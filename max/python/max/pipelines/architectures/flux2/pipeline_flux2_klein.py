"""Flux2 Klein pipeline implementation."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Literal

import numpy as np
from max import functional as F
from max.dtype import DType
from max.interfaces import TokenBuffer
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import PixelModelInputs
from max.tensor import Tensor
from PIL import Image
from tqdm import tqdm

from ..autoencoders import AutoencoderKLFlux2Model
from .model import Flux2TransformerModel
from .pipeline_flux2 import Flux2Pipeline, Flux2PipelineOutput
from ..qwen3.text_encoder import Qwen3TextEncoderModel



@dataclass(kw_only=True)
class Flux2KleinModelInputs(PixelModelInputs):
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    num_images_per_prompt: int = 1
    input_image: Image.Image | None = None


class Flux2KleinPipeline(Flux2Pipeline):
    """Diffusion pipeline for Flux2 Klein image generation."""

    vae: AutoencoderKLFlux2Model
    text_encoder: Qwen3TextEncoderModel
    transformer: Flux2TransformerModel

    components = {
        "vae": AutoencoderKLFlux2Model,
        "text_encoder": Qwen3TextEncoderModel,
        "transformer": Flux2TransformerModel,
    }

    def init_remaining_components(self) -> None:
        super().init_remaining_components()
        self.is_distilled = self.pipeline_config.model.diffusers_config.get(
            "is_distilled", False
        )
        # diffusers Klein passes guidance=None; MAX transformer currently requires Tensor.
        self.transformer_uses_guidance_embeds = bool(
            getattr(self.transformer.config, "guidance_embeds", False)
        )

    @property
    def do_classifier_free_guidance(self) -> bool:
        guidance_scale = getattr(self, "_guidance_scale", 1.0)
        return guidance_scale > 1 and not self.is_distilled

    def prepare_inputs(self, context: PixelContext) -> Flux2KleinModelInputs:  # type: ignore[override]
        if context.input_image is not None and isinstance(
            context.input_image, np.ndarray
        ):
            context.input_image = Image.fromarray(
                context.input_image.astype(np.uint8)
            )
        return Flux2KleinModelInputs.from_context(context)

    def _prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        num_images_per_prompt: int = 1,
        hidden_states_layers: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        layers = hidden_states_layers or [9, 18, 27]
        max_seq = int(tokens.array.shape[-1])

        text_input_ids = Tensor.constant(
            tokens.array, dtype=DType.int64, device=self.text_encoder.devices[0]
        )
        hs_all = self.text_encoder(text_input_ids)
        if not isinstance(hs_all, tuple):
            raise ValueError(
                f"Expected tuple of hidden states from Qwen3, got {type(hs_all)}"
            )

        selected: list[Tensor] = []
        for i in layers:
            if i >= len(hs_all):
                raise ValueError(
                    f"Layer index {i} out of range (model has {len(hs_all)} layers)."
                )
            hs = hs_all[i]
            hs = hs if isinstance(hs, Tensor) else Tensor.from_dlpack(hs)
            if hs.rank == 2:
                hs = F.unsqueeze(hs, axis=0)

            _, seq_len, _ = map(int, hs.shape)
            if seq_len < max_seq:
                hs = F.pad(
                    hs, pad=((0, 0), (0, max_seq - seq_len), (0, 0))
                )
            elif seq_len > max_seq:
                hs = hs[:, :max_seq, :]
            selected.append(hs)

        stacked = F.stack(selected, axis=1)
        stacked = F.permute(stacked, [0, 2, 1, 3])
        batch_size, seq_len, num_layers, hidden_dim = map(int, stacked.shape)
        prompt_embeds = F.reshape(
            stacked, [batch_size, seq_len, num_layers * hidden_dim]
        )

        if num_images_per_prompt != 1:
            prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
            prompt_embeds = F.reshape(
                prompt_embeds, [batch_size * num_images_per_prompt, seq_len, -1]
            )

        text_ids = self._prepare_text_ids(
            batch_size=batch_size * num_images_per_prompt,
            seq_len=seq_len,
            device=self.text_encoder.devices[0],
        )
        return prompt_embeds, text_ids

    def execute(  # type: ignore[override]
        self,
        model_inputs: Flux2KleinModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent"] = "np",
    ) -> Flux2PipelineOutput:
        self._guidance_scale = model_inputs.guidance_scale

        prompt_embeds, text_ids = self._prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )
        batch_size = int(prompt_embeds.shape[0])
        dtype = prompt_embeds.dtype

        negative_prompt_embeds: Tensor | None = None
        negative_text_ids: Tensor | None = None
        if self.do_classifier_free_guidance:
            negative_tokens = model_inputs.negative_tokens
            if negative_tokens is None:
                raise ValueError(
                    "Flux2Klein CFG requires negative tokens. "
                    "Tokenizer should provide empty-string negative tokens when guidance_scale > 1."
                )
            negative_prompt_embeds, negative_text_ids = (
                self._prepare_prompt_embeddings(
                    tokens=negative_tokens,
                    num_images_per_prompt=model_inputs.num_images_per_prompt,
                )
            )

        image_latents = None
        image_latent_ids = None
        if model_inputs.input_image is not None:
            image_tensor = self._pil_image_to_tensor(model_inputs.input_image)
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=[image_tensor],
                batch_size=batch_size,
                device=self.vae.devices[0],
                dtype=self.vae.config.dtype,
            )

        latents: Tensor = (
            Tensor.from_dlpack(model_inputs.latents)
            .to(self.transformer.devices[0])
            .cast(dtype)
        )
        latents = self._patchify_latents(latents)
        latents = self._pack_latents(latents)

        latent_image_ids = Tensor.from_dlpack(
            model_inputs.latent_image_ids.astype(np.int64)
        ).to(self.transformer.devices[0])

        if self.transformer_uses_guidance_embeds:
            guidance = Tensor.full(
                [latents.shape[0]],
                model_inputs.guidance_scale,
                device=self.transformer.devices[0],
                dtype=dtype,
            )
        else:
            # MAX transformer component expects a Tensor input for `guidance`.
            # For Klein models without guidance embeddings, pass a dummy tensor.
            guidance = Tensor.zeros(
                [latents.shape[0]],
                dtype=dtype,
                device=self.transformer.devices[0],
            )

        sigmas = Tensor.from_dlpack(model_inputs.sigmas).to(
            self.transformer.devices[0]
        )
        timesteps: np.ndarray = model_inputs.timesteps
        num_timesteps = timesteps.shape[0]
        timesteps_np = np.broadcast_to(
            timesteps[:, None], (num_timesteps, batch_size)
        )
        timesteps_batched = (
            Tensor.from_dlpack(timesteps_np)
            .to(self.transformer.devices[0])
            .cast(dtype)
        )

        num_noise_tokens = int(latents.shape[1])
        for i in tqdm(range(num_timesteps), desc="Denoising"):
            timestep = timesteps_batched[i]
            if image_latents is not None:
                latent_model_input = F.concat([latents, image_latents], axis=1)
                latent_model_ids = F.concat(
                    [latent_image_ids, image_latent_ids], axis=1
                )
            else:
                latent_model_input = latents
                latent_model_ids = latent_image_ids

            noise_pred = self.transformer(
                latent_model_input,
                prompt_embeds,
                timestep,
                latent_model_ids,
                text_ids,
                guidance,
            )[0]
            noise_pred = Tensor.from_dlpack(noise_pred)
            noise_pred = noise_pred[:, :num_noise_tokens, :]

            if (
                self.do_classifier_free_guidance
                and negative_prompt_embeds is not None
            ):
                assert negative_text_ids is not None
                neg_noise_pred = self.transformer(
                    latent_model_input,
                    negative_prompt_embeds,
                    timestep,
                    latent_model_ids,
                    negative_text_ids,
                    guidance,
                )[0]
                neg_noise_pred = Tensor.from_dlpack(neg_noise_pred)
                neg_noise_pred = neg_noise_pred[:, :num_noise_tokens, :]
                noise_pred = neg_noise_pred + model_inputs.guidance_scale * (
                    noise_pred - neg_noise_pred
                )

            latents = self._scheduler_step(latents, noise_pred, sigmas, i)

            if callback_queue is not None and output_type == "np":
                decoded = self._decode_latents(
                    latents, latent_image_ids, output_type="np"
                )
                if isinstance(decoded, Tensor):
                    decoded = np.array(decoded)
                callback_queue.put_nowait(decoded)

        image_list = []
        for b in range(batch_size):
            latents_b = latents[b : b + 1]
            latent_image_ids_b = latent_image_ids[b : b + 1]
            image_list.append(
                self._decode_latents(
                    latents_b, latent_image_ids_b, output_type=output_type
                )
            )

        return Flux2PipelineOutput(images=image_list)  # type: ignore[arg-type]
