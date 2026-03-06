# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import logging
from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal, cast

import numpy as np
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.interfaces import TokenBuffer
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from tqdm import tqdm

from ..qwen3.text_encoder import Qwen3TextEncoderModel
from .pipeline_flux2 import Flux2ModelInputs, Flux2Pipeline

logger = logging.getLogger("max.pipelines")


@dataclass(kw_only=True)
class Flux2KleinModelInputs(Flux2ModelInputs):
    """Flux2 Klein-specific model inputs."""

    guidance_scale: float = 4.0
    negative_tokens: TokenBuffer | None = None

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.negative_tokens is not None and self.guidance_scale > 1.0


@dataclass
class Flux2KleinPipelineOutput:
    """Container for Flux2 Klein pipeline results."""

    images: np.ndarray | Tensor


class Flux2KleinPipeline(Flux2Pipeline):
    """Flux2 Klein diffusion pipeline with Qwen3 text encoder."""

    prompt_embedding_hidden_states_layers: tuple[int, ...] = (9, 18, 27)

    components = {
        "vae": Flux2Pipeline.components["vae"],
        "text_encoder": Qwen3TextEncoderModel,
        "transformer": Flux2Pipeline.components["transformer"],
    }

    def _ensure_prompt_embedding_postprocess_compiled(self) -> None:
        if "_postprocess_prompt_embeddings" in self.__dict__:
            return
        device = self.text_encoder.devices[0]
        dtype = self.text_encoder.config.dtype
        input_types = [
            TensorType(dtype, shape=["seq", "d0"], device=device),
            TensorType(dtype, shape=["seq", "d1"], device=device),
            TensorType(dtype, shape=["seq", "d2"], device=device),
        ]
        self.__dict__["_postprocess_prompt_embeddings"] = max_compile(
            self._postprocess_prompt_embeddings,
            input_types=input_types,
        )

    def _postprocess_prompt_embeddings(
        self,
        hidden_state_0: Tensor,
        hidden_state_1: Tensor,
        hidden_state_2: Tensor,
    ) -> Tensor:
        prompt_embeds = F.concat(
            [hidden_state_0, hidden_state_1, hidden_state_2], axis=-1
        )
        prompt_embeds = F.unsqueeze(prompt_embeds, axis=0)
        return prompt_embeds

    def prepare_inputs(self, context: PixelContext) -> Flux2KleinModelInputs:  # type: ignore[override]
        base_inputs = super().prepare_inputs(context)

        # Klein/distilled path uses zero guidance embedding in the hot path.
        batch_size = context.num_images_per_prompt
        guidance_key = f"zero_{batch_size}"
        if guidance_key in self._cached_guidance:
            guidance = self._cached_guidance[guidance_key]
        else:
            guidance = Tensor.zeros(
                [batch_size],
                device=self.transformer.devices[0],
                dtype=self.transformer.config.dtype,
            )
            self._cached_guidance[guidance_key] = guidance

        return Flux2KleinModelInputs(
            tokens=base_inputs.tokens,
            latents=base_inputs.latents,
            latent_image_ids=base_inputs.latent_image_ids,
            sigmas=base_inputs.sigmas,
            guidance=guidance,
            latent_h=base_inputs.latent_h,
            latent_w=base_inputs.latent_w,
            image_seq_len=base_inputs.image_seq_len,
            h_carrier=base_inputs.h_carrier,
            w_carrier=base_inputs.w_carrier,
            height=base_inputs.height,
            width=base_inputs.width,
            num_inference_steps=base_inputs.num_inference_steps,
            num_images_per_prompt=base_inputs.num_images_per_prompt,
            input_image=base_inputs.input_image,
            negative_tokens=context.negative_tokens,
            guidance_scale=context.guidance_scale,
        )

    def _fuse_hidden_states(self, hidden_states: list[Tensor]) -> Tensor:
        """Concatenate selected layer hidden states into a fused prompt embedding.

        Args:
            hidden_states: List of tensors each shaped (seq_len, hidden_dim).

        Returns:
            Fused tensor of shape (1, seq_len, num_layers * hidden_dim).
        """
        fused = F.concat(hidden_states, axis=-1)
        return fused.reshape((1, int(fused.shape[0]), int(fused.shape[1])))

    def _get_shape_carriers(
        self, height: int, width: int
    ) -> tuple[Tensor, Tensor]:
        """Compute h/w shape-carrier tensors for ``decode_latents``."""
        packed_h = height // (self.vae_scale_factor * 2)
        packed_w = width // (self.vae_scale_factor * 2)
        for n in (packed_h, packed_w):
            if n not in self._cached_shape_carriers:
                self._cached_shape_carriers[n] = Tensor.from_dlpack(
                    np.empty(n, dtype=np.float32)
                )
        return (
            self._cached_shape_carriers[packed_h],
            self._cached_shape_carriers[packed_w],
        )

    def encode_prompt(
        self,
        tokens: TokenBuffer | Tensor,
        num_images_per_prompt: int = 1,
        hidden_states_layers: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode prompt tokens into fused embeddings via the Qwen3 text encoder.

        Unlike the parent ``prepare_prompt_embeddings`` (which takes a device
        ``Tensor``), this method accepts a ``TokenBuffer`` and selects
        intermediate hidden states from the Qwen3 encoder.

        Args:
            tokens: Token buffer produced by the tokenizer.
            num_images_per_prompt: Number of images per prompt.
            hidden_states_layers: Layer indices to extract (1-indexed).

        Returns:
            Tuple of (prompt_embeds, text_ids).
        """
        layers = hidden_states_layers or list(
            self.prompt_embedding_hidden_states_layers
        )
        if isinstance(tokens, TokenBuffer):
            token_ids = np.asarray(tokens.array, dtype=np.int64)
            if token_ids.ndim != 1:
                raise ValueError(
                    f"Flux2Klein expects 1D tokens, got shape {token_ids.shape}."
                )
            target_seq_len = int(token_ids.shape[0])
            text_input_ids = Tensor.constant(
                token_ids,
                dtype=DType.int64,
                device=self.text_encoder.devices[0],
            )
        else:
            text_input_ids = tokens
            target_device = self.text_encoder.devices[0]
            if text_input_ids.device != target_device:
                text_input_ids = text_input_ids.to(target_device)
            if text_input_ids.dtype != DType.int64:
                text_input_ids = text_input_ids.cast(DType.int64)
            if text_input_ids.rank != 1:
                raise ValueError(
                    "Flux2Klein expects 1D token tensor, "
                    f"got rank {text_input_ids.rank}."
                )
            target_seq_len = int(text_input_ids.shape[0])

        hidden_states_all = self.text_encoder(text_input_ids)

        hidden_states_raw: list[Tensor] = []
        all_match_target_seq_len = True
        for i in layers:
            hs = hidden_states_all[i - 1]
            if hs.rank == 3:
                hs = hs[0]
            hidden_states_raw.append(hs)
            if all_match_target_seq_len and int(hs.shape[0]) != target_seq_len:
                all_match_target_seq_len = False

        if all_match_target_seq_len:
            hidden_states_selected = hidden_states_raw
        else:
            hidden_states_selected = []
            for hs in hidden_states_raw:
                seq_len = int(hs.shape[0])
                hidden_dim = int(hs.shape[1])
                if seq_len < target_seq_len:
                    hs = F.concat(
                        [
                            hs,
                            Tensor.zeros(
                                [target_seq_len - seq_len, hidden_dim],
                                dtype=hs.dtype,
                                device=hs.device,
                            ),
                        ],
                        axis=0,
                    )
                elif seq_len > target_seq_len:
                    hs = hs[:target_seq_len]
                hidden_states_selected.append(hs)

        if (
            len(hidden_states_selected) == 3
            and all(h.rank == 2 for h in hidden_states_selected)
            and num_images_per_prompt == 1
        ):
            self._ensure_prompt_embedding_postprocess_compiled()
            prompt_embeds = self._postprocess_prompt_embeddings(
                hidden_states_selected[0],
                hidden_states_selected[1],
                hidden_states_selected[2],
            )
        else:
            prompt_embeds = F.concat(hidden_states_selected, axis=-1)
            if prompt_embeds.rank == 2:
                prompt_embeds = F.unsqueeze(prompt_embeds, axis=0)

        if num_images_per_prompt != 1:
            prompt_embeds = F.tile(
                prompt_embeds, (1, num_images_per_prompt, 1)
            )
            prompt_embeds = prompt_embeds.reshape(
                (num_images_per_prompt, target_seq_len, -1)
            )

        batch_size = int(prompt_embeds.shape[0])
        seq_len = int(prompt_embeds.shape[1])
        batch_size_final = batch_size
        text_ids_key = f"{batch_size_final}_{seq_len}"
        if text_ids_key in self._cached_text_ids:
            text_ids = self._cached_text_ids[text_ids_key]
        else:
            text_ids = self._prepare_text_ids(
                batch_size=batch_size_final,
                seq_len=seq_len,
                device=self.text_encoder.devices[0],
            )
            self._cached_text_ids[text_ids_key] = text_ids
        return prompt_embeds, text_ids

    def execute(  # type: ignore[override]
        self,
        model_inputs: Flux2KleinModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent"] = "np",
    ) -> Flux2KleinPipelineOutput:
        prompt_embeds, text_ids = self.encode_prompt(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )

        diff_cfg = self.pipeline_config.model.diffusers_config or {}
        is_distilled = bool(diff_cfg.get("is_distilled", False))
        if model_inputs.guidance_scale > 1.0 and is_distilled:
            logger.warning(
                "Guidance scale %s is ignored for distilled Klein models.",
                model_inputs.guidance_scale,
            )

        negative_prompt_embeds: Tensor | None = None
        negative_text_ids: Tensor | None = None
        do_cfg = model_inputs.do_classifier_free_guidance and not is_distilled
        if do_cfg and model_inputs.negative_tokens is not None:
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                tokens=model_inputs.negative_tokens,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )
        elif do_cfg:
            logger.warning(
                "CFG requested but negative prompt tokens are missing; "
                "running without CFG."
            )
            do_cfg = False

        batch_size = int(prompt_embeds.shape[0])

        image_latents = None
        image_latent_ids = None
        if model_inputs.input_image is not None:
            img_np = np.array(model_inputs.input_image, dtype=np.uint8)
            image_tensor = self._numpy_image_to_tensor(img_np)
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=[image_tensor],
                batch_size=batch_size,
                device=self.vae.devices[0],
                dtype=self.vae.config.dtype,
            )

        device = self.transformer.devices[0]
        latents = self.preprocess_latents(model_inputs.latents)
        latent_image_ids = model_inputs.latent_image_ids
        guidance = model_inputs.guidance

        image_seq_len = model_inputs.image_seq_len
        num_inference_steps = model_inputs.num_inference_steps
        sigmas_key = f"{num_inference_steps}_{image_seq_len}"
        if sigmas_key in self._cached_sigmas:
            sigmas = self._cached_sigmas[sigmas_key]
        else:
            sigmas = model_inputs.sigmas.to(device)
            self._cached_sigmas[sigmas_key] = sigmas
        all_timesteps, all_dts = self.prepare_scheduler(sigmas)

        timesteps_seq: Any = all_timesteps
        dts_seq: Any = all_dts
        if hasattr(timesteps_seq, "driver_tensor"):
            timesteps_seq = timesteps_seq.driver_tensor
        if hasattr(dts_seq, "driver_tensor"):
            dts_seq = dts_seq.driver_tensor

        is_img2img = image_latents is not None
        for i in tqdm(range(num_inference_steps), desc="Denoising"):
            timestep = timesteps_seq[i : i + 1]
            dt = dts_seq[i : i + 1]

            if is_img2img:
                assert image_latents is not None
                assert image_latent_ids is not None
                latents_concat, latent_image_ids_concat = (
                    self.concat_image_latents(
                        latents,
                        image_latents,
                        latent_image_ids,
                        image_latent_ids,
                    )
                )
            else:
                latents_concat = latents
                latent_image_ids_concat = latent_image_ids

            noise_pred = self.transformer(
                latents_concat,
                prompt_embeds,
                timestep,
                latent_image_ids_concat,
                text_ids,
                guidance,
            )[0]

            if do_cfg:
                assert negative_prompt_embeds is not None
                assert negative_text_ids is not None
                neg_noise_pred = self.transformer(
                    latents_concat,
                    negative_prompt_embeds,
                    timestep,
                    latent_image_ids_concat,
                    negative_text_ids,
                    guidance,
                )[0]
                noise_pred = neg_noise_pred + model_inputs.guidance_scale * (
                    noise_pred - neg_noise_pred
                )

            latents = self.scheduler_step(latents, noise_pred, dt)

            if callback_queue is not None:
                if hasattr(device, "synchronize"):
                    device.synchronize()
                if output_type == "latent":
                    callback_queue.put_nowait(cast(np.ndarray, latents))
                else:
                    h_c, w_c = self._get_shape_carriers(
                        model_inputs.height, model_inputs.width
                    )
                    callback_queue.put_nowait(
                        cast(
                            np.ndarray,
                            self.decode_latents(latents, h_c, w_c),
                        )
                    )

        if output_type == "latent":
            return Flux2KleinPipelineOutput(images=latents)

        h_carrier, w_carrier = self._get_shape_carriers(
            model_inputs.height, model_inputs.width
        )
        images = self.decode_latents(latents, h_carrier, w_carrier)
        return Flux2KleinPipelineOutput(images=images)
