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

from dataclasses import dataclass
from queue import Queue
from typing import Literal

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import DiffusionPipeline, PixelModelInputs
from tqdm import tqdm

from ..autoencoders import AutoencoderKLModel
from ..qwen3.text_encoder import Qwen3TextEncoderModel
from .model import ZImageTransformerModel


@dataclass(kw_only=True)
class ZImageModelInputs(PixelModelInputs):
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    mask: np.ndarray | None = None
    negative_mask: np.ndarray | None = None
    input_image: np.ndarray | None = None


@dataclass
class ZImagePipelineOutput:
    images: np.ndarray | Tensor


class ZImagePipeline(DiffusionPipeline):
    vae: AutoencoderKLModel
    text_encoder: Qwen3TextEncoderModel
    transformer: ZImageTransformerModel

    components = {
        "vae": AutoencoderKLModel,
        "text_encoder": Qwen3TextEncoderModel,
        "transformer": ZImageTransformerModel,
    }

    def init_remaining_components(self) -> None:
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )

    def prepare_inputs(self, context: PixelContext) -> ZImageModelInputs:  # type: ignore[override]
        return ZImageModelInputs.from_context(context)

    @staticmethod
    def _pack_latents(latents: Tensor) -> Tensor:
        batch_size, num_channels, height, width = map(int, latents.shape)
        latents = F.reshape(
            latents,
            (batch_size, num_channels, height // 2, 2, width // 2, 2),
        )
        # Match diffusers Z-Image patchify order: (pH, pW, C) inside each token.
        latents = F.permute(latents, (0, 2, 4, 3, 5, 1))
        latents = F.reshape(
            latents,
            (
                batch_size,
                (height // 2) * (width // 2),
                num_channels * 4,
            ),
        )
        return latents

    @staticmethod
    def _unpack_latents(
        latents: Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> Tensor:
        batch_size = int(latents.shape[0])
        ch_size = int(latents.shape[2])

        height = 2 * (height // (vae_scale_factor * 2))
        width = 2 * (width // (vae_scale_factor * 2))

        h2 = height // 2
        w2 = width // 2
        latents = F.reshape(
            latents,
            (batch_size, h2, w2, 2, 2, ch_size // 4),
        )
        latents = F.permute(latents, (0, 5, 1, 3, 2, 4))
        latents = F.reshape(
            latents,
            (batch_size, ch_size // 4, height, width),
        )
        return latents

    def _prepare_prompt_embeddings(
        self,
        tokens: np.ndarray,
        mask: np.ndarray | None,
        num_images_per_prompt: int,
    ) -> Tensor:
        if tokens.ndim == 2:
            tokens = tokens[0]
        selected_tokens = tokens
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[0]
            if mask.shape[0] != tokens.shape[0]:
                raise ValueError(
                    "Z-Image mask length must match token length after tokenizer preprocessing. "
                    f"Got mask={mask.shape[0]}, tokens={tokens.shape[0]}."
                )
            if not np.all(mask):
                raise ValueError(
                    "Z-Image expects tokenizer-pretrimmed tokens with dense attention mask. "
                    "Received sparse mask, which indicates an unexpected tokenizer/pipeline mismatch."
                )
            selected_tokens = tokens[mask]

        text_input_ids = Tensor.constant(
            selected_tokens,
            dtype=DType.int64,
            device=self.text_encoder.devices[0],
        )
        hidden_states = self.text_encoder(
            text_input_ids,
            hidden_state_index=-2,
        )

        hidden_states = F.unsqueeze(hidden_states, 0)

        if num_images_per_prompt > 1:
            hidden_states = F.tile(hidden_states, (num_images_per_prompt, 1, 1))

        return hidden_states

    @staticmethod
    def _align_prompt_seq_len(
        embeds: Tensor,
        target_seq_len: int,
    ) -> Tensor:
        cur_len = int(embeds.shape[1])
        if cur_len == target_seq_len:
            return embeds
        if cur_len > target_seq_len:
            return embeds[:, :target_seq_len, :]

        pad_len = target_seq_len - cur_len
        pad = Tensor.zeros(
            (int(embeds.shape[0]), pad_len, int(embeds.shape[2])),
            dtype=embeds.dtype,
            device=embeds.device,
        )
        return F.concat([embeds, pad], axis=1)

    @staticmethod
    def _prepare_text_ids(
        seq_len: int,
        device,
    ) -> Tensor:
        text_ids = np.zeros((seq_len, 3), dtype=np.int64)
        text_ids[:, 0] = np.arange(1, seq_len + 1, dtype=np.int64)
        return Tensor.from_dlpack(text_ids).to(device)

    def _decode_latents(
        self,
        latents: Tensor,
        height: int,
        width: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        if output_type == "latent":
            return latents

        latents = self._unpack_latents(
            latents,
            height,
            width,
            self.vae_scale_factor,
        )
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        return self._to_numpy(self.vae.decode(latents))

    @staticmethod
    def _to_numpy(image: Tensor) -> np.ndarray:
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)

    @staticmethod
    def _vector_norm_per_sample(x: Tensor) -> Tensor:
        squared = x * x
        # x shape: [B, S, C] -> norm shape: [B]
        squared = F.sum(squared, axis=2)
        squared = F.sum(squared, axis=1)
        return F.sqrt(squared + 1e-12)

    @classmethod
    def _apply_cfg_renormalization(
        cls,
        pos: Tensor,
        pred: Tensor,
        cfg_normalization: bool,
    ) -> Tensor:
        if not cfg_normalization:
            return pred

        ori_pos_norm = cls._vector_norm_per_sample(pos)
        new_pos_norm = cls._vector_norm_per_sample(pred)
        while ori_pos_norm.rank > 1:
            ori_pos_norm = F.squeeze(ori_pos_norm, axis=-1)
        while new_pos_norm.rank > 1:
            new_pos_norm = F.squeeze(new_pos_norm, axis=-1)
        max_new_norm = ori_pos_norm
        # Avoid divide-by-zero and clip only when required.
        safe_new_norm = F.where(new_pos_norm > 1e-12, new_pos_norm, 1e-12)
        ratio = max_new_norm / safe_new_norm
        ratio = F.where(new_pos_norm > max_new_norm, ratio, 1.0)
        ratio = F.unsqueeze(F.unsqueeze(ratio, 1), 2)
        return pred * ratio

    @staticmethod
    def _scheduler_step(
        latents: Tensor,
        noise_pred: Tensor,
        sigmas: Tensor,
        step_index: int,
    ) -> Tensor:
        latents_dtype = latents.dtype
        latents = latents.cast(DType.float32)
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        latents = latents + dt * noise_pred
        latents = latents.cast(latents_dtype)
        return latents

    def _image_to_tensor(
        self,
        image: np.ndarray,
        batch_size: int,
        dtype: DType,
    ) -> Tensor:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected input image shape [H, W, 3], got {image.shape}."
            )

        height, width, _ = image.shape
        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0 or width % vae_scale != 0:
            raise ValueError(
                f"Input image dimensions must be divisible by {vae_scale}, "
                f"got {height}x{width}."
            )

        image_f32 = image.astype(np.float32) / 127.5 - 1.0
        image_chw = np.transpose(image_f32, (2, 0, 1))
        image_bchw = np.expand_dims(image_chw, axis=0)
        if batch_size > 1:
            image_bchw = np.repeat(image_bchw, batch_size, axis=0)
        image_bchw = np.ascontiguousarray(image_bchw)

        return (
            Tensor.from_dlpack(image_bchw)
            .to(self.vae.devices[0])
            .cast(dtype)
        )

    def _prepare_img2img_latents(
        self,
        noise_latents: Tensor,
        image: np.ndarray,
        sigmas: Tensor,
    ) -> Tensor:
        batch_size = int(noise_latents.shape[0])
        image_tensor = self._image_to_tensor(
            image=image,
            batch_size=batch_size,
            dtype=self.vae.config.dtype,
        )

        encoder_output = self.vae.encode(image_tensor, return_dict=True)
        posterior = (
            encoder_output["latent_dist"]
            if isinstance(encoder_output, dict)
            else encoder_output
        )
        if not hasattr(posterior, "mode"):
            raise ValueError("VAE encoder output does not expose `mode()`.")

        image_latents = posterior.mode()
        image_latents = (
            image_latents - float(self.vae.config.shift_factor or 0.0)
        ) * float(self.vae.config.scaling_factor)
        image_latents = image_latents.to(self.transformer.devices[0]).cast(
            noise_latents.dtype
        )

        sigma = sigmas[0]
        latents = sigma * noise_latents + (1.0 - sigma) * image_latents
        return latents.cast(noise_latents.dtype)

    def execute(  # type: ignore[override]
        self,
        model_inputs: ZImageModelInputs,
        callback_queue: Queue[np.ndarray | Tensor] | None = None,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> ZImagePipelineOutput:
        prompt_embeds = self._prepare_prompt_embeddings(
            tokens=model_inputs.tokens.array,
            mask=model_inputs.mask,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )

        negative_prompt_embeds: Tensor | None = None
        do_cfg = (
            model_inputs.guidance_scale > 1.0
            and model_inputs.negative_tokens is not None
        )
        if do_cfg:
            negative_prompt_embeds = self._prepare_prompt_embeddings(
                tokens=model_inputs.negative_tokens.array,
                mask=model_inputs.negative_mask,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )
            negative_prompt_embeds = self._align_prompt_seq_len(
                negative_prompt_embeds,
                int(prompt_embeds.shape[1]),
            )

        dtype = prompt_embeds.dtype

        timesteps: np.ndarray = model_inputs.timesteps
        batch_size = int(prompt_embeds.shape[0])
        num_timesteps = timesteps.shape[0]
        if num_timesteps < 1:
            raise ValueError("No timesteps available for denoising.")
        text_seq_len = int(prompt_embeds.shape[1])
        text_seq_len_padded = text_seq_len + (-text_seq_len % 32)

        img_ids_np = model_inputs.latent_image_ids.astype(np.int64, copy=True)
        if img_ids_np.ndim == 3:
            img_ids_np = img_ids_np[0]
        img_ids_np[:, 0] = img_ids_np[:, 0] + text_seq_len_padded + 1
        img_ids = Tensor.from_dlpack(img_ids_np).to(self.transformer.devices[0])
        txt_ids = self._prepare_text_ids(
            text_seq_len,
            self.transformer.devices[0],
        )

        latents = (
            Tensor.from_dlpack(model_inputs.latents)
            .to(self.transformer.devices[0])
            .cast(dtype)
        )
        sigmas = Tensor.from_dlpack(model_inputs.sigmas).to(
            self.transformer.devices[0]
        )
        if model_inputs.input_image is not None:
            latents = self._prepare_img2img_latents(
                noise_latents=latents,
                image=model_inputs.input_image,
                sigmas=sigmas,
            )
        latents = self._pack_latents(latents)
        timesteps_np = np.broadcast_to(timesteps[:, None], (num_timesteps, batch_size))
        timesteps_batched = Tensor.from_dlpack(timesteps_np).to(self.transformer.devices[0])

        for i in tqdm(range(num_timesteps), desc="Denoising"):
            timestep = timesteps_batched[i]
            # Tokenizer scheduler stores normalized timesteps in [0, 1] (sigma domain).
            # Z-Image expects time-aware config value equivalent to (1000 - t_raw) / 1000.
            # Since t_raw = timestep * 1000 here, this reduces to (1 - timestep).
            timestep = 1.0 - timestep
            t_norm = 1.0 - float(timesteps[i])

            current_guidance_scale = model_inputs.guidance_scale
            if do_cfg and model_inputs.cfg_truncation <= 1.0:
                if t_norm > model_inputs.cfg_truncation:
                    current_guidance_scale = 0.0
            apply_cfg = do_cfg and current_guidance_scale > 0.0

            noise_pred = self.transformer(
                latents,
                prompt_embeds,
                timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
            )[0]

            if apply_cfg:
                assert negative_prompt_embeds is not None
                neg_noise_pred = self.transformer(
                    latents,
                    negative_prompt_embeds,
                    timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                )[0]
                pos_noise_pred = noise_pred
                noise_pred = pos_noise_pred + current_guidance_scale * (
                    noise_pred - neg_noise_pred
                )
                noise_pred = self._apply_cfg_renormalization(
                    pos_noise_pred,
                    noise_pred,
                    model_inputs.cfg_normalization,
                )

            noise_pred = -noise_pred
            latents = self._scheduler_step(latents, noise_pred, sigmas, i)

            if callback_queue is not None:
                image = self._decode_latents(
                    latents,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                )
                callback_queue.put_nowait(image)

        outputs = self._decode_latents(
            latents,
            model_inputs.height,
            model_inputs.width,
            output_type=output_type,
        )

        return ZImagePipelineOutput(images=outputs)
