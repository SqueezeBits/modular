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

import os
from dataclasses import dataclass
from queue import Queue
from typing import Any, cast

import numpy as np
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.interfaces import TokenBuffer
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import DiffusionPipeline, PixelModelInputs
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from tqdm import tqdm

from ..autoencoders import AutoencoderKLLTXVideoModel
from ..t5.model import T5Model
from .model import LTXTransformer3DModel


@dataclass(kw_only=True)
class LTXModelInputs(PixelModelInputs):
    """LTX-specific model inputs for text/image-to-video generation."""

    width: int = 704
    height: int = 512
    num_frames: int | None = 161
    frames_per_second: int | None = 25
    guidance_scale: float = 3.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    decode_timestep: float | None = 0.0
    decode_noise_scale: float | None = None
    denoise_strength: float = 1.0
    input_image: np.ndarray | None = None
    image_cond_noise_scale: float = 0.15

    @property
    def do_true_cfg(self) -> bool:
        return self.guidance_scale > 1.0


@dataclass
class LTXPipelineOutput:
    """Output container for LTX video generation."""

    videos: np.ndarray
    frames_per_second: int


class LTXPipeline(DiffusionPipeline):
    """Diffusion pipeline for LTX text-to-video MVP."""

    vae: AutoencoderKLLTXVideoModel
    text_encoder: T5Model
    transformer: LTXTransformer3DModel

    components = {
        "vae": AutoencoderKLLTXVideoModel,
        "text_encoder": T5Model,
        "transformer": LTXTransformer3DModel,
    }

    def init_remaining_components(self) -> None:
        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio
            if getattr(self, "vae", None)
            else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio
            if getattr(self, "vae", None)
            else 8
        )
        self.transformer_spatial_patch_size = int(
            getattr(self.transformer.config, "patch_size", 1)
        )
        self.transformer_temporal_patch_size = int(
            getattr(self.transformer.config, "patch_size_t", 1)
        )

        self.build_prepare_scheduler()
        self.build_scheduler_step()

        self._transformer_device: Device = self.transformer.devices[0]
        self._cached_sigmas: dict[str, Tensor] = {}
        self._cached_rotary: dict[str, tuple[Tensor, Tensor]] = {}

    def prepare_inputs(self, context: PixelContext) -> LTXModelInputs:  # type: ignore[override]
        return LTXModelInputs.from_context(context)

    def build_prepare_scheduler(self) -> None:
        input_types = [
            TensorType(
                DType.float32,
                shape=["num_sigmas"],
                device=self.transformer.devices[0],
            ),
        ]
        self.__dict__["prepare_scheduler"] = max_compile(
            self.prepare_scheduler,
            input_types=input_types,
        )

    def build_scheduler_step(self) -> None:
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                DType.float32,
                shape=["batch", "seq", "channels"],
                device=device,
            ),
            TensorType(
                DType.float32,
                shape=["batch", "seq", "channels"],
                device=device,
            ),
            TensorType(DType.float32, shape=[1], device=device),
        ]
        self.__dict__["scheduler_step"] = max_compile(
            self.scheduler_step,
            input_types=input_types,
        )

    def prepare_scheduler(self, sigmas: Tensor) -> tuple[Tensor, Tensor]:
        sigmas_curr = F.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = F.slice_tensor(sigmas, [slice(1, None)])
        all_dt = F.sub(sigmas_next, sigmas_curr)
        # Diffusers LTX transformer consumes scheduler timesteps in the
        # [0, 1000] domain (scheduler timesteps = sigmas * num_train_timesteps).
        all_timesteps = sigmas_curr.cast(DType.float32) * 1000.0
        return all_timesteps, all_dt

    def scheduler_step(
        self, latents: Tensor, noise_pred: Tensor, dt: Tensor
    ) -> Tensor:
        noise_pred = noise_pred.cast(DType.float32)
        latents = latents.cast(DType.float32)
        return latents + dt * noise_pred

    def prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        mask: np.ndarray | None,
        num_images_per_prompt: int = 1,
    ) -> tuple[Tensor, Tensor]:
        if tokens.array.ndim == 1:
            tokens.array = np.expand_dims(tokens.array, axis=0)

        if mask is None:
            mask_np = np.ones(tokens.array.shape, dtype=np.bool_)
        else:
            mask_np = mask
            if mask_np.ndim == 1:
                mask_np = np.expand_dims(mask_np, axis=0)

        token_ids = Tensor.constant(
            tokens.array,
            dtype=DType.int64,
            device=self.text_encoder.devices[0],
        )
        attention_mask = Tensor.constant(
            mask_np.astype(np.bool_, copy=False),
            dtype=DType.bool,
            device=self.text_encoder.devices[0],
        )
        prompt_embeds = self.text_encoder(token_ids, attention_mask)

        if num_images_per_prompt != 1:
            bs_embed = int(prompt_embeds.shape[0])
            seq_len = int(prompt_embeds.shape[1])
            prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
            prompt_embeds = prompt_embeds.reshape(
                (bs_embed * num_images_per_prompt, seq_len, -1)
            )
            mask_np = np.repeat(mask_np, num_images_per_prompt, axis=0)

        prompt_attention_mask = Tensor.constant(
            mask_np.astype(np.bool_, copy=False),
            dtype=DType.bool,
            device=self.text_encoder.devices[0],
        )

        return prompt_embeds, prompt_attention_mask

    @staticmethod
    def _unpack_latents_np(
        latents: np.ndarray,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> np.ndarray:
        batch_size = latents.shape[0]
        latents = latents.reshape(
            batch_size,
            num_frames,
            height,
            width,
            -1,
            patch_size_t,
            patch_size,
            patch_size,
        )
        latents = np.transpose(latents, (0, 4, 1, 5, 2, 6, 3, 7))
        latents = latents.reshape(
            batch_size,
            -1,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )
        return latents.astype(np.float32, copy=False)

    @staticmethod
    def _pack_latents_np(
        latents: np.ndarray,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> np.ndarray:
        batch_size, channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            channels,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = np.transpose(latents, (0, 2, 4, 6, 1, 3, 5, 7))
        latents = latents.reshape(
            batch_size,
            post_patch_num_frames * post_patch_height * post_patch_width,
            channels * patch_size_t * patch_size * patch_size,
        )
        return np.ascontiguousarray(latents.astype(np.float32, copy=False))

    @staticmethod
    def _normalize_latents_np(
        latents: np.ndarray,
        latents_mean: np.ndarray,
        latents_std: np.ndarray,
        scaling_factor: float = 1.0,
    ) -> np.ndarray:
        num_channels = latents.shape[1]

        latents_mean = np.asarray(latents_mean, dtype=np.float32)
        latents_std = np.asarray(latents_std, dtype=np.float32)

        if latents_mean.size < num_channels:
            latents_mean = np.pad(
                latents_mean,
                (0, num_channels - latents_mean.size),
                mode="constant",
                constant_values=0.0,
            )
        if latents_std.size < num_channels:
            latents_std = np.pad(
                latents_std,
                (0, num_channels - latents_std.size),
                mode="constant",
                constant_values=1.0,
            )

        latents_mean = latents_mean[:num_channels].reshape(1, -1, 1, 1, 1)
        latents_std = latents_std[:num_channels].reshape(1, -1, 1, 1, 1)

        return (latents - latents_mean) * scaling_factor / latents_std

    @staticmethod
    def _denormalize_latents_np(
        latents: np.ndarray,
        latents_mean: np.ndarray,
        latents_std: np.ndarray,
        scaling_factor: float = 1.0,
    ) -> np.ndarray:
        num_channels = latents.shape[1]

        latents_mean = np.asarray(latents_mean, dtype=np.float32)
        latents_std = np.asarray(latents_std, dtype=np.float32)

        if latents_mean.size < num_channels:
            latents_mean = np.pad(
                latents_mean,
                (0, num_channels - latents_mean.size),
                mode="constant",
                constant_values=0.0,
            )
        if latents_std.size < num_channels:
            latents_std = np.pad(
                latents_std,
                (0, num_channels - latents_std.size),
                mode="constant",
                constant_values=1.0,
            )

        latents_mean = latents_mean[:num_channels].reshape(1, -1, 1, 1, 1)
        latents_std = latents_std[:num_channels].reshape(1, -1, 1, 1, 1)

        return latents * latents_std / scaling_factor + latents_mean

    @staticmethod
    def _to_numpy(tensor: Tensor) -> np.ndarray:
        cpu_tensor: Tensor = tensor.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_tensor)

    @staticmethod
    def _prepare_condition_tensor_np(
        input_image: np.ndarray,
        *,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        image_np = np.asarray(input_image)
        if image_np.ndim != 3:
            raise ValueError(
                f"Expected input_image shape [H, W, C], got {image_np.shape}"
            )
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        if image_np.shape[2] == 1:
            image_np = np.repeat(image_np, repeats=3, axis=2)

        if image_np.shape[0] != target_height or image_np.shape[1] != target_width:
            y_idx = np.linspace(
                0, max(image_np.shape[0] - 1, 0), target_height
            ).round().astype(np.int32)
            x_idx = np.linspace(
                0, max(image_np.shape[1] - 1, 0), target_width
            ).round().astype(np.int32)
            image_np = image_np[y_idx][:, x_idx]

        image_np = image_np.astype(np.float32) / 255.0
        image_np = image_np * 2.0 - 1.0
        # [H, W, C] -> [B, C, F, H, W] with F=1
        image_np = np.transpose(image_np, (2, 0, 1))[None, :, None, :, :]
        return np.ascontiguousarray(image_np.astype(np.float32, copy=False))

    @staticmethod
    def _pad_sequence_to_length(
        tensor: Tensor,
        *,
        target_length: int,
        pad_value: float | bool = 0.0,
    ) -> Tensor:
        current_length = int(tensor.shape[1])
        if current_length >= target_length:
            return tensor

        pad_length = target_length - current_length
        if tensor.rank == 3:
            pad_shape = (int(tensor.shape[0]), pad_length, int(tensor.shape[2]))
        elif tensor.rank == 2:
            pad_shape = (int(tensor.shape[0]), pad_length)
        else:
            raise ValueError(
                f"Unsupported rank for sequence padding: {tensor.rank}"
            )

        pad = Tensor.full(
            list(pad_shape),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return F.concat([tensor, pad], axis=1)

    @staticmethod
    def _resize_video_nearest(
        video: np.ndarray,
        *,
        target_frames: int,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        """Resize video tensor [B, F, H, W, C] with nearest-neighbor indexing."""
        if video.ndim != 5:
            return video

        _, num_frames, height, width, _ = video.shape

        if num_frames != target_frames:
            frame_idx = np.linspace(
                0, max(num_frames - 1, 0), target_frames
            ).round().astype(np.int32)
            video = video[:, frame_idx, :, :, :]

        if height != target_height:
            h_idx = np.linspace(
                0, max(height - 1, 0), target_height
            ).round().astype(np.int32)
            video = video[:, :, h_idx, :, :]

        if width != target_width:
            w_idx = np.linspace(
                0, max(width - 1, 0), target_width
            ).round().astype(np.int32)
            video = video[:, :, :, w_idx, :]

        return np.ascontiguousarray(video, dtype=np.float32)

    @staticmethod
    def _linear_quadratic_schedule(
        num_steps: int,
        threshold_noise: float = 0.025,
        linear_steps: int | None = None,
    ) -> np.ndarray:
        if linear_steps is None:
            linear_steps = num_steps // 2
        if num_steps < 2:
            return np.array([1.0], dtype=np.float32)

        linear_sigma_schedule = [
            i * threshold_noise / linear_steps for i in range(linear_steps)
        ]
        threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
        quadratic_steps = num_steps - linear_steps
        quadratic_coef = threshold_noise_step_diff / (
            linear_steps * quadratic_steps**2
        )
        linear_coef = (
            threshold_noise / linear_steps
            - 2 * threshold_noise_step_diff / (quadratic_steps**2)
        )
        const = quadratic_coef * (linear_steps**2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i**2) + linear_coef * i + const
            for i in range(linear_steps, num_steps)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
        return np.asarray(sigma_schedule[:-1], dtype=np.float32)

    @staticmethod
    def _shifted_sigmas_from_timesteps(
        timesteps: np.ndarray,
        *,
        shift: float = 1.0,
        shift_terminal: float | None = 0.1,
        num_train_timesteps: float = 1000.0,
    ) -> np.ndarray:
        """Mirror diffusers scheduler.set_timesteps(custom timesteps) sigma path."""
        sigmas = (
            np.asarray(timesteps, dtype=np.float32) / np.float32(num_train_timesteps)
        ).astype(np.float32, copy=False)
        sigmas = (
            shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        ).astype(np.float32, copy=False)

        if shift_terminal is not None and sigmas.size > 0:
            one_minus = 1.0 - sigmas
            denom = max(1.0 - float(shift_terminal), 1e-6)
            scale_factor = one_minus[-1] / denom
            if abs(scale_factor) > 1e-8:
                sigmas = 1.0 - (one_minus / scale_factor)

        return np.append(sigmas.astype(np.float32, copy=False), np.float32(0.0))

    @staticmethod
    def _get_timesteps_for_strength(
        sigmas: np.ndarray,
        strength: float,
    ) -> tuple[np.ndarray, int]:
        if strength >= 1.0:
            return sigmas.astype(np.float32, copy=False), 0

        num_inference_steps = int(sigmas.shape[0] - 1)
        num_steps = min(int(num_inference_steps * strength), num_inference_steps)
        start_index = max(num_inference_steps - num_steps, 0)
        return sigmas[start_index:].astype(np.float32, copy=False), start_index

    @staticmethod
    def _build_rotary_embeddings_np(
        *,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dim: int,
        patch_size: int,
        patch_size_t: int,
        rope_interpolation_scale: tuple[float, float, float],
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        theta: float = 10000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Matches diffusers LTXVideoRotaryPosEmbed math.
        grid_h = np.arange(height, dtype=np.float32)
        grid_w = np.arange(width, dtype=np.float32)
        grid_f = np.arange(num_frames, dtype=np.float32)
        grid_f_, grid_h_, grid_w_ = np.meshgrid(
            grid_f, grid_h, grid_w, indexing="ij"
        )
        grid = np.stack([grid_f_, grid_h_, grid_w_], axis=0)[None, ...]
        grid = np.repeat(grid, batch_size, axis=0)

        scale_t, scale_h, scale_w = rope_interpolation_scale
        grid[:, 0:1] = (
            grid[:, 0:1] * scale_t * float(patch_size_t) / float(base_num_frames)
        )
        grid[:, 1:2] = (
            grid[:, 1:2] * scale_h * float(patch_size) / float(base_height)
        )
        grid[:, 2:3] = (
            grid[:, 2:3] * scale_w * float(patch_size) / float(base_width)
        )

        # [B, 3, F, H, W] -> [B, S, 3]
        grid = grid.reshape(batch_size, 3, -1).transpose(0, 2, 1)

        rope_dim = dim // 6
        rope_lin = np.linspace(
            0.0,
            1.0,
            rope_dim,
            dtype=np.float32,
        )
        freqs = (theta**rope_lin) * (np.pi / 2.0)

        freqs = freqs * (grid[..., None] * 2.0 - 1.0)
        freqs = np.transpose(freqs, (0, 1, 3, 2)).reshape(
            batch_size, grid.shape[1], -1
        )

        cos_freqs = np.cos(freqs).repeat(2, axis=-1)
        sin_freqs = np.sin(freqs).repeat(2, axis=-1)

        remainder = dim % 6
        if remainder != 0:
            cos_padding = np.ones(
                (batch_size, cos_freqs.shape[1], remainder), dtype=np.float32
            )
            sin_padding = np.zeros(
                (batch_size, sin_freqs.shape[1], remainder), dtype=np.float32
            )
            cos_freqs = np.concatenate([cos_padding, cos_freqs], axis=-1)
            sin_freqs = np.concatenate([sin_padding, sin_freqs], axis=-1)

        return (
            np.ascontiguousarray(cos_freqs, dtype=np.float32),
            np.ascontiguousarray(sin_freqs, dtype=np.float32),
        )

    def _get_rotary_embeddings(
        self,
        *,
        batch_size: int,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        frames_per_second: int,
        dtype: DType,
    ) -> tuple[Tensor, Tensor]:
        key = (
            f"{batch_size}_{latent_num_frames}_{latent_height}_{latent_width}_"
            f"{frames_per_second}_{dtype}"
        )
        if key not in self._cached_rotary:
            rope_interpolation_scale = (
                float(self.vae_temporal_compression_ratio)
                / float(max(frames_per_second, 1)),
                float(self.vae_spatial_compression_ratio),
                float(self.vae_spatial_compression_ratio),
            )
            cos_np, sin_np = self._build_rotary_embeddings_np(
                batch_size=batch_size,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                dim=int(self.transformer.config.num_attention_heads)
                * int(self.transformer.config.attention_head_dim),
                patch_size=self.transformer_spatial_patch_size,
                patch_size_t=self.transformer_temporal_patch_size,
                rope_interpolation_scale=rope_interpolation_scale,
            )
            # Keep rotary frequencies in fp32 to match diffusers LTX behavior.
            # They are consumed in fp32 inside rotary application.
            cos = Tensor.from_dlpack(cos_np).to(self._transformer_device)
            sin = Tensor.from_dlpack(sin_np).to(self._transformer_device)
            self._cached_rotary[key] = (cos, sin)
        return self._cached_rotary[key]

    def execute(  # type: ignore[override]
        self,
        model_inputs: LTXModelInputs,
        callback_queue: Queue[np.ndarray | Tensor] | None = None,
    ) -> LTXPipelineOutput:
        prompt_embeds, prompt_attention_mask = self.prepare_prompt_embeddings(
            model_inputs.tokens,
            model_inputs.mask,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )

        if model_inputs.do_true_cfg and model_inputs.negative_tokens is None:
            raise ValueError(
                "LTX CFG is enabled (guidance_scale > 1.0) but negative prompt "
                "tokens are missing. Tokenizer must provide an unconditional branch."
            )

        negative_prompt_embeds: Tensor | None = None
        negative_prompt_attention_mask: Tensor | None = None
        if model_inputs.do_true_cfg:
            assert model_inputs.negative_tokens is not None
            negative_prompt_embeds, negative_prompt_attention_mask = (
                self.prepare_prompt_embeddings(
                    model_inputs.negative_tokens,
                    model_inputs.negative_mask,
                    num_images_per_prompt=model_inputs.num_images_per_prompt,
                )
            )

        dtype = self.transformer.config.dtype
        device = self._transformer_device
        prompt_embeds = prompt_embeds.to(device).cast(dtype)
        prompt_attention_mask = prompt_attention_mask.to(device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device).cast(dtype)
        if negative_prompt_attention_mask is not None:
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                device
            )

        latents = Tensor.from_dlpack(
            np.ascontiguousarray(model_inputs.latents)
        ).to(device).cast(DType.float32)

        conditioning_mask: Tensor | None = None
        conditioning_mask_np: np.ndarray | None = None
        conditioned_init_latents: Tensor | None = None
        image_cond_noise_scale = float(model_inputs.image_cond_noise_scale)
        if os.getenv("MAX_LTX_IMAGE_COND_NOISE_SCALE") is not None:
            image_cond_noise_scale = float(
                os.getenv("MAX_LTX_IMAGE_COND_NOISE_SCALE", "0.025")
            )
        denoise_strength = float(model_inputs.denoise_strength)
        if os.getenv("MAX_LTX_DENOISE_STRENGTH") is not None:
            denoise_strength = float(
                os.getenv("MAX_LTX_DENOISE_STRENGTH", str(denoise_strength))
            )
        denoise_strength = float(min(max(denoise_strength, 0.0), 1.0))

        override_packed_latents_path = os.getenv(
            "MAX_LTX_OVERRIDE_PACKED_LATENTS"
        )
        if override_packed_latents_path:
            override_packed_latents = np.load(override_packed_latents_path)
            latents = Tensor.from_dlpack(
                np.ascontiguousarray(
                    override_packed_latents.astype(np.float32, copy=False)
                )
            ).to(device)
        debug_dump_prefix = os.getenv("MAX_LTX_DEBUG_DUMP_PREFIX")
        if debug_dump_prefix:
            np.save(
                f"{debug_dump_prefix}_init_latents.npy",
                np.ascontiguousarray(
                    self._to_numpy(latents).astype(np.float32, copy=False)
                ),
            )

        latent_num_frames = (
            (cast(int, model_inputs.num_frames) - 1)
            // self.vae_temporal_compression_ratio
        ) + 1
        latent_height = model_inputs.height // self.vae_spatial_compression_ratio
        latent_width = model_inputs.width // self.vae_spatial_compression_ratio
        is_conditioned = model_inputs.input_image is not None

        if is_conditioned:
            packed_latents_np = self._to_numpy(latents)
            unpacked_latents_np = self._unpack_latents_np(
                packed_latents_np,
                latent_num_frames,
                latent_height,
                latent_width,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            ).copy()

            cond_video_np = self._prepare_condition_tensor_np(
                model_inputs.input_image,
                target_height=model_inputs.height,
                target_width=model_inputs.width,
            )
            cond_video = Tensor.from_dlpack(cond_video_np).to(self.vae.devices[0]).cast(
                self.vae.dtype
            )
            cond_encoded = self.vae.encode(cond_video, return_dict=True)
            if isinstance(cond_encoded, dict):
                cond_posterior = cond_encoded["latent_dist"]
            else:
                cond_posterior = cond_encoded
            condition_latents_np = self._to_numpy(cond_posterior.mode())
            condition_latents_np = self._normalize_latents_np(
                condition_latents_np,
                self.vae.latents_mean,
                self.vae.latents_std,
                self.vae.config.scaling_factor,
            )

            cond_latent_frames = min(
                condition_latents_np.shape[2], unpacked_latents_np.shape[2]
            )
            unpacked_latents_np[:, :, :cond_latent_frames, :, :] = condition_latents_np[
                :, :, :cond_latent_frames, :, :
            ]

            packed_latents_np = self._pack_latents_np(
                unpacked_latents_np,
                patch_size=self.transformer_spatial_patch_size,
                patch_size_t=self.transformer_temporal_patch_size,
            )
            latents = Tensor.from_dlpack(packed_latents_np).to(device).cast(
                DType.float32
            )

            batch_size = int(latents.shape[0])
            seq_len = int(latents.shape[1])
            tokens_per_frame = (
                (latent_height // self.transformer_spatial_patch_size)
                * (latent_width // self.transformer_spatial_patch_size)
            )
            conditioned_seq_frames = max(
                1, cond_latent_frames // self.transformer_temporal_patch_size
            )
            conditioned_seq_tokens = min(
                seq_len, conditioned_seq_frames * tokens_per_frame
            )
            conditioning_mask_np = np.zeros(
                (batch_size, seq_len), dtype=np.float32
            )
            conditioning_mask_np[:, :conditioned_seq_tokens] = 1.0
            conditioning_mask = Tensor.from_dlpack(
                np.ascontiguousarray(conditioning_mask_np)
            ).to(device)
            conditioned_init_latents = latents

        rotary_batch_size = int(latents.shape[0]) * (
            2 if model_inputs.do_true_cfg else 1
        )
        rotary_cos, rotary_sin = self._get_rotary_embeddings(
            batch_size=rotary_batch_size,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            frames_per_second=int(model_inputs.frames_per_second or 25),
            dtype=dtype,
        )

        if is_conditioned:
            condition_timesteps_np = (
                self._linear_quadratic_schedule(
                    int(model_inputs.num_inference_steps)
                )
                * np.float32(1000.0)
            )
            # Diffusers condition pipeline path:
            # set_timesteps(timesteps=linear_quadratic_schedule * 1000)
            # -> scheduler computes shifted sigmas and appends terminal sigma.
            sigmas_np = self._shifted_sigmas_from_timesteps(
                condition_timesteps_np
            )
        else:
            condition_timesteps_np = None
            sigmas_np = np.ascontiguousarray(
                model_inputs.sigmas.astype(np.float32, copy=False)
            )

        sigmas_np, start_index = self._get_timesteps_for_strength(
            sigmas_np,
            denoise_strength,
        )
        if condition_timesteps_np is not None and start_index > 0:
            condition_timesteps_np = condition_timesteps_np[start_index:]

        # img2vid/video refinement starts from input latents and injects noise
        # proportional to the first sigma of the shortened schedule.
        if denoise_strength < 1.0 and sigmas_np.size > 0:
            sigma0 = float(sigmas_np[0])
            latents_np = self._to_numpy(latents)
            noise_np = np.random.standard_normal(latents_np.shape).astype(
                np.float32
            )
            latents_np = sigma0 * noise_np + (1.0 - sigma0) * latents_np
            latents = Tensor.from_dlpack(
                np.ascontiguousarray(latents_np.astype(np.float32, copy=False))
            ).to(device)

        sigmas_key = (
            f"{model_inputs.num_inference_steps}_"
            f"{model_inputs.height}_{model_inputs.width}_{model_inputs.num_frames}_"
            f"{'cond' if is_conditioned else 'text'}_{start_index}_{denoise_strength:.4f}"
        )
        if sigmas_key not in self._cached_sigmas:
            self._cached_sigmas[sigmas_key] = Tensor.from_dlpack(
                np.ascontiguousarray(sigmas_np)
            ).to(device)
        sigmas = self._cached_sigmas[sigmas_key]

        if condition_timesteps_np is None:
            _, all_dts = self.prepare_scheduler(sigmas)
            timesteps_for_condition_np = (
                sigmas_np[:-1] * np.float32(1000.0)
            )
        else:
            all_dts = Tensor.from_dlpack(
                np.ascontiguousarray(
                    (sigmas_np[:-1] - sigmas_np[1:]).astype(
                        np.float32, copy=False
                    )
                )
            ).to(device)
            timesteps_for_condition_np = condition_timesteps_np

        dts_seq: Any = all_dts
        if hasattr(dts_seq, "driver_tensor"):
            dts_seq = dts_seq.driver_tensor

        num_timesteps = int(sigmas_np.shape[0]) - 1

        skip_denoise = os.getenv("MAX_LTX_SKIP_DENOISE", "0") == "1"
        cond_noise_rng = np.random.RandomState(0)
        keep_mask_np: np.ndarray | None = None
        tokens_to_denoise_masks_np: list[np.ndarray] | None = None
        conditioned_init_latents_np: np.ndarray | None = None
        if conditioning_mask_np is not None:
            keep_mask_np = (
                conditioning_mask_np > np.float32(1.0 - 1e-6)
            )[..., None]

            conditioning_threshold_np = (
                1.0 - conditioning_mask_np
            ).astype(np.float32, copy=False)
            tokens_to_denoise_masks_np = []
            for t in timesteps_for_condition_np:
                t_normalized = float(t) / 1000.0
                denoise_mask_np = (
                    (t_normalized - 1e-6) < conditioning_threshold_np
                )[..., None]
                tokens_to_denoise_masks_np.append(
                    np.ascontiguousarray(denoise_mask_np.astype(np.bool_))
                )
            if (
                conditioned_init_latents is not None
                and image_cond_noise_scale > 0.0
            ):
                conditioned_init_latents_np = self._to_numpy(
                    conditioned_init_latents
                )

        if not skip_denoise:
            for i in tqdm(range(num_timesteps), desc="Denoising"):
                dt = dts_seq[i : i + 1]
                t_normalized = float(timesteps_for_condition_np[i]) / 1000.0
                timestep_value = np.float32(timesteps_for_condition_np[i])

                if conditioning_mask_np is not None:
                    timestep_np = np.minimum(
                        timestep_value,
                        (1.0 - conditioning_mask_np).astype(
                            np.float32, copy=False
                        )
                        * np.float32(1000.0),
                    ).astype(np.float32, copy=False)
                else:
                    timestep_np = np.full(
                        (int(latents.shape[0]), int(latents.shape[1])),
                        timestep_value,
                        dtype=np.float32,
                    )

                if (
                    conditioning_mask is not None
                    and conditioned_init_latents is not None
                    and image_cond_noise_scale > 0.0
                ):
                    assert keep_mask_np is not None
                    assert conditioned_init_latents_np is not None
                    cond_noise_np = cond_noise_rng.standard_normal(
                        conditioned_init_latents_np.shape
                    ).astype(np.float32)
                    noised_conditioned_np = conditioned_init_latents_np + (
                        image_cond_noise_scale * (t_normalized**2)
                    ) * cond_noise_np
                    latents_np = self._to_numpy(latents)
                    latents_np = np.where(
                        keep_mask_np, noised_conditioned_np, latents_np
                    ).astype(np.float32, copy=False)
                    latents = Tensor.from_dlpack(
                        np.ascontiguousarray(latents_np)
                    ).to(device)

                if model_inputs.do_true_cfg:
                    assert negative_prompt_embeds is not None
                    assert negative_prompt_attention_mask is not None

                    # Keep cond/uncond text sequence lengths aligned for CFG concat.
                    target_text_seq_len = max(
                        int(prompt_embeds.shape[1]),
                        int(negative_prompt_embeds.shape[1]),
                    )
                    prompt_embeds = self._pad_sequence_to_length(
                        prompt_embeds, target_length=target_text_seq_len
                    )
                    negative_prompt_embeds = self._pad_sequence_to_length(
                        negative_prompt_embeds,
                        target_length=target_text_seq_len,
                    )
                    prompt_attention_mask = self._pad_sequence_to_length(
                        prompt_attention_mask,
                        target_length=target_text_seq_len,
                        pad_value=False,
                    )
                    negative_prompt_attention_mask = (
                        self._pad_sequence_to_length(
                            negative_prompt_attention_mask,
                            target_length=target_text_seq_len,
                            pad_value=False,
                        )
                    )

                    latent_model_input = F.concat(
                        [latents, latents], axis=0
                    ).cast(dtype)
                    model_prompt_embeds = F.concat(
                        [negative_prompt_embeds, prompt_embeds], axis=0
                    )
                    model_prompt_mask = F.concat(
                        [negative_prompt_attention_mask, prompt_attention_mask],
                        axis=0,
                    )
                    timestep_model_np = np.concatenate(
                        [timestep_np, timestep_np], axis=0
                    ).astype(np.float32, copy=False)
                else:
                    latent_model_input = latents.cast(dtype)
                    model_prompt_embeds = prompt_embeds
                    model_prompt_mask = prompt_attention_mask
                    timestep_model_np = timestep_np

                timestep_model = Tensor.from_dlpack(
                    np.ascontiguousarray(timestep_model_np)
                ).to(device)

                noise_pred = self.transformer(
                    latent_model_input,
                    model_prompt_embeds,
                    timestep_model,
                    model_prompt_mask,
                    rotary_cos,
                    rotary_sin,
                )[0]

                if model_inputs.do_true_cfg:
                    noise_pred_uncond, noise_pred_text = F.chunk(
                        noise_pred, 2, axis=0
                    )
                    noise_pred = noise_pred_uncond + model_inputs.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    if debug_dump_prefix and i == 0:
                        np.save(
                            f"{debug_dump_prefix}_step0_noise_uncond.npy",
                            np.ascontiguousarray(
                                self._to_numpy(noise_pred_uncond).astype(
                                    np.float32, copy=False
                                )
                            ),
                        )
                        np.save(
                            f"{debug_dump_prefix}_step0_noise_text.npy",
                            np.ascontiguousarray(
                                self._to_numpy(noise_pred_text).astype(
                                    np.float32, copy=False
                                )
                            ),
                        )

                if debug_dump_prefix and i == 0:
                    np.save(
                        f"{debug_dump_prefix}_step0_noise_cfg.npy",
                        np.ascontiguousarray(
                            self._to_numpy(noise_pred.cast(DType.float32)).astype(
                                np.float32, copy=False
                            )
                        ),
                    )

                step_noise_pred = noise_pred.cast(DType.float32)
                if conditioning_mask is not None:
                    step_noise_pred = -step_noise_pred

                denoised_latents = self.scheduler_step(
                    latents,
                    step_noise_pred,
                    dt,
                )

                if conditioning_mask is not None:
                    assert tokens_to_denoise_masks_np is not None
                    latents_np = self._to_numpy(latents)
                    denoised_latents_np = self._to_numpy(denoised_latents)
                    latents_np = np.where(
                        tokens_to_denoise_masks_np[i],
                        denoised_latents_np,
                        latents_np,
                    ).astype(np.float32, copy=False)
                    latents = Tensor.from_dlpack(
                        np.ascontiguousarray(latents_np)
                    ).to(device)
                else:
                    latents = denoised_latents

                if debug_dump_prefix and i == 0:
                    np.save(
                        f"{debug_dump_prefix}_step1_latents.npy",
                        np.ascontiguousarray(
                            self._to_numpy(latents).astype(np.float32, copy=False)
                        ),
                    )

        packed_latents_np = self._to_numpy(latents)
        save_packed_latents_path = os.getenv("MAX_LTX_SAVE_PACKED_LATENTS")
        if save_packed_latents_path:
            np.save(
                save_packed_latents_path,
                np.ascontiguousarray(
                    packed_latents_np.astype(np.float32, copy=False)
                ),
            )
        latents_np = self._unpack_latents_np(
            packed_latents_np,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents_np = self._denormalize_latents_np(
            latents_np,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        timestep_tensor: Tensor | None = None
        if self.vae.config.timestep_conditioning:
            batch_size = latents_np.shape[0]
            decode_timestep = float(model_inputs.decode_timestep or 0.0)
            decode_noise_scale = model_inputs.decode_noise_scale
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep

            noise = np.random.standard_normal(latents_np.shape).astype(np.float32)
            latents_np = (
                (1.0 - decode_noise_scale) * latents_np
                + decode_noise_scale * noise
            )
            timestep_np = np.full(
                (batch_size,), decode_timestep, dtype=np.float32
            )
            timestep_tensor = Tensor.from_dlpack(timestep_np).to(device).cast(
                dtype
            )

        latents_tensor = Tensor.from_dlpack(
            np.ascontiguousarray(latents_np.astype(np.float32, copy=False))
        ).to(device)
        latents_tensor = latents_tensor.cast(self.vae.dtype)

        video = self.vae.decode(latents_tensor, timestep_tensor)
        video_np = self._to_numpy(video)

        # [B, C, F, H, W] -> [B, F, H, W, C]
        video_np = np.transpose(video_np, (0, 2, 3, 4, 1))
        video_np = self._resize_video_nearest(
            video_np,
            target_frames=int(model_inputs.num_frames or video_np.shape[1]),
            target_height=int(model_inputs.height),
            target_width=int(model_inputs.width),
        )
        if video_np.min() < 0.0 or video_np.max() > 1.0:
            video_np = (video_np * 0.5 + 0.5).clip(0.0, 1.0)
        else:
            video_np = video_np.clip(0.0, 1.0)

        if callback_queue is not None:
            callback_queue.put_nowait(video_np)

        return LTXPipelineOutput(
            videos=video_np.astype(np.float32, copy=False),
            frames_per_second=model_inputs.frames_per_second or 25,
        )
