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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.graph.weights import load_weights
from max.interfaces import PixelGenerationContext, TokenBuffer
from max.pipelines.lib.diffusion_schedulers import UniPCMultistepScheduler
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    PixelModelInputs,
    max_compile,
)
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import Tracer, traced
from tqdm import tqdm

from ..autoencoders import AutoencoderKLWanModel
from ..umt5 import UMT5Model
from .model import WanTransformerModel

logger = logging.getLogger(__name__)


def _maybe_save_debug_npy(path_str: str | None, array: np.ndarray) -> None:
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    logger.info(
        "Saved Wan debug tensor to %s shape=%s dtype=%s min=%.5f max=%.5f mean=%.5f std=%.5f",
        path,
        array.shape,
        array.dtype,
        float(array.min()),
        float(array.max()),
        float(array.mean()),
        float(array.std()),
    )


@dataclass(kw_only=True)
class WanModelInputs(PixelModelInputs):
    mask: npt.NDArray[np.bool_] | None = None
    negative_mask: npt.NDArray[np.bool_] | None = None
    width: int = 832
    height: int = 480
    num_frames: int = 81
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    guidance_scale_2: float | None = None
    boundary_ratio: float | None = None
    expand_timesteps: bool = False
    num_images_per_prompt: int = 1


@dataclass
class WanPipelineOutput:
    images: np.ndarray | Buffer


class WanPipeline(DiffusionPipeline):
    """Wan diffusion pipeline with MAX-native DiT/VAE interfaces.

    Supports Wan 2.2 MoE models with dual transformers (high-noise and
    low-noise experts) when ``transformer_2`` weights are present.
    """

    vae: AutoencoderKLWanModel
    text_encoder: UMT5Model
    transformer: WanTransformerModel
    transformer_2: WanTransformerModel | None

    components = {
        "vae": AutoencoderKLWanModel,
        "text_encoder": UMT5Model,
        "transformer": WanTransformerModel,
    }

    def _load_sub_models(
        self, weight_paths: list[Path],
    ) -> dict[str, ComponentModel]:
        """Load all sub-models, including optional transformer_2 for MoE."""
        diffusers_config = self.pipeline_config.model.diffusers_config or {}
        components_cfg = diffusers_config.get("components", {})
        relative_paths = self._resolve_relative_component_paths()

        models: dict[str, ComponentModel] = {}
        for name, component_cls in tqdm(
            self.components.items(), desc="Loading sub models"
        ):
            if not issubclass(component_cls, ComponentModel):
                continue
            config_dict = self._get_component_config_dict(
                components_cfg, name
            )
            abs_paths = self._resolve_absolute_paths(
                weight_paths, relative_paths[name]
            )
            models[name] = component_cls(
                config=config_dict,
                encoding=self.pipeline_config.model.quantization_encoding,
                devices=self.devices,
                weights=load_weights(abs_paths),
            )

        # Optionally load transformer_2 (low-noise expert) for MoE models.
        if "transformer_2" in relative_paths:
            config_dict = self._get_component_config_dict(
                components_cfg, "transformer_2"
            )
            abs_paths = self._resolve_absolute_paths(
                weight_paths, relative_paths["transformer_2"]
            )
            models["transformer_2"] = WanTransformerModel(
                config=config_dict,
                encoding=self.pipeline_config.model.quantization_encoding,
                devices=self.devices,
                weights=load_weights(abs_paths),
            )
        else:
            self.transformer_2 = None

        return models

    def init_remaining_components(self) -> None:
        self.vae_scale_factor_temporal = int(
            getattr(self.vae.config, "scale_factor_temporal", 4) or 4
        )
        self.vae_scale_factor_spatial = int(
            getattr(self.vae.config, "scale_factor_spatial", 8) or 8
        )
        diffusers_config = self.pipeline_config.model.diffusers_config or {}
        self.boundary_ratio = diffusers_config.get("boundary_ratio")
        components_cfg = diffusers_config.get("components", {})
        scheduler_cfg = components_cfg.get("scheduler", {}).get(
            "config_dict", {}
        )
        self.num_train_timesteps = int(
            scheduler_cfg.get("num_train_timesteps", 1000)
        )
        transformer_cfg = components_cfg.get("transformer", {}).get(
            "config_dict", {}
        )
        self.expand_timesteps = bool(
            transformer_cfg.get("expand_timesteps", False)
        )

        # Initialize UniPC multistep scheduler for proper 2nd-order stepping.
        self._scheduler = UniPCMultistepScheduler(**scheduler_cfg)
        self._base_flow_shift = float(scheduler_cfg.get("flow_shift", 1.0))

        # Pre-compute VAE denormalization constants on GPU (avoids
        # CPU->GPU transfer every call).
        device = self.transformer.devices[0]
        z_dim = int(self.vae.config.z_dim)
        mean_arr = np.asarray(
            self.vae.config.latents_mean, dtype=np.float32,
        ).reshape(1, z_dim, 1, 1, 1)
        std_arr = np.asarray(
            self.vae.config.latents_std, dtype=np.float32,
        ).reshape(1, z_dim, 1, 1, 1)
        self._vae_mean_t = Tensor.from_dlpack(mean_arr).to(device)
        self._vae_std_t = Tensor.from_dlpack(std_arr).to(device)

        self.build_guidance_model()

        # Pre-compile transformers with symbolic dims (no recompilation needed).
        self.transformer.compile_model()
        if self.transformer_2 is not None:
            self.transformer_2.compile_model()
        self.vae.prepare_for_serving()
        self._cached_spatial_shapes: dict[str, Buffer] = {}
        self._cached_batched_timesteps: dict[str, list[Buffer]] = {}

    def build_guidance_model(self) -> None:
        """Compile classifier-free guidance: uncond + scale * (cond - uncond)."""
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype
        latent_type = TensorType(
            dtype,
            shape=["batch", "channels", "frames", "height", "width"],
            device=device,
        )
        input_types = [
            latent_type,  # noise_pred
            latent_type,  # noise_uncond
            TensorType(dtype, shape=[1], device=device),  # guidance_scale
        ]

        self.__dict__["_guidance_model"] = max_compile(
            self._guidance_model, input_types=input_types
        )

    def _guidance_model(
        self, noise_pred: Tensor, noise_uncond: Tensor, scale: Tensor
    ) -> Tensor:
        return noise_uncond + scale * (noise_pred - noise_uncond)

    def prepare_inputs(self, context: PixelGenerationContext) -> WanModelInputs:
        model_inputs = WanModelInputs.from_context(context)
        if hasattr(context, "num_frames") and context.num_frames is not None:
            requested_num_frames = int(context.num_frames)
            model_inputs.num_frames = requested_num_frames
            if model_inputs.latents.ndim == 5:
                latent_frames = int(model_inputs.latents.shape[2])
                model_inputs.num_frames = self._normalize_num_frames_for_wan(
                    requested_num_frames=requested_num_frames,
                    latent_frames=latent_frames,
                )
        if (
            hasattr(context, "guidance_scale_2")
            and context.guidance_scale_2 is not None
        ):
            model_inputs.guidance_scale_2 = context.guidance_scale_2
        return model_inputs

    @traced
    def execute(  # type: ignore[override]
        self,
        model_inputs: WanModelInputs,
        **kwargs: object,
    ) -> WanPipelineOutput:
        del kwargs
        device = self.transformer.devices[0]

        with Tracer("encode_prompt"):
            prompt_embeds = self._get_t5_prompt_embeds(
                tokens=model_inputs.tokens,
                attention_mask=model_inputs.mask,
                num_videos_per_prompt=model_inputs.num_images_per_prompt,
                max_sequence_length=int(
                    model_inputs.tokens.array.shape[-1]
                ),
            )
            do_cfg = (
                model_inputs.guidance_scale > 1.0
                and model_inputs.negative_tokens is not None
            )
            negative_prompt_embeds: Tensor | None = None
            if do_cfg and model_inputs.negative_tokens is not None:
                negative_prompt_embeds = self._get_t5_prompt_embeds(
                    tokens=model_inputs.negative_tokens,
                    attention_mask=model_inputs.negative_mask,
                    num_videos_per_prompt=model_inputs.num_images_per_prompt,
                    max_sequence_length=int(
                        model_inputs.tokens.array.shape[-1]
                    ),
                )

        with Tracer("prepare_latents"):
            latents = (
                Tensor.from_dlpack(model_inputs.latents)
                .to(device)
                .cast(DType.float32)
            )
            transformer_dtype = prompt_embeds.dtype

            boundary_timestep = self.compute_boundary_timestep(
                boundary_ratio=model_inputs.boundary_ratio
                if model_inputs.boundary_ratio is not None
                else self.boundary_ratio,
                num_train_timesteps=self.num_train_timesteps,
            )

            rope_cos, rope_sin = self.transformer.compute_rope(
                num_frames=int(latents.shape[2]),
                height=int(latents.shape[3]),
                width=int(latents.shape[4]),
            )

            num_steps = model_inputs.num_inference_steps
            if self._scheduler.use_flow_sigmas:
                selected_flow_shift = self._select_flow_shift(
                    model_inputs.height, model_inputs.width
                )
                self._scheduler.flow_shift = selected_flow_shift
                logger.info(
                    "Wan scheduler flow_shift=%s for %dx%d",
                    selected_flow_shift,
                    model_inputs.width,
                    model_inputs.height,
                )
            self._scheduler.set_timesteps(num_steps)
            scheduler_timesteps = self._scheduler.timesteps
            assert scheduler_timesteps is not None

            batch_size = int(latents.shape[0])
            batched_timesteps = self._get_batched_timesteps(
                scheduler_timesteps=scheduler_timesteps,
                batch_size=batch_size,
                device=device,
            )

            # Convert prompt_embeds Tensor -> Buffer for transformer
            prompt_embeds_buf = prompt_embeds.driver_tensor
            negative_prompt_embeds_buf: Buffer | None = None
            if negative_prompt_embeds is not None:
                negative_prompt_embeds_buf = negative_prompt_embeds.driver_tensor

            guidance_scale_high: Tensor | None = None
            guidance_scale_low: Tensor | None = None
            if do_cfg:
                guidance_scale_high = Tensor.full(
                    [1],
                    float(model_inputs.guidance_scale),
                    dtype=prompt_embeds.dtype,
                    device=device,
                )
                gs2 = (
                    model_inputs.guidance_scale_2
                    if model_inputs.guidance_scale_2 is not None
                    else model_inputs.guidance_scale
                )
                guidance_scale_low = Tensor.full(
                    [1],
                    float(gs2),
                    dtype=prompt_embeds.dtype,
                    device=device,
                )

            has_moe = (
                self.transformer_2 is not None
                and boundary_timestep is not None
            )
            boundary_step_idx = num_steps
            if has_moe:
                assert boundary_timestep is not None
                for idx in range(num_steps):
                    if float(scheduler_timesteps[idx]) < boundary_timestep:
                        boundary_step_idx = idx
                        break

        # Build a tiny buffer carrying post-patch [frames, height, width]
        # so compiled post-processing can recover symbolic dims.
        p_t, p_h, p_w = self.transformer.config.patch_size
        ppf = int(latents.shape[2]) // p_t
        pph = int(latents.shape[3]) // p_h
        ppw = int(latents.shape[4]) // p_w
        spatial_shape = self._get_spatial_shape(ppf, pph, ppw, device)

        high_noise_model = self.transformer.model
        assert high_noise_model is not None

        latents_np_cache: np.ndarray | None = None
        with Tracer("denoising_high_noise"):
            latents, latents_np_cache = self._run_denoising_phase(
                latents=latents,
                transformer_model=high_noise_model,
                prompt_embeds=prompt_embeds_buf,
                negative_prompt_embeds=negative_prompt_embeds_buf,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                scheduler_timesteps=scheduler_timesteps,
                batched_timesteps=batched_timesteps,
                do_cfg=do_cfg,
                guidance_scale=guidance_scale_high,
                device=device,
                step_range=range(boundary_step_idx),
                desc="Denoising (high-noise)"
                if has_moe
                else "Denoising",
                spatial_shape=spatial_shape,
                latents_np=latents_np_cache,
            )

        if has_moe and boundary_step_idx < num_steps:
            low_noise_model = (
                self.transformer_2.model if self.transformer_2 else None
            )
            assert low_noise_model is not None

            with Tracer("denoising_low_noise"):
                latents, latents_np_cache = self._run_denoising_phase(
                    latents=latents,
                    transformer_model=low_noise_model,
                    prompt_embeds=prompt_embeds_buf,
                    negative_prompt_embeds=negative_prompt_embeds_buf,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    scheduler_timesteps=scheduler_timesteps,
                    batched_timesteps=batched_timesteps,
                    do_cfg=do_cfg,
                    guidance_scale=guidance_scale_low,
                    device=device,
                    step_range=range(boundary_step_idx, num_steps),
                    desc="Denoising (low-noise)",
                    spatial_shape=spatial_shape,
                    latents_np=latents_np_cache,
                )

        with Tracer("denormalize_latents"):
            denorm_latents = latents * self._vae_std_t + self._vae_mean_t
            denorm_latents = denorm_latents.cast(transformer_dtype)
            raw_latents_dump = os.getenv("WAN_DUMP_RAW_LATENTS_NPY")
            denorm_latents_dump = os.getenv("WAN_DUMP_DENORM_LATENTS_NPY")
            if raw_latents_dump:
                assert latents_np_cache is not None
                _maybe_save_debug_npy(
                    raw_latents_dump,
                    np.ascontiguousarray(latents_np_cache, dtype=np.float32),
                )
            if denorm_latents_dump:
                denorm_latents_np = np.from_dlpack(
                    denorm_latents.cast(DType.float32).to(CPU())
                )
                _maybe_save_debug_npy(
                    denorm_latents_dump,
                    np.ascontiguousarray(denorm_latents_np, dtype=np.float32),
                )

        with Tracer("vae_decode"):
            decoded_video = self.vae.decode_5d(denorm_latents)
            decoded_num_frames = int(decoded_video.shape[2])
            target_num_frames = min(
                decoded_num_frames, int(model_inputs.num_frames)
            )
            if decoded_num_frames != int(model_inputs.num_frames):
                logger.warning(
                    "Wan VAE decode produced %d frames for requested %d; "
                    "continuing with %d decoded frames.",
                    decoded_num_frames,
                    int(model_inputs.num_frames),
                    target_num_frames,
                )
            decoded_video = decoded_video[
                :,
                :,
                : target_num_frames,
                : model_inputs.height,
                : model_inputs.width,
            ]

        return WanPipelineOutput(images=self._to_numpy(decoded_video))

    def _run_denoising_phase(
        self,
        latents: Tensor,
        transformer_model: Any,
        prompt_embeds: Buffer,
        negative_prompt_embeds: Buffer | None,
        rope_cos: Buffer,
        rope_sin: Buffer,
        scheduler_timesteps: np.ndarray,
        batched_timesteps: list[Buffer],
        do_cfg: bool,
        guidance_scale: Tensor | None,
        device: Device,
        step_range: range,
        desc: str,
        spatial_shape: Buffer,
        latents_np: np.ndarray | None = None,
    ) -> tuple[Tensor, np.ndarray]:
        """Run a denoising phase using UniPC multistep scheduler.

        Transformer forward passes run on GPU (takes/returns Buffer).
        Guidance model uses compiled Tensor API. Scheduler step runs on
        CPU via numpy (cheap for the small latent tensor, avoids lazy
        graph accumulation that causes quadratic slowdown).
        """
        sched = self._scheduler
        transformer_dtype = DType.bfloat16
        cpu = CPU()

        for i in tqdm(step_range, desc=desc):
            dit_timestep = batched_timesteps[i]

            # Cast latents to bf16 Buffer for transformer
            latent_model_input = latents.cast(transformer_dtype).driver_tensor
            noise_pred_buf: Buffer = transformer_model(
                latent_model_input, dit_timestep, prompt_embeds,
                rope_cos, rope_sin, spatial_shape,
            )
            # Wrap Buffer -> Tensor for guidance model / scheduler
            noise_pred = Tensor.from_dlpack(noise_pred_buf)

            if do_cfg and negative_prompt_embeds is not None:
                assert guidance_scale is not None
                noise_uncond_buf: Buffer = transformer_model(
                    latent_model_input, dit_timestep, negative_prompt_embeds,
                    rope_cos, rope_sin, spatial_shape,
                )
                noise_uncond = Tensor.from_dlpack(noise_uncond_buf)
                noise_pred = self._guidance_model(
                    noise_pred, noise_uncond, guidance_scale,
                )
                del noise_uncond

            # Scheduler step on CPU via numpy.
            # Keep a CPU-side latents cache to avoid re-downloading latents
            # from GPU every step.
            if latents_np is None:
                latents_np = np.from_dlpack(
                    latents.cast(DType.float32).to(cpu)
                )
            noise_np = np.from_dlpack(
                noise_pred.cast(DType.float32).to(cpu)
            )

            latents_np = sched.step(
                noise_np, int(scheduler_timesteps[i]), latents_np
            )

            # Back to GPU as bfloat16.
            latents = Tensor.from_dlpack(
                np.ascontiguousarray(latents_np, dtype=np.float32)
            ).cast(DType.bfloat16).to(device)

        assert latents_np is not None
        return latents, latents_np

    def _get_spatial_shape(
        self, ppf: int, pph: int, ppw: int, device: Device
    ) -> Buffer:
        key = f"{ppf}_{pph}_{ppw}_{device.id}"
        cached = self._cached_spatial_shapes.get(key)
        if cached is not None:
            return cached
        spatial_np = np.zeros((ppf, pph, ppw), dtype=np.int8)
        spatial_shape = Buffer.from_numpy(spatial_np).to(device)
        self._cached_spatial_shapes[key] = spatial_shape
        return spatial_shape

    def _get_batched_timesteps(
        self,
        scheduler_timesteps: np.ndarray,
        batch_size: int,
        device: Device,
    ) -> list[Buffer]:
        key = (
            f"{batch_size}_{len(scheduler_timesteps)}_"
            f"{int(scheduler_timesteps[0])}_{int(scheduler_timesteps[-1])}_"
            f"{device.id}"
        )
        cached = self._cached_batched_timesteps.get(key)
        if cached is not None:
            return cached

        batched_timesteps = [
            Buffer.from_numpy(
                np.full([batch_size], float(int(step_value)), dtype=np.float32)
            ).to(device)
            for step_value in scheduler_timesteps
        ]
        self._cached_batched_timesteps[key] = batched_timesteps
        return batched_timesteps

    def _select_flow_shift(self, height: int, width: int) -> float:
        """Choose scheduler flow shift by resolution for better Wan quality."""
        if height >= 720 or width >= 1280:
            return 5.0
        return 3.0 if self._base_flow_shift <= 3.0 else self._base_flow_shift

    def _get_t5_prompt_embeds(
        self,
        tokens: TokenBuffer,
        attention_mask: npt.NDArray[np.bool_] | None,
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ) -> Tensor:
        token_ids = tokens.array
        if token_ids.ndim == 1:
            token_ids = np.expand_dims(token_ids, axis=0)

        if attention_mask is None:
            # Derive mask from token_ids: non-zero tokens are real.
            attention_mask = token_ids != 0
        if attention_mask.ndim == 1:
            attention_mask = np.expand_dims(attention_mask, axis=0)

        text_input_ids = Tensor.constant(
            token_ids,
            dtype=DType.int64,
            device=self.text_encoder.devices[0],
        )
        text_attention_mask = Tensor.constant(
            attention_mask.astype(np.int64, copy=False),
            dtype=DType.int64,
            device=self.text_encoder.devices[0],
        )
        hidden_states = self.text_encoder(text_input_ids, text_attention_mask)
        return self.get_t5_prompt_embeds_from_hidden(
            hidden_states=hidden_states,
            attention_mask=text_attention_mask,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    @staticmethod
    def get_t5_prompt_embeds_from_hidden(
        hidden_states: Tensor,
        attention_mask: Tensor,
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ) -> Tensor:
        if hidden_states.rank != 3:
            raise ValueError(
                "Expected hidden_states rank=3 [B, S, D] for Wan text encoder."
            )
        if attention_mask.rank != 2:
            raise ValueError(
                "Expected attention_mask rank=2 [B, S] for Wan text encoder."
            )

        batch_size = int(hidden_states.shape[0])
        hidden_dim = int(hidden_states.shape[2])
        rows: list[Tensor] = []
        for batch_idx in range(batch_size):
            seq_len = int(F.sum(attention_mask[batch_idx], axis=0).item())
            seq_len = max(0, min(seq_len, int(hidden_states.shape[1])))
            row = hidden_states[batch_idx, :seq_len, :]
            if seq_len < max_sequence_length:
                pad = Tensor.zeros(
                    [max_sequence_length - seq_len, hidden_dim],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                row = F.concat([row, pad], axis=0)
            elif seq_len > max_sequence_length:
                row = row[:max_sequence_length, :]
            rows.append(row)

        prompt_embeds = F.stack(rows, axis=0)

        if num_videos_per_prompt <= 1:
            return prompt_embeds

        batch_size = int(prompt_embeds.shape[0])
        seq_len = int(prompt_embeds.shape[1])
        prompt_embeds = F.tile(prompt_embeds, (1, num_videos_per_prompt, 1))
        return F.reshape(
            prompt_embeds,
            (
                batch_size * num_videos_per_prompt,
                seq_len,
                int(prompt_embeds.shape[2]),
            ),
        )

    def _normalize_num_frames_for_wan(
        self,
        requested_num_frames: int,
        latent_frames: int,
    ) -> int:
        compatible_num_frames = max(
            1,
            (max(latent_frames, 1) - 1) * self.vae_scale_factor_temporal + 1,
        )
        if requested_num_frames <= compatible_num_frames:
            return requested_num_frames

        logger.warning(
            "Requested Wan num_frames=%d is incompatible with latent temporal "
            "shape (%d latent frames). Auto-adjusting output frame count to %d.",
            requested_num_frames,
            latent_frames,
            compatible_num_frames,
        )
        return compatible_num_frames

    @staticmethod
    def compute_boundary_timestep(
        boundary_ratio: float | None,
        num_train_timesteps: int,
    ) -> float | None:
        if boundary_ratio is None:
            return None
        return boundary_ratio * num_train_timesteps

    @staticmethod
    def _to_numpy(image: Tensor) -> np.ndarray:
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)
