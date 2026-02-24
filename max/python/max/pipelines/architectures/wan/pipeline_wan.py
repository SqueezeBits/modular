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

import gc
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from max import functional as F
from max.driver import CPU, Device
from max.dtype import DType
from max.graph import TensorType, ops
from max.graph.weights import load_weights
from max.interfaces import PixelGenerationContext, TokenBuffer
from max.pipelines.lib.interfaces import (
    CompileWrapper,
    DiffusionPipeline,
    PixelModelInputs,
    max_compile,
)
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.tensor import Tensor
from tqdm import tqdm

from ..autoencoders import AutoencoderKLWanModel
from ..umt5 import UMT5Model
from .model import WanTransformerModel

logger = logging.getLogger(__name__)


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
    images: np.ndarray | Tensor


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
            logger.info("Loaded transformer_2 (low-noise expert) for MoE")
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

        # Compile utility graphs for optimized denoising loop.
        # In VAE-only mode (WAN_LOAD_LATENTS set), skip compilation to avoid
        # allocating GPU memory for transformer-related graphs.
        if not os.environ.get("WAN_LOAD_LATENTS"):
            self._scheduler_step_model = self._build_scheduler_step_model()
            self._timestep_model = self._build_all_timesteps_model()
            self._cfg_model = self._build_cfg_model()
        else:
            self._scheduler_step_model = None
            self._timestep_model = None
            self._cfg_model = None

    def _build_scheduler_step_model(self) -> CompileWrapper:
        """Compile Euler step: latents + dt * noise_pred."""
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype
        input_types = [
            TensorType(
                dtype,
                shape=["batch", "channels", "frames", "height", "width"],
                device=device,
            ),
            TensorType(
                dtype,
                shape=["batch", "channels", "frames", "height", "width"],
                device=device,
            ),
            TensorType(dtype, shape=[1], device=device),
        ]

        def scheduler_step(latents_in, noise_pred_in, dt_in):
            return latents_in + dt_in * noise_pred_in

        return max_compile(scheduler_step, input_types=input_types)

    def _build_all_timesteps_model(self) -> CompileWrapper:
        """Compile graph to precompute all timesteps and dt values."""
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype
        input_types = [
            TensorType(DType.float32, ["num_sigmas"], device=device),
        ]

        def all_timesteps(sigmas):
            sigmas_curr = ops.slice_tensor(sigmas, [slice(0, -1)])
            all_t = ops.cast(sigmas_curr, dtype)
            sigmas_next = ops.slice_tensor(sigmas, [slice(1, None)])
            dt_f32 = ops.sub(sigmas_next, sigmas_curr)
            all_dt = ops.cast(dt_f32, dtype)
            return all_t, all_dt

        return max_compile(all_timesteps, input_types=input_types)

    def _build_cfg_model(self) -> CompileWrapper:
        """Compile CFG: noise_uncond + scale * (noise_pred - noise_uncond)."""
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

        def cfg(noise_pred, noise_uncond, scale):
            return noise_uncond + scale * (noise_pred - noise_uncond)

        return max_compile(cfg, input_types=input_types)

    def prepare_inputs(self, context: PixelGenerationContext) -> WanModelInputs:
        model_inputs = WanModelInputs.from_context(context)
        if hasattr(context, "num_frames") and context.num_frames is not None:
            model_inputs.num_frames = context.num_frames
        if (
            hasattr(context, "guidance_scale_2")
            and context.guidance_scale_2 is not None
        ):
            model_inputs.guidance_scale_2 = context.guidance_scale_2
        return model_inputs

    def execute(  # type: ignore[override]
        self,
        model_inputs: WanModelInputs,
        **kwargs: object,
    ) -> WanPipelineOutput:
        del kwargs
        device = self.transformer.devices[0]

        load_path = os.environ.get("WAN_LOAD_LATENTS")

        if load_path:
            # VAE-only mode: load pre-computed latents, skip denoising.
            logger.info(
                "Loading latents from %s (skipping denoising)", load_path
            )
            latents_np = np.load(load_path)
            latents = (
                Tensor.from_dlpack(latents_np)
                .to(device)
                .cast(self.vae.config.dtype)
            )
        else:
            # Full pipeline: text encoding + denoising.
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

            latents = (
                Tensor.from_dlpack(model_inputs.latents)
                .to(device)
                .cast(prompt_embeds.dtype)
            )

            boundary_timestep = self.compute_boundary_timestep(
                boundary_ratio=model_inputs.boundary_ratio
                if model_inputs.boundary_ratio is not None
                else self.boundary_ratio,
                num_train_timesteps=self.num_train_timesteps,
            )

            # Precompute 3D RoPE once (shape is constant across steps)
            rope_cos, rope_sin = self.transformer.compute_rope(
                num_frames=int(latents.shape[2]),
                height=int(latents.shape[3]),
                width=int(latents.shape[4]),
            )

            # Precompute all timesteps and dt values on device
            sigmas_np = model_inputs.sigmas.astype(
                np.float32, copy=False
            )
            sigmas_tensor = Tensor.from_dlpack(
                np.ascontiguousarray(sigmas_np)
            ).to(device)

            _all_timesteps, all_dts = self._timestep_model(sigmas_tensor)

            # Denoising loop
            timesteps_np = model_inputs.timesteps.astype(
                np.float32, copy=False
            )
            num_steps = len(timesteps_np)
            batch_size = int(latents.shape[0])

            # Precompute guidance scale tensors
            scale_t_high: Tensor | None = None
            scale_t_low: Tensor | None = None
            if do_cfg:
                scale_t_high = Tensor.full(
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
                scale_t_low = Tensor.full(
                    [1],
                    float(gs2),
                    dtype=prompt_embeds.dtype,
                    device=device,
                )

            # Determine the boundary step index for MoE switching.
            has_moe = (
                self.transformer_2 is not None
                and boundary_timestep is not None
            )
            boundary_step_idx = num_steps
            if has_moe:
                for idx in range(num_steps):
                    if float(timesteps_np[idx]) < boundary_timestep:
                        boundary_step_idx = idx
                        break

            # Phase 1: High-noise expert
            self.transformer.compile_model(
                latents, prompt_embeds, rope_cos
            )
            assert self.transformer.model is not None

            latents = self._run_denoising_phase(
                latents=latents,
                transformer_model=self.transformer.model,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                timesteps_np=timesteps_np,
                all_dts=all_dts,
                batch_size=batch_size,
                do_cfg=do_cfg,
                scale_t=scale_t_high,
                device=device,
                step_range=range(boundary_step_idx),
                desc="Denoising (high-noise)"
                if has_moe
                else "Denoising",
            )

            # Phase 2: Low-noise expert (if MoE)
            if has_moe and boundary_step_idx < num_steps:
                assert self.transformer_2 is not None
                self.transformer_2.compile_model(
                    latents, prompt_embeds, rope_cos
                )
                assert self.transformer_2.model is not None

                latents = self._run_denoising_phase(
                    latents=latents,
                    transformer_model=self.transformer_2.model,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    timesteps_np=timesteps_np,
                    all_dts=all_dts,
                    batch_size=batch_size,
                    do_cfg=do_cfg,
                    scale_t=scale_t_low,
                    device=device,
                    step_range=range(boundary_step_idx, num_steps),
                    desc="Denoising (low-noise)",
                )

            # Optionally save denoised latents for later VAE-only runs.
            save_path = os.environ.get("WAN_SAVE_LATENTS")
            if save_path:
                latents_cpu = latents.cast(DType.float32).to(CPU())
                np.save(save_path, np.from_dlpack(latents_cpu))
                logger.info(
                    "Saved denoised latents to %s, shape=%s",
                    save_path,
                    latents_cpu.shape,
                )
                del latents_cpu

        # Free transformer GPU memory before VAE decode.
        # In VAE-only mode this is a no-op (transformers were never compiled).
        self._offload_transformers()

        # Denormalize and decode
        denorm_latents = self.denormalize_vae_latents(
            latents=latents,
            latents_mean=tuple(self.vae.config.latents_mean),
            latents_std=tuple(self.vae.config.latents_std),
            z_dim=int(self.vae.config.z_dim),
        )

        decoded_video = self.vae.decode_5d(denorm_latents)

        decoded_video = decoded_video[
            :,
            :,
            : model_inputs.num_frames,
            : model_inputs.height,
            : model_inputs.width,
        ]
        return WanPipelineOutput(images=self._to_numpy(decoded_video))

    def _run_denoising_phase(
        self,
        latents: Tensor,
        transformer_model: Any,
        prompt_embeds: Tensor,
        negative_prompt_embeds: Tensor | None,
        rope_cos: Tensor,
        rope_sin: Tensor,
        timesteps_np: np.ndarray,
        all_dts: Tensor,
        batch_size: int,
        do_cfg: bool,
        scale_t: Tensor | None,
        device: Device,
        step_range: range,
        desc: str,
    ) -> Tensor:
        """Run a denoising phase with a single transformer model."""
        for i in tqdm(step_range, desc=desc):
            step_value = float(timesteps_np[i])

            dit_timestep = Tensor.full(
                [batch_size],
                step_value,
                dtype=prompt_embeds.dtype,
                device=device,
            )

            noise_pred = transformer_model(
                latents, dit_timestep, prompt_embeds,
                rope_cos, rope_sin,
            )

            if do_cfg and negative_prompt_embeds is not None:
                noise_uncond = transformer_model(
                    latents, dit_timestep, negative_prompt_embeds,
                    rope_cos, rope_sin,
                )
                noise_pred = self._cfg_model(
                    noise_pred, noise_uncond, scale_t,
                )
                del noise_uncond

            dt = all_dts[i : i + 1]
            latents = self._scheduler_step_model(latents, noise_pred, dt)

        return latents

    @staticmethod
    def _clear_compile_wrapper(cw: Any) -> None:
        """Explicitly release GPU resources held by a CompileWrapper."""
        if cw is None:
            return
        if hasattr(cw, "_compiled_module"):
            cw._compiled_module = None
        if hasattr(cw, "_compiled_model"):
            cw._compiled_model = None

    def _offload_transformers(self) -> None:
        """Free transformer compiled models from GPU to make room for VAE.

        The compiled _BlockLevelModel holds GPU weight copies and graph
        workspaces.  We explicitly clear every CompileWrapper's internal
        compiled module/model to ensure GPU buffers are released.
        The CPU-resident state dict is preserved so compile_model() can
        re-compile on the next request.
        """
        # Explicitly release GPU resources from each compiled block.
        for transformer in [self.transformer, self.transformer_2]:
            if transformer is None or transformer.model is None:
                continue
            blm = transformer.model
            self._clear_compile_wrapper(blm.pre)
            for block in blm.blocks:
                self._clear_compile_wrapper(block)
            self._clear_compile_wrapper(blm.post)
            blm.blocks.clear()
            del blm
            transformer.model = None

        # Free utility compiled models (scheduler step, timestep, CFG).
        for attr in (
            "_scheduler_step_model",
            "_timestep_model",
            "_cfg_model",
        ):
            cw = getattr(self, attr, None)
            self._clear_compile_wrapper(cw)

        # Free text encoder compiled model.
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            self.text_encoder.model = None

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
            attention_mask = np.ones_like(token_ids, dtype=np.bool_)
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

    @staticmethod
    def compute_boundary_timestep(
        boundary_ratio: float | None,
        num_train_timesteps: int,
    ) -> float | None:
        if boundary_ratio is None:
            return None
        return boundary_ratio * num_train_timesteps

    @staticmethod
    def use_low_noise_transformer(
        timestep: float,
        boundary_timestep: float | None,
    ) -> bool:
        if boundary_timestep is None:
            return False
        return timestep < boundary_timestep

    @staticmethod
    def denormalize_vae_latents(
        latents: Tensor,
        latents_mean: Sequence[float],
        latents_std: Sequence[float],
        z_dim: int,
    ) -> Tensor:
        if len(latents_mean) != z_dim:
            raise ValueError(
                "latents_mean length must match z_dim. "
                f"Got {len(latents_mean)} vs {z_dim}."
            )
        if len(latents_std) != z_dim:
            raise ValueError(
                "latents_std length must match z_dim. "
                f"Got {len(latents_std)} vs {z_dim}."
            )

        dtype_np = np.float32
        if latents.dtype == DType.float16:
            dtype_np = np.float16

        mean_arr = np.asarray(latents_mean, dtype=dtype_np).reshape(
            1, z_dim, 1, 1, 1
        )
        std_arr = np.asarray(latents_std, dtype=dtype_np).reshape(
            1, z_dim, 1, 1, 1
        )
        recip_std_arr = 1.0 / std_arr

        latents_mean_t = (
            Tensor.from_dlpack(mean_arr).to(latents.device).cast(latents.dtype)
        )
        latents_recip_std_t = (
            Tensor.from_dlpack(recip_std_arr)
            .to(latents.device)
            .cast(latents.dtype)
        )
        return latents / latents_recip_std_t + latents_mean_t

    @staticmethod
    def _to_numpy(image: Tensor) -> np.ndarray:
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)
