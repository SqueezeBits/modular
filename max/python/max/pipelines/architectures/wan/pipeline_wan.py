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
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import Model
from max.graph import Graph, TensorType, ops
from max.graph.weights import load_weights
from max.interfaces import PixelGenerationContext, TokenBuffer
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    PixelModelInputs,
    max_compile,
)
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import Tracer, traced
from tqdm.auto import tqdm

from ..autoencoders import AutoencoderKLWanModel
from ..autoencoders.autoencoder_kl_wan import (
    _buffer_to_numpy_f32,
    _numpy_f32_to_buffer,
)
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
    images: np.ndarray | Buffer


@dataclass
class WanRuntimeCache:
    """Runtime cache for reusable Wan buffers and helper tensors."""

    spatial_shapes: dict[str, Buffer] = field(default_factory=dict)
    batched_timesteps: dict[str, list[Buffer]] = field(default_factory=dict)
    guidance_scales: dict[tuple[float, DType, str], Buffer] = field(
        default_factory=dict
    )


GuidanceFn = Callable[[Buffer, Buffer, Buffer], Buffer]
TensorUniPCStepFn = Callable[
    [Buffer, Buffer, Buffer, Buffer, Buffer, Buffer],
    tuple[Buffer, Buffer, Buffer],
]
DuplicateBatchFn = Callable[[Buffer], Buffer]
ConcatPromptEmbeddingsFn = Callable[[Buffer, Buffer], Buffer]
SplitCfgPredictionsFn = Callable[[Buffer], tuple[Buffer, Buffer]]


@dataclass
class WanCompiled:
    guidance: GuidanceFn
    tensor_unipc_step: TensorUniPCStepFn
    duplicate_cfg_latents: DuplicateBatchFn
    duplicate_cfg_timesteps: DuplicateBatchFn
    concat_cfg_prompt_embeddings: ConcatPromptEmbeddingsFn
    split_cfg_predictions: SplitCfgPredictionsFn
    cast_f32_to_model_dtype: Model
    cast_model_dtype_to_f32: Model
    denorm: Model


WanUniPCState = tuple[Buffer | None, Buffer | None, Buffer | None]


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
        self,
        weight_paths: list[Path],
    ) -> dict[str, ComponentModel]:
        """Load all sub-models, including optional transformer_2 for MoE."""
        diffusers_config = self.pipeline_config.model.diffusers_config or {}
        components_cfg = diffusers_config.get("components", {})
        relative_paths = self._resolve_relative_component_paths()

        # LoRA configuration from diffusers_config
        lora_config = diffusers_config.get("lora", {})
        lora_repo_id = lora_config.get("repo_id")
        lora_subfolder = lora_config.get("subfolder")
        lora_scale = lora_config.get("scale", 1.0)
        lora_scale_2 = lora_config.get("scale_2", lora_scale)
        lora_filenames = lora_config.get("filenames")

        lora_files: dict[str, Path] = {}
        if lora_repo_id and lora_subfolder:
            from .lora_utils import download_wan_lora

            lora_files = download_wan_lora(
                lora_repo_id, lora_subfolder, lora_filenames
            )

        def _load_component(
            name: str,
            component_cls: type[ComponentModel],
            *,
            session: object | None = None,
            eager_load: bool = True,
            lora_path: Path | None = None,
            lora_scale: float = 1.0,
        ) -> ComponentModel:
            logger.info("Loading Wan component: %s", name)
            config_dict = self._get_component_config_dict(components_cfg, name)
            abs_paths = self._resolve_absolute_paths(
                weight_paths, relative_paths[name]
            )
            component_cls_any = component_cls
            component_kwargs: dict[str, Any] = {
                "config": config_dict,
                "encoding": self.pipeline_config.model.quantization_encoding,
                "devices": self.devices,
                "weights": load_weights(abs_paths),
            }
            if session is not None:
                component_kwargs["session"] = session
            if component_cls is WanTransformerModel:
                component_kwargs["eager_load"] = eager_load
                if lora_path is not None:
                    component_kwargs["lora_path"] = lora_path
                    component_kwargs["lora_scale"] = lora_scale
            if component_cls is AutoencoderKLWanModel:
                component_kwargs["eager_load"] = eager_load
            with Tracer(f"load_component:{name}"):
                component = component_cls_any(
                    **component_kwargs,
                )
            logger.info("Loaded Wan component: %s", name)
            return component

        models: dict[str, ComponentModel] = {}
        for name, component_cls in self.components.items():
            if not issubclass(component_cls, ComponentModel):
                continue
            component_session: object | None = None
            component_eager_load = True
            component_lora_path: Path | None = None
            if component_cls is UMT5Model:
                component_session = self.session
            elif component_cls is WanTransformerModel:
                component_session = self.session
                component_lora_path = lora_files.get("high_noise_model")
            elif component_cls is AutoencoderKLWanModel:
                component_eager_load = False
            models[name] = _load_component(
                name,
                component_cls,
                session=component_session,
                eager_load=component_eager_load,
                lora_path=component_lora_path,
                lora_scale=lora_scale,
            )

        # Optionally load transformer_2 (low-noise expert) for MoE models.
        if "transformer_2" in relative_paths:
            models["transformer_2"] = _load_component(
                "transformer_2",
                WanTransformerModel,
                session=self.session,
                eager_load=False,
                lora_path=lora_files.get("low_noise_model"),
                lora_scale=lora_scale_2,
            )
        else:
            self.transformer_2 = None

        return models

    def init_remaining_components(self) -> None:
        """Initialize runtime helpers, caches, and warmup coordination."""
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

        # Pre-compute VAE denormalization constants on GPU (avoids
        # CPU->GPU transfer every call).
        device = self.transformer.devices[0]
        z_dim = int(self.vae.config.z_dim)
        mean_arr = np.asarray(
            self.vae.config.latents_mean,
            dtype=np.float32,
        ).reshape(1, z_dim, 1, 1, 1)
        std_arr = np.asarray(
            self.vae.config.latents_std,
            dtype=np.float32,
        ).reshape(1, z_dim, 1, 1, 1)
        self._vae_mean_buf = Buffer.from_numpy(mean_arr).to(device)
        self._vae_std_buf = Buffer.from_numpy(std_arr).to(device)

        # MoE dual-load: if VRAM is sufficient, compile transformer_2 as a
        # separate model on GPU so high→low switching is instant (no weight
        # swap).  Otherwise fall back to weight-swap mode.
        self._moe_dual_loaded = False
        if self.transformer_2 is not None:
            self.transformer_2.prepare_state_dict()
            if self._try_dual_load_transformer2(device):
                self._moe_dual_loaded = True
                logger.info(
                    "MoE dual-load: transformer_2 compiled on GPU "
                    "(no weight swap needed)"
                )
            else:
                # Pre-build weight registries for swap mode
                sd2 = self.transformer_2.prepare_state_dict()
                self.transformer._get_cached_weight_registries(sd2)
                logger.info(
                    "MoE swap mode: transformer_2 will use weight swap"
                )

        self._compile_runtime_helpers()
        self.vae.prepare_for_serving()
        self.cache: WanRuntimeCache = WanRuntimeCache()
        self._active_transformer_weights = "primary"

    def _compile_runtime_helpers(self) -> None:
        """Compile the reusable helper graphs used by Wan runtime."""
        self.compiled = WanCompiled(
            guidance=self._compile_guidance_model(),
            tensor_unipc_step=self._compile_tensor_unipc_step_model(),
            duplicate_cfg_latents=self._compile_duplicate_cfg_latents(),
            duplicate_cfg_timesteps=self._compile_duplicate_cfg_timesteps(),
            concat_cfg_prompt_embeddings=self._compile_concat_cfg_prompt_embeddings(),
            split_cfg_predictions=self._compile_split_cfg_predictions(),
            cast_f32_to_model_dtype=self._compile_cast_f32_to_model_dtype(),
            cast_model_dtype_to_f32=self._compile_cast_model_dtype_to_f32(),
            denorm=self._compile_denorm_model(),
        )

    def _compile_guidance_model(self) -> Any:
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

        return max_compile(self._guidance_model, input_types=input_types)

    def _compile_tensor_unipc_step_model(self) -> Any:
        """Compile a single on-device UniPC update step for Wan."""
        device = self.transformer.devices[0]
        model_dtype = self.transformer.config.dtype
        latent_type_f32 = TensorType(
            DType.float32,
            shape=["batch", "channels", "frames", "height", "width"],
            device=device,
        )
        latent_type_model = TensorType(
            model_dtype,
            shape=["batch", "channels", "frames", "height", "width"],
            device=device,
        )
        coeff_type = TensorType(DType.float32, shape=[9], device=device)
        input_types = [
            latent_type_f32,  # sample (f32)
            latent_type_model,  # model_output (model dtype, e.g. bf16)
            latent_type_f32,  # last_sample
            latent_type_f32,  # prev_model_output
            latent_type_f32,  # older_model_output
            coeff_type,
        ]
        return max_compile(
            self._tensor_unipc_step_model,
            input_types=input_types,
        )

    def _compile_duplicate_cfg_latents(self) -> Any:
        """Compile CFG latent duplication helper."""
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype

        return max_compile(
            self._duplicate_batch,
            input_types=[
                TensorType(
                    dtype,
                    shape=[
                        1,
                        self.transformer.config.in_channels,
                        "frames",
                        "height",
                        "width",
                    ],
                    device=device,
                )
            ],
        )

    def _compile_duplicate_cfg_timesteps(self) -> Any:
        device = self.transformer.devices[0]
        return max_compile(
            self._duplicate_batch,
            input_types=[TensorType(DType.float32, shape=[1], device=device)],
        )

    def _compile_concat_cfg_prompt_embeddings(self) -> Any:
        device = self.transformer.devices[0]
        return max_compile(
            self._concat_batch_pair,
            input_types=[
                TensorType(
                    self.text_encoder.config.dtype,
                    shape=[1, "seq_text", self.transformer.config.text_dim],
                    device=device,
                ),
                TensorType(
                    self.text_encoder.config.dtype,
                    shape=[1, "seq_text", self.transformer.config.text_dim],
                    device=device,
                ),
            ],
        )

    def _compile_split_cfg_predictions(self) -> Any:
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype
        return max_compile(
            self._split_cfg_predictions,
            input_types=[
                TensorType(
                    dtype,
                    shape=[
                        2,
                        self.transformer.config.out_channels,
                        "frames",
                        "height",
                        "width",
                    ],
                    device=device,
                )
            ],
        )

    def _compile_cast_f32_to_model_dtype(self) -> Any:
        """Compile float32 -> model dtype cast graph."""
        device = self.transformer.devices[0]
        model_dtype = self.transformer.config.dtype
        latent_5d = ["batch", "channels", "frames", "height", "width"]

        with Graph(
            "wan_cast_f32_to_mdtype",
            input_types=[TensorType(DType.float32, latent_5d, device=device)],
        ) as g:
            g.output(ops.cast(g.inputs[0].tensor, model_dtype))
        return self.session.load(g)

    def _compile_cast_model_dtype_to_f32(self) -> Any:
        """Compile model dtype -> float32 cast graph."""
        device = self.transformer.devices[0]
        model_dtype = self.transformer.config.dtype
        latent_5d = ["batch", "channels", "frames", "height", "width"]

        with Graph(
            "wan_cast_mdtype_to_f32",
            input_types=[TensorType(model_dtype, latent_5d, device=device)],
        ) as g:
            g.output(ops.cast(g.inputs[0].tensor, DType.float32))
        return self.session.load(g)

    def _compile_denorm_model(self) -> Any:
        """Compile VAE latent denormalization + dtype cast graph."""
        device = self.transformer.devices[0]
        model_dtype = self.transformer.config.dtype
        z_dim = int(self.vae.config.z_dim)
        input_types = [
            TensorType(
                DType.float32,
                ["batch", z_dim, "f", "h", "w"],
                device=device,
            ),
            TensorType(DType.float32, [1, z_dim, 1, 1, 1], device=device),
            TensorType(DType.float32, [1, z_dim, 1, 1, 1], device=device),
        ]
        with Graph("wan_denorm", input_types=input_types) as g:
            latents, std, mean = (v.tensor for v in g.inputs)
            result = ops.cast(latents * std + mean, model_dtype)
            g.output(result)
        return self.session.load(g)

    @staticmethod
    def _duplicate_batch(value: Any) -> Any:
        return ops.concat([value, value], axis=0)

    @staticmethod
    def _concat_batch_pair(first_value: Any, second_value: Any) -> Any:
        return ops.concat([first_value, second_value], axis=0)

    @staticmethod
    def _split_cfg_predictions(
        batched_predictions: Any,
    ) -> tuple[Any, Any]:
        positive_prediction = ops.slice_tensor(
            batched_predictions,
            [
                slice(0, 1),
                slice(None),
                slice(None),
                slice(None),
                slice(None),
            ],
        )
        negative_prediction = ops.slice_tensor(
            batched_predictions,
            [
                slice(1, 2),
                slice(None),
                slice(None),
                slice(None),
                slice(None),
            ],
        )
        return positive_prediction, negative_prediction

    def _guidance_model(
        self, noise_pred: Any, noise_uncond: Any, scale: Any
    ) -> Any:
        return noise_uncond + scale * (noise_pred - noise_uncond)

    def _get_guidance_scale(
        self,
        value: float,
        *,
        dtype: DType,
        device: Device,
    ) -> Buffer:
        key = (float(value), dtype, str(device.id))
        cached = self.cache.guidance_scales.get(key)
        if cached is not None:
            return cached
        if dtype == DType.bfloat16:
            u16 = float32_to_bfloat16_as_uint16(
                np.array([float(value)], dtype=np.float32)
            )
            scale = (
                Buffer.from_numpy(u16)
                .to(device)
                .view(dtype=DType.bfloat16, shape=[1])
            )
        else:
            scale = Buffer.from_numpy(
                np.array([float(value)], dtype=np.float32)
            ).to(device)
        self.cache.guidance_scales[key] = scale
        return scale

    def _tensor_unipc_step_model(
        self,
        sample: Any,
        model_output: Any,
        last_sample: Any,
        prev_model_output: Any,
        older_model_output: Any,
        coeffs: Any,
    ) -> tuple[Any, Any, Any]:
        # Cast model_output from model dtype (bf16) to float32
        model_output = ops.cast(model_output, DType.float32)

        sigma = coeffs[0:1]
        corrected_input_scale = coeffs[1:2]
        corrector_sample_scale = coeffs[2:3]
        corrector_m0_scale = coeffs[3:4]
        corrector_m1_scale = coeffs[4:5]
        corrector_mt_scale = coeffs[5:6]
        predictor_sample_scale = coeffs[6:7]
        predictor_m0_scale = coeffs[7:8]
        predictor_m1_scale = coeffs[8:9]

        converted = sample - sigma * model_output
        corrected_sample = (
            corrected_input_scale * sample
            + corrector_sample_scale * last_sample
            + corrector_m0_scale * prev_model_output
            + corrector_m1_scale * older_model_output
            + corrector_mt_scale * converted
        )
        previous_sample = (
            predictor_sample_scale * corrected_sample
            + predictor_m0_scale * converted
            + predictor_m1_scale * prev_model_output
        )
        return previous_sample, converted, corrected_sample

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

    def _prepare_prompt_state(
        self,
        model_inputs: WanModelInputs,
    ) -> tuple[Buffer, Buffer | None, Buffer | None, bool]:
        logger.info("Preparing Wan prompt embeddings")
        max_sequence_length = int(model_inputs.tokens.array.shape[-1])
        prompt_embeds = self._get_t5_prompt_embeds(
            tokens=model_inputs.tokens,
            attention_mask=model_inputs.mask,
            num_videos_per_prompt=model_inputs.num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        do_cfg = (
            model_inputs.guidance_scale > 1.0
            and model_inputs.negative_tokens is not None
        )
        negative_prompt_embeds: Buffer | None = None
        if do_cfg and model_inputs.negative_tokens is not None:
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                tokens=model_inputs.negative_tokens,
                attention_mask=model_inputs.negative_mask,
                num_videos_per_prompt=model_inputs.num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        negative_prompt_embeds_buf = negative_prompt_embeds
        batched_prompt_embeds_buf = None
        if (
            do_cfg
            and negative_prompt_embeds_buf is not None
            and int(prompt_embeds.shape[0]) == 1
            and int(negative_prompt_embeds_buf.shape[0]) == 1
        ):
            batched_prompt_embeds_buf = (
                self.compiled.concat_cfg_prompt_embeddings(
                    prompt_embeds,
                    negative_prompt_embeds_buf,
                )
            )
        return (
            prompt_embeds,
            negative_prompt_embeds_buf,
            batched_prompt_embeds_buf,
            do_cfg,
        )

    def _prepare_latents(
        self, model_inputs: WanModelInputs, device: Device
    ) -> Buffer:
        logger.info("Preparing Wan latents")
        return Buffer.from_numpy(
            np.ascontiguousarray(model_inputs.latents, dtype=np.float32)
        ).to(device)

    def _prepare_i2v_condition(
        self,
        model_inputs: WanModelInputs,
        latent_shape: tuple[int, ...],
        device: Device,
    ) -> Buffer | None:
        """Prepare I2V condition tensor [B, 20, T_l, H_l, W_l].

        Encodes the input image via VAE, builds a temporal mask, and
        concatenates them. Returns None for T2V mode.
        """
        if model_inputs.input_image is None:
            return None
        if self.transformer.config.in_channels <= 16:
            return None

        logger.info("Preparing I2V condition")
        image = model_inputs.input_image  # PIL Image or numpy [H, W, 3]
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Normalize to [-1, 1] float32, shape [1, 3, H, W]
        image_f32 = image.astype(np.float32) / 127.5 - 1.0
        if image_f32.ndim == 3:
            image_f32 = image_f32.transpose(2, 0, 1)[np.newaxis]  # [1,3,H,W]

        batch_size = int(latent_shape[0])
        num_frames = int(model_inputs.num_frames)
        h = image_f32.shape[2]
        w = image_f32.shape[3]

        # Build video condition: first frame = image, rest = zeros
        # shape: [B, 3, T, H, W]
        video_cond = np.zeros(
            (batch_size, 3, num_frames, h, w), dtype=np.float32
        )
        video_cond[:, :, 0, :, :] = image_f32

        # Encode via VAE
        video_buf = _numpy_f32_to_buffer(video_cond, self.vae.config.dtype, device)
        latent_cond = self.vae.encode(video_buf)  # [B, z_dim, T_l, H_l, W_l]

        # Normalize: (encoded - mean) * (1/std)
        latent_cond_np = _buffer_to_numpy_f32(latent_cond)
        z_dim = self.vae.config.z_dim
        mean = np.array(self.vae.config.latents_mean, dtype=np.float32).reshape(
            1, z_dim, 1, 1, 1
        )
        inv_std = 1.0 / np.array(
            self.vae.config.latents_std, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        latent_cond_np = (latent_cond_np - mean) * inv_std

        # Build mask [B, vae_scale_factor_temporal, T_l, H_l, W_l]
        # First frame = 1 (conditioned), rest = 0
        t_latent = int(latent_cond.shape[2])
        h_latent = int(latent_cond.shape[3])
        w_latent = int(latent_cond.shape[4])

        mask = np.zeros(
            (batch_size, 1, num_frames, h_latent, w_latent),
            dtype=np.float32,
        )
        mask[:, :, 0, :, :] = 1.0  # First frame is conditioned

        # Expand first frame mask by vae_scale_factor_temporal
        vae_t = self.vae_scale_factor_temporal
        first_mask = np.repeat(mask[:, :, 0:1, :, :], vae_t, axis=2)
        mask_expanded = np.concatenate(
            [first_mask, mask[:, :, 1:, :, :]], axis=2
        )
        # Reshape: [B, 1, T, H_l, W_l] -> [B, vae_t, T_l, H_l, W_l]
        mask_expanded = mask_expanded.reshape(
            batch_size, -1, vae_t, h_latent, w_latent
        )
        mask_expanded = mask_expanded.transpose(0, 2, 1, 3, 4)
        # Now mask_expanded: [B, vae_t, T_l, H_l, W_l]

        # Concat: [mask, latent_condition] -> [B, vae_t + z_dim, T_l, H_l, W_l]
        condition = np.concatenate(
            [mask_expanded, latent_cond_np], axis=1
        ).astype(np.float32)

        return _numpy_f32_to_buffer(condition, self.vae.config.dtype, device)

    def _concat_i2v_condition(
        self, latent_model_input: Buffer, condition: Buffer
    ) -> Buffer:
        """Concat I2V condition [B, 20, T, H, W] with latents [B, 16, T, H, W].

        Returns [B, 36, T, H, W] for the I2V transformer.
        """
        # Both are in model dtype (bfloat16). Concat on CPU via numpy,
        # then transfer back.
        cpu = CPU()
        lat_np = _buffer_to_numpy_f32(latent_model_input, cpu)
        cond_np = _buffer_to_numpy_f32(condition, cpu)
        concat_np = np.concatenate([lat_np, cond_np], axis=1)
        device = latent_model_input.device
        if hasattr(device, "to_device"):
            device = device.to_device()
        return _numpy_f32_to_buffer(
            concat_np, latent_model_input.dtype, device
        )

    def _prepare_guidance_scales(
        self,
        model_inputs: WanModelInputs,
        prompt_embeds: Buffer,
        do_cfg: bool,
        device: Device,
    ) -> tuple[Buffer | None, Buffer | None]:
        if not do_cfg:
            return None, None
        guidance_scale_high = self._get_guidance_scale(
            float(model_inputs.guidance_scale),
            dtype=prompt_embeds.dtype,
            device=device,
        )
        guidance_scale_low = self._get_guidance_scale(
            float(
                model_inputs.guidance_scale_2
                if model_inputs.guidance_scale_2 is not None
                else model_inputs.guidance_scale
            ),
            dtype=prompt_embeds.dtype,
            device=device,
        )
        return guidance_scale_high, guidance_scale_low

    def _prepare_scheduler_state(
        self,
        latents: Buffer,
        model_inputs: WanModelInputs,
        prompt_embeds: Buffer,
        do_cfg: bool,
        device: Device,
    ) -> tuple[
        Buffer,
        Buffer,
        list[Buffer],
        list[Buffer],
        int,
        Buffer,
        bool,
        Buffer | None,
        Buffer | None,
    ]:
        if model_inputs.step_coefficients is None:
            raise ValueError(
                "WanPipeline requires precomputed step_coefficients from PixelGenerationTokenizer."
            )
        scheduler_timesteps = np.ascontiguousarray(
            model_inputs.timesteps, dtype=np.float32
        )
        boundary_timestep = model_inputs.boundary_timestep
        if boundary_timestep is None and self.boundary_ratio is not None:
            boundary_timestep = self.boundary_ratio * self.num_train_timesteps

        rope_cos, rope_sin = self.transformer.compute_rope(
            num_frames=int(latents.shape[2]),
            height=int(latents.shape[3]),
            width=int(latents.shape[4]),
        )
        batched_timesteps = self._get_batched_timesteps(
            scheduler_timesteps=scheduler_timesteps,
            batch_size=int(latents.shape[0]),
            device=device,
        )
        coeff_buffers = [
            Buffer.from_numpy(np.ascontiguousarray(row, dtype=np.float32)).to(
                device
            )
            for row in model_inputs.step_coefficients
        ]
        guidance_scale_high, guidance_scale_low = self._prepare_guidance_scales(
            model_inputs,
            prompt_embeds,
            do_cfg,
            device,
        )
        has_moe = (
            self.transformer_2 is not None and boundary_timestep is not None
        )
        boundary_step_idx = len(scheduler_timesteps)
        if boundary_timestep is not None:
            for idx, timestep in enumerate(scheduler_timesteps):
                if float(timestep) < boundary_timestep:
                    boundary_step_idx = idx
                    break
        p_t, p_h, p_w = self.transformer.config.patch_size
        spatial_shape = self._get_spatial_shape(
            int(latents.shape[2]) // p_t,
            int(latents.shape[3]) // p_h,
            int(latents.shape[4]) // p_w,
            device,
        )
        return (
            rope_cos,
            rope_sin,
            batched_timesteps,
            coeff_buffers,
            boundary_step_idx,
            spatial_shape,
            has_moe,
            guidance_scale_high,
            guidance_scale_low,
        )

    def _run_denoising(
        self,
        latents: Buffer,
        prompt_embeds: Buffer,
        negative_prompt_embeds: Buffer | None,
        batched_prompt_embeds: Buffer | None,
        do_cfg: bool,
        rope_cos: Buffer,
        rope_sin: Buffer,
        batched_timesteps: list[Buffer],
        coeff_buffers: list[Buffer],
        boundary_step_idx: int,
        spatial_shape: Buffer,
        has_moe: bool,
        guidance_scale_high: Buffer | None,
        guidance_scale_low: Buffer | None,
        i2v_condition: Buffer | None = None,
    ) -> Buffer:
        step_state: WanUniPCState = (None, None, None)
        latents, step_state = self._run_denoising_phase(
            latents=latents,
            transformer_model=self.transformer,
            prompt_embeds=prompt_embeds,
            batched_prompt_embeds=batched_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            batched_timesteps=batched_timesteps,
            coeff_buffers=coeff_buffers,
            do_cfg=do_cfg,
            guidance_scale=guidance_scale_high,
            step_range=range(boundary_step_idx),
            desc="Denoising (high-noise)" if has_moe else "Denoising",
            spatial_shape=spatial_shape,
            step_state=step_state,
            i2v_condition=i2v_condition,
        )

        if has_moe and boundary_step_idx < len(batched_timesteps):
            if self._moe_dual_loaded:
                low_noise_model = self.transformer_2
            else:
                self._activate_transformer_weights(use_secondary=True)
                low_noise_model = self.transformer
            latents, _ = self._run_denoising_phase(
                latents=latents,
                transformer_model=low_noise_model,
                prompt_embeds=prompt_embeds,
                batched_prompt_embeds=batched_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                batched_timesteps=batched_timesteps,
                coeff_buffers=coeff_buffers,
                do_cfg=do_cfg,
                guidance_scale=guidance_scale_low,
                step_range=range(boundary_step_idx, len(batched_timesteps)),
                desc="Denoising (low-noise)",
                spatial_shape=spatial_shape,
                step_state=step_state,
                i2v_condition=i2v_condition,
            )

        return latents

    def _denoise_step(
        self,
        latents: Buffer,
        noise_pred: Buffer,
        coeffs: Buffer,
        step_state: WanUniPCState,
    ) -> tuple[Buffer, WanUniPCState]:
        last_sample, prev_model_output, older_model_output = step_state
        if last_sample is None:
            shape = tuple(int(d) for d in latents.shape)
            zero = Buffer.from_numpy(np.zeros(shape, dtype=np.float32)).to(
                latents.device.to_device()
                if hasattr(latents.device, "to_device")
                else latents.device
            )
            last_sample = zero
            prev_model_output = zero
            older_model_output = zero

        assert prev_model_output is not None
        assert older_model_output is not None
        assert last_sample is not None
        previous_sample, converted, corrected_sample = (
            self.compiled.tensor_unipc_step(
                latents,
                noise_pred,
                last_sample,
                prev_model_output,
                older_model_output,
                coeffs,
            )
        )
        next_prev_model_output = getattr(converted, "driver_tensor", converted)
        next_last_sample = getattr(
            corrected_sample, "driver_tensor", corrected_sample
        )
        next_latents = getattr(
            previous_sample, "driver_tensor", previous_sample
        )
        return next_latents, (
            next_last_sample,
            next_prev_model_output,
            prev_model_output,
        )

    def _decode_output(
        self,
        latents: Buffer,
        model_inputs: WanModelInputs,
    ) -> np.ndarray:
        logger.info("Decoding Wan output")
        # _denormalize_vae_latents does f32→model_dtype cast internally
        denorm_latents = self._denormalize_vae_latents(latents)
        decoded_video = self.vae.decode_5d(denorm_latents)
        decoded_np = self._buffer_to_numpy_f32(
            decoded_video, dtype=decoded_video.dtype
        )
        decoded_num_frames = decoded_np.shape[2]
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
        return decoded_np[
            :,
            :,
            :target_num_frames,
            : model_inputs.height,
            : model_inputs.width,
        ]

    @traced(message="WanPipeline.execute")
    def execute(  # type: ignore[override]
        self,
        model_inputs: WanModelInputs,
        **kwargs: object,
    ) -> WanPipelineOutput:
        del kwargs
        device = self.transformer.devices[0]
        if not self._moe_dual_loaded:
            self._activate_transformer_weights(use_secondary=False)
        with Tracer("prepare_prompt_embeddings"):
            (
                prompt_embeds,
                negative_prompt_embeds,
                batched_prompt_embeds,
                do_cfg,
            ) = self._prepare_prompt_state(model_inputs)
        with Tracer("preprocess_latents"):
            latents = self._prepare_latents(model_inputs, device)
        # Prepare I2V condition if input image provided
        with Tracer("prepare_i2v_condition"):
            i2v_condition = self._prepare_i2v_condition(
                model_inputs, tuple(int(d) for d in latents.shape), device
            )
        # Pre-compile VAE decoder for this latent shape so decode doesn't
        # stall after denoising completes.
        self.vae.prewarm_for_latent_shape(tuple(int(d) for d in latents.shape))
        with Tracer("prepare_scheduler"):
            (
                rope_cos,
                rope_sin,
                batched_timesteps,
                coeff_buffers,
                boundary_step_idx,
                spatial_shape,
                has_moe,
                guidance_scale_high,
                guidance_scale_low,
            ) = self._prepare_scheduler_state(
                latents,
                model_inputs,
                prompt_embeds,
                do_cfg,
                device,
            )
        with Tracer("denoising_loop"):
            latents = self._run_denoising(
                latents,
                prompt_embeds,
                negative_prompt_embeds,
                batched_prompt_embeds,
                do_cfg,
                rope_cos,
                rope_sin,
                batched_timesteps,
                coeff_buffers,
                boundary_step_idx,
                spatial_shape,
                has_moe,
                guidance_scale_high,
                guidance_scale_low,
                i2v_condition=i2v_condition,
            )
        with Tracer("decode_outputs"):
            images = self._decode_output(latents, model_inputs)
        return WanPipelineOutput(images=images)

    def _run_denoising_phase(
        self,
        latents: Buffer,
        transformer_model: Any,
        prompt_embeds: Buffer,
        batched_prompt_embeds: Buffer | None,
        negative_prompt_embeds: Buffer | None,
        rope_cos: Buffer,
        rope_sin: Buffer,
        batched_timesteps: list[Buffer],
        coeff_buffers: list[Buffer],
        do_cfg: bool,
        guidance_scale: Buffer | None,
        step_range: range,
        desc: str,
        spatial_shape: Buffer,
        step_state: WanUniPCState,
        i2v_condition: Buffer | None = None,
    ) -> tuple[Buffer, WanUniPCState]:
        progress = tqdm(  # type: ignore[call-arg]
            step_range,
            desc=desc,
            leave=True,
            disable=not sys.stderr.isatty(),
        )
        for i in progress:  # type: ignore[attr-defined]
            with Tracer(f"{desc}:step_{i}"):
                dit_timestep = batched_timesteps[i]
                latent_model_input = (
                    self.compiled.cast_f32_to_model_dtype.execute(latents)[0]
                )
                # I2V: concat condition with latents along channel dim
                if i2v_condition is not None:
                    latent_model_input = self._concat_i2v_condition(
                        latent_model_input, i2v_condition
                    )
                with Tracer("transformer"):
                    noise_pred_buf = self._run_transformer_forward(
                        transformer_model=transformer_model,
                        latent_model_input=latent_model_input,
                        dit_timestep=dit_timestep,
                        prompt_embeds=prompt_embeds,
                        batched_prompt_embeds=batched_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        rope_cos=rope_cos,
                        rope_sin=rope_sin,
                        spatial_shape=spatial_shape,
                        do_cfg=do_cfg,
                        guidance_scale=guidance_scale,
                    )
                with Tracer("scheduler_step"):
                    latents, step_state = self._denoise_step(
                        latents,
                        noise_pred_buf,
                        coeff_buffers[i],
                        step_state,
                    )
        return latents, step_state

    def _run_transformer_forward(
        self,
        *,
        transformer_model: Any,
        latent_model_input: Buffer,
        dit_timestep: Buffer,
        prompt_embeds: Buffer,
        batched_prompt_embeds: Buffer | None,
        negative_prompt_embeds: Buffer | None,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        do_cfg: bool,
        guidance_scale: Buffer | None,
    ) -> Buffer:
        """Run transformer + optional CFG guidance, return noise prediction."""
        if (
            do_cfg
            and batched_prompt_embeds is not None
            and negative_prompt_embeds is not None
        ):
            duplicated_latents = self.compiled.duplicate_cfg_latents(
                latent_model_input
            )
            duplicated_timesteps = self.compiled.duplicate_cfg_timesteps(
                dit_timestep
            )
            batched_predictions = transformer_model(
                duplicated_latents,
                duplicated_timesteps,
                batched_prompt_embeds,
                rope_cos,
                rope_sin,
                spatial_shape,
            )
            batched_predictions = getattr(
                batched_predictions, "driver_tensor", batched_predictions
            )
            positive, negative = self.compiled.split_cfg_predictions(
                batched_predictions
            )
            assert guidance_scale is not None
            guided = self.compiled.guidance(positive, negative, guidance_scale)
            return getattr(guided, "driver_tensor", guided)

        noise_pred_buf = transformer_model(
            latent_model_input,
            dit_timestep,
            prompt_embeds,
            rope_cos,
            rope_sin,
            spatial_shape,
        )
        noise_pred_buf = getattr(
            noise_pred_buf, "driver_tensor", noise_pred_buf
        )

        if (
            do_cfg
            and batched_prompt_embeds is None
            and negative_prompt_embeds is not None
        ):
            assert guidance_scale is not None
            noise_uncond_buf = transformer_model(
                latent_model_input,
                dit_timestep,
                negative_prompt_embeds,
                rope_cos,
                rope_sin,
                spatial_shape,
            )
            noise_uncond_buf = getattr(
                noise_uncond_buf, "driver_tensor", noise_uncond_buf
            )
            guided = self.compiled.guidance(
                noise_pred_buf,
                noise_uncond_buf,
                guidance_scale,
            )
            return getattr(guided, "driver_tensor", guided)

        return noise_pred_buf

    def _get_spatial_shape(
        self, ppf: int, pph: int, ppw: int, device: Device
    ) -> Buffer:
        key = f"{ppf}_{pph}_{ppw}_{device.id}"
        cached = self.cache.spatial_shapes.get(key)
        if cached is not None:
            return cached
        spatial_np = np.zeros((ppf, pph, ppw), dtype=np.int8)
        spatial_shape = Buffer.from_numpy(spatial_np).to(device)
        self.cache.spatial_shapes[key] = spatial_shape
        return spatial_shape

    def _get_batched_timesteps(
        self,
        scheduler_timesteps: np.ndarray,
        batch_size: int,
        device: Device,
    ) -> list[Buffer]:
        key = (
            f"{batch_size}_{len(scheduler_timesteps)}_"
            f"{float(scheduler_timesteps[0]):.4f}_{float(scheduler_timesteps[-1]):.4f}_"
            f"{device.id}"
        )
        cached = self.cache.batched_timesteps.get(key)
        if cached is not None:
            return cached

        batched_timesteps = [
            Buffer.from_numpy(
                np.full([batch_size], float(step_value), dtype=np.float32)
            ).to(device)
            for step_value in scheduler_timesteps
        ]
        self.cache.batched_timesteps[key] = batched_timesteps
        return batched_timesteps

    def _get_t5_prompt_embeds(
        self,
        tokens: TokenBuffer,
        attention_mask: npt.NDArray[np.bool_] | None,
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ) -> Buffer:
        token_ids = tokens.array
        if token_ids.ndim == 1:
            token_ids = np.expand_dims(token_ids, axis=0)

        if attention_mask is None:
            # Derive mask from token_ids: non-zero tokens are real.
            attention_mask = token_ids != 0
        if attention_mask.ndim == 1:
            attention_mask = np.expand_dims(attention_mask, axis=0)

        device = self.text_encoder.devices[0]
        text_input_ids = Buffer.from_dlpack(
            np.ascontiguousarray(token_ids, dtype=np.int64)
        ).to(device)
        text_attention_mask = Buffer.from_dlpack(
            np.ascontiguousarray(attention_mask.astype(np.int64, copy=False))
        ).to(device)
        raw = self.text_encoder(text_input_ids, text_attention_mask)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        hidden_states = getattr(raw, "driver_tensor", raw)
        return self.get_t5_prompt_embeds_from_hidden(
            hidden_states=hidden_states,
            attention_mask=text_attention_mask,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    @staticmethod
    def _buffer_to_numpy_f32(
        value: Buffer,
        *,
        dtype: DType,
    ) -> np.ndarray:
        cpu_value = value.to(CPU())
        if dtype == DType.bfloat16:
            cpu_u16 = np.from_dlpack(
                cpu_value.view(dtype=DType.uint16, shape=cpu_value.shape)
            )
            return (cpu_u16.astype(np.uint32) << 16).view(np.float32)
        return np.from_dlpack(cpu_value).astype(np.float32, copy=False)

    @staticmethod
    def get_t5_prompt_embeds_from_hidden(
        hidden_states: Buffer,
        attention_mask: Buffer,
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ) -> Buffer:
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
        hidden_states_np = WanPipeline._buffer_to_numpy_f32(
            hidden_states,
            dtype=hidden_states.dtype,
        ).reshape(batch_size, int(hidden_states.shape[1]), hidden_dim)
        attention_mask_np = np.from_dlpack(attention_mask.to(CPU())).reshape(
            batch_size, int(attention_mask.shape[1])
        )

        prompt_embeds_np = np.zeros(
            (batch_size, max_sequence_length, hidden_dim),
            dtype=np.float32,
        )
        for batch_idx in range(batch_size):
            seq_len = int(attention_mask_np[batch_idx].sum())
            seq_len = max(0, min(seq_len, hidden_states_np.shape[1]))
            effective_seq_len = min(seq_len, max_sequence_length)
            prompt_embeds_np[batch_idx, :effective_seq_len, :] = (
                hidden_states_np[batch_idx, :effective_seq_len, :]
            )

        if num_videos_per_prompt > 1:
            prompt_embeds_np = np.repeat(
                prompt_embeds_np,
                num_videos_per_prompt,
                axis=0,
            )

        device = (
            hidden_states.device.to_device()
            if hasattr(hidden_states.device, "to_device")
            else hidden_states.device
        )
        if hidden_states.dtype == DType.bfloat16:
            result_u16 = float32_to_bfloat16_as_uint16(
                np.ascontiguousarray(prompt_embeds_np)
            )
            result_buf = Buffer.from_numpy(result_u16).to(device)
            return result_buf.view(
                dtype=DType.bfloat16,
                shape=prompt_embeds_np.shape,
            )

        return Buffer.from_numpy(np.ascontiguousarray(prompt_embeds_np)).to(
            device
        )

    def _denormalize_vae_latents(self, latents: Buffer) -> Buffer:
        """Denormalize latents using compiled denorm model (f32 in, model_dtype out)."""
        result = self.compiled.denorm.execute(
            latents, self._vae_std_buf, self._vae_mean_buf
        )
        return result[0]

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
    def compute_video_latent_shape(
        *,
        batch_size: int,
        z_dim: int,
        num_frames: int,
        height: int,
        width: int,
        scale_factor_temporal: int,
        scale_factor_spatial: int,
    ) -> tuple[int, int, int, int, int]:
        adjusted_num_frames = max(1, int(num_frames))
        if adjusted_num_frames > 1:
            remainder = (adjusted_num_frames - 1) % scale_factor_temporal
            if remainder != 0:
                adjusted_num_frames += scale_factor_temporal - remainder

        latent_frames = (adjusted_num_frames - 1) // scale_factor_temporal + 1
        latent_height = 2 * (int(height) // (scale_factor_spatial * 2))
        latent_width = 2 * (int(width) // (scale_factor_spatial * 2))
        return (
            int(batch_size),
            int(z_dim),
            int(latent_frames),
            int(latent_height),
            int(latent_width),
        )

    @staticmethod
    def _get_free_vram_bytes() -> int | None:
        """Query free GPU VRAM in bytes via nvidia-smi."""
        import subprocess

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            # First GPU, value in MiB
            return int(out.strip().split("\n")[0]) * 1024 * 1024
        except Exception:
            return None

    def _try_dual_load_transformer2(self, device: Device) -> bool:
        """Try to compile transformer_2 on GPU if VRAM is sufficient.

        Estimates required VRAM as the primary transformer's weight size
        (both models have identical architecture). Returns True if
        transformer_2 was successfully compiled on GPU.
        """
        if self.transformer_2 is None:
            return False

        free_vram = self._get_free_vram_bytes()
        if free_vram is None:
            return False

        # Estimate: use primary transformer's state dict size as proxy.
        # Weights are bfloat16 (2 bytes per element). We use the DLPack
        # shape/dtype info without converting to numpy (bf16 unsupported).
        primary_sd = self.transformer.prepare_state_dict()
        estimated_bytes = 0
        for v in primary_sd.values():
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                num_elements = 1
                for d in v.shape:
                    num_elements *= d
                # bfloat16 = 2 bytes
                estimated_bytes += num_elements * 2
        # Add 20% headroom for compilation workspace
        required = int(estimated_bytes * 1.2)
        if free_vram < required:
            logger.info(
                "Insufficient VRAM for dual-load: %.1f GB free, "
                "%.1f GB required",
                free_vram / 1e9,
                required / 1e9,
            )
            return False

        self.transformer_2.load_model()
        return True

    def _activate_transformer_weights(self, *, use_secondary: bool) -> None:
        if not use_secondary:
            if self._active_transformer_weights != "primary":
                self.transformer.reload_model_weights()
                self._active_transformer_weights = "primary"
            return

        if self.transformer_2 is None:
            return
        if self._active_transformer_weights != "secondary":
            self.transformer.reload_model_weights(
                self.transformer_2.prepare_state_dict()
            )
            self._active_transformer_weights = "secondary"

    @staticmethod
    def denormalize_vae_latents(
        latents_np: np.ndarray,
        latents_mean: list[float],
        latents_std: list[float],
        z_dim: int,
    ) -> np.ndarray:
        """Denormalize VAE latents in numpy (used by external callers)."""
        mean = np.asarray(latents_mean, dtype=np.float32).reshape(
            1, z_dim, 1, 1, 1
        )
        std = np.asarray(latents_std, dtype=np.float32).reshape(
            1, z_dim, 1, 1, 1
        )
        return latents_np * std + mean

    @staticmethod
    def _to_numpy(image: Buffer) -> np.ndarray:
        return WanPipeline._buffer_to_numpy_f32(image, dtype=image.dtype)
