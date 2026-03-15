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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from concurrent.futures import Future
from time import perf_counter

import numpy as np
import numpy.typing as npt
from max._mlir_context import MLIRThreadPoolExecutor
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.graph.weights import load_weights
from max.interfaces import PixelGenerationContext, TokenBuffer
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.diffusion_schedulers import UniPCMultistepScheduler
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    PixelModelInputs,
    max_compile,
)
from max.pipelines.lib.interfaces.component_model import ComponentModel

from ..autoencoders import AutoencoderKLWanModel
from ..umt5 import UMT5Model
from .model import WanTransformerModel

logger = logging.getLogger(__name__)


def _as_buffer(value: object) -> Buffer:
    """Extract the underlying Buffer from a max_compile result.

    ``max_compile`` wraps graph outputs in ``max.experimental.tensor.Tensor``.
    This helper transparently unwraps to the raw ``Buffer`` so that callers
    never need to import the v3 Tensor type.
    """
    dt = getattr(value, "driver_tensor", None)
    return dt if dt is not None else value  # type: ignore[return-value]


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


@dataclass
class WanPromptState:
    prompt_embeds_buf: Buffer
    negative_prompt_embeds_buf: Buffer | None
    batched_prompt_embeds_buf: Buffer | None
    do_cfg: bool
    transformer_dtype: DType


@dataclass
class WanSchedulerState:
    rope_cos: Buffer
    rope_sin: Buffer
    scheduler_timesteps: np.ndarray
    batched_timesteps: list[Buffer]
    tensor_unipc: _WanTensorUniPCScheduler | None
    boundary_timestep: float | None
    boundary_step_idx: int
    spatial_shape: Buffer
    has_moe: bool
    guidance_scale_high: Buffer | None
    guidance_scale_low: Buffer | None


@dataclass(frozen=True)
class _WanUniPCStepCoefficients:
    sigma: float
    corrected_input_scale: float
    predictor_order: int
    predictor_sample_scale: float
    predictor_m0_scale: float
    predictor_m1_scale: float = 0.0
    corrector_order: int = 0
    corrector_sample_scale: float = 0.0
    corrector_m0_scale: float = 0.0
    corrector_m1_scale: float = 0.0
    corrector_mt_scale: float = 0.0


class _WanTensorUniPCScheduler:
    """Tensor/GPU UniPC path for Wan's fixed flow-matching configuration.

    This preserves the existing UniPC schedule/state progression while keeping
    latent updates on-device, avoiding the per-step GPU->CPU->GPU roundtrip in
    the Wan denoising loop.
    """

    def __init__(
        self,
        scheduler: UniPCMultistepScheduler,
        device: Device,
        compiled_step: Callable[..., Any],
    ) -> None:
        if not self.supports(scheduler):
            raise ValueError("Unsupported UniPC scheduler configuration.")
        self._device = device
        self._compiled_step = compiled_step
        self._step_coefficients = self._build_step_coefficients(scheduler)
        self._prev_model_output: Buffer | None = None
        self._older_model_output: Buffer | None = None
        self._last_sample: Buffer | None = None
        self._step_index = 0
        self._coeff_buffer_cache: dict[int, Buffer] = {}

    @staticmethod
    def supports(scheduler: UniPCMultistepScheduler) -> bool:
        return (
            scheduler.use_flow_sigmas
            and scheduler.predict_x0
            and scheduler.prediction_type == "flow_prediction"
            and scheduler.solver_type == "bh2"
            and scheduler.solver_order in (1, 2)
            and not scheduler.thresholding
        )

    @staticmethod
    def _lambda_from_sigma(
        scheduler: UniPCMultistepScheduler,
        sigma: float,
    ) -> float:
        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma)
        with np.errstate(divide="ignore"):
            lambda_t = np.log(alpha_t) - np.log(sigma_t)
        return float(lambda_t)

    @classmethod
    def _predictor_coefficients(
        cls,
        scheduler: UniPCMultistepScheduler,
        step_index: int,
        order: int,
    ) -> tuple[float, float, float]:
        sigma_t_raw = float(scheduler.sigmas[step_index + 1])
        sigma_s0_raw = float(scheduler.sigmas[step_index])
        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t_raw)
        _, sigma_s0 = scheduler._sigma_to_alpha_sigma_t(sigma_s0_raw)

        lambda_t = cls._lambda_from_sigma(scheduler, sigma_t_raw)
        lambda_s0 = cls._lambda_from_sigma(scheduler, sigma_s0_raw)
        h = lambda_t - lambda_s0
        hh = -h
        b_h = float(np.expm1(hh))
        sample_scale = float(sigma_t / sigma_s0)
        m0_scale = float(-alpha_t * b_h)
        m1_scale = 0.0

        if order == 2:
            sigma_si_raw = float(scheduler.sigmas[step_index - 1])
            lambda_si = cls._lambda_from_sigma(scheduler, sigma_si_raw)
            rk = (lambda_si - lambda_s0) / h
            m1_scale = float(-alpha_t * b_h * 0.5 / rk)
            m0_scale -= m1_scale

        return sample_scale, m0_scale, m1_scale

    @classmethod
    def _corrector_coefficients(
        cls,
        scheduler: UniPCMultistepScheduler,
        step_index: int,
        order: int,
    ) -> tuple[float, float, float, float]:
        sigma_t_raw = float(scheduler.sigmas[step_index])
        sigma_s0_raw = float(scheduler.sigmas[step_index - 1])
        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t_raw)
        _, sigma_s0 = scheduler._sigma_to_alpha_sigma_t(sigma_s0_raw)

        lambda_t = cls._lambda_from_sigma(scheduler, sigma_t_raw)
        lambda_s0 = cls._lambda_from_sigma(scheduler, sigma_s0_raw)
        h = lambda_t - lambda_s0
        hh = -h
        h_phi_1 = float(np.expm1(hh))
        b_h = float(np.expm1(hh))
        sample_scale = float(sigma_t / sigma_s0)

        if order == 1:
            shared = float(-alpha_t * b_h * 0.5)
            return sample_scale, shared, 0.0, shared

        sigma_si_raw = float(scheduler.sigmas[step_index - 2])
        lambda_si = cls._lambda_from_sigma(scheduler, sigma_si_raw)
        rk = (lambda_si - lambda_s0) / h

        rks = np.array([rk, 1.0], dtype=np.float64)
        h_phi_k = h_phi_1 / hh - 1.0
        factorial_i = 1
        r_matrix = []
        b_vector = []
        for i in range(1, order + 1):
            r_matrix.append(np.power(rks, i - 1))
            b_vector.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i
        rhos_c = np.linalg.solve(
            np.stack(r_matrix), np.array(b_vector, dtype=np.float64)
        )

        m1_scale = float(-alpha_t * b_h * rhos_c[0] / rk)
        m0_scale = float(
            -alpha_t * h_phi_1 + alpha_t * b_h * (rhos_c[0] / rk + rhos_c[-1])
        )
        mt_scale = float(-alpha_t * b_h * rhos_c[-1])
        return sample_scale, m0_scale, m1_scale, mt_scale

    @classmethod
    def _build_step_coefficients(
        cls,
        scheduler: UniPCMultistepScheduler,
    ) -> list[_WanUniPCStepCoefficients]:
        if scheduler.sigmas is None or scheduler.timesteps is None:
            raise ValueError(
                "Scheduler must be initialized with set_timesteps()."
            )

        num_steps = len(scheduler.timesteps)
        lower_order_nums = 0
        previous_predictor_order = 1
        step_coefficients: list[_WanUniPCStepCoefficients] = []

        for step_index in range(num_steps):
            corrector_order = previous_predictor_order if step_index > 0 else 0
            corrected_input_scale = 1.0 if corrector_order == 0 else 0.0
            if scheduler.lower_order_final:
                candidate_order = min(
                    scheduler.solver_order,
                    num_steps - step_index,
                )
            else:
                candidate_order = scheduler.solver_order
            predictor_order = min(candidate_order, lower_order_nums + 1)

            predictor_sample_scale, predictor_m0_scale, predictor_m1_scale = (
                cls._predictor_coefficients(
                    scheduler, step_index, predictor_order
                )
            )

            if corrector_order > 0:
                (
                    corrector_sample_scale,
                    corrector_m0_scale,
                    corrector_m1_scale,
                    corrector_mt_scale,
                ) = cls._corrector_coefficients(
                    scheduler, step_index, corrector_order
                )
            else:
                (
                    corrector_sample_scale,
                    corrector_m0_scale,
                    corrector_m1_scale,
                    corrector_mt_scale,
                ) = (0.0, 0.0, 0.0, 0.0)

            step_coefficients.append(
                _WanUniPCStepCoefficients(
                    sigma=float(scheduler.sigmas[step_index]),
                    corrected_input_scale=corrected_input_scale,
                    predictor_order=predictor_order,
                    predictor_sample_scale=predictor_sample_scale,
                    predictor_m0_scale=predictor_m0_scale,
                    predictor_m1_scale=predictor_m1_scale,
                    corrector_order=corrector_order,
                    corrector_sample_scale=corrector_sample_scale,
                    corrector_m0_scale=corrector_m0_scale,
                    corrector_m1_scale=corrector_m1_scale,
                    corrector_mt_scale=corrector_mt_scale,
                )
            )

            previous_predictor_order = predictor_order
            if lower_order_nums < scheduler.solver_order:
                lower_order_nums += 1

        return step_coefficients

    def _coeff_buffer(self, step_index: int) -> Buffer:
        cached = self._coeff_buffer_cache.get(step_index)
        if cached is not None:
            return cached
        coeffs = self._step_coefficients[step_index]
        buf = Buffer.from_numpy(
            np.asarray(
                [
                    coeffs.sigma,
                    coeffs.corrected_input_scale,
                    coeffs.corrector_sample_scale,
                    coeffs.corrector_m0_scale,
                    coeffs.corrector_m1_scale,
                    coeffs.corrector_mt_scale,
                    coeffs.predictor_sample_scale,
                    coeffs.predictor_m0_scale,
                    coeffs.predictor_m1_scale,
                ],
                dtype=np.float32,
            )
        ).to(self._device)
        self._coeff_buffer_cache[step_index] = buf
        return buf

    def step(self, model_output: Buffer, sample: Buffer) -> Buffer:
        """Advance one UniPC step.

        Args:
            model_output: Noise prediction in model dtype (bf16).
            sample: Current latents in float32.

        Returns:
            Updated latents in float32.
        """
        if self._last_sample is None:
            shape = tuple(int(d) for d in sample.shape)
            zero = Buffer.from_numpy(np.zeros(shape, dtype=np.float32)).to(
                self._device
            )
            self._last_sample = zero
            self._prev_model_output = zero
            self._older_model_output = zero

        assert self._prev_model_output is not None
        assert self._older_model_output is not None
        assert self._last_sample is not None
        previous_sample, converted, corrected_sample = self._compiled_step(
            sample,
            model_output,
            self._last_sample,
            self._prev_model_output,
            self._older_model_output,
            self._coeff_buffer(self._step_index),
        )
        self._older_model_output = self._prev_model_output
        self._prev_model_output = _as_buffer(converted)
        self._last_sample = _as_buffer(corrected_sample)
        self._step_index += 1
        return _as_buffer(previous_sample)


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

        def _load_component(
            name: str,
            component_cls: type[ComponentModel],
            *,
            session: object | None = None,
            eager_load: bool = True,
        ) -> ComponentModel:
            start = perf_counter()
            logger.info("Loading Wan component: %s", name)
            config_dict = self._get_component_config_dict(components_cfg, name)
            abs_paths = self._resolve_absolute_paths(
                weight_paths, relative_paths[name]
            )
            component_cls_any = component_cls
            component_kwargs = {
                "config": config_dict,
                "encoding": self.pipeline_config.model.quantization_encoding,
                "devices": self.devices,
                "weights": load_weights(abs_paths),
            }
            if session is not None:
                component_kwargs["session"] = session
            if component_cls is WanTransformerModel:
                component_kwargs["eager_load"] = eager_load
            if component_cls is AutoencoderKLWanModel:
                component_kwargs["eager_load"] = eager_load
            component = component_cls_any(
                **component_kwargs,
            )
            logger.info(
                "Loaded Wan component %s in %.2fs", name, perf_counter() - start
            )
            return component

        models: dict[str, ComponentModel] = {}
        for name, component_cls in self.components.items():
            if not issubclass(component_cls, ComponentModel):
                continue
            component_session: object | None = None
            component_eager_load = True
            if component_cls is UMT5Model:
                component_session = self.session
            elif component_cls is WanTransformerModel:
                component_session = self.session
            elif component_cls is AutoencoderKLWanModel:
                component_eager_load = False
            models[name] = _load_component(
                name,
                component_cls,
                session=component_session,
                eager_load=component_eager_load,
            )

        # Optionally load transformer_2 (low-noise expert) for MoE models.
        if "transformer_2" in relative_paths:
            models["transformer_2"] = _load_component(
                "transformer_2",
                WanTransformerModel,
                session=self.session,
                eager_load=False,
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

        # Initialize UniPC multistep scheduler for proper 2nd-order stepping.
        self._scheduler = UniPCMultistepScheduler(**scheduler_cfg)
        self._base_flow_shift = float(scheduler_cfg.get("flow_shift", 1.0))

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

        self._compile_runtime_helpers()

        # Warmup transformer GPU kernels with a tiny forward pass.
        self.transformer.warmup()

        self.vae.prepare_for_serving()
        self.cache: WanRuntimeCache = WanRuntimeCache()
        self._warmup_executor = MLIRThreadPoolExecutor(max_workers=2)
        self._low_noise_future: Future[object] | None = None
        self._vae_prewarm_future: Future[object] | None = None
        self._vae_prewarm_shape: tuple[int, int, int, int, int] | None = None
        self._ready_vae_prewarm_shapes: set[tuple[int, int, int, int, int]] = (
            set()
        )
        self._active_transformer_weights = "primary"
        self._maybe_startup_vae_prewarm()

    def _compile_runtime_helpers(self) -> None:
        """Compile the reusable helper graphs used by Wan runtime."""
        self.build_guidance_model()
        self.build_tensor_unipc_step_model()
        self._compile_cfg_fastpath_helpers()
        self._compile_cast_helpers()
        self._compile_denorm_model()

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

    def build_tensor_unipc_step_model(self) -> None:
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
        self.__dict__["_tensor_unipc_step_model"] = max_compile(
            self._tensor_unipc_step_model,
            input_types=input_types,
        )

    def _compile_cfg_fastpath_helpers(self) -> None:
        """Compile helper graphs for Wan CFG batch fast path."""

        def duplicate_batch(value: Any) -> Any:
            return ops.concat([value, value], axis=0)

        def concat_batch_pair(first_value: Any, second_value: Any) -> Any:
            return ops.concat([first_value, second_value], axis=0)

        def split_cfg_predictions(
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

        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype

        self.cached_duplicate_cfg_latents = max_compile(
            duplicate_batch,
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
        self.cached_duplicate_cfg_timesteps = max_compile(
            duplicate_batch,
            input_types=[TensorType(DType.float32, shape=[1], device=device)],
        )
        self.cached_concat_cfg_prompt_embeddings = max_compile(
            concat_batch_pair,
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
        self.cached_split_cfg_predictions = max_compile(
            split_cfg_predictions,
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

    def _compile_cast_helpers(self) -> None:
        """Compile dtype cast graphs (Buffer in → Buffer out)."""
        device = self.transformer.devices[0]
        model_dtype = self.transformer.config.dtype
        latent_5d = ["batch", "channels", "frames", "height", "width"]

        with Graph(
            "wan_cast_f32_to_mdtype",
            input_types=[TensorType(DType.float32, latent_5d, device=device)],
        ) as g:
            g.output(ops.cast(g.inputs[0].tensor, model_dtype))
        self._cast_f32_to_model_dtype = self.session.load(g)

        with Graph(
            "wan_cast_mdtype_to_f32",
            input_types=[TensorType(model_dtype, latent_5d, device=device)],
        ) as g:
            g.output(ops.cast(g.inputs[0].tensor, DType.float32))
        self._cast_model_dtype_to_f32 = self.session.load(g)

    def _compile_denorm_model(self) -> None:
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
        self._denorm_model = self.session.load(g)

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
        if model_inputs.latents.ndim == 5:
            effective_boundary_ratio = (
                model_inputs.boundary_ratio
                if model_inputs.boundary_ratio is not None
                else self.boundary_ratio
            )
            self._start_background_warmups(
                latents_shape=tuple(
                    int(dim) for dim in model_inputs.latents.shape
                ),
                has_moe=(
                    self.transformer_2 is not None
                    and effective_boundary_ratio is not None
                ),
            )
        return model_inputs

    def _prepare_prompt_state(
        self,
        model_inputs: WanModelInputs,
    ) -> WanPromptState:
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
                self.cached_concat_cfg_prompt_embeddings(
                    prompt_embeds,
                    negative_prompt_embeds_buf,
                )
            )
        return WanPromptState(
            prompt_embeds_buf=prompt_embeds,
            negative_prompt_embeds_buf=negative_prompt_embeds_buf,
            batched_prompt_embeds_buf=batched_prompt_embeds_buf,
            do_cfg=do_cfg,
            transformer_dtype=prompt_embeds.dtype,
        )

    def _prepare_latents(
        self, model_inputs: WanModelInputs, device: Device
    ) -> Buffer:
        logger.info("Preparing Wan latents")
        return Buffer.from_numpy(
            np.ascontiguousarray(model_inputs.latents, dtype=np.float32)
        ).to(device)

    def _compute_boundary_step_idx(
        self,
        scheduler_timesteps: np.ndarray,
        boundary_timestep: float | None,
    ) -> int:
        if boundary_timestep is None:
            return len(scheduler_timesteps)
        for idx, timestep in enumerate(scheduler_timesteps):
            if self.use_low_noise_transformer(
                float(timestep), boundary_timestep
            ):
                return idx
        return len(scheduler_timesteps)

    def _prepare_guidance_scales(
        self,
        model_inputs: WanModelInputs,
        prompt_state: WanPromptState,
        device: Device,
    ) -> tuple[Buffer | None, Buffer | None]:
        if not prompt_state.do_cfg:
            return None, None
        guidance_scale_high = self._get_guidance_scale(
            float(model_inputs.guidance_scale),
            dtype=prompt_state.transformer_dtype,
            device=device,
        )
        guidance_scale_low = self._get_guidance_scale(
            float(
                model_inputs.guidance_scale_2
                if model_inputs.guidance_scale_2 is not None
                else model_inputs.guidance_scale
            ),
            dtype=prompt_state.transformer_dtype,
            device=device,
        )
        return guidance_scale_high, guidance_scale_low

    def _prepare_scheduler_state(
        self,
        latents: Buffer,
        model_inputs: WanModelInputs,
        prompt_state: WanPromptState,
        device: Device,
    ) -> WanSchedulerState:
        logger.info("Preparing Wan scheduler/runtime state")
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

        tensor_unipc = None
        if _WanTensorUniPCScheduler.supports(self._scheduler):
            tensor_unipc = _WanTensorUniPCScheduler(
                self._scheduler,
                device,
                self._tensor_unipc_step_model,
            )

        batched_timesteps = self._get_batched_timesteps(
            scheduler_timesteps=scheduler_timesteps,
            batch_size=int(latents.shape[0]),
            device=device,
        )
        guidance_scale_high, guidance_scale_low = self._prepare_guidance_scales(
            model_inputs,
            prompt_state,
            device,
        )
        has_moe = (
            self.transformer_2 is not None and boundary_timestep is not None
        )
        boundary_step_idx = self._compute_boundary_step_idx(
            scheduler_timesteps,
            boundary_timestep,
        )
        p_t, p_h, p_w = self.transformer.config.patch_size
        spatial_shape = self._get_spatial_shape(
            int(latents.shape[2]) // p_t,
            int(latents.shape[3]) // p_h,
            int(latents.shape[4]) // p_w,
            device,
        )
        return WanSchedulerState(
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            scheduler_timesteps=scheduler_timesteps,
            batched_timesteps=batched_timesteps,
            tensor_unipc=tensor_unipc,
            boundary_timestep=boundary_timestep,
            boundary_step_idx=boundary_step_idx,
            spatial_shape=spatial_shape,
            has_moe=has_moe,
            guidance_scale_high=guidance_scale_high,
            guidance_scale_low=guidance_scale_low,
        )

    def _run_denoising(
        self,
        latents: Buffer,
        prompt_state: WanPromptState,
        scheduler_state: WanSchedulerState,
        device: Device,
    ) -> Buffer:
        latents_np_cache: np.ndarray | None = None
        self._start_background_warmups(
            latents_shape=tuple(int(dim) for dim in latents.shape),
            has_moe=scheduler_state.has_moe,
        )
        latents, latents_np_cache = self._run_denoising_phase(
            latents=latents,
            transformer_model=self.transformer,
            prompt_embeds=prompt_state.prompt_embeds_buf,
            batched_prompt_embeds=prompt_state.batched_prompt_embeds_buf,
            negative_prompt_embeds=prompt_state.negative_prompt_embeds_buf,
            rope_cos=scheduler_state.rope_cos,
            rope_sin=scheduler_state.rope_sin,
            scheduler_timesteps=scheduler_state.scheduler_timesteps,
            batched_timesteps=scheduler_state.batched_timesteps,
            do_cfg=prompt_state.do_cfg,
            guidance_scale=scheduler_state.guidance_scale_high,
            device=device,
            step_range=range(scheduler_state.boundary_step_idx),
            desc="Denoising (high-noise)"
            if scheduler_state.has_moe
            else "Denoising",
            spatial_shape=scheduler_state.spatial_shape,
            latents_np=latents_np_cache,
            tensor_unipc=scheduler_state.tensor_unipc,
        )

        if scheduler_state.has_moe and (
            scheduler_state.boundary_step_idx
            < len(scheduler_state.scheduler_timesteps)
        ):
            assert self.transformer_2 is not None
            self._finish_low_noise_warmup()
            latents, latents_np_cache = self._run_denoising_phase(
                latents=latents,
                transformer_model=self.transformer,
                prompt_embeds=prompt_state.prompt_embeds_buf,
                batched_prompt_embeds=prompt_state.batched_prompt_embeds_buf,
                negative_prompt_embeds=prompt_state.negative_prompt_embeds_buf,
                rope_cos=scheduler_state.rope_cos,
                rope_sin=scheduler_state.rope_sin,
                scheduler_timesteps=scheduler_state.scheduler_timesteps,
                batched_timesteps=scheduler_state.batched_timesteps,
                do_cfg=prompt_state.do_cfg,
                guidance_scale=scheduler_state.guidance_scale_low,
                device=device,
                step_range=range(
                    scheduler_state.boundary_step_idx,
                    len(scheduler_state.scheduler_timesteps),
                ),
                desc="Denoising (low-noise)",
                spatial_shape=scheduler_state.spatial_shape,
                latents_np=latents_np_cache,
                tensor_unipc=scheduler_state.tensor_unipc,
            )

        return latents

    def _decode_output(
        self,
        latents: Buffer,
        model_inputs: WanModelInputs,
    ) -> np.ndarray:
        logger.info("Decoding Wan output")
        # _denormalize_vae_latents does f32→model_dtype cast internally
        denorm_latents = self._denormalize_vae_latents(latents)
        self._finish_vae_prewarm(
            tuple(int(dim) for dim in denorm_latents.shape)
        )
        decoded_video = self.vae.decode_5d(denorm_latents)
        decoded_np = self._buffer_to_numpy_f32(
            _as_buffer(decoded_video), dtype=decoded_video.dtype
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

    def execute(  # type: ignore[override]
        self,
        model_inputs: WanModelInputs,
        **kwargs: object,
    ) -> WanPipelineOutput:
        del kwargs
        device = self.transformer.devices[0]
        if (
            self.transformer_2 is not None
            and self._active_transformer_weights != "primary"
        ):
            self.transformer.reload_model_weights()
            self._active_transformer_weights = "primary"
        prompt_state = self._prepare_prompt_state(model_inputs)
        latents = self._prepare_latents(model_inputs, device)
        scheduler_state = self._prepare_scheduler_state(
            latents,
            model_inputs,
            prompt_state,
            device,
        )
        latents = self._run_denoising(
            latents,
            prompt_state,
            scheduler_state,
            device,
        )
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
        scheduler_timesteps: np.ndarray,
        batched_timesteps: list[Buffer],
        do_cfg: bool,
        guidance_scale: Buffer | None,
        device: Device,
        step_range: range,
        desc: str,
        spatial_shape: Buffer,
        latents_np: np.ndarray | None = None,
        tensor_unipc: _WanTensorUniPCScheduler | None = None,
    ) -> tuple[Buffer, np.ndarray | None]:
        """Run a denoising phase using UniPC multistep scheduler.

        Transformer forward passes run on GPU (takes/returns Buffer).
        Guidance model uses compiled graph. Scheduler step runs on CPU
        via numpy or on-device via tensor UniPC.
        """
        sched = self._scheduler
        cpu = CPU()
        log_step_timings = os.getenv("WAN_LOG_STEP_TIMINGS") == "1"
        logger.info("%s start (%d steps)", desc, len(step_range))

        for i in step_range:
            step_start = perf_counter()
            dit_timestep = batched_timesteps[i]

            # Cast latents (f32) to model dtype for transformer
            transformer_start = perf_counter()
            latent_model_input = self._cast_f32_to_model_dtype.execute(latents)[
                0
            ]

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
            transformer_time = perf_counter() - transformer_start

            scheduler_start = perf_counter()
            if tensor_unipc is not None:
                latents = tensor_unipc.step(noise_pred_buf, latents)
            else:
                # Scheduler step on CPU via numpy.
                if latents_np is None:
                    latents_np = np.from_dlpack(latents.to(cpu))
                noise_f32 = self._cast_model_dtype_to_f32.execute(
                    noise_pred_buf
                )[0]
                noise_np = np.from_dlpack(noise_f32.to(cpu))

                latents_np = sched.step(
                    noise_np, int(scheduler_timesteps[i]), latents_np
                )

                # Back to GPU as f32 for next iteration's cast
                latents = Buffer.from_numpy(
                    np.ascontiguousarray(latents_np, dtype=np.float32)
                ).to(device)
            scheduler_time = perf_counter() - scheduler_start

            if log_step_timings:
                logger.info(
                    "%s step=%d transformer=%.3fs scheduler=%.3fs total=%.3fs",
                    desc,
                    i,
                    transformer_time,
                    scheduler_time,
                    perf_counter() - step_start,
                )

        logger.info("%s complete", desc)
        return latents, latents_np

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
            batched_predictions = _as_buffer(
                transformer_model(
                    _as_buffer(
                        self.cached_duplicate_cfg_latents(latent_model_input)
                    ),
                    _as_buffer(
                        self.cached_duplicate_cfg_timesteps(dit_timestep)
                    ),
                    batched_prompt_embeds,
                    rope_cos,
                    rope_sin,
                    spatial_shape,
                )
            )
            positive, negative = self.cached_split_cfg_predictions(
                batched_predictions
            )
            assert guidance_scale is not None
            return _as_buffer(
                self._guidance_model(positive, negative, guidance_scale)
            )

        noise_pred_buf = _as_buffer(
            transformer_model(
                latent_model_input,
                dit_timestep,
                prompt_embeds,
                rope_cos,
                rope_sin,
                spatial_shape,
            )
        )

        if (
            do_cfg
            and batched_prompt_embeds is None
            and negative_prompt_embeds is not None
        ):
            assert guidance_scale is not None
            noise_uncond_buf = _as_buffer(
                transformer_model(
                    latent_model_input,
                    dit_timestep,
                    negative_prompt_embeds,
                    rope_cos,
                    rope_sin,
                    spatial_shape,
                )
            )
            return _as_buffer(
                self._guidance_model(
                    noise_pred_buf,
                    noise_uncond_buf,
                    guidance_scale,
                )
            )

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
            f"{int(scheduler_timesteps[0])}_{int(scheduler_timesteps[-1])}_"
            f"{device.id}"
        )
        cached = self.cache.batched_timesteps.get(key)
        if cached is not None:
            return cached

        batched_timesteps = [
            Buffer.from_numpy(
                np.full([batch_size], float(int(step_value)), dtype=np.float32)
            ).to(device)
            for step_value in scheduler_timesteps
        ]
        self.cache.batched_timesteps[key] = batched_timesteps
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
        hidden_states = _as_buffer(raw)
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
        result = self._denorm_model.execute(
            latents, self._vae_std_buf, self._vae_mean_buf
        )
        return _as_buffer(result[0])

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
    def compute_boundary_timestep(
        boundary_ratio: float | None,
        num_train_timesteps: int,
    ) -> float | None:
        if boundary_ratio is None:
            return None
        return boundary_ratio * num_train_timesteps

    @staticmethod
    def use_low_noise_transformer(
        scheduler_timestep: float,
        boundary_timestep: float | None,
    ) -> bool:
        return (
            boundary_timestep is not None
            and scheduler_timestep < boundary_timestep
        )

    def _start_background_warmups(
        self, latents_shape: tuple[int, int, int, int, int], has_moe: bool
    ) -> None:
        if (
            has_moe
            and self.transformer_2 is not None
            and self._low_noise_future is None
        ):
            logger.info("Starting Wan low-noise expert background warmup")
            self._low_noise_future = self._warmup_executor.submit(
                self.transformer_2.prepare_state_dict
            )

        self._schedule_vae_prewarm(latents_shape)

    def _finish_low_noise_warmup(self) -> None:
        if self.transformer_2 is None:
            return
        if self._low_noise_future is not None:
            logger.info("Waiting for Wan low-noise expert background warmup")
            self._low_noise_future.result()
            self._low_noise_future = None
        self.transformer.reload_model_weights(
            self.transformer_2.prepare_state_dict()
        )
        self._active_transformer_weights = "secondary"

    def _finish_vae_prewarm(
        self, latents_shape: tuple[int, int, int, int, int]
    ) -> None:
        if (
            self._vae_prewarm_future is not None
            and self._vae_prewarm_shape == latents_shape
        ):
            logger.info("Waiting for Wan VAE background prewarm")
            self._vae_prewarm_future.result()
            self._vae_prewarm_future = None
            self._ready_vae_prewarm_shapes.add(latents_shape)
            self._vae_prewarm_shape = None
        if latents_shape not in self._ready_vae_prewarm_shapes:
            self.vae.prewarm_for_latent_shape(latents_shape)
            self._ready_vae_prewarm_shapes.add(latents_shape)

    def _schedule_vae_prewarm(
        self, latents_shape: tuple[int, int, int, int, int]
    ) -> None:
        if latents_shape in self._ready_vae_prewarm_shapes:
            return
        if (
            self._vae_prewarm_future is not None
            and self._vae_prewarm_shape == latents_shape
        ):
            return
        if self._vae_prewarm_future is not None:
            return
        logger.info(
            "Starting Wan VAE background prewarm for latent shape %s",
            latents_shape,
        )
        self._vae_prewarm_shape = latents_shape
        self._vae_prewarm_future = self._warmup_executor.submit(
            self.vae.prewarm_for_latent_shape, latents_shape
        )

    def _maybe_startup_vae_prewarm(self) -> None:
        height_raw = os.getenv("WAN_STARTUP_WARMUP_HEIGHT")
        width_raw = os.getenv("WAN_STARTUP_WARMUP_WIDTH")
        frames_raw = os.getenv("WAN_STARTUP_WARMUP_NUM_FRAMES")
        batch_raw = os.getenv("WAN_STARTUP_WARMUP_BATCH_SIZE")

        if not any((height_raw, width_raw, frames_raw, batch_raw)):
            return
        if not all((height_raw, width_raw, frames_raw)):
            logger.warning(
                "Skipping Wan startup VAE warmup: set all of "
                "WAN_STARTUP_WARMUP_HEIGHT/WIDTH/NUM_FRAMES."
            )
            return

        latents_shape = self.compute_video_latent_shape(
            batch_size=int(batch_raw or "1"),
            z_dim=int(self.vae.config.z_dim),
            num_frames=int(frames_raw),
            height=int(height_raw),
            width=int(width_raw),
            scale_factor_temporal=self.vae_scale_factor_temporal,
            scale_factor_spatial=self.vae_scale_factor_spatial,
        )
        logger.info(
            "Scheduling Wan startup VAE warmup for request %sx%s/%s frames -> latent shape %s",
            int(height_raw),
            int(width_raw),
            int(frames_raw),
            latents_shape,
        )
        self._schedule_vae_prewarm(latents_shape)

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
