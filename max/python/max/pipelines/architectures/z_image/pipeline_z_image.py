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
"""Z-Image diffusion pipeline (Graph API / ModuleV2).

Standalone pipeline that wires together the Qwen3 text encoder, Z-Image
Graph API transformer denoiser, and standard AutoencoderKL VAE.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import TensorType, TensorValue, ops
from max.interfaces import PixelGenerationContext
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    DiffusionPipelineOutput,
    PixelModelInputs,
)
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from max.pipelines.lib.utils import BoundedCache
from max.profiler import Tracer, traced

from ..autoencoders import AutoencoderKLModel
from ..qwen3.text_encoder import Qwen3TextEncoderZImageModel
from .model import ZImageTransformerModel

_DEVICE_TENSOR_FIELDS = frozenset(
    {
        "tokens_tensor",
        "negative_tokens_tensor",
        "txt_ids_tensor",
        "img_ids_tensor",
        "negative_txt_ids_tensor",
        "negative_img_ids_tensor",
        "input_image_tensor",
        "latents_tensor",
        "sigmas_tensor",
        "h_carrier",
        "w_carrier",
    }
)


def _validate_z_image_context(context: PixelGenerationContext) -> None:
    """Fail fast before device uploads."""
    if context.latents.size == 0:
        raise ValueError(
            "ZImagePipeline requires non-empty latents in"
            " PixelGenerationContext."
        )
    for name in ("latent_image_ids", "sigmas", "timesteps"):
        if not hasattr(context, name):
            raise TypeError(
                f"ZImagePipeline requires PixelGenerationContext with"
                f" attribute {name!r}; {type(context).__name__} has no"
                f" {name!r}."
            )
        arr = getattr(context, name)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            raise ValueError(
                f"ZImagePipeline requires non-empty {name} in"
                " PixelGenerationContext."
            )


@dataclass(kw_only=True)
class ZImageModelInputs(PixelModelInputs):
    """Z-Image execution inputs with device tensors and host metadata."""

    width: int = 1024
    height: int = 1024
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    explicit_negative_prompt: bool = False
    do_cfg: bool = False
    tokens_tensor: Buffer
    negative_tokens_tensor: Buffer | None = None
    txt_ids_tensor: Buffer
    img_ids_tensor: Buffer
    negative_txt_ids_tensor: Buffer | None = None
    negative_img_ids_tensor: Buffer | None = None
    input_image_tensor: Buffer | None = None
    latents_tensor: Buffer
    sigmas_tensor: Buffer
    h_carrier: Buffer
    w_carrier: Buffer

    @classmethod
    def kwargs_from_context(
        cls, context: PixelGenerationContext
    ) -> dict[str, Any]:
        """Build kwargs for all fields except device tensors."""
        kwargs: dict[str, Any] = {}
        for dataclass_field in fields(cls):
            name = dataclass_field.name
            if name in _DEVICE_TENSOR_FIELDS:
                continue
            if not hasattr(context, name):
                continue
            v = getattr(context, name)
            if v is None:
                if dataclass_field.default is not MISSING:
                    kwargs[name] = dataclass_field.default
                elif dataclass_field.default_factory is not MISSING:
                    kwargs[name] = dataclass_field.default_factory()
                else:
                    kwargs[name] = None
            else:
                kwargs[name] = v
        return kwargs

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sigmas.size == 0:
            raise ValueError(
                "ZImagePipeline requires non-empty sigmas in context."
            )
        if self.latent_image_ids.size == 0:
            raise ValueError(
                "ZImagePipeline requires non-empty latent image ids in context."
            )


class ZImagePipeline(DiffusionPipeline):
    """Diffusion pipeline for Z-Image generation (Graph API)."""

    unprefixed_weight_component = "transformer"
    default_num_inference_steps = 50
    default_residual_threshold = 0.06

    vae: AutoencoderKLModel
    text_encoder: Qwen3TextEncoderZImageModel
    transformer: ZImageTransformerModel

    components = {
        "vae": AutoencoderKLModel,
        "text_encoder": Qwen3TextEncoderZImageModel,
        "transformer": ZImageTransformerModel,
    }

    # -- Initialisation ------------------------------------------------------

    @traced(message="ZImagePipeline.init_remaining_components")
    def init_remaining_components(self) -> None:
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )

        self.build_preprocess_latents()
        self.build_scheduler_step()
        self.build_decode_latents()
        self.build_cfg_combine()
        self.build_cfg_renormalization()
        self.build_postprocess_image()

        self._cached_text_ids: BoundedCache[str, Buffer] = BoundedCache(32)
        self._cached_sigmas: BoundedCache[str, Buffer] = BoundedCache(32)
        self._cached_img_ids: BoundedCache[str, Buffer] = BoundedCache(32)
        self._cached_img_ids_base_np: BoundedCache[str, np.ndarray] = (
            BoundedCache(32)
        )
        self._cached_shape_carriers: BoundedCache[int, Buffer] = BoundedCache(
            32
        )
        self._cached_prompt_token_tensors: BoundedCache[str, Buffer] = (
            BoundedCache(32)
        )
        self._cached_prompt_padding: BoundedCache[str, Buffer] = BoundedCache(
            32
        )
        self._cached_guidance: BoundedCache[str, Buffer] = BoundedCache(32)
        self._cached_step_tensors: BoundedCache[
            str, tuple[list[Buffer], list[Buffer]]
        ] = BoundedCache(32)

    # -- Build compiled graphs -----------------------------------------------

    @traced(message="ZImagePipeline.build_preprocess_latents")
    def build_preprocess_latents(self) -> None:
        device = self.transformer.devices[0]
        target_dtype = self.transformer.config.dtype

        def _graph(latents: TensorValue) -> TensorValue:
            batch = latents.shape[0]
            c = latents.shape[1]
            h = latents.shape[2]
            w = latents.shape[3]
            latents = ops.rebind(
                latents, [batch, c, (h // 2) * 2, (w // 2) * 2]
            )
            latents = ops.reshape(latents, (batch, c, h // 2, 2, w // 2, 2))
            latents = ops.permute(latents, [0, 2, 4, 3, 5, 1])
            latents = ops.reshape(latents, (batch, (h // 2) * (w // 2), c * 4))
            return ops.cast(latents, target_dtype)

        self._patchify_and_pack = cast(
            Callable[[Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        DType.float32,
                        shape=["batch", "channels", "height", "width"],
                        device=device,
                    ),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_scheduler_step")
    def build_scheduler_step(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]

        def _graph(
            latents: TensorValue, noise_pred: TensorValue, dt: TensorValue
        ) -> TensorValue:
            latents_dtype = latents.dtype
            latents = ops.cast(latents, DType.float32)
            latents = latents - dt * noise_pred
            return ops.cast(latents, latents_dtype)

        self._scheduler_step = cast(
            Callable[[Buffer, Buffer, Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                    TensorType(DType.float32, shape=[1], device=device),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_decode_latents")
    def build_decode_latents(self) -> None:
        device = self.transformer.devices[0]
        if hasattr(self.vae, "build_fused_decode"):
            self._fused_decode = self.vae.build_fused_decode(device)
        else:
            dtype = self.transformer.config.dtype
            scaling = float(self.vae.config.scaling_factor)
            shift = float(self.vae.config.shift_factor or 0.0)

            def _unpack(
                latents: TensorValue,
                h_carrier: TensorValue,
                w_carrier: TensorValue,
            ) -> TensorValue:
                batch = latents.shape[0]
                ch = latents.shape[2]
                half_h = h_carrier.shape[0]
                half_w = w_carrier.shape[0]
                latents = ops.reshape(
                    latents, (batch, half_h, half_w, 2, 2, ch // 4)
                )
                latents = ops.permute(latents, [0, 5, 1, 3, 2, 4])
                latents = ops.reshape(
                    latents, (batch, ch // 4, half_h * 2, half_w * 2)
                )
                return (latents / scaling) + shift

            self._unpack_and_postprocess = cast(
                Callable[[Buffer, Buffer, Buffer], Buffer],
                max_compile(
                    _unpack,
                    input_types=[
                        TensorType(
                            dtype,
                            shape=["batch", "seq", "channels"],
                            device=device,
                        ),
                        TensorType(
                            DType.float32, shape=["half_h"], device=device
                        ),
                        TensorType(
                            DType.float32, shape=["half_w"], device=device
                        ),
                    ],
                ),
            )

    @traced(message="ZImagePipeline.build_postprocess_image")
    def build_postprocess_image(self) -> None:
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype

        def _graph(image: TensorValue) -> TensorValue:
            image = ops.cast(image, DType.float32)
            image = image * 0.5 + 0.5
            image = ops.where(image < 0.0, 0.0, image)
            image = ops.where(image > 1.0, 1.0, image)
            image = ops.permute(image, [0, 2, 3, 1])
            image = image * 255.0
            return ops.cast(image, DType.uint8)

        self._postprocess_image = cast(
            Callable[[Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch", "channels", "height", "width"],
                        device=device,
                    ),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_cfg_combine")
    def build_cfg_combine(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]

        def _graph(
            pos: TensorValue, neg: TensorValue, scale: TensorValue
        ) -> TensorValue:
            result = pos + scale * (pos - neg)
            return ops.cast(result, pos.dtype)

        self._cfg_combine = cast(
            Callable[[Buffer, Buffer, Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                    TensorType(DType.float32, shape=[], device=device),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_cfg_renormalization")
    def build_cfg_renormalization(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]

        def _graph(pos: TensorValue, pred: TensorValue) -> TensorValue:
            ori_norm = ops.sqrt(
                ops.sum(ops.sum(pos * pos, axis=2), axis=1) + 1e-12
            )
            new_norm = ops.sqrt(
                ops.sum(ops.sum(pred * pred, axis=2), axis=1) + 1e-12
            )
            while ori_norm.rank > 1:
                ori_norm = ops.squeeze(ori_norm, -1)
            while new_norm.rank > 1:
                new_norm = ops.squeeze(new_norm, -1)
            safe_new = ops.where(new_norm > 1e-12, new_norm, 1e-12)
            ratio = ori_norm / safe_new
            ratio = ops.where(new_norm > ori_norm, ratio, 1.0)
            ratio = ops.unsqueeze(ops.unsqueeze(ratio, 1), 2)
            return pred * ratio

        self._cfg_renormalization = cast(
            Callable[[Buffer, Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                ],
            ),
        )

    # -- Prepare inputs ------------------------------------------------------

    @traced(message="ZImagePipeline.prepare_inputs")
    def prepare_inputs(
        self, context: PixelGenerationContext
    ) -> ZImageModelInputs:
        _validate_z_image_context(context)
        kwargs = ZImageModelInputs.kwargs_from_context(context)
        device = self.transformer.devices[0]
        text_device = self.text_encoder.devices[0]

        kwargs["latents"] = np.asarray(context.latents)
        kwargs["sigmas"] = np.asarray(context.sigmas)
        kwargs["latent_image_ids"] = np.asarray(context.latent_image_ids)

        latents_np = np.ascontiguousarray(kwargs["latents"])
        latent_h = int(latents_np.shape[-2])
        latent_w = int(latents_np.shape[-1])
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_seq_len = int(np.asarray(context.latent_image_ids).shape[-2])

        tokens_np = self._select_tokens_for_text_encoder(
            context.tokens.array, context.mask
        )
        tokens_buf = self._cache_token_buffer(tokens_np, text_device)
        txt_ids_buf, img_ids_buf = self._prepare_conditioning_ids(
            text_seq_len=int(tokens_np.shape[0]),
            image_seq_len=image_seq_len,
            latent_image_ids=np.asarray(context.latent_image_ids),
            height=int(context.height),
            width=int(context.width),
            device=device,
        )

        neg_tokens_buf: Buffer | None = None
        neg_txt_ids_buf: Buffer | None = None
        neg_img_ids_buf: Buffer | None = None
        if context.negative_tokens is not None:
            neg_np = self._select_tokens_for_text_encoder(
                context.negative_tokens.array, context.negative_mask
            )
            neg_tokens_buf = self._cache_token_buffer(neg_np, text_device)
            if context.explicit_negative_prompt:
                neg_txt_ids_buf, neg_img_ids_buf = (
                    self._prepare_conditioning_ids(
                        text_seq_len=int(neg_np.shape[0]),
                        image_seq_len=image_seq_len,
                        latent_image_ids=np.asarray(context.latent_image_ids),
                        height=int(context.height),
                        width=int(context.width),
                        device=device,
                    )
                )
        do_cfg = (
            float(context.guidance_scale) > 0.0 and neg_tokens_buf is not None
        )

        input_image_buf: Buffer | None = None
        if context.input_image is not None:
            input_image_buf = self._numpy_image_to_buffer(
                image=np.ascontiguousarray(
                    context.input_image.astype(np.uint8, copy=False)
                ),
                batch_size=int(context.num_images_per_prompt),
                dtype=self.vae.config.dtype,
            )

        latents_buf = Buffer.from_dlpack(latents_np).to(device)

        for n in (packed_h, packed_w):
            if n not in self._cached_shape_carriers:
                self._cached_shape_carriers[n] = Buffer.from_dlpack(
                    np.ascontiguousarray(np.empty(n, dtype=np.float32))
                ).to(device)

        num_steps = int(context.num_inference_steps)
        sigmas_key = f"sigmas::{num_steps}::{latent_h}x{latent_w}"
        if sigmas_key in self._cached_sigmas:
            sigmas_buf = self._cached_sigmas[sigmas_key]
        else:
            sigmas_buf = Buffer.from_dlpack(
                np.ascontiguousarray(context.sigmas)
            ).to(device)
            self._cached_sigmas[sigmas_key] = sigmas_buf

        return ZImageModelInputs(
            **kwargs,
            do_cfg=do_cfg,
            tokens_tensor=tokens_buf,
            negative_tokens_tensor=neg_tokens_buf,
            txt_ids_tensor=txt_ids_buf,
            img_ids_tensor=img_ids_buf,
            negative_txt_ids_tensor=neg_txt_ids_buf,
            negative_img_ids_tensor=neg_img_ids_buf,
            input_image_tensor=input_image_buf,
            latents_tensor=latents_buf,
            sigmas_tensor=sigmas_buf,
            h_carrier=self._cached_shape_carriers[packed_h],
            w_carrier=self._cached_shape_carriers[packed_w],
        )

    # -- Utility methods -----------------------------------------------------

    @staticmethod
    def _select_tokens_for_text_encoder(
        tokens: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray:
        if tokens.ndim == 2:
            tokens = tokens[0]
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[0]
            selected = mask.astype(np.bool_, copy=False)
            if not np.any(selected):
                raise ValueError("ZImage mask cannot exclude all tokens.")
            if not np.all(selected):
                tokens = tokens[selected]
        return np.ascontiguousarray(tokens.astype(np.int64, copy=False))

    def _cache_token_buffer(self, tokens: np.ndarray, device: Device) -> Buffer:
        digest = hashlib.sha1(tokens.tobytes()).hexdigest()
        key = f"tokens::{tokens.shape[0]}::{digest}::{device}"
        if key in self._cached_prompt_token_tensors:
            return self._cached_prompt_token_tensors[key]
        buf = Buffer.from_dlpack(tokens).to(device)
        self._cached_prompt_token_tensors[key] = buf
        return buf

    def _prepare_conditioning_ids(
        self,
        text_seq_len: int,
        image_seq_len: int,
        latent_image_ids: np.ndarray,
        height: int,
        width: int,
        device: Device,
    ) -> tuple[Buffer, Buffer]:
        text_seq_len_padded = text_seq_len + (-text_seq_len % 32)

        img_base_key = f"img_ids_base::{image_seq_len}_{height}x{width}"
        if img_base_key in self._cached_img_ids_base_np:
            img_ids_base = self._cached_img_ids_base_np[img_base_key]
        else:
            img_ids_base = np.asarray(latent_image_ids, dtype=np.int64)
            if img_ids_base.ndim == 3:
                img_ids_base = img_ids_base[0]
            img_ids_base = np.ascontiguousarray(img_ids_base)
            self._cached_img_ids_base_np[img_base_key] = img_ids_base

        img_key = (
            f"img_ids::{text_seq_len_padded}_{image_seq_len}_{height}x{width}"
        )
        if img_key in self._cached_img_ids:
            img_buf = self._cached_img_ids[img_key]
        else:
            img_np = img_ids_base.copy()
            img_np[:, 0] = img_np[:, 0] + text_seq_len_padded + 1
            img_buf = Buffer.from_dlpack(np.ascontiguousarray(img_np)).to(
                device
            )
            self._cached_img_ids[img_key] = img_buf

        txt_key = f"text_ids::{text_seq_len}"
        if txt_key in self._cached_text_ids:
            txt_buf = self._cached_text_ids[txt_key]
        else:
            txt_ids = np.zeros((text_seq_len, 3), dtype=np.int64)
            txt_ids[:, 0] = np.arange(1, text_seq_len + 1, dtype=np.int64)
            txt_buf = Buffer.from_dlpack(np.ascontiguousarray(txt_ids)).to(
                device
            )
            self._cached_text_ids[txt_key] = txt_buf

        return txt_buf, img_buf

    def _numpy_image_to_buffer(
        self, image: np.ndarray, batch_size: int, dtype: DType
    ) -> Buffer:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected input image shape [H, W, 3], got {image.shape}."
            )
        img_f32 = image.astype(np.float32) / 127.5 - 1.0
        img_chw = np.transpose(img_f32, (2, 0, 1))
        img_bchw = np.expand_dims(img_chw, axis=0)
        if batch_size > 1:
            img_bchw = np.tile(img_bchw, (batch_size, 1, 1, 1))
        return Buffer.from_dlpack(np.ascontiguousarray(img_bchw)).to(
            self.vae.devices[0]
        )

    def _get_cached_guidance(
        self, guidance_scale: float, device: Device
    ) -> Buffer:
        key = f"{guidance_scale:.8f}::{device}"
        if key in self._cached_guidance:
            return self._cached_guidance[key]
        buf = Buffer.from_dlpack(np.array(guidance_scale, dtype=np.float32)).to(
            device
        )
        self._cached_guidance[key] = buf
        return buf

    # -- Prompt embeddings ---------------------------------------------------

    @traced(message="ZImagePipeline.prepare_prompt_embeddings")
    def prepare_prompt_embeddings(
        self, tokens: Buffer, num_images_per_prompt: int
    ) -> Buffer:
        """Encode prompt tokens via text encoder.

        Wraps Buffer→Tensor for the V3 text encoder, then extracts the
        underlying Buffer from the result.
        """
        prompt_embeds = self.text_encoder(Tensor(storage=tokens))
        return prompt_embeds.storage

    # -- Decode --------------------------------------------------------------

    @traced(message="ZImagePipeline.decode_latents")
    def decode_latents(
        self,
        latents: Buffer,
        h_carrier: Buffer,
        w_carrier: Buffer,
    ) -> npt.NDArray[np.uint8]:
        if hasattr(self, "_fused_decode"):
            decoded = self._fused_decode(latents, h_carrier, w_carrier)
            if isinstance(decoded, Tensor):
                assert decoded.storage is not None
                decoded = decoded.storage
            return np.asarray(np.from_dlpack(decoded.to(CPU())), dtype=np.uint8)

        latents = self._unpack_and_postprocess(latents, h_carrier, w_carrier)
        decoded = self.vae.decode(Tensor(storage=latents))
        assert decoded.storage is not None
        image = self._postprocess_image(decoded.storage)
        return np.asarray(np.from_dlpack(image.to(CPU())), dtype=np.uint8)

    # -- Preprocess ----------------------------------------------------------

    @traced(message="ZImagePipeline.preprocess_latents")
    def preprocess_latents(self, latents: Buffer) -> Buffer:
        """Patchify and pack latents.

        Expects float32 GPU Buffer from prepare_inputs (latents are
        uploaded as float32 via Buffer.from_dlpack().to(device)).
        The compiled _patchify_and_pack graph handles the final dtype
        cast to model precision internally.
        """
        return self._patchify_and_pack(latents)

    # -- Execute -------------------------------------------------------------

    @traced(message="ZImagePipeline.execute")
    def execute(  # type: ignore[override]
        self, model_inputs: ZImageModelInputs
    ) -> DiffusionPipelineOutput:
        """Run the Z-Image denoising loop and decode outputs."""

        # 1) Encode prompt embeddings.
        with Tracer("prepare_prompt_embeddings"):
            prompt_embeds = self.prepare_prompt_embeddings(
                tokens=model_inputs.tokens_tensor,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )

            negative_prompt_embeds: Buffer | None = None
            if (
                model_inputs.do_cfg
                and model_inputs.negative_tokens_tensor is not None
            ):
                negative_prompt_embeds = self.prepare_prompt_embeddings(
                    tokens=model_inputs.negative_tokens_tensor,
                    num_images_per_prompt=model_inputs.num_images_per_prompt,
                )

        latents = model_inputs.latents_tensor
        sigmas = model_inputs.sigmas_tensor
        h_carrier = model_inputs.h_carrier
        w_carrier = model_inputs.w_carrier

        timesteps: np.ndarray = model_inputs.timesteps
        num_timesteps = timesteps.shape[0]
        if num_timesteps < 1:
            raise ValueError("No timesteps were provided for denoising.")

        # 2) Prepare latents.
        device = self.transformer.devices[0]
        img_ids = model_inputs.img_ids_tensor
        txt_ids = model_inputs.txt_ids_tensor
        latents = self.preprocess_latents(latents)

        # 3) Prepare scheduler tensors.
        with Tracer("prepare_scheduler"):
            transformed_timesteps = np.ascontiguousarray(
                (1.0 - model_inputs.timesteps).astype(np.float32, copy=False)
            )
            sigmas_host = np.asarray(model_inputs.sigmas, dtype=np.float32)
            dt_values = np.ascontiguousarray(
                (sigmas_host[1:] - sigmas_host[:-1]).astype(
                    np.float32, copy=False
                )
            )

            combined = np.concatenate([transformed_timesteps, dt_values])
            step_key = (
                f"steps::{num_timesteps}::"
                f"{hashlib.sha1(combined.tobytes()).hexdigest()}"
            )
            if step_key in self._cached_step_tensors:
                timestep_bufs, dt_bufs = self._cached_step_tensors[step_key]
            else:
                timestep_bufs = [
                    Buffer.from_dlpack(
                        np.array([float(t)], dtype=np.float32)
                    ).to(device)
                    for t in transformed_timesteps
                ]
                dt_bufs = [
                    Buffer.from_dlpack(
                        np.array([float(d)], dtype=np.float32)
                    ).to(device)
                    for d in dt_values
                ]
                self._cached_step_tensors[step_key] = (
                    timestep_bufs,
                    dt_bufs,
                )

        cfg_cutoff_step = 0
        if model_inputs.do_cfg:
            if model_inputs.cfg_truncation > 1.0:
                cfg_cutoff_step = num_timesteps
            else:
                mask = transformed_timesteps <= model_inputs.cfg_truncation
                cfg_cutoff_step = int(np.count_nonzero(mask))

        guidance_buf: Buffer | None = None
        if model_inputs.do_cfg:
            guidance_buf = self._get_cached_guidance(
                model_inputs.guidance_scale, device
            )

        # Prepare negative conditioning IDs if needed.
        neg_img_ids = img_ids
        neg_txt_ids = txt_ids
        if model_inputs.do_cfg and negative_prompt_embeds is not None:
            if model_inputs.explicit_negative_prompt:
                assert model_inputs.negative_img_ids_tensor is not None
                assert model_inputs.negative_txt_ids_tensor is not None
                neg_img_ids = model_inputs.negative_img_ids_tensor
                neg_txt_ids = model_inputs.negative_txt_ids_tensor
            else:
                # Create txt_ids matching negative prompt seq length.
                neg_seq_len = int(negative_prompt_embeds.shape[1])
                neg_txt_key = f"text_ids::{neg_seq_len}"
                if neg_txt_key in self._cached_text_ids:
                    neg_txt_ids = self._cached_text_ids[neg_txt_key]
                else:
                    neg_txt_np = np.zeros((neg_seq_len, 3), dtype=np.int64)
                    neg_txt_np[:, 0] = np.arange(
                        1, neg_seq_len + 1, dtype=np.int64
                    )
                    neg_txt_ids = Buffer.from_dlpack(
                        np.ascontiguousarray(neg_txt_np)
                    ).to(device)
                    self._cached_text_ids[neg_txt_key] = neg_txt_ids

        # 4) Denoising loop.
        with Tracer("denoising_loop"):
            for i in range(num_timesteps):
                apply_cfg = i < cfg_cutoff_step
                timestep = timestep_bufs[i]
                dt = dt_bufs[i]

                with Tracer(f"denoising_step_{i}"):
                    with Tracer("transformer"):
                        noise_pred = self.transformer(
                            latents,
                            prompt_embeds,
                            timestep,
                            img_ids,
                            txt_ids,
                        )[0]

                    # Non-batched CFG: separate negative pass.
                    if apply_cfg:
                        assert negative_prompt_embeds is not None
                        with Tracer("cfg_transformer"):
                            neg_noise_pred = self.transformer(
                                latents,
                                negative_prompt_embeds,
                                timestep,
                                neg_img_ids,
                                neg_txt_ids,
                            )[0]
                        assert guidance_buf is not None
                        noise_pred = self._cfg_combine(
                            noise_pred, neg_noise_pred, guidance_buf
                        )
                        if model_inputs.cfg_normalization:
                            noise_pred = self._cfg_renormalization(
                                noise_pred, noise_pred
                            )

                    with Tracer("scheduler_step"):
                        latents = self._scheduler_step(latents, noise_pred, dt)

        with Tracer("decode_outputs"):
            images = self.decode_latents(latents, h_carrier, w_carrier)

        return DiffusionPipelineOutput(images=images)
