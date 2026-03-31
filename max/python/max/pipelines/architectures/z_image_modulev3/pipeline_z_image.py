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
"""Z-Image diffusion pipeline (ModuleV3).

Wires together the Qwen3 text encoder, Z-Image transformer denoiser, and
standard AutoencoderKL VAE. Follows the shared diffusion-pipeline structure
used by the image generation architectures in this directory, including
tracing, module docstrings, and flat weight path assignment.
"""

from __future__ import annotations

import hashlib
from dataclasses import MISSING, dataclass, fields
from typing import Any, Literal

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.interfaces import PixelGenerationContext
from max.pipelines.lib.interfaces import (
    DenoisingCacheState,
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
            "ZImagePipeline requires non-empty latents in PixelGenerationContext."
        )
    for name in ("latent_image_ids", "sigmas", "timesteps"):
        if not hasattr(context, name):
            raise TypeError(
                f"ZImagePipeline requires PixelGenerationContext with attribute "
                f"{name!r} (e.g. max.pipelines.core.PixelContext); "
                f"{type(context).__name__} has no {name!r}."
            )
        arr = getattr(context, name)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            raise ValueError(
                f"ZImagePipeline requires non-empty {name} in PixelGenerationContext."
            )


@dataclass(kw_only=True)
class ZImageModelInputs(PixelModelInputs):
    """Z-Image execution inputs with device tensors and host metadata.

    Scalar and NumPy fields are populated from :class:`PixelGenerationContext`
    (concrete type :class:`~max.pipelines.core.PixelContext`) via
    :meth:`kwargs_from_context`; ``latents_tensor``, ``sigmas_tensor``, and
    shape carriers are required device tensors supplied by
    :meth:`ZImagePipeline.prepare_inputs` in one shot.
    """

    width: int = 1024
    height: int = 1024
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    explicit_negative_prompt: bool = False
    do_cfg: bool = False
    tokens_tensor: Tensor
    negative_tokens_tensor: Tensor | None = None
    txt_ids_tensor: Tensor
    img_ids_tensor: Tensor
    negative_txt_ids_tensor: Tensor | None = None
    negative_img_ids_tensor: Tensor | None = None
    input_image_tensor: Tensor | None = None
    latents_tensor: Tensor
    sigmas_tensor: Tensor
    h_carrier: Tensor
    w_carrier: Tensor

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
    """Diffusion pipeline for Z-Image generation (Qwen3 + transformer + VAE)."""

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

    @traced(message="ZImagePipeline.init_remaining_components")
    def init_remaining_components(self) -> None:
        """Initialize derived attributes and compiled subgraphs."""
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )

        self.build_preprocess_latents()
        self.build_scheduler_step()
        self.build_blend_img2img_latents()
        self.build_decode_latents()
        self.build_cfg_combine()
        self.build_duplicate_batch()
        self.build_duplicate_2d()
        self.build_cfg_finalize_batched()
        self.build_cfg_renormalization()
        self.build_postprocess_image()

        self._init_cache_state(
            dtype=self.transformer.config.dtype,
            device=self.transformer.devices[0],
        )

        self._cached_text_ids: BoundedCache[str, Tensor] = BoundedCache(32)
        self._cached_sigmas: BoundedCache[str, Tensor] = BoundedCache(32)
        self._cached_img_ids: BoundedCache[str, Tensor] = BoundedCache(32)
        self._cached_img_ids_base_np: BoundedCache[
            str, np.ndarray
        ] = BoundedCache(32)
        self._cached_shape_carriers: BoundedCache[
            int, Tensor
        ] = BoundedCache(32)
        self._cached_prompt_token_tensors: BoundedCache[
            str, Tensor
        ] = BoundedCache(32)
        self._cached_prompt_padding: BoundedCache[
            str, Tensor
        ] = BoundedCache(32)
        self._cached_guidance: BoundedCache[
            str, Tensor
        ] = BoundedCache(32)
        self._cached_step_tensors: BoundedCache[
            str, tuple[list[Tensor], list[Tensor]]
        ] = BoundedCache(32)

    @traced(message="ZImagePipeline.prepare_inputs")
    def prepare_inputs(
        self, context: PixelGenerationContext
    ) -> ZImageModelInputs:
        """Convert a :class:`PixelGenerationContext` into model inputs with device tensors."""
        _validate_z_image_context(context)
        kwargs = ZImageModelInputs.kwargs_from_context(context)
        device = self.transformer.devices[0]
        text_device = self.text_encoder.devices[0]

        # Keep host NumPy fields aligned with the tensors we upload.
        kwargs["latents"] = np.asarray(context.latents)
        kwargs["sigmas"] = np.asarray(context.sigmas)
        kwargs["latent_image_ids"] = np.asarray(context.latent_image_ids)

        latents_np = np.ascontiguousarray(kwargs["latents"])
        latent_h = int(latents_np.shape[-2])
        latent_w = int(latents_np.shape[-1])
        packed_h = int(latent_h // 2)
        packed_w = int(latent_w // 2)
        image_seq_len = int(np.asarray(context.latent_image_ids).shape[-2])

        tokens_np = self._select_tokens_for_text_encoder(
            context.tokens.array, context.mask
        )
        tokens_tensor = self._token_tensor_from_numpy(tokens_np, text_device)
        txt_ids_tensor, img_ids_tensor = self._prepare_conditioning_ids(
            text_seq_len=int(tokens_np.shape[0]),
            image_seq_len=image_seq_len,
            latent_image_ids=np.asarray(context.latent_image_ids),
            height=int(context.height),
            width=int(context.width),
            device=device,
        )

        negative_tokens_tensor: Tensor | None = None
        negative_txt_ids_tensor: Tensor | None = None
        negative_img_ids_tensor: Tensor | None = None
        if context.negative_tokens is not None:
            negative_tokens_np = self._select_tokens_for_text_encoder(
                context.negative_tokens.array,
                context.negative_mask,
            )
            negative_tokens_tensor = self._token_tensor_from_numpy(
                negative_tokens_np, text_device
            )
            if context.explicit_negative_prompt:
                (
                    negative_txt_ids_tensor,
                    negative_img_ids_tensor,
                ) = self._prepare_conditioning_ids(
                    text_seq_len=int(negative_tokens_np.shape[0]),
                    image_seq_len=image_seq_len,
                    latent_image_ids=np.asarray(context.latent_image_ids),
                    height=int(context.height),
                    width=int(context.width),
                    device=device,
                )
        do_cfg = (
            float(context.guidance_scale) > 0.0
            and negative_tokens_tensor is not None
        )

        input_image_tensor: Tensor | None = None
        if context.input_image is not None:
            input_image_tensor = self._numpy_image_to_tensor(
                image=np.ascontiguousarray(
                    context.input_image.astype(np.uint8, copy=False)
                ),
                batch_size=int(context.num_images_per_prompt),
                dtype=self.vae.config.dtype,
            )

        latents_tensor = Tensor(
            storage=Buffer.from_dlpack(latents_np).to(device)
        )

        for n in (packed_h, packed_w):
            if n not in self._cached_shape_carriers:
                carrier = np.ascontiguousarray(np.empty(n, dtype=np.float32))
                self._cached_shape_carriers[n] = Tensor(
                    storage=Buffer.from_dlpack(carrier).to(device)
                )
        h_carrier = self._cached_shape_carriers[packed_h]
        w_carrier = self._cached_shape_carriers[packed_w]

        num_steps = int(context.num_inference_steps)
        sigmas_key = f"sigmas::{num_steps}::{latent_h}x{latent_w}"
        if sigmas_key in self._cached_sigmas:
            sigmas_tensor = self._cached_sigmas[sigmas_key]
        else:
            sigmas_tensor = Tensor(
                storage=Buffer.from_dlpack(
                    np.ascontiguousarray(context.sigmas)
                ).to(device)
            )
            self._cached_sigmas[sigmas_key] = sigmas_tensor

        return ZImageModelInputs(
            **kwargs,
            do_cfg=do_cfg,
            tokens_tensor=tokens_tensor,
            negative_tokens_tensor=negative_tokens_tensor,
            txt_ids_tensor=txt_ids_tensor,
            img_ids_tensor=img_ids_tensor,
            negative_txt_ids_tensor=negative_txt_ids_tensor,
            negative_img_ids_tensor=negative_img_ids_tensor,
            input_image_tensor=input_image_tensor,
            latents_tensor=latents_tensor,
            sigmas_tensor=sigmas_tensor,
            h_carrier=h_carrier,
            w_carrier=w_carrier,
        )

    def create_cache_state(
        self,
        batch_size: int,
        seq_len: int,
        transformer_config: Any,
    ) -> DenoisingCacheState:
        """Allocate FBCache / Taylor tensors using Z-Image output layout."""
        cfg = transformer_config
        residual_dim, output_dim = cfg.fbcache_dims()
        state = DenoisingCacheState()

        def _device_zeros(shape: tuple[int, ...]) -> Tensor:
            return Tensor(
                storage=Buffer.zeros(
                    shape, self._cache_dtype, device=self._cache_device
                )
            )

        if self.cache_config.first_block_caching:
            state.prev_residual = _device_zeros(
                (batch_size, seq_len, residual_dim)
            )
            state.prev_output = _device_zeros((batch_size, seq_len, output_dim))

        return state

    def run_transformer(
        self,
        cache_state: DenoisingCacheState,
        **kwargs: Any,
    ) -> tuple[Tensor, ...]:
        return self.transformer(
            kwargs["latents"],
            kwargs["prompt_embeds"],
            kwargs["timestep"],
            kwargs["img_ids"],
            kwargs["txt_ids"],
            prev_residual=cache_state.prev_residual,
            prev_output=cache_state.prev_output,
        )

    @staticmethod
    def duplicate_2d(x: Tensor) -> Tensor:
        """Duplicate [1, D] → [2, D] via broadcast."""
        d = x.shape[1]
        x = F.broadcast_to(x, [2, d])
        return F.rebind(x, [2, d])

    def run_transformer_cfg_split(
        self,
        latents: Tensor,
        prompt_embeds: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
    ) -> Tensor:
        """Run transformer with split preamble (batch=1) + main (batch=2).

        Avoids redundant noise_refiner / embedder computation on the
        duplicated CFG batch.
        """
        # Phase 1: shared preamble at batch=1.
        x, t_emb, unified_freqs, txt_freqs = (
            self.transformer.run_preamble(
                latents, timestep, img_ids, txt_ids,
            )
        )

        # Duplicate shared results to batch=2.
        x_dup = self.duplicate_batch(x)
        t_emb_dup = self.duplicate_2d(t_emb)

        # Phase 2: main at batch=2.
        return self.transformer.run_cfg_main(
            x_dup, prompt_embeds, t_emb_dup, unified_freqs, txt_freqs,
        )[0]

    @traced(message="ZImagePipeline.build_preprocess_latents")
    def build_preprocess_latents(self) -> None:
        device = self.transformer.devices[0]
        target_dtype = self.transformer.config.dtype

        def _patchify_pack_and_cast(latents: Tensor) -> Tensor:
            return ZImagePipeline._patchify_and_pack(latents).cast(
                target_dtype
            )

        self.__dict__["_patchify_and_pack"] = max_compile(
            _patchify_pack_and_cast,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=[
                        "batch",
                        "channels",
                        "height",
                        "width",
                    ],
                    device=device,
                ),
            ],
        )

    @traced(message="ZImagePipeline.build_scheduler_step")
    def build_scheduler_step(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["scheduler_step"] = max_compile(
            self.scheduler_step,
            input_types=[
                TensorType(
                    dtype, shape=["batch", "seq", "channels"], device=device
                ),
                TensorType(
                    dtype, shape=["batch", "seq", "channels"], device=device
                ),
                TensorType(DType.float32, shape=[1], device=device),
            ],
        )

    @traced(message="ZImagePipeline.build_decode_latents")
    def build_decode_latents(self) -> None:
        device = self.transformer.devices[0]
        if hasattr(self.vae, "build_fused_decode"):
            self._fused_decode = self.vae.build_fused_decode(device)
        else:
            dtype = self.transformer.config.dtype
            self.__dict__["_unpack_and_postprocess"] = max_compile(
                self._unpack_and_postprocess,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch", "seq", "channels"],
                        device=device,
                    ),
                    TensorType(
                        DType.float32,
                        shape=["half_h"],
                        device=device,
                    ),
                    TensorType(
                        DType.float32,
                        shape=["half_w"],
                        device=device,
                    ),
                ],
            )

    @traced(message="ZImagePipeline.build_postprocess_image")
    def build_postprocess_image(self) -> None:
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype
        self.__dict__["_postprocess_image"] = max_compile(
            self._postprocess_image,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "channels", "height", "width"],
                    device=device,
                ),
            ],
        )

    @traced(message="ZImagePipeline.build_blend_img2img_latents")
    def build_blend_img2img_latents(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["_blend_img2img_latents"] = max_compile(
            self._blend_img2img_latents,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "channels", "height", "width"],
                    device=device,
                ),
                TensorType(
                    dtype,
                    shape=["batch", "channels", "height", "width"],
                    device=device,
                ),
                TensorType(DType.float32, shape=[1], device=device),
            ],
        )

    @traced(message="ZImagePipeline.build_duplicate_batch")
    def build_duplicate_batch(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["duplicate_batch"] = max_compile(
            self.duplicate_batch,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "seq", "channels"],
                    device=device,
                ),
            ],
        )

    @traced(message="ZImagePipeline.build_duplicate_2d")
    def build_duplicate_2d(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["duplicate_2d"] = max_compile(
            self.duplicate_2d,
            input_types=[
                TensorType(dtype, shape=[1, "dim"], device=device),
            ],
        )

    @traced(message="ZImagePipeline.build_cfg_finalize_batched")
    def build_cfg_finalize_batched(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                dtype,
                shape=["double_batch", "seq", "channels"],
                device=device,
            ),
            TensorType(DType.float32, shape=[], device=device),
        ]
        self.__dict__["cfg_finalize_batched_no_norm"] = max_compile(
            self.cfg_finalize_batched_no_norm,
            input_types=input_types,
        )
        self.__dict__["cfg_finalize_batched_with_norm"] = max_compile(
            self.cfg_finalize_batched_with_norm,
            input_types=input_types,
        )

    @traced(message="ZImagePipeline.build_cfg_renormalization")
    def build_cfg_renormalization(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["_cfg_renormalization_compiled"] = max_compile(
            self._apply_cfg_renormalization_enabled,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "seq", "channels"],
                    device=device,
                ),
                TensorType(
                    dtype,
                    shape=["batch", "seq", "channels"],
                    device=device,
                ),
            ],
        )

    @traced(message="ZImagePipeline.build_cfg_combine")
    def build_cfg_combine(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["cfg_combine"] = max_compile(
            self.cfg_combine,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "seq", "channels"],
                    device=device,
                ),
                TensorType(
                    dtype,
                    shape=["batch", "seq", "channels"],
                    device=device,
                ),
                TensorType(DType.float32, shape=[], device=device),
            ],
        )

    def cfg_combine(
        self,
        noise_pred: Tensor,
        neg_noise_pred: Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        """Apply CFG formula: pos + scale * (pos - neg)."""
        input_dtype = noise_pred.dtype
        diff = noise_pred - neg_noise_pred
        scaled = guidance_scale * diff
        result = noise_pred + scaled
        return result.cast(input_dtype)

    @staticmethod
    def duplicate_batch(x: Tensor) -> Tensor:
        """Duplicate batch dimension as [B, ...] -> [2B, ...] via broadcast."""
        batch = x.shape[0]
        seq = x.shape[1]
        channels = x.shape[2]
        x = F.unsqueeze(x, axis=0)
        x = F.broadcast_to(x, [2, batch, seq, channels])
        x = F.reshape(x, [batch * 2, seq, channels])
        return x

    @classmethod
    def _apply_cfg_renormalization_enabled(
        cls,
        pos: Tensor,
        pred: Tensor,
    ) -> Tensor:
        ori_pos_norm = cls._vector_norm_per_sample(pos)
        new_pos_norm = cls._vector_norm_per_sample(pred)
        while ori_pos_norm.rank > 1:
            ori_pos_norm = F.squeeze(ori_pos_norm, axis=-1)
        while new_pos_norm.rank > 1:
            new_pos_norm = F.squeeze(new_pos_norm, axis=-1)
        max_new_norm = ori_pos_norm
        safe_new_norm = F.where(
            new_pos_norm > 1e-12,
            new_pos_norm,
            1e-12,
        )
        ratio = max_new_norm / safe_new_norm
        ratio = F.where(new_pos_norm > max_new_norm, ratio, 1.0)
        ratio = F.unsqueeze(F.unsqueeze(ratio, 1), 2)
        return pred * ratio

    @classmethod
    def cfg_finalize_batched_no_norm(
        cls,
        noise_pred_cfg: Tensor,
        guidance_scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch2 = noise_pred_cfg.shape[0]
        batch = batch2 // 2
        seq = noise_pred_cfg.shape[1]
        channels = noise_pred_cfg.shape[2]
        pos_noise_pred = F.rebind(
            noise_pred_cfg[:batch],
            [batch, seq, channels],
        )
        neg_noise_pred = F.rebind(
            noise_pred_cfg[batch:],
            [batch, seq, channels],
        )
        input_dtype = pos_noise_pred.dtype
        diff = pos_noise_pred - neg_noise_pred
        scaled = guidance_scale * diff
        noise_pred = (pos_noise_pred + scaled).cast(input_dtype)
        return pos_noise_pred, noise_pred

    @classmethod
    def cfg_finalize_batched_with_norm(
        cls,
        noise_pred_cfg: Tensor,
        guidance_scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pos_noise_pred, noise_pred = cls.cfg_finalize_batched_no_norm(
            noise_pred_cfg,
            guidance_scale,
        )
        noise_pred = cls._apply_cfg_renormalization_enabled(
            pos_noise_pred, noise_pred
        )
        return pos_noise_pred, noise_pred

    @staticmethod
    def _patchify_and_pack(latents: Tensor) -> Tensor:
        batch_size = latents.shape[0]
        num_channels = latents.shape[1]
        height = latents.shape[2]
        width = latents.shape[3]
        latents = F.rebind(
            latents,
            [batch_size, num_channels, (height // 2) * 2, (width // 2) * 2],
        )
        latents = F.reshape(
            latents,
            (
                batch_size,
                num_channels,
                height // 2,
                2,
                width // 2,
                2,
            ),
        )
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

    @traced(message="ZImagePipeline.prepare_prompt_embeddings")
    def prepare_prompt_embeddings(
        self,
        tokens: Tensor,
        num_images_per_prompt: int,
    ) -> Tensor:
        """Encode prompt tokens into text embeddings."""
        prompt_embeds = self.text_encoder(tokens)
        if prompt_embeds.rank == 2:
            prompt_embeds = F.unsqueeze(prompt_embeds, axis=0)
        elif prompt_embeds.rank != 3:
            raise ValueError(
                f"Unexpected prompt_embeds rank={prompt_embeds.rank}; expected 2 or 3."
            )
        if num_images_per_prompt > 1:
            batch = int(prompt_embeds.shape[0])
            seq = int(prompt_embeds.shape[1])
            hidden = int(prompt_embeds.shape[2])
            if batch == 1:
                prompt_embeds = F.broadcast_to(
                    prompt_embeds,
                    [num_images_per_prompt, seq, hidden],
                )
            else:
                prompt_embeds = F.concat(
                    [prompt_embeds] * num_images_per_prompt,
                    axis=0,
                )

        return prompt_embeds

    @staticmethod
    def _select_tokens_for_text_encoder(
        tokens: np.ndarray,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        if tokens.ndim == 2:
            tokens = tokens[0]

        selected_tokens = tokens
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[0]
            if mask.shape[0] != tokens.shape[0]:
                raise ValueError(
                    f"ZImage mask length mismatch: mask={mask.shape[0]}, "
                    f"tokens={tokens.shape[0]}."
                )
            selected_mask = mask.astype(np.bool_, copy=False)
            if not np.any(selected_mask):
                raise ValueError("ZImage mask cannot exclude all tokens.")
            if not np.all(selected_mask):
                selected_tokens = tokens[selected_mask]

        return np.ascontiguousarray(
            selected_tokens.astype(np.int64, copy=False)
        )

    def _token_tensor_from_numpy(
        self,
        tokens: np.ndarray,
        device: Device,
    ) -> Tensor:
        token_digest = hashlib.sha1(tokens.tobytes()).hexdigest()
        token_key = (
            f"prompt_tokens::{tokens.shape[0]}::{token_digest}::{device}"
        )
        if token_key in self._cached_prompt_token_tensors:
            return self._cached_prompt_token_tensors[token_key]

        text_input_ids = Tensor(storage=Buffer.from_dlpack(tokens).to(device))
        self._cached_prompt_token_tensors[token_key] = text_input_ids
        return text_input_ids

    def _prepare_conditioning_ids(
        self,
        text_seq_len: int,
        image_seq_len: int,
        latent_image_ids: np.ndarray,
        height: int,
        width: int,
        device: Device,
    ) -> tuple[Tensor, Tensor]:
        text_seq_len_padded = text_seq_len + (-text_seq_len % 32)
        img_ids_base_key = f"img_ids_base::{image_seq_len}_{height}x{width}"
        if img_ids_base_key in self._cached_img_ids_base_np:
            img_ids_base_np = self._cached_img_ids_base_np[img_ids_base_key]
        else:
            img_ids_base_np = np.asarray(latent_image_ids, dtype=np.int64)
            if img_ids_base_np.ndim == 3:
                img_ids_base_np = img_ids_base_np[0]
            img_ids_base_np = np.ascontiguousarray(img_ids_base_np)
            self._cached_img_ids_base_np[img_ids_base_key] = img_ids_base_np

        img_ids_key = (
            f"img_ids::{text_seq_len_padded}_{image_seq_len}_{height}x{width}"
        )
        if img_ids_key in self._cached_img_ids:
            img_ids_tensor = self._cached_img_ids[img_ids_key]
        else:
            img_ids_np = img_ids_base_np.copy()
            img_ids_np[:, 0] = img_ids_np[:, 0] + text_seq_len_padded + 1
            img_ids_tensor = Tensor(
                storage=Buffer.from_dlpack(np.ascontiguousarray(img_ids_np)).to(
                    device
                )
            )
            self._cached_img_ids[img_ids_key] = img_ids_tensor

        text_ids_key = f"text_ids::{text_seq_len}"
        if text_ids_key in self._cached_text_ids:
            txt_ids_tensor = self._cached_text_ids[text_ids_key]
        else:
            txt_ids_tensor = self._prepare_text_ids(text_seq_len, device)
            self._cached_text_ids[text_ids_key] = txt_ids_tensor

        return txt_ids_tensor, img_ids_tensor

    def _align_prompt_seq_len(
        self,
        embeds: Tensor,
        target_seq_len: int,
    ) -> Tensor:
        cur_len = int(embeds.shape[1])
        if cur_len == target_seq_len:
            return embeds
        if cur_len > target_seq_len:
            return embeds[:, :target_seq_len, :]

        pad_len = target_seq_len - cur_len
        pad_key = (
            f"prompt_pad::{int(embeds.shape[0])}::{pad_len}::"
            f"{int(embeds.shape[2])}::{embeds.dtype}::{embeds.device}"
        )
        if pad_key in self._cached_prompt_padding:
            pad = self._cached_prompt_padding[pad_key]
        else:
            pad = Tensor(
                storage=Buffer.zeros(
                    (int(embeds.shape[0]), pad_len, int(embeds.shape[2])),
                    embeds.dtype,
                    device=embeds.device,
                )
            )
            self._cached_prompt_padding[pad_key] = pad
        return F.concat([embeds, pad], axis=1)

    @staticmethod
    def _prepare_text_ids(
        seq_len: int,
        device: Device,
    ) -> Tensor:
        """Build text position IDs in [1, 2, ..., seq_len] format."""
        text_ids = np.zeros((seq_len, 3), dtype=np.int64)
        text_ids[:, 0] = np.arange(1, seq_len + 1, dtype=np.int64)
        return Tensor(
            storage=Buffer.from_dlpack(np.ascontiguousarray(text_ids)).to(
                device
            )
        )

    @traced(message="ZImagePipeline.decode_latents")
    def decode_latents(
        self,
        latents: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        """Decode packed latents into image output."""
        if output_type == "latent":
            return latents

        if hasattr(self, "_fused_decode"):
            decoded = self._fused_decode(latents, h_carrier, w_carrier)
            return np.from_dlpack(decoded)

        latents = self._unpack_and_postprocess(latents, h_carrier, w_carrier)
        decoded = self.vae.decode(latents)
        image = self._postprocess_image(decoded)
        return np.from_dlpack(image.to(CPU()))

    def _unpack_and_postprocess(
        self,
        latents: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
    ) -> Tensor:
        """Unpack [B, S, C] → [B, C/4, H, W] and apply VAE scaling."""
        batch_size = latents.shape[0]
        ch_size = latents.shape[2]
        half_h = h_carrier.shape[0]
        half_w = w_carrier.shape[0]

        latents = F.reshape(
            latents, (batch_size, half_h, half_w, 2, 2, ch_size // 4)
        )
        latents = F.permute(latents, (0, 5, 1, 3, 2, 4))
        latents = F.reshape(
            latents, (batch_size, ch_size // 4, half_h * 2, half_w * 2)
        )
        latents = (latents / float(self.vae.config.scaling_factor)) + float(
            self.vae.config.shift_factor or 0.0
        )
        return latents

    @staticmethod
    def _postprocess_image(image: Tensor) -> Tensor:
        """Normalize VAE output from [-1, 1] NCHW to [0, 255] uint8 NHWC."""
        image = image.cast(DType.float32)
        image = image * 0.5 + 0.5
        image = F.where(image < 0.0, 0.0, image)
        image = F.where(image > 1.0, 1.0, image)
        image = F.permute(image, (0, 2, 3, 1))
        image = image * 255.0
        return image.cast(DType.uint8)

    @staticmethod
    def _vector_norm_per_sample(x: Tensor) -> Tensor:
        """Compute per-sample norm for [B, S, C] embeddings."""
        squared = x * x
        squared = F.sum(squared, axis=2)
        squared = F.sum(squared, axis=1)
        return F.sqrt(squared + 1e-12)

    @staticmethod
    def scheduler_step(
        latents: Tensor,
        noise_pred: Tensor,
        dt: Tensor,
    ) -> Tensor:
        latents_dtype = latents.dtype
        latents = latents.cast(DType.float32)
        latents = latents - dt * noise_pred
        latents = latents.cast(latents_dtype)
        return latents

    @traced(message="ZImagePipeline.preprocess_latents")
    def preprocess_latents(self, latents: Tensor) -> Tensor:
        """Patchify and pack latents before denoising."""
        with Tracer("host_to_device_latents"):
            latents = latents.to(self.transformer.devices[0]).cast(
                DType.float32
            )

        with Tracer("patchify_and_pack"):
            latents = self._patchify_and_pack(latents)

        return latents

    def _numpy_image_to_tensor(
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
        image_bchw = np.ascontiguousarray(image_bchw)

        image_tensor = Tensor(
            storage=Buffer.from_dlpack(image_bchw).to(self.vae.devices[0])
        ).cast(dtype)
        if batch_size > 1:
            image_tensor = F.broadcast_to(
                image_tensor,
                [batch_size, 3, height, width],
            )
        return image_tensor

    def _blend_img2img_latents(
        self,
        image_latents: Tensor,
        noise_latents: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        """Blend image latents with noise using sigma (compiled graph)."""
        scaling = float(self.vae.config.scaling_factor)
        shift = float(self.vae.config.shift_factor or 0.0)
        image_latents = (image_latents - shift) * scaling
        result = sigma * noise_latents + (1.0 - sigma) * image_latents
        return result.cast(noise_latents.dtype)

    @traced(message="ZImagePipeline.prepare_image_latents")
    def prepare_image_latents(
        self,
        noise_latents: Tensor,
        image_tensor: Tensor,
        sigmas: Tensor,
    ) -> Tensor:
        noise_latents = noise_latents.to(self.transformer.devices[0])

        encoder_output = self.vae.encode(image_tensor, return_dict=True)
        posterior = (
            encoder_output["latent_dist"]
            if isinstance(encoder_output, dict)
            else encoder_output
        )
        if not hasattr(posterior, "mode"):
            raise ValueError("VAE encoder output must expose mode().")

        image_latents = posterior.mode()
        image_latents = image_latents.to(self.transformer.devices[0]).cast(
            noise_latents.dtype
        )

        sigma_0 = sigmas[:1]
        return self._blend_img2img_latents(
            image_latents, noise_latents, sigma_0
        )

    def _get_cached_guidance(
        self,
        guidance_scale: float,
        device: Device,
    ) -> Tensor:
        key = f"{guidance_scale:.8f}::{device}"
        if key in self._cached_guidance:
            return self._cached_guidance[key]
        tensor = Tensor(
            storage=Buffer.from_dlpack(
                np.array(guidance_scale, dtype=np.float32)
            ).to(device)
        )
        self._cached_guidance[key] = tensor
        return tensor

    @traced(message="ZImagePipeline.execute")
    def execute(  # type: ignore[override]
        self,
        model_inputs: ZImageModelInputs,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> DiffusionPipelineOutput:
        """Run the Z-Image denoising loop and decode outputs."""

        # 1) Encode prompt embeddings.
        with Tracer("prepare_prompt_embeddings"):
            prompt_embeds = self.prepare_prompt_embeddings(
                tokens=model_inputs.tokens_tensor,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )

            negative_prompt_embeds: Tensor | None = None
            if (
                model_inputs.do_cfg
                and model_inputs.negative_tokens_tensor is not None
            ):
                negative_prompt_embeds = self.prepare_prompt_embeddings(
                    tokens=model_inputs.negative_tokens_tensor,
                    num_images_per_prompt=model_inputs.num_images_per_prompt,
                )
                if not model_inputs.explicit_negative_prompt:
                    negative_prompt_embeds = self._align_prompt_seq_len(
                        negative_prompt_embeds,
                        int(prompt_embeds.shape[1]),
                    )

        dtype = prompt_embeds.dtype
        latents = model_inputs.latents_tensor
        sigmas = model_inputs.sigmas_tensor
        h_carrier = model_inputs.h_carrier
        w_carrier = model_inputs.w_carrier

        timesteps: np.ndarray = model_inputs.timesteps
        batch_size = int(prompt_embeds.shape[0])
        num_timesteps = timesteps.shape[0]
        if num_timesteps < 1:
            raise ValueError("No timesteps were provided for denoising.")

        # 2) Prepare latents and conditioning tensors.
        device = self.transformer.devices[0]
        img_ids = model_inputs.img_ids_tensor
        txt_ids = model_inputs.txt_ids_tensor

        if model_inputs.input_image_tensor is not None:
            latents = self.prepare_image_latents(
                noise_latents=latents,
                image_tensor=model_inputs.input_image_tensor,
                sigmas=sigmas,
            )
        latents = self.preprocess_latents(latents)

        image_seq_len = int(latents.shape[1])
        use_fast_denoise = not self.cache_config.first_block_caching
        use_batched_cfg_mode = bool(
            model_inputs.do_cfg
            and not model_inputs.explicit_negative_prompt
            and use_fast_denoise
        )

        # Cache states are only needed for the non-split fallback path.
        cache_pos = self.create_cache_state(
            batch_size,
            image_seq_len,
            self.transformer.config,
        )
        cache_neg: DenoisingCacheState | None = None
        if not use_fast_denoise and model_inputs.do_cfg and not use_batched_cfg_mode:
            cache_neg = self.create_cache_state(
                batch_size,
                image_seq_len,
                self.transformer.config,
            )

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

            # Cache per-step scalar tensors to avoid repeated uploads.
            combined = np.concatenate([transformed_timesteps, dt_values])
            step_key = (
                f"steps::{num_timesteps}::"
                f"{hashlib.sha1(combined.tobytes()).hexdigest()}"
            )
            if step_key in self._cached_step_tensors:
                timestep_tensors, dt_tensors = self._cached_step_tensors[
                    step_key
                ]
            else:
                timestep_tensors = [
                    Tensor(
                        storage=Buffer.from_dlpack(
                            np.array([float(t)], dtype=np.float32)
                        ).to(device)
                    )
                    for t in transformed_timesteps
                ]
                dt_tensors = [
                    Tensor(
                        storage=Buffer.from_dlpack(
                            np.array([float(d)], dtype=np.float32)
                        ).to(device)
                    )
                    for d in dt_values
                ]
                self._cached_step_tensors[step_key] = (
                    timestep_tensors,
                    dt_tensors,
                )

        cfg_cutoff_step = 0
        if model_inputs.do_cfg:
            if model_inputs.cfg_truncation > 1.0:
                cfg_cutoff_step = num_timesteps
            else:
                mask = transformed_timesteps <= model_inputs.cfg_truncation
                cfg_cutoff_step = int(np.count_nonzero(mask))

        guidance_scale_tensor: Tensor | None = None
        if model_inputs.do_cfg:
            guidance_scale_tensor = self._get_cached_guidance(
                model_inputs.guidance_scale,
                device,
            )

        # Batched CFG prompt embeddings (only for batched mode).
        cfg_prompt_embeds: Tensor | None = None
        if use_batched_cfg_mode:
            assert negative_prompt_embeds is not None
            cfg_prompt_embeds = F.concat(
                [prompt_embeds, negative_prompt_embeds], axis=0
            )

        # 4) Denoising loop.
        with Tracer("denoising_loop"):
            for i in range(num_timesteps):
                apply_cfg = i < cfg_cutoff_step
                timestep = timestep_tensors[i]
                dt = dt_tensors[i]
                preamble_out: tuple[Tensor, ...] | None = None
                with Tracer(f"denoising_step_{i}"):
                    with Tracer("transformer"):
                        if apply_cfg and use_batched_cfg_mode:
                            assert cfg_prompt_embeds is not None
                            noise_pred_cfg = self.run_transformer_cfg_split(
                                latents=latents,
                                prompt_embeds=cfg_prompt_embeds,
                                timestep=timestep,
                                img_ids=img_ids,
                                txt_ids=txt_ids,
                            )
                            assert guidance_scale_tensor is not None
                            if model_inputs.cfg_normalization:
                                _, noise_pred = self.cfg_finalize_batched_with_norm(
                                    noise_pred_cfg,
                                    guidance_scale_tensor,
                                )
                            else:
                                _, noise_pred = self.cfg_finalize_batched_no_norm(
                                    noise_pred_cfg,
                                    guidance_scale_tensor,
                                )
                        elif apply_cfg:
                            # Non-batched CFG: split preamble shared,
                            # run main separately for pos and neg.
                            if use_fast_denoise:
                                preamble_out = self.transformer.run_preamble(
                                    latents, timestep, img_ids, txt_ids,
                                )
                                x_sh, t_emb_sh, uf_sh, tf_sh = preamble_out
                                noise_pred = self.transformer.run_cfg_main(
                                    x_sh, prompt_embeds, t_emb_sh, uf_sh, tf_sh,
                                )[0]
                            else:
                                noise_pred = self.run_denoising_step(
                                    step=i,
                                    cache_state=cache_pos,
                                    device=device,
                                    latents=latents,
                                    prompt_embeds=prompt_embeds,
                                    timestep=timestep,
                                    img_ids=img_ids,
                                    txt_ids=txt_ids,
                                )
                        else:
                            # No CFG this step.
                            if use_fast_denoise:
                                noise_pred = self.run_transformer(
                                    cache_pos,
                                    latents=latents,
                                    prompt_embeds=prompt_embeds,
                                    timestep=timestep,
                                    img_ids=img_ids,
                                    txt_ids=txt_ids,
                                )[0]
                            else:
                                noise_pred = self.run_denoising_step(
                                    step=i,
                                    cache_state=cache_pos,
                                    device=device,
                                    latents=latents,
                                    prompt_embeds=prompt_embeds,
                                    timestep=timestep,
                                    img_ids=img_ids,
                                    txt_ids=txt_ids,
                                )

                    # Negative pass for non-batched CFG.
                    if apply_cfg and not use_batched_cfg_mode:
                        assert negative_prompt_embeds is not None
                        neg_img_ids = img_ids
                        neg_txt_ids = txt_ids
                        if model_inputs.explicit_negative_prompt:
                            assert model_inputs.negative_img_ids_tensor is not None
                            assert model_inputs.negative_txt_ids_tensor is not None
                            neg_img_ids = model_inputs.negative_img_ids_tensor
                            neg_txt_ids = model_inputs.negative_txt_ids_tensor
                        with Tracer("cfg_transformer"):
                            if use_fast_denoise:
                                assert preamble_out is not None
                                x_sh, t_emb_sh, _, _ = preamble_out
                                neg_uf, neg_tf = self.transformer.compute_rope(
                                    neg_img_ids, neg_txt_ids,
                                )
                                neg_noise_pred = self.transformer.run_cfg_main(
                                    x_sh, negative_prompt_embeds, t_emb_sh,
                                    neg_uf, neg_tf,
                                )[0]
                            else:
                                assert cache_neg is not None
                                neg_noise_pred = self.run_denoising_step(
                                    step=i,
                                    cache_state=cache_neg,
                                    device=device,
                                    latents=latents,
                                    prompt_embeds=negative_prompt_embeds,
                                    timestep=timestep,
                                    img_ids=neg_img_ids,
                                    txt_ids=neg_txt_ids,
                                )
                        assert guidance_scale_tensor is not None
                        noise_pred = self.cfg_combine(
                            noise_pred,
                            neg_noise_pred,
                            guidance_scale_tensor,
                        )
                        if model_inputs.cfg_normalization:
                            noise_pred = self._cfg_renormalization_compiled(
                                noise_pred, noise_pred,
                            )

                    with Tracer("scheduler_step"):
                        latents = self.scheduler_step(latents, noise_pred, dt)

        with Tracer("decode_outputs"):
            outputs = self.decode_latents(
                latents,
                h_carrier,
                w_carrier,
                output_type=output_type,
            )

        return DiffusionPipelineOutput(images=outputs)
