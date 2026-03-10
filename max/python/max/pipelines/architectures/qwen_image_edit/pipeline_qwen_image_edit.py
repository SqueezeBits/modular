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

"""QwenImage edit diffusion pipeline.

Key differences from QwenImagePipeline:
- Multimodal prompt encoding when edit images are present
- VAE image-conditioning path that concatenates condition latents to noise
- True CFG with two forward passes (positive + negative prompts)
"""

import logging
from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.interfaces import TokenBuffer
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import DiffusionPipeline, PixelModelInputs
from max.pipelines.lib.interfaces.diffusion_pipeline import (
    CompileWrapper,
    max_compile,
)
from max.profiler import Tracer, traced

from ..autoencoders.autoencoder_kl_qwen_image import AutoencoderKLQwenImageModel
from ..qwen2_5vl.encoder import (
    Qwen25VLMultimodalEncoderModel,
    Qwen25VLEncoderModel,
)
from .model import QwenImageEditTransformerModel

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class QwenImageEditModelInputs(PixelModelInputs):
    """QwenImage-edit-specific PixelModelInputs.

    For image editing the recommended usage is
    ``--guidance-scale 1.0 --true-cfg-scale 4.0``.
    ``guidance_scale`` is unused (model is not guidance-distilled);
    ``true_cfg_scale`` drives the two-pass CFG behavior.
    """

    width: int = 1024
    height: int = 1024
    guidance_scale: float = 1.0
    true_cfg_scale: float = 4.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    prompt_images: list[npt.NDArray[np.uint8]] | None = None
    vae_images: list[npt.NDArray[np.uint8]] | None = None


@dataclass
class QwenImageEditPipelineOutput:
    """Container for QwenImage edit pipeline results."""

    images: np.ndarray | list


class QwenImageEditPipeline(DiffusionPipeline):
    """Diffusion pipeline for QwenImage image editing.

    Wires together:
    - Qwen2.5-VL prompt encoder
    - QwenImage edit transformer denoiser
    - QwenImage 3D VAE (with latents_mean/std normalization)
    - Image-conditioning path (VAE encode -> normalize -> patchify -> concat)
    """

    vae: AutoencoderKLQwenImageModel
    text_encoder: Qwen25VLEncoderModel
    transformer: QwenImageEditTransformerModel

    components = {
        "vae": AutoencoderKLQwenImageModel,
        "text_encoder": Qwen25VLEncoderModel,
        "transformer": QwenImageEditTransformerModel,
    }

    # NOTE:
    # `prompt_encoder` is intentionally not part of `components`.
    #
    # QwenImageEdit needs a multimodal prompt path that reuses the already-loaded
    # `text_encoder` and layers a vision encoder + prompt/image merge logic on top.
    # That makes it closer to an edit-specific helper than an independent pipeline
    # submodel. Keeping it out of `components` avoids adding special loading rules
    # to the shared DiffusionPipeline base just for this dependency shape.
    prompt_encoder: Qwen25VLMultimodalEncoderModel | None = None
    _prompt_encoder_config: dict[str, Any] | None = None
    _prompt_encoder_weight_paths: list[str] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and len(args) >= 4:
            self._weight_paths = args[3]
        else:
            self._weight_paths = kwargs.get("weight_paths", [])
        super().__init__(*args, **kwargs)

    def init_remaining_components(self) -> None:
        """Initialize derived attributes that depend on loaded components."""
        self.vae_scale_factor = 8

        self.build_patchify()
        self.build_scheduler()
        self.build_scheduler_step()
        self.build_postprocess_latents()
        self.build_cfg_blend()
        self.build_normalize_and_pack()
        self.build_concat_image_latents()

        self._cached_sigmas: dict[str, Buffer] = {}
        self._cached_text_ids: dict[str, Buffer] = {}
        self._cached_fns: dict[str, Any] = {}

        diffusers_config = self.pipeline_config.model.diffusers_config
        components_config = diffusers_config.get("components", {})
        self._prompt_encoder_config = components_config.get(
            "text_encoder", {}
        ).get("config_dict", {})

        relative_paths = self._resolve_relative_component_paths()
        text_encoder_rel_paths = relative_paths.get("text_encoder", [])
        self._prompt_encoder_weight_paths = self._resolve_absolute_paths(
            self._weight_paths, text_encoder_rel_paths
        )

    def _init_prompt_encoder(self) -> None:
        if self.prompt_encoder is not None:
            return

        # NOTE:
        # This is a local assembly step, not a normal ComponentModel load.
        #
        # The edit prompt encoder depends on the already-instantiated
        # `self.text_encoder`, reuses the text-encoder weight set, and adds the
        # Qwen2.5-VL vision path needed for multimodal prompt encoding. If we
        # tried to model it as a regular pipeline component, the shared loader
        # would need special-case dependency wiring for "component B depends on
        # loaded component A", which is more confusing than keeping the assembly
        # here in the edit pipeline.
        from max.graph.weights import load_weights
        if self._prompt_encoder_config is None:
            raise ValueError("prompt encoder config is not initialized")
        if self._prompt_encoder_weight_paths is None:
            raise ValueError("prompt encoder weight paths are not initialized")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.pipeline_config.model.model_path,
            subfolder="tokenizer",
        )

        self.prompt_encoder = Qwen25VLMultimodalEncoderModel(
            text_encoder=self.text_encoder,
            config=self._prompt_encoder_config,
            encoding=self.pipeline_config.model.quantization_encoding,
            devices=self.devices,
            weights=load_weights(self._prompt_encoder_weight_paths),
            session=self.session,
            tokenizer=tokenizer,
        )

    def _get_prompt_encoder(self) -> Qwen25VLMultimodalEncoderModel:
        # NOTE:
        # We only need the multimodal prompt path when edit images are present.
        # Text-only prompt encoding stays on `self.text_encoder`, so we avoid
        # paying the extra vision-side setup cost unless the request actually
        # uses image conditioning.
        if self.prompt_encoder is None:
            self._init_prompt_encoder()
            if self.prompt_encoder is None:
                raise ValueError("failed to initialize prompt_encoder")
        return self.prompt_encoder

    def _encode_prompt(
        self,
        *,
        tokens: TokenBuffer,
        prompt_images: list[npt.NDArray[np.uint8]],
        num_images_per_prompt: int,
        prompt_encoder: Qwen25VLMultimodalEncoderModel | None,
    ) -> Buffer:
        if prompt_images:
            assert prompt_encoder is not None
            return prompt_encoder.encode(
                tokens=tokens,
                images=prompt_images,
                num_images_per_prompt=num_images_per_prompt,
            )

        return self.prepare_prompt_embeddings(
            tokens=tokens,
            num_images_per_prompt=num_images_per_prompt,
        )

    @staticmethod
    def _resolve_condition_images(
        model_inputs: QwenImageEditModelInputs,
    ) -> tuple[list[npt.NDArray[np.uint8]], list[npt.NDArray[np.uint8]]]:
        prompt_images = model_inputs.prompt_images or model_inputs.input_images or []
        vae_images = model_inputs.vae_images or model_inputs.input_images or []
        return prompt_images, vae_images

    def _prepare_negative_prompt_embeddings(
        self,
        *,
        model_inputs: QwenImageEditModelInputs,
        prompt_images: list[npt.NDArray[np.uint8]],
        prompt_encoder: Qwen25VLMultimodalEncoderModel | None,
    ) -> Buffer | None:
        if (
            model_inputs.true_cfg_scale <= 1.0
            or model_inputs.negative_tokens is None
        ):
            return None

        return self._encode_prompt(
            tokens=model_inputs.negative_tokens,
            prompt_images=prompt_images,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
            prompt_encoder=prompt_encoder,
        )

    def _prepare_condition_latents(
        self,
        *,
        vae_images: list[npt.NDArray[np.uint8]],
        batch_size: int,
        device: Device,
    ) -> tuple[Buffer | None, Buffer | None]:
        if not vae_images:
            return None, None

        image_bufs = [self._numpy_image_to_buffer(image) for image in vae_images]
        return self.prepare_image_latents(
            images=image_bufs,
            batch_size=batch_size,
            device=device,
        )

    def _prepare_text_ids_for_embeddings(
        self,
        *,
        embeddings: Buffer,
        batch_size: int,
        device: Device,
        max_vid_index: int,
    ) -> Buffer:
        seq_len = embeddings.shape[1]
        cache_key = f"{batch_size}_{seq_len}_{max_vid_index}"
        if cache_key not in self._cached_text_ids:
            self._cached_text_ids[cache_key] = self._prepare_text_ids(
                batch_size, seq_len, device, max_vid_index
            )
        return self._cached_text_ids[cache_key]

    def prepare_inputs(self, context: PixelContext) -> QwenImageEditModelInputs:  # type: ignore[override]
        """Convert a PixelContext into QwenImageEditModelInputs."""
        return QwenImageEditModelInputs.from_context(context)

    def build_patchify(self) -> None:
        device = self.transformer.devices[0]
        self.__dict__["_patchify_and_pack"] = max_compile(
            self._patchify_and_pack,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=["batch", "channels", "height", 2, "width", 2],
                    device=device,
                )
            ],
        )

    def build_scheduler(self) -> None:
        self.__dict__["prepare_scheduler"] = max_compile(
            self.prepare_scheduler,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=["num_sigmas"],
                    device=self.transformer.devices[0],
                )
            ],
        )

    def build_scheduler_step(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["scheduler_step"] = max_compile(
            self.scheduler_step,
            input_types=[
                TensorType(dtype, shape=["batch", "seq", "channels"], device=device),
                TensorType(dtype, shape=["batch", "pred_seq", "channels"], device=device),
                TensorType(DType.float32, shape=[1], device=device),
                TensorType(DType.int64, shape=[], device=DeviceRef.CPU()),
            ],
        )

    def build_postprocess_latents(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        z_dim = 16
        packed_channels = self.transformer.config.in_channels
        self.__dict__["_postprocess_latents"] = max_compile(
            self._postprocess_latents,
            input_types=[
                TensorType(dtype, shape=["batch", "height", "width", packed_channels], device=device),
                TensorType(dtype, shape=[z_dim], device=device),
                TensorType(dtype, shape=[z_dim], device=device),
            ],
        )

    def build_cfg_blend(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["_cfg_blend"] = max_compile(
            self._cfg_blend,
            input_types=[
                TensorType(dtype, shape=["batch", "seq", "channels"], device=device),
                TensorType(dtype, shape=["batch", "seq", "channels"], device=device),
                TensorType(DType.float32, shape=[1], device=device),
            ],
        )

    def build_normalize_and_pack(self) -> None:
        dtype = self.vae.config.dtype
        device = self.vae.devices[0]
        z_dim = self.vae.config.z_dim
        self.__dict__["_normalize_and_pack_image_latent"] = max_compile(
            self._normalize_and_pack_image_latent,
            input_types=[
                TensorType(dtype, shape=["batch", z_dim, "height", 2, "width", 2], device=device),
                TensorType(dtype, shape=[z_dim], device=device),
                TensorType(dtype, shape=[z_dim], device=device),
            ],
        )

    def build_concat_image_latents(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["concat_image_latents"] = max_compile(
            self.concat_image_latents,
            input_types=[
                TensorType(dtype, shape=["batch", "seq", "channels"], device=device),
                TensorType(dtype, shape=["batch", "img_seq", "channels"], device=device),
                TensorType(DType.int64, shape=["batch", "seq", 3], device=device),
                TensorType(DType.int64, shape=["batch", "img_seq", 3], device=device),
            ],
        )

    def _patchify_and_pack(self, latents: TensorValue) -> TensorValue:
        """(B,C,H//2,2,W//2,2) → (B, H//2*W//2, C*4)"""
        latents = ops.cast(latents, self.transformer.config.dtype)
        batch = latents.shape[0]
        c = latents.shape[1]
        h2 = latents.shape[2]
        w2 = latents.shape[4]
        latents = ops.permute(latents, (0, 1, 3, 5, 2, 4))
        latents = ops.reshape(latents, (batch, c * 4, h2, w2))
        latents = ops.reshape(latents, (batch, c * 4, h2 * w2))
        return ops.permute(latents, (0, 2, 1))

    def _postprocess_latents(
        self,
        latents_bhwc: TensorValue,
        latents_mean: TensorValue,
        latents_std: TensorValue,
    ) -> TensorValue:
        """Unpatchify (B,H,W,C*4) → (B,z_dim,H*2,W*2) and denormalize."""
        batch = latents_bhwc.shape[0]
        h = latents_bhwc.shape[1]
        w = latents_bhwc.shape[2]
        c = latents_bhwc.shape[3]
        z_dim = c // 4
        latents = ops.permute(latents_bhwc, (0, 3, 1, 2))
        latents = ops.reshape(latents, (batch, z_dim, 2, 2, h, w))
        latents = ops.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = ops.reshape(latents, (batch, z_dim, h * 2, w * 2))
        mean_r = ops.reshape(latents_mean, (1, z_dim, 1, 1))
        std_r = ops.reshape(latents_std, (1, z_dim, 1, 1))
        return latents * std_r + mean_r

    def _normalize_and_pack_image_latent(
        self,
        image_latents: TensorValue,
        latents_mean: TensorValue,
        latents_std: TensorValue,
    ) -> TensorValue:
        """Normalize VAE output, then patchify+pack to (B, seq, C*4)."""
        batch = image_latents.shape[0]
        c = image_latents.shape[1]
        h2 = image_latents.shape[2]
        w2 = image_latents.shape[4]
        mean_r = ops.reshape(latents_mean, (1, c, 1, 1))
        std_r = ops.reshape(latents_std, (1, c, 1, 1))
        raw = ops.reshape(image_latents, (batch, c, h2 * 2, w2 * 2))
        raw = (raw - mean_r) / std_r
        packed = ops.reshape(raw, (batch, c, h2, 2, w2, 2))
        packed = ops.permute(packed, (0, 1, 3, 5, 2, 4))
        packed = ops.reshape(packed, (batch, c * 4, h2, w2))
        packed = ops.reshape(packed, (batch, c * 4, h2 * w2))
        return ops.permute(packed, (0, 2, 1))

    def _cfg_blend(
        self,
        cond_pred: TensorValue,
        uncond_pred: TensorValue,
        cfg_scale: TensorValue,
    ) -> TensorValue:
        scale = ops.cast(cfg_scale, cond_pred.dtype)
        return uncond_pred + scale * (cond_pred - uncond_pred)

    def concat_image_latents(
        self,
        latents: TensorValue,
        image_latents: TensorValue,
        latent_image_ids: TensorValue,
        image_latent_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        return (
            ops.concat([latents, image_latents], axis=1),
            ops.concat([latent_image_ids, image_latent_ids], axis=1),
        )

    def scheduler_step(
        self,
        latents: TensorValue,
        noise_pred: TensorValue,
        dt: TensorValue,
        num_noise_tokens: int,
    ) -> TensorValue:
        """Single Euler step that only updates the noise tokens."""
        lat = ops.slice_tensor(
            latents,
            [slice(None), (slice(0, num_noise_tokens), "n"), slice(None)],
        )
        pred = ops.slice_tensor(
            noise_pred,
            [slice(None), (slice(0, num_noise_tokens), "n"), slice(None)],
        )
        lat_dtype = lat.dtype
        lat = ops.cast(lat, DType.float32)
        lat = lat + dt * pred
        return ops.cast(lat, lat_dtype)

    def prepare_scheduler(
        self, sigmas: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Precompute timesteps and dt values from sigmas."""
        sigmas_curr = ops.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = ops.slice_tensor(sigmas, [slice(1, None)])
        return (
            ops.cast(sigmas_curr, self.transformer.config.dtype),
            sigmas_next - sigmas_curr,
        )

    # ── prompt encoding ───────────────────────────────────────────────────

    PROMPT_TEMPLATE_DROP_IDX = 34

    def prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        num_images_per_prompt: int = 1,
    ) -> Buffer:
        device = self.text_encoder.devices[0]
        text_input_ids_np = np.asarray(tokens.array).flatten()
        token_buf = Buffer.from_dlpack(
            np.ascontiguousarray(text_input_ids_np)
        ).to(device)

        hidden_states_all = self.text_encoder(token_buf)
        hs_buf = hidden_states_all[-1]

        hs_cpu = hs_buf.to(CPU())
        if self.text_encoder.config.dtype == DType.bfloat16:
            hs_u16 = np.from_dlpack(
                hs_cpu.view(dtype=DType.uint16, shape=hs_cpu.shape)
            )
            hs_np = (hs_u16.astype(np.uint32) << 16).view(np.float32)
        else:
            hs_np = np.from_dlpack(hs_cpu).astype(np.float32)

        hs_np = hs_np[self.PROMPT_TEMPLATE_DROP_IDX:]
        hs_np = hs_np[np.newaxis, :, :]

        if num_images_per_prompt != 1:
            hs_np = np.tile(hs_np, (num_images_per_prompt, 1, 1))

        from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
        if self.text_encoder.config.dtype == DType.bfloat16:
            result_u16 = float32_to_bfloat16_as_uint16(
                np.ascontiguousarray(hs_np)
            )
            buf = Buffer.from_numpy(result_u16).to(device)
            return buf.view(dtype=DType.bfloat16, shape=hs_np.shape)
        return Buffer.from_numpy(
            np.ascontiguousarray(hs_np)
        ).to(device)

    # ── position ID helpers ───────────────────────────────────────────────

    @staticmethod
    def _prepare_text_ids(
        batch_size: int, seq_len: int, device: Device, max_vid_index: int = 0
    ) -> Buffer:
        """Create 3D text position IDs in (T, H, W) format."""
        tok_positions = np.arange(seq_len, dtype=np.int64) + max_vid_index
        coords = np.stack([tok_positions, tok_positions, tok_positions], axis=-1)
        return Buffer.from_dlpack(
            np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
        ).to(device)

    @staticmethod
    def _prepare_image_ids(
        batch_size: int, height: int, width: int, device: Device
    ) -> Buffer:
        """Create 3D image position IDs in (T, H, W) format."""
        t_coords = np.zeros((height, width), dtype=np.int64)
        h_c = np.arange(height, dtype=np.int64) - (height - height // 2)
        w_c = np.arange(width, dtype=np.int64) - (width - width // 2)
        h_coords, w_coords = np.meshgrid(h_c, w_c, indexing="ij")
        coords = np.stack([t_coords, h_coords, w_coords], axis=-1).reshape(-1, 3)
        return Buffer.from_dlpack(
            np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
        ).to(device)

    @staticmethod
    def _prepare_condition_image_ids(
        batch_size: int, height: int, width: int, device: Device,
        image_index: int = 0,
    ) -> Buffer:
        """Condition-image IDs with T=image_index+1 (noise tokens use T=0).

        For multi-image editing each condition image needs a distinct T
        coordinate so the transformer can distinguish them via RoPE:
        noise → T=0, first image → T=1, second image → T=2, etc.
        """
        t_coords = np.full((height, width), image_index + 1, dtype=np.int64)
        h_c = np.arange(height, dtype=np.int64) - (height - height // 2)
        w_c = np.arange(width, dtype=np.int64) - (width - width // 2)
        h_coords, w_coords = np.meshgrid(h_c, w_c, indexing="ij")
        coords = np.stack([t_coords, h_coords, w_coords], axis=-1).reshape(-1, 3)
        return Buffer.from_dlpack(
            np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
        ).to(device)

    # ── latent preprocessing ──────────────────────────────────────────────

    def preprocess_latents(
        self,
        latents: npt.NDArray[np.float32],
        latent_image_ids: npt.NDArray[np.float32],
    ) -> tuple[Buffer, Buffer]:
        latents_np = np.asarray(latents)
        b, c, h, w = latents_np.shape
        latents_6d = latents_np.reshape(b, c, h // 2, 2, w // 2, 2)
        device = self.transformer.devices[0]
        latents_packed = self._patchify_and_pack(
            Buffer.from_dlpack(np.ascontiguousarray(latents_6d)).to(device)
        )
        ids_buf = Buffer.from_dlpack(
            np.asarray(latent_image_ids, dtype=np.int64)
        ).to(device)
        return latents_packed, ids_buf

    # ── image conditioning ────────────────────────────────────────────────

    def _numpy_image_to_buffer(self, image: npt.NDArray[np.uint8]) -> Buffer:
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        img_array = (image.astype(np.float32) / 127.5) - 1.0
        img_array = np.ascontiguousarray(
            np.expand_dims(np.transpose(img_array, (2, 0, 1)), 0)
        )
        vae_dtype = self.vae.config.dtype
        device = self.vae.devices[0]
        if vae_dtype == DType.bfloat16:
            from max.pipelines.lib.bfloat16_utils import (
                float32_to_bfloat16_as_uint16,
            )

            u16 = float32_to_bfloat16_as_uint16(img_array)
            buf = Buffer.from_numpy(u16).to(device)
            return buf.view(dtype=DType.bfloat16, shape=img_array.shape)
        if vae_dtype == DType.float16:
            img_array = img_array.astype(np.float16)
        return Buffer.from_dlpack(img_array).to(device)

    def _encode_single_image(
        self,
        image: Buffer,
        device: Device,
        image_index: int = 0,
    ) -> tuple[Buffer, Buffer]:
        latents_mean = self.vae.latents_mean_tensor
        latents_std = self.vae.latents_std_tensor
        if latents_mean is None or latents_std is None:
            raise ValueError("VAE latents_mean/latents_std are required.")

        raw_latents = self.vae.encode(image.to(device))
        raw_buf = (
            raw_latents.driver_tensor
            if hasattr(raw_latents, "driver_tensor")
            else raw_latents
        )
        raw_b, raw_c, raw_h, raw_w = raw_buf.shape

        reshape_key = f"vae_reshape_{raw_b}_{raw_c}_{raw_h}_{raw_w}"
        if reshape_key not in self._cached_fns:
            vae_dtype = self.vae.config.dtype

            def _reshape_6d(x: TensorValue) -> TensorValue:
                return ops.reshape(x, (raw_b, raw_c, raw_h // 2, 2, raw_w // 2, 2))

            self._cached_fns[reshape_key] = max_compile(
                _reshape_6d,
                input_types=[
                    TensorType(
                        vae_dtype,
                        shape=[raw_b, raw_c, raw_h, raw_w],
                        device=device,
                    )
                ],
            )
        latents_6d = self._cached_fns[reshape_key](raw_buf)
        image_latents = self._normalize_and_pack_image_latent(
            latents_6d, latents_mean, latents_std
        )
        image_ids = self._prepare_condition_image_ids(
            1,
            raw_h // 2,
            raw_w // 2,
            device,
            image_index=image_index,
        )
        return image_latents, image_ids

    def prepare_image_latents(
        self, images: list[Buffer], batch_size: int, device: Device
    ) -> tuple[Buffer, Buffer]:
        all_latents: list[Buffer] = []
        all_ids: list[Buffer] = []
        for idx, img in enumerate(images):
            lat, ids = self._encode_single_image(img, device, image_index=idx)
            all_latents.append(lat)
            all_ids.append(ids)

        if len(all_latents) == 1:
            image_latents, image_ids = all_latents[0], all_ids[0]
        else:
            image_latents = self._concat_buffers_seq(all_latents, device)
            id_arrays = [np.from_dlpack(ids.to(CPU())) for ids in all_ids]
            image_ids = Buffer.from_dlpack(
                np.ascontiguousarray(np.concatenate(id_arrays, axis=1))
            ).to(device)

        if batch_size > 1:
            lat_np = np.from_dlpack(image_latents.to(CPU()))
            image_latents = Buffer.from_dlpack(
                np.ascontiguousarray(np.tile(lat_np, (batch_size, 1, 1)))
            ).to(device)
            ids_np = np.from_dlpack(image_ids.to(CPU()))
            image_ids = Buffer.from_dlpack(
                np.ascontiguousarray(np.tile(ids_np, (batch_size, 1, 1)))
            ).to(device)

        return image_latents, image_ids

    def _concat_buffers_seq(self, buffers: list[Buffer], device: Device) -> Buffer:
        result = buffers[0]
        for i in range(1, len(buffers)):
            result = self._compiled_seq_concat(result, buffers[i], device)
        return result

    def _compiled_seq_concat(self, a: Buffer, b: Buffer, device: Device) -> Buffer:
        s1, s2, c = a.shape[1], b.shape[1], a.shape[2]
        key = f"seq_concat_{s1}_{s2}_{c}"
        if key not in self._cached_fns:
            dtype = self.transformer.config.dtype

            def _concat_fn(x: TensorValue, y: TensorValue) -> TensorValue:
                return ops.concat([x, y], axis=1)

            self._cached_fns[key] = max_compile(
                _concat_fn,
                input_types=[
                    TensorType(dtype, shape=["batch", s1, c], device=device),
                    TensorType(dtype, shape=["batch", s2, c], device=device),
                ],
            )
        return self._cached_fns[key](a, b)

    # ── decode ────────────────────────────────────────────────────────────

    def _get_reshape_fn(self, h_latent: int, w_latent: int) -> CompileWrapper:
        """Get or create a compiled reshape function for (B,S,C) -> (B,H,W,C)."""
        key = f"reshape_{h_latent}_{w_latent}"
        if key not in self._cached_fns:
            dtype = self.transformer.config.dtype
            device = self.transformer.devices[0]
            packed_channels = self.transformer.config.in_channels
            seq_len = h_latent * w_latent

            def _reshape_fn(x: TensorValue) -> TensorValue:
                return ops.reshape(
                    x, (x.shape[0], h_latent, w_latent, packed_channels)
                )

            self._cached_fns[key] = max_compile(
                _reshape_fn,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch", seq_len, packed_channels],
                        device=device,
                    )
                ],
            )
        return self._cached_fns[key]

    def decode_latents(
        self,
        latents: Buffer,
        height: int,
        width: int,
        output_type: Literal["np", "latent"] = "np",
    ) -> np.ndarray | Buffer:
        """Decode packed latents into an image array."""
        if output_type == "latent":
            return latents

        h_latent = height // (self.vae_scale_factor * 2)
        w_latent = width // (self.vae_scale_factor * 2)

        latents_mean = self.vae.latents_mean_tensor
        latents_std = self.vae.latents_std_tensor
        if latents_mean is None or latents_std is None:
            raise ValueError("VAE latents_mean/latents_std not loaded.")

        latents_bhwc = self._get_reshape_fn(h_latent, w_latent)(latents)
        latents_decoded = self._postprocess_latents(
            latents_bhwc, latents_mean, latents_std
        )
        decoded = self.vae.decode(latents_decoded)
        return self._image_to_flat_hwc(self._to_numpy(decoded))

    def _to_numpy(self, image: Any) -> np.ndarray:
        cpu_image = image.to(CPU())
        try:
            return np.from_dlpack(cpu_image).astype(np.float32)
        except (RuntimeError, TypeError):
            from max.experimental.tensor import Tensor as _Tensor

            if isinstance(cpu_image, _Tensor):
                return np.from_dlpack(cpu_image.cast(DType.float32)).astype(
                    np.float32
                )
            return np.from_dlpack(
                _Tensor(storage=cpu_image).cast(DType.float32)
            ).astype(np.float32)

    @staticmethod
    def _image_to_flat_hwc(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        while img.ndim > 3:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img.astype(np.float32, copy=False)

    # ── main execute ──────────────────────────────────────────────────────

    @traced
    def execute(  # type: ignore[override]
        self,
        model_inputs: QwenImageEditModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent"] = "np",
    ) -> QwenImageEditPipelineOutput:
        """Run the QwenImageEdit denoising loop and decode outputs."""
        device = self.transformer.devices[0]
        prompt_images, vae_images = self._resolve_condition_images(model_inputs)
        has_images = bool(prompt_images)
        prompt_encoder = self._get_prompt_encoder() if has_images else None

        prompt_embeds = self._encode_prompt(
            tokens=model_inputs.tokens,
            prompt_images=prompt_images,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
            prompt_encoder=prompt_encoder,
        )
        batch_size = prompt_embeds.shape[0]

        do_true_cfg = model_inputs.true_cfg_scale > 1.0
        negative_prompt_embeds = self._prepare_negative_prompt_embeddings(
            model_inputs=model_inputs,
            prompt_images=prompt_images,
            prompt_encoder=prompt_encoder,
        )

        latents, latent_image_ids = self.preprocess_latents(
            model_inputs.latents, model_inputs.latent_image_ids
        )

        image_latents, image_latent_ids = self._prepare_condition_latents(
            vae_images=vae_images,
            batch_size=batch_size,
            device=device,
        )

        h_latent = model_inputs.height // (self.vae_scale_factor * 2)
        w_latent = model_inputs.width // (self.vae_scale_factor * 2)
        max_vid_index = max(h_latent // 2, w_latent // 2)

        text_ids = self._prepare_text_ids_for_embeddings(
            embeddings=prompt_embeds,
            batch_size=batch_size,
            device=device,
            max_vid_index=max_vid_index,
        )

        negative_text_ids: Buffer | None = None
        if do_true_cfg and negative_prompt_embeds is not None:
            negative_text_ids = self._prepare_text_ids_for_embeddings(
                embeddings=negative_prompt_embeds,
                batch_size=batch_size,
                device=device,
                max_vid_index=max_vid_index,
            )

        num_inference_steps = model_inputs.num_inference_steps
        sigmas_key = f"{num_inference_steps}_{latents.shape[1]}"
        if sigmas_key not in self._cached_sigmas:
            self._cached_sigmas[sigmas_key] = Buffer.from_dlpack(
                model_inputs.sigmas
            ).to(device)
        with Tracer("prepare_scheduler"):
            all_timesteps, all_dts = self.prepare_scheduler(
                self._cached_sigmas[sigmas_key]
            )

        num_noise_tokens = latents.shape[1]
        cfg_scale_buf: Buffer | None = None
        if do_true_cfg:
            cfg_scale_buf = Buffer.from_dlpack(
                np.array([model_inputs.true_cfg_scale], dtype=np.float32)
            ).to(device)

        nnt = num_noise_tokens if image_latents is not None else None

        with Tracer("denoising_loop"):
            logger.debug("Starting denoising loop (%d steps)", num_inference_steps)
            for i in range(num_inference_steps):
                logger.debug("Denoising step %d/%d", i + 1, num_inference_steps)
                timestep = all_timesteps[i : i + 1]
                dt = all_dts[i : i + 1]

                if image_latents is not None and image_latent_ids is not None:
                    latents_in, ids_in = self.concat_image_latents(
                        latents, image_latents, latent_image_ids, image_latent_ids
                    )
                else:
                    latents_in, ids_in = latents, latent_image_ids

                with Tracer("transformer_pos"):
                    noise_pred = self.transformer(
                        latents_in, prompt_embeds, timestep, ids_in, text_ids,
                        num_noise_tokens=nnt,
                    )[0]

                if (
                    do_true_cfg
                    and negative_prompt_embeds is not None
                    and negative_text_ids is not None
                    and cfg_scale_buf is not None
                ):
                    with Tracer("transformer_neg"):
                        noise_pred_uncond = self.transformer(
                            latents_in,
                            negative_prompt_embeds,
                            timestep,
                            ids_in,
                            negative_text_ids,
                            num_noise_tokens=nnt,
                        )[0]
                    with Tracer("cfg_blend"):
                        noise_pred = self._cfg_blend(
                            noise_pred, noise_pred_uncond, cfg_scale_buf
                        )

                with Tracer("scheduler_step"):
                    latents = self.scheduler_step(
                        latents, noise_pred, dt, num_noise_tokens
                    )

            if callback_queue is not None:
                callback_queue.put_nowait(
                    cast(
                        np.ndarray,
                        self.decode_latents(
                            latents,
                            model_inputs.height,
                            model_inputs.width,
                            output_type,
                        ),
                    )
                )

        image_list = []
        if batch_size == 1:
            image_list.append(
                self.decode_latents(
                    latents,
                    model_inputs.height,
                    model_inputs.width,
                    output_type,
                )
            )
        else:
            lat_np = self._to_numpy(latents)
            for b in range(batch_size):
                latents_b = Buffer.from_dlpack(
                    np.ascontiguousarray(lat_np[b : b + 1])
                ).to(device)
                image_list.append(
                    self.decode_latents(
                        latents_b,
                        model_inputs.height,
                        model_inputs.width,
                        output_type,
                    )
                )

        return QwenImageEditPipelineOutput(images=image_list)
