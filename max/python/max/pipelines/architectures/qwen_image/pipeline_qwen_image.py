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

"""QwenImage diffusion pipeline.

Key differences from Flux2Pipeline:
- True CFG with two forward passes (positive + negative prompts)
- No guidance embedding (timestep only, not timestep+guidance)
- Latent normalization via latents_mean/latents_std instead of BatchNorm
- Text encoder returns last hidden state (not multiple layers)
- 3D position IDs (T, H, W) instead of 4D (T, H, W, L)
"""

import os
from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.interfaces import TokenBuffer
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import DiffusionPipeline, PixelModelInputs
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from max.profiler import Tracer, traced
from tqdm import tqdm

from ..autoencoders.autoencoder_kl_qwen_image import AutoencoderKLQwenImageModel
from .model import QwenImageTransformerModel
from ..qwen2_5vl.text_encoder import Qwen25VLTextEncoderModel


@dataclass(kw_only=True)
class QwenImageModelInputs(PixelModelInputs):
    """QwenImage-specific PixelModelInputs.

    Defaults:
    - width: 1024
    - height: 1024
    - guidance_scale: 4.0 (maps to true_cfg_scale)
    - num_inference_steps: 50
    """

    width: int = 1024
    height: int = 1024
    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1


@dataclass
class QwenImagePipelineOutput:
    """Container for QwenImage pipeline results."""

    images: np.ndarray | Tensor


class QwenImagePipeline(DiffusionPipeline):
    """Diffusion pipeline for QwenImage text-to-image generation.

    Wires together:
    - Qwen2.5-VL text encoder
    - QwenImage transformer denoiser (60 dual-stream blocks)
    - QwenImage 3D VAE (with latents_mean/std normalization)
    """

    vae: AutoencoderKLQwenImageModel
    text_encoder: Qwen25VLTextEncoderModel
    transformer: QwenImageTransformerModel

    components = {
        "vae": AutoencoderKLQwenImageModel,
        "text_encoder": Qwen25VLTextEncoderModel,
        "transformer": QwenImageTransformerModel,
    }

    def init_remaining_components(self) -> None:
        """Initialize derived attributes that depend on loaded components."""
        # QwenImage VAE uses dim_mult [1,2,4,4] with 3 downsample stages
        # Spatial scale factor = 2^3 = 8
        self.vae_scale_factor = 8

        self.build_preprocess_latents()
        self.build_prepare_scheduler()
        self.build_scheduler_step()
        self.build_decode_latents()
        self.build_cfg_blend()

        self._cached_sigmas: dict[str, Tensor] = {}
        self._cached_text_ids: dict[str, Tensor] = {}

    def prepare_inputs(self, context: PixelContext) -> QwenImageModelInputs:  # type: ignore[override]
        """Convert a PixelContext into QwenImageModelInputs."""
        return QwenImageModelInputs.from_context(context)

    def build_preprocess_latents(self) -> None:
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                DType.float32,
                shape=["batch", "channels", "height", 2, "width", 2],
                device=device,
            ),
        ]
        self.__dict__["_patchify_and_pack"] = max_compile(
            self._patchify_and_pack,
            input_types=input_types,
        )

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
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                dtype, shape=["batch", "seq", "channels"], device=device
            ),
            TensorType(
                dtype, shape=["batch", "pred_seq", "channels"], device=device
            ),
            TensorType(DType.float32, shape=[1], device=device),
            TensorType(DType.int64, shape=[], device=DeviceRef.CPU()),
        ]
        self.__dict__["scheduler_step"] = max_compile(
            self.scheduler_step,
            input_types=input_types,
        )

    def build_decode_latents(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        z_dim = 16  # VAE latent channels
        packed_channels = self.transformer.config.in_channels  # 64 = z_dim * patch_size^2

        input_types = [
            TensorType(
                dtype,
                shape=["batch", "height", "width", packed_channels],
                device=device,
            ),
            TensorType(dtype, shape=[z_dim], device=device),
            TensorType(dtype, shape=[z_dim], device=device),
        ]

        self.__dict__["_postprocess_latents"] = max_compile(
            self._postprocess_latents,
            input_types=input_types,
        )

    def build_cfg_blend(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        input_types = [
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
            TensorType(DType.float32, shape=[1], device=device),
        ]
        self.__dict__["_cfg_blend"] = max_compile(
            self._cfg_blend,
            input_types=input_types,
        )

    def _cfg_blend(
        self, cond_pred: Tensor, uncond_pred: Tensor, cfg_scale: Tensor
    ) -> Tensor:
        scale = cfg_scale.cast(cond_pred.dtype)
        return uncond_pred + scale * (cond_pred - uncond_pred)

    # Number of chat template prefix tokens to drop from encoder output.
    # Matches diffusers' prompt_template_encode_start_idx = 34.
    PROMPT_TEMPLATE_DROP_IDX = 34

    def prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        num_images_per_prompt: int = 1,
    ) -> Tensor:
        """Create prompt embeddings from tokens.

        QwenImage uses the last hidden state from the text encoder (layer -1).
        The tokens include a chat template prefix (~34 tokens) that must be
        dropped from the encoder output to match diffusers' behavior.

        Returns:
            prompt_embeds: Tensor of shape (B, S, hidden_dim)
        """
        text_input_ids = Tensor(
            storage=Buffer.from_dlpack(tokens.array).to(
                self.text_encoder.devices[0]
            )
        )
        # Text encoder expects 1D input [total_seq_len]
        seq_len = text_input_ids.shape[-1]
        if text_input_ids.rank == 2:
            text_input_ids = F.reshape(text_input_ids, (-1,))
        hidden_states_all = self.text_encoder(text_input_ids)
        # Use last hidden state
        prompt_embeds = hidden_states_all[-1]

        # Add batch dimension if needed
        if prompt_embeds.rank == 2:
            prompt_embeds = F.unsqueeze(prompt_embeds, 0)

        # Drop chat template prefix tokens (first 34 tokens)
        drop_idx = self.PROMPT_TEMPLATE_DROP_IDX
        prompt_embeds = prompt_embeds[:, drop_idx:, :]

        batch_size = int(prompt_embeds.shape[0])
        seq_len = int(prompt_embeds.shape[1])

        if num_images_per_prompt != 1:
            prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
            prompt_embeds = F.reshape(
                prompt_embeds,
                [batch_size * num_images_per_prompt, seq_len, -1],
            )

        return prompt_embeds

    @staticmethod
    def _prepare_text_ids(
        batch_size: int,
        seq_len: int,
        device: Device,
        max_vid_index: int = 0,
    ) -> Tensor:
        """Create 3D text position IDs in (T, H, W) format.

        QwenImage text tokens use positions [max_vid_index, max_vid_index+1, ...]
        for all 3 axes (matching diffusers scale_rope=True convention).
        """
        tok_positions = np.arange(seq_len, dtype=np.int64) + max_vid_index
        coords = np.stack(
            [tok_positions, tok_positions, tok_positions], axis=-1
        )
        text_ids = np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
        return Tensor(storage=Buffer.from_dlpack(text_ids).to(device))

    @staticmethod
    def _prepare_image_ids(
        batch_size: int,
        height: int,
        width: int,
        device: Device,
    ) -> Tensor:
        """Create 3D image position IDs in (T, H, W) format.

        For image tokens: T=0, H=row_index, W=col_index.
        """
        t_coords = np.zeros((height, width), dtype=np.int64)
        h_coords, w_coords = np.meshgrid(
            np.arange(height, dtype=np.int64),
            np.arange(width, dtype=np.int64),
            indexing="ij",
        )
        coords = np.stack([t_coords, h_coords, w_coords], axis=-1)
        coords = coords.reshape(-1, 3)
        image_ids = np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
        return Tensor(storage=Buffer.from_dlpack(image_ids).to(device))

    def decode_latents(
        self,
        latents: Tensor,
        height: int,
        width: int,
        output_type: Literal["np", "latent"] = "np",
    ) -> np.ndarray | Tensor:
        """Decode packed latents into an image array."""
        if output_type == "latent":
            return latents[0] if int(latents.shape[0]) > 1 else latents

        # Unpack: (B, S, C) -> (B, H, W, C)
        h_latent = height // (self.vae_scale_factor * 2)
        w_latent = width // (self.vae_scale_factor * 2)

        batch = int(latents.shape[0])
        c = int(latents.shape[2])
        latents_bhwc = F.reshape(latents, (batch, h_latent, w_latent, c))

        latents_mean = self.vae.latents_mean_tensor
        latents_std = self.vae.latents_std_tensor
        if latents_mean is None or latents_std is None:
            raise ValueError(
                "VAE latents_mean/latents_std not loaded."
            )

        latents_decoded = self._postprocess_latents(
            latents_bhwc, latents_mean, latents_std
        )

        decoded = self.vae.decode(latents_decoded)
        return self._image_to_flat_hwc(self._to_numpy(decoded))

    def _postprocess_latents(
        self,
        latents_bhwc: Tensor,
        latents_mean: Tensor,
        latents_std: Tensor,
    ) -> Tensor:
        """Unpatchify and denormalize latents for VAE decoding."""
        batch = latents_bhwc.shape[0]
        h = latents_bhwc.shape[1]
        w = latents_bhwc.shape[2]
        c = latents_bhwc.shape[3]
        z_dim = c // 4  # 16

        # Permute (B, H, W, C) -> (B, C, H, W)
        latents = F.permute(latents_bhwc, (0, 3, 1, 2))

        # Unpatchify first: (B, C, H, W) -> (B, z_dim, H*2, W*2)
        latents = F.reshape(latents, (batch, z_dim, 2, 2, h, w))
        latents = F.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = F.reshape(latents, (batch, z_dim, h * 2, w * 2))

        # Then denormalize using latents_mean/std (shape [z_dim])
        mean_r = F.reshape(latents_mean, (1, z_dim, 1, 1))
        std_r = F.reshape(latents_std, (1, z_dim, 1, 1))
        latents = latents * std_r + mean_r

        return latents

    def _to_numpy(self, image: Tensor) -> np.ndarray:
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)

    @staticmethod
    def _image_to_flat_hwc(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        while img.ndim > 3:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img.astype(np.float32, copy=False)

    def preprocess_latents(
        self,
        latents: npt.NDArray[np.float32],
        latent_image_ids: npt.NDArray[np.float32],
    ) -> tuple[Tensor, Tensor]:
        latents_tensor = Tensor(
            storage=Buffer.from_dlpack(latents).to(self.transformer.devices[0])
        )
        batch = latents_tensor.shape[0]
        c = latents_tensor.shape[1]
        h = latents_tensor.shape[2]
        w = latents_tensor.shape[3]
        latents_tensor = F.reshape(
            latents_tensor, (batch, c, h // 2, 2, w // 2, 2)
        )
        latents_tensor = self._patchify_and_pack(latents_tensor)

        latent_image_ids_int64 = np.asarray(latent_image_ids, dtype=np.int64)
        latent_image_ids_tensor = Tensor(
            storage=Buffer.from_dlpack(latent_image_ids_int64).to(
                self.transformer.devices[0]
            )
        )
        return latents_tensor, latent_image_ids_tensor

    def _patchify_and_pack(self, latents: Tensor) -> Tensor:
        """Patchify (B,C,H,W)->(B,C*4,H//2,W//2) then pack to (B,H//2*W//2,C*4)."""
        latents = latents.cast(self.transformer.config.dtype)
        batch = latents.shape[0]
        c = latents.shape[1]
        h2 = latents.shape[2]
        w2 = latents.shape[4]

        latents = F.permute(latents, (0, 1, 3, 5, 2, 4))
        latents = F.reshape(latents, (batch, c * 4, h2, w2))

        c4 = c * 4
        latents = F.reshape(latents, (batch, c4, h2 * w2))
        latents = F.permute(latents, (0, 2, 1))

        return latents

    def scheduler_step(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        dt: Tensor,
        num_noise_tokens: int,
    ) -> Tensor:
        """Apply a single Euler update step."""
        latents_sliced = F.slice_tensor(
            latents,
            [
                slice(None),
                (slice(0, num_noise_tokens), "num_tokens"),
                slice(None),
            ],
        )
        noise_pred_sliced = F.slice_tensor(
            noise_pred,
            [
                slice(None),
                (slice(0, num_noise_tokens), "num_tokens"),
                slice(None),
            ],
        )
        latents_dtype = latents_sliced.dtype
        latents_sliced = latents_sliced.cast(DType.float32)
        latents_sliced = latents_sliced + dt * noise_pred_sliced
        return latents_sliced.cast(latents_dtype)

    def prepare_scheduler(self, sigmas: Tensor) -> tuple[Tensor, Tensor]:
        """Precompute timesteps and dt values from sigmas."""
        sigmas_curr = F.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = F.slice_tensor(sigmas, [slice(1, None)])
        all_dt = F.sub(sigmas_next, sigmas_curr)
        all_timesteps = sigmas_curr.cast(self.transformer.config.dtype)
        return all_timesteps, all_dt

    @traced
    def execute(  # type: ignore[override]
        self,
        model_inputs: QwenImageModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent"] = "np",
    ) -> QwenImagePipelineOutput:
        """Run the QwenImage denoising loop and decode outputs.

        Supports true classifier-free guidance with separate positive and
        negative prompt forward passes.
        """
        # 1) Encode positive prompt
        prompt_embeds = self.prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )
        batch_size = int(prompt_embeds.shape[0])
        dtype = prompt_embeds.dtype
        device = self.transformer.devices[0]

        # 2) Determine if we should do true CFG
        do_true_cfg = (
            model_inputs.true_cfg_scale > 1.0
            and model_inputs.negative_tokens is not None
        )

        # Encode negative prompt if doing true CFG
        negative_prompt_embeds: Tensor | None = None
        if do_true_cfg and model_inputs.negative_tokens is not None:
            negative_prompt_embeds = self.prepare_prompt_embeddings(
                tokens=model_inputs.negative_tokens,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )

        # 3) Prepare latents and conditioning
        latents, latent_image_ids = self.preprocess_latents(
            model_inputs.latents, model_inputs.latent_image_ids
        )

        # Compute max_vid_index for text RoPE offset (scale_rope=True)
        h_latent = model_inputs.height // (self.vae_scale_factor * 2)
        w_latent = model_inputs.width // (self.vae_scale_factor * 2)
        max_vid_index = max(h_latent // 2, w_latent // 2)

        # Prepare text IDs with RoPE offset
        text_seq_len = int(prompt_embeds.shape[1])
        text_ids_key = f"{batch_size}_{text_seq_len}_{max_vid_index}"
        if text_ids_key in self._cached_text_ids:
            text_ids = self._cached_text_ids[text_ids_key]
        else:
            text_ids = self._prepare_text_ids(
                batch_size=batch_size,
                seq_len=text_seq_len,
                device=device,
                max_vid_index=max_vid_index,
            )
            self._cached_text_ids[text_ids_key] = text_ids

        # Negative text IDs (may differ in seq_len)
        negative_text_ids: Tensor | None = None
        if do_true_cfg and negative_prompt_embeds is not None:
            neg_text_seq_len = int(negative_prompt_embeds.shape[1])
            neg_text_ids_key = (
                f"{batch_size}_{neg_text_seq_len}_{max_vid_index}"
            )
            if neg_text_ids_key in self._cached_text_ids:
                negative_text_ids = self._cached_text_ids[neg_text_ids_key]
            else:
                negative_text_ids = self._prepare_text_ids(
                    batch_size=batch_size,
                    seq_len=neg_text_seq_len,
                    device=device,
                    max_vid_index=max_vid_index,
                )
                self._cached_text_ids[neg_text_ids_key] = negative_text_ids

        # 4) Prepare scheduler
        num_inference_steps = model_inputs.num_inference_steps
        image_seq_len = int(latents.shape[1])
        sigmas_key = f"{num_inference_steps}_{image_seq_len}"
        if sigmas_key in self._cached_sigmas:
            sigmas = self._cached_sigmas[sigmas_key]
        else:
            sigmas = Tensor(
                storage=Buffer.from_dlpack(model_inputs.sigmas).to(device)
            )
            self._cached_sigmas[sigmas_key] = sigmas
        with Tracer("prepare_scheduler"):
            all_timesteps, all_dts = self.prepare_scheduler(sigmas)

            # For faster tensor slicing inside the denoising loop.
            timesteps_seq: Any = all_timesteps
            dts_seq: Any = all_dts
            if hasattr(timesteps_seq, "driver_tensor"):
                timesteps_seq = timesteps_seq.driver_tensor
            if hasattr(dts_seq, "driver_tensor"):
                dts_seq = dts_seq.driver_tensor

        # 5) Denoising loop
        num_noise_tokens = int(latents.shape[1])

        cfg_scale_tensor: Tensor | None = None
        if do_true_cfg:
            cfg_scale_tensor = Tensor.full(
                [1],
                model_inputs.true_cfg_scale,
                device=device,
                dtype=DType.float32,
            )

        with Tracer("denoising_loop"):
            for i in tqdm(range(num_inference_steps), desc="Denoising"):
                timestep = timesteps_seq[i : i + 1]
                dt = dts_seq[i : i + 1]

                with Tracer("transformer_pos"):
                    noise_pred = self.transformer(
                        latents,
                        prompt_embeds,
                        timestep,
                        latent_image_ids,
                        text_ids,
                    )[0]

                # True CFG: second forward pass with negative prompt
                if (
                    do_true_cfg
                    and negative_prompt_embeds is not None
                    and negative_text_ids is not None
                    and cfg_scale_tensor is not None
                ):
                    with Tracer("transformer_neg"):
                        noise_pred_uncond = self.transformer(
                            latents,
                            negative_prompt_embeds,
                            timestep,
                            latent_image_ids,
                            negative_text_ids,
                        )[0]

                    with Tracer("cfg_blend"):
                        noise_pred = self._cfg_blend(
                            noise_pred,
                            noise_pred_uncond,
                            cfg_scale_tensor,
                        )

                with Tracer("scheduler_step"):
                    latents = self.scheduler_step(
                        latents, noise_pred, dt, num_noise_tokens
                    )

            # Debug: save per-step latents when env var is set
            _step_dir = os.environ.get("QWEN_SAVE_STEP_DIR")
            if _step_dir:
                _lat = np.from_dlpack(
                    latents.cast(DType.float32).to(CPU())
                )
                np.save(
                    os.path.join(_step_dir, f"step_{i}_latents.npy"), _lat
                )
                if i in {0, 1}:
                    _np_pred = np.from_dlpack(
                        noise_pred.cast(DType.float32).to(CPU())
                    )
                    np.save(
                        os.path.join(
                            _step_dir, f"step_{i}_noise_pred.npy"
                        ),
                        _np_pred,
                    )

            if callback_queue is not None:
                callback_queue.put_nowait(
                    cast(
                        np.ndarray,
                        self.decode_latents(
                            latents,
                            model_inputs.height,
                            model_inputs.width,
                            output_type=output_type,
                        ),
                    )
                )

        # 6) Decode final outputs
        image_list = []
        for b in range(batch_size):
            latents_b = latents[b : b + 1]
            image_list.append(
                self.decode_latents(
                    latents_b,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                )
            )

        return QwenImagePipelineOutput(images=image_list)  # type: ignore[arg-type]
