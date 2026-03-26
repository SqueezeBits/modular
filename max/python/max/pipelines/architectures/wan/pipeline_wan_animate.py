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

"""Wan-Animate pipeline for motion transfer and character replacement.

Extends WanI2VPipeline with:
- Pose conditioning via Conv3d injection
- Face motion encoding (StyleGAN2 bridge) + face encoder (MAX Graph)
- CLIP image conditioning (dual-path cross-attention)
- Multi-segment processing with temporal overlap
- Replace mode (background preservation via mask)
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import Model
from max.graph import Graph, TensorType, ops
from max.graph.weights import load_weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.interfaces import PixelModelInputs
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import Tracer, traced
from tqdm.auto import tqdm

from ..autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWanModel,
    _buffer_to_numpy_f32,
    _numpy_f32_to_buffer,
)
from ..clip import ClipVisionModel
from ..umt5 import UMT5Model
from .model import WanTransformerModel
from .pipeline_wan import WanCompiled, WanModelInputs, WanPipeline, WanPipelineOutput
from .pipeline_wan_i2v import WanI2VPipeline
from .wan_animate_model import WanAnimateTransformerModel

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class WanAnimateModelInputs(WanModelInputs):
    """Extended model inputs for Wan-Animate pipeline."""

    # Preprocessed pose video frames: list of PIL images or [T, H, W, 3] uint8
    pose_video: Any = None
    # Preprocessed face video frames: list of PIL images or [T, 3, 512, 512]
    face_video: Any = None
    # Input image (reference character)
    input_image: Any = None
    # Mode: "animate" or "replace"
    mode: str = "animate"
    # Replace mode inputs
    background_video: Any = None
    mask_video: Any = None
    # Segment parameters
    segment_frame_length: int = 77
    prev_segment_conditioning_frames: int = 1
    # Motion encoder batch size
    motion_encode_batch_size: int = 8
    # Shared inputs dir for parity testing (dumped from diffusers)
    shared_inputs_dir: str | None = None


class WanAnimatePipeline(WanI2VPipeline):
    """Wan-Animate pipeline — motion transfer with pose/face conditioning.

    Extends WanI2VPipeline with CLIP image encoding, motion/face encoding,
    and multi-segment processing.
    """

    transformer: WanAnimateTransformerModel  # type: ignore[assignment]

    components = {
        "vae": AutoencoderKLWanModel,
        "text_encoder": UMT5Model,
        "transformer": WanAnimateTransformerModel,
        "image_encoder": ClipVisionModel,
    }

    def _load_sub_models(
        self,
        weight_paths: list[Path],
    ) -> dict[str, ComponentModel]:
        """Load sub-models including animate transformer."""
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
            logger.info("Loading Wan-Animate component: %s", name)
            config_dict = self._get_component_config_dict(components_cfg, name)
            if name in relative_paths:
                abs_paths = self._resolve_absolute_paths(
                    weight_paths, relative_paths[name]
                )
            else:
                abs_paths = self._download_component_weights(name)
            component_kwargs: dict[str, Any] = {
                "config": config_dict,
                "encoding": self.pipeline_config.model.quantization_encoding,
                "devices": self.devices,
                "weights": load_weights(abs_paths),
            }
            if session is not None:
                component_kwargs["session"] = session
            if component_cls is WanAnimateTransformerModel:
                component_kwargs["eager_load"] = eager_load
            if component_cls is AutoencoderKLWanModel:
                component_kwargs["eager_load"] = eager_load
            with Tracer(f"load_component:{name}"):
                component = component_cls(**component_kwargs)
            logger.info("Loaded Wan-Animate component: %s", name)
            return component

        models: dict[str, ComponentModel] = {}
        for name, component_cls in self.components.items():
            if not issubclass(component_cls, ComponentModel):
                continue
            component_session: object | None = None
            component_eager_load = True
            if component_cls is UMT5Model:
                component_session = self.session
            elif component_cls is WanAnimateTransformerModel:
                component_session = self.session
            elif component_cls is AutoencoderKLWanModel:
                component_eager_load = False
            models[name] = _load_component(
                name,
                component_cls,
                session=component_session,
                eager_load=component_eager_load,
            )
        return models

    def init_remaining_components(self) -> None:
        """Initialize animate-specific runtime components."""
        # WanPipeline expects transformer_2 for MoE — animate doesn't use it.
        self.transformer_2 = None
        super().init_remaining_components()

        # image_encoder is set as an attribute by _load_sub_models via setattr

    @traced(message="WanAnimatePipeline.execute")
    def execute(  # type: ignore[override]
        self,
        model_inputs: WanAnimateModelInputs,
        **kwargs: object,
    ) -> WanPipelineOutput:
        import time as _time

        del kwargs
        device = self.transformer.devices[0]

        t_start = _time.perf_counter()

        # Set up intermediate tensor dumping for parity testing.
        dump_dir = model_inputs.shared_inputs_dir
        if dump_dir:
            max_dump_dir = os.path.join(
                os.path.dirname(dump_dir), "max_dump"
            )
            os.makedirs(max_dump_dir, exist_ok=True)
            logger.info("Dumping MAX intermediates to %s", max_dump_dir)
        else:
            max_dump_dir = None

        def _dump(name: str, arr: Any) -> None:
            if max_dump_dir is None or arr is None:
                return
            if isinstance(arr, Buffer):
                arr = _buffer_to_numpy_f32(arr)
            elif not isinstance(arr, np.ndarray):
                arr = np.asarray(arr, dtype=np.float32)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            path = os.path.join(max_dump_dir, f"{name}.npy")
            np.save(path, arr)
            logger.info("  [dump] %s: %s -> %s", name, arr.shape, path)

        # === One-time setup ===

        h = int(model_inputs.height)
        w = int(model_inputs.width)
        mode = model_inputs.mode
        segment_len = model_inputs.segment_frame_length
        prev_cond_frames = model_inputs.prev_segment_conditioning_frames

        # Pre-compile VAE encoder for this resolution
        self.vae.prewarm_encoder(h, w)

        # 1. Encode text (UMT5)
        with Tracer("prepare_prompt_embeddings"):
            (
                prompt_embeds,
                negative_prompt_embeds,
                batched_prompt_embeds,
                do_cfg,
            ) = self._prepare_prompt_state(model_inputs)

        # 2. Encode reference image via MAX-native CLIP
        with Tracer("clip_encode"):
            clip_features = self.image_encoder.encode(
                model_inputs.input_image
            )

        _dump("prompt_embeds", prompt_embeds)
        _dump("clip_features", clip_features)

        # 3. Prepare latents for I2V condition (will be used per segment)
        with Tracer("prepare_i2v_base"):
            pass  # Condition built per-segment below

        # 4. Prepare pose and face frames
        pose_frames = self._load_video_frames(model_inputs.pose_video)
        face_frames = self._load_video_frames(model_inputs.face_video)
        num_pose_frames = len(pose_frames)

        # Compute segments matching diffusers:
        # effective_seg_len = segment_len - prev_cond_frames (76 for default)
        # Segments overlap by prev_cond_frames: seg0=[0:77], seg1=[76:153], etc.
        effective_seg_len = segment_len - prev_cond_frames
        last_seg_frames = (
            num_pose_frames - prev_cond_frames
        ) % effective_seg_len
        num_padding = (
            0 if last_seg_frames == 0
            else effective_seg_len - last_seg_frames
        )
        num_target_frames = num_pose_frames + num_padding
        num_segments = num_target_frames // effective_seg_len

        # Pad pose/face frames to fill complete segments using
        # reflect-style padding matching diffusers' pad_video_frames.
        def _reflect_pad(
            frames: list[Any], target: int
        ) -> list[Any]:
            if len(frames) >= target:
                return frames[:target]
            idx = 0
            flip = False
            result: list[Any] = []
            while len(result) < target:
                result.append(frames[idx])
                if flip:
                    idx -= 1
                else:
                    idx += 1
                if idx == 0 or idx == len(frames) - 1:
                    flip = not flip
            return result

        total_needed = num_target_frames
        if len(pose_frames) < total_needed:
            pose_frames = _reflect_pad(pose_frames, total_needed)
        if len(face_frames) < total_needed:
            face_frames = _reflect_pad(face_frames, total_needed)

        # Background/mask for replace mode
        bg_frames = None
        mask_frames = None
        if mode == "replace":
            if model_inputs.background_video is not None:
                bg_frames = self._load_video_frames(
                    model_inputs.background_video
                )
            if model_inputs.mask_video is not None:
                mask_frames = self._load_video_frames(model_inputs.mask_video)

        logger.info(
            "Animate: %d pose frames, %d segments (segment_len=%d, prev_cond=%d), mode=%s",
            num_pose_frames, num_segments, segment_len, prev_cond_frames, mode,
        )

        t_prep = _time.perf_counter()

        # === Segment loop ===
        all_out_frames: list[np.ndarray] = []
        prev_segment_cond_video: np.ndarray | None = None

        seg_start = 0
        seg_end = segment_len
        for seg_idx in range(num_segments):
            # Actual pose/face slice for this segment
            seg_pose = pose_frames[seg_start:seg_end]
            seg_face = face_frames[seg_start:seg_end]
            num_seg_frames = len(seg_pose)

            logger.info(
                "Segment %d/%d: frames %d-%d (%d frames)",
                seg_idx + 1, num_segments, seg_start,
                seg_start + num_seg_frames - 1, num_seg_frames,
            )

            # 5. Encode pose segment via VAE
            with Tracer(f"segment_{seg_idx}:vae_encode_pose"):
                pose_latents = self._encode_pose_segment(
                    seg_pose, h, w, device
                )
                logger.info(
                    "Pose latents shape: %s (from %d pixel frames)",
                    tuple(int(d) for d in pose_latents.shape),
                    num_seg_frames,
                )

            _dump(f"pose_latents_seg{seg_idx}", pose_latents)

            # 6. Encode face segment via motion encoder + face encoder
            with Tracer(f"segment_{seg_idx}:encode_face"):
                face_emb, motion_vectors = self._encode_face_segment(
                    seg_face,
                    model_inputs.motion_encode_batch_size,
                    device,
                )

            _dump(f"motion_vectors_seg{seg_idx}", motion_vectors)
            _dump(f"face_emb_seg{seg_idx}", face_emb)

            # 7. Build I2V conditioning (reuse proven I2V method)
            with Tracer(f"segment_{seg_idx}:build_condition"):
                vae_t = self.vae_scale_factor_temporal
                h_l = h // self.vae_scale_factor_spatial
                w_l = w // self.vae_scale_factor_spatial
                z_dim = self.vae.config.z_dim

                # Create segment-level model inputs with correct num_frames
                seg_model_inputs = WanAnimateModelInputs(
                    tokens=model_inputs.tokens,
                    latents=model_inputs.latents,
                    timesteps=model_inputs.timesteps,
                    step_coefficients=model_inputs.step_coefficients,
                    width=model_inputs.width,
                    height=model_inputs.height,
                    num_frames=num_seg_frames,
                    num_inference_steps=model_inputs.num_inference_steps,
                    guidance_scale=model_inputs.guidance_scale,
                    input_image=model_inputs.input_image,
                )

                # Get pose temporal latent dim (T_l) from pose_latents
                t_l = int(pose_latents.shape[2])  # e.g., 20 for 77 frames
                total_latent_t = 1 + t_l  # ref(1) + segment(T_l) = 21

                # Build ref condition: single ref frame VAE encode
                ref_image = model_inputs.input_image
                if not isinstance(ref_image, np.ndarray):
                    ref_image = np.array(ref_image)
                ref_f32 = ref_image.astype(np.float32) / 127.5 - 1.0
                if ref_f32.ndim == 3:
                    ref_f32 = ref_f32.transpose(2, 0, 1)
                if ref_f32.shape[1] != h or ref_f32.shape[2] != w:
                    import PIL.Image
                    pil = PIL.Image.fromarray(
                        ((ref_f32.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    )
                    pil = pil.resize((w, h), PIL.Image.Resampling.LANCZOS)
                    ref_f32 = (np.array(pil).astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)

                # Encode ref as single-frame video [1, 3, 1, H, W]
                ref_video = ref_f32[np.newaxis, :, np.newaxis, :, :]
                ref_buf = _numpy_f32_to_buffer(ref_video, self.vae.config.dtype, device)
                ref_latent_np = _buffer_to_numpy_f32(self.vae.encode(ref_buf))

                # Standardize ref latents
                lat_mean = np.array(self.vae.config.latents_mean, dtype=np.float32).reshape(1, z_dim, 1, 1, 1)
                lat_inv_std = 1.0 / np.array(self.vae.config.latents_std, dtype=np.float32).reshape(1, z_dim, 1, 1, 1)
                ref_latent_std = (ref_latent_np - lat_mean) * lat_inv_std

                # Build ref condition: [mask(vae_t) | latent(z_dim)] = [20, 1, H_l, W_l]
                ref_mask = np.ones((1, vae_t, 1, h_l, w_l), dtype=np.float32)
                y_ref = np.concatenate([ref_mask, ref_latent_std], axis=1)

                # Build prev-segment condition.
                if seg_idx == 0 or prev_segment_cond_video is None:
                    # First segment: encode a zero (black) video through VAE.
                    # The VAE's bias terms produce non-zero latents for zero
                    # input, matching diffusers' behavior.
                    prev_video = np.zeros(
                        (1, 3, num_seg_frames, h, w), dtype=np.float32
                    )
                else:
                    # Subsequent segments: use last frames from decoded output.
                    # prev_segment_cond_video is [1, 3, N, H, W] in [-1, 1].
                    # Pad to full segment length with zeros.
                    prev_n = prev_segment_cond_video.shape[2]
                    remaining = num_seg_frames - prev_n
                    zero_pad = np.zeros(
                        (1, 3, remaining, h, w), dtype=np.float32
                    )
                    prev_video = np.concatenate(
                        [prev_segment_cond_video, zero_pad], axis=2
                    )

                prev_buf = _numpy_f32_to_buffer(
                    prev_video, self.vae.config.dtype, device
                )
                prev_latent_np = _buffer_to_numpy_f32(
                    self.vae.encode(prev_buf)
                )
                prev_latent_std = (prev_latent_np - lat_mean) * lat_inv_std

                # Build mask: 1s for conditioned frames, 0s for the rest
                if seg_idx == 0 or prev_segment_cond_video is None:
                    prev_mask = np.zeros(
                        (1, vae_t, t_l, h_l, w_l), dtype=np.float32
                    )
                else:
                    # Mark first prev_cond_frames latent frames as conditioned
                    prev_mask = np.zeros(
                        (1, vae_t, t_l, h_l, w_l), dtype=np.float32
                    )
                    # prev_cond_frames pixel frames → latent frames
                    cond_lat_t = (
                        prev_cond_frames - 1
                    ) // self.vae_scale_factor_temporal + 1
                    prev_mask[:, :, :cond_lat_t, :, :] = 1.0

                y_prev = np.concatenate(
                    [prev_mask, prev_latent_std], axis=1
                ).astype(np.float32)

                # Full condition: [20, 1+T_l, H_l, W_l]
                y_full = np.concatenate([y_ref, y_prev], axis=2).astype(np.float32)
                condition = _numpy_f32_to_buffer(y_full, self.vae.config.dtype, device)

                if seg_idx == 0:
                    _dump("ref_image_latents", y_ref)
                _dump(f"prev_cond_seg{seg_idx}", y_prev)

                logger.info(
                    "Condition shape: %s, noise T=%d, pose T=%d",
                    y_full.shape, total_latent_t, t_l,
                )

            # 8. Sample noise [16, 1+T_l, H_l, W_l]
            with Tracer(f"segment_{seg_idx}:prepare_latents"):
                noise_shape = (1, z_dim, total_latent_t, h_l, w_l)
                shared_dir = model_inputs.shared_inputs_dir
                seg_noise_path = (
                    os.path.join(shared_dir, f"noise_seg{seg_idx}.npy")
                    if shared_dir
                    else None
                )
                if (
                    seg_noise_path
                    and os.path.exists(seg_noise_path)
                ):
                    latents_np = np.load(seg_noise_path).astype(
                        np.float32
                    )
                    logger.info(
                        "Loaded shared noise for seg %d: %s from %s",
                        seg_idx,
                        latents_np.shape,
                        seg_noise_path,
                    )
                elif (
                    seg_idx == 0
                    and model_inputs.latents is not None
                    and model_inputs.latents.shape == noise_shape
                ):
                    latents_np = model_inputs.latents.astype(np.float32)
                    logger.info("Using provided initial noise: %s", noise_shape)
                else:
                    latents_np = np.random.randn(*noise_shape).astype(
                        np.float32
                    )
                _dump(f"noise_seg{seg_idx}", latents_np)
                latents = Buffer.from_numpy(latents_np).to(device)

            # Compute RoPE for this segment's latent dimensions.
            # compute_rope internally divides by patch_size, so pass
            # latent dims (h_l, w_l) directly.
            rope_cos, rope_sin = self.transformer.compute_rope(
                total_latent_t, h_l, w_l
            )

            # Spatial shape tensor for post-processing.
            # Shape encodes post-patch dims: ppf = T/p_t, pph = H/p_h, ppw = W/p_w
            p_t, p_h, p_w = self.transformer.config.patch_size
            ppf = total_latent_t // p_t
            pph = h_l // p_h
            ppw = w_l // p_w
            spatial_shape = Buffer.from_numpy(
                np.zeros((ppf, pph, ppw), dtype=np.int8)
            ).to(device)

            # Number of temporal frames for face adapter alignment (post-patch)
            num_temporal_frames_buf = Buffer.from_numpy(
                np.array([ppf], dtype=np.int32)
            ).to(device)

            # 9. Pre-compile concat graph for this segment
            # Reset for each segment (temporal dim may differ)
            self._i2v_concat_model = None
            if self._i2v_concat_model is None:
                latent_model_input = (
                    self.compiled.cast_f32_to_model_dtype.execute(latents)[0]
                )
                self._i2v_concat_model = self._compile_i2v_concat(
                    latent_model_input, condition
                )

            # Pre-compile VAE decoder for this segment
            lat_shape = latents.shape
            self.vae.prewarm_for_latent_shape(
                (int(lat_shape[0]), int(lat_shape[1]), int(lat_shape[2]), int(lat_shape[3]), int(lat_shape[4]))
            )

            # Prepare scheduler state for this segment
            with Tracer(f"segment_{seg_idx}:prepare_scheduler"):
                (
                    _rope_cos,
                    _rope_sin,
                    batched_timesteps,
                    coeff_buffers,
                    boundary_step_idx,
                    _spatial_shape,
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

            # 10. Denoising loop
            with Tracer(f"segment_{seg_idx}:denoising"):
                latents = self._run_animate_denoising(
                    latents=latents,
                    condition=condition,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    clip_features=clip_features,
                    pose_latents=pose_latents,
                    face_emb=face_emb,
                    num_temporal_frames=num_temporal_frames_buf,
                    do_cfg=do_cfg,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    batched_timesteps=batched_timesteps,
                    coeff_buffers=coeff_buffers,
                    spatial_shape=spatial_shape,
                    guidance_scale=guidance_scale_high,
                )

            _dump(f"final_latents_seg{seg_idx}", latents)

            # 11. VAE decode (skip first latent frame = conditioned ref image)
            with Tracer(f"segment_{seg_idx}:vae_decode"):
                decoded = self._decode_segment_latents(
                    latents, model_inputs, skip_first=True
                )

            # For segments 1+, strip first prev_cond_frames from output
            if seg_idx > 0 and prev_cond_frames > 0:
                decoded = decoded[:, :, prev_cond_frames:, :, :]

            # Save last frames for next segment conditioning
            if num_segments > 1:
                prev_segment_cond_video = self._extract_last_frames(
                    decoded, prev_cond_frames
                )

            all_out_frames.append(decoded)

            # Advance to next segment (overlapping by prev_cond_frames)
            seg_start += effective_seg_len
            seg_end += effective_seg_len

        t_denoise = _time.perf_counter()

        # === Assembly ===
        video = np.concatenate(all_out_frames, axis=2)
        # Trim to original pose video length
        video = video[:, :, :num_pose_frames, :, :]

        t_total = _time.perf_counter()
        logger.info(
            "Animate timing: prep=%.1fs, denoise=%.1fs, total=%.1fs",
            t_prep - t_start, t_denoise - t_prep, t_total - t_start,
        )

        return WanPipelineOutput(images=video)

    def _load_video_frames(self, video: Any) -> list[Any]:
        """Load video frames from various formats."""
        if video is None:
            return []
        if isinstance(video, list):
            return video
        if isinstance(video, np.ndarray):
            if video.ndim == 4:  # [T, H, W, C] or [T, C, H, W]
                return [video[i] for i in range(video.shape[0])]
            return [video]
        if isinstance(video, str) or isinstance(video, Path):
            from diffusers.utils import load_video
            return load_video(str(video))
        return list(video)

    def _prepare_animate_i2v_condition(
        self,
        model_inputs: WanAnimateModelInputs,
        num_seg_frames: int,
        device: Device,
    ) -> Buffer:
        """Prepare I2V condition for a segment, matching diffusers approach.

        Encodes a full video (ref image at frame 0, zeros elsewhere) through
        VAE at once, then builds [mask | latents] condition tensor.

        Returns:
            Buffer [B, 20, 1+T_l, H_l, W_l]
        """
        image = model_inputs.input_image
        if image is None:
            raise ValueError("Animate pipeline requires input_image")
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        h = int(model_inputs.height)
        w = int(model_inputs.width)
        num_frames = num_seg_frames

        # Normalize to [-1, 1]
        image_f32 = image.astype(np.float32) / 127.5 - 1.0
        if image_f32.ndim == 3:
            image_f32 = image_f32.transpose(2, 0, 1)  # [3, H, W]

        # Resize if needed
        if image_f32.shape[1] != h or image_f32.shape[2] != w:
            import PIL.Image

            pil_img = PIL.Image.fromarray(
                ((image_f32.transpose(1, 2, 0) + 1.0) * 127.5)
                .clip(0, 255)
                .astype(np.uint8)
            )
            pil_img = pil_img.resize((w, h), PIL.Image.Resampling.LANCZOS)
            image_f32 = (
                np.array(pil_img).astype(np.float32) / 127.5 - 1.0
            ).transpose(2, 0, 1)

        # Build full video: ref image at frame 0, zeros elsewhere
        # Shape: [1, 3, num_frames, H, W]
        video_condition = np.zeros(
            (1, 3, num_frames, h, w), dtype=np.float32
        )
        video_condition[:, :, 0:1, :, :] = image_f32[np.newaxis, :, np.newaxis, :, :]

        # VAE encode the full conditioning video
        enc_buf = _numpy_f32_to_buffer(
            video_condition, self.vae.config.dtype, device
        )
        enc_latent = self.vae.encode(enc_buf)
        latent_cond_np = _buffer_to_numpy_f32(enc_latent)

        # Standardize
        z_dim = self.vae.config.z_dim
        mean = np.array(
            self.vae.config.latents_mean, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        inv_std = 1.0 / np.array(
            self.vae.config.latents_std, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        latent_cond_np = (latent_cond_np - mean) * inv_std

        # Build mask: [B, 1, num_frames, H_l, W_l]
        t_latent = latent_cond_np.shape[2]
        h_l = latent_cond_np.shape[3]
        w_l = latent_cond_np.shape[4]

        mask = np.zeros(
            (1, 1, num_frames, h_l, w_l), dtype=np.float32
        )
        mask[:, :, 0, :, :] = 1.0  # First frame is conditioned

        # Expand mask temporally: [B, 1, num_frames, H_l, W_l] -> [B, vae_t, T_l, H_l, W_l]
        vae_t = self.vae_scale_factor_temporal
        first_mask = np.repeat(mask[:, :, 0:1, :, :], vae_t, axis=2)
        mask_expanded = np.concatenate(
            [first_mask, mask[:, :, 1:, :, :]], axis=2
        )
        mask_expanded = mask_expanded.reshape(
            1, -1, vae_t, h_l, w_l
        )
        mask_expanded = mask_expanded.transpose(0, 2, 1, 3, 4)

        # Concat: [mask(vae_t ch), latent_condition(z_dim ch)] -> [B, 20, T_l, H_l, W_l]
        condition = np.concatenate(
            [mask_expanded, latent_cond_np], axis=1
        ).astype(np.float32)

        return _numpy_f32_to_buffer(condition, self.vae.config.dtype, device)

    def _encode_pose_segment(
        self,
        pose_frames: list[Any],
        h: int,
        w: int,
        device: Device,
    ) -> Buffer:
        """Encode pose frames via VAE to get pose latents.

        Returns:
            Buffer [B, 16, T_latent, H_l, W_l] pose latents.
        """
        import PIL.Image

        num_frames = len(pose_frames)

        # Convert to [1, 3, T, H, W] float32 [-1, 1]
        frames_np = []
        for frame in pose_frames:
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            if frame.ndim == 3 and frame.shape[2] == 3:
                # [H, W, 3] -> resize if needed
                if frame.shape[0] != h or frame.shape[1] != w:
                    pil = PIL.Image.fromarray(frame.astype(np.uint8))
                    pil = pil.resize((w, h), PIL.Image.Resampling.LANCZOS)
                    frame = np.array(pil)
                frame = frame.astype(np.float32) / 127.5 - 1.0
                frame = frame.transpose(2, 0, 1)  # [3, H, W]
            frames_np.append(frame)

        # [T, 3, H, W] -> [1, 3, T, H, W]
        video_np = np.stack(frames_np, axis=0)[np.newaxis]  # [1,T,3,H,W]
        video_np = video_np.transpose(0, 2, 1, 3, 4)  # [1,3,T,H,W]

        enc_buf = _numpy_f32_to_buffer(video_np, self.vae.config.dtype, device)
        enc_latent = self.vae.encode(enc_buf)
        pose_latents_np = _buffer_to_numpy_f32(enc_latent)

        # Standardize pose latents (matching diffusers' prepare_pose_latents).
        z_dim = self.vae.config.z_dim
        mean = np.array(
            self.vae.config.latents_mean, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        inv_std = 1.0 / np.array(
            self.vae.config.latents_std, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        pose_latents_std = (pose_latents_np - mean) * inv_std
        return _numpy_f32_to_buffer(
            pose_latents_std.astype(np.float32), self.vae.config.dtype, device
        )

    def _encode_face_segment(
        self,
        face_frames: list[Any],
        batch_size: int,
        device: Device,
    ) -> tuple[Buffer, np.ndarray]:
        """Encode face frames via motion encoder + face encoder.

        Args:
            face_frames: List of face frame images.
            batch_size: Batch size for motion encoder.
            device: Target device.

        Returns:
            Tuple of (face_emb Buffer [B, T//4+1, 5, 5120],
            motion_vectors ndarray [T, 512]).
        """
        # Convert face frames to [T, 3, 512, 512] float32 in [-1, 1]
        import PIL.Image

        face_np_list = []
        for frame in face_frames:
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            if frame.ndim == 3 and frame.shape[2] == 3:
                pil = PIL.Image.fromarray(frame.astype(np.uint8))
                pil = pil.resize(
                    (512, 512), PIL.Image.Resampling.LANCZOS
                )
                frame = (
                    np.array(pil).astype(np.float32) / 127.5 - 1.0
                )
                frame = frame.transpose(2, 0, 1)  # [3, 512, 512]
            elif frame.ndim == 3 and frame.shape[0] == 3:
                if frame.max() > 1.0:
                    frame = frame.astype(np.float32) / 127.5 - 1.0
            face_np_list.append(frame)

        face_pixels = np.stack(face_np_list, axis=0)  # [T, 3, 512, 512]

        # MAX-native motion encoder
        num_frames = face_pixels.shape[0]
        all_motions: list[np.ndarray] = []
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            batch_np = face_pixels[start:end]
            batch_buf = _numpy_f32_to_buffer(
                batch_np.astype(np.float32),
                self.vae.config.dtype,
                device,
            )
            motion_buf = self.transformer.encode_motion(batch_buf)
            all_motions.append(_buffer_to_numpy_f32(motion_buf))
        motion_vectors = np.concatenate(
            all_motions, axis=0
        )  # [T, 512]

        # Face encoder (MAX Graph): [1, T, 512] -> [1, T//4+1, 5, 5120]
        motion_buf = _numpy_f32_to_buffer(
            motion_vectors[np.newaxis].astype(np.float32),
            self.vae.config.dtype,
            device,
        )
        face_emb = self.transformer.encode_face(motion_buf)
        return face_emb, motion_vectors

    def _build_segment_condition(
        self,
        ref_condition: Buffer,
        prev_segment_video: np.ndarray | None,
        seg_idx: int,
        mode: str,
        bg_frames: list[Any] | None,
        mask_frames: list[Any] | None,
        seg_start: int,
        num_seg_frames: int,
        latent_shape: tuple[int, ...],
        device: Device,
    ) -> Buffer:
        """Build the full conditioning tensor for a segment.

        Returns:
            Buffer [B, 20, 1+T_l, H_l, W_l]
        """
        z_dim = int(latent_shape[1])
        t_l = int(latent_shape[2])
        h_l = int(latent_shape[3])
        w_l = int(latent_shape[4])
        vae_t = self.vae_scale_factor_temporal

        if seg_idx == 0 and mode == "animate":
            # First segment, animate mode: zeros for prev conditioning
            prev_latents = np.zeros(
                (1, z_dim, t_l, h_l, w_l), dtype=np.float32
            )
            mask_prev = np.zeros(
                (1, vae_t, t_l, h_l, w_l), dtype=np.float32
            )
        elif seg_idx == 0 and mode == "replace" and bg_frames is not None:
            # First segment, replace mode: use background frames
            prev_latents, mask_prev = self._encode_prev_condition(
                bg_frames[seg_start : seg_start + num_seg_frames],
                mask_frames[seg_start : seg_start + num_seg_frames]
                if mask_frames
                else None,
                latent_shape,
                device,
            )
        elif prev_segment_video is not None:
            # Subsequent segments: use last decoded frames from prev segment
            prev_latents, mask_prev = self._encode_prev_from_decoded(
                prev_segment_video, latent_shape, device
            )
        else:
            prev_latents = np.zeros(
                (1, z_dim, t_l, h_l, w_l), dtype=np.float32
            )
            mask_prev = np.zeros(
                (1, vae_t, t_l, h_l, w_l), dtype=np.float32
            )

        # Combine: [mask(vae_t ch), latents(z_dim ch)] = [B, 20, T_l, H_l, W_l]
        y_prev = np.concatenate(
            [mask_prev, prev_latents], axis=1
        ).astype(np.float32)
        y_prev_buf = _numpy_f32_to_buffer(
            y_prev, self.vae.config.dtype, device
        )

        # Full conditioning: concat ref + prev along temporal dim
        # ref_condition: [B, 20, 1, H_l, W_l]
        # y_prev: [B, 20, T_l, H_l, W_l]
        # Result: [B, 20, 1+T_l, H_l, W_l]
        if self._i2v_concat_model is None:
            # Use numpy concat for simplicity (this is done once per segment)
            ref_np = _buffer_to_numpy_f32(ref_condition)
            full_cond = np.concatenate([ref_np, y_prev], axis=2).astype(
                np.float32
            )
            return _numpy_f32_to_buffer(
                full_cond, self.vae.config.dtype, device
            )
        else:
            ref_np = _buffer_to_numpy_f32(ref_condition)
            full_cond = np.concatenate([ref_np, y_prev], axis=2).astype(
                np.float32
            )
            return _numpy_f32_to_buffer(
                full_cond, self.vae.config.dtype, device
            )

    def _encode_prev_condition(
        self,
        frames: list[Any],
        mask_frames_slice: list[Any] | None,
        latent_shape: tuple[int, ...],
        device: Device,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode previous conditioning frames (for replace mode first segment)."""
        # Simple implementation: encode frames via VAE, build mask
        z_dim = int(latent_shape[1])
        t_l = int(latent_shape[2])
        h_l = int(latent_shape[3])
        w_l = int(latent_shape[4])
        vae_t = self.vae_scale_factor_temporal

        # For now, return zeros (replace mode can be refined later)
        prev_latents = np.zeros(
            (1, z_dim, t_l, h_l, w_l), dtype=np.float32
        )
        mask_prev = np.zeros(
            (1, vae_t, t_l, h_l, w_l), dtype=np.float32
        )
        return prev_latents, mask_prev

    def _encode_prev_from_decoded(
        self,
        decoded_video: np.ndarray,
        latent_shape: tuple[int, ...],
        device: Device,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode previously decoded frames for segment conditioning."""
        z_dim = int(latent_shape[1])
        t_l = int(latent_shape[2])
        h_l = int(latent_shape[3])
        w_l = int(latent_shape[4])
        vae_t = self.vae_scale_factor_temporal

        # VAE encode the decoded video
        enc_buf = _numpy_f32_to_buffer(
            decoded_video.astype(np.float32), self.vae.config.dtype, device
        )
        enc_latent = self.vae.encode(enc_buf)
        prev_latents = _buffer_to_numpy_f32(enc_latent)

        # Standardize
        mean = np.array(
            self.vae.config.latents_mean, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        inv_std = 1.0 / np.array(
            self.vae.config.latents_std, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        prev_latents = (prev_latents - mean) * inv_std

        # Pad/truncate to match expected temporal dimension
        if prev_latents.shape[2] < t_l:
            pad = np.zeros(
                (1, z_dim, t_l - prev_latents.shape[2], h_l, w_l),
                dtype=np.float32,
            )
            prev_latents = np.concatenate([prev_latents, pad], axis=2)
        elif prev_latents.shape[2] > t_l:
            prev_latents = prev_latents[:, :, :t_l, :, :]

        # Build i2v mask: first prev_cond_frames are conditioned
        mask_prev = np.zeros(
            (1, vae_t, t_l, h_l, w_l), dtype=np.float32
        )
        # Mark conditioned frames
        cond_t = min(prev_latents.shape[2], t_l)
        mask_prev[:, :, :cond_t, :, :] = 1.0

        return prev_latents, mask_prev

    def _run_animate_denoising(
        self,
        latents: Buffer,
        condition: Buffer,
        prompt_embeds: Buffer,
        negative_prompt_embeds: Buffer | None,
        clip_features: Buffer,
        pose_latents: Buffer,
        face_emb: Buffer,
        num_temporal_frames: Buffer,
        do_cfg: bool,
        rope_cos: Buffer,
        rope_sin: Buffer,
        batched_timesteps: list[Buffer],
        coeff_buffers: list[Buffer],
        spatial_shape: Buffer,
        guidance_scale: Buffer | None,
    ) -> Buffer:
        """Run denoising loop with animate conditioning."""
        from .pipeline_wan import WanUniPCState

        step_state: WanUniPCState = (None, None, None)
        progress = tqdm(  # type: ignore[call-arg]
            range(len(batched_timesteps)),
            desc="Denoising",
            leave=True,
            disable=not sys.stderr.isatty(),
        )

        for i in progress:  # type: ignore[attr-defined]
            with Tracer(f"denoise_step_{i}"):
                dit_timestep = batched_timesteps[i]
                latent_model_input = (
                    self.compiled.cast_f32_to_model_dtype.execute(latents)[0]
                )

                # Concat condition with latents -> 36 channels
                latent_model_input = self._concat_i2v_condition(
                    latent_model_input, condition
                )

                # Run animate transformer
                with Tracer("transformer"):
                    noise_pred_buf = self.transformer(
                        latent_model_input,
                        dit_timestep,
                        prompt_embeds,
                        clip_features,
                        pose_latents,
                        rope_cos,
                        rope_sin,
                        spatial_shape,
                        face_emb,
                        num_temporal_frames,
                    )
                    noise_pred_buf = getattr(
                        noise_pred_buf, "driver_tensor", noise_pred_buf
                    )


                # CFG (2-pass: positive then negative)
                if do_cfg and negative_prompt_embeds is not None:
                    assert guidance_scale is not None
                    noise_uncond = self.transformer(
                        latent_model_input,
                        dit_timestep,
                        negative_prompt_embeds,
                        clip_features,
                        pose_latents,
                        rope_cos,
                        rope_sin,
                        spatial_shape,
                        # Zero out face for uncond: face * 0 - 1
                        self._get_uncond_face_emb(face_emb),
                        num_temporal_frames,
                    )
                    noise_uncond = getattr(
                        noise_uncond, "driver_tensor", noise_uncond
                    )
                    noise_pred_buf = self.compiled.guidance(
                        noise_pred_buf, noise_uncond, guidance_scale
                    )
                    noise_pred_buf = getattr(
                        noise_pred_buf, "driver_tensor", noise_pred_buf
                    )

                # Scheduler step
                with Tracer("scheduler_step"):
                    latents, step_state = self._denoise_step(
                        latents,
                        noise_pred_buf,
                        coeff_buffers[i],
                        step_state,
                    )

        return latents

    def _get_uncond_face_emb(self, face_emb: Buffer) -> Buffer:
        """Create unconditional face embedding: face * 0 - 1."""
        # Simple approach: compute on CPU and transfer
        face_np = _buffer_to_numpy_f32(face_emb)
        uncond = (face_np * 0 - 1).astype(np.float32)
        return _numpy_f32_to_buffer(
            uncond, self.vae.config.dtype, self.transformer.devices[0]
        )

    def _decode_segment_latents(
        self,
        latents: Buffer,
        model_inputs: WanAnimateModelInputs,
        skip_first: bool = True,
    ) -> np.ndarray:
        """Decode latents to video frames, optionally skipping first frame.

        Matches diffusers: skip first LATENT frame (ref conditioning),
        then VAE decode the remaining frames.
        """
        # Denormalize latents
        denormed = self._denormalize_vae_latents(latents)

        if skip_first:
            # Skip first latent frame (ref conditioning) BEFORE decode
            # (matches diffusers: vae.decode(latents[:, :, 1:]))
            denormed_np = _buffer_to_numpy_f32(denormed)
            denormed_np = denormed_np[:, :, 1:, :, :]
            denormed = _numpy_f32_to_buffer(
                denormed_np, self.vae.config.dtype, self.transformer.devices[0]
            )

        # VAE decode
        decoded = self.vae.decode(denormed)
        return _buffer_to_numpy_f32(decoded[0])

    def _extract_last_frames(
        self,
        decoded: np.ndarray,
        num_frames: int,
    ) -> np.ndarray:
        """Extract last N frames from decoded video for next segment conditioning."""
        return decoded[:, :, -num_frames:, :, :]

    def prepare_inputs(self, context: Any) -> WanAnimateModelInputs:
        """Prepare animate model inputs from context."""
        base_inputs = super().prepare_inputs(context)
        animate_inputs = WanAnimateModelInputs(
            tokens=base_inputs.tokens,
            negative_tokens=base_inputs.negative_tokens,
            mask=base_inputs.mask,
            negative_mask=base_inputs.negative_mask,
            latents=base_inputs.latents,
            timesteps=base_inputs.timesteps,
            step_coefficients=base_inputs.step_coefficients,
            boundary_timestep=base_inputs.boundary_timestep,
            width=base_inputs.width,
            height=base_inputs.height,
            num_frames=base_inputs.num_frames,
            num_inference_steps=base_inputs.num_inference_steps,
            guidance_scale=base_inputs.guidance_scale,
            guidance_scale_2=base_inputs.guidance_scale_2,
        )

        # Copy animate-specific fields from context
        if hasattr(context, "pose_video"):
            animate_inputs.pose_video = context.pose_video
        if hasattr(context, "face_video"):
            animate_inputs.face_video = context.face_video
        if hasattr(context, "input_image"):
            animate_inputs.input_image = context.input_image
        if hasattr(context, "mode"):
            animate_inputs.mode = context.mode
        if hasattr(context, "background_video"):
            animate_inputs.background_video = context.background_video
        if hasattr(context, "mask_video"):
            animate_inputs.mask_video = context.mask_video
        if hasattr(context, "segment_frame_length"):
            animate_inputs.segment_frame_length = context.segment_frame_length
        if hasattr(context, "prev_segment_conditioning_frames"):
            animate_inputs.prev_segment_conditioning_frames = (
                context.prev_segment_conditioning_frames
            )
        if hasattr(context, "motion_encode_batch_size"):
            animate_inputs.motion_encode_batch_size = (
                context.motion_encode_batch_size
            )
        if hasattr(context, "shared_inputs_dir"):
            animate_inputs.shared_inputs_dir = context.shared_inputs_dir

        return animate_inputs
