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

"""Wan Image-to-Video (I2V) pipeline.

Extends WanPipeline with image conditioning: the input image is encoded
via the VAE, combined with a temporal mask, and concatenated with noise
latents at each denoising step to produce 36-channel transformer input.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from max.driver import CPU, Buffer, Device
from max.profiler import Tracer, traced

from ..autoencoders.autoencoder_kl_wan import (
    _buffer_to_numpy_f32,
    _numpy_f32_to_buffer,
)
from .pipeline_wan import WanModelInputs, WanPipeline, WanPipelineOutput

logger = logging.getLogger(__name__)


class WanI2VPipeline(WanPipeline):
    """Wan I2V pipeline — extends WanPipeline with image conditioning.

    When ``input_image`` is provided in model_inputs, the image is encoded
    via VAE and concatenated with noise latents (16 + 4 mask + 16 image = 36
    channels) at each denoising step.
    """

    def init_remaining_components(self) -> None:
        super().init_remaining_components()
        # Pre-compile VAE encoder (dynamic H/W, single compilation).
        self.vae.prewarm_encoder()

    def _prepare_i2v_condition(
        self,
        model_inputs: WanModelInputs,
        latent_shape: tuple[int, ...],
        device: Device,
    ) -> Buffer:
        """Prepare I2V condition tensor [B, 20, T_l, H_l, W_l].

        Encodes the input image via VAE, builds a temporal mask, and
        concatenates them.
        """
        image = model_inputs.input_image
        if image is None:
            raise ValueError("I2V pipeline requires input_image")
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        logger.info("Preparing I2V condition")

        # Normalize to [-1, 1] float32, shape [1, 3, H, W]
        image_f32 = image.astype(np.float32) / 127.5 - 1.0
        if image_f32.ndim == 3:
            image_f32 = image_f32.transpose(2, 0, 1)[np.newaxis]  # [1,3,H,W]

        batch_size = int(latent_shape[0])
        num_frames = int(model_inputs.num_frames)
        # Use target height/width from model_inputs (pixel space)
        h = int(model_inputs.height)
        w = int(model_inputs.width)

        # Resize image to target resolution if needed
        if image_f32.shape[2] != h or image_f32.shape[3] != w:
            import PIL.Image

            pil_img = PIL.Image.fromarray(
                ((image_f32[0].transpose(1, 2, 0) + 1.0) * 127.5)
                .clip(0, 255)
                .astype(np.uint8)
            )
            pil_img = pil_img.resize((w, h), PIL.Image.LANCZOS)
            image_f32 = (
                np.array(pil_img).astype(np.float32) / 127.5 - 1.0
            ).transpose(2, 0, 1)[np.newaxis]

        video_condition_np = np.zeros(
            (batch_size, 3, num_frames, h, w), dtype=np.float32
        )
        video_condition_np[:, :, 0:1, :, :] = image_f32[:, :, np.newaxis, :, :]

        enc_buf = _numpy_f32_to_buffer(
            video_condition_np, self.vae.config.dtype, device
        )
        enc_latent = self.vae.encode(enc_buf)
        latent_cond_np = _buffer_to_numpy_f32(enc_latent)

        logger.debug(
            "VAE encode output: shape=%s min=%.4f max=%.4f mean=%.4f",
            latent_cond_np.shape,
            latent_cond_np.min(),
            latent_cond_np.max(),
            latent_cond_np.mean(),
        )

        expected_t = int(latent_shape[2])
        if latent_cond_np.shape[2] != expected_t:
            raise ValueError(
                "VAE encode temporal shape mismatch for I2V condition: "
                f"got {latent_cond_np.shape[2]}, expected {expected_t} "
                f"for num_frames={num_frames}."
            )

        expected_h = int(latent_shape[3])
        expected_w = int(latent_shape[4])
        if (
            latent_cond_np.shape[3] != expected_h
            or latent_cond_np.shape[4] != expected_w
        ):
            raise ValueError(
                "VAE encode spatial shape mismatch for I2V condition: "
                f"got {latent_cond_np.shape[3:5]}, expected "
                f"({expected_h}, {expected_w})."
            )

        z_dim = self.vae.config.z_dim
        mean = np.array(
            self.vae.config.latents_mean, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        inv_std = 1.0 / np.array(
            self.vae.config.latents_std, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        latent_cond_np = (latent_cond_np - mean) * inv_std

        # Build mask [B, vae_scale_factor_temporal, T_l, H_l, W_l]
        t_latent = latent_cond_np.shape[2]
        h_latent = latent_cond_np.shape[3]
        w_latent = latent_cond_np.shape[4]

        mask = np.zeros(
            (batch_size, 1, num_frames, h_latent, w_latent),
            dtype=np.float32,
        )
        mask[:, :, 0, :, :] = 1.0  # First frame is conditioned

        vae_t = self.vae_scale_factor_temporal
        first_mask = np.repeat(mask[:, :, 0:1, :, :], vae_t, axis=2)
        mask_expanded = np.concatenate(
            [first_mask, mask[:, :, 1:, :, :]], axis=2
        )
        # Reshape: [B, 1, T_pixel, H_l, W_l] -> [B, vae_t, T_l, H_l, W_l]
        mask_expanded = mask_expanded.reshape(
            batch_size, -1, vae_t, h_latent, w_latent
        )
        mask_expanded = mask_expanded.transpose(0, 2, 1, 3, 4)

        # Concat: [mask, latent_condition] -> [B, vae_t+z_dim, T_l, H_l, W_l]
        condition = np.concatenate(
            [mask_expanded, latent_cond_np], axis=1
        ).astype(np.float32)

        self._maybe_dump_i2v_debug_tensors(
            video_condition=video_condition_np,
            latent_condition=latent_cond_np,
            mask=mask_expanded,
            condition=condition,
        )

        return _numpy_f32_to_buffer(condition, self.vae.config.dtype, device)

    def _maybe_dump_i2v_debug_tensors(
        self,
        *,
        video_condition: np.ndarray,
        latent_condition: np.ndarray,
        mask: np.ndarray,
        condition: np.ndarray,
    ) -> None:
        debug_dir = os.environ.get("WAN_I2V_DEBUG_DIR")
        if not debug_dir:
            return

        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "video_condition.npy": video_condition,
            "latent_condition.npy": latent_condition,
            "mask.npy": mask,
            "condition.npy": condition,
        }
        for name, value in artifacts.items():
            out_path = debug_path / name
            np.save(out_path, value)
            logger.info("Saved Wan I2V debug tensor: %s", out_path)

    @staticmethod
    def _concat_i2v_condition(
        latent_model_input: Buffer, condition: Buffer
    ) -> Buffer:
        """Concat latents [B,16,T,H,W] with condition [B,20,T,H,W] -> [B,36,T,H,W]."""
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

    @traced(message="WanI2VPipeline.execute")
    def execute(  # type: ignore[override]
        self,
        model_inputs: WanModelInputs,
        **kwargs: object,
    ) -> WanPipelineOutput:
        import time as _time

        del kwargs
        device = self.transformer.devices[0]
        if not self._moe_dual_loaded:
            self._activate_transformer_weights(use_secondary=False)

        t_start = _time.perf_counter()
        with Tracer("prepare_prompt_embeddings"):
            (
                prompt_embeds,
                negative_prompt_embeds,
                batched_prompt_embeds,
                do_cfg,
            ) = self._prepare_prompt_state(model_inputs)
        t_prompt = _time.perf_counter()

        with Tracer("preprocess_latents"):
            latents = self._prepare_latents(model_inputs, device)

        # Prepare I2V condition from input image (includes VAE encode)
        with Tracer("prepare_i2v_condition"):
            i2v_condition = self._prepare_i2v_condition(
                model_inputs, tuple(int(d) for d in latents.shape), device
            )
        t_encode = _time.perf_counter()

        # Pre-compile VAE decoder
        self.vae.prewarm_for_latent_shape(
            tuple(int(d) for d in latents.shape)
        )
        t_prewarm = _time.perf_counter()

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
            latents = self._run_i2v_denoising(
                latents,
                i2v_condition,
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
            )
        t_denoise = _time.perf_counter()

        with Tracer("decode_outputs"):
            images = self._decode_output(latents, model_inputs)
        t_decode = _time.perf_counter()

        logger.info(
            "I2V timing: prompt=%.1fs, vae_encode=%.1fs, "
            "vae_prewarm=%.1fs, denoise=%.1fs, vae_decode=%.1fs, "
            "total=%.1fs",
            t_prompt - t_start,
            t_encode - t_prompt,
            t_prewarm - t_encode,
            t_denoise - t_prewarm,
            t_decode - t_denoise,
            t_decode - t_start,
        )
        return WanPipelineOutput(images=images)

    def _run_i2v_denoising(
        self,
        latents: Buffer,
        i2v_condition: Buffer,
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
    ) -> Buffer:
        """Denoising loop with I2V condition concatenation."""
        from .pipeline_wan import WanUniPCState

        step_state: WanUniPCState = (None, None, None)
        latents, step_state = self._run_i2v_denoising_phase(
            latents=latents,
            i2v_condition=i2v_condition,
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
        )

        if has_moe and boundary_step_idx < len(batched_timesteps):
            if self._moe_dual_loaded:
                low_noise_model = self.transformer_2
            else:
                self._activate_transformer_weights(use_secondary=True)
                low_noise_model = self.transformer
            latents, _ = self._run_i2v_denoising_phase(
                latents=latents,
                i2v_condition=i2v_condition,
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
            )

        return latents

    def _run_i2v_denoising_phase(
        self,
        latents: Buffer,
        i2v_condition: Buffer,
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
        step_state: tuple,
    ) -> tuple[Buffer, tuple]:
        """Denoising phase with I2V condition concat at each step."""
        import sys

        from tqdm.auto import tqdm

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
                # I2V: concat condition with latents → 36 channels
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
