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

Supports both legacy concat conditioning and Wan2.2 TI2V first-frame
latent conditioning, selected from the loaded transformer config.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.profiler import Tracer, traced

from ..autoencoders.autoencoder_kl_wan import (
    _buffer_to_numpy_f32,
    _numpy_f32_to_buffer,
)
from .pipeline_wan import WanModelInputs, WanPipeline, WanPipelineOutput

logger = logging.getLogger(__name__)


def _resize_with_center_crop(
    image: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """Resize image to target size with aspect-ratio-preserving center crop."""
    import PIL.Image

    pil_img = PIL.Image.fromarray(image.astype(np.uint8))
    ratio = target_width / target_height
    src_ratio = pil_img.width / pil_img.height

    src_w = (
        target_width
        if ratio > src_ratio
        else pil_img.width * target_height // pil_img.height
    )
    src_h = (
        target_height
        if ratio <= src_ratio
        else pil_img.height * target_width // pil_img.width
    )

    resized = pil_img.resize((src_w, src_h), PIL.Image.Resampling.LANCZOS)
    canvas = PIL.Image.new("RGB", (target_width, target_height))
    canvas.paste(
        resized,
        box=(
            target_width // 2 - src_w // 2,
            target_height // 2 - src_h // 2,
        ),
    )
    return np.array(canvas, dtype=np.uint8)


class WanI2VPipeline(WanPipeline):
    """Wan I2V pipeline — extends WanPipeline with image conditioning."""

    _i2v_concat_model: Any = None
    _i2v_mix_model: Any = None

    def _prepare_condition_image(
        self, model_inputs: WanModelInputs
    ) -> np.ndarray:
        image = model_inputs.input_image
        if image is None:
            raise ValueError("I2V pipeline requires input_image")
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        image_f32 = image.astype(np.float32) / 127.5 - 1.0
        if image_f32.ndim == 3:
            image_f32 = image_f32.transpose(2, 0, 1)[np.newaxis]

        height = int(model_inputs.height)
        width = int(model_inputs.width)
        if image_f32.shape[2] == height and image_f32.shape[3] == width:
            return image_f32

        image_u8 = (
            ((image_f32[0].transpose(1, 2, 0) + 1.0) * 127.5)
            .clip(0, 255)
            .astype(np.uint8)
        )
        image_u8 = _resize_with_center_crop(image_u8, width, height)
        return (image_u8.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[
            np.newaxis
        ]

    def _normalize_vae_latents(self, latent_cond_np: np.ndarray) -> np.ndarray:
        z_dim = self.vae.config.z_dim
        mean = np.array(self.vae.config.latents_mean, dtype=np.float32).reshape(
            1, z_dim, 1, 1, 1
        )
        inv_std = 1.0 / np.array(
            self.vae.config.latents_std, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        return (latent_cond_np - mean) * inv_std

    def _prepare_legacy_i2v_condition(
        self,
        model_inputs: WanModelInputs,
        latent_shape: tuple[int, ...],
        device: Device,
    ) -> Buffer:
        """Prepare I2V condition tensor [B, 20, T_l, H_l, W_l].

        Encodes the input image via VAE, builds a temporal mask, and
        concatenates them.
        """
        logger.info("Preparing I2V condition")

        image_f32 = self._prepare_condition_image(model_inputs)
        batch_size = int(latent_shape[0])
        num_frames = int(model_inputs.num_frames)
        h = int(model_inputs.height)
        w = int(model_inputs.width)

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

        expected_spatial = latent_shape[3:5]
        if latent_cond_np.shape[3:5] != expected_spatial:
            raise ValueError(
                "VAE encode spatial shape mismatch for I2V condition: "
                f"got {latent_cond_np.shape[3:5]}, expected "
                f"{expected_spatial}."
            )

        latent_cond_np = self._normalize_vae_latents(latent_cond_np)

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

        return _numpy_f32_to_buffer(condition, self.vae.config.dtype, device)

    def _prepare_ti2v_condition(
        self,
        model_inputs: WanModelInputs,
        latent_shape: tuple[int, ...],
        device: Device,
    ) -> Buffer:
        """Prepare first-frame latent condition [B, C, 1, H_l, W_l]."""
        image_f32 = self._prepare_condition_image(model_inputs)
        batch_size = int(latent_shape[0])
        video_condition_np = np.repeat(
            image_f32[:, :, np.newaxis, :, :], batch_size, axis=0
        )
        enc_buf = _numpy_f32_to_buffer(
            video_condition_np, self.vae.config.dtype, device
        )
        enc_latent = self.vae.encode(enc_buf)
        latent_cond_np = _buffer_to_numpy_f32(enc_latent)

        if latent_cond_np.shape[2] != 1:
            raise ValueError(
                "TI2V I2V expects a single conditioned latent frame, got "
                f"{latent_cond_np.shape[2]}."
            )

        expected_spatial = latent_shape[3:5]
        if latent_cond_np.shape[3:5] != expected_spatial:
            raise ValueError(
                "TI2V I2V VAE encode spatial shape mismatch: "
                f"got {latent_cond_np.shape[3:5]}, expected {expected_spatial}."
            )

        latent_cond_np = self._normalize_vae_latents(latent_cond_np)
        return _numpy_f32_to_buffer(
            latent_cond_np, self.transformer.config.dtype, device
        )

    def _get_first_frame_mask(
        self, latent_shape: tuple[int, ...], device: Device
    ) -> Buffer:
        mask_np = np.ones(
            (latent_shape[0], 1, *latent_shape[2:]),
            dtype=np.float32,
        )
        mask_np[:, :, 0, :, :] = 0.0
        return _numpy_f32_to_buffer(
            mask_np, self.transformer.config.dtype, device
        )

    def _compile_i2v_concat(
        self, latent_model_input: Buffer, condition: Buffer
    ) -> Any:
        """Compile a GPU graph that concatenates latents + condition along axis=1."""
        from max.graph import Graph, TensorType, ops

        device = self.transformer.devices[0]
        dtype = latent_model_input.dtype
        lat_shape = list(latent_model_input.shape)
        cond_shape = list(condition.shape)

        with Graph(
            "wan_i2v_concat",
            input_types=[
                TensorType(dtype, lat_shape, device=device),
                TensorType(dtype, cond_shape, device=device),
            ],
        ) as g:
            lat = g.inputs[0].tensor
            cond = g.inputs[1].tensor
            g.output(ops.concat([lat, cond], axis=1))
        return self.session.load(g)

    def _concat_i2v_condition(
        self, latent_model_input: Buffer, condition: Buffer
    ) -> Buffer:
        """Concat latents [B,C_l,T,H,W] with condition [B,C_c,T,H,W] on GPU."""
        if self._i2v_concat_model is None:
            self._i2v_concat_model = self._compile_i2v_concat(
                latent_model_input, condition
            )
        return self._i2v_concat_model.execute(latent_model_input, condition)[0]

    def _compile_i2v_mix(
        self, latent_model_input: Buffer, condition: Buffer, mask: Buffer
    ) -> Any:
        from max.graph import Graph, TensorType

        device = self.transformer.devices[0]
        dtype = latent_model_input.dtype
        with Graph(
            "wan_i2v_mix",
            input_types=[
                TensorType(
                    dtype, list(latent_model_input.shape), device=device
                ),
                TensorType(dtype, list(condition.shape), device=device),
                TensorType(dtype, list(mask.shape), device=device),
            ],
        ) as g:
            lat = g.inputs[0].tensor
            cond = g.inputs[1].tensor
            mask_tensor = g.inputs[2].tensor
            g.output((1.0 - mask_tensor) * cond + mask_tensor * lat)
        return self.session.load(g)

    def _mix_i2v_condition(
        self, latent_model_input: Buffer, condition: Buffer, mask: Buffer
    ) -> Buffer:
        if self._i2v_mix_model is None:
            self._i2v_mix_model = self._compile_i2v_mix(
                latent_model_input, condition, mask
            )
        return self._i2v_mix_model.execute(latent_model_input, condition, mask)[
            0
        ]

    def _apply_i2v_condition(
        self,
        latent_model_input: Buffer,
        i2v_condition: Buffer,
        first_frame_mask: Buffer | None,
    ) -> Buffer:
        if first_frame_mask is None:
            return self._concat_i2v_condition(latent_model_input, i2v_condition)
        return self._mix_i2v_condition(
            latent_model_input, i2v_condition, first_frame_mask
        )

    def _prepare_i2v_runtime_inputs(
        self,
        model_inputs: WanModelInputs,
        latents: Buffer,
        device: Device,
    ) -> tuple[Buffer, Buffer | None]:
        latent_shape = tuple(int(d) for d in latents.shape)
        if self.expand_timesteps:
            return (
                self._prepare_ti2v_condition(
                    model_inputs, latent_shape, device
                ),
                self._get_first_frame_mask(latent_shape, device),
            )
        return (
            self._prepare_legacy_i2v_condition(
                model_inputs, latent_shape, device
            ),
            None,
        )

    def _ensure_i2v_graph_compiled(
        self,
        latent_model_input: Buffer,
        i2v_condition: Buffer,
        first_frame_mask: Buffer | None,
    ) -> None:
        if first_frame_mask is None:
            if self._i2v_concat_model is None:
                self._i2v_concat_model = self._compile_i2v_concat(
                    latent_model_input, i2v_condition
                )
            return

        if self._i2v_mix_model is None:
            self._i2v_mix_model = self._compile_i2v_mix(
                latent_model_input, i2v_condition, first_frame_mask
            )

    def _first_frame_token_count(self, latents: Buffer) -> int:
        if not self.expand_timesteps:
            return 0

        p_t, p_h, p_w = self.transformer.config.patch_size
        return (
            (1 // p_t)
            * (int(latents.shape[3]) // p_h)
            * (int(latents.shape[4]) // p_w)
        )

    def _clamp_conditioned_first_frame(
        self,
        latents: Buffer,
        i2v_condition: Buffer,
        first_frame_mask: Buffer | None,
    ) -> Buffer:
        if first_frame_mask is None:
            return latents

        latents_model = self._cast_f32_to_model_dtype.execute(latents)[0]
        clamped = self._apply_i2v_condition(
            latents_model, i2v_condition, first_frame_mask
        )
        return self._cast_model_dtype_to_f32.execute(clamped)[0]

    @traced(message="WanI2VPipeline.execute")
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
            prompt_embeds, negative_prompt_embeds, do_cfg = (
                self.prepare_prompt_embeddings(model_inputs)
            )

        with Tracer("preprocess_latents"):
            logger.info("Preparing Wan latents")
            latents = Buffer.from_numpy(
                np.ascontiguousarray(model_inputs.latents, dtype=np.float32)
            ).to(device)
            self._ensure_transformer_compiled(latents, prompt_embeds)

        # Prepare I2V condition from input image.
        with Tracer("prepare_i2v_condition"):
            i2v_condition, first_frame_mask = self._prepare_i2v_runtime_inputs(
                model_inputs, latents, device
            )

        latent_model_input = self._cast_f32_to_model_dtype.execute(latents)[0]
        self._ensure_i2v_graph_compiled(
            latent_model_input, i2v_condition, first_frame_mask
        )

        with Tracer("prepare_scheduler"):
            if model_inputs.step_coefficients is None:
                raise ValueError(
                    "WanPipeline requires precomputed step_coefficients."
                )
            timesteps = np.ascontiguousarray(
                model_inputs.timesteps, dtype=np.float32
            )
            boundary_timestep = model_inputs.boundary_timestep
            if boundary_timestep is None and self.boundary_ratio is not None:
                boundary_timestep = (
                    self.boundary_ratio * self.num_train_timesteps
                )
            rope_cos, rope_sin = self.transformer.compute_rope(
                num_frames=int(latents.shape[2]),
                height=int(latents.shape[3]),
                width=int(latents.shape[4]),
            )
            token_seq_len = self._compute_token_seq_len_from_latents(latents)
            batched_timesteps = self._get_batched_timesteps(
                scheduler_timesteps=timesteps,
                batch_size=int(latents.shape[0]),
                seq_len=token_seq_len,
                device=device,
                prefix_zero_tokens=self._first_frame_token_count(latents),
            )
            coeff_buffers = [
                Buffer.from_numpy(
                    np.ascontiguousarray(row, dtype=np.float32)
                ).to(device)
                for row in model_inputs.step_coefficients
            ]
            guidance_scale_high: Buffer | None = None
            guidance_scale_low: Buffer | None = None
            if do_cfg:
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
            has_moe = (
                self.transformer_2 is not None and boundary_timestep is not None
            )
            boundary_step_idx = len(timesteps)
            if boundary_timestep is not None:
                for idx, t in enumerate(timesteps):
                    if float(t) < boundary_timestep:
                        boundary_step_idx = idx
                        break
            p_t, p_h, p_w = self.transformer.config.patch_size
            spatial_shape = self._get_spatial_shape(
                int(latents.shape[2]) // p_t,
                int(latents.shape[3]) // p_h,
                int(latents.shape[4]) // p_w,
                device,
            )

        with Tracer("denoising_loop"):
            latents = self._run_i2v_denoising(
                latents,
                i2v_condition,
                first_frame_mask,
                prompt_embeds,
                negative_prompt_embeds,
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
            latents = self._clamp_conditioned_first_frame(
                latents, i2v_condition, first_frame_mask
            )
        with Tracer("decode_outputs"):
            images = self.decode_latents(
                latents,
                int(model_inputs.num_frames),
                model_inputs.height,
                model_inputs.width,
            )
        return WanPipelineOutput(images=images)

    def _run_i2v_denoising(
        self,
        latents: Buffer,
        i2v_condition: Buffer,
        first_frame_mask: Buffer | None,
        prompt_embeds: Buffer,
        negative_prompt_embeds: Buffer | None,
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
        """Denoising loop with either concat or first-frame mix conditioning."""
        from .pipeline_wan import WanUniPCState

        step_state: WanUniPCState = (None, None, None)
        latents, step_state = self._run_i2v_denoising_phase(
            latents=latents,
            i2v_condition=i2v_condition,
            first_frame_mask=first_frame_mask,
            transformer_model=self.transformer,
            prompt_embeds=prompt_embeds,
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
                first_frame_mask=first_frame_mask,
                transformer_model=low_noise_model,
                prompt_embeds=prompt_embeds,
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
        first_frame_mask: Buffer | None,
        transformer_model: Any,
        prompt_embeds: Buffer,
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
        """Denoising phase with per-step I2V conditioning."""
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
                latent_model_input = self._cast_f32_to_model_dtype.execute(
                    latents
                )[0]
                latent_model_input = self._apply_i2v_condition(
                    latent_model_input,
                    i2v_condition,
                    first_frame_mask,
                )
                with Tracer("transformer"):
                    noise_pred_buf = self._run_transformer_forward(
                        transformer_model=transformer_model,
                        latent_model_input=latent_model_input,
                        dit_timestep=dit_timestep,
                        prompt_embeds=prompt_embeds,
                        batched_prompt_embeds=None,
                        negative_prompt_embeds=negative_prompt_embeds,
                        rope_cos=rope_cos,
                        rope_sin=rope_sin,
                        spatial_shape=spatial_shape,
                        do_cfg=do_cfg,
                        guidance_scale=guidance_scale,
                    )
                with Tracer("scheduler_step"):
                    latents, step_state = self.scheduler_step(
                        latents,
                        noise_pred_buf,
                        coeff_buffers[i],
                        step_state,
                    )
        return latents, step_state
