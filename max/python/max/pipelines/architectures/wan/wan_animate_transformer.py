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

"""Wan-Animate transformer components (MAX Graph API + PyTorch bridges).

New modules for Wan-Animate that extend the base Wan transformer:
- WanAnimatePosePatchEmbedding: Conv3d for pose latent injection
- WanAnimateMotionEncoder: MAX-native StyleGAN2-based motion encoder
- WanAnimateMotionEncoderBridge: PyTorch bridge (kept for reference/fallback)
- WanAnimateFaceEncoder: CausalConv1d face encoder (MAX Graph API)
- WanAnimateFaceBlock: Face adapter cross-attention (MAX Graph API)
- WanAnimatePreProcess: Extended pre-processing with pose + CLIP
- CLIPImageEncoderBridge: PyTorch bridge for CLIP ViT-H/14
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import Module
from max.nn.linear import Linear

from .model_config import WanConfigBase
from .wan_transformer import (
    WanConv3d,
    WanLayerNorm,
    WanRMSNorm,
    WanTimeTextImageEmbedding,
)

logger = logging.getLogger(__name__)


class WanAnimatePosePatchEmbedding(WanConv3d):
    """Conv3d for pose latent injection (16 input channels -> dim).

    Inherits from WanConv3d so weight keys are directly
    ``pose_patch_embedding.weight`` / ``pose_patch_embedding.bias``
    matching the checkpoint.
    """

    def __init__(
        self,
        config: WanConfigBase,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        dim = config.num_attention_heads * config.attention_head_dim
        super().__init__(
            kernel_size=config.patch_size,
            in_channels=config.latent_channels,
            out_channels=dim,
            stride=config.patch_size,
            dtype=dtype,
            device=device,
        )

    def forward(self, pose_hidden_states: TensorValue) -> TensorValue:
        """Pose patch embedding: [B, 16, T, H, W] -> [B, D, ppf, pph, ppw]."""
        # Conv3d expects NDHWC
        x = ops.permute(pose_hidden_states, [0, 2, 3, 4, 1])
        x = super().__call__(x)
        x = ops.permute(x, [0, 4, 1, 2, 3])  # -> [B, D, ppf, pph, ppw]
        return x


class WanAnimateFaceEncoder(Module):
    """Face encoder: motion vectors -> face embeddings via CausalConv1d.

    Architecture:
        conv1_local(512 -> 4096, k=3) -> reshape to 4 heads ->
        norm1 -> SiLU ->
        conv2(1024 -> 1024, k=3, s=2) -> norm2 -> SiLU ->
        conv3(1024 -> 1024, k=3, s=2) -> norm3 -> SiLU ->
        out_proj(1024 -> 5120) + padding_tokens
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 1024,
        out_dim: int = 5120,
        num_heads: int = 4,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # CausalConv1d weights: [out_channels, in_channels, kernel_size]
        # conv1_local expands to num_heads * hidden_dim
        self.conv1_local = _CausalConv1d(
            in_dim, hidden_dim * num_heads, kernel_size=3, stride=1,
            dtype=dtype, device=device,
        )
        self.norm1 = WanLayerNorm(
            hidden_dim, elementwise_affine=False, dtype=dtype, device=device,
        )

        self.conv2 = _CausalConv1d(
            hidden_dim, hidden_dim, kernel_size=3, stride=2,
            dtype=dtype, device=device,
        )
        self.norm2 = WanLayerNorm(
            hidden_dim, elementwise_affine=False, dtype=dtype, device=device,
        )

        self.conv3 = _CausalConv1d(
            hidden_dim, hidden_dim, kernel_size=3, stride=2,
            dtype=dtype, device=device,
        )
        self.norm3 = WanLayerNorm(
            hidden_dim, elementwise_affine=False, dtype=dtype, device=device,
        )

        self.out_proj = Linear(
            in_dim=hidden_dim, out_dim=out_dim,
            dtype=dtype, device=device, has_bias=True,
        )
        # Learned null token: [1, 1, 1, out_dim]
        self.padding_tokens = Weight(
            "padding_tokens", dtype, [1, 1, 1, out_dim], device,
        )

    def __call__(
        self, motion_vectors: TensorValue
    ) -> TensorValue:
        """Encode motion vectors to face embeddings.

        Args:
            motion_vectors: [B, T, 512] motion vectors from motion encoder.

        Returns:
            [B, T//4 + 1, num_heads + 1, out_dim] face embeddings
            (with prepended zero frame and padding token channel).
        """
        batch_size = motion_vectors.shape[0]
        t_in = motion_vectors.shape[1]

        # conv1_local: [B, T, 512] -> [B, T, 4096]
        x = self.conv1_local(motion_vectors)

        # Reshape to multi-head: [B, T, 4*1024] -> [B, T, 4, 1024] -> [B, 4, T, 1024] -> [B*4, T, 1024]
        x = ops.reshape(x, [batch_size, t_in, self.num_heads, self.hidden_dim])
        x = ops.permute(x, [0, 2, 1, 3])
        x = ops.reshape(x, [batch_size * self.num_heads, t_in, self.hidden_dim])

        # norm1 -> SiLU
        x = self.norm1(x)
        x = ops.silu(x)

        # conv2: stride-2 downsample T -> T//2
        x = self.conv2(x)
        x = self.norm2(x)
        x = ops.silu(x)

        # conv3: stride-2 downsample T//2 -> T//4
        x = self.conv3(x)
        x = self.norm3(x)
        x = ops.silu(x)

        # t_out = T//4
        t_out = x.shape[1]

        # out_proj: [B*4, T//4, 1024] -> [B*4, T//4, 5120]
        x = self.out_proj(x)

        out_dim = x.shape[2]

        # Reshape back: [B*4, T//4, 5120] -> [B, 4, T//4, 5120] -> [B, T//4, 4, 5120]
        x = ops.reshape(x, [batch_size, self.num_heads, t_out, out_dim])
        x = ops.permute(x, [0, 2, 1, 3])

        # Add padding token channel: [B, T//4, 4, D] -> [B, T//4, 5, D]
        padding = ops.broadcast_to(
            self.padding_tokens,
            [batch_size, t_out, 1, out_dim],
        )
        x = ops.concat([x, padding], axis=2)

        # Prepend zero frame: [B, 1, 5, D] of zeros
        zero_frame = x[:, :1, :, :] - x[:, :1, :, :]
        x = ops.concat([zero_frame, x], axis=1)

        return x  # [B, T//4 + 1, 5, 5120]


class _CausalConv1d(Module):
    """Causal 1D convolution with replicate padding.

    Implemented as: pad(kernel_size-1, 0, mode=replicate) -> conv1d.
    In MAX Graph API, we use manual padding + conv1d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # Weight as [k * in_ch, out_ch] for matmul-based conv1d
        self.weight = Weight(
            "weight", dtype, [kernel_size * in_channels, out_channels], device,
        )
        self.bias = Weight("bias", dtype, [out_channels], device)

    def __call__(self, x: TensorValue) -> TensorValue:
        # x: [B, T, C_in]
        # Implement causal conv1d via unfold + matmul (no Conv2d layout issues)

        # Causal padding: replicate first element (kernel_size - 1) times
        pad_size = self.kernel_size - 1
        if pad_size > 0:
            first_elem = x[:, :1, :]  # [B, 1, C]
            padding = ops.broadcast_to(
                first_elem,
                [x.shape[0], pad_size, x.shape[2]],
            )
            x = ops.concat([padding, x], axis=1)

        # Unfold: extract sliding windows of size kernel_size
        # For stride=1: output T positions = T_padded - kernel_size + 1 = T_orig
        # For stride=2: output T positions = (T_padded - kernel_size) // stride + 1
        if self.kernel_size == 3 and self.stride == 1:
            # Concatenate 3 shifted views: [B, T, C*3]
            x0 = x[:, :-2, :]   # positions 0..T-3
            x1 = x[:, 1:-1, :]  # positions 1..T-2
            x2 = x[:, 2:, :]    # positions 2..T-1
            unfolded = ops.concat([x0, x1, x2], axis=2)  # [B, T_out, C*3]
        elif self.kernel_size == 3 and self.stride == 2:
            # Stride-2: take every other position
            x0 = x[:, :-2:2, :]   # even positions
            x1 = x[:, 1:-1:2, :]  # even+1
            x2 = x[:, 2::2, :]    # even+2
            unfolded = ops.concat([x0, x1, x2], axis=2)  # [B, T_out, C*3]
        else:
            raise NotImplementedError(
                f"CausalConv1d: kernel_size={self.kernel_size}, stride={self.stride} not implemented"
            )

        # Matmul: [B, T_out, k*C_in] @ [k*C_in, C_out] -> [B, T_out, C_out]
        result = ops.matmul(unfolded, self.weight)
        # Add bias
        result = result + self.bias
        return result


class WanAnimateFaceBlock(Module):
    """Face adapter cross-attention block.

    Temporally-aligned attention: video tokens attend to face motion tokens.
    Video [B, S, D] reshaped to [B*T, S/T, H, head_dim] attends to
    face [B, T, 5, D] reshaped to [B*T, 5, H, head_dim].
    """

    def __init__(
        self,
        dim: int = 5120,
        num_heads: int = 40,
        head_dim: int = 128,
        eps: float = 1e-6,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = Linear(
            in_dim=dim, out_dim=dim, dtype=dtype, device=device, has_bias=True,
        )
        self.to_k = Linear(
            in_dim=dim, out_dim=dim, dtype=dtype, device=device, has_bias=True,
        )
        self.to_v = Linear(
            in_dim=dim, out_dim=dim, dtype=dtype, device=device, has_bias=True,
        )
        self.to_out = Linear(
            in_dim=dim, out_dim=dim, dtype=dtype, device=device, has_bias=True,
        )
        self.norm_q = WanRMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.norm_k = WanRMSNorm(head_dim, eps=eps, dtype=dtype, device=device)

    def __call__(
        self,
        hidden_states: TensorValue,
        face_emb: TensorValue,
        num_temporal_frames: TensorValue,
    ) -> TensorValue:
        """Apply face adapter cross-attention.

        Args:
            hidden_states: [B, S, D] video features (S = T * H_p * W_p).
            face_emb: [B, T, 5, D] face motion embeddings.
            num_temporal_frames: [1] int32 tensor with T value (for rebind).

        Returns:
            [B, S, D] residual output to add to hidden_states.
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        t_face = face_emb.shape[1]
        face_tokens = face_emb.shape[2]  # 5

        # LayerNorm (elementwise_affine=False) on video features
        x = _layer_norm_no_affine(hidden_states)

        # Q from video features: [B, S, D]
        q = self.to_q(x)

        # LayerNorm (elementwise_affine=False) on face motion
        # face_emb: [B, T, 5, D] — flatten for norm, then restore
        face_flat_for_norm = ops.reshape(
            face_emb,
            [batch_size * t_face * face_tokens, self.dim],
        )
        face_normed = _layer_norm_no_affine(
            ops.reshape(
                face_flat_for_norm,
                [batch_size * t_face, face_tokens, self.dim],
            )
        )

        # K, V from face motion
        face_kv_flat = ops.reshape(
            face_normed,
            [batch_size * t_face * face_tokens, self.dim],
        )
        k_flat = self.to_k(face_kv_flat)
        v_flat = self.to_v(face_kv_flat)

        # Reshape K, V: -> [B*T, 5, H, head_dim]
        k = ops.reshape(
            k_flat,
            [batch_size * t_face, face_tokens, self.num_heads, self.head_dim],
        )
        v = ops.reshape(
            v_flat,
            [batch_size * t_face, face_tokens, self.num_heads, self.head_dim],
        )

        # Reshape Q: [B, S, D] -> [B*T, S/T, H, head_dim]
        # rebind Q to assert seq_len == t_face * spatial_per_frame
        spatial_per_frame = seq_len // t_face
        q = ops.rebind(q, shape=[
            batch_size, t_face * spatial_per_frame, self.dim,
        ])
        q_4d = ops.reshape(
            q,
            [batch_size * t_face, spatial_per_frame, self.num_heads, self.head_dim],
        )

        # QK-norm
        q_4d = self.norm_q(q_4d)
        k = self.norm_k(k)

        # Flash attention
        original_dtype = q_4d.dtype
        scale = 1.0 / (self.head_dim**0.5)
        attn_out = flash_attention_gpu(
            q_4d, k, v,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )

        # Reshape back: [B*T, S/T, H, head_dim] -> [B, S, D]
        attn_out = ops.reshape(
            attn_out, [batch_size, t_face * spatial_per_frame, self.dim]
        )
        attn_out = ops.cast(attn_out, original_dtype)

        return self.to_out(attn_out)


def _layer_norm_no_affine(x: TensorValue) -> TensorValue:
    """LayerNorm without learnable parameters (elementwise_affine=False)."""
    original_dtype = x.dtype
    x = ops.cast(x, DType.float32)
    mean = ops.mean(x, axis=-1)
    x = x - mean
    var = ops.mean(x * x, axis=-1)
    x = x * ops.rsqrt(var + 1e-5)
    return ops.cast(x, original_dtype)


class WanAnimatePreProcess(Module):
    """Animate pre-processing: patch embed + pose injection + CLIP embed.

    Extends WanTransformerPreProcess with:
    - pose_patch_embedding Conv3d
    - image_embedder (GEGLU FFN for CLIP -> model dim)
    - Returns image_embeds as additional output for block cross-attention
    """

    def __init__(
        self,
        config: WanConfigBase,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim

        self.patch_embedding = WanConv3d(
            kernel_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=dim,
            stride=config.patch_size,
            dtype=dtype,
            device=device,
        )
        self.pose_patch_embedding = WanAnimatePosePatchEmbedding(
            config, dtype=dtype, device=device,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            num_layers=config.num_layers,
            image_dim=config.image_dim,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        timestep: TensorValue,
        encoder_hidden_states: TensorValue,
        clip_features: TensorValue,
        pose_hidden_states: TensorValue,
    ) -> tuple[
        TensorValue, TensorValue, TensorValue, TensorValue, TensorValue
    ]:
        batch_size = hidden_states.shape[0]

        # Patch embedding on main input [B, 36, T+1, H, W]
        hs = ops.permute(hidden_states, [0, 2, 3, 4, 1])
        hs = self.patch_embedding(hs)
        hs = ops.permute(hs, [0, 4, 1, 2, 3])  # [B, D, ppf, pph, ppw]

        # Pose patch embedding [B, 16, T, H, W]
        pose_embed = self.pose_patch_embedding.forward(
            pose_hidden_states
        )  # [B, D, ppf_pose, pph, ppw]

        # Add pose to frames 1+ (skip frame 0 = reference image)
        # hs[:, :, 1:] += pose_embed
        hs_ref = hs[:, :, :1, :, :]  # [B, D, 1, pph, ppw]
        hs_rest = hs[:, :, 1:, :, :]  # [B, D, ppf-1, pph, ppw]
        # Rebind hs_rest to match pose_embed shape (ppf-1 == pose_frames at runtime)
        hs_rest = ops.rebind(hs_rest, shape=[
            hs_rest.shape[0], hs_rest.shape[1],
            pose_embed.shape[2],  # use pose_frames dim
            hs_rest.shape[3], hs_rest.shape[4],
        ])
        hs_rest = hs_rest + pose_embed
        hs = ops.concat([hs_ref, hs_rest], axis=2)

        # Flatten spatial: [B, D, ppf, pph, ppw] -> [B, S, D]
        seq_len = hs.shape[2] * hs.shape[3] * hs.shape[4]
        hs = ops.reshape(hs, [batch_size, self.inner_dim, seq_len])
        hs = ops.permute(hs, [0, 2, 1])

        # Condition embeddings (timestep + text)
        temb, timestep_proj, text_emb = self.condition_embedder(
            timestep, encoder_hidden_states
        )

        # Image embeddings (CLIP -> model dim via GEGLU FFN)
        assert self.condition_embedder.image_embedder is not None
        image_embeds = self.condition_embedder.image_embedder(clip_features)

        return hs, temb, timestep_proj, text_emb, image_embeds


# ---------------------------------------------------------------------------
# MAX-native Motion Encoder (Phase 7)
# ---------------------------------------------------------------------------

# Channel sizes for the StyleGAN2-based appearance encoder.
_MOTION_ENCODER_CHANNEL_SIZES: dict[int, int] = {
    4: 512, 8: 512, 16: 512, 32: 512,
    64: 256, 128: 128, 256: 64, 512: 32,
}


class _MotionActivation(Module):
    """FusedLeakyReLU: channel bias + leaky_relu(0.2) * sqrt(2).

    Weight key: ``bias`` (matching diffusers ``act_fn.bias``).
    """

    def __init__(
        self,
        channels: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.bias = Weight("bias", dtype, [channels], device)

    def __call__(self, x: TensorValue) -> TensorValue:
        x = x + self.bias
        # leaky_relu(x, 0.2) * sqrt(2)
        # Use x*0 to get a zero tensor matching shape/dtype/device.
        zero = x * 0
        pos_scale = float(2.0**0.5)
        neg_scale = float(0.2 * 2.0**0.5)
        return ops.where(x > zero, x * pos_scale, x * neg_scale)


class _MotionConv2d(Module):
    """Conv2d with pre-baked weight scaling, optional FIR blur, optional activation.

    Expects NHWC input. Weight stored in RSCF layout ``[K, K, C_in, C_out]``
    with ``1/sqrt(fan_in)`` scaling already applied at load time.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        *,
        has_blur: bool = False,
        has_activation: bool = True,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self._stride = (stride, stride)
        self._padding = (padding, padding, padding, padding)
        self._has_blur = has_blur
        self._has_activation = has_activation

        # Weight in RSCF: [K, K, C_in, C_out] (scale already baked in)
        self.weight = Weight(
            "weight", dtype, [kernel_size, kernel_size, in_channels, out_channels], device,
        )

        if has_activation:
            self.act_fn = _MotionActivation(out_channels, dtype=dtype, device=device)

        if has_blur:
            self._blur_in_ch = in_channels
            # Blur filter as a Weight (gets placed on correct device).
            # Values are injected during weight preprocessing.
            self.blur_filter = Weight(
                "blur_filter", dtype, [4, 4, 1, in_channels], device,
            )
            # Blur padding: p = (blur_k_len - stride) + (kernel_size - 1)
            blur_k_len = 4
            p = (blur_k_len - stride) + (kernel_size - 1)
            ph_before = (p + 1) // 2
            ph_after = p // 2
            self._blur_padding = (ph_before, ph_after, ph_before, ph_after)

    def __call__(self, x: TensorValue) -> TensorValue:
        if self._has_blur:
            x = ops.conv2d(
                x, self.blur_filter,
                stride=(1, 1),
                padding=self._blur_padding,
                groups=self._blur_in_ch,
            )

        x = ops.conv2d(x, self.weight, stride=self._stride, padding=self._padding)

        if self._has_activation:
            x = self.act_fn(x)
        return x


class _MotionResBlock(Module):
    """Residual block: ``(conv2(conv1(x)) + skip(x)) / sqrt(2)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        # conv1: 3×3, stride=1, padding=1, activation
        self.conv1 = _MotionConv2d(
            in_channels, in_channels, 3, stride=1, padding=1,
            has_blur=False, has_activation=True, dtype=dtype, device=device,
        )
        # conv2: 3×3, stride=2, padding=0, blur + activation
        self.conv2 = _MotionConv2d(
            in_channels, out_channels, 3, stride=2, padding=0,
            has_blur=True, has_activation=True, dtype=dtype, device=device,
        )
        # conv_skip: 1×1, stride=2, padding=0, blur, no activation
        self.conv_skip = _MotionConv2d(
            in_channels, out_channels, 1, stride=2, padding=0,
            has_blur=True, has_activation=False, dtype=dtype, device=device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_skip = self.conv_skip(x)
        inv_sqrt2 = float(1.0 / 2.0**0.5)
        return (x_out + x_skip) * inv_sqrt2


class _MotionLinear(Module):
    """Linear layer with pre-baked ``1/sqrt(fan_in)`` scaling.

    Weight stored as ``[in_dim, out_dim]`` (transposed at load time).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.weight = Weight("weight", dtype, [in_dim, out_dim], device)
        self.bias = Weight("bias", dtype, [out_dim], device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.matmul(x, self.weight) + self.bias


class WanAnimateMotionEncoder(Module):
    """MAX-native motion encoder replacing the PyTorch bridge.

    StyleGAN2-based CNN: face crops ``[B, 3, 512, 512]`` → motion vectors
    ``[B, 512]``.

    Weights must be pre-processed before loading:

    - Conv weights: ``OIHW → RSCF``, scale ``1/sqrt(fan_in)`` baked in.
    - Linear weights: ``[out, in] → [in, out]``, scale baked in.
    - ``motion_synthesis_weight`` replaced by pre-computed ``q_matrix``
      from QR decomposition.
    - Indexed keys remapped: ``res_blocks.{i}. → res_blocks_{i}.``,
      ``motion_network.{i}. → motion_network_{i}.``.
    """

    def __init__(
        self,
        size: int = 512,
        style_dim: int = 512,
        motion_dim: int = 20,
        out_dim: int = 512,
        motion_blocks: int = 5,
        channels: dict[int, int] | None = None,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        if channels is None:
            channels = _MOTION_ENCODER_CHANNEL_SIZES

        # conv_in: Conv2d(3 → channels[size], k=1) + activation
        init_ch = channels[size]  # 32
        self.conv_in = _MotionConv2d(
            3, init_ch, 1, stride=1, padding=0,
            has_blur=False, has_activation=True, dtype=dtype, device=device,
        )

        # Res blocks: progressive spatial downsampling 512 → 4
        log_size = int(math.log(size, 2))  # 9
        self._num_res_blocks = log_size - 2  # 7
        in_ch = init_ch
        for i in range(log_size, 2, -1):
            out_ch = channels[2 ** (i - 1)]
            block_idx = log_size - i
            setattr(
                self, f"res_blocks_{block_idx}",
                _MotionResBlock(in_ch, out_ch, dtype=dtype, device=device),
            )
            in_ch = out_ch

        # conv_out: Conv2d(512 → style_dim, k=4), no bias, no activation
        self.conv_out = _MotionConv2d(
            in_ch, style_dim, 4, stride=1, padding=0,
            has_blur=False, has_activation=False, dtype=dtype, device=device,
        )

        # Motion network: linear layers (no intermediate activations)
        self._num_linears = motion_blocks
        for i in range(motion_blocks - 1):
            setattr(
                self, f"motion_network_{i}",
                _MotionLinear(style_dim, style_dim, dtype=dtype, device=device),
            )
        setattr(
            self, f"motion_network_{motion_blocks - 1}",
            _MotionLinear(style_dim, motion_dim, dtype=dtype, device=device),
        )

        # Q matrix from QR decomposition (pre-computed, float32 for precision)
        self.q_matrix = Weight("q_matrix", DType.float32, [out_dim, motion_dim], device)

    def __call__(self, face_image: TensorValue) -> TensorValue:
        """Encode face crops to motion vectors.

        Args:
            face_image: ``[B, 3, 512, 512]`` in ``[-1, 1]`` (NCHW).

        Returns:
            ``[B, 512]`` motion vectors.
        """
        # NCHW → NHWC
        x = ops.permute(face_image, [0, 2, 3, 1])

        # Appearance encoding through conv layers
        x = self.conv_in(x)
        for i in range(self._num_res_blocks):
            x = getattr(self, f"res_blocks_{i}")(x)
        x = self.conv_out(x)  # [B, 1, 1, style_dim]

        # Squeeze spatial dims → [B, style_dim]
        x = ops.reshape(x, [x.shape[0], x.shape[3]])

        # Motion feature extraction through linear layers
        for i in range(self._num_linears):
            x = getattr(self, f"motion_network_{i}")(x)
        # x: [B, motion_dim=20]

        # Motion synthesis via Linear Motion Decomposition:
        #   diag_embed(x) @ Q.T, summed over dim=1, simplifies to x @ Q.T.
        x = ops.cast(x, DType.float32)
        q_t = ops.permute(self.q_matrix, [1, 0])  # [motion_dim, out_dim]
        motion_vec = ops.matmul(x, q_t)  # [B, out_dim=512]
        return ops.cast(motion_vec, DType.bfloat16)


class WanAnimateMotionEncoderBridge:
    """PyTorch bridge for the StyleGAN2-based motion encoder.

    Loads the motion encoder sub-model from the diffusers checkpoint
    and runs get_motion() via PyTorch. This runs once per segment
    (not per denoising step), so performance is not critical.
    """

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        import torch
        from diffusers import WanAnimateTransformer3DModel

        logger.info("Loading motion encoder from %s", model_path)
        # Build model on CPU (not meta) to preserve computed buffers
        # like blur_kernel (FIR antialiasing filter, not in checkpoint).
        diffusers_model = WanAnimateTransformer3DModel.from_config(
            WanAnimateTransformer3DModel.load_config(
                model_path, subfolder="transformer"
            )
        )
        # Load only motion_encoder weights from safetensors
        from safetensors.torch import load_file
        from pathlib import Path

        transformer_dir = Path(model_path)
        if not transformer_dir.is_dir():
            from huggingface_hub import snapshot_download
            transformer_dir = Path(snapshot_download(model_path))
        transformer_dir = transformer_dir / "transformer"

        motion_state_dict: dict[str, torch.Tensor] = {}
        for sf_path in sorted(transformer_dir.glob("*.safetensors")):
            sd = load_file(str(sf_path), device="cpu")
            for k, v in sd.items():
                if k.startswith("motion_encoder."):
                    motion_state_dict[k] = v

        me_state = {
            k[len("motion_encoder."):]: v
            for k, v in motion_state_dict.items()
        }
        # strict=False: blur_kernel buffers are computed at init, not in checkpoint
        diffusers_model.motion_encoder.load_state_dict(me_state, strict=False)
        self.motion_encoder = diffusers_model.motion_encoder.to(
            device=device, dtype=torch.bfloat16
        ).eval()
        self.device = device

    def encode(
        self, face_pixels: np.ndarray, batch_size: int = 8
    ) -> np.ndarray:
        """Encode face crops to motion vectors.

        Args:
            face_pixels: [T, 3, 512, 512] float32 in [-1, 1].

        Returns:
            [T, 512] float32 motion vectors.
        """
        import torch  
        num_frames = face_pixels.shape[0]
        all_motions = []

        with torch.no_grad():
            for start in range(0, num_frames, batch_size):
                end = min(start + batch_size, num_frames)
                batch = torch.from_numpy(face_pixels[start:end]).to(
                    device=self.device, dtype=torch.bfloat16,
                )
                motion = self.motion_encoder(batch)
                all_motions.append(motion.float().cpu().numpy())

        return np.concatenate(all_motions, axis=0)  # [T, 512]


class CLIPImageEncoderBridge:
    """PyTorch bridge for CLIP ViT-H/14 image encoder.

    Loads from the diffusers checkpoint's image_encoder subfolder.
    Runs once per generation — not performance critical.
    """

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        from transformers import CLIPImageProcessor, CLIPVisionModel  
        logger.info("Loading CLIP image encoder from %s", model_path)
        self.model = CLIPVisionModel.from_pretrained(
            model_path, subfolder="image_encoder",
        ).to(device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(
            model_path, subfolder="image_processor",
        )
        self.device = device

    def encode(self, image: Any) -> np.ndarray:
        """Encode image to CLIP features.

        Args:
            image: PIL Image or numpy array.

        Returns:
            [1, 257, 1280] float32 (last_hidden_state).
        """
        import torch  
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=self.device, dtype=self.model.dtype,
        )
        with torch.no_grad():
            outputs = self.model(pixel_values)
        return outputs.last_hidden_state.float().cpu().numpy()
