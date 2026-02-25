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

from math import prod

from max import functional as F
from max.dtype import DType
from max.nn import Conv3d, Linear, Module
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.legacy.kernels import flash_attention_gpu as _flash_attention_gpu
from max.nn.sequential import ModuleList
from max.tensor import Tensor

from ..flux2.layers.embeddings import (
    TimestepEmbedding,
    Timesteps,
    apply_rotary_emb,
)
from .model_config import WanConfigBase

flash_attention_gpu = F.functional(_flash_attention_gpu)


class WanLayerNorm(Module[[Tensor], Tensor]):
    """LayerNorm using decomposed ops for float32 numerical stability.

    The built-in ``layer_norm_gpu_block`` kernel hits
    ``CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`` for dim=5120, so we decompose
    into basic ops (mean, rsqrt, multiply) that each launch small kernels.
    """

    weight: Tensor | None
    bias: Tensor | None

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        *,
        elementwise_affine: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = Tensor.ones([dim])
            self.bias = Tensor.zeros([dim]) if use_bias else None
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype
        x = x.cast(DType.float32)
        mean = F.mean(x, axis=-1)
        x = x - mean
        var = F.mean(x * x, axis=-1)
        x = x * F.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight.cast(DType.float32)
            if self.bias is not None:
                x = x + self.bias.cast(DType.float32)
        return x.cast(original_dtype)


class WanRMSNorm(Module[[Tensor], Tensor]):
    """RMSNorm using decomposed ops for float32 numerical stability.

    Same reason as WanLayerNorm: the built-in ``rms_norm`` custom kernel
    may also hit resource limits for dim=5120.
    """

    weight: Tensor

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.weight = Tensor.ones([dim])
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype
        x = x.cast(DType.float32)
        rms = F.mean(x * x, axis=-1)
        x = x * F.rsqrt(rms + self.eps)
        x = x * self.weight.cast(DType.float32)
        return x.cast(original_dtype)


class WanTextProjection(Module[[Tensor], Tensor]):
    def __init__(self, in_features: int, hidden_size: int):
        self.linear_1 = Linear(in_features, hidden_size, bias=True)
        self.linear_2 = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, caption: Tensor) -> Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(Module[..., tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        dim: int,
        freq_dim: int,
        text_dim: int,
        num_layers: int,
    ):
        self.timesteps_proj = Timesteps(
            num_channels=freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=freq_dim, time_embed_dim=dim
        )
        # Projects SiLU(temb) to 6 modulation params per block
        self.time_proj = Linear(dim, dim * 6, bias=True)
        self.text_embedder = WanTextProjection(
            in_features=text_dim, hidden_size=dim
        )

    def forward(
        self, timestep: Tensor, encoder_hidden_states: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Sinusoidal timestep embedding (computed in float32 for precision).
        # Cast to the model's working dtype (bf16) for the MLP, matching
        # diffusers' behavior: float32 embedding → cast to weight dtype → MLP.
        timesteps_emb = self.timesteps_proj(timestep)  # [B, freq_dim] float32
        timesteps_emb = timesteps_emb.cast(encoder_hidden_states.dtype)  # → bf16
        temb = self.time_embedder(timesteps_emb)  # [B, dim]

        # Timestep projection for modulation: SiLU then linear
        timestep_proj = self.time_proj(F.silu(temb))  # [B, dim*6]
        # Reshape to [B, 6, dim] for per-block modulation
        timestep_proj = F.reshape(
            timestep_proj,
            [timestep_proj.shape[0], 6, timestep_proj.shape[1] // 6],
        )

        # Text projection
        text_emb = self.text_embedder(encoder_hidden_states)  # [B, S, dim]

        return temb, timestep_proj, text_emb


class WanSelfAttention(Module[..., Tensor]):
    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = dim

        self.to_q = Linear(dim, dim, bias=True)
        self.to_k = Linear(dim, dim, bias=True)
        self.to_v = Linear(dim, dim, bias=True)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)
        self.to_out = Linear(dim, dim, bias=True)

    def forward(
        self,
        hidden_states: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # QK-norm applied across all heads (before reshape)
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape to multi-head: [B, S, D] -> [B, S, H, head_dim]
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        query = F.reshape(
            query, [batch_size, seq_len, self.num_heads, self.head_dim]
        )
        key = F.reshape(
            key, [batch_size, seq_len, self.num_heads, self.head_dim]
        )
        value = F.reshape(
            value, [batch_size, seq_len, self.num_heads, self.head_dim]
        )

        # Apply RoPE
        original_dtype = query.dtype
        query = apply_rotary_emb(
            query, rotary_emb, use_real=True, use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        key = apply_rotary_emb(
            key, rotary_emb, use_real=True, use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        query = query.cast(original_dtype)
        key = key.cast(original_dtype)

        # Flash attention
        scale = 1.0 / (self.head_dim ** 0.5)
        hidden_states = flash_attention_gpu(
            query, key, value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )

        # Reshape back: [B, S, H, head_dim] -> [B, S, D]
        hidden_states = F.reshape(
            hidden_states,
            [hidden_states.shape[0], hidden_states.shape[1], self.inner_dim],
        )
        hidden_states = hidden_states.cast(original_dtype)

        return self.to_out(hidden_states)


class WanCrossAttention(Module[..., Tensor]):
    def __init__(
        self,
        dim: int,
        text_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = dim

        self.to_q = Linear(dim, dim, bias=True)
        # Fused K+V projection from text embeddings
        self.to_kv = Linear(text_dim, dim * 2, bias=True)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)
        self.to_out = Linear(dim, dim, bias=True)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
    ) -> Tensor:
        query = self.to_q(hidden_states)

        # Fused KV from text - use explicit slicing instead of F.chunk
        kv = self.to_kv(encoder_hidden_states)
        key = kv[:, :, :self.inner_dim]
        value = kv[:, :, self.inner_dim:]

        # QK-norm across all heads (before reshape)
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape to multi-head
        batch_size = query.shape[0]
        q_seq_len = query.shape[1]
        kv_seq_len = key.shape[1]
        query = F.reshape(
            query, [batch_size, q_seq_len, self.num_heads, self.head_dim]
        )
        key = F.reshape(
            key, [batch_size, kv_seq_len, self.num_heads, self.head_dim]
        )
        value = F.reshape(
            value, [batch_size, kv_seq_len, self.num_heads, self.head_dim]
        )

        # Flash attention (no RoPE for cross-attention)
        original_dtype = query.dtype
        scale = 1.0 / (self.head_dim ** 0.5)
        hidden_states = flash_attention_gpu(
            query, key, value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )

        # Reshape back
        hidden_states = F.reshape(
            hidden_states,
            [hidden_states.shape[0], hidden_states.shape[1], self.inner_dim],
        )
        hidden_states = hidden_states.cast(original_dtype)

        return self.to_out(hidden_states)


class WanFeedForward(Module[[Tensor], Tensor]):
    def __init__(self, dim: int, ffn_dim: int):
        # WAN uses "gelu-approximate" (simple GELU), NOT GEGLU.
        # ffn_dim is the direct projection output size (no 2x expansion).
        self.proj = Linear(dim, ffn_dim, bias=True)
        self.linear_out = Linear(ffn_dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.proj(x)
        hidden = F.gelu(hidden, approximate="tanh")
        return self.linear_out(hidden)


class WanTransformerBlock(Module[..., Tensor]):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        text_dim: int,
        cross_attn_norm: bool,
        eps: float,
    ):
        self.scale_shift_table = Tensor.zeros([1, 6, dim])
        self.norm1 = WanLayerNorm(
            dim, eps=eps, elementwise_affine=False,
        )
        self.attn1 = WanSelfAttention(dim, num_heads, head_dim, eps)
        self.norm2 = WanLayerNorm(
            dim, eps=eps,
            elementwise_affine=cross_attn_norm, use_bias=cross_attn_norm,
        )
        self.attn2 = WanCrossAttention(dim, text_dim, num_heads, head_dim, eps)
        self.norm3 = WanLayerNorm(
            dim, eps=eps, elementwise_affine=False,
        )
        self.ffn = WanFeedForward(dim, ffn_dim)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep_proj: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        rotary_emb = (rope_cos, rope_sin)

        # Modulation: scale_shift_table[1,6,D] + timestep_proj[B,6,D]
        mod = self.scale_shift_table + timestep_proj  # [B, 6, D]

        # Split into 6 modulation parameters
        shift_sa = mod[:, 0:1, :]   # [B, 1, D]
        scale_sa = mod[:, 1:2, :]
        gate_sa = mod[:, 2:3, :]
        shift_ff = mod[:, 3:4, :]
        scale_ff = mod[:, 4:5, :]
        gate_ff = mod[:, 5:6, :]

        # Self-attention
        x = self.norm1(hidden_states)
        x = x * (1 + scale_sa) + shift_sa
        x = self.attn1(x, rotary_emb)
        hidden_states = hidden_states + gate_sa * x

        # Cross-attention
        x = self.norm2(hidden_states)
        x = self.attn2(x, encoder_hidden_states)
        hidden_states = hidden_states + x

        # Feed-forward
        x = self.norm3(hidden_states)
        x = x * (1 + scale_ff) + shift_ff
        x = self.ffn(x)
        hidden_states = hidden_states + gate_ff * x

        return hidden_states


class WanTransformerPreProcess(Module[..., tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Patch embedding + condition embedding (compiled separately)."""

    def __init__(self, config: WanConfigBase) -> None:
        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim

        self.patch_embedding = Conv3d(
            kernel_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=dim,
            stride=config.patch_size,
            has_bias=True,
            permute=False,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            num_layers=config.num_layers,
        )

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = hidden_states.shape[0]
        # Conv3d(permute=False) expects NDHWC input/output.
        hs = F.permute(hidden_states, [0, 2, 3, 4, 1])
        hs = self.patch_embedding(hs)
        hs = F.permute(hs, [0, 4, 1, 2, 3])  # -> [B, D, ppf, pph, ppw]
        seq_len = hs.shape[2] * hs.shape[3] * hs.shape[4]
        hs = F.reshape(hs, [batch_size, self.inner_dim, seq_len])
        hs = F.permute(hs, [0, 2, 1])  # -> [B, S, D]

        temb, timestep_proj, text_emb = self.condition_embedder(
            timestep, encoder_hidden_states
        )
        return hs, temb, timestep_proj, text_emb


class WanTransformerPostProcess(Module[..., Tensor]):
    """Output modulation + unpatchify (compiled separately)."""

    def __init__(
        self,
        config: WanConfigBase,
        ppf: int,
        pph: int,
        ppw: int,
    ) -> None:
        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size
        self.ppf = ppf
        self.pph = pph
        self.ppw = ppw

        self.scale_shift_table = Tensor.zeros([1, 2, dim])
        self.norm_out = WanLayerNorm(
            dim, eps=config.eps, elementwise_affine=False,
        )
        self.proj_out = Linear(
            dim, config.out_channels * prod(config.patch_size), bias=True,
        )

    def forward(self, hidden_states: Tensor, temb: Tensor) -> Tensor:
        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = self.ppf, self.pph, self.ppw

        mod = self.scale_shift_table + F.reshape(
            temb, [batch_size, 1, self.inner_dim]
        )
        shift = mod[:, :1, :]
        scale = mod[:, 1:, :]
        hs = self.norm_out(hidden_states) * (1.0 + scale) + shift
        hs = self.proj_out(hs)

        # Unpatchify: [B, S, C*p_t*p_h*p_w] -> [B, C, T, H, W]
        hs = F.reshape(
            hs,
            [batch_size, ppf, pph, ppw, p_t, p_h, p_w, self.out_channels],
        )
        hs = F.permute(hs, [0, 7, 1, 4, 2, 5, 3, 6])
        hs = F.reshape(
            hs,
            [batch_size, self.out_channels, ppf * p_t, pph * p_h, ppw * p_w],
        )
        return hs.cast(DType.bfloat16)


class WanTransformer3DModel(Module[..., Tensor]):
    """Full transformer (for reference / single-graph compilation)."""

    def __init__(self, config: WanConfigBase):
        super().__init__()
        self.config = config
        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size

        self.patch_embedding = Conv3d(
            kernel_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=dim,
            stride=config.patch_size,
            has_bias=True,
            permute=False,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            num_layers=config.num_layers,
        )
        self.blocks: ModuleList[WanTransformerBlock] = ModuleList(
            [
                WanTransformerBlock(
                    dim=dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_attention_heads,
                    head_dim=config.attention_head_dim,
                    text_dim=dim,
                    cross_attn_norm=config.cross_attn_norm,
                    eps=config.eps,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.scale_shift_table = Tensor.zeros([1, 2, dim])
        self.norm_out = WanLayerNorm(
            dim, eps=config.eps, elementwise_affine=False,
        )
        self.proj_out = Linear(
            dim, config.out_channels * prod(config.patch_size), bias=True
        )

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        orig_T = hidden_states.shape[2]
        orig_H = hidden_states.shape[3]
        orig_W = hidden_states.shape[4]
        p_t, p_h, p_w = self.patch_size
        ppf = orig_T // p_t
        pph = orig_H // p_h
        ppw = orig_W // p_w

        hs = F.permute(hidden_states, [0, 2, 3, 4, 1])
        hs = self.patch_embedding(hs)
        hs = F.permute(hs, [0, 4, 1, 2, 3])
        hs = F.reshape(hs, [batch_size, self.inner_dim, ppf * pph * ppw])
        hs = F.permute(hs, [0, 2, 1])

        temb, timestep_proj, text_emb = self.condition_embedder(
            timestep, encoder_hidden_states
        )

        for block in self.blocks:
            hs = block(hs, text_emb, timestep_proj, rope_cos, rope_sin)

        mod = self.scale_shift_table + F.reshape(
            temb, [batch_size, 1, self.inner_dim]
        )
        shift = mod[:, :1, :]
        scale = mod[:, 1:, :]
        hs = self.norm_out(hs) * (1.0 + scale) + shift
        hs = self.proj_out(hs)

        hs = F.reshape(
            hs,
            [batch_size, ppf, pph, ppw, p_t, p_h, p_w, self.out_channels],
        )
        hs = F.permute(hs, [0, 7, 1, 4, 2, 5, 3, 6])
        hs = F.reshape(
            hs,
            [batch_size, self.out_channels, ppf * p_t, pph * p_h, ppw * p_w],
        )
        return hs.cast(self.config.dtype)
