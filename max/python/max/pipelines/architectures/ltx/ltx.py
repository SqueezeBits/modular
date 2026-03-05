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

import math

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.nn.module_v3 import Linear, Module
from max.nn.module_v3.norm import LayerNorm, RMSNorm
from max.nn.module_v3.sequential import ModuleList

from ..flux1.layers.activations import GELU
from ..flux1.layers.embeddings import PixArtAlphaTextProjection, Timesteps, TimestepEmbedding
from .model_config import LTXConfigBase


def _apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply pair-wise rotary embedding for tensors shaped [B, S, D]."""
    # [B, S, D] -> [B, S, D/2, 2]
    half_dim = int(x.shape[-1]) // 2
    x = x.reshape((x.shape[0], x.shape[1], half_dim, 2))
    x_real = x[:, :, :, 0]
    x_imag = x[:, :, :, 1]

    x_rot = F.stack([-x_imag, x_real], axis=-1)
    x_rot = x_rot.reshape((x_rot.shape[0], x_rot.shape[1], half_dim * 2))

    out = (
        F.cast(x.reshape((x.shape[0], x.shape[1], half_dim * 2)), DType.float32)
        * F.cast(cos, DType.float32)
        + F.cast(x_rot, DType.float32) * F.cast(sin, DType.float32)
    ).cast(x.dtype)
    return out


class _Identity(Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        return x


class _RMSNormNoAffine(Module[[Tensor], Tensor]):
    """RMSNorm without learnable affine parameters."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_f32 = x.cast(DType.float32)
        eps = F.constant(self.eps, dtype=DType.float32, device=x.device)
        variance = F.mean(x_f32 * x_f32, axis=-1)
        if len(variance.shape) < len(x.shape):
            variance = F.unsqueeze(variance, -1)
        inv_rms = F.rsqrt(variance + eps)
        return (x_f32 * inv_rms).cast(x.dtype)


class LTXFeedForward(Module[[Tensor], Tensor]):
    """Diffusers-compatible FFN naming/layout: net.0, net.1, net.2."""

    def __init__(self, dim: int, activation_fn: str = "gelu-approximate") -> None:
        super().__init__()
        if activation_fn != "gelu-approximate":
            raise NotImplementedError(
                f"Unsupported activation_fn={activation_fn!r} for LTX feed-forward"
            )

        inner_dim = dim * 4
        self.net: ModuleList[Module[..., Tensor]] = ModuleList(
            [
                GELU(dim, inner_dim, approximate="tanh", bias=True),
                _Identity(),
                Linear(inner_dim, dim, bias=True),
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class LTXAttention(Module[..., Tensor]):
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        cross_attention_dim: int | None = None,
        bias: bool = True,
        out_bias: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )

        # Diffusers LTX normalizes across all heads (inner_dim), not per-head.
        self.norm_q = RMSNorm(self.inner_dim, eps=1e-5)
        self.norm_k = RMSNorm(self.inner_dim, eps=1e-5)

        self.to_q = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_out: ModuleList[Linear] = ModuleList(
            [Linear(self.inner_dim, query_dim, bias=out_bias)]
        )

    @staticmethod
    def _prepare_attention_bias(
        encoder_attention_mask: Tensor | None,
        hidden_dtype: DType,
    ) -> Tensor | None:
        if encoder_attention_mask is None:
            return None

        # [B, T] bool -> [B, 1, 1, T] additive bias in hidden dtype.
        _ = hidden_dtype
        mask = encoder_attention_mask.cast(DType.float32)
        mask = (1.0 - mask) * -10000.0
        mask = F.unsqueeze(mask, 1)
        mask = F.unsqueeze(mask, 1)
        return mask

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query = _apply_rotary_emb(query, cos, sin)
            key = _apply_rotary_emb(key, cos, sin)

        # [B, S, D] -> [B, H, S, Dh]
        query = query.reshape((batch_size, query.shape[1], self.heads, self.head_dim))
        key = key.reshape((batch_size, key.shape[1], self.heads, self.head_dim))
        value = value.reshape((batch_size, value.shape[1], self.heads, self.head_dim))

        query = F.permute(query, (0, 2, 1, 3))
        key = F.permute(key, (0, 2, 1, 3))
        value = F.permute(value, (0, 2, 1, 3))

        # Match diffusers attention numerics by performing score/value matmuls in fp32.
        query_f32 = F.cast(query, DType.float32)
        key_f32 = F.cast(key, DType.float32)
        value_f32 = F.cast(value, DType.float32)

        scores = F.matmul(query_f32, F.permute(key_f32, (0, 1, 3, 2)))
        scores = scores * math.sqrt(1.0 / self.head_dim)

        attn_bias = self._prepare_attention_bias(attention_mask, hidden_states.dtype)
        if attn_bias is not None:
            scores = scores + attn_bias

        attn = F.softmax(scores, axis=-1)
        hidden_states = F.matmul(attn, value_f32).cast(query.dtype)

        hidden_states = F.permute(hidden_states, (0, 2, 1, 3))
        hidden_states = hidden_states.reshape(
            (batch_size, hidden_states.shape[1], self.inner_dim)
        )

        hidden_states = self.to_out[0](hidden_states)
        return hidden_states


class LTXVideoTransformerBlock(Module[..., Tensor]):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        activation_fn: str,
        attention_bias: bool,
        attention_out_bias: bool,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = _RMSNormNoAffine(eps=eps)
        self.attn1 = LTXAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        self.norm2 = _RMSNormNoAffine(eps=eps)
        self.attn2 = LTXAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        self.ff = LTXFeedForward(dim, activation_fn=activation_fn)
        self.scale_shift_table = Tensor.zeros((6, dim))

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]

        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = int(self.scale_shift_table.shape[0])
        ada_values = self.scale_shift_table.reshape((1, 1, num_ada_params, -1)) + temb.reshape(
            (batch_size, temb.shape[1], num_ada_params, -1)
        )
        shift_msa = ada_values[:, :, 0, :]
        scale_msa = ada_values[:, :, 1, :]
        gate_msa = ada_values[:, :, 2, :]
        shift_mlp = ada_values[:, :, 3, :]
        scale_mlp = ada_values[:, :, 4, :]
        gate_mlp = ada_values[:, :, 5, :]

        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            image_rotary_emb=None,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp
        return hidden_states


class LTXPixArtAlphaCombinedTimestepSizeEmbeddings(Module[[Tensor], Tensor]):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def forward(
        self,
        timestep: Tensor,
        hidden_dtype: DType,
    ) -> Tensor:
        timestep_proj = self.time_proj(timestep.cast(DType.float32))
        return self.timestep_embedder(timestep_proj.cast(hidden_dtype))


class LTXAdaLayerNormSingle(Module[..., tuple[Tensor, Tensor]]):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.emb = LTXPixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim)
        self.linear = Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: Tensor,
        hidden_dtype: DType,
    ) -> tuple[Tensor, Tensor]:
        embedded_timestep = self.emb(
            timestep=timestep,
            hidden_dtype=hidden_dtype,
        )
        temb = self.linear(F.silu(embedded_timestep))
        return temb, embedded_timestep


class LTXVideoTransformer3DModel(Module[..., tuple[Tensor]]):
    """MAX-native implementation of the LTX transformer denoiser."""

    def __init__(self, config: LTXConfigBase) -> None:
        super().__init__()
        self.config = config

        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )

        self.proj_in = Linear(
            self.config.in_channels,
            self.inner_dim,
            bias=True,
        )
        self.scale_shift_table = Tensor.zeros(
            (2, self.inner_dim), dtype=self.config.dtype
        )
        self.time_embed = LTXAdaLayerNormSingle(self.inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=self.config.caption_channels,
            hidden_size=self.inner_dim,
            act_fn="gelu_tanh",
        )

        self.transformer_blocks = ModuleList(
            [
                LTXVideoTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    attention_out_bias=self.config.attention_out_bias,
                    eps=self.config.norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.norm_out = LayerNorm(
            self.inner_dim,
            eps=1e-6,
            keep_dtype=True,
            elementwise_affine=False,
            use_bias=False,
        )
        self.proj_out = Linear(
            self.inner_dim,
            self.config.out_channels,
            bias=True,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.config.dtype,
                shape=["batch", "video_seq", self.config.in_channels],
                device=self.config.device,
            ),
            TensorType(
                self.config.dtype,
                shape=["batch", "text_seq", self.config.caption_channels],
                device=self.config.device,
            ),
            TensorType(
                DType.float32,
                shape=["batch", "video_seq"],
                device=self.config.device,
            ),
            TensorType(
                DType.bool,
                shape=["batch", "text_seq"],
                device=self.config.device,
            ),
            TensorType(
                DType.float32,
                shape=["batch", "video_seq", self.inner_dim],
                device=self.config.device,
            ),
            TensorType(
                DType.float32,
                shape=["batch", "video_seq", self.inner_dim],
                device=self.config.device,
            ),
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        encoder_attention_mask: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
    ) -> tuple[Tensor]:
        batch_size = hidden_states.shape[0]

        hidden_states = self.proj_in(hidden_states)

        timestep = F.reshape(timestep.cast(DType.float32), (-1,))
        temb, embedded_timestep = self.time_embed(
            timestep=timestep,
            hidden_dtype=hidden_states.dtype,
        )

        if temb.rank == 2:
            temb = temb.reshape((batch_size, -1, temb.shape[-1]))
        if embedded_timestep.rank == 2:
            embedded_timestep = embedded_timestep.reshape(
                (batch_size, -1, embedded_timestep.shape[-1])
            )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.reshape(
            (batch_size, -1, hidden_states.shape[-1])
        )

        image_rotary_emb = (rotary_cos, rotary_sin)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
            )

        scale_shift_values = self.scale_shift_table.reshape((1, 1, 2, -1)) + F.unsqueeze(
            embedded_timestep, 2
        )
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        return (output,)
