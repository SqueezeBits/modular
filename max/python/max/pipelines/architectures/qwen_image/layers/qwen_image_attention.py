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

"""QwenImage attention layers: dual-stream attention, FeedForward, and transformer block.

Weight key naming follows HuggingFace diffusers conventions:
- Attention: attn.to_q, attn.to_k, attn.to_v, attn.to_out.0, attn.add_q_proj, etc.
- FeedForward: img_mlp.net.0.proj (SwiGLU), img_mlp.net.2 (output linear)
- Modulation: img_mod.1 (Linear after SiLU), txt_mod.1
- Norms: img_norm1, img_norm2, txt_norm1, txt_norm2 (no affine, no weights)
"""

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu
from max.nn.module_v3 import Linear, Module, module_dataclass
from max.nn.module_v3.norm import LayerNorm, RMSNorm
from max.nn.module_v3.sequential import ModuleList

from max.pipelines.architectures.flux2.layers.embeddings import (
    apply_rotary_emb,
)

flash_attention_gpu = F.functional(_flash_attention_gpu)


# ---------------------------------------------------------------------------
# FeedForward (matches diffusers naming: net.0.proj, net.2)
# ---------------------------------------------------------------------------


class _QwenImageGELU(Module[[Tensor], Tensor]):
    """GELU approximate activation with a Linear projection.

    Weight key: `proj.weight`, `proj.bias`
    In the block: `img_mlp.net.0.proj.weight`
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        self.proj = Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.proj(x))


@module_dataclass
class _QwenImageDropout(Module[[Tensor], Tensor]):
    """No-op dropout for inference. Occupies index 1 in FeedForward.net."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class QwenImageFeedForward(Module[[Tensor], Tensor]):
    """FeedForward matching diffusers key naming.

    Produces keys:
        net.0.proj.weight, net.0.proj.bias  (GELU approximate projection)
        net.2.weight, net.2.bias            (output linear)
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4.0,
        inner_dim: int | None = None,
        bias: bool = True,
    ):
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        self.net: ModuleList = ModuleList(
            [
                _QwenImageGELU(dim, inner_dim, bias=bias),
                _QwenImageDropout(),
                Linear(inner_dim, dim_out, bias=bias),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net[0](x)  # SwiGLU
        # net[1] is dropout (no-op at inference)
        x = self.net[2](x)  # output linear
        return x


# ---------------------------------------------------------------------------
# Attention (matches diffusers key naming: to_q, to_k, to_v, to_out.0, ...)
# ---------------------------------------------------------------------------


class QwenImageAttention(Module[..., Tensor | tuple[Tensor, Tensor]]):
    """Dual-stream attention for QwenImage transformer blocks.

    Key naming matches HuggingFace diffusers:
    - to_q.weight/bias, to_k.weight/bias, to_v.weight/bias
    - to_out.0.weight/bias  (ModuleList for correct .0. indexing)
    - add_q_proj.weight/bias, add_k_proj.weight/bias, add_v_proj.weight/bias
    - to_add_out.weight/bias
    - norm_q.weight, norm_k.weight, norm_added_q.weight, norm_added_k.weight
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = True,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
    ):
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.scale = 1.0 / (self.head_dim**0.5)
        out_dim = out_dim if out_dim is not None else query_dim

        self.to_q = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = Linear(query_dim, self.inner_dim, bias=bias)

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # Use ModuleList so key becomes to_out.0.weight (not to_out_0.weight)
        self.to_out: ModuleList[Linear] = ModuleList()
        self.to_out.append(Linear(self.inner_dim, out_dim, bias=out_bias))

        self.norm_added_q: RMSNorm | None
        self.norm_added_k: RMSNorm | None
        self.add_q_proj: Linear | None
        self.add_k_proj: Linear | None
        self.add_v_proj: Linear | None
        self.to_add_out: Linear | None
        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            self.add_q_proj = Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_k_proj = Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_v_proj = Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.to_add_out = Linear(self.inner_dim, query_dim, bias=out_bias)
        else:
            self.norm_added_q = None
            self.norm_added_k = None
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None
            self.to_add_out = None

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        seq_len = query.shape[1]

        query = F.reshape(
            query, [batch_size, seq_len, self.heads, self.head_dim]
        )
        key = F.reshape(key, [batch_size, seq_len, self.heads, self.head_dim])
        value = F.reshape(
            value, [batch_size, seq_len, self.heads, self.head_dim]
        )

        query = self.norm_q(query)
        key = self.norm_k(key)

        if (
            encoder_hidden_states is not None
            and self.added_kv_proj_dim is not None
        ):
            if (
                self.add_q_proj is None
                or self.add_k_proj is None
                or self.add_v_proj is None
            ):
                raise ValueError("Encoder projections not initialized")
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)
            encoder_seq_len = encoder_query.shape[1]
            encoder_query = F.reshape(
                encoder_query,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )
            encoder_key = F.reshape(
                encoder_key,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )
            encoder_value = F.reshape(
                encoder_value,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )

            if self.norm_added_q is None or self.norm_added_k is None:
                raise ValueError("Encoder normalizations not initialized")
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = F.concat([encoder_query, query], axis=1)
            key = F.concat([encoder_key, key], axis=1)
            value = F.concat([encoder_value, value], axis=1)

        original_dtype = query.dtype
        if image_rotary_emb is not None:
            query = apply_rotary_emb(
                query,
                image_rotary_emb,
                use_real=True,
                use_real_unbind_dim=-1,
                sequence_dim=1,
            )
            key = apply_rotary_emb(
                key,
                image_rotary_emb,
                use_real=True,
                use_real_unbind_dim=-1,
                sequence_dim=1,
            )
            if query.dtype != original_dtype:
                query = query.cast(original_dtype)
            if key.dtype != original_dtype:
                key = key.cast(original_dtype)

        hidden_states = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale,
        )

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = F.reshape(
            hidden_states, [batch_size, seq_len, self.inner_dim]
        )
        if hidden_states.dtype != query.dtype:
            hidden_states = hidden_states.cast(query.dtype)

        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_out = hidden_states[:, :encoder_seq_len, :]
            hidden_out = hidden_states[:, encoder_seq_len:, :]

            hidden_out = self.to_out[0](hidden_out)
            if self.to_add_out is None:
                raise ValueError("Encoder output projection not initialized")
            encoder_out = self.to_add_out(encoder_out)

            return hidden_out, encoder_out
        else:
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states


# ---------------------------------------------------------------------------
# Per-block Modulation (matches diffusers: img_mod.1.weight, txt_mod.1.weight)
# ---------------------------------------------------------------------------


@module_dataclass
class _SiLUPlaceholder(Module[[Tensor], Tensor]):
    """Placeholder at index 0 in ModuleList; SiLU has no learnable params."""

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x)


def _make_block_modulation(dim: int, bias: bool = True) -> ModuleList:
    """Create per-block modulation as ModuleList[SiLU_placeholder, Linear].

    Produces weight keys: `{attr_name}.1.weight` and `{attr_name}.1.bias`
    matching the diffusers convention img_mod.1.weight / txt_mod.1.weight.
    """
    return ModuleList([_SiLUPlaceholder(), Linear(dim, dim * 6, bias=bias)])


# ---------------------------------------------------------------------------
# Transformer Block (per-block img_mod, txt_mod, img_mlp, txt_mlp)
# ---------------------------------------------------------------------------


class QwenImageTransformerBlock(Module[..., tuple[Tensor, Tensor]]):
    """Dual-stream transformer block with per-block modulation.

    Weight key structure per block:
        img_mod.1.{weight,bias}
        txt_mod.1.{weight,bias}
        attn.to_q.{weight,bias}, attn.to_k.{weight,bias}, ...
        img_mlp.net.0.proj.{weight,bias}, img_mlp.net.2.{weight,bias}
        txt_mlp.net.0.proj.{weight,bias}, txt_mlp.net.2.{weight,bias}
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        eps: float = 1e-6,
        bias: bool = True,
    ):
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # Per-block modulation (img_mod, txt_mod)
        # ModuleList[SiLU_placeholder, Linear(dim, 6*dim)]
        # Keys: img_mod.1.weight, img_mod.1.bias, txt_mod.1.weight, txt_mod.1.bias
        self.img_mod: ModuleList = _make_block_modulation(dim, bias=bias)
        self.txt_mod: ModuleList = _make_block_modulation(dim, bias=bias)

        # Norms (elementwise_affine=False → no weights in state_dict)
        self.img_norm1 = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )
        self.img_norm2 = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )
        self.txt_norm1 = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )
        self.txt_norm2 = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )

        # Dual-stream attention
        self.attn = QwenImageAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
        )

        # Feedforward (img_mlp, txt_mlp)
        self.img_mlp = QwenImageFeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias
        )
        self.txt_mlp = QwenImageFeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        # Compute per-block modulation params from temb
        # Compute silu once and reuse for both modulation projections.
        temb_activated = F.silu(temb)
        img_mod = self.img_mod[1](temb_activated)
        txt_mod = self.txt_mod[1](temb_activated)

        if len(img_mod.shape) == 2:
            img_mod = F.unsqueeze(img_mod, 1)
            txt_mod = F.unsqueeze(txt_mod, 1)
        img_mod_chunks = F.chunk(img_mod, 6, axis=-1)
        shift_msa, scale_msa, gate_msa = (
            img_mod_chunks[0],
            img_mod_chunks[1],
            img_mod_chunks[2],
        )
        shift_mlp, scale_mlp, gate_mlp = (
            img_mod_chunks[3],
            img_mod_chunks[4],
            img_mod_chunks[5],
        )

        txt_mod_chunks = F.chunk(txt_mod, 6, axis=-1)
        c_shift_msa, c_scale_msa, c_gate_msa = (
            txt_mod_chunks[0],
            txt_mod_chunks[1],
            txt_mod_chunks[2],
        )
        c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            txt_mod_chunks[3],
            txt_mod_chunks[4],
            txt_mod_chunks[5],
        )

        # Image stream - Attention
        norm_hidden_states = self.img_norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        # Text stream - Attention
        norm_encoder_hidden_states = self.txt_norm1(encoder_hidden_states)
        norm_encoder_hidden_states = (
            1 + c_scale_msa
        ) * norm_encoder_hidden_states + c_shift_msa

        # Dual-stream attention
        attn_result = self.attn.forward(
            norm_hidden_states,
            norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        if isinstance(attn_result, tuple):
            attn_output, context_attn_output = attn_result
        else:
            raise ValueError("Expected tuple from dual-stream attention")

        # Image stream - Apply gate and residual
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        # Image stream - Feedforward
        norm_hidden_states = self.img_norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.img_mlp(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        # Text stream - Apply gate and residual
        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        # Text stream - Feedforward
        norm_encoder_hidden_states = self.txt_norm2(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        )

        context_ff_output = self.txt_mlp(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp * context_ff_output
        )

        if encoder_hidden_states.dtype == DType.float16:
            encoder_hidden_states = encoder_hidden_states.clip(
                min=-65504, max=65504
            )

        return encoder_hidden_states, hidden_states
