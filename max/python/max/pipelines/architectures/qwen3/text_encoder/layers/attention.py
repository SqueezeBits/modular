# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Encoder-only attention without KV cache."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu

from .rotary_embedding import RotaryEmbedding

flash_attention_gpu = F.functional(_flash_attention_gpu)


class EncoderAttention(Module[..., Tensor]):
    """Encoder-only attention without KV cache (Qwen3: interleaved RoPE via rope.forward)."""

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        scale: float,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.scale = scale

        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        self.q_proj = Linear(hidden_size, q_dim, bias=False)
        self.k_proj = Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = Linear(q_dim, hidden_size, bias=False)

        # Qwen3: Q/K norm over head_dim before RoPE (eps matches config.rms_norm_eps)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """Repeat KV heads for GQA (Grouped Query Attention).

        Args:
            x: Input tensor with shape [batch, seq_len, n_kv_heads, head_dim]
            n_rep: Number of times to repeat each head

        Returns:
            Tensor with shape [batch, seq_len, n_kv_heads * n_rep, head_dim]
        """
        if n_rep == 1:
            return x

        batch = x.shape[0]
        seq_len = x.shape[1]
        n_kv_heads = x.shape[2]
        head_dim = x.shape[3]

        # [B, S, H_kv, D] -> [B, S, H_kv, 1, D] -> [B, S, H_kv, n_rep, D]
        # -> [B, S, H, D]
        x = F.unsqueeze(x, 3)
        x = F.tile(x, [1, 1, 1, n_rep, 1])
        x = F.reshape(x, (batch, seq_len, n_kv_heads * n_rep, head_dim))

        return x

    def forward(
        self, x: Tensor, rope: RotaryEmbedding, valid_length: Tensor
    ) -> Tensor:
        """Forward pass computing causal self-attention.

        Args:
            x: Input tensor with shape [batch, total_seq_len, hidden_dim]
            rope: RotaryEmbedding module
            valid_length: Valid token count tensor with shape [batch]
        Returns:
            Output tensor with shape [batch, total_seq_len, hidden_dim]
        """
        batch = x.shape[0]
        total_seq_len = x.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = F.reshape(q, (batch, total_seq_len, self.n_heads, self.head_dim))
        k = F.reshape(k, (batch, total_seq_len, self.n_kv_heads, self.head_dim))
        v = F.reshape(v, (batch, total_seq_len, self.n_kv_heads, self.head_dim))

        # Qwen3: norm over head_dim (per-head), then RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        # GQA: expand K, V if needed
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)

        valid_length = F.rebind(valid_length, [batch])

        attn_out = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
            valid_length=valid_length,
        )

        attn_out = F.reshape(attn_out, (batch, total_seq_len, -1))
        return self.o_proj(attn_out)
