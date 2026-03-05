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

"""Qwen2.5-VL encoder-only attention with bias support.

Unlike Qwen3 (bias=False), Qwen2.5 models use attention_bias=True in their
Q/K/V/O projections.
"""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu
from max.nn.module_v3 import Linear, Module
from max.nn.module_v3.common_layers.rotary_embedding import (
    RotaryEmbedding,
)

flash_attention_gpu = F.functional(_flash_attention_gpu)


class Qwen25VLEncoderAttention(Module[..., Tensor]):
    """Encoder-only attention with bias for Qwen2.5-VL.

    Key difference from Qwen3 EncoderAttention:
    - Uses bias=True in Q/K/V/O projections
    - No per-head Q/K normalization (Qwen2.5 doesn't use qk_norm)
    """

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        scale: float,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.scale = scale

        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        self.q_proj = Linear(hidden_size, q_dim, bias=attention_bias)
        self.k_proj = Linear(hidden_size, kv_dim, bias=attention_bias)
        self.v_proj = Linear(hidden_size, kv_dim, bias=attention_bias)
        self.o_proj = Linear(q_dim, hidden_size, bias=False)

    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        if n_rep == 1:
            return x

        seq_len = x.shape[0]
        n_kv_heads = x.shape[1]
        head_dim = x.shape[2]

        x = F.unsqueeze(x, 2)
        x = F.tile(x, [1, 1, n_rep, 1])
        x = F.reshape(x, (seq_len, n_kv_heads * n_rep, head_dim))

        return x

    def forward(self, x: Tensor, rope: RotaryEmbedding) -> Tensor:
        """Forward pass computing causal self-attention.

        Args:
            x: Input tensor with shape [total_seq_len, hidden_dim]
            rope: RotaryEmbedding module

        Returns:
            Output tensor with shape [total_seq_len, hidden_dim]
        """
        total_seq_len = x.shape[0]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = F.reshape(q, (total_seq_len, self.n_heads, self.head_dim))
        k = F.reshape(k, (total_seq_len, self.n_kv_heads, self.head_dim))
        v = F.reshape(v, (total_seq_len, self.n_kv_heads, self.head_dim))

        # Apply RoPE (no QK norm for Qwen2.5)
        q = F.squeeze(rope(F.unsqueeze(q, 0)), 0)
        k = F.squeeze(rope(F.unsqueeze(k, 0)), 0)

        # GQA: expand K, V if needed
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)

        # flash_attention_gpu expects [B, S, heads, head_dim]
        q = F.unsqueeze(q, 0)
        k = F.unsqueeze(k, 0)
        v = F.unsqueeze(v, 0)

        attn_out = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        attn_out = F.squeeze(attn_out, 0)
        attn_out = F.reshape(attn_out, (total_seq_len, -1))
        return self.o_proj(attn_out)
