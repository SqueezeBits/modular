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

"""Qwen2.5-VL text encoder transformer without KV cache dependency.

Standalone transformer implementation for text encoding in the QwenImage
diffusion pipeline. Returns hidden states from all layers.

Key differences from Qwen3TextEncoder:
- attention_bias=True (Qwen2.5 uses biased Q/K/V projections)
- No per-head Q/K normalization
- Different default dimensions (3584 hidden, 28 heads, 4 kv heads)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.nn.module_v3 import Embedding, Linear, Module
from max.nn.module_v3.norm import RMSNorm
from max.nn.module_v3.sequential import ModuleList

from max.nn.module_v3.common_layers.rotary_embedding import (
    RotaryEmbedding,
)

from .layers import Qwen25VLEncoderAttention

if TYPE_CHECKING:
    from .model_config import Qwen25VLTextEncoderConfigBase


class Qwen25VLMLP(Module[[Tensor], Tensor]):
    """Qwen2.5-VL MLP with SiLU gate activation."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, bias: bool = False
    ) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Qwen25VLEncoderTransformerBlock(Module[..., Tensor]):
    """Transformer block for Qwen2.5-VL encoder-only model."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        scale: float,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen25VLEncoderAttention(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            scale=scale,
            attention_bias=attention_bias,
        )
        self.mlp = Qwen25VLMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x: Tensor, rope: RotaryEmbedding) -> Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, rope)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Qwen25VLTextEncoderTransformer(Module[..., tuple[Tensor, ...]]):
    """Qwen2.5-VL text encoder transformer.

    Returns hidden states from all layers for use in the QwenImage pipeline.
    The pipeline uses the last hidden state (layer -1).
    """

    def __init__(self, config: Qwen25VLTextEncoderConfigBase) -> None:
        super().__init__()

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.device = config.device

        self.rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.device.to_device(),
            head_dim=config.head_dim,
            interleaved=False,  # HF Qwen2 uses rotate_half (non-interleaved)
        )

        self.layers = ModuleList(
            [
                Qwen25VLEncoderTransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                    scale=config.attention_multiplier,
                    attention_bias=config.attention_bias,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                DType.int64,
                shape=["total_seq_len"],
                device=self.device,
            ),
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, ...]:
        """Forward pass returning hidden states from all layers.

        Args:
            tokens: Input token IDs [total_seq_len]

        Returns:
            Tuple of hidden states from all layers plus final normed output.
            The last element is the normed output (matching HF's hidden_states[-1]).
        """
        h = self.embed_tokens(tokens)

        all_hidden_states: list[Tensor] = []
        for layer in self.layers:
            h = layer(h, self.rope)
            all_hidden_states.append(h)

        # Apply final RMSNorm (matches HF Qwen2Model which returns normed
        # output as the last hidden state)
        all_hidden_states.append(self.norm(h))

        return tuple(all_hidden_states)
