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

"""Mistral3 text encoder transformer without KV cache dependency.

This is a standalone transformer implementation for text encoding that does not
require KV cache. Suitable for single-pass encoding in diffusion pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.nn.embedding import Embedding
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding

from .attention import EncoderAttention

if TYPE_CHECKING:
    from .model_config import Mistral3TextEncoderConfigBase


class Mistral3MLP(Module):
    """Mistral3 MLP with SiLU gate activation."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, dtype: DType, device
    ) -> None:
        super().__init__()
        self.gate_proj = Linear(
            hidden_size, intermediate_size, dtype, device, has_bias=False
        )
        self.up_proj = Linear(
            hidden_size, intermediate_size, dtype, device, has_bias=False
        )
        self.down_proj = Linear(
            intermediate_size, hidden_size, dtype, device, has_bias=False
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        gate = ops.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class EncoderTransformerBlock(Module):
    """Transformer block for encoder-only models without KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        scale: float,
        dtype: DType,
        device,
    ) -> None:
        super().__init__()
        self.self_attn = EncoderAttention(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            scale=scale,
            dtype=dtype,
            device=device,
        )
        self.mlp = Mistral3MLP(hidden_size, intermediate_size, dtype, device)
        self.input_layernorm = RMSNorm(hidden_size, dtype=dtype, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, dtype=dtype, eps=rms_norm_eps
        )

    def __call__(self, x: TensorValue, rope: RotaryEmbedding) -> TensorValue:
        """Forward pass without KV cache.

        Args:
            x: Input hidden states [seq_len, hidden_dim]
            rope: RoPE embedding module

        Returns:
            Output hidden states [seq_len, hidden_dim]
        """
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, rope)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Mistral3TextEncoderTransformer(Module):
    """Mistral3 text encoder transformer without KV cache dependency.

    Encodes tokens and returns fused prompt embeddings by stacking hidden
    states from the configured layers and merging the layer/hidden dimensions.
    """

    def __init__(self, config: Mistral3TextEncoderConfigBase) -> None:
        super().__init__()

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.device = config.device
        self._hidden_state_layers = set(config.hidden_state_layers)
        self._sorted_hidden_state_layers = sorted(config.hidden_state_layers)

        self.rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.head_dim,
            interleaved=False,
        )

        self.layers = LayerList(
            [
                EncoderTransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                    scale=config.attention_multiplier,
                    dtype=config.dtype,
                    device=config.device,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.embed_tokens = Embedding(
            config.vocab_size,
            hidden_dim=config.hidden_size,
            dtype=config.dtype,
            device=config.device,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for compilation."""
        return (
            TensorType(
                DType.int64,
                shape=["total_seq_len"],
                device=self.device,
            ),
        )

    def __call__(self, tokens: TensorValue) -> tuple[TensorValue, ...]:
        """Forward pass returning fused prompt embeddings.

        Runs the transformer up to the last configured layer, collects hidden
        states from the configured layers, then stacks and reshapes them into
        a single prompt-embedding tensor.

        Args:
            tokens: Input token IDs [total_seq_len]

        Returns:
            Tensor of shape [1, seq_len, num_layers * hidden_dim] with the
            selected hidden states stacked and the layer/hidden dimensions
            merged, ready for the diffusion transformer.
        """
        h = self.embed_tokens(tokens)

        selected: dict[int, TensorValue] = {}
        max_layer = self._sorted_hidden_state_layers[-1]
        for i, layer in enumerate(self.layers):
            h = layer(h, self.rope)
            if i in self._hidden_state_layers:
                selected[i] = h
            if i == max_layer:
                break

        hidden_states = [selected[i] for i in self._sorted_hidden_state_layers]

        # Stack [L tensors of (S, D)] -> [L, S, D]
        # then fuse into [1, S, L*D] for the diffusion transformer.
        stacked = ops.stack(hidden_states, axis=0)  # [L, S, D]
        stacked = ops.unsqueeze(stacked, axis=0)  # [1, L, S, D]
        stacked = ops.permute(stacked, [0, 2, 1, 3])  # [1, S, L, D]
        # Read L and D directly from the tensor dims to avoid any Python-side
        # constant that could force a device sync at eager execution time.
        seq_len = stacked.shape[1]
        return (
            ops.reshape(
                stacked,
                [1, seq_len, stacked.shape[2] * stacked.shape[3]],
            ),
        )
