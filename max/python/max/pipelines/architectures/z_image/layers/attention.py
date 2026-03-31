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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from ...flux2.layers.embeddings import apply_rotary_emb


class ZImageAttention(Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        qk_norm: bool,
        eps: float,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialize ZImageAttention.

        Args:
            dim: Model dimension.
            n_heads: Number of attention heads.
            qk_norm: Whether to apply RMSNorm to queries and keys.
            eps: Epsilon for RMSNorm.
            dtype: Weight dtype.
            device: Weight device.
        """
        super().__init__()
        self.head_dim = dim // n_heads
        self.inner_dim = dim
        self.n_heads = n_heads

        self.to_q = Linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.to_k = Linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.to_v = Linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

        self.norm_q: RMSNorm | None = (
            RMSNorm(self.head_dim, dtype=dtype, eps=eps) if qk_norm else None
        )
        self.norm_k: RMSNorm | None = (
            RMSNorm(self.head_dim, dtype=dtype, eps=eps) if qk_norm else None
        )

        # Keep LayerList naming for diffusers-compatible key loading.
        self.to_out = LayerList(
            [
                Linear(
                    in_dim=dim,
                    out_dim=dim,
                    dtype=dtype,
                    device=device,
                    has_bias=False,
                )
            ]
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        freqs_cis: tuple[TensorValue, TensorValue],
    ) -> TensorValue:
        """Apply self-attention with rotary position embeddings.

        Args:
            hidden_states: Input tensor of shape [B, S, dim].
            freqs_cis: Tuple of (cos, sin) tensors for RoPE.

        Returns:
            Output tensor of shape [B, S, dim].
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = ops.reshape(
            query, [batch_size, seq_len, self.n_heads, self.head_dim]
        )
        key = ops.reshape(
            key, [batch_size, seq_len, self.n_heads, self.head_dim]
        )
        value = ops.reshape(
            value, [batch_size, seq_len, self.n_heads, self.head_dim]
        )

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = apply_rotary_emb(
            query,
            freqs_cis,
            use_real=True,
            use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        key = apply_rotary_emb(
            key,
            freqs_cis,
            use_real=True,
            use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        query = ops.cast(query, value.dtype)
        key = ops.cast(key, value.dtype)

        out = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0 / (self.head_dim**0.5),
        )

        out = ops.reshape(out, [batch_size, seq_len, self.inner_dim])
        return self.to_out[0](out)
