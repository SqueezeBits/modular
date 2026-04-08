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

"""Ring (context-parallel) Flux2 attention layers.

Ring Attention shards the sequence across devices. Each device keeps its
local Q chunk and gathers the full K, V via allgather. Attention is then
computed locally: attention(Q_local, K_full, V_full). No output
communication is needed since each device's output corresponds exactly
to its Q chunk.

Compared to Ulysses:
  - Simpler: no input/output all-to-all, no head reshuffling
  - No head divisibility requirement (num_heads % num_devices not needed)
  - Same memory profile with allgather emulation

Note: True ring attention uses P2P KV rotation with online softmax to
avoid materializing full K, V. This implementation uses allgather as an
emulation until native P2P send/recv ops become available.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .flux2_attention import Flux2SwiGLU, _apply_flux2_qk_rope


def _replicate_linear(
    linear: Linear, num_devices: int, devices: list[DeviceRef]
) -> list[Linear]:
    linear.weight.sharding_strategy = ShardingStrategy.replicate(num_devices)
    return linear.shard(devices)


def _replicate_norm(
    norm: RMSNorm, num_devices: int, devices: list[DeviceRef]
) -> list[RMSNorm]:
    norm.sharding_strategy = ShardingStrategy.replicate(num_devices)
    return norm.shard(devices)


class RingFlux2Attention(Module):
    """Ring context-parallel dual-stream Flux2 attention.

    Each device holds a sequence chunk with full heads. K and V are
    allgathered so each device can compute attention over the full
    sequence for its local Q chunk. No output communication is needed.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        *,
        dtype: DType,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.num_devices = len(devices)
        n = self.num_devices

        out_dim = out_dim if out_dim is not None else query_dim

        # All projections replicated
        self.to_q = Linear(query_dim, self.inner_dim, dtype, devices[0], has_bias=bias)
        self.to_q_shards = _replicate_linear(self.to_q, n, devices)

        self.to_k = Linear(query_dim, self.inner_dim, dtype, devices[0], has_bias=bias)
        self.to_k_shards = _replicate_linear(self.to_k, n, devices)

        self.to_v = Linear(query_dim, self.inner_dim, dtype, devices[0], has_bias=bias)
        self.to_v_shards = _replicate_linear(self.to_v, n, devices)

        self.norm_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_q_shards = _replicate_norm(self.norm_q, n, devices)

        self.norm_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_k_shards = _replicate_norm(self.norm_k, n, devices)

        _to_out_linear = Linear(self.inner_dim, out_dim, dtype, devices[0], has_bias=out_bias)
        self.to_out = LayerList([_to_out_linear])
        self.to_out_shards = _replicate_linear(_to_out_linear, n, devices)

        # Optional encoder projections for dual-stream
        self.add_q_proj_shards: list[Linear] | None = None
        self.add_k_proj_shards: list[Linear] | None = None
        self.add_v_proj_shards: list[Linear] | None = None
        self.norm_added_q_shards: list[RMSNorm] | None = None
        self.norm_added_k_shards: list[RMSNorm] | None = None
        self.to_add_out_shards: list[Linear] | None = None

        if added_kv_proj_dim is not None:
            proj_bias = False if added_proj_bias is None else added_proj_bias

            self.add_q_proj = Linear(added_kv_proj_dim, self.inner_dim, dtype, devices[0], has_bias=proj_bias)
            self.add_q_proj_shards = _replicate_linear(self.add_q_proj, n, devices)

            self.add_k_proj = Linear(added_kv_proj_dim, self.inner_dim, dtype, devices[0], has_bias=proj_bias)
            self.add_k_proj_shards = _replicate_linear(self.add_k_proj, n, devices)

            self.add_v_proj = Linear(added_kv_proj_dim, self.inner_dim, dtype, devices[0], has_bias=proj_bias)
            self.add_v_proj_shards = _replicate_linear(self.add_v_proj, n, devices)

            self.norm_added_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
            self.norm_added_q_shards = _replicate_norm(self.norm_added_q, n, devices)

            self.norm_added_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
            self.norm_added_k_shards = _replicate_norm(self.norm_added_k, n, devices)

            self.to_add_out = Linear(self.inner_dim, query_dim, dtype, devices[0], has_bias=out_bias)
            self.to_add_out_shards = _replicate_linear(self.to_add_out, n, devices)

    def _per_device_qkv(
        self,
        i: int,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue | None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Project Q, K, V on device i, reshape to [B, S, H, D]."""
        batch_size = hidden_states.shape[0]

        query = self.to_q_shards[i](hidden_states)
        key = self.to_k_shards[i](hidden_states)
        value = self.to_v_shards[i](hidden_states)

        seq_len = query.shape[1]
        query = ops.reshape(query, [batch_size, seq_len, self.heads, self.head_dim])
        key = ops.reshape(key, [batch_size, seq_len, self.heads, self.head_dim])
        value = ops.reshape(value, [batch_size, seq_len, self.heads, self.head_dim])

        query = self.norm_q_shards[i](query)
        key = self.norm_k_shards[i](key)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            enc_q = self.add_q_proj_shards[i](encoder_hidden_states)
            enc_k = self.add_k_proj_shards[i](encoder_hidden_states)
            enc_v = self.add_v_proj_shards[i](encoder_hidden_states)
            enc_seq_len = enc_q.shape[1]

            enc_q = ops.reshape(enc_q, [batch_size, enc_seq_len, self.heads, self.head_dim])
            enc_k = ops.reshape(enc_k, [batch_size, enc_seq_len, self.heads, self.head_dim])
            enc_v = ops.reshape(enc_v, [batch_size, enc_seq_len, self.heads, self.head_dim])

            enc_q = self.norm_added_q_shards[i](enc_q)
            enc_k = self.norm_added_k_shards[i](enc_k)

            query = ops.concat([enc_q, query], axis=1)
            key = ops.concat([enc_k, key], axis=1)
            value = ops.concat([enc_v, value], axis=1)

        return query, key, value

    def __call__(
        self,
        hidden_states_list: list[TensorValue],
        signal_buffers: list[BufferValue],
        encoder_hidden_states_list: list[TensorValue] | None = None,
        image_rotary_emb_list: list[tuple[TensorValue, TensorValue]] | None = None,
    ) -> list[TensorValue] | tuple[list[TensorValue], list[TensorValue]]:
        n = self.num_devices

        # Step 1: Project Q, K, V on each device
        queries, keys, values = [], [], []
        for i in range(n):
            enc = encoder_hidden_states_list[i] if encoder_hidden_states_list is not None else None
            q, k, v = self._per_device_qkv(i, hidden_states_list[i], enc)
            queries.append(q)
            keys.append(k)
            values.append(v)

        # Step 2: Apply RoPE BEFORE allgather (position-dependent)
        if image_rotary_emb_list is not None:
            for i in range(n):
                cos, sin = image_rotary_emb_list[i]
                queries[i], keys[i] = _apply_flux2_qk_rope(
                    queries[i], keys[i], cos, sin
                )

        # Step 3: Allgather K and V along sequence axis (dim=1)
        # Q stays local — each device only needs output for its own chunk
        # [B, S_local, H, D] per device -> [B, S_global, H, D] on every device
        keys_full = ops.allgather(keys, signal_buffers, axis=1)
        values_full = ops.allgather(values, signal_buffers, axis=1)

        # Step 4: Local attention — Q_local × K_full, V_full
        # Output shape: [B, S_local, H, D] (determined by Q)
        attn_outputs = []
        for i in range(n):
            scale = 1.0 / (self.head_dim ** 0.5)
            attn_out = flash_attention_gpu(
                queries[i], keys_full[i], values_full[i],
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=scale,
            )
            attn_outputs.append(attn_out)

        # Step 5: Output projection — no communication needed
        has_encoder = encoder_hidden_states_list is not None
        if has_encoder:
            hidden_outs = []
            encoder_outs = []
            for i in range(n):
                batch_size = attn_outputs[i].shape[0]
                seq_len = attn_outputs[i].shape[1]
                out = ops.reshape(attn_outputs[i], [batch_size, seq_len, self.inner_dim])
                out = ops.cast(out, hidden_states_list[i].dtype)

                encoder_seq_len = encoder_hidden_states_list[i].shape[1]
                enc_out = out[:, :encoder_seq_len, :]
                hid_out = out[:, encoder_seq_len:, :]

                hid_proj = self.to_out_shards[i](hid_out)
                hid_proj = ops.rebind(hid_proj, hidden_states_list[i].shape)
                hidden_outs.append(hid_proj)

                enc_proj = self.to_add_out_shards[i](enc_out)
                enc_proj = ops.rebind(enc_proj, encoder_hidden_states_list[i].shape)
                encoder_outs.append(enc_proj)

            return hidden_outs, encoder_outs
        else:
            results = []
            for i in range(n):
                batch_size = attn_outputs[i].shape[0]
                seq_len = attn_outputs[i].shape[1]
                out = ops.reshape(attn_outputs[i], [batch_size, seq_len, self.inner_dim])
                out = ops.cast(out, hidden_states_list[i].dtype)
                proj = self.to_out_shards[i](out)
                proj = ops.rebind(proj, hidden_states_list[i].shape)
                results.append(proj)

            return results


class RingFlux2FeedForward(Module):
    """Replicated FFN for Ring CP. Identical to Ulysses — local per device."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 3.0,
        inner_dim: int | None = None,
        bias: bool = False,
        *,
        dtype: DType,
        devices: list[DeviceRef],
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.num_devices = len(devices)
        n = self.num_devices

        self.linear_in = Linear(dim, inner_dim * 2, dtype, devices[0], has_bias=bias)
        self.linear_in_shards = _replicate_linear(self.linear_in, n, devices)

        self.act_fn = Flux2SwiGLU()

        self.linear_out = Linear(inner_dim, dim_out, dtype, devices[0], has_bias=bias)
        self.linear_out_shards = _replicate_linear(self.linear_out, n, devices)

    def __call__(self, xs: list[TensorValue]) -> list[TensorValue]:
        outs = [self.linear_in_shards[i](xs[i]) for i in range(self.num_devices)]
        outs = [self.act_fn(o) for o in outs]
        outs = [self.linear_out_shards[i](outs[i]) for i in range(self.num_devices)]
        return outs


class RingFlux2ParallelSelfAttention(Module):
    """Ring CP fused self-attention + MLP for single-stream blocks."""

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
        *,
        dtype: DType,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.num_devices = len(devices)
        out_dim = out_dim if out_dim is not None else query_dim
        n = self.num_devices

        self.mlp_hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        fused_dim = self.inner_dim * 3 + self.mlp_hidden_dim * mlp_mult_factor
        self.to_qkv_mlp_proj = Linear(query_dim, fused_dim, dtype, devices[0], has_bias=bias)
        self.to_qkv_mlp_proj_shards = _replicate_linear(self.to_qkv_mlp_proj, n, devices)

        self.mlp_act_fn = Flux2SwiGLU()

        self.norm_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_q_shards = _replicate_norm(self.norm_q, n, devices)

        self.norm_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_k_shards = _replicate_norm(self.norm_k, n, devices)

        out_in_dim = self.inner_dim + self.mlp_hidden_dim
        self.to_out = Linear(out_in_dim, out_dim, dtype, devices[0], has_bias=out_bias)
        self.to_out_shards = _replicate_linear(self.to_out, n, devices)

    def __call__(
        self,
        hidden_states_list: list[TensorValue],
        signal_buffers: list[BufferValue],
        image_rotary_emb_list: list[tuple[TensorValue, TensorValue]] | None = None,
    ) -> list[TensorValue]:
        n = self.num_devices

        # Step 1: Fused QKV + MLP projection per device
        queries, keys, values, mlp_states = [], [], [], []
        for i in range(n):
            fused = self.to_qkv_mlp_proj_shards[i](hidden_states_list[i])

            qkv_dim = self.inner_dim * 3
            mlp_dim = self.mlp_hidden_dim * self.mlp_mult_factor
            qkv, mlp_hidden = ops.split(fused, [qkv_dim, mlp_dim], axis=-1)

            query, key, value = ops.chunk(qkv, 3, axis=-1)

            query = ops.reshape(query, [query.shape[0], query.shape[1], self.heads, self.head_dim])
            key = ops.reshape(key, [key.shape[0], key.shape[1], self.heads, self.head_dim])
            value = ops.reshape(value, [value.shape[0], value.shape[1], self.heads, self.head_dim])

            query = self.norm_q_shards[i](query)
            key = self.norm_k_shards[i](key)

            queries.append(query)
            keys.append(key)
            values.append(value)
            mlp_states.append(mlp_hidden)

        # Step 2: Apply RoPE before allgather
        if image_rotary_emb_list is not None:
            for i in range(n):
                cos, sin = image_rotary_emb_list[i]
                queries[i], keys[i] = _apply_flux2_qk_rope(
                    queries[i], keys[i], cos, sin
                )

        # Step 3: Allgather K, V (Q stays local)
        keys_full = ops.allgather(keys, signal_buffers, axis=1)
        values_full = ops.allgather(values, signal_buffers, axis=1)

        # Step 4: Local attention
        attn_outputs = []
        for i in range(n):
            attn_out = flash_attention_gpu(
                queries[i], keys_full[i], values_full[i],
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=1.0 / (self.head_dim ** 0.5),
            )
            attn_outputs.append(attn_out)

        # Step 5: Reshape, apply MLP, fuse, project
        results = []
        for i in range(n):
            batch_size = attn_outputs[i].shape[0]
            seq_len = attn_outputs[i].shape[1]
            attn_out = ops.reshape(attn_outputs[i], [batch_size, seq_len, self.inner_dim])
            attn_out = ops.cast(attn_out, hidden_states_list[i].dtype)

            mlp_out = self.mlp_act_fn(mlp_states[i])
            fused = ops.concat([attn_out, mlp_out], axis=-1)
            proj = self.to_out_shards[i](fused)
            proj = ops.rebind(proj, hidden_states_list[i].shape)
            results.append(proj)

        return results
