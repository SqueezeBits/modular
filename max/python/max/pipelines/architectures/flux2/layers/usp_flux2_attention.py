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

"""USP (Unified Sequence Parallelism) Flux2 attention layers.

USP combines Ulysses and Ring within each attention layer:
  1. Ulysses input all-to-all within ulysses subgroup:
     gather sequence, shard heads
  2. Ring allgather K,V within ring subgroup:
     each device gets full K,V for attention
  3. Local attention: Q_local × K_full × V_full
  4. Ulysses output all-to-all within ulysses subgroup:
     gather heads, shard sequence back

Device layout example (ulysses=2, ring=2, 4 GPUs):
  Ulysses groups (contiguous): {0,1}, {2,3}
  Ring groups (strided):       {0,2}, {1,3}

Reference: "USP: A Unified Sequence Parallelism Approach for Long Context
Generative AI" (https://arxiv.org/abs/2405.07719)
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


def _replicate_linear(linear, n, devices):
    linear.weight.sharding_strategy = ShardingStrategy.replicate(n)
    return linear.shard(devices)


def _replicate_norm(norm, n, devices):
    norm.sharding_strategy = ShardingStrategy.replicate(n)
    return norm.shard(devices)


def _compute_device_groups(
    num_devices: int, ulysses_degree: int, ring_degree: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Compute ulysses and ring device group indices.

    Ulysses groups: contiguous chunks of ulysses_degree.
    Ring groups: strided by ulysses_degree.

    Example (u=2, r=2, 4 GPUs):
      Ulysses groups: [[0,1], [2,3]]
      Ring groups:    [[0,2], [1,3]]
    """
    u, r = ulysses_degree, ring_degree
    ulysses_groups = [
        [g * u + j for j in range(u)] for g in range(r)
    ]
    ring_groups = [
        [g + j * u for j in range(r)] for g in range(u)
    ]
    return ulysses_groups, ring_groups


class USPFlux2Attention(Module):
    """USP dual-stream Flux2 attention (Ulysses + Ring)."""

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
        ulysses_degree: int,
        ring_degree: int,
    ) -> None:
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.num_devices = len(devices)
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        self.local_heads = self.heads // ulysses_degree
        n = self.num_devices

        self.ulysses_groups, self.ring_groups = _compute_device_groups(
            n, ulysses_degree, ring_degree
        )

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

        self.add_q_proj_shards = None
        self.add_k_proj_shards = None
        self.add_v_proj_shards = None
        self.norm_added_q_shards = None
        self.norm_added_k_shards = None
        self.to_add_out_shards = None

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

    def _per_device_qkv(self, i, hidden_states, encoder_hidden_states):
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
            enc_seq = enc_q.shape[1]
            enc_q = ops.reshape(enc_q, [batch_size, enc_seq, self.heads, self.head_dim])
            enc_k = ops.reshape(enc_k, [batch_size, enc_seq, self.heads, self.head_dim])
            enc_v = ops.reshape(enc_v, [batch_size, enc_seq, self.heads, self.head_dim])
            enc_q = self.norm_added_q_shards[i](enc_q)
            enc_k = self.norm_added_k_shards[i](enc_k)
            query = ops.concat([enc_q, query], axis=1)
            key = ops.concat([enc_k, key], axis=1)
            value = ops.concat([enc_v, value], axis=1)

        return query, key, value

    def _ulysses_input_alltoall(self, tensors, signal_buffers):
        """Ulysses input: gather sequence within group, shard heads."""
        u = self.ulysses_degree
        for group_devs in self.ulysses_groups:
            group_ts = [tensors[d] for d in group_devs]
            group_bufs = [signal_buffers[d] for d in group_devs]
            gathered = ops.allgather(group_ts, group_bufs, axis=1)
            h_local = tensors[group_devs[0]].shape[2] // u
            for j, d in enumerate(group_devs):
                start = j * h_local
                end = start + h_local
                tensors[d] = gathered[j][:, :, start:end, :]

    def _ulysses_output_alltoall(self, tensors, signal_buffers):
        """Ulysses output: gather heads within group, shard sequence."""
        u = self.ulysses_degree
        for group_devs in self.ulysses_groups:
            group_ts = [tensors[d] for d in group_devs]
            group_bufs = [signal_buffers[d] for d in group_devs]
            gathered = ops.allgather(group_ts, group_bufs, axis=2)
            s_total = gathered[0].shape[1]
            s_local = s_total // u
            for j, d in enumerate(group_devs):
                start = j * s_local
                end = start + s_local
                tensors[d] = gathered[j][:, start:end, :, :]

    def _ring_allgather_kv(self, keys, values, signal_buffers):
        """Ring: allgather K,V within ring group."""
        for group_devs in self.ring_groups:
            group_k = [keys[d] for d in group_devs]
            group_v = [values[d] for d in group_devs]
            group_bufs = [signal_buffers[d] for d in group_devs]
            k_full = ops.allgather(group_k, group_bufs, axis=1)
            v_full = ops.allgather(group_v, group_bufs, axis=1)
            for j, d in enumerate(group_devs):
                keys[d] = k_full[j]
                values[d] = v_full[j]

    def __call__(
        self,
        hidden_states_list: list[TensorValue],
        signal_buffers: list[BufferValue],
        encoder_hidden_states_list: list[TensorValue] | None = None,
        image_rotary_emb_list: list[tuple[TensorValue, TensorValue]] | None = None,
    ) -> list[TensorValue] | tuple[list[TensorValue], list[TensorValue]]:
        n = self.num_devices

        # Step 1: Project Q, K, V locally
        queries = [None] * n
        keys = [None] * n
        values = [None] * n
        for i in range(n):
            enc = encoder_hidden_states_list[i] if encoder_hidden_states_list is not None else None
            queries[i], keys[i], values[i] = self._per_device_qkv(
                i, hidden_states_list[i], enc
            )

        # Step 2: Apply RoPE before any redistribution
        if image_rotary_emb_list is not None:
            for i in range(n):
                cos, sin = image_rotary_emb_list[i]
                queries[i], keys[i] = _apply_flux2_qk_rope(
                    queries[i], keys[i], cos, sin
                )

        # Step 3: Ulysses input all-to-all — gather seq, shard heads
        self._ulysses_input_alltoall(queries, signal_buffers)
        self._ulysses_input_alltoall(keys, signal_buffers)
        self._ulysses_input_alltoall(values, signal_buffers)

        # Step 4: Ring allgather K, V — full sequence for attention
        self._ring_allgather_kv(keys, values, signal_buffers)

        # Step 5: Local attention
        attn_outputs = [None] * n
        for i in range(n):
            attn_outputs[i] = flash_attention_gpu(
                queries[i], keys[i], values[i],
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=1.0 / (self.head_dim ** 0.5),
            )

        # Step 6: Ulysses output all-to-all — gather heads, shard seq
        self._ulysses_output_alltoall(attn_outputs, signal_buffers)

        # Step 7: Output projection with rebind
        has_encoder = encoder_hidden_states_list is not None
        if has_encoder:
            hidden_outs, encoder_outs = [], []
            for i in range(n):
                out = ops.rebind(attn_outputs[i], [
                    hidden_states_list[i].shape[0],
                    hidden_states_list[i].shape[1] + encoder_hidden_states_list[i].shape[1],
                    self.heads, self.head_dim,
                ])
                batch_size = out.shape[0]
                seq_len = out.shape[1]
                out = ops.reshape(out, [batch_size, seq_len, self.inner_dim])
                out = ops.cast(out, hidden_states_list[i].dtype)
                enc_seq = encoder_hidden_states_list[i].shape[1]
                enc_out = out[:, :enc_seq, :]
                hid_out = out[:, enc_seq:, :]
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
                out = ops.rebind(attn_outputs[i], [
                    hidden_states_list[i].shape[0],
                    hidden_states_list[i].shape[1],
                    self.heads, self.head_dim,
                ])
                out = ops.reshape(out, [out.shape[0], out.shape[1], self.inner_dim])
                out = ops.cast(out, hidden_states_list[i].dtype)
                proj = self.to_out_shards[i](out)
                proj = ops.rebind(proj, hidden_states_list[i].shape)
                results.append(proj)
            return results


class USPFlux2FeedForward(Module):
    """Replicated FFN — local per device, no communication."""

    def __init__(self, dim, dim_out=None, mult=3.0, inner_dim=None, bias=False,
                 *, dtype, devices):
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

    def __call__(self, xs):
        outs = [self.linear_in_shards[i](xs[i]) for i in range(self.num_devices)]
        outs = [self.act_fn(o) for o in outs]
        return [self.linear_out_shards[i](outs[i]) for i in range(self.num_devices)]


class USPFlux2ParallelSelfAttention(Module):
    """USP fused self-attention + MLP for single-stream blocks."""

    def __init__(self, query_dim, heads=8, dim_head=64, bias=False, out_bias=True,
                 eps=1e-5, out_dim=None, mlp_ratio=4.0, mlp_mult_factor=2,
                 *, dtype, devices, ulysses_degree, ring_degree):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.num_devices = len(devices)
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        self.local_heads = self.heads // ulysses_degree
        out_dim = out_dim if out_dim is not None else query_dim
        n = self.num_devices

        self.ulysses_groups, self.ring_groups = _compute_device_groups(
            n, ulysses_degree, ring_degree
        )

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

    def _ulysses_input_alltoall(self, tensors, signal_buffers):
        u = self.ulysses_degree
        for group_devs in self.ulysses_groups:
            group_ts = [tensors[d] for d in group_devs]
            group_bufs = [signal_buffers[d] for d in group_devs]
            gathered = ops.allgather(group_ts, group_bufs, axis=1)
            h_local = tensors[group_devs[0]].shape[2] // u
            for j, d in enumerate(group_devs):
                tensors[d] = gathered[j][:, :, j * h_local : (j + 1) * h_local, :]

    def _ulysses_output_alltoall(self, tensors, signal_buffers):
        u = self.ulysses_degree
        for group_devs in self.ulysses_groups:
            group_ts = [tensors[d] for d in group_devs]
            group_bufs = [signal_buffers[d] for d in group_devs]
            gathered = ops.allgather(group_ts, group_bufs, axis=2)
            s_total = gathered[0].shape[1]
            s_local = s_total // u
            for j, d in enumerate(group_devs):
                tensors[d] = gathered[j][:, j * s_local : (j + 1) * s_local, :, :]

    def _ring_allgather_kv(self, keys, values, signal_buffers):
        for group_devs in self.ring_groups:
            group_k = [keys[d] for d in group_devs]
            group_v = [values[d] for d in group_devs]
            group_bufs = [signal_buffers[d] for d in group_devs]
            k_full = ops.allgather(group_k, group_bufs, axis=1)
            v_full = ops.allgather(group_v, group_bufs, axis=1)
            for j, d in enumerate(group_devs):
                keys[d] = k_full[j]
                values[d] = v_full[j]

    def __call__(self, hidden_states_list, signal_buffers, image_rotary_emb_list=None):
        n = self.num_devices
        queries = [None] * n
        keys = [None] * n
        values = [None] * n
        mlp_states = [None] * n

        for i in range(n):
            fused = self.to_qkv_mlp_proj_shards[i](hidden_states_list[i])
            qkv_dim = self.inner_dim * 3
            mlp_dim = self.mlp_hidden_dim * self.mlp_mult_factor
            qkv, mlp_hidden = ops.split(fused, [qkv_dim, mlp_dim], axis=-1)
            q, k, v = ops.chunk(qkv, 3, axis=-1)
            q = ops.reshape(q, [q.shape[0], q.shape[1], self.heads, self.head_dim])
            k = ops.reshape(k, [k.shape[0], k.shape[1], self.heads, self.head_dim])
            v = ops.reshape(v, [v.shape[0], v.shape[1], self.heads, self.head_dim])
            queries[i] = self.norm_q_shards[i](q)
            keys[i] = self.norm_k_shards[i](k)
            values[i] = v
            mlp_states[i] = mlp_hidden

        if image_rotary_emb_list is not None:
            for i in range(n):
                cos, sin = image_rotary_emb_list[i]
                queries[i], keys[i] = _apply_flux2_qk_rope(queries[i], keys[i], cos, sin)

        self._ulysses_input_alltoall(queries, signal_buffers)
        self._ulysses_input_alltoall(keys, signal_buffers)
        self._ulysses_input_alltoall(values, signal_buffers)
        self._ring_allgather_kv(keys, values, signal_buffers)

        attn_outputs = [None] * n
        for i in range(n):
            attn_outputs[i] = flash_attention_gpu(
                queries[i], keys[i], values[i],
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=1.0 / (self.head_dim ** 0.5),
            )

        self._ulysses_output_alltoall(attn_outputs, signal_buffers)

        results = []
        for i in range(n):
            out = ops.rebind(attn_outputs[i], [
                hidden_states_list[i].shape[0], hidden_states_list[i].shape[1],
                self.heads, self.head_dim,
            ])
            out = ops.reshape(out, [out.shape[0], out.shape[1], self.inner_dim])
            out = ops.cast(out, hidden_states_list[i].dtype)
            mlp_out = self.mlp_act_fn(mlp_states[i])
            fused = ops.concat([out, mlp_out], axis=-1)
            proj = self.to_out_shards[i](fused)
            proj = ops.rebind(proj, hidden_states_list[i].shape)
            results.append(proj)
        return results
