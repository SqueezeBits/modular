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

"""Unified Sequence Parallelism (USP) attention layers for Flux2.

A single attention class that handles all context-parallel modes based
on ulysses_degree and ring_degree:

  - ulysses=2, ring=1 → Ulysses only (all-to-all)
  - ulysses=1, ring=2 → Ring only (KV allgather)
  - ulysses=2, ring=2 → USP (Ulysses + Ring combined)

Subgroup operations become no-ops when the degree is 1 (group size = 1),
so no conditional branching is needed in the forward pass.

Reference: "USP: A Unified Sequence Parallelism Approach for Long Context
Generative AI" (https://arxiv.org/abs/2405.07719)
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .flux2_attention import Flux2SwiGLU, _apply_flux2_qk_rope


def _replicate_linear(
    linear: Linear, n: int, devices: list[DeviceRef]
) -> list[Linear]:
    """Replicate a Linear layer across devices."""
    linear.weight.sharding_strategy = ShardingStrategy.replicate(n)
    return linear.shard(devices)


def _replicate_norm(
    norm: RMSNorm, n: int, devices: list[DeviceRef]
) -> Sequence[RMSNorm]:
    """Replicate an RMSNorm layer across devices."""
    norm.sharding_strategy = ShardingStrategy.replicate(n)
    return norm.shard(devices)


def _compute_device_groups(
    num_devices: int, ulysses_degree: int, ring_degree: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Compute ulysses and ring device group indices.

    Ulysses groups: contiguous chunks of ulysses_degree.
    Ring groups: strided by ulysses_degree.

    When a degree is 1, groups are singletons -> subgroup ops become no-ops.
    """
    u, r = ulysses_degree, ring_degree
    ulysses_groups = [[g * u + j for j in range(u)] for g in range(r)]
    ring_groups = [[g + j * u for j in range(r)] for g in range(u)]
    return ulysses_groups, ring_groups


def _subgroup_allgather(
    tensors: list[TensorValue],
    signal_buffers: list[BufferValue],
    group_indices: list[int],
    axis: int,
) -> list[TensorValue]:
    """Allgather within a device subgroup. No-op for groups of size 1."""
    if len(group_indices) <= 1:
        return [tensors[d] for d in group_indices]
    group_ts = [tensors[d] for d in group_indices]
    group_bufs = [signal_buffers[d] for d in group_indices]
    return ops.allgather(group_ts, group_bufs, axis=axis)


def _replicated_prefix_head_slice(
    tensors: list[TensorValue],
    ulysses_degree: int,
    ulysses_groups: list[list[int]],
) -> None:
    """Slice replicated prefix tensors to the local head shard."""
    if ulysses_degree <= 1:
        return
    for group_devs in ulysses_groups:
        h_total = tensors[group_devs[0]].shape[2]
        h_local = h_total // ulysses_degree
        for j, d in enumerate(group_devs):
            start = j * h_local
            tensors[d] = tensors[d][:, :, start : start + h_local, :]


def _replicated_prefix_head_allgather(
    tensors: list[TensorValue],
    signal_buffers: list[BufferValue],
    ulysses_degree: int,
    ulysses_groups: list[list[int]],
) -> None:
    """Allgather prefix heads across Ulysses group to restore full heads."""
    if ulysses_degree <= 1:
        return
    for group_devs in ulysses_groups:
        gathered = _subgroup_allgather(
            tensors, signal_buffers, group_devs, axis=2
        )
        for j, d in enumerate(group_devs):
            tensors[d] = gathered[j]


class USPFlux2Attention(Module):
    """Unified context-parallel dual-stream Flux2 attention.

    Handles Ulysses-only, Ring-only, and USP (combined) modes via
    ulysses_degree and ring_degree parameters. The forward pass:

      1. Project Q, K, V locally (replicated weights)
      2. Apply RoPE
      3. Ulysses input all-to-all: gather seq, shard heads (skip if u=1)
      4. Ring allgather K, V: full sequence for attention (skip if r=1)
      5. Local flash attention
      6. Ulysses output all-to-all: gather heads, shard seq (skip if u=1)
      7. Output projection
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
        ulysses_degree: int = 1,
        ring_degree: int = 1,
    ) -> None:
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.num_devices = len(devices)
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        n = self.num_devices

        self.ulysses_groups, self.ring_groups = _compute_device_groups(
            n, ulysses_degree, ring_degree
        )

        out_dim = out_dim if out_dim is not None else query_dim

        # All projections replicated
        self.to_q = Linear(
            query_dim, self.inner_dim, dtype, devices[0], has_bias=bias
        )
        self.to_q_shards = _replicate_linear(self.to_q, n, devices)
        self.to_k = Linear(
            query_dim, self.inner_dim, dtype, devices[0], has_bias=bias
        )
        self.to_k_shards = _replicate_linear(self.to_k, n, devices)
        self.to_v = Linear(
            query_dim, self.inner_dim, dtype, devices[0], has_bias=bias
        )
        self.to_v_shards = _replicate_linear(self.to_v, n, devices)

        self.norm_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_q_shards = _replicate_norm(self.norm_q, n, devices)
        self.norm_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_k_shards = _replicate_norm(self.norm_k, n, devices)

        _to_out_linear = Linear(
            self.inner_dim, out_dim, dtype, devices[0], has_bias=out_bias
        )
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
            self.add_q_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                dtype,
                devices[0],
                has_bias=proj_bias,
            )
            self.add_q_proj_shards = _replicate_linear(
                self.add_q_proj, n, devices
            )
            self.add_k_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                dtype,
                devices[0],
                has_bias=proj_bias,
            )
            self.add_k_proj_shards = _replicate_linear(
                self.add_k_proj, n, devices
            )
            self.add_v_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                dtype,
                devices[0],
                has_bias=proj_bias,
            )
            self.add_v_proj_shards = _replicate_linear(
                self.add_v_proj, n, devices
            )
            self.norm_added_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
            self.norm_added_q_shards = _replicate_norm(
                self.norm_added_q, n, devices
            )
            self.norm_added_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
            self.norm_added_k_shards = _replicate_norm(
                self.norm_added_k, n, devices
            )
            self.to_add_out = Linear(
                self.inner_dim, query_dim, dtype, devices[0], has_bias=out_bias
            )
            self.to_add_out_shards = _replicate_linear(
                self.to_add_out, n, devices
            )

    def _per_device_qkv(
        self,
        i: int,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue | None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        batch_size = hidden_states.shape[0]
        query = self.to_q_shards[i](hidden_states)
        key = self.to_k_shards[i](hidden_states)
        value = self.to_v_shards[i](hidden_states)

        seq_len = query.shape[1]
        query = ops.reshape(
            query, [batch_size, seq_len, self.heads, self.head_dim]
        )
        key = ops.reshape(key, [batch_size, seq_len, self.heads, self.head_dim])
        value = ops.reshape(
            value, [batch_size, seq_len, self.heads, self.head_dim]
        )

        query = self.norm_q_shards[i](query)
        key = self.norm_k_shards[i](key)

        if (
            encoder_hidden_states is not None
            and self.added_kv_proj_dim is not None
        ):
            assert self.add_q_proj_shards is not None
            assert self.add_k_proj_shards is not None
            assert self.add_v_proj_shards is not None
            assert self.norm_added_q_shards is not None
            assert self.norm_added_k_shards is not None
            enc_q = self.add_q_proj_shards[i](encoder_hidden_states)
            enc_k = self.add_k_proj_shards[i](encoder_hidden_states)
            enc_v = self.add_v_proj_shards[i](encoder_hidden_states)
            enc_seq = enc_q.shape[1]
            enc_q = ops.reshape(
                enc_q, [batch_size, enc_seq, self.heads, self.head_dim]
            )
            enc_k = ops.reshape(
                enc_k, [batch_size, enc_seq, self.heads, self.head_dim]
            )
            enc_v = ops.reshape(
                enc_v, [batch_size, enc_seq, self.heads, self.head_dim]
            )
            enc_q = self.norm_added_q_shards[i](enc_q)
            enc_k = self.norm_added_k_shards[i](enc_k)
            query = ops.concat([enc_q, query], axis=1)
            key = ops.concat([enc_k, key], axis=1)
            value = ops.concat([enc_v, value], axis=1)

        return query, key, value

    def _ulysses_input(
        self,
        tensors: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> None:
        """Ulysses input all-to-all: gather seq, shard heads within group."""
        u = self.ulysses_degree
        if u <= 1:
            return
        for group_devs in self.ulysses_groups:
            gathered = _subgroup_allgather(
                tensors, signal_buffers, group_devs, axis=1
            )
            h_local = tensors[group_devs[0]].shape[2] // u
            for j, d in enumerate(group_devs):
                tensors[d] = gathered[j][
                    :, :, j * h_local : (j + 1) * h_local, :
                ]

    def _ulysses_output(
        self,
        tensors: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> None:
        """Ulysses output all-to-all: gather heads, shard seq within group."""
        u = self.ulysses_degree
        if u <= 1:
            return
        for group_devs in self.ulysses_groups:
            gathered = _subgroup_allgather(
                tensors, signal_buffers, group_devs, axis=2
            )
            s_total = gathered[0].shape[1]
            s_local = s_total // u
            for j, d in enumerate(group_devs):
                tensors[d] = gathered[j][
                    :, j * s_local : (j + 1) * s_local, :, :
                ]

    def _ring_allgather_kv(
        self,
        keys: list[TensorValue],
        values: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> None:
        """Ring: allgather K, V within ring group."""
        r = self.ring_degree
        if r <= 1:
            return
        for group_devs in self.ring_groups:
            k_full = _subgroup_allgather(
                keys, signal_buffers, group_devs, axis=1
            )
            v_full = _subgroup_allgather(
                values, signal_buffers, group_devs, axis=1
            )
            for j, d in enumerate(group_devs):
                keys[d] = k_full[j]
                values[d] = v_full[j]

    def __call__(
        self,
        hidden_states_list: list[TensorValue],
        signal_buffers: list[BufferValue],
        encoder_hidden_states_list: list[TensorValue] | None = None,
        image_rotary_emb_list: list[tuple[TensorValue, TensorValue]]
        | None = None,
    ) -> list[TensorValue] | tuple[list[TensorValue], list[TensorValue]]:
        n = self.num_devices
        has_encoder = encoder_hidden_states_list is not None

        # 1. Project Q, K, V locally
        queries: list[TensorValue] = [None] * n  # type: ignore[list-item]
        keys: list[TensorValue] = [None] * n  # type: ignore[list-item]
        values: list[TensorValue] = [None] * n  # type: ignore[list-item]
        for i in range(n):
            enc = (
                encoder_hidden_states_list[i]
                if encoder_hidden_states_list is not None
                else None
            )
            queries[i], keys[i], values[i] = self._per_device_qkv(
                i, hidden_states_list[i], enc
            )

        # 2. Apply RoPE before any redistribution
        if image_rotary_emb_list is not None:
            for i in range(n):
                cos, sin = image_rotary_emb_list[i]
                queries[i], keys[i] = _apply_flux2_qk_rope(
                    queries[i], keys[i], cos, sin
                )

        # 3. Replicated prefix optimization for dual-stream (encoder present):
        #    Text tokens are replicated across all devices, image tokens
        #    are scattered. Only image (suffix) needs all-to-all; text
        #    (prefix) just needs a local head slice.
        if has_encoder and self.ulysses_degree > 1:
            assert encoder_hidden_states_list is not None

            # Split into prefix (text, replicated) and suffix (image, sharded)
            q_prefix: list[TensorValue] = [None] * n  # type: ignore[list-item]
            k_prefix: list[TensorValue] = [None] * n  # type: ignore[list-item]
            v_prefix: list[TensorValue] = [None] * n  # type: ignore[list-item]
            q_suffix: list[TensorValue] = [None] * n  # type: ignore[list-item]
            k_suffix: list[TensorValue] = [None] * n  # type: ignore[list-item]
            v_suffix: list[TensorValue] = [None] * n  # type: ignore[list-item]
            for i in range(n):
                enc_seq = encoder_hidden_states_list[i].shape[1]
                q_prefix[i] = queries[i][:, :enc_seq, :, :]
                k_prefix[i] = keys[i][:, :enc_seq, :, :]
                v_prefix[i] = values[i][:, :enc_seq, :, :]
                q_suffix[i] = queries[i][:, enc_seq:, :, :]
                k_suffix[i] = keys[i][:, enc_seq:, :, :]
                v_suffix[i] = values[i][:, enc_seq:, :, :]

            # All-to-all only the image suffix
            self._ulysses_input(q_suffix, signal_buffers)
            self._ulysses_input(k_suffix, signal_buffers)
            self._ulysses_input(v_suffix, signal_buffers)

            # Slice replicated prefix to matching head shard
            _replicated_prefix_head_slice(
                q_prefix, self.ulysses_degree, self.ulysses_groups
            )
            _replicated_prefix_head_slice(
                k_prefix, self.ulysses_degree, self.ulysses_groups
            )
            _replicated_prefix_head_slice(
                v_prefix, self.ulysses_degree, self.ulysses_groups
            )

            # Reassemble [prefix_h_local, suffix_gathered]
            for i in range(n):
                queries[i] = ops.concat([q_prefix[i], q_suffix[i]], axis=1)
                keys[i] = ops.concat([k_prefix[i], k_suffix[i]], axis=1)
                values[i] = ops.concat([v_prefix[i], v_suffix[i]], axis=1)

            # Ring allgather K, V (no-op if ring_degree=1)
            self._ring_allgather_kv(keys, values, signal_buffers)

            # Local attention
            attn_outputs: list[TensorValue] = [None] * n  # type: ignore[list-item]
            for i in range(n):
                attn_outputs[i] = flash_attention_gpu(
                    queries[i],
                    keys[i],
                    values[i],
                    mask_variant=MHAMaskVariant.NULL_MASK,
                    scale=1.0 / (self.head_dim**0.5),
                )

            # Split output back into prefix and suffix
            out_prefix: list[TensorValue] = [None] * n  # type: ignore[list-item]
            out_suffix: list[TensorValue] = [None] * n  # type: ignore[list-item]
            for i in range(n):
                enc_seq = encoder_hidden_states_list[i].shape[1]
                out_prefix[i] = attn_outputs[i][:, :enc_seq, :, :]
                out_suffix[i] = attn_outputs[i][:, enc_seq:, :, :]

            # Output all-to-all only on image suffix
            self._ulysses_output(out_suffix, signal_buffers)

            # Allgather prefix heads to restore full heads
            _replicated_prefix_head_allgather(
                out_prefix,
                signal_buffers,
                self.ulysses_degree,
                self.ulysses_groups,
            )

            # Reassemble full output
            for i in range(n):
                attn_outputs[i] = ops.concat(
                    [out_prefix[i], out_suffix[i]], axis=1
                )

        else:
            # No encoder or no Ulysses: original path
            self._ulysses_input(queries, signal_buffers)
            self._ulysses_input(keys, signal_buffers)
            self._ulysses_input(values, signal_buffers)

            self._ring_allgather_kv(keys, values, signal_buffers)

            attn_outputs = [None] * n  # type: ignore[list-item]
            for i in range(n):
                attn_outputs[i] = flash_attention_gpu(
                    queries[i],
                    keys[i],
                    values[i],
                    mask_variant=MHAMaskVariant.NULL_MASK,
                    scale=1.0 / (self.head_dim**0.5),
                )

            self._ulysses_output(attn_outputs, signal_buffers)

        # 7. Output projection with rebind to restore symbolic shapes
        if has_encoder:
            assert encoder_hidden_states_list is not None
            hidden_outs, encoder_outs = [], []
            for i in range(n):
                out = ops.rebind(
                    attn_outputs[i],
                    [
                        hidden_states_list[i].shape[0],
                        hidden_states_list[i].shape[1]
                        + encoder_hidden_states_list[i].shape[1],
                        self.heads,
                        self.head_dim,
                    ],
                )
                batch_size = out.shape[0]
                seq_len = out.shape[1]
                out = ops.reshape(out, [batch_size, seq_len, self.inner_dim])
                out = ops.cast(out, hidden_states_list[i].dtype)
                enc_seq = encoder_hidden_states_list[i].shape[1]
                hid_proj = self.to_out_shards[i](out[:, enc_seq:, :])
                hid_proj = ops.rebind(hid_proj, hidden_states_list[i].shape)
                hidden_outs.append(hid_proj)
                assert self.to_add_out_shards is not None
                enc_proj = self.to_add_out_shards[i](out[:, :enc_seq, :])
                enc_proj = ops.rebind(
                    enc_proj, encoder_hidden_states_list[i].shape
                )
                encoder_outs.append(enc_proj)
            return hidden_outs, encoder_outs
        else:
            results = []
            for i in range(n):
                out = ops.rebind(
                    attn_outputs[i],
                    [
                        hidden_states_list[i].shape[0],
                        hidden_states_list[i].shape[1],
                        self.heads,
                        self.head_dim,
                    ],
                )
                out = ops.reshape(
                    out, [out.shape[0], out.shape[1], self.inner_dim]
                )
                out = ops.cast(out, hidden_states_list[i].dtype)
                proj = self.to_out_shards[i](out)
                proj = ops.rebind(proj, hidden_states_list[i].shape)
                results.append(proj)
            return results


class USPFlux2FeedForward(Module):
    """Replicated FFN -- local per device, no communication."""

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
    ) -> None:
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.num_devices = len(devices)
        n = self.num_devices
        self.linear_in = Linear(
            dim, inner_dim * 2, dtype, devices[0], has_bias=bias
        )
        self.linear_in_shards = _replicate_linear(self.linear_in, n, devices)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = Linear(
            inner_dim, dim_out, dtype, devices[0], has_bias=bias
        )
        self.linear_out_shards = _replicate_linear(self.linear_out, n, devices)

    def __call__(self, xs: list[TensorValue]) -> list[TensorValue]:
        outs = [
            self.linear_in_shards[i](xs[i]) for i in range(self.num_devices)
        ]
        outs = [self.act_fn(o) for o in outs]
        return [
            self.linear_out_shards[i](outs[i]) for i in range(self.num_devices)
        ]


class USPFlux2ParallelSelfAttention(Module):
    """Unified CP fused self-attention + MLP for single-stream blocks."""

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
        ulysses_degree: int = 1,
        ring_degree: int = 1,
    ) -> None:
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.num_devices = len(devices)
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        out_dim = out_dim if out_dim is not None else query_dim
        n = self.num_devices

        self.ulysses_groups, self.ring_groups = _compute_device_groups(
            n, ulysses_degree, ring_degree
        )

        self.mlp_hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        fused_dim = self.inner_dim * 3 + self.mlp_hidden_dim * mlp_mult_factor
        self.to_qkv_mlp_proj = Linear(
            query_dim, fused_dim, dtype, devices[0], has_bias=bias
        )
        self.to_qkv_mlp_proj_shards = _replicate_linear(
            self.to_qkv_mlp_proj, n, devices
        )
        self.mlp_act_fn = Flux2SwiGLU()
        self.norm_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_q_shards = _replicate_norm(self.norm_q, n, devices)
        self.norm_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_k_shards = _replicate_norm(self.norm_k, n, devices)
        out_in_dim = self.inner_dim + self.mlp_hidden_dim
        self.to_out = Linear(
            out_in_dim, out_dim, dtype, devices[0], has_bias=out_bias
        )
        self.to_out_shards = _replicate_linear(self.to_out, n, devices)

    def _ulysses_input(
        self,
        tensors: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> None:
        u = self.ulysses_degree
        if u <= 1:
            return
        for group_devs in self.ulysses_groups:
            gathered = _subgroup_allgather(
                tensors, signal_buffers, group_devs, axis=1
            )
            h_local = tensors[group_devs[0]].shape[2] // u
            for j, d in enumerate(group_devs):
                tensors[d] = gathered[j][
                    :, :, j * h_local : (j + 1) * h_local, :
                ]

    def _ulysses_output(
        self,
        tensors: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> None:
        u = self.ulysses_degree
        if u <= 1:
            return
        for group_devs in self.ulysses_groups:
            gathered = _subgroup_allgather(
                tensors, signal_buffers, group_devs, axis=2
            )
            s_total = gathered[0].shape[1]
            s_local = s_total // u
            for j, d in enumerate(group_devs):
                tensors[d] = gathered[j][
                    :, j * s_local : (j + 1) * s_local, :, :
                ]

    def _ring_allgather_kv(
        self,
        keys: list[TensorValue],
        values: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> None:
        r = self.ring_degree
        if r <= 1:
            return
        for group_devs in self.ring_groups:
            k_full = _subgroup_allgather(
                keys, signal_buffers, group_devs, axis=1
            )
            v_full = _subgroup_allgather(
                values, signal_buffers, group_devs, axis=1
            )
            for j, d in enumerate(group_devs):
                keys[d] = k_full[j]
                values[d] = v_full[j]

    def __call__(
        self,
        hidden_states_list: list[TensorValue],
        signal_buffers: list[BufferValue],
        image_rotary_emb_list: list[tuple[TensorValue, TensorValue]]
        | None = None,
        num_replicated_prefix: int | Dim | None = None,
    ) -> list[TensorValue]:
        n = self.num_devices
        queries: list[TensorValue] = [None] * n  # type: ignore[list-item]
        keys: list[TensorValue] = [None] * n  # type: ignore[list-item]
        values: list[TensorValue] = [None] * n  # type: ignore[list-item]
        mlp_states: list[TensorValue] = [None] * n  # type: ignore[list-item]

        for i in range(n):
            fused = self.to_qkv_mlp_proj_shards[i](hidden_states_list[i])
            qkv_dim = self.inner_dim * 3
            mlp_dim = self.mlp_hidden_dim * self.mlp_mult_factor
            qkv, mlp_hidden = ops.split(fused, [qkv_dim, mlp_dim], axis=-1)
            q, k, v = ops.chunk(qkv, 3, axis=-1)
            q = ops.reshape(
                q, [q.shape[0], q.shape[1], self.heads, self.head_dim]
            )
            k = ops.reshape(
                k, [k.shape[0], k.shape[1], self.heads, self.head_dim]
            )
            v = ops.reshape(
                v, [v.shape[0], v.shape[1], self.heads, self.head_dim]
            )
            queries[i] = self.norm_q_shards[i](q)
            keys[i] = self.norm_k_shards[i](k)
            values[i] = v
            mlp_states[i] = mlp_hidden

        if image_rotary_emb_list is not None:
            for i in range(n):
                cos, sin = image_rotary_emb_list[i]
                queries[i], keys[i] = _apply_flux2_qk_rope(
                    queries[i], keys[i], cos, sin
                )

        # Replicated prefix for single-stream: text tokens are replicated,
        # only image tokens (suffix) go through all-to-all.
        has_prefix = (
            num_replicated_prefix is not None and self.ulysses_degree > 1
        )

        if has_prefix:
            assert num_replicated_prefix is not None
            num_rep = num_replicated_prefix

            # Split QKV and MLP into prefix (text) and suffix (image)
            q_pre = [queries[i][:, :num_rep, :, :] for i in range(n)]
            k_pre = [keys[i][:, :num_rep, :, :] for i in range(n)]
            v_pre = [values[i][:, :num_rep, :, :] for i in range(n)]
            q_suf = [queries[i][:, num_rep:, :, :] for i in range(n)]
            k_suf = [keys[i][:, num_rep:, :, :] for i in range(n)]
            v_suf = [values[i][:, num_rep:, :, :] for i in range(n)]
            mlp_pre = [mlp_states[i][:, :num_rep, :] for i in range(n)]
            mlp_suf = [mlp_states[i][:, num_rep:, :] for i in range(n)]

            # All-to-all only image suffix
            self._ulysses_input(q_suf, signal_buffers)
            self._ulysses_input(k_suf, signal_buffers)
            self._ulysses_input(v_suf, signal_buffers)

            # Slice replicated prefix to matching head shard
            _replicated_prefix_head_slice(
                q_pre, self.ulysses_degree, self.ulysses_groups
            )
            _replicated_prefix_head_slice(
                k_pre, self.ulysses_degree, self.ulysses_groups
            )
            _replicated_prefix_head_slice(
                v_pre, self.ulysses_degree, self.ulysses_groups
            )

            # Reassemble
            for i in range(n):
                queries[i] = ops.concat([q_pre[i], q_suf[i]], axis=1)
                keys[i] = ops.concat([k_pre[i], k_suf[i]], axis=1)
                values[i] = ops.concat([v_pre[i], v_suf[i]], axis=1)

            self._ring_allgather_kv(keys, values, signal_buffers)

            attn_outputs: list[TensorValue] = [None] * n  # type: ignore[list-item]
            for i in range(n):
                attn_outputs[i] = flash_attention_gpu(
                    queries[i],
                    keys[i],
                    values[i],
                    mask_variant=MHAMaskVariant.NULL_MASK,
                    scale=1.0 / (self.head_dim**0.5),
                )

            # Split output
            out_pre = [attn_outputs[i][:, :num_rep, :, :] for i in range(n)]
            out_suf = [attn_outputs[i][:, num_rep:, :, :] for i in range(n)]

            self._ulysses_output(out_suf, signal_buffers)
            _replicated_prefix_head_allgather(
                out_pre,
                signal_buffers,
                self.ulysses_degree,
                self.ulysses_groups,
            )

            for i in range(n):
                attn_outputs[i] = ops.concat([out_pre[i], out_suf[i]], axis=1)

            # MLP is [B, S, mlp_dim] with no head dimension.
            # mlp_pre has full text (replicated), mlp_suf has local image chunk.
            # This matches the attention output layout (text_full + image_local).
            for i in range(n):
                mlp_states[i] = ops.concat([mlp_pre[i], mlp_suf[i]], axis=1)

        else:
            self._ulysses_input(queries, signal_buffers)
            self._ulysses_input(keys, signal_buffers)
            self._ulysses_input(values, signal_buffers)
            self._ring_allgather_kv(keys, values, signal_buffers)

            attn_outputs = [None] * n  # type: ignore[list-item]
            for i in range(n):
                attn_outputs[i] = flash_attention_gpu(
                    queries[i],
                    keys[i],
                    values[i],
                    mask_variant=MHAMaskVariant.NULL_MASK,
                    scale=1.0 / (self.head_dim**0.5),
                )

            self._ulysses_output(attn_outputs, signal_buffers)

        results = []
        for i in range(n):
            out = ops.rebind(
                attn_outputs[i],
                [
                    hidden_states_list[i].shape[0],
                    hidden_states_list[i].shape[1],
                    self.heads,
                    self.head_dim,
                ],
            )
            out = ops.reshape(out, [out.shape[0], out.shape[1], self.inner_dim])
            out = ops.cast(out, hidden_states_list[i].dtype)
            mlp_out = self.mlp_act_fn(mlp_states[i])
            fused = ops.concat([out, mlp_out], axis=-1)
            proj = self.to_out_shards[i](fused)
            proj = ops.rebind(proj, hidden_states_list[i].shape)
            results.append(proj)
        return results
