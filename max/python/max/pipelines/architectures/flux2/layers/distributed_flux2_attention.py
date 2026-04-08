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

"""Distributed (tensor-parallel) Flux2 attention layers.

Uses ShardingStrategy + Linear.shard() so that full weights are loaded
once and sliced lazily at graph-construction time.

IMPORTANT: Original Linear/RMSNorm layers must be stored as `self.X`
attributes so that `load_state_dict` can find them by name and assign
checkpoint weights. The sharded lists (`self.X_shards`) reference
the original weight via ShardingStrategy.
"""

from __future__ import annotations

from functools import partial

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, ShardingStrategy, TensorValue, ops
from max.graph.weight import Weight
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.comm.allreduce import Allreduce
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .flux2_attention import Flux2SwiGLU, _apply_flux2_qk_rope


def _merged_row_sharding_strategy(
    weight: Weight,
    i: int,
    num_devices: int,
    sub_dims: list[int],
) -> TensorValue:
    """Shard a fused weight where multiple sub-matrices are concatenated
    along axis 0. Each sub-matrix is sharded independently by row.

    For example, a fused QKV+MLP weight [Q|K|V|gate|up, dim] needs each
    of Q, K, V, gate, up to be row-sharded independently so that every
    device gets the correct head-aligned slices from each sub-matrix.
    """
    from max.graph.weight import _compute_shard_range

    shards = []
    offset = 0
    for d in sub_dims:
        start, end = _compute_shard_range(d, i, num_devices)
        shards.append(weight[offset + start : offset + end])
        offset += d
    return ops.concat(tuple(shards), axis=0)


def _merged_col_sharding_strategy(
    weight: Weight,
    i: int,
    num_devices: int,
    sub_dims: list[int],
) -> TensorValue:
    """Shard a weight where the input (column) dimension is a concatenation
    of independently-sharded sub-groups.

    For the fused attn+MLP output projection, the input is
    [attn_shard | mlp_shard] where each part was sharded by a different
    amount (inner_dim vs mlp_hidden_dim). We must pick the correct
    non-contiguous columns for each device.
    """
    from max.graph.weight import _compute_shard_range

    shards = []
    offset = 0
    for d in sub_dims:
        start, end = _compute_shard_range(d, i, num_devices)
        shards.append(weight[:, offset + start : offset + end])
        offset += d
    return ops.concat(tuple(shards), axis=1)


class DistributedFlux2FeedForward(Module):
    """Tensor-parallel Flux2 FeedForward (SwiGLU)."""

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

        # Store originals as self.X for load_state_dict name matching
        # gate_up sharding: weight is [gate_weights | up_weights] along axis 0.
        # Each half must be sharded independently so SwiGLU gets correct
        # (gate_shard, up_shard) pairs per device, not contiguous row chunks.
        self.linear_in = Linear(
            dim, inner_dim * 2, dtype, devices[0], has_bias=bias
        )
        self.linear_in.weight.sharding_strategy = ShardingStrategy.gate_up(
            self.num_devices, axis=0
        )
        self.linear_in_shards = self.linear_in.shard(devices)

        self.act_fn = Flux2SwiGLU()

        self.linear_out = Linear(
            inner_dim, dim_out, dtype, devices[0], has_bias=bias
        )
        self.linear_out.weight.sharding_strategy = ShardingStrategy.columnwise(
            self.num_devices
        )
        self.linear_out_shards = self.linear_out.shard(devices)

        self.allreduce = Allreduce(num_accelerators=self.num_devices)

    def __call__(
        self,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        outs = [
            self.linear_in_shards[i](xs[i]) for i in range(self.num_devices)
        ]
        outs = [self.act_fn(o) for o in outs]
        outs = [
            self.linear_out_shards[i](outs[i]) for i in range(self.num_devices)
        ]
        return self.allreduce(outs, signal_buffers)


class DistributedFlux2Attention(Module):
    """Tensor-parallel dual-stream Flux2 attention."""

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
        self.local_heads = self.heads // self.num_devices
        self.local_inner_dim = self.local_heads * self.head_dim
        out_dim = out_dim if out_dim is not None else query_dim
        n = self.num_devices

        # Column-parallel Q/K/V — store originals as self.to_q etc.
        self.to_q = Linear(
            query_dim, self.inner_dim, dtype, devices[0], has_bias=bias
        )
        self.to_q.weight.sharding_strategy = ShardingStrategy.rowwise(n)
        self.to_q_shards = self.to_q.shard(devices)

        self.to_k = Linear(
            query_dim, self.inner_dim, dtype, devices[0], has_bias=bias
        )
        self.to_k.weight.sharding_strategy = ShardingStrategy.rowwise(n)
        self.to_k_shards = self.to_k.shard(devices)

        self.to_v = Linear(
            query_dim, self.inner_dim, dtype, devices[0], has_bias=bias
        )
        self.to_v.weight.sharding_strategy = ShardingStrategy.rowwise(n)
        self.to_v_shards = self.to_v.shard(devices)

        # QK norm (replicated)
        self.norm_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_q.sharding_strategy = ShardingStrategy.replicate(n)
        self.norm_q_shards = self.norm_q.shard(devices)

        self.norm_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_k.sharding_strategy = ShardingStrategy.replicate(n)
        self.norm_k_shards = self.norm_k.shard(devices)

        # Row-parallel output projection.
        # The single-GPU model uses `self.to_out = LayerList([Linear(...)])`,
        # so the checkpoint key is `to_out.0.weight`. We must use the same
        # LayerList wrapping so load_state_dict can match the name.
        _to_out_linear = Linear(
            self.inner_dim, out_dim, dtype, devices[0], has_bias=out_bias
        )
        _to_out_linear.weight.sharding_strategy = ShardingStrategy.columnwise(n)
        self.to_out = LayerList(
            [_to_out_linear]
        )  # to_out.0.weight matches checkpoint
        self.to_out_shards = _to_out_linear.shard(devices)

        # Optional: encoder projections for dual-stream
        self.add_q_proj_shards: list[Linear] | None = None
        self.add_k_proj_shards: list[Linear] | None = None
        self.add_v_proj_shards: list[Linear] | None = None
        self.norm_added_q_shards: list[RMSNorm] | None = None
        self.norm_added_k_shards: list[RMSNorm] | None = None
        self.to_add_out_shards: list[Linear] | None = None

        if added_kv_proj_dim is not None:
            proj_bias = False if added_proj_bias is None else added_proj_bias

            self.add_q_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                dtype,
                devices[0],
                has_bias=proj_bias,
            )
            self.add_q_proj.weight.sharding_strategy = ShardingStrategy.rowwise(
                n
            )
            self.add_q_proj_shards = self.add_q_proj.shard(devices)

            self.add_k_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                dtype,
                devices[0],
                has_bias=proj_bias,
            )
            self.add_k_proj.weight.sharding_strategy = ShardingStrategy.rowwise(
                n
            )
            self.add_k_proj_shards = self.add_k_proj.shard(devices)

            self.add_v_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                dtype,
                devices[0],
                has_bias=proj_bias,
            )
            self.add_v_proj.weight.sharding_strategy = ShardingStrategy.rowwise(
                n
            )
            self.add_v_proj_shards = self.add_v_proj.shard(devices)

            self.norm_added_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
            self.norm_added_q.sharding_strategy = ShardingStrategy.replicate(n)
            self.norm_added_q_shards = self.norm_added_q.shard(devices)  # type: ignore[assignment]

            self.norm_added_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
            self.norm_added_k.sharding_strategy = ShardingStrategy.replicate(n)
            self.norm_added_k_shards = self.norm_added_k.shard(devices)  # type: ignore[assignment]

            self.to_add_out = Linear(
                self.inner_dim, query_dim, dtype, devices[0], has_bias=out_bias
            )
            self.to_add_out.weight.sharding_strategy = (
                ShardingStrategy.columnwise(n)
            )
            self.to_add_out_shards = self.to_add_out.shard(devices)

        self.allreduce = Allreduce(num_accelerators=self.num_devices)

    def _per_device_forward(
        self,
        i: int,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue | None,
        image_rotary_emb: tuple[TensorValue, TensorValue] | None,
    ) -> TensorValue | tuple[TensorValue, TensorValue]:
        batch_size = hidden_states.shape[0]

        query = self.to_q_shards[i](hidden_states)
        key = self.to_k_shards[i](hidden_states)
        value = self.to_v_shards[i](hidden_states)

        seq_len = query.shape[1]
        query = ops.reshape(
            query, [batch_size, seq_len, self.local_heads, self.head_dim]
        )
        key = ops.reshape(
            key, [batch_size, seq_len, self.local_heads, self.head_dim]
        )
        value = ops.reshape(
            value, [batch_size, seq_len, self.local_heads, self.head_dim]
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
            enc_seq_len = enc_q.shape[1]

            enc_q = ops.reshape(
                enc_q,
                [batch_size, enc_seq_len, self.local_heads, self.head_dim],
            )
            enc_k = ops.reshape(
                enc_k,
                [batch_size, enc_seq_len, self.local_heads, self.head_dim],
            )
            enc_v = ops.reshape(
                enc_v,
                [batch_size, enc_seq_len, self.local_heads, self.head_dim],
            )

            enc_q = self.norm_added_q_shards[i](enc_q)
            enc_k = self.norm_added_k_shards[i](enc_k)

            query = ops.concat([enc_q, query], axis=1)
            key = ops.concat([enc_k, key], axis=1)
            value = ops.concat([enc_v, value], axis=1)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = _apply_flux2_qk_rope(query, key, cos, sin)

        scale = 1.0 / (self.head_dim**0.5)
        attn_out = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )

        batch_size = attn_out.shape[0]
        seq_len = attn_out.shape[1]
        attn_out = ops.reshape(
            attn_out, [batch_size, seq_len, self.local_inner_dim]
        )
        attn_out = ops.cast(attn_out, query.dtype)

        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_out = attn_out[:, :encoder_seq_len, :]
            hidden_out = attn_out[:, encoder_seq_len:, :]
            hidden_out = self.to_out_shards[i](hidden_out)
            assert self.to_add_out_shards is not None
            encoder_out = self.to_add_out_shards[i](encoder_out)
            return hidden_out, encoder_out
        else:
            return self.to_out_shards[i](attn_out)

    def __call__(
        self,
        hidden_states_list: list[TensorValue],
        signal_buffers: list[BufferValue],
        encoder_hidden_states_list: list[TensorValue] | None = None,
        image_rotary_emb_list: list[tuple[TensorValue, TensorValue]]
        | None = None,
    ) -> list[TensorValue] | tuple[list[TensorValue], list[TensorValue]]:
        results = [
            self._per_device_forward(
                i,
                hidden_states_list[i],
                encoder_hidden_states_list[i]
                if encoder_hidden_states_list is not None
                else None,
                image_rotary_emb_list[i]
                if image_rotary_emb_list is not None
                else None,
            )
            for i in range(self.num_devices)
        ]

        if encoder_hidden_states_list is not None:
            hidden_outs = [r[0] for r in results]
            encoder_outs = [r[1] for r in results]
            hidden_outs = self.allreduce(hidden_outs, signal_buffers)
            encoder_outs = self.allreduce(encoder_outs, signal_buffers)
            return hidden_outs, encoder_outs
        else:
            tensor_results: list[TensorValue] = list(results)  # type: ignore[arg-type]
            return self.allreduce(tensor_results, signal_buffers)


class DistributedFlux2ParallelSelfAttention(Module):
    """Tensor-parallel fused self-attention + MLP for single-stream blocks."""

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
        self.local_heads = self.heads // self.num_devices
        self.local_inner_dim = self.local_heads * self.head_dim
        out_dim = out_dim if out_dim is not None else query_dim

        self.mlp_hidden_dim = int(query_dim * mlp_ratio)
        self.local_mlp_hidden_dim = self.mlp_hidden_dim // self.num_devices
        self.mlp_mult_factor = mlp_mult_factor
        n = self.num_devices

        # Column-parallel fused QKV + MLP projection.
        # The weight rows are [Q | K | V | MLP_gate | MLP_up]. Each sub-matrix
        # must be row-sharded independently (merged sharding) so every device
        # gets the correct head-aligned slices from each sub-matrix.
        fused_dim = self.inner_dim * 3 + self.mlp_hidden_dim * mlp_mult_factor
        self.to_qkv_mlp_proj = Linear(
            query_dim, fused_dim, dtype, devices[0], has_bias=bias
        )
        qkv_mlp_sub_dims = [self.inner_dim] * 3 + [
            self.mlp_hidden_dim
        ] * mlp_mult_factor
        shard_fn = partial(
            _merged_row_sharding_strategy, sub_dims=qkv_mlp_sub_dims
        )
        self.to_qkv_mlp_proj.weight.sharding_strategy = ShardingStrategy(
            num_devices=n, shard=shard_fn
        )
        self.to_qkv_mlp_proj_shards = self.to_qkv_mlp_proj.shard(devices)

        self.mlp_act_fn = Flux2SwiGLU()

        self.norm_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_q.sharding_strategy = ShardingStrategy.replicate(n)
        self.norm_q_shards = self.norm_q.shard(devices)

        self.norm_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_k.sharding_strategy = ShardingStrategy.replicate(n)
        self.norm_k_shards = self.norm_k.shard(devices)

        # Row-parallel fused output (attention + MLP).
        # The input (column) dimension is [attn_shard | mlp_shard] where the
        # attn and MLP parts were sharded independently by the merged row
        # projection above. We must pick the matching non-contiguous columns
        # for each device — same approach as SGLang's _patch_to_out_weight_loader.
        out_in_dim = self.inner_dim + self.mlp_hidden_dim
        self.to_out = Linear(
            out_in_dim, out_dim, dtype, devices[0], has_bias=out_bias
        )
        out_col_shard_fn = partial(
            _merged_col_sharding_strategy,
            sub_dims=[self.inner_dim, self.mlp_hidden_dim],
        )
        self.to_out.weight.sharding_strategy = ShardingStrategy(
            num_devices=n, shard=out_col_shard_fn
        )
        self.to_out_shards = self.to_out.shard(devices)

        self.allreduce = Allreduce(num_accelerators=self.num_devices)

    def _per_device_forward(
        self,
        i: int,
        hidden_states: TensorValue,
        image_rotary_emb: tuple[TensorValue, TensorValue] | None,
    ) -> TensorValue:
        fused = self.to_qkv_mlp_proj_shards[i](hidden_states)

        qkv_dim = self.local_inner_dim * 3
        mlp_dim = self.local_mlp_hidden_dim * self.mlp_mult_factor
        qkv, mlp_hidden_states = ops.split(fused, [qkv_dim, mlp_dim], axis=-1)

        query, key, value = ops.chunk(qkv, 3, axis=-1)

        query = ops.reshape(
            query,
            [query.shape[0], query.shape[1], self.local_heads, self.head_dim],
        )
        key = ops.reshape(
            key, [key.shape[0], key.shape[1], self.local_heads, self.head_dim]
        )
        value = ops.reshape(
            value,
            [value.shape[0], value.shape[1], self.local_heads, self.head_dim],
        )

        query = self.norm_q_shards[i](query)
        key = self.norm_k_shards[i](key)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = _apply_flux2_qk_rope(query, key, cos, sin)

        attn_out = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0 / (self.head_dim**0.5),
        )

        batch_size = attn_out.shape[0]
        seq_len = attn_out.shape[1]
        attn_out = ops.reshape(
            attn_out, [batch_size, seq_len, self.local_inner_dim]
        )
        attn_out = ops.cast(attn_out, query.dtype)

        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)
        hidden_states = ops.concat([attn_out, mlp_hidden_states], axis=-1)

        return self.to_out_shards[i](hidden_states)

    def __call__(
        self,
        hidden_states_list: list[TensorValue],
        signal_buffers: list[BufferValue],
        image_rotary_emb_list: list[tuple[TensorValue, TensorValue]]
        | None = None,
    ) -> list[TensorValue]:
        outs = [
            self._per_device_forward(
                i,
                hidden_states_list[i],
                image_rotary_emb_list[i]
                if image_rotary_emb_list is not None
                else None,
            )
            for i in range(self.num_devices)
        ]
        return self.allreduce(outs, signal_buffers)
