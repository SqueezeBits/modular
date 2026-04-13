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
"""Multi-GPU AlltoAll collective — v1 pull-based P2P kernel.

Each rank's sendbuf holds ``ngpus`` chunks of size ``chunk_num_elems``:
chunk ``i`` is destined for rank ``i``. After alltoall, each rank's
recvbuf holds the chunks it received from every peer, in peer order:
``recvbuf[peer * chunk + elem] = sendbuf_of(peer)[my_rank * chunk + elem]``.

Uses a pull-based approach: each GPU reads the chunk addressed to it
from every peer's sendbuf via P2P. Single kernel launch with a
start/end multi-GPU barrier, matching the pattern used by
``scatter``/``allgather``. ``self`` (``peer == my_rank``) is copied
via the same P2P path for code simplicity; the self-copy cost shows
up in ``algbw`` but is folded out by the standard alltoall bus
bandwidth formula ``busbw = algbw * (n - 1) / n``.
"""

from layout import TileTensor
from layout.tile_layout import TensorLayout
from std.collections import InlineArray
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    grid_dim,
)
from std.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)

from std.math import ceildiv
from std.sys import simd_width_of
from std.utils import StaticTuple

from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    is_p2p_enabled,
)

# --- Pull kernel: each GPU reads its chunk from every peer's sendbuf. ---


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def alltoall_pull_kernel[
    dtype: DType,
    BLOCK_SIZE: Int,
    ngpus: Int,
    simd_width: Int = simd_width_of[dtype, target=get_gpu_target()](),
](
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ],
    chunk_num_elems: Int,
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
):
    """Pull-based AlltoAll: each GPU reads its chunk from every peer.

    For each peer p in [0, ngpus):
        dst = output_ptr + p * chunk_num_elems
        src = input_ptrs[p] + my_rank * chunk_num_elems
    Each (dst, src) pair is a chunk_num_elems-wide P2P copy executed by
    the full grid, vectorized to simd_width where possible.
    """
    var my_sig = rank_sigs[my_rank]

    var global_tid = global_idx.x
    var stride = grid_dim.x * BLOCK_SIZE

    var num_simd_vectors = chunk_num_elems // simd_width
    var tail_start = num_simd_vectors * simd_width

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

        # Pull chunk destined for my_rank from every peer (including self).
        comptime for peer in range(ngpus):
            var src_base = input_ptrs[peer] + my_rank * chunk_num_elems
            var dst_base = output_ptr + peer * chunk_num_elems

            # Grid-strided vectorized copy.
            for idx in range(global_tid, num_simd_vectors, stride):
                var elem_idx = idx * simd_width
                dst_base.store[width=simd_width](
                    elem_idx,
                    src_base.load[width=simd_width](elem_idx),
                )

            # Tail elements (only active when chunk_num_elems % simd_width != 0).
            var tail_idx = tail_start + global_tid
            if tail_idx < chunk_num_elems:
                dst_base.store[width=1](
                    tail_idx,
                    src_base.load[width=1](tail_idx),
                )

        _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


# --- Wrapper functions ---


@always_inline
@parameter
def alltoall[
    dtype: DType,
    //,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    pdl_level: PDLLevel = PDLLevel(),
](
    input_buffers: InlineArray[TileTensor[dtype, in_layout, in_origin], ngpus],
    output_buffer: TileTensor[mut=True, dtype, ...],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
) raises:
    """Pull-based AlltoAll across ngpus GPUs.

    Each rank's ``input_buffers[my_rank]`` holds ``ngpus`` contiguous
    chunks of equal size; chunk ``i`` is destined for rank ``i``.
    After the call, ``output_buffer`` contains ``ngpus`` chunks where
    chunk ``p`` is the data received from peer ``p``. All ranks must
    call this function with ``input_buffers`` populated with the
    P2P-addressable pointers from every rank.

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        pdl_level: Controls PDL behavior for P2P kernels.

    Args:
        input_buffers: Input send buffers, one per rank, each shaped
            ``ngpus * chunk``. Peer buffers must be P2P-addressable
            from this GPU's context.
        output_buffer: Output receive buffer for THIS GPU, shaped
            ``ngpus * chunk`` matching the input size.
        rank_sigs: Per-GPU Signal pointers for synchronization.
        ctx: Device context for THIS GPU.

    Raises:
        Error: If P2P access is not available between GPUs.
        Error: If the input buffer size is not divisible by ``ngpus``.
    """
    comptime assert ngpus >= 2, "alltoall requires at least 2 GPUs"

    if not is_p2p_enabled():
        raise Error("AlltoAll currently requires P2P access between GPUs")

    var per_rank_elems = input_buffers[0].num_elements()
    if per_rank_elems % ngpus != 0:
        raise Error(
            "AlltoAll input buffer size must be divisible by ngpus"
        )
    var chunk_num_elems = per_rank_elems // ngpus
    if chunk_num_elems == 0:
        return

    # Extract raw pointers for every rank's sendbuf.
    var input_ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ](fill={_unsafe_null = ()})
    for i in range(ngpus):
        input_ptrs[i] = rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
            input_buffers[i].ptr
        )

    comptime BLOCK_SIZE = 256
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    var grid_size = min(
        ceildiv(ceildiv(chunk_num_elems, simd_width), BLOCK_SIZE),
        MAX_NUM_BLOCKS_UPPER_BOUND,
    )

    comptime kernel = alltoall_pull_kernel[dtype, BLOCK_SIZE, ngpus]

    ctx.enqueue_function[kernel, kernel](
        rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](output_buffer.ptr),
        input_ptrs,
        chunk_num_elems,
        rank_sigs,
        Int(ctx.id()),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(pdl_level),
    )
