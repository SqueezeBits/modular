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
"""Multi-GPU AlltoAll collective — v2 block-parallel pull kernel.

Each rank's sendbuf holds ``ngpus`` chunks of size ``chunk_num_elems``:
chunk ``i`` is destined for rank ``i``. After alltoall, each rank's
recvbuf holds the chunks it received from every peer, in peer order:
``recvbuf[peer * chunk + elem] = sendbuf_of(peer)[my_rank * chunk + elem]``.

v2 (current): the grid is partitioned per peer. Let
``total_blocks = blocks_per_peer * ngpus``; block ``b`` handles only
peer ``b // blocks_per_peer`` and iterates over the ``chunk_num_elems``
elements addressed to ``my_rank`` within that peer's sendbuf. This
drives all ``ngpus`` remote NVLink reads concurrently at the grid
level, which v1 (serial per-peer loop inside every block) did not.

v1 (historic): single loop ``for peer in range(ngpus)`` inside each
block. All blocks read from peer 0 first, then peer 1, etc. — only
one remote link saturated at a time. See ``alltoall_v1_benchmark.md``.
"""

from layout import TileTensor
from layout.tile_layout import TensorLayout
from std.collections import InlineArray
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
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

# --- v2 kernel: block-level peer partitioning ---


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
    blocks_per_peer: Int,
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
):
    """Block-parallel pull: each block copies one peer's chunk slice.

    The launch uses ``total_blocks = blocks_per_peer * ngpus`` blocks on
    a single 1D grid (so the existing ``_multi_gpu_barrier`` slot
    indexing by ``blockIdx.x`` still fits inside
    ``MAX_NUM_BLOCKS_UPPER_BOUND``). Block ``b`` is assigned
    ``peer = b // blocks_per_peer`` and works on the portion
    ``[block_in_peer * BLOCK_SIZE, chunk_num_elems)`` of peer ``peer``'s
    chunk via grid-stride with stride ``blocks_per_peer * BLOCK_SIZE``.
    """
    var my_sig = rank_sigs[my_rank]

    # Decompose flat blockIdx.x into (peer, block_in_peer).
    var peer = block_idx.x // blocks_per_peer
    var block_in_peer = block_idx.x % blocks_per_peer

    var src_base = input_ptrs[peer] + my_rank * chunk_num_elems
    var dst_base = output_ptr + peer * chunk_num_elems

    var tid_in_peer = block_in_peer * BLOCK_SIZE + thread_idx.x
    var stride = blocks_per_peer * BLOCK_SIZE

    var num_simd_vectors = chunk_num_elems // simd_width
    var tail_start = num_simd_vectors * simd_width

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

        # Grid-strided vectorized copy within this peer's chunk.
        for idx in range(tid_in_peer, num_simd_vectors, stride):
            var elem_idx = idx * simd_width
            dst_base.store[width=simd_width](
                elem_idx,
                src_base.load[width=simd_width](elem_idx),
            )

        # Tail elements (only active when chunk_num_elems % simd_width != 0).
        var tail_idx = tail_start + tid_in_peer
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
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Pull-based AlltoAll across ngpus GPUs (v2 block-parallel).

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
        _max_num_blocks: Optional cap on the total number of thread
            blocks launched. If provided, the launch will use up to
            ``floor(_max_num_blocks / ngpus) * ngpus`` blocks (i.e. the
            cap is applied to the product ``blocks_per_peer * ngpus``).
            If None, defaults to ``MAX_NUM_BLOCKS_UPPER_BOUND``.

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

    # blocks_per_peer is sized to cover one chunk with one grid-stride pass,
    # then capped so that blocks_per_peer * ngpus <= MAX_NUM_BLOCKS_UPPER_BOUND
    # (or the caller-supplied cap).
    var cap_total = MAX_NUM_BLOCKS_UPPER_BOUND
    if _max_num_blocks:
        cap_total = min(cap_total, _max_num_blocks.value())

    var blocks_for_chunk = ceildiv(
        ceildiv(chunk_num_elems, simd_width), BLOCK_SIZE
    )
    var blocks_per_peer = min(blocks_for_chunk, cap_total // ngpus)
    if blocks_per_peer < 1:
        blocks_per_peer = 1
    var total_blocks = blocks_per_peer * ngpus

    comptime kernel = alltoall_pull_kernel[dtype, BLOCK_SIZE, ngpus]

    ctx.enqueue_function[kernel, kernel](
        rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](output_buffer.ptr),
        input_ptrs,
        chunk_num_elems,
        blocks_per_peer,
        rank_sigs,
        Int(ctx.id()),
        grid_dim=total_blocks,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(pdl_level),
    )
