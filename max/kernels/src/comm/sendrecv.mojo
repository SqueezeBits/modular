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
"""Multi-GPU Ring Send/Recv collective — pull-based P2P kernel.

Each rank's ``input_buffer`` holds ``num_bytes`` of data. One ring hop:
rank ``r`` sends its buffer to ``(r + 1) % ngpus`` and receives into its
``output_buffer`` from ``(r - 1 + ngpus) % ngpus``. Implemented as a
pull: the receiver reads directly from the sender's ``input_buffer`` via
P2P. All ``ngpus`` ranks participate concurrently, so a single kernel
launch with a start/end multi-GPU barrier completes one ring step.

This is intentionally a *ring-pattern* primitive, not a generic
point-to-point send/recv. It matches the communication shape used by
Ring Attention style algorithms and by the ``bench_sendrecv_ccl.mojo``
benchmark, where every rank exchanges with its two neighbors
simultaneously. A fully generic pairwise send/recv (arbitrary source/dest)
is not provided — those use cases should go through the vendor NCCL
``ncclSend``/``ncclRecv`` path until a dedicated Mojo primitive exists.
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

# --- Pull kernel: each rank reads one neighbor's sendbuf. ---


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def ring_sendrecv_pull_kernel[
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
    """Copy ``chunk_num_elems`` elements from the previous neighbor's
    sendbuf into this rank's recvbuf, gated by a full-grid barrier.
    """
    var my_sig = rank_sigs[my_rank]

    # Each rank reads from (my_rank - 1 + ngpus) % ngpus.
    var prev_rank = (my_rank - 1 + ngpus) % ngpus
    var src_base = input_ptrs[prev_rank]
    var dst_base = output_ptr

    var global_tid = global_idx.x
    var stride = grid_dim.x * BLOCK_SIZE

    var num_simd_vectors = chunk_num_elems // simd_width
    var tail_start = num_simd_vectors * simd_width

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

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
def ring_sendrecv[
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
    """Pull-based ring send/recv across ngpus GPUs.

    On each rank ``r``, ``input_buffers[r]`` is the send payload and
    ``output_buffer`` is the receive destination. After the call,
    ``output_buffer`` of rank ``r`` contains the bytes that were in
    ``input_buffers[(r - 1 + ngpus) % ngpus]`` before the call.
    All ranks must call this function with the same
    ``input_buffers`` array populated with the P2P-addressable pointers
    from every rank.

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        pdl_level: Controls PDL behavior for P2P kernels.

    Args:
        input_buffers: Send buffers, one per rank, each holding
            ``chunk_num_elems`` elements. Peer buffers must be
            P2P-addressable from this GPU's context.
        output_buffer: Receive buffer for THIS GPU, same size as the
            input buffers.
        rank_sigs: Per-GPU Signal pointers for synchronization.
        ctx: Device context for THIS GPU.
        _max_num_blocks: Optional cap on the number of thread blocks
            launched. Defaults to ``MAX_NUM_BLOCKS_UPPER_BOUND``.

    Raises:
        Error: If P2P access is not available between GPUs.
    """
    comptime assert ngpus >= 2, "ring_sendrecv requires at least 2 GPUs"

    if not is_p2p_enabled():
        raise Error(
            "ring_sendrecv currently requires P2P access between GPUs"
        )

    var chunk_num_elems = input_buffers[0].num_elements()
    if chunk_num_elems == 0:
        return

    # Extract raw pointers for every rank's sendbuf so the kernel can
    # address this rank's previous neighbor.
    var input_ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ](fill={_unsafe_null = ()})
    for i in range(ngpus):
        input_ptrs[i] = rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
            input_buffers[i].ptr
        )

    comptime BLOCK_SIZE = 256
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()

    var cap = MAX_NUM_BLOCKS_UPPER_BOUND
    if _max_num_blocks:
        cap = min(cap, _max_num_blocks.value())

    var grid_size = min(
        ceildiv(ceildiv(chunk_num_elems, simd_width), BLOCK_SIZE),
        cap,
    )
    if grid_size < 1:
        grid_size = 1

    comptime kernel = ring_sendrecv_pull_kernel[dtype, BLOCK_SIZE, ngpus]

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
