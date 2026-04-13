# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
#
# Minimal benchmark comparing Mojo P2P scatter vs NCCL scatter (2.28.3+).
# Root GPU 0 holds ngpus * per_rank_length elements; each rank i receives
# chunk i of per_rank_length elements. Uses dp_size == ngpus (no DP
# broadcast, pure scatter) on the Mojo side.

from std.collections import InlineArray
from std.sys.defines import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
)
from std.sys import size_of, simd_width_of

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from layout import Idx, TileTensor, row_major
from comm.sync import enable_p2p
from comm.scatter import scatter
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from comm.vendor.ccl import (
    _ccl_scatter,
    _dtype_to_ccl,
    _get_global_comms,
)
from std.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import arg_parse, human_readable_size, CacheBustingBuffer

from std.testing import assert_true


def bench_scatter_ccl[
    dtype: DType,
    ngpus: Int,
    *,
    cache_busting: Bool,
    use_vendor_ccl: Bool,
](
    mut b: Bench,
    list_of_ctx: List[DeviceContext],
    num_bytes: Int,
) raises:
    """num_bytes is per-rank receive size."""
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"

    var recv_length = num_bytes // size_of[dtype]()
    # Root holds ngpus chunks; total scattered bytes == ngpus * num_bytes.
    var total_bytes = num_bytes * ngpus

    var vendorccl_tag = "-vendorccl" if use_vendor_ccl else ""
    var name = String(
        "scatter-",
        dtype,
        "-",
        ngpus,
        "gpus",
        vendorccl_tag,
        "-",
        human_readable_size(total_bytes),
    )
    print("Running " + name)

    comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()

    # Root (GPU 0) input chunks: ngpus separate CacheBusting buffers, each
    # of size recv_length. This matches the Mojo scatter API which takes
    # dp_size separate TileTensors; for NCCL we'll stitch them into a
    # conceptual contiguous sendbuf by calling _ccl_scatter with the first
    # buffer's pointer (only meaningful if all chunks are contiguous).
    #
    # To keep NCCL's contiguous-sendbuf assumption safe, we allocate ONE
    # root buffer of size ngpus*recv_length for the NCCL path, and ngpus
    # separate buffers for the Mojo path. Both paths use the Mojo-style
    # array for TileTensor construction; the NCCL code path ignores it.
    var cb_root_chunks = List[CacheBustingBuffer[dtype]]()
    var nccl_root_buf: DeviceBuffer[dtype]

    comptime if use_vendor_ccl:
        nccl_root_buf = list_of_ctx[0].enqueue_create_buffer[dtype](
            ngpus * recv_length
        )
        # Dummy single CB buffer of minimum size to satisfy construction
        # paths; not used on the NCCL code path.
        cb_root_chunks.append(
            CacheBustingBuffer[dtype](
                recv_length, simd_size, list_of_ctx[0], cache_busting
            )
        )
    else:
        # Mojo path: allocate ngpus separate chunks on GPU 0.
        nccl_root_buf = list_of_ctx[0].enqueue_create_buffer[dtype](1)
        for _c in range(ngpus):
            cb_root_chunks.append(
                CacheBustingBuffer[dtype](
                    recv_length, simd_size, list_of_ctx[0], cache_busting
                )
            )

    # Per-GPU output buffer (recv_length each).
    var out_bufs = List[DeviceBuffer[dtype]](capacity=ngpus)
    # Signal buffers.
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )

    for gpu_idx in range(ngpus):
        out_bufs.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](recv_length)
        )
        signal_buffers.append(
            list_of_ctx[gpu_idx].create_buffer_sync[DType.uint8](
                size_of[Signal]()
            )
        )
        list_of_ctx[gpu_idx].enqueue_memset[DType.uint8](
            signal_buffers[gpu_idx], 0
        )
        rank_sigs[gpu_idx] = (
            signal_buffers[gpu_idx].unsafe_ptr().bitcast[Signal]()
        )
        list_of_ctx[gpu_idx].synchronize()

    # TileTensor arrays for Mojo path.
    comptime InTileType = type_of(
        TileTensor(
            cb_root_chunks[0].unsafe_ptr(), row_major(Idx(recv_length))
        ).as_immut()
    )
    var tt_in = InlineArray[InTileType, ngpus](uninitialized=True)

    comptime OutTileType = type_of(
        TileTensor(out_bufs[0].unsafe_ptr(), row_major(Idx(recv_length)))
    )
    var tt_out = InlineArray[OutTileType, ngpus](uninitialized=True)
    for gpu_idx in range(ngpus):
        tt_out[gpu_idx] = TileTensor(
            out_bufs[gpu_idx].unsafe_ptr(), row_major(Idx(recv_length))
        )

    # NCCL comm pre-init.
    comptime if use_vendor_ccl:
        if not vendor_ccl.is_scatter_available():
            raise "Vendor CCL scatter not available (needs NCCL 2.28.3+)."
        vendor_ccl.init_comms(ngpus)

    @parameter
    @always_inline
    def bench_iter(
        mut bencher: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises:
        @parameter
        @always_inline
        def call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            comptime if use_vendor_ccl:
                var dtype_ccl = _dtype_to_ccl[dtype]()
                var comms = _get_global_comms(ngpus)
                var device_rank = Int(ctx_inner.id())
                var sendbuf = nccl_root_buf.unsafe_ptr()  # meaningful only on root
                var recvbuf = out_bufs[device_rank].unsafe_ptr()
                _ = _ccl_scatter(
                    sendbuf.bitcast[NoneType](),
                    recvbuf.bitcast[NoneType](),
                    recv_length,
                    dtype_ccl,
                    0,  # root
                    comms.comms[device_rank],
                    ctx_inner,
                )
            else:
                comptime for dp_idx in range(ngpus):
                    tt_in[dp_idx] = TileTensor(
                        cb_root_chunks[dp_idx].offset_ptr(cache_iter),
                        row_major(Idx(recv_length)),
                    ).as_immut()
                scatter[ngpus=ngpus, dp_size=ngpus](
                    tt_in,
                    tt_out[ctx_idx],
                    rank_sigs,
                    ctx_inner,
                )

        bencher.iter_custom[call_fn](ctx)

    b.bench_multicontext[bench_iter](
        list_of_ctx,
        BenchId(name),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )
    b.dump_report()

    var max_time = b.info_vec[0].result.mean(unit="ms")
    var gbps = Float64(total_bytes) / (max_time * 1000 * 1000)
    # For scatter, busbw = algbw (data leaves root once).
    var busbw = gbps
    print(
        "|",
        name,
        "| slowest mean time",
        max_time,
        "ms |",
        "algbw:",
        gbps,
        "GB/s |",
        "busbw:",
        busbw,
        "GB/s |",
    )

    _ = signal_buffers^
    _ = cb_root_chunks^
    _ = out_bufs^
    _ = nccl_root_buf^


def main() raises:
    var num_bytes = arg_parse("num_bytes", 1 * 1024 * 1024)

    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime num_gpus = get_defined_int["num_gpus", 2]()
    comptime use_vendor_ccl = get_defined_bool["use_vendor_ccl", False]()
    comptime cache_busting = True

    var m = Bench()

    var num_gpus_found = DeviceContext.number_of_devices()
    assert_true(
        num_gpus_found >= num_gpus,
        String(num_gpus_found) + " devices found, expected " + String(num_gpus),
    )
    assert_true(num_bytes % size_of[dtype]() == 0)

    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    if not enable_p2p():
        print("P2P not enabled, skipping benchmark.")
        return

    bench_scatter_ccl[
        dtype=dtype,
        ngpus=num_gpus,
        cache_busting=cache_busting,
        use_vendor_ccl=use_vendor_ccl,
    ](m, ctx, num_bytes)
