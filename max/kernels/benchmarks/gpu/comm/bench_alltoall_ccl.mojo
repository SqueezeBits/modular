# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
#
# Minimal benchmark for AlltoAll. Uses the ncclAlltoAll binding (NCCL
# 2.28.3+) from comm.vendor.ccl. A Mojo P2P kernel does not yet exist
# in comm/; the dual-path scaffolding is kept here so the Mojo branch
# can be filled in later without rewriting the benchmark.
#
# num_bytes is per-rank per-peer chunk size. Total per-rank traffic
# (sent == received) is ngpus * num_bytes.

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
from comm.sync import enable_p2p
from comm.alltoall import alltoall
from comm import MAX_GPUS, Signal
from layout import Idx, TileTensor, row_major
import comm.vendor.ccl as vendor_ccl
from comm.vendor.ccl import (
    _ccl_alltoall,
    _dtype_to_ccl,
    _get_global_comms,
)
from std.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import arg_parse, human_readable_size, CacheBustingBuffer

from std.testing import assert_true


def bench_alltoall_ccl[
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
    """num_bytes is per-chunk size (ngpus chunks per rank on send/recv side)."""
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"

    var chunk_length = num_bytes // size_of[dtype]()
    var per_rank_length = chunk_length * ngpus
    var per_rank_bytes = per_rank_length * size_of[dtype]()
    # Throughput: per-rank send (== recv) volume in one collective.
    var total_bytes = per_rank_bytes

    var vendorccl_tag = "-vendorccl" if use_vendor_ccl else ""
    var name = String(
        "alltoall-",
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

    # Per-GPU sendbuf: ngpus chunks = per_rank_length elements. Cache-busted.
    var cb_sends = List[CacheBustingBuffer[dtype]]()
    # Per-GPU recvbuf: same size.
    var recv_bufs = List[DeviceBuffer[dtype]](capacity=ngpus)
    # Signal buffers (unused on NCCL path, kept for structural symmetry with
    # the forthcoming Mojo kernel path).
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )

    for gpu_idx in range(ngpus):
        cb_sends.append(
            CacheBustingBuffer[dtype](
                per_rank_length, simd_size, list_of_ctx[gpu_idx], cache_busting
            )
        )
        recv_bufs.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](per_rank_length)
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

    # TileTensors for the Mojo path: one immut view per rank's sendbuf and
    # one mut view of this rank's recvbuf.
    comptime InTileType = type_of(
        TileTensor(
            cb_sends[0].unsafe_ptr(), row_major(Idx(per_rank_length))
        ).as_immut()
    )
    var tt_in = InlineArray[InTileType, ngpus](uninitialized=True)

    comptime OutTileType = type_of(
        TileTensor(recv_bufs[0].unsafe_ptr(), row_major(Idx(per_rank_length)))
    )
    var tt_out = InlineArray[OutTileType, ngpus](uninitialized=True)
    for gpu_idx in range(ngpus):
        tt_out[gpu_idx] = TileTensor(
            recv_bufs[gpu_idx].unsafe_ptr(), row_major(Idx(per_rank_length))
        )

    # NCCL comm pre-init.
    comptime if use_vendor_ccl:
        if not vendor_ccl.is_alltoall_available():
            raise "Vendor CCL alltoall not available (requires NCCL 2.28.3+)."
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
                var sendbuf = cb_sends[device_rank].offset_ptr(cache_iter)
                var recvbuf = recv_bufs[device_rank].unsafe_ptr()
                _ = _ccl_alltoall(
                    sendbuf.bitcast[NoneType](),
                    recvbuf.bitcast[NoneType](),
                    chunk_length,
                    dtype_ccl,
                    comms.comms[device_rank],
                    ctx_inner,
                )
            else:
                comptime for i in range(ngpus):
                    tt_in[i] = TileTensor(
                        cb_sends[i].offset_ptr(cache_iter),
                        row_major(Idx(per_rank_length)),
                    ).as_immut()
                alltoall[ngpus=ngpus](
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
    # For alltoall, busbw = algbw * (n-1)/n (self-chunk stays local).
    var busbw = gbps * Float64(ngpus - 1) / Float64(ngpus)
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
    _ = cb_sends^
    _ = recv_bufs^


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

    bench_alltoall_ccl[
        dtype=dtype,
        ngpus=num_gpus,
        cache_busting=cache_busting,
        use_vendor_ccl=use_vendor_ccl,
    ](m, ctx, num_bytes)
