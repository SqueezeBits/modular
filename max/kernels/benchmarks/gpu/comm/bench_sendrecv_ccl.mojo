# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
#
# Minimal benchmark for NCCL point-to-point Send/Recv measured in a ring
# pattern: each rank sends to (rank + 1) % ngpus and receives from
# (rank - 1 + ngpus) % ngpus, wrapped in a single ncclGroupStart/End so
# the send and recv are submitted together (required to avoid deadlock).
#
# NCCL-only: Modular has no Mojo point-to-point send/recv primitive, so
# there is nothing to compare against on the Mojo side. The data gathered
# here feeds the "true Ring Attention could save X" analysis.
#
# num_bytes is the size of one ring hop. Per-rank traffic is num_bytes
# sent + num_bytes received = 2 * num_bytes per iteration.

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
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from comm.vendor.ccl import (
    _ccl_send,
    _ccl_recv,
    _dtype_to_ccl,
    _get_global_comms,
)
from std.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import arg_parse, human_readable_size, CacheBustingBuffer

from std.testing import assert_true


def bench_sendrecv_ccl[
    dtype: DType,
    ngpus: Int,
    *,
    cache_busting: Bool,
](
    mut b: Bench,
    list_of_ctx: List[DeviceContext],
    num_bytes: Int,
) raises:
    """One ring hop: rank r sends num_bytes to (r+1) and recvs from (r-1)."""
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"

    var chunk_length = num_bytes // size_of[dtype]()
    # algbw is computed from num_bytes (one direction); busbw factor is 1
    # for point-to-point send/recv (nccl-tests SendRecv convention).
    var total_bytes = num_bytes

    var name = String(
        "sendrecv-",
        dtype,
        "-",
        ngpus,
        "gpus-vendorccl-",
        human_readable_size(total_bytes),
    )
    print("Running " + name)

    comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()

    # Per-rank sendbuf (cache-busted) and recvbuf.
    var cb_sends = List[CacheBustingBuffer[dtype]]()
    var recv_bufs = List[DeviceBuffer[dtype]](capacity=ngpus)
    # Signal buffers (kept for structural parity; not used on the NCCL path).
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )

    for gpu_idx in range(ngpus):
        cb_sends.append(
            CacheBustingBuffer[dtype](
                chunk_length, simd_size, list_of_ctx[gpu_idx], cache_busting
            )
        )
        recv_bufs.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](chunk_length)
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

    if not vendor_ccl.is_send_available() or not vendor_ccl.is_recv_available():
        raise "Vendor CCL send/recv not available (requires NCCL 2.7+)."
    vendor_ccl.init_comms(ngpus)

    @parameter
    @always_inline
    def bench_iter(
        mut bencher: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises:
        @parameter
        @always_inline
        def call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            var dtype_ccl = _dtype_to_ccl[dtype]()
            var comms = _get_global_comms(ngpus)
            var device_rank = Int(ctx_inner.id())
            var next_peer = (device_rank + 1) % ngpus
            var prev_peer = (device_rank - 1 + ngpus) % ngpus
            var sendbuf = cb_sends[device_rank].offset_ptr(cache_iter)
            var recvbuf = recv_bufs[device_rank].unsafe_ptr()

            # Batch send + recv in one ncclGroupStart/End to avoid deadlock.
            with vendor_ccl.group():
                _ = _ccl_send(
                    sendbuf.bitcast[NoneType](),
                    chunk_length,
                    dtype_ccl,
                    next_peer,
                    comms.comms[device_rank],
                    ctx_inner,
                )
                _ = _ccl_recv(
                    recvbuf.bitcast[NoneType](),
                    chunk_length,
                    dtype_ccl,
                    prev_peer,
                    comms.comms[device_rank],
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
    # Point-to-point send/recv: busbw == algbw (no reduce/scatter factor).
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
    _ = cb_sends^
    _ = recv_bufs^


def main() raises:
    var num_bytes = arg_parse("num_bytes", 1 * 1024 * 1024)

    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime num_gpus = get_defined_int["num_gpus", 2]()
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

    bench_sendrecv_ccl[
        dtype=dtype,
        ngpus=num_gpus,
        cache_busting=cache_busting,
    ](m, ctx, num_bytes)
