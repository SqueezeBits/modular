# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
#
# Minimal benchmark comparing Mojo P2P reducescatter vs NCCL reducescatter
# for uniform per-rank output sizes. Companion to bench_allgather_ccl.mojo.
# NCCL path uses the ncclReduceScatter binding added to comm.vendor.ccl.

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
from comm.reducescatter import reducescatter
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from comm.vendor.ccl import (
    _ccl_reducescatter,
    _dtype_to_ccl,
    _get_global_comms,
    ncclRedOp_t,
)
from std.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import arg_parse, human_readable_size, CacheBustingBuffer

from std.testing import assert_true


def bench_reducescatter_ccl[
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
    """num_bytes is per-rank OUTPUT size (the scattered shard)."""
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"

    var out_length = num_bytes // size_of[dtype]()
    var in_length = out_length * ngpus
    var in_bytes = in_length * size_of[dtype]()
    # Throughput-wise, full reduction-target size is the useful quantity.
    var total_bytes = in_bytes

    var vendorccl_tag = "-vendorccl" if use_vendor_ccl else ""
    var name = String(
        "reducescatter-",
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

    # Per-GPU full-size cache-busting inputs (ngpus * out_length elements).
    var cb_inputs = List[CacheBustingBuffer[dtype]]()
    # Per-GPU per-rank output buffer (Mojo path only needs this).
    var out_bufs = List[DeviceBuffer[dtype]](capacity=ngpus)
    # Signal buffers.
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )

    for gpu_idx in range(ngpus):
        cb_inputs.append(
            CacheBustingBuffer[dtype](
                in_length, simd_size, list_of_ctx[gpu_idx], cache_busting
            )
        )
        out_bufs.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](out_length)
        )
        signal_buffers.append(
            list_of_ctx[gpu_idx].create_buffer_sync[DType.uint8](
                size_of[Signal]() + ngpus * in_bytes
            )
        )
        list_of_ctx[gpu_idx].enqueue_memset[DType.uint8](
            signal_buffers[gpu_idx], 0
        )
        rank_sigs[gpu_idx] = (
            signal_buffers[gpu_idx].unsafe_ptr().bitcast[Signal]()
        )

    # TileTensors for Mojo path.
    comptime InTileType = type_of(
        TileTensor(
            cb_inputs[0].unsafe_ptr(), row_major(Idx(in_length))
        ).as_immut()
    )
    var tt_in = InlineArray[InTileType, ngpus](uninitialized=True)

    comptime OutTileType = type_of(
        TileTensor(out_bufs[0].unsafe_ptr(), row_major(Idx(out_length)))
    )
    var tt_out = InlineArray[OutTileType, ngpus](uninitialized=True)

    for gpu_idx in range(ngpus):
        tt_out[gpu_idx] = TileTensor(
            out_bufs[gpu_idx].unsafe_ptr(), row_major(Idx(out_length))
        )
        list_of_ctx[gpu_idx].synchronize()

    # NCCL comm pre-init.
    comptime if use_vendor_ccl:
        if not vendor_ccl.is_reducescatter_available():
            raise "Vendor CCL reducescatter not available."
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
                var sendbuf = cb_inputs[device_rank].offset_ptr(cache_iter)
                var recvbuf = out_bufs[device_rank].unsafe_ptr()
                _ = _ccl_reducescatter(
                    sendbuf.bitcast[NoneType](),
                    recvbuf.bitcast[NoneType](),
                    out_length,
                    dtype_ccl,
                    ncclRedOp_t.ncclSum,
                    comms.comms[device_rank],
                    ctx_inner,
                )
            else:
                comptime for i in range(ngpus):
                    tt_in[i] = TileTensor(
                        cb_inputs[i].offset_ptr(cache_iter),
                        row_major(Idx(in_length)),
                    ).as_immut()
                reducescatter[ngpus=ngpus](
                    tt_in,
                    tt_out[ctx_idx],
                    rank_sigs,
                    ctx_inner,
                    Optional[Int](),
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
    # For reducescatter, busbw = algbw * (n-1)/n.
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
    _ = cb_inputs^
    _ = out_bufs^


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

    bench_reducescatter_ccl[
        dtype=dtype,
        ngpus=num_gpus,
        cache_busting=cache_busting,
        use_vendor_ccl=use_vendor_ccl,
    ](m, ctx, num_bytes)
