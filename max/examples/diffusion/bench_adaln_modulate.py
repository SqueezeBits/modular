#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Benchmark: unfused vs fused AdaLN modulate implementations.

Unfused (original):
    norm_x = LayerNorm(x, gamma=1, beta=0)        # standalone kernel
    out    = (1 + scale) * norm_x + shift          # elementwise kernel

Fused (mo.adaln_modulate):
    out    = adaln_modulate(x, scale, shift)        # single kernel

Usage:
    python bench_adaln_modulate.py
    python bench_adaln_modulate.py --seq-len 512 --seq-len 4096 --seq-len 4608 --warmup 10 --iters 50
    python bench_adaln_modulate.py --trace-dir traces/
"""

from __future__ import annotations

import argparse
import os
import time
from typing import NamedTuple

import numpy as np
import torch

from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_unfused_graph(
    B: int, S: int, D: int, dtype: DType, device: DeviceRef
) -> Graph:
    """Original two-step AdaLN: LayerNorm then elementwise (1+scale)*x+shift."""
    with Graph(
        "adaln_unfused",
        input_types=[
            TensorType(dtype, [B, S, D], device),   # x
            TensorType(dtype, [B, 1, D], device),   # scale
            TensorType(dtype, [B, 1, D], device),   # shift
        ],
    ) as g:
        x, scale, shift = (v.tensor for v in g.inputs)

        # LayerNorm with gamma=1, beta=0 (elementwise_affine=False)
        gamma = ops.constant(
            np.ones(D, dtype=np.float32), DType.float32, device
        ).cast(dtype)
        beta = ops.constant(
            np.zeros(D, dtype=np.float32), DType.float32, device
        ).cast(dtype)
        norm_x = ops.layer_norm(x, gamma=gamma, beta=beta, epsilon=1e-6)

        # Elementwise modulate: (1 + scale) * norm_x + shift
        # scale/shift are [B, 1, D], norm_x is [B, S, D] — broadcast over S
        one = ops.constant(1.0, dtype, device)
        out = (one + scale) * norm_x + shift

        g.output(out)
    return g


def build_fused_graph(
    B: int, S: int, D: int, dtype: DType, device: DeviceRef
) -> Graph:
    """Fused single-kernel AdaLN: mo.adaln_modulate."""
    with Graph(
        "adaln_fused",
        input_types=[
            TensorType(dtype, [B, S, D], device),   # x
            TensorType(dtype, [B, 1, D], device),   # scale
            TensorType(dtype, [B, 1, D], device),   # shift
        ],
    ) as g:
        x, scale, shift = (v.tensor for v in g.inputs)

        epsilon = ops.constant(1e-6, dtype, DeviceRef.CPU())
        out = ops.custom(
            "mo.adaln_modulate",
            device=device,
            values=[x, scale, shift, epsilon],
            out_types=[TensorType(dtype=dtype, shape=[B, S, D], device=device)],
        )[0].tensor

        g.output(out)
    return g


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------


class BenchResult(NamedTuple):
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    times_ms: list[float]


_DTYPE_TO_TORCH = {
    DType.float32: torch.float32,
    DType.float16: torch.float16,
    DType.bfloat16: torch.bfloat16,
}


def make_inputs(B: int, S: int, D: int, dtype: DType, device: Accelerator) -> tuple[Buffer, Buffer, Buffer]:
    """Create random Buffer inputs on the accelerator."""
    rng = np.random.default_rng(42)
    torch_dtype = _DTYPE_TO_TORCH[dtype]
    x_np = rng.standard_normal((B, S, D)).astype(np.float32)
    scale_np = rng.standard_normal((B, 1, D)).astype(np.float32)
    shift_np = rng.standard_normal((B, 1, D)).astype(np.float32)

    def to_buf(arr: np.ndarray) -> Buffer:
        t = torch.from_numpy(arr).to(torch_dtype).cuda()
        return Buffer.from_dlpack(t)

    return to_buf(x_np), to_buf(scale_np), to_buf(shift_np)


def bench_model(
    model,
    inputs: tuple[Buffer, Buffer, Buffer],
    warmup: int = 5,
    iters: int = 20,
) -> BenchResult:
    """Warm up then time `iters` executions."""
    for _ in range(warmup):
        model.execute(*inputs)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.execute(*inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    arr = np.array(times)
    return BenchResult(
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        times_ms=times,
    )


def profile_model(
    model,
    inputs: tuple[Buffer, Buffer, Buffer],
    trace_path: str,
    warmup: int = 3,
    iters: int = 5,
) -> None:
    """Run model under torch.profiler and export a Chrome trace JSON."""
    for _ in range(warmup):
        model.execute(*inputs)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            model.execute(*inputs)
        torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"    trace → {trace_path}")


def buf_to_float32(buf: Buffer) -> np.ndarray:
    """Convert a Buffer (any float dtype) to a float32 numpy array."""
    t = torch.from_dlpack(buf.to(CPU()))
    return t.float().numpy()


def check_correctness(
    unfused_model, fused_model, inputs: tuple[Buffer, Buffer, Buffer]
) -> float:
    """Return max absolute difference between unfused and fused outputs."""
    out_unfused = unfused_model.execute(*inputs)[0]
    out_fused = fused_model.execute(*inputs)[0]
    a = buf_to_float32(out_unfused)
    b = buf_to_float32(out_fused)
    return float(np.abs(a - b).max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seq-len",
        type=int,
        action="append",
        dest="seq_lens",
        default=None,
        metavar="S",
        help="Sequence lengths to benchmark (repeatable). "
             "Default: 512 4096 4608  (Flux2 text / image / single-stream)",
    )
    p.add_argument("--batch", type=int, default=1, metavar="B")
    p.add_argument("--hidden-dim", type=int, default=6144, metavar="D")
    p.add_argument(
        "--dtype",
        choices=["bf16", "f16", "f32"],
        default="bf16",
        help="Data type (default: bf16)",
    )
    p.add_argument("--warmup", type=int, default=5, metavar="N")
    p.add_argument("--iters", type=int, default=30, metavar="N")
    p.add_argument(
        "--no-correctness",
        action="store_true",
        help="Skip numerical correctness check",
    )
    p.add_argument(
        "--trace-dir",
        default=None,
        metavar="DIR",
        help="Directory to write Chrome trace JSON files (one per shape × variant). "
             "Skipped if not set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    seq_lens = args.seq_lens or [512, 4096, 4608]
    B = args.batch
    D = args.hidden_dim
    dtype_map = {"bf16": DType.bfloat16, "f16": DType.float16, "f32": DType.float32}
    dtype = dtype_map[args.dtype]
    dtype_str = args.dtype

    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    print(f"AdaLN Modulate Benchmark  — dtype={dtype_str}  B={B}  D={D}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup={args.warmup}  Iters={args.iters}")
    print()

    hdr = f"{'Shape':<22}  {'Unfused (ms)':>14}  {'Fused (ms)':>14}  {'Speedup':>8}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    for S in seq_lens:
        shape_str = f"[{B},{S},{D}]"

        # Build and compile both graphs
        g_unfused = build_unfused_graph(B, S, D, dtype, device_ref)
        g_fused = build_fused_graph(B, S, D, dtype, device_ref)
        m_unfused = session.load(g_unfused)
        m_fused = session.load(g_fused)

        # Inputs (same for both)
        inputs = make_inputs(B, S, D, dtype, device)

        # Correctness
        if not args.no_correctness:
            max_err = check_correctness(m_unfused, m_fused, inputs)
            # Tolerances: implementations differ in accumulation order, especially
            # for low-precision dtypes. bfloat16 has ~0.8% relative eps, so
            # outputs in [-10, 10] range can differ by up to ~0.5 in absolute terms.
            tol_map = {
                DType.float32: 1e-4,
                DType.float16: 5e-2,
                DType.bfloat16: 5e-1,
            }
            tol = tol_map.get(dtype, 0.5)
            status = "OK" if max_err < tol else f"FAIL(err={max_err:.3e}>tol={tol:.1e})"
            print(f"  Correctness {shape_str}: max_abs_err={max_err:.2e}  [{status}]")

        # Benchmark
        r_unfused = bench_model(m_unfused, inputs, args.warmup, args.iters)
        r_fused = bench_model(m_fused, inputs, args.warmup, args.iters)

        speedup = r_unfused.mean_ms / r_fused.mean_ms

        unfused_str = f"{r_unfused.mean_ms:.3f} ± {r_unfused.std_ms:.3f}"
        fused_str = f"{r_fused.mean_ms:.3f} ± {r_fused.std_ms:.3f}"
        speedup_str = f"{speedup:.2f}×"

        print(
            f"  {shape_str:<20}  {unfused_str:>14}  {fused_str:>14}  {speedup_str:>8}"
        )

        # Detail
        print(f"    unfused: min={r_unfused.min_ms:.3f}  max={r_unfused.max_ms:.3f} ms")
        print(f"    fused:   min={r_fused.min_ms:.3f}  max={r_fused.max_ms:.3f} ms")

        # Torch profiler traces
        if args.trace_dir:
            os.makedirs(args.trace_dir, exist_ok=True)
            tag = f"B{B}_S{S}_D{D}_{dtype_str}"
            profile_model(m_unfused, inputs, os.path.join(args.trace_dir, f"adaln_unfused_{tag}.json"))
            profile_model(m_fused,   inputs, os.path.join(args.trace_dir, f"adaln_fused_{tag}.json"))

        print()

    print(sep)
    print("Speedup > 1 means fused is faster.")


if __name__ == "__main__":
    main()
