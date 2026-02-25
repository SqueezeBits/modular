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

"""Analyze Chrome trace JSON files to compare CUDA kernel timings.

Extracts GPU kernel events, groups them by operation category, and
compares MAX vs diffusers across flux1, flux2-t2i, flux2-i2i.
"""

import json
from collections import defaultdict
from pathlib import Path


def load_trace(path: str) -> list[dict]:
    """Load trace events from a Chrome trace JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data


def extract_gpu_kernels(events: list[dict]) -> list[dict]:
    """Extract GPU kernel events (cat == 'kernel' or on GPU stream)."""
    kernels = []
    for ev in events:
        # Look for CUDA kernel events
        if ev.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(ev)
        elif ev.get("cat") == "cuda_runtime":
            continue
        # Also capture events on GPU (pid sometimes indicates GPU)
    return kernels


def classify_kernel(name: str) -> str:
    """Classify a kernel name into a high-level operation category."""
    n = name.lower()

    # Matmul / GEMM
    if any(
        k in n
        for k in [
            "gemm",
            "matmul",
            "mm_",
            "cublas",
            "cutlass",
            "triton_mm",
            "ampere_bf16",
            "sm90_xmma",
            "sm80_xmma",
            "_sgemm",
            "_hgemm",
            "batched_gemm",
        ]
    ):
        return "matmul/gemm"

    # Attention
    if any(
        k in n
        for k in [
            "attention",
            "flash_attn",
            "fmha",
            "sdpa",
            "flash_fwd",
            "fused_attention",
        ]
    ):
        return "attention"

    # Softmax
    if "softmax" in n:
        return "softmax"

    # LayerNorm / RMSNorm
    if any(k in n for k in ["layernorm", "layer_norm", "rmsnorm", "rms_norm"]):
        return "layernorm/rmsnorm"

    # Elementwise (activation, add, mul, etc.)
    if any(
        k in n
        for k in [
            "elementwise",
            "pointwise",
            "gelu",
            "silu",
            "relu",
            "sigmoid",
            "tanh",
            "add_kernel",
            "mul_kernel",
            "vectorized",
            "fused_add",
            "unrolled_elementwise",
        ]
    ):
        return "elementwise"

    # Reduction (sum, mean, etc.)
    if any(k in n for k in ["reduce", "reduction", "welford", "sum_kernel"]):
        return "reduction"

    # Memory operations
    if any(k in n for k in ["memcpy", "memset", "copy_", "contiguous"]):
        return "memory_ops"

    # Concatenation / split / reshape
    if any(
        k in n
        for k in [
            "cat_kernel",
            "concat",
            "split",
            "chunk",
            "index",
            "scatter",
            "gather",
        ]
    ):
        return "concat/index"

    # Embedding / position encoding
    if any(k in n for k in ["embedding", "rotary", "rope"]):
        return "embedding/rope"

    # Triton kernels (generic)
    if "triton_" in n or "triton__" in n:
        return "triton_fused"

    return "other"


def summarize_per_iteration(
    events: list[dict], num_iters: int
) -> dict[str, float]:
    """Aggregate kernel durations by category, averaged per iteration.

    Duration is in microseconds in the trace; we return milliseconds.
    """
    cat_dur: dict[str, float] = defaultdict(float)
    for ev in events:
        dur_us = ev.get("dur", 0)
        cat = classify_kernel(ev.get("name", ""))
        cat_dur[cat] += dur_us

    # Convert to ms and average per iteration
    return {
        cat: dur / 1000.0 / num_iters
        for cat, dur in sorted(cat_dur.items(), key=lambda x: -x[1])
    }


def print_comparison(name: str, diffusers_cats: dict, max_cats: dict) -> None:
    """Print side-by-side comparison for one input shape."""
    all_cats = sorted(
        set(diffusers_cats) | set(max_cats),
        key=lambda c: -(diffusers_cats.get(c, 0) + max_cats.get(c, 0)),
    )

    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"{'=' * 80}")
    print(
        f"{'category':<25} {'diffusers (ms)':>14} {'MAX (ms)':>14} {'diff (ms)':>12} {'ratio':>8}"
    )
    print("-" * 80)

    total_d, total_m = 0.0, 0.0
    for cat in all_cats:
        d = diffusers_cats.get(cat, 0.0)
        m = max_cats.get(cat, 0.0)
        diff = m - d
        ratio = m / d if d > 0.001 else float("inf")
        total_d += d
        total_m += m
        print(f"{cat:<25} {d:>14.3f} {m:>14.3f} {diff:>+12.3f} {ratio:>8.2f}x")

    diff_total = total_m - total_d
    ratio_total = total_m / total_d if total_d > 0.001 else float("inf")
    print("-" * 80)
    print(
        f"{'TOTAL':<25} {total_d:>14.3f} {total_m:>14.3f} {diff_total:>+12.3f} {ratio_total:>8.2f}x"
    )


def main():
    trace_dir = Path("/workspace/modular/traces")
    num_iters = 3  # number of profiled iterations

    configs = [
        (
            "flux1",
            "transformer_diffusers_flux1.json",
            "transformer_max_flux1.json",
        ),
        (
            "flux2-t2i",
            "transformer_diffusers_flux2-t2i.json",
            "transformer_max_flux2-t2i.json",
        ),
        (
            "flux2-i2i",
            "transformer_diffusers_flux2-i2i.json",
            "transformer_max_flux2-i2i.json",
        ),
    ]

    all_results = {}

    for config_name, diff_file, max_file in configs:
        diff_events = load_trace(str(trace_dir / diff_file))
        max_events = load_trace(str(trace_dir / max_file))

        diff_kernels = extract_gpu_kernels(diff_events)
        max_kernels = extract_gpu_kernels(max_events)

        diff_cats = summarize_per_iteration(diff_kernels, num_iters)
        max_cats = summarize_per_iteration(max_kernels, num_iters)

        all_results[config_name] = (diff_cats, max_cats)
        print_comparison(config_name, diff_cats, max_cats)

    # Print cross-config comparison (using flux2-t2i as reference)
    print(f"\n\n{'=' * 80}")
    print("  CROSS-CONFIG ANALYSIS (flux2-t2i as reference)")
    print(f"{'=' * 80}")

    ref_diff, ref_max = all_results["flux2-t2i"]
    for config_name in ["flux1", "flux2-i2i"]:
        d_cats, m_cats = all_results[config_name]
        print(f"\n--- {config_name} vs flux2-t2i ---")
        print(
            f"{'category':<25} {'d_ratio':>10} {'m_ratio':>10} {'d_diff(ms)':>12} {'m_diff(ms)':>12}"
        )
        print("-" * 75)
        all_cats = sorted(
            set(d_cats) | set(m_cats) | set(ref_diff) | set(ref_max),
            key=lambda c: -(
                abs(m_cats.get(c, 0) - ref_max.get(c, 0))
                + abs(d_cats.get(c, 0) - ref_diff.get(c, 0))
            ),
        )
        for cat in all_cats:
            d_val = d_cats.get(cat, 0)
            m_val = m_cats.get(cat, 0)
            rd = ref_diff.get(cat, 0)
            rm = ref_max.get(cat, 0)
            d_ratio = d_val / rd if rd > 0.001 else float("inf")
            m_ratio = m_val / rm if rm > 0.001 else float("inf")
            d_diff = d_val - rd
            m_diff = m_val - rm
            print(
                f"{cat:<25} {d_ratio:>10.2f}x {m_ratio:>10.2f}x {d_diff:>+12.3f} {m_diff:>+12.3f}"
            )

    # Also dump top individual kernels by duration for each trace
    print(f"\n\n{'=' * 80}")
    print("  TOP 15 INDIVIDUAL KERNEL TYPES BY DURATION (per iteration, ms)")
    print(f"{'=' * 80}")

    for config_name, diff_file, max_file in configs:
        diff_events = load_trace(str(trace_dir / diff_file))
        max_events = load_trace(str(trace_dir / max_file))

        diff_kernels = extract_gpu_kernels(diff_events)
        max_kernels = extract_gpu_kernels(max_events)

        for framework, kernels in [
            ("diffusers", diff_kernels),
            ("MAX", max_kernels),
        ]:
            # Group by kernel name
            kernel_dur: dict[str, float] = defaultdict(float)
            kernel_count: dict[str, int] = defaultdict(int)
            for ev in kernels:
                name = ev.get("name", "unknown")
                kernel_dur[name] += ev.get("dur", 0) / 1000.0 / num_iters
                kernel_count[name] += 1

            top = sorted(kernel_dur.items(), key=lambda x: -x[1])[:15]
            print(f"\n--- {framework} / {config_name} ---")
            print(f"{'kernel':<70} {'ms':>8} {'count':>6}")
            print("-" * 88)
            total = sum(kernel_dur.values())
            for name, dur in top:
                cnt = kernel_count[name] // num_iters
                print(f"{name[:70]:<70} {dur:>8.3f} {cnt:>6}")
            print(f"{'TOTAL':<70} {total:>8.3f}")


if __name__ == "__main__":
    main()
