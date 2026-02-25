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

"""Analyze Chrome trace files produced by profile_transformer.py.

Reports per-group GPU kernel latency and its share of the total, using groups
that are meaningful across both MAX and diffusers (torch.compile) traces.

Usage — single trace:
    python traces/analyze_detailed.py traces/transformer_max_flux1.json

Usage — comparison (diffusers vs MAX):
    python traces/analyze_detailed.py \\
        traces/transformer_diffusers_flux1.json \\
        traces/transformer_max_flux1.json \\
        --labels diffusers max

Groups
------
attention    : flash attention / MAX MHA kernels (SM90, SM100)
matmul       : MAX linalg_matmul/linalg_gemv kernels; cuBLAS nvjet / cutlass
normalization: RMSNorm / LayerNorm (MAX: separate; diffusers: triton-fused)
elementwise  : activation, scale, add, concat, RoPE (MAX: separate; diffusers: triton-fused)
else         : memory ops, unknown kernels

MAX kernel naming on SM90 (H100):
  attention : nn_mha_sm90__mha_sm90_*
  matmul    : linalg_matmul_gpu_sm90_*, linalg_gemv_gemv_split_k_*
  norm      : nn_normalization_rms_norm_gpu_*, nn_normalization_layer_norm_gp*
  elemwise  : std_algorithm_functional__ele*, nn_concat__*

Note: RoPE kernels are included in elementwise because MAX compiles them to
std_algorithm_functional__ele* CUDA kernels (no "rope"/"rotary" in name).
Use --top-elementwise to identify RoPE candidates by call count.

For diffusers traces, fused triton kernels are sub-classified by inspecting the
kernel name so that the group totals are comparable with MAX.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Ordered list of groups shown in every table.
GROUPS = ["attention", "matmul", "normalization", "elementwise", "else"]


# ---------------------------------------------------------------------------
# Kernel loading
# ---------------------------------------------------------------------------


def load_kernels(path: str) -> list[dict]:
    """Return only GPU kernel / memcpy / memset events from a Chrome trace."""
    with open(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    return [
        e
        for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]


# ---------------------------------------------------------------------------
# Kernel classification
# ---------------------------------------------------------------------------


def classify_kernel(name: str) -> str:
    """Map a CUDA kernel name to one of the six analysis groups."""
    n = name.lower()

    # --- Attention ---
    # MAX SM90: nn_mha_sm90__mha_sm90_*
    # MAX SM100: nn_mha_sm100_2q_SM100MHA2Q_ke*
    # diffusers: flash_fwd_kernel, flash_fwd_splitkv_*, fmha_*, sdpa_*
    if "nn_mha" in n or "sm100mha" in n:
        return "attention"
    if any(k in n for k in ["flash_fwd", "flash_attn", "fmha", "sdpa"]):
        return "attention"
    # triton attention (rare, but handle it)
    if "triton_" in n and any(
        k in n for k in ["attention", "flash", "dot_product", "scaled_dot"]
    ):
        return "attention"

    # --- Matmul / linear ---
    # MAX SM90: linalg_matmul_gpu_sm90_*, linalg_gemv_gemv_split_k_*
    # diffusers: nvjet_tst_* (cuBLAS), cutlass*
    if any(k in n for k in ["linalg_matmul", "linalg_gemv"]):
        return "matmul"
    if "nvjet_" in n or "cutlass" in n:
        return "matmul"
    if any(
        k in n
        for k in [
            "sm90_xmma",
            "sm80_xmma",
            "ampere_bf16",
            "ampere_h16",
            "volta_h16",
            "gemm",
            "_sgemm",
            "_hgemm",
            "batched_gemm",
        ]
    ):
        return "matmul"

    # --- Normalization ---
    # MAX: nn_normalization_rms_norm_gpu, nn_normalization_layer_norm_gp*
    if any(k in n for k in ["rms_norm", "layer_norm", "rmsnorm", "layernorm"]):
        return "normalization"
    # diffusers: triton kernels fusing norm + elementwise
    if "triton_" in n and any(
        k in n for k in ["norm", "rsqrt", "welford", "rms"]
    ):
        return "normalization"

    # --- Elementwise ---
    # MAX: std_algorithm_functional__ele*, nn_concat__fused_concat_inner, …
    if "std_algorithm_functional__ele" in n:
        return "elementwise"
    if any(
        k in n
        for k in [
            "nn_concat",
            "cat_kernel",
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
    # diffusers: triton kernels fusing activation / concat / scale
    if "triton_" in n:
        return "elementwise"

    # --- Everything else ---
    return "else"


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------


def analyze(
    path: str, num_iters: int
) -> tuple[dict[str, float], dict[str, tuple[float, int]]]:
    """Analyze a trace and return per-group totals and per-kernel stats.

    Returns:
        groups: {group_name: total_ms_per_iter}
        kernels: {kernel_name: (total_ms_per_iter, calls_per_iter)}
    """
    raw_kernels = load_kernels(path)
    buckets: dict[str, float] = defaultdict(float)
    kernel_dur: dict[str, float] = defaultdict(float)
    kernel_cnt: dict[str, int] = defaultdict(int)
    for k in raw_kernels:
        name = k.get("name", "")
        dur_ms = k.get("dur", 0) / 1000.0 / num_iters
        group = classify_kernel(name)
        buckets[group] += dur_ms
        kernel_dur[name] += dur_ms
        kernel_cnt[name] += 1
    groups = {g: buckets.get(g, 0.0) for g in GROUPS}
    kernels = {
        n: (kernel_dur[n], kernel_cnt[n] // num_iters)
        for n in kernel_dur
    }
    return groups, kernels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _infer_label(path: str) -> str:
    """Guess a short framework label from the filename."""
    stem = Path(path).stem.lower()
    for fw in ("diffusers", "max"):
        if fw in stem:
            return fw
    return Path(path).name


def print_single(
    path: str,
    data: dict[str, float],
    label: str,
    kernels: dict[str, tuple[float, int]],
    top_ele: int = 0,
) -> None:
    """Print a per-group breakdown for one trace.

    Args:
        top_ele: If > 0, also print a breakdown of the top-N elementwise
                 kernels with their call counts.  Useful for identifying
                 which kernels correspond to RoPE (they will have call counts
                 matching the architecture block count and scale linearly with
                 sequence length).
    """
    total = sum(data.values())
    width = 60
    print(f"\n{'=' * width}")
    print(f"  Trace : {Path(path).name}")
    print(f"  Label : {label}")
    print(f"  Total : {total:.2f} ms/iter")
    print(f"{'=' * width}")
    print(f"{'Group':<16} {'ms/iter':>10} {'% total':>9}")
    print(f"{'-' * 37}")
    for g in GROUPS:
        v = data[g]
        pct = v / total * 100 if total > 0 else 0.0
        print(f"{g:<16} {v:>10.2f} {pct:>8.1f}%")
    print(f"{'-' * 37}")
    print(f"{'TOTAL':<16} {total:>10.2f} {'100.0%':>9}")

    if top_ele > 0:
        # Show per-kernel breakdown for the elementwise group.
        # Call counts that match architecture block counts (e.g. 56 = 8+48 for
        # Flux2) identify specific operations — particularly useful for RoPE:
        #   pre-opt (apply_rotary_emb): scattered across many kernels
        #   post-opt (fused_qk_rope_vision): one kernel, 56 calls/iter (all blocks)
        # Both compile to std_algorithm_functional__ele* so the group name alone
        # cannot distinguish RoPE from other elementwise ops.
        ele_kernels = {
            n: (ms, cnt)
            for n, (ms, cnt) in kernels.items()
            if "std_algorithm_functional" in n or "nn_concat" in n
        }
        if not ele_kernels:
            return
        ele_total = sum(ms for ms, _ in ele_kernels.values())
        top = sorted(ele_kernels.items(), key=lambda x: -x[1][0])[:top_ele]
        shown_total = sum(ms for _, (ms, _) in top)
        print(f"\n  Top-{top_ele} elementwise kernels  "
              f"(group total: {ele_total:.2f} ms/iter)")
        print(f"  {'kernel (type + hash)':<32} {'ms/iter':>9} {'%ele':>6} "
              f"{'calls/iter':>11}  note")
        print(f"  {'-' * 80}")
        for name, (ms, cnt) in top:
            # Build a readable short form: show the semantic prefix + hash
            if "nn_concat" in name:
                ktype = "concat"
            elif "std_algorithm_functional" in name:
                ktype = "elementwise"
            else:
                ktype = name.split("_")[0][:12]
            short = f"{ktype}:{name[-12:]}"
            pct = ms / ele_total * 100 if ele_total > 0 else 0.0
            # Flag counts that match common Flux2 architecture numbers
            note = ""
            if cnt == 56:
                note = "<-- fused QK RoPE (all 8+48 blocks)"
            elif cnt == 48:
                note = "<-- 48 single blocks (RoPE candidate)"
            elif cnt == 16:
                note = "<-- 8 double blocks ×2 (RoPE candidate)"
            elif cnt == 8:
                note = "<-- 8 double blocks"
            print(f"  {short:<32} {ms:>9.3f} {pct:>5.1f}% "
                  f"{cnt:>11}  {note}")
        if len(top) < len(ele_kernels):
            remaining_ms = ele_total - shown_total
            print(f"  {'...(remaining kernels)':<32} {remaining_ms:>9.3f} "
                  f"{remaining_ms/ele_total*100:>5.1f}%")


def print_comparison(
    path_a: str,
    data_a: dict[str, float],
    label_a: str,
    path_b: str,
    data_b: dict[str, float],
    label_b: str,
) -> None:
    """Print a side-by-side comparison of two traces with a gain/loss summary."""
    total_a = sum(data_a.values())
    total_b = sum(data_b.values())
    delta_total = total_b - total_a

    # --- Per-group table ---
    col_a = f"{label_a}(ms)"
    col_b = f"{label_b}(ms)"
    width = 72
    print(f"\n{'=' * width}")
    print(f"  Comparison")
    print(f"    A [{label_a}]: {Path(path_a).name}")
    print(f"    B [{label_b}]: {Path(path_b).name}")
    print(f"{'=' * width}")

    hdr = (
        f"{'Group':<16} {col_a:>12} {'%A':>6} {col_b:>12} {'%B':>6} "
        f"{'delta(ms)':>11}"
    )
    print(hdr)
    print(f"{'-' * len(hdr)}")
    for g in GROUPS:
        va = data_a[g]
        vb = data_b[g]
        pct_a = va / total_a * 100 if total_a > 0 else 0.0
        pct_b = vb / total_b * 100 if total_b > 0 else 0.0
        delta = vb - va
        print(
            f"{g:<16} {va:>12.2f} {pct_a:>5.1f}% {vb:>12.2f} {pct_b:>5.1f}% "
            f"{delta:>+11.2f}"
        )
    print(f"{'-' * len(hdr)}")
    print(
        f"{'TOTAL':<16} {total_a:>12.2f} {'100.0%':>6} {total_b:>12.2f} "
        f"{'100.0%':>6} {delta_total:>+11.2f}"
    )

    # --- Gain / loss summary ---
    attn_delta = data_b["attention"] - data_a["attention"]
    norm_delta = data_b["normalization"] - data_a["normalization"]
    elem_delta = data_b["elementwise"] - data_a["elementwise"]
    unfused_delta = norm_delta + elem_delta  # positive = B has more overhead

    net_direction = f"{label_b} {'faster' if delta_total < 0 else 'slower'}"

    print(f"\n{'─' * 60}")
    print(f"  Gain / loss summary  ({label_a} → {label_b})")
    print(f"{'─' * 60}")
    print(
        f"  Attention gain        : {-attn_delta:>+8.2f} ms"
        f"  ({data_a['attention']:.2f} → {data_b['attention']:.2f})"
    )
    print(
        f"  Unfused overhead      : {unfused_delta:>+8.2f} ms"
        f"  ({data_a['normalization'] + data_a['elementwise']:.2f}"
        f" → {data_b['normalization'] + data_b['elementwise']:.2f})"
    )
    print(
        f"    normalization delta : {norm_delta:>+8.2f} ms"
        f"  ({data_a['normalization']:.2f} → {data_b['normalization']:.2f})"
    )
    print(
        f"    elementwise delta   : {elem_delta:>+8.2f} ms"
        f"  ({data_a['elementwise']:.2f} → {data_b['elementwise']:.2f})"
    )
    print(
        f"  Net (B - A)           : {delta_total:>+8.2f} ms"
        f"  ({net_direction} by {abs(delta_total):.2f} ms)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one or two Chrome trace files from profile_transformer.py. "
            "Reports per-group GPU kernel latency and share of total. "
            "When two traces are given, prints a comparison with gain/loss summary."
        ),
    )
    parser.add_argument(
        "traces",
        nargs="+",
        metavar="TRACE",
        help="Path(s) to Chrome trace JSON file(s). Provide one or two.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        help=(
            "Short label(s) for each trace (e.g. --labels diffusers max). "
            "Inferred from the filename if omitted."
        ),
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=3,
        help=(
            "Number of profiled iterations recorded in the trace "
            "(used for per-iteration averaging, default: 3)."
        ),
    )
    parser.add_argument(
        "--top-elementwise",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Print a breakdown of the top-N individual elementwise kernels "
            "with their call counts per iteration.  Useful for identifying "
            "which kernel corresponds to RoPE (call count matches block count, "
            "e.g. 56 = 8+48 for Flux2 post-optimization).  Default: 0 (off)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if len(args.traces) > 2:
        raise SystemExit("Error: at most two trace files can be specified.")

    # Resolve labels
    labels: list[str] = list(args.labels or [])
    while len(labels) < len(args.traces):
        labels.append(_infer_label(args.traces[len(labels)]))

    results = [analyze(p, args.num_iters) for p in args.traces]
    groups_list = [r[0] for r in results]
    kernels_list = [r[1] for r in results]

    # Per-trace breakdown
    for path, groups, kernels, label in zip(
        args.traces, groups_list, kernels_list, labels
    ):
        print_single(path, groups, label, kernels, top_ele=args.top_elementwise)

    # Comparison (when two traces are provided)
    if len(args.traces) == 2:
        print_comparison(
            args.traces[0], groups_list[0], labels[0],
            args.traces[1], groups_list[1], labels[1],
        )


if __name__ == "__main__":
    main()
