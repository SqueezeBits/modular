#!/usr/bin/env python3
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

"""Compare intermediate tensors between diffusers and MAX for parity testing.

Reads .npy files dumped by:
- ``wan_animate_move_diffusers.py --dump-intermediates``
- MAX pipeline with ``--shared-inputs`` (which saves MAX intermediates)

Reports per-stage cosine similarity, MSE, and max absolute difference.

Usage:
    python wan_animate_compare_against_diffusers.py \\
        --diffusers-dump outputs/v1_redux/diffusers_dump \\
        --max-dump outputs/v1_redux/max_dump

    # Compare a single pair of tensors:
    python wan_animate_compare_against_diffusers.py \\
        --diffusers-dump outputs/v1_redux/diffusers_dump \\
        --max-dump outputs/v1_redux/max_dump \\
        --only final_latents_seg0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened tensors."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def compare_tensors(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 0.99,
) -> dict:
    """Compare two tensors and return metrics."""
    if a.shape != b.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
        }

    cos = cosine_similarity(a, b)
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff**2))
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))

    passed = cos >= threshold
    status = "PASS" if passed else "FAIL"

    return {
        "name": name,
        "status": status,
        "cos_sim": cos,
        "mse": mse,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "shape": list(a.shape),
        "threshold": threshold,
        "a_mean": float(np.mean(a)),
        "b_mean": float(np.mean(b)),
        "a_std": float(np.std(a)),
        "b_std": float(np.std(b)),
    }


# Tolerance targets per stage.
THRESHOLDS = {
    "prompt_embeds": 0.9999,
    "clip_features": 0.99,
    "ref_image_latents": 0.999,
    "noise_seg": 1.0,  # Must be exact (loaded from same file)
    "pose_latents_seg": 0.999,
    "motion_vectors_seg": 0.999,
    "face_emb_seg": 0.999,
    "prev_cond_seg": 0.999,
    "latents_after_step0_seg": 0.99,
    "final_latents_seg": 0.995,
}


def get_threshold(name: str) -> float:
    """Look up threshold for a tensor name, with prefix matching."""
    for prefix, thresh in THRESHOLDS.items():
        if name.startswith(prefix):
            return thresh
    return 0.99  # default


def find_matching_pairs(
    diffusers_dir: Path, max_dir: Path
) -> list[tuple[str, Path, Path]]:
    """Find .npy files present in both directories."""
    diffusers_files = {f.stem: f for f in diffusers_dir.glob("*.npy")}
    max_files = {f.stem: f for f in max_dir.glob("*.npy")}

    common = sorted(set(diffusers_files) & set(max_files))
    return [
        (name, diffusers_files[name], max_files[name]) for name in common
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare diffusers vs MAX intermediate tensors."
    )
    parser.add_argument(
        "--diffusers-dump",
        required=True,
        help="Directory with diffusers .npy dumps.",
    )
    parser.add_argument(
        "--max-dump",
        required=True,
        help="Directory with MAX .npy dumps.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Compare only this tensor name (stem without .npy).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-tensor statistics.",
    )
    args = parser.parse_args()

    diffusers_dir = Path(args.diffusers_dump)
    max_dir = Path(args.max_dump)

    if not diffusers_dir.exists():
        print(f"Error: diffusers dump dir not found: {diffusers_dir}")
        sys.exit(1)
    if not max_dir.exists():
        print(f"Error: MAX dump dir not found: {max_dir}")
        sys.exit(1)

    pairs = find_matching_pairs(diffusers_dir, max_dir)
    if args.only:
        pairs = [(n, d, m) for n, d, m in pairs if n == args.only]

    if not pairs:
        d_files = sorted(f.stem for f in diffusers_dir.glob("*.npy"))
        m_files = sorted(f.stem for f in max_dir.glob("*.npy"))
        print("No matching .npy files found.")
        print(f"  Diffusers: {d_files}")
        print(f"  MAX:       {m_files}")
        sys.exit(1)

    # List files unique to each dir.
    d_all = {f.stem for f in diffusers_dir.glob("*.npy")}
    m_all = {f.stem for f in max_dir.glob("*.npy")}
    d_only = sorted(d_all - m_all)
    m_only = sorted(m_all - d_all)

    results = []
    for name, d_path, m_path in pairs:
        d_arr = np.load(str(d_path))
        m_arr = np.load(str(m_path))
        threshold = get_threshold(name)
        result = compare_tensors(name, d_arr, m_arr, threshold)
        results.append(result)

    # Print results table.
    print(f"\n{'=' * 90}")
    print("  Diffusers vs MAX Parity Comparison")
    print(f"  Diffusers: {diffusers_dir}")
    print(f"  MAX:       {max_dir}")
    print(f"{'=' * 90}\n")

    print(
        f"{'Tensor':<35} {'Status':<6} {'Cos Sim':>10} "
        f"{'Threshold':>10} {'MaxAbsDiff':>12} {'Shape'}"
    )
    print("-" * 100)

    all_passed = True
    for r in results:
        if r["status"] == "SHAPE_MISMATCH":
            print(
                f"{r['name']:<35} {'SHAPE':<6} {'N/A':>10} "
                f"{'N/A':>10} {'N/A':>12} "
                f"d={r['a_shape']} m={r['b_shape']}"
            )
            all_passed = False
        else:
            print(
                f"{r['name']:<35} {r['status']:<6} "
                f"{r['cos_sim']:>10.6f} "
                f"{r['threshold']:>10.4f} "
                f"{r['max_abs_diff']:>12.6f} "
                f"{r['shape']}"
            )
            if r["status"] == "FAIL":
                all_passed = False

    if args.verbose:
        print(f"\n{'=' * 90}")
        print("Detailed Statistics:")
        print(f"{'=' * 90}")
        for r in results:
            if r["status"] == "SHAPE_MISMATCH":
                continue
            print(f"\n  {r['name']}:")
            print(f"    cos_sim:       {r['cos_sim']:.8f}")
            print(f"    mse:           {r['mse']:.8e}")
            print(f"    max_abs_diff:  {r['max_abs_diff']:.8e}")
            print(f"    mean_abs_diff: {r['mean_abs_diff']:.8e}")
            print(
                f"    diffusers: mean={r['a_mean']:.6f} "
                f"std={r['a_std']:.6f}"
            )
            print(
                f"    MAX:       mean={r['b_mean']:.6f} "
                f"std={r['b_std']:.6f}"
            )

    if d_only:
        print(f"\nDiffusers-only (not in MAX): {d_only}")
    if m_only:
        print(f"\nMAX-only (not in diffusers): {m_only}")

    print(f"\n{'=' * 90}")
    if all_passed:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED - see FAIL entries above")
        for r in results:
            if r.get("status") in ("FAIL", "SHAPE_MISMATCH"):
                print(f"  First failure: {r['name']}")
                break
    print(f"{'=' * 90}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)
    raise SystemExit(main())
