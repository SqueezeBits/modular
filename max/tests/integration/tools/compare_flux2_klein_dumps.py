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

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


_STEP_RE = re.compile(
    r"^(?P<prefix>diffusers|max)_(?P<kind>prompt_embeds|latent_model_input|noise_pred)(?:_(?P<step>\d+))?\.pt$"
)


@dataclass(frozen=True)
class PairStats:
    name: str
    stage_rank: int
    step: int
    shape_equal: bool
    dtype_a: str
    dtype_b: str
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    exact_equal: bool
    allclose: bool
    max_abs_diff: float | None
    mean_abs_diff: float | None
    rms_diff: float | None
    cosine: float | None
    first_diff_index: tuple[int, ...] | None
    first_diff_values: tuple[float, float] | None
    zero_seq_rows_a: int | None
    zero_seq_rows_b: int | None
    active_span_a: tuple[int, int] | None
    active_span_b: tuple[int, int] | None
    best_seq_shift: int | None
    best_shift_cosine: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Flux.2 Klein intermediate dump tensors saved as "
            "diffusers_*.pt and max_*.pt."
        )
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the dump files. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance used for allclose checks.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance used for allclose checks.",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=1e-6,
        help="Row norm threshold used to detect zero-padded sequence rows.",
    )
    return parser.parse_args()


def _stage_order(kind: str, step: int) -> tuple[int, int]:
    if kind == "prompt_embeds":
        return (0, -1)
    if kind == "latent_model_input":
        return (1, step)
    if kind == "noise_pred":
        return (2, step)
    return (99, step)


def _pair_key(path: Path) -> tuple[str, int]:
    match = _STEP_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected dump filename: {path.name}")
    kind = match.group("kind")
    step = int(match.group("step") or -1)
    return kind, step


def _flatten_seq_rows(tensor: torch.Tensor) -> torch.Tensor | None:
    if tensor.ndim < 2:
        return None
    if tensor.ndim == 2:
        return tensor.float()
    return tensor.float().reshape(tensor.shape[0] * tensor.shape[1], -1)


def _seq_row_metadata(
    tensor: torch.Tensor, zero_threshold: float
) -> tuple[int, tuple[int, int] | None]:
    rows = _flatten_seq_rows(tensor)
    if rows is None:
        return (0, None)
    row_norms = rows.norm(dim=-1)
    nonzero = (row_norms >= zero_threshold).nonzero().flatten()
    zero_rows = int((row_norms < zero_threshold).sum().item())
    if nonzero.numel() == 0:
        return (zero_rows, None)
    return (zero_rows, (int(nonzero[0]), int(nonzero[-1])))


def _best_seq_shift(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float] | None:
    if a.ndim != b.ndim or a.shape != b.shape or a.ndim < 2:
        return None
    seq_len = a.shape[-2]
    if seq_len > 1024:
        return None

    a_rows = a.float().reshape(-1, seq_len, a.shape[-1])
    b_rows = b.float().reshape(-1, seq_len, b.shape[-1])
    best_shift = 0
    best_cosine = -math.inf

    for shift in range(-seq_len + 1, seq_len):
        if shift >= 0:
            a_slice = a_rows[:, shift:, :]
            b_slice = b_rows[:, : seq_len - shift, :]
        else:
            a_slice = a_rows[:, : seq_len + shift, :]
            b_slice = b_rows[:, -shift:, :]

        if a_slice.numel() == 0:
            continue

        cosine = F.cosine_similarity(
            a_slice.reshape(1, -1), b_slice.reshape(1, -1)
        ).item()
        if cosine > best_cosine:
            best_shift = shift
            best_cosine = cosine

    return (best_shift, best_cosine)


def _compare_pair(
    diffusers_path: Path,
    max_path: Path,
    *,
    atol: float,
    rtol: float,
    zero_threshold: float,
) -> PairStats:
    match = _STEP_RE.match(diffusers_path.name)
    assert match is not None
    kind = match.group("kind")
    step = int(match.group("step") or -1)
    stage_rank, ordered_step = _stage_order(kind, step)

    a = torch.load(diffusers_path, map_location="cpu")
    b = torch.load(max_path, map_location="cpu")
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError(
            f"Expected tensors in {diffusers_path.name} and {max_path.name}"
        )

    shape_equal = tuple(a.shape) == tuple(b.shape)
    exact_equal = shape_equal and torch.equal(a, b)
    allclose = shape_equal and torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)

    max_abs_diff = None
    mean_abs_diff = None
    rms_diff = None
    cosine = None
    first_diff_index = None
    first_diff_values = None

    if shape_equal:
        a_float = a.float()
        b_float = b.float()
        diff = (a_float - b_float).abs()
        max_abs_diff = float(diff.max().item())
        mean_abs_diff = float(diff.mean().item())
        rms_diff = float((a_float - b_float).pow(2).mean().sqrt().item())
        cosine = float(
            F.cosine_similarity(a_float.reshape(1, -1), b_float.reshape(1, -1)).item()
        )

        nonzero = (diff > 0).nonzero()
        if nonzero.numel() > 0:
            idx = tuple(int(i) for i in nonzero[0].tolist())
            first_diff_index = idx
            first_diff_values = (float(a_float[idx].item()), float(b_float[idx].item()))

    zero_seq_rows_a, active_span_a = _seq_row_metadata(a, zero_threshold)
    zero_seq_rows_b, active_span_b = _seq_row_metadata(b, zero_threshold)
    shift_info = _best_seq_shift(a, b)
    best_seq_shift = shift_info[0] if shift_info is not None else None
    best_shift_cosine = shift_info[1] if shift_info is not None else None
    breakpoint()
    return PairStats(
        name=kind if step < 0 else f"{kind}_{step}",
        stage_rank=stage_rank,
        step=ordered_step,
        shape_equal=shape_equal,
        dtype_a=str(a.dtype),
        dtype_b=str(b.dtype),
        shape_a=tuple(int(x) for x in a.shape),
        shape_b=tuple(int(x) for x in b.shape),
        exact_equal=exact_equal,
        allclose=allclose,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        rms_diff=rms_diff,
        cosine=cosine,
        first_diff_index=first_diff_index,
        first_diff_values=first_diff_values,
        zero_seq_rows_a=zero_seq_rows_a,
        zero_seq_rows_b=zero_seq_rows_b,
        active_span_a=active_span_a,
        active_span_b=active_span_b,
        best_seq_shift=best_seq_shift,
        best_shift_cosine=best_shift_cosine,
    )


def _collect_pairs(root: Path) -> list[tuple[Path, Path]]:
    diffusers_files = sorted(root.glob("diffusers_*.pt"))
    pairs: list[tuple[Path, Path]] = []
    for diffusers_path in diffusers_files:
        if _STEP_RE.match(diffusers_path.name) is None:
            continue
        max_path = root / diffusers_path.name.replace("diffusers_", "max_", 1)
        if max_path.exists():
            pairs.append((diffusers_path, max_path))
    return sorted(pairs, key=lambda p: _stage_order(*_pair_key(p[0])))


def _print_stats(stats: PairStats) -> None:
    print(f"{stats.name}:")
    print(
        f"  shape {stats.shape_a} vs {stats.shape_b}, dtype {stats.dtype_a} vs {stats.dtype_b}"
    )
    print(
        f"  exact_equal={stats.exact_equal} allclose={stats.allclose}"
    )
    if stats.max_abs_diff is not None:
        print(
            "  diff "
            f"max_abs={stats.max_abs_diff:.6g} "
            f"mean_abs={stats.mean_abs_diff:.6g} "
            f"rms={stats.rms_diff:.6g} "
            f"cos={stats.cosine:.9f}"
        )
    if stats.first_diff_index is not None and stats.first_diff_values is not None:
        a_val, b_val = stats.first_diff_values
        print(
            f"  first_diff idx={stats.first_diff_index} "
            f"diffusers={a_val:.6g} max={b_val:.6g}"
        )
    if stats.active_span_a is not None or stats.active_span_b is not None:
        print(
            "  seq_rows "
            f"diffusers_zero={stats.zero_seq_rows_a} active_span={stats.active_span_a} "
            f"max_zero={stats.zero_seq_rows_b} active_span={stats.active_span_b}"
        )
    if stats.best_seq_shift is not None and stats.best_shift_cosine is not None:
        print(
            f"  best_shift shift={stats.best_seq_shift} "
            f"cos={stats.best_shift_cosine:.9f}"
        )


def main() -> None:
    args = _parse_args()
    pairs = _collect_pairs(args.dir)
    if not pairs:
        raise SystemExit(
            f"No matching diffusers_*.pt / max_*.pt pairs found under {args.dir}"
        )

    stats_list = [
        _compare_pair(
            diffusers_path,
            max_path,
            atol=args.atol,
            rtol=args.rtol,
            zero_threshold=args.zero_threshold,
        )
        for diffusers_path, max_path in pairs
    ]

    print(f"Compared {len(stats_list)} tensor pairs under {args.dir}")
    print()
    for stats in stats_list:
        _print_stats(stats)
        print()

    first_divergent = next((s for s in stats_list if not s.allclose), None)
    if first_divergent is not None:
        print(
            "First divergent stage: "
            f"{first_divergent.name} "
            f"(max_abs={first_divergent.max_abs_diff:.6g}, "
            f"cos={first_divergent.cosine:.9f})"
        )
        if (
            first_divergent.best_seq_shift is not None
            and first_divergent.best_shift_cosine is not None
            and abs(first_divergent.best_seq_shift) > 0
        ):
            print(
                "Likely sequence alignment issue: "
                f"best_shift={first_divergent.best_seq_shift} "
                f"still gives cosine={first_divergent.best_shift_cosine:.9f}"
            )


if __name__ == "__main__":
    main()
