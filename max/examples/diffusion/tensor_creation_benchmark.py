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
import gc
import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Any

import torch
from max import driver, engine
from max.driver import Accelerator, Buffer, CPU, Device, accelerator_count
from max.dtype import DType
from max.experimental import realization_context as rc
from max.experimental.tensor import Tensor


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    op: str
    shape: tuple[int, ...]
    dtype: DType
    value: int | float | bool = 0

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def bytes(self) -> int:
        return self.numel * _DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class BenchmarkResult:
    first_ms: float
    min_ms: float
    median_ms: float
    avg_ms: float
    p95_ms: float
    max_ms: float
    realized_after_create_fraction: float | None = None


_DTYPE_BYTES = {
    DType.bool: 1,
    DType.int8: 1,
    DType.uint8: 1,
    DType.float8_e8m0fnu: 1,
    DType.float8_e4m3fn: 1,
    DType.float8_e4m3fnuz: 1,
    DType.float8_e5m2: 1,
    DType.float8_e5m2fnuz: 1,
    DType.float16: 2,
    DType.bfloat16: 2,
    DType.int16: 2,
    DType.uint16: 2,
    DType.float32: 4,
    DType.int32: 4,
    DType.uint32: 4,
    DType.float64: 8,
    DType.int64: 8,
    DType.uint64: 8,
}

_CASES = {
    "prev_residual": BenchmarkCase(
        name="prev_residual",
        op="zeros",
        shape=(1, 4096, 6144),
        dtype=DType.bfloat16,
    ),
    "prev_output": BenchmarkCase(
        name="prev_output",
        op="zeros",
        shape=(1, 4096, 128),
        dtype=DType.bfloat16,
    ),
    "prompt_embeds_like": BenchmarkCase(
        name="prompt_embeds_like",
        op="zeros",
        shape=(1, 512, 15360),
        dtype=DType.bfloat16,
    ),
    "step_cache_flag": BenchmarkCase(
        name="step_cache_flag",
        op="full",
        shape=(1,),
        dtype=DType.bool,
        value=True,
    ),
    "rdt_tensor": BenchmarkCase(
        name="rdt_tensor",
        op="full",
        shape=(1,),
        dtype=DType.float32,
        value=0.08,
    ),
}

_DTYPE_TO_TORCH = {
    DType.bool: torch.bool,
    DType.int8: torch.int8,
    DType.int16: torch.int16,
    DType.int32: torch.int32,
    DType.int64: torch.int64,
    DType.uint8: torch.uint8,
    DType.float16: torch.float16,
    DType.bfloat16: torch.bfloat16,
    DType.float32: torch.float32,
    DType.float64: torch.float64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FLUX.2 step-cache tensor creation against lower-level "
            "MAX Buffer allocation and Torch."
        )
    )
    parser.add_argument(
        "--device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Device to benchmark on. Defaults to gpu.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Accelerator id to use when --device=gpu.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Measured iterations per case and backend.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=5,
        help="Warmup iterations per case and backend before measurement.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(_CASES),
        help=(
            "Case to run. Repeat to select multiple cases. Defaults to "
            "prev_residual, prev_output, step_cache_flag, rdt_tensor."
        ),
    )
    parser.add_argument(
        "--skip-buffer",
        action="store_true",
        help="Skip the MAX Buffer.zeros allocator baseline.",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip the Torch reference benchmark.",
    )
    parser.add_argument(
        "--session-num-threads",
        type=int,
        help=(
            "Override the MAX eager InferenceSession thread count. If unset, "
            "the runtime default is used."
        ),
    )
    return parser.parse_args()


def _resolve_device(args: argparse.Namespace) -> Device:
    if args.device == "cpu":
        return CPU()
    if accelerator_count() == 0:
        raise RuntimeError("No accelerator available for --device=gpu")
    return Accelerator(args.device_id)


def _resolve_torch_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable for --device=gpu")
    return torch.device(f"cuda:{args.device_id}")


def _configure_eager_session(num_threads: int | None) -> engine.InferenceSession | None:
    if num_threads is None:
        return None
    if num_threads <= 0:
        raise ValueError("--session-num-threads must be positive")

    device_specs = driver.scan_available_devices()
    if (cpu := driver.DeviceSpec.cpu()) not in device_specs:
        device_specs.append(cpu)
    devices = driver.load_devices(device_specs)
    session = engine.InferenceSession(devices=devices, num_threads=num_threads)
    rc._SESSION.set(session)
    return session


def _make_tensor(case: BenchmarkCase, device: Device) -> Tensor:
    if case.op == "zeros":
        return Tensor.zeros(case.shape, dtype=case.dtype, device=device)
    if case.op == "full":
        return Tensor.full(
            case.shape,
            case.value,
            dtype=case.dtype,
            device=device,
        )
    raise ValueError(f"Unsupported tensor op: {case.op}")


def _make_buffer(case: BenchmarkCase, device: Device) -> Buffer:
    return Buffer.zeros(case.shape, dtype=case.dtype, device=device)


def _make_torch_tensor(case: BenchmarkCase, device: torch.device) -> torch.Tensor:
    torch_dtype = _DTYPE_TO_TORCH.get(case.dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported torch dtype mapping for {case.dtype}")
    if case.op == "zeros":
        return torch.zeros(case.shape, dtype=torch_dtype, device=device)
    if case.op == "full":
        return torch.full(
            case.shape,
            fill_value=case.value,
            dtype=torch_dtype,
            device=device,
        )
    raise ValueError(f"Unsupported torch op: {case.op}")


def _sync(device: Device, torch_device: torch.device) -> None:
    device.synchronize()
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def _collect_samples(
    create: Callable[[], Any],
    cleanup: Callable[[Any], None],
    device: Device,
    torch_device: torch.device,
    *,
    iterations: int,
    warmups: int,
) -> list[float]:
    for _ in range(warmups):
        value = create()
        _sync(device, torch_device)
        cleanup(value)
        _sync(device, torch_device)

    samples_ms: list[float] = []
    for _ in range(iterations):
        _sync(device, torch_device)
        t0 = perf_counter()
        value = create()
        _sync(device, torch_device)
        samples_ms.append((perf_counter() - t0) * 1000.0)
        cleanup(value)
        _sync(device, torch_device)
    return samples_ms


def _summarize_samples(
    samples_ms: list[float],
    *,
    realized_after_create_fraction: float | None = None,
) -> BenchmarkResult:
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return BenchmarkResult(
        first_ms=samples_ms[0],
        min_ms=ordered[0],
        median_ms=median(samples_ms),
        avg_ms=sum(samples_ms) / len(samples_ms),
        p95_ms=ordered[p95_index],
        max_ms=ordered[-1],
        realized_after_create_fraction=realized_after_create_fraction,
    )


def _cleanup_max(value: Any) -> None:
    del value
    gc.collect()


def _cleanup_torch(value: torch.Tensor) -> None:
    del value
    gc.collect()


def _measure_tensor_case(
    case: BenchmarkCase,
    device: Device,
    torch_device: torch.device,
    *,
    iterations: int,
    warmups: int,
) -> BenchmarkResult:
    for _ in range(warmups):
        tensor = _make_tensor(case, device)
        _sync(device, torch_device)
        _cleanup_max(tensor)
        _sync(device, torch_device)

    realized_after_create_count = 0
    samples_ms: list[float] = []
    for _ in range(iterations):
        _sync(device, torch_device)
        t0 = perf_counter()
        tensor = _make_tensor(case, device)
        if tensor.real:
            realized_after_create_count += 1
        _sync(device, torch_device)
        samples_ms.append((perf_counter() - t0) * 1000.0)
        _cleanup_max(tensor)
        _sync(device, torch_device)

    return _summarize_samples(
        samples_ms,
        realized_after_create_fraction=realized_after_create_count / iterations,
    )


def _measure_buffer_case(
    case: BenchmarkCase,
    device: Device,
    torch_device: torch.device,
    *,
    iterations: int,
    warmups: int,
) -> BenchmarkResult:
    samples_ms = _collect_samples(
        lambda: _make_buffer(case, device),
        _cleanup_max,
        device,
        torch_device,
        iterations=iterations,
        warmups=warmups,
    )
    return _summarize_samples(samples_ms)


def _measure_torch_case(
    case: BenchmarkCase,
    device: Device,
    torch_device: torch.device,
    *,
    iterations: int,
    warmups: int,
) -> BenchmarkResult:
    samples_ms = _collect_samples(
        lambda: _make_torch_tensor(case, torch_device),
        _cleanup_torch,
        device,
        torch_device,
        iterations=iterations,
        warmups=warmups,
    )
    return _summarize_samples(samples_ms)


def _measure_sync_only(
    device: Device,
    torch_device: torch.device,
    *,
    iterations: int,
    warmups: int,
) -> BenchmarkResult:
    for _ in range(warmups):
        _sync(device, torch_device)

    samples_ms: list[float] = []
    for _ in range(iterations):
        _sync(device, torch_device)
        t0 = perf_counter()
        _sync(device, torch_device)
        samples_ms.append((perf_counter() - t0) * 1000.0)

    return _summarize_samples(samples_ms)


def _format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def _print_result_table(
    title: str,
    results: list[tuple[BenchmarkCase, BenchmarkResult]],
) -> None:
    print(title)
    print(
        f"{'case':<18} {'shape':<18} {'dtype':<10} {'size':>10} "
        f"{'first':>10} {'avg':>10} {'median':>10} {'p95':>10} {'max':>10} "
        f"{'real@create':>12}"
    )
    for case, result in results:
        real_at_create = (
            f"{result.realized_after_create_fraction:.2f}"
            if result.realized_after_create_fraction is not None
            else "-"
        )
        print(
            f"{case.name:<18} "
            f"{str(case.shape):<18} "
            f"{case.dtype.name:<10} "
            f"{_format_mib(case.bytes):>10} "
            f"{result.first_ms:>10.3f} "
            f"{result.avg_ms:>10.3f} "
            f"{result.median_ms:>10.3f} "
            f"{result.p95_ms:>10.3f} "
            f"{result.max_ms:>10.3f} "
            f"{real_at_create:>12}"
        )
    print()


def _print_ratio_table(
    tensor_results: list[tuple[BenchmarkCase, BenchmarkResult]],
    buffer_results: list[tuple[BenchmarkCase, BenchmarkResult]] | None,
    torch_results: list[tuple[BenchmarkCase, BenchmarkResult]] | None,
) -> None:
    buffer_by_case = {case.name: result for case, result in buffer_results or []}
    torch_by_case = {case.name: result for case, result in torch_results or []}

    print("Average Latency Ratios")
    print(
        f"{'case':<18} {'tensor/buffer':>14} {'tensor/torch':>14} "
        f"{'buffer/torch':>14}"
    )
    for case, tensor_result in tensor_results:
        buffer_result = buffer_by_case.get(case.name)
        torch_result = torch_by_case.get(case.name)
        tensor_buffer = (
            f"{tensor_result.avg_ms / buffer_result.avg_ms:.1f}x"
            if buffer_result and buffer_result.avg_ms > 0
            else "-"
        )
        tensor_torch = (
            f"{tensor_result.avg_ms / torch_result.avg_ms:.1f}x"
            if torch_result and torch_result.avg_ms > 0
            else "-"
        )
        buffer_torch = (
            f"{buffer_result.avg_ms / torch_result.avg_ms:.1f}x"
            if buffer_result and torch_result and torch_result.avg_ms > 0
            else "-"
        )
        print(
            f"{case.name:<18} {tensor_buffer:>14} {tensor_torch:>14} "
            f"{buffer_torch:>14}"
        )
    print()


def main() -> int:
    args = parse_args()
    device = _resolve_device(args)
    torch_device = _resolve_torch_device(args)
    eager_session = _configure_eager_session(args.session_num_threads)

    selected_cases = (
        args.case
        if args.case
        else [
            "prev_residual",
            "prev_output",
            "step_cache_flag",
            "rdt_tensor",
        ]
    )
    cases = [_CASES[name] for name in selected_cases]

    print(
        f"Device: {device} (api={device.api}, arch="
        f"{device.architecture_name if not device.is_host else 'cpu'})"
    )
    print(f"Torch device: {torch_device}")
    print(
        "MAX eager session threads: "
        f"{eager_session.num_threads if eager_session else 'default'}"
    )
    print(f"Iterations: {args.iterations}, warmups: {args.warmups}")
    print()

    sync_only = _measure_sync_only(
        device,
        torch_device,
        iterations=args.iterations,
        warmups=args.warmups,
    )
    print("Sync Only Baseline")
    print(
        f"{'first':>10} {'avg':>10} {'median':>10} {'p95':>10} {'max':>10}"
    )
    print(
        f"{sync_only.first_ms:>10.3f} "
        f"{sync_only.avg_ms:>10.3f} "
        f"{sync_only.median_ms:>10.3f} "
        f"{sync_only.p95_ms:>10.3f} "
        f"{sync_only.max_ms:>10.3f}"
    )
    print()

    tensor_results = [
        (
            case,
            _measure_tensor_case(
                case,
                device,
                torch_device,
                iterations=args.iterations,
                warmups=args.warmups,
            ),
        )
        for case in cases
    ]
    _print_result_table(
        "MAX Tensor.zeros/full + Synchronize",
        tensor_results,
    )

    buffer_results: list[tuple[BenchmarkCase, BenchmarkResult]] | None = None
    if not args.skip_buffer:
        buffer_results = [
            (
                case,
                _measure_buffer_case(
                    case,
                    device,
                    torch_device,
                    iterations=args.iterations,
                    warmups=args.warmups,
                ),
            )
            for case in cases
        ]
        print(
            "Note: MAX Buffer baseline uses Buffer.zeros with the same shape "
            "and dtype for each case."
        )
        print()
        _print_result_table(
            "MAX Buffer.zeros + Synchronize",
            buffer_results,
        )

    torch_results: list[tuple[BenchmarkCase, BenchmarkResult]] | None = None
    if not args.skip_torch:
        torch_results = [
            (
                case,
                _measure_torch_case(
                    case,
                    device,
                    torch_device,
                    iterations=args.iterations,
                    warmups=args.warmups,
                ),
            )
            for case in cases
        ]
        _print_result_table(
            "Torch zeros/full + Synchronize",
            torch_results,
        )

    _print_ratio_table(tensor_results, buffer_results, torch_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
