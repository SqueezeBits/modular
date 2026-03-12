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

"""Run multiple diffusion requests with different shapes in one process.

This runner keeps a single tokenizer + pipeline alive and executes multiple
requests sequentially, which is useful for checking same-process shape/step
behavior without paying per-process startup costs.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from max.engine import InferenceSession
from max.examples.diffusion.libs.runtime_libs import (
    preload_bundled_nvidia_runtime_libraries,
)
from max.examples.diffusion.offline_generation_utils import (
    build_context_and_inputs,
    build_generation_request,
    build_pipeline_and_tokenizer,
    load_input_image_data_uris,
    postprocess_output,
    save_generation_output,
)


@dataclass(frozen=True)
class GenerationCase:
    """One same-process generation case."""

    width: int
    height: int
    num_inference_steps: int


@dataclass(frozen=True)
class GenerationShape:
    """One generated shape used to build a case matrix."""

    width: int
    height: int


@dataclass
class SessionLoadCounter:
    """Count InferenceSession.load calls as a compile/recompile proxy."""

    total_loads: int = 0

    def mark(self) -> int:
        """Return the current counter value."""
        return self.total_loads

    def delta_since(self, previous_mark: int) -> int:
        """Return how many new load calls happened since a mark."""
        return self.total_loads - previous_mark


def _parse_case(case_spec: str) -> GenerationCase:
    """Parse a CLI case in WIDTHxHEIGHT:STEPS format."""
    try:
        size_spec, steps_spec = case_spec.split(":", maxsplit=1)
        width_spec, height_spec = size_spec.lower().split("x", maxsplit=1)
        case = GenerationCase(
            width=int(width_spec),
            height=int(height_spec),
            num_inference_steps=int(steps_spec),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Cases must use WIDTHxHEIGHT:STEPS, for example 768x1024:2."
        ) from exc

    if case.width <= 0 or case.height <= 0:
        raise argparse.ArgumentTypeError(
            "Case width and height must be positive integers."
        )
    if case.num_inference_steps <= 0:
        raise argparse.ArgumentTypeError(
            "Case step counts must be positive integers."
        )
    return case


def _parse_shape(shape_spec: str) -> GenerationShape:
    """Parse WIDTHxHEIGHT shape syntax."""
    try:
        width_spec, height_spec = shape_spec.lower().split("x", maxsplit=1)
        shape = GenerationShape(width=int(width_spec), height=int(height_spec))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Shapes must use WIDTHxHEIGHT, for example 768x1024."
        ) from exc

    if shape.width <= 0 or shape.height <= 0:
        raise argparse.ArgumentTypeError(
            "Shape width and height must be positive integers."
        )
    return shape


@contextmanager
def count_session_loads() -> Iterator[SessionLoadCounter]:
    """Count InferenceSession.load calls during runner execution."""
    counter = SessionLoadCounter()
    original_load = InferenceSession.load
    original_load_fn = cast(Any, original_load)

    def wrapped_load(self: InferenceSession, *args: Any, **kwargs: Any) -> Any:
        counter.total_loads += 1
        return original_load_fn(self, *args, **kwargs)

    InferenceSession.load = wrapped_load  # type: ignore[method-assign]
    try:
        yield counter
    finally:
        InferenceSession.load = original_load  # type: ignore[method-assign]


def _resolve_cases(args: argparse.Namespace) -> list[GenerationCase]:
    """Combine explicit cases and generated shape/step matrices."""
    cases: list[GenerationCase] = list(args.cases or [])
    if args.shapes and args.step_counts:
        for shape in args.shapes:
            for step_count in args.step_counts:
                cases.append(
                    GenerationCase(
                        width=shape.width,
                        height=shape.height,
                        num_inference_steps=step_count,
                    )
                )

    if not cases:
        raise ValueError(
            "Provide at least one --case or a --shape/--step-count matrix."
        )
    return cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the same-process multi-shape runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple diffusion requests with different shapes/steps in "
            "one process using a shared tokenizer and pipeline."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument(
        "--case",
        dest="cases",
        type=_parse_case,
        action="append",
        help="A run case in WIDTHxHEIGHT:STEPS format. Repeat this flag.",
    )
    parser.add_argument(
        "--shape",
        dest="shapes",
        type=_parse_shape,
        action="append",
        help=(
            "A shape in WIDTHxHEIGHT format. Combine with repeated "
            "--step-count values to auto-generate a case matrix."
        ),
    )
    parser.add_argument(
        "--step-count",
        dest="step_counts",
        type=int,
        action="append",
        help="A step count used with --shape to auto-generate cases.",
    )
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--true-cfg-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--secondary-max-length", type=int, default=None)
    parser.add_argument(
        "--input-image",
        type=str,
        action="append",
        default=None,
        help="Optional input image(s) for image-edit runs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="/tmp/same_process_multi_shape",
        help="Prefix for saved output files.",
    )
    parser.add_argument(
        "--output-ext",
        type=str,
        default="jpg",
        help="Image extension to save for each case, e.g. jpg or png.",
    )
    return parser.parse_args(argv)


async def run_cases(args: argparse.Namespace) -> None:
    """Execute all requested cases sequentially in one process."""
    preload_bundled_nvidia_runtime_libraries()
    cases = _resolve_cases(args)

    with count_session_loads() as load_counter:
        print(f"Loading model once for same-process runs: {args.model}")
        _, arch, tokenizer, pipeline = build_pipeline_and_tokenizer(
            args.model,
            max_length=args.max_length,
            secondary_max_length=args.secondary_max_length,
        )
        input_image_data_uris = load_input_image_data_uris(args.input_image)
        init_loads = load_counter.mark()
        last_case_mark = init_loads
        dynamic_loads_total = 0

        print(f"Running {len(cases)} cases in one process")
        print(f"[summary] init_session_loads={init_loads}")
        for index, case in enumerate(cases, start=1):
            case_start_mark = load_counter.mark()
            request, _, _ = build_generation_request(
                arch_name=arch.name,
                model_path=args.model,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                width=case.width,
                height=case.height,
                num_inference_steps=case.num_inference_steps,
                guidance_scale=args.guidance_scale,
                true_cfg_scale=args.true_cfg_scale,
                seed=args.seed,
                input_image_data_uris=input_image_data_uris,
            )
            context, inputs = await build_context_and_inputs(tokenizer, request)
            prepare_loads = load_counter.delta_since(case_start_mark)

            print(
                f"[case {index}/{len(cases)}] "
                f"{case.width}x{case.height}, steps={case.num_inference_steps}"
            )
            execute_start_mark = load_counter.mark()
            start_time = perf_counter()
            outputs = pipeline.execute(inputs)
            elapsed_s = perf_counter() - start_time
            execute_loads = load_counter.delta_since(execute_start_mark)
            postprocess_start_mark = load_counter.mark()
            output = await postprocess_output(tokenizer, outputs, context)
            postprocess_loads = load_counter.delta_since(postprocess_start_mark)

            if not output.is_done:
                raise RuntimeError(
                    f"Case {index} finished with status {output.final_status}"
                )

            output_path = (
                f"{args.output_prefix}_{index}_{case.width}x{case.height}_"
                f"{case.num_inference_steps}steps.{args.output_ext.lstrip('.')}"
            )
            saved_paths = save_generation_output(output, output_path)
            saved_path_summary = ", ".join(
                str(Path(saved_path).resolve()) for saved_path in saved_paths
            )

            case_loads = load_counter.delta_since(last_case_mark)
            dynamic_loads_total += case_loads
            last_case_mark = load_counter.mark()
            print(
                f"[case {index}/{len(cases)}] "
                f"elapsed={elapsed_s:.3f}s "
                f"session_loads_prepare={prepare_loads} "
                f"session_loads_execute={execute_loads} "
                f"session_loads_postprocess={postprocess_loads} "
                f"session_loads_total={case_loads} "
                f"saved={saved_path_summary}"
            )

        print(
            f"[summary] dynamic_session_loads_after_init={dynamic_loads_total}"
        )


def main(argv: list[str] | None = None) -> int:
    """Program entrypoint."""
    args = parse_args(argv)
    try:
        asyncio.run(run_cases(args))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
