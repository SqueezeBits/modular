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

"""Simple offline pixel generation example using diffusion models.

This module demonstrates end-to-end pixel generation using:
- OpenResponsesRequest: Create generation requests with prompts
- PixelGenerationTokenizer: Tokenize prompts and prepare model context
- PixelGenerationPipeline: Execute the diffusion model to generate pixels

Usage:
    ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
        --model black-forest-labs/FLUX.2-dev \
        --prompt "A cat in a garden"
"""

from __future__ import annotations

import argparse
import asyncio
import os

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
from max.examples.diffusion.profiler import profile_execute


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pixel generation example.

    Args:
        argv: Optional explicit list of argument strings. If None, arguments
            are read from sys.argv[1:].

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate images with a diffusion model.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Identifier of the model to use for generation (e.g., black-forest-labs/FLUX.2-dev).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt describing the image to generate.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt to guide what NOT to generate.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of generated image in pixels. None uses model's native resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of generated image in pixels. None uses model's native resolution.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps. More steps = higher quality but slower.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help=(
            "Guidance scale for classifier-free guidance. "
            "If omitted, defaults to 1.0 for QwenImage family and 3.5 otherwise."
        ),
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=None,
        help=(
            "True classifier-free guidance scale. "
            "If omitted, defaults to 4.0 for QwenImage family when negative prompt is provided, "
            "and 1.0 otherwise."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output filename for the generated image.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum length of tokenizer",
    )
    parser.add_argument(
        "--secondary-max-length",
        type=int,
        default=None,
        help="Maximum length of secondary tokenizer",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        action="append",
        default=None,
        help="Input image for image-to-image generation. Can be specified multiple times.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Profile timings of the pipeline.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help=(
            "Number of warmup iterations to run before the timed execution. "
            "Use >=1 to pre-compile JIT graphs and obtain steady-state timings."
        ),
    )
    parser.add_argument(
        "--num-profile-iterations",
        type=int,
        default=3,
        help="Number of iterations to run for profiling.",
    )

    args = parser.parse_args(argv)

    # Validate arguments
    assert args.prompt, "Prompt must be a non-empty string."
    if args.height is not None:
        assert args.height > 0, "Height must be a positive integer."
    if args.width is not None:
        assert args.width > 0, "Width must be a positive integer."
    assert args.num_inference_steps > 0, (
        "num-inference-steps must be a positive integer."
    )
    if args.guidance_scale is not None:
        assert args.guidance_scale > 0.0, "guidance-scale must be positive."
    if args.true_cfg_scale is not None:
        assert args.true_cfg_scale > 0.0, "true-cfg-scale must be positive."

    return args


async def generate_image(args: argparse.Namespace) -> None:
    """Main generation logic.

    Args:
        args: Parsed command-line arguments
    """
    preload_bundled_nvidia_runtime_libraries()

    print(f"Loading model: {args.model}")
    _, arch, tokenizer, pipeline = build_pipeline_and_tokenizer(
        args.model,
        max_length=args.max_length,
        secondary_max_length=args.secondary_max_length,
    )

    print(f"Generating image for prompt: '{args.prompt}'")

    input_image_data_uris = load_input_image_data_uris(args.input_image)
    request, guidance_scale, true_cfg_scale = build_generation_request(
        arch_name=arch.name,
        model_path=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        seed=args.seed,
        input_image_data_uris=input_image_data_uris,
    )

    print(
        "Parameters: "
        f"steps={args.num_inference_steps}, guidance={guidance_scale}, true_cfg={true_cfg_scale}"
    )

    context, inputs = await build_context_and_inputs(tokenizer, request)
    print(
        f"Context created: {context.height}x{context.width}, "
        f"{context.num_inference_steps} steps"
    )

    # Step 6-1: Warmup — run before profiling or timed execution so that JIT
    # compilation completes and steady-state performance can be measured.
    if args.num_warmups > 0:
        request_warmup, _, _ = build_generation_request(
            arch_name=arch.name,
            model_path=args.model,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            seed=args.seed,
            input_image_data_uris=input_image_data_uris,
        )
        _, inputs_warmup = await build_context_and_inputs(
            tokenizer, request_warmup
        )
        for i in range(args.num_warmups):
            print(f"Running warmup {i + 1} of {args.num_warmups}")
            pipeline.execute(inputs_warmup)
        print("Warmup complete")

    # Step 7: Execute the pipeline
    print("Running diffusion model...")
    if args.profile_timings:
        with profile_execute(pipeline) as prof:
            for i in range(args.num_profile_iterations):
                print(
                    f"Running inference {i + 1} of {args.num_profile_iterations}"
                )
                outputs = pipeline.execute(inputs)
        prof.report(unit="ms")
    else:
        outputs = pipeline.execute(inputs)

    # Step 8: Get the output for our request
    output = await postprocess_output(tokenizer, outputs, context)

    # Check if generation completed successfully
    if not output.is_done:
        print(f"WARNING: Generation status: {output.final_status}")
        return

    print("Generation complete!")

    save_generation_output(output, args.output)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the pixel generation example.

    Args:
        argv: Optional explicit list of argument strings. If None, arguments
            are read from sys.argv[1:].

    Returns:
        Process exit code. 0 indicates success.
    """
    args = parse_args(argv)

    try:
        asyncio.run(generate_image(args))
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    raise SystemExit(main())
