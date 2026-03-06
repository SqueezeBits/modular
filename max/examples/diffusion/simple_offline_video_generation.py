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

"""Simple offline video generation example using diffusion models.

Usage:
    ./bazelw run //max/examples/diffusion:simple_offline_video_generation -- \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --prompt "A cat playing piano" \
        --output output.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import subprocess
from io import BytesIO
from typing import cast

import numpy as np
from max.driver import DeviceSpec
from max.examples.diffusion.profiler import profile_execute
from max.interfaces import (
    PipelineTask,
    PixelGenerationInputs,
    RequestID,
)
from max.interfaces.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
    VideoProviderOptions,
)
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import (
    OpenResponsesRequestBody,
    OutputImageContent,
)
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(name)s %(levelname)s %(message)s"
)


DEFAULT_WAN_WARMUP_HEIGHT = 480
DEFAULT_WAN_WARMUP_WIDTH = 832
DEFAULT_WAN_WARMUP_NUM_FRAMES = 5
DEFAULT_WAN_WARMUP_NUM_INFERENCE_STEPS = 5
WAN_TEMPORAL_FRAME_STRIDE = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos with a diffusion model.",
    )
    parser.add_argument(
        "--model", required=True, help="Model identifier."
    )
    parser.add_argument(
        "--prompt", required=True, help="Text prompt for video generation."
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry, distorted, deformed, ugly, bad, poor, worst quality",
        help="Negative prompt to guide what NOT to generate.",
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Video height in pixels."
    )
    parser.add_argument(
        "--width", type=int, default=832, help="Video width in pixels."
    )
    parser.add_argument(
        "--num-frames", type=int, default=81, help="Number of video frames."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=40,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=3.0,
        help="Secondary guidance scale for low-noise expert (MoE models).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video filename.",
    )
    parser.add_argument(
        "--fps", type=int, default=16, help="Frames per second."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum length of tokenizer.",
    )
    parser.add_argument(
        "--secondary-max-length",
        type=int,
        default=None,
        help="Maximum length of secondary tokenizer.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Profile timings of the pipeline.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=1,
        help="Number of warmup runs before profiling.",
    )
    parser.add_argument(
        "--num-profile-iterations",
        type=int,
        default=1,
        help="Number of iterations for profiling.",
    )
    parser.add_argument(
        "--warmup-prompt",
        type=str,
        default="warmup",
        help="Prompt to use for warmup runs during profiling.",
    )
    parser.add_argument(
        "--warmup-negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt override for warmup runs.",
    )
    parser.add_argument(
        "--warmup-height",
        type=int,
        default=None,
        help=(
            "Optional warmup video height override. "
            "If omitted, Wan profiling defaults to 480."
        ),
    )
    parser.add_argument(
        "--warmup-width",
        type=int,
        default=None,
        help=(
            "Optional warmup video width override. "
            "If omitted, Wan profiling defaults to 832."
        ),
    )
    parser.add_argument(
        "--warmup-num-frames",
        type=int,
        default=None,
        help=(
            "Optional warmup frame-count override. "
            "If omitted, Wan profiling defaults to 5."
        ),
    )
    parser.add_argument(
        "--warmup-num-inference-steps",
        type=int,
        default=None,
        help=(
            "Optional warmup denoising-step override. "
            "If omitted, Wan profiling defaults to 5."
        ),
    )
    parser.add_argument(
        "--warmup-guidance-scale",
        type=float,
        default=None,
        help="Optional warmup CFG scale override.",
    )
    parser.add_argument(
        "--warmup-guidance-scale-2",
        type=float,
        default=None,
        help="Optional warmup low-noise CFG scale override.",
    )

    args = parser.parse_args(argv)
    assert args.prompt, "Prompt must be a non-empty string."
    assert args.height > 0, "Height must be positive."
    assert args.width > 0, "Width must be positive."
    assert args.num_frames > 0, "num-frames must be positive."
    assert args.num_inference_steps > 0, "num-inference-steps must be positive."
    assert args.guidance_scale > 0.0, "guidance-scale must be positive."
    if args.warmup_height is not None:
        assert args.warmup_height > 0, "warmup-height must be positive."
    if args.warmup_width is not None:
        assert args.warmup_width > 0, "warmup-width must be positive."
    if args.warmup_num_frames is not None:
        assert args.warmup_num_frames > 0, "warmup-num-frames must be positive."
    if args.warmup_num_inference_steps is not None:
        assert (
            args.warmup_num_inference_steps > 0
        ), "warmup-num-inference-steps must be positive."
    if args.warmup_guidance_scale is not None:
        assert (
            args.warmup_guidance_scale > 0.0
        ), "warmup-guidance-scale must be positive."
    if args.warmup_guidance_scale_2 is not None:
        assert (
            args.warmup_guidance_scale_2 > 0.0
        ), "warmup-guidance-scale-2 must be positive."
    return args


def _normalize_wan_num_frames(num_frames: int, *, phase: str) -> int:
    """Round Wan frame counts up to the nearest valid 1 + 4k size."""
    if num_frames <= 1:
        return 1

    remainder = (num_frames - 1) % WAN_TEMPORAL_FRAME_STRIDE
    if remainder == 0:
        return num_frames

    adjusted_num_frames = (
        num_frames + WAN_TEMPORAL_FRAME_STRIDE - remainder
    )
    print(
        "WanPipeline adjusted "
        f"{phase} num_frames from {num_frames} to {adjusted_num_frames}; "
        "Wan VAE temporal decode is stable on frame counts of the form 1 + 4k."
    )
    return adjusted_num_frames


def _build_request_body(
    args: argparse.Namespace,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    guidance_scale_2: float | None = None,
) -> OpenResponsesRequestBody:
    """Build an OpenResponsesRequestBody for video generation."""
    negative_prompt = (
        args.negative_prompt if negative_prompt is None else negative_prompt
    )
    height = args.height if height is None else height
    width = args.width if width is None else width
    num_frames = args.num_frames if num_frames is None else num_frames
    num_inference_steps = (
        args.num_inference_steps
        if num_inference_steps is None
        else num_inference_steps
    )
    guidance_scale = (
        args.guidance_scale if guidance_scale is None else guidance_scale
    )
    guidance_scale_2 = (
        args.guidance_scale_2
        if guidance_scale_2 is None
        else guidance_scale_2
    )

    return OpenResponsesRequestBody(
        model=args.model,
        input=prompt,
        seed=args.seed,
        provider_options=ProviderOptions(
            image=ImageProviderOptions(
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ),
            video=VideoProviderOptions(
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                steps=num_inference_steps,
                num_frames=num_frames,
                frames_per_second=args.fps,
                guidance_scale_2=guidance_scale_2,
            ),
        ),
    )


def save_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    """Encode frames to mp4 using ffmpeg."""
    if not frames:
        print("ERROR: No frames to save")
        return

    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "18",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_data = b"".join(f.astype(np.uint8).tobytes() for f in frames)
    _, stderr_bytes = proc.communicate(input=raw_data)
    if proc.returncode != 0:
        print(f"WARNING: ffmpeg returned {proc.returncode}: {stderr_bytes.decode()}")
    else:
        print(f"Video saved to: {output_path}")


async def generate_video(args: argparse.Namespace) -> None:
    print(f"Loading model: {args.model}")

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=args.model,
            device_specs=[DeviceSpec.accelerator()],
        ),
        prefer_module_v3=True,
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        prefer_module_v3=config.prefer_module_v3,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, "No matching diffusion architecture found."

    # Tokenizer setup
    diffusers_config = config.model.diffusers_config
    max_length = args.max_length
    secondary_max_length = args.secondary_max_length
    has_tokenizer_2 = False
    if (
        max_length is None
        and
        diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and components_config.get("tokenizer", None) is not None
    ):
        max_length = components_config["tokenizer"]["config_dict"].get(
            "model_max_length", None
        )
        if arch.name == "WanPipeline":
            max_length = 512
        print(f"Using max length: {max_length} for tokenizer")

    if (
        diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and components_config.get("tokenizer_2", None) is not None
    ):
        has_tokenizer_2 = True
        if secondary_max_length is None:
            secondary_max_length = components_config["tokenizer_2"][
                "config_dict"
            ].get("model_max_length", None)
            print(
                "Using secondary max length: "
                f"{secondary_max_length} for tokenizer_2"
            )

    tokenizer = PixelGenerationTokenizer(
        model_path=args.model,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=max_length,
        subfolder_2="tokenizer_2" if has_tokenizer_2 else None,
        secondary_max_length=secondary_max_length if has_tokenizer_2 else None,
    )

    # Pipeline setup
    assert issubclass(arch.pipeline_model, DiffusionPipeline), (
        f"Architecture does not implement DiffusionPipeline: {arch.pipeline_model}"
    )
    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )

    effective_num_frames = args.num_frames
    if arch.name == "WanPipeline":
        effective_num_frames = _normalize_wan_num_frames(
            args.num_frames, phase="main"
        )

    print(f"Generating video for prompt: '{args.prompt}'")
    print(
        f"Parameters: {args.height}x{args.width}, {effective_num_frames} frames, "
        f"steps={args.num_inference_steps}, guidance={args.guidance_scale}"
    )

    # Create main request
    body = _build_request_body(
        args, args.prompt, num_frames=effective_num_frames
    )
    request = OpenResponsesRequest(request_id=RequestID(), body=body)
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )

    # Execute (with optional profiling)
    if args.profile_timings:
        # Warmup
        warmup_negative_prompt = (
            args.warmup_negative_prompt
            if args.warmup_negative_prompt is not None
            else args.negative_prompt
        )
        if arch.name == "WanPipeline":
            warmup_height = (
                args.warmup_height
                if args.warmup_height is not None
                else DEFAULT_WAN_WARMUP_HEIGHT
            )
            warmup_width = (
                args.warmup_width
                if args.warmup_width is not None
                else DEFAULT_WAN_WARMUP_WIDTH
            )
            warmup_num_frames = (
                args.warmup_num_frames
                if args.warmup_num_frames is not None
                else DEFAULT_WAN_WARMUP_NUM_FRAMES
            )
            warmup_num_inference_steps = (
                args.warmup_num_inference_steps
                if args.warmup_num_inference_steps is not None
                else DEFAULT_WAN_WARMUP_NUM_INFERENCE_STEPS
            )
        else:
            warmup_height = (
                args.warmup_height
                if args.warmup_height is not None
                else args.height
            )
            warmup_width = (
                args.warmup_width
                if args.warmup_width is not None
                else args.width
            )
            warmup_num_frames = (
                args.warmup_num_frames
                if args.warmup_num_frames is not None
                else args.num_frames
            )
            warmup_num_inference_steps = (
                args.warmup_num_inference_steps
                if args.warmup_num_inference_steps is not None
                else args.num_inference_steps
            )
        if arch.name == "WanPipeline":
            warmup_num_frames = _normalize_wan_num_frames(
                warmup_num_frames, phase="warmup"
            )
        warmup_guidance_scale = (
            args.warmup_guidance_scale
            if args.warmup_guidance_scale is not None
            else args.guidance_scale
        )
        warmup_guidance_scale_2 = (
            args.warmup_guidance_scale_2
            if args.warmup_guidance_scale_2 is not None
            else args.guidance_scale_2
        )
        body_warmup = _build_request_body(
            args,
            args.warmup_prompt,
            negative_prompt=warmup_negative_prompt,
            height=warmup_height,
            width=warmup_width,
            num_frames=warmup_num_frames,
            num_inference_steps=warmup_num_inference_steps,
            guidance_scale=warmup_guidance_scale,
            guidance_scale_2=warmup_guidance_scale_2,
        )
        request_warmup = OpenResponsesRequest(
            request_id=RequestID(), body=body_warmup
        )
        context_warmup = await tokenizer.new_context(request_warmup)
        inputs_warmup = PixelGenerationInputs[PixelContext](
            batch={context_warmup.request_id: context_warmup}
        )

        for i in range(args.num_warmups):
            print(f"Running warmup {i + 1} of {args.num_warmups}")
            print(
                "Warmup parameters: "
                f"{warmup_height}x{warmup_width}, {warmup_num_frames} frames, "
                f"steps={warmup_num_inference_steps}, guidance={warmup_guidance_scale}"
            )
            pipeline.execute(inputs_warmup)

        with profile_execute(
            pipeline, patch_concat=True, patch_tensor_ops=True
        ) as prof:
            for i in range(args.num_profile_iterations):
                print(
                    f"Running inference {i + 1} of {args.num_profile_iterations}"
                )
                outputs = pipeline.execute(inputs)
        print(f"Method timings:\n{prof.report(unit='ms')}")
        print(f"Module timings:\n{prof.report_modules(unit='ms')}")
    else:
        outputs = pipeline.execute(inputs)

    # Post-process
    output = outputs[context.request_id]
    output = await tokenizer.postprocess(output)

    if not output.is_done:
        print(f"WARNING: Generation status: {output.final_status}")
        return

    print("Generation complete!")
    if not output.output:
        print("ERROR: No output generated")
        return

    # Extract frames and save video
    frames = []
    for image_content in output.output:
        if not isinstance(image_content, OutputImageContent):
            continue
        if image_content.image_data:
            image_bytes = base64.b64decode(image_content.image_data)
            frame = Image.open(BytesIO(image_bytes)).convert("RGB")
            frames.append(np.array(frame))

    if not frames:
        print("ERROR: No frames generated")
        return

    print(f"Saving {len(frames)} frames as video to {args.output}")
    save_video(frames, args.output, args.fps)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(generate_video(args))
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
