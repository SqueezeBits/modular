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
import logging
import os
import subprocess
from typing import Any, cast

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
)
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)

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
    parser.add_argument("--model", required=True, help="Model identifier.")
    parser.add_argument(
        "--quantization-encoding",
        type=str,
        default=None,
        help="Override quantization encoding (e.g. 'bfloat16' for models that declare float32).",
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
        "--height", type=int, default=None, help="Video height in pixels. Auto-computed from input image if omitted."
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Video width in pixels. Auto-computed from input image if omitted."
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
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
    parser.add_argument(
        "--manual-warmup",
        action="store_true",
        default=False,
        help="Run one full pipeline iteration as warmup before the timed run.",
    )
    parser.add_argument(
        "--lora-repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID for LoRA weights (e.g. lightx2v/Wan2.2-Lightning).",
    )
    parser.add_argument(
        "--lora-subfolder",
        type=str,
        default=None,
        help="Subfolder in the LoRA repo containing safetensors files.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA strength multiplier for primary transformer (high-noise).",
    )
    parser.add_argument(
        "--lora-scale-2",
        type=float,
        default=None,
        help="LoRA strength multiplier for transformer_2 (low-noise). Defaults to --lora-scale.",
    )
    parser.add_argument(
        "--lora-weight-name",
        type=str,
        default=None,
        help=(
            "Single LoRA safetensors filename (for non-MoE models). "
            "When set, --lora-subfolder is treated as the HF subfolder "
            "and this file is downloaded instead of high/low noise pairs."
        ),
    )

    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Path to input image for I2V (image-to-video) generation.",
    )
    parser.add_argument(
        "--initial-noise",
        type=str,
        default=None,
        help="Path to .npy file with pre-generated initial noise (for parity testing).",
    )
    parser.add_argument(
        "--shared-inputs",
        type=str,
        default=None,
        help="Dir with dumped diffusers intermediates for strict parity testing.",
    )

    # Wan-Animate arguments
    parser.add_argument(
        "--driving-video",
        type=str,
        default=None,
        help="Raw driving video for Wan-Animate. Used with --preprocess.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess --driving-video to extract pose skeleton and face crops.",
    )
    parser.add_argument(
        "--save-preprocessed",
        type=str,
        default=None,
        help="Directory to save preprocessed pose/face videos for reuse.",
    )
    parser.add_argument(
        "--pose-video",
        type=str,
        default=None,
        help="Path to preprocessed pose video for Wan-Animate.",
    )
    parser.add_argument(
        "--face-video",
        type=str,
        default=None,
        help="Path to preprocessed face video for Wan-Animate.",
    )
    parser.add_argument(
        "--mode",
        choices=["animate", "replace"],
        default="animate",
        help="Wan-Animate mode: 'animate' (motion transfer) or 'replace' (character replacement).",
    )
    parser.add_argument(
        "--background-video",
        type=str,
        default=None,
        help="Path to background video for Wan-Animate replace mode.",
    )
    parser.add_argument(
        "--mask-video",
        type=str,
        default=None,
        help="Path to mask video for Wan-Animate replace mode.",
    )
    parser.add_argument(
        "--segment-frame-length",
        type=int,
        default=77,
        help="Number of frames per segment for Wan-Animate.",
    )
    parser.add_argument(
        "--prev-segment-conditioning-frames",
        type=int,
        default=1,
        help="Number of overlap frames between segments for Wan-Animate.",
    )
    parser.add_argument(
        "--motion-encode-batch-size",
        type=int,
        default=8,
        help="Batch size for motion encoder in Wan-Animate.",
    )
    args = parser.parse_args(argv)
    assert args.prompt, "Prompt must be a non-empty string."

    # Validate --preprocess / --driving-video / --pose-video mutual exclusivity
    if args.preprocess:
        assert args.driving_video, (
            "--driving-video is required when --preprocess is set."
        )
        assert args.pose_video is None and args.face_video is None, (
            "--pose-video and --face-video cannot be used with --preprocess."
        )
    elif args.driving_video:
        assert False, (
            "--driving-video requires --preprocess to extract pose and face."
        )
    elif args.pose_video:
        assert args.face_video, (
            "--face-video is required when --pose-video is provided."
        )

    # Auto-compute height/width from input image if not specified
    if args.height is None or args.width is None:
        if args.input_image:
            import PIL.Image as _PIL

            _img = _PIL.open(args.input_image)
            aspect_ratio = _img.height / _img.width
            # Default to 720p area
            max_area = 720 * 1280
            mod_value = 16  # vae_scale_factor_spatial(8) * patch_size(2)
            h = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            w = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            if args.height is None:
                args.height = h
            if args.width is None:
                args.width = w
            print(f"Auto-computed resolution: {args.width}x{args.height} (from {_img.size[0]}x{_img.size[1]})")
        else:
            if args.height is None:
                args.height = 720
            if args.width is None:
                args.width = 1280

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
        assert args.warmup_num_inference_steps > 0, (
            "warmup-num-inference-steps must be positive."
        )
    if args.warmup_guidance_scale is not None:
        assert args.warmup_guidance_scale > 0.0, (
            "warmup-guidance-scale must be positive."
        )
    if args.warmup_guidance_scale_2 is not None:
        assert args.warmup_guidance_scale_2 > 0.0, (
            "warmup-guidance-scale-2 must be positive."
        )
    return args


def _normalize_wan_num_frames(num_frames: int, *, phase: str) -> int:
    """Round Wan frame counts up to the nearest valid 1 + 4k size."""
    if num_frames <= 1:
        return 1

    remainder = (num_frames - 1) % WAN_TEMPORAL_FRAME_STRIDE
    if remainder == 0:
        return num_frames

    adjusted_num_frames = num_frames + WAN_TEMPORAL_FRAME_STRIDE - remainder
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
        args.guidance_scale_2 if guidance_scale_2 is None else guidance_scale_2
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
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_data = b"".join(f.astype(np.uint8).tobytes() for f in frames)
    _, stderr_bytes = proc.communicate(input=raw_data)
    if proc.returncode != 0:
        print(
            f"WARNING: ffmpeg returned {proc.returncode}: {stderr_bytes.decode()}"
        )
    else:
        print(f"Video saved to: {output_path}")


def _load_video_frames_ffmpeg(video_path: str) -> list[Any]:
    """Load video frames as list of PIL Images using ffmpeg."""
    import PIL.Image

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path,
    ]
    info = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    w, h = (int(x) for x in info.split(","))

    cmd = [
        "ffmpeg", "-i", video_path, "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-v", "error", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    raw = proc.stdout
    frame_size = w * h * 3
    num_frames = len(raw) // frame_size
    frames: list[Any] = []
    for i in range(num_frames):
        frame = np.frombuffer(
            raw, dtype=np.uint8, count=frame_size, offset=i * frame_size
        ).reshape(h, w, 3)
        frames.append(PIL.Image.fromarray(frame))
    return frames


def _video_frames_from_raw_output(images: np.ndarray) -> list[np.ndarray]:
    """Convert raw pipeline video output [B, C, T, H, W] to uint8 RGB frames."""
    if images.ndim != 5:
        raise ValueError(
            f"Expected video output with rank 5 [B, C, T, H, W], got {images.shape}."
        )

    video = (images[0] * 0.5 + 0.5).clip(min=0.0, max=1.0)
    video = np.transpose(video, (1, 2, 3, 0))
    video = (video * 255.0).round().astype(np.uint8, copy=False)
    return [video[t] for t in range(video.shape[0])]


async def generate_video(args: argparse.Namespace) -> None:
    print(f"Loading model: {args.model}")

    model_kwargs: dict[str, Any] = {
        "model_path": args.model,
        "device_specs": [DeviceSpec.accelerator()],
    }
    if args.quantization_encoding:
        model_kwargs["quantization_encoding"] = args.quantization_encoding
    config = PipelineConfig(
        model=MAXModelConfig(**model_kwargs),
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, "No matching diffusion architecture found."

    # Animate auto-routing: if pose/driving-video provided, switch to WanAnimatePipeline
    is_animate = args.pose_video or (args.driving_video and args.preprocess)
    if is_animate and arch.name in ("WanPipeline", "WanImageToVideoPipeline"):
        animate_arch = PIPELINE_REGISTRY.architectures.get("WanAnimatePipeline")
        if animate_arch is not None:
            print(
                "Animate auto-routing → WanAnimatePipeline"
                " (--pose-video provided)"
            )
            arch = animate_arch

    # TI2V auto-routing: if resolved as WanPipeline but --input-image provided,
    # switch to WanI2VPipeline for image conditioning support.
    if arch.name == "WanPipeline" and args.input_image:
        i2v_arch = PIPELINE_REGISTRY.architectures.get(
            "WanImageToVideoPipeline"
        )
        if i2v_arch is not None:
            print(
                "TI2V auto-routing: WanPipeline → WanI2VPipeline"
                " (--input-image provided)"
            )
            arch = i2v_arch

    # Wan-Animate defaults to 30 fps (matching diffusers/official config)
    if arch.name == "WanAnimatePipeline" and args.fps == 16:
        args.fps = 30
        print("Wan-Animate: auto-set fps=30 (use --fps to override)")

    # Inject LoRA config into diffusers_config if CLI args provided
    diffusers_config = config.model.diffusers_config
    if args.lora_repo_id and diffusers_config is not None:
        lora_cfg: dict[str, Any] = {
            "repo_id": args.lora_repo_id,
            "subfolder": args.lora_subfolder or "",
            "scale": args.lora_scale,
            "scale_2": args.lora_scale_2
            if args.lora_scale_2 is not None
            else args.lora_scale,
        }
        if args.lora_weight_name:
            lora_cfg["filenames"] = [args.lora_weight_name]
        diffusers_config["lora"] = lora_cfg

    # Tokenizer setup
    max_length = args.max_length
    secondary_max_length = args.secondary_max_length
    has_tokenizer_2 = False
    if (
        max_length is None
        and diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and components_config.get("tokenizer", None) is not None
    ):
        max_length = components_config["tokenizer"]["config_dict"].get(
            "model_max_length", None
        )
        if arch.name in ("WanPipeline", "WanImageToVideoPipeline"):
            max_length = 226
        elif arch.name == "WanAnimatePipeline":
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
    if arch.name == "WanAnimatePipeline" and (args.pose_video or args.driving_video):
        # Derive frame count from pose or driving video using ffprobe
        video_for_frames = args.pose_video or args.driving_video
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video_for_frames],
            capture_output=True, text=True,
        )
        effective_num_frames = int(result.stdout.strip())
        print(f"Wan-Animate: derived {effective_num_frames} frames from {video_for_frames}")
    elif arch.name in ("WanPipeline", "WanImageToVideoPipeline"):
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
    # Load input image for I2V if provided
    input_image = None
    if args.input_image:
        import PIL.Image
        input_image = PIL.Image.open(args.input_image).convert("RGB")
        print(f"Input image: {args.input_image} ({input_image.size[0]}x{input_image.size[1]})")
    context = await tokenizer.new_context(request, input_image=input_image)

    # Attach animate-specific fields to context
    if is_animate and arch.name == "WanAnimatePipeline":
        if args.preprocess and args.driving_video:
            from max.examples.diffusion.wan_animate_preprocess import (
                preprocess_driving_video,
                save_preprocessed,
            )

            raw_frames = _load_video_frames_ffmpeg(args.driving_video)
            print(f"Preprocessing {len(raw_frames)} driving video frames...")
            pose_frames, face_frames = preprocess_driving_video(
                raw_frames, device="cuda"
            )
            if args.save_preprocessed:
                save_preprocessed(
                    pose_frames, face_frames, args.save_preprocessed
                )
            context.pose_video = pose_frames
            context.face_video = face_frames
            print(
                f"Preprocessing complete: {len(pose_frames)} pose frames, "
                f"{len(face_frames)} face frames."
            )
        else:
            context.pose_video = _load_video_frames_ffmpeg(args.pose_video)
            context.face_video = _load_video_frames_ffmpeg(args.face_video) if args.face_video else None
        context.input_image = input_image
        context.mode = args.mode
        context.segment_frame_length = args.segment_frame_length
        context.prev_segment_conditioning_frames = args.prev_segment_conditioning_frames
        context.motion_encode_batch_size = args.motion_encode_batch_size
        if args.background_video:
            context.background_video = _load_video_frames_ffmpeg(args.background_video)
        if args.mask_video:
            context.mask_video = _load_video_frames_ffmpeg(args.mask_video)
        print(
            f"Wan-Animate: {len(context.pose_video)} pose frames, "
            f"mode={args.mode}, segment_len={args.segment_frame_length}"
        )

    # Override initial noise if provided (for parity testing)
    if args.initial_noise:
        noise = np.load(args.initial_noise).astype(np.float32)
        context.latents = noise
        print(f"Loaded initial noise from {args.initial_noise}: {noise.shape}")

    # Shared inputs directory for strict parity testing
    if args.shared_inputs:
        context.shared_inputs_dir = args.shared_inputs
        # Also load seg 0 noise as initial latents if available
        seg0_noise = os.path.join(args.shared_inputs, "noise_seg0.npy")
        if os.path.exists(seg0_noise) and not args.initial_noise:
            noise = np.load(seg0_noise).astype(np.float32)
            context.latents = noise
            print(f"Loaded seg0 noise from shared inputs: {noise.shape}")
        print(f"Using shared inputs from {args.shared_inputs}")

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
        if arch.name in ("WanPipeline", "WanImageToVideoPipeline", "WanAnimatePipeline"):
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
        if arch.name in ("WanPipeline", "WanImageToVideoPipeline", "WanAnimatePipeline"):
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

        if getattr(args, "manual_warmup", False):
            # Use full-resolution warmup (avoids CLIP preprocessing
            # crashes with small dummy inputs for animate pipelines).
            print("Running manual warmup (full pipeline)...")
            wu_inputs = pipeline._pipeline_model.prepare_inputs(context)
            pipeline._pipeline_model.execute(wu_inputs)
            print("Manual warmup complete.")
        else:
            for i in range(args.num_warmups):
                print(f"Running warmup {i + 1} of {args.num_warmups}")
                print(
                    "Warmup parameters: "
                    f"{warmup_height}x{warmup_width}, {warmup_num_frames} frames, "
                    f"steps={warmup_num_inference_steps}, guidance={warmup_guidance_scale}"
                )
                pipeline.execute(inputs_warmup)

        with profile_execute(
            pipeline, patch_tensor_ops=True
        ) as prof:
            for i in range(args.num_profile_iterations):
                print(
                    f"Running inference {i + 1} of {args.num_profile_iterations}"
                )
                try:
                    outputs = pipeline.execute(inputs)
                except Exception:
                    # Video output post-processing may fail through the
                    # PixelGenerationPipeline wrapper; timing is still valid.
                    outputs = None
        prof.report(unit="ms")
        if outputs is None:
            print("Profiling complete (output post-processing skipped).")
            return

        output = outputs[context.request_id]
        output = await tokenizer.postprocess(output)

        if not output.is_done:
            print(f"WARNING: Generation status: {output.final_status}")
            return

        print("Generation complete!")
        if not output.output:
            print("ERROR: No output generated")
            return

        frames = []
        for image_content in output.output:
            if not getattr(image_content, "image_data", None):
                continue
            print(
                "ERROR: profile mode still returned encoded image payloads; "
                "run without --profile-timings for the raw fast path."
            )
            return
    else:
        import time as _time

        # Optional manual warmup: run once to compile all graphs, then measure.
        if getattr(args, "manual_warmup", False):
            print("Running manual warmup iteration...")
            wu_inputs = pipeline._pipeline_model.prepare_inputs(context)
            pipeline._pipeline_model.execute(wu_inputs)
            print("Warmup complete. Starting timed run...")

        t0 = _time.perf_counter()
        model_inputs = pipeline._pipeline_model.prepare_inputs(context)
        t1 = _time.perf_counter()
        model_outputs = pipeline._pipeline_model.execute(model_inputs)
        t2 = _time.perf_counter()
        if not isinstance(model_outputs.images, np.ndarray):
            raise TypeError(
                "Expected raw numpy video output from the diffusion pipeline."
            )
        print("Generation complete!")
        total = t2 - t0
        print(
            f"Timing: prepare_inputs={t1 - t0:.2f}s, "
            f"execute={t2 - t1:.2f}s, "
            f"total={total:.2f}s"
        )
        frames = _video_frames_from_raw_output(model_outputs.images)

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
