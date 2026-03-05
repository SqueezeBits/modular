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
import base64
import os
from io import BytesIO
from typing import Any, cast

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
    InputImageContent,
    InputTextContent,
    OpenResponsesRequestBody,
    OutputImageContent,
    OutputVideoContent,
    UserMessage,
)
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)
from PIL import Image


ENV_FILES = ("/root/.env", "/workspace/.env")


def apply_env_overrides() -> None:
    """Load runtime environment overrides from local .env files."""
    for env_path in ENV_FILES:
        if not os.path.isfile(env_path):
            continue

        with open(env_path, encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                if key.startswith("export "):
                    key = key[len("export ") :].strip()
                if not key:
                    continue

                value = value.strip()
                if (
                    len(value) >= 2
                    and value[0] == value[-1]
                    and value[0] in {'"', "'"}
                ):
                    value = value[1:-1]
                os.environ[key] = value


def resolve_local_model_snapshot(model_path: str) -> str:
    """Resolve a local HF snapshot path when available.

    This keeps runs deterministic in offline/cached environments and avoids
    online metadata dependence for quantization detection.
    """
    if os.path.isdir(model_path):
        return model_path

    # Prefer explicit HF cache roots from env, before asking huggingface_hub.
    model_cache_name = "models--" + model_path.replace("/", "--")
    cache_roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
    ]
    for cache_root in cache_roots:
        if not cache_root:
            continue

        repo_cache_dir = os.path.join(cache_root, model_cache_name)
        snapshots_dir = os.path.join(repo_cache_dir, "snapshots")
        refs_main = os.path.join(repo_cache_dir, "refs", "main")

        revision: str | None = None
        if os.path.isfile(refs_main):
            with open(refs_main, encoding="utf-8") as f:
                revision = f.read().strip() or None

        if revision is not None:
            resolved = os.path.join(snapshots_dir, revision)
            if os.path.isdir(resolved):
                return resolved

        if os.path.isdir(snapshots_dir):
            candidates = [
                os.path.join(snapshots_dir, name)
                for name in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, name))
            ]
            if candidates:
                return max(candidates, key=os.path.getmtime)

    try:
        import huggingface_hub

        return huggingface_hub.snapshot_download(
            repo_id=model_path,
            local_files_only=True,
        )
    except Exception:
        return model_path


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
        "--num-frames",
        type=int,
        default=161,
        help="Number of frames for video generation models (e.g., LTX).",
    )
    parser.add_argument(
        "--frames-per-second",
        type=int,
        default=25,
        help="FPS for video generation models (e.g., LTX).",
    )
    parser.add_argument(
        "--decode-timestep",
        type=float,
        default=None,
        help="Decode timestep for timestep-aware video VAE decode.",
    )
    parser.add_argument(
        "--decode-noise-scale",
        type=float,
        default=None,
        help="Decode noise scale for timestep-aware video VAE decode.",
    )
    parser.add_argument(
        "--denoise-strength",
        type=float,
        default=None,
        help="Denoise strength in [0,1] for latent refinement.",
    )
    parser.add_argument(
        "--image-cond-noise-scale",
        type=float,
        default=None,
        help="Noise scale injected into hard image-conditioning latents.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale for classifier-free guidance. Set to 1.0 to disable CFG.",
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
        default=None,
        help="Input image for image-to-image generation.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Profile timings of the pipeline.",
    )
    parser.add_argument(
        "--save-packed-latents-npy",
        type=str,
        default=None,
        help=(
            "LTX only: save packed denoised latents ([B,S,D], float32) to this "
            ".npy path before VAE decode."
        ),
    )
    parser.add_argument(
        "--load-packed-latents-npy",
        type=str,
        default=None,
        help=(
            "LTX only: override initial packed latents ([B,S,D], float32) from "
            "this .npy path."
        ),
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
    assert args.num_frames > 0, "num-frames must be a positive integer."
    assert args.frames_per_second > 0, (
        "frames-per-second must be a positive integer."
    )
    if args.denoise_strength is not None:
        assert 0.0 <= args.denoise_strength <= 1.0, (
            "denoise-strength must be in [0, 1]."
        )
    if args.image_cond_noise_scale is not None:
        assert args.image_cond_noise_scale >= 0.0, (
            "image-cond-noise-scale must be >= 0."
        )
    assert args.guidance_scale > 0.0, "guidance-scale must be positive."

    return args


def save_image(image_data: str, output_path: str) -> None:
    """Save base64-encoded image data to a file.

    Args:
        image_data: Base64-encoded image data string
        output_path: Path where the image should be saved

    Raises:
        ImportError: If PIL is not available
    """
    try:
        from PIL import Image

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image.save(output_path)
        print(f"Image saved to: {output_path}")
    except ImportError:
        print("WARNING: PIL not available, cannot save image")
        print(f"Base64 data length: {len(image_data)} chars")


def save_video(video_data: str, output_path: str) -> None:
    """Save base64-encoded video data to a file."""
    video_bytes = base64.b64decode(video_data)
    actual_output_path = output_path
    if video_bytes.startswith((b"GIF87a", b"GIF89a")) and not output_path.lower().endswith(".gif"):
        base, _ = os.path.splitext(output_path)
        actual_output_path = f"{base}.gif"
        print(
            "WARNING: Encoded output is GIF data but output path had a non-GIF extension. "
            f"Saving as: {actual_output_path}"
        )

    with open(actual_output_path, "wb") as f:
        f.write(video_bytes)
    print(f"Video saved to: {actual_output_path}")


def save_pil_frames_as_gif(
    frames: list[Image.Image], output_path: str, fps: int
) -> None:
    """Save PIL frames to GIF."""
    if not frames:
        raise ValueError("No frames to save.")
    duration_ms = max(1, int(1000 / fps))
    first, rest = frames[0], frames[1:]
    first.save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )
    print(f"Video saved to: {output_path}")


def load_image_as_data_uri(image_path: str | None) -> str | None:
    """Load an image from a file and convert to base64 data URI.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 data URI string, or None if no path provided.
    """
    if image_path is None:
        return None

    # Load image
    image = Image.open(image_path)

    # Convert to bytes
    buffer = BytesIO()
    image_format = image.format or "PNG"
    image.save(buffer, format=image_format)
    image_bytes = buffer.getvalue()

    # Encode as base64
    base64_data = base64.b64encode(image_bytes).decode("utf-8")

    # Determine MIME type
    mime_type = f"image/{image_format.lower()}"

    # Return as data URI
    return f"data:{mime_type};base64,{base64_data}"


async def generate_image(args: argparse.Namespace) -> None:
    """Main generation logic.

    Args:
        args: Parsed command-line arguments
    """
    print(f"Loading model: {args.model}")

    # Step 1: Initialize pipeline configuration
    resolved_model_path = resolve_local_model_snapshot(args.model)
    if resolved_model_path != args.model:
        print(f"Resolved local model snapshot: {resolved_model_path}")

    model_kwargs: dict[str, Any] = {
        "model_path": resolved_model_path,
        "device_specs": [DeviceSpec.accelerator()],
    }
    if "ltx" in args.model.lower():
        # Public LTX checkpoints are fp32 safetensors. Request bf16 runtime
        # and allow fp32<->bf16 safetensors casting during model loading.
        model_kwargs["quantization_encoding"] = "bfloat16"
        model_kwargs[
            "allow_safetensors_weights_fp32_bf6_bidirectional_cast"
        ] = True
    config = PipelineConfig(
        model=MAXModelConfig(**model_kwargs),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        prefer_module_v3=config.runtime.prefer_module_v3,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, (
        "No matching diffusion architecture found for the provided model."
    )

    # Step 2: Initialize the tokenizer
    # The tokenizer handles prompt encoding and context preparation
    has_tokenizer_2 = False
    diffusers_config = config.model.diffusers_config
    max_length = args.max_length
    secondary_max_length = args.secondary_max_length
    if (
        max_length is None
        and diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and (components_config.get("tokenizer", None) is not None)
    ):
        max_length = components_config["tokenizer"]["config_dict"].get(
            "model_max_length", None
        )
        if arch.name == "Flux2Pipeline":
            max_length = 512
        print(f"Using max length: {max_length} for tokenizer")

    if (
        secondary_max_length is None
        and diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and (components_config.get("tokenizer_2", None) is not None)
    ):
        has_tokenizer_2 = True
        secondary_max_length = components_config["tokenizer_2"][
            "config_dict"
        ].get("model_max_length", None)
        print(
            f"Using secondary max length: {secondary_max_length} for tokenizer_2"
        )

    tokenizer = PixelGenerationTokenizer(
        model_path=args.model,
        pipeline_config=config,
        subfolder="tokenizer",  # Tokenizer is in a subfolder for diffusion models
        max_length=max_length,
        subfolder_2="tokenizer_2" if has_tokenizer_2 else None,
        secondary_max_length=secondary_max_length if has_tokenizer_2 else None,
    )

    # Step 3: Initialize the pipeline
    # The pipeline executes the diffusion model
    if not issubclass(arch.pipeline_model, DiffusionPipeline):
        raise TypeError(
            "Selected architecture does not implement DiffusionPipeline: "
            f"{arch.pipeline_model}"
        )
    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )

    print(f"Generating image for prompt: '{args.prompt}'")

    is_ltx = arch.name == "LTXPipeline"
    provider_options = ProviderOptions(
        image=ImageProviderOptions(
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ),
        video=(
            VideoProviderOptions(
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                steps=args.num_inference_steps,
                num_frames=args.num_frames,
                frames_per_second=args.frames_per_second,
                decode_timestep=args.decode_timestep,
                decode_noise_scale=args.decode_noise_scale,
                denoise_strength=args.denoise_strength,
                image_cond_noise_scale=args.image_cond_noise_scale,
            )
            if is_ltx
            else None
        ),
    )

    # Step 4: Create an OpenResponsesRequest
    # Load input image if provided and convert to data URI
    input_image_data_uri = load_image_as_data_uri(args.input_image)

    # Create request with structured message if image is provided
    if input_image_data_uri:
        # Image-to-image: Use structured message with InputImageContent + InputTextContent
        body = OpenResponsesRequestBody(
            model=args.model,
            input=[
                UserMessage(
                    role="user",
                    content=[
                        InputImageContent(
                            type="input_image",
                            image_url=input_image_data_uri,
                        ),
                        InputTextContent(
                            type="input_text",
                            text=args.prompt,
                        ),
                    ],
                )
            ],
            seed=args.seed,
            provider_options=provider_options,
        )
    else:
        # Text-to-image: Use simple string prompt
        body = OpenResponsesRequestBody(
            model=args.model,
            input=args.prompt,
            seed=args.seed,
            provider_options=provider_options,
        )

    request = OpenResponsesRequest(request_id=RequestID(), body=body)

    print(
        f"Parameters: steps={args.num_inference_steps}, guidance={args.guidance_scale}"
    )

    # Step 5: Create a PixelContext object from the request
    # The tokenizer handles prompt tokenization, timestep scheduling,
    # latent initialization, and all other preprocessing
    # Image is now extracted from the message content automatically
    context = await tokenizer.new_context(request)

    print(
        f"Context created: {context.height}x{context.width}, {context.num_inference_steps} steps"
    )

    # Step 6: Prepare inputs for the pipeline
    # Create a batch with a single context
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )

    # Step 6-1: Warmup — run before profiling or timed execution so that JIT
    # compilation completes and steady-state performance can be measured.
    if args.num_warmups > 0:
        body_warmup = OpenResponsesRequestBody(
            model=args.model,
            input=args.prompt,
            seed=args.seed,
            provider_options=provider_options,
        )
        request_warmup = OpenResponsesRequest(
            request_id=RequestID(), body=body_warmup
        )
        input_image = Image.open(args.input_image) if args.input_image else None
        context_warmup = await tokenizer.new_context(
            request_warmup, input_image=input_image
        )
        inputs_warmup = PixelGenerationInputs[PixelContext](
            batch={context_warmup.request_id: context_warmup}
        )
        for i in range(args.num_warmups):
            print(f"Running warmup {i + 1} of {args.num_warmups}")
            pipeline.execute(inputs_warmup)
        print("Warmup complete")

    # Step 7: Execute the pipeline
    print("Running diffusion model...")
    if args.save_packed_latents_npy:
        os.environ["MAX_LTX_SAVE_PACKED_LATENTS"] = args.save_packed_latents_npy
    if args.load_packed_latents_npy:
        os.environ["MAX_LTX_OVERRIDE_PACKED_LATENTS"] = args.load_packed_latents_npy

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
    output = outputs[context.request_id]
    output = await tokenizer.postprocess(output)

    # Check if generation completed successfully
    if not output.is_done:
        print(f"WARNING: Generation status: {output.final_status}")
        return

    print("Generation complete!")

    # Step 9: Extract and save generated media from output content
    if not output.output:
        print("ERROR: No media generated")
        return

    # Save each generated item (image/video)
    for idx, media_content in enumerate(output.output):
        # Determine output filename
        if len(output.output) > 1:
            # Multiple outputs: add index to filename
            base_name, ext = os.path.splitext(args.output)
            output_path = f"{base_name}_{idx}{ext}"
        else:
            output_path = args.output

        if isinstance(media_content, OutputImageContent):
            if media_content.image_data:
                save_image(media_content.image_data, output_path)
            elif media_content.image_url:
                print(f"Image available at URL: {media_content.image_url}")
            else:
                print("ERROR: No image data or URL in output")
        elif isinstance(media_content, OutputVideoContent):
            if media_content.video_data:
                save_video(media_content.video_data, output_path)
            elif media_content.video_url:
                print(f"Video available at URL: {media_content.video_url}")
            else:
                print("ERROR: No video data or URL in output")
        else:
            print(f"ERROR: Unsupported output content type: {type(media_content)}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the pixel generation example.

    Args:
        argv: Optional explicit list of argument strings. If None, arguments
            are read from sys.argv[1:].

    Returns:
        Process exit code. 0 indicates success.
    """
    apply_env_overrides()
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
