# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import argparse
import os
import time
from pathlib import Path

from max.dtype import DType
from max.entrypoints.diffusion import DiffusionPipeline
from max.graph import DeviceRef
from max.pipelines import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flux inference")
    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Model ID from HuggingFace Hub (default: FLUX.1-schnell)",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="max",
        choices=["max", "torch"],
        help="Framework to use for inference (default: max)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat holding a sign that says hello world",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flux_output.png",
        help="Output image path",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps (default: 4 for schnell)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale (default: 0.0 for schnell)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images to generate per prompt (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for inference",
    )
    parser.add_argument(
        "--use-torch-randn",
        action="store_true",
        help="Use torch's random normal generation for latent initialization.",
    )

    args = parser.parse_args()

    if args.framework == "torch":
        import torch
        from diffusers import FluxPipeline

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[args.dtype]

        pipe = FluxPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
        )
        pipe = pipe.to(args.device)

        # Enable memory optimizations if on CUDA
        if args.device == "cuda":
            # pipe.enable_model_cpu_offload()  # Use this if you have limited VRAM
            pipe.enable_attention_slicing()
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
    else:
        max_device = (
            DeviceRef.GPU() if args.device == "cuda" else DeviceRef.CPU()
        )
        max_dtype = (
            DType.bfloat16
            if args.dtype == "bfloat16"
            else DType.float16
            if args.dtype == "float16"
            else DType.float32
        )
        pipeline_config = PipelineConfig(model_path=args.model_id)
        pipe = DiffusionPipeline(pipeline_config)
        if args.use_torch_randn:
            os.environ["USE_TORCH_RANDN"] = "1"
            os.environ["SEED"] = str(args.seed)

    print(f"\nPrompt: {args.prompt}")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Batch size: {args.batch_size}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")

    print("\nRunning warmup...")
    result = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=3,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.batch_size,
    )
    # torch.cuda.synchronize()

    print(f"\nGenerating {'images' if args.batch_size > 1 else 'image'}...")
    time_start = time.time()
    # Run inference
    if args.framework == "max":
        result = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.batch_size,
        )
    else:
        result = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.batch_size,
            generator=generator,
        )

    images = result.images
    # torch.cuda.synchronize()
    time_end = time.time()

    # Save the output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.batch_size == 1:
        images[0].save(output_path)
        print(f"\n✓ Image saved to: {output_path}")
        print(f"  Size: {images[0].size}")
    else:
        stem = output_path.stem
        suffix = output_path.suffix
        for i, image in enumerate(images):
            numbered_path = output_path.parent / f"{stem}_{i + 1}{suffix}"
            image.save(numbered_path)
        print(f"\n✓ {len(images)} images saved to: {output_path.parent}")
        print(f"  Filenames: {stem}_1{suffix} to {stem}_{len(images)}{suffix}")
        print(f"  Size: {images[0].size}")

    print(f"  Inference time: {time_end - time_start:.3f} seconds")


if __name__ == "__main__":
    main()
