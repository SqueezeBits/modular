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

"""Generate reference .npy files from diffusers WanPipeline for quality comparison.

Usage:
    pip install diffusers transformers accelerate
    python wan_reference_dump.py --output-dir /tmp/wan_reference \
        --prompt "A cat walking" --num-frames 81 --num-inference-steps 50

Then compare against MAX pipeline outputs at each checkpoint:
    - ref_prompt_embeds.npy: text encoder output
    - ref_latents_step{N}.npy: latents at selected steps
    - ref_final_latents.npy: latents before VAE
    - ref_denorm_latents.npy: after denormalization
    - ref_vae_output.npy: final decoded video tensor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reference .npy files from diffusers WanPipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="HuggingFace model ID",
    )
    parser.add_argument("--prompt", type=str, default="A cat walking")
    parser.add_argument(
        "--negative-prompt", type=str, default="low quality"
    )
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/wan_reference",
        help="Directory to save .npy reference files",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=str,
        default="0,1,24,49",
        help="Comma-separated step indices to save latent checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_steps = set(int(s) for s in args.checkpoint_steps.split(","))

    # Import diffusers here so the script fails fast if not installed.
    from diffusers import WanPipeline  # type: ignore[import-untyped]

    print(f"Loading model: {args.model}")
    pipe = WanPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Hook into the pipeline to capture intermediates.
    saved_latents: dict[int, np.ndarray] = {}
    original_step = pipe.scheduler.step

    step_counter = [0]

    def patched_step(model_output, timestep, sample, **kwargs):
        result = original_step(model_output, timestep, sample, **kwargs)
        step_idx = step_counter[0]
        if step_idx in checkpoint_steps:
            if hasattr(result, "prev_sample"):
                latent = result.prev_sample
            else:
                latent = result
            saved_latents[step_idx] = (
                latent.detach().float().cpu().numpy()
            )
            print(f"  Saved latents at step {step_idx}")
        step_counter[0] += 1
        return result

    pipe.scheduler.step = patched_step

    # Capture prompt embeds by hooking into the text encoder call.
    prompt_embeds_holder: list[np.ndarray] = []
    original_encode = pipe.encode_prompt

    def patched_encode(*a, **kw):
        result = original_encode(*a, **kw)
        # Result is typically (prompt_embeds, negative_prompt_embeds)
        if isinstance(result, tuple) and len(result) >= 1:
            prompt_embeds_holder.append(
                result[0].detach().float().cpu().numpy()
            )
        return result

    pipe.encode_prompt = patched_encode

    print(f"Running inference with {args.num_inference_steps} steps...")
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    # Save prompt embeds.
    if prompt_embeds_holder:
        path = output_dir / "ref_prompt_embeds.npy"
        np.save(path, prompt_embeds_holder[0])
        print(f"Saved {path}")

    # Save latent checkpoints.
    for step_idx, latent_np in sorted(saved_latents.items()):
        path = output_dir / f"ref_latents_step{step_idx}.npy"
        np.save(path, latent_np)
        print(f"Saved {path}")

    # Save final video output.
    if hasattr(output, "frames") and output.frames is not None:
        frames = output.frames
        if isinstance(frames, list):
            # diffusers returns list of PIL images per batch
            print(
                f"Output is {len(frames)} batches of PIL frames "
                "(saving as list is not straightforward; skipping video .npy)"
            )
        elif isinstance(frames, torch.Tensor):
            video_np = frames.detach().float().cpu().numpy()
            path = output_dir / "ref_vae_output.npy"
            np.save(path, video_np)
            print(f"Saved {path} shape={video_np.shape}")
        elif isinstance(frames, np.ndarray):
            path = output_dir / "ref_vae_output.npy"
            np.save(path, frames)
            print(f"Saved {path} shape={frames.shape}")

    print(f"\nAll reference files saved to {output_dir}")
    print("Files:")
    for f in sorted(output_dir.glob("*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
