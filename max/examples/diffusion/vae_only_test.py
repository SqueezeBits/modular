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

"""VAE-only test: dump latents, then decode with MAX or diffusers VAE.

Three modes:
  dump   - Run diffusers WanPipeline, capture final latents before VAE,
           save as .npy (also saves diffusers VAE output for comparison).
  decode - Load .npy latents, run MAX VAE decoder only, save video.
  decode-diffusers - Load .npy latents, run diffusers VAE decoder, save video.

Usage:
    # Step 1: Dump latents from diffusers (requires diffusers+torch)
    python vae_only_test.py dump \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
        --negative-prompt "low quality" \
        --num-inference-steps 40 --guidance-scale 4.2 \
        --output-dir /tmp/wan_vae_test

    # Step 2: Decode with MAX VAE
    ./bazelw run //max/examples/diffusion:vae_only_test -- decode \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --latents-npy /tmp/wan_vae_test/final_latents.npy \
        --output /tmp/wan_vae_test/max_vae_output.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAE-only test for Wan pipeline debugging"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -- dump mode --
    dump_p = subparsers.add_parser(
        "dump", help="Run diffusers pipeline and save latents before VAE"
    )
    dump_p.add_argument(
        "--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    )
    dump_p.add_argument("--prompt", required=True)
    dump_p.add_argument("--negative-prompt", default="low quality")
    dump_p.add_argument("--num-frames", type=int, default=81)
    dump_p.add_argument("--height", type=int, default=480)
    dump_p.add_argument("--width", type=int, default=832)
    dump_p.add_argument("--num-inference-steps", type=int, default=40)
    dump_p.add_argument("--guidance-scale", type=float, default=4.2)
    dump_p.add_argument("--seed", type=int, default=42)
    dump_p.add_argument(
        "--output-dir", default="/tmp/wan_vae_test",
        help="Directory to save .npy files",
    )

    # -- decode mode --
    dec_p = subparsers.add_parser(
        "decode", help="Load .npy latents and decode with MAX VAE only"
    )
    dec_p.add_argument(
        "--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    )
    dec_p.add_argument(
        "--latents-npy", required=True,
        help="Path to the .npy file with final latents (before VAE)",
    )
    dec_p.add_argument(
        "--latents-format",
        choices=("auto", "raw", "denorm"),
        default="auto",
        help=(
            "Interpretation for input latents. "
            "'raw' means pre-denormalization latents, "
            "'denorm' means VAE input latents, "
            "'auto' infers from filename."
        ),
    )
    dec_p.add_argument("--num-frames", type=int, default=81)
    dec_p.add_argument("--height", type=int, default=480)
    dec_p.add_argument("--width", type=int, default=832)
    dec_p.add_argument("--fps", type=int, default=16)
    dec_p.add_argument(
        "--output", default="max_vae_output.mp4",
        help="Output video file",
    )

    # -- decode-diffusers mode --
    diff_dec_p = subparsers.add_parser(
        "decode-diffusers",
        help="Load .npy latents and decode with diffusers VAE only",
    )
    diff_dec_p.add_argument(
        "--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    )
    diff_dec_p.add_argument(
        "--latents-npy",
        required=True,
        help="Path to the .npy file with final latents (before VAE)",
    )
    diff_dec_p.add_argument(
        "--latents-format",
        choices=("auto", "raw", "denorm"),
        default="auto",
        help=(
            "Interpretation for input latents. "
            "'raw' means pre-denormalization latents, "
            "'denorm' means VAE input latents, "
            "'auto' infers from filename."
        ),
    )
    diff_dec_p.add_argument("--num-frames", type=int, default=81)
    diff_dec_p.add_argument("--height", type=int, default=480)
    diff_dec_p.add_argument("--width", type=int, default=832)
    diff_dec_p.add_argument("--fps", type=int, default=16)
    diff_dec_p.add_argument(
        "--output",
        default="diffusers_vae_output.mp4",
        help="Output video file",
    )
    diff_dec_p.add_argument(
        "--save-decoded-npy",
        action="store_true",
        help="Save decoded tensor as .npy next to output video.",
    )

    return parser.parse_args()


# ---- dump mode ----


def run_dump(args: argparse.Namespace) -> None:
    import torch
    from diffusers import WanPipeline

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading diffusers model: {args.model}")
    pipe = WanPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Hook into the VAE decode to capture input latents.
    original_vae_decode = pipe.vae.decode

    captured_latents: list[np.ndarray] = []
    captured_vae_output: list[np.ndarray] = []

    def hooked_vae_decode(z, *a, **kw):
        captured_latents.append(z.detach().float().cpu().numpy())
        print(f"  [hook] Captured latents before VAE: shape={z.shape}, dtype={z.dtype}")
        result = original_vae_decode(z, *a, **kw)
        # Capture the decoded output too
        if isinstance(result, tuple):
            decoded = result[0]
        elif hasattr(result, "sample"):
            decoded = result.sample
        else:
            decoded = result
        captured_vae_output.append(decoded.detach().float().cpu().numpy())
        print(f"  [hook] Captured VAE output: shape={decoded.shape}")
        return result

    pipe.vae.decode = hooked_vae_decode

    # Also capture the raw denoised latents BEFORE denormalization.
    # Hook into the pipeline's _decode_latents or the denorm step.
    # In diffusers WanPipeline, latents are denormalized then passed to VAE.
    # We'll capture the raw latents by hooking the scheduler's final output.
    raw_latents_holder: list[np.ndarray] = []

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

    # Save captured latents (these are the denormalized latents that went into VAE)
    if captured_latents:
        path = output_dir / "final_latents_denorm.npy"
        np.save(path, captured_latents[0])
        print(f"Saved denormalized latents (VAE input): {path}  shape={captured_latents[0].shape}")

    if captured_vae_output:
        path = output_dir / "diffusers_vae_output.npy"
        np.save(path, captured_vae_output[0])
        print(f"Saved diffusers VAE output: {path}  shape={captured_vae_output[0].shape}")

    # Also save the raw latents before denorm by recomputing them.
    # In diffusers, denorm is: latents = latents * std + mean
    # So raw = (denorm - mean) / std
    if captured_latents:
        vae_config = pipe.vae.config
        if hasattr(vae_config, "latents_mean") and vae_config.latents_mean is not None:
            mean = np.array(vae_config.latents_mean, dtype=np.float32).reshape(1, -1, 1, 1, 1)
            std = np.array(vae_config.latents_std, dtype=np.float32).reshape(1, -1, 1, 1, 1)
            raw = (captured_latents[0] - mean) / std
            path = output_dir / "final_latents_raw.npy"
            np.save(path, raw)
            print(f"Saved raw latents (before denorm): {path}  shape={raw.shape}")
        else:
            # No denorm config; the captured latents ARE the raw latents
            path = output_dir / "final_latents_raw.npy"
            np.save(path, captured_latents[0])
            print(f"Saved raw latents: {path}  shape={captured_latents[0].shape}")

    # Save video from diffusers for comparison
    if hasattr(output, "frames") and output.frames is not None:
        frames = output.frames
        if isinstance(frames, list) and len(frames) > 0:
            # diffusers returns list of list of PIL images
            pil_frames = frames[0] if isinstance(frames[0], list) else frames
            frame_arrays = [np.array(f) for f in pil_frames]
            if frame_arrays:
                video_np = np.stack(frame_arrays)
                path = output_dir / "diffusers_video_frames.npy"
                np.save(path, video_np)
                print(f"Saved diffusers video frames: {path}  shape={video_np.shape}")

                # Also save as mp4
                _save_mp4(frame_arrays, output_dir / "diffusers_output.mp4", args.height, args.width, fps=16)

    print(f"\nAll files saved to {output_dir}")
    for f in sorted(output_dir.glob("*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: shape={arr.shape} dtype={arr.dtype}")
        print(f"    min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} std={arr.std():.4f}")


# ---- decode mode ----


def _resolve_is_denorm(
    latents_npy_path: str, latents_format: str,
) -> bool:
    if latents_format == "denorm":
        return True
    if latents_format == "raw":
        return False
    return "denorm" in latents_npy_path.lower()


def _print_array_stats(name: str, arr: np.ndarray) -> None:
    print(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.4f}, max={arr.max():.4f}, "
        f"mean={arr.mean():.4f}, std={arr.std():.4f}"
    )


def run_decode(args: argparse.Namespace) -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s %(message)s')

    from typing import cast
    from max.driver import CPU, DeviceSpec, load_devices
    from max.dtype import DType
    from max.engine import InferenceSession
    from max.experimental.tensor import Tensor
    from max.graph.weights import load_weights
    from max.interfaces import PipelineTask
    from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
    from max.pipelines.lib.interfaces import DiffusionPipeline
    from max.pipelines.lib.pipeline_variants.pixel_generation import (
        PixelGenerationPipeline,
    )
    from max.pipelines.lib.pipeline_variants.utils import get_weight_paths
    from max.pipelines.core import PixelContext

    print(f"Loading latents from: {args.latents_npy}")
    latents_np = np.load(args.latents_npy).astype(np.float32)
    _print_array_stats("Latents", latents_np)

    is_denorm = _resolve_is_denorm(args.latents_npy, args.latents_format)
    print(
        "  Latents interpreted as "
        + ("denormalized" if is_denorm else "raw (will denormalize)")
    )

    # Use the full pipeline infrastructure to load the model (including VAE)
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=args.model,
            device_specs=[DeviceSpec.accelerator()],
        ),
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None

    # Initialize full pipeline (loads all sub-models including VAE)
    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    devices = load_devices(config.model.device_specs)
    session = InferenceSession(devices=devices)
    config.configure_session(session)
    weight_paths = get_weight_paths(config.model)

    print("Loading pipeline (for VAE)...")
    wan_pipeline = pipeline_model(
        pipeline_config=config,
        session=session,
        devices=devices,
        weight_paths=weight_paths,
    )

    device = devices[0]
    vae = wan_pipeline.vae  # type: ignore[attr-defined]

    # If latents are raw (not denormalized), apply denormalization
    if not is_denorm:
        z_dim = int(vae.config.z_dim)
        mean = np.array(vae.config.latents_mean, dtype=np.float32).reshape(1, z_dim, 1, 1, 1)
        std = np.array(vae.config.latents_std, dtype=np.float32).reshape(1, z_dim, 1, 1, 1)
        latents_np = latents_np * std + mean
        _print_array_stats("After denorm", latents_np)

    # Move latents to GPU as bfloat16
    latents_t = Tensor.from_dlpack(latents_np).cast(DType.bfloat16).to(device)
    print(f"Latents tensor: shape={latents_t.shape}, dtype={latents_t.dtype}, device={latents_t.device}")

    # Decode with MAX VAE
    print("Running MAX VAE decode...")
    decoded = vae.decode_5d(latents_t)
    print(f"Decoded shape: {decoded.shape}")

    # Crop to target dimensions
    decoded = decoded[
        :, :, : args.num_frames, : args.height, : args.width
    ]
    print(f"Cropped shape: {decoded.shape}")

    # Convert to numpy
    decoded_np = np.from_dlpack(decoded.cast(DType.float32).to(CPU()))
    _print_array_stats("Decoded numpy", decoded_np)

    # Save raw decoded output for comparison
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "max_vae_output.npy"
    np.save(npy_path, decoded_np)
    print(f"Saved MAX VAE output: {npy_path}")

    # Compare with diffusers if available
    diffusers_npy = output_dir / "diffusers_vae_output.npy"
    if diffusers_npy.exists():
        diff_out = np.load(diffusers_npy)
        diff_out = diff_out[
            :, :, : args.num_frames, : args.height, : args.width
        ]
        abs_diff = np.abs(decoded_np - diff_out)
        print(f"\n=== Comparison with diffusers VAE ===")
        print(f"  Diffusers shape: {diff_out.shape}")
        print(f"  MAX shape: {decoded_np.shape}")
        print(f"  Abs diff: mean={abs_diff.mean():.6f} max={abs_diff.max():.6f} std={abs_diff.std():.6f}")
        cos_sim = np.sum(decoded_np * diff_out) / (
            np.linalg.norm(decoded_np) * np.linalg.norm(diff_out) + 1e-8
        )
        print(f"  Cosine similarity: {cos_sim:.6f}")

    # Convert decoded video to frames and save mp4
    # decoded_np shape: [B, C, T, H, W] -> [T, H, W, C]
    video = decoded_np[0]  # [C, T, H, W]
    video = np.transpose(video, (1, 2, 3, 0))  # [T, H, W, C]
    video = np.clip(video, 0, 1)
    video = (video * 255).astype(np.uint8)
    frames = [video[t] for t in range(video.shape[0])]

    print(f"Saving {len(frames)} frames as video to {args.output}")
    _save_mp4(frames, args.output, args.height, args.width, args.fps)


def run_decode_diffusers(args: argparse.Namespace) -> None:
    import torch
    from diffusers import AutoencoderKLWan

    print(f"Loading latents from: {args.latents_npy}")
    latents_np = np.load(args.latents_npy).astype(np.float32)
    _print_array_stats("Latents", latents_np)

    is_denorm = _resolve_is_denorm(args.latents_npy, args.latents_format)
    print(
        "  Latents interpreted as "
        + ("denormalized" if is_denorm else "raw (will denormalize)")
    )

    print(f"Loading diffusers VAE from: {args.model}")
    vae = AutoencoderKLWan.from_pretrained(
        args.model,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to("cuda")

    if not is_denorm:
        z_dim = int(vae.config.z_dim)
        mean = np.array(vae.config.latents_mean, dtype=np.float32).reshape(
            1, z_dim, 1, 1, 1
        )
        std = np.array(vae.config.latents_std, dtype=np.float32).reshape(
            1, z_dim, 1, 1, 1
        )
        latents_np = latents_np * std + mean
        _print_array_stats("After denorm", latents_np)

    latents_t = torch.from_numpy(latents_np).to(
        device="cuda", dtype=vae.dtype
    )
    print(
        f"Latents tensor: shape={tuple(latents_t.shape)}, "
        f"dtype={latents_t.dtype}, device={latents_t.device}"
    )

    print("Running diffusers VAE decode...")
    with torch.inference_mode():
        decoded_out = vae.decode(latents_t)
        if isinstance(decoded_out, tuple):
            decoded_t = decoded_out[0]
        elif hasattr(decoded_out, "sample"):
            decoded_t = decoded_out.sample
        else:
            decoded_t = decoded_out

    decoded_t = decoded_t[
        :, :, : args.num_frames, : args.height, : args.width
    ]
    decoded_np = decoded_t.float().cpu().numpy()
    _print_array_stats("Decoded numpy", decoded_np)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_decoded_npy:
        npy_path = output_dir / "diffusers_vae_from_npy.npy"
        np.save(npy_path, decoded_np)
        print(f"Saved diffusers decoded output: {npy_path}")

    video = decoded_np[0]  # [C, T, H, W]
    video = np.transpose(video, (1, 2, 3, 0))  # [T, H, W, C]
    video = np.clip(video, 0, 1)
    video = (video * 255).astype(np.uint8)
    frames = [video[t] for t in range(video.shape[0])]

    print(f"Saving {len(frames)} frames as video to {args.output}")
    _save_mp4(frames, args.output, args.height, args.width, args.fps)


# ---- util ----


def _save_mp4(
    frames: list[np.ndarray],
    output_path: str | Path,
    height: int,
    width: int,
    fps: int = 16,
) -> None:
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "18",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = b"".join(f.astype(np.uint8).tobytes() for f in frames)
    _, stderr = proc.communicate(input=raw)
    if proc.returncode != 0:
        print(f"WARNING: ffmpeg returned {proc.returncode}: {stderr.decode()}")
    else:
        print(f"Video saved to: {output_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "dump":
        run_dump(args)
    elif args.mode == "decode":
        run_decode(args)
    elif args.mode == "decode-diffusers":
        run_decode_diffusers(args)


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)
    main()
