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

"""Run Wan Animate motion transfer with the diffusers library.

This script executes the Wan Animate pipeline in motion-transfer mode using:

- a reference character image
- a preprocessed pose video
- a preprocessed face video

When ``--preprocess`` is given together with ``--driving-video``, the script
automatically extracts skeleton pose renders and cropped face frames from the
raw driving video using DWPose (via ONNX Runtime).

The Wan Animate diffusers API uses ``mode="animate"`` for motion transfer.
There is no separate ``"move"`` mode in the public pipeline API.

Example (with preprocessing):
    python max/examples/diffusion/wan_animate_move_diffusers.py \
        --image assets/character.png \
        --driving-video assets/motion.mp4 \
        --preprocess \
        --prompt "A character dancing under stage lights." \
        --output wan_animate.mp4

Example (pre-processed inputs):
    python max/examples/diffusion/wan_animate_move_diffusers.py \
        --image assets/character.png \
        --pose-video assets/pose.mp4 \
        --face-video assets/face.mp4 \
        --prompt "A character dancing under stage lights." \
        --output wan_animate.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Allow importing sibling modules when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanAnimatePipeline
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
from wan_animate_preprocess import preprocess_driving_video, save_preprocessed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute the Wan Animate motion-transfer pipeline."
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-Animate-14B-Diffusers",
        help="Model id or local path for the Wan Animate diffusers pipeline.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Reference character image.",
    )
    parser.add_argument(
        "--driving-video",
        default=None,
        help="Raw driving video. Used with --preprocess to auto-extract pose and face.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess --driving-video to extract pose skeleton and face crops.",
    )
    parser.add_argument(
        "--save-preprocessed",
        default=None,
        help="Directory to save preprocessed pose/face videos for reuse.",
    )
    parser.add_argument(
        "--pose-video",
        default=None,
        help="Preprocessed pose video that provides body motion.",
    )
    parser.add_argument(
        "--face-video",
        default=None,
        help="Preprocessed face video that provides facial motion.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt that describes the generated video.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt.",
    )
    parser.add_argument(
        "--output",
        default="wan_animate_output.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--mode",
        choices=["animate", "replace"],
        default="animate",
        help="Wan Animate generation mode.",
    )
    parser.add_argument(
        "--background-video",
        default=None,
        help="Background video required for replacement mode.",
    )
    parser.add_argument(
        "--mask-video",
        default=None,
        help="Mask video required for replacement mode.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height. If unset, a compatible size is derived.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width. If unset, a compatible size is derived.",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=720 * 1280,
        help="Target pixel budget used when height and width are omitted.",
    )
    parser.add_argument(
        "--segment-frame-length",
        type=int,
        default=77,
        help="Frames per segment processed by the pipeline.",
    )
    parser.add_argument(
        "--prev-segment-conditioning-frames",
        type=int,
        default=1,
        help="Frames reused for temporal conditioning between segments.",
    )
    parser.add_argument(
        "--motion-encode-batch-size",
        type=int,
        default=None,
        help="Optional batch size for motion encoding.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=20,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale. Wan Animate often uses 1.0.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for the exported mp4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Set to a negative value to disable seeding.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--transformer-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype used for the transformer weights.",
    )
    parser.add_argument(
        "--vae-dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Torch dtype used for the VAE.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable sequential CPU offload instead of moving the full model.",
    )
    parser.add_argument(
        "--dump-intermediates",
        default=None,
        help="Directory to save all intermediate tensors for parity testing.",
    )

    args = parser.parse_args(argv)

    if args.preprocess:
        if args.driving_video is None:
            parser.error(
                "--driving-video is required when --preprocess is set."
            )
        if args.pose_video is not None or args.face_video is not None:
            parser.error(
                "--pose-video and --face-video cannot be used with --preprocess."
            )
    else:
        if args.pose_video is None or args.face_video is None:
            parser.error(
                "--pose-video and --face-video are required when "
                "--preprocess is not set."
            )

    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be provided together.")

    if args.mode == "replace":
        if args.background_video is None:
            parser.error("--background-video is required for --mode replace.")
        if args.mask_video is None:
            parser.error("--mask-video is required for --mode replace.")

    if args.height is not None and args.height <= 0:
        parser.error("--height must be positive.")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be positive.")
    if args.max_area <= 0:
        parser.error("--max-area must be positive.")
    if args.segment_frame_length <= 0:
        parser.error("--segment-frame-length must be positive.")
    if args.prev_segment_conditioning_frames < 0:
        parser.error("--prev-segment-conditioning-frames must be non-negative.")
    if args.motion_encode_batch_size is not None:
        if args.motion_encode_batch_size <= 0:
            parser.error("--motion-encode-batch-size must be positive.")
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be positive.")
    if args.guidance_scale <= 0.0:
        parser.error("--guidance-scale must be positive.")
    if args.fps <= 0:
        parser.error("--fps must be positive.")

    return args


def parse_torch_dtype(name: str) -> torch.dtype:
    """Convert a CLI dtype name into a ``torch.dtype``."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def validate_device(device: str) -> None:
    """Raise a helpful error if the requested device is unavailable."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no CUDA device is available. "
            "Pass --device cpu to run on CPU."
        )


def _spatial_multiple(pipe: WanAnimatePipeline) -> int:
    """Return the spatial multiple required by the VAE and transformer."""
    patch_size = pipe.transformer.config.patch_size
    patch_width = patch_size if isinstance(patch_size, int) else patch_size[1]
    return pipe.vae_scale_factor_spatial * patch_width


def _round_down(value: int, multiple: int) -> int:
    """Round ``value`` down to the nearest positive multiple."""
    return max(multiple, value // multiple * multiple)


def resolve_output_size(
    image: Image.Image,
    pipe: WanAnimatePipeline,
    requested_height: int | None,
    requested_width: int | None,
    max_area: int,
) -> tuple[Image.Image, int, int]:
    """Resize the image to a shape compatible with Wan Animate."""
    spatial_multiple = _spatial_multiple(pipe)

    if requested_height is not None and requested_width is not None:
        height = _round_down(requested_height, spatial_multiple)
        width = _round_down(requested_width, spatial_multiple)
    else:
        aspect_ratio = image.height / image.width
        height = _round_down(
            round(math.sqrt(max_area * aspect_ratio)),
            spatial_multiple,
        )
        width = _round_down(
            round(math.sqrt(max_area / aspect_ratio)),
            spatial_multiple,
        )

    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    return resized, height, width


def build_generator(device: str, seed: int) -> torch.Generator | None:
    """Create a seeded generator when the device supports it."""
    if seed < 0:
        return None

    if device.startswith("cuda"):
        return torch.Generator(device=device).manual_seed(seed)
    if device == "cpu":
        return torch.Generator(device="cpu").manual_seed(seed)
    return None


def load_pipeline(
    model_id: str,
    device: str,
    transformer_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    cpu_offload: bool,
) -> WanAnimatePipeline:
    """Load and place the Wan Animate pipeline."""
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=vae_dtype,
    )
    pipe = WanAnimatePipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=transformer_dtype,
    )

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe




class IntermediateDumper:
    """Captures and saves intermediate tensors from the diffusers pipeline."""

    def __init__(self, dump_dir: str):
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self._seg_idx = 0
        self._step_idx = 0
        self._motion_vectors: list[torch.Tensor] = []
        self._face_emb: list[torch.Tensor] = []
        self._hooks: list[Any] = []
        self._final_latents_per_seg: dict[int, torch.Tensor] = {}
        self._prepare_latents_call_count = 0

    def save(self, name: str, tensor: torch.Tensor | np.ndarray) -> None:
        """Save a tensor as .npy (float32)."""
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().float().cpu().numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float32)
        path = self.dump_dir / f"{name}.npy"
        np.save(str(path), arr)
        print(f"  [dump] {name}: {arr.shape} → {path}")

    def save_config(self, args: argparse.Namespace, height: int, width: int,
                    num_segments: int) -> None:
        """Save generation config as JSON."""
        config = {
            "model": args.model,
            "prompt": args.prompt,
            "height": height,
            "width": width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "segment_frame_length": args.segment_frame_length,
            "prev_segment_conditioning_frames": (
                args.prev_segment_conditioning_frames
            ),
            "mode": args.mode,
            "num_segments": num_segments,
            "transformer_dtype": args.transformer_dtype,
            "vae_dtype": args.vae_dtype,
        }
        path = self.dump_dir / "config.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  [dump] config → {path}")

    def patch_pipeline(self, pipe: WanAnimatePipeline) -> None:
        """Monkey-patch pipeline methods to capture intermediates."""
        dumper = self

        # Patch prepare_latents to save noise per segment.
        # This is called once per segment — use it to track segment index.
        orig_prepare_latents = pipe.prepare_latents
        dumper._prepare_latents_call_count = 0

        def patched_prepare_latents(*a, **kw):
            dumper._seg_idx = dumper._prepare_latents_call_count
            dumper._motion_vectors.clear()
            dumper._face_emb.clear()
            result = orig_prepare_latents(*a, **kw)
            dumper.save(f"noise_seg{dumper._seg_idx}", result)
            dumper._prepare_latents_call_count += 1
            return result

        pipe.prepare_latents = patched_prepare_latents

        # Patch prepare_reference_image_latents (called once)
        orig_prepare_ref = pipe.prepare_reference_image_latents

        def patched_prepare_ref(*a, **kw):
            result = orig_prepare_ref(*a, **kw)
            dumper.save("ref_image_latents", result)
            return result

        pipe.prepare_reference_image_latents = patched_prepare_ref

        # Patch prepare_pose_latents (called once per segment)
        orig_prepare_pose = pipe.prepare_pose_latents

        def patched_prepare_pose(*a, **kw):
            result = orig_prepare_pose(*a, **kw)
            dumper.save(f"pose_latents_seg{dumper._seg_idx}", result)
            return result

        pipe.prepare_pose_latents = patched_prepare_pose

        # Patch prepare_prev_segment_cond_latents (called once per segment)
        orig_prepare_prev = pipe.prepare_prev_segment_cond_latents

        def patched_prepare_prev(*a, **kw):
            result = orig_prepare_prev(*a, **kw)
            dumper.save(f"prev_cond_seg{dumper._seg_idx}", result)
            return result

        pipe.prepare_prev_segment_cond_latents = patched_prepare_prev

        # Patch encode_prompt to save text embeddings (called once)
        orig_encode_prompt = pipe.encode_prompt

        def patched_encode_prompt(*a, **kw):
            result = orig_encode_prompt(*a, **kw)
            prompt_embeds, negative_prompt_embeds = result
            dumper.save("prompt_embeds", prompt_embeds)
            if negative_prompt_embeds is not None:
                dumper.save("negative_prompt_embeds", negative_prompt_embeds)
            return result

        pipe.encode_prompt = patched_encode_prompt

        # Patch encode_image to save CLIP features (called once)
        orig_encode_image = pipe.encode_image

        def patched_encode_image(*a, **kw):
            result = orig_encode_image(*a, **kw)
            dumper.save("clip_features", result)
            return result

        pipe.encode_image = patched_encode_image

        # Hook motion_encoder to capture per-batch motion vectors
        transformer = pipe.transformer
        if hasattr(transformer, "motion_encoder"):

            def motion_hook(module, input, output):
                dumper._motion_vectors.append(output.detach().cpu())

            h = transformer.motion_encoder.register_forward_hook(
                motion_hook
            )
            self._hooks.append(h)

        # Hook face_encoder to capture face embeddings
        if hasattr(transformer, "face_encoder"):

            def face_hook(module, input, output):
                dumper._face_emb.append(output.detach().cpu())

            h = transformer.face_encoder.register_forward_hook(face_hook)
            self._hooks.append(h)

    def make_step_callback(self):
        """Return a callback that saves latents after step 0 and the final
        step of each segment."""
        dumper = self

        def callback(pipe, step_idx, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            if step_idx == 0:
                dumper.save(
                    f"latents_after_step0_seg{dumper._seg_idx}",
                    latents,
                )
            # Always update — the last call per segment gives final latents.
            dumper._final_latents_per_seg[dumper._seg_idx] = (
                latents.detach().cpu().clone()
            )

            # Save motion/face from the first denoising step (the
            # transformer is called every step, but motion/face encoding
            # only happens on the first call per segment in diffusers).
            if step_idx == 0:
                if dumper._motion_vectors:
                    all_motion = torch.cat(dumper._motion_vectors, dim=0)
                    dumper.save(
                        f"motion_vectors_seg{dumper._seg_idx}", all_motion
                    )
                if dumper._face_emb:
                    dumper.save(
                        f"face_emb_seg{dumper._seg_idx}",
                        dumper._face_emb[-1],
                    )
            return callback_kwargs

        return callback

    def save_final_latents(self) -> None:
        """Save final latents for all segments (captured via callback)."""
        for seg_idx, latents in self._final_latents_per_seg.items():
            self.save(f"final_latents_seg{seg_idx}", latents)

    def cleanup(self) -> None:
        """Remove hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def run(args: argparse.Namespace) -> Path:
    """Run Wan Animate and write the generated mp4 to disk."""
    validate_device(args.device)

    transformer_dtype = parse_torch_dtype(args.transformer_dtype)
    vae_dtype = parse_torch_dtype(args.vae_dtype)
    pipe = load_pipeline(
        model_id=args.model,
        device=args.device,
        transformer_dtype=transformer_dtype,
        vae_dtype=vae_dtype,
        cpu_offload=args.cpu_offload,
    )

    image = load_image(args.image)
    if args.preprocess:
        print("Preprocessing driving video...")
        raw_frames = load_video(args.driving_video)
        pose_video, face_video = preprocess_driving_video(
            raw_frames, device=args.device
        )
        if args.save_preprocessed:
            save_preprocessed(
                pose_video, face_video, args.save_preprocessed
            )
        print(
            f"Preprocessing complete: {len(pose_video)} pose frames, "
            f"{len(face_video)} face frames."
        )
    else:
        pose_video = load_video(args.pose_video)
        face_video = load_video(args.face_video)
    image, height, width = resolve_output_size(
        image=image,
        pipe=pipe,
        requested_height=args.height,
        requested_width=args.width,
        max_area=args.max_area,
    )

    generator = build_generator(args.device, args.seed)

    # Set up intermediate tensor dumper if requested.
    dumper = None
    if args.dump_intermediates:
        dumper = IntermediateDumper(args.dump_intermediates)
        dumper.patch_pipeline(pipe)
        print(f"Dumping intermediates to {args.dump_intermediates}")

    pipe_kwargs: dict[str, Any] = {
        "image": image,
        "pose_video": pose_video,
        "face_video": face_video,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": height,
        "width": width,
        "segment_frame_length": args.segment_frame_length,
        "num_inference_steps": args.num_inference_steps,
        "mode": args.mode,
        "prev_segment_conditioning_frames": (
            args.prev_segment_conditioning_frames
        ),
        "guidance_scale": args.guidance_scale,
    }
    if generator is not None:
        pipe_kwargs["generator"] = generator
    if args.motion_encode_batch_size is not None:
        pipe_kwargs["motion_encode_batch_size"] = args.motion_encode_batch_size
    if args.mode == "replace":
        pipe_kwargs["background_video"] = load_video(args.background_video)
        pipe_kwargs["mask_video"] = load_video(args.mask_video)
    if dumper is not None:
        pipe_kwargs["callback_on_step_end"] = dumper.make_step_callback()
        pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    frames = pipe(**pipe_kwargs).frames[0]

    if dumper is not None:
        # Compute num_segments for config saving.
        cond_frames = len(pose_video)
        eff_seg = args.segment_frame_length - (
            args.prev_segment_conditioning_frames
        )
        last_seg = (
            cond_frames - args.prev_segment_conditioning_frames
        ) % eff_seg
        padding = 0 if last_seg == 0 else eff_seg - last_seg
        num_segments = (cond_frames + padding) // eff_seg

        dumper.save_config(args, height, width, num_segments)
        dumper.save_final_latents()
        dumper.cleanup()
        print(
            f"Saved {len(list(Path(args.dump_intermediates).glob('*.npy')))} "
            f"tensor files to {args.dump_intermediates}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=args.fps)
    return output_path


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    output_path = run(args)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
