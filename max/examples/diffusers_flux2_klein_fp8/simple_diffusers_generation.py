"""Minimal diffusers smoke test for FLUX.2-Klein FP8.

Example:
  ./bazelw run //max/examples/diffusers_flux2_klein_fp8:simple_diffusers_generation -- \
    --model black-forest-labs/FLUX.2-klein-4B \
    --transformer-model black-forest-labs/FLUX.2-klein-4b-fp8 \
    --prompt "A cat holding a sign that says hello world" \
    --num-inference-steps 8 \
    --guidance-scale 1.0 \
    --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HF repo id or local diffusers-compatible base pipeline path.",
    )
    parser.add_argument(
        "--transformer-model",
        default="black-forest-labs/FLUX.2-klein-4b-fp8",
        help="HF repo id containing the flat FP8 transformer checkpoint.",
    )
    parser.add_argument(
        "--transformer-filename",
        default="flux-2-klein-4b-fp8.safetensors",
        help="Filename of the flat FP8 transformer checkpoint inside transformer-model.",
    )
    parser.add_argument(
        "--transformer-file",
        default=None,
        help="Optional local path to a flat FP8 transformer safetensors file.",
    )
    parser.add_argument(
        "--prompt",
        default="A cat holding a sign that says hello world",
    )
    parser.add_argument("--output", default="output_diffusers_flux2_klein_fp8.png")
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached model files.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(args.torch_dtype)

    transformer_path = (
        args.transformer_file
        if args.transformer_file is not None
        else hf_hub_download(
            repo_id=args.transformer_model,
            filename=args.transformer_filename,
            local_files_only=args.local_files_only,
        )
    )
    try:
        transformer = Flux2Transformer2DModel.from_single_file(
            transformer_path,
            config=args.model,
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        )
    except RuntimeError as exc:
        if "chunk expects at least a 1-dimensional tensor" in str(exc):
            raise RuntimeError(
                "Diffusers 0.36.0 cannot load the raw FLUX.2-klein-4b-fp8 "
                "single-file checkpoint because its Flux2 single-file loader "
                "treats legacy qkv scale tensors like fused qkv weights."
            ) from exc
        raise

    pipe = Flux2Pipeline.from_pretrained(
        args.model,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe = pipe.to(device)

    image = pipe(
        prompt=args.prompt,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    ).images[0]

    output_path = Path(args.output)
    image.save(output_path)
    print(output_path.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
