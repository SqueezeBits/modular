#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""Wan T2V/I2V performance comparison: diffusers (torch.compile) vs MAX.

Runs at 480p and 720p, with and without LoRA, prints a summary table.
Videos are saved to --output-dir for quality comparison.

Usage:
    ./bazelw run //max/examples/diffusion:wan_comparison
    ./bazelw run //max/examples/diffusion:wan_comparison -- --resolution 720p --variant lora
    ./bazelw run //max/examples/diffusion:wan_comparison -- --runner max --mode t2v
    ./bazelw run //max/examples/diffusion:wan_comparison -- --target wan2.2-t2v-a14b wan2.2-i2v-a14b
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# diffusers' Wan pipeline requires ftfy. Stub it if unavailable.
try:
    import ftfy  # type: ignore[import-not-found] # noqa: F401
except ModuleNotFoundError:
    import builtins
    import importlib
    import importlib.util
    import types as _types

    _ftfy = _types.ModuleType("ftfy")
    _ftfy.fix_text = lambda t: t  # type: ignore[attr-defined]
    _ftfy.__version__ = "0.0.0"  # type: ignore[attr-defined]
    _ftfy.__spec__ = importlib.util.spec_from_loader("ftfy", loader=None)
    sys.modules["ftfy"] = _ftfy
    builtins.ftfy = _ftfy  # type: ignore[attr-defined]

import numpy as np

T2V_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves "
    "fight intensely on a spotlighted stage."
)
I2V_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on "
    "a surfboard. The fluffy-furred feline gazes directly at the camera "
    "with a relaxed expression."
)


# ── Target configurations ────────────────────────────────────


@dataclass
class WanTarget:
    name: str  # e.g. "wan2.2-ti2v-5b-turbo"
    repo_id: str  # HuggingFace repo
    modes: list[str]  # ["t2v"], ["i2v"], or ["t2v", "i2v"]
    max_pixels: tuple[int, int]  # (max_width, max_height)
    max_frames: int
    default_steps: int
    default_guidance: float
    default_guidance_2: float | None  # None for non-MoE models
    # mode -> {repo, subfolder, weight_name?} or {repo, filename?}
    lora: dict[str, dict[str, str]] | None = None
    lora_steps: int = 4
    lora_guidance: float = 1.0
    lora_guidance_2: float = 1.0
    encoding: str | None = None  # override quantization encoding


TARGETS: dict[str, WanTarget] = {
    "wan2.2-t2v-a14b": WanTarget(
        name="wan2.2-t2v-a14b",
        repo_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        modes=["t2v"],
        max_pixels=(1280, 720),
        max_frames=81,
        default_steps=40,
        default_guidance=3.0,
        default_guidance_2=4.0,
        lora={
            "t2v": {
                "repo": "lightx2v/Wan2.2-Lightning",
                "subfolder": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0",
            },
        },
    ),
    "wan2.2-i2v-a14b": WanTarget(
        name="wan2.2-i2v-a14b",
        repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        modes=["i2v"],
        max_pixels=(1280, 720),
        max_frames=81,
        default_steps=40,
        default_guidance=3.0,
        default_guidance_2=4.0,
        lora={
            "i2v": {
                "repo": "lightx2v/Wan2.2-Lightning",
                "subfolder": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
            },
        },
    ),
    "wan2.2-ti2v-5b-turbo": WanTarget(
        name="wan2.2-ti2v-5b-turbo",
        repo_id="yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers",
        modes=["t2v", "i2v"],
        max_pixels=(1280, 704),
        max_frames=121,
        default_steps=4,
        default_guidance=1.0,
        default_guidance_2=1.0,
        lora=None,
    ),
    "wan2.2-ti2v-5b": WanTarget(
        name="wan2.2-ti2v-5b",
        repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        modes=["t2v", "i2v"],
        max_pixels=(1280, 704),
        max_frames=121,
        default_steps=40,
        default_guidance=3.0,
        default_guidance_2=4.0,
        lora=None,
    ),
    "wan2.1-t2v-14b": WanTarget(
        name="wan2.1-t2v-14b",
        repo_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        modes=["t2v"],
        max_pixels=(1280, 720),
        max_frames=81,
        default_steps=40,
        default_guidance=5.0,
        default_guidance_2=None,
        lora={
            "t2v": {
                "repo": "lightx2v/Wan2.1-Distill-Loras",
                "weight_name": "wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors",
            },
        },
    ),
    "wan2.1-i2v-14b": WanTarget(
        name="wan2.1-i2v-14b",
        repo_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        modes=["i2v"],
        max_pixels=(1280, 720),
        max_frames=81,
        default_steps=40,
        default_guidance=5.0,
        default_guidance_2=None,
        lora={
            "i2v": {
                "repo": "lightx2v/Wan2.1-Distill-Loras",
                "weight_name": "wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors",
            },
        },
    ),
}

DEFAULT_TARGETS = [
    "wan2.2-t2v-a14b",
    "wan2.2-i2v-a14b",
    "wan2.1-t2v-14b",
    "wan2.1-i2v-14b",
]

RESOLUTIONS_T2V = {
    "480p": {"height": 480, "width": 832},
    "720p": {"height": 720, "width": 1280},
}


@dataclass
class Result:
    label: str
    durations: list[float] = field(default_factory=list)
    phase_timings: list[str] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.durations) if self.durations else 0.0

    @property
    def summary(self) -> str:
        if not self.durations:
            return "FAILED"
        m = self.mean
        if len(self.durations) > 1:
            return f"{m:.1f}s (std {statistics.stdev(self.durations):.1f}s)"
        return f"{m:.1f}s"

    @property
    def phase_summary(self) -> str:
        """Return the last phase timing string if available."""
        return self.phase_timings[-1] if self.phase_timings else ""


def _compute_i2v_resolution(image_path: str, max_area: int) -> tuple[int, int]:
    """Compute aspect-preserving resolution from an input image.

    Returns (height, width) aligned to 16px, capped by max_area.
    """
    from PIL import Image

    img = Image.open(image_path)
    aspect = img.height / img.width
    mod = 16
    h = round(np.sqrt(max_area * aspect)) // mod * mod
    w = round(np.sqrt(max_area / aspect)) // mod * mod
    return h, w


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wan T2V/I2V: diffusers vs MAX")
    p.add_argument("--num-warmups", type=int, default=1)
    p.add_argument("--num-iterations", type=int, default=1)
    p.add_argument(
        "--runner",
        nargs="+",
        default=["diffusers", "max"],
        choices=["diffusers", "max"],
    )
    p.add_argument(
        "--mode",
        nargs="+",
        default=["t2v", "i2v"],
        choices=["t2v", "i2v"],
    )
    p.add_argument(
        "--resolution",
        nargs="+",
        default=["480p", "720p"],
        choices=["480p", "720p"],
    )
    p.add_argument(
        "--variant",
        nargs="+",
        default=["base", "lora"],
        choices=["base", "lora"],
    )
    p.add_argument(
        "--input-image",
        default=str(Path(__file__).resolve().parent / "cat.jpg"),
    )
    p.add_argument("--output-dir", default="outputs")
    p.add_argument(
        "--target",
        nargs="+",
        default=None,
        help=f"Target names (choices: {', '.join(TARGETS.keys())}). "
        f"Default: {', '.join(DEFAULT_TARGETS)}.",
    )
    args = p.parse_args(argv)
    args.runners = set(args.runner)
    args.modes = set(args.mode)
    args.resolutions = set(args.resolution)
    args.variants = set(args.variant)
    return args


def _get_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    res_keys = sorted(args.resolutions & {"480p", "720p"})

    # Resolve selected targets
    if args.target:
        target_names = args.target
        for tn in target_names:
            if tn not in TARGETS:
                raise ValueError(
                    f"Unknown target '{tn}'. "
                    f"Valid targets: {', '.join(TARGETS.keys())}"
                )
    else:
        target_names = DEFAULT_TARGETS

    cases: list[dict[str, Any]] = []
    for tname in target_names:
        target = TARGETS[tname]
        max_area = target.max_pixels[0] * target.max_pixels[1]
        for mode in target.modes:
            if mode not in args.modes:
                continue
            prompt = T2V_PROMPT if mode == "t2v" else I2V_PROMPT
            for res in res_keys:
                if mode == "t2v":
                    r = RESOLUTIONS_T2V[res]
                else:
                    # I2V: compute resolution from input image
                    if res == "480p":
                        area = min(480 * 832, max_area)
                    else:
                        area = max_area
                    h, w = _compute_i2v_resolution(args.input_image, area)
                    r = {"height": h, "width": w}

                has_lora = target.lora is not None and mode in target.lora
                label_prefix = (
                    f"{tname} {mode}"
                    if len(target_names) > 1 or target_names != DEFAULT_TARGETS
                    else mode
                )

                if "base" in args.variants:
                    cases.append(
                        {
                            "label": f"{label_prefix} {res} base",
                            "mode": mode,
                            "lora": False,
                            "prompt": prompt,
                            "num_inference_steps": target.default_steps,
                            "guidance_scale": target.default_guidance,
                            "guidance_scale_2": target.default_guidance_2,
                            "num_frames": target.max_frames,
                            "repo_id": target.repo_id,
                            "target": target,
                            **r,
                        }
                    )
                if "lora" in args.variants and has_lora:
                    cases.append(
                        {
                            "label": f"{label_prefix} {res} LoRA",
                            "mode": mode,
                            "lora": True,
                            "prompt": prompt,
                            "num_inference_steps": target.lora_steps,
                            "guidance_scale": target.lora_guidance,
                            "guidance_scale_2": target.lora_guidance_2
                            if target.default_guidance_2 is not None
                            else None,
                            "num_frames": target.max_frames,
                            "repo_id": target.repo_id,
                            "target": target,
                            **r,
                        }
                    )
    return cases


def _save_video(frames: list[Any], path: str, fps: int = 16) -> None:
    """Save PIL/numpy frames to mp4 via PyAV (no system ffmpeg needed)."""
    import av
    import av.video
    from PIL import Image as _PILImage

    # Normalize frames to uint8 numpy arrays
    normalized: list[np.ndarray] = []
    for f in frames:
        if isinstance(f, _PILImage.Image):
            arr = np.array(f.convert("RGB"), dtype=np.uint8)
        else:
            arr = np.array(f)
            if arr.dtype in (np.float32, np.float64):
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        normalized.append(arr)

    h, w = normalized[0].shape[:2]
    container = av.open(path, mode="w")
    stream: av.video.VideoStream = container.add_stream("libx264", rate=fps)  # type: ignore[assignment]
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.codec_context.options = {"crf": "18"}

    for arr in normalized:
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush
    for packet in stream.encode():
        container.mux(packet)
    container.close()


# -- Diffusers ---------------------------------------------------------------


def _load_diffusers_pipe(
    repo_id: str, mode: str, lora_info: dict[str, str] | None
) -> Any:
    import torch

    # Workaround for transformers <=5.2 tied-embedding bug where
    # encoder.embed_tokens.weight is reported MISSING (randomly initialized)
    # during from_pretrained, even though it should be tied to shared.weight.
    # This causes garbled text encoding and distorted output.
    # See: https://github.com/huggingface/transformers/issues/43992
    from transformers import UMT5EncoderModel

    _orig_post_init = UMT5EncoderModel.__init__

    def _patched_init(self, config):
        _orig_post_init(self, config)
        # Re-tie: copy loaded shared.weight → encoder.embed_tokens
        self.encoder.embed_tokens = self.shared

    UMT5EncoderModel.__init__ = _patched_init

    if mode == "t2v":
        from diffusers import WanPipeline

        pipe = WanPipeline.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16
        ).to("cuda")
    else:
        from diffusers import WanImageToVideoPipeline

        pipe = WanImageToVideoPipeline.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16
        ).to("cuda")

    # Restore original __init__ to avoid side effects
    UMT5EncoderModel.__init__ = _orig_post_init

    if lora_info is not None:
        print("  Fusing LoRA...")
        if "weight_name" in lora_info:
            # Single-file LoRA (e.g. Wan2.1 distill)
            pipe.load_lora_weights(
                lora_info["repo"],
                weight_name=lora_info["weight_name"],
            )
        else:
            # A14B MoE: high/low noise expert LoRAs
            pipe.load_lora_weights(
                lora_info["repo"],
                weight_name=f"{lora_info['subfolder']}/high_noise_model.safetensors",
                adapter_name="high",
            )
            pipe.load_lora_weights(
                lora_info["repo"],
                weight_name=f"{lora_info['subfolder']}/low_noise_model.safetensors",
                adapter_name="low",
            )
            pipe.set_adapters(["high", "low"], adapter_weights=[1.0, 1.0])
        pipe.fuse_lora()
        pipe.unload_lora_weights()

    print("  Compiling...")
    for attr in ["transformer", "transformer_2"]:
        module = getattr(pipe, attr, None)
        if module is not None and hasattr(module, "layers"):
            for i, block in enumerate(module.layers):
                module.layers[i] = torch.compile(
                    block, dynamic=True, fullgraph=True
                )
    pipe.text_encoder = torch.compile(
        pipe.text_encoder, dynamic=True, fullgraph=True
    )
    pipe.vae = torch.compile(pipe.vae, dynamic=True, fullgraph=True)
    return pipe


def _run_diffusers(args: argparse.Namespace) -> list[Result]:
    import torch
    from PIL import Image

    cases = _get_cases(args)
    out_dir = Path(args.output_dir) / "diffusers"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Diffusers (torch.compile) ===")
    results: list[Result] = []
    pipe: Any = None
    pipe_key: tuple[str, str, bool] | None = None
    cat_img: Any = None

    for case in cases:
        target: WanTarget = case["target"]
        lora_info: dict[str, str] | None = None
        if case["lora"] and target.lora and case["mode"] in target.lora:
            lora_info = target.lora[case["mode"]]

        key = (str(case["repo_id"]), str(case["mode"]), bool(case["lora"]))
        if pipe is None or key != pipe_key:
            if pipe is not None:
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
            tag = f"{'LoRA' if case['lora'] else 'base'} {case['mode']}"
            print(f"  Loading {tag} pipeline ({case['repo_id']})...")
            pipe = _load_diffusers_pipe(
                str(case["repo_id"]), str(case["mode"]), lora_info
            )
            pipe_key = key

        # Pre-generate noise with torch for reproducibility + parity with MAX.
        vae_t = getattr(pipe, "vae_scale_factor_temporal", 4)
        vae_s = getattr(pipe, "vae_scale_factor_spatial", 8)
        num_ch = pipe.transformer.config.in_channels
        lat_f = (case["num_frames"] - 1) // vae_t + 1
        lat_h = case["height"] // vae_s
        lat_w = case["width"] // vae_s
        noise = torch.randn(
            (1, num_ch, lat_f, lat_h, lat_w),
            generator=torch.Generator("cuda").manual_seed(0),
            device="cuda",
            dtype=torch.bfloat16,
        )

        gen_kwargs: dict[str, Any] = dict(
            prompt=case["prompt"],
            negative_prompt="low quality",
            height=case["height"],
            width=case["width"],
            num_frames=case["num_frames"],
            guidance_scale=case["guidance_scale"],
            num_inference_steps=case["num_inference_steps"],
            latents=noise,
        )
        if case["guidance_scale_2"] is not None:
            gen_kwargs["guidance_scale_2"] = case["guidance_scale_2"]
        if case["mode"] == "i2v":
            if cat_img is None:
                cat_img = Image.open(args.input_image).convert("RGB")
            gen_kwargs["image"] = cat_img

        warmup_kwargs: dict[str, Any] = dict(
            prompt="warmup",
            negative_prompt="low quality",
            height=288,
            width=512,
            num_frames=9,
            guidance_scale=case["guidance_scale"],
            num_inference_steps=2,
        )
        if case["guidance_scale_2"] is not None:
            warmup_kwargs["guidance_scale_2"] = case["guidance_scale_2"]
        if case["mode"] == "i2v":
            warmup_kwargs["image"] = Image.new(
                "RGB", (512, 288), (128, 128, 128)
            )

        for i in range(args.num_warmups):
            print(f"  {case['label']} warmup {i + 1}/{args.num_warmups}")
            with torch.no_grad():
                pipe(**warmup_kwargs)
            torch.cuda.synchronize()

        result = Result(label=str(case["label"]))
        for i in range(args.num_iterations):
            # Track denoising start/end via callback
            _step_times: list[float] = []

            def _step_cb(
                _pipe: Any, _step: int, _ts: Any, cb_kwargs: Any
            ) -> Any:
                torch.cuda.synchronize()
                _step_times.append(time.perf_counter())
                return cb_kwargs

            with torch.no_grad():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                output = pipe(
                    **gen_kwargs,
                    callback_on_step_end=_step_cb,
                )
                torch.cuda.synchronize()
                t_end = time.perf_counter()

            dt = t_end - t0
            result.durations.append(dt)

            # Breakdown: prompt+encode is before first step,
            # denoise is first..last step, decode is after last step
            if _step_times:
                t_denoise_start = _step_times[0]
                t_denoise_end = _step_times[-1]
                prep = t_denoise_start - t0
                denoise = t_denoise_end - t_denoise_start
                decode = t_end - t_denoise_end
                phase = (
                    f"prep={prep:.1f}s, denoise={denoise:.1f}s, "
                    f"decode={decode:.1f}s, total={dt:.1f}s"
                )
                result.phase_timings.append(phase)
                print(f"  {case['label']} iter {i + 1}: {phase}")
            else:
                print(f"  {case['label']} iter {i + 1}: {dt:.1f}s")

            if i == 0:
                fname = str(case["label"]).replace(" ", "_").lower()
                _save_video(output.frames[0], str(out_dir / f"{fname}.mp4"))

        results.append(result)

    if pipe is not None:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    return results


# -- MAX ----------------------------------------------------------------------



def _build_max_argv(
    args: argparse.Namespace,
    case: dict[str, Any],
    output_file: str,
) -> list[str]:
    """Build CLI argv for ``simple_offline_video_generation``."""
    target: WanTarget = case["target"]
    argv = [
        "--model",
        str(case["repo_id"]),
        "--prompt",
        str(case["prompt"]),
        "--negative-prompt",
        "low quality",
        "--num-inference-steps",
        str(case["num_inference_steps"]),
        "--guidance-scale",
        str(case["guidance_scale"]),
        "--output",
        output_file,
        "--num-frames",
        str(case["num_frames"]),
        "--seed",
        "0",
        "--height",
        str(case["height"]),
        "--width",
        str(case["width"]),
    ]
    if case["guidance_scale_2"] is not None:
        argv.extend(["--guidance-scale-2", str(case["guidance_scale_2"])])
    if case["mode"] == "i2v":
        argv.extend(["--input-image", os.path.abspath(args.input_image)])
    if case["lora"] and target.lora and case["mode"] in target.lora:
        info = target.lora[case["mode"]]
        argv.extend(["--lora-repo-id", info["repo"]])
        if "weight_name" in info:
            argv.extend(["--lora-weight-name", info["weight_name"]])
        else:
            argv.extend(["--lora-subfolder", info["subfolder"]])
        argv.extend(["--lora-scale", "1.0"])
    if target.encoding:
        argv.extend(["--quantization-encoding", target.encoding])
    return argv


def _parse_max_e2e_time(output: str) -> float | None:
    """Extract E2E execute time (seconds) from profiling report."""
    import re

    m = re.search(r"E2E execute\s+\d+\s+([\d.]+)", output)
    if m:
        return float(m.group(1)) / 1000.0  # ms → s
    return None


def _run_max(args: argparse.Namespace) -> list[Result]:
    """Run MAX via subprocess to avoid torch/CUDA memory conflicts."""
    import subprocess

    # Locate the simple_offline_video_generation binary.
    # Under bazel run, sibling binaries live next to this script's binary.
    self_bin = Path(sys.argv[0]).resolve()
    max_bin = self_bin.parent / "simple_offline_video_generation"
    if not max_bin.exists():
        # Fallback: use bazel-bin output path directly.
        max_bin = Path(
            "bazel-bin/max/examples/diffusion/simple_offline_video_generation"
        )
    if not max_bin.exists():
        raise FileNotFoundError(
            "Cannot find simple_offline_video_generation binary. "
            "Run: ./bazelw build //max/examples/diffusion:simple_offline_video_generation"
        )

    print(f"\n=== MAX (subprocess: {max_bin.name}) ===")
    cases = _get_cases(args)
    out_dir = Path(args.output_dir) / "max"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []

    for case in cases:
        fname = str(case["label"]).replace(" ", "_").lower()
        output_file = str(out_dir / f"{fname}.mp4")
        argv = _build_max_argv(args, case, output_file)

        result = Result(label=str(case["label"]))
        for i in range(args.num_iterations):
            print(f"  {case['label']} iter {i + 1}/{args.num_iterations}")
            t0 = time.perf_counter()
            proc = subprocess.run(
                [str(max_bin)] + argv,
                capture_output=True,
                text=True,
            )
            t_wall = time.perf_counter() - t0

            if proc.returncode != 0:
                print("    FAILED")
                # Show last few lines of stderr/stdout for diagnostics.
                for line in (proc.stderr or proc.stdout or "").strip().split("\n")[-5:]:
                    print(f"      {line}")
                continue

            # Parse E2E time from profiling output; fall back to wall time.
            e2e = _parse_max_e2e_time(proc.stdout + proc.stderr)
            dt = e2e if e2e is not None else t_wall
            result.durations.append(dt)
            phase = f"e2e={dt:.1f}s, wall={t_wall:.1f}s"
            result.phase_timings.append(phase)
            print(f"    {phase}")

        results.append(result)

    return results


# -- Summary ------------------------------------------------------------------


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"{name} ({gb:.0f}GB)"
    except Exception:
        pass
    return "unknown"


def _build_summary(
    args: argparse.Namespace,
    diffusers: list[Result],
    max_results: list[Result],
    run_dir: Path,
) -> str:
    """Build a markdown summary and return it as a string."""
    gpu = _gpu_name()
    ts = run_dir.name  # timestamp
    lines: list[str] = []
    lines.append(f"# Wan Comparison — {ts}")
    lines.append("")
    lines.append(f"- **GPU**: {gpu}")
    lines.append(f"- **Runners**: {', '.join(sorted(args.runners))}")
    lines.append(f"- **Modes**: {', '.join(sorted(args.modes))}")
    lines.append(f"- **Resolutions**: {', '.join(sorted(args.resolutions))}")
    lines.append(f"- **Variants**: {', '.join(sorted(args.variants))}")
    lines.append("")

    labels = [str(c["label"]) for c in _get_cases(args)]
    d_map = {r.label: r for r in diffusers}
    m_map = {r.label: r for r in max_results}

    # Build markdown table
    header = "| Case |"
    separator = "|---|"
    if diffusers:
        header += " Diffusers |"
        separator += "---|"
    if max_results:
        header += " MAX |"
        separator += "---|"
    if diffusers and max_results:
        header += " Speedup |"
        separator += "---|"
    lines.append(header)
    lines.append(separator)

    for label in labels:
        d = d_map.get(label)
        m = m_map.get(label)
        row = f"| {label} |"
        if diffusers:
            row += f" {d.summary if d else 'N/A'} |"
        if max_results:
            row += f" {m.summary if m else 'N/A'} |"
        if diffusers and max_results and d and m and d.mean > 0 and m.mean > 0:
            row += f" {d.mean / m.mean:.2f}x |"
        elif diffusers and max_results:
            row += " — |"
        lines.append(row)

    # Phase breakdown section
    has_phases = any(
        (d_map.get(l) and d_map[l].phase_summary)
        or (m_map.get(l) and m_map[l].phase_summary)
        for l in labels
    )
    if has_phases:
        lines.append("")
        lines.append("## Phase Breakdown")
        lines.append("")
        for label in labels:
            d = d_map.get(label)
            m = m_map.get(label)
            if (d and d.phase_summary) or (m and m.phase_summary):
                lines.append(f"**{label}**")
                if d and d.phase_summary:
                    lines.append(f"- Diffusers: {d.phase_summary}")
                if m and m.phase_summary:
                    lines.append(f"- MAX: {m.phase_summary}")
                lines.append("")

    # Video list
    videos = sorted(run_dir.rglob("*.mp4"))
    if videos:
        lines.append("## Videos")
        lines.append("")
        for v in videos:
            lines.append(f"- `{v.relative_to(run_dir)}`")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Create timestamped run directory: outputs/<timestamp>/
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / ts
    # Override output_dir so _run_diffusers / _run_max write into run_dir/videos/
    args.output_dir = str(run_dir / "videos")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    diffusers: list[Result] = []
    max_results: list[Result] = []

    if "diffusers" in args.runners:
        try:
            diffusers = _run_diffusers(args)
        except Exception as e:
            print(f"ERROR diffusers: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    if "max" in args.runners:
        try:
            max_results = _run_max(args)
        except Exception as e:
            print(f"ERROR MAX: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    # Build and print summary
    summary = _build_summary(args, diffusers, max_results, run_dir)
    print(f"\n{summary}")

    # Write all_result.md
    result_path = run_dir / "all_result.md"
    result_path.write_text(summary)
    print(f"Results saved to: {result_path}")
    return 0


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)
    raise SystemExit(main())
