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
    ./bazelw run //max/examples/diffusion:wan_comparison -- --skip-max --only-480p
    ./bazelw run //max/examples/diffusion:wan_comparison -- --skip-diffusers
    ./bazelw run //max/examples/diffusion:wan_comparison
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import subprocess
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

LORA = {
    "t2v": {
        "repo": "lightx2v/Wan2.2-Lightning",
        "subfolder": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0",
    },
    "i2v": {
        "repo": "lightx2v/Wan2.2-Lightning",
        "subfolder": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
    },
}

RESOLUTIONS = {
    "t2v": {
        "480p": {"height": 480, "width": 832},
        "720p": {"height": 720, "width": 1280},
    },
    "i2v": {
        "480p": {"height": 832, "width": 480},
        "720p": {"height": 1280, "width": 720},
    },
}


@dataclass
class Result:
    label: str
    durations: list[float] = field(default_factory=list)

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wan T2V/I2V: diffusers vs MAX")
    p.add_argument("--num-warmups", type=int, default=1)
    p.add_argument("--num-iterations", type=int, default=1)
    p.add_argument("--skip-diffusers", action="store_true")
    p.add_argument("--skip-max", action="store_true")
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--skip-lora", action="store_true")
    p.add_argument("--skip-t2v", action="store_true")
    p.add_argument("--skip-i2v", action="store_true")
    p.add_argument("--only-720p", action="store_true")
    p.add_argument("--only-480p", action="store_true")
    p.add_argument(
        "--input-image",
        default=str(Path(__file__).resolve().parent / "cat.jpg"),
    )
    p.add_argument("--output-dir", default="wan_comparison_output")
    return p.parse_args(argv)


def _get_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    res_keys = ["480p", "720p"]
    if args.only_480p:
        res_keys = ["480p"]
    if args.only_720p:
        res_keys = ["720p"]

    cases: list[dict[str, Any]] = []
    for mode in ["t2v", "i2v"]:
        if mode == "t2v" and args.skip_t2v:
            continue
        if mode == "i2v" and args.skip_i2v:
            continue
        prompt = T2V_PROMPT if mode == "t2v" else I2V_PROMPT
        for res in res_keys:
            r = RESOLUTIONS[mode][res]
            if not args.skip_base:
                cases.append({
                    "label": f"{mode} {res} base",
                    "mode": mode,
                    "lora": False,
                    "prompt": prompt,
                    "num_inference_steps": 40,
                    "guidance_scale": 3.0,
                    "guidance_scale_2": 4.0,
                    **r,
                })
            if not args.skip_lora:
                cases.append({
                    "label": f"{mode} {res} LoRA",
                    "mode": mode,
                    "lora": True,
                    "prompt": prompt,
                    "num_inference_steps": 4,
                    "guidance_scale": 1.0,
                    "guidance_scale_2": 1.0,
                    **r,
                })
    return cases


def _save_video_ffmpeg(frames: list[Any], path: str, fps: int = 16) -> None:
    """Save PIL/numpy frames to mp4 via ffmpeg."""
    # Normalize frames to uint8 numpy arrays
    from PIL import Image as _PILImage

    normalized: list[np.ndarray] = []
    for f in frames:
        # Convert PIL → numpy uint8 RGB
        if isinstance(f, _PILImage.Image):
            arr = np.array(f.convert("RGB"), dtype=np.uint8)
        else:
            arr = np.array(f)
            if arr.dtype in (np.float32, np.float64):
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        normalized.append(arr)

    h, w = normalized[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = b"".join(f.tobytes() for f in normalized)
    proc.communicate(input=raw)


# ── Diffusers ────────────────────────────────────────────────


def _load_diffusers_pipe(mode: str, lora: bool) -> Any:
    import torch

    if mode == "t2v":
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", torch_dtype=torch.bfloat16
        ).to("cuda")
    else:
        from diffusers import WanImageToVideoPipeline
        pipe = WanImageToVideoPipeline.from_pretrained(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers", torch_dtype=torch.bfloat16
        ).to("cuda")

    if lora:
        info = LORA[mode]
        print("  Fusing LoRA...")
        pipe.load_lora_weights(
            info["repo"],
            weight_name=f"{info['subfolder']}/high_noise_model.safetensors",
            adapter_name="high",
        )
        pipe.load_lora_weights(
            info["repo"],
            weight_name=f"{info['subfolder']}/low_noise_model.safetensors",
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
    pipe_key: tuple[str, bool] | None = None
    cat_img: Any = None

    for case in cases:
        key = (str(case["mode"]), bool(case["lora"]))
        if pipe is None or key != pipe_key:
            if pipe is not None:
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
            tag = f"{'LoRA' if case['lora'] else 'base'} {case['mode']}"
            print(f"  Loading {tag} pipeline...")
            pipe = _load_diffusers_pipe(str(case["mode"]), bool(case["lora"]))
            pipe_key = key

        gen_kwargs: dict[str, Any] = dict(
            prompt=case["prompt"],
            negative_prompt="low quality",
            height=case["height"],
            width=case["width"],
            num_frames=81,
            guidance_scale=case["guidance_scale"],
            guidance_scale_2=case["guidance_scale_2"],
            num_inference_steps=case["num_inference_steps"],
        )
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
            guidance_scale_2=case["guidance_scale_2"],
            num_inference_steps=2,
        )
        if case["mode"] == "i2v":
            warmup_kwargs["image"] = Image.new("RGB", (512, 288), (128, 128, 128))

        for i in range(args.num_warmups):
            print(f"  {case['label']} warmup {i+1}/{args.num_warmups}")
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
                print(
                    f"  {case['label']} iter {i+1}: "
                    f"prep={prep:.1f}s, denoise={denoise:.1f}s, "
                    f"decode={decode:.1f}s, total={dt:.1f}s"
                )
            else:
                print(f"  {case['label']} iter {i+1}: {dt:.1f}s")

            if i == 0:
                fname = str(case["label"]).replace(" ", "_").lower()
                _save_video_ffmpeg(
                    output.frames[0], str(out_dir / f"{fname}.mp4")
                )

        results.append(result)

    if pipe is not None:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    return results


# ── MAX ──────────────────────────────────────────────────────


def _run_max(args: argparse.Namespace) -> list[Result]:
    print("\n=== MAX ===")
    cases = _get_cases(args)
    out_dir = Path(args.output_dir) / "max"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []

    for case in cases:
        fname = str(case["label"]).replace(" ", "_").lower()
        output_file = str(out_dir / f"{fname}.mp4")
        model = (
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
            if case["mode"] == "t2v"
            else "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        )

        cmd = [
            "./bazelw", "run",
            "//max/examples/diffusion:simple_offline_video_generation", "--",
            "--model", model,
            "--prompt", str(case["prompt"]),
            "--negative-prompt", "low quality",
            "--num-inference-steps", str(case["num_inference_steps"]),
            "--guidance-scale", str(case["guidance_scale"]),
            "--guidance-scale-2", str(case["guidance_scale_2"]),
            "--output", output_file,
            "--num-frames", "81",
            "--height", str(case["height"]),
            "--width", str(case["width"]),
        ]
        if case["mode"] == "i2v":
            cmd.extend(["--input-image", os.path.abspath(args.input_image)])
        if case["lora"]:
            info = LORA[str(case["mode"])]
            cmd.extend([
                "--lora-repo-id", info["repo"],
                "--lora-subfolder", info["subfolder"],
                "--lora-scale", "1.0",
            ])

        result = Result(label=str(case["label"]))
        for i in range(args.num_iterations):
            print(f"  {case['label']} iter {i+1}/{args.num_iterations}")
            # Stream stderr (tqdm/logs) live, capture stdout for timing
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=None,  # stderr → terminal
                text=True,
            )
            assert proc.stdout is not None
            stdout = proc.stdout.read()
            proc.wait()
            if proc.returncode != 0:
                print(f"    FAILED (exit {proc.returncode})")
                continue

            for line in stdout.split("\n"):
                if "Timing:" in line:
                    try:
                        total_str = line.split("total=")[1].split("s")[0]
                        dt = float(total_str)
                        result.durations.append(dt)
                        print(f"    {dt:.1f}s")
                    except (IndexError, ValueError):
                        pass
                    break

        results.append(result)
    return results


# ── Summary ──────────────────────────────────────────────────


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    diffusers: list[Result] = []
    max_results: list[Result] = []

    if not args.skip_diffusers:
        try:
            diffusers = _run_diffusers(args)
        except Exception as e:
            print(f"ERROR diffusers: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    if not args.skip_max:
        try:
            max_results = _run_max(args)
        except Exception as e:
            print(f"ERROR MAX: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # Summary
    w = 62
    print(f"\n{'='*w}")
    print(f"  Wan Comparison — {datetime.now():%Y-%m-%d}")
    print(f"  GPU: {_gpu_name()}")
    print(f"  Output: {os.path.abspath(args.output_dir)}/")
    print(f"{'='*w}")

    labels = [str(c["label"]) for c in _get_cases(args)]
    d_map = {r.label: r for r in diffusers}
    m_map = {r.label: r for r in max_results}

    cols = ""
    if diffusers:
        cols += f"  {'Diffusers':>12s}"
    if max_results:
        cols += f"  {'MAX':>12s}"
    if diffusers and max_results:
        cols += f"  {'Speedup':>9s}"
    print(f"  {'':18s}{cols}")
    print(f"  {'-'*(18 + len(cols))}")

    for label in labels:
        d = d_map.get(label)
        m = m_map.get(label)
        row = f"  {label:18s}"
        if diffusers:
            row += f"  {d.summary if d else 'N/A':>12s}"
        if max_results:
            row += f"  {m.summary if m else 'N/A':>12s}"
        if diffusers and max_results and d and m and d.mean > 0 and m.mean > 0:
            row += f"  {d.mean / m.mean:>8.2f}x"
        print(row)

    print(f"{'='*w}")

    # Only show videos from current run's test cases
    case_fnames = {
        str(c["label"]).replace(" ", "_").lower() for c in _get_cases(args)
    }
    videos = sorted(
        v for v in Path(args.output_dir).rglob("*.mp4")
        if v.stem in case_fnames
    )
    if videos:
        print(f"\n  Output videos:")
        for v in videos:
            print(f"    {v}")
    print()
    return 0


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)
    raise SystemExit(main())
