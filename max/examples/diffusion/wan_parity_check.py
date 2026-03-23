#!/usr/bin/env python3
"""Wan T2V module-level parity check: diffusers vs MAX.

Dumps intermediate tensors from both pipelines and compares them.
Uses minimal steps (2) and 480p for speed.

Usage:
    python max/examples/diffusion/wan_parity_check.py
    python max/examples/diffusion/wan_parity_check.py --model Wan-AI/Wan2.1-T2V-14B-Diffusers
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wan parity check")
    p.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="HF model repo",
    )
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--num-steps", type=int, default=2)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="wan_parity_output")
    p.add_argument(
        "--skip-diffusers", action="store_true", help="Skip diffusers run"
    )
    p.add_argument("--skip-max", action="store_true", help="Skip MAX run")
    return p.parse_args()


# ── Diffusers ────────────────────────────────────────────────


def run_diffusers(args: argparse.Namespace) -> None:
    import torch
    from diffusers import WanPipeline

    out = Path(args.output_dir) / "diffusers"
    out.mkdir(parents=True, exist_ok=True)

    print("=== Diffusers: Loading pipeline ===")
    pipe = WanPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")

    generator = torch.Generator("cuda").manual_seed(args.seed)

    # 1. Text encoder output
    print("  Dumping text encoder output...")
    prompt = "A cat walking"
    negative_prompt = "low quality"

    # Get prompt embeds via the pipeline's encode_prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        *_rest,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device="cuda",
    )
    np.save(
        out / "prompt_embeds.npy",
        prompt_embeds.detach().float().cpu().numpy(),
    )
    np.save(
        out / "negative_prompt_embeds.npy",
        negative_prompt_embeds.detach().float().cpu().numpy(),
    )
    print(f"    prompt_embeds: {prompt_embeds.shape}")
    print(f"    negative_prompt_embeds: {negative_prompt_embeds.shape}")

    # 2. Run pipeline with step callback to capture intermediates
    step_latents: list[np.ndarray] = []
    step_noise_preds: list[np.ndarray] = []

    def step_callback(pipe_obj, step_idx, timestep, cb_kwargs):
        # Capture latents at each step
        latents = cb_kwargs["latents"]
        step_latents.append(latents.detach().float().cpu().numpy())
        return cb_kwargs

    print("  Running pipeline...")
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            generator=generator,
            callback_on_step_end=step_callback,
            output_type="latent",
        )

    # 3. Save latents at each step
    for i, lat in enumerate(step_latents):
        np.save(out / f"latents_step{i}.npy", lat)
        print(f"    latents_step{i}: {lat.shape}, mean={lat.mean():.6f}, std={lat.std():.6f}")

    # 4. Get final latents (last step_latents is after final scheduler step)
    final_latents = torch.from_numpy(step_latents[-1]).to("cuda", dtype=torch.bfloat16) if step_latents else None
    if final_latents is None:
        print("    ERROR: no step latents captured")
        return
    final_lat_np = final_latents.detach().float().cpu().numpy()
    np.save(out / "final_latents.npy", final_lat_np)
    print(f"    final_latents: {final_lat_np.shape}, mean={final_lat_np.mean():.6f}")

    # 5. Save initial noise (reconstruct with same seed)
    generator2 = torch.Generator("cuda").manual_seed(args.seed)
    num_channels = pipe.transformer.config.in_channels
    vae_scale_t = pipe.vae.config.temporal_compression_ratio if hasattr(pipe.vae.config, "temporal_compression_ratio") else 4
    vae_scale_s = pipe.vae.config.spatial_compression_ratio if hasattr(pipe.vae.config, "spatial_compression_ratio") else 8
    latent_frames = (args.num_frames - 1) // vae_scale_t + 1
    latent_h = args.height // vae_scale_s
    latent_w = args.width // vae_scale_s
    shape = (1, num_channels, latent_frames, latent_h, latent_w)
    noise = torch.randn(shape, generator=generator2, device="cuda", dtype=torch.bfloat16)
    np.save(out / "initial_noise.npy", noise.detach().float().cpu().numpy())
    print(f"    initial_noise: {noise.shape}, mean={noise.float().mean():.6f}")

    # 6. Scheduler sigmas/timesteps
    pipe.scheduler.set_timesteps(args.num_steps, device="cuda")
    timesteps = pipe.scheduler.timesteps.detach().float().cpu().numpy()
    sigmas = pipe.scheduler.sigmas.detach().float().cpu().numpy() if hasattr(pipe.scheduler, 'sigmas') else None
    np.save(out / "timesteps.npy", timesteps)
    print(f"    timesteps: {timesteps}")
    if sigmas is not None:
        np.save(out / "sigmas.npy", sigmas)
        print(f"    sigmas: {sigmas}")

    # 7. Decode with VAE
    print("  Decoding with VAE...")
    with torch.no_grad():
        # Denormalize
        latents_mean = torch.tensor(
            pipe.vae.config.latents_mean, dtype=torch.float32, device="cuda"
        ).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(
            pipe.vae.config.latents_std, dtype=torch.float32, device="cuda"
        ).view(1, -1, 1, 1, 1)
        denorm = final_latents.float() * latents_std + latents_mean
        np.save(out / "denorm_latents.npy", denorm.cpu().numpy())
        print(f"    denorm_latents: mean={denorm.mean():.6f}")

        decoded = pipe.vae.decode(denorm.to(torch.bfloat16), return_dict=False)[0]
        decoded_np = decoded.detach().float().cpu().numpy()
        np.save(out / "decoded_video.npy", decoded_np)
        print(f"    decoded_video: {decoded_np.shape}, mean={decoded_np.mean():.6f}")

    del pipe
    torch.cuda.empty_cache()
    print("  Diffusers done.\n")


# ── MAX ──────────────────────────────────────────────────────


def run_max(args: argparse.Namespace) -> None:
    out = Path(args.output_dir) / "max"
    out.mkdir(parents=True, exist_ok=True)

    # Set env var to enable MAX intermediate dumps
    os.environ["WAN_PARITY_DUMP_DIR"] = str(out)

    import asyncio

    from max.driver import CPU, DeviceSpec
    from max.interfaces import PipelineTask, PixelGenerationInputs, RequestID
    from max.interfaces.provider_options import (
        ImageProviderOptions,
        ProviderOptions,
        VideoProviderOptions,
    )
    from max.interfaces.request import OpenResponsesRequest
    from max.interfaces.request.open_responses import OpenResponsesRequestBody
    from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
    from max.pipelines.core import PixelContext
    from max.pipelines.lib import PixelGenerationTokenizer
    from max.pipelines.lib.interfaces import DiffusionPipeline
    from max.pipelines.lib.pipeline_variants.pixel_generation import (
        PixelGenerationPipeline,
    )

    print("=== MAX: Loading pipeline ===")
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

    diffusers_config = config.model.diffusers_config
    max_length = 226
    tokenizer = PixelGenerationTokenizer(
        model_path=args.model,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=max_length,
    )

    from typing import Any, cast

    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )

    # Access the underlying wan pipeline
    wan_pipe: Any = pipeline._pipeline_model

    # 1. Text encoder output
    print("  Dumping text encoder output...")
    prompt = "A cat walking"
    negative_prompt = "low quality"

    body = OpenResponsesRequestBody(
        model=args.model,
        input=prompt,
        seed=args.seed,
        provider_options=ProviderOptions(
            image=ImageProviderOptions(
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                steps=args.num_steps,
                guidance_scale=args.guidance_scale,
            ),
            video=VideoProviderOptions(
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                steps=args.num_steps,
                num_frames=args.num_frames,
                guidance_scale_2=args.guidance_scale,
            ),
        ),
    )
    request = OpenResponsesRequest(request_id=RequestID(), body=body)
    context = asyncio.run(tokenizer.new_context(request))
    model_inputs = wan_pipe.prepare_inputs(context)

    # Get prompt embeddings
    (
        prompt_embeds_buf,
        negative_prompt_embeds_buf,
        batched_prompt_embeds,
        do_cfg,
    ) = wan_pipe._prepare_prompt_state(model_inputs)

    from max.dtype import DType
    from max.pipelines.architectures.autoencoders.autoencoder_kl_wan import (
        _buffer_to_numpy_f32,
        _numpy_f32_to_buffer,
    )

    cpu = CPU()
    prompt_embeds_np = _buffer_to_numpy_f32(prompt_embeds_buf, cpu)
    np.save(out / "prompt_embeds.npy", prompt_embeds_np)
    print(f"    prompt_embeds: {prompt_embeds_np.shape}")

    if negative_prompt_embeds_buf is not None:
        neg_np = _buffer_to_numpy_f32(negative_prompt_embeds_buf, cpu)
        np.save(out / "negative_prompt_embeds.npy", neg_np)
        print(f"    negative_prompt_embeds: {neg_np.shape}")

    # 2. Initial latents — use diffusers noise if available for parity
    device = wan_pipe.transformer.devices[0]
    diffusers_dir = Path(args.output_dir) / "diffusers"
    diffusers_noise_path = diffusers_dir / "initial_noise.npy"
    diffusers_timesteps_path = diffusers_dir / "timesteps.npy"
    diffusers_sigmas_path = diffusers_dir / "sigmas.npy"

    if diffusers_noise_path.exists():
        print("    Loading diffusers noise for parity comparison...")
        noise_f32 = np.load(diffusers_noise_path).astype(np.float32)
        latents = _numpy_f32_to_buffer(noise_f32, DType.float32, device)
        print(f"    initial_noise (from diffusers): {noise_f32.shape}, mean={noise_f32.mean():.6f}")
    else:
        latents = wan_pipe._prepare_latents(model_inputs, device)
        print("    initial_noise: generated by MAX (no diffusers noise found)")

    latents_np = _buffer_to_numpy_f32(latents, cpu)
    np.save(out / "initial_noise.npy", latents_np)

    # 3. Scheduler timesteps
    print(f"    timesteps (MAX): {model_inputs.timesteps}")
    if diffusers_timesteps_path.exists():
        d_ts = np.load(diffusers_timesteps_path)
        print(f"    timesteps (diffusers): {d_ts}")
        print(f"    timestep diff: {np.abs(model_inputs.timesteps - d_ts).max():.6f}")

    timesteps_np = np.ascontiguousarray(model_inputs.timesteps, dtype=np.float32)
    np.save(out / "timesteps.npy", timesteps_np)

    coefficients = model_inputs.step_coefficients
    np.save(out / "step_coefficients.npy", np.array(coefficients, dtype=np.float32))

    # 4. Run denoising and capture per-step latents
    print("  Running denoising...")
    wan_pipe.vae.prewarm_for_latent_shape(tuple(int(d) for d in latents.shape))

    (
        rope_cos, rope_sin,
        batched_timesteps, coeff_buffers,
        boundary_step_idx, spatial_shape,
        has_moe, guidance_scale_high, guidance_scale_low,
    ) = wan_pipe._prepare_scheduler_state(
        latents, model_inputs, prompt_embeds_buf, do_cfg, device,
    )

    # Manual denoising loop to capture intermediates
    from max.driver import Buffer

    step_state = (None, None, None)
    if not wan_pipe._moe_dual_loaded:
        wan_pipe._activate_transformer_weights(use_secondary=False)

    for i in range(len(batched_timesteps)):
        dit_timestep = batched_timesteps[i]
        latent_model_input = wan_pipe.compiled.cast_f32_to_model_dtype.execute(latents)[0]

        noise_pred_buf = wan_pipe._run_transformer_forward(
            transformer_model=wan_pipe.transformer,
            latent_model_input=latent_model_input,
            dit_timestep=dit_timestep,
            prompt_embeds=prompt_embeds_buf,
            batched_prompt_embeds=batched_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds_buf,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            spatial_shape=spatial_shape,
            do_cfg=do_cfg,
            guidance_scale=guidance_scale_high,
        )

        # Save noise pred
        noise_pred_np = _buffer_to_numpy_f32(noise_pred_buf, cpu)
        np.save(out / f"noise_pred_step{i}.npy", noise_pred_np)
        print(f"    noise_pred_step{i}: mean={noise_pred_np.mean():.6f}, std={noise_pred_np.std():.6f}")

        latents, step_state = wan_pipe._denoise_step(
            latents, noise_pred_buf, coeff_buffers[i], step_state,
        )

        latents_np = _buffer_to_numpy_f32(latents, cpu)
        np.save(out / f"latents_step{i}.npy", latents_np)
        print(f"    latents_step{i}: mean={latents_np.mean():.6f}, std={latents_np.std():.6f}")

    # 5. Final latents
    final_np = _buffer_to_numpy_f32(latents, cpu)
    np.save(out / "final_latents.npy", final_np)
    print(f"    final_latents: mean={final_np.mean():.6f}")

    # 6. Denormalize + VAE decode
    denorm_latents = wan_pipe._denormalize_vae_latents(latents)
    denorm_np = _buffer_to_numpy_f32(denorm_latents, cpu)
    np.save(out / "denorm_latents.npy", denorm_np)
    print(f"    denorm_latents: mean={denorm_np.mean():.6f}")

    decoded = wan_pipe.vae.decode_5d(denorm_latents)
    decoded_np = _buffer_to_numpy_f32(decoded, cpu)
    np.save(out / "decoded_video.npy", decoded_np)
    print(f"    decoded_video: {decoded_np.shape}, mean={decoded_np.mean():.6f}")

    print("  MAX done.\n")


# ── Compare ──────────────────────────────────────────────────


def compare(args: argparse.Namespace) -> None:
    d_dir = Path(args.output_dir) / "diffusers"
    m_dir = Path(args.output_dir) / "max"

    if not d_dir.exists() or not m_dir.exists():
        print("Skipping comparison (missing one side)")
        return

    print("=== Parity Comparison ===")
    print(f"{'Module':<30s} {'Shape':>20s} {'MaxAbsDiff':>12s} {'MeanAbsDiff':>12s} {'CosSim':>10s}")
    print("-" * 90)

    files = [
        "prompt_embeds.npy",
        "negative_prompt_embeds.npy",
        "initial_noise.npy",
        "timesteps.npy",
        "latents_step0.npy",
        "latents_step1.npy",
        "final_latents.npy",
        "denorm_latents.npy",
        "decoded_video.npy",
    ]

    for fname in files:
        d_path = d_dir / fname
        m_path = m_dir / fname
        if not d_path.exists() or not m_path.exists():
            print(f"  {fname:<30s} {'MISSING':>20s}")
            continue

        d = np.load(d_path).astype(np.float32)
        m = np.load(m_path).astype(np.float32)

        if d.shape != m.shape:
            print(f"  {fname:<30s} SHAPE MISMATCH: {d.shape} vs {m.shape}")
            continue

        diff = np.abs(d - m)
        max_diff = diff.max()
        mean_diff = diff.mean()

        # Cosine similarity
        d_flat = d.flatten()
        m_flat = m.flatten()
        dot = np.dot(d_flat, m_flat)
        norm_d = np.linalg.norm(d_flat)
        norm_m = np.linalg.norm(m_flat)
        cos_sim = dot / (norm_d * norm_m + 1e-8)

        shape_str = str(d.shape)
        print(
            f"  {fname:<30s} {shape_str:>20s} {max_diff:>12.6f} {mean_diff:>12.6f} {cos_sim:>10.6f}"
        )


def main() -> int:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.skip_diffusers:
        try:
            run_diffusers(args)
        except Exception as e:
            print(f"ERROR diffusers: {e}")
            import traceback
            traceback.print_exc()

    if not args.skip_max:
        try:
            run_max(args)
        except Exception as e:
            print(f"ERROR MAX: {e}")
            import traceback
            traceback.print_exc()

    compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
