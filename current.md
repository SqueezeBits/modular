# Wan I2V Current State

## What was done (committed)
- Chunked VAE encoder matching diffusers' cache-based approach (rel L2 0.011)
- WanEncoder3dCached with 24 cache slots, mean extraction in-graph
- I2V pipeline in dedicated pipeline_wan_i2v.py
- 2-pass CFG (positive/negative separate) to halve activation memory → dual-load MoE works at 720p
- wan_comparison.py benchmark (diffusers vs MAX, T2V + I2V)
- Per-stage timing instrumentation

## What was done (uncommitted, on top of last push)
- Dynamic H/W for encoder compilation — compile once, works for any resolution
- `prewarm_encoder()` called in `init_remaining_components()` so first encode doesn't stall
- Replaced per-resolution cache dict with single `_chunked_encoder` field
- I2V resolution fix in wan_comparison.py (portrait for cat.jpg)
- Video save fix (PIL → numpy uint8 RGB properly)
- Diffusers timing breakdown via callback_on_step_end
- MAX stderr streaming (tqdm/logs visible during comparison runs)
- ftfy stub for bazel sandbox (diffusers dependency)

## Known issues
- T2V base 40-step CFG still OOMs with dual-load on H200 (runtime/kernel merge regression, not our code)
- pyright warnings: _WanVAEEncoder, _WanVAEDecoderFirstFrame unused (kept for reference)
- out_shape_4d unused in decode_4d
- wan_comparison.py diffusers video quality not verified yet (PIL save path)

## How to run
```bash
# MAX I2V only, 480p LoRA
./bazelw run //max/examples/diffusion:wan_comparison -- --only-480p --skip-base --skip-max --skip-t2v

# Diffusers only, 480p LoRA
./bazelw run //max/examples/diffusion:wan_comparison -- --only-480p --skip-base --skip-diffusers --skip-t2v

# Full comparison (480p + 720p, base + LoRA, T2V + I2V)
./bazelw run //max/examples/diffusion:wan_comparison

# Direct MAX run
./bazelw run //max/examples/diffusion:simple_offline_video_generation -- \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --prompt "..." --input-image cat.jpg --num-frames 81 \
  --num-inference-steps 4 --guidance-scale 1.0 --guidance-scale-2 1.0 \
  --lora-repo-id lightx2v/Wan2.2-Lightning \
  --lora-subfolder Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1 \
  --lora-scale 1.0 --output i2v_lora.mp4
```

## Key files
- `max/python/max/pipelines/architectures/autoencoders/autoencoder_kl_wan.py` — chunked encoder
- `max/python/max/pipelines/architectures/wan/pipeline_wan_i2v.py` — I2V pipeline
- `max/python/max/pipelines/architectures/wan/pipeline_wan.py` — 2-pass CFG, dual-load
- `max/examples/diffusion/wan_comparison.py` — benchmark script
- `max/examples/diffusion/simple_offline_video_generation.py` — timing instrumentation
