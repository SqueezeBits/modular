# Diffusion Pipeline Warmup Test Guide

## Purpose

This document describes the methodology for testing JIT warmup behavior in the
Flux2 diffusion pipeline. The goal is to identify which pipeline parameters
trigger graph recompilation when changed between warmup and profile runs, and
to establish best practices for warmup configuration.

## Background

The MAX diffusion pipeline uses JIT compilation for its compute graphs. The
first execution compiles these graphs, and subsequent runs reuse the compiled
versions. If a parameter change alters tensor shapes in the compiled graph, a
recompilation is triggered, adding overhead to the first execution with the new
shape.

Understanding which parameters cause recompilation is critical for:
- Designing efficient warmup strategies that cover expected input variations
- Avoiding unexpected latency spikes in production serving
- Minimizing total warmup time by only warming up the dimensions that matter

## Test Infrastructure

### Modified Files

- `max/examples/diffusion/simple_offline_generation.py` — Extended with
  warmup-specific CLI arguments and per-iteration profiling breakdown.
- `run_max.sh` — Shell script to configure and launch tests.

### Warmup-Specific CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `--warmup-prompt` | Text prompt for warmup runs | Same as `--prompt` |
| `--warmup-height` | Image height(s) for warmup (supports multiple values) | Same as `--height` |
| `--warmup-width` | Image width(s) for warmup (supports multiple values) | Same as `--width` |
| `--warmup-num-inference-steps` | Denoising steps for warmup | Same as `--num-inference-steps` |

Multiple resolutions can be warmed up by passing space-separated values:
```bash
--warmup-height 512 1024 --warmup-width 512 1024
```
This runs warmup iterations for each (height, width) pair sequentially.

### Per-Iteration Profiling

The profiler reports a per-iteration breakdown of each pipeline component's
execution time (in ms), allowing direct comparison of iteration 1 (potentially
recompiling) against steady-state iterations.

## Methodology

### Step 1: Isolate Individual Factors

To determine which parameter causes recompilation, change only one variable at
a time between warmup and profile runs while keeping all others constant.

**Test matrix:**

| Test | Warmup | Profile | Variable |
|---|---|---|---|
| Prompt only | Short prompt, 1024x1024, 4 steps | Long prompt, 1024x1024, 4 steps | Prompt length |
| Resolution only | Long prompt, 512x512, 4 steps | Long prompt, 1024x1024, 4 steps | Resolution |
| Steps only | Long prompt, 1024x1024, 2 steps | Long prompt, 1024x1024, 4 steps | Step count |

**Detection criteria:** If iteration 1 of the profile run is significantly
slower than iterations 2-3, recompilation occurred for that factor.

### Step 2: Per-Component Breakdown

Once the offending factor is identified, use the per-iteration profiling
breakdown to pinpoint which specific pipeline component triggers the
recompilation. Compare each component's timing on iteration 1 vs iterations 2-3.

### Step 3: Multi-Resolution Warmup Validation

After identifying resolution as a factor, verify that warming up with all
target resolutions eliminates recompilation:
```bash
--warmup-height 512 1024 --warmup-width 512 1024
```

## Observations

### Experiment 1: Factor Isolation

**Setup:** Warmup 2 iterations, profile 3 iterations. Model: FLUX.2-dev.

| Test | Factor Changed | Iter 1 | Iter 2 | Iter 3 | Recompilation? |
|---|---|---:|---:|---:|---|
| Prompt only | Prompt length (42 -> 732 chars) | 2.82s | 2.83s | 2.79s | No |
| Resolution only | Resolution (512x512 -> 1024x1024) | 6.65s | 2.84s | 2.84s | Yes (~3.8s overhead) |
| Steps only | Steps (2 -> 4) | 2.85s | 2.82s | 2.86s | No |

**Conclusion:** Only image resolution triggers recompilation. Prompt length and
step count can vary freely without overhead.

### Experiment 2: Multi-Resolution Warmup

**Setup:** Warmup with both 512x512 and 1024x1024, profile at 1024x1024 with
16 steps and a long prompt (different from warmup prompt).

| Phase | Config | Time |
|---|---|---:|
| Warmup 1 | 512x512, 2 steps | 12.36s (initial JIT) |
| Warmup 2 | 512x512, 2 steps | 0.54s (cached) |
| Warmup 3 | 1024x1024, 2 steps | 5.14s (recompile for new resolution) |
| Warmup 4 | 1024x1024, 2 steps | 1.76s (cached) |
| Profile 1 | 1024x1024, 16 steps | 9.38s |
| Profile 2 | 1024x1024, 16 steps | 9.40s |
| Profile 3 | 1024x1024, 16 steps | 9.43s |

**Conclusion:** Pre-warming all target resolutions eliminates recompilation
during profiling, regardless of prompt or step count differences.

### Experiment 3: Per-Component Breakdown

**Setup:** Warmup at 512x512 (4 steps), profile at 1024x1024 (4 steps), same
prompt and step count — isolating resolution only.

| Component | Iter 1 (ms) | Iter 2 (ms) | Iter 3 (ms) | Recompiles? |
|---|---:|---:|---:|---|
| **decode_latents** | **4,100.9** | 307.6 | 307.3 | **Yes (~3,793ms)** |
| component/transformer | 2,119.1 | 2,113.3 | 2,123.1 | No |
| prepare_embeddings | 78.1 | 77.2 | 77.1 | No |
| component/text_encoder | 78.0 | 77.1 | 77.0 | No |
| scheduler_step | 1.9 | 2.4 | 2.5 | No |
| preprocess_latents | 0.2 | 0.2 | 0.2 | No |
| prepare_scheduler | 0.1 | 0.1 | 0.1 | No |

**Conclusion:** `decode_latents` is the sole component that recompiles when
resolution changes. It accounts for the entire ~3.8s overhead observed in the
end-to-end timing. All other components — including the transformer, text
encoder, scheduler, and latent preprocessing — handle resolution changes via
dynamic shapes without recompilation.

### Root Cause Analysis

The `decode_latents` component calls the VAE decoder, which reshapes latents
from the packed sequence format `(B, image_seq_len, C)` back to spatial
dimensions `(B, latent_h, latent_w, C)` before decoding. When the resolution
changes, `latent_h` and `latent_w` change, causing the VAE decode graph to be
recompiled.

The resolution flows through this chain:
```
Resolution (H, W)
  -> Latent dims (H/8, W/8)
    -> image_seq_len = (H/16) x (W/16)
      -> decode_latents reshape and VAE decode graph
```

Key dimensions by resolution:
- 512x512: latent 64x64, image_seq_len = 1,024
- 1024x1024: latent 128x128, image_seq_len = 4,096

## Best Practices for Warmup

1. **Always warm up each target resolution.** Resolution is the only parameter
   that triggers recompilation. If your serving workload handles multiple
   resolutions, include each in the warmup sequence.

2. **Use minimal steps and a short prompt for warmup.** Steps and prompt length
   do not affect compilation, so use small values (e.g., 2 steps, "warmup run")
   to minimize warmup wall-clock time.

3. **A single warmup iteration per resolution is sufficient for steady state,
   but two are recommended** to confirm the compiled graph is cached and
   performing at expected throughput.

4. **Example warmup configuration for multi-resolution serving:**
   ```bash
   --warmup-prompt "warmup run" \
   --warmup-height 512 768 1024 \
   --warmup-width 512 768 1024 \
   --warmup-num-inference-steps 2 \
   --num-warmups 1
   ```
