# Flux Transformer Profiling

This document describes the tooling and methodology for profiling the transformer
component of Flux diffusion pipelines across two frameworks (MAX and diffusers)
and three input shapes (flux1, flux2-t2i, flux2-i2i).

## Scripts

### `profile_transformer.py`

Profiles one (framework, input-shape) combination per invocation and dumps a
Chrome-trace JSON file.  Run six times with different arguments to cover all
combinations.

**Frameworks**

| Framework | Details |
|-----------|---------|
| diffusers | `DiffusionPipeline.from_pretrained` with `torch.compile(mode="max-autotune", fullgraph=True)` |
| MAX | `PixelGenerationPipeline` with the compiled MAX transformer |

**Input shapes**

| Shape | Seq len (latents) | in\_channels | encoder\_hidden\_states | Notes |
|-------|------------------:|-------------:|------------------------:|-------|
| flux1 | 4096 | 64 | 4096 (T5-XXL) | + pooled 768 (CLIP) |
| flux2-t2i | 4096 | 128 | 15360 (Mistral3 × 3 layers) | img\_ids shape (1, 4096, 4) int64 |
| flux2-i2i | 8192 | 128 | 15360 (Mistral3 × 3 layers) | noise + image latents concatenated |

**CLI**

```
--model MODEL           HuggingFace model ID (required)
--framework {diffusers,max}   Framework to use (required)
--input-shape {flux1,flux2-t2i,flux2-i2i}   Input shape (required)
--output PATH           Output Chrome trace JSON (default: transformer_{framework}_{input_shape}.json)
--num-warmups N         Warmup iterations (default: 3)
--num-iterations N      Profiled iterations in trace (default: 3)
```

**Usage — all 6 combinations**

```bash
# Diffusers (activate conda env first):
conda activate diffusers
for SHAPE in flux1 flux2-t2i flux2-i2i; do
  MODEL=black-forest-labs/FLUX.1-dev
  [[ $SHAPE == flux2* ]] && MODEL=black-forest-labs/FLUX.2-dev
  python max/examples/diffusion/profile_transformer.py \
      --model $MODEL --framework diffusers --input-shape $SHAPE \
      --output traces/transformer_diffusers_${SHAPE}.json
done

# MAX (via bazel so MAX packages are on the path):
for SHAPE in flux1 flux2-t2i flux2-i2i; do
  MODEL=black-forest-labs/FLUX.1-dev
  [[ $SHAPE == flux2* ]] && MODEL=black-forest-labs/FLUX.2-dev
  ./bazelw run //max/examples/diffusion:profile_transformer -- \
      --model $MODEL --framework max --input-shape $SHAPE \
      --output traces/transformer_max_${SHAPE}.json
done
```

**Output**

One Chrome trace JSON per run.  Warmup latencies are printed to stdout:

```
Warmup: iter 0: 437.64 ms,  iter 1: 107.92 ms,  iter 2: 105.75 ms
Stable mean (iter 1+): 106.84 ms
Trace saved to: traces/transformer_max_flux1.json
```

The first iteration includes torch.compile autotuning (diffusers) or JIT
compilation (MAX), so only iter 1+ reflect steady-state latency.

**Design notes**

- Only the transformer is moved to GPU; other pipeline components stay on CPU
  to avoid OOM when loading multiple models sequentially.
- `_free_gpu()` calls `synchronize`, `empty_cache` (×2), and
  `reset_peak_memory_stats` to aggressively reclaim VRAM between runs.
- For diffusers, `_orig_mod` is used to inspect the forward signature after
  `torch.compile` wraps the module in `OptimizedModule`.

---

### `traces/analyze_traces.py`

First-pass coarse analysis: loads the six trace files, classifies every GPU
kernel into a high-level category, and prints a per-category comparison table.
Useful for a quick overview; operates on all six hard-coded files at once.

**Usage**

```bash
python traces/analyze_traces.py
```

Prints three comparison tables (one per input shape) and a cross-config analysis
using flux2-t2i as reference, plus a "Top 15 kernel types" section for each
framework/config combination.

---

### `traces/analyze_detailed.py`

Per-trace analysis with a comparison mode.  Classifies GPU kernels into six
groups that are meaningful across both MAX and diffusers (torch.compile) traces,
and reports latency + percentage share for each group.

**Groups**

| Group | MAX kernels | Diffusers kernels |
|-------|-------------|-------------------|
| attention | `nn_mha_sm100_2q_SM100MHA2Q_ke*` | `flash_fwd_kernel` |
| matmul | `nvjet_*`, cutlass | (same) |
| rope | rotary/rope kernels | triton\_\* with "rotary"/"rope" |
| normalization | `nn_normalization_rms_norm_gpu`, `nn_normalization_layer_norm_gp*` | triton\_\* with "norm"/"rsqrt" |
| elementwise | `std_algorithm_functional__ele*`, `nn_concat__*` | triton\_\* (activation, concat, scale) |
| else | memory ops, unknown | memory ops, unknown |

For diffusers, normalization and elementwise are typically fused into a single
triton kernel.  The classifier inspects the kernel name to assign it to the
correct group so the totals are comparable with MAX.

**CLI**

```
TRACE [TRACE ...]    One or two Chrome trace JSON files
--labels LABEL ...   Short labels (inferred from filename if omitted)
--num-iters N        Profiled iterations in the trace (default: 3)
```

**Usage — single trace**

```bash
python traces/analyze_detailed.py traces/transformer_max_flux1.json
```

```
============================================================
  Trace : transformer_max_flux1.json
  Label : max
  Total : 102.90 ms/iter
============================================================
Group               ms/iter   % total
-------------------------------------
attention             12.28     11.9%
matmul                34.29     33.3%
rope                   0.00      0.0%
normalization          6.40      6.2%
elementwise           48.76     47.4%
else                   1.18      1.1%
-------------------------------------
TOTAL                102.90    100.0%
```

**Usage — comparison**

```bash
python traces/analyze_detailed.py \
    traces/transformer_diffusers_flux1.json \
    traces/transformer_max_flux1.json \
    --labels diffusers max
```

Prints individual breakdowns for each trace followed by a side-by-side table
and a gain/loss summary:

```
────────────────────────────────────────────────────────────
  Gain / loss summary  (diffusers → max)
────────────────────────────────────────────────────────────
  Attention gain        :   +31.30 ms  (43.58 → 12.28)
  Unfused overhead      :   +46.06 ms  (9.10 → 55.15)
    normalization delta :    +3.69 ms  (2.71 → 6.40)
    elementwise delta   :   +42.37 ms  (6.39 → 48.76)
  Net (B - A)           :   +16.25 ms  (max slower by 16.25 ms)
```

This summary is designed for tracking optimization progress: as MAX fuses
elementwise and normalization kernels, the "Unfused overhead" line should
decrease while "Attention gain" remains constant, moving the net delta negative
(MAX faster).

---

## Key Findings

### Transformer-only latency (avg of iterations 1+, ms)

| Config | diffusers | MAX | Ratio |
|--------|----------:|----:|------:|
| flux1 | 87.9 | 106.8 | MAX 1.22× slower |
| flux2-t2i | 260.2 | 282.6 | MAX 1.09× slower |
| flux2-i2i | 614.7 | 579.9 | MAX **1.06× faster** |

### The crossover mechanism

Two opposing forces determine whether MAX wins or loses versus diffusers (with
`torch.compile`):

**MAX advantage — SM100 attention kernel**

MAX uses a custom SM100 MHA2Q kernel that is ~3.5× faster than PyTorch's
`flash_fwd_kernel`:

| Config | diffusers attn | MAX attn | MAX saves |
|--------|---------------:|----------:|----------:|
| flux1 | ~39 ms | ~8 ms | ~31 ms |
| flux2-t2i | ~93 ms | ~26 ms | ~67 ms |
| flux2-i2i | ~185 ms | ~52 ms | ~133 ms |

**MAX disadvantage — unfused elementwise/concat/norm kernels**

`torch.compile` fuses many small ops (normalization, activation, concat) into
single Triton kernels. MAX currently issues these as separate CUDA kernels:

| Config | diffusers (triton fused) | MAX (unfused overhead) |
|--------|-------------------------:|-----------------------:|
| flux1 | ~23 ms | ~78 ms (+55 ms) |
| flux2-t2i | ~62 ms | ~92 ms (+30 ms) |
| flux2-i2i | ~100 ms | ~161 ms (+61 ms) |

The crossover occurs because the attention savings grow faster with sequence
length than the unfused overhead:

- flux1: saves 31 ms, loses 55 ms → MAX slower by ~24 ms
- flux2-t2i (reference): saves 67 ms, loses 30 ms → MAX slower by ~23 ms
- flux2-i2i: saves 133 ms, loses 61 ms → MAX **faster** by ~72 ms

### Kernel names for trace inspection

When opening traces in `chrome://tracing` or Perfetto, look for:

| Op | diffusers kernel name | MAX kernel name |
|----|----------------------|-----------------|
| Attention | `flash_fwd_kernel` | `nn_mha_sm100_2q_SM100MHA2Q_ke*` |
| Fused norm+elem | `triton_*` (various) | `nn_normalization_rms_norm_gpu`, `std_algorithm_functional__ele*` |
| Concat | `triton_*cat*` | `nn_concat__fused_concat_inner` |
