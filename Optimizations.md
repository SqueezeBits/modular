# Flux2 Pipeline Optimization Summary
This document summarizes the optimization efforts for the Flux2 pipeline on branch `ts/test-flux2-0202-local`.
---
## Overview
| Optimization | Commit | Component | Improvement |
| :--- | :--- | :--- | :--- |
| Compiled Scheduler Step | `ebd995895` | Denoising Loop | ~5x faster per step |
| Pre-compile VAE | `d2a885c49` | VAE Decode | Reduced warmup overhead |
| Fused VAE Prep Model | (this session) | VAE Decode | ~4.1s reduction per step |
**Net Result:** Total per-step latency reduced from **~9.6s** to **~5.5s**.
---
## Optimization 1: Compiled Scheduler Step (`ebd995895`)
**Problem:** The original scheduler step logic (Euler discrete flow matching) ran as Python/eager operations, incurring overhead on every denoising iteration.
**Solution:** Compiled the scheduler step logic into a MAX Graph:
- Sigma calculations (shift, flow matching)
- Latent update (`latents = prev_sample + noise_pred * dt`)
**Impact:**
- Denoising loop: **~1.5s → ~0.3s** (1 step, 1024x1024)
- Also fixed bug in patch/pack ordering that was causing noisy output.
---
## Optimization 2: Pre-compile VAE (`d2a885c49`)
**Problem:** VAE decode was being compiled on every inference call, even after warmup.
**Solution:** Pre-compile VAE model during pipeline initialization to eliminate JIT overhead during inference.
**Impact:** Reduced latency variability after warmup. VAE decode now runs at model-execution speed.
---
## Optimization 3: Fused VAE Prep Model (This Session)
**Problem:** After the above optimizations, profiling still showed ~5s spent in VAE decode block (expected: ~0.8s).
**Root Cause:** Eager operations (`F.reshape`, `F.permute`, `F.sqrt`) in:
- `_unpack_latents_with_ids`: Scatter/reshape operations
- `_unpatchify_latents`: Reshape/permute operations
- `_pack_latents`: Reshape/permute operations
Each call triggered per-call graph compilation, adding **~4.2s overhead**.
**Solution:** Implemented cached, compiled graphs:
| Method | Description |
| :--- | :--- |
| `_ensure_vae_prep_model` | Fuses unpack + BatchNorm inverse + unpatchify into single kernel |
| `_ensure_pack_model` | Compiles pack reshape/permute into single kernel |
**Handling Symbolic Tensors:** VAE BatchNorm parameters were symbolic tensors in the pipeline context. Fixed by forcing eager materialization (`x + 0.0`) before passing to compiled graph.
**Impact:**
- VAE Decode Block: **~5.0s → ~0.9s**
- Removed legacy methods: `_unpack_latents_with_ids`, `_unpatchify_latents`, `_precompute_bn_tensors`
---
## Final Benchmark Results
**Configuration:** 1024x1024, 1 step, `benchmark_flux2_vs_torch`
| Component | Before All Optimizations | After All Optimizations |
| :--- | :--- | :--- |
| Prompt Encoding | ~1.6s | ~1.6s |
| Denoising Loop | ~1.5s | ~0.3s |
| VAE Decode Block | ~5.0s | ~0.9s |
| Pipeline Overhead | ~1.5s | ~2.7s |
| **TOTAL** | **~9.6s** | **~5.5s** |
## Optimization 4: VAE Warmup & Prep Graph Fix (This Session)
**Problem 1:** VAE prep graph recompilation.
**Solution:** Added `warmup_vae` with correct packed channel count (128). Latency: ~2.8s -> ~1.7s.
**Problem 2:** "BN Prep" overhead (~928ms).
- Root Cause: `bn.running_mean` was loaded inside `with F.lazy():` in `load_model`, making it a symbolic tensor.
- This forced usage of a slow workaround (`bn + zero`) to materialize it as eager.
**Solution:**
- Removed `F.lazy()` wrapper for BN stats loading in `AutoencoderKLFlux2Model`.
- Removed `bn + zero` workaround in `Flux2Pipeline`.
**Impact:**
- BN Prep time: **928ms -> 0.02ms**.
- VAE Decode (Total pipeline): **~1.7s -> ~0.9s**.
- VAE Decode (Compute only): **~523ms** (matching standalone benchmark).
---
## Final Benchmark Results
**Configuration:** 1024x1024, 1 step, `profile_flux2_components`
| Component | Before Optimizations | After Optimizations |
| :--- | :--- | :--- |
| Prompt Encoding | ~1.6s | ~1.6s |
| Denoising Loop | ~1.5s | ~0.3s |
| VAE Decode Block | ~5.0s | **~0.9s** |
| Pipeline Overhead | ~1.5s | ~2.7s |
| **TOTAL** | **~9.6s** | **~5.5s** |
> **Note:** Conv kernel optimization (`85472d3a9d`) modified `conv.mojo` but its impact on Flux2 latency was not directly measured in this session.
---
## Eager Tensor Overhead Optimization (2026-02-05)
### Problem
Eager tensor operations (stack, reshape, cast, Tensor.full) caused ~2.3s overhead per inference.
### Optimizations Implemented
| Optimization | Component | Before | After | Improvement |
|:---|:---|:---|:---|:---|
| **Option 1:** Compiled prompt embed graph | `stack_reshape` | 942ms | 409ms | **57%** |
| **Option 2:** Fused latent prep graph | `cast+patchify+pack` | 385ms | 0.8ms | **99%** |
| **Option 3:** Cache invariant tensors | `guidance_ids` | 489ms | 0.1ms | **99%** |
### Implementation Details
1. **`_ensure_prompt_embed_model`**: Compiles `F.stack`, `F.permute`, `F.reshape` into single graph
2. **`_ensure_latent_prep_model`**: Fuses cast (float32→bfloat16) + patchify + pack
3. **`_cached_guidance` / `_cached_latent_image_ids`**: Caches tensors by shape/dtype key
### Results
| Metric | Before | After |
|:---|:---|:---|
| **Steady-state latency (3 steps)** | 4100ms | **2650ms** |
| **latent_prep overhead** | 1171ms | **164ms** |
| **Improvement** | - | **35% faster** |
### 50-Step Generation Performance
| Metric | Value |
|:---|:---|
| Total latency | **17.2s** |
| Per-step latency | **305ms** |
| Transformer | 15.3s (303ms/step) |
| VAE decode | 0.58s |
---
## Prompt Encoding Optimization Phase 2 (2026-02-05)
### Problem
`layer_extraction` in `_prepare_prompt_embeddings` was taking ~524ms (steady-state) due to:
1. Eager `F.reshape` calls triggering graph compilation (~180ms/call × 3 layers = 540ms)
2. `text_ids` recreation overhead (~188ms)
### Optimizations Implemented
| Optimization | Before | After | Improvement |
|:---|:---|:---|:---|
| **`text_ids` caching** | 188ms | 0.1ms | **99%+** |
| **Compiled unsqueeze graph** | ~180ms/call | 0.0ms | **99%+** |
| **Fast shape access** | `Tensor.shape[X].dim` | `driver_tensor.shape` | Avoids GPU sync |
### Implementation Details
1. **`_cached_text_ids`**: Caches text position ID tensors by shape/device key
2. **`_ensure_unsqueeze_model`**: Pre-compiles reshape `(seq_len, hidden_dim) → (1, seq_len, hidden_dim)`
3. **`driver_tensor.shape`**: Uses underlying buffer shape for fast access without GPU sync
### Results
| Metric | Before | After |
|:---|:---|:---|
| `layer_extraction` | 524ms | **0.3ms** |
| `prompt_encoding` (steady-state) | 1632ms | **335ms** |
| **Pipeline (3 steps)** | 2650ms | **2290ms** |
| **Pipeline (50 steps)** | 17.5s | **17.2s** |
---
# Additional Optimizations (From Local File)
# Flux2 Performance Optimizations
This document summarizes the optimization efforts undertaken to improve the performance of the Flux2 pipeline on MAX.
## 1. Fused Q/K RoPE Kernel (`fused_qk_rope_vision`)
A custom Mojo kernel was implemented to replace the inefficient two-pass `apply_rotary_emb` operation for Rotary Position Embeddings in the attention layers.
### **Optimization Strategy**
- **Fusion**: Fused the Cosine/Sine lookup and the rotation application for both Query (Q) and Key (K) tensors into a single kernel launch.
- **Elementwise Pattern**: Utilized the `elementwise` parallelization pattern in Mojo to ensure efficient execution on GPUs, leveraging the massive parallelism for the element-wise rotations.
- **Edge Case Handling**: Handled `width=1` edge cases and ensured compatibility with both `Flux2Attention` (Dual-Stream) and `Flux2ParallelSelfAttention` (Single-Stream).
- **Reduced Graph Complexity**: By replacing a subgraph of multiple operations (slice, sin, cos, mul, add, cat) with a single custom op, the overall computational graph size was reduced, significantly lowering compilation overhead.
### **Performance Impact**
| Metric | Baseline (`apply_rotary_emb`) | Optimized (`fused_qk_rope_vision`) | Improvement |
| :--- | :--- | :--- | :--- |
| **Cold Compilation Time** | ~200s | **~70s** | **~2.8x Faster** |
| **Warm Denoise Step** | ~300ms | **~280ms** | **~6.6% Faster** |
| **E2E Latency (Warm)** | 2.84s | **2.55s** | **~10% Faster** |
> **Note**: The fused kernel eliminates significant Python-side overhead and graph compilation costs, making the pipeline much more responsive, especially during the first run.