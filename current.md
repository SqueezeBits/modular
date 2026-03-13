# WAN Pipeline — Current Status (2026-03-13)

## Branch / Commit
- Branch: `add/wan-pipeline/full-pipeline`
- Last pushed commit: `05c2f4f8ee`
  - `[Pipelines] Fix Wan video generation and startup`

## What is already committed/pushed
The pushed commit includes:
- Wan video generation path working end-to-end
- `simple_offline_generation` video-argument delegation to video entrypoint
- Wan VAE dtype/runtime bug fix (`ops.constant` bf16 mismatch)
- Wan startup cleanup / component-owned load-model structure
- shared session wiring for Wan components
- startup latency improvements:
  - `transformer_2` moved off startup critical path
  - VAE prewarm improvements
- Wan tests updated and passing

## Current uncommitted worktree state
These are **NOT committed yet**:
- `max/examples/diffusion/BUILD.bazel` (adds comparison targets)
- `max/examples/diffusion/wan_compare_against_diffusers.py` (diffusers vs MAX comparison script)
- `max/examples/diffusion/wan_transformer_block_compare.py` (deeper block-level comparison script)

Do **not** assume these scripts exist on another server unless copied/cherry-picked.

## End-to-end runtime status
### Successful runs
- smoke run (video entrypoint): success
- smoke run via image entrypoint delegation: success
- full run: success

Generated artifacts in this workspace:
- `smoke_t2v.mp4`
- `smoke_t2v_refactored.mp4`
- `smoke_t2v_via_image_entrypoint.mp4`
- `smoke_t2v_overlap.mp4`
- `smoke_t2v_startup_optimized.mp4`
- `smoke_t2v_startup_optimized2.mp4`
- `t2v_output.mp4`

### Tests/builds that passed recently
- `./bazelw test //max/tests/integration/architectures/wan:wan`
- `./bazelw build //max/examples/diffusion:simple_offline_generation //max/examples/diffusion:simple_offline_video_generation`

## Comparison work summary
### Baseline reduced diffusers vs MAX comparison
Artifacts:
- `outputs/20260313_042629/`
  - `comparison_report.json`
  - `report_ko.md`
  - paired `diffusers_*.npy` / `max_*.npy`

Reduced settings used for parity:
- `height=160`
- `width=256`
- `num_frames=5`
- `num_inference_steps=4`
- `guidance_scale=4.0`
- `guidance_scale_2=3.0`

Main findings from baseline:
- Scheduler matches exactly
- Text encoder difference is small (~2–3% relative L2)
- VAE on the same diffusers denorm latents is relatively close (~2.9% relative L2)
- Main divergence is in DiT and accumulates into final latents/output
- Example baseline numbers:
  - `dit.step_0.guided` rel L2: `4.106695e-02`
  - `dit.step_2.guided` rel L2: `7.781990e-02` (after one modulation/residual experiment)
  - `pipeline.final_decoded_output` rel L2: `2.654759e-01` (best observed from that experiment)

### Deep block-level comparison
Artifacts:
- `outputs/20260313_053536/`
  - `block_comparison_report.json`
  - `block_report_ko.md`

Important conclusions from block-level analysis:
- First noticeably divergent block in high-noise stage: block `15`
- First noticeably divergent block in low-noise stage: block `12`
- Pre-block inputs are still very close:
  - `pre hidden` rel L2 ~ `7e-06`
  - `timestep_proj` rel L2 ~ `0.003`
  - `text_emb` rel L2 ~ `0.003`
- Within the selected block, the first *visible* divergence was around `norm1_out` / `sa_input`
- However, later analysis showed LayerNorm itself is probably **not** the root cause

### Forward-hook / finer-grained block comparison
Artifacts:
- `outputs/20260313_070221/`
  - copied from runfiles into repo outputs
  - `block_comparison_report.json`
  - `block_report_ko.md`

Important conclusions:
- Using actual diffusers forward hooks reduced the chance of “wrong probe point” interpretation
- `norm1_in` is already divergent by ~1.8–2.0% relative L2 in the first problematic block
- `norm1_out` is similar magnitude to `norm1_in`
- modulation tensors themselves are much smaller error (~0.3–0.4% relative L2)
- Therefore:
  - LayerNorm is **not creating** the large error from nothing
  - the error is already entering the block from earlier blocks
  - self-attn / cross-attn / FFN then amplify it

## Experiments already tried and their outcomes
### 1) Modulation/residual float32 alignment experiment
Result:
- Improved numbers somewhat
- Suggested modulation/residual numerics matter
- But this was not kept as final committed state

### 2) LayerNorm rewrite toward diffusers FP32LayerNorm
Artifacts:
- `outputs/20260313_055839/`
  - `comparison_report.json`
  - `report_ko.md`

Result:
- Made metrics worse, not better
- Conclusion: `WanLayerNorm` formula itself is **not** the main culprit
- This patch was reverted

### 3) Cross-attention split-K/V experiment
Artifacts:
- `outputs/20260313_062139/`
  - `comparison_report.json`
  - `report_ko.md`

Result:
- Changing MAX from fused `attn2.to_kv` to split `to_k` / `to_v` gave effectively no improvement
- Conclusion: cross-attention K/V fusion is **not** the primary root cause
- This patch was reverted

## Best current interpretation
### Ruled out / lower priority
- scheduler
- text encoder
- VAE same-input decode
- LayerNorm formula itself
- cross-attention fused K/V as primary culprit

### Still most suspicious
1. **attention backend / attention numerics**
   - MAX: `flash_attention_gpu`
   - diffusers: `WanAttnProcessor` / `dispatch_attention_fn`
2. **RoPE application differences**
   - MAX rotary helper vs diffusers local rotary path
3. **attention-adjacent numeric flow across earlier blocks**
   - not block-local LayerNorm formula, but block-to-block accumulation through attention paths

## Important caution
There was concern that comparison probes might be misleading. The later block analysis used actual diffusers forward hooks and showed:
- the selected block is receiving already-divergent input (`norm1_in`)
- so “the first place we *see* it” is not necessarily the place it is *born*

Translation:
- the root cause may live in **earlier attention computation** and only become clearly visible later.

## Recommended next steps on another server
If continuing from scratch on another server, do this in order:

1. Restore or copy the uncommitted comparison scripts if needed:
   - `wan_compare_against_diffusers.py`
   - `wan_transformer_block_compare.py`
   - matching `BUILD.bazel` changes

2. Reproduce reduced comparison baseline first
   - same reduced settings (`160x256`, `5f`, `4 steps`)

3. Focus next on **attention backend / rotary path**, not LayerNorm or K/V fusion
   - compare self-attn and cross-attn outputs more directly
   - ideally compare q/k/v or post-RoPE tensors if instrumentation is feasible

4. Keep checking runtime impact
   - any parity experiment should also track whether startup / total run time regresses

## Useful artifact directories to inspect manually
- `outputs/20260313_042629/`
- `outputs/20260313_053536/`
- `outputs/20260313_055839/`
- `outputs/20260313_062139/`
- `outputs/20260313_070221/`

## Files most relevant for next debugging pass
- `max/python/max/pipelines/architectures/wan/wan_transformer.py`
- `max/python/max/pipelines/architectures/wan/model.py`
- `max/examples/diffusion/wan_compare_against_diffusers.py` (uncommitted)
- `max/examples/diffusion/wan_transformer_block_compare.py` (uncommitted)
- diffusers reference:
  - `diffusers/models/transformers/transformer_wan.py`
  - `diffusers/models/normalization.py`
  - `diffusers/models/attention.py`
