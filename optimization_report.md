# FLUX.2-klein-4B Optimization Report

## Executive Summary (2026-03-06)

### Current scorecard
- Current best inner MAX checkpoint:
  - Latest local validation after commit `80f671dd66` (`perf(klein): replace tiled kv repeat in text encoder`)
  - `component/text_encoder`: **7.895 ms**
  - `encode_prompt`: **8.256 ms**
  - `E2E execute`: **249.955 ms**
- Current served MAX benchmark after switching response encoding to JPEG:
  - `/workspace/modular/logs/max_serve_flux2_klein_jpeg_apples_to_apples_20260306.txt`
  - strict `1` warmup + `3` timed requests: **358.819 ms/image**
  - confirmed hot steady-state (`run4`-`run6`): **277.762 ms/image**
- Current `sglang` targets:
  - text-encoder target `/workspace/modular/logs/sglang_text_encoder_probe_nooffload.json`: **37.102 ms**
  - same-run in-server request time: **231.732 ms**
  - broader steady-state served target from the earlier benchmark: **315.924 ms/image**

### Current gaps
- MAX hot served E2E vs `sglang` served E2E: **-38.162 ms**
- MAX strict `1w+3t` served E2E vs `sglang` served E2E: **+42.895 ms**
- MAX hot served E2E vs current MAX inner `E2E execute`: **+27.807 ms**
- MAX strict `1w+3t` served E2E vs current MAX inner `E2E execute`: **+108.864 ms**
- MAX `component/text_encoder` vs current `sglang` text-encoder target: **-29.207 ms**

### What is already fixed
- The large eager overhead around `encode_prompt` is no longer on the hot path.
- The redundant token `to(device).cast(int64)` step was removed.
- Qwen3 hidden-state emission is limited to the three layers Klein actually uses.
- The temporary `encode_prompt` > `component/text_encoder` split after recent text-encoder work was a profiling artifact caused by the lazy-compile guard in `pipeline_flux2_klein.py`; that issue is fixed.
- Replacing tiled KV replication in Qwen3 text attention with concat-based replication cut text encoder latency from roughly **104 ms** to roughly **8 ms**.
- Switching served image responses from PNG to JPEG removed the dominant worker-side response packaging cost and reduced payload size from **1,771,053** bytes to **120,774** bytes.
- Token padding remains matched at 512 tokens in both MAX and `sglang`; token compaction is intentionally out of scope because it affects FLUX.2-Klein output quality.

### Current conclusion
- The text encoder is no longer the bottleneck on the current inner MAX checkpoint.
- The old served E2E gap was dominated by PNG response packaging in the worker path.
- With JPEG output, hot served MAX requests now run in the **268-286 ms** band and are below the earlier `sglang` served baseline of **315.924 ms/image**.
- The first timed JPEG request after warmup is still elevated, so the report keeps both the strict `1w+3t` average and the confirmed hot steady-state average.
- Highest-priority follow-ups are:
  - explain or eliminate the still-elevated first post-warmup JPEG request
  - decide whether JPEG is acceptable as the default OpenResponses surface for FLUX.2-Klein
  - compare image quality / compatibility expectations between MAX JPEG output and `sglang`
  - keep the scorecard on served E2E rather than inner `--profile-timings`

## Update (2026-03-06): MAX Serve JPEG Benchmark

### Goal
- Validate whether the remaining served E2E gap is mainly response packaging cost by changing only the output encoding format in the pixel-generation wrapper.

### Inputs
- JPEG served benchmark:
  - `/workspace/modular/logs/max_serve_flux2_klein_jpeg_apples_to_apples_20260306.txt`
- Historical PNG served benchmark:
  - `/workspace/modular/logs/max_serve_flux2_klein_apples_to_apples_20260306.txt`
- `sglang` served baseline:
  - steady-state client-observed latency: **315.924 ms/image**

### Code change
- File:
  - `max/python/max/pipelines/lib/pipeline_variants/pixel_generation.py`
- Change:
  - `OutputImageContent.from_numpy(img, format="png")`
  - ->
  - `OutputImageContent.from_numpy(img, format="jpeg")`

### Setup
- Server launch:
  - `MAX_SERVE_API_TYPES='["responses"]' MAX_SERVE_DISABLE_TELEMETRY=1 MAX_SERVE_LOGS_CONSOLE_LEVEL=DEBUG ./bazelw run //max/python/max/entrypoints:pipelines -- --log-level DEBUG serve --model black-forest-labs/FLUX.2-klein-4B --devices gpu:0 --port 8013 --max-batch-size 1`
- Request surface:
  - `POST http://127.0.0.1:8013/v1/responses`
- Request body:
  - `model=black-forest-labs/FLUX.2-klein-4B`
  - `input="A cat holding a sign that says hello world"`
  - `seed=42`
  - `provider_options.image.guidance_scale=1.0`
  - `provider_options.image.height=1024`
  - `provider_options.image.width=1024`
  - `provider_options.image.steps=4`
  - `provider_options.image.num_images=1`

### Results
- warmup: **2045.083 ms**
- run1: **535.065 ms**
- run2: **267.715 ms**
- run3: **273.678 ms**
- strict `1w+3t` avg: **358.819 ms**
- additional hot requests:
  - run4: **286.289 ms**
  - run5: **268.807 ms**
  - run6: **278.189 ms**
- confirmed hot avg (`run4`-`run6`): **277.762 ms**
- hot avg (`run2`-`run6`): **274.936 ms**
- response payload size: **120,774 bytes**

### Request timing split
- Timed hot request `run2` (`request_id=8040799c435249d7b93df9be3ff28738`) from DEBUG logs:
  - handler start -> worker start: about **15 ms**
  - worker start -> worker complete: about **249 ms**
  - worker complete -> handler complete: about **3 ms**
- This matches the inner `E2E execute = 249.955 ms` checkpoint closely, which means the served path is no longer paying a large extra worker-side packaging penalty once JPEG is used.

### Interpretation
1. The old served gap was real, but it was mostly response encoding.
- Historical PNG served average: **709.392 ms**
- JPEG hot served average: **277.762 ms**
- Improvement: about **431.630 ms**

2. Hot served MAX is now below the earlier `sglang` served baseline.
- `277.762 - 315.924 = -38.162 ms`
- The result is now in the same regime as the optimized inner checkpoint rather than hundreds of milliseconds above it.

3. The first timed JPEG request is still elevated.
- Strict `1w+3t` average is **358.819 ms**, but the next five timed requests settle into **~275 ms**.
- This suggests one-time response-path initialization or encoder warmup that is not covered by the original single warmup request.

4. Payload size changed dramatically.
- Historical PNG payload: **1,771,053 bytes**
- JPEG payload: **120,774 bytes**
- That reduction aligns with the measured latency drop and explains why route-side JSON / HTTP overhead also became negligible.

## Historical Update (2026-03-06): MAX Serve PNG Benchmark

### Goal
- Compare MAX and `sglang` on the same served surface instead of mixing MAX inner `--profile-timings` with `sglang` client-observed request latency.

### Inputs
- MAX served benchmark:
  - `/workspace/modular/logs/max_serve_flux2_klein_apples_to_apples_20260306.txt`
- `sglang` served baseline:
  - steady-state client-observed latency: **315.924 ms/image**
  - methodology matched the earlier benchmark: `1` warmup, `3` timed requests

### MAX serve setup
- Server launch:
  - `MAX_SERVE_API_TYPES='["responses"]' MAX_SERVE_DISABLE_TELEMETRY=1 ./bazelw run //max/python/max/entrypoints:pipelines -- serve --model black-forest-labs/FLUX.2-klein-4B --devices gpu:0 --port 8010 --max-batch-size 1`
- Request surface:
  - `POST http://127.0.0.1:8010/v1/responses`
- Request body:
  - `model=black-forest-labs/FLUX.2-klein-4B`
  - `input="A cat holding a sign that says hello world"`
  - `seed=42`
  - `provider_options.image.guidance_scale=1.0`
  - `provider_options.image.height=1024`
  - `provider_options.image.width=1024`
  - `provider_options.image.steps=4`
  - `provider_options.image.num_images=1`
- Methodology:
  - wait for `/health`
  - send `1` warmup request
  - send `3` timed requests
  - measure full client-observed latency including HTTP response read

### Results
- warmup: **2083.854 ms**
- run1: **671.812 ms**
- run2: **673.606 ms**
- run3: **782.758 ms**
- avg: **709.392 ms**
- response payload size: **1,771,053 bytes**

### Interpretation
1. The inner optimization succeeded, but it does not close served E2E.
- Current MAX inner `E2E execute` is **249.955 ms**, which is below the earlier `sglang` served baseline of **315.924 ms**.
- Current MAX served latency is **709.392 ms**, so there is about **459.437 ms** outside the inner `E2E execute` surface on the MAX served path.

2. The served gap is still basically the same residual gap we were seeing before.
- `MAX serve - sglang serve = 709.392 - 315.924 = 393.468 ms`
- This is effectively the same order of magnitude as the earlier post-JPEG diagnostic gap (`~392 ms`), which suggests the remaining problem is still outside the inner diffusion model timing.

3. The scoreboard should now prioritize served E2E.
- `--profile-timings` remains useful for inner-model attribution.
- It is no longer sufficient as the main scorecard once inner `E2E execute` is below the `sglang` served baseline but `max serve` is still much slower.
- The next optimization target is the served request path: pipeline wrapper, response construction, image materialization / serialization, and any host synchronization not visible in inner timing.

## Historical Update (2026-03-06): Text Encoder Nsight Comparison Before `_repeat_kv` Fix

This section is preserved for pre-fix attribution only. It predates the current checkpoint where text encoder latency is about **7.895 ms** and served E2E is the main remaining problem.

### Goal
- Understand the remaining prompt-encoding gap after the eager `encode_prompt` fixes.
- Keep the padded 512-token path on both MAX and `sglang`, since trimming padding affects FLUX.2-Klein output quality.

### Inputs (current text-encoder pass)
- Plain MAX timing anchor:
  - `/workspace/modular/logs/max_text_encoder_probe_refresh.log`
- Plain `sglang` timing anchor with offload disabled:
  - `/workspace/modular/logs/sglang_text_encoder_probe_nooffload.json`
- Warm-window `nsys` captures (`start-later` after warmup):
  - `/workspace/modular/logs/max_text_encoder_compare.nsys-rep`
  - `/workspace/modular/logs/max_text_encoder_compare.log`
  - `/workspace/modular/logs/sglang_text_encoder_compare.nsys-rep`
  - `/workspace/modular/logs/sglang_text_encoder_compare.log`
  - `/workspace/modular/logs/sglang_text_encoder_compare_perf.json`
- Full-run `nsys` captures used for kernel-family attribution:
  - `/workspace/modular/logs/max_text_encoder_full_detailed.nsys-rep`
  - `/workspace/modular/logs/max_text_encoder_full_detailed.log`
  - `/workspace/modular/logs/sglang_text_encoder_full_autograd.nsys-rep`
  - `/workspace/modular/logs/sglang_text_encoder_full_autograd.log`
  - `/workspace/modular/logs/sglang_text_encoder_full_autograd_perf.json`

### Methodology notes
- Token padding is matched:
  - MAX FLUX.2-Klein keeps `padding="max_length"` and preserves all 512 tokens.
  - `sglang` FLUX.2-Klein also uses `padding="max_length"` and `max_length=512`.
- `sglang` text-encoder and DiT offload were disabled for the target comparison path.
- The `start-later` `nsys` captures isolated the warmed request, but on this machine they did not retain CUDA GPU kernel tables.
  - Treat those captures as useful for warm-request boundary confirmation and CUDA API sanity only.
  - Do not use them for kernel-family attribution.
- The full-run `nsys` captures do retain GPU kernels, but they perturb absolute stage timings:
  - MAX with `MODULAR_ENABLE_PROFILING=detailed` changed `component/text_encoder` from **135.104 ms** (plain) to **74.733 ms**.
  - `sglang` with `--pytorch=autograd-nvtx` changed `TextEncodingStage` from **37.102 ms** (plain) to **61.810 ms**.
- Therefore:
  - use plain MAX / `sglang` runs for latency targets
  - use full-run `nsys` only for kernel-family attribution
- `--pytorch=functions-trace` was not usable for `sglang` here.
  - It crashed the Qwen3 encoder with `'tuple' object has no attribute 'requires_grad'`, so `autograd-nvtx` was used instead.

### Latency anchors used in this comparison
- Current plain MAX checkpoint:
  - `component/text_encoder`: **135.104 ms**
  - `encode_prompt`: **135.435 ms**
  - `E2E execute`: **374.941 ms**
- Best recent MAX checkpoint before the current regression:
  - `component/text_encoder`: **77.743 ms**
  - `encode_prompt`: **78.091 ms**
  - `E2E execute`: **317.530 ms**
- `sglang` plain `generate --perf-dump-path` with offload disabled:
  - `TextEncodingStage`: **37.102 ms**
  - `total_duration_ms`: **231.732 ms**
- Current plain text-encoder gaps:
  - current MAX `component/text_encoder` - best earlier MAX `component/text_encoder` = **57.361 ms**
  - current MAX `component/text_encoder` - `sglang` `TextEncodingStage` = **98.002 ms**

### Main findings
1. Padding is not the cause of the remaining gap.
- Both stacks are running the padded 512-token path.
- Do not pursue token compaction for Klein; it changes output quality.

2. The earlier prompt-path regression has been removed.
- The transient split between `encode_prompt` and `component/text_encoder` after the recent text-encoder edits was a profiling / lazy-compile interaction, not a true hot-path cost.
- That measurement issue is now fixed and should not be used as an optimization target.

3. The current gap is no longer an eager prompt-path problem.
- In the fresh MAX run, `encode_prompt` (**135.435 ms**) is effectively the same as `component/text_encoder` (**135.104 ms**).
- That means the earlier large non-encoder overhead inside `encode_prompt` has been removed from the hot path.

4. I do not see an explicit host-materialization step inside the MAX text-encoder hot path.
- The current MAX `encode_prompt` path stays on device through Qwen3 forward and prompt-embedding assembly.
- The meaningful explicit device-to-host boundary remains later in image decode / postprocess, not inside prompt encoding.

5. The remaining text-encoder gap is device-side and attention / norm heavy.
- MAX text-encoder attention still uses a generic assembly in:
  - `max/python/max/pipelines/architectures/qwen3/text_encoder/layers/attention.py`
- `sglang` uses a more specialized Qwen3 encoder stack in:
  - `/workspace/modular/.venv-sglang-bench/lib/python3.12/site-packages/sglang/multimodal_gen/runtime/models/encoders/qwen3.py`
- This is consistent with the current regression from **77.743 ms** back to **135.104 ms** on MAX.

### Code-path comparison summary
- MAX attention path:
  - manual fused QKV matmul from three `Linear` weights
  - separate Q / K RMSNorm
  - generic RoPE application
  - explicit KV replication via `_repeat_kv()` (`tile + reshape`)
  - generic `flash_attention_gpu(...)`
- MAX block / MLP path:
  - gate / up already merged into one matmul in `max/python/max/pipelines/architectures/qwen3/text_encoder/qwen3.py`
  - residual add and RMSNorm are still not fused across the block
- `sglang` attention path:
  - `QKVParallelLinear`
  - `LocalAttention`
  - fused rotary / QK-norm-related kernels in the trace
  - no explicit Python-level KV repeat path like MAX `_repeat_kv()`
- `sglang` block / MLP path:
  - `MergedColumnParallelLinear` for gate+up
  - residual-carry RMSNorm path in the decoder block

### Nsight kernel evidence (attribution only, not scorekeeping)
- MAX full-run `nsys` still shows generic text-encoder-relevant kernel families:
  - `nn_mha_sm100_kernel_SM100MHA...`: **43.424 ms** across **200** calls
  - `nn_normalization_rms_norm_gpu...`: **10.483 ms** across **320** calls
  - `nn_concat__fused_concat_inner...`: **1.252 ms** across **54** calls
- `sglang` full-run `nsys` shows more specialized encoder kernels:
  - `pytorch_flash::flash_fwd_kernel...`: **77.301 ms** across **125** calls
  - `flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel...`: **2.138 ms** across **125** calls
  - `_rms_norm_tiled_onepass`: **1.620 ms** across **344** calls
  - `flashinfer::norm::FusedAddRMSNormKernel...`: **0.386 ms** across **144** calls
  - `fused_qknorm_warp`: **0.684 ms** across **50** calls
- Interpretation:
  - `sglang` is spending its text-encoder time in specialized Flash / FlashInfer / fused norm kernels.
  - MAX is still leaning on a more generic MHA + RMSNorm + concat / layout stack.

### What this means for optimization priority
1. Keep the padded 512-token path unchanged.
- It is matched across MAX and `sglang`, and compacting tokens is not acceptable for Klein quality.

2. The next optimization target is MAX attention, not tokenization.
- Best candidate: eliminate explicit KV replication in `max/.../attention.py`.
- Second candidate: fuse or specialize QK-norm + rotary preparation.

3. Residual + RMSNorm fusion is the next block-level win.
- `sglang` trace exposes fused residual/norm behavior that MAX does not currently match.

4. MLP is no longer the main bottleneck.
- MAX already has the merged gate/up matmul.
- `sglang` also uses a merged gate/up path, so the remaining gap is now mostly attention + norm.

5. Use plain runs as the benchmark score.
- Keep using plain `--profile-timings` / `--perf-dump-path` numbers to measure progress.
- Do not use `nsys`-instrumented timings themselves as the latency scoreboard.

## Historical Update (2026-03-06): Prompt Encoding Target Before Encode-Prompt Fixes

These numbers are preserved as the pre-fix prompt-path baseline. They are no longer the current prompt-encoding state.

### Inputs
- MAX profiling log with expanded method coverage:
  - `/workspace/modular/logs/max_profile_timings_encode_prompt_latest.log`
- `sglang` warmed stage-metrics dumps:
  - `/workspace/modular/logs/sglang_encode_prompt_validate.json`
  - `/workspace/modular/logs/sglang_encode_prompt_run2.json`
  - `/workspace/modular/logs/sglang_encode_prompt_run3.json`

### Historical checkpoint values
- MAX `encode_prompt`: **445.872 ms**
- MAX `component/text_encoder`: **148.620 ms**
- `sglang` `TextEncodingStage` (3 warmed runs): **37.586 ms avg**
  - Per run: **37.746 ms**, **37.274 ms**, **37.739 ms**
- MAX `encode_prompt` gap vs `sglang` `TextEncodingStage`: **408.286 ms**
- MAX `component/text_encoder` gap vs `sglang` `TextEncodingStage`: **111.034 ms**
- Non-text-encoder overhead currently inside MAX `encode_prompt`: **297.252 ms**

### Interpretation update
- At that point, the prompt-encoding gap was not primarily the text encoder forward itself.
- MAX was spending about **148.620 ms** in `component/text_encoder`, but **445.872 ms** in `encode_prompt`, leaving about **297 ms** in surrounding prompt-path work.
- That made `encode_prompt` the next concrete decomposition target after end-to-end and decode analysis.
- The next breakdown needed to isolate:
  - tokenization / text preprocess
  - token tensor materialization and H2D / cast
  - text encoder forward
  - prompt embedding postprocess

## Historical Update (2026-03-06): Earlier E2E Target and Response-Format Diagnostic

These numbers are preserved as an earlier end-to-end diagnostic checkpoint. They predate the later prompt-encoding optimizations and the current plain MAX `E2E execute = 374.941 ms` anchor above.

### Optimization target
- Target steady-state latency is the current `sglang` baseline: **315.924 ms/image**.
- The MAX optimization loop is the `E2E execute` output from:
  - `./bazelw run //max/examples/diffusion:simple_offline_generation -- --model black-forest-labs/FLUX.2-klein-4B --prompt "A cat holding a sign that says hello world" --num-inference-steps 4 --guidance-scale 1.0 --seed 42 --num-warmups 1 --profile-timings`

### Historical benchmark anchor
- `sglang` steady-state latency (`1` warmup, `3` timed iters): **315.924 ms/image**
- MAX diagnostic run with JPEG response encoding: **708.265 ms/image**
- Residual E2E gap vs `sglang`: **392.341 ms/image**

### Interpretation update
- The PNG/JPEG experiment did not remove the main performance problem.
- Treat the **392 ms** residual as the gap measured at that point after response-format effects were bounded.
- The JPEG response change was a diagnostic only and was reverted; the useful result is the remaining gap size, not the temporary output format.

## Update (2026-03-05): NSYS Re-Comparison with MAX Start-Later After Warmup

### Inputs (new artifacts)
- MAX (`start-later`, trigger after warmup):
  - `/workspace/modular/logs/max_flux2_klein_nsys_start_later_after_warmup.nsys-rep`
  - `/workspace/modular/logs/max_flux2_klein_nsys_start_later_after_warmup.log`
- sglang serve (`start-later`, timed request):
  - `/workspace/modular/logs/sglang_flux2_klein_serve_nsys_start_later_fix1.nsys-rep`
  - `/workspace/modular/logs/sglang_flux2_klein_serve_nsys_start_later_fix1.log`

### MAX start-later script change used for this run
- File: `/workspace/modular/profile_max_nsys_start_later.sh`
- Default trigger updated to: `START_PATTERN="Warmup complete"`
- Poll interval tightened to avoid late start miss: `POLL_INTERVAL_S=0.05`

### NSYS Side-by-Side (new baseline)

| Metric | MAX (after warmup) | sglang (timed serve request) |
|---|---:|---:|
| Capture span | 2530.053 ms | 309.684 ms |
| Kernel span | 405.670 ms | 308.210 ms |
| Kernel total / calls | 191.819 ms / 1695 | 272.076 ms / 4776 |
| CUDA runtime total / calls | 202.279 ms / 6767 | 168.572 ms / 6014 |
| CUDA sync total / calls | 184.255 ms / 1092 | 72.591 ms / 24 |
| Memcpy total / calls | 0.378 ms / 10 | 0.172 ms / 28 |
| Memcpy bytes | 27.374 MB | 3.414 MB |

### Main findings from this re-comparison

1. MAX trace quality issue is fixed.
- Previous MAX captures missed kernel data; the new after-warmup capture includes `KERNEL` and `MEMCPY` tables.

2. MAX E2E regression signature is host/synchronization-heavy.
- MAX runtime top API: `cudaDeviceSynchronize_v3020` = **181.907 ms** across 19 calls.
- MAX sync type split: `Context sync` = **181.874 ms** (dominant), `Event sync` = 1.687 ms, `Stream wait sync` = 0.694 ms.
- sglang has no `cudaDeviceSynchronize` in this trace; sync is mostly `cudaStreamSynchronize_v3020` (72.623 ms, with one large wait).

3. Kernel totals alone do not explain MAX E2E gap.
- MAX log still reports `E2E execute = 959.257 ms`, `decode_latents = 298.121 ms`, `prepare_embeddings = 239.695 ms`.
- Same run includes `iteration_total_ms = 1587.579` and `shutdown_ms = 784.342`, so capture contains substantial non-hot-path time.

4. Data-movement overhead is not dominant by time but differs in volume.
- MAX D2H in traced window: ~12.638 MB (4 calls, 0.366 ms).
- sglang D2H in traced window: ~3.279 MB (19 calls, 0.156 ms).

### Updated optimization priorities (based on latest NSYS)

1. Reduce/relocate `cudaDeviceSynchronize` on MAX hot path.
- Replace global/context sync points with stream/event-based fencing where safe.
- Confirm syncs are not inside per-step loop or decode critical path.

2. Isolate hot-path measurement window in MAX traces.
- Start after warmup and stop immediately after timed iteration to avoid shutdown/save pollution.
- Keep image save and teardown outside profiled window for apples-to-apples comparisons.

3. Continue decode-path kernel work (still important).
- Prior trace evidence for decoder conv/norm/pointwise bottlenecks remains valid.
- Host sync cleanup should be done in parallel with decoder kernel optimization.

4. Audit D2H touchpoints in MAX.
- Verify whether prompt/image output handling forces avoidable device-to-host copies during timed run.

### Current interpretation

The latest traces support this split:
- **Primary new gap driver:** host-side synchronization behavior in MAX (`cudaDeviceSynchronize` / context sync).
- **Still relevant structural gap:** decoder-side kernel efficiency (from prior kernel-family analysis).

## Update (2026-03-05): Prompt Embedding Postprocess Hot-Path De-Eagering

### Scope
- File updated: `max/python/max/pipelines/architectures/flux2/pipeline_flux2_klein.py`
- Goal: remove eager prompt-embedding postprocess ops from the text-to-image hot path (`num_images_per_prompt=1`).

### What changed
1. Added a compiled helper for prompt postprocess:
- `_postprocess_prompt_embeddings(hidden_state_0, hidden_state_1, hidden_state_2)`
- Contains `concat + unsqueeze` and is compiled via `max_compile`.

2. Added lazy compile initialization:
- `_ensure_prompt_embedding_postprocess_compiled()` compiles once on first hot-path use.

3. Routed hot path in `prepare_prompt_embeddings(...)`:
- When exactly 3 rank-2 hidden states are present and `num_images_per_prompt == 1`, the compiled helper is used.
- This removes eager `F.concat` / `F.unsqueeze` from the common text-to-image case.

### Remaining eager ops (expected for non-hot-path branches)
- Sequence-length mismatch branch still uses eager `F.concat` padding/truncation.
- `num_images_per_prompt != 1` still uses eager `F.tile` + reshape.

### Validation
- 1-step sanity run completed successfully:
  - `./bazelw run //max/examples/diffusion:simple_offline_generation -- --model /workspace/models/FLUX.2-klein-4B --prompt "A cat holding a sign that says hello world" --num-inference-steps 1 --guidance-scale 1.0 --seed 42 --num-warmups 0 --num-profile-iterations 1 --skip-save-output`
- Run produced end-to-end output without runtime errors.

### Interpretation update
- Prompt postprocess eager overhead is reduced on the single-image hot path.
- Host-side synchronization (`cudaDeviceSynchronize` / context sync) remains the primary suspected host regression lever from NSYS.

## Inputs (fresh artifacts)
- MAX full-pipeline trace:
  - `/workspace/modular/logs/max_flux2_klein_steps4_fresh.trace.json`
- sglang full-pipeline traces (warmup enabled, no offload):
  - `/workspace/modular/logs/72049fb4-a12c-409e-995e-f607d022c903-4_steps-global-rank0.trace.json.gz` (`vae-precision=fp16`)
  - `/workspace/modular/logs/5a1f81cc-8224-4c67-b103-925fcaca5f74-4_steps-global-rank0.trace.json.gz` (`vae-precision=bf16`)
- sglang logs:
  - `/workspace/modular/sglang_klein_steps4_warm_profile_fresh.log`
  - `/workspace/modular/sglang_klein_steps4_warm_profile_bf16vae.log`
- MAX VAE decode benchmark CSV (fresh):
  - `/workspace/modular/logs/max_klein_vae_decode_bench_fresh.csv`

## Confirmed Latency Split

### MAX (`simple_offline_generation`, steps=4, guidance=1.0)
- `component/vae.decode`: **458.755 ms/image** (1 call)
- `component/transformer`: **43.455 ms/call** (4 calls)
- `decode_latents`: **682.371 ms/image**
- `E2E execute`: **1448.481 ms/image**

### sglang (`--warmup=true --dit-cpu-offload=false --text-encoder-cpu-offload=false`)
- `TextEncodingStage`: **0.0303 s** (`fp16` VAE run), **0.0255 s** (`bf16` VAE run)
- `DenoisingStage`: **0.1473 s** (`fp16`), **0.1444 s** (`bf16`)
- `DecodingStage`: **0.0098 s** (`fp16`), **0.0078 s** (`bf16`)
- Warmed-up request: **1.87 s** (`fp16`), **1.82 s** (`bf16`)

Interpretation: MAX gap remains decode/VAE-heavy.

## VAE Decode Benchmark (MAX) and Dispatch Interpretation

From `/workspace/modular/logs/max_klein_vae_decode_bench_fresh.csv`:
- `steady_sync_each`: **457.614 ms**
- `stream_total`: **455.824 ms**
- `stream_dispatch`: **374.590 ms**
- `stream_wait`: **81.234 ms**

Important nuance:
- With short loops (`iters=3`), dispatch was ~**8.6 ms** and wait ~**447.5 ms**.
- With longer loops (`iters=20`), dispatch includes queue backpressure and absorbs most compute time.

Conclusion: for single-image latency, this is still primarily GPU kernel time, not pure host launch overhead.

## Kernel-Level Comparison (fresh traces, per image)

MAX (`max_flux2_klein_steps4_fresh.trace.json`)
- Kernel total: **722.394 ms**
- Launch API: **9.176 ms** across **3444** launch calls
- Family split:
  - `vae_conv`: **375.981 ms**
  - `vae_norm`: **133.433 ms**
  - `vae_pointwise`: **83.331 ms**
  - `layout_cat`: **24.614 ms**
  - `attention`: **22.219 ms**

sglang (`72049...trace.json.gz`, fp16 VAE)
- Kernel total: **204.598 ms**
- Launch API: **8.559 ms** across **2347** launch calls

Delta (MAX - sglang):
- Total kernel time: **+517.796 ms**
- `vae_conv`: **+375.981 ms**
- `vae_norm`: **+129.805 ms**
- `vae_pointwise`: **+35.336 ms**
- `layout_cat`: **+16.477 ms**
- Launch API: **+0.618 ms** (not dominant)

Top MAX kernels:
1. `precomputed_convolve_sgemm...`: **286.565 ms**
2. `implicit_convolve_sgemm...`: **89.416 ms**
3. `nvjet_tst_256x256...`: **39.464 ms**
4. `nn_normalization_group_norm...`: **~38 ms** (multiple kernels)
5. `std_algorithm_functional__...`: **26.525 ms**

## Updated Optimization Priorities (klein-specific)

1. Fix decoder conv backend path first.
- Why: `precomputed_convolve_sgemm + implicit_convolve_sgemm` alone are ~**376 ms**.
- Target: move decoder convs onto a faster backend/layout path for these shapes.

2. Optimize GroupNorm path in decoder.
- Why: `vae_norm` is ~**133 ms** in MAX and far above sglang.
- Target: reduce GroupNorm cost via kernel improvements or fusion around norm/activation.

3. Fuse decoder pointwise chains.
- Why: `vae_pointwise` is ~**83 ms** and spread across many kernels.
- Target: reduce kernel count and memory traffic in SiLU/add/mul-style chains.

4. Reduce layout/concat churn.
- Why: `layout_cat` still contributes ~**25 ms**.
- Target: keep stable tensor layout through decoder blocks and avoid avoidable permute/concat ops.

5. Treat launch count as secondary.
- Why: launch overhead delta is small compared to compute delta.
- Target: launch-count reductions are useful but not a primary lever for current gap.

## Validation Loop

For each candidate patch:
1. Run:
   - `./bazelw run --config=disable-mypy //max/examples/diffusion:benchmark_flux2_vae_decode_latency -- ...`
2. Re-run full pipeline trace:
   - `./bazelw run //max/examples/diffusion:simple_offline_generation -- ... --dump-trace ...`
3. Compare against current fresh baseline:
   - MAX kernel total: **722.394 ms**
   - MAX `component/vae.decode`: **458.755 ms**
4. Accept only if both decode and end-to-end improve.

## sglang Serve nsys Profiling Runbook (2026-03-05)

Use this flow to collect an `nsys` trace for FLUX.2-klein-4B via `sglang serve` with one warmup and one timed request.

Important environment note for this machine:
- Use `--attention-backend torch_sdpa`.
- Default FA4/CuTe path failed here due CUDA toolkit requirement (`CuTe Experimental module is only supported on Cuda toolkit 13.1 and above`).

### 0) Prerequisites (install once)

Verify `nsys`:

```bash
which nsys
nsys --version
```

Install Python deps required to run local `/workspace/sglang` serve for this profiling flow:

```bash
python -m pip install \
  torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 \
  sgl-kernel==0.3.21 quack-kernels==0.2.4 \
  diffusers==0.36.0 transformers==4.57.6 openai==2.6.1 \
  tqdm pybase64 aiohttp fastapi uvicorn uvloop \
  pydantic pyzmq psutil orjson msgspec \
  partial_json_parser interegular outlines==0.1.11 llguidance xgrammar==0.1.27 \
  python-multipart setproctitle nvidia-ml-py \
  blobfile datasets compressed-tensors gguf scipy \
  remote-pdb imageio==2.36.0 imageio-ffmpeg==0.5.1 moviepy \
  opencv-python-headless==4.10.0.84 runai_model_streamer cache-dit==1.2.0 addict
```

### 1) Start sglang serve under nsys

Run in terminal A:

```bash
nsys profile \
  --trace=cuda,cublas,osrt,nvtx \
  --trace-fork-before-exec=true \
  --output /workspace/modular/logs/sglang_flux2_klein_serve_nsys_torchsdpa \
  --force-overwrite=true \
  env PYTHONPATH=/workspace/sglang/python \
  python3 -m sglang.multimodal_gen.runtime.entrypoints.cli.main serve \
    --model-path /workspace/models/FLUX.2-klein-4B \
    --host 127.0.0.1 --port 30000 \
    --num-gpus 1 --tp-size 1 \
    --backend auto \
    --attention-backend torch_sdpa \
    --dit-precision bf16 --vae-precision fp16 \
    --dit-cpu-offload=false --text-encoder-cpu-offload=false
```

Wait until server shows:
- `Uvicorn running on http://127.0.0.1:30000`

### 2) Send warmup + timed request

Run in terminal B:

```bash
python3 benchmark_sglang_flux2.py \
  --base-url http://127.0.0.1:30000 \
  --endpoint images \
  --model /workspace/models/FLUX.2-klein-4B \
  --prompt "A cat holding a sign that says hello world" \
  --steps 4 \
  --guidance-scale 1.0 \
  --seed 42 \
  --warmup 1 \
  --iters 1 \
  --save-last-image /workspace/modular/output_sglang_nsys.png \
  --save-last-response-json /workspace/modular/logs/sglang_flux2_klein_serve_nsys_torchsdpa_last_response.json
```

Expected output shape:
- `warmup 1/1: ... status=200, images=1, ok`
- `iter 1/1: ... status=200, images=1, ok`

### 3) Stop server and finalize report

Back in terminal A, press `Ctrl+C`.

`nsys` then writes:
- `/workspace/modular/logs/sglang_flux2_klein_serve_nsys_torchsdpa.nsys-rep`

Companion artifacts:
- `/workspace/modular/output_sglang_nsys.png`
- `/workspace/modular/logs/sglang_flux2_klein_serve_nsys_torchsdpa_last_response.json`

### Latest measured latency (this run)

- Warmup: `2109.693 ms`
- Timed iteration: `333.757 ms`
