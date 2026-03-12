# Qwen-Image Current State

## Branch State

- Working branch: `add/qwen-image/merged`
- Latest merged branch commits already included:
  - `add/qwen-image/scheduler`
  - `add/qwen-image/vae`
  - `add/qwen-image/encoder`
  - `add/qwen-image/runtime`
  - `add/qwen-image/pipeline`
  - `add/qwen-image/edit`
- `merged` now also includes the latest `origin/add/qwen-image/edit` formatter cleanup.

## What Is Stable

- `Qwen/Qwen-Image-2512` runs.
- `Qwen/Qwen-Image-Edit-2511` runs.
- `bash e2e-test.sh` works on `merged`.
- Guidance defaults in `simple_offline_generation.py` are aligned for the
  Qwen image family:
  - `guidance_scale = 1.0`
  - `true_cfg_scale = 4.0` when `negative_prompt` is provided

## E2E Script

- Main script: [`e2e-test.sh`](./e2e-test.sh)
- Outputs:
  - `woman_e2e.png`
  - `man_e2e.png`
  - `output_combined_e2e.png`

## Profiling Scripts

- Short cold profile:
  - [`profile-qwen-short.sh`](./profile-qwen-short.sh)
- Warmup + profile:
  - [`profile-qwen-warm.sh`](./profile-qwen-warm.sh)
- Same-process multi-shape runner:
  - [`max/examples/diffusion/same_process_multi_shape_runner.py`](./max/examples/diffusion/same_process_multi_shape_runner.py)
  - Supports explicit `--case WIDTHxHEIGHT:STEPS`
  - Supports auto-generated case matrices via repeated
    `--shape WIDTHxHEIGHT` and `--step-count N`
  - Prints `InferenceSession.load` proxy counts during same-process runs to
    help spot compile/recompile behavior across cases
- Perfetto exporter:
  - [`tools/export_qwen_profile_perfetto.py`](./tools/export_qwen_profile_perfetto.py)
- Summary JSON exporter:
  - [`tools/export_qwen_profile_json.py`](./tools/export_qwen_profile_json.py)

## Perfetto Files Already Generated

- Cold:
  - `profiles/qwen_t2i_short.perfetto.json`
  - `profiles/qwen_edit_short.perfetto.json`
- Warm:
  - `profiles/qwen_t2i_warm.perfetto.json`
  - `profiles/qwen_edit_warm.perfetto.json`

## Key Profiling Findings

- `Qwen-Image` t2i is still the main optimization target.
- Warmup did **not** materially improve `t2i`.
  - cold span: about `25.7s`
  - warm span: about `26.6s`
- Warmup **did** materially improve `edit`.
  - cold span: about `94.8s`
  - warm span: about `35.3s`
- This strongly suggests:
  - `edit` still had significant first-run / compile / init overhead
  - `t2i` is dominated more by repeated host-driven overhead than by cold-start compile

## Most Important Numbers

### qwen_t2i_short

- runtime span: `25730.62 ms`
- total kernel time: `890.57 ms`
- HtoD memcpy: `7140.42 ms`
- launch calls: `12035`

### qwen_t2i_warm

- runtime span: `26570.89 ms`
- total kernel time: `891.37 ms`
- HtoD memcpy: `7410.16 ms`
- launch calls: `12035`

### qwen_edit_short

- runtime span: `94767.93 ms`
- total kernel time: `4287.49 ms`
- HtoD memcpy: `7655.79 ms`
- launch calls: `14468`

### qwen_edit_warm

- runtime span: `35271.03 ms`
- total kernel time: `4283.29 ms`
- HtoD memcpy: `7618.23 ms`
- launch calls: `14468`

## Interpretation

- `t2i` gap is not mainly "first compile" anymore.
- `t2i` still looks host-bound:
  - too much HtoD
  - too many small launches
  - too much orchestration outside the main denoise compute
- `edit` improved after warmup, so its cold-start problem is more
  significant than `t2i`.

## Optimization Work Already Done

- Removed major CPU round-trips in the multimodal prompt path.
- Moved more prompt/image merge work onto device-side ops.
- Reduced some repeated host scalar / token / id uploads via caching.
- Switched scheduler token counting away from host `int(...)` into shape-driven logic.
- Added denoising step tracer structure closer to Flux2.
- Removed the edit transformer's shape-keyed lazy compile path:
  - `QwenImageEditTransformerModel` now compiles once and reuses a single
    module-v2/MAX-native graph.
  - `zero_cond_t` condition-token handling is now derived dynamically from
    image-token `T` IDs instead of a host `num_noise_tokens` scalar.
- Switched Qwen image block split modulation / gating to device-side
  `condition_token_mask` application via `ops.where(...)`, keeping the edit
  path shape-dynamic across sequential runs with different image sizes.
- Moved edit image-latent concatenation out of the denoising loop:
  - the combined latent/image sequence is now built once before the loop
  - scheduler updates preserve condition tokens using the precomputed image
    IDs
- Added a MAX-native Qwen t2i CFG fast path for the common single-image case:
  - when positive/negative prompt embedding lengths match, the transformer
    runs one batched CFG forward instead of two separate forwards
  - otherwise it safely falls back to the existing two-pass path

## Latest Validation

- `python -m compileall` passes for the touched Qwen image/edit pipeline files.
- Local diagnostics were clean for the touched pipeline/model files.
- `./bazelw run //max/examples/diffusion:simple_offline_generation -- --help`
  succeeds after the changes.
- `./bazelw run //max/examples/diffusion:same_process_multi_shape_runner -- ...`
  succeeds for multi-shape same-process smoke runs on both:
  - `Qwen/Qwen-Image-2512`
  - `Qwen/Qwen-Image-Edit-2511`
- Smoke runs succeeded:
  - `Qwen/Qwen-Image-2512`, 1-step t2i, profiling enabled
  - `Qwen/Qwen-Image-Edit-2511`, 1-step edit, profiling enabled

## Best Next Steps

### 1. Focus on `Qwen-Image` t2i first

- The current worst gap is there.
- `edit` still matters, but `t2i` is the less graph-friendly path now.

### 2. Reduce repeated HtoD uploads

- Cache more of:
  - prompt token buffers
  - text ids
  - latent image ids
  - scheduler helper inputs
  - any static CFG / shape carrier inputs still rebuilt per run

### 3. Reduce tiny helper kernels

- Look for repeated:
  - concat
  - tile
  - reshape
  - small normalization / bookkeeping kernels
- Prefer larger compiled paths and loop-external preparation where possible.

### 4. Compare against Flux2 orchestration

- The gap is not because this is module-v2.
- Flux2 is still more graph-friendly in how it prepares / reuses step inputs.
- The next pass should compare:
  - loop input preparation
  - per-step helper calls
  - cached vs rebuilt buffers

## Notes

- The warm profiling script currently does warmup in one CLI run and then
  profiles in another CLI run.
- That is still useful for comparison, but it is not the same as
  "same-process second invocation".
- If needed later, a more exact benchmark would add same-process profiling
  support directly in the example entrypoint.
