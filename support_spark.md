# DGX Spark Support Notes

This document summarizes the changes made while bringing `FLUX.2-klein-4b`
inference up on the DGX Spark environment.

## Kernel and cuDNN fixes

- Fixed cuDNN FFI ABI mismatches in
  `max/kernels/src/_cudnn/infer.mojo` and
  `max/kernels/src/_cudnn/cnn_infer.mojo`.
  Descriptor dimension arguments and several enum-like types now use
  `Int32`, which matches the cuDNN C API instead of truncating values
  through narrower integer types.
- Improved cuDNN error reporting in `max/kernels/src/nn/conv.mojo`.
  Failures now include the cuDNN status code, `cudnnGetErrorString(...)`,
  and the specific cuDNN API call that failed.
- Updated cuDNN convolution call sites in
  `max/kernels/src/nn/conv.mojo` and
  `max/kernels/src/nn/conv_transpose.mojo`
  to pass `Int32` descriptor parameters.

## Diffusion profiling

- Removed the `torch` dependency from
  `max/examples/diffusion/profiler.py`.
- Profiling now synchronizes through `max.driver.Accelerator` instead of
  `torch.cuda.synchronize()`, so `--profile-timings` works without PyTorch.

## Klein and Qwen3 path

- Replaced `F.tile(...)` with `F.concat([x] * n_rep, axis=2)` in
  `max/python/max/pipelines/architectures/qwen3/text_encoder/layers/attention.py`.
  This keeps grouped-query KV repetition on the concat path used elsewhere
  in the repo and avoids the tile behavior we were debugging for Klein.

## Web demo

- Added `max/examples/diffusion/web_demo.py`, a simple browser-based demo
  for prompt-driven image generation with height, width, seed, and latency.
- Added the `//max/examples/diffusion:web_demo` Bazel target.
- Refactored the HTML renderer in `web_demo.py` to use Jinja templates and
  added `requirement("jinja2")` to the Bazel target.

## Verified builds

- `./bazelw build //max/examples/diffusion:simple_offline_generation`
- `./bazelw build //max/examples/diffusion:web_demo`

## Useful commands

Benchmark with timings:

```bash
./bazelw run //max/examples/diffusion:simple_offline_generation -- \
  --model black-forest-labs/FLUX.2-klein-4b \
  --prompt "A cat holding a sign that says hello world" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 4 \
  --num-warmups 1 \
  --guidance-scale 1.0 \
  --seed 42 \
  --profile-timings
```

Web demo:

```bash
./bazelw run //max/examples/diffusion:web_demo -- \
  --model black-forest-labs/FLUX.2-klein-4b \
  --host 0.0.0.0 \
  --port 8000
```
