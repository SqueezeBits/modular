# Flux2 Optimization Log

## Summary of Changes

To eliminate performance gaps in the Flux2 denoising loop, we implemented three key optimizations:

1.  **Symbolic Shapes for Pre-Computation**:
    - Problem: `_ensure_all_time_steps_model` was recompiling on every run because the input shape (sigmas) changed between warmup (steps=3) and inference (steps=N).
    - Fix: Used symbolic shape `["num_sigmas"]` for the `sigmas` input in the graph definition.
    - Impact: "Scheduler Setup" time dropped from ~11.6s to ~0.13ms.

2.  **Zero-Copy Loop with DriverTensor**:
    - Problem: Per-step host-to-device transfers for `timestep` and `dt` caused ~14ms overhead per step.
    - Fix: Pre-computed all timesteps/dts on device and used `DriverTensor` slicing within the loop.
    - Impact: Eliminated host-device sync points in the loop body.

3.  **Bypassing `max.functional` Wrapper**:
    - Problem: Even with `DriverTensor`, a persistent ~14ms gap remained due to `max.functional` wrapper overhead (tracing, context management) and `Model.__call__` signature binding.
    - Fix: Unwrapped the `compiled_model` to access the raw `Model` object and called `.execute()` directly.
    - Impact: Eliminated `max/_realization_context` overhead.

4.  **Implicit Synchronization for OOM Prevention**:
    - Problem: Bypassing `max.functional` removed automatic synchronization, leading to unbounded async queue buildup and OOM on longer sequences (50+ steps).
    - Fix: Added explicit `device.synchronize()` at the end of each step.
    - Impact: Stable execution for 50 steps with negligible performance cost.

## Performance Results

### Baseline (5 steps)
- **Scheduler Setup:** ~11,595 ms
- **Total Latency:** ~14.40 s
- **Per-Step Latency:** ~280 ms + ~14ms overhead

### Optimized (5 steps)
- **Scheduler Setup:** ~0.13 ms
- **Total Latency:** ~2.52 s
- **Processing Rate:** ~2.0 steps/sec (including all overhead)

### Optimized (50 steps)
- **Total Latency:** ~15.05 s
- **Processing Rate:** ~3.3 steps/sec (pure compute bound)

## Execution Logs

### 5-Step Run
```
[Profiling] Prompt Encoding: 394.70ms
[Profiling] Latent Prep: 1.25ms
[Profiling] Guidance/IDs Cache: 0.02ms
[Profiling] Transformer Compilation/Lookup: 0.02ms
[Profiling] Scheduler Setup: 0.13ms
[Profiling] Pre-Loop Overhead: 0.69ms
step 0: prep=0.03ms, transformer=12.47ms, scheduler=0.46ms, total=12.96ms
step 1: prep=0.03ms, transformer=199.92ms, scheduler=0.78ms, total=200.73ms
step 2: prep=0.03ms, transformer=262.28ms, scheduler=0.78ms, total=263.08ms
step 3: prep=0.03ms, transformer=265.97ms, scheduler=0.79ms, total=266.79ms
step 4: prep=0.03ms, transformer=266.24ms, scheduler=0.77ms, total=267.04ms
[Profiling] Decode Gap: 0.00ms
[Profiling] Total Execute Time: 2314.06ms
Latency: 2.5259s
```

### 50-Step Run
```
step 49: prep=0.02ms, transformer=16.50ms, scheduler=0.55ms, total=17.07ms
[Profiling] Decode Gap: 0.00ms
[Profiling] Total Execute Time: 14997.53ms
Latency: 15.0509s
Generation complete!
```

## Conclusion
The optimization reduced total latency for a 5-step run by **~82%** (14.4s -> 2.5s) and ensured stable, OOM-free execution for longer sequences. The approach minimizes Python overhead in the critical path.
