# Bug Report: `comm.scatter` hangs at grid_size > 512 due to missing clamp

## Summary

`max/kernels/src/comm/scatter.mojo` computes a kernel launch `grid_size`
directly from input element count without clamping to
`MAX_NUM_BLOCKS_UPPER_BOUND = 512`. When `grid_size` exceeds 512,
`_multi_gpu_barrier` writes past the end of its per-block counter array
inside the `Signal` struct, producing out-of-bounds memory writes and a
deadlocked barrier. The process hangs indefinitely with no error.

All other collectives (`allreduce`, `allgather`, `reducescatter`,
`broadcast`) respect this bound. Scatter is the sole outlier.

## Reproducer

```bash
# With NCCL binding + custom bench added, but the same hang reproduces
# with the upstream bench_scatter.mojo at large num_elems.
./bazelw build //max/kernels/benchmarks:gpu/comm/bench_scatter

# Works (num_elems small enough that grid_size <= 512):
./bazel-bin/max/kernels/benchmarks/gpu/comm/bench_scatter --num_elems=524288
#   -> slowest mean time 0.011 ms, OK

# Hangs forever:
./bazel-bin/max/kernels/benchmarks/gpu/comm/bench_scatter --num_elems=1048576
#   -> never returns; must be SIGTERMed
```

With `bfloat16` (`simd_width=8`, `BLOCK_SIZE=256`) on an NVIDIA B200
8-GPU node, the hang threshold is exactly between per-rank 2 MB and 2.25 MB,
which corresponds to `grid_size == 512` and `grid_size == 576` respectively.

### Observed binary search results

| Per-rank size | Elements (bf16) | `grid_size` | Result |
|---|---|---|---|
| 2.00 MB | 1,048,576 | 512 | ✅ works (boundary) |
| 2.25 MB | 1,179,648 | **576** | ❌ **hangs** |
| 2.50 MB | 1,310,720 | **640** | ❌ **hangs** |
| 4.00 MB | 2,097,152 | **1024** | ❌ **hangs** |
| 8.00 MB | 4,194,304 | **2048** | ❌ **hangs** |

Environment:
- NCCL path of the same benchmark works correctly at all sizes up to
  256 MB, confirming the hang is in the Mojo kernel and not in benchmark
  harness code.
- GPU: NVIDIA B200 × 8, not throttled.
- NCCL: 2.28.9.

## Root Cause

### 1. Signal struct has a fixed 512-slot counter array

`max/kernels/src/comm/sync.mojo`:

```mojo
comptime MAX_NUM_BLOCKS_UPPER_BOUND = 512

# (inside the Signal struct)
#   A 2D array of counters with shape (MAX_NUM_BLOCKS_UPPER_BOUND, MAX_GPUS).
#   ...
#   A 3D array of counters with shape (2, MAX_NUM_BLOCKS_UPPER_BOUND, MAX_GPUS).
```

`_multi_gpu_barrier` is then indexed by `blockIdx.x`. Any block with
`blockIdx.x >= 512` writes past the counter array, corrupting the next
struct field (or adjacent memory) and — more importantly — never
participating in a consistent barrier count, so the other ranks spin
forever waiting for the missing ticks.

### 2. `scatter` launches without respecting that bound

`max/kernels/src/comm/scatter.mojo:180`:

```mojo
comptime BLOCK_SIZE = 256
comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
var grid_size = ceildiv(ceildiv(max_elems, simd_width), BLOCK_SIZE)
# <-- no clamp against MAX_NUM_BLOCKS_UPPER_BOUND
```

There is **no** `min(..., MAX_NUM_BLOCKS_UPPER_BOUND)` and no
`max_num_blocks` parameter that callers could use to cap it
externally.

### 3. Other collectives do clamp

For comparison, `allreduce` (`comm/allreduce.mojo:365`):

```mojo
var grid_size = min(max_num_blocks, ceildiv(num_elements, BLOCK_SIZE))
```

It also validates at `allreduce.mojo:899`:

```mojo
if max_num_blocks > MAX_NUM_BLOCKS_UPPER_BOUND:
    raise Error(...)
```

`allgather`, `reducescatter`, and `broadcast` follow the same pattern.
`scatter` is the only collective missing both the external parameter
and the internal clamp.

## Why the bug went undetected

- `bench_scatter.mojo` defaults to `num_elems = 16`, which produces
  `grid_size = 1`. All existing tests exercise tiny payloads.
- The kernel body already contains a correct grid-stride loop
  (`for idx in range(global_tid, num_simd_vectors, stride)`), so
  functional correctness is preserved as long as no barrier
  corruption occurs — the bug is latent until a caller passes enough
  data to push `grid_size` past 512.
- `scatter` appears to only be used in narrow DP/TP broadcast code
  paths today; none of them happen to push past 2 MB per replica on
  bf16.

## Fix

Two possible levels of fix; we apply the minimal one here.

### Minimal fix (correctness only, 1 line + 1 import)

In `max/kernels/src/comm/scatter.mojo`:

```diff
-from comm.sync import ..., is_p2p_enabled
+from comm.sync import ..., is_p2p_enabled, MAX_NUM_BLOCKS_UPPER_BOUND
 ...
-    var grid_size = ceildiv(ceildiv(max_elems, simd_width), BLOCK_SIZE)
+    var grid_size = min(
+        ceildiv(ceildiv(max_elems, simd_width), BLOCK_SIZE),
+        MAX_NUM_BLOCKS_UPPER_BOUND,
+    )
```

Because the kernel already uses `stride = grid_dim.x * BLOCK_SIZE`
inside a grid-stride loop, clamping `grid_size` preserves the
semantics exactly — it only reduces parallelism above the 2 MB mark.

### Ideal fix (parameter parity with other collectives)

Expose `_max_num_blocks: Optional[Int]` on the public `scatter`
function to match `allreduce`/`allgather`/`reducescatter`, with the
same upper-bound validation, so callers that know they will launch
large scatters can tune it.

## Severity

- **Correctness**: the current implementation silently deadlocks above
  2 MB per rank (bf16). The deadlock has no timeout or error — the
  host process hangs until killed.
- **Safety**: the out-of-bounds writes in the barrier counter array
  produce undefined behavior, including potential corruption of
  adjacent struct fields on the device.
- **Exposure today**: limited, because existing consumers of
  `comm.scatter` use tiny payloads. But any future caller attempting
  to scatter model weights, KV cache shards, or activation tensors at
  realistic sizes will trip this bug.

## Recommended next step

File upstream as a bug against Modular with this report. Minimal
patch is ~3 lines; passes existing `bench_scatter` correctness path
and unlocks the full sweep range to match other collectives.
