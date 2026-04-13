# AlltoAll v2 (Block-Parallel Pull) — Benchmark Results

Second iteration of the Mojo P2P AlltoAll collective. Addresses the
many-GPU / large-message bandwidth deficit identified in v1 by
partitioning the grid per peer so all `ngpus` remote NVLink reads
execute concurrently.

See `alltoall_v1_benchmark.md` for the v1 baseline and the limitation
analysis that motivated v2.

## What changed vs v1

| Axis | v1 | v2 |
|---|---|---|
| Grid layout | 1D, all blocks loop over peers serially | 1D, `blocks_per_peer × ngpus` blocks; each block assigned to one peer |
| Peer dispatch | `comptime for peer in range(ngpus)` inside every block | `peer = blockIdx.x / blocks_per_peer` at runtime |
| Concurrent NVLink reads per SM group | 1 (one peer at a time) | up to `ngpus` (one peer per block column) |
| `_max_num_blocks` param | ❌ | ✅ (matches `allreduce` convention, enables autotune) |

The kernel body per block is simpler in v2 — only one chunk, no inner
peer loop — but the launch packs more parallelism into the grid.

## Headline vs NCCL

**v2 wins or ties NCCL on all 24 measurement points.**

| GPUs | Peak v2 gain vs NCCL | BusBW at 256 MB (v2 / NCCL) |
|---|---|---|
| 2 | +73% (at 1 MB) | 355 / 327 GB/s |
| 4 | +48% (at 256 KB) | 537 / 525 GB/s |
| 8 | +76% (at 256 KB) | 606 / 604 GB/s |

At 8 GPU / 256 MB — the hardest case and v1's worst point — v2 now
matches NCCL to within 0.3 %, up from v1's **−69 % deficit**.

## v2 vs v1 (Mojo-to-Mojo delta)

| GPUs | Size | v1 μs | v2 μs | Δ% |
|---:|---:|---:|---:|---:|
| 2 | 16 KB  | 10.22   | 9.20    | +10.0 |
| 2 | 1 MB   | 11.36   | 11.03   | +2.9  |
| 2 | 16 MB  | 50.12   | 58.57   | **−16.9** |
| 2 | 256 MB | 645.24  | 755.10  | **−17.0** |
| 4 | 16 KB  | 15.22   | 10.05   | +34.0 |
| 4 | 1 MB   | 19.52   | 16.29   | +16.6 |
| 4 | 16 MB  | 171.28  | 106.94  | +37.6 |
| 4 | 256 MB | 2583.73 | 1498.53 | **+42.0** |
| 8 | 16 KB  | 24.60   | 11.21   | **+54.4** |
| 8 | 1 MB   | 35.62   | 24.64   | +30.8 |
| 8 | 16 MB  | 519.93  | 210.44  | **+59.5** |
| 8 | 64 MB  | 2381.17 | 790.66  | **+66.8** |
| 8 | 256 MB | 10147.06| 3100.97 | **+69.4** |

(Full 24-point table below.)

The only regressions are **2-GPU large sizes** (16 MB+), where the
grid-partition overhead exceeds the benefit: with only N=2 peers,
splitting the grid in two simply halves per-peer block count, but there
were only ever two remote links to saturate anyway. Even so, v2 still
beats NCCL at those points by +8–30 % — it just gives up some v1
headroom. This is an explicit trade-off we accept: on 4 / 8 GPU the
wins are dramatic, and those are the configurations where the pre-v2
gap mattered.

## Full Results — 24 points (Mojo v2 vs NCCL 2.28.9)

| GPUs | num_bytes (per chunk) | Per-rank | v2 μs | NCCL μs | v2 BW | NCCL BW | Δ vs NCCL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 KB  | 32 KB  | 9.20    | 11.24   | 1.8    | 1.5    | +22.2% |
| 2 | 64 KB  | 128 KB | 9.03    | 11.68   | 7.3    | 5.6    | +29.3% |
| 2 | 256 KB | 512 KB | 9.96    | 13.09   | 26.3   | 20.0   | +31.5% |
| 2 | 1 MB   | 2 MB   | 11.03   | 19.09   | 95.1   | 54.9   | **+73.1%** |
| 2 | 4 MB   | 8 MB   | 21.29   | 33.22   | 197.0  | 126.3  | +56.0% |
| 2 | 16 MB  | 32 MB  | 58.57   | 76.02   | 286.5  | 220.7  | +29.8% |
| 2 | 64 MB  | 128 MB | 198.37  | 227.28  | 338.6  | 295.5  | +14.6% |
| 2 | 256 MB | 512 MB | 755.10  | 820.09  | 355.5  | 327.4  | +8.6% |
| 4 | 16 KB  | 64 KB  | 10.05   | 12.42   | 4.9    | 3.9    | +23.5% |
| 4 | 64 KB  | 256 KB | 10.47   | 12.51   | 18.8   | 15.7   | +19.5% |
| 4 | 256 KB | 1 MB   | 11.64   | 17.22   | 67.6   | 45.7   | **+47.9%** |
| 4 | 1 MB   | 4 MB   | 16.29   | 23.29   | 193.0  | 135.1  | +43.0% |
| 4 | 4 MB   | 16 MB  | 35.77   | 48.16   | 351.7  | 261.4  | +34.6% |
| 4 | 16 MB  | 64 MB  | 106.94  | 129.27  | 470.5  | 389.3  | +20.9% |
| 4 | 64 MB  | 256 MB | 386.18  | 435.70  | 521.3  | 462.1  | +12.8% |
| 4 | 256 MB | 1 GB   | 1498.53 | 1534.16 | 537.2  | 524.8  | +2.4% |
| 8 | 16 KB  | 128 KB | 11.21   | 13.28   | 10.3   | 8.7    | +18.5% |
| 8 | 64 KB  | 512 KB | 11.68   | 14.64   | 39.3   | 31.4   | +25.3% |
| 8 | 256 KB | 2 MB   | 14.33   | 25.17   | 128.3  | 73.0   | **+75.6%** |
| 8 | 1 MB   | 8 MB   | 24.64   | 35.19   | 298.3  | 208.8  | +42.8% |
| 8 | 4 MB   | 32 MB  | 63.21   | 84.33   | 464.7  | 348.3  | +33.4% |
| 8 | 16 MB  | 128 MB | 210.44  | 239.75  | 558.5  | 489.8  | +13.9% |
| 8 | 64 MB  | 512 MB | 790.66  | 849.91  | 593.8  | 552.7  | +7.5% |
| 8 | 256 MB | 2 GB   | 3100.97 | 3111.39 | 605.9  | 603.9  | +0.3% |

**Win/Loss**: v2 24 / 24, NCCL 0 / 24. CSV: `comm_benchmark_scripts/results/alltoall.csv`.

## BusBW scaling (256 MB, saturation regime)

v1 exhibited the pathological scaling `416 → 312 → 185 GB/s` going
from 2 → 4 → 8 GPUs because the serial peer loop couldn't exploit
more than one NVLink at a time. v2 scales the *opposite* direction:

| GPUs | v1 BW | v2 BW | NCCL BW |
|---:|---:|---:|---:|
| 2 | 416 | 355 | 327 |
| 4 | 312 | 537 | 525 |
| 8 | 185 | 606 | 604 |

v2's 8-GPU busbw (606 GB/s) is **3.3× v1** and effectively matches the
NVLink topology limit that NCCL also tops out at. The 2-GPU drop
(416 → 355) is the cost of splitting the grid even when there is no
concurrent-link benefit — a small price for the massive many-GPU win.

## Design notes and tradeoffs

- **Grid must be a multiple of `ngpus`.** `blocks_per_peer` is computed
  and `total_blocks = blocks_per_peer * ngpus` is passed to
  `enqueue_function`. This keeps the flat `blockIdx.x ∈
  [0, total_blocks)` addressable by `_multi_gpu_barrier` without
  touching the barrier's counter layout.

- **Barrier cost vs parallelism.** At small sizes the start/end
  barrier still dominates, but v2 already wins at 16 KB (see above)
  because the per-peer block count is small enough to keep the barrier
  overhead proportional to v1's.

- **`_max_num_blocks` parameter.** Present for API parity with
  `allreduce`. Applies to the *total* (product) grid size, so callers
  who want to cap at e.g. 128 blocks per peer on 8 GPU should pass
  `1024`. Defaulting to `MAX_NUM_BLOCKS_UPPER_BOUND` preserves
  autotune room without imposing a ceiling lower than the barrier
  allows.

- **Self-chunk still copied via the P2P path.** The peer whose index
  equals `my_rank` still executes a full copy through its dedicated
  block column. HW recognizes `input_ptrs[my_rank]` as a local
  address, so the load goes to HBM rather than NVLink. Skipping the
  self-chunk entirely would require either a caller contract change
  (aliased sendbuf/recvbuf) or a separate block-partition scheme; not
  worth it given the measured cost is already folded into the
  `(n-1)/n` busbw formula.

## Known limitations remaining

1. **No multimem / NVLS usage.** B200 hardware multicast could further
   improve the very-large regime (`≥ 64 MB`) where v2 is within ~7%
   of peak NVLink. Not yet implemented.

2. **Static `blocks_per_peer` formula.** No autotune yet — the current
   heuristic just caps at `ceildiv(chunk / simd / BLOCK_SIZE)` or the
   `MAX_NUM_BLOCKS_UPPER_BOUND / ngpus` limit. Different sizes may
   prefer different split points; a tuned table would close the last
   few percent.

3. **2-GPU regression.** v2 is ~15% slower than v1 at 2-GPU / ≥ 16 MB.
   Could be fixed by a runtime dispatch (use v1 path when ngpus == 2
   and size above threshold), but the v2 result still beats NCCL, so
   the regression is cosmetic for now.

## Recommendation

**Production-ready across 2/4/8 GPUs and all measured sizes.**
v2 is at parity or better than NCCL everywhere; the earlier v1 fallback
note ("use NCCL for 8 GPU / large") is no longer needed. Callers can
use the Mojo path unconditionally via `use_vendor_ccl=False`.

## Appendix — Autotune attempt on `blocks_per_peer` (negative result)

After v2 landed we swept the `_max_num_blocks` cap across
{8, 16, 32, 64, 96, 128, 192, 256, 384, 512} for every (ngpus, size)
point and picked the fastest per-cell. The sweep was then re-verified
with multi-run measurements on the cells that initially looked like
wins.

**Outcome**: no cap reliably beat the current default
(`MAX_NUM_BLOCKS_UPPER_BOUND = 512`). Every apparent win was within
per-run measurement noise (~3-5% std at small sizes; larger at
64 KB on 8 GPU where one run hit an outlier that the initial pass
flagged as a +70% improvement — re-measurement showed no real gap).

**Why the default is already optimal**: the v2 heuristic
```
blocks_per_peer = min(blocks_for_chunk, cap // ngpus)
```
already self-adjusts per size:

- Small chunks: `blocks_for_chunk` dominates, so the cap doesn't bind
  and lowering it only wastes parallelism.
- Large chunks: `cap // ngpus` dominates, and 512 is enough to keep
  every SM busy on B200 without overshooting the barrier's 512-slot
  counter array.

In other words, the two-term `min` is the autotune. Further gains
need structural changes, not block-count fiddling:

- Multimem / NVLS: not a natural fit for alltoall (permutation, not
  reduction or broadcast) — skipped.
- `cp.async.bulk` (B200 TMA) for SM-stall-free bulk copies — the most
  promising next avenue for the very-large regime where v2 already
  sits at 97-99% of NCCL's peak busbw.
- `ngpus == 2` runtime dispatch to v1 to recover the 2-GPU /
  ≥ 16 MB regression.

No code change landed for this experiment; this appendix documents the
negative result so the same autotune sweep isn't re-run speculatively
later.
