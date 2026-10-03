# kde2: KernelDensity.score_samples on the Apple GPU (FAST), profile and causes

Lane af-kde2, branch lane/apple-fast-kde2, cut from main 8897404da (0.8.35). Board lane `kde`
(AFC_FAMILY=classical, tools/classical_two_datasets.py: 100,000 fit rows, 2,000 queries,
gaussian, euclidean, bandwidth 1.0, standardized; `score_samples` alone is timed, the fit is
before the clock). M3 Ultra, board 0834: taxi FAST 60 ms / IDENTICAL 112 ms; istella (d = 220)
FAST 1,539 ms / IDENTICAL 181 ms.

## Finding 1: why FAST is 8.5x slower than IDENTICAL on istella

The FAST tier takes DEVIATION 2490's fused pass (`kde/impl/neighbors/kernel_density.mojo`,
`_kde_score_samples_fused`, dispatched from `kde_score_samples_device` under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST`). Its shape, not its arithmetic, is the regression:

1. **One thread per query, 128 queries per block, grid = ceil(n_query / 128).** At 2,000
   queries that is 16 threadgroups of 128 threads (`_kde_fused_launch`, `grid_dim=((n_query +
   127) // 128, 1, 1)`). The M3 Ultra has 80 GPU cores: 16 of them hold one threadgroup each
   (4 simdgroups, no latency hiding), 64 idle. The same 16 blocks run on taxi, so this alone does
   not explain the ratio; it explains why nothing scales.
2. **Every thread walks all 100,000 train rows serially, so the per-thread work is
   `n_train x d` FMAs.** taxi (d = 11, DPAD = 12): 1.2 M FMAs per thread. istella: 22 M. The
   serial chain is 20x longer and there is no other parallelism to absorb it. IDENTICAL does not
   have this chain: DEVIATION 2625's `kde_tiled_logk_kernel` runs a 2D grid (16 query blocks x
   391 train chunks = 6,256 blocks) with 64 SIMD accumulators per thread, then pays for an 800 MB
   log-kernel matrix round trip (`kde_lse_block_sum_kernel`). On taxi the matrix dominates and the
   fused pass wins (60 against 112); on istella the serial chain dominates and loses (1,539
   against 181).
3. **d > KDE_FUSED_QREG (64) takes the wide arm (`kde_fused_wide_kernel`), which is worse per
   cell.** The shared tile is `KDE_FUSED_TILE_FLOATS // d` = 3072 // 220 = 13 train rows, so the
   100,000 rows are 7,693 tile steps, each with two `barrier()`s and a 2,860-float tile fill by
   128 threads. Per step each thread RE-READS its query row from global memory in chunks of 64
   (`query.unsafe_load(qbase + c0 + f)` inside the `while c0 < d` loop): 220 global loads per
   thread per step, 1.7 M redundant global loads per thread per call. The per-row accumulators
   `acc` and `tn` are 47-float `stack_allocation`s indexed by the runtime `j`, so they live in
   thread-private (device) memory and cost a load and a store per cell per feature chunk; the
   `if f < width` predicate sits inside the 64-wide unrolled loop. taxi's arm (`DPAD = 12`) keeps
   the query row in 12 registers, holds 256 rows per tile, and runs 391 steps.

So the FAST path scales as `n_train x d` on 2,048 threads with a slow inner loop, while IDENTICAL
spreads the same cells over the whole GPU. lane/apple-fast-core's unmerged `MOJOLEARN_KDE_FAST_SLICES`
(12dd5697f, 8 commits) attacks (1) and (2) by splitting the train rows over `grid.y` and merging
per-slice `(m, s)` pairs, keeping the one-thread-per-query kernels and their wide arm (3). This lane
does not duplicate it: it replaces the kernel shape (define 1 below) and keeps a chunk merge only as
the second half of a 2D tile, with a different kernel and a different merge.

## Profile of one FAST `score_samples` call (resident fit, main 8897404da, Apple)

Per call, `kde/resident_fit.mojo::kde_score_samples_resident` then
`kde/impl/kde.mojo::score_samples` then `kde_score_samples_device` then `_kde_score_samples_fused`:

| step | where | cost class |
|---|---|---|
| kernel/metric name lookup, registry lookup, shape checks | resident_fit.mojo | host, trivial |
| `kde_validate_data_ptr(query)`: finiteness scan of n_query x d | kernel_density.mojo:289 | host, 440 K floats on istella |
| `enqueue_create_buffer` dquery, `enqueue_create_host_buffer` host, `copy_f32` (1.76 MB memcpy), `enqueue_copy` upload | resident_fit.mojo | 2 buffers, 1 host memcpy, 1 DMA |
| `enqueue_create_buffer` dout | resident_fit.mojo | 1 buffer |
| `trace.header(...)` string build | resident_fit.mojo | host, trivial |
| `validate_metric_arg`, `enqueue_create_buffer` lse (n_query), logw (1 float, unweighted dummy) | kernel_density.mojo `_kde_score_samples_fused` | 2 buffers |
| `log_weights_kernel` (weighted only) | | 1 launch |
| the fused pass: 16 blocks x 128 threads, `kde_fused_wide_kernel` at d = 220, `kde_fused_logsumexp_kernel[12]` at d = 11 | | 1 launch, THE cost (1,539 / 60 ms) |
| `log_kernel_norm` (host float64 formula), `normalize_scores_kernel` | | 1 launch |
| `ctx.synchronize()` | `_kde_score_samples_fused` end | drain 1 |
| `enqueue_create_host_buffer` hout, `enqueue_copy` download, `ctx.synchronize()`, `copy_f32` to scores | resident_fit.mojo | 1 buffer, drain 2, 8 KB memcpy |
| frees: hout, host, dquery, dout, lse, logw | | 6 frees |

Totals per unweighted call: 2 launches, 2 synchronizes, 6 buffers created and freed, 2 host
memcpys, 1 upload, 1 download, 1 host scan. Nothing but the fused launch matters until that
launch is ~20 ms; then the two drains (~0.2 ms each on Apple), the six buffer lifetimes and the
separate normalize launch become visible.

## Design chosen for this lane (all FAST + Apple, each behind its own -D define, default OFF)

Details per define in docs/apple-fast/ab/kde2.md. Summary:

- `MOJOLEARN_KDE_DIMTILE`: a 2D register-tiled kernel (block = 64 queries x 64 train rows, 128
  threads, 4 x 8 cells per thread, 32-feature chunks of both tiles staged feature-major with a
  pad in threadgroup memory, online log-sum-exp per query in registers, the 8 train groups of a
  block merged through threadgroup memory, train rows chunked over `grid.y` so ~2,048 blocks run),
  then a one-thread-per-query merge of the chunk `(m, s)` pairs. Fixes (1), (2) and (3).
- `MOJOLEARN_KDE_LSE_FUSED`: the chunk merge and the normalization in one launch, no `lse` buffer.
- `MOJOLEARN_KDE_NORM_FUSED`: euclidean / sqeuclidean as `|q|^2 + |t|^2 - 2 q.t` with the row
  norms from one kernel, so the inner loop is one FMA per feature instead of a subtract and an FMA.
- `MOJOLEARN_KDE_KERNEL_VARIANTS`: the tile kernel instantiated per metric (and the gaussian x
  euclidean epilog) at compile time; no runtime metric branch in the kernel body.
- `MOJOLEARN_KDE_SAMPLE_FUSED`: the resident score call drains once (the download is enqueued
  behind the kernels and the scratch outlives the one synchronize); no dummy log-weight buffer.
- `MOJOLEARN_KDE2_ALL`: all five.
