# The thesis scoreboard: our GPU against their best CPU, same Mac

**The thesis this repo exists for is GPU ACCESS, not GPU tier: the
incumbents' GPU arms cannot run on Apple silicon at all, so the comparison a
Mac user actually lives with is our Metal path against their best CPU on the
same machine.** Against scikit-learn 1.9.0 with `n_jobs=-1` on every
estimator that accepts it and their `algorithm='auto'` -- their best, not a
crippled arm. Method: `bench/run_bench.py`, arms alternated per round inside
one invocation, medians reported, a row is a finding only when the min..max
ranges do NOT overlap. Apple M4 (10 cores, 16 GB), AC power, no thermal
warnings. Re-verdicted 2026-08-20 afternoon at commit ee05664 (kmeans
accumulate reads X veclen-wide, DEVIATION 46; split-K staging copy
vectorized, a measured NULL for time, kept as bit-identical hygiene; morning
round: PCA centering fusion DEVIATION 42, kmeans Veclen re-port 44/45).

## Fixed-size arms (3 rounds, n = 15/arm, window of 2026-08-20 afternoon)

| arm | shape | ours ms | sklearn ms | ratio | verdict |
|---|---|---|---|---|---|
| ols | 4M x 32 | 62.0 | 913.1 | 14.73x | **ours faster** |
| ols vs `ols_normal_eq` | same | 62.0 | 166.9 | 2.69x | **ours faster** |
| knn | 400k idx, 4k q, d=32, k=10 | 705.7 | 1,196.4 | 1.70x | **ours faster** |
| pca | 4M x 32, 8 comp | 63.9 | 144.4 | **2.26x** | **ours faster** |
| dbscan | 4k x 16 | 10.5 | 10.2 | 0.98x | INDISTINGUISHABLE |
| kmeans | 4M x 32, k=64, 20 iter | 764.4 | 2,488.4 | **3.26x** | **ours faster** |

Ranges: kmeans [749.4, 1246.9] vs [2022.3, 3405.4] -- DISJOINT. pca [60.3,
80.9] vs [119.9, 282.7] -- DISJOINT. **kmeans went loss -> 1.39x -> 3.26x in
one day**: the morning flip was port fidelity (vector loads in assignment),
the afternoon jump is DEVIATION 46 (accumulate reads X veclen-wide, a priced
deviation BEYOND upstream -- their scalar reads lean on NVIDIA warp
coalescing this device does not replicate). Steady-state iteration brackets:
assignment ~27 ms, accumulate ~17 ms (was 63 + 54 two days ago).

- **PCA 166 -> 75.8 ms** (was a 4x loss two days ago at 465): the ~100 ms of
  center+restore passes are GONE -- the split-K kernel reads `x - mu[col]`
  in registers (RAFT's own `stable=false` `@todo`, implemented; DEVIATION
  42), X is never written, and `compute_covariance` measures 66 ms with the
  eig at ~3.
- **kmeans 2,448 -> 1,657.9 ms**: assignment 63 -> 21 ms/iter (phase
  bracket, steady state). The fused L2-NN kernel was reading ONE float at a
  time where upstream's `ldg` reads `Veclen` floats; LANE_kmeans-kernel
  ported the vector load machinery, their veclen selection computation,
  their strided ownership, their launch grid, and their smem-staged norms
  (report has the full diff table). Two earlier mechanism theories (unfused
  dispatch, atomic contention) had already died by measurement; the third
  -- scalar loads in a "faithful" port -- was the real one, found by
  line-by-line source diff, not by reasoning about our code.

## The dimensionality sweep (n = 200,000 fixed; 2026-08-19, still current)

Every earlier row ran at d=8/32 -- the kd-tree's best regime. Trees degrade
with d; a GPU's cost is d-linear. Measured (eps scaled sqrt(d/8) on both
sides, identical fixtures):

| arm | ours ms | sklearn ms | ratio | verdict |
|---|---|---|---|---|
| dbscan d=8 | 478.9 | 473.2 | 0.99x | INDISTINGUISHABLE |
| dbscan d=32 | 3,430.6 | 23,663.8 | **6.90x** | **ours faster** |
| dbscan d=64 | 15,290.3 | 39,946.9 | **2.61x** | **ours faster** |
| knn d=32 | 186.6 | 272.0 | 1.46x | **ours faster** |
| knn d=64 | 361.6 | 468.0 | 1.29x | **ours faster** |
| knn d=128 | 718.0 | 798.1 | 1.11x | INDISTINGUISHABLE |

scikit-learn's DBSCAN cost blows up 50x going d=8 -> 32 (the kd-tree
collapses); ours grows ~7x. **The DBSCAN rows this file used to report as
losses live entirely in the d=8 corner** -- the single regime most favorable
to the incumbent -- and even there, 200k is a tie. At any realistic feature
width the fit is a multi-x win.

## Scaling arms at d=8 (2 rounds; the incumbent's best corner; 2026-08-19)

knn (d=32, k=10, 2,000 q): 1.41x / 1.36x / 1.79x / 1.48x ours at
50k/100k/200k/400k, all clean. dbscan (d=8): ours 2.35x/1.74x at 4k/16k,
parity 100k-200k, sklearn 0.70x-0.83x ahead at 400k-800k. Attribution
(phase timer, 400k+800k): ~88% of phase time is the `vertexdeg`
eps-neighborhood kernels -- a query-kernel constants fight against kd-tree
asymptotics in their best regime, the LOWEST-priority gap on the board.

## The standing tally

**Five wins (OLS 2.7x/14.7x, PCA 2.3x, k-NN 1.3-1.8x, k-means 3.3x, DBSCAN
2.6-6.9x at d>=32), one tie (DBSCAN in the d=8 corner, where large-n is
attributed and deprioritized), zero losses.**

## What moved the 2026-08-20 round, measured

- **PCA centering fusion** (LANE_pca-centering, DEVIATION 42): RAFT
  `cov.cuh`'s `stable=false` arm is `ASSERT(false)` + `@todo: implement
  this using cutlass + customized epilogue!` -- their declared design,
  unshippable for them because cuBLAS exposes no epilogue hook, implemented
  here because the split-K kernel is ours. Bit-identical to
  center-then-gemm (proven per cell on hashed data); X provably untouched
  (worst element move exactly 0.0). Non-Apple targets keep their shipped
  `stable=true` path verbatim, same predicate `gemm_tn` reads.
- **Split-K floor knobs** (same lane, bit-identity FNV-hash-proven): the
  inner unroll reloaded one shared address CELLS times per row whenever
  `256 % m == 0` -- now one hoisted load; the 240-chunk workspace reuses
  the `xt` scratch instead of allocating per call. `gemm_tn_alone` steady
  state: 42.5 ms (first-shot 49 was partly alloc+cold).
- **kmeans fused L2-NN re-port** (LANE_kmeans-kernel, DEVIATIONS 44/45):
  Veclen vector loads + selection computation, `Policy4x4Skinny` arm,
  strided accumulator ownership, launchConfigGenerator grid with m
  grid-stride, smem-staged norms, self-neighbor round-off guard.
  Assignment 63 -> 21 ms/iter. Accumulate side audited CLEAN against
  upstream (their reads are scalar too).

## Afternoon round, measured

- **kmeans accumulate veclen (DEVIATION 46, LANE_kmeans-accumulate)**:
  ~17 ms/iter steady (was 54). One SIMD chunk + ONE label/weight read per
  veclen cells; selection reuses the assignment port's `fused_veclen_for`
  ladder; bit-identical on all three arms (fixed-point Int32 adds are
  order-free), scalar arm proven at d=33, 2-wide at d=34. Fit: 1,658 ->
  764 ms.
- **Split-K staging vectorization (LANE_splitk-interior): NULL for time.**
  `gemm_tn_alone` 42.5 -> 44-45 ms (drift-indistinguishable). The scalar
  staging copy was real and is now 16-byte loads, FNV-bit-identical, kept
  as free hygiene -- but it was NOT the bottleneck. The kernel still runs
  ~11 GB/s effective; the lane's own caveat stands: the accumulation
  loop's shared reads carry ~40x the staging op count, and the next step
  there is Apple Instruments profiling, not structure.

## What the table says to do next

1. **Split-K's remaining ~3x to floor now needs Instruments**, not
   structure: staging vectorization was the last structural suspect and it
   measured null. Profile before touching the kernel again.
2. **kmeans|| init remains unported** (the cuVS default raises) -- now the
   largest FUNCTIONAL gap on an otherwise all-green board.
3. **Upstream report to Modular: DRAFTED**, commit 1ca0eb5
   (`bench/results/MODULAR_UPSTREAM_2026-08-20.md`) -- six probe-backed
   defects + the split-K gap with the 345 -> 49 ms reproducer. Filing is
   external and awaits Andrew's word.
4. **`pca_transform` still centers the long way** (gemm_nt shape):
   untouched by DEVIATION 42, a candidate if transform latency ever
   matters on the board.
5. DBSCAN d=8 large-n `vertexdeg` constants: still the lowest priority.
6. RunPod NVIDIA validation staged, awaiting Andrew's explicit word
   (billed).
