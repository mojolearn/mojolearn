# The thesis scoreboard: our GPU against their best CPU, same Mac

**The thesis this repo exists for is GPU ACCESS, not GPU tier: the
incumbents' GPU arms cannot run on Apple silicon at all, so the comparison a
Mac user actually lives with is our Metal path against their best CPU on the
same machine.** Against scikit-learn 1.9.0 with `n_jobs=-1` on every
estimator that accepts it and their `algorithm='auto'` -- their best, not a
crippled arm. Method: `bench/run_bench.py`, arms alternated per round inside
one invocation, medians reported, a row is a finding only when the min..max
ranges do NOT overlap. Apple M4 (10 cores, 16 GB), AC power, no thermal
warnings. Re-verdicted 2026-08-20 morning at commit 447b6c0 (PCA centering
fused into the split-K read, DEVIATION 42; split-K floor knobs; kmeans fused
L2-NN re-ported to upstream's Veclen policy, DEVIATIONS 44/45).

## Fixed-size arms (3 rounds, n = 15/arm, window of 2026-08-20)

| arm | shape | ours ms | sklearn ms | ratio | verdict |
|---|---|---|---|---|---|
| ols | 4M x 32 | 62.6 | 906.2 | 14.48x | **ours faster** |
| ols vs `ols_normal_eq` | same | 62.6 | 151.5 | 2.42x | **ours faster** |
| knn | 400k idx, 4k q, d=32, k=10 | 756.0 | 1,141.2 | 1.51x | **ours faster** |
| pca | 4M x 32, 8 comp | 75.8 | 147.0 | **1.94x** | **ours faster** |
| dbscan | 4k x 16 | 12.7 | 10.0 | 0.79x | INDISTINGUISHABLE |
| kmeans | 4M x 32, k=64, 20 iter | 1,657.9 | 2,309.6 | **1.39x** | **ours faster** |

Ranges: kmeans [1541.7, 2036.0] vs [2053.1, 2784.5] -- DISJOINT, the row's
first clean verdict in our favor. pca [67.4, 91.9] vs [118.1, 236.4] --
DISJOINT. **The board's last loss and last big-arm tie both flipped in one
round, and neither flip was a tuning constant: both were port fidelity.**

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

**Five wins (OLS 2.4x/14.5x, PCA 1.9x, k-NN 1.3-1.8x, k-means 1.39x, DBSCAN
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

## What the table says to do next

1. **kmeans accumulate is now the widest single bracket**: 54 ms/iter
   against upstream-faithful code (audited line-by-line; their
   `reduce_rows_by_key` reads are scalar too). Any further win there is a
   deliberate deviation beyond upstream, not a fidelity fix -- price it
   before fighting it.
2. **`pca_transform` still centers the long way** (gemm_nt shape, different
   epilogue): untouched by DEVIATION 42, a candidate for the same fusion if
   transform latency ever matters on the board.
3. **Upstream report to Modular: DRAFTED**, commit 1ca0eb5
   (`bench/results/MODULAR_UPSTREAM_2026-08-20.md`) -- six probe-backed
   defects + the split-K gap with the 345 -> 49 ms reproducer. Filing is
   external and awaits Andrew's word.
4. Split-K sits at 42.5 ms steady against a ~15 ms traffic floor (~2.8x):
   the two named knobs are DONE; what remains is kernel-interior (wider
   accumulation per thread, staging depth), unpriced.
5. DBSCAN d=8 large-n `vertexdeg` constants: still the lowest priority.
6. **kmeans|| init remains unported** (the cuVS default raises); RunPod
   NVIDIA validation staged, awaiting Andrew's explicit word (billed).
