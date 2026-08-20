# The thesis scoreboard: our GPU against their best CPU, same Mac

**The thesis this repo exists for is GPU ACCESS, not GPU tier: the
incumbents' GPU arms cannot run on Apple silicon at all, so the comparison a
Mac user actually lives with is our Metal path against their best CPU on the
same machine.** Against scikit-learn 1.9.0 with `n_jobs=-1` on every
estimator that accepts it and their `algorithm='auto'` -- their best, not a
crippled arm. Method: `bench/run_bench.py`, arms alternated per round inside
one invocation, medians reported, a row is a finding only when the min..max
ranges do NOT overlap. Apple M4 (10 cores, 16 GB), AC power, no thermal
warnings. Re-verdicted 2026-08-19 late night at commit be226f8 (split-K Gram
kernel in; k-NN AUTO; DBSCAN RBC + two-loop).

## Fixed-size arms (3 rounds, n = 15/arm)

| arm | shape | ours ms | sklearn ms | ratio | verdict |
|---|---|---|---|---|---|
| ols | 4M x 32 | 64.3 | 821.4 | 12.77x | **ours faster** |
| ols vs `ols_normal_eq` | same | 64.3 | 143.0 | 2.2x | **ours faster** |
| knn | 400k idx, 4k q, d=32, k=10 | 693.2 | 916.4 | 1.32x | **ours faster** |
| pca | 4M x 32, 8 comp | 166.0 | 119.8 | 0.72x | INDISTINGUISHABLE |
| dbscan | 4k x 16 | 11.2 | 8.9 | 0.80x | INDISTINGUISHABLE |
| kmeans | 4M x 32, k=64, 20 iter | 2,427.2 | 2,016.6 | 0.83x | sklearn faster |

**OLS flipped from the board's honest loss to its biggest win in one
change.** Before the split-K Gram kernel: 362 ms of solve, 2.6x BEHIND
`Ridge(alpha=0, solver="cholesky")` -- the same algorithm class as ours.
After: 64 ms, 2.2x AHEAD of it, and 12.8x ahead of `LinearRegression`'s SVD
route. **PCA moved 465 -> 166 ms in the same change**: from a clean 4x loss
to range-overlap with scikit-learn (their LAPACK arm still medians ahead;
the remaining ~100 ms of ours is two full centering passes over 512 MB, and
fusing them into the Gram read is the named next step). kmeans is now the
only clean fixed-size loss and the only unprofiled row.

## The dimensionality sweep (n = 200,000 fixed; NEW, and it redraws the map)

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
to the incumbent -- and even there, 200k is a tie in this window. At any
realistic feature width the fit is a multi-x win.

## Scaling arms at d=8 (2 rounds; the incumbent's best corner)

knn (d=32, k=10, 2,000 q): 1.41x / 1.36x / 1.79x / 1.48x ours at
50k/100k/200k/400k, all clean. dbscan (d=8): ours 2.35x/1.74x at 4k/16k,
parity 100k-200k, sklearn 0.70x-0.83x ahead at 400k-800k. Attribution
(phase timer, 400k+800k): ~88% of phase time is the `vertexdeg`
eps-neighborhood kernels -- a query-kernel constants fight against kd-tree
asymptotics in their best regime, now the LOWEST-priority gap on the board
given the d-sweep.

## What moved this round, measured

- **Split-K Gram kernel** (`core/gram_splitk.mojo`, LANE_gram-splitk): MAX's
  matmul runs the 32x32x4M Gram shape at ~25 GFLOP/s because its Apple arm
  launches one threadgroup per 64x64 output tile, and its split-K paths are
  comptime-gated `not has_apple_gpu_accelerator()` (verified in Modular's
  source at max/v26.5.0). Ours: 240 fixed k-chunks, deterministic ascending
  fold, zero transposes. The Gram call: 345 -> 49 ms. OLS solve: 362 -> 65.
  PCA fit: 465 -> 166.
- **RBC two-loop max_k**: 1.31x-1.57x clean at 200k-800k, labels identical.
- **k-NN AUTO default** (DEVIATION 36 revised): fused iff the ported launch
  computation says grid_x == 1; verified across k=10..64.
- **DBSCAN batch-size lever confirmed dead**; cuML's 80% default vindicated.

## What the table says to do next

1. **kmeans, the last unprofiled row** (0.83x, clean loss this window):
   per-iteration cost sits ~9x above its traffic bound -- the starved-
   parallelism signature that explained every other gap this week.
2. **PCA's outright win**: fuse mean-subtract into the split-K Gram read and
   drop the restore pass (~100 ms of the remaining 166).
3. **Target-keying debt**: `launch_config_generator`'s M4 constants, the
   split-K dispatch predicate/chunk count, and the k-NN AUTO decision are
   Apple-fed; on NVIDIA/AMD they must come from the kernel matrix's target
   column, and non-Apple targets should hand the Gram shape back to MAX's
   own split-K. Correctness is unaffected (scheduling only); NVIDIA/AMD
   remain supported-not-validated.
4. **Upstream report to Modular**: two confirmed silent-wrong bugs
   (`transpose_b` n==1 non-write; col-major views) + the Gram-shape gap,
   with the 345 -> 49 ms reproducer.
5. The split-K call itself has ~3x left to its traffic floor (~15 ms):
   hoist the shared load when 256 % m == 0; move the workspace alloc off
   the timed path.
