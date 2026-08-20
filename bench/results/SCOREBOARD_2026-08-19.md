# The thesis scoreboard: our GPU against their best CPU, same Mac

**The thesis this repo exists for is GPU ACCESS, not GPU tier: the
incumbents' GPU arms cannot run on Apple silicon at all, so the comparison a
Mac user actually lives with is our Metal path against their best CPU on the
same machine.** This is that table, taken 2026-08-19 evening at the shipped
defaults (commit b2d438c: DBSCAN = RBC + two-loop max_k, k-NN =
KNN_METHOD_AUTO), against scikit-learn 1.9.0 with `n_jobs=-1` on every
estimator that accepts it and their `algorithm='auto'` -- their best, not a
crippled arm.

Method: `bench/run_bench.py`, arms alternated per round inside one
invocation, medians reported, and a row is only a finding when the two
min..max ranges do NOT overlap ("INDISTINGUISHABLE" otherwise). Environment
at close: Apple M4 (10 cores, 16 GB), macOS 26.5.2, AC power, no thermal
warning recorded, Python 3.14.6 / numpy 2.4.4 / scikit-learn 1.9.0.

## Fixed-size arms (3 rounds, n = 15/arm)

| arm | shape | ours ms | sklearn ms | ratio | verdict |
|---|---|---|---|---|---|
| kmeans | 4M x 32, k=64, 20 iter | 2,700.2 | 2,127.4 | 0.79x | INDISTINGUISHABLE |
| knn | 400k idx, 4k q, d=32, k=10 | 769.6 | 961.5 | 1.25x | **ours faster** |
| pca | 4M x 32, 8 comp | 520.0 | 130.4 | 0.25x | sklearn faster |
| ols | 4M x 32 | 397.8 | 866.2 | 2.18x | **ours faster** |
| ols vs `ols_normal_eq` | same | 397.8 | 152.4 | 0.38x | sklearn faster |
| dbscan | 4k x 16 | 9.5 | 9.1 | 0.95x | INDISTINGUISHABLE |

**The honest OLS row is the last-but-one's neighbour, not the 2.18x.**
`LinearRegression` takes LAPACK's SVD route; `Ridge(alpha=0,
solver="cholesky")` forms X^T X and solves it -- the SAME algorithm class as
our lstsq_eig -- and at 152.4 ms it beats us 2.6x. The 2.18x is a win over
an algorithm we did not implement. Both are printed so the reader can pick,
and the headline is the loss.

## Scaling arms (2 rounds, n = 6/arm, splitmix64 fixtures)

k-NN: ours = the shipped AUTO dispatch (fused at these shapes), d=32, k=10,
2,000 queries. sklearn = `NearestNeighbors(algorithm='auto', n_jobs=-1)`.

| n index | ours ms | sklearn ms | ratio | verdict |
|---|---|---|---|---|
| 20,000 | 26.9 | 31.2 | 1.16x | INDISTINGUISHABLE |
| 50,000 | 49.4 | 69.4 | 1.41x | **ours faster** |
| 100,000 | 117.0 | 159.5 | 1.36x | **ours faster** |
| 200,000 | 236.0 | 422.8 | 1.79x | **ours faster** |
| 400,000 | 514.9 | 763.2 | 1.48x | **ours faster** |

DBSCAN: ours = the shipped RBC default with the two-loop max_k dispatch,
d=8, eps=0.30. sklearn = `DBSCAN(algorithm='auto', n_jobs=-1)` (a kd-tree
at this dimensionality).

| n | ours ms | sklearn ms | ratio | verdict |
|---|---|---|---|---|
| 4,000 | 7.9 | 18.6 | 2.35x | **ours faster** |
| 16,000 | 22.7 | 39.4 | 1.74x | **ours faster** |
| 50,000 | 121.3 | 92.7 | 0.76x | sklearn faster |
| 100,000 | 228.9 | 216.1 | 0.94x | INDISTINGUISHABLE |
| 200,000 | 528.8 | 514.3 | 0.97x | INDISTINGUISHABLE |
| 400,000 | 1,615.8 | 1,345.8 | 0.83x | sklearn faster |
| 800,000 | 5,052.8 | 3,539.4 | 0.70x | sklearn faster |

## What moved today, measured

**The RBC two-loop max_k dispatch (LANE_rbc-maxk) is a clean win at scale.**
A/B at the scaling fixture, HEAD against pre-lane e4eb7cc, alternated,
labels identical:

| n | before ms | after ms | speedup | verdict |
|---|---|---|---|---|
| 50,000 | 232.9 | 157.1 | 1.48x | overlap |
| 200,000 | 900.2 | 588.5 | 1.53x | **after faster** |
| 400,000 | 2,709.7 | 1,720.6 | 1.57x | **after faster** |
| 800,000 | 7,375.8 | 5,620.0 | 1.31x | **after faster** |

(The scoreboard's DBSCAN rows above were taken in a later, cooler window
than this A/B; compare within a table, never across tables.)

**The k-NN default flipped to AUTO (DEVIATION 36 revised, PORTING.md 36).**
With their `launchConfigGenerator` ported and M4-fed, fused never loses a
median in the `grid_x == 1` regime and wins 1.25x clean at 32,000 queries;
the x-split regime loses catastrophically (0.19x at 500 queries), so AUTO
takes fused iff the launch computation says `grid_x == 1`. At the fixed
bench shape (400k x 4,000) a same-window interleaved check measured fused
and tiled a dead tie (759 vs 750 ms median, full overlap), which also
explains the knn fixed-row movement between windows as thermal drift, not
the flip.

**The DBSCAN batch-size lever is dead.** On the two-loop runner the phase
timer attributes every added batch as pure re-run `vertexdeg` cost
(`weak_cc` <= 1.3 ms/fit at 50k, acquitted); cuML's 80%-one-batch default is
right on this device too. Details in DBSCAN_RBC_2026-08-19.md.

## What the table says to do next

1. **PCA is the worst row (0.25x) and the covariance matmul is NOT the
   cause** (LANE_covariance-unblock: `gemm_tn` has served the T-N shape in
   PCA/tSVD/OLS since 048f3da and caps at ~50-60 ms of the 520; the zero-copy
   col-major route is BANNED, silently wrong across most shapes). The 4x and
   the OLS 0.38x need a per-phase timer; the reading-visible suspects are
   `xty_kernel`'s 32-block launch and the fit's sync/alloc pattern.
2. **DBSCAN at 400k-800k trails 0.70x-0.83x**: sklearn's kd-tree asymptotics
   at d=8 against our RBC constants. The gap is flat, not diverging.
3. **kmeans is a tie that shouldn't be**: 4M x 32 is exactly the shape a GPU
   should own; nobody has profiled the fit since the control-plane rounds.
