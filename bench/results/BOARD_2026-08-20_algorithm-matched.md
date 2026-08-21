# The board, with every unmatched arm removed. 2026-08-20, M4, quiet box.

    pixi run -e bench python bench/run_bench.py --mojo-bin build/bench_main --rounds 3

Three rounds alternating ours/theirs inside one invocation, 15 samples a
side, medians reported, a row is a finding only when the [min, max] ranges
are DISJOINT. scikit-learn 1.9.0, `n_jobs=-1` where accepted.

**Every row below is algorithm-matched. The device is the only variable.**

| arm | shape | ours ms | sklearn ms | ratio | verdict |
|---|---|---|---|---|---|
| ols | 4M x 32 | 34.79 | 149.22 | **4.29x** | ours faster |
| pca | 4M x 32, 8 comp | 37.63 | 132.53 | **3.52x** | ours faster |
| kmeans | 4M x 32, k=64, 20 it | 759.49 | 2,055.22 | **2.71x** | ours faster |
| knn | 400k idx, 4k q, d=32, k=10 | 695.94 | 943.53 | **1.36x** | ours faster |
| dbscan | 4k x 16 | 11.75 | 9.08 | 0.77x | INDISTINGUISHABLE |

Ranges: ols [33.41, 37.31] vs [143.66, 192.31]; pca [36.22, 43.09] vs
[120.66, 189.47]; kmeans [745.31, 1249.87] vs [1911.38, 2457.05]; knn
[687.90, 755.36] vs [880.83, 1201.55] -- all DISJOINT. dbscan
[8.45, 16.58] vs [8.62, 11.65] -- OVERLAP, no verdict.

**The board now emits five arms and every one is algorithm-matched.** Two
were deleted today for failing that bar, both unfair in OUR favour; a third
became an assertion. See the corrections below.

## The spread across three invocations of this board today

A single median is not a stable quantity on this box, so here is every run
this afternoon rather than the flattering one. Runs 2 and 3 are after the
`ols` arm was replaced; run 1's `ols` is the deleted gelsd arm and is struck
out rather than averaged in.

    arm       run 1    run 2    run 3     read as
    ------  -------  -------  -------  ----------
    ols     (25.6x)   4.31x    4.29x   ~4.3x
    pca       4.30x   3.38x    3.52x   ~3.4-3.5x  (run 1 sklearn sample noisy)
    kmeans    2.96x   2.73x    2.71x   ~2.7-3.0x
    knn       1.45x   1.35x    1.36x   ~1.35-1.45x
    dbscan    0.86x   0.86x    0.77x   TIE at every run

Quote the ranges, not the best cell.

## Three corrections this round made, each of which lowered our own number

**1. The OLS arm was deleted, not adjusted.** `ols` was
`LinearRegression(fit_intercept=False)`, which is LAPACK `gelsd`: an SVD of
the full 4,000,000 x 32 design. It printed **25.6x** in the run before this
one and it was the headline of the board. It is a different algorithm doing
substantially more work, and the more numerically stable one -- we are partly
faster because we do less and are more fragile on collinear X. `ols` is now
`Ridge(alpha=0, solver="cholesky")`, the normal equations, which is our
route. **25.6x -> 4.31x.** The import and the docstring table were purged
too, and a note at the call site says why it must not come back.

**2. `pca` at 4.30x was a noisy sample, and `auto` really is our solver.**
The previous run read sklearn PCA at 162.18 ms against `pca_cov_eigh` at
120.77, a 34% gap that looked like `auto` choosing a different solver at
4M rows -- the file's claim had only ever been verified at 200,000. Probed
directly (`_fit_svd_solver` after fit): `auto` resolves to
`covariance_eigh` at **both** 200,000 x 32 and 4,000,000 x 32. This run
reads 127.21 vs 118.31, which is the agreement that claim predicts.
**PCA is genuinely apples-to-apples and the number is 3.38x, not 4.30x.**

**3. `dbscan_brute` is NOT our matched arm and its comment said it was.**
Read from `bench/bench_main.mojo:252`: our arm calls `dbscan_fit_impl` with
the default `eps_nn_method`, which is `EPS_NN_RBC` -- a ball-cover INDEX.
The source comment read "All n^2 pairs, like ours", which has been false
since RBC became the default. Their unindexed path against our indexed one
is unfair IN OUR FAVOUR. The matched arm is `dbscan` (their `auto` picks a
kd-tree or ball tree): spatial index against spatial index, each side's own
best. `dbscan_brute` has been DELETED from the harness rather than relabelled:
a one-sided row that flatters us has no place on this board.

**4. `pca_cov_eigh` is no longer timed; it is an assertion.** It was never
unfair -- it is the same solver `auto` picks -- but two timings of one solver
is noise, and on run 1 they differed 34% from sampling alone, which invited
exactly the misreading it existed to prevent. The harness now probes
`_fit_svd_solver` once and HARD-FAILS if scikit-learn's `auto` ever stops
choosing `covariance_eigh`. Stronger guard, no clutter.

## What this board does not cover

- **DBSCAN at 4k x 16 is a toy** and the tie there says little. The large
  cells are unmeasured this window: the sweep that would have supplied them
  crashed, see `DEFECT_2026-08-20_dbscan-rbc-maxk-large-highd.md`.
- **The d=8 large-n DBSCAN loss is not on this board and has not been
  re-run.** `SCOREBOARD_2026-08-19.md` records sklearn 0.70x-0.83x ahead at
  400k-800k and then tallies "zero losses", which is wrong on its own
  evidence.
- Our GPU against their CPU throughout. sklearn's estimators either refuse
  MPS outright or run slower on it than on torch CPU
  (`SKLEARN_GPU_BASELINE_2026-08-20.md`), so there is no GPU arm on their
  side to run on this machine. That is the thesis, and it belongs beside
  every ratio here.
- One machine, one window, synthetic fixtures.
