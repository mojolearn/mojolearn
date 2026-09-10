# Mac FAST, classical estimators: standings and the next lanes (2026-09-10)

Directive (Andrew, 2026-09-10 evening): performance work is trees and
classical only, the FAST tier on the MacBook and the IDENTICAL tier on
NVIDIA. This file is the Mac FAST half: what was landed tonight, where every
Python-visible classical estimator stands against scikit-learn on the M4,
and the next lanes in order.

## Landed tonight

| commit | what | measured (Apple M4, FAST) |
|---|---|---|
| 9f25051e | KernelDensity fused score pass, DEVIATION 2490 | 16k x 16k x 8: sklearn KDTree 7,671 ms, ours 30.3 ms (was 540.7 ms); 100k x 100k: 499 ms, previously unallocatable. `bench/results/kde_fast_2026-09-10/` |
| b1ba346d | SVM: shuffle arg-reductions in the block solve (2491, all tiers), fused FAST RBF tile (2492), SMO stage clock | HIGGS 50k x 28 fit 7.45 s to 1.87 s; vs sklearn libsvm at 20k: fit 10,561 vs 531 ms, predict 10k 6,561 vs 68 ms. `bench/results/svm_fast_2026-09-10/` |
| a93bef9d | SVR rows | 20k: 10.9 s vs 1.2 s fit, RMSE equal |

The recurring shape of the win: upstream materializes an `n x m` matrix
between library calls (numba kernels, cub, cuBLAS) and walks it again;
one fused pass over tiles in shared memory removes the matrix, the
traffic and the batching that its size forced.

## Standings, M4 FAST vs scikit-learn 1.9 on ten cores

Quick race, min of two after a warm-up, same data both arms, 2026-09-10.
Not a certified window (no interleaving beyond one pass); it ranks the
lanes, it does not price them.

| estimator | shape | ours | sklearn | sklearn / ours |
|---|---|---:|---:|---:|
| PCA (fit, 8 components) | 2M x 32 | 65 ms | 643 ms | 9.9x |
| KMeans (fit, k 64, 20 iter) | 2M x 32 | 1,311 ms | 6,433 ms | 4.9x |
| LinearRegression (fit) | 2M x 32 | 163 ms | 442 ms | 2.7x |
| KernelDensity (score) | 16k x 16k x 8 | 30 ms | 7,671 ms | 253x |
| SVC (fit) | 20k x 28 | 531 ms | 10,561 ms | 19.9x |
| NearestNeighbors (fit + kneighbors, k 10) | 400k x 32, 4k queries | 767 ms | 889 ms | 1.16x |
| NearestNeighbors | 1M x 16, 10k queries | 2,509 ms | 3,840 ms | 1.53x |
| NearestNeighbors (k 5) | 100k x 8, 100k queries | 2,534 ms | 2,866 ms | 1.13x |
| DBSCAN (fit, eps .3, min 5, uniform [0,4)^8) | 200k x 8 | 653 ms | 352 ms | 0.54x |
| DBSCAN (eps .5, min 10, normal) | 200k x 8 | 2,287 ms | 1,040 ms | 0.45x |
| DBSCAN (eps .3, min 5, uniform) | 1M x 8 | 7,811 ms | 3,026 ms | 0.39x |

RBFSampler and Nystroem (kernel_methods) lose their ladder at 1e6 rows
(0.30x, 0.08x) but have no Python surface, so they are not customer-visible
and are not ranked here.

## Next lanes, in order

1. **DBSCAN fused eps-neighborhood (FAST).** The loss is structural: the fit
   runs 20 batches at 200k rows because the eps-neighborhood is a
   materialized `batch x n` matrix (`dbscan/impl/runner.mojo`, the RBC
   arm's `rbc_eps_nn_query_count` then the adjacency), and each batch's
   vertexdeg is ~15 ms. A count pass (one query thread, index rows through
   shared memory, `dist2 <= eps2` counted) then an exclusive scan then a fill
   pass writing the CSR directly, the KDE/SVM tile shape, removes the matrix
   and the batching. Keep the RBC pruning if the landmark structure is
   cheap to consult per tile; otherwise brute is fine at d = 8. Gate:
   labels equal to the current FAST arm on the phase fixture and the
   `dbscan/checks`, then the race above at 200k and 1M. Owner: none.
2. **SVM `select_ws`.** 0.58 s of a 2.2 s fit at 50k: a 32-pass one-bit LSD
   radix sort of `f` over n_train, four launches per bit, about 130 launches
   per outer iteration. Either a 4-bit stable radix pass (shared with GBDT's
   quantiles, so IDENTICAL-gated by fingerprints) or a FAST k-selection of
   the n_ws/2 smallest and largest eligible `f` (radix select with the
   (value, index) tie rule so the set equals the sort's). Stage clock:
   `MOJOLEARN_STAGE_TIMES=1` on any SVC/SVR fit.
3. **kNN brute force on the Mac.** 4 Gcells/s at 400k x 32 / 4k queries is
   far below the fused kernels above (19 Gcells/s at d = 8 for KDE with an
   exp per cell). But `lane/knn-selector` (2026-09-09, unmerged, 8453e883
   unbuilt) owns the kNN kernels and the H100 IDENTICAL column just moved
   (bench/OPPONENT_REFERENCE.md, Sep 10 rows); coordinate before touching
   `neighbors/impl/detail/fused_l2_knn.mojo`.
4. **SVC predict host boundary.** 100k predictions vs 39k SVs took 783 ms,
   of which the fused kernel is roughly half; the rest is the List staging
   of X and the support matrix on every call (DEVIATION 873). Pointer-through
   and a resident support matrix, the WP2/WP3 moves from the trees brief.

## Open collision

A parallel session (Andrew's checkout) is drafting a one-tier rule: only
the tree lanes ship fast and deterministic, `bindings/build_svm.sh` and
`build_estimators.sh` refuse `MOJOLEARN_NUMERIC_MODE=fast`. That rule and
tonight's KDE/SVM FAST arms cannot both stand. The rule's own text admits
a fast tier "where it has a measured win over the opponent's own CPU"; the
KDE and SVC rows above are those measurements. Andrew decides.
