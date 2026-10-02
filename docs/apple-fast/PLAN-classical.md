# Apple FAST classical: state and plan (from lane apple-fast-classical, 2026-10-02)

Branch `lane/apple-fast-classical`, head ee03623fa (pushed), merged into `lane/apple-fast`
(ce7b0ebf2, no conflicts). Tools: `tools/afc_board_summary.py` (classical rows of a board.json,
worst FAST/best-opponent first), `tools/afc_ab.sh <tag> <lane> <ds> <reps> <rounds> <envA> [<envB>]`
(alternating FAST A/B of ONE build by env switch, board shape rows-full; AFC_FAMILY=algos|classical2|classical,
AFC_ARM=ours for IDENTICAL; log at ~/mq/out/race-<tag>/race.log, grep `AFC-AB-SUMMARY|AFC-FAIL`).

## Commits (all FAST + Apple comptime; IDENTICAL compiles the old path). NOTHING MEASURED YET.

| commit | change (cause file:line) | A/B switch (runtime env, `=0` = old path) |
|---|---|---|
| b86277359 | BayesianRidge/ARD: per-iteration sse from the normal equations (yy - 2w'q + w'Gw; Bayes in the eigenbasis O(d), ARD O(d^2) on the team) instead of `_t_sse`, a pass over all rows on ONE block every iteration (`x_linear/bayes.mojo` `_t_sse` calls in `bayes_ridge_fit` / `ard_fit`; board Istella 92 s / 48 s vs sklearn 6.8 / 10.4) | `MOJOLEARN_X_LINEAR_GRAM_SSE=0` (ip[5] set in `x_linear/device.mojo` fit_device) |
| 282545199 | LassoCV/ElasticNetCV: new `x_linear/enetcv_fast.mojo`, every row pass on the grid: per-fold sums + per-fold centered Grams of [X,y] (8192-row chunks, 32x32 tiles), training-set Grams by the parallel-axis rule, CD paths one block per (fold,l1_ratio), held-out MSE on the grid, choice + refit on device, one readback; fold ids checked vs KFold bounds on device (mismatch -> team fit). Cause: `x_linear/cd.mojo` `_prep_team` (lead-built row lists, one thread per Gram cell over all rows, 7x) on ONE block (board taxi 3.2 s / Istella 74 s vs 0.2 / 7) | `MOJOLEARN_X_LINEAR_ENETCV_FAST=0` |
| 6a642df25 | k-NN past 32 features: `fast_mma_bigd_kernel` in `neighbors/impl/detail/fast_mma_knn.mojo` (128-row x 32-feature shared tile, simdgroup MMA accumulated over feature chunks, norms kernel, same admission/fold/merge). Cause: d>32 fell to scalar/tiled FAST arms, 2-4x slower than IDENTICAL's Apple MMA tile (board knn-clf Istella FAST 835 vs IDENT 213 ms; knn 1592 vs 768) | `MOJOLEARN_KNN_FAST_MMA_BIGD=0` |
| ee03623fa | NearestCentroid stats chunked over rows too (`x_neighbors/iter_device.mojo` op_nc_stats: nc_means_kernel/nc_std_kernel were 14 blocks at 220 features each walking every row) | `MOJOLEARN_XN_NC_CHUNKS=0` |

All compile FAST and IDENTICAL on the laptop (x_linear, x_neighbors, core). Correctness/quality unverified on Metal.
IDENTICAL M3 hash check owed (all FAST-gated): identical races of bayesian-ridge, ard, lasso-cv, enet-cv, knn-clf, knn, nearest-centroid at head vs main.

## Queued on m3 (branch lane/apple-fast-classical; results UNREAD)

| # | tag | what |
|---|---|---|
| 37 | afc-base2 | FAST baseline (builds core): knn-clf/knn-reg Istella, nystroem, rbf-sampler, gpr, lasso, elasticnet (classical2), knn/pca Istella (classical). NOTE nystroem/rbf/gpr/lasso/elasticnet bindings not built in that job: expect REFUSED |
| 39 | afc-m1build | FAST build x_linear + core at head |
| 40 | afc-m1bayes | A/B bayesian-ridge, ard on taxi+istella: A new, B GRAM_SSE=0 (2 reps x 2 rounds) |
| 41 | afc-m1enet | A/B enet-cv, lasso-cv taxi+istella: A new, B ENETCV_FAST=0 |
| 42 | afc-m1knn | A/B knn-clf (classical2), knn (classical) Istella: A new, B MMA_BIGD=0 (3x3) |
| 44 | afc-dbg | afc_ab.sh debug (huber taxi 1x1) |

Done: #34 afc-build1 (FAST x_* builds ok), #35 afc-base1 VOID (afc_ab.sh used timeout(1), absent on macOS; fixed 7c0bcdca2+). The M3 resolves the branch head at run time, so jobs 39-44 run at the latest pushed head of lane/apple-fast-classical (ee03623fa, NC change included but x_neighbors not rebuilt by m1build). Requeue the baseline (base1 list) for real FAST numbers at head.

## Worst FAST classical rows, M3 0834 board (FAST ms / best opponent ms; board is wheel 0.8.34, main moved since)

huber taxi 168,025/1,855 (90x; FAST 8x slower than IDENTICAL 20,184; main now has the Huber grid, likely fixed); knn-imputer taxi 44x; enet-cv taxi 15x, Istella 10.6x; lasso-cv taxi 14x, Istella 12.8x; lle taxi 14x, Istella 6.3x; bayesian-ridge Istella 13.5x (FAST r2 -4.2e4 vs sklearn -890: quality already below opponent), taxi 6x; meanshift Istella 13x; connected-components 12-13x; label-encoder 11-13x; label-binarizer 10-12x; isotonic 12-16x; theta taxi-hourly 10x (FAST 1,747 vs IDENTICAL 388); minmax-scaler Istella 5.7x; multilabel-binarizer 5.4-5.6x; nearest-centroid Istella 5.5x, taxi 4.9x; minibatch-kmeans Istella 5.5x; lars/lasso-lars taxi 5.3x; ridge-clf taxi 5.3x; lda-clf Istella 5.1x; pca Istella 4.8x (FAST 988 vs IDENT 814); ard Istella 4.6x; incremental-pca 3.5x; onehot/ordinal taxi 3.6-3.8x; knn-clf/knn-reg Istella 2.9x (FAST 4x slower than IDENTICAL); knn Istella 2.8x; qda Istella 2.6x; kernel-pca 1.8-1.9x; nystroem taxi 2.0x; rbf-sampler Istella 1.8x (FAST slower than IDENTICAL).
Neural-ish rows (pools, conv, optimizers, graphsage, moe) are out of scope.

## Next experiments, ranked (cause file:line)

1. Read 40/41/42; flip nothing (new paths are already default-on under FAST Apple); if quality/speed fail, default the switch to 0.
2. FAST slower than IDENTICAL rows (cheap wins: take IDENTICAL's arm in FAST): theta (`sequence/theta.mojo` op_theta, one thread per series), rbf-sampler, pca Istella. Measure FAST vs IDENTICAL at head first (afc_ab.sh with AFC_ARM=ours).
3. PCA Istella: one-block Jacobi `jacobi_eigh_kernel` grid 1 at 220x220 (`decomposition/impl/linalg/detail/pca.mojo` eig_and_truncate ~line 262) plus host ordering; FAST: x_decomp's jacobi2 or a top-k GPU solver (trace gives ratio/noise); also unused n*d `xa2` buffer (`decomposition/estimator.mojo` pca_fit_host) in the fused split-K path.
4. Lars/LassoLars/ridge-clf/RidgeCV: same Gram-on-grid pattern as enetcv_fast (one-block `t_centered_gram`/xty in `x_linear/lars.mojo`, `ridge.mojo`); reuse enetcv_fast's sums/gram kernels (generalize ef_gram_kernel into a shared FAST grid Gram).
5. Isotonic (12-16x): `x_linear/isotonic.mojo` PAVA on one block; FAST parallel PAVA (block-merge).
6. knn-imputer (44x), lle (14x): kNN inside; check they route through fast_mma_knn (now d>32 capable) — `x_neighbors` knn_sq_item path likely bypasses it.
7. label-encoder/binarizer, multilabel, onehot/ordinal, minmax: Python host overhead per label (`python/mojolearn/_expansion_prep.py`); profile at head (neural-pass137 moved some to device after 0.8.34).
8. meanshift Istella 13x (`x_cluster`, one block per seed; MOJOLEARN_MEANSHIFT_BLOCK WIP opt-in exists, unbuilt).
9. lda-clf/qda Istella: covariance per class on one block (x_prep), same grid-Gram pattern.
