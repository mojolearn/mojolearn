# Cloud peer family branches: status index (2026-10-02)

Every branch is `lane/apple-fast-<family>` off `origin/lane/apple-fast`, FAST + Apple only, GPU only,
every change behind a default-off switch; IDENTICAL compiles the old code. None of it has been
compiled (no Mojo toolchain on the cloud side): the first M3 build of each branch is the compile
check, and each family's `<family>.md` names the risky sites. Queue lines live in `<family>.txt`.

## Finished (draft PR against lane/apple-fast; queue lines in docs/apple-fast/ab/<family>.txt)

| family | PR | lanes |
|---|---|---|
| tier | #132 | FAST slower than IDENTICAL: pca/ols/kmeans GEMM arms, rbf-sampler/nystroem pinned GEMM, theta fma + registers |
| pca-eig | #133 | pca, tsvd, incremental-pca |
| trees-scan | #134 | GBDT segmented sort scans, RF bootstrap sort, CTR vector scan |
| seq | #135 | croston, garch |
| gram | #136 | lars, lasso-lars, ridge-clf, ridge-cv, lda-clf, qda |
| trees-yeti | #137 | yetirank task kernel, symmetric histogram block |
| prep | #138 | minmax-scaler, onehot, ordinal, multilabel-binarizer |
| cluster | #139 | meanshift, minibatch-kmeans |
| trees-depthwise | #140 | gbdt-categorical CTR prep (fused chain is on lane/apple-fast-depthwise) |
| trees-io | #141 | iforest sampled upload and raw query path |
| kernel | #142 | bayesian-ridge, ard, gpr, nystroem |
| isotonic-knn | #143 | isotonic, tiled kNN MMA route (lof, label-prop), lle |
| linear | #144 | lasso, elasticnet, logreg, linearsvc, linearsvr, gmm |
| ann | #145 | tsne, cagra, ivf-pq, ivf-sq, ivf-rabitq, ivf-refine, ivf-filter, ivf |
| prep2 | #146 | target-encoder, simple-imputer, robust-scaler, iterative-imputer |
| cluster2 | #147 | affinity-prop, bayesian-gmm, bisecting-kmeans, optics |
| decomp-sparse | #148 | sparse-coder, dict-learning, sparse-pca, fastica, mds, isomap, gaussian-rp, sparse-rp |
| resample | #149 | bootstrap, permutation-test, resample, cross-val-score |

## Unfinished (the writing agent stopped on a rate limit; saved as-is, no PR)

| family | state on origin |
|---|---|
| core | 4 finished commits (kmeans, knn k<=64 MMA, kde slices, dbscan CSR scan) + a WIP commit in glm/estimator.mojo, bindings/_mojolearn_estimators.mojo, python/mojolearn/linear_model.py. No queue file. |
| decomp-linalg | 6 finished commits (eigh, lu-solve, qr, svd CholeskyQR2, tiled kit gemm, fused LU panel, pls, fa) + a WIP commit adding decomp-linalg.txt. No note. |
| neighbors2 | 4 finished commits with neighbors2.txt and .md (kNN via MMA, pagerank reductions, svgp device solve, three more). Looks complete; PR not opened. |
| tsa | one WIP commit: arima/impl/batched_arima.mojo, batched_kalman.mojo, tsa/arima_common.mojo, new arima/impl/fast_eval_ws.mojo. Unfinished, untested. |
| trees-ensembles | 1 finished commit with trees-ensembles.txt and .md (bagging members on one device X, native cv folds). |
| trees-symmetric | empty (no commit). |

## Known overlaps to resolve at merge
- trees-scan and trees-depthwise both parallelise `launch_scan_vector_u32` (gbdt/gpu_util/kernel/scan.mojo) under different defines: keep one.
- gram and kernel both edit x_linear/device.mojo (different functions).
- cluster and cluster2 both edit x_cluster/device_ops.mojo, ops.mojo, host/host_ops.mojo (different methods).
- prep and prep2, isotonic-knn and decomp-sparse each touch python/mojolearn/_expansion_prep.py / _expansion_decomp.py (different functions).
