# Wave 2: leftovers after the first family pass

The rules are the same as in COMMON_BRIEF.md. Code changes only; run nothing.
Files under the c-svm, c-cluster and c-metrics-prep prefixes stay off-limits while those lanes run.

- w2-core-scope: drop the core/host_lanes, host_predict_threads, classical_host_predict, knn_host_predict and forest_host_predict imports from GPU-scoped code; make forest_inference_model's scan_finite_f32 a GPU scan; cut the linalg GPU binding's path to decomposition/host/linalg_public, pca_oracle and identical_gemm.
- w2-linear: the x_linear fit_kernel one-block fits (Lasso, Huber, logistic, weighted isotonic), the isotonic scans, xq_step, the per-sample SGD lane==0 loop, the blocks_gram tid0 fold and the cholesky/checks/trsm round trips.
- w2-trees: xtrees weighted_sample (fixed-order f64 prefix sum on the device), the GBDT non-GreedyLogSum border builders, boost_from_average, the _expansion_trees.py per-row loops and the ensemble.py _gbdt_host imports.
- w2-pyglue: a device label encoder in _labels.py, used by the np.unique rows (x_sequence, CNN); the NearestCentroid manhattan median and the OneClassSVM weight filtering; the Hessian/LTSA/modified-LLE per-sample Python loops; the GP Python loops.
