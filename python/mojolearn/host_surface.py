# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CPU SURFACE OF MOJOLEARN, DECLARED ONCE (the host surface manifest
lane, 2026-09-14).

Before this file the CPU surface had no single statement in code. The
covered training lanes lived in the CPU identity gate's YAML env, the
routing table in `_backend._HOST_MODULES`, the inference lanes in two gate
tools, the recording directories in the workflow, and the README restated
all of it by hand, which is how it came to say k-NN had no CPU path the day
after the k-NN host lane merged. This file is the one source; everything
else READS it:

  `_backend._HOST_MODULES`           is `routed_modules()`
  .github/workflows/cpu-identity-gate.yml
                                     fills COVERED_LANES, HOST_FAMILIES,
                                     HOST_BINDINGS and CLASSICAL_RECORDED
                                     from the command line below, in one step
  tools/identity_break.py            records `host.surface` beside
                                     `host.families` in every CPU column
  tools/docs_facts.py                checks the marked spans of README.md,
                                     SUPPORT_MATRIX.md and
                                     docs/BYTE_LM_CPU_TRAINING.md against it
  python/mojolearn/tests/test_host_surface.py
                                     fails when a binding exports a name
                                     this file does not list, when this file
                                     lists a family with no binding source,
                                     when a build shim does not exec
                                     bindings/build_host_family.sh, or when a lane
                                     named here is unknown to the gate that
                                     is supposed to run it

Per host family the manifest declares: the binding basename under
mojolearn/host/, the build shim, the GPU family it routes on a CPU-only
install (None for the three bindings loaded by path, the byte LM's, the
forest's and the tokenizer's, which has no GPU binding at all), the sabotage define its
gate's negative control passes, the identity_break lanes it covers for
TRAINING (the CPU column must read STABLE and IDENTICAL x4 on them), the
lanes and public classes it serves for INFERENCE from a saved model, the
Mojo host modules that ship inside it, the function names it exports, and
whether it ships in a wheel (every family does since 0.8.6, the packaging
lane of 2026-09-14; the two wheel builders and the packer read
`--wheel-families` and `--wheel-bindings` below instead of naming the byte
LM's binding by hand, and packaging/check_ext_lists.py fails when any of
them carries a host list of its own).

This file imports nothing from the package on purpose. It runs by path
before the package can import (the gate runner has no binding built yet):

    python3 python/mojolearn/host_surface.py --covered-lanes
    python3 python/mojolearn/host_surface.py --routed-families
    python3 python/mojolearn/host_surface.py --bindings --sep ,
    python3 python/mojolearn/host_surface.py --classical-recorded
    python3 python/mojolearn/host_surface.py --wheel-families
    python3 python/mojolearn/host_surface.py --wheel-bindings
    python3 python/mojolearn/host_surface.py --markdown
    python3 python/mojolearn/host_surface.py --json

and `python3 -m mojolearn.host_surface ...` says the same thing on a box
where the package imports.
"""
import argparse
import json
import sys

#: Where this manifest lives, recorded into every CPU column.
SOURCE = "python/mojolearn/host_surface.py"

#: The one builder every family compiles through; bindings/build_<family>_host.sh
#: is a two-line shim that execs it with the family name.
BUILDER = "bindings/build_host_family.sh"

#: The GPU columns the TRAINING gate diffs the CPU column against
#: (cpu-identity-gate.yml, --require-columns 4 on the covered lanes). The
#: record must not lag the surface: the 2026-09-13 three-column record
#: predates the kde and svc model cells and the holtwinters infer cells the
#: CPU column now carries, so gate run 34832840859 at 7bf4f4cc9 failed on
#: all seven runners with "require-columns 4 ... 27 short". The 47-lane
#: record taken at 7bf4f4cc9 (pca-whiten included) was the one this surface
#: was diffed against until 2026-09-14 afternoon; the 136-lane record at
#: 4048e1b51 (the 2712/2713 fix, the sixteen one-device par-* lanes) is the
#: one now; the manifest step of the workflow fails, before any build, when a
#: column named here is not in the checkout. Its AMD column is the MI325X one
#: (DigitalOcean, gfx942): the MI300X column of the same record is incomplete
#: (the 60-minute Hot Aisle cap cut it before iforest, iforest-tuned and five
#: par-* lanes), and iforest is a covered lane, so require-columns 4 could not
#: hold against it.
TRAINING_GPU_COLUMNS = (
    "bench/results/identity_break/2026-09-14_136-lanes/apple-m4.json",
    "bench/results/identity_break/2026-09-14_136-lanes/nvidia-h100-sm_90a.json",
    "bench/results/identity_break/2026-09-14_136-lanes/amd-mi325x-gfx942.json",
)

#: The GPU columns the classical INFERENCE gate compares each host identity
#: hash against (tools/classical_host_gate.py check --gpu-column).
CLASSICAL_GPU_COLUMNS = (
    "bench/results/identity_break/2026-09-14_46-lanes/apple-m4.json",
    "bench/results/identity_break/2026-09-14_46-lanes/nvidia-h100-sm_90a.json",
    "bench/results/identity_break/2026-09-14_46-lanes/amd-mi300x-gfx942.json",
)

#: The classical inference recordings (one directory per GPU box and lane
#: set; every fixture under each must be RECORDED or the gate exits 2).
CLASSICAL_RECORDED = (
    "bench/results/classical_host/2026-09-13-apple-m4",
    "bench/results/classical_host/2026-09-14-nvidia-h100",
    "bench/results/classical_host/2026-09-14-amd-mi300x",
    "bench/results/classical_host/2026-09-14-apple-m4-kde-svc",
    "bench/results/classical_host/2026-09-14-apple-m4-knn",
    "bench/results/classical_host/2026-09-14-nvidia-h100-b",
    "bench/results/classical_host/2026-09-14-amd-mi300x-b",
    "bench/results/classical_host/2026-09-14-apple-m4-multiclass",
)

#: The forest inference recordings: every directory under this root whose
#: expected.json says RECORDED (the workflows sort them at run time).
FOREST_RECORDED_ROOT = "bench/results/forest_host"

#: The identity_break lanes with a CPU TRAINING path, in the gate's order,
#: with the name the docs use for each.
TRAINING_LANE_NAMES = {
    "gemm-pinned": "pinned GEMM",
    "kde": "kernel density",
    "holtwinters": "Holt-Winters",
    "lasso": "lasso",
    "elasticnet": "elasticnet",
    "svc": "SVC",
    "agglomerative": "agglomerative clustering",
    "et-clf": "the Extra Trees classifier",
    "et-reg": "the Extra Trees regressor",
    "iforest": "the isolation forest",
    # Workstream E (lane/cpu-training-e, 2026-09-14). The k-NN lanes' fit
    # stores the index and their train cell is the host search the knn host
    # inference lane already serves; pca, pca-whiten, tsvd, ols and ridge
    # train through decomposition/host/pca_oracle.mojo and
    # glm/host/glm_oracle.mojo. The seven-runner gate read the four-column
    # diff IDENTICAL on every covered cell at 2b7f991b6 (run 34871479957).
    "knn": "nearest neighbors",
    "knn-clf": "the k-NN classifier",
    "knn-reg": "the k-NN regressor",
    "pca": "PCA",
    "pca-whiten": "whitened PCA",
    "tsvd": "truncated SVD",
    "ols": "linear regression",
    "ridge": "ridge",
    "dbscan": "DBSCAN",
    # Workstream E batch 2 (lane/cpu-training-e2, 2026-09-14): k-means trains
    # through cluster/host/kmeans_oracle.mojo, exported as kmeans_fit from
    # the core host binding. The seven-runner gate read all nine fixtures
    # IDENTICAL x4 (run 34884487749).
    "kmeans": "k-means",
    # Workstream E batch 2: the five metrics of the lane (accuracy, ARI,
    # v-measure, r2, silhouette) through metrics/host/metrics_oracle.mojo,
    # the metrics family's own host binding; the lane also fits a KMeans,
    # served by the core family above.
    "metrics": "the metrics",
    # The spectral lane, same batch: the spectral host oracle moved to
    # spectral/host/spectral_oracle.mojo with the k-NN graph, the symmetrize
    # kernel and the k-means recluster restated, exported as
    # spectral_fit_predict_dataset from the metrics host binding.
    "spectral": "spectral clustering",
    # The two scaler lanes, same batch: preprocessing/host/scaler_oracle.mojo
    # through the preprocessing family's own host binding.
    "standard-scaler": "the standard scaler",
    "minmax-scaler": "the min-max scaler",
    # The logistic lane, same batch: glm/host/qn_oracle.mojo, the L-BFGS
    # arm of the quasi-Newton solver, exported as qn_fit from the
    # estimators host binding.
    "logistic": "logistic regression",
    # Workstream E batch 3 (2026-09-14): the random forests train through
    # ensemble/host/rf_oracle.mojo, the device trainer restated on the host,
    # exported under the GPU binding's names from the rf family's own host
    # binding. Gate run 34884487749 at ed6c06526: all nine fixtures
    # IDENTICAL x4 (train, infer and model), the sabotage build DIVERGENT on
    # every one.
    "rf-clf": "the random forest classifier",
    "rf-reg": "the random forest regressor",
    # CPU training batch 2 declared (lane/cpu-training-batch2-declare,
    # 2026-09-14): lanes the host bindings above already serve that the
    # 47-lane record lacked, so they could not be declared against it; the
    # 136-lane record carries every one, IDENTICAL on its three columns.
    # k-means with the random start (three restarts), an explicit start and
    # sample weights, all through the core family's kmeans_fit and
    # host_fit_main, whose quantized accumulation carries the sabotage unit.
    "kmeans-random": "k-means with a random start",
    "kmeans-array": "k-means from given centroids",
    "kmeans-weighted": "weighted k-means",
    # The scalers without centering, without scaling, and with a
    # non-default range and the clamp, through the preprocessing family's
    # host binding (the shifted slab tree reaches the first two, the
    # min-max offset arm the third).
    "standard-scaler-no-mean": "the standard scaler without centering",
    "standard-scaler-no-std": "the standard scaler without scaling",
    "minmax-scaler-clip": "the clipped min-max scaler",
    # Spectral clustering on a precomputed affinity: spectral_fit_predict_graph
    # in the metrics host binding over host_spectral_fit_predict_coo, and the
    # dense affinity's COO scan (nonzero_f64_count, nonzero_f64_fill) in the
    # core host binding.
    "spectral-precomputed": "spectral clustering on a precomputed affinity",
    # Workstream E, the gp host lane (2026-09-14): the Gaussian process fit
    # and predict through gaussian_process/host/gpr_oracle.mojo over
    # cholesky/host/chol_oracle.mojo and gemm_oracle, exported under the GPU
    # binding's names from the gp family's own host binding. There is no
    # optimizer on any column (DEVIATION 1761), so the fit is the kernel
    # matrix, the ridge, the factorization, the solve and three scalars.
    # Gate run 34895158657 at bafab59ef: all nine training and infer cells of
    # each lane IDENTICAL x4, the sabotage build DIVERGENT on every one.
    "gp": "the Gaussian process with an RBF kernel",
    "gp-matern12": "the Gaussian process with a Matern kernel at nu 0.5",
    "gp-matern32": "the Gaussian process with a Matern kernel at nu 1.5",
    "gp-matern52-ard": "the Gaussian process with an ARD Matern kernel at nu 2.5",
    # CPU training batch 3 (lane/cpu-training-batch3, 2026-09-14): option
    # variants of families that already had a host path, every one in the
    # 136-lane record and IDENTICAL x4 against its three GPU columns on the
    # M4's CPU column (one core) before the gate ran. Served by the host
    # entries as they stood: the k-NN squared euclidean metric and the
    # distance-weighted vote and mean (core), the transposed GEMM ops
    # (linalg), brute-force L1 DBSCAN, the five kernel and metric pairs and
    # the weighted KDE, OLS without an intercept and with weights, ridge
    # without an intercept, unpenalized logistic regression without an
    # intercept (estimators), elasticnet at the l2 end without an intercept
    # (solver), the multiplicative Holt-Winters (tsa), the linear SVC and the
    # tuned isolation forest (svm).
    "knn-sqeuclidean": "nearest neighbors under squared euclidean distance",
    "knn-clf-distance": "the distance-weighted k-NN classifier",
    "knn-reg-distance": "the distance-weighted k-NN regressor",
    "gemm-transposed": "the transposed GEMM ops",
    "dbscan-brute-l1": "brute-force DBSCAN under manhattan distance",
    "kde-tophat-sqeuclidean": "kernel density with the tophat kernel under squared euclidean distance",
    "kde-epanechnikov-l1": "kernel density with the Epanechnikov kernel under manhattan distance",
    "kde-exponential-chebyshev": "kernel density with the exponential kernel under chebyshev distance",
    "kde-linear-cosine": "kernel density with the linear kernel under cosine distance",
    "kde-cosine-minkowski": "kernel density with the cosine kernel under minkowski distance",
    "kde-weighted": "weighted kernel density",
    "ols-no-intercept": "linear regression without an intercept",
    "ols-weighted": "weighted linear regression",
    "ridge-no-intercept": "ridge without an intercept",
    "logistic-unpenalized-no-intercept": "unpenalized logistic regression without an intercept",
    "elasticnet-l2end-no-intercept": "elasticnet at the l2 end without an intercept",
    "holtwinters-multiplicative": "multiplicative Holt-Winters",
    "svc-linear": "the linear SVC",
    "iforest-tuned": "the tuned isolation forest",
    # Same batch, each needing a host restatement it did not have: the
    # cosine, manhattan, chebyshev and minkowski k-NN metrics
    # (core/knn_host_predict.mojo over metric_distance_kernel's cores), the
    # ball cover's radius and k-NN queries as an exhaustive scan (the cover
    # prunes exactly), the weighted DBSCAN core test
    # (dbscan/host/dbscan_oracle.mojo), the OWL-QN arm and the softmax loss
    # (glm/host/qn_oracle.mojo), the KPSS test (tsa/checks/kpss_oracle.mojo),
    # and epsilon-SVR (svm/host/smo_oracle.mojo's regression arm).
    "knn-manhattan": "nearest neighbors under manhattan distance",
    "knn-chebyshev": "nearest neighbors under chebyshev distance",
    "knn-cosine": "nearest neighbors under cosine distance",
    "knn-minkowski-p3": "nearest neighbors under minkowski distance at p 3",
    "knn-rbc": "nearest neighbors over the random ball cover",
    "radius": "radius neighbors",
    "radius-manhattan": "radius neighbors under manhattan distance",
    "radius-chebyshev": "radius neighbors under chebyshev distance",
    "radius-minkowski-p3": "radius neighbors under minkowski distance at p 3",
    "dbscan-weighted": "weighted DBSCAN",
    "logistic-l1": "l1-penalized logistic regression",
    "logistic-elasticnet": "elasticnet-penalized logistic regression",
    "logistic-multiclass": "multiclass logistic regression",
    "kpss": "the KPSS stationarity test",
    "svr": "SVR",
    "svr-linear": "the linear SVR",
    # The pca-full-whiten lane (lane/cpu-training-pca-whiten, 2026-09-14):
    # PCA with svd_solver='full' trains through
    # decomposition/host/pca_full_oracle.mojo, the tall TSQR Householder QR
    # and the one-sided Jacobi of svd_full.mojo restated on the host,
    # exported as pca_fit_full from the estimators host binding (a wide
    # matrix refuses by name). IDENTICAL x4 on all 27 train, infer and model
    # cells on the M4's CPU column (one core) before the gate ran, and the
    # sabotage build DIVERGENT on all 27.
    "pca-full-whiten": "whitened PCA through the full SVD",
    # The metrics-classification lane (lane/cpu-training-metrics-classification,
    # 2026-09-14): precision, recall and F1 under every average, the
    # zero-division arms, the log loss, the ROC AUC, the confusion matrix, the
    # precision-recall curve, the three regression errors, the Rand index, the
    # KL divergence and trustworthiness through
    # metrics/host/classification_oracle.mojo, exported under the GPU binding's
    # names from the metrics host binding (the log loss's probability check,
    # probability_rows_f32, from the core host binding). IDENTICAL x4 on all
    # nine train cells on the M4's CPU column (one core) before the gate ran,
    # and the sabotage build DIVERGENT on all nine.
    "metrics-classification": "the classification, ranking and regression metrics",
    # Workstream E batch 3 (2026-09-14): gradient boosting on its default
    # symmetric tree with the Logloss loss trains through
    # gbdt/host/gbdt_oracle.mojo, the device trainer restated on the host,
    # exported under the GPU binding's names from the gbdt family's own host
    # binding. The other GBDT lanes refuse by name. Gate run 34900811380 at
    # e767b829b read the four GBDT lanes' 108 train, infer and model cells
    # IDENTICAL x4 and the sabotage build DIVERGENT on all 108.
    "gbdt-symmetric": "gradient boosting on symmetric trees with the Logloss loss",
    # Workstream E batch 3 (2026-09-14): the same tree with the RMSE loss
    # trains through gbdt/host/gbdt_oracle_rmse.mojo (the seeded cursor and
    # the searcher's own leaves, DEVIATION 64) from the same binding. Gate
    # run 34900811380, as above.
    "gbdt-rmse": "gradient boosting on symmetric trees with the RMSE loss",
    # Same batch: the Depthwise and Lossguide policies with the Logloss loss
    # train through gbdt/host/gbdt_oracle_depthwise.mojo (the non-symmetric
    # driver) and gbdt/host/gbdt_oracle_lossguide.mojo, in the same binding.
    # Gate run 34900811380, as above.
    "gbdt-depthwise": "gradient boosting on depthwise trees with the Logloss loss",
    "gbdt-lossguide": "gradient boosting on lossguide trees with the Logloss loss",
    # Workstream E (lane/cpu-training-arima, 2026-09-14): batched ARIMA
    # trains and forecasts through arima/host/arima_oracle.mojo, the device
    # lane restated on the host, exported under the GPU binding's names from
    # the arima family's own host binding. par-arima is not declared. Gate
    # run 34895909493 at 422a1b9e5: all 27 training and 27 infer cells
    # IDENTICAL x4; under the sabotage build 26 of 27 of each DIVERGENT
    # (arima-011/wide keeps its hash).
    "arima": "ARIMA",
    "arima-011": "differenced ARIMA",
    "arima-seasonal-c": "seasonal ARIMA",
}

#: The lanes with NO CPU path of any kind, as the README states them. A
#: lane leaves this list the day its host lane merges; docs_facts fails the
#: README until the marked span is rewritten.
NO_CPU_PATH = (
    "UMAP",
    "the neural blocks",
    "gradient boosting training other than symmetric trees with the Logloss or RMSE loss and depthwise and lossguide trees with the Logloss loss",
)

#: The read-back trio every host binding exports under its own prefix,
#: plus the sabotage flag: `<prefix>_numeric_mode()` must answer 1,
#: `<prefix>_vendor()` "cpu", `<prefix>_column()` "cpu" (the kernel
#: matrix's CPU column, asserted at build time), `<prefix>_sabotage()` False
#: outside the gate.
READBACK = ("numeric_mode", "vendor", "column", "sabotage")

FAMILIES = (
    dict(
        family="byte_lm",
        binding="_mojolearn_byte_lm_host",
        routes=None,
        loaded_by="python/mojolearn/_byte_lm_host.py",
        sabotage_define="MOJOLEARN_BYTE_LM_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=(),
        forest_kinds=(),
        classes=("LanguageModelInference", "LanguageModelHostTrainer"),
        display="the byte LM forward pass and one training step",
        host_modules=(
            "training/byte_lm_host.mojo",
            "training/byte_lm_host_backward.mojo",
            "training/byte_lm_host_kernels.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "byte_lm_host_numeric_mode", "byte_lm_host_vendor",
            "byte_lm_host_column", "byte_lm_host_sabotage",
            "byte_lm_host_profile", "byte_lm_host_logits", "byte_lm_host_loss",
            "byte_lm_host_train_step", "all_finite_f32", "all_finite_f64",
            "cast_f64_to_f32",
        ),
        gate=".github/workflows/byte-lm-cpu-gate.yml",
        ships_in_wheel=True,
    ),
    dict(
        family="forest",
        binding="_mojolearn_forest_host",
        routes=None,
        loaded_by="python/mojolearn/_forest_host.py, python/mojolearn/_gbdt_host.py",
        sabotage_define="MOJOLEARN_FOREST_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=(),
        forest_kinds=(
            "rf_classifier", "rf_regressor", "et_classifier", "et_regressor",
            "gbdt_symmetric", "gbdt_depthwise", "gbdt_lossguide", "gbdt_rmse",
        ),
        classes=(
            "RandomForestClassifier", "RandomForestRegressor",
            "ExtraTreesClassifier", "ExtraTreesRegressor", "GradientBoosting",
        ),
        display="random forests, Extra Trees and the four gradient boosting variants",
        host_modules=("core/forest_host_predict.mojo", "core/gbdt_host_predict.mojo"),
        exports=(
            "forest_host_numeric_mode", "forest_host_vendor", "forest_host_column",
            "forest_host_sabotage", "forest_host_rf_predict_proba",
            "forest_host_rf_predict_reg", "forest_host_et_predict",
            "forest_host_gbdt_predict", "forest_host_gbdt_sigmoid",
            "all_finite_f32", "all_finite_f64", "cast_f64_to_f32",
            "argmax_rows_f32", "argmax_rows_f64", "gather_i64", "gather_f64",
        ),
        gate="tools/forest_host_gate.py (.github/workflows/forest-host-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        # The expose-tokenizer lane, 2026-09-14. Not a CPU twin of a GPU
        # entry: `tokenizer/` is host integers and tables with no GPU
        # binding, so this binding IS the family's only door, loaded by path
        # like the byte LM's and the forest's, and the same binary serves a
        # GPU box and a CPU-only install.
        family="tokenizer",
        binding="_mojolearn_tokenizer_host",
        routes=None,
        loaded_by="python/mojolearn/tokenizer.py",
        sabotage_define="MOJOLEARN_TOKENIZER_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=(),
        forest_kinds=(),
        classes=("GPT2Tokenizer",),
        display="the GPT-2 byte-level BPE tokenizer",
        host_modules=(
            "tokenizer/encoding.mojo", "tokenizer/impl/bpe.mojo",
            "tokenizer/impl/pretokenize.mojo", "tokenizer/impl/ranks.mojo",
            "tokenizer/impl/unicode_class.mojo", "tokenizer/impl/byte_unicode.mojo",
        ),
        exports=(
            "tokenizer_host_numeric_mode", "tokenizer_host_vendor",
            "tokenizer_host_column", "tokenizer_host_sabotage", "gpt2_load",
            "gpt2_n_vocab", "gpt2_max_token_bytes", "gpt2_encode", "gpt2_decode",
        ),
        gate="pixi run check-tokenizer and python/mojolearn/tests/test_tokenizer_surface.py",
        ships_in_wheel=True,
    ),
    dict(
        family="core",
        binding="_mojolearn_core_host",
        routes="_mojolearn",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(
            "knn", "knn-clf", "knn-reg", "kmeans", "kmeans-random", "kmeans-array",
            "kmeans-weighted", "knn-sqeuclidean", "knn-clf-distance", "knn-reg-distance",
            "knn-manhattan", "knn-chebyshev", "knn-cosine", "knn-minkowski-p3", "knn-rbc",
            "radius", "radius-manhattan", "radius-chebyshev", "radius-minkowski-p3",
        ),
        inference_lanes=("knn", "knn-clf", "knn-reg"),
        forest_kinds=(),
        classes=(
            "NearestNeighbors", "KNeighborsClassifier", "KNeighborsRegressor", "KMeans",
            "RadiusNeighbors",
        ),
        display="nearest neighbors, k-NN classification and k-NN regression",
        host_modules=(
            "core/knn_host_predict.mojo", "bindings/host_helpers.mojo",
            "cluster/host/kmeans_oracle.mojo",
        ),
        exports=(
            "core_host_numeric_mode", "core_host_vendor", "core_host_column",
            "core_host_sabotage", "mojolearn_vendor", "mojolearn_numeric_mode",
            "knn_search", "knn_classify", "knn_regress", "kmeans_fit",
            "radius_neighbors_count", "radius_neighbors_fill", "rbc_knn_search", "transpose_f32",
            "cast_colmajor_f64_to_f32", "nonzero_f64_count", "nonzero_f64_fill", "cast_f64_to_f32", "all_finite_f32",
            "all_finite_f64", "gather_i64", "gather_f64", "argmax_rows_f32",
            "argmax_rows_f64", "column_mean_f64", "center_columns_f32",
            "scale_rows_f32", "probability_rows_f32",
        ),
        gate="tools/classical_host_gate.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        family="linalg",
        binding="_mojolearn_linalg_host",
        routes="_mojolearn_linalg",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("gemm-pinned", "gemm-transposed"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("linalg.gemm", "linalg.gemv"),
        display="pinned GEMM",
        host_modules=("gemm/host/gemm_oracle.mojo",),
        exports=(
            "linalg_host_numeric_mode", "linalg_host_vendor", "linalg_host_column",
            "linalg_host_sabotage", "linalg_vendor", "linalg_numeric_mode",
            "linalg_profile_version", "gemm",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        family="estimators",
        binding="_mojolearn_estimators_host",
        routes="_mojolearn_estimators",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(
            "kde", "pca", "pca-whiten", "tsvd", "ols", "ridge", "dbscan", "logistic",
            "dbscan-brute-l1", "kde-tophat-sqeuclidean", "kde-epanechnikov-l1",
            "kde-exponential-chebyshev", "kde-linear-cosine", "kde-cosine-minkowski",
            "kde-weighted", "ols-no-intercept", "ols-weighted", "ridge-no-intercept",
            "logistic-unpenalized-no-intercept", "dbscan-weighted", "logistic-l1",
            "logistic-elasticnet", "logistic-multiclass", "pca-full-whiten",
        ),
        inference_lanes=("ols", "ridge", "tsvd", "logistic", "logistic-multiclass", "pca", "pca-whiten", "kde"),
        forest_kinds=(),
        classes=(
            "LinearRegression", "Ridge", "TruncatedSVD", "LogisticRegression",
            "PCA", "KernelDensity", "DBSCAN",
        ),
        display="linear regression, ridge, truncated SVD, logistic regression, PCA with and without whitening and kernel density",
        host_modules=(
            "kde/host/kde_oracle.mojo", "core/classical_host_predict.mojo",
            "decomposition/host/pca_oracle.mojo", "glm/host/glm_oracle.mojo",
            "dbscan/host/dbscan_oracle.mojo", "glm/host/qn_oracle.mojo",
            "decomposition/host/pca_full_oracle.mojo",
        ),
        exports=(
            "estimators_host_numeric_mode", "estimators_host_vendor",
            "estimators_host_column", "estimators_host_sabotage",
            "estimators_vendor", "estimators_numeric_mode", "kde_score_samples",
            "pca_fit", "pca_fit_full", "tsvd_fit", "ols_fit", "ridge_fit", "dbscan_fit", "qn_fit",
            "ols_predict", "tsvd_transform", "pca_transform",
            "pca_whiten_transform", "pca_whiten_inverse_transform",
            "qn_decision_function", "qn_sigmoid", "qn_softmax",
        ),
        gate="tools/identity_break.py and tools/classical_host_gate.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        # Workstream E batch 2 (lane/cpu-training-e2, 2026-09-14): the
        # metrics family's first host binding. It routes `_mojolearn_metrics`
        # on a CPU-only install and carries the five metrics the identity
        # harness's metrics lane computes plus the four label metrics that
        # share their integer kernels; the spectral, UMAP and remaining
        # metric entries stay absent and refuse by name.
        family="metrics",
        binding="_mojolearn_metrics_host",
        routes="_mojolearn_metrics",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("metrics", "spectral", "spectral-precomputed", "metrics-classification"),
        inference_lanes=(),
        forest_kinds=(),
        classes=(
            "SpectralClustering",
            "metrics.accuracy_score", "metrics.adjusted_rand_score",
            "metrics.entropy", "metrics.mutual_info_score",
            "metrics.homogeneity_score", "metrics.completeness_score",
            "metrics.v_measure_score", "metrics.r2_score",
            "metrics.silhouette_score", "metrics.silhouette_samples",
            "metrics.rand_score", "metrics.precision_score", "metrics.recall_score",
            "metrics.f1_score", "metrics.log_loss", "metrics.roc_auc_score",
            "metrics.confusion_matrix", "metrics.precision_recall_curve",
            "metrics.mean_squared_error", "metrics.mean_absolute_error",
            "metrics.root_mean_squared_error", "metrics.kl_divergence",
            "metrics.trustworthiness",
        ),
        display="the label, classification, ranking, regression, r2, KL, silhouette and trustworthiness metrics and spectral clustering",
        host_modules=(
            "metrics/host/metrics_oracle.mojo",
            "metrics/host/classification_oracle.mojo",
            "spectral/host/spectral_oracle.mojo",
            "cluster/host/kmeans_oracle.mojo",
            "core/knn_host_predict.mojo",
        ),
        exports=(
            "metrics_host_numeric_mode", "metrics_host_vendor",
            "metrics_host_column", "metrics_host_sabotage", "metrics_vendor",
            "metrics_numeric_mode", "accuracy_score", "adjusted_rand_score",
            "entropy", "mutual_info_score", "homogeneity_score",
            "completeness_score", "v_measure_score", "r2_score", "silhouette",
            "spectral_fit_predict_dataset", "spectral_fit_predict_graph",
            "rand_score", "mean_squared_error", "mean_absolute_error",
            "root_mean_squared_error", "roc_auc_score", "precision_recall_curve",
            "log_loss", "confusion_matrix", "precision_recall_fscore",
            "kl_divergence", "trustworthiness",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        # Workstream E batch 2 (lane/cpu-training-e2, 2026-09-14): the
        # preprocessing family's host binding, the whole GPU binding's
        # surface (four entries) restated.
        family="preprocessing",
        binding="_mojolearn_preprocessing_host",
        routes="_mojolearn_preprocessing",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(
            "standard-scaler", "minmax-scaler", "standard-scaler-no-mean",
            "standard-scaler-no-std", "minmax-scaler-clip",
        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=("StandardScaler", "MinMaxScaler"),
        display="the standard and min-max scalers",
        host_modules=("preprocessing/host/scaler_oracle.mojo",),
        exports=(
            "preprocessing_host_numeric_mode", "preprocessing_host_vendor",
            "preprocessing_host_column", "preprocessing_host_sabotage",
            "preprocessing_numeric_mode", "preprocessing_vendor",
            "standard_fit", "standard_transform", "minmax_fit", "minmax_transform",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        family="tsa",
        binding="_mojolearn_tsa_host",
        routes="_mojolearn_tsa",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("holtwinters", "holtwinters-multiplicative", "kpss"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("ExponentialSmoothing", "kpss_test"),
        display="Holt-Winters",
        host_modules=("holtwinters/host/hw_oracle.mojo", "tsa/checks/kpss_oracle.mojo"),
        exports=(
            "tsa_host_numeric_mode", "tsa_host_vendor", "tsa_host_column",
            "tsa_host_sabotage", "tsa_vendor", "holtwinters_fit",
            "holtwinters_forecast", "kpss_test",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        family="solver",
        binding="_mojolearn_solver_host",
        routes="_mojolearn_solver",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("lasso", "elasticnet", "agglomerative", "elasticnet-l2end-no-intercept"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("Lasso", "ElasticNet", "AgglomerativeClustering"),
        display="lasso, elasticnet and agglomerative clustering",
        host_modules=(
            "solver/host/cd_oracle.mojo", "gemm/host/gemm_oracle.mojo",
            "hierarchy/checks/linkage_oracle.mojo",
        ),
        exports=(
            "solver_host_numeric_mode", "solver_host_vendor", "solver_host_column",
            "solver_host_sabotage", "solver_vendor", "cd_fit", "cd_predict",
            "linkage_fit",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        family="svm",
        binding="_mojolearn_svm_host",
        routes="_mojolearn_svm",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("svc", "iforest", "svc-linear", "iforest-tuned", "svr", "svr-linear"),
        inference_lanes=("svc",),
        forest_kinds=(),
        classes=("SVC", "IsolationForest", "SVR"),
        display="SVC",
        host_modules=(
            "svm/host/smo_oracle.mojo", "gemm/host/gemm_oracle.mojo",
            "isolation_forest/checks/if_oracle.mojo",
            "isolation_forest/impl/rng/xorwow.mojo",
        ),
        exports=(
            "svm_host_numeric_mode", "svm_host_vendor", "svm_host_column",
            "svm_host_sabotage", "svm_vendor", "svm_numeric_mode", "svc_fit",
            "svc_predict", "svr_fit", "svr_predict", "iforest_run",
        ),
        gate="tools/identity_break.py and tools/classical_host_gate.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        family="trees",
        binding="_mojolearn_trees_host",
        routes="_mojolearn_trees",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("et-clf", "et-reg"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("ExtraTreesClassifier", "ExtraTreesRegressor"),
        display="the Extra Trees classifier and regressor",
        host_modules=(
            "extratrees/estimator.mojo", "extratrees/checks/pcg_rng.mojo",
            "core/forest_host_predict.mojo",
        ),
        exports=(
            "trees_host_numeric_mode", "trees_host_vendor", "trees_host_column",
            "trees_host_sabotage", "trees_vendor", "trees_numeric_mode",
            "et_classifier_fit", "et_classifier_fit_export",
            "et_classifier_fit_rowmajor", "et_classifier_fit_rowmajor_export",
            "et_regressor_fit", "et_regressor_fit_export",
            "et_regressor_fit_rowmajor", "et_regressor_fit_rowmajor_export",
            "forest_export", "forest_export_legacy", "forest_export_release",
            "et_predict",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        # Workstream E batch 3 (2026-09-14): the RandomForest family's host
        # binding. It routes `_mojolearn_rf` on a CPU-only install with the
        # GPU binding's fit, export and predict names; the class-weighted and
        # shard fits and the GPU engines stay absent and refuse by name. Not
        # the forest host binding: that one is loaded by path under its own
        # names, and `_backend` must not route `_mojolearn_rf` to it.
        family="rf",
        binding="_mojolearn_rf_host",
        routes="_mojolearn_rf",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("rf-clf", "rf-reg"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("RandomForestClassifier", "RandomForestRegressor"),
        display="the random forest classifier and regressor",
        host_modules=("ensemble/host/rf_oracle.mojo", "core/forest_host_predict.mojo"),
        exports=(
            "rf_host_numeric_mode", "rf_host_vendor", "rf_host_column",
            "rf_host_sabotage", "rf_vendor", "rf_numeric_mode",
            "rf_classifier_fit", "rf_classifier_fit_export",
            "rf_classifier_fit_rowmajor", "rf_classifier_fit_rowmajor_export",
            "rf_regressor_fit", "rf_regressor_fit_export",
            "rf_regressor_fit_rowmajor", "rf_regressor_fit_rowmajor_export",
            "forest_export", "forest_export_legacy", "forest_export_release",
            "rf_predict_proba", "rf_predict_reg",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        # Workstream E, the gp host lane (2026-09-14): the Gaussian process
        # family's host binding. It routes `_mojolearn_gp` on a CPU-only
        # install with the GPU binding's gpr_fit and gpr_predict and the
        # Cholesky door workstream D put on the same binding
        # (cholesky_factor, cholesky_solve, cholesky_profile_jitter; the
        # cholesky lane is not declared covered because the 136-lane record
        # predates it). gp_parallel_available stays absent, so the ordered
        # multi-GPU driver refuses by name.
        family="gp",
        binding="_mojolearn_gp_host",
        routes="_mojolearn_gp",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("gp", "gp-matern12", "gp-matern32", "gp-matern52-ard"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("GaussianProcessRegressor", "Cholesky"),
        display="the Gaussian process regressor and the Cholesky door",
        host_modules=(
            "gaussian_process/host/gpr_oracle.mojo",
            "cholesky/host/chol_oracle.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "gp_host_numeric_mode", "gp_host_vendor", "gp_host_column",
            "gp_host_sabotage", "gp_vendor", "gp_numeric_mode",
            "gpr_fit", "gpr_predict", "cholesky_profile_jitter",
            "cholesky_factor", "cholesky_solve",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        # Workstream E batch 3 (2026-09-14): the GradientBoosting family's
        # host binding. It routes `_mojolearn_gbdt` on a CPU-only install
        # with the GPU binding's fit, predict, model-dim and sigmoid names;
        # gbdt_fit refuses by name every value outside the gbdt-symmetric,
        # gbdt-rmse, gbdt-depthwise and gbdt-lossguide configurations, and the multi-dimensional predict, the ordered and
        # FeatureFreq fits and the adapters' device transforms stay absent.
        # Not the forest host binding: that one is loaded by path under its
        # own names and takes the model as flat arrays.
        family="gbdt",
        binding="_mojolearn_gbdt_host",
        routes="_mojolearn_gbdt",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("gbdt-symmetric", "gbdt-rmse", "gbdt-depthwise", "gbdt-lossguide"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("GradientBoosting",),
        display="gradient boosting on symmetric trees with the Logloss or RMSE loss and depthwise and lossguide trees with the Logloss loss",
        host_modules=(
            "gbdt/host/gbdt_oracle.mojo", "gbdt/host/gbdt_oracle_rmse.mojo",
            "gbdt/host/gbdt_oracle_depthwise.mojo", "gbdt/host/gbdt_oracle_lossguide.mojo",
            "core/gbdt_host_predict.mojo",
        ),
        exports=(
            "gbdt_host_numeric_mode", "gbdt_host_vendor", "gbdt_host_column",
            "gbdt_host_sabotage", "gbdt_vendor", "gbdt_numeric_mode",
            "gbdt_fit", "gbdt_predict", "gbdt_model_dim", "gbdt_sigmoid",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        # Workstream E (lane/cpu-training-arima, 2026-09-14): batched
        # ARIMA's host binding. It routes `_mojolearn_arima` on a CPU-only
        # install with the GPU binding's whole surface (fit, predict,
        # forecast); p, q or P above 1, any Q, d + D of 2, p + q + k of 0
        # and an in-sample prediction refuse by name. par-arima is not
        # declared.
        family="arima",
        binding="_mojolearn_arima_host",
        routes="_mojolearn_arima",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("arima", "arima-011", "arima-seasonal-c"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("ARIMA",),
        display="batched ARIMA",
        host_modules=("arima/host/arima_oracle.mojo",),
        exports=(
            "arima_host_numeric_mode", "arima_host_vendor", "arima_host_column",
            "arima_host_sabotage", "arima_vendor", "arima_numeric_mode",
            "arima_fit", "arima_predict", "arima_forecast",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
)


def families():
    """Every host family name, in build order."""
    return [f["family"] for f in FAMILIES]


def family(name):
    for f in FAMILIES:
        if f["family"] == name:
            return f
    raise KeyError(f"no host family named {name!r}; the manifest lists {families()}")


def bindings():
    """Every host binding basename, in the same order."""
    return [f["binding"] for f in FAMILIES]


def wheel_families():
    """The families whose host binding ships in both wheels, in build
    order: what packaging/linux/build_sets.sh and
    packaging/macos/build_release_wheel.sh build through the shims, and what
    packaging/linux/pack_wheel.py requires under mojolearn/host/."""
    return [f["family"] for f in FAMILIES if f["ships_in_wheel"]]


def wheel_bindings():
    """The basenames of `wheel_families()`, in the same order."""
    return [f["binding"] for f in FAMILIES if f["ships_in_wheel"]]


def training_gpu_column_record():
    """The one record directory the training GPU columns live in, by its
    last path component (`2026-09-14_47-lanes`). The wheels carry the three
    columns under mojolearn/identity_columns/<record>/ so `python -m
    mojolearn identity` can name the record it diffed against; a manifest
    naming columns from two records is refused here, because one diff has
    one record."""
    dirs = sorted({c.rsplit("/", 1)[0] for c in TRAINING_GPU_COLUMNS})
    if len(dirs) != 1:
        raise ValueError(f"TRAINING_GPU_COLUMNS spans {len(dirs)} record directories: {dirs}")
    return dirs[0].rsplit("/", 1)[1]


def binding_source(name):
    """The Mojo source of a family's binding."""
    return f"bindings/_mojolearn_{name}_host.mojo"


def build_shim(name):
    """The two-line shim that execs BUILDER for a family."""
    return f"bindings/build_{name}_host.sh"


def routed_modules():
    """`_MODULES` name -> host binding basename: the table `_backend` routes a
    CPU-only install through. The three bindings loaded by path (byte_lm,
    forest, tokenizer) are deliberately absent."""
    return {f["routes"]: f["binding"] for f in FAMILIES if f["routes"]}


def routed_families():
    """The families with a route, the phase 1 set the gate builds in a loop."""
    return [f["family"] for f in FAMILIES if f["routes"]]


def routed_bindings():
    return [f["binding"] for f in FAMILIES if f["routes"]]


def covered_lanes():
    """The identity_break lanes with a CPU TRAINING path, in the gate's
    order (TRAINING_LANE_NAMES' order, the order the lanes landed in). Every
    lane a family declares must have a name there, and the reverse."""
    declared = []
    for f in FAMILIES:
        for lane in f["training_lanes"]:
            if lane not in declared:
                declared.append(lane)
    named = list(TRAINING_LANE_NAMES)
    if sorted(named) != sorted(declared):
        raise RuntimeError(
            f"host_surface: TRAINING_LANE_NAMES {named} and the families' training lanes "
            f"{declared} disagree"
        )
    return named


def inference_lanes():
    """The classical gate lanes served from a saved model, in gate order."""
    out = []
    for f in FAMILIES:
        for lane in f["inference_lanes"]:
            if lane not in out:
                out.append(lane)
    return out


def forest_kinds():
    return list(family("forest")["forest_kinds"])


def sabotage_define(name):
    return family(name)["sabotage_define"]


def training_sentence():
    """The training list as the README states it."""
    return _join([TRAINING_LANE_NAMES[lane] for lane in covered_lanes()])


def inference_sentence():
    """The inference list as the README states it, one clause per family
    that serves a saved model."""
    parts = [f["display"] for f in FAMILIES if f["inference_lanes"] or f["forest_kinds"]]
    return "; ".join(parts)


def no_cpu_path_sentence():
    return _join(list(NO_CPU_PATH))


def _join(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def markdown_table():
    """The CPU surface as one table, for the marked spans in
    SUPPORT_MATRIX.md and docs/BYTE_LM_CPU_TRAINING.md."""
    rows = [
        "| family | binding under `mojolearn/host/` | routes (CPU-only install) | trains on a CPU (identity_break lanes) | predicts on a CPU from a saved model | gate | in a wheel |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in FAMILIES:
        trains = ", ".join(f["training_lanes"]) or "no"
        if f["forest_kinds"]:
            predicts = ", ".join(f["classes"]) + " (" + ", ".join(f["forest_kinds"]) + ")"
        elif f["inference_lanes"]:
            predicts = ", ".join(f["classes"]) + " (" + ", ".join(f["inference_lanes"]) + ")"
        elif f["routes"] is None and f["classes"]:
            # A surface of its own, loaded by path (the byte LM's two
            # classes; the tokenizer's encode and decode, which fit no
            # "saved model" wording but are what the binding serves).
            predicts = ", ".join(f["classes"])
        else:
            predicts = "no"
        rows.append(
            f"| {f['family']} | `{f['binding']}.so` | {('`' + f['routes'] + '`') if f['routes'] else 'loaded by path'} "
            f"| {trains} | {predicts} | {f['gate']} | {'yes' if f['ships_in_wheel'] else 'no, `' + build_shim(f['family']) + '`'} |"
        )
    return "\n".join(rows)


def as_dict():
    return dict(
        source=SOURCE,
        builder=BUILDER,
        families=[dict(f) for f in FAMILIES],
        routed=routed_modules(),
        covered_lanes=covered_lanes(),
        inference_lanes=inference_lanes(),
        forest_kinds=forest_kinds(),
        classical_recorded=list(CLASSICAL_RECORDED),
        classical_gpu_columns=list(CLASSICAL_GPU_COLUMNS),
        training_gpu_columns=list(TRAINING_GPU_COLUMNS),
        training_gpu_column_record=training_gpu_column_record(),
        wheel_families=wheel_families(),
        wheel_bindings=wheel_bindings(),
        forest_recorded_root=FOREST_RECORDED_ROOT,
        no_cpu_path=list(NO_CPU_PATH),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="the CPU surface manifest; one flag per list")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--families", action="store_true", help="every host family")
    g.add_argument("--routed-families", action="store_true", help="families routed by _backend._HOST_MODULES")
    g.add_argument("--bindings", action="store_true", help="every host binding basename")
    g.add_argument("--routed-bindings", action="store_true", help="the routed families' basenames")
    g.add_argument("--covered-lanes", action="store_true", help="identity_break lanes with a CPU training path (comma separated)")
    g.add_argument("--inference-lanes", action="store_true", help="classical gate lanes served from a saved model (comma separated)")
    g.add_argument("--forest-kinds", action="store_true", help="forest gate kinds (comma separated)")
    g.add_argument("--wheel-families", action="store_true", help="families whose host binding ships in the wheels")
    g.add_argument("--wheel-bindings", action="store_true", help="the wheel families' basenames")
    g.add_argument("--training-gpu-column-record", action="store_true", help="the record directory name of the training GPU columns")
    g.add_argument("--classical-recorded", action="store_true", help="classical gate recording directories")
    g.add_argument("--classical-gpu-columns", action="store_true", help="the GPU columns the classical gate compares against")
    g.add_argument("--training-gpu-columns", action="store_true", help="the GPU columns the training gate diffs against")
    g.add_argument("--markdown", action="store_true", help="the surface as a Markdown table")
    g.add_argument("--json", action="store_true", help="the whole manifest as JSON")
    p.add_argument("--sep", default=None, help="separator for list output (default: comma for lanes and kinds, space otherwise)")
    args = p.parse_args(argv)
    if args.json:
        print(json.dumps(as_dict(), indent=2, sort_keys=True))
        return 0
    if args.markdown:
        print(markdown_table())
        return 0
    comma = (args.covered_lanes or args.inference_lanes or args.forest_kinds)
    sep = args.sep if args.sep is not None else ("," if comma else " ")
    if args.families:
        items = families()
    elif args.routed_families:
        items = routed_families()
    elif args.bindings:
        items = bindings()
    elif args.routed_bindings:
        items = routed_bindings()
    elif args.wheel_families:
        items = wheel_families()
    elif args.wheel_bindings:
        items = wheel_bindings()
    elif args.training_gpu_column_record:
        items = [training_gpu_column_record()]
    elif args.covered_lanes:
        items = covered_lanes()
    elif args.inference_lanes:
        items = inference_lanes()
    elif args.forest_kinds:
        items = forest_kinds()
    elif args.classical_recorded:
        items = list(CLASSICAL_RECORDED)
    elif args.classical_gpu_columns:
        items = list(CLASSICAL_GPU_COLUMNS)
    else:
        items = list(TRAINING_GPU_COLUMNS)
    print(sep.join(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
