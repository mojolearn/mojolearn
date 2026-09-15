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
whether it ships in a wheel (inference dependencies and the published byte-LM
trainer ship; training-only families remain source reference bindings; the two wheel builders and the packer read
`--wheel-families` and `--wheel-bindings` below instead of naming the byte
LM's binding by hand, and packaging/check_ext_lists.py fails when any of
them carries a host list of its own).

This file imports nothing from the package on purpose. It runs by path
before the package can import (the gate runner has no binding built yet):

    python3 python/mojolearn/host_surface.py --covered-lanes
    python3 python/mojolearn/host_surface.py --record-covered-lanes
    python3 python/mojolearn/host_surface.py --fix-covered-lanes
    python3 python/mojolearn/host_surface.py --training-fix-columns
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
#: hold against it. Since 2026-09-14 night the 166-lane record at 1eea14f80
#: (the batch part, fifteen more par-* lanes, every lane complete on all three
#: columns, the AMD column MI325X again) is the one; it carries one DIVERGENT
#: training cell, kmeans-sqrt/wide (the H100 inertia stands alone,
#: docs/lanes/BRIEF_kmeans_sqrt_wide_h100_inertia_2026-09-14.md), on a lane no
#: CPU column covers, and the workflow asserts that count exactly. That cell is
#: fixed since 9fde8f5f7 (DEVIATIONS 2715 and 2716); the kmeans lanes' cells
#: after the fix are bench/results/identity_break/2026-09-14_kmeans-sqrt-fix,
#: which the workflow asserts IDENTICAL x3 in its own step.
TRAINING_GPU_COLUMNS = (
    "bench/results/identity_break/2026-09-14_166-lanes/apple-m4.json",
    "bench/results/identity_break/2026-09-14_166-lanes/nvidia-h100-sm_90a.json",
    "bench/results/identity_break/2026-09-14_166-lanes/amd-mi325x-gfx942.json",
)

#: The covered lanes whose cells in TRAINING_GPU_COLUMNS predate a fix of
#: the lane on every GPU column, and the three columns of the same boxes
#: taken after the fix, which the training gate diffs those lanes against
#: instead (--require-columns 4, the same verdict rule). kmeans-sqrt is the
#: one (lane/cpu-training-misc, 2026-09-15): the 166-lane record carries its
#: pre-fix labels on all three columns and the H100's pre-fix inertia on
#: `wide` (DEVIATIONS 2716 and 2715), the fix record at 9fde8f5f7 carries the
#: fixed cells IDENTICAL x3, and the host oracle was fixed in the same
#: commit, so the CPU column reads IDENTICAL x4 against the fix record and
#: DIVERGENT on all nine fixtures against the 166-lane record. A lane leaves
#: this list the day TRAINING_GPU_COLUMNS names a record taken after its
#: fix. `python -m mojolearn identity` diffs against TRAINING_GPU_COLUMNS
#: alone, so on a CPU-only install it runs `record_covered_lanes()`.
#:
#: The same list also carries covered lanes the record does not have at all
#: (lane/cpu-training-embedding-ivf, 2026-09-15): embedding and
#: embedding-sort, whose three GPU columns are
#: bench/results/identity_break/2026-09-15_embedding-sort (Apple M4, H100,
#: MI325X at ba4a108bb and e2d770ba8), and ivf and ivf-euclidean, whose three
#: are bench/results/identity_break/2026-09-14_ivf-euclidean (Apple M4, H100,
#: MI300X at 76a170dcf, after the L2SqrtExpanded fix). `identity_break --diff`
#: takes every JSON here at once: a column that lacks a lane reads "(not
#: run)" on its cells and is not counted, so each of these lanes rests on
#: its own record's three GPU hashes plus the CPU column's, which
#: `--require-columns 4` demands, and a JSON that carried a lane it should
#: not would add a fifth hash to the cell rather than hide one. They leave
#: this list the day TRAINING_GPU_COLUMNS names a record that carries them
#: (lane/identity-record-next's 178-lane record does).
TRAINING_FIX_COLUMNS = (
    "bench/results/identity_break/2026-09-14_kmeans-sqrt-fix/apple-m4.json",
    "bench/results/identity_break/2026-09-14_kmeans-sqrt-fix/nvidia-h100-sm_90a.json",
    "bench/results/identity_break/2026-09-14_kmeans-sqrt-fix/amd-mi325x-gfx942.json",
    "bench/results/identity_break/2026-09-15_embedding-sort/identity_break.apple-m4.json",
    "bench/results/identity_break/2026-09-15_embedding-sort/identity_break.nvidia-nvidia-h100-80gb-hbm3-sm_90a.json",
    "bench/results/identity_break/2026-09-15_embedding-sort/identity_break.amd-gfx942.json",
    "bench/results/identity_break/2026-09-14_ivf-euclidean/identity_break.apple-m4.json",
    "bench/results/identity_break/2026-09-14_ivf-euclidean/identity_break.nvidia-nvidia-h100-80gb-hbm3-sm_90a.json",
    "bench/results/identity_break/2026-09-14_ivf-euclidean/identity_break.amd-mi300x-gfx942.json",
)
TRAINING_FIX_LANES = ("kmeans-sqrt", "embedding", "embedding-sort", "ivf", "ivf-euclidean")

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
    "bench/results/classical_host/2026-09-15-apple-m4-neighbors-density",
    # lane/inference-linear-svm (2026-09-15): the 17 saved-model lanes that
    # joined the estimators family's inference lanes, recorded on the M4's
    # Metal set on three fixtures. The NVIDIA and AMD recordings of these
    # lanes are owed to the next release record; their infer cells in the
    # 166-lane record are the cross-vendor comparison until then.
    "bench/results/classical_host/2026-09-15-apple-m4-linear-kernel",
    # lane/inference-forecast-umap-pca (2026-09-15): pca-full-whiten and umap,
    # recorded on the M4's Metal set; both bind families the gate builds.
    "bench/results/classical_host/2026-09-15-apple-m4-umap-pca",
    # lane/inference-svm (2026-09-15): svc-linear, svc-poly, svr and
    # svr-linear, recorded on the M4's Metal set on all nine fixtures. The
    # NVIDIA and AMD recordings are owed to the next release record.
    "bench/results/classical_host/2026-09-15-apple-m4-svm",
)

#: The saved ARIMA recordings (lane/inference-forecast-umap-pca, 2026-09-15),
#: checked with `tools/classical_host_gate.py check` like CLASSICAL_RECORDED
#: but kept apart from it: they bind `_mojolearn_forecast_host`, a family
#: with no route. The CPU identity gate workflow builds every family the
#: manifest declares and checks this list, with SEARCH_LOOKUP_RECORDED and
#: INFERENCE_ONLY_RECORDED, as `saved_model_recorded()` (production must
#: match, the sabotage host set must differ on every lane).
FORECAST_RECORDED = (
    "bench/results/classical_host/2026-09-15-apple-m4-arima",
)

#: The iforest, GMM and HDBSCAN saved-model recordings (the neighbors and
#: density inference lane, 2026-09-15), checked with
#: `tools/classical_host_gate.py check` like CLASSICAL_RECORDED but kept apart
#: from it, as FORECAST_RECORDED is: the GMM and HDBSCAN models bind the
#: unrouted mixture_infer and hdbscan_infer families (the gp recordings bind
#: gp_infer). The CPU identity gate workflow checks it through
#: `saved_model_recorded()`.
INFERENCE_ONLY_RECORDED = (
    "bench/results/classical_host/2026-09-15-apple-m4-iforest-gmm-hdbscan",
    "bench/results/classical_host/2026-09-15-apple-m4-gp-gmm-sample",
)

#: The saved IVF-Flat index and embedding table recordings
#: (lane/inference-embedding-ivf-cholesky, 2026-09-15), checked with
#: `tools/classical_host_gate.py check` like FORECAST_RECORDED and kept apart
#: from CLASSICAL_RECORDED for the same reason: they bind
#: `_mojolearn_ivf_search_host` and `_mojolearn_embedding_infer_host`,
#: families with no route. The CPU identity gate workflow checks it through
#: `saved_model_recorded()`.
SEARCH_LOOKUP_RECORDED = (
    "bench/results/classical_host/2026-09-15-apple-m4-ivf-embedding",
)

def saved_model_recorded():
    """The saved-model recordings that bind the unrouted inference-only
    families (forecast, mixture_infer, gp_infer, hdbscan_infer, ivf_search,
    embedding_infer), checked by the CPU identity gate apart from
    CLASSICAL_RECORDED."""
    return list(FORECAST_RECORDED + INFERENCE_ONLY_RECORDED + SEARCH_LOOKUP_RECORDED)


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
    # lane/cpu-training-small-gaps (2026-09-15): normalize_y=True, the folds
    # through the preprocessing host binding's standard_fit and standard_transform.
    "gp-normalize-y": "the Gaussian process with normalized targets",
    # Gaussian process classification (lane/gaussian-process-classifier,
    # 2026-09-15): gaussian_process/host/gpc_oracle.mojo over the gp oracle's
    # kernel matrix, the Cholesky oracle and gemm_oracle, the Newton steps in
    # gaussian_process/host/gpc_steps.mojo (the GPU path compiles the same
    # file). No GPU record carries these lanes yet, so their cells are OWED.
    "gpc": "the binary Gaussian process classifier",
    "gpc-multiclass": "the one-vs-rest Gaussian process classifier",
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
    # lane/cpu-training-small-gaps (2026-09-15): SVC(kernel='poly') through the
    # svm host binding's svc_fit and svc_predict (degree and coef0 in params).
    "svc-poly": "the polynomial SVC",
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
    # The metrics-fowlkes-mallows lane (lane/cpu-training-small-gaps,
    # 2026-09-15): scikit-learn's fowlkes_mallows_score over the integer
    # contingency matrix, host_fowlkes_mallows in
    # metrics/host/metrics_oracle.mojo, exported under the GPU binding's name.
    "metrics-fowlkes-mallows": "the Fowlkes-Mallows index",
    # The weighted score lanes (lane/cpu-training-small-gaps, 2026-09-15):
    # score(X, y, sample_weight) of the gradient boosting adapters and the
    # random forests through host_weighted_accuracy and host_weighted_r2 in
    # metrics/host/metrics_oracle.mojo, exported from the metrics host binding.
    "gbdt-adapter-score-weighted": "the weighted scores of the gradient boosting classifier and regressor",
    "rf-score-weighted": "the weighted scores of the random forest classifier and regressor",
    # The mlp lane (lane/cpu-training-mlp, 2026-09-14): SmallMLPTrainer's
    # step through the training family's host binding (the three MLP
    # operations in training/host/mlp_oracle.mojo, the loss and AdamW over
    # training/checks/loss_oracle.mojo and optimizer_oracle.mojo) and the
    # linalg host GEMM. IDENTICAL x4 on all 27 train, infer and model cells on
    # the M4's CPU column (one core) before the gate ran.
    "mlp": "the small MLP",
    # Workstream E batch 3 (2026-09-14): gradient boosting on its default
    # symmetric tree with the Logloss loss trains through
    # gbdt/host/gbdt_oracle.mojo, the device trainer restated on the host,
    # exported under the GPU binding's names from the gbdt family's own host
    # binding. The other GBDT lanes refused by name until 2026-09-15. Gate run 34900811380 at
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
    # CPU training for more GBDT lanes (lane/cpu-training-gbdt-losses,
    # 2026-09-15), from the same binding. nan_mode Min and Max on an X that
    # carries NaN: the oracle's grid already placed the NaN border and
    # substituted the value, and the binding refused the NaN by name only.
    # The two sklearn-style adapters: the regressor is RMSE at the adapter's
    # defaults and needed no change; the classifier is Logloss plus the
    # adapter's binary probability and class transforms, restated per
    # element in the binding (gbdt/binary_prediction.mojo's kernel). On the
    # M4's CPU column (one core) all 27 train, 36 infer and model and 27
    # batch cells read IDENTICAL x4 against the 166-lane record, and the
    # sabotage build DIVERGENT on every one of them.
    "gbdt-nan-modes": "gradient boosting with the Min and Max NaN modes",
    "gbdt-adapter-clf": "the gradient boosting classifier",
    "gbdt-adapter-reg": "the gradient boosting regressor",
    # Same lane, the rest of it: the ten pointwise losses of
    # gbdt-parametric-losses and the Exact leaves with the Poisson bootstrap
    # of gbdt-exact-mae train through gbdt/host/gbdt_oracle_losses.mojo (the
    # pointwise target kernels, the row bootstraps, the Gradient and Newton
    # walkers and the Exact weighted quantile); gbdt-lossguide-newtoncosine
    # through gbdt/host/gbdt_oracle_depthwise.mojo with the NewtonCosine
    # score, the child-Hessian and split-gain thresholds, the leaf size, the
    # feature sample, the score noise, the Bernoulli bootstrap and Gradient
    # leaves; gbdt-multiclass and gbdt-onevsall through
    # gbdt/host/gbdt_oracle_multiclass.mojo (the multilogit and one-vs-all
    # planes at stat_count 1 + dim, class weights, the blocked Newton step)
    # and the binding's gbdt_predict_multi. On the M4's CPU column (one core)
    # the twelve GBDT lanes read all 108 train, 198 infer and model and 108
    # batch cells IDENTICAL x4 against the 166-lane record, and the sabotage
    # build DIVERGENT on every one of them.
    "gbdt-parametric-losses": "gradient boosting with the Quantile, MAE, LogLinQuantile, MAPE, Poisson, Lq, Expectile, Tweedie, Huber and CrossEntropy losses",
    "gbdt-exact-mae": "gradient boosting with Exact leaves and the Poisson bootstrap",
    "gbdt-lossguide-newtoncosine": "gradient boosting on lossguide trees with the NewtonCosine score and the searcher options",
    "gbdt-multiclass": "multiclass gradient boosting",
    "gbdt-onevsall": "one-vs-all gradient boosting",
    # lane/cpu-training-gbdt-ordered (2026-09-15): OrderedRMSE trains
    # through gbdt/host/gbdt_oracle_ordered.mojo (the pointwise searcher's
    # fold arm, its 8-bit fixed-point and half-byte float histograms, the
    # dynamic cosine scorer and the ordered Newton leaves restated on the
    # host) and ExperimentalTwoLevelFeatureFreq through
    # gbdt/host/gbdt_oracle_feature_freq.mojo (the synchronized two-level
    # tensor search over the symmetric oracle's histograms), exported as
    # gbdt_fit_ordered_rmse and gbdt_fit_two_level_feature_freq from the
    # gbdt host binding; sample_weight refuses by name on both. On the M4's
    # CPU column (one core) both lanes read all 18 train, 36 infer and model
    # and 18 batch cells IDENTICAL x4 against the 166-lane columns before
    # the gate ran, and the sabotage build DIVERGENT on every cell.
    "gbdt-ordered-rmse": "ordered boosting with the RMSE loss (OrderedRMSE)",
    "gbdt-feature-freq": "the two-level FeatureFreq estimator",
    # The same lane branch: the pointwise searcher with L2 scores, the
    # Bayesian bootstrap, boost from average on Logloss, row weights and an
    # eval set with the Iter detector and best-model truncation trains
    # through gbdt/host/gbdt_oracle_pointwise.mojo (the ordered oracle's
    # single-task structure search with the plain L2 scorer, the weighted
    # Newton walker, the bootstrap draws and the test arm restated on the
    # host) inside gbdt_fit's use_pointwise_searcher arm, which refuses every
    # other value of those options by name. IDENTICAL x4 on all 9 train, 18
    # infer and model and 9 batch cells on the M4's CPU column (one core)
    # before the gate ran, and the sabotage build DIVERGENT on every cell.
    "gbdt-pointwise-l2-bayesian-eval": "gradient boosting with the pointwise searcher, L2 scores, the Bayesian bootstrap and an eval set",
    # The same lane branch: gradient boosting with categorical and one-hot
    # columns trains through gbdt/host/gbdt_oracle_onehot.mojo (the flags,
    # train's categorical validation, the one-hot grid inside
    # gbdt_oracle.mojo's fit, the cat model records). The lane's categorical
    # column has two categories, so train makes it one-hot and no CTR column
    # is built on any fixture (the model texts carry no ctr record); a
    # categorical column above one_hot_max_size refuses by name. IDENTICAL
    # x4 on all 9 train, 18 infer and model and 9 batch cells on the M4's
    # CPU column (one core) before the gate ran, the sabotage build
    # DIVERGENT on every cell.
    "gbdt-categorical-ctr": "gradient boosting with one-hot categorical columns",
    # lane/gbdt-learning-to-rank stage 2 (2026-09-15): the QueryRMSE ranking
    # loss on query groups trains through gbdt/host/gbdt_oracle_losses.mojo
    # with the querywise target restated in gbdt/host/gbdt_oracle_query.mojo,
    # from the same binding.
    "gbdt-query-rmse": "gradient boosting with the QueryRMSE ranking loss on query groups",
    # lane/gbdt-learning-to-rank stage 3 (2026-09-15): the PairLogit ranking
    # loss on generated and explicit pairs trains through
    # gbdt/host/gbdt_oracle_losses.mojo with the pairwise target restated in
    # gbdt/host/gbdt_oracle_pair.mojo and the pairs of gbdt/data/pairs.mojo,
    # from the same binding.
    "gbdt-pair-logit": "gradient boosting with the PairLogit ranking loss on generated and explicit pairs",
    # lane/gbdt-learning-to-rank stage 4 (2026-09-15): the YetiRank ranking
    # loss trains through gbdt/host/gbdt_oracle_losses.mojo with the sampled
    # permutations restated in gbdt/host/gbdt_oracle_yeti.mojo over the task
    # table of gbdt/data/yeti_rank_tasks.mojo, from the same binding.
    "gbdt-yeti-rank": "gradient boosting with the YetiRank ranking loss on query groups",
    # Workstream E (lane/cpu-training-arima, 2026-09-14): batched ARIMA
    # trains and forecasts through arima/host/arima_oracle.mojo, the device
    # lane restated on the host, exported under the GPU binding's names from
    # the arima family's own host binding. par-arima was not declared until
    # lane/cpu-training-par-classical (2026-09-15, below). Gate
    # run 34895909493 at 422a1b9e5: all 27 training and 27 infer cells
    # IDENTICAL x4; under the sabotage build 26 of 27 of each DIVERGENT
    # (arima-011/wide keeps its hash).
    "arima": "ARIMA",
    "arima-011": "differenced ARIMA",
    "arima-seasonal-c": "seasonal ARIMA",
    # The umap host lane (lane/cpu-training-umap-b, 2026-09-14): UMAP fits
    # and transforms through umap/host/umap_oracle.mojo, exported under the
    # GPU binding's names from the metrics host binding. The fit's optimizer
    # is the IDENTICAL DEVICE epoch fold (kernel-matrix row
    # umap_device_optimizer_for) restated vertex by vertex, not the serial
    # host loop, which produces different bits. Gate run 34914545371 at
    # 5988700d9 (136-lane record): all nine train and nine infer cells
    # IDENTICAL x4 on the seven runners, the sabotage build DIVERGENT on all
    # eighteen; the 166-lane record carries the same umap hashes.
    "umap": "UMAP",
    # lane/cpu-training-misc batch 1 (2026-09-15): the two k-means option
    # lanes the core family's kmeans_fit already serves through
    # cluster/host/kmeans_oracle.mojo (the rooted euclidean metric, fixed
    # with the device in 9fde8f5f7, and the classic sequential k-means++
    # start), the cosine metric's refusal, which the oracle raises in the
    # device's words so the cell is the same refusal sentence on every
    # column, and cross_val_score over the gbdt family's RMSE fit, the core
    # family's fold-row gather (gather_rows_bytes) and the metrics family's
    # r2. On the M4, one core: all 36 cells IDENTICAL x4 (kmeans-sqrt
    # against TRAINING_FIX_COLUMNS), and the sabotage set DIVERGENT on
    # kmeans-sqrt, kmeans-classic-pp and cross-val on every fixture.
    "kmeans-sqrt": "k-means under the rooted euclidean metric",
    "kmeans-classic-pp": "k-means from the classic k-means++ start",
    "kmeans-cosine": "the refusal of k-means under cosine distance",
    "cross-val": "cross-validation of gradient boosting",
    # lane/cpu-training-misc batch 2 (2026-09-15): the resampling functions
    # through the resample family's own host binding
    # (resample/host/resample_host.mojo: the bootstrap replicate folds and
    # the sorted order statistics, the permutation ranks and masked folds,
    # the Monte Carlo chunk trees, restated from resample/estimator.mojo's
    # kernels, and the host finish that file already runs). On the M4, one
    # core: all 27 train cells IDENTICAL x4 against the 166-lane record
    # before the gate ran.
    "bootstrap": "the bootstrap",
    "permutation-test": "the permutation test",
    "monte-carlo": "Monte Carlo integration",
    # lane/cpu-training-misc batch 3 (2026-09-15): the neural primitives
    # through the training family's host binding. optim-sgd and
    # cross-entropy-arms reach optimizer_step and ce_loss (the mlp lane's
    # entries); optim-adam-clip adds clip_grad_norm over
    # training/checks/optimizer_oracle.mojo and accumulate over
    # training/host/samba_ops_oracle.mojo; training-primitives adds the
    # embedding, RMSNorm and linear operations over the same file (the
    # embedding and GEMM oracles and the RMSNorm kernels' statements with
    # the caller's eps). On the M4, one core: all 36 train, 9 infer and 18
    # batch cells IDENTICAL x4 before the gate ran.
    "optim-sgd": "SGD with momentum, Nesterov and dampening",
    "optim-adam-clip": "Adam and AdamW with the gradient clip and accumulation",
    "cross-entropy-arms": "the cross-entropy loss arms",
    "training-primitives": "the embedding, RMSNorm and linear training primitives",
    # CPU training for the workstream D estimators
    # (lane/cpu-training-d-estimators, 2026-09-15). The Cholesky door through
    # the gp host binding's cholesky_factor and cholesky_solve (the factor is
    # now chol_host_potrf over chol_host_factor_lower, same bits); KernelRidge,
    # Nystroem and RBFSampler through kernel_methods/host/km_host_oracle.mojo;
    # GaussianMixture through mixture/host/gmm_host_oracle.mojo; HDBSCAN
    # through hdbscan/host/hdbscan_host_oracle.mojo (the Boruvka round count
    # included). On the M4's CPU column (one core, shared machine) every
    # train, infer and batch cell of the eight lanes read IDENTICAL x4
    # against the 166-lane record's three GPU columns before the gate ran,
    # and each family's sabotage build read DIVERGENT on every train cell.
    "cholesky": "the Cholesky factorization and solve",
    "rbf-sampler": "random Fourier features",
    "kernel-ridge": "kernel ridge",
    "nystroem": "the Nystroem kernel approximation",
    "gmm": "the Gaussian mixture",
    "gmm-random-init": "the Gaussian mixture with a random start",
    "hdbscan": "HDBSCAN",
    "hdbscan-leaf": "HDBSCAN with leaf selection",
    # The forest variant lanes (lane/cpu-training-forest-variants,
    # 2026-09-15). rf-clf-entropy-log2-noboot was already served by
    # ensemble/host/rf_oracle.mojo (entropy, log2 features, no bootstrap, the
    # level-order leaf cap); the oracle gained the POISSON, GAMMA and
    # INVERSE_GAUSSIAN gains and the class-weighted bootstrap, the rf binding
    # rf_classifier_fit_weighted, and both forest bindings the resident
    # parallel_groves entries over core/forest_host_groves.mojo; ExtraTrees'
    # best-first growth fits through train_tree_exact_bestfirst. On the M4's
    # CPU column (one core), all 54 train, 108 infer and model and 54 batch
    # cells IDENTICAL x4 against the 166-lane record before the gate ran, and
    # the sabotage build DIVERGENT on every one.
    "rf-clf-entropy-log2-noboot": "the random forest classifier with entropy splits, log2 features and no bootstrap",
    "rf-clf-balanced-parallel": "the class-weighted random forest classifier with the parallel groves engine",
    "rf-reg-poisson": "the random forest regressor with the Poisson criterion",
    "rf-reg-gamma-ig": "the random forest regressor with the gamma and inverse Gaussian criteria",
    "et-clf-entropy-bestfirst": "the best-first Extra Trees classifier with entropy splits",
    "et-reg-bootstrap-parallel": "the bootstrapped Extra Trees regressor with the parallel groves engine",
    # The Mamba block lanes (lane/cpu-training-mamba, 2026-09-15): the three
    # blocks' forward, carried-state prefill and decode step over their host
    # oracles, and the zero-state prefill backward over mamba/host/gen/, the
    # device VJP written out for the host by tools/mamba_host_gen.py (the
    # Mamba-1 host backward oracle differs from the device in low bits, so
    # it is not the one used). On the M4, one core, shared machine: all 36
    # train (forward, prefill, step and backward parts), 36 infer and 36
    # batch cells IDENTICAL x4 against the 166-lane record before the gate
    # ran, and the MOJOLEARN_HOST_SABOTAGE build DIVERGENT.
    "mamba2": "the Mamba-2 block",
    "mamba2-dtlimit": "the Mamba-2 block with an active dt clamp",
    "mamba1": "the Mamba-1 block",
    "mamba3": "the Mamba-3 block",
    # The Transformer block lanes (lane/cpu-training-transformer,
    # 2026-09-15). TransformerBlock's forward (the stateless prefill, the
    # carried-state prefill and the decode step) and its zero-state prefill
    # backward run through the lane's own host oracles,
    # transformer/checks/transformer_oracle.mojo and
    # transformer_backward_oracle.mojo, composed by
    # transformer/host/transformer_block_host.mojo, which also converts the KV
    # cache between the device's packed (or ring) layout and the oracle's.
    # The sliding window is the same oracles' window argument.
    "transformer": "the Transformer block",
    "transformer-window": "the sliding-window Transformer block",
    # The Samba stack lanes (lane/cpu-training-samba, 2026-09-15). SambaStack
    # is Python over the training, mamba and transformer bindings; the one
    # operation it reached with no host entry was the neural RNG (the
    # initializers and dropout), now the training host binding's neural_rng
    # over core/philox_neural.mojo written out by tools/mamba_host_gen.py. On
    # the M4, one core, shared machine: all 18 train, 36 infer and model and
    # 18 batch cells IDENTICAL x4 against the 166-lane record before the gate
    # ran, the MOJOLEARN_HOST_SABOTAGE set DIVERGENT on every one, and a
    # throwaway dropout-mask arm DIVERGENT on samba-untied-dropout-accum only.
    "samba": "the Samba stack",
    "samba-untied-dropout-accum": "the Samba stack with untied embeddings, dropout, accumulation, clipping and a cosine schedule",
    # The byte LM host lanes (lane/cpu-training-host-only-lanes, 2026-09-15).
    # Host code on every box, so the 166-lane record's three columns are the
    # host CPUs of the Apple M4, the H100 box and the MI325X box, IDENTICAL x3
    # on every cell; the CPU column is a fourth CPU. This list is the set the
    # full CPU column runs, not only fits: two of these are inference, and
    # LanguageModelHostTrainer is the published CPU trainer the inference
    # boundary (docs/lanes/CPU_INFERENCE_BOUNDARY_2026-09-15.md) keeps.
    "byte-lm-host-infer": "the byte LM forward pass on its reference path (inference)",
    "byte-lm-host-infer-threaded": "the byte LM forward pass on its threaded path (inference)",
    "byte-lm-host-train": "the published byte LM host training step",
    # CPU training for the par-* lanes whose driver shards in Python
    # (lane/cpu-training-par-classical, 2026-09-15). fit_scaler and
    # transform_scaler (four column shards), fit_arima and
    # fit_exponential_smoothing (two series per shard) run each shard as a
    # host fit in its own worker process and merge in shard order through
    # the drivers' unchanged code (_parallel_pool.CPU_OPERATIONS); the
    # cooperative drivers refuse by name. On the M4's CPU column (one core)
    # every train, infer and batch cell of the three lanes read IDENTICAL x4
    # against the 166-lane record before the gate ran, and the sabotage set
    # DIVERGENT on every train cell.
    "par-scaler": "the column-sharded standard scaler",
    "par-arima": "series-sharded ARIMA",
    "par-holtwinters": "series-sharded Holt-Winters",
    # Wave 2 (lane/cpu-training-par-wave2, 2026-09-15): the neighbor
    # drivers. ParallelQueries cuts query rows in Python (four shards of 16
    # rows); ReferenceShardedNeighbors cuts the reference into four shards
    # of 1024 rows, merges the shard candidates by composite key in Python
    # and sends one vote request, served on CPU by the core host binding's
    # knn_classify_neighbors and knn_regress_neighbors.
    "par-queries-knn": "query-sharded k-NN classification",
    "par-queries-radius": "query-sharded radius neighbors",
    "par-queries-kde": "query-sharded kernel density",
    "par-reference-knn": "reference-sharded k-NN classification",
    "par-reference-knn-reg": "reference-sharded k-NN regression",
    # Wave 2, the forest driver: fit_forest cuts 16 trees into four global
    # tree ID ranges in Python; each range is the rf or trees host
    # binding's shard fit (rf_classifier_fit_shard, et_regressor_fit_shard),
    # the GPU bindings' tree_start offset restated on the host, and the
    # trees concatenate in ID order.
    "par-forest": "the tree-range-sharded random forest classifier",
    "par-forest-et": "the tree-range-sharded Extra Trees regressor",
    # Wave 2, par-mlp: ParallelNeuralTrainer sends one mlp_gradient request
    # per logical shard (three of 64 rows) and one mlp_update that folds
    # them in shard order in Python and steps the optimizer, on the
    # training host binding. The update pool is cooperative and is admitted
    # on CPU only at one device, where the GPU binding's range split is the
    # plain path; two devices refuse by name.
    "par-mlp": "the small MLP trained over ordered logical gradient shards",
    # The Embedding layer and IVFIndex (lane/cpu-training-embedding-ivf,
    # 2026-09-15). Embedding's gather and fold, both execution plans, the
    # padding row and the microbatch carry, through
    # embedding/host/embedding_host.mojo (the device launch restated, not
    # the contract's oracle) in the embedding family's own host binding.
    # Their GPU cells are not in the 166-lane record; they are diffed against
    # TRAINING_EXTRA_RECORDS below.
    "embedding": "the Embedding layer",
    "embedding-sort": "the Embedding layer on its sorted execution plan",
    # IVFIndex's build and search through ivf/host/ivf_host.mojo (the k-means
    # quantizer through cluster/host/kmeans_oracle.mojo) in the ivf family's
    # own host binding, under both L2 metrics.
    "ivf": "the IVF-Flat index",
    "ivf-euclidean": "the IVF-Flat index under euclidean distance",
    # lane/inference-embedding-ivf-cholesky stage 2 (2026-09-15): the rows
    # added to a built index by IVFIndex.extend. No GPU record carries the
    # lane yet, so every cell is OWED against the record.
    "ivf-extend": "extending a built IVF-Flat index",
    # The GPU byte LM trainer's lanes (same branch): SmallByteLanguageModelTrainer,
    # stateless and on its resident session, whose single-device entries a
    # CPU-only install serves from the byte LM host binding's step, loss and
    # logits (ADAPTED_MODULES below). Both lanes are in the 166-lane record.
    "byte-lm": "the byte LM trainer",
    "byte-lm-resident": "the byte LM trainer on its resident session",
}

#: GPU binding families a CPU-only install serves through a Python adapter
#: over a host binding loaded by path, rather than through a routed host
#: binding exporting the GPU names (lane/cpu-training-embedding-ivf,
#: 2026-09-15). `_backend.binding(name)` returns `<module>.binding()` when
#: the family's host binding is built, and the installed stub refuses by
#: name when it is not. The byte LM trainer's GPU binding carries a resident
#: session ABI and multi-GPU entries a Mojo host binding would have to
#: restate as state; the adapter holds the session's bookkeeping and the
#: CPU byte LM binding's step, loss and logits hold all of the arithmetic.
ADAPTED_MODULES = {
    "_mojolearn_byte_lm": dict(family="byte_lm", module="_byte_lm_trainer_host"),
}

#: The lanes with NO CPU path of any kind, as the README states them. A
#: lane leaves this list the day its host lane merges; docs_facts fails the
#: README until the marked span is rewritten.
NO_CPU_PATH = (
    "gradient boosting training outside its declared lanes (CTR categorical features, and sample weights, eval sets and the pointwise searcher outside the gbdt-pointwise-l2-bayesian-eval configuration, among them)",
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
        # lane/cpu-training-host-only-lanes (2026-09-15). The CPU identity
        # gate loads this binding from MOJOLEARN_HOST_DIR too:
        # identity_break's host_record loads every host binding through
        # _backend.load_host_module under the module name _byte_lm_host.py
        # reuses. Its sabotage set (-D MOJOLEARN_HOST_SABOTAGE=1) reads
        # DIVERGENT on all nine fixtures of each lane (the threaded head
        # through training/byte_lm_host.mojo's reverse flag since this
        # branch); this family's own define moves the two inference lanes
        # and not the training step, which the byte LM CPU gate's
        # wrong-gradient build covers.
        # lane/cpu-training-embedding-ivf (2026-09-15): the GPU trainer's two
        # lanes, through python/mojolearn/_byte_lm_trainer_host.py
        # (ADAPTED_MODULES). Loaded through _backend.load_host_module for
        # them, so the CPU identity gate's sabotage set (built with
        # MOJOLEARN_HOST_SABOTAGE, gemm_oracle's descending leaf) reaches
        # them, and byte_lm_host_sabotage reports that arm too.
        training_lanes=("byte-lm-host-infer", "byte-lm-host-infer-threaded", "byte-lm-host-train",
                        "byte-lm", "byte-lm-resident"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("LanguageModelInference", "LanguageModelHostTrainer", "SmallByteLanguageModelTrainer"),
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
        gate=".github/workflows/byte-lm-cpu-gate.yml and tools/identity_break.py (cpu-identity-gate.yml)",
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
            # lane/inference-gbdt-modes (2026-09-15): saved OrderedRMSE,
            # ExperimentalTwoLevelFeatureFreq, pointwise Bayesian eval and
            # one-hot categorical models through HostGBDT. Their CPU
            # predictions read IDENTICAL against the 166-lane GPU columns
            # (identity_break MOJOLEARN_IDENTITY_HOST_INFER on the
            # gbdt-ordered-rmse, gbdt-feature-freq,
            # gbdt-pointwise-l2-bayesian-eval and gbdt-categorical-ctr lanes);
            # their forest gate fixtures are OWED to the next release record.
            "gbdt_ordered_rmse", "gbdt_feature_freq", "gbdt_pointwise_bayesian_eval",
            "gbdt_categorical_onehot",
        ),
        classes=(
            "RandomForestClassifier", "RandomForestRegressor",
            "ExtraTreesClassifier", "ExtraTreesRegressor", "GradientBoosting",
            "OrderedRMSE", "ExperimentalTwoLevelFeatureFreq",
        ),
        display="random forests, Extra Trees and eight gradient boosting variants",
        # lane/inference-gbdt-ctr-tables (2026-09-15): the CTR and tensor CTR
        # step of a saved GBDT model, reusing expand_raw_columns and the
        # tensor apply module the GPU predict calls
        host_modules=("core/forest_host_predict.mojo", "core/gbdt_host_predict.mojo",
                      "core/gbdt_host_ctr.mojo", "gbdt/models/tensor_ctr_apply.mojo"),
        exports=(
            "forest_host_numeric_mode", "forest_host_vendor", "forest_host_column",
            "forest_host_sabotage", "forest_host_rf_predict_proba",
            "forest_host_rf_predict_reg", "forest_host_et_predict",
            "forest_host_gbdt_predict", "forest_host_gbdt_sigmoid",
            "forest_host_gbdt_expand_ctr", "forest_host_gbdt_ctr_sabotage",
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
        # REFUSED from the covered lanes (lane/cpu-training-host-only-lanes,
        # 2026-09-15), though the `tokenizer` lane reads IDENTICAL x4 against
        # the 166-lane record on the M4 (9 infer and model cells). The full
        # CPU gate's sabotage step points MOJOLEARN_HOST_DIR at a set built
        # from byte_lm, forest and the routed families only, so this binding
        # is absent there: the lane read REFUSED on all nine fixtures
        # ("_mojolearn_tokenizer_host.so is not built"), and
        # cpu_identity_gate_check.py fails a covered lane that is not
        # STABLE. Even built into that set it could not be caught: the
        # binding holds integers and tables with no float fold, and
        # MOJOLEARN_HOST_SABOTAGE reaches nothing here (only this family's
        # define reverses gpt2_encode's ids). Covering it needs the gate to
        # build the tokenizer binding with its own define into the sabotage
        # set; until then test_tokenizer_surface.py is its gate.
        # mojolearn ships no vocabulary (2026-09-15): the lane loads the
        # synthetic one (python/mojolearn/_tokenizer_synthetic.py) at
        # identity_break's LANE_REVISIONS["tokenizer"], so the records above
        # hashed older input and its cells are owed to the next record.
        # gpt2_encode_batch (lane/inference-tokenizer-neural, 2026-09-15)
        # has its own negative control, -D MOJOLEARN_TOKENIZER_BATCH_SABOTAGE=1,
        # which the lane's batch part reads BATCH_MOVED.
        training_lanes=(),
        inference_lanes=(),
        forest_kinds=(),
        classes=("GPT2Tokenizer",),
        display="the byte-level BPE tokenizer (GPT-2 format, user-supplied vocabulary)",
        host_modules=(
            "tokenizer/encoding.mojo", "tokenizer/impl/bpe.mojo",
            "tokenizer/impl/pretokenize.mojo", "tokenizer/impl/ranks.mojo",
            "tokenizer/impl/unicode_class.mojo", "tokenizer/impl/byte_unicode.mojo",
        ),
        exports=(
            "tokenizer_host_numeric_mode", "tokenizer_host_vendor",
            "tokenizer_host_column", "tokenizer_host_sabotage", "gpt2_load",
            "gpt2_n_vocab", "gpt2_max_token_bytes", "gpt2_encode", "gpt2_encode_batch",
            "gpt2_decode",
        ),
        gate="pixi run check-tokenizer and python/mojolearn/tests/test_tokenizer_surface.py",
        ships_in_wheel=True,
    ),
    dict(
        # lane/inference-tokenizer-neural, 2026-09-15. The INFERENCE half of
        # the training and transformer families (which stay source reference
        # builds): the small MLP's logits and the TransformerBlock stateless
        # prefill, forward only, loaded by path and shipped. No optimizer,
        # loss, backward or decode export, so none of that is compiled in.
        # The mlp, transformer and transformer-window lanes' held-out and
        # batch cells run through MLPInference and TransformerBlockInference
        # on a CPU column; their training rows stay the training and
        # transformer families' covered lanes.
        family="neural",
        binding="_mojolearn_neural_host",
        routes=None,
        loaded_by="python/mojolearn/neural_inference.py",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=(),
        forest_kinds=(),
        # lane/inference-neural-forward, 2026-09-15: the Mamba-1, Mamba-2
        # and Mamba-3 zero-state forwards and the Samba stack's embedding,
        # final norm and head joined, forward only; the mamba family stays a
        # source reference build.
        classes=("MLPInference", "TransformerBlockInference", "Mamba1BlockInference",
                 "Mamba2BlockInference", "Mamba3BlockInference", "SambaInference"),
        display="the small MLP's logits, the Transformer and Mamba blocks' zero-state forward and the Samba stack's logits (inference only)",
        host_modules=(
            "training/host/mlp_oracle.mojo",
            "training/host/samba_ops_oracle.mojo",
            "transformer/host/transformer_block_host.mojo",
            "transformer/checks/transformer_oracle.mojo",
            "mamba/checks/mamba_oracle.mojo",
            "mamba/checks/mamba2_oracle.mojo",
            "mamba/checks/mamba3_oracle.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "neural_host_numeric_mode", "neural_host_vendor", "neural_host_column",
            "neural_host_sabotage", "mlp_forward_logits", "transformer_forward_fresh",
            "mamba1_forward_fresh", "mamba2_forward_fresh", "mamba3_forward_fresh",
            "embedding_forward", "rms_norm_forward", "linear_forward",
        ),
        gate="python/mojolearn/tests/test_neural_inference.py and tools/identity_break.py (mlp, transformer, transformer-window, mamba1, mamba2, mamba3, mamba2-dtlimit, samba, samba-untied-dropout-accum)",
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
            "kmeans-sqrt", "kmeans-classic-pp", "kmeans-cosine",
            "par-queries-knn", "par-queries-radius", "par-reference-knn",
            "par-reference-knn-reg",
        ),
        # The neighbors and density inference lane (2026-09-15) adds every
        # k-NN metric, the ball cover, the distance-weighted vote and mean
        # and RadiusNeighbors on its four metrics, all from a saved model.
        inference_lanes=("knn", "knn-clf", "knn-reg", "knn-sqeuclidean", "knn-manhattan",
                         "knn-chebyshev", "knn-cosine", "knn-minkowski-p3", "knn-rbc",
                         "knn-clf-distance", "knn-reg-distance", "radius", "radius-manhattan",
                         "radius-chebyshev", "radius-minkowski-p3"),
        forest_kinds=(),
        classes=(
            "NearestNeighbors", "KNeighborsClassifier", "KNeighborsRegressor", "KMeans",
            "RadiusNeighbors",
        ),
        display="nearest neighbors on every metric and the ball cover, k-NN classification and k-NN regression with either weighting and radius neighbors",
        host_modules=(
            "core/knn_host_predict.mojo", "bindings/host_helpers.mojo",
            "cluster/host/kmeans_oracle.mojo",
        ),
        exports=(
            "core_host_numeric_mode", "core_host_vendor", "core_host_column",
            "core_host_sabotage", "mojolearn_vendor", "mojolearn_numeric_mode",
            "knn_search", "knn_classify", "knn_regress", "kmeans_fit", "kmeans_predict", "kmeans_transform",
            "knn_classify_neighbors", "knn_regress_neighbors",
            "radius_neighbors_count", "radius_neighbors_fill", "rbc_knn_search", "transpose_f32",
            "cast_colmajor_f64_to_f32", "nonzero_f64_count", "nonzero_f64_fill", "cast_f64_to_f32", "all_finite_f32",
            "all_finite_f64", "gather_i64", "gather_f64", "gather_rows_bytes", "argmax_rows_f32",
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
        # The Cholesky door joined this family on
        # lane/inference-embedding-ivf-cholesky (2026-09-15) so that public
        # CPU Cholesky inference (a saved factor, or a factor of a given
        # matrix, then solve) ships: this family is in the wheel and gp is
        # not. On a CPU-only install `Cholesky` binds `_mojolearn_linalg`,
        # so the cholesky lane reads through this binding.
        training_lanes=("gemm-pinned", "gemm-transposed", "cholesky"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("linalg.gemm", "linalg.gemv", "Cholesky"),
        display="pinned GEMM and the Cholesky factorization and solve",
        host_modules=("gemm/host/gemm_oracle.mojo", "cholesky/host/chol_oracle.mojo"),
        exports=(
            "linalg_host_numeric_mode", "linalg_host_vendor", "linalg_host_column",
            "linalg_host_sabotage", "linalg_vendor", "linalg_numeric_mode",
            "linalg_profile_version", "gemm", "cholesky_profile_jitter",
            "cholesky_factor", "cholesky_solve",
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
            "par-queries-kde",
        ),
        # lane/inference-linear-svm (2026-09-15): the option variants of
        # ols, ridge and logistic load through the same formats; the
        # scalers, lasso and elasticnet and the three kernel methods load
        # through formats of their own, and their transform and predict
        # entries are served here (the reference-only preprocessing, solver
        # and kernel_methods bindings keep the fits).
        inference_lanes=(
            "ols", "ridge", "tsvd", "logistic", "logistic-multiclass", "pca", "pca-whiten", "kde",
            "ols-no-intercept", "ols-weighted", "ridge-no-intercept", "logistic-l1",
            "logistic-elasticnet", "logistic-unpenalized-no-intercept",
            "standard-scaler", "standard-scaler-no-mean", "standard-scaler-no-std",
            "minmax-scaler", "minmax-scaler-clip", "lasso", "elasticnet",
            "elasticnet-l2end-no-intercept", "kernel-ridge", "nystroem", "rbf-sampler", "pca-full-whiten",
            "kde-tophat-sqeuclidean", "kde-epanechnikov-l1", "kde-exponential-chebyshev",
            "kde-linear-cosine", "kde-cosine-minkowski", "kde-weighted",
        ),
        forest_kinds=(),
        classes=(
            "LinearRegression", "Ridge", "TruncatedSVD", "LogisticRegression",
            "PCA", "KernelDensity", "DBSCAN", "StandardScaler", "MinMaxScaler",
            "Lasso", "ElasticNet", "KernelRidge", "Nystroem", "RBFSampler",
            "AgglomerativeClustering",
        ),
        display="linear regression, ridge, truncated SVD, logistic regression, PCA with and without whitening (either solver), kernel density on every kernel, metric and weighting, the standard and min-max scalers, lasso, elasticnet, kernel ridge, the Nystroem approximation and random Fourier features",
        # lane/inference-transductive-predict (2026-09-15): `dbscan_fit_core`
        # (the fit's core mask for DBSCAN(prediction_data=True)) and
        # `labeled_reference_predict`, the out-of-sample labels of DBSCAN
        # and AgglomerativeClustering (DEVIATION 2740). The agglomerative FIT
        # stays in the solver family, which does not ship; its predict entry
        # is here so a saved model predicts from the inference wheel.
        host_modules=(
            "kde/host/kde_oracle.mojo", "core/classical_host_predict.mojo",
            "decomposition/host/pca_oracle.mojo", "glm/host/glm_oracle.mojo",
            "dbscan/host/dbscan_oracle.mojo", "glm/host/qn_oracle.mojo",
            "decomposition/host/pca_full_oracle.mojo",
            "core/labeled_reference_host_predict.mojo",
            "preprocessing/host/scaler_oracle.mojo",
            "kernel_methods/host/km_host_oracle.mojo",
            "kernel_methods/checks/random_features.mojo",
            "cholesky/host/chol_oracle.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "estimators_host_numeric_mode", "estimators_host_vendor",
            "estimators_host_column", "estimators_host_sabotage",
            "estimators_vendor", "estimators_numeric_mode", "kde_score_samples",
            "pca_fit", "pca_fit_full", "tsvd_fit", "ols_fit", "ridge_fit", "dbscan_fit", "qn_fit",
            "dbscan_fit_core", "labeled_reference_predict",
            "ols_predict", "tsvd_transform", "pca_transform",
            "pca_whiten_transform", "pca_whiten_inverse_transform",
            "qn_decision_function", "qn_sigmoid", "qn_softmax",
            "standard_transform", "minmax_transform", "cd_predict",
            "kernel_ridge_predict", "nystroem_transform", "rbf_sampler_transform",
        ),
        gate="tools/identity_break.py and tools/classical_host_gate.py (cpu-identity-gate.yml)",
        ships_in_wheel=True,
    ),
    dict(
        # Workstream E batch 2 (lane/cpu-training-e2, 2026-09-14): the
        # metrics family's first host binding. It routes `_mojolearn_metrics`
        # on a CPU-only install and carries the five metrics the identity
        # harness's metrics lane computes plus the four label metrics that
        # share their integer kernels; the spectral entries joined in the
        # same batch and the UMAP entries on lane/cpu-training-umap-b
        # (umap_fit_transform, umap_transform, umap_numeric_mode); the
        # remaining metric entries stay absent and refuse by name.
        family="metrics",
        binding="_mojolearn_metrics_host",
        routes="_mojolearn_metrics",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("metrics", "spectral", "spectral-precomputed", "umap", "metrics-classification",
                        "metrics-fowlkes-mallows"),
        # UMAP.transform from a saved embedding (lane/inference-forecast-
        # umap-pca, 2026-09-15). Its answer depends on the query batch by the
        # transform's contract, so the claim is the GPU's bytes for the same
        # batch; `inference_display` says so in the README sentence.
        inference_lanes=("umap",),
        inference_display="UMAP transform of a saved embedding (the GPU's bytes for the same query batch; a row's embedding depends on the batch it is asked in)",
        forest_kinds=(),
        classes=(
            "SpectralClustering", "UMAP",
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
            "metrics.trustworthiness", "metrics.fowlkes_mallows_score",
        ),
        display="the label, classification, ranking, regression, r2, KL, silhouette and trustworthiness metrics, spectral clustering and UMAP",
        host_modules=(
            "metrics/host/metrics_oracle.mojo",
            "metrics/host/classification_oracle.mojo",
            "spectral/host/spectral_oracle.mojo",
            "cluster/host/kmeans_oracle.mojo",
            "core/knn_host_predict.mojo",
            "umap/host/umap_oracle.mojo",
            "umap/sparse_graph.mojo",
            "umap/graph.mojo",
            "umap/curve.mojo",
            "umap/params.mojo",
        ),
        exports=(
            "metrics_host_numeric_mode", "metrics_host_vendor",
            "metrics_host_column", "metrics_host_sabotage", "metrics_vendor",
            "metrics_numeric_mode", "accuracy_score", "adjusted_rand_score",
            "entropy", "mutual_info_score", "homogeneity_score",
            "completeness_score", "v_measure_score", "r2_score", "silhouette",
            "spectral_fit_predict_dataset", "spectral_fit_predict_graph",
            "umap_fit_transform", "umap_transform", "umap_numeric_mode",
            "rand_score", "mean_squared_error", "mean_absolute_error",
            "root_mean_squared_error", "roc_auc_score", "precision_recall_curve",
            "log_loss", "confusion_matrix", "precision_recall_fscore",
            "kl_divergence", "trustworthiness", "fowlkes_mallows_score",
            "accuracy_score_weighted", "r2_score_weighted",
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
            "standard-scaler-no-std", "minmax-scaler-clip", "par-scaler",
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
        ships_in_wheel=False,
    ),
    dict(
        family="tsa",
        binding="_mojolearn_tsa_host",
        routes="_mojolearn_tsa",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("holtwinters", "holtwinters-multiplicative", "kpss", "par-holtwinters"),
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
        ships_in_wheel=False,
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
        ships_in_wheel=False,
    ),
    dict(
        family="svm",
        binding="_mojolearn_svm_host",
        routes="_mojolearn_svm",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("svc", "iforest", "svc-linear", "iforest-tuned", "svr", "svr-linear", "svc-poly"),
        # The neighbors and density inference lane (2026-09-15): a saved
        # IsolationForest scores through iforest_run, the same forest rebuild
        # every GPU scoring call runs (DEVIATION 874).
        inference_lanes=("svc", "svc-linear", "svc-poly", "svr", "svr-linear", "iforest", "iforest-tuned"),
        forest_kinds=(),
        classes=("SVC", "IsolationForest", "SVR"),
        display="SVC and the isolation forest",
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
        training_lanes=("et-clf", "et-reg", "et-clf-entropy-bestfirst", "et-reg-bootstrap-parallel", "par-forest-et"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("ExtraTreesClassifier", "ExtraTreesRegressor"),
        display="the Extra Trees classifier and regressor",
        host_modules=(
            "extratrees/estimator.mojo", "extratrees/checks/pcg_rng.mojo",
            "core/forest_host_predict.mojo", "core/forest_host_groves.mojo",
        ),
        exports=(
            "trees_host_numeric_mode", "trees_host_vendor", "trees_host_column",
            "trees_host_sabotage", "trees_vendor", "trees_numeric_mode",
            "et_classifier_fit", "et_classifier_fit_export",
            "et_classifier_fit_rowmajor", "et_classifier_fit_rowmajor_export",
            "et_regressor_fit", "et_regressor_fit_export",
            "et_regressor_fit_rowmajor", "et_regressor_fit_rowmajor_export",
            "forest_export", "forest_export_legacy", "forest_export_release",
            "et_classifier_fit_shard", "et_regressor_fit_shard",
            "et_predict", "forest_prepare_gpu", "forest_predict_resident_reuse_gpu",
            "forest_release_gpu",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
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
        training_lanes=(
            "rf-clf", "rf-reg", "rf-clf-entropy-log2-noboot", "rf-clf-balanced-parallel",
            "rf-reg-poisson", "rf-reg-gamma-ig", "par-forest", "rf-score-weighted",
        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=("RandomForestClassifier", "RandomForestRegressor"),
        display="the random forest classifier and regressor",
        host_modules=(
            "ensemble/host/rf_oracle.mojo", "core/forest_host_predict.mojo",
            "core/forest_host_groves.mojo",
        ),
        exports=(
            "rf_host_numeric_mode", "rf_host_vendor", "rf_host_column",
            "rf_host_sabotage", "rf_vendor", "rf_numeric_mode",
            "rf_classifier_fit", "rf_classifier_fit_export",
            "rf_classifier_fit_rowmajor", "rf_classifier_fit_rowmajor_export",
            "rf_regressor_fit", "rf_regressor_fit_export",
            "rf_regressor_fit_rowmajor", "rf_regressor_fit_rowmajor_export",
            "forest_export", "forest_export_legacy", "forest_export_release",
            "rf_predict_proba", "rf_predict_reg",
            "rf_classifier_fit_weighted", "rf_classifier_fit_weighted_export",
            "rf_classifier_fit_shard", "rf_regressor_fit_shard",
            "forest_prepare_gpu", "forest_predict_resident_reuse_gpu", "forest_release_gpu",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        # Workstream E, the gp host lane (2026-09-14): the Gaussian process
        # family's host binding. It routes `_mojolearn_gp` on a CPU-only
        # install with the GPU binding's gpr_fit and gpr_predict and the
        # Cholesky door workstream D put on the same binding
        # (cholesky_factor, cholesky_solve, cholesky_profile_jitter). The
        # cholesky lane, which the 136-lane record predated, is covered since
        # lane/cpu-training-d-estimators (2026-09-15) against the 166-lane
        # record. gp_parallel_available stays absent, so the ordered
        # multi-GPU driver refuses by name. Since
        # lane/inference-embedding-ivf-cholesky (2026-09-15) the cholesky
        # lane and the Cholesky class are the linalg family's: a CPU-only
        # install binds Cholesky to `_mojolearn_linalg`, which ships. This
        # binding still exports the three door names for its own GP.
        family="gp",
        binding="_mojolearn_gp_host",
        routes="_mojolearn_gp",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        # GaussianProcessClassifier joined on
        # lane/gaussian-process-classifier (2026-09-15): gpc_fit and
        # gpc_predict under the GPU binding's contract.
        training_lanes=("gp", "gp-matern12", "gp-matern32", "gp-matern52-ard", "gp-normalize-y",
                        "gpc", "gpc-multiclass"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("GaussianProcessRegressor", "GaussianProcessClassifier"),
        display="the Gaussian process regressor and classifier",
        host_modules=(
            "gaussian_process/host/gpr_oracle.mojo",
            "gaussian_process/host/gpc_oracle.mojo",
            "gaussian_process/host/gpc_steps.mojo",
            "cholesky/host/chol_oracle.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "gp_host_numeric_mode", "gp_host_vendor", "gp_host_column",
            "gp_host_sabotage", "gp_vendor", "gp_numeric_mode",
            "gpr_fit", "gpr_predict", "gpr_sample_y", "gpc_fit", "gpc_predict", "cholesky_profile_jitter",
            "cholesky_factor", "cholesky_solve",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        # CPU training for the workstream D estimators
        # (lane/cpu-training-d-estimators, 2026-09-15): the kernel methods
        # family's host binding. It routes `_mojolearn_kernel_methods` on a
        # CPU-only install with the GPU binding's fit, predict and transform
        # names for KernelRidge, Nystroem and RBFSampler at the linear and
        # rbf kernels; the polynomial, sigmoid and laplacian kernels refuse
        # by name, and kernel_methods_rows_parallel_available stays absent,
        # so the multi-GPU driver refuses by name.
        family="kernel_methods",
        binding="_mojolearn_kernel_methods_host",
        routes="_mojolearn_kernel_methods",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("rbf-sampler", "kernel-ridge", "nystroem"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("KernelRidge", "Nystroem", "RBFSampler"),
        display="kernel ridge, the Nystroem approximation and random Fourier features",
        host_modules=(
            "kernel_methods/host/km_host_oracle.mojo",
            "kernel_methods/checks/random_features.mojo",
            "cholesky/host/chol_oracle.mojo",
            "decomposition/host/pca_oracle.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "kernel_methods_host_numeric_mode", "kernel_methods_host_vendor",
            "kernel_methods_host_column", "kernel_methods_host_sabotage",
            "kernel_methods_vendor", "kernel_methods_numeric_mode",
            "kernel_ridge_fit", "kernel_ridge_predict", "nystroem_fit",
            "nystroem_transform", "rbf_sampler_fit", "rbf_sampler_transform",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        # CPU training for the workstream D estimators
        # (lane/cpu-training-d-estimators, 2026-09-15): the Gaussian mixture
        # family's host binding. It routes `_mojolearn_mixture` on a CPU-only
        # install with the GPU binding's fit and scoring names;
        # gmm_parallel_available stays absent, so the multi-GPU driver
        # refuses by name.
        family="mixture",
        binding="_mojolearn_mixture_host",
        routes="_mojolearn_mixture",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("gmm", "gmm-random-init"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("GaussianMixture",),
        display="the Gaussian mixture",
        host_modules=(
            "mixture/host/gmm_host_oracle.mojo",
            "cluster/host/kmeans_oracle.mojo",
            "cholesky/host/chol_oracle.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "mixture_host_numeric_mode", "mixture_host_vendor",
            "mixture_host_column", "mixture_host_sabotage",
            "mixture_vendor", "mixture_numeric_mode", "gmm_fit",
            "gmm_score_samples", "gmm_predict_proba", "gmm_predict",
            "gmm_score_bic_aic", "gmm_sample",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        # The neighbors and density inference lane (2026-09-15): the
        # INFERENCE-ONLY mixture binding a wheel ships. It registers the four
        # scoring entries (bindings/mixture_host_scoring.mojo, the same
        # functions the reference binding above registers) and no fit, so
        # gmmh_fit and the starts are not compiled in; the neural family's
        # pattern. `routes` stays None (the reference binding keeps the
        # route); a saved model is served through `mojolearn.host_model`,
        # whose host class binds this file, as lane/inference-linear-svm
        # serves the scalers through the estimators binding.
        family="mixture_infer",
        binding="_mojolearn_mixture_infer_host",
        routes=None,
        loaded_by="python/mojolearn/_classical_host.py (mojolearn.host_model)",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=("gmm", "gmm-random-init", "gmm-sample", "gmm-random-init-sample"),
        forest_kinds=(),
        classes=("GaussianMixture",),
        display="the Gaussian mixture's scores, probabilities, labels and samples",
        host_modules=(
            "bindings/mixture_host_scoring.mojo",
            "mixture/host/gmm_host_oracle.mojo",
            "mixture/checks/sample.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "mixture_infer_host_numeric_mode", "mixture_infer_host_vendor",
            "mixture_infer_host_column", "mixture_infer_host_sabotage",
            "mixture_vendor", "mixture_numeric_mode",
            "gmm_score_samples", "gmm_predict_proba", "gmm_predict",
            "gmm_score_bic_aic", "gmm_sample",
        ),
        gate="tools/classical_host_gate.py",
        ships_in_wheel=True,
    ),
    dict(
        # CPU training for the workstream D estimators
        # (lane/cpu-training-d-estimators, 2026-09-15): the HDBSCAN family's
        # host binding. It routes `_mojolearn_hdbscan` on a CPU-only install
        # with the GPU binding's fit name; hdbscan_rows_parallel_available
        # stays absent, so the multi-GPU driver refuses by name.
        family="hdbscan",
        binding="_mojolearn_hdbscan_host",
        routes="_mojolearn_hdbscan",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("hdbscan", "hdbscan-leaf"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("HDBSCAN",),
        display="HDBSCAN",
        host_modules=(
            "hdbscan/host/hdbscan_host_oracle.mojo",
            "core/knn_host_predict.mojo",
            "hierarchy/checks/linkage_oracle.mojo",
            "hdbscan/impl/detail/condense.mojo",
            "hdbscan/impl/detail/extract.mojo",
            "hdbscan/impl/detail/utils.mojo",
            "hdbscan/impl/condensed_hierarchy.mojo",
        ),
        exports=(
            "hdbscan_host_numeric_mode", "hdbscan_host_vendor",
            "hdbscan_host_column", "hdbscan_host_sabotage",
            "hdbscan_vendor", "hdbscan_numeric_mode", "hdbscan_fit",
            "hdbscan_generate_prediction_data", "hdbscan_approximate_predict",
            "hdbscan_host_predict_sabotage", "hdbscan_membership_vector",
            "hdbscan_all_points_membership_vectors",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        # The neighbors and density inference lane (2026-09-15): the
        # INFERENCE-ONLY gp binding a wheel ships, gpr_predict from a saved
        # model (bindings/gp_host_predict.mojo, the function the reference
        # binding registers) and no fit, log marginal likelihood or Cholesky
        # door. normalize_y's scale-back is host Python. Loaded like
        # mixture_infer.
        family="gp_infer",
        binding="_mojolearn_gp_infer_host",
        routes=None,
        loaded_by="python/mojolearn/_classical_host.py (mojolearn.host_model)",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=("gp", "gp-matern12", "gp-matern32", "gp-matern52-ard", "gp-normalize-y",
                         "gpc", "gpc-multiclass"),
        forest_kinds=(),
        classes=("GaussianProcessRegressor", "GaussianProcessClassifier"),
        display="the Gaussian process regressor's predictive mean and std, normalized targets included, and the Gaussian process classifier's labels and probabilities",
        host_modules=(
            "bindings/gp_host_predict.mojo",
            "gaussian_process/host/gpr_oracle.mojo",
            "gaussian_process/host/gpc_oracle.mojo",
            "gaussian_process/host/gpc_steps.mojo",
        ),
        exports=(
            "gp_infer_host_numeric_mode", "gp_infer_host_vendor",
            "gp_infer_host_column", "gp_infer_host_sabotage",
            "gp_vendor", "gp_numeric_mode", "gpr_predict", "gpc_predict",
        ),
        gate="tools/classical_host_gate.py",
        ships_in_wheel=True,
    ),
    dict(
        # The neighbors and density inference lane (2026-09-15): the
        # INFERENCE-ONLY hdbscan binding a wheel ships, approximate_predict,
        # membership_vector and all_points_membership_vectors from a saved
        # model (bindings/hdbscan_host_predict.mojo, the same functions the
        # reference binding registers) and no fit, prediction data
        # generation or tree building. Loaded like mixture_infer.
        family="hdbscan_infer",
        binding="_mojolearn_hdbscan_infer_host",
        routes=None,
        loaded_by="python/mojolearn/_classical_host.py (mojolearn.host_model)",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=("hdbscan", "hdbscan-leaf"),
        forest_kinds=(),
        classes=(
            "hdbscan.approximate_predict", "hdbscan.membership_vector",
            "hdbscan.all_points_membership_vectors",
        ),
        display="HDBSCAN's approximate_predict, membership_vector and all_points_membership_vectors",
        host_modules=(
            "bindings/hdbscan_host_predict.mojo",
            "hdbscan/host/hdbscan_host_oracle.mojo",
        ),
        exports=(
            "hdbscan_infer_host_numeric_mode", "hdbscan_infer_host_vendor",
            "hdbscan_infer_host_column", "hdbscan_infer_host_sabotage",
            "hdbscan_vendor", "hdbscan_numeric_mode", "hdbscan_approximate_predict",
            "hdbscan_membership_vector", "hdbscan_all_points_membership_vectors",
        ),
        gate="tools/classical_host_gate.py",
        ships_in_wheel=True,
    ),
    dict(
        # Workstream E batch 3 (2026-09-14): the GradientBoosting family's
        # host binding. It routes `_mojolearn_gbdt` on a CPU-only install
        # with the GPU binding's fit, predict, model-dim and sigmoid names;
        # gbdt_fit refuses by name every value outside the declared GBDT
        # configurations. The classifier adapter's binary probability and
        # class transforms and the multi-dimensional predict are exported
        # since 2026-09-15 (lane/cpu-training-gbdt-losses);
        # gbdt_fit_ordered_rmse and gbdt_fit_two_level_feature_freq train the
        # gbdt-ordered-rmse and gbdt-feature-freq lanes, gbdt_fit's
        # use_pointwise_searcher arm the gbdt-pointwise-l2-bayesian-eval lane
        # and its one-hot categorical columns the gbdt-categorical-ctr lane
        # (lane/cpu-training-gbdt-ordered, 2026-09-15).
        # Not the forest host binding: that one is loaded by path under its
        # own names and takes the model as flat arrays.
        family="gbdt",
        binding="_mojolearn_gbdt_host",
        routes="_mojolearn_gbdt",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(
            "gbdt-symmetric", "gbdt-rmse", "gbdt-depthwise", "gbdt-lossguide", "cross-val",
            "gbdt-nan-modes", "gbdt-adapter-clf", "gbdt-adapter-reg",
            "gbdt-parametric-losses", "gbdt-exact-mae",
            "gbdt-lossguide-newtoncosine", "gbdt-multiclass", "gbdt-onevsall",
            "gbdt-ordered-rmse", "gbdt-feature-freq",
            "gbdt-pointwise-l2-bayesian-eval", "gbdt-categorical-ctr",
            "gbdt-adapter-score-weighted",
            "gbdt-query-rmse", "gbdt-pair-logit", "gbdt-yeti-rank",
        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=(
            "GradientBoosting", "GradientBoostingClassifier", "GradientBoostingRegressor",
            "model_selection.cross_val_score", "OrderedRMSE", "ExperimentalTwoLevelFeatureFreq",
        ),
        display="gradient boosting on symmetric trees with the pointwise, multiclass, QueryRMSE, PairLogit and YetiRank losses, either NaN mode and the classifier and regressor adapters, and on depthwise and lossguide trees with the Logloss loss; one-hot categorical columns, the pointwise searcher with L2 scores, the Bayesian bootstrap and an eval set, OrderedRMSE and the two-level FeatureFreq estimator",
        host_modules=(
            "gbdt/host/gbdt_oracle.mojo", "gbdt/host/gbdt_oracle_rmse.mojo",
            "gbdt/host/gbdt_oracle_depthwise.mojo", "gbdt/host/gbdt_oracle_lossguide.mojo",
            "gbdt/host/gbdt_oracle_losses.mojo", "gbdt/host/gbdt_oracle_multiclass.mojo",
            "gbdt/host/gbdt_oracle_ordered.mojo", "gbdt/host/gbdt_oracle_feature_freq.mojo",
            "gbdt/host/gbdt_oracle_pointwise.mojo", "gbdt/host/gbdt_oracle_onehot.mojo",
            "gbdt/host/gbdt_oracle_query.mojo", "gbdt/host/gbdt_oracle_pair.mojo",
            "gbdt/data/pairs.mojo",
            "gbdt/host/gbdt_oracle_yeti.mojo", "gbdt/data/yeti_rank_tasks.mojo",
            "core/gbdt_host_predict.mojo",
        ),
        exports=(
            "gbdt_host_numeric_mode", "gbdt_host_vendor", "gbdt_host_column",
            "gbdt_host_sabotage", "gbdt_vendor", "gbdt_numeric_mode",
            "gbdt_fit", "gbdt_predict", "gbdt_predict_multi", "gbdt_model_dim",
            "gbdt_sigmoid", "gbdt_binary_probabilities", "gbdt_binary_classes",
            "gbdt_fit_ordered_rmse", "gbdt_fit_two_level_feature_freq",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        # The mlp lane (lane/cpu-training-mlp, 2026-09-14): the training
        # family's host binding. It routes `_mojolearn_training` on a
        # CPU-only install with the GPU binding's optimizer_step, ce_loss and
        # the three small MLP operations, so SmallMLPTrainer (and the
        # optimizers and cross_entropy on their own) run unchanged;
        # lane/cpu-training-misc batch 3 (2026-09-15) adds clip_grad_norm,
        # accumulate, accumulation_is_aligned and the embedding, RMSNorm and
        # linear forward and backward; lane/cpu-training-samba (2026-09-15)
        # adds neural_rng (core/philox_neural.mojo's kernel as
        # tools/mamba_host_gen.py writes it out for the host), the last
        # operation SambaStack reaches that was missing, so the samba lanes
        # train on the CPU through this family with the mamba and
        # transformer families' blocks. The multi-GPU probes stay absent and
        # refuse by name.
        family="training",
        binding="_mojolearn_training_host",
        routes="_mojolearn_training",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("mlp", "optim-sgd", "optim-adam-clip", "cross-entropy-arms", "training-primitives", "par-mlp",
                        "samba", "samba-untied-dropout-accum"),
        inference_lanes=(),
        forest_kinds=(),
        classes=(
            "SmallMLPTrainer", "SGD", "Adam", "AdamW", "cross_entropy", "clip_grad_norm_",
            "accumulate_grads", "embedding_forward", "embedding_backward", "rms_norm_forward",
            "rms_norm_backward", "linear_forward", "linear_backward", "training.Generator",
            "SambaConfig", "SambaStack",
        ),
        display="the small MLP trainer, the optimizers, the gradient clip, the cross-entropy loss, the training primitives, the neural random stream and the Samba stack",
        host_modules=(
            "training/host/mlp_oracle.mojo",
            "training/checks/loss_oracle.mojo",
            "training/checks/optimizer_oracle.mojo",
            "training/host/samba_ops_oracle.mojo",
            "mamba/host/gen/philox_neural.mojo",
            "mamba/host/gen/philox.mojo",
            "embedding/checks/embedding_oracle.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "training_host_numeric_mode", "training_host_vendor",
            "training_host_column", "training_host_sabotage",
            "training_numeric_mode", "training_vendor", "optimizer_step",
            "ce_loss", "mlp_bias_activation", "mlp_relu_backward", "mlp_sum_rows",
            "clip_grad_norm", "accumulate", "accumulation_is_aligned",
            "embedding_forward", "embedding_backward", "rms_norm_forward",
            "rms_norm_backward", "linear_forward", "linear_backward",
            "neural_rng",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        # lane/cpu-training-misc batch 2 (2026-09-15): the resampling
        # family's host binding. It routes `_mojolearn_resample` on a
        # CPU-only install with the GPU binding's bootstrap,
        # permutation_test and monte_carlo_integrate over
        # resample/host/resample_host.mojo (resample/estimator.mojo's entry
        # points with every device kernel restated on the host);
        # resample_ranges_parallel_available is absent, so the multi-GPU
        # range drivers refuse by name.
        family="resample",
        binding="_mojolearn_resample_host",
        routes="_mojolearn_resample",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("bootstrap", "permutation-test", "monte-carlo"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("resample.bootstrap", "resample.permutation_test", "resample.monte_carlo_integrate"),
        display="bootstrap, the permutation test and Monte Carlo integration",
        host_modules=("resample/host/resample_host.mojo",),
        exports=(
            "resample_host_numeric_mode", "resample_host_vendor", "resample_host_column",
            "resample_host_sabotage", "resample_vendor", "resample_numeric_mode",
            "bootstrap", "permutation_test", "monte_carlo_integrate",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        # lane/cpu-training-mamba (2026-09-15): the Mamba blocks' host
        # binding. It routes `_mojolearn_mamba` on a CPU-only install with
        # the GPU binding's whole surface: the forwards and decode steps
        # over the three block oracles (mamba/checks/mamba{,2,3}_oracle.mojo),
        # and the three zero-state prefill VJPs over mamba/host/gen/, the
        # device passes (forward stages included) written out for the host
        # by tools/mamba_host_gen.py: each kernel a serial loop over its
        # launch grid, the device GEMM through gemm_oracle
        # (mamba/host/device_shim.mojo). The gate's manifest step fails when
        # the generated files lag the device source.
        family="mamba",
        binding="_mojolearn_mamba_host",
        routes="_mojolearn_mamba",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("mamba2", "mamba2-dtlimit", "mamba1", "mamba3"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("Mamba1Block", "Mamba2Block", "Mamba3Block"),
        display="the Mamba-1, Mamba-2 and Mamba-3 blocks' forward, decode step and prefill backward",
        host_modules=(
            "mamba/checks/mamba_oracle.mojo",
            "mamba/checks/mamba2_oracle.mojo",
            "mamba/checks/mamba3_oracle.mojo",
            "mamba/host/device_shim.mojo",
            "mamba/host/gen/modeling_mamba_prefill_backward.mojo",
            "mamba/host/gen/mamba2_prefill_backward.mojo",
            "mamba/host/gen/mamba3_prefill_backward.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "mamba_host_numeric_mode", "mamba_host_vendor", "mamba_host_column",
            "mamba_host_sabotage", "mamba_vendor", "mamba_numeric_mode",
            "mamba1_forward", "mamba1_backward", "mamba1_decode_step",
            "mamba2_forward", "mamba2_decode_step", "mamba2_backward",
            "mamba3_forward", "mamba3_forward_fresh", "mamba3_decode_step",
            "mamba3_backward",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        # Training-only reference family: source builds for internal bitwise
        # verification, not shipped (docs/lanes/CPU_INFERENCE_BOUNDARY_2026-09-15.md).
        ships_in_wheel=False,
    ),
    dict(
        # Workstream E (lane/cpu-training-arima, 2026-09-14): batched
        # ARIMA's host binding. It routes `_mojolearn_arima` on a CPU-only
        # install with the GPU binding's whole surface (fit, predict,
        # forecast); p, q or P above 1, any Q, d + D of 2 and p + q + k of 0
        # refuse by name (an in-sample prediction runs since 2026-09-15, and
        # the forecast family below ships the prediction half). par-arima is declared
        # since lane/cpu-training-par-classical (2026-09-15): fit_arima's
        # series shards run as host fits in their own workers.
        family="arima",
        binding="_mojolearn_arima_host",
        routes="_mojolearn_arima",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("arima", "arima-011", "arima-seasonal-c", "par-arima"),
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
        ships_in_wheel=False,
    ),
    dict(
        # lane/cpu-training-embedding-ivf (2026-09-15): the Embedding
        # layer's host binding. It routes `_mojolearn_embedding` on a
        # CPU-only install with the GPU binding's whole surface
        # (embedding_forward, embedding_backward and the two read-backs),
        # the refusals in its words and order.
        family="embedding",
        binding="_mojolearn_embedding_host",
        routes="_mojolearn_embedding",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("embedding", "embedding-sort"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("Embedding",),
        display="the Embedding layer",
        host_modules=(
            "embedding/host/embedding_host.mojo",
            "embedding/checks/embedding_oracle.mojo",
        ),
        exports=(
            "embedding_host_numeric_mode", "embedding_host_vendor",
            "embedding_host_column", "embedding_host_sabotage",
            "embedding_vendor", "embedding_numeric_mode",
            "embedding_forward", "embedding_backward",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        # lane/inference-embedding-ivf-cholesky (2026-09-15): public CPU
        # lookup in a saved embedding table, with no backward in the binary.
        # The reference embedding family above carries the backward fold and
        # stays out of the wheels; this binding registers embedding_forward
        # from the same source (bindings/embedding_host_forward.mojo) and
        # ships. It serves `_mojolearn_embedding` on a CPU-only install when
        # the reference binding is not built.
        family="embedding_infer",
        binding="_mojolearn_embedding_infer_host",
        routes=None,
        serves=("_mojolearn_embedding",),
        loaded_by="_backend._HOST_INFERENCE_MODULES and python/mojolearn/_classical_host.py",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=("embedding",),
        forest_kinds=(),
        classes=("Embedding",),
        display="Embedding lookup in a saved table",
        host_modules=(
            "embedding/host/embedding_host.mojo", "bindings/embedding_host_forward.mojo",
            "embedding/checks/embedding_oracle.mojo",
        ),
        exports=(
            "embedding_infer_host_numeric_mode", "embedding_infer_host_vendor",
            "embedding_infer_host_column", "embedding_infer_host_sabotage",
            "embedding_vendor", "embedding_numeric_mode", "embedding_forward",
        ),
        gate="tools/classical_host_gate.py and tools/identity_break.py",
        ships_in_wheel=True,
    ),
    dict(
        # lane/cpu-training-embedding-ivf (2026-09-15): IVFIndex's host
        # binding. It routes `_mojolearn_ivf` on a CPU-only install with the
        # GPU binding's whole surface (ivf_flat_build_and_search and the two
        # read-backs): the build's k-means quantizer through the k-means
        # lane's host restatement, the CSR layout and probe merge the device
        # path already runs on the host, and the pinned distance tile and
        # the identical top-k restated.
        family="ivf",
        binding="_mojolearn_ivf_host",
        routes="_mojolearn_ivf",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("ivf", "ivf-euclidean", "ivf-extend"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("IVFIndex",),
        display="the IVF-Flat index",
        host_modules=(
            "ivf/host/ivf_host.mojo",
            "cluster/host/kmeans_oracle.mojo",
            "ivf/checks/list_layout.mojo",
            "ivf/impl/neighbors/ivf_common.mojo",
            "ivf/impl/neighbors/ivf_flat/ivf_flat_index.mojo",
        ),
        exports=(
            "ivf_host_numeric_mode", "ivf_host_vendor", "ivf_host_column",
            "ivf_host_sabotage", "ivf_vendor", "ivf_numeric_mode",
            "ivf_flat_build_and_search", "ivf_flat_build", "ivf_flat_search",
            "ivf_flat_extend",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        # lane/inference-embedding-ivf-cholesky (2026-09-15): public CPU
        # search over a saved, GPU-built IVF-Flat index, with no build in the
        # binary. The reference ivf family above carries the k-means
        # quantizer fit and stays out of the wheels; this binding registers
        # ivf_flat_search from the same source (bindings/ivf_host_search.mojo)
        # and ships. It serves `_mojolearn_ivf` on a CPU-only install when the
        # reference binding is not built.
        family="ivf_search",
        binding="_mojolearn_ivf_search_host",
        routes=None,
        serves=("_mojolearn_ivf",),
        loaded_by="_backend._HOST_INFERENCE_MODULES and python/mojolearn/_classical_host.py",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=("ivf", "ivf-euclidean", "ivf-extend"),
        forest_kinds=(),
        classes=("IVFIndex",),
        display="IVF-Flat search over a saved index and extending it",
        host_modules=(
            "ivf/host/ivf_host.mojo", "bindings/ivf_host_search.mojo",
            "bindings/ivf_index_arrays.mojo",
            "ivf/impl/neighbors/ivf_flat/ivf_flat_index.mojo",
        ),
        exports=(
            "ivf_search_host_numeric_mode", "ivf_search_host_vendor",
            "ivf_search_host_column", "ivf_search_host_sabotage",
            "ivf_vendor", "ivf_numeric_mode", "ivf_flat_search", "ivf_flat_extend",
        ),
        gate="tools/classical_host_gate.py and tools/identity_break.py",
        ships_in_wheel=True,
    ),
    dict(
        # lane/inference-forecast-umap-pca (2026-09-15): public CPU inference
        # for saved ARIMA models, with no fit in the binary. The reference
        # arima family above carries the whole fit and stays out of the
        # wheels; this binding registers arima_predict and arima_forecast
        # from the same source (bindings/arima_host_predict.mojo) and ships.
        # `routes` is None because `_mojolearn_arima` routes to the reference
        # binding when it is built; `serves` names the route this binding
        # takes when it is not (`_backend._HOST_INFERENCE_MODULES`), which is
        # an installed CPU-only wheel.
        family="forecast",
        binding="_mojolearn_forecast_host",
        routes=None,
        serves=("_mojolearn_arima",),
        loaded_by="_backend._HOST_INFERENCE_MODULES and python/mojolearn/_classical_host.py",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=("arima", "arima-011", "arima-seasonal-c"),
        forest_kinds=(),
        classes=("ARIMA",),
        display="batched ARIMA prediction, in sample and out of sample, and forecasts",
        host_modules=("arima/host/arima_oracle.mojo", "bindings/arima_host_predict.mojo"),
        exports=(
            "forecast_host_numeric_mode", "forecast_host_vendor", "forecast_host_column",
            "forecast_host_sabotage", "arima_vendor", "arima_numeric_mode",
            "arima_predict", "arima_forecast",
        ),
        gate="tools/classical_host_gate.py and tools/identity_break.py",
        ships_in_wheel=True,
    ),
    dict(
        family="transformer",
        binding="_mojolearn_transformer_host",
        routes="_mojolearn_transformer",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("transformer", "transformer-window"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("TransformerBlock",),
        display="the Transformer block forward, decode step and backward",
        host_modules=(
            "transformer/host/transformer_block_host.mojo",
            "transformer/checks/transformer_oracle.mojo",
            "transformer/checks/transformer_backward_oracle.mojo",
            "transformer/checks/transformer_fixture.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "transformer_host_numeric_mode", "transformer_host_vendor",
            "transformer_host_column", "transformer_host_sabotage",
            "transformer_vendor", "transformer_numeric_mode",
            "transformer_forward", "transformer_forward_fresh",
            "transformer_decode_step", "transformer_backward",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        # Training-only reference family: source builds for internal bitwise
        # verification, not shipped (docs/lanes/CPU_INFERENCE_BOUNDARY_2026-09-15.md).
        ships_in_wheel=False,
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


def public_reference_lanes():
    """Small explicit verification surface available in an inference wheel.

    Full CPU training verification uses source bindings and covered_lanes().
    These probes need only public inference dependencies, including linalg.
    """
    return ["gemm-pinned", "kde", "ols", "ridge", "knn", "svc", "pca", "cholesky"] + list(PUBLIC_HOST_ONLY_LANES)


#: Public reference lanes that are not CPU training lanes, {lane: family}: a
#: shipped host family with no GPU path to cover (2026-09-15). The tokenizer
#: lane loads the synthetic vocabulary the harness trains itself, so it needs
#: no vocabulary file on the install.
PUBLIC_HOST_ONLY_LANES = {"tokenizer": "tokenizer"}


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


def fix_covered_lanes():
    """The covered lanes the training gate diffs against
    TRAINING_FIX_COLUMNS, in the gate's order. Every one must be covered."""
    covered = covered_lanes()
    unknown = [lane for lane in TRAINING_FIX_LANES if lane not in covered]
    if unknown:
        raise RuntimeError(f"host_surface: TRAINING_FIX_LANES names uncovered lanes {unknown}")
    return [lane for lane in covered if lane in TRAINING_FIX_LANES]


def record_covered_lanes():
    """The covered lanes the training gate diffs against
    TRAINING_GPU_COLUMNS: every covered lane not in TRAINING_FIX_LANES."""
    fixed = fix_covered_lanes()
    return [lane for lane in covered_lanes() if lane not in fixed]


def inference_lanes():
    """The classical gate lanes served from a saved model, in gate order."""
    out = []
    for f in FAMILIES:
        for lane in f["inference_lanes"]:
            if lane not in out:
                out.append(lane)
    return out


def inference_routes():
    """`_MODULES` name -> the inference-only host binding that serves it on
    a CPU-only install when the route's reference binding is not built (the
    `serves` key; lane/inference-forecast-umap-pca, 2026-09-15). A route may
    be served by one such binding, and only a binding that ships."""
    out = {}
    for f in FAMILIES:
        for route in f.get("serves", ()):
            if route in out:
                raise RuntimeError(f"host_surface: {route} is served by {out[route]} and {f['binding']}")
            if not f["ships_in_wheel"]:
                raise RuntimeError(f"host_surface: {f['binding']} serves {route} but does not ship")
            out[route] = f["binding"]
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
    parts = [f.get("inference_display", f["display"]) for f in FAMILIES if f["inference_lanes"] or f["forest_kinds"]]
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
        "| family | binding under `mojolearn/host/` | routes (CPU-only install) | internal CPU reference lanes | predicts on a CPU from a saved model | gate | in a wheel |",
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
        if f["routes"]:
            route = "`" + f["routes"] + "`"
        elif f.get("serves"):
            route = ", ".join("`" + r + "`" for r in f["serves"]) + " when its reference binding is not built"
        else:
            route = "loaded by path"
        rows.append(
            f"| {f['family']} | `{f['binding']}.so` | {route} "
            f"| {trains} | {predicts} | {f['gate']} | {'yes' if f['ships_in_wheel'] else 'no, `' + build_shim(f['family']) + '`'} |"
        )
    return "\n".join(rows)


def as_dict():
    return dict(
        source=SOURCE,
        builder=BUILDER,
        families=[dict(f) for f in FAMILIES],
        routed=routed_modules(),
        inference_routes=inference_routes(),
        covered_lanes=covered_lanes(),
        record_covered_lanes=record_covered_lanes(),
        fix_covered_lanes=fix_covered_lanes(),
        inference_lanes=inference_lanes(),
        forest_kinds=forest_kinds(),
        classical_recorded=list(CLASSICAL_RECORDED),
        forecast_recorded=list(FORECAST_RECORDED),
        inference_only_recorded=list(INFERENCE_ONLY_RECORDED),
        search_lookup_recorded=list(SEARCH_LOOKUP_RECORDED),
        classical_gpu_columns=list(CLASSICAL_GPU_COLUMNS),
        training_gpu_columns=list(TRAINING_GPU_COLUMNS),
        training_fix_columns=list(TRAINING_FIX_COLUMNS),
        training_gpu_column_record=training_gpu_column_record(),
        wheel_families=wheel_families(),
        wheel_bindings=wheel_bindings(),
        forest_recorded_root=FOREST_RECORDED_ROOT,
        no_cpu_path=list(NO_CPU_PATH),
        adapted_modules={k: dict(v) for k, v in ADAPTED_MODULES.items()},
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="the CPU surface manifest; one flag per list")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--families", action="store_true", help="every host family")
    g.add_argument("--routed-families", action="store_true", help="families routed by _backend._HOST_MODULES")
    g.add_argument("--bindings", action="store_true", help="every host binding basename")
    g.add_argument("--routed-bindings", action="store_true", help="the routed families' basenames")
    g.add_argument("--covered-lanes", action="store_true", help="identity_break lanes with a CPU training path (comma separated)")
    g.add_argument("--record-covered-lanes", action="store_true", help="covered lanes diffed against --training-gpu-columns (comma separated)")
    g.add_argument("--fix-covered-lanes", action="store_true", help="covered lanes diffed against --training-fix-columns (comma separated)")
    g.add_argument("--inference-lanes", action="store_true", help="classical gate lanes served from a saved model (comma separated)")
    g.add_argument("--forest-kinds", action="store_true", help="forest gate kinds (comma separated)")
    g.add_argument("--wheel-families", action="store_true", help="families whose host binding ships in the wheels")
    g.add_argument("--wheel-bindings", action="store_true", help="the wheel families' basenames")
    g.add_argument("--training-gpu-column-record", action="store_true", help="the record directory name of the training GPU columns")
    g.add_argument("--classical-recorded", action="store_true", help="classical gate recording directories")
    g.add_argument("--saved-model-recorded", action="store_true", help="the forecast, inference-only and search/lookup recording directories")
    g.add_argument("--classical-gpu-columns", action="store_true", help="the GPU columns the classical gate compares against")
    g.add_argument("--training-gpu-columns", action="store_true", help="the GPU columns the training gate diffs against")
    g.add_argument("--training-fix-columns", action="store_true", help="the GPU columns the training gate diffs --fix-covered-lanes against")
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
    comma = (args.covered_lanes or args.record_covered_lanes or args.fix_covered_lanes
             or args.inference_lanes or args.forest_kinds)
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
    elif args.record_covered_lanes:
        items = record_covered_lanes()
    elif args.fix_covered_lanes:
        items = fix_covered_lanes()
    elif args.training_fix_columns:
        items = list(TRAINING_FIX_COLUMNS)
    elif args.inference_lanes:
        items = inference_lanes()
    elif args.forest_kinds:
        items = forest_kinds()
    elif args.classical_recorded:
        items = list(CLASSICAL_RECORDED)
    elif args.saved_model_recorded:
        items = saved_model_recorded()
    elif args.classical_gpu_columns:
        items = list(CLASSICAL_GPU_COLUMNS)
    else:
        items = list(TRAINING_GPU_COLUMNS)
    print(sep.join(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
