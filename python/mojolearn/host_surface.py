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
whether it ships in a wheel. SINCE 2026-09-16 EVERY FAMILY SHIPS
(lane/ship-cpu-host-families). The "inference only, CPU training internal"
boundary held the sixteen training families back, and the price was the number
that matters: an installed wheel could check 39 of the harness's 211 identity
lanes, because a lane's host binding was not in it. Bitwise reproducibility a
user cannot re-run on their own machine is a claim, not a result. The
objection was size, and size was measured rather than guessed: the sixteen add
7.66 MB uncompressed and 2.19 MB compressed to a wheel whose weight is GPU
bindings (312 MB uncompressed on Linux, 91 files). `ships_in_wheel=False` now
means a deliberate, stated exclusion, and nothing carries one. What did NOT
change is the RUN-TIME boundary: an ordinary `fit` on a CPU-only install still
refuses (`_cpu_reference.py`), and these bindings answer the verifier, which
fits inside `reference_training()`. The two wheel builders and the packer read
`--wheel-families` and `--wheel-bindings` below instead of naming the byte
LM's binding by hand, and packaging/check_ext_lists.py fails when any of
them carries a host list of its own.

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
    python3 python/mojolearn/host_surface.py --wheel-notes
    python3 python/mojolearn/host_surface.py --public-reference-lanes
    python3 python/mojolearn/host_surface.py --public-reference-candidates
    python3 python/mojolearn/host_surface.py --saved-model-inference-owed
    python3 python/mojolearn/host_surface.py --markdown
    python3 python/mojolearn/host_surface.py --json

Every family carries a `wheel_note` saying why it ships
(lane/expose-inference-surface, 2026-09-16, which introduced the field so that
an exclusion could never be silent). Every note now begins "Ships:", and
python/mojolearn/tests/test_host_surface.py fails if one does not: the two
that once read OPEN, `resample` and the `tsa` family's `kpss_test`, were
settled by shipping them, and lane/ship-cpu-host-families then shipped the
rest. A family held back again must say so in its note, which is the only
place the reason belongs.

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
#: CPU column now carries, so the gate at 7bf4f4cc9 failed on all seven
#: runners with "require-columns 4 ... 27 short". The 47-lane
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
    # lane/saved-model-reference-gaps (2026-09-16): dbscan, agglomerative,
    # spectral and spectral-precomputed, the four predicts that shipped on
    # 2026-09-15 with no recording at all, on all nine fixtures each. Recorded
    # on a RunPod NVIDIA A100 (sm_80), which is the first recording in this
    # list taken anywhere but the M4; `check` on the same box read
    # `gate verdict IDENTICAL (36 fixtures, exit 0)` and the predict-only
    # sabotage host set read `EXPECTED MISMATCH SEEN` with `unmoved` EMPTY
    # under --every-fixture, so every one of the 36 cells was watched to fail
    # before it was believed. The Apple and AMD recordings are owed to the
    # next release record; the identity columns behind these cells are
    # bench/results/identity_break/2026-09-16_predict-nvidia.
    "bench/results/classical_host/2026-09-16-nvidia-predict",
    # lane/classical-host-recordings (2026-09-16): the six FITTED k-means
    # lanes, the last entry SAVED_MODEL_INFERENCE_OWED had, on all nine
    # fixtures each. Recorded on a RunPod NVIDIA A100-SXM4-80GB (sm_80) and
    # checked on BOTH host architectures: `gate verdict IDENTICAL (54
    # fixtures, exit 0)` on the box's x86-64 bindings AND on the M4's arm64
    # ones, so a model fitted on an A100 answers predict and transform with
    # the A100's bits on a machine with no GPU. Both sabotage host sets, the
    # family define and MOJOLEARN_KMEANS_PREDICT_SABOTAGE, read `EXPECTED
    # MISMATCH SEEN` with `unmoved` EMPTY under --every-fixture on both
    # architectures. The family define had NO arm on this side of the
    # library until that lane added one, and its --every-fixture arm read
    # `SABOTAGE NOT CAUGHT` on all 54 when it was first rehearsed. The Apple
    # and AMD recordings are owed to the next release record; the identity
    # column behind these cells is
    # bench/results/identity_break/2026-09-16_kmeans-and-spectral-cpu.
    "bench/results/classical_host/2026-09-16-nvidia-kmeans",
    # The AMD column of the same six lanes, same nine fixtures, on a RunPod
    # MI300X (gfx942): `gate verdict IDENTICAL (54 fixtures, exit 0)` and the
    # predict-define sabotage arm caught on all 54 under --every-fixture. The
    # AMD PREDICT recording from the same box is 3 of 36 and is deliberately
    # NOT listed here: its `record` died with hipErrorOutOfMemory in
    # dbscan_fit_core, so it is kept, named partial, at
    # bench/results/classical_host/2026-09-16-amd-predict-partial.
    #
    # WHY IT DIED, ANSWERED BY lane/amd-dbscan-oom THE SAME DAY: not a leak
    # and not the code. A second RunPod MI300X reproduced the same refusal,
    # and its card had 360.4 MiB of 196592.0 MiB FREE before `import
    # mojolearn` ran, with 178.6 GiB held by a kfd process outside the
    # container. A dedicated DigitalOcean MI325X ran the same three DBSCAN
    # lanes at `cells=27 stable=27 moved=0 refused=0` with device memory flat
    # at a 1.5 GiB plateau. The AMD predict column is owed on a card that is
    # ours, and there is nothing to fix first. See
    # docs/lanes/LANE_STATUS_lane-amd-dbscan-oom.md.
    "bench/results/classical_host/2026-09-16-amd-kmeans",
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
    # lane/inference-holtwinters (2026-09-15): holtwinters and
    # holtwinters-multiplicative, recorded on the M4's Metal set.
    "bench/results/classical_host/2026-09-15-apple-m4-holtwinters",
    # lane/arima-exog (2026-09-15): arima-exog and arima-exog-seasonal, the
    # saved mojolearn-arima-2 models (the fit's regressors travel in the
    # file), recorded on the M4's Metal set, nine fixtures each.
    "bench/results/classical_host/2026-09-15-apple-m4-arima-exog",
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
    "kernel-ridge-poly": "kernel ridge poly kernel variant",
    "kernel-ridge-sigmoid": "kernel ridge sigmoid kernel variant",
    "kernel-ridge-laplacian": "kernel ridge laplacian kernel variant",
    "nystroem-poly": "nystroem poly kernel variant",
    "nystroem-sigmoid": "nystroem sigmoid kernel variant",
    "nystroem-laplacian": "nystroem laplacian kernel variant",

    # Logical row shards; pending independent reference qualification.
    "par-rbf-sampler": "the row-sharded random Fourier feature transform",
    "select-d": "ordinary differencing order selection",
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
    # diff IDENTICAL on every covered cell at 2b7f991b6.
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
    # IDENTICAL x4.
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
    # binding. The sabotage arm of these two lanes is DECLARED and has not
    # been observed to fire in any committed column.
    # .github/workflows/cpu-identity-gate.yml builds the sabotage host set and
    # runs them under it on every run, and uploads the result as an artifact
    # that no step commits.
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
    # At bafab59ef all nine training and infer cells of each lane read
    # IDENTICAL x4, and the sabotage build moves every one. That move is
    # carried by committed columns under bench/results/identity_break, so it
    # can be re-checked here rather than in a CI log.
    "gp": "the Gaussian process with an RBF kernel",
    "gp-matern12": "the Gaussian process with a Matern kernel at nu 0.5",
    "gp-matern32": "the Gaussian process with a Matern kernel at nu 1.5",
    "gp-matern52-ard": "the Gaussian process with an ARD Matern kernel at nu 2.5",
    # lane/cpu-training-small-gaps (2026-09-15): normalize_y=True, the folds
    # through the preprocessing host binding's standard_fit and standard_transform.
    "gp-normalize-y": "the Gaussian process with normalized targets",
    "gp-optimize": "Gaussian process hyperparameter optimization",
    "gp-optimize-restarts": "Gaussian process hyperparameter optimization with restarts",
    # Gaussian process classification (lane/gaussian-process-classifier,
    # 2026-09-15): gaussian_process/host/gpc_oracle.mojo over the gp oracle's
    # kernel matrix, the Cholesky oracle and gemm_oracle, the Newton steps in
    # gaussian_process/host/gpc_steps.mojo (the GPU path compiles the same
    # file). No GPU record carries these lanes yet, so their cells are OWED.
    "gpc": "the binary Gaussian process classifier",
    "gpc-multiclass": "the one-vs-rest Gaussian process classifier",
    # lane/unlaned-public-algorithms (2026-09-20): the two
    # `parallel_gaussian_process` entries `tools/verification_matrix.py`
    # reported with NO identity lane at all. Their shards are CLASSES, cut in
    # the driver's own Python and merged there in class order, and each shard
    # is this family's host `gpc_fit`/`gpc_predict` -- the same arithmetic the
    # two lanes above hash -- so `gpc_class_fit` and `gpc_class_predict`
    # joined `_parallel_pool.CPU_OPERATIONS` and these two take a CPU column.
    # `PUBLIC_EXCLUDED_PREFIXES` keeps every `par-*` lane out of the public
    # reference set, so being covered here does not make them public.
    "par-gpc-fit": "the class-sharded Gaussian process classifier fit",
    "par-gpc-predict": "the class-sharded Gaussian process classifier prediction",
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

    # lane/linalg-public (2026-09-19): the three decompositions this tree
    # already computed inside PCA, TruncatedSVD, Nystroem, SpectralClustering
    # and the ARIMA least squares, under their own numpy names.
    "linalg-qr": "the Householder QR's R factor, both slice arms",
    "linalg-eigh": "the symmetric Jacobi eigendecomposition, ascending",
    "linalg-svdvals": "the singular values, descending",
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
    "metrics-homogeneity-completeness": "the combined homogeneity, completeness and V-measure scores",
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
    # binding. The other GBDT lanes refused by name until 2026-09-15. The
    # sabotage arm of the four GBDT lanes is DECLARED and has not been
    # observed to fire in any committed column. The CPU identity gate builds
    # the sabotage host set and runs them under it on every run, and uploads
    # the result as an artifact that no step commits.
    "gbdt-symmetric": "gradient boosting on symmetric trees with the Logloss loss",
    # Workstream E batch 3 (2026-09-14): the same tree with the RMSE loss
    # trains through gbdt/host/gbdt_oracle_rmse.mojo (the seeded cursor and
    # the searcher's own leaves, DEVIATION 64) from the same binding. Its
    # sabotage arm is declared and unobserved, as above.
    "gbdt-rmse": "gradient boosting on symmetric trees with the RMSE loss",
    # Same batch: the Depthwise and Lossguide policies with the Logloss loss
    # train through gbdt/host/gbdt_oracle_depthwise.mojo (the non-symmetric
    # driver) and gbdt/host/gbdt_oracle_lossguide.mojo, in the same binding.
    # Their sabotage arms are declared and unobserved, as above.
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
    "gbdt-border-types": "gradient boosting with the six non-default feature border types",
    "gbdt-ordered": "ordered boosting (boosting_type='Ordered') with the Logloss and RMSE losses",
    "gbdt-ordered-bayesian-noise": "ordered boosting with the Bayesian bootstrap and score noise",
    "gbdt-bfa-quantile": "boost from average on the MAE, Quantile and MAPE losses",
    "gbdt-catboost-defaults": "gradient boosting at CatBoost's GPU defaults (auto learning rate, Bayesian bootstrap, score noise)",
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
    # lane/cpu-training-par-classical (2026-09-15, below). At 422a1b9e5 all 27
    # training and 27 infer cells read IDENTICAL x4, and under the sabotage
    # build 26 of 27 of each move (arima-011/wide keeps its hash). That move
    # is carried by committed columns, not by a CI log.
    "arima": "ARIMA",
    "arima-011": "differenced ARIMA",
    "arima-seasonal-c": "seasonal ARIMA",
    # lane/arima-exog (2026-09-15): regression with ARIMA errors, the
    # regressors restated in arima/host/arima_oracle.mojo beside the lane.
    "arima-exog": "ARIMA with exogenous regressors",
    "arima-exog-seasonal": "differenced seasonal ARIMA with exogenous regressors",
    # The umap host lane (lane/cpu-training-umap-b, 2026-09-14): UMAP fits
    # and transforms through umap/host/umap_oracle.mojo, exported under the
    # GPU binding's names from the metrics host binding. The fit's optimizer
    # is the IDENTICAL DEVICE epoch fold (kernel-matrix row
    # umap_device_optimizer_for) restated vertex by vertex, not the serial
    # host loop, which produces different bits. At 5988700d9 (the 136-lane
    # record) all nine train and nine infer cells read IDENTICAL x4, and the
    # sabotage build moves all eighteen; the 166-lane record carries the same
    # umap hashes. That move is carried by committed columns, not by a CI log.
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
    "ordered-gradient-sum": "the ordered shard gradient reduction",
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
    # lane/laneless-public-classes (2026-09-19): LanguageModelConfig, the
    # public name of ByteLanguageModelConfig, at two shapes no other lane
    # runs. The forward is LanguageModelInference's, this family's host
    # binding, so the family's route is the lane's route; the profile string
    # it hashes comes back from the same binding through
    # _byte_lm_trainer_host.py's byte_lm_config_profile. Host-only, exactly
    # as byte-lm-host-infer is: tools/lane_applicability.py calls it
    # `cpu-host-route-only`, so a GPU column measures that box's CPU and the
    # DEVICE axis is degenerate there. No record carries it yet, and it is
    # held out of public_reference_lanes() until one does.
    "language-model-config": "the byte LM shape object at two non-default shapes",
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
    "par-scaler-minmax": "the column-sharded min-max scaler",
    "par-arima": "series-sharded ARIMA",
    "par-holtwinters": "series-sharded Holt-Winters",
    # lane/lm-attention-fallback (2026-09-19): the four
    # `parallel_forecasting` drivers, the PREDICTION half of the same series
    # partition. They send one worker operation, `forecast_predict`, whose
    # split is `parallel_forecasting._ranges` in Python and whose merge is
    # the driver's own ordered memcopy; nothing is split inside a binding
    # (the worker runs the bare `state.predict`, and arima/ and holtwinters/
    # name no device count and no multi_gpu module), so
    # `_parallel_pool.CPU_OPERATIONS` admits it at any device count and the
    # shard's own predict and forecast are this family's and the tsa
    # family's host bindings. Before that all four entries refused by name
    # on every CPU install and had no lane at all.
    #
    # MEASURED 2026-09-19 on the M4's CPU column, one core, at --repeats 2,
    # against arima and tsa host bindings built from this source in the same
    # session: both lanes STABLE on all four columns (train, infer, model,
    # batch); every one of those eight cells MOVED under
    # MOJOLEARN_HOST_SABOTAGE=1 rebuilds of those two families, so the lanes
    # reach the host arithmetic and are not hashing Python; the batch part
    # read BATCH_MOVED on both under MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1; and
    # five deliberate breaks of the DRIVER (a shard handed the series one to
    # its left, the merge pairing reversed, and a shard dropped) made both
    # cells REFUSE -- three of them through the in-cell oracle by byte count
    # ("predict_arima(0, n_obs) and plain ARIMA.predict(0, n_obs) differ:
    # 1836 bytes of 8192"), two through the drivers' own shape guard.
    "par-forecast-arima": "the series-sharded ARIMA prediction and forecast drivers",
    "par-forecast-holtwinters": "the series-sharded Holt-Winters prediction and forecast drivers",
    # Wave 2 (lane/cpu-training-par-wave2, 2026-09-15): the neighbor
    # drivers. ParallelQueries cuts query rows in Python (four shards of 16
    # rows); ReferenceShardedNeighbors cuts the reference into four shards
    # of 1024 rows, merges the shard candidates by composite key in Python
    # and sends one vote request, served on CPU by the core host binding's
    # knn_classify_neighbors and knn_regress_neighbors.
    "par-queries-knn": "query-sharded k-NN classification",
    "par-queries-nn": "query-sharded nearest-neighbor distances and indices",
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
    "par-forest-et-clf": "the tree-range-sharded Extra Trees classifier",
    "par-forest-reg": "the tree-range-sharded random forest regressor",
    # Wave 2, par-mlp: ParallelNeuralTrainer sends one mlp_gradient request
    # per logical shard (three of 64 rows) and one mlp_update that folds
    # them in shard order in Python and steps the optimizer, on the
    # training host binding. The update pool is cooperative and is admitted
    # on CPU only at one device, where the GPU binding's range split is the
    # plain path; two devices refuse by name.
    "par-mlp": "the small MLP trained over ordered logical gradient shards",
    # Wave 3 (lane/cpu-verifier-par-samba, 2026-09-16): the same driver over
    # the Samba stack. ParallelNeuralTrainer sends one samba_gradient request
    # per logical shard (two windows of (2, 17)) from the non-cooperative
    # pool, and one samba_update that folds them in shard order with
    # ordered_sum_gradients and steps the optimizer, on the training host
    # binding with the mamba and transformer families' blocks. The clip lane
    # adds the global norm (max_norm=0.5), the arithmetic the covered
    # samba-untied-dropout-accum lane checks. One device only, as par-mlp.
    "par-samba": "the Samba stack trained over ordered logical gradient shards",
    "par-samba-clip": "the Samba stack trained over ordered logical gradient shards under a global norm clip",
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
    # lane/laneless-public-classes (2026-09-19): DistributedIVFIndex, the
    # disjoint-shard driver over a built index. Its workers call the ivf
    # binding's `ivf_flat_partial_search` and `ivf_finalize_distances`,
    # which the host bindings now register from bindings/ivf_host_search.mojo
    # over `host_ivf_search`'s partial_storage arms, and which
    # `_parallel_pool.CPU_OPERATIONS` admits, so the driver's Python
    # partition, its local-id maps and its global merge run unchanged on a
    # CPU. Being covered means it HAS a CPU route, not that a CPU column
    # runs it: it is a multi-device driver, so tools/lane_applicability.py
    # refuses it on a zero-device column and identity_break's
    # RECORD_EXCLUDED_PREFIXES keeps every par-* lane out of a release
    # record. Its cells come from the CPU identity gate and from the
    # two-device par legs, as the other par-* lanes' do.
    "par-ivf": "the shard-distributed IVF-Flat index",
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
    # lane/cpu-verifier-gaps-7 (2026-09-15): seven one-device lanes whose CPU
    # host functions had merged with their own lanes, and which the gate now
    # runs. GaussianMixture.sample through the mixture host binding's
    # gmm_sample (mixture/checks/sample.mojo) on the gmm lanes' host fits;
    # GaussianProcessRegressor.sample_y through the gp host binding's
    # gpr_sample_y (gaussian_process/checks/sample_y.mojo), the normalized
    # arm over the preprocessing binding's folds. No committed GPU record
    # carries them, so their cells are OWED against the record.
    "gmm-sample": "samples from the Gaussian mixture",
    "gmm-random-init-sample": "samples from the Gaussian mixture with a random start",
    "gp-sample-y": "posterior draws from the Gaussian process",
    "gp-sample-y-normalize": "posterior draws from the Gaussian process with normalized targets",
    # The tokenizer lane: host integers and tables through the tokenizer
    # binding, the synthetic vocabulary at identity_break's
    # LANE_REVISIONS["tokenizer"], so the record's older cells read as
    # absent and every part is OWED. The gate's sabotage set builds this
    # binding with its own define (GATE_SABOTAGE_OWN_DEFINES).
    "tokenizer": "the byte-level BPE tokenizer (inference, host integers)",
    # THE THREE THE TOKENIZER FAMILY ALREADY SERVED AND NEVER DECLARED
    # (2026-09-19, lane/laneless-public-classes). lane/bpe-builder-native
    # (2026-09-18) put `bpe_train`, `bpe_trained_sizes` and
    # `bpe_trained_copy` in the tokenizer binding and named them in this
    # family's `exports` and `host_modules`, and lane/tokenized-corpus added
    # the three lanes that reach them, but the hand-written `training_lanes`
    # tuple stayed at `("tokenizer",)`. Because tools/lane_applicability.py
    # derives `has_cpu_route` from `covered_lanes()`, all three read
    # DEGENERATE and tools/verify_lanes.py REFUSED them on the CPU column
    # while the binding was built, loaded and producing hashes: a lane that
    # cannot be run is indistinguishable in a total from one that passes.
    # Measured on the M4 against the prebuilt host set at the fix:
    # bpe-trainer, bpe-vocabulary and tokenized-corpus each STABLE.
    # No GPU column can carry them -- vocabulary training has no GPU path in
    # any library -- which is why they are also PUBLIC_HOST_ONLY_LANES; being
    # covered is about having a CPU route, not about owing a GPU one.
    "bpe-trainer": "byte-level BPE vocabulary training",
    "bpe-vocabulary": "a trained BPE vocabulary written, loaded back and used",
    "tokenized-corpus": "a corpus tokenized once, cached and read back as batches",
    # THE `mojolearn.models` NAMESPACE, which had no lane of any kind
    # (lane/models-namespace-lanes, 2026-09-19). tools/verification_matrix.py
    # reported fourteen public entries with no lane and every one of them was
    # the Hugging Face loading path; what stood in for a lane was
    # python/mojolearn/tests/test_models_loader.py, which says the loader
    # agrees with itself on one box and nothing about two boxes agreeing on a
    # byte. Three lanes, cut where the code paths cut, each declared by the
    # family whose binding computes its CPU cell:
    #   hf-checkpoint  linalg   (widen_bf16 -> from_bf16; see the lane)
    #   hf-tokenizer   tokenizer (the GPT-2 door, BpeTokenizer.from_token_bytes)
    #   hf-causal-lm   neural   (_CpuPrimitives and the *Inference blocks)
    # No GPU record carries them yet, so their cells are OWED against the
    # record until the next one is taken.
    "hf-checkpoint": "the Hugging Face checkpoint reader and the option matrix",
    "hf-tokenizer": "the Hugging Face byte-level BPE tokenizer (three pre-tokenization patterns)",
    "hf-causal-lm": "a Hugging Face causal language model loaded and run",
    # The CTR table lanes: CPU TRAINING of CTR tables refuses by name
    # (NO_CPU_PATH), so the CPU column LOADS the Metal-saved model of each
    # fixture from GBDT_CTR_MODELS_DIR and predicts through the forest host
    # binding's CTR and tensor CTR step (HostGBDT). What the CPU column
    # computes is the prediction from a GPU-fitted model, not a fit.
    "gbdt-categorical-ctr-tables": "predictions of Metal-saved gradient boosting models with CTR tables (inference)",
    "gbdt-tensor-ctr-tables": "predictions of Metal-saved gradient boosting models with tensor CTRs (inference)",
    # lane/identical-lowbit-inference (2026-09-17): the two low-bit GEMM
    # profiles, and every neural block with bf16- or int8-stored projection
    # weights materialized exactly and run through the fp32 path.
    "gemm-bf16": "the bf16-storage GEMM profile",
    "gemm-int8": "the int8 GEMM profile with power-of-two scales",
    "transformer-bf16w": "the Transformer block with bf16-stored weights",
    "transformer-int8w": "the Transformer block with int8-stored weights",
    "mamba1-bf16w": "the Mamba-1 block with bf16-stored weights",
    "mamba1-int8w": "the Mamba-1 block with int8-stored weights",
    "mamba2-bf16w": "the Mamba-2 block with bf16-stored weights",
    "mamba2-int8w": "the Mamba-2 block with int8-stored weights",
    "mamba3-bf16w": "the Mamba-3 block with bf16-stored weights",
    "mamba3-int8w": "the Mamba-3 block with int8-stored weights",
    "mlp-bf16w": "the small MLP with bf16-stored weights",
    "mlp-int8w": "the small MLP with int8-stored weights",
    "samba-bf16w": "the Samba stack with bf16-stored weights",
    "samba-int8w": "the Samba stack with int8-stored weights",
    # lane/laneless-public-classes (2026-09-19): three public surfaces that
    # tools/verification_matrix.py reported with NO identity lane at all.
    # Each is a CPU route already in the tree, so the missing thing was the
    # lane and not the implementation.
    #
    # `saved-model-host-infer` is HostForest, HostGBDT, host_predict and
    # host_predict_proba. tools/forest_host_gate.py already holds
    # host_model's two entries to 24 committed GPU recordings and it passes;
    # what it does not reach is the two one-call entries and the
    # parallel_groves HOST engine, which that gate still refuses by name
    # ("the host engine is sequential") although core/forest_host_groves.mojo
    # landed on lane/forest-groves-cpu-and-speed. It moves under the forest
    # family's OWN define, not the generic one.
    #
    # `lowbit-conversions` is the four conversion seams of contract L-1..L-6.
    # They had NO sabotage arm before this lane: MOJOLEARN_HOST_SABOTAGE
    # reaches gemm_oracle's leaf and stops. GATE_SABOTAGE_OWN_DEFINES now
    # names MOJOLEARN_LOWBIT_CONVERT_SABOTAGE for linalg.
    #
    # `grad-accumulation` is clause 9.2's balanced tree and its alignment
    # predicate, which only the opt-in --batch-grad part had ever reached.
    "saved-model-host-infer": "saved forest and gradient boosting models predicted on the CPU (inference)",
    "lowbit-conversions": "the bf16 and int8 weight-storage conversions",
    "grad-accumulation": "gradient accumulation across microbatches",
}

#: The saved models the CTR table lanes load on a CPU column, one
#: `<lane>.<fixture>.npz` per lane and fixture, each written by the Apple M4
#: Metal column (identity_break's GBDT_CTR_MODELS_ENV, repeat 0): base, ties
#: and odd by lane/inference-gbdt-ctr-tables at 1386833b4, the other six
#: fixtures by lane/cpu-verifier-gaps-7 from that lane's same Metal build.
#: The CPU identity gate exports it as MOJOLEARN_IDENTITY_GBDT_CTR_MODELS.
GBDT_CTR_MODELS_DIR = "bench/results/identity_break/2026-09-15_gbdt-ctr-tables/models"
GBDT_CTR_MODEL_LANES = ("gbdt-categorical-ctr-tables", "gbdt-tensor-ctr-tables")

#: Families the CPU identity gate's sabotage host set builds with defines of
#: their own BESIDE -D MOJOLEARN_HOST_SABOTAGE=1 (lane/cpu-verifier-gaps-7,
#: 2026-09-15), because MOJOLEARN_HOST_SABOTAGE reaches nothing in them and a
#: covered lane rests on them. The tokenizer's reverses every encoded
#: document's ids (the tokenizer lane's train, infer and batch parts); the
#: forest's CTR arm rotates every CTR table's counts by one category (the CTR
#: table lanes' train, infer and batch parts), which leaves every other
#: forest and GBDT prediction alone. byte_lm keeps building clean here.
#:
#: A FAMILY MAY NEED MORE THAN ONE, and the tokenizer does since
#: lane/laneless-public-classes (2026-09-19) made `bpe-trainer` a covered
#: lane. That lane trains a vocabulary and hashes the two files it renders;
#: it never encodes, so the ENCODER arm above cannot reach it and under
#: -D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1 alone its cell read
#: 6ed8b49585df3d85 -- the clean value, every part unmoved (measured on the
#: M4, base fixture, --repeats 2). The trainer's own arm,
#: MOJOLEARN_BPE_TRAINER_SABOTAGE, reverses the merge tie-break and is what
#: moves it, so it is listed here too and the gate's set now carries both.
#: A negative control that leaves a covered lane where it found it is not a
#: negative control for that lane.
#:
#: THE LINALG CONVERSION ARM, the same shape as the tokenizer's trainer arm
#: and for the same reason (lane/laneless-public-classes, 2026-09-19). The
#: linalg family's own define is MOJOLEARN_HOST_SABOTAGE, and through
#: `gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE` it moves the GEMM
#: leaf and `gemm_int8_oracle`'s dequantized cell. It reaches NOTHING in
#: `widen_bf16`, `narrow_bf16`, `quantize_rows_int8` or
#: `dequantize_rows_int8`, which are the whole of contract clauses L-1
#: through L-6 and every byte the `lowbit-conversions` lane hashes: measured
#: on the M4, that cell read UNMOVED under the family define alone.
#: MOJOLEARN_LOWBIT_CONVERT_SABOTAGE is those four seams' own arm and the
#: gate's set now carries it too.
GATE_SABOTAGE_OWN_DEFINES = {
    # lane/catboost-parity (2026-09-19): the non-default border types' own
    # arm (drops `select_borders`' middle border). MOJOLEARN_HOST_SABOTAGE
    # already moves gbdt-border-types through the Newton walker; this one
    # reaches the border branch itself, and moves no GreedyLogSum lane
    # (gbdt-symmetric reads its clean hash under it, measured on the M4).
    # And Ordered boosting's own arm (every fold estimated on its whole fold,
    # the look-ahead Ordered prevents): it moves every column of gbdt-ordered
    # and gbdt-ordered-bayesian-noise on the CPU column AND on Metal, to the
    # same hashes (1053bc113326b3cb and ae3714a675da4713 on base), and leaves
    # gbdt-symmetric and gbdt-ordered-rmse at their clean hashes.
    # And the quantile constant's arm (their delta adjust skipped): it moves
    # gbdt-bfa-quantile only.
    "gbdt": ("MOJOLEARN_BORDER_TYPES_SABOTAGE", "MOJOLEARN_ORDERED_SABOTAGE",
             "MOJOLEARN_SAMPLE_QUANTILE_SABOTAGE"),
    "forest": ("MOJOLEARN_GBDT_CTR_HOST_SABOTAGE",),
    "linalg": ("MOJOLEARN_LOWBIT_CONVERT_SABOTAGE",),
    "tokenizer": ("MOJOLEARN_TOKENIZER_HOST_SABOTAGE", "MOJOLEARN_BPE_TRAINER_SABOTAGE"),
}


def sabotage_build_defines(name):
    """The MOJOLEARN_BUILD_EXTRA_DEFINES of family `name` in the CPU identity
    gate's sabotage host set."""
    family(name)
    defines = ["-D MOJOLEARN_HOST_SABOTAGE=1"]
    for own in GATE_SABOTAGE_OWN_DEFINES.get(name, ()):
        defines.append(f"-D {own}=1")
    return " ".join(defines)

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
                        "byte-lm", "byte-lm-resident", "language-model-config"),
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
            # Greedy next bytes, the decode entry a9d933e4f added while
            # skipping discarded decode logits. It shipped as an EXPORT
            # without a manifest row, which is the exact drift
            # test_binding_exports_exactly_the_manifest exists to catch:
            # a function a user can call that the surface does not declare.
            "byte_lm_host_next",
        ),
        gate=".github/workflows/byte-lm-cpu-gate.yml and tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note=(
            "Ships: the byte LM's two public CPU surfaces, LanguageModelInference (forward from a "
            "checkpoint) and the published LanguageModelHostTrainer, the one training entry the "
            "inference boundary keeps public."
        ),
        ships_in_wheel=True,
    ),
    dict(
        family="forest",
        binding="_mojolearn_forest_host",
        routes=None,
        loaded_by="python/mojolearn/_forest_host.py, python/mojolearn/_gbdt_host.py",
        sabotage_define="MOJOLEARN_FOREST_HOST_SABOTAGE",
        # The CTR table lanes (lane/cpu-verifier-gaps-7, 2026-09-15): their
        # CPU cells are this binding's predictions from the Metal-saved
        # models under GBDT_CTR_MODELS_DIR (no CPU fit; see
        # TRAINING_LANE_NAMES), moved in the gate by the CTR arm
        # (GATE_SABOTAGE_OWN_DEFINES).
        training_lanes=GBDT_CTR_MODEL_LANES + ("saved-model-host-infer",),
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
        host_modules=("core/forest_host_predict.mojo", "core/forest_host_groves.mojo",
                      "core/gbdt_host_predict.mojo",
                      "core/gbdt_host_ctr.mojo", "gbdt/models/tensor_ctr_apply.mojo"),
        exports=(
            "forest_host_numeric_mode", "forest_host_vendor", "forest_host_column",
            "forest_host_sabotage", "forest_host_rf_predict_proba",
            "forest_host_rf_predict_reg", "forest_host_et_predict",
            # lane/forest-groves-cpu-and-speed (2026-09-17): the parallel_groves
            # engine over core/forest_host_groves.mojo, and DEVIATION 2961's
            # association sabotage read-back
            "forest_host_groves_sabotage", "forest_host_groves_prepare",
            "forest_host_groves_predict", "forest_host_groves_release",
            "forest_host_gbdt_predict", "forest_host_gbdt_sigmoid", "forest_host_gbdt_sigmoid_pair",
            "forest_host_gbdt_expand_ctr", "forest_host_gbdt_ctr_sabotage",
            "all_finite_f32", "all_finite_f64", "cast_f64_to_f32",
            "argmax_rows_f32", "argmax_rows_f64", "gather_i64", "gather_f64",
        ),
        gate="tools/forest_host_gate.py (.github/workflows/forest-host-gate.yml)",
        wheel_note=(
            "Ships: predicts on a CPU from saved random forest, Extra Trees and gradient boosting "
            "models (forest_kinds), and serves the two CTR table lanes, whose CPU cells are "
            "predictions from GPU-saved models."
        ),
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
        # A covered lane since lane/cpu-verifier-gaps-7 (2026-09-15). It was
        # kept out (lane/cpu-training-host-only-lanes) because the gate's
        # sabotage set held no tokenizer binding (the lane read REFUSED
        # there) and MOJOLEARN_HOST_SABOTAGE reaches nothing in it: the
        # binding holds integers and tables with no float fold. The gate now
        # builds every family into that set and this one with its own define
        # (GATE_SABOTAGE_OWN_DEFINES), which reverses the ids of bpe_encode
        # and of every document of bpe_encode_batch, so the train, infer and
        # batch parts move.
        # mojolearn ships no vocabulary (2026-09-15): the lane loads the
        # synthetic one (python/mojolearn/_tokenizer_synthetic.py) at
        # identity_break's LANE_REVISIONS["tokenizer"], so the records above
        # hashed older input and its cells are owed to the next record.
        # bpe_encode_batch (lane/inference-tokenizer-neural, 2026-09-15)
        # has its own negative control, -D MOJOLEARN_TOKENIZER_BATCH_SABOTAGE=1,
        # which the lane's batch part reads BATCH_MOVED.
        # bpe-trainer, bpe-vocabulary and tokenized-corpus joined on
        # 2026-09-19 (lane/laneless-public-classes); TRAINING_LANE_NAMES
        # carries why they were missing. It takes BOTH of this family's own
        # defines to move all four: MOJOLEARN_TOKENIZER_HOST_SABOTAGE is the
        # ENCODER's arm and bpe-trainer never encodes, so under it alone that
        # lane read 6ed8b49585df3d85, every part unmoved, on the M4 at the
        # fix. MOJOLEARN_BPE_TRAINER_SABOTAGE is the trainer's, which is why
        # GATE_SABOTAGE_OWN_DEFINES now names both for this family.
        # `hf-tokenizer` joined on 2026-09-19 (lane/models-namespace-lanes).
        # `mojolearn.models.Tokenizer` reads a Hugging Face `tokenizer.json`
        # and makes the pre-tokenization PATTERN a parameter (DEVIATION
        # 2960); for the `gpt2` pattern it encodes through this binding's
        # compiled door (`BpeTokenizer.from_token_bytes`), and for `llama3`
        # and `qwen2` through the Python cut and the Python merge, which have
        # no binding at all. So the ENCODER arm moves the GPT-2 parts, the
        # TRAINER arm moves every id part (it moves the vocabulary the lane
        # trains on the fixture), and the Python patterns' parts move under
        # neither -- what guards those is the recorded hash.
        training_lanes=("tokenizer", "bpe-trainer", "bpe-vocabulary", "tokenized-corpus",
                        "hf-tokenizer"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("BpeTokenizer",),
        display="the byte-level BPE tokenizer (GPT-2 format, user-supplied vocabulary)",
        host_modules=(
            "tokenizer/encoding.mojo", "tokenizer/impl/bpe.mojo",
            "tokenizer/impl/pretokenize.mojo", "tokenizer/impl/ranks.mojo",
            "tokenizer/impl/unicode_class.mojo", "tokenizer/impl/byte_unicode.mojo",
            # lane/bpe-builder-native (2026-09-18): the vocabulary TRAINER is
            # compiled in too (bpe_train, bpe_trained_sizes, bpe_trained_copy),
            # the default backend of BpeVocabularyTrainer. Its own negative
            # control is -D MOJOLEARN_BPE_TRAINER_SABOTAGE=1 (reversed
            # tie-break), which tokenizer_host_sabotage() also reads True for.
            "tokenizer/train/bpe_train.mojo",
        ),
        exports=(
            "tokenizer_host_numeric_mode", "tokenizer_host_vendor",
            "tokenizer_host_column", "tokenizer_host_sabotage", "bpe_load",
            "bpe_n_vocab", "bpe_max_token_bytes", "bpe_encode", "bpe_encode_batch",
            "bpe_decode", "bpe_train", "bpe_trained_sizes", "bpe_trained_copy",
        ),
        gate="pixi run check-tokenizer and python/mojolearn/tests/test_tokenizer_surface.py",
        wheel_note=(
            "Ships: encode and decode are inference over a user-supplied vocabulary, and there is no "
            "GPU binding to route from at all."
        ),
        ships_in_wheel=True,
    ),
    dict(
        # lane/inference-tokenizer-neural, 2026-09-15. The INFERENCE half of
        # the training and transformer families (which stay source reference
        # builds): the small MLP's logits and the Transformer and Mamba
        # blocks' forward, forward only, loaded by path and shipped. No
        # optimizer, loss or backward export, so none of that is compiled
        # in. The mlp, transformer and transformer-window lanes' held-out
        # and batch cells run through MLPInference and
        # TransformerBlockInference on a CPU column; their training rows
        # stay the training and transformer families' covered lanes.
        # lane/stateful-cpu-decoding (2026-09-16) added the state-carrying
        # entries, so a decode cache IS exported here now: the same host
        # functions the fresh entries call, handed the caller's state, which
        # is what makes a step-by-step decode bitwise the one-shot prefill.
        family="neural",
        binding="_mojolearn_neural_host",
        routes=None,
        loaded_by="python/mojolearn/neural_inference.py",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        # THIS FAMILY'S FIRST COVERED LANE (lane/models-namespace-lanes,
        # 2026-09-19). It served the held-out and batch cells of `mlp`,
        # `transformer` and the Mamba lanes, whose TRAINING rows belong to
        # the training, transformer and mamba families, so it declared none
        # of its own. `hf-causal-lm` is different: `models.CausalLM` with the
        # public default `device="auto"` assembles
        # `TransformerBlockInference`/`Mamba1BlockInference` and
        # `_CpuPrimitives`'s embedding, rms_norm and linear over THIS
        # binding on a CPU column, and nothing else computes that cell. The
        # generic -D MOJOLEARN_HOST_SABOTAGE=1 (gemm_oracle's descending
        # leaf) reaches it, so no own define is needed.
        training_lanes=("hf-causal-lm",),
        inference_lanes=(),
        forest_kinds=(),
        # lane/inference-neural-forward, 2026-09-15: the Mamba-1, Mamba-2
        # and Mamba-3 zero-state forwards and the Samba stack's embedding,
        # final norm and head joined, forward only; the mamba family stays a
        # source reference build.
        classes=("MLPInference", "TransformerBlockInference", "Mamba1BlockInference",
                 "Mamba2BlockInference", "Mamba3BlockInference", "SambaInference"),
        display="the small MLP's logits, the Transformer and Mamba blocks' forward and decode step and the Samba stack's logits (inference only)",
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
            "transformer_forward", "transformer_decode_step",
            "mamba1_forward_fresh", "mamba1_forward", "mamba1_decode_step",
            "mamba2_forward_fresh", "mamba2_forward", "mamba2_decode_step",
            "mamba3_forward_fresh", "mamba3_forward", "mamba3_decode_step",
            "embedding_forward", "rms_norm_forward", "linear_forward",
        ),
        gate="python/mojolearn/tests/test_neural_inference.py, tools/step_vs_full_check.py and tools/identity_break.py (mlp, transformer, transformer-window, transformer-decode, mamba1, mamba2, mamba3, mamba1-decode, mamba2-decode, mamba3-decode, mamba2-dtlimit, samba, samba-decode, samba-untied-dropout-accum)",
        wheel_note=(
            "Ships: forward-only. Carries the MLP, Transformer, Mamba-1/2/3 and Samba forwards of the "
            "training-only mamba, transformer and training families, their decode steps and caches "
            "(lane/stateful-cpu-decoding), and registers no backward, optimizer or loss symbol."
        ),
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
            "kmeans-sqrt", "kmeans-classic-pp",
            "par-queries-knn", "par-queries-nn", "par-queries-radius", "par-reference-knn",
            "par-reference-knn-reg",
        ),
        # The neighbors and density inference lane (2026-09-15) adds every
        # k-NN metric, the ball cover, the distance-weighted vote and mean
        # and RadiusNeighbors on its four metrics, all from a saved model.
        # lane/classical-host-recordings (2026-09-16): the six FITTED k-means
        # lanes, from a saved `mojolearn-kmeans-1` file through `HostKMeans`
        # on this binding (`kmeans_predict` and `kmeans_transform`, the
        # arithmetic of cluster/host/kmeans_oracle.mojo). One format carries
        # every metric and every start, and since 2026-09-18
        # (lane/kmeans-cosine-capability) the SIX fitted k-means lanes are
        # the whole set: the seventh, `kmeans-cosine`, was a refusal with no
        # model to save, and the metric behind it was deleted.
        inference_lanes=("knn", "knn-clf", "knn-reg", "knn-sqeuclidean", "knn-manhattan",
                         "knn-chebyshev", "knn-cosine", "knn-minkowski-p3", "knn-rbc",
                         "knn-clf-distance", "knn-reg-distance", "radius", "radius-manhattan",
                         "radius-chebyshev", "radius-minkowski-p3",
                         "kmeans", "kmeans-random", "kmeans-array", "kmeans-weighted",
                         "kmeans-sqrt", "kmeans-classic-pp"),
        forest_kinds=(),
        classes=(
            "NearestNeighbors", "KNeighborsClassifier", "KNeighborsRegressor", "KMeans",
            "RadiusNeighbors",
        ),
        display="nearest neighbors on every metric and the ball cover, k-NN classification and k-NN regression with either weighting, radius neighbors and k-means assignment and distances",
        host_modules=(
            "core/knn_host_predict.mojo", "bindings/host_helpers.mojo",
            "cluster/host/kmeans_oracle.mojo", "bindings/hotpath_helpers.mojo",
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
            # lane/python-hotpath (2026-09-17, DEVIATIONS 3100-3104): the helpers
            # of bindings/hotpath_helpers.mojo that stand in for per-row Python,
            # and the ORDER RULE's label encoder (DEVIATION 2500) a CPU-only
            # install used to run as a Python loop.
            "cast_elements", "reduce_stat", "equal_elements",
            "encode_labels_f32", "encode_labels_f64", "encode_labels_i32",
            "encode_labels_i64", "encode_labels_u32", "encode_labels_u8",
            "gather_i32", "check_indices_i64", "indices_overlap_i64",
            "fold_ids", "select_fold_i64",
        ),
        gate="tools/classical_host_gate.py (cpu-identity-gate.yml)",
        wheel_note=(
            "Ships: k-NN, radius-neighbor and k-means inference from a saved model, twenty-one "
            "inference lanes."
        ),
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
        # lane/linalg-public (2026-09-19): qr, eigh and svdvals. They take
        # the HOST route on every box, a GPU box included -- there is no
        # one-shot device door for these kernels, they are reached through an
        # estimator that owns a DeviceContext -- so this family IS their
        # route, not their fallback.
        training_lanes=("gemm-pinned", "gemm-transposed", "cholesky", "gemm-bf16", "gemm-int8",
                        "linalg-qr", "linalg-eigh", "linalg-svdvals",
                        # lane/laneless-public-classes (2026-09-19): the
                        # conversion seams themselves, under their own arm
                        "lowbit-conversions",
                        # lane/models-namespace-lanes (2026-09-19): the
                        # safetensors reader and the option matrix. The ONE
                        # native call in that lane is `lowbit.widen_bf16`,
                        # which is this binding's `from_bf16`, so this family
                        # is where it belongs and the arm that moves it is
                        # this family's own MOJOLEARN_LOWBIT_CONVERT_SABOTAGE
                        # and not MOJOLEARN_HOST_SABOTAGE, for the reason
                        # GATE_SABOTAGE_OWN_DEFINES states above.
                        "hf-checkpoint"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("linalg.gemm", "linalg.gemv", "Cholesky",
                 "linalg.qr", "linalg.eigh", "linalg.svdvals"),
        display="pinned GEMM, the Cholesky factorization and solve, and the QR, symmetric eigen and singular-value decompositions",
        host_modules=("gemm/host/gemm_oracle.mojo", "gemm/host/gemm_lowbit_oracle.mojo",
                      "cholesky/host/chol_oracle.mojo",
                      "decomposition/host/linalg_public.mojo"),
        exports=(
            "linalg_host_numeric_mode", "linalg_host_vendor", "linalg_host_column",
            "linalg_host_sabotage", "linalg_vendor", "linalg_numeric_mode",
            "linalg_profile_version", "gemm", "cholesky_profile_jitter",
            "cholesky_factor", "cholesky_solve",
            # lane/identical-lowbit-inference (2026-09-17): the bf16f32.v1 and
            # int8i32.v1 profiles, gemm/IDENTICAL_LOWBIT_CONTRACT.md.
            "lowbit_profile_version", "gemm_bf16", "gemm_int8", "quantize_int8",
            "dequantize_int8", "to_bf16", "from_bf16",
            # lane/linalg-public (2026-09-19)
            "qr_r", "eigh", "svdvals",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note=(
            "Ships: gemm and gemv, and the saved Cholesky factor's solve. The public reference lanes "
            "gemm-pinned and cholesky need it on an inference wheel."
        ),
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
            "kernel-ridge-poly", "kernel-ridge-sigmoid", "kernel-ridge-laplacian",
            "nystroem-poly", "nystroem-sigmoid", "nystroem-laplacian",
            "kde-tophat-sqeuclidean", "kde-epanechnikov-l1", "kde-exponential-chebyshev",
            "kde-linear-cosine", "kde-cosine-minkowski", "kde-weighted",
            # lane/saved-model-reference-gaps (2026-09-16): DBSCAN.predict and
            # AgglomerativeClustering.predict from a saved model, both through
            # `labeled_reference_predict` on this binding (DEVIATION 2740).
            "dbscan", "agglomerative",
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
        wheel_note=(
            "Ships: the largest saved-model inference surface, thirty-two lanes (the linear, "
            "logistic, decomposition, density, scaler, coordinate-descent and kernel-method classes)."
        ),
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
                        "metrics-fowlkes-mallows", "metrics-homogeneity-completeness"),
        # UMAP.transform from a saved embedding (lane/inference-forecast-
        # umap-pca, 2026-09-15). Its answer depended on the query batch by the
        # transform's contract until lane/umap-batch-fix (2026-09-16) made all
        # four couplings per row, so the claim is now the GPU's bytes for a
        # row whatever else is asked with it; `inference_display` says so in
        # the README sentence.
        # SpectralClustering.predict joins it (lane/saved-model-reference-gaps,
        # 2026-09-16, DEVIATION 2860): the Nystrom extension from a saved
        # `prediction_data=True` fit, through `spectral_predict` on this
        # binding, for both affinities the estimator accepts.
        inference_lanes=("umap", "spectral", "spectral-precomputed"),
        inference_display="UMAP transform of a saved embedding (the GPU's bytes for a row, whatever else is asked in the same batch)",
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
            "spectral/host/spectral_predict_host.mojo",
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
            # lane/spectral-predict (2026-09-15): the fit entries that keep
            # the prediction data and SpectralClustering.predict (DEVIATION
            # 2860), so a saved model predicts from the inference wheel.
            "spectral_fit_predict_dataset_state", "spectral_fit_predict_graph_state",
            "spectral_predict",
            "umap_fit_transform", "umap_transform", "umap_numeric_mode",
            "rand_score", "mean_squared_error", "mean_absolute_error",
            "root_mean_squared_error", "roc_auc_score", "precision_recall_curve",
            "log_loss", "confusion_matrix", "precision_recall_fscore",
            "kl_divergence", "trustworthiness", "fowlkes_mallows_score",
            "accuracy_score_weighted", "r2_score_weighted",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note=(
            "Ships: the metric functions, and the transform of a saved UMAP embedding."
        ),
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
            "standard-scaler-no-std", "minmax-scaler-clip", "par-scaler", "par-scaler-minmax",
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
        wheel_note=(
            "Ships: the scaler fits, so the six scaler lanes can be checked on an installed CPU. "
            "StandardScaler and MinMaxScaler already transformed from a saved model through the "
            "shipped estimators binding; standard_fit and minmax_fit are only here."
        ),
        ships_in_wheel=True,
    ),
    dict(
        family="tsa",
        binding="_mojolearn_tsa_host",
        routes="_mojolearn_tsa",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("holtwinters", "holtwinters-multiplicative", "kpss", "select-d", "par-holtwinters",
                        "par-forecast-holtwinters"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("ExponentialSmoothing", "kpss_test", "select_d"),
        display="Holt-Winters",
        host_modules=("holtwinters/host/hw_oracle.mojo", "tsa/checks/kpss_oracle.mojo",
                      "holtwinters/host/hw_predict.mojo", "bindings/holtwinters_host_predict.mojo",
                      "bindings/kpss_host_test.mojo"),
        exports=(
            "tsa_host_numeric_mode", "tsa_host_vendor", "tsa_host_column",
            "tsa_host_sabotage", "tsa_vendor", "holtwinters_fit",
            "holtwinters_forecast", "holtwinters_predict", "kpss_test",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note=(
            "Ships: holtwinters_fit and the KPSS test (also used by the Python select_d controller), "
            "so the holtwinters, kpss and select-d lanes can be "
            "checked on an installed CPU. A saved ExponentialSmoothing still forecasts through the "
            "shipped forecast binding, which registers kpss_test from the same shared module "
            "(bindings/kpss_host_test.mojo), so the two binaries cannot drift."
        ),
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
        wheel_note=(
            "Ships: the coordinate descent and the linkage, so the lasso, elasticnet and "
            "agglomerative lanes can be checked on an installed CPU. Saved Lasso and ElasticNet "
            "models still predict through the shipped estimators binding."
        ),
        ships_in_wheel=True,
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
        wheel_note=(
            "Ships: predicts on a CPU from saved SVC, SVR and isolation forest models, seven "
            "inference lanes."
        ),
        ships_in_wheel=True,
    ),
    dict(
        family="trees",
        binding="_mojolearn_trees_host",
        routes="_mojolearn_trees",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("et-clf", "et-reg", "et-clf-entropy-bestfirst", "et-reg-bootstrap-parallel", "par-forest-et", "par-forest-et-clf"),
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
        wheel_note=(
            "Ships: the Extra Trees fit, including the best-first growth and the shard offsets, so "
            "the five Extra Trees lanes can be checked on an installed CPU. Saved models still "
            "predict through the shipped forest binding."
        ),
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
        training_lanes=(
            "rf-clf", "rf-reg", "rf-clf-entropy-log2-noboot", "rf-clf-balanced-parallel",
            "rf-reg-poisson", "rf-reg-gamma-ig", "par-forest", "par-forest-reg", "rf-score-weighted",
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
        wheel_note=(
            "Ships: the random forest fit, the four criteria and the class-weighted bootstrap, so "
            "the eight random forest lanes can be checked on an installed CPU. Saved models still "
            "predict through the shipped forest binding."
        ),
        ships_in_wheel=True,
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
                        "gpc", "gpc-multiclass", "gp-sample-y", "gp-sample-y-normalize",
                        "gp-optimize", "gp-optimize-restarts",
                        # lane/unlaned-public-algorithms (2026-09-20): the class-sharded
                        # GPC drivers, whose shard IS this family's gpc_fit/gpc_predict
                        "par-gpc-fit", "par-gpc-predict"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("GaussianProcessRegressor", "GaussianProcessClassifier"),
        display="the Gaussian process regressor and classifier",
        host_modules=(
            "gaussian_process/host/gpr_oracle.mojo",
            "gaussian_process/host/gpr_grad_oracle.mojo",
            "gaussian_process/host/gp_theta.mojo",
            "gaussian_process/host/gpc_oracle.mojo",
            "gaussian_process/host/gpc_steps.mojo",
            "cholesky/host/chol_oracle.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "gp_host_numeric_mode", "gp_host_vendor", "gp_host_column",
            "gp_host_sabotage", "gp_vendor", "gp_numeric_mode",
            "gpr_fit", "gpr_predict", "gpr_sample_y", "gpr_lml_grad", "gp_log64", "gp_theta_params",
            "gp_restart_uniforms", "gpc_fit", "gpc_predict", "cholesky_profile_jitter",
            "cholesky_factor", "cholesky_solve",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note=(
            "Ships: the Gaussian process fit, the log marginal likelihood gradient and the "
            "posterior draws, so the nine GP lanes can be checked on an installed CPU. Saved "
            "models still predict through the shipped gp_infer binding, which carries no fit."
        ),
        ships_in_wheel=True,
    ),
    dict(
        # CPU training for the workstream D estimators
        # (lane/cpu-training-d-estimators, 2026-09-15): the kernel methods
        # family's host binding. It routes `_mojolearn_kernel_methods` on a
        # CPU-only install with the GPU binding's fit, predict and transform
        # names for KernelRidge, Nystroem and RBFSampler at the linear and
        # rbf kernels and now polynomial, sigmoid and Laplacian;
        # kernel_methods_rows_parallel_available stays absent,
        # so cooperative kernel-method drivers refuse by name. The RBF sampler
        # transform splits whole rows in Python and uses the ordinary host
        # transform per shard; its non-cooperative CPU route is declared here.
        family="kernel_methods",
        binding="_mojolearn_kernel_methods_host",
        routes="_mojolearn_kernel_methods",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("rbf-sampler", "kernel-ridge", "nystroem", "par-rbf-sampler",
                        "kernel-ridge-poly", "kernel-ridge-sigmoid", "kernel-ridge-laplacian", "nystroem-poly", "nystroem-sigmoid", "nystroem-laplacian"),
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
        wheel_note=(
            "Ships: the kernel ridge, Nystroem and random Fourier feature fits, plus the "
            "logical row-sharded Fourier transform. Its parallel lane remains pending "
            "reference qualification. Saved models still predict and transform "
            "through the shipped estimators binding."
        ),
        ships_in_wheel=True,
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
        training_lanes=("gmm", "gmm-random-init", "gmm-sample", "gmm-random-init-sample"),
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
        wheel_note=(
            "Ships: the EM fit and its two starts, so the four Gaussian mixture lanes can be "
            "checked on an installed CPU. Saved models still score, predict and sample through the "
            "shipped mixture_infer binding, which shares this family's scoring source and carries "
            "no fit."
        ),
        ships_in_wheel=True,
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
        wheel_note=(
            "Ships: inference-only binding. The mixture family's fit stays a source reference build; "
            "the scoring and sampling entries are shared through bindings/mixture_host_scoring.mojo "
            "and carry no fit symbol."
        ),
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
        wheel_note=(
            "Ships: the Boruvka rounds, the condensed tree and the extraction, so the two HDBSCAN "
            "lanes can be checked on an installed CPU. Saved models still answer "
            "approximate_predict and the membership vectors through the shipped hdbscan_infer "
            "binding."
        ),
        ships_in_wheel=True,
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
        wheel_note=(
            "Ships: inference-only binding for saved GaussianProcessRegressor and "
            "GaussianProcessClassifier models; the gp family's fit and optimizer do not ship."
        ),
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
        wheel_note=(
            "Ships: inference-only binding for approximate_predict, membership_vector and "
            "all_points_membership_vectors over a saved model; carries no fit."
        ),
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
            # lane/catboost-parity (2026-09-19): the non-default border
            # types run the device fit's own host function
            # (`select_borders`) on the CPU column
            "gbdt-border-types",
            # lane/catboost-parity: Ordered boosting through
            # gbdt/host/gbdt_oracle_ordered.mojo::gbdt_ordered_host_fit
            "gbdt-ordered", "gbdt-ordered-bayesian-noise",
            # lane/catboost-parity: the MAE / Quantile / MAPE starting point
            # (`gbdt/metrics/sample_quantile.mojo`, shared host code)
            "gbdt-bfa-quantile",
            # lane/catboost-parity: the SymmetricTree defaults, the Bayesian
            # bootstrap and score noise through gbdt_oracle.mojo::gbdt_host_fit
            "gbdt-catboost-defaults",
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
            # lane/catboost-parity: the Ordered plan and the quantile
            # constant, host code the device fit calls too
            "gbdt/data/ordered_plan.mojo", "gbdt/metrics/sample_quantile.mojo",
            "core/gbdt_host_predict.mojo",
        ),
        exports=(
            "gbdt_host_numeric_mode", "gbdt_host_vendor", "gbdt_host_column",
            "gbdt_host_sabotage", "gbdt_vendor", "gbdt_numeric_mode",
            "gbdt_fit", "gbdt_predict", "gbdt_predict_multi", "gbdt_model_dim",
            "gbdt_sigmoid", "gbdt_sigmoid_pair", "gbdt_binary_probabilities", "gbdt_binary_classes",
            "gbdt_fit_ordered_rmse", "gbdt_fit_two_level_feature_freq",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note=(
            "Ships: the boosting fit, which is twenty-one covered lanes, the largest block of "
            "checkable surface any one binding holds. Saved models still predict through the "
            "shipped forest binding. CPU training of CTR categorical features still refuses by "
            "name (NO_CPU_PATH). At about 1.3 MB it is the largest host binding."
        ),
        ships_in_wheel=True,
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
        training_lanes=("mlp", "optim-sgd", "optim-adam-clip", "cross-entropy-arms", "training-primitives", "ordered-gradient-sum", "par-mlp",
                        "samba", "samba-untied-dropout-accum", "par-samba", "par-samba-clip",
                        "mlp-bf16w", "mlp-int8w", "samba-bf16w", "samba-int8w",
                        # lane/laneless-public-classes (2026-09-19)
                        "grad-accumulation"),
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
        wheel_note=(
            "Ships: the optimizers, the losses, the gradient clip, the accumulation and the "
            "backward primitives, so the ten training lanes can be checked on an installed CPU. "
            "The forwards a user infers with remain in the shipped neural family. Shipping the "
            "binding does not make CPU training public: an ordinary fit still refuses outside the "
            "verifier's scope."
        ),
        ships_in_wheel=True,
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
        wheel_note=(
            "Ships: bootstrap, permutation_test and monte_carlo_integrate compute an answer from a "
            "user's own data and a user's own function. They train no model and there is nothing to "
            "save, so the saved-model inference boundary never had a side for them to fall on and "
            "they used to refuse on a CPU-only install, which is indefensible for cheap analysis "
            "functions a user calls on their laptop. Andrew's call (2026-09-16, "
            "lane/expose-inference-surface): the boundary exists to keep CPU TRAINING OF MODELS "
            "internal, not to exclude analysis. This binding registers the three entries and no "
            "fit, and _backend already routes _mojolearn_resample here, so shipping it is the whole "
            "change."
        ),
        ships_in_wheel=True,
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
        training_lanes=("mamba2", "mamba2-dtlimit", "mamba1", "mamba3",
                        "mamba1-bf16w", "mamba1-int8w", "mamba2-bf16w", "mamba2-int8w", "mamba3-bf16w", "mamba3-int8w"),
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
        wheel_note=(
            "Ships: the three blocks' prefill backward, so the mamba1, mamba2 and mamba3 lanes can "
            "be checked on an installed CPU; the shipped neural family is forward only and cannot "
            "answer their train part. About 1.06 MB, the second largest host binding."
        ),
        ships_in_wheel=True,
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
        training_lanes=("arima", "arima-011", "arima-seasonal-c", "par-arima", "arima-exog",
                        "arima-exog-seasonal", "par-forecast-arima"),
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
        wheel_note=(
            "Ships: the Kalman filter fit and the L-BFGS, so the six ARIMA lanes can be checked on "
            "an installed CPU. Saved models still predict and forecast through the shipped "
            "forecast binding, which carries no fit."
        ),
        ships_in_wheel=True,
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
        wheel_note=(
            "Ships: the embedding backward fold and both execution plans, so the embedding lanes "
            "can be checked on an installed CPU. Lookup in a saved table is still served by the "
            "shipped embedding_infer binding."
        ),
        ships_in_wheel=True,
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
        wheel_note=(
            "Ships: inference-only binding serving _mojolearn_embedding, lookup in a saved embedding "
            "table; the embedding family's fit does not ship."
        ),
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
        training_lanes=("ivf", "ivf-euclidean", "ivf-extend", "par-ivf"),
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
            "ivf_flat_extend", "ivf_flat_partial_search", "ivf_finalize_distances",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note=(
            "Ships: the IVF-Flat index build with its k-means quantizer, so the IVF lanes can be "
            "checked on an installed CPU. Search over a saved index is still served by the shipped "
            "ivf_search binding, which carries no build."
        ),
        ships_in_wheel=True,
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
            "ivf_flat_partial_search", "ivf_finalize_distances",
        ),
        gate="tools/classical_host_gate.py and tools/identity_break.py",
        wheel_note=(
            "Ships: inference-only binding serving _mojolearn_ivf, search over and extension of a "
            "saved IVF-Flat index; the index build does not ship."
        ),
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
        # an installed CPU-only wheel. lane/inference-holtwinters (2026-09-15)
        # adds saved Holt-Winters models the same way: holtwinters_forecast
        # and holtwinters_predict from bindings/holtwinters_host_predict.mojo,
        # the source the reference tsa binding registers them from, and the
        # `_mojolearn_tsa` route when that binding is not built.
        family="forecast",
        binding="_mojolearn_forecast_host",
        routes=None,
        serves=("_mojolearn_arima", "_mojolearn_tsa"),
        loaded_by="_backend._HOST_INFERENCE_MODULES and python/mojolearn/_classical_host.py",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=("arima", "arima-011", "arima-seasonal-c", "holtwinters", "holtwinters-multiplicative",
                         "arima-exog", "arima-exog-seasonal"),
        forest_kinds=(),
        classes=("ARIMA", "ExponentialSmoothing", "kpss_test"),
        display=("batched ARIMA prediction, in sample and out of sample, and forecasts, with or without"
                 " exogenous regressors, and Holt-Winters"
                 " forecasts and in-sample one-step predictions, additive and multiplicative"),
        host_modules=("arima/host/arima_oracle.mojo", "bindings/arima_host_predict.mojo",
                      "holtwinters/host/hw_predict.mojo", "bindings/holtwinters_host_predict.mojo",
                      "tsa/checks/kpss_oracle.mojo", "bindings/kpss_host_test.mojo"),
        exports=(
            "forecast_host_numeric_mode", "forecast_host_vendor", "forecast_host_column",
            "forecast_host_sabotage", "arima_vendor", "arima_numeric_mode",
            "arima_predict", "arima_forecast", "tsa_vendor", "holtwinters_forecast",
            "holtwinters_predict", "kpss_test",
        ),
        gate="tools/classical_host_gate.py and tools/identity_break.py",
        wheel_note=(
            "Ships: inference-only binding serving _mojolearn_arima and _mojolearn_tsa, predict and "
            "forecast from saved ARIMA and Holt-Winters models; neither fit ships. Since "
            "lane/expose-inference-surface (2026-09-16) it also carries kpss_test, which trains no "
            "model at all: it computes a statistic from the caller's own series, so it belongs on "
            "the shipped side, and registering it here puts it there without shipping the tsa "
            "family's holtwinters_fit. Both bindings register it from bindings/kpss_host_test.mojo, "
            "so they answer through one source."
        ),
        ships_in_wheel=True,
    ),
    dict(
        family="transformer",
        binding="_mojolearn_transformer_host",
        routes="_mojolearn_transformer",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("transformer", "transformer-window", "transformer-bf16w", "transformer-int8w"),
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
        wheel_note=(
            "Ships: the block's decode step and prefill backward, which the transformer and "
            "transformer-window lanes hash, so they can be checked on an installed CPU; the "
            "shipped neural family carries the forward only."
        ),
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


#: LANES A RELEASE RECORD DOES NOT COVER, so the public set does not either.
#: `tools/identity_break.py` excludes every `par-*` lane from a full-column
#: record (its RECORD_EXCLUDED_PREFIXES, 2026-09-16): in a record they are
#: one-device runs, several cannot be covered honestly at all, and the
#: two-device claim is made by the dedicated legs instead. Their references in
#: the shipped table therefore come from a record whose scope no longer
#: includes them and which nothing will refresh, and unlike a shrunk fixture
#: there is no LANE_REVISIONS entry to catch it going stale. A user's `verify`
#: must not rest on that. They stay covered lanes, and the CPU identity gate
#: still runs all thirteen.
#:
#: ASKED AGAIN 2026-09-19 (lane/lm-attention-fallback), because the columns
#: LOOK finished now: thirteen of the twenty-one covered par-* lanes --
#: `par-arima`, `par-forest`, `par-forest-et`, `par-holtwinters`, `par-mlp`,
#: `par-queries-kde`, `par-queries-knn`, `par-queries-radius`,
#: `par-reference-knn`, `par-reference-knn-reg`, `par-samba`,
#: `par-samba-clip` and `par-scaler` -- each carry amd + apple + cpu +
#: nvidia on every real part of all nine fixtures, which is a stronger table
#: position than any of the eighteen PUBLIC_REFERENCE_CANDIDATES holds.
#: They are still not
#: promoted, and the reason above is not a formality -- it is visible in the
#: provenance of the very cells that look complete. Reading each part's
#: `cols` back through the table's `records` list:
#:
#:   par-forest        apple + nvidia <- 2026-09-14_166-lanes
#:   par-arima         apple + nvidia + amd <- 2026-09-14_166-lanes
#:   par-mlp           apple + nvidia + amd <- 2026-09-14_166-lanes
#:   par-samba         apple + nvidia + amd <- 2026-09-14_166-lanes
#:   par-queries-knn   apple + nvidia + amd <- 2026-09-14_166-lanes
#:   par-holtwinters   apple <- 2026-09-14_166-lanes
#:
#: 2026-09-14_166-lanes is the record RECORD_EXCLUDED_PREFIXES was added to
#: EXCLUDE par-* from two days later. So these witnesses come from a scope
#: that no longer runs them, exactly as this note says, and no release record
#: will ever refresh them: the number looks complete because it was frozen,
#: not because it is current. The remainder (par-holtwinters' nvidia and amd,
#: from 2026-09-15_holtwinters-linesearch-fix) are one-off fix records, which
#: is the same problem in a smaller package.
#:
#: And the deeper one, which no column can fix: on ONE device a par-* driver
#: is DEGENERATE -- `_verify_reference.admit` now says so in its own
#: docstring, and `lane_applicability.degenerate('apple-metal')` holds all
#: thirteen -- so a one-device column passes whatever the code does, while a
#: CPU-only install has exactly one device. Promoting them would put a cell
#: in a user's `verify --all` that cannot fail on their machine. Two devices
#: is what states their claim, and that is what the dedicated legs do.
PUBLIC_EXCLUDED_PREFIXES = ("par-",)

#: WHY EACH EXCLUDED PREFIX IS EXCLUDED, in the words a user reads
#: (lane/verifier-full-exposure, 2026-09-20). Until this existed the prefix
#: removed 55 of the harness's 256 lanes from `verify --all` and said nothing:
#: the command printed `186 lanes` and a user had no way to learn that 70
#: lanes were not in that number, let alone why. AN ABSENCE IS NOT A REPORT.
#: A prefix in `PUBLIC_EXCLUDED_PREFIXES` with no sentence here is refused by
#: `lane_exposure()` rather than silently dropping its lanes again.
#:
#: The sentence has to carry three things, because the reader is a user on one
#: box who wants to know whether they were shortchanged: what the lane claims,
#: why this machine cannot state it, and the command that can. The answer is
#: never "you are missing a feature"; it is "this claim is about two devices
#: and you have one", which is a fact about the claim, not about the install.
PUBLIC_EXCLUDED_PREFIX_REASONS = {
    "par-": ("claim requires two devices: a par-* driver claims that a "
             "TWO-DEVICE column hashes equal to the one-device column cell "
             "for cell. `verify` is always a one-device run (it refuses "
             "MOJOLEARN_PAR_DEVICES), and on one device the comparison is a "
             "run against itself, which passes whatever the code does. Owed: "
             "a two-GPU box running `MOJOLEARN_PAR_DEVICES=0,1 python3 "
             "tools/identity_break.py --repeats 2 --lanes <lane> --json "
             "par2.json` beside the same lanes at MOJOLEARN_PAR_DEVICES=0, "
             "admitted with _verify_reference.admit(par_axis=True)"),
}

#: THE STATUS EVERY HARNESS LANE GETS IN `lane_exposure()`, and the whole
#: vocabulary of it. Only `EXPOSED` may ever contribute to a pass; the other
#: four are the shapes an absence takes, spelled out so a reader cannot mistake
#: one for coverage. `UNDECLARED` exists so that the default for a lane nobody
#: has thought about is a REFUSAL, not a silent omission: it is what the gate
#: `tools/lane_accounting.py` fails on.
LANE_EXPOSED = "EXPOSED"
LANE_NOT_APPLICABLE = "NOT APPLICABLE"
LANE_OWED = "OWED"
LANE_HELD = "HELD"
LANE_UNDECLARED = "UNDECLARED"
LANE_STATUSES = (LANE_EXPOSED, LANE_NOT_APPLICABLE, LANE_OWED, LANE_HELD,
                 LANE_UNDECLARED)

#: The `PUBLIC_PENDING_LANES` reasons that mean NO COMMITTED RECORD CARRIES A
#: HASH, which is the one hold whose honest run-time status is OWED rather
#: than HELD: the lane could be run, and every part of it would read OWED,
#: which is not a pass. Every other reason is a hold on the COMPARISON, so
#: running the lane would not settle it and the status stays HELD.
PUBLIC_PENDING_OWED_REASONS = ("no reference",)

#: COVERED LANES HELD BACK FROM THE PUBLIC SET, each with the reason, checked
#: by python/mojolearn/tests/test_host_surface.py against the harness, the
#: shipped table and the measured run rather than trusted as prose
#: (lane/ship-cpu-host-families, 2026-09-16). A lane leaves this dict the day
#: its reason stops being true, which for `stale reference` is the next
#: release record.
#:
#:   stale reference   the lane's fixture moved past the hash the shipped
#:                     table carries (identity_break LANE_REVISIONS, the
#:                     thirteen lanes lane/identity-fixtures-light shrank).
#:                     `_verify_all` would drop and name them anyway; keeping
#:                     them out means the public set is a set that PASSES,
#:                     not one that reports thirteen lanes it cannot compare.
#:   no reference      no committed record carries a single hash for the lane,
#:                     so every part would read OWED, which is not a pass.
#:   own record        diffed against TRAINING_FIX_COLUMNS, not the release
#:                     record (TRAINING_FIX_LANES). The shipped table is built
#:                     from every committed record, so these do have hashes,
#:                     but `record_covered_lanes()` is what the public set is
#:                     held to and they are not in it.
#:   unwatched         the shipped table DOES carry cells for the lane at the
#:                     current fixture revision, so nothing static holds it
#:                     back any more. What is missing is the one thing the
#:                     promotion rule will not do without: a CPU-only
#:                     `verify --all` watched to read IDENTICAL for it. Added
#:                     2026-09-16 by lane/expose-stepfull, which regenerated
#:                     the table and so cleared `stale reference` for four
#:                     lanes at once without being able to run that column.
#:                     The same regeneration gave SIX `no reference`
#:                     lanes their first cells, so they moved here too.
#:   one column        the shipped table DOES carry cells for the lane at the
#:                     current fixture revision, but every one of them rests
#:                     on a SINGLE device class, so the reference has one
#:                     witness and nothing has ever reproduced it. A user who
#:                     disagreed with such a number could not tell their own
#:                     machine apart from our one column. This is the bar
#:                     `PUBLIC_REFERENCE_CANDIDATES` already holds `svc-poly`
#:                     to, written as a condition the table answers rather
#:                     than as prose: `test_host_surface` counts the classes
#:                     in each cell's `cols` and fails BOTH when a lane here
#:                     has gained a second class and when one held as
#:                     `unwatched` has only one. It leaves the day a second
#:                     column carries the lane, which for the lanes a fixture
#:                     change emptied is the next release record
#:                     (lane/reference-regen, 2026-09-17).
#:   measured          a CPU-only `verify --all` at this commit WATCHED the
#:                     lane and it did not read clean. This reason is the only
#:                     one that comes from a run rather than from a static
#:                     condition, and it carries what the run said.
#:   qualification pending
#:                     RETIRED 2026-09-20 (lane/verifier-full-exposure). It
#:                     said "the expanded property/hardware completion plan is
#:                     not met", which named no artifact, so nobody reading it
#:                     could tell whether the debt was a run, a rental, a file
#:                     or a merge, nor when it was paid. Five lanes carried it.
#:                     They now carry `owed artifact`, which names the thing.
#:   owed artifact <class> ...
#:                     the lane's cells are current and admissible, but one
#:                     DEVICE CLASS is missing from them and the reason names
#:                     which, what file or command would supply it, and what is
#:                     wrong with the nearest thing the tree already has. The
#:                     third word is the device class, and
#:                     `test_public_reference_lanes_are_derived_and_every_pending_reason_is_true`
#:                     fails BOTH when that class is absent from the reason's
#:                     vocabulary and when the shipped table's cells have
#:                     GAINED it on every part, so the hold cannot outlive the
#:                     column that pays it.
#:   no cpu route      THE ONE REASON A COVERED LANE CANNOT HAVE, and the
#:                     only entry here that is not a covered lane
#:                     (lane/unlaned-public-algorithms, 2026-09-20). The
#:                     lane's public surface REFUSES BY NAME on a CPU-only
#:                     install, so no CPU column can ever carry it and no
#:                     host family can ever declare it: the reference it owes
#:                     is owed on a GPU column and on no other. Every other
#:                     reason here says "the evidence has not been taken
#:                     yet"; this one says "this box is the wrong box, and
#:                     always will be". It is CHECKED against
#:                     `identity_break.GPU_ONLY_LANES`, which is what
#:                     `tools/identity_break.py` uses to drop the lane from a
#:                     full-column CPU run, so the reason cannot be used as
#:                     an escape hatch for a lane that merely has no CPU
#:                     column yet. `tools/lane_accounting.py`'s invariant is
#:                     satisfied by the entry; `verification_matrix`'s
#:                     `cpu_vacuous_lanes()` derives the same fact
#:                     independently from `lane_applicability`.
#:
#: THE THIRTEEN `stale reference` LANES WERE RESOLVED ON 2026-09-16 by the
#: regeneration lane/expose-stepfull landed, and the split is the one the
#: rule predicted: NINE lost every cell, because each record that carried
#: them predates the fixture shrink, so what they owe is a RECORD and their
#: reason is now `no reference`; FOUR kept cells at the current revision and
#: owe only the watched run, so their reason is now `unwatched`.
#:
#: AND THE REASON CAME BACK THE SAME DAY, which is the point of keeping it in
#: the vocabulary. `samba-untied-dropout-accum` (lane/shrink-floors) and then
#: four more from lane/dead-arms are `stale reference` again: `mamba2-dtlimit`,
#: whose dt clamp moved, and `mamba3`, `transformer` and `transformer-window`,
#: whose two same-shape norm weights stopped being one tensor. The last three
#: were PUBLIC that morning. A fixture change recreates this reason on the
#: same day it is declared resolved.
PUBLIC_PENDING_LANES = {
    # lane/laneless-public-classes (2026-09-19). The lane is new, so no
    # committed record and no shipped table cell describes it yet, and a
    # public lane with no reference makes an installed `verify --all` read
    # OWED for something it could have been told not to ask. It joins
    # `public_reference_lanes()` the day a record carries it.
    #
    # THE ONLY ONE OF THE TEN STILL HERE FOR THIS REASON
    # (lane/new-lane-reference-promotion, 2026-09-20). The other nine had
    # admissible committed columns by then and were promoted into the shipped
    # table; this one could not be, and the merge guard is what said so
    # rather than a reading of the record:
    #
    #     merge_reference_lanes -> language-model-config/base: missing parts ['batch']
    #
    # bench/results/identity_break/2026-09-19_language-model-config/ carries
    # ONE fixture (base) and three parts (train, infer, model) for it, and a
    # scoped admission requires all nine fixtures with train, infer, model
    # and batch. So its debt is unchanged and is a RECORD: one CPU column
    # over all nine fixtures at `--repeats 2` with the batch part run. It is
    # not a GPU debt and not a rental.
    "language-model-config": "no reference",

    # lane/linalg-public (2026-09-19). Adding these three to the linalg
    # family's `training_lanes` made them public reference lanes with no
    # column of any kind behind them --
    # `test_public_reference_lanes_are_derived_and_every_pending_reason_is_true`
    # named `linalg-eigh` and `linalg-qr` and failed from that commit onward.
    #
    # THE REASON MOVED ON 2026-09-20 (lane/new-lane-reference-promotion), and
    # it moved because the columns the comment above was waiting for had
    # landed in the tree and nothing had admitted them into the artifact the
    # wheel ships. A scoped, additive admission from the committed records --
    # no binding, no GPU, no rental -- gave all three cells on all nine
    # fixtures from 2026-09-19_vendor-class-gaps (amd + nvidia, @0845c6de7)
    # and 2026-09-19_apple-column-gaps (@ef8c2b311), 36 parts per class per
    # lane, with 0 conflicts. Intersected over every part of every fixture
    # they carry amd + apple + nvidia, which is all three vendor classes.
    #
    # So `no reference` is no longer true and `one column` never was: what is
    # left is the ONE thing the promotion rule will not do without, a CPU-only
    # `verify --all` WATCHED to read IDENTICAL for them. This box cannot run
    # it -- it has no binding built -- so the reason says exactly that.
    # Note the CPU class is deliberately absent from the intersection: the
    # 2026-09-19_linalg-public CPU column covers the base fixture only (3 of
    # 36 parts), which is why the watched run is still the open item.
    "linalg-qr": "unwatched",
    "linalg-eigh": "unwatched",
    "linalg-svdvals": "unwatched",

    # lane/laneless-public-classes (2026-09-19). Three lanes joined their
    # families' `training_lanes` with no committed column of any kind. Their
    # CPU evidence is in `bench/results/identity_break/
    # 2026-09-19_laneless-public-classes/`: 27 of 27 cells STABLE over all
    # nine fixtures at `--repeats 2`, both declared batch parts STABLE and
    # seen BATCH_MOVED under the harness switch, and each one watched failing
    # under a build define (`saved-model-host-infer` and `lowbit-conversions`
    # 9/9 under their families' OWN defines and 0/9 under the generic one,
    # `grad-accumulation` 9/9 under the training family's).
    #
    # TWO OF THE THREE SPLIT ON 2026-09-20 (lane/new-lane-reference-promotion)
    # and the split is the one the vocabulary predicts.
    # `lowbit-conversions` and `grad-accumulation` gained amd + apple +
    # nvidia GPU columns on top of that CPU one -- four classes on every part
    # of all nine fixtures, from 2026-09-19_vendor-class-gaps (@0845c6de7)
    # and 2026-09-19_apple-column-gaps (@ef8c2b311) -- so nothing static holds
    # them and their reason is the watched CPU-only run.
    # `saved-model-host-infer` gained NOTHING but the CPU column it already
    # had: intersected over every part it rests on `cpu` ALONE, one witness,
    # a number nothing has ever reproduced. That is `one column`, not
    # `unwatched`, and test_host_surface checks the difference both ways --
    # it fails if a lane held here gains a second class and if one held as
    # `unwatched` has only one. What it owes is a GPU column, which is the
    # one thing on this list that needs hardware.
    "saved-model-host-infer": "one column",
    "lowbit-conversions": "unwatched",
    "grad-accumulation": "unwatched",

    # lane/models-namespace-lanes (2026-09-19). The three `mojolearn.models`
    # lanes joined the linalg, tokenizer and neural families' `training_lanes`
    # with no committed column. Their CPU evidence is in
    # `bench/results/identity_break/2026-09-19_models-namespace/`: 27 of 27
    # cells STABLE over all nine fixtures at `--repeats 2`, a clean replay
    # IDENTICAL on all 27, `hf-causal-lm`'s batch part STABLE, and each lane
    # watched failing 9/9 under a build define -- `hf-checkpoint` under
    # MOJOLEARN_LOWBIT_CONVERT_SABOTAGE (and 0/9 under the generic one,
    # which is recorded there and is why linalg needs its own arm),
    # `hf-tokenizer` under both of the tokenizer family's own arms, and
    # `hf-causal-lm` under the generic define on the neural family.
    #
    # ALL THREE MOVED TO `unwatched` ON 2026-09-20
    # (lane/new-lane-reference-promotion). The cross-vendor claim the comment
    # above said "has never been taken on any column" HAD been taken, in
    # 2026-09-19_vendor-class-gaps (amd + nvidia, @0845c6de7) and
    # 2026-09-19_apple-column-gaps (@ef8c2b311); the columns were committed
    # and admissible and simply had not been admitted into the shipped table.
    # They now carry amd + apple + cpu + nvidia on every part of all nine
    # fixtures, 0 conflicts. What is left for each is the watched CPU-only
    # `verify --all`, which this box has no binding to run.
    "hf-checkpoint": "unwatched",
    "hf-tokenizer": "unwatched",
    "hf-causal-lm": "unwatched",

    # 2026-09-18: current all-nine, full-property AMD captures now agree
    # with the CPU references for these five neural routes. The independent
    # second column clears their former "one column" reason. All-nine,
    # twice-repeated installed CPU core replay subsequently passed; see
    # LANE_STATUS_verifier-installed-continuation.md. Extended installed
    # properties and NVIDIA/current Apple completion remain owed, as does
    # the exact expanded 0.8.7 release certificate. Keep the holds explicit.
    #
    # THE APPLE HALF OF THAT DEBT IS ALREADY PAID AND JUST UNADMITTED
    # (measured 2026-09-19, lane/lm-attention-fallback). These five read
    # amd + cpu in the shipped table and nothing else, and the obvious
    # reading -- that a Mac owes them all nine fixtures -- is wrong.
    # bench/results/identity_break/2026-09-18_installed-apple-properties/
    # carries one Apple M4 column per lane at commit aff968968, two repeats,
    # identical mode, `admit()` clean, and every one of them agrees with the
    # SHIPPED refs on every comparable part with ZERO disagreements:
    # mamba3 72, transformer 72, transformer-window 72, samba 81,
    # samba-untied-dropout-accum 81. That is all nine fixtures each.
    #
    # NVIDIA IS THE REAL DEBT, and it is a rental. Every NVIDIA column in
    # the tree that names these lanes predates the fixture change that made
    # them `stale reference` on 2026-09-17 (their two same-shape norm
    # weights stopped being one tensor), and it shows: the best of them,
    # 2026-09-14_166-lanes, agrees on 36 parts and DISAGREES on 117. An old
    # column of the old bytes proves nothing here, so this one cannot be
    # closed by an admission -- it needs a current NVIDIA box running the
    # five lanes over all nine fixtures. That is the ONE piece of GPU time
    # these five still owe.
    #
    # MEASURED 2026-09-20 (lane/verifier-full-exposure), by calling
    # `_verify_reference.admit()` on all 302 committed columns under
    # bench/results/ that name any of these five and keeping the ones that
    # are admissible AND at the harness's current LANE_REVISIONS entry:
    #
    #   mamba3               7 columns, classes {amd, apple, cpu}. TWENTY
    #                        NVIDIA columns name it and NONE is at the current
    #                        revision.
    #   transformer          13 columns, classes {amd, apple, cpu, nvidia}
    #   transformer-window   13 columns, classes {amd, apple, cpu, nvidia}
    #   samba                14 columns, classes {amd, apple, cpu, nvidia}
    #   samba-untied-...     13 columns, classes {amd, apple, cpu, nvidia}
    #
    # So the phrase `qualification pending` was hiding TWO DIFFERENT DEBTS of
    # different size, and the four that read `nvidia` above are much closer
    # than the one that does not. The nvidia column the four share is
    # bench/results/attention_replay_vendors_2026-09-18/nvidia_identity/remote/
    # attn-replay-extra/identity_break.{before,after}.nvidia-sm_90a.json: nine
    # fixtures, two repeats, mode identical, `admit()` returns None. What it
    # lacks is parts. Five of the nine carry a usable value at every fixture
    # (train, infer, model, batch, rlpair) and four do not: `stepfull`, which
    # is a DEFAULT compared part, and batchgrad, batchscale and ragged. It is
    # also outside bench/results/identity_break/, the only tree
    # `_verify_reference.build_table` walks, so nothing would find it.
    #
    # THE APPLE COLUMN IS A THIRD, SMALLER DEBT AND IT IS NOT GPU TIME. All
    # five read `apple` above, from
    # bench/results/identity_break/2026-09-18_installed-apple-properties/,
    # which is committed, `admit()` clean, nine fixtures, two repeats and
    # complete on all nine parts -- and is in NO cell of the shipped table for
    # these lanes, which still read amd + cpu. Admitting it is a scoped
    # `verify --all --batch-checks --emit-reference --reference-table`, a pure
    # function over JSON that needs no GPU and no binding. It is not done here
    # because lane/new-lane-reference-promotion owns the shipped table.
    "mamba3": (
        "owed artifact nvidia column. No committed column anywhere under "
        "bench/results/ carries mamba3 on an NVIDIA device at fixture revision "
        "'norms-near-one-1'; the twenty that name it all predate the 2026-09-16 "
        "norm change (lane/dead-arms) and describe bytes this harness no longer "
        "produces. Unlike the other four neural holds there is no near miss to "
        "repair. Owed, on a current NVIDIA box: `MOJOLEARN_COMMIT=$(git rev-parse HEAD) "
        "python3 tools/identity_break.py --repeats 2 --lanes mamba3 --step-full "
        "--batch-grad --batch-scale --ragged --json identity_break.nvidia.json`, "
        "landed under bench/results/identity_break/"),
    "transformer": (
        "owed artifact nvidia stepfull. An admissible NVIDIA column at the current "
        "fixture revision exists (bench/results/attention_replay_vendors_2026-09-18/"
        "nvidia_identity/remote/attn-replay-extra/identity_break.after.nvidia-sm_90a.json, "
        "admit() None, nine fixtures, two repeats) but carries only train, infer, model, "
        "batch and rlpair. The DEFAULT part `stepfull` is absent, as are batchgrad, "
        "batchscale and ragged, and the file sits outside bench/results/identity_break/ "
        "so build_table never sees it. Owed: re-run those lanes on a current NVIDIA box "
        "with `--step-full --batch-grad --batch-scale --ragged` and land the column under "
        "bench/results/identity_break/"),
    "transformer-window": (
        "owed artifact nvidia stepfull. Same column and same gap as `transformer`: "
        "bench/results/attention_replay_vendors_2026-09-18/nvidia_identity/remote/"
        "attn-replay-extra/identity_break.after.nvidia-sm_90a.json is admissible at the "
        "current revision on nine fixtures but has no `stepfull`, batchgrad, batchscale or "
        "ragged value, and is outside the tree build_table walks. Owed: the same NVIDIA "
        "re-run with `--step-full --batch-grad --batch-scale --ragged`, landed under "
        "bench/results/identity_break/"),
    "samba": (
        "owed artifact nvidia stepfull. Same column and same gap as `transformer`: the "
        "attention-replay NVIDIA column is admissible at revision 'steps-1-1' on nine "
        "fixtures but carries no `stepfull`, batchgrad, batchscale or ragged value, and "
        "is outside bench/results/identity_break/. Owed: the same NVIDIA re-run with "
        "`--step-full --batch-grad --batch-scale --ragged`, landed under "
        "bench/results/identity_break/"),
    "samba-untied-dropout-accum": (
        "owed artifact nvidia stepfull. Same column and same gap as `transformer`, at "
        "revision 'steps-3-1'. This is the one lane whose PRE-revision NVIDIA records "
        "also agree, because lane/shrink-floors reversed its step cut, but an old column "
        "of coincidentally equal bytes is not a current witness. Owed: the same NVIDIA "
        "re-run with `--step-full --batch-grad --batch-scale --ragged`, landed under "
        "bench/results/identity_break/"),
    # Spectral, Fowlkes-Mallows and both ARIMA-exog lanes passed all nine
    # fixtures twice through the public CPU verifier, with native negative
    # controls. See LANE_STATUS_cpu_public_promotion.md for artifact scope.
    # CTR saved-model lanes were promoted after all nine fixtures passed twice
    # from an installed CPU development wheel with bundled, digest-checked models.
    # See docs/lanes/LANE_STATUS_verification_evidence_audit.md. This is CPU
    # inference replay, not CPU CTR training or final release qualification.
    # The other fix-record lanes passed installed reference-table replay;
    # legacy `identity` still restricts itself to its original column scope.
    # All twelve low-bit weight lanes gained strict all-nine CPU references
    # and complete native control pairs in 2026-09-17_cpu-complete-dependencies.
    #
    # lane/unlaned-public-algorithms (2026-09-20): the six public entries
    # tools/verification_matrix.py reported with NO identity lane at all.
    # None of the six has a shipped-table cell, and `tools/lane_accounting.py`
    # is right to demand that each say so here rather than sit between the two
    # mechanisms. They split by WHY the cell is missing, and the split is not
    # cosmetic: two owe a record, four owe a different box.
    #
    # The class-sharded Gaussian process classifier drivers ARE covered lanes
    # -- they take a CPU column, recorded in
    # bench/results/identity_break/2026-09-20_unlaned-public-algorithms/ with
    # a sabotage pair that moved every cell -- but no committed record carries
    # a GPU hash for them and the shipped table has no cell, so an installed
    # `verify --all` would read OWED for every part. `PUBLIC_EXCLUDED_PREFIXES`
    # already keeps every `par-` lane out of the public set; this says why the
    # table is empty as well.
    "par-gpc-fit": "no reference",
    "par-gpc-predict": "no reference",
    # The four with no CPU column at all. `mamba1_session_create` and
    # `transformer_decode_session_create` exist only in
    # bindings/_mojolearn_mamba.mojo and bindings/_mojolearn_transformer.mojo;
    # ParallelCausalLM and parallel cross-validation refuse unless
    # `_backend.vendor()` is cuda or hip, before either builds a layer or
    # prepares a fold. A CPU run cannot make these covered lanes, today or
    # ever, so `no reference` would be the wrong reason: it promises a record
    # this box can take.
    "mamba1-decode-session": "no cpu route",
    "transformer-decode-session": "no cpu route",
    "par-causal-lm": "no cpu route",
    "par-cross-val": "no cpu route",
}


def public_reference_lanes():
    """THE LANES AN INSTALLED WHEEL CAN CHECK, on a CPU-only install, as
    `python -m mojolearn verify --all` and `python -m mojolearn identity`
    select them.

    Until 2026-09-16 this was eight hand-named probes plus the tokenizer, and
    then thirty-nine after lane/expose-inference-surface measured thirty more.
    Both lists were bounded by the same thing: a lane whose host binding was
    not in the wheel could not be re-run on the machine it was installed on,
    whatever the shipped table said. Every host family ships now (see the
    module docstring), so the limit is no longer what is installed but what
    can be honestly compared: a covered lane, in a release record's scope,
    with a reference in the shipped table, WATCHED to read clean by a CPU-only
    run at this commit.

    It is DERIVED, never hand-listed, so a lane that gains a CPU path joins on
    the day it is declared, and one that cannot be compared is held back in
    `PUBLIC_PENDING_LANES` with the reason written down. The promotion rule
    lane/expose-inference-surface set is kept: a lane is public only because a
    run was watched to read IDENTICAL for it, which is why every lane this
    returns is in the measured run recorded in
    docs/lanes/LANE_STATUS_lane-ship-cpu-host-families.md, and why
    `PUBLIC_REFERENCE_CANDIDATES` stays out until its own condition is met.
    """
    lanes = [lane for lane in covered_lanes()
             if not lane.startswith(PUBLIC_EXCLUDED_PREFIXES)
             and lane not in PUBLIC_PENDING_LANES
             and lane not in PUBLIC_REFERENCE_CANDIDATES
             and lane not in PUBLIC_HOST_ONLY_LANES]
    return lanes + list(PUBLIC_HOST_ONLY_LANES)


def excluded_prefix_reason(lane):
    """The sentence for the `PUBLIC_EXCLUDED_PREFIXES` prefix `lane` starts
    with, or None. Raises when a prefix carries no sentence, which is the
    whole point: adding a prefix without saying why would put its lanes back
    into silent absence."""
    for prefix in PUBLIC_EXCLUDED_PREFIXES:
        if lane.startswith(prefix):
            try:
                return PUBLIC_EXCLUDED_PREFIX_REASONS[prefix]
            except KeyError:
                raise RuntimeError(
                    f"host_surface: PUBLIC_EXCLUDED_PREFIXES carries {prefix!r} but "
                    f"PUBLIC_EXCLUDED_PREFIX_REASONS does not say why. Every lane under "
                    f"that prefix would vanish from `verify --all` with no reason given, "
                    f"which is the defect lane/verifier-full-exposure exists to remove."
                ) from None
    return None


def lane_exposure(lanes):
    """EVERY LANE ACCOUNTED FOR: `{lane: {"status", "reason", "exposed"}}`
    over `lanes`, which is meant to be the whole harness lane list.

    THE DEFECT THIS REPLACES (lane/verifier-full-exposure, 2026-09-20). The
    identity harness defines 256 lanes; `public_reference_lanes()` returns
    186. The other 70 -- 55 removed by `PUBLIC_EXCLUDED_PREFIXES` and 15 held
    in `PUBLIC_PENDING_LANES` -- were not reported as anything. They were
    simply not in the set, so `verify --all` printed a lane count that was the
    number it ran and said nothing about the number it did not, and a user
    reading `186 lanes` had no way to find out that a fifth of the harness was
    missing or why. A silently absent lane is indistinguishable from a lane
    that does not exist, and both are indistinguishable from one that passed.

    Every status but `LANE_EXPOSED` is a gap, and the caller must treat it as
    one: `_verify_all.verdict()` refuses to print VERIFIED when any lane in
    the run's own scope is not `LANE_EXPOSED` and clean. Statuses:

      EXPOSED         the installed verifier runs it and compares it
      NOT APPLICABLE  the claim cannot be STATED on this kind of run at all,
                      whatever the code does (the `par-*` drivers, whose
                      claim is about two devices)
      OWED            it could be run and every part would read OWED, because
                      no committed record carries a hash for it
      HELD            it is held out of the public set on a named condition
                      that running it would not settle
      UNDECLARED      nothing in this manifest says anything about it. This is
                      the failure state, not a category: `tools/
                      lane_accounting.py` refuses it.

    It is DERIVED from the same three tables `public_reference_lanes()` is
    derived from, so a lane cannot be exposed here and pending there.
    """
    public = set(public_reference_lanes())
    candidates = set(PUBLIC_REFERENCE_CANDIDATES)
    out = {}
    for lane in lanes:
        if lane in public:
            out[lane] = dict(status=LANE_EXPOSED, reason=None, exposed=True)
            continue
        prefix_reason = excluded_prefix_reason(lane)
        if prefix_reason is not None:
            out[lane] = dict(status=LANE_NOT_APPLICABLE, reason=prefix_reason, exposed=False)
        elif lane in PUBLIC_PENDING_LANES:
            why = PUBLIC_PENDING_LANES[lane]
            owed = any(why.startswith(r) for r in PUBLIC_PENDING_OWED_REASONS)
            out[lane] = dict(
                status=LANE_OWED if owed else LANE_HELD,
                reason=("no committed record carries a hash for this lane, so every part of it "
                        "would read OWED; owed: a release record that runs it" if owed else why),
                exposed=False)
        elif lane in candidates:
            out[lane] = dict(status=LANE_HELD,
                             reason="reference qualification pending (PUBLIC_REFERENCE_CANDIDATES)",
                             exposed=False)
        else:
            out[lane] = dict(status=LANE_UNDECLARED, reason=None, exposed=False)
    return out


def lane_exposure_counts(lanes):
    """`{status: n}` over `lane_exposure(lanes)`, every status present even at
    zero, so a reader sees the category that is empty rather than inferring
    it from an absent key."""
    exposure = lane_exposure(lanes)
    counts = {s: 0 for s in LANE_STATUSES}
    for row in exposure.values():
        counts[row["status"]] += 1
    return counts


#: Public reference lanes of a shipped host family with no GPU path to cover,
#: {lane: family} (2026-09-15). The tokenizer lane loads the synthetic
#: vocabulary the harness trains itself, so it needs no vocabulary file on the
#: install. Since lane/cpu-verifier-gaps-7 it is also a covered lane, which
#: the full CPU gate runs; this list is the inference wheel's reference set.
#: It is public though no record carries its hashes yet (the GPT-2 table left
#: the tree and the lane now loads the synthetic vocabulary the harness trains
#: itself), so its parts read OWED until the next record. It was public before
#: lane/ship-cpu-host-families widened the set and narrowing it would be a
#: regression, which is why the derivation adds it back rather than deriving
#: it: it is the one public lane a run selects without comparing.
# BPE vocabulary training and fold construction are pure host operations.
# Vocabulary serialization and corpus preparation also run on the host, but
# use the shipped tokenizer binding to encode and decode. Their replay and
# both negative controls are recorded in 2026-09-18_tokenized-corpus; this is
# CPU-only evidence, not a claim of independent GPU implementation equality.
# Missing reference hashes must still read OWED.
# None means pure Python: no native host binding is required. bpe-trainer
# names the tokenizer family since lane/bpe-builder-native (2026-09-18):
# BpeVocabularyTrainer trains through that binding's bpe_train by default and
# falls back to the pure Python reference only when the binding lacks it.
PUBLIC_HOST_ONLY_LANES = {"tokenizer": "tokenizer", "bpe-trainer": "tokenizer",
                          "cross-val-folds": None, "bpe-vocabulary": "tokenizer",
                          "tokenized-corpus": "tokenizer"}

#: Lanes that PASS every static condition for `public_reference_lanes()` and
#: are not in it (lane/expose-inference-surface, 2026-09-16). Each one:
#:
#:   * is a lane tools/identity_break.py defines;
#:   * is in `record_covered_lanes()`, so it is diffed against
#:     TRAINING_GPU_COLUMNS rather than a fix record;
#:   * has a real (not `n/a`, not conflicted) train reference in the shipped
#:     table `mojolearn/verify_reference/table.json` on ALL NINE fixtures,
#:     so it can read IDENTICAL rather than OWED;
#:   * already carries a `cpu` column in that table, so some CPU box has
#:     reproduced it once;
#:   * has all nine fixtures on ALL THREE of TRAINING_GPU_COLUMNS, so the
#:     `identity` command's `--require-columns 4` can be met. This condition
#:     was added after the fact: `svc-poly` met every other one and was in
#:     this list until the columns were counted, and its cells turned out to
#:     rest on two columns only (apple and cpu, from
#:     2026-09-15_inference-svm; its NVIDIA and AMD recordings are owed to the
#:     next release record, as CLASSICAL_RECORDED already notes). It rejoins
#:     this list the day a record carries those two columns;
#:   * is reachable from a binding that SHIPS, so promoting it adds no binary
#:     to the wheel. Either the declaring family ships, or a shipped
#:     inference-only binding serves its route: `kpss` is declared by `tsa`,
#:     which holds holtwinters_fit and stays a source build, while the shipped
#:     `forecast` binding serves `_mojolearn_tsa` and registers `kpss_test`.
#:     Requiring the declaring family itself to ship would wrongly reject a
#:     lane a user can call.
#:
#: They are NOT live yet, and the reason is not a policy one: nothing has run
#: them through `verify --all` on a CPU-only install at this commit. Promoting
#: a lane here into `public_reference_lanes()` without that run would ship a
#: claim no one has watched succeed, and would turn a user's `verify` into
#: REFUSED or DIVERGENT if it were wrong. The run is one command per family
#: and needs no GPU; docs/lanes/LANE_STATUS_lane-expose-inference-surface.md
#: carries it. A lane moves from here into `public_reference_lanes()` on the
#: day that run reads IDENTICAL for it and the sabotage host build reads
#: DIVERGENT for it.
#:
#: WHAT ALL EIGHTEEN ARE WAITING ON, RE-MEASURED 2026-09-19
#: (lane/lm-attention-fallback). One thing, and it is the same thing for
#: every entry: the shipped table carries NO `nvidia` column for any of them,
#: on any part of any fixture. Fifteen of the eighteen are otherwise
#: complete -- `amd` + `apple` + `cpu` on every real part of all nine
#: fixtures, at the current revision, unconflicted. The other three
#: (gbdt-query-rmse, gmm-sample, gmm-random-init-sample) hold amd + cpu
#: everywhere and apple on some fixtures only; see their entry below, where
#: that gap turns out to be unadmitted rather than unrecorded too.
#:
#: THE PER-ENTRY COMMENTS BELOW HAD ALL GONE STALE, and that is the reason
#: the condition is now ASSERTED in test_host_surface rather than described
#: here. `gp-optimize` said "only Apple GPU witnesses"; the nine from
#: lane/ship-cpu-host-families said "rests on the APPLE column alone, with no
#: NVIDIA and no AMD"; `svc-poly` said "apple and cpu". The 2026-09-18 AMD
#: columns landed and made all three sentences false, and nothing failed,
#: because a condition kept in prose is a condition nothing checks. Each has
#: been rewritten to the measured state below.
#:
#: AND THE NVIDIA COLUMN THEY NEED IS ALREADY IN THE TREE, COMMITTED AND
#: ADMISSIBLE. It is not a rental:
#:
#:   bench/results/identity_break/2026-09-19_gpu-class-gaps/
#:     nvidia-nvidia-geforce-rtx-4090-sm_89.classical.json   (commit 14e0c6bbe)
#:       -- 16 of the 18, all nine fixtures, identical mode, repeats 2
#:   bench/results/identity_break/2026-09-19_single-device-gaps/
#:     nvidia-nvidia-geforce-rtx-4090-sm_89.single-device-gaps.json
#:       -- gp-normalize-y and gp-sample-y-normalize, same terms
#:
#: `_verify_reference.admit()` returns None (admissible) for both: mode
#: identical, par_devices 0, heldout_seed 1, no sabotage flag, no `gpu_slot`
#: held by another run. AND THEY AGREE WITH WHAT IS ALREADY SHIPPED: held
#: against the table's own refs through `_part_value` at two repeats, the two
#: columns match on 612 of 612 comparable parts across the eighteen lanes,
#: with ZERO disagreements, so the admission adds columns and changes no
#: cell.
#:
#: THAT ADMISSION LANDED (2026-09-19, this lane). The scoped, additive
#: `verify --all --emit-reference --reference-table --batch-checks` -- the
#: same operation 91a2a68d5 used to add 63 GBDT cells without touching an
#: existing one -- ran from the committed records with no binding and no
#: rented box, and the result was diffed against the shipped table before it
#: was installed: 0 reference hashes changed, 0 vendor classes lost, 0 cells
#: removed, 0 non-candidate lanes' admission witness touched, the global
#: harness witness unchanged, and exactly 162 cells (18 lanes x 9 fixtures)
#: GAINING a class. The merge guard earned its keep on the first attempt: run
#: without `--batch-checks` it REFUSED rather than drop `gbdt-query-rmse`'s
#: batchgrad/batchscale/ragged cells.
#:
#: So the list is EMPTY, and empty is this mechanism's success state, not a
#: deletion: every lane that was waiting on a column is now a public
#: reference lane, and `public_reference_lanes()` derives them because
#: nothing excludes them any more. The eighteen are kept by name in
#: `PUBLIC_REFERENCE_PROMOTED` so the condition that admitted them stays
#: checkable -- a table edit that took a class back off any part of any
#: fixture fails test_host_surface rather than quietly un-proving a public
#: claim. A NEW candidate belongs here, with the column it waits on named,
#: rather than nowhere.
#:
#: What this did NOT discharge: `SAVED_MODEL_INFERENCE_OWED` keeps all six
#: kernel/nystroem entries. Their debt is an NVIDIA `classical_host` record,
#: a different artifact from an identity-table cell -- the only kernel-variant
#: saved-model recordings in the tree are Apple's
#: (bench/results/classical_host/2026-09-18-apple-kernel-variants) -- so the
#: RTX 4090 identity column could not and did not pay it. Promoting a lane's
#: identity claim and discharging its saved-model replay debt are two
#: different promises; this took the first only.
PUBLIC_REFERENCE_CANDIDATES = ()

#: The eighteen promoted 2026-09-19 by the admission described above, kept by
#: name so the evidence stays load-bearing rather than becoming prose. Each
#: carries amd + apple + cpu + nvidia on EVERY part of ALL NINE fixtures in
#: the shipped table; test_host_surface asserts that of every entry, which is
#: what makes a later regression fail here instead of in a user's install.
#: Intersected over parts, never unioned -- three of them (gbdt-query-rmse,
#: gmm-sample, gmm-random-init-sample) had a UNION that read three classes
#: while single fixtures rested on one, and a union test would have promoted
#: them a day early.
PUBLIC_REFERENCE_PROMOTED = (
    "gbdt-query-rmse",
    "gmm-random-init-sample",
    "gmm-sample",
    "gp-normalize-y",
    "gp-optimize",
    "gp-optimize-restarts",
    "gp-sample-y",
    "gp-sample-y-normalize",
    "gpc",
    "gpc-multiclass",
    "ivf-extend",
    "kernel-ridge-laplacian",
    "kernel-ridge-poly",
    "kernel-ridge-sigmoid",
    "nystroem-laplacian",
    "nystroem-poly",
    "nystroem-sigmoid",
    "svc-poly",
)

#: Implemented saved-model CPU inference with recording or qualification debt.
#: `inference_lanes()` names selectable classical_host_gate.py routes; a route
#: can now be declared and carry Apple recordings while still owing the other
#: vendor columns and installed replay. The debt must remain visible until its
#: named gates pass. Recording itself always requires a GPU, so the CPU binding
#: cannot manufacture its own expected answer.
#:
#: lane/saved-model-reference-gaps (2026-09-16) emptied this list of the four
#: entries it took a GPU box for: `dbscan`, `agglomerative`, `spectral` and
#: `spectral-precomputed` are declared inference lanes now, recorded at
#: bench/results/classical_host/2026-09-16-nvidia-predict. That lane also
#: found WHY none of them had a recording: `classical_host_gate.py record`
#: had been raising AttributeError on main since `--lane-rule-only` was added,
#: ahead of every other refusal, so the tool that makes a recording could not
#: start.
#:
#: lane/classical-host-recordings (2026-09-16) took the last one. `kmeans`,
#: `kmeans-random`, `kmeans-array`, `kmeans-weighted`, `kmeans-sqrt` and
#: `kmeans-classic-pp` are declared inference lanes now, recorded at
#: bench/results/classical_host/2026-09-16-nvidia-kmeans, so THE LIST IS
#: EMPTY. Empty means every saved-model inference the classical host door
#: dispatches is also a declared, gated and recorded inference lane; it does
#: NOT mean nothing is left to implement. A new `save` that ships without a
#: recording belongs here, with the reason, rather than nowhere.
SAVED_MODEL_INFERENCE_OWED = {
    "kernel-ridge-poly": "The serialization format is implemented; Apple and AMD saved-model recordings and CPU replay passed all nine fixtures (2026-09-18 kernel records); NVIDIA records and installed qualification remain owed.",
    "kernel-ridge-sigmoid": "The serialization format is implemented; Apple and AMD saved-model recordings and CPU replay passed all nine fixtures (2026-09-18 kernel records); NVIDIA records and installed qualification remain owed.",
    "kernel-ridge-laplacian": "The serialization format is implemented; Apple and AMD saved-model recordings and CPU replay passed all nine fixtures (2026-09-18 kernel records); NVIDIA records and installed qualification remain owed.",
    "nystroem-poly": "The serialization format is implemented; Apple and AMD saved-model recordings and CPU replay passed all nine fixtures (2026-09-18 kernel records); NVIDIA records and installed qualification remain owed.",
    "nystroem-sigmoid": "The serialization format is implemented; Apple and AMD saved-model recordings and CPU replay passed all nine fixtures (2026-09-18 kernel records); NVIDIA records and installed qualification remain owed.",
    "nystroem-laplacian": "The serialization format is implemented; Apple and AMD saved-model recordings and CPU replay passed all nine fixtures (2026-09-18 kernel records); NVIDIA records and installed qualification remain owed.",
}


def public_reference_candidates():
    """The lanes measured addable to `public_reference_lanes()`, in order."""
    return list(PUBLIC_REFERENCE_CANDIDATES)


def saved_model_inference_owed():
    """{lane: remaining saved-model recording or qualification debt}."""
    return dict(SAVED_MODEL_INFERENCE_OWED)


def wheel_notes():
    """{family: why it does or does not ship in the wheels}."""
    return {f["family"]: f["wheel_note"] for f in FAMILIES}


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
        public_reference_lanes=public_reference_lanes(),
        public_reference_candidates=public_reference_candidates(),
        saved_model_inference_owed=saved_model_inference_owed(),
        wheel_notes=wheel_notes(),
        adapted_modules={k: dict(v) for k, v in ADAPTED_MODULES.items()},
        gbdt_ctr_models_dir=GBDT_CTR_MODELS_DIR,
        sabotage_build_defines={name: sabotage_build_defines(name) for name in families()},
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
    g.add_argument("--public-reference-lanes", action="store_true",
                   help="lanes `verify --all` runs on a CPU-only install (comma separated)")
    g.add_argument("--public-reference-candidates", action="store_true",
                   help="lanes measured addable to --public-reference-lanes (comma separated)")
    g.add_argument("--saved-model-inference-owed", action="store_true",
                   help="implemented saved-model inference that is not a declared inference lane")
    g.add_argument("--wheel-notes", action="store_true",
                   help="why each host family does or does not ship in the wheels")
    g.add_argument("--markdown", action="store_true", help="the surface as a Markdown table")
    g.add_argument("--json", action="store_true", help="the whole manifest as JSON")
    g.add_argument("--gbdt-ctr-models", action="store_true", help="the saved-model directory the CTR table lanes load on a CPU column")
    g.add_argument("--sabotage-build-defines", metavar="FAMILY", default=None,
                   help="MOJOLEARN_BUILD_EXTRA_DEFINES of FAMILY in the gate's sabotage host set")
    p.add_argument("--sep", default=None, help="separator for list output (default: comma for lanes and kinds, space otherwise)")
    args = p.parse_args(argv)
    if args.sabotage_build_defines is not None:
        print(sabotage_build_defines(args.sabotage_build_defines))
        return 0
    if args.gbdt_ctr_models:
        print(GBDT_CTR_MODELS_DIR)
        return 0
    if args.public_reference_lanes:
        print(",".join(public_reference_lanes()))
        return 0
    if args.public_reference_candidates:
        print(",".join(public_reference_candidates()))
        return 0
    if args.saved_model_inference_owed:
        for lane, why in saved_model_inference_owed().items():
            print(f"{lane}: {why}")
        return 0
    if args.wheel_notes:
        for name, why in wheel_notes().items():
            print(f"{name}: {why}")
        return 0
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
