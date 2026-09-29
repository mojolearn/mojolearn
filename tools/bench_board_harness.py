# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHERE THE BOARD'S SETTINGS COME FROM: NVIDIA's two benchmark harnesses.

Andrew (2026-09-29): "use the harness for nvidia for classical ml and for
decision trees and use their tuning params for us and the opponent", and "why
would our settings EVER differ from theirs? the point is so they don't".

For every lane one of NVIDIA's harnesses covers, each parameter that harness
sets EXPLICITLY is the board's value, on OUR arm and on EVERY opponent arm.
Where the harness leaves a work setting at each library's default (bins,
seed, sampling, forest bins, ...), the board keeps pinning ONE value on every
arm (`PINNED`), because library defaults differ and an unset default is how
two arms end up solving two problems. Lanes with no harness entry keep the
board's own settings (`NOT_COVERED`).

The values live in the drivers' lane configs (tools/speed_gbdt_arm.py
lane_config, tools/classical_two_datasets.py, tools/bench_board_more.py,
tools/bench_board_algos.py); each carries `harness_source(lane)` so every
race's settings name the file and commit its values were copied from. This
module is the provenance record and the one place the commits are written.

    python3 tools/bench_board_harness.py        # print the table as JSON
"""
import json
import sys

# The commits below were read with `git ls-remote` on 2026-09-29 and the files
# fetched at exactly those commits.
CUML_COMMIT = "e0f7a4e31578c8eeef376f3ce715d846bfee8d4c"
GBM_BENCH_COMMIT = "73a976b036249ff9d8cb30cf9082bb414b911379"
CATBOOST_COMMIT = "e628c03fb0e6b760592652a995163f26be7ea7d3"

CUML_BENCH = {
    "name": "cuML benchmark (RAPIDS)",
    "repo": "https://github.com/rapidsai/cuml",
    "commit": CUML_COMMIT,
    "file": "python/cuml/cuml/benchmark/algorithms.py",
    "url": "https://github.com/rapidsai/cuml/blob/%s/python/cuml/cuml/benchmark/algorithms.py"
           % CUML_COMMIT,
    "timing": "runners.py BenchmarkTimer: min over --n-reps (default 1) of perf_counter around "
              "the bench function; cli.py sizes 10,000 to 100,000 rows (2 sizes) x 64, 256, 512 "
              "features, blobs",
}
GBM_BENCH = {
    "name": "NVIDIA gbm-bench",
    "repo": "https://github.com/NVIDIA/gbm-bench",
    "commit": GBM_BENCH_COMMIT,
    "file": "algorithms.py (shared_params and each library's configure), runme.py (-ntrees)",
    "url": "https://github.com/NVIDIA/gbm-bench/blob/%s/algorithms.py" % GBM_BENCH_COMMIT,
}
CATBOOST_DEFAULTS = {
    "name": "CatBoost boosting options",
    "repo": "https://github.com/catboost/catboost",
    "commit": CATBOOST_COMMIT,
    "file": "catboost/private/libs/options/boosting_options.cpp",
    "url": "https://github.com/catboost/catboost/blob/%s/catboost/private/libs/options/"
           "boosting_options.cpp#L13" % CATBOOST_COMMIT,
    "line": 'IterationCount("iterations", 1000)',
}

# ---------------------------------------------------------------------------
# gbm-bench: shared_params (the XGBoost spelling) go to every library;
# runme.py -ntrees defaults to 500; LightGBM adds max_leaves 256; XGBoost,
# LightGBM and CatBoost binary classification add
# scale_pos_weight = len(y_train) / count_nonzero(y_train). The RF classes
# (sklearn, cuML) take shared_params minus reg_lambda and learning_rate, with
# n_estimators = ntrees.
# ---------------------------------------------------------------------------
GBM = {
    "shared_params": {"max_depth": 8, "learning_rate": 0.1, "reg_lambda": 1},
    "ntrees": 500,
    "lightgbm": {"max_leaves": 256},
    "binary": "scale_pos_weight = len(y_train) / count_nonzero(y_train) on XGBoost, "
              "LightGBM and CatBoost",
    "rf": {"max_depth": 8, "n_estimators": 500},
}

# ---------------------------------------------------------------------------
# cuML: AlgorithmPair(name, shared_args, cpu_args, cuml_args), copied for the
# pairs the board races. shared_args go to every arm; cpu_args to the CPU
# library; cuml_args to cuML (and to ours where ours has the same parameter,
# because ours implements cuML's algorithm and the parameter changes the work).
# ---------------------------------------------------------------------------
CUML = {
    "KMeans": dict(shared_args=dict(init="k-means++", n_clusters=8, max_iter=300, n_init=1),
                   cuml_args=dict(oversampling_factor=0)),
    "AgglomerativeClustering": dict(shared_args=dict(n_clusters=8, metric="euclidean",
                                                     linkage="single")),
    "SpectralClustering": dict(shared_args=dict(n_clusters=8, affinity="nearest_neighbors",
                                                n_neighbors=10, n_init=1, random_state=42)),
    "PCA": dict(shared_args=dict(n_components=10)),
    "IncrementalPCA": dict(shared_args=dict(n_components=10)),
    "tSVD": dict(shared_args=dict(n_components=10)),
    "GaussianRandomProjection": dict(shared_args=dict(n_components=10)),
    "SparseRandomProjection": dict(shared_args=dict(n_components=10)),
    "NearestNeighbors": dict(shared_args=dict(n_neighbors=64),
                             cpu_args=dict(algorithm="brute", n_jobs=-1)),
    "KernelDensity": dict(shared_args=dict(kernel="gaussian", bandwidth=1.0)),
    "DBSCAN": dict(shared_args=dict(eps=3, min_samples=2), cpu_args=dict(algorithm="brute")),
    "LinearRegression": dict(shared_args={}),
    "ElasticNet": dict(shared_args=dict(alpha=0.1, l1_ratio=0.5)),
    "Lasso": dict(shared_args={}),
    "Ridge": dict(shared_args={}),
    "KernelRidge": dict(shared_args={}),
    "LogisticRegression": dict(shared_args={}),
    "TSNE": dict(shared_args={}),
    "SVC-RBF": dict(shared_args=dict(kernel="rbf")),
    "SVR-RBF": dict(shared_args=dict(kernel="rbf")),
    "LinearSVC": dict(shared_args={}),
    "LinearSVR": dict(shared_args={}),
    "KNeighborsClassifier": dict(shared_args={}),
    "KNeighborsRegressor": dict(shared_args={}),
    "GaussianNB": dict(shared_args={}), "MultinomialNB": dict(shared_args={}),
    "BernoulliNB": dict(shared_args={}), "ComplementNB": dict(shared_args={}),
    "CategoricalNB": dict(shared_args={}),
    "RobustScaler": dict(shared_args={}), "MaxAbsScaler": dict(shared_args={}),
    "Normalizer": dict(shared_args={}), "Binarizer": dict(shared_args={}),
    "PowerTransformer": dict(shared_args={}), "PolynomialFeatures": dict(shared_args={}),
    "SimpleImputer": dict(shared_args={}), "OrdinalEncoder": dict(shared_args={}),
    "KBinsDiscretizer": dict(shared_args={}), "QuantileTransformer": dict(shared_args={}),
    "LabelEncoder": dict(shared_args={}), "LabelBinarizer": dict(shared_args={}),
    "OneHotEncoder": dict(shared_args=dict(sparse_output=False, handle_unknown="ignore")),
    "TargetEncoder": dict(shared_args=dict(smooth=0.0), cpu_args=dict(cv=4, random_state=42),
                          cuml_args=dict(n_folds=4, seed=42, split_method="interleaved",
                                         multi_feature_mode="independent")),
    "HDBSCAN": dict(shared_args={}, cpu_args=dict(core_dist_n_jobs=-1)),
    "UMAP-Unsupervised": dict(shared_args=dict(n_neighbors=5, n_epochs=500)),
    "MBSGDClassifier": dict(shared_args={}, cuml_args=dict(eta0=0.005, epochs=100)),
    "MBSGDRegressor": dict(shared_args={}, cuml_args=dict(eta0=0.005, epochs=100)),
}

#: board lane -> (harness, entry). "gbm" entries name gbm-bench's learner set.
COVERED = {
    # trees, gbm-bench
    "gbdt-symmetric": ("gbm-bench", "xgb/lgbm/cat shared_params, ntrees 500"),
    "gbdt-symmetric-1000": ("gbm-bench", "xgb/lgbm/cat shared_params, 1000 trees (CatBoost's "
                            "own default iteration count; see CATBOOST_DEFAULTS)"),
    "gbdt-depthwise": ("gbm-bench", "xgb/lgbm/cat shared_params, ntrees 500"),
    "gbdt-lossguide": ("gbm-bench", "xgb/lgbm/cat shared_params, ntrees 500"),
    "gbdt-multiclass": ("gbm-bench", "xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task)"),
    "gbdt-categorical": ("gbm-bench", "xgb/lgbm/cat shared_params, ntrees 500 (binary task)"),
    "gbdt-ordered": ("gbm-bench", "cat shared_params, ntrees 500, boosting_type 'Ordered'"),
    "rf": ("gbm-bench", "skrf/cumlrf: max_depth 8, n_estimators 500"),
    # classical, cuML
    "kmeans": ("cuml", "KMeans"), "pca": ("cuml", "PCA"), "ols": ("cuml", "LinearRegression"),
    "knn": ("cuml", "NearestNeighbors"), "kde": ("cuml", "KernelDensity"),
    "svc": ("cuml", "SVC-RBF"), "dbscan": ("cuml", "DBSCAN"), "hdbscan": ("cuml", "HDBSCAN"),
    # classical2, cuML
    "umap": ("cuml", "UMAP-Unsupervised"), "logreg": ("cuml", "LogisticRegression"),
    "linearsvc": ("cuml", "LinearSVC"), "ridge": ("cuml", "Ridge"), "lasso": ("cuml", "Lasso"),
    "elasticnet": ("cuml", "ElasticNet"), "linearsvr": ("cuml", "LinearSVR"),
    "tsvd": ("cuml", "tSVD"), "knn-clf": ("cuml", "KNeighborsClassifier"),
    "knn-reg": ("cuml", "KNeighborsRegressor"), "spectral": ("cuml", "SpectralClustering"),
    "agglomerative": ("cuml", "AgglomerativeClustering"), "svr": ("cuml", "SVR-RBF"),
    "kernel-ridge": ("cuml", "KernelRidge"),
    # algos, cuML
    "algos/incremental-pca": ("cuml", "IncrementalPCA"),
    "algos/gaussian-rp": ("cuml", "GaussianRandomProjection"),
    "algos/sparse-rp": ("cuml", "SparseRandomProjection"),
    "algos/tsne": ("cuml", "TSNE"),
    "algos/sgd-clf": ("cuml", "MBSGDClassifier"), "algos/sgd-reg": ("cuml", "MBSGDRegressor"),
    "algos/gaussian-nb": ("cuml", "GaussianNB"), "algos/bernoulli-nb": ("cuml", "BernoulliNB"),
    "algos/categorical-nb": ("cuml", "CategoricalNB"),
    "algos/multinomial-nb": ("cuml", "MultinomialNB"),
    "algos/complement-nb": ("cuml", "ComplementNB"),
    "algos/robust-scaler": ("cuml", "RobustScaler"), "algos/maxabs-scaler": ("cuml", "MaxAbsScaler"),
    "algos/normalizer": ("cuml", "Normalizer"), "algos/binarizer": ("cuml", "Binarizer"),
    "algos/power-transformer": ("cuml", "PowerTransformer"),
    "algos/poly-features": ("cuml", "PolynomialFeatures"),
    "algos/simple-imputer": ("cuml", "SimpleImputer"), "algos/ordinal": ("cuml", "OrdinalEncoder"),
    "algos/onehot": ("cuml", "OneHotEncoder"), "algos/target-encoder": ("cuml", "TargetEncoder"),
    "algos/label-encoder": ("cuml", "LabelEncoder"),
    "algos/label-binarizer": ("cuml", "LabelBinarizer"),
    "algos/kbins": ("cuml", "KBinsDiscretizer"),
    "algos/quantile-transformer": ("cuml", "QuantileTransformer"),
}

#: Lanes neither harness has; they keep the board's own settings.
NOT_COVERED = {
    "et": "neither harness races ExtraTrees",
    "iforest": "neither harness races IsolationForest",
    "gbdt-rank-yetirank": "gbm-bench has regression, binary and multiclass tasks only",
    "gbdt-rank-pairlogit": "gbm-bench has regression, binary and multiclass tasks only",
    "classical2 gmm, gpr, gpc, nystroem, rbf-sampler, arima, ets, ivf, spectral-embedding":
        "no cuML benchmark AlgorithmPair",
    "neural (every lane)": "neither harness races neural models; the neural races keep their own",
    "algos (every lane not in COVERED)": "no cuML benchmark AlgorithmPair",
}

#: Work settings the harness leaves at each library's default, where the
#: defaults differ between libraries: ONE value on every arm, with the reason.
PINNED = [
    ("seed", "7 (42 on spectral and target-encoder, where cuML's benchmark sets 42)",
     "gbm-bench sets none and most cuML pairs set none; each library's default differs "
     "(XGBoost 0, CatBoost 0, LightGBM its own, scikit-learn and cuML None)"),
    ("bins", "254 borders = 255 bins on every boosted arm",
     "defaults differ (ours 128, CatBoost CPU 254 borders, CatBoost GPU 128, XGBoost 256, "
     "LightGBM 255 bins)"),
    ("row and column sampling", "none (CatBoost and ours bootstrap_type 'No', subsample 1.0 "
     "and colsample 1.0 elsewhere)",
     "CatBoost's default bootstrap samples rows; the others do not"),
    ("boosting_type", "Plain", "CatBoost's default is data-dependent (Ordered on small pools)"),
    ("max_leaves", "2 ** max_depth = 256 on every boosted arm",
     "gbm-bench gives LightGBM 256; the others take the same cap at depth 8"),
    ("leaf estimation, split floors, borders, nan_mode, boost_from_average",
     "as speed_gbdt_arm.lane_config pins them",
     "the libraries' defaults differ in meaning (lane_config docstring)"),
    ("rf n_bins", "128 (ours and cuML)", "cuML's default; scikit-learn searches exact "
     "thresholds and has no bin count"),
    ("rf max_features, bootstrap, max_samples, min_samples_leaf",
     "'sqrt' (classification) or 1.0, True, 1.0, 1",
     "each library's own default for the task, pinned so a default change cannot move one arm"),
    ("kmeans tol", "1e-7", "scikit-learn and cuML default 1e-4, ours its own; the board "
     "keeps one value (ours and cuML refuse 0)"),
    ("every other parameter a lane sets today", "its current value, on every arm",
     "the harness leaves it at the library default"),
]


def harness_source(lane):
    """The provenance a covered lane's settings carry, or None."""
    hit = COVERED.get(lane)
    if hit is None:
        return None
    src = GBM_BENCH if hit[0] == "gbm-bench" else CUML_BENCH
    out = {"harness": src["name"], "entry": hit[1], "url": src["url"], "commit": src["commit"]}
    if lane == "gbdt-symmetric-1000":
        out["trees_source"] = CATBOOST_DEFAULTS["url"]
    return out


if __name__ == "__main__":
    json.dump({"sources": [CUML_BENCH, GBM_BENCH, CATBOOST_DEFAULTS], "gbm": GBM, "cuml": CUML,
               "covered": COVERED, "not_covered": NOT_COVERED, "pinned": PINNED},
              sys.stdout, indent=1)
    sys.stdout.write("\n")
