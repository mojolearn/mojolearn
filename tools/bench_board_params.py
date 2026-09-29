# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BENCH BOARD'S PARAMETER CHECK: same seed, same tuning parameters, enforced.

Andrew (2026-09-29): "they need to be comparable same seed same tuning params".
Every board driver (trees, classical, classical2, neural, algos) calls
`enforce(...)` once per race, after it has CONSTRUCTED every arm and before the
first timed round:

    import bench_board_params as BP
    report = BP.enforce("rf", {"ours": est, "ours-fast": est_fast,
                               "sklearn-rf-cpu": skl, "lightgbm-cpu": lgb_model},
                        family="trees")        # prints BOARD-PARAMS, raises on a mismatch

A driver whose arms live in separate worker processes has each worker send
`BP.arm_record(obj)` for the object it constructed, and the conductor calls
`enforce` with those records. The arm names are the board's cell names
(`ours`, `ours-fast` / `ours-ab`, the opponent arm names).

1. It READS BACK what each arm really got, from the constructed object:
   `get_params()` for scikit-learn-shaped estimators (scikit-learn, ours, the
   XGBoost / LightGBM / CatBoost scikit-learn wrappers, umap-learn), the
   optimizer's `defaults` for a torch optimizer, the scalar attributes for a
   torch module, the params dict for a raw XGBoost / LightGBM booster
   config. A plain dict is accepted only for an arm that is a function call
   (statsmodels, faiss, NumPy), and the report says `source: declared`.
2. It maps names that mean the same thing across libraries to one canonical
   name through ONE table, `ALIASES` (with value transforms, e.g. CatBoost's
   `border_count` is `max_bin - 1`).
3. It REFUSES the race by name (`ParamsRefused`, and a `BOARD-PARAMS-REFUSED`
   line) when an arm's seed differs from ours, or when a canonical parameter
   that both our arm and that arm have differs, or is set on one side and
   left to the library default (None) on the other, unless that exact
   (lane, parameter, arm) is listed in `EXCEPTIONS` with its reason.
   THE SEED. Every arm that has a seed parameter must hold the lane's seed
   (`seed_for(lane)`: 7, or the NVIDIA harness's own where it sets one,
   `LANE_SEED`); anything else refuses unless EXCEPTIONS names it. An arm
   whose constructor has NO seed parameter at all (no seed name in its
   read-back, and for a constructed object none in its constructor's
   signature either) draws nothing and is recorded as
   `seed: none (deterministic)` without an EXCEPTIONS entry. A third-party
   arm that DOES draw random numbers but has no seed argument is the one
   case EXCEPTIONS keeps for the seed: its row's reason is recorded instead.
   Execution-only settings (threads, device, verbosity, numeric mode) are in
   `IGNORE` and never compared.
4. It prints one `BOARD-PARAMS <json>` line with every arm's resolved
   canonical parameters, what was compared, and the exceptions applied; the
   board copies it into the race's cells and renders it in BOARD.md.

The reference arm is `ours` (IDENTICAL); every other arm, our FAST arm
included, is compared with it.

    python3 tools/bench_board_params.py table     # print ALIASES, IGNORE, EXCEPTIONS
"""
import fnmatch
import inspect
import json
import math
import sys

SEED = 7

#: The NVIDIA harness's own seed, where it sets one, for the lanes it covers
#: (tools/bench_board_harness.py records the source and commit); every other
#: lane uses SEED.
LANE_SEED = {
    # cuML benchmark algorithms.py SpectralClustering shared_args random_state=42
    "spectral": 42,
    # cuML benchmark algorithms.py TargetEncoder cpu_args random_state=42, cuml_args seed=42
    "algos/target-encoder": 42,
}


def seed_for(lane):
    """The seed every arm of `lane` is given."""
    return LANE_SEED.get(lane, SEED)
MARK = "BOARD-PARAMS"
MARK_REFUSED = "BOARD-PARAMS-REFUSED"

# ---------------------------------------------------------------------------
# ONE alias table. library -> {library's name: canonical name or
# (canonical name, transform to the canonical value)}. "*" applies to every
# library unless the library's own table names the parameter.
# ---------------------------------------------------------------------------


def _plus1(v):
    return None if v is None else v + 1


def _unit_if_none(v):
    """scale_pos_weight: None is every library's documented unit weight 1."""
    return 1.0 if v is None else v


def _pos_over_neg(w):
    """Ours' class_weights [w0, w1] as scale_pos_weight w1 / w0 (the weight
    CatBoost, XGBoost and LightGBM put on the positive class); None is unit
    weights, 1.0. More than two classes is not a scale_pos_weight."""
    if w is None:
        return 1.0
    w = list(w)
    return w[1] / w[0] if len(w) == 2 and w[0] else repr(w)


ALIASES = {
    "*": {
        "random_state": "seed", "seed": "seed", "random_seed": "seed", "rng_seed": "seed",
        "n_estimators": "n_estimators", "num_boost_round": "n_estimators",
        "iterations": "n_estimators", "num_iterations": "n_estimators",
        "num_trees": "n_estimators", "n_trees": "n_estimators",
        "max_depth": "max_depth", "depth": "max_depth",
        "learning_rate": "learning_rate", "eta": "learning_rate", "lr": "learning_rate",
        "min_child_weight": "min_child_weight", "min_sum_hessian_in_leaf": "min_child_weight",
        "min_child_hessian": "min_child_weight",
        "max_bin": "max_bin", "max_bins": "max_bin",
        "reg_lambda": "reg_lambda", "l2_leaf_reg": "reg_lambda", "lambda_l2": "reg_lambda",
        "reg_alpha": "reg_alpha", "lambda_l1": "reg_alpha",
        "max_leaves": "max_leaves", "num_leaves": "max_leaves", "max_leaf_nodes": "max_leaves",
        "min_samples_leaf": "min_samples_leaf", "min_child_samples": "min_samples_leaf",
        "min_data_in_leaf": "min_samples_leaf",
        "min_split_gain": "min_split_gain", "gamma_split": "min_split_gain",
        "min_impurity_decrease": "min_split_gain",
        "subsample": "subsample", "bagging_fraction": "subsample",
        "colsample_bytree": "feature_fraction", "feature_fraction": "feature_fraction",
        "max_features": "max_features", "colsample_bynode": "feature_fraction_bynode",
        "feature_fraction_bynode": "feature_fraction_bynode",
        "grow_policy": "grow_policy", "bootstrap": "bootstrap", "criterion": "criterion",
        "max_samples": "max_samples", "contamination": "contamination",
        "n_clusters": "n_clusters", "n_components": "n_components", "n_init": "n_init",
        "init": "init", "max_iter": "max_iter", "tol": "tol", "algorithm": "algorithm",
        "C": "C", "alpha": "alpha", "l1_ratio": "l1_ratio", "kernel": "kernel",
        "degree": "degree", "coef0": "coef0", "epsilon": "epsilon",
        "fit_intercept": "fit_intercept", "penalty": "penalty", "loss": "loss",
        "solver": "solver", "whiten": "whiten", "svd_solver": "svd_solver",
        "n_neighbors": "n_neighbors", "metric": "metric", "p": "p", "weights": "weights",
        "min_dist": "min_dist", "spread": "spread", "n_epochs": "n_epochs",
        "eps": "eps", "min_samples": "min_samples", "min_cluster_size": "min_cluster_size",
        "bandwidth": "bandwidth", "linkage": "linkage", "affinity": "affinity",
        "covariance_type": "covariance_type", "reg_covar": "reg_covar",
        "n_iter": "n_iter", "perplexity": "perplexity", "early_exaggeration": "early_exaggeration",
        "betas": "betas", "weight_decay": "weight_decay", "momentum": "momentum",
        "dampening": "dampening", "nesterov": "nesterov", "amsgrad": "amsgrad",
        "hidden_size": "hidden_size", "num_layers": "num_layers", "dropout": "dropout",
        "batch_size": "batch_size", "epochs": "epochs", "shuffle": "shuffle",
        "nlist": "nlist", "nprobe": "nprobe", "n_lists": "nlist", "n_probes": "nprobe",
        "order": "order", "seasonal_order": "seasonal_order", "trend": "trend",
        "seasonal": "seasonal", "seasonal_periods": "seasonal_periods",
        "damped_trend": "damped_trend", "initialization_method": "initialization_method",
        "normalize": "normalize", "positive": "positive", "class_weight": "class_weight",
        "leaf_size": "leaf_size", "boosting_type": "boosting_type",
        "border_count": ("max_bin", _plus1),
        # classical and classical2 (tools/classical_two_datasets.py, tools/bench_board_more.py)
        "gamma": "gamma", "atol": "atol", "rtol": "rtol", "breadth_first": "breadth_first",
        "cluster_selection_method": "cluster_selection_method",
        "cluster_selection_epsilon": "cluster_selection_epsilon",
        "max_cluster_size": "max_cluster_size", "allow_single_cluster": "allow_single_cluster",
        "selection": "selection", "precompute": "precompute", "init_params": "init_params",
        "n_restarts_optimizer": "n_restarts_optimizer", "max_iter_predict": "max_iter_predict",
        "maxiter": "max_iter", "penalized_intercept": "penalized_intercept",
        "set_op_mix_ratio": "set_op_mix_ratio", "local_connectivity": "local_connectivity",
        "negative_sample_rate": "negative_sample_rate", "repulsion_strength": "repulsion_strength",
        "assign_labels": "assign_labels", "start_periods": "start_periods",
        # gbm-bench's binary classification weight (tools/bench_board_harness.py)
        "scale_pos_weight": ("scale_pos_weight", _unit_if_none),
        # the cuML benchmark's KMeans cuml_args, TargetEncoder and MBSGD values
        "oversampling_factor": "oversampling_factor", "smooth": "smooth", "cv": "cv",
        "eta0": "eta0",
    },
    # XGBoost's `gamma` is the minimum split loss reduction
    "xgboost": {"gamma": "min_split_gain", "max_bin": "max_bin", "booster": "boosting_type"},
    # LightGBM's `min_child_weight` is min_sum_hessian_in_leaf
    "lightgbm": {"min_child_weight": "min_child_weight", "boosting_type": "boosting_type"},
    # CatBoost and ours count borders, not bins: 254 borders = 255 bins
    "catboost": {"border_count": ("max_bin", _plus1), "boosting_type": "boosting_type",
                 # trees: the CatBoost knobs ours carries under the same name
                 "loss_function": "loss", "random_strength": "random_strength",
                 "score_function": "score_function", "bootstrap_type": "bootstrap_type",
                 "leaf_estimation_method": "leaf_estimation_method",
                 "leaf_estimation_iterations": "leaf_estimation_iterations",
                 "feature_border_type": "feature_border_type", "nan_mode": "nan_mode",
                 # Ordered boosting's knobs (read back as None when unset, below)
                 "permutation_count": "permutation_count",
                 "fold_len_multiplier": "fold_len_multiplier",
                 "fold_permutation_block": "fold_permutation_block"},
    "mojolearn": {"border_count": ("max_bin", _plus1), "boosting_type": "boosting_type",
                  "random_strength": "random_strength", "score_function": "score_function",
                  "bootstrap_type": "bootstrap_type",
                  "leaf_estimation_method": "leaf_estimation_method",
                  "leaf_estimation_iterations": "leaf_estimation_iterations",
                  "feature_border_type": "feature_border_type", "nan_mode": "nan_mode",
                  "permutation_count": "permutation_count",
                  "fold_len_multiplier": "fold_len_multiplier",
                  "fold_permutation_block": "fold_permutation_block",
                  # the RF quantile bin count (cuML's n_bins) is a bin count
                  "n_bins": "max_bin",
                  # GradientBoosting's class weights as the positive-class weight
                  "class_weights": ("scale_pos_weight", _pos_over_neg)},
    "cuml": {"n_bins": "max_bin", "n_folds": "cv"},
    # torch: lr / betas / eps / weight_decay come from the optimizer's defaults
    "torch": {"lr": "learning_rate"},
}

#: Execution-only settings: never compared (they choose where and how verbosely
#: the same computation runs, not what it computes).
IGNORE = frozenset({
    "n_jobs", "nthread", "num_threads", "thread_count", "verbose", "verbosity", "silent",
    "logging_level", "device", "device_type", "task_type", "numeric_mode", "copy_X", "copy",
    "allow_writing_files", "train_dir", "importance_type", "callbacks", "warm_start",
    "used_ram_limit", "gpu_ram_part", "gpu_id", "devices", "tree_method", "predictor",
    "output_type", "handle", "compute_uv", "memory", "low_memory", "validate_parameters",
})

#: The ONLY accepted differences. Each entry: (lane glob, canonical parameter,
#: arm glob, reason). A difference not listed here refuses the race.
EXCEPTIONS = [
    # e.g. ("umap", "seed", "umap-learn-unseeded", "raced unseeded on purpose: seeded umap-learn runs one thread"),
    ("gbdt-ordered", "permutation_count", "catboost-*", "CatBoost's Python constructor does not "
     "take permutation_count (TypeError, 1.2.10), so their arm runs its default 4 "
     "(boosting_options.cpp:14); ours is given 4 explicitly"),
    # ---- trees (tools/speed_gbdt_arm.py lane_config; each also a FSPEED-NOTE mismatch line)
    ("gbdt-*", "subsample", "*", "no row sampling on any arm: ours and CatBoost bootstrap_type "
     "'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM "
     "subsample 1.0"),
    ("gbdt-*", "boosting_type", "xgboost-*", "different vocabularies: ours and CatBoost 'Plain' "
     "(not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting"),
    ("gbdt-*", "boosting_type", "lightgbm-*", "different vocabularies: ours and CatBoost 'Plain' "
     "(not Ordered), LightGBM 'gbdt'; both plain gradient boosting"),
    ("gbdt-*", "min_child_weight", "lightgbm-*", "LightGBM 4.7.0 aborts a boosted tree at "
     "min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor"),
    ("gbdt-*", "min_samples_leaf", "lightgbm-*", "LightGBM keeps min_child_samples 20 (it aborts at "
     "the other arms' value); ours and CatBoost min_data_in_leaf 1"),
    ("gbdt-depthwise", "min_child_weight", "*", "ours takes min_child_hessian only with a Newton "
     "score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian "
     "floor: XGBoost min_child_weight 0"),
    ("gbdt-lossguide", "score_function", "catboost-cpu", "CatBoost's CPU learner scores splits with "
     "Cosine or L2 only; ours and catboost-gpu NewtonL2, the Newton L2 gain of XGBoost and LightGBM"),
    ("gbdt-categorical", "score_function", "catboost-cpu", "CatBoost's CPU learner scores splits "
     "with Cosine or L2 only; ours and catboost-gpu NewtonL2, the Newton L2 gain of XGBoost and "
     "LightGBM"),
] + [
    (lane, "min_child_weight", "*", "ours takes min_child_hessian on Depthwise and Lossguide only; "
     "on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)")
    for lane in ("gbdt-symmetric*", "gbdt-rank-*", "gbdt-multiclass", "gbdt-ordered")
] + [
    (lane, "min_split_gain", "*", "ours takes min_split_gain on Depthwise and Lossguide only; on "
     "the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)")
    for lane in ("gbdt-symmetric*", "gbdt-rank-*", "gbdt-multiclass", "gbdt-ordered")
] + [
    (lane, "grow_policy", "xgboost-*", "ours and CatBoost fit this loss on the symmetric grower "
     "only; XGBoost has none and runs depthwise at the same depth")
    for lane in ("gbdt-rank-*", "gbdt-multiclass")
] + [
    ("rf", "max_leaves", "*", "no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), "
     "sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM "
     "num_leaves 65536 = 2 ** max_depth, never reached at depth 16"),
    ("et", "max_leaves", "*", "no leaf cap on any arm: ours and sklearn max_leaf_nodes None, "
     "LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16"),
    ("rf", "class_weight", "*", "None on every arm is unit class weights, the value each library "
     "defines for None"),
    ("et", "class_weight", "*", "None on every arm is unit class weights, the value each library "
     "defines for None"),
    ("et", "max_samples", "*", "bootstrap False on ours and sklearn: every row in every tree; "
     "sklearn refuses max_samples without bootstrap, so it stays None on both"),
    ("iforest", "max_depth", "*", "ours None is the auto depth ceil(log2(max_samples)) = 8; "
     "sklearn has no max_depth parameter and fixes the same value"),
    # --- classical (tools/classical_two_datasets.py) and classical2 (tools/bench_board_more.py).
    # An arm with no seed argument at all needs no row (check() records it as
    # `seed: none (deterministic)`). A seed row below names either an arm whose
    # constructor HAS a seed parameter left None on purpose, or a third-party
    # arm that draws random numbers and has no seed argument.
    ("svc", "seed", "ours*", "mojolearn SVC refuses random_state without probability=True (the fit "
     "draws nothing); scikit-learn and cuML get 7"),
    ("svc", "class_weight", "*", "None (unweighted) set explicitly on every arm"),
    ("dbscan", "algorithm", "sklearn-cpu*", "an exact eps search on every arm: ours 'rbc' (its "
     "default), scikit-learn has no 'rbc' and runs 'brute' (the cuML benchmark's cpu_args)"),
    ("dbscan", "algorithm", "cuml-gpu", "an exact eps search on every arm: ours 'rbc', cuml-gpu "
     "'brute' (cuml-gpu-rbc races 'rbc')"),
    ("hdbscan", "max_cluster_size", "sklearn-cpu*", "no limit on every arm: ours and cuML spell "
     "it 0, scikit-learn None"),
    ("umap", "seed", "umap-learn-cpu-unseeded", "raced unseeded on purpose: seeded umap-learn runs "
     "one thread (its rule); this arm is random_state=None, n_jobs=-1 (umap-learn-cpu has 7)"),
    ("logreg", "solver", "sklearn-cpu*", "L-BFGS on every arm: ours and cuML 'qn', scikit-learn 'lbfgs'"),
    ("logreg", "class_weight", "*", "None (unweighted) set explicitly on every arm"),
    ("logreg", "l1_ratio", "*", "penalty='l2' on every arm: ours None, scikit-learn None or 0.0 "
     "(its l2 spelling from 1.8)"),
    ("linearsvc", "class_weight", "*", "None (unweighted) set explicitly on every arm"),
    ("ridge", "solver", "sklearn-cpu*", "ours and cuML 'eig' (eigendecomposition of the normal "
     "equations); scikit-learn has no 'eig' and runs 'cholesky' on the same normal equations"),
    ("lasso", "seed", "ours*", "mojolearn Lasso refuses random_state: it selects nothing with "
     "selection='cyclic'"),
    ("elasticnet", "seed", "ours*", "mojolearn ElasticNet refuses random_state: it selects nothing "
     "with selection='cyclic'"),
    ("ivf", "seed", "cuvs-gpu", "cuVS ivf_flat IndexParams takes no seed and its k-means training "
     "samples rows; ours and faiss get 7"),
    ("tsvd", "algorithm", "sklearn-cpu*", "scikit-learn TruncatedSVD has no 'covariance_eigh'; it "
     "runs 'arpack' at tol=0"),
    ("tsvd", "algorithm", "cuml-gpu", "cuML TruncatedSVD has no 'covariance_eigh'; it runs 'full'"),
    ("spectral*", "gamma", "*", "affinity='nearest_neighbors' reads no gamma: ours refuses any "
     "value (None), scikit-learn holds its default (1.0 clustering, None embedding)"),
    ("spectral", "degree", "*", "affinity='nearest_neighbors' reads no degree: ours refuses any "
     "value (None), scikit-learn holds 3"),
    ("spectral", "coef0", "*", "affinity='nearest_neighbors' reads no coef0: ours refuses any "
     "value (None), scikit-learn holds 1"),
    ("gp[rc]", "kernel", "sklearn-cpu*", "the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + "
     "WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ"),
    ("gpc", "seed", "ours*", "mojolearn GaussianProcessClassifier refuses random_state "
     "(optimizer=None draws nothing); scikit-learn gets 7"),
    # ---- neural (tools/bench_board_neural.py; lane ids "neural/<lane>"): ours'
    # neural classes take no seed argument (recorded automatically); torch arms
    # call torch.manual_seed(7)
    # ---- algos (tools/bench_board_algos.py; lane ids "algos/<slug>"): the arms
    # with no seed argument that draw nothing (the deterministic lanes, ours'
    # weightless layers, SVGP) are recorded automatically; the third-party arms
    # below draw random numbers and take no seed, or draw through a seeded
    # function argument rather than a seed argument of their own
] + [
    ("algos/" + lane, "seed", "*", "SelectKBest takes no seed argument on either side; the "
     "noise seed 7 is bound into score_func=%s(random_state=7) on both arms" % fn)
    for lane, fn in (("select-mutual-info", "mutual_info_classif"),
                     ("select-mutual-info-reg", "mutual_info_regression"))
] + [
    ("algos/louvain", "seed", "cugraph-gpu", "cuGraph louvain takes no seed; its GPU move order "
     "is not seeded; ours and networkx get 7"),
    ("algos/cagra", "seed", "faiss-cpu", "faiss IndexHNSWFlat takes no seed argument (its level "
     "draw uses faiss' fixed internal seed)"),
] + [
    ("algos/" + lane, "seed", "cuvs-gpu", "cuVS IndexParams take no seed and the index build "
     "samples rows; ours and faiss get 7")
    for lane in ("ivf-pq", "ivf-sq", "ivf-refine", "cagra")
] + [
    ("algos/dart", "max_leaves", "xgboost-*", "XGBoost DART grows depth-wise (max_depth 8, no "
     "leaf cap); ours and LightGBM leaf-wise with num_leaves 255"),
    ("algos/dart-reg", "max_leaves", "xgboost-*", "XGBoost DART grows depth-wise (max_depth 8, "
     "no leaf cap); ours and LightGBM leaf-wise with num_leaves 255"),
    ("algos/mds", "metric", "sklearn-cpu", "the same metric MDS on Euclidean distances, spelled "
     "differently: ours metric='euclidean' (scikit-learn 1.9's name), the pinned scikit-learn "
     "1.7.2 metric=True with dissimilarity='euclidean'"),
    ("algos/tsne", "init", "cuml-gpu", "cuML TSNE takes only init='random' and draws its own "
     "start; ours and scikit-learn start from the same seed-7 array"),
    ("algos/tsne", "learning_rate", "cuml-gpu", "cuML names its rate schedule "
     "learning_rate_method='adaptive'; ours and scikit-learn learning_rate='auto'"),
    ("algos/sgd-clf", "max_iter", "cuml-gpu", "cuML MBSGD reads epochs=100, not max_iter"),
    ("algos/sgd-clf", "tol", "cuml-gpu", "cuML MBSGD tol=0.0 is its no-early-stop value; ours "
     "and scikit-learn tol=None"),
    ("algos/sgd-reg", "max_iter", "cuml-gpu", "cuML MBSGD reads epochs=100, not max_iter"),
    ("algos/sgd-reg", "tol", "cuml-gpu", "cuML MBSGD tol=0.0 is its no-early-stop value; ours "
     "and scikit-learn tol=None"),
    ("algos/sgd-reg", "loss", "cuml-gpu", "cuML spells squared error 'squared_loss'"),
]


class ParamsRefused(RuntimeError):
    """The race's arms do not share the seed or a tuning parameter."""


# ---------------------------------------------------------------------------
# Reading back what an arm really got
# ---------------------------------------------------------------------------

def library_of(obj):
    mod = type(obj).__module__ or ""
    top = mod.split(".")[0]
    return {"sklearn": "sklearn", "xgboost": "xgboost", "lightgbm": "lightgbm",
            "catboost": "catboost", "mojolearn": "mojolearn", "torch": "torch",
            "umap": "umap-learn", "faiss": "faiss", "statsmodels": "statsmodels"}.get(top, top or "?")


def _scalar(v):
    return v is None or isinstance(v, (bool, int, float, str)) or (
        isinstance(v, (tuple, list)) and all(isinstance(x, (bool, int, float, str)) or x is None for x in v))


def read_params(obj):
    """(library, source, {name: value}) of what the constructed arm holds."""
    if isinstance(obj, dict) and obj.get("__record__"):
        # read back in another process (a worker) by arm_record()
        return obj["library"], obj["source"], dict(obj["params"])
    if isinstance(obj, dict):
        return obj.get("__library__", "declared"), "declared", {
            k: v for k, v in obj.items() if k != "__library__"}
    lib = library_of(obj)
    # torch optimizer: its defaults are the hyperparameters every group starts from
    defaults = getattr(obj, "defaults", None)
    if lib == "torch" and isinstance(defaults, dict) and hasattr(obj, "param_groups"):
        return lib, "optimizer.defaults", {k: v for k, v in defaults.items() if _scalar(v)}
    if lib == "catboost" and hasattr(obj, "get_params"):
        # CatBoost's get_params lists only what was set; a known name that is
        # absent is the library default and reads None
        got = dict(obj.get_params())
        for name in ("random_seed", "iterations", "depth", "learning_rate", "l2_leaf_reg",
                     "border_count", "min_data_in_leaf", "max_leaves", "grow_policy",
                     "bootstrap_type", "subsample", "random_strength", "boosting_type"):
            got.setdefault(name, None)
        if str(got.get("boosting_type")) == "Ordered":
            # the Ordered knobs, whose unset defaults differ between their CPU
            # and GPU (fold_permutation_block 1 vs 64): compared only there
            for name in ("permutation_count", "fold_len_multiplier",
                         "fold_permutation_block"):
                got.setdefault(name, None)
        return lib, "get_params", got
    if hasattr(obj, "get_params"):
        try:
            got = obj.get_params(deep=False)
        except TypeError:
            got = obj.get_params()
        if lib in ("xgboost", "lightgbm"):
            got = dict(got)
            kw = got.pop("kwargs", None)
            if isinstance(kw, dict):
                got.update(kw)
        return lib, "get_params", dict(got)
    if lib == "torch":
        return lib, "module attributes", {
            k: v for k, v in vars(obj).items() if not k.startswith("_") and _scalar(v)}
    # a constructor-signature object without get_params
    try:
        names = [p for p in inspect.signature(type(obj).__init__).parameters if p != "self"]
    except (TypeError, ValueError):
        names = []
    got = {n: getattr(obj, n) for n in names if hasattr(obj, n)}
    if not got:
        got = {k: v for k, v in vars(obj).items() if not k.startswith("_") and _scalar(v)}
    return lib, "attributes", got


def arm_record(obj):
    """What a WORKER process sends back to its conductor for one arm: the
    read-back of the object it constructed, JSON-safe. The conductor passes
    the records to enforce() in place of the objects."""
    lib, source, raw = read_params(obj)
    return {"__record__": True, "library": lib, "source": source,
            "params": {k: _jsonable(v) for k, v in raw.items() if k not in IGNORE}}


def canonical(lib, params):
    """{canonical name: (value, the library's own name)} for the aliased,
    non-ignored parameters."""
    own = ALIASES.get(lib, {})
    shared = ALIASES["*"]
    out = {}
    for name, value in params.items():
        if name in IGNORE:
            continue
        rule = own.get(name, shared.get(name))
        if rule is None:
            continue
        canon, fn = (rule, None) if isinstance(rule, str) else rule
        try:
            value = fn(value) if fn else value
        except TypeError:
            pass
        out[canon] = (_jsonable(value), name)
    return out


def _jsonable(v):
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return v
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    try:
        import numpy as np                              # noqa: PLC0415
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:
        pass
    if callable(v) or isinstance(v, type):
        return getattr(v, "__name__", repr(v))
    return repr(v)


def _equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


#: What the report records for an arm with no seed parameter at all.
NO_SEED_ARGUMENT = "none (deterministic)"
#: ... and for one that draws random numbers without a seed argument (an
#: EXCEPTIONS row names it and gives the reason).
NO_SEED_DRAWS = "none (no argument; draws random numbers, see exceptions)"


def _seed_names():
    """Every library spelling that maps to the canonical `seed`."""
    return {n for table in ALIASES.values() for n, r in table.items()
            if (r if isinstance(r, str) else r[0]) == "seed"}


def has_no_seed_argument(obj, raw):
    """True when the arm has no seed parameter at all: none of the seed
    spellings is in its read-back `raw` and, for a constructed object, none
    is in its constructor's signature (a constructor that takes one but whose
    read-back lost it is NOT this case; the check refuses it by name). A
    worker's record and a declared dict are what was really constructed or
    passed, so their read-back is the whole answer."""
    names = _seed_names()
    if any(n in raw for n in names):
        return False
    if isinstance(obj, dict):
        return True
    try:
        params = inspect.signature(type(obj).__init__).parameters
    except (TypeError, ValueError):
        return False
    return not any(n in params for n in names)


def exception_for(lane, param, arm):
    for lg, p, ag, why in EXCEPTIONS:
        if p == param and fnmatch.fnmatch(lane, lg) and fnmatch.fnmatch(arm, ag):
            return why
    return None


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def check(lane, arms, family=None, reference="ours", seed=None, extra_exceptions=()):
    """The report for one race. `arms`: {arm name: constructed object or
    declared dict}. `extra_exceptions`: (param, arm glob, reason) the driver
    adds for this race (the same shape as EXCEPTIONS without the lane).
    `seed`: None is `seed_for(lane)`."""
    seed = seed_for(lane) if seed is None else seed
    resolved, sources, libs, no_seed, draws = {}, {}, {}, set(), set()
    for name, obj in arms.items():
        lib, source, raw = read_params(obj)
        libs[name], sources[name] = lib, source
        resolved[name] = canonical(lib, raw)
        if "seed" not in resolved[name] and has_no_seed_argument(obj, raw):
            no_seed.add(name)
    ref = reference if reference in resolved else next(iter(resolved), None)
    problems, compared, applied = [], [], []

    def _excused(param, arm):
        why = exception_for(lane, param, arm)
        if why is None:
            for p, ag, r in extra_exceptions:
                if p == param and fnmatch.fnmatch(arm, ag):
                    why = r
                    break
        return why

    for name, canon in resolved.items():
        # the seed: every arm, ours included, must carry the board's seed
        if "seed" in canon:
            got = canon["seed"][0]
            if got is None or not _equal(got, seed):
                why = _excused("seed", name)
                if why:
                    applied.append({"arm": name, "param": "seed", "value": got, "reason": why})
                else:
                    problems.append("%s: seed is %r (%s), the board's seed is %d"
                                    % (name, got, canon["seed"][1], seed))
        elif name in no_seed:
            # no seed argument: deterministic, unless a row says it draws
            why = _excused("seed", name)
            if why:
                draws.add(name)
                applied.append({"arm": name, "param": "seed", "value": None, "reason": why})
        else:
            # the constructor takes a seed but the read-back lost it
            why = _excused("seed", name)
            if why:
                applied.append({"arm": name, "param": "seed", "value": None, "reason": why})
            else:
                problems.append("%s (%s): its constructor takes a seed but none was read back"
                                % (name, libs[name]))
        if name == ref:
            continue
        for param, (val, own_name) in sorted(canon.items()):
            if param == "seed" or param not in resolved[ref]:
                continue
            rval, rname = resolved[ref][param]
            same = _equal(val, rval) and not (val is None and rval is None)
            compared.append({"arm": name, "param": param, "ours": rval, "theirs": val, "same": same})
            if same:
                continue
            why = _excused(param, name)
            if why:
                applied.append({"arm": name, "param": param, "ours": rval, "theirs": val, "reason": why})
                continue
            if val is None or rval is None:
                problems.append("%s: %s is %s on %s and %s on %s (a library default is not a "
                                "matched value; set it explicitly on both)"
                                % (name, param, "unset" if rval is None else repr(rval), ref,
                                   "unset" if val is None else repr(val), name))
            else:
                problems.append("%s: %s is %r (%s) on %s and %r (%s) on %s"
                                % (name, param, rval, rname, ref, val, own_name, name))
    return {"lane": lane, "family": family, "reference": ref, "seed": seed,
            "arms": {n: {"library": libs[n], "source": sources[n],
                         "params": dict(sorted(
                             [(p, v) for p, (v, _) in c.items()]
                             + ([("seed", NO_SEED_DRAWS if n in draws else NO_SEED_ARGUMENT)]
                                if n in no_seed else [])))}
                     for n, c in resolved.items()},
            "seed_none_deterministic": sorted(no_seed - draws),
            "compared": compared, "exceptions": applied, "problems": problems,
            "verdict": "REFUSED" if problems else "MATCHED"}


def emit(report, stream=None):
    stream = stream or sys.stdout
    stream.write("%s %s\n" % (MARK, json.dumps(report, sort_keys=True, default=repr)))
    if report["problems"]:
        stream.write("%s lane=%s reason=%s\n" % (MARK_REFUSED, report["lane"],
                                                  " | ".join(report["problems"])[:2000]))
    stream.flush()


def enforce(lane, arms, family=None, reference="ours", seed=None, extra_exceptions=(), stream=None):
    """check + emit; raises ParamsRefused when the arms do not match."""
    report = check(lane, arms, family=family, reference=reference, seed=seed,
                   extra_exceptions=extra_exceptions)
    emit(report, stream)
    if report["problems"]:
        raise ParamsRefused("parameters do not match (%d): %s"
                            % (len(report["problems"]), "; ".join(report["problems"])))
    return report


def parse_lines(text):
    """[report] from a driver log (the board's side)."""
    out = []
    for line in text.splitlines():
        if line.startswith(MARK + " {"):
            try:
                out.append(json.loads(line[len(MARK) + 1:]))
            except ValueError:
                pass
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        print(json.dumps({"ALIASES": {k: {n: (r if isinstance(r, str) else r[0] + " (transformed)")
                                          for n, r in v.items()} for k, v in ALIASES.items()},
                          "IGNORE": sorted(IGNORE), "LANE_SEED": LANE_SEED,
                          "EXCEPTIONS": EXCEPTIONS}, indent=1))
    else:
        print(__doc__)
