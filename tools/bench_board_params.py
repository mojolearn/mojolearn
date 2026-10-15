# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BENCH BOARD'S PARAMETER CHECK: same seed, same tuning parameters, enforced.

The rule (2026-09-29): races are comparable only at the same seed and the same
tuning parameters.
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
   differs or is missing from either arm's read-back. Unknown names are
   compared too. Library defaults (including None on both sides) need an exact
   (lane, parameter, arm) entry in `EXCEPTIONS` with its reason.
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
import os
import sys

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board_pairing  # noqa: E402  (smoke-fixes r13: the own-arm pairing table)

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


def _minus1(v):
    return None if v is None else v - 1


def _cuml_pca_solver(v):
    """cuML PCA's svd_solver 'full' is the eigendecomposition of the
    covariance matrix (cuml/decomposition/pca.pyx: 'auto' and 'full' both set
    solver.COV_EIG_DQ), scikit-learn's and ours' 'covariance_eigh'; its own
    scikit-learn interop maps any scikit-learn solver but 'auto' to 'full'.
    'jacobi' (COV_EIG_JACOBI) is another algorithm and keeps its name."""
    return "covariance_eigh" if v == "full" else v


def _cuml_split_criterion(v):
    """cuML RandomForest split_criterion as scikit-learn's criterion name (cuML 26.08
    ensemble/randomforest_common.pyx _split_criterion_to_criterion: gini/0 -> 'gini',
    entropy/1 -> 'entropy', mse/2 -> 'squared_error', poisson/4 -> 'poisson'). Any other value
    keeps its own spelling, so it is compared and refuses."""
    return {"gini": "gini", 0: "gini", "0": "gini", "entropy": "entropy", 1: "entropy",
            "1": "entropy", "mse": "squared_error", 2: "squared_error", "2": "squared_error",
            "poisson": "poisson", 4: "poisson", "4": "poisson"}.get(v, v)


def _hw_component(v):
    """Holt-Winters component names: cuML and statsmodels also spell
    'additive'/'multiplicative' as 'add'/'mul' (the same model)."""
    return {"add": "additive", "mul": "multiplicative"}.get(v, v)


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
        "feature_range": "feature_range", "clip": "clip",
        "solver": "solver", "whiten": "whiten", "svd_solver": "svd_solver",
        "n_neighbors": "n_neighbors", "metric": "metric", "p": "p", "weights": "weights",
        "min_dist": "min_dist", "spread": "spread", "n_epochs": "n_epochs",
        "eps": "eps", "min_samples": "min_samples", "min_cluster_size": "min_cluster_size",
        "bandwidth": "bandwidth", "linkage": "linkage", "affinity": "affinity",
        "covariance_type": "covariance_type", "reg_covar": "reg_covar",
        "n_iter": "n_iter", "perplexity": "perplexity", "early_exaggeration": "early_exaggeration",
        # release-0845: ours' TSNE option (python/mojolearn/_expansion_ann.py TSNE) is the factor
        # compared as post_exaggeration_rate_factor (cuML's is derived in _read_params_object)
        "post_exaggeration_learning_rate_factor": "post_exaggeration_rate_factor",
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
        # FAIRNESS FIX (2026-10-13 audit): neural Mamba blocks (tools/bench_board_neural.py): ours'
        # profile constants and the mamba-ssm arms' constructed modules, the ssm_ prefix so no other
        # library's get_params can collide. Results raced before this change were wrong for this
        # lane; apply extra scrutiny.
        "ssm_d_state": "ssm_d_state", "ssm_d_conv": "ssm_d_conv", "ssm_expand": "ssm_expand",
        "ssm_dt_rank": "ssm_dt_rank", "ssm_headdim": "ssm_headdim", "ssm_ngroups": "ssm_ngroups",
        "ssm_chunk_size": "ssm_chunk_size", "ssm_dt_limit": "ssm_dt_limit",
        "ssm_rope_angles": "ssm_rope_angles",
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
        # FAIRNESS FIX (2026-10-13 audit): PageRank's stopping threshold, the L1
        # change below which every arm stops (n x tol), declared by every graph
        # arm of tools/bench_board_algos.py and compared here; before, cuGraph's
        # unscaled tol was not comparable. Results raced before this change were
        # wrong for this lane; apply extra scrutiny.
        "l1_stop_threshold": "l1_stop_threshold",
        # FAIRNESS FIX (2026-10-13 audit): the spectral eigensolver tolerance, pinned on every
        # arm of classical2 spectral and spectral-embedding (it was each library's default, and
        # never compared). Results raced before this change were wrong for those lanes; apply
        # extra scrutiny.
        "eigen_tol": "eigen_tol",
        # gbm-bench's binary classification weight (tools/bench_board_harness.py)
        "scale_pos_weight": ("scale_pos_weight", _unit_if_none),
        # the cuML benchmark's KMeans cuml_args, TargetEncoder and MBSGD values
        "oversampling_factor": "oversampling_factor", "smooth": "smooth", "cv": "cv",
        "eta0": "eta0",
        # FAIRNESS FIX (2026-10-13 audit): the naive Bayes settings were never compared (cuML ran
        # its defaults unchecked); they are now read back on every arm that has them. Results
        # raced before this change were wrong for those lanes; apply extra scrutiny.
        "fit_prior": "fit_prior", "binarize": "binarize", "var_smoothing": "var_smoothing",
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
                 "fold_permutation_block": "fold_permutation_block",
                 # smoke-fixes (2026-10-14; gbdt-symmetric/-1000, gbdt-multiclass, gbdt-ordered:
                 # "feature_fraction, border_build_max_samples missing from catboost-gpu
                 # read-back"): CatBoost's names for ours' per-tree feature fraction (rsm,
                 # oblivious_tree_options.cpp:20) and border-search row subsample
                 # (dev_max_subset_size_for_build_borders, binarization_options.cpp:22)
                 "rsm": "feature_fraction",
                 "dev_max_subset_size_for_build_borders": "border_build_max_samples"},
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
    "cuvs": {"init_method": "init"},
    # smoke-fixes (2026-10-14; decision-tree-clf/-reg: "split_criterion missing from ours read-back",
    # "criterion missing from cuml-gpu read-back"): cuML's forests name the criterion split_criterion
    # and spell squared error 'mse' (or the enum code); cuML 26.08 ensemble/randomforest_common.pyx
    # _split_criterion_to_criterion is its own map to scikit-learn's criterion names, used here so
    # the VALUE is compared under one name.
    "cuml": {"split_criterion": ("criterion", _cuml_split_criterion),
             "n_bins": "max_bin", "n_folds": "cv",
             "svd_solver": ("svd_solver", _cuml_pca_solver),
             "seasonal": ("seasonal", _hw_component)},
    "statsmodels": {"seasonal": ("seasonal", _hw_component), "trend": ("trend", _hw_component)},
    # torch: lr / betas / eps / weight_decay come from the optimizer's defaults
    "torch": {"lr": "learning_rate"},
    # scikit-learn's HDBSCAN counts the point itself in min_samples
    # (hdbscan.py: kneighbors(X, min_samples)[:, -1]; its Notes: "must be 1
    # greater than ... scikit-learn-contrib/hdbscan"). Ours and cuML's do not
    # (cuML runner.h:68-80 runs the k-NN at min_samples + 1). The canonical
    # value is the count WITHOUT the point, so scikit-learn's 11 reads 10 and
    # a real mismatch (scikit-learn 10 against ours 10) still refuses.
    "sklearn/HDBSCAN": {"min_samples": ("min_samples", _minus1)},
}

#: Execution-only settings: never compared (they choose where and how verbosely
#: the same computation runs, not what it computes).
IGNORE = frozenset({
    "n_jobs", "nthread", "num_threads", "thread_count", "verbose", "verbosity", "silent",
    "logging_level", "device", "device_type", "task_type", "numeric_mode",
    "allow_writing_files", "train_dir", "importance_type",
    "used_ram_limit", "gpu_ram_part", "gpu_id", "devices",
    "output_type", "handle", "validate_parameters",
})

#: The ONLY accepted differences. Each entry: (lane glob, canonical parameter,
#: arm glob, reason). A difference not listed here refuses the race.
EXCEPTIONS = [
    # e.g. ("umap", "seed", "umap-learn-unseeded", "raced unseeded on purpose: seeded umap-learn runs one thread"),
    ("gbdt-ordered", "permutation_count", "catboost-*", "CatBoost's Python constructor does not "
     "take permutation_count (TypeError, 1.2.10), so their arm runs its default 4 "
     "(boosting_options.cpp:14); ours is given 4 explicitly"),
    # ---- classical
    ("pca", "tol", "cuml-gpu", "cuML PCA's tol is read only by its 'jacobi' solver "
     "(cuml/decomposition/pca.pyx: 'Used if algorithm = \"jacobi\"'); the raced 'full' solver, "
     "COV_EIG_DQ, the covariance eigendecomposition of ours' covariance_eigh, has no tolerance. "
     "cuML's own scikit-learn interop maps scikit-learn's tol 0.0 to its 1e-7"),
    # ---- trees (tools/speed_gbdt_arm.py lane_config; each also a FSPEED-NOTE mismatch line)
    ("gbdt-*", "subsample", "*", "no row sampling on any arm: ours and CatBoost bootstrap_type "
     "'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM "
     "subsample 1.0"),
    ("gbdt-*", "boosting_type", "xgboost-*", "different vocabularies: ours and CatBoost 'Plain' "
     "(not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting"),
    # FAIRNESS FIX (2026-10-13 audit, second pass): three LightGBM rows removed here
    # (boosting_type, min_child_weight and min_samples_leaf on gbdt-*). They excused LightGBM
    # racing min_child_samples 20 and min_child_weight 1e-3 against every other arm's 1 and 0,
    # another stopping rule; every boosted LightGBM arm is now refused by name
    # (tools/speed_gbdt_arm.py lightgbm_arms, lightgbm_rank_arms). Results raced before this
    # change were wrong for these lanes; apply extra scrutiny.
    # FAIRNESS FIX (2026-10-13 audit): three rows removed here. ("gbdt-depthwise",
    # min_child_weight, *) excused ours' missing hessian floor while Depthwise ran Cosine on ours
    # against XGBoost's Newton L2 gain; Depthwise now scores NewtonL2 and ours takes
    # min_child_hessian 0.0, XGBoost's min_child_weight 0. ("gbdt-lossguide" and
    # "gbdt-categorical", score_function, catboost-cpu) excused CatBoost's CPU learner racing
    # Cosine against ours' NewtonL2; that arm is now refused by name where it lacks the lane's
    # score function (tools/speed_gbdt_arm.py _catboost_cpu_refusal), and gbdt-categorical
    # scores L2 on every arm. Results raced before this change were wrong for these lanes;
    # apply extra scrutiny.
] + [
    (lane, "min_child_weight", "*", "ours takes min_child_hessian on Depthwise and Lossguide only; "
     "on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)")
    for lane in ("gbdt-symmetric*", "gbdt-rank-*", "gbdt-multiclass", "gbdt-ordered")
] + [
    (lane, "min_split_gain", "*", "ours takes min_split_gain on Depthwise and Lossguide only; on "
     "the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)")
    for lane in ("gbdt-symmetric*", "gbdt-rank-*", "gbdt-multiclass", "gbdt-ordered")
] + [
    # FAIRNESS FIX (2026-10-13 audit, second pass): the (gbdt-rank-*, gbdt-multiclass,
    # grow_policy, xgboost-*) rows are removed: they excused XGBoost racing depthwise trees
    # against ours' symmetric ones, a different algorithm; those XGBoost arms are now refused by
    # name (tools/speed_gbdt_arm.py xgboost_arms, xgboost_rank_arms). Results raced before this
    # change were wrong for these lanes; apply extra scrutiny.

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
    # FAIRNESS FIX (2026-10-13 audit): two rows here excused ours' DBSCAN
    # algorithm='rbc' against 'brute' on cuml-gpu and scikit-learn, a different
    # neighbour search. Ours now races 'brute' (tools/classical_two_datasets.py
    # OursDBSCAN), so the rows are gone and a differing algorithm refuses the
    # race. Results raced before this change were wrong for this lane; apply
    # extra scrutiny.
    # FAIRNESS FIX (2026-10-13 audit): k-means stopping rules differed (ours and
    # cuML an absolute squared center shift < tol, scikit-learn tol relative to
    # the mean feature variance). scikit-learn now gets tol / mean(var(X)), the
    # same absolute threshold, so its tol VALUE differs by design and is excused
    # here. Results raced before this change were wrong for this lane; apply
    # extra scrutiny.
    ("kmeans", "tol", "sklearn-cpu*", "the same stopping rule on every arm: total squared center "
     "shift below 1e-7 (ours and cuML read tol as that absolute threshold, cuVS kmeans.cuh "
     "sqrdNormError < tol); scikit-learn reads tol relative to mean(var(X, axis=0)) "
     "(_kmeans.py _tolerance), so it is given 1e-7 / mean(var(X)) of the same rows"),
    ("hdbscan", "max_cluster_size", "sklearn-cpu*", "no limit on every arm: ours and cuML spell "
     "it 0, scikit-learn None"),
    ("umap", "seed", "umap-learn-cpu-unseeded", "raced unseeded on purpose: seeded umap-learn runs "
     "one thread (its rule); this arm is random_state=None, n_jobs=-1 (umap-learn-cpu has 7)"),
    ("logreg", "solver", "sklearn-cpu*", "L-BFGS on every arm: ours and cuML 'qn', scikit-learn 'lbfgs'"),
    ("logreg", "class_weight", "*", "None (unweighted) set explicitly on every arm"),
    ("logreg", "l1_ratio", "*", "penalty='l2' on every arm: ours None, scikit-learn None or 0.0 "
     "(its l2 spelling from 1.8)"),
    ("linearsvc", "class_weight", "*", "None (unweighted) set explicitly on every arm"),
    # FAIRNESS FIX (2026-10-13 audit): the old reason said ours ran 'eig' like cuML; ours' IDENTICAL
    # default is the Cholesky of the centered Gram, and cuML is refused on ridge. Results raced
    # before this change were wrong for this lane; apply extra scrutiny.
    ("ridge", "solver", "sklearn-cpu*", "ours is constructed solver='eig' (the only name it takes "
     "besides 'auto') but its IDENTICAL fit solves the centered normal equations by Cholesky "
     "(glm/impl/gram_solve.mojo; eigendecomposition only when its trust gate rejects the factor); "
     "scikit-learn 'cholesky' is that algorithm"),
    ("lasso", "seed", "ours*", "mojolearn Lasso refuses random_state: it selects nothing with "
     "selection='cyclic'"),
    ("elasticnet", "seed", "ours*", "mojolearn ElasticNet refuses random_state: it selects nothing "
     "with selection='cyclic'"),
    # FAIRNESS FIX (2026-10-13 audit): the race declared precompute=False on every arm while ours'
    # coordinate descent sweeps the Gram; scikit-learn now races precompute=True. Results raced
    # before this change were wrong for these lanes; apply extra scrutiny.
    ("lasso", "precompute", "sklearn-cpu*", "ours takes only precompute=False but its coordinate "
     "descent forms X^T X once and sweeps the Gram (solver/impl/cd_gram_rule.mojo); scikit-learn "
     "precompute=True is that algorithm"),
    ("elasticnet", "precompute", "sklearn-cpu*", "ours takes only precompute=False but its "
     "coordinate descent forms X^T X once and sweeps the Gram (solver/impl/cd_gram_rule.mojo); "
     "scikit-learn precompute=True is that algorithm"),
    ("ivf", "seed", "cuvs-gpu", "cuVS ivf_flat IndexParams takes no seed and its k-means training "
     "samples rows; ours and faiss get 7"),
    # FAIRNESS FIX (2026-10-13 audit): hipVS (AMD's cuVS port) replaces the harness-written torch
    # arms on AMD; its Python API takes no seed, as cuVS's. Results raced before this change were
    # wrong for these lanes; apply extra scrutiny.
    ("ivf", "seed", "hipvs-gpu", "hipVS (AMD's cuVS port) ivf_flat IndexParams takes no seed and its "
     "k-means training samples rows; ours and faiss get 7"),
    ("kmeans", "seed", "hipvs-gpu", "hipVS (AMD's cuVS port) cuvs.cluster.kmeans KMeansParams takes "
     "no seed: its k-means++ draws from cuVS's own generator; ours, cuML and scikit-learn get 7"),
    ("tsvd", "algorithm", "sklearn-cpu*", "scikit-learn TruncatedSVD has no 'covariance_eigh'; it "
     "runs 'arpack' at tol=0"),
    ("tsvd", "algorithm", "cuml-gpu", "cuML TruncatedSVD has no 'covariance_eigh'; it runs 'full'"),
    ("tsvd", "n_iter", "cuml-gpu", "cuML TruncatedSVD's n_iter is read only by its 'jacobi' solver "
     "(cuml/decomposition/tsvd.pyx: 'Used in Jacobi solver'); the raced 'full' solver, COV_EIG_DQ, "
     "a covariance eigendecomposition, iterates nothing. cuML's own scikit-learn interop maps n_iter "
     "5 to its 15"),
    ("tsvd", "tol", "cuml-gpu", "cuML TruncatedSVD's tol is read only by its 'jacobi' solver "
     "(cuml/decomposition/tsvd.pyx: 'Used if algorithm = \"jacobi\"'); the raced 'full' solver "
     "has no tolerance. cuML's own scikit-learn interop maps tol 0.0 to its 1e-7"),
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
    ("algos/" + lane, "seed", arm, "cuVS IndexParams take no seed and the index build "
     "samples rows; ours and faiss get 7")
    for lane in ("ivf-pq", "ivf-sq", "ivf-refine", "cagra")
    for arm in ("cuvs-gpu", "hipvs-gpu")    # hipvs-gpu: FAIRNESS FIX (2026-10-13 audit), hipVS on AMD
] + [
    # refused-param-mappings (2026-10-15): the unconditional ("algos/decision-tree-*", "max_leaves",
    # "cuml-gpu") row waived ANY leaf cap on either arm; it is now the value-conditional mapping in
    # semantic_exception (ours None <-> cuML -1 only).
    ("algos/dart", "max_leaves", "xgboost-*", "XGBoost DART grows depth-wise (max_depth 8, no "
     "leaf cap); ours and LightGBM leaf-wise with num_leaves 255"),
    ("algos/dart-reg", "max_leaves", "xgboost-*", "XGBoost DART grows depth-wise (max_depth 8, "
     "no leaf cap); ours and LightGBM leaf-wise with num_leaves 255"),
    ("algos/mds", "metric", "sklearn-cpu", "the same metric MDS on Euclidean distances, spelled "
     "differently: ours metric='euclidean' (scikit-learn 1.9's name), the pinned scikit-learn "
     "1.7.2 metric=True with dissimilarity='euclidean'"),
    ("algos/tsne", "init", "cuml-gpu", "cuML TSNE takes only init='random' and draws its own "
     "start; ours and scikit-learn start from the same seed-7 array"),
    # FAIRNESS FIX (2026-10-13 audit): cuML t-SNE ran method='fft' with
    # learning_rate_method='adaptive', which at n = 20,000 rewrites its rate to
    # n / 3 = 6,667, early_exaggeration to 24 and n_neighbors to 78
    # (t_sne.pyx _init_params), against ours' 'auto' 416.7, 12 and 91. cuML now
    # runs method='exact' with learning_rate_method='none' and learning_rate =
    # ours' resolved 'auto' value (tools/bench_board_algos.py _derived_params),
    # so the VALUE is ours' rate and only its spelling differs. Results raced
    # before this change were wrong for this lane; apply extra scrutiny.
    ("algos/tsne", "learning_rate", "cuml-gpu", "the same rate, spelled differently: ours and "
     "scikit-learn learning_rate='auto' (max(n / early_exaggeration / 4, 50)); cuML gets that "
     "value as a float with learning_rate_method='none' (its 'adaptive' rewrites the rate, the "
     "exaggeration and n_neighbors). cuML's non-adaptive schedule doubles the rate after the "
     "exaggeration phase (t_sne.pyx post_learning_rate = 2 x learning_rate), which its Python "
     "API cannot switch off; that doubling is compared separately as post_exaggeration_rate_factor "
     "(_read_params_object), and ours runs it too (post_exaggeration_learning_rate_factor=2.0)"),
    # FAIRNESS FIX (2026-10-13 audit): cuGraph pagerank stopped when the
    # unscaled L1 change fell below tol (1e-6) while ours and networkx stop below
    # n x tol (0.02 at 20,000 nodes), 20,000x tighter on cuGraph. cuGraph now
    # gets tol = n x tol; l1_stop_threshold (n x tol) is declared and compared
    # on every arm. Results raced before this change were wrong for this lane;
    # apply extra scrutiny.
    ("algos/pagerank", "tol", "cugraph-gpu", "the same stopping rule, spelled differently: ours "
     "and networkx stop when the L1 change of the scores is below n x tol (networkx pagerank: "
     "err < N * tol), cuGraph when it is below its tol (unscaled), so cuGraph is given n x tol; "
     "l1_stop_threshold is equal on every arm"),
    # smoke-fixes (2026-10-14; sgd-clf/-reg: "epochs missing from ours read-back"): the two
    # unconditional max_iter rows that stood here never fired (max_iter is absent from cuML's
    # read-back, not different); semantic_exception now maps epochs <-> max_iter by value.
    ("algos/sgd-clf", "tol", "cuml-gpu", "cuML MBSGD tol=0.0 is its no-early-stop value; ours "
     "and scikit-learn tol=None"),
    ("algos/sgd-reg", "tol", "cuml-gpu", "cuML MBSGD tol=0.0 is its no-early-stop value; ours "
     "and scikit-learn tol=None"),
    # refused-param-mappings (2026-10-15): the unconditional sgd-reg and qn-reg "loss" rows for
    # cuml-gpu waived ANY loss on either arm (ours 'huber' against cuML 'l2' matched); they are now
    # the value-conditional spellings in semantic_exception (ours 'squared_error' only).
    ("algos/qn-reg", "tol", "sklearn-cpu", "scikit-learn LinearRegression's tol is read only by "
     "the sparse lsqr path; on dense X it solves in closed form (scipy lstsq) and has no "
     "stopping tolerance"),
]


def _no_row_sampling(ours, theirs):
    """gbdt-* subsample: ours bootstrap 'No' with subsample unset, and the opponent either CatBoost
    bootstrap 'No' with subsample unset or a library whose subsample is exactly 1.0 (every row)."""
    if not (_r6_is(ours.get("bootstrap_type"), "No") and ours.get("subsample") is None):
        return False
    if _r6_is(theirs.get("bootstrap_type"), "No") and theirs.get("subsample") is None:
        return True
    return theirs.get("bootstrap_type") is None and _r6_is(theirs.get("subsample"), 1.0)


#: refused-param-mappings (2026-10-15): the values each EXCEPTIONS row was reviewed for. A row
#: listed here applies only when (ours, theirs) is one of its pairs, or its predicate over the two
#: arms' canonical settings holds; any other difference on that parameter is compared and refuses.
#: Before this, every row applied to ANY two values (e.g. "class_weight" None/None also waived
#: ours 'balanced' against cuML None; "boosting_type" Plain/gbtree also waived ours Ordered or
#: XGBoost dart). The pairs are the ones the rows fired on in the 0.8.45 smoke read-backs
#: (smoke-0845, NVIDIA f4c365b4c and AMD), the values each row's reason describes. Rows not listed
#: (seed rows and data-dependent values: tsne init/learning_rate, pagerank tol) keep their
#: source-reviewed reason only; tools/test_refused_param_mappings.py replays those read-backs.
EXCEPTION_VALUES = {
    ("gbdt-ordered", "permutation_count", "catboost-*"): [(4, None)],
    ("gbdt-*", "subsample", "*"): _no_row_sampling,
    ("gbdt-*", "boosting_type", "xgboost-*"): [("Plain", "gbtree")],
    ("pca", "tol", "cuml-gpu"): [(0.0, 1e-7)],
    ("tsvd", "algorithm", "cuml-gpu"): [("covariance_eigh", "full")],
    ("tsvd", "n_iter", "cuml-gpu"): [(5, 15)],
    ("tsvd", "tol", "cuml-gpu"): [(0.0, 1e-7)],
    ("algos/sgd-clf", "tol", "cuml-gpu"): [(None, 0.0)],
    ("algos/sgd-reg", "tol", "cuml-gpu"): [(None, 0.0)],
    ("iforest", "max_depth", "*"): [(None, None), (None, 8)],
    ("logreg", "l1_ratio", "*"): [(None, None), (None, 0.0)],
    ("linearsvc", "class_weight", "*"): [(None, None)],
    ("logreg", "class_weight", "*"): [(None, None)],
    ("svc", "class_weight", "*"): [(None, None)],
    ("rf", "class_weight", "*"): [(None, None)],
    ("et", "class_weight", "*"): [(None, None)],
}


def _row_values_hold(row, ours, theirs, param):
    """Whether an EXCEPTIONS row's reviewed values (EXCEPTION_VALUES) hold for this difference."""
    pins = EXCEPTION_VALUES.get(row)
    if pins is None:
        return True
    if ours is None or theirs is None:
        return False
    if callable(pins):
        return bool(pins(ours, theirs))
    o, t = ours.get(param), theirs.get(param)
    return any(_r6_is(o, po) and _r6_is(t, pt) for po, pt in pins)


class ParamsRefused(RuntimeError):
    """The race's arms do not share the seed or a tuning parameter."""


# ---------------------------------------------------------------------------
# Reading back what an arm really got
# ---------------------------------------------------------------------------

#: An ESTIMATOR whose parameter means something else than the same name in
#: the rest of its library gets its own ALIASES key, "<library>/<class>".
#: scikit-learn's HDBSCAN counts the point itself in min_samples; its DBSCAN
#: does too, but so do ours and cuML's DBSCAN, so only HDBSCAN is keyed.
ESTIMATOR_LIBRARY = {("sklearn", "HDBSCAN"): "sklearn/HDBSCAN"}


def library_of(obj):
    mod = type(obj).__module__ or ""
    top = mod.split(".")[0]
    lib = {"sklearn": "sklearn", "xgboost": "xgboost", "lightgbm": "lightgbm",
           "catboost": "catboost", "mojolearn": "mojolearn", "torch": "torch",
           "umap": "umap-learn", "faiss": "faiss", "statsmodels": "statsmodels"}.get(top, top or "?")
    return ESTIMATOR_LIBRARY.get((lib, type(obj).__name__), lib)


def _scalar(v):
    return v is None or isinstance(v, (bool, int, float, str)) or (
        isinstance(v, (tuple, list)) and all(isinstance(x, (bool, int, float, str)) or x is None for x in v))


def read_params(obj):
    """(library, source, {name: value}) of what the constructed arm holds.

    smoke-fixes r13 (2026-10-14): an estimator the trees harness gives a pre-binned block
    (tools/speed_gbdt_arm.py attach_input) carries that block's record, read back as `input_binning`
    and compared like any setting (both arms of an XGBoost pairing must hold the same bins)."""
    lib, source, got = _read_params_object(obj)
    tag = None if isinstance(obj, dict) else getattr(obj, "_board_input_binning", None)
    if tag is not None:
        got = dict(got, input_binning=tag)
        source += " + input_binning (the harness's input block, speed_gbdt_arm.attach_input)"
    return lib, source, got


def _read_params_object(obj):
    """read_params without the harness's input record."""
    if isinstance(obj, dict) and obj.get("__record__"):
        # read back in another process (a worker) by arm_record()
        return obj["library"], obj["source"], dict(obj.get("raw_params", obj["params"]))
    if isinstance(obj, dict):
        return obj.get("__library__", "declared"), "declared", {
            k: v for k, v in obj.items() if k not in ("__library__", "__implementation__")}
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
        if type(obj).__name__ == "TSNE" and lib in ("mojolearn", "cuml"):
            # smoke-fixes (2026-10-14; algos/tsne): the step size after the exaggeration phase, as a factor of
            # learning_rate, compared on both arms. cuML 26.08 under method='exact' always sets
            # post_learning_rate = 2 x learning_rate (python/cuml/cuml/manifold/t_sne.pyx:201-202; no Python
            # switch). This difference changes the work, so it is compared, never waived.
            # release-0845: ours has the option, post_exaggeration_learning_rate_factor (get_params, aliased
            # to post_exaggeration_rate_factor in ALIASES; x_ann/tsne_device.mojo tsne_fit_device applies it),
            # and the algos/tsne race pins it to cuML's 2.0 (tools/bench_board_algos.py). A wheel without the
            # option keeps learning_rate for every step: factor 1.0.
            got = dict(got)
            if lib == "mojolearn":
                if "post_exaggeration_learning_rate_factor" not in got:
                    got["post_exaggeration_rate_factor"] = 1.0
            elif str(got.get("method", "")).lower() == "exact":
                got["post_exaggeration_rate_factor"] = 2.0
        # FAIRNESS FIX (2026-10-13 audit): cuML 26.8's Binarizer and
        # Normalizer constructors store these attributes but inherit a
        # get_params list that omits them (_thirdparty/sklearn/preprocessing/
        # _data.py). Read the actual object; an absent attribute stays absent.
        # smoke-fixes (2026-10-14; simple-imputer: "add_indicator, missing_values missing from
        # cuml-gpu read-back"): cuML 26.08 SimpleImputer stores both in _BaseImputer.__init__
        # (_thirdparty/sklearn/preprocessing/_imputation.py:109-111) but its _get_param_names
        # (lines 261-267) lists only strategy, fill_value, verbose and copy. Same recovery.
        if lib == "cuml" and type(obj).__name__ in ("Binarizer", "Normalizer", "SimpleImputer"):
            names = {"Binarizer": ("threshold", "copy"), "Normalizer": ("norm", "copy"),
                     "SimpleImputer": ("missing_values", "add_indicator")}[type(obj).__name__]
            got = dict(got)
            recovered = [name for name in names if name not in got and hasattr(obj, name)]
            got.update({name: getattr(obj, name) for name in recovered})
            return lib, "get_params + constructor attributes " + ",".join(recovered), got
        if lib == "mojolearn" and type(obj).__name__ in ("DecisionTreeClassifier", "DecisionTreeRegressor"):
            # smoke-fixes (2026-10-14; decision-tree-clf/-reg: "n_estimators, bootstrap, max_samples,
            # oob_score, n_streams, max_batch_size missing from ours read-back"): ours' DecisionTree*
            # is its forest builder with settings fixed in __init__ (python/mojolearn/_expansion_trees.py:
            # n_estimators=1, bootstrap=False, n_streams=1; max_samples, oob_score and max_batch_size the
            # forest defaults) that get_params does not list. The object holds them in `_cfg`, the dict
            # its fit hands the binding (python/mojolearn/randomforest.py _RandomForestBase._fit_params);
            # read them from there. An absent key stays absent.
            cfg = getattr(obj, "_cfg", None) or {}
            names = ("n_estimators", "bootstrap", "max_samples", "oob_score", "n_streams", "max_batch_size")
            got = dict(got)
            recovered = [name for name in names if name not in got and name in cfg]
            got.update({name: cfg[name] for name in recovered})
            return lib, "get_params + forest settings _cfg " + ",".join(recovered), got
        if (lib == "mojolearn" and type(obj).__name__ == "TSNE" and "exaggeration_iter" not in got
                and isinstance(getattr(obj, "_EXPLORATION_MAX_ITER", None), int)):
            # smoke-fixes (2026-10-14; algos/tsne: "exaggeration_iter missing from ours read-back"):
            # ours' TSNE fixes its early-exaggeration phase in the class (python/mojolearn/
            # _expansion_ann.py TSNE._EXPLORATION_MAX_ITER = 250; fit runs min(it, max_iter) steps
            # with the exaggeration, as cuML clips exaggeration_iter to max_iter, t_sne.pyx:173).
            # get_params does not list it; read the object's own constant.
            return lib, "get_params + class constant _EXPLORATION_MAX_ITER", dict(
                got, exaggeration_iter=obj._EXPLORATION_MAX_ITER)
        if lib == "mojolearn" and type(obj).__name__ == "LogisticRegression":
            # smoke-fixes (2026-10-14; logreg: "lbfgs_memory, penalty_normalized missing from ours
            # read-back"): ours' LogisticRegression fixes both in __init__ (python/mojolearn/
            # linear_model.py:1445-1447, lbfgs_memory=5, penalty_normalized=True) and hands them to
            # the QN binding on every fit (linear_model.py:1530-1531), but get_params does not list
            # them. Read the object's values; an absent attribute stays absent.
            names = ("lbfgs_memory", "penalty_normalized")
            got = dict(got)
            recovered = [name for name in names if name not in got and hasattr(obj, name)]
            got.update({name: getattr(obj, name) for name in recovered})
            return lib, "get_params + fixed QN fields " + ",".join(recovered), got
        if lib == "mojolearn" and type(obj).__name__ in ("RandomForestClassifier", "RandomForestRegressor") \
                and "max_leaf_nodes" in got and got["max_leaf_nodes"] is None and "max_leaves" in got:
            # smoke-fixes (2026-10-14; rf: "conflicting aliases for max_leaves: max_leaf_nodes and
            # max_leaves", which left "no parameter records"): ours' forests carry cuML's leaf cap
            # as max_leaves and keep scikit-learn's max_leaf_nodes only to refuse it: any value but
            # None raises (python/mojolearn/randomforest.py _RandomForestBase.__init__,
            # _MAX_LEAF_NODES_WHY), so None is no setting and the cap is max_leaves alone (compared).
            got = dict(got)
            del got["max_leaf_nodes"]
            return lib, "get_params without the refusal-only max_leaf_nodes=None", got
        recurrent = _recurrent_effective_options(obj, lib, got)
        if recurrent is not None:
            # smoke-mappings (2026-10-15): ours' recurrent models (get_params from ParamsMixin)
            return lib, "get_params + " + recurrent[0], recurrent[1]
        if lib == "cuml" and type(obj).__name__ == "MinMaxScaler" and "clip" not in got:
            # The pinned cuML 26.08 transform only scales and translates. Its
            # documented out-of-range example returns 1.5, not 1.0:
            # https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.preprocessing.minmaxscaler/
            # Report the fixed semantic value, not an unsupported constructor
            # option. A future cuML read-back that includes clip takes priority.
            return lib, "get_params + fixed unclipped MinMaxScaler behavior", dict(got, clip=False)
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
    recurrent = _recurrent_effective_options(obj, lib, got)
    if recurrent is not None:
        return lib, "attributes + " + recurrent[0], recurrent[1]
    return lib, "attributes", got


def _recurrent_effective_options(obj, lib, got):
    """(source note, read-back) with ours' recurrent model's effective optimizer settings, or None.

    smoke-fixes (2026-10-14; algos/lstm-*/gru-*/rnn-*: "betas, eps, weight_decay missing from ours
    (ours-fast) read-back"): ours' recurrent models hand the trainer the optimizer's PyTorch
    defaults updated by optimizer_options (python/mojolearn/_x_sequence_rnn.py optimizer_arguments:
    o = dict(_OPT_DEFAULTS[name]); o.update(options); _RecurrentBase._fit passes its slots, the
    values the Mojo trainer receives, to rnn_fit). Read those effective values from the model's own
    module; optimizer_options stays as given. An unknown optimizer or option adds nothing (the
    trainer itself raises on them).
    smoke-mappings (2026-10-15; 12 smoke lines algos/{rnn,lstm,gru}-{clf,reg}: "betas missing from
    ours read-back", the same for eps): the recurrent models get get_params from ParamsMixin
    (python/mojolearn/_mode.py), so read_params took the get_params branch and returned before the
    attribute fallback that held this read-back. Both branches now call this helper."""
    if lib != "mojolearn" or type(obj).__name__ not in _OURS_RECURRENT or "optimizer" not in got:
        return None
    mod = sys.modules.get(type(obj).__module__)
    defaults = getattr(mod, "_OPT_DEFAULTS", {}).get(got["optimizer"])
    options = got.get("optimizer_options") or {}
    if not (isinstance(defaults, dict) and isinstance(options, dict) and set(options) <= set(defaults)):
        return None
    effective = dict(defaults, **options)
    recovered = sorted(n for n in effective if n not in got)
    got = dict(got)
    got.update({n: effective[n] for n in recovered})
    return "effective optimizer options " + ",".join(recovered), got


#: ours' recurrent estimators (python/mojolearn/_x_sequence_rnn.py), read by read_params above
_OURS_RECURRENT = frozenset({"%s%s" % (c, t) for c in ("LSTM", "GRU", "RNN") for t in ("Classifier", "Regressor")})


def arm_record(obj):
    """What a WORKER process sends back to its conductor for one arm: the
    read-back of the object it constructed, JSON-safe. The conductor passes
    the records to enforce() in place of the objects."""
    # FAIRNESS FIX (2026-10-13 audit): ignored and unrecognized settings were
    # discarded, making the recorded recipe incomplete. Keep all raw values;
    # canonical() alone decides which names can be compared across libraries.
    lib, source, raw = read_params(obj)
    return {"__record__": True, "library": lib, "source": source,
            "implementation": implementation_of(obj),
            "params": {k: _jsonable(v) for k, v in raw.items()}}


def implementation_of(obj):
    if isinstance(obj, dict):
        if obj.get("__record__"):
            return obj.get("implementation")
        fn = obj.get("__implementation__")
        if callable(fn):
            return {"module": fn.__module__, "name": fn.__qualname__}
        return None
    return {"module": type(obj).__module__, "name": type(obj).__qualname__}


def canonical(lib, params):
    """All work settings, including new/unknown names; only execution knobs are ignored."""
    own = ALIASES.get(lib, {})
    shared = ALIASES["*"]
    out = {}
    for name, value in params.items():
        if name in IGNORE:
            continue
        # FAIRNESS FIX (2026-10-13 audit): unknown names and one-sided settings
        # used to disappear. Keep them so an incomplete mapping refuses.
        rule = own.get(name, shared.get(name, name))
        canon, fn = (rule, None) if isinstance(rule, str) else rule
        try:
            value = fn(value) if fn else value
        except TypeError:
            pass
        value = _jsonable(value)
        if canon in out and not _equal(out[canon][0], value):
            raise ParamsRefused("conflicting aliases for %s: %s and %s" % (canon, out[canon][1], name))
        out[canon] = (value, name)
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
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
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


def exception_for(lane, param, arm, ours=None, theirs=None):
    """The EXCEPTIONS reason for this difference. `ours` / `theirs`: the two arms' canonical
    settings; a row with reviewed values (EXCEPTION_VALUES) applies only when they hold."""
    for lg, p, ag, why in EXCEPTIONS:
        if p == param and fnmatch.fnmatch(lane, lg) and fnmatch.fnmatch(arm, ag) \
                and _row_values_hold((lg, p, ag), ours, theirs, param):
            return why
    return None


#: Lanes whose torch arm constructs a torch.optim optimizer (algos optimizer lanes, without the "algos/" prefix,
#: and the neural train-step lanes with torch.optim.AdamW).
_TORCH_OPTIMIZER_LANES = frozenset({
    "sgd", "adam", "adamw", "rmsprop", "adagrad", "adamax", "nadam", "adafactor",
    "mlp-train-step", "lm-train-step", "samba-train-step", "lm-host-train-step"})
#: torch.optim classes that fix decoupled_weight_decay by class, by lane.
_TORCH_DECOUPLED_BY_CLASS = {"adam": False, "adamw": True, "mlp-train-step": True, "lm-train-step": True,
                             "samba-train-step": True, "lm-host-train-step": True}


def _semantic_r4(lane, param, arm, ours, theirs, library):
    """smoke-fixes sub-lane r4 (2026-10-14): source-reviewed, value-conditional mappings for
    algos/tsne (cuML TSNE settings ours' TSNE fixes in its code and does not take)."""
    # smoke-fixes (2026-10-14; algos/tsne: "<14 settings> missing from ours read-back"). Sources:
    # cuML 26.08 python/cuml/cuml/manifold/t_sne.pyx (TSNEParams fields :34-59, _init_params
    # :144-232), cpp/src/tsne/tsne_runner.cuh (k-NN, squaring, perplexity search :195-260),
    # cpp/src/tsne/exact_tsne.cuh (the EXACT loop :83-157); ours python/mojolearn/_expansion_ann.py
    # TSNE (takes n_components, perplexity, early_exaggeration, learning_rate, max_iter, init,
    # random_state), x_ann/tsne_core.mojo (:12-27 design; ts_knn_cell :104 squared distances;
    # bisection :156, 100 steps), x_ann/tsne_device.mojo (:869-870 exaggeration 1.0 and momentum
    # 0.5 -> 0.8 after the exploration steps). Each fires only for cuML on algos/tsne, only when
    # the setting is absent from ours, and only for the values that prove the same work.
    if lane != "algos/tsne" or arm != "cuml-gpu" or library != "cuml":
        return None
    if param in ours or param not in theirs:
        return None
    v = theirs[param]
    exact = theirs.get("method") == "exact"
    sq_euclid = ((theirs.get("metric") == "euclidean" and theirs.get("square_distances") is True)
                 or (theirs.get("metric") == "sqeuclidean" and theirs.get("square_distances") is False))
    if param == "method" and v == "exact":
        return ("cuML method='exact' (TSNE_ALGORITHM.EXACT, exact_tsne.cuh): attraction over the "
                "sparse k-NN P, exact all-pairs repulsion; ours' only algorithm (tsne_core.mojo)")
    if param == "angle" and exact:
        return ("angle is the Barnes-Hut theta; cuML's exact loop never reads it (exact_tsne.cuh "
                "has no theta); ours has no tree")
    if param == "late_exaggeration" and exact and _equal(v, 1.0):
        return ("cuML's exact loop sets the exaggeration to 1.0 after exaggeration_iter "
                "(exact_tsne.cuh:92-95; late_exaggeration is FFT only), ours too "
                "(tsne_device.mojo:869)")
    if param == "learning_rate_method" and exact and v in ("none", None):
        return ("no adaptive rewrite of rate, exaggeration or n_neighbors (t_sne.pyx:189: never "
                "under EXACT, and 'none' off); NOTE cuML's non-adaptive path still doubles the "
                "rate after exaggeration_iter (t_sne.pyx:201-202), ours keeps it: compared under "
                "learning_rate, not waived here")
    if param == "metric" and sq_euclid:
        return ("squared Euclidean k-NN distances on both: cuML metric='euclidean' then "
                "square_distances squares them (tsne_runner.cuh:207-217); ours ranks and weighs "
                "by squared distance (tsne_core.mojo:104)")
    if param == "square_distances" and sq_euclid:
        return "with metric='euclidean' it makes cuML's distances squared Euclidean, ours' (tsne_core.mojo:104)"
    if param == "metric_params" and v is None and theirs.get("metric") != "minkowski":
        return "metric_params is read only for minkowski's p (t_sne.pyx:232); unset and inactive"
    if param == "min_grad_norm" and _equal(v, 0.0):
        return ("min_grad_norm=0.0: cuML stops only when the gradient norm is < 0 "
                "(exact_tsne.cuh:146), never; ours has no early stop: both run exactly max_iter")
    if param == "n_iter_without_progress":
        return ("cuML stores n_iter_without_progress but never passes it to its solver "
                "(TSNEParams, t_sne.pyx:34-59, has no such field); no early stop on either arm")
    if (param == "n_neighbors" and isinstance(v, int) and not isinstance(v, bool)
            and isinstance(ours.get("perplexity"), (int, float))
            and v == math.floor(3.0 * float(ours["perplexity"])) + 1):
        return ("cuML n_neighbors equals ours' fixed rule k = floor(3 x perplexity) + 1 "
                "(tsne_core.mojo:12, clipped to n - 1 as cuML clips to n, t_sne.pyx:180); both "
                "exact brute-force k-NN")
    if param == "perplexity_max_iter" and v == 100:
        return ("100 perplexity bisection steps on both, tolerance 1e-5 on both (ours "
                "tsne_core.mojo:14,156; cuML params.perplexity_tol, t_sne.pyx:215)")
    if param == "pre_momentum" and _equal(v, 0.5):
        return "momentum 0.5 during the exaggeration phase on both (ours tsne_device.mojo:870)"
    if param == "post_momentum" and _equal(v, 0.8):
        return "momentum 0.8 after the exaggeration phase on both (ours tsne_device.mojo:870)"
    if param == "precomputed_knn" and v is None:
        return ("no supplied k-NN graph: cuML computes its own exact brute-force k-NN "
                "(distances.cuh get_distances, cuvs brute_force), ours its own exact k-NN")


# smoke-fixes (2026-10-14; batch 3 sub-lane r5): NVIDIA classical ols, svc and classical2 agglomerative, arima,
# ets, spectral-embedding, spectral, tsvd, umap. Every mapping names its lane (bare, as the classical drivers
# pass it), arm and library, and fires only on the values that prove the same work. Sources: cuML v26.08.00
# (python/cuml/cuml/..., cpp/src/...; cuVS v26.08.00 for the spectral eigensolver) and python/mojolearn/.
def _semantic_r5(lane, param, arm, ours, theirs, library):
    """Source-reviewed, value-conditional mappings for the r5 recipes (see the comment above)."""
    # smoke-fixes (2026-10-14; ols: "torch-gpu: driver missing from ours read-back"): the torch arm
    # runs torch.linalg.lstsq(driver='gels'), QR without pivoting of the centered design
    # (tools/classical_two_datasets.py TorchOLS); ours' tall-design fit is the blocked TSQR of the
    # centered [X | y] (python/mojolearn/linear_model.py:474-478 _ols_tsqr_on, :527-551 _ols_tsqr;
    # FAST-only normal equations at :719-722 never run in identical mode). Both factor X = QR.
    if lane == "ols" and arm == "torch-gpu" and library == "torch":
        if param == "driver" and param not in ours and theirs.get(param) == "gels" \
                and ours.get("fit_intercept") is True and theirs.get("fit_intercept") is True:
            return ("QR least squares on both: torch driver='gels' (QR, no pivoting) and ours' TSQR of the "
                    "centered design; fit_intercept is compared")
        return None
    if library != "cuml" or arm != "cuml-gpu":
        return None
    # smoke-fixes (2026-10-14; ols: "algorithm, copy_X missing from ours read-back"). cuML 26.08
    # linear_model/linear_regression.pyx: algorithm='qr' "uses QR decomposition and solves
    # Rx = Q^T y" (lines 104-105) on the dense single-target path (lines 352-385); copy_X=True means
    # "X will never be mutated" (lines 113-115, may_mutate_X at 383). Ours (linear_model.py:474-551)
    # solves the same least squares by the TSQR of the centered design and never writes the
    # caller's X (as_f32_c copies or reads it; the centering runs on the device copy).
    if lane == "ols" and param not in ours:
        if param == "algorithm" and theirs.get(param) == "qr":
            return "QR least squares on both: cuML algorithm='qr' (Rx = Q^T y) and ours' TSQR of the centered design"
        if param == "copy_X" and theirs.get(param) is True:
            return "neither arm mutates the caller's X: cuML copy_X=True; ours centers a device copy"
        return None
    # smoke-fixes (2026-10-14; svc: "break_ties, probability missing from cuml-gpu read-back").
    # cuML 26.08 svm/svc.py has neither option: no Platt scaling (_params_from_cpu refuses
    # probability=True, lines 169-170; _get_param_names lines 157-163) and multiclass through
    # sklearn's OneVsOne/OneVsRest wrappers chosen by decision_function_shape (lines 291-310). Ours
    # python/mojolearn/_svm_impl.py: probability=False skips _fit_probability (line 1041);
    # break_ties is refused with 'ovo' (lines 910-915), so False is the only 'ovo' value.
    if lane == "svc" and param not in theirs:
        if param == "probability" and ours.get(param) is False:
            return "no Platt-scaling cross-validation fits: ours probability=False; cuML SVC has no probability option"
        if param == "break_ties" and ours.get(param) is False \
                and ours.get("decision_function_shape") == theirs.get("decision_function_shape") == "ovo":
            return "break_ties is inactive under the compared decision_function_shape='ovo' (ours refuses True there)"
        return None
    # smoke-fixes (2026-10-14; agglomerative: "compute_distances, compute_full_tree, distance_threshold,
    # memory, prediction_data missing from cuml-gpu read-back"). cuML 26.08 cluster/agglomerative.pyx:
    # only n_clusters, metric, linkage, connectivity and c (lines 108-135); single_linkage builds the
    # whole dendrogram, children_ of shape (n_samples - 1, 2) (line 100), and cuts it at n_clusters
    # (lines 168-195); no merge distances, cache or predict. Ours python/mojolearn/_hierarchy_impl.py:
    # with a string connectivity the full tree is always built (lines 485-492), cut at n_clusters when
    # distance_threshold is None (lines 499-500, 522-523); distances_ only with compute_distances or a
    # threshold (lines 518-521); memory must be None (lines 262-266); prediction_data keeps a row copy
    # for predict only (lines 528-529).
    if lane == "agglomerative" and param not in theirs \
            and ours.get("linkage") == theirs.get("linkage") == "single" \
            and ours.get("connectivity") == theirs.get("connectivity") == "pairwise":
        if param == "compute_full_tree" and ours.get(param) in ("auto", True):
            return "the full single-linkage tree on both (ours always without a connectivity matrix; cuML children_ n-1 merges)"
        if param == "distance_threshold" and param in ours and ours[param] is None \
                and ours.get("n_clusters") == theirs.get("n_clusters"):
            return "both cut the tree at the compared n_clusters; ours distance_threshold=None, cuML has no threshold cut"
        if param == "compute_distances" and ours.get(param) is False:
            return "no merge-distance output on either arm: ours compute_distances=False; cuML returns none"
        if param == "memory" and param in ours and ours[param] is None:
            return "no tree cache on either arm: ours memory=None; cuML has no memory option"
        if param == "prediction_data" and ours.get(param) is False:
            return "no stored training copy for predict: ours prediction_data=False; cuML has no predict"
        return None
    # smoke-fixes (2026-10-14; arima: "max_iter, method, order, seasonal_order, trend missing from cuml-gpu
    # read-back"; "missing real library implementation read-back: cuml-gpu"): cuML 26.08 tsa/arima.pyx
    # get_params raises NotImplementedError (lines 596-603); the driver now records the constructed
    # object's order struct (p, d, q, P, D, Q, s, k, n_exog, lines 365-372), simple_differencing (line
    # 374) and the fit keywords it passes (method, maxiter, lines 860-866) (tools/bench_board_more.py,
    # arima cuml-gpu). Ours python/mojolearn/_arima_impl.py: trend is cuML's fit_intercept (DEVIATION
    # 993, lines 371-384: 'c' -> k = 1, 'n' -> k = 0); simple_differencing only the True arm (lines
    # 345-348); fit(y) takes exog only when given (the driver passes none).
    if lane == "arima":
        k_ours = {"c": True, "n": False}.get(str(ours.get("trend")).lower())
        if param == "trend" and param not in theirs and k_ours is not None \
                and theirs.get("fit_intercept") is k_ours:
            return "ours trend=%r is cuML fit_intercept=%s (the same k in the ARIMA order)" % (ours["trend"], k_ours)
        if param == "fit_intercept" and param not in ours and k_ours is not None \
                and theirs.get(param) is k_ours:
            return "cuML fit_intercept=%s is ours trend=%r (the same k in the ARIMA order)" % (k_ours, ours["trend"])
        if param == "simple_differencing" and param not in ours and theirs.get(param) is True:
            return "both difference the series before the fit: cuML simple_differencing=True, the only arm ours implements"
        if param == "n_exog" and param not in ours and theirs.get(param) == 0:
            return "no exogenous regressors on either arm (cuML n_exog=0; ours' fit receives none)"
        return None
    # smoke-fixes (2026-10-14; ets: "initialization_method missing from cuml-gpu read-back"). cuML 26.08
    # tsa/holtwinters.pyx has one initialization: the moving-average decomposition of the first
    # start_periods seasons, then the optimizer over alpha/beta/gamma only (lines 137-140, 265-300).
    # Ours python/mojolearn/_tsa_impl.py: 'heuristic' (alias 'cuml') is that fit (lines 113-116,
    # 421-440); 'estimated' also estimates the initial states (different work), so it never maps.
    if lane == "ets":
        if param == "initialization_method" and param not in theirs and ours.get(param) in ("heuristic", "cuml") \
                and ours.get("start_periods") == theirs.get("start_periods"):
            return "cuML's only initialization is ours 'heuristic': start_periods-season decomposition, then smoothing parameters only"
        return None
    # smoke-fixes (2026-10-14; spectral-embedding: "eigen_solver, eigen_tol, gamma missing from cuml-gpu
    # read-back"). cuML 26.08 manifold/spectral_embedding.pyx passes only n_components, n_neighbors,
    # norm_laplacian, drop_first and seed (lines 31-35, cpp/src/spectral/spectral_embedding.hpp:14-25),
    # so cuVS v26.08.00 runs RAFT's Lanczos (src/preprocessing/spectral/detail/spectral_embedding.cuh:
    # 62-77) at its default tolerance 1e-5 (include/cuvs/preprocessing/spectral_embedding.hpp:59) on the
    # kNN affinity. Ours python/mojolearn/_spectral_impl.py SpectralEmbedding: eigen_solver None is the
    # only value, RAFT's thick-restart Lanczos (lines 1164-1168); gamma only for affinity='rbf'
    # (lines 1157-1162).
    if lane == "spectral-embedding" and param not in theirs:
        if param == "eigen_tol" and isinstance(ours.get(param), float) and ours[param] == 1e-5:
            return "Lanczos tolerance 1e-5 on both (ours eigen_tol=1e-05; cuVS spectral_embedding params.tolerance default)"
        if param == "eigen_solver" and param in ours and ours[param] is None:
            return "RAFT Lanczos on both: ours eigen_solver=None (its only solver); cuML/cuVS has no other"
        if param == "gamma" and param in ours and ours[param] is None \
                and ours.get("affinity") == theirs.get("affinity") == "nearest_neighbors":
            return "gamma parameterizes only the RBF affinity; both use the compared nearest_neighbors affinity"
        return None
    # smoke-fixes (2026-10-14; spectral: "assign_labels, gamma, prediction_data missing from cuml-gpu
    # read-back"). cuML 26.08 cluster/spectral_clustering.pyx: "Kmeans is used for assigning labels"
    # (line 137; _params_to_cpu reports assign_labels='kmeans', line 188; other values refused, lines
    # 165-168); affinity only nearest_neighbors/precomputed (lines 161-164); no predict. Ours
    # python/mojolearn/_spectral_impl.py SpectralClustering: assign_labels (lines 359-363), gamma read
    # only for affinity='rbf' (lines 375-378, 461), prediction_data keeps Nystrom data for predict only.
    if lane == "spectral" and param not in theirs:
        if param == "assign_labels" and ours.get(param) == "kmeans":
            return "k-means label assignment on both (ours assign_labels='kmeans'; cuML's only assignment); n_init compared"
        if param == "gamma" and param in ours and ours[param] is None \
                and ours.get("affinity") == theirs.get("affinity") == "nearest_neighbors":
            return "gamma parameterizes only the RBF affinity; both use the compared nearest_neighbors affinity"
        if param == "prediction_data" and ours.get(param) is False:
            return "no prediction data kept: ours prediction_data=False; cuML SpectralClustering has no predict"
        # smoke-mappings (2026-10-15; classical2/spectral/{istella,taxi}: "eigen_solver missing from cuml-gpu
        # read-back"). Full pass over the lane's read-backs: cuML 26.08 SpectralClustering get_params carries
        # affinity, eigen_tol, n_clusters, n_components, n_init, n_neighbors, random_state (all compared) and
        # output_type, verbose (IGNORE). Ours python/mojolearn/_spectral_impl.py SpectralClustering adds
        # assign_labels, gamma, prediction_data (mapped above), degree, coef0, kernel_params (below), n_jobs,
        # verbose (IGNORE) and eigen_solver: ours accepts only None, its one Lanczos solver (docstring
        # "Only None is accepted"); cuML's spectral clustering runs cuVS/RAFT Lanczos and takes no
        # eigen_solver. eigen_tol (1e-5 on both arms) is compared.
        if param == "eigen_solver" and param in ours and ours[param] is None \
                and ours.get("affinity") == theirs.get("affinity") == "nearest_neighbors":
            return "Lanczos on both: ours eigen_solver=None (its only solver); cuML/cuVS has no other; eigen_tol compared"
        # smoke-mappings (2026-10-15; classical2/spectral/{istella,taxi}: "coef0 missing from cuml-gpu
        # read-back", the same for degree). degree, coef0 and kernel_params parameterize only scikit-learn's
        # polynomial/sigmoid/callable kernel affinities. Ours python/mojolearn/_spectral_impl.py
        # SpectralClustering refuses any value but None for all three (lines 414-425: "it parameterizes an
        # RBF or polynomial affinity"); cuML 26.08 cluster/spectral_clustering.pyx takes none of them and
        # builds only the nearest_neighbors/precomputed affinity (lines 161-164). Fires only when ours holds
        # None and both arms run the compared nearest_neighbors affinity.
        if param in ("degree", "coef0", "kernel_params") and param in ours and ours[param] is None \
                and ours.get("affinity") == theirs.get("affinity") == "nearest_neighbors":
            return ("%s parameterizes only polynomial/sigmoid kernel affinities: ours holds None (any value is "
                    "refused), cuML SpectralClustering has none; both use the compared nearest_neighbors "
                    "affinity" % param)
        return None
    # smoke-fixes (2026-10-14; tsvd: "n_oversamples, power_iteration_normalizer missing from cuml-gpu
    # read-back"). Ours python/mojolearn/decomposition.py:785-796 reads both only when
    # algorithm='randomized'; cuML 26.08 decomposition/tsvd.pyx 'full' is a full eigendecomposition
    # (lines 80-81) and has no randomized solver (lines 141-156).
    if lane == "tsvd" and param in ("n_oversamples", "power_iteration_normalizer") and param not in theirs \
            and ours.get("algorithm") == "covariance_eigh" and theirs.get("algorithm") == "full":
        return "randomized-solver controls are inactive: ours covariance_eigh and cuML 'full' are eigendecompositions"
    # smoke-fixes (2026-10-14; umap). cuML 26.08 manifold/umap/umap.pyx against
    # python/mojolearn/_umap_impl.py and umap/.
    if lane == "umap":
        # "a, b unset on ours and unset on cuml-gpu": None fits umap-learn's curve 1/(1 + a x^(2b)) to
        # the offset exponential on the same 300-point grid of [0, 3 spread] from the compared spread and
        # min_dist on both: cuML find_ab_params (lines 62-94, used at 482-483); ours a = b = None sends 0
        # (_umap_impl.py:256-259) and umap/curve.mojo fit_umap_curve fits it (umap/params.mojo:113-119,
        # checked against scipy within 2e-4 in umap/checks/curve_check.mojo).
        if param in ("a", "b") and param in ours and param in theirs and ours[param] is None \
                and theirs[param] is None and ours.get("a") is None and ours.get("b") is None \
                and isinstance(ours.get("spread"), float) and isinstance(ours.get("min_dist"), float) \
                and ours.get("spread") == theirs.get("spread") and ours.get("min_dist") == theirs.get("min_dist"):
            return ("a/b None: both fit umap-learn's curve from the compared spread=%r, min_dist=%r"
                    % (ours["spread"], ours["min_dist"]))
        # "metric_kwds unset on both": no keyword overrides for the compared euclidean metric
        # (ours _umap_impl.py:247-254 reads only minkowski 'p'; cuML passes none for euclidean).
        if param == "metric_kwds" and param in ours and param in theirs and ours[param] is None \
                and theirs[param] is None and ours.get("metric") == theirs.get("metric") == "euclidean":
            return "no metric keyword overrides for the compared euclidean metric on either arm"
        if param in theirs and param not in ours:
            value = theirs[param]
            # cuML build_algo='brute_force_knn' (lines 627-628) is the exact kNN ours always builds
            # (umap/estimator.mojo:22, 68: exact k-NN); build_kwds None keeps knn_n_clusters=1, one
            # unpartitioned exact search (lines 597-599); device_ids is read only with nn_descent and
            # knn_n_clusters > 1 (lines 898-902).
            if param == "build_algo" and value == "brute_force_knn":
                return "exact brute-force kNN graph on both (cuML build_algo='brute_force_knn'; ours always exact)"
            if param == "build_kwds" and value is None and theirs.get("build_algo") == "brute_force_knn":
                return "no kNN build overrides: knn_n_clusters=1, one unpartitioned exact search"
            if param == "device_ids" and value is None:
                return "single-device fit; cuML reads device_ids only for partitioned nn_descent"
            if param == "callback" and value is None:
                return "no per-epoch callback on cuML (lines 592-595); ours has none"
            if param == "precomputed_knn" and value is None:
                return "both build the kNN graph from X; no precomputed graph"
            # hash_input only lets a later transform(X) return the fit embedding (lines 776-785) and
            # transform_queue_size only steers transform's neighbor search (lines 746-750); the race
            # times fit_transform on both arms (tools/bench_board_more.py).
            if param == "hash_input" and value is False:
                return "no input hashing; read only by a later transform, and the race times fit_transform"
            if param == "transform_queue_size" and isinstance(value, float):
                return "read only by transform's neighbor search; the race times fit_transform"
            # force_serial_epochs: the driver pins True, cuML's own choice for a spectral fit (lines
            # 562-582: per-vertex serial kernel). Ours has no switch: its identical optimizer is a
            # per-vertex fold of the epoch snapshot (umap/optimizer_identical_device.mojo:1-40), the same
            # attractive/repulsive update, learning rate schedule, epochs and negative sampling (compared).
            # The fold order is each library's parallel schedule of that update; quality is scored.
            if param == "force_serial_epochs" and value is True and ours.get("init") == theirs.get("init") == "spectral":
                return ("cuML's per-vertex serial epoch kernel (its spectral-fit choice, pinned) and ours' per-vertex "
                        "snapshot fold schedule the same SGD update; epochs, learning rate and negative sampling compared")
        if param in ours and param not in theirs:
            # ours refuses densmap=True and any output_metric but euclidean (_umap_impl.py:234-241);
            # cuML has neither (plain UMAP objective, euclidean embedding space).
            if param == "densmap" and ours[param] is False:
                return "plain UMAP objective on both: ours densmap=False; cuML has no densMAP"
            if param == "output_metric" and ours[param] in ("euclidean", "l2"):
                return "euclidean embedding space on both: ours output_metric='euclidean'; cuML's is fixed"
        return None


def _r6_is(v, want):
    """`v` equals `want` (strings case-insensitively, as _equal compares them)."""
    if isinstance(v, str) and isinstance(want, str):
        return v.lower() == want.lower()
    if isinstance(v, bool) or isinstance(want, bool) or v is None or want is None:
        return v is want
    return isinstance(v, (int, float)) and v == want


def _semantic_r6(lane, param, arm, ours, theirs, library):
    """smoke-fixes (2026-10-14, batch 3, r6; trees/gbdt-depthwise/taxi and trees/gbdt-lossguide/taxi
    on NVIDIA: "<knob> missing from catboost-gpu read-back", "<knob> missing from xgboost-gpu
    read-back", "<knob> missing from ours read-back").

    Ours is python/mojolearn/ensemble.py GradientBoosting (get_params lists every constructor
    argument); CatBoost 1.2.10's get_params lists only what the driver set (read_params above), and
    XGBoost 3.2.0's lists its scikit-learn names, None for "native default" (core.py:2357
    Booster.set_param drops None, so the native default runs). The driver (tools/speed_gbdt_arm.py
    catboost_arms / xgboost_arms, bench/speed/forest_speed_arm.py) fits every arm on (X, y) with no
    eval set, no callbacks and no categorical columns on these lanes. Each row fires only for the
    named library, only when the absent side really lacks the field, and only at the values that
    make the knob inactive or equal; any other value still refuses. Sources were read at the box's
    versions (catboost v1.2.10, dmlc/xgboost v3.2.0 on GitHub). NOT mapped on raw input (different
    work, see the lane's reply): XGBoost's feature_border_type / border_build_max_samples (its hist
    cuts are a weighted quantile sketch over every row, src/common/hist_util.cu, not GreedyLogSum
    over a 200,000-row sample; no XGBoost setting selects them; smoke-fixes r13 maps them only when
    both arms read back the same pre-binned input, see below) and CatBoost's min_split_gain under
    Lossguide (greedy_search_helper.cpp:297-309, 321-328: the best leaf splits with no sign test,
    where ours' min_split_gain 0 requires a positive gain). smoke-fixes r13 (Andrew 2026-10-14):
    on gbdt-lossguide CatBoost is compared only with the own arm at min_split_gain None (no gain
    check, its own growth; tools/board_pairing.py), mapped below; ours at 0.0 (`ours-gain0`) is
    compared only with XGBoost, so no waiver is needed. 0.0 against CatBoost Lossguide still refuses.
    """
    if not lane.startswith("gbdt-"):
        return None
    cb = library == "catboost" and arm.startswith("catboost-")
    xg = library == "xgboost" and arm.startswith("xgboost-")
    if not (cb or xg):
        return None
    lacks_theirs = param in ours and param not in theirs
    lacks_ours = param in theirs and param not in ours
    if not (lacks_theirs or lacks_ours):
        return None
    policy = str(ours.get("grow_policy") or "")
    nonsym = policy.lower() in ("depthwise", "lossguide") and _r6_is(theirs.get("grow_policy"), policy)
    plain = _r6_is(ours.get("boosting_type"), "Plain")
    no_cats = (ours.get("cat_features") is None and ours.get("one_hot_features") is None
               and theirs.get("cat_features") is None)
    no_od = all(ours.get(k) is None for k in ("use_best_model", "od_type", "od_pvalue", "od_wait"))
    logloss = _r6_is(ours.get("loss"), "Logloss")
    if cb:
        cb_plain = plain and _r6_is(theirs.get("boosting_type"), "Plain")
        if not lacks_theirs:
            return None
        # smoke-fixes (2026-10-14; gbdt-depthwise/lossguide: "bagging_temperature missing from
        # catboost-gpu read-back"): CatBoost refuses bagging_temperature unless the bootstrap is
        # Bayesian (bootstrap_options.cpp:21-24, "you shoudn't provide bootstrap options if bootstrap
        # is disabled"); ours reads it under Bayesian only (ensemble.py:1243-1247).
        if (param == "bagging_temperature" and _r6_is(ours.get("bootstrap_type"), "No")
                and _r6_is(theirs.get("bootstrap_type"), "No")):
            return ("bootstrap_type 'No' on both: bagging_temperature is read by the Bayesian bootstrap "
                    "only, and CatBoost refuses it with bootstrap 'No' (bootstrap_options.cpp:21-24)")
        # smoke-fixes (2026-10-14; "best_model_min_trees, use_best_model, od_type, od_pvalue, od_wait
        # missing from catboost-gpu read-back"): with no eval set CatBoost's overfitting detector
        # and best-model truncation are off (core.py:2672-2673 refuses use_best_model without an
        # eval_set; options_helper.cpp:106-108 defaults it on only with a test set;
        # overfitting_detector_options.cpp:24-32); ours raises for either without an eval_set
        # (ensemble.py docs od_type / use_best_model). best_model_min_trees bounds that truncation only.
        if param in ("best_model_min_trees", "use_best_model", "od_type", "od_pvalue", "od_wait") and no_od \
                and not any(k in theirs for k in ("use_best_model", "od_type", "od_pval", "od_wait")):
            return ("no eval set on either arm (tools/speed_gbdt_arm.py catboost_arms _fit, "
                    "forest_speed_arm.py fit): no overfitting detector and no best-model truncation; "
                    "ours' od_*/use_best_model unset and CatBoost's absent")
        # smoke-fixes (2026-10-14; "border_build_max_samples missing from catboost-gpu read-back"):
        # CatBoost's border search subsamples MaxSubsetSizeForBuildBorders = 200000 rows
        # (data_processing_options.cpp:13-19, binarization_options.cpp:22); CatBoostClassifier's
        # constructor has no argument for it (core.py: only Pool.quantize takes
        # dev_max_subset_size_for_build_borders, :1313), so it is fixed at ours' value.
        if param == "border_build_max_samples" and _r6_is(ours.get(param), 200000):
            return ("CatBoost samples 200000 rows for its border search, fixed "
                    "(data_processing_options.cpp:18, no constructor argument); ours 200000")
        # smoke-fixes (2026-10-14; "cat_features, one_hot_features, ctr_estimation_permutation_id
        # missing from catboost-gpu read-back"): no categorical column on either arm; the CTR
        # estimation permutation is read by the categorical path only (ours ensemble.py:1480-1484).
        if param in ("cat_features", "one_hot_features", "ctr_estimation_permutation_id") \
                and no_cats and ours.get(param) is None:
            return "no categorical columns on either arm; the CTR and one-hot options are inactive"
        # smoke-fixes (2026-10-14; "feature_fraction missing from catboost-gpu read-back"): CatBoost's
        # rsm defaults to 1.0 (oblivious_tree_options.cpp:20) and its GPU learner refuses any other
        # value outside pairwise modes (catboost_options.cpp:638); ours 1.0 samples no features.
        if param == "feature_fraction" and _r6_is(ours.get(param), 1.0) and "rsm" not in theirs:
            return "no feature sampling on either arm: ours feature_fraction 1.0, CatBoost rsm 1.0 (default)"
        # smoke-fixes (2026-10-14; "fold_len_multiplier, fold_permutation_block, permutation_count
        # missing from catboost-gpu read-back"): Plain boosting with no CTR features runs ONE
        # permutation on CatBoost's GPU (cuda/train_lib/train.cpp:102-107), so the Ordered fold
        # knobs are unread; ours resolves permutation_count 1 for that fit and reads the fold knobs
        # under Ordered only (ensemble.py docs permutation_count, fold_len_multiplier).
        if param in ("fold_len_multiplier", "fold_permutation_block", "permutation_count") \
                and cb_plain and no_cats and _r6_is(ours.get("permutation_count"), None):
            return ("boosting_type 'Plain' with no categorical columns on both: one permutation, "
                    "no Ordered folds (CatBoost cuda/train_lib/train.cpp:102-107)")
        # smoke-fixes (2026-10-14; "loss_alpha, loss_border, loss_delta, loss_q, loss_variance_power
        # missing from catboost-gpu read-back"): Logloss reads only `border`, whose default 0.5 is
        # CatBoost's (loss_description.cpp:90-92) and ours (ensemble.py docs loss_border); the
        # others belong to Quantile/Expectile, Lq, Huber and Tweedie.
        if param in ("loss_alpha", "loss_border", "loss_delta", "loss_q", "loss_variance_power") \
                and logloss and _r6_is(theirs.get("loss"), "Logloss") and ours.get(param) is None:
            return ("loss Logloss on both: its target border is the shared default 0.5; the other "
                    "loss parameters belong to other losses")
        # smoke-fixes (2026-10-14; "min_child_weight missing from catboost-gpu read-back"): CatBoost
        # has no hessian floor (oblivious_tree_options.cpp:12-41 lists min_data_in_leaf only); ours'
        # min_child_hessian 0.0 admits every child (equality allowed), the same as none.
        if param == "min_child_weight" and ours.get(param) in (None, 0.0):
            return "no hessian floor on either arm: ours min_child_hessian %r, CatBoost has none" % (
                ours.get(param),)
        # smoke-fixes (2026-10-14; "min_split_gain missing from catboost-gpu read-back"): CatBoost has
        # no gain threshold. Depthwise splits exactly the leaves whose best split improves (Score < 0,
        # greedy_search_helper.cpp:356-359), ours' min_split_gain 0.0 (a strictly positive gain);
        # ours' None is CatBoost's own growth on any policy. Lossguide at 0.0 is NOT this case.
        if param == "min_split_gain" and (
                ours.get(param) is None
                or (_r6_is(ours.get(param), 0.0) and policy.lower() == "depthwise" and nonsym)):
            return ("CatBoost has no gain threshold; ours %r is its growth here (Depthwise splits "
                    "only improving leaves, greedy_search_helper.cpp:356-359)" % (ours.get(param),))
        # smoke-fixes (2026-10-14; "use_pointwise_searcher missing from catboost-gpu read-back"):
        # off the symmetric grower CatBoost's GPU grows with TGreedySubsetsSearcher
        # (pointwise_non_symmetric.cpp), the searcher ours runs at False; ours refuses True there
        # (ensemble.py:673).
        if param == "use_pointwise_searcher" and ours.get(param) is False and nonsym:
            return "Depthwise/Lossguide: the greedy subsets searcher on both arms"
        return None
    # ---- XGBoost (sklearn.py XGBClassifier, src/tree/param.h, src/tree/gpu_hist/*)
    if lane not in ("gbdt-depthwise", "gbdt-lossguide") or not nonsym:
        return None
    no_sampling = _r6_is(theirs.get("subsample"), 1.0) and theirs.get("sampling_method") in (None, "uniform")
    no_col = (_r6_is(ours.get("feature_fraction"), 1.0)
              and all(_r6_is(theirs.get(k), 1.0) for k in ("feature_fraction", "colsample_bylevel",
                                                            "feature_fraction_bynode")))
    plain_leaf = _r6_is(theirs.get("reg_alpha"), 0.0) and _r6_is(theirs.get("max_delta_step"), 0.0)
    binary = logloss and _r6_is(theirs.get("objective"), "binary:logistic")
    xg_no_cats = (no_cats and theirs.get("enable_categorical") is False
                  and theirs.get("feature_types") is None)
    xg_no_es = all(theirs.get(k) is None for k in ("early_stopping_rounds", "callbacks", "eval_metric"))
    if lacks_theirs:
        # smoke-fixes (2026-10-14; "bagging_temperature, bootstrap_type missing from xgboost-gpu
        # read-back"): no row sampling on either arm: ours bootstrap 'No' (bagging_temperature is
        # read by Bayesian only, ensemble.py:1243-1247); XGBoost subsample 1.0 builds no sampler
        # (src/tree/gpu_hist/sampler.cu:348-354).
        if param in ("bagging_temperature", "bootstrap_type") and _r6_is(ours.get("bootstrap_type"), "No") \
                and no_sampling:
            return "no row sampling on either arm: ours bootstrap_type 'No', XGBoost subsample 1.0"
        # smoke-fixes (2026-10-14; "best_model_min_trees, use_best_model, od_* missing from
        # xgboost-gpu read-back"): no eval set, no early stopping and no callbacks on either arm.
        if param in ("best_model_min_trees", "use_best_model", "od_type", "od_pvalue", "od_wait") \
                and no_od and xg_no_es:
            return "no eval set and no early stopping on either arm; ours' od_*/use_best_model unset"
        # smoke-fixes (2026-10-14; "boost_from_average missing from xgboost-gpu read-back"): both
        # start every row at margin 0: ours boost_from_average False; XGBoost's explicit base_score
        # disables its intercept fit (src/learner.cc:104-107, 172-173) and 0.5 is logit 0 under
        # binary:logistic (src/objective/regression_loss.h:74-77).
        if param == "boost_from_average" and ours.get(param) is False and binary \
                and _r6_is(theirs.get("base_score"), 0.5):
            return "margin 0 on both: ours boost_from_average False, XGBoost base_score 0.5 (logit 0)"
        # smoke-fixes (2026-10-14; "cat_features, one_hot_features, ctr_estimation_permutation_id
        # missing from xgboost-gpu read-back"): no categorical columns on either arm.
        if param in ("cat_features", "one_hot_features", "ctr_estimation_permutation_id") \
                and xg_no_cats and ours.get(param) is None:
            return "no categorical columns on either arm (XGBoost enable_categorical False)"
        # smoke-fixes (2026-10-14; "fold_len_multiplier, fold_permutation_block, permutation_count
        # missing from xgboost-gpu read-back"): Ordered-boosting knobs, unread by ours' Plain fit
        # without categorical columns (ensemble.py docs permutation_count, fold_len_multiplier).
        if param in ("fold_len_multiplier", "fold_permutation_block", "permutation_count") \
                and plain and xg_no_cats and ours.get("permutation_count") is None:
            return "ours boosting_type 'Plain' with no categorical columns: no Ordered folds to match"
        # smoke-fixes (2026-10-14; "leaf_estimation_method, leaf_estimation_iterations missing from
        # xgboost-gpu read-back"): one Newton step per leaf on both: ours Newton x 1; XGBoost's leaf
        # weight is -G / (H + lambda) (src/tree/param.h:234-246) with reg_alpha 0 and
        # max_delta_step 0 (no soft threshold, no clip).
        if param in ("leaf_estimation_method", "leaf_estimation_iterations") and plain_leaf \
                and _r6_is(ours.get("leaf_estimation_method"), "Newton") \
                and _r6_is(ours.get("leaf_estimation_iterations"), 1):
            return "one Newton step per leaf on both (XGBoost param.h CalcWeight, alpha 0, no clip)"
        # smoke-fixes (2026-10-14; "loss missing from xgboost-gpu read-back"): Logloss is XGBoost's
        # binary:logistic (src/objective/regression_loss.h LogisticClassification).
        if param == "loss" and binary:
            return "binary log loss on both: ours loss 'Logloss', XGBoost objective 'binary:logistic'"
        if param in ("loss_alpha", "loss_border", "loss_delta", "loss_q", "loss_variance_power") \
                and binary and ours.get(param) is None:
            return ("binary log loss on both; ours' other-loss parameters unset and its Logloss border "
                    "the default 0.5 (0/1 labels)")
        # smoke-fixes (2026-10-14; "min_samples_leaf missing from xgboost-gpu read-back"): ours
        # min_data_in_leaf 1 only stops a one-row leaf; XGBoost refuses any split with an empty
        # child (src/tree/gpu_hist/expand_entry.cuh:37-39), so a one-row node never splits there
        # either, at min_child_weight 0 on both.
        if param == "min_samples_leaf" and _r6_is(ours.get(param), 1) \
                and _r6_is(theirs.get("min_child_weight"), 0.0) and _r6_is(ours.get("min_child_weight"), 0.0):
            return ("no row-count floor beyond a non-empty child on either arm (XGBoost "
                    "expand_entry.cuh:37-39; ours min_data_in_leaf 1)")
        # smoke-fixes (2026-10-14; "nan_mode missing from xgboost-gpu read-back"): with no NaN in the
        # data neither missing-value rule is read; XGBoost's dense matrix carries no missing gradient
        # (src/tree/gpu_hist/evaluate_splits.cu:90-91). xgboost_arms refuses the arm by name when
        # the data holds a NaN (tools/speed_gbdt_arm.py, NAN-INPUT).
        if param == "nan_mode" and _r6_is(ours.get(param), "Min") and isinstance(theirs.get("missing"), float) \
                and math.isnan(theirs["missing"]):
            return "no NaN in the data (the driver refuses XGBoost otherwise): neither missing-value rule is read"
        # smoke-fixes (2026-10-14; "random_strength missing from xgboost-gpu read-back"): ours 0.0
        # adds no score noise; XGBoost has none (src/tree/param.h).
        if param == "random_strength" and _r6_is(ours.get(param), 0.0):
            return "no split-score noise on either arm (ours random_strength 0.0)"
        # smoke-fixes (2026-10-14; "score_function missing from xgboost-gpu read-back"): NewtonL2 is
        # sum G^2 / (H + lambda) over the children, XGBoost's CalcGain at reg_alpha 0 and
        # max_delta_step 0 (src/tree/param.h:248-256); lane_config pins NewtonL2 for this reason.
        if param == "score_function" and _r6_is(ours.get(param), "NewtonL2") and plain_leaf:
            return "the Newton L2 gain on both (XGBoost param.h CalcGain, alpha 0, no clip)"
        # smoke-fixes r13 (Andrew 2026-10-14, DECISIONS.md decision 1; "feature_border_type,
        # border_build_max_samples missing from xgboost-gpu read-back"): XGBoost has neither (its hist
        # cuts are a weighted quantile sketch over every row, src/common/hist_util.cu). Mapped ONLY
        # when both arms read back the same pre-binned input record (tools/speed_gbdt_arm.py
        # prebinned_data: GreedyLogSum bin ids, the bins' sha256 inside the record): every feature
        # then holds at most 255 distinct values, each its own bin on both arms (ours 254
        # GreedyLogSum borders over every row, border_build_max_samples 0; XGBoost max_bin 255,
        # compared), so both arms see the same split candidates. Raw input still refuses.
        if param in ("feature_border_type", "border_build_max_samples"):
            record = ours.get("input_binning")
            if (isinstance(record, str) and record.startswith("pre-binned input on both arms")
                    and theirs.get("input_binning") == record
                    and _r6_is(ours.get("border_build_max_samples"), 0)
                    and _r6_is(ours.get("feature_border_type"), "GreedyLogSum")
                    and ours.get("max_bin") is not None and ours.get("max_bin") == theirs.get("max_bin")):
                return ("the same pre-binned input on both arms (%s): every bin id is its own bin on "
                        "both (ours GreedyLogSum over every row, max_bin %s on both)"
                        % (record, ours.get("max_bin")))
            return None
        # smoke-fixes (2026-10-14; "use_pointwise_searcher missing from xgboost-gpu read-back"): ours
        # False is its greedy depthwise/lossguide searcher (True is refused off SymmetricTree,
        # ensemble.py:673); XGBoost has one hist grower.
        if param == "use_pointwise_searcher" and ours.get(param) is False:
            return "ours' greedy Depthwise/Lossguide searcher (the only one off SymmetricTree)"
        return None
    # lacks_ours: XGBoost scikit-learn fields ours has no name for
    # smoke-fixes (2026-10-14; "base_score missing from ours read-back"): see boost_from_average.
    if param == "base_score" and binary and _r6_is(theirs.get(param), 0.5) and ours.get("boost_from_average") is False:
        return "margin 0 on both: XGBoost base_score 0.5 (logit 0, learner.cc:172-173), ours boost_from_average False"
    # smoke-fixes (2026-10-14; "objective missing from ours read-back"): see loss.
    if param == "objective" and binary:
        return "binary log loss on both: XGBoost 'binary:logistic', ours 'Logloss'"
    # smoke-fixes (2026-10-14; "callbacks, early_stopping_rounds, eval_metric missing from ours
    # read-back"): no eval set; eval_metric is only evaluated on one (sklearn.py:447-503).
    if param in ("callbacks", "early_stopping_rounds", "eval_metric") and theirs.get(param) is None and no_od:
        return "no eval set, early stopping or callback on either arm"
    # smoke-fixes (2026-10-14; "colsample_bylevel, feature_fraction_bynode, feature_weights missing
    # from ours read-back"): no column sampling anywhere (ours feature_fraction 1.0, XGBoost every
    # colsample 1.0); feature_weights is read by column sampling only (sklearn.py:404-408).
    if param in ("colsample_bylevel", "feature_fraction_bynode") and no_col:
        return "no column sampling on either arm (ours feature_fraction 1.0, XGBoost colsample_* 1.0)"
    if param == "feature_weights" and theirs.get(param) is None and no_col:
        return "feature_weights unset and unread: no column sampling on either arm"
    # smoke-fixes (2026-10-14; "enable_categorical, feature_types, max_cat_threshold,
    # max_cat_to_onehot missing from ours read-back"): no categorical columns on either arm; the
    # max_cat_* knobs need enable_categorical (sklearn.py:410-432).
    if param in ("enable_categorical", "feature_types", "max_cat_threshold", "max_cat_to_onehot") \
            and xg_no_cats and theirs.get(param) in (None, False):
        return "no categorical columns on either arm (XGBoost enable_categorical False, ours cat_features None)"
    # smoke-fixes (2026-10-14; "interaction_constraints, monotone_constraints missing from ours
    # read-back"): unset, so XGBoost applies no constraint; ours has none.
    if param in ("interaction_constraints", "monotone_constraints") and theirs.get(param) is None:
        return "no split constraints on either arm"
    # smoke-fixes (2026-10-14; "max_delta_step, reg_alpha missing from ours read-back"): 0 disables
    # XGBoost's leaf clip and L1 soft threshold (src/tree/param.h:216-264); ours has neither.
    if param in ("max_delta_step", "reg_alpha") and _r6_is(theirs.get(param), 0.0):
        return "%s 0: XGBoost's leaf weight and gain are -G/(H+lambda) and G^2/(H+lambda), as ours" % param
    # smoke-fixes (2026-10-14; "missing missing from ours read-back"): see nan_mode.
    if param == "missing" and isinstance(theirs.get(param), float) and math.isnan(theirs[param]) \
            and _r6_is(ours.get("nan_mode"), "Min"):
        return "NaN marks a missing value, and the data has none (the driver refuses XGBoost otherwise)"
    # smoke-fixes (2026-10-14; "multi_strategy, num_parallel_tree missing from ours read-back"):
    # None reaches XGBoost as its defaults (core.py:2357): one output per tree
    # (learner.cc:309) for the one binary output, and one tree per round (gbm/gbtree_model.h:46).
    if param == "multi_strategy" and theirs.get(param) is None and binary:
        return "one binary output: XGBoost's default one_output_per_tree, one tree per round as ours"
    if param == "num_parallel_tree" and theirs.get(param) in (None, 1):
        return "one tree per boosting round on both (XGBoost num_parallel_tree default 1)"
    # smoke-fixes (2026-10-14; "sampling_method missing from ours read-back"): subsample 1.0 builds
    # no sampler whatever the method (src/tree/gpu_hist/sampler.cu:348-354).
    if param == "sampling_method" and no_sampling and _r6_is(ours.get("bootstrap_type"), "No"):
        return "no row sampling on either arm (XGBoost subsample 1.0, ours bootstrap_type 'No')"
    # smoke-fixes (2026-10-14; "tree_method missing from ours read-back"): 'hist' is XGBoost's
    # quantized-histogram grower, ours' only one; the bin count is compared as max_bin.
    if param == "tree_method" and _r6_is(theirs.get(param), "hist"):
        return "histogram growth on quantized features on both (bin count compared as max_bin)"


#: smoke-fixes (2026-10-14) sub-lane r7: the trees lanes whose CatBoost GPU mappings live in
#: _semantic_r7 (tools/speed_gbdt_arm.py lane_config, R7_SEARCHER_LANES).
_R7_CATBOOST_LANES = frozenset({"gbdt-symmetric", "gbdt-symmetric-1000", "gbdt-multiclass", "gbdt-ordered"})
#: losses that read none of ours' loss_alpha/loss_delta/loss_q/loss_variance_power (Quantile,
#: LogLinQuantile and Expectile read alpha, Huber delta, Lq q, Tweedie variance_power; ours
#: ensemble.py GradientBoosting docstring, CatBoost loss_description.cpp)
_R7_PLAIN_LOSSES = ("Logloss", "MultiClass", "RMSE")


def _semantic_querywise_pointwise(lane, param, arm, ours, theirs, library):
    """Map the explicitly selected GPU ranking searcher, never its old false route."""
    # FAIRNESS FIX (2026-10-13 audit, completed 2026-10-15): CatBoost v1.2.10
    # cuda/train_lib/querywise.cpp includes train_template_pointwise.h. Plain
    # DocParallel instantiates TDocParallelObliviousTree with its pointwise
    # searcher. Native f63ef223 implements its querywise statistics; this is
    # not an alias for our previous greedy-subsets algorithm. Keep CPU,
    # FeatureParallel, unsupported leaf methods and other losses refused.
    loss = {"gbdt-rank-pairlogit": "PairLogit", "gbdt-rank-yetirank": "YetiRank"}.get(lane)
    if loss is None or library != "catboost" or arm != "catboost-gpu":
        return None
    if not (ours.get("loss") == theirs.get("loss") == loss and
            ours.get("grow_policy") == theirs.get("grow_policy") == "SymmetricTree" and
            ours.get("boosting_type") == theirs.get("boosting_type") == "Plain" and
            ours.get("leaf_estimation_method") == theirs.get("leaf_estimation_method") and
            ours.get("leaf_estimation_method") in ("Newton", "Gradient") and
            "cat_features" in ours and ours["cat_features"] is None and
            theirs.get("cat_features") is None and
            ours.get("use_pointwise_searcher") is True and
            theirs.get("data_partition") == "DocParallel"):
        return None
    if (param == "use_pointwise_searcher" and param not in theirs) or (
            param == "data_partition" and param not in ours):
        return ("the same GPU querywise oblivious searcher: native grouped-target pointwise "
                "search and CatBoost Plain DocParallel TDocParallelObliviousTreeSearcher; "
                "loss and leaf estimator match explicitly")
    return None


def _semantic_r7(lane, param, arm, ours, theirs, library):
    """smoke-fixes (2026-10-14) sub-lane r7: trees/gbdt-symmetric, gbdt-symmetric-1000,
    gbdt-multiclass, gbdt-ordered (CatBoost 1.2.10 GPU) and trees/rf (cuML 26.08), absent API
    fields only, each conditional on the values that make it the same work. Sources: CatBoost
    v1.2.10 catboost/private/libs/options/*.cpp and catboost/cuda/{train_lib,methods,data}
    (github.com/catboost/catboost/tree/v1.2.10); cuML v26.08.00
    python/cuml/cuml/ensemble/randomforest_common.pyx; ours python/mojolearn/ensemble.py and
    randomforest.py."""
    if lane == "rf" and library == "cuml" and param in ours and param not in theirs:
        # smoke-fixes (2026-10-14; rf: the scikit-learn tree options ours' forest exposes and
        # cuML's does not): cuML's own scikit-learn interop refuses every non-default value
        # (randomforest_common.pyx BaseRandomForestModel._params_from_cpu: warm_start,
        # monotonic_cst not None, ccp_alpha != 0, min_weight_fraction_leaf != 0 raise
        # UnsupportedOnGPU), and ours refuses the same values (randomforest.py
        # _RandomForestBase.__init__ and RandomForestClassifier/Regressor.__init__), so each
        # maps only at the value that means "off".
        if param == "warm_start" and ours[param] is False:
            return "both fits build every tree from scratch: ours warm_start=False (True refused); cuML has no warm start"
        if param == "monotonic_cst" and ours[param] is None:
            return "no monotonic constraints on either arm: ours None (any other refused); cuML has none"
        if param == "ccp_alpha" and ours[param] == 0.0:
            return "no cost-complexity pruning on either arm: ours 0.0 (any other refused); cuML has none"
        if param == "min_weight_fraction_leaf" and ours[param] == 0.0:
            return ("no leaf weight-fraction floor on either arm: ours 0.0 (any other refused); cuML has "
                    "none; min_samples_leaf is compared")
        if param == "inference_engine" and ours[param] in ("auto", "sequential"):
            # implementation only: how ours' predict reduces over the same fitted trees
            # (randomforest.py: 'auto' is 'sequential' outside fast mode)
            return "ours' prediction reduction over the same fitted forest; not a model setting"
        return None
    if lane not in _R7_CATBOOST_LANES or library != "catboost":
        return None
    plain = ours.get("boosting_type") == "Plain" and theirs.get("boosting_type") == "Plain"
    ordered = ours.get("boosting_type") == "Ordered" and theirs.get("boosting_type") == "Ordered"
    symmetric = ours.get("grow_policy") == "SymmetricTree" and theirs.get("grow_policy") == "SymmetricTree"
    no_cats = "cat_features" in ours and ours["cat_features"] is None and "cat_features" not in theirs
    loss = ours.get("loss") if ours.get("loss") == theirs.get("loss") else None
    if param == "data_partition" and param in theirs and param not in ours and symmetric:
        # the reverse direction of use_pointwise_searcher below (R7_SEARCHER_LANES)
        param, mirrored = "use_pointwise_searcher", True
    else:
        mirrored = False
    if param == "use_pointwise_searcher" and (mirrored or (param in ours and param not in theirs)) \
            and symmetric and no_cats:
        part = theirs.get("data_partition")
        if ours.get(param) is True and part == "DocParallel" and plain and loss in ("Logloss", "RMSE"):
            # pointwise.cpp registers train_template_pointwise.h's trainer for these losses;
            # DocParallel trains TDocParallelObliviousTree (train_template_pointwise.h:49-50),
            # searched by TDocParallelObliviousTreeSearcher (doc_parallel_pointwise_oblivious_tree.h:45)
            return ("the same oblivious searcher: CatBoost data_partition='DocParallel' trains %s with "
                    "TDocParallelObliviousTreeSearcher, ours use_pointwise_searcher=True" % loss)
        if ours.get(param) is False and part == "DocParallel" and plain and loss == "MultiClass":
            # multiclass.cpp:1 builds the MultiClass trainer from
            # train_template_pointwise_greedy_subsets_searcher.h (TGreedySubsetsSearcher, line 35)
            return ("the same oblivious searcher: CatBoost's GPU MultiClass trainer is TGreedySubsetsSearcher "
                    "(doc-parallel), ours use_pointwise_searcher=False, the greedy subsets searcher")
        if ours.get(param) is False and part == "FeatureParallel" and ordered:
            # boosting_options.cpp:70-72 refuses DocParallel with Ordered; FeatureParallel trains
            # TDynamicBoosting with TFeatureParallelPointwiseObliviousTree (train_template_pointwise.h:27-29)
            return ("the same Ordered fold searcher: CatBoost data_partition='FeatureParallel' "
                    "(TDynamicBoosting), ours Ordered with use_pointwise_searcher=False (True is refused)")
        return None
    if param not in ours or param in theirs:
        return None
    v = ours[param]
    if param == "bagging_temperature" and v == 1.0 and ours.get("bootstrap_type") == "No" \
            and theirs.get("bootstrap_type") == "No":
        # CatBoost refuses any bagging_temperature beside bootstrap 'No' (bootstrap_options.cpp:21-24);
        # ours reads it under Bayesian only and refuses a non-default value otherwise
        return "no row sampling on either arm (bootstrap_type 'No', compared); bagging_temperature is never read"
    if param in ("use_best_model", "od_type", "od_wait", "od_pvalue") and v is None:
        # neither driver passes an eval set (forest_speed_arm._our_fit, speed_gbdt_arm.catboost_arms
        # _fit): ours' detector and best-model truncation require one (ensemble.py), CatBoost's
        # stop_pvalue 0 leaves its detector off (overfitting_detector_options.cpp:9) and
        # use_best_model is false without one (output_file_options.cpp:76)
        return "no eval set on either fit: no overfitting detector and no best-model truncation on either arm"
    if param == "best_model_min_trees" and ours.get("use_best_model") is None and ours.get("od_type") is None:
        return "read only by best-model truncation, which neither arm runs (no eval set on either fit)"
    if param in ("cat_features", "one_hot_features", "ctr_estimation_permutation_id") and v is None and no_cats:
        # CatBoost's arm passes cat_features only for a dataset with categorical columns
        # (speed_gbdt_arm.catboost_arms); ours reads one_hot_features and the CTR permutation only
        # for declared columns (ensemble.py)
        return "no categorical columns on either arm: no one-hot encoding and no CTRs"
    if param in ("loss_alpha", "loss_delta", "loss_q", "loss_variance_power") and v is None \
            and loss in _R7_PLAIN_LOSSES:
        return "read only by Quantile/Expectile/Huber/Lq/Tweedie; the raced loss (compared) is %s on both" % loss
    if param == "loss_border" and v is None and loss in _R7_PLAIN_LOSSES:
        # ours None is Logloss's 0.5 (ensemble.py); CatBoost's Logloss border defaults to the
        # default target border (loss_description.cpp:90-92); other losses read no border
        return ("Logloss's target threshold is the 0.5 default on both arms" if loss == "Logloss"
                else "%s reads no target border on either arm" % loss)
    if param in ("min_child_weight", "min_split_gain") and v is None and symmetric:
        # ours takes both on Depthwise/Lossguide only (ensemble.py); CatBoost's oblivious grower has
        # neither, and min_data_in_leaf (compared) is its only leaf floor
        return "no hessian floor and no split-gain threshold on the symmetric grower of either arm"
    if param == "permutation_count" and v is None and plain and no_cats:
        # ours None resolves to 1 for a Plain fit without CTRs (ensemble.py); CatBoost's GPU sets 1
        # for Plain without permutation-dependent features (train.cpp:101-108)
        return "one permutation on both arms: Plain boosting without CTRs"
    if param == "fold_len_multiplier" and v == 2.0 and plain:
        # read only by Ordered's TDynamicBoosting (dynamic_boosting.h:595), never by the Plain
        # TBoosting (doc_parallel_boosting.h); ours refuses another value on a Plain fit
        return "Ordered-only fold growth; both arms boost Plain (compared)"
    if param == "fold_permutation_block" and v is None and plain and no_cats \
            and ours.get("bootstrap_type") == "No" and theirs.get("bootstrap_type") == "No":
        # CatBoost's GPU sets 64 and uses it only to block-shuffle a permutation's row order
        # (train.cpp:118-121, permutation.cpp:7-17); with one permutation, no CTRs and no
        # bootstrap, Plain boosting sums every row of every leaf, so the order changes no split,
        # leaf or model value. Ours reads it only under Ordered (ensemble.py)
        return ("only a row-order block size for CatBoost's one Plain permutation; with no CTRs and no "
                "bootstrap the order changes no split or leaf")
    if param == "boost_from_average" and v is None and loss == "MultiClass":
        # CatBoost refuses boost_from_average for MultiClass (catboost_options.cpp:703-709) and
        # defaults it false (boosting_options.cpp:17); ours None is False for MultiClass (ensemble.py)
        return "MultiClass starts from zero on both arms; no arm can start it from an average"
    if param == "scale_pos_weight" and v == 1.0 and loss == "MultiClass" \
            and not any(k in theirs for k in ("class_weights", "auto_class_weights")):
        # ours 1.0 here is class_weights None (_pos_over_neg); CatBoost's arm is given no weights
        return "no class weights on either arm (ours class_weights None, CatBoost none)"
    return None


_R8_SEQ_LANES = frozenset("algos/%s-%s" % (c, t) for c in ("lstm", "gru", "rnn") for t in ("clf", "reg"))


def _semantic_r8(lane, param, arm, ours, theirs, library):
    """smoke-fixes (2026-10-14) sub-lane r8: algos/lstm-*, gru-*, rnn-* and cnn-clf against torch,
    absent API fields only, each conditional on the values that make it the same work. Sources:
    python/mojolearn/_x_sequence_rnn.py, python/mojolearn/_expansion_cnn.py, the torch arms in
    tools/bench_board_algos.py (_build_seqmodel, _build_cnnclf) and torch/nn/modules/rnn.py,
    torch/optim/sgd.py."""
    if library != "torch" or not arm.startswith("torch-"):
        return None
    if lane in _R8_SEQ_LANES and param == "lr_schedule" and "lr_schedule" not in theirs \
            and "lr_schedule" in ours and ours["lr_schedule"] is None \
            and "learning_rate" in ours and "learning_rate" in theirs \
            and _equal(ours["learning_rate"], theirs["learning_rate"]):
        # smoke-fixes (2026-10-14; algos/lstm-*/gru-*/rnn-*: "lr_schedule missing from torch-* read-back"):
        # ours with lr_schedule None steps at learning_rate on every step (_x_sequence_rnn.py:186-195,
        # _RecurrentBase._lrs: np.full(n_steps, learning_rate)); the torch arm builds torch.optim.Adam
        # (lr=learning_rate) with no LR scheduler (bench_board_algos.py _build_seqmodel fit()). Fires only
        # when ours' lr_schedule is None and both learning rates are equal.
        return ("no learning-rate schedule on either arm: ours' lr_schedule None steps at the constant "
                "learning_rate (_x_sequence_rnn.py _RecurrentBase._lrs), the torch arm's Adam has no scheduler")
    if lane in _R8_SEQ_LANES and not lane.startswith("algos/rnn-") and param == "nonlinearity" \
            and "nonlinearity" not in theirs and ours.get("nonlinearity") == "tanh":
        # smoke-fixes (2026-10-14; algos/lstm-*/gru-*: "nonlinearity missing from torch-* read-back"):
        # ours' LSTM/GRU ignore nonlinearity (_x_sequence_rnn.py:111-116, _RecurrentBase._cell reads it for
        # the RNN cell only; the LSTM*/GRU* docstrings: "Unused by this model"); torch.nn.LSTM / nn.GRU take
        # no nonlinearity (torch/nn/modules/rnn.py; only nn.RNN does): both run the fixed sigmoid gates and
        # tanh candidate/cell. Fires only on the lstm/gru lanes and ours' default 'tanh'.
        return ("LSTM/GRU have fixed sigmoid gates and tanh cell activations on both arms; ours' "
                "nonlinearity is unused there (_x_sequence_rnn.py _RecurrentBase._cell) and torch.nn.LSTM/GRU "
                "take none")
    if lane == "algos/cnn-clf" and param in ("betas", "eps") and param not in theirs \
            and str(ours.get("optimizer", "")).lower() == "sgd" and str(theirs.get("optimizer", "")).lower() == "sgd":
        # smoke-fixes (2026-10-14; algos/cnn-clf: "betas, eps missing from torch-* read-back"): ours'
        # CNNClassifier reads betas and eps only on its Adam/AdamW branch (_expansion_cnn.py:1557-1563 and
        # 1620-1637: optimizer == 'sgd' hands x_cnn_fit_epoch_d / x_cnn_sgd_r lr, momentum, weight_decay,
        # dampening and nesterov only); torch.optim.SGD has no betas or eps (torch/optim/sgd.py). Fires only
        # when both arms run SGD.
        return ("SGD on both arms: ours' CNNClassifier uses betas/eps only for adam/adamw "
                "(_expansion_cnn.py fit), torch.optim.SGD has neither")
def _semantic_r11(lane, param, arm, ours, theirs, library):
    """smoke-fixes (2026-10-14, batch 4, r11; NVIDIA algos/kbins, robust-scaler, sparse-rp,
    label-encoder, onehot vs cuML 26.08). Each row fires only for the named lane, the cuml-gpu arm
    and only at the values that make the absent setting inactive or equal. cuML sources read at
    github.com/rapidsai/cuml v26.08.00 python/cuml/cuml/ (paths below); ours python/mojolearn/."""
    if library != "cuml" or arm != "cuml-gpu" or not lane.startswith("algos/"):
        return None
    algo = lane.removeprefix("algos/")
    lacks_theirs = param in ours and param not in theirs
    lacks_ours = param in theirs and param not in ours
    # smoke-fixes (2026-10-14; kbins: "dtype, quantile_method, subsample missing from cuml-gpu
    # read-back"). cuML _thirdparty/sklearn/preprocessing/_discretization.py: __init__ takes only
    # n_bins/encode/strategy (line 140); fit keeps X's float dtype (check_inputs dtype=FLOAT_DTYPES,
    # line 171) and takes quantile edges from every row with cupy's np.percentile (lines 203-205;
    # `import cupy as np`, line 21), whose default method is 'linear'. Ours _expansion_prep.py
    # KBinsDiscretizer: quantile_method 'linear' is method 2 (lines 2460-2462), subsample None fits
    # every row (lines 2473, 2478: no resample), dtype is accepted only (output always float32,
    # line 2407-2408) and the lane times fit (bin edges), which dtype never reaches.
    if algo == "kbins" and lacks_theirs and ours.get("strategy") == "quantile" \
            and theirs.get("strategy") == "quantile":
        if param == "quantile_method" and ours.get(param) == "linear":
            return "linear quantile edges on both: ours quantile_method='linear', cuML cupy.percentile's linear default"
        if param == "subsample" and param in ours and ours[param] is None:
            return "every fit row on both: ours subsample None draws no resample; cuML has no subsample"
        if param == "dtype" and param in ours and ours[param] is None:
            return ("dtype sets only transform's output type (ours always float32, cuML X's float dtype); "
                    "the lane times fit, the bin edges")
    # smoke-fixes (2026-10-14; robust-scaler: "unit_variance missing from cuml-gpu read-back").
    # cuML _thirdparty/sklearn/preprocessing/_data.py RobustScaler: __init__ has no unit_variance
    # (lines 1270-1271) and scale_ is the plain interquantile range (lines 1341-1347). Ours
    # _expansion_prep.py:877 rescales by the normal ppf only when unit_variance is True.
    if algo == "robust-scaler" and param == "unit_variance" and lacks_theirs and ours.get(param) is False:
        return "scale = the plain interquantile range on both (ours unit_variance False; cuML has no option)"
    # smoke-fixes (2026-10-14; sparse-rp: "compute_inverse_components missing from cuml-gpu
    # read-back"). cuML random_projection/random_projection.py _BaseRandomProjection (__init__
    # lines 53-66, fit lines 81-121; SparseRandomProjection lines 341-408) has no such option and
    # never forms the inverse; ours _expansion_decomp.py:2098 computes it only when True.
    if algo == "sparse-rp" and param == "compute_inverse_components" and lacks_theirs \
            and ours.get(param) is False:
        return "no pseudo-inverse of the components on either arm; cuML has no inverse option"
    # smoke-fixes (2026-10-14; label-encoder: "handle_unknown missing from ours read-back"). cuML
    # preprocessing/_label.py LabelEncoder: handle_unknown='error' raises ValueError on a label not
    # seen in fit (transform, lines 181-184; inverse_transform 210-215). Ours _expansion_prep.py
    # LabelEncoder takes no handle_unknown and always raises ValueError on an unseen label
    # (docstring and transform, lines 5450, 5521, 5530).
    if algo == "label-encoder" and param == "handle_unknown" and lacks_ours and theirs.get(param) == "error":
        return "an unseen label raises ValueError on both (cuML handle_unknown='error'; ours always)"
    # smoke-fixes (2026-10-14; onehot: "drop is unset on ours and unset on cuml-gpu",
    # "feature_name_combiner, max_categories, min_frequency missing from cuml-gpu read-back").
    # cuML preprocessing/encoders.py OneHotEncoder: drop None keeps every category (drop_idx_ None,
    # _compute_drop_idx lines 240-243) and must be None with handle_unknown='ignore' (lines
    # 228-231); __init__ (lines 189-206) has no infrequent grouping and no feature-name option.
    # Ours _expansion_prep.py OneHotEncoder: drop None drops nothing (_set_drop_idx), grouping is
    # on only when min_frequency or max_categories is set (_check_infrequent_params, line 1174),
    # feature_name_combiner is read only by get_feature_names_out (docstring line 1511).
    if algo == "onehot":
        both_ignore = ours.get("handle_unknown") == "ignore" and theirs.get("handle_unknown") == "ignore"
        if (param == "drop" and param in ours and param in theirs and ours[param] is None
                and theirs[param] is None and both_ignore):
            return "no category dropped on either arm (drop None; cuML requires it with handle_unknown='ignore')"
        if param in ("min_frequency", "max_categories") and lacks_theirs and ours.get(param, 0) is None \
                and ours.get("min_frequency") is None and ours.get("max_categories") is None:
            return "no infrequent-category grouping on either arm (ours both None; cuML has none)"
        if param == "feature_name_combiner" and lacks_theirs and ours.get(param) == "concat":
            return "feature names only (get_feature_names_out); the encoding is unchanged"
    return None


def _semantic_r15(lane, param, arm, ours, theirs, library):
    """smoke-fixes (2026-10-14; algos/als/taxi-zones: "use_gpu missing from ours read-back"): implicit's use_gpu
    picks its CUDA or CPU class (implicit/als.py factory); the math either class runs is its least-squares solver,
    now compared on every arm as use_cg and cg_steps (bench_board_algos.py als lane). Ours has no device argument
    (the race places it). Fires only when both arms read back the same use_cg and cg_steps."""
    if lane != "algos/als" or library != "implicit" or param != "use_gpu" or param in ours:
        return None
    # refused-param-mappings (2026-10-15): any bool used to pass, so an implicit-gpu arm reading back
    # use_gpu=False (implicit's CPU class) matched; the class must be the one the arm is named for.
    if theirs.get("use_gpu") is arm.endswith("-gpu") and ours.get("use_cg") is not None \
            and ours.get("use_cg") == theirs.get("use_cg") and ours.get("cg_steps") == theirs.get("cg_steps"):
        return ("use_gpu only selects implicit's CUDA or CPU class; the solver both run is compared (use_cg=%s, "
                "cg_steps=%s)" % (ours.get("use_cg"), ours.get("cg_steps")))
    return None


#: smoke-fixes (2026-10-14) sub-lane r18: the CatBoost GPU task lanes whose mappings live in
#: _semantic_r18 (tools/speed_gbdt_arm.py TASK_LANES)
_R18_RANK_LANES = frozenset({"gbdt-rank-pairlogit", "gbdt-rank-yetirank"})
_R18_CATBOOST_LANES = _R18_RANK_LANES | {"gbdt-categorical"}


def _semantic_r18(lane, param, arm, ours, theirs, library):
    """smoke-fixes (2026-10-14) sub-lane r18: trees/gbdt-categorical/taxi,
    gbdt-rank-pairlogit/istella and gbdt-rank-yetirank/istella against CatBoost 1.2.10's GPU
    learner (arm catboost-gpu only), absent API fields only, each conditional on the values that
    make it the same work. Sources: CatBoost v1.2.10 catboost/private/libs/options/*.cpp,
    catboost/cuda/{train_lib,methods,data,gpu_data} (github.com/catboost/catboost/tree/v1.2.10,
    cached in ~/mojolearn-evidence/smoke-fixes/r7-src/); ours python/mojolearn/ensemble.py,
    gbdt/train.mojo, gbdt/ctrs/ctr_order.mojo. boost_from_average on the ranking lanes is pinned
    False on both arms instead (speed_gbdt_arm.boost_from_average_for)."""
    if lane not in _R18_CATBOOST_LANES or library != "catboost" or arm != "catboost-gpu":
        return None
    plain = ours.get("boosting_type") == "Plain" and theirs.get("boosting_type") == "Plain"
    loss = ours.get("loss") if ours.get("loss") == theirs.get("loss") else None
    # the one-identity-permutation CTR setup of gbdt-categorical (TASK_LANES ctr_permutations):
    # CatBoost has_time=True forces PermutationCount = 1 (catboost_options.cpp:1042-1044, an
    # assignment), its estimation permutation is PermutationsCount() - 1 = 0
    # (doc_parallel_boosting.h:101-103), permutation 0 is the identity (permutation.cpp:14-17), and
    # has_time skips the load-time pool shuffle (private/libs/algo/preprocess.cpp:161-199). Ours
    # never shuffles the pool, and permutation_count 1 with ctr_estimation_permutation_id 0 is its
    # one identity order (gbdt/train.mojo perm_count/est_perm, gbdt/ctrs/ctr_order.mojo)
    identity_ctrs = (lane == "gbdt-categorical" and plain and theirs.get("has_time") is True
                     and ours.get("permutation_count") == 1
                     and ours.get("ctr_estimation_permutation_id") == 0)
    # FAIRNESS FIX (2026-10-13 audit): None means no forced one-hot columns,
    # not no automatic one-hot encoding. Native TCatFeatureParams.default()
    # fixes the threshold at 2; require CatBoost's explicit read-back of 2
    # and the same categorical columns before mapping either absent API field.
    automatic_one_hot = (identity_ctrs and ours.get("one_hot_features") is None
                         and bool(ours.get("cat_features"))
                         and ours.get("cat_features") == theirs.get("cat_features")
                         and theirs.get("one_hot_max_size") == 2)
    if param == "one_hot_max_size" and param in theirs and param not in ours and automatic_one_hot:
        return "automatic one-hot at cardinality <= 2 on both: native fixed threshold and explicit CatBoost read-back"
    if param == "has_time" and param in theirs and param not in ours:
        # smoke-fixes (2026-10-14; gbdt-categorical/taxi: "has_time missing from ours read-back"):
        # ours has no has_time; its CTRs are always computed over the rows as passed, in the
        # estimation permutation's order, here the identity
        if identity_ctrs:
            return ("CTRs over the rows in the order given on both arms: CatBoost has_time=True (one "
                    "permutation, the identity, no pool shuffle); ours permutation_count 1, "
                    "ctr_estimation_permutation_id 0 (compared)")
        return None
    if param not in ours or param in theirs:
        return None
    v = ours[param]
    if param == "one_hot_features" and automatic_one_hot:
        return "no forced one-hot columns; both use automatic cardinality <= 2 on the same declared columns"
    if param == "scale_pos_weight" and v == 1.0 and lane in _R18_RANK_LANES \
            and loss in ("PairLogit", "YetiRank") \
            and not any(k in theirs for k in ("class_weights", "auto_class_weights")):
        # FAIRNESS FIX (2026-10-13 audit): the canonical ratio of absent
        # class weights is 1; ranking has no positive-class reweighting.
        # This does NOT waive the unresolved ranking searcher difference.
        return "no class weighting on either ranking arm (unit canonical ratio)"
    if param in ("ctr_estimation_permutation_id", "permutation_count") and identity_ctrs:
        # smoke-fixes (2026-10-14; gbdt-categorical/taxi: "ctr_estimation_permutation_id,
        # permutation_count missing from catboost-gpu read-back"): CatBoost's Python constructor
        # takes neither (1.2.10); has_time=True fixes both on its side (above)
        return ("one CTR permutation, the identity, on both arms: CatBoost has_time=True forces "
                "permutation_count 1 and estimates on permutation 0; ours is given 1 and 0")
    if param == "fold_len_multiplier" and v == 2.0 and plain:
        # smoke-fixes (2026-10-14; gbdt-categorical/taxi: "fold_len_multiplier missing from
        # catboost-gpu read-back"; with CTRs, so the earlier no-categorical rules do not fire):
        # read only by Ordered's TDynamicBoosting (dynamic_boosting.h:595); the Plain TBoosting of
        # doc_parallel_boosting.h, the only one CatBoost's Lossguide trainer runs, never reads it.
        # Ours refuses another value on a Plain fit (ensemble.py)
        return "Ordered-only fold growth; both arms boost Plain (compared)"
    if param == "fold_permutation_block" and v is None and plain:
        # smoke-fixes (2026-10-14; gbdt-categorical/taxi: "fold_permutation_block missing from
        # catboost-gpu read-back"; with CTRs, so the earlier no-categorical rules do not fire):
        # the GPU sets the option to 64 (train.cpp:118-121), but only TDynamicBoosting reads it
        # (dynamic_boosting.h:127-143). The Plain doc-parallel datasets, including every CTR
        # permutation, are built with GetPermutation(DataProvider, permutationId)
        # (doc_parallel_dataset_builder.cpp:48), whose blockSize defaults to 1 (permutation.h
        # GetPermutation), so the option reaches no order. Ours reads it only
        # under Ordered and refuses it on a Plain fit (ensemble.py)
        return "read only by Ordered boosting; CatBoost's Plain datasets shuffle with block size 1"
    if param in ("loss_alpha", "loss_delta", "loss_q", "loss_variance_power", "loss_border") \
            and v is None and lane in _R18_RANK_LANES and loss in ("PairLogit", "YetiRank"):
        # smoke-fixes (2026-10-14; gbdt-rank-*: "loss_alpha, loss_border, loss_delta missing from
        # catboost-gpu read-back"): alpha is read by Quantile/LogLinQuantile/Expectile, delta by
        # Huber, q by Lq, variance_power by Tweedie (ensemble.py _REQUIRED_PARAM and the
        # GradientBoosting docstring), the border by Logloss only (GetLogLossBorder asserts
        # Logloss, loss_description.cpp:90-92; ours loss_border is Logloss's threshold). The
        # pairwise losses read the grades within each query and none of these
        return "the raced loss (compared) is %s on both, which reads no %s" % (loss, param[5:])
    return None


def semantic_exception(lane, param, arm, ours, theirs, library):
    """Source-reviewed inactive/fixed API settings, conditional on actual values.

    FAIRNESS FIX (2026-10-13 audit): an absent API option need not mean different
    work. Retain raw readback and the explanation, never fabricate a parameter.
    cuML 26.8 source: decomposition/pca.pyx, neighbors/kernel_density.pyx,
    cluster/kmeans.pyx and neighbors/nearest_neighbors.pyx. Our matching source:
    decomposition.py, density.py, cluster.py and neighbors.py. Reasons below
    apply only to the named recipe, library and resolved settings.
    """
    # smoke-fixes (2026-10-14): the per-sub-lane source-reviewed mappings, each value-conditional and lane-named
    for _mapping in (_semantic_r4, _semantic_r5, _semantic_r6, _semantic_querywise_pointwise, _semantic_r7, _semantic_r8, _semantic_r11, _semantic_r15, _semantic_r18):
        why = _mapping(lane, param, arm, ours, theirs, library)
        if why:
            return why
    # FAIRNESS FIX (2026-10-13 audit): optimizer defaults exposed by torch
    # were treated as absent algorithm settings on our narrower optimizer
    # API. Preserve the raw settings and explicitly distinguish the library's
    # execution variants from changes to the update equation. Source:
    # torch.optim.{adam,adamw,sgd,...}, _training_impl.py and
    # _x_sequence_optim.py. Never waive an unknown or active algorithm option.
    optimizer = lane.removeprefix("algos/")
    if library == "torch" and arm.startswith("torch-"):
        unset = {"avgpool2d": {"divisor_override": "None selects the kernel-area divisor on both AvgPool2d implementations"},
                 "embedding": {"padding_idx": "None means there is no padding token on either embedding",
                               "max_norm": "None disables embedding-vector renormalization on both arms"}}
        if param in unset.get(optimizer, {}) and param in ours and param in theirs and ours[param] is None and theirs[param] is None:
            return unset[optimizer][param] + "; all active arguments remain compared"
    # preflight-smoke (2026-10-14): the math-changing flags (amsgrad, maximize, decoupled_weight_decay) are no
    # longer waived here. Ours' read-back carries the values its implementation fixes (bench_board_algos.py
    # _OURS_OPT_SEMANTICS, bench_board_neural.py OURS_ADAMW_SEMANTICS), the torch arm passes the same values
    # explicitly, and the two are compared like any other setting. Only torch's implementation choices are
    # exceptions, on every lane whose torch arm builds a torch.optim optimizer (the algos optimizer lanes and
    # the neural train-step lanes, whose torch arm is torch.optim.AdamW).
    if library == "torch" and arm.startswith("torch-") and optimizer in _TORCH_OPTIMIZER_LANES:
        if param not in ours and param in theirs:
            value = theirs[param]
            if param in ("foreach", "fused") and (value is None or isinstance(value, bool)):
                return "torch's foreach/fused selects its own implementation of the same optimizer update; the raw execution choice remains recorded"
            if param in ("capturable", "differentiable") and value is False:
                return "no captured-graph or differentiable optimizer step is requested; our standalone update has neither mode"
        # An older torch whose AdamW/Adam takes no decoupled_weight_decay and does not read it back: the class
        # itself fixes the choice (torch.optim.AdamW decoupled, torch.optim.Adam coupled). Waived only when
        # ours' read-back value is that class's fixed choice.
        if param == "decoupled_weight_decay" and param in ours and param not in theirs:
            fixed = _TORCH_DECOUPLED_BY_CLASS.get(optimizer)
            if fixed is not None and ours[param] is fixed:
                return ("this torch's %s takes no decoupled_weight_decay argument and fixes it to %s; ours reads back "
                        "the same" % ("AdamW" if fixed else "Adam", fixed))
    if library == "cuvs" and arm == "hipvs-gpu":
        if lane == "kmeans":
            if param == "init" and ours.get(param) == "k-means++" and theirs.get(param) == "KMeansPlusPlus":
                return "same k-means++ initializer, spelled KMeansPlusPlus by hipVS KMeansParams; oversampling is compared separately"
            if param == "metric" and ours.get(param) == "euclidean" and theirs.get(param) == "sqeuclidean":
                return "both KMeans objectives minimize squared Euclidean distances; hipVS names the squared distance explicitly"
            if param == "init_centroids" and ours.get(param) is None and param not in theirs:
                return "no supplied initial centroids; hipVS fit(params, X) uses the separately compared init_method"
            if param in ("hierarchical", "hierarchical_n_iters") and param not in ours and theirs.get("hierarchical") is False:
                return "hipVS hierarchical=False selects ordinary KMeans; hierarchical_n_iters is inactive"
        if lane == "knn" and param == "query_tile" and ours.get("algorithm") == theirs.get("algorithm") == "brute" and isinstance(ours.get(param), int) and ours[param] >= 0 and param not in theirs:
            return "both brute-force searches cover every index/query pair; query_tile only partitions our complete query work"
        return None
    if library != "cuml" or arm != "cuml-gpu":
        return None
    algo = lane.removeprefix("algos/") if lane.startswith("algos/") else None
    # smoke-fixes (2026-10-14; bernoulli-nb, categorical-nb, multinomial-nb, complement-nb:
    # "class_prior is unset on ours and unset on cuml-gpu", "force_alpha missing from cuml-gpu
    # read-back"). Source: cuML 26.08 python/cuml/cuml/naive_bayes/naive_bayes.py
    # (_update_class_log_prior, lines 591-608; partial_fit alpha check, line 661;
    # _update_feature_log_prob adds self.alpha unclipped, lines 876, 1018, 1146, 1388) against
    # python/mojolearn/_expansion_prep.py (_DiscreteNB._prior_stages, lines 3560-3568; CategoricalNB
    # lines 7432-7435; _check_alpha, line 3140: alpha used as given, force_alpha only stored).
    if algo in ("bernoulli-nb", "categorical-nb", "multinomial-nb", "complement-nb"):
        if (param == "class_prior" and param in ours and param in theirs and ours[param] is None
                and theirs[param] is None and ours.get("fit_prior") is True
                and theirs.get("fit_prior") is True):
            return ("class_prior None with fit_prior True learns log(class_count / n) from the fit "
                    "rows on both (cuML _update_class_log_prior; ours class_log_prior stage); "
                    "fit_prior and alpha remain compared")
        if param == "force_alpha" and param not in theirs and ours.get(param) is True:
            return ("cuML has no force_alpha and adds alpha unclipped; ours force_alpha=True is "
                    "scikit-learn's unclipped alpha (ours never clips); alpha remains compared")
        if (algo == "categorical-nb" and param == "min_categories" and param not in theirs
                and isinstance(ours.get(param), list) and ours[param]
                and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in ours[param])):
            # the driver refuses the cuML arm unless these equal cuML's fit-derived counts
            # (tools/bench_board_algos.py _est_factory, categorical-nb)
            return ("cuML sizes each feature by its highest fit code + 1 (naive_bayes.py lines "
                    "1380-1388); ours' min_categories equals those counts on every raced dataset "
                    "(the driver refuses the cuML arm when a query code exceeds the fit maximum)")
    # smoke-fixes (2026-10-14; gaussian-nb: "priors is unset on ours and unset on cuml-gpu").
    # cuML 26.08 naive_bayes.py GaussianNB.fit lines 387-388 (priors None: class_count_ / sum);
    # ours _expansion_prep.py:3166-3168 (None uses the training class frequencies) and 3222-3228.
    if (algo == "gaussian-nb" and param == "priors" and param in ours and param in theirs
            and ours[param] is None and theirs[param] is None):
        return "priors None means the fit rows' class frequencies on both; var_smoothing remains compared"
    # smoke-fixes (2026-10-14; gaussian-rp: "compute_inverse_components missing from cuml-gpu
    # read-back"). cuML 26.08 random_projection/random_projection.py: _BaseRandomProjection
    # (__init__ lines 53-66, fit lines 81-121) has no such option and never forms the inverse;
    # ours _expansion_decomp.py:2098 computes the pseudo-inverse only when it is True.
    if (algo == "gaussian-rp" and param == "compute_inverse_components" and param not in theirs
            and ours.get(param) is False):
        return "no pseudo-inverse of the components on either arm; cuML has no inverse option"
    # smoke-fixes (2026-10-14; lars: "jitter missing from cuml-gpu read-back"). cuML 26.08
    # linear_model/lars.pyx (__init__ lines 134-151, fit lines 202-222) takes no jitter and adds no
    # noise to y; ours _expansion_linear.py:1444 refuses any jitter but None (no noise).
    if algo == "lars" and param == "jitter" and param not in theirs and param in ours and ours[param] is None:
        return "no target jitter on either arm: ours None adds none, cuML has no jitter option"
    # smoke-fixes (2026-10-14; simple-imputer: "keep_empty_features missing from cuml-gpu
    # read-back", "fill_value is unset on ours and unset on cuml-gpu"). cuML 26.08
    # _thirdparty/sklearn/preprocessing/_imputation.py: transform drops columns whose statistic is
    # NaN unless strategy='constant' (lines 448-463); fill_value None means 0 for numeric input
    # (lines 336-344) and is read only by strategy='constant' (_dense_fit, lines 424-426). Ours
    # _expansion_prep.py:2226-2234: keep_empty_features False drops them; fill_value None means 0.
    if algo == "simple-imputer":
        same_stat = (ours.get("strategy") == theirs.get("strategy")
                     and ours.get("strategy") in ("mean", "median", "most_frequent"))
        if param == "keep_empty_features" and param not in theirs and ours.get(param) is False and same_stat:
            return "both drop columns with no observed fit value (non-constant strategy); strategy remains compared"
        if (param == "fill_value" and param in ours and param in theirs and ours[param] is None
                and theirs[param] is None):
            return "fill_value None means 0 on both and is read only by strategy='constant'"
    # smoke-fixes (2026-10-14; target-encoder: "categories, shuffle, target_type missing from
    # cuml-gpu read-back", "multi_feature_mode, split_method, stat missing from ours read-back").
    # cuML 26.08 preprocessing/_target_encoder.py: categories_ are the fit rows' sorted uniques
    # (lines 326-330); multi_feature_mode='independent' encodes each column alone over all fit
    # rows and never reads the fold column (_fit_transform_independent, lines 421-470,
    # _compute_single_feature_encoding lines 472-487: (sum + smooth * mean) / (count + smooth)).
    # Ours _expansion_prep.py:1901-1960 (TargetEncoder): categories 'auto' from the fit rows,
    # fit() (line 2099) builds no folds. The race times fit() then transform(); the driver refuses
    # the cuML arm for a timed fit_transform or a target that is not 0/1
    # (tools/bench_board_algos.py _est_factory, target-encoder).
    if algo == "target-encoder" and theirs.get("multi_feature_mode") == "independent":
        if param == "categories" and param not in theirs and ours.get(param) == "auto":
            return "categories from the fit rows' unique values on both (ours 'auto', cuML categories_)"
        if param == "multi_feature_mode" and param not in ours:
            return "cuML 'independent' encodes each column separately, as ours always does"
        if (param == "target_type" and param not in theirs and ours.get(param) == "binary"
                and theirs.get("stat") == "mean"):
            return "ours' binary positive-class rate is cuML's mean(y) on the 0/1 target the driver requires"
        if (param == "stat" and param not in ours and theirs.get(param) == "mean"
                and ours.get("target_type") == "binary"):
            return "cuML stat='mean' is ours' mean target encoding; smooth remains compared"
        if param == "split_method" and param not in ours:
            return "fold assignment is inactive: fit() cross-fits on neither arm (cuML independent mode never reads folds)"
        if param == "shuffle" and param not in theirs and isinstance(ours.get(param), bool):
            return "fold shuffling is inactive: ours' fit() builds no folds; cuML independent mode never reads folds"
    if lane == "pca" and ours.get("svd_solver") == theirs.get("svd_solver") == "covariance_eigh":
        if param in ("iterated_power", "n_oversamples", "power_iteration_normalizer"):
            return "covariance eigendecomposition on both arms: randomized/Jacobi iteration controls are inactive (cuML pca.pyx; mojolearn decomposition.py)"
    if lane == "kde":
        fixed = {"algorithm": "auto", "atol": 0.0, "rtol": 0.0, "breadth_first": True, "leaf_size": 40}
        if param in fixed and ours.get(param) == fixed[param] and param not in theirs:
            return "both implementations use exact brute-force KDE; ours refuses nondefault tree controls and cuML exposes no tree (density.py; kernel_density.pyx)"
        if param == "metric_params" and ours.get(param) is None and theirs.get(param) is None:
            return "None means no metric keyword overrides, p=2 on both implementations (density.py; kernel_density.pyx)"
    if lane == "kmeans":
        if param in ("device_buffer_samples", "init_size") and param not in ours and theirs.get("device_buffer_samples") == 0:
            return "cuML device_buffer_samples=0 uses the full in-memory dataset; init_size only applies to its out-of-core path (kmeans.pyx)"
        if param == "init_centroids" and ours.get(param) is None and param not in theirs:
            return "no supplied centroids; both use the separately compared init method (cluster.py; kmeans.pyx)"
        if param == "metric" and ours.get(param) == "euclidean" and param not in theirs:
            return "cuML KMeans fixes DistanceType.L2Expanded, matching our squared Euclidean objective (kmeans.pyx set_params)"
        if param == "max_samples_per_batch" and param not in ours and isinstance(theirs.get(param), int) and theirs[param] > 0:
            return "cuML distance-computation batch size partitions all rows without subsampling; library implementation tiling is retained, not a model setting (kmeans.pyx)"
    # FAIRNESS FIX (2026-10-13 audit): the classifier/regressor inherit
    # cuML's same NearestNeighbors implementation. Their inactive overrides
    # and complete-query tiling were incorrectly treated as model differences.
    # Labels, weights, k and metric remain ordinary compared parameters.
    if lane in ("knn", "knn-clf", "knn-reg") and ours.get("algorithm") == theirs.get("algorithm") == "brute":
        if param in ("algo_params", "metric_params") and theirs.get(param) is None and param not in ours:
            return "no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)"
        if param == "radius" and param not in ours:
            return "kneighbors uses the compared n_neighbors, never the radius-neighbors query option (nearest_neighbors.pyx)"
        if param == "query_tile" and isinstance(ours.get(param), int) and ours[param] >= 0 and param not in theirs:
            return "our query_tile partitions all queries without changing k or the exact distance metric; internal tiling is logged (neighbors.py)"
    # FAIRNESS FIX (2026-10-13 audit): retain implementation scheduling and
    # inactive callable-kernel arguments without inventing missing readback.
    # Reviewed against cuML 26.8 svm/linear_svc.py and
    # kernel_ridge/kernel_ridge.py::_get_kernel.
    if lane == "linearsvc" and param == "n_streams" and param not in ours:
        value = theirs.get(param)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return "cuML n_streams schedules fitting in parallel streams; loss, penalty, multiclass strategy and stopping controls are compared separately"
    if lane == "kernel-ridge" and param == "kernel_params" and param not in ours and theirs.get(param) is None:
        if ours.get("kernel") == theirs.get("kernel") and ours.get("kernel") in ("linear", "rbf", "poly", "polynomial", "sigmoid", "cosine"):
            return "cuML's named kernel uses gamma/degree/coef0; kernel_params=None supplies no callable-kernel overrides"
    # FAIRNESS FIX (2026-10-13 audit): fixed-point completion and default
    # output/cache controls were missing from one of these APIs. These are
    # value-conditional mappings, not permission to ignore active settings.
    # Sources: density.py; cuML 26.8 cluster/dbscan.pyx; hdbscan.py and
    # hdbscan/impl/runner.mojo; cuML cluster/hdbscan/hdbscan.pyx.
    if lane == "dbscan":
        if param == "max_iterations" and param in ours and ours[param] is None and param not in theirs:
            return "both propagate labels to the fixed point; cuML exposes no iteration cap, and our None requests completion"
        if param == "prediction_data" and ours.get(param) is False and param not in theirs:
            return "no additional prediction cache is requested; cuML DBSCAN exposes no prediction-data option"
        if param == "calc_core_sample_indices" and param not in ours and theirs.get(param) is True:
            return "our DBSCAN always produces core_sample_indices_ and components_; cuML is explicitly set to retain both"
        if param == "max_mbytes_per_batch" and param in ours and param in theirs and ours[param] is None and theirs[param] is None:
            return "both libraries choose distance-batch memory automatically; neither setting subsamples the input"
    if lane == "hdbscan" and theirs.get("build_algo") == "brute_force":
        if param == "build_algo" and param not in ours:
            return "our HDBSCAN fixes exhaustive neighbors; cuML explicitly selects the same exact brute-force graph construction"
        if param in ("build_kwds", "device_ids") and param not in ours and theirs.get(param) is None:
            return "cuML's exact graph uses the default unpartitioned single-device path; no approximate or multi-device overrides are requested"
        if param == "gen_min_span_tree" and param not in ours and theirs.get(param) is False:
            return "both construct the internal MST for clustering; neither requests an additional exported minimum-spanning-tree object"
        if param == "p" and param not in ours and theirs.get(param) is None and ours.get("metric") == theirs.get("metric") == "euclidean":
            return "p only parameterizes Minkowski distance; both arms explicitly use Euclidean distance"
    # smoke-fixes (2026-10-14; decision-tree-clf/-reg: "ccp_alpha, min_weight_fraction_leaf,
    # monotonic_cst, splitter, inference_engine missing from cuml-gpu read-back"; decision-tree-clf:
    # "class_weight is unset on ours and unset on cuml-gpu"): cuML's one-tree forest has none of
    # ours' scikit-learn tree options. Its own scikit-learn interop refuses every non-default value
    # (cuML 26.08 ensemble/randomforest_common.pyx BaseRandomForestModel._params_from_cpu:
    # ccp_alpha != 0, min_weight_fraction_leaf != 0 and monotonic_cst not None raise
    # UnsupportedOnGPU), so each maps only at the value that means "off". Ours: _expansion_trees.py
    # (ccp_alpha, min_weight_fraction_leaf and monotonic_cst accept only these values; splitter
    # 'best' searches every quantile bin of every candidate feature, as cuML's builder does).
    if lane in ("algos/decision-tree-clf", "algos/decision-tree-reg") and theirs.get("n_estimators") == 1 \
            and theirs.get("bootstrap") is False and ours.get("bootstrap") is False:
        if param not in theirs:
            if param == "ccp_alpha" and ours.get(param) == 0.0:
                return "no cost-complexity pruning on either arm: ours ccp_alpha=0.0; cuML has no pruning (randomforest_common.pyx refuses ccp_alpha != 0)"
            if param == "min_weight_fraction_leaf" and ours.get(param) == 0.0:
                return "no leaf weight-fraction floor on either arm: ours 0.0; cuML has none (randomforest_common.pyx refuses != 0); min_samples_leaf is compared"
            if param == "monotonic_cst" and param in ours and ours[param] is None:
                return "no monotonic constraints on either arm: ours None; cuML has none (randomforest_common.pyx refuses non-None)"
            if param == "splitter" and ours.get(param) == "best":
                return ("best split on both arms: ours splitter='best' and cuML's builder search every quantile-bin "
                        "threshold of every candidate feature (max_bin and max_features compared); cuML has no random splitter")
            if param == "inference_engine" and ours.get(param) in ("auto", "sequential"):
                # implementation only: the prediction kernel that walks the same fitted tree
                # (_expansion_trees.py: 'auto' predicts as 'sequential' in identical mode)
                return "ours' prediction-kernel choice walks the same fitted tree; cuML predicts its own forest; not a model setting"
        # refused-param-mappings (2026-10-15; replaces an unconditional EXCEPTIONS row): no leaf cap on
        # either arm. Ours and scikit-learn max_leaf_nodes None (a value switches both to best-first
        # growth; ours refuses it with splitter='best'), cuML max_leaves -1, its unlimited sentinel
        # (cuML 26.08 decisiontree.hpp); max_depth (compared) bounds every arm. Any other value on
        # either side is a real leaf cap and is compared.
        if param == "max_leaves" and param in ours and param in theirs and ours[param] is None \
                and theirs[param] == -1 and not isinstance(theirs[param], bool):
            return ("no leaf cap on either arm: ours max_leaf_nodes=None, cuML max_leaves=-1 (its unlimited "
                    "sentinel); max_depth compared")
        if param == "class_weight" and lane == "algos/decision-tree-clf" and param in ours and param in theirs \
                and ours[param] is None and theirs[param] is None:
            # cuML 26.08 common/classification.py process_class_weight: None returns the
            # sample_weight unchanged (None, unweighted); ours randomforest.py _class_weight_rows: None
            # gives no row weights. A dict of unit weights would route both through their weighted
            # kernels instead, so None is the explicit unweighted setting on both.
            return "None is unweighted on both (cuML process_class_weight; ours _class_weight_rows); no class weights"
    # refused-param-mappings (2026-10-15; replaces two unconditional EXCEPTIONS rows): cuML spells
    # squared error 'squared_loss' (MBSGD, cuML 26.08 linear_model/mbsgd_regressor.pyx) and 'l2'
    # (QN, linear_model/qn.pyx); ours spells it 'squared_error'. Only that pair maps; any other
    # loss on either arm is compared and refuses.
    _sq_loss = {"algos/sgd-reg": "squared_loss", "algos/qn-reg": "l2"}.get(lane)
    if _sq_loss and arm == "cuml-gpu" and param == "loss" and _r6_is(ours.get(param), "squared_error") \
            and _r6_is(theirs.get(param), _sq_loss):
        return "the same squared-error loss, spelled differently: ours 'squared_error', cuML %r" % _sq_loss
    # smoke-fixes (2026-10-14; sgd-clf/-reg: "epochs missing from ours read-back", "max_iter, average,
    # early_stopping, epsilon, warm_start (and sgd-clf class_weight) missing from cuml-gpu read-back"):
    # cuML 26.08 cpp/src/solver/sgd.cuh: `for (i = 0; i < epochs; i++)` runs full passes of
    # batch_size-row minibatches, and the stop rule is read only when tol > 0 (sgd.cuh:235); ours
    # x_linear/sgd.mojo minibatch fit runs `for epoch in range(max_iter)` full passes and reads its
    # stop rule only when tol is not None (-3e38 sentinel, _expansion_linear.py _sgd_fit). So
    # epochs == max_iter is the same pass count only when neither arm can stop early.
    if lane in ("algos/sgd-clf", "algos/sgd-reg"):
        no_stop = ours.get("tol", 1.0) is None and theirs.get("tol") == 0.0
        if param in ("epochs", "max_iter") and no_stop and isinstance(ours.get("max_iter"), int) \
                and ours.get("max_iter") == theirs.get("epochs") and "max_iter" not in theirs and "epochs" not in ours:
            return ("the same number of full passes: ours max_iter=%d, cuML epochs=%d, neither arm stops early "
                    "(ours tol=None, cuML tol=0.0; sgd.cuh reads n_iter_no_change only when tol > 0)"
                    % (ours["max_iter"], theirs["epochs"]))
        if param in theirs:
            return None
        if param == "average" and ours.get(param) is False:
            return "no iterate averaging on either arm: ours average=False; cuML sgd.cuh has no averaging"
        if param == "early_stopping" and ours.get(param) is False:
            return "no validation-split early stop on either arm: ours early_stopping=False; cuML MBSGD has none"
        if param == "warm_start" and ours.get(param) is False:
            return "both fits start from zero coefficients: ours warm_start=False; cuML sgd.pyx fit_sgd allocates cp.zeros each fit"
        if param == "epsilon" and ours.get("loss") in ("hinge", "log_loss", "squared_error"):
            return ("epsilon is read only by ours' huber/epsilon-insensitive/modified_huber losses; the raced loss "
                    "(compared) is %s, and cuML MBSGD has only hinge/log/squared_loss (sgd.pyx _LOSSES)" % ours["loss"])
        if param == "class_weight" and lane == "algos/sgd-clf" and param in ours and ours[param] is None:
            return "no class weights on either arm: ours class_weight=None; cuML MBSGDClassifier has no class weighting"
    return None


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def check(lane, arms, family=None, reference="ours", seed=None, extra_exceptions=(), pairing=None):
    """The report for one race. `arms`: {arm name: constructed object or
    declared dict}. `extra_exceptions`: (param, arm glob, reason) the driver
    adds for this race (the same shape as EXCEPTIONS without the lane).
    `seed`: None is `seed_for(lane)`.
    `pairing` (smoke-fixes r13, 2026-10-14): {own arm: [opponent arm globs]} of a lane that races
    more than one explicit own configuration (tools/board_pairing.py OUR_PAIRINGS). Each opponent
    is then compared with the own arm it is paired with, never with another; an own arm of the
    pairing other than `reference` is a separate configuration and is not compared with
    `reference` (its own settings are read back and recorded under `arms`)."""
    seed = seed_for(lane) if seed is None else seed
    resolved, sources, libs, raw_params, no_seed, draws = {}, {}, {}, {}, set(), set()
    for name, obj in arms.items():
        lib, source, raw = read_params(obj)
        libs[name], sources[name] = lib, source
        raw_params[name] = {k: _jsonable(v) for k, v in raw.items()}
        resolved[name] = canonical(lib, raw)
        if "seed" not in resolved[name] and has_no_seed_argument(obj, raw):
            no_seed.add(name)
    ref = reference if reference in resolved else next(iter(resolved), None)
    pairing = {own: list(globs) for own, globs in (pairing or {}).items()}
    refs = {}
    for name in resolved:
        if name in pairing:
            continue                        # an own configuration of the pairing: not compared
        own = board_pairing.reference_for(name, pairing, default=ref)
        refs[name] = own if own in resolved else ref
    problems, compared, applied = [], [], []
    if not resolved:
        problems.append("no parameter records")

    def _excused(param, arm):
        own = {k: v[0] for k, v in resolved.get(refs.get(arm, ref), {}).items()}
        theirs = {k: v[0] for k, v in resolved.get(arm, {}).items()}
        why = semantic_exception(lane, param, arm, own, theirs, libs.get(arm))
        if why is None:
            why = exception_for(lane, param, arm, own, theirs)
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
        if name == ref or name not in refs:
            continue
        rref = refs[name]
        for param in sorted(set(canon) | set(resolved[rref])):
            if param == "seed":
                continue
            if param not in canon or param not in resolved[rref]:
                # Only a centrally reviewed conditional mapping may explain an
                # absent API field. A driver-supplied exception cannot waive it.
                why = semantic_exception(lane, param, name,
                    {k: v[0] for k, v in resolved[rref].items()},
                    {k: v[0] for k, v in canon.items()}, libs[name])
                if why:
                    applied.append({"arm": name, "param": param,
                                    "ours": resolved[rref].get(param, (None,))[0],
                                    "theirs": canon.get(param, (None,))[0],
                                    "reason": why, "absent_api_field": True})
                    continue
                problems.append("%s: %s missing from %s read-back; explicit semantic mapping required"
                                % (name, param, name if param not in canon else rref))
                continue
            val, own_name = canon[param]
            rval, rname = resolved[rref][param]
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
                                % (name, param, "unset" if rval is None else repr(rval), rref,
                                   "unset" if val is None else repr(val), name))
            else:
                problems.append("%s: %s is %r (%s) on %s and %r (%s) on %s"
                                % (name, param, rval, rname, rref, val, own_name, name))
    if pairing:
        # each opponent's own reference arm (the "ours" value of its compared/exceptions rows)
        for row in compared + applied:
            if row.get("arm") in refs and row.get("param") != "seed":
                row["reference"] = refs[row["arm"]]
    return {"schema": "mojolearn.parameter-parity.v2", "comparison": "union of all work settings",
            "lane": lane, "family": family, "reference": ref, "seed": seed,
            **({"pairing": pairing, "references": dict(sorted(refs.items()))} if pairing else {}),
            "arms": {n: {"library": libs[n], "source": sources[n],
                         "implementation": implementation_of(arms[n]),
                         "raw_params": raw_params[n],
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


def enforce(lane, arms, family=None, reference="ours", seed=None, extra_exceptions=(), stream=None,
            pairing=None):
    """check + emit; raises ParamsRefused when the arms do not match."""
    report = check(lane, arms, family=family, reference=reference, seed=seed,
                   extra_exceptions=extra_exceptions, pairing=pairing)
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
