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


def _minus1(v):
    return None if v is None else v - 1


def _cuml_pca_solver(v):
    """cuML PCA's svd_solver 'full' is the eigendecomposition of the
    covariance matrix (cuml/decomposition/pca.pyx: 'auto' and 'full' both set
    solver.COV_EIG_DQ), scikit-learn's and ours' 'covariance_eigh'; its own
    scikit-learn interop maps any scikit-learn solver but 'auto' to 'full'.
    'jacobi' (COV_EIG_JACOBI) is another algorithm and keeps its name."""
    return "covariance_eigh" if v == "full" else v


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
    "cuvs": {"init_method": "init"},
    "cuml": {"n_bins": "max_bin", "n_folds": "cv",
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
    ("algos/decision-tree-*", "max_leaves", "cuml-gpu", "no leaf cap on any arm: ours and "
     "scikit-learn max_leaf_nodes None (a value would switch both to best-first growth; ours "
     "refuses it with splitter='best'), cuML max_leaves -1, its unlimited sentinel "
     "(decisiontree.hpp); depth 16 bounds every arm"),
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
     "API cannot switch off"),
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
    ("algos/sgd-clf", "max_iter", "cuml-gpu", "cuML MBSGD reads epochs=100, not max_iter"),
    ("algos/sgd-clf", "tol", "cuml-gpu", "cuML MBSGD tol=0.0 is its no-early-stop value; ours "
     "and scikit-learn tol=None"),
    ("algos/sgd-reg", "max_iter", "cuml-gpu", "cuML MBSGD reads epochs=100, not max_iter"),
    ("algos/sgd-reg", "tol", "cuml-gpu", "cuML MBSGD tol=0.0 is its no-early-stop value; ours "
     "and scikit-learn tol=None"),
    ("algos/sgd-reg", "loss", "cuml-gpu", "cuML spells squared error 'squared_loss'"),
    ("algos/qn-reg", "loss", "cuml-gpu", "cuML QN spells squared error 'l2'"),
    ("algos/qn-reg", "tol", "sklearn-cpu", "scikit-learn LinearRegression's tol is read only by "
     "the sparse lsqr path; on dense X it solves in closed form (scipy lstsq) and has no "
     "stopping tolerance"),
]


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
    """(library, source, {name: value}) of what the constructed arm holds."""
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
    return lib, "attributes", got


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


def exception_for(lane, param, arm):
    for lg, p, ag, why in EXCEPTIONS:
        if p == param and fnmatch.fnmatch(lane, lg) and fnmatch.fnmatch(arm, ag):
            return why
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
    if library == "cuvs" and arm == "hipvs-gpu":
        if lane == "kmeans":
            if param == "init" and ours.get(param) == "k-means++" and theirs.get(param) == "KMeansPlusPlus":
                return "same k-means++ initializer, spelled KMeansPlusPlus by hipVS KMeansParams; oversampling is compared separately"
            if param == "metric" and ours.get(param) == "euclidean" and theirs.get(param) == "sqeuclidean":
                return "both KMeans objectives minimize squared Euclidean distances; hipVS names the squared distance explicitly"
            if param == "init_centroids" and ours.get(param) is None and param not in theirs:
                return "no supplied initial centroids; hipVS fit(params, X) uses the separately compared init_method"
            if param in ("hierarchical", "hierarchical_n_iters") and theirs.get("hierarchical") is False:
                return "hipVS hierarchical=False selects ordinary KMeans; hierarchical_n_iters is inactive"
        if lane == "knn" and param == "query_tile" and ours.get("algorithm") == theirs.get("algorithm") == "brute" and isinstance(ours.get(param), int) and ours[param] >= 0 and param not in theirs:
            return "both brute-force searches cover every index/query pair; query_tile only partitions our complete query work"
        return None
    if library != "cuml" or arm != "cuml-gpu":
        return None
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
        if param in ("device_buffer_samples", "init_size") and theirs.get("device_buffer_samples") == 0:
            return "cuML device_buffer_samples=0 uses the full in-memory dataset; init_size only applies to its out-of-core path (kmeans.pyx)"
        if param == "init_centroids" and ours.get(param) is None and param not in theirs:
            return "no supplied centroids; both use the separately compared init method (cluster.py; kmeans.pyx)"
        if param == "metric" and ours.get(param) == "euclidean" and param not in theirs:
            return "cuML KMeans fixes DistanceType.L2Expanded, matching our squared Euclidean objective (kmeans.pyx set_params)"
        if param == "max_samples_per_batch" and isinstance(theirs.get(param), int) and theirs[param] > 0:
            return "cuML distance-computation batch size partitions all rows without subsampling; library implementation tiling is retained, not a model setting (kmeans.pyx)"
    if lane == "knn" and ours.get("algorithm") == theirs.get("algorithm") == "brute":
        if param in ("algo_params", "metric_params") and theirs.get(param) is None and param not in ours:
            return "no algorithm or distance keyword overrides on the explicit brute-force kneighbors call (nearest_neighbors.pyx)"
        if param == "radius" and param not in ours:
            return "kneighbors uses the compared n_neighbors, never the radius-neighbors query option (nearest_neighbors.pyx)"
        if param == "query_tile" and isinstance(ours.get(param), int) and ours[param] >= 0 and param not in theirs:
            return "our query_tile partitions all queries without changing k or the exact distance metric; internal tiling is logged (neighbors.py)"
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
    resolved, sources, libs, raw_params, no_seed, draws = {}, {}, {}, {}, set(), set()
    for name, obj in arms.items():
        lib, source, raw = read_params(obj)
        libs[name], sources[name] = lib, source
        raw_params[name] = {k: _jsonable(v) for k, v in raw.items()}
        resolved[name] = canonical(lib, raw)
        if "seed" not in resolved[name] and has_no_seed_argument(obj, raw):
            no_seed.add(name)
    ref = reference if reference in resolved else next(iter(resolved), None)
    problems, compared, applied = [], [], []
    if not resolved:
        problems.append("no parameter records")

    def _excused(param, arm):
        own = {k: v[0] for k, v in resolved.get(ref, {}).items()}
        theirs = {k: v[0] for k, v in resolved.get(arm, {}).items()}
        why = semantic_exception(lane, param, arm, own, theirs, libs.get(arm))
        if why is None:
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
        for param in sorted(set(canon) | set(resolved[ref])):
            if param == "seed":
                continue
            if param not in canon or param not in resolved[ref]:
                # Only a centrally reviewed conditional mapping may explain an
                # absent API field. A driver-supplied exception cannot waive it.
                why = semantic_exception(lane, param, name,
                    {k: v[0] for k, v in resolved[ref].items()},
                    {k: v[0] for k, v in canon.items()}, libs[name])
                if why:
                    applied.append({"arm": name, "param": param,
                                    "ours": resolved[ref].get(param, (None,))[0],
                                    "theirs": canon.get(param, (None,))[0],
                                    "reason": why, "absent_api_field": True})
                    continue
                problems.append("%s: %s missing from %s read-back; explicit semantic mapping required"
                                % (name, param, name if param not in canon else ref))
                continue
            val, own_name = canon[param]
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
    return {"schema": "mojolearn.parameter-parity.v2", "comparison": "union of all work settings",
            "lane": lane, "family": family, "reference": ref, "seed": seed,
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
