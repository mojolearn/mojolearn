#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OUR side of the gradient-boosting and forest speed run, and the one
process that interleaves it with the NVIDIA-native opponents.

    pixi run -e gbmbench python bench/speed/forest_speed_arm.py --lane rf
    MOJOLEARN_SPEED_SIZE=smoke \
      pixi run -e gbmbench python bench/speed/forest_speed_arm.py \
        --lane gbdt-symmetric

THE QUESTION THIS FILE ANSWERS
-------------------------------
Measure the explicitly selected numeric mode against competitors on the
same accelerator. NVIDIA comparisons use IDENTICAL; FAST decision-tree
performance runs on MacBook remain supported. Before timing, require the
estimator's resolved mode and its own native compiled-mode/vendor readbacks
to agree with the requested mode and vendor. A namespace or environment
label alone is not compiled-mode evidence. Quality is reported alongside
full public fit time; cross-library model identity is not asserted.

HOW WE ARE INVOKED, AND WHY THROUGH PYTHON
--------------------------------------------
Through the Python bindings, because that is how `gbdt/`, `ensemble/`,
`extratrees/` and `isolation_forest/` are actually reached from outside
Mojo: `bindings/_mojolearn_gbdt.mojo`, `_mojolearn_rf.mojo`,
`_mojolearn_trees.mojo` and `_mojolearn_svm.mojo` (which carries the
isolation forest) are built into extensions by `bindings/build_*.sh`, and
`python/mojolearn/` is the surface every existing comparison already uses --
`bench/external/gbm_bench/mojolearn_algorithm.py` calls exactly these
classes. Shelling a Mojo entry point instead would time a different program
than the one a user runs and would have no accuracy column at all.

The cost of that choice is named rather than hidden. See DEVIATION 1840
below: our `fit` transposes X to column-major INSIDE the timer, and that
copy is a real part of what a caller of this surface pays.

WHAT LIVES WHERE
-----------------
Everything shared -- the datasets, the hyper-parameter tables, the output
contract, the metrics, the runner -- lives in `tools/speed_gbdt_arm.py`,
which imports NOTHING from mojolearn. This file adds the `ours` arm and the
lane wiring. The direction is deliberate: on a box whose CUDA
build of `ensemble/` has failed, the opponents still run from that file
alone and the lease is not wasted.

ONE LANE PER PROCESS
---------------------
`--lane` takes exactly one lane. CUDA forest measurements already exist in
`bench/results/trees_identical/`; a new build still needs its own mode/vendor
witness and successful fit. Separate lane processes contain build or runtime
failures without losing the other comparisons.

INTERLEAVED BY DEFAULT, SEPARABLE ON PURPOSE
---------------------------------------------
By default this process runs ours AND the opponents, ALTERNATING one round
each. That is the only comparison format this repository quotes: a rented
box throttles both arms together, and a ratio survives what an absolute
number does not.

`--ours-only` exists for the case the interleaving cannot survive: mojolearn
reaches CUDA through MAX's runtime while cuML, CatBoost-GPU and XGBoost-GPU
reach it through their own, and two runtimes contending for one context in
one process is a plausible way to lose an hour. If the interleaved run
crashes, fall back to `--ours-only` here plus `tools/speed_gbdt_arm.py
--lane <same>` in a second process, and say in the writeup that the ratio
then spans two processes and is exposed to drift between them.

DEVIATION 1840, THE TRANSPOSE, AND WHY IT IS LEFT IN
------------------------------------------------------
`GradientBoosting.fit` and the forests' `_fit_arrays` both call
`np.asfortranarray` on X, because the builders are column-major inside
(cuML's `data` is column-major, and CatBoost's pool is). On an 800,000 x 100
float32 matrix that is a 320 MB host copy, INSIDE the timer.

It stays inside the timer, because a user calling this Python surface pays
it and a benchmark that deletes a cost the user cannot delete is measuring
something nobody can buy. `MOJOLEARN_SPEED_FORTRAN=1` hands our arm an
already-Fortran-ordered copy, prepared once outside every timer, so the
transpose can be PRICED rather than argued about. Run it both ways and the
difference is the number. The opponents always receive the C-ordered array,
which is what every one of them documents as its preferred layout.
"""

import argparse
import os
import sys
import time

import numpy as np

# `tools/` is not a package and never has been, so the spec is imported by
# path rather than by name. The repository root is two levels up from this
# file (bench/speed/ -> bench/ -> root), and `python/` goes on the path too
# so an in-repo, not-yet-installed `mojolearn` is importable exactly the way
# `bench/external/run_gbm_bench.sh` arranges it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
#
# MOJOLEARN_BENCH_INSTALLED=1 (tools/bench_board.py) keeps `python/` OFF the
# path, so `import mojolearn` resolves to the INSTALLED wheel. The shipped
# tree carries `python/mojolearn/` without compiled bindings, so on a box that
# measures the PyPI wheel the in-repo copy would shadow the wheel and then
# refuse for a missing .so (or, worse, time a stale local build).
_PATHS = [os.path.join(_ROOT, "tools")]
if os.environ.get("MOJOLEARN_BENCH_INSTALLED", "0").strip() in ("", "0"):
    _PATHS.append(os.path.join(_ROOT, "python"))
for _p in _PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec           # noqa: E402


#: What each lane calls in `python/mojolearn/`, for the report and for the
#: `--list-arms` output. `training/` is absent on purpose: it is the neural
#: training-step lane and it has no boosting or forest estimator, so there is
#: no entry point here to time.
#:
#: CORRECTED 2026-08-31. This comment used to add that, as of 2026-08-25,
#: `archive/plans/training/TRAINING_LOOP_PLAN.md` stated in its own first paragraph that no
#: `mojo` process had ever read any of its three files. Commit `5ce6eb17`
#: falsified that plan's banner (the lane compiles and its step gate ran green
#: on one device) and the banner is now corrected in place. The reason
#: `training/` is absent from THIS file never depended on it: no forest, no
#: boosting, nothing to time.
OUR_ENTRY_POINTS = {
    "gbdt-symmetric": "mojolearn.GradientBoosting(grow_policy='SymmetricTree')"
                      " -> gbdt/ via _mojolearn_gbdt",
    "gbdt-symmetric-1000": "mojolearn.GradientBoosting(grow_policy='SymmetricTree',"
                           " n_estimators=1000) -> gbdt/ via _mojolearn_gbdt",
    "gbdt-depthwise": "mojolearn.GradientBoosting(grow_policy='Depthwise')"
                      " -> gbdt/ via _mojolearn_gbdt",
    "gbdt-lossguide": "mojolearn.GradientBoosting(grow_policy='Lossguide')"
                      " -> gbdt/ via _mojolearn_gbdt",
    "rf": "mojolearn.RandomForest{Classifier,Regressor}(device='gpu')"
          " -> ensemble/ via _mojolearn_rf",
    "et": "mojolearn.ExtraTrees{Classifier,Regressor}(device='gpu')"
          " -> extratrees/ via _mojolearn_trees",
    "iforest": "mojolearn.IsolationForest()"
               " -> isolation_forest/ via _mojolearn_svm",
    "gbdt-rank-yetirank": "mojolearn.GradientBoosting(loss='YetiRank',"
                          " grow_policy='SymmetricTree').fit(X, y, group_id=qid)"
                          " -> gbdt/ via _mojolearn_gbdt",
    "gbdt-rank-pairlogit": "mojolearn.GradientBoosting(loss='PairLogit',"
                           " grow_policy='SymmetricTree').fit(X, y, group_id=qid)"
                           " -> gbdt/ via _mojolearn_gbdt",
    "gbdt-multiclass": "mojolearn.GradientBoosting(loss='MultiClass',"
                       " grow_policy='SymmetricTree') -> gbdt/ via _mojolearn_gbdt",
    "gbdt-categorical": "mojolearn.GradientBoosting(grow_policy='Lossguide',"
                        " cat_features=[...]) -> gbdt/ via _mojolearn_gbdt",
    "gbdt-ordered": "mojolearn.GradientBoosting(grow_policy='SymmetricTree',"
                    " boosting_type='Ordered') -> gbdt/ via _mojolearn_gbdt",
}


def _fortran_requested():
    return os.environ.get("MOJOLEARN_SPEED_FORTRAN", "0").strip() not in (
        "", "0", "no", "false")


def prepare_our_inputs(data):
    """Everything our arm needs in host memory, in the dtype and layout it
    will consume, prepared ONCE and outside every timer.

    What is NOT prepared here is the Fortran copy, unless
    `MOJOLEARN_SPEED_FORTRAN=1` asks for it. See DEVIATION 1840 in the module
    docstring: the transpose is a real cost of this surface and deleting it
    silently would flatter us by exactly the size of a 320 MB memcpy."""
    x = np.ascontiguousarray(data.X_train, dtype=np.float32)
    data._ours_X = np.asfortranarray(x) if _fortran_requested() else x
    data._ours_Xtest = np.ascontiguousarray(data.X_test, dtype=np.float32)
    data._ours_y = np.ascontiguousarray(data.y_train, dtype=np.float32)


# --------------------------------------------------------------------------
# The `ours` arm, one builder per lane.
# --------------------------------------------------------------------------

def our_gbdt_arm(lane, cfg, data, extra=None):
    """`mojolearn.GradientBoosting`, the CatBoost GPU tree learner implementation.

    These explicit controls are shared with the CatBoost arm. Other defaults
    and searcher dispatch can differ; this is not algorithm equivalence:

        n_estimators <- iterations       max_depth   <- depth
        learning_rate                    l2_leaf_reg <- reg lambda
        border_count (BORDERS, not bins) grow_policy
        bootstrap_type='No'              random_state <- random_seed

    `boost_from_average` is passed explicitly since 2026-09-29 (same seed,
    same tuning parameters): True for RMSE and False for Logloss, the
    data-dependent default both libraries resolve (`check-bfa-oracle` proves
    our bias bit-equal to their `get_scale_and_bias`), and the value XGBoost
    and LightGBM are now pinned to as well.

    MULTICLASS IS REFUSED BY NAME (DEVIATION 1838). The Mojo layer implements
    `MultiClass`; this Python wrapper is one-dimensional, so a 7-class
    dataset cannot be run by this arm at all. It refuses rather than
    silently fitting a different problem than the CatBoost arm beside it."""
    import mojolearn

    # The task lanes (speed_gbdt_arm.TASK_LANES) name their loss: YetiRank or
    # PairLogit on query groups, MultiClass, Logloss with categorical columns.
    # MultiClass is on the public surface now (0.8.22), so the refusal below
    # binds the original binary/regression lanes only.
    task_loss = cfg.get("loss")
    if data.task == "multiclass" and task_loss != "MultiClass":
        raise RuntimeError(
            "mojolearn.GradientBoosting has no MultiClass on the Python "
            "surface (ensemble.py's _UNREACHABLE_LOSSES); run --dataset "
            "covtype2 or year instead of a 7-class task"
        )
    params = dict(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        l2_leaf_reg=cfg["l2"],
        border_count=cfg["borders"],
        random_state=cfg["seed"],
        bootstrap_type="No",                 # DEVIATION 1833
        grow_policy=cfg["grow_policy"],
        loss=task_loss or ("RMSE" if data.task == "regression" else "Logloss"),
        # SAME SEED, SAME TUNING PARAMETERS (speed_gbdt_arm.lane_config): every
        # knob the opponents also have, set to the value they are given
        max_leaves=cfg["max_leaves"],        # 2 ** depth, accepted off Lossguide
        min_data_in_leaf=cfg["min_data_in_leaf"],
        random_strength=cfg["random_strength"],
        score_function=cfg["score_function"],
        leaf_estimation_method=cfg["leaf_estimation_method"],
        leaf_estimation_iterations=cfg["leaf_estimation_iterations"],
        feature_border_type=cfg["feature_border_type"],
        nan_mode=cfg["nan_mode"],
        boosting_type=cfg["boosting_type"],
    )
    if cfg.get("min_split_gain") is not None:
        params["min_split_gain"] = cfg["min_split_gain"]
    if cfg.get("min_child_hessian") is not None:
        params["min_child_hessian"] = cfg["min_child_hessian"]
    bfa = spec.boost_from_average_for(data)
    if bfa is not None:
        params["boost_from_average"] = bfa
    spw = spec.scale_pos_weight_for(cfg, data)
    if spw is not None:
        # gbm-bench's scale_pos_weight on the other arms: CatBoost's own
        # equivalent is class weights [1, scale_pos_weight]
        params["class_weights"] = [1.0, spw]
    if data.cat_idx:
        # DEVIATION 2634's OTHER SIDE. `cat_features` is the only way to reach
        # the CTR target prep at all: with no categorical column 2634 skips it,
        # which is the only path anything had ever measured (its 0.9603 verdict
        # of 2026-09-12 came entirely from the skip side, on taxi and
        # Istella-S, neither of which has a categorical column). criteo spans 3
        # to 371,237 distinct per column, so CatBoost's own dispatch
        # (binarizations_manager.cpp:106-115) takes one-hot on the small
        # columns and target statistics on the large ones, and ours follows it.
        #
        # The codes must be DENSE 0..k-1, which `_decode_criteo` guarantees by
        # assigning them as the rank in sorted-unique order.
        #
        # NOTE the refusal this must not trip: `ensemble.py` raises
        # NotImplementedError for feature_fraction < 1 together with
        # cat_features. cfg pins no bagging (DEVIATION 1833) and never sets
        # feature_fraction, so the default 1.0 stands -- but an `--ours-ab`
        # that lowers it on criteo will refuse by name rather than silently
        # dropping the categorical path.
        params["cat_features"] = list(data.cat_idx)
    if extra:
        # `--ours-ab`: one estimator keyword changed, everything else equal.
        params.update(extra)

    def make():
        return mojolearn.GradientBoosting(**params)

    if data.task == "ranking":
        # `group_id` is the query id per row; the run lengths are formed
        # inside fit, inside the clock, as CatBoost's Pool is on its arm.
        def fit(model, d):
            return model.fit(d._ours_X, d._ours_y, group_id=d.qid_train)

        def score(model, d):
            return spec.score_ranking(model.predict(d._ours_Xtest), d)

        return spec.Arm("ours", make, fit, score, sync=_our_sync, library="mojolearn")
    return spec.Arm("ours", make, _our_fit, _our_score,
                    sync=_our_sync, library="mojolearn")


def our_rf_arm(lane, cfg, data, extra=None):
    """`mojolearn.RandomForest*`, the cuML RandomForest implementation (`ensemble/`).

    `device='gpu'` is explicit and never 'auto', so a run that cannot reach
    the accelerator fails loudly instead of quietly reporting a host number
    under a GPU label. There is no CPU arm here because the library has no
    CPU path at all -- `kernel_matrix.mojo` says "There is no CPU column."

    `n_streams` is left at each library's public default. Explicit one-stream
    comparisons must configure both arms. Mojolearn currently pipelines trees
    on one queue; cuML uses its CUDA stream pool, so these controls do not imply
    identical scheduling semantics."""
    import mojolearn

    common = dict(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        max_features=spec.max_features_for(data),
        n_bins=cfg["n_bins"],
        min_samples_leaf=cfg["min_samples_leaf"],
        min_samples_split=cfg["min_samples_split"],
        min_impurity_decrease=cfg["min_impurity_decrease"],
        bootstrap=cfg["bootstrap"],
        max_samples=cfg["max_samples"],   # 1.0: n draws with replacement, as sklearn's
        random_state=cfg["seed"],
        device="gpu",
    )
    if extra:
        common.update(extra)       # `--ours-ab`: one keyword changed

    def make():
        if data.task == "regression":
            return mojolearn.RandomForestRegressor(
                criterion="squared_error", **common)
        return mojolearn.RandomForestClassifier(criterion="gini", **common)

    return spec.Arm("ours", make, _our_fit, _our_score,
                    sync=_our_sync, library="mojolearn")


def our_et_arm(lane, cfg, data, extra=None):
    """`mojolearn.ExtraTrees*`, the cuML-design ExtraTrees implementation
    (`extratrees/`).

    IT TAKES NO `n_bins`, and that is not an omission on this side. Extremely
    randomized trees draw a UNIFORM RANDOM threshold inside each candidate
    feature's observed range; there is no per-feature quantile grid to size.
    The lane's `n_bins` therefore reaches the `rf` arms and not these, which
    is why `rf` and `et` are separate lanes rather than one forest lane with
    a flag."""
    import mojolearn

    common = dict(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        max_features=spec.max_features_for(data),
        min_samples_leaf=cfg["min_samples_leaf"],
        min_samples_split=cfg["min_samples_split"],
        min_impurity_decrease=cfg["min_impurity_decrease"],
        bootstrap=cfg["bootstrap"],
        random_state=cfg["seed"],
        device="gpu",
    )
    if extra:
        common.update(extra)       # `--ours-ab`: one keyword changed

    def make():
        if data.task == "regression":
            return mojolearn.ExtraTreesRegressor(
                criterion="squared_error", **common)
        return mojolearn.ExtraTreesClassifier(criterion="gini", **common)

    return spec.Arm("ours", make, _our_fit, _our_score,
                    sync=_our_sync, library="mojolearn")


def our_iforest_arm(lane, cfg, data, extra=None):
    """`mojolearn.IsolationForest`, the cuML IsolationForest implementation.

    WHAT `fit` ACTUALLY DOES HERE, AND IT CHANGES WHAT THIS ROW MEANS
    (DEVIATION 874, restated as DEVIATION 1836 for this harness). The forest
    is NOT KEPT. `fit` builds the forest over X and scores ONE row, which is
    the cheapest call the entry point accepts, and then every later scoring
    call REBUILDS IT. So:

      * the timed number IS a real full forest build over the training rows,
        plus a one-row scoring pass, and is comparable to sklearn's `fit`;
      * the ACCURACY column comes from a DIFFERENT forest than the one that
        was timed, because `score_samples` built its own;
      * repeated-build hashes must be interpreted under the selected mode;
        rebuilding alone does not imply that IDENTICAL hashes may change.

    None of that is corrected here. It is a property of the surface under
    test and correcting it in the harness would measure a library that does
    not exist."""
    import mojolearn

    params = dict(
        n_estimators=cfg["n_estimators"],
        max_samples=cfg["max_samples"],
        max_features=cfg["max_features"],
        bootstrap=cfg["bootstrap"],
        contamination="auto",
        random_state=cfg["seed"],
    )
    if extra:
        # `--ours-ab`: one keyword changed (numeric_mode='fast' is the Apple
        # board's FAST arm, interleaved beside IDENTICAL).
        params.update(extra)

    def make():
        return mojolearn.IsolationForest(**params)

    def fit(model, d):
        return model.fit(d._ours_X)

    def score(model, d):
        s = -np.asarray(model.score_samples(d._ours_Xtest), dtype=np.float64)
        return [("auc", spec.auc(d.y_anom, s), s)]

    return spec.Arm("ours", make, fit, score,
                    sync=_our_sync, library="mojolearn")


def _our_fit(model, data):
    return model.fit(data._ours_X, data._ours_y)


def _our_sync():
    """A named no-op.

    Every one of these bindings returns HOST arrays -- the forests hand back
    `offsets`, `colid`, `quesval`, `left_child`, `leaves`, and the GBDT hands
    back a model object -- so the call cannot return before the device
    finished producing them. The synchronization proof is the return itself.
    Spelled out rather than left as `None` so that reading the arm table
    tells you which arms were checked and which were assumed."""
    return None


def _our_score(model, data):
    """Scored exactly like every opponent, through the same helper, on the
    same held-out rows. Our surface is scikit-learn-shaped on purpose, and
    the one place it is not -- `GradientBoosting.predict` returns RAW SCORES
    for every loss, as CatBoost's does -- is handled by scoring through
    `predict_proba`, which applies the link, for the classification tasks."""
    view = _TestView(data)
    return spec.score_sklearn_like(model, view)


class _TestView(object):
    """The scoring helper reads `X_test`, `y_test` and `task`; hand it our
    already-prepared float32 test block so no arm is charged for a cast at
    scoring time either."""

    def __init__(self, data):
        self.X_test = data._ours_Xtest
        self.y_test = data.y_test
        self.task = data.task
        self.n_classes = data.n_classes


OUR_BUILDERS = {
    "gbdt-rank-yetirank": our_gbdt_arm,
    "gbdt-rank-pairlogit": our_gbdt_arm,
    "gbdt-multiclass": our_gbdt_arm,
    "gbdt-categorical": our_gbdt_arm,
    "gbdt-ordered": our_gbdt_arm,
    "gbdt-symmetric": our_gbdt_arm,
    "gbdt-symmetric-1000": our_gbdt_arm,
    "gbdt-depthwise": our_gbdt_arm,
    "gbdt-lossguide": our_gbdt_arm,
    "rf": our_rf_arm,
    "et": our_et_arm,
    "iforest": our_iforest_arm,
}


def verify_our_arm(arm, requested=None):
    """Resolve and verify this arm before any fit timer starts. `requested`
    overrides the environment's tier for an `--ours-ab numeric_mode=...` arm."""
    requested = (requested or os.environ.get("MOJOLEARN_NUMERIC_MODE", "")).strip().lower()
    codes = {0: "fast", 1: "identical", 2: "deterministic"}
    if requested not in codes.values():
        raise RuntimeError("set MOJOLEARN_NUMERIC_MODE explicitly before benchmarking")
    model = arm.make()
    resolved = model.numeric_mode_used()
    vendor = model.vendor_used()
    # FOUND 2026-09-11 (lane forest-speed, H100): IsolationForest inherits the
    # mixin's `_BINDING = "_mojolearn"` while its code is compiled into the
    # svm extension, so this readback asked the base binding for
    # `_numeric_mode` and REFUSED our iforest arm on taxi and Istella-S.
    # Its tier and vendor are read from `_mojolearn_svm` (`svm_numeric_mode`).
    binding_name = model._BINDING
    if type(model).__name__ == "IsolationForest":
        binding_name = "_mojolearn_svm"
    binding = model._bind(binding_name)
    prefix = binding_name.removeprefix("_mojolearn_")
    getter = getattr(binding, prefix + "_numeric_mode", None)
    if getter is None:
        raise RuntimeError("native compiled-mode readback missing for " + model._BINDING)
    compiled = codes.get(int(getter()), "unknown")
    expected_vendor = os.environ.get("MOJOLEARN_SPEED_EXPECTED_VENDOR", "").strip().lower()
    if expected_vendor not in ("cuda", "metal", "hip"):
        raise RuntimeError("set MOJOLEARN_SPEED_EXPECTED_VENDOR to cuda, metal, or hip")
    if resolved != requested or compiled != requested or vendor != expected_vendor:
        raise RuntimeError("mode/vendor mismatch: requested=%s/%s resolved=%s compiled=%s/%s"
                           % (requested, expected_vendor, resolved, compiled, vendor))
    print("BENCH_BINDING arm=%s requested=%s resolved=%s compiled=%s vendor=%s path=%s"
          % (arm.name, requested, resolved, compiled, vendor, binding.__file__), flush=True)
    # CONTRIBUTING.md (Non-default paths): the compiled side of every opt-in
    # switch this binding carries, beside the timing (DEVIATIONS 2550, 2551,
    # 2580, 2581).
    paths = getattr(binding, prefix + "_per_round_paths", None)
    if paths is not None:
        print("BENCH_PATHS arm=%s %s" % (arm.name, paths()), flush=True)
    return dict(requested=requested, resolved=resolved, compiled=compiled,
                vendor=vendor, path=binding.__file__)


def build_ours(lane, cfg, data, name="ours", extra=None):
    """Our verified arm, or an explicit import/mode/vendor refusal. `extra`
    (the `--ours-ab` arm) changes one GBDT estimator keyword."""
    try:
        if extra:
            if lane not in ("rf", "et", "iforest") and not lane.startswith("gbdt-"):
                raise RuntimeError("--ours-ab reaches the gbdt, rf, et and iforest lanes only")
            arm = OUR_BUILDERS[lane](lane, cfg, data, extra=extra)
            arm.name = name
        else:
            arm = OUR_BUILDERS[lane](lane, cfg, data)
        verify_our_arm(arm, requested=(extra or {}).get("numeric_mode"))
        return [arm]
    except Exception as exc:                       # noqa: BLE001
        spec.emit_refused(lane, name, "%s: %s"
                          % (exc.__class__.__name__,
                             " ".join(str(exc).split())))
        return []


# --------------------------------------------------------------------------
# INFERENCE (`--infer`, OFF by default: without it this file's output is
# unchanged). After the fit rounds, every arm predicts with ITS OWN last model
# from those rounds (no fit is retimed), on the same rows, in the same output
# kind its accuracy column scores: P(class 1) for a binary task, the value for
# regression, `score_samples` for iforest. One untimed warm-up per arm and
# batch, then the rounds, arms interleaved. Lines (all new heads, so a parser
# of the fit lines never sees them):
#
#   FSPEED-INFER-PATH    lane arm call          what each arm's clock covers
#   FSPEED-INFER-WARMUP  lane arm batch rows ms
#   FSPEED-INFER         lane arm batch rows round ms hash
#   FSPEED-INFER-ACC     lane arm batch metric value   (same metric as FSPEED-ACC)
#   FSPEED-INFER-AGREE   lane batch arms rows bits_equal max_abs_diff  (ours vs ours-ab)
#   FSPEED-INFER-REFUSED lane arm batch reason
#   FSPEED-INFER-NOTE    lane arm note
#
# Batches: `test` is the held-out split the accuracy column scores; `large` is
# the first `--infer-large-rows` (default 1,000,000) training rows, a bigger
# batch from the same loader. Every clock is host rows in, host predictions
# out: an opponent whose fastest documented path is on its device copies the
# rows up and the result back INSIDE its clock, as ours does.
# --------------------------------------------------------------------------

INFER_LARGE_ROWS = 1_000_000
#: the dataset of this process's inference phase, for a proxy arm's batch
_INFER_DATA = [None]


def _p1(raw):
    a = np.asarray(raw, dtype=np.float64)
    if a.ndim == 2:
        return np.ascontiguousarray(a[:, 1] if a.shape[1] > 1 else a[:, 0])
    return np.ascontiguousarray(a.reshape(-1))


def _vec(raw):
    return np.ascontiguousarray(np.asarray(raw, dtype=np.float64).reshape(-1))


def _mat(raw):
    """A multiclass probability matrix, (rows, n_classes) float64."""
    return np.ascontiguousarray(np.asarray(raw, dtype=np.float64))


def infer_spec(arm_name, lane, task, model, n_features=None, frame=None):
    """(call(X) -> raw, post(raw) -> float64 vector, call text) for one arm.
    `call` is what the clock covers; `post` (a host dtype view) runs outside
    it. The call text names the path and why it is the library's fastest
    documented one. It never contains '=' (the lines are key=value)."""
    iforest = lane == "iforest"
    binary = task == "binary"
    if hasattr(model, "board_infer_spec"):
        # a board arm in another process (forest_board_arms.py, `--ours-cpu`)
        return model.board_infer_spec(lane, task, _INFER_DATA[0])
    if task not in ("binary", "regression", "multiclass", "ranking") and not iforest:
        raise RuntimeError("inference timing covers binary, regression, multiclass and "
                           "ranking tasks; %s is %s" % (lane, task))
    if task in ("multiclass", "ranking"):
        return _task_infer_spec(arm_name, task, model, n_features)
    if frame is not None:
        # gbdt-categorical: CatBoost and XGBoost predict from the frame kind
        # their fit took, built from the host rows INSIDE the clock (as the
        # fit clock builds it); ours and LightGBM take the float32 codes.
        return _categorical_infer_spec(arm_name, model, frame)
    if arm_name in ("ours", "ours-ab"):
        if iforest:
            return (model.score_samples, _vec,
                    "mojolearn IsolationForest.score_samples(X) (the forest is rebuilt "
                    "inside every scoring call, DEVIATION 874, so this clock includes a "
                    "forest build)")
        if binary:
            return (model.predict_proba, _p1,
                    "mojolearn %s.predict_proba(X), column 1" % type(model).__name__)
        return (model.predict, _vec, "mojolearn %s.predict(X)" % type(model).__name__)
    if arm_name.startswith("catboost-"):
        tt = "GPU" if arm_name.endswith("-gpu") else "CPU"
        note = ""
        if tt == "GPU":
            # CatBoost's own GPU apply (predict's task_type). Probed once on
            # two rows outside every clock; a build that refuses it applies
            # on the CPU and the call text says so.
            try:
                probe = np.zeros((2, n_features or model.n_features_in_), dtype=np.float32)
                (model.predict_proba if binary else model.predict)(probe, task_type="GPU")
            except Exception as exc:              # noqa: BLE001
                tt = "CPU"
                note = " (task_type GPU refused: %s)" % " ".join(str(exc).split())[:80]
        if binary:
            return ((lambda X: model.predict_proba(X, task_type=tt)), _p1,
                    "catboost predict_proba(X, task_type %s, thread_count -1)%s, column 1"
                    % (tt, note.replace("=", ":")))
        return ((lambda X: model.predict(X, task_type=tt)), _vec,
                "catboost predict(X, task_type %s, thread_count -1)%s" % (tt, note.replace("=", ":")))
    if arm_name.startswith("xgboost-"):
        booster = model.get_booster()
        what = "probability" if binary else "value"
        if arm_name.endswith("-gpu"):
            # XGBoost's documented fastest path is inplace_predict with the
            # data on the booster's device; host rows on a CUDA booster fall
            # back to a DMatrix. So the rows go up as a cupy array and the
            # result comes back, both inside the clock.
            try:
                import cupy
            except ImportError:
                return (booster.inplace_predict, _vec,
                        "xgboost Booster.inplace_predict(host X) on a CUDA booster (no cupy "
                        "here, so XGBoost's own device-mismatch fallback), %s" % what)
            return ((lambda X: cupy.asnumpy(booster.inplace_predict(cupy.asarray(X)))), _vec,
                    "xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, "
                    "the rows uploaded and the %s copied back inside the clock" % what)
        return (booster.inplace_predict, _vec,
                "xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented "
                "fastest path), %s" % what)
    if arm_name.startswith("lightgbm-"):
        booster = model.booster_
        return (booster.predict, _vec,
                "lightgbm Booster.predict(X) (%s; LightGBM predicts on the CPU whatever "
                "device trained it)" % ("probability" if binary else "value"))
    if arm_name.startswith("sklearn-"):
        if iforest:
            return (model.score_samples, _vec, "sklearn IsolationForest.score_samples(X), n_jobs -1")
        if binary:
            return (model.predict_proba, _p1,
                    "sklearn %s.predict_proba(X), n_jobs -1, column 1" % type(model).__name__)
        return (model.predict, _vec, "sklearn %s.predict(X), n_jobs -1" % type(model).__name__)
    if arm_name.startswith("cuml-"):
        if iforest:
            return (model.score_samples, _vec, "cuml IsolationForest.score_samples(host X)")
        # cuML's forest inference is FIL. The model is converted once, outside
        # every clock (a model load); predict then runs FIL on host rows.
        try:
            fil = model.convert_to_fil_model()
            fn = fil.predict_proba if binary else fil.predict
            text = "cuml RandomForest.convert_to_fil_model() once untimed, then FIL %s(host X)" % (
                "predict_proba" if binary else "predict")
        except Exception as exc:                  # noqa: BLE001
            fn = model.predict_proba if binary else model.predict
            text = "cuml RandomForest.%s(host X) (FIL conversion refused: %s)" % (
                "predict_proba" if binary else "predict",
                " ".join(str(exc).split())[:80].replace("=", ":"))
        return (fn, _p1 if binary else _vec, text)
    raise RuntimeError("no inference path wired for arm %s" % arm_name)


def _task_infer_spec(arm_name, task, model, n_features):
    """Inference for the multiclass and ranking lanes: the probability
    matrix (multiclass) or the raw ranking scores, each library's own call."""
    multi = task == "multiclass"
    post = _mat if multi else _vec
    what = "the (rows, n_classes) probability matrix" if multi else "raw ranking scores"
    if arm_name in ("ours", "ours-ab"):
        fn = model.predict_proba if multi else model.predict
        return (fn, post, "mojolearn GradientBoosting.%s(X), %s"
                % ("predict_proba" if multi else "predict", what))
    if arm_name.startswith("catboost-") and not multi:
        # CatBoostRanker.predict takes no task_type (catboost 1.2.10): its
        # apply runs on the host whatever device trained it.
        return ((lambda X: model.predict(X, thread_count=-1)), post,
                "catboost CatBoostRanker.predict(X, thread_count -1) (no task_type on the "
                "ranker: a host apply on every box), %s" % what)
    if arm_name.startswith("catboost-"):
        tt = "GPU" if arm_name.endswith("-gpu") else "CPU"
        note = ""
        if tt == "GPU":
            try:
                probe = np.zeros((2, n_features or model.n_features_in_), dtype=np.float32)
                model.predict_proba(probe, task_type="GPU")
            except Exception as exc:              # noqa: BLE001
                tt = "CPU"
                note = " (task_type GPU refused: %s)" % " ".join(str(exc).split())[:80]
        return ((lambda X: model.predict_proba(X, task_type=tt)), post,
                "catboost predict_proba(X, task_type %s)%s, %s"
                % (tt, note.replace("=", ":"), what))
    if arm_name.startswith("xgboost-"):
        booster = model.get_booster()
        if arm_name.endswith("-gpu"):
            try:
                import cupy
            except ImportError:
                return (booster.inplace_predict, post,
                        "xgboost Booster.inplace_predict(host X) on a CUDA booster (no cupy), %s"
                        % what)
            return ((lambda X: cupy.asnumpy(booster.inplace_predict(cupy.asarray(X)))), post,
                    "xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, %s"
                    % what)
        return (booster.inplace_predict, post,
                "xgboost Booster.inplace_predict(X) (no DMatrix), %s" % what)
    if arm_name.startswith("lightgbm-"):
        return (model.booster_.predict, post,
                "lightgbm Booster.predict(X) (on the CPU whatever device trained it), %s" % what)
    raise RuntimeError("no inference path wired for arm %s" % arm_name)


def _categorical_infer_spec(arm_name, model, frame):
    """Inference for the categorical lane: P(class 1) from each library's
    own call on its own frame kind (`frame(X)`, built inside the clock)."""
    if arm_name in ("ours", "ours-ab"):
        return (model.predict_proba, _p1,
                "mojolearn GradientBoosting.predict_proba(X) on the float32 codes, column 1")
    if arm_name.startswith("catboost-"):
        tt = "GPU" if arm_name.endswith("-gpu") else "CPU"
        return ((lambda X: model.predict_proba(frame("catboost", X), task_type=tt)), _p1,
                "catboost predict_proba(int64 categorical frame built in the clock, task_type %s), "
                "column 1" % tt)
    if arm_name.startswith("xgboost-"):
        return ((lambda X: model.predict_proba(frame("xgboost", X))), _p1,
                "xgboost XGBClassifier.predict_proba(pandas CategoricalDtype frame built in the "
                "clock; inplace_predict takes no category frame here), column 1")
    if arm_name.startswith("lightgbm-"):
        return (model.booster_.predict, _vec,
                "lightgbm Booster.predict(X) on the float32 codes (its categorical columns are "
                "recorded in the model), probability")
    raise RuntimeError("no inference path wired for arm %s" % arm_name)


def _categorical_frame(data):
    """`frame(library, X)` for the categorical lane's inference: CatBoost's
    int64 object matrix and XGBoost's category frame, the kinds their fit and
    scorer build (tools/speed_gbdt_arm.py catboost_arms, xgboost_arms)."""
    cat_idx = list(data.cat_idx)
    levels = {j: int(max(float(np.max(data.X_train[:, j])),
                         float(np.max(data.X_test[:, j])))) + 1 for j in cat_idx}

    def frame(library, x):
        if library == "catboost":
            out = x.astype(object)
            for j in cat_idx:
                out[:, j] = x[:, j].astype(np.int64)
            return out
        import pandas as pd
        from pandas.api.types import CategoricalDtype
        df = pd.DataFrame(x)
        for j in cat_idx:
            df[j] = df[j].astype("int64").astype(CategoricalDtype(categories=list(range(levels[j]))))
        return df
    return frame


def emit_infer(head, lane, fields):
    print("%s lane=%s %s" % (head, lane, " ".join("%s=%s" % kv for kv in fields)), flush=True)


def _one_line(text, n=240):
    return " ".join(str(text).split())[:n]


def infer_metrics(lane, data, vec):
    """The FSPEED-ACC metric(s), recomputed from the timed inference output
    on the held-out rows, so a reader sees the timed call produced the scored
    predictions."""
    if lane == "iforest":
        return [("auc", spec.auc(data.y_anom, -vec))]
    if data.task == "ranking":
        return spec.ranking_metrics(data.bounds_test, data.y_test, vec)
    if data.task == "multiclass":
        return [("mlogloss", spec.mlogloss(data.y_test, vec)),
                ("accuracy", spec.accuracy(data.y_test, np.argmax(vec, axis=1)))]
    if data.task == "regression":
        return [("rmse", spec.rmse(data.y_test, vec))]
    return [("logloss", spec.logloss(data.y_test, vec)), ("auc", spec.auc(data.y_test, vec))]


def run_inference(lane, arms, models, data, n_rounds, large_rows, deadline):
    """The inference phase. `models` holds each arm's last fitted model."""
    budget = spec.per_arm_budget_s()
    _INFER_DATA[0] = data
    n_large = int(min(large_rows, data.X_train.shape[0]))
    xl = np.ascontiguousarray(data.X_train[:n_large], dtype=np.float32)
    batches = [("test", data._ours_Xtest, data.X_test), ("large", xl, xl)]
    specs = {}
    # categorical frames (lane gbdt-categorical, criteo): CatBoost and XGBoost
    # predict from the frame kind their fit took, built inside the clock
    frame = _categorical_frame(data) if data.cat_idx and data.task == "binary" else None
    for arm in arms:
        model = models.get(arm.name)
        if model is None:
            emit_infer("FSPEED-INFER-REFUSED", lane,
                       [("arm", arm.name), ("batch", "all"),
                        ("reason", "no fitted model from the fit rounds")])
            continue
        try:
            call, post, text = infer_spec(arm.name, lane, data.task, model,
                                          n_features=data.X_train.shape[1], frame=frame)
        except Exception as exc:                   # noqa: BLE001
            emit_infer("FSPEED-INFER-REFUSED", lane,
                       [("arm", arm.name), ("batch", "all"),
                        ("reason", "%s: %s" % (exc.__class__.__name__, _one_line(exc)))])
            continue
        specs[arm.name] = (call, post)
        emit_infer("FSPEED-INFER-PATH", lane, [("arm", arm.name), ("call", _one_line(text, 400))])
    spent = {name: 0.0 for name in specs}
    for batch, x_ours, x_them in batches:
        rows = int(x_ours.shape[0])
        live, last = [], {}
        for name, (call, post) in specs.items():
            x = x_ours if name in ("ours", "ours-ab") else x_them
            if time.time() > deadline:
                emit_infer("FSPEED-INFER-REFUSED", lane, [("arm", name), ("batch", batch),
                           ("reason", "process deadline reached before warm-up")])
                continue
            try:
                t0 = time.perf_counter()
                call(x)
                ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:               # noqa: BLE001
                emit_infer("FSPEED-INFER-REFUSED", lane, [("arm", name), ("batch", batch),
                           ("reason", "%s during warm-up: %s" % (exc.__class__.__name__, _one_line(exc)))])
                continue
            spent[name] += ms / 1000.0
            emit_infer("FSPEED-INFER-WARMUP", lane, [("arm", name), ("batch", batch),
                                                     ("rows", rows), ("ms", "%.3f" % ms)])
            live.append(name)
        for r in range(1, n_rounds + 1):
            for name in list(live):
                call, post = specs[name]
                x = x_ours if name in ("ours", "ours-ab") else x_them
                if time.time() > deadline or spent[name] > budget:
                    emit_infer("FSPEED-INFER-REFUSED", lane, [("arm", name), ("batch", batch),
                               ("reason", "process deadline or per-arm budget %.0fs reached at "
                                          "round %d" % (budget, r))])
                    live.remove(name)
                    continue
                try:
                    t0 = time.perf_counter()
                    raw = call(x)
                    ms = (time.perf_counter() - t0) * 1000.0
                    vec = post(raw)
                except Exception as exc:           # noqa: BLE001
                    emit_infer("FSPEED-INFER-REFUSED", lane, [("arm", name), ("batch", batch),
                               ("reason", "%s at round %d: %s" % (exc.__class__.__name__, r,
                                                                  _one_line(exc)))])
                    live.remove(name)
                    continue
                spent[name] += ms / 1000.0
                last[name] = vec
                emit_infer("FSPEED-INFER", lane, [("arm", name), ("batch", batch), ("rows", rows),
                                                  ("round", r), ("ms", "%.3f" % ms),
                                                  ("hash", spec.hash_predictions(vec))])
        if batch == "test":
            for name, vec in last.items():
                try:
                    for metric, value in infer_metrics(lane, data, vec):
                        emit_infer("FSPEED-INFER-ACC", lane, [("arm", name), ("batch", batch),
                                                              ("metric", metric),
                                                              ("value", "%.6f" % value)])
                except Exception as exc:           # noqa: BLE001
                    emit_infer("FSPEED-INFER-REFUSED", lane, [("arm", name), ("batch", batch),
                               ("reason", "%s while scoring: %s" % (exc.__class__.__name__,
                                                                    _one_line(exc)))])
        # FAST (ours-ab on the Apple board) and our CPU tier (ours-cpu)
        # against IDENTICAL (ours): the same rows through two models,
        # compared bit for bit.
        for other in [o for o in ("ours-ab", "ours-cpu") if "ours" in last and o in last]:
            a, b = last["ours"], last[other]
            same_shape = a.shape == b.shape
            emit_infer("FSPEED-INFER-AGREE", lane, [
                ("batch", batch), ("arms", "ours," + other), ("rows", rows),
                ("bits_equal", "yes" if same_shape and a.tobytes() == b.tobytes() else "no"),
                ("max_abs_diff", ("%.6g" % float(np.max(np.abs(a - b)))) if same_shape and a.size
                 else "-")])


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="forest_speed_arm",
        description="mojolearn's explicitly selected mode against native "
                    "opponents, one lane per process",
    )
    p.add_argument("--lane", required=True, choices=spec.LANE_NAMES)
    p.add_argument("--dataset", default=None,
                   help="taxi, taxireg, istella, istellareg, higgs (retired), year, covtype, covtype2, synth, "
                        "synthclf, anomaly; the lane's own default if "
                        "unset. `higgsreg` is higgs with its label as a "
                        "float target (the RMSE cell). `higgs` "
                        "is the LARGE-LOAD dataset (11M x 28) and is what "
                        "--rows climbs.")
    p.add_argument("--devices", default="auto",
                   help="which device arms of each opponent to run. `auto` "
                        "(the default) is GPU-ONLY wherever an accelerator "
                        "is visible and cpu on the MacBook. On NVIDIA we "
                        "compare against the vendor's GPU path only; on AMD "
                        "(CONTRIBUTING.md (Comparing against libraries without a GPU path), 2026-09-11) a "
                        "library with no AMD GPU path runs on the box's CPU "
                        "when `cpu` is listed, labeled in the arm name. "
                        "`opencl` selects LightGBM's USE_GPU learner. An "
                        "explicit list still wins, and the refusal lines say "
                        "when one was applied.")
    p.add_argument("--arms", default=None,
                   help="comma-separated opponent arm names to keep (a subset "
                        "of the lane's roster, e.g. catboost-cpu,xgboost-gpu); "
                        "a name asked for and not built is REFUSED by name. "
                        "`ours` always runs; the filter reads opponents only.")
    p.add_argument("--ours-ab", default=None, metavar="PARAM=VALUE",
                   help="add a second ours arm, `ours-ab`, equal to `ours` "
                        "except one estimator keyword (a Python literal, "
                        "e.g. use_pointwise_searcher=True or "
                        "numeric_mode='fast'), timed round by round beside "
                        "`ours` in this process; gbdt, rf, et and iforest lanes, and "
                        "it runs under --ours-only too")
    p.add_argument("--opponents-first", action="store_true",
                   help="import and construct the opponents BEFORE our "
                        "binding in this process (the import order that "
                        "exposed the _buffer.py ctypes clash with cuML); the "
                        "round rotation still starts with ours")
    p.add_argument("--rows", type=int, default=None,
                   help="cap the training rows. On `higgs` this IS the load "
                        "ladder: rungs are nested prefixes of the same "
                        "data scored against the same fixed 500,000-row "
                        "tail, so 1000000 vs 5000000 is a comparison of "
                        "LOAD and not of two problems. Every line carries "
                        "the row count in shape=.")
    p.add_argument("--ours-only", action="store_true",
                   help="skip the opponents; use when two CUDA runtimes in "
                        "one process will not coexist")
    p.add_argument("--list-arms", action="store_true",
                   help="print the roster for the lane and exit")
    p.add_argument("--infer", action="store_true",
                   help="after the fit rounds, time batch prediction too: every arm "
                        "predicts with its own last fitted model on the held-out rows "
                        "and on a large batch (FSPEED-INFER lines). Off by default; "
                        "without it the output is unchanged")
    p.add_argument("--ours-cpu", action="store_true",
                   help="add `ours-cpu`: the same ours estimator in a worker process under "
                        "MOJOLEARN_VENDOR=cpu (the wheel's CPU tier, IDENTICAL), in the "
                        "round-robin beside the other arms (bench/speed/forest_board_arms.py)")
    p.add_argument("--mem", action="store_true",
                   help="print FSPEED-MEM: each arm's peak host memory per round (and the "
                        "process's GPU figure), read outside the clock (forest_board_arms.py)")
    p.add_argument("--infer-large-rows", type=int, default=INFER_LARGE_ROWS,
                   help="rows of the `large` inference batch (the first N training "
                        "rows; default 1,000,000, capped at the training rows)")
    return p


def seed_draws(lane, cfg, data):
    """What the seed drives in this lane's fit on the arms that draw random
    numbers, or '' when no arm draws one (then the seed note says so)."""
    if lane in ("rf", "et"):
        return ("row bootstrap (rf), per-node feature draws, ET thresholds; LightGBM "
                "derives its bagging, feature and extra-trees seeds from seed 7")
    if lane == "iforest":
        return "row subsamples, split features and thresholds"
    parts = []
    if cfg.get("random_strength"):
        parts.append("split-score noise (random_strength %g) on ours and CatBoost"
                     % cfg["random_strength"])
    if cfg.get("loss") == "YetiRank":
        parts.append("YetiRank's sampled permutations on ours and CatBoost")
    if getattr(data, "cat_idx", None):
        parts.append("CatBoost's and ours CTR permutations")
    return "; ".join(parts)


def main(argv=None):
    started = time.time()
    args = build_parser().parse_args(argv)
    lane = args.lane
    size = spec.size_tag()
    dataset = args.dataset or spec.LANE_DEFAULT_DATASET[lane]
    devices, devices_auto = spec.resolve_devices(args.devices, lane)

    if lane.startswith("gbdt-") and dataset == "covtype" and lane != "gbdt-multiclass":
        # DEVIATION 1838. Refuse the 7-class task by name and move to the
        # derived binary one, which every arm can run, rather than quietly
        # swapping the problem under the reader.
        spec.emit_refused(
            lane, "ours",
            "covtype is 7-class and mojolearn.GradientBoosting has no "
            "MultiClass on the Python surface; running covtype2 (y == 2 vs "
            "rest) on EVERY arm instead")
        dataset = "covtype2"

    # DEVIATION 2634's REACH, turned on for every run of this harness
    # (2026-09-12, lane harness-honesty). `gbdt/train.mojo` gates the CTR
    # target prep on `len(dependent_configs) > 0 and ctr_prep_wanted` and
    # nothing logged either term, so the 2.1% band its criteo A/B measured
    # could not be attributed to the branch under test -- the reached-but-inert
    # trap, recorded in OPPONENT_REFERENCE.md. The fit now prints the CTR
    # config split and a prep=ran/skipped marker when this is set, and the
    # LIBRARY stays silent unless it is: a shipped fit does not print.
    # The harness always asks, so no leg has to remember to.
    os.environ.setdefault("MOJOLEARN_CTR_TRACE", "1")
    print("FSPEED-CTR-TRACE lane=%s MOJOLEARN_CTR_TRACE=%s (the fit prints its "
          "CTR config split and prep marker; DEVIATION 2634 is observable)"
          % (lane, os.environ["MOJOLEARN_CTR_TRACE"]), flush=True)

    data = spec.load_with_fallback(dataset, size, args.rows)
    cfg = spec.lane_config(lane, size)
    task = spec.task_of(lane)
    if task:
        # A task lane runs its own task only: a dataset of another task would
        # time a different problem under the lane's name.
        if data.task not in spec.task_names(task):
            spec.emit_refused(lane, "all", "dataset %s is a %s task; lane %s races %s "
                              "(its datasets: %s)" % (data.name, data.task, lane,
                                                     "/".join(spec.task_names(task)),
                                                     ",".join(task["datasets"])))
            return 1
        if lane == "gbdt-categorical" and not data.cat_idx:
            spec.emit_refused(lane, "all", "dataset %s declares no categorical column; lane %s "
                              "races the categorical path (its datasets: %s)"
                              % (data.name, lane, ",".join(task["datasets"])))
            return 1
        if data.task == "multiclass":
            cfg["n_classes"] = data.n_classes
        # every mismatch the lane could not remove, one line each, on the card
        for i, why in enumerate(task["mismatches"]):
            spec.emit_note(lane, ["*"], "mismatch", float(i + 1), why)
        spec.emit_note(lane, ["*"], "objectives", float(len(task["objectives"])),
                       "; ".join("%s %s" % kv for kv in sorted(task["objectives"].items())))
    spec.prepare_cuml_labels(data)
    spec.prepare_anomaly_labels(lane, data)
    prepare_our_inputs(data)

    if args.list_arms:
        print("ours  %s" % OUR_ENTRY_POINTS[lane])
        for names, _ in spec.opponent_builders(lane, cfg, data, devices):
            for name in names:
                print(name)
        return 0

    # THE DEVICE POLICY IS ON THE CARD, not only in this file. A reader who
    # sees three arms where the Apple table had six has to be able to find
    # out why without reading the harness.
    if spec.accel_vendor() == "amd":
        policy = ("AMD box: an opponent with an AMD GPU path runs on the GPU, "
                  "one without runs on this box's CPU on all cores "
                  "(CONTRIBUTING.md (Comparing against libraries without a GPU path)); the arm name says which")
    else:
        policy = ("on an NVIDIA box the vendors' CPU arms do not run, so a "
                  "lane whose only opponent is a CPU library has NO legal "
                  "opponent here and says so")
    spec.emit_note(lane, ["ours"], "devices", float(len(devices)),
                   "devices=%s (%s); %s"
                   % (",".join(devices), "auto" if devices_auto else "explicit",
                      policy))
    opponents = []
    # --arms: only the wanted builders run (an unwanted one's refusal would
    # otherwise reach the board as a cell it never planned)
    wanted = [n.strip() for n in (args.arms or "").split(",") if n.strip()] or None
    if not args.ours_only and args.opponents_first:
        print("FSPEED-IMPORT-ORDER lane=%s first=opponents" % lane, flush=True)
        opponents = spec.build_opponents(lane, cfg, data, devices, wanted)
    arms = build_ours(lane, cfg, data)
    if args.ours_ab:
        import ast
        key, _, raw = args.ours_ab.partition("=")
        value = ast.literal_eval(raw)
        print("FSPEED-AB lane=%s arm=ours-ab %s=%r (arm ours keeps the "
              "estimator default)" % (lane, key.strip(), value), flush=True)
        arms.extend(build_ours(lane, cfg, data, name="ours-ab",
                               extra={key.strip(): value}))
    proxies = []
    if args.ours_cpu:
        import forest_board_arms
        proxies = forest_board_arms.build_cpu_arm(lane, dataset, args.rows, args.infer_large_rows,
                                                  spec.emit_refused)
        arms.extend(proxies)
    if not args.ours_only:
        if not args.opponents_first:
            opponents = spec.build_opponents(lane, cfg, data, devices, wanted)
        if args.arms:
            have = {a.name for a in opponents}
            for name in wanted:
                if name not in have:
                    spec.emit_refused(lane, name,
                                      "asked for by --arms and not built here; "
                                      "built: %s" % ",".join(sorted(have)))
            opponents = [a for a in opponents if a.name in wanted]
        # The opponents are appended AFTER ours so that the rotation starts
        # with our arm; the runner alternates from there and no arm ever runs
        # two rounds in a row.
        arms.extend(opponents)

    if not arms:
        spec.emit_refused(lane, "all", "nothing could be constructed on this "
                                       "box; see the refusals above")
        return 1
    # THE PARAMETER CHECK (tools/bench_board_params.py; Andrew, 2026-09-29:
    # "same seed same tuning params"), before the first timed round. Each
    # arm's model is constructed, not fitted, and its parameters are read
    # back from the object; a seed or a shared parameter that differs
    # refuses the race by name. The CPU proxy arms are ours on another
    # column and carry the same parameters, so they are not compared.
    # `ours-ab` differs from ours by the one key --ours-ab names, on purpose.
    import bench_board_params as BP
    proxy_names = {getattr(a, "name", None) for a in proxies}
    records = {}
    for arm in arms:
        if arm.name in proxy_names:
            continue
        try:
            records[arm.name] = arm.make()
        except Exception as exc:  # noqa: BLE001
            records[arm.name] = {"__record__": True, "library": "?",
                                 "source": "construction failed (%s)" % exc, "params": {}}
    extra = ()
    if args.ours_ab:
        extra = ((args.ours_ab.partition("=")[0].strip(), "ours-ab",
                  "--ours-ab changes this one key on purpose (the A/B arm)"),)
    try:
        BP.enforce(lane, records, family="trees", extra_exceptions=extra)
    except BP.ParamsRefused as exc:
        spec.emit_refused(lane, "all", "PARAMETER CHECK REFUSED: %s" % exc)
        return 1
    # SAME SEED, SAME TUNING PARAMETERS: rule 1's note, then the board's check
    # on one constructed estimator per arm, before any warm-up or timed round
    spec.emit_seed_note(lane, [a.name for a in arms], seed_draws(lane, cfg, data))
    if not spec.enforce_board_params(lane, arms, skip={p.name for p in proxies}):
        return 2
    # `cfg` reaches the runner so the FIT-EQUIVALENCE check can hold each
    # arm's FITTED tree count against the count this lane asked for. Without
    # it the shapes are still reported and that one check is skipped; an
    # expectation is never invented.
    models = {}
    if args.infer:
        # Keep each arm's LAST fitted model for the inference phase. The
        # wrapper records the model after `fit` returns; nothing inside the
        # fit clock changes.
        for arm in arms:
            def _fit(model, d, _orig=arm.fit, _name=arm.name):
                out = _orig(model, d)
                models[_name] = model
                return out
            arm.fit = _fit
    fit_context = None
    if args.mem:
        import forest_board_arms
        fit_context = forest_board_arms.TreeMem(lane, proxies).context
    live = spec.run(lane, arms, data, spec.rounds(), size, cfg=cfg, fit_context=fit_context)
    if args.infer:
        names = {a.name for a in live}
        run_inference(lane, [a for a in arms if a.name in names],
                      models, data, spec.rounds(), args.infer_large_rows,
                      started + spec.process_deadline_s())
    for proxy in proxies:
        proxy.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
