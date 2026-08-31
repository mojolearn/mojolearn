#!/usr/bin/env python3
"""OUR side of the gradient-boosting and forest speed run, and the one
process that interleaves it with the NVIDIA-native opponents.

    pixi run -e speedbench python bench/speed/forest_speed_arm.py --lane rf
    MOJOLEARN_SPEED_SIZE=smoke \
      pixi run -e speedbench python bench/speed/forest_speed_arm.py \
        --lane gbdt-symmetric

THE QUESTION THIS FILE ANSWERS
-------------------------------
How fast is mojolearn's FAST path -- the DEFAULT build, NOT
`-D MOJOLEARN_NUMERIC_IDENTICAL=1` -- against what an NVIDIA user would
actually run, on an NVIDIA GPU. That is a pure speed question about the
explicitly non-deterministic, non-bitwise-identical arm. It is not the
identity question, no line here gates a hash, and `MOJOLEARN_NUMERIC_MODE`
is reported on every header only so that a run of the wrong arm is
impossible to mislabel.

**We expect to lose the GPU columns, possibly by a lot. Recording how much
is the entire point.**

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
lane wiring. The direction is deliberate: on a box whose first-ever CUDA
build of `ensemble/` has just failed, the opponents still run from that file
alone and the lease is not wasted.

ONE LANE PER PROCESS
---------------------
`--lane` takes exactly one lane. No line of `ensemble/` or `extratrees/` has
ever been compiled for CUDA; `TARGET_COLUMN` has been `COLUMN_APPLE` for
every build ever made in this repository, the NVIDIA rows of the kernel
matrix are arithmetic rather than measurement, and the float atomic flush
branch is unreachable on Apple and is the path NVIDIA takes. **Treat the
first run as a BUILD, not a benchmark. If it produces numbers on the first
attempt, be suspicious rather than pleased.** A lane that takes the process
down with it must not take the other five.

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

import numpy as np

# `tools/` is not a package and never has been, so the spec is imported by
# path rather than by name. The repository root is two levels up from this
# file (bench/speed/ -> bench/ -> root), and `python/` goes on the path too
# so an in-repo, not-yet-installed `mojolearn` is importable exactly the way
# `bench/external/run_gbm_bench.sh` arranges it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec           # noqa: E402


#: What each lane calls in `python/mojolearn/`, for the report and for the
#: `--list-arms` output. `training/` is absent on purpose: it is the neural
#: training-step lane and it has no boosting or forest estimator, so there is
#: no entry point here to time.
#:
#: CORRECTED 2026-08-31. This comment used to add that, as of 2026-08-25,
#: `training/TRAINING_LOOP_PLAN.md` stated in its own first paragraph that no
#: `mojo` process had ever read any of its three files. Commit `5ce6eb17`
#: falsified that plan's banner (the lane compiles and its step gate ran green
#: on one device) and the banner is now corrected in place. The reason
#: `training/` is absent from THIS file never depended on it: no forest, no
#: boosting, nothing to time.
OUR_ENTRY_POINTS = {
    "gbdt-symmetric": "mojolearn.GradientBoosting(grow_policy='SymmetricTree')"
                      " -> gbdt/ via _mojolearn_gbdt",
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

def our_gbdt_arm(lane, cfg, data):
    """`mojolearn.GradientBoosting`, the CatBoost GPU tree learner port.

    Every knob is the lane's, so the only difference between this arm and
    the CatBoost arm beside it is the device and the implementation:

        n_estimators <- iterations       max_depth   <- depth
        learning_rate                    l2_leaf_reg <- reg lambda
        border_count (BORDERS, not bins) grow_policy
        bootstrap_type='No'              random_state <- random_seed

    `boost_from_average` is passed by NEITHER arm, deliberately. Both
    libraries resolve the same data-dependent default -- auto-true for RMSE,
    false for Logloss, since Logloss is not on CatBoost's
    `AdjustBoostFromAverageDefaultValue` list -- and `check-bfa-oracle`
    proves our bias bit-equal to their `get_scale_and_bias`. Passing it would
    be overriding a default that already agrees.

    MULTICLASS IS REFUSED BY NAME (DEVIATION 1838). The Mojo layer implements
    `MultiClass`; this Python wrapper is one-dimensional, so a 7-class
    dataset cannot be run by this arm at all. It refuses rather than
    silently fitting a different problem than the CatBoost arm beside it."""
    import mojolearn

    if data.task == "multiclass":
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
        loss="RMSE" if data.task == "regression" else "Logloss",
    )
    if cfg["grow_policy"] == "Lossguide":
        params["max_leaves"] = cfg["max_leaves"]

    def make():
        return mojolearn.GradientBoosting(**params)

    return spec.Arm("ours", make, _our_fit, _our_score,
                    sync=_our_sync, library="mojolearn")


def our_rf_arm(lane, cfg, data):
    """`mojolearn.RandomForest*`, the cuML RandomForest port (`ensemble/`).

    `device='gpu'` is explicit and never 'auto', so a run that cannot reach
    the accelerator fails loudly instead of quietly reporting a host number
    under a GPU label. There is no CPU arm here because the library has no
    CPU path at all -- `kernel_matrix.mojo` says "There is no CPU column."

    `n_streams` is left at the class default, which is cuML's, for the same
    reason the cuML arm leaves it: a value above 1 makes the fit
    non-reproducible, and this whole slice measures the non-deterministic
    FAST path. Pinning it would benchmark a configuration nobody runs."""
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
        random_state=cfg["seed"],
        device="gpu",
    )

    def make():
        if data.task == "regression":
            return mojolearn.RandomForestRegressor(
                criterion="squared_error", **common)
        return mojolearn.RandomForestClassifier(criterion="gini", **common)

    return spec.Arm("ours", make, _our_fit, _our_score,
                    sync=_our_sync, library="mojolearn")


def our_et_arm(lane, cfg, data):
    """`mojolearn.ExtraTrees*`, the cuML-design ExtraTrees port
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

    def make():
        if data.task == "regression":
            return mojolearn.ExtraTreesRegressor(
                criterion="squared_error", **common)
        return mojolearn.ExtraTreesClassifier(criterion="gini", **common)

    return spec.Arm("ours", make, _our_fit, _our_score,
                    sync=_our_sync, library="mojolearn")


def our_iforest_arm(lane, cfg, data):
    """`mojolearn.IsolationForest`, the cuML IsolationForest port.

    WHAT `fit` ACTUALLY DOES HERE, AND IT CHANGES WHAT THIS ROW MEANS
    (DEVIATION 874, restated as DEVIATION 1836 for this harness). The forest
    is NOT KEPT. `fit` builds the forest over X and scores ONE row, which is
    the cheapest call the entry point accepts, and then every later scoring
    call REBUILDS IT. So:

      * the timed number IS a real full forest build over the training rows,
        plus a one-row scoring pass, and is comparable to sklearn's `fit`;
      * the ACCURACY column comes from a DIFFERENT forest than the one that
        was timed, because `score_samples` built its own;
      * a `hash=` that changes between rounds is expected here twice over --
        once for the FAST path's non-determinism and once because the forest
        was rebuilt.

    None of that is corrected here. It is a property of the surface under
    test and correcting it in the harness would measure a library that does
    not exist."""
    import mojolearn

    def make():
        return mojolearn.IsolationForest(
            n_estimators=cfg["n_estimators"],
            max_samples=cfg["max_samples"],
            max_features=cfg["max_features"],
            bootstrap=cfg["bootstrap"],
            contamination="auto",
            random_state=cfg["seed"],
        )

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
    "gbdt-symmetric": our_gbdt_arm,
    "gbdt-depthwise": our_gbdt_arm,
    "gbdt-lossguide": our_gbdt_arm,
    "rf": our_rf_arm,
    "et": our_et_arm,
    "iforest": our_iforest_arm,
}


def build_ours(lane, cfg, data):
    """Our arm, or a refusal. An import error here is the EXPECTED failure on
    the first CUDA build, and it must print a refusal rather than a
    traceback: the opponents in the same process still have a lane to run."""
    try:
        return [OUR_BUILDERS[lane](lane, cfg, data)]
    except Exception as exc:                       # noqa: BLE001
        spec.emit_refused(lane, "ours", "%s: %s"
                          % (exc.__class__.__name__,
                             " ".join(str(exc).split())))
        return []


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="forest_speed_arm",
        description="mojolearn's FAST path against the NVIDIA-native "
                    "opponents, one lane per process",
    )
    p.add_argument("--lane", required=True, choices=spec.LANE_NAMES)
    p.add_argument("--dataset", default=None,
                   help="higgs, year, covtype, covtype2, synth, synthclf, "
                        "anomaly; the lane's own default if unset. `higgs` "
                        "is the LARGE-LOAD dataset (11M x 28) and is what "
                        "--rows climbs.")
    p.add_argument("--devices", default="auto",
                   help="which device arms of each opponent to run. `auto` "
                        "(the default) is GPU-ONLY wherever an accelerator "
                        "is visible and cpu on the MacBook: on NVIDIA and "
                        "AMD we compare against the vendor's GPU path only, "
                        "because a GPU-versus-CPU ratio is not the claim "
                        "this project makes. An explicit list still wins, "
                        "and the refusal lines say when one was applied.")
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
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    lane = args.lane
    size = spec.size_tag()
    dataset = args.dataset or spec.LANE_DEFAULT_DATASET[lane]
    devices, devices_auto = spec.resolve_devices(args.devices, lane)

    if lane.startswith("gbdt-") and dataset == "covtype":
        # DEVIATION 1838. Refuse the 7-class task by name and move to the
        # derived binary one, which every arm can run, rather than quietly
        # swapping the problem under the reader.
        spec.emit_refused(
            lane, "ours",
            "covtype is 7-class and mojolearn.GradientBoosting has no "
            "MultiClass on the Python surface; running covtype2 (y == 2 vs "
            "rest) on EVERY arm instead")
        dataset = "covtype2"

    data = spec.load_with_fallback(dataset, size, args.rows)
    cfg = spec.lane_config(lane, size)
    spec.prepare_cuml_labels(data)
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
    spec.emit_note(lane, ["ours"], "devices", float(len(devices)),
                   "devices=%s (%s); on a GPU vendor's box the vendors' CPU "
                   "arms do not run, so a lane whose only opponent is a CPU "
                   "library has NO legal opponent here and says so"
                   % (",".join(devices), "auto" if devices_auto else "explicit"))
    arms = build_ours(lane, cfg, data)
    if not args.ours_only:
        # The opponents are appended AFTER ours so that the rotation starts
        # with our arm; the runner alternates from there and no arm ever runs
        # two rounds in a row.
        arms.extend(spec.build_opponents(lane, cfg, data, devices))

    if not arms:
        spec.emit_refused(lane, "all", "nothing could be constructed on this "
                                       "box; see the refusals above")
        return 1
    spec.run(lane, arms, data, spec.rounds(), size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
