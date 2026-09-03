#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The NVIDIA-native opponents for the gradient-boosting and forest slice,
plus the dataset loading, the hyper-parameter tables and the output contract
that `bench/speed/forest_speed_arm.py` shares with them.

    pixi run -e gbmbench python tools/speed_gbdt_arm.py --lane rf
    pixi run -e gbmbench python tools/speed_gbdt_arm.py --download year

WHY THIS FILE IS THE BASE MODULE AND NOT THE LEAF
-------------------------------------------------
`bench/speed/forest_speed_arm.py` imports this one, and not the other way
round, for one reason: NOTHING HERE IMPORTS `mojolearn`. On a rented box
whose first-ever CUDA build of `ensemble/` has just failed, the opponents
still run and still print a table, and the run is not wasted. Put the shared
spec in the file that has our dependency and a broken build takes the
opponents down with it.

It also means the hyper-parameters live in ONE place (`lane_config`) that
both sides read. Two files that each spell out "learning_rate 0.1" is exactly how
a benchmark drifts into comparing two different problems while both sides
still print numbers.

WHAT CHANGES ON NVIDIA, AND IT IS THE WHOLE FRAMING
---------------------------------------------------
On Apple, CatBoost's `task_type="GPU"` raises and LightGBM ships no GPU
learner, so the M4 tables in this repository compare our GPU against their
CPU because that is their strongest LEGAL arm there. On NVIDIA that is
false. CatBoost ships a CUDA learner, XGBoost ships `device="cuda"`, cuML
is NVIDIA's own forest library, and LightGBM can be built with
`USE_CUDA=ON`. Those are what an NVIDIA user would actually run, so those
are the opponents. Every lane therefore times BOTH arms of each opponent:
the CPU arm for continuity with the Apple tables, and the GPU arm because
it is the honest one.

**We expect to lose the GPU columns, possibly by a lot. Recording how much
is the entire point.** No arm is dropped for winning and no dataset is
chosen for flattering us.

WHICH OPPONENT GOES IN WHICH LANE, AND THE STANDING ORDER BEHIND IT
--------------------------------------------------------------------
Andrew's standing order (2026-08-22): **the symmetric-tree comparison is
CatBoost ONLY.** LightGBM has no symmetric mode -- leaf-wise is its only
growth algorithm -- so a LightGBM arm beside the symmetric learner compares
two different algorithms and is excluded from it.

DEVIATION 1831 extends that order to XGBoost by the same argument, because
the argument is about the algorithm and not about the vendor: XGBoost's
`grow_policy` is `depthwise` or `lossguide` and it has no symmetric mode
either. So:

    lane gbdt-symmetric   ours grow_policy='SymmetricTree'
                          opponents: catboost-cpu, catboost-gpu ONLY
    lane gbdt-depthwise   ours grow_policy='Depthwise' (the level-wise
                          binary tree)
                          opponents: xgboost-cpu, xgboost-gpu
                          (grow_policy='depthwise'), catboost-cpu/gpu
                          under the SAME policy
    lane gbdt-lossguide   ours grow_policy='Lossguide' (leaf-wise, one leaf
                          per step)
                          opponents: lightgbm-cpu, lightgbm-cuda -- this IS
                          LightGBM's own algorithm, which is why LightGBM
                          belongs here and not beside the symmetric lane --
                          plus xgboost-*/catboost-* under lossguide
    lane rf               opponents: cuml-rf-gpu, sklearn-rf-cpu,
                          lightgbm-cpu/cuda in boosting_type='rf'
    lane et               opponents: sklearn-et-cpu, lightgbm-cpu/cuda in
                          rf + extra_trees; cuML has no ExtraTrees and
                          REFUSES by name
    lane iforest          opponents: sklearn-iforest-cpu, and cuml-iforest-gpu
                          IF cuML ships one (DEVIATION 1837)

THE TIMED REGION (DEVIATION 1830)
----------------------------------
What is timed is `fit(X, y)` on raw host numpy, on every arm, and nothing
else. Loading, splitting, dtype conversion and the train/test split all
happen before the timer starts. Prediction and scoring happen after it
stops.

This DIFFERS from `bench/interleaved/`, which hands CatBoost a pre-quantized
`Pool` because the Mojo side there consumes a prebuilt compressed index.
Here every arm quantizes inside its own `fit` -- ours computes borders in
`fit`, CatBoost bins in `fit`, XGBoost builds its `QuantileDMatrix` in
`fit`, LightGBM builds its `Dataset` in `fit`, cuML bins in `fit`. Timing
`fit` therefore includes the same phase of work on every arm, which is the
property that makes the ratio mean something. It is NOT the same number as
the interleaved harness's and must never be quoted beside one.

A GPU fit is only finished when the device says so. Every arm carries a
`sync` that runs INSIDE the timer; for the libraries whose `fit` blocks it
is a no-op and says so, and for cuML it is a device synchronize.

THE OUTPUT CONTRACT
--------------------
Header, once per process:

    FSPEED-HEADER family=forest lane=<lane> arm=<arm> mode=<FAST|IDENTICAL> \
        device=<string> rounds=<n> size=<shipped|smoke>

One line per timed round, one warm-up line per arm never in the table, one
accuracy line per arm and metric, and a refusal wherever an opponent could
not be installed or could not run:

    FSPEED lane=<l> arm=<a> shape=<tag> round=<i> ms=<float> hash=<16 hex|->
    FSPEED-WARMUP lane=<l> arm=<a> shape=<tag> ms=<float>
    FSPEED-ACC lane=<l> arm=<a> metric=<rmse|logloss|accuracy|auc> value=<f>
    FSPEED-REFUSED lane=<l> arm=<a> reason=<one line>

One line type is NOT in the contract the orchestrator handed down, and it is
additive rather than a change to the four above (DEVIATION 1839):

    FSPEED-NOTE lane=<l> arms=<a,b> metric=<m> delta=<f> reason=<one line>

It fires when two arms OF THE SAME LIBRARY differ in accuracy by more than
`_ACC_TOL`. Two arms of one library that disagree about the answer were not
given identical configurations, whatever the config dict says, and a timing
table that does not say so is reporting the speed of two different problems.
A parser keyed on the four contract prefixes ignores it.

BUDGET, BECAUSE THE BOX IS RENTED AND THE HOUR IS SHORT
--------------------------------------------------------
`MOJOLEARN_SPEED_BUDGET_S` (default 300) is a per-arm wall budget and
`MOJOLEARN_SPEED_DEADLINE_S` (default 2400) is a whole-process one. An arm
that runs past its budget is dropped from the rotation with a REFUSED line
carrying the reason, so a slow CPU arm cannot eat the lease that the GPU
arms are the point of. Nothing here waits unbounded on anything.
"""

import argparse
import hashlib
import os
import platform
import subprocess
import shutil
import sys
import time

import numpy as np

FAMILY = "forest"

#: Accuracy tolerance between two arms of the SAME library, as a relative
#: difference. Wider than float noise and much narrower than a config
#: mistake; a CPU and a CUDA learner of one library legitimately differ in
#: the last few digits because they sum in different orders.
_ACC_TOL = 0.02

#: The lanes this file knows. `training/` is deliberately absent: it is the
#: neural-network training-step lane (`training/TRAINING_LOOP_PLAN.md`) and it
#: has no gradient-boosting or forest surface, so there is no entry point to
#: time.
#:
#: CORRECTED 2026-08-31. This comment used to add "and as of 2026-08-25
#: nothing in it has ever been compiled". Commit `5ce6eb17` falsified that;
#: the lane compiles and its step gate ran green on one device. The absence
#: from this tuple was never for that reason.
LANE_NAMES = (
    "gbdt-symmetric",
    "gbdt-depthwise",
    "gbdt-lossguide",
    "rf",
    "et",
    "iforest",
)


# --------------------------------------------------------------------------
# The output contract.
# --------------------------------------------------------------------------

def size_tag():
    """`shipped` (the default, the real dataset) or `smoke` (a plumbing
    check). Every emitted line carries it so the two can never be confused
    in a results file."""
    tag = os.environ.get("MOJOLEARN_SPEED_SIZE", "shipped").strip().lower()
    if tag not in ("shipped", "smoke"):
        raise SystemExit(
            "MOJOLEARN_SPEED_SIZE must be 'shipped' or 'smoke', got "
            + repr(tag)
        )
    return tag


def rounds():
    """Timed rounds after the one untimed warm-up. Three by default for this
    slice rather than the larger counts the kernel lanes use, because a fit
    is seconds to minutes and the lease is an hour."""
    return max(1, int(os.environ.get("MOJOLEARN_SPEED_ROUNDS", "3")))


def per_arm_budget_s():
    return float(os.environ.get("MOJOLEARN_SPEED_BUDGET_S", "300"))


def process_deadline_s():
    return float(os.environ.get("MOJOLEARN_SPEED_DEADLINE_S", "2400"))


def numeric_mode_label():
    """FAST or IDENTICAL. This slice's whole question is the FAST path -- the
    default build, NOT `-D MOJOLEARN_NUMERIC_IDENTICAL=1` -- so FAST is the
    expected value here and a run that reports IDENTICAL is answering a
    different question. The label is read from the environment rather than
    assumed, because a mislabelled arm is worse than a missing one."""
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
    return "IDENTICAL" if mode == "identical" else "FAST"


def device_string():
    """What this box is, in one token-safe string. `nvidia-smi` first,
    because on the box this file exists for the GPU is the answer; the host
    platform is the fallback so a line is never emitted without a device."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        name = out.stdout.strip().splitlines()
        if out.returncode == 0 and name:
            return name[0].strip().replace(" ", "_")
    except (OSError, subprocess.SubprocessError):
        pass
    return (platform.system() + "_" + platform.machine()).replace(" ", "_")


def emit_header(lane, arm, dev, n_rounds, size):
    print(
        "FSPEED-HEADER family=%s lane=%s arm=%s mode=%s device=%s "
        "rounds=%d size=%s"
        % (FAMILY, lane, arm, numeric_mode_label(), dev, n_rounds, size)
    )
    sys.stdout.flush()


def emit_warmup(lane, arm, shape, ms):
    print("FSPEED-WARMUP lane=%s arm=%s shape=%s ms=%.3f"
          % (lane, arm, shape, ms))
    sys.stdout.flush()


def emit_round(lane, arm, shape, index, ms, digest):
    print("FSPEED lane=%s arm=%s shape=%s round=%d ms=%.3f hash=%s"
          % (lane, arm, shape, index, ms, digest or "-"))
    sys.stdout.flush()


def emit_acc(lane, arm, metric, value):
    print("FSPEED-ACC lane=%s arm=%s metric=%s value=%.6f"
          % (lane, arm, metric, value))
    sys.stdout.flush()


def emit_refused(lane, arm, reason):
    one_line = " ".join(str(reason).split())[:240]
    print("FSPEED-REFUSED lane=%s arm=%s reason=%s" % (lane, arm, one_line))
    sys.stdout.flush()


def emit_note(lane, arms, metric, delta, reason):
    one_line = " ".join(str(reason).split())[:240]
    print("FSPEED-NOTE lane=%s arms=%s metric=%s delta=%.6f reason=%s"
          % (lane, ",".join(arms), metric, delta, one_line))
    sys.stdout.flush()


def hash_predictions(vec):
    """sha256 over dtype + shape + the exact bytes, truncated to 16 hex
    digits, the same recipe `bench/external/patch_gbm_bench.py` puts in the
    gbm-bench results.

    IT IS NOT A DETERMINISM CLAIM HERE. This slice measures the FAST path,
    which is the explicitly non-deterministic arm: cuML's `n_streams`, our
    own atomics and CatBoost's GPU reductions all reorder run to run. Two
    equal hashes across rounds are informative, two unequal ones are
    expected, and neither is a defect. Across arms the hashes are not even
    comparable, because the dtypes differ."""
    if vec is None:
        return None
    arr = np.ascontiguousarray(vec)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# Metrics. Computed OUTSIDE every timer, on a held-out split.
# --------------------------------------------------------------------------

def rmse(y, p):
    d = np.asarray(y, dtype=np.float64) - np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def logloss(y, p1):
    """Binary log loss with the same clip sklearn uses since it dropped its
    `eps` argument in 1.5 (the sharp edge `patch_gbm_bench.py` already had to
    work around upstream)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p1, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(y, labels):
    return float(np.mean(np.asarray(y).ravel() == np.asarray(labels).ravel()))


def auc(y, score):
    """Rank-based AUC in numpy, with ties averaged. Written out rather than
    imported so the metric does not become a reason the run needs sklearn on
    a box where only the CUDA libraries installed."""
    y = np.asarray(y, dtype=np.float64).ravel()
    s = np.asarray(score, dtype=np.float64).ravel()
    pos = float(np.sum(y > 0.5))
    neg = float(y.size) - pos
    if pos == 0.0 or neg == 0.0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Average the ranks inside each tie group, which is what makes this
    # agree with sklearn's roc_auc_score rather than merely resemble it.
    sorted_s = s[order]
    i = 0
    while i < sorted_s.size:
        j = i
        while j + 1 < sorted_s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return float((np.sum(ranks[y > 0.5]) - pos * (pos + 1.0) / 2.0)
                 / (pos * neg))


# --------------------------------------------------------------------------
# Datasets.
# --------------------------------------------------------------------------

class Data(object):
    """Train/test split, already in host memory, already the right dtype.

    `task` is 'regression', 'binary', 'multiclass' or 'anomaly'. `tag` is
    what every emitted line carries as `shape=`, and it names the dataset AND
    the training shape so a smoke line can never be mistaken for a shipped
    one even if somebody loses the header."""

    def __init__(self, name, x_train, x_test, y_train, y_test, task,
                 n_classes=0, y_anom=None):
        self.name = name
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.task = task
        self.n_classes = n_classes
        # Only for the anomaly lane: the planted anomaly labels of X_test.
        self.y_anom = y_anom
        self.tag = "%s-%dx%d" % (name, x_train.shape[0], x_train.shape[1])


def data_root():
    """Where the multi-gigabyte downloads live. Same environment variable
    `bench/external/run_gbm_bench.sh` uses, so a box that already has the
    gbm-bench store does not fetch anything twice."""
    return os.environ.get(
        "GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"),
                                       "datasets", "gbm-bench"))


def _smoke_rows(x, y, limit):
    if limit is None or x.shape[0] <= limit:
        return x, y
    return x[:limit], y[:limit]


def _synth_regression(rows, feats, seed=7):
    """The same generator `tools/interleaved_prep.py` uses, at the same
    default shape, so a synthetic fallback number here sits beside the
    interleaved harness's synthetic numbers rather than beside nothing."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, feats)).astype(np.float32)
    y = (3.0 * x[:, 0] - 2.0 * x[:, 3] + 1.5 * x[:, 7] * x[:, 0]
         + 0.1 * rng.normal(size=rows)).astype(np.float32)
    return x, y


def _synth_binary(rows, feats, seed=7):
    x, raw = _synth_regression(rows, feats, seed)
    y = (raw > np.median(raw)).astype(np.float32)
    return x, y


def _split_tail(x, y, task, name, n_classes=0, frac=0.1):
    """A deterministic TAIL split, no shuffle, no sklearn. The test rows are
    only ever used for the accuracy column, so what matters is that every arm
    of every lane scores the same rows -- not that the split is clever."""
    n_test = max(1, int(x.shape[0] * frac))
    n_train = x.shape[0] - n_test
    return Data(name, x[:n_train], x[n_train:], y[:n_train], y[n_train:],
                task, n_classes)


def load_year(size, rows_cap=None):
    """YearPredictionMSD, the regression dataset gbm-bench uses and the one
    the M4 GBDT tables were taken on.

    THE DOWNLOAD IS 211 MB and it is a SEPARATE, EXPLICITLY NAMED STEP
    (`--download year`), never something a timed run does on its own, because
    the orchestrator has to budget a one-hour GPU lease around it. The
    train/test split is gbm-bench's own: the first 463,715 rows train, the
    remaining 51,630 test, no shuffle, which that dataset's own
    documentation requires (an artist's tracks must not straddle the split).

    Read straight from the zip with pandas rather than through gbm-bench's
    `year.pkl`. Their pickle carries their `Data` and `LearningTask` classes,
    so unpickling it would make this file depend on their checkout being
    importable, which on a fresh pod it is not."""
    import pandas as pd

    folder = os.path.join(data_root(), "year")
    zip_path = os.path.join(folder, "YearPredictionMSD.txt.zip")
    npz_path = os.path.join(folder, "year_speed.npz")
    if os.path.exists(npz_path):
        z = np.load(npz_path)
        x, y = z["x"], z["y"]
    else:
        if not os.path.isfile(zip_path):
            raise RuntimeError(
                "year is not downloaded: %s is missing. Run "
                "`python tools/speed_gbdt_arm.py --download year` first "
                "(211 MB), OUTSIDE the timed run." % zip_path
            )
        frame = pd.read_csv(zip_path, header=None)
        x = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
        y = frame.iloc[:, 0].to_numpy(dtype=np.float32)
        np.savez(npz_path, x=x, y=y)
    n_train = 463715
    if size == "smoke" or rows_cap:
        cap = rows_cap or 50000
        x, y = _smoke_rows(x, y, cap + max(1, cap // 10))
        return _split_tail(x, y, "regression", "year")
    return Data("year", x[:n_train], x[n_train:], y[:n_train], y[n_train:],
                "regression")


def load_covtype(size, binary=False, rows_cap=None):
    """Forest covertype, 581,012 x 54, downloaded by scikit-learn itself into
    `~/scikit_learn_data` (about 11 MB compressed). The forest lanes' shipped
    dataset.

    `binary=True` is the `covtype2` task: y == 2 (Lodgepole Pine, the
    majority class) against the rest. It exists because our `GradientBoosting`
    Python surface has no `MultiClass` -- the Mojo layer implements it, the
    wrapper is one-dimensional -- so the GBDT lanes cannot run the 7-class
    problem at all (DEVIATION 1838). The derived task is applied IDENTICALLY
    to every arm, it is named differently in the `shape=` tag so it can never
    be read as the 7-class result, and it is not a dataset chosen for
    flattering anyone: it is the only covtype an arm that refuses multiclass
    can run."""
    from sklearn.datasets import fetch_covtype

    ds = fetch_covtype()
    x = np.ascontiguousarray(ds.data, dtype=np.float32)
    if binary:
        y = (ds.target == 2).astype(np.float32)
        name, task, n_classes = "covtype2", "binary", 2
    else:
        y = (ds.target.astype(np.int64) - 1).astype(np.float32)
        name, task, n_classes = "covtype", "multiclass", 7
    if size == "smoke" or rows_cap:
        x, y = _smoke_rows(x, y, rows_cap or 50000)
    return _split_tail(x, y, task, name, n_classes)


def load_synth(size, task, rows_cap=None):
    rows = 800000 if size == "shipped" else 50000
    feats = 100 if size == "shipped" else 32
    if rows_cap:
        rows = min(rows, rows_cap)
    if task == "binary":
        x, y = _synth_binary(rows, feats)
        return _split_tail(x, y, "binary", "synthclf", 2)
    x, y = _synth_regression(rows, feats)
    return _split_tail(x, y, "regression", "synth")


def load_anomaly(size, rows_cap=None):
    """A planted-anomaly fixture for the isolation-forest lane.

    covtype and year carry no anomaly labels, so an isolation forest scored
    on them has no accuracy column at all, and this slice's whole premise is
    that a timing without an accuracy column is meaningless. So the iforest
    lane runs a synthetic fixture whose anomalies are known by construction:
    a 99% inlier Gaussian blob and a 1% uniform outlier shell, AUC of the
    anomaly score against the planted label. Stated here rather than buried,
    because it is a weaker dataset than the other lanes get."""
    rows = 500000 if size == "shipped" else 50000
    feats = 32 if size == "shipped" else 16
    if rows_cap:
        rows = min(rows, rows_cap)
    rng = np.random.default_rng(11)
    n_out = max(1, rows // 100)
    n_in = rows - n_out
    inliers = rng.normal(size=(n_in, feats)).astype(np.float32)
    outliers = rng.uniform(-8.0, 8.0, size=(n_out, feats)).astype(np.float32)
    x = np.vstack((inliers, outliers))
    lab = np.concatenate((np.zeros(n_in, np.float32),
                          np.ones(n_out, np.float32)))
    order = rng.permutation(rows)
    x = np.ascontiguousarray(x[order])
    lab = lab[order]
    # The anomaly lane fits and scores the SAME rows, which is what
    # `IsolationForest` is for; the test split is the scoring set.
    d = Data("anomaly", x, x, lab, lab, "anomaly")
    d.y_anom = lab
    return d


def load_higgs(size, rows_cap=None):
    """HIGGS, 11,000,000 x 28, binary. THE LARGE-LOAD DATASET.

    `year` and `covtype` both stop near half a million rows, and half a
    million rows is not where a GPU tree learner is decided. Every ratio in
    the first NVIDIA trees table was taken at that size, and the question
    it cannot answer is the one that matters: our per-row work has measured
    FASTER than CatBoost's while our fixed per-tree cost measured 5.7x
    theirs, so the ratios should improve monotonically with rows and the
    forest gap should close hardest. That is a prediction, and it is
    untestable on a dataset that ends at 522,911 rows.

    HIGGS is the dataset NVIDIA's own gbm-bench uses for exactly this, so
    the ladder runs on data the vendors already benchmark themselves on
    rather than on a synthetic fixture we chose. The split is gbm-bench's:
    the LAST 500,000 rows are test, everything before them is train, no
    shuffle.

    `rows_cap` is the ladder. It caps the TRAINING rows only and takes them
    from the FRONT, deterministically, so `--rows 1000000` and
    `--rows 5000000` are nested prefixes of the same data and every arm at
    every rung scores the SAME 500,000 test rows. A rung is therefore a
    comparison of load, not of two different problems.

    THE DOWNLOAD IS 2.6 GB and it is a SEPARATE, EXPLICITLY NAMED STEP
    (`--download higgs`), never something a timed run does on its own, for
    the reason `load_year` gives: the orchestrator budgets a lease around
    it. The decoded cache is another 1.3 GB of float32.
    """
    folder = os.path.join(data_root(), "higgs")
    gz_path = os.path.join(folder, "HIGGS.csv.gz")
    npz_path = os.path.join(folder, "higgs_speed.npz")
    if os.path.exists(npz_path):
        z = np.load(npz_path)
        x, y = z["x"], z["y"]
    else:
        if not os.path.isfile(gz_path):
            raise RuntimeError(
                "higgs is not downloaded: %s is missing. Run "
                "`python tools/speed_gbdt_arm.py --download higgs` first "
                "(2.6 GB), OUTSIDE the timed run." % gz_path
            )
        import pandas as pd
        # Chunked, because the frame is 11M x 29 and reading it whole
        # alongside the float32 copy peaks near 5 GB of host memory on a
        # box whose whole job is the GPU.
        xs, ys = [], []
        for chunk in pd.read_csv(gz_path, header=None, dtype=np.float32,
                                 chunksize=1_000_000):
            arr = chunk.to_numpy(dtype=np.float32)
            ys.append(arr[:, 0])
            xs.append(arr[:, 1:])
        x = np.ascontiguousarray(np.concatenate(xs))
        y = np.ascontiguousarray(np.concatenate(ys))
        del xs, ys
        np.savez(npz_path, x=x, y=y)
    n_test = 500000
    n_train = x.shape[0] - n_test
    if size == "smoke":
        rows_cap = min(rows_cap or 50000, 50000)
    if rows_cap:
        n_train = min(n_train, rows_cap)
    # THE TEST ROWS ARE THE FIXED TAIL AT EVERY RUNG. Slicing them off the
    # end rather than off `n_train` is what makes two rungs comparable.
    x_train = np.ascontiguousarray(x[:n_train])
    y_train = np.ascontiguousarray(y[:n_train])
    x_test = np.ascontiguousarray(x[-n_test:])
    y_test = np.ascontiguousarray(y[-n_test:])
    return Data("higgs", x_train, x_test, y_train, y_test, "binary", 2)


#: Which dataset each lane runs by default, and what it falls back to.
#: `year` for the boosting lanes because it is the regression dataset the
#: existing M4 GBDT tables were taken on; `covtype` for the forest lanes
#: because the RF and ET tables were taken on it.
LANE_DEFAULT_DATASET = {
    "gbdt-symmetric": "year",
    "gbdt-depthwise": "year",
    "gbdt-lossguide": "year",
    "rf": "covtype",
    "et": "covtype",
    "iforest": "anomaly",
}


def load_dataset(name, size, rows_cap=None):
    if name == "higgs":
        return load_higgs(size, rows_cap)
    if name == "year":
        return load_year(size, rows_cap)
    if name == "covtype":
        return load_covtype(size, False, rows_cap)
    if name == "covtype2":
        return load_covtype(size, True, rows_cap)
    if name == "synth":
        return load_synth(size, "regression", rows_cap)
    if name == "synthclf":
        return load_synth(size, "binary", rows_cap)
    if name == "anomaly":
        return load_anomaly(size, rows_cap)
    raise SystemExit("unknown dataset " + repr(name))


def load_with_fallback(name, size, rows_cap=None):
    """A download that failed must not cost the lease. Fall back to the
    synthetic fixture of the SAME task, say so on stderr, and let every line
    carry the fallback's own `shape=` tag so nobody reads a synth number as a
    year number."""
    try:
        return load_dataset(name, size, rows_cap)
    except Exception as exc:                       # noqa: BLE001
        sys.stderr.write(
            "speed_gbdt_arm: dataset %r unavailable (%s); falling back to "
            "the synthetic fixture. Every line will say so in shape=.\n"
            % (name, exc)
        )
        if name in ("covtype", "covtype2", "synthclf", "higgs"):
            return load_dataset("synthclf", size, rows_cap)
        if name == "anomaly":
            return load_dataset("anomaly", size, rows_cap)
        return load_dataset("synth", size, rows_cap)


def download(name):
    """The explicitly named, untimed fetch step. Prints the size it pulled so
    the orchestrator can budget the lease around it."""
    if name == "higgs":
        import urllib.request
        folder = os.path.join(data_root(), "higgs")
        os.makedirs(folder, exist_ok=True)
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "00280/HIGGS.csv.gz")
        dest = os.path.join(folder, "HIGGS.csv.gz")
        if os.path.isfile(dest):
            print("higgs already present: %s (%.1f MB)"
                  % (dest, os.path.getsize(dest) / 1e6))
        else:
            print("downloading %s -> %s (about 2.6 GB)" % (url, dest))
            urllib.request.urlretrieve(url, dest)
            print("higgs: %.1f MB" % (os.path.getsize(dest) / 1e6))
        # Decode once, here. The gzip csv parse is several MINUTES and it
        # must not happen inside a timed run's setup on a leased box.
        d = load_higgs("shipped")
        print("higgs decoded to %s (train %d x %d, test %d)"
              % (os.path.join(folder, "higgs_speed.npz"),
                 d.X_train.shape[0], d.X_train.shape[1], d.X_test.shape[0]))
        return
    if name == "year":
        import urllib.request
        folder = os.path.join(data_root(), "year")
        os.makedirs(folder, exist_ok=True)
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "00203/YearPredictionMSD.txt.zip")
        dest = os.path.join(folder, "YearPredictionMSD.txt.zip")
        if os.path.isfile(dest):
            print("year already present: %s (%.1f MB)"
                  % (dest, os.path.getsize(dest) / 1e6))
        else:
            print("downloading %s -> %s (about 211 MB)" % (url, dest))
            urllib.request.urlretrieve(url, dest)
            print("year: %.1f MB" % (os.path.getsize(dest) / 1e6))
        # Decode once, here, so the first timed run does not pay a
        # two-minute csv parse inside its setup.
        load_year("shipped")
        print("year decoded to %s" % os.path.join(folder, "year_speed.npz"))
        return
    if name in ("covtype", "covtype2"):
        from sklearn.datasets import fetch_covtype
        fetch_covtype()
        print("covtype fetched into ~/scikit_learn_data (about 11 MB "
              "compressed, 581012 x 54)")
        return
    if name in ("synth", "synthclf", "anomaly"):
        print("%s is generated in-process; nothing to download" % name)
        return
    raise SystemExit("nothing known to download for " + repr(name))


# --------------------------------------------------------------------------
# The hyper-parameters, in ONE place, held equal across every arm of a lane.
# --------------------------------------------------------------------------

def lane_config(lane, size):
    """The knobs every arm of `lane` is given, spelled once.

    IDENTICAL CONFIG ON EVERY ARM; THE DEVICE IS THE ONLY VARIABLE. Where a
    default differs between libraries it is set EXPLICITLY on all of them and
    the difference is named in a comment, because an unset default is how two
    arms end up solving two problems.

    The GBDT knobs:

      n_estimators 100   Matches `tools/nvidia_forest_bench.sh`'s default so
                         this run and that one are the same size of job.
                         CatBoost's own default is 1000 and sklearn's is 100.
      max_depth 6        CatBoost's default. XGBoost's is 6 too; LightGBM has
                         no depth limit by default (-1) and is pinned here.
      learning_rate 0.1  Set explicitly on every arm. CatBoost's constructor
                         value is 0.03 but a CatBoost user with the rate
                         unset gets a value FITTED from the pool
                         (`options_helper.cpp:252-288`, about 0.097 at 800k
                         rows), which is not ported. Pinning it removes the
                         whole question.
      l2 1.0             CatBoost `l2_leaf_reg`, XGBoost `reg_lambda`,
                         LightGBM `lambda_l2`. Same quantity, three
                         spellings. CatBoost's default is 3.0, XGBoost's 1.0,
                         LightGBM's 0.0.
      borders 254        DEVIATION 1832, AND IT IS AN OFF-BY-ONE. CatBoost's
                         `border_count` counts BORDERS; XGBoost's `max_bin`
                         and LightGBM's `max_bin` count BINS. 254 borders is
                         255 bins, so the XGBoost and LightGBM arms get
                         max_bin=255 to quantize the same way. Our
                         `border_count` is CatBoost's, so it takes 254.
                         (Defaults would have been 128 ours, 254 CatBoost
                         CPU, 128 CatBoost GPU, 256 XGBoost, 255 LightGBM --
                         five different grids.)
      no bagging         DEVIATION 1833. `bootstrap_type='No'` on CatBoost
                         and ours, `subsample=1.0`/`colsample_bytree=1.0` on
                         XGBoost, `bagging_fraction=1.0`/
                         `feature_fraction=1.0` on LightGBM. Row and column
                         sampling are the largest RNG term in a boosting fit
                         and the arms cannot share a generator, so they are
                         switched off rather than matched. It makes the fits
                         SLOWER on every arm equally.
      boosting Plain     DEVIATION 1841, AND THIS ONE IS A TRAP. CatBoost's
                         `boosting_type` default is data-dependent: `Ordered`
                         on small pools and `Plain` on large ones, and
                         `Ordered` is a DIFFERENT ALGORITHM that trains
                         several permutations. A smoke-size run would have
                         silently timed Ordered against everyone else's
                         Plain. Both CatBoost arms are pinned to `Plain`.
      seed 7             Every arm.

    The forest knobs:

      n_estimators 100   cuML's, sklearn's and ours.
      max_depth 16       Set explicitly on every arm, and as of 2026-09-01
                         it is NOBODY's default. It was cuML's until
                         v26.08.00 -- this repo's pin -- changed it to None
                         (randomforestclassifier.py:68-74). sklearn's is
                         also None. Both mean grow until pure, a different
                         and much more expensive tree, so every arm is
                         pinned to 16 rather than left to a library.
      max_features       'sqrt' for classification, 1.0 for regression -- the
                         RF definition and every library's own default for
                         that task. Set explicitly on all.
      n_bins 128         cuML's and ours. DEVIATION 1834 AND IT IS NOT
                         REMOVABLE: sklearn searches EXACT thresholds and has
                         no bin count. That is an ALGORITHM difference in
                         sklearn's favour on accuracy and against it on
                         speed, it is what `PARITY_NOTES['rf-quantile-splits']`
                         already records for the gbm-bench arms, and it is
                         recorded rather than corrected because correcting it
                         would mean not running cuML's algorithm.
      bootstrap          True for rf, False for et. LightGBM's `rf` boosting
                         REFUSES `bagging_fraction=1.0`, so its forest arms
                         run 0.632/freq=1 -- DEVIATION 1835, forced by
                         LightGBM, already recorded as
                         `PARITY_NOTES['lgbm-rf-bagging']`.
    """
    smoke = size == "smoke"
    common = dict(
        seed=7,
        n_estimators=10 if smoke else 100,
    )
    if lane.startswith("gbdt-"):
        cfg = dict(common)
        cfg.update(
            max_depth=6,
            learning_rate=0.1,
            l2=1.0,
            borders=254,       # CatBoost border count; max_bin = borders + 1
            max_leaves=64,     # 2 ** 6, so the lossguide lane matches depth 6
            grow_policy={
                "gbdt-symmetric": "SymmetricTree",
                "gbdt-depthwise": "Depthwise",
                "gbdt-lossguide": "Lossguide",
            }[lane],
        )
        return cfg
    if lane in ("rf", "et"):
        cfg = dict(common)
        cfg.update(
            max_depth=16,
            n_bins=128,
            min_samples_leaf=1,
            min_samples_split=2,
            min_impurity_decrease=0.0,
            bootstrap=(lane == "rf"),
        )
        return cfg
    if lane == "iforest":
        cfg = dict(common)
        cfg.update(
            max_samples=256,
            max_features=1.0,
            bootstrap=False,
        )
        return cfg
    raise SystemExit("unknown lane " + repr(lane))


def max_features_for(data):
    """'sqrt' for a classification forest, 1.0 for a regression forest. The
    RF definition, and each library's own default for the task."""
    return "sqrt" if data.task in ("binary", "multiclass") else 1.0


# --------------------------------------------------------------------------
# An arm: build a model, fit it inside the timer, score it outside.
# --------------------------------------------------------------------------

class Arm(object):
    """One timed competitor.

    `make()` returns a fresh unfitted estimator; it runs OUTSIDE the timer,
    so any host-side setup a constructor does is not charged to the fit.
    `fit(model, data)` is the ONLY thing inside the timer. `sync()` also runs
    inside it, because a fit is not finished until the device says so, and
    for the libraries whose `fit` already blocks it is a documented no-op.
    `score(model, data)` runs outside and returns `(metric_name, value,
    prediction_vector)`."""

    def __init__(self, name, make, fit, score, sync=None, library=None):
        self.name = name
        self.make = make
        self.fit = fit
        self.score = score
        self.sync = sync or (lambda: None)
        #: Arms sharing a library are cross-checked for accuracy agreement.
        self.library = library or name.split("-")[0]


def _cuda_sync():
    """Device synchronize for the arms whose `fit` returns before the device
    is done. cupy ships with every RAPIDS install; if it is absent the sync
    is a no-op and the arm's number is then a LOWER BOUND rather than a
    measurement, which is why the absence is printed rather than swallowed."""
    try:
        import cupy
    except ImportError:
        return
    cupy.cuda.runtime.deviceSynchronize()


def _blocking(name):
    """A named no-op for a library whose `fit` is synchronous on the host.
    Spelled out rather than passed as `None` so that reading the arm table
    tells you which arms were checked and which were assumed."""
    del name
    return None


# ---- CatBoost -------------------------------------------------------------

def catboost_arms(lane, cfg, data, devices):
    """CatBoost CPU and CatBoost's CUDA learner. The symmetric-tree opponent,
    and per Andrew's standing order the ONLY opponent in `gbdt-symmetric`."""
    import catboost

    def _params(task_type):
        p = dict(
            iterations=cfg["n_estimators"],
            depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            l2_leaf_reg=cfg["l2"],
            border_count=cfg["borders"],
            random_seed=cfg["seed"],
            bootstrap_type="No",       # DEVIATION 1833
            boosting_type="Plain",     # DEVIATION 1841, the data-dependent trap
            grow_policy=cfg["grow_policy"],
            task_type=task_type,
            verbose=False,
            allow_writing_files=False,
        )
        if cfg["grow_policy"] == "Lossguide":
            p["max_leaves"] = cfg["max_leaves"]
        if task_type == "GPU":
            p["devices"] = "0"
        return p

    def make(task_type):
        p = _params(task_type)
        if data.task == "regression":
            return catboost.CatBoostRegressor(loss_function="RMSE", **p)
        if data.task == "binary":
            return catboost.CatBoostClassifier(loss_function="Logloss", **p)
        return catboost.CatBoostClassifier(loss_function="MultiClass", **p)

    out = []
    for dev in devices:
        task_type = "CPU" if dev == "cpu" else "GPU"
        out.append(Arm(
            "catboost-" + dev,
            (lambda tt: (lambda: make(tt)))(task_type),
            lambda m, d: m.fit(d.X_train, d.y_train),
            _score_sklearn_like,
            sync=lambda: _blocking("catboost"),
            library="catboost",
        ))
    return out


# ---- XGBoost --------------------------------------------------------------

def xgboost_arms(lane, cfg, data, devices):
    """XGBoost `tree_method='hist'` on CPU and on CUDA.

    DEVIATION 1831: XGBoost never appears in `gbdt-symmetric`. Its
    `grow_policy` is depthwise or lossguide and it has no symmetric mode, so
    the same argument that keeps LightGBM out of the symmetric pair keeps
    XGBoost out of it."""
    import xgboost as xgb

    policy = {"Depthwise": "depthwise", "Lossguide": "lossguide"}.get(
        cfg["grow_policy"])
    if policy is None:
        raise RuntimeError(
            "xgboost has no symmetric growth policy; the symmetric-tree "
            "comparison is CatBoost ONLY (standing order 2026-08-22, "
            "DEVIATION 1831)"
        )

    def _params(device):
        p = dict(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            reg_lambda=cfg["l2"],
            reg_alpha=0.0,
            # DEVIATION 1832: their max_bin counts BINS, CatBoost's
            # border_count counts BORDERS.
            max_bin=cfg["borders"] + 1,
            subsample=1.0,             # DEVIATION 1833
            colsample_bytree=1.0,
            colsample_bylevel=1.0,
            colsample_bynode=1.0,
            min_child_weight=1.0,
            tree_method="hist",
            grow_policy=policy,
            random_state=cfg["seed"],
            device=device,
            verbosity=0,
        )
        if policy == "lossguide":
            p["max_leaves"] = cfg["max_leaves"]
        return p

    def make(device):
        p = _params(device)
        if data.task == "regression":
            return xgb.XGBRegressor(objective="reg:squarederror", **p)
        if data.task == "binary":
            return xgb.XGBClassifier(objective="binary:logistic", **p)
        return xgb.XGBClassifier(objective="multi:softprob",
                                 num_class=data.n_classes, **p)

    out = []
    for dev in devices:
        device = "cpu" if dev == "cpu" else "cuda"
        out.append(Arm(
            "xgboost-" + dev,
            (lambda dv: (lambda: make(dv)))(device),
            lambda m, d: m.fit(d.X_train, d.y_train),
            _score_sklearn_like,
            sync=_cuda_sync,
            library="xgboost",
        ))
    return out


# ---- LightGBM -------------------------------------------------------------

def lightgbm_arms(lane, cfg, data, devices):
    """LightGBM CPU and LightGBM CUDA.

    THE PIP AND CONDA WHEELS HAVE NO CUDA SUPPORT. `device_type='cuda'`
    requires a source build with `USE_CUDA=ON`, so the cuda arm is expected
    to REFUSE on a box where nobody built it, and its refusal must be visible
    as a refusal rather than as a missing row.

    LightGBM never appears in `gbdt-symmetric`: leaf-wise is its only growth
    algorithm (Andrew's standing order, 2026-08-22). In the forest lanes it
    runs `boosting_type='rf'`, which is what
    `PARITY_NOTES['lgbm-rf-bagging']` and `tools/nvidia_forest_bench.sh`
    already compare against."""
    import lightgbm as lgb

    forest = lane in ("rf", "et")
    p = dict(
        n_estimators=cfg["n_estimators"],
        random_state=cfg["seed"],
        verbose=-1,
        min_child_samples=1,
        min_child_weight=0.0,
        min_split_gain=0.0,
    )
    if forest:
        # LightGBM REFUSES rf boosting with bagging_fraction=1.0
        # (DEVIATION 1835): the asymmetry is forced by LightGBM, not chosen.
        mf = max_features_for(data)
        n_feat = data.X_train.shape[1]
        frac = (float(np.sqrt(n_feat)) / n_feat) if mf == "sqrt" else 1.0
        p.update(
            boosting_type="rf",
            bagging_fraction=0.632,
            bagging_freq=1,
            feature_fraction=frac,
            max_depth=cfg["max_depth"],
            num_leaves=2 ** min(cfg["max_depth"], 15),
            max_bin=255,
            reg_lambda=0.0,
            learning_rate=1.0,   # ignored by rf boosting; pinned, not left
        )
        if lane == "et":
            p["extra_trees"] = True
    else:
        if cfg["grow_policy"] == "SymmetricTree":
            raise RuntimeError(
                "the symmetric-tree comparison is CatBoost ONLY (standing "
                "order 2026-08-22); LightGBM has no symmetric mode"
            )
        p.update(
            max_depth=cfg["max_depth"],
            num_leaves=cfg["max_leaves"],
            learning_rate=cfg["learning_rate"],
            reg_lambda=cfg["l2"],
            max_bin=cfg["borders"] + 1,      # DEVIATION 1832
            bagging_fraction=1.0,            # DEVIATION 1833
            feature_fraction=1.0,
        )

    def make(device_type):
        q = dict(p)
        q["device_type"] = device_type
        if data.task == "regression":
            return lgb.LGBMRegressor(objective="regression", **q)
        if data.task == "binary":
            return lgb.LGBMClassifier(objective="binary", **q)
        return lgb.LGBMClassifier(objective="multiclass",
                                  num_class=data.n_classes, **q)

    out = []
    for dev in devices:
        device_type = "cpu" if dev == "cpu" else "cuda"
        out.append(Arm(
            "lightgbm-" + ("cpu" if dev == "cpu" else "cuda"),
            (lambda dt: (lambda: make(dt)))(device_type),
            lambda m, d: m.fit(d.X_train, d.y_train),
            _score_sklearn_like,
            sync=_cuda_sync,
            library="lightgbm",
        ))
    return out


# ---- cuML and scikit-learn forests ---------------------------------------

def cuml_rf_arm(lane, cfg, data):
    """cuML's RandomForest on the GPU: NVIDIA's own forest, and the library
    `ensemble/` is a port of. The honest opponent for the `rf` lane.

    `n_streams` is left at cuML's default. Their own documentation says a
    value above 1 makes the fit NON-REPRODUCIBLE, which is fine here and only
    here: this slice measures the FAST path, which is the explicitly
    non-deterministic arm. Pinning it to 1 would be benchmarking a
    configuration no cuML user runs."""
    from cuml.ensemble import RandomForestClassifier, RandomForestRegressor

    if lane != "rf":
        raise RuntimeError(
            "cuML has no ExtraTrees estimator: its RandomForest searches "
            "quantile splits, not the uniform-random thresholds that define "
            "ExtraTrees. The like-for-like comparator for `et` is sklearn."
        )
    mf = max_features_for(data)
    common = dict(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        max_features=mf,
        n_bins=cfg["n_bins"],
        min_samples_leaf=cfg["min_samples_leaf"],
        min_samples_split=cfg["min_samples_split"],
        min_impurity_decrease=cfg["min_impurity_decrease"],
        bootstrap=cfg["bootstrap"],
        random_state=cfg["seed"],
    )

    def make():
        if data.task == "regression":
            return RandomForestRegressor(split_criterion=2, **common)
        # 0 = GINI, cuML's default and ours.
        return RandomForestClassifier(split_criterion=0, **common)

    def fit(model, d):
        # cuML's classifier wants int32 labels. The cast is a HOST cast on an
        # already-loaded array; it is prepared once in `_cuml_labels` outside
        # the timer and only indexed here.
        return model.fit(d.X_train, d._cuml_y)

    return Arm("cuml-rf-gpu", make, fit, _score_sklearn_like,
               sync=_cuda_sync, library="cuml")


def sklearn_forest_arm(lane, cfg, data):
    """scikit-learn's RandomForest / ExtraTrees on every core.

    `n_jobs=-1` deliberately, reusing `PARITY_NOTES['skl-threads']`'s reason:
    gbm-bench's own sklearn arms pass no `n_jobs` and therefore run on ONE
    core, and beating a single-core sklearn is not a result.

    `max_depth` is pinned to 16 rather than left at sklearn's `None`. Their
    default grows every tree until its leaves are pure, which is a different
    and far more expensive tree than cuML's depth-16 default, and comparing
    them would be comparing tree sizes."""
    import sklearn.ensemble as sken

    common = dict(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        max_features=max_features_for(data),
        min_samples_leaf=cfg["min_samples_leaf"],
        min_samples_split=cfg["min_samples_split"],
        min_impurity_decrease=cfg["min_impurity_decrease"],
        bootstrap=cfg["bootstrap"],
        random_state=cfg["seed"],
        n_jobs=-1,
    )

    def make():
        if lane == "et":
            if data.task == "regression":
                return sken.ExtraTreesRegressor(**common)
            return sken.ExtraTreesClassifier(**common)
        if data.task == "regression":
            return sken.RandomForestRegressor(**common)
        return sken.RandomForestClassifier(**common)

    name = "sklearn-et-cpu" if lane == "et" else "sklearn-rf-cpu"
    return Arm(name, make,
               lambda m, d: m.fit(d.X_train, d.y_train),
               _score_sklearn_like,
               sync=lambda: _blocking("sklearn"),
               library="sklearn")


# ---- isolation forests ----------------------------------------------------

def sklearn_iforest_arm(cfg, data):
    """scikit-learn's IsolationForest, on every core.

    LABEL IT HONESTLY: this is a CPU opponent, and unless the cuML arm below
    runs it is the ONLY opponent this lane has. A GPU-versus-CPU ratio in
    this lane is not the same claim as a GPU-versus-GPU ratio in the others,
    and the FOREST_SPEED.md table says so in the lane's own row."""
    from sklearn.ensemble import IsolationForest

    def make():
        return IsolationForest(
            n_estimators=cfg["n_estimators"],
            max_samples=cfg["max_samples"],
            max_features=cfg["max_features"],
            bootstrap=cfg["bootstrap"],
            contamination="auto",
            random_state=cfg["seed"],
            n_jobs=-1,
        )

    def score(model, d):
        s = -np.asarray(model.score_samples(d.X_test), dtype=np.float64)
        return [("auc", auc(d.y_anom, s), s)]

    return Arm("sklearn-iforest-cpu", make,
               lambda m, d: m.fit(d.X_train),
               score, sync=lambda: _blocking("sklearn"), library="sklearn")


def cuml_iforest_arm(cfg, data):
    """cuML's IsolationForest, IF this cuML has one.

    DEVIATION 1837, AND IT IS A DISAGREEMENT WITH THE BRIEF. The brief for
    this file states that cuML has no IsolationForest. Our own port says
    otherwise: `python/mojolearn/_iforest_impl.py` cites
    `isolation_forest.pyx:663-702` for cuML's `max_samples` resolution, and
    `isolation_forest/` is described throughout as a port of cuML's. Both
    cannot be right, and the cheap way to settle it is to try the import on
    the box and print what happens. If it is absent the arm REFUSES by name
    and the lane falls back to sklearn alone; if it is present the lane gets
    the GPU-versus-GPU column every other lane has.

    Settling it also closes a real open question the port carries: DEVIATION
    750, cuML's `curand_u64` word order, has never been checked against a
    cuML binary because there has never been one on the same machine."""
    try:
        from cuml.ensemble import IsolationForest
    except ImportError:
        from cuml import IsolationForest    # older layouts

    def make():
        return IsolationForest(
            n_estimators=cfg["n_estimators"],
            max_samples=cfg["max_samples"],
            max_features=cfg["max_features"],
            bootstrap=cfg["bootstrap"],
            contamination="auto",
            random_state=cfg["seed"],
        )

    def score(model, d):
        s = -np.asarray(model.score_samples(d.X_test), dtype=np.float64)
        return [("auc", auc(d.y_anom, s), s)]

    return Arm("cuml-iforest-gpu", make,
               lambda m, d: m.fit(d.X_train),
               score, sync=_cuda_sync, library="cuml")


# ---- scoring, shared by every sklearn-shaped arm --------------------------

def _score_sklearn_like(model, d):
    """`(metric, value, prediction_vector)` triples for an estimator with the
    scikit-learn surface -- which is all of them, ours included.

    Regression gets RMSE. Binary gets BOTH logloss and AUC, because the two
    answer different questions and a boosting table with only one of them
    invites the reader to pick. Multiclass gets accuracy.

    A note about log loss that this repository has already been bitten by:
    gbm-bench's own Log_Loss column is ASYMMETRIC for CatBoost because it
    scores their RAW MARGINS (bench/external/README.md). Nothing here has
    that bug -- every arm is scored through `predict_proba`, which every one
    of these libraries defines as the probability after the link -- but the
    resemblance is close enough to be worth naming."""
    out = []
    if d.task == "regression":
        p = np.asarray(model.predict(d.X_test)).ravel()
        out.append(("rmse", rmse(d.y_test, p), p))
        return out
    proba = np.asarray(model.predict_proba(d.X_test), dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack((1.0 - proba, proba))
    if d.task == "binary":
        p1 = proba[:, 1]
        out.append(("logloss", logloss(d.y_test, p1), p1))
        out.append(("auc", auc(d.y_test, p1), None))
        return out
    labels = np.argmax(proba, axis=1)
    out.append(("accuracy", accuracy(d.y_test, labels), labels))
    return out


#: Public alias. `bench/speed/forest_speed_arm.py` scores our arm through the
#: SAME helper every opponent uses, on the same held-out rows; reaching for a
#: private name to do that would invite somebody to write a second scorer,
#: and two scorers is how an accuracy column stops being comparable.
score_sklearn_like = _score_sklearn_like


# --------------------------------------------------------------------------
# The opponent roster for a lane.
# --------------------------------------------------------------------------

def opponent_builders(lane, cfg, data, devices):
    """`[(arm_name, thunk)]`. Each thunk returns a LIST of arms or raises;
    the raise becomes an FSPEED-REFUSED line for every arm it would have
    produced, which is why the names are known before the thunk runs. An
    opponent that cannot be installed must be visible as a refusal, never as
    an absent row."""
    builders = []
    if lane == "gbdt-symmetric":
        # CatBoost ONLY. Standing order, 2026-08-22.
        builders.append((["catboost-cpu", "catboost-gpu"],
                         lambda: catboost_arms(lane, cfg, data, devices)))
    elif lane in ("gbdt-depthwise", "gbdt-lossguide"):
        builders.append((["catboost-cpu", "catboost-gpu"],
                         lambda: catboost_arms(lane, cfg, data, devices)))
        builders.append((["xgboost-cpu", "xgboost-gpu"],
                         lambda: xgboost_arms(lane, cfg, data, devices)))
        if lane == "gbdt-lossguide":
            # Leaf-wise growth IS LightGBM's algorithm; this is the only
            # boosting lane it belongs in.
            builders.append((["lightgbm-cpu", "lightgbm-cuda"],
                             lambda: lightgbm_arms(lane, cfg, data, devices)))
    elif lane == "rf":
        builders.append((["cuml-rf-gpu"],
                         lambda: [cuml_rf_arm(lane, cfg, data)]))
        builders.append((["sklearn-rf-cpu"],
                         lambda: [sklearn_forest_arm(lane, cfg, data)]))
        builders.append((["lightgbm-cpu", "lightgbm-cuda"],
                         lambda: lightgbm_arms(lane, cfg, data, devices)))
    elif lane == "et":
        # Named `cuml-et-gpu` so the refusal reads as "cuML has no ExtraTrees"
        # rather than as a missing RandomForest row.
        builders.append((["cuml-et-gpu"],
                         lambda: [cuml_rf_arm(lane, cfg, data)]))
        builders.append((["sklearn-et-cpu"],
                         lambda: [sklearn_forest_arm(lane, cfg, data)]))
        builders.append((["lightgbm-cpu", "lightgbm-cuda"],
                         lambda: lightgbm_arms(lane, cfg, data, devices)))
    elif lane == "iforest":
        builders.append((["cuml-iforest-gpu"],
                         lambda: [cuml_iforest_arm(cfg, data)]))
        builders.append((["sklearn-iforest-cpu"],
                         lambda: [sklearn_iforest_arm(cfg, data)]))
    else:
        raise SystemExit("unknown lane " + repr(lane))
    return builders


def accel_visible():
    """Is this a GPU vendor's box?"""
    try:
        import torch                                   # noqa: PLC0415
        if torch.cuda.is_available():
            return True
    except Exception:                                  # noqa: BLE001
        pass
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if os.environ.get(var, "").strip() not in ("", "-1"):
            return True
    return bool(shutil.which("nvidia-smi") or shutil.which("rocm-smi"))


def resolve_devices(requested, lane=None):
    """Which device arms of each opponent may run here.

    THE RULE, AND IT IS NOT A PREFERENCE. On NVIDIA and on AMD we compare
    against the vendor's GPU path ONLY. Their CPU path is for the MacBook,
    where it is the only path they have.

    A GPU-versus-CPU ratio is not the claim this project makes. Beating
    CatBoost's CPU learner by 1.70x on an H100 is not a result, it is a
    category error, and printing it beside the GPU column invites the table
    to be graded on the easy comparison. It also costs the lease:
    `lightgbm-cpu` took 89 SECONDS on 522,911 rows in the rf lane, and at a
    5,000,000-row rung it would spend most of a per-arm budget measuring
    something nobody asked about.

    `auto` -- the default -- is `gpu` wherever an accelerator is visible and
    `cpu` on the MacBook. An explicit list still wins, because the Apple
    runs need to ask for `cpu` by name, and because a deliberate override
    should be possible; what is not possible is getting the CPU arms by
    ACCIDENT on a box that is billing by the minute.

    THE CONSEQUENCE IS STATED, NOT WORKED AROUND. cuML ships no ExtraTrees
    and no IsolationForest, so on NVIDIA the `et` and `iforest` lanes have
    NO legal opponent, and every arm they would have had is refused BY NAME.
    That is a finding about the vendor's GPU coverage. It is not a licence
    to run scikit-learn on the host CPU and call it an opponent.
    """
    want = [d.strip().lower() for d in (requested or "").split(",")
            if d.strip()]
    if not want or want == ["auto"]:
        want = ["gpu"] if accel_visible() else ["cpu"]
        auto = True
    else:
        auto = False
    if "cpu" in want and accel_visible():
        dropped = [d for d in want if d == "cpu"]
        want = [d for d in want if d != "cpu"]
        if lane is not None and dropped:
            emit_refused(
                lane, "*-cpu",
                "GPU-PATH-ONLY: an accelerator is visible on this box, so the "
                "vendors' CPU arms do not run. On NVIDIA and AMD we compare "
                "against the vendor's GPU arm only; the CPU arm is the "
                "MacBook's. Set MOJOLEARN_SPEED_DEVICES=cpu to override, and "
                "then say out loud beside the number that you did.")
    if not want:
        want = ["gpu"]
    return want, auto


def build_opponents(lane, cfg, data, devices):
    """Run every builder inside its own `try` and turn a failure into
    refusals. Nothing here may take the process down: an opponent that will
    not install on a rented box is the NORMAL case, not the exception."""
    arms = []
    allow_cpu = "cpu" in devices
    for names, thunk in opponent_builders(lane, cfg, data, devices):
        # THE CHOKEPOINT FOR THE GPU-PATH-ONLY RULE, and it is here rather
        # than in each builder because two of the forest builders --
        # `sklearn_forest_arm` and `sklearn_iforest_arm` -- never took
        # `devices` at all. Gating at the call sites would have left those
        # two running scikit-learn on an H100's host CPU while every other
        # arm obeyed the rule, which is the worst of both: the table would
        # look GPU-only and would not be.
        blocked = [n for n in names if n.endswith("-cpu")] if not allow_cpu else []
        for name in blocked:
            emit_refused(lane, name,
                         "GPU-PATH-ONLY: %s is a CPU arm and this box has an "
                         "accelerator. On NVIDIA and AMD we compare against "
                         "the vendor's GPU path only; their CPU path is the "
                         "MacBook's." % name)
        if blocked and len(blocked) == len(names):
            continue
        try:
            arms.extend(thunk())
        except Exception as exc:                   # noqa: BLE001
            reason = " ".join(str(exc).split()) or exc.__class__.__name__
            for name in names:
                emit_refused(lane, name, "%s: %s" % (exc.__class__.__name__,
                                                     reason))
    return arms


def prepare_cuml_labels(data):
    """cuML's classifier wants int32 labels; make the cast ONCE, outside
    every timer, so no arm is charged for it."""
    data._cuml_y = (
        data.y_train
        if data.task == "regression"
        else np.ascontiguousarray(data.y_train, dtype=np.int32)
    )


# --------------------------------------------------------------------------
# The runner. Arms ALTERNATE; they never run in blocks.
# --------------------------------------------------------------------------

def run(lane, arms, data, n_rounds, size, dev=None):
    """One untimed warm-up per arm, then `n_rounds` timed rounds in which
    every surviving arm takes one turn before any arm takes its second.

    THE ALTERNATION IS THE POINT AND NOT A STYLE CHOICE. A rented box may
    throttle mid-run. Blocks give you the first arm's cold clocks against the
    last arm's hot ones and no way to tell; alternation spreads the drift
    across every arm, and a ratio survives what an absolute number does not.

    A per-arm wall budget and a whole-process deadline keep a slow CPU arm
    from eating the lease that the GPU arms are the point of. An arm that
    exceeds either is dropped from the rotation with a refusal carrying the
    reason, so the table says what happened."""
    dev = dev or device_string()
    budget = per_arm_budget_s()
    deadline = time.time() + process_deadline_s()
    spent = {a.name: 0.0 for a in arms}
    last_model = {}

    for arm in arms:
        emit_header(lane, arm.name, dev, n_rounds, size)

    live = []
    for arm in arms:
        if time.time() > deadline:
            emit_refused(lane, arm.name, "process deadline reached before "
                                         "warm-up")
            continue
        try:
            t0 = time.perf_counter()
            model = arm.make()
            arm.fit(model, data)
            arm.sync()
            ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:                   # noqa: BLE001
            emit_refused(lane, arm.name, "%s during warm-up: %s"
                         % (exc.__class__.__name__,
                            " ".join(str(exc).split())))
            continue
        emit_warmup(lane, arm.name, data.tag, ms)
        spent[arm.name] += ms / 1000.0
        last_model[arm.name] = model
        live.append(arm)

    for r in range(1, n_rounds + 1):
        for arm in list(live):
            if time.time() > deadline:
                emit_refused(lane, arm.name,
                             "process deadline reached at round %d" % r)
                live.remove(arm)
                continue
            if spent[arm.name] > budget:
                emit_refused(lane, arm.name,
                             "per-arm budget %.0fs exceeded after %.1fs; "
                             "remaining rounds skipped"
                             % (budget, spent[arm.name]))
                live.remove(arm)
                continue
            try:
                t0 = time.perf_counter()
                model = arm.make()
                arm.fit(model, data)
                arm.sync()
                ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:               # noqa: BLE001
                emit_refused(lane, arm.name, "%s at round %d: %s"
                             % (exc.__class__.__name__, r,
                                " ".join(str(exc).split())))
                live.remove(arm)
                continue
            spent[arm.name] += ms / 1000.0
            last_model[arm.name] = model
            digest = None
            try:
                triples = arm.score(model, data)
                for _, _, vec in triples:
                    if vec is not None:
                        digest = hash_predictions(vec)
                        break
            except Exception:                      # noqa: BLE001
                digest = None
            emit_round(lane, arm.name, data.tag, r, ms, digest)

    # Accuracy, once per arm, from the last model each arm produced. Outside
    # every timer, and after the whole table, so a scoring failure cannot
    # perturb a timing.
    scores = {}
    for arm in arms:
        model = last_model.get(arm.name)
        if model is None:
            continue
        try:
            for metric, value, _ in arm.score(model, data):
                emit_acc(lane, arm.name, metric, value)
                scores.setdefault((arm.library, metric), []).append(
                    (arm.name, value))
        except Exception as exc:                   # noqa: BLE001
            emit_refused(lane, arm.name, "%s while scoring: %s"
                         % (exc.__class__.__name__,
                            " ".join(str(exc).split())))

    _check_same_library_agreement(lane, scores)
    return live


def _check_same_library_agreement(lane, scores):
    """DEVIATION 1839. Two arms of the SAME library that disagree about the
    answer were not given identical configurations, whatever the config dict
    says. This is the only automatic check that the fairness rule was
    actually obeyed rather than merely intended, and it is worth more than
    the config dict because it reads the RESULT."""
    for (library, metric), entries in sorted(scores.items()):
        if len(entries) < 2:
            continue
        vals = [v for _, v in entries if v == v]     # drop NaN
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        scale = max(abs(lo), abs(hi), 1e-12)
        delta = (hi - lo) / scale
        if delta > _ACC_TOL:
            emit_note(
                lane, [n for n, _ in entries], metric, delta,
                "two arms of %s differ by %.2f%% on %s (tolerance %.2f%%); "
                "their configurations were NOT identical, so the timing "
                "ratio compares two different problems"
                % (library, 100.0 * delta, metric, 100.0 * _ACC_TOL),
            )


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser(prog=None):
    p = argparse.ArgumentParser(
        prog=prog,
        description="the NVIDIA-native opponents for the forest slice",
    )
    p.add_argument("--lane", choices=LANE_NAMES,
                   help="which lane to run; ONE lane per process, because a "
                        "lane that segfaults must not take the others down")
    p.add_argument("--dataset", default=None,
                   help="higgs, year, covtype, covtype2, synth, synthclf, "
                        "anomaly; the lane's own default if unset. `higgs` "
                        "is the LARGE-LOAD dataset (11M x 28) and is what "
                        "--rows climbs.")
    p.add_argument("--devices", default="cpu,gpu",
                   help="which device arms of each opponent to run; the "
                        "default runs BOTH, which is the whole point on "
                        "NVIDIA")
    p.add_argument("--rows", type=int, default=None,
                   help="cap the training rows. On `higgs` this IS the load "
                        "ladder: rungs are nested prefixes of the same "
                        "data scored against the same fixed 500,000-row "
                        "tail, so 1000000 vs 5000000 is a comparison of "
                        "LOAD and not of two problems. Every line carries "
                        "the row count in shape=.")
    p.add_argument("--download", default=None,
                   help="fetch a dataset and exit; a SEPARATE step, outside "
                        "any timed run")
    p.add_argument("--list-arms", action="store_true",
                   help="print the arm roster for the lane and exit")
    p.add_argument("--arms", default=None,
                   help="comma-separated arm names to run, a SUBSET of the "
                        "lane's roster. This exists because the opponents do "
                        "not all share an interpreter: cuml and cuvs are not "
                        "on conda-forge and live on the pod's system python, "
                        "while catboost, lightgbm and our own extension live "
                        "in the pixi `gbmbench` environment (pixi.toml). One "
                        "process cannot import both sets, so a vendor leg "
                        "runs the lane TWICE with different --arms and the "
                        "tables are merged. An arm named here that the "
                        "roster does not have is a REFUSAL, not a silent "
                        "empty run.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.download:
        download(args.download)
        return 0
    if not args.lane:
        build_parser().error("--lane is required (or --download)")

    size = size_tag()
    dataset = args.dataset or LANE_DEFAULT_DATASET[args.lane]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    if args.lane.startswith("gbdt-") and dataset == "covtype":
        # DEVIATION 1838: our GBDT Python surface has no MultiClass, so the
        # 7-class problem is not something every arm can run. Refuse by name
        # rather than quietly swapping the task underneath the reader.
        emit_refused(args.lane, "ours",
                     "covtype is 7-class and mojolearn.GradientBoosting has "
                     "no MultiClass on the Python surface; use "
                     "--dataset covtype2 (the derived binary task) or year")
        dataset = "covtype2"

    data = load_with_fallback(dataset, size, args.rows)
    cfg = lane_config(args.lane, size)
    prepare_cuml_labels(data)

    if args.list_arms:
        for names, _ in opponent_builders(args.lane, cfg, data, devices):
            for name in names:
                print(name)
        return 0

    arms = build_opponents(args.lane, cfg, data, devices)
    if args.arms:
        # A FILTER IS EXPLICIT INTENT AND AN ABSENT ROW IS NOT. `build_opponents`
        # already turns an uninstallable opponent into a visible refusal rather
        # than a missing line; selecting a subset must keep that property, so a
        # name asked for and not found is refused BY NAME here.
        wanted = [n for n in (x.strip() for x in args.arms.split(",")) if n]
        have = {a.name for a in arms}
        for name in wanted:
            if name not in have:
                emit_refused(args.lane, name,
                             "not in this lane's roster on this interpreter; "
                             "the roster here is: %s" % ",".join(sorted(have)))
        arms = [a for a in arms if a.name in wanted]
    if not arms:
        emit_refused(args.lane, "all-opponents",
                     "no opponent could be constructed on this box")
        return 1
    run(args.lane, arms, data, rounds(), size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
