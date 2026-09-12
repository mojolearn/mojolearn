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
import contextlib
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
#: neural-network training-step lane (`archive/plans/training/TRAINING_LOOP_PLAN.md`) and it
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
    """Requested MojoLearn tier, not a claim about competitor arithmetic.

    The MojoLearn runner separately verifies the actual binding before timing.
    Opponent-only runs retain this label solely as experiment context.
    """
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
    if mode not in ("fast", "deterministic", "identical"):
        raise ValueError("invalid MOJOLEARN_NUMERIC_MODE: " + repr(mode))
    return mode.upper()


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
    # An AMD box has no nvidia-smi; `rocm-smi --showproductname` names the
    # card ("Card Series: AMD Instinct MI325X").
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        for line in out.stdout.splitlines():
            if "Card Series" in line and ":" in line:
                return line.rsplit(":", 1)[1].strip().replace(" ", "_")
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

    Equal repeated hashes are a same-device witness, not cross-device
    qualification. Competitor GPU fits may vary. An IDENTICAL MojoLearn arm
    must retain its own exact predictions; different libraries need not match.
    """
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
                 n_classes=0, y_anom=None, cat_idx=None):
        self.name = name
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.task = task
        self.n_classes = n_classes
        # Only for the anomaly lane: the planted anomaly labels of X_test.
        self.y_anom = y_anom
        # Column indices holding DENSE CATEGORY CODES 0..k-1, or () when every
        # column is numeric. Until criteo landed (2026-09-12) this was always
        # empty, which is why no benchmark had ever exercised the categorical
        # path: DEVIATION 2634 skips the CTR target prep when no column is
        # categorical, so its measured 0.9603 came entirely from the SKIP side.
        # An arm that can use categoricals reads this; one that cannot ignores
        # it and is then timed on a different problem, so a lane that passes
        # cat_idx to one arm must pass it to all of them.
        self.cat_idx = tuple(cat_idx or ())
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

    folder = os.path.join(data_root(), "year")
    zip_path = os.path.join(folder, "YearPredictionMSD.txt.zip")
    npz_path = os.path.join(folder, "year_speed.npz")
    # A CACHE ENTRY THAT EXISTS IS NOT A CACHE ENTRY THAT LOADS.
    #
    # `os.path.exists` stood here and it sent a ZERO-BYTE stub down the
    # cached branch: on 2026-09-03 a rented box was destroyed while the
    # decode's writes were still in page cache, and the volume came back
    # holding a complete 2,816,407,858-byte HIGGS.csv.gz beside an
    # `higgs_speed.npz` of 0 bytes. numpy raises "No data left in file" on
    # that, which reads like a truncated DOWNLOAD and is not one -- the
    # download was perfect and the decode was the casualty.
    #
    # A half-written cache must be treated as an absent one, because the
    # ingredient it is derived from is still sitting right next to it. The
    # decode is a couple of minutes; failing the whole leg to avoid it is
    # the wrong trade, and reporting it as a missing download sends the
    # next person to re-fetch 2.6 GB they already have.
    cached = None
    if os.path.exists(npz_path) and os.path.getsize(npz_path) > 0:
        try:
            cached = np.load(npz_path)
        except Exception as exc:                   # noqa: BLE001
            sys.stderr.write(
                "speed_gbdt_arm: %s is unreadable (%s); re-decoding from "
                "the gzip beside it\n" % (npz_path, exc))
            cached = None
    if cached is not None:
        z = cached
        x, y = z["x"], z["y"]
    else:
        if not os.path.isfile(zip_path):
            raise RuntimeError(
                "year is not downloaded: %s is missing. Run "
                "`python tools/speed_gbdt_arm.py --download year` first "
                "(211 MB), OUTSIDE the timed run." % zip_path
            )
        import pandas as pd
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
    # A CACHE ENTRY THAT EXISTS IS NOT A CACHE ENTRY THAT LOADS.
    #
    # `os.path.exists` stood here and it sent a ZERO-BYTE stub down the
    # cached branch: on 2026-09-03 a rented box was destroyed while the
    # decode was still in page cache, and the volume came back holding a
    # COMPLETE 2,816,407,858-byte HIGGS.csv.gz beside an higgs_speed.npz of
    # 0 bytes. numpy raises "No data left in file" on that, which reads like
    # a truncated DOWNLOAD and is not one -- the download was perfect and
    # the decode was the casualty.
    #
    # A half-written cache is treated as an absent one, because the gzip it
    # is derived from is sitting right beside it and the decode is a couple
    # of minutes. Failing the leg to avoid that is the wrong trade, and
    # reporting it as a missing download sends the next run to re-fetch
    # 2.6 GB it already has.
    cached = None
    if os.path.exists(npz_path) and os.path.getsize(npz_path) > 0:
        try:
            cached = np.load(npz_path)
        except Exception as exc:                   # noqa: BLE001
            sys.stderr.write(
                "speed_gbdt_arm: %s is unreadable (%s); re-decoding from "
                "the gzip beside it\n" % (npz_path, exc))
            cached = None
    if cached is not None:
        z = cached
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


ISTELLA_URL = "http://library.istella.it/dataset/istella-s-letor.tar.gz"
ISTELLA_FEATURES = 220
ISTELLA_N_TEST = 500000


def _decode_letor(path, n_features):
    """SVMlight/LETOR text ("rel qid:N 1:v ... F:v" per line, every feature
    present and in order) to (x float32 [rows, F], y float32 [rows]). The
    'k:' prefixes are stripped per 64 MB chunk with one regex and the rest
    is one `np.fromstring`, which is what makes a 2M-line file a few
    minutes instead of an hour. Istella marks a missing value with the
    float64 maximum (1.797e308); that is above float32 and would become
    inf, so anything at or above 1e300 is clamped to the float32 maximum:
    still the largest value in its column, so every threshold rule sees
    the same ordering, and finite for every library's binning."""
    import re
    strip = re.compile(rb" \d+:")
    xs, ys = [], []
    tail = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(64 << 20)
            if not block:
                break
            block = tail + block
            cut = block.rfind(b"\n")
            if cut < 0:
                tail = block
                continue
            chunk, tail = block[:cut + 1], block[cut + 1:]
            chunk = strip.sub(b" ", chunk.replace(b" qid:", b" "))
            arr = np.fromstring(chunk.decode("ascii"), sep=" ",
                                dtype=np.float64)
            arr = arr.reshape(-1, n_features + 2)
            ys.append(arr[:, 0].astype(np.float32))
            feats = arr[:, 2:]
            big = feats >= 1e300
            if big.any():
                feats = feats.copy()
                feats[big] = np.finfo(np.float32).max
            xs.append(feats.astype(np.float32))
    if tail.strip():
        raise RuntimeError("%s: trailing partial line" % path)
    return (np.ascontiguousarray(np.concatenate(xs)),
            np.ascontiguousarray(np.concatenate(ys)))


def load_istella(size, rows_cap=None, regression=False):
    """Istella-S LETOR, 3,408,630 x 220, THE HIGH-FEATURE LARGE DATASET
    (ENGINEERING_RULES.md section 9, the second kind beside NYC taxi; HIGGS is retired).

    Real web-search query/document feature vectors from the istella search
    engine (Dato et al., ACM TOIS 2016), dense, 220 numeric features,
    graded relevance 0..4. train.txt is 2,043,304 rows, test.txt 681,250.
    Binary target: relevance > 0 (about 11% positive); `regression=True`
    keeps the 0..4 grade as a float target (the RMSE cell). The test rows
    are the first ISTELLA_N_TEST rows of test.txt at every rung, the train
    rows the first `rows_cap` of train.txt, so rungs are comparable the way
    HIGGS rungs are. Direct download, no credentials (Bosch needs Kaggle).

    THE DOWNLOAD IS 472 MB and is a SEPARATE, EXPLICITLY NAMED STEP
    (`--download istella`); the decode is minutes and is done there too."""
    folder = os.path.join(data_root(), "istella")
    npz_path = os.path.join(folder, "istella_speed.npz")
    cached = None
    if os.path.exists(npz_path) and os.path.getsize(npz_path) > 0:
        try:
            cached = np.load(npz_path)
        except Exception as exc:                   # noqa: BLE001
            sys.stderr.write(
                "speed_gbdt_arm: %s is unreadable (%s); re-decoding from "
                "the text files beside it\n" % (npz_path, exc))
            cached = None
    if cached is not None:
        x_tr, r_tr = cached["x_train"], cached["r_train"]
        x_te, r_te = cached["x_test"], cached["r_test"]
    else:
        train_txt = _find_file(folder, "train.txt")
        test_txt = _find_file(folder, "test.txt")
        if train_txt is None or test_txt is None:
            raise RuntimeError(
                "istella is not downloaded: train.txt/test.txt missing under "
                "%s. Run `python tools/speed_gbdt_arm.py --download istella` "
                "first (472 MB), OUTSIDE the timed run." % folder)
        x_tr, r_tr = _decode_letor(train_txt, ISTELLA_FEATURES)
        x_te, r_te = _decode_letor(test_txt, ISTELLA_FEATURES)
        x_te = np.ascontiguousarray(x_te[:ISTELLA_N_TEST])
        r_te = np.ascontiguousarray(r_te[:ISTELLA_N_TEST])
        np.savez(npz_path, x_train=x_tr, r_train=r_tr, x_test=x_te,
                 r_test=r_te)
    n_train = x_tr.shape[0]
    if size == "smoke":
        rows_cap = min(rows_cap or 50000, 50000)
    if rows_cap:
        n_train = min(n_train, rows_cap)
    x_train = np.ascontiguousarray(x_tr[:n_train])
    r_train = np.ascontiguousarray(r_tr[:n_train])
    if regression:
        return Data("istellareg", x_train, x_te, r_train, r_te,
                    "regression", 0)
    y_train = (r_train > 0).astype(np.float32)
    y_test = (r_te > 0).astype(np.float32)
    return Data("istella", x_train, x_te, y_train, y_test, "binary", 2)


def _find_file(folder, name):
    for root, _dirs, files in os.walk(folder):
        if name in files:
            return os.path.join(root, name)
    return None


TAXI_MONTHS = ("2024-01", "2024-02")
TAXI_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_%s.parquet"
TAXI_N_TEST = 500000
#: The 16 tree features, in column order. Categorical ids stay as their
#: integer codes (the same code for every arm; CatBoost is NOT told they are
#: categorical, per "same everything except GPU"). A missing value is -1.
TAXI_FEATURES = ("vendor", "passengers", "distance_mi", "ratecode",
                 "store_fwd", "pu_zone", "do_zone", "pickup_hour",
                 "pickup_weekday", "pickup_day", "duration_min", "extra",
                 "mta_tax", "tolls", "congestion", "airport_fee")
#: The columns a classical (kNN, k-means, PCA, OLS) lane takes: the numeric
#: ones, no ids.
TAXI_NUMERIC = ("passengers", "distance_mi", "pickup_hour", "pickup_weekday",
                "pickup_day", "duration_min", "extra", "mta_tax", "tolls",
                "congestion", "airport_fee")


def _decode_taxi_month(path):
    """One TLC yellow-taxi parquet month to (features float32 [rows, 16],
    fare float32, tip float32, card bool). Needs pyarrow; only the untimed
    `--download taxi` step calls it."""
    import pyarrow.parquet as pq
    cols = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "passenger_count", "trip_distance", "RatecodeID",
            "store_and_fwd_flag", "PULocationID", "DOLocationID",
            "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount",
            "tolls_amount", "congestion_surcharge", "Airport_fee"]
    t = pq.read_table(path, columns=cols)

    def col(name, fill=-1.0):
        a = t.column(name).to_numpy(zero_copy_only=False)
        if a.dtype.kind in "OU":
            return a
        a = a.astype(np.float64)
        a[~np.isfinite(a)] = fill
        return a

    pu = t.column("tpep_pickup_datetime").to_numpy(zero_copy_only=False)
    do = t.column("tpep_dropoff_datetime").to_numpy(zero_copy_only=False)
    pu_us = pu.astype("datetime64[us]").astype(np.int64)
    do_us = do.astype("datetime64[us]").astype(np.int64)
    duration_min = (do_us - pu_us) / 60e6
    pu_min = pu_us // 60_000_000
    pickup_hour = (pu_min // 60) % 24
    pu_day = pu_min // (60 * 24)
    pickup_weekday = (pu_day + 3) % 7          # 1970-01-01 was a Thursday
    pickup_day = (pu.astype("datetime64[D]").astype(np.int64)
                  - pu.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)) + 1
    flag = t.column("store_and_fwd_flag").to_numpy(zero_copy_only=False)
    store_fwd = np.full(len(flag), -1.0)
    store_fwd[flag == "N"] = 0.0
    store_fwd[flag == "Y"] = 1.0
    fare = col("fare_amount")
    tip = col("tip_amount")
    distance = col("trip_distance")
    payment = col("payment_type")
    feats = np.stack([
        col("VendorID"), col("passenger_count"), distance, col("RatecodeID"),
        store_fwd, col("PULocationID"), col("DOLocationID"),
        pickup_hour.astype(np.float64), pickup_weekday.astype(np.float64),
        pickup_day.astype(np.float64), duration_min, col("extra"),
        col("mta_tax"), col("tolls_amount"), col("congestion_surcharge"),
        col("Airport_fee")], axis=1)
    # Plausible trips only: a positive fare under $500, a positive distance
    # under 100 miles, one minute to three hours. Same filter for every arm.
    keep = ((fare > 0) & (fare <= 500) & (distance > 0) & (distance <= 100)
            & (duration_min >= 1) & (duration_min <= 180))
    return (feats[keep].astype(np.float32), fare[keep].astype(np.float32),
            tip[keep].astype(np.float32), payment[keep] == 1)


def load_taxi(size, rows_cap=None, regression=False):
    """NYC TLC yellow taxi trips, January and February 2024, THE MIXED-TYPE
    LARGE DATASET (ENGINEERING_RULES.md section 9, the second kind beside
    Istella-S).

    About 5.8M plausible trips after the filter in `_decode_taxi_month`,
    16 features of the kind a business table has: categorical ids (vendor,
    rate code, pickup and dropoff zone), small integers (passengers, hour,
    weekday, day), skewed positives (distance, duration, tolls), columns
    that are mostly missing (congestion surcharge, airport fee; -1 marks a
    missing value for every arm). `taxi` is the classification task: on
    card-paid trips (cash tips are not recorded), did the rider tip 20% of
    the fare or more. `taxireg` is fare_amount on every trip. The test rows
    are the LAST TAXI_N_TEST rows (late February) at every rung, the train
    rows the first `rows_cap`, so rungs are comparable and the split is
    the temporal one a real deployment has.

    THE DOWNLOAD IS ABOUT 100 MB of parquet and is a SEPARATE, EXPLICITLY
    NAMED STEP (`--download taxi`, needs pyarrow); timed runs load the
    NumPy cache only."""
    folder = os.path.join(data_root(), "taxi")
    npz_path = os.path.join(folder, "taxi_speed.npz")
    cached = None
    if os.path.exists(npz_path) and os.path.getsize(npz_path) > 0:
        try:
            cached = np.load(npz_path)
        except Exception as exc:                   # noqa: BLE001
            sys.stderr.write(
                "speed_gbdt_arm: %s is unreadable (%s); re-decoding from "
                "the parquet files beside it\n" % (npz_path, exc))
            cached = None
    if cached is not None:
        x, fare, tip, card = (cached["x"], cached["fare"], cached["tip"],
                              cached["card"])
    else:
        paths = [os.path.join(folder, "yellow_tripdata_%s.parquet" % m)
                 for m in TAXI_MONTHS]
        if not all(os.path.isfile(q) for q in paths):
            raise RuntimeError(
                "taxi is not downloaded: %s missing. Run `python "
                "tools/speed_gbdt_arm.py --download taxi` first (about "
                "100 MB, needs pyarrow), OUTSIDE the timed run." % folder)
        parts = [_decode_taxi_month(q) for q in paths]
        x = np.ascontiguousarray(np.concatenate([p[0] for p in parts]))
        fare = np.concatenate([p[1] for p in parts])
        tip = np.concatenate([p[2] for p in parts])
        card = np.concatenate([p[3] for p in parts])
        np.savez(npz_path, x=x, fare=fare, tip=tip, card=card)
    if regression:
        y = fare
        name = "taxireg"
    else:
        x, fare, tip = x[card], fare[card], tip[card]
        y = (tip >= 0.2 * fare).astype(np.float32)
        name = "taxi"
    n_test = TAXI_N_TEST
    n_train = x.shape[0] - n_test
    if size == "smoke":
        rows_cap = min(rows_cap or 50000, 50000)
    if rows_cap:
        n_train = min(n_train, rows_cap)
    x_train = np.ascontiguousarray(x[:n_train])
    y_train = np.ascontiguousarray(y[:n_train])
    x_test = np.ascontiguousarray(x[-n_test:])
    y_test = np.ascontiguousarray(y[-n_test:])
    if regression:
        return Data(name, x_train, x_test, y_train, y_test, "regression", 0)
    return Data(name, x_train, x_test, y_train, y_test, "binary", 2)


#: Which dataset each lane runs by default. Since 2026-09-11 every tree
#: lane defaults to `taxi` (ENGINEERING_RULES.md section 9); a leg runs
#: `taxi` and `istella` both, and HIGGS is retired (its loader stays so old
#: evidence can be re-read). `year` and `covtype` remain as small fixtures.
LANE_DEFAULT_DATASET = {
    "gbdt-symmetric": "taxi",
    "gbdt-depthwise": "taxi",
    "gbdt-lossguide": "taxi",
    "rf": "taxi",
    "et": "taxi",
    "iforest": "anomaly",
}


#: Criteo display-advertising click logs, the CATEGORICAL dataset. Public on
#: HuggingFace, ungated (the API reports gated:false and an unauthenticated
#: range request returns 206), so it needs no credentials -- which is why it is
#: usable where Bosch was not. One `part-*` file is about 97 MB and 1,529,035
#: rows; three reach ~4.6M, past the section 9 million-row floor.
CRITEO_BASE = ("https://huggingface.co/datasets/criteo/CriteoClickLogs/"
               "resolve/main/data/day=2015-02-15/")
CRITEO_PARTS = (
    "part-00015-99c339d5-fbac-4110-9dcf-75453a61a5c1.c000.snappy.parquet",
    "part-00079-99c339d5-fbac-4110-9dcf-75453a61a5c1.c000.snappy.parquet",
    "part-00104-99c339d5-fbac-4110-9dcf-75453a61a5c1.c000.snappy.parquet",
)
CRITEO_N_INT = 13
CRITEO_N_CAT = 26
CRITEO_N_TEST = 500000


def _decode_criteo(paths):
    """Criteo parquet -> (X float32, y float32, cat_idx).

    Layout is `label`, `integer_feature_1..13`, `categorical_feature_1..26`
    (40 columns, verified against the file). The integers keep -1 for missing,
    the same convention `_decode_taxi_month` uses, so one arm cannot read a
    missing value as a real one while another refuses it.

    THE CATEGORY CODES ARE DETERMINISTIC BY CONSTRUCTION, and that is
    load-bearing rather than tidiness. Our surface wants DENSE CODES 0..k-1
    (`ensemble.py`: CatBoost's own dispatch then picks one-hot for a small
    cardinality and target statistics for a large one). Criteo ships 32-bit
    HASHED STRINGS instead, so the codes have to be assigned here -- and if
    they came from dict or hash iteration order, two decodes of the same rows
    would produce different codes, hence different borders, different splits
    and a different model hash. The identity property would break on the one
    dataset whose reason for existing is to exercise the categorical path.
    So: codes are the rank of the string in SORTED UNIQUE order per column,
    computed once over the decoded rows and frozen in the npz beside the
    matrix. NULL is a category of its own (`cat_14`, `cat_16`, `cat_17` and
    `cat_23` are 34.3% null and `cat_1` 3.9%: the missingness clusters, so
    folding it into an arbitrary code would destroy signal).

    Measured on `part-00015` (1,529,035 rows, 3.21% positive): cardinality
    runs from 3 (`cat_6`, `cat_17`) to 371,237 (`cat_20`), 1,697,108 distinct
    over the 26 columns. That spread is the point -- the small columns take
    CatBoost's one-hot branch and the large ones force the CTR branch, so one
    dataset covers both sides of the dispatch."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError(
            "criteo needs pyarrow to decode parquet; it is a download-step "
            "dependency only, never imported inside a timed run")
    tables = [pq.read_table(p) for p in paths]
    int_cols = ["integer_feature_%d" % i for i in range(1, CRITEO_N_INT + 1)]
    cat_cols = ["categorical_feature_%d" % i for i in range(1, CRITEO_N_CAT + 1)]
    y = np.concatenate([
        np.asarray(t.column("label").to_numpy(zero_copy_only=False),
                   dtype=np.float32) for t in tables])
    n = y.shape[0]
    x = np.empty((n, CRITEO_N_INT + CRITEO_N_CAT), dtype=np.float32)
    for j, name in enumerate(int_cols):
        col = np.concatenate([
            np.asarray(t.column(name).to_numpy(zero_copy_only=False),
                       dtype=np.float64) for t in tables])
        # -1 marks missing for every arm, as in taxi
        col = np.where(np.isnan(col), -1.0, col)
        x[:, j] = col.astype(np.float32)
    tables_cat = {}
    for j, name in enumerate(cat_cols):
        parts = []
        for t in tables:
            parts.append(np.asarray(t.column(name).to_pylist(), dtype=object))
        col = np.concatenate(parts)
        col = np.where(col == None, "\x00NULL", col)        # noqa: E711
        col = col.astype(str)
        # sorted unique -> rank. np.unique sorts, so the mapping is a pure
        # function of the value set and never of iteration order.
        uniq, codes = np.unique(col, return_inverse=True)
        x[:, CRITEO_N_INT + j] = codes.astype(np.float32)
        tables_cat[name] = uniq
    cat_idx = tuple(range(CRITEO_N_INT, CRITEO_N_INT + CRITEO_N_CAT))
    return x, y, cat_idx, tables_cat


def load_criteo(size, rows_cap=None):
    """Criteo click logs, 13 integer + 26 categorical, THE CATEGORICAL SET.

    NOT a section 9 gating dataset. taxi and Istella-S remain the two kinds
    every flip verdict is computed over; criteo exists to tune and check the
    categorical and CTR paths, which taxi's low-cardinality ids and
    Istella-S's 220 dense numerics never reach. A win measured here alone is
    a one-kind win and stays opt-in, exactly as the rules say.

    Binary target `label` (about 3.2% positive, so far more skewed than
    taxi's 76% or Istella-S's 11%). The test rows are the LAST
    CRITEO_N_TEST rows at every rung and the train rows the first
    `rows_cap`, so rungs are comparable.

    THE DOWNLOAD IS ABOUT 291 MB of parquet over three parts and is a
    SEPARATE, EXPLICITLY NAMED STEP (`--download criteo`, needs pyarrow);
    timed runs load the NumPy cache only."""
    folder = os.path.join(data_root(), "criteo")
    npz_path = os.path.join(folder, "criteo_speed.npz")
    cached = None
    if os.path.exists(npz_path) and os.path.getsize(npz_path) > 0:
        try:
            cached = np.load(npz_path, allow_pickle=False)
        except Exception as exc:                   # noqa: BLE001
            sys.stderr.write(
                "speed_gbdt_arm: %s is unreadable (%s); re-decoding from the "
                "parquet parts beside it\n" % (npz_path, exc))
            cached = None
    if cached is not None:
        x_all, y_all = cached["x_all"], cached["y_all"]
        cat_idx = tuple(int(v) for v in cached["cat_idx"])
    else:
        paths = [os.path.join(folder, p) for p in CRITEO_PARTS]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise RuntimeError(
                "criteo is not downloaded: %d of %d parquet parts missing "
                "under %s. Run `python tools/speed_gbdt_arm.py --download "
                "criteo` first (about 291 MB), OUTSIDE the timed run."
                % (len(missing), len(paths), folder))
        x_all, y_all, cat_idx, _codes = _decode_criteo(paths)
        os.makedirs(folder, exist_ok=True)
        np.savez(npz_path, x_all=x_all, y_all=y_all,
                 cat_idx=np.asarray(cat_idx, dtype=np.int32))
    n_all = x_all.shape[0]
    n_test = min(CRITEO_N_TEST, max(1, n_all // 5))
    x_te = np.ascontiguousarray(x_all[n_all - n_test:])
    y_te = np.ascontiguousarray(y_all[n_all - n_test:])
    n_train = n_all - n_test
    if size == "smoke":
        rows_cap = min(rows_cap or 50000, 50000)
    if rows_cap:
        n_train = min(n_train, rows_cap)
    x_train = np.ascontiguousarray(x_all[:n_train])
    y_train = np.ascontiguousarray(y_all[:n_train])
    return Data("criteo", x_train, x_te, y_train, y_te, "binary", 2,
                cat_idx=cat_idx)


def load_dataset(name, size, rows_cap=None):
    if name == "criteo":
        return load_criteo(size, rows_cap)
    if name == "higgs":
        return load_higgs(size, rows_cap)
    if name == "higgsreg":
        # HIGGS with its 0/1 label as a float TARGET: the RMSE cell of the
        # boosting lanes on the same bytes as the Logloss cell, so the two
        # objectives (one with a Newton walker, one without) are timed on
        # one fixture. RMSE on a 0/1 target is a legitimate regression
        # (a Brier-style fit); it is not what CatBoost users run on HIGGS.
        d = load_higgs(size, rows_cap)
        return Data("higgsreg", d.X_train, d.X_test, d.y_train, d.y_test,
                    "regression", 0)
    if name == "istella":
        return load_istella(size, rows_cap)
    if name == "istellareg":
        return load_istella(size, rows_cap, regression=True)
    if name == "taxi":
        return load_taxi(size, rows_cap)
    if name == "taxireg":
        return load_taxi(size, rows_cap, regression=True)
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
        if name in ("covtype", "covtype2", "synthclf", "higgs", "istella",
                    "taxi"):
            return load_dataset("synthclf", size, rows_cap)
        if name in ("higgsreg", "istellareg", "taxireg"):
            return load_dataset("synth", size, rows_cap)
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
    if name == "taxi":
        import urllib.request
        folder = os.path.join(data_root(), "taxi")
        os.makedirs(folder, exist_ok=True)
        for m in TAXI_MONTHS:
            dest = os.path.join(folder, "yellow_tripdata_%s.parquet" % m)
            if os.path.isfile(dest):
                print("taxi %s already present (%.1f MB)"
                      % (m, os.path.getsize(dest) / 1e6))
                continue
            print("downloading %s -> %s" % (TAXI_URL % m, dest))
            # An explicit agent: the TLC CloudFront has refused Python-urllib's
            # default one (MI325X trees leg, 2026-09-11). It also refuses
            # DigitalOcean droplets BY ADDRESS whatever the agent (curl with a
            # browser agent got 403 on droplet 599682588, 2026-09-11), so a
            # DigitalOcean leg needs taxi from the mojolearn-data-tor1 volume.
            # Written to .part and renamed, so a failed fetch leaves no
            # truncated file for the "already present" test to accept.
            req = urllib.request.Request(
                TAXI_URL % m,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) mojolearn-bench"})
            part = dest + ".part"
            with urllib.request.urlopen(req) as resp, open(part, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            os.replace(part, dest)
            print("taxi %s: %.1f MB" % (m, os.path.getsize(dest) / 1e6))
        d = load_taxi("shipped")
        print("taxi decoded to %s (train %d x %d, test %d, positives %.3f)"
              % (os.path.join(folder, "taxi_speed.npz"), d.X_train.shape[0],
                 d.X_train.shape[1], d.X_test.shape[0],
                 float(d.y_train.mean())))
        return
    if name == "istella":
        import tarfile
        import urllib.request
        folder = os.path.join(data_root(), "istella")
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, "istella-s-letor.tar.gz")
        if os.path.isfile(dest):
            print("istella already present: %s (%.1f MB)"
                  % (dest, os.path.getsize(dest) / 1e6))
        else:
            print("downloading %s -> %s (about 472 MB)" % (ISTELLA_URL, dest))
            urllib.request.urlretrieve(ISTELLA_URL, dest)
            print("istella: %.1f MB" % (os.path.getsize(dest) / 1e6))
        if _find_file(folder, "train.txt") is None:
            with tarfile.open(dest) as tar:
                tar.extractall(folder)
        d = load_istella("shipped")
        print("istella decoded to %s (train %d x %d, test %d)"
              % (os.path.join(folder, "istella_speed.npz"),
                 d.X_train.shape[0], d.X_train.shape[1], d.X_test.shape[0]))
        return
    if name == "criteo":
        import urllib.request
        folder = os.path.join(data_root(), "criteo")
        os.makedirs(folder, exist_ok=True)
        for part in CRITEO_PARTS:
            dest = os.path.join(folder, part)
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                print("criteo part already present: %s (%.1f MB)"
                      % (part, os.path.getsize(dest) / 1e6))
                continue
            url = CRITEO_BASE + part
            print("downloading %s -> %s (about 97 MB)" % (url, dest))
            # .part then rename, so an interrupted fetch cannot leave a
            # truncated file that the size>0 check above would accept as
            # complete. urlretrieve cannot resume, and a half file that
            # counts as present is how a leg once measured synthetic data
            # while believing it had Istella-S.
            tmp = dest + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
            print("criteo part: %.1f MB" % (os.path.getsize(dest) / 1e6))
        d = load_criteo("shipped")
        print("criteo decoded to %s (train %d x %d, test %d, %d categorical "
              "columns at %s)"
              % (os.path.join(folder, "criteo_speed.npz"),
                 d.X_train.shape[0], d.X_train.shape[1], d.X_test.shape[0],
                 len(d.cat_idx), ",".join(str(i) for i in d.cat_idx[:4]) + ",..."))
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
                         rows), which is not implemented. Pinning it removes the
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

    `make()` returns a fresh unfitted estimator. The shared runner times
    construction, `fit(model, data)` and `sync()` together: constructor work
    and input conversion in fit are included. Synchronization is included
    because a fit is not finished until the device says so, and
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
        # DEVIATION 2634's other side. `cat_features` is what makes CatBoost
        # run the CTR machinery at all: its own dispatch
        # (binarizations_manager.cpp:106-115) takes one-hot below a
        # cardinality threshold and target statistics above it. criteo spans
        # 3 to 371,237 distinct per column, so one dataset exercises both
        # branches. Passing the indices is also what makes the comparison
        # honest: without them CatBoost would treat 26 hashed-id columns as
        # ORDERED NUMBERS and be timed on a different, easier problem than
        # ours.
        if data.cat_idx:
            p["cat_features"] = list(data.cat_idx)
        if data.task == "regression":
            return catboost.CatBoostRegressor(loss_function="RMSE", **p)
        if data.task == "binary":
            return catboost.CatBoostClassifier(loss_function="Logloss", **p)
        return catboost.CatBoostClassifier(loss_function="MultiClass", **p)

    def _cat_frame(x, cat_idx):
        """float32 matrix -> object matrix with the declared columns integral.

        CatBoost refuses float columns named in `cat_features` ("Invalid type
        for cat_feature"), so those columns go in as integers. The codes are
        already dense 0..k-1 integers held exactly in float32 (the largest is
        371,237, well inside float32's 24-bit exact-integer range), so the
        cast moves no value."""
        out = x.astype(object)
        for j in cat_idx:
            out[:, j] = x[:, j].astype(np.int64)
        return out

    def _fit(m, d):
        if not d.cat_idx:
            return m.fit(d.X_train, d.y_train)
        return m.fit(_cat_frame(d.X_train, d.cat_idx), d.y_train)

    def _score(m, d):
        if not d.cat_idx:
            return _score_sklearn_like(m, d)
        # predict MUST see the same column types the fit saw. Without this the
        # scorer would hand raw float32 test rows to a model fitted with
        # cat_features and CatBoost would raise -- a failure that would have
        # looked like "criteo is broken" rather than "the scorer was not
        # wired". Reuse the shared scorer so the metric stays defined once.
        shim = Data(d.name, d.X_train, _cat_frame(d.X_test, d.cat_idx),
                    d.y_train, d.y_test, d.task, d.n_classes,
                    y_anom=d.y_anom, cat_idx=d.cat_idx)
        return _score_sklearn_like(m, shim)

    out = []
    for dev in devices:
        if dev == "opencl":
            continue                   # LightGBM's device name only
        task_type = "CPU" if dev == "cpu" else "GPU"
        out.append(Arm(
            "catboost-" + dev,
            (lambda tt: (lambda: make(tt)))(task_type),
            _fit,
            _score,
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
        # XGBoost's categorical support is opt-in and needs the columns to
        # arrive as pandas `category` dtype; raw integer codes would be split
        # as ORDERED NUMBERS, which is a different (easier) problem than the
        # partition search CatBoost and ours do. Declaring it is what keeps
        # the three arms on the same problem.
        if data.cat_idx:
            p["enable_categorical"] = True
            p["max_cat_to_onehot"] = 1      # force partition search, not one-hot
        if data.task == "regression":
            return xgb.XGBRegressor(objective="reg:squarederror", **p)
        if data.task == "binary":
            return xgb.XGBClassifier(objective="binary:logistic", **p)
        return xgb.XGBClassifier(objective="multi:softprob",
                                 num_class=data.n_classes, **p)

    def _frame(x, cat_idx):
        """float32 matrix -> DataFrame with the declared columns as category.

        Built once per fit and per predict, OUTSIDE the timed region's
        intent... which is exactly why it is measured: the conversion is part
        of what an XGBoost user pays to use categoricals, the same way our
        staging cost is part of ours. It is not hidden from the clock for one
        arm and charged to another."""
        import pandas as pd
        df = pd.DataFrame(x)
        for j in cat_idx:
            df[j] = df[j].astype("int64").astype("category")
        return df

    def _fit(m, d):
        if not d.cat_idx:
            return m.fit(d.X_train, d.y_train)
        return m.fit(_frame(d.X_train, d.cat_idx), d.y_train)

    def _score(m, d):
        if not d.cat_idx:
            return _score_sklearn_like(m, d)
        # predict must see the SAME dtypes the fit saw, or XGBoost raises on
        # the category mismatch. Swap in a framed test matrix and reuse the
        # shared scorer so the metric definition stays in one place.
        shim = Data(d.name, d.X_train, _frame(d.X_test, d.cat_idx),
                    d.y_train, d.y_test, d.task, d.n_classes,
                    y_anom=d.y_anom, cat_idx=d.cat_idx)
        return _score_sklearn_like(m, shim)

    out = []
    for dev in devices:
        if dev == "opencl":
            continue                   # LightGBM's device name only
        # "cuda" is also the device string of AMD's ROCm build (amd_xgboost).
        device = "cpu" if dev == "cpu" else "cuda"
        out.append(Arm(
            "xgboost-" + dev,
            (lambda dv: (lambda: make(dv)))(device),
            _fit,
            _score,
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
    # MOJOLEARN_SPEED_LGBM_PARAMS=name=value,... (lane trees-hotaisle,
    # 2026-09-11): LightGBM 4.7.0 refused every AMD cell with "Check failed:
    # (best_split_info.left_count) > (0)" under min_child_weight=0.0, so ONE
    # retry runs with other params. Unset means the params above, unchanged;
    # set means every LightGBM arm in the process takes them, and the log says so.
    raw = os.environ.get("MOJOLEARN_SPEED_LGBM_PARAMS", "").strip()
    if raw:
        import ast
        over = {}
        for item in raw.split(","):
            key, _, val = item.partition("=")
            over[key.strip()] = ast.literal_eval(val.strip())
        p.update(over)
        emit_note(lane, ["lightgbm-*"], "params", float(len(over)),
                  "MOJOLEARN_SPEED_LGBM_PARAMS overrides the harness LightGBM "
                  "params: %s" % ", ".join("%s=%r" % kv for kv in sorted(over.items())))

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
        # `opencl` is LightGBM's OpenCL learner (device_type 'gpu', a
        # USE_GPU=ON build), the GPU build that can run on an AMD box.
        device_type = {"cpu": "cpu", "opencl": "gpu"}.get(dev, "cuda")
        out.append(Arm(
            "lightgbm-" + {"cpu": "cpu", "opencl": "opencl"}.get(dev, "cuda"),
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
    `ensemble/` is an implementation of. The honest opponent for the `rf` lane.

    `n_streams` is left at cuML's default for the primary competitor baseline.
    A separately labeled one-stream arm can diagnose scheduling effects.
    Neither requires cuML to implement MojoLearn's IDENTICAL contract.
    """
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
    and the benchmark output must say so in the lane's own row."""
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
    this file states that cuML has no IsolationForest. Our own implementation says
    otherwise: `python/mojolearn/_iforest_impl.py` cites
    `isolation_forest.pyx:663-702` for cuML's `max_samples` resolution, and
    `isolation_forest/` is described throughout as an implementation of cuML's. Both
    cannot be right, and the cheap way to settle it is to try the import on
    the box and print what happens. If it is absent the arm REFUSES by name
    and the lane falls back to sklearn alone; if it is present the lane gets
    the GPU-versus-GPU column every other lane has.

    Settling it also closes a real open question the implementation carries: DEVIATION
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
            builders.append((["lightgbm-cpu", "lightgbm-cuda", "lightgbm-opencl"],
                             lambda: lightgbm_arms(lane, cfg, data, devices)))
    elif lane == "rf":
        builders.append((["cuml-rf-gpu"],
                         lambda: [cuml_rf_arm(lane, cfg, data)]))
        builders.append((["sklearn-rf-cpu"],
                         lambda: [sklearn_forest_arm(lane, cfg, data)]))
        builders.append((["lightgbm-cpu", "lightgbm-cuda", "lightgbm-opencl"],
                         lambda: lightgbm_arms(lane, cfg, data, devices)))
    elif lane == "et":
        # Named `cuml-et-gpu` so the refusal reads as "cuML has no ExtraTrees"
        # rather than as a missing RandomForest row.
        builders.append((["cuml-et-gpu"],
                         lambda: [cuml_rf_arm(lane, cfg, data)]))
        builders.append((["sklearn-et-cpu"],
                         lambda: [sklearn_forest_arm(lane, cfg, data)]))
        builders.append((["lightgbm-cpu", "lightgbm-cuda", "lightgbm-opencl"],
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


def accel_vendor():
    """'nvidia', 'amd' or None: which GPU vendor's box this is. NVIDIA wins
    when both tools are present (a CUDA image with rocm-smi installed is
    still an NVIDIA box)."""
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocm-smi") or os.path.exists("/dev/kfd"):
        return "amd"
    return None


def resolve_devices(requested, lane=None):
    """Which device arms of each opponent may run here.

    CORRECTED 2026-09-11 (ENGINEERING_RULES.md section 10): on AMD a library
    with no AMD GPU path runs on the box's CPU on all cores when `cpu` is
    requested by name, and every arm name says -cpu or -gpu; the GPU-only
    rule below now binds NVIDIA only.

    THE RULE, AND IT IS NOT A PREFERENCE. On NVIDIA we compare
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
    CORRECTED 2026-09-11: `et` on NVIDIA with `cpu` requested by name runs
    scikit-learn's ExtraTrees on the pod CPU, labeled CPU (lane trees-taxi-h100).
    """
    want = [d.strip().lower() for d in (requested or "").split(",")
            if d.strip()]
    if not want or want == ["auto"]:
        want = ["gpu"] if accel_visible() else ["cpu"]
        auto = True
    else:
        auto = False
    if "cpu" in want and accel_visible() and accel_vendor() == "amd":
        if lane is not None:
            emit_note(
                lane, want, "devices", float(len(want)),
                "AMD box: CPU arms run for libraries with no AMD GPU path, "
                "on all cores (ENGINEERING_RULES.md section 10); every arm "
                "name carries its device")
    elif "cpu" in want and accel_visible() and lane == "et" and not auto:
        # CORRECTED 2026-09-11 (lane trees-taxi-h100): NVIDIA has no GPU
        # ExtraTrees, so an explicit `cpu` admits scikit-learn's
        # ExtraTreesClassifier on the pod CPU for `et` only, labeled -cpu.
        if lane is not None:
            emit_note(
                lane, want, "devices", float(len(want)),
                "NVIDIA box, lane et: no GPU ExtraTrees exists, so the CPU "
                "arm requested by name runs on all cores and is labeled CPU")
    elif "cpu" in want and accel_visible():
        dropped = [d for d in want if d == "cpu"]
        want = [d for d in want if d != "cpu"]
        if lane is not None and dropped:
            emit_refused(
                lane, "*-cpu",
                "GPU-PATH-ONLY: an accelerator is visible on this box, so the "
                "vendors' CPU arms do not run. On NVIDIA we compare against "
                "the vendor's GPU arm only; the CPU arm is the MacBook's. An "
                "explicit cpu is admitted on NVIDIA for lane et only.")
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
                         "GPU-PATH-ONLY: %s is a CPU arm and cpu was not "
                         "requested on this accelerator box. On NVIDIA we "
                         "compare against the vendor's GPU path only; on AMD "
                         "request cpu by name (ENGINEERING_RULES.md section "
                         "10)." % name)
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

def prepare_anomaly_labels(lane, data):
    """The iforest lane on a labeled dataset (taxi, Istella-S; ENGINEERING_RULES.md
    section 9, lane forest-speed 2026-09-11). Neither has planted anomalies, so
    the AUC column is a PROXY: the anomaly score against the training set's
    minority class on the test rows (taxi: no 20% tip, about 24%; Istella-S:
    relevance above 0, about 11%). It checks that two forests rank the same
    rows alike; it is not a detection-quality claim. The fixture lane keeps
    its planted labels."""
    if lane != "iforest" or getattr(data, "y_anom", None) is not None:
        return
    if data.task != "binary":
        raise SystemExit("iforest on %r needs a binary dataset for the proxy "
                         "AUC; use taxi, istella or anomaly" % data.name)
    positive_share = float(np.mean(np.asarray(data.y_train) > 0.5))
    minority = 1.0 if positive_share <= 0.5 else 0.0
    data.y_anom = (np.asarray(data.y_test) > 0.5).astype(np.float32)
    if minority == 0.0:
        data.y_anom = (1.0 - data.y_anom).astype(np.float32)
    emit_note(lane, ["*"], "auc", positive_share,
              "proxy AUC: anomaly score against the minority class (label %d) "
              "of %s, no planted anomalies" % (int(minority), data.name))


def dataset_scale(data):
    """Visible workload heuristic, not a performance acceptance criterion."""
    rows, features = data.X_train.shape
    input_bytes = data.X_train.nbytes
    return dict(rows=int(rows), features=int(features), input_bytes=int(input_bytes),
                large_candidate=bool(rows >= 1_000_000 or input_bytes >= 256 * 1024**2))


def emit_scale_reminder(data, stage):
    scale = dataset_scale(data)
    print("FSPEED-SCALE stage=%s rows=%d features=%d input_mib=%.1f large_candidate=%s"
          % (stage, scale['rows'], scale['features'], scale['input_bytes'] / 1024**2,
             str(scale['large_candidate']).lower()))
    print("FSPEED-REMINDER stage=%s %s"
          % (stage, "Optimize for large datasets; confirm representative feature/class counts, "
             "tree depth and memory pressure before making speed/default decisions."
             if scale['large_candidate'] else
             "SMALL WORKLOAD: useful for correctness/smoke diagnostics; do not use this "
             "run alone for training-speed claims or optimization/default decisions. "
             "Also test representative large datasets (taxi and istella at 1M rows or more)."))


def run(lane, arms, data, n_rounds, size, dev=None, *, rotate_order=False, fit_context=None):
    """One untimed warm-up per arm, then `n_rounds` timed rounds in which
    every surviving arm takes one turn before any arm takes its second.
    Optional rotate_order advances the first arm each round to distribute
    position effects; it does not remove noise or establish timing stability.
    fit_context optionally returns a context manager for (arm_name, round),
    with round 0 for warm-up. It surrounds constructor/fit/sync, excluding
    scoring; entry/exit are outside the timer. Use for diagnostic traces only.

    THE ALTERNATION IS THE POINT AND NOT A STYLE CHOICE. A rented box may
    throttle mid-run. Blocks give you the first arm's cold clocks against the
    last arm's hot ones and no way to tell; alternation spreads the drift
    across every arm, and a ratio survives what an absolute number does not.

    A per-arm wall budget and a whole-process deadline keep a slow CPU arm
    from eating the lease that the GPU arms are the point of. An arm that
    exceeds either is dropped from the rotation with a refusal carrying the
    reason, so the table says what happened."""
    emit_scale_reminder(data, "before")
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
            with (fit_context(arm.name, 0) if fit_context is not None
                  else contextlib.nullcontext()):
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
        ordered = list(live)
        if rotate_order and ordered:
            offset = (r - 1) % len(ordered)
            ordered = ordered[offset:] + ordered[:offset]
        print("FSPEED-ORDER lane=%s round=%d arms=%s"
              % (lane, r, ",".join(arm.name for arm in ordered)))
        for arm in ordered:
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
                with (fit_context(arm.name, r) if fit_context is not None
                      else contextlib.nullcontext()):
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
    emit_scale_reminder(data, "after")
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
                   help="taxi, taxireg, istella, istellareg, higgs (retired), year, covtype, covtype2, synth, synthclf, "
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
