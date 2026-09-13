#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""identity_break: TRY TO BREAK cross-vendor bit-identity, on every estimator.

`tools/repeat_run_stability.py` asks one lane one question on one benign
fixture. This asks every public estimator EIGHT hostile questions, under the
`identical` tier, and writes a per-cell fingerprint so three vendors' runs
can be diffed cell by cell:

    ties      integer-grid data, every distance and split value tied many ways
    hashed    values derived from a hash, no structure, no ties, no symmetry
    wide      column magnitudes spanning 1e-4 .. 1e4 (accumulation order bites)
    denormal  a slice of the data sits below float32 normal (flush-to-zero)
    denormal_ftz  the same with subnormals flushed: the diagnostic twin
    dupes     duplicated rows, a constant column, an all-zero column
    odd       n = 12345, d = 17 -- nothing a block, warp or tile divides evenly
    negative  every value negative, shifted off the origin

    plus `base`, the stability harness's own fixture, as the control.

Every cell is fitted TWICE in one process so a run-to-run mover on this box
is separated from a cross-vendor divergence. A lane that raises reports
REFUSED with the message; nothing is swallowed.

THREE COLUMNS PER CELL, each supporting one public claim (2026-09-13). Every
column is computed on BOTH fits of the cell, so a MOVED column is this box
disagreeing with itself and a DIVERGENT column is two vendors disagreeing.

    train   the cell hash as it has always been (JSON keys `hashes`, `parts`,
            `verdict`), the just-fitted model read back on mostly its own
            training rows and, for the linear and decomposition lanes, its
            fitted state. It supports TRAINING identity, the same fit on
            every vendor. Its bytes and its keys are unchanged, so an old
            JSON diffs against a new one on this column.
    infer   sha256 of the fitted model's outputs on HELD-OUT rows the model
            never saw, the same fixture kind drawn from seed 1 (`odd` keeps
            its odd n and d), through the same public methods the train
            column hashes (predict, predict_proba, transform,
            decision_function, score_samples, kneighbors). It supports
            INFERENCE identity, the same predictions on unseen rows. A lane
            whose estimator has no out-of-sample method records
            `n/a:<reason>` and never a hash of training-row output.
            transductive means the labels belong to the fitted rows only
            (DBSCAN, agglomerative, spectral); no-predict means KMeans has
            no predict or transform; forecast means Holt-Winters takes no
            new rows; function means the lane is not an estimator (linalg,
            metrics).
    model   sha256 of the bytes `save(path)` wrote, for every estimator that
            has `save` and `load` (the random forests, the extra trees and
            every GradientBoosting lane). It supports ARTIFACT identity, the
            same model file on every vendor. The saved file is then loaded
            back and asked for the held-out rows again; if that hash differs
            from `infer` the column is RELOAD-MOVED, a file that does not
            predict what the model in memory predicts. Estimators without
            save/load record `n/a:no-save`.

    MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py --json apple.json
    python3 tools/identity_break.py --diff apple.json nvidia.json amd.json

A `--diff` prints, per lane, per fixture: IDENTICAL (all columns agree),
DIVERGENT (they do not), MOVED (a column disagreed with itself), or the
refusal, first for the train column and then in a second table for the
infer and model columns, where RELOAD-MOVED is also possible and a JSON
that predates the columns reads NOT-COMPARED, never DIVERGENT. It exits
non-zero on any DIVERGENT, MOVED or RELOAD-MOVED cell. FAST is refused on
purpose: a bitwise question to a FAST arm is a category error
(fast-is-not-identical).
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import traceback

import numpy as np


def _h(*arrays):
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode())
        m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def _hfile(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# ---------------------------------------------------------------- fixtures
# Small on purpose: the Mac side of this runs under the no-heavy-local-compute
# rule and the point is the ARITHMETIC PATH, which n=20000 already exercises
# multi-block. Sizes are fixed so a hash means the same thing on every box.

N, D = 20000, 16

#: the seed of the held-out draw; the training draw is seed 0
HELDOUT_SEED = 1


def _hashed_uniform(n, d, seed):
    """Values from sha256 of the index, so no RNG-family assumption and no
    repeated structure; per-cell distinct with overwhelming probability."""
    out = np.empty(n * d, dtype=np.float32)
    ctr = np.arange(n * d, dtype=np.uint64)
    # vectorised: hash 8-byte blocks in chunks
    raw = ctr.tobytes()
    dig = hashlib.sha256(raw + str(seed).encode()).digest()
    # expand deterministically with a counter-mode stream
    stream = bytearray()
    blk = 0
    while len(stream) < n * d * 4:
        stream += hashlib.sha256(dig + blk.to_bytes(8, "little")).digest()
        blk += 1
    u32 = np.frombuffer(bytes(stream[: n * d * 4]), dtype=np.uint32)
    out[:] = (u32.astype(np.float64) / 2**32).astype(np.float32)
    return out.reshape(n, d)


def fixture(kind, n=N, d=D, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "base":
        X = rng.standard_normal((n, d)).astype(np.float32)
    elif kind == "ties":
        X = rng.integers(0, 6, size=(n, d)).astype(np.float32)
    elif kind == "hashed":
        X = (_hashed_uniform(n, d, seed) * 4.0 - 2.0).astype(np.float32)
    elif kind == "wide":
        X = rng.standard_normal((n, d)).astype(np.float32)
        scale = np.logspace(-4, 4, d).astype(np.float32)
        X = (X * scale).astype(np.float32)
    elif kind == "denormal":
        X = rng.standard_normal((n, d)).astype(np.float32)
        # a quarter of the rows, three columns, pushed into the subnormal range
        X[: n // 4, :3] = (X[: n // 4, :3] * np.float32(1e-40)).astype(np.float32)
        assert np.any((X != 0) & (np.abs(X) < np.finfo(np.float32).tiny))
    elif kind == "denormal_ftz":
        # THE DIAGNOSTIC TWIN of `denormal`: the same bytes with every
        # subnormal replaced by a signed zero, which is what a flush-to-zero
        # backend sees. A vendor whose `denormal` cell equals another vendor's
        # `denormal_ftz` cell is flushing where the other is not.
        X, _, _ = fixture("denormal", n, d, seed)
        sub = (X != 0) & (np.abs(X) < np.finfo(np.float32).tiny)
        X = X.copy()
        X[sub] = np.copysign(np.float32(0.0), X[sub])
    elif kind == "dupes":
        X = rng.standard_normal((n, d)).astype(np.float32)
        X[n // 2:] = X[: n - n // 2]          # second half duplicates the first
        X[:, d - 2] = np.float32(3.5)         # a constant column
        X[:, d - 1] = np.float32(0.0)         # an all-zero column
    elif kind == "odd":
        n, d = 12345, 17
        X = rng.standard_normal((n, d)).astype(np.float32)
    elif kind == "negative":
        X = (-np.abs(rng.standard_normal((n, d))) - 2.0).astype(np.float32)
    else:
        raise ValueError(kind)
    return (X,) + labels_for(X, seed)


def heldout(kind):
    """Rows the model never saw: the same fixture kind from HELDOUT_SEED, so
    it has the same shape, scale and pathology as the training draw (`odd`
    keeps its odd n and d) and no row in common with it. Only X is handed to
    the lanes; the held-out labels are not part of any column."""
    X, _, _ = fixture(kind, seed=HELDOUT_SEED)
    return X


def labels_for(X, seed=0):
    """The fixture targets for any X (shared with
    `checks/gbdt_sub_byte_identity_check.py`, so its integer-grid fixtures
    carry labels by the same rule, byte for byte)."""
    # targets: a signed rule on two columns, and a linear regression target
    # Labels come from columns 3 and 4, which NO fixture perturbs (denormal
    # rewrites columns 0-2), so `denormal` and `denormal_ftz` hand every lane
    # the same labels and their twin comparison is about the features alone.
    # Until 2026-08-29 this read columns 0 and 1 and the twins carried 2493
    # different labels, which made the classifier twins uncomparable.
    s01 = X[:, 3] + 0.5 * X[:, 4]
    y_clf = (s01 > np.median(s01)).astype(np.int32)
    w = np.random.default_rng(seed + 1).standard_normal(X.shape[1]).astype(np.float32)
    # FIXED-ORDER, ELEMENTWISE, NO BLAS. `X @ w` in float32 goes through the
    # host BLAS (Accelerate on the Mac, OpenBLAS on a Linux box) and the
    # accumulation order differs between them: measured 2026-08-29, 13876 of
    # 20000 targets differed on ONE Mac between `X @ w` and this loop, and
    # every regression lane read DIVERGENT Apple-vs-AMD for that reason alone.
    # A cross-vendor probe must hand every vendor the same bytes.
    y_reg = np.zeros(X.shape[0], dtype=np.float32)
    for j in range(X.shape[1]):
        y_reg = (y_reg + X[:, j] * w[j]).astype(np.float32)
    return y_clf, y_reg


FIXTURES = ["base", "ties", "hashed", "wide", "denormal", "denormal_ftz", "dupes", "odd", "negative"]

LANES = {}


def lane(name):
    def deco(fn):
        LANES[name] = fn
        return fn
    return deco


class Fit(dict):
    """A lane's answer. It IS the training-row parts dict, hashed exactly as
    it has always been (the train column), and it carries the fitted
    estimator and the held-out probe as attributes so the SAME fit answers
    the infer and model columns. `probe` is either a callable taking an
    estimator and returning the arrays to hash on the held-out rows, or an
    `n/a:<reason>` string for an estimator with no out-of-sample method.
    `checks/gbdt_sub_byte_identity_check.py` calls a lane with four
    arguments and reads only the dict; that contract is unchanged."""
    est = None
    probe = "n/a:function"


def _fit(parts, est=None, probe="n/a:function"):
    f = Fit(parts)
    f.est = est
    f.probe = probe
    return f


# ---------------------------------------------------------------- lanes
# One per PUBLIC estimator, plus linalg and metrics. Each returns a hash of
# the things a user would read back, plus the fitted estimator and what to
# ask it on the held-out rows `Xh`. The held-out slice sizes mirror the
# training-row slice sizes the train column already uses, so the infer
# column costs what the train column costs.

@lane("rf-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("rf-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("et-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("et-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.ExtraTreesRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-symmetric")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss").fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-depthwise")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, grow_policy="Depthwise",
                            loss="Logloss").fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-lossguide")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GradientBoosting(n_estimators=20, max_leaves=32, grow_policy="Lossguide",
                            loss="Logloss").fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-rmse")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="RMSE").fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("kmeans")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KMeans(n_clusters=8, random_state=3).fit(X)
    # mojolearn's KMeans has fit and fit_predict only, no predict or transform
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_)), m, "n/a:no-predict")


@lane("knn")
def _(ml, X, yc, yr, Xh=None):
    m = ml.NearestNeighbors(n_neighbors=8).fit(X[:4096])
    d, i = m.kneighbors(X[4096:4160])
    return _fit(dict(dist=_h(d), idx=_h(i)), m, lambda e: e.kneighbors(Xh[:64]))


@lane("knn-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KNeighborsClassifier(n_neighbors=8).fit(X[:4096], yc[:4096])
    return _fit(dict(predict=_h(m.predict(X[4096:4160])), proba=_h(m.predict_proba(X[4096:4160]))),
                m, lambda e: (e.predict(Xh[:64]), e.predict_proba(Xh[:64])))


@lane("knn-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KNeighborsRegressor(n_neighbors=8).fit(X[:4096], yr[:4096])
    return _fit(dict(predict=_h(m.predict(X[4096:4160]))), m, lambda e: (e.predict(Xh[:64]),))


@lane("dbscan")
def _(ml, X, yc, yr, Xh=None):
    m = ml.DBSCAN(eps=0.9, min_samples=5).fit(X[:6000, :4])
    return _fit(dict(labels=_h(m.labels_)), m, "n/a:transductive")


@lane("pca")
def _(ml, X, yc, yr, Xh=None):
    m = ml.PCA(n_components=4).fit(X)
    return _fit(dict(components=_h(m.components_), variance=_h(m.explained_variance_), transform=_h(m.transform(X[:256]))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("tsvd")
def _(ml, X, yc, yr, Xh=None):
    m = ml.TruncatedSVD(n_components=4).fit(X)
    return _fit(dict(components=_h(m.components_), transform=_h(m.transform(X[:256]))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("ols")
def _(ml, X, yc, yr, Xh=None):
    m = ml.LinearRegression().fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("ridge")
def _(ml, X, yc, yr, Xh=None):
    m = ml.Ridge(alpha=1.0).fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("logistic")
def _(ml, X, yc, yr, Xh=None):
    m = ml.LogisticRegression(max_iter=50).fit(X, yc)
    return _fit(dict(coef=_h(m.coef_), proba=_h(m.predict_proba(X[:256]))), m, lambda e: (e.predict_proba(Xh[:256]),))


@lane("lasso")
def _(ml, X, yc, yr, Xh=None):
    m = ml.Lasso(alpha=0.01, max_iter=200).fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("elasticnet")
def _(ml, X, yc, yr, Xh=None):
    m = ml.ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=200).fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("svc")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SVC(C=1.0, kernel="rbf", max_iter=200).fit(X[:2000], yc[:2000])
    return _fit(dict(decision=_h(m.decision_function(X[2000:2256])), predict=_h(m.predict(X[2000:2256]))),
                m, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("kde")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KernelDensity(bandwidth=0.7).fit(X[:4096, :4])
    return _fit(dict(scores=_h(m.score_samples(X[4096:4352, :4]))), m, lambda e: (e.score_samples(Xh[:256, :4]),))


@lane("agglomerative")
def _(ml, X, yc, yr, Xh=None):
    m = ml.AgglomerativeClustering(n_clusters=4).fit(X[:2000, :4])
    return _fit(dict(labels=_h(m.labels_)), m, "n/a:transductive")


@lane("spectral")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SpectralClustering(n_clusters=4, random_state=3).fit(X[:2000, :4])
    # SpectralClustering.predict raises NotImplementedError by design
    return _fit(dict(labels=_h(m.labels_)), m, "n/a:transductive")


@lane("holtwinters")
def _(ml, X, yc, yr, Xh=None):
    series = (np.cumsum(X[:512, 0]) + 50.0).astype(np.float32)
    series = series - series.min() + 1.0     # positive, for the multiplicative path
    m = ml.ExponentialSmoothing(series, seasonal="additive", seasonal_periods=12).fit()
    # the forecast is already the train column; there are no new rows to feed
    return _fit(dict(forecast=_h(m.forecast(24))), m, "n/a:forecast")


@lane("gemm-pinned")
def _(ml, X, yc, yr, Xh=None):
    a = X[:256].T.copy()                     # 16 x 256 or 17 x 256
    b = X[256:256 + 128].copy()              # 128 x d
    wide = np.ascontiguousarray(X[:4096, :].reshape(-1)[: 256 * 4096].reshape(256, 4096)) \
        if X.size >= 256 * 4096 else X[:256, :]
    c1 = ml.linalg.matmul(a.astype(np.float32), X[:256].astype(np.float32), identical=True)
    c2 = ml.linalg.matmul(wide.astype(np.float32),
                          np.ascontiguousarray(wide[:128].T).astype(np.float32),
                          identical=True) if wide.shape[1] == 4096 else c1
    return _fit(dict(small=_h(c1), wide=_h(c2)))


@lane("metrics")
def _(ml, X, yc, yr, Xh=None):
    # labels_ is a mojolearn Array since the NumPy-free change; the modulo
    # below and the metrics take ndarrays, so view it once.
    labels = np.asarray(ml.KMeans(n_clusters=4, random_state=3).fit(X[:3000, :4]).labels_)
    mt = ml.metrics
    return _fit(dict(
        accuracy=_h(np.float64(mt.accuracy_score(yc[:3000], (labels % 2).astype(np.int32)))),
        ari=_h(np.float64(mt.adjusted_rand_score(yc[:3000], labels))),
        vmeasure=_h(np.float64(mt.v_measure_score(yc[:3000], labels))),
        r2=_h(np.float64(mt.r2_score(yr[:3000], yr[:3000] * np.float32(0.9)))),
        silhouette=_h(np.float64(mt.silhouette_score(X[:3000, :4], labels))),
    ))


# LAST ON PURPOSE (2026-08-29): on a RunPod RTX 4090 the isolation forest
# binding hung at its first fit in every tier (a second DeviceContext beside
# the caller's deadlocked at teardown on sm_89; fixed by DEVIATION 1944, the
# estimator now uses the caller's context), and a SECOND 4090 defect made the
# next GPU call after one RandomForest.fit hang -- DEVIATION 1946, the same
# class one call later: the binding's context was destroyed at its last use
# and its own buffers were freed behind it. 1946 is FIXED IN SOURCE AND UNRUN
# ON A 4090 (see bench/results/identity_break/RESULTS.md, RUN OWED). A hang
# cannot be caught from inside this process, so the JSON is written after
# every lane, the caller wraps the run in `timeout`, and the historically
# hanging lane runs last; a killed run still leaves every earlier lane on
# disk.
@lane("iforest")
def _(ml, X, yc, yr, Xh=None):
    m = ml.IsolationForest(n_estimators=16, random_state=5).fit(X)
    return _fit(dict(scores=_h(m.score_samples(X)), predict=_h(m.predict(X[:512]))),
                m, lambda e: (e.score_samples(Xh), e.predict(Xh[:512])))


# ---------------------------------------------------------------- run / diff

def _has_save_load(est):
    return est is not None and callable(getattr(est, "save", None)) \
        and callable(getattr(type(est), "load", None))


def _probe_fit(fit, name):
    """The infer and model columns of ONE fit. Returns (infer, model, reload,
    error) where infer and model are a hash or an `n/a:<reason>` string,
    reload is the held-out hash of the loaded-back file or None, and error is
    the exception text of whichever stage raised (with the stage name). A
    stage that raised leaves its column None, which reads REFUSED."""
    if not callable(fit.probe):
        return fit.probe, "n/a:no-save", None, None
    try:
        infer = _h(*fit.probe(fit.est))
    except Exception as exc:
        return None, None, None, f"infer: {type(exc).__name__}: {exc}"
    if not _has_save_load(fit.est):
        return infer, "n/a:no-save", None, None
    try:
        with tempfile.TemporaryDirectory(prefix="identity_break_") as tmp:
            path = os.path.join(tmp, f"{name}.npz")
            fit.est.save(path)
            model = _hfile(path)
            back = type(fit.est).load(path)
            reload = _h(*fit.probe(back))
    except Exception as exc:
        return infer, None, None, f"model: {type(exc).__name__}: {exc}"
    return infer, model, reload, None


def _column_verdict(values, reloads=None, infers=None):
    """STABLE, MOVED, RELOAD-MOVED, N/A or REFUSED for one column over the
    repeats of a cell. `values` holds one entry per repeat, a hash or an
    `n/a:` string, or None where that repeat's stage raised. `reloads` and
    `infers` are passed for the model column only; a repeat whose loaded
    file predicts differently from the model in memory is RELOAD-MOVED."""
    if any(v is None for v in values):
        return "REFUSED"
    if all(isinstance(v, str) and v.startswith("n/a") for v in values):
        return "N/A"
    if reloads is not None and any(r != i for r, i in zip(reloads, infers)):
        return "RELOAD-MOVED"
    return "STABLE" if len(set(values)) == 1 else "MOVED"


def _shown(verdict, values):
    if verdict in ("STABLE", "N/A"):
        return values[0]
    if verdict == "MOVED":
        return "MOVED " + values[0][:8]
    return verdict


def run(args):
    import mojolearn as ml
    mode = ml.numeric_mode()
    want = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower() or "fast"
    if mode != want:
        raise SystemExit(f"REFUSING TO MEASURE: asked for {want!r}, loaded {mode!r}")
    if mode == "fast" and not args.allow_fast:
        raise SystemExit("REFUSING: this is a bitwise question and FAST makes no "
                         "bitwise promise; use MOJOLEARN_NUMERIC_MODE=identical")

    lanes = [n for n in LANES if not args.lanes or n in args.lanes.split(",")]
    skip = set(x for x in args.skip.split(",") if x)
    lanes = [n for n in lanes if n not in skip]
    if skip:
        print(f"# SKIPPED on request: {sorted(skip)}")
    fixtures = [f for f in FIXTURES if not args.fixtures or f in args.fixtures.split(",")]
    data = {f: fixture(f) for f in fixtures}
    held = {f: heldout(f) for f in fixtures}
    # THE FIXTURE IS PART OF THE RESULT. Every vendor must be handed the same
    # bytes, and --diff refuses to compare columns whose fixtures differ. The
    # held-out draw is recorded under its own key so a JSON that predates it
    # still matches on `fixtures`.
    fixture_hashes = {f: dict(X=_h(X), y_clf=_h(yc), y_reg=_h(yr))
                      for f, (X, yc, yr) in data.items()}
    heldout_hashes = {f: dict(X=_h(X)) for f, X in held.items()}
    cells = {}

    def dump(complete):
        with open(args.json, "w") as fh:
            json.dump(dict(mode=mode, repeats=args.repeats, platform=platform.platform(),
                           vendor=args.vendor, commit=os.environ.get("MOJOLEARN_COMMIT", ""),
                           heldout_seed=HELDOUT_SEED, fixtures=fixture_hashes,
                           heldout=heldout_hashes, cells=cells, complete=complete,
                           skipped=sorted(skip)), fh, indent=1)

    print(f"# identity_break  mode={mode}  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"{platform.platform()}")
    print("# per lane: the train row, then an `infer` row (held-out rows) and a `model` row "
          "(saved bytes) where the lane has them")
    W = 22
    print(f"| {'lane':<{W}} | " + " | ".join(f"{f:<16}" for f in fixtures) + " |")
    print(f"|{'-'*(W+2)}|" + "|".join("-" * 18 for _ in fixtures) + "|")
    na = {}
    for name in lanes:
        row, row_infer, row_model = [], [], []
        for f in fixtures:
            X, yc, yr = data[f]
            hs, parts, err = [], [], None
            infers, models, reloads, errs2 = [], [], [], []
            for _ in range(args.repeats):
                try:
                    # each fit gets its own held-out bytes, so a lane cannot
                    # hand the next fit rows it wrote to
                    p = LANES[name](ml, X, yc, yr, held[f].copy())
                    parts.append(dict(p))
                    hs.append(_h(np.frombuffer("|".join(f"{k}={v}" for k, v in sorted(p.items())).encode(), dtype=np.uint8)))
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    if args.verbose:
                        traceback.print_exc()
                    break
                inf, mod, rel, err2 = _probe_fit(p, name)
                infers.append(inf); models.append(mod); reloads.append(rel)
                if err2:
                    errs2.append(err2[:300])
                    if args.verbose:
                        traceback.print_exc()
            if err:
                cell = dict(verdict="REFUSED", error=err[:300], hashes=hs, parts=parts)
                shown = "REFUSED"
            elif len(set(hs)) == 1:
                cell = dict(verdict="STABLE", hashes=hs, parts=parts)
                shown = hs[0]
            else:
                cell = dict(verdict="MOVED", hashes=hs, parts=parts)
                shown = "MOVED " + hs[0][:8]
            # the two new columns; a REFUSED train column has no fit to probe
            if not err:
                iv = _column_verdict(infers)
                has_reload = all(r is not None for r in reloads)
                mv = _column_verdict(models, reloads if has_reload else None, infers)
                cell.update(infer=infers, infer_verdict=iv, model=models, model_verdict=mv)
                if any(r is not None for r in reloads):
                    cell["reload"] = reloads
                if errs2:
                    cell["probe_error"] = errs2[0]
                row_infer.append(_shown(iv, infers))
                row_model.append(_shown(mv, models))
                if iv == "N/A":
                    na.setdefault(f"{name} infer", infers[0])
                if mv == "N/A":
                    na.setdefault(f"{name} model", models[0])
            else:
                row_infer.append("REFUSED"); row_model.append("REFUSED")
            cells[f"{name}/{f}"] = cell
            row.append(f"{shown:<16}")
        print(f"| {name:<{W}} | " + " | ".join(row) + " |", flush=True)
        for label, r in (("infer", row_infer), ("model", row_model)):
            if any(not s.startswith("n/a") for s in r):
                print(f"| {name + ' ' + label:<{W}} | " + " | ".join(f"{s:<16}" for s in r) + " |", flush=True)
        if args.json:
            # written after EVERY lane so a hang that gets killed by the
            # caller's timeout still leaves the finished lanes on disk
            dump(False)

    refused = {k: v["error"] for k, v in cells.items() if v["verdict"] == "REFUSED"}
    moved = [k for k, v in cells.items() if v["verdict"] == "MOVED"]
    moved2 = [k for k, v in cells.items()
              if v.get("infer_verdict") == "MOVED" or v.get("model_verdict") in ("MOVED", "RELOAD-MOVED")]
    refused2 = {k: v["probe_error"] for k, v in cells.items() if v.get("probe_error")}
    print()
    print(f"cells={len(cells)} stable={len(cells)-len(refused)-len(moved)} "
          f"moved={len(moved)} refused={len(refused)}")
    for col in ("infer", "model"):
        vs = [v.get(f"{col}_verdict") for v in cells.values() if v.get(f"{col}_verdict")]
        print(f"{col}: " + " ".join(f"{k.lower()}={vs.count(k)}" for k in
                                    ("STABLE", "MOVED", "RELOAD-MOVED", "REFUSED", "N/A") if vs.count(k)))
    if na:
        print("n/a: " + ", ".join(f"{k} {v}" for k, v in sorted(na.items())))
    for k in moved:
        print(f"MOVED   {k}: {cells[k]['hashes']}")
    for k in moved2:
        c = cells[k]
        print(f"MOVED   {k}: infer={c['infer_verdict']} {c['infer']} model={c['model_verdict']} {c['model']}"
              + (f" reload={c['reload']}" if c.get("reload") else ""))
    for k, e in refused.items():
        print(f"REFUSED {k}: {e}")
    for k, e in refused2.items():
        print(f"REFUSED {k}: {e}")
    if args.json:
        dump(True)
        print(f"wrote {args.json}")
    return 1 if (moved or moved2) else 0


def _diff_column(cols, k, col):
    """One row of the second table: the per-column shown values and the
    verdict for `col` (infer or model) of cell `k`. A column that lacks the
    key is `(no column)` and is never compared."""
    shown, hashes, missing, na = [], [], False, False
    for _, j in cols:
        c = j["cells"].get(k)
        if c is None:
            shown.append("(not run)"); continue
        if c.get("verdict") == "REFUSED":
            shown.append("REFUSED"); continue
        if f"{col}_verdict" not in c:
            shown.append("(no column)"); missing = True; continue
        v = c[f"{col}_verdict"]
        if v == "N/A":
            shown.append(c[col][0]); na = True; continue
        if v in ("MOVED", "RELOAD-MOVED", "REFUSED"):
            shown.append(v); hashes.append(v); continue
        shown.append(c[col][0]); hashes.append(c[col][0])
    real = [h for h in hashes if h not in ("MOVED", "RELOAD-MOVED", "REFUSED")]
    if "RELOAD-MOVED" in hashes:
        verdict = "RELOAD-MOVED"
    elif "MOVED" in hashes:
        verdict = "MOVED"
    elif len(real) >= 2:
        verdict = f"IDENTICAL x{len(real)}" if len(set(real)) == 1 else "DIVERGENT"
    elif missing:
        verdict = "NOT-COMPARED"
    elif len(real) == 1:
        verdict = "ONE-COLUMN"
    elif na:
        verdict = "N/A"
    else:
        verdict = "REFUSED"
    return verdict, shown


def diff(paths):
    cols = []
    for p in paths:
        with open(p) as fh:
            j = json.load(fh)
        cols.append((j.get("vendor") or os.path.basename(p), j))
    keys = sorted(set(k for _, j in cols for k in j["cells"]))
    names = [c for c, _ in cols]
    for n, (_, j) in zip(names, cols):
        if not j.get("complete", True):
            print(f"NOTE: column {n} is INCOMPLETE (the run was killed); lanes after the last one written are absent, not clean")
        if j.get("skipped"):
            print(f"NOTE: column {n} SKIPPED {j['skipped']} on request")
    # Refuse to blame the library for a fixture the vendors did not share.
    fx = [j.get("fixtures") for _, j in cols]
    if all(fx):
        for f in sorted(set(k for x in fx for k in x)):
            vals = [json.dumps(x.get(f), sort_keys=True) for x in fx]
            if len(set(vals)) != 1:
                print(f"FIXTURE MISMATCH on {f!r}: the columns were not handed the same bytes; "
                      f"cells on it are the fixture's divergence, not the library's")
                for n, x in zip(names, fx):
                    print(f"    {n}: {x.get(f)}")
    else:
        print("NOTE: a column carries no fixture hashes (older harness); fixture equality unchecked")
    hx = [j.get("heldout") for _, j in cols]
    if all(hx):
        for f in sorted(set(k for x in hx for k in x)):
            vals = [json.dumps(x.get(f), sort_keys=True) for x in hx]
            if len(set(vals)) != 1:
                print(f"HELD-OUT MISMATCH on {f!r}: the columns were not handed the same held-out bytes; "
                      f"infer and model cells on it are the fixture's divergence, not the library's")
                for n, x in zip(names, hx):
                    print(f"    {n}: {x.get(f)}")
    elif any(hx):
        print("NOTE: a column carries no held-out hashes (it predates the infer/model columns); "
              "its infer and model cells read NOT-COMPARED")
    print(f"| {'lane/fixture':<28} | {'verdict':<10} | " + " | ".join(f"{n:<16}" for n in names) + " |")
    print(f"|{'-'*30}|{'-'*12}|" + "|".join("-" * 18 for _ in names) + "|")
    bad, counts = 0, {}
    for k in keys:
        vals, shown = [], []
        for _, j in cols:
            c = j["cells"].get(k)
            if c is None:
                shown.append("(not run)"); continue
            if c["verdict"] == "REFUSED":
                shown.append("REFUSED"); continue
            if c["verdict"] == "MOVED":
                shown.append("MOVED"); vals.append("MOVED"); continue
            shown.append(c["hashes"][0]); vals.append(c["hashes"][0])
        ran = [v for v in vals]
        if "MOVED" in ran:
            verdict = "MOVED"
        elif len(ran) < 2:
            verdict = "ONE-COLUMN" if len(ran) == 1 else "REFUSED"
        elif len(set(ran)) == 1:
            verdict = f"IDENTICAL x{len(ran)}"
        else:
            verdict = "DIVERGENT"
        if verdict in ("MOVED", "DIVERGENT"):
            bad += 1
        if verdict == "DIVERGENT":
            # localise: which named part disagrees
            per = {}
            for n, (_, j) in zip(names, cols):
                c = j["cells"].get(k)
                if c and c.get("parts"):
                    for pk, pv in c["parts"][0].items():
                        per.setdefault(pk, []).append(pv)
            diverging = [pk for pk, pv in per.items() if len(set(pv)) > 1]
            agreeing = [pk for pk, pv in per.items() if len(set(pv)) == 1]
            if per:
                shown[0] = f"parts differ: {','.join(diverging) or '?'}; agree: {','.join(agreeing) or '-'}"
        counts[verdict.split(" ")[0]] = counts.get(verdict.split(" ")[0], 0) + 1
        print(f"| {k:<28} | {verdict:<10} | " + " | ".join(f"{s:<16}" for s in shown) + " |")
    print()
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # THE SECOND TABLE: infer (held-out rows) and model (saved bytes), cell by
    # cell, printed where at least one column carries the key. Cells no
    # column carries are counted as not compared, never as divergent.
    rows, uncarried, counts2 = [], 0, {}
    for k in keys:
        for col in ("infer", "model"):
            carried = any(f"{col}_verdict" in (j["cells"].get(k) or {}) for _, j in cols)
            if not carried:
                uncarried += 1
                continue
            verdict, shown = _diff_column(cols, k, col)
            if verdict in ("MOVED", "DIVERGENT", "RELOAD-MOVED"):
                bad += 1
            counts2[verdict.split(" ")[0]] = counts2.get(verdict.split(" ")[0], 0) + 1
            rows.append((k, col, verdict, shown))
    print()
    if rows:
        print(f"| {'lane/fixture':<28} | {'column':<6} | {'verdict':<12} | " + " | ".join(f"{n:<16}" for n in names) + " |")
        print(f"|{'-'*30}|{'-'*8}|{'-'*14}|" + "|".join("-" * 18 for _ in names) + "|")
        for k, col, verdict, shown in rows:
            print(f"| {k:<28} | {col:<6} | {verdict:<12} | " + " | ".join(f"{s:<16}" for s in shown) + " |")
        print()
    if uncarried:
        counts2["NOT-COMPARED"] = counts2.get("NOT-COMPARED", 0) + uncarried
        print(f"infer/model: {uncarried} column cells not compared (no JSON here carries them; "
              f"they predate the columns)")
    print("summary (infer/model): " + ", ".join(f"{k}={v}" for k, v in sorted(counts2.items())))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default="")
    ap.add_argument("--lanes", default="")
    ap.add_argument("--skip", default="", help="lanes to leave out, comma separated; each is reported as SKIPPED")
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--vendor", default=platform.machine())
    ap.add_argument("--allow-fast", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--diff", nargs="+", default=None, metavar="JSON",
                    help="compare JSONs cell by cell: the train column, then infer and model where carried")
    args = ap.parse_args()
    if args.diff:
        return diff(args.diff)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
