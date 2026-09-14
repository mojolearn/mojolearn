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
            (DBSCAN, agglomerative, spectral: `fit` and `fit_predict`, and
            SpectralClustering.predict raises NotImplementedError by
            design); no-predict means KMeans has `fit` and `fit_predict`
            only, no predict or transform (`cluster.py`, verified
            2026-09-14); function means the lane is not an estimator
            (linalg, metrics). THE FORECASTERS (holtwinters, arima; the
            infer probes of 2026-09-14) take no new rows, so their held-out
            axis is TIME: the infer column hashes the forecast beyond the
            fitted series at FORECAST_HORIZON steps, the fitted length, a
            horizon the train column (24 steps) does not hash, through
            every public out-of-sample entry (`forecast`, and ARIMA's
            `predict(n_obs, n_obs + h)`, which its docstring says is the
            same answer; the probe holds it to that). Until then those
            two lanes recorded `n/a:forecast`; a JSON that predates the
            probe reads ONE-COLUMN against a new one, never DIVERGENT.
    model   sha256 of the bytes `save(path)` wrote, for every estimator that
            has `save` and `load` (the random forests, the extra trees and
            every GradientBoosting lane). It supports ARTIFACT identity, the
            same model file on every vendor. The saved file is then loaded
            back and asked for the held-out rows again; if that hash differs
            from `infer` the column is RELOAD-MOVED, a file that does not
            predict what the model in memory predicts. Estimators without
            save/load record `n/a:no-save`. The trainers and SambaStack
            offer `save_checkpoint`/`from_checkpoint` instead; that pair
            feeds the same column (2026-09-13).

THE LANES, 46 (2026-09-13), one per public estimator plus linalg and metrics.

    trees      rf-clf rf-reg et-clf et-reg gbdt-symmetric gbdt-depthwise
               gbdt-lossguide gbdt-rmse gbdt-ordered-rmse gbdt-feature-freq
               iforest (last on purpose, see its comment)
    classical  kmeans knn knn-clf knn-reg radius dbscan pca tsvd ols ridge
               logistic lasso elasticnet svc svr kde gp agglomerative
               spectral holtwinters arima umap standard-scaler minmax-scaler
    neural     mlp byte-lm byte-lm-host-infer byte-lm-host-train mamba1
               mamba2 mamba3 transformer samba
    functions  gemm-pinned metrics

The 18 lanes added on 2026-09-13 (svr through samba above) are fed the SAME
fixture bytes in the shape their estimator wants; the derivation rules are
the helpers under "derived inputs". Where an estimator's own docstring says
its identity card is one vendor or two, the lane's docstring repeats that;
the tool measures, it does not claim.

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

THE FOURTH COLUMN, a CPU (the CPU training lane, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md section 3.3). On an install whose
`mojolearn.vendor()` is 'cpu' (no GPU set, a host binding under
mojolearn/host/ built), `--vendor` defaults to `cpu-<cpu model slug>` read
from the machine, the JSON gains a `host` object (cpu model, arch, the host
bindings built and what each reads back as its kernel-matrix column), and a
lane whose family has no host fit records REFUSED with the by-name sentence
"no CPU implementation of <binding>.<function> yet", never a hash of
something else. Two provenance rules apply to EVERY column since the same
day. `--vendor` must be a box label matching ^[a-z0-9][a-z0-9_.-]*$ and not
a placeholder (the 2026-09-13 NVIDIA column recorded "box-arch", an
unexpanded env default in the leg body), and `commit` is REQUIRED, from
MOJOLEARN_COMMIT, else `git rev-parse HEAD` of this checkout, else a COMMIT
or commit.txt witness at the repository root; a JSON with an empty commit
is not written (all three 2026-09-13 GPU columns carry "commit": "").
`--diff ... --require-columns N --lanes a,b` exits non-zero when any named
lane has fewer than N real hashes on a compared cell, because `IDENTICAL x3`
on a lane the CPU column should cover is the CPU binding refusing, not a
pass.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: A box label: lowercase, digits, `_ . -`, never a placeholder.
VENDOR_LABEL = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
PLACEHOLDER_LABELS = frozenset({
    "box-arch", "box", "arch", "vendor", "unknown", "none", "cpu", "gpu",
    "label", "todo", "tbd", "placeholder", "x", "test",
})
COMMIT_HASH = re.compile(r"^[0-9a-f]{7,40}$")


def cpu_model():
    """This machine's CPU model string, or None. `sysctl -n
    machdep.cpu.brand_string` on macOS; `model name` from /proc/cpuinfo, then
    `lscpu` "Model name" on Linux (an ARM64 /proc/cpuinfo has no model name)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            return out or None
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    key, _, value = line.partition(":")
                    if key.strip().lower() in ("model name", "cpu model") and value.strip():
                        return value.strip()
        except OSError:
            pass
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            key, _, value = line.partition(":")
            if key.strip().lower() == "model name" and value.strip():
                return value.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def slug(text):
    """`Apple M4` -> `apple-m4`, `Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz`
    -> `intel-r-xeon-r-platinum-8272cl-cpu-2.60ghz`."""
    s = re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-.")
    return re.sub(r"-{2,}", "-", s)


def check_vendor_label(label):
    """Refuse a label that is not a box: the regex, the placeholders, and any
    unexpanded `$` or `{`."""
    bad = (not label or not VENDOR_LABEL.match(label) or label in PLACEHOLDER_LABELS
           or "$" in label or "{" in label)
    if bad:
        raise SystemExit(
            f"REFUSING: --vendor {label!r} is not a box label. It must match "
            "^[a-z0-9][a-z0-9_.-]*$ and must not be a placeholder "
            f"({', '.join(sorted(PLACEHOLDER_LABELS))}). Name the box: apple-m4, "
            "nvidia-h100-sm_90a, amd-mi325x-gfx942, cpu-<cpu model slug>. The "
            "2026-09-13 NVIDIA column recorded \"box-arch\" from an unexpanded env "
            "default in the leg body; this refusal is what stops the next one."
        )
    return label


def default_vendor_label(ml):
    """`cpu-<cpu model slug>` on a CPU-only install, derived from the machine
    and never typed; `platform.machine()` elsewhere, as before."""
    if ml.vendor() == "cpu":
        model = cpu_model()
        if not model:
            raise SystemExit(
                "REFUSING: on a CPU-only install --vendor defaults to cpu-<cpu "
                "model slug> and this machine's CPU model could not be read; "
                "pass --vendor cpu-<model> explicitly"
            )
        return "cpu-" + slug(model)
    return platform.machine().lower()


def commit_witness():
    """(commit, source). MOJOLEARN_COMMIT wins; else `git rev-parse HEAD` of
    the checkout this tool lives in; else a COMMIT or commit.txt witness at
    its root (what a leg archive carries). Refuses an empty or malformed
    value: a column with no commit cannot be tied to the source it ran."""
    candidates = [("MOJOLEARN_COMMIT", os.environ.get("MOJOLEARN_COMMIT", "").strip())]
    try:
        git = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git = ""
    candidates.append(("git rev-parse HEAD", git))
    for name in ("COMMIT", "commit.txt"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            with open(p) as fh:
                text = fh.read().strip()
            candidates.append((name, text.split()[0] if text else ""))
    for source, value in candidates:
        if not value:
            continue
        if not COMMIT_HASH.match(value):
            raise SystemExit(f"REFUSING: commit {value!r} from {source} is not a git hash")
        return value, source
    raise SystemExit(
        "REFUSING: no commit witness. Set MOJOLEARN_COMMIT, run from a git "
        "checkout, or place a COMMIT or commit.txt file at the repository root. "
        "A JSON with an empty commit is not written."
    )


def host_record(ml):
    """The `host` object of a CPU column: the machine, the host bindings
    built, and what each reads back (its kernel-matrix column, the column
    the accelerator predicates detected, its sabotage flag). A binding that
    refuses to load records the refusal, so a REFUSED cell is attributable
    to an unbuilt or refused family rather than a bug."""
    from mojolearn import _backend, host_surface
    families = {}
    for basename in _backend.host_families_built():
        prefix = basename[len("_mojolearn_"):]
        try:
            m = _backend.load_host_module(basename)
            fam = dict(column=str(getattr(m, prefix + "_column")()))
            detected = getattr(m, prefix + "_detected_column", None)
            if detected is not None:
                fam["detected_column"] = str(detected())
            sab = getattr(m, prefix + "_sabotage", None)
            if sab is not None:
                fam["sabotage"] = bool(sab())
        except Exception as exc:
            fam = dict(error=f"{type(exc).__name__}: {exc}"[:300])
        families[basename] = fam
    columns = sorted(set(f.get("column", "unreadable") for f in families.values()))
    return dict(
        cpu_model=cpu_model(), arch=platform.machine(),
        target_cpu=os.environ.get("MOJOLEARN_TARGET_CPU", "unrecorded"),
        mojo_version=os.environ.get("MOJOLEARN_MOJO_VERSION", "unrecorded"),
        python=platform.python_version(),
        column=(columns[0] if len(columns) == 1 else ("none-built" if not columns else "MIXED:" + ",".join(columns))),
        routed=dict(_backend._HOST_MODULES),
        # The manifest the routing table and the gate lists are read from
        # (the host surface manifest lane, 2026-09-14), and every binding it
        # declares, so a family that is declared but absent from `families`
        # reads as NOT BUILT on this box rather than as a bug.
        surface=host_surface.SOURCE,
        declared=host_surface.bindings(),
        covered_lanes=host_surface.covered_lanes(),
        families=families,
    )


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

#: the forecasters' held-out axis is time: steps beyond the fitted series
#: in the infer column, equal to the fitted length (the two lanes fit 512
#: observations per series), as the held-out row counts mirror the
#: training row counts elsewhere. The train column keeps its 24 steps.
FORECAST_HORIZON = 512


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


@lane("pca-whiten")
def _(ml, X, yc, yr, Xh=None):
    # The whitened transform (the kde svc host lane, 2026-09-14): the same
    # fit as `pca` with `whiten=True`, so the whiten scale kernel and its
    # host restatement have a cell of their own on every column.
    m = ml.PCA(n_components=4, whiten=True).fit(X)
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
    # ExponentialSmoothing takes endog in the constructor and has no
    # predict(X): the only out-of-sample output is the forecast, and its
    # held-out axis is the horizon. The train column keeps forecast(24) as
    # it has always been; the infer column (2026-09-14) asks for
    # FORECAST_HORIZON steps through both return paths, the flat
    # single-series buffer and the `index=0` strided read.
    return _fit(dict(forecast=_h(m.forecast(24))), m,
                lambda e: _same_bytes("forecast(h)", e.forecast(FORECAST_HORIZON),
                                      "forecast(h, index=0)", e.forecast(FORECAST_HORIZON, index=0)))


def _same_bytes(name_a, a, name_b, b):
    """The forecasters' infer probe: two public entries the estimator
    documents as the same answer. Returns both for hashing when their bytes
    agree; raises, naming the pair and the byte count, when they do not, so
    the infer column reads REFUSED with the message instead of a hash that
    hides which of the two moved."""
    ba, bb = np.asarray(a).ravel().tobytes(), np.asarray(b).ravel().tobytes()
    if ba != bb:
        n = sum(x != y for x, y in zip(ba, bb)) + abs(len(ba) - len(bb))
        raise ValueError(f"{name_a} and {name_b} differ: {n} bytes of {max(len(ba), len(bb))}")
    return a, b


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


# ---------------------------------------------------------------- derived inputs
# The lanes below (2026-09-13) cover the estimators that were not in the
# probe, the ones whose natural input is not a feature matrix (a batch of
# series, a token stream, a (batch, length, d_model) activation slab) get it
# DERIVED FROM THE FIXTURE BYTES by the rules here, so every vendor is handed
# the same hostile values in a different shape. Anything that has to be the
# same on every box and is NOT part of the answer (weights, permutations, a
# radius) comes from the hashed stream or from fixed-order host arithmetic,
# never from a host BLAS or a host transcendental (labels_for's lesson).

def _hw(shape, seed, lo, hi):
    """A float32 tensor of `shape` from the hashed stream, uniform on
    [lo, hi); the seed is a string naming the lane and the tensor."""
    n = int(np.prod(shape))
    u = _hashed_uniform(n, 1, seed).reshape(-1)
    return np.ascontiguousarray((u * np.float32(hi - lo) + np.float32(lo)).astype(np.float32).reshape(shape))


def _seq(X, b, l, dm, skip=0):
    """A (b, l, dm) float32 activation slab, the fixture's values in
    row-major order, `skip` values in, the next b*l*dm of them. The
    `odd` fixture has 12345*17 values, enough for every slab asked for."""
    flat = np.ascontiguousarray(X).reshape(-1)
    return np.ascontiguousarray(flat[skip: skip + b * l * dm].reshape(b, l, dm)).astype(np.float32)


def _ids(X, b, l, vocab=256):
    """A (b, l) int32 token stream, the first b*l BYTES of the fixture
    (its float32 values viewed as bytes), so the `ties` fixture hands the
    language models a stream heavy in repeated bytes and `hashed` a flat
    one."""
    raw = np.frombuffer(np.ascontiguousarray(X).tobytes()[: b * l], dtype=np.uint8)
    return np.ascontiguousarray((raw.astype(np.int32) % vocab).reshape(b, l))


def _three_class(X):
    """Targets in 0..2 for the fixed 8-16-3 MLP, from the two columns
    no fixture perturbs (the same ones labels_for reads)."""
    return ((X[:, 3] > 0).astype(np.int32) + (X[:, 4] > 0).astype(np.int32)).astype(np.int32)


def _coded(X):
    """The FeatureFreq estimator wants dense categorical codes in its
    source columns, two code columns (a 2-way and a 4-way split on
    columns 4, 5 and 6 at their medians) in front of the first eight
    numeric columns, which vary in every fixture (`dupes` holds its
    constant and its zero column at the end)."""
    c0 = (X[:, 4] > np.median(X[:, 4])).astype(np.float32)
    c1 = (2 * (X[:, 5] > np.median(X[:, 5])).astype(np.int32)
          + (X[:, 6] > np.median(X[:, 6])).astype(np.int32)).astype(np.float32)
    return np.ascontiguousarray(np.column_stack([c0, c1, X[:, :8]]).astype(np.float32))


def _radius_for(index, queries):
    """A radius scaled to the fixture, six tenths of the median distance
    from the query rows to the first index row, in fixed-order float64
    host arithmetic (elementwise, no BLAS), then rounded to float32. A
    fixed radius would return nothing on `ties` and everything on
    `denormal`."""
    ref = index[0].astype(np.float64)
    d2 = np.zeros(queries.shape[0], dtype=np.float64)
    for j in range(index.shape[1]):
        d2 = d2 + (queries[:, j].astype(np.float64) - ref[j]) ** 2
    return float(np.float32(0.6 * np.median(np.sqrt(d2))))


def _ragged(result):
    """The arrays to hash for a ragged radius query. Per-row counts, then
    every distance and every index in row order (sorted within a row by
    the estimator, `sort_results=True`)."""
    dists, idx = result
    lens = np.asarray([np.asarray(a).size for a in idx], dtype=np.int64)
    dd = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in dists])
    ii = np.concatenate([np.asarray(a).reshape(-1).astype(np.int64) for a in idx])
    return lens, dd, ii


def _byte_lm_params(shape):
    """Named tensors for the byte LM from the hashed stream, norms at one
    and everything else uniform on [-1/8, 1/8) (the fan-in scale of a
    32-wide model as a dyadic rational, so no host sqrt). The same on
    every fixture, because the fixture is the TOKEN stream, not the weights.
    Returns (named dict, flat registry-order vector)."""
    named = {}
    for name, shp in zip(shape.parameter_names, shape.parameter_shapes):
        if name.endswith("norm1_w") or name.endswith("norm2_w"):
            named[name] = np.ones(shp, dtype=np.float32)
        else:
            named[name] = _hw(shp, "byte-lm:" + name, -0.125, 0.125)
    flat = np.ascontiguousarray(np.concatenate([named[n].reshape(-1) for n in shape.parameter_names]))
    return named, flat


def _block_weights(lane, shapes, ones=()):
    """A weight dict for a sequence block. `ones` names get a vector of
    ones (the norms), the rest are hashed uniform on [-1/8, 1/8)."""
    return {name: (np.ones(shp, dtype=np.float32) if name in ones
                   else _hw(shp, f"{lane}:{name}", -0.125, 0.125))
            for name, shp in shapes.items()}


def _block_fit(blk, x, g, state_kw):
    """The train column shared by the four sequence blocks. A stateless
    prefill, the same prefill into a fresh state followed by one decode
    step, and the backward pass (IDENTICAL only, by the blocks' own
    contract) with a fixture-derived cotangent."""
    y = np.asarray(blk.forward(x))
    st = blk.allocate_state(x.shape[0], **state_kw)
    y_state = np.asarray(blk.forward(x, st))
    y_step = np.asarray(blk.step(np.ascontiguousarray(x[:, :1, :]), st))
    grads = blk.backward(x, g)
    return dict(forward=_h(y), prefill=_h(y_state), step=_h(y_step),
                backward=_h(*[np.asarray(grads[k]) for k in sorted(grads)]))


# ---------------------------------------------------------------- lanes (2026-09-13)
# The estimators that were not in the probe. Cross-vendor standing is per
# estimator and is NOT this tool's claim; where the estimator's own
# docstring says its identity card is one vendor or two, the lane says so
# and measures anyway.

@lane("svr")
def _(ml, X, yc, yr, Xh=None):
    """SVR's docstring says it makes no cross-vendor claim and a three-vendor
    card is owed. Sized like the svc lane."""
    m = ml.SVR(C=1.0, kernel="rbf", epsilon=0.1, max_iter=200).fit(X[:2000], yr[:2000])
    return _fit(dict(predict=_h(m.predict(X[2000:2256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("arima")
def _(ml, X, yc, yr, Xh=None):
    """Four series of 512 observations, the first four columns of the
    fixture, transposed. ARIMA's docstring says the three-vendor card
    covers the filter and the fit is one vendor. Like holtwinters there
    are no new rows to feed: the held-out axis is the horizon. The train
    column keeps forecast(24); the infer column (2026-09-14) asks for
    FORECAST_HORIZON steps through both public out-of-sample entries,
    `forecast(h)` and `predict(n_obs, n_obs + h)`, which `_arima_impl.py`
    says are the same answer; `_same_bytes` holds it to that, so a byte
    between them reads REFUSED with its name, never a quiet hash."""
    series = np.ascontiguousarray(X[:512, :4].T)
    m = ml.ARIMA(order=(1, 0, 0)).fit(series)
    return _fit(dict(ar=_h(m.ar_), mu=_h(m.mu_), sigma2=_h(m.sigma2_), forecast=_h(m.forecast(24))),
                m, lambda e: _same_bytes("forecast(h)", e.forecast(FORECAST_HORIZON),
                                         "predict(n_obs, n_obs + h)",
                                         e.predict(e.n_obs_, e.n_obs_ + FORECAST_HORIZON)))


@lane("gp")
def _(ml, X, yc, yr, Xh=None):
    """The dense Cholesky is n^2 memory, so 256 rows of four columns, with
    a white-noise term so duplicate rows (`ties`) still factor. The GP's
    docstring says Apple and AMD IDENTICAL card, no NVIDIA card."""
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
    m = ml.GaussianProcessRegressor(kernel=k).fit(X[:256, :4], yr[:256])
    mean, std = m.predict(X[256:320, :4], return_std=True)
    return _fit(dict(alpha=_h(m.alpha_), L=_h(m.L_), lml=_h(np.float64(m.log_marginal_likelihood_value_)),
                     mean=_h(mean), std=_h(std)),
                m, lambda e: e.predict(Xh[:64, :4], return_std=True))


@lane("umap")
def _(ml, X, yc, yr, Xh=None):
    """Exact neighbor search is quadratic, so 1024 rows of eight columns
    and eight epochs. UMAP's docstring says fit certificates do not certify
    transform, which is what the infer column asks."""
    m = ml.UMAP(n_neighbors=8, n_components=2, n_epochs=8, random_state=3).fit(X[:1024, :8])
    return _fit(dict(embedding=_h(m.embedding_)), m, lambda e: (e.transform(Xh[:64, :8]),))


@lane("radius")
def _(ml, X, yc, yr, Xh=None):
    """Sized like the knn lane; the radius is scaled to the fixture by
    _radius_for and is part of the train column. Results sorted within a
    row so the hash asks about membership and order, not device order."""
    index, q = X[:4096], X[4096:4160]
    r = _radius_for(index, q)
    m = ml.RadiusNeighbors(radius=r).fit(index)
    lens, dd, ii = _ragged(m.radius_neighbors(q, sort_results=True))
    return _fit(dict(radius=_h(np.float32(r)), counts=_h(lens), dist=_h(dd), idx=_h(ii)),
                m, lambda e: _ragged(e.radius_neighbors(Xh[:64], sort_results=True)))


@lane("standard-scaler")
def _(ml, X, yc, yr, Xh=None):
    m = ml.StandardScaler().fit(X)
    t = m.transform(X[:256])
    return _fit(dict(mean=_h(m.mean_), var=_h(m.var_), scale=_h(m.scale_), transform=_h(t),
                     inverse=_h(m.inverse_transform(t))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("minmax-scaler")
def _(ml, X, yc, yr, Xh=None):
    m = ml.MinMaxScaler().fit(X)
    t = m.transform(X[:256])
    return _fit(dict(data_min=_h(m.data_min_), data_max=_h(m.data_max_), scale=_h(m.scale_), min=_h(m.min_),
                     transform=_h(t), inverse=_h(m.inverse_transform(t))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("gbdt-ordered-rmse")
def _(ml, X, yc, yr, Xh=None):
    """OrderedRMSE takes one explicit row permutation; it is the stable
    argsort of a hashed stream, the same on every box."""
    perm = np.argsort(_hashed_uniform(X.shape[0], 1, "ordered-permutation").reshape(-1), kind="stable")
    m = ml.OrderedRMSE(n_estimators=20, max_depth=6).fit(X, yr, permutation=perm)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-feature-freq")
def _(ml, X, yc, yr, Xh=None):
    """ExperimentalTwoLevelFeatureFreq, one depth-two tree over two coded
    source columns (_coded) and eight numeric columns."""
    m = ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7).fit(_coded(X), yr)
    return _fit(dict(predict=_h(m.predict(_coded(X)))), m, lambda e: (e.predict(_coded(Xh)),))


@lane("mlp")
def _(ml, X, yc, yr, Xh=None):
    """SmallMLPTrainer, the fixed 8-16-3 network. Three AdamW steps on
    three 64-row batches of the first eight columns, targets from
    _three_class, starting weights the surface test's arange rule. Train
    column is the three losses, the weights after the steps and the
    logits on the 256 training rows; the checkpoint is the model column."""
    w = [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
         for s in ((16, 8), (16,), (3, 16), (3,))]
    m = ml.SmallMLPTrainer(*w, data_schedule={"dataset": "identity_break", "order": "sequential"})
    Xm = np.ascontiguousarray(X[:256, :8])
    t = _three_class(X[:256])
    losses = [np.float64(m.train_step(Xm[64 * k:64 * (k + 1)], t[64 * k:64 * (k + 1)])["loss"]) for k in range(3)]
    return _fit(dict(loss=_h(np.asarray(losses)),
                     weights=_h(*[np.asarray(m.weights_[k]) for k in sorted(m.weights_)]),
                     logits=_h(np.asarray(m.predict_logits(Xm)))),
                m, lambda e: (np.asarray(e.predict_logits(np.ascontiguousarray(Xh[:256, :8]))),))


@lane("byte-lm")
def _(ml, X, yc, yr, Xh=None):
    """SmallByteLanguageModelTrainer (LanguageModelTrainer is the same
    class) at the default b2-l32-d32 profile, 34944 parameters. Three
    AdamW steps on three (2, 33) windows of the fixture bytes (_ids),
    weights from _byte_lm_params. The identity claim is per shape by the
    trainer's own contract; this is one shape."""
    shape = ml.ByteLanguageModelConfig()
    named, _ = _byte_lm_params(shape)
    m = ml.SmallByteLanguageModelTrainer(named, data_schedule={"dataset": "identity_break", "order": "sequential"},
                                         shape=shape)
    ids = _ids(X, 3 * shape.batch, shape.length + 1)
    losses = [np.float64(m.train_step(ids[2 * k:2 * k + 2])["loss"]) for k in range(3)]
    return _fit(dict(loss=_h(np.asarray(losses)), params=_h(np.asarray(m.parameters_)),
                     logits=_h(np.asarray(m.logits(ids[:2, :-1])))),
                m, lambda e: (np.asarray(e.logits(_ids(Xh, shape.batch, shape.length))),))


@lane("byte-lm-host-infer")
def _(ml, X, yc, yr, Xh=None):
    """LanguageModelInference, the CPU forward path, on the reference
    (unthreaded) arm, from the same starting weights as byte-lm. Its
    certificate is per CPU (docs/BYTE_LM_CPU_INFERENCE.md); this lane
    measures whatever CPU the box has."""
    shape = ml.ByteLanguageModelConfig()
    _, flat = _byte_lm_params(shape)
    m = ml.LanguageModelInference(flat, shape=shape, threaded=False)
    ids = _ids(X, shape.batch, shape.length + 1)
    return _fit(dict(loss_bits=_h(np.uint32(m.loss_bits(ids))), logits=_h(np.asarray(m.logits(ids[:, :-1]))),
                     next=_h(np.asarray(m.next_bytes(ids[:, :-1])))),
                m, lambda e: (np.asarray(e.logits(_ids(Xh, shape.batch, shape.length))),))


@lane("byte-lm-host-train")
def _(ml, X, yc, yr, Xh=None):
    """LanguageModelHostTrainer, one CPU training step (its docstring says not
    certified yet). It has no evaluation-only entry (`loss` IS a step), so
    the held-out probe is the loss bits of a second step on the held-out
    window."""
    shape = ml.ByteLanguageModelConfig()
    _, flat = _byte_lm_params(shape)
    m = ml.LanguageModelHostTrainer(flat, shape=shape)
    bits = m.train_step(_ids(X, shape.batch, shape.length + 1))
    return _fit(dict(loss_bits=_h(np.uint32(bits)), params=_h(np.asarray(m.parameters_)), m=_h(np.asarray(m.m_))),
                m, lambda e: (np.uint32(e.train_step(_ids(Xh, shape.batch, shape.length + 1))),))


# The sequence blocks: (2, 16, 32) slabs from the fixture (_seq), hashed
# weights at the smallest legal d_model (32, the Mamba-2/3 rule). Their
# docstrings say the IDENTICAL card is one vendor (transformer) or that
# broader backward qualification is open (Mamba); measured here on all.

@lane("mamba1")
def _(ml, X, yc, yr, Xh=None):
    dm, di, r = 32, 64, 2
    w = _block_weights("mamba1", {
        "norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4),
        "conv1d.bias": (di,), "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
        "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (dm, di)},
        ones=("norm.weight",))
    blk = ml.Mamba1Block(w)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), {})
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


@lane("mamba2")
def _(ml, X, yc, yr, Xh=None):
    dm, di, nh = 32, 64, 1
    cd, dip = di + 256, 2 * di + 256 + nh
    w = _block_weights("mamba2", {
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
        "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
        "out_proj.weight": (dm, di)},
        ones=("block_norm.weight", "norm.weight"))
    blk = ml.Mamba2Block(w)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), {})
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


@lane("mamba3")
def _(ml, X, yc, yr, Xh=None):
    dm, di, nh = 32, 64, 1
    dip = 2 * di + 256 + 3 * nh + 32
    w = _block_weights("mamba3", {
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "dt_bias": (nh,),
        "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128), "C_bias": (nh, 128),
        "D": (nh,), "out_proj.weight": (dm, di)},
        ones=("block_norm.weight", "B_norm.weight", "C_norm.weight"))
    blk = ml.Mamba3Block(w)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), {})
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


@lane("transformer")
def _(ml, X, yc, yr, Xh=None):
    dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
    w = _block_weights("transformer", {
        "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
        "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm), "v_proj.weight": (nkv * hd, dm),
        "o_proj.weight": (dm, nh * hd), "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
        "down_proj.weight": (dm, it)},
        ones=("input_layernorm.weight", "post_attention_layernorm.weight"))
    blk = ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), dict(max_tokens=32))
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


@lane("samba")
def _(ml, X, yc, yr, Xh=None):
    """SambaStack, one Mamba-3 layer and one attention layer at d_model 32
    over a 256-byte vocabulary, weights from the stack's own seeded
    generator, three AdamW steps on three (2, 17) windows of the fixture
    bytes. SUPPORT_MATRIX says cross-vendor qualification of the training
    surface is open. The checkpoint is the model column."""
    cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)
    m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3)
    ids = _ids(X, 6, 17)
    losses = [np.float64(m.train_step(ids[2 * k:2 * k + 2, :-1], ids[2 * k:2 * k + 2, 1:])["loss"])
              for k in range(3)]
    params = m.parameters()
    return _fit(dict(loss=_h(np.asarray(losses)), logits=_h(np.asarray(m.forward(ids[:2, :-1]))),
                     params=_h(*[np.asarray(params[k]) for k in sorted(params)])),
                m, lambda e: (np.asarray(e.forward(_ids(Xh, 2, 16))),))


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

def _save_load(est):
    """The (save method, load classmethod, suffix) an estimator offers.
    `save`/`load` on the forests, the boosting lanes and, since the
    classical host inference lane (2026-09-13 evening), ols, ridge, tsvd,
    logistic and pca, plus kde and svc since the kde svc host lane
    (2026-09-14), written to `.npz` as always, or
    `save_checkpoint`/`from_checkpoint` on the trainers and SambaStack
    (2026-09-13), a JSON envelope. None where it has neither."""
    if est is None:
        return None
    for s, l, suffix in (("save", "load", ".npz"), ("save_checkpoint", "from_checkpoint", ".json")):
        if callable(getattr(est, s, None)) and callable(getattr(type(est), l, None)):
            return s, l, suffix
    return None


def _has_save_load(est):
    return _save_load(est) is not None


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
    save, load, suffix = _save_load(fit.est)
    try:
        with tempfile.TemporaryDirectory(prefix="identity_break_") as tmp:
            path = os.path.join(tmp, f"{name}{suffix}")
            getattr(fit.est, save)(path)
            model = _hfile(path)
            back = getattr(type(fit.est), load)(path)
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
    # PROVENANCE BEFORE THE FIRST FIT. The label and the commit are refused
    # here, in seconds, not after nine fixtures of fits.
    vendor = check_vendor_label(args.vendor if args.vendor is not None else default_vendor_label(ml))
    commit, commit_source = commit_witness()
    host = host_record(ml) if ml.vendor() == "cpu" else None
    print(f"# vendor={vendor} commit={commit} ({commit_source})"
          + (f" host.cpu_model={host['cpu_model']!r} host.column={host['column']} "
             f"host.families={sorted(host['families'])}" if host else ""))

    if args.lanes:
        unknown = [n for n in args.lanes.split(",") if n and n not in LANES]
        if unknown:
            raise SystemExit(f"REFUSING: --lanes names no lane: {unknown}; lanes are {sorted(LANES)}")
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
        record = dict(mode=mode, repeats=args.repeats, platform=platform.platform(),
                      vendor=vendor, commit=commit, commit_source=commit_source,
                      heldout_seed=HELDOUT_SEED, fixtures=fixture_hashes,
                      heldout=heldout_hashes, cells=cells, complete=complete,
                      skipped=sorted(skip))
        if host is not None:
            record["host"] = host
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=1)

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


def _real_count(verdict):
    """How many real hashes a diff verdict rests on: `IDENTICAL xK` is K,
    ONE-COLUMN is 1, REFUSED is 0; DIVERGENT, MOVED and RELOAD-MOVED already
    fail on their own and are not counted here; N/A and NOT-COMPARED are
    None (no requirement applies)."""
    if verdict.startswith("IDENTICAL x"):
        return int(verdict.split("x", 1)[1])
    if verdict == "ONE-COLUMN":
        return 1
    if verdict == "REFUSED":
        return 0
    return None


def diff(paths, require_columns=0, require_lanes=None):
    cols = []
    for p in paths:
        with open(p) as fh:
            j = json.load(fh)
        cols.append((j.get("vendor") or os.path.basename(p), j))
    keys = sorted(set(k for _, j in cols for k in j["cells"]))
    names = [c for c, _ in cols]
    if require_columns and require_columns > len(cols):
        print(f"REQUIRE FAIL: --require-columns {require_columns} with {len(cols)} JSONs given")
    required = set(require_lanes) if require_lanes else (set(k.split("/")[0] for k in keys) if require_columns else set())
    short = []

    def require(key, col, verdict):
        if not require_columns or key.split("/")[0] not in required:
            return
        n = _real_count(verdict)
        if n is not None and n < require_columns:
            short.append((key, col, verdict, n))

    for n_, (_, j) in zip(names, cols):
        if j.get("host"):
            h = j["host"]
            print(f"NOTE: column {n_} is a CPU column: cpu_model={h.get('cpu_model')!r} "
                  f"column={h.get('column')} families={sorted(h.get('families', {}))} commit={j.get('commit')}")
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
        require(k, "train", verdict)
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
            require(k, col, verdict)
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
    if require_columns:
        if require_columns > len(cols):
            bad += 1
        for key, col, verdict, n in short:
            print(f"REQUIRE FAIL {key} {col}: {verdict} rests on {n} real hash(es), "
                  f"--require-columns {require_columns} demands that many; a column that "
                  "should cover this lane is refusing, which is not a pass")
        missing = sorted(required - set(k.split("/")[0] for k in keys))
        for lane_name in missing:
            print(f"REQUIRE FAIL {lane_name}: no JSON carries a cell for this lane")
        bad += len(short) + len(missing)
        print(f"require-columns {require_columns} over {sorted(required)}: "
              f"{'OK' if not short and not missing else str(len(short) + len(missing)) + ' short'}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default="")
    ap.add_argument("--lanes", default="")
    ap.add_argument("--skip", default="", help="lanes to leave out, comma separated; each is reported as SKIPPED")
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--vendor", default=None,
                    help="the box label, ^[a-z0-9][a-z0-9_.-]*$ and not a placeholder; default "
                         "cpu-<cpu model slug> on a CPU-only install, platform.machine() elsewhere")
    ap.add_argument("--allow-fast", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--diff", nargs="+", default=None, metavar="JSON",
                    help="compare JSONs cell by cell: the train column, then infer and model where carried")
    ap.add_argument("--require-columns", type=int, default=0, metavar="N",
                    help="with --diff: exit non-zero unless every compared cell of the lanes named by "
                         "--lanes (every lane when --lanes is empty) rests on at least N real hashes")
    args = ap.parse_args()
    if args.diff:
        lanes = [n for n in args.lanes.split(",") if n] if args.lanes else None
        if lanes:
            unknown = [n for n in lanes if n not in LANES]
            if unknown:
                raise SystemExit(f"REFUSING: --lanes names no lane: {unknown}; lanes are {sorted(LANES)}")
        return diff(args.diff, args.require_columns, lanes)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
