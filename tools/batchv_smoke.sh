#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# batchv_smoke.sh: import smoke, FAST mode, for the lane/apple-fast-batchv
# defaults (x_prep/fastpt.mojo PowerTransformer / col_stats row-tiled grids,
# kde/impl/neighbors/kernel_density.mojo KDE_DIMTILE,
# hdbscan HDB_SMR_TILED). One line per
# estimator, then BATCHV-SMOKE status=ok|FAIL. Run in a built tree (wrap in
# ~/mq/ensure_so.sh). Times nothing.
set -u
cd "$(dirname "$0")/../python"
PY=${BATCHV_PY:-$HOME/board-0834/cache/venv/bin/python}
PYTHONPATH=$PWD MOJOLEARN_NUMERIC_MODE=fast "$PY" - <<'PYEOF'
import traceback
import numpy as np
from mojolearn._expansion_prep import PowerTransformer, SimpleImputer, _ptimpute_flags
from mojolearn.density import KernelDensity
from mojolearn.hdbscan import HDBSCAN

r = np.random.default_rng(0)
bad = []


def check(name, fn):
    try:
        ok, msg = fn()
    except Exception:
        traceback.print_exc()
        ok, msg = False, "raised"
    print("BATCHV %s %s %s" % (name, "ok" if ok else "FAIL", msg))
    if not ok:
        bad.append(name)


def pt():
    X = r.lognormal(0, 0.7, (5000, 13)).astype(np.float32)
    m = PowerTransformer(method="yeo-johnson", standardize=True).fit(X)
    T = np.asarray(m.transform(X), dtype=np.float64)
    lam = np.asarray(m.lambdas_, dtype=np.float64)
    ok = T.shape == X.shape and np.isfinite(T).all() and np.isfinite(lam).all() \
        and abs(T.mean()) < 1e-2 and abs(T.std() - 1) < 1e-2
    return ok, "flags=%d mean=%.2e std=%.4f" % (_ptimpute_flags("fast"), T.mean(), T.std())


def si():
    X = r.standard_normal((5000, 9)).astype(np.float32)
    X[r.random(X.shape) < 0.1] = np.nan
    m = SimpleImputer(strategy="mean").fit(X)
    s = np.asarray(m.statistics_, dtype=np.float64)
    ref = np.nanmean(X.astype(np.float64), axis=0)
    err = float(np.abs(s - ref).max())
    return err < 1e-4, "max_err=%.2e" % err


def kde():
    X = r.standard_normal((3000, 7)).astype(np.float32)
    k = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(X[:2500])
    s = np.asarray(k.score_samples(X[2500:]), dtype=np.float64)
    return s.shape == (500,) and np.isfinite(s).all(), "mean_ll=%.4f" % s.mean()


def hdb():
    # d = 80 > 64: the sparse arm, HDB_SMR_TILED's kernel
    c = r.standard_normal((4, 80)) * 6
    X = (c[r.integers(0, 4, 3000)] + r.standard_normal((3000, 80))).astype(np.float32)
    lab = np.asarray(HDBSCAN(min_cluster_size=50).fit(X).labels_)
    k = len(set(lab.tolist()) - {-1})
    return lab.shape == (3000,) and k >= 2, "n_clusters=%d noise=%.3f" % (k, float((lab == -1).mean()))


check("power-transformer", pt)
check("hdbscan", hdb)
check("simple-imputer", si)
check("kde", kde)
print("BATCHV-SMOKE status=%s%s" % ("FAIL " if bad else "ok", " ".join(bad)))
PYEOF
