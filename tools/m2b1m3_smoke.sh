#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# m2b1m3_smoke.sh: import smoke, FAST mode, for the lane/apple-fast-m2b1-m3
# defaults: IVF k-means++ device seeding (IVFIndex, IVFPQIndex), MaxAbsScaler
# direct fit, MultinomialNB on CSR, LinearSVR (LS_BATCH + SLIM; LSVR_ALL at
# n_features <= 32, the old path above). One line per estimator, then
# M2B1M3-SMOKE status=ok|FAIL. Run in a built tree.
set -u
cd "$(dirname "$0")/../python"
PY=${M2B1M3_PY:-$HOME/board-0834/cache/venv/bin/python}
MOJOLEARN_NUMERIC_MODE=fast "$PY" - <<'PYEOF'
import traceback
import numpy as np
import mojolearn as ml
from mojolearn._ivf_impl import IVFIndex
from mojolearn._expansion_ann import IVFPQIndex
from mojolearn._expansion_prep import MaxAbsScaler, MultinomialNB

r = np.random.default_rng(0)
bad = []


def check(name, fn):
    try:
        ok, msg = fn()
    except Exception:
        traceback.print_exc()
        ok, msg = False, "raised"
    print("M2B1M3 %s %s %s" % (name, "ok" if ok else "FAIL", msg))
    if not ok:
        bad.append(name)


def _ids(res):
    if isinstance(res, tuple):
        res = res[-1]
    return np.asarray(res)


def ivf(cls, floor, **kw):
    def run():
        X = r.random((20000, 32), dtype=np.float32)
        Q = X[:200] + np.float32(1e-3)
        idx = cls(n_lists=64, n_probes=8, n_neighbors=10, random_state=0, **kw).fit(X)
        got = _ids(idx.search(Q))
        hit = float(np.mean(got[:, 0] == np.arange(200)))
        return hit >= floor and got.shape == (200, 10), "shape=%s self_hit=%.3f" % (got.shape, hit)
    return run


def maxabs():
    X = (r.standard_normal((5000, 17)) * 3).astype(np.float32)
    m = MaxAbsScaler().fit(X)
    ref = np.abs(X).max(0)
    err = float(np.abs(np.asarray(m.max_abs_) - ref).max())
    return err == 0.0, "max_abs_err=%g" % err


def mnb():
    import scipy.sparse as sp
    X = r.poisson(0.3, (3000, 200)).astype(np.float32)
    y = (X[:, :100].sum(1) > X[:, 100:].sum(1)).astype(np.int32)
    a = MultinomialNB().fit(sp.csr_matrix(X), y)
    b = MultinomialNB().fit(X, y)
    pa = np.asarray(a.predict(sp.csr_matrix(X)))
    pb = np.asarray(b.predict(X))
    agree = float(np.mean(pa == pb))
    return agree >= 0.999, "csr_vs_dense_agree=%.4f acc=%.4f" % (agree, float(np.mean(pa == y)))


def lsvr(d):
    def run():
        X = r.standard_normal((20000, d)).astype(np.float32)
        w = r.standard_normal(d).astype(np.float32)
        y = (X @ w + 0.1 * r.standard_normal(20000)).astype(np.float32)
        p = np.asarray(ml.LinearSVR(random_state=0).fit(X, y).predict(X))
        r2 = 1.0 - float(((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        return r2 > 0.95, "d=%d r2=%.4f" % (d, r2)
    return run


check("ivf-flat", ivf(IVFIndex, 0.5))
check("ivf-pq", ivf(IVFPQIndex, 0.1, pq_dim=16))
check("maxabs-scaler", maxabs)
check("multinomial-nb-csr", mnb)
check("linearsvr-d11", lsvr(11))
check("linearsvr-d64", lsvr(64))
print("M2B1M3-SMOKE status=%s%s" % ("ok" if not bad else "FAIL", "" if not bad else " failed=" + ",".join(bad)))
PYEOF
