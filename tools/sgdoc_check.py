#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""sgdoc_check.py <rows-dir> <dataset>: quality check of the FAST
SGDOneClassSVM built with -D MOJOLEARN_SGDOC_FAST_PAR (run with
MOJOLEARN_NUMERIC_MODE=fast) on rows-small: its objective J against
scikit-learn's (same params) and against a float64 Frank-Wolfe reference,
on the cls block as is (centered: optimum w = 0, J = nu) and shifted by +1
per column (an off-origin optimum, so the Frank-Wolfe steps run).
Verification only (no timing); lane/apple-fast-sgdoc-parallel."""
import sys
import numpy as np
from sklearn.linear_model import SGDOneClassSVM as SkOC
import mojolearn

rows, ds = sys.argv[1], sys.argv[2]
z = np.load("%s/cls-%s.npz" % (rows, ds))
nu = 0.1


def obj(X, w, rho):
    sc = X.astype(np.float64) @ np.asarray(w, np.float64)
    return 0.5 * nu * float(w @ w) + nu * (1.0 - rho) + float(np.maximum(0.0, rho - sc).mean())


def fw_ref(X, iters=3000):
    """float64 Frank-Wolfe on the dual (x_linear/sgdoc_fast.mojo's steps)."""
    X = X.astype(np.float64)
    n = X.shape[0]
    R = int(np.ceil(nu * n - n * 1e-7))
    c = 1.0 / (nu * n)
    u = X.mean(0)
    for it in range(iters):
        sc = X @ u
        o = np.argsort(sc, kind="stable")
        tau = sc[o[R - 1]]
        less = sc < tau
        eq = sc == tau
        wt = (1.0 - less.sum() * c) / eq.sum()
        s = c * X[less].sum(0) + wt * X[eq].sum(0)
        g = u @ (u - s)
        dm = (u - s) @ (u - s)
        if g <= 1e-9 * max(u @ u, s @ s) or dm <= 0:
            break
        u = u + min(1.0, max(0.0, g / dm)) * (s - u)
    us = u @ s
    t = us / (u @ u) if us > 0 and u @ u > 0 else 0.0
    return t * u, t * tau, it


for name, sh in (("as-is", 0.0), ("shift+1", 1.0)):
    X = (z["X"] + np.float32(sh)).astype(np.float32)
    Xq = (z["Xq"] + np.float32(sh)).astype(np.float32)
    m = mojolearn.SGDOneClassSVM(nu=nu, max_iter=20, tol=None, random_state=7).fit(X)
    w = np.asarray(m.coef_, np.float64).reshape(-1)
    rho = float(np.asarray(m.offset_).reshape(-1)[0])
    ff = float((np.asarray(m.predict(Xq)) < 0).mean())
    k = SkOC(nu=nu, max_iter=20, tol=None, random_state=7).fit(X)
    wk = k.coef_.astype(np.float64).reshape(-1)
    rk = float(k.offset_[0])
    wr, rr, itr = fw_ref(X)
    print("SGDOC-CHECK %s %s ours J=%.9f |w|=%.4e offset=%.5e n_iter=%s flagged=%.5f | sklearn J=%.9f flagged=%.5f"
          " | ref64 J=%.9f iters=%d flagged=%.5f" % (
              ds, name, obj(X, w, rho), float(np.sqrt(w @ w)), rho, getattr(m, "n_iter_", "?"), ff,
              obj(X, wk, rk), float((k.predict(Xq) < 0).mean()),
              obj(X, wr, rr), itr, float(((Xq.astype(np.float64) @ wr - rr) < 0).mean())))
