#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""sgdoc_objective.py <rows-dir> <dataset> [seeds]: scikit-learn's
SGDOneClassSVM (the board's sgd-ocsvm params: nu=0.1, max_iter=20,
tol=None) on the cls block, one fit per seed, with its objective
J = nu/2 |w|^2 + nu (1 - rho) + mean max(0, rho - w.x) on the fit rows and
the fraction of held-out rows flagged; plus the optimum's certificate on
this X (the column mean's norm: a feasible dual point, J* >= nu - nu/2 |mean|^2,
and (w, rho) = (0, 0) has J = nu). lane/apple-fast-sgdoc-parallel; a small
run on the M2 (rows-small, 50k rows), never on the M3."""
import sys, time
import numpy as np
from sklearn.linear_model import SGDOneClassSVM

rows, ds = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "7,0,1,2,3").split(",")]
z = np.load("%s/cls-%s.npz" % (rows, ds))
X, Xq = z["X"], z["Xq"]
nu = 0.1


def obj(w, rho):
    sc = X.astype(np.float64) @ np.asarray(w, np.float64)
    return 0.5 * nu * float(w @ w) + nu * (1.0 - rho) + float(np.maximum(0.0, rho - sc).mean())


mu = X.astype(np.float64).mean(0)
print("SGDOC-OBJ %s n=%d d=%d nq=%d |colmean|=%.3e lower_bound=%.9f J(0,0)=%.9f" % (
    ds, X.shape[0], X.shape[1], Xq.shape[0], float(np.sqrt(mu @ mu)), nu - 0.5 * nu * float(mu @ mu), nu))
for s in seeds:
    t0 = time.perf_counter()
    m = SGDOneClassSVM(nu=nu, max_iter=20, tol=None, random_state=s).fit(X)
    ms = (time.perf_counter() - t0) * 1e3
    w = m.coef_.astype(np.float64).reshape(-1)
    rho = float(m.offset_[0])
    ff = float((m.predict(Xq) < 0).mean())
    print("SGDOC-OBJ %s sklearn seed=%d J=%.9f |w|=%.4e offset=%.4e fraction_flagged=%.5f fit_ms=%.0f" % (
        ds, s, obj(w, rho), float(np.sqrt(w @ w)), rho, ff, ms))
