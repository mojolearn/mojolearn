# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""bayesq_probe.py <data dir> <dataset>: why BayesianRidge FAST quality on a
board block differs from scikit-learn's (lane/apple-fast-bayesq). One fit per
arm, no timing. Prints BQ lines: the column spread of the fit rows, each arm's
alpha_, lambda_, n_iter_, held-out r2/rmse, then the f64 eigenbasis of the
centered fit Gram with each arm's coefficient in it (z_k = V_k' coef) and the
held-out spread along V_k, so the directions that carry the held-out error are
named. Arm "ours@sk" refits ours at scikit-learn float32's alpha_ and lambda_
with max_iter=0 (the solve alone, no evidence iteration)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_board_algos as bba  # noqa: E402

data, ds = sys.argv[1], sys.argv[2]
lane = "bayesian-ridge"
B, _ = bba._load_block(lane, ds, data)
D = bba.lane_arrays(lane, B)
X, y, Xq, yq = D["X"], D["y"], D["Xq"], D["yq"]
print("BQ data", ds, "X", X.shape, X.dtype, "Xq", Xq.shape, "y std %.4g" % float(np.std(y.astype(np.float64))))
X64 = X.astype(np.float64)
mu = X64.mean(0)
sd = X64.std(0)
sdq = Xq.astype(np.float64).std(0)
order = np.argsort(sd)
print("BQ colspread fit-std smallest:", " ".join("%d:%.3g/q%.3g" % (j, sd[j], sdq[j]) for j in order[:12]))
print("BQ colspread n(fit std<1e-6)=%d n(<1e-3)=%d max|Xq|=%.4g max|X|=%.4g" % (
    int((sd < 1e-6).sum()), int((sd < 1e-3).sum()), float(np.abs(Xq).max()), float(np.abs(X).max())))


def reg(p):
    r = bba._reg(yq, p)
    return "r2=%.6g rmse=%.6g finite=%s" % (r["r2"], r["rmse"], r["finite"])


fits = {}
from sklearn.linear_model import BayesianRidge as SkBR  # noqa: E402
for name, dt in (("sk32", np.float32), ("sk64", np.float64)):
    m = SkBR(max_iter=300, tol=1e-3).fit(X.astype(dt), y.astype(dt))
    fits[name] = (np.asarray(m.coef_, np.float64), float(m.intercept_))
    print("BQ arm", name, "alpha=%.6g lambda=%.6g n_iter=%d |coef|=%.6g max|coef|=%.6g" % (
        m.alpha_, m.lambda_, m.n_iter_, np.linalg.norm(m.coef_), np.abs(m.coef_).max()), reg(m.predict(Xq.astype(dt))))
    if name == "sk32":
        a32, l32 = float(m.alpha_), float(m.lambda_)

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "fast")
import mojolearn as ml  # noqa: E402
for name, kw in (("ours", dict(max_iter=300, tol=1e-3)),
                 ("ours@sk", dict(max_iter=0, alpha_init=a32, lambda_init=l32))):
    try:
        m = ml.BayesianRidge(**kw).fit(X, y)
        c = np.asarray(m.coef_, np.float64)
        fits[name] = (c, float(m.intercept_))
        p = np.asarray(m.predict(Xq), np.float64)
        print("BQ arm", name, "alpha=%.6g lambda=%.6g n_iter=%d |coef|=%.6g max|coef|=%.6g" % (
            m.alpha_, m.lambda_, m.n_iter_, np.linalg.norm(c), np.abs(c).max()), reg(p))
    except Exception as exc:  # noqa: BLE001
        print("BQ arm", name, "ERROR", repr(exc)[:300])

Xc = X64 - mu
G = Xc.T @ Xc
ev, V = np.linalg.eigh(G)
print("BQ spectrum f64 ev max=%.6g min=%.6g n(ev<1e-7 max)=%d n(ev<1e-12 max)=%d" % (
    ev[-1], ev[0], int((ev < 1e-7 * ev[-1]).sum()), int((ev < 1e-12 * ev[-1]).sum())))
Pq = (Xq.astype(np.float64) - mu) @ V          # held-out coordinates in the eigenbasis
spread = np.sqrt((Pq * Pq).mean(0))
Z = {k: V.T @ c for k, (c, _) in fits.items()}
names = list(Z)
# each direction's held-out rms contribution per arm: |z_k| * spread_k
contrib = {k: np.abs(Z[k]) * spread for k in names}
print("BQ dir header k ev fitspread qspread " + " ".join("z_%s q_%s" % (k, k) for k in names))
show = sorted(set(list(range(12)) + list(np.argsort(-contrib.get("ours", contrib[names[0]]))[:12])))
for k in show:
    print("BQ dir %d %.4g %.4g %.4g " % (k, ev[k], np.sqrt(max(ev[k], 0) / X.shape[0]), spread[k])
          + " ".join("%.4g %.4g" % (Z[n][k], contrib[n][k]) for n in names))
for n in names:
    c = contrib[n]
    top = np.argsort(-c)[:5]
    print("BQ top", n, " ".join("%d:%.3g" % (k, c[k]) for k in top), "total_rms_bound=%.4g" % float(np.sqrt((c * c).sum())))
