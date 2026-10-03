# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""fa_diag.py <dataset> <data dir>: one FactorAnalysis fit (ours, the board's
parameters and rows, MOJOLEARN_NUMERIC_MODE from the env) and a few lines on
where its held-out log-likelihood comes from (lane/apple-fast-quality-glmfa).
No opponent runs. Prints FA-DIAG lines only."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import bench_board_algos as bba  # noqa: E402

ds, data = sys.argv[1], sys.argv[2]
lane = "factor-analysis"
B, _rec = bba._load_block(lane, ds, data)
D = bba.lane_arrays(lane, B)
make, _name, params = bba._est_factory(lane, "ours-fast", D)
est = make().fit(D["X"])
X = D["X"].astype(np.float64)
Xq = D["Xq"].astype(np.float64)
W = np.asarray(est.components_, np.float64)
psi = np.asarray(est.noise_variance_, np.float64)
mu = np.asarray(est.mean_, np.float64)


def ll(W, psi, mu):
    C = W.T @ W + np.diag(psi)
    R = Xq - mu
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, R.T)
    return float(np.mean(-0.5 * (z * z).sum(0)) - np.log(np.diag(L)).sum() - 0.5 * R.shape[1] * np.log(2 * np.pi))


lk = list(getattr(est, "loglike_", []))
dl = np.diff(lk[-6:]) if len(lk) > 1 else []
var = X.var(0)
const = np.ptp(X, 0) == 0
mu64 = X.mean(0)
print("FA-DIAG n_iter=%s mode=%s last_dll=%s" % (est.n_iter_, os.environ.get("MOJOLEARN_NUMERIC_MODE"),
                                                 [round(float(v), 3) for v in dl]))
print("FA-DIAG const_cols=%d const_nonzero=%d const_in_Xq_too=%d max|mu-mu64|/(|mu64|+1)=%.3g"
      % (const.sum(), (const & (mu64 != 0)).sum(), (const & (np.ptp(Xq, 0) == 0)).sum(),
         float(np.max(np.abs(mu - mu64) / (np.abs(mu64) + 1)))))
r = psi / np.maximum(var, 1e-300)
print("FA-DIAG psi<=1e-11:%d psi/var<1e-6:%d psi/var<1e-9:%d min_psi=%.3g"
      % ((psi <= 1e-11).sum(), (r < 1e-6).sum(), (r < 1e-9).sum(), psi.min()))
p2 = psi.copy()
p2[const] = 1e-12
mu2 = mu.copy()
mu2[const] = mu64[const]
print("FA-DIAG ll_ours=%.4f ll_constfix=%.4f" % (ll(W, psi, mu), ll(W, p2, mu2)))
