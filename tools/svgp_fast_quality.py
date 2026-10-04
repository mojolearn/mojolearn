#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVGP FAST quality dump and main/candidate comparison (lane apple-fast-w2-svgp).

dump OUT.npz --dataset taxi|istella [--data DIR]
    The board's svgp lane exactly (tools/bench_board_algos.py: 100k stride
    rows, 512 stride inducing points, kernel variance / lengthscale / noise
    1.0, jitter 1e-6, 20k stride query rows): fit, predict, and save the
    predictions, the query targets, elbo_, alpha, C, q_mu and q_sqrt.
compare A.npz B.npz
    A = main, B = candidate. Gate, fixed before any result (2026-10-04):
      r2_B   >= r2_A - 1e-4
      rmse_B <= rmse_A * (1 + 1e-4)
      elbo_B >= elbo_A - 1e-4 * |elbo_A|
    One-sided: the candidate may be better, never worse beyond 1e-4.
    Why 1e-4: BLKCHOL and RBFTILE keep every chain's operation order (the
    float-float Cholesky entries, each kernel cell's feature fold), so only
    FAST's free a*b+c contraction can move a word (<= 1 ulp, ~6e-8 relative,
    per kernel cell); BSPLIT re-associates float-float sums (~1e-14 relative
    in B and b). 1e-4 is ~1000x a float32 rounding of the metrics and well
    below the board's displayed digits. The prediction difference, bit
    identity of every array and the C / q_mu relative differences are
    printed for the record, not gated (Sigma's conditioning on taxi, 1e-6 ..
    9e5, can move the inducing-space state without moving predictions).
Prints one line: SVGP-FAST-AB status=PASS|FAIL ...
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "svgp-board-v1"
TOL = 1e-4


def _board():
    sys.path.insert(0, str(ROOT / "tools"))
    import bench_board_algos as bba
    return bba


def default_data():
    for b in ("board-0834", "board-0833"):
        p = Path.home() / b / "cache/algos-data/rows-full"
        if p.is_dir():
            return str(p)
    raise SystemExit("no board data dir (~/board-0834/cache/algos-data/rows-full)")


def dump(path, dataset, data):
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast", "MOJOLEARN_NUMERIC_MODE=fast required"
    sys.path.insert(0, str(ROOT / "python"))
    bba = _board()
    B, _rec = bba._load_block("svgp", dataset, data)
    D = bba.lane_arrays("svgp", B)
    p = bba.LANES["svgp"]["params"]
    X, y, Xq, yq = D["X"], D["y"], D["Xq"], D["yq"]
    Z0 = bba._stride(X, p["n_inducing"])
    hyp = {k: p[k] for k in ("kernel_variance", "lengthscale", "noise_variance", "jitter")}
    import mojolearn as ml
    from mojolearn import _expansion_neighbors as xn
    model = ml.SVGP(inducing_points=Z0, **hyp).fit(X, y)
    pred = np.asarray(model.predict(Xq), dtype=np.float32).reshape(-1)
    out = dict(pred=pred, yq=np.asarray(yq, dtype=np.float32), elbo=np.float64(model.elbo_),
               alpha=np.asarray(model._alpha), C=np.asarray(model._C), q_mu=np.asarray(model.q_mu_),
               q_sqrt=np.asarray(model.q_sqrt_))
    np.savez(path, **out)
    so = Path(xn.__file__).with_name("_mojolearn_x_neighbors.so")
    meta = dict(fixture=FIXTURE, dataset=dataset, n=int(X.shape[0]), m=int(Z0.shape[0]), d=int(X.shape[1]),
                nq=int(Xq.shape[0]), binding_sha256=hashlib.sha256(so.read_bytes()).hexdigest() if so.exists() else None)
    print("SVGP-FAST-CAPTURE " + json.dumps(meta, sort_keys=True))


def _metrics(z):
    y = z["yq"].astype(np.float64)
    p = z["pred"].astype(np.float64)
    r = y - p
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((r * r).sum()) / ss, float(np.sqrt(np.mean(r * r))), float(z["elbo"])


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b)) / n if n else float(np.linalg.norm(a - b))


def compare(fa, fb):
    a, b = np.load(fa), np.load(fb)
    assert a["yq"].tobytes() == b["yq"].tobytes(), "different query targets"
    r2a, rma, ea = _metrics(a)
    r2b, rmb, eb = _metrics(b)
    finite = all(np.isfinite(b[k]).all() for k in ("pred", "alpha", "C", "q_mu", "q_sqrt")) and np.isfinite(eb)
    ok = (finite and r2b >= r2a - TOL and rmb <= rma * (1 + TOL) and eb >= ea - TOL * abs(ea))
    exact = all(a[k].tobytes() == b[k].tobytes() for k in ("pred", "alpha", "C", "q_mu", "q_sqrt", "elbo"))
    dmax = float(np.max(np.abs(a["pred"].astype(np.float64) - b["pred"].astype(np.float64))))
    print("SVGP-FAST-AB status=%s r2 %.6f -> %.6f rmse %.6f -> %.6f elbo %.6g -> %.6g tol=%g exact=%s "
          "pred_maxabs=%.3g C_rel=%.3g qmu_rel=%.3g alpha_rel=%.3g" % (
              "PASS" if ok else "FAIL", r2a, r2b, rma, rmb, ea, eb, TOL, exact, dmax,
              _rel(a["C"], b["C"]), _rel(a["q_mu"], b["q_mu"]), _rel(a["alpha"], b["alpha"])))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("dump", "compare"))
    ap.add_argument("first")
    ap.add_argument("second", nargs="?")
    ap.add_argument("--dataset", choices=("taxi", "istella"))
    ap.add_argument("--data")
    args = ap.parse_args()
    if args.action == "dump":
        assert args.dataset, "--dataset required"
        dump(args.first, args.dataset, args.data or default_data())
    else:
        sys.exit(0 if compare(args.first, args.second) else 1)


if __name__ == "__main__":
    main()
