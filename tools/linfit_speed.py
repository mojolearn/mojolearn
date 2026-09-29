#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/linfit-speed: time and hash our iterative linear fits at the bench
board's shapes and settings (tools/bench_board_algos.py LANES: sgd-clf,
sgd-reg, poisson), one fit per (lane, dataset), in THIS tree's bindings.

    python3 tools/linfit_speed.py --data DIR [--lanes poisson,sgd-reg,sgd-clf]
        [--datasets taxi,istella] [--rows N] [--max-iter K] [--env K=V ...] --out F.json

DIR holds the board's prep blocks (bench_board_algos.py prep: reg-<ds>.npz,
cls-<ds>.npz). Each record: seconds of `fit` (one fit, the board's clock),
n_iter_, and the sha256 of coef_ | intercept_ bytes (the bits the A/B and
the gate compare across trees and settings). --env sets environment
variables for the fits (e.g. MOJOLEARN_X_LINEAR_GLM_TEAM=1, the one-block
GLM; MOJOLEARN_X_LINEAR_GW_STEPS=4096, tiny bounded launches)."""
import argparse
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "python"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="the board's prep blocks (or --synthetic)")
    ap.add_argument("--synthetic", default="", help="N,D: seeded synthetic data instead of --data")
    ap.add_argument("--lanes", default="poisson,sgd-reg,sgd-clf")
    ap.add_argument("--datasets", default="taxi,istella")
    ap.add_argument("--rows", type=int, default=0, help="first N fit rows (0 = all)")
    ap.add_argument("--max-iter", type=int, default=0, help="override max_iter (0 = the board's)")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], help="param override K=V (V as Python literal)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    import numpy as np
    import bench_board_algos as bba
    import mojolearn as ml
    recs = []
    for lane in [x for x in a.lanes.split(",") if x]:
        s = bba.LANES[lane]
        params = dict(bba._SHARED.get(lane, {}))
        params.update(s["params"])
        if a.max_iter:
            params["max_iter"] = a.max_iter
        import ast
        for kv in a.set:
            k, v = kv.split("=", 1)
            params[k] = ast.literal_eval(v)
        for ds in [x for x in a.datasets.split(",") if x]:
            if a.synthetic:
                sn, sd = (int(v) for v in a.synthetic.split(","))
                rng = np.random.default_rng(7 + len(ds))
                X = rng.standard_normal((sn, sd)).astype(np.float32)
                w = (rng.standard_normal(sd) / np.sqrt(sd)).astype(np.float32)
                z = X @ w
                if s["task"] == "reg":
                    y = (np.exp(0.5 * z) if s.get("target") else z + 0.1 * rng.standard_normal(sn)).astype(np.float32)
                else:
                    y = (z > 0).astype(np.int64)
            else:
                with np.load(os.path.join(a.data, "%s-%s.npz" % (s["block"], ds))) as z:
                    X = np.ascontiguousarray(z["X"], dtype=np.float32)
                    y = np.ascontiguousarray(z["y"])
            if s.get("target") in ("poisson", "tweedie"):
                y = np.maximum(y, 0)
            if a.rows:
                X, y = np.ascontiguousarray(X[:a.rows]), np.ascontiguousarray(y[:a.rows])
            y = y.astype(np.float32) if s["task"] == "reg" else y
            m = getattr(ml, s["ours"][0])(**params)
            t0 = time.perf_counter()
            m.fit(X, y)
            sec = time.perf_counter() - t0
            h = hashlib.sha256()
            h.update(np.asarray(m.coef_, dtype=np.float32).tobytes())
            h.update(np.atleast_1d(np.asarray(m.intercept_, dtype=np.float32)).tobytes())
            r = dict(lane=lane, dataset=ds, shape=list(X.shape), fit_s=round(sec, 3),
                     n_iter=int(getattr(m, "n_iter_", -1)), sha=h.hexdigest()[:16],
                     env={k: os.environ[k] for k in os.environ if k.startswith("MOJOLEARN_X_LINEAR")},
                     max_iter=params.get("max_iter"), set=a.set)
            print(json.dumps(r), flush=True)
            recs.append(r)
    with open(a.out, "w") as fh:
        json.dump(recs, fh, indent=1)


if __name__ == "__main__":
    main()
