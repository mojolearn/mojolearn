# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Per-iteration cost of the iterative linear solvers on the GPU binding
(lane linear-apple, the Metal profile).

Fits the same problem at several iteration caps with the tolerance at 1e-12, so
every fit runs exactly its cap; the slope of fit time against the cap is the
cost of one iteration (one coordinate-descent epoch, one L-BFGS iteration) and
the intercept is the fixed cost (transfers, setup, the first evaluation).

    python bench/linear_apple_profile.py [--rows 1000000] [--caps 1,2,4,8,16]

Data as bench/x_linear_speed.py (taxi 16 standardized columns, HIGGS).
Lines: `LAPROF <case> cap=<k> fit=<s>` then `LAPROF <case> per_iter=<s> fixed=<s>`.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from x_linear_speed import _load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--caps", default="1,2,4,8,16")
    a = ap.parse_args()
    d = _load(a.rows)
    import mojolearn as ML
    from mojolearn import linear_model as LM
    from mojolearn import svm as SV
    caps = [int(c) for c in a.caps.split(",")]
    cases = [
        ("lasso", lambda k: ML.Lasso(alpha=0.001, max_iter=k, tol=1e-12), d["tx"], d["fare"]),
        ("logistic", lambda k: LM.LogisticRegression(max_iter=k, tol=1e-12), d["hx"], d["hy"]),
        ("linear-svr", lambda k: SV.LinearSVR(max_iter=k, tol=1e-12), d["tx"], d["fare"]),
    ]
    for name, make, X, y in cases:
        # one warm fit (loads the binding, compiles nothing new)
        make(1).fit(X, y)
        ts = []
        for k in caps:
            t0 = time.perf_counter()
            make(k).fit(X, y)
            t = time.perf_counter() - t0
            ts.append(t)
            print(f"LAPROF {name} cap={k} fit={t:.4f}", flush=True)
        slope, icept = np.polyfit(np.asarray(caps, float), np.asarray(ts), 1)
        print(f"LAPROF {name} per_iter={slope:.5f} fixed={icept:.4f}", flush=True)


if __name__ == "__main__":
    main()
