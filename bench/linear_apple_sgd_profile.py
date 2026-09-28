# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Where the SGD family's GPU time goes (lane linear-apple2, the Metal profile).

Fits SGDClassifier on HIGGS (100k x 28 by default) on the GPU binding with one
part of the row pass switched off at a time (the shuffle, the penalty, the
per-row learning rate, the objective's stopping test), plus one epoch against
five, so the difference of two lines is that part's cost. Every line prints
a digest of coef_ and intercept_.

    python bench/linear_apple_sgd_profile.py [--rows 100000] [--column gpu|host]

Lines: `SGDPROF <case> fit=<s> <digest>`.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from x_linear_speed import _load, _pin, _modules, _digest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--column", default="gpu", choices=["gpu", "host"])
    a = ap.parse_args()
    d = _load(a.rows)
    import mojolearn._expansion_linear as lm
    X, y = d["hx"], d["hy"]
    base = dict(max_iter=5, tol=None, random_state=0)
    cases = [
        ("base", {}),
        ("epoch1", dict(max_iter=1)),
        ("noshuffle", dict(shuffle=False)),
        ("nopenalty", dict(penalty=None)),
        ("constlr", dict(learning_rate="constant", eta0=0.01)),
        ("bare", dict(shuffle=False, penalty=None, learning_rate="constant", eta0=0.01)),
        ("bare-epoch1", dict(max_iter=1, shuffle=False, penalty=None, learning_rate="constant", eta0=0.01)),
        ("tol", dict(tol=1e-3, n_iter_no_change=1000)),
    ]
    mod = _modules(a.column)
    for name, kw in cases:
        args = dict(base)
        args.update(kw)
        est = _pin(lm.SGDClassifier(**args), mod)
        t0 = time.perf_counter()
        est.fit(X, y)
        t = time.perf_counter() - t0
        print(f"SGDPROF {name} {a.column} fit={t:.3f} {_digest([est.coef_, est.intercept_])}", flush=True)


if __name__ == "__main__":
    main()
