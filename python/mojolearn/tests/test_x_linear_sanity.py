# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE'S SANITY CHECK against scikit-learn (lane/algos-linear).

    cd python && python -m mojolearn.tests.test_x_linear_sanity [name ...]

A tolerance check on small data, not an identity claim: each case fits the
mojolearn estimator and scikit-learn's on the same data and compares the
quantity a user reads (score, coefficients or predictions) within a stated
tolerance. NumPy and scikit-learn are test oracles here only.
"""
import sys

import numpy as np

import mojolearn as ml

CASES = {}


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def _data(n=600, d=8, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    yr = (X @ w + noise * rng.standard_normal(n)).astype(np.float32)
    yc = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
    y3 = np.digitize(X[:, 0] + 0.3 * X[:, 2], [-0.5, 0.5]).astype(np.int64)
    return X, yr, yc, y3


def _close(tag, ours, theirs, tol):
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    err = float(np.max(np.abs(ours - theirs))) if ours.size else 0.0
    ok = err <= tol
    print(f"  {tag}: max|diff| {err:.4g} (tol {tol}) {'ok' if ok else 'FAIL'}")
    return ok


@case("sgd")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data()
    ok = True
    for loss in ("hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"):
        for y in (yc, y3):
            a = ml.SGDClassifier(loss=loss, random_state=0).fit(X, y).score(X, y)
            b = sk.SGDClassifier(loss=loss, random_state=0).fit(X, y).score(X, y)
            ok &= _close(f"SGDClassifier {loss} k={len(set(y))} accuracy {a:.3f} vs {b:.3f}", a, b, 0.06)
    for loss in ("squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"):
        for pen in ("l2", "l1", "elasticnet"):
            a = ml.SGDRegressor(loss=loss, penalty=pen, random_state=0).fit(X, yr)
            b = sk.SGDRegressor(loss=loss, penalty=pen, random_state=0).fit(X, yr)
            ok &= _close(f"SGDRegressor {loss}/{pen} R2 {a.score(X, yr):.3f} vs {b.score(X, yr):.3f}",
                         a.score(X, yr), b.score(X, yr), 0.05)
    return ok


def main(argv):
    names = argv or list(CASES)
    bad = []
    for name in names:
        print(f"[{name}]")
        if not CASES[name]():
            bad.append(name)
    print("SANITY:", "PASS" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
