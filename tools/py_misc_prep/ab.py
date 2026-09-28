#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane py-misc-prep A/B: the Python reference route (before) against the
native route (after) of IterativeImputer(estimator=...) and
CalibratedClassifierCV, in ONE process on one build.

    python tools/py_misc_prep/ab.py bits          every case, both routes, sha256 of every output; exit 1 on a mismatch
    python tools/py_misc_prep/ab.py time [--rows N]   the audit's shapes, each route once, seconds

The vendor is whatever the process resolves (MOJOLEARN_VENDOR=cpu for the
host bindings). The routes are switched by the modules' own flags
(`_expansion_prep._II_NATIVE`, `_expansion_trees._CAL_NATIVE`)."""
import argparse
import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, "python")
import mojolearn as ml  # noqa: E402
from mojolearn import _expansion_prep as P, _expansion_trees as T  # noqa: E402


def route(native):
    P._II_NATIVE = native
    T._CAL_NATIVE = native


def h(*arrs):
    d = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        d.update(str(a.dtype).encode() + str(a.shape).encode())
        d.update(a.tobytes())
    return d.hexdigest()[:16]


class MeanReg:
    """float32 mean of y (the lane's own user estimator)."""

    def get_params(self, deep=True):
        return {}

    def fit(self, X, y):
        self.m = np.float32(np.asarray(y, dtype=np.float32).mean(dtype=np.float64))
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.m, dtype=np.float32)


class LinReg:
    """float64 least squares (predictions float64: the clip and the float32 store matter)."""

    def __init__(self, ridge=1e-3):
        self.ridge = ridge

    def get_params(self, deep=True):
        return {"ridge": self.ridge}

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            self.w, self.b = None, float(y.mean()) if y.size else 0.0
            return self
        mu = X.mean(axis=0)
        A = X - mu
        self.w = np.linalg.solve(A.T @ A + self.ridge * np.eye(A.shape[1]), A.T @ (y - y.mean()))
        self.b = float(y.mean() - mu @ self.w)
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=np.float64)
        p = X @ self.w + self.b if self.w is not None else np.full(X.shape[0], self.b)
        if return_std:
            return p, np.full(p.shape[0], 0.5) * (1 + (np.arange(p.shape[0]) % 3))
        return p


class ListReg(LinReg):
    """Predictions as a Python list (the non-buffer route)."""

    def predict(self, X, return_std=False):
        r = super().predict(X, return_std)
        return ([float(v) for v in r[0]], list(r[1])) if return_std else [float(v) for v in r]


def with_nan(X, every=7):
    X = np.array(X, dtype=np.float32, copy=True)
    n, d = X.shape
    idx = np.arange(n * d)
    X.reshape(-1)[(idx * 2654435761 % 97) < 97 // every] = np.nan
    return X


def data(n, d, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, 1] += 0.7 * X[:, 0]
    X[:, 2] = 0.5 * X[:, 1] - X[:, 3]
    return X


def ii_cases():
    X = with_nan(data(1500, 6, 1))
    Xh = with_nan(data(300, 6, 2))
    out = {}
    for name, kw in dict(
            mean=dict(estimator=MeanReg(), max_iter=3),
            lin=dict(estimator=LinReg(), max_iter=6, min_value=-2.0, max_value=2.5),
            lin_stop=dict(estimator=LinReg(), max_iter=25, tol=5e-2),
            lin_tight=dict(estimator=LinReg(), max_iter=25, tol=1e-6),
            lst=dict(estimator=ListReg(), max_iter=4, imputation_order="descending"),
            rand=dict(estimator=LinReg(), max_iter=4, imputation_order="random", random_state=3),
            nnf=dict(estimator=LinReg(), max_iter=3, n_nearest_features=3, random_state=4),
            post=dict(estimator=LinReg(), max_iter=2, sample_posterior=True, random_state=5,
                      min_value=-1.5, max_value=1.5),
            ind=dict(estimator=LinReg(), max_iter=3, add_indicator=True, skip_complete=True),
    ).items():
        m = ml.IterativeImputer(**kw)
        a = m.fit_transform(X)
        out[f"ii/{name}"] = h(a, np.array([m.n_iter_]), m.transform(Xh))
    return out


def cal_cases():
    X = data(1200, 8, 3)
    Xh = data(300, 8, 4)
    y3 = (np.abs(X[:, 0] * 3 + X[:, 1]).astype(np.int64) % 3)
    y2 = (X[:, 0] + 0.3 * X[:, 4] > 0).astype(np.int64)
    y5 = (np.abs(X[:, 0] * 5 + X[:, 2]).astype(np.int64) % 5)
    out = {}
    for yn, y in (("b", y2), ("m3", y3), ("m5", y5)):
        for meth in ("sigmoid", "isotonic"):
            for ens in (True, False):
                m = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), method=meth, cv=3,
                                              ensemble=ens).fit(X, y)
                out[f"cal/{yn}/{meth}/{ens}"] = h(m.predict_proba(X), m.predict_proba(Xh), m.predict(Xh))
    return out


def bits():
    res = {}
    for native in (False, True):
        route(native)
        res["native" if native else "python"] = {**ii_cases(), **cal_cases()}
    bad = [k for k in res["python"] if res["python"][k] != res["native"].get(k)]
    for k in sorted(res["python"]):
        print(f"{k:28s} python {res['python'][k]} native {res['native'][k]} "
              f"{'SAME' if k not in bad else 'DIFFER'}")
    print(json.dumps({"vendor": vendor(), "cases": len(res["python"]), "differ": bad}))
    print("RESULT", "SAME" if not bad else "DIFFER")
    return 1 if bad else 0


def vendor():
    import os
    return os.environ.get("MOJOLEARN_VENDOR", "default")


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return time.perf_counter() - t, r


def time_cmd(rows):
    X = with_nan(data(rows, 10, 7))
    Xc = data(rows, 10, 8)
    yc = (np.abs(Xc[:, 0] * 10 + Xc[:, 1]).astype(np.int64) % 10)
    cal = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), method="sigmoid", cv=5).fit(Xc, yc)
    cal_i = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), method="isotonic", cv=5).fit(Xc, yc)
    rows_out = []
    for native in (False, True):
        route(native)
        tag = "after" if native else "before"
        m = ml.IterativeImputer(estimator=MeanReg(), max_iter=10, tol=0.0)
        s, a = timed(lambda: m.fit_transform(X))
        rows_out.append((f"ii_fit_transform_meanreg_{rows}x10_10rounds", tag, s, h(a, np.array([m.n_iter_]))))
        s, a = timed(lambda: m.transform(X))
        rows_out.append((f"ii_transform_{rows}x10", tag, s, h(a)))
        s, p = timed(lambda: cal.predict_proba(Xc))
        rows_out.append((f"cal_sigmoid_predict_proba_k10_cv5_{rows}", tag, s, h(p)))
        s, p = timed(lambda: cal_i.predict_proba(Xc))
        rows_out.append((f"cal_isotonic_predict_proba_k10_cv5_{rows}", tag, s, h(p)))
        c2 = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), method="sigmoid", cv=5)
        s, _ = timed(lambda: c2.fit(Xc, yc))
        rows_out.append((f"cal_sigmoid_fit_k10_cv5_{rows}", tag, s, h(c2.predict_proba(Xc[:1000]))))
    for r in rows_out:
        print(f"TIME {vendor():8s} {r[0]:44s} {r[1]:6s} {r[2]:9.3f} s  {r[3]}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("bits", "time"))
    ap.add_argument("--rows", type=int, default=1_000_000)
    a = ap.parse_args()
    sys.exit(bits() if a.cmd == "bits" else time_cmd(a.rows))
