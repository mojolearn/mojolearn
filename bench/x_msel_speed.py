# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Model selection speed board with PHASES (lane metrics-apple3).

Times cross validation, the searches and the curves over tree and classical
estimators at the timing floor (1M rows, taxi + HIGGS from R2), under the
numeric mode of the environment (MOJOLEARN_NUMERIC_MODE). Every case prints
its wall seconds, a digest of its result, its scores, and the seconds spent
in each phase (exclusive of the phases nested inside it):

    folds       check_cv + the splitter + index validation
    take_rows   the per-fold row gathers of X and y
    clone       estimator clones
    fit         estimator.fit
    predict     predict / predict_proba / decision_function
    score       estimator.score or the scorer, less its predict
    other       the rest of the wall time (Python bookkeeping)

    python bench/x_msel_speed.py [--rows 1000000] [--reps 1] [--only a,b] [--cprofile N]

Lines: `XMSEL <case> <seconds> <digest> | phase=seconds(calls) ...`,
`XMSEL-SCORE <case> <values>`, `XMSEL-FIT <estimator> first=<s> next=<s>`
(a standalone fit on the training share, first call and the best later
call: their difference is the one-time cost of the binding and its
pipelines), `XMSEL-TOTAL <seconds>`.
"""
import argparse
import collections
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from x_metrics_speed import _digest, _load  # noqa: E402


class Phases:
    def __init__(self):
        self.t = collections.defaultdict(float)
        self.c = collections.Counter()
        self.stack = []

    def reset(self):
        self.t.clear()
        self.c.clear()
        del self.stack[:]

    def wrap(self, name, fn):
        ph = self

        def timed(*a, **k):
            ph.stack.append(0.0)
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                inner = ph.stack.pop()
                ph.t[name] += dt - inner
                ph.c[name] += 1
                if ph.stack:
                    ph.stack[-1] += dt
        timed.__wrapped__ = fn
        timed.__name__ = getattr(fn, "__name__", name)
        return timed

    def line(self, wall):
        known = sum(self.t.values())
        parts = ["%s=%.4f(%d)" % (k, self.t[k], self.c[k]) for k in sorted(self.t, key=lambda k: -self.t[k])]
        parts.append("other=%.4f" % (wall - known))
        return " ".join(parts)


PH = Phases()


def instrument(S, classes):
    for name, phase in (("_take_rows", "take_rows"), ("_cv_folds", "folds"), ("_prepare_folds", "folds"),
                        ("_clone", "clone"), ("_score", "score"), ("_fit_score_fold", "fold_other")):
        if hasattr(S, name):
            setattr(S, name, PH.wrap(phase, getattr(S, name)))
    for cls in classes:
        for meth, phase in (("fit", "fit"), ("predict", "predict"), ("predict_proba", "predict"),
                            ("decision_function", "predict"), ("score", "score")):
            fn = cls.__dict__.get(meth)
            if fn is None:
                for base in cls.__mro__[1:]:
                    if meth in base.__dict__:
                        fn = base.__dict__[meth]
                        break
            if fn is None or not callable(fn):
                continue
            setattr(cls, meth, PH.wrap(phase, fn))


def estimators(ml):
    def mk(name, **kw):
        cls = getattr(ml, name, None)
        if cls is None:
            return None
        return lambda: cls(**kw)
    return dict(
        ridge=mk("Ridge"),
        logreg=mk("LogisticRegression"),
        gnb=mk("GaussianNB"),
        dtc=mk("DecisionTreeClassifier", max_depth=8),
        rfc=mk("RandomForestClassifier", n_estimators=10, max_depth=8),
        gbr=mk("GradientBoostingRegressor", n_estimators=20, max_depth=4),
    )


def _scores(v):
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x):
                if "time" in str(k) or k == "params":
                    continue
                walk(x[k])
            return
        if isinstance(x, (list, tuple)):
            for e in x:
                walk(e)
            return
        try:
            a = np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, dtype=np.float64).ravel()
        except Exception:
            return
        out.extend(a[:12].tolist())
    walk(v)
    return " ".join("%.6g" % s for s in out[:24])


def _strip(v):
    """The result without wall times (they are not part of the digest)."""
    if isinstance(v, dict):
        return {k: _strip(x) for k, x in v.items() if "time" not in str(k) and k != "params"}
    return v


def cases(ml, S, raw, n):
    E = estimators(ml)
    tx = np.ascontiguousarray(raw["taxi"]["x"][:, :16])
    fare = raw["taxi"]["target"]
    hx = np.ascontiguousarray(raw["higgs"]["x"][:, 1:17])
    hy = raw["higgs"]["label"]
    sub = max(n // 5, 1000)
    out = []

    def add(name, need, fn):
        if all(E.get(k) is not None for k in need):
            out.append((name, fn))
        else:
            print("XMSEL %-24s   SKIPPED no estimator %s" % (name, ",".join(need)), flush=True)

    add("cvs_ridge", ["ridge"], lambda: S.cross_val_score(E["ridge"](), tx, fare, cv=5))
    add("cvs_logreg", ["logreg"], lambda: S.cross_val_score(E["logreg"](), hx, hy, cv=5))
    add("cvs_gnb", ["gnb"], lambda: S.cross_val_score(E["gnb"](), hx, hy, cv=5))
    add("cvs_dtc", ["dtc"], lambda: S.cross_val_score(E["dtc"](), hx, hy, cv=5))
    add("cvs_rfc", ["rfc"], lambda: S.cross_val_score(E["rfc"](), hx, hy, cv=5))
    add("cvs_gbr", ["gbr"], lambda: S.cross_val_score(E["gbr"](), tx, fare, cv=5))
    add("cv_multi_gnb", ["gnb"], lambda: _strip(S.cross_validate(
        E["gnb"](), hx, hy, cv=5, scoring=["roc_auc", "accuracy", "neg_log_loss"], return_train_score=True)))
    add("cv_shuffle_ridge", ["ridge"], lambda: _strip(S.cross_validate(
        E["ridge"](), tx, fare, cv=S.KFold(5, shuffle=True, random_state=0),
        scoring=["r2", "neg_mean_absolute_error"])))
    add("grid_ridge", ["ridge"], lambda: _strip(S.GridSearchCV(
        E["ridge"](), {"alpha": [0.01, 0.1, 1.0, 10.0]}, cv=5).fit(tx, fare).cv_results_))
    add("grid_dtc_auc", ["dtc"], lambda: _strip(S.GridSearchCV(
        E["dtc"](), {"max_depth": [4, 8]}, cv=3, scoring="roc_auc").fit(hx, hy).cv_results_))
    add("rand_gbr", ["gbr"], lambda: _strip(S.RandomizedSearchCV(
        E["gbr"](), {"max_depth": [2, 3, 4], "n_estimators": [5, 10]}, n_iter=4, cv=3,
        random_state=0).fit(tx, fare).cv_results_))
    add("valcurve_ridge", ["ridge"], lambda: S.validation_curve(
        E["ridge"](), tx, fare, param_name="alpha", param_range=[0.01, 0.1, 1.0, 10.0], cv=5))
    add("learncurve_ridge", ["ridge"], lambda: S.learning_curve(E["ridge"](), tx, fare, cv=5))
    add("learncurve_shuf_gnb", ["gnb"], lambda: S.learning_curve(
        E["gnb"](), hx, hy, cv=5, shuffle=True, random_state=0))
    add("cvpredict_ridge", ["ridge"], lambda: S.cross_val_predict(E["ridge"](), tx, fare, cv=5))
    add("cvpredict_proba_gnb", ["gnb"], lambda: S.cross_val_predict(
        E["gnb"](), hx[:sub], hy[:sub], cv=5, method="predict_proba"))
    add("permtest_gnb", ["gnb"], lambda: S.permutation_test_score(
        E["gnb"](), hx, hy, cv=5, n_permutations=5, random_state=0))
    fits = dict(ridge=(tx, fare), logreg=(hx, hy), gnb=(hx, hy), dtc=(hx, hy), rfc=(hx, hy), gbr=(tx, fare))
    return out, E, fits


def _profile(name, fn, top):
    import cProfile
    import io
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    try:
        fn()
    finally:
        pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top)
    for line in buf.getvalue().splitlines():
        if line.strip():
            print("XMPROFILE %s | %s" % (name, line), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--cprofile", type=int, default=0)
    ap.add_argument("--fits", type=int, default=1, help="1: also time a standalone fit per estimator")
    ap.add_argument("--warm", type=int, default=1, help="1: one untimed run of each case first")
    a = ap.parse_args()
    t0 = time.time()
    raw = _load(a.rows)
    import mojolearn as ml
    import mojolearn.model_selection as S
    print("XMSEL-DATA rows=%d load_s=%.2f mode=%s vendor=%s msel3=%s" % (
        a.rows, time.time() - t0, os.environ.get("MOJOLEARN_NUMERIC_MODE", "default"),
        os.environ.get("MOJOLEARN_VENDOR", "auto"), os.environ.get("MOJOLEARN_MSEL3", "")), flush=True)
    cs, E, fits = cases(ml, S, raw, a.rows)
    only = set(x for x in a.only.split(",") if x)
    if a.fits:
        m = a.rows * 4 // 5
        for k in sorted(E):
            if E[k] is None or (only and not any(k in c for c in only)):
                continue
            X, y = fits[k]
            Xs, ys = np.ascontiguousarray(X[:m]), np.ascontiguousarray(y[:m])
            try:
                ts = []
                for _ in range(3):
                    est = E[k]()
                    t = time.perf_counter()
                    est.fit(Xs, ys)
                    ts.append(time.perf_counter() - t)
                t = time.perf_counter()
                sc = est.score(np.ascontiguousarray(X[m:]), np.ascontiguousarray(y[m:]))
                tsc = time.perf_counter() - t
                print("XMSEL-FIT %-8s first=%.4f next=%.4f score_s=%.4f score=%.6g" % (
                    k, ts[0], min(ts[1:]), tsc, float(sc)), flush=True)
                if a.cprofile:
                    _profile("fit_" + k, lambda: E[k]().fit(Xs, ys), a.cprofile)
            except Exception as e:
                print("XMSEL-FIT %-8s FAILED %s: %s" % (k, type(e).__name__, str(e)[:200]), flush=True)
                E[k] = None
    classes = []
    for k in sorted(E):
        if E[k] is not None:
            try:
                classes.append(type(E[k]()))
            except Exception:
                pass
    instrument(S, classes)
    total = 0.0
    for name, fn in cs:
        if only and name not in only:
            continue
        try:
            if a.warm:
                fn()
            best = float("inf")
            line = ""
            v = None
            for _ in range(a.reps):
                PH.reset()
                t = time.perf_counter()
                v = fn()
                dt = time.perf_counter() - t
                if dt < best:
                    best = dt
                    line = PH.line(dt)
            total += best
            print("XMSEL %-24s %9.4f %s | %s" % (name, best, _digest(v), line), flush=True)
            print("XMSEL-SCORE %-24s %s" % (name, _scores(v)), flush=True)
            if a.cprofile:
                _profile(name, fn, a.cprofile)
        except Exception as e:  # a case that fails is reported, never hidden
            print("XMSEL %-24s   FAILED %s: %s" % (name, type(e).__name__, str(e)[:200]), flush=True)
    print("XMSEL-TOTAL %.4f" % total, flush=True)


if __name__ == "__main__":
    main()
