"""Before/after digests of AdaBoost fit, the operation lane mf-ada split into UNPACK / typed RUN (`ada_fit_run`) /
PACK. Host column. One line per configuration: name, then the sha256 of the fitted state (estimator_weights_,
estimator_errors_, every member's five tree arrays and classes) and of the outputs (predict; predict_proba and
decision_function for a classifier), dtype and shape included; a refusal is digested by its words. Test tool, never
product runtime.

Configurations: AdaBoostClassifier x base {default stump, DecisionTreeClassifier(max_depth=3),
DecisionTreeClassifier(splitter='random', max_depth=3), ExtraTreesClassifier(n_estimators=3, max_depth=3)} x
sample_weight {None, given} x class_weight {None, 'balanced'} (on the base) x n_estimators {1, 25}; AdaBoostRegressor
x base {default depth-3 tree, DecisionTreeRegressor(max_depth=5), splitter='random', ExtraTreesRegressor} x
sample_weight x loss {linear, square, exponential} x n_estimators {1, 25}; plus the refusals (worse than random, a
NaN in X, a non-tree base). The DecisionTree* splitter='best' rows take `ada_fit_run`; the others keep the door
rounds (the same digests are expected for both).

Base: origin/lane/member-fits 4d76a9e16e; head: lane/mf-ada. Every line is expected equal.

    python tools/mf_ada_digests.py > digests.txt

The configurations are independent and run one per process across the host's cores (`os.cpu_count()` workers;
each fit is small, a few MB).
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import os
import sys

import numpy as np


def _digest(*values):
    h = hashlib.sha256()
    for value in values:
        a = np.asarray(value)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(seed=11, n=257, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = (X @ w + 0.3 * rng.normal(size=n)).astype(np.float32)
    edges = np.quantile(y, [1 / 3, 2 / 3])
    c = np.searchsorted(edges, y)
    sw = rng.random(n) + 0.25
    return X, y, c, sw


def _base(ml, classifier, which, class_weight):
    if classifier:
        kw = {} if class_weight is None else {"class_weight": class_weight}
        if which == "default":
            return ml.DecisionTreeClassifier(max_depth=1, **kw) if kw else None
        if which == "depth3":
            return ml.DecisionTreeClassifier(max_depth=3, **kw)
        if which == "random":
            return ml.DecisionTreeClassifier(splitter="random", max_depth=3, random_state=0, **kw)
        return ml.ExtraTreesClassifier(n_estimators=3, max_depth=3, random_state=0, **kw)
    if which == "default":
        return None
    if which == "depth3":
        return ml.DecisionTreeRegressor(max_depth=5)
    if which == "random":
        return ml.DecisionTreeRegressor(splitter="random", max_depth=3, random_state=0)
    return ml.ExtraTreesRegressor(n_estimators=3, max_depth=3, random_state=0)


def _state(m, classifier):
    parts = [np.asarray(m.estimator_weights_, np.float64), np.asarray(m.estimator_errors_, np.float64)]
    for e in m.estimators_:
        f = e.__dict__
        for key in ("_offsets", "_colid", "_quesval", "_left_child", "_leaves"):
            parts.append(np.asarray(f[key]))
        if classifier:
            parts.append(np.asarray(f["classes_"]))
    return _digest(*parts)


def _case(job):
    import mojolearn as ml
    classifier, which, weighted, option, n_estimators = job
    name = (f"{'c' if classifier else 'r'}_{which}_{'w' if weighted else 'u'}_{option}_{n_estimators}")
    X, y, c, sw = _data()
    Xt = _data(12, n=97)[0]
    try:
        if classifier:
            base = _base(ml, True, which, None if option == "none" else "balanced")
            m = ml.AdaBoostClassifier(base, n_estimators=n_estimators, random_state=3)
            m.fit(X, c, sample_weight=sw if weighted else None)
            out = _digest(m.predict(Xt), m.predict_proba(Xt), m.decision_function(Xt))
        else:
            base = _base(ml, False, which, None)
            m = ml.AdaBoostRegressor(base, n_estimators=n_estimators, loss=option, random_state=3)
            m.fit(X, y, sample_weight=sw if weighted else None)
            out = _digest(m.predict(Xt))
        return f"{name} n={len(m.estimators_)} state={_state(m, classifier)} out={out}"
    except Exception as e:  # noqa: BLE001 - the words are the digest
        return f"{name} raised {type(e).__name__}: {e}"


def _refusal(job):
    import mojolearn as ml
    X, y, c, _ = _data()
    try:
        if job == "worse":
            ml.AdaBoostClassifier(n_estimators=3).fit(np.zeros((8, 2), np.float32), np.array([0, 1] * 4))
        elif job == "nan_c" or job == "nan_r":
            bad = X.copy()
            bad[5, 1] = np.nan
            if job == "nan_c":
                ml.AdaBoostClassifier(n_estimators=3).fit(bad, c)
            else:
                ml.AdaBoostRegressor(n_estimators=3).fit(bad, y)
        else:
            ml.AdaBoostClassifier(ml.LogisticRegression(), n_estimators=3).fit(X, c)
        return f"refuse_{job} no-refusal"
    except Exception as e:  # noqa: BLE001 - the words are the digest
        return f"refuse_{job} {type(e).__name__}: {e}"


def _run(job):
    return _refusal(job[1]) if job[0] == "refusal" else _case(job[1])


def main():
    bases = ("default", "depth3", "random", "extra")
    jobs = [("case", (True, b, w, cw, m))
            for b, w, cw, m in itertools.product(bases, (False, True), ("none", "balanced"), (1, 25))]
    jobs += [("case", (False, b, w, loss, m))
             for b, w, loss, m in itertools.product(bases, (False, True), ("linear", "square", "exponential"),
                                                    (1, 25))]
    jobs += [("refusal", r) for r in ("worse", "nan_c", "nan_r", "nontree")]
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        for line in pool.map(_run, jobs):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
