"""Before/after digests of the meta-estimators' member fits with no fit door left (lane mf-del8, 2026-11-23). One line
per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is digested by
its exception type and words. Base: origin/lane/member-fits-8 f1385f8f1c (a member the native set did not take, a
draw or fold lacking a class, a non-finite X or y fell to the member's Python fit door: `_fit_member`,
`_fit_units_native` False, Bagging's `native = False` loop, CalibratedClassifierCV's `_CalUnits`), head lane/mf-del8
(no door: THE UNIT CLASSES re-code, native refusals). Test tool, never product runtime.

Units:
  * bag_*: BaggingClassifier(DecisionTreeClassifier) on an imbalanced 3-class y with small bootstrap draws (several
    draws lack a class) x max_samples {6, 20} x n_jobs {1, 4}: predict_proba, predict, each member's classes_ and
    its tree's feature importances; BaggingRegressor over a NaN X and a NaN y (refusal words).
  * vote_*: VotingClassifier soft/hard and VotingRegressor over DecisionTree / LogisticRegression / Ridge members:
    predict(_proba); VotingRegressor refusals: NaN X, inf y, y of another length.
  * stack_*: StackingClassifier with an unshuffled KFold over sorted 3-class y (every train fold lacks a class):
    its fitted final estimator's coef_ or its refusal words; with shuffled StratifiedKFold: predict_proba.
  * cal_*: CalibratedClassifierCV over LogisticRegression / DecisionTreeClassifier / GaussianNB, sigmoid and
    isotonic, ensemble True / False: predict_proba.
  * mor_*: MultiOutputRegressor(Ridge) and (DecisionTreeRegressor) over a 2-output y: predict.

EXPECTED, host column:
  * every vote_ / cal_ / mor_ / stack_ (StratifiedKFold) line equal base vs head (they were native members before);
  * bag_ lines: equal base vs head (the lacking draws were door-fitted on their drawn classes; now natively re-coded
    onto the same classes with the same rows: the same tree);
  * stack_ (unshuffled KFold) lines: equal words or equal coef_ (the door fitted the lacking folds; the native
    re-code fits the same classes);
  * the refusal lines: the words may change: base raised the member fit door's words (or the forest door's), head
    the set's (TREES_NONFINITE for Bagging X, scikit-learn's check_array words for y and for Voting, scikit-learn's
    check_consistent_length words for a y of another length). Report each changed refusal.
L40S (device image): the same lines equal to the host column (bits must match across vendors and the host column).
Configurations are independent: one process each across the allocated cores.

    python tools/mf_del8_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _imbalanced(n=60, d=4, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    y[:3] = 2
    y[3:20] = 1
    X[:, 0] += y
    return X, y


def _regression(n=50, d=4, seed=9, outputs=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d, outputs)).astype(np.float32)
    y = X @ w
    return X, (y[:, 0] if outputs == 1 else y)


def _bag(samples, jobs):
    import mojolearn as ml
    X, y = _imbalanced()
    m = ml.BaggingClassifier(ml.DecisionTreeClassifier(random_state=0), n_estimators=10, max_samples=samples,
                             random_state=0, n_jobs=jobs).fit(X, y)
    out = [("proba", m.predict_proba(X)), ("predict", m.predict(X))]
    for i, est in enumerate(m.estimators_):
        out.append((f"m{i}_classes", np.asarray(est.classes_)))
        out.append((f"m{i}_importances", np.asarray(est.feature_importances_)))
    return out


def _bag_refusal(which):
    import mojolearn as ml
    X, y = _regression()
    if which == "x":
        X[3, 1] = np.nan
    else:
        y = y.copy()
        y[4] = np.nan
    ml.BaggingRegressor(ml.DecisionTreeRegressor(), n_estimators=3, random_state=0).fit(X, y)
    return []


def _vote(kind):
    import mojolearn as ml
    if kind == "reg":
        X, y = _regression()
        m = ml.VotingRegressor([("t", ml.DecisionTreeRegressor(random_state=0)), ("r", ml.Ridge())]).fit(X, y)
        return [("predict", m.predict(X))]
    X, y = _imbalanced()
    m = ml.VotingClassifier([("t", ml.DecisionTreeClassifier(random_state=0)), ("l", ml.LogisticRegression())],
                            voting=kind).fit(X, y)
    out = [("predict", m.predict(X))]
    if kind == "soft":
        out.append(("proba", m.predict_proba(X)))
    return out


def _vote_refusal(which):
    import mojolearn as ml
    X, y = _regression()
    if which == "x":
        X[2, 0] = np.nan
    elif which == "y":
        y = y.copy()
        y[1] = np.inf
    else:
        y = y[:-1]
    ml.VotingRegressor([("t", ml.DecisionTreeRegressor(random_state=0))]).fit(X, y)
    return []


def _stack(cv_kind):
    import mojolearn as ml
    from mojolearn.model_selection import KFold, StratifiedKFold
    X, y = _imbalanced()
    order = np.argsort(y, kind="stable")
    X, y = X[order], y[order]
    cv = KFold(3) if cv_kind == "kfold" else StratifiedKFold(3, shuffle=True, random_state=0)
    m = ml.StackingClassifier([("t", ml.DecisionTreeClassifier(random_state=0)), ("l", ml.LogisticRegression())],
                              final_estimator=ml.LogisticRegression(), cv=cv).fit(X, y)
    return [("coef", np.asarray(m.final_estimator_.coef_)), ("proba", m.predict_proba(X))]


def _cal(base, method, ensemble):
    import mojolearn as ml
    X, y = _imbalanced(n=90)
    b = {"lr": ml.LogisticRegression, "dt": lambda: ml.DecisionTreeClassifier(random_state=0),
         "gnb": ml.GaussianNB}[base]()
    m = ml.CalibratedClassifierCV(b, method=method, cv=3, ensemble=ensemble).fit(X, y)
    return [("proba", m.predict_proba(X))]


def _mor(base):
    import mojolearn as ml
    X, y = _regression(outputs=2)
    b = ml.Ridge() if base == "ridge" else ml.DecisionTreeRegressor(random_state=0)
    m = ml.MultiOutputRegressor(b).fit(X, y)
    return [("predict", m.predict(X))]


CONFIGS = (
    [("bag", (s, j)) for s in (6, 20) for j in (1, 4)]
    + [("bag_refusal", (w,)) for w in ("x", "y")]
    + [("vote", (k,)) for k in ("soft", "hard", "reg")]
    + [("vote_refusal", (w,)) for w in ("x", "y", "length")]
    + [("stack", (c,)) for c in ("kfold", "stratified")]
    + [("cal", (b, m, e)) for b in ("lr", "dt", "gnb") for m in ("sigmoid", "isotonic") for e in (True, False)]
    + [("mor", (b,)) for b in ("ridge", "tree")]
)
RUNS = {"bag": _bag, "bag_refusal": _bag_refusal, "vote": _vote, "vote_refusal": _vote_refusal, "stack": _stack,
        "cal": _cal, "mor": _mor}


def _unit(config):
    kind, args = config
    name = "del8_" + kind + "_" + "_".join(str(a) for a in args)
    out = []
    try:
        for suffix, value in RUNS[kind](*args):
            out.append((f"{name}_{suffix}", _digest(value)))
        if not out:
            out.append((name + "_fitted", "no refusal"))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _workers(units):
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    return max(1, min(units, cores))


def main():
    with ProcessPoolExecutor(max_workers=_workers(len(CONFIGS))) as pool:
        for lines in pool.map(_unit, CONFIGS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
