"""Before/after digests for lanes mf-del5 and mf-del5b (2026-11-23): one line per configuration, outputs AND fitted
state as sha256 of raw bytes (a refusal by its words). Written, not run; the orchestrator runs it on the CPU column
for base origin/lane/member-fits-5 a327b129fc and for the head of lane/mf-del5b, then diffs the two outputs.

Section del5b_rfe (lane mf-del5b): THE RFE KIND over a META estimator, an RFE with a dotted importance_getter
('best_estimator_.coef_', 'estimator_.coef_', 'final_estimator_.coef_') used as a member of a VotingClassifier /
VotingRegressor, a GridSearchCV, a CalibratedClassifierCV and cross_val_score (before: the RFE kind answered False
and each meta kept its door units; after: the RFE kind's META state, the META estimator admitted by the work-list),
plus RFE's own door over a search classifier ('best_estimator_.coef_', whose estimator type now comes from the
META kind's mask). EXPECTED: equal outputs for every served case (the same RFE RUN `rfe_fit_run`, the same rounds
and member fits); `*_raised` lines may change wording only where a door refusal became the member's refusal of the
same input.

Section del5 (lane mf-del5, no digest script of its own): CalibratedClassifierCV with LeaveOneOut and with
cv > class count (the refusal words), and a stratified splitter on a regressor (StackingRegressor, GridSearchCV
over Ridge, RidgeCV) on targets with repeated values (folds) and continuous targets (the refusal words). EXPECTED:
equal, except `*_raised` lines that came from the door before and from the member's RUN now (same words).

Configurations are independent: one process each across the host cores.

    python tools/mf_del5_digests.py > digests.txt
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


def _data(seed, n, d, k):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d), dtype=np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    s = X @ w
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return X, np.searchsorted(edges, s).astype(np.int64), (s + 0.1 * rng.standard_normal(n)).astype(np.float32)


def _dotted_rfe(getter, regression):
    import mojolearn
    from mojolearn.feature_selection import RFE
    from mojolearn.model_selection import GridSearchCV
    if getter == "best_estimator_.coef_":
        base = mojolearn.Ridge(alpha=1.0) if regression else mojolearn.LogisticRegression(max_iter=200)
        grid = {"alpha": [0.5, 1.0]} if regression else {"C": [0.5, 1.0]}
        inner = GridSearchCV(base, grid, cv=3)
    elif getter == "estimator_.coef_":
        base = mojolearn.Ridge(alpha=1.0) if regression else mojolearn.LogisticRegression(max_iter=200)
        inner = RFE(base, n_features_to_select=6)
    else:
        if regression:
            inner = mojolearn.StackingRegressor([("r", mojolearn.Ridge(alpha=1.0))],
                                                final_estimator=mojolearn.Ridge(alpha=0.5), cv=3)
        else:
            inner = mojolearn.StackingClassifier([("lr", mojolearn.LogisticRegression(max_iter=200))],
                                                 final_estimator=mojolearn.LogisticRegression(max_iter=200), cv=3)
    return RFE(inner, n_features_to_select=3, importance_getter=getter)


def _del5b_rfe(arg):
    _, meta, getter, regression = arg
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    from mojolearn.model_selection import GridSearchCV, cross_val_score
    X, y, t = _data(23, 180, 6, 3)
    target = t if regression else y
    tag = f"del5b_rfe_{meta}_{getter.split('.')[0]}_{'reg' if regression else 'cls'}"
    out = []
    try:
        if meta == "door":
            m = _dotted_rfe(getter, regression).fit(X, target)
            out.append((tag + "_support", _digest(np.asarray(m.support_, dtype=bool))))
            out.append((tag + "_ranking", _digest(m.ranking_)))
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif meta == "vote":
            cls = mojolearn.VotingRegressor if regression else mojolearn.VotingClassifier
            m = cls([("a", _dotted_rfe(getter, regression)), ("b", mojolearn.Ridge(alpha=1.0) if regression
                                                                   else mojolearn.LogisticRegression(max_iter=200))])
            m.fit(X, target)
            out.append((tag + "_predict", _digest(m.predict(X))))
            sub = m.estimators_[0]
            out.append((tag + "_member_support", _digest(np.asarray(sub.support_, dtype=bool))))
            out.append((tag + "_member_ranking", _digest(sub.ranking_)))
        elif meta == "cal":
            m = CalibratedClassifierCV(_dotted_rfe(getter, False), cv=3).fit(X, y)
            out.append((tag + "_proba", _digest(m.predict_proba(X))))
        elif meta == "search":
            m = GridSearchCV(_dotted_rfe(getter, regression), {"n_features_to_select": [2, 4]}, cv=3).fit(X, target)
            out.append((tag + "_scores", _digest(np.asarray(m.cv_results_["mean_test_score"], dtype=np.float64))))
            out.append((tag + "_best_support", _digest(np.asarray(m.best_estimator_.support_, dtype=bool))))
            out.append((tag + "_predict", _digest(m.predict(X))))
        else:
            got = cross_val_score(_dotted_rfe(getter, regression), X, target, cv=3)
            out.append((tag + "_cv", _digest(np.asarray(got, dtype=np.float64))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _del5(arg):
    _, case = arg
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    from mojolearn.model_selection import GridSearchCV, KFold, LeaveOneOut, StratifiedKFold
    X, y, t = _data(29, 120, 5, 3)
    tag = f"del5_{case}"
    out = []
    try:
        repeated = np.round(t * 2.0).astype(np.float32)
        if case == "cal_loo":
            m = CalibratedClassifierCV(mojolearn.LogisticRegression(max_iter=200), cv=LeaveOneOut()).fit(X, y)
            out.append((tag + "_proba", _digest(m.predict_proba(X))))
        elif case == "cal_too_many":
            m = CalibratedClassifierCV(mojolearn.LogisticRegression(max_iter=200), cv=200).fit(X, y)
            out.append((tag + "_proba", _digest(m.predict_proba(X))))
        elif case in ("stack_repeated", "stack_continuous"):
            target = repeated if case == "stack_repeated" else t
            m = mojolearn.StackingRegressor([("r", mojolearn.Ridge(alpha=1.0))], cv=StratifiedKFold(2)).fit(X, target)
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif case in ("search_repeated", "search_continuous"):
            target = repeated if case == "search_repeated" else t
            m = GridSearchCV(mojolearn.Ridge(), {"alpha": [0.5, 1.0]}, cv=StratifiedKFold(2)).fit(X, target)
            out.append((tag + "_scores", _digest(np.asarray(m.cv_results_["mean_test_score"], dtype=np.float64))))
        elif case == "ridgecv_kfold":
            m = mojolearn.RidgeCV(alphas=(0.5, 1.0), cv=KFold(3)).fit(X, t)
            out.append((tag + "_coef", _digest(m.coef_)))
            out.append((tag + "_alpha", _digest(np.asarray(m.alpha_, dtype=np.float64))))
        else:
            m = mojolearn.RidgeCV(alphas=(0.5, 1.0), cv=StratifiedKFold(2)).fit(X, repeated)
            out.append((tag + "_coef", _digest(m.coef_)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _unit(arg):
    if arg[0] == "del5b_rfe":
        return _del5b_rfe(arg)
    return _del5(arg)


GETTERS = ("best_estimator_.coef_", "estimator_.coef_", "final_estimator_.coef_")
UNITS = ([("del5b_rfe", m, g, r) for m in ("door", "vote", "search", "cv") for g in GETTERS for r in (False, True)]
         + [("del5b_rfe", "cal", g, False) for g in GETTERS]
         + [("del5", c) for c in ("cal_loo", "cal_too_many", "stack_repeated", "stack_continuous", "search_repeated",
                                  "search_continuous", "ridgecv_kfold", "ridgecv_strat_repeated")])


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
