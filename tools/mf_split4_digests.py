"""Before/after digests of lane mf-split4 (2026-11-23): the configurations that took a door path before and are
native META members (or native leaf members) now. Host column. One line per configuration and output: name and
sha256 of the output bytes (dtype and shape included); a refusal is digested by its words, and the warnings a fit
raised by their category and words, in order. Base: origin/lane/member-fits b0c2f733fd (the door paths), head
lane/mf-split4. Test tool, never product runtime.

EQUAL (every line expected):
  * `bagcw_*` (item 0): a BaggingClassifier whose DecisionTree carries a class_weight (dict keyed by codes, and
    'balanced'), on the batched route (no weights, every feature), as a member of a Voting, a Calibrated, a
    Stacking and a GridSearchCV: the base fitted it through its door (`_bagging_fit_batched`); the head through
    the kind's batched forest with the door's class rows.
  * `adacw_*` (item 0b(ii)): an AdaBoostClassifier whose DecisionTree / RandomForest base carries a class_weight,
    as a member of the same callers.
  * `adarand_*` (item 0b(i)): an AdaBoostClassifier / AdaBoostRegressor over a DecisionTree splitter='random'.
  * `nestcw_*` (item 0c(i)): a Voting / Stacking / OneVsRest whose LogisticRegression / LinearSVC member has a
    class_weight dict, inside a CalibratedClassifierCV and a GridSearchCV (the labels walk down the admission).
  * `subcls_*` (item 0c(ii)): a str subclass 'balanced' and a dict subclass class_weight on LogisticRegression,
    SGDClassifier, RidgeClassifier and SVC, inside Calibrated, Voting and cross_val_score.
  * `split_*` (item 1): a StackingClassifier with cv=KFold (plain and shuffled) or StratifiedKFold (plain,
    shuffled, and one whose least populated class is below n_splits: its UserWarning in the words), and a
    StackingRegressor with cv=KFold, as members of a Voting, a Calibrated and a GridSearchCV. A warning's ORDER
    relative to the member fits' own warnings may differ (the head raises it at PACK, first among the stack's
    notices): the `_warnings` lines list them sorted, so they are expected equal.
Lane mf-split4b (sections below SPLIT): base origin/lane/member-fits af90e5f0b6, head lane/mf-split4b. EQUAL
(every line expected):
  * `calcv_*` (item 2): a CalibratedClassifierCV with cv = an index list (all rows of 150, so it refuses inside a
    grid's 100-row folds: the same words), KFold(3), shuffled KFold and shuffled StratifiedKFold, ensemble True and
    False, as a member of a Voting, a Calibrated and a GridSearchCV (base: the door's `_splits` + `_CalUnits`; head:
    the kind's typed folds).
  * `usersplit_*` (item 1): a user's own splitter object as cv of a nested Stacking, Calibrated, GridSearchCV and
    LassoCV: a refusal, the same words (base: the door refused it; head: the admission).
  * `ridgecv_*` (item 2, RidgeCV's split path in ONE RUN with typed gathers): cv = shuffled KFold, an index list
    with weights, and a list whose second split has one test row (the undefined-R^2 warning): coef_, intercept_,
    alpha_, best_score_ and the warnings.
  * `adasw_*` (item 3): an AdaBoostClassifier / AdaBoostRegressor member fitted with sample weights inside a Voting
    and a Stacking (base: the door; head: the kind normalizing the same float32 words).
  * `svccw_*` (item 4): SVC(class_weight=3.5) inside a Calibrated, a Voting and cross_val_score: the settings
    admission's words.
  * `bagpm_*` (item 6): a BaggingClassifier whose tree has a class_weight (dict, 'balanced') on the per-member
    route (max_features=3), unweighted and weighted, in a Voting, a Calibrated, a Stacking and a GridSearchCV; and a
    dict naming a class a small draw lacks (the tree door's refusal).
Configurations are independent: one process each across the host cores.

    python tools/mf_split4_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(seed, n, k, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    s = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return X, np.searchsorted(edges, s).astype(np.int64)


class _Str(str):
    pass


class _Dict(dict):
    pass


CALLERS = ("vote", "cal", "stack", "grid")


def _in_caller(caller, est, X, y, regressor=False):
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    from mojolearn.model_selection import GridSearchCV
    if regressor:
        if caller == "vote":
            m = mojolearn.VotingRegressor([("a", est), ("b", mojolearn.Ridge())]).fit(X, y)
        else:
            m = GridSearchCV(mojolearn.VotingRegressor([("a", est)]), {"weights": [None]}, cv=3).fit(X, y)
        return [("predict", m.predict(X))]
    if caller == "vote":
        m = mojolearn.VotingClassifier([("a", est), ("b", mojolearn.LogisticRegression())], voting="soft").fit(X, y)
        return [("predict", m.predict(X)), ("proba", m.predict_proba(X))]
    if caller == "cal":
        m = CalibratedClassifierCV(est, cv=3).fit(X, y)
        return [("proba", m.predict_proba(X))]
    if caller == "stack":
        m = mojolearn.StackingClassifier([("a", est)], cv=3).fit(X, y)
        return [("proba", m.predict_proba(X))]
    m = GridSearchCV(mojolearn.VotingClassifier([("a", est)], voting="soft"), {"weights": [None]}, cv=3).fit(X, y)
    return [("mean_test", np.asarray(m.cv_results_["mean_test_score"], dtype=np.float64)),
            ("proba", m.predict_proba(X))]


def _lines(tag, call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = [(f"{tag}_{name}", _digest(np.asarray(v).astype(str) if np.asarray(v).dtype == object else v))
                   for name, v in call()]
        except Exception as e:  # noqa: BLE001 - the words are the digest
            out = [(tag + "_raised", f"{type(e).__name__}: {e}")]
    words = sorted(f"{w.category.__name__}: {w.message}" for w in caught)
    out.append((tag + "_warnings", _digest(np.asarray(words, dtype=str)) if words else "none"))
    return out


BAGCW = [(cw, c) for cw in ("dict", "balanced") for c in CALLERS]
ADACW = [(base, cw, c) for base in ("dt", "rf") for cw in ("dict", "balanced") for c in CALLERS]
ADARAND = [(kind, c) for kind in ("clf", "reg") for c in CALLERS]
NESTCW = [(meta, member, c) for meta in ("vote", "stack", "ovr") for member in ("lr", "svc") for c in ("cal", "grid")]
SUBCLS = [(member, form, c) for member in ("lr", "sgd", "ridgec", "svk") for form in ("str", "dict")
          for c in ("cal", "vote", "cvs")]
SPLITS = ("kfold", "kfold_shuffle", "strat", "strat_shuffle", "strat_least", "reg_kfold")
SPLIT = [(sp, c) for sp in SPLITS for c in ("vote", "cal", "grid")]


def _cw(form):
    return "balanced" if form == "balanced" else {0: 0.5, 2: 1.8}


def _bagcw(index):
    import mojolearn
    cw, caller = BAGCW[index]
    X, y = _data(61, 160, 3)
    est = mojolearn.BaggingClassifier(mojolearn.DecisionTreeClassifier(max_depth=4, class_weight=_cw(cw)),
                                      n_estimators=6, random_state=0)
    return _lines(f"bagcw_{cw}_{caller}", lambda: _in_caller(caller, est, X, y))


def _adacw(index):
    import mojolearn
    base, cw, caller = ADACW[index]
    X, y = _data(67, 160, 3)
    tree = (mojolearn.DecisionTreeClassifier(max_depth=1, class_weight=_cw(cw)) if base == "dt" else
            mojolearn.RandomForestClassifier(n_estimators=4, max_depth=3, class_weight=_cw(cw), random_state=1))
    est = mojolearn.AdaBoostClassifier(tree, n_estimators=5, random_state=0)
    return _lines(f"adacw_{base}_{cw}_{caller}", lambda: _in_caller(caller, est, X, y))


def _adarand(index):
    import mojolearn
    kind, caller = ADARAND[index]
    X, y = _data(71, 160, 3)
    if kind == "reg":
        if caller not in ("vote", "grid"):
            return []
        yr = (X @ np.arange(1, 6, dtype=np.float32)).astype(np.float32)
        est = mojolearn.AdaBoostRegressor(mojolearn.DecisionTreeRegressor(max_depth=3, splitter="random"),
                                          n_estimators=5, random_state=0)
        return _lines(f"adarand_reg_{caller}", lambda: _in_caller(caller, est, X, yr, regressor=True))
    est = mojolearn.AdaBoostClassifier(mojolearn.DecisionTreeClassifier(max_depth=2, splitter="random"),
                                       n_estimators=5, random_state=0)
    return _lines(f"adarand_clf_{caller}", lambda: _in_caller(caller, est, X, y))


def _nestcw(index):
    import mojolearn
    meta, member, caller = NESTCW[index]
    X, y = _data(73, 160, 3)
    leaf = (mojolearn.LogisticRegression(max_iter=200, class_weight={0: 0.5, 2: 1.8}) if member == "lr" else
            mojolearn.LinearSVC(C=0.5, class_weight={0: 0.5, 2: 1.8}))
    if meta == "vote":
        inner = mojolearn.VotingClassifier([("a", leaf), ("b", mojolearn.LogisticRegression())], voting="hard")
    elif meta == "stack":
        inner = mojolearn.StackingClassifier([("a", leaf)], cv=3)
    else:
        leaf.set_params(class_weight={0: 0.5, 1: 1.8})  # OneVsRest fits each base on its 0 / 1 column
        inner = mojolearn.OneVsRestClassifier(leaf)

    def run():
        from mojolearn.calibration import CalibratedClassifierCV
        from mojolearn.model_selection import GridSearchCV
        if caller == "cal":
            return [("proba", CalibratedClassifierCV(inner, cv=3).fit(X, y).predict_proba(X))]
        m = GridSearchCV(inner, {}, cv=3).fit(X, y)
        return [("mean_test", np.asarray(m.cv_results_["mean_test_score"], dtype=np.float64)),
                ("predict", m.predict(X))]
    return _lines(f"nestcw_{meta}_{member}_{caller}", run)


def _subcls(index):
    import mojolearn
    from mojolearn.model_selection import cross_val_score
    member, form, caller = SUBCLS[index]
    X, y = _data(79, 150, 3)
    cw = _Str("balanced") if form == "str" else _Dict({0: 0.5, 2: 1.8})
    if member == "lr":
        est = mojolearn.LogisticRegression(max_iter=200, class_weight=cw)
    elif member == "sgd":
        est = mojolearn.SGDClassifier(max_iter=20, random_state=3, class_weight=cw)
    elif member == "ridgec":
        est = mojolearn.RidgeClassifier(alpha=0.7, class_weight=cw)
    else:
        est = mojolearn.SVC(C=1.0, gamma=0.5, class_weight=cw)
    if caller == "cvs":
        return _lines(f"subcls_{member}_{form}_cvs", lambda: [("scores", cross_val_score(est, X, y, cv=3))])
    if caller == "vote":
        return _lines(f"subcls_{member}_{form}_vote", lambda: [
            ("predict", mojolearn.VotingClassifier([("a", est), ("b", mojolearn.LogisticRegression())],
                                                   voting="hard").fit(X, y).predict(X))])
    return _lines(f"subcls_{member}_{form}_cal", lambda: _in_caller("cal", est, X, y))


def _split(index):
    import mojolearn
    from mojolearn.model_selection import KFold, StratifiedKFold
    sp, caller = SPLIT[index]
    if sp == "strat_least":
        X, y = _data(83, 120, 3)
        y[:118] = np.where(y[:118] == 2, 1, y[:118])  # class 2 keeps two rows: below n_splits=3
        y[118:] = 2
    else:
        X, y = _data(83, 150, 3)
    cv = {"kfold": lambda: KFold(3), "kfold_shuffle": lambda: KFold(3, shuffle=True, random_state=4),
          "strat": lambda: StratifiedKFold(3), "strat_shuffle": lambda: StratifiedKFold(3, shuffle=True, random_state=4),
          "strat_least": lambda: StratifiedKFold(3), "reg_kfold": lambda: KFold(3, shuffle=True, random_state=4)}[sp]()
    if sp == "reg_kfold":
        if caller == "cal":
            return []
        yr = (X @ np.arange(1, 6, dtype=np.float32)).astype(np.float32)
        est = mojolearn.StackingRegressor([("a", mojolearn.Ridge()), ("b", mojolearn.Lasso(alpha=0.1))], cv=cv)
        return _lines(f"split_{sp}_{caller}", lambda: _in_caller(caller, est, X, yr, regressor=True))
    est = mojolearn.StackingClassifier([("a", mojolearn.LogisticRegression(max_iter=200)),
                                        ("b", mojolearn.DecisionTreeClassifier(max_depth=3))], cv=cv)
    return _lines(f"split_{sp}_{caller}", lambda: _in_caller(caller, est, X, y))


class _UserSplit:
    """A user's own splitter object: never called by Mojolearn (refused)."""

    def get_n_splits(self, X=None, y=None, groups=None):
        return 3

    def split(self, X, y=None, groups=None):
        n = len(X)
        idx = np.arange(n)
        for f in range(3):
            yield idx[idx % 3 != f], idx[idx % 3 == f]


CALCV_FORMS = ("list", "kfold", "kfold_shuffle", "strat_shuffle")
CALCV = [(form, ens, c) for form in CALCV_FORMS for ens in (True, False) for c in ("vote", "cal", "grid")]
USERSPLIT = [(meta, c) for meta in ("stack", "cal", "grid", "lassocv") for c in ("vote", "cal")]
RIDGECV = [("kfold_shuffle",), ("list_weighted",), ("list_one_test",)]
ADASW = [(kind, c) for kind in ("clf", "reg") for c in ("vote", "stack")]
SVCCW = [(c,) for c in ("cal", "vote", "cvs")]
BAGPM = [(cw, w, c) for cw in ("dict", "balanced") for w in (False, True) for c in CALLERS] + [("absent", False, "vote")]


def _calcv(index):
    from mojolearn.calibration import CalibratedClassifierCV
    from mojolearn.model_selection import KFold, StratifiedKFold
    import mojolearn
    form, ens, caller = CALCV[index]
    X, y = _data(89, 150, 3)
    idx = np.arange(150)
    cv = {"list": lambda: [(idx[idx % 3 != f], idx[idx % 3 == f]) for f in range(3)],
          "kfold": lambda: KFold(3), "kfold_shuffle": lambda: KFold(3, shuffle=True, random_state=5),
          "strat_shuffle": lambda: StratifiedKFold(3, shuffle=True, random_state=5)}[form]()
    est = CalibratedClassifierCV(mojolearn.LogisticRegression(max_iter=200), cv=cv, ensemble=ens)
    return _lines(f"calcv_{form}_{'ens' if ens else 'one'}_{caller}", lambda: _in_caller(caller, est, X, y))


def _usersplit(index):
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    from mojolearn.model_selection import GridSearchCV
    meta, caller = USERSPLIT[index]
    X, y = _data(97, 150, 3)
    if meta == "lassocv":
        yr = (X @ np.arange(1, 6, dtype=np.float32)).astype(np.float32)
        est = mojolearn.LassoCV(cv=_UserSplit())
        if caller != "vote":
            return []
        return _lines(f"usersplit_{meta}_{caller}", lambda: _in_caller("vote", est, X, yr, regressor=True))
    if meta == "stack":
        est = mojolearn.StackingClassifier([("a", mojolearn.LogisticRegression(max_iter=200))], cv=_UserSplit())
    elif meta == "cal":
        est = CalibratedClassifierCV(mojolearn.LogisticRegression(max_iter=200), cv=_UserSplit())
    else:
        est = GridSearchCV(mojolearn.LogisticRegression(max_iter=200), {"C": [0.5, 1.0]}, cv=_UserSplit())
    return _lines(f"usersplit_{meta}_{caller}", lambda: _in_caller(caller, est, X, y))


def _ridgecv(index):
    import mojolearn
    from mojolearn.model_selection import KFold
    (form,) = RIDGECV[index]
    X, _ = _data(101, 120, 3)
    yr = (X @ np.arange(1, 6, dtype=np.float32) + 0.1).astype(np.float32)
    idx = np.arange(120)
    w = None
    if form == "kfold_shuffle":
        cv = KFold(4, shuffle=True, random_state=6)
    elif form == "list_weighted":
        cv = [(idx[idx % 3 != f], idx[idx % 3 == f]) for f in range(3)]
        w = (1.0 + (idx % 5)).astype(np.float32)
    else:
        cv = [(idx[2:], idx[:2]), (idx[1:], idx[:1]), (idx[:-3], idx[-3:])]

    def run():
        m = mojolearn.RidgeCV(alphas=(0.1, 1.0, 10.0), cv=cv).fit(X, yr, sample_weight=w)
        return [("coef", m.coef_), ("intercept", np.asarray(m.intercept_)), ("alpha", np.asarray(m.alpha_)),
                ("best_score", np.asarray(m.best_score_))]
    return _lines(f"ridgecv_{form}", run)


def _adasw(index):
    import mojolearn
    kind, caller = ADASW[index]
    X, y = _data(103, 160, 3)
    w = (0.5 + (np.arange(160) % 4)).astype(np.float32)
    if kind == "reg":
        yr = (X @ np.arange(1, 6, dtype=np.float32)).astype(np.float32)
        est = mojolearn.AdaBoostRegressor(mojolearn.DecisionTreeRegressor(max_depth=3), n_estimators=5, random_state=0)
        meta = (mojolearn.VotingRegressor([("a", est), ("b", mojolearn.Ridge())]) if caller == "vote" else
                mojolearn.StackingRegressor([("a", est)], cv=3))
        return _lines(f"adasw_{kind}_{caller}", lambda: [("predict", meta.fit(X, yr, sample_weight=w).predict(X))])
    est = mojolearn.AdaBoostClassifier(mojolearn.DecisionTreeClassifier(max_depth=2), n_estimators=5, random_state=0)
    meta = (mojolearn.VotingClassifier([("a", est), ("b", mojolearn.LogisticRegression())], voting="soft")
            if caller == "vote" else mojolearn.StackingClassifier([("a", est)], cv=3))
    return _lines(f"adasw_{kind}_{caller}", lambda: [("proba", meta.fit(X, y, sample_weight=w).predict_proba(X))])


def _svccw(index):
    import mojolearn
    from mojolearn.model_selection import cross_val_score
    (caller,) = SVCCW[index]
    X, y = _data(107, 150, 3)
    est = mojolearn.SVC(C=1.0, gamma=0.5, class_weight=3.5)
    if caller == "cvs":
        return _lines("svccw_cvs", lambda: [("scores", cross_val_score(est, X, y, cv=3))])
    return _lines(f"svccw_{caller}", lambda: _in_caller(caller, est, X, y))


def _bagpm(index):
    import mojolearn
    cw, weighted, caller = BAGPM[index]
    X, y = _data(109, 160, 3)
    if cw == "absent":
        tree = mojolearn.DecisionTreeClassifier(max_depth=3, class_weight={1: 2.0})
        y = y.copy()
        y[:3] = 1
        y[3:] = np.where(y[3:] == 1, 0, y[3:])  # class 1 keeps three rows: a draw of 4 rows may lack it
        est = mojolearn.BaggingClassifier(tree, n_estimators=6, max_samples=4, max_features=3, random_state=0)
        return _lines("bagpm_absent_vote", lambda: _in_caller("vote", est, X, y))
    tree = mojolearn.DecisionTreeClassifier(max_depth=4, class_weight=_cw(cw))
    est = mojolearn.BaggingClassifier(tree, n_estimators=6, max_features=3, random_state=0)
    if not weighted:
        return _lines(f"bagpm_{cw}_plain_{caller}", lambda: _in_caller(caller, est, X, y))
    if caller not in ("vote", "stack"):
        return []
    w = (0.5 + (np.arange(160) % 3)).astype(np.float32)
    meta = (mojolearn.VotingClassifier([("a", est), ("b", mojolearn.LogisticRegression())], voting="soft")
            if caller == "vote" else mojolearn.StackingClassifier([("a", est)], cv=3))
    return _lines(f"bagpm_{cw}_weighted_{caller}", lambda: [("proba", meta.fit(X, y, sample_weight=w).predict_proba(X))])


SECTIONS = ((BAGCW, _bagcw), (ADACW, _adacw), (ADARAND, _adarand), (NESTCW, _nestcw), (SUBCLS, _subcls),
            (SPLIT, _split), (CALCV, _calcv), (USERSPLIT, _usersplit), (RIDGECV, _ridgecv), (ADASW, _adasw),
            (SVCCW, _svccw), (BAGPM, _bagpm))


def _run(index):
    for table, fn in SECTIONS:
        if index < len(table):
            return fn(index)
        index -= len(table)
    return []


def main():
    total = sum(len(table) for table, _ in SECTIONS)
    with ProcessPoolExecutor(max_workers=min(total, os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(total)):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
