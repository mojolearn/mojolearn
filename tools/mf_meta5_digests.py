"""Before/after digests of lane mf-meta5 (2026-11-23), host column. One line per configuration and output: name and
sha256 of the output bytes (dtype and shape included); a refusal is digested by its words. Base: origin/lane/
member-fits-5 89a4f6be5d (RFE and ExperimentalTwoLevelFeatureFreq carried no member mark: every caller used their
own doors; a search's units over LabelPropagation with -1 labels ran the Python units), head lane/mf-meta5 (THE RFE
KIND, the FeatureFreq Gbdt member, the unlabeled marker in the LabelPropagation member; RFE's own door runs the one
typed `rfe_fit_run`). Test tool, never product runtime.

Configurations:
  * `rfe_door_*`: RFE.fit / predict / transform over LDA, LogisticRegression, Ridge, a RandomForest
    (feature_importances_) and GaussianNB (the refusal), n_features_to_select None / 2 / 0.5, step 1 / 2: the
    door's member path now runs `rfe_fit_run` whole (support_, ranking_, estimator_ predictions).
  * `rfe_in_*`: RFE(LogisticRegression) and RFE(Ridge) inside BaggingClassifier is not served (its base is a
    tree); inside VotingClassifier, StackingClassifier, CalibratedClassifierCV, GridSearchCV
    (n_features_to_select grid) and cross_val_score; RFE(Ridge) inside VotingRegressor and GridSearchCV.
  * `ff_*`: ExperimentalTwoLevelFeatureFreq alone, and inside VotingRegressor, BaggingRegressor is not served (tree
    base), GridSearchCV (learning_rate grid) and cross_val_score; predict and model_.
  * `lp_units_*`: cross_val_score and GridSearchCV over LabelPropagation / LabelSpreading with every fourth label
    -1 (the units now fit the member with the marker word).

Section meta5b (lane mf-meta5b; base origin/lane/member-fits-5 0d7bb545e9, head lane/mf-meta5b):
  * `b_params_*`: RFE.fit(X, y, **fit_params) over LogisticRegression / Ridge / a RandomForest with
    sample_weight (and the forest's feature_importances=True), an unknown key (the compiled fit's TypeError),
    a GradientBoosting booster with cat_features and an eval_set (numeric and object held-out X), and the
    classifier adapter over three classes with an eval_set (its multiclass refusal).
  * `b_lp_*`: RFE over LabelPropagation with every fourth label -1 (no round: the final fit; a round: the
    no-importance refusal), and CalibratedClassifierCV(LabelPropagation / LabelSpreading) over the same labels,
    ensemble True and False, 4 classes with the marker (predict_proba, calibrated_classifiers_ classes).
  * `b_dotted_*`: RFE(GridSearchCV(LogisticRegression), importance_getter='best_estimator_.coef_'),
    RFE(RFE(Ridge), 'estimator_.coef_') and RFE(StackingClassifier, 'final_estimator_.coef_').
  EXPECTED meta5b: equal outputs for every served case (the member path runs the same fits and programs); the
  `*_raised` words of a refusal the native UNPACK now raises may differ in their prefix only; CalibratedClassifierCV
  over LabelPropagation with -1 labels moved from the door's units to the RUN (same fits; any bit difference
  there is a defect to report). Every b_* unit is independent: one process each.

EXPECTED: every line equal (same RUNs, same programs, same order), except, stated: `*_raised` lines whose words
change from a door refusal to the member's refusal of the same input, and the `lp_units_*` and `ff_*`/`rfe_in_*`
search scores computed by the typed metric program where the old units used the Python scorer (the last bits may
differ, as lane mf-metric4 noted). Configurations are independent: one process each across the host cores.

    python tools/mf_meta5_digests.py > digests.txt
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


def _base(name):
    import mojolearn
    if name == "lda":
        return mojolearn.LinearDiscriminantAnalysis()
    if name == "logreg":
        return mojolearn.LogisticRegression(max_iter=200)
    if name == "ridge":
        return mojolearn.Ridge(alpha=1.0)
    if name == "rf":
        return mojolearn.RandomForestClassifier(n_estimators=8, random_state=0)
    return mojolearn.GaussianNB()


def _rfe_door(arg):
    _, base, nsel, step = arg
    import mojolearn
    X, y, t = _data(3, 200, 8, 3)
    tag = f"rfe_door_{base}_{nsel}_{step}"
    out = []
    try:
        target = t if base == "ridge" else y
        m = mojolearn.RFE(_base(base), n_features_to_select=nsel, step=step).fit(X, target)
        out.append((tag + "_support", _digest(np.asarray(m.support_, dtype=bool))))
        out.append((tag + "_ranking", _digest(m.ranking_)))
        out.append((tag + "_predict", _digest(m.predict(X))))
        out.append((tag + "_transform", _digest(m.transform(X))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _rfe_in(arg):
    _, meta, base = arg
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    from mojolearn.model_selection import GridSearchCV, cross_val_score
    X, y, t = _data(5, 180, 6, 3)
    tag = f"rfe_in_{meta}_{base}"
    out = []

    def rfe():
        return mojolearn.RFE(_base(base), n_features_to_select=3)
    try:
        regression = base == "ridge"
        target = t if regression else y
        if meta == "vote":
            cls = mojolearn.VotingRegressor if regression else mojolearn.VotingClassifier
            m = cls([("a", rfe()), ("b", _base(base))]).fit(X, target)
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif meta == "stack":
            m = mojolearn.StackingClassifier([("a", rfe())], cv=3).fit(X, target)
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif meta == "cal":
            m = CalibratedClassifierCV(rfe(), cv=3).fit(X, target)
            out.append((tag + "_proba", _digest(m.predict_proba(X))))
        elif meta == "search":
            m = GridSearchCV(rfe(), {"n_features_to_select": [2, 4]}, cv=3).fit(X, target)
            out.append((tag + "_scores", _digest(np.asarray(m.cv_results_["mean_test_score"], dtype=np.float64))))
            out.append((tag + "_best_support", _digest(np.asarray(m.best_estimator_.support_, dtype=bool))))
        else:
            out.append((tag + "_cv", _digest(np.asarray(cross_val_score(rfe(), X, target, cv=3), dtype=np.float64))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _ff(arg):
    _, meta = arg
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, cross_val_score
    rng = np.random.default_rng(11)
    n = 240
    X = np.column_stack([rng.integers(0, 3, n), rng.integers(0, 4, n), rng.random(n), rng.random(n)]).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.25 + X[:, 2] + 0.05 * rng.standard_normal(n)).astype(np.float32)
    tag = f"ff_{meta}"
    out = []

    def ff():
        return mojolearn.ExperimentalTwoLevelFeatureFreq(sources=(0, 1))
    try:
        if meta == "alone":
            m = ff().fit(X, y)
            out.append((tag + "_predict", _digest(m.predict(X))))
            out.append((tag + "_model", hashlib.sha256(m.model_.encode()).hexdigest()[:24]))
        elif meta == "vote":
            m = mojolearn.VotingRegressor([("a", ff()), ("b", mojolearn.Ridge())]).fit(X, y)
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif meta == "search":
            m = GridSearchCV(ff(), {"learning_rate": [0.03, 0.1]}, cv=3).fit(X, y)
            out.append((tag + "_scores", _digest(np.asarray(m.cv_results_["mean_test_score"], dtype=np.float64))))
        elif meta == "weighted":
            m = ff().fit(X, y, sample_weight=np.linspace(0.5, 1.5, n).astype(np.float32))
            out.append((tag + "_predict", _digest(m.predict(X))))
        else:
            out.append((tag + "_cv", _digest(np.asarray(cross_val_score(ff(), X, y, cv=3), dtype=np.float64))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _lp_units(arg):
    _, name, how = arg
    import mojolearn
    from mojolearn.model_selection import GridSearchCV, cross_val_score
    X, y, _ = _data(7, 90, 3, 3)
    marked = y.copy()
    marked[::4] = -1
    est = mojolearn.LabelPropagation(max_iter=200) if name == "lp" else mojolearn.LabelSpreading(alpha=0.2)
    tag = f"lp_units_{name}_{how}"
    out = []
    try:
        if how == "cv":
            out.append((tag + "_cv", _digest(np.asarray(cross_val_score(est, X, marked, cv=3), dtype=np.float64))))
        else:
            grid = {"max_iter": [5, 50]} if name == "lp" else {"alpha": [0.2, 0.5]}
            m = GridSearchCV(est, grid, cv=3).fit(X, marked)
            out.append((tag + "_scores", _digest(np.asarray(m.cv_results_["mean_test_score"], dtype=np.float64))))
            out.append((tag + "_best_dist", _digest(m.best_estimator_.label_distributions_)))
            out.append((tag + "_best_classes", repr(list(m.best_estimator_.classes_))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _b_params(arg):
    _, case = arg
    import mojolearn
    from mojolearn.feature_selection import RFE
    X, y, t = _data(11, 160, 6, 3)
    w = np.linspace(0.5, 2.0, len(X)).astype(np.float32)
    tag = f"b_params_{case}"
    out = []
    try:
        if case == "logreg_sw":
            m = RFE(mojolearn.LogisticRegression(max_iter=200), n_features_to_select=3).fit(X, y, sample_weight=w)
        elif case == "ridge_sw":
            m = RFE(mojolearn.Ridge(alpha=1.0), n_features_to_select=2, step=2).fit(X, t, sample_weight=w)
        elif case == "rf_fi":
            m = RFE(mojolearn.RandomForestClassifier(n_estimators=6, random_state=0), n_features_to_select=3).fit(
                X, y, sample_weight=w, feature_importances=True)
        elif case == "unknown":
            m = RFE(mojolearn.Ridge(alpha=1.0), n_features_to_select=3).fit(X, t, bogus=1)
        elif case.startswith("gb_cat"):
            Xc = X.copy()
            Xc[:, 0] = np.floor(Xc[:, 0] * 4)
            held = Xc[:40].copy()
            if case == "gb_cat_object":
                held = held.astype(object)
                held[:, 0] = np.array(["a", "b", "c", "d"] * 10, dtype=object)
            est = mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[0])
            m = RFE(est, n_features_to_select=6).fit(Xc, t, eval_set=(held, t[:40]))
        else:  # the classifier adapter over three classes
            est = mojolearn.GradientBoostingClassifier(n_estimators=4)
            m = RFE(est, n_features_to_select=6).fit(X, y, eval_set=(X[:30], y[:30]))
        out.append((tag + "_support", _digest(np.asarray(m.support_, dtype=bool))))
        out.append((tag + "_ranking", _digest(m.ranking_)))
        out.append((tag + "_predict", _digest(m.predict(X))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _b_lp(arg):
    _, case = arg
    import mojolearn
    from mojolearn.feature_selection import RFE
    X, y, _ = _data(13, 120, 4, 4)
    marked = y.copy()
    marked[::4] = -1
    tag = f"b_lp_{case}"
    out = []
    try:
        if case.startswith("rfe"):
            m = RFE(mojolearn.LabelPropagation(max_iter=200), n_features_to_select=4 if case == "rfe_final" else 2)
            m.fit(X, marked)
            out.append((tag + "_predict", _digest(m.predict(X))))
            out.append((tag + "_classes", repr(list(m.estimator_.classes_))))
        else:
            est = mojolearn.LabelPropagation(max_iter=200) if "lp" in case else mojolearn.LabelSpreading(alpha=0.2)
            m = mojolearn.CalibratedClassifierCV(est, cv=3, ensemble=case.endswith("_ens")).fit(X, marked)
            out.append((tag + "_proba", _digest(m.predict_proba(X))))
            out.append((tag + "_classes", repr([list(c.estimator.classes_) for c in m.calibrated_classifiers_])))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _b_dotted(arg):
    _, case = arg
    import mojolearn
    from mojolearn.feature_selection import RFE
    from mojolearn.model_selection import GridSearchCV
    X, y, t = _data(17, 150, 6, 2)
    tag = f"b_dotted_{case}"
    out = []
    try:
        if case == "search":
            inner = GridSearchCV(mojolearn.LogisticRegression(max_iter=200), {"C": [0.5, 1.0]}, cv=3)
            m = RFE(inner, n_features_to_select=3, importance_getter="best_estimator_.coef_").fit(X, y)
        elif case == "rfe":
            m = RFE(RFE(mojolearn.Ridge(alpha=1.0), n_features_to_select=6), n_features_to_select=3,
                    importance_getter="estimator_.coef_").fit(X, t)
        else:
            stack = mojolearn.StackingClassifier([("lr", mojolearn.LogisticRegression(max_iter=200))],
                                                 final_estimator=mojolearn.LogisticRegression(max_iter=200), cv=3)
            m = RFE(stack, n_features_to_select=3, importance_getter="final_estimator_.coef_").fit(X, y)
        out.append((tag + "_support", _digest(np.asarray(m.support_, dtype=bool))))
        out.append((tag + "_ranking", _digest(m.ranking_)))
        out.append((tag + "_predict", _digest(m.predict(X))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _unit(arg):
    kind = arg[0]
    if kind == "b_params":
        return _b_params(arg)
    if kind == "b_lp":
        return _b_lp(arg)
    if kind == "b_dotted":
        return _b_dotted(arg)
    if kind == "rfe_door":
        return _rfe_door(arg)
    if kind == "rfe_in":
        return _rfe_in(arg)
    if kind == "ff":
        return _ff(arg)
    return _lp_units(arg)


UNITS = ([("rfe_door", b, s, st) for b in ("lda", "logreg", "ridge", "rf", "gnb") for s in (None, 2, 0.5)
          for st in (1, 2)]
         + [("rfe_in", m, b) for m in ("vote", "stack", "cal", "search", "cv") for b in ("logreg", "ridge")]
         + [("ff", m) for m in ("alone", "vote", "search", "weighted", "cv")]
         + [("lp_units", n, h) for n in ("lp", "ls") for h in ("cv", "search")]
         + [("b_params", c) for c in ("logreg_sw", "ridge_sw", "rf_fi", "unknown", "gb_cat_numeric", "gb_cat_object",
                                      "gbc_multiclass")]
         + [("b_lp", c) for c in ("rfe_final", "rfe_round", "cal_lp_ens", "cal_lp_oof", "cal_ls_ens", "cal_ls_oof")]
         + [("b_dotted", c) for c in ("search", "rfe", "stack")])


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
