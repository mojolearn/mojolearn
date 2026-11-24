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

Section hostent5 (lane mf-hostent5; base origin/lane/member-fits-5 996f24a41b vs the head of lane/mf-hostent5):
OrderedRMSE as Gbdt member kind 4 and the saved HostForest / HostGBDT as predict-only kinds.
  * ordered_fit / ordered_fit_w: OrderedRMSE's own fit (its door now runs the ONE typed body of its checks):
    EXPECTED equal (same checks, same `xg_ordered_fit`, same order words).
  * ordered_bad_*: the permutation refusals (a repeat, a float, a short list): EXPECTED equal words.
  * ordered_rfe_final (n_features_to_select = all: no round, only the final fit), ordered_rfe_round (one round:
    the Gbdt kinds' no-importance refusal) and ordered_rfe_no_perm: before, the estimator's own doors (`_fit_clone`);
    after, the member kind with `permutation` carried from the fit parameters. EXPECTED: the final fit's model text
    and predict equal; the round's and the missing permutation's refusals may change wording only where the door's
    became the member's (the missing argument is Python's TypeError either way).
  * ordered_vote / ordered_search: before, refused at the meta admission (no plan); after, the member kind, whose
    fit raises Python's missing-argument TypeError (as scikit-learn's meta-estimators fit it: no permutation).
    EXPECTED: `*_raised` with new words.
  * ordered_shap, host_forest_* and host_gbdt_*: Kernel / Permutation SHAP over a fitted OrderedRMSE and over saved
    HostForest (rf / et, classifier / regressor, sequential and parallel_groves) and HostGBDT (RMSE, Logloss;
    MultiClass refused) models: before, refused (no plan: "requires a mojolearn estimator"); after, the member
    kind's values. The host model's own predict is printed beside them (EXPECTED equal before and after: the
    class's own door is unchanged), and the values must equal those of the same explainer over the estimator that
    saved the file (the `*_twin` lines: the same predict bits by the contract).

Section cal5 (lane mf-cal5; base origin/lane/member-fits-5 3dbe1143cc vs the head of lane/mf-cal5): THE FALLBACK
RULE cases this lane moved off the doors.
  * cw_eq_*: an SGDClassifier / LogisticRegressionCV whose class_weight is an object equal to 'balanced' (not a str),
    as an RFE estimator, a CalibratedClassifierCV base and a VotingClassifier member: before, the member was not
    admitted and each caller kept its door; after, the member's 'balanced' tail. EXPECTED equal outputs.
  * rfe_fi_*: RFE(RandomForestClassifier).fit(..., feature_importances=np.bool_(True)) (before: the door; after:
    the member: EXPECTED equal), feature_importances=1 (the forest door's words both times: EXPECTED equal words),
    RFE(ExtraTreesClassifier).fit(..., feature_importances="yes") (ignored by et_fit; before the door, after the
    member: EXPECTED equal).
  * rfe_one_*: RFE over a one-class y: a DecisionTreeClassifier (before the door, after the member: EXPECTED equal
    support, ranking and predict), a LogisticRegression (its fit's one-class words: EXPECTED the same words; a
    change of wording is a defect to send back).
  * rfe_gb_cat_outside: RFE(GradientBoosting(cat_features=[9])).fit(X, t, eval_set=...) on 6 columns: the fit
    door's `_cat_slots` words both times (EXPECTED equal words).
  * host_late: model_selection loaded, then HostForest / HostGBDT touched (a pool worker may have loaded them
    already; for the late-import case proper call `_cal5(("cal5", "host_late"))` in a fresh interpreter): their
    member plans are registered at their import (EXPECTED after: True / True; before: whatever the rescan gave).

Section refuse5 (lane mf-refuse5; base origin/lane/member-fits-5 65eb618002 vs the head of lane/mf-refuse5): THE
DOOR'S REFUSAL. Each case is a meta-estimator with one constructor value its fit door refuses, fitted (a) at the
top level (`*_door`: its own fit door) and (b) as a member of a VotingClassifier / VotingRegressor whose fit runs
its members through the native member set (`*_member`: before, the META admission answered False and the member
set fell back to the meta's door to raise the words; after, the META kind's FIT raises them, from the same check
function, bindings/trees_meta_validate_unpack.mojo). `*_same` says whether the two words are equal.
EXPECTED: before and after, every `*_door` line equal; after, every `*_same` True (a False is a defect to send
back); `*_member` lines equal before and after (the words do not change, only the path that raises them).

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


def _ordered(n_estimators=4):
    import mojolearn
    return mojolearn.OrderedRMSE(n_estimators=n_estimators, max_depth=3)


def _explain(model, X, which):
    import mojolearn as ml
    explainer = (ml.KernelExplainer if which == "K" else ml.PermutationExplainer)(model, X[:16], random_state=0)
    kw = {"nsamples": 64} if which == "K" else {}
    return np.asarray(explainer.shap_values(X[:6], **kw))


def _saved(case, X, y, t, path):
    """The estimator that saves the file of a host case, fitted."""
    import mojolearn
    if case.startswith("host_forest"):
        _, _, family, kind, engine = case.split("_")
        cls = {("rf", "cls"): mojolearn.RandomForestClassifier, ("rf", "reg"): mojolearn.RandomForestRegressor,
               ("et", "cls"): mojolearn.ExtraTreesClassifier, ("et", "reg"): mojolearn.ExtraTreesRegressor}[(family, kind)]
        kw = {"inference_engine": "parallel_groves"} if engine == "groves" else {}
        model = cls(n_estimators=8, random_state=0, **kw).fit(X, y if kind == "cls" else t)
    else:
        loss = case.split("_")[2]
        target = t if loss == "RMSE" else (y if loss == "MultiClass" else (y > 0).astype(np.float32))
        model = mojolearn.GradientBoosting(loss=loss, n_estimators=8).fit(X, target)
    model.save(path)
    return model


def _hostent5(arg):
    _, case = arg
    import mojolearn
    from mojolearn.feature_selection import RFE
    from mojolearn.model_selection import GridSearchCV
    X, y, t = _data(31, 160, 6, 3)
    n = X.shape[0]
    perm = np.random.default_rng(5).permutation(n)
    tag = f"hostent5_{case}"
    out = []
    try:
        if case == "ordered_fit":
            m = _ordered().fit(X, t, permutation=perm)
            out.append((tag + "_model", hashlib.sha256(m.model_.encode()).hexdigest()[:24]))
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif case == "ordered_fit_w":
            w = np.linspace(0.5, 1.5, n, dtype=np.float32)
            m = _ordered().fit(X, t, permutation=list(perm), sample_weight=w)
            out.append((tag + "_model", hashlib.sha256(m.model_.encode()).hexdigest()[:24]))
        elif case.startswith("ordered_bad"):
            bad = {"ordered_bad_repeat": np.zeros(n, dtype=np.int64),
                   "ordered_bad_float": perm.astype(np.float64),
                   "ordered_bad_short": list(perm[:-1])}[case]
            _ordered().fit(X, t, permutation=bad)
            out.append((tag + "_answered", "fitted"))
        elif case in ("ordered_rfe_final", "ordered_rfe_round", "ordered_rfe_no_perm"):
            keep = 6 if case != "ordered_rfe_round" else 3
            params = {} if case == "ordered_rfe_no_perm" else {"permutation": perm}
            m = RFE(_ordered(), n_features_to_select=keep).fit(X, t, **params)
            out.append((tag + "_model", hashlib.sha256(m.estimator_.model_.encode()).hexdigest()[:24]))
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif case == "ordered_vote":
            mojolearn.VotingRegressor([("o", _ordered()), ("r", mojolearn.Ridge())]).fit(X, t)
            out.append((tag + "_answered", "fitted"))
        elif case == "ordered_search":
            GridSearchCV(_ordered(), {"max_depth": [2, 3]}, cv=3).fit(X, t)
            out.append((tag + "_answered", "fitted"))
        elif case == "ordered_shap":
            m = _ordered().fit(X, t, permutation=perm)
            out.append((tag + "_kernel", _digest(_explain(m, X, "K"))))
            out.append((tag + "_permutation", _digest(_explain(m, X, "P"))))
        else:
            import tempfile
            from mojolearn import HostForest, HostGBDT
            with tempfile.TemporaryDirectory() as root:
                path = os.path.join(root, "model.npz")
                twin = _saved(case, X, y, t, path)
                host = (HostForest if case.startswith("host_forest") else HostGBDT).from_file(path)
                out.append((tag + "_host_predict", _digest(host.predict(X))))
                for which in ("K", "P"):
                    try:
                        out.append((tag + "_" + which, _digest(_explain(host, X, which))))
                    except Exception as e:  # noqa: BLE001 - the words are the digest
                        out.append((tag + "_" + which + "_raised", f"{type(e).__name__}: {e}"))
                    try:
                        out.append((tag + "_" + which + "_twin", _digest(_explain(twin, X, which))))
                    except Exception as e:  # noqa: BLE001 - the words are the digest
                        out.append((tag + "_" + which + "_twin_raised", f"{type(e).__name__}: {e}"))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


class _EqBalanced:
    """Not a str; equal to 'balanced' (the x_linear / SGD doors test class_weight by ==)."""

    def __eq__(self, other):
        return other == "balanced"

    def __hash__(self):
        return hash("balanced")


def _cal5(arg):
    _, case = arg
    import mojolearn
    from mojolearn.feature_selection import RFE
    X, y, t = _data(37, 120, 6, 3)
    tag = f"cal5_{case}"
    out = []
    try:
        if case.startswith("cw_eq_"):
            from mojolearn.calibration import CalibratedClassifierCV
            if case.endswith("lrcv"):
                est = mojolearn.LogisticRegressionCV(cv=3, class_weight=_EqBalanced())
            else:
                est = mojolearn.SGDClassifier(random_state=0, max_iter=50, class_weight=_EqBalanced())
            if "_rfe" in case:
                m = RFE(est, n_features_to_select=3).fit(X, y)
                out.append((tag + "_support", _digest(np.asarray(m.support_, dtype=bool))))
                out.append((tag + "_ranking", _digest(m.ranking_)))
            elif "_cal" in case:
                m = CalibratedClassifierCV(est, cv=3).fit(X, y)
                out.append((tag + "_proba", _digest(m.predict_proba(X))))
            else:
                m = mojolearn.VotingClassifier([("a", est), ("b", mojolearn.LogisticRegression(max_iter=200))]).fit(X, y)
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif case.startswith("rfe_fi_"):
            if case == "rfe_fi_et":
                est = mojolearn.ExtraTreesClassifier(n_estimators=6, random_state=0)
                m = RFE(est, n_features_to_select=3).fit(X, y, feature_importances="yes")
            else:
                value = np.bool_(True) if case == "rfe_fi_bool_" else 1
                est = mojolearn.RandomForestClassifier(n_estimators=6, random_state=0)
                m = RFE(est, n_features_to_select=3).fit(X, y, feature_importances=value)
            out.append((tag + "_support", _digest(np.asarray(m.support_, dtype=bool))))
            out.append((tag + "_ranking", _digest(m.ranking_)))
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif case.startswith("rfe_one_"):
            one = np.zeros_like(y)
            est = (mojolearn.DecisionTreeClassifier(random_state=0) if case == "rfe_one_tree"
                   else mojolearn.LogisticRegression(max_iter=200))
            m = RFE(est, n_features_to_select=3).fit(X, one)
            out.append((tag + "_support", _digest(np.asarray(m.support_, dtype=bool))))
            out.append((tag + "_ranking", _digest(m.ranking_)))
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif case == "rfe_gb_cat_outside":
            Xc = X.copy()
            Xc[:, 0] = np.floor(Xc[:, 0] * 4)
            est = mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[9])
            m = RFE(est, n_features_to_select=6).fit(Xc, t, eval_set=(Xc[:40].copy(), t[:40]))
            out.append((tag + "_predict", _digest(m.predict(Xc))))
        else:  # host_late
            from mojolearn import model_selection
            registry = model_selection._NATIVE_CONTEXT.get("family_msel_estimators", {})
            forest = mojolearn.HostForest
            gbdt = mojolearn.HostGBDT
            out.append((tag + "_forest", str(forest in registry)))
            out.append((tag + "_gbdt", str(gbdt in registry)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _refuse5_meta(case):
    """The meta-estimator of a refuse5 case and whether it is a classifier."""
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    lr = mojolearn.LogisticRegression(max_iter=200)
    tree = mojolearn.DecisionTreeClassifier(max_depth=2, random_state=0)
    two = [("a", mojolearn.LogisticRegression(max_iter=200)), ("b", mojolearn.LogisticRegression(C=0.5, max_iter=200))]
    if case == "voting_bad":
        return mojolearn.VotingClassifier(two, voting="bad"), True
    if case == "voting_verbose":
        return mojolearn.VotingClassifier(two, verbose=True), True
    if case == "voting_weights_count":
        return mojolearn.VotingClassifier(two, weights=[1.0, 2.0, 3.0]), True
    if case == "voting_all_drop":
        return mojolearn.VotingClassifier([("a", "drop"), ("b", "drop")]), True
    if case == "stack_method_bad":
        return mojolearn.StackingClassifier(two, stack_method="bad"), True
    if case == "stack_cv_one":
        return mojolearn.StackingClassifier(two, cv=1), True
    if case == "ovr_jobs_zero":
        return mojolearn.OneVsRestClassifier(lr, n_jobs=0), True
    if case == "cal_method_bad":
        return CalibratedClassifierCV(lr, method="bad", cv=3), True
    if case == "cal_ensemble_bad":
        return CalibratedClassifierCV(lr, ensemble="bad", cv=3), True
    if case == "bag_oob_no_bootstrap":
        return mojolearn.BaggingClassifier(tree, n_estimators=3, oob_score=True, bootstrap=False), True
    if case == "bag_zero_estimators":
        return mojolearn.BaggingClassifier(tree, n_estimators=0), True
    if case == "bag_samples_outside":
        return mojolearn.BaggingClassifier(tree, n_estimators=3, max_samples=1.5), True
    if case == "bag_base_not_tree":
        return mojolearn.BaggingClassifier(lr, n_estimators=3), True
    if case == "bag_cw_absent":
        return mojolearn.BaggingClassifier(mojolearn.DecisionTreeClassifier(max_depth=2, class_weight={99: 2.0}),
                                           n_estimators=3, random_state=0), True
    if case == "ada_algorithm_bad":
        return mojolearn.AdaBoostClassifier(tree, n_estimators=3, algorithm="SAMME.R"), True
    if case == "ada_rate_zero":
        return mojolearn.AdaBoostRegressor(mojolearn.DecisionTreeRegressor(max_depth=2), n_estimators=3,
                                           learning_rate=0.0), False
    return mojolearn.AdaBoostClassifier(tree, n_estimators=3, random_state=-1), True  # ada_seed_negative


def _refuse5_words(fit):
    try:
        fit()
    except Exception as e:  # noqa: BLE001 - the words are the digest
        return f"{type(e).__name__}: {e}"
    return "fitted"


def _refuse5(arg):
    _, case = arg
    import mojolearn
    X, y, t = _data(41, 90, 4, 3)
    tag = f"refuse5_{case}"
    meta, classifier = _refuse5_meta(case)
    target = y if classifier else t
    door = _refuse5_words(lambda: meta.fit(X, target))
    meta2, _ = _refuse5_meta(case)
    if classifier:
        outer = mojolearn.VotingClassifier([("m", meta2), ("l", mojolearn.LogisticRegression(max_iter=200))])
    else:
        outer = mojolearn.VotingRegressor([("m", meta2), ("r", mojolearn.Ridge())])
    member = _refuse5_words(lambda: outer.fit(X, target))
    return [(tag + "_door", door), (tag + "_member", member), (tag + "_same", str(door == member))]


REFUSE5 = ("voting_bad", "voting_verbose", "voting_weights_count", "voting_all_drop", "stack_method_bad",
           "stack_cv_one", "ovr_jobs_zero", "cal_method_bad", "cal_ensemble_bad", "bag_oob_no_bootstrap",
           "bag_zero_estimators", "bag_samples_outside", "bag_base_not_tree", "bag_cw_absent", "ada_algorithm_bad",
           "ada_rate_zero", "ada_seed_negative")


def _unit(arg):
    if arg[0] == "refuse5":
        return _refuse5(arg)
    if arg[0] == "del5b_rfe":
        return _del5b_rfe(arg)
    if arg[0] == "hostent5":
        return _hostent5(arg)
    if arg[0] == "cal5":
        return _cal5(arg)
    return _del5(arg)


GETTERS = ("best_estimator_.coef_", "estimator_.coef_", "final_estimator_.coef_")
UNITS = ([("del5b_rfe", m, g, r) for m in ("door", "vote", "search", "cv") for g in GETTERS for r in (False, True)]
         + [("del5b_rfe", "cal", g, False) for g in GETTERS]
         + [("del5", c) for c in ("cal_loo", "cal_too_many", "stack_repeated", "stack_continuous", "search_repeated",
                                  "search_continuous", "ridgecv_kfold", "ridgecv_strat_repeated")]
         + [("hostent5", c) for c in (
             "ordered_fit", "ordered_fit_w", "ordered_bad_repeat", "ordered_bad_float", "ordered_bad_short",
             "ordered_rfe_final", "ordered_rfe_round", "ordered_rfe_no_perm", "ordered_vote", "ordered_search",
             "ordered_shap")]
         + [("hostent5", f"host_forest_{f}_{k}_{e}") for f in ("rf", "et") for k in ("cls", "reg")
            for e in ("seq", "groves")]
         + [("hostent5", f"host_gbdt_{loss}") for loss in ("RMSE", "Logloss", "MultiClass")]
         + [("cal5", c) for c in (
             "cw_eq_rfe_sgd", "cw_eq_cal_sgd", "cw_eq_vote_sgd", "cw_eq_rfe_lrcv", "cw_eq_cal_lrcv",
             "rfe_fi_bool_", "rfe_fi_int", "rfe_fi_et", "rfe_one_tree", "rfe_one_logistic", "rfe_gb_cat_outside",
             "host_late")]
         + [("refuse5", c) for c in REFUSE5])


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
