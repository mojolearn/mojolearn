"""Before/after digests of the META member kind (lane mf-meta3, 2026-11-23; core/ensemble_member.mojo): every
caller that now reaches a VotingClassifier / VotingRegressor / StackingClassifier / StackingRegressor as a NATIVE
member through the member ABI, host column. Base = origin/lane/member-fits ccbef84e9b (those callers ran the
meta-estimator's own fit and predict doors: CalibratedClassifierCV's door units `_CalUnits`, the outer meta's door
loop `_fit_member`, IterativeImputer's door rounds, SHAP's `_DoorModel`), head = lane/mf-meta3.

One line per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal
is digested by its words. EVERY line must be equal: the META kind runs the door's own RUNs in the door's order
(xtrees/trees_meta_vote_run.mojo vote_* / stack_place_run, the same member sets, the same folds of
`xt_device_folds`, the same float32 rounding of the meta features), so no bits change. A differing line is a defect
to send back, with one known exception class: a nested member's fit NOTICE (a convergence warning) is now raised at
the META's PACK, so its position among the warnings of one fit may move (warnings are not digested).

Configurations (each with predict, predict_proba where it exists, and the fitted nested members' coef_ /
intercept_ where they have them):
  * CalibratedClassifierCV(base) for base in {soft VotingClassifier(lr, gnb), StackingClassifier(lr, lda; cv 3)} x
    method {sigmoid, isotonic} x ensemble {True, False} x n_jobs {1, 4};
  * VotingClassifier([("inner", soft VotingClassifier(lr, gnb)), ("lr", lr)], voting soft) (a META member of a
    meta: the inner vote is admitted as META, its members through the leaf dispatch);
  * StackingClassifier([("vote", soft VotingClassifier(lr, gnb)), ("lda", lda)], cv 3) and its SHAP Kernel /
    Permutation values (the inner vote is served by its held META state, no `_DoorModel`);
  * IterativeImputer(estimator=VotingRegressor(ridge, linreg)) and (estimator=StackingRegressor(ridge, linreg;
    cv 3, passthrough True)), max_iter 3, transform of the fit X;
  * hard VotingClassifier as a calibration base (no predict_proba: the door units still answer: equal words);
  * GridSearchCV(VotingClassifier soft, {"weights": [[1, 1], [1, 2]]}, cv 3): best_params_, predict_proba.

(B, lane mf-meta3b, base = origin/lane/member-fits 82eb3f39ea, head = lane/mf-meta3b) the UNIT KINDS
(OneVsRestClassifier, CalibratedClassifierCV, BaggingClassifier / BaggingRegressor) and the stacking cases:
  * CalibratedClassifierCV(base) for base in {OneVsRest(lr), CalibratedClassifierCV(lr, cv 3), Bagging(RF 5 trees,
    4 members, max_samples 0.8, bootstrap_features)} x method x ensemble x n_jobs {1, 4};
  * GridSearchCV(Bagging(RF), {"n_estimators": [2, 3]}, cv 3) and GridSearchCV(OneVsRest(lr), {"n_jobs": [1, 2]}):
    the best estimator refitted through its META member;
  * Kernel / Permutation SHAP over OneVsRest(lr) and over Bagging(RF): the explained model's own META member;
  * IterativeImputer(estimator=BaggingRegressor(RF regressor), max_iter 2);
  * VotingClassifier([("s", StackingClassifier(lr, lda, cv=<index list>)), ("lr", lr)]) and the same with
    stack_method='predict', and Bagging(RF, oob_score=True) inside a VotingClassifier (the door vote fits its
    members as one native set: the Stacking and Bagging members as META units).
EVERY (B) line must be equal except: `cal_hard_*` (the hard vote base is now refused in scikit-learn's words,
"AttributeError: VotingClassifier has none of the following attributes: decision_function, predict_proba.") and
`vote_stack_predict_*`, `vote_stack_usercv_*` when the base raised (a user index list and stack_method='predict' on
a classifier now fit natively; before, the door's own fit answered: equal words expected there too, a differing
line is a defect).

(C, lane mf-meta3c, base = origin/lane/member-fits 7150562308, head = lane/mf-meta3c) THE BATCHED BAGGING ROUTE
inside the Bagging META kind and META estimators nested two and three deep (before: the door kept both):
  * CalibratedClassifierCV(BaggingClassifier(n_estimators=4, random_state=0)) (the default decision-tree base:
    the batched forest fit) x method x ensemble, n_jobs 1;
  * VotingClassifier([("bag", <bagging>), ("lr", lr)], soft) for <bagging> in {the default batched bag,
    bootstrap=False with max_samples=1.0 (batched over every row), max_features=0.6 (the per-member route)}, and
    the batched bag's vote fitted with sample_weight (the per-member route: the batched route takes no weights);
  * IterativeImputer(estimator=BaggingRegressor(n_estimators=3, random_state=0), max_iter 2) (a regressor tree
    left at n_bins=None: the image's bin count in the batched forest's slot 7);
  * CalibratedClassifierCV(StackingClassifier([("vote", soft vote), ("lda", lda)], cv 3)) (two deep in a
    calibration), VotingClassifier([("v2", VotingClassifier([("v1", soft vote), ("lr", lr)], soft)), ("gnb",
    gnb)], soft) (three deep), GridSearchCV(VotingClassifier([("inner", soft vote), ("lr", lr)], soft),
    {"weights": [[1, 1], [1, 2]]}, cv 3) (a nested META admitted, refitted and packed through the work-lists);
    each with the nested fitted estimators' attributes.
EVERY (C) line must be equal (the same forest fit, the same RUNs; the nested states run the same sets).

(D, lane mf-ada4, base = origin/lane/member-fits dcb76ed253, head = lane/mf-ada4) THE ADABOOST KIND: an
AdaBoostClassifier / AdaBoostRegressor as a native member (before: the callers' door units, `_CalUnits`, the outer
meta's door loop, IterativeImputer's door rounds, SHAP's `_DoorModel`):
  * CalibratedClassifierCV(AdaBoostClassifier(n_estimators=5, random_state=0)) (the default stump: the session
    rounds) x method x ensemble, n_jobs 1;
  * VotingClassifier([("ada", <ada>), ("lr", lr)], soft) for <ada> in {the default stump, a RandomForestClassifier
    (n_estimators=3, max_depth=2) base (the member rounds), an ExtraTreesClassifier(n_estimators=3, max_depth=2)
    base, a DecisionTreeClassifier(splitter="random") base (kept on the door: equal words), a
    DecisionTreeClassifier(class_weight="balanced") base (kept on the door)}, with every member's
    estimator_weights_ / estimator_errors_ and each round's leaves;
  * StackingClassifier([("ada", default ada), ("lda", lda)], cv 3) and its Kernel / Permutation SHAP values;
  * IterativeImputer(estimator=AdaBoostRegressor(n_estimators=4, loss="square", random_state=0)) and with a
    RandomForestRegressor(n_estimators=3, max_depth=3) base, max_iter 2;
  * GridSearchCV(AdaBoostClassifier(random_state=0), {"n_estimators": [3, 5]}, cv 3): best_params_, predicts.
EVERY (D) line must be equal: the kind runs the door's `ada_fit_run` (same session, same rounds, same seeds) and its
`ada_classifier_run` / `ada_regressor_run` over the same members' predicts.

(E, lane mf-search4, base = origin/lane/member-fits bb6c1e7ea1, head = lane/mf-search4) THE SEARCH KIND and THE
MULTI-OUTPUT KIND (before: `_CalUnits`, the outer meta's door loop, SHAP's `_DoorModel`, cross_validate's Python
units over a search):
  * CalibratedClassifierCV(GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0, 10.0]}, cv 3, scoring None /
    'accuracy')) x method, n_jobs 1: predicts, every pair's best_params_ and cv_results_ tables;
  * VotingClassifier([("search", RandomizedSearchCV(LogisticRegression(), ..., n_iter 2, random_state 0)),
    ("gnb", GaussianNB())], soft): predicts and the search's cv_results_;
  * cross_val_score(GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv 3), X, y, cv 3): the nested scores;
  * a search with error_score=-1.0 over a refusing candidate; return_train_score=True; scoring 'r2' (kept on the
    door: equal words);
  * GridSearchCV(MultiOutputRegressor(Lasso()), {"estimator__alpha": [0.01, 0.1]}, cv 3) on a 2-D y (its public
    fit keeps the door; its members are the multi-output kind wherever a caller admits them);
  * KernelExplainer over StackingClassifier([("search", <grid search>), ("lda", lda)]): the values.
EVERY (E) line must be equal. Bits: none expected (the same member fits on the same folds; accuracy is a count).

The configurations are independent: one process each across the host cores.

    python tools/mf_meta3_digests.py > digests.txt
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


def _data(seed, n, k, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    s = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    if k == 0:
        return X, s.astype(np.float32)
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return X, np.searchsorted(edges, s).astype(np.int64)


def _soft_vote(ml, weights=None):
    return ml.VotingClassifier([("lr", ml.LogisticRegression(max_iter=200)), ("gnb", ml.GaussianNB())],
                               voting="soft", weights=weights)


def _stack(ml):
    return ml.StackingClassifier([("lr", ml.LogisticRegression(max_iter=200)),
                                  ("lda", ml.LinearDiscriminantAnalysis())], cv=3)


def _members(model, tag):
    out = []
    for name in ("estimators_", "calibrated_classifiers_"):
        for i, e in enumerate(getattr(model, name, None) or ()):
            inner = getattr(e, "estimator", e)
            for attr in ("coef_", "intercept_", "theta_", "var_"):
                v = getattr(inner, attr, None)
                if v is not None:
                    out.append((f"{tag}_{name}{i}_{attr}", _digest(v)))
    return out


def _predicts(model, X, tag):
    out = [(f"{tag}_predict", _digest(model.predict(X)))]
    if hasattr(model, "predict_proba"):
        try:
            out.append((f"{tag}_proba", _digest(model.predict_proba(X))))
        except Exception as e:  # noqa: BLE001 - the words are the digest
            out.append((f"{tag}_proba_raised", f"{type(e).__name__}: {e}"))
    return out


def _configs():
    rows = []
    for base in ("vote", "stack", "hard"):
        for method in ("sigmoid", "isotonic"):
            for ensemble in (True, False):
                for jobs in (1, 4):
                    rows.append(("cal", base, method, ensemble, jobs))
    rows += [("nested_vote",), ("stack_of_vote",), ("shap", "K"), ("shap", "P"), ("iterimp", "vote"),
             ("iterimp", "stack"), ("search",)]
    # (B) lane mf-meta3b
    for base in ("ovr", "calib", "bag"):
        for method in ("sigmoid", "isotonic"):
            for ensemble in (True, False):
                for jobs in (1, 4):
                    rows.append(("unitcal", base, method, ensemble, jobs))
    rows += [("unitsearch", "bag"), ("unitsearch", "ovr"), ("unitshap", "ovr", "K"), ("unitshap", "ovr", "P"),
             ("unitshap", "bag", "K"), ("unitshap", "bag", "P"), ("unititerimp",), ("vote", "stack_usercv"),
             ("vote", "stack_predict"), ("vote", "bag_oob")]
    # (C) lane mf-meta3c
    for method in ("sigmoid", "isotonic"):
        for ensemble in (True, False):
            rows.append(("batchcal", method, ensemble))
    rows += [("batchvote", "default"), ("batchvote", "every_row"), ("batchvote", "subset"),
             ("batchvote", "weighted"), ("batchiterimp",), ("deep", "cal_stack_vote"), ("deep", "vote3"),
             ("deep", "search")]
    # (D) lane mf-ada4
    for method in ("sigmoid", "isotonic"):
        for ensemble in (True, False):
            rows.append(("adacal", method, ensemble))
    rows += [("adavote", "default"), ("adavote", "rf"), ("adavote", "et"), ("adavote", "random"),
             ("adavote", "balanced"), ("adastack",), ("adashap", "K"), ("adashap", "P"), ("adaiterimp", "default"),
             ("adaiterimp", "rf"), ("adasearch",)]
    # (E) lane mf-search4
    for scoring in (None, "accuracy"):
        for method in ("sigmoid", "isotonic"):
            rows.append(("searchcal", scoring, method))
    rows += [("searchvote",), ("searchcv",), ("searcherr",), ("searchtrain",), ("searchr2",), ("searchmulti",),
             ("searchshap",)]
    return rows


def _ada(ml, base="default"):
    if base == "rf":
        est = ml.RandomForestClassifier(n_estimators=3, max_depth=2, random_state=0)
    elif base == "et":
        est = ml.ExtraTreesClassifier(n_estimators=3, max_depth=2, random_state=0)
    elif base == "random":
        est = ml.DecisionTreeClassifier(splitter="random", max_depth=2)
    elif base == "balanced":
        est = ml.DecisionTreeClassifier(class_weight="balanced", max_depth=1)
    else:
        est = None
    return ml.AdaBoostClassifier(est, n_estimators=5, random_state=0)


def _ada_attrs(model, tag):
    out = []
    for i, e in enumerate(getattr(model, "estimators_", None) or ()):
        for attr in ("estimator_weights_", "estimator_errors_"):
            v = getattr(e, attr, None)
            if v is not None:
                out.append((f"{tag}_m{i}_{attr}", _digest(np.asarray(v, dtype=np.float64))))
        for j, r in enumerate(getattr(e, "estimators_", None) or ()):
            v = getattr(r, "_leaves", None)
            if v is not None and hasattr(e, "estimator_weights_"):
                out.append((f"{tag}_m{i}_round{j}_leaves", _digest(v)))
    return out


def _bag(ml, **kw):
    return ml.BaggingClassifier(ml.RandomForestClassifier(n_estimators=5, random_state=0), n_estimators=4,
                                max_samples=0.8, random_state=0, **kw)


def _unit_base(ml, name):
    if name == "ovr":
        return ml.OneVsRestClassifier(ml.LogisticRegression(max_iter=200))
    if name == "calib":
        return ml.CalibratedClassifierCV(ml.LogisticRegression(max_iter=200), cv=3)
    return _bag(ml, bootstrap_features=True)


def _index_cv(n, folds=3):
    rows = np.arange(n)
    return [(rows[rows % folds != f], rows[rows % folds == f]) for f in range(folds)]


CONFIGS = _configs()


def _search_tables(model, tag):
    out = [(tag + "_best", str(model.best_params_)), (tag + "_best_index", str(model.best_index_)),
           (tag + "_best_score", repr(float(model.best_score_)))]
    for key in sorted(model.cv_results_):
        if key == "params" or key.startswith("param_"):
            out.append((f"{tag}_{key}", str(list(model.cv_results_[key]))))
        else:
            out.append((f"{tag}_{key}", _digest(np.asarray(model.cv_results_[key]))))
    return out


def _search_row(ml, row, tag, X, y, Xt):
    from mojolearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
    grid = {"C": [0.1, 1.0, 10.0]}
    if row[0] == "searchcal":
        _, scoring, method = row
        search = GridSearchCV(ml.LogisticRegression(), grid, cv=3, scoring=scoring)
        model = ml.CalibratedClassifierCV(search, method=method, cv=3, n_jobs=1).fit(X, y)
        out = _predicts(model, Xt, tag)
        for i, (e, _cals) in enumerate(model.calibrated_classifiers_):
            out += _search_tables(e, f"{tag}_pair{i}")
        return out
    if row[0] == "searchvote":
        search = RandomizedSearchCV(ml.LogisticRegression(), grid, n_iter=2, random_state=0, cv=3)
        model = ml.VotingClassifier([("search", search), ("gnb", ml.GaussianNB())], voting="soft").fit(X, y)
        return _predicts(model, Xt, tag) + _search_tables(model.estimators_[0], tag + "_search")
    Xr, _ = _data(5, 150, 0, d=5)
    yr = (Xr @ np.arange(1, 6, dtype=np.float32)).astype(np.float32)
    if row[0] == "searchcv":
        scores = cross_val_score(GridSearchCV(ml.Ridge(), {"alpha": [0.1, 1.0]}, cv=3), Xr, yr, cv=3)
        return [(tag + "_scores", _digest(np.asarray(scores, dtype=np.float64)))]
    if row[0] == "searcherr":
        search = GridSearchCV(ml.LogisticRegression(), {"C": [1.0, -1.0]}, cv=3, error_score=-1.0)
        model = ml.CalibratedClassifierCV(search, cv=3).fit(X, y)
        return _predicts(model, Xt, tag) + _search_tables(model.calibrated_classifiers_[0][0], tag)
    if row[0] == "searchtrain":
        search = GridSearchCV(ml.LogisticRegression(), grid, cv=3, return_train_score=True)
        model = ml.CalibratedClassifierCV(search, cv=3).fit(X, y)
        return _predicts(model, Xt, tag) + _search_tables(model.calibrated_classifiers_[0][0], tag)
    if row[0] == "searchr2":
        search = GridSearchCV(ml.Ridge(), {"alpha": [0.1, 1.0]}, cv=3, scoring="r2")
        model = ml.VotingRegressor([("search", search), ("lin", ml.LinearRegression())]).fit(Xr, yr)
        return [(tag + "_predict", _digest(model.predict(Xr[:20])))]
    if row[0] == "searchmulti":
        Y = np.stack([yr, yr * 0.5 + 1.0], axis=1).astype(np.float32)
        model = GridSearchCV(ml.MultiOutputRegressor(ml.Lasso()), {"estimator__alpha": [0.01, 0.1]}, cv=3).fit(Xr, Y)
        return [(tag + "_predict", _digest(model.predict(Xr[:20])))] + _search_tables(model, tag)
    search = GridSearchCV(ml.LogisticRegression(), grid, cv=3)
    model = ml.StackingClassifier([("search", search), ("lda", ml.LinearDiscriminantAnalysis())], cv=3).fit(X, y)
    explainer = ml.KernelExplainer(model, X[:20])
    phi = np.asarray(explainer.shap_values(Xt[:5]))
    return [(tag + "_values", _digest(phi))]


def _run(index):
    import mojolearn as ml
    row = CONFIGS[index]
    tag = "_".join(str(w) for w in row)
    X, y = _data(index % 3 + 1, 150, 3)
    Xt, _ = _data(97, 40, 3)
    try:
        if row[0] == "cal":
            _, base, method, ensemble, jobs = row
            est = _soft_vote(ml) if base == "vote" else (_stack(ml) if base == "stack" else
                                                         ml.VotingClassifier([("lr", ml.LogisticRegression()),
                                                                              ("gnb", ml.GaussianNB())],
                                                                             voting="hard"))
            model = ml.CalibratedClassifierCV(est, method=method, ensemble=ensemble, cv=3, n_jobs=jobs).fit(X, y)
            return _predicts(model, Xt, tag) + _members(model, tag)
        if row[0] == "nested_vote":
            model = ml.VotingClassifier([("inner", _soft_vote(ml)), ("lr", ml.LogisticRegression())],
                                        voting="soft").fit(X, y)
            return _predicts(model, Xt, tag) + _members(model, tag)
        if row[0] == "stack_of_vote":
            model = ml.StackingClassifier([("vote", _soft_vote(ml)), ("lda", ml.LinearDiscriminantAnalysis())],
                                          cv=3).fit(X, y)
            return _predicts(model, Xt, tag) + _members(model, tag)
        if row[0] == "shap":
            model = ml.StackingClassifier([("vote", _soft_vote(ml)), ("lda", ml.LinearDiscriminantAnalysis())],
                                          cv=3).fit(X, y)
            explainer = (ml.KernelExplainer if row[1] == "K" else ml.PermutationExplainer)(model, X[:20])
            phi = np.asarray(explainer.shap_values(Xt[:5]))
            return [(tag + "_values", _digest(phi)),
                    (tag + "_expected", _digest(np.asarray(explainer.expected_value, dtype=np.float64)))]
        if row[0] == "iterimp":
            Xr, _ = _data(11, 120, 0, d=5)
            Xr = Xr.copy()
            rng = np.random.default_rng(3)
            Xr[rng.random(Xr.shape) < 0.15] = np.nan
            if row[1] == "vote":
                est = ml.VotingRegressor([("ridge", ml.Ridge(alpha=1.0)), ("lin", ml.LinearRegression())])
            else:
                est = ml.StackingRegressor([("ridge", ml.Ridge(alpha=1.0)), ("lin", ml.LinearRegression())], cv=3,
                                           passthrough=True)
            imp = ml.IterativeImputer(estimator=est, max_iter=3, random_state=0).fit(Xr)
            return [(tag + "_transform", _digest(imp.transform(Xr)))]
        if row[0] == "search":
            from mojolearn.model_selection import GridSearchCV
            model = GridSearchCV(_soft_vote(ml), {"weights": [[1, 1], [1, 2]]}, cv=3).fit(X, y)
            return [(tag + "_best", str(model.best_params_))] + _predicts(model, Xt, tag)
        if row[0] == "unitcal":
            _, base, method, ensemble, jobs = row
            model = ml.CalibratedClassifierCV(_unit_base(ml, base), method=method, ensemble=ensemble, cv=3,
                                              n_jobs=jobs).fit(X, y)
            return _predicts(model, Xt, tag) + _members(model, tag)
        if row[0] == "unitsearch":
            from mojolearn.model_selection import GridSearchCV
            if row[1] == "bag":
                model = GridSearchCV(_bag(ml), {"n_estimators": [2, 3]}, cv=3).fit(X, y)
            else:
                model = GridSearchCV(_unit_base(ml, "ovr"), {"n_jobs": [1, 2]}, cv=3).fit(X, y)
            return ([(tag + "_best", str(model.best_params_))] + _predicts(model, Xt, tag)
                    + _members(model.best_estimator_, tag))
        if row[0] == "unitshap":
            model = (_unit_base(ml, "ovr") if row[1] == "ovr" else _bag(ml)).fit(X, y)
            explainer = (ml.KernelExplainer if row[2] == "K" else ml.PermutationExplainer)(model, X[:20])
            phi = np.asarray(explainer.shap_values(Xt[:5]))
            return [(tag + "_values", _digest(phi)),
                    (tag + "_expected", _digest(np.asarray(explainer.expected_value, dtype=np.float64)))]
        if row[0] == "unititerimp":
            Xr, _ = _data(11, 120, 0, d=5)
            Xr = Xr.copy()
            rng = np.random.default_rng(3)
            Xr[rng.random(Xr.shape) < 0.15] = np.nan
            est = ml.BaggingRegressor(ml.RandomForestRegressor(n_estimators=4, random_state=0), n_estimators=3,
                                      max_samples=0.8, random_state=0)
            imp = ml.IterativeImputer(estimator=est, max_iter=2, random_state=0).fit(Xr)
            return [(tag + "_transform", _digest(imp.transform(Xr)))]
        if row[0] == "vote":
            if row[1] == "stack_usercv":
                inner = ml.StackingClassifier([("lr", ml.LogisticRegression(max_iter=200)),
                                               ("lda", ml.LinearDiscriminantAnalysis())], cv=_index_cv(len(X)))
            elif row[1] == "stack_predict":
                inner = ml.StackingClassifier([("lr", ml.LogisticRegression(max_iter=200)),
                                               ("lda", ml.LinearDiscriminantAnalysis())], cv=3,
                                              stack_method="predict")
            else:
                inner = _bag(ml, oob_score=True)
            model = ml.VotingClassifier([("inner", inner), ("lr", ml.LogisticRegression(max_iter=200))],
                                        voting="soft").fit(X, y)
            out = _predicts(model, Xt, tag) + _members(model, tag)
            for attr in ("oob_score_", "oob_decision_function_"):
                v = getattr(model.estimators_[0], attr, None)
                if v is not None:
                    out.append((f"{tag}_{attr}", _digest(v)))
            return out
        if row[0] == "batchcal":
            _, method, ensemble = row
            model = ml.CalibratedClassifierCV(ml.BaggingClassifier(n_estimators=4, random_state=0), method=method,
                                              ensemble=ensemble, cv=3, n_jobs=1).fit(X, y)
            return _predicts(model, Xt, tag) + _members(model, tag)
        if row[0] == "batchvote":
            kw = {"n_estimators": 4, "random_state": 0}
            if row[1] == "every_row":
                kw.update(bootstrap=False, max_samples=1.0)
            elif row[1] == "subset":
                kw.update(max_features=0.6)
            model = ml.VotingClassifier([("bag", ml.BaggingClassifier(**kw)),
                                         ("lr", ml.LogisticRegression(max_iter=200))], voting="soft")
            if row[1] == "weighted":
                w = np.linspace(0.5, 1.5, len(X)).astype(np.float32)
                model.fit(X, y, sample_weight=w)
            else:
                model.fit(X, y)
            bag = model.estimators_[0]
            return (_predicts(model, Xt, tag) + _members(model, tag)
                    + [(tag + "_bag_proba", _digest(bag.predict_proba(Xt))),
                       (tag + "_bag_batched", str(getattr(bag, "_batched", None) is not None))])
        if row[0] == "batchiterimp":
            Xr, _ = _data(11, 120, 0, d=5)
            Xr = Xr.copy()
            rng = np.random.default_rng(3)
            Xr[rng.random(Xr.shape) < 0.15] = np.nan
            imp = ml.IterativeImputer(estimator=ml.BaggingRegressor(n_estimators=3, random_state=0), max_iter=2,
                                      random_state=0).fit(Xr)
            return [(tag + "_transform", _digest(imp.transform(Xr)))]
        if row[0] == "deep":
            if row[1] == "cal_stack_vote":
                inner = ml.StackingClassifier([("vote", _soft_vote(ml)), ("lda", ml.LinearDiscriminantAnalysis())],
                                              cv=3)
                model = ml.CalibratedClassifierCV(inner, cv=3).fit(X, y)
                return _predicts(model, Xt, tag) + _members(model, tag)
            if row[1] == "vote3":
                v2 = ml.VotingClassifier([("v1", _soft_vote(ml)), ("lr", ml.LogisticRegression(max_iter=200))],
                                         voting="soft")
                model = ml.VotingClassifier([("v2", v2), ("gnb", ml.GaussianNB())], voting="soft").fit(X, y)
                out = _predicts(model, Xt, tag) + _members(model, tag)
                return out + _members(model.estimators_[0], tag + "_v2") + _members(
                    model.estimators_[0].estimators_[0], tag + "_v1")
            from mojolearn.model_selection import GridSearchCV
            est = ml.VotingClassifier([("inner", _soft_vote(ml)), ("lr", ml.LogisticRegression(max_iter=200))],
                                      voting="soft")
            model = GridSearchCV(est, {"weights": [[1, 1], [1, 2]]}, cv=3).fit(X, y)
            return ([(tag + "_best", str(model.best_params_))] + _predicts(model, Xt, tag)
                    + _members(model.best_estimator_, tag) + _members(model.best_estimator_.estimators_[0],
                                                                       tag + "_inner"))
        if row[0] == "adacal":
            _, method, ensemble = row
            model = ml.CalibratedClassifierCV(_ada(ml), method=method, ensemble=ensemble, cv=3, n_jobs=1).fit(X, y)
            out = _predicts(model, Xt, tag)
            for i, (e, _cals) in enumerate(model.calibrated_classifiers_):
                out.append((f"{tag}_pair{i}_weights", _digest(np.asarray(e.estimator_weights_, dtype=np.float64))))
                out.append((f"{tag}_pair{i}_decision", _digest(e.decision_function(Xt))))
            return out
        if row[0] == "adavote":
            model = ml.VotingClassifier([("ada", _ada(ml, row[1])), ("lr", ml.LogisticRegression(max_iter=200))],
                                        voting="soft").fit(X, y)
            return _predicts(model, Xt, tag) + _members(model, tag) + _ada_attrs(model, tag)
        if row[0] == "adastack":
            model = ml.StackingClassifier([("ada", _ada(ml)), ("lda", ml.LinearDiscriminantAnalysis())],
                                          cv=3).fit(X, y)
            return _predicts(model, Xt, tag) + _members(model, tag) + _ada_attrs(model, tag)
        if row[0] == "adashap":
            model = ml.StackingClassifier([("ada", _ada(ml)), ("lda", ml.LinearDiscriminantAnalysis())],
                                          cv=3).fit(X, y)
            explainer = (ml.KernelExplainer if row[1] == "K" else ml.PermutationExplainer)(model, X[:20])
            phi = np.asarray(explainer.shap_values(Xt[:5]))
            return [(tag + "_values", _digest(phi)),
                    (tag + "_expected", _digest(np.asarray(explainer.expected_value, dtype=np.float64)))]
        if row[0] == "adaiterimp":
            Xr, _ = _data(11, 120, 0, d=5)
            Xr = Xr.copy()
            rng = np.random.default_rng(3)
            Xr[rng.random(Xr.shape) < 0.15] = np.nan
            base = (ml.RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0) if row[1] == "rf"
                    else None)
            est = ml.AdaBoostRegressor(base, n_estimators=4, loss="square", random_state=0)
            imp = ml.IterativeImputer(estimator=est, max_iter=2, random_state=0).fit(Xr)
            return [(tag + "_transform", _digest(imp.transform(Xr)))]
        if row[0] == "adasearch":
            from mojolearn.model_selection import GridSearchCV
            model = GridSearchCV(ml.AdaBoostClassifier(random_state=0), {"n_estimators": [3, 5]}, cv=3).fit(X, y)
            return ([(tag + "_best", str(model.best_params_))] + _predicts(model, Xt, tag)
                    + [(tag + "_best_weights", _digest(np.asarray(model.best_estimator_.estimator_weights_,
                                                                  dtype=np.float64)))])
        if row[0].startswith("search"):
            return _search_row(ml, row, tag, X, y, Xt)
    except Exception as e:  # noqa: BLE001 - the words are the digest
        return [(tag + "_raised", f"{type(e).__name__}: {e}")]
    return []


def main():
    with ProcessPoolExecutor(max_workers=min(len(CONFIGS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(CONFIGS))):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
