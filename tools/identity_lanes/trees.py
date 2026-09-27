# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE TREES LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `trees` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("trees-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "trees-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_trees_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


@lane("trees-dt-clf")
def _(ml, X, yc, yr, Xh=None):
    """One CART tree, entropy, every feature (the weighted entry has no CPU arm yet)."""
    m = ml.DecisionTreeClassifier(max_depth=8, criterion="entropy", random_state=7).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)), depth=_h(np.int64(m.get_depth()))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("trees-dt-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, random_state=7).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X)), leaves=_h(np.int64(m.get_n_leaves()))),
                m, lambda e: (e.predict(Xh),))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-dt-clf")
_batch_decl(_rows_calls("predict"), "trees-dt-reg")


@lane("trees-bagging-clf")
def _(ml, X, yc, yr, Xh=None):
    """Bootstrap rows, a feature subset without replacement, proba averaging."""
    m = ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=6), n_estimators=6, max_samples=0.8,
                             max_features=0.75, random_state=7).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("trees-bagging-reg")
def _(ml, X, yc, yr, Xh=None):
    """Rows without replacement, bootstrapped features, the members' mean."""
    m = ml.BaggingRegressor(ml.DecisionTreeRegressor(max_depth=6), n_estimators=6, max_samples=0.7,
                            bootstrap=False, bootstrap_features=True, random_state=7).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-bagging-clf")
_batch_decl(_rows_calls("predict"), "trees-bagging-reg")


@lane("trees-adaboost-clf")
def _(ml, X, yc, yr, Xh=None):
    """SAMME: weighted stumps-plus (depth 2), the weight update and the vote."""
    m = ml.AdaBoostClassifier(ml.DecisionTreeClassifier(max_depth=2), n_estimators=8, learning_rate=0.8,
                              random_state=7).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)),
                     weights=_h(np.asarray(m.estimator_weights_, dtype=np.float64))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("trees-adaboost-reg")
def _(ml, X, yc, yr, Xh=None):
    """AdaBoost.R2: the weighted bootstrap, the square loss, the weighted median."""
    m = ml.AdaBoostRegressor(ml.DecisionTreeRegressor(max_depth=3), n_estimators=6, loss="square",
                             random_state=7).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X)), weights=_h(np.asarray(m.estimator_weights_, dtype=np.float64))),
                m, lambda e: (e.predict(Xh),))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-adaboost-clf")
_batch_decl(_rows_calls("predict"), "trees-adaboost-reg")


@lane("trees-dart-reg")
def _(ml, X, yc, yr, Xh=None):
    """DART, L2: drops forced often (skip_drop 0, drop_rate 0.5), Newton leaves."""
    m = ml.DARTRegressor(n_estimators=8, num_leaves=15, max_depth=5, min_child_samples=5, drop_rate=0.5,
                         skip_drop=0.0, reg_lambda=1.0, drop_seed=3, random_state=7).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X)), coefs=_h(np.asarray(m.tree_coefs_, dtype=np.float64))),
                m, lambda e: (e.predict(Xh),))


@lane("trees-dart-clf")
def _(ml, X, yc, yr, Xh=None):
    """DART, binary logloss (yc folded to two classes), xgboost_dart_mode, uniform_drop."""
    y2 = (np.asarray(yc) % 2).astype(np.int64)
    m = ml.DARTClassifier(n_estimators=8, num_leaves=15, max_depth=5, min_child_samples=5, drop_rate=0.5,
                          skip_drop=0.0, xgboost_dart_mode=True, uniform_drop=True, random_state=7).fit(X, y2)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-dart-clf")
_batch_decl(_rows_calls("predict"), "trees-dart-reg")


@lane("trees-random-embedding")
def _(ml, X, yc, yr, Xh=None):
    """Random uniform targets, ExtraTrees with one feature per split, the leaf one-hot."""
    m = ml.RandomTreesEmbedding(n_estimators=8, max_depth=4, random_state=7).fit(X)
    return _fit(dict(embedding=_h(m.transform(X)), leaves=_h(m.apply(X))),
                m, lambda e: (e.transform(Xh),))


_batch_decl(_rows_calls("transform", "apply"), "trees-random-embedding")


@lane("trees-voting-clf")
def _(ml, X, yc, yr, Xh=None):
    """Soft voting with weights over three different members."""
    m = ml.VotingClassifier([("dt", ml.DecisionTreeClassifier(max_depth=5)),
                             ("rf", ml.RandomForestClassifier(n_estimators=4, max_depth=5, random_state=3)),
                             ("bag", ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=4), n_estimators=3,
                                                          random_state=5))],
                            voting="soft", weights=[1.0, 2.0, 0.5]).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)), transform=_h(m.transform(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("trees-voting-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.VotingRegressor([("dt", ml.DecisionTreeRegressor(max_depth=5)),
                            ("rf", ml.RandomForestRegressor(n_estimators=4, max_depth=5, random_state=3))],
                           weights=[3.0, 1.0]).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-voting-clf")
_batch_decl(_rows_calls("predict"), "trees-voting-reg")


@lane("trees-stacking-clf")
def _(ml, X, yc, yr, Xh=None):
    """Three stratified folds of held-out probabilities, passthrough, a tree on top."""
    m = ml.StackingClassifier([("dt", ml.DecisionTreeClassifier(max_depth=4)),
                               ("rf", ml.RandomForestClassifier(n_estimators=3, max_depth=4, random_state=3))],
                              final_estimator=ml.DecisionTreeClassifier(max_depth=3), cv=3,
                              passthrough=True).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)), meta=_h(m.transform(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("trees-stacking-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.StackingRegressor([("dt", ml.DecisionTreeRegressor(max_depth=4)),
                              ("rf", ml.RandomForestRegressor(n_estimators=3, max_depth=4, random_state=3))],
                             final_estimator=ml.DecisionTreeRegressor(max_depth=3), cv=3).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-stacking-clf")
_batch_decl(_rows_calls("predict"), "trees-stacking-reg")


@lane("trees-multioutput")
def _(ml, X, yc, yr, Xh=None):
    """A regressor per column of Y, and a classifier per column of a label matrix."""
    Y = np.stack([yr, yr[::-1].copy()], axis=1).astype(np.float32)
    r = ml.MultiOutputRegressor(ml.DecisionTreeRegressor(max_depth=4)).fit(X, Y)
    Yc = np.stack([np.asarray(yc), np.asarray(yc) % 2], axis=1).astype(np.int64)
    c = ml.MultiOutputClassifier(ml.DecisionTreeClassifier(max_depth=4)).fit(X, Yc)
    return _fit(dict(reg=_h(r.predict(X)), clf=_h(c.predict(X)), proba=_h(*c.predict_proba(X))),
                r, lambda e: (e.predict(Xh),))


_batch_decl(_rows_calls("predict"), "trees-multioutput")


@lane("trees-onevsrest")
def _(ml, X, yc, yr, Xh=None):
    m = ml.OneVsRestClassifier(ml.DecisionTreeClassifier(max_depth=4)).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-onevsrest")


@lane("trees-calibrated")
def _(ml, X, yc, yr, Xh=None):
    """Sigmoid (Platt, per fold, ensembled) and isotonic (cross-validated scores) calibration."""
    s = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), method="sigmoid", cv=3).fit(X, yc)
    i = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), method="isotonic", cv=3,
                                  ensemble=False).fit(X, yc)
    return _fit(dict(sigmoid=_h(s.predict_proba(X)), isotonic=_h(i.predict_proba(X)), predict=_h(s.predict(X))),
                s, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-calibrated")


@lane("trees-rf-weighted")
def _(ml, X, yc, yr, Xh=None):
    """The RF weighted objective (class weights without bootstrap): a balanced
    entropy forest on column samples, and a gini tree on per-row sample weights
    that zero some rows out."""
    f = ml.RandomForestClassifier(n_estimators=4, max_depth=6, random_state=7, class_weight="balanced",
                                  bootstrap=False, criterion="entropy", max_features=0.6).fit(X, yc)
    w = np.array([(1.0, 2.5, 0.0, 0.75)[i % 4] for i in range(len(X))], dtype=np.float32)
    t = ml.DecisionTreeClassifier(max_depth=7, min_samples_leaf=2).fit(X, yc, sample_weight=w)
    return _fit(dict(forest=_h(f.predict_proba(X)), tree=_h(t.predict_proba(X)), predict=_h(t.predict(X))),
                t, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-rf-weighted")


@lane("trees-shap-tree")
def _(ml, X, yc, yr, Xh=None):
    """Exact TreeSHAP over a classifier forest (three outputs) and a DART model."""
    f = ml.RandomForestClassifier(n_estimators=4, max_depth=6, random_state=7).fit(X, yc)
    e = ml.TreeExplainer(f, data=X[:256])
    dm = ml.DARTRegressor(n_estimators=6, num_leaves=15, max_depth=5, min_child_samples=5, random_state=7).fit(X, yr)
    de = ml.TreeExplainer(dm, data=X[:256])
    return _fit(dict(forest=_h(e.shap_values(X[:64])), forest_ev=_h(np.asarray(e.expected_value)),
                     dart=_h(de.shap_values(X[:64])), dart_ev=_h(np.float64(de.expected_value))),
                e, lambda x: (x.shap_values(Xh[:64]),))


_batch_decl(_rows_calls("shap_values", sl=slice(0, 24)), "trees-shap-tree")


@lane("trees-shap-kernel")
def _(ml, X, yc, yr, Xh=None):
    """Kernel SHAP on six features (full coalition enumeration) and on all of
    them (sampled coalitions, the counter RNG), over a regression tree."""
    X6 = np.ascontiguousarray(X[:, :6])
    m6 = ml.DecisionTreeRegressor(max_depth=5).fit(X6, yr)
    k6 = ml.KernelExplainer(m6, X6[:12])
    m = ml.DecisionTreeRegressor(max_depth=5).fit(X, yr)
    k = ml.KernelExplainer(m, X[:6], random_state=3)
    return _fit(dict(full=_h(k6.shap_values(X6[:4])), sampled=_h(k.shap_values(X[:2], nsamples=300))),
                k6, lambda e: (e.shap_values(np.ascontiguousarray(Xh[:3, :6])),))


_batch_decl(_rows_calls("shap_values", sl=slice(0, 12), prep=lambda Xh: np.ascontiguousarray(Xh[:, :6])),
            "trees-shap-kernel")
