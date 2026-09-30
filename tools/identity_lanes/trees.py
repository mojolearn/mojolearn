# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE TREES LANE'S IDENTITY LANES.
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


@lane("trees-dt-random")
def _(ml, X, yc, yr, Xh=None):
    """splitter='random': one ExtraTrees tree; best-first growth to 24 leaves."""
    c = ml.DecisionTreeClassifier(splitter="random", max_depth=8, random_state=5).fit(X, yc)
    r = ml.DecisionTreeRegressor(splitter="random", max_leaf_nodes=24, random_state=5).fit(X, yr)
    return _fit(dict(proba=_h(c.predict_proba(X)), predict=_h(c.predict(X)), reg=_h(r.predict(X)),
                     leaves=_h(np.int64(r.get_n_leaves()))),
                c, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-dt-random")


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


@lane("trees-dart-options")
def _(ml, X, yc, yr, Xh=None):
    """DART's LightGBM options: L1 and max_delta_step leaves, per-tree feature
    sampling, row bagging; multiclass softmax (one tree per class) and its TreeSHAP."""
    r = ml.DARTRegressor(n_estimators=8, num_leaves=15, max_depth=5, min_child_samples=5, drop_rate=0.5,
                         skip_drop=0.0, reg_lambda=1.0, reg_alpha=2.0, max_delta_step=40.0, colsample_bytree=0.6,
                         subsample=0.7, subsample_freq=2, random_state=7).fit(X, yr)
    m = ml.DARTClassifier(n_estimators=6, num_leaves=15, max_depth=5, min_child_samples=5, drop_rate=0.5,
                          skip_drop=0.0, reg_alpha=0.5, max_delta_step=0.7, colsample_bytree=0.8, subsample=0.8,
                          subsample_freq=1, random_state=7).fit(X, yc)
    e = ml.TreeExplainer(m, data=X[:128])
    return _fit(dict(reg=_h(r.predict(X)), predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)),
                     raw=_h(m.decision_function(X)), shap=_h(e.shap_values(X[:32])),
                     ev=_h(np.asarray(e.expected_value))),
                m, lambda x: (x.predict(Xh), x.predict_proba(Xh)))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-dart-options")


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


class _trees_Shuffled3:
    """A splitter object (sklearn's `split(X, y)` protocol): three folds of a
    fixed row permutation, so the cv-object path runs without sklearn."""

    def split(self, X, y=None):
        n = len(X)
        perm = np.random.RandomState(11).permutation(n)
        for f in range(3):
            te = np.sort(perm[f::3])
            yield np.setdiff1d(np.arange(n), te), te


@lane("trees-oob-cv-link")
def _(ml, X, yc, yr, Xh=None):
    """Bagging oob_score (classifier and regressor), cv as a splitter object and
    as (train, test) pairs (stacking, calibration), Kernel SHAP with link='logit'."""
    bc = ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=5), n_estimators=6, max_features=0.8,
                              oob_score=True, random_state=7).fit(X, yc)
    br = ml.BaggingRegressor(ml.DecisionTreeRegressor(max_depth=5), n_estimators=6, max_samples=0.8,
                             oob_score=True, random_state=7).fit(X, yr)
    n = len(X)
    pairs = [(np.arange(n)[np.arange(n) % 2 != f], np.arange(n)[np.arange(n) % 2 == f]) for f in (0, 1)]
    st = ml.StackingRegressor([("dt", ml.DecisionTreeRegressor(max_depth=4))],
                              final_estimator=ml.DecisionTreeRegressor(max_depth=3), cv=pairs).fit(X, yr)
    ca = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), method="isotonic",
                                   cv=_trees_Shuffled3(), ensemble=False).fit(X, yc)
    y2 = (np.asarray(yc) % 2).astype(np.int64)
    X6 = np.ascontiguousarray(X[:, :6])
    dm = ml.DARTClassifier(n_estimators=4, num_leaves=7, max_depth=3, min_child_samples=5, random_state=7).fit(X6, y2)
    ke = ml.KernelExplainer(dm, X6[:10], link="logit")
    return _fit(dict(oob_c=_h(np.float64(bc.oob_score_)), oob_df=_h(bc.oob_decision_function_),
                     oob_r=_h(np.float64(br.oob_score_)), oob_p=_h(br.oob_prediction_),
                     bag=_h(bc.predict_proba(X)), stack=_h(st.predict(X)), cal=_h(ca.predict_proba(X)),
                     shap=_h(ke.shap_values(X6[:3])), ev=_h(np.asarray(ke.expected_value))),
                ca, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-oob-cv-link")


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


@lane("trees-et-deviance")
def _(ml, X, yc, yr, Xh=None):
    """ExtraTreesRegressor on cuML's deviance criteria (Poisson, Gamma,
    InverseGaussian) over a strictly positive target; one bootstrap forest,
    one best-first."""
    y = _pos(yr)
    p = ml.ExtraTreesRegressor(n_estimators=8, max_depth=8, random_state=7, criterion="poisson").fit(X, y)
    g = ml.ExtraTreesRegressor(n_estimators=6, max_depth=7, random_state=7, criterion="gamma",
                               bootstrap=True, max_samples=0.7).fit(X, y)
    i = ml.ExtraTreesRegressor(n_estimators=4, max_leaf_nodes=24, random_state=7,
                               criterion="inverse_gaussian").fit(X, y)
    return _fit(dict(poisson=_h(p.predict(X)), gamma=_h(g.predict(X)), ig=_h(i.predict(X))),
                p, lambda e: (e.predict(Xh),))


_batch_decl(_rows_calls("predict"), "trees-et-deviance")


def _trees_multi_target(X, yr):
    """Three regression targets, `(n, 3)`, derived from the fixture's own
    regression target and two columns by one float32 operation each (exact
    IEEE elementwise, so every host builds the same bytes)."""
    x = np.asarray(X, dtype=np.float32)
    y = np.asarray(yr, dtype=np.float32)
    return np.ascontiguousarray(np.stack(
        [y, np.float32(0.5) * y + x[:, 0], x[:, 1] - x[:, 2]], axis=1
    ).astype(np.float32))


@lane("trees-gbdt-multirmse")
def _(ml, X, yc, yr, Xh=None):
    """GradientBoosting loss='MultiRMSE' on three targets: 12 depth-6
    SymmetricTree trees, Newton leaves through the BLOCKED Hessian (their
    `GetHessianType()` is Symmetric for MultiRMSE, `multiclass_targets.h:
    118-123`), predict RAW `(n, 3)`."""
    m = _gbdt(ml.GradientBoosting, n_estimators=12, max_depth=6, loss="MultiRMSE").fit(X, _trees_multi_target(X, yr))
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


_batch_decl(_rows_calls("predict"), "trees-gbdt-multirmse")


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


@lane("trees-shap-permutation")
def _(ml, X, yc, yr, Xh=None):
    """Permutation SHAP over a classifier's probabilities."""
    m = ml.DecisionTreeClassifier(max_depth=5).fit(X, yc)
    p = ml.PermutationExplainer(m, X[:8], random_state=5)
    return _fit(dict(values=_h(p.shap_values(X[:4], npermutations=6)), ev=_h(np.asarray(p.expected_value))),
                p, lambda e: (e.shap_values(Xh[:3], npermutations=4),))


_batch_decl("n/a:position-seeded (each explained row draws its permutations from the stream of its position"
            " in the call, as shap's sequential RNG does)", "trees-shap-permutation")
