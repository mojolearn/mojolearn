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
    """One CART tree, entropy, every feature (the weighted entry has no CPU arm yet); the same tree
    on a float64 Fortran-order copy of 2000 rows, the column-major fit entry (rf_classifier_fit_export:
    every fixture is float32 C-order, which takes the row-major entry). Part `saved` (lane w3i11-endpoints,
    2026-11-25): the bytes `save` writes for the fitted tree (a saved file does not depend on the device)."""
    m = ml.DecisionTreeClassifier(max_depth=8, criterion="entropy", random_state=7).fit(X, yc)
    mf = ml.DecisionTreeClassifier(max_depth=8, criterion="entropy", random_state=7).fit(
        np.asfortranarray(X[:2000], dtype=np.float64), yc[:2000])
    with tempfile.TemporaryDirectory(prefix="ib-trees-dt-save-") as d:
        path = os.path.join(d, "tree.mlf")
        m.save(path)
        with open(path, "rb") as fh:
            saved = np.frombuffer(fh.read(), dtype=np.uint8)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)), depth=_h(np.int64(m.get_depth())),
                     colmajor=_h(mf.predict_proba(X)), saved=_h(saved)),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("trees-dt-reg")
def _(ml, X, yc, yr, Xh=None):
    """One CART regression tree; the same tree on a float64 Fortran-order copy of 2000 rows, the
    column-major fit entry (rf_regressor_fit_export)."""
    m = ml.DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, random_state=7).fit(X, yr)
    mf = ml.DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, random_state=7).fit(
        np.asfortranarray(X[:2000], dtype=np.float64), yr[:2000])
    return _fit(dict(predict=_h(m.predict(X)), leaves=_h(np.int64(m.get_n_leaves())), colmajor=_h(mf.predict(X))),
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
    # lane w3f3-i (2026-11-24): the mf-del8 `lacking` part (six-row draws over a rare class) is moved to
    # trees-stacking-clf. Its bag (a best-split tree base, every feature, bootstrap, no weights) is one batched
    # forest fit (bindings/trees_public_boost_door.mojo `_bagging_batched_ok`), which never reads THE UNIT CLASSES,
    # so the coverage-mf-del8-unit_classes arm could not move it (gate2-af00 sabotage: MISSED).
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
                     weights=_h(np.asarray(m.estimator_weights_, dtype=np.float64)),
                     score=_h(np.float64(m.score(X[:16], yc[:16])))),
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
@floor(rows=(4097, "every member's root node must exceed T05_NODE_ROWS = 4096 (ensemble/decisiontree/batched_levelalgo/"
                   "builder.mojo:209) so the multi-block histogram route runs at the root and the small-node walk "
                   "below it; voting fits each member on all rows; 2026-11-05 lane/slow-verification-lanes"))
def _(ml, X, yc, yr, Xh=None):
    """Soft voting with weights over three different members. The members fit
    on the first `rows` rows and predict the whole fixture."""
    # 5001 rows (2026-11-05, lane/slow-verification-lanes; was the whole
    # 20000-row fixture): above the 4096-row T05 small-node bound, so every
    # member's root takes the multi-block histogram and its children the fused
    # small-node walk, both as before; above the 512-row quantile RNG arm
    # (ensemble/decisiontree/batched_levelalgo/quantiles.mojo:53); odd, so the
    # last histogram block is partial. Fewer trees and one level less per
    # member: the member count and depth are not what the lane checks.
    rows = 5001
    m = ml.VotingClassifier([("dt", ml.DecisionTreeClassifier(max_depth=4)),
                             ("rf", ml.RandomForestClassifier(n_estimators=2, max_depth=4, random_state=3)),
                             ("bag", ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=3), n_estimators=2,
                                                          random_state=5))],
                            voting="soft", weights=[1.0, 2.0, 0.5]).fit(X[:rows], yc[:rows])
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)), transform=_h(m.transform(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("trees-voting-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.VotingRegressor([("dt", ml.DecisionTreeRegressor(max_depth=5)),
                            ("rf", ml.RandomForestRegressor(n_estimators=4, max_depth=5, random_state=3))],
                           weights=[3.0, 1.0]).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


# 2001 held-out rows for the batch protocol (2026-11-05, lane/slow-verification-lanes; was all 20000):
# odd, eight 256-thread blocks with a partial last one; the per-row and split checks are unchanged.
_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 2001)), "trees-voting-clf")
_batch_decl(_rows_calls("predict"), "trees-voting-reg")


@lane("trees-stacking-clf")
@floor(rows=(6145, "each of the three stratified folds trains a member on 2/3 of the rows, which must exceed "
                   "T05_NODE_ROWS = 4096 (ensemble/decisiontree/batched_levelalgo/builder.mojo:209) so the fold "
                   "fits keep the multi-block root and small-node children; 2026-11-05 lane/slow-verification-lanes"))
def _(ml, X, yc, yr, Xh=None):
    """Three stratified folds of held-out probabilities, passthrough, a tree on
    top. Fitted on the first `rows` rows; predicts the whole fixture."""
    # 7001 rows (2026-11-05, lane/slow-verification-lanes; was the whole
    # 20000-row fixture): a fold's training part is ~4667 rows, above the
    # 4096-row T05 small-node bound and the 512-row quantile RNG arm, so the
    # member fits take the same routes as before; odd, so blocks and folds are
    # uneven. One level less and two forest trees: not what the lane checks.
    rows = 7001
    m = ml.StackingClassifier([("dt", ml.DecisionTreeClassifier(max_depth=3)),
                               ("rf", ml.RandomForestClassifier(n_estimators=2, max_depth=3, random_state=3))],
                              final_estimator=ml.DecisionTreeClassifier(max_depth=3), cv=3,
                              passthrough=True).fit(X[:rows], yc[:rows])
    # lane w3f3-i (2026-11-24): a cv fold whose training rows lack a class. 60 rows, one row of class 2: the three
    # stratified folds put it in one test fold, so that fold's training rows hold classes 0 and 1 only and its
    # member is fitted on the classes it holds, its codes re-coded onto them (THE UNIT CLASSES,
    # core/member_unit_classes.mojo `member_unit_recode`, called by bindings/trees_public_meta_door.mojo
    # `_fit_units`), its probability columns placed at their classes (scikit-learn `_enforce_prediction_order`:
    # scikit-learn 1.9 cross_val_predict fits that fold's member on y[train] and warns "Number of classes in
    # training fold (2) does not match total number of classes (3)"). The final tree is fitted on those stacked
    # columns, so a broken re-code moves `lacking`.
    # w3f5-e (2026-11-24): the 60 rows are every 117th of the floored `rows` (0, 117, ..., 6903), not `X[:60]`: a
    # literal row cut of the fixture in a floored lane is what tools/fixture_floors.py refuses (rule 3), since it
    # can be shrunk without walking past `rows = 7001` and its reason.
    xs = np.ascontiguousarray(X[:rows:117], dtype=np.float32)
    ys = (xs[:, 0] > np.median(xs[:, 0])).astype(np.int64)
    ys[int(np.argmax(xs[:, 0]))] = 2
    s = ml.StackingClassifier([("dt", ml.DecisionTreeClassifier(max_depth=3))],
                              final_estimator=ml.DecisionTreeClassifier(max_depth=3), cv=3).fit(xs, ys)
    # The same stack as a MEMBER of a soft vote: fitted by THE META MEMBER KIND (core/ensemble_member.mojo
    # `_held_and_codes`), whose lacking fold reads the same re-code body (`member_unit_recode_rows`).
    v = ml.VotingClassifier([("st", ml.StackingClassifier([("dt", ml.DecisionTreeClassifier(max_depth=3))],
                                                          final_estimator=ml.DecisionTreeClassifier(max_depth=3),
                                                          cv=3))], voting="soft").fit(xs, ys)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)), meta=_h(m.transform(X)),
                     lacking=_h(s.predict_proba(xs)), lacking_predict=_h(s.predict(xs)),
                     lacking_member=_h(v.predict_proba(xs))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("trees-stacking-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.StackingRegressor([("dt", ml.DecisionTreeRegressor(max_depth=4)),
                              ("rf", ml.RandomForestRegressor(n_estimators=3, max_depth=4, random_state=3))],
                             final_estimator=ml.DecisionTreeRegressor(max_depth=3), cv=3).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


# 2001 held-out rows for the batch protocol (2026-11-05, lane/slow-verification-lanes; was all 20000):
# odd, eight 256-thread blocks with a partial last one; the per-row and split checks are unchanged.
_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 2001)), "trees-stacking-clf")
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
@floor(rows=(6145, "each of the three calibration folds fits its tree on 2/3 of the rows, which must exceed "
                   "T05_NODE_ROWS = 4096 (ensemble/decisiontree/batched_levelalgo/builder.mojo:209), keeping the "
                   "multi-block root split in every fold fit; 2026-11-05 lane/slow-verification-lanes"))
def _(ml, X, yc, yr, Xh=None):
    """Sigmoid (Platt, per fold, ensembled) and isotonic (cross-validated
    scores) calibration. Fitted on the first `rows` rows; predicts the whole
    fixture."""
    # 7001 rows (2026-11-05, lane/slow-verification-lanes; was the whole
    # 20000-row fixture): a fold's training part is ~4667 rows, above the
    # 4096-row T05 small-node bound and the 512-row quantile RNG arm, so the
    # fold trees take the same routes as before; the calibrators still see
    # ~2334 held-out scores per fold. Both methods, cv=3 and ensemble on/off stay.
    rows = 7001
    s = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=3), method="sigmoid", cv=3).fit(
        X[:rows], yc[:rows])
    i = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=3), method="isotonic", cv=3,
                                  ensemble=False).fit(X[:rows], yc[:rows])
    return _fit(dict(sigmoid=_h(s.predict_proba(X)), isotonic=_h(i.predict_proba(X)), predict=_h(s.predict(X))),
                s, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


# 2001 held-out rows for the batch protocol (2026-11-05, lane/slow-verification-lanes; was all 20000):
# odd, eight 256-thread blocks with a partial last one; the per-row and split checks are unchanged.
_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 2001)), "trees-calibrated")


@lane("trees-oob-cv-link")
def _(ml, X, yc, yr, Xh=None):
    """Bagging oob_score (classifier and regressor), cv as a splitter object and
    as (train, test) pairs (stacking, calibration), Kernel SHAP with link='logit'.
    Revised 2026-10-25 (lane verify-lane-contracts-20261025): calibration's
    splitter object was a Python class with split(X, y), which the native
    owner refuses (no Python protocol in the runtime) and the lane read
    REFUSED; it is now the native splitter KFold(3, shuffle=True). Random
    forests with oob_score=True and feature_importances=True (the aux fit
    entries) added 2026-10-26 (lane precise-coverage-0859)."""
    bc = ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=5), n_estimators=6, max_features=0.8,
                              oob_score=True, random_state=7).fit(X, yc)
    br = ml.BaggingRegressor(ml.DecisionTreeRegressor(max_depth=5), n_estimators=6, max_samples=0.8,
                             oob_score=True, random_state=7).fit(X, yr)
    n = len(X)
    pairs = [(np.arange(n)[np.arange(n) % 2 != f], np.arange(n)[np.arange(n) % 2 == f]) for f in (0, 1)]
    st = ml.StackingRegressor([("dt", ml.DecisionTreeRegressor(max_depth=4))],
                              final_estimator=ml.DecisionTreeRegressor(max_depth=3), cv=pairs).fit(X, yr)
    ca = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), method="isotonic",
                                   cv=ml.model_selection.KFold(3, shuffle=True, random_state=11),
                                   ensemble=False).fit(X, yc)
    y2 = (np.asarray(yc) % 2).astype(np.int64)
    X6 = np.ascontiguousarray(X[:, :6])
    dm = ml.DARTClassifier(n_estimators=4, num_leaves=7, max_depth=3, min_child_samples=5, random_state=7).fit(X6, y2)
    ke = ml.KernelExplainer(dm, X6[:10], link="logit")
    # The forests' requested fit outputs (oob_score=True, feature_importances=True)
    # take the aux fit entries (rf_classifier_fit_aux_export, rf_regressor_fit_aux_export).
    rfc = ml.RandomForestClassifier(n_estimators=8, max_depth=6, oob_score=True, random_state=7).fit(
        X[:2000], yc[:2000], feature_importances=True)
    rfr = ml.RandomForestRegressor(n_estimators=8, max_depth=6, oob_score=True, random_state=7).fit(
        X[:2000], yr[:2000], feature_importances=True)
    return _fit(dict(oob_c=_h(np.float64(bc.oob_score_)), oob_df=_h(bc.oob_decision_function_),
                     oob_r=_h(np.float64(br.oob_score_)), oob_p=_h(br.oob_prediction_),
                     rf_oob_df=_h(rfc.oob_decision_function_), rf_oob=_h(np.float64(rfc.oob_score_)),
                     rf_imp=_h(rfc.feature_importances_),
                     rfr_oob_p=_h(rfr.oob_prediction_), rfr_oob=_h(np.float64(rfr.oob_score_)),
                     rfr_imp=_h(rfr.feature_importances_),
                     bag=_h(bc.predict_proba(X)), stack=_h(st.predict(X)), cal=_h(ca.predict_proba(X)),
                     shap=_h(ke.shap_values(X6[:3])), ev=_h(np.asarray(ke.expected_value))),
                ca, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


_batch_decl(_rows_calls("predict", "predict_proba"), "trees-oob-cv-link")


@lane("trees-rf-weighted")
def _(ml, X, yc, yr, Xh=None):
    """The RF weighted objective (class weights without bootstrap): a balanced
    entropy forest on column samples, and a gini tree on per-row sample weights
    that zero some rows out; a regression tree on the same weights."""
    f = ml.RandomForestClassifier(n_estimators=4, max_depth=6, random_state=7, class_weight="balanced",
                                  bootstrap=False, criterion="entropy", max_features=0.6).fit(X, yc)
    w = np.array([(1.0, 2.5, 0.0, 0.75)[i % 4] for i in range(len(X))], dtype=np.float32)
    t = ml.DecisionTreeClassifier(max_depth=7, min_samples_leaf=2).fit(X, yc, sample_weight=w)
    # the weighted regression entry (rf_regressor_fit_weighted_export) on the same weights
    tr = ml.DecisionTreeRegressor(max_depth=7, min_samples_leaf=2).fit(X, yr, sample_weight=w)
    return _fit(dict(forest=_h(f.predict_proba(X)), tree=_h(t.predict_proba(X)), predict=_h(t.predict(X)),
                     tree_reg=_h(tr.predict(X))),
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


@lane("trees-gbdt-importances")
def _(ml, X, yc, yr, Xh=None):
    """GradientBoosting `feature_importances_` (lane mf-imp6: core/gbdt_importance.mojo on the host column,
    gbdt/gpu_importance.mojo on a GPU, the same bodies and fold order), light: 8 SymmetricTree trees (RMSE),
    the same on fixture-derived sample weights (the exact integer leaf weights), 8 Depthwise trees (the
    non-symmetric descent). The weighted fit uses the public regressor adapter, and a separate fit uses the
    classifier adapter. The digest covers the
    importances' bytes beside each model's predictions.

    THE CTR MODELS (lane vd-leftovers, 2026-11-24): a categorical model's importances walk its MODEL columns,
    which the device column expands on the device (gbdt/gpu_importance.mojo `gbdt_imp_expand_device`, the
    resident predict's kernels) and the host column on the host (`gbdt_importance_model_x`). Three fits reach
    the three expansions: simple CTR and one-hot columns (`_gbdt_cat_x`, every column), feature combinations
    (max_ctr_complexity 2: the tree-CTR columns), and ExperimentalTwoLevelFeatureFreq (the tensor CTR
    column). Each is a fit whose trees split on its CTR columns, so a wrong model column moves its bytes."""
    w = np.array([(1.0, 2.5, 0.5, 0.75)[i % 4] for i in range(len(X))], dtype=np.float32)
    sym = _gbdt(ml.GradientBoosting, n_estimators=8, max_depth=4).fit(X, yr)
    weighted = _gbdt(ml.GradientBoostingRegressor, n_estimators=8, max_depth=4).fit(X, yr, sample_weight=w)
    depthwise = _gbdt(ml.GradientBoosting, n_estimators=8, max_depth=4, grow_policy="Depthwise").fit(X, yr)
    clf = ml.GradientBoostingClassifier(n_estimators=8, max_depth=4).fit(X, yc)
    rows = min(_GBDT_CAT_ROWS, X.shape[0])
    Xc = _gbdt_cat_x(X[:rows])
    cat_kw = dict(n_estimators=8, max_depth=4, learning_rate=_GBDT_CAT_RATE, l2_leaf_reg=3.0, border_count=128,
                  leaf_estimation_iterations=1, boosting_type="Plain", random_state=7, loss="RMSE",
                  **_GBDT_CAT_PLAIN)
    cols, cats = _GBDT_CAT_ALL
    simple = ml.GradientBoosting(cat_features=list(cats), **cat_kw).fit(np.ascontiguousarray(Xc[:, cols]), yr[:rows])
    cols2, cats2 = _GBDT_CAT_MCC2
    comb = ml.GradientBoosting(cat_features=list(cats2), max_ctr_complexity=2, **cat_kw).fit(
        np.ascontiguousarray(Xc[:, cols2]), yr[:rows])
    Xt = _tensor_ctr_x(X)
    tensor = ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7).fit(Xt, _tensor_ctr_y(Xt))
    return _fit(dict(sym=_h(sym.feature_importances_), weighted=_h(weighted.feature_importances_),
                     depthwise=_h(depthwise.feature_importances_), clf=_h(clf.feature_importances_),
                     ctr_simple=_h(simple.feature_importances_), ctr_combination=_h(comb.feature_importances_),
                     ctr_tensor=_h(tensor.feature_importances_),
                     predict=_h(sym.predict(X))),
                sym, lambda e: (e.predict(Xh),))


_batch_decl(_rows_calls("predict"), "trees-gbdt-importances")


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
