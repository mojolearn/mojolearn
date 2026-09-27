# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Trees lane (algorithm expansion) sanity vs scikit-learn on small data.

A tolerance check, not an identity claim: our trees split on per-feature
quantiles (n_bins) where sklearn's split on every midpoint, so the scores are
compared, never the predictions. Run on a GPU box:
    python -m pytest python/mojolearn/tests/test_x_trees_sanity.py -q
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.datasets import make_classification, make_regression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402

import mojolearn as ml  # noqa: E402


def _clf(n_classes=3, seed=0):
    X, y = make_classification(n_samples=1500, n_features=10, n_informative=6, n_classes=n_classes,
                               random_state=seed)
    return train_test_split(X.astype(np.float32), y, test_size=0.3, random_state=seed)


def _reg(seed=0):
    X, y = make_regression(n_samples=1500, n_features=10, n_informative=6, noise=5.0, random_state=seed)
    return train_test_split(X.astype(np.float32), y.astype(np.float32), test_size=0.3, random_state=seed)


def test_decision_tree_classifier():
    from sklearn.tree import DecisionTreeClassifier
    Xa, Xb, ya, yb = _clf()
    ours = ml.DecisionTreeClassifier(max_depth=6).fit(Xa, ya)
    ref = DecisionTreeClassifier(max_depth=6, random_state=0).fit(Xa, ya)
    a, r = accuracy_score(yb, np.asarray(ours.predict(Xb))), accuracy_score(yb, ref.predict(Xb))
    assert a >= r - 0.05, (a, r)
    assert ours.get_depth() <= 6
    p = np.asarray(ours.predict_proba(Xb))
    np.testing.assert_allclose(p.sum(1), 1.0, rtol=1e-5)
    w = np.where(ya == 0, 5.0, 1.0).astype(np.float32)
    wt = ml.DecisionTreeClassifier(max_depth=6).fit(Xa, ya, sample_weight=w)
    # upweighting class 0 must not reduce its recall
    rec = lambda m: np.mean(np.asarray(m.predict(Xb))[yb == 0] == 0)  # noqa: E731
    assert rec(wt) >= rec(ours) - 0.02


def test_decision_tree_regressor():
    from sklearn.tree import DecisionTreeRegressor
    Xa, Xb, ya, yb = _reg()
    ours = ml.DecisionTreeRegressor(max_depth=8).fit(Xa, ya)
    ref = DecisionTreeRegressor(max_depth=8, random_state=0).fit(Xa, ya)
    a, r = r2_score(yb, np.asarray(ours.predict(Xb))), r2_score(yb, ref.predict(Xb))
    assert a >= r - 0.05, (a, r)


def test_bagging_classifier():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    Xa, Xb, ya, yb = _clf()
    ours = ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=8), n_estimators=10, random_state=0).fit(Xa, ya)
    ref = BaggingClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=10, random_state=0).fit(Xa, ya)
    a, r = accuracy_score(yb, np.asarray(ours.predict(Xb))), accuracy_score(yb, ref.predict(Xb))
    assert a >= r - 0.05, (a, r)
    np.testing.assert_allclose(np.asarray(ours.predict_proba(Xb)).sum(1), 1.0, rtol=1e-6)


def test_bagging_regressor():
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
    Xa, Xb, ya, yb = _reg()
    ours = ml.BaggingRegressor(ml.DecisionTreeRegressor(max_depth=8), n_estimators=10, max_features=0.8,
                               random_state=0).fit(Xa, ya)
    ref = BaggingRegressor(DecisionTreeRegressor(max_depth=8), n_estimators=10, max_features=0.8,
                           random_state=0).fit(Xa, ya)
    a, r = r2_score(yb, np.asarray(ours.predict(Xb))), r2_score(yb, ref.predict(Xb))
    assert a >= r - 0.05, (a, r)


def test_adaboost_classifier():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    Xa, Xb, ya, yb = _clf()
    ours = ml.AdaBoostClassifier(ml.DecisionTreeClassifier(max_depth=2), n_estimators=30, random_state=0).fit(Xa, ya)
    ref = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=30, random_state=0).fit(Xa, ya)
    a, r = accuracy_score(yb, np.asarray(ours.predict(Xb))), accuracy_score(yb, ref.predict(Xb))
    assert a >= r - 0.05, (a, r)
    np.testing.assert_allclose(np.asarray(ours.predict_proba(Xb)).sum(1), 1.0, rtol=1e-9)
    Xa2, Xb2, ya2, yb2 = _clf(n_classes=2)
    ours2 = ml.AdaBoostClassifier(n_estimators=30).fit(Xa2, ya2)
    ref2 = AdaBoostClassifier(n_estimators=30).fit(Xa2, ya2)
    a, r = accuracy_score(yb2, np.asarray(ours2.predict(Xb2))), accuracy_score(yb2, ref2.predict(Xb2))
    assert a >= r - 0.05, (a, r)


def test_adaboost_regressor():
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    Xa, Xb, ya, yb = _reg()
    ours = ml.AdaBoostRegressor(ml.DecisionTreeRegressor(max_depth=4), n_estimators=30, random_state=0).fit(Xa, ya)
    ref = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=30, random_state=0).fit(Xa, ya)
    a, r = r2_score(yb, np.asarray(ours.predict(Xb))), r2_score(yb, ref.predict(Xb))
    assert a >= r - 0.05, (a, r)


def test_dart():
    lightgbm = pytest.importorskip("lightgbm")
    Xa, Xb, ya, yb = _reg()
    ours = ml.DARTRegressor(n_estimators=60, learning_rate=0.1, random_state=0).fit(Xa, ya)
    ref = lightgbm.LGBMRegressor(boosting_type="dart", n_estimators=60, learning_rate=0.1, verbose=-1).fit(Xa, ya)
    a, r = r2_score(yb, np.asarray(ours.predict(Xb))), r2_score(yb, ref.predict(Xb))
    assert a >= r - 0.08, (a, r)
    Xa, Xb, ya, yb = _clf(n_classes=2)
    ours = ml.DARTClassifier(n_estimators=60, random_state=0).fit(Xa, ya)
    ref = lightgbm.LGBMClassifier(boosting_type="dart", n_estimators=60, verbose=-1).fit(Xa, ya)
    a, r = accuracy_score(yb, np.asarray(ours.predict(Xb))), accuracy_score(yb, ref.predict(Xb))
    assert a >= r - 0.05, (a, r)


def test_random_trees_embedding():
    from sklearn.linear_model import LogisticRegression
    Xa, Xb, ya, yb = _clf()
    emb = ml.RandomTreesEmbedding(n_estimators=30, max_depth=5, random_state=0).fit(Xa)
    Ea, Eb = np.asarray(emb.transform(Xa)), np.asarray(emb.transform(Xb))
    assert Ea.shape[1] == emb.n_output_features_
    np.testing.assert_array_equal(Ea.sum(1), 30)          # one leaf per tree
    assert set(np.unique(Ea)) <= {0.0, 1.0}
    acc = accuracy_score(yb, LogisticRegression(max_iter=500).fit(Ea, ya).predict(Eb))
    assert acc > 0.6, acc                                   # the embedding carries signal


def test_voting():
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    Xa, Xb, ya, yb = _clf()
    ours = ml.VotingClassifier([("a", ml.DecisionTreeClassifier(max_depth=4)),
                                ("b", ml.DecisionTreeClassifier(max_depth=8))], voting="soft").fit(Xa, ya)
    ref = VotingClassifier([("a", DecisionTreeClassifier(max_depth=4, random_state=0)),
                            ("b", DecisionTreeClassifier(max_depth=8, random_state=0))], voting="soft").fit(Xa, ya)
    a, r = accuracy_score(yb, np.asarray(ours.predict(Xb))), accuracy_score(yb, ref.predict(Xb))
    assert a >= r - 0.05, (a, r)
    hard = ml.VotingClassifier([("a", ml.DecisionTreeClassifier(max_depth=4)),
                                ("b", ml.DecisionTreeClassifier(max_depth=8)),
                                ("c", ml.DecisionTreeClassifier(max_depth=6))]).fit(Xa, ya)
    assert accuracy_score(yb, np.asarray(hard.predict(Xb))) >= r - 0.08
    Xa, Xb, ya, yb = _reg()
    ours = ml.VotingRegressor([("a", ml.DecisionTreeRegressor(max_depth=4)),
                               ("b", ml.DecisionTreeRegressor(max_depth=8))], weights=[1, 2]).fit(Xa, ya)
    ref = VotingRegressor([("a", DecisionTreeRegressor(max_depth=4, random_state=0)),
                           ("b", DecisionTreeRegressor(max_depth=8, random_state=0))], weights=[1, 2]).fit(Xa, ya)
    a, r = r2_score(yb, np.asarray(ours.predict(Xb))), r2_score(yb, ref.predict(Xb))
    assert a >= r - 0.05, (a, r)


def test_stacking_folds_match_sklearn():
    from sklearn.model_selection import StratifiedKFold, KFold
    from mojolearn._expansion_trees import _trees_stratified_folds, _trees_kfolds
    y = np.random.RandomState(0).randint(0, 4, size=103)
    ours = _trees_stratified_folds(y.tolist(), 5)
    for i, (_, te) in enumerate(StratifiedKFold(5).split(np.zeros((103, 1)), y)):
        assert sorted(te.tolist()) == [r for r, f in enumerate(ours) if f == i]
    ours = _trees_kfolds(103, 4)
    for i, (_, te) in enumerate(KFold(4).split(np.zeros((103, 1)))):
        assert te.tolist() == [r for r, f in enumerate(ours) if f == i]


def test_stacking():
    from sklearn.ensemble import StackingClassifier, StackingRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    Xa, Xb, ya, yb = _clf()
    ours = ml.StackingClassifier([("a", ml.DecisionTreeClassifier(max_depth=4)),
                                  ("b", ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=8), random_state=0))],
                                 final_estimator=ml.DecisionTreeClassifier(max_depth=4)).fit(Xa, ya)
    ref = StackingClassifier([("a", DecisionTreeClassifier(max_depth=4, random_state=0)),
                              ("b", DecisionTreeClassifier(max_depth=8, random_state=0))],
                             final_estimator=DecisionTreeClassifier(max_depth=4, random_state=0)).fit(Xa, ya)
    a, r = accuracy_score(yb, np.asarray(ours.predict(Xb))), accuracy_score(yb, ref.predict(Xb))
    assert a >= r - 0.06, (a, r)
    Xa, Xb, ya, yb = _reg()
    ours = ml.StackingRegressor([("a", ml.DecisionTreeRegressor(max_depth=4)),
                                 ("b", ml.DecisionTreeRegressor(max_depth=8))],
                                final_estimator=ml.DecisionTreeRegressor(max_depth=4)).fit(Xa, ya)
    ref = StackingRegressor([("a", DecisionTreeRegressor(max_depth=4, random_state=0)),
                             ("b", DecisionTreeRegressor(max_depth=8, random_state=0))],
                            final_estimator=DecisionTreeRegressor(max_depth=4, random_state=0)).fit(Xa, ya)
    a, r = r2_score(yb, np.asarray(ours.predict(Xb))), r2_score(yb, ref.predict(Xb))
    assert a >= r - 0.08, (a, r)


def test_multioutput():
    Xa, Xb, ya, yb = _reg()
    Y = np.stack([ya, -2 * ya], axis=1)
    m = ml.MultiOutputRegressor(ml.DecisionTreeRegressor(max_depth=8)).fit(Xa, Y)
    P = np.asarray(m.predict(Xb))
    assert P.shape == (len(Xb), 2)
    np.testing.assert_allclose(P[:, 1], -2 * P[:, 0], rtol=1e-4, atol=1e-3)
    Xa, Xb, ya, yb = _clf()
    c = ml.MultiOutputClassifier(ml.DecisionTreeClassifier(max_depth=6)).fit(Xa, np.stack([ya, ya % 2], 1))
    Pc = np.asarray(c.predict(Xb))
    assert Pc.shape == (len(Xb), 2) and accuracy_score(yb, Pc[:, 0]) > 0.5


def test_onevsrest():
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.tree import DecisionTreeClassifier
    Xa, Xb, ya, yb = _clf()
    ours = ml.OneVsRestClassifier(ml.DecisionTreeClassifier(max_depth=6)).fit(Xa, ya)
    ref = OneVsRestClassifier(DecisionTreeClassifier(max_depth=6, random_state=0)).fit(Xa, ya)
    a, r = accuracy_score(yb, np.asarray(ours.predict(Xb))), accuracy_score(yb, ref.predict(Xb))
    assert a >= r - 0.05, (a, r)
    np.testing.assert_allclose(np.asarray(ours.predict_proba(Xb)).sum(1), 1.0, rtol=1e-9)


def test_calibration():
    from sklearn.calibration import CalibratedClassifierCV, _sigmoid_calibration
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import log_loss
    from sklearn.tree import DecisionTreeClassifier
    rs = np.random.RandomState(0)
    f = rs.normal(size=400)
    y = (f + rs.normal(size=400) > 0).astype(np.int32)
    b = ml.CalibratedClassifierCV()._bind()
    ab = np.zeros(2)
    b.x_trees_platt_fit(f.ctypes.data, y.ctypes.data, ab.ctypes.data, [400])
    a_ref, b_ref = _sigmoid_calibration(f, y)
    np.testing.assert_allclose(ab, [a_ref, b_ref], rtol=1e-3, atol=1e-4)
    kx, ky = np.zeros(400), np.zeros(400)
    y64 = y.astype(np.float64)
    m = int(b.x_trees_isotonic_fit(f.ctypes.data, y64.ctypes.data, kx.ctypes.data, ky.ctypes.data, [400]))
    t = np.linspace(-4, 4, 97)
    out = np.zeros(97)
    b.x_trees_isotonic_predict(kx.ctypes.data, ky.ctypes.data, t.ctypes.data, out.ctypes.data, [m, 97])
    ref = IsotonicRegression(out_of_bounds="clip").fit(f, y64).predict(t)
    np.testing.assert_allclose(out, ref, atol=1e-12)
    Xa, Xb, ya, yb = _clf()
    for method in ("sigmoid", "isotonic"):
        ours = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=6), method=method).fit(Xa, ya)
        ref = CalibratedClassifierCV(DecisionTreeClassifier(max_depth=6, random_state=0), method=method).fit(Xa, ya)
        lo, lr = log_loss(yb, np.asarray(ours.predict_proba(Xb))), log_loss(yb, ref.predict_proba(Xb))
        assert lo <= lr * 1.15 + 0.02, (method, lo, lr)


def test_tree_explainer_matches_shap_recursion():
    shap = pytest.importorskip("shap")
    Xa, Xb, ya, yb = _clf()
    m = ml.RandomForestClassifier(n_estimators=5, max_depth=5, random_state=0).fit(Xa, ya)
    ex = ml.TreeExplainer(m, data=Xa)   # every leaf holds a training row, so every node is covered
    phi = np.asarray(ex.shap_values(Xb[:40]))
    # additivity: sum of values + expected value == the forest's probability
    np.testing.assert_allclose(phi.sum(1) + np.asarray(ex.expected_value), np.asarray(m.predict_proba(Xb[:40])),
                               atol=2e-6)
    # the same recursion in the shap package, on our trees and our cover
    arrays, k, scale, cover = ex._parts[0]
    off, col, q, left, leaves = (np.asarray(a) for a in arrays)
    cover = np.asarray(cover)
    trees = []
    for t in range(len(off) - 1):
        lo, hi = off[t], off[t + 1]
        lc = left[lo:hi].astype(np.int64)
        trees.append(dict(children_left=lc, children_right=np.where(lc == -1, -1, lc + 1),
                          children_default=lc, features=np.where(lc == -1, -2, col[lo:hi]).astype(np.int64),
                          thresholds=q[lo:hi].astype(np.float64),
                          values=leaves[lo * k:hi * k].reshape(hi - lo, k).astype(np.float64) * scale,
                          node_sample_weight=cover[lo:hi]))
    ref = shap.TreeExplainer(dict(trees=trees, base_offset=0), feature_perturbation="tree_path_dependent")
    rv = np.asarray(ref.shap_values(Xb[:40].astype(np.float64), check_additivity=False))
    if rv.shape != phi.shape:
        rv = np.moveaxis(rv, 0, -1)
    np.testing.assert_allclose(phi, rv, rtol=1e-9, atol=1e-12)
    Xr, Xrb, yra, yrb = _reg()
    d = ml.DARTRegressor(n_estimators=20, random_state=0).fit(Xr, yra)
    e = ml.TreeExplainer(d, data=Xr[:200])
    p = np.asarray(e.shap_values(Xrb[:30]))
    np.testing.assert_allclose(p.sum(1) + e.expected_value, np.asarray(d.predict(Xrb[:30])), rtol=1e-5, atol=1e-3)


def test_kernel_explainer():
    shap = pytest.importorskip("shap")
    Xa, Xb, ya, yb = _reg()
    Xa, Xb = Xa[:, :6].copy(), Xb[:, :6].copy()
    m = ml.DecisionTreeRegressor(max_depth=5).fit(Xa, ya)
    bg = Xa[:20]
    ke = ml.KernelExplainer(m, bg)
    phi = np.asarray(ke.shap_values(Xb[:5]))
    f = lambda X: np.asarray(m.predict(np.asarray(X, dtype=np.float32)), dtype=np.float64)  # noqa: E731
    ref = shap.KernelExplainer(f, bg.astype(np.float64)).shap_values(Xb[:5].astype(np.float64), silent=True)
    np.testing.assert_allclose(phi, np.asarray(ref), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(phi.sum(1) + ke.expected_value, f(Xb[:5]), rtol=1e-6, atol=1e-5)
    Xc, Xcb, yc, ycb = _clf()
    c = ml.RandomForestClassifier(n_estimators=4, max_depth=4, random_state=0).fit(Xc[:, :5], yc)
    kc = ml.KernelExplainer(c, Xc[:15, :5])
    pc = np.asarray(kc.shap_values(Xcb[:3, :5]))
    assert pc.shape == (3, 5, 3)
    np.testing.assert_allclose(pc.sum(1) + np.asarray(kc.expected_value), np.asarray(c.predict_proba(Xcb[:3, :5])),
                               atol=1e-5)


def test_permutation_explainer():
    Xa, Xb, ya, yb = _reg()
    Xa, Xb = Xa[:, :6].copy(), Xb[:, :6].copy()
    m = ml.DecisionTreeRegressor(max_depth=5).fit(Xa, ya)
    bg = Xa[:20]
    f = lambda X: np.asarray(m.predict(np.asarray(X, dtype=np.float32)), dtype=np.float64)  # noqa: E731
    exact = np.asarray(ml.KernelExplainer(m, bg).shap_values(Xb[:5]))   # full enumeration: exact SHAP
    pe = ml.PermutationExplainer(m, bg, random_state=0)
    pp = np.asarray(pe.shap_values(Xb[:5], npermutations=300))
    np.testing.assert_allclose(pp.sum(1) + pe.expected_value, f(Xb[:5]), rtol=1e-6, atol=1e-5)
    assert np.abs(pp - exact).max() <= 0.15 * np.abs(exact).max() + 1e-6
