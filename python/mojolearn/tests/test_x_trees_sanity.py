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
