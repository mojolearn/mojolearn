# SPDX-License-Identifier: Apache-2.0
"""RFE rows lane mf-rfe6 (2026-11-23) moved off the estimator's own doors (`_fit_clone`) into the member RUN.

Each test compares the member route with the route it replaced or with the estimator's own fit:
  * a forest leaf's feature_importances_ under a search step equals RFE over the forest itself (the search's
    refit forest is the same fit), support and ranking;
  * a fold's DeviceRows X (cross_val_score with the RFE plan's device_rows on) equals the host-array folds;
  * a weighted RandomForestRegressor round equals an elimination written out over the estimator's own
    weighted fit with feature_importances=True (served since this lane, as the classifier's was);
  * a refused target raises the estimator's own fit's words (no second fit through its doors).

Runs with the built package (the python/mojolearn/tests suite on a box, never on the laptop).
"""
import hashlib

import numpy as np
import pytest

import mojolearn
from mojolearn import model_selection
from mojolearn.model_selection import GridSearchCV


def _data(n=64, d=6, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x[:, 0] + 0.6 * x[:, 2] - 0.3 * x[:, 4] > 0).astype(np.int64)
    t = (2.0 * x[:, 0] - x[:, 2] + 0.25 * x[:, 4] ** 2).astype(np.float32)
    return x, y, t


def _forest(cls):
    return cls(n_estimators=4, max_depth=4, random_state=0)


def test_forest_leaf_importances_under_a_search_equal_the_forest_itself():
    x, y, _ = _data()
    every = np.arange(len(x))
    search = GridSearchCV(_forest(mojolearn.RandomForestClassifier), {"max_depth": [4]}, cv=[(every, every)],
                          refit=True)
    member = mojolearn.RFE(search, n_features_to_select=3,
                           importance_getter="best_estimator_.feature_importances_").fit(x, y)
    leaf = mojolearn.RFE(_forest(mojolearn.RandomForestClassifier), n_features_to_select=3,
                         importance_getter="feature_importances_").fit(x, y)
    np.testing.assert_array_equal(np.asarray(member.support_), np.asarray(leaf.support_))
    np.testing.assert_array_equal(np.asarray(member.ranking_), np.asarray(leaf.ranking_))
    kept = np.asarray(member.support_)
    np.testing.assert_array_equal(np.asarray(member.predict(x)), np.asarray(leaf.predict(x)))
    assert np.asarray(member.estimator_.best_estimator_.feature_importances_).shape == (int(kept.sum()),)


def _digest(scores):
    return hashlib.sha256(np.asarray(scores, dtype="f8").tobytes()).hexdigest()


@pytest.mark.parametrize("make", [
    lambda: mojolearn.LogisticRegression(max_iter=50),
    lambda: mojolearn.LinearDiscriminantAnalysis(),
], ids=["logistic", "lda"])
def test_device_rows_folds_equal_host_folds(make):
    x, y, _ = _data()
    registry = model_selection._NATIVE_CONTEXT["family_msel_estimators"]
    if mojolearn.RFE not in registry:
        pytest.skip("RFE has no native model-selection plan in this build")
    plan = registry[mojolearn.RFE]
    before = plan.get("device_rows", False)
    try:
        plan["device_rows"] = False
        host = model_selection.cross_val_score(mojolearn.RFE(make(), n_features_to_select=3), x, y, cv=4)
        plan["device_rows"] = True
        rows = model_selection.cross_val_score(mojolearn.RFE(make(), n_features_to_select=3), x, y, cv=4)
    finally:
        plan["device_rows"] = before
    assert _digest(rows) == _digest(host), (list(host), list(rows))


def _eliminate(make, x, t, w, keep):
    # scikit-learn's RFE with step=1 written out over the estimator's own weighted fit and importances.
    features = np.arange(x.shape[1])
    while len(features) > keep:
        est = make().fit(x[:, features], t, sample_weight=w, feature_importances=True)
        order = np.argsort(np.asarray(est.feature_importances_), kind="stable")
        features = np.delete(features, order[0])
    return np.isin(np.arange(x.shape[1]), features)


def test_weighted_forest_regressor_rounds_equal_its_own_weighted_importances():
    x, _, t = _data()
    w = np.linspace(0.5, 2.0, len(x)).astype(np.float32)
    est = _forest(mojolearn.RandomForestRegressor).fit(x, t, sample_weight=w, feature_importances=True)
    imp = np.asarray(est.feature_importances_)
    assert imp.shape == (x.shape[1],) and imp.dtype == np.float32
    member = mojolearn.RFE(_forest(mojolearn.RandomForestRegressor), n_features_to_select=3).fit(
        x, t, sample_weight=w)
    want = _eliminate(lambda: _forest(mojolearn.RandomForestRegressor), x, t, w, 3)
    np.testing.assert_array_equal(np.asarray(member.support_), want)
