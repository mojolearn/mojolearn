"""RFE.fit(X, y, **fit_params) over a GradientBoosting member (lane mf-gbdt5): the fit parameters are admitted
natively with the estimator's own fit door's checks and words, and RFE's member fits read them, the held-out X
restricted to the fit's columns. With n_features_to_select keeping every column RFE runs no round (scikit-learn
reads no importances then), so its final estimator is the estimator's own fit on X: same predictions, same
curves. A round over a family with no importance block (GradientBoosting, GaussianNB) refuses before it fits.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn._expansion_prep import RFE


def _data(seed=3, n=160, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X @ np.arange(1, d + 1, dtype=np.float32) + 0.1 * rng.normal(size=n)).astype(np.float32)
    return X, y


def _same_fit(rfe, own, X):
    est = rfe.estimator_
    np.testing.assert_array_equal(np.asarray(est.predict(X)), np.asarray(own.predict(X)))
    np.testing.assert_array_equal(np.asarray(est.loss_curve_), np.asarray(own.loss_curve_))
    if own.test_loss_curve_ is None:
        assert est.test_loss_curve_ is None
    else:
        np.testing.assert_array_equal(np.asarray(est.test_loss_curve_), np.asarray(own.test_loss_curve_))
    assert est.best_iteration_ == own.best_iteration_
    assert list(rfe.support_) == [True] * X.shape[1]


def test_booster_eval_set_equals_its_own_fit():
    X, y = _data()
    kw = dict(n_estimators=12, use_best_model=True)
    eval_set = (X[120:], y[120:])
    rfe = RFE(ml.GradientBoosting(**kw), n_features_to_select=X.shape[1]).fit(X[:120], y[:120], eval_set=eval_set)
    own = ml.GradientBoosting(**kw).fit(X[:120], y[:120], eval_set=eval_set)
    _same_fit(rfe, own, X)
    assert rfe.estimator_.test_loss_curve_ is not None


def test_booster_sample_weight_equals_its_own_fit():
    X, y = _data(5)
    w = np.linspace(0.5, 2.0, X.shape[0]).astype(np.float32)
    rfe = RFE(ml.GradientBoosting(n_estimators=10), n_features_to_select=X.shape[1]).fit(X, y, sample_weight=w)
    own = ml.GradientBoosting(n_estimators=10).fit(X, y, sample_weight=w)
    _same_fit(rfe, own, X)


def test_booster_pairs_and_groups_equal_its_own_fit():
    X, y = _data(7)
    groups = np.repeat(np.arange(16), 10)
    pairs = np.asarray([[g * 10, g * 10 + 1] for g in range(16)], dtype=np.int64)
    kw = dict(loss="PairLogit", n_estimators=8)
    rfe = RFE(ml.GradientBoosting(**kw), n_features_to_select=X.shape[1]).fit(X, y, group_id=groups, pairs=pairs)
    own = ml.GradientBoosting(**kw).fit(X, y, group_id=groups, pairs=pairs)
    _same_fit(rfe, own, X)


def test_regressor_adapter_eval_set_equals_its_own_fit():
    X, y = _data(11)
    eval_set = [(X[120:], y[120:])]
    rfe = RFE(ml.GradientBoostingRegressor(n_estimators=12), n_features_to_select=X.shape[1]).fit(
        X[:120], y[:120], eval_set=eval_set)
    own = ml.GradientBoostingRegressor(n_estimators=12).fit(X[:120], y[:120], eval_set=eval_set)
    _same_fit(rfe, own, X)


def test_classifier_adapter_eval_set_equals_its_own_fit():
    X, y = _data(13)
    labels = np.where(y > np.median(y), "hi", "lo").astype(object)
    eval_set = (X[120:], labels[120:])
    rfe = RFE(ml.GradientBoostingClassifier(n_estimators=12), n_features_to_select=X.shape[1]).fit(
        X[:120], labels[:120], eval_set=eval_set)
    own = ml.GradientBoostingClassifier(n_estimators=12).fit(X[:120], labels[:120], eval_set=eval_set)
    _same_fit(rfe, own, X)
    np.testing.assert_array_equal(np.asarray(rfe.predict(X)), np.asarray(own.predict(X)))


def _words(fit):
    with pytest.raises(Exception) as caught:
        fit()
    return f"{type(caught.value).__name__}: {caught.value}"


@pytest.mark.parametrize("params", [
    {"eval_set": [(np.zeros((4, 5), np.float32), np.zeros(4, np.float32))] * 2},
    {"eval_set": (np.zeros((4, 4), np.float32), np.zeros(4, np.float32))},
    {"eval_set": (np.zeros((4, 5), np.float32), np.zeros(3, np.float32))},
    {"pairs_weight": [1.0]},
    {"subgroup_id": np.zeros(160)},
    {"pairs": [[0, 1]], "group_id": np.zeros(160)},
])
def test_booster_refusals_are_the_fit_doors_words(params):
    X, y = _data()
    ours = _words(lambda: RFE(ml.GradientBoosting(n_estimators=4), n_features_to_select=5).fit(X, y, **params))
    own = _words(lambda: ml.GradientBoosting(n_estimators=4).fit(X, y, **params))
    assert ours == own


def test_a_round_over_gbdt_refuses_before_it_fits():
    X, y = _data()
    with pytest.raises(Exception, match="RFE needs an estimator with coef_ or feature_importances_"):
        RFE(ml.GradientBoosting(n_estimators=4), n_features_to_select=2).fit(X, y, eval_set=(X[:20], y[:20]))
    with pytest.raises(AttributeError, match="'GradientBoosting' object has no attribute 'coef_'"):
        RFE(ml.GradientBoosting(n_estimators=4), n_features_to_select=2, importance_getter="coef_").fit(X, y)


def test_no_round_needs_no_importances():
    X, y = _data()
    codes = (y > np.median(y)).astype(np.int64)
    rfe = RFE(ml.GaussianNB(), n_features_to_select=X.shape[1]).fit(X, codes)
    assert list(rfe.support_) == [True] * X.shape[1]
    np.testing.assert_array_equal(np.asarray(rfe.predict(X)), np.asarray(ml.GaussianNB().fit(X, codes).predict(X)))


def test_an_unknown_key_keeps_the_fit_door():
    X, y = _data()
    ours = _words(lambda: RFE(ml.GradientBoosting(n_estimators=4), n_features_to_select=5).fit(X, y, bogus=1))
    own = _words(lambda: ml.GradientBoosting(n_estimators=4).fit(X, y, bogus=1))
    assert ours == own
