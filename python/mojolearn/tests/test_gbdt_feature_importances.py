"""`feature_importances_` of the gradient-boosting estimators (lane mf-imp6) against what is known without the
product: the target's driving columns rank first, the values are non-negative and sum to 1, a constant target gives
zeros, and a constant power-of-two sample weight leaves them exactly unchanged (the integer leaf weights scale
exactly). The value is computed in Mojo at fit (core/gbdt_importance.mojo on the host column,
gbdt/gpu_importance.mojo on a GPU); its digest on both sides is the verify lane trees-gbdt-importances
(tools/identity_lanes/trees.py), and the numerical definition is checked natively
(checks/native_tests/mf_imp6_gbdt_importances.mojo). The returned object, the not-fitted error, save/load, pickle,
HostGBDT, the adapters and scikit-learn's SelectFromModel are in test_gbdt_boundary.py.
"""
import numpy as np

import mojolearn as ml


def _data(seed=11, n=400, d=5):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d), dtype=np.float32)
    y = (4.0 * X[:, 2] + 0.5 * X[:, 0] + 0.01 * rng.normal(size=n)).astype(np.float32)
    return X, y


def _check(imp, d):
    assert type(imp) is np.ndarray
    assert imp.shape == (d,)
    assert imp.dtype == np.float64
    assert np.all(imp >= 0.0)
    assert abs(float(imp.sum()) - 1.0) <= 1e-6


def test_booster_returns_normalised_float64_ndarray():
    X, y = _data()
    model = ml.GradientBoosting(n_estimators=20, max_depth=4).fit(X, y)
    imp = model.feature_importances_
    _check(imp, X.shape[1])
    # the target is driven by column 2, then column 0
    assert int(np.argmax(imp)) == 2
    assert imp[0] > imp[1] and imp[0] > imp[3] and imp[0] > imp[4]


def test_constant_target_gives_zeros():
    X, _ = _data()
    model = ml.GradientBoosting(n_estimators=4).fit(X, np.full(X.shape[0], 2.5, np.float32))
    imp = model.feature_importances_
    assert imp.shape == (X.shape[1],) and imp.dtype == np.float64
    np.testing.assert_array_equal(imp, np.zeros(X.shape[1]))


def test_constant_power_of_two_weight_is_exact():
    X, y = _data()
    plain = ml.GradientBoosting(n_estimators=6).fit(X, y)
    weighted = ml.GradientBoosting(n_estimators=6).fit(X, y, sample_weight=np.full(X.shape[0], 2.0, np.float32))
    np.testing.assert_array_equal(np.asarray(plain.predict(X)), np.asarray(weighted.predict(X)))
    np.testing.assert_array_equal(plain.feature_importances_, weighted.feature_importances_)


def test_constant_power_of_two_weight_is_exact_under_l2_score():
    # Under score_function='L2' (no noise term) every split score, leaf sum and the scaled l2 (l2 * sum(w) / n)
    # double exactly with a constant weight 2, so the trees and leaves are equal by construction. The default
    # Cosine score instead scales by sqrt(2) together with its random_strength noise (the derivative std over
    # the object count), equal up to rounding: if only the test above fails, a near-tie in the Cosine ranking
    # is the cause, not the weighting.
    X, y = _data()
    plain = ml.GradientBoosting(n_estimators=6, score_function="L2").fit(X, y)
    weighted = ml.GradientBoosting(n_estimators=6, score_function="L2").fit(
        X, y, sample_weight=np.full(X.shape[0], 2.0, np.float32))
    np.testing.assert_array_equal(np.asarray(plain.predict(X)), np.asarray(weighted.predict(X)))
    np.testing.assert_array_equal(plain.feature_importances_, weighted.feature_importances_)
