"""`feature_importances_` of the gradient-boosting estimators (lane mf-imp6): the public boundary.

The value is computed in Mojo at fit (core/gbdt_importance.mojo on the host column, gbdt/gpu_importance.mojo on a
GPU) and read through ONE native call. Checked here: the returned object (a NumPy ndarray), its shape
(n_features_in_,), dtype float64 (scikit-learn's), non-negative and summing to 1, a fresh array per read, the
not-fitted error in scikit-learn's words, the same values after save/load, pickle and in HostGBDT, the constant
target's zeros, and exact invariance under a constant power-of-two sample weight (the integer leaf weights scale
exactly). The numerical definition itself is checked natively (checks/native_tests/mf_imp6_gbdt_importances.mojo).
"""
import pickle

import numpy as np
import pytest

import mojolearn as ml
from mojolearn.exceptions import NotFittedError


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


def test_each_read_is_a_fresh_array():
    X, y = _data()
    model = ml.GradientBoosting(n_estimators=8).fit(X, y)
    first = model.feature_importances_
    kept = first.copy()
    first[:] = -1.0
    np.testing.assert_array_equal(model.feature_importances_, kept)


@pytest.mark.parametrize("make", [
    lambda: ml.GradientBoosting(n_estimators=4),
    lambda: ml.GradientBoostingRegressor(n_estimators=4),
    lambda: ml.GradientBoostingClassifier(n_estimators=4),
])
def test_not_fitted_is_sklearns_error(make):
    est = make()
    with pytest.raises(NotFittedError, match="instance is not fitted yet. Call 'fit' with appropriate arguments"):
        est.feature_importances_


def test_adapters_expose_the_learners_importances():
    X, y = _data()
    reg = ml.GradientBoostingRegressor(n_estimators=10).fit(X, y)
    own = ml.GradientBoosting(n_estimators=10).fit(X, y)
    _check(reg.feature_importances_, X.shape[1])
    np.testing.assert_array_equal(reg.feature_importances_, own.feature_importances_)
    codes = (y > np.median(y)).astype(np.int64)
    clf = ml.GradientBoostingClassifier(n_estimators=10).fit(X, codes)
    _check(clf.feature_importances_, X.shape[1])


def test_multiclass_sums_its_dimensions():
    X, y = _data()
    codes = np.digitize(y, np.quantile(y, [0.33, 0.66])).astype(np.float32)
    model = ml.GradientBoosting(loss="MultiClass", n_estimators=10).fit(X, codes)
    _check(model.feature_importances_, X.shape[1])


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


def test_save_load_pickle_and_host_twin(tmp_path):
    X, y = _data()
    model = ml.GradientBoosting(n_estimators=8).fit(X, y)
    path = tmp_path / "gb.npz"
    model.save(path)
    np.testing.assert_array_equal(ml.GradientBoosting.load(path).feature_importances_, model.feature_importances_)
    np.testing.assert_array_equal(ml.HostGBDT.from_file(path).feature_importances_, model.feature_importances_)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).feature_importances_, model.feature_importances_)


def test_set_params_clears_them():
    X, y = _data()
    model = ml.GradientBoosting(n_estimators=4).fit(X, y)
    model.set_params(n_estimators=5)
    with pytest.raises(NotFittedError):
        model.feature_importances_


def test_sklearn_select_from_model_reads_them():
    feature_selection = pytest.importorskip("sklearn.feature_selection")
    X, y = _data()
    sel = feature_selection.SelectFromModel(ml.GradientBoostingRegressor(n_estimators=10), max_features=1,
                                            threshold=-np.inf).fit(X, y)
    assert list(np.flatnonzero(sel.get_support())) == [2]
