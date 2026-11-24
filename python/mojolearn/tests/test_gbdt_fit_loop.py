"""`fit_loop_` of the gradient-boosting estimators (lane ab-loop-mode): the public boundary.

The loop is timed in Mojo (core/fit_loop_clock.mojo: two clock reads around the boosting loop, on the host column
and on a GPU) and the attribute is set by the fit's native PACK. Checked here: it exists after fit as a tuple
(int, float), its rounds equal `n_estimators` when no eval set can stop the fit, it raises AttributeError before fit
and on a loaded model (the archive does not carry it), and the scikit-learn style adapters copy it. The rounds per
boosting loop are checked natively (checks/native_tests/gbdt_public_fits.mojo `fit_loop_rounds`).
"""
import numpy as np
import pytest

import mojolearn as ml


def _data(seed=5, n=300, d=4):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d), dtype=np.float32)
    y = (2.0 * X[:, 0] + X[:, 1]).astype(np.float32)
    return X, y


def _check(loop, rounds):
    assert type(loop) is tuple and len(loop) == 2
    assert type(loop[0]) is int and type(loop[1]) is float
    assert loop[0] == rounds
    assert loop[1] >= 0.0


@pytest.mark.parametrize("grow_policy", ["SymmetricTree", "Depthwise"])
def test_fit_loop_after_fit(grow_policy):
    X, y = _data()
    model = ml.GradientBoosting(n_estimators=6, max_depth=3, grow_policy=grow_policy).fit(X, y)
    _check(model.fit_loop_, 6)


def test_fit_loop_raises_before_fit():
    with pytest.raises(AttributeError):
        ml.GradientBoosting(n_estimators=4).fit_loop_


def test_fit_loop_not_carried_by_the_archive(tmp_path):
    X, y = _data()
    model = ml.GradientBoosting(n_estimators=4).fit(X, y)
    path = tmp_path / "model.npz"
    model.save(str(path))
    loaded = ml.GradientBoosting.load(str(path))
    with pytest.raises(AttributeError):
        loaded.fit_loop_


def test_adapters_copy_fit_loop():
    X, y = _data()
    _check(ml.GradientBoostingRegressor(n_estimators=5).fit(X, y).fit_loop_, 5)
    labels = (y > np.median(y)).astype(np.int64)
    _check(ml.GradientBoostingClassifier(n_estimators=5).fit(X, labels).fit_loop_, 5)
