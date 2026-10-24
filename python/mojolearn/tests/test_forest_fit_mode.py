# SPDX-License-Identifier: Apache-2.0
"""Fit-mode retention through the real public forest methods (small inputs)."""
import pickle

import numpy as np
import pytest
from mojolearn import (ExtraTreesClassifier, ExtraTreesRegressor,
                       RandomForestClassifier, RandomForestRegressor)
from mojolearn import _backend

CLASSES = [ExtraTreesClassifier, ExtraTreesRegressor,
           RandomForestClassifier, RandomForestRegressor]


def _data(cls):
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    if cls._estimator_type == 'classifier':
        return X, (X[:, 0] > 0.5).astype(np.int32)
    return X, (2.0 * X[:, 0] + X[:, 1]).astype(np.float32)


def _model(cls, **kw):
    return cls(n_estimators=2, max_depth=3, random_state=0, **kw)


def _other(mode):
    return 'fast' if mode == 'identical' else 'identical'


@pytest.mark.parametrize('cls', CLASSES)
def test_capture_default_refit_reset_pickle(cls):
    X, y = _data(cls)
    default = _backend.default_mode()
    model = _model(cls).fit(X, y)
    assert model._fit_numeric_mode == default
    assert model.get_params()['numeric_mode'] is None
    first = np.asarray(model.predict(X))
    restored = pickle.loads(pickle.dumps(model))
    assert restored._fit_numeric_mode == default
    np.testing.assert_array_equal(np.asarray(restored.predict(X)), first)
    model.numeric_mode = _other(default)
    with pytest.raises(ValueError, match='was fitted'):
        model.predict(X)
    if _backend._CPU_ONLY is not None and _other(default) != 'identical':
        # CPU-only install: host bindings are identical only and the other
        # tier is refused by name (_backend._cpu_only_binding).
        with pytest.raises(ValueError, match="tier on a CPU-only install"):
            model.fit(X, y)
    else:
        try:
            model.fit(X, y)
        except ImportError as error:
            pytest.skip(f"this install has no {_other(default)} tree binding: {error}")
        assert model._fit_numeric_mode == _other(default)
    model.set_params(numeric_mode='identical')
    assert not model.__sklearn_is_fitted__()
    assert not hasattr(model, '_fit_numeric_mode')


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('mode', ['', 0, False, 1, 'other'])
def test_invalid_direct_mode_refused_at_call(cls, mode):
    X, y = _data(cls)
    model = _model(cls, numeric_mode='identical').fit(X, y)
    model.numeric_mode = mode
    with pytest.raises(ValueError, match='numeric_mode'):
        model.predict(X)
    with pytest.raises(ValueError, match='numeric_mode'):
        _model(cls, numeric_mode=mode).fit(X, y)


@pytest.mark.parametrize('cls', CLASSES)
def test_score_captured_mode_and_equivalent_spelling(cls):
    from sklearn.metrics import accuracy_score, r2_score
    X, y = _data(cls)
    model = _model(cls, numeric_mode=' IDENTICAL ').fit(X, y)
    assert model._fit_numeric_mode == 'identical'
    model.numeric_mode = None
    pred = np.asarray(model.predict(X))
    expected = accuracy_score(y, pred) if cls._estimator_type == 'classifier' else r2_score(y, pred)
    assert model.score(X, y) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert model._fit_numeric_mode == 'identical'


@pytest.mark.parametrize('cls', CLASSES)
def test_legacy_archive_fallback(cls, tmp_path):
    """A sequential archive records no numeric mode: the loaded model
    predicts under its stored `numeric_mode`, else the process default."""
    X, y = _data(cls)
    model = _model(cls, numeric_mode='identical', inference_engine='sequential').fit(X, y)
    path = tmp_path / 'model.npz'
    model.save(path)
    loaded = cls.load(path)
    assert '_fit_numeric_mode' not in loaded.__dict__
    expected = np.asarray(model.predict(X))
    loaded.numeric_mode = 'identical'
    np.testing.assert_array_equal(np.asarray(loaded.predict(X)), expected)
    loaded.numeric_mode = None
    if _backend.default_mode() == 'identical':
        np.testing.assert_array_equal(np.asarray(loaded.predict(X)), expected)
    else:
        assert np.asarray(loaded.predict(X)).shape == expected.shape
