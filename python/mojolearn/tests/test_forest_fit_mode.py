# SPDX-License-Identifier: Apache-2.0
"""Fit-mode retention with host stubs; no native code or GPU work."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from mojolearn import (ExtraTreesClassifier, ExtraTreesRegressor,
                       RandomForestClassifier, RandomForestRegressor)
from mojolearn import _backend, _metrics_impl

CLASSES = [ExtraTreesClassifier, ExtraTreesRegressor,
           RandomForestClassifier, RandomForestRegressor]


@pytest.fixture
def boundary(monkeypatch):
    default = ['identical']
    calls = []
    monkeypatch.setattr(_backend, 'default_mode', lambda: default[0])
    native = SimpleNamespace(**{name: object() for name in (
        'rf_classifier_fit', 'rf_regressor_fit', 'et_classifier_fit', 'et_regressor_fit')})
    def binding(name, mode=None):
        calls.append(mode)
        return native
    monkeypatch.setattr(_backend, 'binding', binding)
    return default, calls


def fit_stub(self, *args, **kwargs):
    self._offsets = np.array([0, 1], dtype=np.int32)
    return self


@pytest.mark.parametrize('cls', CLASSES)
def test_capture_default_refit_reset_pickle(cls, monkeypatch, boundary):
    # Sentinel entrypoints isolate mode capture; export routing has its own gate.
    monkeypatch.setenv("MOJOLEARN_FOREST_EXPORT", "legacy")
    default, calls = boundary
    monkeypatch.setattr(cls, '_fit_arrays', fit_stub)
    model = cls()
    model.fit(np.ones((2, 1)), np.array([0, 1]))
    assert calls[-1] == 'identical'
    assert model.get_params()['numeric_mode'] is None
    default[0] = 'fast'
    model._bind()
    assert calls[-1] == 'identical'
    restored = pickle.loads(pickle.dumps(model))
    restored._bind()
    assert calls[-1] == 'identical'
    model.numeric_mode = 'fast'
    with pytest.raises(ValueError, match='was fitted'):
        model._bind()
    model.fit(np.ones((2, 1)), np.array([0, 1]))
    assert calls[-1] == 'fast'
    model.set_params(numeric_mode='deterministic')
    assert not model.__sklearn_is_fitted__()
    assert not hasattr(model, '_fit_numeric_mode')


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('mode', ['', 0, False, 1, 'other'])
def test_invalid_direct_mode_refused_at_call(cls, mode, boundary):
    model = cls(numeric_mode='identical')
    model._capture_fit_mode()
    model.numeric_mode = mode
    with pytest.raises(ValueError, match='numeric_mode'):
        model._bind()
    assert not boundary[1]


@pytest.mark.parametrize('cls', CLASSES)
def test_score_captured_mode_and_equivalent_spelling(cls, monkeypatch, boundary):
    model = cls(numeric_mode=' IDENTICAL ')
    model._capture_fit_mode()
    model.numeric_mode = None
    boundary[0][0] = 'fast'
    monkeypatch.setattr(model, 'predict', lambda X: np.array([0, 1]))
    modes = []
    def metric(*args, numeric_mode):
        modes.append(numeric_mode)
        return 1.0
    monkeypatch.setattr(_metrics_impl, 'accuracy_score', metric)
    monkeypatch.setattr(_metrics_impl, 'r2_score', metric)
    assert model.score([[0], [1]], [0, 1]) == 1.0
    assert modes == ['identical']


@pytest.mark.parametrize('cls', CLASSES)
def test_legacy_archive_fallback(cls, boundary):
    model = cls.__new__(cls)
    model._bind()
    assert boundary[1][-1] == 'identical'
    boundary[0][0] = 'fast'
    model._bind()
    assert boundary[1][-1] == 'fast'
    model.numeric_mode = 'deterministic'
    model._bind()
    assert boundary[1][-1] == 'deterministic'
