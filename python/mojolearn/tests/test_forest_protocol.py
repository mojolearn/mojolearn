# SPDX-License-Identifier: Apache-2.0
"""Real sklearn orchestration; native training/scoring mocked in CPU unit checks."""
import inspect
import pickle

import numpy as np
import pytest

sklearn = pytest.importorskip('sklearn')
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from mojolearn import (ExtraTreesClassifier, ExtraTreesRegressor,
                       RandomForestClassifier, RandomForestRegressor)
from mojolearn import _metrics_impl

CLASSES = [ExtraTreesClassifier, ExtraTreesRegressor,
           RandomForestClassifier, RandomForestRegressor]


@pytest.mark.parametrize('cls', CLASSES)
def test_clone_raw_parameters_mode_and_fitted_state(cls):
    original_count = np.int64(3)
    model = cls(n_estimators=original_count, max_depth=None, numeric_mode='identical')
    assert model.get_params()['n_estimators'] is original_count
    assert 'numeric_mode' in inspect.signature(cls).parameters
    assert model.max_depth is None
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    model._offsets = np.array([0, 1])
    model.classes_ = np.array(['a', 'b'])
    check_is_fitted(model)
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    assert not hasattr(copied, '_offsets') and not hasattr(copied, 'classes_')
    assert copied.numeric_mode == 'identical'
    assert is_classifier(copied) == ('Classifier' in cls.__name__)
    assert is_regressor(copied) == ('Regressor' in cls.__name__)
    assert pickle.loads(pickle.dumps(copied)).get_params() == copied.get_params()


@pytest.mark.parametrize('cls', CLASSES)
def test_set_params_refreshes_native_config_atomically(cls):
    model = cls(max_depth=3, numeric_mode='fast')
    model._offsets = np.array([0, 1])
    with pytest.raises(ValueError):
        model.set_params(unknown=1)
    with pytest.raises(ValueError):
        model.set_params(device='invalid')
    with pytest.raises(ValueError, match='numeric_mode'):
        model.set_params(numeric_mode='typo')
    assert model._cfg['max_depth'] == 3 and model.__sklearn_is_fitted__()
    assert model.set_params(max_depth=2, numeric_mode='identical') is model
    assert model._cfg['max_depth'] == 2 and not model.__sklearn_is_fitted__()
    assert model.numeric_mode == 'identical'
    model.max_depth = 4
    model._refresh_config()
    assert model._cfg['max_depth'] == 4
    assert model.set_params() is model


@pytest.mark.parametrize('cls', CLASSES)
def test_pipeline_serial_search_and_score_mode(monkeypatch, cls):
    modes = []
    fitted_configs = []
    def fitted(self, X, y, n_classes, fit_fn):
        fitted_configs.append((self._cfg['max_depth'], self.numeric_mode))
        self._offsets = np.array([0, 1], dtype=np.int32)
        self.n_features_in_ = X.shape[1]
        return self
    monkeypatch.setattr(cls, '_fit_arrays', fitted)
    # Fit still executes the public label encoding/domain checks and config refresh.
    class Binding:
        def __getattr__(self, name):
            return lambda *args: None
    monkeypatch.setattr(cls, '_bind', lambda self, *args: Binding())
    classifier = 'Classifier' in cls.__name__
    def predict(self, X):
        values = (X[:, 0] > 0).astype(np.int32)
        if self._cfg['max_depth'] == 1:
            values[:] = 0
        return np.asarray(self.classes_)[values] if classifier else values.astype(np.float32)
    monkeypatch.setattr(cls, 'predict', predict)
    def accuracy(yt, yp, *, numeric_mode):
        yt, yp = np.asarray(yt), np.asarray(yp)
        modes.append(numeric_mode)
        assert yt.dtype == yp.dtype == np.int32
        return float(np.mean(yt == yp))
    def r2(yt, yp, *, numeric_mode):
        yt, yp = np.asarray(yt), np.asarray(yp)
        modes.append(numeric_mode)
        assert yt.dtype == yp.dtype == np.float32
        return float(1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2))
    monkeypatch.setattr(_metrics_impl, 'accuracy_score', accuracy)
    monkeypatch.setattr(_metrics_impl, 'r2_score', r2)
    X = np.tile(np.array([[-2.], [2.]], np.float32), (12, 1))
    y = np.tile(np.array(['negative', 'positive']) if classifier else np.array([0., 1.]), 12)
    pipeline = Pipeline([('scale', StandardScaler()),
                         ('model', cls(numeric_mode='identical'))])
    search = GridSearchCV(pipeline, {'model__max_depth': [1, 2]}, cv=3,
                          n_jobs=1, error_score='raise')
    assert search.fit(X, y) is search
    assert search.best_params_ == {'model__max_depth': 2}
    assert search.score(X, y) == 1
    assert set(modes) == {'identical'}
    assert set(fitted_configs) == {(1, 'identical'), (2, 'identical')}
    with pytest.raises(NotImplementedError, match='sample_weight'):
        search.best_estimator_[-1].score(X, y, sample_weight=np.ones(len(y)))
    with pytest.raises(ValueError, match='one-dimensional'):
        search.best_estimator_[-1].score(X, y[:, None])
    with pytest.raises(ValueError, match='lengths differ'):
        search.best_estimator_[-1].score(X, y[:-1])


@pytest.mark.parametrize('cls', CLASSES)
def test_inference_only_archive_does_not_invent_fit_params(cls):
    loaded = cls.__new__(cls)
    loaded._offsets = np.array([0, 1])
    check_is_fitted(loaded)
    with pytest.raises(ValueError, match='inference-only'):
        clone(loaded)


@pytest.mark.parametrize('cls', CLASSES)
def test_failed_refit_cannot_reuse_old_model(monkeypatch, cls):
    model = cls()
    model._offsets = np.array([0, 1])
    model.classes_ = np.array(['old', 'labels'])
    class Binding:
        def __getattr__(self, name):
            return lambda *args: None
    monkeypatch.setattr(cls, '_bind', lambda self, *args: Binding())
    def failure(*args):
        raise ValueError('native fixture failure')
    monkeypatch.setattr(cls, '_fit_arrays', failure)
    with pytest.raises(ValueError, match='native fixture failure'):
        model.fit(np.ones((2, 1)), np.array([0, 1]))
    assert not model.__sklearn_is_fitted__()
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_custom_constructor_does_not_silently_drop_parameters():
    class CustomForest(RandomForestRegressor):
        def __init__(self, custom=7, **kwargs):
            self.custom = custom
            super().__init__(**kwargs)
    with pytest.raises(TypeError, match="custom constructor"):
        clone(CustomForest(custom=9))
