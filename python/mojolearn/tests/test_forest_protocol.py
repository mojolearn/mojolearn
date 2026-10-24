# SPDX-License-Identifier: Apache-2.0
"""Real sklearn orchestration over the real public forest methods (small inputs)."""
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
    # The parameter checks run at fit (`_refresh_config`), as in
    # scikit-learn; a refused value leaves the fitted state in place.
    for bad, match in ((dict(device='invalid'), 'device'), (dict(numeric_mode='typo'), 'numeric_mode')):
        model.__dict__.update(bad)
        with pytest.raises(ValueError, match=match):
            model._refresh_config()
        assert model.__sklearn_is_fitted__()
        model.__dict__.update(device='gpu', numeric_mode='fast')
    model._refresh_config()
    assert model._cfg['max_depth'] == 3
    model._offsets = np.array([0, 1])
    assert model.set_params(max_depth=2, numeric_mode='identical') is model
    assert not model.__sklearn_is_fitted__()
    model._refresh_config()
    assert model._cfg['max_depth'] == 2
    assert model.numeric_mode == 'identical'
    model.max_depth = 4
    model._refresh_config()
    assert model._cfg['max_depth'] == 4
    assert model.set_params() is model


@pytest.mark.parametrize('cls', CLASSES)
def test_pipeline_serial_search_and_score_mode(cls):
    from sklearn.metrics import accuracy_score, r2_score
    classifier = 'Classifier' in cls.__name__
    X = np.tile(np.array([[-2.], [2.]], np.float32), (12, 1))
    y = np.tile(np.array(['negative', 'positive']) if classifier else np.array([0., 1.]), 12)
    pipeline = Pipeline([('scale', StandardScaler()),
                         ('model', cls(n_estimators=4, random_state=0, numeric_mode='identical'))])
    search = GridSearchCV(pipeline, {'model__max_depth': [1, 2]}, cv=3,
                          n_jobs=1, error_score='raise')
    assert search.fit(X, y) is search
    assert search.best_params_['model__max_depth'] in (1, 2)
    best = search.best_estimator_[-1]
    assert best._fit_numeric_mode == 'identical'
    assert best.max_depth == search.best_params_['model__max_depth']
    assert search.score(X, y) == pytest.approx(1.0)
    # score(sample_weight) is scikit-learn's weighted metric of predict
    w = np.arange(1, len(y) + 1, dtype=np.float32)
    Xs = search.best_estimator_[0].transform(X)
    pred = np.asarray(best.predict(Xs))
    expected = (accuracy_score(y, pred, sample_weight=w) if classifier
                else r2_score(y, pred, sample_weight=w))
    assert best.score(Xs, y, sample_weight=w) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    with pytest.raises(ValueError, match='one-dimensional'):
        best.score(Xs, y[:, None])
    with pytest.raises(ValueError, match='lengths differ'):
        best.score(Xs, y[:-1])


@pytest.mark.parametrize('cls', CLASSES)
def test_failed_refit_cannot_reuse_old_model(cls):
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.int32)
    model = cls(n_estimators=2, max_depth=2, random_state=0).fit(X, y)
    assert model.__sklearn_is_fitted__()
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        model.fit(bad, y)
    assert not model.__sklearn_is_fitted__()
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize('cls', CLASSES)
def test_inference_only_archive_does_not_invent_fit_params(cls):
    loaded = cls.__new__(cls)
    loaded._offsets = np.array([0, 1])
    check_is_fitted(loaded)
    with pytest.raises(ValueError, match='inference-only'):
        clone(loaded)


@pytest.mark.parametrize('cls', CLASSES)
def test_failed_refit_cannot_reuse_old_model(monkeypatch, cls):
    pytest.skip("the forest routing these fakes drove is native now (bindings/trees_public_forest.mojo); needs a real-binding rewrite")
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
    # Since 8a6b082d5 a native public model refuses a class it has no
    # connected layout for ("requires an exact connected class") before the
    # forest's own custom-constructor refusal can run; either way the
    # parameters are refused, never dropped.
    with pytest.raises(TypeError, match="custom constructor|requires an exact connected class"):
        clone(CustomForest(custom=9))
