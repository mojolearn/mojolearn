# SPDX-License-Identifier: Apache-2.0
"""`fit_loop_` at the public boundary of the forests (lane alm-forest,
2026-11-24): after `fit` it is a tuple `(int, float)` whose count is the trees
built (`n_estimators`), before `fit` it raises AttributeError like the other
fitted attributes, and a loaded model has none (it is not saved). The loop
itself is checked natively in checks/native_tests/forest_fit_loop.mojo."""
import numpy as np
import pytest
from mojolearn import (ExtraTreesClassifier, ExtraTreesRegressor, IsolationForest,
                       RandomForestClassifier, RandomForestRegressor)

CLASSES = [ExtraTreesClassifier, ExtraTreesRegressor,
           RandomForestClassifier, RandomForestRegressor, IsolationForest]
K = 3


def _data(cls):
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    if cls is IsolationForest:
        return X, None
    if cls._estimator_type == 'classifier':
        return X, (X[:, 0] > 0.5).astype(np.int32)
    return X, (2.0 * X[:, 0] + X[:, 1]).astype(np.float32)


def _fit(cls):
    X, y = _data(cls)
    model = cls(n_estimators=K, random_state=0)
    return model.fit(X) if y is None else model.fit(X, y)


@pytest.mark.parametrize('cls', CLASSES)
def test_fit_loop_is_a_count_and_milliseconds(cls):
    loop = _fit(cls).fit_loop_
    assert type(loop) is tuple and len(loop) == 2
    assert type(loop[0]) is int and type(loop[1]) is float
    assert loop[0] == K
    assert loop[1] >= 0.0


@pytest.mark.parametrize('cls', CLASSES)
def test_fit_loop_before_fit_raises(cls):
    model = cls(n_estimators=K, random_state=0)
    with pytest.raises(AttributeError):
        model.fit_loop_
    assert not hasattr(model, 'fit_loop_')


@pytest.mark.parametrize('cls', CLASSES)
def test_fit_loop_is_not_saved(cls, tmp_path):
    model = _fit(cls)
    path = tmp_path / 'model.mojolearn'
    model.save(str(path))
    loaded = cls.load(str(path))
    with pytest.raises(AttributeError):
        loaded.fit_loop_
