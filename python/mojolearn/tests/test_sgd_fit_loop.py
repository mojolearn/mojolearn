"""The SGD family's `fit_loop_` at the Python boundary; run only against the integrated native build.

After a fit capped at K epochs with stopping off, `fit_loop_` is a tuple (int, float) of (K, milliseconds >= 0) and its
first item equals `n_iter_`; before a fit it raises the error the estimator's other fitted attributes raise.
"""
import numpy as np
import pytest

import mojolearn as ml

_K = 3

# (class, constructor options, classification targets)
_CASES = (
    (ml.SGDClassifier, {"tol": None, "early_stopping": False}, True),
    (ml.SGDRegressor, {"tol": None, "early_stopping": False}, False),
    (ml.Perceptron, {"tol": None, "early_stopping": False}, True),
    (ml.PassiveAggressiveClassifier, {"tol": None, "early_stopping": False}, True),
    (ml.PassiveAggressiveRegressor, {"tol": None, "early_stopping": False}, False),
    (ml.SGDOneClassSVM, {"tol": None}, None),
)


def _data():
    rng = np.random.default_rng(0)
    X = rng.random((64, 4), dtype=np.float32)
    return X, (X[:, 0] > 0.5).astype(np.int32), (X @ np.arange(1, 5, dtype=np.float32)).astype(np.float32)


@pytest.mark.parametrize("cls, options, classes", _CASES, ids=[case[0].__name__ for case in _CASES])
def test_fit_loop_after_fit(cls, options, classes):
    X, labels, targets = _data()
    model = cls(max_iter=_K, random_state=0, **options)
    if classes is None:
        model.fit(X)
    else:
        model.fit(X, labels if classes else targets)
    loop = model.fit_loop_
    assert type(loop) is tuple and len(loop) == 2
    assert type(loop[0]) is int and type(loop[1]) is float
    assert loop[0] == _K == model.n_iter_
    assert loop[1] >= 0.0


@pytest.mark.parametrize("cls, options, classes", _CASES, ids=[case[0].__name__ for case in _CASES])
def test_fit_loop_before_fit(cls, options, classes):
    model = cls(max_iter=_K, **options)
    with pytest.raises(Exception) as other:
        model.n_iter_
    with pytest.raises(type(other.value)):
        model.fit_loop_
