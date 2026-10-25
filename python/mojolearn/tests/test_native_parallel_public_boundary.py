# SPDX-License-Identifier: Apache-2.0
"""Runtime admission probes for the installed public parallel boundary."""
import sys

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import parallel_classical as classical, parallel_ensemble as ensemble


class Poison:
    calls = []

    def __iter__(self):
        self.calls.append('iter')
        raise AssertionError('Python iteration hook executed')

    def __array__(self, *args, **kwargs):
        self.calls.append('array')
        raise AssertionError('Python input protocol executed')

    def __bool__(self):
        self.calls.append('bool')
        raise AssertionError('Python truth hook executed')

    def __float__(self):
        self.calls.append('float')
        raise AssertionError('Python scalar hook executed')


@pytest.mark.parametrize('argument', ['devices', 'X', 'sample_weight', 'numeric_mode'])
def test_parallel_admission_refuses_protocols_without_executing_them(argument):
    estimator = ml.KMeans(n_clusters=2, numeric_mode='identical')
    X = np.zeros((4, 2), dtype=np.float32)
    kwargs = {'devices': (0,), 'sample_weight': None}
    Poison.calls.clear()
    if argument == 'X':
        X = Poison()
    elif argument == 'numeric_mode':
        estimator.numeric_mode = Poison()
    else:
        kwargs[argument] = Poison()
    with pytest.raises((TypeError, ValueError), match='native|Python|numeric_mode|unsupported|input'):
        classical.fit_kmeans(estimator, X, **kwargs)
    assert Poison.calls == []


def test_parallel_shell_and_native_rejection_do_not_execute_python_preparation():
    estimator = ml.KMeans(n_clusters=2, numeric_mode='identical')
    frames = []
    def trace(frame, event, arg):
        if event == 'call' and frame.f_globals.get('__name__', '').startswith('mojolearn.'):
            frames.append((frame.f_globals['__name__'], frame.f_code.co_name))
    sys.setprofile(trace)
    try:
        with pytest.raises(ValueError, match='distinct nonnegative'):
            classical.fit_kmeans(estimator, [[1.0], [2.0]], devices=(0, 0))
    finally:
        sys.setprofile(None)
    assert frames == [('mojolearn.parallel_classical', 'fit_kmeans')]


def test_predictor_rejects_subclass_without_estimator_hooks():
    calls = []
    class CallbackForest(ml.RandomForestRegressor):
        def _effective_mode(self):
            calls.append('mode')
            return 'identical'
        def _prediction_engine(self):
            calls.append('engine')
            return 'parallel_groves'
    # Native allocation is not needed to test the exact-class guard.
    estimator = object.__new__(CallbackForest)
    with pytest.raises(TypeError, match='requires.*RandomForest'):
        ensemble.ParallelForestPredictor(estimator, devices=(0,))
    assert calls == []


def test_predictor_snapshot_and_repeated_close():
    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    forest = ml.RandomForestRegressor(n_estimators=3, max_depth=2,
        numeric_mode='identical', inference_engine='parallel_groves').fit(X, X[:, 0])
    pool = ensemble.ParallelForestPredictor(forest, devices=range(1))
    saved = pool._model
    assert saved is not forest
    assert saved._resident_holder is not forest._resident_holder
    assert pool.n_estimators_ == 3 and pool.n_features_in_ == 2
    forest.n_estimators = 5
    assert saved.n_estimators == 3
    assert pool.__enter__() is pool
    pool.close()
    pool.close()
    assert pool._model is None
    with pytest.raises(RuntimeError, match='closed'):
        pool.predict(X)
