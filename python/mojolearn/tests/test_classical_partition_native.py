# SPDX-License-Identifier: Apache-2.0
"""Opt-in native logical partition checks, not physical multi-GPU proof.

Run under the shared native slot with MOJOLEARN_NATIVE_CLASSICAL_PARTITIONS=1.
The sequential local transport deliberately isolates partition correctness.
"""
import os
import pickle
import numpy as np
import pytest
from mojolearn import parallel_forecasting as pf, parallel_gaussian_process as pg
from mojolearn._parallel_worker import execute
from mojolearn._cpu_reference import reference_training

pytestmark = pytest.mark.skipif(os.environ.get('MOJOLEARN_NATIVE_CLASSICAL_PARTITIONS') != '1',
                                reason='opt-in native check; requires shared slot')


class LocalTransport:
    def __init__(self, devices):
        pass
    def map(self, requests):
        # Exercise serialization as a real subprocess transport would.
        return [pickle.loads(pickle.dumps(execute(pickle.loads(pickle.dumps(r))))) for r in requests]
    def close(self):
        pass


@pytest.fixture(autouse=True)
def local_transport(monkeypatch):
    monkeypatch.setattr(pf, 'DevicePool', LocalTransport)
    monkeypatch.setattr(pg, 'DevicePool', LocalTransport)


def same(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    assert aa.shape == bb.shape
    assert aa.dtype == bb.dtype
    assert aa.tobytes() == bb.tobytes()


@pytest.mark.parametrize('width', [1, 2, 3])
def test_native_arima(width):
    from mojolearn import ARIMA
    y = np.random.default_rng(42).normal(size=(5, 32)).astype(np.float32)
    with reference_training():
        m = ARIMA(order=(1, 1, 0), maxiter=3, numeric_mode='identical').fit(y)
    same(m.predict(0, 36), pf.predict_arima(m, 0, 36, series_per_shard=width))
    same(m.forecast(4), pf.forecast_arima(m, 4, series_per_shard=width))


@pytest.mark.parametrize('width', [1, 2, 3])
@pytest.mark.parametrize('seasonal', ['additive', 'multiplicative'])
def test_native_holtwinters(width, seasonal):
    from mojolearn import ExponentialSmoothing
    y = np.asarray([[4 + s + 0.05 * t + (t % 4) * 0.1 for t in range(24)] for s in range(5)], dtype=np.float32)
    with reference_training():
        m = ExponentialSmoothing(y, ts_num=5, seasonal_periods=4, seasonal=seasonal).fit()
    same(m.forecast(7), pf.forecast_exponential_smoothing(m, 7, series_per_shard=width))
    same(m.predict(26, 31, index=3), pf.predict_exponential_smoothing(m, 26, 31, index=3, series_per_shard=width))


@pytest.mark.parametrize('classes', [2, 3])
def test_native_gpc(classes):
    from mojolearn import GaussianProcessClassifier as GPC
    x = np.random.default_rng(75).normal(size=(9, 3)).astype(np.float32)
    y = [i % classes for i in range(9)]
    with reference_training():
        plain = GPC(max_iter_predict=3, numeric_mode='identical').fit(x, y)
        parallel = pg.fit_gaussian_process_classifier(GPC(max_iter_predict=3, numeric_mode='identical'), x, y, devices=(0, 1))
    for one, two in zip(plain.estimators_, parallel.estimators_):
        for name in ('y_train_', 'L_', 'pi_', 'W_sr_'):
            same(getattr(one, name), getattr(two, name))
    assert plain.log_marginal_likelihood_value_ == parallel.log_marginal_likelihood_value_
    for method in ('predict', 'predict_proba'):
        same(getattr(plain, method)(x[:5]), pg.predict_gaussian_process_classifier(parallel, x[:5], method=method, devices=(0, 1)))
