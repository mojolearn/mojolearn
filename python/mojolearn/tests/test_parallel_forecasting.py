# SPDX-License-Identifier: Apache-2.0
"""Partition/assembly tests; these do not claim physical GPU qualification."""
import numpy as np
import pytest
from mojolearn import parallel_forecasting as pf
from mojolearn._arima_impl import ARIMA
from mojolearn._tsa_impl import ExponentialSmoothing
from mojolearn._buffer import Array


def array(x):
    return Array.from_list(np.asarray(x, dtype=np.float32).tolist(), '<f4')


class Pool:
    calls = []
    closed = False
    fail = False
    def __init__(self, devices):
        self.devices = devices
        type(self).closed = False
    def map(self, requests):
        type(self).calls = requests
        if self.fail:
            raise RuntimeError('worker failure')
        return [state.predict(*args[1]) for _, state, args in requests]
    def close(self):
        type(self).closed = True


@pytest.fixture(autouse=True)
def pool(monkeypatch):
    Pool.fail = False
    monkeypatch.setattr(pf, 'DevicePool', Pool)
    monkeypatch.setattr('mojolearn._backend.default_mode', lambda: 'identical')


def arima(batch=5):
    m = ARIMA(order=(1, 0, 0))
    m.batch_size_, m.n_obs_, m.n_exog_ = batch, 8, 0
    m._y = array(np.arange(batch * 8).reshape(batch, 8))
    m.params_ = array(np.arange(batch * 2).reshape(batch, 2))
    m.x_ = m._y
    return m


def arima_prediction(self, start, end):
    # Pure test oracle: each row depends on its own transported data and state.
    y, p = np.asarray(self._y), np.asarray(self.params_)
    return array(y[:, :1] + p[:, :1] + np.arange(start, end)[None, :])


@pytest.mark.parametrize('shard', [1, 2, 3, 9])
def test_arima_exact_uneven_and_immutable(monkeypatch, shard):
    monkeypatch.setattr(ARIMA, 'predict', arima_prediction)
    model = arima()
    expected = model.predict(2, 12)
    before = model.__dict__.copy()
    got = pf.predict_arima(model, 2, 12, devices=(1, 0), series_per_shard=shard)
    assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()
    assert model.__dict__ == before
    assert all('x_' not in state.__dict__ for _, state, _ in Pool.calls)
    assert Pool.closed
    assert np.asarray(pf.forecast_arima(model, 4, series_per_shard=shard)).tobytes() == np.asarray(model.predict(8, 12)).tobytes()


def hw(batch=5):
    m = ExponentialSmoothing.__new__(ExponentialSmoothing)
    m.n, m.seasonal_periods, m.seasonal = 8, 2, 'additive'
    m.ts_num, m.fit_executed_flag = batch, True
    m._comps = array(np.arange(3 * 6 * batch))
    return m


def hw_prediction(self, start, end):
    c = np.asarray(self._comps).reshape(3, 6, self.ts_num)
    values = c[0, -1] + c[1, -1] * np.arange(start, end)[:, None] + c[2, -1]
    return self._shaped(array(values.ravel()), end - start, None)


@pytest.mark.parametrize('batch,shard', [(1, 1), (5, 1), (5, 2), (5, 3), (5, 9)])
@pytest.mark.parametrize('index', [None, 0])
def test_hw_component_partition_time_major_assembly(monkeypatch, batch, shard, index):
    monkeypatch.setattr(ExponentialSmoothing, 'predict', hw_prediction)
    model = hw(batch)
    before = np.asarray(model._comps).tobytes()
    expected = model.predict(10, 13)
    if index is not None and batch > 1:
        expected = np.asarray(expected)[:, index]
    got = pf.predict_exponential_smoothing(model, 10, 13, index=index, series_per_shard=shard)
    assert got.shape == expected.shape
    assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()
    assert np.asarray(model._comps).tobytes() == before
    assert Pool.closed


def test_failures_close_pool_and_do_not_mutate(monkeypatch):
    Pool.fail = True
    for model, invoke in [(arima(), lambda m: pf.forecast_arima(m, 3)),
                          (hw(), lambda m: pf.forecast_exponential_smoothing(m, 3))]:
        before = model.__dict__.copy()
        with pytest.raises(RuntimeError, match='worker failure'):
            invoke(model)
        assert Pool.closed
        assert model.__dict__ == before


@pytest.mark.parametrize('width', [0, -1, True, 1.5])
def test_invalid_partition(width):
    with pytest.raises(ValueError, match='positive integer'):
        pf.forecast_arima(arima(), 3, series_per_shard=width)


def test_explicit_boundaries():
    with pytest.raises(NotImplementedError, match='host arithmetic'):
        pf.predict_exponential_smoothing(hw(), 0, 10)
    with pytest.raises(NotImplementedError, match='exogenous'):
        pf.predict_arima(arima(), 0, 10, exog=array([1]))
    with pytest.raises(IndexError):
        pf.forecast_exponential_smoothing(hw(), index=5)
