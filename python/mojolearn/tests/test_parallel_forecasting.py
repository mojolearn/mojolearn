"""Partition/assembly tests of the parallel forecasting drivers; these do not
claim physical GPU qualification.

The four drivers are one native call each (`parallel_forecast_call` on the
native process pool, lane migrate-neural-2): the shards, the worker rebuild,
the per-shard predict and the merge all run natively, so these tests compare
the drivers against the estimator's own predict on real fitted models instead
of substituting a Python pool."""
import numpy as np
import pytest
from mojolearn import parallel_forecasting as pf
from mojolearn._arima_impl import ARIMA
from mojolearn._tsa_impl import ExponentialSmoothing


def arima(batch=5):
    rng = np.random.default_rng(3)
    series = np.ascontiguousarray(rng.standard_normal((batch, 48)).astype(np.float32))
    return ARIMA(order=(1, 0, 0)).fit(series)


def hw(batch=5):
    rng = np.random.default_rng(4)
    y = np.ascontiguousarray((10.0 + rng.standard_normal((batch, 40))).astype(np.float32))
    return ExponentialSmoothing(y, ts_num=batch, seasonal_periods=4, seasonal="additive").fit()


@pytest.mark.parametrize('batch,shard', [(1, 1), (5, 1), (5, 2), (5, 3), (5, 9)])
def test_arima_partition_matches_the_estimator(batch, shard):
    model = arima(batch)
    expected = model.predict(40, 52)
    got = pf.predict_arima(model, 40, 52, series_per_shard=shard)
    assert np.asarray(got).shape == np.asarray(expected).shape
    assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()
    forecast = pf.forecast_arima(model, 4, series_per_shard=shard)
    assert np.asarray(forecast).tobytes() == np.asarray(model.forecast(4)).tobytes()


@pytest.mark.parametrize('batch,shard', [(1, 1), (5, 1), (5, 2), (5, 3), (5, 9)])
def test_hw_partition_matches_the_estimator(batch, shard):
    model = hw(batch)
    expected = model.forecast(3)
    got = pf.forecast_exponential_smoothing(model, 3, series_per_shard=shard)
    assert np.asarray(got).shape == np.asarray(expected).shape
    assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize('width', [0, -1, True, 1.5])
def test_invalid_partition(width):
    with pytest.raises(ValueError, match='positive integer'):
        pf.forecast_arima(arima(), 3, series_per_shard=width)


def test_explicit_boundaries():
    with pytest.raises(NotImplementedError, match='exogenous'):
        pf.predict_arima(arima(), 0, 10, exog=np.ones((5, 10), np.float32))
    with pytest.raises(IndexError):
        pf.forecast_exponential_smoothing(hw(), index=5)
