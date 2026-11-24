# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the forecast family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""

# moved from python/mojolearn/tests/test_parallel_forecasting.py
import numpy as np
import pytest

from mojolearn import parallel_forecasting as _pfc_pf
from mojolearn._arima_impl import ARIMA as _PfcARIMA
from mojolearn._tsa_impl import ExponentialSmoothing as _PfcExponentialSmoothing


def _pfc_arima(batch=5):
    rng = np.random.default_rng(3)
    series = np.ascontiguousarray(rng.standard_normal((batch, 48)).astype(np.float32))
    return _PfcARIMA(order=(1, 0, 0)).fit(series)


def _pfc_hw(batch=5):
    rng = np.random.default_rng(4)
    y = np.ascontiguousarray((10.0 + rng.standard_normal((batch, 40))).astype(np.float32))
    return _PfcExponentialSmoothing(y, ts_num=batch, seasonal_periods=4, seasonal="additive").fit()


@pytest.mark.parametrize('width', [0, -1, True, 1.5])
def test_parallel_forecast_invalid_partition(width):
    with pytest.raises(ValueError, match='positive integer'):
        _pfc_pf.forecast_arima(_pfc_arima(), 3, series_per_shard=width)


def test_parallel_forecast_explicit_boundaries():
    with pytest.raises(NotImplementedError, match='exogenous'):
        _pfc_pf.predict_arima(_pfc_arima(), 0, 10, exog=np.ones((5, 10), np.float32))
    with pytest.raises(IndexError):
        _pfc_pf.forecast_exponential_smoothing(_pfc_hw(), index=5)
