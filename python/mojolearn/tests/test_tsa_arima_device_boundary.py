# SPDX-License-Identifier: Apache-2.0
"""`device=` on ARIMA and AutoARIMA at the Python boundary, and the check form of the three time-series estimators
(ExponentialSmoothing, ARIMA, AutoARIMA): the same estimator on the other device, in this process, holding the
fitted state of the first.

The Holt-Winters construction and fit rows are in test_tsa_hw_device_boundary.py.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import ARIMA, AutoARIMA, ExponentialSmoothing, _check_both

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def series(batch=3, n=96, seed=3):
    rng = np.random.default_rng(seed)
    y = np.zeros((batch, n), dtype=np.float32)
    shocks = rng.normal(0.0, 1.0, size=(batch, n)).astype(np.float32)
    for t in range(1, n):
        y[:, t] = 0.6 * y[:, t - 1] + shocks[:, t]
    return np.ascontiguousarray(y + 5.0, dtype=np.float32)


def _bits(value):
    return np.ascontiguousarray(np.asarray(value)).tobytes()


def test_arima_device_is_a_constructor_parameter():
    assert ARIMA(order=(1, 0, 1)).get_params()["device"] is None
    model = ARIMA(order=(1, 0, 1), device="cpu")
    assert model.get_params()["device"] == "cpu"
    assert model.set_params(device=None).get_params()["device"] is None
    assert AutoARIMA(series()).get_params()["device"] is None
    assert AutoARIMA(series(), device="cpu").get_params()["device"] == "cpu"


@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        ARIMA(order=(1, 0, 1), device=bad)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        AutoARIMA(series(), device=bad)


def test_cpu_arima_fits_and_forecasts():
    y = series()
    model = ARIMA(order=(1, 0, 1), device="cpu").fit(y)
    assert np.asarray(model.forecast(4)).shape == (y.shape[0], 4)
    assert model.vendor_used() == "cpu"


def test_cpu_autoarima_searches_fits_and_forecasts():
    y = series()
    model = AutoARIMA(y, device="cpu")
    # The CPU ARIMA covers p, q, P and Q of at most 1 and d + D of at most 1 (arima/host/arima_oracle.mojo,
    # 422a1b9e5b); every candidate here is one both devices fit.
    model.search(d=range(2), D=range(1), p=range(1, 2), q=range(2), P=range(1), Q=range(1))
    model.fit()
    assert np.asarray(model.forecast(4)).shape[-1] == 4


@two_devices
@pytest.mark.parametrize("order", [(1, 0, 1), (0, 1, 1), (1, 0, 0)])
def test_arima_two_devices_return_the_same_bits(order):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    y = series()
    gpu = ARIMA(order=order, device="gpu").fit(y)
    cpu = ARIMA(order=order, device="cpu").fit(y)
    assert gpu.vendor_used() == mojolearn.vendor() and cpu.vendor_used() == "cpu"
    first = _bits(gpu.forecast(5))
    assert _bits(cpu.forecast(5)) == first
    assert _bits(gpu.forecast(5)) == first             # interleaved: each keeps its own route and resident copy
    assert _bits(cpu.predict(80, 100)) == _bits(gpu.predict(80, 100))


@two_devices
def test_arima_fitted_state_answers_on_the_other_device():
    y = series()
    gpu = ARIMA(order=(1, 0, 1)).fit(y)
    before = _bits(gpu.forecast(5))
    twin = _check_both.on_device(gpu, "cpu")
    assert twin is not gpu and twin.get_params()["device"] == "cpu" and twin.vendor_used() == "cpu"
    assert _bits(twin.forecast(5)) == before
    assert _bits(gpu.forecast(5)) == before             # the owner's resident copy was not taken or released
    del twin
    assert _bits(gpu.forecast(5)) == before


@two_devices
def test_autoarima_fitted_state_answers_on_the_other_device():
    y = series()
    gpu = AutoARIMA(y)
    gpu.search(d=range(2), D=range(1), p=range(1, 2), q=range(2), P=range(1), Q=range(1))
    gpu.fit()
    before = _bits(gpu.forecast(4))
    twin = _check_both.on_device(gpu, "cpu")
    assert twin is not gpu and twin.get_params()["device"] == "cpu"
    assert _bits(twin.forecast(4)) == before
    assert _bits(gpu.forecast(4)) == before


@two_devices
@pytest.mark.parametrize("seasonal", ["add", "mul"])
def test_holtwinters_fitted_state_answers_on_the_other_device(seasonal):
    rng = np.random.default_rng(5)
    t = np.arange(48, dtype=np.float32)
    y = np.ascontiguousarray(10.0 + 0.2 * t + 2.0 * np.sin(2.0 * np.pi * t / 4.0)
                             + rng.normal(0.0, 0.1, size=(3, 48)), dtype=np.float32)
    gpu = ExponentialSmoothing(y, seasonal=seasonal, seasonal_periods=4, ts_num=3).fit()
    before = _bits(gpu.forecast(6))
    twin = _check_both.on_device(gpu, "cpu")
    assert twin is not gpu and twin.vendor_used() == "cpu"
    assert _bits(twin.forecast(6)) == before             # the twin holds the fitted components, no second fit
    assert _bits(gpu.forecast(6)) == before
    unfitted = _check_both.on_device(gpu, "cpu", fitted=False)
    assert _bits(unfitted.fit().forecast(6)) == before  # and a fit on the CPU returns the same bits
