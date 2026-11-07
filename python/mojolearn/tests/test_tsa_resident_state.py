"""Resident fitted state of the time-series forecasters (plan
NATIVE_ENTRY_POINTS_AFTER_0866 section A, rows N028-N031).

Each case checks the rules of the resident scheme: the resident answer has
the per-call path's bits (a model whose fitted array is a plain ndarray
cannot be frozen and keeps the per-call upload), a fitted array behind a
device copy is read-only, replacing it is honoured, and the device copy is
released with the model.
"""
import copy
import gc

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend


def _count(binding, name):
    module = _backend.binding(binding)
    entry = getattr(module, name, None)
    if entry is None:
        pytest.skip(binding + " has no resident entries (host column)")
    return entry


def _per_call_twin(model, names):
    """A copy whose fitted arrays are writable ndarrays: never frozen, so it
    runs the per-call upload path."""
    twin = copy.deepcopy(model)
    fields = object.__getattribute__(twin, "__dict__")
    for name in names:
        if fields.get(name) is not None:
            fields[name] = np.array(np.asarray(fields[name]), copy=True)
    return twin


def test_arima_resident_matches_per_call_and_lifetime():
    count = _count("_mojolearn_arima", "arima_resident_count")
    y = np.random.default_rng(7).normal(size=(3, 80)).astype(np.float32)
    before = count()
    model = ml.ARIMA(order=(1, 1, 1), maxiter=5).fit(y)
    twin = _per_call_twin(model, ("_y", "params_"))
    first = np.asarray(model.forecast(6))
    assert count() == before + 1
    again = np.asarray(model.forecast(9))
    inside = np.asarray(model.predict(10, 90))
    assert count() == before + 1
    np.testing.assert_array_equal(first.view(np.uint32), np.asarray(twin.forecast(6)).view(np.uint32))
    np.testing.assert_array_equal(again.view(np.uint32), np.asarray(twin.forecast(9)).view(np.uint32))
    np.testing.assert_array_equal(inside.view(np.uint32), np.asarray(twin.predict(10, 90)).view(np.uint32))
    with pytest.raises(ValueError):
        np.asarray(model.params_)[0, 0] = 1.0
    model.fit(y[:, ::-1].copy())  # a refit stores new arrays: the next query mints again
    _ = model.forecast(3)
    gc.collect()
    assert count() == before + 1
    del model
    gc.collect()
    assert count() == before


def test_arima_exog_resident_matches_per_call():
    _count("_mojolearn_arima", "arima_resident_count")
    rng = np.random.default_rng(11)
    y = rng.normal(size=(2, 64)).astype(np.float32)
    exog = rng.normal(size=(2, 64, 1)).astype(np.float32)
    future = rng.normal(size=(2, 5, 1)).astype(np.float32)
    model = ml.ARIMA(order=(1, 0, 0), maxiter=5).fit(y, exog)
    twin = _per_call_twin(model, ("_y", "params_", "_exog"))
    got = np.asarray(model.forecast(5, future))
    np.testing.assert_array_equal(got.view(np.uint32), np.asarray(twin.forecast(5, future)).view(np.uint32))


def test_holtwinters_resident_one_call_matches_per_call():
    count = _count("_mojolearn_tsa", "holtwinters_resident_count")
    y = (np.sin(np.arange(4 * 48) * 0.5).reshape(4, 48) + 3.0).astype(np.float32)
    before = count()
    model = ml.ExponentialSmoothing(y, seasonal="add", seasonal_periods=4, ts_num=4).fit()
    twin = _per_call_twin(model, ("_comps",))
    spans = [(10, 40), (40, 56), (50, 60)]
    for start, end in spans:
        got = np.asarray(model.predict(start, end))
        want = np.asarray(twin.predict(start, end))
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    got = np.asarray(model.forecast(7))
    np.testing.assert_array_equal(got.view(np.uint32), np.asarray(twin.forecast(7)).view(np.uint32))
    assert count() == before + 1
    del model
    gc.collect()
    assert count() == before


def test_theta_new_horizon_runs_the_forecast_tail():
    y = (np.arange(3 * 60, dtype=np.float32).reshape(3, 60) % 7 + 10.0).astype(np.float32)
    model = ml.AutoTheta(season_length=7).fit(y)
    model.predict(7)
    long = model.predict(14)  # same fit, new horizon: the forecast-only tail
    other = ml.AutoTheta(season_length=7).fit(y)
    fresh = other.predict(14)  # the whole fit at h=14
    for key in fresh:
        np.testing.assert_array_equal(np.asarray(long[key]).view(np.uint32), np.asarray(fresh[key]).view(np.uint32))
    np.testing.assert_array_equal(np.asarray(model.info_).view(np.uint32), np.asarray(other.info_).view(np.uint32))
    assert list(model.model_) == list(other.model_)


def test_prophet_predict_repeats_bits():
    t = np.arange(120, dtype=np.float64)
    y = (np.sin(t / 7.0) + t / 50.0).astype(np.float32)
    model = ml.ProphetForecaster().fit(t, y)
    future = np.arange(130, dtype=np.float64)
    a = np.asarray(model.predict(future))
    b = np.asarray(model.predict(future))
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
