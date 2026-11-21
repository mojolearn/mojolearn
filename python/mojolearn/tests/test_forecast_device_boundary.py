# SPDX-License-Identifier: Apache-2.0
"""`device=` on the forecasters at the Python boundary: the Theta family, ETS, DampedETS, GARCH, VAR, STL and
ProphetForecaster.

* `device` is a constructor parameter (default None, stored as passed, returned by `get_params`, carried by
  `set_params`); an unknown value is refused at construction in the shared words;
* a forecaster built with `device="cpu"` fits and forecasts;
* on an installation with a GPU, in `identical` mode, the CPU forecaster returns the bits of the GPU forecaster, and
  two forecasters on two devices keep their own images across interleaved calls. `VARResults` forecasts on the
  device of the `VAR` that fitted it.
"""
import numpy as np
import pytest

import mojolearn

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def series(batch=3, n=96, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    season = 2.0 * np.sin(2.0 * np.pi * t / 12.0)
    y = 20.0 + 0.05 * t + season + rng.normal(0.0, 0.5, size=(batch, n))
    return np.ascontiguousarray(y, dtype=np.float32)


def returns(batch=3, n=160, seed=9):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(0.0, 1.0, size=(batch, n)), dtype=np.float32)


def _flat(value):
    if isinstance(value, dict):
        return [np.asarray(value[key]) for key in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [np.asarray(item) for item in value]
    return [np.asarray(value)]


def _same(a, b):
    left, right = _flat(a), _flat(b)
    return len(left) == len(right) and all(x.tobytes() == y.tobytes() for x, y in zip(left, right))


def _theta(cls):
    return lambda **kw: (getattr(mojolearn, cls)(season_length=12, **kw), lambda m: m.fit(series()).predict(6))


def _ets(**kw):
    return mojolearn.ETS(season_length=12, model="AAA", **kw), lambda m: m.fit(series()).predict(6)


def _ets_multiplicative(**kw):
    # a multiplicative season reads the positive-data scans of the fit
    return mojolearn.ETS(season_length=12, model="MAM", **kw), lambda m: m.fit(series()).predict(6)


def _damped(**kw):
    return mojolearn.DampedETS(season_length=1, **kw), lambda m: m.fit(series()).predict(6)


def _garch(**kw):
    return mojolearn.GARCH(**kw), lambda m: m.fit(returns(), horizon=3).forecast(3)


def _var(**kw):
    y = np.ascontiguousarray(series(batch=3).T)
    return mojolearn.VAR(y, **kw), lambda m: m.fit(maxlags=2).forecast(y[-2:], 4)


def _stl(**kw):
    def run(m):
        result = m.fit()
        return (result.seasonal, result.trend, result.resid)
    return mojolearn.STL(series(), period=12, **kw), run


def _prophet(**kw):
    def run(m):
        t = np.arange(96, dtype=np.float64)          # days
        return m.fit(t, series()).predict(t[-8:])
    return mojolearn.ProphetForecaster(n_changepoints=5, max_iter=50, **kw), run


CASES = [
    ("AutoTheta", _theta("AutoTheta")), ("Theta", _theta("Theta")), ("OptimizedTheta", _theta("OptimizedTheta")),
    ("DynamicTheta", _theta("DynamicTheta")), ("DynamicOptimizedTheta", _theta("DynamicOptimizedTheta")),
    ("ETS", _ets), ("ETS-multiplicative", _ets_multiplicative), ("DampedETS", _damped), ("GARCH", _garch),
    ("VAR", _var), ("STL", _stl), ("ProphetForecaster", _prophet),
]
case = pytest.mark.parametrize("spec", CASES, ids=[c[0] for c in CASES])


@case
def test_device_is_a_constructor_parameter(spec):
    model, _ = spec[1]()
    assert model.get_params()["device"] is None
    model, _ = spec[1](device="cpu")
    assert model.get_params()["device"] == "cpu"
    assert model.set_params(device=None).get_params()["device"] is None


@case
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(spec, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        spec[1](device=bad)


@case
def test_cpu_forecaster_fits_and_forecasts(spec):
    model, run = spec[1](device="cpu")
    answer = _flat(run(model))
    assert all(np.isfinite(item.astype(np.float64)).all() for item in answer)


@two_devices
@case
def test_two_devices_return_the_same_bits(spec):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    gpu, run_gpu = spec[1](device="gpu")
    cpu, run_cpu = spec[1](device="cpu")
    first = run_gpu(gpu)
    assert _same(run_cpu(cpu), first)
    assert _same(run_gpu(gpu), first)          # interleaved: each keeps its own route
    assert _same(run_cpu(cpu), first)


@two_devices
def test_var_results_forecast_on_the_device_of_their_var():
    y = np.ascontiguousarray(series(batch=3).T)
    results = mojolearn.VAR(y, device="cpu").fit(maxlags=2)
    assert results.device == "cpu"
    assert mojolearn.VAR(y).fit(maxlags=2).device is None
