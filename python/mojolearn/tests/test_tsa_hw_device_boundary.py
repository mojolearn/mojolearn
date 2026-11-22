# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on ExponentialSmoothing, at the Python boundary.

The device is a constructor parameter. None is this installation's device: the GPU where the installation has a GPU
set, else the CPU. An explicit `device="gpu"` on a CPU-only installation is refused in words that name
`device="cpu"`; `device="cpu"` always runs. It is checked when it is given and read once, by the estimator's first
operation, into its native route owner; the instance keeps what was passed and pickle carries it. On a GPU
installation both Holt-Winters operations tables exist in one process and `device="cpu"` runs the CPU binary
(`vendor_used() == "cpu"`); in `identical` mode it returns the bits the GPU returns. These tests run on a GPU
installation and on a CPU-only one; what needs two devices is skipped by name on the CPU-only one.
"""
import gc
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn import ExponentialSmoothing

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def series(ts_num=4, n=48, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(ts_num * n, dtype=np.float32).reshape(ts_num, n)
    return (np.sin(t * 0.5) + 3.0 + 0.05 * rng.random((ts_num, n), dtype=np.float32)).astype(np.float32)


def model(device=None, y=None, **settings):
    y = series() if y is None else y
    settings.setdefault("seasonal", "add")
    settings.setdefault("seasonal_periods", 4)
    settings.setdefault("ts_num", y.shape[0])
    return ExponentialSmoothing(y, device=device, **settings)


def outputs(fitted):
    # the fit's statistics, both prediction ranges (in sample, across the end, past the end) and a forecast
    parts = [np.asarray(fitted.score()), np.asarray(fitted.get_level()), np.asarray(fitted.get_trend()),
             np.asarray(fitted.get_season()), np.asarray(fitted.predict(10, 40)), np.asarray(fitted.predict(40, 56)),
             np.asarray(fitted.predict(50, 60)), np.asarray(fitted.forecast(7))]
    return b"".join(part.tobytes() for part in parts)


def test_device_is_a_constructor_parameter_with_default_none():
    assert model().device is None
    assert model(device=None).device is None
    assert model(device="cpu").device == "cpu"


@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        model(device=bad)


def test_explicit_gpu_follows_the_installation():
    if TWO_DEVICES:
        assert model(device="gpu").fit().vendor_used() == mojolearn.vendor()
    else:
        with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*device='cpu'"):
            model(device="gpu")


def test_default_runs_on_this_installations_device():
    assert model().fit().vendor_used() == mojolearn.vendor()


def test_cpu_device_runs_the_cpu_binary():
    fitted = model(device="cpu").fit()
    assert fitted.vendor_used() == "cpu"
    assert np.asarray(fitted.forecast(5)).shape == (5, 4)          # (h, ts_num), as documented
    assert np.asarray(fitted.predict(0, 50)).shape == (50, 4)


@two_devices
@pytest.mark.parametrize("settings", [dict(seasonal="add"), dict(seasonal="mul"),
                                      dict(seasonal="add", initialization_method="heuristic")])
def test_identical_bits_across_the_two_devices(settings):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("the two devices return the same bits in identical mode")
    gpu = model(device="gpu", **settings).fit()
    cpu = model(device="cpu", **settings).fit()
    assert (gpu.vendor_used(), cpu.vendor_used()) == (mojolearn.vendor(), "cpu")
    assert outputs(gpu) == outputs(cpu)


@two_devices
def test_two_estimators_on_two_devices_interleave():
    gpu = model(device="gpu").fit()
    cpu = model(device="cpu").fit()
    first = (np.asarray(gpu.forecast(3)).tobytes(), np.asarray(cpu.forecast(3)).tobytes())
    for _ in range(3):
        assert np.asarray(cpu.forecast(3)).tobytes() == first[1]
        assert np.asarray(gpu.forecast(3)).tobytes() == first[0]
    assert (gpu.vendor_used(), cpu.vendor_used()) == (mojolearn.vendor(), "cpu")


@two_devices
def test_cpu_device_keeps_no_resident_copy_and_the_gpu_one_is_released():
    from mojolearn import _backend
    count = _backend.binding("_mojolearn_tsa").holtwinters_resident_count
    before = count()
    cpu = model(device="cpu").fit()
    cpu.forecast(4)
    cpu.predict(10, 50)
    assert count() == before
    gpu = model(device="gpu").fit()
    gpu.forecast(4)
    assert count() == before + 1
    del gpu
    gc.collect()
    assert count() == before


def test_constructing_again_reads_the_new_device():
    y = series()
    fitted = model(device="cpu", y=y).fit()
    assert fitted.vendor_used() == "cpu"
    fitted.__init__(y, seasonal="add", seasonal_periods=4, ts_num=4, device=None)
    assert fitted.device is None
    assert fitted.fit().vendor_used() == mojolearn.vendor()


def test_refused_device_at_construction_leaves_the_model_unchanged():
    fitted = model(device="cpu").fit()
    before = np.asarray(fitted.forecast(3)).tobytes()
    with pytest.raises(ValueError, match="device must be"):
        fitted.__init__(series(), seasonal="add", seasonal_periods=4, ts_num=4, device="tpu")
    assert fitted.device == "cpu" and fitted.vendor_used() == "cpu"
    assert np.asarray(fitted.forecast(3)).tobytes() == before


def test_pickle_carries_the_device():
    fitted = model(device="cpu").fit()
    again = pickle.loads(pickle.dumps(fitted))
    assert again.device == "cpu" and again.vendor_used() == "cpu"
    assert np.asarray(again.forecast(3)).tobytes() == np.asarray(fitted.forecast(3)).tobytes()


def test_saved_model_loads_on_this_installations_device(tmp_path):
    fitted = model(device="cpu").fit()
    path = str(tmp_path / "hw.npz")
    fitted.save(path)
    loaded = ExponentialSmoothing.load(path)
    assert np.asarray(loaded.forecast(3)).tobytes() == np.asarray(fitted.forecast(3)).tobytes()


def test_python_events_per_call_do_not_grow_with_the_series():
    import sys

    def events(fitted):
        seen = [0]

        def tracer(frame, event, arg):
            seen[0] += 1
            return tracer
        fitted.forecast(4)
        sys.settrace(tracer)
        try:
            fitted.forecast(4)
        finally:
            sys.settrace(None)
        return seen[0]

    small = model(y=series(n=48)).fit()
    large = model(y=series(n=4096)).fit()
    assert events(small) == events(large)
