# SPDX-License-Identifier: Apache-2.0
"""`score` of a `device="cpu"` Ridge or LinearRegression runs its R^2 program on the CPU metrics binary.

On a GPU installation a `device="cpu"` estimator's route holds the CPU identical metrics entry
(`_mojolearn_x_metrics_host`), resolved once with its fit and predict entries. Its `score` launches nothing on the
GPU and uploads nothing; in `identical` mode it returns the bits the GPU estimator's `score` returns. The tests that
need two devices are skipped by name on a CPU-only installation.
"""
import struct
import sys

import numpy as np
import pytest

import mojolearn
from mojolearn import LinearRegression, Ridge
from mojolearn import _check_both as cb

CLASSES = [Ridge, LinearRegression]
TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def data(rows=300, cols=5, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    noise = rng.random(rows, dtype=np.float32) * np.float32(0.25)
    y = (x @ np.arange(1, cols + 1, dtype=np.float32) + noise).astype(np.float32)
    return x, y


def bits(value):
    return struct.pack("<d", float(value))


@pytest.mark.parametrize("cls", CLASSES)
def test_score_on_the_default_device_is_a_float(cls):
    x, y = data()
    value = cls().fit(x, y).score(x, y)
    assert type(value) is float and 0.0 < value <= 1.0


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_score_loads_the_cpu_metrics_binary(cls):
    x, y = data()
    cls(device="cpu").fit(x, y).score(x, y)
    assert "mojolearn._host._mojolearn_x_metrics_host" in sys.modules


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_score_does_nothing_on_the_gpu(cls):
    x, y = data()
    model = cls(device="cpu").fit(x, y)
    model.score(x, y)                                   # the route is resolved before the record starts
    if not cb.activity_begin(mojolearn):
        pytest.skip("this installation has no activity record")
    model.score(x, y)
    activity = cb.activity_end(mojolearn)
    assert activity is not None
    assert activity["launches"] == 0, activity["launch_names"]
    assert activity["upload_calls"] == 0 and activity["download_calls"] == 0


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_default_device_score_still_runs_on_the_gpu_beside_a_cpu_estimator(cls):
    x, y = data()
    cpu = cls(device="cpu").fit(x, y)
    gpu = cls().fit(x, y)
    cpu.score(x, y)
    gpu.score(x, y)
    if not cb.activity_begin(mojolearn):
        pytest.skip("this installation has no activity record")
    gpu.score(x, y)
    activity = cb.activity_end(mojolearn)
    assert activity is not None and activity["launches"] > 0


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_identical_mode_cpu_device_score_returns_the_gpu_bits(cls):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("the process loaded the fast binaries, which make no promise about bits")
    x, y = data()
    gpu = cls(numeric_mode="identical").fit(x, y)
    cpu = cls(numeric_mode="identical", device="cpu").fit(x, y)
    assert bits(cpu.score(x, y)) == bits(gpu.score(x, y))


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_set_params_to_cpu_moves_score_to_the_cpu_metrics_binary(cls):
    x, y = data()
    model = cls().fit(x, y)
    before = model.score(x, y)
    model.set_params(device="cpu")
    model.score(x, y)
    if not cb.activity_begin(mojolearn):
        pytest.skip("this installation has no activity record")
    after = model.score(x, y)
    activity = cb.activity_end(mojolearn)
    assert activity is not None and activity["launches"] == 0
    if mojolearn.numeric_mode() == "identical":
        assert bits(after) == bits(before)


@pytest.mark.parametrize("cls", CLASSES)
def test_score_refuses_a_target_of_another_length_on_every_device(cls):
    x, y = data()
    for device in (None, "cpu"):
        model = cls(device=device).fit(x, y)
        with pytest.raises(ValueError, match="different shapes"):
            model.score(x, y[:-1])
