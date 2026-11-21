# SPDX-License-Identifier: Apache-2.0
"""PLSRegression, PLSCanonical and CCA on both devices: the fit-level rows of `device=`.

Construction, refusals, `set_params`, clone and pickle of the parameter itself are covered for every class that takes
`device=` by `test_device_parameter.py`. Here: on a GPU installation `device="cpu"` runs the CPU binary beside the GPU
one in one process (`vendor_used() == "cpu"`, nothing launched on the GPU by fit, transform, predict or score); in
`identical` mode it returns the bits the GPU returns. What needs two devices is skipped by name on a CPU-only
installation.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import _check_both as cb
from mojolearn import CCA, PLSCanonical, PLSRegression

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")
case = pytest.mark.parametrize("cls", [PLSRegression, PLSCanonical, CCA], ids=["pls-regression", "pls-canonical", "cca"])


def data(rows=120, cols=7, targets=3, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    y = (x[:, :targets] + np.float32(0.25) * rng.random((rows, targets), dtype=np.float32)).astype(np.float32)
    return x, y


def build(cls, **extra):
    return cls(**dict(dict(n_components=2), **extra))


def answers(model, x, y):
    xs, ys = model.transform(x, y)
    return [np.asarray(model.x_weights_), np.asarray(model.y_weights_), np.asarray(model.coef_), np.asarray(xs),
            np.asarray(ys), np.asarray(model.predict(x)), np.asarray(model.inverse_transform(np.asarray(xs))),
            np.float64(model.score(x, y))]


@case
def test_default_device_is_this_installations_device(cls):
    x, y = data()
    model = build(cls)
    assert model.vendor_used() == mojolearn.vendor()           # answers before the first fit
    model.fit(x, y)
    assert model.get_params()["device"] is None
    assert model.vendor_used() == mojolearn.vendor()
    assert answers(model, x, y)[3].shape == (120, 2)


@case
def test_cpu_device_runs_the_cpu_binary(cls):
    x, y = data()
    model = build(cls, device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"
    assert answers(model, x, y)[5].shape == (120, 3)


@two_devices
@case
def test_cpu_device_does_nothing_on_the_gpu(cls):
    x, y = data()
    answers(build(cls, device="cpu").fit(x, y), x, y)
    if not cb.activity_begin(mojolearn):
        pytest.skip("this installation has no activity record")
    answers(build(cls, device="cpu").fit(x, y), x, y)
    activity = cb.activity_end(mojolearn)
    assert activity is not None
    assert activity["launches"] == 0, activity["launch_names"]
    assert activity["upload_calls"] == 0 and activity["download_calls"] == 0


@two_devices
@case
def test_identical_mode_cpu_device_returns_the_gpu_bits(cls):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x, y = data()
    gpu = answers(build(cls, numeric_mode="identical").fit(x, y), x, y)
    cpu = answers(build(cls, numeric_mode="identical", device="cpu").fit(x, y), x, y)
    for left, right in zip(gpu, cpu):
        assert left.tobytes() == right.tobytes()


@two_devices
@case
def test_set_params_moves_the_estimator_to_the_other_device(cls):
    x, y = data()
    model = build(cls).fit(x, y)
    assert model.vendor_used() == mojolearn.vendor()
    model.set_params(device="cpu")
    assert model.vendor_used() == "cpu"
    # a fitted model queried on the other device: the fitted operands are re-admitted on that image
    assert np.asarray(model.predict(x)).shape == (120, 3)
    model.fit(x, y)
    assert model.vendor_used() == "cpu"
    model.set_params(device=None)
    model.fit(x, y)
    assert model.vendor_used() == mojolearn.vendor()
