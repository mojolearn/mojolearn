# SPDX-License-Identifier: Apache-2.0
"""AlternatingLeastSquares on both devices: the fit-level rows of `device=`.

Construction, refusals, `set_params`, clone and pickle of the parameter itself are covered for every class that takes
`device=` by `test_device_parameter.py`. Here: on a GPU installation `device="cpu"` runs the CPU binary beside the GPU
one in one process (`vendor_used() == "cpu"`, nothing launched on the GPU by fit, recommend or similar_items); in
`identical` mode it returns the bits the GPU returns. What needs two devices is skipped by name on a CPU-only
installation.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import _check_both as cb
from mojolearn._expansion_decomp import AlternatingLeastSquares

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def interactions(users=30, items=18, seed=6):
    rng = np.random.default_rng(seed)
    return (rng.random((users, items), dtype=np.float32) > np.float32(0.7)).astype(np.float32)


def build(**extra):
    return AlternatingLeastSquares(**dict(dict(factors=4, iterations=4, random_state=1), **extra))


def answers(model, r):
    ids, scores = model.recommend(3, r, N=5)
    near, weights = model.similar_items(2, N=4)
    return [np.asarray(model.user_factors), np.asarray(model.item_factors), np.asarray(ids), np.asarray(scores),
            np.asarray(near), np.asarray(weights)]


def test_default_device_is_this_installations_device():
    r = interactions()
    model = build().fit(r)
    assert model.get_params()["device"] is None
    assert model.vendor_used() == mojolearn.vendor()
    assert answers(model, r)[0].shape == (30, 4)


def test_cpu_device_runs_the_cpu_binary():
    r = interactions()
    model = build(device="cpu").fit(r)
    assert model.vendor_used() == "cpu"
    assert answers(model, r)[1].shape == (18, 4)


@two_devices
def test_cpu_device_does_nothing_on_the_gpu():
    r = interactions()
    answers(build(device="cpu").fit(r), r)
    if not cb.activity_begin(mojolearn):
        pytest.skip("this installation has no activity record")
    answers(build(device="cpu").fit(r), r)
    activity = cb.activity_end(mojolearn)
    assert activity is not None
    assert activity["launches"] == 0, activity["launch_names"]
    assert activity["upload_calls"] == 0 and activity["download_calls"] == 0


@two_devices
def test_identical_mode_cpu_device_returns_the_gpu_bits():
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    r = interactions()
    gpu = answers(build(numeric_mode="identical").fit(r), r)
    cpu = answers(build(numeric_mode="identical", device="cpu").fit(r), r)
    for left, right in zip(gpu, cpu):
        assert left.tobytes() == right.tobytes()


@two_devices
def test_set_params_moves_the_estimator_to_the_other_device():
    r = interactions()
    model = build().fit(r)
    assert model.vendor_used() == mojolearn.vendor()
    model.set_params(device="cpu")
    model.fit(r)
    assert model.vendor_used() == "cpu"
    model.set_params(device=None)
    model.fit(r)
    assert model.vendor_used() == mojolearn.vendor()
