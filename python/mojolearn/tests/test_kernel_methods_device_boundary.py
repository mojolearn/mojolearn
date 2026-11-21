# SPDX-License-Identifier: Apache-2.0
"""KernelRidge, Nystroem and RBFSampler on both devices: the fit-level rows of `device=`.

Construction, refusals, `set_params`, clone and pickle of the parameter itself are covered for every class that takes
`device=` by `test_device_parameter.py`, and KernelRidge's shared rows by `test_family_device_boundary.py`. Here: the
two transformers (which have no `predict`), every fit route of the door (fit then transform, the one-call
fit_transform, gamma='scale', a weighted and a multi-target ridge), and KernelRidge's resident copy of the fit when
the estimator moves to the other device. On a GPU installation `device="cpu"` runs the CPU binary beside the GPU one
in one process; in `identical` mode it returns the bits the GPU returns. What needs two devices is skipped by name on
a CPU-only installation.
"""
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn import _check_both as cb
from mojolearn.kernel_methods import KernelRidge, Nystroem, RBFSampler

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def data(rows=160, cols=6, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    y = (x @ np.arange(1, cols + 1, dtype=np.float32) + np.float32(0.5)).astype(np.float32)
    return x, y


def nystroem(model, x, y):
    model.fit(x)
    return [np.asarray(model.components_), np.asarray(model.component_indices_), np.asarray(model.normalization_),
            np.asarray(model.transform(x))]


def nystroem_joint(model, x, y):
    return [np.asarray(model.fit_transform(x)), np.asarray(model.components_), np.asarray(model.transform(x))]


def sampler(model, x, y):
    model.fit(x)
    return [np.asarray(model.random_weights_), np.asarray(model.random_offset_), np.asarray(model.transform(x))]


def sampler_joint(model, x, y):
    return [np.asarray(model.fit_transform(x)), np.asarray(model.random_weights_), np.asarray(model.transform(x))]


def ridge(model, x, y):
    model.fit(x, y)
    # the second predict reads the resident copy of the fit where the module keeps one
    return [np.asarray(model.dual_coef_), np.asarray(model.predict(x)), np.asarray(model.predict(x[:17])),
            np.float64(model.score(x, y))]


def ridge_weighted(model, x, y):
    weights = np.linspace(0.5, 1.5, len(x)).astype(np.float64)
    model.fit(x, np.stack([y, -y], axis=1), sample_weight=weights)
    return [np.asarray(model.dual_coef_), np.asarray(model.predict(x))]


# (id, class, constructor keywords, the answers compared bit for bit)
CASES = [
    ("nystroem-rbf", Nystroem, dict(kernel="rbf", n_components=24, random_state=0), nystroem),
    ("nystroem-poly-joint", Nystroem, dict(kernel="poly", degree=2, n_components=24, random_state=0), nystroem_joint),
    ("rbf-sampler", RBFSampler, dict(gamma=0.5, n_components=32, random_state=0), sampler),
    ("rbf-sampler-scale-joint", RBFSampler, dict(gamma="scale", n_components=32, random_state=0), sampler_joint),
    ("kernel-ridge-laplacian", KernelRidge, dict(kernel="laplacian", alpha=0.5), ridge),
    ("kernel-ridge-weighted-two-targets", KernelRidge, dict(kernel="rbf", alpha=1.0), ridge_weighted),
]
case = pytest.mark.parametrize("spec", CASES, ids=[row[0] for row in CASES])


def build(spec, **extra):
    return spec[1](**dict(spec[2], **extra))


@case
def test_default_device_is_this_installations_device(spec):
    x, y = data()
    model = build(spec)
    assert model.vendor_used() == mojolearn.vendor()           # answers before the first fit
    out = spec[3](model, x, y)
    assert model.get_params()["device"] is None
    assert model.vendor_used() == mojolearn.vendor()
    assert all(np.isfinite(np.asarray(part, dtype=np.float64)).all() for part in out)


@case
def test_cpu_device_runs_the_cpu_binary(spec):
    x, y = data()
    model = build(spec, device="cpu")
    out = spec[3](model, x, y)
    assert model.vendor_used() == "cpu"
    assert all(np.isfinite(np.asarray(part, dtype=np.float64)).all() for part in out)


@two_devices
@case
def test_cpu_device_does_nothing_on_the_gpu(spec):
    x, y = data()
    spec[3](build(spec, device="cpu"), x, y)
    if not cb.activity_begin(mojolearn):
        pytest.skip("this installation has no activity record")
    spec[3](build(spec, device="cpu"), x, y)
    activity = cb.activity_end(mojolearn)
    assert activity is not None
    assert activity["launches"] == 0, activity["launch_names"]
    assert activity["upload_calls"] == 0 and activity["download_calls"] == 0


@two_devices
@case
def test_identical_mode_cpu_device_returns_the_gpu_bits(spec):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x, y = data()
    gpu = spec[3](build(spec, numeric_mode="identical"), x, y)
    cpu = spec[3](build(spec, numeric_mode="identical", device="cpu"), x, y)
    assert len(gpu) == len(cpu)
    for left, right in zip(gpu, cpu):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()


@case
def test_pickle_keeps_the_device_and_the_answers(spec):
    x, y = data()
    model = build(spec, device="cpu")
    spec[3](model, x, y)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params()["device"] == "cpu" and restored.vendor_used() == "cpu"
    query = (lambda m: m.predict(x)) if spec[1] is KernelRidge else (lambda m: m.transform(x))
    assert np.asarray(query(restored)).tobytes() == np.asarray(query(model)).tobytes()


@two_devices
def test_set_params_moves_a_fitted_kernel_ridge_and_its_resident_copy():
    x, y = data()
    model = KernelRidge(kernel="rbf", alpha=1.0).fit(x, y)
    first = np.asarray(model.predict(x)).copy()                # the device copy of the fit is minted here
    assert np.asarray(model.predict(x)).tobytes() == first.tobytes()
    model.set_params(device="cpu")
    assert model.vendor_used() == "cpu"
    moved = np.asarray(model.predict(x))                       # the CPU module: per-call entry, the copy is released
    if mojolearn.numeric_mode() == "identical":
        assert moved.tobytes() == first.tobytes()
    model.set_params(device=None)
    assert model.vendor_used() == mojolearn.vendor()
    again = np.asarray(model.predict(x))                       # uploaded again, once
    assert again.tobytes() == first.tobytes()
    assert np.asarray(model.predict(x)).tobytes() == first.tobytes()


@case
def test_check_runs_both_devices_in_this_process(spec, capsys):
    x, y = data()
    model = build(spec)
    spec[3](model, x, y)
    result = model.check(x)
    capsys.readouterr()
    if not TWO_DEVICES or mojolearn.numeric_mode() != "identical":
        assert not result["ran"]
        return
    assert result["ran"] and result["transfer"] == "device"
    assert result["sides"]["gpu"]["route"] == dict(device=None, vendor_used=mojolearn.vendor())
    assert result["sides"]["cpu"]["route"] == dict(device="cpu", vendor_used="cpu")
