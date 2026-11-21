# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on the Gaussian process estimators, at the Python boundary.

The device is a constructor parameter of GaussianProcessRegressor and GaussianProcessClassifier. None is this
installation's device: the GPU where the installation has a GPU set, else the CPU. `get_params` returns what was
passed (None stays None). The constructor stores the value as given, as it stores every other parameter of these
classes; it is read once, by the estimator's first operation, into its route record. An unknown value, and an
explicit `device="gpu"` on a CPU-only installation, are refused there in words. `get_params`, `set_params`, clone
and pickle carry it. On a GPU installation `device="cpu"` runs the CPU binaries beside the GPU ones in one process
(`vendor_used() == "cpu"`); in `identical` mode it returns the bits the GPU returns. These tests run on a GPU
installation and on a CPU-only one; what needs two devices is skipped by name on the CPU-only one.
"""
import pickle
import sys

import numpy as np
import pytest

import mojolearn
from mojolearn import _check_both as cb
from mojolearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor

CLASSES = [GaussianProcessRegressor, GaussianProcessClassifier]
TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def data(cls, rows=48, cols=3, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    target = x @ np.arange(1, cols + 1, dtype=np.float32)
    if cls is GaussianProcessClassifier:
        return x, (target > np.median(target)).astype(np.int64)
    return x, target.astype(np.float32)


def answer(model, x):
    return np.asarray(model.predict(x))


@pytest.mark.parametrize("cls", CLASSES)
def test_device_is_a_constructor_parameter_with_default_none(cls):
    assert cls().get_params()["device"] is None
    assert cls(device=None).get_params()["device"] is None
    assert cls(device="cpu").get_params()["device"] == "cpu"
    assert "device" in cls._get_param_names()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_by_the_first_operation(cls, bad):
    x, y = data(cls)
    model = cls(device=bad)                      # stored as given, like every parameter of these classes
    assert model.get_params()["device"] is bad
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        model.fit(x, y)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        model.vendor_used()


@pytest.mark.parametrize("cls", CLASSES)
def test_default_device_is_this_installations_device(cls):
    x, y = data(cls)
    model = cls().fit(x, y)
    assert model.vendor_used() == mojolearn.vendor()
    assert model.get_params()["device"] is None
    assert answer(model, x).shape == (len(x),)


@pytest.mark.parametrize("cls", CLASSES)
def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(cls):
    x, y = data(cls)
    if TWO_DEVICES:
        model = cls(device="gpu").fit(x, y)
        assert model.vendor_used() == mojolearn.vendor() and model.get_params()["device"] == "gpu"
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        cls(device="gpu").fit(x, y)
    model = cls().fit(x, y)
    model.set_params(device="gpu")
    with pytest.raises(ValueError, match="Use device='cpu'"):
        model.fit(x, y)


@pytest.mark.parametrize("cls", CLASSES)
def test_a_refused_device_leaves_the_fitted_model_answering_after_it_is_set_back(cls):
    x, y = data(cls)
    model = cls().fit(x, y)
    before = answer(model, x).copy()
    model.device = "tpu"
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got 'tpu'"):
        model.predict(x)
    model.device = None
    np.testing.assert_array_equal(answer(model, x), before)


@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_runs_the_cpu_binary(cls):
    x, y = data(cls)
    model = cls(device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"
    assert answer(model, x).shape == (len(x),)
    assert isinstance(model.score(x, y), float)


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_default_device_runs_the_gpu_binary_beside_a_cpu_estimator_in_one_process(cls):
    x, y = data(cls)
    cpu = cls(device="cpu").fit(x, y)
    gpu = cls().fit(x, y)
    assert gpu.vendor_used() == mojolearn.vendor() != "cpu"
    assert cpu.vendor_used() == "cpu"
    for _ in range(2):                           # each keeps its own route across interleaved calls
        assert answer(gpu, x).shape == answer(cpu, x).shape
        assert gpu.vendor_used() != cpu.vendor_used()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_does_nothing_on_the_gpu(cls):
    x, y = data(cls)
    model = cls(device="cpu").fit(x, y)
    model.predict(x)
    model.score(x, y)
    if not cb.activity_begin(mojolearn):
        pytest.skip("this installation has no activity record")
    cls(device="cpu").fit(x, y).predict(x)
    model.predict(x)
    model.score(x, y)
    activity = cb.activity_end(mojolearn)
    assert activity is not None
    assert activity["launches"] == 0, activity["launch_names"]
    assert activity["upload_calls"] == 0 and activity["download_calls"] == 0


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_identical_mode_cpu_device_returns_the_gpu_bits(cls):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x, y = data(cls)
    gpu = cls(numeric_mode="identical").fit(x, y)
    cpu = cls(numeric_mode="identical", device="cpu").fit(x, y)
    assert answer(gpu, x).tobytes() == answer(cpu, x).tobytes()
    assert float(gpu.score(x, y)).hex() == float(cpu.score(x, y)).hex()
    if cls is GaussianProcessClassifier:
        assert np.asarray(gpu.predict_proba(x)).tobytes() == np.asarray(cpu.predict_proba(x)).tobytes()
    else:
        assert np.asarray(gpu.alpha_).tobytes() == np.asarray(cpu.alpha_).tobytes()
        gpu_mean, gpu_std = gpu.predict(x, return_std=True)
        cpu_mean, cpu_std = cpu.predict(x, return_std=True)
        assert np.asarray(gpu_std).tobytes() == np.asarray(cpu_std).tobytes()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_never_runs_a_fast_binary(cls):
    x, y = data(cls)
    with pytest.raises(ValueError, match="device='cpu' runs in identical mode only"):
        cls(device="cpu", numeric_mode="fast").fit(x, y)


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_set_params_moves_the_estimator_to_the_other_device(cls):
    x, y = data(cls)
    model = cls().fit(x, y)
    assert model.set_params(device="cpu") is model
    assert model.get_params()["device"] == "cpu"
    model.fit(x, y)
    assert model.vendor_used() == "cpu"
    model.set_params(device=None).fit(x, y)
    assert model.get_params()["device"] is None
    assert model.vendor_used() == mojolearn.vendor()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_an_assigned_device_is_seen_by_the_next_operation(cls):
    x, y = data(cls)
    model = cls().fit(x, y)
    assert model.vendor_used() == mojolearn.vendor()
    model.device = "cpu"
    assert model.vendor_used() == "cpu"
    model.device = None
    assert model.vendor_used() == mojolearn.vendor()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_clone_and_pickle_carry_the_device(cls, device):
    if device == "gpu" and not TWO_DEVICES:
        pytest.skip("an explicit device='gpu' is refused on a CPU-only installation")
    x, y = data(cls)
    model = cls(device=device).fit(x, y)
    expected = answer(model, x).copy()
    used = model.vendor_used()

    rebuilt = cls(**model.get_params())
    assert rebuilt.get_params()["device"] == device          # None stays None
    assert rebuilt.fit(x, y).vendor_used() == used

    try:
        from sklearn.base import clone
    except ImportError:
        clone = None
    if clone is not None:
        cloned = clone(model)
        assert cloned.get_params()["device"] == device
        assert cloned.fit(x, y).vendor_used() == used

    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params()["device"] == device
    np.testing.assert_array_equal(answer(restored, x), expected)
    assert restored.vendor_used() == used


@pytest.mark.parametrize("cls", CLASSES)
def test_an_estimator_pickled_without_the_field_runs_this_installations_device(cls):
    x, y = data(cls)
    model = cls().fit(x, y)
    expected = answer(model, x).copy()
    state = dict(model.__dict__)
    del state["device"]                          # what an older version pickled
    older = cls.__new__(cls)
    older.__dict__.update(state)
    np.testing.assert_array_equal(answer(older, x), expected)
    assert older.vendor_used() == mojolearn.vendor()


@pytest.mark.parametrize("cls", CLASSES)
def test_a_changed_numeric_mode_resolves_the_route_again(cls):
    x, y = data(cls)
    model = cls(device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"
    model.numeric_mode = "identical"
    assert model.numeric_mode_used() == "identical"
    assert model.vendor_used() == "cpu"
    answer(model, x)


@pytest.mark.parametrize("cls", CLASSES)
def test_check_runs_both_devices_in_this_process(cls, capsys):
    x, y = data(cls)
    model = cls().fit(x, y)
    result = model.check(x)
    capsys.readouterr()
    if not TWO_DEVICES or mojolearn.numeric_mode() != "identical":
        assert not result["ran"]
        return
    assert result["ran"] and result["transfer"] == "device"
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["route"] == dict(device=None, vendor_used=mojolearn.vendor())
    assert cpu["route"] == dict(device="cpu", vendor_used="cpu")
    assert result["verdict"]["match"]
    assert gpu["outputs"][0]["sha256"] == cpu["outputs"][0]["sha256"]
    # the given estimator is untouched
    assert model.get_params()["device"] is None and model.vendor_used() == mojolearn.vendor()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_the_check_takes_the_device_form_not_a_saved_file(cls):
    model = cls()
    assert cb.takes_device(model)
    twin = cb.on_device(model, "cpu", fitted=False)
    assert type(twin) is cls and twin is not model and twin.get_params()["device"] == "cpu"
    assert cb.on_device(model, "gpu", fitted=False) is model


def _python_events(call):
    """Python-level events while `call` runs: every Python function call and every C function called from a
    Python frame."""
    counts = {"call": 0, "c_call": 0}

    def profile(frame, event, arg):
        if event in counts:
            counts[event] += 1

    sys.setprofile(profile)
    try:
        call()
    finally:
        sys.setprofile(None)
    return counts


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("device", [None, "cpu"])
def test_python_operations_per_call_do_not_grow_with_rows(cls, device):
    counts = {}
    for name, rows in (("small", 24), ("large", 96)):
        x, y = data(cls, rows=rows)
        model = cls(device=device).fit(x, y)
        model.predict(x)                         # the route and the fitted-state record are resolved
        counts[name] = _python_events(lambda: model.predict(x))
    assert counts["small"] == counts["large"]
    assert counts["small"]["call"] <= 2          # the public method and nothing under it
