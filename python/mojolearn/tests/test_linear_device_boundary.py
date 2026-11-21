# SPDX-License-Identifier: Apache-2.0
"""`device="gpu"` (default) and `device="cpu"` on Ridge and LinearRegression, at the Python boundary.

The device is a constructor parameter. It is checked when it is given and read once, by the estimator's first
operation, into its native route owner; `get_params`, `set_params`, clone and pickle carry it. On a GPU installation
both entry tables exist in one process and `device="cpu"` runs the CPU binaries (`vendor_used() == "cpu"`); in
`identical` mode it returns the bits the GPU returns. These tests run on a GPU installation and on a CPU-only one;
what needs two devices is skipped by name on the CPU-only one.
"""
import pickle
import sys

import numpy as np
import pytest

import mojolearn
from mojolearn import LinearRegression, Ridge

CLASSES = [Ridge, LinearRegression]
TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def data(rows=200, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    y = (x @ np.arange(1, cols + 1, dtype=np.float32) + np.float32(0.5)).astype(np.float32)
    return x, y


@pytest.mark.parametrize("cls", CLASSES)
def test_device_is_a_constructor_parameter_with_default_gpu(cls):
    assert cls().get_params()["device"] == "gpu"
    assert cls(device="cpu").get_params()["device"] == "cpu"
    assert "device" in cls._get_param_names()
    assert sorted(cls._get_param_names()) == list(cls._get_param_names())


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", None, 0, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(cls, bad):
    with pytest.raises(ValueError, match="device must be 'gpu' or 'cpu', got"):
        cls(device=bad)


@pytest.mark.parametrize("cls", CLASSES)
def test_set_params_refuses_an_unknown_device_and_leaves_the_estimator_unchanged(cls):
    x, y = data()
    model = cls().fit(x, y)
    before = np.asarray(model.predict(x)).copy()
    with pytest.raises(ValueError, match="device must be 'gpu' or 'cpu', got 'tpu'"):
        model.set_params(device="tpu")
    assert model.get_params()["device"] == "gpu"
    np.testing.assert_array_equal(np.asarray(model.predict(x)), before)


@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_runs_the_cpu_binary(cls):
    x, y = data()
    model = cls(device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"
    assert model.numeric_mode_used() == "identical" or not TWO_DEVICES
    assert np.asarray(model.predict(x)).shape == (len(x),)
    assert isinstance(model.score(x, y), float)


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_default_device_runs_the_gpu_binary_beside_a_cpu_estimator_in_one_process(cls):
    x, y = data()
    cpu = cls(device="cpu").fit(x, y)
    gpu = cls().fit(x, y)
    assert gpu.vendor_used() == mojolearn.vendor() != "cpu"
    assert cpu.vendor_used() == "cpu"
    # each keeps its own route across interleaved calls
    for _ in range(2):
        assert np.asarray(gpu.predict(x)).shape == np.asarray(cpu.predict(x)).shape
        assert gpu.vendor_used() != cpu.vendor_used()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("weighted", [False, True])
def test_identical_mode_cpu_device_returns_the_gpu_bits(cls, weighted):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x, y = data(rows=400, cols=8)
    weight = np.linspace(0.5, 2.0, len(x), dtype=np.float32) if weighted else None
    gpu = cls(numeric_mode="identical").fit(x, y, weight)
    cpu = cls(numeric_mode="identical", device="cpu").fit(x, y, weight)
    assert np.asarray(gpu.coef_).tobytes() == np.asarray(cpu.coef_).tobytes()
    assert float(gpu.intercept_) == float(cpu.intercept_)
    assert np.asarray(gpu.predict(x)).tobytes() == np.asarray(cpu.predict(x)).tobytes()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_never_runs_a_fast_binary(cls):
    x, y = data()
    model = cls(device="cpu", numeric_mode="fast")
    try:
        model.fit(x, y)
    except ValueError as refusal:
        # where `fast` names its own binaries (Apple): refused in words
        assert "device='cpu' runs in identical mode only" in str(refusal)
    except ImportError:
        # where no fast binaries are installed, the mode's own refusal stands
        pass
    else:
        # where the identical set answers `fast` for this family
        assert model.numeric_mode_used() == "identical" and model.vendor_used() == "cpu"


@pytest.mark.parametrize("cls", CLASSES)
def test_set_params_moves_the_estimator_to_the_other_device(cls):
    x, y = data()
    model = cls().fit(x, y)
    assert model.set_params(device="cpu") is model
    assert model.get_params()["device"] == "cpu"
    model.fit(x, y)
    assert model.vendor_used() == "cpu"
    model.set_params(device="gpu").fit(x, y)
    assert model.vendor_used() == mojolearn.vendor()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_clone_and_pickle_carry_the_device(cls, device):
    x, y = data()
    model = cls(device=device).fit(x, y)
    expected = np.asarray(model.predict(x)).copy()
    used = model.vendor_used()

    rebuilt = cls(**model.get_params())
    assert rebuilt.get_params() == model.get_params()
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
    # the unpickled estimator was built without its constructor: its first
    # operation reads the device from its restored parameters
    np.testing.assert_array_equal(np.asarray(restored.predict(x)), expected)
    assert restored.vendor_used() == used


@pytest.mark.parametrize("cls", CLASSES)
def test_a_changed_numeric_mode_resolves_the_route_again(cls):
    x, y = data()
    model = cls(device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"
    model.numeric_mode = "identical"
    assert model.numeric_mode_used() == "identical"
    assert model.vendor_used() == "cpu"
    np.asarray(model.predict(x))


@pytest.mark.parametrize("cls", CLASSES)
def test_a_saved_model_loads_on_the_default_device(cls, tmp_path):
    x, y = data()
    model = cls(device="cpu").fit(x, y)
    path = str(tmp_path / "model.npz")
    model.save(path)
    loaded = cls.load(path)
    assert loaded.get_params()["device"] == "gpu"
    assert np.asarray(loaded.predict(x)).shape == (len(x),)
    host = mojolearn.host_model(path)
    assert host.vendor_used() == "cpu"


@pytest.mark.parametrize("cls", CLASSES)
def test_check_runs_both_devices_in_this_process(cls, capsys):
    x, y = data()
    model = cls().fit(x, y)
    result = model.check(x)
    printed = capsys.readouterr().out
    if not TWO_DEVICES or mojolearn.numeric_mode() != "identical":
        assert not result["ran"]
        return
    assert result["ran"] and result["transfer"] == "device"
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["route"] == dict(device="gpu", vendor_used=mojolearn.vendor())
    assert cpu["route"] == dict(device="cpu", vendor_used="cpu")
    assert result["verdict"]["match"]
    assert gpu["outputs"][0]["sha256"] == cpu["outputs"][0]["sha256"]
    assert "device='gpu' in this process" in printed and "device='cpu' in this process" in printed
    # the given estimator is untouched
    assert model.get_params()["device"] == "gpu" and model.vendor_used() == mojolearn.vendor()


def _python_events(call):
    """Python-level events while `call` runs: every Python function call and
    every C function called from a Python frame."""
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
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_python_operations_per_call_do_not_grow_with_rows(cls, device):
    small_x, small_y = data(rows=64)
    large_x, large_y = data(rows=8192)
    counts = {}
    for name, (x, y) in (("small", (small_x, small_y)), ("large", (large_x, large_y))):
        model = cls(device=device)
        model.fit(x, y)                    # the first operation resolves the route
        counts[name] = (_python_events(lambda: model.fit(x, y)),
                        _python_events(lambda: model.predict(x)),
                        _python_events(lambda: model.score(x, y)))
    assert counts["small"] == counts["large"]
    # one public method is one Python frame and one call into the binding
    # (plus the measuring lambda's own frame)
    for per_operation in counts["small"]:
        assert per_operation["call"] <= 2
