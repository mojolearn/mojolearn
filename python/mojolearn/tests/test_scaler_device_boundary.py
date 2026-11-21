# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on the four scalers, at the Python boundary.

StandardScaler, MinMaxScaler, MaxAbsScaler and RobustScaler take the device as a constructor parameter, with the rule
and the words of Ridge and LinearRegression (test_linear_device_boundary.py). None is this installation's device: the
GPU where the installation has a GPU set, else the CPU. `get_params` returns what was passed (None stays None). An
explicit `device="gpu"` on a CPU-only installation is refused in words that name `device="cpu"`; `device="cpu"` always
runs. It is checked when it is given and read once, by the scaler's first data operation, into its native route owner;
`get_params`, `set_params`, clone and pickle carry it. On a GPU installation both connected contexts exist in one
process and `device="cpu"` runs the CPU binaries (`vendor_used() == "cpu"`); in `identical` mode it returns the bits
the GPU returns. These tests run on a GPU installation and on a CPU-only one; what needs two devices is skipped by
name on the CPU-only one.
"""
import pickle
import sys

import numpy as np
import pytest

import mojolearn
from mojolearn import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

CLASSES = [StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler]
#: `set_params` of these two rebuilds the estimator (it is unfitted afterwards); the other two keep their fit.
REBUILT_BY_SET_PARAMS = (StandardScaler, MinMaxScaler)
SAVED = [StandardScaler, MinMaxScaler]
TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def data(rows=200, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, cols)) * np.arange(1, cols + 1)).astype(np.float32)


def fitted_arrays(model):
    """The fitted statistics of a scaler, as bytes by name."""
    found = {}
    for name in ("mean_", "var_", "scale_", "min_", "data_min_", "data_max_", "data_range_", "max_abs_", "center_"):
        value = getattr(model, name, None)
        if value is not None:
            found[name] = np.asarray(value).tobytes()
    return found


@pytest.mark.parametrize("cls", CLASSES)
def test_device_is_a_constructor_parameter_with_default_none(cls):
    assert cls().get_params()["device"] is None
    assert cls(device=None).get_params()["device"] is None
    assert cls(device="cpu").get_params()["device"] == "cpu"
    assert cls(device="cpu").get_params(deep=False)["device"] == "cpu"


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(cls, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        cls(device=bad)


@pytest.mark.parametrize("cls", CLASSES)
def test_default_device_is_this_installations_device(cls):
    x = data()
    model = cls().fit(x)
    assert model.vendor_used() == mojolearn.vendor()          # the GPU set where there is one, else the CPU
    assert model.get_params()["device"] is None               # what was passed, not what it resolved to
    assert np.asarray(model.transform(x)).shape == x.shape


@pytest.mark.parametrize("cls", CLASSES)
def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(cls):
    x = data()
    if TWO_DEVICES:
        model = cls(device="gpu").fit(x)
        assert model.vendor_used() == mojolearn.vendor() and model.get_params()["device"] == "gpu"
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        cls(device="gpu")
    model = cls().fit(x)
    with pytest.raises(ValueError, match="Use device='cpu'"):
        model.set_params(device="gpu")
    assert model.get_params()["device"] is None
    # an instance that carries 'gpu' without its constructor's check is refused by its first data operation
    carried = cls()
    carried.device = "gpu"
    with pytest.raises(ValueError, match="Use device='cpu'"):
        carried.fit(x)


@pytest.mark.parametrize("cls", CLASSES)
def test_set_params_refuses_an_unknown_device_and_leaves_the_estimator_unchanged(cls):
    x = data()
    model = cls().fit(x)
    before = np.asarray(model.transform(x)).copy()
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got 'tpu'"):
        model.set_params(device="tpu")
    assert model.get_params()["device"] is None
    np.testing.assert_array_equal(np.asarray(model.transform(x)), before)


@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_runs_the_cpu_binary(cls):
    x = data()
    model = cls(device="cpu").fit(x)
    assert model.vendor_used() == "cpu"
    assert np.asarray(model.transform(x)).shape == x.shape
    assert np.asarray(cls(device="cpu").fit_transform(x)).shape == x.shape


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_default_device_runs_the_gpu_binary_beside_a_cpu_estimator_in_one_process(cls):
    x = data()
    cpu = cls(device="cpu").fit(x)
    gpu = cls().fit(x)
    assert gpu.vendor_used() == mojolearn.vendor() != "cpu"
    assert cpu.vendor_used() == "cpu"
    # each keeps its own route across interleaved calls
    for _ in range(2):
        assert np.asarray(gpu.transform(x)).shape == np.asarray(cpu.transform(x)).shape
        assert gpu.vendor_used() != cpu.vendor_used()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("with_nan", [False, True])
def test_identical_mode_cpu_device_returns_the_gpu_bits(cls, with_nan):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x = data(rows=400, cols=8)
    if with_nan:
        x[::7, 2] = np.nan
    gpu = cls(numeric_mode="identical").fit(x)
    cpu = cls(numeric_mode="identical", device="cpu").fit(x)
    assert fitted_arrays(gpu) == fitted_arrays(cpu) and fitted_arrays(gpu)
    assert np.asarray(gpu.transform(x)).tobytes() == np.asarray(cpu.transform(x)).tobytes()
    assert (np.asarray(cls(numeric_mode="identical").fit_transform(x)).tobytes()
            == np.asarray(cls(numeric_mode="identical", device="cpu").fit_transform(x)).tobytes())


@two_devices
@pytest.mark.parametrize("cls", [StandardScaler, MinMaxScaler])
def test_identical_mode_partial_fit_on_the_cpu_device_returns_the_gpu_bits(cls):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x = data(rows=300, cols=5)
    gpu = cls(numeric_mode="identical")
    cpu = cls(numeric_mode="identical", device="cpu")
    for start in (0, 100, 200):
        gpu.partial_fit(x[start:start + 100])
        cpu.partial_fit(x[start:start + 100])
    assert cpu.vendor_used() == "cpu" != gpu.vendor_used()
    assert fitted_arrays(gpu) == fitted_arrays(cpu)


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_never_runs_a_fast_binary(cls):
    x = data()
    model = cls(device="cpu", numeric_mode="fast")
    try:
        model.fit(x)
    except (ValueError, ImportError) as refusal:
        # either the CPU context's own words, or the mode's own refusal where no fast binaries are installed
        assert "device='cpu' runs in identical mode only" in str(refusal) or "fast" in str(refusal)
    else:
        # where the identical set answers `fast` for this family
        assert model.vendor_used() == "cpu"


@pytest.mark.parametrize("cls", CLASSES)
def test_set_params_moves_the_estimator_to_the_other_device(cls):
    x = data()
    model = cls().fit(x)
    assert model.set_params(device="cpu") is model
    assert model.get_params()["device"] == "cpu"
    model.fit(x)
    assert model.vendor_used() == "cpu"
    model.set_params(device=None).fit(x)
    assert model.get_params()["device"] is None
    assert model.vendor_used() == mojolearn.vendor()


@two_devices
@pytest.mark.parametrize("cls", [MaxAbsScaler, RobustScaler])
def test_a_fitted_scaler_moved_to_the_cpu_transforms_there_with_the_same_bits(cls):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    assert cls not in REBUILT_BY_SET_PARAMS
    x = data()
    model = cls(numeric_mode="identical").fit(x)
    on_gpu = np.asarray(model.transform(x)).tobytes()
    model.set_params(device="cpu")                 # these two keep their fit through set_params
    assert model.vendor_used() == "cpu"
    assert np.asarray(model.transform(x)).tobytes() == on_gpu


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_clone_and_pickle_carry_the_device(cls, device):
    if device == "gpu" and not TWO_DEVICES:
        pytest.skip("an explicit device='gpu' is refused on a CPU-only installation")
    x = data()
    model = cls(device=device).fit(x)
    expected = np.asarray(model.transform(x)).copy()
    used = model.vendor_used()

    rebuilt = cls(**model.get_params())
    assert rebuilt.get_params() == model.get_params()
    assert rebuilt.get_params()["device"] == device          # None stays None
    assert rebuilt.fit(x).vendor_used() == used

    try:
        from sklearn.base import clone
    except ImportError:
        clone = None
    if clone is not None:
        cloned = clone(model)
        assert cloned.get_params()["device"] == device
        assert cloned.fit(x).vendor_used() == used

    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params()["device"] == device
    # the unpickled scaler was built without its constructor: its first data
    # operation reads the device from its restored parameters
    np.testing.assert_array_equal(np.asarray(restored.transform(x)), expected)
    assert restored.vendor_used() == used


@pytest.mark.parametrize("cls", CLASSES)
def test_an_instance_without_the_parameter_runs_this_installations_device(cls):
    # an estimator pickled before `device` existed carries no such field
    x = data()
    model = cls().fit(x)
    del model.__dict__["device"]
    assert model.get_params()["device"] is None
    assert np.asarray(model.transform(x)).shape == x.shape
    assert model.vendor_used() == mojolearn.vendor()


@pytest.mark.parametrize("cls", SAVED)
def test_a_saved_scaler_loads_on_the_default_device(cls, tmp_path):
    x = data()
    model = cls(device="cpu").fit(x)
    path = str(tmp_path / "scaler.npz")
    model.save(path)
    loaded = cls.load(path)
    assert loaded.get_params()["device"] is None
    assert loaded.vendor_used() == mojolearn.vendor()
    if mojolearn.numeric_mode() == "identical":
        # fitted on the CPU, transformed on this installation's device: the same bits
        assert np.asarray(loaded.transform(x)).tobytes() == np.asarray(model.transform(x)).tobytes()
    host = mojolearn.host_model(path)
    assert np.asarray(host.transform(x)).shape == x.shape


@pytest.mark.parametrize("cls", CLASSES)
def test_check_runs_both_devices_in_this_process(cls, capsys):
    x = data()
    model = cls().fit(x)
    result = model.check(x)
    printed = capsys.readouterr().out
    if not TWO_DEVICES or mojolearn.numeric_mode() != "identical":
        assert not result["ran"]
        return
    assert result["ran"] and result["transfer"] == "device"
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["route"] == dict(device=None, vendor_used=mojolearn.vendor())
    assert cpu["route"] == dict(device="cpu", vendor_used="cpu")
    assert result["verdict"]["match"]
    assert gpu["outputs"][0]["sha256"] == cpu["outputs"][0]["sha256"]
    assert "device=None (this installation's device) in this process" in printed
    assert "device='cpu' in this process" in printed
    # the given estimator is untouched
    assert model.get_params()["device"] is None and model.vendor_used() == mojolearn.vendor()


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
@pytest.mark.parametrize("device", [None, "cpu"])
def test_python_operations_per_call_do_not_grow_with_rows_or_with_the_device(cls, device):
    counts = {}
    for name, rows in (("small", 64), ("large", 8192)):
        x = data(rows=rows)
        model = cls(device=device)
        model.fit(x)                       # the first data operation resolves the route
        counts[name] = (_python_events(lambda: model.fit(x)), _python_events(lambda: model.transform(x)))
    assert counts["small"] == counts["large"]
    if device is None:
        return
    # the CPU route costs the same Python events per call as the default route
    x = data(rows=64)
    default = cls()
    default.fit(x)
    assert (_python_events(lambda: default.fit(x)), _python_events(lambda: default.transform(x))) == counts["small"]
