# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on Sequential, at the Python boundary.

The device is a constructor parameter (and a parameter of `from_checkpoint` / `load`). None is this installation's
device. It is checked and refused where it is given and read once, by the model's first operation, into the shared
route owner (no call reads it again); pickle and copy carry it. An explicit `device="gpu"` on a CPU-only installation is refused in words that name
`device="cpu"`; `device="cpu"` always runs, in identical mode. On a GPU installation both entry tables exist in one
process; in identical mode the two devices return the same bits. Which binary ran is read from the verification
sidecar a saved model writes (its `install.vendor` is the word the binary itself reports). What needs two devices is
skipped by name on a CPU-only installation.
"""
import copy
import json
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn import Sequential

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def data(rows=96, cols=5, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    y = (x @ np.arange(1, cols + 1, dtype=np.float32)).astype(np.float32)
    return x, y


def build(device=None, numeric_mode="identical"):
    model = Sequential(numeric_mode, device=device).linear(8).tanh().linear(1)
    return model.compile(loss="mse", optimizer="adam", lr=0.01)


def fitted(device=None, numeric_mode="identical"):
    x, y = data()
    return build(device, numeric_mode).fit(x, y, epochs=3, batch_size=32, seed=0), x


def vendor_ran(model, tmp_path, name):
    """The vendor word of the binary that saves and runs this model."""
    path = tmp_path / (name + ".ckpt")
    model.save_checkpoint(str(path))
    with open(str(path) + ".verify.json") as stream:
        return json.load(stream)["install"]["vendor"]


def test_device_is_a_constructor_parameter_with_default_none():
    assert Sequential().device is None
    assert Sequential(None, input_shape=None, device=None).device is None
    assert Sequential(device="cpu").device == "cpu"


@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        Sequential(device=bad)


def test_a_refused_constructor_leaves_an_existing_model_unchanged():
    model = Sequential("identical", device="cpu").linear(3)
    with pytest.raises(ValueError, match="device must be"):
        model.__init__(device="tpu")
    assert model.device == "cpu" and model.numeric_mode == "identical" and len(model.layers) == 1   # `layers` is a property


def test_default_device_is_this_installations_device(tmp_path):
    model, x = fitted()
    assert model.device is None
    assert vendor_ran(model, tmp_path, "default") == mojolearn.vendor()
    assert np.asarray(model.predict(x)).shape == (len(x),)


def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(tmp_path):
    if TWO_DEVICES:
        model, _ = fitted("gpu")
        assert model.device == "gpu" and vendor_ran(model, tmp_path, "gpu") == mojolearn.vendor()
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        Sequential(device="gpu")
    # a model that carries 'gpu' without its constructor's check is refused by its first operation
    carried = build()
    carried.device = "gpu"
    x, y = data()
    with pytest.raises(ValueError, match="Use device='cpu'"):
        carried.fit(x, y, epochs=1)


def test_cpu_device_runs_the_cpu_binary(tmp_path):
    model, x = fitted("cpu")
    assert vendor_ran(model, tmp_path, "cpu") == "cpu"
    assert np.all(np.isfinite(np.asarray(model.predict(x))))


@two_devices
def test_identical_mode_cpu_device_returns_the_gpu_bits():
    gpu, x = fitted(None)
    cpu, _ = fitted("cpu")
    assert np.asarray(gpu.params_).tobytes() == np.asarray(cpu.params_).tobytes()
    assert np.asarray(gpu.predict(x)).tobytes() == np.asarray(cpu.predict(x)).tobytes()
    assert list(gpu.loss_curve_) == list(cpu.loss_curve_)


@two_devices
def test_two_models_on_two_devices_interleave_in_one_process(tmp_path):
    gpu, x = fitted(None)
    cpu, _ = fitted("cpu")
    first = np.asarray(gpu.predict(x)).tobytes()
    second = np.asarray(cpu.predict(x)).tobytes()
    third = np.asarray(gpu.predict(x)).tobytes()
    assert first == second == third
    assert vendor_ran(gpu, tmp_path, "g") != "cpu" and vendor_ran(cpu, tmp_path, "c") == "cpu"


@two_devices
def test_cpu_device_never_runs_a_fast_binary():
    if mojolearn.vendor() != "metal":
        pytest.skip("only an Apple installation carries a fast set")
    x, y = data()
    with pytest.raises(ValueError, match="device='cpu' runs in identical mode only"):
        build("cpu", "fast").fit(x, y, epochs=1)


@pytest.mark.parametrize("device", [None, "cpu"])
def test_pickle_and_deepcopy_carry_the_device(device, tmp_path):
    model, x = fitted(device)
    wanted = np.asarray(model.predict(x)).tobytes()
    for twin in (pickle.loads(pickle.dumps(model)), copy.deepcopy(model)):
        assert twin.device == device
        assert np.asarray(twin.predict(x)).tobytes() == wanted
        assert vendor_ran(twin, tmp_path, "twin") == ("cpu" if device == "cpu" else mojolearn.vendor())


def test_a_changed_numeric_mode_resolves_the_route_again():
    model, x = fitted("cpu")
    model.numeric_mode = "nonsense"
    with pytest.raises(ValueError, match="numeric_mode must be fast, identical or None"):
        model.predict(x)
    model.numeric_mode = "identical"
    assert np.asarray(model.predict(x)).shape == (len(x),)


@pytest.mark.parametrize("device", [None, "cpu"])
def test_a_checkpoint_loads_on_the_device_it_is_given(device, tmp_path):
    model, x = fitted(None)
    path = str(tmp_path / "model.ckpt")
    model.save_checkpoint(path)
    wanted = np.asarray(model.predict(x)).tobytes()
    for loader in (Sequential.from_checkpoint, Sequential.load):
        loaded = loader(path, "identical", device=device)
        assert loaded.device == device
        assert np.asarray(loaded.predict(x)).tobytes() == wanted
        assert vendor_ran(loaded, tmp_path, "loaded") == ("cpu" if device == "cpu" else mojolearn.vendor())
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        Sequential.load(path, device="tpu")


@pytest.mark.parametrize("device", [None, "cpu"])
def test_python_operations_per_predict_do_not_grow_with_rows(device):
    import sys
    model, _ = fitted(device)
    counts = []
    for rows in (64, 8192):
        x, _ = data(rows=rows)
        model.predict(x)                                    # the route and the admission are resolved
        events = [0]

        def count(frame, event, arg):
            events[0] += 1
            return None

        sys.setprofile(count)
        try:
            model.predict(x)
        finally:
            sys.setprofile(None)
        counts.append(events[0])
    assert counts[0] == counts[1]


def test_vendor_used_answers_for_the_models_own_device():
    gpu, _ = fitted(None)
    assert gpu.vendor_used() == mojolearn.vendor()
    if TWO_DEVICES:
        cpu, _ = fitted("cpu")
        assert cpu.vendor_used() == "cpu"
        # the first model still answers for its own device
        assert gpu.vendor_used() == mojolearn.vendor()


def test_check_runs_both_devices_in_this_process(capsys):
    model, x = fitted(None)
    before = np.asarray(model.params_).tobytes()
    result = model.check(x)
    capsys.readouterr()
    if not TWO_DEVICES:
        assert not result["ran"]
        return
    assert result["ran"] and result["transfer"] == "device"
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["route"]["vendor_used"] == mojolearn.vendor()
    assert cpu["route"]["vendor_used"] == "cpu"
    assert result["verdict"]["match"]
    assert gpu["outputs"][0]["sha256"] == cpu["outputs"][0]["sha256"]
    # the given model is untouched: same device, same binary, same parameters
    assert model.device is None and model.vendor_used() == mojolearn.vendor()
    assert np.asarray(model.params_).tobytes() == before


@two_devices
def test_the_check_twin_is_another_model_on_the_other_device_with_the_same_fitted_state():
    from mojolearn._check_both import on_device, takes_device
    model, x = fitted(None)
    assert takes_device(model)
    assert on_device(model, "gpu") is model
    twin = on_device(model, "cpu")
    assert type(twin) is Sequential and twin is not model
    assert twin.device == "cpu" and twin.vendor_used() == "cpu" and model.vendor_used() == mojolearn.vendor()
    assert twin._layers == model._layers and twin._layers is not model._layers
    assert np.asarray(twin.params_).tobytes() == np.asarray(model.params_).tobytes()
    assert np.asarray(twin.predict(x)).tobytes() == np.asarray(model.predict(x)).tobytes()
    # a layer added to the twin never reaches the checked model
    count = len(model._layers)
    twin.relu()
    assert len(model._layers) == count and len(twin._layers) == count + 1
    # an unfitted twin carries the description and no fitted state
    bare = on_device(model, "cpu", fitted=False)
    assert bare._layers == model._layers and getattr(bare, "params_", None) is None
