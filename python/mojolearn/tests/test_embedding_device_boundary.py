# SPDX-License-Identifier: Apache-2.0
"""The Embedding layer on both devices: the operation-level rows of `device=`.

`device=None | "gpu" | "cpu"` is the last constructor parameter (`forward(ids, device=...)` is another thing: where the
output lives). On a GPU installation a `device="cpu"` layer runs the CPU binary beside the GPU one in one process:
nothing is launched or uploaded, its forward and backward return the bits the GPU layer returns in `identical` mode,
and it has no device I/O (`forward(device=True)` and `to_device` are refused in words). What needs two devices is
skipped by name on a CPU-only installation.
"""
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn import _check_both as cb
from mojolearn import Embedding

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def table(rows=50, width=6, seed=2):
    return np.random.default_rng(seed).random((rows, width), dtype=np.float32)


def ids():
    return np.array([[1, 7, 7, 3], [49, 0, 3, 1]], dtype=np.int64)


def test_device_is_the_last_constructor_parameter_with_default_none():
    layer = Embedding.from_pretrained(table())
    assert layer.get_params()["device"] is None
    assert Embedding.from_pretrained(table(), device="cpu").get_params()["device"] == "cpu"
    assert Embedding(50, 6, weight=table(), device="cpu").get_params()["device"] == "cpu"


@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction_and_set_params(bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        Embedding.from_pretrained(table(), device=bad)
    layer = Embedding.from_pretrained(table())
    before = np.asarray(layer.forward(ids())).copy()
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        layer.set_params(device=bad)
    assert layer.get_params()["device"] is None
    np.testing.assert_array_equal(np.asarray(layer.forward(ids())), before)


def test_cpu_device_runs_the_cpu_binary():
    layer = Embedding.from_pretrained(table(), device="cpu")
    out = np.asarray(layer.forward(ids()))
    assert out.shape == (2, 4, 6)
    np.testing.assert_array_equal(out, table()[ids()])
    assert layer.vendor_used() == "cpu"
    grad = np.asarray(layer.backward(ids(), np.ones((2, 4, 6), dtype=np.float32)))
    assert grad.shape == (50, 6)


@two_devices
def test_cpu_device_does_nothing_on_the_gpu():
    layer = Embedding.from_pretrained(table(), device="cpu")
    layer.forward(ids())
    if not cb.activity_begin(mojolearn):
        pytest.skip("this installation has no activity record")
    layer.forward(ids())
    layer.backward(ids(), np.ones((2, 4, 6), dtype=np.float32))
    activity = cb.activity_end(mojolearn)
    assert activity is not None
    assert activity["launches"] == 0, activity["launch_names"]
    assert activity["upload_calls"] == 0 and activity["download_calls"] == 0


@two_devices
def test_identical_mode_cpu_device_returns_the_gpu_bits():
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    gpu = Embedding.from_pretrained(table(), numeric_mode="identical")
    cpu = Embedding.from_pretrained(table(), numeric_mode="identical", device="cpu")
    assert gpu.vendor_used() == mojolearn.vendor() and cpu.vendor_used() == "cpu"
    dy = np.random.default_rng(4).random((2, 4, 6), dtype=np.float32)
    assert np.asarray(gpu.forward(ids())).tobytes() == np.asarray(cpu.forward(ids())).tobytes()
    assert np.asarray(gpu.backward(ids(), dy)).tobytes() == np.asarray(cpu.backward(ids(), dy)).tobytes()


@two_devices
def test_cpu_device_has_no_device_io_and_never_runs_fast():
    layer = Embedding.from_pretrained(table(), device="cpu")
    with pytest.raises(RuntimeError, match="no device I/O entries"):
        layer.forward(ids(), device=True)
    with pytest.raises(RuntimeError, match="no device I/O entries"):
        layer.to_device(np.ones((2, 4, 6), dtype=np.float32))
    fast = Embedding.from_pretrained(table(), device="cpu", numeric_mode="fast")
    with pytest.raises(ValueError, match="identical"):
        fast.forward(ids())


@two_devices
def test_set_params_moves_the_layer_and_releases_its_device_table():
    layer = Embedding.from_pretrained(table())
    expected = np.asarray(layer.forward(ids())).copy()
    assert layer.set_params(device="cpu") is layer
    assert layer.vendor_used() == "cpu"
    np.testing.assert_array_equal(np.asarray(layer.forward(ids())), expected)
    layer.set_params(device=None)
    assert layer.vendor_used() == mojolearn.vendor()
    np.testing.assert_array_equal(np.asarray(layer.forward(ids())), expected)


@pytest.mark.parametrize("device", [None, "cpu"])
def test_pickle_and_saved_layers(device, tmp_path):
    layer = Embedding.from_pretrained(table(), device=device)
    expected = np.asarray(layer.forward(ids())).copy()
    restored = pickle.loads(pickle.dumps(layer))
    assert restored.get_params()["device"] == device
    np.testing.assert_array_equal(np.asarray(restored.forward(ids())), expected)
    path = str(tmp_path / "layer.npz")
    layer.save(path)
    loaded = Embedding.load(path)
    assert loaded.get_params()["device"] is None          # a constructor parameter, not saved state
    np.testing.assert_array_equal(np.asarray(loaded.forward(ids())), expected)
