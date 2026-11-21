# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on SmallMLPTrainer, at the Python boundary.
Same rule and same words as Ridge and LinearRegression (test_linear_device_boundary.py).

The device is a constructor parameter (and a keyword of `from_checkpoint`), checked where it is given and
read once into the trainer's one native route owner; no operation reads it again. On a GPU installation both
sets of compiled entries exist in one process and `device="cpu"` runs the CPU binary. What needs two devices
is skipped by name on a CPU-only installation.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn._mlp_impl import SmallMLPTrainer

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def weights():
    rng = np.random.default_rng(11)
    return ((rng.standard_normal((16, 8)) * .1).astype(np.float32), np.zeros(16, np.float32),
            (rng.standard_normal((3, 16)) * .1).astype(np.float32), np.zeros(3, np.float32))


def batch(rows=32, seed=5):
    rng = np.random.default_rng(seed)
    return rng.random((rows, 8), dtype=np.float32), rng.integers(0, 3, rows).astype(np.int32)


def trainer(**options):
    return SmallMLPTrainer(*weights(), data_schedule={"batch_size": 32}, **options)


def raw(value):
    return np.asarray(value).tobytes()


def test_default_device_is_none_and_trains():
    model = trainer()
    assert model.device is None
    X, y = batch()
    assert np.isfinite(model.train_step(X, y)["loss"])
    assert model.step_ == 1


@pytest.mark.parametrize("value", ["CPU", "cuda", 0, True, b"cpu"])
def test_unknown_device_is_refused_at_construction(value):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        trainer(device=value)


def test_refused_reconstruction_leaves_the_trainer_unchanged():
    model = trainer()
    X, y = batch()
    model.train_step(X, y)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        model.__init__(*weights(), data_schedule={"batch_size": 32}, device="metal")
    assert model.device is None and model.step_ == 1
    assert np.isfinite(model.train_step(X, y)["loss"])


def test_unknown_device_is_refused_at_restore(tmp_path):
    model = trainer()
    path = tmp_path / "mlp.json"
    model.save_checkpoint(path)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        SmallMLPTrainer.from_checkpoint(path, device="metal")
    assert SmallMLPTrainer.from_checkpoint(path).device is None


def test_explicit_gpu_follows_the_installation():
    if TWO_DEVICES:
        assert trainer(device="gpu").device == "gpu"
    else:
        with pytest.raises(ValueError, match="device='gpu' needs a GPU build"):
            trainer(device="gpu")


def test_cpu_device_trains_everywhere():
    model = trainer(device="cpu")
    assert model.device == "cpu"
    X, y = batch()
    out = model.train_step(X, y)
    assert out["step"] == 1 and np.isfinite(out["loss"])
    assert np.asarray(model.predict_logits(X)).shape == (32, 3)


@two_devices
def test_every_operation_returns_the_same_bits_on_both_devices(tmp_path):
    gpu, cpu = trainer(), trainer(device="cpu")
    X, y = batch()
    for step in range(3):
        a, b = gpu.train_step(X, y, return_input_grad=True), cpu.train_step(X, y, return_input_grad=True)
        assert a["step"] == b["step"] == step + 1
        assert a["loss"] == b["loss"]
        assert raw(a["logits"]) == raw(b["logits"]) and raw(a["input_grad"]) == raw(b["input_grad"])
        for name in ("weight1", "bias1", "weight2", "bias2"):
            assert raw(a["gradients"][name]) == raw(b["gradients"][name])
    X4, y4 = batch(rows=64, seed=9)
    a, b = gpu.train_steps(X4, y4, 4), cpu.train_steps(X4, y4, 4)
    assert list(a["losses"]) == list(b["losses"]) and raw(a["logits"]) == raw(b["logits"])
    assert raw(gpu.predict_logits(X)) == raw(cpu.predict_logits(X))
    assert raw(gpu.predict_logits_sessions([X, X4])) == raw(cpu.predict_logits_sessions([X, X4]))
    (la, ga), (lb, gb) = gpu.loss_and_grads(X, y), cpu.loss_and_grads(X, y)
    assert la == lb and [raw(g) for g in ga] == [raw(g) for g in gb]
    assert gpu.apply_gradients(ga) == cpu.apply_gradients(gb) == 8
    for name, value in gpu.weights_.items():
        assert raw(value) == raw(cpu.weights_[name])
    sa, sb = gpu.state_dict(), cpu.state_dict()
    assert sa["numeric_mode"] == sb["numeric_mode"] == "identical"
    assert raw(sa["optimizer"]["m"]) == raw(sb["optimizer"]["m"])
    assert raw(sa["optimizer"]["v"]) == raw(sb["optimizer"]["v"])
    gpu.save_checkpoint(tmp_path / "gpu.json")
    cpu.save_checkpoint(tmp_path / "cpu.json")
    assert (tmp_path / "gpu.json").read_bytes() == (tmp_path / "cpu.json").read_bytes()


@two_devices
def test_a_state_moves_between_devices(tmp_path):
    gpu = trainer()
    X, y = batch()
    gpu.train_step(X, y)
    cpu = trainer(device="cpu")
    assert cpu.load_state_dict(gpu.state_dict()) is cpu
    assert cpu.step_ == 1
    assert raw(gpu.train_step(X, y)["logits"]) == raw(cpu.train_step(X, y)["logits"])
    path = tmp_path / "mlp.json"
    gpu.save_checkpoint(path)
    restored = SmallMLPTrainer.from_checkpoint(path, device="cpu")
    assert restored.device == "cpu" and restored.step_ == 2
    assert raw(restored.predict_logits(X)) == raw(gpu.predict_logits(X))


@two_devices
def test_a_constructor_run_again_takes_the_new_device():
    model = trainer()
    X, y = batch()
    first = raw(model.train_step(X, y)["logits"])
    model.__init__(*weights(), data_schedule={"batch_size": 32}, device="cpu")
    assert model.device == "cpu" and model.step_ == 0
    assert raw(model.train_step(X, y)["logits"]) == first


@two_devices
def test_cpu_device_refuses_fast_mode_at_construction():
    from mojolearn import _backend
    if _backend.numeric_mode() != "fast":
        pytest.skip("the process runs identical mode")
    with pytest.raises(ValueError, match="device='cpu' runs in identical mode only"):
        trainer(device="cpu")
