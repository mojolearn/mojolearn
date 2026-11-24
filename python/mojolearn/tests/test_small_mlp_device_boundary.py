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


# moved from python/mojolearn/tests/test_neural_inference.py
# MLPInference at the public boundary: returned shape and dtype, checkpoint
# refusals by name and constructor/query refusals.
import json
import os
import tempfile

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend

_NINF_MLP_SHAPES = ((16, 8), (16,), (3, 16), (3,))


@pytest.fixture(scope="module")
def ninf_mlp_binding():
    try:
        return _backend.load_host_module("_mojolearn_neural_host")
    except ImportError as exc:
        pytest.skip(f"neural host binding not built: {exc}")


def _ninf_mlp_weights():
    return [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
            for s in _NINF_MLP_SHAPES]


def _ninf_x(rows, cols=8, seed=0):
    return np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)


def _ninf_checkpoint_from_weights(tmp):
    """A trainer checkpoint written by the public trainer's own
    `save_checkpoint`, before any step."""
    path = os.path.join(tmp, "mlp.json")
    ml.SmallMLPTrainer(*_ninf_mlp_weights(), data_schedule={"dataset": "test"}).save_checkpoint(path)
    return path


def test_mlp_logits_shape_rows_alone_and_checkpoint(ninf_mlp_binding):
    inf = ml.MLPInference(*_ninf_mlp_weights())
    X = _ninf_x(64)
    got = np.asarray(inf.predict_logits(X))
    # rows alone equal the batch: the native check training_neural_inference
    assert got.shape == (64, 3) and got.dtype == np.float32
    assert set(inf.weights_) == {"weight1", "bias1", "weight2", "bias2"}


def test_mlp_from_checkpoint_equals_constructor_and_refuses_by_name(ninf_mlp_binding):
    tmp = tempfile.mkdtemp()
    path = _ninf_checkpoint_from_weights(tmp)
    X = _ninf_x(32, seed=3)
    a = np.asarray(ml.MLPInference.from_checkpoint(path).predict_logits(X))
    b = np.asarray(ml.MLPInference(*_ninf_mlp_weights()).predict_logits(X))
    assert a.tobytes() == b.tobytes()
    env = json.load(open(path))
    env["payload"]["data_schedule"] = {"dataset": "tampered"}
    bad = os.path.join(tmp, "bad.json")
    json.dump(env, open(bad, "w"))
    with pytest.raises(ValueError, match="integrity mismatch"):
        ml.MLPInference.from_checkpoint(bad)
    env["schema"] = "other"
    json.dump(env, open(bad, "w"))
    with pytest.raises(ValueError, match="schema mismatch"):
        ml.MLPInference.from_checkpoint(bad)
    with open(bad, "wb") as fh:
        fh.write(b"{" * 40000)
    with pytest.raises(ValueError, match="size bound"):
        ml.MLPInference.from_checkpoint(bad)


def test_mlp_refusals(ninf_mlp_binding):
    w = _ninf_mlp_weights()
    with pytest.raises(ValueError, match="shape"):
        ml.MLPInference(w[0].T.copy(), *w[1:])
    inf = ml.MLPInference(*w)
    with pytest.raises(ValueError, match=r"shape \(batch, 8\)"):
        inf.predict_logits(_ninf_x(4, cols=7))
    with pytest.raises(ValueError, match=r"\[1, 256\]"):
        inf.predict_logits(_ninf_x(257))
    bad = w[1].copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ml.MLPInference(w[0], bad, w[2], w[3])
