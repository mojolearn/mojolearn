# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on `ops.Session`, at the Python boundary.

The device is a constructor parameter, read once when the session is opened: the session keeps the connection of the
compiled binary it runs on, and every operation, and every Program captured from it, runs there. None is this
installation's device. An explicit `device="gpu"` on a CPU-only installation is refused in words that name
`device="cpu"`; `device="cpu"` always runs, in identical mode. On a GPU installation the CPU binary is loaded beside
the GPU one and the two kinds of session work in one process; in identical mode they return the same bits. What needs
two devices is skipped by name on a CPU-only installation.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import ops

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def vendor_of(session):
    """The vendor word the session's own compiled binary reports."""
    return session._b.x_sequence_vendor()


def step(session, seed=5):
    """A small recorded composition: forward, loss, gradient, one Adam step, a captured program run."""
    rng = np.random.default_rng(seed)
    x = session.tensor(rng.standard_normal((32, 4), dtype=np.float32))
    y = session.tensor(rng.standard_normal((32, 2), dtype=np.float32))
    w = session.tensor(rng.standard_normal((4, 2), dtype=np.float32), requires_grad=True)
    out = session.linear(x, w, activation="tanh")
    loss = session.mse(out, y)
    session.backward(loss)
    grad = np.array(session.grad(w), copy=True)
    forward = np.array(session.numpy(out), copy=True)
    opt = session.optimizer("adam", lr=0.01)
    session.step(opt, [w])
    stepped = np.array(session.numpy(w), copy=True)
    return forward, grad, stepped, float(session.scalar(loss))


def test_device_is_a_constructor_parameter_with_default_none():
    with ops.Session() as s:
        assert s.device is None
        assert vendor_of(s) == mojolearn.vendor()
    with ops.Session(False, None, None) as s:
        assert s.device is None
    with ops.Session(device="cpu") as s:
        assert s.device == "cpu"
        assert vendor_of(s) == "cpu"


@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_before_a_session_exists(bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        ops.Session(device=bad)


def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one():
    if TWO_DEVICES:
        with ops.Session(device="gpu") as s:
            assert s.device == "gpu" and vendor_of(s) == mojolearn.vendor()
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        ops.Session(device="gpu")


def test_a_cpu_session_computes_and_trains():
    with ops.Session(record=True, numeric_mode="identical", device="cpu") as s:
        forward, grad, stepped, loss = step(s)
    assert forward.shape == (32, 2) and grad.shape == (4, 2) and stepped.shape == (4, 2)
    assert np.isfinite(loss) and np.all(np.isfinite(stepped))


@two_devices
def test_identical_mode_cpu_session_returns_the_gpu_bits():
    with ops.Session(record=True, numeric_mode="identical") as gpu:
        on_gpu = step(gpu)
    with ops.Session(record=True, numeric_mode="identical", device="cpu") as cpu:
        on_cpu = step(cpu)
    for a, b in zip(on_gpu[:3], on_cpu[:3]):
        assert a.tobytes() == b.tobytes()
    assert np.float32(on_gpu[3]).tobytes() == np.float32(on_cpu[3]).tobytes()


@two_devices
def test_two_sessions_on_two_devices_interleave_in_one_process():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((16, 3), dtype=np.float32)
    w = rng.standard_normal((3, 2), dtype=np.float32)
    with ops.Session(numeric_mode="identical") as gpu, ops.Session(numeric_mode="identical", device="cpu") as cpu:
        assert vendor_of(gpu) != "cpu" and vendor_of(cpu) == "cpu"
        xg, xc = gpu.tensor(a), cpu.tensor(a)
        wg, wc = gpu.tensor(w), cpu.tensor(w)
        yg = gpu.matmul(xg, wg)
        yc = cpu.matmul(xc, wc)
        first = np.array(gpu.numpy(yg), copy=True)
        second = np.array(cpu.numpy(yc), copy=True)
        again = np.array(gpu.numpy(gpu.matmul(xg, wg)), copy=True)
    assert first.tobytes() == second.tobytes() == again.tobytes()


def test_a_program_captured_from_a_cpu_session_runs_on_the_cpu_session():
    rng = np.random.default_rng(2)
    w1 = rng.standard_normal((4, 8), dtype=np.float32)
    batch = rng.standard_normal((32, 4), dtype=np.float32)
    answers = []
    devices = ["cpu", None] if TWO_DEVICES else ["cpu"]
    for device in devices:
        with ops.Session(numeric_mode="identical", device=device) as s:
            x = s.tensor(np.zeros((32, 4), np.float32))
            y = s.linear(x, s.tensor(w1), activation="relu")
            model = s.capture(inputs=[x], outputs=[y])
            s.reset()
            first = np.array(model.run(batch), copy=True)
            second = np.array(model.run(batch), copy=True)    # the repeated run: the kept route of its own image
            assert first.tobytes() == second.tobytes()
            answers.append(first)
    if TWO_DEVICES:
        assert answers[0].tobytes() == answers[1].tobytes()


@two_devices
def test_cpu_session_never_runs_a_fast_binary():
    if mojolearn.vendor() != "metal":
        pytest.skip("only an Apple installation carries a fast set")
    with pytest.raises(ValueError, match="device='cpu' runs in identical mode only"):
        ops.Session(numeric_mode="fast", device="cpu")


def test_a_closed_cpu_session_refuses_like_any_closed_session():
    s = ops.Session(device="cpu")
    s.close()
    with pytest.raises(Exception):
        s.tensor(np.ones((2, 2), dtype=np.float32))
