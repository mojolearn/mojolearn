# SPDX-License-Identifier: Apache-2.0
"""Python Session optimizer options, result packing and checkpoint ownership.

Numerical training parity, independent optimizer equations, schedule values,
checkpoint continuation and gradient reset moved to the direct native
ops_optimizer_{cpu,gpu}.mojo checks. These cases retain the actual public
Python interface: keyword decoding, scalar/dict/tensor results, checkpoint
arrays surviving Session.close, and exception categories.
"""
import numpy as np
import pytest

from mojolearn import _backend


def _has_step_entry():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_session_call")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_step_entry(),
                                reason="the sequence binding has no ops_session_call entry (rebuild it)")


def _ops():
    from mojolearn import ops
    return ops


def _tensors(s, weights=None):
    if weights is None:
        weights = np.array([[0.25, -0.125], [0.5, 0.25]], dtype=np.float32)
    w = s.tensor(weights, requires_grad=True)
    x = s.tensor(np.array([[1.0, 0.5], [-0.5, 1.0]], dtype=np.float32))
    y = s.tensor(np.zeros((2, 2), dtype=np.float32))
    return w, x, y


def _backward(s, w, x, y, *, accumulate=False):
    s.backward(s.mse(s.matmul(x, w), y), accumulate=accumulate)


CASES = {
    "sgd_step_schedule": ("sgd", dict(lr=0.05, schedule="step", step_size=2, gamma=0.5)),
    "sgd_nesterov": ("sgd", dict(lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01)),
    "adam": ("adam", dict(lr=0.01)),
    "adam_exponential": ("adam", dict(lr=0.02, schedule="exponential", gamma=0.9, weight_decay=0.001)),
    "adamw": ("adamw", dict(lr=0.01, weight_decay=0.05, beta1=0.8, beta2=0.99)),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_optimizer_options_and_output_publication(case):
    kind, options = CASES[case]
    with _ops().Session(record=True) as s:
        w, x, y = _tensors(s)
        opt = s.optimizer(kind, **options)
        assert type(opt) is int
        _backward(s, w, x, y)
        s.step(opt, [w])
        result = s.numpy(w)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32 and result.shape == (2, 2)
        state = s.optimizer_state(opt, w)
        assert type(state) is dict and type(state["step"]) is int
        assert state["step"] == 1


def test_learning_rate_python_scalar():
    with _ops().Session(record=True) as s:
        opt = s.optimizer("sgd", lr=0.1, schedule="step", step_size=2, gamma=0.5)
        assert type(s.learning_rate(opt)) is float
        s.scheduler_step(opt)
        assert type(s.learning_rate(opt)) is float


@pytest.mark.parametrize("kind,opts", [("adam", dict(lr=0.01)), ("sgd", dict(lr=0.05, momentum=0.9))])
def test_optimizer_checkpoint_dict_and_owner_boundary(kind, opts):
    buffer_keys = {"exp_avg", "exp_avg_sq"} if kind == "adam" else {"momentum_buffer"}
    scalar_keys = {"step", "beta1_power", "beta2_power"} if kind == "adam" else {"step"}
    with _ops().Session(record=True) as s:
        w, x, y = _tensors(s)
        opt = s.optimizer(kind, **opts)
        _backward(s, w, x, y)
        s.step(opt, w)
        state = s.optimizer_state(opt, w)
        assert type(state) is dict and set(state) == buffer_keys | scalar_keys
        assert type(state["step"]) is int and state["step"] == 1
        for key in scalar_keys - {"step"}:
            assert type(state[key]) is float
        weights = s.numpy(w)
        saved = {key: s.numpy(value) if key in buffer_keys else value
                 for key, value in state.items()}
        for key in buffer_keys:
            assert type(state[key]) is int
            assert isinstance(saved[key], np.ndarray)
            assert saved[key].shape == (2, 2) and saved[key].dtype == np.float32
        snapshot_bytes = {key: saved[key].tobytes() for key in buffer_keys}
    # These arrays must remain owned after the exporting Session closes.
    with _ops().Session(record=True) as s:
        w, x, y = _tensors(s, weights)
        opt = s.optimizer(kind, **opts)
        restored = {key: s.tensor(value) if key in buffer_keys else value
                    for key, value in saved.items()}
        s.load_optimizer_state(opt, w, restored)
        assert s.optimizer_state(opt, w)["step"] == 1
        _backward(s, w, x, y)
        s.step(opt, [w])
        assert s.optimizer_state(opt, w)["step"] == 2
    assert {key: saved[key].tobytes() for key in buffer_keys} == snapshot_bytes


def test_zero_grad_public_result_and_refusal():
    with _ops().Session(record=True) as s:
        w, x, y = _tensors(s)
        _backward(s, w, x, y)
        assert s.grad(w).shape == (2, 2)
        s.zero_grad()
        with pytest.raises(Exception):
            s.grad(w)
        _backward(s, w, x, y, accumulate=True)
        result = s.grad(w)
        assert result.shape == (2, 2) and result.dtype == np.float32


def test_step_refusals():
    ops = _ops()
    with ops.Session(record=True) as s:
        w = s.tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
        c = s.tensor(np.ones((2, 2), dtype=np.float32))
        opt = s.optimizer("adam")
        with pytest.raises(Exception):
            s.step(opt, [w])               # no backward yet
        s.backward(s.sum(s.matmul(c, w)))
        with pytest.raises(Exception):
            s.step(opt, [c])               # not trainable
        with pytest.raises(Exception):
            s.step(opt, [w, w])            # listed twice
        with pytest.raises(ValueError):
            s.step(opt + 1, [w])           # not an optimizer of this session
        with pytest.raises(Exception):
            s.optimizer("sgd", nesterov=True)   # Nesterov without momentum
        with pytest.raises(Exception):
            s.optimizer("adam", lr=0.0)
        with pytest.raises(ValueError):
            s.optimizer("lion")
        s.step(opt, w)
        assert s.optimizer_state(opt, w)["step"] == 1
    with ops.Session() as s:
        with pytest.raises(Exception):
            s.zero_grad()                  # not a recording session
