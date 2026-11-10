# SPDX-License-Identifier: Apache-2.0
"""Python admission, dispatch, packing and refusal coverage of ops.Session.

The full numerical oracles and scheduler state checks live in
checks/native_tests/ops_session_cases.mojo and execute directly against typed
host/device sessions. Python retains tiny method-routing sentinels, returned
buffer/scalar contracts, and Python argument/error translation checks.
"""
import numpy as np
import pytest

from mojolearn import _backend


def _has_ops_entries():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_session_call")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_ops_entries(),
                                reason="the sequence binding has no ops_session_call entry (rebuild it)")

A = np.array([[1.25, -2.75], [3.5, 4.0]], np.float32)
B = np.array([[2.0, 2.0], [1.0, 0.5]], np.float32)


def _ops():
    from mojolearn import ops
    return ops


# A single output sentinel distinguishes each public method's dispatch route.
# Full numerical arrays and independent oracles are tested in Mojo.
BINARY = [
    ("sub", lambda s, a, b: s.sub(a, b), -0.75),
    ("div", lambda s, a, b: s.div(a, b), 0.625),
    ("minimum", lambda s, a, b: s.minimum(a, b), 1.25),
    ("pow", lambda s, a, b: s.pow(b, a), 2.378414230005442),
    ("equal", lambda s, a, b: s.equal(a, s.copy(a)), 1.0),
    ("not_equal", lambda s, a, b: s.not_equal(a, b), 1.0),
    ("greater_equal", lambda s, a, b: s.greater_equal(a, b), 0.0),
    ("less_equal", lambda s, a, b: s.less_equal(a, b), 1.0),
    ("logical_and", lambda s, a, b: s.logical_and(a, s.sub(b, b)), 0.0),
    ("logical_or", lambda s, a, b: s.logical_or(s.sub(a, a), b), 1.0),
]
UNARY = [
    ("ceil", lambda s, a: s.ceil(a), 2.0),
    ("sign", lambda s, a: s.sign(a), 1.0),
    ("round", lambda s, a: s.round(a), 1.0),
    ("logical_not", lambda s, a: s.logical_not(s.sub(a, a)), 1.0),
    ("pow_scalar", lambda s, a: s.pow_scalar(a, 2.0), 1.5625),
    ("unsqueeze", lambda s, a: s.unsqueeze(a, 0), 1.25),
]


def _assert_packed_buffer(value, shape):
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float32
    assert value.shape == shape


@pytest.mark.parametrize("name,call,sentinel", BINARY, ids=[r[0] for r in BINARY])
def test_binary_method_routes_and_packs_numpy_buffer(name, call, sentinel):
    with _ops().Session() as s:
        result = s.numpy(call(s, s.tensor(A), s.tensor(B)))
    _assert_packed_buffer(result, (2, 2))
    assert result.flat[0] == pytest.approx(sentinel, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("name,call,sentinel", UNARY, ids=[r[0] for r in UNARY])
def test_unary_method_routes_and_packs_numpy_buffer(name, call, sentinel):
    with _ops().Session() as s:
        result = s.numpy(call(s, s.tensor(A)))
    _assert_packed_buffer(result, (1, 2, 2) if name == "unsqueeze" else (2, 2))
    assert result.flat[0] == pytest.approx(sentinel, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_loss_keyword_admission_and_result_packing(reduction):
    t = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    with _ops().Session() as s:
        a, b, target = s.tensor(A), s.tensor(B), s.tensor(t)
        for h in (s.mae(a, b, reduction=reduction),
                  s.huber(a, b, delta=1.0, reduction=reduction),
                  s.bce_with_logits(a, target, reduction=reduction)):
            _assert_packed_buffer(s.numpy(h), (1, 1))
            assert isinstance(s.scalar(h), float)
        # These sentinels cover the string-to-native reduction mapping.
        assert s.scalar(s.mae(a, b, reduction=reduction)) == (2.875 if reduction == "mean" else 11.5)


def test_metric_scalar_permutation_and_pooling_result_packing():
    y = np.array([[1.0], [0.0], [1.0], [0.0]], np.float32)
    score = np.array([[0.9], [0.2], [0.4], [0.6]], np.float32)
    pred = np.array([[1.0], [0.0], [0.0], [1.0]], np.float32)
    with _ops().Session() as s:
        yt, sc = s.tensor(y), s.tensor(score)
        assert isinstance(s.accuracy(yt, s.tensor(pred)), float)
        assert isinstance(s.roc_auc(yt, sc), float)
        assert isinstance(s.log_loss(yt, sc), float)
        _assert_packed_buffer(s.numpy(s.permutation(7, seed=3)), (7, 1))
        assert isinstance(s.scalar(s.sum(s.tensor(A))), float)
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        _assert_packed_buffer(s.numpy(s.avg_pool2d(s.tensor(x), 2)), (1, 1, 2, 2))


def test_optimizer_and_profile_python_result_types():
    with _ops().Session(record=True) as s:
        opt = s.optimizer("sgd", lr=0.5, schedule="step", step_size=1, gamma=0.5)
        assert isinstance(s.learning_rate(opt), float)
        assert s.scheduler_step(opt) is None
        assert s.profile_reset() is None
        assert isinstance(s.profile_report(), str)


# A dead handle is refused by name by every operation.
DEAD = [(n, lambda s, a, d, c=c: c(s, a, d)) for n, c, _ in BINARY if n != "equal"] + \
       [(n, lambda s, a, d, c=c: c(s, d)) for n, c, _ in UNARY] + [
    ("equal", lambda s, a, d: s.equal(a, d)),
    ("mae", lambda s, a, d: s.mae(a, d)), ("huber", lambda s, a, d: s.huber(a, d)),
    ("bce_with_logits", lambda s, a, d: s.bce_with_logits(a, d)),
    ("accuracy", lambda s, a, d: s.accuracy(d, a)), ("log_loss", lambda s, a, d: s.log_loss(d, a)),
    ("roc_auc", lambda s, a, d: s.roc_auc(d, a)), ("scalar", lambda s, a, d: s.scalar(d)),
    ("avg_pool2d", lambda s, a, d: s.avg_pool2d(d, 2))]


@pytest.mark.parametrize("name,call", DEAD, ids=[r[0] for r in DEAD])
def test_dead_handle_is_refused_by_name(name, call):
    with _ops().Session() as s:
        a = s.tensor(A)
        dead = s.tensor(B)
        s.free(dead)
        with pytest.raises(ValueError, match="is not a live tensor of this session"):
            call(s, a, dead)
        with pytest.raises(TypeError, match="handle must be an integer"):
            call(s, a, "x")


# Poisoned arguments the native session refuses, naming what is wrong.
POISON = [
    ("sub", lambda s, a, c: s.sub(a, c), "incompatible tensor shapes"),
    ("div", lambda s, a, c: s.div(a, c), "incompatible tensor shapes"),
    ("minimum", lambda s, a, c: s.minimum(a, c), "incompatible tensor shapes"),
    ("pow", lambda s, a, c: s.pow(a, c), "incompatible tensor shapes"),
    ("equal", lambda s, a, c: s.equal(a, c), "incompatible tensor shapes"),
    ("not_equal", lambda s, a, c: s.not_equal(a, c), "incompatible tensor shapes"),
    ("greater_equal", lambda s, a, c: s.greater_equal(a, c), "incompatible tensor shapes"),
    ("less_equal", lambda s, a, c: s.less_equal(a, c), "incompatible tensor shapes"),
    ("logical_and", lambda s, a, c: s.logical_and(a, c), "incompatible tensor shapes"),
    ("logical_or", lambda s, a, c: s.logical_or(a, c), "incompatible tensor shapes"),
    ("unsqueeze", lambda s, a, c: s.unsqueeze(a, 7), "axis 7 is out of range"),
    ("avg_pool2d", lambda s, a, c: s.avg_pool2d(a, 2), r"pooling takes x \(N, C, H, W\)"),
    ("avg_pool2d_kernel", lambda s, a, c: s.avg_pool2d(s.tensor(np.ones((1, 1, 4, 4), np.float32)), 0),
     "kernel_size must be >= 1"),
    ("mae", lambda s, a, c: s.mae(a, c), "target must have the prediction's shape"),
    ("mae_reduction", lambda s, a, c: s.mae(a, a, reduction="max"), "reduction must be 'mean' or 'sum'"),
    ("huber", lambda s, a, c: s.huber(a, a, delta=0.0), "delta must be positive"),
    ("bce_with_logits", lambda s, a, c: s.bce_with_logits(a, a, reduction="max"),
     "reduction must be 'mean' or 'sum'"),
    ("accuracy", lambda s, a, c: s.accuracy(a, c), r"y_true must be \(n, 1\)"),
    ("log_loss", lambda s, a, c: s.log_loss(a, c), r"y_true must be \(n, 1\)"),
    ("roc_auc", lambda s, a, c: s.roc_auc(a, c), r"y_true must be \(n, 1\)"),
    ("permutation", lambda s, a, c: s.permutation(0), "n must be >= 1"),
    ("permutation_seed", lambda s, a, c: s.permutation(4, seed=-1), "seed must be in"),
    ("scalar", lambda s, a, c: s.scalar(a), "takes a tensor of one value"),
    ("scheduler_step", lambda s, a, c: s.scheduler_step(987654), "is not an optimizer of this session"),
]


@pytest.mark.parametrize("name,call,match", POISON, ids=[r[0] for r in POISON])
def test_poisoned_argument_is_refused_by_name(name, call, match):
    with _ops().Session() as s:
        a, c = s.tensor(A), s.tensor(np.ones((3, 3), np.float32))
        with pytest.raises(Exception, match=match):
            call(s, a, c)


@pytest.mark.parametrize("name", ["sub", "ceil", "mae", "scalar", "profile_report", "profile_reset"])
def test_closed_session_refuses_every_operation(name):
    s = _ops().Session()
    a = s.tensor(A)
    s.close()
    calls = {"sub": lambda: s.sub(a, a), "ceil": lambda: s.ceil(a), "mae": lambda: s.mae(a, a),
             "scalar": lambda: s.scalar(a)}
    if name in calls:
        with pytest.raises(ValueError, match="this session is closed"):
            calls[name]()
    else:
        # the launch profile is process-wide, so it still answers after close
        getattr(s, name)()


# Scalar arguments that are not native real scalars are refused naming the argument
# (no conversion hook such as __float__ runs).
SCALAR = [
    ("pow_scalar", lambda s, a: s.pow_scalar(a, "two"), "exponent"),
    ("scale", lambda s, a: s.scale(a, "x"), "factor"),
    ("leaky_relu", lambda s, a: s.leaky_relu(a, alpha="x"), "alpha"),
    ("elu", lambda s, a: s.elu(a, alpha=None), "alpha"),
    ("hard_sigmoid", lambda s, a: s.hard_sigmoid(a, beta="x"), "beta"),
    ("full", lambda s, a: s.full((2, 2), "x"), "value"),
    ("clip", lambda s, a: s.clip(a, min="x"), "min"),
    ("pad", lambda s, a: s.pad(a, (1, 1), value="x"), "value"),
    ("dropout", lambda s, a: s.dropout(a, "x"), "p"),
    ("huber", lambda s, a: s.huber(a, a, delta="x"), "delta"),
    ("optimizer", lambda s, a: s.optimizer("sgd", lr="x"), "lr"),
]


@pytest.mark.parametrize("name,call,argument", SCALAR, ids=[r[0] for r in SCALAR])
def test_non_numeric_scalar_argument_is_refused_by_name(name, call, argument):
    with _ops().Session() as s:
        a = s.tensor(A)
        with pytest.raises(TypeError, match=f"^{argument} must be a native real scalar without conversion hooks"):
            call(s, a)
