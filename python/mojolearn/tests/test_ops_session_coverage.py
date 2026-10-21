# SPDX-License-Identifier: Apache-2.0
"""Release-gate coverage of mojolearn.ops.Session operations changed since
0.8.55 (docs/plans/TEST_COVERAGE_20261021.md): every operation below is
called on the real native session and checked against NumPy, then called
with a poisoned input that the native session must refuse by name. Tiny
data, no measurements."""
import numpy as np
import pytest

from mojolearn import _backend


def _has_ops_entries():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_open")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_ops_entries(),
                                reason="the sequence binding has no ops_open entry (rebuild it)")

A = np.array([[1.25, -2.75], [3.5, 4.0]], np.float32)
B = np.array([[2.0, 2.0], [1.0, 0.5]], np.float32)
ATOL = 1e-6


def _ops():
    from mojolearn import ops
    return ops


# (name, call(s, a, b) -> handle or float, NumPy reference(A, B))
BINARY = [
    ("sub", lambda s, a, b: s.sub(a, b), lambda x, y: x - y),
    ("div", lambda s, a, b: s.div(a, b), lambda x, y: x / y),
    ("minimum", lambda s, a, b: s.minimum(a, b), np.minimum),
    ("pow", lambda s, a, b: s.pow(b, a), lambda x, y: np.power(y, x)),
    ("equal", lambda s, a, b: s.equal(a, s.copy(a)), lambda x, y: np.ones_like(x)),
    ("not_equal", lambda s, a, b: s.not_equal(a, b), lambda x, y: (x != y).astype(np.float32)),
    ("greater_equal", lambda s, a, b: s.greater_equal(a, b), lambda x, y: (x >= y).astype(np.float32)),
    ("less_equal", lambda s, a, b: s.less_equal(a, b), lambda x, y: (x <= y).astype(np.float32)),
    ("logical_and", lambda s, a, b: s.logical_and(a, s.sub(b, b)), lambda x, y: np.zeros_like(x)),
    ("logical_or", lambda s, a, b: s.logical_or(s.sub(a, a), b), lambda x, y: np.ones_like(x)),
]
UNARY = [
    ("ceil", lambda s, a: s.ceil(a), np.ceil),
    ("sign", lambda s, a: s.sign(a), np.sign),
    ("round", lambda s, a: s.round(a), np.round),
    ("logical_not", lambda s, a: s.logical_not(s.sub(a, a)), lambda x: np.ones_like(x)),
    ("pow_scalar", lambda s, a: s.pow_scalar(a, 2.0), np.square),
    ("unsqueeze", lambda s, a: s.unsqueeze(a, 0), lambda x: x[None]),
]


@pytest.mark.parametrize("name,call,ref", BINARY, ids=[r[0] for r in BINARY])
def test_binary_operation_matches_numpy(name, call, ref):
    with _ops().Session() as s:
        out = s.numpy(call(s, s.tensor(A), s.tensor(B)))
    np.testing.assert_allclose(out, ref(A, B), rtol=1e-6, atol=ATOL)


@pytest.mark.parametrize("name,call,ref", UNARY, ids=[r[0] for r in UNARY])
def test_unary_operation_matches_numpy(name, call, ref):
    with _ops().Session() as s:
        out = s.numpy(call(s, s.tensor(A)))
    np.testing.assert_allclose(out, ref(A), rtol=1e-6, atol=ATOL)


def test_losses_match_numpy_and_honor_reduction():
    t = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    d = A - B
    huber = np.where(np.abs(d) <= 1.0, 0.5 * d * d, np.abs(d) - 0.5)
    bce = np.maximum(A, 0) - A * t + np.log1p(np.exp(-np.abs(A)))
    with _ops().Session() as s:
        a, b, tt = s.tensor(A), s.tensor(B), s.tensor(t)
        assert s.numpy(s.mae(a, b)).shape == (1, 1)
        np.testing.assert_allclose(s.scalar(s.mae(a, b)), np.abs(d).mean(), rtol=1e-6)
        np.testing.assert_allclose(s.scalar(s.mae(a, b, reduction="sum")), np.abs(d).sum(), rtol=1e-6)
        np.testing.assert_allclose(s.scalar(s.huber(a, b, delta=1.0)), huber.mean(), rtol=1e-6)
        np.testing.assert_allclose(s.scalar(s.bce_with_logits(a, tt)), bce.mean(), rtol=1e-5)
        np.testing.assert_allclose(s.scalar(s.bce_with_logits(a, tt, reduction="sum")), bce.sum(), rtol=1e-5)


def test_metrics_permutation_scalar_and_pooling_match_numpy():
    y = np.array([[1.0], [0.0], [1.0], [0.0]], np.float32)
    score = np.array([[0.9], [0.2], [0.4], [0.6]], np.float32)
    with _ops().Session() as s:
        yt, sc = s.tensor(y), s.tensor(score)
        assert s.accuracy(yt, s.tensor((score > 0.5).astype(np.float32))) == 0.5
        np.testing.assert_allclose(s.roc_auc(yt, sc), 0.75, rtol=1e-6)
        ll = -np.mean(y * np.log(score) + (1 - y) * np.log(1 - score))
        np.testing.assert_allclose(s.log_loss(yt, sc), ll, rtol=1e-5)
        perm = s.numpy(s.permutation(7, seed=3)).reshape(-1)
        assert sorted(perm.astype(int).tolist()) == list(range(7))
        assert s.scalar(s.sum(s.tensor(A))) == pytest.approx(float(A.sum()))
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        pooled = s.numpy(s.avg_pool2d(s.tensor(x), 2))
        np.testing.assert_allclose(pooled, x.reshape(1, 1, 2, 2, 2, 2).mean(axis=(3, 5)), rtol=1e-6)


def test_scheduler_step_advances_learning_rate_and_profile_text():
    with _ops().Session(record=True) as s:
        opt = s.optimizer("sgd", lr=0.5, schedule="step", step_size=1, gamma=0.5)
        assert s.learning_rate(opt) == 0.5
        s.scheduler_step(opt)
        assert s.learning_rate(opt) == 0.25
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
