# SPDX-License-Identifier: Apache-2.0
"""mojolearn.ops.Session, the operation session over the autograd tape
(autograd/ops_api.mojo): a two-expert mixture built from the ops (a hard
route chosen by one scalar read back, and a soft gate-weighted sum), a tape
loss and metrics on frozen predictions, a permutation gather, and a
recording session's gradient. Tiny data. Every case asserts shapes and
finite values and the same digest across two independent sessions, and
prints OPS_DIGEST lines to diff across CPU, Metal, NVIDIA and AMD."""
import hashlib

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


def _ops():
    from mojolearn import ops
    return ops


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _emit(name, digest):
    print(f"OPS_DIGEST {name} {digest}")


def _moe_inputs():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 4)).astype(np.float32)
    W0 = rng.standard_normal((4, 3)).astype(np.float32)
    W1 = rng.standard_normal((4, 3)).astype(np.float32)
    Wg = rng.standard_normal((4, 2)).astype(np.float32)
    b0 = rng.standard_normal((1, 3)).astype(np.float32)
    b1 = rng.standard_normal((1, 3)).astype(np.float32)
    bg = rng.standard_normal((1, 2)).astype(np.float32)
    return X, W0, b0, W1, b1, Wg, bg


def _moe_run():
    X, W0, b0, W1, b1, Wg, bg = _moe_inputs()
    with _ops().Session() as s:
        x = s.tensor(X)
        y0 = s.linear(x, s.tensor(W0), s.tensor(b0))
        y1 = s.linear(x, s.tensor(W1), s.tensor(b1), activation="tanh")
        gate = s.linear(x, s.tensor(Wg), s.tensor(bg), activation="softmax")
        # hard route: the expert with the larger mean gate, one scalar read back
        mean_gate = s.mean(gate, axis=0)
        margin = s.matmul(mean_gate, s.tensor(np.array([[1.0], [-1.0]], dtype=np.float32)))
        first = s.scalar(margin) > 0
        hard = s.relu(y0) if first else s.relu(y1)
        # soft route: each row's experts weighted by its gate, on the device
        g0 = s.matmul(gate, s.tensor(np.array([[1.0], [0.0]], dtype=np.float32)))
        g1 = s.matmul(gate, s.tensor(np.array([[0.0], [1.0]], dtype=np.float32)))
        soft = s.add(s.mul(g0, y0), s.mul(g1, y1))
        assert s.shape(gate) == (16, 2)
        assert s.shape(margin) == (1, 1)
        return s.numpy(gate), s.numpy(hard), s.numpy(soft), first


def test_mixture_of_two_experts():
    gate, hard, soft, first = _moe_run()
    assert gate.shape == (16, 2) and hard.shape == (16, 3) and soft.shape == (16, 3)
    assert np.isfinite(gate).all() and np.isfinite(hard).all() and np.isfinite(soft).all()
    assert np.allclose(gate.sum(axis=1), 1.0, atol=1e-5)
    assert (hard >= 0).all()
    gate2, hard2, soft2, first2 = _moe_run()
    assert first == first2
    d = _digest(gate, hard, soft)
    assert d == _digest(gate2, hard2, soft2)
    _emit("moe_gate", _digest(gate))
    _emit("moe_hard", _digest(hard))
    _emit("moe_soft", _digest(soft))


def _frozen_run():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((32, 3)).astype(np.float32)
    codes = rng.integers(0, 3, size=(32, 1)).astype(np.float32)
    target = rng.standard_normal((32, 3)).astype(np.float32)
    binary = (rng.random((32, 1)) > 0.5).astype(np.float32)
    binary[0, 0], binary[1, 0] = 0.0, 1.0
    with _ops().Session() as s:
        z = s.tensor(logits)
        c = s.tensor(codes)
        ce = s.cross_entropy(z, c)
        ce_sum = s.cross_entropy(z, c, reduction="sum")
        mse = s.mse(z, s.tensor(target))
        hub = s.huber(z, s.tensor(target), delta=0.5)
        prob = s.softmax(z)
        acc = s.accuracy(c, z)
        ll = s.log_loss(c, prob)
        auc = s.roc_auc(s.tensor(binary), s.tensor(logits[:, :1]))
        losses = np.array([s.scalar(ce), s.scalar(ce_sum), s.scalar(mse), s.scalar(hub)], dtype=np.float32)
        metrics = np.array([acc, ll, auc], dtype=np.float32)
        return losses, metrics


def test_loss_and_metrics_on_frozen_predictions():
    losses, metrics = _frozen_run()
    assert np.isfinite(losses).all() and np.isfinite(metrics).all()
    assert losses[0] > 0 and losses[1] > losses[0]
    assert 0.0 <= metrics[0] <= 1.0 and metrics[1] > 0 and 0.0 <= metrics[2] <= 1.0
    losses2, metrics2 = _frozen_run()
    assert _digest(losses, metrics) == _digest(losses2, metrics2)
    _emit("frozen_losses", _digest(losses))
    _emit("frozen_metrics", _digest(metrics))


def _gather_run():
    X = np.arange(40, dtype=np.float32).reshape(10, 4)
    with _ops().Session() as s:
        x = s.tensor(X)
        perm = s.permutation(10, seed=7, epoch=0)
        rows = s.gather_rows(x, perm)
        batch = s.gather_rows(x, perm, start=2, count=3)
        assert s.shape(rows) == (10, 4) and s.shape(batch) == (3, 4)
        return s.numpy(perm), s.numpy(rows), s.numpy(batch)


def test_permutation_and_gather():
    perm, rows, batch = _gather_run()
    assert sorted(perm[:, 0].astype(int).tolist()) == list(range(10))
    assert np.array_equal(np.sort(rows[:, 0]), np.arange(0, 40, 4, dtype=np.float32))
    assert np.array_equal(batch, rows[2:5])
    perm2, rows2, batch2 = _gather_run()
    assert _digest(perm, rows, batch) == _digest(perm2, rows2, batch2)
    _emit("gather", _digest(perm, rows, batch))


def _grad_run():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((24, 5)).astype(np.float32)
    W = rng.standard_normal((5, 2)).astype(np.float32)
    Y = rng.standard_normal((24, 2)).astype(np.float32)
    with _ops().Session(record=True) as s:
        w = s.tensor(W, requires_grad=True)
        pred = s.matmul(s.tensor(X), w)
        loss = s.mse(pred, s.tensor(Y))
        s.backward(loss)
        return X, W, Y, s.scalar(loss), s.grad(w)


def test_recording_session_gradient():
    X, W, Y, loss, g = _grad_run()
    assert g.shape == W.shape and np.isfinite(g).all()
    reference = (2.0 / Y.size) * X.T @ (X @ W - Y)
    assert np.allclose(g, reference, rtol=1e-4, atol=1e-5)
    assert np.isclose(loss, np.mean((X @ W - Y) ** 2), rtol=1e-5)
    _, _, _, loss2, g2 = _grad_run()
    assert _digest([loss], g) == _digest([loss2], g2)
    _emit("grad", _digest([loss], g))


def test_refusals():
    ops = _ops()
    with ops.Session() as s:
        x = s.tensor(np.ones((3, 2), dtype=np.float32))
        with pytest.raises(Exception):
            s.tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
        with pytest.raises(Exception):
            s.gather_rows(x, x)
        with pytest.raises(Exception):
            s.matmul(x, x)
        s.free(x)
        with pytest.raises(Exception):
            s.numpy(x)
    with pytest.raises(ValueError):
        s.numpy(1)
