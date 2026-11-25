# SPDX-License-Identifier: Apache-2.0
"""Session.custom_op / Session.custom (sub-lane nn-blocks-cbw): the Python
boundary of a user-defined backward rule. Admission and refusals in clear
words, the MyTanh example against the built-in tanh bit for bit, an input used
along two paths under a seed other than one, several outputs, and a program
holding the operation saved, loaded and trained one step to the same bits.
The native cases are checks/native_tests/nn_cbw.mojo."""
import numpy as np
import pytest

from mojolearn import _backend


def _has_entries():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        return hasattr(b, "ops_session_call")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_entries(),
                                reason="the sequence binding has no ops_session_call entry (rebuild it)")


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _my_tanh(cap, shape):
    """MyTanh's programs: forward y = tanh(x) (saving y), backward g * (1 - y * y)."""
    zeros = np.zeros(shape, np.float32)
    x = cap.tensor(zeros)
    forward = cap.capture(inputs=[x], outputs=[cap.tanh(x)])
    y, g = cap.tensor(zeros), cap.tensor(zeros)
    one = cap.full((1, 1), 1.0)
    backward = cap.capture(inputs=[y, g], outputs=[cap.mul(g, cap.sub(one, cap.mul(y, y)))])
    return forward, backward


def _x(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_my_tanh_equals_builtin_tanh_bit_for_bit():
    from mojolearn import ops
    x0 = _x((8, 5), 0)
    with ops.Session() as cap:
        forward, backward = _my_tanh(cap, x0.shape)
        with ops.Session(record=True) as s:
            op = s.custom_op(forward, backward, saved=[0])
            assert isinstance(op, int)
            w = s.tensor(x0, requires_grad=True)
            y = s.custom(op, w)
            s.backward(s.scale(s.sum(y), 3.0))
            got = s.grad(w)
            values = s.numpy(y)
    with ops.Session(record=True) as r:
        w = r.tensor(x0, requires_grad=True)
        y = r.tanh(w)
        r.backward(r.scale(r.sum(y), 3.0))
        want = r.grad(w)
        want_values = r.numpy(y)
    assert np.array_equal(_words(values), _words(want_values))
    assert np.array_equal(_words(got), _words(want))


def test_two_paths_with_seed_three_get_both_contributions():
    from mojolearn import ops
    x0 = _x((6, 4), 1)
    with ops.Session() as cap:
        forward, backward = _my_tanh(cap, x0.shape)
        with ops.Session(record=True) as s:
            op = s.custom_op(forward, backward, saved=[0])
            x = s.tensor(x0, requires_grad=True)
            t = s.custom(op, x)          # recorded first: its gradient is added last
            z = s.mul(x, x)
            s.backward(s.scale(s.add(s.sum(t), s.sum(z)), 3.0))
            got = s.grad(x).astype(np.float64)
    x64 = x0.astype(np.float64)
    want = 3.0 * (1.0 - np.tanh(x64) ** 2) + 6.0 * x64
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_several_outputs_and_saved_extras():
    from mojolearn import ops
    x0 = _x((5, 3), 2)
    zeros = np.zeros(x0.shape, np.float32)
    with ops.Session() as cap:
        x = cap.tensor(zeros)
        # outputs (tanh(x), x * x); x itself is a saved extra (output 2)
        forward = cap.capture(inputs=[x], outputs=[cap.tanh(x), cap.mul(x, x), x])
        y, xs, g0, g1 = (cap.tensor(zeros) for _ in range(4))
        one, two = cap.full((1, 1), 1.0), cap.full((1, 1), 2.0)
        dx = cap.add(cap.mul(g0, cap.sub(one, cap.mul(y, y))), cap.mul(g1, cap.mul(two, xs)))
        backward = cap.capture(inputs=[y, xs, g0, g1], outputs=[dx])
        with ops.Session(record=True) as s:
            op = s.custom_op(forward, backward, saved=[0, 2], outputs=2)
            w = s.tensor(x0, requires_grad=True)
            a, b = s.custom(op, w)
            assert s.shape(a) == x0.shape and s.shape(b) == x0.shape
            s.backward(s.add(s.scale(s.sum(a), 3.0), s.scale(s.sum(b), 0.5)))
            got = s.grad(w).astype(np.float64)
    x64 = x0.astype(np.float64)
    np.testing.assert_allclose(got, 3.0 * (1.0 - np.tanh(x64) ** 2) + x64, rtol=1e-5, atol=1e-5)


def test_saved_program_trains_one_step_to_the_same_bits(tmp_path):
    from mojolearn import ops
    rng = np.random.default_rng(3)
    X = rng.standard_normal((16, 3)).astype(np.float32)
    T = rng.standard_normal((16, 3)).astype(np.float32)
    W = rng.standard_normal((3, 3)).astype(np.float32)
    path = str(tmp_path / "custom.mlp")
    with ops.Session() as s:
        forward, backward = _my_tanh(s, (16, 3))
        op = s.custom_op(forward, backward, saved=[0])
        x = s.tensor(np.zeros((16, 3), np.float32))
        w = s.tensor(W)
        model = s.capture(inputs=[x], outputs=[s.custom(op, s.matmul(x, w))])
        assert model.inspect() is not None
        model.save(path, verify=False)
        before = np.asarray(model.run(X))
        opt = s.optimizer("sgd", lr=0.1)
        model.fit(X, targets=T, optimizer=opt, loss="mse", epochs=1, batch_size=16, shuffle=False)
        trained = s.numpy(w)
    with ops.Session() as t:
        loaded = t.load_program(path)
        assert np.array_equal(_words(np.asarray(loaded.run(X))), _words(before))
        opt = t.optimizer("sgd", lr=0.1)
        loaded.fit(X, targets=T, optimizer=opt, loss="mse", epochs=1, batch_size=16, shuffle=False)
        again = t.numpy(loaded.params[0])
    assert not np.array_equal(_words(trained), _words(W))
    assert np.array_equal(_words(again), _words(trained))


def _refusal(cap, s, forward, backward, words, **kw):
    with pytest.raises(ValueError, match=words):
        s.custom_op(forward, backward, **kw)


def test_signature_mismatches_are_refused_by_name():
    from mojolearn import ops
    zeros = np.zeros((4, 3), np.float32)
    with ops.Session() as cap:
        forward, backward = _my_tanh(cap, (4, 3))
        x = cap.tensor(zeros)
        two_out = cap.capture(inputs=[x], outputs=[cap.tanh(x), x])
        g = cap.tensor(zeros)
        one_in = cap.capture(inputs=[g], outputs=[cap.tanh(g)])
        y, gs = cap.tensor(zeros), cap.tensor(np.zeros((1, 3), np.float32))
        wrong_grad = cap.capture(inputs=[y, gs], outputs=[gs])
        y2, g2 = cap.tensor(zeros), cap.tensor(zeros)
        wrong_out = cap.capture(inputs=[y2, g2], outputs=[cap.sum(g2)])
        weight = cap.tensor(np.ones((4, 3), np.float32))
        reads_param = cap.capture(inputs=[x], outputs=[cap.mul(x, weight)])
        with ops.Session(record=True) as s:
            _refusal(cap, s, forward, one_in, "the backward program takes 1 inputs; it must take 2", saved=[0])
            _refusal(cap, s, forward, wrong_grad, r"backward input 1 \(the gradient of output 0\) has shape", saved=[0])
            _refusal(cap, s, forward, wrong_out, "gradient 0 has shape", saved=[0])
            _refusal(cap, s, forward, backward, "saved entry 0 names forward output 3, which does not exist", saved=[3])
            _refusal(cap, s, two_out, one_in, "forward output 1 is neither returned nor saved", outputs=1)
            _refusal(cap, s, reads_param, backward, "reads 1 session tensors as parameters", saved=[0])
            with pytest.raises(ValueError, match="is not a custom operation of this session"):
                s.custom(99, s.tensor(zeros))


def test_call_refusals_and_first_order_only():
    from mojolearn import ops
    zeros = np.zeros((4, 3), np.float32)
    with ops.Session() as cap:
        forward, backward = _my_tanh(cap, (4, 3))
        with ops.Session(record=True) as s:
            op = s.custom_op(forward, backward, saved=[0])
            with pytest.raises(ValueError, match="takes 1 inputs, got 2"):
                s.custom(op, s.tensor(zeros), s.tensor(zeros))
            with pytest.raises(ValueError, match="the rank must match"):
                s.custom(op, s.tensor(np.zeros((2, 4, 3), np.float32)))
            w = s.tensor(zeros, requires_grad=True)
            loss = s.sum(s.custom(op, w))
            with pytest.raises(ValueError, match="first order only"):
                s.backward(loss, create_graph=True)
            with pytest.raises(ValueError, match="first order only"):
                s.hvp(loss, [w], [s.full((4, 3), 1.0)])
