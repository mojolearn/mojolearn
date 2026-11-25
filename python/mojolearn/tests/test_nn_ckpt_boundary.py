# SPDX-License-Identifier: Apache-2.0
"""Public boundary of ops.Session.checkpoint and Session.scan(checkpoint_every=k) (lane activation-ckpt, 2026-11-25):
what they return, the refusals in the user's words, and a captured checkpoint saved and loaded. The numerics (the
checkpointed gradients against the plain ones word for word, segments, chains, scans forward and reverse, lengths,
two states, Program.fit) are the native check checks/native_tests/nn_ckpt.mojo and the verify lane
ops-composed-cases (case file composed-checkpoint.json)."""
import os

import numpy as np
import pytest

from mojolearn import ops

B, F, H, T = 2, 3, 4, 5


def _segment(c, two=False):
    x = c.tensor(np.zeros((B, F), np.float32))
    w1 = c.tensor(np.full((F, H), 0.2, np.float32))
    w2 = c.tensor(np.full((H, F), -0.1, np.float32))
    h = c.tanh(c.matmul(x, w1))
    y = c.tanh(c.matmul(h, w2))
    return c.capture(inputs=[x], outputs=[y, h] if two else [y])


def _cell(c):
    h = c.tensor(np.zeros((B, H), np.float32))
    x = c.tensor(np.zeros((B, F), np.float32))
    w = c.tensor(np.full((F, H), 0.1, np.float32))
    h2 = c.tanh(c.add(c.matmul(x, w), h))
    return c.capture(inputs=[h, x], outputs=[h2, h2])


def _x():
    return np.arange(B * F, dtype=np.float32).reshape(B, F) / 10.0


def test_one_output_is_a_handle_and_several_a_tuple():
    with ops.Session() as c:
        one, two = _segment(c), _segment(c, two=True)
        x = c.tensor(_x())
        y = c.checkpoint(one, x)
        assert isinstance(y, int)
        assert c.numpy(y).shape == (B, F)
        got = c.checkpoint(two, x)
        assert isinstance(got, tuple) and len(got) == 2
        assert c.numpy(got[1]).shape == (B, H)
        np.testing.assert_array_equal(c.numpy(got[0]), c.numpy(y))


def test_a_checkpoint_returns_the_plain_forms_words():
    with ops.Session() as c:
        seg = _segment(c)
        with ops.Session(record=True) as s:
            w1 = s.tensor(np.full((F, H), 0.2, np.float32), requires_grad=True)
            w2 = s.tensor(np.full((H, F), -0.1, np.float32), requires_grad=True)
            x = s.tensor(_x(), requires_grad=True)
            y = s.checkpoint(seg, x, params=[w1, w2])
            y_plain = s.checkpoint(seg, x, params=[w1, w2], recompute=False)
            np.testing.assert_array_equal(s.numpy(y), s.numpy(y_plain))
            s.backward(s.sum(s.mul(y, y)))
            assert s.grad(w1).shape == (F, H)
            assert s.grad(x).shape == (B, F)


def test_a_checkpointed_scan_returns_the_plain_scans_words():
    with ops.Session() as c:
        cell = _cell(c)
        with ops.Session(record=True) as s:
            xs = s.tensor(np.arange(T * B * F, dtype=np.float32).reshape(T, B, F) / 50.0, requires_grad=True)
            h0 = s.tensor(np.zeros((B, H), np.float32), requires_grad=True)
            w = s.tensor(np.full((F, H), 0.1, np.float32), requires_grad=True)
            ys, h_n = s.scan(cell, xs, h0, params=[w], checkpoint_every=2)
            ys_plain, _ = s.scan(cell, xs, h0, params=[w])
            assert isinstance(h_n, int)
            np.testing.assert_array_equal(s.numpy(ys), s.numpy(ys_plain))
            s.backward(s.sum(s.mul(ys, ys)))
            assert s.grad(h0).shape == (B, H)


@pytest.mark.parametrize("bad, words", [
    ("count", "the segment takes 1 inputs"),
    ("shape", "input 0 is (2, 5) but the segment's input 0 is (2, 3)"),
    ("params", "the segment reads 2 parameters; params names 1"),
    ("counter", "counter must be >= 0"),
])
def test_refusals_name_the_mismatch(bad, words):
    with ops.Session() as c:
        seg = _segment(c)
        x = c.tensor(_x())
        kwargs = {}
        args = [x]
        if bad == "count":
            args = [x, x]
        elif bad == "shape":
            args = [c.tensor(np.zeros((B, 5), np.float32))]
        elif bad == "params":
            kwargs["params"] = [seg.params[0]]
        elif bad == "counter":
            kwargs["counter"] = -1
        with pytest.raises(ValueError, match=words.replace("(", r"\(").replace(")", r"\)")):
            c.checkpoint(seg, *args, **kwargs)


def test_a_segment_returning_its_input_is_refused():
    with ops.Session() as c:
        x = c.tensor(np.zeros((B, F), np.float32))
        seg = c.capture(inputs=[x], outputs=[c.tanh(x), x])
        with pytest.raises(ValueError, match="is its input 0 returned unchanged"):
            c.checkpoint(seg, c.tensor(_x()))


def test_a_nested_checkpoint_is_refused():
    with ops.Session() as c:
        inner = _segment(c)
        x = c.tensor(np.zeros((B, F), np.float32))
        outer = c.capture(inputs=[x], outputs=[c.checkpoint(inner, x)])
        with pytest.raises(ValueError, match="holds a checkpoint"):
            c.checkpoint(outer, c.tensor(_x()))


def test_checkpoint_every_refusals():
    with ops.Session() as c:
        h = c.tensor(np.zeros((B, H), np.float32))
        x = c.tensor(np.zeros((B, H), np.float32))
        passthrough = c.capture(inputs=[h, x], outputs=[c.tanh(c.add(h, x)), x])
        with ops.Session(record=True) as s:
            xs = s.tensor(np.zeros((T, B, H), np.float32), requires_grad=True)
            h0 = s.tensor(np.zeros((B, H), np.float32), requires_grad=True)
            with pytest.raises(ValueError, match="returns an input unchanged"):
                s.scan(passthrough, xs, h0, checkpoint_every=2)
            with pytest.raises(ValueError, match="checkpoint_every must be a positive"):
                s.scan(passthrough, xs, h0, checkpoint_every=-1)


def test_create_graph_through_a_checkpoint_is_refused():
    with ops.Session() as c:
        seg = _segment(c)
        with ops.Session(record=True) as s:
            w1 = s.tensor(np.full((F, H), 0.2, np.float32), requires_grad=True)
            w2 = s.tensor(np.full((H, F), -0.1, np.float32), requires_grad=True)
            y = s.checkpoint(seg, s.tensor(_x()), params=[w1, w2])
            with pytest.raises(ValueError, match="first order only"):
                s.backward(s.sum(y), create_graph=True)


def test_a_captured_checkpoint_saves_and_loads(tmp_path):
    with ops.Session() as c:
        seg = _segment(c)
        x = c.tensor(np.zeros((B, F), np.float32))
        prog = c.capture(inputs=[x], outputs=[c.checkpoint(seg, x)])
        want = prog.run(_x())
        path = os.path.join(tmp_path, "checkpointed.mlp")
        prog.save(path)
        with ops.Session() as d:
            loaded = d.load_program(path)
            np.testing.assert_array_equal(loaded.run(_x()), want)
