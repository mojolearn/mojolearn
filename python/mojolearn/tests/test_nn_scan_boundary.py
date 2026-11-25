# SPDX-License-Identifier: Apache-2.0
"""Public boundary of ops.Session.scan (lane nn-blocks-scan, 2026-11-25): what it returns, the refusals in the
user's words, and a captured scan saved and loaded. The numerics (a GRU cell against float64 and Session.gru, the
linear recurrence's closed form, lengths, every gradient against central differences, host versus device words)
are the native check checks/native_tests/nn_scan.mojo and the verify lane ops-scan-cell."""
import os

import numpy as np
import pytest

from mojolearn import ops

B, F, H = 2, 3, 4


def _cell(s):
    h = s.tensor(np.zeros((B, H), np.float32))
    x = s.tensor(np.zeros((B, F), np.float32))
    w = s.tensor(np.full((F, H), 0.1, np.float32))
    h2 = s.tanh(s.add(s.matmul(x, w), h))
    return s.capture(inputs=[h, x], outputs=[h2, h2])


def _xs(steps=5):
    return np.arange(steps * B * F, dtype=np.float32).reshape(steps, B, F) / 50.0


def test_returns_steps_first_and_the_final_state_in_the_init_state_form():
    with ops.Session() as s:
        cell = _cell(s)
        xs = s.tensor(_xs())
        ys, h_n = s.scan(cell, xs, s.tensor(np.zeros((B, H), np.float32)))
        assert isinstance(h_n, int)
        assert s.numpy(ys).shape == (5, B, H)
        np.testing.assert_array_equal(s.numpy(ys)[-1], s.numpy(h_n))
        ys2, finals = s.scan(cell, xs, [s.tensor(np.zeros((B, H), np.float32))])
        assert isinstance(finals, tuple) and len(finals) == 1
        np.testing.assert_array_equal(s.numpy(ys), s.numpy(ys2))


def test_reverse_keeps_time_order_and_ends_at_step_zero():
    with ops.Session() as s:
        cell = _cell(s)
        ys, h_n = s.scan(cell, s.tensor(_xs()), s.tensor(np.zeros((B, H), np.float32)), reverse=True)
        np.testing.assert_array_equal(s.numpy(ys)[0], s.numpy(h_n))


@pytest.mark.parametrize("bad, words", [
    ("count", "the cell carries 1 state tensors; init_state gives 2"),
    ("shape", "init_state 0 is (2, 5) but the cell's state 0 is (2, 4)"),
    ("params", "the cell reads 1 parameters; params names 0"),
    ("step", "each step of xs is (2, 7) but the cell's step input is (2, 3)"),
])
def test_refusals_name_the_mismatch(bad, words):
    with ops.Session() as s:
        cell = _cell(s)
        xs = s.tensor(_xs())
        h0 = s.tensor(np.zeros((B, H), np.float32))
        kwargs = {}
        state = h0
        if bad == "count":
            state = [h0, h0]
        elif bad == "shape":
            state = s.tensor(np.zeros((B, 5), np.float32))
        elif bad == "params":
            kwargs["params"] = []
        else:
            xs = s.tensor(np.zeros((5, B, 7), np.float32))
        with pytest.raises(ValueError, match="scan: " + words.replace("(", r"\(").replace(")", r"\)")):
            s.scan(cell, xs, state, **kwargs)


def test_a_cell_of_another_session_needs_params():
    with ops.Session() as c:
        cell = _cell(c)
        with ops.Session(record=True) as s:
            with pytest.raises(ValueError, match="pass params= with this Session's tensors"):
                s.scan(cell, s.tensor(_xs()), s.tensor(np.zeros((B, H), np.float32)))
            w = s.tensor(np.full((F, H), 0.1, np.float32), requires_grad=True)
            h0 = s.tensor(np.zeros((B, H), np.float32), requires_grad=True)
            ys, h_n = s.scan(cell, s.tensor(_xs()), h0, params=[w])
            s.backward(s.mean(s.mul(ys, ys)))
            assert np.all(np.isfinite(s.grad(w))) and np.any(s.grad(w) != 0)
            assert s.grad(h0).shape == (B, H)


def test_a_captured_scan_saves_and_loads(tmp_path):
    xs_np = _xs()
    h0_np = np.full((B, H), 0.25, np.float32)
    with ops.Session() as s:
        cell = _cell(s)
        xs = s.tensor(xs_np)
        h0 = s.tensor(h0_np)
        ys, h_n = s.scan(cell, xs, h0)
        eager = s.numpy(ys)
        prog = s.capture(inputs=[xs, h0], outputs=[ys, h_n])
        path = os.path.join(str(tmp_path), "scan.mlp")
        prog.save(path)
    with ops.Session() as d:
        got = d.load_program(path).run(xs_np, h0_np)
        np.testing.assert_array_equal(np.asarray(got[0]), eager)
