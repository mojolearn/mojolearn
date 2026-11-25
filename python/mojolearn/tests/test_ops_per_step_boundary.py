# SPDX-License-Identifier: Apache-2.0
"""The Python boundary of ops.Session's per-step blocks (lane rnn-parity-B, 2026-11-25): the parameter names of
the weighted losses and of the recurrent options, and their refusals in words. The values are checked natively
(checks/native_tests/ops_per_step.mojo) and by the verify lane ops-per-step-weighted-embedding."""
import inspect

import numpy as np
import pytest

from mojolearn import _backend


def _has_entries():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_session_call")
    except Exception:
        return False


needs_binding = pytest.mark.skipif(not _has_entries(),
                                   reason="the sequence binding has no ops_session_call entry (rebuild it)")


def _params(name):
    from mojolearn import ops
    return list(inspect.signature(getattr(ops.Session, name)).parameters)


@pytest.mark.parametrize("name, expected", [
    ("mse", ["self", "prediction", "target", "reduction", "weight"]),
    ("mae", ["self", "prediction", "target", "reduction", "weight"]),
    ("huber", ["self", "prediction", "target", "delta", "reduction", "weight"]),
    ("bce_with_logits", ["self", "logits", "target", "reduction", "weight"]),
    ("cross_entropy", ["self", "logits", "codes", "reduction", "weight", "probabilities"]),
    ("lstm", ["self", "x", "weight_ih", "weight_hh", "bias_ih", "bias_hh", "h_0", "c_0", "lengths", "reverse",
              "return_state"]),
    ("gru", ["self", "x", "weight_ih", "weight_hh", "bias_ih", "bias_hh", "h_0", "lengths", "reverse",
             "return_state"]),
    ("rnn", ["self", "x", "weight_ih", "weight_hh", "bias_ih", "bias_hh", "h_0", "lengths", "reverse",
             "return_state"]),
])
def test_parameter_names(name, expected):
    assert _params(name) == expected


def _gru_weights(s, inputs, hidden, gates=3):
    rng = np.random.default_rng(0)
    shapes = ((gates * hidden, inputs), (gates * hidden, hidden), (1, gates * hidden), (1, gates * hidden))
    return [s.tensor(rng.uniform(-0.3, 0.3, sh).astype(np.float32)) for sh in shapes]


@needs_binding
def test_weight_refusals():
    from mojolearn import ops
    with ops.Session() as s:
        p = s.tensor(np.zeros((6, 3), dtype=np.float32))
        y = s.tensor(np.ones((6, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="weight must hold"):
            s.mse(p, y, weight=s.tensor(np.ones((7, 1), dtype=np.float32)))
        with pytest.raises(ValueError, match="reduction must be"):
            s.bce_with_logits(p, y, reduction="none", weight=s.tensor(np.ones((6, 1), dtype=np.float32)))
        with pytest.raises(ValueError, match="floating-point"):
            s.mae(p, y, weight=s.ids(np.ones((6, 1), dtype=np.int32)))
        # accepted shapes: one per element, one per row
        assert s.shape(s.mse(p, y, weight=s.tensor(np.ones((6, 3), dtype=np.float32)))) == (1, 1)
        assert s.shape(s.huber(p, y, weight=s.tensor(np.ones((6,), dtype=np.float32)))) == (1, 1)


@needs_binding
def test_recurrent_option_refusals():
    from mojolearn import ops
    T, B, F, H = 4, 3, 2, 5
    with ops.Session() as s:
        x = s.tensor(np.zeros((T, B, F), dtype=np.float32))
        w = _gru_weights(s, F, H)
        with pytest.raises(ValueError, match="h_0 must be"):
            s.gru(x, *w, h_0=s.tensor(np.zeros((B, H + 1), dtype=np.float32)))
        with pytest.raises(ValueError, match="lengths must hold"):
            s.gru(x, *w, lengths=s.tensor(np.ones((B + 1,), dtype=np.float32)))
        lw = _gru_weights(s, F, H, gates=4)
        with pytest.raises(ValueError, match="c_0 must be"):
            s.lstm(x, *lw, c_0=s.tensor(np.zeros((B, 2), dtype=np.float32)))
        out, h_n = s.gru(x, *w, return_state=True)
        assert s.shape(out) == (T, B, H) and s.shape(h_n) == (B, H)
        out, h_n, c_n = s.lstm(x, *lw, h_0=s.tensor(np.zeros((1, B, H), dtype=np.float32)), return_state=True)
        assert s.shape(h_n) == (B, H) and s.shape(c_n) == (B, H)
