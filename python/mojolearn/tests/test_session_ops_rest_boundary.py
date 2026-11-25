# SPDX-License-Identifier: Apache-2.0
"""The Python boundary of the ops.Session forms of Sequential's norms, prelu and mish, multi-head attention and
positional encoding, losses and raw state-space scans: PyTorch's parameter names, the admission of each argument,
the refusals in words, the returned shapes, and a captured Program saved and loaded again. The values and the
gradients are checked natively (checks/native_tests/session_ops_rest.mojo, against the Sequential layers' own
records and float64 references) and by the verify lanes ops-layer-* through device=."""
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


@pytest.mark.parametrize("name, expected", [
    ("batch_norm", ["self", "input", "running_mean", "running_var", "weight", "bias", "training", "momentum", "eps"]),
    ("group_norm", ["self", "input", "num_groups", "weight", "bias", "eps"]),
    ("instance_norm", ["self", "input", "running_mean", "running_var", "weight", "bias", "use_input_stats",
                       "momentum", "eps"]),
    ("prelu", ["self", "input", "weight"]),
    ("mish", ["self", "input"]),
    ("multi_head_attention", ["self", "query", "key", "value", "num_heads", "in_proj_weight", "in_proj_bias",
                              "out_proj_weight", "out_proj_bias", "key_lengths", "attn_mask", "is_causal",
                              "dropout_p", "training", "seed", "counter"]),
    ("positional_encoding", ["self", "x", "dropout_p", "training", "seed", "counter"]),
    ("smooth_l1_loss", ["self", "input", "target", "reduction", "beta"]),
    ("nll_loss", ["self", "input", "target", "weight", "ignore_index", "reduction"]),
    ("kl_div", ["self", "input", "target", "reduction", "log_target"]),
    ("binary_cross_entropy_with_logits", ["self", "input", "target", "weight", "reduction", "pos_weight"]),
    ("ctc_loss", ["self", "log_probs", "targets", "input_lengths", "target_lengths", "blank", "reduction",
                  "zero_infinity"]),
    ("selective_scan", ["self", "u", "delta", "A_log", "B", "C", "D"]),
    ("ssd_scan", ["self", "x", "dt", "A_log", "D"]),
    ("trapezoid_scan", ["self", "v", "k", "q", "dt", "A", "trap", "D", "heads"]),
])
def test_parameter_names(name, expected):
    from mojolearn import ops
    assert list(inspect.signature(getattr(ops.Session, name)).parameters) == expected


@needs_binding
def test_shapes():
    from mojolearn import ops
    rng = np.random.default_rng(0)
    with ops.Session() as s:
        x = s.tensor(rng.standard_normal((2, 4, 3), dtype=np.float32))
        w, b = s.tensor(np.ones(4, np.float32)), s.tensor(np.zeros(4, np.float32))
        rm, rv = s.state(np.zeros(4, np.float32)), s.state(np.ones(4, np.float32))
        assert s.shape(s.batch_norm(x, rm, rv, w, b, training=True)) == (2, 4, 3)
        assert s.shape(s.batch_norm(x, rm, rv, w, b)) == (2, 4, 3)
        assert s.shape(s.group_norm(x, 2, w, b)) == (2, 4, 3)
        assert s.shape(s.instance_norm(x)) == (2, 4, 3)
        assert s.shape(s.prelu(x, w)) == (2, 4, 3)
        assert s.shape(s.mish(x)) == (2, 4, 3)
        q = s.tensor(rng.standard_normal((2, 5, 4), dtype=np.float32))
        w_in, w_out = s.tensor(np.zeros((4, 12), np.float32)), s.tensor(np.zeros((4, 4), np.float32))
        assert s.shape(s.multi_head_attention(q, q, q, 2, w_in, None, w_out, None)) == (2, 5, 4)
        assert s.shape(s.positional_encoding(q)) == (2, 5, 4)
        p, t = s.tensor(np.zeros((3, 4), np.float32)), s.tensor(np.ones((3, 4), np.float32))
        for loss in (s.smooth_l1_loss(p, t), s.kl_div(p, t), s.binary_cross_entropy_with_logits(p, t, pos_weight=2.0),
                     s.nll_loss(p, s.ids(np.array([0, 3, 1])))):  # driver-loop: over four losses
            assert np.asarray(s.numpy(loss)).size == 1


@needs_binding
def test_refusals_in_words():
    from mojolearn import ops
    with ops.Session() as s:
        x = s.tensor(np.ones((2, 4, 3), np.float32))
        w = s.tensor(np.ones(4, np.float32))
        m, v = s.tensor(np.zeros(4, np.float32)), s.tensor(np.ones(4, np.float32))
        with pytest.raises(ValueError, match="weight and bias must be given together"):
            s.batch_norm(x, m, v, w, None)
        with pytest.raises(ValueError, match="running_mean and running_var must be given together"):
            s.batch_norm(x, m, None, training=True)
        with pytest.raises(ValueError, match="training=False normalizes with running_mean and running_var"):
            s.batch_norm(x, None, None)
        with pytest.raises(ValueError, match="make them with Session.state"):
            s.batch_norm(x, m, v, training=True)
        with pytest.raises(ValueError, match="momentum=None"):
            s.batch_norm(x, m, v, momentum=None)
        with pytest.raises(ValueError, match="num_groups"):
            s.group_norm(x, 3)
        with pytest.raises(ValueError, match="one per channel"):
            s.prelu(x, s.tensor(np.ones(3, np.float32)))
        q = s.tensor(np.ones((1, 3, 4), np.float32))
        k = s.tensor(np.ones((1, 2, 4), np.float32))
        w_in, w_out = s.tensor(np.zeros((4, 12), np.float32)), s.tensor(np.zeros((4, 4), np.float32))
        with pytest.raises(ValueError, match="key and value must be the same tensor"):
            s.multi_head_attention(q, k, q, 2, w_in, None, w_out, None)
        with pytest.raises(ValueError, match="attn_mask is not supported"):
            s.multi_head_attention(q, q, q, 2, w_in, None, w_out, None, attn_mask=q)
        with pytest.raises(TypeError, match="cross-attention .* takes in_proj_weight as a pair"):
            s.multi_head_attention(q, k, k, 2, w_in, None, w_out, None)
        with pytest.raises(ValueError, match="num_heads"):
            s.multi_head_attention(q, q, q, 3, w_in, None, w_out, None)
        with pytest.raises(ValueError, match="d_model must be even"):
            s.positional_encoding(s.tensor(np.ones((1, 2, 3), np.float32)))
        p, t = s.tensor(np.zeros((3, 4), np.float32)), s.tensor(np.ones((3, 4), np.float32))
        with pytest.raises(ValueError, match="reduction='none' is not supported"):
            s.smooth_l1_loss(p, t, reduction="none")
        with pytest.raises(ValueError, match="log_target=True is not supported"):
            s.kl_div(p, t, log_target=True)
        with pytest.raises(TypeError, match="pos_weight must be a float"):
            s.binary_cross_entropy_with_logits(p, t, pos_weight=t)
        with pytest.raises(ValueError, match="nll_loss: weight"):
            s.nll_loss(p, s.ids(np.array([0, 1, 2])), weight=t)
        with pytest.raises(ValueError, match="ignore_index is not supported"):
            s.nll_loss(p, s.ids(np.array([0, 1, 2])), ignore_index=0)
        with pytest.raises(ValueError, match=r"target class 4 at position 1 is outside \[0, 4\)"):
            s.nll_loss(p, s.ids(np.array([0, 4, 2])))
        with pytest.raises(ValueError, match="target must be class ids made with Session.ids"):
            s.nll_loss(p, t)
        lp = s.tensor(np.full((4, 2, 3), -1.0986123, np.float32))
        tg = s.ids(np.array([[1, 2], [0, 0]]))
        with pytest.raises(ValueError, match="must not hold the blank index"):
            s.ctc_loss(lp, tg, s.ids(np.array([4, 4])), s.ids(np.array([2, 1])))
        with pytest.raises(ValueError, match="input_lengths must be whole numbers from 0 to the 4 steps"):
            s.ctc_loss(lp, s.ids(np.array([[1, 2], [2, 1]])), s.ids(np.array([5, 4])), s.ids(np.array([2, 1])))


@needs_binding
def test_program_save_load_round_trip(tmp_path):
    from mojolearn import ops
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((2, 4, 2), dtype=np.float32)
    with ops.Session() as s:
        x = s.tensor(x0)
        w, b = s.tensor(np.full(4, 1.5, np.float32)), s.tensor(np.full(4, 0.25, np.float32))
        rm, rv = s.tensor(np.zeros(4, np.float32)), s.tensor(np.ones(4, np.float32))
        y = s.add(s.batch_norm(x, rm, rv, w, b), s.mish(s.group_norm(x, 2, w, b)))
        y = s.positional_encoding(s.prelu(y, w))
        expected = s.numpy(y)
        program = s.capture(inputs=[x], outputs=[y])
        path = str(tmp_path / "layers.program")
        program.save(path, verify=False)
        loaded = s.load_program(path)
        again = loaded.run(x0)
        again = again[0] if isinstance(again, (list, tuple)) else again
        assert np.asarray(again).tobytes() == np.asarray(expected).tobytes()
