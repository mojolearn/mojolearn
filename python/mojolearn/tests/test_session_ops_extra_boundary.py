# SPDX-License-Identifier: Apache-2.0
"""The Python boundary of ops.Session.sequence_mask, topk, gather, logsumexp and cross_entropy with a target
distribution (sub-lane session-ops-extra, 2026-11-25): the parameter names, admission, the refusals in words, the
returned shapes and kinds, and a captured Program saved and loaded again. Values, tie order and gradients are
checked natively (checks/native_tests/session_ops_extra.mojo) and by lane ops-composed-cases (composed-extra.json)."""
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
    ("sequence_mask", ["self", "lengths", "max_length"]),
    ("topk", ["self", "x", "k", "axis", "largest", "sorted"]),
    ("gather", ["self", "x", "axis", "index"]),
    ("logsumexp", ["self", "x", "axis", "keepdims"]),
    ("cross_entropy", ["self", "logits", "codes", "reduction", "weight"]),
])
def test_parameter_names(name, expected):
    from mojolearn import ops
    assert list(inspect.signature(getattr(ops.Session, name)).parameters) == expected


@needs_binding
def test_shapes_and_kinds():
    from mojolearn import ops
    with ops.Session() as s:
        lens = s.tensor(np.array([2.0, 0.0, 3.0], dtype=np.float32))
        assert s.numpy(s.sequence_mask(lens, 4)).shape == (3, 4)
        x = s.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        values, indices = s.topk(x, 2, axis=1)
        assert s.numpy(values).shape == (2, 2, 4)
        assert s.numpy(indices).view(np.int32).max() <= 2
        # topk's indices are an integer tensor: gather and take accept them
        assert s.numpy(s.gather(x, 1, indices)).tobytes() == s.numpy(values).tobytes()
        assert s.numpy(s.logsumexp(x, 2)).shape == (2, 3)
        assert s.numpy(s.logsumexp(x, 0, keepdims=True)).shape == (1, 3, 4)
        z = s.tensor(np.zeros((3, 4), dtype=np.float32))
        t = s.tensor(np.full((3, 4), 0.25, dtype=np.float32))
        assert s.numpy(s.cross_entropy(z, t)).size == 1
        assert s.numpy(s.cross_entropy(z, t, weight=s.tensor(np.ones((3, 1), np.float32)))).size == 1


@needs_binding
def test_refusals_in_words():
    from mojolearn import ops
    with ops.Session() as s:
        x = s.tensor(np.ones((2, 3), dtype=np.float32))
        with pytest.raises(ValueError, match=r"topk: k must be in \[1, 3\]"):
            s.topk(x, 4)
        with pytest.raises(TypeError, match="gather: index must be an integer tensor made with Session.ids"):
            s.gather(x, 1, np.zeros((2, 3), dtype=np.int64))
        with pytest.raises(ValueError, match="gather: index must be an integer tensor made with Session.ids"):
            s.gather(x, 1, s.tensor(np.zeros((2, 3), dtype=np.float32)))
        with pytest.raises(ValueError, match=r"gather: index 3 at position 1 is outside \[0, 3\)"):
            s.gather(x, 1, s.ids(np.array([[0, 3], [1, 2]])))
        with pytest.raises(ValueError, match="gather: index must have x's size on every dimension except axis"):
            s.gather(x, 1, s.ids(np.zeros((3, 2), dtype=np.int64)))
        with pytest.raises(ValueError, match="sequence_mask: lengths must be a floating-point tensor"):
            s.sequence_mask(s.ids(np.array([1, 2])), 3)
        with pytest.raises(ValueError, match=r"sequence_mask: max_length must be an integer in \[1, 2\^24\)"):
            s.sequence_mask(s.tensor(np.ones((2, 1), np.float32)), 0)
        with pytest.raises(TypeError, match="logsumexp: axis must be one integer"):
            s.logsumexp(x, (0, 1))
        with pytest.raises(Exception, match="axis 2 is out of range"):
            s.logsumexp(x, 2)


@needs_binding
def test_program_save_load_round_trip(tmp_path):
    from mojolearn import ops
    data = np.array([[1.0, 5.0, 3.0, 5.0], [7.0, -1.0, 7.0, 2.0]], dtype=np.float32)
    lens = np.array([[1.0], [3.0]], dtype=np.float32)
    target = np.array([[0.7, 0.1, 0.1, 0.1], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    with ops.Session() as s:
        x, ln, t = s.tensor(data), s.tensor(lens), s.tensor(target)
        values, _ = s.topk(x, 2)
        out = s.add(s.sum(values), s.sum(s.logsumexp(x, 1)))
        out = s.add(out, s.cross_entropy(x, t, weight=s.reshape(s.sum(s.sequence_mask(ln, 4), axis=1), (2, 1))))
        expected = s.numpy(out)
        program = s.capture(inputs=[x, ln, t], outputs=[out])
        path = str(tmp_path / "extra.program")
        program.save(path, verify=False)
        loaded = s.load_program(path)
        again = loaded.run(data, lens, target)
        again = again[0] if isinstance(again, (list, tuple)) else again
        assert np.asarray(again).tobytes() == np.asarray(expected).tobytes()
