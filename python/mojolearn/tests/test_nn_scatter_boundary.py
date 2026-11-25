# SPDX-License-Identifier: Apache-2.0
"""The Python boundary of ops.Session.scatter_add, segment_sum, segment_mean and segment_max (lane
nn-blocks-scatter, 2026-11-25): the parameter names, the admission of the index and segment ids (an ids() tensor),
the refusals in words, the returned shapes, and a captured Program saved and loaded again. The values, the fold
order and the gradients are checked natively (checks/native_tests/nn_scatter.mojo) and by the verify lane
ops-scatter-segment through device=."""
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
    ("scatter_add", ["self", "x", "dim", "index", "src"]),
    ("segment_sum", ["self", "data", "segment_ids", "num_segments"]),
    ("segment_mean", ["self", "data", "segment_ids", "num_segments"]),
    ("segment_max", ["self", "data", "segment_ids", "num_segments"]),
])
def test_parameter_names(name, expected):
    from mojolearn import ops
    assert list(inspect.signature(getattr(ops.Session, name)).parameters) == expected


@needs_binding
def test_shapes_and_admission():
    from mojolearn import ops
    with ops.Session() as s:
        data = s.tensor(np.ones((5, 3), dtype=np.float32))
        ids = s.ids(np.array([0, 3, 3, 1, 0]))
        for name in ("segment_sum", "segment_mean", "segment_max"):
            assert s.numpy(getattr(s, name)(data, ids, 4)).shape == (4, 3)
        x = s.tensor(np.zeros((2, 4, 3), dtype=np.float32))
        src = s.tensor(np.ones((2, 6, 3), dtype=np.float32))
        index = s.ids(np.zeros((2, 6, 3), dtype=np.int64))
        assert s.numpy(s.scatter_add(x, 1, index, src)).shape == (2, 4, 3)
        # the ids may be any integer dtype; a float array or a float tensor is refused
        with pytest.raises(TypeError, match="segment_ids must be an integer tensor made with Session.ids"):
            s.segment_sum(data, np.array([0, 3, 3, 1, 0]), 4)
        with pytest.raises(ValueError, match="segment_ids must be an integer tensor made with Session.ids"):
            s.segment_sum(data, s.tensor(np.zeros((5, 1), dtype=np.float32)), 4)
        with pytest.raises(TypeError, match="index must be an integer tensor made with Session.ids"):
            s.scatter_add(x, 1, np.zeros((2, 6, 3), dtype=np.int64), src)
        with pytest.raises(ValueError, match="data must be a floating-point tensor"):
            s.segment_mean(ids, ids, 4)


@needs_binding
def test_refusals_in_words():
    from mojolearn import ops
    with ops.Session() as s:
        data = s.tensor(np.ones((4, 2), dtype=np.float32))
        with pytest.raises(ValueError, match=r"segment_sum: segment id 7 at position 2 is outside \[0, 3\)"):
            s.segment_sum(data, s.ids(np.array([0, 1, 7, 2])), 3)
        with pytest.raises(ValueError, match=r"segment_max: segment id -1 at position 0 is outside \[0, 3\)"):
            s.segment_max(data, s.ids(np.array([-1, 1, 0, 2])), 3)
        with pytest.raises(ValueError, match="num_segments must be a positive integer, got -2"):
            s.segment_mean(data, s.ids(np.array([0, 1, 0, 1])), -2)
        with pytest.raises(TypeError, match="num_segments must be an integer"):
            s.segment_mean(data, s.ids(np.array([0, 1, 0, 1])), 2.5)
        with pytest.raises(ValueError, match=r"one id per row of data \(4 rows, 3 ids\)"):
            s.segment_sum(data, s.ids(np.array([0, 1, 0])), 2)
        x = s.tensor(np.zeros((3, 2), dtype=np.float32))
        src = s.tensor(np.ones((4, 2), dtype=np.float32))
        with pytest.raises(ValueError, match=r"scatter_add: index 3 at position 3 is outside \[0, 3\)"):
            s.scatter_add(x, 0, s.ids(np.array([[0, 1], [2, 3], [0, 0], [1, 1]])), src)
        with pytest.raises(ValueError, match="index must have the shape of src"):
            s.scatter_add(x, 0, s.ids(np.zeros((4, 1), dtype=np.int64)), src)
        with pytest.raises(ValueError, match="src must have x's size on every dimension except dim"):
            s.scatter_add(x, 0, s.ids(np.zeros((4, 3), dtype=np.int64)), s.tensor(np.ones((4, 3), dtype=np.float32)))
        with pytest.raises(ValueError, match="dim 2 is out of range for a tensor of 2 dimensions"):
            s.scatter_add(x, 2, s.ids(np.zeros((4, 2), dtype=np.int64)), src)


@needs_binding
def test_program_save_load_round_trip(tmp_path):
    from mojolearn import ops
    data = np.arange(15, dtype=np.float32).reshape(5, 3)
    seg = np.array([2, 0, 2, 2, 0])
    with ops.Session() as s:
        d = s.tensor(data)
        ids = s.ids(seg)
        base = s.tensor(np.zeros((3, 3), dtype=np.float32))
        index = s.ids(np.repeat(seg[:, None], 3, axis=1))
        out = s.add(s.segment_mean(d, ids, 3), s.scatter_add(base, 0, index, d))
        expected = s.numpy(out)
        program = s.capture(inputs=[d, ids, index], outputs=[out])
        path = str(tmp_path / "segments.program")
        program.save(path, verify=False)
        loaded = s.load_program(path)
        again = loaded.run(data, seg, np.repeat(seg[:, None], 3, axis=1))
        again = again[0] if isinstance(again, (list, tuple)) else again
        assert np.asarray(again).tobytes() == np.asarray(expected).tobytes()
