# SPDX-License-Identifier: Apache-2.0
"""mojolearn.ops.Session, the Python side: refusals, ndarray program
inputs through one buffer view, and the operation-name table. The session's
compositions (the two-expert mixture, losses and metrics on frozen
predictions, permutation and row gather, a recording session's gradient)
run natively in checks/native_tests/core_ops_session_*.mojo."""
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


def _ops():
    from mojolearn import ops
    return ops


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


def test_program_run_reads_ndarray_inputs_through_one_buffer_view():
    # Row N143: an ndarray input is described by one buffer export (address,
    # shape, strides, format). Strided, Fortran-order and wider-dtype inputs
    # give the same bits as their C-contiguous float32 copies; byte-swapped
    # input keeps its refusal.
    ops = _ops()
    rng = np.random.default_rng(4)
    w = rng.standard_normal((4, 3), dtype=np.float32)
    with ops.Session() as s:
        x = s.tensor(np.zeros((6, 4), np.float32))
        y = s.softmax(s.linear(x, s.tensor(w)))
        prog = s.capture(inputs=[x], outputs=[y])
        s.reset()
        base = rng.standard_normal((6, 8), dtype=np.float32)
        strided = base[:, ::2]
        assert not strided.flags.c_contiguous
        want = np.asarray(prog.run(np.ascontiguousarray(strided)))
        assert np.asarray(prog.run(strided)).tobytes() == want.tobytes()
        fortran = np.asfortranarray(strided)
        assert np.asarray(prog.run(fortran)).tobytes() == want.tobytes()
        wide = strided.astype(np.float64)
        assert np.asarray(prog.run(wide)).tobytes() == want.tobytes()
        with pytest.raises(TypeError):
            prog.run(strided.astype(">f4"))


def test_session_operation_dispatch_by_code_keeps_results_and_refusals():
    # Row N089: the operation name is looked up once in a native code table.
    # Known operations give the same values; an unknown name is refused by name.
    ops = _ops()
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    with ops.Session() as s:
        x = s.tensor(a)
        assert s.numpy(s.add(x, x)).tobytes() == (a + a).tobytes()
        assert s.numpy(s.relu(s.sub(x, s.scale(x, 2.0)))).tobytes() == np.zeros_like(a).tobytes()
        with pytest.raises(Exception, match="unknown native session operation: no_such_op"):
            ops._SESSION_CALL(s, "no_such_op", ())
