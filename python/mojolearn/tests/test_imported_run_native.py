# SPDX-License-Identifier: Apache-2.0
"""ImportedProgram.run is one native call (autograd/program_pyapi.mojo
`ops_imported_run_py`). These tests hold it word for word to the former
Python path, rebuilt here as an oracle (NumPy views in the session shapes,
`Session.run`, the outputs viewed and reshaped), and hold the refusals'
classes and wording."""
import numpy as np
import pytest


# ------------------------------------------------------------ ImportedProgram.run

def _import_ready():
    try:
        from mojolearn.tests.test_import_lower import _has_entries
        from mojolearn import _backend
        return _has_entries() and hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_imported_run")
    except Exception:
        return False


needs_import = pytest.mark.skipif(not _import_ready(), reason="the sequence binding has no ops_imported_run (rebuild it)")


def _former_run(prog, *arrays, **named):
    # the former ImportedProgram.run, verbatim but for the checks
    from mojolearn.importers._table import session_shape
    if named:
        arrays = tuple(named[k] for k, _, _ in prog.inputs)
    views = []
    for (_, shape, _), x in zip(prog.inputs, arrays):
        a = np.asarray(x)
        views.append(a.reshape(session_shape(shape)) if a.ndim < 2 else a)
    got = prog._s.run(prog, *views)
    got = got if isinstance(got, tuple) else (got,)
    outs = [(y.view(np.int32) if dtype == "int32" else y).reshape(shape) for (_, shape, dtype), y in zip(prog.outputs, got)]
    return outs[0] if len(outs) == 1 else tuple(outs)


def _same(got, want):
    got = got if isinstance(got, tuple) else (got,)
    want = want if isinstance(want, tuple) else (want,)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert isinstance(g, np.ndarray) and g.shape == w.shape and g.dtype == w.dtype, (g.shape, w.shape, g.dtype)
        assert np.array_equal(np.ascontiguousarray(g).view(np.uint32), np.ascontiguousarray(w).view(np.uint32))


def _vector_relu(directory, n=6):
    # x [6] -> Relu -> y [6]: a rank-1 model value (held as a (1, 6) row)
    from mojolearn.tests import _onnx_fixture as fx
    nodes = [fx.node("Relu", ["x"], ["y"], name="relu")]
    return fx._write(directory, "vector_relu.onnx", nodes, [("x", "f32", (n,))], [("y", "f32", (n,))], [],
                     lambda b: {"y": (n,)})


@needs_import
def test_imported_run_equals_former_path(tmp_path):
    from mojolearn import importers
    from mojolearn.tests import _onnx_fixture as fx
    from mojolearn.tests.test_import_lower import _mixed
    f = fx.write_fixture("linear_layernorm_linear", tmp_path)
    prog = importers.import_model(f.path, shapes={"x": (3, 6)})
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 6)).astype(np.float32)
    wide = rng.standard_normal((3, 12))
    _same(prog.run(x), _former_run(prog, x))
    _same(prog.run(x=x), _former_run(prog, x=x))
    _same(prog.run(x.tolist()), _former_run(prog, x.tolist()))
    _same(prog.run(wide[:, ::2]), _former_run(prog, wide[:, ::2]))  # a float64 strided view
    _same(prog.run(x.astype(np.int32)), _former_run(prog, x.astype(np.int32)))
    # rank-1 outputs and an int32 argmax output
    m = _mixed(str(tmp_path))
    mp = importers.import_model(m.path, shapes={"x": (3, 7)})
    xm = rng.standard_normal((3, 7)).astype(np.float32)
    _same(mp.run(xm), _former_run(mp, xm))
    # a rank-1 input, from an array, a reversed view and a list
    v = _vector_relu(str(tmp_path))
    vp = importers.import_model(v.path)
    xv = rng.standard_normal(6).astype(np.float32)
    _same(vp.run(xv), _former_run(vp, xv))
    _same(vp.run(xv[::-1]), _former_run(vp, xv[::-1]))
    _same(vp.run(xv.tolist()), _former_run(vp, xv.tolist()))


@needs_import
def test_imported_run_refusals(tmp_path):
    from mojolearn import importers
    from mojolearn.tests import _onnx_fixture as fx
    f = fx.write_fixture("linear_layernorm_linear", tmp_path)
    prog = importers.import_model(f.path, shapes={"x": (3, 6)})
    x = np.zeros((3, 6), np.float32)
    with pytest.raises(ValueError, match="either in order or by name"):
        prog.run(x, x=x)
    with pytest.raises(ValueError, match=r"the model has no input 'z' \(inputs: x\)"):
        prog.run(z=x)
    with pytest.raises(ValueError, match=r"the model takes 1 inputs, got 2"):
        prog.run(x, x)
    with pytest.raises(ValueError, match=r"input 'x' must have the shape \(3, 6\), got \(3, 5\)"):
        prog.run(np.zeros((3, 5), np.float32))
    with pytest.raises(ValueError, match=r"input 'x' must have the shape \(3, 6\), got \(18,\)"):
        prog.run(np.zeros(18, np.float32))
