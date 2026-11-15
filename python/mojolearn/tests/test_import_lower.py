# SPDX-License-Identifier: Apache-2.0
"""The lowering of imported models (docs/plans/MODEL_IMPORT.md lane 2): the
Python boundary of the imported program.

Each shared fixture of tests/_onnx_fixture.py (written without ``onnx`` or
``torch``) is imported, run on fixed-seed inputs and compared word for word
with an eager ``ops.Session`` composition; a second run, the inputs by name
and ``save`` then ``Session.load_program`` give the same words. The
embedding program takes int32 ids by name, refuses float ids, and its saved
program keeps the ids input. The refusals (Resize, a dilated Conv, a
data-dependent Reshape, Gather by indices the model also reads as values)
come in one error naming every node, before anything is uploaded.

The mixed operations, grouped and depthwise Conv with SAME pads, the padded
pools, the embedding program against the eager composition, the BatchNorm
prologue and the Gemm ``transB`` weight are native checks
(checks/native_tests/core_import_lower.mojo).
Prints IMPORT_DIGEST lines to diff across CPU, Metal, NVIDIA and AMD."""
import hashlib

import numpy as np
import pytest

from mojolearn import _backend
from mojolearn import importers
from mojolearn.importers import UnsupportedModel
from mojolearn.tests import _onnx_fixture as fx

BATCH = 3  # odd and small: no board shape


def _has_entries():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        if not hasattr(b, "ops_session_call"):
            return False
        from mojolearn import ops
        with ops.Session() as s:
            return s.shape(s.erf(s.tensor(np.zeros((2, 3), dtype=np.float32)))) == (2, 3)
    except Exception:
        return False


needs_binding = pytest.mark.skipif(not _has_entries(),
                                   reason="the sequence binding has no capture, weight import or import kinds "
                                          "(rebuild it)")


def _ops():
    from mojolearn import ops
    return ops


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _same(got, want, what):
    g, w = _words(got).reshape(-1), _words(want).reshape(-1)
    assert got.shape == want.shape, (what, got.shape, want.shape)
    diff = np.flatnonzero(g != w)
    assert diff.size == 0, f"{what}: {diff.size} words differ, first at {diff[0]}"


def _init(f, name):
    t = f.initializers[name]
    return np.frombuffer(t.data, dtype="<f4").reshape(t.shape).copy()


def _row(a):
    return np.ascontiguousarray(a, dtype=np.float32).reshape(1, -1)


def _input(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# ------------------------------------------------------------ references --
# Each reference calls the session operations the lowering emits, in order,
# so the words must be equal.

def _ref_linear_ln_linear(s, f, x):
    h = s.linear(x, s.tensor(np.ascontiguousarray(_init(f, "w1").T)), s.tensor(_row(_init(f, "b1"))))
    h = s.layer_norm(h, s.tensor(_row(_init(f, "ln_scale"))), s.tensor(_row(_init(f, "ln_bias"))), eps=1e-5)
    return s.linear(h, s.tensor(np.ascontiguousarray(_init(f, "w2").T)), s.tensor(_row(_init(f, "b2"))))


def _ref_conv(s, f, x):
    h = s.conv2d(x, s.tensor(_init(f, "conv_w")), s.tensor(_row(_init(f, "conv_b"))), stride=1, padding=1)
    h = s.max_pool2d(s.relu(h), 2, stride=2)
    h = s.reshape(h, (BATCH, 64))
    return s.linear(h, s.tensor(np.ascontiguousarray(_init(f, "fc_w").T)), s.tensor(_row(_init(f, "fc_b"))))


def _ref_attention(s, f, x, t=5, d=8, heads=2):
    dh = d // heads

    def heads_of(w):
        return s.reshape(s.matmul(x, s.tensor(_init(f, w))), (BATCH, t, heads, dh))

    q = s.permute(heads_of("wq"), (0, 2, 1, 3))
    k = s.permute(heads_of("wk"), (0, 2, 3, 1))
    v = s.permute(heads_of("wv"), (0, 2, 1, 3))
    sc = s.mul(s.matmul(q, k), s.tensor(_init(f, "scale").reshape(1, 1)))
    p = s.softmax(s.add(sc, s.tensor(_init(f, "mask"))), axis=-1)
    o = s.reshape(s.permute(s.matmul(p, v), (0, 2, 1, 3)), (BATCH, t, d))
    return s.matmul(o, s.tensor(_init(f, "wo")))


def _bn_fold(s, f, c=4):
    sc, b = s.tensor(_row(_init(f, "bn_scale"))), s.tensor(_row(_init(f, "bn_bias")))
    mean, var = s.tensor(_row(_init(f, "bn_mean"))), s.tensor(_row(_init(f, "bn_var")))
    fs = s.div(sc, s.sqrt(s.add(var, s.full((1, 1), 1e-5))))
    ft = s.sub(b, s.mul(mean, fs))
    return s.reshape(fs, (1, c, 1, 1)), s.reshape(ft, (1, c, 1, 1))


def _ref_batchnorm(s, f, x):
    fs, ft = _bn_fold(s, f)
    return s.relu(s.add(s.mul(x, fs), ft))


def _ref_shape_fold(s, f, x):
    return s.relu(s.reshape(x, (BATCH, 6)))


def _ref_padded_maxpool(s, f, x):
    # the lowering's own composition: the border filled with the lowest float32, then the unpadded pool
    return s.relu(s.max_pool2d(s.pad(x, [0, 0, 1, 1, 0, 0, 1, 1], -3.4028234663852886e38), 3, stride=2))


CASES = {
    "linear_layernorm_linear": ((BATCH, 6), _ref_linear_ln_linear),
    "external_sharded": ((BATCH, 6), _ref_linear_ln_linear),
    "conv_relu_maxpool_flatten_gemm": ((BATCH, 3, 8, 8), _ref_conv),
    "attention_two_head_causal": ((BATCH, 5, 8), _ref_attention),
    "batchnorm_relu": ((BATCH, 4, 5, 5), _ref_batchnorm),
    "shape_fold": ((BATCH, 2, 3), _ref_shape_fold),
    "padded_maxpool": ((BATCH, 3, 7, 7), _ref_padded_maxpool),
}


@needs_binding
@pytest.mark.parametrize("name", sorted(CASES))
def test_fixture_equals_eager(tmp_path, name):
    shape, ref = CASES[name]
    f = fx.write_fixture(name, tmp_path)
    prog = importers.import_model(f.path, shapes={"x": shape})
    assert prog.inputs == [("x", shape, "float32")]
    want_out = f.output_shape(BATCH)
    assert [(o[0], o[1]) for o in prog.outputs] == [(k, want_out[k]) for k in want_out]
    x = _input(shape, 7)
    y = prog.run(x)
    again = prog.run(x)
    _same(again, y, f"{name}: a second run")
    ops = _ops()
    with ops.Session() as s:
        want = s.numpy(ref(s, f, s.tensor(x)))
    _same(y, want, f"{name}: the program against the eager composition")
    # by name too
    _same(prog.run(x=x), y, f"{name}: the inputs by name")
    path = tmp_path / f"{name}.mlprog"
    prog.save(path)
    with ops.Session() as s2:
        loaded = s2.load_program(path)
        _same(loaded.run(x), y, f"{name}: the saved program")
    print(f"IMPORT_DIGEST {name} {_digest(y)}")


EMB_VOCAB, EMB_T = 11, 4


@needs_binding
def test_embedding_program_ids_boundary(tmp_path):
    f = fx.write_fixture("embedding_linear", tmp_path)
    prog = importers.import_model(f.path, shapes={"ids": (BATCH, EMB_T)})
    assert prog.inputs == [("ids", (BATCH, EMB_T), "int64")]
    assert prog.input_kinds == (1,)
    assert [(o[0], o[1]) for o in prog.outputs] == [("y", (BATCH, EMB_T, 3))]
    ids = np.random.default_rng(13).integers(0, EMB_VOCAB, size=(BATCH, EMB_T))
    y = prog.run(ids)
    _same(prog.run(ids=ids.astype(np.int32)), y, "embedding_linear: int32 ids by name")
    with pytest.raises(TypeError, match="integer"):
        prog.run(ids.astype(np.float32))
    path = tmp_path / "embedding_linear.mlprog"
    prog.save(path)
    with _ops().Session() as s2:
        loaded = s2.load_program(path)
        assert loaded.input_kinds == (1,)
        _same(loaded.run(ids.reshape(BATCH, EMB_T)), y, "embedding_linear: the saved program")
    print(f"IMPORT_DIGEST embedding_linear {_digest(y)}")


# ------------------------------------------------------------ refusals ----

def _refused_model(directory):
    inits = [fx.weight("w", (4, 2, 3, 3), 51), fx.tensor("scales", "f32", (4,), (1.0, 1.0, 2.0, 2.0)),
             fx.weight("table", (10, 3), 52)]
    nodes = [fx.node("Conv", ["x", "w"], ["c"], name="conv_d2", kernel_shape=(3, 3), group=2, dilations=(2, 2),
                     pads=(1, 1, 1, 1)),
             fx.node("Resize", ["c", "", "scales"], ["r"], name="up", mode="nearest"),
             fx.node("Floor", ["y"], ["yf"], name="floor"),
             fx.node("Cast", ["yf"], ["target"], name="cast", to=7),
             fx.node("Reshape", ["x2", "target"], ["xr"], name="reshape_dd"),
             fx.node("Gather", ["table", "ids"], ["e"], name="emb", axis=0),
             fx.node("Cast", ["ids"], ["idf"], name="ids_values", to=1)]
    return fx.make_model(f"{directory}/refused.onnx", nodes,
                         [("x", "f32", (2, 4, 6, 6)), ("y", "f32", (2,)), ("x2", "f32", (2, 6)),
                          ("ids", "i64", (5,))],
                         [("r", "f32", (2, 4, 8, 8)), ("xr", "f32", (3, 4)), ("e", "f32", (5, 3)),
                          ("idf", "f32", (5,))], inits)


def test_refusals_list_every_node_before_any_upload(tmp_path, monkeypatch):
    # the refusal pass runs in the one onnx_import call before any upload
    # (autograd/import_lower.mojo); the session it was given is closed again
    from mojolearn.importers import _program

    opened = []
    real = _program.Session

    def tracked(*a, **k):
        s = real(*a, **k)
        opened.append(s)
        return s

    monkeypatch.setattr(_program, "Session", tracked)
    path = _refused_model(str(tmp_path))
    with pytest.raises(UnsupportedModel) as e:
        importers.import_model(path)
    names = [p[0] for p in e.value.problems]
    for node in ("conv_d2", "up", "reshape_dd", "emb"):
        assert node in names, (node, str(e.value))
    msg = str(e.value)
    assert "dilations=(2, 2)" in msg and "no kernel" in msg and "indices" in msg
    assert isinstance(e.value, ValueError)
    assert all(s._id is None for s in opened), "the refused import left its session open"


def test_supported_ops_lists_the_table():
    table = dict(importers.supported_ops())
    for op in ("Gemm", "Conv", "LayerNormalization", "BatchNormalization", "Softmax", "MatMul", "Gather"):
        assert op in table and table[op]
    assert "Resize" not in table
