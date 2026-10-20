# SPDX-License-Identifier: Apache-2.0
"""The lowering of imported models (docs/plans/MODEL_IMPORT.md lane 2).

Each shared fixture of tests/_onnx_fixture.py (written without ``onnx`` or
``torch``) is imported, run on fixed-seed inputs and compared word for word
with an eager ``ops.Session`` composition of the same operations on the same
weights; a second run gives the same words; ``save`` then
``Session.load_program`` gives them again; the BatchNorm prologue equals the
eager fold; a ``Gemm`` ``transB`` weight equals the transposed weight.
Grouped and depthwise Conv, ``auto_pad`` SAME_UPPER / SAME_LOWER, padded
MaxPool and AveragePool (``count_include_pad`` 0 and 1) equal their eager
compositions word for word and a float64 NumPy reference to rounding. The
refusals (Resize, a dilated Conv, a data-dependent Reshape, Gather by an
index input) come in one error naming every node, before anything is
eager fold; a ``Gemm`` ``transB`` weight equals the transposed weight; the
embedding fixture (a ``Gather`` by an int64 ids input) equals the eager
``embedding`` on an ids tensor and on the ids array. The refusals (Resize,
Conv group=2, a data-dependent Reshape, Gather by indices the model also
reads as values) come in one error naming every node, before anything is
uploaded.
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
        if not (hasattr(b, "ops_open") and hasattr(b, "ops_capture") and hasattr(b, "ops_load_weights")):
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


@needs_binding
def test_batchnorm_prologue_is_the_eager_fold(tmp_path):
    f = fx.write_fixture("batchnorm_relu", tmp_path)
    prog = importers.import_model(f.path, shapes={"x": (BATCH, 4, 5, 5)})
    s = prog._s
    params = [s.numpy(h) for h in prog.params if s.shape(h) == (1, 4, 1, 1)]
    assert len(params) == 2, "the folded scale and shift are the program's parameters"
    ops = _ops()
    with ops.Session() as r:
        fs, ft = _bn_fold(r, f)
        want = [r.numpy(fs), r.numpy(ft)]
    _same(params[0], want[0], "the folded scale")
    _same(params[1], want[1], "the folded shift")
    print(f"IMPORT_DIGEST batchnorm_fold {_digest(*params)}")


@needs_binding
def test_gemm_transb_weight_is_the_transpose(tmp_path):
    f = fx.write_fixture("linear_layernorm_linear", tmp_path)
    prog = importers.import_model(f.path, shapes={"x": (BATCH, 6)})
    s = prog._s
    w1 = [s.numpy(h) for h in prog.params if s.shape(h) == (6, 5)]
    assert len(w1) == 1
    _same(w1[0], np.ascontiguousarray(_init(f, "w1").T), "the transposed Gemm weight")


def _mixed(directory):
    """x [batch, 7] -> Slice [:, 1:6] -> Gather columns (0, 2, 3) -> Clip(-0.5, 0.5) -> Pow 2 -> Add a [3]
    bias -> ReduceMean over axis 1 (keepdims 0) [batch]; the clipped values also go through Softmax
    (axis 1) and ArgMax (axis 1, keepdims 0) [batch]."""
    inits = [fx.tensor("starts", "i64", (1,), (1,)), fx.tensor("ends", "i64", (1,), (6,)),
             fx.tensor("axes", "i64", (1,), (1,)), fx.tensor("cols", "i64", (3,), (0, 2, 3)),
             fx.tensor("lo", "f32", (), (-0.5,)), fx.tensor("hi", "f32", (), (0.5,)),
             fx.tensor("two", "i64", (), (2,)), fx.weight("bias", (3,), 41), fx.tensor("mean_axes", "i64", (1,), (1,))]
    nodes = [fx.node("Slice", ["x", "starts", "ends", "axes"], ["xs"], name="slice"),
             fx.node("Gather", ["xs", "cols"], ["xg"], name="gather", axis=1),
             fx.node("Clip", ["xg", "lo", "hi"], ["xc"], name="clip"),
             fx.node("Pow", ["xc", "two"], ["xp"], name="pow"),
             fx.node("Add", ["xp", "bias"], ["xb"], name="bias"),
             fx.node("ReduceMean", ["xb", "mean_axes"], ["m"], name="mean", keepdims=0),
             fx.node("Softmax", ["xc"], ["p"], name="softmax", axis=1),
             fx.node("ArgMax", ["xc"], ["a"], name="argmax", axis=1, keepdims=0)]
    return fx._write(directory, "mixed.onnx", nodes, [("x", "f32", ("batch", 7))],
                     [("m", "f32", ("batch",)), ("p", "f32", ("batch", 3)), ("a", "i64", ("batch",))], inits,
                     lambda b: {"m": (b,), "p": (b, 3), "a": (b,)}, opset=18)


@needs_binding
def test_mixed_ops_equal_eager(tmp_path):
    f = _mixed(str(tmp_path))
    prog = importers.import_model(f.path, shapes={"x": (BATCH, 7)})
    assert prog.outputs == [("m", (BATCH,), "float32"), ("p", (BATCH, 3), "float32"), ("a", (BATCH,), "int32")]
    x = _input((BATCH, 7), 11)
    m, p, a = prog.run(x)
    assert a.dtype == np.int32 and m.shape == (BATCH,)
    ops = _ops()
    with ops.Session() as s:
        h = s.tensor(x)
        xs = s.split(h, [1, 5, 1], axis=1)[1]
        xg = s.concat([s.split(xs, [1, 4], axis=1)[0], s.split(xs, [2, 2, 1], axis=1)[1]], axis=1)
        xc = s.minimum(s.maximum(xg, s.tensor(np.full((1, 1), -0.5, np.float32))),
                       s.tensor(np.full((1, 1), 0.5, np.float32)))
        xb = s.add(s.pow_scalar(xc, 2.0), s.tensor(_row(_init(f, "bias"))))
        want_m = s.numpy(s.reshape(s.mean(xb, axis=(1,), keepdims=True), (1, BATCH))).reshape(BATCH)
        want_p = s.numpy(s.softmax(xc, axis=-1))
        want_a = s.numpy(s.reshape(s.argmax(xc, axis=1, keepdims=True, int32=True), (1, BATCH)))
    _same(m, want_m, "ReduceMean")
    _same(p, want_p, "Softmax")
    assert np.array_equal(a, want_a.view(np.int32).reshape(BATCH))
    print(f"IMPORT_DIGEST mixed {_digest(m, p, a)}")


# ------------------------------------------------- grouped conv and pads --
# float64 NumPy references of the ONNX definitions (test oracles only)

def _np_conv(x, w, b, stride, pads, groups):
    """ONNX Conv (2-D): pads (top, left, bottom, right), zeros."""
    x = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    n, c, h, wd = x.shape
    oc, cg, kh, kw = w.shape
    oh, ow = (h - kh) // stride + 1, (wd - kw) // stride + 1
    out = np.zeros((n, oc, oh, ow))
    ocg = oc // groups
    for o in range(oc):
        g = o // ocg
        xs = x[:, g * cg:(g + 1) * cg]
        for i in range(oh):
            for j in range(ow):
                win = xs[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
                out[:, o, i, j] = np.tensordot(win, w[o].astype(np.float64), axes=3)
        if b is not None:
            out[:, o] += b[o]
    return out


def _np_pool(x, k, stride, pads, mode, include_pad=0):
    """ONNX MaxPool / AveragePool (2-D, square kernel)."""
    n, c, h, w = x.shape
    oh = (h + pads[0] + pads[2] - k) // stride + 1
    ow = (w + pads[1] + pads[3] - k) // stride + 1
    out = np.zeros((n, c, oh, ow))
    x = x.astype(np.float64)
    for i in range(oh):
        for j in range(ow):
            h0, w0 = i * stride - pads[0], j * stride - pads[1]
            hs, ws = max(h0, 0), max(w0, 0)
            he, we = min(h0 + k, h), min(w0 + k, w)
            win = x[:, :, hs:he, ws:we]
            if mode == "max":
                out[:, :, i, j] = win.max(axis=(2, 3))
            else:
                out[:, :, i, j] = win.sum(axis=(2, 3)) / (k * k if include_pad else (he - hs) * (we - ws))
    return out


def _grouped_conv(directory):
    """x [batch, 4, 8, 9] -> Conv group=2, weight [6, 2, 3, 3], bias, strides 2, auto_pad
    SAME_UPPER (pads (0, 1, 1, 1): the odd total of the height at the end) [batch, 6, 4, 5] ->
    depthwise Conv group=6, weight [6, 1, 3, 3], pads 1, no bias [batch, 6, 4, 5]."""
    inits = [fx.weight("gw", (6, 2, 3, 3), 61), fx.weight("gb", (6,), 62), fx.weight("dw", (6, 1, 3, 3), 63)]
    nodes = [fx.node("Conv", ["x", "gw", "gb"], ["g"], name="conv_g2", kernel_shape=(3, 3), strides=(2, 2),
                     auto_pad="SAME_UPPER", group=2),
             fx.node("Conv", ["g", "dw"], ["y"], name="conv_dw", kernel_shape=(3, 3), pads=(1, 1, 1, 1), group=6)]
    return fx._write(directory, "grouped_conv.onnx", nodes, [("x", "f32", ("batch", 4, 8, 9))],
                     [("y", "f32", ("batch", 6, 4, 5))], inits, lambda b: {"y": (b, 6, 4, 5)})


@needs_binding
def test_grouped_conv_and_same_pads(tmp_path):
    shape = (BATCH, 4, 8, 9)
    f = _grouped_conv(str(tmp_path))
    prog = importers.import_model(f.path, shapes={"x": shape})
    x = _input(shape, 13)
    y = prog.run(x)
    assert y.shape == (BATCH, 6, 4, 5)
    ops = _ops()
    with ops.Session() as s:
        h = s.pad(s.tensor(x), [0, 0, 0, 1, 0, 0, 1, 1])
        h = s.conv2d(h, s.tensor(_init(f, "gw")), s.tensor(_row(_init(f, "gb"))), stride=2, padding=0, groups=2)
        want = s.numpy(s.conv2d(h, s.tensor(_init(f, "dw")), None, stride=1, padding=1, groups=6))
    _same(y, want, "grouped conv: the program against the eager composition")
    ref = _np_conv(x, _init(f, "gw"), _init(f, "gb"), 2, (0, 1, 1, 1), 2)
    ref = _np_conv(ref, _init(f, "dw"), None, 1, (1, 1, 1, 1), 6)
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)
    print(f"IMPORT_DIGEST grouped_conv {_digest(y)}")


def _padded_pools(directory):
    """x [batch, 3, 7, 8] -> MaxPool 3 strides 2 pads 1 [batch, 3, 4, 4]; AveragePool 3 strides 2
    auto_pad SAME_LOWER (pads (1, 1, 1, 0)), count_include_pad 0 [batch, 3, 4, 4]; AveragePool 3
    strides 2 pads 1, count_include_pad 1 [batch, 3, 4, 4]."""
    nodes = [fx.node("MaxPool", ["x"], ["m"], name="max_p1", kernel_shape=(3, 3), strides=(2, 2),
                     pads=(1, 1, 1, 1)),
             fx.node("AveragePool", ["x"], ["a0"], name="avg_same", kernel_shape=(3, 3), strides=(2, 2),
                     auto_pad="SAME_LOWER"),
             fx.node("AveragePool", ["x"], ["a1"], name="avg_incl", kernel_shape=(3, 3), strides=(2, 2),
                     pads=(1, 1, 1, 1), count_include_pad=1)]
    outs = [(o, "f32", ("batch", 3, 4, 4)) for o in ("m", "a0", "a1")]
    return fx._write(directory, "padded_pools.onnx", nodes, [("x", "f32", ("batch", 3, 7, 8))], outs, [],
                     lambda b: {o: (b, 3, 4, 4) for o in ("m", "a0", "a1")})


@needs_binding
def test_padded_pools(tmp_path):
    shape = (BATCH, 3, 7, 8)
    f = _padded_pools(str(tmp_path))
    prog = importers.import_model(f.path, shapes={"x": shape})
    x = _input(shape, 17)
    m, a0, a1 = prog.run(x)
    again = prog.run(x)
    for got, want, what in zip((m, a0, a1), again, ("max", "avg same", "avg include pad")):
        _same(got, want, f"{what}: a second run")
    ops = _ops()
    with ops.Session() as s:
        h = s.tensor(x)
        want_m = s.numpy(s.max_pool2d(s.pad(h, [0, 0, 1, 1, 0, 0, 1, 1], float("-inf")), 3, stride=2))
        c = [0, 0, 1, 1, 0, 0, 1, 0]
        total = s.conv2d(s.pad(h, c, 0.0), s.full((3, 1, 3, 3), 1.0), None, stride=2, padding=0, groups=3)
        count = s.conv2d(s.pad(s.full((1, 1, 7, 8), 1.0), c, 0.0), s.full((1, 1, 3, 3), 1.0), None, stride=2)
        want_a0 = s.numpy(s.div(total, count))
        want_a1 = s.numpy(s.avg_pool2d(s.pad(h, [0, 0, 1, 1, 0, 0, 1, 1], 0.0), 3, stride=2))
    _same(m, want_m, "MaxPool pads")
    _same(a0, want_a0, "AveragePool SAME_LOWER count_include_pad=0")
    _same(a1, want_a1, "AveragePool count_include_pad=1")
    np.testing.assert_array_equal(m, _np_pool(x, 3, 2, (1, 1, 1, 1), "max").astype(np.float32))
    np.testing.assert_allclose(a0, _np_pool(x, 3, 2, (1, 1, 1, 0), "avg", 0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(a1, _np_pool(x, 3, 2, (1, 1, 1, 1), "avg", 1), rtol=1e-5, atol=1e-6)
    print(f"IMPORT_DIGEST padded_pools {_digest(m, a0, a1)}")
EMB_VOCAB, EMB_T, EMB_D = 11, 4, 6


def _ref_embedding_linear(s, f, ids):
    """``ids``: an ids tensor (Session.ids) or the ids array (the caller-array form)."""
    e = s.reshape(s.embedding(s.tensor(_init(f, "table")), ids), (BATCH, EMB_T, EMB_D))
    m = s.matmul(e, s.tensor(_init(f, "proj")))
    return s.add(m, s.tensor(_row(_init(f, "proj_b"))))


@needs_binding
def test_embedding_fixture_equals_eager(tmp_path):
    f = fx.write_fixture("embedding_linear", tmp_path)
    prog = importers.import_model(f.path, shapes={"ids": (BATCH, EMB_T)})
    assert prog.inputs == [("ids", (BATCH, EMB_T), "int64")]
    assert prog.input_kinds == (1,)
    assert [(o[0], o[1]) for o in prog.outputs] == [("y", (BATCH, EMB_T, 3))]
    rng = np.random.default_rng(13)
    ids = rng.integers(0, EMB_VOCAB, size=(BATCH, EMB_T))
    other = rng.integers(0, EMB_VOCAB, size=(BATCH, EMB_T))
    y = prog.run(ids)
    _same(prog.run(ids), y, "embedding_linear: a second run")
    moved = prog.run(other)
    ops = _ops()
    with ops.Session() as s:
        want = s.numpy(_ref_embedding_linear(s, f, s.ids(ids)))
        want_arr = s.numpy(_ref_embedding_linear(s, f, ids))
        want_other = s.numpy(_ref_embedding_linear(s, f, s.ids(other)))
    _same(y, want, "embedding_linear: the program against the eager composition")
    _same(want_arr, want, "embedding_linear: the ids tensor against the ids array")
    _same(moved, want_other, "embedding_linear: new ids against the eager composition")
    _same(prog.run(ids=ids.astype(np.int32)), y, "embedding_linear: int32 ids by name")
    with pytest.raises(TypeError, match="integer"):
        prog.run(ids.astype(np.float32))
    path = tmp_path / "embedding_linear.mlprog"
    prog.save(path)
    with ops.Session() as s2:
        loaded = s2.load_program(path)
        assert loaded.input_kinds == (1,)
        _same(loaded.run(ids.reshape(BATCH, EMB_T)), y, "embedding_linear: the saved program")
    print(f"IMPORT_DIGEST embedding_linear {_digest(y)}")
    print(f"IMPORT_DIGEST embedding_linear_new_ids {_digest(moved)}")


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
    from mojolearn.importers import _program

    def no_session(*a, **k):
        raise AssertionError("the import opened a session before refusing the model")

    monkeypatch.setattr(_program, "Session", no_session)
    path = _refused_model(str(tmp_path))
    with pytest.raises(UnsupportedModel) as e:
        importers.import_model(path)
    names = [p[0] for p in e.value.problems]
    for node in ("conv_d2", "up", "reshape_dd", "emb"):
        assert node in names, (node, str(e.value))
    msg = str(e.value)
    assert "dilations=(2, 2)" in msg and "no kernel" in msg and "indices" in msg
    assert isinstance(e.value, ValueError)


def test_supported_ops_lists_the_table():
    table = dict(importers.supported_ops())
    for op in ("Gemm", "Conv", "LayerNormalization", "BatchNormalization", "Softmax", "MatMul", "Gather"):
        assert op in table and table[op]
    assert "Resize" not in table
