# SPDX-License-Identifier: Apache-2.0
"""The Mojo ONNX reader against the Python one (the test oracle).

``mojolearn.importers.onnx.read`` reads a file through the Mojo reader
(autograd/import_reader.mojo); ``tests/_onnx_reader_oracle.py`` is the Python
reader it replaced. For every fixture of tests/_onnx_fixture.py and every
refusal case, both readers must give the same GraphIR field for field (the
opset, the inputs and outputs, every node with its attributes, every
initializer's dtype, shape and bytes, the source), the same refusal (the
same problems in the same order, the same message), and ``resolve`` on both
graphs the same Resolved (shapes, dtypes, folded integers, nodes without a
rule, problems). Needs the x_sequence binding with the onnx_read entry."""
import pytest

from mojolearn import _backend
from mojolearn.importers import UnsupportedModel
from mojolearn.importers import onnx as native_reader
from mojolearn.importers._graph import GraphIR, Node, TensorInfo, resolve
from mojolearn.tests import _onnx_fixture as fx
from mojolearn.tests import _onnx_reader_oracle as oracle


def _has_reader():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "onnx_read")
    except Exception:  # noqa: BLE001  no binding in this install
        return False


pytestmark = pytest.mark.skipif(not _has_reader(), reason="the x_sequence binding has no ONNX reader")


# ------------------------------------------------------------- comparing --

def _tensor_key(t):
    assert isinstance(t, TensorInfo)
    return (t.name, t.dtype, tuple(t.shape), bytes(t.data))


def _attr_key(v):
    if isinstance(v, TensorInfo):
        return ("tensor", _tensor_key(v))
    if isinstance(v, tuple):
        return ("tuple", tuple(_attr_key(x) for x in v))
    return (type(v).__name__, v)


def _node_key(n):
    assert isinstance(n, Node)
    return (n.op, n.domain, n.name, tuple(n.inputs), tuple(n.outputs),
            [(k, _attr_key(v)) for k, v in n.attrs.items()])


def _graph_key(g):
    assert isinstance(g, GraphIR)
    return {"opset": g.opset, "inputs": g.inputs, "outputs": g.outputs,
            "nodes": [_node_key(n) for n in g.nodes],
            "initializers": [(k, _tensor_key(t)) for k, t in g.initializers.items()],
            "source": g.source}


def _read_both(path):
    """(kind, payload) of each reader: ("graph", g), ("unsupported", (problems, message)) or
    ("error", (type name, message))."""
    out = []
    for reader in (oracle.read, native_reader.read):
        try:
            out.append(("graph", reader(path)))
        except UnsupportedModel as e:
            out.append(("unsupported", (e.problems, str(e))))
        except (ValueError, OSError) as e:
            out.append(("error", (type(e).__name__, str(e))))
    return out


def _same_read(path):
    """Assert both readers agree on ``path``; return (oracle graph, native graph) or None."""
    (ka, a), (kb, b) = _read_both(path)
    assert ka == kb, (path, ka, a, kb, b)
    if ka != "graph":
        assert a == b, (path, a, b)
        return None
    ga, gb = _graph_key(a), _graph_key(b)
    for field in ga:
        assert ga[field] == gb[field], (path, field, ga[field], gb[field])
    return a, b


def _same_resolve(ga, gb, shapes):
    ra, rb = resolve(ga, shapes), resolve(gb, shapes)
    assert getattr(ga, "_native", None) is None and getattr(gb, "_native", None) is not None
    assert list(ra.shapes.items()) == list(rb.shapes.items()), (shapes, ra.shapes, rb.shapes)
    assert ra.dtypes == rb.dtypes, (shapes, ra.dtypes, rb.dtypes)
    assert ra.folded == rb.folded, (shapes, ra.folded, rb.folded)
    assert ra.no_rule == rb.no_rule, (shapes, ra.no_rule, rb.no_rule)
    assert ra.problems == rb.problems, (shapes, ra.problems, rb.problems)
    return rb


# -------------------------------------------------------------- fixtures --

@pytest.mark.parametrize("name", sorted(fx.FIXTURES))
def test_fixture_graph_and_resolve_match(tmp_path, name):
    f = fx.write_fixture(name, tmp_path)
    ga, gb = _same_read(f.path)
    _same_resolve(ga, gb, None)  # the open batch axis, as a problem
    iname, _, ishape = f.inputs[0]
    for b in (1, 7):
        rb = _same_resolve(ga, gb, {iname: (b,) + tuple(ishape[1:])})
        assert not rb.problems, rb.problems
        for out, shp in f.output_shape(b).items():
            assert rb.shapes[out] == shp, (name, out)
    _same_resolve(ga, gb, {iname: (3,) + tuple(ishape[1:]), "not_an_input": (1,)})
    _same_resolve(ga, gb, {iname: (0,) + tuple(ishape[1:])})
    _same_resolve(ga, gb, {iname: (2,)})


# ------------------------------------------------------- refusal cases ---

def _model(tmp_path, fname, nodes, inputs, outputs, inits=(), **kw):
    return fx.make_model(str(tmp_path / fname), nodes, inputs, outputs, list(inits), **kw)


def test_refuse_opset_12(tmp_path):
    _same_read(_model(tmp_path, "old.onnx", [fx.node("Relu", ["x"], ["y"], name="r")],
                      [("x", "f32", (2, 2))], [("y", "f32", (2, 2))], opset=12))


def test_refuse_ml_domain(tmp_path):
    nodes = [fx.node("Relu", ["x"], ["h"], name="r"),
             fx.node("LinearClassifier", ["h"], ["y"], name="clf", domain="ai.onnx.ml", coefficients=(1.0, 2.0))]
    _same_read(_model(tmp_path, "ml.onnx", nodes, [("x", "f32", (2, 1))], [("y", "f32", (2, 1))],
                      extra_opsets=[("ai.onnx.ml", 3)]))


def test_refuse_float8_and_uint8_and_many(tmp_path):
    t = fx.tensor("w8", "f8e4m3fn", (2, 2), raw=bytes(4))
    _same_read(_model(tmp_path, "f8.onnx", [fx.node("MatMul", ["x", "w8"], ["y"], name="mm")],
                      [("x", "f32", (2, 2))], [("y", "f32", (2, 2))], [t]))
    nodes = [fx.node("MatMul", ["x", "w8"], ["h"], name="mm"),
             fx.node("TreeEnsembleRegressor", ["h"], ["y"], name="trees", domain="ai.onnx.ml")]
    _same_read(_model(tmp_path, "many.onnx", nodes, [("x", "f32", (2, 2))], [("y", "f32", (2, 1))],
                      [fx.tensor("w8", "u8", (2, 2), raw=bytes(4))], opset=11, extra_opsets=[("ai.onnx.ml", 3)]))


def _ext_body(name, loc, nbytes, offset=None):
    """A float32 (2, 2) TensorProto body whose data sits in external file ``loc``."""
    body = (fx.f_packed_ints(1, (2, 2)) + fx.f_int(2, 1) + fx.f_str(8, name)
            + fx.f_bytes(13, fx.f_str(1, "location") + fx.f_str(2, loc)))
    if offset is not None:
        body += fx.f_bytes(13, fx.f_str(1, "offset") + fx.f_str(2, offset))
    if nbytes is not None:
        body += fx.f_bytes(13, fx.f_str(1, "length") + fx.f_str(2, str(nbytes)))
    return body + fx.f_int(14, 1)


def test_external_data_cases(tmp_path):
    f = fx.write_fixture("external_sharded", tmp_path)
    ga, gb = _same_read(f.path)
    assert bytes(gb.initializers["w2"].data) == f.initializers["w2"].data
    # a location outside the model's directory
    sub = tmp_path / "sub"
    sub.mkdir()
    t = fx.weight("w", (2, 2), 1, external="../escape.bin")
    _same_read(fx.make_model(str(sub / "m.onnx"), [fx.node("MatMul", ["x", "w"], ["y"], name="mm")],
                             [("x", "f32", (2, 2))], [("y", "f32", (2, 2))], [t]))
    data = fx.weight("w", (2, 2), 1).data
    (tmp_path / "short.bin").write_bytes(data)
    cases = {"abs.onnx": _ext_body("w", "/abs/w.bin", 16),
             "missing.onnx": _ext_body("w", "missing.bin", 16),
             "noloc.onnx": _ext_body("w", "", 16),
             "offset_ok.onnx": _ext_body("w", "short.bin", 16, offset=" 0_0 "),
             "badlen.onnx": _ext_body("w", "short.bin", 16, offset="1"),
             "badoff.onnx": _ext_body("w", "short.bin", None, offset="x1"),
             "wronglen.onnx": _ext_body("w", "short.bin", 12)}
    for fname, body in cases.items():
        _same_read(_graph_file(tmp_path, fname, [body]))


def _graph_file(tmp_path, fname, init_bodies, nodes=None, inputs=None, outputs=None):
    """A model of MatMul(x, w) whose initializers are the given TensorProto bodies."""
    nodes = nodes if nodes is not None else [fx.node("MatMul", ["x", "w"], ["y"], name="mm")]
    inputs = inputs if inputs is not None else [("x", "f32", (2, 2))]
    outputs = outputs if outputs is not None else [("y", "f32", (2, 2))]
    g = b"".join(fx.f_bytes(1, fx.node_proto(n)) for n in nodes) + fx.f_str(2, "fixture")
    g += b"".join(fx.f_bytes(5, b) for b in init_bodies)
    g += b"".join(fx.f_bytes(11, fx.value_info(*v)) for v in inputs)
    g += b"".join(fx.f_bytes(12, fx.value_info(*v)) for v in outputs)
    m = fx.f_int(1, 8) + fx.f_bytes(8, fx.f_str(1, "") + fx.f_int(2, 17)) + fx.f_bytes(7, g)
    path = str(tmp_path / fname)
    with open(path, "wb") as fh:
        fh.write(m)
    return path


def test_typed_fields_and_constants(tmp_path):
    inits = [fx.tensor("wf", "f32", (2, 2), (1.0, -2.0, 3.5, 0.25), typed=True),
             fx.tensor("ids", "i64", (3,), (0, -1, 1), typed=True)]
    i32 = (fx.f_packed_ints(1, (2,)) + fx.f_int(2, 6) + fx.f_str(8, "i32w") + fx.f_packed_ints(5, (7, -9)))
    f16 = (fx.f_packed_ints(1, (2,)) + fx.f_int(2, 10) + fx.f_str(8, "h16") + fx.f_packed_ints(5, (0x3C00, 0xBC00)))
    bools = (fx.f_packed_ints(1, (3,)) + fx.f_int(2, 9) + fx.f_str(8, "flags") + fx.f_packed_ints(5, (1, 0, 1)))
    empty = fx.f_packed_ints(1, (0, 3)) + fx.f_int(2, 1) + fx.f_str(8, "nothing")
    nodes = [fx.node("Constant", [], ["cf"], name="c_float", value_float=0.5),
             fx.node("Constant", [], ["cfs"], name="c_floats", value_floats=(1.5, 2.5)),
             fx.node("Constant", [], ["ci"], name="c_int", value_int=-3),
             fx.node("Constant", [], ["ct"], name="c_tensor", value=fx.tensor("v", "i64", (2,), (4, 5))),
             fx.node("MatMul", ["x", "wf"], ["m"], name="mm"),
             fx.node("Mul", ["m", "cf"], ["y"], name="mul"),
             fx.node("ConstantOfShape", ["ct"], ["z"], name="cos", value=fx.tensor("fill", "i64", (1,), (2,))),
             fx.node("Shape", ["z"], ["zs"], name="zshape"),
             fx.node("Range", ["ci", "ct", "ci"], ["rg"], name="range"),
             fx.node("Cast", ["ids"], ["idb"], name="tobool", to=9),
             fx.node("Softmax", ["y"], ["p"], name="sm", axis=-1),
             fx.node("Pad", ["x", "padv"], ["xp"], name="pad", mode="constant"),
             fx.node("Gemm", ["x", "wf"], ["gy"], name="g", transA=1, alpha=1.0)]
    padv = fx.tensor("padv", "i64", (4,), (0, 1, 0, 1))
    bodies = [fx.tensor_proto(t) for t in inits + [padv]] + [i32, f16, bools, empty]
    path = _graph_file(tmp_path, "typed.onnx", bodies, nodes=nodes, inputs=[("x", "f32", (2, 2))],
                       outputs=[("p", "f32", (2, 2)), ("z", "i64", (4, 5)), ("rg", "i64", ("r",)),
                                ("idb", "bool", (3,)), ("xp", "f32", (2, 4)), ("gy", "f32", (2, 2)),
                                ("zs", "i64", (2,))])
    ga, gb = _same_read(path)
    _same_resolve(ga, gb, None)


def test_graph_order_and_definition_problems(tmp_path):
    nodes = [fx.node("Relu", ["h"], ["y"], name="second"), fx.node("Sigmoid", ["x"], ["h"], name="first")]
    ga, gb = _same_read(_model(tmp_path, "unsorted.onnx", nodes, [("x", "f32", (2, 3))], [("y", "f32", (2, 3))]))
    _same_resolve(ga, gb, None)
    cyc = [fx.node("Relu", ["b"], ["a"], name="ra"), fx.node("Relu", ["a"], ["b"], name="rb"),
           fx.node("Relu", ["x"], ["y"], name="ok")]
    _same_read(_model(tmp_path, "cycle.onnx", cyc, [("x", "f32", (2,))], [("y", "f32", (2,))]))
    dup = [fx.node("Relu", ["x"], ["y"], name="one"), fx.node("Sigmoid", ["x"], ["y"], name="two"),
           fx.node("Relu", ["nope"], ["z"])]
    _same_read(_model(tmp_path, "dup.onnx", dup, [("x", "f32", (2,))], [("y", "f32", (2,)), ("w", "f32", (2,))]))
    ir3 = [fx.node("MatMul", ["x", "w"], ["y"], name="mm")]
    _same_read(_model(tmp_path, "ir3.onnx", ir3, [("x", "f32", ("n", 2)), ("w", "f32", (2, 2))],
                      [("y", "f32", ("n", 2))], [fx.weight("w", (2, 2), 3)]))


def test_resolve_refusals(tmp_path):
    nodes = [fx.node("Reshape", ["x", "shp"], ["y"], name="dyn_reshape")]
    ga, gb = _same_read(_model(tmp_path, "dyn.onnx", nodes, [("x", "f32", (2, 3)), ("shp", "i64", (2,))],
                               [("y", "f32", (3, 2))]))
    _same_resolve(ga, gb, None)
    nodes = [fx.node("Resize", ["x"], ["y"], name="up"), fx.node("Relu", ["y"], ["z"], name="after")]
    ga, gb = _same_read(_model(tmp_path, "resize.onnx", nodes, [("x", "f32", (1, 1, 2, 2))],
                               [("z", "f32", (1, 1, 4, 4))]))
    _same_resolve(ga, gb, None)
    inits = [fx.weight("w", (4, 2, 3, 3), 51), fx.tensor("scales", "f32", (4,), (1.0, 1.0, 2.0, 2.0)),
             fx.weight("table", (10, 3), 52)]
    nodes = [fx.node("Conv", ["x", "w"], ["c"], name="conv_g2", kernel_shape=(3, 3), group=2),
             fx.node("Resize", ["c", "", "scales"], ["r"], name="up", mode="nearest"),
             fx.node("Floor", ["y"], ["yf"], name="floor"),
             fx.node("Cast", ["yf"], ["target"], name="cast", to=7),
             fx.node("Reshape", ["x2", "target"], ["xr"], name="reshape_dd"),
             fx.node("Gather", ["table", "ids"], ["e"], name="emb", axis=0),
             fx.node("MatMul", ["x2", "x2"], ["bad"], name="inner"),
             fx.node("Transpose", ["x2"], ["xt"], name="tp", perm=(0, 0)),
             fx.node("Squeeze", ["x2", "ax0"], ["sq"], name="sq"),
             fx.node("Split", ["x2"], ["s1", "s2", "s3", "s4"], name="split", axis=1),
             fx.node("Concat", ["x2", "y"], ["cc"], name="cat", axis=0),
             fx.node("Gather", ["x2", "big"], ["gb"], name="oob", axis=1)]
    path = _model(tmp_path, "refused.onnx", nodes,
                  [("x", "f32", (2, 4, 6, 6)), ("y", "f32", (2,)), ("x2", "f32", (2, 6)), ("ids", "i64", (5,))],
                  [("r", "f32", (2, 4, 8, 8)), ("xr", "f32", (3, 4)), ("e", "f32", (5, 3)), ("xt", "f32", (6, 2))],
                  inits + [fx.tensor("ax0", "i64", (1,), (0,)), fx.tensor("big", "i64", (2,), (7, -9))])
    ga, gb = _same_read(path)
    _same_resolve(ga, gb, None)
    _same_resolve(ga, gb, {"x": (2, 4, 6, 6), "ids": (5,)})


def test_not_an_onnx_file(tmp_path):
    for k, body in enumerate((b"\xff\xff\xff", b"\x08", b"\x3a\x05ab", b"\x0f", b"")):
        p = tmp_path / f"junk{k}.onnx"
        p.write_bytes(body)
        _same_read(str(p))
    _same_read(str(tmp_path / "does_not_exist.onnx"))
