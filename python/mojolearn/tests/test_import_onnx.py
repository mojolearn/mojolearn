# SPDX-License-Identifier: Apache-2.0
"""The ONNX reader and the import graph (docs/plans/MODEL_IMPORT.md lane 1).

Pure Python, no binding, no ``onnx`` and no ``torch``: every model is written
by tests/_onnx_fixture.py. Checks the round trip of every shared fixture
(ops, attributes, initializer bytes, inputs and outputs), static shape
resolution on a dynamic batch, the folding of a Shape/Gather/Concat target,
external data views, the topological sort, and the refusals by version,
domain, dtype and open axis, each listed in one error."""
import mmap
import os

import pytest

from mojolearn import importers
from mojolearn.importers import UnsupportedModel
from mojolearn.importers import onnx as onnx_reader
from mojolearn.importers._graph import (GraphIR, Node, TensorInfo, int_values, resolve, resolve_shapes)
from mojolearn.tests import _onnx_fixture as fx

ROUND_TRIP = ["linear_layernorm_linear", "conv_relu_maxpool_flatten_gemm", "attention_two_head_causal",
              "batchnorm_relu", "external_sharded", "shape_fold"]


@pytest.mark.parametrize("name", ROUND_TRIP)
def test_fixture_round_trip(tmp_path, name):
    f = fx.write_fixture(name, tmp_path)
    g = onnx_reader.read(f.path)
    assert isinstance(g, GraphIR)
    assert g.opset == 17 and g.source == f.path
    assert [n.op for n in g.nodes] == [op for op in f.ops if op != "Constant"]
    assert all(isinstance(n, Node) and n.domain == "" and n.name for n in g.nodes)
    assert g.inputs == tuple((nm, dt, tuple(shp)) for nm, dt, shp in f.inputs)
    assert g.outputs == tuple((nm, dt, tuple(shp)) for nm, dt, shp in f.outputs)
    for nm, t in f.initializers.items():
        got = g.initializers[nm]
        assert isinstance(got, TensorInfo)
        assert (got.dtype, got.shape) == (t.dtype, t.shape), nm
        assert bytes(got.data) == t.data, nm
    # every output shape on two batch sizes
    for b in (1, 7):
        shapes = resolve_shapes(g, {"x": (b,) + tuple(f.inputs[0][2][1:])})
        for out, shp in f.output_shape(b).items():
            assert shapes[out] == shp, (name, out, shapes[out], shp)


def test_attributes_and_types(tmp_path):
    g = onnx_reader.read(fx.write_fixture("conv_relu_maxpool_flatten_gemm", tmp_path).path)
    conv, _, pool, flat, gemm = g.nodes
    assert conv.attrs == {"kernel_shape": (3, 3), "pads": (1, 1, 1, 1), "strides": (1, 1)}
    assert pool.attrs["kernel_shape"] == (2, 2) and flat.attrs == {"axis": 1}
    assert gemm.attrs == {"transB": 1} and gemm.inputs == ("f", "fc_w", "fc_b")
    g = onnx_reader.read(fx.write_fixture("linear_layernorm_linear", tmp_path).path)
    ln = g.nodes[1]
    assert ln.attrs["axis"] == -1 and abs(ln.attrs["epsilon"] - 1e-5) < 1e-12


def test_intermediate_shapes_attention(tmp_path):
    g = onnx_reader.read(fx.write_fixture("attention_two_head_causal", tmp_path).path)
    s = resolve_shapes(g, {"x": (3, 5, 8)})
    assert s["qh"] == (3, 5, 2, 4) and s["qt"] == (3, 2, 5, 4) and s["kt"] == (3, 2, 4, 5)
    assert s["s"] == (3, 2, 5, 5) and s["sm"] == (3, 2, 5, 5) and s["o"] == (3, 2, 5, 4)
    assert s["om"] == (3, 5, 8) and s["y"] == (3, 5, 8)


def test_shape_subgraph_folds(tmp_path):
    g = onnx_reader.read(fx.write_fixture("shape_fold", tmp_path).path)
    assert "axes0" in g.initializers and int_values(g.initializers["axes0"]) == (0,)
    assert int_values(g.initializers["minus_one"]) == (-1,)
    r = resolve(g, {"x": (4, 2, 3)})
    assert not r.problems and not r.no_rule
    assert r.folded["s"] == (4, 2, 3) and r.folded["b"] == (4,) and r.folded["target"] == (4, -1)
    assert r.shapes["r"] == (4, 6) and r.dtypes["s"] == "i64" and r.dtypes["y"] == "f32"


def test_external_data_is_a_view_of_the_shard(tmp_path):
    f = fx.write_fixture("external_sharded", tmp_path)
    assert (tmp_path / "shard0.bin").stat().st_size == 4 * (30 + 5)
    assert (tmp_path / "shard1.bin").stat().st_size == 4 * (5 + 5 + 15 + 3)
    g = onnx_reader.read(f.path)
    w2 = g.initializers["w2"]
    assert isinstance(w2.data, memoryview) and isinstance(w2.data.obj, mmap.mmap)
    assert bytes(w2.data) == f.initializers["w2"].data
    # a location outside the model's directory is refused by name
    sub = tmp_path / "sub"
    sub.mkdir()
    t = fx.weight("w", (2, 2), 1, external="../escape.bin")
    path = fx.make_model(str(sub / "m.onnx"), [fx.node("MatMul", ["x", "w"], ["y"], name="mm")],
                         [("x", "f32", (2, 2))], [("y", "f32", (2, 2))], [t])
    with pytest.raises(UnsupportedModel) as e:
        onnx_reader.read(path)
    assert "w" == e.value.problems[0][0] and "outside the model's directory" in str(e.value)


def test_nodes_are_sorted_topologically(tmp_path):
    nodes = [fx.node("Relu", ["h"], ["y"], name="second"), fx.node("Sigmoid", ["x"], ["h"], name="first")]
    path = fx.make_model(str(tmp_path / "unsorted.onnx"), nodes, [("x", "f32", (2, 3))], [("y", "f32", (2, 3))])
    g = onnx_reader.read(path)
    assert [n.name for n in g.nodes] == ["first", "second"]
    assert resolve_shapes(g)["y"] == (2, 3)


def test_refuse_opset_12_by_version(tmp_path):
    path = fx.make_model(str(tmp_path / "old.onnx"), [fx.node("Relu", ["x"], ["y"], name="r")],
                         [("x", "f32", (2, 2))], [("y", "f32", (2, 2))], opset=12)
    with pytest.raises(UnsupportedModel) as e:
        onnx_reader.read(path)
    assert isinstance(e.value, ValueError)
    assert "opset 12" in str(e.value) and "version_converter" in str(e.value)


def test_refuse_ml_domain_by_domain(tmp_path):
    nodes = [fx.node("Relu", ["x"], ["h"], name="r"),
             fx.node("LinearClassifier", ["h"], ["y"], name="clf", domain="ai.onnx.ml", coefficients=(1.0, 2.0))]
    path = fx.make_model(str(tmp_path / "ml.onnx"), nodes, [("x", "f32", (2, 1))], [("y", "f32", (2, 1))],
                         extra_opsets=[("ai.onnx.ml", 3)])
    with pytest.raises(UnsupportedModel) as e:
        onnx_reader.read(path)
    assert e.value.problems == (("clf", "LinearClassifier",
                                 "domain 'ai.onnx.ml' is not supported (only the default ONNX domain is)"),)
    assert "clf (LinearClassifier)" in str(e.value)


def test_refuse_float8_initializer_by_name(tmp_path):
    t = fx.tensor("w8", "f8e4m3fn", (2, 2), raw=bytes(4))
    path = fx.make_model(str(tmp_path / "f8.onnx"), [fx.node("MatMul", ["x", "w8"], ["y"], name="mm")],
                         [("x", "f32", (2, 2))], [("y", "f32", (2, 2))], [t])
    with pytest.raises(UnsupportedModel) as e:
        onnx_reader.read(path)
    assert e.value.problems[0][0] == "w8" and "float8e4m3fn" in e.value.problems[0][2]


def test_refuse_unresolved_axis_by_input_name(tmp_path):
    g = onnx_reader.read(fx.write_fixture("linear_layernorm_linear", tmp_path).path)
    with pytest.raises(UnsupportedModel) as e:
        resolve_shapes(g)
    assert e.value.problems[0][0] == "x" and "open" in e.value.problems[0][2]
    assert "shapes={'x'" in str(e.value)
    with pytest.raises(UnsupportedModel) as e:
        resolve_shapes(g, {"x": (3, 7)})  # the fixed axis is 6
    assert "does not match" in str(e.value)
    with pytest.raises(UnsupportedModel) as e:
        resolve_shapes(g, {"x": (3, 6), "z": (1,)})
    assert e.value.problems[0][0] == "z"


def test_every_problem_in_one_error(tmp_path):
    nodes = [fx.node("MatMul", ["x", "w8"], ["h"], name="mm"),
             fx.node("TreeEnsembleRegressor", ["h"], ["y"], name="trees", domain="ai.onnx.ml")]
    path = fx.make_model(str(tmp_path / "many.onnx"), nodes, [("x", "f32", (2, 2))], [("y", "f32", (2, 1))],
                         [fx.tensor("w8", "u8", (2, 2), raw=bytes(4))], opset=11, extra_opsets=[("ai.onnx.ml", 3)])
    with pytest.raises(UnsupportedModel) as e:
        onnx_reader.read(path)
    names = [p[0] for p in e.value.problems]
    assert names == ["many.onnx", "w8", "trees"], names
    assert str(e.value).startswith("3 items of many.onnx cannot be imported:")


def test_data_dependent_reshape_is_refused_by_node(tmp_path):
    nodes = [fx.node("Reshape", ["x", "shp"], ["y"], name="dyn_reshape")]
    path = fx.make_model(str(tmp_path / "dyn.onnx"), nodes, [("x", "f32", (2, 3)), ("shp", "i64", (2,))],
                         [("y", "f32", (3, 2))])
    g = onnx_reader.read(path)
    r = resolve(g)
    assert len(r.problems) == 1 and r.problems[0][:2] == ("dyn_reshape", "Reshape")
    assert "data-dependent" in r.problems[0][2]
    with pytest.raises(UnsupportedModel) as e:
        resolve_shapes(g)
    assert str(e.value).startswith("1 node of dyn.onnx cannot be imported:")


def test_op_without_a_shape_rule_is_left_to_the_lowering(tmp_path):
    nodes = [fx.node("Resize", ["x"], ["y"], name="up"), fx.node("Relu", ["y"], ["z"], name="after")]
    path = fx.make_model(str(tmp_path / "resize.onnx"), nodes, [("x", "f32", (1, 1, 2, 2))],
                         [("z", "f32", (1, 1, 4, 4))])
    r = resolve(onnx_reader.read(path))
    assert r.no_rule == ("up",) and not r.problems and "z" not in r.shapes


def test_not_an_onnx_file(tmp_path):
    p = tmp_path / "junk.onnx"
    p.write_bytes(b"\xff\xff\xff")
    with pytest.raises(ValueError, match="not an ONNX model"):
        onnx_reader.read(str(p))


def test_supported_ops_is_a_list():
    assert isinstance(importers.supported_ops(), list)
