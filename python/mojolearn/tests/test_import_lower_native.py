# SPDX-License-Identifier: Apache-2.0
"""The Mojo lowering of imported models against the Python one (the oracle).

``mojolearn.import_model`` lowers a model in one ``onnx_import`` call
(autograd/import_lower.mojo); ``tests/_lower_oracle.py`` is the Python
lowering it replaced (``check`` + ``lower`` through ``Session`` calls, then
the capture). For every fixture of tests/_onnx_fixture.py, the models of
tests/test_import_lower.py (mixed ops, grouped conv with SAME pads, padded
pools, the refusals) and the sweeps below (unary, binary and comparison ops,
reductions, normalizations, 1-D conv, constants folded at import, float16 /
float64 / bfloat16 weights, a refusal per subset check), both imports must
give: the same refusal (the same problems in the same order, the same
message), or programs with the same inputs, outputs, input kinds and
parameter count, the same saved file byte for byte (the program's
description words and its parameter words) and the same run outputs word
for word. A GraphIR built in Python (the Python reader's, and a
torch.export program's) takes the words path (``onnx_from_words``) and must
give the oracle's program too. Needs the x_sequence binding with the
``onnx_import`` entry. Prints LOWER_PARITY lines."""
import hashlib

import numpy as np
import pytest

from mojolearn import _backend
from mojolearn import importers
from mojolearn.importers import UnsupportedModel
from mojolearn.tests import _lower_oracle as oracle
from mojolearn.tests import _onnx_fixture as fx

BATCH = 3  # odd and small: no board shape


def _has_entry():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        return hasattr(b, "onnx_import") and hasattr(b, "onnx_from_words") and hasattr(b, "ops_capture")
    except Exception:  # noqa: BLE001  no binding in this install
        return False


pytestmark = pytest.mark.skipif(not _has_entry(), reason="the x_sequence binding has no onnx_import entry")


# ------------------------------------------------------------- comparing --

def _attempt(fn, source, shapes):
    try:
        return "program", fn(source, shapes=shapes)
    except UnsupportedModel as e:
        return "unsupported", (e.problems, str(e))
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        return "error", (type(e).__name__, str(e))


def _words(a):
    return np.ascontiguousarray(a).view(np.uint32 if a.dtype.itemsize == 4 else np.uint8).reshape(-1)


def _inputs(prog, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _, shape, dtype in prog.inputs:
        if dtype in ("int64", "int32"):
            out.append(rng.integers(0, 3, size=shape).astype(np.int64))
        else:
            out.append(rng.standard_normal(shape).astype(np.float32))
    return out


def _digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _same_import(tmp_path, tag, source, shapes, oracle_source=None):
    """Import ``source`` through both lowerings and assert they agree."""
    ka, a = _attempt(oracle.import_model, oracle_source if oracle_source is not None else source, shapes)
    kb, b = _attempt(importers.import_model, source, shapes)
    assert ka == kb, (tag, ka, a, kb, b)
    if ka != "program":
        assert a == b, (tag, a, b)
        print(f"LOWER_PARITY {tag} {ka} {len(a[0]) if ka == 'unsupported' else a[0]}")
        return None
    assert a.inputs == b.inputs, (tag, a.inputs, b.inputs)
    assert a.outputs == b.outputs, (tag, a.outputs, b.outputs)
    assert a.input_kinds == b.input_kinds and a.input_shapes == b.input_shapes, tag
    assert a.output_shapes == b.output_shapes and len(a.params) == len(b.params), tag
    pa, pb = tmp_path / f"{tag}_oracle.mlprog", tmp_path / f"{tag}_native.mlprog"
    a.save(pa)
    b.save(pb)
    assert pa.read_bytes() == pb.read_bytes(), f"{tag}: the saved programs differ (description or parameter words)"
    xs = _inputs(a, 5)
    ya, yb = a.run(*xs), b.run(*xs)
    ya = ya if isinstance(ya, tuple) else (ya,)
    yb = yb if isinstance(yb, tuple) else (yb,)
    for k, (u, v) in enumerate(zip(ya, yb)):
        assert u.shape == v.shape and u.dtype == v.dtype, (tag, k, u.shape, v.shape)
        diff = np.flatnonzero(_words(u) != _words(v))
        assert diff.size == 0, f"{tag}: output {k}: {diff.size} words differ, first at {diff[0]}"
    print(f"LOWER_PARITY {tag} program {_digest(yb)}")
    return b


# -------------------------------------------------------------- fixtures --

@pytest.mark.parametrize("name", sorted(fx.FIXTURES))
def test_fixture_lowering_matches_oracle(tmp_path, name):
    f = fx.write_fixture(name, tmp_path)
    iname, _, ishape = f.inputs[0]
    shape = (BATCH,) + tuple(ishape[1:])
    _same_import(tmp_path, name, f.path, {iname: shape})
    _same_import(tmp_path, f"{name}_open", f.path, None)  # the open batch axis: a refusal
    _same_import(tmp_path, f"{name}_bad_shape", f.path, {iname: (2,)})
    _same_import(tmp_path, f"{name}_not_an_input", f.path, {iname: shape, "not_an_input": (1,)})


def test_fixture_graph_words_path_matches_oracle(tmp_path):
    # a GraphIR built in Python (the Python reader's) goes to Mojo as words
    from mojolearn.tests import _onnx_reader_oracle as py_reader
    for name in sorted(fx.FIXTURES):
        f = fx.write_fixture(name, tmp_path)
        iname, _, ishape = f.inputs[0]
        g = py_reader.read(f.path)
        assert getattr(g, "_native", None) is None
        _same_import(tmp_path, f"{name}_words", g, {iname: (BATCH,) + tuple(ishape[1:])}, oracle_source=f.path)


def test_lowering_test_models_match_oracle(tmp_path):
    from mojolearn.tests import test_import_lower as tl
    d = str(tmp_path)
    _same_import(tmp_path, "mixed", tl._mixed(d).path, {"x": (BATCH, 7)})
    _same_import(tmp_path, "grouped_conv", tl._grouped_conv(d).path, {"x": (BATCH, 4, 8, 9)})
    _same_import(tmp_path, "padded_pools", tl._padded_pools(d).path, {"x": (BATCH, 3, 7, 8)})
    _same_import(tmp_path, "refused", tl._refused_model(d), None)


# ------------------------------------------------------------- the sweeps --

def _ops_sweep(directory):
    """x [batch, 6]: the elementwise, comparison and reduction ops, Gemm forms and shape ops."""
    n = fx.node
    inits = [fx.weight("gw", (4, 6), 71), fx.weight("gb", (4,), 72), fx.weight("gw2", (6, 4), 73),
             fx.weight("gb2", (1, 4), 74), fx.tensor("two", "i64", (), (2,)), fx.tensor("zero", "f32", (), (0.0,)),
             fx.tensor("lo", "f32", (), (-0.25,)), fx.tensor("split", "i64", (2,), (2, 4)),
             fx.tensor("rs_axes", "i64", (1,), (1,)), fx.tensor("unsq", "i64", (1,), (0,)),
             fx.tensor("exp_shape", "i64", (3,), (2, BATCH, 6)), fx.weight("pw", (1, 6), 75, 0.25)]
    nodes = [n("Gemm", ["x", "gw", "gb"], ["g1"], name="gemm_tb", transB=1, alpha=0.5, beta=2.0),
             n("Gemm", ["x", "gw2", "gb2"], ["g2"], name="gemm_alpha", alpha=1.5),
             n("Neg", ["x"], ["neg"], name="neg"),
             n("LeakyRelu", ["x"], ["lrelu"], name="lrelu", alpha=0.1),
             n("Elu", ["x"], ["elu"], name="elu"),
             n("HardSigmoid", ["x"], ["hsig"], name="hsig"),
             n("Gelu", ["x"], ["gelu"], name="gelu", approximate="tanh"),
             n("Erf", ["x"], ["erf"], name="erf"),
             n("Abs", ["x"], ["ab"], name="abs"),
             n("Sqrt", ["ab"], ["sq"], name="sqrt"),
             n("Exp", ["x"], ["ex"], name="exp"),
             n("Log", ["ex"], ["lg"], name="log"),
             n("Reciprocal", ["ex"], ["rc"], name="recip"),
             n("Floor", ["x"], ["fl"], name="floor"), n("Ceil", ["x"], ["ce"], name="ceil"),
             n("Round", ["x"], ["ro"], name="round"), n("Sin", ["x"], ["si"], name="sin"),
             n("Cos", ["x"], ["co"], name="cos"), n("Sign", ["x"], ["sg"], name="sign"),
             n("HardSwish", ["x"], ["hsw"], name="hswish"), n("Softplus", ["x"], ["spl"], name="softplus"),
             n("Sigmoid", ["x"], ["sgm"], name="sigmoid"), n("Tanh", ["x"], ["th"], name="tanh"),
             n("Max", ["x", "neg", "zero"], ["mx"], name="max3"), n("Min", ["x", "neg"], ["mn"], name="min2"),
             n("Greater", ["x", "zero"], ["gt"], name="gt"), n("Less", ["x", "lo"], ["lt"], name="lt"),
             n("GreaterOrEqual", ["x", "pw"], ["ge"], name="ge"), n("LessOrEqual", ["x", "pw"], ["le"], name="le"),
             n("Equal", ["fl", "ce"], ["eq"], name="eq"), n("And", ["gt", "ge"], ["an"], name="and"),
             n("Or", ["lt", "le"], ["orr"], name="or"), n("Not", ["an"], ["nt"], name="not"),
             n("Where", ["gt", "x", "neg"], ["wh"], name="where"),
             n("Pow", ["x", "two"], ["p2"], name="pow_int"), n("Pow", ["ab", "pw"], ["pt"], name="pow_tensor"),
             n("Clip", ["x", "lo"], ["cl"], name="clip_lo"),
             n("ReduceSum", ["x", "rs_axes"], ["rsum"], name="rsum", keepdims=1),
             n("ReduceMax", ["x"], ["rmax"], name="rmax", axes=(1,), keepdims=0),
             n("ReduceMin", ["x", "rs_axes"], ["rmin"], name="rmin", keepdims=0),
             n("ReduceL2", ["x"], ["rl2"], name="rl2", keepdims=1),
             n("ReduceMean", ["x"], ["rmean"], name="rmean", keepdims=0),
             n("Softmax", ["x"], ["sm0"], name="softmax_0", axis=0),
             n("LogSoftmax", ["x"], ["lsm"], name="log_softmax"),
             n("Split", ["x", "split"], ["s_a", "s_b"], name="split", axis=1),
             n("Concat", ["s_b", "s_a", "s_a"], ["cat"], name="concat", axis=-1),
             n("Transpose", ["cat"], ["tr"], name="transpose"),
             n("Unsqueeze", ["x", "unsq"], ["ux"], name="unsqueeze"),
             n("Expand", ["ux", "exp_shape"], ["xp"], name="expand"),
             n("ArgMax", ["x"], ["am"], name="argmax", axis=1, keepdims=1),
             n("Identity", ["sq"], ["idn"], name="identity")]
    outs = ["g1", "g2", "lrelu", "elu", "hsig", "gelu", "erf", "lg", "rc", "ro", "si", "co", "sg", "hsw", "spl",
            "sgm", "th", "mx", "mn", "eq", "orr", "nt", "wh", "p2", "pt", "cl", "rsum", "rmax", "rmin", "rl2",
            "rmean", "sm0", "lsm", "tr", "xp", "am", "idn"]
    return fx.make_model(f"{directory}/ops_sweep.onnx", nodes, [("x", "f32", ("batch", 6))],
                         [(o, "i64" if o == "am" else "f32", ()) for o in outs], inits, opset=20)


def _norms_sweep(directory):
    """x [batch, 4, 6]: instance, group, RMS and layer norms, a 1-D grouped conv, pools on a 4-D view."""
    n = fx.node
    inits = [fx.weight("in_s", (4,), 81), fx.weight("in_b", (4,), 82), fx.weight("gn_s", (4,), 83),
             fx.weight("gn_b", (4,), 84), fx.weight("gg_s", (2,), 85), fx.weight("gg_b", (2,), 86),
             fx.weight("rms_s", (6,), 87), fx.weight("cw", (6, 2, 3), 88), fx.weight("cb", (6,), 89),
             fx.tensor("img", "i64", (4,), (0, 2, 2, 6))]
    nodes = [n("InstanceNormalization", ["x", "in_s", "in_b"], ["inn"], name="inorm", epsilon=1e-4),
             n("GroupNormalization", ["x", "gn_s", "gn_b"], ["gnn"], name="gnorm_c", num_groups=2),
             n("GroupNormalization", ["x", "gg_s", "gg_b"], ["ggn"], name="gnorm_g", num_groups=2),
             n("RMSNormalization", ["x", "rms_s"], ["rms"], name="rms", axis=-1),
             n("Conv", ["x", "cw", "cb"], ["c1"], name="conv1d", kernel_shape=(3,), pads=(1, 1), group=2),
             n("Reshape", ["x", "img"], ["x4"], name="as_image"),
             n("AveragePool", ["x4"], ["ap"], name="avgpool", kernel_shape=(2, 2), strides=(2, 2)),
             n("GlobalAveragePool", ["x4"], ["gap"], name="gap"),
             n("GlobalMaxPool", ["x4"], ["gmp"], name="gmp"),
             n("MaxPool", ["x4"], ["mp"], name="maxpool_same", kernel_shape=(2, 2), strides=(2, 2),
               auto_pad="SAME_UPPER")]
    outs = ["inn", "gnn", "ggn", "rms", "c1", "ap", "gap", "gmp", "mp"]
    return fx.make_model(f"{directory}/norms_sweep.onnx", nodes, [("x", "f32", ("batch", 4, 6))],
                         [(o, "f32", ()) for o in outs], inits, opset=23)


def _bf16_words(vals):
    import struct
    return [struct.unpack("<I", struct.pack("<f", v))[0] >> 16 for v in vals]


def _consts_sweep(directory):
    """x [batch, 5, 5]: constants folded at import (a constant Gather, ConstantOfShape), Trilu, Pad with
    a fill value, float16 / float64 / bfloat16 weights, a Gemm transB weight in bfloat16."""
    n = fx.node
    table = fx.weight("table", (4, 5), 91)
    inits = [table, fx.tensor("rows", "i64", (3,), (2, 0, -1)), fx.tensor("cs_shape", "i64", (2,), (1, 5)),
             fx.tensor("k1", "i64", (), (1,)), fx.tensor("pads", "i64", (6,), (0, 1, 0, 0, 0, 2)),
             fx.tensor("fill", "f32", (), (0.75,)),
             fx.tensor("h16", "f16", (5,), (0.5, -1.0, 2.0, 0.25, 1.5)),
             fx.tensor("d64", "f64", (5,), (0.1, 0.2, 0.3, 0.4, 0.5)),
             fx.tensor("b16", "bf16", (5,), _bf16_words((1.0, -0.5, 0.25, 2.0, -1.0))),
             fx.tensor("wb16", "bf16", (3, 5), _bf16_words(fx.values(15, 92, 0.5))),
             fx.tensor("flat", "i64", (2,), (-1, 5))]
    nodes = [n("Gather", ["table", "rows"], ["picked"], name="const_gather", axis=0),
             n("ReduceSum", ["picked"], ["psum"], name="const_sum", axes=(0,), keepdims=1),
             n("ConstantOfShape", ["cs_shape"], ["cs"], name="cos_value",
               value=fx.tensor("v", "f32", (1,), (0.125,))),
             n("Add", ["x", "psum"], ["a1"], name="add_folded"),
             n("Add", ["a1", "cs"], ["a2"], name="add_cos"),
             n("Trilu", ["a2", "k1"], ["tu"], name="triu", upper=1),
             n("Trilu", ["a2"], ["tl"], name="tril", upper=0),
             n("Pad", ["tu", "pads", "fill"], ["pd"], name="pad_fill"),
             n("Mul", ["x", "h16"], ["m16"], name="mul_f16"),
             n("Mul", ["m16", "d64"], ["m64"], name="mul_f64"),
             n("Add", ["m64", "b16"], ["mb"], name="add_bf16"),
             n("Reshape", ["mb", "flat"], ["mf"], name="flat"),
             n("Gemm", ["mf", "wb16"], ["gy"], name="gemm_bf16", transB=1)]
    outs = [("tl", "f32", ()), ("pd", "f32", ()), ("gy", "f32", ())]
    return fx.make_model(f"{directory}/consts_sweep.onnx", nodes, [("x", "f32", ("batch", 5, 5))], outs, inits)


def _refusal_sweep(directory):
    """One node per subset check, each refused; every text must match the oracle's, in order."""
    n = fx.node
    inits = [fx.weight("w", (4, 4, 3, 3), 101), fx.weight("ln_s", (5,), 102), fx.weight("bn", (4,), 103),
             fx.tensor("steps", "i64", (1,), (2,)), fx.tensor("st", "i64", (1,), (0,)), fx.tensor("en", "i64", (1,), (4,)),
             fx.tensor("ax", "i64", (1,), (3,)), fx.tensor("pads", "i64", (8,), (0, 0, 1, 1, 0, 0, 1, 1))]
    nodes = [n("Gelu", ["x"], ["o1"], name="gelu_bad", approximate="fast"),
             n("LayerNormalization", ["x", "ln_s"], ["o2"], name="ln_axis", axis=1),
             n("BatchNormalization", ["x", "bn", "bn", "o1", "bn"], ["o3"], name="bn_train", training_mode=1),
             n("Conv", ["x", "w"], ["o4"], name="conv_strides", kernel_shape=(3, 3), strides=(1, 2), dilations=(2, 1)),
             n("MaxPool", ["x"], ["o5"], name="pool_rect", kernel_shape=(2, 3), storage_order=1),
             n("AveragePool", ["x"], ["o6"], name="pool_pads", kernel_shape=(2, 2), pads=(2, 0, 0, 0)),
             n("Pad", ["x", "pads"], ["o7"], name="pad_reflect", mode="reflect"),
             n("Slice", ["x", "st", "en", "ax", "steps"], ["o8"], name="slice_steps"),
             n("ArgMax", ["x"], ["o9"], name="argmax_last", axis=1, select_last_index=1),
             n("GroupNormalization", ["x", "bn", "bn"], ["o10"], name="gn_groups", num_groups=3),
             n("Resize", ["x"], ["o11"], name="resize"),
             n("Cast", ["x"], ["o12"], name="cast_int", to=7),
             n("Dropout", ["x"], ["o13", "mask"], name="dropout_mask")]
    outs = [(f"o{k}", "f32", ()) for k in range(1, 12)] + [("o12", "i64", ()), ("o13", "f32", ()),
                                                          ("mask", "bool", ())]
    return fx.make_model(f"{directory}/refusal_sweep.onnx", nodes,
                         [("x", "f32", (2, 4, 6, 6)), ("xb", "bf16", (2, 3))], outs, inits, opset=20)


def test_sweeps_match_oracle(tmp_path):
    d = str(tmp_path)
    _same_import(tmp_path, "ops_sweep", _ops_sweep(d), {"x": (BATCH, 6)})
    _same_import(tmp_path, "norms_sweep", _norms_sweep(d), {"x": (BATCH, 4, 6)})
    _same_import(tmp_path, "consts_sweep", _consts_sweep(d), {"x": (BATCH, 5, 5)})
    _same_import(tmp_path, "refusal_sweep", _refusal_sweep(d), None)


def test_refusal_sweep_lists_every_check(tmp_path):
    with pytest.raises(UnsupportedModel) as e:
        importers.import_model(_refusal_sweep(str(tmp_path)))
    msg = str(e.value)
    names = [p[0] for p in e.value.problems]
    assert "xb" in names and "bfloat16 inputs" in msg
    for node in ("gelu_bad", "conv_strides", "pool_rect", "pad_reflect", "argmax_last", "resize", "cast_int"):
        assert node in names, (node, msg)


def test_torch_export_matches_oracle(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.LayerNorm(5), torch.nn.GELU(),
                                torch.nn.Linear(5, 3)).eval()
    torch.manual_seed(0)
    ep = torch.export.export(model, (torch.zeros(BATCH, 6),))
    _same_import(tmp_path, "torch_mlp", ep, None)
