# SPDX-License-Identifier: Apache-2.0
"""Write small ONNX model files for the import tests, with no ``onnx`` and
no ``torch``: a protobuf encoder for the fields the importer reads (field
numbers of onnx/onnx.proto3), ``make_model`` to write a file, and the
fixtures the import lanes share (``FIXTURES`` / ``write_fixture``).

Weights are deterministic float32 words from a fixed-seed generator, so a
fixture's bytes are the same on every machine.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

DTYPE_CODE = {"f32": 1, "u8": 2, "i8": 3, "i32": 6, "i64": 7, "bool": 9, "f16": 10, "f64": 11, "bf16": 16,
              "f8e4m3fn": 17}
_FMT = {"f32": "f", "f64": "d", "i64": "q", "i32": "i", "bool": "B", "u8": "B", "i8": "b", "f16": "e",
        "f8e4m3fn": "B", "bf16": "H"}


# ------------------------------------------------------------ wire format --

def _varint(v):
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(num, wt):
    return _varint((num << 3) | wt)


def f_int(num, v):
    return _key(num, 0) + _varint(int(v))


def f_bytes(num, b):
    b = bytes(b)
    return _key(num, 2) + _varint(len(b)) + b


def f_str(num, s):
    return f_bytes(num, s.encode("utf-8"))


def f_float(num, x):
    return _key(num, 5) + struct.pack("<f", x)


def f_packed_ints(num, vals):
    return f_bytes(num, b"".join(_varint(int(v)) for v in vals))


def f_packed_floats(num, vals):
    return f_bytes(num, struct.pack(f"<{len(vals)}f", *vals))


# ------------------------------------------------------------- messages ----

@dataclass
class Tensor:
    """An initializer (or a Constant's value): ``data`` is its raw bytes.
    ``typed`` stores it in float_data/int64_data instead of raw_data;
    ``external`` names the file it goes into (set by make_model)."""
    name: str
    dtype: str
    shape: tuple
    data: bytes
    typed: bool = False
    external: str | None = None
    ext_offset: int = 0


def tensor(name, dtype, shape, values=None, raw=None, typed=False, external=None):
    """A Tensor from Python values (packed with the dtype's struct format) or raw bytes."""
    if raw is None:
        n = 1
        for d in shape:
            n *= d
        values = list(values) if values is not None else [0] * n
        assert len(values) == n, (name, len(values), shape)
        raw = struct.pack(f"<{n}{_FMT[dtype]}", *values)
    return Tensor(name, dtype, tuple(shape), bytes(raw), typed, external)


def tensor_proto(t):
    out = f_packed_ints(1, t.shape) if t.shape else b""
    out += f_int(2, DTYPE_CODE[t.dtype]) + f_str(8, t.name)
    if t.external is not None:
        out += f_bytes(13, f_str(1, "location") + f_str(2, t.external))
        out += f_bytes(13, f_str(1, "offset") + f_str(2, str(t.ext_offset)))
        out += f_bytes(13, f_str(1, "length") + f_str(2, str(len(t.data))))
        out += f_int(14, 1)
    elif t.typed and t.dtype == "f32":
        out += f_bytes(4, t.data)  # packed float_data is the raw little-endian words
    elif t.typed and t.dtype == "i64":
        n = len(t.data) // 8
        out += f_packed_ints(7, struct.unpack(f"<{n}q", t.data))
    else:
        out += f_bytes(9, t.data)
    return out


def value_info(name, dtype, shape):
    dims = b"".join(f_bytes(1, f_str(2, d) if isinstance(d, str) else f_int(1, d)) for d in shape)
    tt = f_int(1, DTYPE_CODE[dtype]) + f_bytes(2, dims)
    return f_str(1, name) + f_bytes(2, f_bytes(1, tt))


def attribute(name, v):
    out = f_str(1, name)
    if isinstance(v, Tensor):
        return out + f_int(20, 4) + f_bytes(5, tensor_proto(v))
    if isinstance(v, bool) or isinstance(v, int):
        return out + f_int(20, 2) + f_int(3, int(v))
    if isinstance(v, float):
        return out + f_int(20, 1) + f_float(2, v)
    if isinstance(v, str):
        return out + f_int(20, 3) + f_str(4, v)
    v = tuple(v)
    if all(isinstance(x, int) for x in v):
        return out + f_int(20, 7) + f_packed_ints(8, v)
    if all(isinstance(x, (int, float)) for x in v):
        return out + f_int(20, 6) + f_packed_floats(7, [float(x) for x in v])
    raise TypeError(f"attribute {name}: {v!r}")


@dataclass
class N:
    """A node: ``N("Gemm", ["x", "w", "b"], ["y"], transB=1)``."""
    op: str
    inputs: list
    outputs: list
    name: str = ""
    domain: str = ""
    attrs: dict = field(default_factory=dict)


def node(op, inputs, outputs, name="", domain="", **attrs):
    return N(op, list(inputs), list(outputs), name, domain, attrs)


def node_proto(n):
    out = b"".join(f_str(1, i) for i in n.inputs) + b"".join(f_str(2, o) for o in n.outputs)
    if n.name:
        out += f_str(3, n.name)
    out += f_str(4, n.op)
    if n.domain:
        out += f_str(7, n.domain)
    out += b"".join(f_bytes(5, attribute(k, v)) for k, v in n.attrs.items())
    return out


def make_model(path, nodes, inputs, outputs, initializers=(), opset=17, extra_opsets=(), ir_version=8):
    """Write a valid ONNX file at ``path``. ``inputs``/``outputs`` are
    ``(name, dtype, shape)`` with str axes symbolic; initializers with
    ``external`` set go into that file beside the model, appended in order."""
    base = os.path.dirname(os.path.abspath(path))
    ext_files = {}
    for t in initializers:
        if t.external is not None:
            buf = ext_files.setdefault(t.external, bytearray())
            t.ext_offset = len(buf)
            buf += t.data
    for name, buf in ext_files.items():
        with open(os.path.join(base, name), "wb") as f:
            f.write(bytes(buf))
    g = b"".join(f_bytes(1, node_proto(n)) for n in nodes) + f_str(2, "fixture")
    g += b"".join(f_bytes(5, tensor_proto(t)) for t in initializers)
    g += b"".join(f_bytes(11, value_info(*v)) for v in inputs)
    g += b"".join(f_bytes(12, value_info(*v)) for v in outputs)
    m = f_int(1, ir_version) + f_str(2, "mojolearn-test")
    m += f_bytes(8, f_str(1, "") + f_int(2, opset))
    m += b"".join(f_bytes(8, f_str(1, d) + f_int(2, v)) for d, v in extra_opsets)
    m += f_bytes(7, g)
    with open(path, "wb") as f:
        f.write(m)
    return path


# --------------------------------------------------------------- weights ---

def values(n, seed, scale=1.0):
    """n deterministic float32-representable values in [-scale, scale)."""
    out, s = [], (seed * 0x9E3779B97F4A7C15 + 1) % (1 << 64)
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(((s >> 40) / float(1 << 24) * 2.0 - 1.0) * scale)
    return struct.unpack(f"<{n}f", struct.pack(f"<{n}f", *out))


def weight(name, shape, seed, scale=0.5, **kw):
    n = 1
    for d in shape:
        n *= d
    return tensor(name, "f32", shape, values(n, seed, scale), **kw)


# -------------------------------------------------------------- fixtures ---

@dataclass
class Fixture:
    """A written fixture: its path, initializers by name, graph inputs and
    the output shape for a batch size."""
    path: str
    initializers: dict
    inputs: tuple
    outputs: tuple
    output_shape: object  # batch -> {output name: shape}
    ops: tuple


def _write(directory, fname, nodes, inputs, outputs, inits, out_shape, opset=17):
    path = make_model(os.path.join(directory, fname), nodes, inputs, outputs, inits, opset=opset)
    return Fixture(path, {t.name: t for t in inits}, tuple(inputs), tuple(outputs), out_shape,
                   tuple(n.op for n in nodes))


def linear_layernorm_linear(directory, external=None):
    """x [batch, 6] -> Gemm(transB) [batch, 5] -> LayerNormalization -> Gemm(transB) [batch, 3].
    ``external``: None, or a pair of file names (the first holds the first Gemm's weights, the
    second the rest), the sharded external-data form."""
    e0, e1 = external if external else (None, None)
    inits = [weight("w1", (5, 6), 1, external=e0), weight("b1", (5,), 2, external=e0),
             weight("ln_scale", (5,), 3, external=e1), weight("ln_bias", (5,), 4, external=e1),
             weight("w2", (3, 5), 5, external=e1), weight("b2", (3,), 6, external=e1)]
    nodes = [node("Gemm", ["x", "w1", "b1"], ["h"], name="fc1", transB=1),
             node("LayerNormalization", ["h", "ln_scale", "ln_bias"], ["hn"], name="ln", axis=-1, epsilon=1e-5),
             node("Gemm", ["hn", "w2", "b2"], ["y"], name="fc2", transB=1)]
    fname = "linear_ln_linear_ext.onnx" if external else "linear_ln_linear.onnx"
    return _write(directory, fname, nodes, [("x", "f32", ("batch", 6))], [("y", "f32", ("batch", 3))], inits,
                  lambda b: {"y": (b, 3)})


def conv_relu_maxpool_flatten_gemm(directory):
    """x [batch, 3, 8, 8] -> Conv 3x3 pad 1 [batch, 4, 8, 8] -> Relu -> MaxPool 2 [batch, 4, 4, 4]
    -> Flatten [batch, 64] -> Gemm(transB) [batch, 10]."""
    inits = [weight("conv_w", (4, 3, 3, 3), 11), weight("conv_b", (4,), 12),
             weight("fc_w", (10, 64), 13, 0.1), weight("fc_b", (10,), 14)]
    nodes = [node("Conv", ["x", "conv_w", "conv_b"], ["c"], name="conv", kernel_shape=(3, 3), pads=(1, 1, 1, 1),
                  strides=(1, 1)),
             node("Relu", ["c"], ["r"], name="relu"),
             node("MaxPool", ["r"], ["p"], name="pool", kernel_shape=(2, 2), strides=(2, 2)),
             node("Flatten", ["p"], ["f"], name="flatten", axis=1),
             node("Gemm", ["f", "fc_w", "fc_b"], ["y"], name="fc", transB=1)]
    return _write(directory, "conv_pool_gemm.onnx", nodes, [("x", "f32", ("batch", 3, 8, 8))],
                  [("y", "f32", ("batch", 10))], inits, lambda b: {"y": (b, 10)})


def attention_two_head_causal(directory, t=5, d=8, heads=2):
    """x [batch, t, d] -> Q, K, V projections -> two heads -> softmax(QK^T * scale + causal mask) V
    -> merged heads -> output projection [batch, t, d]. The mask is a [1, 1, t, t] initializer."""
    dh = d // heads
    mask = [0.0 if r >= c else -1e9 for r in range(t) for c in range(t)]
    inits = [weight("wq", (d, d), 21), weight("wk", (d, d), 22), weight("wv", (d, d), 23),
             weight("wo", (d, d), 24),
             tensor("heads_shape", "i64", (4,), (0, 0, heads, dh)),
             tensor("merge_shape", "i64", (3,), (0, 0, d)),
             tensor("scale", "f32", (), (1.0 / dh ** 0.5,)),
             tensor("mask", "f32", (1, 1, t, t), mask)]
    nodes = [node("MatMul", ["x", "wq"], ["q"], name="q_proj"),
             node("MatMul", ["x", "wk"], ["k"], name="k_proj"),
             node("MatMul", ["x", "wv"], ["v"], name="v_proj"),
             node("Reshape", ["q", "heads_shape"], ["qh"], name="q_heads"),
             node("Reshape", ["k", "heads_shape"], ["kh"], name="k_heads"),
             node("Reshape", ["v", "heads_shape"], ["vh"], name="v_heads"),
             node("Transpose", ["qh"], ["qt"], name="q_t", perm=(0, 2, 1, 3)),
             node("Transpose", ["kh"], ["kt"], name="k_t", perm=(0, 2, 3, 1)),
             node("Transpose", ["vh"], ["vt"], name="v_t", perm=(0, 2, 1, 3)),
             node("MatMul", ["qt", "kt"], ["s"], name="scores"),
             node("Mul", ["s", "scale"], ["ss"], name="scaled"),
             node("Add", ["ss", "mask"], ["sm"], name="masked"),
             node("Softmax", ["sm"], ["p"], name="softmax", axis=-1),
             node("MatMul", ["p", "vt"], ["o"], name="pv"),
             node("Transpose", ["o"], ["ot"], name="o_t", perm=(0, 2, 1, 3)),
             node("Reshape", ["ot", "merge_shape"], ["om"], name="merge"),
             node("MatMul", ["om", "wo"], ["y"], name="o_proj")]
    return _write(directory, "attention_causal.onnx", nodes, [("x", "f32", ("batch", t, d))],
                  [("y", "f32", ("batch", t, d))], inits, lambda b: {"y": (b, t, d)})


def batchnorm_relu(directory, c=4, hw=5):
    """x [batch, c, hw, hw] -> BatchNormalization (inference) -> Relu."""
    var = [abs(v) + 0.5 for v in values(c, 34)]
    inits = [weight("bn_scale", (c,), 31), weight("bn_bias", (c,), 32), weight("bn_mean", (c,), 33),
             tensor("bn_var", "f32", (c,), var)]
    nodes = [node("BatchNormalization", ["x", "bn_scale", "bn_bias", "bn_mean", "bn_var"], ["n"], name="bn",
                  epsilon=1e-5),
             node("Relu", ["n"], ["y"], name="relu")]
    return _write(directory, "batchnorm_relu.onnx", nodes, [("x", "f32", ("batch", c, hw, hw))],
                  [("y", "f32", ("batch", c, hw, hw))], inits, lambda b: {"y": (b, c, hw, hw)})


def shape_fold(directory):
    """x [batch, 2, 3] -> Reshape to [batch, -1] with the target built from Shape, Gather,
    Unsqueeze and Concat (a Constant node and a typed int64_data initializer), then Relu."""
    inits = [tensor("zero", "i64", (), (0,)), tensor("minus_one", "i64", (1,), (-1,), typed=True)]
    nodes = [node("Shape", ["x"], ["s"], name="shape"),
             node("Gather", ["s", "zero"], ["b"], name="batch_dim", axis=0),
             node("Constant", [], ["axes0"], name="axes_const", value_ints=(0,)),
             node("Unsqueeze", ["b", "axes0"], ["bu"], name="unsq"),
             node("Concat", ["bu", "minus_one"], ["target"], name="target", axis=0),
             node("Reshape", ["x", "target"], ["r"], name="reshape"),
             node("Relu", ["r"], ["y"], name="relu")]
    return _write(directory, "shape_fold.onnx", nodes, [("x", "f32", ("batch", 2, 3))],
                  [("y", "f32", ("batch", 6))], inits, lambda b: {"y": (b, 6)})


FIXTURES = {
    "linear_layernorm_linear": linear_layernorm_linear,
    "conv_relu_maxpool_flatten_gemm": conv_relu_maxpool_flatten_gemm,
    "attention_two_head_causal": attention_two_head_causal,
    "batchnorm_relu": batchnorm_relu,
    "external_sharded": lambda d: linear_layernorm_linear(d, external=("shard0.bin", "shard1.bin")),
    "shape_fold": shape_fold,
}


def write_fixture(name, directory):
    """Write fixture ``name`` into ``directory`` and return its Fixture."""
    return FIXTURES[name](str(directory))
