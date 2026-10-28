# SPDX-License-Identifier: Apache-2.0
"""The intermediate graph every model reader fills and the lowering reads
(docs/plans/MODEL_IMPORT.md section 7), and the static shape resolution.

The vocabulary is ONNX's: op names, attribute names, NCHW images and
``[out, in]`` Gemm weights. ``resolve`` turns the inputs' resolved shapes
into every value's static shape and dtype, and folds the integer shape
subgraphs (``Shape``, ``Gather`` on shapes, ``Concat`` of ints, ``Range``,
``ConstantOfShape``, integer ``Reshape``/``Slice``/``Unsqueeze``/``Cast``)
into description integers. It never reads a floating-point element.
"""
# Engineering note: the resolution is Mojo's (autograd/import_resolve.mojo,
# onnx_resolve), on the handle of the graph the Mojo reader holds. The former
# Python rules are the test oracle in tests/_resolve_oracle.py. (The
# torch.export reader and its GraphIR-to-words path were removed 2026-10-22,
# lane purge-user-objects: a torch model is exported to ONNX first.)
from __future__ import annotations

from dataclasses import dataclass, field

# ------------------------------------------------------------------ dtypes --

# ONNX TensorProto.DataType codes the import reads, by IR name
ONNX_DTYPE = {1: "f32", 10: "f16", 16: "bf16", 11: "f64", 7: "i64", 6: "i32", 9: "bool"}
# every other code, by name, for refusal messages
ONNX_DTYPE_REFUSED = {
    0: "undefined", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 8: "string", 12: "uint32",
    13: "uint64", 14: "complex64", 15: "complex128", 17: "float8e4m3fn", 18: "float8e4m3fnuz",
    19: "float8e5m2", 20: "float8e5m2fnuz", 21: "uint4", 22: "int4", 23: "float4e2m1",
}
ITEMSIZE = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8, "i64": 8, "i32": 4, "bool": 1}
FLOAT_DTYPES = frozenset(("f32", "f16", "bf16", "f64"))
INT_DTYPES = frozenset(("i64", "i32", "bool"))

def onnx_dtype_name(code):
    """The readable name of an ONNX DataType code."""
    return _READER_NATIVE("dtype_name", code, None, _READER_CONTEXT)


# ------------------------------------------------------------------ errors --

class UnsupportedModel(ValueError):
    """The model uses operations, attributes, types or shapes the import does
    not support. ``problems`` lists every one, in graph order, as
    ``(node_or_value_name, op_or_None, reason)``."""

    def __init__(self, source, problems):
        return _READER_NATIVE("unsupported_construct", self, (source, problems), _READER_CONTEXT)


def _message(source, problems):
    return _READER_NATIVE("unsupported_message", source, problems, _READER_CONTEXT)


# ---------------------------------------------------------------- the IR ----

@dataclass(frozen=True)
class TensorInfo:
    """An initializer or a constant: name, dtype, static shape and its raw
    little-endian row-major bytes (``data`` is None for a graph value)."""
    name: str
    dtype: str
    shape: tuple
    data: object = None  # memoryview | None

    def __init__(self, name, dtype, shape, data=None):
        return _READER_NATIVE("record_construct", self, (name, dtype, shape, data), _READER_CONTEXT)


_ATTRS_DEFAULT = object()


@dataclass(frozen=True)
class Node:
    """One operation in ONNX's vocabulary. ``inputs`` holds "" for an absent
    optional input."""
    op: str
    domain: str
    name: str
    inputs: tuple
    outputs: tuple
    attrs: dict = field(default_factory=dict)

    def __init__(self, op, domain, name, inputs, outputs, attrs=_ATTRS_DEFAULT):
        return _READER_NATIVE("record_construct", self, (op, domain, name, inputs, outputs, attrs), _READER_CONTEXT)


@dataclass
class GraphIR:
    """A model as the lowering reads it: graph inputs and outputs as
    ``(name, dtype, shape)`` with symbolic axes as str, nodes in topological
    order, initializers by name, and the source for messages."""
    opset: int
    inputs: tuple
    outputs: tuple
    nodes: tuple
    initializers: dict
    source: str

    def __init__(self, opset, inputs, outputs, nodes, initializers, source):
        return _READER_NATIVE("record_construct", self, (opset, inputs, outputs, nodes, initializers, source), _READER_CONTEXT)


# -------------------------------------------------- tensor buffers ----

def bytes_view(data):
    """A NumPy uint8 view of a buffer (no copy), or None when it is empty."""
    return _READER_NATIVE("bytes_view", data, None, _READER_CONTEXT)


def buffer_address(a):
    return _READER_NATIVE("buffer_address", a, None, _READER_CONTEXT)


# ---------------------------------------------------- shape resolution ----

@dataclass
class Resolved:
    """What ``resolve`` knows about every value: its static shape, its dtype,
    the row-major integers of the values that fold statically, the nodes it
    has no shape rule for (left to the lowering's refusal pass) and the
    problems found (unresolved axes, data-dependent shapes)."""
    shapes: dict
    dtypes: dict
    folded: dict
    no_rule: tuple
    problems: tuple

    def __init__(self, shapes, dtypes, folded, no_rule, problems):
        return _READER_NATIVE("record_construct", self, (shapes, dtypes, folded, no_rule, problems), _READER_CONTEXT)


def resolve(g, shapes=None):
    """Every value's static shape and dtype from the inputs' resolved shapes,
    with the integer shape subgraphs folded. Never raises for a graph it
    cannot resolve: the problems (an input axis left open, a shape that
    depends on tensor values, inconsistent shapes) are in ``problems`` and
    the nodes with no shape rule in ``no_rule``, so the lowering can report
    them together with its own refusals in one error."""
    return _READER_NATIVE("resolve", g, shapes, _READER_CONTEXT)


def resolve_shapes(g, shapes=None):
    """Every value's static shape from the inputs' resolved shapes; integer
    shape subgraphs folded; raises UnsupportedModel on an unresolved axis or
    a data-dependent shape, by node name."""
    return _READER_NATIVE("resolve_shapes", g, shapes, _READER_CONTEXT)


class _Native:
    """The retained reader handle and its native buffer views."""

    def __init__(self, b, h, views=()):
        return _READER_NATIVE("reader_construct", self, (b, h, views), _READER_CONTEXT)

    def resolve(self, shapes=None):
        return _READER_NATIVE("reader_resolve", self, shapes, _READER_CONTEXT)


def _connect_reader():
    # glue: one import-time connection, also for manually constructed GraphIR.
    from ._native_reader_bridge import connect
    return connect(GraphIR, _Native, Node, TensorInfo, Resolved, UnsupportedModel,
                   _ATTRS_DEFAULT, ONNX_DTYPE, ONNX_DTYPE_REFUSED)


_READER_NATIVE, _READER_CONTEXT = _connect_reader()

# glue: the static table helper shares this connection without an import cycle.
from . import _table
_table._READER_NATIVE, _table._READER_CONTEXT = _READER_NATIVE, _READER_CONTEXT
