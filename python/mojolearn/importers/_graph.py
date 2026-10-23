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

import os
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
    return ONNX_DTYPE.get(code) or ONNX_DTYPE_REFUSED.get(code) or f"data type {code}"


# ------------------------------------------------------------------ errors --

class UnsupportedModel(ValueError):
    """The model uses operations, attributes, types or shapes the import does
    not support. ``problems`` lists every one, in graph order, as
    ``(node_or_value_name, op_or_None, reason)``."""

    def __init__(self, source, problems):
        self.source = source
        self.problems = tuple(problems)
        super().__init__(_message(source, self.problems))


def _message(source, problems):
    src = os.path.basename(str(source)) or str(source)
    n = len(problems)
    nodes = all(op is not None for _, op, _ in problems)  # glue: the refusal entries of the error message
    what = ("node" if n == 1 else "nodes") if nodes else ("item" if n == 1 else "items")
    lines = [f"{n} {what} of {src} cannot be imported:"]
    for name, op, reason in problems:  # glue: one line per refusal entry of the error message
        lines.append(f"  {name} ({op}): {reason}" if op is not None else f"  {name}: {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the IR ----

@dataclass(frozen=True)
class TensorInfo:
    """An initializer or a constant: name, dtype, static shape and its raw
    little-endian row-major bytes (``data`` is None for a graph value)."""
    name: str
    dtype: str
    shape: tuple
    data: object = None  # memoryview | None


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


# -------------------------------------------------- tensor buffers ----

def bytes_view(data):
    """A NumPy uint8 view of a buffer (no copy), or None when it is empty."""
    from .._optional_numpy import require_numpy
    np = require_numpy("mojolearn.importers")
    raw = memoryview(data).cast("B") if data is not None else memoryview(b"")
    return np.frombuffer(raw, dtype=np.uint8) if raw.nbytes else None


def buffer_address(a):
    return 0 if a is None else int(a.ctypes.data)


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


def resolve(g, shapes=None):
    """Every value's static shape and dtype from the inputs' resolved shapes,
    with the integer shape subgraphs folded. Never raises for a graph it
    cannot resolve: the problems (an input axis left open, a shape that
    depends on tensor values, inconsistent shapes) are in ``problems`` and
    the nodes with no shape rule in ``no_rule``, so the lowering can report
    them together with its own refusals in one error."""
    native = getattr(g, "_native", None)
    if native is None:
        raise TypeError("resolve takes a graph read by mojolearn.importers.onnx.read")
    # a graph the Mojo reader holds (importers/onnx.py): resolved on its handle
    return native.resolve(shapes)


def resolve_shapes(g, shapes=None):
    """Every value's static shape from the inputs' resolved shapes; integer
    shape subgraphs folded; raises UnsupportedModel on an unresolved axis or
    a data-dependent shape, by node name."""
    r = resolve(g, shapes)
    if r.problems:
        raise UnsupportedModel(g.source, r.problems)
    return r.shapes
