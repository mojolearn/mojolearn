# SPDX-License-Identifier: Apache-2.0
"""Read an ONNX model file into the import graph.

:func:`read` parses the file with a protobuf reader of mojolearn's own (no
``onnx`` package needed), checks that the model is opset 13 or later in the
default ONNX domain with tensor types the import supports, and returns the
graph with its nodes in topological order. The weights are not copied: each
initializer's bytes are a view into the file (or into its external data
file) until they are uploaded to the device.
"""
# cpu-route: model file parsed on the host. This module is the explicit
# input step before the runtime: it reads names, shapes, attributes and byte
# ranges from a file and never computes on tensor element values.
from __future__ import annotations

import heapq
import mmap
import os
import struct

from . import _onnx_proto as _pb
from ._graph import (FLOAT_DTYPES, ITEMSIZE, ONNX_DTYPE, GraphIR, Node, TensorInfo, UnsupportedModel,
                     onnx_dtype_name)

OPSET_MIN = 13
_DEFAULT_DOMAINS = ("", "ai.onnx")
_SUPPORTED_TYPES = "float32, float16, bfloat16, float64, int64, int32 and bool"


class _Problem(Exception):
    pass


def _map_file(path):
    """A read-only memoryview of the whole file (memory-mapped; the view keeps the map alive)."""
    with open(path, "rb") as f:
        try:
            return memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        except ValueError:  # an empty file cannot be mapped
            return memoryview(f.read())


class _External:
    """External data files beside the model, mapped once each."""

    def __init__(self, base):
        self.base = os.path.realpath(base)
        self.maps = {}

    def slice(self, entries, nbytes):
        loc = entries.get("location", "")
        if not loc:
            raise _Problem("external data without a location")
        if os.path.isabs(loc):
            raise _Problem(f"external data location '{loc}' is an absolute path (it must sit beside the model)")
        full = os.path.realpath(os.path.join(self.base, loc))
        if os.path.commonpath([full, self.base]) != self.base:
            raise _Problem(f"external data location '{loc}' is outside the model's directory")
        if full not in self.maps:
            if not os.path.isfile(full):
                raise _Problem(f"external data file '{loc}' not found beside the model")
            self.maps[full] = _map_file(full)
        buf = self.maps[full]
        try:
            off = int(entries.get("offset", "0") or 0)
            length = int(entries["length"]) if entries.get("length") else nbytes
        except ValueError:
            raise _Problem(f"external data offset/length of '{loc}' are not integers") from None
        if off < 0 or length != nbytes or off + length > len(buf):
            raise _Problem(f"external data '{loc}' [offset {off}, length {length}] does not hold the "
                           f"{nbytes} bytes the tensor needs (file size {len(buf)})")
        return buf[off:off + length]


def _join(chunks):
    return chunks[0] if len(chunks) == 1 else memoryview(b"".join(bytes(c) for c in chunks))  # cpu-route: model file parsed on the host (field occurrences)


def _typed_bytes(t, dtype):
    """The raw bytes of a tensor whose data sits in a typed repeated field.
    Exporters write weights as raw_data; the typed fields carry small
    constants, re-encoded here to the raw layout without arithmetic."""
    typed = t["typed"]
    if dtype in ("f32",) and 4 in typed:
        return _join([v for wt, v in typed[4]])  # cpu-route: model file parsed on the host (field occurrences)
    if dtype == "f64" and 10 in typed:
        return _join([v for wt, v in typed[10]])  # cpu-route: model file parsed on the host (field occurrences)
    if dtype in ("f16", "bf16", "i32", "bool") and 5 in typed:
        vals = []
        for wt, v in typed[5]:  # cpu-route: model file parsed on the host (field occurrences)
            vals.extend(_pb._packed_varints(v, wt))
        fmt = {"f16": "H", "bf16": "H", "i32": "i", "bool": "B"}[dtype]
        if fmt == "H":
            vals = [v & 0xFFFF for v in vals]  # cpu-route: model file parsed on the host (bit patterns re-encoded, no arithmetic on values)
        return memoryview(struct.pack(f"<{len(vals)}{fmt}", *vals))
    if dtype == "i64" and 7 in typed:
        vals = []
        for wt, v in typed[7]:  # cpu-route: model file parsed on the host (field occurrences)
            vals.extend(_pb._packed_varints(v, wt))
        return memoryview(struct.pack(f"<{len(vals)}q", *vals))
    return None


def _tensor_info(t, ext, name=None):
    """A TensorProto dict as a TensorInfo; raises _Problem with the reason."""
    name = name or t["name"]
    code = t["data_type"]
    dtype = ONNX_DTYPE.get(code)
    if dtype is None:
        raise _Problem(f"{onnx_dtype_name(code)} tensors are not supported ({_SUPPORTED_TYPES} are)")
    if t["segment"]:
        raise _Problem("segmented tensors are not supported")
    dims = tuple(t["dims"])
    if any(d < 0 for d in dims):  # cpu-route: model file parsed on the host (one per axis)
        raise _Problem(f"negative dimension in {dims}")
    numel = 1
    for d in dims:  # cpu-route: model file parsed on the host (one per axis)
        numel *= d
    nbytes = numel * ITEMSIZE[dtype]
    if t["data_location"] == 1:
        data = ext.slice(t["external"], nbytes)
    elif t["raw"] is not None:
        data = t["raw"]
    else:
        data = _typed_bytes(t, dtype)
        if data is None:
            if numel:
                raise _Problem("has no data")
            data = memoryview(b"")
    if len(data) != nbytes:
        raise _Problem(f"holds {len(data)} bytes, its shape {dims} of {dtype} needs {nbytes}")
    return TensorInfo(name, dtype, dims, data)


def _attr(a, ext):
    """An AttributeProto dict as an IR attribute value; raises _Problem."""
    if a["ref_attr_name"]:
        raise _Problem(f"attribute '{a['name']}' refers to a function attribute")
    if a["has_graph"]:
        raise _Problem(f"attribute '{a['name']}' holds a subgraph (control flow is not supported)")
    if a["has_sparse"]:
        raise _Problem(f"attribute '{a['name']}' holds a sparse tensor")
    if a["has_type_proto"]:
        raise _Problem(f"attribute '{a['name']}' holds a type")
    ty = a["type"]
    if ty == 0:  # writers older than IR 3 leave the type out: infer it from the field present
        ty = (_pb.ATTR_FLOAT if "f" in a else _pb.ATTR_INT if "i" in a else _pb.ATTR_STRING if "s" in a
              else _pb.ATTR_TENSOR if "t" in a else _pb.ATTR_FLOATS if a["floats"] else _pb.ATTR_INTS
              if a["ints"] else _pb.ATTR_STRINGS if a["strings"] else _pb.ATTR_TENSORS if a["tensors"] else 0)
    if ty == _pb.ATTR_FLOAT:
        return float(a.get("f", 0.0))
    if ty == _pb.ATTR_INT:
        return int(a.get("i", 0))
    if ty == _pb.ATTR_STRING:
        return a.get("s", b"").decode("utf-8", "replace")
    if ty == _pb.ATTR_TENSOR:
        if "t" not in a:
            raise _Problem(f"attribute '{a['name']}' has no tensor")
        return _tensor_info(a["t"], ext, a["t"]["name"] or a["name"])
    if ty == _pb.ATTR_FLOATS:
        return tuple(float(x) for x in a["floats"])  # cpu-route: model file parsed on the host (attribute list)
    if ty == _pb.ATTR_INTS:
        return tuple(int(x) for x in a["ints"])  # cpu-route: model file parsed on the host (attribute list)
    if ty == _pb.ATTR_STRINGS:
        return tuple(s.decode("utf-8", "replace") for s in a["strings"])  # cpu-route: model file parsed on the host (attribute list)
    if ty == _pb.ATTR_TENSORS:
        return tuple(_tensor_info(t, ext, t["name"] or a["name"]) for t in a["tensors"])  # cpu-route: model file parsed on the host (attribute list)
    raise _Problem(f"attribute '{a['name']}' has an unsupported type {ty}")


def _constant_tensor(name, attrs):
    """The tensor a Constant node holds, from whichever value attribute it sets."""
    if "value" in attrs and isinstance(attrs["value"], TensorInfo):
        t = attrs["value"]
        return TensorInfo(name, t.dtype, t.shape, t.data)
    if "value_float" in attrs:
        return TensorInfo(name, "f32", (), memoryview(struct.pack("<f", attrs["value_float"])))
    if "value_floats" in attrs:
        v = attrs["value_floats"]
        return TensorInfo(name, "f32", (len(v),), memoryview(struct.pack(f"<{len(v)}f", *v)))
    if "value_int" in attrs:
        return TensorInfo(name, "i64", (), memoryview(struct.pack("<q", attrs["value_int"])))
    if "value_ints" in attrs:
        v = attrs["value_ints"]
        return TensorInfo(name, "i64", (len(v),), memoryview(struct.pack(f"<{len(v)}q", *v)))
    raise _Problem("a Constant without a numeric value (string and sparse constants are not supported)")


def _value(vi, what):
    """(name, dtype, dims) of a graph input or output; raises _Problem."""
    ty = vi["type"]
    if ty is None or ty["kind"] != "tensor":
        kind = "no type" if ty is None else f"a {ty['kind']} type"
        raise _Problem(f"{what} has {kind} (only tensors are supported)")
    dtype = ONNX_DTYPE.get(ty["elem_type"])
    if dtype is None:
        raise _Problem(f"{what} is {onnx_dtype_name(ty['elem_type'])} ({_SUPPORTED_TYPES} are supported)")
    if ty["shape"] is None:
        raise _Problem(f"{what} has no shape (its rank must be known)")
    dims = tuple(f"?{i}" if d is None else d for i, d in enumerate(ty["shape"]))  # cpu-route: model file parsed on the host (one per axis)
    return vi["name"], dtype, dims


def _toposort(nodes, available, problems):
    """Nodes in a topological order, stable for a graph already sorted."""
    producer = {}
    for i, n in enumerate(nodes):  # cpu-route: model file parsed on the host (graph nodes)
        for o in n.outputs:  # cpu-route: model file parsed on the host (node outputs)
            if not o:
                continue
            if o in producer or o in available:
                problems.append((n.name, n.op, f"output '{o}' is already defined"))
            producer.setdefault(o, i)
    deps = []
    users = [[] for _ in nodes]  # cpu-route: model file parsed on the host (graph nodes)
    for i, n in enumerate(nodes):  # cpu-route: model file parsed on the host (graph nodes)
        d = set()
        for x in n.inputs:  # cpu-route: model file parsed on the host (node inputs)
            if not x or x in available:
                continue
            p = producer.get(x)
            if p is None:
                problems.append((n.name, n.op, f"input '{x}' is not produced by any node, initializer or input"))
            elif p != i:
                d.add(p)
        deps.append(len(d))
        for p in d:  # cpu-route: model file parsed on the host (node inputs)
            users[p].append(i)
    ready = [i for i, k in enumerate(deps) if k == 0]  # cpu-route: model file parsed on the host (graph nodes)
    heapq.heapify(ready)
    order = []
    while ready:  # cpu-route: model file parsed on the host (graph nodes)
        i = heapq.heappop(ready)
        order.append(nodes[i])
        for u in users[i]:  # cpu-route: model file parsed on the host (node users)
            deps[u] -= 1
            if deps[u] == 0:
                heapq.heappush(ready, u)
    if len(order) != len(nodes):
        placed = {id(n) for n in order}  # cpu-route: model file parsed on the host (graph nodes)
        for n in nodes:  # cpu-route: model file parsed on the host (graph nodes)
            if id(n) not in placed:
                problems.append((n.name, n.op, "is part of a cycle or depends on a node that is"))
    return tuple(order)


def read(path):
    """Read the ONNX model at ``path`` into an import graph.

    Returns a :class:`~mojolearn.importers._graph.GraphIR` whose nodes are
    in topological order and whose initializers are views of the file's
    bytes (``Constant`` nodes become initializers). Raises
    :class:`UnsupportedModel` listing every problem found at once: an opset
    older than 13, a node outside the default ONNX domain, an initializer,
    input or output of an unsupported type, a subgraph attribute; raises
    ``ValueError`` when the file is not an ONNX model.
    """
    path = os.fspath(path)  # cpu-route: model file parsed on the host
    buf = _map_file(path)
    try:
        m = _pb.model(buf)
    except _pb.ProtoError as e:
        raise ValueError(f"{path} is not an ONNX model: {e}") from None
    g = m["graph"]
    if g is None:
        raise ValueError(f"{path} is not an ONNX model: it has no graph")
    src = os.path.basename(path)
    problems = []
    opset = None
    for o in m["opset_import"]:  # cpu-route: model file parsed on the host (opset entries)
        if o["domain"] in _DEFAULT_DOMAINS:
            opset = o["version"]
    if opset is None:
        problems.append((src, None, "declares no opset for the default ONNX domain"))
        opset = 0
    elif opset < OPSET_MIN:
        problems.append((src, None, f"opset {opset} (opset {OPSET_MIN} or later is needed; convert the "
                                    f"model once with onnx.version_converter)"))
    ext = _External(os.path.dirname(os.path.abspath(path)))
    inits = {}
    for t in g["initializers"]:  # cpu-route: model file parsed on the host (initializers)
        try:
            inits[t["name"]] = _tensor_info(t, ext)
        except _Problem as e:
            problems.append((t["name"], None, str(e)))
    if g["sparse_initializers"]:
        problems.append((src, None, f"{g['sparse_initializers']} sparse initializers (not supported)"))
    nodes = []
    dead = set()  # outputs of refused Constant nodes: defined, so users are not reported twice
    for i, nd in enumerate(g["nodes"]):  # cpu-route: model file parsed on the host (graph nodes)
        op, dom = nd["op_type"], nd["domain"]
        name = nd["name"] or f"{op}_{i}"
        if dom not in _DEFAULT_DOMAINS:
            problems.append((name, op, f"domain '{dom}' is not supported (only the default ONNX domain is)"))
            nodes.append(Node(op, dom, name, tuple(nd["inputs"]), tuple(nd["outputs"]), {}))
            continue
        attrs, bad = {}, False
        for a in nd["attributes"]:  # cpu-route: model file parsed on the host (node attributes)
            try:
                attrs[a["name"]] = _attr(a, ext)
            except _Problem as e:
                problems.append((name, op, str(e)))
                bad = True
        if op == "Constant":
            if len(nd["outputs"]) != 1 or not nd["outputs"][0]:
                problems.append((name, op, "a Constant needs one output"))
                continue
            try:
                if bad:
                    raise _Problem("its value cannot be read")
                inits[nd["outputs"][0]] = _constant_tensor(nd["outputs"][0], attrs)
            except _Problem as e:
                problems.append((name, op, str(e)))
                dead.add(nd["outputs"][0])
            continue
        nodes.append(Node(op, "", name, tuple(nd["inputs"]), tuple(nd["outputs"]), attrs))
    inputs = []
    for vi in g["inputs"]:  # cpu-route: model file parsed on the host (graph inputs)
        if vi["name"] in inits:
            continue  # an initializer listed as an input (IR 3 style) stays a constant
        try:
            inputs.append(_value(vi, "input"))
        except _Problem as e:
            problems.append((vi["name"], None, str(e)))
    outputs = []
    for vi in g["outputs"]:  # cpu-route: model file parsed on the host (graph outputs)
        try:
            outputs.append(_value(vi, "output"))
        except _Problem as e:
            problems.append((vi["name"], None, str(e)))
    available = set(inits) | dead | {x[0] for x in inputs}  # cpu-route: model file parsed on the host (graph inputs)
    ordered = _toposort(nodes, available, problems)
    produced = available | {o for n in nodes for o in n.outputs}  # cpu-route: model file parsed on the host (node outputs)
    for name, _, _ in outputs:  # cpu-route: model file parsed on the host (graph outputs)
        if name not in produced:
            problems.append((name, None, "output is not produced by any node"))
    if problems:
        raise UnsupportedModel(path, problems)
    return GraphIR(opset=opset, inputs=tuple(inputs), outputs=tuple(outputs), nodes=ordered,
                   initializers=inits, source=path)
