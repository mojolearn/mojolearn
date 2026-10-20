# SPDX-License-Identifier: Apache-2.0
"""A minimal protobuf wire reader for the ONNX messages an inference import
needs (ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto with
external data, ValueInfoProto, TypeProto, TensorShapeProto,
OperatorSetIdProto, StringStringEntryProto).

No ``onnx`` and no ``protobuf`` dependency: the wire format is varints and
length-delimited fields, and every field this module does not name is
skipped. Messages decode into plain dicts of the fields read; tensor bytes
stay memoryviews into the file's buffer (no copy before the device upload).
Field numbers are those of onnx/onnx.proto3.
"""
# cpu-route: model file parsed on the host. Everything here is the explicit
# input step before the runtime: it reads a file's description (names,
# shapes, attributes, byte ranges) and never computes on element values.
from __future__ import annotations

import struct

# wire types
_VARINT, _FIXED64, _LEN, _START_GROUP, _END_GROUP, _FIXED32 = 0, 1, 2, 3, 4, 5


class ProtoError(ValueError):
    """The bytes are not a well-formed protobuf message."""


def _varint(buf, pos, end):
    """(value, next_pos) of the unsigned varint at ``pos``."""
    shift = 0
    value = 0
    while True:  # cpu-route: model file parsed on the host (at most 10 bytes of one varint)
        if pos >= end:
            raise ProtoError("truncated varint")
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if b < 0x80:
            return value, pos
        shift += 7
        if shift >= 70:
            raise ProtoError("varint longer than 10 bytes")


def _signed64(v):
    """A varint read as a two's-complement int64 (protobuf int64/int32 fields)."""
    return v - (1 << 64) if v >= (1 << 63) else v


def fields(buf, start=0, end=None):
    """Yield ``(field_number, wire_type, value)`` for each field of the
    message in ``buf[start:end]``: an int for a varint, a memoryview for a
    length-delimited field, the raw little-endian memoryview of 4 or 8 bytes
    for fixed32/fixed64. Groups are refused (ONNX has none)."""
    if end is None:
        end = len(buf)
    pos = start
    while pos < end:  # cpu-route: model file parsed on the host (walks message fields, never tensor words)
        key, pos = _varint(buf, pos, end)
        num, wt = key >> 3, key & 7
        if num == 0:
            raise ProtoError("field number 0")
        if wt == _VARINT:
            v, pos = _varint(buf, pos, end)
            yield num, wt, v
        elif wt == _LEN:
            n, pos = _varint(buf, pos, end)
            if pos + n > end:
                raise ProtoError("truncated length-delimited field")
            yield num, wt, buf[pos:pos + n]
            pos += n
        elif wt == _FIXED32:
            if pos + 4 > end:
                raise ProtoError("truncated fixed32")
            yield num, wt, buf[pos:pos + 4]
            pos += 4
        elif wt == _FIXED64:
            if pos + 8 > end:
                raise ProtoError("truncated fixed64")
            yield num, wt, buf[pos:pos + 8]
            pos += 8
        else:
            raise ProtoError(f"unsupported wire type {wt} (groups)")


def _str(v):
    return bytes(v).decode("utf-8")


def _packed_varints(v, wt, signed=True):
    """Every value of a repeated varint field occurrence (packed or not)."""
    if wt == _VARINT:
        return [_signed64(v) if signed else v]
    if wt != _LEN:
        raise ProtoError("repeated integer field with a fixed wire type")
    out = []
    pos, end = 0, len(v)
    while pos < end:  # cpu-route: model file parsed on the host (dims, axes and small integer lists)
        x, pos = _varint(v, pos, end)
        out.append(_signed64(x) if signed else x)
    return out


# ---------------------------------------------------------------- messages --

def opset_id(buf):
    """OperatorSetIdProto -> {"domain": str, "version": int}."""
    d = {"domain": "", "version": 0}
    for num, wt, v in fields(buf):  # cpu-route: model file parsed on the host
        if num == 1 and wt == _LEN:
            d["domain"] = _str(v)
        elif num == 2 and wt == _VARINT:
            d["version"] = _signed64(v)
    return d


def string_entry(buf):
    """StringStringEntryProto -> (key, value)."""
    k, val = "", ""
    for num, wt, v in fields(buf):  # cpu-route: model file parsed on the host
        if num == 1 and wt == _LEN:
            k = _str(v)
        elif num == 2 and wt == _LEN:
            val = _str(v)
    return k, val


def tensor(buf):
    """TensorProto -> dict with ``name``, ``data_type``, ``dims``, ``raw``
    (memoryview or None), the typed data fields that were present as
    (wire_type, memoryview or int) occurrences, ``external`` (dict of the
    external_data entries), ``data_location`` and ``segment`` (True when a
    segment field was present: refused by the reader)."""
    d = {"name": "", "data_type": 0, "dims": [], "raw": None, "typed": {}, "external": {},
         "data_location": 0, "segment": False}
    for num, wt, v in fields(buf):  # cpu-route: model file parsed on the host
        if num == 1:
            d["dims"].extend(_packed_varints(v, wt))
        elif num == 2 and wt == _VARINT:
            d["data_type"] = _signed64(v)
        elif num == 3:
            d["segment"] = True
        elif num == 8 and wt == _LEN:
            d["name"] = _str(v)
        elif num == 9 and wt == _LEN:
            d["raw"] = v
        elif num in (4, 5, 7, 10, 11):  # float_data int32_data int64_data double_data uint64_data
            d["typed"].setdefault(num, []).append((wt, v))
        elif num == 6:  # string_data
            d["typed"].setdefault(6, []).append((wt, v))
        elif num == 13 and wt == _LEN:
            k, val = string_entry(v)
            d["external"][k] = val
        elif num == 14 and wt == _VARINT:
            d["data_location"] = _signed64(v)
    return d


def tensor_shape(buf):
    """TensorShapeProto -> list of dims: an int (dim_value), a str
    (dim_param) or None (neither)."""
    dims = []
    for num, wt, v in fields(buf):  # cpu-route: model file parsed on the host
        if num == 1 and wt == _LEN:
            val = None
            for n2, w2, v2 in fields(v):  # cpu-route: model file parsed on the host
                if n2 == 1 and w2 == _VARINT:
                    val = _signed64(v2)
                elif n2 == 2 and w2 == _LEN:
                    val = _str(v2)
            dims.append(val)
    return dims


def type_proto(buf):
    """TypeProto -> {"kind": "tensor"|"sequence"|"map"|"optional"|"sparse"|"unknown",
    "elem_type": int, "shape": list or None}."""
    d = {"kind": "unknown", "elem_type": 0, "shape": None}
    for num, wt, v in fields(buf):  # cpu-route: model file parsed on the host
        if num == 1 and wt == _LEN:
            d["kind"] = "tensor"
            for n2, w2, v2 in fields(v):  # cpu-route: model file parsed on the host
                if n2 == 1 and w2 == _VARINT:
                    d["elem_type"] = _signed64(v2)
                elif n2 == 2 and w2 == _LEN:
                    d["shape"] = tensor_shape(v2)
        elif num == 4:
            d["kind"] = "sequence"
        elif num == 5:
            d["kind"] = "map"
        elif num == 9:
            d["kind"] = "optional"
        elif num == 8:
            d["kind"] = "sparse"
    return d


def value_info(buf):
    """ValueInfoProto -> {"name": str, "type": TypeProto dict or None}."""
    d = {"name": "", "type": None}
    for num, wt, v in fields(buf):  # cpu-route: model file parsed on the host
        if num == 1 and wt == _LEN:
            d["name"] = _str(v)
        elif num == 2 and wt == _LEN:
            d["type"] = type_proto(v)
    return d


# AttributeProto.AttributeType
ATTR_FLOAT, ATTR_INT, ATTR_STRING, ATTR_TENSOR, ATTR_GRAPH = 1, 2, 3, 4, 5
ATTR_FLOATS, ATTR_INTS, ATTR_STRINGS, ATTR_TENSORS, ATTR_GRAPHS = 6, 7, 8, 9, 10
ATTR_SPARSE_TENSOR, ATTR_SPARSE_TENSORS, ATTR_TYPE_PROTO, ATTR_TYPE_PROTOS = 11, 12, 13, 14


def attribute(buf):
    """AttributeProto -> dict with ``name``, ``type`` (the declared
    AttributeType, 0 when absent), ``ref_attr_name`` and the values present:
    ``f``, ``i``, ``s`` (bytes), ``t`` (TensorProto dict), ``floats``,
    ``ints``, ``strings``, ``tensors``; ``has_graph`` / ``has_sparse`` /
    ``has_type_proto`` flag the kinds an inference import refuses."""
    d = {"name": "", "type": 0, "ref_attr_name": "", "floats": [], "ints": [], "strings": [],
         "tensors": [], "has_graph": False, "has_sparse": False, "has_type_proto": False}
    for num, wt, v in fields(buf):  # cpu-route: model file parsed on the host
        if num == 1 and wt == _LEN:
            d["name"] = _str(v)
        elif num == 20 and wt == _VARINT:
            d["type"] = _signed64(v)
        elif num == 21 and wt == _LEN:
            d["ref_attr_name"] = _str(v)
        elif num == 2 and wt == _FIXED32:
            d["f"] = struct.unpack("<f", v)[0]
        elif num == 3 and wt == _VARINT:
            d["i"] = _signed64(v)
        elif num == 4 and wt == _LEN:
            d["s"] = bytes(v)
        elif num == 5 and wt == _LEN:
            d["t"] = tensor(v)
        elif num == 7:
            if wt == _FIXED32:
                d["floats"].append(struct.unpack("<f", v)[0])
            elif wt == _LEN:
                if len(v) % 4:
                    raise ProtoError("packed floats not a multiple of 4 bytes")
                d["floats"].extend(struct.unpack(f"<{len(v) // 4}f", v))
        elif num == 8:
            d["ints"].extend(_packed_varints(v, wt))
        elif num == 9 and wt == _LEN:
            d["strings"].append(bytes(v))
        elif num == 10 and wt == _LEN:
            d["tensors"].append(tensor(v))
        elif num in (6, 11):
            d["has_graph"] = True
        elif num in (22, 23):
            d["has_sparse"] = True
        elif num in (14, 15):
            d["has_type_proto"] = True
    return d


def node(buf):
    """NodeProto -> {"op_type", "domain", "name", "inputs", "outputs", "attributes"}."""
    d = {"op_type": "", "domain": "", "name": "", "inputs": [], "outputs": [], "attributes": []}
    for num, wt, v in fields(buf):  # cpu-route: model file parsed on the host
        if num == 1 and wt == _LEN:
            d["inputs"].append(_str(v))
        elif num == 2 and wt == _LEN:
            d["outputs"].append(_str(v))
        elif num == 3 and wt == _LEN:
            d["name"] = _str(v)
        elif num == 4 and wt == _LEN:
            d["op_type"] = _str(v)
        elif num == 7 and wt == _LEN:
            d["domain"] = _str(v)
        elif num == 5 and wt == _LEN:
            d["attributes"].append(attribute(v))
    return d


def graph(buf):
    """GraphProto -> {"name", "nodes", "initializers", "sparse_initializers"
    (count), "inputs", "outputs", "value_info"}."""
    d = {"name": "", "nodes": [], "initializers": [], "sparse_initializers": 0,
         "inputs": [], "outputs": [], "value_info": []}
    for num, wt, v in fields(buf):  # cpu-route: model file parsed on the host
        if wt != _LEN:
            continue
        if num == 1:
            d["nodes"].append(node(v))
        elif num == 2:
            d["name"] = _str(v)
        elif num == 5:
            d["initializers"].append(tensor(v))
        elif num == 15:
            d["sparse_initializers"] += 1
        elif num == 11:
            d["inputs"].append(value_info(v))
        elif num == 12:
            d["outputs"].append(value_info(v))
        elif num == 13:
            d["value_info"].append(value_info(v))
    return d


def model(buf):
    """ModelProto -> {"ir_version", "opset_import", "graph" (or None),
    "functions" (count), "training_info" (count)}."""
    d = {"ir_version": 0, "opset_import": [], "graph": None, "functions": 0, "training_info": 0}
    for num, wt, v in fields(buf):  # cpu-route: model file parsed on the host
        if num == 1 and wt == _VARINT:
            d["ir_version"] = _signed64(v)
        elif num == 8 and wt == _LEN:
            d["opset_import"].append(opset_id(v))
        elif num == 7 and wt == _LEN:
            d["graph"] = graph(v)
        elif num == 25:
            d["functions"] += 1
        elif num == 20:
            d["training_info"] += 1
    return d
