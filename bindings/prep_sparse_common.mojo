# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Transitional CPython boundary for the native sparse entry points.

This module still needs CPython and does NOT meet the owner's no-product-
Python requirement. The independently callable native implementation is in
x_prep/sparse_input*.mojo and x_prep/host/sparse_input.mojo. No SciPy
conversion routine is called here. LIL/DOK objects are transcribed into raw
coordinates; all sorting, duplicate handling and narrowing happens natively.
"""
from std.python import Python, PythonObject
from x_prep.sparse_input import SparseShape, SparseView, BP, SP_MAX, validate_shape, validate_view


def sparse_shape(sizes: PythonObject) raises -> SparseShape:
    if Int(py=len(sizes)) != 7:
        raise Error("sparse input: expected seven shape fields")
    var s = SparseShape(Int(py=sizes[0]), Int(py=sizes[1]), Int(py=sizes[2]), Int(py=sizes[3]),
                        Int(py=sizes[4]), Int(py=sizes[5]), Int(py=sizes[6]))
    validate_shape(s)
    return s


def sparse_view(meta: PythonObject, indices: Bool = False) raises -> SparseView:
    if Int(py=len(meta)) != 11:
        raise Error("sparse input: expected eleven buffer descriptor fields")
    var addr = Int(py=meta[0])
    var code = Int(py=meta[1])
    var width = Int(py=meta[2])
    var d0 = Int(py=meta[3])
    var d1 = Int(py=meta[4])
    var d2 = Int(py=meta[5])
    var span = Int(py=meta[10])
    if code < 0 or code > 11 or (abs(width) != 1 and abs(width) != 2 and abs(width) != 4 and abs(width) != 8):
        raise Error("sparse input: unsupported buffer dtype")
    if d0 < 0 or d1 < 1 or d2 < 1 or d1 > SP_MAX // d2 or d0 > SP_MAX // (d1 * d2):
        raise Error("sparse input: buffer shape exceeds Int32 indexing")
    if span < 0 or (d0 > 0 and (span == 0 or addr == 0)):
        raise Error("sparse input: null or empty buffer span")
    if indices and (code == 0 or code == 1 or code == 8):
        raise Error("sparse input: integer indices required")
    var view = SparseView(BP(unsafe_from_address=addr), code, width, d1, d2,
                          Int(py=meta[6]), Int(py=meta[7]), Int(py=meta[8]), Int(py=meta[9]), d0 * d1 * d2)
    validate_view(view, indices)
    var lo = view.origin
    var hi = view.origin + abs(width)
    lo += min(0, (d0 - 1) * view.stride0) + min(0, (d1 - 1) * view.stride1) + min(0, (d2 - 1) * view.stride2)
    hi += max(0, (d0 - 1) * view.stride0) + max(0, (d1 - 1) * view.stride1) + max(0, (d2 - 1) * view.stride2)
    if d0 > 0 and (lo < 0 or hi > span):
        raise Error("sparse input: strided buffer exceeds its declared byte span")
    return view


def sparse_check_outputs(outputs: PythonObject) raises:
    if Int(py=len(outputs)) != 4:
        raise Error("sparse input: expected three output addresses and capacity")
    var capacity = Int(py=outputs[3])
    if capacity < 0 or capacity > SP_MAX or Int(py=outputs[0]) == 0:
        raise Error("sparse input: invalid output capacity or row-pointer address")
    if capacity > 0 and (Int(py=outputs[1]) == 0 or Int(py=outputs[2]) == 0):
        raise Error("sparse input: null nonempty CSR output")


def sparse_object_size(kind: PythonObject, x: PythonObject, n_: PythonObject) raises -> PythonObject:
    """LIL row-container lengths / DOK mapping size; no numerical computation."""
    var n = Int(py=n_)
    var count = 0
    if Int(py=kind) == 0:
        var rows = x.rows
        var values = x.data
        if Int(py=len(rows)) != n or Int(py=len(values)) != n:
            raise Error("sparse input: LIL container length differs from shape")
        for r in range(n):
            var length = Int(py=len(rows[r]))
            if Int(py=len(values[r])) != length:
                raise Error("sparse input: LIL row/data lengths differ")
            count += length
            if count > SP_MAX:
                raise Error("sparse input: too many LIL entries")
    else:
        count = Int(py=len(x))
    return PythonObject(count)


def sparse_object_pack(kind: PythonObject, x: PythonObject, sizes: PythonObject, outputs: PythonObject) raises -> PythonObject:
    """Copy host object fields into caller-owned i64/i64/f64 buffers.

    The object protocol itself remains a CPython interface dependency. This
    entry is not a Python-free public interface and must not be certified as
    one. It preserves LIL/DOK interoperability during the native migration.
    """
    var rp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=outputs[0]))
    var cp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=outputs[1]))
    var vp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=outputs[2]))
    var n = Int(py=sizes[0])
    var cap = Int(py=sizes[1])
    var unsigned64 = String(py=x.dtype.kind) == "u" and Int(py=x.dtype.itemsize) == 8
    var at = 0
    if Int(py=kind) == 0:
        var rows = x.rows
        var values = x.data
        for r in range(n):
            var rr = rows[r]
            var vv = values[r]
            var length = Int(py=len(rr))
            if Int(py=len(vv)) != length:
                raise Error("sparse input: LIL changed during buffer marshalling")
            for j in range(length):
                if at >= cap:
                    raise Error("sparse input: LIL changed during buffer marshalling")
                rp.unsafe_store(at, Int64(r))
                cp.unsafe_store(at, Int64(Int(py=rr[j])))
                vp.unsafe_store(at, Float64(Int(py=vv[j])) if unsigned64 else Float64(py=vv[j]))
                at += 1
    else:
        # Builtin iteration only, never scipy's conversion helpers. Values
        # are copied in their mapping order; native COO sorting is explicit.
        var builtins = Python.import_module("builtins")
        var items = builtins.list(x.items())
        if Int(py=len(items)) != cap:
            raise Error("sparse input: DOK changed during buffer marshalling")
        for j in range(cap):
            var item = items[j]
            var key = item[0]
            rp.unsafe_store(j, Int64(Int(py=key[0])))
            cp.unsafe_store(j, Int64(Int(py=key[1])))
            vp.unsafe_store(j, Float64(Int(py=item[1])) if unsigned64 else Float64(py=item[1]))
        at = cap
    if at != cap:
        raise Error("sparse input: object changed during buffer marshalling")
    return PythonObject(at)
