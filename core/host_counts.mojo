# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel.
"""CPU-only grouped class counts for stratified group splitting."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.hotpath_helpers import _ptr, HOTPATH_SABOTAGE


def _bincount2_i32_host(
    a_addr: Int, b_addr: Int, count: Int, ka: Int, kb: Int, counts_addr: Int,
) raises -> Int:
    """int64 counts[v * ka + u] over the n rows of two int32 code columns
    (u = a[i], v = b[i]): the host column's twin of the GPU base binding's
    `bincount2_i32` (core/hotpath_device.mojo device_bincount2_i32,
    StratifiedGroupKFold's group x class table). The counts are integers, so
    the table is the device's bit for bit. A code outside [0, ka) x [0, kb)
    raises before any count is written (the device leaves the table
    untouched then too)."""
    if count < 0 or ka < 1 or kb < 1:
        raise Error("bincount2_i32: 0 <= n, 1 <= ka, kb")
    var ap = _ptr[DType.int32](a_addr)
    var bp = _ptr[DType.int32](b_addr)
    var cp = _ptr[DType.int64](counts_addr)
    for i in range(count):
        var u = Int(ap.unsafe_load(i))
        var v = Int(bp.unsafe_load(i))
        if u < 0 or u >= ka or v < 0 or v >= kb:
            raise Error("bincount2_i32: a code is out of range")
    for j in range(ka * kb):
        cp.unsafe_store(j, 0)
    for i in range(count):
        var u = Int(ap.unsafe_load(i))
        comptime if HOTPATH_SABOTAGE:
            u = (u + 1) % ka
        var at = Int(bp.unsafe_load(i)) * ka + u
        cp.unsafe_store(at, cp.unsafe_load(at) + 1)
    return 0


def bincount2_i32_host_binding(
    a_addr: PythonObject, b_addr: PythonObject, dims: PythonObject, counts_addr: PythonObject,
) raises -> PythonObject:
    """dims = [n, ka, kb]: int64 counts[b * ka + a] over the n rows of two
    int32 code columns, on the host (the core host binding's
    `bincount2_i32`, the GPU base binding's twin). Raises when a code is
    outside [0, ka) x [0, kb)."""
    if len(dims) != 3:
        raise Error("bincount2_i32: dims [n, ka, kb]")
    var count = Int(py=dims[0])
    var ka = Int(py=dims[1])
    var kb = Int(py=dims[2])
    var ca = Int(py=counts_addr)
    if ca == 0:
        raise Error("bincount2_i32: null counts address")
    var a = Int(py=a_addr)
    var b = Int(py=b_addr)
    if count > 0 and (a == 0 or b == 0):
        raise Error("bincount2_i32: null code address")
    var r: Int
    with GILReleased(Python()):
        r = _bincount2_i32_host(a, b, count, ka, kb, ca)
    return PythonObject(r)


