# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of `core/py2mojo_rows.mojo` (lane/apple-fast-py2mojo-linear):
PythonObject arguments to a checked typed `RowsCall`, the GIL released around
the typed device or host pass. No arithmetic on data."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceContext
from core.py2mojo_rows import ROWS_HUBER_OUT, RowsCall, rows_call, rows_device, rows_host, rows_param_count


def rows_call_from_python(mode: PythonObject, src_addr: PythonObject, dst_addr: PythonObject,
                          params: PythonObject) raises -> RowsCall:
    var md = Int(py=mode)
    var want = rows_param_count(md)
    if len(params) != want:
        raise Error("py2mojo_rows: params must contain n, k (and, for HUBER_OUT, the pred address and thr)")
    var src2 = 0
    var thr = Float64(0)
    if md == ROWS_HUBER_OUT:
        src2 = Int(py=params[2])
        thr = Float64(py=params[3])
    return rows_call(md, Int(py=params[0]), Int(py=params[1]), Int(py=src_addr), Int(py=dst_addr), src2, thr)


def py2mojo_rows_device_binding(
    ctx: DeviceContext, mode: PythonObject, src_addr: PythonObject, dst_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The GPU bindings' `py2mojo_rows`; returns n."""
    var call = rows_call_from_python(mode, src_addr, dst_addr, params)
    with GILReleased(Python()):
        rows_device(ctx, call)
    return PythonObject(call.n)


def py2mojo_rows_host_binding(
    mode: PythonObject, src_addr: PythonObject, dst_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """The CPU bindings' `py2mojo_rows` (the same per-row words); returns n."""
    var call = rows_call_from_python(mode, src_addr, dst_addr, params)
    if call.n > 0:
        with GILReleased(Python()):
            rows_host(call)
    return PythonObject(call.n)
