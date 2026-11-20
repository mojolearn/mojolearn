# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of `core/py2mojo_rows.mojo` (lane/apple-fast-py2mojo-linear):
PythonObject arguments to a checked typed `RowsCall`, the GIL released around
the typed device or host pass. No arithmetic on data."""
# The PACK half of the former `providers/python/core_py2mojo_rows_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/core_py2mojo_rows_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceContext
from core.py2mojo_rows import ROWS_HUBER_OUT, RowsCall, rows_call, rows_device, rows_host, rows_param_count


from providers.python.core_py2mojo_rows_pyapi_unpack import (
    rows_call_from_python,
)


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
