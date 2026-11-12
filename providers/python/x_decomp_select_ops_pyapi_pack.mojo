# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/select_ops.mojo: the `reduce_py` and
`order_small_py` doors both bindings register (host buffers). Each unpacks
addresses and integers and calls one typed `*_run` entry in
x_decomp/select_ops.mojo; no data work happens here."""
# The PACK half of the former `providers/python/x_decomp_select_ops_pyapi.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `providers/python/x_decomp_select_ops_pyapi_unpack.mojo`.
from std.python import PythonObject

from x_decomp.cells import F32Ptr
from x_decomp.select_ops import order_small_run, reduce_run


from providers.python.x_decomp_select_ops_pyapi_unpack import (
    _addr,
    _cnt,
)


def reduce_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst[0] = the op (p = [count, op]: 0 max |x|, 1 max, 2 min) over the
    first count floats at host address a."""
    var count = _cnt(p, 0)
    var op = Int(py=p[1])
    var src = _addr(a)
    var out = _addr(dst)
    reduce_run(src, out, count, op)
    return PythonObject(1)


def order_small_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst (p = [n] exact floats) = the stable ascending order of the n
    floats at host address a (see `order_less`)."""
    var n = _cnt(p, 0)
    var src = _addr(a)
    var out = _addr(dst)
    order_small_run(src, out, n)
    return PythonObject(n)
