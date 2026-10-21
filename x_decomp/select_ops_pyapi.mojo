# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/select_ops.mojo: the `reduce_py` and
`order_small_py` doors both bindings register (host buffers). Each unpacks
addresses and integers and calls one typed `*_run` entry in
x_decomp/select_ops.mojo; no data work happens here."""
from std.python import PythonObject

from x_decomp.cells import F32Ptr
from x_decomp.select_ops import order_small_run, reduce_run


def _addr(o: PythonObject) raises -> F32Ptr:
    var a = Int(py=o)
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def _cnt(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v


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
