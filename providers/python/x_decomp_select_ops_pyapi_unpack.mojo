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
