# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The decomp lane's Python shell: the PythonObject <-> typed-value doors that
`bindings/_mojolearn_x_decomp.mojo` (`*_py[DevExec]`) and
`bindings/_mojolearn_x_decomp_host.mojo` (`*_py[HostExec]`) register under the
same names. Each door unpacks addresses and integers, releases the GIL and
calls one typed entry in x_decomp/api.mojo or the kernel module it names; no
data work happens here. Every address is a host buffer the caller owns."""
from std.python import PythonObject

from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.moves import F64Ptr

def _f(addr: PythonObject) raises -> F32Ptr:
    var a = Int(py=addr)
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def _i(addr: PythonObject) raises -> I32Ptr:
    var a = Int(py=addr)
    if a == 0:
        raise Error("x_decomp: null int32 buffer address")
    return I32Ptr(unsafe_from_address=a)


def _n(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension dst of range")
    return v


def _d(addr: PythonObject) raises -> F64Ptr:
    var a = Int(py=addr)
    if a == 0:
        raise Error("x_decomp: null float64 buffer address")
    return F64Ptr(unsafe_from_address=a)
