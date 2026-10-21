# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ann lane's Python address contract, shared by
`bindings/_mojolearn_x_ann.mojo` and `bindings/_mojolearn_x_ann_host.mojo`:
PythonObject unpacking and plain copies, no arithmetic."""

from std.python import PythonObject
from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from x_ann.ivf_pq_core import F32P, I32P
from x_ann.abi import out_f32_to, out_i32_to


def p_int(params: PythonObject, i: Int) raises -> Int:
    return Int(py=params[i])


def a_int(addrs: PythonObject, i: Int) raises -> Int:
    return Int(py=addrs[i])


def py_int_list(obj: PythonObject) raises -> List[Int]:
    """A Python sequence of ints (addresses or parameters) as a typed list."""
    var out = List[Int]()
    for t in range(len(obj)):
        out.append(Int(py=obj[t]))
    return out^


def in_f32(addrs: PythonObject, i: Int, n: Int) raises -> List[Float32]:
    return read_f32(a_int(addrs, i), n)


def in_i32(addrs: PythonObject, i: Int, n: Int) raises -> List[Int32]:
    return read_i32(a_int(addrs, i), n)


def ptr_f32(addrs: PythonObject, i: Int) raises -> F32P:
    """The caller's float32 array itself (no copy): the host binding reads
    and writes it in place, under the GIL-released call that owns it."""
    return rebind[F32P](f32_ptr(a_int(addrs, i)))


def ptr_i32(addrs: PythonObject, i: Int) raises -> I32P:
    return rebind[I32P](i32_ptr(a_int(addrs, i)))


def out_f32(values: List[Float32], addrs: PythonObject, i: Int) raises:
    """Into the caller's array (`out_f32_to`)."""
    out_f32_to(values, a_int(addrs, i))


def out_i32(values: List[Int32], addrs: PythonObject, i: Int) raises:
    out_i32_to(values, a_int(addrs, i))
