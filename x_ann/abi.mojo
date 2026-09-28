# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ann lane's address contract, shared by `bindings/_mojolearn_x_ann.mojo`
and `bindings/_mojolearn_x_ann_host.mojo`: plain copies, no arithmetic."""

from std.memory import memcpy
from std.python import PythonObject
from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from x_ann.ivf_pq_core import F32P, I32P
from x_ann.switches import ANN3_HOST_PASSES


def p_int(params: PythonObject, i: Int) raises -> Int:
    return Int(py=params[i])


def a_int(addrs: PythonObject, i: Int) raises -> Int:
    return Int(py=addrs[i])


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
    """Into the caller's array. Under `ANN3_HOST_PASSES` (lane ann-apple3)
    one memcpy; one store per word otherwise (the IVF code arrays are
    n x dim words)."""
    var p = f32_ptr(a_int(addrs, i))
    comptime if ANN3_HOST_PASSES:
        if len(values) > 0:
            memcpy(dest=p, src=values.unsafe_ptr(), count=len(values))
    else:
        for e in range(len(values)):
            p.unsafe_store(e, values[e])


def out_i32(values: List[Int32], addrs: PythonObject, i: Int) raises:
    var p = i32_ptr(a_int(addrs, i))
    comptime if ANN3_HOST_PASSES:
        if len(values) > 0:
            memcpy(dest=p, src=values.unsafe_ptr(), count=len(values))
    else:
        for e in range(len(values)):
            p.unsafe_store(e, values[e])


def check_search(n_lists: Int, m: Int, k: Int, n_probes: Int) raises:
    if m <= 0 or k <= 0:
        raise Error("x_ann: search needs at least one query and k >= 1")
    if n_probes <= 0 or n_probes > n_lists:
        raise Error("x_ann: n_probes must be in [1, n_lists]; it is not clamped")
