# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/mds_iso.mojo: the PythonObject <-> typed-value
doors that `bindings/_mojolearn_x_decomp_host.mojo` registers under the same
names. Each door unpacks addresses and integers, releases the GIL where it
did and calls one typed entry in x_decomp/mds_iso.mojo; no data work
happens here. Every address is a host buffer the caller owns."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.mds_iso import mds_disp_host, mds_setup_host


def _fp(o: PythonObject) raises -> F32Ptr:
    var a = Int(py=o)
    if a == 0:
        raise Error("x_decomp: null buffer address")
    return F32Ptr(unsafe_from_address=a)


def _ip(o: PythonObject) raises -> I32Ptr:
    var a = Int(py=o)
    if a == 0:
        raise Error("x_decomp: null buffer address")
    return I32Ptr(unsafe_from_address=a)


def mds_setup_host_py(
    dis: PythonObject, keys: PythonObject, idx: PythonObject, gid: PythonObject, gst: PythonObject, p: PythonObject
) raises -> PythonObject:
    """The setup on host buffers (p = [n]; keys, idx, gid hold n (n - 1) / 2,
    gst n (n - 1) / 2 + 1). Returns (m, G)."""
    var r = mds_setup_host(_fp(dis), _fp(keys), _ip(idx), _ip(gid), _ip(gst), Int(py=p[0]))
    return Python.tuple(r[0], r[1])


def mds_disp_host_py(d: PythonObject, dst: PythonObject, a: PythonObject, p: PythonObject) raises -> PythonObject:
    """One iteration's upper-triangle disparities into dst (n x n, zeroed
    here) on host buffers. a = [keys, idx, gid, gst, sm, wt, end, prv,
    last, hf, gv] addresses; p = [n, m, G, first]."""
    var li = List[Int]()
    for i in range(11):  # small-loop(the 11 buffer address words of a)
        li.append(Int(py=a[i]))
    var m = Int(py=p[1])
    var pd = _fp(d)
    var po = _fp(dst)
    var n = Int(py=p[0])
    var G = Int(py=p[2])
    var first = Int(py=p[3]) != 0
    with GILReleased(Python()):
        mds_disp_host(pd, po, li, n, m, G, first)
    return PythonObject(m)
