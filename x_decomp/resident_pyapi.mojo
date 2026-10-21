# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the device-resident matrices (x_decomp/resident.mojo):
the doors the GPU binding registers. Each door unpacks ids and integers and
calls one typed entry in x_decomp/resident.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.resident import (
    pool_alloc,
    pool_free,
    dev_upload_run,
    dev_download_run,
    dev_ew_run,
    dev_mcd_cov_run,
    dev_gemm_run,
    dev_project_run,
)


def _id(o: PythonObject) raises -> Int:
    return Int(py=o)


def _n(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v


def dev_alloc_py(n: PythonObject) raises -> PythonObject:
    return PythonObject(pool_alloc(Int(py=n)))


def dev_free_py(id: PythonObject) raises -> PythonObject:
    pool_free(_id(id))
    return PythonObject(None)


def dev_upload_py(id: PythonObject, addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """Host floats into the device matrix; waits (the host store may change
    after this returns)."""
    var cnt = Int(py=n)
    var src = F32Ptr(unsafe_from_address=Int(py=addr))
    var i = _id(id)
    var r = 0
    with GILReleased(Python()):
        r = dev_upload_run(i, src, cnt)
    return PythonObject(r)


def dev_download_py(id: PythonObject, addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """The device matrix into host floats, after every launch before it."""
    var cnt = Int(py=n)
    var dst = F32Ptr(unsafe_from_address=Int(py=addr))
    var i = _id(id)
    var r = 0
    with GILReleased(Python()):
        r = dev_download_run(i, dst, cnt)
    return PythonObject(r)


def dev_ew_py(
    a: PythonObject, b: PythonObject, c: PythonObject, dst: PythonObject, p: PythonObject, s: PythonObject
) raises -> PythonObject:
    # p = [op, count, d, lb, bm, lc, cm], as x_decomp_ew
    var op = Int(py=p[0])
    var count = _n(p, 1)
    var d = _n(p, 2)
    var lb = _n(p, 3)
    var bm = Int(py=p[4])
    var lc = _n(p, 5)
    var cm = Int(py=p[6])
    return PythonObject(dev_ew_run(_id(a), _id(b), _id(c), _id(dst), op, count, d, lb, bm, lc, cm, Float64(py=s)))


def dev_mcd_cov_py(a: PythonObject, c: PythonObject, p: PythonObject) raises -> PythonObject:
    """Opt-in MCD final masked covariance Gram; no generic GEMM dispatch change."""
    return PythonObject(dev_mcd_cov_run(_id(a), _id(c), _n(p, 0), _n(p, 1)))


def dev_gemm_py(a: PythonObject, b: PythonObject, c: PythonObject, p: PythonObject) raises -> PythonObject:
    var m = _n(p, 0)
    var k = _n(p, 1)
    var n = _n(p, 2)
    var ta = Int(py=p[3]) != 0
    var tb = Int(py=p[4]) != 0
    return PythonObject(dev_gemm_run(_id(a), _id(b), _id(c), m, k, n, ta, tb))


def dev_project_py(a: PythonObject, b: PythonObject, c: PythonObject, flag: PythonObject, p: PythonObject) raises -> PythonObject:
    """c = a b^T (a m x k, b n x k) by the tiled projection kernel, enqueued
    (no sync); flag[0] = 1 when an entry of a is not finite. Returns -1,
    launching nothing, on a device whose shared memory does not hold the tile."""
    var m = _n(p, 0)
    var k = _n(p, 1)
    var n = _n(p, 2)
    return PythonObject(dev_project_run(_id(a), _id(b), _id(c), _id(flag), m, k, n))
