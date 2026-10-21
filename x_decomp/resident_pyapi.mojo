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
    dev_rand_run,
    dev_trisolve_run,
    dev_knn_select_run,
    dev_colsum_run,
    dev_rowsum_run,
    dev_sqdist_run,
    dev_absmax_run,
    dev_orth_run,
    dev_orth_diag_run,
    dev_lda_rows_run,
    dev_lda_bound_run,
    dev_als_rows_run,
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


def dev_rand_py(dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [count, seed, stream, kind]: the counter-based Philox draws written
    into the device matrix, enqueued (no sync, no download)."""
    var count = _n(p, 0)
    var seed = Int(py=p[1])
    var stream = Int(py=p[2])
    var kind = Int(py=p[3])
    return PythonObject(dev_rand_run(_id(dst), count, seed, stream, kind))


def dev_trisolve_py(lu: PythonObject, idx: PythonObject, src: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """`trisolve_py` on device ids, enqueued (no sync)."""
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    var trans = _n(p, 2)
    return PythonObject(dev_trisolve_run(_id(lu), _id(idx), _id(src), _id(dst), n, nrhs, trans))


def dev_knn_select_py(dmat: PythonObject, dist: PythonObject, idx: PythonObject, p: PythonObject) raises -> PythonObject:
    """`knn_select_py` on device ids, enqueued (no sync)."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var k = _n(p, 2)
    var ex = _n(p, 3)
    return PythonObject(dev_knn_select_run(_id(dmat), _id(dist), _id(idx), n, m, k, ex))


def dev_colsum_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    return PythonObject(dev_colsum_run(_id(a), _id(dst), _n(p, 0), _n(p, 1)))


def dev_rowsum_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    return PythonObject(dev_rowsum_run(_id(a), _id(dst), _n(p, 0), _n(p, 1)))


def dev_sqdist_py(a: PythonObject, b: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    # p = [na, nb, d] or [na, nb, d, kind, pw], as x_decomp_sqdist
    var na = _n(p, 0)
    var nb = _n(p, 1)
    var d = _n(p, 2)
    var kind = _n(p, 3) if len(p) > 3 else 0
    var pw = Float64(py=p[4]) if len(p) > 4 else Float64(2)
    return PythonObject(dev_sqdist_run(_id(a), _id(b), _id(dst), na, nb, d, kind, pw))


def dev_absmax_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    # p = [n, d, by_col], as x_decomp_absmax_sign
    var n = _n(p, 0)
    var d = _n(p, 1)
    var by_col = Int(py=p[2]) != 0
    return PythonObject(dev_absmax_run(_id(a), _id(dst), n, d, by_col))


def dev_orth_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst = the orthonormalized columns of a (a is unchanged)."""
    var m = _n(p, 0)
    var l = _n(p, 1)
    var ia = _id(a)
    var io = _id(dst)
    var r = 0
    with GILReleased(Python()):
        r = dev_orth_run(ia, io, m, l)
    return PythonObject(r)


def dev_orth_diag_py(a: PythonObject, dst: PythonObject, diag: PythonObject, p: PythonObject) raises -> PythonObject:
    """`dev_orth_py`, and the host floats at `diag` (l) = the product of the
    two passes' R diagonals."""
    var m = _n(p, 0)
    var l = _n(p, 1)
    var ia = _id(a)
    var io = _id(dst)
    var pd = F32Ptr(unsafe_from_address=Int(py=diag))
    var r = 0
    with GILReleased(Python()):
        r = dev_orth_diag_run(ia, io, pd, m, l)
    return PythonObject(r)


def dev_lda_rows_py(
    x: PythonObject, ew: PythonObject, d: PythonObject, e: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`x_decomp_lda_rows` on device matrices: d and e (n x k) updated in place
    on the device. p = [n, k, v, max_iter], f = [prior, tol]."""
    var n = _n(p, 0)
    var k = _n(p, 1)
    var v = _n(p, 2)
    var max_iter = _n(p, 3)
    var prior = Float64(py=f[0])
    var tol = Float64(py=f[1])
    return PythonObject(dev_lda_rows_run(_id(x), _id(ew), _id(d), _id(e), n, k, v, max_iter, prior, tol))


def dev_lda_bound_py(
    x: PythonObject, ddt: PythonObject, dcomp: PythonObject, dst: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """LatentDirichletAllocation's word-term bound cells on device matrices.
    p = [n, k, v], f = [floor]."""
    var n = _n(p, 0)
    var k = _n(p, 1)
    var v = _n(p, 2)
    var floor = Float64(py=f[0])
    return PythonObject(dev_lda_bound_run(_id(x), _id(ddt), _id(dcomp), _id(dst), n, k, v, floor))


def dev_als_rows_py(
    c: PythonObject, y: PythonObject, yty: PythonObject, x: PythonObject, flags: PythonObject, p: PythonObject,
    reg: PythonObject,
) raises -> PythonObject:
    """`x_decomp_als_rows` on device matrices. p = [n, m, f, su, si, c_len]."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var f = _n(p, 2)
    var su = _n(p, 3)
    var si = _n(p, 4)
    var cl = _n(p, 5)
    var rg = Float64(py=reg)
    return PythonObject(
        dev_als_rows_run(_id(c), _id(y), _id(yty), _id(x), _id(flags), n, m, f, su, si, cl, rg)
    )
