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
    dev_colsum_run,
    dev_rowsum_run,
    dev_sqdist_run,
    dev_absmax_run,
    dev_orth_run,
    dev_orth_diag_run,
    dev_lda_rows_run,
    dev_lda_bound_run,
    dev_cd_rows_run,
    dev_svd_run,
    dev_lu_run,
    dev_lu_aux_run,
    dev_qr_r_run,
    dev_code_rows_run,
    dev_eigh_run,
    GRP_CLS2_NOSCAN,
    GRP_CLS2_DEVSCAN,
    GRP_CLS2_LAZY,
    GRP_FAST_FUSED,
    GRP_FAST_SRP_STRAT,
    dev_first_nonfinite_run,
    dev_move_run,
    grp_fit_fused_run,
    dev_maxabs_run,
    mcd_cov_probe_run,
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


def dev_cd_rows_py(
    w: PythonObject, hht: PythonObject, xht: PythonObject, perm: PythonObject, viol: PythonObject, p: PythonObject
) raises -> PythonObject:
    """`x_decomp_cd_rows` on device matrices: w (n x k, swept in place), hht
    (k x k), xht (n x k), viol (n, written); perm the host address of k
    int32. p = [n, k]. Waits."""
    var n = _n(p, 0)
    var k = _n(p, 1)
    var pp = I32Ptr(unsafe_from_address=Int(py=perm))
    return PythonObject(dev_cd_rows_run(_id(w), _id(hht), _id(xht), pp, _id(viol), n, k))


def dev_svd_py(a: PythonObject, s: PythonObject, v: PythonObject, p: PythonObject) raises -> PythonObject:
    """`x_decomp_svd` of the device matrix a (m x n, left as it is); s (n)
    and v (n x n) are host addresses. p = [m, n]. Waits."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    var ia = _id(a)
    var ps = F32Ptr(unsafe_from_address=Int(py=s))
    var pv = F32Ptr(unsafe_from_address=Int(py=v))
    var r = 0
    with GILReleased(Python()):
        r = dev_svd_run(ia, ps, pv, m, n)
    return PythonObject(r)


def dev_lu_py(a: PythonObject, lu: PythonObject, piv: PythonObject, info: PythonObject, p: PythonObject) raises -> PythonObject:
    """`x_decomp_lu` on device matrices: lu (n x n) = the factor of a (n x n,
    left as it is); piv (n int32) and info (1 float32) are host addresses.
    p = [n]. Waits."""
    var n = _n(p, 0)
    var ia = _id(a)
    var il = _id(lu)
    var pp = I32Ptr(unsafe_from_address=Int(py=piv))
    var pi = F32Ptr(unsafe_from_address=Int(py=info))
    var r = 0
    with GILReleased(Python()):
        r = dev_lu_run(ia, il, pp, pi, n)
    return PythonObject(r)


def dev_lu_aux_py(
    lu: PythonObject, piv: PythonObject, pm: PythonObject, im: PythonObject, diag: PythonObject,
    stats: PythonObject, p: PythonObject,
) raises -> PythonObject:
    """`x_decomp_lu_aux` on the device factor lu (n x n, clamped in place
    with p[1]); piv (n int32), pm, im, diag (n each) and stats (4) are host
    addresses. p = [n, clamp]. Waits."""
    var n = _n(p, 0)
    var clamp = _n(p, 1)
    var il = _id(lu)
    var pv = I32Ptr(unsafe_from_address=Int(py=piv))
    var p1 = F32Ptr(unsafe_from_address=Int(py=pm))
    var p2 = F32Ptr(unsafe_from_address=Int(py=im))
    var pd = F32Ptr(unsafe_from_address=Int(py=diag))
    var ps = F32Ptr(unsafe_from_address=Int(py=stats))
    var r = 0
    with GILReleased(Python()):
        r = dev_lu_aux_run(il, pv, p1, p2, pd, ps, n, clamp)
    return PythonObject(r)


def dev_qr_r_py(a: PythonObject, r: PythonObject, p: PythonObject) raises -> PythonObject:
    """`x_decomp_qr_r` of the device matrix a (m x n); r (n x n) is a host
    address. p = [m, n] or [m, n, consume]. Returns 1 when a was consumed,
    else 0. Waits."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    var consume = len(p) > 2 and Int(py=p[2]) != 0
    var ia = _id(a)
    var pr = F32Ptr(unsafe_from_address=Int(py=r))
    var res = 0
    with GILReleased(Python()):
        res = dev_qr_r_run(ia, pr, m, n, consume)
    return PythonObject(res)


def dev_code_rows_py(g: PythonObject, q: PythonObject, w: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """g (k x k), q (n x k) and w (n x k) are device matrices.
    p = [kind, n, k, a, b]: kind 0 Lasso CD, kind 1 Lars, kind 2 OMP;
    f = [alpha, tol]. Waits."""
    var kind = Int(py=p[0])
    var n = _n(p, 1)
    var k = _n(p, 2)
    var a = _n(p, 3)
    var b = _n(p, 4)
    var ig = _id(g)
    var iq = _id(q)
    var iw = _id(w)
    var alpha = Float64(py=f[0])
    var tol = Float64(py=f[1])
    var r = 0
    with GILReleased(Python()):
        r = dev_code_rows_run(ig, iq, iw, kind, n, k, a, b, alpha, tol)
    return PythonObject(r)


def dev_eigh_py(a: PythonObject, w: PythonObject, v: PythonObject, p: PythonObject) raises -> PythonObject:
    """`x_decomp_eigh` of the device matrix a (n x n, left as it is); w (n)
    and v (n x n) are host addresses. p = [n, uplo]. Waits."""
    var n = _n(p, 0)
    var uplo = _n(p, 1)
    var ia = _id(a)
    var pw = F32Ptr(unsafe_from_address=Int(py=w))
    var pv = F32Ptr(unsafe_from_address=Int(py=v))
    var r = 0
    with GILReleased(Python()):
        r = dev_eigh_run(ia, pw, pv, n, uplo)
    return PythonObject(r)


def grp_cls2_py() raises -> PythonObject:
    """The random projections' FAST Apple fit switches compiled in: bit 1
    NOSCAN, bit 2 DEVSCAN, bit 4 LAZY, bit 8 FUSED, bit 16 SRP_STRAT (0 on
    every other build)."""
    var f = 0
    comptime if GRP_CLS2_NOSCAN:
        f |= 1
    comptime if GRP_CLS2_DEVSCAN:
        f |= 2
    comptime if GRP_CLS2_LAZY:
        f |= 4
    comptime if GRP_FAST_FUSED:
        f |= 8
    comptime if GRP_FAST_SRP_STRAT:
        f |= 16
    return PythonObject(f)


def dev_first_nonfinite_py(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """The first flat index of a NaN or infinity among the n floats at
    `addr`, or -1. Registered only under GRP_CLS2_DEVSCAN."""
    var cnt = Int(py=n)
    var src = F32Ptr(unsafe_from_address=Int(py=addr))
    var r = -1
    with GILReleased(Python()):
        r = dev_first_nonfinite_run(src, cnt)
    return PythonObject(r)


def dev_move_py(src: PythonObject, idx: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """`x_decomp_move` on device ids, enqueued (no sync); p = [op, count, a1,
    a2, a3, ist, ioff, nsrc, ndst, nidx]."""
    var op = Int(py=p[0])
    var count = _n(p, 1)
    var a1 = _n(p, 2)
    var a2 = _n(p, 3)
    var a3 = _n(p, 4)
    var ist = _n(p, 5)
    var ioff = _n(p, 6)
    var nsrc = _n(p, 7)
    var ndst = _n(p, 8)
    var nidx = _n(p, 9)
    return PythonObject(
        dev_move_run(_id(src), _id(idx), _id(dst), op, count, a1, a2, a3, ist, ioff, nsrc, ndst, nidx)
    )


def grp_fit_fused_py(
    xaddr: PythonObject, n: PythonObject, dst: PythonObject, hout: PythonObject, p: PythonObject, s: PythonObject
) raises -> PythonObject:
    """p = [count, seed, mode] (or [count, seed, 3, n_features, D]), s =
    [scale, threshold]: the first flat index of a NaN or infinity among the
    n floats at xaddr (or -1), the projection matrix into the device matrix
    dst, and its words copied to the host floats at hout. Registered only
    under GRP_FAST_FUSED."""
    var cnt = Int(py=n)
    var count = _n(p, 0)
    var seed = Int(py=p[1])
    var mode = Int(py=p[2])
    var sd = 1
    var sgrid = 0
    comptime if GRP_FAST_SRP_STRAT:
        if mode == 3:
            sd = _n(p, 3)
            sgrid = _n(p, 4)
    var sc = Float64(py=s[0])
    var thr = Float64(py=s[1])
    var src = F32Ptr(unsafe_from_address=Int(py=xaddr))
    var host_out = F32Ptr(unsafe_from_address=Int(py=hout))
    var idst = _id(dst)
    var r = -1
    with GILReleased(Python()):
        r = grp_fit_fused_run(src, cnt, idst, host_out, count, seed, mode, sd, sgrid, sc, thr)
    return PythonObject(r)


def dev_maxabs_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst (a device matrix of >= 1 value) = max |a| over the first p[0]
    values of the device matrix a (NaN if any is NaN); no wait."""
    return PythonObject(dev_maxabs_run(_id(a), _id(dst), _n(p, 0)))


# Test-only direct batched covariance seam, identical API on both arms.
def mcd_cov_probe_py(xaddr: PythonObject, gaddr: PythonObject, outaddr: PythonObject,
                     p: PythonObject) raises -> PythonObject:
    var nc = _n(p, 0)
    var rows = _n(p, 1)
    var d = _n(p, 2)
    var px = F32Ptr(unsafe_from_address=Int(py=xaddr))
    var pg = I32Ptr(unsafe_from_address=Int(py=gaddr))
    var pout = F32Ptr(unsafe_from_address=Int(py=outaddr))
    return PythonObject(mcd_cov_probe_run(px, pg, pout, nc, rows, d))
