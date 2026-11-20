# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the device-resident matrices (x_decomp/resident.mojo):
the doors the GPU binding registers. Each door unpacks ids and integers and
calls one typed entry in x_decomp/resident.mojo; no data work happens here."""
# The PACK half of the former `providers/python/x_decomp_resident_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/x_decomp_resident_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.resident import (
    pool_alloc,
    pool_free,
    pool_live_count,
    dev_upload_run,
    dev_download_run,
    dev_ew_run,
    dev_gemm_run,
    dev_rand_run,
    dev_colsum_run,
    dev_rowsum_run,
    dev_sqdist_run,
    dev_absmax_run,
    dev_orth_run,
    dev_orth_diag_run,
    dev_svd_run,
    dev_lu_run,
    dev_lu_aux_run,
    dev_code_rows_run,
    dev_eigh_run,
    dev_eigh_to_run,
    GRP_CLS2_DEVSCAN,
    GRP_CLS2_LAZY,
    GRP_FAST_FUSED,
    GRP_FAST_SRP_STRAT,
    dev_move_run,
)


from providers.python.x_decomp_resident_pyapi_unpack import (
    _id,
    _n,
)


def dev_alloc_py(n: PythonObject) raises -> PythonObject:
    return PythonObject(pool_alloc(Int(py=n)))


def dev_live_py() raises -> PythonObject:
    """The pool ids in use (a test-only readout; lane ne1-a-decomp-resident)."""
    return PythonObject(pool_live_count())


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


# TOMBSTONE: dev_upload_finite_py (named entry x_decomp_dev_upload_finite, lane a8c-device-x-linear-prep-rfe@2141a3451b)
# folded 2026-11-09 at integration (integ/pre-0867-w5) into the kernel-call op XD_DEV_UPLOAD_FINITE (lane a8d), the
# same body (x_decomp/resident.mojo dev_upload_finite_run); it never reached main, so no patch: recoverable at
# lane/a8c-device-x-linear-prep-rfe@2141a3451b.

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


# TOMBSTONE: MOJOLEARN_MCD_ORDERED_COV (its door dev_mcd_cov_py) deleted 2026-11-08 by 65b756872d; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/MOJOLEARN_MCD_ORDERED_COV.patch


# TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_decomp.patch

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

def dev_eigh_to_py(a: PythonObject, w: PythonObject, v: PythonObject, p: PythonObject) raises -> PythonObject:
    """`x_decomp_eigh` of the device matrix a (n x n, left as it is) into the
    DEVICE matrices w (n) and v (n x n). p = [n, uplo]. Waits once."""
    var n = _n(p, 0)
    var uplo = _n(p, 1)
    var ia = _id(a)
    var iw = _id(w)
    var iv = _id(v)
    var r = 0
    with GILReleased(Python()):
        r = dev_eigh_to_run(ia, iw, iv, n, uplo)
    return PythonObject(r)


def grp_cls2_py() raises -> PythonObject:
    """The random projections' FAST Apple fit switches compiled in: bit 1
    NOSCAN, bit 2 DEVSCAN, bit 4 LAZY, bit 8 FUSED, bit 16 SRP_STRAT (0 on
    every other build)."""
    var f = 0
    comptime if GRP_CLS2_DEVSCAN:
        f |= 2
    comptime if GRP_CLS2_LAZY:
        f |= 4
    comptime if GRP_FAST_FUSED:
        f |= 8
    comptime if GRP_FAST_SRP_STRAT:
        f |= 16
    return PythonObject(f)

# TOMBSTONE: MOJOLEARN_MCD_ORDERED_COV (its test seam door mcd_cov_probe_py) deleted 2026-11-08 by 65b756872d; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/MOJOLEARN_MCD_ORDERED_COV.patch
