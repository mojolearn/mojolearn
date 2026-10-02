# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LocallyLinearEmbedding's null space on the SPARSE factor (lane/apple-fast-lle,
2026-10-02). FAST on Apple only, behind `-D MOJOLEARN_LLE_SPARSE_EIG`: the
x_decomp binding registers `x_decomp_dev_lle_sparse_eig` only under
`LLE_SPARSE_EIG` below, and IDENTICAL registers main's entries alone.

Main's route (python/mojolearn/_expansion_decomp.py `_lle_smallest`) writes
the dense n x n factor F = I - W, factors it with one O(n^3) LU and runs
shift-invert subspace iteration with dense products and a Jacobi SVD a
step. F has k + 1 nonzeros a row (the kNN lists `idx`, the barycenter
weights `wb`): here M = F^T F is never formed. F x is one launch over
(row, column) cells of the block X; F^T y is the same over the transpose,
held as a device CSC of W (keys (column, entry) sorted by a bitonic
network: one writer a cell, no atomics, the same order on every vendor).
The n_components smallest eigenpairs of M past the constant (u = 1 /
sqrt(n), F's exact null vector, which sklearn's ARPACK returns first and
drops) by LOBPCG on a block of b = n_components + LLE_EXTRA vectors kept
orthogonal to u: residual R = T (F^T (F X) - X Lambda) with T = diag(M)^-1
(the Jacobi preconditioner: diag(M)_j = 1 + sum of the squared in-weights
of j, from the CSC), R projected off u and X, the search space S = [X | R |
P] (n x 3b, one row-major matrix, so its Gram matrices are two `launch_gemm`
calls) reduced on the device by ONE single-block Rayleigh-Ritz kernel
(`lle_rr_kernel`, 3b <= LLE_RR_MAX = 36 cells of threadgroup memory): the
eigh of B = S^T S (two-sided Jacobi in the round-robin order of
x_decomp/rr.mojo, a fixed LLE_RR_SWEEPS sweeps), directions whose B
eigenvalue is under LLE_RR_DROP times the largest dropped (the robust
LOBPCG basis selection), A' = Z^T (F S)^T (F S) Z (the Gram of F S, so the
small eigenvalues keep float32's RELATIVE accuracy, as main's Ritz step on
F^ X does), its eigh, the b smallest Ritz pairs ascending and the P
coefficients (the R and P rows of the same vectors). The host enqueues
launches only; every LLE_SPARSE_EVERY iterations one folded read (the b
Ritz values, the wanted columns' change since the last read, ||F||_F^2)
decides main's stopping rules: the wanted subspace moved by at most
_LLE_SUBSPACE_TOL, or stalled under _LLE_STALL_TOL, or every wanted Ritz
singular value is under _LLE_NULL_FLOOR float32 epsilons of F's rms row
norm (a null space wider than the constant: any basis is the answer).
Not settled in the budget returns -1 and the caller runs main's route (an
unconverged embedding is never returned as one).
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.python import PythonObject
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from x_decomp.cells import F32Ptr, I32Ptr, rand_cell
from x_decomp.device import (
    TPB,
    ShF32,
    _blocks,
    xd_ctx,
    launch_gemm,
    gemm_scratch,
    launch_colsum,
    colsum_scratch,
    launch_ew,
)
from x_decomp.resident import pool_alloc, pool_free, _ptr, _id, _n
from x_decomp.rr import pj_first, pj_second

comptime KP = MutPointer[UInt64, MutAnyOrigin]
comptime LLE_RR_MAX = 36
"""The Rayleigh-Ritz dimension cap: 3 b, b = n_components + LLE_EXTRA, and
the caller keeps n_components + 1 < 10 (sklearn's ARPACK policy)."""
comptime LLE_B_MAX = LLE_RR_MAX // 3
comptime LLE_RR_TPB = 256
comptime LLE_RR_SWEEPS = 10
"""Cyclic (round-robin) Jacobi sweeps on a <= 36 x 36 Gram: it converges
quadratically after the first few; ten reach float32's floor."""
comptime LLE_RR_DROP = Float32(5.0e-5)
"""A B eigenvalue under this fraction of the largest is a direction float32's
Gram cannot tell from the others (unit columns): dropped."""
comptime LLE_RR_BIG = Float32(1.0e30)
comptime OP_SQ = 8
comptime LLE_RR_BYTES = (4 * LLE_RR_MAX * LLE_RR_MAX + 3 * LLE_RR_MAX + 4) * 4
comptime LLE_RR_FITS = lib_smem_page_fits_for[TARGET_COLUMN, LLE_RR_BYTES]()
comptime LLE_SPARSE_EIG = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_LLE_SPARSE_EIG"]()
    and LLE_RR_FITS
)
"""FAST on Apple with -D MOJOLEARN_LLE_SPARSE_EIG: the binding registers
`x_decomp_dev_lle_sparse_eig`. IDENTICAL never reaches this module."""


@always_inline
def _t() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _kp(p: F32Ptr) -> KP:
    """A pooled float buffer's words as uint64 keys (host side, before launch)."""
    return KP(unsafe_from_address=Int(p))


def _ip(p: F32Ptr) -> I32Ptr:
    return I32Ptr(unsafe_from_address=Int(p))


@always_inline
def _pad_key() -> UInt64:
    return ~UInt64(0)


# ---- the CSC of W (column, row) ---------------------------------------------


def lle_key_kernel(idx: F32Ptr, keys: KP, n: Int32, k: Int32, npad: Int32):
    """keys[e] = column * (n k) + e for entry e = (row, slot) of the kNN
    lists (a negative index is no entry); positions past n k, and no-entry
    slots, hold the pad key (sorts last)."""
    var t = _t()
    if t < Int(npad):
        var nk = Int(n) * Int(k)
        var key = _pad_key()
        if t < nk:
            var j = Int(idx.unsafe_load(t))
            if j >= 0 and j < Int(n):
                key = UInt64(j) * UInt64(nk) + UInt64(t)
        keys.unsafe_store(t, key)


def lle_bitonic_kernel(keys: KP, count: Int32, j: Int32, kk: Int32):
    """One compare-exchange stage of the bitonic network over `count` (a
    power of two) keys, ascending."""
    var i = _t()
    if i < Int(count):
        var ixj = i ^ Int(j)
        if ixj > i:
            var a = keys.unsafe_load(i)
            var b = keys.unsafe_load(ixj)
            var up = (i & Int(kk)) == 0
            if (up and a > b) or ((not up) and a < b):
                keys.unsafe_store(i, b)
                keys.unsafe_store(ixj, a)


def lle_csc_fill_kernel(keys: KP, wb: F32Ptr, crow: I32Ptr, cw: F32Ptr, colptr: I32Ptr, n: Int32, k: Int32, npad: Int32):
    """Sorted position t: its row and weight, and colptr[c] = t for every
    column c that starts here (the first position whose column is >= c);
    the pad run (column n) sets colptr[n] = the entry count."""
    var t = _t()
    if t < Int(npad):
        var nn = Int(n)
        var nk = nn * Int(k)
        var key = keys.unsafe_load(t)
        var col = nn
        if key != _pad_key():
            var e = Int(key % UInt64(nk))
            col = Int(key // UInt64(nk))
            crow.unsafe_store(t, Int32(e // Int(k)))
            cw.unsafe_store(t, wb.unsafe_load(e))
        var prev = -1
        if t > 0:
            var kp = keys.unsafe_load(t - 1)
            prev = nn if kp == _pad_key() else Int(kp // UInt64(nk))
        for c in range(prev + 1, col + 1):
            colptr.unsafe_store(c, Int32(t))
        if t == Int(npad) - 1 and col < nn:
            for c in range(col + 1, nn + 1):
                colptr.unsafe_store(c, Int32(npad))


def lle_diag_kernel(colptr: I32Ptr, cw: F32Ptr, dinv: F32Ptr, n: Int32):
    """dinv[j] = 1 / (1 + sum of the squared in-weights of column j): the
    Jacobi preconditioner on M = F^T F (idx excludes the row itself)."""
    var j = _t()
    if j < Int(n):
        var s = Float32(1.0)
        var p0 = Int(colptr.unsafe_load(j))
        var p1 = Int(colptr.unsafe_load(j + 1))
        for p in range(p0, p1):
            var w = cw.unsafe_load(p)
            s += w * w
        dinv.unsafe_store(j, Float32(1.0) / s)


# ---- the block operator on S = [X | R | P] (n x ld, ld = 3 b) ---------------


def lle_init_kernel(s: F32Ptr, fs: F32Ptr, n: Int32, ld: Int32, b: Int32, seed: UInt32, stream: UInt32):
    """X = uniform draws centred at 0 (main's start, `k.rand` kind 0 minus
    0.5), R and P zero; FS zero (the P block must read 0 at the first
    Rayleigh-Ritz)."""
    var t = _t()
    if t < Int(n) * Int(ld):
        var c = t % Int(ld)
        var v = Float32(0.0)
        if c < Int(b):
            var i = t // Int(ld)
            v = rand_cell(i * Int(b) + c, seed, stream, 0) - Float32(0.5)
        s.unsafe_store(t, v)
        fs.unsafe_store(t, Float32(0.0))


def lle_fx_kernel(s: F32Ptr, fs: F32Ptr, idx: F32Ptr, wb: F32Ptr, n: Int32, k: Int32, ld: Int32, c0: Int32, cnt: Int32):
    """FS[:, c0 + c] = F S[:, c0 + c] = S[i, .] - sum_a wb[i, a] S[idx[i, a], .]."""
    var t = _t()
    if t < Int(n) * Int(cnt):
        var i = t // Int(cnt)
        var c = Int(c0) + t % Int(cnt)
        var v = s.unsafe_load(i * Int(ld) + c)
        for a in range(Int(k)):
            var j = Int(idx.unsafe_load(i * Int(k) + a))
            if j >= 0:
                v -= wb.unsafe_load(i * Int(k) + a) * s.unsafe_load(j * Int(ld) + c)
        fs.unsafe_store(i * Int(ld) + c, v)


def lle_resid_kernel(
    s: F32Ptr, fs: F32Ptr, colptr: I32Ptr, crow: I32Ptr, cw: F32Ptr, dinv: F32Ptr, lam: F32Ptr, n: Int32, ld: Int32, b: Int32
):
    """R[j, c] = dinv[j] ((F^T (F X))[j, c] - X[j, c] lambda_c), written to
    S's R block; F^T y = y - W^T y over the CSC segment of column j."""
    var t = _t()
    if t < Int(n) * Int(b):
        var j = t // Int(b)
        var c = t % Int(b)
        var v = fs.unsafe_load(j * Int(ld) + c)
        var p0 = Int(colptr.unsafe_load(j))
        var p1 = Int(colptr.unsafe_load(j + 1))
        for p in range(p0, p1):
            v -= cw.unsafe_load(p) * fs.unsafe_load(Int(crow.unsafe_load(p)) * Int(ld) + c)
        v -= s.unsafe_load(j * Int(ld) + c) * lam.unsafe_load(c)
        s.unsafe_store(j * Int(ld) + Int(b) + c, v * dinv.unsafe_load(j))


def lle_deflate_kernel(s: F32Ptr, sums: F32Ptr, n: Int32, ld: Int32, c0: Int32, cnt: Int32):
    """S[:, c0 + c] -= u (u^T S[:, c0 + c]) = the column mean: kept
    orthogonal to the constant, F's null vector."""
    var t = _t()
    if t < Int(n) * Int(cnt):
        var i = t // Int(cnt)
        var c = Int(c0) + t % Int(cnt)
        var o = i * Int(ld) + c
        s.unsafe_store(o, s.unsafe_load(o) - sums.unsafe_load(c) / Float32(Int(n)))


def lle_proj_kernel(s: F32Ptr, bg: F32Ptr, n: Int32, ld: Int32, b: Int32):
    """R -= X (X^T R), X^T R read from the Gram B = S^T S (ld x ld)."""
    var t = _t()
    if t < Int(n) * Int(b):
        var i = t // Int(b)
        var c = t % Int(b)
        var o = i * Int(ld) + Int(b) + c
        var v = s.unsafe_load(o)
        for q in range(Int(b)):
            v -= s.unsafe_load(i * Int(ld) + q) * bg.unsafe_load(q * Int(ld) + Int(b) + c)
        s.unsafe_store(o, v)


def lle_scale_kernel(s: F32Ptr, fs: F32Ptr, ss: F32Ptr, n: Int32, ld: Int32, b: Int32):
    """The R and P columns of S to unit norm (a zero or non-finite column to
    zero: the Rayleigh-Ritz drops it), FS's P block by the same factors (its
    R block is computed after)."""
    var t = _t()
    var w = 2 * Int(b)
    if t < Int(n) * w:
        var i = t // w
        var c = Int(b) + t % w
        var q = ss.unsafe_load(c)
        var inv = Float32(0.0)
        if q > Float32(0.0) and q == q and q < Float32(3.0e38):
            inv = Float32(1.0) / sqrt(q)
        var o = i * Int(ld) + c
        s.unsafe_store(o, s.unsafe_load(o) * inv)
        if c >= 2 * Int(b):
            fs.unsafe_store(o, fs.unsafe_load(o) * inv)


def lle_place_kernel(s: F32Ptr, fs: F32Ptr, outs: F32Ptr, outf: F32Ptr, n: Int32, ld: Int32, b: Int32):
    """The new X (out[:, :b]) and P (out[:, b:2b]) into S's X and P blocks,
    FS alike."""
    var t = _t()
    var w = 2 * Int(b)
    if t < Int(n) * w:
        var i = t // w
        var c = t % w
        var d = c if c < Int(b) else Int(b) + c
        s.unsafe_store(i * Int(ld) + d, outs.unsafe_load(t))
        fs.unsafe_store(i * Int(ld) + d, outf.unsafe_load(t))


def lle_copy_cols_kernel(s: F32Ptr, dst: F32Ptr, n: Int32, ld: Int32, c0: Int32, cnt: Int32):
    """dst (n x cnt, contiguous) = S[:, c0 : c0 + cnt]."""
    var t = _t()
    if t < Int(n) * Int(cnt):
        var i = t // Int(cnt)
        var c = t % Int(cnt)
        dst.unsafe_store(t, s.unsafe_load(i * Int(ld) + Int(c0) + c))


def lle_err_kernel(xw: F32Ptr, yp: F32Ptr, g: F32Ptr, e: F32Ptr, n: Int32, nc: Int32):
    """E = Xw - Yp (Yp^T Xw): the wanted columns' part outside the previous
    read's wanted subspace (main's `E`), g = Yp^T Xw (nc x nc)."""
    var t = _t()
    if t < Int(n) * Int(nc):
        var i = t // Int(nc)
        var c = t % Int(nc)
        var v = xw.unsafe_load(t)
        for q in range(Int(nc)):
            v -= yp.unsafe_load(i * Int(nc) + q) * g.unsafe_load(q * Int(nc) + c)
        e.unsafe_store(t, v)


def lle_gather_kernel(lam: F32Ptr, b: Int32, err: F32Ptr, nc: Int32, fro: F32Ptr, dst: F32Ptr, cnt: Int32):
    """dst = [lam (b), err (nc), fro (1)]: the one read of a check."""
    var t = _t()
    if t < Int(cnt):
        if t < Int(b):
            dst.unsafe_store(t, lam.unsafe_load(t))
        elif t < Int(b) + Int(nc):
            dst.unsafe_store(t, err.unsafe_load(t - Int(b)))
        else:
            dst.unsafe_store(t, fro.unsafe_load(0))


# ---- the single-block Rayleigh-Ritz --------------------------------------------


@always_inline
def _rr_pair(r: Int, bb: Int, mm: Int) -> SIMD[DType.int32, 2]:
    return SIMD[DType.int32, 2](Int32(pj_first(r, bb, mm)), Int32(pj_second(r, bb, mm)))


@always_inline
def _rr_jacobi(m: ShF32, v: ShF32, cs: ShF32, dim: Int, t: Int):
    """m = J^T m J, v = v J over LLE_RR_SWEEPS sweeps of the round-robin
    pairs (x_decomp/rr.mojo's circle order): a round's pairs are disjoint,
    so its rotations commute; the (c, s) of every pair from the cells before
    the round, then the column step (m and v), a barrier, the row step. m
    ends diagonal to the sweeps' resolution, its eigenvectors in v's
    columns. Golub-Van Loan sym.schur2: tau = (a_qq - a_pp) / (2 a_pq),
    t = sgn(tau) / (|tau| + sqrt(1 + tau^2)), J = [[c, s], [-s, c]]."""
    var mm = dim + (dim % 2)
    var h = mm // 2
    for _sweep in range(LLE_RR_SWEEPS):
        for r in range(mm - 1):
            if t < h:
                var pq = _rr_pair(r, t, mm)
                var p = Int(pq[0])
                var q = Int(pq[1])
                var c = Float32(1.0)
                var s = Float32(0.0)
                if p < dim and q < dim:
                    var apq = m[p * dim + q]
                    if apq != Float32(0.0):
                        var tau = (m[q * dim + q] - m[p * dim + p]) / (Float32(2.0) * apq)
                        var sg = Float32(1.0) if tau >= Float32(0.0) else Float32(-1.0)
                        var tt = sg / (abs(tau) + sqrt(Float32(1.0) + tau * tau))
                        c = Float32(1.0) / sqrt(Float32(1.0) + tt * tt)
                        s = tt * c
                cs[2 * t] = c
                cs[2 * t + 1] = s
            barrier()
            var e = t
            while e < dim * h:
                var bb = e % h
                var kk = e // h
                var pq = _rr_pair(r, bb, mm)
                var p = Int(pq[0])
                var q = Int(pq[1])
                if p < dim and q < dim:
                    var c = cs[2 * bb]
                    var s = cs[2 * bb + 1]
                    var x = m[kk * dim + p]
                    var y = m[kk * dim + q]
                    m[kk * dim + p] = c * x - s * y
                    m[kk * dim + q] = s * x + c * y
                    x = v[kk * dim + p]
                    y = v[kk * dim + q]
                    v[kk * dim + p] = c * x - s * y
                    v[kk * dim + q] = s * x + c * y
                e += LLE_RR_TPB
            barrier()
            e = t
            while e < dim * h:
                var bb = e % h
                var kk = e // h
                var pq = _rr_pair(r, bb, mm)
                var p = Int(pq[0])
                var q = Int(pq[1])
                if p < dim and q < dim:
                    var c = cs[2 * bb]
                    var s = cs[2 * bb + 1]
                    var x = m[p * dim + kk]
                    var y = m[q * dim + kk]
                    m[p * dim + kk] = c * x - s * y
                    m[q * dim + kk] = s * x + c * y
                e += LLE_RR_TPB
            barrier()


def lle_rr_kernel(ag: F32Ptr, bg: F32Ptr, ycat: F32Ptr, lam: F32Ptr, dim_in: Int32, bw_in: Int32):
    """ONE block over the dim x dim (dim = 3 b <= LLE_RR_MAX, a compile-time
    cap, not a runtime size) Gram matrices A = (F S)^T (F S) and B = S^T S:
    eigh(B) = Q diag(beta); Z = Q diag(beta^-1/2) with the columns under
    LLE_RR_DROP beta_max zeroed (dropped directions); A' = Z^T A Z, a
    dropped direction's diagonal set to LLE_RR_BIG (sorts last); eigh(A') =
    Y' diag(theta); Y = Z Y'. Out: ycat (dim x 2 b): columns 0..b the b
    smallest theta's vectors ascending, columns b..2b the same with their X
    rows zeroed (the next P = R c_r + P c_p); lam (b) the theta's."""
    var dim = Int(dim_in)
    var bw = Int(bw_in)
    var t = Int(thread_idx.x)
    var a = stack_allocation[LLE_RR_MAX * LLE_RR_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var g = stack_allocation[LLE_RR_MAX * LLE_RR_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var v = stack_allocation[LLE_RR_MAX * LLE_RR_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var z = stack_allocation[LLE_RR_MAX * LLE_RR_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var cs = stack_allocation[LLE_RR_MAX + 4, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var w = stack_allocation[LLE_RR_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rk = stack_allocation[LLE_RR_MAX, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var dd = dim * dim
    var e = t
    while e < dd:
        g[e] = bg.unsafe_load(e)
        a[e] = ag.unsafe_load(e)
        v[e] = Float32(1.0) if e // dim == e % dim else Float32(0.0)
        e += LLE_RR_TPB
    barrier()
    _rr_jacobi(g, v, cs, dim, t)
    if t < dim:
        w[t] = g[t * dim + t]
    barrier()
    var bmax = Float32(0.0)
    for c in range(dim):
        if w[c] > bmax:
            bmax = w[c]
    var cut = LLE_RR_DROP * bmax
    e = t
    while e < dd:
        var c = e % dim
        var beta = w[c]
        z[e] = v[e] / sqrt(beta) if beta > cut else Float32(0.0)
        e += LLE_RR_TPB
    barrier()
    e = t
    while e < dd:                       # g = a z
        var r = e // dim
        var c = e % dim
        var acc = Float32(0.0)
        for q in range(dim):
            acc += a[r * dim + q] * z[q * dim + c]
        g[e] = acc
        e += LLE_RR_TPB
    barrier()
    e = t
    while e < dd:                       # a = z^T g, dropped diagonal BIG, v = I
        var r = e // dim
        var c = e % dim
        var acc = Float32(0.0)
        for q in range(dim):
            acc += z[q * dim + r] * g[q * dim + c]
        if r == c and not (w[c] > cut):
            acc = LLE_RR_BIG
        a[e] = acc
        v[e] = Float32(1.0) if r == c else Float32(0.0)
        e += LLE_RR_TPB
    barrier()
    _rr_jacobi(a, v, cs, dim, t)
    if t < dim:
        w[t] = a[t * dim + t]
    barrier()
    e = t
    while e < dd:                       # g = z v (the Ritz vectors' coefficients)
        var r = e // dim
        var c = e % dim
        var acc = Float32(0.0)
        for q in range(dim):
            acc += z[r * dim + q] * v[q * dim + c]
        g[e] = acc
        e += LLE_RR_TPB
    if t < dim:                         # ascending theta, ties to the lower column
        var cnt = 0
        for j in range(dim):
            if w[j] < w[t] or (w[j] == w[t] and j < t):
                cnt += 1
        rk[t] = Int32(cnt)
    barrier()
    e = t
    while e < dd:
        var r = e // dim
        var c = e % dim
        var rr = Int(rk[c])
        if rr < bw:
            var y = g[e]
            ycat.unsafe_store(r * (2 * bw) + rr, y)
            ycat.unsafe_store(r * (2 * bw) + bw + rr, y if r >= bw else Float32(0.0))
            if r == 0:
                lam.unsafe_store(rr, w[c])
        e += LLE_RR_TPB


# ---- the driver ------------------------------------------------------------------


def lle_sparse_eig(
    ctx: DeviceContext, idx: F32Ptr, wb: F32Ptr, xo: F32Ptr, lo: F32Ptr, n: Int, k: Int, nc: Int, b: Int,
    maxit: Int, every: Int, seed: UInt32, tol_sub: Float32, tol_stall: Float32, floor_eps: Float32,
) raises -> Int:
    """The module docstring's LOBPCG on device pointers: idx, wb (n x k),
    xo (n x nc) and lo (nc, the Ritz values ascending) out. Returns the
    iterations run when settled, -1 when not (xo, lo then unspecified)."""
    var nk = n * k
    var ld = 3 * b
    var npad = 1
    while npad < nk:
        npad *= 2
    var kid = pool_alloc(2 * npad)
    var rid = pool_alloc(npad)
    var wid = pool_alloc(npad)
    var cid = pool_alloc(n + 1)
    var did = pool_alloc(n)
    var sid = pool_alloc(n * ld)
    var fid = pool_alloc(n * ld)
    var ntmp = max(n * ld, nk)
    var tmid = pool_alloc(ntmp)
    var ssid = pool_alloc(ld)
    var smid = pool_alloc(ld)
    var nsc = colsum_scratch(n, ld)
    var scid = pool_alloc(nsc)
    var bid = pool_alloc(ld * ld)
    var aid = pool_alloc(ld * ld)
    var ngs = gemm_scratch(ld, n, ld)
    var gsid = pool_alloc(ngs)
    var yid = pool_alloc(ld * 2 * b)
    var lid = pool_alloc(b)
    var osid = pool_alloc(n * 2 * b)
    var ofid = pool_alloc(n * 2 * b)
    var xwid = pool_alloc(n * nc)
    var ypid = pool_alloc(n * nc)
    var g2id = pool_alloc(nc * nc)
    var ng2 = gemm_scratch(nc, n, nc)
    var g2sid = pool_alloc(ng2)
    var eid = pool_alloc(n * nc)
    var erid = pool_alloc(nc)
    var nes = colsum_scratch(n, nc)
    var esid = pool_alloc(nes)
    var frid = pool_alloc(1)
    var nfs = colsum_scratch(nk, 1)
    var fsid = pool_alloc(nfs)
    var nread = b + nc + 1
    var keys = _kp(_ptr(kid, 2 * npad))
    var crow = _ip(_ptr(rid, npad))
    var cw = _ptr(wid, npad)
    var colptr = _ip(_ptr(cid, n + 1))
    var dinv = _ptr(did, n)
    var s = _ptr(sid, n * ld)
    var fs = _ptr(fid, n * ld)
    var tmp = _ptr(tmid, ntmp)
    var ss = _ptr(ssid, ld)
    var sums = _ptr(smid, ld)
    var csc = _ptr(scid, nsc)
    var bg = _ptr(bid, ld * ld)
    var ag = _ptr(aid, ld * ld)
    var gsc = _ptr(gsid, ngs)
    var ycat = _ptr(yid, ld * 2 * b)
    var lam = _ptr(lid, b)
    var outs = _ptr(osid, n * 2 * b)
    var outf = _ptr(ofid, n * 2 * b)
    var xw = _ptr(xwid, n * nc)
    var yp = _ptr(ypid, n * nc)
    var g2 = _ptr(g2id, nc * nc)
    var g2s = _ptr(g2sid, ng2)
    var ee = _ptr(eid, n * nc)
    var err = _ptr(erid, nc)
    var esc = _ptr(esid, nes)
    var fro = _ptr(frid, 1)
    var frs = _ptr(fsid, nfs)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](nread)
    var db = ctx.enqueue_create_buffer[DType.float32](nread)
    var dbp = db.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    # the CSC of W: keys sorted by the bitonic network, then one writer a cell
    ctx.enqueue_function[lle_key_kernel](idx, keys, Int32(n), Int32(k), Int32(npad), grid_dim=_blocks(npad), block_dim=TPB)
    var kk = 2
    while kk <= npad:
        var j = kk // 2
        while j >= 1:
            ctx.enqueue_function[lle_bitonic_kernel](
                keys, Int32(npad), Int32(j), Int32(kk), grid_dim=_blocks(npad), block_dim=TPB
            )
            j //= 2
        kk *= 2
    ctx.enqueue_function[lle_csc_fill_kernel](
        keys, wb, crow, cw, colptr, Int32(n), Int32(k), Int32(npad), grid_dim=_blocks(npad), block_dim=TPB
    )
    ctx.enqueue_function[lle_diag_kernel](colptr, cw, dinv, Int32(n), grid_dim=_blocks(n), block_dim=TPB)
    # ||F||_F^2 - n = the sum of the squared weights (the null floor's scale)
    launch_ew(ctx, OP_SQ, wb, wb, 0, wb, 0, tmp, nk, k, Float32(0.0))
    launch_colsum(ctx, tmp, fro, frs, nk, 1)

    # X: the seeded start, orthogonal to u, unit columns, then its own
    # Rayleigh-Ritz (R and P zero: the kernel drops them)
    ctx.enqueue_function[lle_init_kernel](
        s, fs, Int32(n), Int32(ld), Int32(b), seed, UInt32(0x11E), grid_dim=_blocks(n * ld), block_dim=TPB
    )
    launch_colsum(ctx, s, sums, csc, n, ld)
    ctx.enqueue_function[lle_deflate_kernel](
        s, sums, Int32(n), Int32(ld), Int32(0), Int32(b), grid_dim=_blocks(n * b), block_dim=TPB
    )
    ctx.enqueue_function[lle_fx_kernel](
        s, fs, idx, wb, Int32(n), Int32(k), Int32(ld), Int32(0), Int32(b), grid_dim=_blocks(n * b), block_dim=TPB
    )
    launch_gemm(ctx, s, s, bg, gsc, ld, n, ld, True, False)
    launch_gemm(ctx, fs, fs, ag, gsc, ld, n, ld, True, False)
    ctx.enqueue_function[lle_rr_kernel](ag, bg, ycat, lam, Int32(ld), Int32(b), grid_dim=1, block_dim=LLE_RR_TPB)
    launch_gemm(ctx, s, ycat, outs, gsc, n, ld, 2 * b, False, False)
    launch_gemm(ctx, fs, ycat, outf, gsc, n, ld, 2 * b, False, False)
    ctx.enqueue_function[lle_place_kernel](
        s, fs, outs, outf, Int32(n), Int32(ld), Int32(b), grid_dim=_blocks(n * 2 * b), block_dim=TPB
    )

    var settled = -1
    var have_prev = False
    var checks = 0
    var e_prev = Float64(1.0e300)
    for it in range(maxit):
        # R = T (F^T F X - X Lambda), off u, off X, unit columns (P alike)
        ctx.enqueue_function[lle_resid_kernel](
            s, fs, colptr, crow, cw, dinv, lam, Int32(n), Int32(ld), Int32(b), grid_dim=_blocks(n * b), block_dim=TPB
        )
        launch_colsum(ctx, s, sums, csc, n, ld)
        ctx.enqueue_function[lle_deflate_kernel](
            s, sums, Int32(n), Int32(ld), Int32(b), Int32(b), grid_dim=_blocks(n * b), block_dim=TPB
        )
        launch_gemm(ctx, s, s, bg, gsc, ld, n, ld, True, False)
        ctx.enqueue_function[lle_proj_kernel](s, bg, Int32(n), Int32(ld), Int32(b), grid_dim=_blocks(n * b), block_dim=TPB)
        launch_ew(ctx, OP_SQ, s, s, 0, s, 0, tmp, n * ld, ld, Float32(0.0))
        launch_colsum(ctx, tmp, ss, csc, n, ld)
        ctx.enqueue_function[lle_scale_kernel](
            s, fs, ss, Int32(n), Int32(ld), Int32(b), grid_dim=_blocks(n * 2 * b), block_dim=TPB
        )
        ctx.enqueue_function[lle_fx_kernel](
            s, fs, idx, wb, Int32(n), Int32(k), Int32(ld), Int32(b), Int32(b), grid_dim=_blocks(n * b), block_dim=TPB
        )
        # the Rayleigh-Ritz on [X | R | P], then the new X and P
        launch_gemm(ctx, s, s, bg, gsc, ld, n, ld, True, False)
        launch_gemm(ctx, fs, fs, ag, gsc, ld, n, ld, True, False)
        ctx.enqueue_function[lle_rr_kernel](ag, bg, ycat, lam, Int32(ld), Int32(b), grid_dim=1, block_dim=LLE_RR_TPB)
        launch_gemm(ctx, s, ycat, outs, gsc, n, ld, 2 * b, False, False)
        launch_gemm(ctx, fs, ycat, outf, gsc, n, ld, 2 * b, False, False)
        ctx.enqueue_function[lle_place_kernel](
            s, fs, outs, outf, Int32(n), Int32(ld), Int32(b), grid_dim=_blocks(n * 2 * b), block_dim=TPB
        )
        if (it + 1) % every != 0 and it != maxit - 1:
            continue
        # one folded read: the Ritz values, the wanted columns' change, ||F||^2
        ctx.enqueue_function[lle_copy_cols_kernel](
            s, xw, Int32(n), Int32(ld), Int32(0), Int32(nc), grid_dim=_blocks(n * nc), block_dim=TPB
        )
        if have_prev:
            launch_gemm(ctx, yp, xw, g2, g2s, nc, n, nc, True, False)
            ctx.enqueue_function[lle_err_kernel](xw, yp, g2, ee, Int32(n), Int32(nc), grid_dim=_blocks(n * nc), block_dim=TPB)
            launch_ew(ctx, OP_SQ, ee, ee, 0, ee, 0, ee, n * nc, nc, Float32(0.0))
            launch_colsum(ctx, ee, err, esc, n, nc)
        ctx.enqueue_function[lle_gather_kernel](
            lam, Int32(b), err, Int32(nc), fro, dbp, Int32(nread), grid_dim=_blocks(nread), block_dim=TPB
        )
        ctx.enqueue_copy(dst_buf=hb, src_buf=db)
        ctx.synchronize()
        var hp = hb.unsafe_ptr()
        var fro2 = Float64(hp.unsafe_load(b + nc)) + Float64(n)
        var floor2 = Float64(floor_eps) * Float64(floor_eps) * fro2 / Float64(n)
        var lam_max = Float64(0.0)
        var finite = True
        for c in range(nc):
            var l = Float64(hp.unsafe_load(c))
            if not (l == l and l < 1.0e300):
                finite = False
            if l > lam_max:
                lam_max = l
        if not finite:
            break
        if checks >= 1 and lam_max <= floor2:
            settled = it + 1
        elif have_prev:
            var e2 = Float64(0.0)
            for c in range(nc):
                e2 += Float64(hp.unsafe_load(b + c))
            var e = sqrt(e2) if e2 > 0.0 else 0.0
            if e <= Float64(tol_sub) or (e <= Float64(tol_stall) and e >= e_prev):
                settled = it + 1
            e_prev = e
        var swap = xw
        xw = yp
        yp = swap
        have_prev = True
        checks += 1
        if settled >= 0:
            break
    if settled >= 0:
        ctx.enqueue_function[lle_copy_cols_kernel](
            s, xo, Int32(n), Int32(ld), Int32(0), Int32(nc), grid_dim=_blocks(n * nc), block_dim=TPB
        )
        ctx.enqueue_function[lle_copy_cols_kernel](
            lam, lo, Int32(1), Int32(b), Int32(0), Int32(nc), grid_dim=_blocks(nc), block_dim=TPB
        )
    ctx.synchronize()
    _ = db^
    _ = hb^
    pool_free(fsid)
    pool_free(frid)
    pool_free(esid)
    pool_free(erid)
    pool_free(eid)
    pool_free(g2sid)
    pool_free(g2id)
    pool_free(ypid)
    pool_free(xwid)
    pool_free(ofid)
    pool_free(osid)
    pool_free(lid)
    pool_free(yid)
    pool_free(gsid)
    pool_free(aid)
    pool_free(bid)
    pool_free(scid)
    pool_free(smid)
    pool_free(ssid)
    pool_free(tmid)
    pool_free(fid)
    pool_free(sid)
    pool_free(did)
    pool_free(cid)
    pool_free(wid)
    pool_free(rid)
    pool_free(kid)
    return settled


def dev_lle_sparse_eig_py(
    idx: PythonObject, wb: PythonObject, xo: PythonObject, lo: PythonObject, p: PythonObject, tp: PythonObject
) raises -> PythonObject:
    """Resident ids idx, wb (n x k), xo (n x nc), lo (nc); p = [n, k, nc, b,
    maxit, every, seed]; tp = [tol_sub, tol_stall, floor_eps]. Returns the
    iterations run when settled, -1 when not."""
    var n = _n(p, 0)
    var k = _n(p, 1)
    var nc = _n(p, 2)
    var b = _n(p, 3)
    var maxit = _n(p, 4)
    var every = _n(p, 5)
    var seed = UInt32(Int(py=p[6]) & 0xFFFFFFFF)
    if n < 2 or k < 1 or nc < 1 or b < nc or b > LLE_B_MAX or b >= n or maxit < 1 or every < 1:
        raise Error("x_decomp: lle_sparse_eig shape out of range")
    if n * k > 1 << 30 or n * 3 * b > 2147483647 or n > 2147483647 // 4:
        raise Error("x_decomp: lle_sparse_eig exceeds the Int32 index bound")
    var tol_sub = Float32(Float64(py=tp[0]))
    var tol_stall = Float32(Float64(py=tp[1]))
    var floor_eps = Float32(Float64(py=tp[2]))
    var got = lle_sparse_eig(
        xd_ctx(), _ptr(_id(idx), n * k), _ptr(_id(wb), n * k), _ptr(_id(xo), n * nc), _ptr(_id(lo), nc),
        n, k, nc, b, maxit, every, seed, tol_sub, tol_stall, floor_eps,
    )
    return PythonObject(got)
