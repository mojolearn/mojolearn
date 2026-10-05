# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-s-linalg (2026-10-04): TSVD_FAST_CHOLQR3, default OFF,
FAST + Apple only (`-D MOJOLEARN_TSVD_FAST_CHOLQR3`).

#: What: TruncatedSVD's TSVD_QFIX route (x_decomp/qfix.mojo bit 2) needs only
#: the R of a tall X (1M x 220 on the board) for the one-sided Jacobi of R.
#: Main gets R from the blocked Householder TSQR (x_decomp/tsqr_device.mojo):
#: scalar leaf panels of 16 columns, a WY trailing update per panel that
#: re-reads each 4096-row block ~14 times, 2 device barriers per column, and
#: a serial-per-column combine tree. rab8-tsvd: tsvd istella 363 -> 864 ms,
#: taxi 34.5 -> 42.2 ms when TSVD_QFIX replaced the Gram route.
#: This candidate gets R by shifted CholeskyQR3 (Fukaya, Kannan, Nakatsukasa,
#: Yamamoto, Yanagisawa 2020) on the matrix unit: every pass is
#:   G = Y^T Y (launch_gemm, the DECOMP_FAST_GEMM_MMA route),
#:   G_jj += c G_jj (pass 0 and 1 only: a diagonal-proportional shift, the
#:           Gram's rounding is relative to sqrt(G_ii G_jj)), floor the
#:           diagonal at CQ_FLOOR max_j G_jj (an all-zero column stays 0),
#:   L = chol(G) (the column driver `chol_step_kernel`, n launches),
#:   R_p = L^T, Y <- Y R_p^-1 (R_p^-1 by one thread per column, then
#:           launch_gemm on the matrix unit), R <- R_p R.
#: 2 shifted passes cut cond(X) (column-scaled) by ~1/sqrt(c) = 32 each,
#: then 2 plain CholeskyQR passes orthonormalize. The last pass is the
#: guard: its Gram's live diagonal must be within CQ_TOL of 1 (Y already
#: orthonormal) and no Cholesky may have failed; else the caller runs main's
#: TSQR, so quality never drops below main's. Traffic: 4 Gram reads and 3
#: read+write GEMM passes over X, against the TSQR's ~14 panel sweeps per block.
#: Quality bar (the A/B): tsvd istella relative_reconstruction_error stays
#: at the TSVD_QFIX level (1.22e-4, sklearn's), not the Gram's 2.55e-3.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import fma
from std.memory import stack_allocation
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from x_decomp.cells import F32Ptr
from x_decomp.device import (
    TPB, XD_FAST_APPLE, _blocks, _down, _p, _up, chol_step_kernel, gemm_scratch, launch_gemm,
    lu_info_init_kernel, xd_ctx,
)

comptime TSVD_FAST_CHOLQR3 = XD_FAST_APPLE and is_defined["MOJOLEARN_TSVD_FAST_CHOLQR3"]()
#: threads of the n-sized diagonal kernels (n <= TS_MAX_N = 512 columns)
comptime CQ3_TPB = 256
#: shifted passes, then plain passes (the last plain pass is the guard)
comptime CQ3_SHIFTED = 2
comptime CQ3_PASSES = 4
#: the diagonal-proportional shift of a shifted pass, 2^-10 (the MMA Gram's
#: split-K float32 sums carry ~1e-5 relative error an entry; 2^-10 keeps
#: G + c diag(G) positive definite with a wide margin)
comptime CQ3_SHIFT = Float32(0.0009765625)
#: diagonal floor relative to max_j G_jj, 2^-40: an all-zero column gets a
#: positive pivot and stays exactly 0 in Y
comptime CQ3_FLOOR = Float32(9.094947017729282e-13)
#: guard: a last-pass column is dead below 2^-30 max_j G_jj, live within
#: 2^-6 of 1
comptime CQ3_DEAD = Float32(9.313225746154785e-10)
comptime CQ3_TOL = Float32(0.015625)


def cq3_diag_kernel(g: F32Ptr, dd: F32Ptr, n_in: Int32, c: Float32):
    """small-launch(n: Gram columns): dd = diag(G) as formed, then G_jj =
    max(G_jj, CQ3_FLOOR max_j G_jj) + c G_jj. One block: thread t folds the
    max over j = t, t + CQ3_TPB, ... then a tree."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var sm = stack_allocation[CQ3_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var mx = Float32(0)
    var j = tid
    while j < n:
        var d = g.unsafe_load(j * n + j)
        dd.unsafe_store(j, d)
        if d > mx:
            mx = d
        j += CQ3_TPB
    sm[tid] = mx
    barrier()
    var half = CQ3_TPB // 2
    while half > 0:
        if tid < half and sm[tid + half] > sm[tid]:
            sm[tid] = sm[tid + half]
        barrier()
        half //= 2
    var fl = CQ3_FLOOR * sm[0]
    j = tid
    while j < n:
        var d = g.unsafe_load(j * n + j)
        var e = d if d > fl else fl
        g.unsafe_store(j * n + j, fma(c, d, e))
        j += CQ3_TPB


def cq3_rinv_kernel(l: F32Ptr, ri: F32Ptr, n_in: Int32):
    """ri = R^-1 for R = L^T (L lower n x n from `chol_step_kernel`, its upper
    triangle zero): column j of R^-1 by thread j, back substitution
    i = j - 1 .. 0, R[i, t] = L[t, i]; ri's strict lower triangle is 0."""
    var n = Int(n_in)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= n:
        return
    for i in range(j + 1, n):
        ri.unsafe_store(i * n + j, Float32(0))
    ri.unsafe_store(j * n + j, Float32(1) / l.unsafe_load(j * n + j))
    var i = j - 1
    while i >= 0:
        var s = Float32(0)
        for t in range(i + 1, j + 1):
            s = fma(l.unsafe_load(t * n + i), ri.unsafe_load(t * n + j), s)
        ri.unsafe_store(i * n + j, -s / l.unsafe_load(i * n + i))
        i -= 1


def cq3_lt_kernel(l: F32Ptr, r: F32Ptr, n_in: Int32):
    """r = L^T with an explicit zero strict lower triangle (n x n)."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < n * n:
        var i = t // n
        var j = t - i * n
        r.unsafe_store(t, l.unsafe_load(j * n + i) if j >= i else Float32(0))


def cq3_guard_kernel(dd: F32Ptr, info: F32Ptr, flag: F32Ptr, n_in: Int32, passes: Int32):
    """small-launch(n: Gram columns): flag[0] = 1 when every pass's Cholesky
    succeeded (info 0) and every last-pass diagonal is dead (<= CQ3_DEAD
    max) or within CQ3_TOL of 1; else 0. One block, a max tree then a bad tree."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var sm = stack_allocation[CQ3_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sb = stack_allocation[CQ3_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var mx = Float32(0)
    var j = tid
    while j < n:
        var d = dd.unsafe_load(j)
        if d > mx:
            mx = d
        j += CQ3_TPB
    sm[tid] = mx
    barrier()
    var half = CQ3_TPB // 2
    while half > 0:
        if tid < half and sm[tid + half] > sm[tid]:
            sm[tid] = sm[tid + half]
        barrier()
        half //= 2
    var dead = CQ3_DEAD * sm[0]
    var bad = Float32(0)
    j = tid
    while j < n:
        var d = dd.unsafe_load(j)
        if not (d <= dead or abs(d - Float32(1)) <= CQ3_TOL):
            bad = Float32(1)
        j += CQ3_TPB
    if tid < Int(passes) and info.unsafe_load(tid) != Float32(0):
        bad = Float32(1)
    sb[tid] = bad
    barrier()
    half = CQ3_TPB // 2
    while half > 0:
        if tid < half and sb[tid + half] > sb[tid]:
            sb[tid] = sb[tid + half]
        barrier()
        half //= 2
    if tid == 0:
        flag.unsafe_store(0, Float32(1) if sb[0] == Float32(0) and n > 0 else Float32(0))


def tsvd_cholqr_r_py(a: PythonObject, r: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [m, n]: r (host n x n, row major) = the upper-triangular R of a
    (host m x n, row major, m >= n) by shifted CholeskyQR3 on the device.
    Returns 1 when the guard passed (r holds R), 0 when it tripped (r is not
    meaningful; the caller runs the TSQR)."""
    var m = Int(py=p[0])
    var n = Int(py=p[1])
    if n < 1 or n > 512 or m < n or m * n > 2147483647:
        raise Error("x_decomp: tsvd_cholqr_r needs 1 <= n <= 512, m >= n and m * n inside Int32")
    var pa = F32Ptr(unsafe_from_address=Int(py=a))
    var pr = F32Ptr(unsafe_from_address=Int(py=r))
    var ok = 0
    with GILReleased(Python()):
        var ctx = xd_ctx()
        var dy = _up(ctx, pa, m * n)
        var dq = ctx.enqueue_create_buffer[DType.float32](m * n)
        var dg = ctx.enqueue_create_buffer[DType.float32](n * n)
        var dri = ctx.enqueue_create_buffer[DType.float32](n * n)
        var drp = ctx.enqueue_create_buffer[DType.float32](n * n)
        var drt = ctx.enqueue_create_buffer[DType.float32](n * n)
        var dtmp = ctx.enqueue_create_buffer[DType.float32](n * n)
        var dd = ctx.enqueue_create_buffer[DType.float32](n)
        var dinfo = ctx.enqueue_create_buffer[DType.float32](CQ3_PASSES)
        var dflag = ctx.enqueue_create_buffer[DType.float32](1)
        var g1 = gemm_scratch(n, m, n)
        var g2 = gemm_scratch(m, n, n)
        var g3 = gemm_scratch(n, n, n)
        var gs = max(g1, max(g2, g3))
        var dscr = ctx.enqueue_create_buffer[DType.float32](gs if gs > 0 else 1)
        for ps in range(CQ3_PASSES):
            var src = dy if ps % 2 == 0 else dq
            var dst = dq if ps % 2 == 0 else dy
            launch_gemm(ctx, _p(src), _p(src), _p(dg), _p(dscr), n, m, n, True, False)
            var c = CQ3_SHIFT if ps < CQ3_SHIFTED else Float32(0)
            ctx.enqueue_function[cq3_diag_kernel](  # small-launch(n: Gram columns)
                _p(dg), _p(dd), Int32(n), c, grid_dim=1, block_dim=CQ3_TPB
            )
            ctx.enqueue_function[lu_info_init_kernel](_p(dinfo) + ps, grid_dim=1, block_dim=1)
            for j in range(n):
                ctx.enqueue_function[chol_step_kernel](
                    _p(dg), _p(dinfo) + ps, Int32(j), Int32(n), grid_dim=_blocks(n - j), block_dim=TPB
                )
            ctx.enqueue_function[cq3_lt_kernel](_p(dg), _p(drp), Int32(n), grid_dim=_blocks(n * n), block_dim=TPB)
            if ps == 0:
                ctx.enqueue_copy(dst_buf=drt, src_buf=drp)
            else:
                launch_gemm(ctx, _p(drp), _p(drt), _p(dtmp), _p(dscr), n, n, n, False, False)
                ctx.enqueue_copy(dst_buf=drt, src_buf=dtmp)
            if ps < CQ3_PASSES - 1:
                ctx.enqueue_function[cq3_rinv_kernel](_p(dg), _p(dri), Int32(n), grid_dim=_blocks(n), block_dim=TPB)
                launch_gemm(ctx, _p(src), _p(dri), _p(dst), _p(dscr), m, n, n, False, False)
            # one pass per command buffer (macOS cuts a long Metal command buffer)
            ctx.synchronize()
            _ = src^
            _ = dst^
        ctx.enqueue_function[cq3_guard_kernel](  # small-launch(n: Gram columns)
            _p(dd), _p(dinfo), _p(dflag), Int32(n), Int32(CQ3_PASSES), grid_dim=1, block_dim=CQ3_TPB
        )
        var hflag = ctx.enqueue_create_host_buffer[DType.float32](1)
        ctx.enqueue_copy(dst_ptr=hflag.unsafe_ptr(), src_buf=dflag)
        _down(ctx, drt, pr, n * n)
        ctx.synchronize()
        ok = 1 if hflag.unsafe_ptr().unsafe_load(0) != Float32(0) else 0
        _ = hflag^
        _ = dy^
        _ = dq^
        _ = dg^
        _ = dri^
        _ = drp^
        _ = drt^
        _ = dtmp^
        _ = dd^
        _ = dinfo^
        _ = dflag^
        _ = dscr^
    return PythonObject(ok)
