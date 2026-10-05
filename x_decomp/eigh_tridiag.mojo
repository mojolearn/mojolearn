# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-w4-eigh (2026-10-04): FAST + Apple eigh by Householder
tridiagonalization, double-float bisection and twisted-factorization vectors,
FAST + Apple; rollback -D MOJOLEARN_EIGH_FAST_TRIDIAG_PANELS_OFF.

Why: main's eigh on the board (n = 4096) is the round-robin Jacobi. Every
round of every sweep reads and writes the whole matrix (4095 rounds a sweep,
two launches each), 43.7 s on the M3 against numpy's 4.8 s, and its float32
rotations leave a 6.1e-5 eigenvalue error (numpy 3.5e-8). The tangent and
cache repairs (EIGH_FAST_TANGENT, EIGH_TANGENT_CACHE) fixed the error but kept
the Jacobi's cost. This is a different algorithm (LAPACK's ssytrd + bisection
+ inverse iteration, laid out for the GPU):

1. Tridiagonalize, A = Q T Q^T, in panels of TD_NB columns (LAPACK latrd,
   lower). Per column four grid launches: `td_col_kernel` (the column
   updated by the panel's earlier reflectors, block partial norms, d_j),
   `td_gemv_kernel` (every block folds the partials into the reflector, then
   y = A_trail v one threadgroup a row, the 2 jj panel dot products, and the
   v / e_j / tau_j stores; v is kept in A's column j below the diagonal for
   step 4), `td_p_kernel` (p = tau (y - V W^T v - W V^T v), partial p . v)
   and `td_w_kernel` (w = p - tau/2 (p . v) v). Per panel one rank-2 TD_NB
   update of the trailing square (`td_syr2k_kernel`, 32 x 32 tiles).
2. Eigenvalues of T by Sturm-count bisection in double-float (two float32
   words, error-free two-sum / fma two-product, the compensated arithmetic
   131a0d78a already ran on the M3), one thread per eigenvalue index, T
   scaled by a power of two (exact). df64 matters: the float32 eigenvalue of
   T is not accurate enough for the vectors below.
3. Eigenvectors of T by the twisted factorization (Dhillon's one-shot inverse
   iteration): T - lambda = L+ D+ L+^T = U- D- U-^T, r = argmin |gamma_r|,
   z_r = 1 and the two bidiagonal recurrences out from r, all in df64, one
   thread per eigenvalue. Vector error ~ eps_df ||T|| / gap, eps_df ~ 1e-14,
   so no reorthogonalization is needed above TD_GAP_MIN_REL.
4. V = Q Z: panels last to first in compact WY form, Z -= Y (T_p (Y^T Z))
   (`td_gram_kernel`, `td_tfac_kernel`, `td_bt_s_kernel` split over rows,
   `td_bt_t_kernel`, `td_bt_z_kernel`).

Refusal: a gap below TD_GAP_MIN_REL max|lambda| (repeated or clustered
eigenvalues, where one-shot vectors are not orthogonal), a nonfinite or
out-of-range T, or a nonfinite vector norm sets info[2]; the caller then runs
main's round-robin Jacobi on the untouched input. One readback decides.
"""
from std.math import fma, sqrt
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.device_zero import enqueue_fill
from x_decomp.cells import F32Ptr

#: Promotion preparation, 2026-10-04; source e79a96e03d0331450a65544a7f55d06c7b140576.
#: M3 w2-eigh-panels-q-20261004: original main-relative criterion PASS across
#: eight cases, exact finite B<=A for every metric; no tolerance added.
#: Board eigerr 6.08669e-5 -> 4.26951e-7, residual 5.37499e-5 -> 6.13409e-7,
#: orthogonality .0011713 -> 1.68868e-6. Absolute 3.5e-7 target still FAIL;
#: Gram fallback orthogonality FAIL unchanged. Opponent-quality HOLD remains.
#: The handoff explicitly permits main-relative acceptance, not a fake strict PASS.
#: M3 w2-eigh-panels-t-20261004: 43765.6 -> 700.3 ms (one run/arm).
#: Default/OFF builds required before merge; same synchronized output storage.
#: OFF restores prior Jacobi, including all existing fallback behavior.
comptime EIGH_FAST_TRIDIAG = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_EIGH_FAST_TRIDIAG_PANELS_OFF"]()
)
#: FAST Apple candidate, default OFF: `-D MOJOLEARN_EIGH_FAST_PANEL_DF`
#: (`_OFF` wins). Source lane/apple-fast-eigh-panel-df@21eeacf90 (ported
#: 2026-10-04, lane apple-fast-rec-misc). Panel tridiagonalization with the
#: trailing matrix kept as double-float (hi in A, lo in an n x n buffer):
#: the rank-2 tile update, the column read (`_td_cell_df`) and the GEMV row
#: dot (y = A v, thread and block sums) run in double-float.
#: Known: M3 eigh-panel-df-q-v1 HOLD, "strict no-regression failed" (all
#: eight cases, finite B <= A per metric, no allowance). Cause: 21eeacf90
#: compensated only the syr2k update and then rounded each cell back to
#: float32 at every panel, so ~128 roundings per cell at n 4096 stayed, and
#: y = A v (the reflector's fp32 GEMV over up to n terms) stayed float32;
#: B moved no metric beyond noise. Fixed here: low words survive between
#: panels and the GEMV reads and sums them. T, reflectors, p/w, bisection,
#: vectors, back-transform and refusals unchanged; orthogonality is set by
#: the back-transform and may still compare as noise. Board 4096 eigerr
#: 4.27e-7 (A) vs numpy ~3.49e-8; one float32 rounding of T is ~6e-8.
comptime EIGH_FAST_PANEL_DF = (
    EIGH_FAST_TRIDIAG
    and is_defined["MOJOLEARN_EIGH_FAST_PANEL_DF"]()
    and not is_defined["MOJOLEARN_EIGH_FAST_PANEL_DF_OFF"]()
)
#: Panel width (reflectors per WY block). TD_TPB == 8 TD_NB is assumed by
#: the tile loads below.
comptime TD_NB = 32
#: Smallest n routed here. Was 512 ("below it main's Jacobi is cheap"; it
#: split the board's 11- and 220-wide matrices from 4096); removed as
#: benchmark-tuned on 2026-10-04, replacement UNMEASURED. The rule is now
#: the kernel's: one full TD_NB panel plus a trailing matrix for its rank-2
#: update, n >= 2 TD_NB. A refusal still falls back to the Jacobi. `-D
#: MOJOLEARN_LEGACY_NARROW_EIGH_TD` restores 512.
comptime TD_MIN_N = 512 if is_defined[
    "MOJOLEARN_LEGACY_NARROW_EIGH_TD"
]() else 2 * TD_NB
comptime TD_NBP = TD_NB + 1
comptime TD_TPB = 256
#: Threads per block of the one-thread-per-eigenvalue kernels (small blocks
#: spread n threads over more cores).
comptime TD_EIG_TPB = 32
#: Row splits of the back-transform's Y^T Z.
comptime TD_KSPLIT = 16
#: Panels between waits (bounds a command buffer's length).
comptime TD_SYNC_PANELS = 4
#: Bisection stops at this relative width (df64 carries ~48 bits).
comptime TD_EPS_DF = Float32(1.4210855e-14)
comptime TD_BISECT_MAX = 64
#: Refuse when two eigenvalues are closer than this times max |lambda|.
comptime TD_GAP_MIN_REL = Float32(1.0e-7)
#: Pivot floor of the scaled Sturm / twisted recurrences.
comptime TD_PIVMIN = Float32(1.0e-30)

comptime DF = SIMD[DType.float32, 2]


# ===========================================================================
# double-float arithmetic (hi, lo), every helper inlined (Metal pointer rule)
# ===========================================================================


@always_inline
def _two_sum(a: Float32, b: Float32) -> DF:
    var s = a + b
    var bb = s - a
    var e = (a - (s - bb)) + (b - bb)
    return DF(s, e)


@always_inline
def _quick(a: Float32, b: Float32) -> DF:
    var s = a + b
    return DF(s, b - (s - a))


@always_inline
def df_add(a: DF, b: DF) -> DF:
    var s = _two_sum(a[0], b[0])
    var t = _two_sum(a[1], b[1])
    var e = s[1] + t[0]
    var u = _quick(s[0], e)
    e = t[1] + u[1]
    return _quick(u[0], e)


@always_inline
def df_neg(a: DF) -> DF:
    return DF(-a[0], -a[1])


@always_inline
def df_sub(a: DF, b: DF) -> DF:
    return df_add(a, df_neg(b))


@always_inline
def df_mul(a: DF, b: DF) -> DF:
    var p = a[0] * b[0]
    var e = fma(a[0], b[0], -p)
    e = fma(a[0], b[1], e)
    e = fma(a[1], b[0], e)
    return _quick(p, e)


@always_inline
def df_div(a: DF, b: DF) -> DF:
    var q1 = a[0] / b[0]
    var r = df_sub(a, df_mul(b, DF(q1, Float32(0.0))))
    var q2 = r[0] / b[0]
    r = df_sub(r, df_mul(b, DF(q2, Float32(0.0))))
    var q3 = r[0] / b[0]
    return df_add(_quick(q1, q2), DF(q3, Float32(0.0)))


@always_inline
def df_sq(x: Float32) -> DF:
    var p = x * x
    return DF(p, fma(x, x, -p))


@always_inline
def df_guard(x: DF) -> DF:
    """LAPACK's pivot floor: a pivot below TD_PIVMIN becomes -TD_PIVMIN."""
    if abs(x[0]) < TD_PIVMIN:
        return DF(-TD_PIVMIN, Float32(0.0))
    return x


@always_inline
def td_finite(x: Float32) -> Bool:
    return abs(x) <= Float32(3.0e38)


@always_inline
def td_sturm_kern(dd: F32Ptr, ee: F32Ptr, n: Int, sc: Float32, x: DF) -> Int:
    """Eigenvalues of the scaled T below x (negative pivots of T - x)."""
    var q = df_guard(df_sub(DF(dd.unsafe_load(0) * sc, Float32(0.0)), x))
    var cnt = 0
    if q[0] < Float32(0.0):
        cnt += 1
    for i in range(1, n):
        var e = ee.unsafe_load(i - 1) * sc
        var dl = df_sub(DF(dd.unsafe_load(i) * sc, Float32(0.0)), x)
        q = df_guard(df_sub(dl, df_div(df_sq(e), q)))
        if q[0] < Float32(0.0):
            cnt += 1
    return cnt


# ===========================================================================
# 1. Tridiagonalization (panels of TD_NB reflectors)
# ===========================================================================


def td_copy_kernel(src: F32Ptr, dst: F32Ptr, count_in: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(count_in):
        dst.unsafe_store(t, src.unsafe_load(t))


@always_inline
def _td_cell_kern(a: F32Ptr, vp: F32Ptr, wp: F32Ptr, n: Int, i: Int, c: Int, jj: Int) -> Float32:
    """A_cur[i, c] = A[i, c] - sum_p (V[i, p] W[c, p] + W[i, p] V[c, p]) over
    the panel's first jj reflectors (A is stale by exactly those)."""
    var x = a.unsafe_load(i * n + c)
    for p in range(jj):
        x -= vp.unsafe_load(i * TD_NB + p) * wp.unsafe_load(c * TD_NB + p) + wp.unsafe_load(i * TD_NB + p) * vp.unsafe_load(c * TD_NB + p)
    return x


@always_inline
def _td_cell_df(a: F32Ptr, alo: F32Ptr, vp: F32Ptr, wp: F32Ptr, n: Int, i: Int, c: Int, jj: Int) -> Float32:
    """EIGH_FAST_PANEL_DF: `_td_cell` from the retained (hi, lo) trailing
    cell, the panel's pending rank-2 terms subtracted in double-float, one
    final rounding."""
    var x = DF(a.unsafe_load(i * n + c), alo.unsafe_load(i * n + c))
    for p in range(jj):
        var t = df_add(
            df_mul(DF(vp.unsafe_load(i * TD_NB + p), Float32(0.0)), DF(wp.unsafe_load(c * TD_NB + p), Float32(0.0))),
            df_mul(DF(wp.unsafe_load(i * TD_NB + p), Float32(0.0)), DF(vp.unsafe_load(c * TD_NB + p), Float32(0.0))),
        )
        x = df_sub(x, t)
    return x[0] + x[1]


@always_inline
def _td_cell_any(a: F32Ptr, alo: F32Ptr, vp: F32Ptr, wp: F32Ptr, n: Int, i: Int, c: Int, jj: Int) -> Float32:
    comptime if EIGH_FAST_PANEL_DF:
        return _td_cell_df(a, alo, vp, wp, n, i, c, jj)
    else:
        return _td_cell_kern(a, vp, wp, n, i, c, jj)


def td_col_kernel(
    a: F32Ptr, alo: F32Ptr, vp: F32Ptr, wp: F32Ptr, xcol: F32Ptr, part: F32Ptr, dd: F32Ptr, n_in: Int32, j_in: Int32,
    jj_in: Int32,
):
    """Column j updated by the panel's earlier reflectors: xcol[i] = A_cur[i, j]
    (rows > j), part[b] = block b's sum of xcol[i]^2 over rows > j + 1, and
    (block 0) d_j = A_cur[j, j]; at j = n - 2 also d_{n-1} (the last
    reflector is the identity). grid ceil((n - j - 1) / TD_TPB)."""
    var n = Int(n_in)
    var j = Int(j_in)
    var jj = Int(jj_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var red = stack_allocation[TD_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var i = j + 1 + b * TD_TPB + tid
    var s = Float32(0.0)
    if i < n:
        var x = _td_cell_any(a, alo, vp, wp, n, i, j, jj)
        xcol.unsafe_store(i, x)
        if i >= j + 2:
            s = x * x
    red[tid] = s
    barrier()
    var w = TD_TPB // 2
    while w > 0:
        if tid < w:
            red[tid] = red[tid] + red[tid + w]
        barrier()
        w = w // 2
    if tid == 0:
        part.unsafe_store(b, red[0])
    if b == 0 and tid == 1:
        dd.unsafe_store(j, _td_cell_any(a, alo, vp, wp, n, j, j, jj))
        if j == n - 2:
            dd.unsafe_store(n - 1, _td_cell_any(a, alo, vp, wp, n, n - 1, n - 1, jj))


@always_inline
def _td_house(alpha: Float32, xn2: Float32) -> SIMD[DType.float32, 4]:
    """slarfg from alpha = x[0] and xn2 = ||x[1:]||^2: (beta, tau,
    scale = 1 / (alpha - beta), 0); xn2 = 0 is the identity (tau 0)."""
    var t = Float32(0.0)
    var beta = alpha
    var scale = Float32(0.0)
    if xn2 > Float32(0.0):
        var nrm = sqrt(alpha * alpha + xn2)
        beta = -nrm if alpha >= Float32(0.0) else nrm
        t = (beta - alpha) / beta
        scale = Float32(1.0) / (alpha - beta)
    return SIMD[DType.float32, 4](beta, t, scale, Float32(0.0))


@always_inline
def _td_v(xcol: F32Ptr, j: Int, scale: Float32, c: Int) -> Float32:
    """v[c] for c > j: 1 at j + 1, xcol[c] scale below."""
    if c == j + 1:
        return Float32(1.0)
    return xcol.unsafe_load(c) * scale


def td_gemv_kernel(
    a: F32Ptr, alo: F32Ptr, xcol: F32Ptr, part: F32Ptr, vp: F32Ptr, wp: F32Ptr, vv: F32Ptr, tau: F32Ptr, ee: F32Ptr,
    y: F32Ptr, tv: F32Ptr, n_in: Int32, j_in: Int32, jj_in: Int32, np_in: Int32,
):
    """Every block first forms column j's reflector (`_td_house` of the
    td_col_kernel partials), then:
    blocks 0 .. m - 1 (m = n - j - 1): y[j + 1 + b] = A[j + 1 + b, j + 1 :] . v
    (stale A; td_p_kernel folds in the panel's earlier reflectors);
    blocks m .. m + 2 jj - 1: tv[q] = W[:, q] . v (q < jj), V[:, q - jj] . v;
    the last ceil(m / TD_TPB) blocks store v (vv, V[:, jj], A[j + 1 :, j])
    and e_j, tau_j."""
    var n = Int(n_in)
    var j = Int(j_in)
    var jj = Int(jj_in)
    var m = n - j - 1
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var red = stack_allocation[TD_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    # EIGH_FAST_PANEL_DF's low words of `red` (one float when off)
    var redl = stack_allocation[
        TD_TPB if EIGH_FAST_PANEL_DF else 1, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    # column j's reflector from the partial sums, folded in the same order
    # in every block (so every block holds the same scalars)
    var np_ = Int(np_in)
    var s0 = Float32(0.0)
    var q0 = tid
    while q0 < np_:
        s0 += part.unsafe_load(q0)
        q0 += TD_TPB
    red[tid] = s0
    barrier()
    var w0 = TD_TPB // 2
    while w0 > 0:
        if tid < w0:
            red[tid] = red[tid] + red[tid + w0]
        barrier()
        w0 = w0 // 2
    var hh = _td_house(xcol.unsafe_load(j + 1), red[0])
    barrier()
    var scale = hh[2]
    if b >= m + 2 * jj:
        var i = j + 1 + (b - m - 2 * jj) * TD_TPB + tid
        if i < n:
            var v = _td_v(xcol, j, scale, i)
            vv.unsafe_store(i, v)
            vp.unsafe_store(i * TD_NB + jj, v)
            a.unsafe_store(i * n + j, v)
        if b == m + 2 * jj and tid == 0:
            ee.unsafe_store(j, hh[0])
            tau.unsafe_store(j, hh[1])
        return
    var s = Float32(0.0)
    comptime if EIGH_FAST_PANEL_DF:
        if b < m:
            # y row from the retained (hi, lo) cells, products and the
            # thread's and block's sums in double-float, one final rounding.
            var row = (j + 1 + b) * n
            var c = j + 1 + tid
            var acc = DF(Float32(0.0), Float32(0.0))
            while c < n:
                acc = df_add(acc, df_mul(DF(a.unsafe_load(row + c), alo.unsafe_load(row + c)),
                                         DF(_td_v(xcol, j, scale, c), Float32(0.0))))
                c += TD_TPB
            red[tid] = acc[0]
            redl[tid] = acc[1]
            barrier()
            var wd = TD_TPB // 2
            while wd > 0:
                if tid < wd:
                    var u = df_add(DF(red[tid], redl[tid]), DF(red[tid + wd], redl[tid + wd]))
                    red[tid] = u[0]
                    redl[tid] = u[1]
                barrier()
                wd = wd // 2
            if tid == 0:
                y.unsafe_store(j + 1 + b, red[0] + redl[0])
            return
    if b < m:
        var row = (j + 1 + b) * n
        var c = j + 1 + tid
        while c < n:
            s += a.unsafe_load(row + c) * _td_v(xcol, j, scale, c)
            c += TD_TPB
    else:
        var q = b - m
        var i = j + 1 + tid
        if q < jj:
            while i < n:
                s += wp.unsafe_load(i * TD_NB + q) * _td_v(xcol, j, scale, i)
                i += TD_TPB
        else:
            var q2 = q - jj
            while i < n:
                s += vp.unsafe_load(i * TD_NB + q2) * _td_v(xcol, j, scale, i)
                i += TD_TPB
    red[tid] = s
    barrier()
    var w = TD_TPB // 2
    while w > 0:
        if tid < w:
            red[tid] = red[tid] + red[tid + w]
        barrier()
        w = w // 2
    if tid == 0:
        if b < m:
            y.unsafe_store(j + 1 + b, red[0])
        else:
            tv.unsafe_store(b - m, red[0])


def td_p_kernel(
    vp: F32Ptr, wp: F32Ptr, vv: F32Ptr, y: F32Ptr, tv: F32Ptr, tau: F32Ptr, pb: F32Ptr, part: F32Ptr,
    n_in: Int32, j_in: Int32, jj_in: Int32,
):
    """pb[i] = tau (y - V (W^T v) - W (V^T v))[i] for rows > j, part[b] = block
    b's sum of pb[i] v[i]. grid ceil((n - j - 1) / TD_TPB)."""
    var n = Int(n_in)
    var j = Int(j_in)
    var jj = Int(jj_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var red = stack_allocation[TD_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sh = stack_allocation[2 * TD_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if tid < 2 * jj:
        sh[tid] = tv.unsafe_load(tid)
    barrier()
    var i = j + 1 + b * TD_TPB + tid
    var s = Float32(0.0)
    if i < n:
        var u = y.unsafe_load(i)
        for p in range(jj):
            u -= vp.unsafe_load(i * TD_NB + p) * sh[p] + wp.unsafe_load(i * TD_NB + p) * sh[jj + p]
        var pv = tau.unsafe_load(j) * u
        pb.unsafe_store(i, pv)
        s = pv * vv.unsafe_load(i)
    red[tid] = s
    barrier()
    var w = TD_TPB // 2
    while w > 0:
        if tid < w:
            red[tid] = red[tid] + red[tid + w]
        barrier()
        w = w // 2
    if tid == 0:
        part.unsafe_store(b, red[0])


def td_w_kernel(
    wp: F32Ptr, vv: F32Ptr, pb: F32Ptr, part: F32Ptr, tau: F32Ptr, n_in: Int32, j_in: Int32, jj_in: Int32, np_in: Int32
):
    """W[i, jj] = pb[i] - tau/2 (p . v) v[i], rows > j; every block folds the
    np partial dots in the same order. grid ceil((n - j - 1) / TD_TPB)."""
    var n = Int(n_in)
    var j = Int(j_in)
    var jj = Int(jj_in)
    var np_ = Int(np_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var red = stack_allocation[TD_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s = Float32(0.0)
    var q = tid
    while q < np_:
        s += part.unsafe_load(q)
        q += TD_TPB
    red[tid] = s
    barrier()
    var w = TD_TPB // 2
    while w > 0:
        if tid < w:
            red[tid] = red[tid] + red[tid + w]
        barrier()
        w = w // 2
    var alpha2 = Float32(-0.5) * tau.unsafe_load(j) * red[0]
    var i = j + 1 + b * TD_TPB + tid
    if i < n:
        wp.unsafe_store(i * TD_NB + jj, pb.unsafe_load(i) + alpha2 * vv.unsafe_load(i))


def td_syr2k_kernel(a: F32Ptr, alo: F32Ptr, vp: F32Ptr, wp: F32Ptr, n_in: Int32, kend_in: Int32, g_in: Int32):
    """A[kend :, kend :] -= V W^T + W V^T (the panel's TD_NB columns; unused
    columns are zero). One block of TD_TPB per 32 x 32 tile, 2 x 2 cells a
    thread; g x g tiles."""
    var n = Int(n_in)
    var kend = Int(kend_in)
    var g = Int(g_in)
    var b = Int(block_idx.x)
    var by = b // g
    var bx = b - by * g
    var r0 = kend + by * 32
    var c0 = kend + bx * 32
    var tid = Int(thread_idx.x)
    var ty = tid // 16
    var tx = tid - ty * 16
    var svr = stack_allocation[32 * TD_NBP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var swr = stack_allocation[32 * TD_NBP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var svc = stack_allocation[32 * TD_NBP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var swc = stack_allocation[32 * TD_NBP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    comptime for s in range(32 * TD_NB // TD_TPB):
        var idx = tid + TD_TPB * s
        var rr = idx // TD_NB
        var q = idx - rr * TD_NB
        var ri = r0 + rr
        var ci = c0 + rr
        var a1 = Float32(0.0)
        var a2 = Float32(0.0)
        var a3 = Float32(0.0)
        var a4 = Float32(0.0)
        if ri < n:
            a1 = vp.unsafe_load(ri * TD_NB + q)
            a2 = wp.unsafe_load(ri * TD_NB + q)
        if ci < n:
            a3 = vp.unsafe_load(ci * TD_NB + q)
            a4 = wp.unsafe_load(ci * TD_NB + q)
        svr[rr * TD_NBP + q] = a1
        swr[rr * TD_NBP + q] = a2
        svc[rr * TD_NBP + q] = a3
        swc[rr * TD_NBP + q] = a4
    barrier()
    comptime if EIGH_FAST_PANEL_DF:
        # Keep the two products' rounding residuals and all 32 rank-2
        # contributions. Include A in the same compensated subtraction:
        # rounding the accumulated update before A - update loses low
        # bits when cancellation leaves a small trailing cell.
        var acc_df = InlineArray[DF, 4](fill=DF(Float32(0.0), Float32(0.0)))
        comptime for q in range(TD_NB):
            comptime for ra in range(2):
                var vr = DF(svr[(ty + 16 * ra) * TD_NBP + q], Float32(0.0))
                var wr = DF(swr[(ty + 16 * ra) * TD_NBP + q], Float32(0.0))
                comptime for cb in range(2):
                    var wc = DF(swc[(tx + 16 * cb) * TD_NBP + q], Float32(0.0))
                    var vc = DF(svc[(tx + 16 * cb) * TD_NBP + q], Float32(0.0))
                    var pair = df_add(df_mul(vr, wc), df_mul(wr, vc))
                    acc_df[ra * 2 + cb] = df_add(acc_df[ra * 2 + cb], pair)
        comptime for ra in range(2):
            comptime for cb in range(2):
                var ri = r0 + ty + 16 * ra
                var ci = c0 + tx + 16 * cb
                if ri < n and ci < n:
                    # The cell's low word survives to the next panel
                    # (21eeacf90 rounded it away here: the HOLD's cause).
                    var prior = DF(a.unsafe_load(ri * n + ci), alo.unsafe_load(ri * n + ci))
                    var updated = df_sub(prior, acc_df[ra * 2 + cb])
                    a.unsafe_store(ri * n + ci, updated[0])
                    alo.unsafe_store(ri * n + ci, updated[1])
    else:
        var acc = InlineArray[Float32, 4](fill=Float32(0.0))
        comptime for q in range(TD_NB):
            comptime for ra in range(2):
                var vr = svr[(ty + 16 * ra) * TD_NBP + q]
                var wr = swr[(ty + 16 * ra) * TD_NBP + q]
                comptime for cb in range(2):
                    acc[ra * 2 + cb] += vr * swc[(tx + 16 * cb) * TD_NBP + q] + wr * svc[(tx + 16 * cb) * TD_NBP + q]
        comptime for ra in range(2):
            comptime for cb in range(2):
                var ri = r0 + ty + 16 * ra
                var ci = c0 + tx + 16 * cb
                if ri < n and ci < n:
                    a.unsafe_store(ri * n + ci, a.unsafe_load(ri * n + ci) - acc[ra * 2 + cb])


# ===========================================================================
# 2-3. Eigenvalues and eigenvectors of T (df64, one thread per eigenvalue)
# ===========================================================================


@always_inline
def td_scale_kern(dd: F32Ptr, ee: F32Ptr, n: Int) -> SIMD[DType.float32, 2]:
    """(sc, bad): sc the power of two bringing max(|d|, |e|) into [0.5, 2)
    (exact scaling), bad = 1 when T is nonfinite or max outside
    [1e-30, 1e30]. Every thread computes it (O(n), the same everywhere)."""
    var m = Float32(0.0)
    var bad = Float32(0.0)
    for i in range(n):
        var x = dd.unsafe_load(i)
        if not td_finite(x):
            bad = Float32(1.0)
        else:
            m = max(m, abs(x))
        if i < n - 1:
            var e = ee.unsafe_load(i)
            if not td_finite(e):
                bad = Float32(1.0)
            else:
                m = max(m, abs(e))
    var sc = Float32(1.0)
    if bad > Float32(0.0) or not (m <= Float32(1.0e30)) or not (m >= Float32(1.0e-30)):
        return SIMD[DType.float32, 2](sc, Float32(1.0))
    while m >= Float32(2.0):
        m = m * Float32(0.5)
        sc = sc * Float32(0.5)
    while m < Float32(0.5):
        m = m * Float32(2.0)
        sc = sc * Float32(2.0)
    return SIMD[DType.float32, 2](sc, Float32(0.0))


def td_bisect_kernel(dd: F32Ptr, ee: F32Ptr, info: F32Ptr, wh: F32Ptr, wl: F32Ptr, n_in: Int32):
    """Thread k: the k-th smallest eigenvalue of the scaled T by Sturm-count
    bisection in df64 from the Gershgorin interval, to TD_EPS_DF relative
    width (or TD_BISECT_MAX halvings)."""
    var n = Int(n_in)
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k >= n:
        return
    var sb = td_scale_kern(dd, ee, n)
    var sc = sb[0]
    if sb[1] != Float32(0.0):
        info.unsafe_store(2, Float32(1.0))
    if k == 0:
        info.unsafe_store(0, sc)  # read by td_gap_kernel / td_vec_kernel
    var glo = Float32(3.0e38)
    var ghi = Float32(-3.0e38)
    for i in range(n):
        var r = Float32(0.0)
        if i > 0:
            r += abs(ee.unsafe_load(i - 1) * sc)
        if i < n - 1:
            r += abs(ee.unsafe_load(i) * sc)
        var d = dd.unsafe_load(i) * sc
        glo = min(glo, d - r)
        ghi = max(ghi, d + r)
    var pad = Float32(1.0e-4) * max(ghi - glo, max(abs(glo), abs(ghi))) + Float32(1.0e-20)
    var lo = DF(glo - pad, Float32(0.0))
    var hi = DF(ghi + pad, Float32(0.0))
    for _ in range(TD_BISECT_MAX):
        var wd = df_sub(hi, lo)
        if wd[0] <= TD_EPS_DF * max(abs(lo[0]), abs(hi[0])) + TD_PIVMIN:
            break
        var mid = df_add(lo, DF(wd[0] * Float32(0.5), wd[1] * Float32(0.5)))
        if td_sturm_kern(dd, ee, n, sc, mid) > k:
            hi = mid
        else:
            lo = mid
    var wf = df_sub(hi, lo)
    var lam = df_add(lo, DF(wf[0] * Float32(0.5), wf[1] * Float32(0.5)))
    wh.unsafe_store(k, lam[0])
    wl.unsafe_store(k, lam[1])


def td_gap_kernel(wh: F32Ptr, wl: F32Ptr, info: F32Ptr, w_out: F32Ptr, n_in: Int32):
    """w_out[k] = lambda_k unscaled (float32); info[2] = 1 when a neighbor
    gap is below TD_GAP_MIN_REL max |lambda| (or not ascending / finite)."""
    var n = Int(n_in)
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k >= n:
        return
    var sc = info.unsafe_load(0)
    var wk = DF(wh.unsafe_load(k), wl.unsafe_load(k))
    var g = Float32(3.0e38)
    if k > 0:
        g = min(g, df_sub(wk, DF(wh.unsafe_load(k - 1), wl.unsafe_load(k - 1)))[0])
    if k < n - 1:
        g = min(g, df_sub(DF(wh.unsafe_load(k + 1), wl.unsafe_load(k + 1)), wk)[0])
    var scale = max(abs(wh.unsafe_load(0)), abs(wh.unsafe_load(n - 1)))
    if not (g >= TD_GAP_MIN_REL * scale) or not td_finite(wk[0]):
        info.unsafe_store(2, Float32(1.0))
    w_out.unsafe_store(k, (wk[0] + wk[1]) / sc)


def td_vec_kernel(
    dd: F32Ptr, ee: F32Ptr, info: F32Ptr, wh: F32Ptr, wl: F32Ptr,
    dph: F32Ptr, dpl: F32Ptr, dmh: F32Ptr, dml: F32Ptr, z: F32Ptr, n_in: Int32,
):
    """Thread k: T's unit eigenvector for lambda_k into column k of z
    (z[i n + k]) by the twisted factorization in df64. dp* / dm* (n x n,
    [i n + k]) hold D+ and D-. A nonfinite norm sets info[2]."""
    var n = Int(n_in)
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k >= n:
        return
    var sc = info.unsafe_load(0)
    var lam = DF(wh.unsafe_load(k), wl.unsafe_load(k))
    # D+ (top down)
    var dp = df_guard(df_sub(DF(dd.unsafe_load(0) * sc, Float32(0.0)), lam))
    dph.unsafe_store(k, dp[0])
    dpl.unsafe_store(k, dp[1])
    for i in range(1, n):
        var e = ee.unsafe_load(i - 1) * sc
        var dl = df_sub(DF(dd.unsafe_load(i) * sc, Float32(0.0)), lam)
        dp = df_guard(df_sub(dl, df_div(df_sq(e), dp)))
        dph.unsafe_store(i * n + k, dp[0])
        dpl.unsafe_store(i * n + k, dp[1])
    # D- (bottom up) and the twist index r = argmin |gamma|
    var dl0 = df_sub(DF(dd.unsafe_load(n - 1) * sc, Float32(0.0)), lam)
    var dm = df_guard(dl0)
    dmh.unsafe_store((n - 1) * n + k, dm[0])
    dml.unsafe_store((n - 1) * n + k, dm[1])
    var g0 = df_sub(df_add(dp, dm), dl0)
    var best = abs(g0[0])
    var r = n - 1
    for tb in range(n - 1):
        var ib = n - 2 - tb
        var e = ee.unsafe_load(ib) * sc
        var dl = df_sub(DF(dd.unsafe_load(ib) * sc, Float32(0.0)), lam)
        dm = df_guard(df_sub(dl, df_div(df_sq(e), dm)))
        dmh.unsafe_store(ib * n + k, dm[0])
        dml.unsafe_store(ib * n + k, dm[1])
        var dpi = DF(dph.unsafe_load(ib * n + k), dpl.unsafe_load(ib * n + k))
        var g = df_sub(df_add(dpi, dm), dl)
        if abs(g[0]) < best:
            best = abs(g[0])
            r = ib
    # z_r = 1, then out from r in both directions (df64 running values)
    z.unsafe_store(r * n + k, Float32(1.0))
    var nrm = Float32(1.0)
    var zc = DF(Float32(1.0), Float32(0.0))
    var zp = DF(Float32(0.0), Float32(0.0))
    var ii = r - 1
    while ii >= 0:
        var e = ee.unsafe_load(ii) * sc
        var zn = DF(Float32(0.0), Float32(0.0))
        if zc[0] != Float32(0.0):
            var dpi = DF(dph.unsafe_load(ii * n + k), dpl.unsafe_load(ii * n + k))
            zn = df_neg(df_mul(df_div(DF(e, Float32(0.0)), dpi), zc))
        elif e != Float32(0.0):
            # z_{i+1} = 0: row i + 1 gives z_i = -(e_{i+1} / e_i) z_{i+2}
            var e1 = ee.unsafe_load(ii + 1) * sc
            zn = df_neg(df_mul(df_div(DF(e1, Float32(0.0)), DF(e, Float32(0.0))), zp))
        var zf = zn[0] + zn[1]
        z.unsafe_store(ii * n + k, zf)
        nrm += zf * zf
        zp = zc
        zc = zn
        ii -= 1
    zc = DF(Float32(1.0), Float32(0.0))
    zp = DF(Float32(0.0), Float32(0.0))
    ii = r
    while ii < n - 1:
        var e = ee.unsafe_load(ii) * sc
        var zn = DF(Float32(0.0), Float32(0.0))
        if zc[0] != Float32(0.0):
            var dmi = DF(dmh.unsafe_load((ii + 1) * n + k), dml.unsafe_load((ii + 1) * n + k))
            zn = df_neg(df_mul(df_div(DF(e, Float32(0.0)), dmi), zc))
        elif ii >= 1 and e != Float32(0.0):
            # z_i = 0: row i gives z_{i+1} = -(e_{i-1} / e_i) z_{i-1}
            var e1 = ee.unsafe_load(ii - 1) * sc
            zn = df_neg(df_mul(df_div(DF(e1, Float32(0.0)), DF(e, Float32(0.0))), zp))
        var zf = zn[0] + zn[1]
        z.unsafe_store((ii + 1) * n + k, zf)
        nrm += zf * zf
        zp = zc
        zc = zn
        ii += 1
    if not td_finite(nrm) or not (nrm > Float32(0.0)):
        info.unsafe_store(2, Float32(1.0))
        return
    var inv = Float32(1.0) / sqrt(nrm)
    for t in range(n):
        z.unsafe_store(t * n + k, z.unsafe_load(t * n + k) * inv)


# ===========================================================================
# 4. Back-transform V = Q Z, panels last to first, compact WY
# ===========================================================================


@always_inline
def _td_y(a: F32Ptr, n: Int, k: Int, cnt: Int, i: Int, q: Int) -> Float32:
    """Y[i, q]: reflector k + q (stored in A's column k + q, rows > k + q)."""
    if q < cnt and i >= k + q + 1 and i < n:
        return a.unsafe_load(i * n + k + q)
    return Float32(0.0)


def td_gram_kernel(a: F32Ptr, gm: F32Ptr, n_in: Int32):
    """gm[qa TD_NB + qb] = Y[:, qa] . Y[:, qb]; one block of TD_TPB per pair,
    TD_NB x TD_NB blocks per panel; every panel runs in grid y."""
    var n = Int(n_in)
    var panel = Int(block_idx.y)
    var k = panel * TD_NB
    var cnt = min(TD_NB, n - 1 - k)
    var b = Int(block_idx.x)
    var qa = b // TD_NB
    var qb = b - qa * TD_NB
    var tid = Int(thread_idx.x)
    var red = stack_allocation[TD_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s = Float32(0.0)
    if qa < cnt and qb < cnt:
        var i = k + 1 + tid
        while i < n:
            s += _td_y(a, n, k, cnt, i, qa) * _td_y(a, n, k, cnt, i, qb)
            i += TD_TPB
    red[tid] = s
    barrier()
    var w = TD_TPB // 2
    while w > 0:
        if tid < w:
            red[tid] = red[tid] + red[tid + w]
        barrier()
        w = w // 2
    if tid == 0:
        gm.unsafe_store(panel * TD_NB * TD_NB + b, red[0])


def td_tfac_kernel(gm: F32Ptr, tau: F32Ptr, tf: F32Ptr, n_in: Int32):
    """LAPACK larft (forward, columnwise): T upper triangular, T[i, i] =
    tau_i, T[r, i] = -tau_i sum_{s=r}^{i-1} T[r, s] G[s, i]. Thread r owns
    row r (no exchange). One block per panel, all panels in parallel."""
    var panel = Int(block_idx.x)
    var k = panel * TD_NB
    var cnt = min(TD_NB, Int(n_in) - 1 - k)
    var offset = panel * TD_NB * TD_NB
    var r = Int(thread_idx.x)
    var row = InlineArray[Float32, TD_NB](fill=Float32(0.0))
    for i in range(TD_NB):
        if i < cnt:
            var ti = tau.unsafe_load(k + i)
            if r == i:
                row[i] = ti
            elif r < i:
                var s = Float32(0.0)
                for q in range(r, i):
                    s += row[q] * gm.unsafe_load(offset + q * TD_NB + i)
                row[i] = -ti * s
    for i in range(TD_NB):
        tf.unsafe_store(offset + r * TD_NB + i, row[i])


def td_bt_s_kernel(
    a: F32Ptr, z: F32Ptr, sp: F32Ptr, n_in: Int32, k_in: Int32, cnt_in: Int32, chunk_in: Int32
):
    """sp[(sy TD_NB + q) n + c] = sum over split sy's rows of Y[i, q] Z[i, c].
    grid (ceil(n / TD_TPB), TD_KSPLIT), thread = column c."""
    var n = Int(n_in)
    var k = Int(k_in)
    var cnt = Int(cnt_in)
    var chunk = Int(chunk_in)
    var tid = Int(thread_idx.x)
    var c = Int(block_idx.x) * TD_TPB + tid
    var sy = Int(block_idx.y)
    var ib = k + 1 + sy * chunk
    var ie = min(n, ib + chunk)
    var sy_t = stack_allocation[8 * TD_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[Float32, TD_NB](fill=Float32(0.0))
    var i0 = ib
    while i0 < ie:
        var rr = tid // TD_NB
        var q = tid - rr * TD_NB
        var iy = i0 + rr
        var val = Float32(0.0)
        if iy < ie:
            val = _td_y(a, n, k, cnt, iy, q)
        sy_t[tid] = val
        barrier()
        if c < n:
            comptime for r in range(8):
                if i0 + r < ie:
                    var zv = z.unsafe_load((i0 + r) * n + c)
                    comptime for qq in range(TD_NB):
                        acc[qq] += sy_t[r * TD_NB + qq] * zv
        barrier()
        i0 += 8
    if c < n:
        comptime for qq in range(TD_NB):
            sp.unsafe_store((sy * TD_NB + qq) * n + c, acc[qq])


def td_bt_t_kernel(sp: F32Ptr, tf: F32Ptr, s2: F32Ptr, n_in: Int32):
    """s2[a n + c] = sum_{b >= a} T[a, b] sum_sy sp[(sy TD_NB + b) n + c]."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var c = Int(block_idx.x) * TD_TPB + tid
    var tsh = stack_allocation[TD_NB * TD_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    comptime for s in range(TD_NB * TD_NB // TD_TPB):
        tsh[tid + TD_TPB * s] = tf.unsafe_load(tid + TD_TPB * s)
    barrier()
    if c < n:
        var sv = InlineArray[Float32, TD_NB](fill=Float32(0.0))
        for sy in range(TD_KSPLIT):
            comptime for q in range(TD_NB):
                sv[q] += sp.unsafe_load((sy * TD_NB + q) * n + c)
        comptime for ra in range(TD_NB):
            var acc = Float32(0.0)
            comptime for q in range(ra, TD_NB):
                acc += tsh[ra * TD_NB + q] * sv[q]
            s2.unsafe_store(ra * n + c, acc)


def td_bt_z_kernel(a: F32Ptr, z: F32Ptr, s2: F32Ptr, n_in: Int32, k_in: Int32, cnt_in: Int32):
    """Z[i, c] -= sum_q Y[i, q] s2[q, c] for 16 rows i >= k + 1 a block row;
    grid (ceil(n / TD_TPB), ceil((n - k - 1) / 16))."""
    var n = Int(n_in)
    var k = Int(k_in)
    var cnt = Int(cnt_in)
    var tid = Int(thread_idx.x)
    var c = Int(block_idx.x) * TD_TPB + tid
    var i0 = k + 1 + Int(block_idx.y) * 16
    var sy_t = stack_allocation[16 * TD_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    comptime for s in range(16 * TD_NB // TD_TPB):
        var idx = tid + TD_TPB * s
        var rr = idx // TD_NB
        var q = idx - rr * TD_NB
        sy_t[idx] = _td_y(a, n, k, cnt, i0 + rr, q)
    barrier()
    if c < n:
        var sv = InlineArray[Float32, TD_NB](fill=Float32(0.0))
        comptime for q in range(TD_NB):
            sv[q] = s2.unsafe_load(q * n + c)
        comptime for r in range(16):
            var i = i0 + r
            if i < n:
                var acc = Float32(0.0)
                comptime for q in range(TD_NB):
                    acc += sy_t[r * TD_NB + q] * sv[q]
                z.unsafe_store(i * n + c, z.unsafe_load(i * n + c) - acc)


# ===========================================================================
# The host side
# ===========================================================================


def _td_grid(count: Int, tpb: Int) -> Int:
    return (count + tpb - 1) // tpb if count > 0 else 1


def eigh_td_on(
    ctx: DeviceContext,
    mut da: DeviceBuffer[DType.float32],
    n: Int,
    mut dz: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float32],
) raises -> Bool:
    """The four stages on the symmetric n x n matrix in `da` (overwritten:
    reflectors below the diagonal). On True, dw (n) holds the eigenvalues
    ascending and dz (n x n) the unit eigenvectors in COLUMNS (not yet sign
    flipped). False: refused (see the module note); dz / dw undefined.
    n >= 2."""
    var nref = n - 1
    var dV = ctx.enqueue_create_buffer[DType.float32](n * TD_NB)
    var dW = ctx.enqueue_create_buffer[DType.float32](n * TD_NB)
    var dvv = ctx.enqueue_create_buffer[DType.float32](n)
    var dy = ctx.enqueue_create_buffer[DType.float32](n)
    var dtv = ctx.enqueue_create_buffer[DType.float32](2 * TD_NB)
    var dtau = ctx.enqueue_create_buffer[DType.float32](n)
    var ddd = ctx.enqueue_create_buffer[DType.float32](n)
    var dee = ctx.enqueue_create_buffer[DType.float32](n)
    enqueue_fill(ctx, dee, Float32(0.0))
    var dx = ctx.enqueue_create_buffer[DType.float32](n)
    var dpb = ctx.enqueue_create_buffer[DType.float32](n)
    var dpart = ctx.enqueue_create_buffer[DType.float32](_td_grid(n, TD_TPB))
    var dpart2 = ctx.enqueue_create_buffer[DType.float32](_td_grid(n, TD_TPB))
    # EIGH_FAST_PANEL_DF: the trailing matrix's low words (n x n, zero = the
    # input is exact in float32); one float otherwise (never read).
    var dlo = ctx.enqueue_create_buffer[DType.float32](n * n if EIGH_FAST_PANEL_DF else 1)
    comptime if EIGH_FAST_PANEL_DF:
        enqueue_fill(ctx, dlo, Float32(0.0))
    # 1. tridiagonalize
    var k = 0
    var panel = 0
    while k < nref:
        var cnt = min(TD_NB, nref - k)
        enqueue_fill(ctx, dV, Float32(0.0))
        enqueue_fill(ctx, dW, Float32(0.0))
        for jj in range(cnt):
            var j = k + jj
            var m = n - j - 1
            var nbm = _td_grid(m, TD_TPB)
            ctx.enqueue_function[td_col_kernel](
                da.unsafe_ptr(), dlo.unsafe_ptr(), dV.unsafe_ptr(), dW.unsafe_ptr(), dx.unsafe_ptr(), dpart.unsafe_ptr(), ddd.unsafe_ptr(),
                Int32(n), Int32(j), Int32(jj),
                grid_dim=nbm, block_dim=TD_TPB,
            )
            ctx.enqueue_function[td_gemv_kernel](
                da.unsafe_ptr(), dlo.unsafe_ptr(), dx.unsafe_ptr(), dpart.unsafe_ptr(), dV.unsafe_ptr(), dW.unsafe_ptr(), dvv.unsafe_ptr(),
                dtau.unsafe_ptr(), dee.unsafe_ptr(), dy.unsafe_ptr(), dtv.unsafe_ptr(),
                Int32(n), Int32(j), Int32(jj), Int32(nbm),
                grid_dim=m + 2 * jj + nbm, block_dim=TD_TPB,
            )
            ctx.enqueue_function[td_p_kernel](
                dV.unsafe_ptr(), dW.unsafe_ptr(), dvv.unsafe_ptr(), dy.unsafe_ptr(), dtv.unsafe_ptr(), dtau.unsafe_ptr(),
                dpb.unsafe_ptr(), dpart2.unsafe_ptr(), Int32(n), Int32(j), Int32(jj),
                grid_dim=nbm, block_dim=TD_TPB,
            )
            ctx.enqueue_function[td_w_kernel](
                dW.unsafe_ptr(), dvv.unsafe_ptr(), dpb.unsafe_ptr(), dpart2.unsafe_ptr(), dtau.unsafe_ptr(),
                Int32(n), Int32(j), Int32(jj), Int32(nbm),
                grid_dim=nbm, block_dim=TD_TPB,
            )
        var kend = k + cnt
        var mm = n - kend
        if mm > 0:
            var g = (mm + 31) // 32
            ctx.enqueue_function[td_syr2k_kernel](
                da.unsafe_ptr(), dlo.unsafe_ptr(), dV.unsafe_ptr(), dW.unsafe_ptr(), Int32(n), Int32(kend), Int32(g),
                grid_dim=g * g, block_dim=TD_TPB,
            )
        k = kend
        panel += 1
        if panel % TD_SYNC_PANELS == 0:
            ctx.synchronize()
    # 2. eigenvalues
    var dinfo = ctx.enqueue_create_buffer[DType.float32](4)
    enqueue_fill(ctx, dinfo, Float32(0.0))
    var dwh = ctx.enqueue_create_buffer[DType.float32](n)
    var dwl = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[td_bisect_kernel](
        ddd.unsafe_ptr(), dee.unsafe_ptr(), dinfo.unsafe_ptr(), dwh.unsafe_ptr(), dwl.unsafe_ptr(), Int32(n),
        grid_dim=_td_grid(n, TD_EIG_TPB), block_dim=TD_EIG_TPB,
    )
    ctx.enqueue_function[td_gap_kernel](
        dwh.unsafe_ptr(), dwl.unsafe_ptr(), dinfo.unsafe_ptr(), dw.unsafe_ptr(), Int32(n),
        grid_dim=_td_grid(n, TD_EIG_TPB), block_dim=TD_EIG_TPB,
    )
    ctx.synchronize()
    # the tridiagonalization is complete (synchronized): low words released
    _ = dlo^
    # 3. eigenvectors of T
    var dph = ctx.enqueue_create_buffer[DType.float32](n * n)
    var dpl = ctx.enqueue_create_buffer[DType.float32](n * n)
    var dmh = ctx.enqueue_create_buffer[DType.float32](n * n)
    var dml = ctx.enqueue_create_buffer[DType.float32](n * n)
    ctx.enqueue_function[td_vec_kernel](
        ddd.unsafe_ptr(), dee.unsafe_ptr(), dinfo.unsafe_ptr(), dwh.unsafe_ptr(), dwl.unsafe_ptr(),
        dph.unsafe_ptr(), dpl.unsafe_ptr(), dmh.unsafe_ptr(), dml.unsafe_ptr(), dz.unsafe_ptr(), Int32(n),
        grid_dim=_td_grid(n, TD_EIG_TPB), block_dim=TD_EIG_TPB,
    )
    var hinfo = ctx.enqueue_create_host_buffer[DType.float32](4)
    ctx.enqueue_copy(dst_ptr=hinfo.unsafe_ptr(), src_buf=dinfo)
    ctx.synchronize()
    var refused = hinfo.unsafe_ptr().unsafe_load(2) != Float32(0.0)
    _ = dph^
    _ = dpl^
    _ = dmh^
    _ = dml^
    if refused:
        _ = dV^
        _ = dW^
        _ = dvv^
        _ = dy^
        _ = dtv^
        _ = dtau^
        _ = ddd^
        _ = dee^
        _ = dx^
        _ = dpb^
        _ = dpart^
        _ = dpart2^
        _ = dinfo^
        _ = dwh^
        _ = dwl^
        _ = hinfo^
        return False
    # 4. V = Q Z
    var dS = ctx.enqueue_create_buffer[DType.float32](TD_KSPLIT * TD_NB * n)
    var dS2 = ctx.enqueue_create_buffer[DType.float32](TD_NB * n)
    var npan = (nref + TD_NB - 1) // TD_NB
    var dG = ctx.enqueue_create_buffer[DType.float32](npan * TD_NB * TD_NB)
    var dT = ctx.enqueue_create_buffer[DType.float32](npan * TD_NB * TD_NB)
    # Reflectors and tau are immutable after tridiagonalization. Prepare all
    # panel factors together, preserving every row's arithmetic order.
    ctx.enqueue_function[td_gram_kernel](
        da.unsafe_ptr(), dG.unsafe_ptr(), Int32(n),
        grid_dim=(TD_NB * TD_NB, npan, 1), block_dim=TD_TPB,
    )
    ctx.enqueue_function[td_tfac_kernel](
        dG.unsafe_ptr(), dtau.unsafe_ptr(), dT.unsafe_ptr(), Int32(n),
        grid_dim=npan, block_dim=TD_NB,
    )
    var gx = _td_grid(n, TD_TPB)
    var p = npan - 1
    var done = 0
    while p >= 0:
        var kp = p * TD_NB
        var cnt = min(TD_NB, nref - kp)
        var mrows = n - kp - 1
        var chunk = (mrows + TD_KSPLIT - 1) // TD_KSPLIT
        ctx.enqueue_function[td_bt_s_kernel](
            da.unsafe_ptr(), dz.unsafe_ptr(), dS.unsafe_ptr(), Int32(n), Int32(kp), Int32(cnt), Int32(chunk),
            grid_dim=(gx, TD_KSPLIT, 1), block_dim=(TD_TPB, 1, 1),
        )
        ctx.enqueue_function[td_bt_t_kernel](
            dS.unsafe_ptr(), dT.unsafe_ptr() + p * TD_NB * TD_NB, dS2.unsafe_ptr(), Int32(n), grid_dim=gx, block_dim=TD_TPB
        )
        ctx.enqueue_function[td_bt_z_kernel](
            da.unsafe_ptr(), dz.unsafe_ptr(), dS2.unsafe_ptr(), Int32(n), Int32(kp), Int32(cnt),
            grid_dim=(gx, _td_grid(mrows, 16), 1), block_dim=(TD_TPB, 1, 1),
        )
        done += 1
        if done % TD_SYNC_PANELS == 0:
            ctx.synchronize()
        p -= 1
    ctx.synchronize()
    _ = dS^
    _ = dS2^
    _ = dG^
    _ = dT^
    _ = dV^
    _ = dW^
    _ = dvv^
    _ = dy^
    _ = dtv^
    _ = dtau^
    _ = ddd^
    _ = dee^
    _ = dx^
    _ = dpb^
    _ = dpart^
    _ = dpart2^
    _ = dinfo^
    _ = dwh^
    _ = dwl^
    _ = hinfo^
    return True
