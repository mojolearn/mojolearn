# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host-safe half of x_decomp/eigh_tridiag.mojo (lane
apple-neighbors-gpu-cpu, 2026-10-13): the route switch and constants, the
double-float helpers, the per-cell / per-row helpers and the per-thread items
of the bisection, gap test, twisted vectors and larft rows. No GPU import, so
the CPU column (x_decomp/eigh_tridiag_host.mojo) runs the same statements as
the device kernels (x_decomp/eigh_tridiag.mojo, which imports everything
here). Moved verbatim; no arithmetic changed."""
from std.math import fma
from std.sys.compile import is_defined

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from x_decomp.cells import F32Ptr

#: The IDENTICAL tridiagonal route (x_decomp/eigh_tridiag.mojo's module note):
#: the device's `DevExec._eigh_on` and, since 2026-10-13, the host column's
#: `HostExec.eigh` (x_decomp/eigh_tridiag_host.mojo) read it.
comptime EIGH_TD_IDN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_EIGH_TRIDIAG_OFF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
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

comptime TD_TPB = 256

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
    # the product pinned so the two-product's error term is exact (a
    # contracted `p + e` would fold the correction away)
    var p = identical_mul(a[0], b[0])
    var e = fma(a[0], b[0], -p)
    e = fma(a[0], b[1], e)
    e = fma(a[1], b[0], e)
    return _quick(p, e)


@always_inline
def df_div(a: DF, b: DF) -> DF:
    var q1 = identical_div(a[0], b[0])
    var r = df_sub(a, df_mul(b, DF(q1, Float32(0.0))))
    var q2 = identical_div(r[0], b[0])
    r = df_sub(r, df_mul(b, DF(q2, Float32(0.0))))
    var q3 = identical_div(r[0], b[0])
    return df_add(_quick(q1, q2), DF(q3, Float32(0.0)))


@always_inline
def df_sq(x: Float32) -> DF:
    var p = identical_mul(x, x)
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


@always_inline
def _td_cell_kern(a: F32Ptr, vp: F32Ptr, wp: F32Ptr, n: Int, i: Int, c: Int, jj: Int) -> Float32:
    """A_cur[i, c] = A[i, c] - sum_p (V[i, p] W[c, p] + W[i, p] V[c, p]) over
    the panel's first jj reflectors (A is stale by exactly those)."""
    var x = a.unsafe_load(i * n + c)
    for p in range(jj):
        # the V W^T term, then the W V^T term, one fused multiply-add each
        x = identical_mul_add(-vp.unsafe_load(i * TD_NB + p), wp.unsafe_load(c * TD_NB + p), x)
        x = identical_mul_add(-wp.unsafe_load(i * TD_NB + p), vp.unsafe_load(c * TD_NB + p), x)
    return x


@always_inline
def _td_house(alpha: Float32, xn2: Float32) -> SIMD[DType.float32, 4]:
    """slarfg from alpha = x[0] and xn2 = ||x[1:]||^2: (beta, tau,
    scale = 1 / (alpha - beta), 0); xn2 = 0 is the identity (tau 0)."""
    var t = Float32(0.0)
    var beta = alpha
    var scale = Float32(0.0)
    if xn2 > Float32(0.0):
        var nrm = identical_sqrt(identical_mul_add(alpha, alpha, xn2))
        beta = -nrm if alpha >= Float32(0.0) else nrm
        t = identical_div(beta - alpha, beta)
        scale = identical_div(Float32(1.0), alpha - beta)
    return SIMD[DType.float32, 4](beta, t, scale, Float32(0.0))


@always_inline
def _td_v(xcol: F32Ptr, j: Int, scale: Float32, c: Int) -> Float32:
    """v[c] for c > j: 1 at j + 1, xcol[c] scale below."""
    if c == j + 1:
        return Float32(1.0)
    return identical_mul(xcol.unsafe_load(c), scale)


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


@always_inline
def _td_y(a: F32Ptr, n: Int, k: Int, cnt: Int, i: Int, q: Int) -> Float32:
    """Y[i, q]: reflector k + q (stored in A's column k + q, rows > k + q)."""
    if q < cnt and i >= k + q + 1 and i < n:
        return a.unsafe_load(i * n + k + q)
    return Float32(0.0)



@always_inline
def td_bisect_item(dd: F32Ptr, ee: F32Ptr, info: F32Ptr, wh: F32Ptr, wl: F32Ptr, n: Int, k: Int):
    """`td_bisect_kernel`'s thread k (k < n): the k-th smallest eigenvalue of
    the scaled T by Sturm-count bisection in df64. The host column
    (x_decomp/eigh_tridiag_host.mojo) runs the same item."""
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
    var pad = identical_mul_add(Float32(1.0e-4), max(ghi - glo, max(abs(glo), abs(ghi))), Float32(1.0e-20))
    var lo = DF(glo - pad, Float32(0.0))
    var hi = DF(ghi + pad, Float32(0.0))
    for _ in range(TD_BISECT_MAX):
        var wd = df_sub(hi, lo)
        if wd[0] <= identical_mul_add(TD_EPS_DF, max(abs(lo[0]), abs(hi[0])), TD_PIVMIN):
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


@always_inline
def td_gap_item(wh: F32Ptr, wl: F32Ptr, info: F32Ptr, w_out: F32Ptr, n: Int, k: Int):
    """`td_gap_kernel`'s thread k (k < n): w_out[k] unscaled, the gap refusal."""
    var sc = info.unsafe_load(0)
    var wk = DF(wh.unsafe_load(k), wl.unsafe_load(k))
    var g = Float32(3.0e38)
    if k > 0:
        g = min(g, df_sub(wk, DF(wh.unsafe_load(k - 1), wl.unsafe_load(k - 1)))[0])
    if k < n - 1:
        g = min(g, df_sub(DF(wh.unsafe_load(k + 1), wl.unsafe_load(k + 1)), wk)[0])
    var scale = max(abs(wh.unsafe_load(0)), abs(wh.unsafe_load(n - 1)))
    if not (g >= identical_mul(TD_GAP_MIN_REL, scale)) or not td_finite(wk[0]):
        info.unsafe_store(2, Float32(1.0))
    w_out.unsafe_store(k, identical_div(wk[0] + wk[1], sc))


@always_inline
def td_vec_item(
    dd: F32Ptr, ee: F32Ptr, info: F32Ptr, wh: F32Ptr, wl: F32Ptr,
    dph: F32Ptr, dpl: F32Ptr, dmh: F32Ptr, dml: F32Ptr, z: F32Ptr, n: Int, k: Int,
):
    """`td_vec_kernel`'s thread k (k < n): T's unit eigenvector for lambda_k
    into column k of z by the twisted factorization in df64."""
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
        nrm = identical_mul_add(zf, zf, nrm)
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
        nrm = identical_mul_add(zf, zf, nrm)
        zp = zc
        zc = zn
        ii += 1
    if not td_finite(nrm) or not (nrm > Float32(0.0)):
        info.unsafe_store(2, Float32(1.0))
        return
    var inv = identical_div(Float32(1.0), identical_sqrt(nrm))
    for t in range(n):
        z.unsafe_store(t * n + k, identical_mul(z.unsafe_load(t * n + k), inv))


@always_inline
def td_tfac_item(gm: F32Ptr, tau: F32Ptr, tf: F32Ptr, n_in: Int, panel: Int, r: Int):
    """`td_tfac_kernel`'s thread r of block `panel`: row r of the panel's
    larft T (forward, columnwise)."""
    var k = panel * TD_NB
    var cnt = min(TD_NB, Int(n_in) - 1 - k)
    var offset = panel * TD_NB * TD_NB
    var row = InlineArray[Float32, TD_NB](fill=Float32(0.0))
    for i in range(TD_NB):
        if i < cnt:
            var ti = tau.unsafe_load(k + i)
            if r == i:
                row[i] = ti
            elif r < i:
                var s = Float32(0.0)
                for q in range(r, i):
                    s = identical_mul_add(row[q], gm.unsafe_load(offset + q * TD_NB + i), s)
                row[i] = identical_mul(-ti, s)
    for i in range(TD_NB):
        tf.unsafe_store(offset + r * TD_NB + i, row[i])
