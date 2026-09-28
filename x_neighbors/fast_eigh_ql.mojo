# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST's dense symmetric eigensolver on the host: Householder
tridiagonalization and the implicit QL iteration in binary64 (lane
neighbors-apple3, 2026-09-28). Its own module, imported only where a build
selects it (x_neighbors/eigh.mojo, kernel_methods' Nystroem)."""
from std.sys.info import simd_width_of

from x_neighbors.fast_eigh import EigP

# ---------------------------------------------------------------------------
# FAST: Householder tridiagonalization and the implicit QL iteration
# ---------------------------------------------------------------------------
#
# lane neighbors-apple3 (2026-09-28). The Jacobi of fast_eigh.mojo is about
# 0.4 us a rotation at n = 500 and a solve is about a million rotations. The classical
# dense route (EISPACK's tred2 and tql2, the statements of JAMA's
# EigenvalueDecomposition, public domain) is a tenth of the arithmetic:
# reduce to tridiagonal form by Householder reflections, accumulate them,
# then the implicit QL iteration with its rotations applied to the basis.
# In binary64 on the host, the basis held TRANSPOSED so that every inner
# loop of the three O(n^3) phases runs over contiguous memory. The float32
# outputs are the binary64 results rounded once. FAST only: the answer is
# the same decomposition by another route, so FAST's words move and the
# paired quality check decides (bench/x_neighbors_fast_quality.py: kpca,
# nystroem).
from std.math import sqrt as _sqrt64

comptime QL_W = simd_width_of[DType.float64]()
comptime QlV = SIMD[DType.float64, QL_W]
comptime QlP = MutPointer[Float64, MutAnyOrigin]
comptime QL_MAX_ITER = 60


@always_inline
def _dot64(a: QlP, b: QlP, lo: Int, hi: Int) -> Float64:
    """sum a[k] * b[k], k in [lo, hi)."""
    var s0 = QlV(0.0)
    var s1 = QlV(0.0)
    var s2 = QlV(0.0)
    var s3 = QlV(0.0)
    var k = lo
    while k + 4 * QL_W <= hi:
        s0 = s0 + a.unsafe_load[width=QL_W](k) * b.unsafe_load[width=QL_W](k)
        s1 = s1 + a.unsafe_load[width=QL_W](k + QL_W) * b.unsafe_load[width=QL_W](k + QL_W)
        s2 = s2 + a.unsafe_load[width=QL_W](k + 2 * QL_W) * b.unsafe_load[width=QL_W](k + 2 * QL_W)
        s3 = s3 + a.unsafe_load[width=QL_W](k + 3 * QL_W) * b.unsafe_load[width=QL_W](k + 3 * QL_W)
        k += 4 * QL_W
    var s = ((s0 + s1) + (s2 + s3)).reduce_add()
    while k < hi:
        s += a.unsafe_load(k) * b.unsafe_load(k)
        k += 1
    return s


@always_inline
def _axpy64(y: QlP, x: QlP, alpha: Float64, lo: Int, hi: Int):
    """y[k] += alpha * x[k], k in [lo, hi)."""
    var av = QlV(alpha)
    var k = lo
    while k + QL_W <= hi:
        y.unsafe_store(k, y.unsafe_load[width=QL_W](k) + av * x.unsafe_load[width=QL_W](k))
        k += QL_W
    while k < hi:
        y.unsafe_store(k, y.unsafe_load(k) + alpha * x.unsafe_load(k))
        k += 1


@always_inline
def _axpy2_64(y: QlP, x1: QlP, a1: Float64, x2: QlP, a2: Float64, lo: Int, hi: Int):
    """y[k] -= a1 * x1[k] + a2 * x2[k], k in [lo, hi)."""
    var v1 = QlV(a1)
    var v2 = QlV(a2)
    var k = lo
    while k + QL_W <= hi:
        y.unsafe_store(
            k,
            y.unsafe_load[width=QL_W](k)
            - (v1 * x1.unsafe_load[width=QL_W](k) + v2 * x2.unsafe_load[width=QL_W](k)),
        )
        k += QL_W
    while k < hi:
        y.unsafe_store(k, y.unsafe_load(k) - (a1 * x1.unsafe_load(k) + a2 * x2.unsafe_load(k)))
        k += 1


@always_inline
def _rot64(lo_row: QlP, hi_row: QlP, n: Int, c: Float64, s: Float64):
    """tql2's accumulation over rows i (lo_row) and i + 1 (hi_row) of the
    transposed basis: h = hi[k]; hi[k] = s * lo[k] + c * h;
    lo[k] = c * lo[k] - s * h."""
    var cv = QlV(c)
    var sv = QlV(s)
    var k = 0
    while k + QL_W <= n:
        var h = hi_row.unsafe_load[width=QL_W](k)
        var l = lo_row.unsafe_load[width=QL_W](k)
        hi_row.unsafe_store(k, sv * l + cv * h)
        lo_row.unsafe_store(k, cv * l - sv * h)
        k += QL_W
    while k < n:
        var h1 = hi_row.unsafe_load(k)
        var l1 = lo_row.unsafe_load(k)
        hi_row.unsafe_store(k, s * l1 + c * h1)
        lo_row.unsafe_store(k, c * l1 - s * h1)
        k += 1


@always_inline
def _hypot64(a: Float64, b: Float64) -> Float64:
    var aa = abs(a)
    var bb = abs(b)
    if aa > bb:
        var r = bb / aa
        return aa * _sqrt64(Float64(1.0) + r * r)
    if bb != Float64(0.0):
        var r2 = aa / bb
        return bb * _sqrt64(Float64(1.0) + r2 * r2)
    return Float64(0.0)


def symmetric_eig_ql(src: EigP, n: Int, evals: EigP, evecs: EigP) raises -> Int:
    """`symmetric_eig_rows`'s contract (upper triangle and diagonal of `src`
    read; eigenvalues ascending by (value, index); eigenvector c in COLUMN
    c, signs pinned by DEVIATION 770's rule) by tred2 and tql2 in binary64.
    Returns the largest QL iteration count of any eigenvalue."""
    if n <= 0:
        raise Error("symmetric_eig_ql: n must be positive")
    var zero = Float64(0.0)
    var one = Float64(1.0)
    var vt_store = List[Float64](length=n * n, fill=zero)
    var d_store = List[Float64](length=n, fill=zero)
    var e_store = List[Float64](length=n, fill=zero)
    var vt = QlP(unsafe_from_address=Int(vt_store.unsafe_ptr()))
    var d = QlP(unsafe_from_address=Int(d_store.unsafe_ptr()))
    var e = QlP(unsafe_from_address=Int(e_store.unsafe_ptr()))
    # vt[j][k] is V[k][j]; the input is symmetric, so vt starts as the input
    for r0 in range(n):
        vt.unsafe_store(r0 * n + r0, Float64(src.unsafe_load(r0 * n + r0)))
        for c0 in range(r0 + 1, n):
            var x = Float64(src.unsafe_load(r0 * n + c0))
            vt.unsafe_store(r0 * n + c0, x)
            vt.unsafe_store(c0 * n + r0, x)

    # ---- tred2
    for j in range(n):
        d.unsafe_store(j, vt.unsafe_load(j * n + n - 1))
    var hi_ = n - 1
    while hi_ > 0:
        var scale = zero
        var h = zero
        for k in range(hi_):
            scale = scale + abs(d.unsafe_load(k))
        if scale == zero:
            e.unsafe_store(hi_, d.unsafe_load(hi_ - 1))
            for j in range(hi_):
                d.unsafe_store(j, vt.unsafe_load(j * n + hi_ - 1))
                vt.unsafe_store(j * n + hi_, zero)
                vt.unsafe_store(hi_ * n + j, zero)
        else:
            for k in range(hi_):
                var dk = d.unsafe_load(k) / scale
                d.unsafe_store(k, dk)
                h += dk * dk
            var f = d.unsafe_load(hi_ - 1)
            var g = _sqrt64(h)
            if f > zero:
                g = -g
            e.unsafe_store(hi_, scale * g)
            h = h - f * g
            d.unsafe_store(hi_ - 1, f - g)
            for j in range(hi_):
                e.unsafe_store(j, zero)
            for j in range(hi_):
                var rj = vt.unsafe_offset(j * n)
                f = d.unsafe_load(j)
                vt.unsafe_store(hi_ * n + j, f)
                g = e.unsafe_load(j) + rj.unsafe_load(j) * f
                g += _dot64(rj, d, j + 1, hi_)
                _axpy64(e, rj, f, j + 1, hi_)
                e.unsafe_store(j, g)
            f = zero
            for j in range(hi_):
                var ej = e.unsafe_load(j) / h
                e.unsafe_store(j, ej)
                f += ej * d.unsafe_load(j)
            var hh = f / (h + h)
            for j in range(hi_):
                e.unsafe_store(j, e.unsafe_load(j) - hh * d.unsafe_load(j))
            for j in range(hi_):
                var rj2 = vt.unsafe_offset(j * n)
                f = d.unsafe_load(j)
                g = e.unsafe_load(j)
                _axpy2_64(rj2, e, f, d, g, j, hi_)
                d.unsafe_store(j, rj2.unsafe_load(hi_ - 1))
                rj2.unsafe_store(hi_, zero)
        d.unsafe_store(hi_, h)
        hi_ -= 1

    # ---- accumulate the reflections
    for a in range(n - 1):
        var ra = vt.unsafe_offset(a * n)
        var rn = vt.unsafe_offset((a + 1) * n)
        ra.unsafe_store(n - 1, ra.unsafe_load(a))
        ra.unsafe_store(a, one)
        var h2 = d.unsafe_load(a + 1)
        if h2 != zero:
            for k in range(a + 1):
                d.unsafe_store(k, rn.unsafe_load(k) / h2)
            for j in range(a + 1):
                var rj3 = vt.unsafe_offset(j * n)
                var g2 = _dot64(rn, rj3, 0, a + 1)
                _axpy64(rj3, d, -g2, 0, a + 1)
        for k in range(a + 1):
            rn.unsafe_store(k, zero)
    for j in range(n):
        d.unsafe_store(j, vt.unsafe_load(j * n + n - 1))
        vt.unsafe_store(j * n + n - 1, zero)
    vt.unsafe_store((n - 1) * n + n - 1, one)
    e.unsafe_store(0, zero)

    # ---- tql2
    for a in range(1, n):
        e.unsafe_store(a - 1, e.unsafe_load(a))
    e.unsafe_store(n - 1, zero)
    var fsh = zero
    var tst1 = zero
    var eps = Float64(2.220446049250313e-16)
    var worst_iter = 0
    for l in range(n):
        var t1 = abs(d.unsafe_load(l)) + abs(e.unsafe_load(l))
        if t1 > tst1:
            tst1 = t1
        var m = l
        while m < n:
            if abs(e.unsafe_load(m)) <= eps * tst1:
                break
            m += 1
        if m >= n:
            m = n - 1
        if m > l:
            var iters = 0
            while True:
                iters += 1
                if iters > QL_MAX_ITER:
                    raise Error(
                        "symmetric_eig_ql: the QL iteration did not converge in "
                        + String(QL_MAX_ITER)
                        + " steps at eigenvalue "
                        + String(l)
                    )
                var g3 = d.unsafe_load(l)
                var el = e.unsafe_load(l)
                var p = (d.unsafe_load(l + 1) - g3) / (Float64(2.0) * el)
                var r = _hypot64(p, one)
                if p < zero:
                    r = -r
                d.unsafe_store(l, el / (p + r))
                d.unsafe_store(l + 1, el * (p + r))
                var dl1 = d.unsafe_load(l + 1)
                var h3 = g3 - d.unsafe_load(l)
                for a in range(l + 2, n):
                    d.unsafe_store(a, d.unsafe_load(a) - h3)
                fsh = fsh + h3
                p = d.unsafe_load(m)
                var c = one
                var c2 = one
                var c3 = one
                var el1 = e.unsafe_load(l + 1)
                var s = zero
                var s2 = zero
                var a2 = m - 1
                while a2 >= l:
                    c3 = c2
                    c2 = c
                    s2 = s
                    var ea = e.unsafe_load(a2)
                    var g4 = c * ea
                    var h4 = c * p
                    r = _hypot64(p, ea)
                    e.unsafe_store(a2 + 1, s * r)
                    s = ea / r
                    c = p / r
                    var da = d.unsafe_load(a2)
                    p = c * da - s * g4
                    d.unsafe_store(a2 + 1, h4 + s * (c * g4 + s * da))
                    _rot64(vt.unsafe_offset(a2 * n), vt.unsafe_offset((a2 + 1) * n), n, c, s)
                    a2 -= 1
                p = -s * s2 * c3 * el1 * e.unsafe_load(l) / dl1
                e.unsafe_store(l, s * p)
                d.unsafe_store(l, c * p)
                if not (abs(e.unsafe_load(l)) > eps * tst1):
                    break
            if iters > worst_iter:
                worst_iter = iters
        d.unsafe_store(l, d.unsafe_load(l) + fsh)
        e.unsafe_store(l, zero)

    # ---- ascending by (value, index) of the float32 values, signs pinned
    var w32 = List[Float32](capacity=n)
    for a in range(n):
        w32.append(Float32(d.unsafe_load(a)))
    var order = List[Int](capacity=n)
    for a in range(n):
        order.append(a)
    for a in range(1, n):
        var key = order[a]
        var b = a - 1
        while b >= 0:
            var ob = order[b]
            # the binary64 value orders; equal values keep their index order
            if not (d.unsafe_load(ob) > d.unsafe_load(key)):
                break
            order[b + 1] = ob
            b -= 1
        order[b + 1] = key
    for cc in range(n):
        var col = order[cc]
        evals.unsafe_store(cc, w32[col])
        var row = vt.unsafe_offset(col * n)
        var rr = 0
        while rr < n and Float32(row.unsafe_load(rr)) == Float32(0):
            rr += 1
        var negate = rr < n and row.unsafe_load(rr) < zero
        for r2 in range(n):
            var ev = Float32(row.unsafe_load(r2))
            evecs.unsafe_store(r2 * n + cc, -ev if negate else ev)
    _ = vt_store^
    _ = d_store^
    _ = e_store^
    return worst_iter
