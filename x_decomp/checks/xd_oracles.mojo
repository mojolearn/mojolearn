# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE DECOMP LANE'S HOST ORACLES (pass 2, lane/algos-decomp): every seam of
x_decomp/cells.mojo restated as plain host code, written apart from the cells
so a change to a cell shows up as a difference here. Each oracle takes `alt`:
0 is the pinned spelling the cells promise, and a nonzero `alt` is the
unpinned spelling the seam's DEVIATION rules out, which a check uses to show
its fixture SEPARATES the two before trusting an equality. The shared pinned
primitives (`ftz`, the fused multiply-add pin, the portable transcendentals,
Philox) are the tree's own and carry their own checks."""
from std.math import fma, sqrt

from checks.numerics import (
    ftz,
    identical_cos,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
    identical_tanh,
)
from core.philox import philox4x32_10
from decomposition.host.linalg_public import host_qr_r
from x_decomp.qr_sliced import QS_ROWS, XD_QR_SLICED


#: The fold block, restated (x_decomp/cells.mojo FOLD_BLOCK).
comptime O_BLOCK = 4096


# ------------------------------------------------------------ scalar helpers
def o_add(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


def o_sub(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) - ftz(b))


def o_mul(a: Float32, b: Float32) -> Float32:
    return ftz(identical_mul(ftz(a), ftz(b)))


def o_fma(a: Float32, b: Float32, c: Float32) -> Float32:
    return ftz(identical_mul_add(ftz(a), ftz(b), ftz(c)))


def o_div0(a: Float32, b: Float32) -> Float32:
    if ftz(b) == Float32(0):
        return Float32(0)
    return ftz(identical_div(ftz(a), ftz(b)))


def o_sqrt0(a: Float32) -> Float32:
    var x = ftz(a)
    if not (x > Float32(0)):
        return Float32(0)
    return ftz(identical_sqrt(x))


def o_logf(a: Float32, floor: Float32) -> Float32:
    var x = ftz(a)
    if not (x > ftz(floor)):
        x = ftz(floor)
    if not (x > Float32(0)):
        return Float32(0)
    return ftz(identical_log(x))


def o_exp(a: Float32) -> Float32:
    var x = ftz(a)
    if x > Float32(88.0):
        x = Float32(88.0)
    if x < Float32(-103.0):
        return Float32(0)
    return ftz(identical_exp(x))


def o_digamma(a: Float32, alt: Int = 0) -> Float32:
    """psi(x): recurrence to 6 then the asymptotic series; alt 1 skips the
    recurrence (the series alone, which is wrong below 6)."""
    var x = ftz(a)
    if not (x > Float32(0)):
        return Float32(0)
    var r = Float32(0)
    if alt == 0:
        while x < Float32(6):
            r = o_sub(r, o_div0(Float32(1), x))
            x = o_add(x, Float32(1))
    var inv = o_div0(Float32(1), x)
    var inv2 = o_mul(inv, inv)
    var t = o_mul(inv2, Float32(0.003968253968253968))
    t = o_sub(Float32(0.008333333333333333), t)
    t = o_mul(inv2, t)
    t = o_sub(Float32(0.08333333333333333), t)
    t = o_mul(inv2, t)
    r = o_add(r, o_logf(x, Float32(0)))
    r = o_sub(r, o_mul(Float32(0.5), inv))
    return o_sub(r, t)


def o_lgamma(a: Float32, alt: Int = 0) -> Float32:
    var x = ftz(a)
    if not (x > Float32(0)):
        return Float32(0)
    var shift = Float32(0)
    if alt == 0:
        while x < Float32(6):
            shift = o_add(shift, o_logf(x, Float32(0)))
            x = o_add(x, Float32(1))
    var inv = o_div0(Float32(1), x)
    var inv2 = o_mul(inv, inv)
    var t = o_mul(inv2, Float32(0.0007936507936507937))
    t = o_sub(Float32(0.002777777777777778), t)
    t = o_mul(inv2, t)
    t = o_sub(Float32(0.08333333333333333), t)
    t = o_mul(inv, t)
    var r = o_mul(o_sub(x, Float32(0.5)), o_logf(x, Float32(0)))
    r = o_sub(r, x)
    r = o_add(r, Float32(0.9189385332046727))
    r = o_add(r, t)
    return o_sub(r, shift)


# ------------------------------------------------------------ folds (5300-5302)
def oracle_gemm(
    a: List[Float32], b: List[Float32], m: Int, k: Int, n: Int, ta: Bool, tb: Bool, alt: Int = 0
) -> List[Float32]:
    """C[i, j] = sum_p A[i, p] B[p, j]: p ascending, one fused multiply-add
    per term. alt 1: p descending; alt 2: product rounded, then added."""
    # alt 3: one sequential fold even past O_BLOCK (the unblocked spelling)
    var c = List[Float32](length=m * n, fill=Float32(0))
    var blk = k if alt == 3 else O_BLOCK
    var nb = (k + blk - 1) // blk
    for i in range(m):
        for j in range(n):
            var tot = Float32(0)
            for bl in range(nb):
                var p0 = bl * blk
                var p1 = min(k, p0 + blk)
                var acc = Float32(0)
                for q in range(p1 - p0):
                    var p = p1 - 1 - q if alt == 1 else p0 + q
                    var x = a[p * m + i] if ta else a[i * k + p]
                    var y = b[j * k + p] if tb else b[p * n + j]
                    if alt == 2:
                        acc = o_add(acc, o_mul(x, y))
                    else:
                        acc = o_fma(x, y, acc)
                tot = acc if nb == 1 else o_add(tot, acc)
            c[i * n + j] = tot
    return c^


def oracle_colsum(a: List[Float32], n: Int, d: Int, alt: Int = 0) -> List[Float32]:
    """alt 1: rows descending; alt 2: the adds unflushed (a subnormal partial sum kept)."""
    var out = List[Float32](length=d, fill=Float32(0))
    var blk = n if alt == 3 else O_BLOCK
    var nb = (n + blk - 1) // blk
    for j in range(d):
        var tot = Float32(0)
        for bl in range(nb):
            var r0 = bl * blk
            var r1 = min(n, r0 + blk)
            var acc = Float32(0)
            for q in range(r1 - r0):
                var i = r1 - 1 - q if alt == 1 else r0 + q
                acc = (acc + a[i * d + j]) if alt == 2 else o_add(acc, a[i * d + j])
            tot = acc if nb == 1 else o_add(tot, acc)
        out[j] = tot
    return out^


def oracle_rowsum(a: List[Float32], n: Int, d: Int, alt: Int = 0) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    var blk = d if alt == 3 else O_BLOCK
    var nb = (d + blk - 1) // blk
    for i in range(n):
        var tot = Float32(0)
        for bl in range(nb):
            var c0 = bl * blk
            var c1 = min(d, c0 + blk)
            var acc = Float32(0)
            for q in range(c1 - c0):
                var j = c1 - 1 - q if alt == 1 else c0 + q
                acc = o_add(acc, a[i * d + j])
            tot = acc if nb == 1 else o_add(tot, acc)
        out[i] = tot
    return out^


def oracle_pdist(a: List[Float32], na: Int, b: List[Float32], nb: Int, d: Int, kind: Int, pw: Float32,
                 alt: Int = 0) -> List[Float32]:
    """DEVIATION 5319 restated: kind 1 manhattan, 2 chebyshev, 3 minkowski
    pw, 4 cosine; alt 1 folds the features descending."""
    var tiny = Float32(1.1754943508222875e-38)
    var out = List[Float32](length=na * nb, fill=Float32(0))
    for i in range(na):
        for j in range(nb):
            var r = Float32(0)
            if kind == 4:
                var ab = Float32(0)
                var aa = Float32(0)
                var bb = Float32(0)
                for q in range(d):
                    var p = d - 1 - q if alt == 1 else q
                    var x = a[i * d + p]
                    var y = b[j * d + p]
                    ab = o_fma(x, y, ab)
                    aa = o_fma(x, x, aa)
                    bb = o_fma(y, y, bb)
                var sa = o_sqrt0(aa)
                var sb = o_sqrt0(bb)
                r = o_sub(Float32(1), o_div0(ab, o_mul(sa if sa != Float32(0) else Float32(1), sb if sb != Float32(0) else Float32(1))))
                r = Float32(0) if r < Float32(0) else (Float32(2) if r > Float32(2) else r)
            else:
                for q in range(d):
                    var p = d - 1 - q if alt == 1 else q
                    var t = abs(o_sub(a[i * d + p], b[j * d + p]))
                    if kind == 1:
                        r = o_add(r, t)
                    elif kind == 2:
                        r = t if t > r else r
                    elif t > Float32(0):
                        r = o_add(r, o_exp(o_mul(pw, o_logf(t, tiny))))
                if kind == 3:
                    r = o_exp(o_div0(o_logf(r, tiny), pw)) if r > Float32(0) else Float32(0)
            out[i * nb + j] = r
    return out^


def oracle_sqdist(a: List[Float32], na: Int, b: List[Float32], nb: Int, d: Int, alt: Int = 0) -> List[Float32]:
    var out = List[Float32](length=na * nb, fill=Float32(0))
    for i in range(na):
        for j in range(nb):
            var acc = Float32(0)
            for q in range(d):
                var p = d - 1 - q if alt == 1 else q
                var t = o_sub(a[i * d + p], b[j * d + p])
                if alt == 2:
                    acc = o_add(acc, o_mul(t, t))
                else:
                    acc = o_fma(t, t, acc)
            out[i * nb + j] = acc
    return out^


def oracle_absmax_sign(a: List[Float32], n: Int, d: Int, by_col: Bool, alt: Int = 0) -> List[Float32]:
    """The sign of each column (row) by its largest-|.| entry, ties to the
    LOWER index (alt 1: the higher)."""
    var cnt = d if by_col else n
    var out = List[Float32](length=cnt, fill=Float32(1))
    for t in range(cnt):
        var best = Float32(-1)
        var val = Float32(0)
        for q in range(n if by_col else d):
            var v = ftz(a[q * d + t]) if by_col else ftz(a[t * d + q])
            if (abs(v) >= best) if alt == 1 else (abs(v) > best):
                best = abs(v)
                val = v
        out[t] = Float32(-1) if val < Float32(0) else Float32(1)
    return out^


# ------------------------------------------------------------ elementwise (5303-5305)
def oracle_ew(op: Int, x_in: Float32, y_in: Float32, z_in: Float32, s_in: Float32, alt: Int = 0) -> Float32:
    """x_decomp/cells.mojo `ew_cell` restated. alt 1: the raw quotient and
    the raw exp/log/sqrt with no zero guard or clamp (Clause B's computed
    NaN/inf); alt 2: no flush of the operands of an add."""
    var x = x_in if alt == 2 else ftz(x_in)
    var y = y_in if alt == 2 else ftz(y_in)
    var z = ftz(z_in)
    var s = ftz(s_in)
    var r = Float32(0)
    if op == 0:
        r = (x + y) if alt == 2 else o_add(x, y)
    elif op == 1:
        r = o_sub(x, y)
    elif op == 2:
        r = o_mul(x, y)
    elif op == 3:
        r = (x / y) if alt == 1 else o_div0(x, y)
    elif op == 4:
        r = ftz(identical_mul_add(s, y, x))
    elif op == 5:
        r = x if x > s else s
    elif op == 6:
        r = o_div0(o_mul(x, y), o_add(z, s))
    elif op == 7:
        r = sqrt(x) if alt == 1 else o_sqrt0(x)
    elif op == 8:
        r = o_mul(x, x)
    elif op == 9:
        r = identical_exp(x) if alt == 1 else o_exp(x)
    elif op == 10:
        r = identical_log(x) if alt == 1 else o_logf(x, s)
    elif op == 11:
        r = ftz(identical_tanh(x))
    elif op == 12:
        r = o_sub(Float32(1), o_mul(x, x))
    elif op == 13:
        r = abs(x)
    elif op == 14:
        r = o_mul(x, s)
    elif op == 15:
        r = ftz(identical_mul_add(x, y, z))
    elif op == 16:
        r = (Float32(1) / x) if alt == 1 else o_div0(Float32(1), x)
    elif op == 17:
        var m = o_sub(abs(x), s)
        r = (m if x > Float32(0) else -m) if m > Float32(0) else Float32(0)
    elif op == 18:
        r = o_mul(o_sub(x, y), z)
    elif op == 19:
        r = x if x < s else s
    elif op == 20:
        r = y
    elif op == 21:
        var t = o_sub(x, y)
        r = o_mul(t, t)
    elif op == 22:
        r = o_add(x, s)
    elif op == 23:
        r = Float32(1) if x > s else Float32(0)
    elif op == 24:
        r = o_digamma(x, 1 if alt == 3 else 0)
    elif op == 25:
        r = o_mul(x, o_exp(o_mul(Float32(-0.5), o_mul(x, x))))
    elif op == 26:
        var x2 = o_mul(x, x)
        r = o_mul(o_sub(Float32(1), x2), o_exp(o_mul(Float32(-0.5), x2)))
    elif op == 27:
        r = o_mul(o_mul(x, x), x)
    elif op == 28:
        r = o_mul(Float32(3), o_mul(x, x))
    elif op == 30:
        r = x if x > y else y
    elif op == 31:
        r = x if x < y else y
    elif op == 33:
        r = Float32(1) if x > Float32(0) else (Float32(-1) if x < Float32(0) else Float32(0))
    elif op == 34:
        r = Float32(1) if x <= y else Float32(0)
    elif op == 35:
        r = y if x > s else z
    elif op == 36:
        r = (o_mul(x, y / z)) if alt == 1 else o_mul(x, o_div0(y, z if z != Float32(0) else s))
    elif op == 37:
        r = o_lgamma(x, 1 if alt == 3 else 0)
    return r if alt == 1 else ftz(r)


# ------------------------------------------------------------ RNG (5306)
def oracle_rand(i: Int, seed: UInt32, stream: UInt32, kind: Int, alt: Int = 0) -> Float32:
    """Philox4x32-10 at counter (i, stream, i >> 32, 0), key (seed, 0x5EED).
    kind 0 uniform (r0 >> 8) * 2^-24; kind 1 Box-Muller cos arm on ((r0 >>
    8) + 1) 2^-24 and (r1 >> 8) 2^-24; kind 2 the sign of r0's low bit.
    alt 1: r0 >> 9 for the uniform, the sin arm for the normal."""
    var r = philox4x32_10(
        SIMD[DType.uint32, 4](UInt32(i & 0xFFFFFFFF), stream, UInt32(i >> 32), 0),
        SIMD[DType.uint32, 2](seed, UInt32(0x5EED)),
    )
    var g = Float32(5.9604644775390625e-08)
    if kind == 0:
        return o_mul(Float32(r[0] >> UInt32(9 if alt == 1 else 8)), g)
    if kind == 2:
        return Float32(1) if (r[0] & 1) == 1 else Float32(-1)
    var u1 = o_mul(Float32((r[0] >> 8) + 1), g)
    var u2 = o_mul(Float32(r[1] >> 8), g)
    var rad = o_sqrt0(o_mul(Float32(-2), o_logf(u1, Float32(0))))
    var ang = o_mul(Float32(6.2831854820251465), u2)
    if alt == 1:
        ang = o_add(ang, Float32(1.5707963705062866))
    return o_mul(rad, ftz(identical_cos(ang)))


def oracle_gamma(i: Int, seed: UInt32, stream: UInt32, shape: Float32, alt: Int = 0) -> Float32:
    """Marsaglia-Tsang restated; alt 1 takes the first proposal unexamined."""
    var d = o_sub(shape, Float32(0.3333333333333333))
    var c = o_div0(Float32(1), o_sqrt0(o_mul(Float32(9), d)))
    var g = Float32(5.9604644775390625e-08)
    var last = d
    for j in range(64):
        var r = philox4x32_10(
            SIMD[DType.uint32, 4](UInt32(i & 0xFFFFFFFF), stream, UInt32(j), UInt32(0x6A33)),
            SIMD[DType.uint32, 2](seed, UInt32(0x5EED)),
        )
        var u1 = o_mul(Float32((r[0] >> 8) + 1), g)
        var u2 = o_mul(Float32(r[1] >> 8), g)
        var x = o_mul(o_sqrt0(o_mul(Float32(-2), o_logf(u1, Float32(0)))), ftz(identical_cos(o_mul(Float32(6.2831854820251465), u2))))
        var v = o_add(Float32(1), o_mul(c, x))
        if not (v > Float32(0)):
            continue
        v = o_mul(o_mul(v, v), v)
        var u = o_mul(Float32((r[2] >> 8) + 1), g)
        last = o_mul(d, v)
        if alt == 1:
            return last
        var x2 = o_mul(x, x)
        if u < o_sub(Float32(1), o_mul(Float32(0.0331), o_mul(x2, x2))):
            return last
        var rhs = o_add(o_mul(Float32(0.5), x2), o_mul(d, o_add(o_sub(Float32(1), v), o_logf(v, Float32(0)))))
        if o_logf(u, Float32(0)) < rhs:
            return last
    return last


# ------------------------------------------------------------ dense factorizations (5307-5309)
def oracle_lu(a: List[Float32], n: Int, alt: Int = 0) -> Tuple[List[Float32], List[Int32]]:
    """getrf, partial pivoting on max |a[i, k]|, ties to the LOWEST row
    (strict >); alt 1: ties to the LAST row (>=)."""
    var w = a.copy()
    var piv = List[Int32](length=n, fill=Int32(0))
    for k in range(n):
        var p = k
        var best = abs(ftz(w[k * n + k]))
        for i in range(k + 1, n):
            var v = abs(ftz(w[i * n + k]))
            if (v >= best) if alt == 1 else (v > best):
                best = v
                p = i
        piv[k] = Int32(p)
        if p != k:
            for j in range(n):
                var t = w[k * n + j]
                w[k * n + j] = w[p * n + j]
                w[p * n + j] = t
        var d = ftz(w[k * n + k])
        if d == Float32(0):
            continue
        for i in range(k + 1, n):
            var l = o_div0(w[i * n + k], d)
            w[i * n + k] = l
            for j in range(k + 1, n):
                w[i * n + j] = o_fma(-l, w[k * n + j], w[i * n + j])
    return (w^, piv^)


def oracle_lu_solve(lu: List[Float32], piv: List[Int32], b: List[Float32], n: Int, nrhs: Int, alt: Int = 0) -> List[Float32]:
    """getrs; alt 1 folds each substitution's inner sum descending."""
    var x = b.copy()
    for k in range(n):
        var p = Int(piv[k])
        if p != k:
            for c in range(nrhs):
                var t = x[k * nrhs + c]
                x[k * nrhs + c] = x[p * nrhs + c]
                x[p * nrhs + c] = t
    for c in range(nrhs):
        for i in range(n):
            var acc = ftz(x[i * nrhs + c])
            for q in range(i):
                var j = i - 1 - q if alt == 1 else q
                acc = o_fma(-lu[i * n + j], x[j * nrhs + c], acc)
            x[i * nrhs + c] = acc
        for ii in range(n):
            var i = n - 1 - ii
            var acc = ftz(x[i * nrhs + c])
            for q in range(n - i - 1):
                var j = n - 1 - q if alt == 1 else i + 1 + q
                acc = o_fma(-lu[i * n + j], x[j * nrhs + c], acc)
            x[i * nrhs + c] = o_div0(acc, lu[i * n + i])
    return x^


def oracle_lu_solve_t(lu: List[Float32], piv: List[Int32], b: List[Float32], n: Int, nrhs: Int, alt: Int = 0) -> List[Float32]:
    """getrs 'T' (A^T X = B), written independently of the cell: U^T forward,
    unit L^T back, the swaps undone last to first; alt 1 folds each inner sum
    descending."""
    var x = b.copy()
    for c in range(nrhs):
        for i in range(n):
            var acc = ftz(x[i * nrhs + c])
            for q in range(i):
                var j = i - 1 - q if alt == 1 else q
                acc = o_fma(-lu[j * n + i], x[j * nrhs + c], acc)
            x[i * nrhs + c] = o_div0(acc, lu[i * n + i])
        for ii in range(n):
            var i = n - 1 - ii
            var acc = ftz(x[i * nrhs + c])
            for q in range(n - i - 1):
                var j = n - 1 - q if alt == 1 else i + 1 + q
                acc = o_fma(-lu[j * n + i], x[j * nrhs + c], acc)
            x[i * nrhs + c] = acc
    var k = n - 1
    while k >= 0:
        var p = Int(piv[k])
        if p != k:
            for c in range(nrhs):
                var t = x[k * nrhs + c]
                x[k * nrhs + c] = x[p * nrhs + c]
                x[p * nrhs + c] = t
        k -= 1
    return x^


def oracle_chol(a: List[Float32], n: Int, alt: Int = 0) -> List[Float32]:
    """Left-looking lower Cholesky, sums ascending (alt 1: descending); a
    non-positive pivot is replaced by 1 as the cell does."""
    var w = a.copy()
    for j in range(n):
        var acc = ftz(w[j * n + j])
        for q in range(j):
            var p = j - 1 - q if alt == 1 else q
            acc = o_fma(-w[j * n + p], w[j * n + p], acc)
        if not (acc > Float32(0)):
            acc = Float32(1)
        var d = o_sqrt0(acc)
        w[j * n + j] = d
        for i in range(j + 1, n):
            var s = ftz(w[i * n + j])
            for q in range(j):
                var p = j - 1 - q if alt == 1 else q
                s = o_fma(-w[i * n + p], w[j * n + p], s)
            w[i * n + j] = o_div0(s, d)
        for i in range(j + 1, n):
            w[j * n + i] = Float32(0)
    return w^


def oracle_orth(a: List[Float32], m: Int, l: Int, alt: Int = 0) raises -> List[Float32]:
    """Two passes (alt 1: one) of R = host_qr_r (decomposition/'s host twin
    of qr_factor), the rank guard (alt 2: none), Q = A R^-1 by forward
    substitution per row, ascending."""
    var q = a.copy()
    var passes = 1 if alt == 1 else 2
    for _ in range(passes):
        var r = host_qr_r(q, m, l)
        if alt != 2:
            # DEVIATION 5318: column j is dependent when its residual R[j, j]
            # is at most 2^-16 of the column's norm (both over max |R[., j]|).
            for j in range(l):
                var col = List[Float32]()
                var big = Float32(0)
                for t in range(j + 1):
                    col.append(ftz(r[t * l + j]))
                    big = max(big, abs(col[t]))
                if big > Float32(0):
                    var ss = Float32(0)
                    for t in range(j + 1):
                        var u = o_div0(col[t], big)
                        ss = o_fma(u, u, ss)
                    var dj = o_div0(col[j], big)
                    if o_mul(dj, dj) <= o_mul(Float32(2.3283064365386963e-10), ss):
                        r[j * l + j] = Float32(0)
        var nq = List[Float32](length=m * l, fill=Float32(0))
        for i in range(m):
            for j in range(l):
                var acc = ftz(q[i * l + j])
                for t in range(j):
                    acc = o_fma(-nq[i * l + t], r[t * l + j], acc)
                nq[i * l + j] = o_div0(acc, r[j * l + j])
        q = nq^
    return q^


# ------------------------------------------------------------ row solvers (5310-5312)
def oracle_cd(w: List[Float32], hht: List[Float32], xht: List[Float32], n: Int, k: Int, alt: Int = 0) -> List[Float32]:
    """One `_update_cdnmf_fast` sweep per row, components ascending (alt 1: descending)."""
    var W = w.copy()
    for i in range(n):
        for s in range(k):
            var t = k - 1 - s if alt == 1 else s
            var grad = -ftz(xht[i * k + t])
            for r in range(k):
                grad = o_fma(hht[t * k + r], W[i * k + r], grad)
            var hess = ftz(hht[t * k + t])
            if hess != Float32(0):
                var nw = o_sub(W[i * k + t], o_div0(grad, hess))
                W[i * k + t] = nw if nw > Float32(0) else Float32(0)
    return W^


def oracle_lasso(g: List[Float32], q: List[Float32], w0: List[Float32], n: Int, k: Int, alpha: Float32, sweeps: Int, alt: Int = 0) -> List[Float32]:
    """Lasso CD on the Gram, `sweeps` full sweeps, coordinates ascending
    (alt 1: descending), the residual correlation recomputed from w (the
    cell keeps it incrementally; a fixed sweep count makes the two agree
    only when they fold alike, which is the point of the oracle)."""
    var W = w0.copy()
    for i in range(n):
        var H = List[Float32](length=k, fill=Float32(0))
        for j in range(k):
            var acc = Float32(0)
            for l in range(k):
                acc = o_fma(g[j * k + l], W[i * k + l], acc)
            H[j] = acc
        for _ in range(sweeps):
            for jj in range(k):
                var j = k - 1 - jj if alt == 1 else jj
                var gjj = ftz(g[j * k + j])
                if gjj == Float32(0):
                    continue
                var wj = ftz(W[i * k + j])
                if wj != Float32(0):
                    for l in range(k):
                        H[l] = o_fma(-wj, g[l * k + j], H[l])
                var tmp = o_sub(q[i * k + j], H[j])
                var nw = Float32(0)
                var mm = o_sub(abs(tmp), alpha)
                if mm > Float32(0):
                    nw = o_div0(mm if tmp > Float32(0) else -mm, gjj)
                W[i * k + j] = nw
                if nw != Float32(0):
                    for l in range(k):
                        H[l] = o_fma(nw, g[l * k + j], H[l])
    return W^


def oracle_omp_first(g: List[Float32], q: List[Float32], n: Int, k: Int, alt: Int = 0) -> List[Int32]:
    """OMP's first atom per row: argmax |q_j|, ties to the LOWER index
    (alt 1: the higher)."""
    var out = List[Int32](length=n, fill=Int32(0))
    for i in range(n):
        var lam = 0
        var best = Float32(-1)
        for j in range(k):
            var v = abs(ftz(q[i * k + j]))
            if (v >= best) if alt == 1 else (v > best):
                best = v
                lam = j
        out[i] = Int32(lam)
    return out^


# ------------------------------------------------------------ LDA (5313)
def oracle_lda_step(x: List[Float32], ew: List[Float32], d0: List[Float32], e0: List[Float32], n: Int, k: Int, v: Int, prior: Float32, alt: Int = 0) -> List[Float32]:
    """ONE `_update_doc_distribution` iteration per document (words ascending
    in both folds; alt 1: descending in the topic fold). Returns the new
    doc-topic rows."""
    var out = List[Float32](length=n * k, fill=Float32(0))
    var eps = Float32(2.220446049250313e-16)
    for i in range(n):
        var r = List[Float32](length=v, fill=Float32(0))
        for w in range(v):
            if ftz(x[i * v + w]) == Float32(0):
                continue
            var acc = Float32(0)
            for t in range(k):
                acc = o_fma(e0[i * k + t], ew[t * v + w], acc)
            r[w] = o_div0(x[i * v + w], o_add(acc, eps))
        for t in range(k):
            var acc = Float32(0)
            for ww in range(v):
                var w = v - 1 - ww if alt == 1 else ww
                if ftz(x[i * v + w]) == Float32(0):
                    continue
                acc = o_fma(r[w], ew[t * v + w], acc)
            out[i * k + t] = o_add(o_mul(e0[i * k + t], acc), prior)
    return out^


# ------------------------------------------------------------ graph (5314, 5315)
def oracle_dijkstra(w: List[Float32], n: Int, alt: Int = 0) -> List[Float32]:
    """All-pairs shortest paths, undirected: edge u-v weighs the smaller
    nonzero of W[u, v] and W[v, u] (alt 1: W[u, v] alone, directed); -1 for
    an unreachable pair. Plain Bellman-Ford relaxation to a fixed point:
    a different algorithm reaching the same minimum."""
    var out = List[Float32](length=n * n, fill=Float32(-1))
    for s in range(n):
        out[s * n + s] = Float32(0)
        var changed = True
        while changed:
            changed = False
            for u in range(n):
                var du = out[s * n + u]
                if du < Float32(0):
                    continue
                for v2 in range(n):
                    var a = ftz(w[u * n + v2])
                    var b = ftz(w[v2 * n + u])
                    var e = a
                    if alt == 0 and (e == Float32(0) or (b != Float32(0) and b < e)):
                        e = b
                    if e == Float32(0):
                        continue
                    var nd = o_add(du, e)
                    var dv = out[s * n + v2]
                    if dv < Float32(0) or nd < dv:
                        out[s * n + v2] = nd
                        changed = True
    return out^


def oracle_barycenter(x: List[Float32], y: List[Float32], nbr: List[Int], n: Int, d: Int, k: Int, reg: Float32, alt: Int = 0) -> List[Float32]:
    """sklearn barycenter_weights: G = Z Z^T + R I with R = reg * trace(G)
    (alt 1: R = reg), w = G^-1 1 by Cholesky, normalized to sum 1."""
    var out = List[Float32](length=n * k, fill=Float32(0))
    for i in range(n):
        var z = List[Float32](length=k * d, fill=Float32(0))
        for a in range(k):
            for f in range(d):
                z[a * d + f] = o_sub(y[nbr[i * k + a] * d + f], x[i * d + f])
        var g = List[Float32](length=k * k, fill=Float32(0))
        var tr = Float32(0)
        for a in range(k):
            for b in range(k):
                var acc = Float32(0)
                for f in range(d):
                    acc = o_fma(z[a * d + f], z[b * d + f], acc)
                g[a * k + b] = acc
            tr = o_add(tr, g[a * k + a])
        var R = reg if (alt == 1 or not (tr > Float32(0))) else o_mul(reg, tr)
        for a in range(k):
            g[a * k + a] = o_add(g[a * k + a], R)
        var L = oracle_chol(g, k)
        var w = List[Float32](length=k, fill=Float32(0))
        for a in range(k):
            var acc = Float32(1)
            for p in range(a):
                acc = o_fma(-L[a * k + p], w[p], acc)
            w[a] = o_div0(acc, L[a * k + a])
        for aa in range(k):
            var a = k - 1 - aa
            var acc = ftz(w[a])
            for p in range(a + 1, k):
                acc = o_fma(-L[p * k + a], w[p], acc)
            w[a] = o_div0(acc, L[a * k + a])
        var tot = Float32(0)
        for a in range(k):
            tot = o_add(tot, w[a])
        for a in range(k):
            out[i * k + a] = o_div0(w[a], tot)
    return out^


# ------------------------------------------------------------ ALS (5316)
def oracle_als(c: List[Float32], y: List[Float32], n: Int, m: Int, f: Int, reg: Float32, alt: Int = 0) -> List[Float32]:
    """implicit's exact least_squares per row: YtY (items ascending), + reg
    I, + (c - 1) y y^T and b += c y over the items ascending (alt 1:
    descending), then the Cholesky solve."""
    var yty = List[Float32](length=f * f, fill=Float32(0))
    for a in range(f):
        for b in range(f):
            var acc = Float32(0)
            for i in range(m):
                acc = o_fma(y[i * f + a], y[i * f + b], acc)
            yty[a * f + b] = acc
    var out = List[Float32](length=n * f, fill=Float32(0))
    for u in range(n):
        var A = yty.copy()
        var bv = List[Float32](length=f, fill=Float32(0))
        for j in range(f):
            A[j * f + j] = o_add(A[j * f + j], reg)
        for ii in range(m):
            var i = m - 1 - ii if alt == 1 else ii
            var conf = ftz(c[u * m + i])
            if conf == Float32(0):
                continue
            if conf > Float32(0):
                for j in range(f):
                    bv[j] = o_fma(conf, y[i * f + j], bv[j])
            else:
                conf = -conf
            var cm1 = o_sub(conf, Float32(1))
            for j in range(f):
                var t = o_mul(cm1, y[i * f + j])
                for l in range(f):
                    A[j * f + l] = o_fma(t, y[i * f + l], A[j * f + l])
        var L = oracle_chol(A, f)
        for a in range(f):
            var acc = ftz(bv[a])
            for p in range(a):
                acc = o_fma(-L[a * f + p], bv[p], acc)
            bv[a] = o_div0(acc, L[a * f + a])
        for aa in range(f):
            var a = f - 1 - aa
            var acc = ftz(bv[a])
            for p in range(a + 1, f):
                acc = o_fma(-L[p * f + a], bv[p], acc)
            bv[a] = o_div0(acc, L[a * f + a])
        for q in range(f):
            out[u * f + q] = bv[q]
    return out^


def _o_tree(p_in: List[Float32]) -> Float32:
    """x_decomp/qr_sliced.mojo's tree restated: stride h = 1, 2, 4, ..., p[a]
    absorbs p[a + h] for a = 0, 2h, ...; 0 for no partials."""
    var p = p_in.copy()
    var ns = len(p)
    if ns == 0:
        return Float32(0)
    var h = 1
    while h < ns:
        var a = 0
        while a + h < ns:
            p[a] = o_add(p[a], p[a + h])
            a += 2 * h
        h *= 2
    return p[0]


def _o_pair(s1: Float32, q1: Float32, s2: Float32, q2: Float32) -> Tuple[Float32, Float32]:
    """dlassq's combine: the larger scale kept (a tie keeps the left)."""
    var bs = s1
    var bq = q1
    var ss = s2
    var sq = q2
    if s2 > s1:
        bs = s2
        bq = q2
        ss = s1
        sq = q1
    if ss == Float32(0):
        return (bs, bq)
    var r = o_div0(ss, bs)
    return (bs, o_fma(o_mul(r, r), sq, bq))


def _o_pair_tree(s_in: List[Float32], q_in: List[Float32]) -> Tuple[Float32, Float32]:
    var s = s_in.copy()
    var q = q_in.copy()
    var ns = len(s)
    if ns == 0:
        return (Float32(0), Float32(0))
    var h = 1
    while h < ns:
        var a = 0
        while a + h < ns:
            var r = _o_pair(s[a], q[a], s[a + h], q[a + h])
            s[a] = r[0]
            q[a] = r[1]
            a += 2 * h
        h *= 2
    return (s[0], q[0])


def _o_slice_rows(k: Int, c: Int, m: Int, alt: Int) -> List[Int]:
    """Slice c's rows at step k, ascending (alt 1: descending)."""
    var lo = k + 1 + c * QS_ROWS
    var hi = min(k + 1 + (c + 1) * QS_ROWS, m)
    var out = List[Int]()
    for t in range(hi - lo):
        out.append((hi - 1 - t) if alt == 1 else (lo + t))
    return out^


def _o_sliced_dot(xv: List[Float32], xs: Int, xc: Int, yv: List[Float32], ys: Int, yc: Int, k: Int, m: Int, alt: Int) -> Float32:
    """w = y[k] + sum_{i>k} x[i] y[i] in the sliced order: each slice's chain
    from 0, the tree, then the seed added."""
    var ns = (m - k - 1 + QS_ROWS - 1) // QS_ROWS if m - k - 1 > 0 else 0
    var parts = List[Float32]()
    for c in range(ns):
        var acc = Float32(0)
        for i in _o_slice_rows(k, c, m, alt):
            acc = o_fma(xv[i * xs + xc], yv[i * ys + yc], acc)
        parts.append(acc)
    return o_add(yv[k * ys + yc], _o_tree(parts))


def oracle_geqrf_sliced(a_in: List[Float32], m: Int, n: Int, alt: Int = 0) -> Tuple[List[Float32], List[Float32]]:
    """geqrf in x_decomp/qr_sliced.mojo's order, restated: the norm from the
    slices' (scale, sum of squares) pairs by the tree, alpha's pair last;
    every reflector product by slices, the tree, the seed added last (alt 1:
    every slice's chain descending)."""
    var a = a_in.copy()
    var kk = m if m < n else n
    var tau = List[Float32](length=kk, fill=Float32(0))
    for k in range(kk):
        var alpha = ftz(a[k * n + k])
        var ns = (m - k - 1 + QS_ROWS - 1) // QS_ROWS if m - k - 1 > 0 else 0
        var ps = List[Float32]()
        var pq = List[Float32]()
        for c in range(ns):
            var rows = _o_slice_rows(k, c, m, alt)
            var mx = Float32(0)
            for i in rows:
                if abs(ftz(a[i * n + k])) > mx:
                    mx = abs(ftz(a[i * n + k]))
            var acc = Float32(0)
            if mx != Float32(0):
                for i in rows:
                    var v = o_div0(a[i * n + k], mx)
                    acc = o_fma(v, v, acc)
            ps.append(mx)
            pq.append(acc)
        var t = _o_pair_tree(ps, pq)
        if t[0] == Float32(0):
            continue
        var aa = abs(alpha)
        var f = _o_pair(t[0], t[1], aa, Float32(1) if aa != Float32(0) else Float32(0))
        var nrm = o_mul(o_sqrt0(f[1]), f[0])
        var beta = -nrm if alpha >= Float32(0) else nrm
        tau[k] = o_div0(o_sub(beta, alpha), beta)
        var scale = o_sub(alpha, beta)
        for i in range(k + 1, m):
            a[i * n + k] = o_div0(a[i * n + k], scale)
        a[k * n + k] = beta
        for j in range(k + 1, n):
            var w = _o_sliced_dot(a, n, k, a, n, j, k, m, alt)
            var tw = o_mul(tau[k], w)
            a[k * n + j] = o_sub(a[k * n + j], tw)
            for i in range(k + 1, m):
                a[i * n + j] = o_fma(-tw, a[i * n + k], a[i * n + j])
    return (a^, tau^)


def oracle_orgqr_sliced(h: List[Float32], tau: List[Float32], m: Int, n: Int, kk: Int, qc: Int, alt: Int = 0) -> List[Float32]:
    """orgqr in the sliced order, restated (alt 1: slice chains descending;
    alt 2: H_0 first)."""
    var q = List[Float32](length=m * qc, fill=Float32(0))
    for j in range(qc):
        var x = List[Float32](length=m, fill=Float32(0))
        x[j] = Float32(1)
        for r in range(kk):
            var k = r if alt == 2 else kk - 1 - r
            var t = ftz(tau[k])
            if t == Float32(0):
                continue
            var w = _o_sliced_dot(h, n, k, x, 1, 0, k, m, alt)
            var tw = o_mul(t, w)
            x[k] = o_sub(x[k], tw)
            for i in range(k + 1, m):
                x[i] = o_fma(-tw, h[i * n + k], x[i])
        for i in range(m):
            q[i * qc + j] = x[i]
    return q^


def oracle_geqrf(a_in: List[Float32], m: Int, n: Int, alt: Int = 0) -> Tuple[List[Float32], List[Float32]]:
    """LAPACK geqrf restated (DEVIATION 5320): per column k, dlarfg on
    (a[k, k], a[k+1:, k]) with the norm scaled by the largest |entry|, its
    squares ascending (alt 1: descending), beta = -sign(alpha) ||.||, then
    H_k applied to every later column with w = a[k, j] + sum_{i>k} v_i
    a[i, j] ascending (alt 1: descending). Returns (h, tau). The sliced
    order (lane hr-qr) unless -D MOJOLEARN_XD_QR_CHAIN."""
    comptime if XD_QR_SLICED:
        return oracle_geqrf_sliced(a_in, m, n, alt)
    var a = a_in.copy()
    var kk = m if m < n else n
    var tau = List[Float32](length=kk, fill=Float32(0))
    for k in range(kk):
        var alpha = ftz(a[k * n + k])
        var xmax = Float32(0)
        for i in range(k + 1, m):
            if abs(ftz(a[i * n + k])) > xmax:
                xmax = abs(ftz(a[i * n + k]))
        if xmax == Float32(0):
            continue
        var mx = Float32(0)
        for i in range(k, m):
            if abs(ftz(a[i * n + k])) > mx:
                mx = abs(ftz(a[i * n + k]))
        var acc = Float32(0)
        for ii in range(m - k):
            var i = (m - 1 - ii) if alt == 1 else (k + ii)
            var v = ftz(identical_div(ftz(a[i * n + k]), mx))
            acc = o_fma(v, v, acc)
        var nrm = ftz(identical_mul(o_sqrt0(acc), mx))
        var beta = -nrm if alpha >= Float32(0) else nrm
        tau[k] = o_div0(o_sub(beta, alpha), beta)
        var scale = o_sub(alpha, beta)
        for i in range(k + 1, m):
            a[i * n + k] = o_div0(a[i * n + k], scale)
        a[k * n + k] = beta
        for j in range(k + 1, n):
            var w = ftz(a[k * n + j])
            for ii in range(m - k - 1):
                var i = (m - 1 - ii) if alt == 1 else (k + 1 + ii)
                w = o_fma(a[i * n + k], a[i * n + j], w)
            var tw = o_mul(tau[k], w)
            a[k * n + j] = o_sub(a[k * n + j], tw)
            for i in range(k + 1, m):
                a[i * n + j] = o_fma(-tw, a[i * n + k], a[i * n + j])
    return (a^, tau^)


def oracle_orgqr(h: List[Float32], tau: List[Float32], m: Int, n: Int, kk: Int, qc: Int, alt: Int = 0) -> List[Float32]:
    """orgqr restated (DEVIATION 5320): column j of Q is e_j with H_{kk-1},
    ..., H_0 applied in that order (alt 2: H_0 first), each w = x[k] + sum
    v_i x[i] ascending (alt 1: descending)."""
    comptime if XD_QR_SLICED:
        return oracle_orgqr_sliced(h, tau, m, n, kk, qc, alt)
    var q = List[Float32](length=m * qc, fill=Float32(0))
    for j in range(qc):
        var x = List[Float32](length=m, fill=Float32(0))
        x[j] = Float32(1)
        for r in range(kk):
            var k = r if alt == 2 else kk - 1 - r
            var t = ftz(tau[k])
            if t == Float32(0):
                continue
            var w = ftz(x[k])
            for ii in range(m - k - 1):
                var i = (m - 1 - ii) if alt == 1 else (k + 1 + ii)
                w = o_fma(h[i * n + k], x[i], w)
            var tw = o_mul(t, w)
            x[k] = o_sub(x[k], tw)
            for i in range(k + 1, m):
                x[i] = o_fma(-tw, h[i * n + k], x[i])
        for i in range(m):
            q[i * qc + j] = x[i]
    return q^


def _o_ytyx(yty: List[Float32], reg: Float32, v: List[Float32], f: Int, sign: Float32) -> List[Float32]:
    var res = List[Float32](length=f, fill=Float32(0))
    for j in range(f):
        var acc = Float32(0)
        for l in range(f):
            var e = ftz(yty[j * f + l])
            if l == j:
                e = o_add(e, reg)
            acc = o_fma(e, v[l], acc)
        res[j] = o_mul(sign, acc)
    return res^


def _o_ydot(y: List[Float32], i: Int, v: List[Float32], f: Int) -> Float32:
    var acc = Float32(0)
    for j in range(f):
        acc = o_fma(y[i * f + j], v[j], acc)
    return acc


def _o_vdot(a: List[Float32], b: List[Float32], rev: Bool) -> Float32:
    var acc = Float32(0)
    for jj in range(len(a)):
        var j = len(a) - 1 - jj if rev else jj
        acc = o_fma(a[j], b[j], acc)
    return acc


def oracle_als_cg(c: List[Float32], y: List[Float32], x0: List[Float32], n: Int, m: Int, f: Int, reg: Float32,
                  steps: Int, alt: Int = 0) -> List[Float32]:
    """implicit's `_least_squares_cg` restated (DEVIATION 5321): YtY items
    ascending, r = b - (YtY + reg I) x over the items ascending, then
    `steps` CG steps with every dot ascending in the factor index (alt 1:
    the p.Ap dot descending)."""
    var yty = List[Float32](length=f * f, fill=Float32(0))
    for a in range(f):
        for b in range(f):
            var acc = Float32(0)
            for i in range(m):
                acc = o_fma(y[i * f + a], y[i * f + b], acc)
            yty[a * f + b] = acc
    var out = x0.copy()
    for u in range(n):
        var x = List[Float32](length=f, fill=Float32(0))
        for j in range(f):
            x[j] = out[u * f + j]
        var r = _o_ytyx(yty, reg, x, f, Float32(-1))
        for i in range(m):
            var conf = ftz(c[u * m + i])
            if conf == Float32(0):
                continue
            var temp = Float32(0)
            if conf > Float32(0):
                temp = conf
            else:
                conf = -conf
            temp = o_sub(temp, o_mul(o_sub(conf, Float32(1)), _o_ydot(y, i, x, f)))
            for j in range(f):
                r[j] = o_fma(temp, y[i * f + j], r[j])
        var p = r.copy()
        var rsold = _o_vdot(r, r, False)
        if rsold < Float32(1e-20):
            continue
        for _ in range(steps):
            var ap = _o_ytyx(yty, reg, p, f, Float32(1))
            for i in range(m):
                var conf = ftz(c[u * m + i])
                if conf == Float32(0):
                    continue
                if conf < Float32(0):
                    conf = -conf
                var temp = o_mul(o_sub(conf, Float32(1)), _o_ydot(y, i, p, f))
                for j in range(f):
                    ap[j] = o_fma(temp, y[i * f + j], ap[j])
            var alpha = o_div0(rsold, _o_vdot(p, ap, alt == 1))
            for j in range(f):
                x[j] = o_fma(alpha, p[j], x[j])
            for j in range(f):
                r[j] = o_fma(-alpha, ap[j], r[j])
            var rsnew = _o_vdot(r, r, False)
            if rsnew < Float32(1e-20):
                break
            var beta = o_div0(rsnew, rsold)
            for j in range(f):
                p[j] = o_fma(beta, p[j], r[j])
            rsold = rsnew
        for j in range(f):
            out[u * f + j] = x[j]
    return out^
