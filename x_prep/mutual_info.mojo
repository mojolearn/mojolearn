# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""mutual_info_classif / mutual_info_regression units, in the prep lane's
program model (x_prep/common.mojo).

Reference: scikit-learn 1.9 `sklearn/feature_selection/_mutual_info.py`
(`_estimate_mi`: scale without centring, add 1e-10 * max(1, mean|x|) *
N(0, 1) noise; `_compute_mi_cc`: the Kraskov estimator, Chebyshev k-th
neighbour radius, marginal counts within nextafter(radius, 0);
`_compute_mi_cd`: Ross's estimator). Every neighbour search is a brute
force scan in ascending index order, one unit per (point, feature), so the
k-th distance and the counts are exact functions of the input words. The
noise is ours: splitmix64 of (seed, element) through Box-Muller with the
portable log / cos, where the reference draws numpy's generator.
`count within nextafter(r, 0)` is `dist < r` for r > 0 and `dist == 0` for
r == 0, the same set in float32.
"""
from std.memory import bitcast
from checks.numerics import identical_cos
from x_prep.common import FP, IP, p, ld, st, ldi, sti
from x_prep.prims import add, sub, mul, div, logf, sqrtf, zero_to_one

comptime MAX_K = 32
comptime TWO_PI = Float32(6.2831855)


def digammaf(x_in: Float32) -> Float32:
    """psi(x) for x > 0: the recurrence up to x >= 6, then the asymptotic
    series."""
    var x = x_in
    var acc = Float32(0)
    while x < Float32(6):
        acc = sub(acc, div(Float32(1), x))
        x = add(x, Float32(1))
    var inv = div(Float32(1), x)
    var inv2 = mul(inv, inv)
    var tail = mul(inv2, sub(Float32(0.083333333), mul(inv2, sub(Float32(0.0083333333), mul(inv2, Float32(0.0039682540))))))
    return add(acc, sub(sub(logf(x), mul(Float32(0.5), inv)), tail))


@always_inline
def _splitmix(v: UInt64) -> UInt64:
    var z = v + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def gauss(seed: Int, t: Int) -> Float32:
    """A standard normal from counter t: two splitmix64 words, their top 24
    bits as uniforms, Box-Muller (DEVIATION 5406)."""
    var base = UInt64(seed) * UInt64(0x100000000) + UInt64(2 * t)
    var z1 = _splitmix(base)
    var z2 = _splitmix(base + UInt64(1))
    var u1 = mul(Float32(Int((z1 >> 40) + UInt64(1))), Float32(5.9604645e-08))
    var u2 = mul(Float32(Int(z2 >> 40)), Float32(5.9604645e-08))
    var r = sqrtf(mul(Float32(-2), logf(u1)))
    return mul(r, identical_cos(mul(TWO_PI, u2)))


def mi_colscale_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, ST, SCALE, MABS]; t = column. SCALE = population std
    (zero -> one); MABS = max(1, mean |x / SCALE|)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var s = zero_to_one(sqrtf(ld(f, p(q, 3) + 2 * d + t)))
    st(f, p(q, 4) + t, s)
    var acc = Float32(0)
    for i in range(n):
        acc = add(acc, abs(div(ld(f, p(q, 0) + i * d + t), s)))
    var m = div(acc, Float32(n))
    st(f, p(q, 5) + t, m if m > Float32(1) else Float32(1))


def mi_noise_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, SCALE, MABS, SEED, OUT, SEC]; t = element. SEC == 0:
    OUT = x / SCALE + 1e-10 * MABS * N(0, 1) in float32. SEC > 0: the
    reference's noise kept exactly as a second word (DEVIATION 5407):
    OUT = x / SCALE and SEC - 1 holds MABS * N(0, 1), the value being
    OUT + 1e-10 * SEC word, which float32 cannot add without losing it."""
    var c = t % p(q, 2)
    var v = div(ld(f, p(q, 0) + t), ld(f, p(q, 3) + c))
    var g = gauss(p(q, 5), t)
    if p(q, 7) > 0:
        st(f, p(q, 6) + t, v)
        st(f, p(q, 7) - 1 + t, mul(ld(f, p(q, 4) + c), g))
        return
    st(f, p(q, 6) + t, add(v, mul(mul(Float32(1.0e-10), ld(f, p(q, 4) + c)), g)))


@always_inline
def _sec(f: FP, base1: Int, i: Int) -> Float32:
    """The secondary word of element i (base1 = offset + 1; 0: none, 0)."""
    if base1 == 0:
        return Float32(0)
    return ld(f, base1 - 1 + i)


@always_inline
def _dsec(zi: Float32, zj: Float32, si: Float32, sj: Float32) -> Float32:
    """The secondary word of |(zj + e*sj) - (zi + e*si)| for an infinitesimal
    e: e*(sj - si) carries the sign of zj - zi, or is taken absolute when the
    primary words tie."""
    var dz = sub(zj, zi)
    var ds = sub(sj, si)
    if dz > Float32(0):
        return ds
    if dz < Float32(0):
        return -ds
    return abs(ds)


@always_inline
def _less(ap: Float32, as_: Float32, bp: Float32, bs: Float32) -> Bool:
    """(ap, as_) < (bp, bs): the primary word first, the secondary on a tie."""
    return ap < bp or (ap == bp and as_ < bs)


@always_inline
def _within(dp: Float32, ds: Float32, rp: Float32, rs: Float32) -> Bool:
    """DEVIATION 5407: within nextafter(r, 0) is strictly inside r, compared
    as (primary, secondary) pairs (DEVIATION 5407); at r == 0 exactly, the
    reference's radius-0 query, only distance 0 counts. With no secondary
    words (all 0) this is dist < r, and dist == 0 at r == 0."""
    if dp != rp:
        return dp < rp
    if rp == Float32(0) and rs == Float32(0):
        return ds == Float32(0)
    return ds < rs


def mi_cc_unit(t: Int, f: FP, q: IP):
    """q = [Z, n, d, Y, k, TERM, ZS, YS]; t = i*d + c. TERM = psi(nx) +
    psi(ny), the marginal counts (self included) within the k-th Chebyshev
    neighbour radius of the joint (x, y) sample. ZS / YS (offset + 1, or 0
    for none) hold the noise words (DEVIATION 5407): every distance is a
    (primary, secondary) pair, compared lexicographically."""
    var n = p(q, 1)
    var d = p(q, 2)
    var k = p(q, 4)
    var i = t // d
    var c = t % d
    var zs = p(q, 6)
    var ys = p(q, 7)
    var xi = ld(f, p(q, 0) + i * d + c)
    var yi = ld(f, p(q, 3) + i)
    var sxi = _sec(f, zs, i * d + c)
    var syi = _sec(f, ys, i)
    var bp = InlineArray[Float32, MAX_K](fill=Float32(3.4028235e38))
    var bs = InlineArray[Float32, MAX_K](fill=Float32(3.4028235e38))
    for j in range(n):
        if j == i:
            continue
        var xj = ld(f, p(q, 0) + j * d + c)
        var yj = ld(f, p(q, 3) + j)
        var dx = abs(sub(xj, xi))
        var dy = abs(sub(yj, yi))
        var sx = _dsec(xi, xj, sxi, _sec(f, zs, j * d + c))
        var sy = _dsec(yi, yj, syi, _sec(f, ys, j))
        var dp = dx
        var dsec = sx
        if _less(dx, sx, dy, sy):
            dp = dy
            dsec = sy
        if _less(dp, dsec, bp[k - 1], bs[k - 1]):
            var m = k - 1
            while m > 0 and _less(dp, dsec, bp[m - 1], bs[m - 1]):
                bp[m] = bp[m - 1]
                bs[m] = bs[m - 1]
                m -= 1
            bp[m] = dp
            bs[m] = dsec
    var rp = bp[k - 1]
    var rs = bs[k - 1]
    var nx = 0
    var ny = 0
    for j in range(n):
        var xj = ld(f, p(q, 0) + j * d + c)
        var yj = ld(f, p(q, 3) + j)
        if _within(abs(sub(xj, xi)), _dsec(xi, xj, sxi, _sec(f, zs, j * d + c)), rp, rs):
            nx += 1
        if _within(abs(sub(yj, yi)), _dsec(yi, yj, syi, _sec(f, ys, j)), rp, rs):
            ny += 1
    st(f, p(q, 5) + t, add(digammaf(Float32(nx)), digammaf(Float32(ny))))


def _cd_term(
    f: FP, i: Int, n: Int, zb: Int, zs: Int, sb: Int, lb: Int, ls: Int, cb: Int, k: Int
) -> Float32:
    """Ross's per-point term for point i: the continuous values at
    Z[zb + j*zs] (their noise words at sb - 1 + j*zs, sb = 0 for none;
    DEVIATION 5407), the labels at L[lb + j*ls], each label's count at
    CNT[cb + label]. For a point whose class has more than one member:
    kl = min(k, count - 1), r the kl-th nearest same-class distance, m the
    points (of classes with more than one member, self included) within r;
    psi(kl) - psi(count) - psi(m). Otherwise 0 (outside the mean)."""
    var li = Int(ld(f, lb + i * ls))
    var cnt = Int(ld(f, cb + li))
    if cnt <= 1:
        return Float32(0)
    var kl = k if k < cnt - 1 else cnt - 1
    var xi = ld(f, zb + i * zs)
    var si = _sec(f, sb, i * zs)
    var bp = InlineArray[Float32, MAX_K](fill=Float32(3.4028235e38))
    var bs = InlineArray[Float32, MAX_K](fill=Float32(3.4028235e38))
    for j in range(n):
        if j == i or Int(ld(f, lb + j * ls)) != li:
            continue
        var xj = ld(f, zb + j * zs)
        var dp = abs(sub(xj, xi))
        var dsec = _dsec(xi, xj, si, _sec(f, sb, j * zs))
        if _less(dp, dsec, bp[kl - 1], bs[kl - 1]):
            var m = kl - 1
            while m > 0 and _less(dp, dsec, bp[m - 1], bs[m - 1]):
                bp[m] = bp[m - 1]
                bs[m] = bs[m - 1]
                m -= 1
            bp[m] = dp
            bs[m] = dsec
    var rp = bp[kl - 1]
    var rs = bs[kl - 1]
    var mall = 0
    for j in range(n):
        if Int(ld(f, cb + Int(ld(f, lb + j * ls)))) <= 1:
            continue
        var xj = ld(f, zb + j * zs)
        if _within(abs(sub(xj, xi)), _dsec(xi, xj, si, _sec(f, sb, j * zs)), rp, rs):
            mall += 1
    return sub(sub(digammaf(Float32(kl)), digammaf(Float32(cnt))), digammaf(Float32(mall)))


def mi_cd_unit(t: Int, f: FP, q: IP):
    """q = [Z, n, d, Y, LABCNT, k, TERM, ZS]; t = i*d + c. A continuous
    feature against the classes Y (counts LABCNT): Ross's term of point i
    (`_cd_term`); ZS the noise words (offset + 1, or 0 for none)."""
    var d = p(q, 2)
    var i = t // d
    var c = t % d
    var sb = p(q, 7) + c if p(q, 7) > 0 else 0
    st(f, p(q, 6) + t, _cd_term(f, i, p(q, 1), p(q, 0) + c, d, sb, p(q, 3), 1, p(q, 4), p(q, 5)))


def mi_dc_unit(t: Int, f: FP, q: IP):
    """q = [ZY, n, d, XC, CNT, KS, k, TERM, ZYS]; t = i*d + c. A discrete feature
    against a continuous target (the reference's `_compute_mi_cd(y, x)`):
    the labels are column c's codes XC[i*d + c], their counts CNT[c*KS + code],
    the continuous values the noised target ZY (noise words ZYS, offset + 1,
    or 0 for none)."""
    var d = p(q, 2)
    var i = t // d
    var c = t % d
    st(f, p(q, 7) + t, _cd_term(f, i, p(q, 1), p(q, 0), 1, p(q, 8), p(q, 3) + c, d, p(q, 4) + c * p(q, 5), p(q, 6)))


def mi_dd_unit(t: Int, f: FP, q: IP):
    """q = [XC, n, d, YC, ky, KX, TB, S, OUT]; t = column c. The contingency
    mutual information of column c's codes XC[i*d + c] (KX[c] categories)
    against the class codes YC (ky classes), the reference's
    `mutual_info_score`: the table T (int32, at TB + c*S, KX[c]*ky entries,
    then its row sums and column sums; the arena arrives zeroed), then over
    the nonzero cells in row-major order
    c_ab / N * (log c_ab - log N) + c_ab / N * (log N + log N - log a - log b),
    a term below float32 eps in magnitude read as 0 (the reference: its
    dtype's eps), the sum clipped at 0; one category or one class is 0."""
    var n = p(q, 1)
    var d = p(q, 2)
    var ky = p(q, 4)
    var kx = Int(ld(f, p(q, 5) + t))
    var tb = p(q, 6) + t * p(q, 7)
    var rb = tb + kx * ky
    var cb = rb + kx
    for i in range(n):
        var a = Int(ld(f, p(q, 0) + i * d + t))
        var b = Int(ld(f, p(q, 3) + i))
        sti(f, tb + a * ky + b, ldi(f, tb + a * ky + b) + 1)
        sti(f, rb + a, ldi(f, rb + a) + 1)
        sti(f, cb + b, ldi(f, cb + b) + 1)
    if kx <= 1 or ky <= 1:
        st(f, p(q, 8) + t, Float32(0))
        return
    var nf = Float32(n)
    var logn = logf(nf)
    var s = Float32(0)
    for a in range(kx):
        for b in range(ky):
            var cab = ldi(f, tb + a * ky + b)
            if cab == 0:
                continue
            var cf = Float32(cab)
            var cnm = div(cf, nf)
            var outer = add(logf(Float32(ldi(f, rb + a))), logf(Float32(ldi(f, cb + b))))
            var term = add(mul(cnm, sub(logf(cf), logn)), mul(cnm, sub(add(logn, logn), outer)))
            if abs(term) < Float32(1.1920929e-07):
                term = Float32(0)
            s = add(s, term)
    st(f, p(q, 8) + t, s if s > Float32(0) else Float32(0))


def mi_reduce_unit(t: Int, f: FP, q: IP):
    """q = [TERM, n, d, KIND, k, NUSED, OUT, CNT, KS]; t = column. KIND 0
    (cc): psi(n) + psi(k) - mean(TERM); KIND 1 (cd): psi(NUSED) +
    sum(TERM) / NUSED; KIND 2 (cd per column): KIND 1 with NUSED the sum of
    column t's counts CNT[t*KS : (t+1)*KS] that exceed 1; KIND 3 (cd, lane
    cpu2-l3-prep): KIND 2 over the one table CNT[0 : KS] (the class counts)
    for every column, in place of the host's sum. Negative estimates are
    0."""
    var n = p(q, 1)
    var d = p(q, 2)
    var s = Float32(0)
    for i in range(n):
        s = add(s, ld(f, p(q, 0) + i * d + t))
    var mi: Float32
    if p(q, 3) == 0:
        mi = sub(add(digammaf(Float32(n)), digammaf(Float32(p(q, 4)))), div(s, Float32(n)))
    else:
        var used = p(q, 5)
        if p(q, 3) >= 2:
            # KIND 3 (lane cpu2-l3-prep): one count table shared by every column
            var cb = p(q, 7) + (0 if p(q, 3) == 3 else t * p(q, 8))
            used = 0
            for k in range(p(q, 8)):
                var cnt = Int(ld(f, cb + k))
                if cnt > 1:
                    used += cnt
        mi = add(digammaf(Float32(used)), div(s, Float32(used))) if used > 0 else Float32(0)
    st(f, p(q, 6) + t, mi if mi > Float32(0) else Float32(0))
