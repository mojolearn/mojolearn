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
from x_prep.common import FP, IP, p, ld, st
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
    """q = [X, n, d, SCALE, MABS, SEED, OUT]; t = element:
    x / SCALE + 1e-10 * MABS * N(0, 1)."""
    var c = t % p(q, 2)
    var v = div(ld(f, p(q, 0) + t), ld(f, p(q, 3) + c))
    st(f, p(q, 6) + t, add(v, mul(mul(Float32(1.0e-10), ld(f, p(q, 4) + c)), gauss(p(q, 5), t))))


@always_inline
def _within(dist: Float32, r: Float32) -> Bool:
    """DEVIATION 5407: within nextafter(r, 0) is dist < r (dist == 0 at r == 0)."""
    if r > Float32(0):
        return dist < r
    return dist == Float32(0)


def mi_cc_unit(t: Int, f: FP, q: IP):
    """q = [Z, n, d, Y, k, TERM]; t = i*d + c. TERM = psi(nx) + psi(ny), the
    marginal counts (self included) within the k-th Chebyshev neighbour
    radius of the joint (x, y) sample."""
    var n = p(q, 1)
    var d = p(q, 2)
    var k = p(q, 4)
    var i = t // d
    var c = t % d
    var xi = ld(f, p(q, 0) + i * d + c)
    var yi = ld(f, p(q, 3) + i)
    var best = InlineArray[Float32, MAX_K](fill=Float32(3.4028235e38))
    for j in range(n):
        if j == i:
            continue
        var dx = abs(sub(ld(f, p(q, 0) + j * d + c), xi))
        var dy = abs(sub(ld(f, p(q, 3) + j), yi))
        var dist = dx if dx > dy else dy
        if dist < best[k - 1]:
            var m = k - 1
            while m > 0 and best[m - 1] > dist:
                best[m] = best[m - 1]
                m -= 1
            best[m] = dist
    var r = best[k - 1]
    var nx = 0
    var ny = 0
    for j in range(n):
        if _within(abs(sub(ld(f, p(q, 0) + j * d + c), xi)), r):
            nx += 1
        if _within(abs(sub(ld(f, p(q, 3) + j), yi)), r):
            ny += 1
    st(f, p(q, 5) + t, add(digammaf(Float32(nx)), digammaf(Float32(ny))))


def mi_cd_unit(t: Int, f: FP, q: IP):
    """q = [Z, n, d, Y, LABCNT, k, TERM]; t = i*d + c. For a point whose
    class has more than one member: kl = min(k, count - 1), r the kl-th
    nearest same-class distance, m the points (of classes with more than one
    member, self included) within r; TERM = psi(kl) - psi(count) - psi(m).
    Otherwise TERM = 0 and the point is outside the mean."""
    var n = p(q, 1)
    var d = p(q, 2)
    var k = p(q, 5)
    var i = t // d
    var c = t % d
    var li = Int(ld(f, p(q, 3) + i))
    var cnt = Int(ld(f, p(q, 4) + li))
    if cnt <= 1:
        st(f, p(q, 6) + t, Float32(0))
        return
    var kl = k if k < cnt - 1 else cnt - 1
    var xi = ld(f, p(q, 0) + i * d + c)
    var best = InlineArray[Float32, MAX_K](fill=Float32(3.4028235e38))
    for j in range(n):
        if j == i or Int(ld(f, p(q, 3) + j)) != li:
            continue
        var dist = abs(sub(ld(f, p(q, 0) + j * d + c), xi))
        if dist < best[kl - 1]:
            var m = kl - 1
            while m > 0 and best[m - 1] > dist:
                best[m] = best[m - 1]
                m -= 1
            best[m] = dist
    var r = best[kl - 1]
    var mall = 0
    for j in range(n):
        if Int(ld(f, p(q, 4) + Int(ld(f, p(q, 3) + j)))) <= 1:
            continue
        if _within(abs(sub(ld(f, p(q, 0) + j * d + c), xi)), r):
            mall += 1
    st(f, p(q, 6) + t, sub(sub(digammaf(Float32(kl)), digammaf(Float32(cnt))), digammaf(Float32(mall))))


def mi_reduce_unit(t: Int, f: FP, q: IP):
    """q = [TERM, n, d, KIND, k, NUSED, OUT]; t = column. KIND 0 (cc):
    psi(n) + psi(k) - mean(TERM); KIND 1 (cd): psi(NUSED) + sum(TERM) / NUSED.
    Negative estimates are 0."""
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
        mi = add(digammaf(Float32(used)), div(s, Float32(used))) if used > 0 else Float32(0)
    st(f, p(q, 6) + t, mi if mi > Float32(0) else Float32(0))
