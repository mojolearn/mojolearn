# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE'S SEAM ORACLES (DEVIATIONS 5000-5009, IDENTITY_PATHS rows
100-109): each seam of x_linear/ restated as plain host code, independent of
x_linear/ops.mojo, together with the UNPINNED spelling the fixture must
separate it from. Built and run only under IDENTICAL
(tools/with_identical_mode.sh); x_linear/checks/seams_check.mojo drives it.
"""
from std.memory import bitcast
from std.math import fma, exp, log
from checks.numerics import portable_expf, portable_logf, portable_divf, portable_sqrtf, portable_powf


def flush(x: Float32) -> Float32:
    """IDENTITY_PATHS row 10's policy restated on the bits: a subnormal
    becomes its signed zero."""
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x7F800000)) == 0 and (b & UInt32(0x007FFFFF)) != 0:
        return bitcast[DType.float32](b & UInt32(0x80000000))
    return x


def mul(a: Float32, b: Float32) -> Float32:
    """A product rounded on its own: fma(a, b, -0.0) is exactly round(a*b)."""
    return flush(fma(flush(a), flush(b), Float32(-0.0)))


def add(a: Float32, b: Float32) -> Float32:
    return flush(flush(a) + flush(b))


def sub(a: Float32, b: Float32) -> Float32:
    return flush(flush(a) - flush(b))


def div(a: Float32, b: Float32) -> Float32:
    return flush(portable_divf(flush(a), flush(b)))


def sqrt_(a: Float32) -> Float32:
    return flush(portable_sqrtf(flush(a)))


# 5000: the dot fold -----------------------------------------------------------

def dot_pinned(a: List[Float32], b: List[Float32]) -> Float32:
    """j ascending, one fused multiply-add per term, every operand flushed."""
    var acc = Float32(0)
    for j in range(len(a)):
        acc = flush(fma(flush(a[j]), flush(b[j]), flush(acc)))
    return acc


def dot_reversed(a: List[Float32], b: List[Float32]) -> Float32:
    var acc = Float32(0)
    for jj in range(len(a)):
        var j = len(a) - 1 - jj
        acc = flush(fma(flush(a[j]), flush(b[j]), flush(acc)))
    return acc


def dot_unfused(a: List[Float32], b: List[Float32]) -> Float32:
    var acc = Float32(0)
    for j in range(len(a)):
        acc = add(acc, mul(a[j], b[j]))
    return acc


# 5001: the operand flush --------------------------------------------------------

def add_unflushed(a: Float32, b: Float32) -> Float32:
    return a + b


# 5002: Cholesky -----------------------------------------------------------------

def cholesky_pinned(mut a: List[Float32], m: Int):
    """Column j ascending; each inner sum k ascending; every product rounded."""
    for j in range(m):
        var s = a[j * m + j]
        for k in range(j):
            s = sub(s, mul(a[j * m + k], a[j * m + k]))
        var r = sqrt_(s)
        a[j * m + j] = r
        for i in range(j + 1, m):
            var t = a[i * m + j]
            for k in range(j):
                t = sub(t, mul(a[i * m + k], a[j * m + k]))
            a[i * m + j] = div(t, r)


def cholesky_reversed_inner(mut a: List[Float32], m: Int):
    for j in range(m):
        var s = a[j * m + j]
        for kk in range(j):
            var k = j - 1 - kk
            s = sub(s, mul(a[j * m + k], a[j * m + k]))
        var r = sqrt_(s)
        a[j * m + j] = r
        for i in range(j + 1, m):
            var t = a[i * m + j]
            for kk in range(j):
                var k = j - 1 - kk
                t = sub(t, mul(a[i * m + k], a[j * m + k]))
            a[i * m + j] = div(t, r)


# 5003: the Jacobi sweep ---------------------------------------------------------

def _abs(x: Float32) -> Float32:
    return -x if x < 0 else x


def _sign(x: Float32) -> Float32:
    if x > 0:
        return Float32(1)
    if x < 0:
        return Float32(-1)
    return Float32(0)


def jacobi(mut a: List[Float32], mut v: List[Float32], m: Int, sweeps: Int, q_descending: Bool):
    """Cyclic Jacobi, p ascending and q ascending (or q descending, the
    unpinned order), Rutishauser's rotation, the 1e-9 relative skip."""
    for i in range(m):
        for j in range(m):
            v[i * m + j] = Float32(1) if i == j else Float32(0)
    for _ in range(sweeps):
        var rotated = False
        for p in range(m):
            for qq in range(p + 1, m):
                var q = (m - 1 - (qq - p - 1)) if q_descending else qq
                var apq = a[p * m + q]
                var app = a[p * m + p]
                var aqq = a[q * m + q]
                var scale = sqrt_(_abs(mul(app, aqq)))
                if _abs(apq) <= mul(Float32(1e-9), scale) or apq == 0:
                    continue
                rotated = True
                var theta = div(sub(aqq, app), mul(Float32(2), apq))
                var t = div(_sign(theta) if theta != 0 else Float32(1),
                            add(_abs(theta), sqrt_(add(mul(theta, theta), Float32(1)))))
                var c = div(Float32(1), sqrt_(add(mul(t, t), Float32(1))))
                var s = mul(t, c)
                for k in range(m):
                    var akp = a[k * m + p]
                    var akq = a[k * m + q]
                    a[k * m + p] = sub(mul(c, akp), mul(s, akq))
                    a[k * m + q] = add(mul(s, akp), mul(c, akq))
                for k in range(m):
                    var apk = a[p * m + k]
                    var aqk = a[q * m + k]
                    a[p * m + k] = sub(mul(c, apk), mul(s, aqk))
                    a[q * m + k] = add(mul(s, apk), mul(c, aqk))
                for k in range(m):
                    var vkp = v[k * m + p]
                    var vkq = v[k * m + q]
                    v[k * m + p] = sub(mul(c, vkp), mul(s, vkq))
                    v[k * m + q] = add(mul(s, vkp), mul(c, vkq))
        if not rotated:
            return


# 5004: the shuffle --------------------------------------------------------------

def splitmix(mut s: UInt64) -> UInt64:
    s = s + UInt64(0x9E3779B97F4A7C15)
    var z = s
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def shuffle_mod(n: Int, seed: UInt64) -> List[Int]:
    """Fisher-Yates, i descending, j = draw mod (i + 1)."""
    var idx = List[Int]()
    for i in range(n):
        idx.append(i)
    var s = seed
    var i = n - 1
    while i > 0:
        var j = Int(splitmix(s) % UInt64(i + 1))
        var t = idx[i]
        idx[i] = idx[j]
        idx[j] = t
        i -= 1
    return idx^


def shuffle_mulshift(n: Int, seed: UInt64) -> List[Int]:
    """The unpinned mapping: j = ((draw >> 32) * (i + 1)) >> 32."""
    var idx = List[Int]()
    for i in range(n):
        idx.append(i)
    var s = seed
    var i = n - 1
    while i > 0:
        var j = Int(((splitmix(s) >> 32) * UInt64(i + 1)) >> 32)
        var t = idx[i]
        idx[i] = idx[j]
        idx[j] = t
        i -= 1
    return idx^


# 5005: first index on a tie -----------------------------------------------------

def argmax_abs_first(v: List[Float32]) -> Int:
    var best = 0
    for j in range(1, len(v)):
        if _abs(v[j]) > _abs(v[best]):
            best = j
    return best


def argmax_abs_last(v: List[Float32]) -> Int:
    var best = 0
    for j in range(1, len(v)):
        if _abs(v[j]) >= _abs(v[best]):
            best = j
    return best


# 5006: the NaN word -------------------------------------------------------------

comptime CANONICAL_NAN_BITS = UInt32(0x7FC00000)


# 5007: the intercept joins last ---------------------------------------------------

def score_intercept_last(x: List[Float32], w: List[Float32], b: Float32) -> Float32:
    return add(dot_pinned(x, w), b)


def score_intercept_first(x: List[Float32], w: List[Float32], b: Float32) -> Float32:
    var acc = b
    for j in range(len(x)):
        acc = flush(fma(flush(x[j]), flush(w[j]), flush(acc)))
    return acc


# 5008: portable transcendentals ----------------------------------------------------

def exp_pinned(x: Float32) -> Float32:
    return flush(portable_expf(flush(x)))


def log_pinned(x: Float32) -> Float32:
    return flush(portable_logf(flush(x)))


def exp_libm(x: Float32) -> Float32:
    return exp(x)


def log_libm(x: Float32) -> Float32:
    return log(x)


# 5009: the alpha grid ---------------------------------------------------------------

def grid_pinned(amax: Float32, eps: Float32, k: Int, a_n: Int) -> Float32:
    """alpha_k = alpha_max * exp((k / (A - 1)) * log(eps))."""
    var frac = div(Float32(k), Float32(a_n - 1))
    return mul(amax, exp_pinned(mul(frac, log_pinned(eps))))


def grid_logspace(amax: Float32, eps: Float32, k: Int, a_n: Int) -> Float32:
    """The unpinned spelling, np.geomspace's: exp of a linear interpolation
    between log(alpha_max) and log(alpha_max * eps)."""
    var frac = div(Float32(k), Float32(a_n - 1))
    var lo = log_pinned(amax)
    var hi = log_pinned(mul(amax, eps))
    return exp_pinned(add(lo, mul(frac, sub(hi, lo))))
