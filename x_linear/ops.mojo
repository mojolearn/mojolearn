# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE'S SHARED ARITHMETIC (lane/algos-linear, 2026-09-27).

Every fit in x_linear/ is ONE plain function over raw pointers. The host
binding calls it directly; the GPU binding calls the SAME function from a
one-thread kernel (x_linear/device.mojo). Pass 1 buys identity with a fully
sequential schedule: every reduction runs in ascending index order, every
operand goes through `ftz`, every product through `identical_mul`, every
multiply-add through `identical_mul_add`, every division through
`identical_div`, and exp/log/sqrt through the portable spellings. No heap
allocation happens inside a fit; the caller hands in the work buffers.
Speed (a parallel schedule with the same fold order) is pass 2.
"""
from std.sys.compile import is_defined
from checks.numerics import (
    ftz, identical_mul, identical_mul_add, identical_div, identical_sqrt,
    identical_exp, identical_log,
)

comptime FP = MutPointer[Float32, MutAnyOrigin]
# The CPU gate's negative control: a host build with this define folds every
# dot product in DESCENDING order, so each lane's CPU hash must move.
comptime X_LINEAR_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()
comptime IP = MutPointer[Int32, MutAnyOrigin]


@always_inline
def fa(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


@always_inline
def fs(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) - ftz(b))


@always_inline
def fm(a: Float32, b: Float32) -> Float32:
    return ftz(identical_mul(ftz(a), ftz(b)))


@always_inline
def fd(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(ftz(a), ftz(b)))


@always_inline
def fmad(a: Float32, b: Float32, c: Float32) -> Float32:
    """a * b + c, one rounding under IDENTICAL."""
    return ftz(identical_mul_add(ftz(a), ftz(b), ftz(c)))


@always_inline
def fsqrt(a: Float32) -> Float32:
    return ftz(identical_sqrt(ftz(a)))


@always_inline
def fexp(a: Float32) -> Float32:
    return ftz(identical_exp(ftz(a)))


@always_inline
def flog(a: Float32) -> Float32:
    return ftz(identical_log(ftz(a)))


@always_inline
def fabs(a: Float32) -> Float32:
    return -a if a < 0 else a


@always_inline
def fmax(a: Float32, b: Float32) -> Float32:
    return a if a >= b else b


@always_inline
def fmin(a: Float32, b: Float32) -> Float32:
    return a if a <= b else b


@always_inline
def fsign(a: Float32) -> Float32:
    if a > 0:
        return Float32(1)
    if a < 0:
        return Float32(-1)
    return Float32(0)


@always_inline
def ld(p: FP, i: Int) -> Float32:
    return p.unsafe_load(i)


@always_inline
def st(p: FP, i: Int, v: Float32):
    p.unsafe_store(i, v)


@always_inline
def ldi(p: IP, i: Int) -> Int:
    return Int(p.unsafe_load(i))


@always_inline
def sti(p: IP, i: Int, v: Int):
    p.unsafe_store(i, Int32(v))


@always_inline
def i2f(i: Int) -> Float32:
    """An integer count as float32: one correctly rounded conversion on every target."""
    return Float32(i)


def fill(p: FP, off: Int, count: Int, v: Float32):
    for i in range(count):
        st(p, off + i, v)


def copy(dst: FP, doff: Int, src: FP, soff: Int, count: Int):
    for i in range(count):
        st(dst, doff + i, ld(src, soff + i))


def dot(a: FP, ia: Int, b: FP, ib: Int, count: Int) -> Float32:
    """sum_j a[ia+j] * b[ib+j], j ascending, one fused multiply-add per term."""
    var acc = Float32(0)
    for jj in range(count):
        var j = count - 1 - jj if X_LINEAR_HOST_SABOTAGE else jj
        acc = fmad(ld(a, ia + j), ld(b, ib + j), acc)
    return acc


def row_dot(x: FP, i: Int, d: Int, w: FP, woff: Int) -> Float32:
    return dot(x, i * d, w, woff, d)


# ----------------------------------------------------------------- RNG
# splitmix64 (Steele, Lea, Flood 2014): integer only, so every target draws
# the same stream from the same seed.

@always_inline
def rng_next(mut s: UInt64) -> UInt64:
    s = s + UInt64(0x9E3779B97F4A7C15)
    var z = s
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def shuffle(idx: IP, n: Int, mut s: UInt64):
    """Fisher-Yates, i descending, j = draw mod (i + 1)."""
    var i = n - 1
    while i > 0:
        var j = Int(rng_next(s) % UInt64(i + 1))
        var t = ldi(idx, i)
        sti(idx, i, ldi(idx, j))
        sti(idx, j, t)
        i -= 1


# ------------------------------------------------------ dense linear algebra

def cholesky(a: FP, aoff: Int, m: Int) -> Bool:
    """In-place lower Cholesky of the m x m row-major block at `aoff`
    (upper triangle untouched). False on a non-positive pivot."""
    for j in range(m):
        var s = ld(a, aoff + j * m + j)
        for k in range(j):
            var l = ld(a, aoff + j * m + k)
            s = fs(s, fm(l, l))
        if not (s > 0):
            return False
        var r = fsqrt(s)
        st(a, aoff + j * m + j, r)
        for i in range(j + 1, m):
            var t = ld(a, aoff + i * m + j)
            for k in range(j):
                t = fs(t, fm(ld(a, aoff + i * m + k), ld(a, aoff + j * m + k)))
            st(a, aoff + i * m + j, fd(t, r))
    return True


def chol_solve(l: FP, loff: Int, m: Int, b: FP, boff: Int):
    """Solve L L^T x = b in place (b <- x)."""
    for i in range(m):
        var t = ld(b, boff + i)
        for k in range(i):
            t = fs(t, fm(ld(l, loff + i * m + k), ld(b, boff + k)))
        st(b, boff + i, fd(t, ld(l, loff + i * m + i)))
    var i = m - 1
    while i >= 0:
        var t = ld(b, boff + i)
        for k in range(i + 1, m):
            t = fs(t, fm(ld(l, loff + k * m + i), ld(b, boff + k)))
        st(b, boff + i, fd(t, ld(l, loff + i * m + i)))
        i -= 1


def jacobi_eig(a: FP, aoff: Int, v: FP, voff: Int, m: Int, max_sweeps: Int):
    """Cyclic Jacobi (Golub & Van Loan, Algorithm 8.5.3, the classical
    rotation of Rutishauser's form) on the symmetric m x m block at `aoff`:
    eigenvalues land on its diagonal, eigenvectors in the COLUMNS of `v`.
    Sweep order p ascending, q ascending; stops when a whole sweep rotates
    nothing (every |a_pq| below 1e-9 * sqrt(a_pp a_qq)) or at `max_sweeps`."""
    for i in range(m):
        for j in range(m):
            st(v, voff + i * m + j, Float32(1) if i == j else Float32(0))
    for _ in range(max_sweeps):
        var rotated = False
        for p in range(m):
            for q in range(p + 1, m):
                var apq = ld(a, aoff + p * m + q)
                var app = ld(a, aoff + p * m + p)
                var aqq = ld(a, aoff + q * m + q)
                var scale = fsqrt(fabs(fm(app, aqq)))
                if fabs(apq) <= fm(Float32(1e-9), scale) or apq == 0:
                    continue
                rotated = True
                var theta = fd(fs(aqq, app), fm(Float32(2), apq))
                var tt = fd(fsign(theta) if theta != 0 else Float32(1),
                            fa(fabs(theta), fsqrt(fa(fm(theta, theta), Float32(1)))))
                var c = fd(Float32(1), fsqrt(fa(fm(tt, tt), Float32(1))))
                var s = fm(tt, c)
                for k in range(m):
                    var akp = ld(a, aoff + k * m + p)
                    var akq = ld(a, aoff + k * m + q)
                    st(a, aoff + k * m + p, fs(fm(c, akp), fm(s, akq)))
                    st(a, aoff + k * m + q, fa(fm(s, akp), fm(c, akq)))
                for k in range(m):
                    var apk = ld(a, aoff + p * m + k)
                    var aqk = ld(a, aoff + q * m + k)
                    st(a, aoff + p * m + k, fs(fm(c, apk), fm(s, aqk)))
                    st(a, aoff + q * m + k, fa(fm(s, apk), fm(c, aqk)))
                for k in range(m):
                    var vkp = ld(v, voff + k * m + p)
                    var vkq = ld(v, voff + k * m + q)
                    st(v, voff + k * m + p, fs(fm(c, vkp), fm(s, vkq)))
                    st(v, voff + k * m + q, fa(fm(s, vkp), fm(c, vkq)))
        if not rotated:
            return


def col_means(x: FP, n: Int, d: Int, rows: IP, n_rows: Int, use_rows: Bool, res: FP, ooff: Int):
    """Column means over all rows (or the listed rows), rows ascending."""
    for j in range(d):
        var acc = Float32(0)
        for r in range(n_rows):
            var i = ldi(rows, r) if use_rows else r
            acc = fa(acc, ld(x, i * d + j))
        st(res, ooff + j, fd(acc, i2f(n_rows)))


def mean_of(y: FP, n: Int) -> Float32:
    var acc = Float32(0)
    for i in range(n):
        acc = fa(acc, ld(y, i))
    return fd(acc, i2f(n))


def centered_gram(x: FP, n: Int, d: Int, xm: FP, xmoff: Int, g: FP, goff: Int):
    """G = (X - 1 xm^T)^T (X - 1 xm^T), rows ascending, both triangles written."""
    for j in range(d):
        for k in range(j, d):
            var acc = Float32(0)
            var mj = ld(xm, xmoff + j)
            var mk = ld(xm, xmoff + k)
            for i in range(n):
                acc = fmad(fs(ld(x, i * d + j), mj), fs(ld(x, i * d + k), mk), acc)
            st(g, goff + j * d + k, acc)
            st(g, goff + k * d + j, acc)


def centered_xty(x: FP, y: FP, n: Int, d: Int, xm: FP, xmoff: Int, ym: Float32, res: FP, ooff: Int):
    for j in range(d):
        var acc = Float32(0)
        var mj = ld(xm, xmoff + j)
        for i in range(n):
            acc = fmad(fs(ld(x, i * d + j), mj), fs(ld(y, i), ym), acc)
        st(res, ooff + j, acc)
