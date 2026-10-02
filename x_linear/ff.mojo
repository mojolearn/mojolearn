# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FLOAT-FLOAT arithmetic (lane/neural-pass93, 2026-10-01; Andrew: Ridge on an
ill-conditioned Gram such as istella's solves in paired float32, about 48
bits). A value is hi + lo with |lo| <= ulp(hi) / 2. Every operation is the
lane's IDENTICAL float32 arithmetic (x_linear/ops.mojo `fa`, `fs`, `fm`,
`fd`, `fmad`, `fsqrt`: correctly rounded, the products pinned, every result
flushed by `fz`), composed into the classical error-free transformations
(Knuth's two-sum, Dekker's fast two-sum, the fused-multiply-add two-product;
Hida, Li and Bailey's double-double add, multiply, divide and square root).
So a float-float result is a fixed sequence of IDENTICAL operations: the same
words on the host and on every device. The flush only touches error terms
below 2^-126, far under the precision any caller needs.
"""
from x_linear.ops import FP, fa, fs, fm, fd, fmad, fsqrt, ld, st


@fieldwise_init
struct FF(ImplicitlyCopyable, Movable):
    var hi: Float32
    var lo: Float32


@always_inline
def ff_of(a: Float32) -> FF:
    return FF(a, Float32(0))


@always_inline
def two_sum(a: Float32, b: Float32) -> FF:
    var s = fa(a, b)
    var bb = fs(s, a)
    var e = fa(fs(a, fs(s, bb)), fs(b, bb))
    return FF(s, e)


@always_inline
def fast_two_sum(a: Float32, b: Float32) -> FF:
    """|a| >= |b| (or a == 0)."""
    var s = fa(a, b)
    return FF(s, fs(b, fs(s, a)))


@always_inline
def two_prod(a: Float32, b: Float32) -> FF:
    var p = fm(a, b)
    return FF(p, fmad(a, b, -p))


@always_inline
def ff_add(x: FF, y: FF) -> FF:
    var s = two_sum(x.hi, y.hi)
    var t = two_sum(x.lo, y.lo)
    var e = fa(s.lo, t.hi)
    var u = fast_two_sum(s.hi, e)
    e = fa(u.lo, t.lo)
    return fast_two_sum(u.hi, e)


@always_inline
def ff_neg(x: FF) -> FF:
    return FF(-x.hi, -x.lo)


@always_inline
def ff_sub(x: FF, y: FF) -> FF:
    return ff_add(x, ff_neg(y))


@always_inline
def ff_add_f(x: FF, b: Float32) -> FF:
    var s = two_sum(x.hi, b)
    var e = fa(s.lo, x.lo)
    return fast_two_sum(s.hi, e)


@always_inline
def ff_mul(x: FF, y: FF) -> FF:
    var p = two_prod(x.hi, y.hi)
    var e = fmad(x.hi, y.lo, p.lo)
    e = fmad(x.lo, y.hi, e)
    return fast_two_sum(p.hi, e)


@always_inline
def ff_mul_f(x: FF, b: Float32) -> FF:
    var p = two_prod(x.hi, b)
    var e = fmad(x.lo, b, p.lo)
    return fast_two_sum(p.hi, e)


@always_inline
def ff_div(x: FF, y: FF) -> FF:
    var q1 = fd(x.hi, y.hi)
    var r = ff_sub(x, ff_mul_f(y, q1))
    var q2 = fd(r.hi, y.hi)
    r = ff_sub(r, ff_mul_f(y, q2))
    var q3 = fd(r.hi, y.hi)
    var q = fast_two_sum(q1, q2)
    return ff_add_f(q, q3)


@always_inline
def ff_sqrt(a: FF) -> FF:
    """a > 0."""
    var s = fsqrt(a.hi)
    var r = ff_sub(a, two_prod(s, s))
    var c = fd(r.hi, fm(Float32(2), s))
    return fast_two_sum(s, c)


@always_inline
def ff_f32(x: FF) -> Float32:
    return fa(x.hi, x.lo)


@always_inline
def ff_ld(h: FP, l: FP, i: Int) -> FF:
    return FF(ld(h, i), ld(l, i))


@always_inline
def ff_st(h: FP, l: FP, i: Int, v: FF):
    st(h, i, v.hi)
    st(l, i, v.lo)


def ff_cholesky(ah: FP, al: FP, m: Int) -> Bool:
    """`cholesky`'s order (column j ascending, every inner sum k ascending) in
    float-float on the m x m block (hi words ah, lo words al), in place,
    lower triangle. False on a pivot that is not positive."""
    for j in range(m):
        var s = ff_ld(ah, al, j * m + j)
        for k in range(j):
            var l = ff_ld(ah, al, j * m + k)
            s = ff_sub(s, ff_mul(l, l))
        if not (s.hi > 0):
            return False
        var r = ff_sqrt(s)
        ff_st(ah, al, j * m + j, r)
        for i in range(j + 1, m):
            var tv = ff_ld(ah, al, i * m + j)
            for k in range(j):
                tv = ff_sub(tv, ff_mul(ff_ld(ah, al, i * m + k), ff_ld(ah, al, j * m + k)))
            ff_st(ah, al, i * m + j, ff_div(tv, r))
    return True


def ff_chol_solve(lh: FP, ll: FP, m: Int, bh: FP, bl: FP):
    """`chol_solve`'s order in float-float: L L' x = b, b <- x."""
    for i in range(m):
        var tv = ff_ld(bh, bl, i)
        for k in range(i):
            tv = ff_sub(tv, ff_mul(ff_ld(lh, ll, i * m + k), ff_ld(bh, bl, k)))
        ff_st(bh, bl, i, ff_div(tv, ff_ld(lh, ll, i * m + i)))
    var i = m - 1
    while i >= 0:
        var tv = ff_ld(bh, bl, i)
        for k in range(i + 1, m):
            tv = ff_sub(tv, ff_mul(ff_ld(lh, ll, k * m + i), ff_ld(bh, bl, k)))
        ff_st(bh, bl, i, ff_div(tv, ff_ld(lh, ll, i * m + i)))
        i -= 1


# ------------------------------------------------ the centered statistics
def ff_col_mean(v: FP, step: Int, off: Int, n: Int, s: Int, e: Int, w: FP, wo: Int, sw: Bool, wsum: FF) -> FF:
    """The (weighted) mean of v[off + i*step] over the rows outside [s, e), ascending."""
    var acc = FF(Float32(0), Float32(0))
    for i in range(n):
        if i >= s and i < e:
            continue
        if sw:
            acc = ff_add(acc, two_prod(ld(w, wo + i), ld(v, off + i * step)))
        else:
            acc = ff_add_f(acc, ld(v, off + i * step))
    return ff_div(acc, wsum)


@always_inline
def ff_centered(a: Float32, m: FF) -> FF:
    """a - m, a float32 word."""
    var s = two_sum(a, -m.hi)
    return fast_two_sum(s.hi, fs(s.lo, m.lo))


def ff_cross(a: FP, astep: Int, aoff: Int, ma: FF, b: FP, bstep: Int, boff: Int, mb: FF,
             n: Int, s: Int, e: Int, w: FP, wo: Int, sw: Bool) -> FF:
    """sum (w_i) (a_i - ma)(b_i - mb) over the rows outside [s, e), ascending, in float-float."""
    var acc = FF(Float32(0), Float32(0))
    for i in range(n):
        if i >= s and i < e:
            continue
        var ca = ff_centered(ld(a, aoff + i * astep), ma)
        var cb = ff_centered(ld(b, boff + i * bstep), mb)
        var p = ff_mul(ca, cb)
        if sw:
            p = ff_mul_f(p, ld(w, wo + i))
        acc = ff_add(acc, p)
    return acc
