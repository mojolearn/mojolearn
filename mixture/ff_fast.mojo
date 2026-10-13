# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Float-float (hi + lo, about 48 bits) arithmetic for GaussianMixture's FAST
Apple covariance and precision Cholesky (lane apple-holes, 2026-10-12;
`GMM_FAST_FF_COV` in mixture/checks/mstep.mojo).

x_linear/ff.mojo builds the same transformations on `identical_mul_add`,
which is `a * b + c` (two roundings, or a compiler's contraction) outside
IDENTICAL, so its two-product loses the error term in a FAST build. Here
every step is pinned in every mode: sums are `llvm.fma.f32(a, 1, b)` (one
rounding, never reassociated), products `pinned_mul_f32`, error terms
`llvm.fma.f32`. Quotients and roots start from the hardware's float32 value
and are corrected by exact residuals, so an approximate divide or sqrt only
costs a correction step, not accuracy. The error-free transformations are
Knuth's two-sum, Dekker's fast two-sum and the fma two-product; add,
multiply, divide and square root are Hida, Li and Bailey's double-double
forms."""
from std.math import sqrt
from std.sys import llvm_intrinsic
from checks.numerics import pinned_mul_f32


@fieldwise_init
struct GF(ImplicitlyCopyable, Movable):
    var hi: Float32
    var lo: Float32


@always_inline
def gf_rn_add(a: Float32, b: Float32) -> Float32:
    """a + b, one rounding, kept out of any reassociation."""
    return llvm_intrinsic["llvm.fma.f32", Float32, has_side_effect=False](a, Float32(1), b)


@always_inline
def gf_rn_fma(a: Float32, b: Float32, c: Float32) -> Float32:
    return llvm_intrinsic["llvm.fma.f32", Float32, has_side_effect=False](a, b, c)


@always_inline
def gf_of(a: Float32) -> GF:
    return GF(a, Float32(0))


@always_inline
def gf_two_sum(a: Float32, b: Float32) -> GF:
    var s = gf_rn_add(a, b)
    var bb = gf_rn_add(s, -a)
    var e = gf_rn_add(gf_rn_add(a, -gf_rn_add(s, -bb)), gf_rn_add(b, -bb))
    return GF(s, e)


@always_inline
def gf_fast_two_sum(a: Float32, b: Float32) -> GF:
    """|a| >= |b| (or a == 0)."""
    var s = gf_rn_add(a, b)
    return GF(s, gf_rn_add(b, -gf_rn_add(s, -a)))


@always_inline
def gf_two_prod(a: Float32, b: Float32) -> GF:
    var p = pinned_mul_f32(a, b)
    return GF(p, gf_rn_fma(a, b, -p))


@always_inline
def gf_add(x: GF, y: GF) -> GF:
    var s = gf_two_sum(x.hi, y.hi)
    var t = gf_two_sum(x.lo, y.lo)
    var e = gf_rn_add(s.lo, t.hi)
    var u = gf_fast_two_sum(s.hi, e)
    e = gf_rn_add(u.lo, t.lo)
    return gf_fast_two_sum(u.hi, e)


@always_inline
def gf_add_f(x: GF, b: Float32) -> GF:
    var s = gf_two_sum(x.hi, b)
    var e = gf_rn_add(s.lo, x.lo)
    return gf_fast_two_sum(s.hi, e)


@always_inline
def gf_sub(x: GF, y: GF) -> GF:
    return gf_add(x, GF(-y.hi, -y.lo))


@always_inline
def gf_mul(x: GF, y: GF) -> GF:
    var p = gf_two_prod(x.hi, y.hi)
    var e = gf_rn_fma(x.hi, y.lo, p.lo)
    e = gf_rn_fma(x.lo, y.hi, e)
    return gf_fast_two_sum(p.hi, e)


@always_inline
def gf_mul_f(x: GF, b: Float32) -> GF:
    var p = gf_two_prod(x.hi, b)
    var e = gf_rn_fma(x.lo, b, p.lo)
    return gf_fast_two_sum(p.hi, e)


@always_inline
def gf_div(x: GF, y: GF) -> GF:
    """y != 0."""
    var q1 = x.hi / y.hi
    var r = gf_sub(x, gf_mul_f(y, q1))
    var q2 = r.hi / y.hi
    r = gf_sub(r, gf_mul_f(y, q2))
    var q3 = r.hi / y.hi
    var q = gf_fast_two_sum(q1, q2)
    return gf_add_f(q, q3)


@always_inline
def gf_sqrt(a: GF) -> GF:
    """a.hi > 0."""
    var s = sqrt(a.hi)
    var r = gf_sub(a, gf_two_prod(s, s))
    var c = r.hi / pinned_mul_f32(Float32(2), s)
    return gf_fast_two_sum(s, c)


@always_inline
def gf_f32(x: GF) -> Float32:
    """x rounded once to float32."""
    return gf_rn_add(x.hi, x.lo)
