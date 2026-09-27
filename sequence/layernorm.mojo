# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LayerNorm over the last dimension, `torch.nn.LayerNorm` /
`torch.nn.functional.layer_norm` (biased variance, eps inside the rsqrt,
optional elementwise weight and bias), forward and backward. Every
reduction is one thread's ascending loop: a row's mean and variance, a
column's weight and bias gradients over rows. The backward is the
reference's closed form dx = rstd (g - mean(g) - xhat mean(g xhat)) with
g = dy w."""
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import ftz, identical_div, identical_rsqrt


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def _stats(x: FP, base: Int, D: Int, eps: Float32) -> Tuple[Float32, Float32]:
    var s = Float32(0.0)
    for c in range(D):
        s = add(s, ld(x, base + c))
    var mean = div(s, Float32(D))
    var q = Float32(0.0)
    for c in range(D):
        var d = sub(ld(x, base + c), mean)
        q = fma3(d, d, q)
    var rstd = ftz(identical_rsqrt(add(div(q, Float32(D)), eps)))
    return (mean, rstd)


def op_ln_fwd(t: Int, a: Args):
    """Row t of x p0 [M, D]: y p3 = (x - mean) rstd (w p1) (+ b p2); p4 mean,
    p5 rstd per row. i0 D, i1 has weight, i2 has bias; f0 eps."""
    var D = a.i0
    var base = t * D
    var ms = _stats(a.p0, base, D, a.f0)
    st(a.p4, t, ms[0])
    st(a.p5, t, ms[1])
    for c in range(D):
        var y = mul(sub(ld(a.p0, base + c), ms[0]), ms[1])
        if a.i1 != 0:
            y = mul(y, ld(a.p1, c))
        if a.i2 != 0:
            y = add(y, ld(a.p2, c))
        st(a.p3, base + c, y)


def op_ln_bwd_x(t: Int, a: Args):
    """Row t: dx p3 from dy p0, x p1, w p2 (i1 has weight), mean p4, rstd p5;
    i0 D."""
    var D = a.i0
    var base = t * D
    var mean = ld(a.p4, t)
    var rstd = ld(a.p5, t)
    var sg = Float32(0.0)
    var sgx = Float32(0.0)
    for c in range(D):
        var g = ld(a.p0, base + c)
        if a.i1 != 0:
            g = mul(g, ld(a.p2, c))
        var xh = mul(sub(ld(a.p1, base + c), mean), rstd)
        sg = add(sg, g)
        sgx = fma3(g, xh, sgx)
    var mg = div(sg, Float32(D))
    var mgx = div(sgx, Float32(D))
    for c in range(D):
        var g = ld(a.p0, base + c)
        if a.i1 != 0:
            g = mul(g, ld(a.p2, c))
        var xh = mul(sub(ld(a.p1, base + c), mean), rstd)
        st(a.p3, base + c, mul(rstd, sub(sub(g, mg), mul(xh, mgx))))


def op_ln_bwd_w(t: Int, a: Args):
    """Column t: dw p2[t] = sum_r dy xhat, db p3[t] = sum_r dy, rows
    ascending. p0 dy, p1 x, p4 mean, p5 rstd; i0 D, i1 M."""
    var D = a.i0
    var sw = Float32(0.0)
    var sb = Float32(0.0)
    for r in range(a.i1):
        var dy = ld(a.p0, r * D + t)
        var xh = mul(sub(ld(a.p1, r * D + t), ld(a.p4, r)), ld(a.p5, r))
        sw = fma3(dy, xh, sw)
        sb = add(sb, dy)
    st(a.p2, t, sw)
    st(a.p3, t, sb)
