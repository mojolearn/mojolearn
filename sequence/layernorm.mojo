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
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_rsqrt
from std.sys.compile import is_defined

#: lane idn-loss-norm-folds (2026-10-04): under IDENTICAL the dweight / dbias
#: column folds are BLOCKED on every column (device and host run this same
#: op): rows are cut into consecutive blocks of ln_fold_rows(M) rows, one
#: thread per (block, column) folds its rows ascending from +0.0, then one
#: thread per column adds the block partials ascending from +0.0. The block
#: size is a function of M alone. One block (M <= 64) is the single chain.
#: It replaced one M-term chain per column.
#: `-D MOJOLEARN_LN_FOLD_BLOCK_OFF` restores the single chain.
comptime LN_FOLD_BLOCK = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_LN_FOLD_BLOCK_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())


def ln_fold_rows(M: Int) -> Int:
    """Rows per block: the smallest power of two R >= 64 with R * R >= M."""
    var r = 64
    while r * r < M:
        r *= 2
    return r


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


#: rows whose loads are issued before they are folded (apple2): a column's
#: fold over a million rows waited out every strided load; the same fmas and
#: adds run in the same order, so the bits are the plain loop's.
comptime LN_STAGE = 16


@always_inline
def _ln_w_fold(a: Args, col: Int, r0: Int, r1: Int) -> Tuple[Float32, Float32]:
    """(sum dy xhat, sum dy) of column col over rows r0 .. r1 - 1, rows
    ascending, from zero. p0 dy, p1 x, p4 mean, p5 rstd; i0 D."""
    var D = a.i0
    var sw = Float32(0.0)
    var sb = Float32(0.0)
    var r = r0
    while r + LN_STAGE <= r1:
        var dv = SIMD[DType.float32, LN_STAGE]()
        var xv = SIMD[DType.float32, LN_STAGE]()
        comptime for j in range(LN_STAGE):
            dv[j] = ld(a.p0, (r + j) * D + col)
            xv[j] = mul(sub(ld(a.p1, (r + j) * D + col), ld(a.p4, r + j)), ld(a.p5, r + j))
        comptime for j in range(LN_STAGE):
            sw = fma3(dv[j], xv[j], sw)
            sb = add(sb, dv[j])
        r += LN_STAGE
    while r < r1:
        var dy = ld(a.p0, r * D + col)
        var xh = mul(sub(ld(a.p1, r * D + col), ld(a.p4, r)), ld(a.p5, r))
        sw = fma3(dy, xh, sw)
        sb = add(sb, dy)
        r += 1
    return (sw, sb)


def op_ln_bwd_w(t: Int, a: Args):
    """Column t: dw p2[t] = sum_r dy xhat, db p3[t] = sum_r dy, rows
    ascending. p0 dy, p1 x, p4 mean, p5 rstd; i0 D, i1 M.
    Split (FAST, and IDENTICAL's blocked fold; i2 = S > 0, i3 rows per split): thread t is split t // D of
    column t % D and writes its partials to p6 / p7 [S, D]; with i4 != 0 the
    thread (one per column) adds the S partials in order into p2 / p3."""
    var D = a.i0
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL or LN_FOLD_BLOCK:
        if a.i4 != 0:
            var sw = Float32(0.0)
            var sb = Float32(0.0)
            for s in range(a.i2):
                sw = add(sw, ld(a.p6, s * D + t))
                sb = add(sb, ld(a.p7, s * D + t))
            st(a.p2, t, sw)
            st(a.p3, t, sb)
            return
        if a.i2 > 0:
            var s = t // D
            var col = t - s * D
            var r0 = s * a.i3
            var r1 = min(a.i1, r0 + a.i3)
            var pw = _ln_w_fold(a, col, r0, r1)
            st(a.p6, t, pw[0])
            st(a.p7, t, pw[1])
            return
    var w = _ln_w_fold(a, t, 0, a.i1)
    st(a.p2, t, w[0])
    st(a.p3, t, w[1])
