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
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST, ftz, identical_div, identical_rsqrt
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

# S04 A/B — NOT TESTED — NOT COMPILED — NOT MEASURED. Default OFF.
# Canonical row statistics are written once, then independent feature cells
# run in parallel. Costs two additional launches and two M-word backward
# arrays; no size or vendor selects another fold. Full SQ forward/backward
# timing, quality and all-column identity remain pending.
comptime LN_SPLIT_APPLY = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NEURAL_S04_LN_SPLIT_APPLY"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

# V02 A/B — NOT TESTED — NOT COMPILED — NOT MEASURED. Default OFF.
# Named SEQUENCE-only profile ln-adjacent-tree-v1. Every scalar term is a
# +0-seeded leaf, adjacent pairs add with FTZ, odd nodes carry unchanged.
# Variance/dot leaves use one FMA from +0; epsilon and divisions retain
# their original placement. Forward, dx, dweight/dbias and HostExec share
# this implementation; the independent oracle restates the tree separately.
# This does not revise transformer/Mamba RMSNorm; those remain prerequisites
# for the broader V02 card. Within this profile every vendor MUST match;
# across-profile bits may differ. Task quality remains unestablished.
comptime LN_ADJACENT_TREE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NEURAL_V02_LN_ADJACENT_TREE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: lane idn-loss-norm-folds (2026-10-04): under IDENTICAL the dweight / dbias
#: column folds are BLOCKED on every column (device and host run this same
#: op): rows are cut into consecutive blocks of ln_fold_rows(M) rows, one
#: thread per (block, column) folds its rows ascending from +0.0, then one
#: thread per column adds the block partials ascending from +0.0. The block
#: size is a function of M alone. One block (M <= 64) is the single chain.
#: It replaced one M-term chain per column.
#: `-D MOJOLEARN_LN_FOLD_BLOCK_OFF` restores the single chain.
# F20 M3 2026-10-06 MEASURED FAST block-fold; stays OFF. Shapes131x17,137x65,
# 257x129: forward B/A0.9739/1.1342/0.9834, backward0.8322/2.0792/1.0993,
# downstream0.6184/1.2856/0.9373. Mixed/regression, one warmup+score; FAST quality
# varies within recorded contract. Existing IDENTICAL decision separate. ab-20261006/repairs-54c1f35a5/F20.
comptime LN_FOLD_BLOCK = (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not LN_ADJACENT_TREE and not (is_defined["MOJOLEARN_LN_FOLD_BLOCK_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())) or (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_LN_FAST_BLOCK_FOLD"]())


def ln_fold_rows(M: Int) -> Int:
    """Rows per block: the smallest power of two R >= 64 with R * R >= M."""
    var r = 64
    while r * r < M:
        r *= 2
    return r


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def _ln_tree[KIND: Int](a: Args, index: Int, n: Int, center: Float32 = 0.0, scale: Float32 = 1.0) -> Float32:
    """V02 fixed logical leaves. KIND 0=sum x, 1=centered square,
    2=sum g, 3=sum g*xhat, 4=dweight column, 5=dbias column.
    The binary-carry stack reproduces adjacent-pair levels with odd carry;
    its 64 slots cover the full Int indexing domain, independent of hardware.
    """
    var stack = InlineArray[Float32, 64](fill=Float32(0.0))
    var occupied = UInt64(0)
    for k in range(n):
        var term = Float32(0.0)
        comptime if KIND == 0:
            term = add(Float32(0.0), ld(a.p0, index + k))
        elif KIND == 1:
            var d = sub(ld(a.p0, index + k), center)
            term = fma3(d, d, Float32(0.0))
        elif KIND == 2 or KIND == 3:
            var g = ld(a.p0, index + k)
            if a.i1 != 0:
                g = mul(g, ld(a.p2, k))
            comptime if KIND == 2:
                term = add(Float32(0.0), g)
            else:
                var xh = mul(sub(ld(a.p1, index + k), center), scale)
                term = fma3(g, xh, Float32(0.0))
        else:
            var dy = ld(a.p0, k * a.i0 + index)
            comptime if KIND == 4:
                var xh = mul(sub(ld(a.p1, k * a.i0 + index), ld(a.p4, k)), ld(a.p5, k))
                term = fma3(dy, xh, Float32(0.0))
            else:
                term = add(Float32(0.0), dy)
        var level = 0
        var bit = UInt64(1)
        while (occupied & bit) != 0:
            term = add(stack[level], term)
            occupied = occupied ^ bit
            level += 1
            bit = bit << 1
        stack[level] = term
        occupied = occupied | bit
    var result = Float32(0.0)
    var have = False
    # Newest/smallest pending subtree joins the older subtree on its left.
    # This distinction matters for 7, 11, ... leaves with ragged tree tails.
    for level in range(64):
        if (occupied & (UInt64(1) << level)) != 0:
            result = add(stack[level], result) if have else stack[level]
            have = True
    return result


def _stats(x: FP, base: Int, D: Int, eps: Float32) -> Tuple[Float32, Float32]:
    comptime if LN_ADJACENT_TREE:
        var a = Args()
        a.p0 = x
        var mean = div(_ln_tree[0](a, base, D), Float32(D))
        var q = _ln_tree[1](a, base, D, mean)
        return (mean, ftz(identical_rsqrt(add(div(q, Float32(D)), eps))))
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


def op_ln_stats(t: Int, a: Args):
    """S04 stats only; the same Args and _stats as op_ln_fwd."""
    var ms = _stats(a.p0, t * a.i0, a.i0, a.f0)
    st(a.p4, t, ms[0])
    st(a.p5, t, ms[1])


def op_ln_apply(t: Int, a: Args):
    """S04 one cell, preserving the original subtraction/product/add seams."""
    var r = t // a.i0
    var c = t - r * a.i0
    var y = mul(sub(ld(a.p0, t), ld(a.p4, r)), ld(a.p5, r))
    if a.i1 != 0:
        y = mul(y, ld(a.p1, c))
    if a.i2 != 0:
        y = add(y, ld(a.p2, c))
    st(a.p3, t, y)


def op_ln_bwd_stats(t: Int, a: Args):
    """S04 the exact original backward row chains, means into p6/p7."""
    var D = a.i0
    var base = t * D
    var mean = ld(a.p4, t)
    var rstd = ld(a.p5, t)
    comptime if LN_ADJACENT_TREE:
        st(a.p6, t, div(_ln_tree[2](a, base, D, mean, rstd), Float32(D)))
        st(a.p7, t, div(_ln_tree[3](a, base, D, mean, rstd), Float32(D)))
        return
    var sg = Float32(0.0)
    var sgx = Float32(0.0)
    for c in range(D):
        var g = ld(a.p0, base + c)
        if a.i1 != 0:
            g = mul(g, ld(a.p2, c))
        var xh = mul(sub(ld(a.p1, base + c), mean), rstd)
        sg = add(sg, g)
        sgx = fma3(g, xh, sgx)
    st(a.p6, t, div(sg, Float32(D)))
    st(a.p7, t, div(sgx, Float32(D)))


def op_ln_bwd_apply(t: Int, a: Args):
    """S04 one dx cell; saved p6/p7 are immutable for this launch."""
    var r = t // a.i0
    var c = t - r * a.i0
    var g = ld(a.p0, t)
    if a.i1 != 0:
        g = mul(g, ld(a.p2, c))
    var rstd = ld(a.p5, r)
    var xh = mul(sub(ld(a.p1, t), ld(a.p4, r)), rstd)
    st(a.p3, t, mul(rstd, sub(sub(g, ld(a.p6, r)), mul(xh, ld(a.p7, r)))))


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
    comptime if LN_ADJACENT_TREE:
        sg = _ln_tree[2](a, base, D, mean, rstd)
        sgx = _ln_tree[3](a, base, D, mean, rstd)
    else:
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
    comptime if LN_ADJACENT_TREE:
        st(a.p2, t, _ln_tree[4](a, t, a.i1))
        st(a.p3, t, _ln_tree[5](a, t, a.i1))
        return
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
