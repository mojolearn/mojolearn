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
from std.sys.compile import is_defined, get_defined_int
from std.sys.info import has_apple_gpu_accelerator

#: lane layernorm-idn N3 (2026-10-09), `-D MOJOLEARN_IDN_LN_ROW_WARP`, IDENTICAL,
#: default OFF, BITS CHANGE (a fold order; allowed within one version, every
#: column together). The row reductions of the forward (sum x, sum (x-mean)^2)
#: and of the backward-x (sum g, sum g xhat) stop being one ascending chain of
#: D terms and become a FIXED 32-LANE FOLD: logical lane j (0 <= j < 32) folds
#: elements c = j, j + 32, j + 64, ... ascending from +0.0, then a five-level
#: pairwise tree (strides 16, 8, 4, 2, 1; level h adds partial[j + h] into
#: partial[j] for j < h) gives the row total. The tree is spelled on LN_RW_LANES
#: = 32 LOGICAL lanes everywhere: the device warp cell (sequence/coop.mojo,
#: `shuffle_xor` with strides < 32 never leaves a 32-lane half of a 64-wide
#: CDNA wavefront), the one-thread device op (D < 32 rows, or the coop path
#: off) and the host twin (`_rw_fold` below) run the same 32 partials and the
#: same tree, so NVIDIA == AMD == host bit for bit.
#: Cost reasoning: the one-thread chain (and the coop chain, which every lane
#: replays over broadcast words) costs D dependent adds per row, about 4 D
#: cycles of latency, while the row's bytes (4 D) would take ~D/32 coalesced
#: loads per lane: at any D >= 32 the chain, not the memory, bounds the kernel.
#: The 32-lane fold cuts the dependent chain to D/32 + 5 steps per reduction
#: (32x fewer at D = 1024, 1 instead of ~30 us per row-wave); the two trees
#: per reduction stage cost 5 shuffles each. 32 lanes (not 64 or a block) so
#: one cell is one warp on every vendor with no shared memory or barrier and
#: the tree shape does not depend on the row width; a wider row only gives
#: each lane more sequential terms (D/32), still bandwidth-paced.
#: dweight / dbias column folds are untouched (LN_FOLD_BLOCK above).
comptime LN_ROW_WARP = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_LN_ROW_WARP"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: logical lanes of the row fold (the tree has log2(LN_RW_LANES) = 5 levels)
comptime LN_RW_LANES = 32

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
comptime LN_FOLD_BLOCK = (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_LN_FOLD_BLOCK_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())) or (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_LN_FAST_BLOCK_FOLD"]())


def ln_fold_rows(M: Int) -> Int:
    """Rows per block: the smallest power of two R >= 64 with R * R >= M."""
    # NI51 independent V arm: more independent parameter-gradient leaves;
    # sqrt(M) growth still bounds scratch. Shared HostExec/DeviceExec operation
    # source; only neural LayerNorm changes. Default OFF, validation NOT RUN.
    # L11 (2026-10-07): a leaf parameter, -D MOJOLEARN_IDN_SEQ_LN_LEAF=64|32
    # (default 64; legal set asserted in core/six_lane_experiment_guards).
    var r = 64
    comptime if (
        GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and LN_FOLD_BLOCK
        and get_defined_int["MOJOLEARN_IDN_SEQ_LN_LEAF", 64]() == 32
        and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    ):
        r = 32
    while r * r < M:
        r *= 2
    return r


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def rw_tree(p: SIMD[DType.float32, LN_RW_LANES]) -> Float32:
    """The row fold's tree over the 32 lane partials: level h (16, 8, 4, 2, 1)
    adds p[j + h] into p[j] for j < h; p[0] is the total. The device cell
    computes exactly these adds (`coop_rw_total`, sequence/coop.mojo)."""
    var v = p
    comptime for lv in range(5):
        comptime h = LN_RW_LANES >> (lv + 1)
        comptime for j in range(h):
            v[j] = add(v[j], v[j + h])
    return v[0]


@always_inline
def _rw_fold_sum(x: FP, base: Int, D: Int) -> Float32:
    """sum of x[base + c], c < D, as the 32-lane fold (LN_ROW_WARP): lane j
    adds c = j, j + 32, ... ascending from +0.0, then `rw_tree`."""
    var p = SIMD[DType.float32, LN_RW_LANES](0.0)
    var k = 0
    while k < D:
        var m = min(LN_RW_LANES, D - k)
        for j in range(m):
            p[j] = add(p[j], ld(x, base + k + j))
        k += LN_RW_LANES
    return rw_tree(p)


@always_inline
def _rw_fold_sq(x: FP, base: Int, D: Int, mean: Float32) -> Float32:
    """sum of (x[base + c] - mean)^2 as the 32-lane fold: lane j's chain is
    fma3(d, d, acc) over c = j, j + 32, ... from +0.0, then `rw_tree`."""
    var p = SIMD[DType.float32, LN_RW_LANES](0.0)
    var k = 0
    while k < D:
        var m = min(LN_RW_LANES, D - k)
        for j in range(m):
            var d = sub(ld(x, base + k + j), mean)
            p[j] = fma3(d, d, p[j])
        k += LN_RW_LANES
    return rw_tree(p)


def _stats(x: FP, base: Int, D: Int, eps: Float32) -> Tuple[Float32, Float32]:
    var s = Float32(0.0)
    var q = Float32(0.0)
    var mean = Float32(0.0)
    comptime if LN_ROW_WARP:
        s = _rw_fold_sum(x, base, D)
        mean = div(s, Float32(D))
        q = _rw_fold_sq(x, base, D, mean)
    else:
        for c in range(D):
            s = add(s, ld(x, base + c))
        mean = div(s, Float32(D))
        for c in range(D):
            var d = sub(ld(x, base + c), mean)
            q = fma3(d, d, q)
    var rstd = ftz(identical_rsqrt(add(div(q, Float32(D)), eps)))
    return (mean, rstd)


@always_inline
def ln_bwd_g(a: Args, base: Int, c: Int, mean: Float32, rstd: Float32) -> Tuple[Float32, Float32]:
    """(g, xhat) of column c of a backward-x row: g = dy (w), xhat = (x - mean)
    rstd; the one expression every LN backward body shares."""
    var g = ld(a.p0, base + c)
    if a.i1 != 0:
        g = mul(g, ld(a.p2, c))
    var xh = mul(sub(ld(a.p1, base + c), mean), rstd)
    return (g, xh)


@always_inline
def _rw_fold_bwd(a: Args, base: Int, D: Int, mean: Float32, rstd: Float32) -> Tuple[Float32, Float32]:
    """(sum g, sum g xhat) of a backward-x row as the 32-lane fold: lane j
    runs add(sg, g) and fma3(g, xh, sgx) over c = j, j + 32, ... from +0.0,
    then one `rw_tree` per sum."""
    var pg = SIMD[DType.float32, LN_RW_LANES](0.0)
    var pgx = SIMD[DType.float32, LN_RW_LANES](0.0)
    var k = 0
    while k < D:
        var m = min(LN_RW_LANES, D - k)
        for j in range(m):
            var gx = ln_bwd_g(a, base, k + j, mean, rstd)
            pg[j] = add(pg[j], gx[0])
            pgx[j] = fma3(gx[0], gx[1], pgx[j])
        k += LN_RW_LANES
    return (rw_tree(pg), rw_tree(pgx))


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
    comptime if LN_ROW_WARP:
        var ss = _rw_fold_bwd(a, base, D, mean, rstd)
        sg = ss[0]
        sgx = ss[1]
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
