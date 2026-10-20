# SPDX-License-Identifier: Apache-2.0
# THE RANK-4 CONVOLUTION AND POOLING KERNELS OF THE TAPE (lane ag-rankn-conv,
# docs/plans/AUTOGRAD_RANKN.md section 2.8). One `seq_kernel` body a code,
# shared by HostExec and DeviceExec through sequence/dispatch.mojo, so the
# host column and every GPU run the same bodies.
#
# Each body is x_cnn's element function for that element (x_cnn/ops.mojo
# im2col_val, conv_out_val, dout_rows_at, col2im_at, _maxpool_fwd,
# maxpool_bwd_val, avgpool_fwd_at, avgpool_bwd_at), read against the same
# parameter words: the CP_* block of a convolution or the PP_* block of a
# pool, held in a descriptor buffer bound to Args.p11 (int32 bit patterns
# in tape storage, read with rn_word: never an int-cast pointer, never ld,
# whose ftz would zero a small integer's pattern). The max pool's winners
# are int32 patterns in a float buffer too (bitcast, as autograd/
# embedding_ops.mojo _ld_i). So each stored word is x_cnn's word for that
# element; the GEMM between the layout kernels is the tape's OP_GEMM.
#
# ORDER. Every body is parallel over its output words (one thread a word).
# The only sums are the gathers of col2im, the max pool's VJP and the avg
# pool's forward and VJP: each one output's taps in (kh, kw) ascending order
# from +0.0, x_cnn's order (the reference scatter-adds; the gather has one
# order on every vendor). No atomics, no two threads write one word. The
# tiling shortcut of the pools' VJPs (kernel == stride, no padding, no
# dilation) visits the one window the general loop would visit and no
# other: the same terms in the same order, so the same words.
#
# Index decoding divides in 32 bits as x_cnn does (DEVIATION 5720): every
# operand is a non-negative element index or dimension below 2^31 (the
# executor bounds every launch to Int32).
#
# Codes OP_RNV_FIRST..OP_RNV_LAST (4500..4519) are this family; 4508..4519
# are reserved for this lane's later bodies. sequence/dispatch.mojo routes
# and fences them.
from std.memory import bitcast
from sequence.ops import Args, FP
from checks.numerics import ftz, identical_div
from autograd.rankn import rn_word
from x_cnn.ops import (
    CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_SH, CP_SW, CP_PH, CP_PW, CP_DH, CP_DW,
    CP_OH, CP_OW, CP_BIAS,
    PP_H, PP_W, PP_KH, PP_KW, PP_SH, PP_SW, PP_PH, PP_PW, PP_DH, PP_DW, PP_OH, PP_OW,
    PP_INCPAD, PP_DIVOVR,
)

comptime OP_RNV_IM2COL = 4500
comptime OP_RNV_CONV_OUT = 4501
comptime OP_RNV_CONV_GROWS = 4502
comptime OP_RNV_COL2IM = 4503
comptime OP_RNV_MAXPOOL_FWD = 4504
comptime OP_RNV_MAXPOOL_VJP = 4505
comptime OP_RNV_AVGPOOL_FWD = 4506
comptime OP_RNV_AVGPOOL_VJP = 4507
comptime OP_RNV_FIRST = 4500
comptime OP_RNV_LAST = 4519

# The plan's names for the same codes (section 6, lane 5).
comptime OP_RN_IM2COL = OP_RNV_IM2COL
comptime OP_RN_CONV_OUT = OP_RNV_CONV_OUT
comptime OP_RN_CONV_GROWS = OP_RNV_CONV_GROWS
comptime OP_RN_COL2IM = OP_RNV_COL2IM
comptime OP_RN_MAXPOOL_FWD = OP_RNV_MAXPOOL_FWD
comptime OP_RN_MAXPOOL_VJP = OP_RNV_MAXPOOL_VJP
comptime OP_RN_AVGPOOL_FWD = OP_RNV_AVGPOOL_FWD
comptime OP_RN_AVGPOOL_VJP = OP_RNV_AVGPOOL_VJP


@always_inline
def _ud(a: Int, b: Int) -> Int:
    return Int(UInt32(a) // UInt32(b))


@always_inline
def _um(a: Int, b: Int) -> Int:
    return Int(UInt32(a) % UInt32(b))


@always_inline
def _canon(v: Float32) -> Float32:
    """x_cnn/ops.mojo canon: a NaN is stored as the one word 0x7FC00000."""
    if v != v:
        return bitcast[DType.float32](UInt32(0x7FC00000))
    return v


@always_inline
def _ld_i(p: FP, i: Int) -> Int:
    """The int32 held as bits in word i of p."""
    return Int(bitcast[DType.int32](p.unsafe_load(i)))


@always_inline
def _st_i(p: FP, i: Int, v: Int):
    """Word i of p holds int32 v as its bit pattern."""
    p.unsafe_store(i, bitcast[DType.float32](Int32(v)))


# ---------------------------------------------------------------- conv2d
@always_inline
def op_rnv_im2col(t: Int, a: Args):
    """cols[r, q] (r = (n OH + oh) OW + ow, q = (c KH + kh) KW + kw) = the
    input pixel under that tap, or +0.0 in the zero padding: x_cnn
    im2col_val. p0 x (NCHW), p1 cols; p11 the CP_* descriptor."""
    var d = a.p11
    var C = rn_word(d, CP_C); var H = rn_word(d, CP_H); var W = rn_word(d, CP_W)
    var KH = rn_word(d, CP_KH); var KW = rn_word(d, CP_KW)
    var OH = rn_word(d, CP_OH); var OW = rn_word(d, CP_OW)
    var ckk = C * KH * KW
    var r = _ud(t, ckk)
    var q = t - r * ckk
    var n = _ud(r, OH * OW)
    var rem = r - n * OH * OW
    var oh = _ud(rem, OW)
    var ow = rem - oh * OW
    var c = _ud(q, KH * KW)
    var k = q - c * KH * KW
    var kh = _ud(k, KW)
    var kw = k - kh * KW
    var h = oh * rn_word(d, CP_SH) - rn_word(d, CP_PH) + kh * rn_word(d, CP_DH)
    var w = ow * rn_word(d, CP_SW) - rn_word(d, CP_PW) + kw * rn_word(d, CP_DW)
    var v = Float32(0)
    if h >= 0 and h < H and w >= 0 and w < W:
        v = ftz(a.p0.unsafe_load(((n * C + c) * H + h) * W + w))
    a.p1.unsafe_store(t, v)


@always_inline
def op_rnv_conv_out(t: Int, a: Args):
    """out[n, oc, oh, ow] (NCHW) = y2[r, oc] (+ bias[oc] when CP_BIAS), one
    add: x_cnn conv_out_at / conv_out_val. p0 y2 [rows, OC], p1 bias (read
    only when CP_BIAS != 0), p2 out; p11 the CP_* descriptor."""
    var d = a.p11
    var OC = rn_word(d, CP_OC); var OH = rn_word(d, CP_OH); var OW = rn_word(d, CP_OW)
    var ow = _um(t, OW)
    var s = _ud(t, OW)
    var oh = _um(s, OH)
    s = _ud(s, OH)
    var oc = _um(s, OC)
    var n = _ud(s, OC)
    var r = (n * OH + oh) * OW + ow
    var v = ftz(a.p0.unsafe_load(r * OC + oc))
    if rn_word(d, CP_BIAS) != 0:
        v = ftz(v + ftz(a.p1.unsafe_load(oc)))
    a.p2.unsafe_store(t, _canon(v))


@always_inline
def op_rnv_conv_grows(t: Int, a: Args):
    """grow[r, oc] = dout[n, oc, oh, ow]: NCHW to the GEMM's row layout, a
    copy (x_cnn dout_rows_at). p0 dout, p1 grow; p11 the CP_* descriptor."""
    var d = a.p11
    var OC = rn_word(d, CP_OC); var OH = rn_word(d, CP_OH); var OW = rn_word(d, CP_OW)
    var r = _ud(t, OC)
    var oc = t - r * OC
    var n = _ud(r, OH * OW)
    var rem = r - n * OH * OW
    a.p1.unsafe_store(t, ftz(a.p0.unsafe_load((n * OC + oc) * OH * OW + rem)))


@always_inline
def op_rnv_col2im(t: Int, a: Args):
    """dx[n, c, h, w] = the sum of every dcols entry whose tap reads this
    pixel, gathered in (kh, kw) ascending order from +0.0 (x_cnn col2im_at,
    DEVIATION 5700). p0 dcols [rows, C KH KW], p1 dx; p11 the CP_*
    descriptor."""
    var d = a.p11
    var C = rn_word(d, CP_C); var H = rn_word(d, CP_H); var W = rn_word(d, CP_W)
    var KH = rn_word(d, CP_KH); var KW = rn_word(d, CP_KW)
    var OH = rn_word(d, CP_OH); var OW = rn_word(d, CP_OW)
    var SH = rn_word(d, CP_SH); var SW = rn_word(d, CP_SW)
    var PH = rn_word(d, CP_PH); var PW = rn_word(d, CP_PW)
    var DH = rn_word(d, CP_DH); var DW = rn_word(d, CP_DW)
    var w = _um(t, W)
    var s = _ud(t, W)
    var h = _um(s, H)
    s = _ud(s, H)
    var c = _um(s, C)
    var n = _ud(s, C)
    var ckk = C * KH * KW
    var acc = Float32(0)
    for kh in range(KH):
        var th = h + PH - kh * DH
        if th < 0 or _um(th, SH) != 0:
            continue
        var oh = _ud(th, SH)
        if oh >= OH:
            continue
        for kw in range(KW):
            var tw = w + PW - kw * DW
            if tw < 0 or _um(tw, SW) != 0:
                continue
            var ow = _ud(tw, SW)
            if ow >= OW:
                continue
            var r = (n * OH + oh) * OW + ow
            acc = ftz(acc + ftz(a.p0.unsafe_load(r * ckk + (c * KH + kh) * KW + kw)))
    a.p1.unsafe_store(t, acc)


# ---------------------------------------------------------------- pools
@always_inline
def op_rnv_maxpool_fwd(t: Int, a: Args):
    """out[n, c, oh, ow] = the max over the window, taps in (kh, kw)
    ascending order; the first maximum wins a tie, a NaN wins (x_cnn
    _maxpool_fwd, DEVIATION 5706). The winner's flat h W + w goes to p2 as
    an int32 pattern. p0 x, p1 out, p2 winners; p11 the PP_* descriptor."""
    var d = a.p11
    var H = rn_word(d, PP_H); var W = rn_word(d, PP_W)
    var KH = rn_word(d, PP_KH); var KW = rn_word(d, PP_KW)
    var OH = rn_word(d, PP_OH); var OW = rn_word(d, PP_OW)
    var SH = rn_word(d, PP_SH); var SW = rn_word(d, PP_SW)
    var PH = rn_word(d, PP_PH); var PW = rn_word(d, PP_PW)
    var DH = rn_word(d, PP_DH); var DW = rn_word(d, PP_DW)
    var ow = _um(t, OW)
    var s = _ud(t, OW)
    var oh = _um(s, OH)
    var nc = _ud(s, OH)
    var base = nc * H * W
    var best = Float32(0)
    var bi = -1
    for kh in range(KH):
        var h = oh * SH - PH + kh * DH
        if h < 0 or h >= H:
            continue
        for kw in range(KW):
            var w = ow * SW - PW + kw * DW
            if w < 0 or w >= W:
                continue
            var v = ftz(a.p0.unsafe_load(base + h * W + w))
            if bi < 0 or v > best or v != v:
                best = v
                bi = h * W + w
    a.p1.unsafe_store(t, best)
    _st_i(a.p2, t, bi)


@always_inline
def op_rnv_maxpool_vjp(t: Int, a: Args):
    """dx[n, c, h, w] = the sum of dout over every window whose winner is
    this pixel, gathered in (kh, kw) ascending order from +0.0 (x_cnn
    maxpool_bwd_val). p0 dout, p1 winners, p2 dx; p11 the PP_* descriptor."""
    var d = a.p11
    var H = rn_word(d, PP_H); var W = rn_word(d, PP_W)
    var KH = rn_word(d, PP_KH); var KW = rn_word(d, PP_KW)
    var OH = rn_word(d, PP_OH); var OW = rn_word(d, PP_OW)
    var SH = rn_word(d, PP_SH); var SW = rn_word(d, PP_SW)
    var PH = rn_word(d, PP_PH); var PW = rn_word(d, PP_PW)
    var DH = rn_word(d, PP_DH); var DW = rn_word(d, PP_DW)
    var w = _um(t, W)
    var s = _ud(t, W)
    var h = _um(s, H)
    var nc = _ud(s, H)
    var me = h * W + w
    var acc = Float32(0)
    if KH == SH and KW == SW and PH == 0 and PW == 0 and DH == 1 and DW == 1:
        # tiling windows: the one window holding this pixel (the loop below
        # visits it and no other)
        var oh1 = _ud(h, KH)
        var ow1 = _ud(w, KW)
        if oh1 < OH and ow1 < OW:
            var o1 = (nc * OH + oh1) * OW + ow1
            if _ld_i(a.p1, o1) == me:
                acc = ftz(acc + ftz(a.p0.unsafe_load(o1)))
        a.p2.unsafe_store(t, acc)
        return
    for kh in range(KH):
        var th = h + PH - kh * DH
        if th < 0 or _um(th, SH) != 0:
            continue
        var oh = _ud(th, SH)
        if oh >= OH:
            continue
        for kw in range(KW):
            var tw = w + PW - kw * DW
            if tw < 0 or _um(tw, SW) != 0:
                continue
            var ow = _ud(tw, SW)
            if ow >= OW:
                continue
            var o = (nc * OH + oh) * OW + ow
            if _ld_i(a.p1, o) == me:
                acc = ftz(acc + ftz(a.p0.unsafe_load(o)))
    a.p2.unsafe_store(t, acc)


@always_inline
def _avg_divisor(oh: Int, ow: Int, d: FP) -> Int:
    """x_cnn _avg_divisor: the window clipped to the padded input, or to the
    input itself when count_include_pad is 0; divisor_override when set."""
    if rn_word(d, PP_DIVOVR) > 0:
        return rn_word(d, PP_DIVOVR)
    var H = rn_word(d, PP_H); var W = rn_word(d, PP_W)
    var PH = rn_word(d, PP_PH); var PW = rn_word(d, PP_PW)
    var hs = oh * rn_word(d, PP_SH) - PH
    var ws = ow * rn_word(d, PP_SW) - PW
    var he = min(hs + rn_word(d, PP_KH), H + PH)
    var we = min(ws + rn_word(d, PP_KW), W + PW)
    if rn_word(d, PP_INCPAD) != 0:
        return (he - hs) * (we - ws)
    return (min(he, H) - max(hs, 0)) * (min(we, W) - max(ws, 0))


@always_inline
def op_rnv_avgpool_fwd(t: Int, a: Args):
    """out = (the window's sum in (kh, kw) ascending order from +0.0) /
    divisor, one correctly rounded division (x_cnn avgpool_fwd_at,
    DEVIATION 5707). p0 x, p1 out; p11 the PP_* descriptor."""
    var d = a.p11
    var H = rn_word(d, PP_H); var W = rn_word(d, PP_W)
    var KH = rn_word(d, PP_KH); var KW = rn_word(d, PP_KW)
    var OH = rn_word(d, PP_OH); var OW = rn_word(d, PP_OW)
    var SH = rn_word(d, PP_SH); var SW = rn_word(d, PP_SW)
    var PH = rn_word(d, PP_PH); var PW = rn_word(d, PP_PW)
    var ow = _um(t, OW)
    var s = _ud(t, OW)
    var oh = _um(s, OH)
    var nc = _ud(s, OH)
    var base = nc * H * W
    var acc = Float32(0)
    for kh in range(KH):
        var h = oh * SH - PH + kh
        if h < 0 or h >= H:
            continue
        for kw in range(KW):
            var w = ow * SW - PW + kw
            if w < 0 or w >= W:
                continue
            acc = ftz(acc + ftz(a.p0.unsafe_load(base + h * W + w)))
    a.p1.unsafe_store(t, ftz(identical_div(acc, Float32(_avg_divisor(oh, ow, d)))))


@always_inline
def op_rnv_avgpool_vjp(t: Int, a: Args):
    """dx[n, c, h, w] = the sum over every window holding this pixel of
    dout / divisor, gathered in (kh, kw) ascending order from +0.0 (x_cnn
    avgpool_bwd_at). p0 dout, p1 dx; p11 the PP_* descriptor."""
    var d = a.p11
    var H = rn_word(d, PP_H); var W = rn_word(d, PP_W)
    var KH = rn_word(d, PP_KH); var KW = rn_word(d, PP_KW)
    var OH = rn_word(d, PP_OH); var OW = rn_word(d, PP_OW)
    var SH = rn_word(d, PP_SH); var SW = rn_word(d, PP_SW)
    var PH = rn_word(d, PP_PH); var PW = rn_word(d, PP_PW)
    var w = _um(t, W)
    var s = _ud(t, W)
    var h = _um(s, H)
    var nc = _ud(s, H)
    var acc = Float32(0)
    if KH == SH and KW == SW and PH == 0 and PW == 0:
        # tiling windows: the one window holding this pixel
        var oh1 = _ud(h, KH)
        var ow1 = _ud(w, KW)
        if oh1 < OH and ow1 < OW:
            var o1 = (nc * OH + oh1) * OW + ow1
            var g1 = ftz(identical_div(ftz(a.p0.unsafe_load(o1)), Float32(_avg_divisor(oh1, ow1, d))))
            acc = ftz(acc + g1)
        a.p1.unsafe_store(t, acc)
        return
    for kh in range(KH):
        var th = h + PH - kh
        if th < 0 or _um(th, SH) != 0:
            continue
        var oh = _ud(th, SH)
        if oh >= OH:
            continue
        for kw in range(KW):
            var tw = w + PW - kw
            if tw < 0 or _um(tw, SW) != 0:
                continue
            var ow = _ud(tw, SW)
            if ow >= OW:
                continue
            var o = (nc * OH + oh) * OW + ow
            var g = ftz(identical_div(ftz(a.p0.unsafe_load(o)), Float32(_avg_divisor(oh, ow, d))))
            acc = ftz(acc + g)
    a.p1.unsafe_store(t, acc)


@always_inline
def apply_rankn_conv[OP: Int](t: Int, a: Args):
    comptime if OP == OP_RNV_IM2COL:
        op_rnv_im2col(t, a)
    elif OP == OP_RNV_CONV_OUT:
        op_rnv_conv_out(t, a)
    elif OP == OP_RNV_CONV_GROWS:
        op_rnv_conv_grows(t, a)
    elif OP == OP_RNV_COL2IM:
        op_rnv_col2im(t, a)
    elif OP == OP_RNV_MAXPOOL_FWD:
        op_rnv_maxpool_fwd(t, a)
    elif OP == OP_RNV_MAXPOOL_VJP:
        op_rnv_maxpool_vjp(t, a)
    elif OP == OP_RNV_AVGPOOL_FWD:
        op_rnv_avgpool_fwd(t, a)
    elif OP == OP_RNV_AVGPOOL_VJP:
        op_rnv_avgpool_vjp(t, a)
