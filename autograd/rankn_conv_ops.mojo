# SPDX-License-Identifier: Apache-2.0
# THE RANK-4 CONVOLUTION AND POOLING KERNELS OF THE TAPE (lane ag-rankn-conv,
# plan AUTOGRAD_RANKN (git history) section 2.8). One `seq_kernel` body a code,
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
# Codes OP_RNV_FIRST..OP_RNV_LAST (4500..4519) are this family; 4508..4517
# are the batch norm of Sequential (lane seq-blocks-3, below), 4518..4519
# reserved. sequence/dispatch.mojo routes and fences them.
from std.memory import bitcast
from std.sys.compile import is_defined
from sequence.ops import Args, FP
from checks.numerics import ftz, identical_div, identical_rsqrt
from autograd.rankn import rn_word
from x_cnn.ops import FP as XFP, IP as XIP
from x_cnn.ops import (
    BN_GAMMA, BN_BETA, BN_SUMG, BN_SUMGX,
    bn_blk_sum_at, bn_blk_mean_at, bn_blk_sq_at, bn_blk_var_at, bn_blk_var_running_at, bn_eval_stats_at, bn_apply_at,
    bn_blk_red_at, bn_blk_red_fin_at, bn_bwd_dx_at, bn_bwd_eval_dx_at,
)
from x_cnn.ops import (
    CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_SH, CP_SW, CP_PH, CP_PW, CP_DH, CP_DW,
    CP_OH, CP_OW, CP_BIAS, CP_LEN,
    PP_H, PP_W, PP_KH, PP_KW, PP_SH, PP_SW, PP_PH, PP_PW, PP_DH, PP_DW, PP_OH, PP_OW,
    PP_INCPAD, PP_DIVOVR, PP_LEN,
)

comptime OP_RNV_IM2COL = 4500
comptime OP_RNV_CONV_OUT = 4501
comptime OP_RNV_CONV_GROWS = 4502
comptime OP_RNV_COL2IM = 4503
comptime OP_RNV_MAXPOOL_FWD = 4504
comptime OP_RNV_MAXPOOL_VJP = 4505
comptime OP_RNV_AVGPOOL_FWD = 4506
comptime OP_RNV_AVGPOOL_VJP = 4507
comptime OP_RNV_BN_PREP = 4508
comptime OP_RNV_BN_SUM = 4509
comptime OP_RNV_BN_MEAN = 4510
comptime OP_RNV_BN_SQ = 4511
comptime OP_RNV_BN_VAR = 4512
comptime OP_RNV_BN_APPLY = 4513
comptime OP_RNV_BN_RED = 4514
comptime OP_RNV_BN_RED_FIN = 4515
comptime OP_RNV_BN_DX = 4516
comptime OP_RNV_BN_DX_EVAL = 4517
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


# ---------------------------------------------------------------- the depth axis (lane nn-conv3d, 2026-11-23)
# Every descriptor the conv and pool bodies below read is x_cnn's block (CP_LEN words for a convolution,
# PP_LEN for a pool) followed by RN_DEPTH_LEN depth words [D, KD, SD, PD, DD, OD]: the input depth, the
# window's depth, its step, its padding (both ends), its dilation and the output depth. A 1-D or 2-D node
# writes [1, 1, 1, 0, 1, 1] (autograd/rankn_conv.mojo rn_flat_depth): the depth loop runs once with z = 0,
# the decodes skip their division by 1 (_uq), so a 2-D node's words are the same terms in the same order
# as before the depth axis (the same bits). conv3d, max_pool3d and avg_pool3d are these same bodies with a
# depth (AGENTS.md "One body per operation": there is no separate 3-D implementation).
comptime RN_DEPTH_D = 0
comptime RN_DEPTH_KD = 1
comptime RN_DEPTH_SD = 2
comptime RN_DEPTH_PD = 3
comptime RN_DEPTH_DD = 4
comptime RN_DEPTH_OD = 5
comptime RN_DEPTH_LEN = 6

# SABOTAGE ARMS (lane nn-conv3d), test builds only, never defaults: each perturbs the depth axis of one family so
# checks/native_tests/nn_conv3d.mojo is shown to catch it; a node without depth (KD = 1) is untouched.
# CONV: im2col reads the depth taps in reverse. POOL: the max pool skips its last depth tap and the average
# pool's divisor leaves out the depth extent.
comptime NN_CONV3D_SABOTAGE_CONV = is_defined["MOJOLEARN_NN_CONV3D_SABOTAGE_CONV"]()
comptime NN_CONV3D_SABOTAGE_POOL = is_defined["MOJOLEARN_NN_CONV3D_SABOTAGE_POOL"]()


@always_inline
def _uq(a: Int, b: Int) -> Int:
    """a // b, with no division when b is 1 (the depth extents of a flat node)."""
    if b == 1:
        return a
    return _ud(a, b)


@always_inline
def _umq(a: Int, b: Int) -> Int:
    """a % b, with no division when b is 1."""
    if b == 1:
        return 0
    return _um(a, b)


# ---------------------------------------------------------------- conv2d / conv3d
@always_inline
def op_rnv_im2col(t: Int, a: Args):
    """cols[r, q] (r = ((n OD + od) OH + oh) OW + ow, q = ((c KD + kd) KH + kh) KW + kw) = the
    input voxel under that tap, or +0.0 in the zero padding: x_cnn im2col_val with the depth axis.
    p0 x (NCDHW; NCHW when D = 1), p1 cols; p11 the CP_* descriptor and its depth words."""
    var d = a.p11
    var C = rn_word(d, CP_C); var H = rn_word(d, CP_H); var W = rn_word(d, CP_W)
    var KH = rn_word(d, CP_KH); var KW = rn_word(d, CP_KW)
    var OH = rn_word(d, CP_OH); var OW = rn_word(d, CP_OW)
    var D = rn_word(d, CP_LEN + RN_DEPTH_D); var KD = rn_word(d, CP_LEN + RN_DEPTH_KD)
    var OD = rn_word(d, CP_LEN + RN_DEPTH_OD)
    var kk = KH * KW
    var ckk = C * KD * kk
    var r = _ud(t, ckk)
    var q = t - r * ckk
    var sp = OH * OW
    var nd = _ud(r, sp)
    var rem = r - nd * sp
    var oh = _ud(rem, OW)
    var ow = rem - oh * OW
    var n = _uq(nd, OD)
    var od = nd - n * OD
    var cd = _ud(q, kk)
    var k = q - cd * kk
    var kh = _ud(k, KW)
    var kw = k - kh * KW
    var c = _uq(cd, KD)
    var kd = cd - c * KD
    comptime if NN_CONV3D_SABOTAGE_CONV:
        kd = KD - 1 - kd
    var z = od * rn_word(d, CP_LEN + RN_DEPTH_SD) - rn_word(d, CP_LEN + RN_DEPTH_PD) + kd * rn_word(d, CP_LEN + RN_DEPTH_DD)
    var h = oh * rn_word(d, CP_SH) - rn_word(d, CP_PH) + kh * rn_word(d, CP_DH)
    var w = ow * rn_word(d, CP_SW) - rn_word(d, CP_PW) + kw * rn_word(d, CP_DW)
    var v = Float32(0)
    if z >= 0 and z < D and h >= 0 and h < H and w >= 0 and w < W:
        v = ftz(a.p0.unsafe_load((((n * C + c) * D + z) * H + h) * W + w))
    a.p1.unsafe_store(t, v)


@always_inline
def op_rnv_conv_out(t: Int, a: Args):
    """out[n, oc, s] (s the flat output position, OD OH OW of them) = y2[n S + s, oc] (+ bias[oc] when
    CP_BIAS), one add: x_cnn conv_out_at / conv_out_val. p0 y2 [rows, OC], p1 bias (read only when
    CP_BIAS != 0), p2 out; p11 the CP_* descriptor and its depth words."""
    var d = a.p11
    var OC = rn_word(d, CP_OC)
    var S = rn_word(d, CP_LEN + RN_DEPTH_OD) * rn_word(d, CP_OH) * rn_word(d, CP_OW)
    var sp = _um(t, S)
    var s = _ud(t, S)
    var oc = _um(s, OC)
    var n = _ud(s, OC)
    var r = n * S + sp
    var v = ftz(a.p0.unsafe_load(r * OC + oc))
    var with_bias = rn_word(d, CP_BIAS) != 0
    comptime if SEQ_BLOCKS_SABOTAGE_CONV:
        with_bias = False
    if with_bias:
        v = ftz(v + ftz(a.p1.unsafe_load(oc)))
    a.p2.unsafe_store(t, _canon(v))


@always_inline
def op_rnv_conv_grows(t: Int, a: Args):
    """grow[r, oc] = dout[n, oc, s] (r = n S + s): the output layout to the GEMM's rows, a copy
    (x_cnn dout_rows_at). p0 dout, p1 grow; p11 the CP_* descriptor and its depth words."""
    var d = a.p11
    var OC = rn_word(d, CP_OC)
    var S = rn_word(d, CP_LEN + RN_DEPTH_OD) * rn_word(d, CP_OH) * rn_word(d, CP_OW)
    var r = _ud(t, OC)
    var oc = t - r * OC
    var n = _ud(r, S)
    var rem = r - n * S
    a.p1.unsafe_store(t, ftz(a.p0.unsafe_load((n * OC + oc) * S + rem)))


@always_inline
def op_rnv_col2im(t: Int, a: Args):
    """dx[n, c, z, h, w] = the sum of every dcols entry whose tap reads this voxel, gathered in
    (kd, kh, kw) ascending order from +0.0 (x_cnn col2im_at, DEVIATION 5700, with the depth axis).
    p0 dcols [rows, C KD KH KW], p1 dx; p11 the CP_* descriptor and its depth words."""
    var d = a.p11
    var C = rn_word(d, CP_C); var H = rn_word(d, CP_H); var W = rn_word(d, CP_W)
    var KH = rn_word(d, CP_KH); var KW = rn_word(d, CP_KW)
    var OH = rn_word(d, CP_OH); var OW = rn_word(d, CP_OW)
    var SH = rn_word(d, CP_SH); var SW = rn_word(d, CP_SW)
    var PH = rn_word(d, CP_PH); var PW = rn_word(d, CP_PW)
    var DH = rn_word(d, CP_DH); var DW = rn_word(d, CP_DW)
    var D = rn_word(d, CP_LEN + RN_DEPTH_D); var KD = rn_word(d, CP_LEN + RN_DEPTH_KD)
    var SD = rn_word(d, CP_LEN + RN_DEPTH_SD); var PD = rn_word(d, CP_LEN + RN_DEPTH_PD)
    var DD = rn_word(d, CP_LEN + RN_DEPTH_DD); var OD = rn_word(d, CP_LEN + RN_DEPTH_OD)
    var w = _um(t, W)
    var s = _ud(t, W)
    var h = _um(s, H)
    s = _ud(s, H)
    var z = _umq(s, D)
    s = _uq(s, D)
    var c = _um(s, C)
    var n = _ud(s, C)
    var ckk = C * KD * KH * KW
    var acc = Float32(0)
    for kd in range(KD):
        var tz = z + PD - kd * DD
        if tz < 0 or _umq(tz, SD) != 0:
            continue
        var od = _uq(tz, SD)
        if od >= OD:
            continue
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
                var r = ((n * OD + od) * OH + oh) * OW + ow
                acc = ftz(acc + ftz(a.p0.unsafe_load(r * ckk + ((c * KD + kd) * KH + kh) * KW + kw)))
    a.p1.unsafe_store(t, acc)


# ---------------------------------------------------------------- pools (2-D and 3-D)
@always_inline
def op_rnv_maxpool_fwd(t: Int, a: Args):
    """out[n, c, od, oh, ow] = the max over the window, taps in (kd, kh, kw) ascending order; the
    first maximum wins a tie, a NaN wins (x_cnn _maxpool_fwd, DEVIATION 5706). The winner's flat
    (z H + h) W + w goes to p2 as an int32 pattern. p0 x, p1 out, p2 winners; p11 the PP_*
    descriptor and its depth words."""
    var d = a.p11
    var H = rn_word(d, PP_H); var W = rn_word(d, PP_W)
    var KH = rn_word(d, PP_KH); var KW = rn_word(d, PP_KW)
    var OH = rn_word(d, PP_OH); var OW = rn_word(d, PP_OW)
    var SH = rn_word(d, PP_SH); var SW = rn_word(d, PP_SW)
    var PH = rn_word(d, PP_PH); var PW = rn_word(d, PP_PW)
    var DH = rn_word(d, PP_DH); var DW = rn_word(d, PP_DW)
    var D = rn_word(d, PP_LEN + RN_DEPTH_D); var KD = rn_word(d, PP_LEN + RN_DEPTH_KD)
    var SD = rn_word(d, PP_LEN + RN_DEPTH_SD); var PD = rn_word(d, PP_LEN + RN_DEPTH_PD)
    var DD = rn_word(d, PP_LEN + RN_DEPTH_DD); var OD = rn_word(d, PP_LEN + RN_DEPTH_OD)
    var ow = _um(t, OW)
    var s = _ud(t, OW)
    var oh = _um(s, OH)
    s = _ud(s, OH)
    var nc = _uq(s, OD)
    var od = s - nc * OD
    var base = nc * D * H * W
    var best = Float32(0)
    var bi = -1
    var kd_end = KD
    comptime if NN_CONV3D_SABOTAGE_POOL:
        if KD > 1:
            kd_end = KD - 1
    for kd in range(kd_end):
        var z = od * SD - PD + kd * DD
        if z < 0 or z >= D:
            continue
        for kh in range(KH):
            var h = oh * SH - PH + kh * DH
            if h < 0 or h >= H:
                continue
            for kw in range(KW):
                var w = ow * SW - PW + kw * DW
                if w < 0 or w >= W:
                    continue
                var v = ftz(a.p0.unsafe_load(base + (z * H + h) * W + w))
                if bi < 0 or v > best or v != v:
                    best = v
                    bi = (z * H + h) * W + w
    a.p1.unsafe_store(t, best)
    _st_i(a.p2, t, bi)


@always_inline
def op_rnv_maxpool_vjp(t: Int, a: Args):
    """dx[n, c, z, h, w] = the sum of dout over every window whose winner is this voxel, gathered
    in (kd, kh, kw) ascending order from +0.0 (x_cnn maxpool_bwd_val). p0 dout, p1 winners, p2 dx;
    p11 the PP_* descriptor and its depth words."""
    var d = a.p11
    var H = rn_word(d, PP_H); var W = rn_word(d, PP_W)
    var KH = rn_word(d, PP_KH); var KW = rn_word(d, PP_KW)
    var OH = rn_word(d, PP_OH); var OW = rn_word(d, PP_OW)
    var SH = rn_word(d, PP_SH); var SW = rn_word(d, PP_SW)
    var PH = rn_word(d, PP_PH); var PW = rn_word(d, PP_PW)
    var DH = rn_word(d, PP_DH); var DW = rn_word(d, PP_DW)
    var D = rn_word(d, PP_LEN + RN_DEPTH_D); var KD = rn_word(d, PP_LEN + RN_DEPTH_KD)
    var SD = rn_word(d, PP_LEN + RN_DEPTH_SD); var PD = rn_word(d, PP_LEN + RN_DEPTH_PD)
    var DD = rn_word(d, PP_LEN + RN_DEPTH_DD); var OD = rn_word(d, PP_LEN + RN_DEPTH_OD)
    var w = _um(t, W)
    var s = _ud(t, W)
    var h = _um(s, H)
    s = _ud(s, H)
    var nc = _uq(s, D)
    var z = s - nc * D
    var me = (z * H + h) * W + w
    var acc = Float32(0)
    if (KH == SH and KW == SW and KD == SD and PH == 0 and PW == 0 and PD == 0
            and DH == 1 and DW == 1 and DD == 1):
        # tiling windows: the one window holding this voxel (the loop below
        # visits it and no other)
        var od1 = _uq(z, KD)
        var oh1 = _ud(h, KH)
        var ow1 = _ud(w, KW)
        if od1 < OD and oh1 < OH and ow1 < OW:
            var o1 = ((nc * OD + od1) * OH + oh1) * OW + ow1
            if _ld_i(a.p1, o1) == me:
                acc = ftz(acc + ftz(a.p0.unsafe_load(o1)))
        a.p2.unsafe_store(t, acc)
        return
    for kd in range(KD):
        var tz = z + PD - kd * DD
        if tz < 0 or _umq(tz, SD) != 0:
            continue
        var od = _uq(tz, SD)
        if od >= OD:
            continue
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
                var o = ((nc * OD + od) * OH + oh) * OW + ow
                if _ld_i(a.p1, o) == me:
                    acc = ftz(acc + ftz(a.p0.unsafe_load(o)))
    a.p2.unsafe_store(t, acc)


@always_inline
def _avg_divisor(od: Int, oh: Int, ow: Int, d: FP) -> Int:
    """x_cnn _avg_divisor with the depth axis: the window clipped to the padded input, or to the
    input itself when count_include_pad is 0; divisor_override when set."""
    if rn_word(d, PP_DIVOVR) > 0:
        return rn_word(d, PP_DIVOVR)
    var H = rn_word(d, PP_H); var W = rn_word(d, PP_W)
    var PH = rn_word(d, PP_PH); var PW = rn_word(d, PP_PW)
    var D = rn_word(d, PP_LEN + RN_DEPTH_D); var PD = rn_word(d, PP_LEN + RN_DEPTH_PD)
    var ds = od * rn_word(d, PP_LEN + RN_DEPTH_SD) - PD
    var hs = oh * rn_word(d, PP_SH) - PH
    var ws = ow * rn_word(d, PP_SW) - PW
    var de = min(ds + rn_word(d, PP_LEN + RN_DEPTH_KD), D + PD)
    var he = min(hs + rn_word(d, PP_KH), H + PH)
    var we = min(ws + rn_word(d, PP_KW), W + PW)
    var depth: Int
    var plane: Int
    if rn_word(d, PP_INCPAD) != 0:
        depth = de - ds
        plane = (he - hs) * (we - ws)
    else:
        depth = min(de, D) - max(ds, 0)
        plane = (min(he, H) - max(hs, 0)) * (min(we, W) - max(ws, 0))
    comptime if NN_CONV3D_SABOTAGE_POOL:
        if rn_word(d, PP_LEN + RN_DEPTH_KD) > 1:
            depth = 1
    return depth * plane


@always_inline
def op_rnv_avgpool_fwd(t: Int, a: Args):
    """out = (the window's sum in (kd, kh, kw) ascending order from +0.0) / divisor, one correctly
    rounded division (x_cnn avgpool_fwd_at, DEVIATION 5707). p0 x, p1 out; p11 the PP_* descriptor
    and its depth words."""
    var d = a.p11
    var H = rn_word(d, PP_H); var W = rn_word(d, PP_W)
    var KH = rn_word(d, PP_KH); var KW = rn_word(d, PP_KW)
    var OH = rn_word(d, PP_OH); var OW = rn_word(d, PP_OW)
    var SH = rn_word(d, PP_SH); var SW = rn_word(d, PP_SW)
    var PH = rn_word(d, PP_PH); var PW = rn_word(d, PP_PW)
    var D = rn_word(d, PP_LEN + RN_DEPTH_D); var KD = rn_word(d, PP_LEN + RN_DEPTH_KD)
    var SD = rn_word(d, PP_LEN + RN_DEPTH_SD); var PD = rn_word(d, PP_LEN + RN_DEPTH_PD)
    var OD = rn_word(d, PP_LEN + RN_DEPTH_OD)
    var ow = _um(t, OW)
    var s = _ud(t, OW)
    var oh = _um(s, OH)
    s = _ud(s, OH)
    var nc = _uq(s, OD)
    var od = s - nc * OD
    var base = nc * D * H * W
    var acc = Float32(0)
    for kd in range(KD):
        var z = od * SD - PD + kd
        if z < 0 or z >= D:
            continue
        for kh in range(KH):
            var h = oh * SH - PH + kh
            if h < 0 or h >= H:
                continue
            for kw in range(KW):
                var w = ow * SW - PW + kw
                if w < 0 or w >= W:
                    continue
                acc = ftz(acc + ftz(a.p0.unsafe_load(base + (z * H + h) * W + w)))
    a.p1.unsafe_store(t, ftz(identical_div(acc, Float32(_avg_divisor(od, oh, ow, d)))))


@always_inline
def op_rnv_avgpool_vjp(t: Int, a: Args):
    """dx[n, c, z, h, w] = the sum over every window holding this voxel of dout / divisor, gathered
    in (kd, kh, kw) ascending order from +0.0 (x_cnn avgpool_bwd_at). p0 dout, p1 dx; p11 the PP_*
    descriptor and its depth words."""
    var d = a.p11
    var H = rn_word(d, PP_H); var W = rn_word(d, PP_W)
    var KH = rn_word(d, PP_KH); var KW = rn_word(d, PP_KW)
    var OH = rn_word(d, PP_OH); var OW = rn_word(d, PP_OW)
    var SH = rn_word(d, PP_SH); var SW = rn_word(d, PP_SW)
    var PH = rn_word(d, PP_PH); var PW = rn_word(d, PP_PW)
    var D = rn_word(d, PP_LEN + RN_DEPTH_D); var KD = rn_word(d, PP_LEN + RN_DEPTH_KD)
    var SD = rn_word(d, PP_LEN + RN_DEPTH_SD); var PD = rn_word(d, PP_LEN + RN_DEPTH_PD)
    var OD = rn_word(d, PP_LEN + RN_DEPTH_OD)
    var w = _um(t, W)
    var s = _ud(t, W)
    var h = _um(s, H)
    s = _ud(s, H)
    var nc = _uq(s, D)
    var z = s - nc * D
    var acc = Float32(0)
    if KH == SH and KW == SW and KD == SD and PH == 0 and PW == 0 and PD == 0:
        # tiling windows: the one window holding this voxel
        var od1 = _uq(z, KD)
        var oh1 = _ud(h, KH)
        var ow1 = _ud(w, KW)
        if od1 < OD and oh1 < OH and ow1 < OW:
            var o1 = ((nc * OD + od1) * OH + oh1) * OW + ow1
            var g1 = ftz(identical_div(ftz(a.p0.unsafe_load(o1)), Float32(_avg_divisor(od1, oh1, ow1, d))))
            acc = ftz(acc + g1)
        a.p1.unsafe_store(t, acc)
        return
    for kd in range(KD):
        var tz = z + PD - kd
        if tz < 0 or _umq(tz, SD) != 0:
            continue
        var od = _uq(tz, SD)
        if od >= OD:
            continue
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
                var o = ((nc * OD + od) * OH + oh) * OW + ow
                var g = ftz(identical_div(ftz(a.p0.unsafe_load(o)), Float32(_avg_divisor(od, oh, ow, d))))
                acc = ftz(acc + g)
    a.p1.unsafe_store(t, acc)


# ---------------------------------------------------------------- batch norm (lane seq-blocks-3, 2026-11-22)
# Sequential's batch_norm1d / batch_norm2d over (N, C, HW) samples. Every body below is x_cnn's own
# element function (x_cnn/ops.mojo bn_blk_sum_at .. bn_bwd_eval_dx_at), called as it is: the same
# math, one body (AGENTS.md "One body per operation"). x_cnn reads its [N, C, HW] parameter words
# through an int32 pointer; here they are the tape descriptor bound to p11 (int32 bit patterns,
# Tape.descriptor), read through a bitcast of that pointer (never an integer-made pointer). The
# statistics are x_cnn's BLOCKED folds (bn_fold_block: blocks whose size depends on N * HW alone),
# so the order is the same on every vendor and the host. x_cnn's own IDENTICAL default for its
# BatchNorm layers is the grid-stats form (bn_gs_*, a shared-memory tree in x_cnn/device.mojo);
# the tape executor runs plain element launches, so it takes the blocked bodies.
# aux: [eps, momentum] then seven per-channel arrays (x_cnn's BN_* slots).


# SABOTAGE ARMS (lane seq-blocks-3), test builds only, never defaults: each perturbs one block family so its
# check is shown to catch it (checks/native_tests/sequence_blocks.mojo, sequential_lengths.mojo). BN: the
# unbiased variance normalizes. CONV: the conv output without its bias. (MASK: sequence/ops.mojo op_cell_fwd.)
comptime SEQ_BLOCKS_SABOTAGE_BN = is_defined["MOJOLEARN_SEQ_BLOCKS_SABOTAGE_BN"]()
comptime SEQ_BLOCKS_SABOTAGE_CONV = is_defined["MOJOLEARN_SEQ_BLOCKS_SABOTAGE_CONV"]()


@always_inline
def _xf(p: FP) -> XFP:
    return p.unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _xd(p: FP) -> XIP:
    return p.bitcast[Int32]().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def op_rnv_bn_prep(c: Int, a: Args):
    """Channel c: aux's eps (f0) and momentum (f1) words, gamma and beta (p0, p4) into their
    slots; i0 = 1 (eval): the running statistics (p2 = [mean C | var C]) as this call's
    statistics (bn_eval_stats_at). p1 aux; p11 the [N, C, HW] descriptor. Every channel's
    thread stores the same two scalar words."""
    var C = rn_word(a.p11, 1)
    a.p1.unsafe_store(0, a.f0)
    a.p1.unsafe_store(1, a.f1)
    if a.i1 != 0:
        # no affine (lane nn-norms: the group and instance norms' statistics): gamma 1, beta 0
        a.p1.unsafe_store(2 + BN_GAMMA * C + c, Float32(1))
        a.p1.unsafe_store(2 + BN_BETA * C + c, Float32(0))
    else:
        a.p1.unsafe_store(2 + BN_GAMMA * C + c, ftz(a.p0.unsafe_load(c)))
        a.p1.unsafe_store(2 + BN_BETA * C + c, ftz(a.p4.unsafe_load(c)))
    if a.i0 != 0:
        var x = _xf(a.p2)
        bn_eval_stats_at(c, x, _xf(a.p1), x, x, _xd(a.p11), _xd(a.p11))


@always_inline
def op_rnv_bn_sum(t: Int, a: Args):
    """Block t's sum of x (p0) into part (p1): bn_blk_sum_at."""
    var x = _xf(a.p0)
    bn_blk_sum_at(t, x, _xf(a.p1), x, x, _xd(a.p11), _xd(a.p11))


@always_inline
def op_rnv_bn_mean(c: Int, a: Args):
    """Channel c's mean from part (p0) into aux (p1): bn_blk_mean_at."""
    var x = _xf(a.p0)
    bn_blk_mean_at(c, x, _xf(a.p1), x, x, _xd(a.p11), _xd(a.p11))


@always_inline
def op_rnv_bn_sq(t: Int, a: Args):
    """Block t's sum of (x - mean)^2: x p0, part p1, aux p2 (bn_blk_sq_at)."""
    var x = _xf(a.p0)
    bn_blk_sq_at(t, x, _xf(a.p1), _xf(a.p2), x, _xd(a.p11), _xd(a.p11))


@always_inline
def op_rnv_bn_var(c: Int, a: Args):
    """Channel c's biased variance and invstd, then PyTorch's running update of p2 in place
    (bn_blk_var_running_at): part p0, aux p1, running p2. i0 = 1: no running statistics (lane nn-norms:
    the group norm and an untracked instance norm), bn_blk_var_at alone."""
    var x = _xf(a.p0)
    if a.i0 != 0:
        bn_blk_var_at(c, x, _xf(a.p1), x, x, _xd(a.p11), _xd(a.p11))
    else:
        bn_blk_var_running_at(c, x, _xf(a.p1), _xf(a.p2), x, _xd(a.p11), _xd(a.p11))
    comptime if SEQ_BLOCKS_SABOTAGE_BN:
        # the unbiased variance normalizes (PyTorch normalizes with the biased one)
        var C = rn_word(a.p11, 1)
        var count = Float32(rn_word(a.p11, 0) * rn_word(a.p11, 2))
        var vu = a.p1.unsafe_load(2 + C + c) * count / (count - Float32(1))
        a.p1.unsafe_store(2 + 2 * C + c, ftz(identical_rsqrt(ftz(vu + a.p1.unsafe_load(0)))))


@always_inline
def op_rnv_bn_apply(i: Int, a: Args):
    """y = ((x - mean) invstd) gamma + beta: x p0, aux p1, y p2 (bn_apply_at)."""
    var x = _xf(a.p0)
    bn_apply_at(i, x, _xf(a.p1), _xf(a.p2), x, _xd(a.p11), _xd(a.p11))


@always_inline
def op_rnv_bn_red(t: Int, a: Args):
    """Block t's sum of g and of g xhat: x p0, g p1, aux p2, part p3 (bn_blk_red_at)."""
    bn_blk_red_at(t, _xf(a.p0), _xf(a.p1), _xf(a.p2), _xf(a.p3), _xd(a.p11), _xd(a.p11))


@always_inline
def op_rnv_bn_red_fin(c: Int, a: Args):
    """Channel c's sum_g and sum_gx (bn_blk_red_fin_at: part p0, aux p1), then the parameter
    gradients d gamma = sum_gx into p2 (i0 = 1) and d beta = sum_g into p3 (i1 = 1)."""
    var x = _xf(a.p0)
    bn_blk_red_fin_at(c, x, _xf(a.p1), x, x, _xd(a.p11), _xd(a.p11))
    var C = rn_word(a.p11, 1)
    if a.i0 != 0:
        a.p2.unsafe_store(c, a.p1.unsafe_load(2 + BN_SUMGX * C + c))
    if a.i1 != 0:
        a.p3.unsafe_store(c, a.p1.unsafe_load(2 + BN_SUMG * C + c))


@always_inline
def op_rnv_bn_dx(i: Int, a: Args):
    """Training-mode dx: x p0, g p1, aux p2, dx p3 (bn_bwd_dx_at)."""
    bn_bwd_dx_at(i, _xf(a.p0), _xf(a.p1), _xf(a.p2), _xf(a.p3), _xd(a.p11), _xd(a.p11))


@always_inline
def op_rnv_bn_dx_eval(i: Int, a: Args):
    """Eval-mode dx = g invstd gamma: x p0, g p1, aux p2, dx p3 (bn_bwd_eval_dx_at)."""
    bn_bwd_eval_dx_at(i, _xf(a.p0), _xf(a.p1), _xf(a.p2), _xf(a.p3), _xd(a.p11), _xd(a.p11))


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
    elif OP == OP_RNV_BN_PREP:
        op_rnv_bn_prep(t, a)
    elif OP == OP_RNV_BN_SUM:
        op_rnv_bn_sum(t, a)
    elif OP == OP_RNV_BN_MEAN:
        op_rnv_bn_mean(t, a)
    elif OP == OP_RNV_BN_SQ:
        op_rnv_bn_sq(t, a)
    elif OP == OP_RNV_BN_VAR:
        op_rnv_bn_var(t, a)
    elif OP == OP_RNV_BN_APPLY:
        op_rnv_bn_apply(t, a)
    elif OP == OP_RNV_BN_RED:
        op_rnv_bn_red(t, a)
    elif OP == OP_RNV_BN_RED_FIN:
        op_rnv_bn_red_fin(t, a)
    elif OP == OP_RNV_BN_DX:
        op_rnv_bn_dx(t, a)
    elif OP == OP_RNV_BN_DX_EVAL:
        op_rnv_bn_dx_eval(t, a)
