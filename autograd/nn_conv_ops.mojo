# SPDX-License-Identifier: Apache-2.0
# THE KERNEL BODIES OF THE CONVOLUTION AND SHAPE LAYERS OF SEQUENTIAL (lane nn-conv, 2026-11-23).
#
# One `seq_kernel` body a code, shared by HostExec and DeviceExec through sequence/dispatch.mojo, so
# the host column and every GPU run the same bodies. Codes OP_NNC_FIRST..OP_NNC_LAST (5560..5579).
#
# ONE BODY PER OPERATION. Where the product already has the math, the body here is a call of it:
#   OP_NNC_CT_OUT          the transposed convolution's output: autograd/rankn_conv_ops.mojo
#                          op_rnv_col2im (the conv's own input-gradient gather), then + bias
#   OP_NNC_PAD / _PAD_VJP  reflection and replication padding: x_cnn/ops.mojo pad_fwd_at /
#                          pad_bwd_at (the CNN layers' bodies; constant padding is the import
#                          family's imp_pad, autograd/import_api.mojo)
#   OP_NNC_AAVG_BLK / _FIN / _VJP   adaptive average pooling: x_cnn adapt_avg_blk_at,
#                          adapt_avg_fin_at, adapt_avg_bwd_at
#   OP_NNC_AMAX / _VJP     adaptive max pooling: x_cnn adapt_max_fwd_at / adapt_max_bwd_at
# New math (no copy existed in the product): the upsampling bodies (nearest and bilinear, PyTorch's
# index rules), forward and gather-form backward.
#
# ORDER. Every body is parallel over its output words (one thread a word); no atomics, no two threads
# write one word. The gathers (backward of the pads, the pools and the upsampling) add their terms in
# ascending output order from +0.0, so one order on the host and on every vendor.
from std.memory import bitcast
from std.sys.compile import is_defined
from sequence.ops import Args, FP
from checks.numerics import ftz, identical_mul
from autograd.rankn import rn_word
from autograd.rankn_conv_ops import op_rnv_col2im
from x_cnn.ops import FP as XFP, IP as XIP
from x_cnn.ops import CP_C, CP_H, CP_W
from x_cnn.ops import (
    pad_fwd_at, pad_bwd_at, adapt_avg_blk_at, adapt_avg_fin_at, adapt_avg_bwd_at,
    adapt_max_fwd_at, adapt_max_bwd_at,
)

comptime OP_NNC_CT_OUT = 5560
comptime OP_NNC_PAD = 5561
comptime OP_NNC_PAD_VJP = 5562
comptime OP_NNC_AAVG_BLK = 5563
comptime OP_NNC_AAVG_FIN = 5564
comptime OP_NNC_AAVG_VJP = 5565
comptime OP_NNC_AMAX = 5566
comptime OP_NNC_AMAX_VJP = 5567
comptime OP_NNC_UP = 5568
comptime OP_NNC_UP_VJP = 5569
comptime OP_NNC_FIRST = 5560
comptime OP_NNC_LAST = 5579

# SABOTAGE ARMS (test builds only, never defaults): each perturbs one module family so its native check
# (checks/native_tests/nn_conv.mojo) is shown to catch it. CT: the transposed convolution drops its bias.
# PAD: reflection's outermost left column reads +0.0. POOL: the adaptive average is scaled by 7/8. UP:
# bilinear swaps the two height weights. (SHUFFLE: autograd/nn_conv.mojo.)
comptime NN_CONV_SABOTAGE_CT = is_defined["MOJOLEARN_NN_CONV_SABOTAGE_CT"]()
comptime NN_CONV_SABOTAGE_PAD = is_defined["MOJOLEARN_NN_CONV_SABOTAGE_PAD"]()
comptime NN_CONV_SABOTAGE_POOL = is_defined["MOJOLEARN_NN_CONV_SABOTAGE_POOL"]()
comptime NN_CONV_SABOTAGE_UP = is_defined["MOJOLEARN_NN_CONV_SABOTAGE_UP"]()

# The upsampling descriptor (autograd/nn_conv.mojo nn_upsample): words
# [NC, H, W, OH, OW, mode (0 nearest, 1 bilinear), align_corners, scale_h bits, scale_w bits].
comptime UP_NC = 0
comptime UP_H = 1
comptime UP_W = 2
comptime UP_OH = 3
comptime UP_OW = 4
comptime UP_MODE = 5
comptime UP_ALIGN = 6
comptime UP_SH = 7
comptime UP_SW = 8
comptime UP_WORDS = 9


@always_inline
def _xf(p: FP) -> XFP:
    return p.unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _xd(p: FP) -> XIP:
    return p.bitcast[Int32]().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _ud(a: Int, b: Int) -> Int:
    return Int(UInt32(a) // UInt32(b))


@always_inline
def _um(a: Int, b: Int) -> Int:
    return Int(UInt32(a) % UInt32(b))


# ---------------------------------------------------------------- transposed convolution
@always_inline
def op_nnc_ct_out(t: Int, a: Args):
    """out[n, c, h, w] of a transposed convolution: the sum of every dcols entry whose tap lands on this
    pixel (op_rnv_col2im, the convolution's input-gradient gather, (kh, kw) ascending from +0.0), then
    + bias[c] when i0 = 1 (one add). p0 dcols [rows, C KH KW], p1 out, p2 bias; p11 the CP_* descriptor
    of the convolution whose input is this output."""
    op_rnv_col2im(t, a)
    var with_bias = a.i0 != 0
    comptime if NN_CONV_SABOTAGE_CT:
        with_bias = False
    if with_bias:
        var d = a.p11
        var hw = rn_word(d, CP_H) * rn_word(d, CP_W)
        var c = _um(_ud(t, hw), rn_word(d, CP_C))
        a.p1.unsafe_store(t, ftz(a.p1.unsafe_load(t) + ftz(a.p2.unsafe_load(c))))


# ---------------------------------------------------------------- padding (x_cnn's bodies)
@always_inline
def op_nnc_pad(i: Int, a: Args):
    """Padded word i (x_cnn pad_fwd_at): p0 x, p1 out; p11 [N, C, H, W, top, bottom, left, right, mode]."""
    var x = _xf(a.p0)
    pad_fwd_at(i, x, x, _xf(a.p1), x, _xd(a.p11), _xd(a.p11))
    comptime if NN_CONV_SABOTAGE_PAD:
        # reflection's outermost left column reads +0.0 instead of its mirrored source
        var Wp = rn_word(a.p11, 3) + rn_word(a.p11, 6) + rn_word(a.p11, 7)
        if rn_word(a.p11, 8) == 1 and rn_word(a.p11, 6) > 0 and i % Wp == 0:
            a.p1.unsafe_store(i, Float32(0))


@always_inline
def op_nnc_pad_vjp(i: Int, a: Args):
    """Source word i's gradient (x_cnn pad_bwd_at): p0 upstream, p1 dx; p11 the pad words."""
    var g = _xf(a.p0)
    pad_bwd_at(i, g, g, _xf(a.p1), g, _xd(a.p11), _xd(a.p11))


# ---------------------------------------------------------------- adaptive pooling (x_cnn's bodies)
@always_inline
def op_nnc_aavg_blk(t: Int, a: Args):
    """Slot t's block chain of an adaptive average (x_cnn adapt_avg_blk_at): p0 x, p1 part; p11
    [N, C, H, W, OH, OW]."""
    var x = _xf(a.p0)
    adapt_avg_blk_at(t, x, x, _xf(a.p1), x, _xd(a.p11), _xd(a.p11))


@always_inline
def op_nnc_aavg_fin(i: Int, a: Args):
    """Output i: its block partials, then one division (x_cnn adapt_avg_fin_at): p0 part, p1 out."""
    var x = _xf(a.p0)
    adapt_avg_fin_at(i, x, x, _xf(a.p1), x, _xd(a.p11), _xd(a.p11))
    comptime if NN_CONV_SABOTAGE_POOL:
        # the mean scaled by 7/8
        a.p1.unsafe_store(i, ftz(a.p1.unsafe_load(i) * Float32(0.875)))


@always_inline
def op_nnc_aavg_vjp(i: Int, a: Args):
    """Input word i's gradient (x_cnn adapt_avg_bwd_at): p0 upstream, p1 dx."""
    var g = _xf(a.p0)
    adapt_avg_bwd_at(i, g, g, _xf(a.p1), g, _xd(a.p11), _xd(a.p11))


@always_inline
def op_nnc_amax(i: Int, a: Args):
    """Output i, the first maximum of its window (a NaN wins), and its winner's h W + w as an int32
    pattern in p2 (x_cnn adapt_max_fwd_at): p0 x, p1 out."""
    var x = _xf(a.p0)
    adapt_max_fwd_at(i, x, x, _xf(a.p1), x, _xd(a.p2), _xd(a.p11))


@always_inline
def op_nnc_amax_vjp(i: Int, a: Args):
    """Input word i's gradient: the upstream of every window it won (x_cnn adapt_max_bwd_at): p0
    upstream, p1 dx, p2 the winners."""
    var g = _xf(a.p0)
    adapt_max_bwd_at(i, g, g, _xf(a.p1), g, _xd(a.p2), _xd(a.p11))


# ---------------------------------------------------------------- upsampling
@always_inline
def _scale(d: FP, k: Int) -> Float32:
    return bitcast[DType.float32](Int32(rn_word(d, k)))


@always_inline
def up_nearest_index(o: Int, isize: Int, scale: Float32) -> Int:
    """PyTorch's 'nearest' source index: min(floor(o * scale), isize - 1) (o * scale >= 0, one product)."""
    var s = Int(identical_mul(Float32(o), scale))
    return s if s < isize - 1 else isize - 1


@always_inline
def up_linear_taps(o: Int, isize: Int, scale: Float32, align: Bool) -> Tuple[Int, Int, Float32, Float32]:
    """(i0, i1, lambda0, lambda1) of output index o (PyTorch's area_pixel_compute_source_index, linear):
    src = scale * o (align_corners) or max(scale * (o + 0.5) - 0.5, 0); i0 = floor(src), i1 = i0 + 1
    below the last index (else i0), lambda1 = src - i0, lambda0 = 1 - lambda1."""
    var src: Float32
    if align:
        src = ftz(identical_mul(scale, Float32(o)))
    else:
        src = ftz(ftz(identical_mul(scale, Float32(o) + Float32(0.5))) - Float32(0.5))
        if src < Float32(0):
            src = Float32(0)
    var i0 = Int(src)
    var i1 = i0 + 1 if i0 < isize - 1 else i0
    var l1 = ftz(src - Float32(i0))
    var l0 = ftz(Float32(1) - l1)
    return (i0, i1, l0, l1)


@always_inline
def op_nnc_up(t: Int, a: Args):
    """Output word t of an upsampling (NCHW): nearest, the source pixel; bilinear,
    l0h (l0w x00 + l1w x01) + l1h (l0w x10 + l1w x11) (PyTorch's order). p0 x, p1 out; p11 UP_*."""
    var d = a.p11
    var H = rn_word(d, UP_H); var W = rn_word(d, UP_W)
    var OH = rn_word(d, UP_OH); var OW = rn_word(d, UP_OW)
    var ow = _um(t, OW)
    var s = _ud(t, OW)
    var oh = _um(s, OH)
    var nc = _ud(s, OH)
    var base = nc * H * W
    if rn_word(d, UP_MODE) == 0:
        var h = up_nearest_index(oh, H, _scale(d, UP_SH))
        var w = up_nearest_index(ow, W, _scale(d, UP_SW))
        a.p1.unsafe_store(t, ftz(a.p0.unsafe_load(base + h * W + w)))
        return
    var align = rn_word(d, UP_ALIGN) != 0
    var th = up_linear_taps(oh, H, _scale(d, UP_SH), align)
    var tw = up_linear_taps(ow, W, _scale(d, UP_SW), align)
    var l0h = th[2]
    var l1h = th[3]
    comptime if NN_CONV_SABOTAGE_UP:
        l0h = th[3]
        l1h = th[2]
    var x00 = ftz(a.p0.unsafe_load(base + th[0] * W + tw[0]))
    var x01 = ftz(a.p0.unsafe_load(base + th[0] * W + tw[1]))
    var x10 = ftz(a.p0.unsafe_load(base + th[1] * W + tw[0]))
    var x11 = ftz(a.p0.unsafe_load(base + th[1] * W + tw[1]))
    var r0 = ftz(ftz(identical_mul(tw[2], x00)) + ftz(identical_mul(tw[3], x01)))
    var r1 = ftz(ftz(identical_mul(tw[2], x10)) + ftz(identical_mul(tw[3], x11)))
    a.p1.unsafe_store(t, ftz(ftz(identical_mul(l0h, r0)) + ftz(identical_mul(l1h, r1))))


@always_inline
def up_candidates(i: Int, osize: Int, scale: Float32) -> Tuple[Int, Int]:
    """[lo, hi): a range of output indices holding every output that reads input index i, for both
    modes (the source index is non-decreasing in the output index; reading i needs a source position in
    [i - 1, i + 2)), widened by one on each side for rounding. scale 0: every output reads index 0."""
    if not (scale > Float32(0)):
        return (0, osize)
    # Float32 only: this runs inside a device kernel and Metal has no float64 (the Apple build of
    # _mojolearn_native_sequence_m1 failed "Failed to verify LLVM IR for Metal" on the float64 converts).
    # The range only bounds the candidate loop; every candidate is tested for membership and the
    # accumulation stays ascending, so a wider range gives the same bits. The float32 error of
    # (i + 2) / scale is under osize * 2^-22, so a margin of 2 + osize / 2^20 covers it; the bounds are
    # clamped in float before the integer conversion so a tiny scale cannot overflow it.
    var margin = 2 + (osize >> 20)
    var top = Float32(osize)
    var flo = min(max(Float32(i - 1) / scale, Float32(0)), top)
    var fhi = min(max(Float32(i + 2) / scale, Float32(0)), top)
    var lo = Int(flo) - margin
    var hi = Int(fhi) + margin
    if lo < 0:
        lo = 0
    if hi > osize:
        hi = osize
    return (lo, hi)


@always_inline
def op_nnc_up_vjp(t: Int, a: Args):
    """Input word t's gradient: over the outputs (oh, ow) ascending that read this pixel, the upstream
    times its weight, each output's taps in the forward's order (h0 w0, h0 w1, h1 w0, h1 w1; the
    weight lh * lw first, PyTorch's h0lambda * w0lambda * g), added from +0.0. p0 upstream, p1 dx."""
    var d = a.p11
    var H = rn_word(d, UP_H); var W = rn_word(d, UP_W)
    var OH = rn_word(d, UP_OH); var OW = rn_word(d, UP_OW)
    var sh = _scale(d, UP_SH); var sw = _scale(d, UP_SW)
    var w = _um(t, W)
    var s = _ud(t, W)
    var h = _um(s, H)
    var nc = _ud(s, H)
    var gbase = nc * OH * OW
    var nearest = rn_word(d, UP_MODE) == 0
    var align = rn_word(d, UP_ALIGN) != 0
    var rh = up_candidates(h, OH, sh)
    var rw = up_candidates(w, OW, sw)
    var acc = Float32(0)
    for oh in range(rh[0], rh[1]):
        if nearest:
            if up_nearest_index(oh, H, sh) != h:
                continue
            for ow in range(rw[0], rw[1]):
                if up_nearest_index(ow, W, sw) == w:
                    acc = ftz(acc + ftz(a.p0.unsafe_load(gbase + oh * OW + ow)))
            continue
        var th = up_linear_taps(oh, H, sh, align)
        var l0h = th[2]
        var l1h = th[3]
        comptime if NN_CONV_SABOTAGE_UP:
            l0h = th[3]
            l1h = th[2]
        if th[0] != h and th[1] != h:
            continue
        for ow in range(rw[0], rw[1]):
            var tw = up_linear_taps(ow, W, sw, align)
            if tw[0] != w and tw[1] != w:
                continue
            var g = ftz(a.p0.unsafe_load(gbase + oh * OW + ow))
            if th[0] == h and tw[0] == w:
                acc = ftz(acc + ftz(identical_mul(ftz(identical_mul(l0h, tw[2])), g)))
            if th[0] == h and tw[1] == w:
                acc = ftz(acc + ftz(identical_mul(ftz(identical_mul(l0h, tw[3])), g)))
            if th[1] == h and tw[0] == w:
                acc = ftz(acc + ftz(identical_mul(ftz(identical_mul(l1h, tw[2])), g)))
            if th[1] == h and tw[1] == w:
                acc = ftz(acc + ftz(identical_mul(ftz(identical_mul(l1h, tw[3])), g)))
    a.p1.unsafe_store(t, acc)


@always_inline
def apply_nn_conv[OP: Int](t: Int, a: Args):
    comptime if OP == OP_NNC_CT_OUT:
        op_nnc_ct_out(t, a)
    elif OP == OP_NNC_PAD:
        op_nnc_pad(t, a)
    elif OP == OP_NNC_PAD_VJP:
        op_nnc_pad_vjp(t, a)
    elif OP == OP_NNC_AAVG_BLK:
        op_nnc_aavg_blk(t, a)
    elif OP == OP_NNC_AAVG_FIN:
        op_nnc_aavg_fin(t, a)
    elif OP == OP_NNC_AAVG_VJP:
        op_nnc_aavg_vjp(t, a)
    elif OP == OP_NNC_AMAX:
        op_nnc_amax(t, a)
    elif OP == OP_NNC_AMAX_VJP:
        op_nnc_amax_vjp(t, a)
    elif OP == OP_NNC_UP:
        op_nnc_up(t, a)
    elif OP == OP_NNC_UP_VJP:
        op_nnc_up_vjp(t, a)
