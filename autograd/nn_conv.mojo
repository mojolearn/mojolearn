# SPDX-License-Identifier: Apache-2.0
# THE CONVOLUTION AND SHAPE OPERATIONS OF SEQUENTIAL ON THE TAPE (lane nn-conv, 2026-11-23).
#
# Layer codes 56..66 (autograd/model_blocks.mojo): conv_transpose1d / 2d, adaptive_avg_pool2d,
# adaptive_max_pool2d / 1d, global_max_pool, constant / zero padding, reflection and replication
# padding, upsampling, pixel shuffle / unshuffle. NCHW, x_cnn's layout. Each operation below is either
# a composition of tape ops the product already has, or a CUSTOM node (tags NNC_TAG_FIRST..LAST,
# NnConvRules below) whose launches run the bodies of autograd/nn_conv_ops.mojo, which call x_cnn's and
# the tape convolution's own element functions.
#
# LAUNCHES (per forward / per backward, one layer):
#   conv_transpose   forward 3: OP_RNV_CONV_GROWS (x to the GEMM's row layout), one OP_GEMM (dcols =
#                    xrows @ W; groups > 1: one OP_RN_BMM over every group), OP_NNC_CT_OUT (the conv's
#                    col2im gather + bias). Backward up to 5 plus the bias fold: OP_RNV_IM2COL of the
#                    upstream, dW (one GEMM / BMM), dx (one GEMM / BMM, then OP_RNV_CONV_OUT), and the
#                    bias: OP_RNV_CONV_GROWS of the upstream and the tape's blocked fold over its rows.
#                    It is exactly the adjoint of the tape's convolution: its forward is that
#                    convolution's input-gradient launches, its backward that convolution's forward and
#                    weight-gradient launches.
#   adaptive avg     forward 2 (x_cnn's blocked window fold: block chains, then their totals and one
#                    division), backward 1 (the gather).
#   adaptive max     forward 1 (values and winners), backward 1.
#   global max pool  rankn_max over the spatial axes (the tape's blocked parallel max, autograd/
#                    rankn_fold.mojo; NaN wins, the first maximum takes the gradient): its launches.
#   constant pad     imp_pad (autograd/import_api.mojo): forward 1, backward 1.
#   reflect / replicate pad   forward 1, backward 1 (x_cnn's pad bodies).
#   upsample         forward 1, backward 1 (the gather).
#   pixel (un)shuffle a view, a permute (a strided view) and a view: forward 1 copy, backward 1.
# Host work here is shape metadata and descriptor words, once per recorded node: no loop over data.
from std.memory import bitcast
from std.sys.compile import is_defined
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_api import rn_shape
from autograd.fold import fold_partials, blocked_fold
from autograd.ops import AXIS_ROWS
from autograd.import_api import imp_pad
from autograd.rankn_fold import rankn_max
from autograd.rankn_conv_ops import OP_RNV_CONV_GROWS, OP_RNV_IM2COL, OP_RNV_CONV_OUT
from autograd.rules_conv2d import rn_conv_gemm, rn_conv_group_gemm, rn_conv_group_words, RNG_FWD, RNG_DW, RNG_DCOLS
from autograd.nn_conv_ops import (
    OP_NNC_CT_OUT, OP_NNC_PAD, OP_NNC_PAD_VJP, OP_NNC_AAVG_BLK, OP_NNC_AAVG_FIN, OP_NNC_AAVG_VJP,
    OP_NNC_AMAX, OP_NNC_AMAX_VJP, OP_NNC_UP, OP_NNC_UP_VJP,
)
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr
from x_cnn.ops import conv_params, CP_OH, CP_OW, PAD_REFLECT, PAD_REPLICATE, gap_fold_blocks

comptime NNC_TAG_CONV_TRANSPOSE = 560
comptime NNC_TAG_PAD = 561
comptime NNC_TAG_ADAPT_AVG = 562
comptime NNC_TAG_ADAPT_MAX = 563
comptime NNC_TAG_UPSAMPLE = 564
comptime NNC_TAG_FIRST = 560
comptime NNC_TAG_LAST = 575

comptime NNC_PAD_REFLECT = PAD_REFLECT
comptime NNC_PAD_REPLICATE = PAD_REPLICATE

#: the pixel shuffle's sabotage arm (test builds only): the two factor axes are interleaved in the wrong
#: order (the permutation's r axes swapped), so its native check is shown to catch it
comptime NN_CONV_SABOTAGE_SHUFFLE = is_defined["MOJOLEARN_NN_CONV_SABOTAGE_SHUFFLE"]()


def _sh(dims: List[Int]) raises -> ShapeN:
    return rn_shape(dims, "nn_conv")


def _words(block: List[Int32]) -> List[Int]:
    var words = List[Int](capacity=len(block))
    for v in block:  # small-loop(len(block): the CP_ parameter block, 17 words)
        words.append(Int(v))
    return words^


def _f32_word(v: Float32) -> Int:
    return Int(bitcast[DType.int32](v))


# ---------------------------------------------------------------- transposed convolution
def nn_conv_transpose2d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor],
    stride_h: Int, stride_w: Int, padding_h: Int, padding_w: Int, out_pad_h: Int, out_pad_w: Int,
    dilation_h: Int, dilation_w: Int, groups: Int,
) raises -> Tensor:
    """The transposed convolution of x (N, C_in, H, W) with weight (C_in, C_out // groups, KH, KW), as
    PyTorch's conv_transpose2d: the output (N, C_out, OH, OW), OH = (H - 1) stride - 2 padding +
    dilation (KH - 1) + output_padding + 1, is the adjoint of the convolution of that output with the
    same weight, stride, padding and dilation (whose output is x), plus bias (C_out)."""
    var sx = tape.shape_of(x)
    var sw = tape.shape_of(weight)
    if sx.ndim != 4 or sw.ndim != 4:
        raise Error("conv_transpose: x and weight must be rank-4 tensors")
    var N = sx.d[0]
    var Cin = sx.d[1]
    var H = sx.d[2]
    var W = sx.d[3]
    var KH = sw.d[2]
    var KW = sw.d[3]
    if groups < 1 or Cin % groups != 0 or sw.d[0] != Cin:
        raise Error("conv_transpose: weight's first dimension must equal x's channels, and groups divide it")
    var Cout = sw.d[1] * groups
    if stride_h < 1 or stride_w < 1 or dilation_h < 1 or dilation_w < 1 or padding_h < 0 or padding_w < 0:
        raise Error("conv_transpose: stride and dilation must be positive and padding non-negative")
    if out_pad_h < 0 or out_pad_w < 0 or out_pad_h >= stride_h or out_pad_w >= stride_w:
        raise Error("conv_transpose: output_padding must be non-negative and smaller than stride")
    var OH = (H - 1) * stride_h - 2 * padding_h + dilation_h * (KH - 1) + out_pad_h + 1
    var OW = (W - 1) * stride_w - 2 * padding_w + dilation_w * (KW - 1) + out_pad_w + 1
    if OH < 1 or OW < 1:
        raise Error("conv_transpose: the output would be empty (padding too large)")
    var has_bias = Bool(bias)
    if has_bias:
        var b = bias.value()
        if tape.rows(b) * tape.cols(b) != Cout:
            raise Error("conv_transpose: bias must hold out_channels values")
    # the convolution whose input is this output (C = C_out, OC = C_in); its CP_BIAS word stays 0 (the
    # transposed output adds its own bias, and the backward's OP_RNV_CONV_OUT must add none)
    var raw: List[Int] = [N, Cout, OH, OW, Cin, KH, KW, stride_h, stride_w, padding_h, padding_w,
                          dilation_h, dilation_w, 0, 0, 0, 0]
    var block = conv_params(raw)
    if Int(block[CP_OH]) != H or Int(block[CP_OW]) != W:
        raise Error("conv_transpose: the geometry does not invert (output_padding must be smaller than stride)")
    var rows = N * H * W
    var ckk = Cout * KH * KW
    var desc = tape.descriptor(ex, _words(block))
    var xv = tape.value_of(ex, x)
    var wv = tape.value_of(ex, weight)
    var bv = dummy_ptr()
    if has_bias:
        bv = tape.value_of(ex, bias.value())
    # x in the GEMM's row layout [rows, C_in]
    var xrows = tape.reserve(ex, rows * Cin)
    var g = Args()
    g.p0 = xv
    g.p1 = xrows
    g.p11 = desc
    ex.launch[OP_RNV_CONV_GROWS](g, rows * Cin)
    # dcols [rows, C_out KH KW] = xrows @ W
    var dcols = tape.reserve(ex, rows * ckk)
    if groups == 1:
        rn_conv_gemm(ex, dcols, xrows, wv, rows, ckk, Cin, Cin, 1, ckk, 1)
    else:
        var cdesc = tape.descriptor(ex, rn_conv_group_words(RNG_DCOLS, rows, Cout, Cin, KH * KW, groups))
        rn_conv_group_gemm(ex, dcols, xrows, wv, cdesc, rows * ckk)
    var count = N * Cout * OH * OW
    var out = tape.reserve(ex, count)
    var o = Args()
    o.p0 = dcols
    o.p1 = out
    o.p2 = bv
    o.p11 = desc
    o.i0 = 1 if has_bias else 0
    ex.launch[OP_NNC_CT_OUT](o, count)
    var rec = tape.is_recording()
    var need_x = rec and tape.requires_grad(x)
    var need_w = rec and tape.requires_grad(weight)
    var need_b = rec and has_bias and tape.requires_grad(bias.value())
    var saved = Args()
    saved.p0 = xrows
    saved.p11 = desc
    if need_x or need_w:
        saved.p1 = tape.reserve(ex, rows * ckk)  # the upstream's im2col
    if need_x:
        saved.p2 = tape.reserve(ex, rows * Cin)  # dx in the row layout
    var orows = N * OH * OW
    if need_b:
        saved.p3 = tape.reserve(ex, orows * Cout)
        saved.p5 = tape.reserve(ex, fold_partials(orows, Cout, AXIS_ROWS))
        # a 1x1 convolution over the output: its CP_* words put the upstream in row layout
        var braw: List[Int] = [N, Cout, OH, OW, Cout, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
        saved.p4 = tape.descriptor(ex, _words(conv_params(braw)))
    if groups > 1:
        if need_w:
            saved.p6 = tape.descriptor(ex, rn_conv_group_words(RNG_DW, rows, Cout, Cin, KH * KW, groups))
        if need_x:
            saved.p7 = tape.descriptor(ex, rn_conv_group_words(RNG_FWD, rows, Cout, Cin, KH * KW, groups))
    saved.i0 = 1 if has_bias else 0
    saved.i1 = rows
    saved.i2 = ckk
    saved.i3 = Cin
    saved.i4 = N * Cin * H * W
    saved.i5 = 1 if need_x else 0
    saved.i6 = 1 if need_w else 0
    saved.i7 = 1 if need_b else 0
    saved.i8 = groups
    saved.i9 = orows
    saved.i10 = Cout
    var inputs: List[Tensor] = [x, weight]
    if has_bias:
        inputs.append(bias.value())
    var od: List[Int] = [N, Cout, OH, OW]
    return tape.custom_nd(ex, inputs, out, _sh(od), NNC_TAG_CONV_TRANSPOSE, saved)


def nn_conv_transpose1d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor],
    stride: Int, padding: Int, output_padding: Int, dilation: Int, groups: Int,
) raises -> Tensor:
    """nn_conv_transpose2d over (N, C_in, 1, L) with a (C_in, C_out // groups, 1, K) weight: x (N, C_in,
    L), weight (C_in, C_out // groups, K); the output (N, C_out, OL)."""
    var sx = tape.shape_of(x)
    var sw = tape.shape_of(weight)
    if sx.ndim != 3 or sw.ndim != 3:
        raise Error("conv_transpose1d: x and weight must be rank-3 tensors")
    var x4d: List[Int] = [sx.d[0], sx.d[1], 1, sx.d[2]]
    var w4d: List[Int] = [sw.d[0], sw.d[1], 1, sw.d[2]]
    var y4 = nn_conv_transpose2d(tape, ex, tape.view_nd(ex, x, _sh(x4d)), tape.view_nd(ex, weight, _sh(w4d)),
                                 bias, 1, stride, 0, padding, 0, output_padding, 1, dilation, groups)
    var sy = tape.shape_of(y4)
    var y3: List[Int] = [sy.d[0], sy.d[1], sy.d[3]]
    return tape.view_nd(ex, y4, _sh(y3))


def _conv_transpose_vjp[E: Exec](mut ex: E, saved: Args, upstream: FP, inputs: List[FP], gradients: List[FP]) raises:
    """The transposed convolution's VJP (the header's launches). inputs: x, weight[, bias]."""
    var rows = saved.i1
    var ckk = saved.i2
    var cin = saved.i3
    var grouped = saved.i8 > 1
    if saved.i5 != 0 or saved.i6 != 0:
        var c = Args()
        c.p0 = upstream
        c.p1 = saved.p1
        c.p11 = saved.p11
        ex.launch[OP_RNV_IM2COL](c, rows * ckk)
    if saved.i6 != 0:
        if grouped:
            rn_conv_group_gemm(ex, gradients[1], saved.p0, saved.p1, saved.p6, cin * (ckk // saved.i8))
        else:
            # dW(ci, q) = sum over r of xrows[r, ci] gcols[r, q]
            rn_conv_gemm(ex, gradients[1], saved.p0, saved.p1, cin, ckk, rows, 1, cin, ckk, 1)
    if saved.i5 != 0:
        if grouped:
            rn_conv_group_gemm(ex, saved.p2, saved.p1, inputs[1], saved.p7, rows * cin)
        else:
            # dx2(r, ci) = sum over q of gcols[r, q] W[ci, q]
            rn_conv_gemm(ex, saved.p2, saved.p1, inputs[1], rows, cin, ckk, ckk, 1, 1, ckk)
        var o = Args()
        o.p0 = saved.p2
        o.p2 = gradients[0]
        o.p11 = saved.p11
        ex.launch[OP_RNV_CONV_OUT](o, saved.i4)
    if saved.i0 != 0 and saved.i7 != 0:
        var b = Args()
        b.p0 = upstream
        b.p1 = saved.p3
        b.p11 = saved.p4
        ex.launch[OP_RNV_CONV_GROWS](b, saved.i9 * saved.i10)
        blocked_fold(ex, gradients[2], saved.p3, saved.i9, saved.i10, AXIS_ROWS, Float32(1), saved.p5)


# ---------------------------------------------------------------- padding
def nn_pad[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, left: Int, right: Int, top: Int, bottom: Int, mode: Int,
    value: Float32,
) raises -> Tensor:
    """x (N, C, L) or (N, C, H, W) padded on its last axis by (left, right) and, for rank 4, its height
    by (top, bottom), as PyTorch's ConstantPad / ReflectionPad / ReplicationPad: mode 0 constant
    `value` (a negative pad crops), NNC_PAD_REFLECT the mirror without the edge, NNC_PAD_REPLICATE the
    edge repeated."""
    var s = tape.shape_of(x)
    if s.ndim != 3 and s.ndim != 4:
        raise Error("pad: x must be (N, C, L) or (N, C, H, W)")
    var two = s.ndim == 4
    if mode == 0:
        var pads = List[Int](length=2 * s.ndim, fill=0)
        pads[s.ndim - 1] = left
        pads[2 * s.ndim - 1] = right
        if two:
            pads[2] = top
            pads[s.ndim + 2] = bottom
        return imp_pad(tape, ex, x, pads, value)
    var N = s.d[0]
    var C = s.d[1]
    var H = s.d[2] if two else 1
    var W = s.d[s.ndim - 1]
    if not two and (top != 0 or bottom != 0):
        raise Error("pad: a (N, C, L) input has no height to pad")
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        raise Error("pad: reflection and replication pads must be non-negative")
    if mode == NNC_PAD_REFLECT and (left >= W or right >= W or top >= H or bottom >= H):
        raise Error("pad: a reflection pad must be smaller than the padded dimension")
    if mode != NNC_PAD_REFLECT and mode != NNC_PAD_REPLICATE:
        raise Error("pad: unknown mode")
    var words: List[Int] = [N, C, H, W, top, bottom, left, right, mode]
    var desc = tape.descriptor(ex, words)
    var Hp = H + top + bottom
    var Wp = W + left + right
    var count = N * C * Hp * Wp
    var out = tape.reserve(ex, count)
    var a = Args()
    a.p0 = tape.value_of(ex, x)
    a.p1 = out
    a.p11 = desc
    ex.launch[OP_NNC_PAD](a, count)
    var saved = Args()
    saved.p11 = desc
    saved.i4 = s.count()
    saved.i5 = 1 if (tape.is_recording() and tape.requires_grad(x)) else 0
    var od = List[Int]()
    od.append(N)
    od.append(C)
    if two:
        od.append(Hp)
    od.append(Wp)
    return tape.custom_nd(ex, [x], out, _sh(od), NNC_TAG_PAD, saved)


# ---------------------------------------------------------------- adaptive pooling
def _adapt_words(N: Int, C: Int, H: Int, W: Int, OH: Int, OW: Int) raises -> List[Int]:
    if OH < 1 or OW < 1:
        raise Error("adaptive pool: output_size must be positive")
    var words: List[Int] = [N, C, H, W, OH, OW]
    return words^


def nn_adaptive_avg_pool2d[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, OH: Int, OW: Int) raises -> Tensor:
    """PyTorch's adaptive_avg_pool2d of x (N, C, H, W) to (N, C, OH, OW): output (oh, ow) is the mean of
    rows [floor(oh H / OH), ceil((oh + 1) H / OH)) and the columns likewise, folded in x_cnn's blocked
    (h, w) order."""
    var s = tape.shape_of(x)
    if s.ndim != 4:
        raise Error("adaptive_avg_pool2d: x must be (N, C, H, W)")
    var N = s.d[0]; var C = s.d[1]; var H = s.d[2]; var W = s.d[3]
    var desc = tape.descriptor(ex, _adapt_words(N, C, H, W, OH, OW))
    var count = N * C * OH * OW
    var NB = gap_fold_blocks(H, W, OH, OW)
    var part = tape.reserve(ex, count * NB)
    var b = Args()
    b.p0 = tape.value_of(ex, x)
    b.p1 = part
    b.p11 = desc
    ex.launch[OP_NNC_AAVG_BLK](b, count * NB)
    var out = tape.reserve(ex, count)
    var f = Args()
    f.p0 = part
    f.p1 = out
    f.p11 = desc
    ex.launch[OP_NNC_AAVG_FIN](f, count)
    var saved = Args()
    saved.p11 = desc
    saved.i4 = s.count()
    saved.i5 = 1 if (tape.is_recording() and tape.requires_grad(x)) else 0
    var od: List[Int] = [N, C, OH, OW]
    return tape.custom_nd(ex, [x], out, _sh(od), NNC_TAG_ADAPT_AVG, saved)


def nn_adaptive_max_pool2d[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, OH: Int, OW: Int) raises -> Tensor:
    """PyTorch's adaptive_max_pool2d of x (N, C, H, W) to (N, C, OH, OW) over the same windows as the
    adaptive average: the first maximum in (h, w) order (a NaN wins) takes the gradient."""
    var s = tape.shape_of(x)
    if s.ndim != 4:
        raise Error("adaptive_max_pool2d: x must be (N, C, H, W)")
    var N = s.d[0]; var C = s.d[1]; var H = s.d[2]; var W = s.d[3]
    var desc = tape.descriptor(ex, _adapt_words(N, C, H, W, OH, OW))
    var count = N * C * OH * OW
    var out = tape.reserve(ex, count)
    var winners = tape.reserve(ex, count)
    var a = Args()
    a.p0 = tape.value_of(ex, x)
    a.p1 = out
    a.p2 = winners
    a.p11 = desc
    ex.launch[OP_NNC_AMAX](a, count)
    var saved = Args()
    saved.p10 = winners
    saved.p11 = desc
    saved.i4 = s.count()
    saved.i5 = 1 if (tape.is_recording() and tape.requires_grad(x)) else 0
    var od: List[Int] = [N, C, OH, OW]
    return tape.custom_nd(ex, [x], out, _sh(od), NNC_TAG_ADAPT_MAX, saved)


def nn_adaptive_max_pool1d[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, OL: Int) raises -> Tensor:
    """adaptive_max_pool1d of x (N, C, L) to (N, C, OL): nn_adaptive_max_pool2d over (N, C, 1, L)."""
    var s = tape.shape_of(x)
    if s.ndim != 3:
        raise Error("adaptive_max_pool1d: x must be (N, C, L)")
    var x4: List[Int] = [s.d[0], s.d[1], 1, s.d[2]]
    var y = nn_adaptive_max_pool2d(tape, ex, tape.view_nd(ex, x, _sh(x4)), 1, OL)
    var y3: List[Int] = [s.d[0], s.d[1], OL]
    return tape.view_nd(ex, y, _sh(y3))


def nn_adaptive_avg_pool1d[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, OL: Int) raises -> Tensor:
    """adaptive_avg_pool1d of x (N, C, L) to (N, C, OL) for any OL (nn_adaptive_avg_pool2d over
    (N, C, 1, L))."""
    var s = tape.shape_of(x)
    if s.ndim != 3:
        raise Error("adaptive_avg_pool1d: x must be (N, C, L)")
    var x4: List[Int] = [s.d[0], s.d[1], 1, s.d[2]]
    var y = nn_adaptive_avg_pool2d(tape, ex, tape.view_nd(ex, x, _sh(x4)), 1, OL)
    var y3: List[Int] = [s.d[0], s.d[1], OL]
    return tape.view_nd(ex, y, _sh(y3))


def nn_global_max_pool[E: Exec](mut tape: Tape, mut ex: E, x: Tensor) raises -> Tensor:
    """The maximum of each (sample, channel) over every spatial position of x (N, C, L) or (N, C, H, W):
    (N, C). The tape's blocked parallel max (rankn_max): a NaN wins, the first maximum takes the
    gradient."""
    var s = tape.shape_of(x)
    if s.ndim != 3 and s.ndim != 4:
        raise Error("global_max_pool: x must be (N, C, L) or (N, C, H, W)")
    var axes = List[Int]()
    for k in range(2, s.ndim):  # small-loop(at most two spatial axes): shape metadata
        axes.append(k)
    var m = rankn_max(tape, ex, x, axes)
    var od: List[Int] = [s.d[0], s.d[1]]
    return tape.view_nd(ex, m, _sh(od))


# ---------------------------------------------------------------- upsampling
def up_scale(isize: Int, osize: Int, scale_factor: Float64, align: Bool, nearest: Bool) -> Float32:
    """PyTorch's source-index scale of one axis (area_pixel_compute_scale): with align_corners (linear
    only) (isize - 1) / (osize - 1), 0 for one output; otherwise 1 / scale_factor when one was given,
    else isize / osize. Rounded to float32 once (the kernels' arithmetic)."""
    if align and not nearest:
        if osize > 1:
            return Float32(Float64(isize - 1) / Float64(osize - 1))
        return Float32(0)
    if scale_factor > 0:
        return Float32(Float64(1) / scale_factor)
    return Float32(Float64(isize) / Float64(osize))


def nn_upsample[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, OH: Int, OW: Int, scale_h: Float32, scale_w: Float32,
    bilinear: Bool, align: Bool,
) raises -> Tensor:
    """x (N, C, H, W) resampled to (N, C, OH, OW), as PyTorch's interpolate (mode 'nearest' or
    'bilinear'), with the per-axis source scales of up_scale."""
    var s = tape.shape_of(x)
    if s.ndim != 4:
        raise Error("upsample: x must be (N, C, H, W)")
    if OH < 1 or OW < 1:
        raise Error("upsample: the output size must be positive")
    var N = s.d[0]; var C = s.d[1]; var H = s.d[2]; var W = s.d[3]
    var words: List[Int] = [N * C, H, W, OH, OW, 1 if bilinear else 0, 1 if align else 0,
                            _f32_word(scale_h), _f32_word(scale_w)]
    var desc = tape.descriptor(ex, words)
    var count = N * C * OH * OW
    var out = tape.reserve(ex, count)
    var a = Args()
    a.p0 = tape.value_of(ex, x)
    a.p1 = out
    a.p11 = desc
    ex.launch[OP_NNC_UP](a, count)
    var saved = Args()
    saved.p11 = desc
    saved.i4 = s.count()
    saved.i5 = 1 if (tape.is_recording() and tape.requires_grad(x)) else 0
    var od: List[Int] = [N, C, OH, OW]
    return tape.custom_nd(ex, [x], out, _sh(od), NNC_TAG_UPSAMPLE, saved)


# ---------------------------------------------------------------- pixel shuffle
def nn_pixel_shuffle[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, r: Int, unshuffle: Bool) raises -> Tensor:
    """PyTorch's pixel_shuffle (C r^2, H, W) -> (C, H r, W r) per sample, out[c, h r + i, w r + j] =
    x[c r^2 + i r + j, h, w]; with `unshuffle` its inverse pixel_unshuffle. A view, a permute and a view:
    no arithmetic, one copy each way."""
    var s = tape.shape_of(x)
    if s.ndim != 4 or r < 1:
        raise Error("pixel_shuffle: x must be (N, C, H, W) and the factor positive")
    var N = s.d[0]
    var perm: List[Int]
    var split: List[Int]
    var od: List[Int]
    if not unshuffle:
        if s.d[1] % (r * r) != 0:
            raise Error("pixel_shuffle: the channels must be divisible by upscale_factor squared")
        var C = s.d[1] // (r * r)
        var H = s.d[2]
        var W = s.d[3]
        split = [N, C, r, r, H, W]
        perm = [0, 1, 4, 2, 5, 3]   # (N, C, H, r, W, r)
        comptime if NN_CONV_SABOTAGE_SHUFFLE:
            perm = [0, 1, 4, 3, 5, 2]
        od = [N, C, H * r, W * r]
    else:
        if s.d[2] % r != 0 or s.d[3] % r != 0:
            raise Error("pixel_unshuffle: the height and width must be divisible by downscale_factor")
        var C = s.d[1]
        var H = s.d[2] // r
        var W = s.d[3] // r
        split = [N, C, H, r, W, r]
        perm = [0, 1, 3, 5, 2, 4]   # (N, C, r, r, H, W)
        comptime if NN_CONV_SABOTAGE_SHUFFLE:
            perm = [0, 1, 5, 3, 2, 4]
        od = [N, C * r * r, H, W]
    var p = tape.permute(ex, tape.view_nd(ex, x, _sh(split)), perm)
    return tape.view_nd(ex, p, _sh(od))


# ---------------------------------------------------------------- backward
@fieldwise_init
struct NnConvRules(BackwardRules, ImplicitlyCopyable, Movable):
    """The backward of the custom nodes above (tags NNC_TAG_FIRST..NNC_TAG_LAST). Every destination
    word this rule touches is written once; an input that needs no gradient is never touched."""

    @staticmethod
    def owns(tag: Int) -> Bool:
        return tag >= NNC_TAG_FIRST and tag <= NNC_TAG_LAST

    def supports(self, tag: Int) -> Bool:
        return (tag == NNC_TAG_CONV_TRANSPOSE or tag == NNC_TAG_PAD or tag == NNC_TAG_ADAPT_AVG
                or tag == NNC_TAG_ADAPT_MAX or tag == NNC_TAG_UPSAMPLE)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == NNC_TAG_CONV_TRANSPOSE:
            _conv_transpose_vjp(ex, saved, upstream, inputs, gradients)
            return
        if saved.i5 == 0:
            return
        var a = Args()
        a.p0 = upstream
        a.p1 = gradients[0]
        a.p11 = saved.p11
        if tag == NNC_TAG_PAD:
            ex.launch[OP_NNC_PAD_VJP](a, saved.i4)
        elif tag == NNC_TAG_ADAPT_AVG:
            ex.launch[OP_NNC_AAVG_VJP](a, saved.i4)
        elif tag == NNC_TAG_ADAPT_MAX:
            a.p2 = saved.p10
            ex.launch[OP_NNC_AMAX_VJP](a, saved.i4)
        elif tag == NNC_TAG_UPSAMPLE:
            ex.launch[OP_NNC_UP_VJP](a, saved.i4)
        else:
            raise Error("nn_conv rules: unknown tag")
