# SPDX-License-Identifier: Apache-2.0
# conv2d, max_pool2d and avg_pool2d on the tape's rank-4 tensors (lane
# ag-rankn-conv, docs/plans/AUTOGRAD_RANKN.md section 2.8). NCHW, x_cnn's
# layout: x [N, C, H, W], weight [OC, C, KH, KW], bias [OC].
#
# Each op is a CUSTOM node (autograd/rules_conv2d.mojo Conv2dRules, tags
# 176..178) whose forward launches run here, queued on the executor and
# resident in tape storage (no host wait, no copy to the host):
#   conv2d      OP_RNV_IM2COL (one thread a cols word), OP_GEMM y2 [rows, OC]
#               = cols @ W^T (W read through strides, B(k, n) = W[n, k]: the
#               tape's GEMM chain over the C KH KW terms in im2col order),
#               OP_RNV_CONV_OUT (NCHW, + bias). 3 launches.
#               groups > 1 (grouped and depthwise): the same im2col over all
#               C channels (group g's taps are the contiguous columns
#               [g Cg KH KW, (g + 1) Cg KH KW) of cols, Cg = C / groups), one
#               OP_RN_BMM launch for every group's product (free axes rows,
#               g, OC / groups; one fma chain over group g's Cg KH KW terms in
#               im2col order from +0.0, op_gemm's chain), then the same
#               OP_RNV_CONV_OUT. groups = 1 is the path above, unchanged.
#   max_pool2d  OP_RNV_MAXPOOL_FWD (values and winners). 1 launch.
#   avg_pool2d  OP_RNV_AVGPOOL_FWD. 1 launch.
# The kernels are x_cnn's element functions on the same parameter words
# (CP_* / PP_*, validated and completed by x_cnn's conv_params /
# pool_params), uploaded once per node as a descriptor (Tape.descriptor).
# The x_cnn entries themselves are not called (they run on cnn_ctx with a
# host wait each and fuse pool(relu(conv))). The GEMM is the sequence GEMM's
# OP_GEMM, not CNNClassifier's pinned plan, so a tape conv2d has its own
# words (one order on the host and every vendor); CNNClassifier keeps its
# path and its digests.
#
# Host work here is shape metadata only: no loop over rows or elements.
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.fold import fold_partials
from autograd.ops import AXIS_ROWS
from autograd.rankn_conv_ops import (
    OP_RNV_IM2COL, OP_RNV_CONV_OUT, OP_RNV_MAXPOOL_FWD, OP_RNV_AVGPOOL_FWD,
)
from autograd.rules_conv2d import (
    RN_CONV2D, RN_MAXPOOL2D, RN_AVGPOOL2D, rn_conv_gemm, rn_conv_group_gemm, rn_conv_group_words,
    RNG_FWD, RNG_DW, RNG_DCOLS,
)
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr
from x_cnn.ops import conv_params, pool_params, CP_OH, CP_OW, PP_OH, PP_OW


def _shape4(a: Int, b: Int, c: Int, d: Int) -> ShapeN:
    var dims = InlineArray[Int, RN_MAX_RANK](fill=1)
    dims[0] = a
    dims[1] = b
    dims[2] = c
    dims[3] = d
    return ShapeN(4, dims)


def _dims4(tape: Tape, x: Tensor, what: String) raises -> ShapeN:
    var s = tape.shape_of(x)
    if s.ndim != 4:
        raise Error(what + " must be a rank-4 tensor")
    return s


def _words(block: List[Int32]) -> List[Int]:
    var words = List[Int](capacity=len(block))
    for v in block:  # small-loop(len(block): the CP_/PP_ parameter block, at most 18 words)
        words.append(Int(v))
    return words^


def rankn_conv2d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: Int = 1, padding: Int = 0, groups: Int = 1,
) raises -> Tensor:
    """Two-dimensional convolution (cross-correlation) of a batch of images.

    With groups > 1 the C input channels and the out_channels outputs are
    split into groups equal blocks, and output block g sees input block g
    only; groups = C is a depthwise convolution.

    Args:
        tape: The tape that records the operation.
        ex: The executor.
        x: The input, shape (N, C, H, W).
        weight: The kernels, shape (out_channels, C // groups, kh, kw).
        bias: Optional, out_channels values added to each output channel.
        stride: The step between windows, in both directions.
        padding: Zeros added on every side of each image.
        groups: The number of channel groups; it divides C and
            out_channels.

    Returns:
        The output, shape (N, out_channels, OH, OW) with
        OH = (H + 2 padding - kh) // stride + 1 and OW likewise.
    """
    var sx = _dims4(tape, x, "conv2d: x")
    var sw = _dims4(tape, weight, "conv2d: weight")
    var N = sx.d[0]
    var C = sx.d[1]
    var H = sx.d[2]
    var W = sx.d[3]
    var OC = sw.d[0]
    var KH = sw.d[2]
    var KW = sw.d[3]
    if groups < 1:
        raise Error("conv2d: groups must be positive")
    if groups == 1 and sw.d[1] != C:
        raise Error("conv2d: weight's second dimension must equal x's channels")
    if groups > 1:
        if C % groups != 0 or OC % groups != 0:
            raise Error("conv2d: groups must divide the input channels and out_channels")
        if sw.d[1] * groups != C:
            raise Error("conv2d: weight's second dimension must equal x's channels // groups")
    if stride <= 0 or padding < 0:
        raise Error("conv2d: stride must be positive and padding non-negative")
    var has_bias = Bool(bias)
    if has_bias:
        var b = bias.value()
        if tape.rows(b) * tape.cols(b) != OC:
            raise Error("conv2d: bias must hold out_channels values")
    var raw: List[Int] = [N, C, H, W, OC, KH, KW, stride, stride, padding, padding, 1, 1, 0, 0,
                          1 if has_bias else 0, 0]
    var block = conv_params(raw)
    var OH = Int(block[CP_OH])
    var OW = Int(block[CP_OW])
    var rows = N * OH * OW
    var ckk = C * KH * KW
    var desc = tape.descriptor(ex, _words(block))
    var xv = tape.value_of(ex, x)
    var wv = tape.value_of(ex, weight)
    var bv = dummy_ptr()
    if has_bias:
        bv = tape.value_of(ex, bias.value())
    # im2col
    var cols = tape.reserve(ex, rows * ckk)
    var a = Args()
    a.p0 = xv
    a.p1 = cols
    a.p11 = desc
    ex.launch[OP_RNV_IM2COL](a, rows * ckk)
    var y2 = tape.reserve(ex, rows * OC)
    if groups == 1:
        # y2 [rows, OC] = cols @ W^T, W(oc, q) at oc ckk + q
        rn_conv_gemm(ex, y2, cols, wv, rows, OC, ckk, ckk, 1, 1, ckk)
    else:
        # y2[r, g OCg + o] = sum over group g's taps of cols x W (one launch)
        var fdesc = tape.descriptor(ex, rn_conv_group_words(RNG_FWD, rows, C, OC, KH * KW, groups))
        rn_conv_group_gemm(ex, y2, cols, wv, fdesc, rows * OC)
    # NCHW output (+ bias)
    var out = tape.reserve(ex, N * OC * OH * OW)
    var o = Args()
    o.p0 = y2
    if has_bias:
        o.p1 = bv
    o.p2 = out
    o.p11 = desc
    ex.launch[OP_RNV_CONV_OUT](o, N * OC * OH * OW)
    var rec = tape.is_recording()
    var need_x = rec and tape.requires_grad(x)
    var need_w = rec and tape.requires_grad(weight)
    var need_b = rec and has_bias and tape.requires_grad(bias.value())
    var saved = Args()
    saved.p0 = cols
    saved.p2 = y2
    if need_x:
        saved.p4 = tape.reserve(ex, rows * ckk)
    if need_b:
        saved.p5 = tape.reserve(ex, fold_partials(rows, OC, AXIS_ROWS))
    if groups > 1:
        # the VJP's two grouped products, described once here (shape words)
        if need_w:
            saved.p6 = tape.descriptor(ex, rn_conv_group_words(RNG_DW, rows, C, OC, KH * KW, groups))
        if need_x:
            saved.p7 = tape.descriptor(ex, rn_conv_group_words(RNG_DCOLS, rows, C, OC, KH * KW, groups))
    saved.p11 = desc
    saved.i0 = 1 if has_bias else 0
    saved.i1 = rows
    saved.i2 = ckk
    saved.i3 = OC
    saved.i4 = N * C * H * W
    saved.i5 = 1 if need_x else 0
    saved.i6 = 1 if need_w else 0
    saved.i7 = 1 if need_b else 0
    saved.i8 = groups
    var inputs: List[Tensor] = [x, weight]
    if has_bias:
        inputs.append(bias.value())
    return tape.custom_nd(ex, inputs, out, _shape4(N, OC, OH, OW), RN_CONV2D, saved)


def _pool_block(sx: ShapeN, kernel_size: Int, stride: Int, padding: Int) raises -> List[Int32]:
    if kernel_size <= 0 or stride < 0 or padding < 0:
        raise Error("pool: kernel_size must be positive, stride and padding non-negative")
    var s = kernel_size if stride == 0 else stride
    # [N, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW, count_include_pad,
    #  reverse, ceil_mode, divisor_override]
    var raw: List[Int] = [sx.d[0], sx.d[1], sx.d[2], sx.d[3], kernel_size, kernel_size, s, s,
                          padding, padding, 1, 1, 0, 0, 1, 0, 0, 0]
    return pool_params(raw)


def rankn_max_pool2d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, kernel_size: Int, stride: Int = 0, padding: Int = 0,
) raises -> Tensor:
    """Two-dimensional max pooling of a batch of images.

    Each output is the largest value of its kernel_size x kernel_size
    window; when several values tie, the first in row-major window order
    is the one that receives the gradient.

    Args:
        tape: The tape that records the operation.
        ex: The executor.
        x: The input, shape (N, C, H, W).
        kernel_size: The window's height and width.
        stride: The step between windows; 0 means kernel_size.
        padding: Positions added on every side, never chosen as a maximum
            (at most half of kernel_size).

    Returns:
        The output, shape (N, C, OH, OW).
    """
    var sx = _dims4(tape, x, "max_pool2d: x")
    var block = _pool_block(sx, kernel_size, stride, padding)
    var OH = Int(block[PP_OH])
    var OW = Int(block[PP_OW])
    var count = sx.d[0] * sx.d[1] * OH * OW
    var desc = tape.descriptor(ex, _words(block))
    var xv = tape.value_of(ex, x)
    var out = tape.reserve(ex, count)
    var winners = tape.reserve(ex, count)
    var a = Args()
    a.p0 = xv
    a.p1 = out
    a.p2 = winners
    a.p11 = desc
    ex.launch[OP_RNV_MAXPOOL_FWD](a, count)
    var saved = Args()
    saved.p10 = winners
    saved.p11 = desc
    saved.i4 = sx.count()
    saved.i5 = 1 if (tape.is_recording() and tape.requires_grad(x)) else 0
    return tape.custom_nd(ex, [x], out, _shape4(sx.d[0], sx.d[1], OH, OW), RN_MAXPOOL2D, saved)


def rankn_avg_pool2d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, kernel_size: Int, stride: Int = 0, padding: Int = 0,
) raises -> Tensor:
    """Two-dimensional average pooling of a batch of images.

    Each output is the mean of its kernel_size x kernel_size window, the
    zero padding counted in the window's size.

    Args:
        tape: The tape that records the operation.
        ex: The executor.
        x: The input, shape (N, C, H, W).
        kernel_size: The window's height and width.
        stride: The step between windows; 0 means kernel_size.
        padding: Zeros added on every side (at most half of kernel_size).

    Returns:
        The output, shape (N, C, OH, OW).
    """
    var sx = _dims4(tape, x, "avg_pool2d: x")
    var block = _pool_block(sx, kernel_size, stride, padding)
    var OH = Int(block[PP_OH])
    var OW = Int(block[PP_OW])
    var count = sx.d[0] * sx.d[1] * OH * OW
    var desc = tape.descriptor(ex, _words(block))
    var xv = tape.value_of(ex, x)
    var out = tape.reserve(ex, count)
    var a = Args()
    a.p0 = xv
    a.p1 = out
    a.p11 = desc
    ex.launch[OP_RNV_AVGPOOL_FWD](a, count)
    var saved = Args()
    saved.p11 = desc
    saved.i4 = sx.count()
    saved.i5 = 1 if (tape.is_recording() and tape.requires_grad(x)) else 0
    return tape.custom_nd(ex, [x], out, _shape4(sx.d[0], sx.d[1], OH, OW), RN_AVGPOOL2D, saved)
