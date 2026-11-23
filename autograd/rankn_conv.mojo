# SPDX-License-Identifier: Apache-2.0
# conv2d, max_pool2d and avg_pool2d on the tape's rank-4 tensors (lane
# ag-rankn-conv, plan AUTOGRAD_RANKN (git history) section 2.8). NCHW, x_cnn's
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
    RN_DEPTH_LEN, OP_RNV_IM2COL, OP_RNV_CONV_OUT, OP_RNV_MAXPOOL_FWD, OP_RNV_AVGPOOL_FWD,
    OP_RNV_BN_PREP, OP_RNV_BN_SUM, OP_RNV_BN_MEAN, OP_RNV_BN_SQ, OP_RNV_BN_VAR, OP_RNV_BN_APPLY,
)
from autograd.rules_conv2d import (
    RN_CONV2D, RN_MAXPOOL2D, RN_AVGPOOL2D, RN_BATCH_NORM, rn_conv_gemm, rn_conv_group_gemm, rn_conv_group_words,
    RNG_FWD, RNG_DW, RNG_DCOLS,
)
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr
from x_cnn.ops import conv_params, pool_params, CP_OH, CP_OW, PP_OH, PP_OW, BN_SLOTS, bn_fold_blocks


def _shape4(a: Int, b: Int, c: Int, d: Int) -> ShapeN:
    var dims = Array[Int, RN_MAX_RANK](fill=1)
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


def rn_flat_depth() -> List[Int]:
    """The depth words of a node without a depth axis (1-D and 2-D): D 1, KD 1, SD 1, PD 0, DD 1,
    OD 1 (autograd/rankn_conv_ops.mojo RN_DEPTH_*; lane nn-conv3d)."""
    var out: List[Int] = [1, 1, 1, 0, 1, 1]
    return out^


def _words(block: List[Int32], depth: List[Int]) -> List[Int]:
    """The descriptor of a conv or pool node: x_cnn's parameter block, then its RN_DEPTH_LEN depth words."""
    var words = List[Int](capacity=len(block) + RN_DEPTH_LEN)
    for v in block:  # small-loop(len(block): the CP_/PP_ parameter block, at most 18 words)
        words.append(Int(v))
    for v in depth:  # small-loop(RN_DEPTH_LEN = 6 depth words)
        words.append(v)
    return words^


def _shape5(a: Int, b: Int, c: Int, d: Int, e: Int) -> ShapeN:
    var dims = Array[Int, RN_MAX_RANK](fill=1)
    dims[0] = a
    dims[1] = b
    dims[2] = c
    dims[3] = d
    dims[4] = e
    return ShapeN(5, dims)


def _conv_out(N: Int, OC: Int, OD: Int, OH: Int, OW: Int, vol: Bool) -> ShapeN:
    if vol:
        return _shape5(N, OC, OD, OH, OW)
    return _shape4(N, OC, OH, OW)


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
    return _rankn_conv(tape, ex, x, weight, bias, stride, stride, padding, padding, groups)


def rankn_conv1d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: Int = 1, padding: Int = 0, groups: Int = 1,
) raises -> Tensor:
    """One-dimensional convolution (cross-correlation) of a batch of sequences.

    The same node as rankn_conv2d over (N, C, 1, L) with a (OC, C // groups, 1, K) kernel,
    step (1, stride) and pads (0, padding): one im2col, one GEMM chain over the C // groups K
    terms, one output launch, and the conv2d backward.

    Args:
        tape: The tape that records the operation.
        ex: The executor.
        x: The input, shape (N, C, L).
        weight: The kernels, shape (out_channels, C // groups, K).
        bias: Optional, out_channels values added to each output channel.
        stride: The step between windows.
        padding: Zeros added at both ends of each sequence.
        groups: The number of channel groups; it divides C and out_channels.

    Returns:
        The output, shape (N, out_channels, OL), OL = (L + 2 padding - K) // stride + 1.
    """
    var sx = tape.shape_of(x)
    var sw = tape.shape_of(weight)
    if sx.ndim != 3:
        raise Error("conv1d: x must be a rank-3 tensor")
    if sw.ndim != 3:
        raise Error("conv1d: weight must be a rank-3 tensor")
    var x4 = tape.view_nd(ex, x, _shape4(sx.d[0], sx.d[1], 1, sx.d[2]))
    var w4 = tape.view_nd(ex, weight, _shape4(sw.d[0], sw.d[1], 1, sw.d[2]))
    var y4 = _rankn_conv(tape, ex, x4, w4, bias, 1, stride, 0, padding, groups)
    var sy = tape.shape_of(y4)
    return tape.view_nd(ex, y4, _shape3(sy.d[0], sy.d[1], sy.d[3]))


def _rankn_conv[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor],
    stride_h: Int, stride_w: Int, padding_h: Int, padding_w: Int, groups: Int,
    dilation_h: Int = 1, dilation_w: Int = 1, stride_d: Int = 1, padding_d: Int = 0, dilation_d: Int = 1,
) raises -> Tensor:
    """rankn_conv2d with the height and width steps and pads given apart (x_cnn's CP_* words hold
    them apart); the one convolution of the tape, used by conv1d, conv2d and conv3d. A rank-5 x
    (N, C, D, H, W) with a rank-5 weight (OC, C // groups, KD, KH, KW) is the 3-D convolution
    (lane nn-conv3d): the same im2col, GEMM and output launches with the depth words set."""
    var sx = tape.shape_of(x)
    var sw = tape.shape_of(weight)
    var vol = sx.ndim == 5
    var what = String("conv3d") if vol else String("conv2d")
    if sx.ndim != 4 and not vol:
        raise Error("conv2d: x must be a rank-4 tensor")
    if sw.ndim != sx.ndim:
        raise Error(what + ": weight must be a rank-" + String(sx.ndim) + " tensor")
    var o = 1 if vol else 0
    var N = sx.d[0]
    var C = sx.d[1]
    var D = sx.d[2] if vol else 1
    var H = sx.d[2 + o]
    var W = sx.d[3 + o]
    var OC = sw.d[0]
    var KD = sw.d[2] if vol else 1
    var KH = sw.d[2 + o]
    var KW = sw.d[3 + o]
    if groups < 1:
        raise Error(what + ": groups must be positive")
    if groups == 1 and sw.d[1] != C:
        raise Error(what + ": weight's second dimension must equal x's channels")
    if groups > 1:
        if C % groups != 0 or OC % groups != 0:
            raise Error(what + ": groups must divide the input channels and out_channels")
        if sw.d[1] * groups != C:
            raise Error(what + ": weight's second dimension must equal x's channels // groups")
    if stride_h <= 0 or stride_w <= 0 or padding_h < 0 or padding_w < 0 or stride_d <= 0 or padding_d < 0:
        raise Error(what + ": stride must be positive and padding non-negative")
    if dilation_h < 1 or dilation_w < 1 or dilation_d < 1:
        raise Error(what + ": dilation must be positive")
    var has_bias = Bool(bias)
    if has_bias:
        var b = bias.value()
        if tape.rows(b) * tape.cols(b) != OC:
            raise Error(what + ": bias must hold out_channels values")
    var raw: List[Int] = [N, C, H, W, OC, KH, KW, stride_h, stride_w, padding_h, padding_w, dilation_h,
                          dilation_w, 0, 0, 1 if has_bias else 0, 0]
    var block = conv_params(raw)
    var depth = rn_flat_depth()
    var OD = 1
    if vol:
        # the depth's output size by the same rule (x_cnn conv_params over the depth as its height)
        var rawd: List[Int] = [N, C, D, 1, OC, KD, 1, stride_d, 1, padding_d, 0, dilation_d, 1, 0, 0, 0, 0]
        OD = Int(conv_params(rawd)[CP_OH])
        depth = [D, KD, stride_d, padding_d, dilation_d, OD]
    var OH = Int(block[CP_OH])
    var OW = Int(block[CP_OW])
    var rows = N * OD * OH * OW
    var ckk = C * KD * KH * KW
    var desc = tape.descriptor(ex, _words(block, depth))
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
        var fdesc = tape.descriptor(ex, rn_conv_group_words(RNG_FWD, rows, C, OC, KD * KH * KW, groups))
        rn_conv_group_gemm(ex, y2, cols, wv, fdesc, rows * OC)
    # NCHW / NCDHW output (+ bias)
    var out = tape.reserve(ex, N * OC * OD * OH * OW)
    var o = Args()
    o.p0 = y2
    if has_bias:
        o.p1 = bv
    o.p2 = out
    o.p11 = desc
    ex.launch[OP_RNV_CONV_OUT](o, N * OC * OD * OH * OW)
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
            saved.p6 = tape.descriptor(ex, rn_conv_group_words(RNG_DW, rows, C, OC, KD * KH * KW, groups))
        if need_x:
            saved.p7 = tape.descriptor(ex, rn_conv_group_words(RNG_DCOLS, rows, C, OC, KD * KH * KW, groups))
    saved.p11 = desc
    saved.i0 = 1 if has_bias else 0
    saved.i1 = rows
    saved.i2 = ckk
    saved.i3 = OC
    saved.i4 = sx.count()
    saved.i5 = 1 if need_x else 0
    saved.i6 = 1 if need_w else 0
    saved.i7 = 1 if need_b else 0
    saved.i8 = groups
    var inputs: List[Tensor] = [x, weight]
    if has_bias:
        inputs.append(bias.value())
    return tape.custom_nd(ex, inputs, out, _conv_out(N, OC, OD, OH, OW, vol), RN_CONV2D, saved)


def _pool_block(sx: ShapeN, kernel_size: Int, stride: Int, padding: Int) raises -> List[Int32]:
    if kernel_size <= 0 or stride < 0 or padding < 0:
        raise Error("pool: kernel_size must be positive, stride and padding non-negative")
    var s = kernel_size if stride == 0 else stride
    return _pool_block_hw(sx, kernel_size, kernel_size, s, s, padding, padding)


def _pool_block_hw(sx: ShapeN, kh: Int, kw: Int, sh: Int, sw: Int, ph: Int, pw: Int) raises -> List[Int32]:
    # [N, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW, count_include_pad,
    #  reverse, ceil_mode, divisor_override]
    var raw: List[Int] = [sx.d[0], sx.d[1], sx.d[2], sx.d[3], kh, kw, sh, sw,
                          ph, pw, 1, 1, 0, 0, 1, 0, 0, 0]
    return pool_params(raw)


def _pool_block_1d(sx: ShapeN, kernel_size: Int, stride: Int, padding: Int) raises -> List[Int32]:
    """The PP_* words of a 1-D pool over (N, C, 1, L): window (1, kernel_size),
    step (1, stride; 0 means kernel_size), pads (0, padding)."""
    if kernel_size <= 0 or stride < 0 or padding < 0:
        raise Error("pool1d: kernel_size must be positive, stride and padding non-negative")
    if 2 * padding > kernel_size:
        raise Error("pool1d: padding must be at most half of kernel_size")
    var s = kernel_size if stride == 0 else stride
    return _pool_block_hw(sx, 1, kernel_size, 1, s, 0, padding)


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
    return _max_pool(tape, ex, x, sx, _pool_block(sx, kernel_size, stride, padding), rn_flat_depth())


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
    return _avg_pool(tape, ex, x, sx, _pool_block(sx, kernel_size, stride, padding), rn_flat_depth())


def _max_pool[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, sx: ShapeN, block: List[Int32], depth: List[Int],
) raises -> Tensor:
    """The max pool node of x (N, C, H, W), or of x (N, C, D, H, W) with its depth words (lane nn-conv3d)."""
    var OH = Int(block[PP_OH])
    var OW = Int(block[PP_OW])
    var OD = depth[5]
    var count = sx.d[0] * sx.d[1] * OD * OH * OW
    var desc = tape.descriptor(ex, _words(block, depth))
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
    return tape.custom_nd(ex, [x], out, _conv_out(sx.d[0], sx.d[1], OD, OH, OW, sx.ndim == 5), RN_MAXPOOL2D, saved)


def _avg_pool[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, sx: ShapeN, block: List[Int32], depth: List[Int],
) raises -> Tensor:
    """The average pool node of x (N, C, H, W), or of x (N, C, D, H, W) with its depth words (lane nn-conv3d)."""
    var OH = Int(block[PP_OH])
    var OW = Int(block[PP_OW])
    var OD = depth[5]
    var count = sx.d[0] * sx.d[1] * OD * OH * OW
    var desc = tape.descriptor(ex, _words(block, depth))
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
    return tape.custom_nd(ex, [x], out, _conv_out(sx.d[0], sx.d[1], OD, OH, OW, sx.ndim == 5), RN_AVGPOOL2D, saved)


def _shape3(a: Int, b: Int, c: Int) -> ShapeN:
    var dims = Array[Int, RN_MAX_RANK](fill=1)
    dims[0] = a
    dims[1] = b
    dims[2] = c
    return ShapeN(3, dims)


def rankn_max_pool1d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, kernel_size: Int, stride: Int = 0, padding: Int = 0,
) raises -> Tensor:
    """One-dimensional max pooling of a batch of sequences.

    Each output is the largest value of its kernel_size window; padded
    positions are never chosen, and of tied values the first receives the
    gradient. The max_pool2d node over (N, C, 1, L).

    Args:
        tape: The tape that records the operation.
        ex: The executor.
        x: The input, shape (N, C, L).
        kernel_size: The window's length.
        stride: The step between windows; 0 means kernel_size.
        padding: Positions added at both ends (at most half of kernel_size).

    Returns:
        The output, shape (N, C, OL).
    """
    var s3 = tape.shape_of(x)
    if s3.ndim != 3:
        raise Error("max_pool1d: x must be a rank-3 tensor")
    var sx = _shape4(s3.d[0], s3.d[1], 1, s3.d[2])
    var x4 = tape.view_nd(ex, x, sx)
    var y4 = _max_pool(tape, ex, x4, sx, _pool_block_1d(sx, kernel_size, stride, padding), rn_flat_depth())
    var sy = tape.shape_of(y4)
    return tape.view_nd(ex, y4, _shape3(sy.d[0], sy.d[1], sy.d[3]))


def rankn_avg_pool1d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, kernel_size: Int, stride: Int = 0, padding: Int = 0,
) raises -> Tensor:
    """One-dimensional average pooling of a batch of sequences: each output is
    the mean of its kernel_size window, the zero padding counted in the
    window's size. The avg_pool2d node over (N, C, 1, L).

    Args:
        tape: The tape that records the operation.
        ex: The executor.
        x: The input, shape (N, C, L).
        kernel_size: The window's length.
        stride: The step between windows; 0 means kernel_size.
        padding: Zeros added at both ends (at most half of kernel_size).

    Returns:
        The output, shape (N, C, OL).
    """
    var s3 = tape.shape_of(x)
    if s3.ndim != 3:
        raise Error("avg_pool1d: x must be a rank-3 tensor")
    var sx = _shape4(s3.d[0], s3.d[1], 1, s3.d[2])
    var x4 = tape.view_nd(ex, x, sx)
    var y4 = _avg_pool(tape, ex, x4, sx, _pool_block_1d(sx, kernel_size, stride, padding), rn_flat_depth())
    var sy = tape.shape_of(y4)
    return tape.view_nd(ex, y4, _shape3(sy.d[0], sy.d[1], sy.d[3]))


def rankn_batch_norm[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, gamma: Tensor, beta: Tensor, running: FP, eps: Float32,
    momentum: Float32, train: Bool,
) raises -> Tensor:
    """Batch normalization over the channel axis of x (N, C, ...): every channel's values over the
    batch and the remaining axes are normalized, then scaled by gamma and shifted by beta (C each).

    Training (`train`): the batch's mean and biased variance normalize, and the running
    statistics `running` ([mean C | var C], executor storage) take PyTorch's update
    r = (1 - momentum) r + momentum stat, the variance unbiased. Otherwise the running
    statistics normalize. x_cnn's BatchNorm bodies (autograd/rankn_conv_ops.mojo
    OP_RNV_BN_*): launches, training 6 (prep, block sums, mean, block squares, variance with
    the running update, apply), eval 2 (prep with the running statistics, apply); the
    backward 3 (block reductions, their totals with d gamma and d beta, dx). Recorded as
    RN_BATCH_NORM (autograd/rules_conv2d.mojo)."""
    var s = tape.shape_of(x)
    if s.ndim < 2:
        raise Error("batch norm: x must be (N, C, ...)")
    var N = s.d[0]
    var C = s.d[1]
    var count = s.count()
    var HW = count // (N * C)
    if tape.rows(gamma) * tape.cols(gamma) != C or tape.rows(beta) * tape.cols(beta) != C:
        raise Error("batch norm: gamma and beta must hold one value per channel")
    if train and N * HW < 2:
        raise Error("batch norm: training needs more than one value per channel")
    var words: List[Int] = [N, C, HW]
    var desc = tape.descriptor(ex, words)
    var xv = tape.value_of(ex, x)
    var aux = tape.reserve(ex, 2 + BN_SLOTS * C)
    var pr = Args()
    pr.p0 = tape.value_of(ex, gamma)
    pr.p1 = aux
    pr.p2 = running
    pr.p4 = tape.value_of(ex, beta)
    pr.p11 = desc
    pr.f0 = eps
    pr.f1 = momentum
    pr.i0 = 0 if train else 1
    ex.launch[OP_RNV_BN_PREP](pr, C)
    var NB = bn_fold_blocks(N * HW)
    var part = tape.reserve(ex, 2 * C * NB)
    if train:
        var a = Args()
        a.p0 = xv
        a.p1 = part
        a.p11 = desc
        ex.launch[OP_RNV_BN_SUM](a, C * NB)
        var m = Args()
        m.p0 = part
        m.p1 = aux
        m.p11 = desc
        ex.launch[OP_RNV_BN_MEAN](m, C)
        var q = Args()
        q.p0 = xv
        q.p1 = part
        q.p2 = aux
        q.p11 = desc
        ex.launch[OP_RNV_BN_SQ](q, C * NB)
        var v = Args()
        v.p0 = part
        v.p1 = aux
        v.p2 = running
        v.p11 = desc
        ex.launch[OP_RNV_BN_VAR](v, C)
    var out = tape.reserve(ex, count)
    var ap = Args()
    ap.p0 = xv
    ap.p1 = aux
    ap.p2 = out
    ap.p11 = desc
    ex.launch[OP_RNV_BN_APPLY](ap, count)
    var rec = tape.is_recording()
    var saved = Args()
    saved.p0 = xv
    saved.p1 = aux
    saved.p2 = part
    saved.p11 = desc
    saved.i0 = count
    saved.i1 = C * NB
    saved.i2 = C
    saved.i3 = 1 if train else 0
    saved.i5 = 1 if rec and tape.requires_grad(x) else 0
    saved.i6 = 1 if rec and tape.requires_grad(gamma) else 0
    saved.i7 = 1 if rec and tape.requires_grad(beta) else 0
    var inputs: List[Tensor] = [x, gamma, beta]
    return tape.custom_nd(ex, inputs, out, s, RN_BATCH_NORM, saved)

