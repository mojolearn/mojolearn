# SPDX-License-Identifier: Apache-2.0
# The 3-D convolution and pools of the tape (lane nn-conv3d, 2026-11-23): conv3d, max_pool3d,
# avg_pool3d and adaptive_avg_pool3d on rank-5 (N, C, D, H, W) tensors, PyTorch's semantics.
#
# There is no 3-D implementation here: each op is the tape's ONE convolution or pool node
# (autograd/rankn_conv.mojo _rankn_conv, _max_pool, _avg_pool), whose bodies
# (autograd/rankn_conv_ops.mojo) carry a depth axis in the RN_DEPTH_* words after x_cnn's
# parameter block. The backward is the same rule (autograd/rules_conv2d.mojo Conv2dRules tags
# RN_CONV2D, RN_MAXPOOL2D, RN_AVGPOOL2D). Host work here is shape metadata only.
#
# LAUNCHES (each a parallel element launch, one thread a word):
#   conv3d        forward 3 (im2col, GEMM, output + bias); backward up to 5 (grow copy, dW GEMM,
#                 db blocked fold, dcols GEMM, col2im) plus the fold's. groups > 1: the GEMMs are
#                 one grouped OP_RN_BMM each.
#   max_pool3d    forward 1 (values and winners); backward 1.
#   avg_pool3d    forward 1; backward 1.
#   adaptive_avg_pool3d  output (1, 1, 1): tape.mean's blocked fold over D H W (its launches);
#                 output sizes that divide the input: avg_pool3d with kernel = stride = in / out.
#
# ORDER. The sums are the GEMM's chain over the C KD KH KW taps in im2col order (kd, kh, kw
# ascending inside each channel), col2im's and the pools' gathers in (kd, kh, kw) ascending order
# from +0.0: one order on the host column and every vendor.
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_conv import _rankn_conv, _max_pool, _avg_pool, _shape5
from sequence.exec_trait import Exec
from x_cnn.ops import pool_params, PP_OH


#: one axis's value of a packed (d, h, w) triple: 10 bits each (Sequential's description holds a
#: conv3d's kernel, stride, padding and dilation in one integer argument each)
comptime DHW_BITS = 10
comptime DHW_MASK = (1 << DHW_BITS) - 1


@fieldwise_init
struct Dhw(ImplicitlyCopyable, Movable):
    """A (depth, height, width) triple: a window, a step, a padding or a dilation."""
    var d: Int
    var h: Int
    var w: Int

    @staticmethod
    def cube(v: Int) -> Self:
        return Self(v, v, v)

    @staticmethod
    def unpack(v: Int) -> Self:
        """The triple dhw_pack wrote."""
        return Self(v & DHW_MASK, (v >> DHW_BITS) & DHW_MASK, (v >> (2 * DHW_BITS)) & DHW_MASK)

    def volume(self) -> Int:
        return self.d * self.h * self.w


def dhw_pack(d: Int, h: Int, w: Int) raises -> Int:
    """One integer holding (d, h, w), each in 0 .. 1023."""
    if d < 0 or h < 0 or w < 0 or d > DHW_MASK or h > DHW_MASK or w > DHW_MASK:
        raise Error("3-D layer: each of a triple's values must be in 0 .. 1023")
    return d | (h << DHW_BITS) | (w << (2 * DHW_BITS))


def _dims5(tape: Tape, x: Tensor, what: String) raises -> ShapeN:
    var s = tape.shape_of(x)
    if s.ndim != 5:
        raise Error(what + ": x must be a rank-5 (N, C, D, H, W) tensor")
    return s


def rankn_conv3d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor], stride: Dhw, padding: Dhw,
    dilation: Dhw, groups: Int = 1,
) raises -> Tensor:
    """Three-dimensional convolution (cross-correlation) of a batch of volumes, as PyTorch's
    `conv3d` with zero padding.

    Args:
        tape: The tape that records the operation.
        ex: The executor.
        x: The input, shape (N, C, D, H, W).
        weight: The kernels, shape (out_channels, C // groups, kd, kh, kw).
        bias: Optional, out_channels values added to each output channel.
        stride: The steps between windows along depth, height and width.
        padding: Zeros added at both ends of each axis.
        dilation: The spacing between kernel taps along each axis.
        groups: The number of channel groups; it divides C and out_channels.

    Returns:
        The output, shape (N, out_channels, OD, OH, OW), OD = (D + 2 pd - dd (kd - 1) - 1) // sd + 1
        and the others likewise.
    """
    _ = _dims5(tape, x, "conv3d")
    return _rankn_conv(tape, ex, x, weight, bias, stride.h, stride.w, padding.h, padding.w, groups,
                       dilation.h, dilation.w, stride.d, padding.d, dilation.d)


def _pool3d_block(
    sx: ShapeN, kernel: Dhw, stride: Dhw, padding: Dhw, dilation: Dhw, ceil_mode: Bool, include_pad: Bool,
    divisor: Int, what: String, mut depth: List[Int],
) raises -> List[Int32]:
    """x_cnn's PP_* block of the (H, W) plane; `depth` receives the depth words [D, KD, SD, PD, DD, OD], the depth's
    output size by the same rule (pool_params over the depth as its height: PyTorch's
    pooling_output_shape, floor or ceil mode, and its padding bound)."""
    if kernel.d < 1 or kernel.h < 1 or kernel.w < 1 or stride.d < 1 or stride.h < 1 or stride.w < 1:
        raise Error(what + ": kernel_size and stride must be positive")
    if padding.d < 0 or padding.h < 0 or padding.w < 0:
        raise Error(what + ": padding must be non-negative")
    if 2 * padding.d > kernel.d or 2 * padding.h > kernel.h or 2 * padding.w > kernel.w:
        raise Error(what + ": padding must be at most half of kernel_size")
    if dilation.d < 1 or dilation.h < 1 or dilation.w < 1:
        raise Error(what + ": dilation must be positive")
    if divisor < 0:
        raise Error(what + ": divisor_override must be positive")
    var N = sx.d[0]
    var C = sx.d[1]
    var inc = 1 if include_pad else 0
    var ceil = 1 if ceil_mode else 0
    var raw: List[Int] = [N, C, sx.d[3], sx.d[4], kernel.h, kernel.w, stride.h, stride.w, padding.h, padding.w,
                          dilation.h, dilation.w, 0, 0, inc, 0, ceil, divisor]
    var block = pool_params(raw)
    var rawd: List[Int] = [N, C, sx.d[2], 1, kernel.d, 1, stride.d, 1, padding.d, 0, dilation.d, 1, 0, 0, inc, 0,
                           ceil, 0]
    var od = Int(pool_params(rawd)[PP_OH])
    depth = [sx.d[2], kernel.d, stride.d, padding.d, dilation.d, od]
    return block^


def rankn_max_pool3d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, kernel: Dhw, stride: Dhw, padding: Dhw, dilation: Dhw,
    ceil_mode: Bool = False,
) raises -> Tensor:
    """Three-dimensional max pooling of a batch of volumes, as PyTorch's `max_pool3d`.

    Each output is the largest value of its window; padded positions are never chosen, a NaN
    wins, and of tied values the first in (depth, height, width) order receives the gradient.

    Args:
        tape: The tape that records the operation.
        ex: The executor.
        x: The input, shape (N, C, D, H, W).
        kernel: The window along depth, height and width.
        stride: The steps between windows.
        padding: Positions added at both ends of each axis (at most half of the window).
        dilation: The spacing between window taps.
        ceil_mode: Round the output sizes up (PyTorch's rule: the last window starts inside the
            input or its leading padding).

    Returns:
        The output, shape (N, C, OD, OH, OW).
    """
    var sx = _dims5(tape, x, "max_pool3d")
    var depth = List[Int]()
    var block = _pool3d_block(sx, kernel, stride, padding, dilation, ceil_mode, True, 0, "max_pool3d", depth)
    return _max_pool(tape, ex, x, sx, block, depth)


def rankn_avg_pool3d[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, kernel: Dhw, stride: Dhw, padding: Dhw, ceil_mode: Bool = False,
    count_include_pad: Bool = True, divisor_override: Int = 0,
) raises -> Tensor:
    """Three-dimensional average pooling of a batch of volumes, as PyTorch's `avg_pool3d`.

    Each output is the window's sum divided by the window's size: the size clipped to the padded
    input, counting the zero padding when count_include_pad (PyTorch's default), or only the input
    positions otherwise; divisor_override, when positive, is the divisor of every window.

    Args:
        tape: The tape that records the operation.
        ex: The executor.
        x: The input, shape (N, C, D, H, W).
        kernel: The window along depth, height and width.
        stride: The steps between windows.
        padding: Zeros added at both ends of each axis (at most half of the window).
        ceil_mode: Round the output sizes up (PyTorch's rule).
        count_include_pad: Count the padding in each window's size.
        divisor_override: The divisor of every window when positive; 0 for none.

    Returns:
        The output, shape (N, C, OD, OH, OW).
    """
    var sx = _dims5(tape, x, "avg_pool3d")
    var depth = List[Int]()
    var block = _pool3d_block(sx, kernel, stride, padding, Dhw.cube(1), ceil_mode, count_include_pad,
                              divisor_override, "avg_pool3d", depth)
    return _avg_pool(tape, ex, x, sx, block, depth)


def adaptive3d_supported(D: Int, H: Int, W: Int, size: Dhw) -> Bool:
    """True when adaptive_avg_pool3d takes this input and output: the output (1, 1, 1), or output sizes
    that divide the input's (PyTorch's windows are then the equal tiles avg_pool3d reads)."""
    if size.d < 1 or size.h < 1 or size.w < 1:
        return False
    if size.d == 1 and size.h == 1 and size.w == 1:
        return True
    return D % size.d == 0 and H % size.h == 0 and W % size.w == 0


def rankn_adaptive_avg_pool3d[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, output: Dhw) raises -> Tensor:
    """Adaptive average pooling of a batch of volumes to (OD, OH, OW), as PyTorch's
    `adaptive_avg_pool3d` for an output of (1, 1, 1) (the global average) or output sizes that divide
    the input's (each output the mean of one equal tile).

    Args:
        tape: The tape that records the operation.
        ex: The executor.
        x: The input, shape (N, C, D, H, W).
        output: The output sizes along depth, height and width.

    Returns:
        The output, shape (N, C, OD, OH, OW).
    """
    var sx = _dims5(tape, x, "adaptive_avg_pool3d")
    var D = sx.d[2]
    var H = sx.d[3]
    var W = sx.d[4]
    if not adaptive3d_supported(D, H, W, output):
        raise Error("adaptive_avg_pool3d: the output sizes must be 1 or divide the input's ("
                    + String(D) + ", " + String(H) + ", " + String(W) + ")")
    if output.d == 1 and output.h == 1 and output.w == 1:
        # the global mean of each (sample, channel) row: one blocked fold over D H W (the (N C, D H W)
        # matrix view), parallel over rows and blocks
        var flat = tape.view_nd(ex, x, ShapeN.matrix(sx.d[0] * sx.d[1], D * H * W))
        var m = tape.mean(ex, flat, 1)
        return tape.view_nd(ex, m, _shape5(sx.d[0], sx.d[1], 1, 1, 1))
    var k = Dhw(D // output.d, H // output.h, W // output.w)
    return rankn_avg_pool3d(tape, ex, x, k, k, Dhw.cube(0), False, True, 0)
