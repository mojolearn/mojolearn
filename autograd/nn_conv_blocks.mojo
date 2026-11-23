# SPDX-License-Identifier: Apache-2.0
"""The convolution and shape layers of a Sequential description (lane nn-conv, 2026-11-23).

autograd/model_blocks.mojo answers these codes by one call each into this file (its `# nn-conv` groups),
so the shared tables hold one line per entry point:

  code  layer                 arguments (a0..a5; f)                         op (autograd/nn_conv.mojo)
  56    conv_transpose1d      out_channels, kernel, stride, padding,         nn_conv_transpose1d
                              output_padding, groups * 2 + bias; f dilation
  57    conv_transpose2d      the same, square kernel                        nn_conv_transpose2d
  58    adaptive_avg_pool2d   out height, out width (0 keeps the input's)    nn_adaptive_avg_pool2d
  59    adaptive_max_pool2d   out height, out width (0 keeps the input's)    nn_adaptive_max_pool2d
  60    adaptive_max_pool1d   out length                                     nn_adaptive_max_pool1d
  61    global_max_pool       none ((C, L) or (C, H, W) to (C,))             nn_global_max_pool
  62    constant_pad          left, right, top, bottom, dims; f value         nn_pad mode 0 (imp_pad)
  63    reflection_pad        left, right, top, bottom, dims                  nn_pad reflect
  64    replication_pad       left, right, top, bottom, dims                  nn_pad replicate
  65    upsample              out height, out width / length (0: from the     nn_upsample
                              scale), mode (0 nearest, 1 linear / bilinear),
                              align_corners, dims; f scale_factor (0: size)
  66    pixel_shuffle         factor, 0 shuffle / 1 unshuffle                 nn_pixel_shuffle

Only the transposed convolutions have parameters: weight (in_channels, out_channels // groups, kernel
[, kernel]) and an optional bias (out_channels), PyTorch's layout and init (uniform in +-1 / sqrt(fan_in),
fan_in = out_channels // groups * kernel^dims, both tensors).
"""
from std.math import isfinite
from checks.numerics import ftz, identical_div, identical_sqrt
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_shape, rn_dims
from autograd.nn_conv import (
    nn_conv_transpose1d, nn_conv_transpose2d, nn_adaptive_avg_pool2d, nn_adaptive_max_pool2d,
    nn_adaptive_max_pool1d, nn_global_max_pool, nn_pad, nn_upsample, nn_pixel_shuffle, up_scale,
    NNC_PAD_REFLECT, NNC_PAD_REPLICATE,
)

comptime LAYER_CONV_TRANSPOSE1D = 56
comptime LAYER_CONV_TRANSPOSE2D = 57
comptime LAYER_ADAPTIVE_AVGPOOL2D = 58
comptime LAYER_ADAPTIVE_MAXPOOL2D = 59
comptime LAYER_ADAPTIVE_MAXPOOL1D = 60
comptime LAYER_GLOBAL_MAXPOOL = 61
comptime LAYER_CONSTANT_PAD = 62
comptime LAYER_REFLECTION_PAD = 63
comptime LAYER_REPLICATION_PAD = 64
comptime LAYER_UPSAMPLE = 65
comptime LAYER_PIXEL_SHUFFLE = 66
comptime NNC_FIRST = 56
comptime NNC_LAST = 66

comptime _ARGS = 6


def is_nn_conv(code: Int) -> Bool:
    return code >= NNC_FIRST and code <= NNC_LAST


def _x(xargs: List[Int], i: Int, k: Int) -> Int:
    return xargs[_ARGS * i + k]


def _ct(code: Int) -> Bool:
    return code == LAYER_CONV_TRANSPOSE1D or code == LAYER_CONV_TRANSPOSE2D


def nnc_is_projection(code: Int) -> Bool:
    """True for a layer that counts as the model's learned projection (a transposed convolution, as a
    convolution's output layer of an autoencoder)."""
    return _ct(code)


def _ct_groups(xargs: List[Int], i: Int) -> Int:
    return _x(xargs, i, 5) // 2


def _ct_bias(xargs: List[Int], i: Int) -> Bool:
    return _x(xargs, i, 5) % 2 == 1


def _ct_dims(code: Int) -> Int:
    return 1 if code == LAYER_CONV_TRANSPOSE1D else 2


def _ct_out(isize: Int, k: Int, s: Int, p: Int, op: Int, d: Int) -> Int:
    return (isize - 1) * s - 2 * p + d * (k - 1) + op + 1


def _up_out(isize: Int, size: Int, f: Float32) -> Int:
    if size > 0:
        return size
    # PyTorch's floor(input_size * scale_factor), in float64
    return Int(Float64(isize) * Float64(f))


def _sh(dims: List[Int], name: String) raises -> ShapeN:
    return rn_shape(dims, name)


# ---------------------------------------------------------------- description
def nnc_shape(code: Int, xargs: List[Int], i: Int, f: Float32, cur: ShapeN, name: String) raises -> ShapeN:
    """Validate layer i on its per-sample input shape `cur`; its output shape."""
    var a0 = _x(xargs, i, 0)
    var a1 = _x(xargs, i, 1)
    if _ct(code):
        var dims = _ct_dims(code)
        var what = name + (" (ConvTranspose1d)" if dims == 1 else " (ConvTranspose2d)")
        if cur.ndim != dims + 1:
            raise Error(what + ": the input must be (channels, length)" if dims == 1
                        else what + ": the input must be (channels, height, width)")
        var k = a1
        var s = _x(xargs, i, 2)
        var p = _x(xargs, i, 3)
        var op = _x(xargs, i, 4)
        var g = _ct_groups(xargs, i)
        var dl = Int(f)
        if a0 < 1 or k < 1 or s < 1 or p < 0 or g < 1 or not (f >= Float32(1)) or Float32(dl) != f:
            raise Error(what + ": out_channels, kernel_size, stride, groups and dilation must be positive "
                        + "integers and padding >= 0")
        if op < 0 or op >= s:
            raise Error(what + ": output_padding must be smaller than stride")
        if cur.d[0] % g != 0 or a0 % g != 0:
            raise Error(what + ": groups must divide the input channels and out_channels")
        var od = List[Int]()
        od.append(a0)
        for ax in range(1, cur.ndim):  # small-loop(one or two spatial axes): shape metadata
            var o = _ct_out(cur.d[ax], k, s, p, op, dl)
            if o < 1:
                raise Error(what + ": the output would be empty (padding too large)")
            od.append(o)
        return _sh(od, what)
    if code == LAYER_ADAPTIVE_AVGPOOL2D or code == LAYER_ADAPTIVE_MAXPOOL2D:
        var what = name + (" (AdaptiveAvgPool2d)" if code == LAYER_ADAPTIVE_AVGPOOL2D else " (AdaptiveMaxPool2d)")
        if cur.ndim != 3:
            raise Error(what + ": the input must be (channels, height, width)")
        if a0 < 0 or a1 < 0:
            raise Error(what + ": output_size must be positive (or None to keep the input's)")
        var od: List[Int] = [cur.d[0], a0 if a0 > 0 else cur.d[1], a1 if a1 > 0 else cur.d[2]]
        return _sh(od, what)
    if code == LAYER_ADAPTIVE_MAXPOOL1D:
        if cur.ndim != 2:
            raise Error(name + " (AdaptiveMaxPool1d): the input must be (channels, length)")
        if a0 < 1:
            raise Error(name + " (AdaptiveMaxPool1d): output_size must be positive")
        var od: List[Int] = [cur.d[0], a0]
        return _sh(od, name + " (AdaptiveMaxPool1d)")
    if code == LAYER_GLOBAL_MAXPOOL:
        if cur.ndim != 2 and cur.ndim != 3:
            raise Error(name + " (GlobalMaxPool): the input must be (channels, length) or (channels, height, width)")
        var od: List[Int] = [cur.d[0]]
        return _sh(od, name + " (GlobalMaxPool)")
    if code == LAYER_CONSTANT_PAD or code == LAYER_REFLECTION_PAD or code == LAYER_REPLICATION_PAD:
        var what = name + (" (ConstantPad)" if code == LAYER_CONSTANT_PAD else
                           (" (ReflectionPad)" if code == LAYER_REFLECTION_PAD else " (ReplicationPad)"))
        var dims = _x(xargs, i, 4)
        if dims != 1 and dims != 2:
            raise Error(what + ": dims must be 1 or 2")
        if cur.ndim != dims + 1:
            raise Error(what + ": the input must be (channels, length)" if dims == 1
                        else what + ": the input must be (channels, height, width)")
        var top = _x(xargs, i, 2)
        var bottom = _x(xargs, i, 3)
        if dims == 1 and (top != 0 or bottom != 0):
            raise Error(what + ": a 1-D pad has only left and right")
        if code == LAYER_CONSTANT_PAD:
            if not isfinite(f):
                raise Error(what + ": value must be finite")
        elif a0 < 0 or a1 < 0 or top < 0 or bottom < 0:
            raise Error(what + ": padding must be non-negative")
        var W = cur.d[cur.ndim - 1]
        var H = cur.d[1] if dims == 2 else 1
        if code == LAYER_REFLECTION_PAD and (a0 >= W or a1 >= W or top >= H or bottom >= H):
            raise Error(what + ": padding must be smaller than the padded dimension")
        var od = rn_dims(cur)
        od[cur.ndim - 1] = W + a0 + a1
        if dims == 2:
            od[1] = H + top + bottom
        if od[cur.ndim - 1] < 1 or (dims == 2 and od[1] < 1):
            raise Error(what + ": the output would be empty")
        return _sh(od, what)
    if code == LAYER_UPSAMPLE:
        var dims = _x(xargs, i, 4)
        var mode = _x(xargs, i, 2)
        var align = _x(xargs, i, 3)
        var what = name + " (Upsample)"
        if dims != 1 and dims != 2:
            raise Error(what + ": dims must be 1 or 2")
        if cur.ndim != dims + 1:
            raise Error(what + ": the input must be (channels, length)" if dims == 1
                        else what + ": the input must be (channels, height, width)")
        if (mode != 0 and mode != 1) or (align != 0 and align != 1) or (mode == 0 and align == 1):
            raise Error(what + ": mode must be nearest or (bi)linear; align_corners only with (bi)linear")
        var sized = a1 > 0
        if sized == (f > Float32(0)):
            raise Error(what + ": give either size or scale_factor")
        if not sized and not isfinite(f):
            raise Error(what + ": scale_factor must be finite")
        var od = rn_dims(cur)
        od[cur.ndim - 1] = _up_out(cur.d[cur.ndim - 1], a1, f)
        if dims == 2:
            od[1] = _up_out(cur.d[1], a0, f)
        if od[cur.ndim - 1] < 1 or od[1] < 1:
            raise Error(what + ": the output would be empty")
        return _sh(od, what)
    if code == LAYER_PIXEL_SHUFFLE:
        var what = name + (" (PixelUnshuffle)" if a1 == 1 else " (PixelShuffle)")
        if cur.ndim != 3:
            raise Error(what + ": the input must be (channels, height, width)")
        if a0 < 1 or (a1 != 0 and a1 != 1):
            raise Error(what + ": the factor must be positive")
        if a1 == 0:
            if cur.d[0] % (a0 * a0) != 0:
                raise Error(what + ": the channels must be divisible by upscale_factor squared")
            var od: List[Int] = [cur.d[0] // (a0 * a0), cur.d[1] * a0, cur.d[2] * a0]
            return _sh(od, what)
        if cur.d[1] % a0 != 0 or cur.d[2] % a0 != 0:
            raise Error(what + ": height and width must be divisible by downscale_factor")
        var ud: List[Int] = [cur.d[0] * a0 * a0, cur.d[1] // a0, cur.d[2] // a0]
        return _sh(ud, what)
    raise Error(name + ": unknown layer code " + String(code))


def _ct_taps(code: Int, xargs: List[Int], i: Int) -> Int:
    var k = _x(xargs, i, 1)
    return k if code == LAYER_CONV_TRANSPOSE1D else k * k


def nnc_param_sizes(code: Int, xargs: List[Int], i: Int, cur: ShapeN) -> List[Int]:
    var out = List[Int]()
    if _ct(code):
        var oc = _x(xargs, i, 0)
        out.append(cur.d[0] * (oc // _ct_groups(xargs, i)) * _ct_taps(code, xargs, i))
        if _ct_bias(xargs, i):
            out.append(oc)
    return out^


def nnc_param_names(code: Int) -> List[String]:
    """The checkpoint names of nnc_param_sizes' tensors (a layer without a bias uses the first only)."""
    var out = List[String]()
    if _ct(code):
        out.append(".weight")
        out.append(".bias")
    return out^


def nnc_init_bounds(code: Int, xargs: List[Int], i: Int, cur: ShapeN) -> List[Float32]:
    """PyTorch's ConvTranspose init: weight and bias uniform in +-1 / sqrt(fan_in), fan_in =
    out_channels // groups * kernel^dims (the weight's second dimension times its receptive field)."""
    var out = List[Float32]()
    if _ct(code):
        var fan = (_x(xargs, i, 0) // _ct_groups(xargs, i)) * _ct_taps(code, xargs, i)
        var b = ftz(identical_div(Float32(1), identical_sqrt(Float32(fan))))
        out.append(b)
        if _ct_bias(xargs, i):
            out.append(b)
    return out^


def nnc_tensor_dims(code: Int, xargs: List[Int], i: Int, cur: ShapeN, slot: Int) -> List[Int]:
    """The shape of the layer's tensor `slot`, the same in PyTorch's layout and params_' own."""
    var out = List[Int]()
    if _ct(code):
        if slot == 0:
            var k = _x(xargs, i, 1)
            out.append(cur.d[0])
            out.append(_x(xargs, i, 0) // _ct_groups(xargs, i))
            out.append(k)
            if code == LAYER_CONV_TRANSPOSE2D:
                out.append(k)
        else:
            out.append(_x(xargs, i, 0))
    return out^


# ---------------------------------------------------------------- forward
def _p[E: Exec](mut tape: Tape, mut ex: E, P: FP, Gr: FP, off: Int, rows: Int, cols: Int, train: Bool) raises -> Tensor:
    if train:
        return tape.variable_into(ex, P + off, Gr + off, rows, cols)
    return tape.variable(ex, P + off, rows, cols, False)


def nnc_forward[E: Exec](
    mut tape: Tape, mut ex: E, code: Int, xargs: List[Int], i: Int, f: Float32, cur: ShapeN,
    P: FP, Gr: FP, off: Int, h: Tensor, train: Bool,
) raises -> Tensor:
    """Record layer i on the tape from h ((B,) + cur)."""
    var B = tape.rows(h) * tape.cols(h) // cur.count()
    var bd = List[Int]()
    bd.append(B)
    for k in range(cur.ndim):  # small-loop(the sample rank, at most 3 here): shape metadata
        bd.append(cur.d[k])
    var x = tape.view_nd(ex, h, rn_shape(bd, "model"))
    var a0 = _x(xargs, i, 0)
    var a1 = _x(xargs, i, 1)
    if _ct(code):
        var C = cur.d[0]
        var g = _ct_groups(xargs, i)
        var taps = _ct_taps(code, xargs, i)
        var ocg = a0 // g
        var w2 = _p(tape, ex, P, Gr, off, C, ocg * taps, train)
        var wd = List[Int]()
        wd.append(C)
        wd.append(ocg)
        wd.append(a1)
        if code == LAYER_CONV_TRANSPOSE2D:
            wd.append(a1)
        var w = tape.view_nd(ex, w2, rn_shape(wd, "model"))
        var bias = Optional[Tensor]()
        if _ct_bias(xargs, i):
            var b2 = _p(tape, ex, P, Gr, off + C * ocg * taps, 1, a0, train)
            var bdims: List[Int] = [a0]
            bias = Optional[Tensor](tape.view_nd(ex, b2, rn_shape(bdims, "model")))
        var s = _x(xargs, i, 2)
        var p = _x(xargs, i, 3)
        var op = _x(xargs, i, 4)
        var dl = Int(f)
        if code == LAYER_CONV_TRANSPOSE1D:
            return nn_conv_transpose1d(tape, ex, x, w, bias, s, p, op, dl, g)
        return nn_conv_transpose2d(tape, ex, x, w, bias, s, s, p, p, op, op, dl, dl, g)
    if code == LAYER_ADAPTIVE_AVGPOOL2D:
        return nn_adaptive_avg_pool2d(tape, ex, x, a0 if a0 > 0 else cur.d[1], a1 if a1 > 0 else cur.d[2])
    if code == LAYER_ADAPTIVE_MAXPOOL2D:
        return nn_adaptive_max_pool2d(tape, ex, x, a0 if a0 > 0 else cur.d[1], a1 if a1 > 0 else cur.d[2])
    if code == LAYER_ADAPTIVE_MAXPOOL1D:
        return nn_adaptive_max_pool1d(tape, ex, x, a0)
    if code == LAYER_GLOBAL_MAXPOOL:
        return nn_global_max_pool(tape, ex, x)
    if code == LAYER_CONSTANT_PAD or code == LAYER_REFLECTION_PAD or code == LAYER_REPLICATION_PAD:
        var mode = 0 if code == LAYER_CONSTANT_PAD else (
            NNC_PAD_REFLECT if code == LAYER_REFLECTION_PAD else NNC_PAD_REPLICATE)
        return nn_pad(tape, ex, x, a0, a1, _x(xargs, i, 2), _x(xargs, i, 3), mode,
                      f if code == LAYER_CONSTANT_PAD else Float32(0))
    if code == LAYER_UPSAMPLE:
        var nearest = _x(xargs, i, 2) == 0
        var align = _x(xargs, i, 3) == 1
        var sf = Float64(f) if a1 <= 0 else Float64(0)
        var two = cur.ndim == 3
        var H = cur.d[1] if two else 1
        var W = cur.d[cur.ndim - 1]
        var OW = _up_out(W, a1, f)
        var OH = _up_out(H, a0, f) if two else 1
        var sh = up_scale(H, OH, sf if two else Float64(0), align, nearest)
        var sw = up_scale(W, OW, sf, align, nearest)
        var x4 = x
        if not two:
            var d4: List[Int] = [B, cur.d[0], 1, W]
            x4 = tape.view_nd(ex, x, rn_shape(d4, "model"))
        var y = nn_upsample(tape, ex, x4, OH, OW, sh, sw, not nearest, align)
        if two:
            return y
        var d3: List[Int] = [B, cur.d[0], OW]
        return tape.view_nd(ex, y, rn_shape(d3, "model"))
    if code == LAYER_PIXEL_SHUFFLE:
        return nn_pixel_shuffle(tape, ex, x, a0, a1 == 1)
    raise Error("model: unknown nn-conv code " + String(code))
