# SPDX-License-Identifier: Apache-2.0
"""The 3-D layers of a Sequential description (lane nn-conv3d, 2026-11-23): layer codes 73 to 76.

  code  layer                 arguments (a0 .. a5; a triple is dhw_pack's one integer)     tape op
  73    conv3d                out_channels, kernel, stride, padding, bias + 2 groups,        rankn_conv3d
                              dilation
  74    max_pool3d            kernel, stride, padding, dilation, ceil_mode                   rankn_max_pool3d
  75    avg_pool3d            kernel, stride, padding, count_include_pad, ceil_mode,         rankn_avg_pool3d
                              divisor_override (0 none)
  76    adaptive_avg_pool3d   output size                                                   rankn_adaptive_avg_pool3d

The input of each is a (channels, depth, height, width) sample. autograd/model_blocks.mojo answers these
codes through the functions here (its `# nn-conv3d` groups); the ops are autograd/nn_conv3d.mojo, the
tape's one convolution and pool nodes with their depth words, and their backward is Conv2dRules (reached
through ModelRules -> ImportRules -> RankNRules), so no rule is added. Parameters: a conv3d's weight
(out_channels, channels // groups, kd, kh, kw) then its bias (out_channels), PyTorch's layout and init
(uniform in +-1/sqrt(fan_in), fan_in = channels // groups * kd kh kw, for both).
"""
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_api import rn_shape
from autograd.nn_conv3d import (
    Dhw, rankn_conv3d, rankn_max_pool3d, rankn_avg_pool3d, rankn_adaptive_avg_pool3d, adaptive3d_supported,
    _pool3d_block,
)
from autograd.rankn_conv import _shape5
from sequence.exec_trait import Exec
from sequence.ops import FP
from checks.numerics import ftz, identical_div, identical_sqrt
from x_cnn.ops import PP_OH, PP_OW

comptime LAYER_CONV3D = 73
comptime LAYER_MAXPOOL3D = 74
comptime LAYER_AVGPOOL3D = 75
comptime LAYER_ADAPTIVE_AVGPOOL3D = 76
comptime _ARGS = 6


def is_conv3d_layer(code: Int) -> Bool:
    return code >= LAYER_CONV3D and code <= LAYER_ADAPTIVE_AVGPOOL3D


def _x(xargs: List[Int], i: Int, k: Int) -> Int:
    return xargs[_ARGS * i + k]


def _t(xargs: List[Int], i: Int, k: Int) -> Dhw:
    return Dhw.unpack(_x(xargs, i, k))


def conv3d_bias(xargs: List[Int], i: Int) -> Bool:
    return (_x(xargs, i, 4) & 1) == 1


def conv3d_groups(xargs: List[Int], i: Int) -> Int:
    var g = _x(xargs, i, 4) >> 1
    return g if g > 0 else 1


def _sample5(cur: ShapeN) -> ShapeN:
    return _shape5(1, cur.d[0], cur.d[1], cur.d[2], cur.d[3])


def _out_size(size: Int, k: Int, p: Int, s: Int, dil: Int) -> Int:
    return (size + 2 * p - dil * (k - 1) - 1) // s + 1


# ---------------------------------------------------------------- description
def conv3d_layer_shape(code: Int, xargs: List[Int], i: Int, cur: ShapeN, name: String) raises -> ShapeN:
    """Validate 3-D layer i on its per-sample input shape `cur` (C, D, H, W); its output shape."""
    if cur.ndim != 4:
        raise Error(name + ": 3-D convolution and pooling take a (channels, depth, height, width) input")
    var C = cur.d[0]
    var D = cur.d[1]
    var H = cur.d[2]
    var W = cur.d[3]
    var dims = List[Int]()
    if code == LAYER_CONV3D:
        var oc = _x(xargs, i, 0)
        var k = _t(xargs, i, 1)
        var s = _t(xargs, i, 2)
        var p = _t(xargs, i, 3)
        var dl = _t(xargs, i, 5)
        var bias = _x(xargs, i, 4) & 1
        var groups = conv3d_groups(xargs, i)
        if oc < 1 or k.d < 1 or k.h < 1 or k.w < 1 or s.d < 1 or s.h < 1 or s.w < 1:
            raise Error(name + " (Conv3d): out_channels, kernel_size and stride must be positive")
        if dl.d < 1 or dl.h < 1 or dl.w < 1:
            raise Error(name + " (Conv3d): dilation must be positive")
        if bias != 0 and bias != 1:
            raise Error(name + " (Conv3d): bias must be 0 or 1")
        if C % groups != 0 or oc % groups != 0:
            raise Error(name + " (Conv3d): groups must divide the input channels and out_channels")
        var od = _out_size(D, k.d, p.d, s.d, dl.d)
        var oh = _out_size(H, k.h, p.h, s.h, dl.h)
        var ow = _out_size(W, k.w, p.w, s.w, dl.w)
        if od < 1 or oh < 1 or ow < 1:
            raise Error(name + " (Conv3d): the kernel is larger than the (padded) input")
        dims = [oc, od, oh, ow]
    elif code == LAYER_MAXPOOL3D or code == LAYER_AVGPOOL3D:
        var is_max = code == LAYER_MAXPOOL3D
        var what = name + (" (MaxPool3d)" if is_max else " (AvgPool3d)")
        var dil = _t(xargs, i, 3) if is_max else Dhw.cube(1)
        var inc = True if is_max else _x(xargs, i, 3) == 1
        var ceil = _x(xargs, i, 4)
        if ceil != 0 and ceil != 1:
            raise Error(what + ": ceil_mode must be 0 or 1")
        if not is_max and _x(xargs, i, 3) != 0 and _x(xargs, i, 3) != 1:
            raise Error(what + ": count_include_pad must be 0 or 1")
        var div = 0 if is_max else _x(xargs, i, 5)
        var bd = _pool3d_block(_sample5(cur), _t(xargs, i, 0), _t(xargs, i, 1), _t(xargs, i, 2), dil, ceil == 1,
                               inc, div, what)
        dims = [C, bd[1][5], Int(bd[0][PP_OH]), Int(bd[0][PP_OW])]
    elif code == LAYER_ADAPTIVE_AVGPOOL3D:
        var o = _t(xargs, i, 0)
        if not adaptive3d_supported(D, H, W, o):
            raise Error(name + " (AdaptiveAvgPool3d): output_size must be 1 or divide the input's ("
                        + String(D) + ", " + String(H) + ", " + String(W) + ")")
        dims = [C, o.d, o.h, o.w]
    else:
        raise Error(name + ": unknown 3-D layer code " + String(code))
    return rn_shape(dims, name)


def _fan_in(xargs: List[Int], i: Int, cur: ShapeN) -> Int:
    return (cur.d[0] // conv3d_groups(xargs, i)) * _t(xargs, i, 1).volume()


def conv3d_layer_sizes(code: Int, xargs: List[Int], i: Int, cur: ShapeN) -> List[Int]:
    """The element counts of layer i's parameter tensors (a conv3d's weight, then its bias)."""
    var out = List[Int]()
    if code == LAYER_CONV3D:
        var oc = _x(xargs, i, 0)
        out.append(oc * _fan_in(xargs, i, cur))
        if conv3d_bias(xargs, i):
            out.append(oc)
    return out^


def conv3d_layer_names(code: Int) -> List[String]:
    var out = List[String]()
    if code == LAYER_CONV3D:
        out.append(".weight")
        out.append(".bias")
    return out^


def conv3d_layer_bounds(code: Int, xargs: List[Int], i: Int, cur: ShapeN) -> List[Float32]:
    """PyTorch's Conv3d init: weight and bias uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)] (kaiming_uniform
    with a = sqrt(5) gives the weight that bound)."""
    var out = List[Float32]()
    if code == LAYER_CONV3D:
        var b = ftz(identical_div(Float32(1), identical_sqrt(Float32(_fan_in(xargs, i, cur)))))
        out.append(b)
        if conv3d_bias(xargs, i):
            out.append(b)
    return out^


def conv3d_layer_dims(code: Int, xargs: List[Int], i: Int, cur: ShapeN, slot: Int) -> List[Int]:
    """The shape of a conv3d's tensor `slot` (PyTorch's layout is params_' own)."""
    var out = List[Int]()
    if code == LAYER_CONV3D:
        out.append(_x(xargs, i, 0))
        if slot == 0:
            var k = _t(xargs, i, 1)
            out.append(cur.d[0] // conv3d_groups(xargs, i))
            out.append(k.d)
            out.append(k.h)
            out.append(k.w)
    return out^


# ---------------------------------------------------------------- forward
def _p[E: Exec](mut tape: Tape, mut ex: E, P: FP, Gr: FP, off: Int, rows: Int, cols: Int, train: Bool) raises -> Tensor:
    if train:
        return tape.variable_into(ex, P + off, Gr + off, rows, cols)
    return tape.variable(ex, P + off, rows, cols, False)


def conv3d_layer_forward[E: Exec](
    mut tape: Tape, mut ex: E, code: Int, xargs: List[Int], i: Int, cur: ShapeN, P: FP, Gr: FP, off: Int,
    h: Tensor, train: Bool,
) raises -> Tensor:
    """Record 3-D layer i on the tape from h (B,) + cur."""
    var B = tape.rows(h) * tape.cols(h) // cur.count()
    var x = tape.view_nd(ex, h, _shape5(B, cur.d[0], cur.d[1], cur.d[2], cur.d[3]))
    if code == LAYER_CONV3D:
        var oc = _x(xargs, i, 0)
        var k = _t(xargs, i, 1)
        var cg = cur.d[0] // conv3d_groups(xargs, i)
        var nw = oc * cg * k.volume()
        var w2 = _p(tape, ex, P, Gr, off, oc, cg * k.volume(), train)
        var w5 = tape.view_nd(ex, w2, _shape5(oc, cg, k.d, k.h, k.w))
        var bias = Optional[Tensor]()
        if conv3d_bias(xargs, i):
            bias = Optional[Tensor](_p(tape, ex, P, Gr, off + nw, 1, oc, train))
        return rankn_conv3d(tape, ex, x, w5, bias, _t(xargs, i, 2), _t(xargs, i, 3), _t(xargs, i, 5),
                            conv3d_groups(xargs, i))
    if code == LAYER_MAXPOOL3D:
        return rankn_max_pool3d(tape, ex, x, _t(xargs, i, 0), _t(xargs, i, 1), _t(xargs, i, 2), _t(xargs, i, 3),
                                _x(xargs, i, 4) == 1)
    if code == LAYER_AVGPOOL3D:
        return rankn_avg_pool3d(tape, ex, x, _t(xargs, i, 0), _t(xargs, i, 1), _t(xargs, i, 2), _x(xargs, i, 4) == 1,
                                _x(xargs, i, 3) == 1, _x(xargs, i, 5))
    if code == LAYER_ADAPTIVE_AVGPOOL3D:
        return rankn_adaptive_avg_pool3d(tape, ex, x, _t(xargs, i, 0))
    raise Error("model: unknown 3-D layer code " + String(code))
