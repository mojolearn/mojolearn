# SPDX-License-Identifier: Apache-2.0
# Capturable native RMSNorm and full-width Llama rotary embeddings. These
# native rules add first derivatives; higher-order admission remains explicit.
from std.math import isfinite
from sequence.exec_trait import Exec
from sequence.ops import Args, FP
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.transformer_primitives_ops import OP_TFP_RMS, OP_TFP_RMS_DX, OP_TFP_RMS_DW, OP_TFP_ROPE_TABLE, OP_TFP_ROPE, OP_TFP_LINEAR
from checks.numerics import ftz, identical_div, identical_mul, portable_powf

# The import-family tags occupy 192..207; keep these disjoint.
comptime GRAPH_RMS_NORM = 208
comptime GRAPH_ROTARY = 209
comptime GRAPH_TRANSFORMER_LINEAR = 210


@fieldwise_init
struct TransformerPrimitiveRules(BackwardRules):
    def supports(self, tag: Int) -> Bool:
        return tag == GRAPH_RMS_NORM or tag == GRAPH_ROTARY or tag == GRAPH_TRANSFORMER_LINEAR

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
                      inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
        var a = saved
        if tag == GRAPH_TRANSFORMER_LINEAR:
            a.p0 = upstream
            a.p1 = inputs[1]
            a.p2 = gradients[0]
            a.i0 = rows[0]
            a.i1 = cols[0]
            a.i2 = rows[1]
            a.i3 = rows[1]
            a.i4 = 1
            a.i5 = cols[0]
            a.i6 = 1
            ex.launch[OP_TFP_LINEAR](a, rows[0] * cols[0])
            a.p1 = inputs[0]
            a.p2 = gradients[1]
            a.i0 = rows[1]
            a.i1 = cols[1]
            a.i2 = rows[0]
            a.i3 = 1
            a.i4 = rows[1]
            a.i5 = cols[0]
            ex.launch[OP_TFP_LINEAR](a, rows[1] * cols[1])
            return
        if tag == GRAPH_ROTARY:
            a.p0 = upstream
            a.p1 = saved.p4
            a.p2 = saved.p5
            a.p3 = gradients[0]
            a.i3 = 1
            ex.launch[OP_TFP_ROPE](a, rows[0] * cols[0])
            return
        if tag != GRAPH_RMS_NORM:
            raise Error("unknown transformer primitive backward rule")
        a.p0 = inputs[0]
        a.p1 = inputs[1]
        a.p2 = upstream
        a.p3 = gradients[0]
        a.p4 = saved.p4
        a.p5 = saved.p5
        a.i1 = rows[0]
        ex.launch[OP_TFP_RMS_DX](a, rows[0])
        a.p3 = gradients[1]
        ex.launch[OP_TFP_RMS_DW](a, cols[0])


def graph_rms_norm[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor, eps: Float32) raises -> Tensor:
    var shape = tape.shape_of(x)
    var d = tape.cols(x)
    var m = tape.rows(x)
    if tape.rows(weight) != 1 or tape.cols(weight) != d:
        raise Error("ValueError: RMSNorm weight must have one value per last-axis feature")
    if not eps > 0 or not isfinite(eps):
        raise Error("ValueError: RMSNorm epsilon must be positive and finite")
    var xp = tape.value_of(ex, x)
    var wp = tape.value_of(ex, weight)
    var output = tape.reserve(ex, m * d)
    var rstd = tape.reserve(ex, m)
    var scratch = tape.reserve(ex, m * d)
    var a = Args()
    a.p0 = xp
    a.p1 = wp
    a.p2 = output
    a.p3 = rstd
    a.i0 = d
    a.f0 = eps
    ex.launch[OP_TFP_RMS](a, m)
    a.p4 = rstd
    a.p5 = scratch
    return tape.custom_nd(ex, [x, weight], output, shape, GRAPH_RMS_NORM, a)


def graph_rotary[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, hd: Int, offset: Int, theta: Float32) raises -> Tensor:
    var shape = tape.shape_of(x)
    if shape.ndim < 2:
        raise Error("ValueError: rotary input needs sequence and feature axes")
    var width = tape.cols(x)
    var length = shape.d[shape.ndim - 2]
    if hd <= 0 or hd % 2 != 0 or width % hd != 0:
        raise Error("ValueError: rotary head_dim must be positive, even and divide the feature width")
    if offset < 0 or offset > 2147483647 - length:
        raise Error("ValueError: rotary position offset is out of range")
    if not theta > 0 or not isfinite(theta):
        raise Error("ValueError: rotary theta must be positive and finite")
    var half = hd // 2
    var inv = List[Float32]()
    for j in range(half):  # cpu-route: native rotary frequency constants, shared host contract S6
        var exponent = ftz(identical_div(Float32(2 * j), Float32(hd)))
        var frequency = ftz(identical_div(Float32(1), portable_powf(theta, exponent)))
        var angle = ftz(identical_mul(Float32(offset + length - 1), frequency))
        if not isfinite(angle) or angle >= Float32(8192):
            raise Error("ValueError: rotary angle outside the supported domain [0, 8192)")
        inv.append(frequency)
    var ip = tape.reserve(ex, half)
    ex.upload(ip, FP(unsafe_from_address=Int(inv.unsafe_ptr())), half)
    _ = inv^  # host upload source remains alive through the supported transfer
    var cos = tape.reserve(ex, length * half)
    var sin = tape.reserve(ex, length * half)
    var a = Args()
    a.p0 = ip
    a.p1 = cos
    a.p2 = sin
    a.i0 = half
    a.i1 = offset
    ex.launch[OP_TFP_ROPE_TABLE](a, length * half)
    var xp = tape.value_of(ex, x)
    var output = tape.reserve(ex, tape.rows(x) * width)
    a.p0 = xp
    a.p1 = cos
    a.p2 = sin
    a.p3 = output
    a.i0 = width
    a.i1 = length
    a.i2 = hd
    a.i3 = 0
    ex.launch[OP_TFP_ROPE](a, tape.rows(x) * width)
    a.p4 = cos
    a.p5 = sin
    return tape.custom_nd(ex, [x], output, shape, GRAPH_ROTARY, a)


def graph_transformer_linear[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor) raises -> Tensor:
    var shape = tape.shape_of(x)
    var weight_shape = tape.shape_of(weight)
    if shape.ndim < 1 or weight_shape.ndim != 2 or tape.cols(x) != tape.cols(weight):
        raise Error("ValueError: transformer projection needs (..., inputs) and (outputs, inputs) weights")
    var a = Args()
    a.i0 = tape.rows(x)
    a.i1 = tape.rows(weight)
    a.i2 = tape.cols(x)
    a.i3 = a.i2
    a.i4 = 1
    a.i5 = 1
    a.i6 = a.i2
    a.p0 = tape.value_of(ex, x)
    a.p1 = tape.value_of(ex, weight)
    var output = tape.reserve(ex, a.i0 * a.i1)
    a.p2 = output
    ex.launch[OP_TFP_LINEAR](a, a.i0 * a.i1)
    shape.d[shape.ndim - 1] = a.i1
    return tape.custom_nd(ex, [x, weight], output, shape, GRAPH_TRANSFORMER_LINEAR, a)
