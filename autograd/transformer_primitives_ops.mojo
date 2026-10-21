# SPDX-License-Identifier: Apache-2.0
# Native scalar bodies shared by HostExec and DeviceExec. Forward seams match
# transformer/impl/llama/modeling_llama.mojo S1..S10; no float algebra rewrites.
from sequence.ops import Args, FP
from checks.numerics import ftz, identical_mul, identical_mul_add, identical_div, identical_rsqrt, identical_sin, identical_cos
from transformer.experiments.norm_profile_contract import NN24_NORM_LANES8, NN24_LANES, norm_profile_dot
from gemm.contract import contract_leaf_size, leaf_count
from gemm.experiments.neural_profile import NEURAL_PROFILE_CHANGED, NEURAL_LEAF, NEURAL_CHAINS, neural_partition, neural_cell

comptime OP_TFP_FIRST = 4700
comptime OP_TFP_RMS = 4700
comptime OP_TFP_RMS_DX = 4701
comptime OP_TFP_RMS_DW = 4702
comptime OP_TFP_ROPE_TABLE = 4703
comptime OP_TFP_ROPE = 4704
comptime OP_TFP_LINEAR = 4705
comptime OP_TFP_LAST = 4705


def _linear(cell: Int, a: Args):
    # Same neural GEMM partition/profile as the specialized Transformer. The
    # ordinary sequence matmul has a different reduction contract and cannot
    # substitute for this model's leaf + balanced-tree projection arithmetic.
    var leaf = contract_leaf_size(a.i2)
    var leaves = leaf_count(a.i2, leaf)
    comptime if NEURAL_PROFILE_CHANGED:
        var part = neural_partition[NEURAL_LEAF](a.i2)
        leaf = part[0]
        leaves = part[1]
    var value = neural_cell[NEURAL_CHAINS](a.p0, a.p1, cell // a.i1, cell % a.i1,
        a.i2, leaf, leaves, a.i3, a.i4, a.i5, a.i6)
    a.p2.unsafe_store(cell, value)


def _rms(row: Int, a: Args):
    var d = a.i0
    var base = row * d
    var acc = Float32(0)
    for j in range(d):  # device-loop: pinned ascending fused sum of squares
        var x = ftz(a.p0.unsafe_load(base + j))
        acc = ftz(identical_mul_add(x, x, acc))
    comptime if NN24_NORM_LANES8:
        acc = norm_profile_dot[NN24_LANES](a.p0, a.p0, base, base, d)
    var rstd = ftz(identical_rsqrt(ftz(ftz(identical_div(acc, Float32(d))) + a.f0)))
    a.p3.unsafe_store(row, rstd)
    for j in range(d):  # device-loop: two separate rounded products per feature
        var inner = ftz(identical_mul(ftz(a.p0.unsafe_load(base + j)), rstd))
        a.p2.unsafe_store(base + j, ftz(identical_mul(ftz(a.p1.unsafe_load(j)), inner)))


def _rms_dx(row: Int, a: Args):
    var d = a.i0
    var base = row * d
    var dot = Float32(0)
    for j in range(d):  # device-loop: native row derivative and fused dot order
        var dh = ftz(identical_mul(ftz(a.p2.unsafe_load(base + j)), ftz(a.p1.unsafe_load(j))))
        a.p5.unsafe_store(base + j, dh)
        dot = ftz(identical_mul_add(dh, ftz(a.p0.unsafe_load(base + j)), dot))
    comptime if NN24_NORM_LANES8:
        dot = norm_profile_dot[NN24_LANES](a.p5, a.p0, base, base, d)
    var rstd = ftz(a.p4.unsafe_load(row))
    var r2 = ftz(identical_mul(rstd, rstd))
    var r3 = ftz(identical_mul(r2, rstd))
    var dv = ftz(identical_div(ftz(identical_mul(Float32(-0.5), ftz(identical_mul(dot, r3)))), Float32(d)))
    for j in range(d):  # device-loop: same unfused dx1 + dx2 seams as the block
        var dx1 = ftz(identical_mul(ftz(a.p5.unsafe_load(base + j)), rstd))
        var twice_x = ftz(identical_mul(Float32(2), ftz(a.p0.unsafe_load(base + j))))
        var dx2 = ftz(identical_mul(dv, twice_x))
        a.p3.unsafe_store(base + j, ftz(dx1 + dx2))


def _rms_dw(column: Int, a: Args):
    var acc = Float32(0)
    for row in range(a.i1):  # device-loop: graph weight gradient, ascending rows
        var at = row * a.i0 + column
        var inner = ftz(identical_mul(ftz(a.p0.unsafe_load(at)), ftz(a.p4.unsafe_load(row))))
        var term = ftz(identical_mul(ftz(a.p2.unsafe_load(at)), inner))
        acc = ftz(identical_mul_add(Float32(1), term, acc))
    a.p3.unsafe_store(column, acc)


def _rope_table(cell: Int, a: Args):
    var half = a.i0
    var pos = cell // half + a.i1
    var frequency = cell % half
    var angle = ftz(identical_mul(Float32(pos), ftz(a.p0.unsafe_load(frequency))))
    a.p1.unsafe_store(cell, ftz(identical_cos(angle)))
    a.p2.unsafe_store(cell, ftz(identical_sin(angle)))


def _rope(cell: Int, a: Args):
    var width = a.i0
    var length = a.i1
    var hd = a.i2
    var half = hd // 2
    var j = cell % hd
    var table_at = ((cell // width) % length) * half + (j % half)
    var paired = cell + half if j < half else cell - half
    var rotated = ftz(a.p0.unsafe_load(paired))
    if (j < half) != (a.i3 != 0):
        rotated = -rotated
    var own = ftz(identical_mul(ftz(a.p0.unsafe_load(cell)), ftz(a.p1.unsafe_load(table_at))))
    var other = ftz(identical_mul(rotated, ftz(a.p2.unsafe_load(table_at))))
    a.p3.unsafe_store(cell, ftz(own + other))


def apply_transformer_primitive[OP: Int](cell: Int, a: Args):
    comptime if OP == OP_TFP_RMS:
        _rms(cell, a)
    elif OP == OP_TFP_RMS_DX:
        _rms_dx(cell, a)
    elif OP == OP_TFP_RMS_DW:
        _rms_dw(cell, a)
    elif OP == OP_TFP_ROPE_TABLE:
        _rope_table(cell, a)
    elif OP == OP_TFP_ROPE:
        _rope(cell, a)
    elif OP == OP_TFP_LINEAR:
        _linear(cell, a)
