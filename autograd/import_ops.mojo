# SPDX-License-Identifier: Apache-2.0
# THE IMPORT ELEMENTWISE FAMILY (lane ag-import-kernels, docs/plans/
# MODEL_IMPORT.md section 3.3): the element bodies an imported ONNX or
# torch.export graph needs beyond the tape's own (erf, gelu, softplus, silu,
# leaky relu, elu, hard sigmoid / swish, clip, abs, floor, ceil, round, sin,
# cos, sign, reciprocal, logical not, pow; the broadcasting binary kinds pow,
# minimum, maximum, where, the comparisons and logical and / or; argmax's
# position words; constant pad; lane ag-import-lm-ops: the nonzero test of
# any / all, cumsum's blocked scan and the flat ids of a multi-tensor index).
# Shared by HostExec and DeviceExec through
# sequence/dispatch.mojo, so the host column and every GPU run the same
# bodies.
#
# Codes OP_IMP_FIRST..OP_IMP_LAST (4600..4619) are this family;
# sequence/dispatch.mojo routes and fences them. 4600..4609 are used,
# 4610..4619 are reserved for this family.
#
# Every body is one thread per output word with no order to declare: a word
# is a function of the words it reads, through ld / st (ftz) and the
# identical_* numerics of checks/numerics.mojo, as autograd/ops.mojo's unary
# body does. floor, ceil, round (half to even), abs, sign, the comparisons,
# where, minimum and maximum are exact. The only sums are a broadcast
# operand's gradient, which the tape calls fold with OP_RN_REDUCE_PART and
# the blocked fold (autograd/import_api.mojo), order (F) of
# autograd/rankn_fold_ops.mojo.
#
# THE SCAN ORDER (cumsum, OP_IMP_SCAN_TOTALS / OP_IMP_SCAN_FINAL, lane
# ag-import-lm-ops). A scan along an axis of length L is cut into leaves of
# IMP_SCAN_LEAF = 64 consecutive positions in scan order (the reversed order
# for a reverse scan). Within a leaf the terms are added in ascending scan
# order, left to right, one add at a time (add: ftz). The words of an
# output at scan position j are:
#   L <= 64:  ((v0 + v1) + v2) ... + vj
#   L > 64:   Q[b - 1] + (((v[64b] + v[64b + 1]) ... + vj)), b = j // 64,
#             where T[b] is leaf b's total (its 64 terms in the same order)
#             and Q is the inclusive scan of T by this same rule
#             (recursively, one level per factor of 64); leaf 0 adds no Q.
# An exclusive scan's word at j is the inclusive word at j - 1 (0 at j = 0).
# Every word is one thread's sum of at most 64 terms plus one add, so no
# sequential chain runs along the axis; the order is the same on every
# executor and every launch geometry. The gradient is the scan of the
# upstream in the other direction, same exclusive flag, same rule.
#
# THE FLAT IDS DESCRIPTOR (OP_IMP_FLAT_IDS, a multi-tensor index): word 0
# nk (the broadcast index shape's rank), word 1 the index tensor's base
# offset, then nk records of IMP_IDS_REC = 2 words [extent, stride] (stride
# 0 on an axis the index tensor broadcasts over). Index values are float
# words holding integers; a negative one counts from the end of its axis.
# A NaN, a fraction or an index outside its axis makes the id -1, which the
# embedding's id check refuses before any row is read.
#
# THE BINARY DESCRIPTOR (int32 bit patterns in tape storage, read with
# rn_word, never ld; bound to Args.p11):
#   word 0  nk: axis records (outermost first), the output's rank
#   word 1  0 (reserved)
#   words 2, 3, 4  base offsets of operands a, b and c in their buffers
#   then nk records of IMP_BIN_REC = 5 words, one an axis:
#     [extent, output stride, a stride, b stride, c stride]
#   (a stride is 0 on an axis the operand broadcasts over).
# A linear index decodes over the records innermost (last record) first: a
# per-thread loop over at most 8 records (the tensor rank): shape metadata,
# no data, and no host loop.
#
# THE PAD DESCRIPTOR: word 0 nd, word 1 the source's base offset, then nd
# records of IMP_PAD_REC = 4 words. Forward: [output extent, pad before,
# source extent, source stride]. VJP: [source extent, pad before, output
# row-major stride, output extent]. A negative pad crops.
#
# Metal rules (autograd/ops.mojo _ld_right): an operand bound once for two
# roles is read through a branch on a flag, never through a pointer select;
# every function a body reaches is @always_inline.
from std.math import floor, ceil
from std.memory import bitcast
from sequence.ops import Args, FP, ld, st, add, sub, mul
from autograd.ops import div
from autograd.rankn import rn_word
from checks.numerics import (
    ftz, identical_clamp, identical_cos, identical_erf, identical_exp, identical_fmax,
    identical_gelu_erf, identical_gelu_tanh, identical_log, identical_pow, identical_sigmoid,
    identical_silu, identical_sin, identical_softplus, identical_tanh,
)

comptime OP_IMP_UNARY_FWD = 4600
comptime OP_IMP_UNARY_VJP = 4601
comptime OP_IMP_BIN_FWD = 4602
comptime OP_IMP_BIN_VJP = 4603
comptime OP_IMP_ARGMAX = 4604
comptime OP_IMP_PAD = 4605
comptime OP_IMP_PAD_VJP = 4606
comptime OP_IMP_SCAN_TOTALS = 4607
comptime OP_IMP_SCAN_FINAL = 4608
comptime OP_IMP_FLAT_IDS = 4609
comptime OP_IMP_FIRST = 4600
comptime OP_IMP_LAST = 4619

# Unary kinds (Args.i1 of OP_IMP_UNARY_*; f0, f1 their parameters).
comptime IMP_ERF = 1
comptime IMP_GELU_ERF = 2
comptime IMP_GELU_TANH = 3
comptime IMP_SOFTPLUS = 4
comptime IMP_SILU = 5
comptime IMP_LEAKY_RELU = 6      # f0 alpha
comptime IMP_ELU = 7             # f0 alpha
comptime IMP_HARD_SIGMOID = 8    # f0 alpha, f1 beta
comptime IMP_HARD_SWISH = 9      # f0 alpha, f1 beta (1/6 and 1/2)
comptime IMP_CLIP = 10           # f0 low, f1 high
comptime IMP_ABS = 11
comptime IMP_FLOOR = 12
comptime IMP_CEIL = 13
comptime IMP_ROUND = 14          # half to even
comptime IMP_SIN = 15
comptime IMP_COS = 16
comptime IMP_SIGN = 17
comptime IMP_RECIPROCAL = 18
comptime IMP_NOT = 19            # 1 where x == 0, else 0
comptime IMP_POW_SCALAR = 20     # f0 the exponent
comptime IMP_BOOL = 21           # 1 where x != 0 (NaN included), else 0
comptime IMP_UNARY_LAST = 21

# Binary kinds (Args.i1 of OP_IMP_BIN_*).
comptime IMP_POW = 1
comptime IMP_MINIMUM = 2
comptime IMP_MAXIMUM = 3
comptime IMP_WHERE = 4           # c != 0 ? a : b (c the condition)
comptime IMP_EQ = 5
comptime IMP_GT = 6
comptime IMP_LT = 7
comptime IMP_GE = 8
comptime IMP_LE = 9
comptime IMP_NE = 10
comptime IMP_AND = 11
comptime IMP_OR = 12
comptime IMP_BINARY_LAST = 12

# argmax output forms (Args.i1 of OP_IMP_ARGMAX)
comptime IMP_ARGMAX_INT32 = 0    # int32 words in float storage
comptime IMP_ARGMAX_FLOAT = 1    # the position as a float32 value

comptime IMP_BIN_HEAD = 5
comptime IMP_BIN_REC = 5
comptime IMP_PAD_HEAD = 2
comptime IMP_PAD_REC = 4
comptime IMP_IDS_HEAD = 2
comptime IMP_IDS_REC = 2
# the scan's leaf length (THE SCAN ORDER above); fixed, so cumsum's words
# never depend on the fold lane's settings
comptime IMP_SCAN_LEAF = 64

# derivative constants, float32 roundings of the closed forms
comptime IMP_TWO_OVER_SQRT_PI = Float32(1.1283791670955126)
comptime IMP_INV_SQRT2 = Float32(0.7071067811865476)
comptime IMP_INV_SQRT_2PI = Float32(0.3989422804014327)
comptime IMP_GELU_K = Float32(0.7978845608028654)
comptime IMP_GELU_C = Float32(0.044715)
comptime IMP_GELU_C3 = Float32(0.134145)


def unary_has_gradient(kind: Int) -> Bool:
    """False for the piecewise-constant kinds (floor, ceil, round, sign,
    not, bool): their gradient is zero."""
    return not (kind == IMP_FLOOR or kind == IMP_CEIL or kind == IMP_ROUND or kind == IMP_SIGN or kind == IMP_NOT
                or kind == IMP_BOOL)


def binary_has_gradient(kind: Int) -> Bool:
    """False for the comparisons and logical kinds (0/1 results)."""
    return kind == IMP_POW or kind == IMP_MINIMUM or kind == IMP_MAXIMUM or kind == IMP_WHERE


# ---------------------------------------------------------------- scalar helpers
@always_inline
def _round_even(x: Float32) -> Float32:
    """x to the nearest integer, a half to the even one (exact: floor and
    the difference of two nearby floats round nothing)."""
    var r = floor(x)
    var diff = x - r
    if diff > Float32(0.5):
        return r + Float32(1)
    if diff < Float32(0.5):
        return r
    var half = r * Float32(0.5)
    if floor(half) == half:
        return r
    return r + Float32(1)


@always_inline
def _abs(x: Float32) -> Float32:
    return bitcast[DType.float32](bitcast[DType.uint32](x) & UInt32(0x7FFFFFFF))


@always_inline
def _order_key(v: Float32) -> UInt32:
    """A float as an unsigned key whose integer order is the float's order
    (-0 below +0): the total order identical_fmax selects by."""
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


@always_inline
def _fmin(x: Float32, y: Float32) -> Float32:
    """The total-order minimum: a NaN operand gives the canonical NaN, -0 is
    below +0, the result does not depend on the operand order."""
    if x != x or y != y:
        return bitcast[DType.float32](UInt32(0x7FC00000))
    var a = ftz(x)
    var b = ftz(y)
    if _order_key(b) < _order_key(a):
        return b
    return a


@always_inline
def _pow(x: Float32, p: Float32) -> Float32:
    """x ** p: identical_pow for x >= 0; a negative x with an integral
    exponent is |x| ** p with the sign of an odd exponent (NaN otherwise)."""
    if x < Float32(0) and floor(p) == p:
        var m = ftz(identical_pow(-x, p))
        var h = p * Float32(0.5)
        if floor(h) != h:
            return -m
        return m
    return ftz(identical_pow(x, p))


@always_inline
def _unit(b: Bool) -> Float32:
    if b:
        return Float32(1)
    return Float32(0)


@always_inline
def imp_unary_value(kind: Int, x: Float32, f0: Float32, f1: Float32) -> Float32:
    """The forward word of unary kind `kind` at input word x."""
    if kind == IMP_ERF:
        return ftz(identical_erf(x))
    if kind == IMP_GELU_ERF:
        return ftz(identical_gelu_erf(x))
    if kind == IMP_GELU_TANH:
        return ftz(identical_gelu_tanh(x))
    if kind == IMP_SOFTPLUS:
        return ftz(identical_softplus(x))
    if kind == IMP_SILU:
        return ftz(identical_silu(x))
    if kind == IMP_LEAKY_RELU:
        if x >= Float32(0):
            return x
        return mul(f0, x)
    if kind == IMP_ELU:
        if x >= Float32(0):
            return x
        return mul(f0, sub(ftz(identical_exp(x)), Float32(1)))
    if kind == IMP_HARD_SIGMOID:
        return ftz(identical_clamp(add(mul(f0, x), f1), Float32(0), Float32(1)))
    if kind == IMP_HARD_SWISH:
        return mul(x, ftz(identical_clamp(add(mul(f0, x), f1), Float32(0), Float32(1))))
    if kind == IMP_CLIP:
        return ftz(identical_clamp(x, f0, f1))
    if kind == IMP_ABS:
        return _abs(x)
    if (kind == IMP_FLOOR or kind == IMP_CEIL or kind == IMP_ROUND) and x != x:
        # a NaN passes through as its own word, never a library's NaN
        return x
    if kind == IMP_FLOOR:
        return floor(x)
    if kind == IMP_CEIL:
        return ceil(x)
    if kind == IMP_ROUND:
        return _round_even(x)
    if kind == IMP_SIN:
        return ftz(identical_sin(x))
    if kind == IMP_COS:
        return ftz(identical_cos(x))
    if kind == IMP_SIGN:
        if x != x:
            return x
        if x > Float32(0):
            return Float32(1)
        if x < Float32(0):
            return Float32(-1)
        return Float32(0)
    if kind == IMP_RECIPROCAL:
        return div(Float32(1), x)
    if kind == IMP_NOT:
        return _unit(x == Float32(0))
    if kind == IMP_BOOL:
        return _unit(x != Float32(0))
    # IMP_POW_SCALAR
    return _pow(x, f0)


@always_inline
def imp_unary_vjp_value(kind: Int, g: Float32, x: Float32, y: Float32, f0: Float32, f1: Float32) -> Float32:
    """The VJP word of unary kind `kind`: g the upstream word, x the forward
    input, y the forward output."""
    if kind == IMP_ERF:
        var e = ftz(identical_exp(-mul(x, x)))
        return mul(g, mul(IMP_TWO_OVER_SQRT_PI, e))
    if kind == IMP_GELU_ERF:
        var cdf = mul(Float32(0.5), add(Float32(1), ftz(identical_erf(mul(x, IMP_INV_SQRT2)))))
        var pdf = mul(IMP_INV_SQRT_2PI, ftz(identical_exp(mul(Float32(-0.5), mul(x, x)))))
        return mul(g, add(cdf, mul(x, pdf)))
    if kind == IMP_GELU_TANH:
        var x2 = mul(x, x)
        var u = mul(IMP_GELU_K, add(x, mul(IMP_GELU_C, mul(x2, x))))
        var th = ftz(identical_tanh(u))
        var left = mul(Float32(0.5), add(Float32(1), th))
        var du = mul(IMP_GELU_K, add(Float32(1), mul(IMP_GELU_C3, x2)))
        var right = mul(mul(mul(Float32(0.5), x), sub(Float32(1), mul(th, th))), du)
        return mul(g, add(left, right))
    if kind == IMP_SOFTPLUS:
        return mul(g, ftz(identical_sigmoid(x)))
    if kind == IMP_SILU:
        var s = ftz(identical_sigmoid(x))
        return mul(g, mul(s, add(Float32(1), mul(x, sub(Float32(1), s)))))
    if kind == IMP_LEAKY_RELU:
        if x > Float32(0):
            return g
        return mul(g, f0)
    if kind == IMP_ELU:
        if x > Float32(0):
            return g
        return mul(g, mul(f0, ftz(identical_exp(x))))
    if kind == IMP_HARD_SIGMOID or kind == IMP_HARD_SWISH:
        var v = add(mul(f0, x), f1)
        var inside = v > Float32(0) and v < Float32(1)
        if kind == IMP_HARD_SIGMOID:
            if inside:
                return mul(g, f0)
            return Float32(0)
        var hs = ftz(identical_clamp(v, Float32(0), Float32(1)))
        if inside:
            return mul(g, add(hs, mul(x, f0)))
        return mul(g, hs)
    if kind == IMP_CLIP:
        if x >= f0 and x <= f1:
            return g
        return Float32(0)
    if kind == IMP_ABS:
        if x > Float32(0):
            return g
        if x < Float32(0):
            return -g
        return Float32(0)
    if kind == IMP_SIN:
        return mul(g, ftz(identical_cos(x)))
    if kind == IMP_COS:
        return -mul(g, ftz(identical_sin(x)))
    if kind == IMP_RECIPROCAL:
        return -mul(g, mul(y, y))
    if kind == IMP_POW_SCALAR:
        if f0 == Float32(0):
            return Float32(0)
        return mul(g, mul(f0, _pow(x, sub(f0, Float32(1)))))
    # floor, ceil, round, sign, not, bool: piecewise constant
    return Float32(0)


@always_inline
def imp_binary_value(kind: Int, x: Float32, y: Float32) -> Float32:
    """The forward word of binary kind `kind` (every kind but WHERE)."""
    if kind == IMP_POW:
        return _pow(x, y)
    if kind == IMP_MINIMUM:
        return _fmin(x, y)
    if kind == IMP_MAXIMUM:
        return ftz(identical_fmax(x, y))
    if kind == IMP_EQ:
        return _unit(x == y)
    if kind == IMP_GT:
        return _unit(x > y)
    if kind == IMP_LT:
        return _unit(x < y)
    if kind == IMP_GE:
        return _unit(x >= y)
    if kind == IMP_LE:
        return _unit(x <= y)
    if kind == IMP_NE:
        return _unit(x != y)
    if kind == IMP_AND:
        return _unit(x != Float32(0) and y != Float32(0))
    # IMP_OR
    return _unit(x != Float32(0) or y != Float32(0))


@always_inline
def imp_binary_vjp_value(kind: Int, e: Int, g: Float32, x: Float32, y: Float32, z: Float32, c: Float32) -> Float32:
    """The VJP term of operand e (0: a, 1: b) at one output word: g the
    upstream word, x and y the operands, z the forward output, c the
    condition (WHERE)."""
    if kind == IMP_POW:
        if e == 0:
            if y == Float32(0):
                return Float32(0)
            return mul(g, mul(y, _pow(x, sub(y, Float32(1)))))
        if x == Float32(0) and y >= Float32(0):
            return Float32(0)
        return mul(g, mul(z, ftz(identical_log(x))))
    if kind == IMP_MINIMUM or kind == IMP_MAXIMUM:
        # the selected operand takes the gradient, a tie splits it in two
        var mine = x if e == 0 else y
        var other = y if e == 0 else x
        if mine == other:
            return mul(g, Float32(0.5))
        var wins = mine != mine
        if not wins and other == other:
            if kind == IMP_MAXIMUM:
                wins = mine > other
            else:
                wins = mine < other
        if wins:
            return g
        return Float32(0)
    if kind == IMP_WHERE:
        var take_a = c != Float32(0)
        if (e == 0 and take_a) or (e == 1 and not take_a):
            return g
        return Float32(0)
    return Float32(0)


# ---------------------------------------------------------------- operand reads
@always_inline
def _rec(p: FP, k: Int, col: Int) -> Int:
    return rn_word(p, IMP_BIN_HEAD + k * IMP_BIN_REC + col)


@always_inline
def _decode(p: FP, count: Int, index: Int) -> Tuple[Int, Int, Int]:
    """(a, b, c) offsets of output word `index` decoded over records
    0 .. count - 1 (innermost last), columns 2, 3, 4, plus the base offsets."""
    var rem = index
    var oa = rn_word(p, 2)
    var ob = rn_word(p, 3)
    var oc = rn_word(p, 4)
    var k = count - 1
    while k >= 0:  # small-loop(count <= 8: tensor rank, shape metadata)
        var ext = _rec(p, k, 0)
        var c = rem % ext
        rem = rem // ext
        oa += c * _rec(p, k, 2)
        ob += c * _rec(p, k, 3)
        oc += c * _rec(p, k, 4)
        k -= 1
    return (oa, ob, oc)


@always_inline
def _ld_b(a: Args, i: Int) -> Float32:
    # i9 = 1: operand b is operand a's buffer (p1 unbound). A branch, never
    # a pointer select (autograd/ops.mojo _ld_right).
    if a.i9 == 1:
        return ld(a.p0, i)
    return ld(a.p1, i)


@always_inline
def _ld_c(a: Args, i: Int) -> Float32:
    # i10 = 1: the condition is operand a's buffer; i11 = 1: operand b's
    # (p2 unbound in both). Branches, never a pointer select.
    if a.i10 == 1:
        return ld(a.p0, i)
    if a.i11 == 1:
        return _ld_b(a, i)
    return ld(a.p2, i)


# ---------------------------------------------------------------- bodies
@always_inline
def op_imp_unary_fwd(t: Int, a: Args):
    """p1[t] = kind(p0[t]). p0 x, p1 y; i1 kind, f0 / f1 its parameters.
    Thread t = element t."""
    st(a.p1, t, imp_unary_value(a.i1, ld(a.p0, t), a.f0, a.f1))


@always_inline
def op_imp_unary_vjp(t: Int, a: Args):
    """p3[t] = the VJP at element t. p0 the forward input, p1 the forward
    output, p2 upstream, p3 destination; i1 kind, f0 / f1."""
    st(a.p3, t, imp_unary_vjp_value(a.i1, ld(a.p2, t), ld(a.p0, t), ld(a.p1, t), a.f0, a.f1))


@always_inline
def op_imp_bin_fwd(t: Int, a: Args):
    """p3[t] = a op b (WHERE: c != 0 ? a : b) at output word t with NumPy
    broadcasting. p0 a, p1 b (unbound when i9 = 1), p2 c (WHERE only;
    unbound when i10 or i11 = 1), p3 output, p11 the binary descriptor;
    i1 kind."""
    var nk = rn_word(a.p11, 0)
    var d = _decode(a.p11, nk, t)
    var v: Float32
    if a.i1 == IMP_WHERE:
        # branch and load the selected operand only
        if _ld_c(a, d[2]) != Float32(0):
            v = ld(a.p0, d[0])
        else:
            v = _ld_b(a, d[1])
    else:
        v = imp_binary_value(a.i1, ld(a.p0, d[0]), _ld_b(a, d[1]))
    st(a.p3, t, v)


@always_inline
def op_imp_bin_vjp(t: Int, a: Args):
    """The VJP term of operand i8 (0: a, 1: b) at output word t, written to
    p4[t]: the operand's gradient word when it has the output's shape, else
    a scratch word the broadcast fold reduces. op_imp_bin_fwd's slots and
    descriptor plus p3 the forward output, p5 upstream, p4 destination."""
    var nk = rn_word(a.p11, 0)
    var d = _decode(a.p11, nk, t)
    var kind = a.i1
    var x = ld(a.p0, d[0])
    var y = _ld_b(a, d[1])
    var c = Float32(0)
    if kind == IMP_WHERE:
        c = _ld_c(a, d[2])
    st(a.p4, t, imp_binary_vjp_value(kind, a.i8, ld(a.p5, t), x, y, ld(a.p3, t), c))


@always_inline
def op_imp_argmax(t: Int, a: Args):
    """Output word t of argmax from the max fold's position words (p0, int32
    patterns, autograd/rankn_fold.mojo rankn_max_argmax): i1 =
    IMP_ARGMAX_INT32 copies the pattern (raw, never through st: ftz would
    zero a small integer's pattern), IMP_ARGMAX_FLOAT writes the position
    as a float32 value. p1 output."""
    var bits = a.p0.unsafe_load(t)
    if a.i1 == IMP_ARGMAX_INT32:
        a.p1.unsafe_store(t, bits)
    else:
        st(a.p1, t, Float32(Int(bitcast[DType.int32](bits))))


@always_inline
def op_imp_pad(t: Int, a: Args):
    """Output word t of a constant pad: the source word at the output's
    coordinates less the leading pads, or f0 where a coordinate falls
    outside the source. p0 source, p1 output, p11 the pad descriptor."""
    var nd = rn_word(a.p11, 0)
    var src = rn_word(a.p11, 1)
    var rem = t
    var inside = True
    var k = nd - 1
    while k >= 0:  # small-loop(nd <= 8: tensor rank, shape metadata)
        var at = IMP_PAD_HEAD + k * IMP_PAD_REC
        var ext = rn_word(a.p11, at)
        var c = rem % ext
        rem = rem // ext
        var s = c - rn_word(a.p11, at + 1)
        if s < 0 or s >= rn_word(a.p11, at + 2):
            inside = False
        else:
            src += s * rn_word(a.p11, at + 3)
        k -= 1
    if inside:
        st(a.p1, t, ld(a.p0, src))
    else:
        st(a.p1, t, a.f0)


@always_inline
def op_imp_pad_vjp(t: Int, a: Args):
    """Source word t's gradient: the upstream word at its padded position
    (zero for a word a negative pad cropped away). p0 upstream, p1
    destination, p11 the pad VJP descriptor."""
    var nd = rn_word(a.p11, 0)
    var rem = t
    var dst = 0
    var inside = True
    var k = nd - 1
    while k >= 0:  # small-loop(nd <= 8: tensor rank, shape metadata)
        var at = IMP_PAD_HEAD + k * IMP_PAD_REC
        var ext = rn_word(a.p11, at)
        var c = rem % ext
        rem = rem // ext
        var o = c + rn_word(a.p11, at + 1)
        if o < 0 or o >= rn_word(a.p11, at + 3):
            inside = False
        else:
            dst += o * rn_word(a.p11, at + 2)
        k -= 1
    if inside:
        st(a.p1, t, ld(a.p0, dst))
    else:
        st(a.p1, t, Float32(0))


@always_inline
def _scan_src(o: Int, j: Int, q: Int, L: Int, I: Int, rev: Int) -> Int:
    """The source word of scan position j (scan order) of outer o, inner q."""
    var r = j
    if rev == 1:
        r = L - 1 - j
    return (o * L + r) * I + q


@always_inline
def _leaf_sum(p: FP, o: Int, lo: Int, hi: Int, q: Int, L: Int, I: Int, rev: Int) -> Float32:
    """Scan positions lo .. hi (inclusive, one leaf) added in ascending scan
    order, left to right."""
    var s = ld(p, _scan_src(o, lo, q, L, I, rev))
    var j = lo + 1
    while j <= hi:  # small-loop(at most IMP_SCAN_LEAF - 1 = 63 terms: one leaf of the declared scan order)
        s = add(s, ld(p, _scan_src(o, j, q, L, I, rev)))
        j += 1
    return s


@always_inline
def op_imp_scan_totals(t: Int, a: Args):
    """Leaf totals of a scan: p1[t] = the sum of leaf b of outer o, inner q
    (t = (o * nb + b) * I + q, nb = ceil(L / 64)), in scan order. p0 the
    source (outer, L, inner), i1 L, i2 I, i3 1 for a reverse scan."""
    var L = a.i1
    var I = a.i2
    var nb = (L + IMP_SCAN_LEAF - 1) // IMP_SCAN_LEAF
    var q = t % I
    var rest = t // I
    var b = rest % nb
    var o = rest // nb
    var lo = b * IMP_SCAN_LEAF
    var hi = min(lo + IMP_SCAN_LEAF, L) - 1
    st(a.p1, t, _leaf_sum(a.p0, o, lo, hi, q, L, I, a.i3))


@always_inline
def op_imp_scan_final(t: Int, a: Args):
    """Output word t = (o * L + r) * I + q of a scan (THE SCAN ORDER): p0 the
    source, p1 the output, p2 Q, the inclusive scan of the leaf totals
    (outer, nb, inner; unbound when i5 = 0, a single leaf); i1 L, i2 I, i3
    1 for a reverse scan, i4 1 for an exclusive one."""
    var L = a.i1
    var I = a.i2
    var q = t % I
    var rest = t // I
    var r = rest % L
    var o = rest // L
    var j = r
    if a.i3 == 1:
        j = L - 1 - r
    if a.i4 == 1:
        j -= 1
    if j < 0:
        st(a.p1, t, Float32(0))
        return
    var b = j // IMP_SCAN_LEAF
    var s = _leaf_sum(a.p0, o, b * IMP_SCAN_LEAF, j, q, L, I, a.i3)
    if a.i5 == 1 and b > 0:
        var nb = (L + IMP_SCAN_LEAF - 1) // IMP_SCAN_LEAF
        s = add(ld(a.p2, (o * nb + b - 1) * I + q), s)
    st(a.p1, t, s)


@always_inline
def op_imp_flat_ids(t: Int, a: Args):
    """Flat id t of a multi-tensor index (THE FLAT IDS DESCRIPTOR): the
    index tensor p0's word at broadcast position t, wrapped to [0, i1) and
    times i2, written (i3 = 1) or added to the id already at p1[t] (i3 = 0)
    as an int32 word; -1 where the index or the earlier id is invalid."""
    var nk = rn_word(a.p11, 0)
    var off = rn_word(a.p11, 1)
    var rem = t
    var k = nk - 1
    while k >= 0:  # small-loop(nk <= 8: tensor rank, shape metadata)
        var at = IMP_IDS_HEAD + k * IMP_IDS_REC
        var ext = rn_word(a.p11, at)
        var c = rem % ext
        rem = rem // ext
        off += c * rn_word(a.p11, at + 1)
        k -= 1
    var v = ld(a.p0, off)
    var dim = a.i1
    var ok = v == v and floor(v) == v and v > Float32(-2147483648.0) and v < Float32(2147483648.0)
    var iv = 0
    if ok:
        iv = Int(v)
        if iv < 0:
            iv += dim
        ok = iv >= 0 and iv < dim
    var prev = 0
    if a.i3 == 0:
        prev = Int(bitcast[DType.int32](a.p1.unsafe_load(t)))
        if prev < 0:
            ok = False
    var w = Int32(-1)
    if ok:
        w = Int32(prev + iv * a.i2)
    # raw, never through st: ftz would zero a small integer's pattern
    a.p1.unsafe_store(t, bitcast[DType.float32](w))


@always_inline
def apply_import[OP: Int](t: Int, a: Args):
    """The OP_IMP_FIRST..OP_IMP_LAST family (sequence/dispatch.mojo)."""
    comptime if OP == OP_IMP_UNARY_FWD:
        op_imp_unary_fwd(t, a)
    elif OP == OP_IMP_UNARY_VJP:
        op_imp_unary_vjp(t, a)
    elif OP == OP_IMP_BIN_FWD:
        op_imp_bin_fwd(t, a)
    elif OP == OP_IMP_BIN_VJP:
        op_imp_bin_vjp(t, a)
    elif OP == OP_IMP_ARGMAX:
        op_imp_argmax(t, a)
    elif OP == OP_IMP_PAD:
        op_imp_pad(t, a)
    elif OP == OP_IMP_PAD_VJP:
        op_imp_pad_vjp(t, a)
    elif OP == OP_IMP_SCAN_TOTALS:
        op_imp_scan_totals(t, a)
    elif OP == OP_IMP_SCAN_FINAL:
        op_imp_scan_final(t, a)
    elif OP == OP_IMP_FLAT_IDS:
        op_imp_flat_ids(t, a)
