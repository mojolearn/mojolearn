# SPDX-License-Identifier: Apache-2.0
# Lane ag-import-kernels (docs/plans/MODEL_IMPORT.md 3.3): checks of the
# import elementwise family on any executor (HostExec, Metal, NVIDIA, AMD).
# Each case prints one `IMPORT <kind> <case> digest <d>` line; the digests
# must be equal on every executor. Every kind runs once forward and once
# through a gradient:
#   forward words are compared bit for bit with the same scalar function
#     (autograd/import_ops.mojo imp_*_value) applied on the host to the
#     flushed input words, so a mismatch is the launch's plumbing, and the
#     cross-executor digest diff checks the numerics;
#   a gradient is compared bit for bit with the scalar VJP where it is
#     elementwise, and with a float64 closed form written here (check_values'
#     tolerance) for the mathematics and for a broadcast operand's fold.
# Kinds without a gradient check that the gradient is zero. Also: where with
# a broadcast condition, expand on a middle axis equal to mul(ones, x) word
# for word (value and gradient), argmax with ties and NaN in both output
# forms, constant pad with a negative pad. Shapes are small, odd and off any
# board row; one fold has two leaves (R = 70).
# Lane ag-import-lm-ops: the nonzero test (bool) with the step kinds; any and
# all over one and two axes against host loops; cumsum on one leaf, on one
# level of leaf totals (L = 150) and on two levels (L = 4500), forward,
# reverse and exclusive, value and gradient bit for bit against a host
# replica of THE SCAN ORDER (autograd/import_ops.mojo) and within tolerance
# of a float64 running sum; index with two index tensors on leading,
# middle (consecutive) and split axes, a negative and a repeated position,
# value bit for bit and gradient against a host scatter-add.
from std.memory import bitcast
from std.math import nan, exp, log, sqrt, tanh, sin, cos, erf
from std.collections import InlineArray
from autograd.tape import Tape, Tensor
from autograd.ops import MULTIPLY
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_fold import rankn_binary
from autograd.import_ops import (
    IMP_ERF, IMP_GELU_ERF, IMP_GELU_TANH, IMP_SOFTPLUS, IMP_SILU, IMP_LEAKY_RELU, IMP_ELU,
    IMP_HARD_SIGMOID, IMP_HARD_SWISH, IMP_CLIP, IMP_ABS, IMP_FLOOR, IMP_CEIL, IMP_ROUND, IMP_SIN,
    IMP_COS, IMP_SIGN, IMP_RECIPROCAL, IMP_NOT, IMP_POW_SCALAR, IMP_POW, IMP_MINIMUM, IMP_MAXIMUM,
    IMP_EQ, IMP_GT, IMP_LT, IMP_GE, IMP_LE, IMP_NE, IMP_AND, IMP_OR, IMP_ARGMAX_INT32,
    IMP_ARGMAX_FLOAT, IMP_BOOL, IMP_SCAN_LEAF, imp_unary_value, imp_unary_vjp_value, imp_binary_value, imp_binary_vjp_value,
    unary_has_gradient, binary_has_gradient,
)
from autograd.import_api import (
    ImportRules, imp_unary, imp_binary, imp_where, imp_expand, imp_argmax, imp_pad, imp_cumsum, imp_gather_nd,
)
from autograd.rankn_fold import rankn_max
from sequence.ops import add
from autograd.checks.support import check_values
from sequence.exec_trait import Exec
from checks.numerics import ftz


# ---------------------------------------------------------------- data and digests
def _data(n: Int, salt: Int) -> List[Float32]:
    """Values in [-0.615, 0.614], zero included."""
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(((i * 7919 + salt * 104729) % 1000) - 500) * Float32(0.00123)
    return v^


def _positive(n: Int, salt: Int) -> List[Float32]:
    """Values in [0.5, 1.7)."""
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(0.5) + Float32((i * 7919 + salt * 104729) % 1000) * Float32(0.0012)
    return v^


def _quarters(n: Int, salt: Int) -> List[Float32]:
    """Multiples of 0.25 in [-2.75, 2.75]: halves for round, ties for
    minimum and maximum, zeros for sign and not, a NaN every 41 words."""
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(((i * 37 + salt * 11) % 23) - 11) * Float32(0.25)
        if (i * 13 + salt) % 41 == 0:
            v[i] = nan[DType.float32]()
    return v^


def _digest(v: List[Float32]) -> UInt64:
    var digest = UInt64(1469598103934665603)
    for i in range(len(v)):  # small-loop(len(v): check output digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)
    return digest


def _mix(d: UInt64, e: UInt64) -> UInt64:
    return (d ^ e) * UInt64(1099511628211)


def _same(name: String, got: List[Float32], want: List[Float32]) raises -> UInt64:
    if len(got) != len(want):
        raise Error(name + ": lengths differ")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            print("FAIL", name, "index", i, "got", got[i], "want", want[i])
            raise Error(name + ": bits differ")
    return _digest(got)


def _same_grad(name: String, got: List[Float32], want: List[Float32]) raises -> UInt64:
    """_same for a gradient: the tape may add a contribution to a zeroed
    leaf gradient, which turns a -0.0 contribution into +0.0, so the two
    zeros compare equal; every other word bit for bit."""
    if len(got) != len(want):
        raise Error(name + ": lengths differ")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        var both_zero = got[i] == Float32(0) and want[i] == Float32(0)
        if not both_zero and bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            print("FAIL", name, "index", i, "got", got[i], "want", want[i])
            raise Error(name + ": bits differ")
    return _digest(got)


def _close(name: String, got: List[Float32], want: List[Float64]) raises -> UInt64:
    var w32 = List[Float32](length=len(want), fill=Float32(0))
    for i in range(len(want)):  # small-loop(len(want): check oracle, host side)
        w32[i] = Float32(want[i])
    check_values(name, got, w32)
    return _digest(got)


def _shape3(a: Int, b: Int, c: Int) -> ShapeN:
    var d = InlineArray[Int, RN_MAX_RANK](fill=1)
    d[0] = a
    d[1] = b
    d[2] = c
    return ShapeN(3, d)


def _src3(s: ShapeN, o: ShapeN, i: Int, j: Int, k: Int) -> Int:
    """The word of a rank-3 operand of shape s that output (i, j, k) of
    shape o reads under broadcasting."""
    var si = 0 if s.d[0] == 1 else i
    var sj = 0 if s.d[1] == 1 else j
    var sk = 0 if s.d[2] == 1 else k
    return (si * s.d[1] + sj) * s.d[2] + sk


def _bcast3(v: List[Float32], s: ShapeN, o: ShapeN) -> List[Float32]:
    var r = List[Float32](length=o.count(), fill=Float32(0))
    for i in range(o.d[0]):  # small-loop(check data, host side)
        for j in range(o.d[1]):  # small-loop(check data, host side)
            for k in range(o.d[2]):  # small-loop(check data, host side)
                r[(i * o.d[1] + j) * o.d[2] + k] = v[_src3(s, o, i, j, k)]
    return r^


def _name(kind: Int) -> String:
    if kind == IMP_ERF:
        return "erf"
    if kind == IMP_GELU_ERF:
        return "gelu"
    if kind == IMP_GELU_TANH:
        return "gelu_tanh"
    if kind == IMP_SOFTPLUS:
        return "softplus"
    if kind == IMP_SILU:
        return "silu"
    if kind == IMP_LEAKY_RELU:
        return "leaky_relu"
    if kind == IMP_ELU:
        return "elu"
    if kind == IMP_HARD_SIGMOID:
        return "hard_sigmoid"
    if kind == IMP_HARD_SWISH:
        return "hard_swish"
    if kind == IMP_CLIP:
        return "clip"
    if kind == IMP_ABS:
        return "abs"
    if kind == IMP_FLOOR:
        return "floor"
    if kind == IMP_CEIL:
        return "ceil"
    if kind == IMP_ROUND:
        return "round"
    if kind == IMP_SIN:
        return "sin"
    if kind == IMP_COS:
        return "cos"
    if kind == IMP_SIGN:
        return "sign"
    if kind == IMP_RECIPROCAL:
        return "reciprocal"
    if kind == IMP_NOT:
        return "not"
    if kind == IMP_BOOL:
        return "bool"
    return "pow_scalar"


def _bname(kind: Int) -> String:
    if kind == IMP_POW:
        return "pow"
    if kind == IMP_MINIMUM:
        return "minimum"
    if kind == IMP_MAXIMUM:
        return "maximum"
    if kind == IMP_EQ:
        return "equal"
    if kind == IMP_GT:
        return "greater"
    if kind == IMP_LT:
        return "less"
    if kind == IMP_GE:
        return "greater_equal"
    if kind == IMP_LE:
        return "less_equal"
    if kind == IMP_NE:
        return "not_equal"
    if kind == IMP_AND:
        return "and"
    return "or"


# ---------------------------------------------------------------- closed forms
def _sig64(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x))


def _dclosed(kind: Int, x: Float64, f0: Float64, f1: Float64) -> Float64:
    """d kind(x) / dx in float64 (the kinds with a gradient)."""
    var pi = 3.141592653589793
    if kind == IMP_ERF:
        return 2.0 / sqrt(pi) * exp(-x * x)
    if kind == IMP_GELU_ERF:
        return 0.5 * (1.0 + erf(x / sqrt(Float64(2)))) + x * exp(-0.5 * x * x) / sqrt(2.0 * pi)
    if kind == IMP_GELU_TANH:
        var k = sqrt(2.0 / pi)
        var u = k * (x + 0.044715 * x * x * x)
        var t = tanh(u)
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * k * (1.0 + 3.0 * 0.044715 * x * x)
    if kind == IMP_SOFTPLUS:
        return _sig64(x)
    if kind == IMP_SILU:
        var s = _sig64(x)
        return s * (1.0 + x * (1.0 - s))
    if kind == IMP_LEAKY_RELU:
        if x > 0.0:
            return Float64(1)
        return f0
    if kind == IMP_ELU:
        if x > 0.0:
            return Float64(1)
        return f0 * exp(x)
    if kind == IMP_HARD_SIGMOID or kind == IMP_HARD_SWISH:
        var v = f0 * x + f1
        var inside = v > 0.0 and v < 1.0
        if kind == IMP_HARD_SIGMOID:
            if inside:
                return f0
            return Float64(0)
        var hs = v
        if v <= 0.0:
            hs = Float64(0)
        elif v >= 1.0:
            hs = Float64(1)
        if inside:
            return hs + x * f0
        return hs
    if kind == IMP_CLIP:
        if x >= f0 and x <= f1:
            return Float64(1)
        return Float64(0)
    if kind == IMP_ABS:
        if x > 0.0:
            return Float64(1)
        if x < 0.0:
            return Float64(-1)
        return Float64(0)
    if kind == IMP_SIN:
        return cos(x)
    if kind == IMP_COS:
        return -sin(x)
    if kind == IMP_RECIPROCAL:
        return -1.0 / (x * x)
    # IMP_POW_SCALAR (x > 0)
    return f0 * exp((f0 - 1.0) * log(x))


# ---------------------------------------------------------------- unary
def _unary_case[E: Exec](
    mut ex: E, mut tape: Tape, kind: Int, xv: List[Float32], f0: Float32, f1: Float32,
) raises -> UInt64:
    """kind(x) on [3, 7, 5]: the value bit for bit against the scalar
    function, the gradient bit for bit against the scalar VJP and within
    tolerance of the float64 closed form (zero for a kind without one)."""
    var s = _shape3(3, 7, 5)
    var n = s.count()
    var gv = _data(n, 90 + kind)
    var name = _name(kind)
    tape.reset()
    var x = tape.tensor_nd(ex, xv, s, True)
    var y = imp_unary(tape, ex, x, kind, f0, f1)
    if not tape.shape_of(y).same(s):
        raise Error(name + ": the output shape differs from the input's")
    tape.backward_with_rules(ex, y, tape.tensor_nd(ex, gv, s, False), ImportRules())
    var yv = tape.values(ex, y)
    var dx = tape.gradients(ex, x)
    var want = List[Float32](length=n, fill=Float32(0))
    var wdx = List[Float32](length=n, fill=Float32(0))
    var cdx = List[Float64](length=n, fill=Float64(0))
    for i in range(n):  # small-loop(n: check oracle, host side)
        var xi = ftz(xv[i])
        want[i] = ftz(imp_unary_value(kind, xi, f0, f1))
        if unary_has_gradient(kind):
            wdx[i] = ftz(imp_unary_vjp_value(kind, ftz(gv[i]), xi, want[i], f0, f1))
            cdx[i] = Float64(gv[i]) * _dclosed(kind, Float64(xi), Float64(f0), Float64(f1))
    var d = _same(name + "/value", yv, want)
    d = _mix(d, _same_grad(name + "/dx", dx, wdx))
    if unary_has_gradient(kind):
        d = _mix(d, _close(name + "/dx-closed", dx, cdx))
    print("IMPORT", name, "3x7x5 digest", d)
    return d^


# ---------------------------------------------------------------- binary
def _bin_case[E: Exec](
    mut ex: E, mut tape: Tape, kind: Int, sa: ShapeN, sb: ShapeN,
    av: List[Float32], bv: List[Float32], permuted: Bool, check_grad: Bool = True,
) raises -> UInt64:
    """a kind b with a [sa] and b [sb] broadcast to [3, 4, 5]. `permuted`:
    a is read through a pending permute of a [5, 3, 4] tensor (operand_nd's
    strided read). The value bit for bit against the scalar function; an
    operand of the output's shape gets its gradient bit for bit against the
    scalar VJP, a broadcast one within tolerance of the float64 sum of the
    closed-form terms."""
    var so = _shape3(3, 4, 5)
    var n = so.count()
    var gv = _data(n, 70 + kind)
    var name = _bname(kind)
    var cname = String(sa.d[0]) + "x" + String(sa.d[1]) + "x" + String(sa.d[2]) + "-" + String(sb.d[0]) + "x" + String(sb.d[1]) + "x" + String(sb.d[2])
    if permuted:
        cname = cname + "-permuted"
    tape.reset()
    var a: Tensor
    var a_src: Tensor
    if permuted:
        # the [5, 3, 4] tensor whose permute (1, 2, 0) is a: src[k, i, j] = a[i, j, k]
        var srcv = List[Float32](length=n, fill=Float32(0))
        for i in range(3):  # small-loop(check data, host side)
            for j in range(4):  # small-loop(check data, host side)
                for k in range(5):  # small-loop(check data, host side)
                    srcv[(k * 3 + i) * 4 + j] = av[(i * 4 + j) * 5 + k]
        a_src = tape.tensor_nd(ex, srcv, _shape3(5, 3, 4), True)
        var perm: List[Int] = [1, 2, 0]
        a = tape.permute(ex, a_src, perm)
    else:
        a = tape.tensor_nd(ex, av, sa, True)
        a_src = a
    var b = tape.tensor_nd(ex, bv, sb, True)
    var z = imp_binary(tape, ex, a, b, kind)
    var zv = tape.values(ex, z)
    var afull = _bcast3(av, sa, so)
    var bfull = _bcast3(bv, sb, so)
    var want = List[Float32](length=n, fill=Float32(0))
    for t in range(n):  # small-loop(n: check oracle, host side)
        want[t] = ftz(imp_binary_value(kind, ftz(afull[t]), ftz(bfull[t])))
    var d = _same(name + "/" + cname + "/value", zv, want)
    if binary_has_gradient(kind) and check_grad and not permuted:
        tape.backward_with_rules(ex, z, tape.tensor_nd(ex, gv, so, False), ImportRules())
        for e in range(2):  # small-loop(2: the two operands)
            var s = sa if e == 0 else sb
            var got = tape.gradients(ex, a if e == 0 else b)
            var exact = List[Float32](length=n, fill=Float32(0))
            var closed = List[Float64](length=s.count(), fill=Float64(0))
            for i in range(3):  # small-loop(check oracle, host side)
                for j in range(4):  # small-loop(check oracle, host side)
                    for k in range(5):  # small-loop(check oracle, host side)
                        var t = (i * 4 + j) * 5 + k
                        var x = ftz(afull[t])
                        var y = ftz(bfull[t])
                        var g = ftz(gv[t])
                        var term = ftz(imp_binary_vjp_value(kind, e, g, x, y, want[t], Float32(0)))
                        exact[t] = term
                        var c: Float64
                        if kind == IMP_POW:
                            var x64 = Float64(x)
                            var y64 = Float64(y)
                            if e == 0:
                                c = Float64(g) * y64 * exp((y64 - 1.0) * log(x64))
                            else:
                                c = Float64(g) * exp(y64 * log(x64)) * log(x64)
                        else:
                            # minimum / maximum: the selected operand, a tie splits in two
                            var mine = x if e == 0 else y
                            var other = y if e == 0 else x
                            if mine == other:
                                c = 0.5 * Float64(g)
                            elif mine != mine:
                                c = Float64(g)
                            elif other != other:
                                c = 0.0
                            elif (kind == IMP_MAXIMUM and mine > other) or (kind == IMP_MINIMUM and mine < other):
                                c = Float64(g)
                            else:
                                c = 0.0
                        closed[_src3(s, so, i, j, k)] += c
            if s.same(so):
                d = _mix(d, _same_grad(name + "/" + cname + "/d" + String(e), got, exact))
            d = _mix(d, _close(name + "/" + cname + "/d" + String(e) + "-closed", got, closed))
    elif not binary_has_gradient(kind):
        tape.backward_with_rules(ex, z, tape.tensor_nd(ex, gv, so, False), ImportRules())
        d = _mix(d, _same_grad(name + "/" + cname + "/da-zero", tape.gradients(ex, a_src), List[Float32](length=n, fill=Float32(0))))
    print("IMPORT", name, cname, "digest", d)
    return d^


def _where_case[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    """where(c [3, 1, 5], a [3, 4, 5], b [1, 4, 1]): the value bit for bit;
    a's gradient (the output's shape) bit for bit, b's (summed over two
    broadcast axes) within tolerance, the condition's zero. One condition
    word is NaN (nonzero: a)."""
    var sc = _shape3(3, 1, 5)
    var sa = _shape3(3, 4, 5)
    var sb = _shape3(1, 4, 1)
    var so = sa
    var n = so.count()
    var cv = List[Float32](length=15, fill=Float32(0))
    for i in range(15):  # small-loop(15: check data, host side)
        cv[i] = Float32(1) if (i * 7) % 3 == 0 else Float32(0)
    cv[4] = nan[DType.float32]()
    var av = _data(n, 61)
    var bv = _data(4, 62)
    var gv = _data(n, 63)
    tape.reset()
    var c = tape.tensor_nd(ex, cv, sc, True)
    var a = tape.tensor_nd(ex, av, sa, True)
    var b = tape.tensor_nd(ex, bv, sb, True)
    var z = imp_where(tape, ex, c, a, b)
    tape.backward_with_rules(ex, z, tape.tensor_nd(ex, gv, so, False), ImportRules())
    var cfull = _bcast3(cv, sc, so)
    var bfull = _bcast3(bv, sb, so)
    var want = List[Float32](length=n, fill=Float32(0))
    var wda = List[Float32](length=n, fill=Float32(0))
    var wdb = List[Float64](length=4, fill=Float64(0))
    for i in range(3):  # small-loop(check oracle, host side)
        for j in range(4):  # small-loop(check oracle, host side)
            for k in range(5):  # small-loop(check oracle, host side)
                var t = (i * 4 + j) * 5 + k
                var take_a = cfull[t] != Float32(0)
                want[t] = ftz(av[t]) if take_a else ftz(bfull[t])
                if take_a:
                    wda[t] = ftz(gv[t])
                else:
                    wdb[j] += Float64(ftz(gv[t]))
    var d = _same("where/value", tape.values(ex, z), want)
    d = _mix(d, _same_grad("where/da", tape.gradients(ex, a), wda))
    d = _mix(d, _close("where/db", tape.gradients(ex, b), wdb))
    d = _mix(d, _same_grad("where/dc-zero", tape.gradients(ex, c), List[Float32](length=15, fill=Float32(0))))
    print("IMPORT where 3x1x5-3x4x5-1x4x1 digest", d)
    return d^


# ---------------------------------------------------------------- expand
def _expand_case[E: Exec](mut ex: E, mut tape: Tape, a: Int, b: Int, c: Int) raises -> UInt64:
    """expand(x [a, 1, c], (a, b, c)) against mul(ones [a, b, c], x), the
    rank-N broadcast product of the fold lane: the same words, value and
    gradient (the gradient's fold has ceil(b / 64) leaves an output)."""
    var sx = _shape3(a, 1, c)
    var so = _shape3(a, b, c)
    var n = so.count()
    var xv = _data(a * c, 81)
    var gv = _data(n, 82)
    var name = String(a) + "x1x" + String(c) + "-to-" + String(a) + "x" + String(b) + "x" + String(c)
    tape.reset()
    var x = tape.tensor_nd(ex, xv, sx, True)
    var dims: List[Int] = [a, b, c]
    var y = imp_expand(tape, ex, x, dims)
    if not tape.shape_of(y).same(so):
        raise Error("expand: the output shape differs from the broadcast shape")
    tape.backward_with_rules(ex, y, tape.tensor_nd(ex, gv, so, False), ImportRules())
    var yv = tape.values(ex, y)
    var dx = tape.gradients(ex, x)
    tape.reset()
    var x2 = tape.tensor_nd(ex, xv, sx, True)
    var ones = tape.tensor_nd(ex, List[Float32](length=n, fill=Float32(1)), so, False)
    var y2 = rankn_binary(tape, ex, ones, x2, MULTIPLY)
    tape.backward_with_rules(ex, y2, tape.tensor_nd(ex, gv, so, False), ImportRules())
    var d = _same("expand/" + name + "/value", yv, tape.values(ex, y2))
    d = _mix(d, _same("expand/" + name + "/dx", dx, tape.gradients(ex, x2)))
    print("IMPORT expand", name, "digest", d)
    return d^


# ---------------------------------------------------------------- argmax
def _argmax_case[E: Exec](mut ex: E, mut tape: Tape, with_nan: Bool) raises -> UInt64:
    """argmax along axis 1 of x [5, 7, 6] (many ties; NaN at some words when
    asked) in both forms against a host scan (a NaN wins, the first position
    on ties); keepdims off gives (5, 6), on gives (5, 1, 6)."""
    var sx = _shape3(5, 7, 6)
    var n = sx.count()
    var xv = _quarters(n, 3)
    if not with_nan:
        for i in range(n):  # small-loop(n: check data, host side)
            if xv[i] != xv[i]:
                xv[i] = Float32(0.5)
    var name = String("argmax/5x7x6/axis1") + ("/nan" if with_nan else "")
    tape.reset()
    var x = tape.tensor_nd(ex, xv, sx, False)
    var pf = imp_argmax(tape, ex, x, 1, False, IMP_ARGMAX_FLOAT)
    var pi = imp_argmax(tape, ex, x, -2, True, IMP_ARGMAX_INT32)
    var sf = tape.shape_of(pf)
    var si = tape.shape_of(pi)
    if sf.ndim != 2 or sf.d[0] != 5 or sf.d[1] != 6:
        raise Error(name + ": keepdims=False must give (5, 6)")
    if si.ndim != 3 or si.d[0] != 5 or si.d[1] != 1 or si.d[2] != 6:
        raise Error(name + ": keepdims=True must give (5, 1, 6)")
    var gf = tape.values(ex, pf)
    var gi = tape.values(ex, pi)
    var want = List[Float32](length=30, fill=Float32(0))
    for i in range(5):  # small-loop(check oracle, host side)
        for k in range(6):  # small-loop(check oracle, host side)
            var best = 0
            var bv = xv[(i * 7) * 6 + k]
            for j in range(1, 7):  # small-loop(check oracle, host side)
                var v = xv[(i * 7 + j) * 6 + k]
                var bn = bv != bv
                if not bn and (v != v or v > bv):
                    best = j
                    bv = v
            want[i * 6 + k] = Float32(best)
            var got_i = Int(bitcast[DType.int32](gi[i * 6 + k]))
            if got_i != best:
                print("FAIL", name, "output", i * 6 + k, "int32 position", got_i, "want", best)
                raise Error(name + ": int32 positions differ")
    var d = _same(name + "/float", gf, want)
    d = _mix(d, _digest(gi))
    print("IMPORT", name, "digest", d)
    return d^


# ---------------------------------------------------------------- pad
def _pad_case[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    """pad x [3, 4, 5] with pads (1, 0, 2) before and (0, 3, -1) after, value
    0.5: output [4, 7, 6]; the value and the gradient (the upstream narrowed
    back, zero at the cropped words) bit for bit against host loops."""
    var sx = _shape3(3, 4, 5)
    var so = _shape3(4, 7, 6)
    var nx = sx.count()
    var no = so.count()
    var xv = _data(nx, 101)
    var gv = _data(no, 102)
    var before: List[Int] = [1, 0, 2]
    var pads: List[Int] = [1, 0, 2, 0, 3, -1]
    tape.reset()
    var x = tape.tensor_nd(ex, xv, sx, True)
    var y = imp_pad(tape, ex, x, pads, Float32(0.5))
    if not tape.shape_of(y).same(so):
        raise Error("pad: the output shape must be (4, 7, 6)")
    tape.backward_with_rules(ex, y, tape.tensor_nd(ex, gv, so, False), ImportRules())
    var want = List[Float32](length=no, fill=Float32(0.5))
    var wdx = List[Float32](length=nx, fill=Float32(0))
    for i in range(3):  # small-loop(check oracle, host side)
        for j in range(4):  # small-loop(check oracle, host side)
            for k in range(5):  # small-loop(check oracle, host side)
                var oi = i + before[0]
                var oj = j + before[1]
                var ok = k + before[2]
                if ok < 6:
                    var o = (oi * 7 + oj) * 6 + ok
                    want[o] = ftz(xv[(i * 4 + j) * 5 + k])
                    wdx[(i * 4 + j) * 5 + k] = ftz(gv[o])
    var d = _same("pad/value", tape.values(ex, y), want)
    d = _mix(d, _same_grad("pad/dx", tape.gradients(ex, x), wdx))
    print("IMPORT pad 3x4x5-to-4x7x6 digest", d)
    return d^


# ---------------------------------------------------------------- any and all
def _any_all_case[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    """any / all of x [5, 7, 6] (mostly zeros, some NaN) over axis 1 and
    over axes (0, 2), composed as the session composes them (the nonzero or
    zero test, the max fold, not), against host loops."""
    var sx = _shape3(5, 7, 6)
    var n = sx.count()
    var xv = _quarters(n, 5)
    for i in range(n):  # small-loop(n: check data, host side)
        if i % 3 != 0 and i % 7 != 0:
            xv[i] = Float32(0)
    for i in range(6):  # small-loop(6: rows that are all nonzero, for all)
        xv[(2 * 7 + i) * 6 + 1] = Float32(0.75)
    tape.reset()
    var x = tape.tensor_nd(ex, xv, sx, False)
    var ax1: List[Int] = [1]
    var ax02: List[Int] = [0, 2]
    var any1 = rankn_max(tape, ex, imp_unary(tape, ex, x, IMP_BOOL), ax1)
    var all1 = imp_unary(tape, ex, rankn_max(tape, ex, imp_unary(tape, ex, x, IMP_NOT), ax1), IMP_NOT)
    var any02 = rankn_max(tape, ex, imp_unary(tape, ex, x, IMP_BOOL), ax02)
    var all02 = imp_unary(tape, ex, rankn_max(tape, ex, imp_unary(tape, ex, x, IMP_NOT), ax02), IMP_NOT)
    var w_any1 = List[Float32](length=30, fill=Float32(0))
    var w_all1 = List[Float32](length=30, fill=Float32(1))
    var w_any02 = List[Float32](length=7, fill=Float32(0))
    var w_all02 = List[Float32](length=7, fill=Float32(1))
    for i in range(5):  # small-loop(check oracle, host side)
        for j in range(7):  # small-loop(check oracle, host side)
            for k in range(6):  # small-loop(check oracle, host side)
                var nz = xv[(i * 7 + j) * 6 + k] != Float32(0)
                if nz:
                    w_any1[i * 6 + k] = Float32(1)
                    w_any02[j] = Float32(1)
                else:
                    w_all1[i * 6 + k] = Float32(0)
                    w_all02[j] = Float32(0)
    var d = _same("any/5x7x6/axis1", tape.values(ex, any1), w_any1)
    d = _mix(d, _same("all/5x7x6/axis1", tape.values(ex, all1), w_all1))
    d = _mix(d, _same("any/5x7x6/axes02", tape.values(ex, any02), w_any02))
    d = _mix(d, _same("all/5x7x6/axes02", tape.values(ex, all02), w_all02))
    print("IMPORT any_all 5x7x6 digest", d)
    return d^


# ---------------------------------------------------------------- cumsum
def _host_leaf(v: List[Float32], o: Int, lo: Int, hi: Int, q: Int, L: Int, I: Int, rev: Bool) -> Float32:
    var s = Float32(0)
    for j in range(lo, hi + 1):  # small-loop(at most 64: one leaf, check oracle)
        var r = L - 1 - j if rev else j
        var w = ftz(v[(o * L + r) * I + q])
        s = w if j == lo else add(s, w)
    return s


def _host_scan(v: List[Float32], O: Int, L: Int, I: Int, rev: Bool, excl: Bool) -> List[Float32]:
    """THE SCAN ORDER of autograd/import_ops.mojo replayed on the host."""
    var out = List[Float32](length=O * L * I, fill=Float32(0))
    var nb = (L + IMP_SCAN_LEAF - 1) // IMP_SCAN_LEAF
    var q_scan = List[Float32]()
    if nb > 1:
        var totals = List[Float32](length=O * nb * I, fill=Float32(0))
        for o in range(O):  # small-loop(check oracle, host side)
            for b in range(nb):  # small-loop(check oracle, host side)
                for q in range(I):  # small-loop(check oracle, host side)
                    var lo = b * IMP_SCAN_LEAF
                    var hi = min(lo + IMP_SCAN_LEAF, L) - 1
                    totals[(o * nb + b) * I + q] = _host_leaf(v, o, lo, hi, q, L, I, rev)
        q_scan = _host_scan(totals, O, nb, I, False, False)
    for o in range(O):  # small-loop(check oracle, host side)
        for r in range(L):  # small-loop(check oracle, host side)
            for q in range(I):  # small-loop(check oracle, host side)
                var j = L - 1 - r if rev else r
                if excl:
                    j -= 1
                var w = Float32(0)
                if j >= 0:
                    var b = j // IMP_SCAN_LEAF
                    w = _host_leaf(v, o, b * IMP_SCAN_LEAF, j, q, L, I, rev)
                    if nb > 1 and b > 0:
                        w = add(q_scan[(o * nb + b - 1) * I + q], w)
                out[(o * L + r) * I + q] = w
    return out^


def _f64_scan(v: List[Float32], O: Int, L: Int, I: Int, rev: Bool, excl: Bool) -> List[Float64]:
    var out = List[Float64](length=O * L * I, fill=Float64(0))
    for o in range(O):  # small-loop(check oracle, host side)
        for q in range(I):  # small-loop(check oracle, host side)
            var acc = Float64(0)
            for k in range(L):  # small-loop(check oracle, host side)
                var r = L - 1 - k if rev else k
                var x = Float64(ftz(v[(o * L + r) * I + q]))
                if excl:
                    out[(o * L + r) * I + q] = acc
                    acc += x
                else:
                    acc += x
                    out[(o * L + r) * I + q] = acc
    return out^


def _cumsum_case[E: Exec](
    mut ex: E, mut tape: Tape, O: Int, L: Int, I: Int, rev: Bool, excl: Bool,
) raises -> UInt64:
    """cumsum of x [O, L, I] along axis 1: value and gradient bit for bit
    against the host replica of the scan order, and within tolerance of a
    float64 running sum."""
    var sx = _shape3(O, L, I)
    var n = sx.count()
    var xv = _data(n, 111 + L)
    var gv = _data(n, 112 + L)
    var name = String("cumsum/") + String(O) + "x" + String(L) + "x" + String(I)
    if rev:
        name += "/reverse"
    if excl:
        name += "/exclusive"
    tape.reset()
    var x = tape.tensor_nd(ex, xv, sx, True)
    var y = imp_cumsum(tape, ex, x, 1, excl, rev)
    if not tape.shape_of(y).same(sx):
        raise Error(name + ": the output shape differs from the input's")
    tape.backward_with_rules(ex, y, tape.tensor_nd(ex, gv, sx, False), ImportRules())
    var yv = tape.values(ex, y)
    var dx = tape.gradients(ex, x)
    var d = _same(name + "/value", yv, _host_scan(xv, O, L, I, rev, excl))
    d = _mix(d, _same_grad(name + "/dx", dx, _host_scan(gv, O, L, I, not rev, excl)))
    d = _mix(d, _close(name + "/value-f64", yv, _f64_scan(xv, O, L, I, rev, excl)))
    d = _mix(d, _close(name + "/dx-f64", dx, _f64_scan(gv, O, L, I, not rev, excl)))
    print("IMPORT", name, "digest", d)
    return d^


# ---------------------------------------------------------------- index
def _index_case[E: Exec](mut ex: E, mut tape: Tape, a0: Int, a1: Int) raises -> UInt64:
    """x [4, 5, 6] indexed on axes (a0, a1) by i [3, 1] and j [1, 2] (a
    negative and a repeated position): the result is [3, 2, rest] with the
    broadcast axes first, or in place of the indexed axes when they are
    consecutive and not leading. Value bit for bit against a host gather;
    x's gradient against a host scatter-add in ascending position order."""
    var sx = _shape3(4, 5, 6)
    var nx = sx.count()
    var xv = _data(nx, 121 + a0 + a1)
    var other = 3 - a0 - a1
    var dims: List[Int] = [4, 5, 6]
    var iv: List[Float32] = [Float32(1), Float32(-1), Float32(1)]
    var jv: List[Float32] = [Float32(2), Float32(0)]
    var d0 = dims[a0]
    var rest = dims[other]
    var name = String("index/4x5x6/axes") + String(a0) + String(a1)
    tape.reset()
    var x = tape.tensor_nd(ex, xv, sx, True)
    var si = InlineArray[Int, RN_MAX_RANK](fill=1)
    si[0] = 3
    var sj = InlineArray[Int, RN_MAX_RANK](fill=1)
    sj[1] = 2
    var ti = tape.tensor_nd(ex, iv, ShapeN(2, si), False)
    var tj = tape.tensor_nd(ex, jv, ShapeN(2, sj), False)
    var idx = List[Tensor]()
    idx.append(ti)
    idx.append(tj)
    var axes: List[Int] = [a0, a1]
    var y = imp_gather_nd(tape, ex, x, idx, axes)
    var middle = a0 == 1 and a1 == 2
    var so = _shape3(3, 2, rest)
    if middle:
        so = _shape3(rest, 3, 2)
    if not tape.shape_of(y).same(so):
        raise Error(name + ": the output shape is wrong")
    var no = so.count()
    var gv = _data(no, 131 + a0 + a1)
    tape.backward_with_rules(ex, y, tape.tensor_nd(ex, gv, so, False), ImportRules())
    var want = List[Float32](length=no, fill=Float32(0))
    var wdx = List[Float32](length=nx, fill=Float32(0))
    var seen = List[Bool](length=nx, fill=False)
    # the gather's position order: broadcast index (p, q), then the rest axis
    for p in range(3):  # small-loop(check oracle, host side)
        for q in range(2):  # small-loop(check oracle, host side)
            for r in range(rest):  # small-loop(check oracle, host side)
                var c = InlineArray[Int, 3](fill=0)
                var vi = Int(iv[p])
                c[a0] = vi + d0 if vi < 0 else vi
                c[a1] = Int(jv[q])
                c[other] = r
                var src = (c[0] * 5 + c[1]) * 6 + c[2]
                var o = (p * 2 + q) * rest + r
                if middle:
                    o = (r * 3 + p) * 2 + q
                want[o] = ftz(xv[src])
                var g = ftz(gv[o])
                wdx[src] = g if not seen[src] else add(wdx[src], g)
                seen[src] = True
    var d = _same(name + "/value", tape.values(ex, y), want)
    d = _mix(d, _same_grad(name + "/dx", tape.gradients(ex, x), wdx))
    print("IMPORT", name, "digest", d)
    return d^


def run_import_checks[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var summary = UInt64(1469598103934665603)
    var cases = 0
    var n = 105
    var smooth = _data(n, 1)
    var steps = _quarters(n, 2)
    var pos = _positive(n, 3)
    # the smooth kinds on [-0.615, 0.614]
    var smooth_kinds: List[Int] = [IMP_ERF, IMP_GELU_ERF, IMP_GELU_TANH, IMP_SOFTPLUS, IMP_SILU, IMP_SIN, IMP_COS]
    for i in range(len(smooth_kinds)):  # small-loop(7: check kinds)
        summary = _mix(summary, _unary_case(ex, tape, smooth_kinds[i], smooth, Float32(0), Float32(0)))
        cases += 1
    summary = _mix(summary, _unary_case(ex, tape, IMP_LEAKY_RELU, smooth, Float32(0.1), Float32(0)))
    summary = _mix(summary, _unary_case(ex, tape, IMP_ELU, smooth, Float32(1.3), Float32(0)))
    summary = _mix(summary, _unary_case(ex, tape, IMP_HARD_SIGMOID, smooth, Float32(1.5), Float32(0.5)))
    summary = _mix(summary, _unary_case(ex, tape, IMP_HARD_SWISH, smooth, Float32(1) / Float32(6), Float32(0.5)))
    summary = _mix(summary, _unary_case(ex, tape, IMP_CLIP, smooth, Float32(-0.3), Float32(0.25)))
    summary = _mix(summary, _unary_case(ex, tape, IMP_ABS, smooth, Float32(0), Float32(0)))
    summary = _mix(summary, _unary_case(ex, tape, IMP_RECIPROCAL, pos, Float32(0), Float32(0)))
    summary = _mix(summary, _unary_case(ex, tape, IMP_POW_SCALAR, pos, Float32(2.5), Float32(0)))
    cases += 8
    # the piecewise-constant kinds on quarters with NaN (gradient zero)
    var step_kinds: List[Int] = [IMP_FLOOR, IMP_CEIL, IMP_ROUND, IMP_SIGN, IMP_NOT, IMP_BOOL]
    for i in range(len(step_kinds)):  # small-loop(6: check kinds)
        summary = _mix(summary, _unary_case(ex, tape, step_kinds[i], steps, Float32(0), Float32(0)))
        cases += 1
    # binary kinds: a middle-axis broadcast b [3, 1, 5] and a leading-and-
    # trailing one b [1, 4, 1] (R = 15 terms an output)
    var sa = _shape3(3, 4, 5)
    var mid = _shape3(3, 1, 5)
    var lead = _shape3(1, 4, 1)
    summary = _mix(summary, _bin_case(ex, tape, IMP_POW, sa, mid, _positive(60, 4), _data(15, 5), False))
    summary = _mix(summary, _bin_case(ex, tape, IMP_POW, sa, lead, _positive(60, 6), _data(4, 7), False))
    summary = _mix(summary, _bin_case(ex, tape, IMP_MINIMUM, sa, mid, _quarters(60, 8), _quarters(15, 9), False))
    summary = _mix(summary, _bin_case(ex, tape, IMP_MAXIMUM, sa, lead, _quarters(60, 10), _quarters(4, 11), False))
    summary = _mix(summary, _bin_case(ex, tape, IMP_MAXIMUM, sa, mid, _quarters(60, 12), _quarters(15, 13), True))
    cases += 5
    var cmp_kinds: List[Int] = [IMP_EQ, IMP_GT, IMP_LT, IMP_GE, IMP_LE, IMP_NE, IMP_AND, IMP_OR]
    for i in range(len(cmp_kinds)):  # small-loop(8: check kinds)
        summary = _mix(summary, _bin_case(ex, tape, cmp_kinds[i], sa, mid, _quarters(60, 20 + i), _quarters(15, 30 + i), False))
        cases += 1
    # a negative base with integral exponents (pow's sign rule) on equal
    # shapes, forward only (the exponent's gradient needs log of the base)
    var negs = List[Float32](length=60, fill=Float32(0))
    var ints = List[Float32](length=60, fill=Float32(0))
    for i in range(60):  # small-loop(60: check data, host side)
        negs[i] = Float32(-0.5) - Float32(i % 7) * Float32(0.25)
        ints[i] = Float32((i % 5) - 2)
    summary = _mix(summary, _bin_case(ex, tape, IMP_POW, sa, sa, negs, ints, False, False))
    cases += 1
    summary = _mix(summary, _where_case(ex, tape))
    summary = _mix(summary, _expand_case(ex, tape, 3, 4, 5))
    summary = _mix(summary, _expand_case(ex, tape, 3, 70, 80))
    summary = _mix(summary, _argmax_case(ex, tape, False))
    summary = _mix(summary, _argmax_case(ex, tape, True))
    summary = _mix(summary, _pad_case(ex, tape))
    cases += 6
    # lane ag-import-lm-ops: the attention-mask ops
    summary = _mix(summary, _any_all_case(ex, tape))
    summary = _mix(summary, _cumsum_case(ex, tape, 3, 7, 5, False, False))
    summary = _mix(summary, _cumsum_case(ex, tape, 2, 150, 3, False, False))
    summary = _mix(summary, _cumsum_case(ex, tape, 2, 150, 3, True, False))
    summary = _mix(summary, _cumsum_case(ex, tape, 2, 150, 3, False, True))
    summary = _mix(summary, _cumsum_case(ex, tape, 1, 4500, 2, True, True))
    summary = _mix(summary, _index_case(ex, tape, 0, 1))
    summary = _mix(summary, _index_case(ex, tape, 1, 2))
    summary = _mix(summary, _index_case(ex, tape, 0, 2))
    cases += 9
    print("PASS import cases", cases, "summary", summary)
