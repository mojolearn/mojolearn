# SPDX-License-Identifier: Apache-2.0
"""The elementwise operations of imported models in the session
(docs/plans/MODEL_IMPORT.md section 3.3), moved from test_ops_import_kinds.py
(2026-11-14): values against independent float64 formulas (exactly where the
operation rounds nothing), gradients against float64 closed forms or central
differences, small odd non-board shapes. Every fixture runs on two
independent sessions and the second must give the same float32 words (the
former OPS_IMPORT_DIGEST twice-run check).

  0..11  smooth unary values (erf, gelu, gelu tanh, softplus, silu,
         leaky relu 0.1, elu 1.3, hard sigmoid, hard swish, sin, cos,
         reciprocal of |x| + 0.25) on (3, 7, 5) in [-2, 2): rtol 2e-6, atol 2e-6
  12     exact unary values: abs, floor, ceil, round (half to even), sign,
         logical_not, clip(-1.3, 2.1), clip(min=0.5) on arange(-23, 82) / 4
         with a -0.0
  13     pow (broadcast, a negative base with integer exponents), pow_scalar
         2.5 (rtol 3e-6), minimum and maximum broadcast (exact)
  14     the six comparisons, logical and/or, where with a broadcast condition
  15     expand (and equal to mul(full(1), x) word for word), argmax with ties
         and a NaN in float and int32 form, keepdims, constant pad with a
         negative count
  16..23 unary gradients (erf, gelu, gelu tanh, softplus, silu, elu, sin, cos)
         against float64 central differences (h = 1e-3): rtol 1e-4, atol 1e-5
  24     broadcast gradients through pow, where, expand and pad: rtol 1e-5,
         atol 1e-5
  25     any and all over one axis, two axes with keepdims, the last axis, and
         every element (a NaN counts as nonzero)
  26..41 cumsum on (3, 7, 5) axis 1, (3, 150, 2) axis 1, (2, 4500) axis -1,
         (70, 3) axis 0, each inclusive/exclusive and forward/reverse, value and
         the gradient of sum(W * cumsum(x)) against float64 running sums
         (rtol 1e-5, atol 1e-3)
  42..44 index by two index tensors (a negative and a repeated position) on
         axes (0, 1), (1, 2) and (0, 2) (NumPy's placement), value exact and
         gradient a float64 scatter-add (rtol 1e-6, atol 1e-6)

The NumPy draws are core_ops_common's seeded values over the same ranges and
shapes. The refusals stay in the Python test.
"""
from std.math import erf, exp, log, log1p, tanh, sin, cos, sqrt, floor, ceil, nan
from std.memory import bitcast
from autograd.ops_api import OpsSession
from sequence.exec_trait import Exec
from checks.native_tests.ops_higher import tensor, total, values, gradient, same_words
from checks.native_tests.core_ops_common import uniform, wide, allclose, exact, dims2, dims3, same_shape, opi
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime CORE_OPS_IMPORT_KINDS_FIXTURES = 45

comptime _ADD = 2
comptime _MUL = 4
comptime _FULL = 30
comptime _ERF = 48
comptime _GELU = 49
comptime _SOFTPLUS = 50
comptime _SILU = 51
comptime _LEAKY_RELU = 52
comptime _ELU = 53
comptime _HARD_SIGMOID = 54
comptime _HARD_SWISH = 55
comptime _CLIP = 56
comptime _ABS = 57
comptime _FLOOR = 58
comptime _CEIL = 59
comptime _ROUND = 60
comptime _SIN = 61
comptime _COS = 62
comptime _SIGN = 63
comptime _RECIPROCAL = 64
comptime _POW = 65
comptime _MINIMUM = 66
comptime _MAXIMUM = 67
comptime _WHERE = 68
comptime _COMPARE = 69
comptime _LOGICAL = 70
comptime _EXPAND = 71
comptime _ARGMAX = 72
comptime _PAD = 73
comptime _ANY_ALL = 74
comptime _CUMSUM = 75
comptime _INDEX = 76


def core_ops_import_kinds_records(fixture_id: Int) -> Bool:
    """The gradient fixtures record; the value fixtures run in an inference session."""
    return (fixture_id >= 16 and fixture_id <= 24) or fixture_id >= 26


# ------------------------------------------------------------------ shapes

def _count(dims: List[Int]) -> Int:
    var n = 1
    for d in dims:
        n *= d
    return n


def _unflat(i: Int, dims: List[Int]) -> List[Int]:
    var idx = List[Int](length=len(dims), fill=0)
    var r = i
    var k = len(dims) - 1
    while k >= 0:
        idx[k] = r % dims[k]
        r //= dims[k]
        k -= 1
    return idx^


def _flat(idx: List[Int], dims: List[Int]) -> Int:
    var f = 0
    for k in range(len(dims)):
        f = f * dims[k] + idx[k]
    return f


def _bsrc(i: Int, out_dims: List[Int], in_dims: List[Int]) -> Int:
    """The flat position of an equal-rank broadcast operand read by output i."""
    var idx = _unflat(i, out_dims)
    for k in range(len(in_dims)):
        if in_dims[k] == 1:
            idx[k] = 0
    return _flat(idx, in_dims)


def _broadcast(v: List[Float32], in_dims: List[Int], out_dims: List[Int]) -> List[Float32]:
    var n = _count(out_dims)
    var out = List[Float32](length=n, fill=0)
    for i in range(n):
        out[i] = v[_bsrc(i, out_dims, in_dims)]
    return out^


def _f32(v: List[Float64]) -> List[Float32]:
    var out = List[Float32](length=len(v), fill=0)
    for i in range(len(v)):
        out[i] = Float32(v[i])
    return out^


# ------------------------------------------------------------------ session spellings

def _u[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int) raises -> Int:
    var hs: List[Int] = [a]
    return opi(s, code, hs, List[Int](), List[Float32]())


def _uf[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, fs: List[Float32]) raises -> Int:
    var hs: List[Int] = [a]
    return opi(s, code, hs, List[Int](), fs)


def _ui[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, ints: List[Int]) raises -> Int:
    var hs: List[Int] = [a]
    return opi(s, code, hs, ints, List[Float32]())


def _b[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, b: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    return opi(s, code, hs, List[Int](), List[Float32]())


def _bi[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, b: Int, kind: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    var ints: List[Int] = [kind]
    return opi(s, code, hs, ints, List[Float32]())


def _smooth_code(k: Int) -> Int:
    var codes: List[Int] = [_ERF, _GELU, _GELU, _SOFTPLUS, _SILU, _LEAKY_RELU, _ELU, _HARD_SIGMOID,
                            _HARD_SWISH, _SIN, _COS, _RECIPROCAL]
    return codes[k]


def _apply[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, h: Int) raises -> Int:
    """The former `_apply(s, name, h)` for smooth kind k (Session defaults:
    hard_sigmoid alpha 0.2, beta 0.5; the test's leaky relu 0.1, elu 1.3)."""
    var code = _smooth_code(k)
    if code == _GELU:
        var approx: List[Int] = [1 if k == 2 else 0]
        return _ui(s, code, h, approx)
    if code == _LEAKY_RELU:
        var alpha: List[Float32] = [0.1]
        return _uf(s, code, h, alpha)
    if code == _ELU:
        var alpha: List[Float32] = [1.3]
        return _uf(s, code, h, alpha)
    if code == _HARD_SIGMOID:
        var ab: List[Float32] = [0.2, 0.5]
        return _uf(s, code, h, ab)
    return _u(s, code, h)


def _reference(k: Int, x: Float64) -> Float64:
    """The former `_unary_reference`, in float64."""
    if k == 0:
        return erf(x)
    if k == 1:
        return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
    if k == 2:
        return 0.5 * x * (1.0 + tanh(sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * x * x * x)))
    if k == 3:
        return log1p(exp(x))
    if k == 4:
        return x / (1.0 + exp(-x))
    if k == 5:
        return x if x >= 0 else 0.1 * x
    if k == 6:
        return x if x >= 0 else 1.3 * (exp(x) - 1.0)
    if k == 7:
        return min(max(0.2 * x + 0.5, 0.0), 1.0)
    if k == 8:
        return x * min(max(x / 6.0 + 0.5, 0.0), 1.0)
    if k == 9:
        return sin(x)
    if k == 10:
        return cos(x)
    return 1.0 / x


# ------------------------------------------------------------------ fixtures (outputs of one session)

def _run[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises -> List[List[Float32]]:
    var out = List[List[Float32]]()
    if fixture_id < 12:
        var x = uniform(105, 1, -2.0, 2.0)
        if fixture_id == 11:
            for i in range(105):
                x[i] = abs(x[i]) + Float32(0.25)
        var d = dims3(3, 7, 5)
        out.append(values(s, _apply(s, fixture_id, tensor(s, x, d)), d))
    elif fixture_id == 12:
        var d = dims3(3, 5, 7)
        var x = tensor(s, _exact_input(), d)
        var codes: List[Int] = [_ABS, _FLOOR, _CEIL, _ROUND, _SIGN]
        for c in codes:
            out.append(values(s, _u(s, c, x), d))
        var not_kind: List[Int] = [0]
        out.append(values(s, _ui(s, _LOGICAL, x, not_kind), d))
        var bounds: List[Float32] = [-1.3, 2.1]
        out.append(values(s, _uf(s, _CLIP, x, bounds), d))
        var low_only: List[Float32] = [0.5, Float32(1.0) / Float32(0.0)]
        out.append(values(s, _uf(s, _CLIP, x, low_only), d))
    elif fixture_id == 13:
        var d = dims3(3, 4, 5)
        var a = tensor(s, uniform(60, 2, 0.5, 2.0), d)
        var b = tensor(s, uniform(15, 3, -1.5, 1.5), dims3(3, 1, 5))
        out.append(values(s, _b(s, _POW, a, b), d))
        var e: List[Float32] = [2.5]
        out.append(values(s, _uf(s, _POW, a, e), d))
        out.append(values(s, _b(s, _POW, tensor(s, _neg_base(), d), tensor(s, _int_exponent(), d)), d))
        out.append(values(s, _b(s, _MINIMUM, a, b), d))
        out.append(values(s, _b(s, _MAXIMUM, a, b), d))
    elif fixture_id == 14:
        var d = dims3(3, 4, 5)
        var a = tensor(s, _cmp_a(), d)
        var b = tensor(s, _cmp_b(), dims3(3, 1, 5))
        var c = tensor(s, _cmp_c(), dims3(3, 1, 5))
        var y = tensor(s, uniform(4, 4, -2.0, 2.0), dims3(1, 4, 1))
        var kinds: List[Int] = [0, 5, 1, 3, 2, 4]  # equal, not_equal, greater, greater_equal, less, less_equal
        for k in kinds:
            out.append(values(s, _bi(s, _COMPARE, a, b, k), d))
        out.append(values(s, _bi(s, _LOGICAL, a, b, 1), d))
        out.append(values(s, _bi(s, _LOGICAL, a, b, 2), d))
        var hs: List[Int] = [c, a, y]
        out.append(values(s, opi(s, _WHERE, hs, List[Int](), List[Float32]()), d))
    elif fixture_id == 15:
        var d = dims3(3, 4, 5)
        var x = tensor(s, uniform(15, 5, -2.0, 2.0), dims3(3, 1, 5))
        var t = tensor(s, _ties(), dims3(5, 7, 6))
        out.append(values(s, _ui(s, _EXPAND, x, d), d))
        var one: List[Float32] = [1.0]
        var full = opi(s, _FULL, List[Int](), d, one)
        out.append(values(s, _b(s, _MUL, full, x), d))
        var am: List[Int] = [1, 0, 1]
        out.append(values(s, _ui(s, _ARGMAX, t, am), dims2(5, 6)))
        var ak: List[Int] = [-2, 1, 0]
        out.append(values(s, _ui(s, _ARGMAX, t, ak), dims3(5, 1, 6)))
        var pads: List[Int] = [1, 0, 2, 0, 3, -1]
        var hs: List[Int] = [x]
        var half: List[Float32] = [0.5]
        out.append(values(s, opi(s, _PAD, hs, pads, half), dims3(4, 4, 6)))
    elif fixture_id < 24:
        var k = _grad_kind(fixture_id - 16)
        var d = dims3(2, 3, 5)
        var x = tensor(s, uniform(30, 7, -1.5, 1.5), d, True)
        s.backward(total(s, _apply(s, k, x)), False)
        out.append(gradient(s, x, d))
    elif fixture_id == 24:
        var d = dims3(3, 4, 5)
        var av = uniform(60, 8, 0.5, 2.0)
        var a = tensor(s, av, d, True)
        var b = tensor(s, uniform(15, 9, -1.0, 1.0), dims3(3, 1, 5), True)
        var c = tensor(s, _cmp_c(), dims3(3, 1, 5))
        var x = tensor(s, uniform(15, 10, -2.0, 2.0), dims3(3, 1, 5), True)
        var hs: List[Int] = [c, a, b]
        var loss = _b(s, _ADD, total(s, _b(s, _POW, a, b)), total(s, opi(s, _WHERE, hs, List[Int](), List[Float32]())))
        loss = _b(s, _ADD, loss, total(s, _b(s, _MUL, _ui(s, _EXPAND, x, d), tensor(s, av, d))))
        var pads: List[Int] = [0, 1, 0, 2, 0, 0]
        var px: List[Int] = [x]
        var three: List[Float32] = [3.0]
        loss = _b(s, _ADD, loss, total(s, opi(s, _PAD, px, pads, three)))
        s.backward(loss, False)
        out.append(gradient(s, a, d))
        out.append(gradient(s, b, dims3(3, 1, 5)))
        out.append(gradient(s, x, dims3(3, 1, 5)))
    elif fixture_id == 25:
        var x = tensor(s, _mask_data(), dims3(4, 5, 6))
        var a1: List[Int] = [0, -1, 1]
        var l1: List[Int] = [1, -1, 1]
        var a02: List[Int] = [0, 1, 0, 2]
        var l2: List[Int] = [1, 1, -1]
        var aa: List[Int] = [0, -1]
        var la: List[Int] = [1, -1]
        out.append(values(s, _ui(s, _ANY_ALL, x, a1), dims2(4, 6)))
        out.append(values(s, _ui(s, _ANY_ALL, x, l1), dims2(4, 6)))
        out.append(values(s, _ui(s, _ANY_ALL, x, a02), dims3(1, 5, 1)))
        out.append(values(s, _ui(s, _ANY_ALL, x, l2), dims3(4, 5, 1)))
        out.append(values(s, _ui(s, _ANY_ALL, x, aa), dims2(1, 1)))
        out.append(values(s, _ui(s, _ANY_ALL, x, la), dims2(1, 1)))
    elif fixture_id < 42:
        var c = _cumsum_case(fixture_id - 26)
        var d = c.dims.copy()
        var n = _count(d)
        var x = tensor(s, uniform(n, 22, -2.0, 2.0), d, True)
        var words: List[Int] = [c.axis, Int(c.exclusive), Int(c.reverse)]
        var y = _ui(s, _CUMSUM, x, words)
        same_shape(s, y, d, "cumsum keeps the shape")
        s.backward(total(s, _b(s, _MUL, y, tensor(s, uniform(n, 23, -2.0, 2.0), d))), False)
        out.append(values(s, y, d))
        out.append(gradient(s, x, d))
    else:
        var axes = _index_axes(fixture_id - 42)
        var d = dims3(4, 5, 6)
        var x = tensor(s, uniform(120, 24, -2.0, 2.0), d, True)
        var iv: List[Float32] = [1.0, -1.0, 1.0]
        var jv: List[Float32] = [2.0, 0.0]
        var od = _index_dims(fixture_id - 42)
        var hs: List[Int] = [x, tensor(s, iv, dims2(3, 1)), tensor(s, jv, dims2(1, 2))]
        var y = opi(s, _INDEX, hs, axes, List[Float32]())
        same_shape(s, y, od, "index result shape")
        s.backward(total(s, _b(s, _MUL, y, tensor(s, uniform(_count(od), 25, -2.0, 2.0), od))), False)
        out.append(values(s, y, od))
        out.append(gradient(s, x, d))
    return out^


# ------------------------------------------------------------------ inputs

def _exact_input() -> List[Float32]:
    var x = List[Float32](length=105, fill=0)
    for i in range(105):
        x[i] = Float32(i - 23) * Float32(0.25)
    x[0] = Float32(-0.0)
    return x^


def _neg_base() -> List[Float32]:
    var v = List[Float32](length=60, fill=0)
    for i in range(60):
        v[i] = -Float32(i % 7) * Float32(0.25) - Float32(0.5)
    return v^


def _int_exponent() -> List[Float32]:
    var v = List[Float32](length=60, fill=0)
    for i in range(60):
        v[i] = Float32(i % 5) - Float32(2)
    return v^


def _cmp_a() -> List[Float32]:
    return _int_exponent()


def _cmp_b() -> List[Float32]:
    var v = List[Float32](length=15, fill=0)
    for i in range(15):
        v[i] = Float32(i % 3) - Float32(1)
    return v^


def _cmp_c() -> List[Float32]:
    var v = List[Float32](length=15, fill=0)
    for i in range(15):
        v[i] = Float32(i % 2)
    return v^


def _half_steps(v: List[Float32]) -> List[Float32]:
    """np.round(v * 2) / 2 (half to even), many exact ties."""
    var out = List[Float32](length=len(v), fill=0)
    for i in range(len(v)):
        out[i] = _round_even(v[i] * Float32(2)) / Float32(2)
    return out^


def _round_even(x: Float32) -> Float32:
    var f = floor(Float64(x))
    var r = Float64(x) - f
    if r > 0.5 or (r == 0.5 and Int(f) % 2 != 0):
        f += 1.0
    return Float32(f)


def _ties() -> List[Float32]:
    var t = _half_steps(uniform(210, 6, -2.0, 2.0))
    t[(1 * 7 + 3) * 6 + 2] = nan[DType.float32]()
    return t^


def _mask_data() -> List[Float32]:
    var x = _half_steps(uniform(120, 21, -2.0, 2.0))
    var i = 0
    while i < 120:
        x[i] = nan[DType.float32]()
        i += 17
    for t in range(5):
        x[(2 * 5 + t) * 6 + 1] = 0.75
    return x^


def _grad_kind(k: Int) -> Int:
    var kinds: List[Int] = [0, 1, 2, 3, 4, 6, 9, 10]  # erf gelu gelu_tanh softplus silu elu sin cos
    return kinds[k]


@fieldwise_init
struct CumsumCase(Movable):
    var dims: List[Int]
    var axis: Int
    var exclusive: Bool
    var reverse: Bool


def _cumsum_case(k: Int) -> CumsumCase:
    var shape = k // 4
    var flags = k % 4
    var dims: List[Int]
    var axis: Int
    if shape == 0:
        dims = dims3(3, 7, 5)
        axis = 1
    elif shape == 1:
        dims = dims3(3, 150, 2)
        axis = 1
    elif shape == 2:
        dims = dims2(2, 4500)
        axis = -1
    else:
        dims = dims2(70, 3)
        axis = 0
    return CumsumCase(dims^, axis, flags == 1 or flags == 3, flags >= 2)


def _index_axes(k: Int) -> List[Int]:
    if k == 0:
        return dims2(0, 1)
    if k == 1:
        return dims2(1, 2)
    return dims2(0, 2)


def _index_dims(k: Int) -> List[Int]:
    """NumPy's placement: consecutive indexed axes keep the broadcast (3, 2)
    where the first of them stood; split axes put it first."""
    if k == 0:
        return dims3(3, 2, 6)
    if k == 1:
        return dims3(4, 3, 2)
    return dims3(3, 2, 5)


# ------------------------------------------------------------------ oracles

def _check(fixture_id: Int, r: List[List[Float32]]) raises:
    var tag = "import fixture " + String(fixture_id)
    if fixture_id < 12:
        var x = uniform(105, 1, -2.0, 2.0)
        if fixture_id == 11:
            for i in range(105):
                x[i] = abs(x[i]) + Float32(0.25)
        var want = List[Float64](length=105, fill=0)
        for i in range(105):
            want[i] = _reference(fixture_id, Float64(x[i]))
        allclose(r[0], want, 2e-6, 2e-6, tag + " unary value")
    elif fixture_id == 12:
        var x = _exact_input()
        var n = 105
        var w = List[List[Float32]]()
        for _ in range(8):
            w.append(List[Float32](length=n, fill=0))
        for i in range(n):
            var v = x[i]
            w[0][i] = abs(v)
            w[1][i] = Float32(floor(Float64(v)))
            w[2][i] = Float32(ceil(Float64(v)))
            w[3][i] = _round_even(v)
            w[4][i] = Float32(1) if v > 0 else (Float32(-1) if v < 0 else v)
            w[5][i] = Float32(1) if v == 0 else Float32(0)
            w[6][i] = min(max(v, Float32(-1.3)), Float32(2.1))
            w[7][i] = max(v, Float32(0.5))
        var names: List[String] = ["abs", "floor", "ceil", "round", "sign", "logical_not", "clip", "clip min"]
        for k in range(8):
            exact(r[k], w[k], tag + " " + names[k])
    elif fixture_id == 13:
        var d = dims3(3, 4, 5)
        var bd = dims3(3, 1, 5)
        var a = uniform(60, 2, 0.5, 2.0)
        var b = uniform(15, 3, -1.5, 1.5)
        var nb = _neg_base()
        var ie = _int_exponent()
        var p = List[Float64](length=60, fill=0)
        var ps = List[Float64](length=60, fill=0)
        var pn = List[Float64](length=60, fill=0)
        var mn = List[Float32](length=60, fill=0)
        var mx = List[Float32](length=60, fill=0)
        for i in range(60):
            var bv = b[_bsrc(i, d, bd)]
            p[i] = Float64(a[i]) ** Float64(bv)
            ps[i] = Float64(a[i]) ** 2.5
            pn[i] = Float64(nb[i]) ** Float64(ie[i])
            mn[i] = min(a[i], bv)
            mx[i] = max(a[i], bv)
        allclose(r[0], p, 3e-6, 0.0, tag + " pow broadcast")
        allclose(r[1], ps, 3e-6, 0.0, tag + " pow_scalar")
        allclose(r[2], pn, 3e-6, 0.0, tag + " pow of a negative base")
        exact(r[3], mn, tag + " minimum")
        exact(r[4], mx, tag + " maximum")
    elif fixture_id == 14:
        var d = dims3(3, 4, 5)
        var bd = dims3(3, 1, 5)
        var a = _cmp_a()
        var b = _cmp_b()
        var c = _cmp_c()
        var y = uniform(4, 4, -2.0, 2.0)
        var w = List[List[Float32]]()
        for _ in range(9):
            w.append(List[Float32](length=60, fill=0))
        for i in range(60):
            var av = a[i]
            var bv = b[_bsrc(i, d, bd)]
            w[0][i] = Float32(1) if av == bv else Float32(0)
            w[1][i] = Float32(1) if av != bv else Float32(0)
            w[2][i] = Float32(1) if av > bv else Float32(0)
            w[3][i] = Float32(1) if av >= bv else Float32(0)
            w[4][i] = Float32(1) if av < bv else Float32(0)
            w[5][i] = Float32(1) if av <= bv else Float32(0)
            w[6][i] = Float32(1) if (av != 0 and bv != 0) else Float32(0)
            w[7][i] = Float32(1) if (av != 0 or bv != 0) else Float32(0)
            var cv = c[_bsrc(i, d, bd)]
            w[8][i] = av if cv != 0 else y[_bsrc(i, d, dims3(1, 4, 1))]
        var names: List[String] = ["equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
                                   "logical_and", "logical_or", "where"]
        for k in range(9):
            exact(r[k], w[k], tag + " " + names[k])
    elif fixture_id == 15:
        var d = dims3(3, 4, 5)
        var xd = dims3(3, 1, 5)
        var x = uniform(15, 5, -2.0, 2.0)
        exact(r[0], _broadcast(x, xd, d), tag + " expand")
        same_words(r[1], r[0], tag + " expand equals mul(full(1), x)")
        var t = _ties()
        var am = List[Float32](length=30, fill=0)
        for i in range(5):
            for k in range(6):
                # NumPy's argmax: a NaN is the largest (its first position
                # wins), else the first of equal largest values
                var best = 0
                for j in range(1, 7):
                    var cur = t[(i * 7 + best) * 6 + k]
                    if cur != cur:
                        break
                    var v = t[(i * 7 + j) * 6 + k]
                    if v != v or v > cur:
                        best = j
                am[i * 6 + k] = Float32(best)
        exact(r[2], am, tag + " argmax (NaN largest, first of equals)")
        require(len(r[3]) == 30, tag + " argmax int32 extent")
        for i in range(30):
            require(Int(bitcast[DType.int32](r[3][i])) == Int(am[i]), tag + " argmax int32 words at " + String(i))
        var pd = List[Float32](length=96, fill=0)
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    var si = i - 1
                    var sk = k - 2
                    var inside = si >= 0 and si < 3 and j == 0 and sk >= 0 and sk < 5
                    pd[(i * 4 + j) * 6 + k] = x[si * 5 + sk] if inside else Float32(0.5)
        exact(r[4], pd, tag + " pad")
    elif fixture_id < 24:
        var k = _grad_kind(fixture_id - 16)
        var x = uniform(30, 7, -1.5, 1.5)
        var want = List[Float64](length=30, fill=0)
        for i in range(30):
            var v = Float64(x[i])
            want[i] = (_reference(k, v + 1e-3) - _reference(k, v - 1e-3)) / 2e-3
        allclose(r[0], want, 1e-4, 1e-5, tag + " unary gradient")
    elif fixture_id == 24:
        var d = dims3(3, 4, 5)
        var bd = dims3(3, 1, 5)
        var a = wide(uniform(60, 8, 0.5, 2.0))
        var b = wide(uniform(15, 9, -1.0, 1.0))
        var c = _cmp_c()
        var want_a = List[Float64](length=60, fill=0)
        var want_b = List[Float64](length=15, fill=0)
        var want_x = List[Float64](length=15, fill=1.0)
        for i in range(60):
            var j = _bsrc(i, d, bd)
            var take = c[j] != 0
            want_a[i] = b[j] * (a[i] ** (b[j] - 1.0)) + (1.0 if take else 0.0)
            want_b[j] += (a[i] ** b[j]) * log(a[i]) + (0.0 if take else 1.0)
            want_x[j] += a[i]
        allclose(r[0], want_a, 1e-5, 1e-5, tag + " d a")
        allclose(r[1], want_b, 1e-5, 1e-5, tag + " d b")
        allclose(r[2], want_x, 1e-5, 1e-5, tag + " d x")
    elif fixture_id == 25:
        var x = _mask_data()
        var nz = List[Bool](length=120, fill=False)
        for i in range(120):
            nz[i] = x[i] != 0  # a NaN compares unequal: nonzero
        var a1 = List[Float32](length=24, fill=0)
        var l1 = List[Float32](length=24, fill=0)
        for i in range(4):
            for k in range(6):
                var any_ = False
                var all_ = True
                for j in range(5):
                    any_ = any_ or nz[(i * 5 + j) * 6 + k]
                    all_ = all_ and nz[(i * 5 + j) * 6 + k]
                a1[i * 6 + k] = Float32(1) if any_ else Float32(0)
                l1[i * 6 + k] = Float32(1) if all_ else Float32(0)
        var a02 = List[Float32](length=5, fill=0)
        for j in range(5):
            var any_ = False
            for i in range(4):
                for k in range(6):
                    any_ = any_ or nz[(i * 5 + j) * 6 + k]
            a02[j] = Float32(1) if any_ else Float32(0)
        var l2 = List[Float32](length=20, fill=0)
        for i in range(4):
            for j in range(5):
                var all_ = True
                for k in range(6):
                    all_ = all_ and nz[(i * 5 + j) * 6 + k]
                l2[i * 5 + j] = Float32(1) if all_ else Float32(0)
        exact(r[0], a1, tag + " any over axis 1")
        exact(r[1], l1, tag + " all over axis 1")
        exact(r[2], a02, tag + " any over axes (0, 2), keepdims")
        exact(r[3], l2, tag + " all over the last axis, keepdims")
        require(r[4][0] == 1.0 and r[5][0] == 0.0, tag + " any and all of every element")
    elif fixture_id < 42:
        var c = _cumsum_case(fixture_id - 26)
        var n = _count(c.dims)
        var x = wide(uniform(n, 22, -2.0, 2.0))
        var w = wide(uniform(n, 23, -2.0, 2.0))
        allclose(r[0], _scan(x, c.dims, c.axis, c.exclusive, c.reverse), 1e-5, 1e-3, tag + " cumsum value")
        allclose(r[1], _scan(w, c.dims, c.axis, c.exclusive, not c.reverse), 1e-5, 1e-3, tag + " cumsum gradient")
    else:
        var k = fixture_id - 42
        var x = uniform(120, 24, -2.0, 2.0)
        var od = _index_dims(k)
        var m = _count(od)
        var wv = wide(uniform(m, 25, -2.0, 2.0))
        var want = List[Float32](length=m, fill=0)
        var gw = List[Float64](length=120, fill=0)
        var d = dims3(4, 5, 6)
        for o in range(m):
            var oi = _unflat(o, od)
            var src = List[Int](length=3, fill=0)
            var bi: Int
            var bj: Int
            if k == 0:
                bi = oi[0]
                bj = oi[1]
                src[0] = _wrap(_i_at(bi), 4)
                src[1] = _j_at(bj)
                src[2] = oi[2]
            elif k == 1:
                bi = oi[1]
                bj = oi[2]
                src[0] = oi[0]
                src[1] = _wrap(_i_at(bi), 5)
                src[2] = _j_at(bj)
            else:
                bi = oi[0]
                bj = oi[1]
                src[0] = _wrap(_i_at(bi), 4)
                src[1] = oi[2]
                src[2] = _j_at(bj)
            var f = _flat(src, d)
            want[o] = x[f]
            gw[f] += wv[o]
        exact(r[0], want, tag + " index value")
        allclose(r[1], gw, 1e-6, 1e-6, tag + " index gradient (scatter-add)")


def _i_at(b: Int) -> Int:
    """I = [[1], [-1], [1]], broadcast row b of (3, 2)."""
    var i: List[Int] = [1, -1, 1]
    return i[b]


def _j_at(b: Int) -> Int:
    """J = [[2, 0]], broadcast column b of (3, 2)."""
    return 2 if b == 0 else 0


def _wrap(i: Int, n: Int) -> Int:
    return i + n if i < 0 else i


def _scan(v: List[Float64], dims: List[Int], axis_in: Int, exclusive: Bool, reverse: Bool) -> List[Float64]:
    """The former `_np_cumsum` in float64: running sums along `axis`."""
    var axis = axis_in + len(dims) if axis_in < 0 else axis_in
    var n = len(v)
    var out = List[Float64](length=n, fill=0)
    var length = dims[axis]
    var inner = 1
    for k in range(axis + 1, len(dims)):
        inner *= dims[k]
    var outer = n // (length * inner)
    for o in range(outer):
        for q in range(inner):
            var acc = Float64(0)
            for step in range(length):
                var j = length - 1 - step if reverse else step
                var at = (o * length + j) * inner + q
                if exclusive:
                    out[at] = acc
                    acc += v[at]
                else:
                    acc += v[at]
                    out[at] = acc
    return out^


def run_core_ops_import_kinds_fixture[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], mut again: OpsSession[E], fixture_id: Int
) raises:
    require(fixture_id >= 0 and fixture_id < CORE_OPS_IMPORT_KINDS_FIXTURES, "unknown import-kinds fixture")
    var first = _run(s, fixture_id)
    var second = _run(again, fixture_id)
    require(len(first) == len(second), "two runs give different output counts")
    for k in range(len(first)):
        same_words(first[k], second[k], "import fixture " + String(fixture_id) + ": two runs differ, output " + String(k))
    _check(fixture_id, first)


# ---- The host column (was core_ops_import_kinds_cpu_suite.mojo) ----
# The import elementwise kinds against float64 oracles: each fixture on its own two host sessions, the fixtures run
# across host cores with a private failure slot each.
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(CORE_OPS_IMPORT_KINDS_FIXTURES)
    def task(fixture_id: Int) {imm cases}:
        try:
            var record = core_ops_import_kinds_records(fixture_id)
            var s = OpsSession[HostExec](HostExec(), record)
            var again = OpsSession[HostExec](HostExec(), record)
            try:
                run_core_ops_import_kinds_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, CORE_OPS_IMPORT_KINDS_FIXTURES)


# ---- The device column (was core_ops_import_kinds_gpu_suite.mojo) ----
# The import elementwise kinds against float64 oracles on the device, in order.
#
# DeviceExec shares sequence_ctx() and its pool, so independent concurrent
# sessions would race that context's ownership (as in ops_higher_gpu); the
# fixtures run one after another and each kernel keeps its own parallelism.
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(CORE_OPS_IMPORT_KINDS_FIXTURES)
    for fixture_id in range(CORE_OPS_IMPORT_KINDS_FIXTURES):
        try:
            var record = core_ops_import_kinds_records(fixture_id)
            var s = OpsSession[DeviceExec](DeviceExec(), record)
            var again = OpsSession[DeviceExec](DeviceExec(), record)
            try:
                run_core_ops_import_kinds_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            cases.fail(fixture_id, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("core_ops_import_kinds", cpu=True, gpu=True, serial="", run=run_cases)
