# SPDX-License-Identifier: Apache-2.0
"""Higher-order gradients of the native ops session (autograd/ops_api.mojo
`backward_graph`, `grad_handle`, `vjp`, `jvp`, `hvp`), moved from
test_ops_higher.py and test_higher_rank_native.py (2026-11-14).

No Python participates. The session calls are the ones the public
`mojolearn.ops.Session` makes (providers/python/autograd_ops_frontend_pyapi_door.mojo):
the same op codes, ints and floats.

Oracles and tolerances are the former tests':
  * ops_higher fixtures 0..6: float64 NumPy nested central differences
    (Richardson from h and h/2) and Jacobians, stored with their inputs under
    checks/native_tests/fixtures/ops_higher/ by its generate.py; the closed
    forms (6 x v for the vjp of x*x*x, the tanh network's du/dx and d2u/dx2)
    are computed here in float64. Tolerance |a - e| <= 3e-4 + 3e-4 |e|, every
    value finite. Each fixture that the former test repeated runs again on the
    second session and must give the same float32 words.
  * higher-rank fixtures 7..15: value = arange(12)/16, direction =
    arange(1, 13)/8 in rank 3 and rank 4 shapes, against float64 exp;
    tolerance |a - e| <= 2e-6 + 2e-5 |e| (NumPy assert_allclose's rule).
The refusals (create_graph with accumulate, grad_tensor after an unrecorded
backward, a direction of the wrong full shape) stay in the Python tests.
"""
from std.math import exp, isfinite, tanh
from std.memory import bitcast, unsafe_memcpy
from autograd.ops_api import OpsSession
from autograd.rankn import ShapeN
from core.cast_in import CastIn
from core.hostptr import f32_ptr
from sequence.exec_trait import Exec

comptime OPS_HIGHER_FIXTURES = 16
comptime _DIR = "checks/native_tests/fixtures/ops_higher/"

# Op codes and words of the public session (autograd/ops_api.mojo).
comptime _MATMUL = 1
comptime _ADD = 2
comptime _MUL = 4
comptime _TANH = 8
comptime _EXP = 9
comptime _SCALE = 12
comptime _RESHAPE = 14
comptime _LINEAR = 19
comptime _LAYER_NORM = 20
comptime _MSE = 24
comptime _CROSS_ENTROPY = 28
comptime _RESHAPE_N = 33
comptime _SUM_N = 35
comptime _ACT_NONE = 0
comptime _ACT_TANH = 3
comptime _REDUCE_MEAN = 0


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


# ------------------------------------------------------------------ fixtures on disk

def load_f32(case: String, name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](length=count, fill=0)
    with open(_DIR + case + "-" + name + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 4, "ops_higher input fixture extent changed: " + case + "-" + name)
        unsafe_memcpy(dest=result.unsafe_ptr().bitcast[UInt8](), src=bytes.unsafe_ptr(), count=count * 4)
        _ = bytes^
    return result^


def load_f64(case: String, name: String, count: Int) raises -> List[Float64]:
    var result = List[Float64](length=count, fill=0)
    with open(_DIR + case + "-" + name + ".f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 8, "ops_higher oracle fixture extent changed: " + case + "-" + name)
        unsafe_memcpy(dest=result.unsafe_ptr().bitcast[UInt8](), src=bytes.unsafe_ptr(), count=count * 8)
        _ = bytes^
    return result^


# ------------------------------------------------------------------ session helpers

def _count(dims: List[Int]) -> Int:
    var n = 1
    for d in dims:
        n *= d
    return n


def tensor[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], values: List[Float32], dims: List[Int], requires_grad: Bool = False
) raises -> Int:
    var cols = dims[len(dims) - 1]
    var count = _count(dims)
    require(len(values) == count, "fixture extent does not match its shape")
    var src = CastIn.f32_c(Int(values.unsafe_ptr()), count // cols, cols)
    return s.upload_nd(src, ShapeN.of(dims), requires_grad)


def op1[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int) raises -> Int:
    var hs: List[Int] = [a]
    return s.op(code, hs, List[Int](), List[Float32]())


def op2[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, b: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    return s.op(code, hs, List[Int](), List[Float32]())


def total[E: Exec & Movable & Deinitable](mut s: OpsSession[E], a: Int) raises -> Int:
    """`Session.sum(a)`: every element into a (1, 1) tensor."""
    var hs: List[Int] = [a]
    var ints: List[Int] = [-1]
    return s.op(_SUM_N, hs, ints, List[Float32]())


def scale[E: Exec & Movable & Deinitable](mut s: OpsSession[E], a: Int, factor: Float32) raises -> Int:
    var hs: List[Int] = [a]
    var fs: List[Float32] = [factor]
    return s.op(_SCALE, hs, List[Int](), fs)


def reshape[E: Exec & Movable & Deinitable](mut s: OpsSession[E], a: Int, dims: List[Int]) raises -> Int:
    var hs: List[Int] = [a]
    var ints = List[Int]()
    var code = _RESHAPE
    if len(dims) != 2:
        code = _RESHAPE_N
        ints.append(0)
    for d in dims:
        ints.append(d)
    return s.op(code, hs, ints, List[Float32]())


def linear[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], x: Int, w: Int, b: Int, activation: Int
) raises -> Int:
    var hs: List[Int] = [x, w]
    if b != 0:
        hs.append(b)
    var ints: List[Int] = [activation]
    return s.op(_LINEAR, hs, ints, List[Float32]())


def loss[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, b: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    var ints: List[Int] = [_REDUCE_MEAN]
    return s.op(code, hs, ints, List[Float32]())


def values[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, dims: List[Int]) raises -> List[Float32]:
    """`Session.numpy(h)`, after checking the full shape."""
    require(s.shape_nd(h).same(ShapeN.of(dims)), "result shape")
    var out = List[Float32](length=_count(dims), fill=0)
    s.download(h, f32_ptr(Int(out.unsafe_ptr())))
    return out^


def gradient[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, dims: List[Int]) raises -> List[Float32]:
    """`Session.grad(h)`."""
    var out = List[Float32](length=_count(dims), fill=0)
    s.gradient_into(h, f32_ptr(Int(out.unsafe_ptr())))
    return out^


# ------------------------------------------------------------------ comparisons

def close_ho(actual: List[Float32], expected: List[Float64], what: String) raises:
    """test_ops_higher's `_close`: finite and |a - e| <= 3e-4 + 3e-4 |e|."""
    require(len(actual) == len(expected), what + ": extent")
    for i in range(len(actual)):
        var a = Float64(actual[i])
        require(isfinite(a), what + ": non-finite at " + String(i))
        require(abs(a - expected[i]) <= 3e-4 + 3e-4 * abs(expected[i]),
                what + ": at " + String(i) + " got " + String(a) + ", expected " + String(expected[i]))


def close_hr(actual: List[Float32], expected: List[Float64], what: String) raises:
    """test_higher_rank_native's `close`: rtol 2e-5, atol 2e-6."""
    require(len(actual) == len(expected), what + ": extent")
    for i in range(len(actual)):
        var a = Float64(actual[i])
        require(abs(a - expected[i]) <= 2e-6 + 2e-5 * abs(expected[i]),
                what + ": at " + String(i) + " got " + String(a) + ", expected " + String(expected[i]))


def same_words(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": extent")
    for i in range(len(a)):
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": a second run differs at " + String(i))


def _wide(x: List[Float32]) -> List[Float64]:
    var out = List[Float64](length=len(x), fill=0)
    for i in range(len(x)):
        out[i] = Float64(x[i])
    return out^


# ------------------------------------------------------------------ test_ops_higher

def _example_run[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises -> List[List[Float32]]:
    var dx: List[Int] = [8, 3]
    var dw: List[Int] = [3, 2]
    var dy: List[Int] = [8, 2]
    var x = tensor(s, load_f32("example", "x", 24), dx)
    var w = tensor(s, load_f32("example", "w", 6), dw, True)
    var y = tensor(s, load_f32("example", "y", 16), dy)
    var l = loss(s, _MSE, linear(s, x, w, 0, _ACT_TANH), y)
    s.backward_graph(l)
    var g = s.grad_handle(w)
    var first = values(s, g, dw)
    var v = tensor(s, load_f32("example", "v", 6), dw)
    s.backward(total(s, op2(s, _MUL, g, v)), False)
    var hv = gradient(s, w, dw)
    var hs: List[Int] = [w]
    var vs: List[Int] = [v]
    var hv2 = values(s, s.hvp(l, hs, vs)[0], dw)
    var out: List[List[Float32]] = [first^, hv^, hv2^]
    return out^


def example_hvp[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut again: OpsSession[E]) raises:
    var r = _example_run(s)
    var oracle = load_f64("example", "hvp", 6)
    close_ho(r[0], load_f64("example", "gradient", 6), "first-order gradient under create_graph")
    close_ho(r[1], oracle, "H v by two backward calls")
    close_ho(r[2], oracle, "H v by hvp")
    var r2 = _example_run(again)
    for k in range(3):
        same_words(r[k], r2[k], "example hvp output " + String(k))


def _softmax_run[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises -> List[List[Float32]]:
    var dw: List[Int] = [3, 4]
    var db: List[Int] = [1, 4]
    var dx: List[Int] = [6, 3]
    var dc: List[Int] = [6, 1]
    var w = tensor(s, load_f32("softmax_ce", "w", 12), dw, True)
    var b = tensor(s, load_f32("softmax_ce", "b", 4), db, True)
    var x = tensor(s, load_f32("softmax_ce", "x", 18), dx)
    var c = tensor(s, load_f32("softmax_ce", "c", 6), dc)
    var l = loss(s, _CROSS_ENTROPY, linear(s, x, w, b, _ACT_NONE), c)
    var hs: List[Int] = [w, b]
    var vs: List[Int] = [tensor(s, load_f32("softmax_ce", "vw", 12), dw), tensor(s, load_f32("softmax_ce", "vb", 4), db)]
    var hv = s.hvp(l, hs, vs)
    var out: List[List[Float32]] = [values(s, hv[0], dw), values(s, hv[1], db)]
    return out^


def hvp_softmax_cross_entropy[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut again: OpsSession[E]) raises:
    var r = _softmax_run(s)
    close_ho(r[0], load_f64("softmax_ce", "hw", 12), "softmax cross-entropy H v (weights)")
    close_ho(r[1], load_f64("softmax_ce", "hb", 4), "softmax cross-entropy H v (bias)")
    var r2 = _softmax_run(again)
    same_words(r[0], r2[0], "softmax cross-entropy H v (weights)")
    same_words(r[1], r2[1], "softmax cross-entropy H v (bias)")


def hvp_zero_curvature[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var dx: List[Int] = [4, 3]
    var dw: List[Int] = [3, 2]
    var w = tensor(s, load_f32("zero_curvature", "w", 6), dw, True)
    var l = total(s, op2(s, _MATMUL, tensor(s, load_f32("zero_curvature", "x", 12), dx), w))
    var hs: List[Int] = [w]
    var vs: List[Int] = [tensor(s, List[Float32](length=6, fill=1.0), dw)]
    var hv = values(s, s.hvp(l, hs, vs)[0], dw)
    for i in range(len(hv)):
        require(hv[i] == 0, "zero curvature H v is not zero at " + String(i))


def _adjoint_run[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises -> List[List[Float32]]:
    var dx: List[Int] = [3, 2]
    var dw1: List[Int] = [2, 3]
    var x = tensor(s, load_f32("adjoint", "x", 6), dx, True)
    var w1 = tensor(s, load_f32("adjoint", "w1", 6), dw1, True)
    var w2 = tensor(s, load_f32("adjoint", "w2", 6), dx)
    var out = op1(s, _TANH, op2(s, _MATMUL, op1(s, _TANH, op2(s, _MATMUL, x, w1)), w2))
    var hs: List[Int] = [x, w1]
    var jt = s.vjp(out, hs, tensor(s, load_f32("adjoint", "v", 6), dx), False)
    var us: List[Int] = [tensor(s, load_f32("adjoint", "ux", 6), dx), tensor(s, load_f32("adjoint", "uw", 6), dw1)]
    var ju = s.jvp(out, hs, us)
    var result: List[List[Float32]] = [values(s, jt[0], dx), values(s, jt[1], dw1), values(s, ju, dx)]
    return result^


def jvp_vjp_adjoint[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut again: OpsSession[E]) raises:
    var r = _adjoint_run(s)
    close_ho(r[0], load_f64("adjoint", "jtv_x", 6), "J^T v (x)")
    close_ho(r[1], load_f64("adjoint", "jtv_w", 6), "J^T v (w1)")
    close_ho(r[2], load_f64("adjoint", "ju", 6), "J u")
    var v = _wide(load_f32("adjoint", "v", 6))
    var ux = _wide(load_f32("adjoint", "ux", 6))
    var uw = _wide(load_f32("adjoint", "uw", 6))
    var lhs = Float64(0)
    var rhs = Float64(0)
    for i in range(6):
        lhs += v[i] * Float64(r[2][i])
        rhs += Float64(r[0][i]) * ux[i] + Float64(r[1][i]) * uw[i]
    require(abs(lhs - rhs) <= 1e-4 * (1 + abs(lhs)), "adjoint identity <v, J u> = <J^T v, u>")
    var r2 = _adjoint_run(again)
    for k in range(3):
        same_words(r[k], r2[k], "vjp/jvp output " + String(k))


def vjp_create_graph[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    """A vjp with create_graph is differentiable: d/dx <J^T v, 1> for
    out = x * x * x (elementwise) is 6 x v."""
    var d: List[Int] = [1, 3]
    var xv: List[Float32] = [0.5, -1.0, 1.5]
    var vv: List[Float32] = [1.0, 2.0, -0.5]
    var x = tensor(s, xv, d, True)
    var out = op2(s, _MUL, op2(s, _MUL, x, x), x)
    var hs: List[Int] = [x]
    var g = s.vjp(out, hs, tensor(s, vv, d), True)
    s.backward(total(s, g[0]), False)
    var second = gradient(s, x, d)
    var expected = List[Float64](length=3, fill=0)
    for i in range(3):
        expected[i] = 6 * Float64(xv[i]) * Float64(vv[i])
    close_ho(second, expected, "derivative of a create_graph vjp")


def _physics_run[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises -> List[List[Float32]]:
    var dx: List[Int] = [4, 1]
    var dw1: List[Int] = [1, 3]
    var dw2: List[Int] = [3, 1]
    var xv: List[Float32] = [-0.9, -0.2, 0.4, 1.1]
    var w1v: List[Float32] = [0.8, -1.3, 0.5]
    var b1v: List[Float32] = [0.1, 0.3, -0.2]
    var w2v: List[Float32] = [0.9, 0.4, -0.7]
    var x = tensor(s, xv, dx, True)
    var hidden = op1(s, _TANH, op2(s, _ADD, op2(s, _MATMUL, x, tensor(s, w1v, dw1)), tensor(s, b1v, dw1)))
    var u = op2(s, _MATMUL, hidden, tensor(s, w2v, dw2))
    s.backward_graph(total(s, u))
    var du = s.grad_handle(x)
    var first = values(s, du, dx)
    s.backward(total(s, du), False)
    var result: List[List[Float32]] = [first^, gradient(s, x, dx)]
    return result^


def physics_second_derivative[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut again: OpsSession[E]) raises:
    var xv: List[Float32] = [-0.9, -0.2, 0.4, 1.1]
    var w1v: List[Float32] = [0.8, -1.3, 0.5]
    var b1v: List[Float32] = [0.1, 0.3, -0.2]
    var w2v: List[Float32] = [0.9, 0.4, -0.7]
    var du = List[Float64](length=4, fill=0)
    var d2u = List[Float64](length=4, fill=0)
    for r in range(4):
        for j in range(3):
            var w1 = Float64(w1v[j])
            var w2 = Float64(w2v[j])
            var t = tanh(Float64(xv[r]) * w1 + Float64(b1v[j]))
            du[r] += (1 - t * t) * w1 * w2
            d2u[r] += -2 * t * (1 - t * t) * w1 * w1 * w2
    var got = _physics_run(s)
    close_ho(got[0], du, "du/dx per row")
    close_ho(got[1], d2u, "d2u/dx2 per row")
    var got2 = _physics_run(again)
    same_words(got[0], got2[0], "du/dx")
    same_words(got[1], got2[1], "d2u/dx2")


def layer_norm_hessian_vector[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var dx: List[Int] = [2, 3]
    var dp: List[Int] = [1, 3]
    var x = tensor(s, load_f32("layer_norm", "x", 6), dx, True)
    var w = tensor(s, load_f32("layer_norm", "w", 3), dp, True)
    var b = tensor(s, load_f32("layer_norm", "b", 3), dp, True)
    var vs: List[Int] = [
        tensor(s, load_f32("layer_norm", "vx", 6), dx),
        tensor(s, load_f32("layer_norm", "vw", 3), dp),
        tensor(s, load_f32("layer_norm", "vb", 3), dp),
    ]
    var hs: List[Int] = [x, w, b]
    var fs: List[Float32] = [0.03]
    var out = s.op(_LAYER_NORM, hs, List[Int](), fs)
    var l = total(s, op2(s, _MUL, out, out))
    var hv = s.hvp(l, hs, vs)
    close_ho(values(s, hv[0], dx), load_f64("layer_norm", "hx", 6), "layer_norm Hessian x")
    close_ho(values(s, hv[1], dp), load_f64("layer_norm", "hw", 3), "layer_norm Hessian w")
    close_ho(values(s, hv[2], dp), load_f64("layer_norm", "hb", 3), "layer_norm Hessian b")


# ------------------------------------------------------------------ test_higher_rank_native

def _rank_shape(rank4: Bool) -> List[Int]:
    var d: List[Int] = [2, 3, 2]
    if rank4:
        d = [2, 1, 3, 2]
    return d^


def _value() -> List[Float32]:
    var out = List[Float32](length=12, fill=0)
    for i in range(12):
        out[i] = Float32(i) / 16
    return out^


def _direction() -> List[Float32]:
    var out = List[Float32](length=12, fill=0)
    for i in range(12):
        out[i] = Float32(i + 1) / 8
    return out^


def _exp_times(value: List[Float32], direction: List[Float32], use_direction: Bool) -> List[Float64]:
    var out = List[Float64](length=len(value), fill=0)
    for i in range(len(value)):
        out[i] = exp(Float64(value[i]))
        if use_direction:
            out[i] *= Float64(direction[i])
    return out^


def recorded_gradient_alias[E: Exec & Movable & Deinitable](mut s: OpsSession[E], rank4: Bool) raises:
    var shape = _rank_shape(rank4)
    var value = _value()
    var direction = _direction()
    var x = tensor(s, value, shape, True)
    var alias_dims: List[Int] = [6, 2]
    var alias = reshape(s, x, alias_dims)
    var l = total(s, op1(s, _EXP, alias))
    s.backward_graph(l)
    var gx = s.grad_handle(x)
    var ga = s.grad_handle(alias)
    var plain = _exp_times(value, direction, False)
    close_hr(values(s, gx, shape), plain, "gradient of x")
    close_hr(values(s, ga, alias_dims), plain, "gradient of the alias")
    require(s.shape_nd(x).same(ShapeN.of(shape)), "x keeps its full shape")
    require(s.shape_nd(alias).same(ShapeN.of(alias_dims)), "the alias keeps (6, 2)")
    var v = tensor(s, direction, shape)
    s.backward(total(s, op2(s, _MUL, gx, v)), False)
    close_hr(gradient(s, x, shape), _exp_times(value, direction, True), "second derivative along v")


def hvp_full_rank[E: Exec & Movable & Deinitable](mut s: OpsSession[E], rank4: Bool, curved: Bool) raises:
    var shape = _rank_shape(rank4)
    var value = _value()
    var direction = _direction()
    var x = tensor(s, value, shape, True)
    var y = op1(s, _EXP, x) if curved else scale(s, x, 2)
    var l = total(s, y)
    var hs: List[Int] = [x]
    var vs: List[Int] = [tensor(s, direction, shape)]
    var hv = values(s, s.hvp(l, hs, vs)[0], shape)
    var expected = _exp_times(value, direction, True) if curved else List[Float64](length=12, fill=0)
    close_hr(hv, expected, "H v, curved=" + String(curved))


def hvp_disconnected[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var shape = _rank_shape(False)
    var other = _rank_shape(True)
    var value = _value()
    var direction = _direction()
    var ones = List[Float32](length=12, fill=1.0)
    var x = tensor(s, value, shape, True)
    var unused = tensor(s, ones, other, True)
    var l = total(s, op1(s, _EXP, x))
    var hs: List[Int] = [x, unused]
    var vs: List[Int] = [tensor(s, direction, shape), tensor(s, ones, other)]
    var hv = s.hvp(l, hs, vs)
    close_hr(values(s, hv[0], shape), _exp_times(value, direction, True), "H v of x")
    close_hr(values(s, hv[1], other), List[Float64](length=12, fill=0), "H v of the disconnected parameter")


def jvp_output_rank[E: Exec & Movable & Deinitable](mut s: OpsSession[E], curved: Bool) raises:
    var shape = _rank_shape(False)
    var out_dims = _rank_shape(True)
    var value = _value()
    var direction = _direction()
    var x = tensor(s, value, shape, True)
    var y = op1(s, _EXP, x) if curved else scale(s, x, 2)
    var output = reshape(s, y, out_dims)
    var hs: List[Int] = [x]
    var us: List[Int] = [tensor(s, direction, shape)]
    var ju = values(s, s.jvp(output, hs, us), out_dims)
    var expected = _exp_times(value, direction, True)
    if not curved:
        for i in range(12):
            expected[i] = 2 * Float64(direction[i])
    close_hr(ju, expected, "J u, curved=" + String(curved))


# ------------------------------------------------------------------ dispatch

def run_ops_higher_fixture[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], mut again: OpsSession[E], fixture_id: Int
) raises:
    """`s` and `again` are two fresh recording sessions on one executor."""
    if fixture_id == 0:
        example_hvp(s, again)
    elif fixture_id == 1:
        hvp_softmax_cross_entropy(s, again)
    elif fixture_id == 2:
        hvp_zero_curvature(s)
    elif fixture_id == 3:
        jvp_vjp_adjoint(s, again)
    elif fixture_id == 4:
        vjp_create_graph(s)
    elif fixture_id == 5:
        physics_second_derivative(s, again)
    elif fixture_id == 6:
        layer_norm_hessian_vector(s)
    elif fixture_id == 7 or fixture_id == 8:
        recorded_gradient_alias(s, fixture_id == 8)
    elif fixture_id < 13:
        var k = fixture_id - 9
        hvp_full_rank(s, k >= 2, k % 2 == 1)
    elif fixture_id == 13:
        hvp_disconnected(s)
    elif fixture_id < OPS_HIGHER_FIXTURES:
        jvp_output_rank(s, fixture_id == 15)
    else:
        raise Error("unknown ops_higher fixture")
