# SPDX-License-Identifier: Apache-2.0
"""The session helpers, stored-fixture readers and comparisons of the ops_higher checks, shared with the core_*,
ops_*, training_* and transformer_* session checks (moved out of ops_higher.mojo when it became a TestSuite file:
shared helpers live in a module with no `main()` and no `test_*`, docs/plans/TESTSUITE_CONVERSION.md).

The session calls are the ones the public `mojolearn.ops.Session` makes
(providers/python/autograd_ops_frontend_pyapi_door.mojo): the same op codes, ints and floats.
"""
from std.math import isfinite
from std.memory import bitcast, unsafe_memcpy
from autograd.ops_api import OpsSession
from autograd.rankn import ShapeN
from core.cast_in import CastIn
from core.hostptr import f32_ptr
from sequence.exec_trait import Exec


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


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


# ------------------------------------------------------------------ fixtures on disk

def load_f32(case_index: String, name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](length=count, fill=0)
    with open(_DIR + case_index + "-" + name + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 4, "ops_higher input fixture extent changed: " + case_index + "-" + name)
        unsafe_memcpy(dest=result.unsafe_ptr().bitcast[UInt8](), src=bytes.unsafe_ptr(), count=count * 4)
        _ = bytes^
    return result^


def load_f64(case_index: String, name: String, count: Int) raises -> List[Float64]:
    var result = List[Float64](length=count, fill=0)
    with open(_DIR + case_index + "-" + name + ".f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 8, "ops_higher oracle fixture extent changed: " + case_index + "-" + name)
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
