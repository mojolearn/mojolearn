# SPDX-License-Identifier: Apache-2.0
"""The transformer primitives of the native ops session (autograd/ops_api.mojo
OPS_RMS_NORM, OPS_ROTARY, OPS_TRANSFORMER_LINEAR, OPS_TRANSFORMER_SOFTMAX,
autograd/transformer_primitives.mojo), moved from
test_ops_transformer_program.py (2026-11-14).

No Python participates. The session calls are the ones the public
`mojolearn.ops.Session` makes (providers/python/autograd_ops_frontend_pyapi_door.mojo):
the same op codes, ints and floats (rms_norm 77 [x, w] [] [eps]; rotary 78
[x] [head_dim, offset] [theta]; _transformer_linear 79 [x, w];
_transformer_softmax 80 [x]; permute 34 [x] [0, axes...]; max 37 [x]
[keepdims, axes...]; sub 3; mul 4; exp 9; softmax 17; sum 35 [x] [-1]).

Fixtures (inputs, float64 oracles and tolerances are the former tests'):
  0  rms_norm of a permuted (0, 2, 1) view and both derivatives, against the
     float64 closed forms; rtol 3e-6 / 4e-6, atol 1e-7. The weight is a
     1-D array, which Session.tensor imports as a (4, 1) column, and its
     gradient keeps that shape.
  1  rotary at absolute offset 7, two heads of width 4, theta 4000, value and
     gradient against the float64 rotation; rtol 2e-6, atol 2e-6.
  2  the projection's first-order vjp keeps the full rank of x and the
     parameter's shape; float64 seed @ w and seed^T @ x; rtol 2e-6, atol 2e-7.
  3..5  _transformer_softmax at 5, 33 and 65 columns: value and vjp against
     float64 (rtol 2e-6 / 3e-6, atol 2e-7); its words equal exp / the serial
     float32 sum of the session's own exp(x - max) terms; the ordinary softmax's
     words equal exp / the 32-slot tree sum; and the two differ somewhere (the
     fixture distinguishes the contracts). The inputs are the former test's
     NumPy PCG64 draws, stored by fixtures/transformer_ops_program/generate.py.
The create_graph refusal, the captured-program run/save/load of the softmax,
the permutation-composition pass report and every argument refusal stay in
the Python test.
"""
from std.math import cos, exp, log, max, sin, sqrt
from std.memory import bitcast, unsafe_memcpy
from autograd.ops_api import OpsSession
from checks.native_tests.ops_higher import gradient, op1, op2, tensor, total, values
from sequence.exec_trait import Exec
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime TRANSFORMER_OPS_PROGRAM_FIXTURES = 6
comptime _DIR = "checks/native_tests/fixtures/transformer_ops_program/"

comptime _SUB = 3
comptime _MUL = 4
comptime _EXP = 9
comptime _SOFTMAX = 17
comptime _LAYOUT = 34
comptime _MAX_N = 37
comptime _RMS_NORM = 77
comptime _ROTARY = 78
comptime _TRANSFORMER_LINEAR = 79
comptime _TRANSFORMER_SOFTMAX = 80


def _load_f32(name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](length=count, fill=0)
    with open(_DIR + name + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 4, "transformer_ops_program fixture extent changed: " + name)
        unsafe_memcpy(dest=result.unsafe_ptr().bitcast[UInt8](), src=bytes.unsafe_ptr(), count=count * 4)
        _ = bytes^
    return result^


def _linspace(start: Float64, stop: Float64, n: Int) -> List[Float32]:
    """np.linspace(start, stop, n, dtype=np.float32): float64 steps, the last
    point exactly `stop`, then rounded to float32."""
    var out = List[Float32](length=n, fill=0)
    var step = (stop - start) / Float64(n - 1)
    for i in range(n):
        out[i] = Float32(Float64(i) * step + start)
    out[n - 1] = Float32(stop)
    return out^


def _allclose(actual: List[Float32], expected: List[Float64], rtol: Float64, atol: Float64, what: String) raises:
    """numpy.testing.assert_allclose: |a - e| <= atol + rtol |e|."""
    require(len(actual) == len(expected), what + ": extent")
    for i in range(len(actual)):
        var a = Float64(actual[i])
        require(abs(a - expected[i]) <= atol + rtol * abs(expected[i]),
                what + ": at " + String(i) + " got " + String(a) + ", expected " + String(expected[i]))


def _same_words(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": extent")
    for i in range(len(a)):
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]),
                what + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


# ------------------------------------------------------------------ 0: rms_norm

def rms_norm_view_and_derivatives[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var x = List[Float32](length=24, fill=0)
    for i in range(24):
        x[i] = Float32(i) / Float32(11) - Float32(1)
    var w: List[Float32] = [0.7, -1.1, 0.4, 1.3]
    var seed = _linspace(-0.4, 0.7, 24)
    var eps = 0.03
    var dx3: List[Int] = [2, 4, 3]
    var dw: List[Int] = [4, 1]
    var dy: List[Int] = [2, 3, 4]
    var hx = tensor(s, x, dx3, True)
    var hw = tensor(s, w, dw, True)
    var view = s.op(_LAYOUT, [hx], [0, 0, 2, 1], List[Float32]())
    var fs: List[Float32] = [Float32(eps)]
    var result = s.op(_RMS_NORM, [view, hw], List[Int](), fs)
    s.backward(total(s, op2(s, _MUL, result, tensor(s, seed, dy))), False)
    var got = values(s, result, dy)
    var gx = gradient(s, hx, dx3)
    var gw = gradient(s, hw, dw)
    # float64 reference over z = x.transpose(0, 2, 1), shape (2, 3, 4)
    var expected = List[Float64](length=24, fill=0)
    var dx = List[Float64](length=24, fill=0)  # in x's (2, 4, 3) layout
    var dwe = List[Float64](length=4, fill=0)
    for b in range(2):
        for i in range(3):
            var z = List[Float64](length=4, fill=0)
            var sq = 0.0
            for j in range(4):
                z[j] = Float64(x[(b * 4 + j) * 3 + i])
                sq += z[j] * z[j]
            var inv = 1.0 / sqrt(sq / 4.0 + eps)
            var wz = 0.0
            for j in range(4):
                var weighted = Float64(seed[(b * 3 + i) * 4 + j]) * Float64(w[j])
                wz += weighted * z[j]
            wz /= 4.0
            for j in range(4):
                var at = (b * 3 + i) * 4 + j
                var weighted = Float64(seed[at]) * Float64(w[j])
                expected[at] = z[j] * inv * Float64(w[j])
                dx[(b * 4 + j) * 3 + i] = weighted * inv - z[j] * inv * inv * inv * wz
                dwe[j] += Float64(seed[at]) * z[j] * inv
    _allclose(got, expected, 3e-6, 1e-7, "rms_norm value")
    _allclose(gx, dx, 4e-6, 1e-7, "rms_norm d_x")
    _allclose(gw, dwe, 4e-6, 1e-7, "rms_norm d_weight (a (4, 1) column)")


# ------------------------------------------------------------------ 1: rotary

def _rotate(v: List[Float32], at: Int, j: Int, hd: Int) -> Float64:
    """The rotate-half partner of feature j (head-local) of the 8-wide row at `at`."""
    var head = (j // hd) * hd
    var k = j - head
    var half = hd // 2
    if k < half:
        return -Float64(v[at + head + k + half])
    return Float64(v[at + head + k - half])


def rotary_offset_gradient_heads[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var x = _linspace(-1.0, 1.0, 48)
    var seed = List[Float32](length=48, fill=0)
    for i in range(48):
        seed[i] = Float32(cos(Float64(i)))
    var offset = 7
    var hd = 4
    var theta = 4000.0
    var dims: List[Int] = [2, 3, 8]
    var h = tensor(s, x, dims, True)
    var fs: List[Float32] = [Float32(theta)]
    var y = s.op(_ROTARY, [h], [hd, offset], fs)
    s.backward(total(s, op2(s, _MUL, y, tensor(s, seed, dims))), False)
    var got = values(s, y, dims)
    var gh = gradient(s, h, dims)
    var expected = List[Float64](length=48, fill=0)
    var expected_grad = List[Float64](length=48, fill=0)
    for b in range(2):
        for t in range(3):
            var at = (b * 3 + t) * 8
            for j in range(8):
                var k = (j % hd) % (hd // 2)
                var angle = Float64(offset + t) / exp(Float64(k * 2) / Float64(hd) * log(theta))
                var c = cos(angle)
                var si = sin(angle)
                expected[at + j] = Float64(x[at + j]) * c + _rotate(x, at, j, hd) * si
                expected_grad[at + j] = Float64(seed[at + j]) * c - _rotate(seed, at, j, hd) * si
    _allclose(got, expected, 2e-6, 2e-6, "rotary value")
    _allclose(gh, expected_grad, 2e-6, 2e-6, "rotary gradient")


# ------------------------------------------------------------------ 2: projection vjp

def projection_first_vjp[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var x = List[Float32](length=24, fill=0)
    for i in range(24):
        x[i] = Float32(i) / Float32(16)
    var w = List[Float32](length=20, fill=0)
    for i in range(20):
        w[i] = Float32(i) / Float32(32) - Float32(0.2)
    var seed = _linspace(-0.3, 0.7, 30)
    var dx: List[Int] = [2, 3, 4]
    var dw: List[Int] = [5, 4]
    var dy: List[Int] = [2, 3, 5]
    var hx = tensor(s, x, dx, True)
    var hw = tensor(s, w, dw, True)
    var y = s.op(_TRANSFORMER_LINEAR, [hx, hw], List[Int](), List[Float32]())
    var grads = s.vjp(y, [hx, hw], tensor(s, seed, dy), False)
    require(len(grads) == 2, "vjp returned " + String(len(grads)) + " handles for two inputs")
    var gx = values(s, grads[0], dx)  # values() checks the full shape: x's (2, 3, 4)
    var gw = values(s, grads[1], dw)  # and the parameter's (5, 4)
    var ex = List[Float64](length=24, fill=0)
    var ew = List[Float64](length=20, fill=0)
    for r in range(6):
        for i in range(4):
            var acc = 0.0
            for o in range(5):
                acc += Float64(seed[r * 5 + o]) * Float64(w[o * 4 + i])
            ex[r * 4 + i] = acc
    for o in range(5):
        for i in range(4):
            var acc = 0.0
            for r in range(6):
                acc += Float64(seed[r * 5 + o]) * Float64(x[r * 4 + i])
            ew[o * 4 + i] = acc
    _allclose(gx, ex, 2e-6, 2e-7, "projection d_x")
    _allclose(gw, ew, 2e-6, 2e-7, "projection d_weight")


# ------------------------------------------------------------------ 3..5: transformer softmax

def transformer_softmax_contracts[E: Exec & Movable & Deinitable](mut s: OpsSession[E], columns: Int) raises:
    var n = 6 * columns
    var name = "softmax" + String(columns)
    var x = _load_f32(name + "-x", n)
    var seed = _load_f32(name + "-seed", n)
    var dims: List[Int] = [2, 3, columns]
    var h = tensor(s, x, dims, True)
    var y = s.op(_TRANSFORMER_SOFTMAX, [h], List[Int](), List[Float32]())
    var got = values(s, y, dims)
    var grads = s.vjp(y, [h], tensor(s, seed, dims), False)
    require(len(grads) == 1, "vjp returned " + String(len(grads)) + " handles for one input")
    var gh = values(s, grads[0], dims)
    var reference = List[Float64](length=n, fill=0)
    var derivative = List[Float64](length=n, fill=0)
    for r in range(6):
        var at = r * columns
        var top = Float64(x[at])
        for k in range(columns):
            top = max(top, Float64(x[at + k]))
        var total64 = 0.0
        for k in range(columns):
            reference[at + k] = exp(Float64(x[at + k]) - top)
            total64 += reference[at + k]
        var dot = 0.0
        for k in range(columns):
            reference[at + k] /= total64
            dot += reference[at + k] * Float64(seed[at + k])
        for k in range(columns):
            derivative[at + k] = reference[at + k] * (Float64(seed[at + k]) - dot)
    _allclose(got, reference, 2e-6, 2e-7, name + " value")
    _allclose(gh, derivative, 3e-6, 2e-7, name + " vjp")
    # A separate native exp operation establishes the rounded terms; the
    # check computes each documented float32 sum independently.
    var top_h = s.op(_MAX_N, [h], [1, -1], List[Float32]())
    var e = values(s, op1(s, _EXP, op2(s, _SUB, h, top_h)), dims)
    var expected = List[Float32](length=n, fill=0)
    var by_slots = List[Float32](length=n, fill=0)
    for r in range(6):
        var at = r * columns
        var serial = Float32(0)
        var slots = List[Float32](length=32, fill=0)
        for k in range(columns):
            serial = serial + e[at + k]
            slots[k % 32] = slots[k % 32] + e[at + k]
        var width = 16
        while width >= 1:
            for q in range(width):
                slots[q] = slots[q] + slots[q + width]
            width //= 2
        for k in range(columns):
            expected[at + k] = e[at + k] / serial
            by_slots[at + k] = e[at + k] / slots[0]
    var ordinary = values(s, s.op(_SOFTMAX, [h], List[Int](), List[Float32]()), dims)
    _same_words(got, expected, name + ": transformer softmax against exp / the serial sum")
    _same_words(ordinary, by_slots, name + ": ordinary softmax against exp / the 32-slot tree sum")
    var differs = False
    for i in range(n):
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](ordinary[i]):
            differs = True
    require(differs, name + ": the fixture must distinguish the two contracts")


def run_transformer_ops_program_fixture[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises:
    if fixture_id == 0:
        rms_norm_view_and_derivatives(s)
    elif fixture_id == 1:
        rotary_offset_gradient_heads(s)
    elif fixture_id == 2:
        projection_first_vjp(s)
    elif fixture_id == 3:
        transformer_softmax_contracts(s, 5)
    elif fixture_id == 4:
        transformer_softmax_contracts(s, 33)
    elif fixture_id == 5:
        transformer_softmax_contracts(s, 65)
    else:
        raise Error("transformer_ops_program: no fixture " + String(fixture_id))


# ---- The host column (was transformer_ops_program_cpu_suite.mojo) ----
# The transformer-primitive session fixtures, each on its own recording host
# session, run across host cores.
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(TRANSFORMER_OPS_PROGRAM_FIXTURES)
    def task(fixture_id: Int) {imm cases}:
        try:
            var s = OpsSession[HostExec](HostExec(), True)
            try:
                run_transformer_ops_program_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, TRANSFORMER_OPS_PROGRAM_FIXTURES)


# ---- The device column (was transformer_ops_program_gpu_suite.mojo) ----
# The transformer-primitive session fixtures on the device, in order.
#
# DeviceExec shares sequence_ctx() and its pool, so independent concurrent
# sessions would race that context's ownership (as in ops_higher_gpu); the
# fixtures run one after another and each kernel keeps its own parallelism.
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(TRANSFORMER_OPS_PROGRAM_FIXTURES)
    for fixture_id in range(TRANSFORMER_OPS_PROGRAM_FIXTURES):
        try:
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            try:
                run_transformer_ops_program_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
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
    return Check("transformer_ops_program", cpu=True, gpu=True, serial="", run=run_cases)
