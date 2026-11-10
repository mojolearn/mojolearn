# SPDX-License-Identifier: Apache-2.0
"""Broadcast differentiation and axis softmax with independent scalar oracles.

Same shapes/axis forms as test_ops_rankn; deterministic signed rational
fixtures replace NumPy RNG preparation. Float64 math computes expectations,
with explicit float32 roundings for the exact broadcast forward assertion.
"""
from std.math import exp, log
from autograd.ops_api import OpsSession, OPS_ADD, OPS_MUL, OPS_DIV, OPS_SUM_N, OPS_SOFTMAX_N, OPS_LOG_SOFTMAX_N
from autograd.rankn import ShapeN
from core.cast_in import CastIn, CAST_KIND_FLOAT
from core.hostptr import f32_ptr
from sequence.exec_trait import Exec

comptime RANKN_MATH_FIXTURES = 4


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def _upload[E: Exec & Movable & Deinitable](mut s: OpsSession[E], values: List[Float32], dims: List[Int], grad: Bool = False) raises -> Int:
    var cols = dims[len(dims) - 1]
    var source = CastIn(Int(values.unsafe_ptr()), CAST_KIND_FLOAT, 4, len(values) // cols, cols, cols * 4, 4)
    var h = s.upload_nd(source, ShapeN.of(dims), grad)
    _ = values
    return h


def _binary[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, b: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    return s.run_op(code, hs, List[Int](), List[Float32]())


def _gradient[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, count: Int) raises -> List[Float32]:
    var values = List[Float32](length=count, fill=0)
    s.gradient_into(h, f32_ptr(Int(values.unsafe_ptr())))
    return values^


def _close(actual: List[Float32], expected: List[Float64], rtol: Float64, atol: Float64, label: String) raises:
    require(len(actual) == len(expected), label + " element count")
    for i in range(len(expected)):
        require(abs(Float64(actual[i]) - expected[i]) <= atol + rtol * abs(expected[i]), label + " element " + String(i))


def _broadcast[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var av = List[Float32](length=24, fill=0)
    for i in range(24):
        av[i] = Float32((i * 7) % 19 - 9) * 0.25
    var bv: List[Float32] = [-0.375, 0.625, -0.875]
    var cv: List[Float32] = [-0.5, 0.75, 1.25, -1.5]
    var dv: List[Float32] = [0.75, 1.25, 1.5, 0.5, 1.5, 0.75, 0.5, 1.25]
    var adims: List[Int] = [2, 3, 4]
    var bdims: List[Int] = [1, 3, 1]
    var cdims: List[Int] = [1, 4]
    var ddims: List[Int] = [2, 1, 4]
    var a = _upload(s, av, adims, True)
    var b = _upload(s, bv, bdims, True)
    var c = _upload(s, cv, cdims, True)
    var d = _upload(s, dv, ddims, True)
    var added = _binary(s, OPS_ADD, a, b)
    var multiplied = _binary(s, OPS_MUL, added, c)
    var y = _binary(s, OPS_DIV, multiplied, d)
    require(s.shape_nd(y).same(ShapeN.of(adims)), "broadcast output shape")
    var hs: List[Int] = [y]
    var ints: List[Int] = [-1]
    var loss = s.run_op(OPS_SUM_N, hs, ints, List[Float32]())
    s.backward(loss, False)
    var actual = List[Float32](length=24, fill=0)
    s.download(y, f32_ptr(Int(actual.unsafe_ptr())))
    var ga = _gradient(s, a, 24)
    var gb = _gradient(s, b, 3)
    var gc = _gradient(s, c, 4)
    var gd = _gradient(s, d, 8)
    var expected_ga = List[Float64](length=24, fill=0)
    var expected_gb = List[Float64](length=3, fill=0)
    var expected_gc = List[Float64](length=4, fill=0)
    var expected_gd = List[Float64](length=8, fill=0)
    for i in range(24):
        var row = i // 12
        var middle = (i // 4) % 3
        var channel = i % 4
        var denominator = Float64(dv[row * 4 + channel])
        var numerator = Float64(av[i]) + Float64(bv[middle])
        var coefficient = Float64(cv[channel])
        # Independent scalar operations, each forward stage rounded to fp32
        # as in the former NumPy array operations. Equality matches array_equal
        # (including its equality of positive and negative zero).
        var add32 = Float32(numerator)
        var mul32 = Float32(Float64(add32) * coefficient)
        var expected_y = Float32(Float64(mul32) / denominator)
        require(actual[i] == expected_y, "broadcast forward element " + String(i))
        expected_ga[i] = coefficient / denominator
        expected_gb[middle] += coefficient / denominator
        expected_gc[channel] += numerator / denominator
        expected_gd[row * 4 + channel] -= numerator * coefficient / (denominator * denominator)
    _close(ga, expected_ga, 1e-6, 1e-8, "broadcast gradient A")
    _close(gb, expected_gb, 1e-5, 1e-5, "broadcast gradient B")
    _close(gc, expected_gc, 1e-5, 1e-5, "broadcast gradient C")
    _close(gd, expected_gd, 1e-5, 1e-5, "broadcast gradient D")


def _axis_index(i: Int, axis: Int, coordinate: Int) -> Int:
    var row = i // 30
    var middle = (i // 5) % 6
    var inner = i % 5
    if axis == 0:
        return coordinate * 30 + middle * 5 + inner
    if axis == 1:
        return row * 30 + coordinate * 5 + inner
    return row * 30 + middle * 5 + coordinate


def _softmax[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises:
    var values = List[Float32](length=120, fill=0)
    for i in range(120):
        values[i] = Float32((i * 37) % 127 - 63) / 16.0
    var dims: List[Int] = [4, 6, 5]
    var hs: List[Int] = [_upload(s, values, dims)]
    var axis = 1
    var extent = 6
    var code = OPS_SOFTMAX_N
    if fixture_id == 2:
        axis = 0
        extent = 4
        code = OPS_LOG_SOFTMAX_N
    elif fixture_id == 3:
        axis = 2
        extent = 5
    var ints: List[Int] = [-1 if fixture_id == 3 else axis]
    var result = s.run_op(code, hs, ints, List[Float32]())
    require(s.shape_nd(result).same(ShapeN.of(dims)), "axis softmax output shape")
    var actual = List[Float32](length=120, fill=0)
    s.download(result, f32_ptr(Int(actual.unsafe_ptr())))
    var expected = List[Float64](length=120, fill=0)
    for i in range(120):
        var largest = Float64(-1e300)
        for j in range(extent):
            largest = max(largest, Float64(values[_axis_index(i, axis, j)]))
        var denominator = Float64(0)
        for j in range(extent):
            denominator += exp(Float64(values[_axis_index(i, axis, j)]) - largest)
        var probability = exp(Float64(values[i]) - largest) / denominator
        expected[i] = log(probability) if fixture_id == 2 else probability
    _close(actual, expected, 1e-5, 1e-5 if fixture_id == 2 else 1e-6, "axis softmax/log-softmax")


def run_rankn_math_fixture[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises:
    require(fixture_id >= 0 and fixture_id < RANKN_MATH_FIXTURES, "unknown rank-N math fixture")
    if fixture_id == 0:
        _broadcast(s)
    else:
        _softmax(s, fixture_id)
