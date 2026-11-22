# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
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
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from checks.native_tests.rankn_cases import _upload, _binary, _gradient, _close, require


comptime RANKN_MATH_FIXTURES = 4
comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()


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


# ---- The host column entry (was rankn_math_cpu.mojo) ----
# Four independent native session fixtures run through the host pool.


def _host_fixture(fixture_id: Int) raises:
    var s = OpsSession[HostExec](HostExec(), fixture_id == 0)
    try:
        run_rankn_math_fixture(s, fixture_id)
    except error:
        s.close()
        raise error
    s.close()


def test_fixture() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("fixture", RANKN_MATH_FIXTURES, _host_fixture)
    else:
        # The device column (was rankn_math_gpu.mojo), inline in the comptime branch: TestSuite discovery
        # instantiates every module function and the host column build has no device target.
        # Direct native GPU operations, with ordered shared-context submissions.
        # DeviceExec shares sequence_ctx() and its pool. Respect that ownership and
        # use the existing parallel kernels; this does not prove session concurrency.
        var failed = 0
        var text = String("")
        for fixture_id in range(RANKN_MATH_FIXTURES):  # driver-loop: ordered shared-context submissions
            try:
                var s = OpsSession[DeviceExec](DeviceExec(), fixture_id == 0)
                try:
                    run_rankn_math_fixture(s, fixture_id)
                except error:
                    s.close()
                    raise error
                s.close()
            except error:
                failed += 1
                text += "\n      fixture row " + String(fixture_id) + ": " + String(error)
        if failed:
            raise Error("fixture: " + String(failed) + " of " + String(RANKN_MATH_FIXTURES) + " cases failed" + text)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
