# SPDX-License-Identifier: Apache-2.0
"""Parallel direct host queries over independently owned native fitted state."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from holtwinters.host.hw_predict import hw_predict_in_sample_ptr, hw_forecast_from_state_ptr
from checks.native_tests.holtwinters_state_cases import (
    HW_STATE_OWNERS, HW_STATE_QUERIES, require, hw_shape, hw_span,
    hw_components, hw_output, hw_check,
)


def fixture(fixture_id: Int) raises:
    var owner_id = fixture_id // HW_STATE_QUERIES
    var query_id = fixture_id % HW_STATE_QUERIES
    var shape = hw_shape(owner_id)
    var n = shape[0]
    var batch = shape[1]
    var frequency = shape[2]
    var additive = shape[3]
    var span = hw_span(n, frequency, query_id)
    var start = span[0]
    var end = span[1]
    var packed = hw_components(owner_id)
    var before = packed.copy()
    var actual = hw_output(owner_id, query_id)
    var dst = actual.unsafe_ptr()
    var input = f32_ptr(Int(packed.unsafe_ptr()))
    var plane = (n - frequency) * batch
    var inside = max(0, min(end, n) - start)
    if inside:
        var values = hw_predict_in_sample_ptr(input, plane, n, batch, frequency, additive, start, min(end, n))
        var vp = values.unsafe_ptr()
        def copy_inside(index: Int) {imm dst, imm vp}:
            dst.unsafe_store(index + 1, vp.unsafe_load(index))
        host_parallelize(copy_inside, inside * batch)
        _ = values
    if end > n:
        var values = hw_forecast_from_state_ptr(input, plane, n, batch, frequency, additive, end - n)
        var vp = values.unsafe_ptr()
        var skip = max(0, start - n) * batch
        var count = (end - max(start, n)) * batch
        var offset = inside * batch + 1
        def copy_future(index: Int) {imm dst, imm vp, imm skip, imm offset}:
            dst.unsafe_store(offset + index, vp.unsafe_load(skip + index))
        host_parallelize(copy_future, count)
        _ = values
    hw_check(actual, owner_id, query_id)
    for index in range(len(packed)):
        require(bitcast[DType.uint32](packed[index]) == bitcast[DType.uint32](before[index]), "host query mutated fitted state")
    _ = packed


def main() raises:
    comptime COUNT = HW_STATE_OWNERS * HW_STATE_QUERIES
    var failures = List[Int](length=COUNT, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            fixture(fixture_id)
        except error:
            print("FAIL holtwinters_state_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, COUNT)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "holtwinters_state_cpu failed " + String(failed) + " fixtures")
    print("PASS holtwinters_state_cpu:", COUNT, "parallel native fitted-state queries")
