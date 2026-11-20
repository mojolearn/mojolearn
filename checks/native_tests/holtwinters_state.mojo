# SPDX-License-Identifier: Apache-2.0
"""Independent packed Holt-Winters state and query expectations.

Dyadic level/trend/season formulas give exact representable predictions, so
expected words are computed in Float64 without a product numerical helper.
Shapes include the former public48x4/period4 case plus41x7/period6; both
additive/multiplicative modes cover inside, crossing, future-only, forecast
and warmup spans. No identity hashes or fit-as-its-own-oracle comparison.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from holtwinters.host.hw_predict import hw_predict_in_sample_ptr, hw_forecast_from_state_ptr
from holtwinters.estimator import holtwinters_predict_ptr, holtwinters_forecast_ptr
from holtwinters.resident_state import (
    holtwinters_resident_prepare,
    holtwinters_resident_release,
    holtwinters_resident_count,
    holtwinters_query_resident,
)
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime HW_STATE_OWNERS = 4
comptime HW_STATE_QUERIES = 5
comptime HW_SENTINEL = Float32(31415.5)


def hw_shape(owner_id: Int) -> Tuple[Int, Int, Int, Bool]:
    return (48, 4, 4, owner_id % 2 == 0) if owner_id < 2 else (41, 7, 6, owner_id % 2 == 0)


def hw_span(n: Int, frequency: Int, query_id: Int) -> Tuple[Int, Int]:
    if query_id == 0:
        return (10, n - 8)
    if query_id == 1:
        return (n - 8, n + 8)
    if query_id == 2:
        return (n + 2, n + 12)
    if query_id == 3:
        return (n, n + 7)
    return (0, 2 * frequency + 3)


def level_value(time: Int, series: Int) -> Float64:
    return 2.0 + Float64(time) * 0.125 + Float64(series) * 0.25


def trend_value(time: Int, series: Int) -> Float64:
    return Float64((time + series) % 5 - 2) * 0.03125


def season_value(time: Int, series: Int, frequency: Int) -> Float64:
    return 1.0 + Float64(time % frequency) * 0.0625 + Float64(series) * 0.03125


def hw_components(owner_id: Int) -> List[Float32]:
    var shape = hw_shape(owner_id)
    var n = shape[0]
    var batch = shape[1]
    var frequency = shape[2]
    var plane = (n - frequency) * batch
    var result = List[Float32](length=3 * plane, fill=0)
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm rp, imm plane, imm batch, imm frequency}:
        var kind = index // plane
        var time = (index % plane) // batch
        var series = index % batch
        var value = level_value(time, series) if kind == 0 else (trend_value(time, series) if kind == 1 else season_value(time, series, frequency))
        rp.unsafe_store(index, Float32(value))
    host_parallelize(cell, 3 * plane)
    return result^


def hw_expected(owner_id: Int, query_id: Int) -> List[Float32]:
    var shape = hw_shape(owner_id)
    var n = shape[0]
    var batch = shape[1]
    var frequency = shape[2]
    var additive = shape[3]
    var span = hw_span(n, frequency, query_id)
    var start = span[0]
    var count = (span[1] - start) * batch
    var result = List[Float32](length=count, fill=0)
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm rp, imm n, imm batch, imm frequency, imm additive, imm start}:
        var time = start + index // batch
        var series = index % batch
        if time < 2 * frequency:
            rp.unsafe_store(index, bitcast[DType.float32](UInt32(0x7fc00000)))
            return
        var state_time = time - frequency - 1
        var seasonal_time = time - 2 * frequency
        var horizon = 1
        if time >= n:
            state_time = n - frequency - 1
            seasonal_time = n - 2 * frequency + (time - n) % frequency
            horizon = time - n + 1
        var base = level_value(state_time, series) + Float64(horizon) * trend_value(state_time, series)
        var seasonal = season_value(seasonal_time, series, frequency)
        rp.unsafe_store(index, Float32(base + seasonal if additive else base * seasonal))
    host_parallelize(cell, count)
    return result^


def hw_output(owner_id: Int, query_id: Int) -> List[Float32]:
    var shape = hw_shape(owner_id)
    var span = hw_span(shape[0], shape[2], query_id)
    return List[Float32](length=(span[1] - span[0]) * shape[1] + 2, fill=HW_SENTINEL)


def hw_check(actual: List[Float32], owner_id: Int, query_id: Int) raises:
    var expected = hw_expected(owner_id, query_id)
    require(len(actual) == len(expected) + 2, "Holt-Winters output extent")
    require(actual[0] == HW_SENTINEL and actual[len(actual) - 1] == HW_SENTINEL, "Holt-Winters output guard")
    for index in range(len(expected)):
        require(bitcast[DType.uint32](actual[index + 1]) == bitcast[DType.uint32](expected[index]),
            "independent Holt-Winters state owner " + String(owner_id) + " query " + String(query_id) + " cell " + String(index))


# ---- The host column entry (was holtwinters_state_cpu.mojo) ----
# Parallel direct host queries over independently owned native fitted state.
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


def _run_cpu(mut report: Report) raises:
    comptime COUNT = HW_STATE_OWNERS * HW_STATE_QUERIES
    var cases = report.cases(COUNT)
    def task(fixture_id: Int) {imm cases}:
        try:
            fixture(fixture_id)
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, COUNT)


# ---- The device column entry (was holtwinters_state_gpu.mojo) ----
# Direct GPU per-call/resident fitted-state queries and owner lifetime.
#
# Resident-owner submissions and changing queries are ordered for the existing
# GPU context. Forecast kernels, native data preparation and oracle cells run
# in parallel. No concurrent GPU-query claim is made here.
def fixture_gpu(owner_id: Int) raises:
    var shape = hw_shape(owner_id)
    var n = shape[0]
    var batch = shape[1]
    var frequency = shape[2]
    var seasonal = String("add" if shape[3] else "mul")
    var packed = hw_components(owner_id)
    var preserved = packed.copy()
    var before = holtwinters_resident_count()
    var handle = holtwinters_resident_prepare(f32_ptr(Int(packed.unsafe_ptr())), n, batch, frequency)
    try:
        require(holtwinters_resident_count() == before + 1, "resident component count after prepare")
        # A resident model owns an uploaded snapshot, not the caller's span.
        var pp = packed.unsafe_ptr()
        def mutate(index: Int) {imm pp}:
            pp.unsafe_store(index, Float32(-99))
        host_parallelize(mutate, len(packed))
        for query_id in range(HW_STATE_QUERIES):  # successive queries on one retained native owner
            var span = hw_span(n, frequency, query_id)
            var start = span[0]
            var end = span[1]
            var actual = hw_output(owner_id, query_id)
            var pointer_result = hw_output(owner_id, query_id)
            var count = (end - start) * batch
            require(holtwinters_query_resident(handle, f32_ptr(Int(actual.unsafe_ptr()) + 4), seasonal, start, end) == count, "resident query extent")
            hw_check(actual, owner_id, query_id)
            var inside = max(0, min(end, n) - start)
            if inside:
                require(holtwinters_predict_ptr(f32_ptr(Int(preserved.unsafe_ptr())), f32_ptr(Int(pointer_result.unsafe_ptr()) + 4), n, batch, frequency, seasonal, start, min(end, n)) == inside * batch, "per-call inside extent")
            if end > n:
                var future = List[Float32](length=(end - n) * batch, fill=HW_SENTINEL)
                require(holtwinters_forecast_ptr(f32_ptr(Int(preserved.unsafe_ptr())), f32_ptr(Int(future.unsafe_ptr())), n, batch, frequency, seasonal, end - n) == len(future), "per-call forecast extent")
                var src = future.unsafe_ptr()
                var dst = pointer_result.unsafe_ptr()
                var skip = max(0, start - n) * batch
                var offset = inside * batch + 1
                def copy_future(index: Int) {imm src, imm dst, imm skip, imm offset}:
                    dst.unsafe_store(offset + index, src.unsafe_load(skip + index))
                host_parallelize(copy_future, count - inside * batch)
                _ = future
            hw_check(pointer_result, owner_id, query_id)
            for index in range(count + 2):
                require(bitcast[DType.uint32](actual[index]) == bitcast[DType.uint32](pointer_result[index]), "resident/per-call prediction words differ")
            require(holtwinters_resident_count() == before + 1, "new query duplicated resident owner")
    except error:
        holtwinters_resident_release(handle)
        raise error
    holtwinters_resident_release(handle)
    holtwinters_resident_release(handle)
    require(holtwinters_resident_count() == before, "resident owner release count")
    var refused_output = List[Float32](length=batch, fill=HW_SENTINEL)
    var refused = False
    try:
        _ = holtwinters_query_resident(handle, f32_ptr(Int(refused_output.unsafe_ptr())), seasonal, n, n + 1)
    except:
        refused = True
    require(refused, "released resident handle accepted")
    for value in refused_output:
        require(value == HW_SENTINEL, "stale handle wrote output")
    var original = hw_components(owner_id)
    for index in range(len(preserved)):
        require(preserved[index] == original[index], "per-call query mutated source components")
    _ = packed
    _ = preserved
    _ = refused_output


def _run_gpu(mut report: Report) raises:
    var cases = report.cases(HW_STATE_OWNERS)
    for owner_id in range(HW_STATE_OWNERS):
        try:
            fixture_gpu(owner_id)
        except error:
            cases.fail(owner_id, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("holtwinters_state", cpu=True, gpu=True, serial="", run=run_cases)
