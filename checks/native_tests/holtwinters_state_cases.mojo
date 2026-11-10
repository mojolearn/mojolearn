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

comptime HW_STATE_OWNERS = 4
comptime HW_STATE_QUERIES = 5
comptime HW_SENTINEL = Float32(31415.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


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
