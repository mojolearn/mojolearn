# SPDX-License-Identifier: Apache-2.0
"""Parallel typed host conversion checks with independent integer expectations."""
from std.memory import bitcast
from checks.native_tests.bf16_conversion_cases import (
    BF16_NARROW_COUNT, BF16_WIDEN_COUNT, BF16_CHUNK, BF16_NARROW_UNITS,
    BF16_UNITS, input_word, expected_narrow, expected_widen, require,
)
from core.host_parallel import host_parallelize
from decomposition.linalg_host_impl import to_bf16_host_run, from_bf16_host_run


def check_unit(unit: Int) raises:
    if unit < BF16_NARROW_UNITS:
        var start = unit * BF16_CHUNK
        var count = min(BF16_CHUNK, BF16_NARROW_COUNT - start)
        var source = List[Float32](length=count, fill=0)
        var narrowed = List[UInt16](length=count, fill=0)
        var widened = List[Float32](length=count, fill=0)
        for j in range(count):
            source[j] = bitcast[DType.float32](input_word(start + j))
        to_bf16_host_run(narrowed.unsafe_ptr(), Int(source.unsafe_ptr()), count)
        from_bf16_host_run(widened.unsafe_ptr(), Int(narrowed.unsafe_ptr()), count)
        for j in range(count):
            var expected = expected_narrow(input_word(start + j))
            require(narrowed[j] == expected, "host BF16 narrowing word " + String(start + j))
            require(bitcast[DType.uint32](widened[j]) == expected_widen(expected), "host BF16 roundtrip word " + String(start + j))
        _ = source
        _ = narrowed
        _ = widened
    else:
        var start = (unit - BF16_NARROW_UNITS) * BF16_CHUNK
        var source = List[UInt16](length=BF16_CHUNK, fill=0)
        var widened = List[Float32](length=BF16_CHUNK, fill=0)
        for j in range(BF16_CHUNK):
            source[j] = UInt16(start + j)
        from_bf16_host_run(widened.unsafe_ptr(), Int(source.unsafe_ptr()), BF16_CHUNK)
        for j in range(BF16_CHUNK):
            require(bitcast[DType.uint32](widened[j]) == expected_widen(UInt16(start + j)), "host raw BF16 widening word " + String(start + j))
        _ = source
        _ = widened


def main() raises:
    var failures = List[Int](length=BF16_UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            check_unit(unit)
        except error:
            print("FAIL bf16_conversion_cpu unit", unit, String(error))
            fp.unsafe_store(unit, 1)
    host_parallelize(task, BF16_UNITS)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "bf16_conversion_cpu failed " + String(failed) + " units")
    print("PASS bf16_conversion_cpu:", BF16_NARROW_COUNT, "narrow/roundtrip words and", BF16_WIDEN_COUNT, "raw widening words, parallel")
