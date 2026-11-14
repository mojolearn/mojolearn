# SPDX-License-Identifier: Apache-2.0
"""The host column's float16 widening over every half pattern, 64 chunks of
1024 words run across host cores, each with its own buffers."""
from std.memory import bitcast
from checks.native_tests.core_safetensors_f16_cases import (
    F16_CHUNK, F16_UNITS, F16_WORDS, require, check_words,
)
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from decomposition.linalg_host_impl import from_f16_host_run


def check_unit(unit: Int) raises:
    var start = unit * F16_CHUNK
    var source = List[UInt16](length=F16_CHUNK, fill=0)
    var widened = List[Float32](length=F16_CHUNK, fill=0)
    for j in range(F16_CHUNK):
        source[j] = UInt16(start + j)
    from_f16_host_run(f32_ptr(Int(widened.unsafe_ptr())), Int(source.unsafe_ptr()), F16_CHUNK)
    var words = List[UInt32](length=F16_CHUNK, fill=0)
    for j in range(F16_CHUNK):
        words[j] = bitcast[DType.uint32](widened[j])
    check_words(start, F16_CHUNK, source, words, "host from_f16")
    _ = source
    _ = widened


def run_checks() raises:
    var failures = List[Int](length=F16_UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            check_unit(unit)
        except error:
            print("FAIL core_safetensors_f16_cpu unit", unit, String(error))
            fp.unsafe_store(unit, 1)
    host_parallelize(task, F16_UNITS)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "core_safetensors_f16_cpu failed " + String(failed) + " of " + String(F16_UNITS) + " units")
    print("PASS core_safetensors_f16_cpu:", F16_WORDS, "half patterns, parallel")
