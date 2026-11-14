# SPDX-License-Identifier: Apache-2.0
"""The device float16 widening over every half pattern in one launch (the
kernel is one thread per word); the host checks the words across cores."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.native_tests.core_safetensors_f16_cases import (
    F16_CHUNK, F16_UNITS, F16_WORDS, require, check_words,
)
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, u16_ptr
from decomposition.linalg_door_impl import from_f16_device


def run_checks() raises:
    var ctx = DeviceContext()
    var source = List[UInt16](length=F16_WORDS, fill=0)
    var widened = List[Float32](length=F16_WORDS, fill=0)
    for j in range(F16_WORDS):
        source[j] = UInt16(j)
    from_f16_device(ctx, f32_ptr(Int(widened.unsafe_ptr())), u16_ptr(Int(source.unsafe_ptr())), F16_WORDS)
    var failures = List[Int](length=F16_UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    var sp = source.unsafe_ptr()
    var wp = widened.unsafe_ptr()
    def task(unit: Int) {imm fp, imm sp, imm wp}:
        try:
            var start = unit * F16_CHUNK
            var src = List[UInt16](length=F16_CHUNK, fill=0)
            var words = List[UInt32](length=F16_CHUNK, fill=0)
            for j in range(F16_CHUNK):
                src[j] = sp[start + j]
                words[j] = bitcast[DType.uint32](wp[start + j])
            check_words(start, F16_CHUNK, src, words, "device from_f16")
        except error:
            print("FAIL core_safetensors_f16_gpu unit", unit, String(error))
            fp.unsafe_store(unit, 1)
    host_parallelize(task, F16_UNITS)
    _ = source
    _ = widened
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "core_safetensors_f16_gpu failed " + String(failed) + " of " + String(F16_UNITS) + " units")
    print("PASS core_safetensors_f16_gpu:", F16_WORDS, "half patterns in one launch")
