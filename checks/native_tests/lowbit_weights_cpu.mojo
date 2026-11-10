# SPDX-License-Identifier: Apache-2.0
"""Independent weight fixtures and oracle rows use the existing host pool."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, u16_ptr, i8_ptr, i32_ptr
from decomposition.linalg_host_impl import to_bf16_host_run, from_bf16_host_run, quantize_int8_host_run, dequantize_int8_host_run
from checks.native_tests.lowbit_weights_cases import LOWBIT_WEIGHT_FIXTURES, shape, values, check_bf16
from checks.native_tests.int8_linalg_cases import require, expected_quantized, check_quantized, check_dequantized


def check_fixture(fixture_id: Int) raises:
    var dims = shape(fixture_id)
    var rows = dims[0]
    var cols = dims[1]
    var count = rows * cols
    var source = values(fixture_id)
    var original = source.copy()
    var bits = List[UInt16](length=count, fill=0)
    var restored = List[Float32](length=count, fill=0)
    to_bf16_host_run(u16_ptr(Int(bits.unsafe_ptr())), Int(source.unsafe_ptr()), count)
    from_bf16_host_run(f32_ptr(Int(restored.unsafe_ptr())), Int(bits.unsafe_ptr()), count)
    check_bf16(original, bits, restored, rows, cols)
    var codes = List[Int8](length=count, fill=0)
    var exponents = List[Int32](length=rows, fill=0)
    var expected = expected_quantized(original, rows, cols)
    require(quantize_int8_host_run(i8_ptr(Int(codes.unsafe_ptr())), i32_ptr(Int(exponents.unsafe_ptr())), Int(source.unsafe_ptr()), rows, cols) == count, "weight quantize count")
    check_quantized(codes, exponents, expected[0], expected[1], rows, cols)
    require(dequantize_int8_host_run(f32_ptr(Int(restored.unsafe_ptr())), Int(codes.unsafe_ptr()), Int(exponents.unsafe_ptr()), rows, cols) == count, "weight dequantize count")
    check_dequantized(restored, codes, exponents, rows, cols)
    var again = List[Int8](length=count, fill=0)
    var again_exp = List[Int32](length=rows, fill=0)
    _ = quantize_int8_host_run(i8_ptr(Int(again.unsafe_ptr())), i32_ptr(Int(again_exp.unsafe_ptr())), Int(restored.unsafe_ptr()), rows, cols)
    check_quantized(again, again_exp, codes, exponents, rows, cols)
    for i in range(count):
        require(bitcast[DType.uint32](source[i]) == bitcast[DType.uint32](original[i]), "weight input mutated")
    # Integer addresses carry no ownership. Keep every borrowed allocation
    # alive through its last native call, including re-quantization's input.
    _ = source
    _ = original
    _ = bits
    _ = restored
    _ = codes
    _ = exponents
    _ = again
    _ = again_exp


def main() raises:
    var failures = List[Int](length=LOWBIT_WEIGHT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            check_fixture(fixture_id)
        except error:
            print("FAIL lowbit_weights_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, LOWBIT_WEIGHT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "lowbit_weights_cpu failed " + String(failed) + " fixtures")
    print("PASS lowbit_weights_cpu:", LOWBIT_WEIGHT_FIXTURES, "native BF16/INT8 weight fixtures, parallel")
