# SPDX-License-Identifier: Apache-2.0
"""Typed host entries; independent fixtures, rows and output cells parallel."""
from checks.native_tests.int8_linalg_cases import (
    INT8_LINALG_FIXTURES, require, fixture_shape, make_values,
    expected_quantized, check_quantized, check_product, check_dequantized,
)
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, i8_ptr, i32_ptr
from decomposition.linalg_host_impl import quantize_int8_host_run, dequantize_int8_host_run, gemm_int8_host_run


def check_fixture(fixture_id: Int) raises:
    var shape = fixture_shape(fixture_id)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    var a = make_values(m, k, 30 + k)
    var b = make_values(n, k, 40 + k)
    var expected_a = expected_quantized(a, m, k)
    var expected_b = expected_quantized(b, n, k)
    var qa = List[Int8](length=m * k, fill=0)
    var ea = List[Int32](length=m, fill=0)
    var qb = List[Int8](length=n * k, fill=0)
    var eb = List[Int32](length=n, fill=0)
    require(quantize_int8_host_run(i8_ptr(Int(qa.unsafe_ptr())), i32_ptr(Int(ea.unsafe_ptr())), Int(a.unsafe_ptr()), m, k) == m * k, "host A quantizer count")
    require(quantize_int8_host_run(i8_ptr(Int(qb.unsafe_ptr())), i32_ptr(Int(eb.unsafe_ptr())), Int(b.unsafe_ptr()), n, k) == n * k, "host B quantizer count")
    check_quantized(qa, ea, expected_a[0], expected_a[1], m, k)
    check_quantized(qb, eb, expected_b[0], expected_b[1], n, k)
    var c = List[Float32](length=m * n, fill=0)
    require(gemm_int8_host_run(f32_ptr(Int(c.unsafe_ptr())), Int(qa.unsafe_ptr()), Int(ea.unsafe_ptr()), Int(qb.unsafe_ptr()), Int(eb.unsafe_ptr()), m, n, k) == m * n, "host INT8 product count")
    check_product(c, qa, ea, qb, eb, m, n, k)
    var y = List[Float32](length=n * k, fill=0)
    require(dequantize_int8_host_run(f32_ptr(Int(y.unsafe_ptr())), Int(qb.unsafe_ptr()), Int(eb.unsafe_ptr()), n, k) == n * k, "host dequantizer count")
    check_dequantized(y, qb, eb, n, k)
    _ = a
    _ = b
    _ = qa
    _ = ea
    _ = qb
    _ = eb
    _ = c
    _ = y


def main() raises:
    var failures = List[Int](length=INT8_LINALG_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            check_fixture(fixture_id)
        except error:
            print("FAIL int8_linalg_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, INT8_LINALG_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "int8_linalg_cpu failed " + String(failed) + " fixtures")
    print("PASS int8_linalg_cpu:", INT8_LINALG_FIXTURES, "native quantize/GEMM/dequantize fixtures; parallel rows and cells")
