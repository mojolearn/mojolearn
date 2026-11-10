# SPDX-License-Identifier: Apache-2.0
"""Direct GPU quantizer/GEMM/dequantizer with independent native oracles.

The existing upload/download helpers own pinned buffers and synchronize;
one GPU context submits shapes in order. Kernels, native fixture preparation
and oracle rows/cells run in parallel. No hashes or cross-vendor run here.
"""
from max.gpu.host import DeviceContext
from checks.native_tests.int8_linalg_cases import (
    INT8_LINALG_FIXTURES, require, fixture_shape, make_values,
    expected_quantized, check_quantized, check_product, check_dequantized,
)
from gemm.checks.gemm_lowbit import quantize_rows_int8_device, dequantize_rows_int8_device, identical_gemm_int8_into
from gemm.checks.gemm_lowbit_check import _upload_f32, _download_f32, _download_i8, _download_i32, _poisoned


def check_fixture(ctx: DeviceContext, fixture_id: Int) raises:
    var shape = fixture_shape(fixture_id)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    var a = make_values(m, k, 30 + k)
    var b = make_values(n, k, 40 + k)
    var expected_a = expected_quantized(a, m, k)
    var expected_b = expected_quantized(b, n, k)
    var da = _upload_f32(ctx, a)
    var db = _upload_f32(ctx, b)
    var dqa = ctx.enqueue_create_buffer[DType.int8](m * k)
    var dea = ctx.enqueue_create_buffer[DType.int32](m)
    var dqb = ctx.enqueue_create_buffer[DType.int8](n * k)
    var deb = ctx.enqueue_create_buffer[DType.int32](n)
    var dc = _poisoned(ctx, m * n)
    var dy = _poisoned(ctx, n * k)
    quantize_rows_int8_device(ctx, dqa, dea, da, m, k)
    quantize_rows_int8_device(ctx, dqb, deb, db, n, k)
    identical_gemm_int8_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
    dequantize_rows_int8_device(ctx, dy, dqb, deb, n, k)
    ctx.synchronize()
    var qa = _download_i8(ctx, dqa, m * k)
    var ea = _download_i32(ctx, dea, m)
    var qb = _download_i8(ctx, dqb, n * k)
    var eb = _download_i32(ctx, deb, n)
    check_quantized(qa, ea, expected_a[0], expected_a[1], m, k)
    check_quantized(qb, eb, expected_b[0], expected_b[1], n, k)
    var c = _download_f32(ctx, dc, m * n, "native INT8 product")
    check_product(c, qa, ea, qb, eb, m, n, k)
    var y = _download_f32(ctx, dy, n * k, "native INT8 dequantization")
    check_dequantized(y, qb, eb, n, k)
    _ = da
    _ = db
    _ = dqa
    _ = dea
    _ = dqb
    _ = deb
    _ = dc
    _ = dy


def main() raises:
    var ctx = DeviceContext()
    var failed = 0
    for fixture_id in range(INT8_LINALG_FIXTURES):
        try:
            check_fixture(ctx, fixture_id)
        except error:
            print("FAIL int8_linalg_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "int8_linalg_gpu failed " + String(failed) + " fixtures")
    print("PASS int8_linalg_gpu:", INT8_LINALG_FIXTURES, "native fixtures; parallel kernels and host oracle rows/cells")
