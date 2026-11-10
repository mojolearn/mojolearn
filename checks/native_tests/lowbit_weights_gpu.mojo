# SPDX-License-Identifier: Apache-2.0
"""Direct GPU conversions; parallel kernels and host oracle rows.

One context submits the three fixture graphs in order. This preserves the
existing device test harness and is not a multi-context concurrency claim.
"""
from max.gpu.host import DeviceContext
from checks.native_tests.lowbit_weights_cases import LOWBIT_WEIGHT_FIXTURES, shape, values, check_bf16
from checks.native_tests.int8_linalg_cases import require, expected_quantized, check_quantized, check_dequantized
from gemm.checks.gemm_lowbit import bf16_narrow, bf16_widen, quantize_rows_int8_device, dequantize_rows_int8_device
from gemm.checks.gemm_lowbit_check import _upload_f32, _download_f32, _download_u16, _download_i8, _download_i32, _poisoned


def check_fixture(ctx: DeviceContext, fixture_id: Int) raises:
    var dims = shape(fixture_id)
    var rows = dims[0]
    var cols = dims[1]
    var count = rows * cols
    var source = values(fixture_id)
    var expected = expected_quantized(source, rows, cols)
    var dx = _upload_f32(ctx, source)
    var db = ctx.enqueue_create_buffer[DType.uint16](count)
    var dwide = _poisoned(ctx, count)
    var dq = ctx.enqueue_create_buffer[DType.int8](count)
    var de = ctx.enqueue_create_buffer[DType.int32](rows)
    var dy = _poisoned(ctx, count)
    var dagain = ctx.enqueue_create_buffer[DType.int8](count)
    var deagain = ctx.enqueue_create_buffer[DType.int32](rows)
    bf16_narrow(ctx, db, dx, count)
    bf16_widen(ctx, dwide, db, count)
    quantize_rows_int8_device(ctx, dq, de, dx, rows, cols)
    dequantize_rows_int8_device(ctx, dy, dq, de, rows, cols)
    quantize_rows_int8_device(ctx, dagain, deagain, dy, rows, cols)
    ctx.synchronize()
    var bits = _download_u16(ctx, db, count)
    var widened = _download_f32(ctx, dwide, count, "weight BF16 materialize")
    check_bf16(source, bits, widened, rows, cols)
    var codes = _download_i8(ctx, dq, count)
    var exponents = _download_i32(ctx, de, rows)
    check_quantized(codes, exponents, expected[0], expected[1], rows, cols)
    var restored = _download_f32(ctx, dy, count, "weight INT8 materialize")
    check_dequantized(restored, codes, exponents, rows, cols)
    var again = _download_i8(ctx, dagain, count)
    var again_exp = _download_i32(ctx, deagain, rows)
    check_quantized(again, again_exp, codes, exponents, rows, cols)
    _ = dx
    _ = db
    _ = dwide
    _ = dq
    _ = de
    _ = dy
    _ = dagain
    _ = deagain


def main() raises:
    var ctx = DeviceContext()
    var failed = 0
    for fixture_id in range(LOWBIT_WEIGHT_FIXTURES):
        try:
            check_fixture(ctx, fixture_id)
        except error:
            print("FAIL lowbit_weights_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "lowbit_weights_gpu failed " + String(failed) + " fixtures")
    print("PASS lowbit_weights_gpu:", LOWBIT_WEIGHT_FIXTURES, "native BF16/INT8 weight fixtures; parallel kernels and host oracle rows")
