# SPDX-License-Identifier: Apache-2.0
"""Independent weight conversion fixtures formerly checked with NumPy.

Keep the original 40x32 and 32x24 shapes/uniform range using a native fixture
generator. The former unused special-row helper now supplies a third executed
fixture. BF16 expectations use quotient/remainder rounding; INT8 expectations
use the independent Float64 ties-even oracle in int8_linalg_cases.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from checks.native_tests.bf16_conversion import expected_narrow, expected_widen
from core.hostptr import f32_ptr, u16_ptr, i8_ptr, i32_ptr
from decomposition.linalg_host_impl import (
    to_bf16_host_run,
    from_bf16_host_run,
    quantize_int8_host_run,
    dequantize_int8_host_run,
)
from checks.native_tests.int8_linalg import expected_quantized, check_quantized, check_dequantized
from max.gpu.host import DeviceContext
from gemm.checks.gemm_lowbit import bf16_narrow, bf16_widen, quantize_rows_int8_device, dequantize_rows_int8_device
from gemm.checks.gemm_lowbit_check import (
    _upload_f32,
    _download_f32,
    _download_u16,
    _download_i8,
    _download_i32,
    _poisoned,
)
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime LOWBIT_WEIGHT_FIXTURES = 3


def shape(fixture_id: Int) -> Tuple[Int, Int]:
    if fixture_id == 0:
        return (40, 32)
    if fixture_id == 1:
        return (32, 24)
    return (7, 4)


def values(fixture_id: Int) -> List[Float32]:
    if fixture_id == 2:
        var special: List[Float32] = [
            0.0, -0.0, 0.0, 0.0,
            1e-40, -1e-40, 1.0, -1.0,
            0.5, 1.5, 2.5, -2.5,
            1e-30, 2e-30, -3e-30, 4e-30,
            3e38, -1e38, 1.0, 2.0,
            0.1, -0.2, 0.3, -0.4,
            65504.0, 0.000030517578125, -0.000030517578125, 100.0,
        ]
        return special^
    var dims = shape(fixture_id)
    var rows = dims[0]
    var cols = dims[1]
    var source = List[Float32](length=rows * cols, fill=0)
    var sp = source.unsafe_ptr()
    def row_task(row: Int) {imm sp, imm cols, imm fixture_id}:
        for col in range(cols):
            var h = UInt64(row * cols + col + 1) * UInt64(0x9E3779B97F4A7C15) + UInt64(1 if fixture_id == 0 else 3)
            h ^= h >> 30
            h *= UInt64(0xBF58476D1CE4E5B9)
            h ^= h >> 27
            sp.unsafe_store(row * cols + col, Float32(Int(h & 0xFFFFFF) - 8388608) * 0.00000001490116119384765625)
    host_parallelize(row_task, rows)
    return source^


def check_bf16(source: List[Float32], bits: List[UInt16], restored: List[Float32], rows: Int, cols: Int) raises:
    var sp = source.unsafe_ptr()
    var bp = bits.unsafe_ptr()
    var rp = restored.unsafe_ptr()
    var failures = List[Int](length=rows, fill=0)
    var fp = failures.unsafe_ptr()
    def row_task(row: Int) {imm sp, imm bp, imm rp, imm fp, imm cols}:
        for col in range(cols):
            var i = row * cols + col
            var expected = expected_narrow(bitcast[DType.uint32](sp.unsafe_load(i)))
            if bp.unsafe_load(i) != expected or bitcast[DType.uint32](rp.unsafe_load(i)) != expected_widen(expected):
                fp.unsafe_store(row, 1)
    host_parallelize(row_task, rows)
    for row in range(rows):
        require(failures[row] == 0, "packed BF16 weight row " + String(row))
    _ = source
    _ = bits
    _ = restored


# ---- The host column entry (was lowbit_weights_cpu.mojo) ----
# Independent weight fixtures and oracle rows use the existing host pool.
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


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(LOWBIT_WEIGHT_FIXTURES)
    def task(fixture_id: Int) {imm cases}:
        try:
            check_fixture(fixture_id)
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, LOWBIT_WEIGHT_FIXTURES)


# ---- The device column entry (was lowbit_weights_gpu.mojo) ----
# Direct GPU conversions; parallel kernels and host oracle rows.
#
# One context submits the three fixture graphs in order. This preserves the
# existing device test harness and is not a multi-context concurrency claim.
def check_fixture_gpu(ctx: DeviceContext, fixture_id: Int) raises:
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


def _run_gpu(mut report: Report) raises:
    var ctx = DeviceContext()
    var cases = report.cases(LOWBIT_WEIGHT_FIXTURES)
    for fixture_id in range(LOWBIT_WEIGHT_FIXTURES):
        try:
            check_fixture_gpu(ctx, fixture_id)
        except error:
            cases.fail(fixture_id, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("lowbit_weights", cpu=True, gpu=True, serial="", run=run_cases)
