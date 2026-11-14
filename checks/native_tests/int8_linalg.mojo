# SPDX-License-Identifier: Apache-2.0
"""Independent native INT8 quantizer, product and dequantizer expectations.

The five shapes and 13-bit-significand/eight-binade input distribution come
from test_linalg_lowbit. A native deterministic generator replaces NumPy's
RNG. Float64 quotient/remainder rounding and Int64 dots form the oracle;
no product rounding, GEMM or quantization helper computes expectations.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, i8_ptr, i32_ptr
from decomposition.linalg_host_impl import quantize_int8_host_run, dequantize_int8_host_run, gemm_int8_host_run
from max.gpu.host import DeviceContext
from gemm.checks.gemm_lowbit import quantize_rows_int8_device, dequantize_rows_int8_device, identical_gemm_int8_into
from gemm.checks.gemm_lowbit_check import _upload_f32, _download_f32, _download_i8, _download_i32, _poisoned
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime INT8_LINALG_FIXTURES = 5


def fixture_shape(fixture_id: Int) raises -> Tuple[Int, Int, Int]:
    if fixture_id == 0:
        return (1, 32, 32)
    if fixture_id == 1:
        return (2, 96, 127)
    if fixture_id == 2:
        return (7, 33, 129)
    if fixture_id == 3:
        return (16, 64, 1000)
    if fixture_id == 4:
        return (1, 1024, 512)
    raise Error("unknown INT8 linalg fixture")


def _power2(exponent: Int) -> Float64:
    # Test fixtures' powers are small normal values. This is an exact IEEE
    # encoding, independent of the product's power-of-two helper.
    return bitcast[DType.float64](UInt64(exponent + 1023) << 52)


def make_values(rows: Int, cols: Int, seed: Int) -> List[Float32]:
    var values = List[Float32](length=rows * cols, fill=0)
    var vp = values.unsafe_ptr()
    def row_task(row: Int) {imm vp, imm cols, imm seed}:
        for col in range(cols):
            var h = UInt64(row * cols + col + 1) * UInt64(0x9E3779B97F4A7C15) + UInt64(seed)
            h ^= h >> 30
            h *= UInt64(0xBF58476D1CE4E5B9)
            h ^= h >> 27
            var bits = (UInt32((h >> 19) & 1) << 31) | ((UInt32((h >> 12) & 7) + 123) << 23) | (UInt32(h & 0xFFF) << 11)
            vp.unsafe_store(row * cols + col, bitcast[DType.float32](bits))
    host_parallelize(row_task, rows)
    return values^


def expected_quantized(values: List[Float32], rows: Int, cols: Int) -> Tuple[List[Int8], List[Int32]]:
    var codes = List[Int8](length=rows * cols, fill=0)
    var exponents = List[Int32](length=rows, fill=0)
    var vp = values.unsafe_ptr()
    var qp = codes.unsafe_ptr()
    var ep = exponents.unsafe_ptr()
    def row_task(row: Int) {imm vp, imm qp, imm ep, imm cols}:
        var largest = UInt32(0)
        for col in range(cols):
            var word = bitcast[DType.uint32](vp.unsafe_load(row * cols + col)) & UInt32(0x7FFFFFFF)
            largest = max(largest, word)
        var exponent = Int(largest >> 23) - 133 if largest != 0 else 0
        ep.unsafe_store(row, Int32(exponent))
        var scale = _power2(-exponent)
        for col in range(cols):
            var x = Float64(vp.unsafe_load(row * cols + col)) * scale
            var magnitude = abs(x)
            var rounded = Int(magnitude)
            var remainder = magnitude - Float64(rounded)
            if remainder > 0.5 or (remainder == 0.5 and rounded % 2 != 0):
                rounded += 1
            rounded = min(127, rounded)
            if x < 0.0:
                rounded = -rounded
            qp.unsafe_store(row * cols + col, Int8(rounded))
    host_parallelize(row_task, rows)
    _ = values
    return (codes^, exponents^)


def check_quantized(codes: List[Int8], exponents: List[Int32], expected_codes: List[Int8], expected_exponents: List[Int32], rows: Int, cols: Int) raises:
    var failures = List[Int](length=rows, fill=0)
    var fp = failures.unsafe_ptr()
    var qp = codes.unsafe_ptr()
    var ep = exponents.unsafe_ptr()
    var eqp = expected_codes.unsafe_ptr()
    var eep = expected_exponents.unsafe_ptr()
    def row_task(row: Int) {imm fp, imm qp, imm ep, imm eqp, imm eep, imm cols}:
        if ep.unsafe_load(row) != eep.unsafe_load(row):
            fp.unsafe_store(row, 1)
        for col in range(cols):
            var i = row * cols + col
            if qp.unsafe_load(i) != eqp.unsafe_load(i) or abs(Int(qp.unsafe_load(i))) > 127:
                fp.unsafe_store(row, 1)
    host_parallelize(row_task, rows)
    for row in range(rows):
        require(failures[row] == 0, "quantizer code/exponent row " + String(row))
    _ = codes
    _ = exponents
    _ = expected_codes
    _ = expected_exponents


def check_product(actual: List[Float32], qa: List[Int8], ea: List[Int32], qb: List[Int8], eb: List[Int32], m: Int, n: Int, k: Int) raises:
    var failures = List[Int](length=m * n, fill=0)
    var fp = failures.unsafe_ptr()
    var cp = actual.unsafe_ptr()
    var ap = qa.unsafe_ptr()
    var aep = ea.unsafe_ptr()
    var bp = qb.unsafe_ptr()
    var bep = eb.unsafe_ptr()
    def cell_task(cell: Int) {imm fp, imm cp, imm ap, imm aep, imm bp, imm bep, imm n, imm k}:
        var row = cell // n
        var col = cell % n
        var accumulator = Int64(0)
        for j in range(k):
            accumulator += Int64(ap.unsafe_load(row * k + j)) * Int64(bp.unsafe_load(col * k + j))
        # Preserve the original independent oracle's Int64 -> float32 seam
        # before exact power-of-two scaling, then flush a subnormal result.
        var rounded = Float32(accumulator)
        var expected = Float32(Float64(rounded) * _power2(Int(aep.unsafe_load(row)) + Int(bep.unsafe_load(col))))
        var bits = bitcast[DType.uint32](expected)
        if (bits & 0x7FFFFFFF) < 0x00800000:
            bits = 0
        if bitcast[DType.uint32](cp.unsafe_load(cell)) != bits:
            fp.unsafe_store(cell, 1)
    host_parallelize(cell_task, m * n)
    for cell in range(m * n):
        require(failures[cell] == 0, "INT8 dot-product cell " + String(cell))
    _ = actual
    _ = qa
    _ = ea
    _ = qb
    _ = eb


def check_dequantized(actual: List[Float32], codes: List[Int8], exponents: List[Int32], rows: Int, cols: Int) raises:
    var failures = List[Int](length=rows, fill=0)
    var fp = failures.unsafe_ptr()
    var yp = actual.unsafe_ptr()
    var qp = codes.unsafe_ptr()
    var ep = exponents.unsafe_ptr()
    def row_task(row: Int) {imm fp, imm yp, imm qp, imm ep, imm cols}:
        var scale = _power2(Int(ep.unsafe_load(row)))
        for col in range(cols):
            var i = row * cols + col
            var expected = Float32(Float64(qp.unsafe_load(i)) * scale)
            if bitcast[DType.uint32](yp.unsafe_load(i)) != bitcast[DType.uint32](expected):
                fp.unsafe_store(row, 1)
    host_parallelize(row_task, rows)
    for row in range(rows):
        require(failures[row] == 0, "INT8 dequantization row " + String(row))
    _ = actual
    _ = codes
    _ = exponents


# ---- The host column entry (was int8_linalg_cpu.mojo) ----
# Typed host entries; independent fixtures, rows and output cells parallel.
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


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(INT8_LINALG_FIXTURES)
    def task(fixture_id: Int) {imm cases}:
        try:
            check_fixture(fixture_id)
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, INT8_LINALG_FIXTURES)


# ---- The device column entry (was int8_linalg_gpu.mojo) ----
# Direct GPU quantizer/GEMM/dequantizer with independent native oracles.
#
# The existing upload/download helpers own pinned buffers and synchronize;
# one GPU context submits shapes in order. Kernels, native fixture preparation
# and oracle rows/cells run in parallel. No hashes or cross-vendor run here.
def check_fixture_gpu(ctx: DeviceContext, fixture_id: Int) raises:
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


def _run_gpu(mut report: Report) raises:
    var ctx = DeviceContext()
    var cases = report.cases(INT8_LINALG_FIXTURES)
    for fixture_id in range(INT8_LINALG_FIXTURES):
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
    return Check("int8_linalg", cpu=True, gpu=True, serial="", run=run_cases)
