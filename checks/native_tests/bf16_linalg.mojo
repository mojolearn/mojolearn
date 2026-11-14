# SPDX-License-Identifier: Apache-2.0
"""BF16 GEMM cases preserve the native-host and widened-fp32 relationships.

The same nine shapes/operand forms as test_linalg_lowbit. Conversion bits
are independently constructed by the integer BF16 oracle; the existing
normative fp32 GEMM remains the arithmetic reference, as in the Python test.
No digest, Python binding or changed kernel is involved.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from decomposition.linalg_host_impl import gemm_bf16_host_run, gemm_host_run
from gemm.contract import OP_NN, OP_NT, OP_TN
from checks.native_tests.bf16_conversion import expected_narrow, expected_widen
from checks.native_tests.int8_linalg import make_values
from max.gpu.host import DeviceContext
from gemm.checks.gemm_lowbit import LowbitWorkspace, bf16_widen, identical_gemm_bf16w_into
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.checks.gemm_lowbit_check import _upload_f32, _upload_u16, _download_f32, _poisoned
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime BF16_LINALG_FIXTURES = 9


def fixture_shape(fixture_id: Int) raises -> Tuple[Int, Int, Int, Int]:
    if fixture_id == 0:
        return (1, 32, 32, OP_NT)
    if fixture_id == 1:
        return (1, 64, 128, OP_NT)
    if fixture_id == 2:
        return (2, 96, 127, OP_NT)
    if fixture_id == 3:
        return (7, 33, 129, OP_NT)
    if fixture_id == 4:
        return (5, 9, 256, OP_NN)
    if fixture_id == 5:
        return (9, 5, 300, OP_TN)
    if fixture_id == 6:
        return (129, 129, 256, OP_NT)
    if fixture_id == 7:
        return (1, 1024, 512, OP_NT)
    if fixture_id == 8:
        return (6, 40, 200, OP_NT)
    raise Error("unknown BF16 GEMM fixture")


def narrow_input(values: List[Float32]) -> List[UInt16]:
    var result = List[UInt16](length=len(values), fill=0)
    var vp = values.unsafe_ptr()
    var rp = result.unsafe_ptr()
    var count = len(values)
    def task(unit: Int) {imm vp, imm rp, imm count}:
        var start = unit * 1024
        for i in range(start, min(start + 1024, count)):
            rp.unsafe_store(i, expected_narrow(bitcast[DType.uint32](vp.unsafe_load(i))))
    host_parallelize(task, (count + 1023) // 1024)
    _ = values
    return result^


def widen_input(values: List[UInt16]) -> List[Float32]:
    var result = List[Float32](length=len(values), fill=0)
    var vp = values.unsafe_ptr()
    var rp = result.unsafe_ptr()
    var count = len(values)
    def task(unit: Int) {imm vp, imm rp, imm count}:
        var start = unit * 1024
        for i in range(start, min(start + 1024, count)):
            rp.unsafe_store(i, bitcast[DType.float32](expected_widen(vp.unsafe_load(i))))
    host_parallelize(task, (count + 1023) // 1024)
    _ = values
    return result^


def check_words(actual: List[Float32], expected: List[Float32], label: String) raises:
    require(len(actual) == len(expected), label + " element count")
    var count = len(actual)
    var units = (count + 1023) // 1024
    var failures = List[Int](length=units, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = actual.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    def task(unit: Int) {imm fp, imm ap, imm ep, imm count}:
        var start = unit * 1024
        for i in range(start, min(start + 1024, count)):
            if bitcast[DType.uint32](ap.unsafe_load(i)) != bitcast[DType.uint32](ep.unsafe_load(i)):
                fp.unsafe_store(unit, 1)
    host_parallelize(task, units)
    for unit in range(units):
        require(failures[unit] == 0, label + " output chunk " + String(unit))
    _ = actual
    _ = expected


def host_results(a: List[Float32], abits: List[UInt16], bbits: List[UInt16], fixture_id: Int) raises -> Tuple[List[Float32], List[Float32]]:
    var shape = fixture_shape(fixture_id)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    var op = shape[3]
    var a_address = Int(abits.unsafe_ptr()) if fixture_id == 8 else Int(a.unsafe_ptr())
    var actual = List[Float32](length=m * n, fill=0)
    require(gemm_bf16_host_run(f32_ptr(Int(actual.unsafe_ptr())), a_address, Int(bbits.unsafe_ptr()), op, m, n, k, fixture_id == 8) == m * n, "host BF16 written count")
    var wide_a = widen_input(abits) if fixture_id == 8 else a.copy()
    var wide_b = widen_input(bbits)
    var expected = List[Float32](length=m * n, fill=0)
    require(gemm_host_run(f32_ptr(Int(expected.unsafe_ptr())), Int(wide_a.unsafe_ptr()), Int(wide_b.unsafe_ptr()), op, m, n, k) == m * n, "host fp32 written count")
    check_words(actual, expected, "host BF16 versus exact widened fp32")
    _ = a
    _ = abits
    _ = bbits
    _ = wide_a
    _ = wide_b
    return (actual^, expected^)


def fixture_operands(fixture_id: Int) raises -> Tuple[List[Float32], List[UInt16], List[UInt16]]:
    var shape = fixture_shape(fixture_id)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    var a = make_values(m, k, 3 if fixture_id == 8 else 10 + k)
    var b = make_values(n, k, 4 if fixture_id == 8 else 20 + k)
    var abits = narrow_input(a)
    var bbits = narrow_input(b)
    return (a^, abits^, bbits^)


# ---- The host column entry (was bf16_linalg_cpu.mojo) ----
# Independent host BF16 GEMM fixtures, using typed buffers and native calls.
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(BF16_LINALG_FIXTURES)
    def task(fixture_id: Int) {imm cases}:
        try:
            var operands = fixture_operands(fixture_id)
            _ = host_results(operands[0], operands[1], operands[2], fixture_id)
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, BF16_LINALG_FIXTURES)


# ---- The device column entry (was bf16_linalg_gpu.mojo) ----
# BF16 GPU GEMM against host and GPU widened-fp32 results, without bindings.
#
# One owned GPU context submits shape cases in order. Existing GEMM/widening
# kernels and native host input/word-check chunks provide the parallel work.
# The host reference is the same normative arithmetic used by the replaced
# Python tests; this is functional equivalence coverage, not an identity run.
def check_fixture(ctx: DeviceContext, fixture_id: Int) raises:
    var shape = fixture_shape(fixture_id)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    var op = shape[3]
    var operands = fixture_operands(fixture_id)
    var reference = host_results(operands[0], operands[1], operands[2], fixture_id)
    var da = _upload_f32(ctx, operands[0])
    var da_bits = _upload_u16(ctx, operands[1])
    var db = _upload_u16(ctx, operands[2])
    if fixture_id == 8:
        bf16_widen(ctx, da, da_bits, m * k)
    var dc = _poisoned(ctx, m * n)
    var work = LowbitWorkspace(ctx)
    identical_gemm_bf16w_into(ctx, dc, da, db, work, m, n, k, op)
    var wide_b = ctx.enqueue_create_buffer[DType.float32](n * k)
    bf16_widen(ctx, wide_b, db, n * k)
    var dc32 = _poisoned(ctx, m * n)
    var ws = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(m, n, k))
    identical_gemm_into[False](ctx, dc32, da, wide_b, ws, m, n, k, op)
    ctx.synchronize()
    var actual = _download_f32(ctx, dc, m * n, "BF16 native product")
    var fp32 = _download_f32(ctx, dc32, m * n, "BF16 widened-fp32 product")
    check_words(actual, reference[0], "GPU BF16 versus host")
    check_words(actual, fp32, "GPU BF16 versus exact widened fp32")
    _ = da
    _ = da_bits
    _ = db
    _ = dc
    _ = dc32
    _ = wide_b
    _ = ws
    _ = work^


def _run_gpu(mut report: Report) raises:
    var ctx = DeviceContext()
    var cases = report.cases(BF16_LINALG_FIXTURES)
    for fixture_id in range(BF16_LINALG_FIXTURES):
        try:
            check_fixture(ctx, fixture_id)
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
    return Check("bf16_linalg", cpu=True, gpu=True, serial="", run=run_cases)
