# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""BF16 GEMM cases preserve the native-host and widened-fp32 relationships.

The same nine shapes/operand forms as test_linalg_lowbit. Conversion bits
are independently constructed by the integer BF16 oracle; the existing
normative fp32 GEMM remains the arithmetic reference, as in the Python test.
No digest, Python binding or changed kernel is involved.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, require, fixture_f32
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from decomposition.linalg_host_impl import gemm_bf16_host_run, gemm_host_run
from gemm.contract import OP_NN, OP_NT, OP_TN
from checks.native_tests.lowbit_oracles import expected_narrow, expected_widen
from checks.native_tests.lowbit_oracles import make_values
from max.gpu.host import DeviceContext
from gemm.checks.gemm_lowbit import LowbitWorkspace, bf16_widen, identical_gemm_bf16w_into
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.checks.gemm_lowbit_check import _upload_f32, _upload_u16, _download_f32, _poisoned
from std.sys.compile import is_defined
# Apple fast BF16 (lane py2mojo-D, batch P): the RUN entries of bindings/apple_fast_bf16_public_door.mojo
from gemm.afn_apple_fast import AFN_GEMM_BF16_MMA, afn_gemm_bf16_bits_into, afn_gemm_bf16_into
from gemm.checks.gemm_lowbit import bf16_narrow
from checks.native_tests.py2mojo_fixture_words import fixture_f64



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


def _host_fixture(fixture_id: Int) raises:
    var operands = fixture_operands(fixture_id)
    _ = host_results(operands[0], operands[1], operands[2], fixture_id)


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("bf16_linalg host", BF16_LINALG_FIXTURES, _host_fixture)


# The device column: one owned GPU context submits shape cases in order.
def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        var ctx = DeviceContext()
        var failed = String("")
        for fixture_id in range(BF16_LINALG_FIXTURES):  # driver-loop: one in-order device context
            try:
                check_fixture(ctx, fixture_id)
            except problem:
                failed += "\n      bf16_linalg device row " + String(fixture_id) + ": " + String(problem)
        if failed.byte_length() > 0:
            raise Error("bf16_linalg device: cases failed" + failed)


# ---- Apple fast BF16 (lane py2mojo-D, batch P, 2026-11-22) ----
# Ported from python/mojolearn/tests/test_linalg_fast_bf16.py `test_cold_matmul_entry_keeps_packed_uploads_alive`
# and `test_products_tail_shapes_transposes_and_float32_a`. The calls are the ones `linalg_fast.matmul_bf16` and
# `linalg_fast.to_bf16` make after their admission (bindings/apple_fast_bf16_public_door.mojo): `bf16_narrow` for
# the bits and `afn_gemm_bf16_into` (float32 a) or `afn_gemm_bf16_bits_into` (bf16 a), on one device context.
# They exist only in an Apple FAST build (AFN_GEMM_BF16_MMA); every other build lists the two tests as skipped,
# as the original skips when the loaded binaries are not Metal. The fresh-process subprocess, the no-Python-frame
# trace, ownership, aliases and refusals stay in the Python file.
#
#   fast_cold_one_hot  (m, n, k) in (1,9,7), (4,5,3), (1,9,8), (1,9,9), (4,5,8), every p < k: a = e_p columns
#                      (float32), b = rows p of bf16 1.0 (0x3f80): the product is exactly ones (35 products, the
#                      test's first matmul on a fresh context, before any conversion)
#   fast_products      18 rows: (m, n, k) in (1,9,7), (31,33,17), (65,63,129) x NN / NT / TN x float32 or bf16 a,
#                      operands drawn by NumPy (fixtures/bf16_fast/, tools/py2mojo_native_fixtures_bP.py): every
#                      cell within 4 k eps32 * (|A| . |B|) + 1e-6 of the float64 product of the RNE operands
comptime BF16_FAST_DIR = "checks/native_tests/fixtures/bf16_fast/"
comptime FLOAT32_EPS = 1.1920928955078125e-07


def fast_products_shape(s: Int) -> Tuple[Int, Int, Int]:
    if s == 0:
        return (1, 9, 7)
    if s == 1:
        return (31, 33, 17)
    return (65, 63, 129)


def fast_cold_one_hot(ctx: DeviceContext) raises:
    var shapes: List[Int] = [1, 9, 7, 4, 5, 3, 1, 9, 8, 1, 9, 9, 4, 5, 8]
    var count = 0
    for which in range(5):  # driver-loop: over the original's shapes, in order on the one context
        var m = shapes[3 * which]
        var n = shapes[3 * which + 1]
        var k = shapes[3 * which + 2]
        for p in range(k):  # driver-loop: over the one-hot position
            var a = List[Float32](length=m * k, fill=0)
            for r in range(m):  # driver-loop: building the host operand
                a[r * k + p] = 1.0
            var b = List[UInt16](length=k * n, fill=0)
            for c in range(n):  # driver-loop: building the host operand
                b[p * n + c] = 0x3F80
            var da = _upload_f32(ctx, a)
            var db = _upload_u16(ctx, b)
            var dc = _poisoned(ctx, m * n)
            var served = afn_gemm_bf16_into(ctx, dc, da, db, m, n, k, OP_NN)
            ctx.synchronize()
            require(served, "fast matmul_bf16 refused (" + String(m) + ", " + String(n) + ", " + String(k) + ")")
            var got = _download_f32(ctx, dc, m * n, "one-hot product")
            for i in range(m * n):  # driver-loop: over the downloaded product (test oracle, host)
                require(got[i] == 1.0, "one-hot (m, n, k, p) = (" + String(m) + ", " + String(n) + ", " + String(k)
                    + ", " + String(p) + ") cell " + String(i) + " is " + String(got[i]) + ", expected 1")
            _ = a^
            _ = b^
            _ = da^
            _ = db^
            _ = dc^
            count += 1
    require(count == 35, "one-hot products: " + String(count) + " of 35")


def fast_product_row(ctx: DeviceContext, row: Int) raises:
    var s = row // 6
    var t = (row // 2) % 3
    var a_bits = row % 2
    var shape = fast_products_shape(s)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    var op = OP_NN if t == 0 else (OP_NT if t == 1 else OP_TN)
    var tag = BF16_FAST_DIR + "mm-" + String(s) + "-" + String(t) + "-"
    var a = fixture_f32(tag + "a.f32", m * k)
    var b = fixture_f32(tag + "b.f32", k * n)
    var want = fixture_f64(tag + String(a_bits) + "-want.f64", m * n)
    var scale = fixture_f64(tag + String(a_bits) + "-scale.f64", m * n)
    var da = _upload_f32(ctx, a)
    var db32 = _upload_f32(ctx, b)
    var db = ctx.enqueue_create_buffer[DType.uint16](k * n)
    bf16_narrow(ctx, db, db32, k * n)
    var dc = _poisoned(ctx, m * n)
    var served: Bool
    if a_bits == 1:
        var da_bits = ctx.enqueue_create_buffer[DType.uint16](m * k)
        bf16_narrow(ctx, da_bits, da, m * k)
        served = afn_gemm_bf16_bits_into(ctx, dc, da_bits, db, m, n, k, op)
        ctx.synchronize()
        _ = da_bits^
    else:
        served = afn_gemm_bf16_into(ctx, dc, da, db, m, n, k, op)
        ctx.synchronize()
    var label = "(m, n, k) = (" + String(m) + ", " + String(n) + ", " + String(k) + ") op " + String(t) + " a_bits " + String(a_bits)
    require(served, "fast matmul_bf16 refused " + label)
    var got = _download_f32(ctx, dc, m * n, "fast BF16 product")
    for i in range(m * n):  # driver-loop: over the downloaded product (test oracle, host)
        var bound = 4.0 * Float64(k) * FLOAT32_EPS * scale[i] + 1e-6
        var error = abs(Float64(got[i]) - want[i])
        require(error <= bound, label + " cell " + String(i) + ": " + String(got[i]) + " expected " + String(want[i])
            + " (error " + String(error) + " > bound " + String(bound) + ")")
    _ = a^
    _ = b^
    _ = da^
    _ = db32^
    _ = db^
    _ = dc^


def test_fast_cold_one_hot() raises:
    comptime if AFN_GEMM_BF16_MMA:
        var ctx = DeviceContext()
        fast_cold_one_hot(ctx)


def test_fast_products() raises:
    comptime if AFN_GEMM_BF16_MMA:
        var ctx = DeviceContext()
        var failed = String("")
        for row in range(18):  # driver-loop: one in-order device context
            try:
                fast_product_row(ctx, row)
            except problem:
                failed += "\n      fast_products row " + String(row) + ": " + String(problem)
        if failed.byte_length() > 0:
            raise Error("bf16_linalg fast_products: cases failed" + failed)


def main() raises:
    # Each column registers its own tests by hand. Discovery (`__functions_in_module()`) reflects over every
    # function of the module, which elaborates the device helpers in the host column's build ("target 'unknown'
    # is not supported"), so a two-column file names its tests.
    var suite = TestSuite()
    # The Apple fast BF16 entries first: the cold one-hot runs before any other test has touched a context.
    comptime if AFN_GEMM_BF16_MMA:
        suite.test[test_fast_cold_one_hot]()
        suite.test[test_fast_products]()
    else:
        suite.test[test_fast_cold_one_hot]()
        suite.skip[test_fast_cold_one_hot]()
        suite.test[test_fast_products]()
        suite.skip[test_fast_products]()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
