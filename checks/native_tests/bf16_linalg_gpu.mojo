# SPDX-License-Identifier: Apache-2.0
"""BF16 GPU GEMM against host and GPU widened-fp32 results, without bindings.

One owned GPU context submits shape cases in order. Existing GEMM/widening
kernels and native host input/word-check chunks provide the parallel work.
The host reference is the same normative arithmetic used by the replaced
Python tests; this is functional equivalence coverage, not an identity run.
"""
from max.gpu.host import DeviceContext
from checks.native_tests.bf16_linalg_cases import (
    BF16_LINALG_FIXTURES, require, fixture_shape, fixture_operands, host_results, check_words,
)
from gemm.checks.gemm_lowbit import LowbitWorkspace, bf16_widen, identical_gemm_bf16w_into
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.checks.gemm_lowbit_check import _upload_f32, _upload_u16, _download_f32, _poisoned


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


def main() raises:
    var ctx = DeviceContext()
    var failed = 0
    for fixture_id in range(BF16_LINALG_FIXTURES):
        try:
            check_fixture(ctx, fixture_id)
        except error:
            print("FAIL bf16_linalg_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "bf16_linalg_gpu failed " + String(failed) + " fixtures")
    print("PASS bf16_linalg_gpu:", BF16_LINALG_FIXTURES, "native host/GPU/widened-fp32 equivalence fixtures")
