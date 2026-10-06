# SPDX-License-Identifier: Apache-2.0
"""IDENTICAL schedule experiment probe. Synthetic kernel screening only.

One excluded warmup, one synchronized sample. Correctness cases compare every
word with the flat plan. Every case hashes every output word before and after
the sample. The Python runner compares those hashes between frozen A/B builds.
No allocation, transfer, reference calculation or hashing is inside the timer.
"""
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from gemm.checks.gemm_identical import (
    PLAN_FLAT, GEMM_KSPLIT_SLACK, GEMM_KPACK_RPT, GEMM_KPACK_CPT,
    identical_gemm_into, identical_gemm_with_plan,
    identical_gemm_workspace_max_floats, gemm_shipped_dispatch_name,
)
from gemm.checks.gemm_step_arms import (
    gemm_step_env_int, gemm_step_fill, gemm_step_poison,
    gemm_step_readback, gemm_step_digest, gemm_step_compare,
    gemm_step_poison_left,
)


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var m = gemm_step_env_int("SCHEDULE_M", 65)
    var n = gemm_step_env_int("SCHEDULE_N", 67)
    var k = gemm_step_env_int("SCHEDULE_K", 257)
    var op = gemm_step_env_int("SCHEDULE_OP", 1)
    var tiny = gemm_step_env_int("SCHEDULE_TINY", 0) == 1
    var reference = gemm_step_env_int("SCHEDULE_REFERENCE", 1) == 1
    var timed = gemm_step_env_int("SCHEDULE_TIMED", 0) == 1
    if m < 1 or n < 1 or k < 1 or op < 0 or op > 2:
        raise Error("invalid shape or orientation")
    comptime mask = (
        # A02 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
        Int(is_defined["MOJOLEARN_GEMM_ONE_PAGE"]())
        # I01 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
        + 2 * Int(is_defined["MOJOLEARN_IDN_GEMM_GROUP_SLACK_2"]())
        # I01 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
        + 4 * Int(is_defined["MOJOLEARN_IDN_GEMM_GROUP_SLACK_8"]())
        # I01 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
        + 8 * Int(is_defined["MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY"]())
        # N01 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
        + 16 * Int(is_defined["MOJOLEARN_GEMM_KPACK_RPT4"]())
    )
    if mask != gemm_step_env_int("SCHEDULE_EXPECT_MASK", 0):
        raise Error("compiled experiment does not match requested arm")
    var ctx = DeviceContext()
    print("SCHEDULE_DEVICE", ctx.name())
    print("SCHEDULE_DISPATCH", gemm_shipped_dispatch_name(m, n, k))
    var a = ctx.enqueue_create_buffer[DType.float32](m * k)
    var b = ctx.enqueue_create_buffer[DType.float32](n * k)
    var c = ctx.enqueue_create_buffer[DType.float32](m * n)
    var h = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    var ws_n = max(1, identical_gemm_workspace_max_floats(m, n, k))
    var ws = ctx.enqueue_create_buffer[DType.float32](ws_n)
    gemm_step_fill(ctx, a, m * k, 17, tiny)
    gemm_step_fill(ctx, b, n * k, 31, tiny)
    gemm_step_poison(ctx, c, h, m * n)
    identical_gemm_into(ctx, c, a, b, ws, m, n, k, op)
    gemm_step_readback(ctx, c, h)
    var warm_hash = gemm_step_digest(h, m * n)
    if gemm_step_poison_left(h, m * n) != 0:
        raise Error("GEMM left unwritten output cells")
    if reference:
        var expected = ctx.enqueue_create_buffer[DType.float32](m * n)
        var hr = ctx.enqueue_create_host_buffer[DType.float32](m * n)
        identical_gemm_with_plan(ctx, expected, a, b, ws, m, n, k, op, PLAN_FLAT)
        gemm_step_readback(ctx, expected, hr)
        var diff = gemm_step_compare(h, hr, m * n)
        if diff[0] != 0 or diff[1] != 0:
            raise Error("flat-plan mismatch: " + String(diff[0]) + " cells")
        _ = expected^
        _ = hr^
    var elapsed = 0
    if timed:
        ctx.synchronize()
        var start = perf_counter_ns()
        identical_gemm_into(ctx, c, a, b, ws, m, n, k, op)
        ctx.synchronize()
        elapsed = perf_counter_ns() - start
        gemm_step_readback(ctx, c, h)
        if gemm_step_digest(h, m * n) != warm_hash:
            raise Error("warmup and timed output bits differ")
    print("SCHEDULE_RESULT mode=" + numeric_mode_name()
          + " column=" + column_name(TARGET_COLUMN)
          + " mask=" + String(mask) + " m=" + String(m) + " n=" + String(n)
          + " k=" + String(k) + " op=" + String(op) + " tiny=" + String(Int(tiny))
          + " reference=" + String(Int(reference)) + " timed=" + String(Int(timed))
          + " hash=" + String(warm_hash) + " ns=" + String(elapsed)
          + " workspace_bytes=" + String(ws_n * 4)
          + " slack=" + String(GEMM_KSPLIT_SLACK)
          + " rpt=" + String(GEMM_KPACK_RPT) + " cpt=" + String(GEMM_KPACK_CPT))
