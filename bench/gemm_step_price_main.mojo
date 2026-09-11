# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Per-call price of one GEMM step arm against the shipped plan (DEVIATION 2543).

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_price_main.mojo -o <bin>
    MOJOLEARN_GEMM_ARM=half <bin>

For each of the twelve LM calls of the step at the target shape
(`gemm/checks/gemm_step_arms.mojo`): the shipped dispatch (`choose_gemm_plan`)
and the arm (`gemm_step_arm_geometry` through
`identical_gemm_step_geometry_into`) each write a poisoned `C`, and the two
outputs must be bit-equal with no poison left (BITS line; a MOVED call is
not timed and the process exits non-zero). Then `MOJOLEARN_GEMM_STEP_WARMUPS`
(2) warmups of both and `MOJOLEARN_GEMM_STEP_ROUNDS` (7) rounds alternating
which runs first, host-synchronized, raw samples printed. PRICE and TABLE
lines name the plan and the geometry beside the timing (ENGINEERING_RULES
8). The STEP line weights each call's median by its per-step count: a GEMM
sum, not a step time. `MOJOLEARN_GEMM_STEP_CALLS=<comma list>` restricts the
calls (`proj_fwd` ... `head_dB`).

Operands are the hashed ordinary kind. This is a per-call kernel price; the
default-flip input under ENGINEERING_RULES 9 is the LM step on the two
corpora (`tools/gemm_step_leg.sh`), never this harness.
"""
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import TARGET_COLUMN
from checks.numerics import numeric_mode_name
from gemm.checks.gemm_identical import (
    GEMM_ARM_SABOTAGE,
    GEMM_ARM_TRIAL,
    GEMM_GEOM_SHIPPED,
    choose_gemm_plan,
    gemm_plan_name,
    gemm_step_arm_from_env,
    gemm_step_arm_geometry,
    gemm_step_arm_name,
    gemm_step_geometry_blocks,
    gemm_step_geometry_name,
    identical_gemm_step_geometry_into,
    identical_gemm_with_plan,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import op_name
from gemm.checks.gemm_step_arms import (
    GEMM_STEP_LM_CALLS,
    gemm_step_compare,
    gemm_step_digest,
    gemm_step_env_int,
    gemm_step_fill,
    gemm_step_lm_call,
    gemm_step_lm_call_name,
    gemm_step_median_ms,
    gemm_step_operand_counts,
    gemm_step_poison,
    gemm_step_poison_left,
    gemm_step_readback,
    gemm_step_selected,
)


def _launch(
    ctx: DeviceContext,
    mut dc: DeviceBuffer[DType.float32],
    mut da: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    candidate: Bool,
    plan: Int,
    geom: Int,
) raises:
    if candidate:
        identical_gemm_step_geometry_into(ctx, dc, da, db, dw, m, n, k, op, geom, False)
    else:
        identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, op, plan)
    ctx.synchronize()


def main() raises:
    var arm = gemm_step_arm_from_env()
    if (arm & GEMM_ARM_SABOTAGE) != 0:
        raise Error("gemm_step_price: refuses MOJOLEARN_GEMM_ARM_SABOTAGE=1; a sabotaged arm has no price")
    var name = gemm_step_arm_name(arm)
    var rounds = gemm_step_env_int("MOJOLEARN_GEMM_STEP_ROUNDS", 7)
    var warmups = gemm_step_env_int("MOJOLEARN_GEMM_STEP_WARMUPS", 2)
    if rounds < 1 or warmups < 0:
        raise Error("gemm_step_price: rounds must be positive and warmups non-negative")
    var calls = String(getenv("MOJOLEARN_GEMM_STEP_CALLS"))
    print(
        "== bench/gemm_step_price_main.mojo [" + numeric_mode_name() + "] arm=" + name
        + " trial=" + String(GEMM_ARM_TRIAL) + " column=" + String(TARGET_COLUMN)
        + " rounds=" + String(rounds) + " warmups=" + String(warmups) + " =="
    )
    print("timing=host-synchronized median ms; raw samples below; the BITS pass is not a warmup")
    comptime if not GEMM_ARM_TRIAL:
        print("NOTE: no -D MOJOLEARN_GEMM_ARM_TRIAL=1: every arm runs the shipped plan; this prices the shipped plan against itself")
    var ctx = DeviceContext()
    var step_shipped = Float64(0.0)
    var step_arm = Float64(0.0)
    var priced = 0
    var failures = 0
    print(
        "TABLE call | op | m x n x k | per step | shipped plan | arm geometry | shipped blocks"
        " | arm blocks | shipped median ms | arm median ms | ratio"
    )
    for i in range(GEMM_STEP_LM_CALLS):
        var cname = gemm_step_lm_call_name(i)
        if not gemm_step_selected(calls, cname):
            continue
        var call = gemm_step_lm_call(i)
        var op = call[0]
        var m = call[1]
        var n = call[2]
        var k = call[3]
        var per = call[4]
        var mn = m * n
        var plan = choose_gemm_plan(m, n, k)
        var geom = gemm_step_arm_geometry(arm, m, n, k)
        var counts = gemm_step_operand_counts(m, n, k)
        var nws = identical_gemm_workspace_max_floats(m, n, k)
        var da = ctx.enqueue_create_buffer[DType.float32](counts[0])
        var db = ctx.enqueue_create_buffer[DType.float32](counts[1])
        var dc = ctx.enqueue_create_buffer[DType.float32](mn)
        var dw = ctx.enqueue_create_buffer[DType.float32](nws)
        var hexp = ctx.enqueue_create_host_buffer[DType.float32](mn)
        var hgot = ctx.enqueue_create_host_buffer[DType.float32](mn)
        ctx.synchronize()
        gemm_step_fill(ctx, da, counts[0], 11 + i, False)
        gemm_step_fill(ctx, db, counts[1], 22 + i, False)

        gemm_step_poison(ctx, dc, hexp, mn)
        _launch(ctx, dc, da, db, dw, op, m, n, k, False, plan, geom)
        gemm_step_readback(ctx, dc, hexp)
        gemm_step_poison(ctx, dc, hgot, mn)
        _launch(ctx, dc, da, db, dw, op, m, n, k, True, plan, geom)
        gemm_step_readback(ctx, dc, hgot)
        var cmp = gemm_step_compare(hgot, hexp, mn)
        var left = gemm_step_poison_left(hexp, mn)
        var gname = gemm_step_geometry_name(geom)
        if cmp[0] != 0 or cmp[1] != 0 or left != 0:
            print(
                "BITS " + cname + " MOVED cells=" + String(cmp[0]) + " first=" + String(cmp[2])
                + " poison_arm=" + String(cmp[1]) + " poison_shipped=" + String(left)
                + " geometry=[" + gname + "]"
            )
            failures += 1
            continue
        print("BITS " + cname + " EQUAL cells=" + String(mn) + " digest=" + hex(gemm_step_digest(hexp, mn)))

        for _ in range(warmups):
            _launch(ctx, dc, da, db, dw, op, m, n, k, False, plan, geom)
            _launch(ctx, dc, da, db, dw, op, m, n, k, True, plan, geom)
        var s_shipped = List[Int]()
        var s_arm = List[Int]()
        for r in range(rounds):
            var arm_first = (r % 2) == 1
            for slot in range(2):
                var candidate = (slot == 0) == arm_first
                var t0 = perf_counter_ns()
                _launch(ctx, dc, da, db, dw, op, m, n, k, candidate, plan, geom)
                var dt = Int(perf_counter_ns() - t0)
                if candidate:
                    s_arm.append(dt)
                else:
                    s_shipped.append(dt)
            print(
                "SAMPLE " + cname + " round=" + String(r) + " arm_first=" + String(arm_first)
                + " shipped_ns=" + String(s_shipped[r]) + " arm_ns=" + String(s_arm[r])
            )
        var ms_shipped = gemm_step_median_ms(s_shipped)
        var ms_arm = gemm_step_median_ms(s_arm)
        var ratio = Float64(0.0)
        if ms_shipped > 0.0:
            ratio = ms_arm / ms_shipped
        var flops = Float64(2) * Float64(m) * Float64(n) * Float64(k)
        var blocks_shipped = gemm_step_geometry_blocks(GEMM_GEOM_SHIPPED, m, n)
        var blocks_arm = gemm_step_geometry_blocks(geom, m, n)
        print(
            "PRICE gemm " + name + " " + cname + " shipped_ms=" + String(ms_shipped)
            + " arm_ms=" + String(ms_arm) + " ratio=" + String(ratio)
            + " shipped_tflops=" + String(flops / (ms_shipped * 1.0e9))
            + " arm_tflops=" + String(flops / (ms_arm * 1.0e9))
            + " plan=[" + gemm_plan_name(plan) + "] geometry=[" + gname + "]"
        )
        print(
            "TABLE " + cname + " | " + op_name(op) + " | " + String(m) + " x " + String(n) + " x "
            + String(k) + " | " + String(per) + " | " + gemm_plan_name(plan) + " | " + gname
            + " | " + String(blocks_shipped) + " | " + String(blocks_arm) + " | "
            + String(ms_shipped) + " | " + String(ms_arm) + " | " + String(ratio)
        )
        step_shipped += Float64(per) * ms_shipped
        step_arm += Float64(per) * ms_arm
        priced += 1
        _ = da
        _ = db
        _ = dc
        _ = dw
        _ = hexp
        _ = hgot
    var step_ratio = Float64(0.0)
    if step_shipped > 0.0:
        step_ratio = step_arm / step_shipped
    print(
        "STEP gemm arm=" + name + " ratio=" + String(step_ratio) + " shipped_ms=" + String(step_shipped)
        + " arm_ms=" + String(step_arm) + " calls=" + String(priced) + " moved=" + String(failures)
        + " (per-call medians weighted by per-step counts; a GEMM sum, not a step time)"
    )
    if failures != 0:
        raise Error("gemm_step_price: " + String(failures) + " calls MOVED BITS; an arm that changes the answer has no price")
    if priced == 0:
        raise Error("gemm_step_price: no call selected")
