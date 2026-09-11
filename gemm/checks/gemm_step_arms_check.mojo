# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GEMM step arms gate (DEVIATION 2543).

Every arm geometry of DEVIATIONS 2540 and 2541, forced through
`identical_gemm_step_geometry_into`, must store the SAME BITS as the shipped
`PLAN_TUNED_128_8X8` and as `PLAN_FLAT`, and its sabotage instantiation must
move EXACTLY one cell per block of that geometry (brief section 10.1 item
6), which proves the arm kernel ran and names the geometry that ran. Then
the twelve LM calls of the step at the target shape (brief section 2, the
three vocab-sized head calls among them) go through `identical_gemm_into`,
the entry every GEMM of the step reaches, under `MOJOLEARN_GEMM_ARM` set
for each arm in turn: bits equal to the shipped plan, and reach per call
(one moved cell per block where the arm applies, none where it does not).

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . \\
        gemm/checks/gemm_step_arms_check.mojo -o <bin>
    MOJOLEARN_GEMM_STEP_CHECK_LM=0 <bin>   # a light box: ragged controls and selector only
    <bin>                                  # a GPU box: plus the twelve LM calls

Knobs: `MOJOLEARN_GEMM_STEP_CHECK_LM=0` skips the LM section (and says so);
`MOJOLEARN_GEMM_STEP_CHECK_ARMS=<comma list>` restricts the LM section's
arms; `MOJOLEARN_GEMM_STEP_CHECK_FLOPS` (default 400,000,000) caps
`m n k` of a ragged control.

Without `-D MOJOLEARN_GEMM_ARM_TRIAL=1` every geometry runs the shipped plan,
the sabotage moves nothing, and this check FAILS saying so. It refuses a
build that defines one of `gemm_identical.mojo`'s global sabotage switches,
because its baseline would be a sabotaged kernel.

ENGINEERING_RULES 8: the switch is exercised on both sides by name. The
`shipped` arm through the entry must equal the explicit plan and its
sabotage must move nothing; every other arm must move one cell per block
where it applies.
"""
from std.os import getenv, setenv
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import numeric_mode_name
from gemm.checks.gemm_identical import (
    ANY_SABOTAGE,
    GEMM_ARM_SHIPPED,
    GEMM_ARM_TRIAL,
    GEMM_GEOM_COUNT,
    GEMM_GEOM_SHIPPED,
    PLAN_FLAT,
    PLAN_TUNED_128_8X8,
    choose_gemm_plan,
    gemm_plan_name,
    gemm_sabotage_name,
    gemm_step_arm_geometry,
    gemm_step_arm_name,
    gemm_step_arm_parse,
    gemm_step_geometry_blocks,
    gemm_step_geometry_name,
    identical_gemm_into,
    identical_gemm_step_geometry_into,
    identical_gemm_with_plan,
    identical_gemm_workspace_floats,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN, op_name
from gemm.checks.gemm_step_arms import (
    GEMM_STEP_LM_CALLS,
    gemm_step_compare,
    gemm_step_env_int,
    gemm_step_fill,
    gemm_step_lm_call,
    gemm_step_lm_call_name,
    gemm_step_operand_counts,
    gemm_step_poison,
    gemm_step_poison_left,
    gemm_step_readback,
    gemm_step_selected,
)

comptime RUN_PLAN = 0
comptime RUN_GEOMETRY = 1
comptime RUN_ENTRY = 2


def _trial_hint() -> String:
    comptime if GEMM_ARM_TRIAL:
        return String("")
    return String(
        " (this build lacks -D MOJOLEARN_GEMM_ARM_TRIAL=1: every geometry ran the shipped plan)"
    )


def _arm_names() -> List[String]:
    var names: List[String] = [
        "shipped", "lfold", "half", "half_ks16", "quarter", "head", "half_head"
    ]
    return names^


def _run(
    ctx: DeviceContext,
    mut dc: DeviceBuffer[DType.float32],
    mut da: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float32],
    mut h: HostBuffer[DType.float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    how: Int,
    which: Int,
    sabotage: Bool,
) raises:
    """Poison `C`, launch one way, wait, read `C` back into `h`."""
    gemm_step_poison(ctx, dc, h, m * n)
    if how == RUN_PLAN:
        identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, op, which)
    elif how == RUN_GEOMETRY:
        identical_gemm_step_geometry_into(
            ctx, dc, da, db, dw, m, n, k, op, which, sabotage
        )
    else:
        identical_gemm_into(ctx, dc, da, db, dw, m, n, k, op)
    ctx.synchronize()
    gemm_step_readback(ctx, dc, h)


# ===========================================================================
# THE SELECTOR (DEVIATION 2542)
# ===========================================================================


def check_selector(ctx: DeviceContext, mut failures: List[String]) raises:
    """Every arm name round-trips; the empty name is `shipped`; an unknown
    name raises from the parser AND from `identical_gemm_into` on a trial
    build (the harnesses rely on it)."""
    var names = _arm_names()
    for i in range(len(names)):
        var arm = gemm_step_arm_parse(names[i])
        if gemm_step_arm_name(arm) != names[i]:
            failures.append(
                "selector: '" + names[i] + "' parses to '" + gemm_step_arm_name(arm) + "'"
            )
    if gemm_step_arm_parse(String("")) != GEMM_ARM_SHIPPED:
        failures.append("selector: the empty name is not the shipped arm")
    var parse_raised = False
    try:
        _ = gemm_step_arm_parse(String("not_an_arm"))
    except:
        parse_raised = True
    if not parse_raised:
        failures.append("selector: gemm_step_arm_parse accepted 'not_an_arm'")

    var da = ctx.enqueue_create_buffer[DType.float32](16)
    var db = ctx.enqueue_create_buffer[DType.float32](16)
    var dc = ctx.enqueue_create_buffer[DType.float32](16)
    var dw = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(4, 4, 4)
    )
    ctx.synchronize()
    _ = setenv("MOJOLEARN_GEMM_ARM", "not_an_arm", True)
    var entry_raised = False
    try:
        identical_gemm_into(ctx, dc, da, db, dw, 4, 4, 4, OP_NN)
        ctx.synchronize()
    except:
        entry_raised = True
    _ = setenv("MOJOLEARN_GEMM_ARM", "", True)
    comptime if GEMM_ARM_TRIAL:
        if not entry_raised:
            failures.append(
                "selector: identical_gemm_into did not raise under MOJOLEARN_GEMM_ARM=not_an_arm"
            )
    else:
        failures.append(
            "selector: identical_gemm_into reads no MOJOLEARN_GEMM_ARM on this build (raised="
            + String(entry_raised) + ")" + _trial_hint()
        )
    print("selector: " + String(len(names)) + " names round-trip, unknown raises from the parser="
          + String(parse_raised) + " and from identical_gemm_into=" + String(entry_raised))
    _ = da
    _ = db
    _ = dc
    _ = dw


# ===========================================================================
# RAGGED AND ADVERSARIAL CONTROLS: every geometry forced
# ===========================================================================


def _ragged_case(
    ctx: DeviceContext,
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    subnormal: Bool,
    salt: Int,
    mut failures: List[String],
    mut reach_ok: List[Int],
    mut reach_runs: List[Int],
) raises:
    var mn = m * n
    var counts = gemm_step_operand_counts(m, n, k)
    var na = counts[0] if counts[0] > 0 else 1
    var nb = counts[1] if counts[1] > 0 else 1
    var nws = identical_gemm_workspace_floats(m, n, k, PLAN_FLAT)
    var w128 = identical_gemm_workspace_floats(m, n, k, PLAN_TUNED_128_8X8)
    if w128 > nws:
        nws = w128
    if nws < 1:
        nws = 1
    var da = ctx.enqueue_create_buffer[DType.float32](na)
    var db = ctx.enqueue_create_buffer[DType.float32](nb)
    var dc = ctx.enqueue_create_buffer[DType.float32](mn)
    var dw = ctx.enqueue_create_buffer[DType.float32](nws)
    var hexp = ctx.enqueue_create_host_buffer[DType.float32](mn)
    var hgot = ctx.enqueue_create_host_buffer[DType.float32](mn)
    ctx.synchronize()
    gemm_step_fill(ctx, da, counts[0], salt, subnormal)
    gemm_step_fill(ctx, db, counts[1], salt + 7919, subnormal)
    var tag = op_name(op) + " " + String(m) + "x" + String(n) + "x" + String(k)
    tag += " subnormal" if subnormal else " ordinary"

    # The reference: the shipped 128x128 plan, forced.
    _run(ctx, dc, da, db, dw, hexp, op, m, n, k, RUN_PLAN, PLAN_TUNED_128_8X8, False)
    var left = gemm_step_poison_left(hexp, mn)
    if left != 0:
        failures.append(tag + ": the shipped 128x128 plan left " + String(left) + " cells poisoned")
        return
    # FLAT against it: the two references agree before any arm is judged.
    _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_PLAN, PLAN_FLAT, False)
    var cf = gemm_step_compare(hgot, hexp, mn)
    if cf[0] != 0 or cf[1] != 0:
        failures.append(
            tag + ": FLAT differs from the shipped 128x128 plan in " + String(cf[0])
            + " cells (first " + String(cf[2]) + "), poison " + String(cf[1])
        )
    for geom in range(1, GEMM_GEOM_COUNT):
        var gname = gemm_step_geometry_name(geom)
        _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_GEOMETRY, geom, False)
        var cc = gemm_step_compare(hgot, hexp, mn)
        if cc[0] != 0 or cc[1] != 0:
            failures.append(
                tag + " [" + gname + "]: MOVED " + String(cc[0]) + " cells against the shipped plan (first "
                + String(cc[2]) + "), poison " + String(cc[1])
            )
        _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_GEOMETRY, geom, True)
        var cs = gemm_step_compare(hgot, hexp, mn)
        var blocks = gemm_step_geometry_blocks(geom, m, n)
        reach_runs[geom] += 1
        if cs[0] == blocks and cs[1] == 0:
            reach_ok[geom] += 1
        else:
            failures.append(
                tag + " [" + gname + "]: REACH NOT PROVEN: the sabotage moved " + String(cs[0])
                + " cells and one per block is " + String(blocks) + _trial_hint()
            )
    _ = da
    _ = db
    _ = dc
    _ = dw
    _ = hexp
    _ = hgot


def check_ragged_controls(ctx: DeviceContext, mut failures: List[String]) raises:
    """All three ops; `k` in {0, 1, 128, 129, 300, 1000, 2049, 50257} (P = 0,
    1, 1, 2 with a one-step leaf, 3, 8, 17 with a carry, 393); `m x n` off
    every tile multiple of every geometry (1x3, 33x70, 129x257, 300x129,
    257x520); the subnormal-product kind at 129x257 with k 129 and 1000."""
    var dims: List[Int] = [1, 3, 33, 70, 129, 257, 300, 129, 257, 520]
    var ks: List[Int] = [0, 1, 128, 129, 300, 1000, 2049, 50257]
    var ops: List[Int] = [OP_NN, OP_NT, OP_TN]
    var budget = gemm_step_env_int("MOJOLEARN_GEMM_STEP_CHECK_FLOPS", 400_000_000)
    var reach_ok = List[Int]()
    var reach_runs = List[Int]()
    for _ in range(GEMM_GEOM_COUNT):
        reach_ok.append(0)
        reach_runs.append(0)
    var before = len(failures)
    var cases = 0
    var skipped = 0
    for s in range(len(dims) // 2):
        var m = dims[2 * s]
        var n = dims[2 * s + 1]
        for ki in range(len(ks)):
            var k = ks[ki]
            var kk = k if k > 0 else 1
            if m * n * kk > budget:
                skipped += 1
                continue
            for oi in range(len(ops)):
                var salt = 101 + 13 * s + 7 * ki + oi
                _ragged_case(ctx, ops[oi], m, n, k, False, salt, failures, reach_ok, reach_runs)
                cases += 1
                if m == 129 and n == 257 and (k == 129 or k == 1000):
                    _ragged_case(ctx, ops[oi], m, n, k, True, salt + 1, failures, reach_ok, reach_runs)
                    cases += 1
    for geom in range(1, GEMM_GEOM_COUNT):
        print(
            "REACH ragged [" + gemm_step_geometry_name(geom) + "] "
            + String(reach_ok[geom]) + "/" + String(reach_runs[geom]) + " launches moved one cell per block"
        )
    print(
        "ragged controls: " + String(cases) + " cases x " + String(GEMM_GEOM_COUNT - 1)
        + " geometries, " + String(skipped) + " (m n k) over the " + String(budget)
        + " budget skipped, " + String(len(failures) - before) + " failures"
    )


# ===========================================================================
# THE TWELVE LM CALLS THROUGH THE ENTRY THE STEP REACHES
# ===========================================================================


def check_lm_calls(ctx: DeviceContext, mut failures: List[String]) raises:
    var names = _arm_names()
    var spec = String(getenv("MOJOLEARN_GEMM_STEP_CHECK_ARMS"))
    for i in range(GEMM_STEP_LM_CALLS):
        var call = gemm_step_lm_call(i)
        var op = call[0]
        var m = call[1]
        var n = call[2]
        var k = call[3]
        var mn = m * n
        var cname = gemm_step_lm_call_name(i)
        var plan = choose_gemm_plan(m, n, k)
        if plan != PLAN_TUNED_128_8X8:
            failures.append(
                "LM " + cname + ": choose_gemm_plan returns " + gemm_plan_name(plan)
                + ", not the 128x128 plan the brief's section 2 table names"
            )
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
        _ = setenv("MOJOLEARN_GEMM_ARM", "", True)
        _ = setenv("MOJOLEARN_GEMM_ARM_SABOTAGE", "", True)
        _run(ctx, dc, da, db, dw, hexp, op, m, n, k, RUN_PLAN, PLAN_TUNED_128_8X8, False)
        var left = gemm_step_poison_left(hexp, mn)
        if left != 0:
            failures.append("LM " + cname + ": the shipped plan left " + String(left) + " cells poisoned")
            continue
        for ai in range(len(names)):
            var an = names[ai]
            if not gemm_step_selected(spec, an):
                continue
            var arm = gemm_step_arm_parse(an)
            var geom = gemm_step_arm_geometry(arm, m, n, k)
            var expected = 0
            if geom != GEMM_GEOM_SHIPPED:
                expected = gemm_step_geometry_blocks(geom, m, n)
            _ = setenv("MOJOLEARN_GEMM_ARM", an, True)
            _ = setenv("MOJOLEARN_GEMM_ARM_SABOTAGE", "0", True)
            _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_ENTRY, 0, False)
            var cc = gemm_step_compare(hgot, hexp, mn)
            _ = setenv("MOJOLEARN_GEMM_ARM_SABOTAGE", "1", True)
            _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_ENTRY, 0, False)
            var cs = gemm_step_compare(hgot, hexp, mn)
            _ = setenv("MOJOLEARN_GEMM_ARM_SABOTAGE", "", True)
            var ok = cc[0] == 0 and cc[1] == 0 and cs[0] == expected and cs[1] == 0
            print(
                "LM " + cname + " " + op_name(op) + " " + String(m) + "x" + String(n) + "x" + String(k)
                + " arm=" + an + " geometry=[" + gemm_step_geometry_name(geom) + "] blocks="
                + String(expected) + " clean_moved=" + String(cc[0]) + " sabotage_moved="
                + String(cs[0]) + (" OK" if ok else " FAIL")
            )
            if not ok:
                failures.append(
                    "LM " + cname + " arm=" + an + ": clean moved " + String(cc[0]) + " (poison "
                    + String(cc[1]) + "), sabotage moved " + String(cs[0]) + " of the expected "
                    + String(expected) + _trial_hint()
                )
        _ = setenv("MOJOLEARN_GEMM_ARM", "", True)
        _ = da
        _ = db
        _ = dc
        _ = dw
        _ = hexp
        _ = hgot


def main() raises:
    print(
        "== gemm/checks/gemm_step_arms_check.mojo [" + numeric_mode_name() + "] trial="
        + String(GEMM_ARM_TRIAL) + " sabotage: " + gemm_sabotage_name() + " =="
    )
    print("   DEVIATIONS 2540 to 2543; docs/lanes/BRIEF_gemm_step_2026-09-11.md sections 6 and 10")
    comptime if ANY_SABOTAGE:
        raise Error(
            "gemm_step_arms_check: refuses a build with a global GEMM sabotage ("
            + gemm_sabotage_name() + "): its baseline would be a sabotaged kernel"
        )
    var ctx = DeviceContext()
    var failures = List[String]()
    check_selector(ctx, failures)
    check_ragged_controls(ctx, failures)
    var lm = String(getenv("MOJOLEARN_GEMM_STEP_CHECK_LM"))
    if lm == "0":
        print(
            "LM calls SKIPPED (MOJOLEARN_GEMM_STEP_CHECK_LM=0): the twelve target-shape calls,"
            " the three vocab-sized head calls among them, are NOT checked by this run"
        )
    else:
        check_lm_calls(ctx, failures)
    for i in range(len(failures)):
        if i < 60:
            print("FAIL " + failures[i])
    if len(failures) != 0:
        raise Error("gemm_step_arms_check: " + String(len(failures)) + " failures")
    var scope = String(", LM calls skipped") if lm == "0" else String(", and per LM call through identical_gemm_into")
    print(
        "PASS gemm step arms: every geometry bit-equal to the shipped 128x128 plan and to FLAT,"
        " reach proven per geometry" + scope
    )
