# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Per-call price of one GEMM step arm against the shipped dispatch (DEVIATIONS 2543, 2593 and 2595).

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_price_main.mojo -o <bin>
    MOJOLEARN_GEMM_ARM=half <bin>

For each of the twelve LM calls of the step at the target shape
(`gemm/checks/gemm_step_arms.mojo`): the shipped dispatch
(`identical_gemm_shipped_into`, the lines `identical_gemm_into` runs after its
trial hook) and the arm (`gemm_step_arm_geometry` through
`identical_gemm_step_geometry_into`) each write a poisoned `C`, and the two
outputs must be bit-equal with no poison left (BITS line; a MOVED call is
not timed and the process exits non-zero). Then `MOJOLEARN_GEMM_STEP_WARMUPS`
(2) warmups of both and `MOJOLEARN_GEMM_STEP_ROUNDS` (7) rounds alternating
which runs first, host-synchronized, raw samples printed. PRICE and TABLE
lines name the plan and the geometry beside the timing (ENGINEERING_RULES
8). The STEP line weights each call's median by its per-step count: a GEMM
sum, not a step time. `MOJOLEARN_GEMM_STEP_CALLS=<comma list>` restricts the
calls (`proj_fwd` ... `head_dB`, and the control names below).

DEVIATION 2593 (docs/lanes/BRIEF_gemm_long_k_2026-09-11.md sections 3.2 and
9). `MOJOLEARN_GEMM_STEP_CONTROLS=1` adds the six control calls of
`gemm_step_arms.mojo` (`ctl_*`), which separate the explanations of that
brief's section 3.2: BITS and SAMPLE lines as for an LM call, then ONE
CONTROL line each, never weighted into STEP. Where the arm's geometry at a
call is `ksplit` or `ksplit_leaf`: a PHASEBITS pass
(`identical_gemm_step_ksplit_phase_into` must store the shipped bits too),
then one PHASE line with the medians over the rounds of the workspace
allocation, the group launch and the fold launch, each host-synchronized
apart (the fold cost `F` of that brief's section 4). PRICE, TABLE and
CONTROL lines carry the arm's launched blocks (tiles times groups for
ksplit) and its group leaves (0 for a geometry that is not ksplit).

DEVIATION 2595 (the same brief, section 10). `shipped` is the shipped
DEFAULT: on a column whose `lib_gemm_block_parallelism_for` row is above 0
(NVIDIA) the calls the long-k group rule takes run ksplit and every other
call the plan `choose_gemm_plan` picks; on a column whose row is 0 it is the
old TUNED 128x128 plan. The old plan is priced as the arm `tuned128`. The
header prints one DEFAULT line (the column's row, whether the default is on,
the trial arm's `S`, the shipped summary) and one PLANLABEL line
(`gemm_step_arm_plan_label`). PRICE, TABLE and CONTROL lines name the plan
the shipped dispatch RAN at that call (`shipped_plan=[...]`, with its
launched blocks and group leaves) beside the arm's geometry, so no line can
confuse the new default with the old plan. When the arm's geometry is the
shipped one and the default takes the call, the PHASE line prices the
default's own phases (`phase_of=shipped_default`).
`MOJOLEARN_GEMM_STEP_LABEL_ONLY=1` prints the header and the PLANLABEL line
and exits before any device work (`tools/gemm_step_leg.sh` labels its LM
probes from it). With `MOJOLEARN_GEMM_STEP_LABEL_M`, `_N` and `_K` set (and
an optional `MOJOLEARN_GEMM_STEP_LABEL_CALLER` tag) it also prints one
host-only DISPATCH line for that caller shape under the arm: whether the
ksplit default takes it, the default's group size, `choose_gemm_plan`'s
plan, the shipped dispatch's plan and the arm's geometry
(`tools/gemm_ksplit_classical_leg.sh`, brief section 11).

DEVIATION 2599 (docs/lanes/BRIEF_gemm_kernel_2026-09-11.md section 6).
The arms `kpack` and `kpack_wide` get PRICE and TABLE lines like every arm.
Where the arm's own group rule takes a call, a PHASEBITS pass and a PHASE
line with `phase_of=arm_kpack` price the allocation, the packed group launch
and the fold launch apart (`identical_gemm_step_kpack_phase_into`). Where it
declines, the arm runs the whole leaf range in one launch and there is no
PHASE line (as for `tuned128`).

DEVIATIONS 2640 to 2642 (docs/lanes/BRIEF_gemm_final_2026-09-11.md sections 4
and 6). The arms `kfoldv` and `kfoldv_leaf` get PRICE and TABLE lines like every
arm. Where an arm's rule takes a call, a PHASEBITS pass and a PHASE line with
`phase_of=arm_kfold` price the allocation, the shipped group launch at the
arm's group size and the LANE fold apart
(`identical_gemm_step_kfold_phase_into`). Its `fold_ms` against the
`shipped_default` PHASE line of the `shipped` run on the same pod is what reads
brief section 4.4's models A to C. Where the rule declines, the arm runs the
shipped dispatch and the call carries the shipped default's PHASE line.

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
    GEMM_GEOM_KFOLDV,
    GEMM_GEOM_KFOLDV_LEAF,
    GEMM_GEOM_KPACK,
    GEMM_GEOM_KPACK_WIDE,
    GEMM_GEOM_KSPLIT,
    GEMM_GEOM_KSPLIT_LEAF,
    GEMM_GEOM_SHIPPED,
    GEMM_KSPLIT_DEFAULT_S,
    GEMM_KSPLIT_S,
    choose_gemm_plan,
    gemm_plan_name,
    gemm_shipped_dispatch_name,
    gemm_step_arm_from_env,
    gemm_step_arm_geometry,
    gemm_step_arm_name,
    gemm_step_arm_plan_label,
    gemm_step_geometry_blocks,
    gemm_step_geometry_group_leaves,
    gemm_step_geometry_launched_blocks,
    gemm_step_geometry_name,
    identical_gemm_shipped_into,
    identical_gemm_step_geometry_into,
    identical_gemm_step_kfold_phase_into,
    identical_gemm_step_kpack_phase_into,
    identical_gemm_step_ksplit_phase_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import op_name
from gemm.checks.gemm_step_arms import (
    GEMM_STEP_CONTROL_CALLS,
    GEMM_STEP_LM_CALLS,
    gemm_step_compare,
    gemm_step_control_call,
    gemm_step_control_call_name,
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
    geom: Int,
) raises:
    if candidate:
        identical_gemm_step_geometry_into(ctx, dc, da, db, dw, m, n, k, op, geom, False)
    else:
        # DEVIATION 2595: the reference is the SHIPPED dispatch (the ksplit
        # default where the column's row is above 0), never the old plan by
        # name; the arm `tuned128` prices the old plan against it.
        identical_gemm_shipped_into(ctx, dc, da, db, dw, m, n, k, op)
    ctx.synchronize()


def _price_call(
    ctx: DeviceContext,
    arm: Int,
    name: String,
    cname: String,
    call: Tuple[Int, Int, Int, Int, Int],
    salt: Int,
    rounds: Int,
    warmups: Int,
    control: Bool,
) raises -> Tuple[Float64, Float64, Int, Int]:
    """One call: BITS, warmups, alternated rounds, then PRICE and TABLE (an LM
    call) or CONTROL (a control call), then PHASEBITS and PHASE where the
    arm's geometry is ksplit, or where it is the shipped geometry and the
    shipped default takes the call. `(shipped median ms, arm median ms,
    priced 0 or 1, bits moved 0 or 1)`. The LM calls keep the salts the
    harness always used (`11 + i`, `22 + i`)."""
    var op = call[0]
    var m = call[1]
    var n = call[2]
    var k = call[3]
    var per = call[4]
    var mn = m * n
    var plan = choose_gemm_plan(m, n, k)
    var geom = gemm_step_arm_geometry(arm, m, n, k)
    var sname = gemm_shipped_dispatch_name(m, n, k)
    var counts = gemm_step_operand_counts(m, n, k)
    var nws = identical_gemm_workspace_max_floats(m, n, k)
    var da = ctx.enqueue_create_buffer[DType.float32](counts[0])
    var db = ctx.enqueue_create_buffer[DType.float32](counts[1])
    var dc = ctx.enqueue_create_buffer[DType.float32](mn)
    var dw = ctx.enqueue_create_buffer[DType.float32](nws)
    var hexp = ctx.enqueue_create_host_buffer[DType.float32](mn)
    var hgot = ctx.enqueue_create_host_buffer[DType.float32](mn)
    ctx.synchronize()
    gemm_step_fill(ctx, da, counts[0], 11 + salt, False)
    gemm_step_fill(ctx, db, counts[1], 22 + salt, False)

    gemm_step_poison(ctx, dc, hexp, mn)
    _launch(ctx, dc, da, db, dw, op, m, n, k, False, geom)
    gemm_step_readback(ctx, dc, hexp)
    gemm_step_poison(ctx, dc, hgot, mn)
    _launch(ctx, dc, da, db, dw, op, m, n, k, True, geom)
    gemm_step_readback(ctx, dc, hgot)
    var cmp = gemm_step_compare(hgot, hexp, mn)
    var left = gemm_step_poison_left(hexp, mn)
    var gname = gemm_step_geometry_name(geom)
    if cmp[0] != 0 or cmp[1] != 0 or left != 0:
        print(
            "BITS " + cname + " MOVED cells=" + String(cmp[0]) + " first=" + String(cmp[2])
            + " poison_arm=" + String(cmp[1]) + " poison_shipped=" + String(left)
            + " shipped_plan=[" + sname + "] geometry=[" + gname + "]"
        )
        _ = da
        _ = db
        _ = dc
        _ = dw
        return (Float64(0.0), Float64(0.0), 0, 1)
    print("BITS " + cname + " EQUAL cells=" + String(mn) + " digest=" + hex(gemm_step_digest(hexp, mn)))

    for _ in range(warmups):
        _launch(ctx, dc, da, db, dw, op, m, n, k, False, geom)
        _launch(ctx, dc, da, db, dw, op, m, n, k, True, geom)
    var s_shipped = List[Int]()
    var s_arm = List[Int]()
    for r in range(rounds):
        var arm_first = (r % 2) == 1
        for slot in range(2):
            var candidate = (slot == 0) == arm_first
            var t0 = perf_counter_ns()
            _launch(ctx, dc, da, db, dw, op, m, n, k, candidate, geom)
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
    var launched_shipped = gemm_step_geometry_launched_blocks(GEMM_GEOM_SHIPPED, m, n, k)
    var gleaves_shipped = gemm_step_geometry_group_leaves(GEMM_GEOM_SHIPPED, m, n, k)
    var blocks_arm = gemm_step_geometry_blocks(geom, m, n)
    var launched = gemm_step_geometry_launched_blocks(geom, m, n, k)
    var gleaves = gemm_step_geometry_group_leaves(geom, m, n, k)
    if control:
        print(
            "CONTROL gemm " + name + " " + cname + " " + op_name(op) + " " + String(m) + "x"
            + String(n) + "x" + String(k) + " shipped_ms=" + String(ms_shipped)
            + " arm_ms=" + String(ms_arm) + " ratio=" + String(ratio)
            + " shipped_tflops=" + String(flops / (ms_shipped * 1.0e9))
            + " arm_tflops=" + String(flops / (ms_arm * 1.0e9))
            + " shipped_blocks=" + String(blocks_shipped) + " arm_blocks=" + String(blocks_arm)
            + " shipped_launched_blocks=" + String(launched_shipped)
            + " shipped_group_leaves=" + String(gleaves_shipped)
            + " arm_launched_blocks=" + String(launched) + " group_leaves=" + String(gleaves)
            + " choose_plan=[" + gemm_plan_name(plan) + "] shipped_plan=[" + sname
            + "] geometry=[" + gname + "]"
        )
    else:
        print(
            "PRICE gemm " + name + " " + cname + " shipped_ms=" + String(ms_shipped)
            + " arm_ms=" + String(ms_arm) + " ratio=" + String(ratio)
            + " shipped_tflops=" + String(flops / (ms_shipped * 1.0e9))
            + " arm_tflops=" + String(flops / (ms_arm * 1.0e9))
            + " shipped_launched_blocks=" + String(launched_shipped)
            + " shipped_group_leaves=" + String(gleaves_shipped)
            + " arm_launched_blocks=" + String(launched) + " group_leaves=" + String(gleaves)
            + " choose_plan=[" + gemm_plan_name(plan) + "] shipped_plan=[" + sname
            + "] geometry=[" + gname + "]"
        )
        print(
            "TABLE " + cname + " | " + op_name(op) + " | " + String(m) + " x " + String(n) + " x "
            + String(k) + " | " + String(per) + " | " + sname + " | " + gname
            + " | " + String(launched_shipped) + " | " + String(blocks_arm) + " | "
            + String(ms_shipped) + " | " + String(ms_arm) + " | " + String(ratio)
            + " | " + String(launched) + " | " + String(gleaves) + " | " + String(gleaves_shipped)
        )

    # ---- DEVIATIONS 2593 and 2595: the ksplit phases, apart.
    var moved = 0
    var pleaves = 0
    var pblocks = 0
    var phase_of = String("")
    # DEVIATION 2599: the kpack arms time their own packed group launch.
    var kpack_phase = False
    # DEVIATIONS 2640 and 2641: the lane fold arms time their own fold launch.
    var kfold_phase = False
    if (geom == GEMM_GEOM_KFOLDV or geom == GEMM_GEOM_KFOLDV_LEAF) and gleaves > 0:
        pleaves = gleaves
        pblocks = launched
        phase_of = String("arm_kfold")
        kfold_phase = True
    elif geom == GEMM_GEOM_KSPLIT or geom == GEMM_GEOM_KSPLIT_LEAF:
        pleaves = gleaves
        pblocks = launched
        phase_of = String("arm")
    elif (geom == GEMM_GEOM_KPACK or geom == GEMM_GEOM_KPACK_WIDE) and gleaves > 0:
        pleaves = gleaves
        pblocks = launched
        phase_of = String("arm_kpack")
        kpack_phase = True
    elif geom == GEMM_GEOM_SHIPPED and gleaves_shipped > 0:
        pleaves = gleaves_shipped
        pblocks = launched_shipped
        phase_of = String("shipped_default")
    if pleaves > 0:
        gemm_step_poison(ctx, dc, hgot, mn)
        if kfold_phase:
            _ = identical_gemm_step_kfold_phase_into(ctx, dc, da, db, dw, m, n, k, op, geom)
        elif kpack_phase:
            _ = identical_gemm_step_kpack_phase_into(ctx, dc, da, db, dw, m, n, k, op, geom)
        else:
            _ = identical_gemm_step_ksplit_phase_into(ctx, dc, da, db, dw, m, n, k, op, pleaves)
        gemm_step_readback(ctx, dc, hgot)
        var pc = gemm_step_compare(hgot, hexp, mn)
        if pc[0] != 0 or pc[1] != 0:
            print(
                "PHASEBITS " + cname + " MOVED cells=" + String(pc[0]) + " first=" + String(pc[2])
                + " poison=" + String(pc[1]) + " phase_of=" + phase_of + " geometry=[" + gname + "]"
            )
            moved = 1
        else:
            print("PHASEBITS " + cname + " EQUAL cells=" + String(mn) + " phase_of=" + phase_of)
            var s_alloc = List[Int]()
            var s_group = List[Int]()
            var s_fold = List[Int]()
            var s_sum = List[Int]()
            for _ in range(rounds):
                var ph = (Int(0), Int(0), Int(0))
                if kfold_phase:
                    ph = identical_gemm_step_kfold_phase_into(ctx, dc, da, db, dw, m, n, k, op, geom)
                elif kpack_phase:
                    ph = identical_gemm_step_kpack_phase_into(ctx, dc, da, db, dw, m, n, k, op, geom)
                else:
                    ph = identical_gemm_step_ksplit_phase_into(ctx, dc, da, db, dw, m, n, k, op, pleaves)
                s_alloc.append(ph[0])
                s_group.append(ph[1])
                s_fold.append(ph[2])
                s_sum.append(ph[0] + ph[1] + ph[2])
            var groups = 0
            if blocks_arm > 0:
                groups = pblocks // blocks_arm
            print(
                "PHASE gemm " + name + " " + cname + " phase_of=" + phase_of
                + " group_leaves=" + String(pleaves)
                + " groups=" + String(groups) + " launched_blocks=" + String(pblocks)
                + " alloc_ms=" + String(gemm_step_median_ms(s_alloc))
                + " group_ms=" + String(gemm_step_median_ms(s_group))
                + " fold_ms=" + String(gemm_step_median_ms(s_fold))
                + " sum_ms=" + String(gemm_step_median_ms(s_sum))
                + " arm_ms=" + String(ms_arm) + " shipped_ms=" + String(ms_shipped)
                + " (each phase host-synchronized; medians over " + String(rounds) + " rounds)"
            )
    _ = da
    _ = db
    _ = dc
    _ = dw
    _ = hexp
    _ = hgot
    return (ms_shipped, ms_arm, 1, moved)


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
    var controls = String(getenv("MOJOLEARN_GEMM_STEP_CONTROLS")) == "1"
    print(
        "== bench/gemm_step_price_main.mojo [" + numeric_mode_name() + "] arm=" + name
        + " trial=" + String(GEMM_ARM_TRIAL) + " column=" + String(TARGET_COLUMN)
        + " rounds=" + String(rounds) + " warmups=" + String(warmups)
        + " controls=" + String(controls) + " =="
    )
    # DEVIATION 2595: which plan `shipped` is on this build, and which plan
    # this arm runs, before any timing.
    var default_state = String("off")
    if GEMM_KSPLIT_DEFAULT_S > 0:
        default_state = String("on")
    print(
        "DEFAULT gemm column=" + String(TARGET_COLUMN) + " block_parallelism_row="
        + String(GEMM_KSPLIT_DEFAULT_S) + " ksplit_default=" + default_state
        + " trial_ksplit_S=" + String(GEMM_KSPLIT_S)
        + " shipped=[" + gemm_step_geometry_name(GEMM_GEOM_SHIPPED) + "]"
    )
    print("PLANLABEL arm=" + name + " label=" + gemm_step_arm_plan_label(arm))
    # Brief section 11: one CALLER shape's dispatch on this build under this
    # arm, host only (no DeviceContext), so a classical A/B leg records which
    # plan each GEMM of a caller runs. `op` does not enter the dispatch.
    var label_m = gemm_step_env_int("MOJOLEARN_GEMM_STEP_LABEL_M", 0)
    if label_m > 0:
        var label_n = gemm_step_env_int("MOJOLEARN_GEMM_STEP_LABEL_N", 0)
        var label_k = gemm_step_env_int("MOJOLEARN_GEMM_STEP_LABEL_K", 0)
        var label_geom = gemm_step_arm_geometry(arm, label_m, label_n, label_k)
        var label_leaves = gemm_step_geometry_group_leaves(
            GEMM_GEOM_SHIPPED, label_m, label_n, label_k
        )
        var takes = String("no")
        if label_leaves > 0:
            takes = String("yes")
        print(
            "DISPATCH caller=" + String(getenv("MOJOLEARN_GEMM_STEP_LABEL_CALLER")) + " m="
            + String(label_m) + " n=" + String(label_n) + " k=" + String(label_k) + " arm="
            + name + " ksplit_default_takes=" + takes + " default_leaves_per_group="
            + String(label_leaves) + " choose_plan=["
            + gemm_plan_name(choose_gemm_plan(label_m, label_n, label_k)) + "] shipped_plan=["
            + gemm_shipped_dispatch_name(label_m, label_n, label_k) + "] arm_geometry=["
            + gemm_step_geometry_name(label_geom) + "]"
        )
    if String(getenv("MOJOLEARN_GEMM_STEP_LABEL_ONLY")) == "1":
        print("LABEL ONLY (MOJOLEARN_GEMM_STEP_LABEL_ONLY=1): no device work")
        return
    print("timing=host-synchronized median ms; raw samples below; the BITS pass is not a warmup")
    comptime if not GEMM_ARM_TRIAL:
        print("NOTE: no -D MOJOLEARN_GEMM_ARM_TRIAL=1: every arm runs the shipped default; this prices the shipped default against itself")
    var ctx = DeviceContext()
    var step_shipped = Float64(0.0)
    var step_arm = Float64(0.0)
    var priced = 0
    var controls_priced = 0
    var failures = 0
    print(
        "TABLE call | op | m x n x k | per step | shipped plan (ran) | arm geometry | shipped launched blocks"
        " | arm blocks | shipped median ms | arm median ms | ratio | arm launched blocks | group leaves"
        " | shipped group leaves"
    )
    for i in range(GEMM_STEP_LM_CALLS):
        var cname = gemm_step_lm_call_name(i)
        if not gemm_step_selected(calls, cname):
            continue
        var call = gemm_step_lm_call(i)
        var r = _price_call(ctx, arm, name, cname, call, i, rounds, warmups, False)
        failures += r[3]
        if r[2] == 1:
            step_shipped += Float64(call[4]) * r[0]
            step_arm += Float64(call[4]) * r[1]
            priced += 1
    if controls:
        for i in range(GEMM_STEP_CONTROL_CALLS):
            var cname2 = gemm_step_control_call_name(i)
            if not gemm_step_selected(calls, cname2):
                continue
            var r2 = _price_call(
                ctx, arm, name, cname2, gemm_step_control_call(i), 100 + i, rounds, warmups, True
            )
            failures += r2[3]
            controls_priced += r2[2]
    var step_ratio = Float64(0.0)
    if step_shipped > 0.0:
        step_ratio = step_arm / step_shipped
    print(
        "STEP gemm arm=" + name + " ratio=" + String(step_ratio) + " shipped_ms=" + String(step_shipped)
        + " arm_ms=" + String(step_arm) + " calls=" + String(priced) + " moved=" + String(failures)
        + " (per-call medians weighted by per-step counts; a GEMM sum, not a step time)"
    )
    if controls:
        print("CONTROLS gemm arm=" + name + " priced=" + String(controls_priced) + " (never weighted into STEP)")
    if failures != 0:
        raise Error("gemm_step_price: " + String(failures) + " calls MOVED BITS; an arm that changes the answer has no price")
    if priced == 0 and controls_priced == 0:
        raise Error("gemm_step_price: no call selected")
