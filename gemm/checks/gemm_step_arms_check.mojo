# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GEMM step arms gate (DEVIATIONS 2543, 2592 and 2595).

Every arm geometry of DEVIATIONS 2540, 2541, 2591 and 2595, forced through
`identical_gemm_step_geometry_into`, must store the SAME BITS as the old
shipped `PLAN_TUNED_128_8X8` and as `PLAN_FLAT`, and its sabotage
instantiation must move EXACTLY the cells `gemm_step_geometry_reach` names:
one per block for the 2540 and 2541 geometries (brief section 10.1 item 6),
one per launched `(tile, group)` whose sabotaged cell is in the output for
the ksplit geometries (docs/lanes/BRIEF_gemm_long_k_2026-09-11.md section
5.6), none for `tuned128` (the old plan has no sabotage), which proves the
arm kernel ran and names the geometry (and group count) that ran. The ragged
part also runs `identical_gemm_step_ksplit_into` at explicit group sizes
{1, 2, 4, 16, 64}. Two host checks run first and need no device work:
`check_group_fold_is_the_contract_tree` (the long-k brief's Lemmas A to C,
every `P` in 1 to 1,100, every power-of-two group size up to at least `2 P`)
and `check_group_rule_hand_counts` (the section 4 rule at `S = 132` against
the brief's hand counts at the twelve LM calls, and the shipped default's
rule at row 132 and at row 0). Then `check_default_dispatch` (DEVIATION
2595, below). Then the twelve LM calls of the step at the target shape
(brief section 2, the three vocab-sized head calls among them) go through
`identical_gemm_into`, the entry every GEMM of the step reaches, under
`MOJOLEARN_GEMM_ARM` set for each arm in turn: bits equal to the old plan,
and reach per call (the arm's reach where it applies, the shipped default's
reach where it does not).

DEVIATION 2595 (that brief's section 10): `shipped` is the shipped DEFAULT,
ksplit where the column's `lib_gemm_block_parallelism_for` row is above 0
and the group rule takes the call, else the plan `choose_gemm_plan` picks.
`check_default_dispatch` holds both sides of that switch by reach at every
column: the shipped dispatch's body at the column's own row, at row 0 (the
old plan runs: nothing moves) and at row 132 (the group launch's cells move),
and the shipped entry itself with `MOJOLEARN_GEMM_ARM` unset.

DEVIATION 2599 (docs/lanes/BRIEF_gemm_kernel_2026-09-11.md sections 5 and
6): the arms `kpack` and `kpack_wide` are geometries 10 and 11, so the
ragged part forces both at every case (bits equal to the old plan and FLAT,
reach one cell per tile in their all-leaves launch and one per `(tile, q)` in
their group launch, `gemm_step_kpack_reach`) and the LM part sends the twelve
calls through `identical_gemm_into` under both. Two host checks run first:
`check_kpack_page_is_a_bijection` (every staged `(line, step)` of every
thread, slot, mapping, operand and geometry lands at exactly one packed
address, and every read address holds the pair the reader expects) and
`check_kpack_rule_hand_counts` (the tile-parameterized rule equals
`gemm_step_ksplit_rule` at 128x128, and the 128x256 counts match brief
section 4.2). `check_group_fold_is_the_contract_tree` also requires the
group nodes built with `kpack_wide`'s 12-level stack to equal the 16-level
ones bit for bit.

DEVIATIONS 2640 to 2642 (docs/lanes/BRIEF_gemm_final_2026-09-11.md sections 4
to 6): the arms `kfoldv` and `kfoldv_leaf` are geometries 12 and 13, so the
ragged part forces both at every case (bits equal to the old plan and FLAT,
reach `gemm_step_kfold_reach`: the group launch's cells plus one cell per lane
fold block, minus the cells in both) and the LM part sends the twelve calls
through `identical_gemm_into` under both. Two host checks run first:
`check_kfold_lanes_is_the_stack_fold` (the lane-wise `_fold_push_lanes` and
`_fold_drain_lanes` store, in every lane, the bits `_fold_push`,
`_fold_drain` and the stored `ftz` give that lane's cell alone, for every group
count 1 to 255, and the 8-level stack overflows exactly at the 256th push) and
`check_kfold_rule_hand_counts` (the two arms' rules, group counts and fold
block counts at the twelve LM calls against brief section 3.3's hand counts).

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . \\
        gemm/checks/gemm_step_arms_check.mojo -o <bin>
    MOJOLEARN_GEMM_STEP_CHECK_LM=0 <bin>   # a light box: ragged controls and selector only
    <bin>                                  # a GPU box: plus the twelve LM calls

Knobs: `MOJOLEARN_GEMM_STEP_CHECK_LM=0` skips the LM section (and says so);
`MOJOLEARN_GEMM_STEP_CHECK_ARMS=<comma list>` restricts the LM section's
arms; `MOJOLEARN_GEMM_STEP_CHECK_FLOPS` (default 400,000,000) caps
`m n k` of a ragged or default control.

Without `-D MOJOLEARN_GEMM_ARM_TRIAL=1` the arm kernels and every sabotage
instantiation are not compiled, so this check LAUNCHES NONE OF THEM: no
forced arm geometry but `tuned128`, no explicit group size, no sabotage run
at any default row or through the entry, no arm through the LM entry. It
FAILS, one line per geometry, per group size, per default row, for the
shipped entry and per LM arm, each naming the missing define, and it still
checks what the shipped build can: the two host checks, the old plan against
FLAT, `tuned128` forced, and the shipped dispatch's CLEAN bits at every
default row and through the entry.

Why it launches nothing it cannot reach (the no-trial crash on the Apple M4,
2026-09-11, brief section 11). Since DEVIATION 2595 the non-trial fallback
of `identical_gemm_step_geometry_into` and `identical_gemm_step_ksplit_into`
is the shipped dispatch, which at a ragged shape with `P >= 4` picks a SPLIT
plan (`choose_gemm_plan`: 1x3x1000 SPLIT_16_1X1, 33x70x1000 SPLIT_32_2X2,
129x257x1000 SPLIT_64_4X4) that writes `m n P` floats of workspace. The
ragged part sizes `dw` for FLAT and the old 128x128 plan, which need none,
so `dw` held ONE float and the SPLIT leaf kernel wrote up to 1.2 MB past it.
Before 2595 the fallback was PLAN_TUNED_128_8X8 by name, which needs no
workspace, so the same run only failed on reach.

It refuses a build that defines one of `gemm_identical.mojo`'s global
sabotage switches, because its baseline would be a sabotaged kernel.

ENGINEERING_RULES 8: the switch is exercised on both sides by name. The
`shipped` arm through the entry must equal the explicit old plan and its
sabotage must move exactly the default's reach (nothing on a column whose
row is 0); `tuned128` must equal it and move nothing; every other arm must
move its reach where it applies.
"""
from std.memory import bitcast, stack_allocation
from std.os import getenv, setenv
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import ftz, numeric_mode_name
from gemm.checks.gemm_identical import (
    ANY_SABOTAGE,
    GEMM_ARM_SHIPPED,
    GEMM_ARM_TRIAL,
    GEMM_FOLD_SLOTS,
    GEMM_GEOM_COUNT,
    GEMM_GEOM_KFOLDV,
    GEMM_GEOM_KFOLDV_LEAF,
    GEMM_GEOM_KPACK,
    GEMM_GEOM_KPACK_WIDE,
    GEMM_GEOM_SHIPPED,
    GEMM_GEOM_TUNED128,
    GEMM_KFOLD_FS,
    GEMM_KFOLD_MAX_GROUPS,
    GEMM_KFOLD_TPB,
    GEMM_KFOLD_VB,
    GEMM_KFOLD_W,
    GEMM_KPACKW_BM,
    GEMM_KPACKW_BN,
    GEMM_KPACKW_CPT,
    GEMM_KPACKW_FS,
    GEMM_KPACKW_KS,
    GEMM_KPACKW_RPT,
    GEMM_KPACK_CPT,
    GEMM_KPACK_KS,
    GEMM_KPACK_RPT,
    GEMM_KSPLIT_DEFAULT_S,
    PLAN_FLAT,
    PLAN_TUNED_128_8X8,
    TUNED_FOLD_SLOTS,
    TUNED_TC,
    TUNED_TPB,
    TUNED_VECLEN,
    _fold_drain,
    _fold_drain_lanes,
    _fold_drain_local,
    _fold_push,
    _fold_push_lanes,
    _fold_push_local,
    choose_gemm_plan,
    contract_partition,
    gemm_kfold_blocks,
    gemm_default_ksplit_leaves,
    gemm_default_ksplit_leaves_at,
    gemm_kpack_addr,
    gemm_kpack_register_slots,
    gemm_kpack_stage_outer,
    gemm_kpack_stage_p,
    gemm_plan_name,
    gemm_sabotage_name,
    gemm_shipped_dispatch_name,
    gemm_shipped_dispatch_name_at,
    gemm_step_arm_geometry,
    gemm_step_arm_name,
    gemm_step_arm_parse,
    gemm_step_arm_plan_label,
    gemm_step_geometry_group_leaves,
    gemm_step_geometry_launched_blocks,
    gemm_step_geometry_name,
    gemm_step_geometry_reach,
    gemm_step_kfold_leaves,
    gemm_step_kfold_rule,
    gemm_step_kpack_rule,
    gemm_step_ksplit_reach,
    gemm_step_ksplit_rule,
    identical_gemm_into,
    identical_gemm_shipped_at_row_into,
    identical_gemm_shipped_into,
    identical_gemm_step_geometry_into,
    identical_gemm_step_ksplit_into,
    identical_gemm_with_plan,
    identical_gemm_workspace_floats,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN, fold_balanced_tree, op_name
from gemm.checks.gemm_step_arms import (
    GEMM_STEP_LM_CALLS,
    _value,
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
#: `which` is the leaves per group (DEVIATION 2592).
comptime RUN_KSPLIT = 3
#: `which` is the block parallelism row (DEVIATION 2595): the shipped
#: dispatch's body, `identical_gemm_shipped_at_row_into`.
comptime RUN_ROW = 4
#: DEVIATION 2595: `identical_gemm_shipped_into`, the column's own shipped
#: dispatch, with no environment read.
comptime RUN_SHIPPED = 5

#: Cells per fold in the host group-fold check (the device check's 4).
comptime GROUP_NC = 4

#: DEVIATION 2595: an ENABLED block parallelism row, held on every column
#: (the H100 row the flip was measured at).
comptime DEFAULT_ROW_ON = 132


def _trial_hint() -> String:
    comptime if GEMM_ARM_TRIAL:
        return String("")
    return String(
        " (this build lacks -D MOJOLEARN_GEMM_ARM_TRIAL=1: the arm kernels and the sabotage"
        " instantiations are not compiled, so nothing was launched for them)"
    )


def _arm_names() -> List[String]:
    var names: List[String] = [
        "shipped", "lfold", "half", "half_ks16", "quarter", "head", "half_head",
        "ksplit", "ksplit_leaf", "tuned128", "kpack", "kpack_wide",
        # DEVIATIONS 2640 and 2641 (docs/lanes/BRIEF_gemm_final_2026-09-11.md):
        # geometries 12 and 13, forced in the ragged part by GEMM_GEOM_COUNT.
        "kfoldv", "kfoldv_leaf"
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
    elif how == RUN_KSPLIT:
        identical_gemm_step_ksplit_into(
            ctx, dc, da, db, dw, m, n, k, op, which, sabotage
        )
    elif how == RUN_ROW:
        # The sabotage instantiation exists only on a trial build; without
        # the define the run is clean and the reach check fails, saying so.
        if sabotage:
            comptime if GEMM_ARM_TRIAL:
                identical_gemm_shipped_at_row_into[True](
                    ctx, dc, da, db, dw, m, n, k, op, which
                )
            else:
                identical_gemm_shipped_at_row_into[False](
                    ctx, dc, da, db, dw, m, n, k, op, which
                )
        else:
            identical_gemm_shipped_at_row_into[False](
                ctx, dc, da, db, dw, m, n, k, op, which
            )
    elif how == RUN_SHIPPED:
        identical_gemm_shipped_into(ctx, dc, da, db, dw, m, n, k, op)
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
# HOST: THE GROUP FOLD AND THE GROUP RULE (DEVIATION 2592)
# ===========================================================================


def _bits32(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _group_node[
    FS: Int
](parts: List[Float32], lbeg: Int, lend: Int) raises -> SIMD[DType.float32, GROUP_NC]:
    """One group's stored node, per cell: leaves `[lbeg, lend)` pushed into a
    FRESH stack of `FS` levels with the device's own `_fold_push_local`, then
    the device's own `_fold_drain_local`. `parts[t * GROUP_NC + e]` is leaf
    `t` of cell `e`. `FS` is `TUNED_FOLD_SLOTS` (the shipped kernels) or
    `GEMM_KPACKW_FS` (`kpack_wide`, DEVIATION 2599)."""
    var stack = stack_allocation[FS * GROUP_NC, Scalar[DType.float32]]()
    var occ = 0
    for t in range(lbeg, lend):
        var v = SIMD[DType.float32, GROUP_NC](0.0)
        for e in range(GROUP_NC):
            v[e] = parts[t * GROUP_NC + e]
        if not _fold_push_local[GROUP_NC, FS](stack, occ, v):
            raise Error(
                "_group_node: the thread-local fold stack of " + String(FS)
                + " levels OVERFLOWED at leaf " + String(t)
            )
    return _fold_drain_local[GROUP_NC, FS](stack, occ)


def check_group_fold_is_the_contract_tree(mut failures: List[String]) raises:
    """Lemmas A to C of docs/lanes/BRIEF_gemm_long_k_2026-09-11.md section 5,
    exhaustively on the host, before any device run.

    For every `P` in 1 to 1,100 and every power-of-two group size `gl` from 1
    up to the first at or above `2 P`: the group nodes are built as the group
    kernel builds them (`_group_node`), then folded two ways, as
    `identical_gemm_fold_stack_kernel` folds them (`_fold_push`,
    `_fold_drain`, then the stored `ftz`) and as
    `identical_gemm_fold_kernel[True]` folds them (`fold_balanced_tree`, the
    level-wise tree). Both must equal `fold_balanced_tree` over all `P` leaf
    partials, bit for bit, in every cell. Two partial kinds: the 13-bit
    significand generator (every addition inexact), and the same with about
    one partial in five replaced by `-0.0` (the carry and signed-zero seams).
    """
    var bad = 0
    var runs = 0
    var first = String("")
    # DEVIATION 2599: the same group nodes from `kpack_wide`'s shorter stack.
    var fs_bad = 0
    var fs_runs = 0
    var fs_first = String("")
    for kind in range(2):
        for p in range(1, 1101):
            var parts = List[Float32]()
            for t in range(p):
                for e in range(GROUP_NC):
                    var v = _value(t * 31 + p + 7 * e, 909 + e + 17 * kind)
                    if kind == 1 and (t * 7 + e * 3 + p) % 5 == 0:
                        v = -Float32(0.0)
                    parts.append(v)
            var want = List[Float32]()
            for e in range(GROUP_NC):
                var col = List[Float32]()
                for t in range(p):
                    col.append(parts[t * GROUP_NC + e])
                want.append(fold_balanced_tree(col))
            var gl = 1
            while True:
                var groups = (p + gl - 1) // gl
                var nodes = List[Float32]()
                for q in range(groups):
                    var lend = (q + 1) * gl
                    if lend > p:
                        lend = p
                    var gv = _group_node[TUNED_FOLD_SLOTS](parts, q * gl, lend)
                    var gv12 = _group_node[GEMM_KPACKW_FS](parts, q * gl, lend)
                    for e in range(GROUP_NC):
                        nodes.append(gv[e])
                        fs_runs += 1
                        if _bits32(gv12[e]) != _bits32(gv[e]):
                            fs_bad += 1
                            if fs_first.byte_length() == 0:
                                fs_first = (
                                    "kind=" + String(kind) + " P=" + String(p) + " group="
                                    + String(gl) + " q=" + String(q) + " cell=" + String(e)
                                    + ": FS " + String(GEMM_KPACKW_FS) + " node="
                                    + hex(_bits32(gv12[e])) + " FS " + String(TUNED_FOLD_SLOTS)
                                    + " node=" + hex(_bits32(gv[e]))
                                )
                for e in range(GROUP_NC):
                    var col2 = List[Float32]()
                    var stack = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
                    var occ = 0
                    for q2 in range(groups):
                        var nv = nodes[q2 * GROUP_NC + e]
                        col2.append(nv)
                        if not _fold_push(stack, occ, nv):
                            raise Error(
                                "check_group_fold_is_the_contract_tree: the register stack"
                                " OVERFLOWED at group " + String(q2) + " of " + String(groups)
                            )
                    var got_stack = ftz(_fold_drain(stack, occ))
                    var got_tree = fold_balanced_tree(col2)
                    runs += 1
                    var want_bits = _bits32(want[e])
                    if _bits32(got_stack) != want_bits or _bits32(got_tree) != want_bits:
                        bad += 1
                        if first.byte_length() == 0:
                            first = (
                                "kind=" + String(kind) + " P=" + String(p) + " group="
                                + String(gl) + " cell=" + String(e) + ": stack="
                                + hex(_bits32(got_stack)) + " tree=" + hex(_bits32(got_tree))
                                + " contract=" + hex(want_bits)
                            )
                if gl >= 2 * p:
                    break
                gl = gl * 2
    if bad != 0:
        failures.append(
            "check_group_fold_is_the_contract_tree: " + String(bad) + " of " + String(runs)
            + " (kind, P, group, cell) folds DISAGREE with fold_balanced_tree; first " + first
        )
    if fs_bad != 0:
        failures.append(
            "check_group_fold_is_the_contract_tree: " + String(fs_bad) + " of " + String(fs_runs)
            + " group nodes built with the " + String(GEMM_KPACKW_FS) + "-level stack DIFFER from the "
            + String(TUNED_FOLD_SLOTS) + "-level ones; first " + fs_first
        )
    print(
        "check_group_fold_is_the_contract_tree: " + String(runs) + " folds (2 kinds, P 1..1100,"
        " every power-of-two group up to 2P, 4 cells), " + String(bad) + " disagree; "
        + String(fs_runs) + " group nodes at FS " + String(GEMM_KPACKW_FS) + " against FS "
        + String(TUNED_FOLD_SLOTS) + ", " + String(fs_bad) + " differ"
    )


# ===========================================================================
# HOST: THE PACKED PAGE AND THE KPACK RULE (DEVIATION 2599)
# ===========================================================================


def _kpack_page_case(
    label: String,
    lines: Int,
    group_lines: Int,
    per_thread: Int,
    ks: Int,
    outer: Bool,
    by_row: Bool,
    mut failures: List[String],
) raises:
    """One operand of one geometry under one staging mapping, all 256
    threads. Brief section 5.2, checked rather than argued:

    1. The helper the kernel's staging stores call
       (`gemm_kpack_stage_p` or `gemm_kpack_stage_outer`) names the same
       `(line, step, staged)` as `_tuned_g2r`'s own body, transcribed here.
    2. Every staged pair lands at an address in `[0, lines KS)`, and every
       address receives exactly one store (a bijection, no address shared).
    3. Every address a thread reads (its group `g`, step `c`, element `u`)
       holds line `g + u group_lines` at step `c`, and is `gemm_kpack_addr`
       of that pair.

    `by_row` selects the A reader (`g = tid // TC`) or the B reader
    (`g = tid mod TC`)."""
    comptime NTH = TUNED_TPB
    comptime VEC = TUNED_VECLEN
    var tag = String("check_kpack_page_is_a_bijection [") + label + "]"
    if lines != per_thread * group_lines or ks % VEC != 0:
        failures.append(
            tag + ": " + String(lines) + " lines is not " + String(per_thread) + " x "
            + String(group_lines) + ", or KS " + String(ks) + " is not a VEC multiple"
        )
        return
    var kv = ks // VEC
    var slots = (lines * kv + NTH - 1) // NTH
    # The kernel instantiates `_tuned_g2r` with this padded count, because a
    # SIMD width must be a power of two (brief section 11). Every padded slot
    # must stage nothing under either mapping.
    var reg_slots = gemm_kpack_register_slots(slots)
    var reg_width = reg_slots * VEC
    if reg_slots < slots or reg_width <= 0 or (reg_width & (reg_width - 1)) != 0:
        failures.append(
            tag + ": " + String(slots) + " staging slots pad to " + String(reg_slots)
            + ", register width " + String(reg_width) + " is not a power of two at or above "
            + String(slots * VEC)
        )
        return
    var total = lines * ks
    var hits = List[Int]()
    var at_line = List[Int]()
    var at_step = List[Int]()
    for _ in range(total):
        hits.append(0)
        at_line.append(-1)
        at_step.append(-1)
    var bad = 0
    var first = String("")
    for tid in range(NTH):
        for s in range(reg_slots):
            for e in range(VEC):
                var want_ok = False
                var want_line = 0
                var want_step = 0
                var sp = gemm_kpack_stage_p(tid, s, e, lines, kv, VEC, NTH)
                if outer:
                    sp = gemm_kpack_stage_outer(tid, s, e, lines, ks, VEC, NTH)
                    var idx0 = tid + (s * VEC + e) * NTH
                    want_ok = idx0 < lines * kv * VEC
                    want_line = idx0 % lines
                    want_step = idx0 // lines
                else:
                    var idx = tid + s * NTH
                    want_ok = idx < lines * kv
                    want_line = idx // kv
                    want_step = (idx - want_line * kv) * VEC + e
                if s >= slots:
                    # A padded register slot: `_tuned_g2r` must leave it
                    # `+0.0` and the kernel's stores never visit it.
                    if want_ok or sp[2]:
                        bad += 1
                        if first.byte_length() == 0:
                            first = (
                                "tid " + String(tid) + " padded slot (" + String(s) + ", "
                                + String(e) + ") past " + String(slots) + " would stage line "
                                + String(want_line) + " step " + String(want_step)
                            )
                    continue
                if sp[2] != want_ok or (want_ok and (sp[0] != want_line or sp[1] != want_step)):
                    bad += 1
                    if first.byte_length() == 0:
                        first = (
                            "tid " + String(tid) + " slot (" + String(s) + ", " + String(e)
                            + ") staged as line " + String(sp[0]) + " step " + String(sp[1])
                            + " where _tuned_g2r holds line " + String(want_line) + " step "
                            + String(want_step)
                        )
                    continue
                if not want_ok:
                    continue
                var ad = gemm_kpack_addr(want_line, want_step, group_lines, per_thread, ks)
                if ad < 0 or ad >= total:
                    bad += 1
                    if first.byte_length() == 0:
                        first = (
                            "line " + String(want_line) + " step " + String(want_step)
                            + " packs to address " + String(ad) + " outside [0, " + String(total) + ")"
                        )
                    continue
                hits[ad] += 1
                at_line[ad] = want_line
                at_step[ad] = want_step
    for ad2 in range(total):
        if hits[ad2] != 1:
            bad += 1
            if first.byte_length() == 0:
                first = "address " + String(ad2) + " received " + String(hits[ad2]) + " stores"
    for tid2 in range(NTH):
        var g = tid2 % TUNED_TC
        if by_row:
            g = tid2 // TUNED_TC
        for c0 in range(ks):
            for u in range(per_thread):
                var ad3 = g * ks * per_thread + c0 * per_thread + u
                var line = g + u * group_lines
                if (
                    ad3 >= total
                    or at_line[ad3] != line
                    or at_step[ad3] != c0
                    or gemm_kpack_addr(line, c0, group_lines, per_thread, ks) != ad3
                ):
                    bad += 1
                    if first.byte_length() == 0:
                        first = (
                            "tid " + String(tid2) + " reads line " + String(line) + " step "
                            + String(c0) + " at address " + String(ad3)
                        )
    if bad != 0:
        failures.append(tag + ": " + String(bad) + " disagreements; first " + first)
    print(
        tag + " lines=" + String(lines) + " group_lines=" + String(group_lines) + " per_thread="
        + String(per_thread) + " KS=" + String(ks) + " slots=" + String(slots) + " register_slots=" + String(reg_slots) + " addresses="
        + String(total) + " disagreements=" + String(bad)
    )


def check_kpack_page_is_a_bijection(mut failures: List[String]) raises:
    """`_kpack_page_case` for `kpack` (A and B, K step 16) and `kpack_wide`
    (A and B at K steps 12 and 16, the two values its matrix read can take),
    under both staging mappings. Host only."""
    comptime TR = TUNED_TPB // TUNED_TC
    var before = len(failures)
    var kss: List[Int] = [12, 16]
    for oi in range(2):
        var outer = oi == 1
        var mapping = String("outer") if outer else String("p")
        _kpack_page_case(
            String("kpack A ") + mapping, GEMM_KPACK_RPT * TR, TR, GEMM_KPACK_RPT,
            GEMM_KPACK_KS, outer, True, failures,
        )
        _kpack_page_case(
            String("kpack B ") + mapping, GEMM_KPACK_CPT * TUNED_TC, TUNED_TC, GEMM_KPACK_CPT,
            GEMM_KPACK_KS, outer, False, failures,
        )
        for si in range(len(kss)):
            _kpack_page_case(
                String("kpack_wide A KS=") + String(kss[si]) + " " + mapping, GEMM_KPACKW_BM, TR,
                GEMM_KPACKW_RPT, kss[si], outer, True, failures,
            )
            _kpack_page_case(
                String("kpack_wide B KS=") + String(kss[si]) + " " + mapping, GEMM_KPACKW_BN,
                TUNED_TC, GEMM_KPACKW_CPT, kss[si], outer, False, failures,
            )
    print(
        "check_kpack_page_is_a_bijection: 12 cases (2 geometries, 2 operands, 2 mappings, kpack_wide"
        " at KS 12 and 16; this column's kpack_wide KS=" + String(GEMM_KPACKW_KS) + "), "
        + String(len(failures) - before) + " failures"
    )


def check_kpack_rule_hand_counts(mut failures: List[String]) raises:
    """`gemm_step_kpack_rule` at the 128x128 tile must equal
    `gemm_step_ksplit_rule` at the twelve LM calls, at `S = 132` and at no
    reading; on the 128x256 tile at `S = 132` it must give brief section
    4.2's hand counts. Host only."""
    comptime TR = TUNED_TPB // TUNED_TC
    var want_wide: List[Int] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 32, 0]
    var bm = GEMM_KPACK_RPT * TR
    var bn = GEMM_KPACK_CPT * TUNED_TC
    var before = len(failures)
    for i in range(GEMM_STEP_LM_CALLS):
        var call = gemm_step_lm_call(i)
        var m = call[1]
        var n = call[2]
        var k = call[3]
        var p132 = gemm_step_kpack_rule(m, n, k, 132, bm, bn)
        var k132 = gemm_step_ksplit_rule(m, n, k, 132, True)
        var p0 = gemm_step_kpack_rule(m, n, k, 0, bm, bn)
        var k0 = gemm_step_ksplit_rule(m, n, k, 0, True)
        var w132 = gemm_step_kpack_rule(m, n, k, 132, GEMM_KPACKW_BM, GEMM_KPACKW_BN)
        print(
            "RULE_KPACK " + gemm_step_lm_call_name(i) + " " + op_name(call[0]) + " " + String(m) + "x"
            + String(n) + "x" + String(k) + " kpack(S=132)=" + String(p132) + " ksplit(S=132)="
            + String(k132) + " kpack(S=0)=" + String(p0) + " ksplit(S=0)=" + String(k0)
            + " kpack_wide(S=132)=" + String(w132) + " (brief " + String(want_wide[i]) + ")"
        )
        if p132 != k132 or p0 != k0:
            failures.append(
                "RULE_KPACK " + gemm_step_lm_call_name(i) + ": the 128x128 rule gives " + String(p132)
                + " and " + String(p0) + " where gemm_step_ksplit_rule gives " + String(k132) + " and "
                + String(k0)
            )
        if w132 != want_wide[i]:
            failures.append(
                "RULE_KPACK " + gemm_step_lm_call_name(i) + ": kpack_wide(S=132)=" + String(w132)
                + " (brief " + String(want_wide[i]) + ")"
            )
    print("check_kpack_rule_hand_counts: " + String(len(failures) - before) + " failures")


# ===========================================================================
# HOST: THE LANE FOLD AND ITS RULES (DEVIATIONS 2640 to 2642)
# ===========================================================================


def check_kfold_lanes_is_the_stack_fold(mut failures: List[String]) raises:
    """docs/lanes/BRIEF_gemm_final_2026-09-11.md sections 5.3 to 5.5, on the
    host, before any device work.

    For every group count `G` in 1 to `GEMM_KFOLD_MAX_GROUPS` (255) and three
    node kinds (the 13-bit significand generator; the same with about one node
    in four replaced by `-0.0`; the same with about one in three scaled by
    `1e-38`, so subnormal nodes and sums exercise every `ftz` seam), `W` lanes
    of distinct nodes are pushed with the device's own `_fold_push_lanes` in
    `q` order (the kernel's load batches change when a node loads, never the
    push order) and drained with `_fold_drain_lanes`. In every lane the stored
    `ftz(root)` must equal, bit for bit, what the shipped fold stack kernel
    stores for that lane's cell alone (`_fold_push` into its 12-level register
    stack, `_fold_drain`, `ftz`). For the first two kinds it must also equal
    `fold_balanced_tree` over the lane's nodes. No push may overflow below 256
    groups, `occ` must end at `G`, and after 255 pushes the 256th must overflow
    (section 5.5 both ways)."""
    comptime W = GEMM_KFOLD_W
    comptime FS = GEMM_KFOLD_FS
    var bad = 0
    var runs = 0
    var first = String("")
    for kind in range(3):
        for g in range(1, GEMM_KFOLD_MAX_GROUPS + 1):
            var nodes = List[Float32]()
            for q in range(g):
                for e in range(W):
                    var v = _value(q * 131 + g * 7 + e, 1301 + e + 29 * kind)
                    if kind == 1 and (q * 5 + e * 3 + g) % 4 == 0:
                        v = -Float32(0.0)
                    if kind == 2 and (q + e + g) % 3 == 0:
                        v = v * Float32(1.0e-38)
                    nodes.append(v)
            var stack = SIMD[DType.float32, FS * W](0.0)
            var occ = 0
            var overflow_at = -1
            for q2 in range(g):
                var val = SIMD[DType.float32, W](0.0)
                for e2 in range(W):
                    val[e2] = nodes[q2 * W + e2]
                if not _fold_push_lanes[W, FS](stack, occ, val):
                    if overflow_at < 0:
                        overflow_at = q2
            if overflow_at >= 0 or occ != g:
                bad += 1
                if first.byte_length() == 0:
                    first = (
                        "kind=" + String(kind) + " G=" + String(g) + ": lane stack overflowed at push "
                        + String(overflow_at) + ", occ=" + String(occ)
                    )
            var root = _fold_drain_lanes[W, FS](stack, occ)
            for e3 in range(W):
                var sstack = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
                var socc = 0
                var col = List[Float32]()
                for q3 in range(g):
                    var nv = nodes[q3 * W + e3]
                    col.append(nv)
                    if not _fold_push(sstack, socc, nv):
                        raise Error(
                            "check_kfold_lanes_is_the_stack_fold: the 12-level register stack"
                            " OVERFLOWED at group " + String(q3)
                        )
                var want_stack = ftz(_fold_drain(sstack, socc))
                var got = ftz(root[e3])
                var ok = _bits32(got) == _bits32(want_stack)
                var tree_bits = UInt32(0)
                if kind < 2:
                    tree_bits = _bits32(fold_balanced_tree(col))
                    ok = ok and _bits32(got) == tree_bits
                runs += 1
                if not ok:
                    bad += 1
                    if first.byte_length() == 0:
                        first = (
                            "kind=" + String(kind) + " G=" + String(g) + " lane=" + String(e3)
                            + ": lanes=" + hex(_bits32(got)) + " stack=" + hex(_bits32(want_stack))
                            + " tree=" + hex(tree_bits) + " (tree compared for kinds 0 and 1)"
                        )
    # Section 5.5 the other way: 255 pushes fit, the 256th overflows.
    var st2 = SIMD[DType.float32, FS * W](0.0)
    var occ2 = 0
    var fit = True
    for q4 in range(GEMM_KFOLD_MAX_GROUPS):
        if not _fold_push_lanes[W, FS](st2, occ2, SIMD[DType.float32, W](Float32(q4 % 7) + 1.0)):
            fit = False
    var over = _fold_push_lanes[W, FS](st2, occ2, SIMD[DType.float32, W](1.0))
    if not fit or over:
        failures.append(
            "check_kfold_lanes_is_the_stack_fold: " + String(GEMM_KFOLD_MAX_GROUPS)
            + " pushes fit=" + String(fit) + ", push " + String(GEMM_KFOLD_MAX_GROUPS + 1)
            + " placed=" + String(over) + " (must be True and False)"
        )
    if bad != 0:
        failures.append(
            "check_kfold_lanes_is_the_stack_fold: " + String(bad) + " of " + String(runs)
            + " (kind, G, lane) folds DISAGREE; first " + first
        )
    print(
        "check_kfold_lanes_is_the_stack_fold: " + String(runs) + " lane folds (3 kinds, G 1.."
        + String(GEMM_KFOLD_MAX_GROUPS) + ", W=" + String(W) + ", FS=" + String(FS) + "), "
        + String(bad) + " disagree; overflow bound fit=" + String(fit) + " next_placed=" + String(over)
    )


def check_kfold_rule_hand_counts(mut failures: List[String]) raises:
    """Brief sections 3.3, 4.2 and 4.3 at the twelve LM calls, host only, on
    every column (`gemm_step_kfold_rule` takes `S` as an argument): `kfoldv`
    at `S = 132` equals `gemm_step_ksplit_rule(.., 132, True)` and the ksplit
    hand counts; `kfoldv_leaf` equals `gemm_step_ksplit_rule(.., 0, False)`
    and the ksplit_leaf hand counts; the group counts match the brief and
    never exceed `GEMM_KFOLD_MAX_GROUPS` where an arm takes the call (so the
    lane fold, not the shipped fold, runs there); `gemm_kfold_blocks` matches
    the hand count of fold blocks (`ceil(ceil(m n / 16) / 256)`). The column's
    own answer (`gemm_step_kfold_leaves`, at `GEMM_KSPLIT_S`) is printed."""
    var want_v: List[Int] = [1, 1, 1, 0, 2, 2, 2, 0, 2, 0, 64, 0]
    var want_leaf: List[Int] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 16, 0]
    var want_groups_v: List[Int] = [6, 6, 16, 0, 8, 8, 8, 0, 8, 0, 7, 0]
    var want_groups_leaf: List[Int] = [6, 6, 16, 6, 16, 16, 16, 6, 16, 0, 25, 0]
    var want_blocks: List[Int] = [384, 384, 144, 1024, 384, 384, 384, 1024, 384, 25129, 384, 9424]
    var before = len(failures)
    for i in range(GEMM_STEP_LM_CALLS):
        var call = gemm_step_lm_call(i)
        var m = call[1]
        var n = call[2]
        var k = call[3]
        var cname = gemm_step_lm_call_name(i)
        var p_count = contract_partition(k)[1]
        var gv = gemm_step_kfold_rule(GEMM_GEOM_KFOLDV, m, n, k, 132)
        var gf = gemm_step_kfold_rule(GEMM_GEOM_KFOLDV_LEAF, m, n, k, 132)
        var kv = gemm_step_ksplit_rule(m, n, k, 132, True)
        var kf = gemm_step_ksplit_rule(m, n, k, 0, False)
        var groups_v = 0
        if gv > 0:
            groups_v = (p_count + gv - 1) // gv
        var groups_f = 0
        if gf > 0:
            groups_f = (p_count + gf - 1) // gf
        var blocks = gemm_kfold_blocks(m * n)
        var col_v = gemm_step_kfold_leaves(GEMM_GEOM_KFOLDV, m, n, k)
        var col_f = gemm_step_kfold_leaves(GEMM_GEOM_KFOLDV_LEAF, m, n, k)
        print(
            "RULE_KFOLD " + cname + " " + op_name(call[0]) + " " + String(m) + "x" + String(n) + "x"
            + String(k) + " kfoldv(S=132)=" + String(gv) + " groups=" + String(groups_v)
            + " kfoldv_leaf=" + String(gf) + " groups=" + String(groups_f) + " fold_blocks="
            + String(blocks) + " (W=" + String(GEMM_KFOLD_W) + " tpb=" + String(GEMM_KFOLD_TPB)
            + " VB=" + String(GEMM_KFOLD_VB) + ") column: kfoldv=" + String(col_v) + " kfoldv_leaf="
            + String(col_f)
        )
        if gv != kv or gf != kf:
            failures.append(
                "RULE_KFOLD " + cname + ": kfoldv=" + String(gv) + " (ksplit rule " + String(kv)
                + "), kfoldv_leaf=" + String(gf) + " (ksplit_leaf rule " + String(kf) + ")"
            )
        if gv != want_v[i] or gf != want_leaf[i]:
            failures.append(
                "RULE_KFOLD " + cname + ": kfoldv=" + String(gv) + " (brief " + String(want_v[i])
                + "), kfoldv_leaf=" + String(gf) + " (brief " + String(want_leaf[i]) + ")"
            )
        if groups_v != want_groups_v[i] or groups_f != want_groups_leaf[i]:
            failures.append(
                "RULE_KFOLD " + cname + ": groups kfoldv=" + String(groups_v) + " (brief "
                + String(want_groups_v[i]) + "), kfoldv_leaf=" + String(groups_f) + " (brief "
                + String(want_groups_leaf[i]) + ")"
            )
        if groups_v > GEMM_KFOLD_MAX_GROUPS or groups_f > GEMM_KFOLD_MAX_GROUPS:
            failures.append(
                "RULE_KFOLD " + cname + ": a taken call has more than " + String(GEMM_KFOLD_MAX_GROUPS)
                + " groups, so the shipped fold would run there, not the lane fold"
            )
        if blocks != want_blocks[i]:
            failures.append(
                "RULE_KFOLD " + cname + ": fold blocks " + String(blocks) + " (brief "
                + String(want_blocks[i]) + ")"
            )
    print("check_kfold_rule_hand_counts: " + String(len(failures) - before) + " failures")


def check_group_rule_hand_counts(mut failures: List[String]) raises:
    """The section 4 group rule, at the NVIDIA reading `S = 132` and with no
    reading, against the long-k brief's hand counts at the twelve LM calls
    (leaves per group; 0 declines). Host only: `gemm_step_ksplit_rule` takes
    `S` as an argument, so this holds on every column.

    DEVIATION 2595: the shipped default's rule
    (`gemm_default_ksplit_leaves_at`) must give the `ksplit` hand counts at
    row 132 and 0 at every call at row 0; the column's own answer is
    printed beside them."""
    var want_ksplit: List[Int] = [1, 1, 1, 0, 2, 2, 2, 0, 2, 0, 64, 0]
    var want_leaf: List[Int] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 16, 0]
    var before = len(failures)
    for i in range(GEMM_STEP_LM_CALLS):
        var call = gemm_step_lm_call(i)
        var m = call[1]
        var n = call[2]
        var k = call[3]
        var gk = gemm_step_ksplit_rule(m, n, k, 132, True)
        var gf = gemm_step_ksplit_rule(m, n, k, 0, False)
        var gd_on = gemm_default_ksplit_leaves_at(m, n, k, DEFAULT_ROW_ON)
        var gd_off = gemm_default_ksplit_leaves_at(m, n, k, 0)
        var gd_col = gemm_default_ksplit_leaves(m, n, k)
        print(
            "RULE " + gemm_step_lm_call_name(i) + " " + op_name(call[0]) + " " + String(m) + "x"
            + String(n) + "x" + String(k) + " ksplit(S=132)=" + String(gk) + " ksplit_leaf="
            + String(gf) + " default(row=" + String(DEFAULT_ROW_ON) + ")=" + String(gd_on)
            + " default(row=0)=" + String(gd_off) + " default(column row="
            + String(GEMM_KSPLIT_DEFAULT_S) + ")=" + String(gd_col)
        )
        if gk != want_ksplit[i] or gf != want_leaf[i]:
            failures.append(
                "RULE " + gemm_step_lm_call_name(i) + ": ksplit(S=132)=" + String(gk) + " (brief "
                + String(want_ksplit[i]) + "), ksplit_leaf=" + String(gf) + " (brief "
                + String(want_leaf[i]) + ")"
            )
        if gd_on != want_ksplit[i] or gd_off != 0:
            failures.append(
                "RULE " + gemm_step_lm_call_name(i) + ": default(row=" + String(DEFAULT_ROW_ON)
                + ")=" + String(gd_on) + " (brief " + String(want_ksplit[i]) + "), default(row=0)="
                + String(gd_off) + " (must be 0: the row turns the default off)"
            )
    print("check_group_rule_hand_counts: " + String(len(failures) - before) + " failures")


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
    group_sizes: List[Int],
    mut kreach_ok: List[Int],
    mut kreach_runs: List[Int],
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

    # The reference: the OLD shipped 128x128 plan, forced by name. Since
    # DEVIATION 2595 the shipped default is held to it by check_default_dispatch.
    _run(ctx, dc, da, db, dw, hexp, op, m, n, k, RUN_PLAN, PLAN_TUNED_128_8X8, False)
    var left = gemm_step_poison_left(hexp, mn)
    if left != 0:
        failures.append(tag + ": the old 128x128 plan left " + String(left) + " cells poisoned")
        return
    # FLAT against it: the two references agree before any arm is judged.
    _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_PLAN, PLAN_FLAT, False)
    var cf = gemm_step_compare(hgot, hexp, mn)
    if cf[0] != 0 or cf[1] != 0:
        failures.append(
            tag + ": FLAT differs from the old 128x128 plan in " + String(cf[0])
            + " cells (first " + String(cf[2]) + "), poison " + String(cf[1])
        )
    for geom in range(1, GEMM_GEOM_COUNT):
        var gname = gemm_step_geometry_name(geom)
        # Brief section 11: without the trial define an arm geometry is the
        # shipped dispatch, which at this shape may pick a SPLIT plan whose
        # workspace `dw` (sized above for FLAT and the old plan) does not
        # hold. Nothing is launched; check_ragged_controls fails the geometry
        # by name. `tuned128` is PLAN_TUNED_128_8X8 on every build and runs.
        if not GEMM_ARM_TRIAL and geom != GEMM_GEOM_TUNED128:
            reach_runs[geom] += 1
            continue
        _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_GEOMETRY, geom, False)
        var cc = gemm_step_compare(hgot, hexp, mn)
        if cc[0] != 0 or cc[1] != 0:
            failures.append(
                tag + " [" + gname + "]: MOVED " + String(cc[0]) + " cells against the old plan (first "
                + String(cc[2]) + "), poison " + String(cc[1])
            )
        _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_GEOMETRY, geom, True)
        var cs = gemm_step_compare(hgot, hexp, mn)
        # DEVIATION 2592: one per block for 2540 and 2541, one per launched
        # (tile, group) whose sabotaged cell is in the output for ksplit;
        # DEVIATION 2595: 0 for tuned128.
        var blocks = gemm_step_geometry_reach(geom, m, n, k)
        reach_runs[geom] += 1
        if cs[0] == blocks and cs[1] == 0:
            reach_ok[geom] += 1
        else:
            failures.append(
                tag + " [" + gname + "]: REACH NOT PROVEN: the sabotage moved " + String(cs[0])
                + " cells and the geometry's reach is " + String(blocks) + " (group leaves "
                + String(gemm_step_geometry_group_leaves(geom, m, n, k)) + ")" + _trial_hint()
            )
    # DEVIATION 2592: the group kernel and its fold at explicit group sizes,
    # every size a power of two (a size at or above P is one group).
    for gi in range(len(group_sizes)):
        var gl = group_sizes[gi]
        var kname = String("ksplit group=") + String(gl)
        # Brief section 11: the group launch is not compiled without the
        # trial define and the fallback reads `dw`; launch nothing.
        if not GEMM_ARM_TRIAL:
            kreach_runs[gi] += 1
            continue
        _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_KSPLIT, gl, False)
        var kc = gemm_step_compare(hgot, hexp, mn)
        if kc[0] != 0 or kc[1] != 0:
            failures.append(
                tag + " [" + kname + "]: MOVED " + String(kc[0]) + " cells against the old plan (first "
                + String(kc[2]) + "), poison " + String(kc[1])
            )
        _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_KSPLIT, gl, True)
        var ks = gemm_step_compare(hgot, hexp, mn)
        var kexp = gemm_step_ksplit_reach(m, n, k, gl)
        kreach_runs[gi] += 1
        if ks[0] == kexp and ks[1] == 0:
            kreach_ok[gi] += 1
        else:
            failures.append(
                tag + " [" + kname + "]: REACH NOT PROVEN: the sabotage moved " + String(ks[0])
                + " cells and the group launch's reach is " + String(kexp) + _trial_hint()
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
    var group_sizes: List[Int] = [1, 2, 4, 16, 64]
    var kreach_ok = List[Int]()
    var kreach_runs = List[Int]()
    for _ in range(len(group_sizes)):
        kreach_ok.append(0)
        kreach_runs.append(0)
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
                _ragged_case(
                    ctx, ops[oi], m, n, k, False, salt, failures, reach_ok, reach_runs,
                    group_sizes, kreach_ok, kreach_runs,
                )
                cases += 1
                if m == 129 and n == 257 and (k == 129 or k == 1000):
                    _ragged_case(
                        ctx, ops[oi], m, n, k, True, salt + 1, failures, reach_ok, reach_runs,
                        group_sizes, kreach_ok, kreach_runs,
                    )
                    cases += 1
    for geom in range(1, GEMM_GEOM_COUNT):
        var what = String(" launches moved exactly the geometry's reach")
        if not GEMM_ARM_TRIAL and geom != GEMM_GEOM_TUNED128:
            what = String(" cases NOT LAUNCHED (no trial define: the arm kernel is not compiled)")
        print(
            "REACH ragged [" + gemm_step_geometry_name(geom) + "] "
            + String(reach_ok[geom]) + "/" + String(reach_runs[geom]) + what
        )
    for gi in range(len(group_sizes)):
        var kwhat = String(" launches moved exactly the group launch's reach")
        if not GEMM_ARM_TRIAL:
            kwhat = String(" cases NOT LAUNCHED (no trial define: the group launch is not compiled)")
        print(
            "REACH ragged [ksplit group=" + String(group_sizes[gi]) + "] "
            + String(kreach_ok[gi]) + "/" + String(kreach_runs[gi]) + kwhat
        )
    # Brief section 11: a build without the trial define launched none of
    # these, so each FAILS once, by name, instead of once per case.
    if not GEMM_ARM_TRIAL:
        for geom in range(1, GEMM_GEOM_COUNT):
            if geom == GEMM_GEOM_TUNED128:
                continue
            failures.append(
                "REACH ragged [" + gemm_step_geometry_name(geom) + "]: NOT RUN in "
                + String(reach_runs[geom]) + " cases, bits and reach unproven" + _trial_hint()
            )
        for gi in range(len(group_sizes)):
            failures.append(
                "REACH ragged [ksplit group=" + String(group_sizes[gi]) + "]: NOT RUN in "
                + String(kreach_runs[gi]) + " cases, bits and reach unproven" + _trial_hint()
            )
    print(
        "ragged controls: " + String(cases) + " cases x " + String(GEMM_GEOM_COUNT - 1)
        + " geometries and " + String(len(group_sizes)) + " explicit group sizes, "
        + String(skipped) + " (m n k) over the " + String(budget)
        + " budget skipped, " + String(len(failures) - before) + " failures"
    )


# ===========================================================================
# THE SHIPPED DEFAULT, BOTH SIDES OF ITS SWITCH (DEVIATION 2595)
# ===========================================================================


def check_default_dispatch(ctx: DeviceContext, mut failures: List[String]) raises:
    """DEVIATION 2595: the shipped dispatch, both sides of its switch, by
    reach, on every column.

    Shapes: 257x520 and 520x257 (the ragged outputs over the tuned plans'
    128 K-cell floor, so `choose_gemm_plan` answers PLAN_TUNED_128_8X8), all
    three ops, `k` in {128, 129, 300, 1000} under the flop budget (P = 1, 2,
    3, 8; the M4's 50 M budget stops before 1000). At row 132 the rule takes
    every case with `P >= 2` (15 tiles, one leaf per group) and declines
    `P = 1`.

    Per case, after FLAT is held to the old plan:

    - `identical_gemm_shipped_at_row_into` at three rows: the column's own
      (`GEMM_KSPLIT_DEFAULT_S`), 0, and `DEFAULT_ROW_ON`. Clean it must store
      the old plan's bits. Sabotaged it must move EXACTLY
      `gemm_step_ksplit_reach` at the default's group size where
      `gemm_default_ksplit_leaves_at` takes the call, and NOTHING where it
      does not. Row 0 moving nothing at every shape proves the old plan runs
      with the row at 0; the enabled row moving exactly the group launch's
      cells proves the ksplit body runs where the row enables it.
    - The column's shipped entry: `identical_gemm_shipped_into` clean, then
      `identical_gemm_into` with MOJOLEARN_GEMM_ARM unset and
      MOJOLEARN_GEMM_ARM_SABOTAGE=1 (the trial hook serves a sabotaged
      shipped call through the same body), which must move exactly the
      column's default reach: the group launch's cells where the column's
      row is above 0, nothing where it is 0.

    Guards against a vacuous pass: the enabled row must take at least one
    case under the budget; row 0 must take none; the column's entry must take
    ksplit somewhere when its row is above 0 and nowhere when it is 0.

    Without `-D MOJOLEARN_GEMM_ARM_TRIAL=1` (brief section 11) no sabotage
    runs at any row or through the entry, because the sabotaged body is not
    compiled and a clean run that moves nothing proves no reach. The clean
    launches still run and must store the old plan's bits (the shipped body
    at every row, `identical_gemm_shipped_into` and `identical_gemm_into`),
    and each row and the entry FAIL once, by name, for the unproven reach.
    """
    var budget = gemm_step_env_int("MOJOLEARN_GEMM_STEP_CHECK_FLOPS", 400_000_000)
    var dims: List[Int] = [257, 520, 520, 257]
    var ks: List[Int] = [128, 129, 300, 1000]
    var ops: List[Int] = [OP_NN, OP_NT, OP_TN]
    var rows: List[Int] = [GEMM_KSPLIT_DEFAULT_S, 0, DEFAULT_ROW_ON]
    var row_ok = List[Int]()
    var row_runs = List[Int]()
    var row_took = List[Int]()
    for _ in range(len(rows)):
        row_ok.append(0)
        row_runs.append(0)
        row_took.append(0)
    var entry_ok = 0
    var entry_runs = 0
    var entry_took = 0
    var before = len(failures)
    var cases = 0
    var skipped = 0
    for s in range(len(dims) // 2):
        var m = dims[2 * s]
        var n = dims[2 * s + 1]
        for ki in range(len(ks)):
            var k = ks[ki]
            if m * n * k > budget:
                skipped += 1
                continue
            for oi in range(len(ops)):
                var op = ops[oi]
                var mn = m * n
                var counts = gemm_step_operand_counts(m, n, k)
                var nws = identical_gemm_workspace_max_floats(m, n, k)
                var da = ctx.enqueue_create_buffer[DType.float32](counts[0])
                var db = ctx.enqueue_create_buffer[DType.float32](counts[1])
                var dc = ctx.enqueue_create_buffer[DType.float32](mn)
                var dw = ctx.enqueue_create_buffer[DType.float32](nws)
                var hexp = ctx.enqueue_create_host_buffer[DType.float32](mn)
                var hgot = ctx.enqueue_create_host_buffer[DType.float32](mn)
                ctx.synchronize()
                var salt = 701 + 13 * s + 7 * ki + oi
                gemm_step_fill(ctx, da, counts[0], salt, False)
                gemm_step_fill(ctx, db, counts[1], salt + 7919, False)
                var tag = (
                    String("DEFAULT ") + op_name(op) + " " + String(m) + "x" + String(n) + "x"
                    + String(k)
                )
                if choose_gemm_plan(m, n, k) != PLAN_TUNED_128_8X8:
                    failures.append(
                        tag + ": choose_gemm_plan returns " + gemm_plan_name(choose_gemm_plan(m, n, k))
                        + ", not the 128x128 plan this check is built on"
                    )
                _run(ctx, dc, da, db, dw, hexp, op, m, n, k, RUN_PLAN, PLAN_TUNED_128_8X8, False)
                var left = gemm_step_poison_left(hexp, mn)
                if left != 0:
                    failures.append(tag + ": the old plan left " + String(left) + " cells poisoned")
                    continue
                _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_PLAN, PLAN_FLAT, False)
                var cf = gemm_step_compare(hgot, hexp, mn)
                if cf[0] != 0 or cf[1] != 0:
                    failures.append(
                        tag + ": FLAT differs from the old plan in " + String(cf[0])
                        + " cells (first " + String(cf[2]) + "), poison " + String(cf[1])
                    )
                cases += 1
                for ri in range(len(rows)):
                    var row = rows[ri]
                    var gl = gemm_default_ksplit_leaves_at(m, n, k, row)
                    var reach = 0
                    if gl > 0:
                        reach = gemm_step_ksplit_reach(m, n, k, gl)
                        row_took[ri] += 1
                    _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_ROW, row, False)
                    var cc = gemm_step_compare(hgot, hexp, mn)
                    row_runs[ri] += 1
                    if not GEMM_ARM_TRIAL:
                        # Brief section 11: the sabotaged body is not compiled,
                        # so no reach is provable at ANY row (a clean run that
                        # moves nothing proves nothing about a sabotage). The
                        # clean bits are the shipped build's own check; the
                        # row fails once, by name, after the loop.
                        var clean_ok = cc[0] == 0 and cc[1] == 0
                        if clean_ok:
                            row_ok[ri] += 1
                        else:
                            failures.append(
                                tag + " row=" + String(row) + ": clean moved " + String(cc[0])
                                + " (poison " + String(cc[1]) + ") against the old plan"
                            )
                        print(
                            tag + " row=" + String(row) + " leaves_per_group=" + String(gl)
                            + " reach=" + String(reach) + " clean_moved=" + String(cc[0])
                            + " sabotage=NOT RUN (no trial define) plan=["
                            + gemm_shipped_dispatch_name_at(m, n, k, row) + "]"
                            + (" CLEAN OK" if clean_ok else " FAIL")
                        )
                        continue
                    _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_ROW, row, True)
                    var cs = gemm_step_compare(hgot, hexp, mn)
                    var ok = cc[0] == 0 and cc[1] == 0 and cs[0] == reach and cs[1] == 0
                    if ok:
                        row_ok[ri] += 1
                    print(
                        tag + " row=" + String(row) + " leaves_per_group=" + String(gl)
                        + " reach=" + String(reach) + " clean_moved=" + String(cc[0])
                        + " sabotage_moved=" + String(cs[0]) + " plan=["
                        + gemm_shipped_dispatch_name_at(m, n, k, row) + "]"
                        + (" OK" if ok else " FAIL")
                    )
                    if not ok:
                        failures.append(
                            tag + " row=" + String(row) + ": clean moved " + String(cc[0])
                            + " (poison " + String(cc[1]) + "), sabotage moved " + String(cs[0])
                            + " of the expected " + String(reach) + _trial_hint()
                        )
                # The column's own shipped entry, clean and through the hook.
                var glc = gemm_default_ksplit_leaves(m, n, k)
                var reachc = 0
                if glc > 0:
                    reachc = gemm_step_ksplit_reach(m, n, k, glc)
                    entry_took += 1
                _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_SHIPPED, 0, False)
                var ce = gemm_step_compare(hgot, hexp, mn)
                _ = setenv("MOJOLEARN_GEMM_ARM", "", True)
                # Brief section 11: without the trial define the entry has no
                # hook and would ignore the sabotage variable, so it runs clean
                # and must move nothing; the reach fails once after the loop.
                var want_se = reachc
                var se_label = String(" sabotage_moved=")
                if GEMM_ARM_TRIAL:
                    _ = setenv("MOJOLEARN_GEMM_ARM_SABOTAGE", "1", True)
                else:
                    want_se = 0
                    se_label = String(" sabotage=NOT RUN (no trial define) entry_clean_moved=")
                _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_ENTRY, 0, False)
                var se = gemm_step_compare(hgot, hexp, mn)
                _ = setenv("MOJOLEARN_GEMM_ARM_SABOTAGE", "", True)
                entry_runs += 1
                var eok = ce[0] == 0 and ce[1] == 0 and se[0] == want_se and se[1] == 0
                if eok:
                    entry_ok += 1
                print(
                    tag + " shipped entry (column row=" + String(GEMM_KSPLIT_DEFAULT_S) + ")"
                    + " leaves_per_group=" + String(glc) + " reach=" + String(reachc)
                    + " clean_moved=" + String(ce[0]) + se_label + String(se[0])
                    + " plan=[" + gemm_shipped_dispatch_name(m, n, k) + "]"
                    + (" OK" if eok else " FAIL")
                )
                if not eok:
                    failures.append(
                        tag + " shipped entry: clean moved " + String(ce[0]) + " (poison "
                        + String(ce[1]) + "), sabotage moved " + String(se[0]) + " of the expected "
                        + String(reachc) + _trial_hint()
                    )
                _ = da
                _ = db
                _ = dc
                _ = dw
                _ = hexp
                _ = hgot
    var row_what = String(" launches moved exactly the default's reach")
    var entry_what = String(" calls moved exactly the column's default reach")
    if not GEMM_ARM_TRIAL:
        row_what = String(" clean launches stored the old plan's bits; reach NOT PROVEN (no trial define)")
        entry_what = String(" clean calls stored the old plan's bits; reach NOT PROVEN (no trial define)")
    for ri in range(len(rows)):
        var which = String(" (this column's row)")
        if ri == 1:
            which = String(" (off)")
        elif ri == 2:
            which = String(" (enabled, the H100 row)")
        print(
            "REACH default [row=" + String(rows[ri]) + which + "] " + String(row_ok[ri]) + "/"
            + String(row_runs[ri]) + row_what + "; the ksplit body took " + String(row_took[ri])
            + " of the cases"
        )
    print(
        "REACH default [shipped entry, column row=" + String(GEMM_KSPLIT_DEFAULT_S) + "] "
        + String(entry_ok) + "/" + String(entry_runs) + entry_what + "; ksplit took "
        + String(entry_took) + " of them"
    )
    # Brief section 11: without the trial define no sabotage ran, so the
    # reach of every row and of the entry FAILS once, by name.
    if not GEMM_ARM_TRIAL:
        for ri in range(len(rows)):
            failures.append(
                "check_default_dispatch [row=" + String(rows[ri]) + "]: reach NOT PROVEN in "
                + String(row_runs[ri]) + " cases (clean bits equal in " + String(row_ok[ri]) + ")"
                + _trial_hint()
            )
        failures.append(
            "check_default_dispatch [shipped entry]: reach NOT PROVEN in " + String(entry_runs)
            + " cases (clean bits equal in " + String(entry_ok) + ")" + _trial_hint()
        )
    if cases > 0 and row_took[2] == 0:
        failures.append(
            "check_default_dispatch: the enabled row took ksplit in none of " + String(cases)
            + " cases under the budget: the check is vacuous"
        )
    if row_took[1] != 0:
        failures.append(
            "check_default_dispatch: row 0 took ksplit in " + String(row_took[1])
            + " cases: the row does not turn the default off"
        )
    if GEMM_KSPLIT_DEFAULT_S > 0:
        if cases > 0 and entry_took == 0:
            failures.append(
                "check_default_dispatch: this column's row is " + String(GEMM_KSPLIT_DEFAULT_S)
                + " and the shipped entry took ksplit in none of " + String(cases) + " cases"
            )
    elif entry_took != 0:
        failures.append(
            "check_default_dispatch: this column's row is 0 and the shipped entry took ksplit in "
            + String(entry_took) + " cases"
        )
    if cases == 0:
        failures.append(
            "check_default_dispatch: no case ran under the " + String(budget) + " flop budget"
        )
    print(
        "default dispatch: " + String(cases) + " cases x " + String(len(rows)) + " rows and the"
        + " shipped entry, " + String(skipped) + " (m n k) over the budget skipped, column row="
        + String(GEMM_KSPLIT_DEFAULT_S) + " [" + gemm_step_geometry_name(GEMM_GEOM_SHIPPED) + "], "
        + String(len(failures) - before) + " failures"
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
        # The reference is the OLD plan by name (DEVIATION 2595): the `shipped`
        # arm through the entry is the default, and it must store these bits.
        _run(ctx, dc, da, db, dw, hexp, op, m, n, k, RUN_PLAN, PLAN_TUNED_128_8X8, False)
        var left = gemm_step_poison_left(hexp, mn)
        if left != 0:
            failures.append("LM " + cname + ": the old plan left " + String(left) + " cells poisoned")
            continue
        # Brief section 11: without the trial define the entry reads no arm,
        # so every arm below would run the shipped default and no sabotage
        # exists. Check the shipped entry's CLEAN bits once per call and fail
        # each selected arm once, by name, after the loop.
        if not GEMM_ARM_TRIAL:
            _run(ctx, dc, da, db, dw, hgot, op, m, n, k, RUN_ENTRY, 0, False)
            var ce = gemm_step_compare(hgot, hexp, mn)
            var clean_ok = ce[0] == 0 and ce[1] == 0
            print(
                "LM " + cname + " " + op_name(op) + " " + String(m) + "x" + String(n) + "x" + String(k)
                + " shipped entry (no trial define: no arm, no sabotage) clean_moved=" + String(ce[0])
                + " shipped_plan=[" + gemm_shipped_dispatch_name(m, n, k) + "]"
                + (" CLEAN OK" if clean_ok else " FAIL")
            )
            if not clean_ok:
                failures.append(
                    "LM " + cname + " shipped entry: clean moved " + String(ce[0]) + " (poison "
                    + String(ce[1]) + ") against the old plan"
                )
        for ai in range(len(names)):
            var an = names[ai]
            if not gemm_step_selected(spec, an):
                continue
            if not GEMM_ARM_TRIAL:
                continue
            var arm = gemm_step_arm_parse(an)
            var geom = gemm_step_arm_geometry(arm, m, n, k)
            # DEVIATIONS 2592 and 2595: the default's reach for shipped (0
            # where the old plan runs), 0 for tuned128, one per block, or the
            # ksplit reach.
            var expected = gemm_step_geometry_reach(geom, m, n, k)
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
                + " arm=" + an + " geometry=[" + gemm_step_geometry_name(geom) + "] group_leaves="
                + String(gemm_step_geometry_group_leaves(geom, m, n, k)) + " launched_blocks="
                + String(gemm_step_geometry_launched_blocks(geom, m, n, k)) + " reach="
                + String(expected) + " clean_moved=" + String(cc[0]) + " sabotage_moved="
                + String(cs[0]) + " shipped_plan=[" + gemm_shipped_dispatch_name(m, n, k)
                + "] arm_plan=[" + gemm_step_arm_plan_label(arm) + "]"
                + (" OK" if ok else " FAIL")
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
    # Brief section 11: every selected arm FAILS once, by name, on a build
    # without the trial define (nothing was launched for it).
    if not GEMM_ARM_TRIAL:
        for ai in range(len(names)):
            if gemm_step_selected(spec, names[ai]):
                failures.append(
                    "LM arm=" + names[ai] + ": NOT RUN through identical_gemm_into, bits and"
                    + " reach unproven" + _trial_hint()
                )


def main() raises:
    print(
        "== gemm/checks/gemm_step_arms_check.mojo [" + numeric_mode_name() + "] trial="
        + String(GEMM_ARM_TRIAL) + " sabotage: " + gemm_sabotage_name() + " =="
    )
    print("   DEVIATIONS 2540 to 2543; docs/lanes/BRIEF_gemm_step_2026-09-11.md sections 6 and 10")
    print("   DEVIATIONS 2590 to 2592; docs/lanes/BRIEF_gemm_long_k_2026-09-11.md sections 4, 5 and 9")
    print(
        "   DEVIATION 2595; the same brief, section 10; column block parallelism row="
        + String(GEMM_KSPLIT_DEFAULT_S) + " shipped=[" + gemm_step_geometry_name(GEMM_GEOM_SHIPPED) + "]"
    )
    comptime if ANY_SABOTAGE:
        raise Error(
            "gemm_step_arms_check: refuses a build with a global GEMM sabotage ("
            + gemm_sabotage_name() + "): its baseline would be a sabotaged kernel"
        )
    print(
        "   DEVIATION 2599; docs/lanes/BRIEF_gemm_kernel_2026-09-11.md sections 5 and 6; kpack=["
        + gemm_step_geometry_name(GEMM_GEOM_KPACK) + "] kpack_wide=["
        + gemm_step_geometry_name(GEMM_GEOM_KPACK_WIDE) + "]"
    )
    print(
        "   DEVIATIONS 2640 to 2642; docs/lanes/BRIEF_gemm_final_2026-09-11.md sections 4 to 6; kfoldv=["
        + gemm_step_geometry_name(GEMM_GEOM_KFOLDV) + "] kfoldv_leaf=["
        + gemm_step_geometry_name(GEMM_GEOM_KFOLDV_LEAF) + "]"
    )
    var failures = List[String]()
    check_group_fold_is_the_contract_tree(failures)
    check_group_rule_hand_counts(failures)
    check_kpack_page_is_a_bijection(failures)
    check_kpack_rule_hand_counts(failures)
    check_kfold_lanes_is_the_stack_fold(failures)
    check_kfold_rule_hand_counts(failures)
    var ctx = DeviceContext()
    check_selector(ctx, failures)
    check_ragged_controls(ctx, failures)
    check_default_dispatch(ctx, failures)
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
        "PASS gemm step arms: the group fold is the contract tree on the host, the group rule"
        " matches the hand counts, the kpack page is a bijection and its rule matches (2599),"
        " the lane fold is the stack fold for G 1..255 and the kfold rules match (2640 to 2642),"
        " every geometry and every explicit group size bit-equal to"
        " the old 128x128 plan and to FLAT, reach proven per geometry, the shipped default"
        " proven at its column row, at row 0 and at the enabled row" + scope
    )
