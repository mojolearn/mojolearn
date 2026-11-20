# SPDX-License-Identifier: Apache-2.0
# RUN of the probability, hinge and ranking metrics
# (bindings/metrics_exp_rank_door.mojo): the typed program pieces and the
# post-run checks. No Python import (tools/python_free_dirs.sh).
#
# Lane metric-programs-2 (2026-11-12): `_Proba` held the codes and the
# probabilities as Python objects next to its shape; the shape is the typed
# `ProbaShape` here, the two arrays are the door's pins. `_row_fold`,
# `_row_mean_stage` and `_put_proba`'s stages build on the typed plan; a
# weighted fold declares the weights as the plan's next input and the door
# pins them in that order. Same layout, same records: the bits do not move.
from core.metrics_program_plan import MetricProgramPlan
from bindings.arena_prog_run import MetricArenaOwner
from bindings.metrics_exp_common_run import MX_NONE

# row_metric kinds (`_ROW`)
comptime _ROW_BRIER = 1
comptime _ROW_LOGLOSS = 2
comptime _ROW_HINGE_BIN = 3
comptime _ROW_HINGE_MC = 4
comptime _ROW_DCG = 5
comptime _ROW_COVERAGE = 6
comptime _ROW_LRAP = 7
comptime _ROW_RANKLOSS = 8
comptime _ROW_DCG_IGNORE_TIES = 9
# rank_epi kinds (x_metrics/rank_epi.mojo)
comptime _RANK_MEAN = 0
comptime _RANK_NDCG_ROW = 1
comptime _RANK_D2_LOG = 2
comptime _RANK_D2_BRIER = 3
comptime _RANK_DCG_DISC = 4
# flag_scan tests (x_metrics/tail.mojo)
comptime _SCAN_INDICATOR = 1
comptime _SCAN_NEGATIVE = 2


def row_fold(mut plan: MetricProgramPlan, S: Int, cols: Int, Y: Int, n: Int, kind: Int, weighted: Bool,
             K: Int, Dt: Int) raises -> Tuple[Int, Int, Int]:
    """`_row_fold`: (TOT, SW, W). `weighted`: the weights (n words) are the
    plan's next input, which the door pins right after this call."""
    var out = plan.scratch(n)
    plan.stage_op("row_metric", n, [S, cols, Y, out, kind, K, Dt])
    var zero = plan.scratch(n)
    plan.stage_op("pair_key", n, [0, 0, zero, 1, 2])
    var W = MX_NONE
    if weighted:
        W = plan.input(n)
    var tot = plan.group(zero, n, 1, out, 1, W, 1)[1]
    var sw = MX_NONE
    if weighted:
        sw = plan.group(zero, n, 1, MX_NONE, 1, W, 1)[1]
    return (tot, sw, W)


def row_mean_stage(mut plan: MetricProgramPlan, S: Int, cols: Int, Y: Int, n: Int, kind: Int, weighted: Bool,
                   K: Int, Dt: Int, normalize: Bool, half: Bool) raises -> Int:
    """`_row_mean` up to the run: the folds and the MEAN unit; the result slot."""
    var f = row_fold(plan, S, cols, Y, n, kind, weighted, K, Dt)
    var res = plan.alloc(2)
    plan.stage_op("rank_epi", 1, [_RANK_MEAN, f[0], n, f[1], 1 if normalize else 0, 1 if half else 0, res])
    return res


@fieldwise_init
struct ProbaShape(ImplicitlyCopyable, Movable):
    """`_proba`'s typed part: rows, classes, and the one-column form."""
    var n: Int
    var k: Int
    var binary: Bool


def proba_stages(mut plan: MetricProgramPlan, A: Int, p: ProbaShape) raises -> Tuple[Int, Int]:
    """`_put_proba` after the probabilities' input A: (S, FLAGS)."""
    var flags = plan.alloc(3)
    var S = A
    if p.binary:
        S = plan.scratch(2 * p.n)
    if p.n:
        plan.stage_op("proba_rows", p.n, [A, p.n, 1 if p.binary else p.k, 1 if p.binary else 0, S, flags])
    return (S, flags)


def proba_check(arena: MetricArenaOwner, flags: Int, caller: String) raises:
    var reason = String("")
    if arena.i32(flags) != 0:
        reason = "must be finite"
    elif arena.i32(flags + 1) != 0:
        reason = "must lie in [0, 1]"
    elif arena.i32(flags + 2) != 0:
        reason = "rows must sum to one within sqrt(float32 eps)"
    if reason.byte_length() > 0:
        raise Error("mojolearn " + caller + ": y_proba " + reason)


def dcg_table(mut plan: MetricProgramPlan, c: Int, b: Float64) raises -> Int:
    """`_dcg_table` after the log base's refusals: the discount table's slot."""
    var B = plan.input_f64(b)
    var D = plan.scratch(c)
    if c:
        plan.stage_op("rank_epi", c, [_RANK_DCG_DISC, c, B, D])
    return D
