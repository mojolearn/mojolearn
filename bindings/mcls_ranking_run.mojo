# SPDX-License-Identifier: Apache-2.0
# RUN of roc_auc_score and precision_recall_curve (bindings/mcls_ranking_door.mojo):
# the typed curve-fold and average stages on the program plan. No Python
# import (tools/python_free_dirs.sh). Lane metric-programs-2 (2026-11-12):
# moved from mcls_ranking.mojo, where they built on the Python-holding
# `McProg`; same stage records in the same order.
from std.memory import bitcast
from core.metrics_program_plan import MetricProgramPlan
from bindings.mcls_common_run import MC_NONE

comptime _CF_OUT = 10
comptime _CF_CHUNK = 1024
comptime _OVO_WORDS = 1 << 26
comptime _RANK_ROW_SUM = 5
comptime _RANK_CURVE_SCORE = 6
comptime _RANK_AVERAGE = 7
comptime _RANK_OVO_MASK = 8
comptime _RANK_OVO_PAIR = 9
comptime _CO_PR = 1
comptime _CO_EMPTY = 1
comptime _CO_ZERO_CLASS = 2


def _f32_bits(x: Float64) -> Int:
    return Int(bitcast[DType.int32](Float32(x)))


def mc_fold_stages(mut plan: MetricProgramPlan, s: Int, stride: Int, pos: Int, w: Int, n: Int, problems: Int,
                   partial: Bool, max_fpr: Float64, out_at: Int) raises -> Int:
    """`_fold_stages(..., auc=True, max_fpr, out)`: the compacted curve
    (bin_curve) and its AUC fold (curve_fold); the fold's CF_OUT words per
    problem, at `out_at` (allocated and declared when MC_NONE)."""
    var total = n * problems
    var order = plan.scratch(total)
    var cnt = plan.scratch(problems)
    var fps = plan.scratch(total)
    var tps = plan.scratch(total)
    var thr = plan.scratch(total)
    var keep = plan.scratch(total)
    var cf = plan.scratch(3 * total)
    var cm = plan.scratch(problems)
    plan.stage_op("bin_curve", problems, [s, stride, pos, w, n, order, fps, tps, thr, cnt, keep, 1, cf, cm])
    var out = out_at
    if out == MC_NONE:
        out = plan.want(plan.alloc(_CF_OUT * problems), _CF_OUT * problems, -1, 1)
    var mode = 0
    var mh = 0
    var ml = 0
    if partial:
        # `_f32_split(max_fpr)`: the float-float hi + lo as Int32 bit words
        var hi = Float32(max_fpr)
        mode = 2
        mh = Int(bitcast[DType.int32](hi))
        ml = _f32_bits(max_fpr - Float64(hi))
    plan.stage_op("curve_fold", problems, [n, cf, cf + total, cm, out, mode, mh, ml, _CF_CHUNK])
    return out


def mc_stage_average(mut plan: MetricProgramPlan, sc: Int, sup: Int, m: Int, weighted: Bool) raises -> Int:
    """`_stage_average`: rank_epi AVERAGE; the declared binary64 result."""
    var res = plan.want(plan.alloc(2), 2, -1, 1)
    plan.stage_op("rank_epi", 1, [_RANK_AVERAGE, sc, sup, m, 1 if weighted else 0, res])
    return res
