# SPDX-License-Identifier: Apache-2.0
# RUN of the curve metrics (bindings/metrics_exp_curve_door.mojo): the typed
# program pieces of the curve and its fold. No Python import
# (tools/python_free_dirs.sh).
#
# Lane door-run-split (2026-11-14): `_fold_stages` built these stages on the
# door's Python-holding `MetricDoorProg`; it takes the typed plan here and the
# door passes `prog.plan`. Same scratch, allocations, outputs and stage words
# in the same order: the bits do not move.
from std.memory import bitcast
from core.metrics_program_plan import MetricProgramPlan
from bindings.metrics_exp_common_run import MX_NONE

# The fold's words per problem (`_FoldCurve._words`: s as two words, F, T, k, a, b, u, v, c).
comptime CURVE_CF_OUT = 10
# `curve_fold`'s rows per chunk.
comptime CURVE_CF_CHUNK = 1024


def curve_f32_bits(x: Float32) -> Int:
    return Int(bitcast[DType.int32](x))


def curve_fold_stages(mut plan: MetricProgramPlan, S: Int, stride: Int, POS: Int, W: Int, n: Int, problems: Int,
                      auc: Bool, has_max_fpr: Bool, max_fpr: Float64, out_in: Int) raises -> Int:
    """`_fold_stages`: the curve and its fold; the fold's CURVE_CF_OUT words per problem."""
    var N = n * problems
    var order = plan.scratch(N)
    var cnt = plan.scratch(problems)
    var fps = plan.scratch(N)
    var tps = plan.scratch(N)
    var thr = plan.scratch(N)
    var keep = MX_NONE
    var CF = 0
    var CM = 0
    if auc:
        keep = plan.scratch(N)
        CF = plan.scratch(3 * N)
        CM = plan.scratch(problems)
    plan.stage_op("bin_curve", problems, [S, stride, POS, W, n, order, fps, tps, thr, cnt, keep, 1 if auc else 0, CF, CM])
    var out = out_in
    if out < 0:
        out = plan.want(plan.alloc(CURVE_CF_OUT * problems), CURVE_CF_OUT * problems, -1, 1)
    var mode = 1
    var mh = 0
    var ml = 0
    if auc:
        mode = 0
        if has_max_fpr:
            # `_f32_split`: hi the nearest float32, lo the nearest float32 of the remainder
            var hi = Float32(max_fpr)
            var lo = Float32(max_fpr - Float64(hi))
            mode = 2
            mh = curve_f32_bits(hi)
            ml = curve_f32_bits(lo)
    if auc:
        plan.stage_op("curve_fold", problems, [n, CF, CF + N, CM, out, mode, mh, ml, CURVE_CF_CHUNK])
    else:
        plan.stage_op("curve_fold", problems, [n, fps, tps, cnt, out, mode, mh, ml, CURVE_CF_CHUNK])
    return out
