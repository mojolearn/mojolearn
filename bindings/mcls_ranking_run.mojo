# SPDX-License-Identifier: Apache-2.0
# RUN of roc_auc_score and precision_recall_curve (bindings/mcls_ranking_door.mojo):
# the typed curve-fold and average stages on the program plan. No Python
# import (tools/python_free_dirs.sh). Lane metric-programs-2 (2026-11-12):
# moved from mcls_ranking.mojo, where they built on the Python-holding
# `McProg`; same stage records in the same order.
from std.memory import bitcast
from core.metrics_program_plan import MetricProgramPlan
from bindings.mcls_common_run import MC_NONE
from bindings.arena_prog_run import MetricArenaOwner, MetricRoute, metric_inputs_run
from checks.numerics import pinned_mul_f64

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


@fieldwise_init
struct McAuc(ImplicitlyCopyable, Movable):
    """One folded AUC: `defined` False when the fold saw one class only (the
    door warns `_ROC_ONE_CLASS` and answers NaN)."""

    var defined: Bool
    var value: Float64


def mc_fold_auc_value(arena: MetricArenaOwner, off: Int, partial: Bool, max_fpr: Float64) -> McAuc:
    """`_FoldCurve.auc(max_fpr)` from the fold's words: the one body of the
    AUC finish (lane bm-meta-3, 2026-11-23: moved from the door's
    `mc_fold_auc`, which now only warns), every product pinned."""
    var s = arena.f32(off) + arena.f32(off + 1)
    var f_tot = arena.f32(off + 2)
    var t_tot = arena.f32(off + 3)
    var k = arena.i32(off + 4)
    var a = arena.f32(off + 5)
    var b = arena.f32(off + 6)
    var u = arena.f32(off + 7)
    var v = arena.f32(off + 8)
    var c = arena.i32(off + 9)
    if c <= 0 or not (f_tot > 0 and t_tot > 0):
        return McAuc(False, 0.0)
    var whole = s / pinned_mul_f64(pinned_mul_f64(2.0, f_tot), t_tot)
    if not partial or k >= c:
        return McAuc(True, whole)
    var x0 = a / f_tot
    var x1 = b / f_tot
    var y0 = u / t_tot
    var y1 = v / t_tot
    var yi = y0
    if x1 != x0:
        yi = y0 + pinned_mul_f64(max_fpr - x0, y1 - y0) / (x1 - x0)
    var part = whole + pinned_mul_f64(max_fpr - x0, yi + y0) / 2.0
    var min_area = pinned_mul_f64(pinned_mul_f64(0.5, max_fpr), max_fpr)
    return McAuc(True, pinned_mul_f64(0.5, 1 + (part - min_area) / (max_fpr - min_area)))


@fieldwise_init
struct RocMicro(ImplicitlyCopyable, Movable):
    """`roc_ovr_micro_run`'s answer: `rows_ok` False when a score row does not
    sum to 1 (the door raises `_ROWS_MSG`, checked before the AUC), else the
    micro AUC."""

    var rows_ok: Bool
    var auc: McAuc


def roc_ovr_micro_run(route: MetricRoute, s: Int, s_words: Int, codes: Int, codes_words: Int, w: Int, w_words: Int,
                      n: Int, k: Int) raises -> RocMicro:
    """RUN of `roc_auc_score(..., multi_class="ovr", average="micro")` (lane
    bm-meta-3, 2026-11-23; was composed in the door over Python pins):
    `_micro_inputs` and `_curves(s.reshape((n * k,)), flags, wm, n * k, 1,
    fold="auc")` as ONE x_metrics program, then its ROW_SUM check over the
    untouched scores. `s`: n x k float32 scores; `codes`: n int32 codes into
    the ordered classes; `w`: n float32 weights (`w_words` -1: none) (the caller
    keeps all three alive). `*_words`: each input's word count, as the door
    pinned it (its array's size). The same stages in the same order as the door
    built (lane ne2-b-members, N080): no bits change. No Python."""
    var plan = MetricProgramPlan()
    var addresses = List[Int]()
    var sizes = List[Int]()
    var s2 = plan.input(s_words)
    addresses.append(s)
    sizes.append(s_words)
    var c2 = plan.input(codes_words)
    addresses.append(codes)
    sizes.append(codes_words)
    var pos2 = plan.scratch(n * k)
    plan.stage_op("onehot", n * k, [c2, pos2, n, k, 1])
    var w2 = MC_NONE
    if w_words >= 0:
        var w_in = plan.input(w_words)
        addresses.append(w)
        sizes.append(w_words)
        w2 = plan.scratch(n * k)
        plan.stage_op("rep_rows", n * k, [w_in, w2, k])
    var out2 = mc_fold_stages(plan, s2, 1, pos2, w2, n * k, 1, False, 0.0, MC_NONE)
    var rows2 = plan.want(plan.alloc(1), 1, -1, 1)
    if n:
        plan.stage_op("rank_epi", n, [_RANK_ROW_SUM, s2, n, k, rows2])
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    if arena.i32(rows2) != 0:
        return RocMicro(False, McAuc(False, 0.0))
    return RocMicro(True, mc_fold_auc_value(arena, out2, False, 0.0))
