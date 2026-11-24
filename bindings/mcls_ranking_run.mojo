# SPDX-License-Identifier: Apache-2.0
# RUN of roc_auc_score and precision_recall_curve (bindings/mcls_ranking_door.mojo):
# the typed curve-fold and average stages on the program plan. No Python
# import (tools/python_free_dirs.sh). Lane metric-programs-2 (2026-11-12):
# moved from mcls_ranking.mojo, where they built on the Python-holding
# `McProg`; same stage records in the same order.
from std.memory import bitcast
from core.metrics_program_plan import MetricProgramPlan
from bindings.mcls_common_run import MC_NONE
from bindings.arena_prog_run import MetricArenaOwner, MetricRoute, metric_inputs_run, metric_slots_run
from checks.numerics import pinned_mul_f64
from core.host_finite_scan import host_all_finite_f32

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
    """`roc_ovr_micro_run_slot` with host codes."""
    return roc_ovr_micro_run_slot(route, s, s_words, codes, codes_words, -1, w, w_words, n, k)


def _codes_slots(codes_slot: Int, inputs: Int) -> List[Int]:
    # The slots word of a program whose input 1 is the codes (lane
    # vdf-xm-wire4): empty (every input a host buffer, no allocation) for
    # host codes, else -1 per input but the codes' store slot.
    if codes_slot < 0:
        return List[Int]()
    var slots = List[Int](length=inputs, fill=-1)
    slots[1] = codes_slot
    return slots^


def roc_ovr_micro_run_slot(route: MetricRoute, s: Int, s_words: Int, codes: Int, codes_words: Int, codes_slot: Int,
                           w: Int, w_words: Int, n: Int, k: Int) raises -> RocMicro:
    """RUN of `roc_auc_score(..., multi_class="ovr", average="micro")` (lane
    bm-meta-3, 2026-11-23; was composed in the door over Python pins):
    `_micro_inputs` and `_curves(s.reshape((n * k,)), flags, wm, n * k, 1,
    fold="auc")` as ONE x_metrics program, then its ROW_SUM check over the
    untouched scores. `s`: n x k float32 scores; `codes`: n int32 codes into
    the ordered classes (`codes_slot` >= 0: those codes are a slot of the
    route's resident store, mapped on the device, read in place; lane
    vdf-xm-wire4); `w`: n float32 weights (`w_words` -1: none) (the caller
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
    var arena = metric_slots_run(route, plan, addresses, sizes, _codes_slots(codes_slot, len(addresses)))
    if arena.i32(rows2) != 0:
        return RocMicro(False, McAuc(False, 0.0))
    return RocMicro(True, mc_fold_auc_value(arena, out2, False, 0.0))


# ---------------------------------------------------------------- roc_auc, typed builders
# Lane mf-metric4 (2026-11-23): the programs bindings/mcls_ranking_door.mojo
# `_roc_binary`, `_roc_ovr` and `_roc_ovo` assembled over pins, as typed
# builders (the door calls them; so does the search scorers' RUN,
# core/msel_scorer_run.mojo). The same stages in the same order and layout.


def mc_roc_binary_stages(mut plan: MetricProgramPlan, n: Int, weighted: Bool, partial: Bool,
                         max_fpr: Float64) raises -> Int:
    """`_roc_binary`'s fold program over its inputs: the scores (n), the
    positive flags (n int32) and, weighted, the weights (n). The fold slot."""
    var s_off = plan.input(n)
    var pos = plan.input(n)
    var w_off = MC_NONE
    if weighted:
        w_off = plan.input(n)
    return mc_fold_stages(plan, s_off, 1, pos, w_off, n, 1, partial, max_fpr, MC_NONE)


@fieldwise_init
struct RocOvrSlots(ImplicitlyCopyable, Movable):
    """`_roc_ovr`'s (non-micro) program slots: the ROW_SUM flag, the per-class
    scores and flags, the average (MC_NONE for average=None)."""
    var rows: Int
    var sc: Int
    var fl: Int
    var res: Int


def mc_roc_ovr_stages(mut plan: MetricProgramPlan, n: Int, k: Int, weighted: Bool, averaged: Bool,
                      weighted_avg: Bool) raises -> RocOvrSlots:
    """`_roc_ovr`'s one-vs-rest program over its inputs: the scores (n x k),
    the codes (n int32) and, weighted, the weights (n)."""
    var s_off = plan.input(n * k)
    var c_off = plan.input(n)
    var pos = plan.scratch(n * k)
    plan.stage_op("onehot", n * k, [c_off, pos, n, k, 0])
    var w_off = MC_NONE
    if weighted:
        w_off = plan.input(n)
    var out = mc_fold_stages(plan, s_off, k, pos, w_off, n, k, False, 0.0, MC_NONE)
    # the tail: ROW_SUM, then the supports, scores and average
    var rows = plan.want(plan.alloc(1), 1, -1, 1)
    if n:
        plan.stage_op("rank_epi", n, [_RANK_ROW_SUM, s_off, n, k, rows])
    var got = plan.group(c_off, n, k, MC_NONE, 1, w_off, 1)
    var sc = plan.want(plan.alloc(2 * k), 2 * k, -1, 1)
    var fl = plan.want(plan.alloc(k), k, -1, 1)
    var sup = plan.want(plan.alloc(2 * k), 2 * k, -1, 1)
    plan.stage_op("rank_epi", k, [_RANK_CURVE_SCORE, out, k, 0, sc, fl, got[1] if weighted else got[0],
                                  1 if weighted else 0, sup])
    var res = MC_NONE
    if averaged:
        res = mc_stage_average(plan, sc, sup, k, weighted_avg)
    return RocOvrSlots(rows, sc, fl, res)


#: `RocOvo.status`: the answer, a score row that does not sum to 1, a y_true label not among the classes.
comptime ROC_OVO_OK = 0
comptime ROC_OVO_ROWS = 1
comptime ROC_OVO_LABEL = 2


@fieldwise_init
struct RocOvo(ImplicitlyCopyable, Movable):
    """`roc_ovo_run`'s answer: the status, the count of pair directions
    with one class only (the door warns once each), the average."""
    var status: Int
    var undefined: Int
    var value: Float64


def roc_ovo_run(route: MetricRoute, s: Int, codes: Int, n: Int, k: Int, weighted_avg: Bool) raises -> RocOvo:
    """`roc_ovo_run_slot` with host codes."""
    return roc_ovo_run_slot(route, s, codes, -1, n, k, weighted_avg)


def roc_ovo_run_slot(route: MetricRoute, s: Int, codes: Int, codes_slot: Int, n: Int, k: Int,
                     weighted_avg: Bool) raises -> RocOvo:
    """THE TYPED RUN of `_ovo_native` (was composed in the door over pins):
    per program the scores (n x k float32 at `s`) and codes (n int32 at
    `codes`) go in once, OVO_MASK weights each pair's rows, two curve folds
    per pair, CURVE_SCORE, OVO_PAIR and AVERAGE; pairs past `_OVO_WORDS` go
    to further programs whose pair words meet in one AVERAGE program. The
    caller keeps both buffers alive. `codes_slot` >= 0: the codes are a slot
    of the route's resident store (mapped on the device; every chunk's
    program reads it in place, lane vdf-xm-wire4). The checks of the first
    program stop the run, as the door's raises did. No Python."""
    var n_pairs = k * (k - 1) // 2
    var per_pair = 17 * n + 4 + 4 * _CF_OUT
    var chunk = max(1, min(n_pairs, _OVO_WORDS // max(per_pair, 1)))
    var pair_a = List[Int]()
    var pair_b = List[Int]()
    for a in range(k):  # driver-loop: over the classes (the pair table)
        for b in range(a + 1, k):  # driver-loop: over the classes after a
            pair_a.append(a)
            pair_b.append(b)
    var ps_words = List[Int32]()
    var prev_words = List[Int32]()
    var undefined = 0
    var g0 = 0
    var addresses: List[Int] = [s, codes]
    var sizes: List[Int] = [n * k, n]
    var slots = _codes_slots(codes_slot, 2)
    for _ in range(n_pairs + 1):  # driver-loop: over the chunks of class pairs (one program each)
        if g0 >= n_pairs:
            break
        var m = min(chunk, n_pairs - g0)
        var last_only = g0 == 0 and m == n_pairs
        var plan = MetricProgramPlan()
        var s_off = plan.input(n * k)
        var c_off = plan.input(n)
        var oh = plan.scratch(n * k)
        if n:
            plan.stage_op("onehot", n * k, [c_off, oh, n, k, 0])
        var rows = MC_NONE
        if g0 == 0:
            rows = plan.want(plan.alloc(1), 1, -1, 1)
            if n:
                plan.stage_op("rank_epi", n, [_RANK_ROW_SUM, s_off, n, k, rows])
        var got = plan.group(c_off, n, k, MC_NONE, 1, MC_NONE, 1)
        var off = got[0]
        _ = plan.want(off + k, 1, -1, 1)
        var cf = plan.scratch(2 * m * _CF_OUT)
        for t in range(m):  # driver-loop: over the class pairs of this program
            var a = pair_a[g0 + t]
            var b = pair_b[g0 + t]
            var w = plan.scratch(n)
            if n:
                plan.stage_op("rank_epi", n, [_RANK_OVO_MASK, c_off, n, a, b, w])
            _ = mc_fold_stages(plan, s_off + a, k, oh + a * n, w, n, 1, False, 0.0, cf + 2 * t * _CF_OUT)
            _ = mc_fold_stages(plan, s_off + b, k, oh + b * n, w, n, 1, False, 0.0, cf + (2 * t + 1) * _CF_OUT)
        var sc = plan.scratch(4 * m)
        var fl = plan.want(plan.alloc(2 * m), 2 * m, -1, 1)
        plan.stage_op("rank_epi", 2 * m, [_RANK_CURVE_SCORE, cf, 2 * m, 0, sc, fl, MC_NONE, 0, 0])
        var ps: Int
        var prev: Int
        if last_only:
            ps = plan.scratch(2 * m)
            prev = plan.scratch(2 * m)
        else:
            ps = plan.want(plan.alloc(2 * m), 2 * m, -1, 1)
            prev = plan.want(plan.alloc(2 * m), 2 * m, -1, 1)
        plan.stage_op("rank_epi", m, [_RANK_OVO_PAIR, sc, off, k, n, g0, m, ps, prev])
        var res = MC_NONE
        if last_only:
            res = mc_stage_average(plan, ps, prev, n_pairs, weighted_avg)
        var arena = metric_slots_run(route, plan, addresses, sizes, slots)
        if g0 == 0:
            if arena.i32(rows) != 0:
                return RocOvo(ROC_OVO_ROWS, undefined, 0.0)
            if arena.i32(off + k) != n:
                return RocOvo(ROC_OVO_LABEL, undefined, 0.0)
        for j in range(2 * m):  # driver-loop: over the pair directions of this program
            if arena.i32(fl + j) == 1:
                undefined += 1
        if last_only:
            return RocOvo(ROC_OVO_OK, undefined, arena.f64(res))
        for j in range(2 * m):  # driver-loop: over the pair words of this program
            ps_words.append(Int32(arena.i32(ps + j)))
            prev_words.append(Int32(arena.i32(prev + j)))
        g0 += m
    # the pair words are the AVERAGE program's own words
    var p2 = MetricProgramPlan()
    var ps_off = p2.input_words(ps_words^)
    var prev_off = p2.input_words(prev_words^)
    var res = mc_stage_average(p2, ps_off, prev_off, n_pairs, weighted_avg)
    var arena = metric_inputs_run(route, p2, List[Int](), List[Int]())
    return RocOvo(ROC_OVO_OK, undefined, arena.f64(res))


# ---------------------------------------------------------------- roc_auc over typed scores
#: `roc_auc_run`'s problems (the search scorers' ranking family).
comptime ROC_BINARY = 0
comptime ROC_OVR = 1
comptime ROC_OVR_WEIGHTED = 2
comptime ROC_OVO = 3
comptime ROC_OVO_WEIGHTED = 4
comptime _ROC_ONE_CLASS_WORDS = "Only one class is present in y_true. ROC AUC score is not defined in that case."
comptime _ROC_ROWS_WORDS = "Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes"


def roc_auc_run(route: MetricRoute, problem: Int, truth: Int, scores: Int, n: Int, k: Int) raises -> Float64:
    """`roc_auc_run` unweighted."""
    return roc_auc_run(route, problem, truth, scores, n, k, 0)


def roc_auc_run(route: MetricRoute, problem: Int, truth: Int, scores: Int, n: Int, k: Int,
                w: Int) raises -> Float64:
    """`w` (lane mf-score6): n float32 sample weights (0: none), the
    programs' last input as the door pins it (`mc_roc_binary_stages` /
    `mc_roc_ovr_stages(weighted)`); one-vs-one refuses them in the door's
    words (bindings/mcls_ranking_door.mojo).
    THE TYPED RUN of one unweighted `roc_auc_score` with the scorer's
    options over n int32 class codes in 0 .. k-1: ROC_BINARY (`roc_auc`:
    n float32 scores, the decision values or `predict_proba[:, 1]`), or the
    multiclass one-vs-rest / one-vs-one averages over the n x k float32
    probabilities. The classes are those y_true holds, the refusals the
    door's words. A one-class fold answers NaN (the door warns). No Python."""
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    var rank = List[Int32](length=max(k, 1), fill=Int32(-1))
    for i in range(n):  # driver-loop: over the rows (the classes y_true holds)
        var c = Int(t[i])
        if c < 0 or c >= k:
            raise Error("ValueError: y codes outside [0, n_classes)")
        rank[c] = 0
    var present = 0
    for c in range(k):  # driver-loop: over the class codes (their ranks)
        if rank[c] == 0:
            rank[c] = Int32(present)
            present += 1
    var codes = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # driver-loop: over the rows (the codes over the present classes)
        codes[i] = rank[Int(t[i])]
    var width = 1 if problem == ROC_BINARY else k
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=scores), n * width):
        if problem == ROC_BINARY:
            raise Error("binary ranking scores must be finite")
        raise Error("mojolearn roc_auc_score: y_score must be finite")
    if problem == ROC_BINARY:
        # the default call (`_roc_auc` with its defaults): two classes in y_true
        if k > 2 or present > 2:
            raise Error("multi_class must be in ('ovo', 'ovr')")
        if present != 2:
            raise Error("roc_auc_score requires both positive and negative classes")
        var plan = MetricProgramPlan()
        var out = mc_roc_binary_stages(plan, n, w != 0, False, 0.0)
        var addresses: List[Int] = [scores, Int(codes.unsafe_ptr())]
        var sizes: List[Int] = [n, n]
        if w != 0:
            addresses.append(w)
            sizes.append(n)
        var arena = metric_inputs_run(route, plan, addresses, sizes)
        _ = len(codes)
        var got = mc_fold_auc_value(arena, out, False, 0.0)
        return got.value if got.defined else _roc_nan()
    if k == 2 and present <= 2:
        # `_roc_options`' binary arm reads 1-D scores: an (n, 2) array is refused there
        raise Error("mojolearn roc_auc_score: y_score must be 1-D, got shape (" + String(n) + ", 2)")
    if present != k:
        raise Error("Number of classes in y_true not equal to the number of columns in 'y_score'")
    if problem == ROC_OVR or problem == ROC_OVR_WEIGHTED:
        var plan = MetricProgramPlan()
        var slots = mc_roc_ovr_stages(plan, n, k, w != 0, True, problem == ROC_OVR_WEIGHTED)
        var addresses: List[Int] = [scores, Int(codes.unsafe_ptr())]
        var sizes: List[Int] = [n * k, n]
        if w != 0:
            addresses.append(w)
            sizes.append(n)
        var arena = metric_inputs_run(route, plan, addresses, sizes)
        _ = len(codes)
        if arena.i32(slots.rows) != 0:
            raise Error(_ROC_ROWS_WORDS)
        return arena.f64(slots.res)
    if w != 0:
        raise Error("sample_weight is not supported for multiclass one-vs-one ROC AUC, "
                    + "'sample_weight' must be None in this case.")
    var got = roc_ovo_run(route, scores, Int(codes.unsafe_ptr()), n, k, problem == ROC_OVO_WEIGHTED)
    _ = len(codes)
    if got.status == ROC_OVO_ROWS:
        raise Error(_ROC_ROWS_WORDS)
    if got.status == ROC_OVO_LABEL:
        raise Error("mojolearn roc_auc_score: a y_true label is not among the classes")
    return got.value


#: Lane mf-score8: the one-vs-rest averages the table does not name.
comptime ROC_OVR_MICRO = 5
comptime ROC_OVR_NONE = 6
#: `roc_auc_options_run`'s `undefined` answer: a fold of one class (the door
#: warns `_ROC_ONE_CLASS` and answers NaN).
comptime ROC_ONE_CLASS_WORDS = _ROC_ONE_CLASS_WORDS


def roc_auc_options_run(route: MetricRoute, problem: Int, truth: Int, scores: Int, n: Int, k: Int, w: Int,
                        cols: Int, max_fpr: Float64, max_fpr_shown: String, sel_at: Int, sel_n: Int,
                        mut per_class: List[Float64], mut undefined: Int) raises -> Float64:
    """THE TYPED RUN of `roc_auc_score` with a make_scorer's options (lane
    mf-score8), in bindings/mcls_ranking_door.mojo `_roc_options`' order:
    a side of at most two classes with 1-D scores (`cols` 1: the column a
    binary member hands every predict_proba scorer, or its decision) is the
    binary arm (`_roc_binary`: a one-class side answers NaN, `undefined` 1;
    `max_fpr` > 0 and != 1 the partial AUC, McClish-corrected); otherwise
    multiclass: a partial `max_fpr` refuses (`max_fpr_shown` its words),
    the binary problem refuses as multi_class='raise', `labels` (`sel_at` /
    `sel_n`, their given order; one-vs-rest checks they cover y_true's) are
    the classes, then ROC_OVR / _WEIGHTED / _MICRO / _NONE (the per-class
    AUCs in `per_class`, answering the first) or ROC_OVO / _WEIGHTED. No
    Python."""
    undefined = 0
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    var rank = List[Int32](length=max(k, 1), fill=Int32(-1))
    for i in range(n):  # driver-loop: over the rows (the classes y_true holds)
        var c = Int(t[i])
        if c < 0 or c >= k:
            raise Error("ValueError: y codes outside [0, n_classes)")
        rank[c] = 0
    var present = 0
    for c in range(k):  # driver-loop: over the class codes (their ranks)
        if rank[c] == 0:
            rank[c] = Int32(present)
            present += 1
    var partial = max_fpr > 0.0 and max_fpr != 1.0
    var multiclass = present > 2 or cols > 2
    if not multiclass:
        if cols != 1:
            raise Error("mojolearn roc_auc_score: y_score must be 1-D, got shape (" + String(n) + ", "
                        + String(cols) + ")")
        if present != 2:
            undefined = 1
            return _roc_nan()
        if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=scores), n):
            raise Error("mojolearn roc_auc_score: y_score must be finite")
        var flags = List[Int32](length=max(n, 1), fill=Int32(0))
        for i in range(n):  # driver-loop: over the rows (y_true == the second present class)
            flags[i] = rank[Int(t[i])]
        var plan = MetricProgramPlan()
        var out = mc_roc_binary_stages(plan, n, w != 0, partial, max_fpr if partial else 0.0)
        var addresses: List[Int] = [scores, Int(flags.unsafe_ptr())]
        var sizes: List[Int] = [n, n]
        if w != 0:
            addresses.append(w)
            sizes.append(n)
        var arena = metric_inputs_run(route, plan, addresses, sizes)
        _ = len(flags)
        var got = mc_fold_auc_value(arena, out, partial, max_fpr if partial else 0.0)
        if not got.defined:
            undefined = 1
            return _roc_nan()
        return got.value
    if partial:
        raise Error("Partial AUC computation not available in multiclass setting, 'max_fpr' must be "
                    + "set to `None`, received `max_fpr=" + max_fpr_shown + "` instead")
    if problem == ROC_BINARY:
        raise Error("multi_class must be in ('ovo', 'ovr')")
    var ovo = problem == ROC_OVO or problem == ROC_OVO_WEIGHTED
    var nclasses = present
    var codes = List[Int32](length=max(n, 1), fill=Int32(0))
    if sel_n > 0:
        var sel = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=sel_at)
        var span = k
        for j in range(sel_n):  # small-loop(the caller's labels)
            span = max(span, Int(sel[j]) + 1)
        var at = List[Int32](length=max(span, 1), fill=Int32(-1))
        for j in range(sel_n):  # small-loop(the caller's labels: their places)
            at[Int(sel[j])] = Int32(j)
        for c in range(k):  # small-loop(the classes y_true holds: each among the labels)
            if rank[c] >= 0 and at[c] < 0 and not ovo:
                raise Error("'y_true' contains labels not in parameter 'labels'")
        for i in range(n):  # driver-loop: over the rows (their places among the labels; -1 outside)
            codes[i] = at[Int(t[i])]
        nclasses = sel_n
    else:
        for i in range(n):  # driver-loop: over the rows (the codes over the present classes)
            codes[i] = rank[Int(t[i])]
    if nclasses != cols:
        raise Error("Number of classes in y_true not equal to the number of columns in 'y_score'")
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=scores), n * cols):
        raise Error("mojolearn roc_auc_score: y_score must be finite")
    if ovo:
        if w != 0:
            raise Error("sample_weight is not supported for multiclass one-vs-one ROC AUC, "
                        + "'sample_weight' must be None in this case.")
        var got = roc_ovo_run(route, scores, Int(codes.unsafe_ptr()), n, cols, problem == ROC_OVO_WEIGHTED)
        _ = len(codes)
        if got.status == ROC_OVO_ROWS:
            raise Error(_ROC_ROWS_WORDS)
        if got.status == ROC_OVO_LABEL:
            raise Error("mojolearn roc_auc_score: a y_true label is not among the classes")
        undefined = got.undefined
        return got.value
    if problem == ROC_OVR_MICRO:
        var micro = roc_ovr_micro_run(route, scores, n * cols, Int(codes.unsafe_ptr()), n, w, n if w != 0 else -1, n,
                                      cols)
        _ = len(codes)
        if not micro.rows_ok:
            raise Error(_ROC_ROWS_WORDS)
        if not micro.auc.defined:
            undefined = 1
            return _roc_nan()
        return micro.auc.value
    var plan = MetricProgramPlan()
    var averaged = problem != ROC_OVR_NONE
    var slots = mc_roc_ovr_stages(plan, n, cols, w != 0, averaged, problem == ROC_OVR_WEIGHTED)
    var addresses: List[Int] = [scores, Int(codes.unsafe_ptr())]
    var sizes: List[Int] = [n * cols, n]
    if w != 0:
        addresses.append(w)
        sizes.append(n)
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(codes)
    if arena.i32(slots.rows) != 0:
        raise Error(_ROC_ROWS_WORDS)
    for c in range(cols):  # small-loop(the classes: one warning per class without both labels)
        if arena.i32(slots.fl + c) == 1:
            undefined += 1
    if not averaged:
        per_class = List[Float64](capacity=cols)
        for c in range(cols):  # small-loop(the classes: their AUCs)
            per_class.append(arena.f64(slots.sc + 2 * c))
        return per_class[0]
    return arena.f64(slots.res)


def _roc_nan() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF8000000000000))
