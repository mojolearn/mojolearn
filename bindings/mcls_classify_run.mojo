# SPDX-License-Identifier: Apache-2.0
# RUN of accuracy_score, confusion_matrix, precision/recall/f-score and
# log_loss (bindings/mcls_classify_door.mojo): the typed sums records, the
# COUNT tail, the probabilities' and row-mean stages and the post-run flag
# read. No Python import (tools/python_free_dirs.sh). Lane
# metric-programs-2 (2026-11-12): moved from mcls_classify.mojo, where they
# built on the Python-holding `McProg`; same stage records in the same order.
from core.metrics_program_plan import MetricProgramPlan
from bindings.arena_prog_run import MetricArenaOwner, MetricRoute, metric_inputs_run
from bindings.mcls_common_run import MC_NONE
from bindings.metrics_exp_rank_run import labels_rank
from std.memory import bitcast

# cm_epi kinds (x_metrics/cm_epi.mojo, op 59) and the NORM modes.
comptime _CM_MCC = 1
comptime _CM_COUNT = 2
comptime _CM_NORM = 3
comptime _CM_WSUM_LEAF = 7
comptime _CM_WSUM_FOLD = 8
comptime _WSUM_BLOCK = 256
# row_metric kinds (x_metrics ranking.mojo) and rank_epi MEAN.
comptime _ROW_LOGLOSS = 2
comptime _RANK_MEAN = 0


@fieldwise_init
struct McEpi(Copyable, Movable):
    """One `cls_epi` request of `_Sums(epi=...)`: (kind, k, AVG, ZD, FMODE, beta^2)."""
    var kind: Int
    var k: Int
    var avg: Int
    var zd: Int
    var fmode: Int
    var b2: Float64


struct McSums(Movable):
    """The offsets `_Sums` keeps: the groupings' (OFF, OUT) pairs (match,
    true, pred, then the weight total's), the count slots (unweighted) and
    the epilogue slots; the match key and the weights for a COUNT tail."""
    var groups: List[Int]
    var cnt: List[Int]
    var eos: List[Int]
    var match_off: Int
    var w_off: Int
    var n: Int
    var labels: Int
    var weighted: Bool

    def __init__(out self):
        self.groups = List[Int]()
        self.cnt = List[Int]()
        self.eos = List[Int]()
        self.match_off = MC_NONE
        self.w_off = MC_NONE
        self.n = 0
        self.labels = 0
        self.weighted = False


def mc_wsum_sf64(mut plan: MetricProgramPlan, s: McSums, k: Int) raises -> Int:
    """`_wsum_sf64`: the weighted COUNT's hit and weight total as soft
    binary64 sums (one leaf unit per WSUM_BLOCK rows, then one fold stage
    per tree level)."""
    var nb = max(1, (s.n + _WSUM_BLOCK - 1) // _WSUM_BLOCK)
    var out = plan.scratch(4 * nb)
    plan.stage_op("cm_epi", nb, [_CM_WSUM_LEAF, s.match_off, s.w_off, s.n, out, k])
    var stride = 1
    for _ in range(64):  # driver-loop: over the fold tree levels (log2 of the block count)
        if stride >= nb:
            break
        plan.stage_op("cm_epi", (nb - stride + 2 * stride - 1) // (2 * stride), [_CM_WSUM_FOLD, out, stride, nb])
        stride *= 2
    return out


def mc_count_tail(mut plan: MetricProgramPlan, s: McSums, k: Int) raises -> Int:
    """`_cm_stage(..., _CM_COUNT, 11)`: the COUNT tail's 11-word slot."""
    var col = 1 if s.weighted else 0
    var out = plan.alloc(11)
    var tp = s.groups[col]
    var tot = s.n
    if s.weighted:
        tot = mc_wsum_sf64(plan, s, k)
    plan.stage_op("cm_epi", 1, [_CM_COUNT, tp, k, col, tot, out])
    return out


def mc_row_mean(mut plan: MetricProgramPlan, s: Int, cols: Int, y: Int, n: Int, kind: Int, weighted: Bool,
                normalize: Bool, half: Bool) raises -> Int:
    """`_row_mean`'s stages (`_row_fold`, then rank_epi MEAN) before the
    run: the binary64 result's offset. `weighted`: the weights (n words) are
    the plan's next input, which the door pins right after this call."""
    var out = plan.scratch(n)
    plan.stage_op("row_metric", n, [s, cols, y, out, kind, 0, MC_NONE])
    var zero = plan.scratch(n)
    plan.stage_op("pair_key", n, [0, 0, zero, 1, 2])
    var w_off = MC_NONE
    if weighted:
        w_off = plan.input(n)
    var tot = plan.group(zero, n, 1, out, 1, w_off, 1)[1]
    var sw = MC_NONE
    if weighted:
        sw = plan.group(zero, n, 1, MC_NONE, 1, w_off, 1)[1]
    var res = plan.alloc(2)
    plan.stage_op("rank_epi", 1, [_RANK_MEAN, tot, n, sw, 1 if normalize else 0, 1 if half else 0, res])
    return res


def mc_proba_stages(mut plan: MetricProgramPlan, a_off: Int, n: Int, k: Int, binary: Bool) raises -> Tuple[Int, Int]:
    """`_put_proba` after the probabilities' input: (S, FLAGS)."""
    var flags = plan.alloc(3)
    var s = a_off
    if binary:
        s = plan.scratch(2 * n)
    if n:
        plan.stage_op("proba_rows", n, [a_off, n, 1 if binary else k, 1 if binary else 0, s, flags])
    return (s, flags)


def mc_proba_code(arena: MetricArenaOwner, flags: Int) raises -> Int:
    """`_proba_code`: 0, or the first failed check."""
    if arena.i32(flags) != 0:
        return 1
    if arena.i32(flags + 1) != 0:
        return 2
    if arena.i32(flags + 2) != 0:
        return 3
    return 0


# ---------------------------------------------------------------- the typed sums builder
# Lane mf-metric4 (2026-11-23): `_Sums` was assembled twice, inside
# bindings/mcls_classify_door.mojo `mc_sums` and bindings/metrics_exp_cls_door.mojo
# `_sums`, over Python pins. It is one typed builder now (both doors call
# it, and the search scorers' RUN, core/msel_scorer_run.mojo): the inputs a
# (true codes), b (predicted codes) and, weighted, w (n words each, in that
# order: the caller's buffers), the pair key, the three count groupings (and
# the weight total's), the count slots. The same stages in the same order:
# the bits do not move.


def mc_sums_stages(mut plan: MetricProgramPlan, n: Int, labels: Int, weighted: Bool) raises -> McSums:
    """`_Sums(true, pred, w, order)` up to its epilogues over `labels` codes."""
    var s = McSums()
    s.n = n
    s.labels = labels
    s.weighted = weighted
    var a = plan.input(n)
    var b = plan.input(n)
    var match_off = plan.scratch(n)
    plan.stage_op("pair_key", n, [a, b, match_off, labels, 1])
    var w_off = MC_NONE
    if weighted:
        w_off = plan.input(n)
    s.match_off = match_off
    s.w_off = w_off
    var keys: List[Int] = [match_off, a, b]
    for g in range(3):  # driver-loop: over the three count groupings
        var got = plan.group(keys[g], n, labels, MC_NONE, 1, w_off, 1)
        s.groups.append(got[0])
        s.groups.append(got[1])
    if weighted:
        var zero = plan.scratch(n)
        plan.stage_op("pair_key", n, [a, a, zero, 1, 2])
        var got = plan.group(zero, n, 1, MC_NONE, 1, w_off, 1)
        s.groups.append(got[0])
        s.groups.append(got[1])
    else:
        for g in range(3):  # driver-loop: over the three count slots
            s.cnt.append(plan.counts(s.groups[2 * g], labels))
    return s^


def mc_epi_words(mut plan: MetricProgramPlan, tp: Int, tr: Int, pr: Int, labels: Int, col: Int,
                 e: McEpi) raises -> Int:
    """One `cls_epi` stage over the groupings (tp, true, pred); FMODE 2's
    (1 + beta^2, beta^2) as program-owned binary64 words. The slot."""
    var bo = 0
    if e.fmode == 2:
        var pairs: List[Float64] = [1 + e.b2, e.b2]
        bo = plan.input_f64s(pairs)
    var eo = plan.alloc(8 + 6 * max(labels, 1))
    plan.stage_op("cls_epi", 1, [tp, tr, pr, e.k, col, e.kind, e.avg, e.zd, eo, e.fmode, bo])
    return eo


def mc_epi_stage(mut plan: MetricProgramPlan, mut s: McSums, e: McEpi) raises -> Int:
    """`_Sums(epi=...)`'s epilogue `e` over the sums `s`; its slot (also kept in `s.eos`)."""
    var col = 1 if s.weighted else 0
    var eo = mc_epi_words(plan, s.groups[col], s.groups[2 + col], s.groups[4 + col], s.labels, col, e)
    s.eos.append(eo)
    return eo


def mc_mcc_words(mut plan: MetricProgramPlan, tp: Int, tr: Int, pr: Int, k: Int, col: Int) raises -> Int:
    """`_cm_stage(..., _CM_MCC, 4)`: the matthews tail's 4-word slot."""
    var out = plan.alloc(4)
    plan.stage_op("cm_epi", 1, [_CM_MCC, tp, tr, pr, k, col, out])
    return out


# ---------------------------------------------------------------- one label metric over typed codes
#: `label_metric_run`'s metrics (the search scorers' label family).
comptime LBL_JACCARD = 0
comptime LBL_BALANCED = 1
comptime LBL_PRECISION = 2
comptime LBL_RECALL = 3
comptime LBL_F1 = 4
comptime LBL_MCC = 5
#: cls_epi AVG codes (`_epi_codes`): micro, the plain (macro) mean, weighted;
#: (lane mf-score6) 0 the per-label values (average=None, x_metrics/cls_epi.mojo AVG_NONE).
comptime LBL_NONE = 0
#: x_metrics/cls_epi.mojo CLS_HEAD: the epilogue slot's words before the per-label values.
comptime _CLS_HEAD = 8
comptime LBL_MICRO = 1
comptime LBL_MACRO = 2
comptime LBL_WEIGHTED = 3


def label_present_codes(truth: Int, pred: Int, n: Int, k: Int, mut mapped_truth: List[Int32],
                        mut mapped_pred: List[Int32]) raises -> Int:
    """`_pair`'s sorted label union over class codes 0 .. k-1: the codes
    present in either vector, each mapped to its rank among them (the order
    a metric's `present` labels take; codes keep the labels' sorted order).
    Answers the count of present labels."""
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    var p = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=pred)
    var seen = List[Int32](length=max(k, 1), fill=Int32(-1))
    for i in range(n):  # driver-loop: over the rows (the label union, a mark per code)
        var a = Int(t[i])
        var b = Int(p[i])
        if a < 0 or a >= k or b < 0 or b >= k:
            raise Error("ValueError: y codes outside [0, n_classes)")
        seen[a] = 0
        seen[b] = 0
    var count = 0
    for c in range(k):  # driver-loop: over the class codes (their ranks)
        if seen[c] == 0:
            seen[c] = Int32(count)
            count += 1
    mapped_truth = List[Int32](length=max(n, 1), fill=Int32(0))
    mapped_pred = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # driver-loop: over the rows (the mapped codes)
        mapped_truth[i] = seen[Int(t[i])]
        mapped_pred[i] = seen[Int(p[i])]
    return count


def label_selected_codes(truth: Int, pred: Int, n: Int, k: Int, sel_at: Int, sel_n: Int,
                         mut mapped_truth: List[Int32], mut mapped_pred: List[Int32]) raises -> Int:
    """`mx_label_order(labels, present)` over class codes (lane mf-tail5b):
    the `sel_n` selected codes at `sel_at` (int32; a code >= k is a label
    no row holds) take ranks 0 .. sel_n-1, then every other code present in
    either vector in its sorted order. Answers the count of ordered labels."""
    var sel = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=sel_at)
    var span = k
    for j in range(sel_n):  # small-loop(the caller's labels)
        span = max(span, Int(sel[j]) + 1)
    var rank = List[Int32](length=max(span, 1), fill=Int32(-1))
    for j in range(sel_n):  # small-loop(the caller's labels: their ranks first)
        var c = Int(sel[j])
        if c < 0 or rank[c] >= 0:
            raise Error("labels must be unique")
        rank[c] = Int32(j)
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    var p = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=pred)
    var present = List[Bool](length=max(k, 1), fill=False)
    for i in range(n):  # driver-loop: over the rows (the label union, a mark per code)
        var a = Int(t[i])
        var b = Int(p[i])
        if a < 0 or a >= k or b < 0 or b >= k:
            raise Error("ValueError: y codes outside [0, n_classes)")
        present[a] = True
        present[b] = True
    var count = sel_n
    for c in range(k):  # driver-loop: over the class codes (the present ones after the selection)
        if present[c] and rank[c] < 0:
            rank[c] = Int32(count)
            count += 1
    mapped_truth = List[Int32](length=max(n, 1), fill=Int32(0))
    mapped_pred = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # driver-loop: over the rows (the mapped codes)
        mapped_truth[i] = rank[Int(t[i])]
        mapped_pred[i] = rank[Int(p[i])]
    return count


def label_metric_run(route: MetricRoute, metric: Int, avg: Int, truth: Int, pred: Int, n: Int,
                     k: Int, mut warn: Int) raises -> Float64:
    """`label_metric_options_run` with the metric's defaults (labels None,
    zero_division 'warn', beta 1)."""
    return label_metric_options_run(route, metric, avg, truth, pred, n, k, 0, 2, 1.0, 0, 0, warn)


def label_metric_options_run(route: MetricRoute, metric: Int, avg: Int, truth: Int, pred: Int, n: Int,
                             k: Int, zd: Int, fmode: Int, b2: Float64, sel_at: Int, sel_n: Int,
                             mut warn: Int) raises -> Float64:
    """`label_metric_options_run` unweighted, its average one of the codes."""
    var per_class = List[Float64]()
    return label_metric_options_run(route, metric, avg, truth, pred, n, k, zd, fmode, b2, sel_at, sel_n, 0, per_class,
                                    warn)


def label_metric_options_run(route: MetricRoute, metric: Int, avg: Int, truth: Int, pred: Int, n: Int,
                             k: Int, zd: Int, fmode: Int, b2: Float64, sel_at: Int, sel_n: Int, w: Int,
                             mut per_class: List[Float64], mut warn: Int) raises -> Float64:
    """Lane mf-score6: `w`, n float32 sample weights (0: none; the sums'
    third input, `mc_sums_stages(weighted)`, as the door's weighted
    program reads them); `avg` 0 (LBL_NONE, average=None) on a precision /
    recall / F / jaccard metric fills `per_class` with the epilogue's
    per-label values (the `sel_n` or present labels, in their order) and
    answers the first (NaN when there are none).
    THE ONE BODY of the averaged label metrics (lane mf-tail5b: a
    make_scorer's kwargs as words). `zd`: the cls_epi zero_division code
    (`_epi_zd`: 0 for 'warn' and 0, 1, 2 for nan); `fmode` / `b2`: the
    F-score's mode and beta^2 (`_fmode`: 2 with beta^2, 1 for beta 0, 0 for
    an infinite beta; precision / recall read neither); `sel_at` / `sel_n`:
    the caller's `labels` as int32 class codes (0 / 0: the present labels),
    a code >= k being a label no row holds. The order is
    `mx_label_order(labels, present)`: the selected codes, then the other
    present ones in their sorted order; the epilogue averages the first
    `sel_n` of them, as the doors' `_set_wise_labels` / `_label_order` do.
    THE TYPED RUN of one unweighted label metric with its default options
    (labels None, zero_division 'warn', adjusted False) over n int32 true and
    predicted class codes in 0 .. k-1 (the caller keeps both alive): the
    present labels, the door's program (`mc_sums_stages` + the epilogue or
    the matthews tail) through `metric_inputs_run`, the value. `warn`
    answers the door's warning words' bits (lane mf-score5; the caller
    raises them as the door does): precision / recall / f-score, the
    `cls_epi` flags (1 precision, 2 recall, 4 f-score: a zero denominator);
    jaccard, 1 for a label of no true and no predicted samples; balanced
    accuracy, 1 when y_pred holds classes y_true does not; matthews, 0. No
    Python."""
    warn = 0
    var mt = List[Int32]()
    var mp = List[Int32]()
    var labels: Int
    var chosen: Int
    if sel_n > 0:
        labels = label_selected_codes(truth, pred, n, k, sel_at, sel_n, mt, mp)
        chosen = sel_n
    else:
        labels = label_present_codes(truth, pred, n, k, mt, mp)
        chosen = labels
    var weighted = w != 0
    var col = 1 if weighted else 0
    if avg == LBL_NONE and (metric == LBL_MCC or metric == LBL_BALANCED):
        raise Error("RuntimeError: average=None on a label metric without per-label values")
    var plan = MetricProgramPlan()
    var s = mc_sums_stages(plan, n, labels, weighted)
    var at = MC_NONE
    if metric == LBL_MCC:
        at = mc_mcc_words(plan, s.groups[col], s.groups[2 + col], s.groups[4 + col], labels, col)
    elif metric == LBL_BALANCED:
        # lane mf-score8: `zd` is balanced_accuracy_score's `adjusted` (1),
        # the word the door's `_balanced_accuracy` stages in that slot
        at = mc_epi_stage(plan, s, McEpi(1, labels, 0, 1 if zd == 1 else 0, 0, 0.0))
    elif metric == LBL_JACCARD:
        at = mc_epi_stage(plan, s, McEpi(0, chosen, avg, zd, 0, 0.0))
    elif metric == LBL_PRECISION or metric == LBL_RECALL or metric == LBL_F1:
        at = mc_epi_stage(plan, s, McEpi(2, chosen, avg, zd, fmode, b2))
    else:
        raise Error("RuntimeError: unknown label metric " + String(metric))
    var addresses: List[Int] = [Int(mt.unsafe_ptr()), Int(mp.unsafe_ptr())]
    var sizes: List[Int] = [n, n]
    if weighted:
        addresses.append(w)
        sizes.append(n)
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(mt)  # the program's inputs were copied from these codes above
    _ = len(mp)
    if metric == LBL_MCC:
        if arena.i32(at) != 0:
            raise Error("math domain error")
        return arena.f64(at + 2)
    if metric == LBL_BALANCED:
        # `_balanced_accuracy`: the labels of nonzero true sum (OUT[1]) short of the present ones
        warn = 1 if arena.i32(at + 1) != labels else 0
        return arena.f64(at + 2)
    warn = arena.i32(at)
    if metric == LBL_JACCARD:
        warn = 1 if warn != 0 else 0
    if avg == LBL_NONE:
        # x_metrics/cls_epi.mojo: the per-label binary64 values after the
        # CLS_HEAD words (jaccard at V; precision, recall, F at V, V + 2k,
        # V + 4k, k the epilogue's label count)
        var v = at + _CLS_HEAD
        var base = v
        if metric == LBL_RECALL:
            base = v + 2 * chosen
        elif metric == LBL_F1:
            base = v + 4 * chosen
        per_class = List[Float64](capacity=chosen)
        for i in range(chosen):  # small-loop(the epilogue's labels: their values)
            per_class.append(arena.f64(base + 2 * i))
        if chosen == 0:
            return bitcast[DType.float64](UInt64(0x7FF8000000000000))
        return per_class[0]
    if metric == LBL_RECALL:
        return arena.f64(at + 4)
    if metric == LBL_F1:
        return arena.f64(at + 6)
    return arena.f64(at + 2)


def label_binary_run(route: MetricRoute, metric: Int, truth: Int, pred: Int, n: Int, k: Int,
                     positive: Int, mut warn: Int) raises -> Float64:
    """`label_binary_options_run` with the defaults (zero_division 'warn', beta 1)."""
    return label_binary_options_run(route, metric, truth, pred, n, k, positive, 0, 2, 1.0, warn)


def label_binary_options_run(route: MetricRoute, metric: Int, truth: Int, pred: Int, n: Int, k: Int,
                             positive: Int, zd: Int, fmode: Int, b2: Float64, mut warn: Int) raises -> Float64:
    """`label_binary_options_run` unweighted."""
    return label_binary_options_run(route, metric, truth, pred, n, k, positive, zd, fmode, b2, 0, warn)


def label_binary_options_run(route: MetricRoute, metric: Int, truth: Int, pred: Int, n: Int, k: Int,
                             positive: Int, zd: Int, fmode: Int, b2: Float64, w: Int,
                             mut warn: Int) raises -> Float64:
    """`w` (lane mf-score6): n float32 sample weights (0: none), the sums'
    third input as `label_metric_options_run` reads them.
    THE ONE BODY of the binary label metrics; `zd`, `fmode`, `b2` as
    `label_metric_options_run` reads them (lane mf-tail5b).
    THE TYPED RUN of one unweighted label metric with average='binary'
    (lane mf-search4b; precision, recall, f1 or jaccard, zero_division
    'warn') over n int32 true and predicted class codes in 0 .. k-1, the
    positive label being class code `positive` (mapped from pos_label at
    UNPACK). The door's order (bindings/mcls_classify_door.mojo `_prf` /
    `_prf_weighted`, `_set_wise_labels`): more than two observed labels
    refuse; a positive that is not observed among two refuses; the
    selected labels are [positive] then the other observed ones
    (`mc_extend_order`), the sums run over all of them and the epilogue's
    macro mean over the one selected label is the positive's value. An
    unobserved positive (fewer than two observed) is appended and scores
    0.0 (its zero denominators). `warn`: the epilogue's flags, as
    `label_metric_run` answers them (lane mf-score5). No Python."""
    warn = 0
    if positive < 0 or positive >= k:
        raise Error("RuntimeError: the binary scorer's positive class code is outside [0, n_classes)")
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    var p = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=pred)
    var seen = List[Int32](length=max(k, 1), fill=Int32(-1))
    for i in range(n):  # driver-loop: over the rows (the observed label union, a mark per code)
        var a = Int(t[i])
        var b = Int(p[i])
        if a < 0 or a >= k or b < 0 or b >= k:
            raise Error("ValueError: y codes outside [0, n_classes)")
        seen[a] = 0
        seen[b] = 0
    var observed = 0
    for c in range(k):  # small-loop(the class codes: the observed count)
        if seen[c] == 0:
            observed += 1
    if observed > 2:
        raise Error("average='binary' requires at most two observed classes")
    if seen[positive] != 0 and observed == 2:
        raise Error("pos_label is not a valid observed label")
    # The order: the positive first, then the other observed codes in their
    # sorted order.
    seen[positive] = 0
    var labels = 1
    for c in range(k):  # small-loop(the class codes: their places after the positive)
        if c == positive:
            continue
        if seen[c] == 0:
            seen[c] = Int32(labels)
            labels += 1
    var mt = List[Int32](length=max(n, 1), fill=Int32(0))
    var mp = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # driver-loop: over the rows (the mapped codes)
        mt[i] = seen[Int(t[i])]
        mp[i] = seen[Int(p[i])]
    var plan = MetricProgramPlan()
    var s = mc_sums_stages(plan, n, labels, w != 0)
    var at = MC_NONE
    if metric == LBL_JACCARD:
        at = mc_epi_stage(plan, s, McEpi(0, 1, LBL_MACRO, zd, 0, 0.0))
    elif metric == LBL_PRECISION or metric == LBL_RECALL or metric == LBL_F1:
        at = mc_epi_stage(plan, s, McEpi(2, 1, LBL_MACRO, zd, fmode, b2))
    else:
        raise Error("RuntimeError: unknown binary label metric " + String(metric))
    var addresses: List[Int] = [Int(mt.unsafe_ptr()), Int(mp.unsafe_ptr())]
    var sizes: List[Int] = [n, n]
    if w != 0:
        addresses.append(w)
        sizes.append(n)
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(mt)  # the program's inputs were copied from these codes above
    _ = len(mp)
    warn = arena.i32(at)
    if metric == LBL_JACCARD:
        warn = 1 if warn != 0 else 0
    if metric == LBL_RECALL:
        return arena.f64(at + 4)
    if metric == LBL_F1:
        return arena.f64(at + 6)
    return arena.f64(at + 2)


# ---------------------------------------------------------------- log_loss, typed
# Lane mf-metric4 (2026-11-23): the program bindings/mcls_classify_door.mojo
# `_log_loss` / `_log_loss_options` assembled over pins, as one builder.


def mc_log_loss_stages(mut plan: MetricProgramPlan, n: Int, k: Int, binary: Bool, weighted: Bool,
                       normalize: Bool) raises -> Tuple[Int, Int]:
    """The log-loss program over its inputs: the probabilities (n words
    binary, else n * k), the codes (n) and, weighted, the weights (n), in
    that order. (the proba FLAGS slot, the binary64 result offset)."""
    var a_off = plan.input(n if binary else n * k)
    var put = mc_proba_stages(plan, a_off, n, k, binary)
    var y = plan.input(n)
    var res = mc_row_mean(plan, put[0], k, y, n, _ROW_LOGLOSS, weighted, normalize, False)
    return (put[1], res)


def log_loss_run(route: MetricRoute, truth: Int, proba: Int, n: Int, k: Int) raises -> Float64:
    """`log_loss_run` unweighted."""
    return log_loss_run(route, truth, proba, n, k, 0)


def log_loss_run(route: MetricRoute, truth: Int, proba: Int, n: Int, k: Int, w: Int) raises -> Float64:
    """`log_loss_options_run` with normalize=True."""
    return log_loss_options_run(route, truth, proba, n, k, w, True)


def log_loss_options_run(route: MetricRoute, truth: Int, proba: Int, n: Int, k: Int, w: Int,
                         normalize: Bool) raises -> Float64:
    """`log_loss_options_run` with labels None."""
    return log_loss_options_run(route, truth, proba, n, k, w, normalize, 0, 0)


def log_loss_options_run(route: MetricRoute, truth: Int, proba: Int, n: Int, k: Int, w: Int,
                         normalize: Bool, sel_at: Int, sel_n: Int) raises -> Float64:
    """`sel_at` / `sel_n` (lane mf-score8): log_loss's `labels` as int32
    class codes in their given order (`mc_selected`, then `mc_map_order`;
    a code >= k is a label no row holds); the door's checks in its order:
    fewer than two labels, a y_true label outside them, then the columns.
    `normalize` (lane mf-score8): log_loss's `normalize`, the rank_epi
    MEAN word `mc_log_loss_stages` stages (False: the summed loss).
    `w` (lane mf-score6): n float32 sample weights (0: none), the
    program's third input (`mc_log_loss_stages(weighted)`, the door's
    weighted mean).
    THE TYPED RUN of `log_loss(y_true, y_proba)` with its defaults over n
    int32 class codes in 0 .. k-1 and the member's float32 probabilities
    (n x k; for k == 2 the scorer's one column, n words, as
    `neg_log_loss` hands `predict_proba[:, 1]`): the labels are those
    present in y_true, as `_log_loss` selects them, in its refusal words and
    order. No Python."""
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    var codes = List[Int32](length=max(n, 1), fill=Int32(0))
    var selected = 0
    if sel_n > 0:
        if sel_n < 2:
            raise Error("log_loss requires at least two labels; pass labels for one observed class")
        selected = labels_rank(truth, n, k, sel_at, sel_n, codes, "y_true contains a label missing from labels")
    else:
        var rank = List[Int32](length=max(k, 1), fill=Int32(-1))
        for i in range(n):  # driver-loop: over the rows (the classes y_true holds)
            var c = Int(t[i])
            if c < 0 or c >= k:
                raise Error("ValueError: y codes outside [0, n_classes)")
            rank[c] = 0
        for c in range(k):  # driver-loop: over the class codes (their ranks among the selected)
            if rank[c] == 0:
                rank[c] = Int32(selected)
                selected += 1
        if selected < 2:
            raise Error("log_loss requires at least two labels; pass labels for one observed class")
        for i in range(n):  # driver-loop: over the rows (the encoded labels)
            codes[i] = rank[Int(t[i])]
    var binary = k == 2
    if binary:
        if selected != 2:
            raise Error("one-column probabilities require exactly two labels")
    elif k != selected:
        raise Error("probability columns must match the label count (at least two)")
    var plan = MetricProgramPlan()
    var got = mc_log_loss_stages(plan, n, k, binary, w != 0, normalize)
    var addresses: List[Int] = [proba, Int(codes.unsafe_ptr())]
    var sizes: List[Int] = [n if binary else n * k, n]
    if w != 0:
        addresses.append(w)
        sizes.append(n)
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(codes)  # copied into the program's arena above
    var code = mc_proba_code(arena, got[0])
    if code == 1:
        raise Error("log_loss probabilities must be finite")
    if code == 2:
        raise Error("log_loss probabilities must lie in [0,1]")
    if code == 3:
        raise Error("log_loss probabilities rows must sum to one within sqrt(float32 eps)")
    return arena.f64(got[1])


# ---------------------------------------------------------------- the COUNT tail over typed codes
# Lane mf-fitadm7 (2026-11-23): zero_one_loss and hamming_loss (1-D) as a
# typed RUN, the program bindings/metrics_exp_cls_door.mojo `_cm_count`
# assembles: the sums over the present labels (`_sums(pair, pair.present)`,
# here `label_present_codes` + `mc_sums_stages`) and the COUNT tail
# (`_cm_stage(..., _CM_COUNT, 11)`, here `mc_count_tail`). Same stages in
# the same order; the search scorers' RUN (core/msel_scorer_run.mojo) calls it.
#: `label_count_run`'s answers (the COUNT slot's words the doors read).
comptime CNT_LOSS = 0       # f64 at +6: the normalized loss (zero_one normalize=True, hamming_loss)
comptime CNT_COUNT = 1      # zero_one normalize=False: i32 at +10 (counts) or f64 at +4 (weighted)
#: lane mf-score8: accuracy_score normalize=False, the door's `accuracy_count`
#: (bindings/mcls_classify_door.mojo `_accuracy`): i32 at +1 (counts) or f64 at +2 (weighted)
comptime CNT_CORRECT = 3


def label_count_run(route: MetricRoute, answer: Int, truth: Int, pred: Int, n: Int, k: Int, w: Int,
                    mut zero_div: Bool) raises -> Float64:
    """THE TYPED RUN of the COUNT tail over n int32 true and predicted class
    codes in 0 .. k-1 (the caller keeps both alive) and, `w` != 0, n float32
    sample weights (the sums' third input). `answer` CNT_LOSS: the
    misclassified fraction (zero_one_loss normalize=True, hamming_loss on 1-D
    targets); CNT_COUNT: the misclassified count (an integer for unweighted
    codes, the weighted sum otherwise). `zero_div`: the slot's zero-total
    flag for CNT_LOSS (the caller raises the door's ZeroDivisionError words).
    No Python."""
    zero_div = False
    var mt = List[Int32]()
    var mp = List[Int32]()
    var labels = label_present_codes(truth, pred, n, k, mt, mp)
    var weighted = w != 0
    var plan = MetricProgramPlan()
    var s = mc_sums_stages(plan, n, labels, weighted)
    var out = mc_count_tail(plan, s, labels)
    var addresses: List[Int] = [Int(mt.unsafe_ptr()), Int(mp.unsafe_ptr())]
    var sizes: List[Int] = [n, n]
    if weighted:
        addresses.append(w)
        sizes.append(n)
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(mt)  # the program's inputs were copied from these codes above
    _ = len(mp)
    if answer == CNT_LOSS:
        zero_div = arena.i32(out) != 0
        return arena.f64(out + 6)
    if answer == CNT_CORRECT:
        if weighted:
            return arena.f64(out + 2)
        return Float64(arena.i32(out + 1))
    if weighted:
        return arena.f64(out + 4)
    return Float64(arena.i32(out + 10))


# ---------------------------------------------------------------- cohen_kappa_score, typed
# Lane mf-score8 (2026-11-23): the program bindings/metrics_exp_cls_door.mojo
# `_kappa` assembled inline (`_confusion` over the chosen labels, then the
# four cm_epi KAPPA phases), as one builder: the door calls it over its pins
# and the search scorers' RUN (core/msel_scorer_run.mojo) over typed codes.
# Same stages in the same order, the same arena layout: the bits do not move.
comptime _CM_KAPPA = 4
#: cohen_kappa_score's `weights` words (`_kappa`'s `wm`).
comptime KAPPA_NONE = 0
comptime KAPPA_LINEAR = 1
comptime KAPPA_QUADRATIC = 2
#: `kappa_run`'s flags (the slot's first word): no label of y2 in both y1
#: and the labels; only one label in common (the door's two warnings).
comptime KAPPA_NO_COMMON = 1
comptime KAPPA_ONE_COMMON = 2


def kappa_stages(mut plan: MetricProgramPlan, n: Int, k: Int, weighted: Bool, wm: Int) raises -> Int:
    """`_kappa`'s program over its inputs: the true and the predicted codes
    in the chosen labels' order (n int32 words each, -1 for a label outside
    them) and, weighted, the weights (n), in that order (the door's pins:
    `_confusion`'s two `put_i32`, then `put(w)` after the pair key). The
    4-word result slot (flags, then the binary64 kappa at +2)."""
    if k > 4096:
        raise Error("mojolearn metrics: at most 4096 labels in a confusion matrix")
    var a = plan.input(n)
    var b = plan.input(n)
    var key = plan.scratch(n)
    plan.stage_op("pair_key", n, [a, b, key, k, 0])
    var w_off = MC_NONE
    if weighted:
        w_off = plan.input(n)
    var g = plan.group(key, n, k * k, MC_NONE, 1, w_off, 1)
    var cells = g[1] if weighted else g[0]
    var col = 1 if weighted else 0
    var out = plan.want(plan.alloc(4), 4, -1, 1)
    var sc = plan.scratch(8 * k + 2)
    var phases: List[Int] = [0, 1, 2, 3]
    var units: List[Int] = [k, 1, k, 1]
    for i in range(4):  # small-loop(4: the four kappa stages): fixed
        if units[i]:
            plan.stage_op("cm_epi", units[i], [_CM_KAPPA, cells, k, col, phases[i], wm, out, sc])
    return out


def kappa_run(route: MetricRoute, truth: Int, pred: Int, n: Int, k: Int, sel_at: Int, sel_n: Int, wm: Int, w: Int,
              mut flags: Int) raises -> Float64:
    """THE TYPED RUN of `cohen_kappa_score(y1, y2, labels=..., weights=...)`
    over n int32 true and predicted class codes in 0 .. k-1 (the caller
    keeps both alive) and, `w` != 0, n float32 sample weights. `sel_at` /
    `sel_n`: the caller's `labels` as int32 class codes in their given order
    (a code >= k is a label no row holds; 0 / 0: the labels present in
    either vector, sorted, as `mx_selected(None, ...)` answers). A row whose
    label is outside the chosen ones is coded -1 (`mx_codes`), as the door
    codes it. The door's refusal when no chosen label is one of y1's, in its
    words. `flags`: KAPPA_NO_COMMON / KAPPA_ONE_COMMON when the door warns
    and answers its `replace_undefined_by` (the caller does both); else the
    kappa. No Python."""
    flags = 0
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    var p = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=pred)
    var mt = List[Int32]()
    var mp = List[Int32]()
    var chosen: Int
    if sel_n > 0:
        var sel = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=sel_at)
        var span = k
        for j in range(sel_n):  # small-loop(the caller's labels)
            span = max(span, Int(sel[j]) + 1)
        var rank = List[Int32](length=max(span, 1), fill=Int32(-1))
        for j in range(sel_n):  # small-loop(the caller's labels: their places)
            var c = Int(sel[j])
            if c < 0 or rank[c] >= 0:
                raise Error("labels must be unique")
            rank[c] = Int32(j)
        var in_y1 = False
        mt = List[Int32](length=max(n, 1), fill=Int32(0))
        mp = List[Int32](length=max(n, 1), fill=Int32(0))
        for i in range(n):  # driver-loop: over the rows (their codes among the chosen labels)
            var a = Int(t[i])
            var b = Int(p[i])
            if a < 0 or a >= k or b < 0 or b >= k:
                raise Error("ValueError: y codes outside [0, n_classes)")
            mt[i] = rank[a]
            mp[i] = rank[b]
            in_y1 = in_y1 or rank[a] >= 0
        if not in_y1:
            raise Error("At least one label in `labels` must be present in `y1` (even though "
                        + "`cohen_kappa_score` is otherwise agnostic to the order of `y1` and `y2`).")
        chosen = sel_n
    else:
        chosen = label_present_codes(truth, pred, n, k, mt, mp)
    var plan = MetricProgramPlan()
    var out = kappa_stages(plan, n, chosen, w != 0, wm)
    var addresses: List[Int] = [Int(mt.unsafe_ptr()), Int(mp.unsafe_ptr())]
    var sizes: List[Int] = [n, n]
    if w != 0:
        addresses.append(w)
        sizes.append(n)
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(mt)  # the program's inputs were copied from these codes above
    _ = len(mp)
    var bits = arena.i32(out)
    if (bits & KAPPA_NO_COMMON) != 0:
        flags = KAPPA_NO_COMMON
    elif (bits & KAPPA_ONE_COMMON) != 0:
        flags = KAPPA_ONE_COMMON
    return arena.f64(out + 2)


# ---------------------------------------------------------------- class_likelihood_ratios, typed
# Lane mf-score8 (2026-11-23): `_likelihood_ratios`' program (bindings/
# metrics_exp_cls_door.mojo: the 2 x 2 confusion over the chosen labels, then
# cm_epi CLR), one builder for the door and the search scorers'
# 'positive_likelihood_ratio' / 'neg_negative_likelihood_ratio' rows.
comptime _CM_CLR = 5
#: `clr_run`'s flags (the slot's first word, the door's three warnings).
comptime CLR_NO_POSITIVE = 1
comptime CLR_POS_UNDEFINED = 2
comptime CLR_NEG_UNDEFINED = 4


def clr_stages(mut plan: MetricProgramPlan, n: Int, weighted: Bool) raises -> Int:
    """`_likelihood_ratios`' program over its inputs: the true and the
    predicted codes among the two chosen labels (n int32 words each, -1 for
    a label outside them) and, weighted, the weights, in that order. The
    6-word slot (flags, LR+ at +2, LR- at +4, binary64)."""
    var a = plan.input(n)
    var b = plan.input(n)
    var key = plan.scratch(n)
    plan.stage_op("pair_key", n, [a, b, key, 2, 0])
    var w_off = MC_NONE
    if weighted:
        w_off = plan.input(n)
    var g = plan.group(key, n, 4, MC_NONE, 1, w_off, 1)
    var cells = g[1] if weighted else g[0]
    var out = plan.want(plan.alloc(6), 6, -1, 1)
    plan.stage_op("cm_epi", 1, [_CM_CLR, cells, 1 if weighted else 0, out])
    return out


def clr_run(route: MetricRoute, truth: Int, pred: Int, n: Int, k: Int, w: Int, pick: Int,
            mut flags: Int) raises -> Float64:
    """THE TYPED RUN of `class_likelihood_ratios(y_true, y_pred,
    replace_undefined_by=1.0)[pick]` (pick 0 LR+, 1 LR-) over n int32 class
    codes in 0 .. k-1 (and, `w` != 0, n float32 weights): the labels present
    in either vector, the door's refusals in its order and words (more than
    two: multiclass; one: not a 2 x 2 matrix), its program, then the
    replacement 1.0 of an undefined ratio, as the door's dict
    {'LR+': 1.0, 'LR-': 1.0} gives it; no positive sample answers NaN.
    `flags`: the slot's warning bits (the caller raises the door's words)."""
    flags = 0
    var mt = List[Int32]()
    var mp = List[Int32]()
    var present = label_present_codes(truth, pred, n, k, mt, mp)
    if present > 2:
        raise Error("class_likelihood_ratios only supports binary classification problems, "
                    + "got targets of type: multiclass")
    if present != 2:
        raise Error("class_likelihood_ratios needs a 2 x 2 confusion matrix")
    var plan = MetricProgramPlan()
    var out = clr_stages(plan, n, w != 0)
    var addresses: List[Int] = [Int(mt.unsafe_ptr()), Int(mp.unsafe_ptr())]
    var sizes: List[Int] = [n, n]
    if w != 0:
        addresses.append(w)
        sizes.append(n)
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(mt)  # the program's inputs were copied from these codes above
    _ = len(mp)
    flags = arena.i32(out)
    if (flags & CLR_NO_POSITIVE) != 0:
        return bitcast[DType.float64](UInt64(0x7FF8000000000000))
    if pick == 0:
        return 1.0 if (flags & CLR_POS_UNDEFINED) != 0 else arena.f64(out + 2)
    return 1.0 if (flags & CLR_NEG_UNDEFINED) != 0 else arena.f64(out + 4)
