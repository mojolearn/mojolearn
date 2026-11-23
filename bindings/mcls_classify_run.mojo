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
#: cls_epi AVG codes (`_epi_codes`): micro, the plain (macro) mean, weighted.
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


def label_metric_run(route: MetricRoute, metric: Int, avg: Int, truth: Int, pred: Int, n: Int,
                     k: Int, mut warn: Int) raises -> Float64:
    """THE TYPED RUN of one unweighted label metric with its default options
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
    var labels = label_present_codes(truth, pred, n, k, mt, mp)
    var plan = MetricProgramPlan()
    var s = mc_sums_stages(plan, n, labels, False)
    var at = MC_NONE
    if metric == LBL_MCC:
        at = mc_mcc_words(plan, s.groups[0], s.groups[2], s.groups[4], labels, 0)
    elif metric == LBL_BALANCED:
        at = mc_epi_stage(plan, s, McEpi(1, labels, 0, 0, 0, 0.0))
    elif metric == LBL_JACCARD:
        at = mc_epi_stage(plan, s, McEpi(0, labels, avg, 0, 0, 0.0))
    elif metric == LBL_PRECISION or metric == LBL_RECALL or metric == LBL_F1:
        at = mc_epi_stage(plan, s, McEpi(2, labels, avg, 0, 2, 1.0))
    else:
        raise Error("RuntimeError: unknown label metric " + String(metric))
    var addresses: List[Int] = [Int(mt.unsafe_ptr()), Int(mp.unsafe_ptr())]
    var sizes: List[Int] = [n, n]
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
    if metric == LBL_RECALL:
        return arena.f64(at + 4)
    if metric == LBL_F1:
        return arena.f64(at + 6)
    return arena.f64(at + 2)


def label_binary_run(route: MetricRoute, metric: Int, truth: Int, pred: Int, n: Int, k: Int,
                     positive: Int, mut warn: Int) raises -> Float64:
    """THE TYPED RUN of one unweighted label metric with average='binary'
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
    var s = mc_sums_stages(plan, n, labels, False)
    var at = MC_NONE
    if metric == LBL_JACCARD:
        at = mc_epi_stage(plan, s, McEpi(0, 1, LBL_MACRO, 0, 0, 0.0))
    elif metric == LBL_PRECISION or metric == LBL_RECALL or metric == LBL_F1:
        at = mc_epi_stage(plan, s, McEpi(2, 1, LBL_MACRO, 0, 2, 1.0))
    else:
        raise Error("RuntimeError: unknown binary label metric " + String(metric))
    var addresses: List[Int] = [Int(mt.unsafe_ptr()), Int(mp.unsafe_ptr())]
    var sizes: List[Int] = [n, n]
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
    """THE TYPED RUN of `log_loss(y_true, y_proba)` with its defaults over n
    int32 class codes in 0 .. k-1 and the member's float32 probabilities
    (n x k; for k == 2 the scorer's one column, n words, as
    `neg_log_loss` hands `predict_proba[:, 1]`): the labels are those
    present in y_true, as `_log_loss` selects them, in its refusal words and
    order. No Python."""
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth)
    var rank = List[Int32](length=max(k, 1), fill=Int32(-1))
    for i in range(n):  # driver-loop: over the rows (the classes y_true holds)
        var c = Int(t[i])
        if c < 0 or c >= k:
            raise Error("ValueError: y codes outside [0, n_classes)")
        rank[c] = 0
    var selected = 0
    for c in range(k):  # driver-loop: over the class codes (their ranks among the selected)
        if rank[c] == 0:
            rank[c] = Int32(selected)
            selected += 1
    if selected < 2:
        raise Error("log_loss requires at least two labels; pass labels for one observed class")
    var binary = k == 2
    if binary:
        if selected != 2:
            raise Error("one-column probabilities require exactly two labels")
    elif k != selected:
        raise Error("probability columns must match the label count (at least two)")
    var codes = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # driver-loop: over the rows (the encoded labels)
        codes[i] = rank[Int(t[i])]
    var plan = MetricProgramPlan()
    var got = mc_log_loss_stages(plan, n, k, binary, False, True)
    var addresses: List[Int] = [proba, Int(codes.unsafe_ptr())]
    var sizes: List[Int] = [n if binary else n * k, n]
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
