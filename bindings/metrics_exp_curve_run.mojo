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
from core.family_cast_impl import family_min_f32, family_max_f32
from core.host_finite_scan import host_all_finite_f32
from bindings.metrics_exp_common_run import MX_NONE
from bindings.arena_prog_run import MetricArenaOwner, MetricRoute, metric_inputs_run

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


# ---------------------------------------------------------------- the typed builders of the search scorers
# Lane mf-score5 (2026-11-23): average_precision_score and
# top_k_accuracy_score were assembled inside bindings/metrics_exp_curve_door.mojo
# over pins. Their programs are typed builders here (typed words in, program
# out): the door calls them with its pins in the builder's input order, and
# the search scorers' RUN (core/msel_scorer_run.mojo) calls the same builders
# over its typed buffers. The same stages in the same order: the bits do not
# move.

comptime _RANK_CURVE_SCORE = 6
comptime _RANK_AVERAGE = 7
comptime _ROW_TOPK = 0
comptime _CRF32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _CRI32P = MutPointer[Int32, MutUntrackedOrigin]


@fieldwise_init
struct CurveFoldWords(ImplicitlyCopyable, Movable):
    """`_FoldCurve._words` of one problem: s, F, T, k, a, b, u, v, c."""
    var s: Float64
    var F: Float64
    var T: Float64
    var k: Int
    var a: Float64
    var b: Float64
    var u: Float64
    var v: Float64
    var c: Int


def curve_fold_words(arena: MetricArenaOwner, off: Int) -> CurveFoldWords:
    """The fold's CURVE_CF_OUT words at `off`."""
    return CurveFoldWords(arena.f32(off) + arena.f32(off + 1), arena.f32(off + 2), arena.f32(off + 3),
                          arena.i32(off + 4), arena.f32(off + 5), arena.f32(off + 6), arena.f32(off + 7),
                          arena.f32(off + 8), arena.i32(off + 9))


def ap_fold_value(arena: MetricArenaOwner, off: Int, mut zero_division: Bool) -> Float64:
    """`_FoldCurve.ap`: the average precision of one fold; `zero_division`
    True where the door raises ZeroDivisionError("float division by zero")."""
    zero_division = False
    var f = curve_fold_words(arena, off)
    if f.c <= 0 or f.T == 0:
        return 0.0
    if f.k > 0:
        zero_division = True
        return 0.0
    var q = f.s / f.T
    return q if q > 0.0 else 0.0


def ap_binary_stages(mut plan: MetricProgramPlan, n: Int, weighted: Bool) raises -> Int:
    """The binary average precision's program over its inputs S (n float32
    scores), POS (n int32 positive flags) and, weighted, W (n), in that
    order: one curve and its fold; the fold's words."""
    var S = plan.input(n)
    var POS = plan.input(n)
    var W = MX_NONE
    if weighted:
        W = plan.input(n)
    return curve_fold_stages(plan, S, 1, POS, W, n, 1, False, False, 0.0, -1)


@fieldwise_init
struct ApOvr(ImplicitlyCopyable, Movable):
    """The one-vs-rest average precision's slots: the folds, the per-class
    scores, flags and supports, the average's result (MX_NONE: average None)."""
    var out: Int
    var sc: Int
    var fl: Int
    var sup: Int
    var res: Int


def ap_average_stages(mut plan: MetricProgramPlan, SC: Int, SUP: Int, m: Int, weighted: Bool) raises -> Int:
    """`_stage_average`: the macro (or weighted) mean of m class scores."""
    var res = plan.want(plan.alloc(2), 2, -1, 1)
    plan.stage_op("rank_epi", 1, [_RANK_AVERAGE, SC, SUP, m, 1 if weighted else 0, res])
    return res


def ap_ovr_stages(mut plan: MetricProgramPlan, n: Int, k: Int, weighted: Bool, average: Bool,
                  average_weighted: Bool) raises -> ApOvr:
    """The one-vs-rest average precision's program over its inputs S (n x k
    float32 scores), C (n int32 class codes among the k classes) and,
    weighted, W (n), in that order: the class-major one-hot flags, the folds
    and `_OvrTail("ap", False, average)`."""
    var S = plan.input(n * k)
    var C = plan.input(n)
    var POS = plan.scratch(n * k)
    plan.stage_op("onehot", n * k, [C, POS, n, k, 0])
    var W = MX_NONE
    if weighted:
        W = plan.input(n)
    var out = curve_fold_stages(plan, S, k, POS, W, n, k, False, False, 0.0, -1)
    var g = plan.group(C, n, k, MX_NONE, 1, W, 1)
    var sc = plan.want(plan.alloc(2 * k), 2 * k, -1, 1)
    var fl = plan.want(plan.alloc(k), k, -1, 1)
    var sup = plan.want(plan.alloc(2 * k), 2 * k, -1, 1)
    plan.stage_op("rank_epi", k, [_RANK_CURVE_SCORE, out, k, 1, sc, fl, g[1] if weighted else g[0],
                                  1 if weighted else 0, sup])
    var res = MX_NONE
    if average:
        res = ap_average_stages(plan, sc, sup, k, average_weighted)
    return ApOvr(out, sc, fl, sup, res)


def ap_ovr_edge(arena: MetricArenaOwner, o: ApOvr, k: Int) -> Bool:
    """True when a class's AP needs the fold's own rule (its flag 2)."""
    for c in range(k):  # driver-loop: over the classes' fold flags
        if arena.i32(o.fl + c) == 2:
            return True
    return False


def ap_ovr_scores(arena: MetricArenaOwner, o: ApOvr, k: Int, mut zero_division: Bool) -> List[Float64]:
    """The per-class scores of an edge run: the fold's rule for a class of
    flag 2, the program's score for the others."""
    zero_division = False
    var scores = List[Float64](capacity=k)
    for c in range(k):  # driver-loop: over the classes' scores
        if arena.i32(o.fl + c) == 2:
            var zero = False
            scores.append(ap_fold_value(arena, o.out + CURVE_CF_OUT * c, zero))
            zero_division = zero_division or zero
        else:
            scores.append(arena.f64(o.sc + 2 * c))
    return scores^


def ap_edge_average_plan(scores: List[Float64], sup_words: List[Int32], k: Int, weighted: Bool,
                         mut res: Int) raises -> MetricProgramPlan:
    """The second program of an edge run: the scores and the supports as
    program-owned words, then their average (`res`)."""
    var plan = MetricProgramPlan()
    var SC = plan.input_f64s(scores)
    var SUP = plan.input_words(sup_words.copy())
    res = ap_average_stages(plan, SC, SUP, k, weighted)
    return plan^


def _present(truth: Int, n: Int, k: Int, mut rank: List[Int32]) raises -> Int:
    """The class codes y_true holds, each mapped to its rank among them
    (`rank[c]` -1 for an absent code); the count."""
    var t = _CRI32P(unsafe_from_address=truth)
    rank = List[Int32](length=max(k, 1), fill=Int32(-1))
    for i in range(n):  # driver-loop: over the rows (the classes y_true holds)
        var c = Int(t[i])
        if c < 0 or c >= k:
            raise Error("ValueError: y codes outside [0, n_classes)")
        rank[c] = 0
    var count = 0
    for c in range(k):  # driver-loop: over the class codes (their ranks)
        if rank[c] == 0:
            rank[c] = Int32(count)
            count += 1
    return count


def average_precision_run(route: MetricRoute, truth: Int, scores: Int, n: Int, cols: Int, k: Int,
                          positive: Int) raises -> Float64:
    """`average_precision_run` unweighted."""
    return average_precision_run(route, truth, scores, n, cols, k, positive, 0)


def average_precision_run(route: MetricRoute, truth: Int, scores: Int, n: Int, cols: Int, k: Int,
                          positive: Int, w: Int) raises -> Float64:
    """`w` (lane mf-score6): n float32 sample weights (0: none), the
    builders' W input as the door pins it (`ap_binary_stages` /
    `ap_ovr_stages(weighted)`).
    THE TYPED RUN of `average_precision_score(y_true, y_score)` with its
    defaults (average 'macro', pos_label 1) over n int32 class codes in
    0 .. k-1 and the member's float32 scores (n x cols; cols 1: the one
    column the scorer hands a binary member), `positive` the class code of
    pos_label=1 (-1: 1 is not a label). The door's order and words
    (bindings/metrics_exp_curve_door.mojo `_average_precision`): at most two
    classes in y_true take the binary curve over y_true == pos_label, more
    take the one-vs-rest curves over the classes y_true holds. No Python."""
    var caller = String("average_precision_score")
    var rank = List[Int32]()
    var present = _present(truth, n, k, rank)
    var t = _CRI32P(unsafe_from_address=truth)
    var plan = MetricProgramPlan()
    if present <= 2:
        if present == 2 and (positive < 0 or positive >= k or rank[positive] < 0):
            raise Error("pos_label=1 is not a valid label. It should be one of the labels y_true holds")
        if cols != 1:
            raise Error("mojolearn " + caller + ": y_score must be 1-D, got shape (" + String(n) + ", "
                        + String(cols) + ")")
        if not host_all_finite_f32(_CRF32P(unsafe_from_address=scores), n):
            raise Error("mojolearn " + caller + ": y_score must be finite")
        var flags = List[Int32](length=max(n, 1), fill=Int32(0))
        for i in range(n):  # driver-loop: over the rows (y_true == pos_label)
            flags[i] = Int32(1) if Int(t[i]) == positive else Int32(0)
        var out = ap_binary_stages(plan, n, w != 0)
        var addresses: List[Int] = [scores, Int(flags.unsafe_ptr())]
        var sizes: List[Int] = [n, n]
        if w != 0:
            addresses.append(w)
            sizes.append(n)
        var arena = metric_inputs_run(route, plan, addresses, sizes)
        _ = len(flags)  # copied into the program's arena above
        var zero = False
        var value = ap_fold_value(arena, out, zero)
        if zero:
            raise Error("ZeroDivisionError: float division by zero")
        return value
    if cols < 2:
        raise Error("mojolearn " + caller + ": y_score must be 2-D, got shape (" + String(n) + ",)")
    if not host_all_finite_f32(_CRF32P(unsafe_from_address=scores), n * cols):
        raise Error("mojolearn " + caller + ": y_score must be finite")
    if present != cols:
        raise Error("Number of classes in y_true not equal to the number of columns in 'y_score'")
    var codes = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # driver-loop: over the rows (the codes among the classes y_true holds)
        codes[i] = rank[Int(t[i])]
    var o = ap_ovr_stages(plan, n, cols, w != 0, True, False)
    var addresses: List[Int] = [scores, Int(codes.unsafe_ptr())]
    var sizes: List[Int] = [n * cols, n]
    if w != 0:
        addresses.append(w)
        sizes.append(n)
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(codes)  # copied into the program's arena above
    if not ap_ovr_edge(arena, o, cols):
        return arena.f64(o.res)
    var zero = False
    var per = ap_ovr_scores(arena, o, cols, zero)
    if zero:
        raise Error("ZeroDivisionError: float division by zero")
    var res = MX_NONE
    var p2 = ap_edge_average_plan(per, arena.i32_words(o.sup, 2 * cols), cols, False, res)
    var a2 = metric_inputs_run(route, p2, List[Int](), List[Int]())
    return a2.f64(res)


# ---------------------------------------------------------------- top_k_accuracy_score


def top_k_threshold(p: Int, n: Int) -> Float32:
    """The binary k=1 threshold: 0.5 for scores in [0, 1], else 0.0."""
    var s = _CRF32P(unsafe_from_address=p)
    return Float32(0.5) if (family_min_f32(s, n) >= 0 and family_max_f32(s, n) <= 1) else Float32(0.0)


def top_k_stages(mut plan: MetricProgramPlan, n: Int, binary: Bool, k: Int, cols: Int, threshold: Float32,
                 weighted: Bool) raises -> Tuple[Int, Int]:
    """The top-k program over its inputs: the scores (n x cols float32;
    binary only with k == 1, n words), the codes (n int32) and, weighted,
    the weights (n), in that order. (the hit total's slot, the weight
    total's or MX_NONE)."""
    var S: Int
    var kk: Int
    var c: Int
    if binary:
        if k == 1:
            var S1 = plan.input(n)
            S = plan.scratch(2 * n)
            plan.stage_op("pair_cols", n, [S1, S, curve_f32_bits(threshold)])
            kk = 1
            c = 2
        else:
            S = plan.scratch(2 * n)
            kk = 2
            c = 2
    else:
        S = plan.input(n * cols)
        kk = k
        c = cols
    var Y = plan.input(n)
    var hit = plan.scratch(n)
    plan.stage_op("row_metric", n, [S, c, Y, hit, _ROW_TOPK, kk, 0])
    var zero = plan.scratch(n)
    plan.stage_op("pair_key", n, [0, 0, zero, 1, 2])
    var W = MX_NONE
    if weighted:
        W = plan.input(n)
    var tot = plan.group(zero, n, 1, hit, 1, W, 1)[1]
    var sw = MX_NONE
    if weighted:
        sw = plan.group(zero, n, 1, MX_NONE, 1, W, 1)[1]
    return (tot, sw)


def top_k_reads_scores(binary: Bool, k: Int) -> Bool:
    """True when the top-k program takes the scores as an input."""
    return not binary or k == 1


def top_k_value(arena: MetricArenaOwner, tot: Int, sw: Int, n: Int, normalize: Bool) -> Float64:
    """The hits, or their fraction of the rows (of the weight total)."""
    var hits = arena.f32(tot)
    if not normalize:
        return hits
    if sw == MX_NONE:
        return hits / Float64(n)
    return hits / arena.f32(sw)


def top_k_run(route: MetricRoute, truth: Int, scores: Int, n: Int, cols: Int, classes: Int, k: Int,
              mut warn: Int) raises -> Float64:
    """`top_k_run` unweighted."""
    return top_k_run(route, truth, scores, n, cols, classes, k, 0, warn)


def top_k_run(route: MetricRoute, truth: Int, scores: Int, n: Int, cols: Int, classes: Int, k: Int, w: Int,
              mut warn: Int) raises -> Float64:
    """`w` (lane mf-score6): n float32 sample weights (0: none), the
    program's last input as the door pins it (`top_k_stages(weighted)`).
    THE TYPED RUN of `top_k_accuracy_score(y_true, y_score, k=k)` with
    its other defaults (normalize, labels None) over n int32 class codes in
    0 .. classes-1 and the member's float32 scores (n x cols; cols 1: the
    one column the scorer hands a binary member). The door's refusals in its
    order and words (bindings/metrics_exp_curve_door.mojo `_top_k`); `warn`
    1 where the door warns that k >= n_classes. No Python."""
    warn = 0
    if k < 1:
        raise Error("k must be a positive integer")
    var caller = String("top_k_accuracy_score")
    var rank = List[Int32]()
    var nc = _present(truth, n, classes, rank)
    if not host_all_finite_f32(_CRF32P(unsafe_from_address=scores), n * max(cols, 1)):
        raise Error("mojolearn " + caller + ": y_score must be finite")
    var binary = nc <= 2
    var two_d = cols > 1
    var n_score = cols if two_d else 2
    if binary and two_d:
        raise Error("`y_true` is binary while y_score is 2d with " + String(cols) + " classes. If `y_true` "
                    + "does not contain all the labels, `labels` must be provided.")
    if nc != n_score:
        raise Error("Number of classes in 'y_true' (" + String(nc) + ") not equal to the number of "
                    + "classes in 'y_score' (" + String(n_score) + ").")
    if k >= nc:
        warn = 1
    var t = _CRI32P(unsafe_from_address=truth)
    var codes = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # driver-loop: over the rows (the codes among the classes y_true holds)
        codes[i] = rank[Int(t[i])]
    var threshold = Float32(0.0)
    if binary and k == 1:
        threshold = top_k_threshold(scores, n)
    var plan = MetricProgramPlan()
    var got = top_k_stages(plan, n, binary, k, cols, threshold, w != 0)
    var addresses = List[Int]()
    var sizes = List[Int]()
    if top_k_reads_scores(binary, k):
        addresses.append(scores)
        sizes.append(n * cols)
    addresses.append(Int(codes.unsafe_ptr()))
    sizes.append(n)
    if w != 0:
        addresses.append(w)
        sizes.append(n)
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(codes)  # copied into the program's arena above
    return top_k_value(arena, got[0], got[1], n, True)
