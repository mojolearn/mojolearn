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
from bindings.arena_prog_run import MetricArenaOwner, MetricRoute, metric_inputs_run
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


# ---------------------------------------------------------------- brier_score_loss, typed
# Lane mf-score5 (2026-11-23): `_brier`'s program as one builder (the door
# pins its inputs in the builder's order; the search scorers' RUN calls it
# over typed buffers). The same stages in the same order: the bits do not
# move.

comptime _RKF32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _RKI32P = MutPointer[Int32, MutUntrackedOrigin]


def brier_stages(mut plan: MetricProgramPlan, shape: ProbaShape, weighted: Bool,
                 half: Bool) raises -> Tuple[Int, Int]:
    """`_brier`'s program over its inputs: the probabilities (n words for
    the one-column form, else n x k), the codes (n int32) and, weighted, the
    weights (n), in that order. (the proba FLAGS slot, the result offset)."""
    var A = plan.input(shape.n if shape.binary else shape.n * shape.k)
    var put = proba_stages(plan, A, shape)
    var Y = plan.input(shape.n)
    var res = row_mean_stage(plan, put[0], shape.k, Y, shape.n, _ROW_BRIER, weighted, 0, MX_NONE, True, half)
    return (put[1], res)


#: `brier_run`'s label marks (two bits per class code, `marks >> 2c & 3`): a
#: class whose label is the integer 0, 1 or -1 (other labels: 0). The UNPACK
#: writes them once from the fit's labels; `_pos_label(None, ...)` reads them.
comptime BRIER_MARK_ZERO = 1
comptime BRIER_MARK_ONE = 2
comptime BRIER_MARK_MINUS_ONE = 3


def brier_marks_positive(marks: Int, rank: List[Int32], k: Int) -> Int:
    """`_pos_label(None, kind, present)` over class codes: when the labels
    y_true holds are {0}, {1}, {-1}, {0, 1} or {-1, 1} (integers), the code
    of the label 1 (-1: 1 is not a label: no row is positive); else the
    code of the largest label y_true holds. Negative `marks`: none marked."""
    var m = marks if marks > 0 else 0
    var seen = List[Int]()
    var last = -1
    for c in range(k):  # small-loop(the class codes y_true holds, in label order)
        if rank[c] >= 0:
            seen.append((m >> (2 * c)) & 3)
            last = c
    var ok = False
    if len(seen) == 1:
        ok = seen[0] != 0
    elif len(seen) == 2:
        ok = (seen[0] == BRIER_MARK_ZERO or seen[0] == BRIER_MARK_MINUS_ONE) and seen[1] == BRIER_MARK_ONE
    if not ok:
        return last
    for c in range(k):  # small-loop(the class codes: the label 1's)
        if ((m >> (2 * c)) & 3) == BRIER_MARK_ONE:
            return c
    return -1


def brier_run(route: MetricRoute, truth: Int, proba: Int, n: Int, k: Int, cols: Int, marks: Int) raises -> Float64:
    """THE TYPED RUN of `brier_score_loss(y_true, y_proba)` with its
    defaults (pos_label None, labels None, scale_by_half 'auto') over n
    int32 class codes in 0 .. k-1 and the member's float32 probabilities
    (cols 1: the one column the scorer hands a binary member, `predict_proba[:, 1]`;
    else n x cols), `marks` the labels' marks (`brier_marks_positive`). The
    door's refusals in its order and words (bindings/metrics_exp_rank_door.mojo
    `_proba`, `_brier`). No Python."""
    var caller = String("brier_score_loss")
    var t = _RKI32P(unsafe_from_address=truth)
    var rank = List[Int32](length=max(k, 1), fill=Int32(-1))
    for i in range(n):  # driver-loop: over the rows (the classes y_true holds)
        var c = Int(t[i])
        if c < 0 or c >= k:
            raise Error("ValueError: y codes outside [0, n_classes)")
        rank[c] = 0
    var present = 0
    for c in range(k):  # small-loop(the class codes: their ranks among the present)
        if rank[c] == 0:
            rank[c] = Int32(present)
            present += 1
    var binary = cols == 1
    var codes = List[Int32](length=max(n, 1), fill=Int32(0))
    var width = 2
    if binary:
        if present > 2:
            raise Error("The type of the target inferred from y_true is multiclass but should be binary "
                        + "according to the shape of y_prob.")
        var positive = brier_marks_positive(marks, rank, k)
        for i in range(n):  # driver-loop: over the rows (y_true == pos_label)
            codes[i] = Int32(1) if Int(t[i]) == positive else Int32(0)
    else:
        if present != cols:
            raise Error("y_true and y_proba contain different number of classes: " + String(present)
                        + " vs " + String(cols))
        width = cols
        for i in range(n):  # driver-loop: over the rows (the codes among the classes y_true holds)
            codes[i] = rank[Int(t[i])]
    var shape = ProbaShape(n, width, binary)
    var plan = MetricProgramPlan()
    var got = brier_stages(plan, shape, False, binary or width < 3)
    var addresses: List[Int] = [proba, Int(codes.unsafe_ptr())]
    var sizes: List[Int] = [n if binary else n * width, n]
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(codes)  # copied into the program's arena above
    proba_check(arena, got[0], caller)
    return arena.f64(got[1])
