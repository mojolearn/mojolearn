# SPDX-License-Identifier: Apache-2.0
# RUN of accuracy_score, confusion_matrix, precision/recall/f-score and
# log_loss (bindings/mcls_classify_door.mojo): the typed sums records, the
# COUNT tail, the probabilities' and row-mean stages and the post-run flag
# read. No Python import (tools/python_free_dirs.sh). Lane
# metric-programs-2 (2026-11-12): moved from mcls_classify.mojo, where they
# built on the Python-holding `McProg`; same stage records in the same order.
from core.metrics_program_plan import MetricProgramPlan
from bindings.arena_prog_run import MetricArenaOwner
from bindings.mcls_common_run import MC_NONE

# cm_epi kinds (x_metrics/cm_epi.mojo, op 59) and the NORM modes.
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
