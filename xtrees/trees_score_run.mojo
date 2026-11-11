# SPDX-License-Identifier: Apache-2.0
"""RUN of the tree ensembles' `score` door (bindings/trees_public_score.mojo):
the x_metrics COUNT program of `accuracy_score` as typed words, and the
read of its answer from the arena. The door's UNPACK encodes the labels and
weights into int32 / float32 Arrays; its PACK hands this program's words to
the shared native program runner (bindings/arena_prog_native.mojo
`_metrics_run`, which takes the program as a Python object today) and
passes back the arena address. No Python import:
`tools/python_free_dirs.sh` covers this directory.

Lane doorsplit-trees (2026-11-11): moved from the door's `_Program` and
`_accuracy_encoded`; the same stages, words and offsets in the same order
(no bit change). Building the program is a few dozen words; the program
runs as device launches in the runner.
"""
from std.memory import bitcast

# x_metrics stage ops (`_expansion_metrics._OPS`) and cm_epi kinds.
comptime SCORE_OP_GROUP_SORT = 0
comptime SCORE_OP_GROUP_SUM = 1
comptime SCORE_OP_PAIR_KEY = 2
comptime SCORE_OP_OFF_DIFF = 56
comptime SCORE_OP_CM_EPI = 59
comptime SCORE_CM_COUNT = 2
comptime SCORE_CM_WSUM_LEAF = 7
comptime SCORE_CM_WSUM_FOLD = 8
comptime SCORE_WSUM_BLOCK = 256
comptime SCORE_NONE = -1
comptime SCORE_PARAMS = 14
comptime SCORE_INT32_MAX = 2147483647
# One stage record: op, total, then SCORE_PARAMS parameter words.
comptime SCORE_STAGE_WORDS = SCORE_PARAMS + 2


def score_shape_text(shape: List[Int]) -> String:
    """The Python repr of a shape tuple of ints: `()`, `(3,)`, `(3, 4)`."""
    var text = String("(")
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        if k > 0:
            text += ", "
        text += String(shape[k])
    if len(shape) == 1:
        text += ","
    return text + ")"


struct ScoreProgram(Movable):
    """The fields of an `_expansion_metrics._Prog` the native runner reads:
    its size, the arena offsets of its inputs (in `put` order; the boundary
    pairs each with its Array), the stage records (SCORE_STAGE_WORDS words
    each) and the skipped ranges (start, end pairs)."""
    var size: Int
    var input_offsets: List[Int]
    var stages: List[Int]
    var skip: List[Int]

    def __init__(out self):
        self.size = 0
        self.input_offsets = List[Int]()
        self.stages = List[Int]()
        self.skip = List[Int]()

    def stage_count(self) -> Int:
        return len(self.stages) // SCORE_STAGE_WORDS

    def alloc(mut self, n: Int) raises -> Int:
        var off = self.size
        self.size += max(n, 0)
        if self.size > SCORE_INT32_MAX:
            raise Error("mojolearn metrics: the problem exceeds the Int32 arena bound")
        return off

    def put(mut self, count: Int) raises -> Int:
        """`put` / `put_i32` of a C-order float32 or int32 Array of
        `count` words: its offset."""
        var off = self.alloc(count)
        self.input_offsets.append(off)
        self.skip.append(off)
        self.skip.append(off + count)
        return off

    def scratch(mut self, n: Int) raises -> Int:
        var off = self.alloc(n)
        if n > 0:
            self.skip.append(off)
            self.skip.append(off + n)
        return off

    def stage(mut self, op: Int, total: Int, params: List[Int]) raises:
        if len(params) > SCORE_PARAMS:
            raise Error("x_metrics: too many stage parameters")
        self.stages.append(op)
        self.stages.append(total)
        for i in range(SCORE_PARAMS):  # small-loop(params): the fourteen fixed parameter words of one stage
            self.stages.append(params[i] if i < len(params) else 0)

    def group(mut self, key: Int, n: Int, m: Int, weights: Int) raises -> Tuple[Int, Int]:
        """`_expansion_metrics._group` (values=_NONE, vstride=1, width=1)."""
        var off = self.alloc(m + 1)
        var order = self.scratch(n)
        var out = self.alloc(m)
        self.stage(SCORE_OP_GROUP_SORT, 1, [key, n, m, off, order])
        if m:
            self.stage(SCORE_OP_GROUP_SUM, m, [off, order, SCORE_NONE, 1, weights, out, 1])
        return (off, out)

    def counts(mut self, off: Int, m: Int) raises -> Int:
        """`_expansion_metrics._counts`."""
        var out = self.alloc(m)
        if m:
            self.stage(SCORE_OP_OFF_DIFF, m, [off, m, out])
        return out


struct AccuracyPlan(Movable):
    """The COUNT program of one accuracy score and the arena offset of its
    eleven answer words. Inputs: true codes, predicted codes, then the
    weights when weighted."""
    var program: ScoreProgram
    var out: Int
    var weighted: Bool

    def __init__(out self, var program: ScoreProgram, out_offset: Int, weighted: Bool):
        self.program = program^
        self.out = out_offset
        self.weighted = weighted


def score_accuracy_check(caller: String, n_true: Int, n_pred: Int, kind_true: String, kind_pred: String) raises:
    """`_accuracy_encoded`'s pair refusals."""
    if n_true != n_pred:
        raise Error("mojolearn " + caller + ": y_true and y_pred lengths differ")
    if kind_true != kind_pred:
        raise Error("TypeError: mojolearn " + caller + ": y_true and y_pred must use the same label type")
    if n_true > SCORE_INT32_MAX:
        raise Error("mojolearn " + caller + ": at most INT32_MAX rows")


def score_accuracy_plan(n: Int, classes: Int, weighted: Bool) raises -> AccuracyPlan:
    """`_Sums(true, pred, w, present, mode, tail=_cm_stage(..., COUNT, 11))`
    over `n` rows and `classes` present labels."""
    var L = classes
    var prog = ScoreProgram()
    var a = prog.put(n)
    var bb = prog.put(n)
    var matched = prog.scratch(n)
    prog.stage(SCORE_OP_PAIR_KEY, n, [a, bb, matched, L, 1])
    var W = SCORE_NONE
    if weighted:
        W = prog.put(n)
    # The COUNT epilogue (x_metrics/cm_epi.mojo `_count`) reads only the
    # matched grouping's counts when unweighted, and only the folded wsum
    # slot when weighted (TP is not read). The y_true and y_pred groupings,
    # their counts and the weighted total grouping were queued and never
    # read: two to four O(n) device sorts per score (p3-trees). Only the
    # grouping the epilogue reads is queued; the result is the same.
    var tp = matched
    if not weighted:
        var g0 = prog.group(matched, n, L, W)
        _ = prog.counts(g0[0], L)
        tp = g0[0]
    # `_cm_stage(prog, groups, w, n, L, _CM_COUNT, 11)`.
    var out = prog.alloc(11)
    var tot = n
    if weighted:
        # `_wsum_sf64(prog, L)`: the soft binary64 hit and weight total.
        var nb = max(1, (n + SCORE_WSUM_BLOCK - 1) // SCORE_WSUM_BLOCK)
        var s = prog.scratch(4 * nb)
        prog.stage(SCORE_OP_CM_EPI, nb, [SCORE_CM_WSUM_LEAF, matched, W, n, s, L])
        var stride = 1
        while stride < nb:  # driver-loop: over the fold levels (log2 of the block count)
            prog.stage(SCORE_OP_CM_EPI, (nb - stride + 2 * stride - 1) // (2 * stride), [SCORE_CM_WSUM_FOLD, s, stride, nb])
            stride *= 2
        tot = s
    prog.stage(SCORE_OP_CM_EPI, 1, [SCORE_CM_COUNT, tp, L, 1 if weighted else 0, tot, out])
    return AccuracyPlan(prog^, out, weighted)


def score_accuracy_answer(base: Int, word: Int, weighted: Bool) raises -> Float64:
    """The fraction the COUNT epilogue wrote at arena word `word` (base: the
    arena's address); a zero weight total raises ZeroDivisionError as the
    division did."""
    if weighted and Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=base + 4 * word)[0]) != 0:
        raise Error("ZeroDivisionError: float division by zero")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=base + 4 * (word + 8))[0]
