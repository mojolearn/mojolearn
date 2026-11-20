# SPDX-License-Identifier: Apache-2.0
# Native metric stage/layout preparation. No Python objects, callbacks or provider
# imports: the boundary owns input references and supplies admitted opcode words.
from std.memory import bitcast

def metric_merge_pairs(var pairs: List[Int]) -> List[Int32]:
    # `_arena_io.output_ranges`: [lo, hi, -1, 1] quads of the sorted, merged
    # non-empty pairs.
    var n = len(pairs) // 2
    for i in range(1, n):  # driver-loop: insertion sort of the program's output range pairs
        var lo = pairs[2 * i]
        var hi = pairs[2 * i + 1]
        var j = i - 1
        while j >= 0 and (pairs[2 * j] > lo or (pairs[2 * j] == lo and pairs[2 * j + 1] > hi)):  # driver-loop: insertion shift over range pairs
            pairs[2 * j + 2] = pairs[2 * j]
            pairs[2 * j + 3] = pairs[2 * j + 1]
            j -= 1
        pairs[2 * j + 2] = lo
        pairs[2 * j + 3] = hi
    var out = List[Int32]()
    for i in range(n):  # driver-loop: merge of the sorted output range pairs
        var lo = pairs[2 * i]
        var hi = pairs[2 * i + 1]
        if hi <= lo:
            continue
        var m = len(out) // 4
        if m > 0 and lo <= Int(out[4 * m - 3]):
            out[4 * m - 3] = Int32(max(Int(out[4 * m - 3]), hi))
        else:
            out.append(Int32(lo))
            out.append(Int32(hi))
            out.append(-1)
            out.append(1)
    return out^


# The fixed x_metrics op and term codes, typed (lane metric-programs-2,
# 2026-11-12). They were the Python dicts `_expansion_metrics._OPS` / `_TERM`,
# copied into a per-program Python snapshot (bindings/metrics_plan_admission_
# unpack.mojo `MetricPlanOpcodes`, deleted) and admitted name by name. The
# codes are the stage ids x_metrics/units.mojo `run_unit` dispatches on
# (x_metrics/plan.mojo, tail.mojo, cm_epi.mojo, reg_epi.mojo, ... OP_*); the
# same numbers, so every stage record keeps its words.
def metric_opcode(name: String) raises -> Int:
    """The stage id of an x_metrics op name (the former `_OPS`)."""
    if name == "group_sort":
        return 0
    if name == "group_sum":
        return 1
    if name == "pair_key":
        return 2
    if name == "reg_term":
        return 3
    if name == "col_sort":
        return 4
    if name == "wpercentile":
        return 5
    if name == "col_max":
        return 6
    if name == "bin_curve":
        return 7
    if name == "row_metric":
        return 8
    if name == "row_centroid_dist":
        return 9
    if name == "permute":
        return 10
    if name == "fold_rows":
        return 36
    if name == "rows64":
        return 41
    if name == "strat_codes":
        return 45
    if name == "curve_fold":
        return 46
    if name == "onehot":
        return 52
    if name == "rep_rows":
        return 53
    if name == "pair_cols":
        return 54
    if name == "cls_epi":
        return 55
    if name == "off_diff":
        return 56
    if name == "flag_scan":
        return 57
    if name == "proba_rows":
        return 58
    if name == "cm_epi":
        return 59
    if name == "reg_epi":
        return 60
    if name == "rank_epi":
        return 61
    if name == "cl_epi":
        return 62
    if name == "curve_out":
        return 63
    if name == "auc_xy":
        return 67
    if name == "cont_stats":
        return 70
    if name == "approx_mode":
        return 81
    if name == "group_take":
        return 85
    if name == "take_rows64":
        return 86
    if name == "permute_drawn":
        return 87
    if name == "strat_take":
        return 88
    raise Error("KeyError: unknown x_metrics op '" + name + "'")


def metric_term(name: String) raises -> Int:
    """The reg_term kind of a term name (the former `_TERM`)."""
    if name == "sq":
        return 0
    if name == "abs":
        return 1
    if name == "sqlog":
        return 2
    if name == "ape":
        return 3
    if name == "pinball":
        return 4
    if name == "tweedie":
        return 5
    if name == "diff":
        return 6
    raise Error("KeyError: unknown x_metrics term '" + name + "'")


struct MetricProgramPlan(Movable):
    var size: Int
    var stages: List[Int32]
    var input_offsets: List[Int]
    var skip: List[Int]
    var outputs: List[Int]
    var declared_outputs: Bool
    # Per input record: the index of its program-owned constant words in
    # `consts`, or -1 for a caller buffer the boundary keeps alive (lane
    # entangled-programs: scalar words were a Python array each).
    var input_consts: List[Int]
    var consts: List[List[Int32]]

    def __init__(out self):
        self.size = 0
        # The ABI always receives a valid pointer, even for zero stages.
        self.stages = List[Int32](length=1, fill=0)
        self.input_offsets = List[Int]()
        self.skip = List[Int]()
        self.outputs = List[Int]()
        self.declared_outputs = False
        self.input_consts = List[Int]()
        self.consts = List[List[Int32]]()

    def alloc(mut self, n: Int) raises -> Int:
        var off = self.size
        self.size += max(n, 0)
        if self.size > 2147483647:
            raise Error("mojolearn metrics: the problem exceeds the Int32 arena bound")
        return off

    def input(mut self, n: Int) raises -> Int:
        var off = self.alloc(n)
        self.input_offsets.append(off)
        self.input_consts.append(-1)
        self.skip.append(off)
        self.skip.append(off + n)
        return off

    def input_words(mut self, var words: List[Int32]) raises -> Int:
        """An input of program-owned words (a scalar's float32 bits, a
        binary64 as two Int32 words, low first), copied in as their bits."""
        var off = self.input(len(words))
        self.input_consts[len(self.input_consts) - 1] = len(self.consts)
        self.consts.append(words^)
        return off

    def input_f32(mut self, value: Float64) raises -> Int:
        """`Array.from_list([_f32(value)], "<f4")` as one constant input word."""
        var words = List[Int32](length=1, fill=0)
        words[0] = bitcast[DType.int32](Float32(value))
        return self.input_words(words^)

    def input_f64(mut self, value: Float64) raises -> Int:
        """`_f64_words`: a binary64 scalar as two Int32 words, low first."""
        var bits = bitcast[DType.uint64](value)
        var words = List[Int32](length=2, fill=0)
        words[0] = bitcast[DType.int32](UInt32(bits & 0xFFFFFFFF))
        words[1] = bitcast[DType.int32](UInt32(bits >> 32))
        return self.input_words(words^)

    def buffer_inputs(self) -> Int:
        """The number of caller-buffer inputs (the boundary's owners)."""
        var count = 0
        for k in range(len(self.input_consts)):  # driver-loop: over the program's input records
            if self.input_consts[k] < 0:
                count += 1
        return count

    def const_address(self, k: Int) -> Int:
        """The address of input k's constant words (k must be a constant)."""
        return Int(self.consts[self.input_consts[k]].unsafe_ptr())

    def scratch(mut self, n: Int) raises -> Int:
        var off = self.alloc(n)
        if n > 0:
            self.skip.append(off)
            self.skip.append(off + n)
        return off

    def stage(mut self, opcode: Int, total: Int, params: List[Int]) raises:
        if len(params) > 14:
            raise Error("x_metrics: too many stage parameters")
        if len(self.stages) == 1:
            self.stages.clear()
        self.stages.append(Int32(opcode))
        self.stages.append(Int32(total))
        for word in params:  # small-loop(at most fourteen native stage parameters)
            self.stages.append(Int32(word))
        for _ in range(14 - len(params)):  # small-loop(fixed stage record padding)
            self.stages.append(0)

    def stage_op(mut self, op: String, total: Int, params: List[Int]) raises:
        """`stage` by op name, through the typed op table."""
        self.stage(metric_opcode(op), total, params)

    def input_f64s(mut self, values: List[Float64]) raises -> Int:
        """Binary64 values as program-owned Int32 word pairs, low first (was
        a Python int32 Array per call: `mc_f64_words`, `_f64_words`)."""
        var words = List[Int32](length=2 * len(values), fill=0)
        for i in range(len(values)):  # driver-loop: over a stage's binary64 parameters
            var bits = bitcast[DType.uint64](values[i])
            words[2 * i] = bitcast[DType.int32](UInt32(bits & 0xFFFFFFFF))
            words[2 * i + 1] = bitcast[DType.int32](UInt32(bits >> 32))
        return self.input_words(words^)

    def group(mut self, key: Int, n: Int, m: Int, values: Int, vstride: Int, weights: Int,
              width: Int) raises -> Tuple[Int, Int]:
        """`_group`: the stable counting sort and the PairSums; (OFF, OUT)."""
        var off = self.alloc(m + 1)
        var order = self.scratch(n)
        var out = self.alloc(m * width)
        self.stage_op("group_sort", 1, [key, n, m, off, order])
        if m * width:
            self.stage_op("group_sum", m * width, [off, order, values, vstride, weights, out, width])
        return (off, out)

    def counts(mut self, off: Int, m: Int) raises -> Int:
        """`_counts`: the m group sizes as an Int32 slot (off_diff)."""
        var out = self.alloc(m)
        if m:
            self.stage_op("off_diff", m, [off, m, out])
        return out

    def scan_flag(mut self, src: Int, n: Int, mode: Int) raises -> Int:
        """`_scan_flag`: one flag word set by the flag_scan test over n words."""
        var flag = self.alloc(1)
        if n:
            self.stage_op("flag_scan", n, [src, n, mode, flag])
        return flag

    def want(mut self, off: Int, n: Int, count: Int, mult: Int) -> Int:
        self.declared_outputs = True
        if n > 0:
            self.outputs.append(off)
            self.outputs.append(off + n)
            self.outputs.append(count)
            self.outputs.append(mult)
        return off

    def download(self, mut known: Bool) -> List[Int32]:
        known = True
        if self.declared_outputs:
            var plain = List[Int]()
            for i in range(len(self.outputs) // 4):  # driver-loop: native declared output spans
                if self.outputs[4 * i + 2] < 0:
                    plain.append(self.outputs[4 * i])
                    plain.append(self.outputs[4 * i + 1])
            var result = metric_merge_pairs(plain^)
            for i in range(len(self.outputs) // 4):  # driver-loop: counted ranges retain declaration order
                if self.outputs[4 * i + 2] >= 0:
                    for j in range(4):
                        result.append(Int32(self.outputs[4 * i + j]))
            return result^
        if len(self.skip) == 0:
            known = False
            return List[Int32]()
        var pairs = self.skip.copy()
        var n = len(pairs) // 2
        for i in range(1, n):  # driver-loop: original insertion sort of input/scratch spans
            var lo = pairs[2 * i]
            var hi = pairs[2 * i + 1]
            var j = i - 1
            while j >= 0 and (pairs[2 * j] > lo or (pairs[2 * j] == lo and pairs[2 * j + 1] > hi)):
                pairs[2 * j + 2] = pairs[2 * j]
                pairs[2 * j + 3] = pairs[2 * j + 1]
                j -= 1
            pairs[2 * j + 2] = lo
            pairs[2 * j + 3] = hi
        var result = List[Int32]()
        var at = 0
        for i in range(n):  # driver-loop: same complement of input/scratch spans
            if pairs[2 * i] > at:
                result.append(Int32(at))
                result.append(Int32(pairs[2 * i]))
                result.append(-1)
                result.append(1)
            at = max(at, pairs[2 * i + 1])
        if at < self.size:
            result.append(Int32(at))
            result.append(Int32(self.size))
            result.append(-1)
            result.append(1)
        return result^
