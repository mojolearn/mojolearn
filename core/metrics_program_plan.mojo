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
