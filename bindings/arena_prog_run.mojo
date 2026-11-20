# SPDX-License-Identifier: Apache-2.0
"""RUN of the arena program runner (bindings/arena_prog_native.mojo): the
switch read and the typed input/output range descriptors. No Python import
(tools/python_free_dirs.sh). Lane rs-shared-infra-b (2026-11-12): moved word
for word; descriptor lists of one program, nothing to split across cores."""
from std.os import getenv
from std.memory import unsafe_memcpy, bitcast
from core.metrics_program_plan import MetricProgramPlan
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from x_metrics.descriptor_schema import MP_CAPABILITIES, MP_HAS_RANGES, MP_RUN, MP_RUN_OUT, MP_RUN_RANGES

comptime _RUN_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _STAGE_WORDS = 16


def _switch(name: String) -> Bool:
    # Unset or anything but "0": on (the Python `_switch` / `ranges_enabled`).
    return String(getenv(name, "1").strip()) != "0"


def _add_input(mut ins: List[Int32], lo: Int, hi: Int, src: Int) raises:
    # `_arena_io.input_ranges`: ascending disjoint spans, adjacent host spans
    # (src -1) merged; an overlap is a layout bug.
    var count = len(ins) // 3
    if count > 0:
        var last_hi = Int(ins[3 * count - 2])
        if lo < last_hi:
            raise Error("AssertionError: arena ranges: inputs overlap at [" + String(lo) + ", " + String(hi) + ")")
        if src < 0 and Int(ins[3 * count - 1]) < 0 and lo == last_hi:
            ins[3 * count - 2] = Int32(hi)
            return
    ins.append(Int32(lo))
    ins.append(Int32(hi))
    ins.append(Int32(src))


def _merge_pairs(var pairs: List[Int]) -> List[Int32]:
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


def _pad(var words: List[Int32], fill: List[Int32]) -> List[Int32]:
    # A packed descriptor list is never empty (the entry reads a pointer).
    if len(words) == 0:
        return fill.copy()
    return words^


# ---------------------------------------------------------------- x_metrics, typed
# Lane entangled-programs (2026-11-12): the RUN of a native metric program
# (`_metrics_run_plan`'s execution). The door (bindings/arena_prog_native.mojo)
# admits the inputs to (offset, address, words) records, keeps their Python
# owners alive and releases the GIL around this call; the program runs through
# the image's `metrics_program_native_api_v1` dispatch (x_metrics/
# descriptor_schema.mojo) instead of the Python `x_metrics_run*` entries: the
# same executors (descriptor_host / descriptor_device call the bodies the Python
# entries called), the same words, the same route choice. The arena is a native
# store owned by the result (was a Python `array.array`).


@fieldwise_init
struct MetricProgramInput(ImplicitlyCopyable, Movable):
    """One admitted input of a metric program: its arena offset, the address
    of its words (the door keeps the owner alive) and its word count."""
    var offset: Int
    var address: Int
    var words: Int


struct MetricArenaOwner(Copyable, Movable):
    """A completed metric program's arena: zeroed float32 words (at least one,
    as the Python store was), private to the operation. The base address is
    read from the store itself, so a copy reads its own words."""
    var words: List[Float32]

    def __init__(out self, size: Int):
        self.words = List[Float32](length=max(size, 1), fill=0.0)

    def base(self) -> Int:
        return Int(self.words.unsafe_ptr())

    # Typed reads of the completed arena (lane metric-programs-2, 2026-11-12:
    # one reader set for every metric owner; were MxArena's methods and the
    # mc_arena_* / _arena_* copies in bindings/mcls_common.mojo and
    # bindings/metrics_reg_door.mojo). Each read takes the base from the
    # store this value owns.
    def i32(self, off: Int) -> Int:
        """`prog.ints(off, 1)[0]`: the Int32 word at arena[off]."""
        return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=self.base() + 4 * off)[0])

    def i64(self, off: Int) -> Int:
        """`prog.words(off, 2, "q")[0]`: the Int64 at arena[off], arena[off + 1], low first."""
        var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=self.base() + 4 * off)
        return Int(bitcast[DType.int64](UInt64(words[0]) | (UInt64(words[1]) << 32)))

    def f32(self, off: Int) -> Float64:
        """`prog.floats(off, 1)[0]`: the Float32 word at arena[off], widened exactly."""
        return Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.base() + 4 * off)[0])

    def f64(self, off: Int) -> Float64:
        """`prog.words(off, 2, "d")[0]`: the binary64 at arena[off], arena[off + 1], low first."""
        var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=self.base() + 4 * off)
        return bitcast[DType.float64](UInt64(words[0]) | (UInt64(words[1]) << 32))

    def i32_words(self, off: Int, count: Int) -> List[Int32]:
        """A copy of `count` arena words as Int32 bits (a later program's
        program-owned input; was an Array PACK read back as a caller buffer)."""
        var copied = List[Int32](length=count, fill=0)
        if count > 0:
            unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(copied.unsafe_ptr())),
                          src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=self.base() + 4 * off),
                          count=count)
        return copied^


def metrics_plan_execute(entry: NativeDescriptorCall, plan: MetricProgramPlan, inputs: List[MetricProgramInput],
                         ranges_on: Bool, output_ranges: Bool) raises -> MetricArenaOwner:
    """RUN: copy the inputs into a fresh arena and run the plan's stages.
    `ranges_on`: MOJOLEARN_ARENA_RANGES and the image has the range route
    (device images; MP_CAPABILITIES is read here as well); `output_ranges`:
    the image has the output-range route. No Python object is read."""
    var arena = MetricArenaOwner(plan.size)
    var base = arena.base()
    var ins = List[Int32]()
    for k in range(len(inputs)):  # driver-loop: over the program's input buffers
        var record = inputs[k]
        if record.words == 0:
            continue
        unsafe_memcpy(dest=_RUN_F32P(unsafe_from_address=base + 4 * record.offset),
                      src=_RUN_F32P(unsafe_from_address=record.address), count=record.words)
        _add_input(ins, record.offset, record.offset + record.words, -1)
    var known = True
    var merged = plan.download(known)
    var nst = len(plan.stages) // _STAGE_WORDS
    var pa = Int(plan.stages.unsafe_ptr())
    var size = plan.size
    var ranges = ranges_on and (native_descriptor_call(entry, MP_CAPABILITIES, [], [], []) & MP_HAS_RANGES) != 0
    if ranges:
        # the shared ranges runner (core/arena_io.mojo): only the inputs go
        # up, only the outputs come back
        var outs = merged.copy() if known else [Int32(0), Int32(size), Int32(-1), Int32(1)]
        var n_in = len(ins) // 3
        var n_out = len(outs) // 4
        var ia = _pad(ins.copy(), [Int32(0), Int32(0), Int32(-1)])
        var oa = _pad(outs^, [Int32(0), Int32(0), Int32(-1), Int32(1)])
        _ = native_descriptor_call(entry, MP_RUN_RANGES, [base, pa, Int(ia.unsafe_ptr()), Int(oa.unsafe_ptr())],
                                   [size, nst, n_in, n_out], [])
        # The address words are not owners: keep both lists live past the call.
        _ = ia^
        _ = oa^
    elif known and output_ranges:
        var n_out = len(merged) // 4
        var oa = _pad(merged.copy(), [Int32(0), Int32(0), Int32(-1), Int32(1)])
        _ = native_descriptor_call(entry, MP_RUN_OUT, [base, pa, Int(oa.unsafe_ptr())], [size, nst, n_out], [])
        _ = oa^
    else:
        _ = native_descriptor_call(entry, MP_RUN, [base, pa], [size, nst], [])
    _ = ins^
    return arena^
