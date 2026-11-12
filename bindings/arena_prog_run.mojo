# SPDX-License-Identifier: Apache-2.0
"""RUN of the arena program runner (bindings/arena_prog_native.mojo): the
switch read and the typed input/output range descriptors. No Python import
(tools/python_free_dirs.sh). Lane rs-shared-infra-b (2026-11-12): moved word
for word; descriptor lists of one program, nothing to split across cores."""
from std.os import getenv


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
