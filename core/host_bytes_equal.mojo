# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Bitwise equality of two host byte ranges, across the host's cores (no
Python import). Moved here word for word from x_cnn/graph_run.mojo
(`graph_run_bytes_equal`, lanes fb-neural and fx-rows) by lane fx-rows-rest
(2026-11-18) so a second door (the isotonic knot record of
bindings/x_linear_native_door.mojo) uses the same compare without linking
the graph layers; `graph_run_bytes_equal` calls it.
"""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime _HBE_U8P = MutPointer[UInt8, MutUntrackedOrigin]


def host_bytes_equal(left: Int, right: Int, count: Int) -> Bool:
    """Whether the `count` bytes at address `left` equal those at `right`
    (bitwise: a NaN equals the same NaN bits).

    One read of both buffers split across the host pool; each task compares
    64-byte SIMD blocks and stops at its first differing block. Byte ranges
    are independent, so the split moves no verdict; a mismatch anywhere is
    False. A cache's hit test against the owned copy it took (the graph
    cache's edge buffers, lane fb-neural 2026-11-17; the isotonic knots,
    lane fx-rows-rest 2026-11-18)."""
    if count <= 0:
        return True
    # Units of whole 8-byte words: every unit's start is a multiple of 8
    # bytes from both bases (a cost split, from size alone; no shape rule).
    var words = count // 8
    var units = host_predict_task_count_work(max(words, 1), count)
    var chunk = host_predict_chunk(words, units)
    var failures = List[Int](length=units, fill=0)
    var flags = failures.unsafe_ptr()
    var lw = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=left)
    var rw = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=right)
    # fx-rows (2026-11-18): each task compares 64-byte SIMD blocks (eight
    # words per load, OR-reduced) and stops at the first differing block; the
    # word-at-a-time loop with a break per word did not vectorize. The test
    # is bitwise equality of the same bytes, so the verdict is unchanged.
    def task(unit: Int) {imm lw, imm rw, imm flags, imm chunk, imm words}:
        var differ = False
        var j = unit * chunk
        var hi = min((unit + 1) * chunk, words)
        while j + 8 <= hi:
            var l = lw.unsafe_load[width=8, alignment=1](j)
            var r = rw.unsafe_load[width=8, alignment=1](j)
            if l.ne(r).reduce_or():
                differ = True
                break
            j += 8
        while not differ and j < hi:  # small-loop(at most seven tail words of one task)
            if lw.unsafe_load[width=1, alignment=1](j) != rw.unsafe_load[width=1, alignment=1](j):
                differ = True
            j += 1
        flags[unit] = Int(differ)
    host_parallelize(task, units)
    for flag in failures:  # metadata: one status per independent byte range
        if flag != 0:
            return False
    var lb = _HBE_U8P(unsafe_from_address=left)
    var rb = _HBE_U8P(unsafe_from_address=right)
    for i in range(words * 8, count):  # small-loop(at most seven tail bytes)
        if lb[i] != rb[i]:
            return False
    return True
