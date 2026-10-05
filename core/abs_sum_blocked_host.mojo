# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""cpu2-l6-bindings (2026-10-04): the host column of `core/abs_sum_blocked`
(the blocked binary64 sum of |v|), with no GPU import so host-only
bindings can use it. The order is documented in `core/abs_sum_blocked.mojo`.
"""
from std.memory import bitcast

from checks.soft_f64 import SF64_ZERO, sf64_add, sf64_from_f32

#: rows per chunk (the serial leg of the order); part of the contract.
comptime ABS_SUM_CHUNK = 256


@always_inline
def abs_word_f64(v: Float32) -> UInt64:
    """`|v|` widened exactly to binary64 words (sign bit cleared)."""
    return sf64_from_f32(
        bitcast[DType.float32](bitcast[DType.uint32](v) & UInt32(0x7FFFFFFF))
    )


def host_abs_sum_blocked(addr: Int, n: Int) -> Float64:
    """The host column's restatement of `device_abs_sum_blocked`: the same
    chunks, the same pairwise levels, the same `sf64_add` words. `addr` is
    a Float32 buffer of `n` values."""
    if n <= 0:
        return Float64(0)
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)
    var m = (n + ABS_SUM_CHUNK - 1) // ABS_SUM_CHUNK
    var level = List[UInt64](capacity=m)
    for c in range(m):
        var lo = c * ABS_SUM_CHUNK
        var hi = min(lo + ABS_SUM_CHUNK, n)
        var acc = SF64_ZERO
        for r in range(lo, hi):
            acc = sf64_add(acc, abs_word_f64(src.unsafe_load(r)))
        level.append(acc)
    while m > 1:
        var half = (m + 1) // 2
        var nxt = List[UInt64](capacity=half)
        for i in range(half):
            if 2 * i + 1 < m:
                nxt.append(sf64_add(level[2 * i], level[2 * i + 1]))
            else:
                nxt.append(level[2 * i])
        level = nxt^
        m = half
    return bitcast[DType.float64](level[0])
