# SPDX-License-Identifier: Apache-2.0
"""Exact logical-group UInt64 minimum, independent of physical subgroup width.

Only integer minimum may change its tree here. Floating-point sums must retain
an independently specified arithmetic order. All block threads must call this
primitive convergently, and WIDTH must divide the block size.
"""
from max.gpu import thread_idx
from max.gpu.sync import barrier
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.primitives.warp import shuffle_xor


@always_inline
def shuffle_min_u64[WIDTH: Int](value: UInt64) -> UInt64:
    """WIDTH must divide, and not exceed, the actual hardware subgroup width."""
    comptime assert WIDTH > 0 and (WIDTH & (WIDTH - 1)) == 0
    var result = value
    var offset = 1
    while offset < WIDTH:
        var hi = shuffle_xor(UInt32(result >> UInt64(32)), UInt32(offset))
        var lo = shuffle_xor(UInt32(result & UInt64(0xffffffff)), UInt32(offset))
        var other = (UInt64(hi) << UInt64(32)) | UInt64(lo)
        if other < result:
            result = other
        offset *= 2
    return result

