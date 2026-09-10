# SPDX-License-Identifier: Apache-2.0
"""Exact logical-group UInt64 minimum, independent of physical subgroup width.

Only integer minimum may change its tree here. Floating-point sums must retain
an independently specified arithmetic order. All block threads must call this
primitive convergently, and WIDTH must divide the block size.
"""
from std.gpu import thread_idx
from max.gpu.sync import barrier
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from std.gpu.primitives.warp import shuffle_xor


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


@always_inline
def logical_min_u64[WIDTH: Int, BLOCK: Int, NATIVE: Int, FIXED: Bool](
    value: UInt64,
) -> UInt64:
    """Shared fallback also handles logical groups spanning physical subgroups.

    Scratch contains BLOCK entries. The final barrier permits immediate scratch
    reuse by the next reduction; omitting it races slower readers. NATIVE is
    ignored for variable-width devices: a declared floor is not a guarantee.
    """
    comptime assert WIDTH > 0 and (WIDTH & (WIDTH - 1)) == 0
    comptime assert BLOCK >= WIDTH and BLOCK % WIDTH == 0
    comptime if FIXED and WIDTH <= NATIVE and NATIVE % WIDTH == 0:
        return shuffle_min_u64[WIDTH](value)
    else:
        var scratch = stack_allocation[BLOCK, UInt64, address_space=AddressSpace.SHARED]()
        var tid = Int(thread_idx.x)
        var lane = tid % WIDTH
        var base = tid - lane
        scratch.unsafe_store(tid, value)
        barrier()
        var stride = WIDTH // 2
        while stride > 0:
            if lane < stride:
                var a = scratch.unsafe_load(tid)
                var b = scratch.unsafe_load(tid + stride)
                scratch.unsafe_store(tid, b if b < a else a)
            barrier()
            stride //= 2
        var result = scratch.unsafe_load(base)
        barrier()
        return result
