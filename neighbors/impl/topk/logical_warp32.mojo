# SPDX-License-Identifier: Apache-2.0
"""Keep a32-lane queue topology inside each half of an IDENTICAL CDNA wave.

This changes communication scope, never floating arithmetic or queue order.
FAST/DETERMINISTIC retain their prior physical-width calls and entry refusal.
The caller must converge each logical group; different halves may diverge.
"""
from std.gpu.primitives.id import lane_id
from std.gpu.primitives.warp import max as warp_max
from std.gpu.primitives.warp import shuffle_idx, shuffle_xor
from checks.kernel_matrix import TARGET_COLUMN, lib_lane_width_for
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime LOGICAL32_ON64 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and lib_lane_width_for[TARGET_COLUMN]() == 64


@always_inline
def logical32_source_lane(physical_lane: UInt32, source: UInt32) -> UInt32:
    return (physical_lane & UInt32(0xffffffe0)) | source


@always_inline
def queue_lane_id() -> UInt32:
    comptime if LOGICAL32_ON64:
        return UInt32(lane_id()) & UInt32(31)
    return UInt32(lane_id())


@always_inline
def queue_mask() -> UInt:
    return UInt(0xffffffff) << UInt(UInt32(lane_id()) & UInt32(0xffffffe0))


@always_inline
def queue_shuffle_xor[dtype: DType, //](value: SIMD[dtype, 1], offset: UInt32) -> SIMD[dtype, 1]:
    comptime if LOGICAL32_ON64:
        return shuffle_xor(queue_mask(), value, offset)
    return shuffle_xor(value, offset)


@always_inline
def queue_any(value: Int32) -> Int32:
    comptime if LOGICAL32_ON64:
        var result = value
        var offset = 1
        while offset < 32:
            var other = queue_shuffle_xor(result, UInt32(offset))
            if other > result:
                result = other
            offset *= 2
        return result
    return warp_max(value)


@always_inline
def queue_broadcast(value: Float32, source: UInt32) -> Float32:
    comptime if LOGICAL32_ON64:
        return shuffle_idx(queue_mask(), value, logical32_source_lane(UInt32(lane_id()), source))
    return shuffle_idx(value, source)
