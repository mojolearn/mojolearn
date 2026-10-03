# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One-block integer scans the x_linear grid drivers share (moved out of
x_linear/device.mojo by cgr-linear so x_linear/logcv_grid.mojo reaches them)."""
from std.gpu import thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.ops import IP


# ---------------------------------------------- parallel block scans (cpu-gpu-cleanup c-linear)
# Integer prefix sums, so the order of the adds never shows in a word.
comptime SC_NT = 256


@always_inline
def _sc_block_excl(src: IP, dst: IP, lo: Int, cnt: Int, base: Int32,
                   part: UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]) -> Int32:
    """dst[lo + i] = base + sum(src[lo : lo + i]) for i < cnt, on one block
    of SC_NT threads (each owns a contiguous chunk; src may be dst); returns
    the row's total. Every thread of the block calls it."""
    var tid = Int(thread_idx.x)
    var ch = (cnt + SC_NT - 1) // SC_NT
    var a = lo + min(tid * ch, cnt)
    var b = lo + min(tid * ch + ch, cnt)
    var s = Int32(0)
    for i in range(a, b):
        s += src.unsafe_load(i)
    part[tid] = s
    barrier()
    var off = 1
    while off < SC_NT:
        var v = part[tid] + (part[tid - off] if tid >= off else Int32(0))
        barrier()
        part[tid] = v
        barrier()
        off *= 2
    var acc = base + part[tid] - s
    var total = part[SC_NT - 1]
    barrier()
    for i in range(a, b):
        var c = src.unsafe_load(i)
        dst.unsafe_store(i, acc)
        acc += c
    return total


@always_inline
def _sc_block_sum(src: IP, cnt: Int,
                  part: UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]) -> Int32:
    """sum(src[0:cnt]) on one block of SC_NT threads (a strided pass, then a
    halving tree in shared memory)."""
    var tid = Int(thread_idx.x)
    var s = Int32(0)
    for i in range(tid, cnt, SC_NT):
        s += src.unsafe_load(i)
    part[tid] = s
    barrier()
    var h = SC_NT // 2
    while h > 0:
        if tid < h:
            part[tid] = part[tid] + part[tid + h]
        barrier()
        h //= 2
    var total = part[0]
    barrier()
    return total


