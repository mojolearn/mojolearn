# SPDX-License-Identifier: Apache-2.0
"""Unqualified small-k integer-key selector; no production dispatch.

For 1 <= k <= 16, each of 256 threads retains its k smallest composite
(distance,index) keys, then the block merges those sorted lists. An element
outside a thread's local top-k cannot belong to the block top-k. Every merge
uses UInt64 minima; there are no floating reductions or atomic tie choices.
The key is the EXISTING selector's key, including its select-max ordering,
signed-zero distinction and NaN payload order. Output values are copied from
the original selected index, preserving their bits. The sentinel cannot be a
real key because accepted row lengths are <= INT32_MAX, below UINT32_MAX.

Caller owns nonoverlapping buffers and retains them through synchronization.
This is a candidate, not an established speedup or a certified default.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.select_radix_identical import composite_key

comptime SMALLK_BLOCK = 256
comptime SMALLK_LIMIT = 16


def smallk_identical_kernel(
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    length_in: Int32, k_in: Int32, select_min_in: Int32,
):
    var length = Int(length_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var sentinel = UInt64(18446744073709551615)
    var local_keys = SIMD[DType.uint64, SMALLK_LIMIT](sentinel)
    var heads = stack_allocation[
        SMALLK_BLOCK, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var col = tid
    while col < length:
        var pending = composite_key(values.unsafe_load(row * length + col), UInt32(col), select_min_in != 0)
        if pending < local_keys[k - 1]:
            # Carry insertion; a rejected item does no list maintenance.
            for slot in range(k):
                if pending < local_keys[slot]:
                    var previous = local_keys[slot]
                    local_keys[slot] = pending
                    pending = previous
        col += SMALLK_BLOCK
    for rank in range(k):
        var mine = local_keys[0]
        heads[tid] = mine
        barrier()
        var stride = SMALLK_BLOCK // 2
        while stride > 0:
            if tid < stride:
                var other = heads[tid + stride]
                if other < heads[tid]:
                    heads[tid] = other
            barrier()
            stride //= 2
        var winner = heads[0]
        # No thread may overwrite shared heads before every thread reads it.
        barrier()
        if tid == 0:
            var selected = UInt32(winner & UInt64(4294967295))
            out_indices.unsafe_store(row * k + rank, selected)
            out_values.unsafe_store(row * k + rank, values.unsafe_load(row * length + Int(selected)))
        if mine == winner:
            for slot in range(k - 1):
                local_keys[slot] = local_keys[slot + 1]
            local_keys[k - 1] = sentinel
        barrier()


def smallk_identical_into(
    ctx: DeviceContext, mut values: DeviceBuffer[DType.float32],
    mut out_values: DeviceBuffer[DType.float32], mut out_indices: DeviceBuffer[DType.uint32],
    rows: Int, length: Int, k: Int, select_min: Bool = True,
) raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("small-k candidate requires IDENTICAL")
    if rows <= 0 or rows > 2147483647 or length <= 0 or length > 2147483647:
        raise Error("small-k requires positive Int32 dimensions")
    if k < 1 or k > SMALLK_LIMIT or k > length:
        raise Error("small-k candidate supports only 1 <= k <= min(16, length)")
    if len(values) < rows * length or len(out_values) < rows * k or len(out_indices) < rows * k:
        raise Error("small-k buffer capacity is insufficient")
    ctx.enqueue_function[smallk_identical_kernel](
        values.unsafe_ptr(), out_values.unsafe_ptr(), out_indices.unsafe_ptr(),
        Int32(length), Int32(k), Int32(select_min),
        grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
    )
