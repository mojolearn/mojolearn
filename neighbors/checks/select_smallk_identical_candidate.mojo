# SPDX-License-Identifier: Apache-2.0
"""Small-k composite-key selector for the IDENTICAL tiled k-NN arm.

PRODUCTION DISPATCH SINCE 2026-09-09 through the kernel-matrix row
`knn_smallk_select_for` (NVIDIA and AMD by default), for every 1 <= k <= 64
via `smallk_bucket_kernel` / `smallk_select_launch` at the bottom of this
file; the k in {8, 10, 16} specializations and the runtime baseline above
them are kept for the A/B drivers. `partial_topk_merge_kernel`, also at the
bottom, is the index-axis tiling's merge.

For 1 <= k <= 16, each of 256 threads retains its k smallest composite
(distance,index) keys, then the block merges those sorted lists. An element
outside a thread's local top-k cannot belong to the block top-k. Every merge
uses UInt64 minima; there are no floating reductions or atomic tie choices.
The key is the EXISTING selector's key, including its select-max ordering,
signed-zero distinction and NaN payload order. Output values are copied from
the original selected index, preserving their bits. The sentinel cannot be a
real key because accepted row lengths are <= INT32_MAX, below UINT32_MAX.

Caller owns nonoverlapping buffers and retains them through synchronization.
The optional specialized arm fixes K=8/10/16 at compile time, unrolling
insertion and shifts so local SIMD indexing cannot spill via runtime indices.
Other K values use the retained runtime baseline.
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


# Specialized counterpart preserves the baseline kernel above for A/B gates.
def smallk_specialized_kernel[K: Int](
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    length_in: Int32, k_in: Int32, select_min_in: Int32,
):
    var length = Int(length_in)
    # K is fixed at compile time; every SIMD lane access is constant.
    var k = K
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
        if pending < local_keys[K - 1]:
            # Carry insertion; a rejected item does no list maintenance.
            comptime for slot in range(K):
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
            comptime for slot in range(K - 1):
                local_keys[slot] = local_keys[slot + 1]
            local_keys[K - 1] = sentinel
        barrier()


def smallk_identical_into(
    ctx: DeviceContext, mut values: DeviceBuffer[DType.float32],
    mut out_values: DeviceBuffer[DType.float32], mut out_indices: DeviceBuffer[DType.uint32],
    rows: Int, length: Int, k: Int, select_min: Bool = True, specialized: Bool = False,
) raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("small-k candidate requires IDENTICAL")
    if rows <= 0 or rows > 2147483647 or length <= 0 or length > 2147483647:
        raise Error("small-k requires positive Int32 dimensions")
    if k < 1 or k > SMALLK_LIMIT or k > length:
        raise Error("small-k candidate supports only 1 <= k <= min(16, length)")
    if len(values) < rows * length or len(out_values) < rows * k or len(out_indices) < rows * k:
        raise Error("small-k buffer capacity is insufficient")
    if specialized:
        if k == 8:
            ctx.enqueue_function[smallk_specialized_kernel[8]](
                values.unsafe_ptr(), out_values.unsafe_ptr(), out_indices.unsafe_ptr(),
                Int32(length), Int32(k), Int32(select_min),
                grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
            )
            return
        elif k == 10:
            ctx.enqueue_function[smallk_specialized_kernel[10]](
                values.unsafe_ptr(), out_values.unsafe_ptr(), out_indices.unsafe_ptr(),
                Int32(length), Int32(k), Int32(select_min),
                grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
            )
            return
        elif k == 16:
            ctx.enqueue_function[smallk_specialized_kernel[16]](
                values.unsafe_ptr(), out_values.unsafe_ptr(), out_indices.unsafe_ptr(),
                Int32(length), Int32(k), Int32(select_min),
                grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
            )
            return
    # Other k values retain the baseline runtime implementation.
    ctx.enqueue_function[smallk_identical_kernel](
        values.unsafe_ptr(), out_values.unsafe_ptr(), out_indices.unsafe_ptr(),
        Int32(length), Int32(k), Int32(select_min),
        grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
    )


# ---------------------------------------------------------------------------
# The generalized selector, 2026-09-09: every 1 <= k <= SMALLK_MAX_K.
#
# Same algorithm as `smallk_specialized_kernel` with the per-thread capacity a
# comptime bucket (16 / 32 / 64) and k a runtime argument, so three
# instantiations cover the whole band instead of one per k. Every list index
# is a comptime constant, so the local list stays in registers. The block
# merge pops one winner per rank through a shared-memory min tree; both the
# min and the pop are exact integer operations on unique keys, so the output
# is the k smallest composite keys ascending, which is what the radix
# selector's rank pass writes too.
# ---------------------------------------------------------------------------

comptime SMALLK_MAX_K = 64


def smallk_bucket_kernel[CAP: Int](
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
    var local_keys = SIMD[DType.uint64, CAP](sentinel)
    # `local_keys[k - 1]`, kept in its own register so the reject test never
    # indexes the list at a runtime position.
    var threshold = sentinel
    var heads = stack_allocation[
        SMALLK_BLOCK, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var col = tid
    while col < length:
        var pending = composite_key(values.unsafe_load(row * length + col), UInt32(col), select_min_in != 0)
        if pending < threshold:
            comptime for slot in range(CAP):
                if slot < k:
                    if pending < local_keys[slot]:
                        var previous = local_keys[slot]
                        local_keys[slot] = pending
                        pending = previous
            comptime for slot in range(CAP):
                if slot == k - 1:
                    threshold = local_keys[slot]
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
        barrier()
        if tid == 0:
            var selected = UInt32(winner & UInt64(4294967295))
            out_indices.unsafe_store(row * k + rank, selected)
            out_values.unsafe_store(row * k + rank, values.unsafe_load(row * length + Int(selected)))
        if mine == winner:
            comptime for slot in range(CAP - 1):
                local_keys[slot] = local_keys[slot + 1]
            local_keys[CAP - 1] = sentinel
        barrier()


def smallk_select_launch(
    ctx: DeviceContext,
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, length: Int, k: Int, select_min: Bool = True,
) raises:
    """One query tile's top-k for 1 <= k <= SMALLK_MAX_K, bucketed by capacity.

    Pointer form, so the caller can offset into the outer output buffer the
    way the radix launch does. `length >= k` is the caller's to guarantee:
    with fewer real keys than k the sentinel would win a rank.
    """
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("small-k selector requires IDENTICAL")
    if rows <= 0 or rows > 2147483647 or length <= 0 or length > 2147483647:
        raise Error("small-k selector requires positive Int32 dimensions")
    if k < 1 or k > SMALLK_MAX_K or k > length:
        raise Error("small-k selector supports only 1 <= k <= min(64, length)")
    if k <= 16:
        ctx.enqueue_function[smallk_bucket_kernel[16]](
            values, out_values, out_indices,
            Int32(length), Int32(k), Int32(select_min),
            grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
        )
    elif k <= 32:
        ctx.enqueue_function[smallk_bucket_kernel[32]](
            values, out_values, out_indices,
            Int32(length), Int32(k), Int32(select_min),
            grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
        )
    else:
        ctx.enqueue_function[smallk_bucket_kernel[64]](
            values, out_values, out_indices,
            Int32(length), Int32(k), Int32(select_min),
            grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
        )


# ---------------------------------------------------------------------------
# The partial top-k merge, 2026-09-09: index-axis tiling's second half.
#
# `running` holds a row's k best (distance, index) pairs so far, ascending by
# composite key with GLOBAL indices; `partial` holds the k best of the next
# column tile, ascending, with indices LOCAL to that tile (add `base`). Keys
# are unique, so the rank of an element in the union is its own position
# plus the number of elements of the other list below it, and the k smallest
# ranks are written back in place. No arithmetic on the distances, no
# arrival order anywhere: the result is a pure function of the two lists.
# ---------------------------------------------------------------------------

comptime MERGE_BLOCK = 256
comptime MERGE_MAX_K = 1024


def partial_topk_merge_kernel(
    running_values: MutPointer[Float32, MutAnyOrigin],
    running_indices: MutPointer[UInt32, MutAnyOrigin],
    partial_values: MutPointer[Float32, MutAnyOrigin],
    partial_indices: MutPointer[UInt32, MutAnyOrigin],
    k_in: Int32, base_in: Int32, select_min_in: Int32,
):
    var k = Int(k_in)
    var base = UInt32(Int(base_in))
    var select_min = select_min_in != 0
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var keys = stack_allocation[
        2 * MERGE_MAX_K, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var vals = stack_allocation[
        2 * MERGE_MAX_K, Scalar[DType.float32], address_space=AddressSpace.SHARED,
    ]()
    var slot = tid
    while slot < 2 * k:
        if slot < k:
            var v = running_values.unsafe_load(row * k + slot)
            keys[slot] = composite_key(v, running_indices.unsafe_load(row * k + slot), select_min)
            vals[slot] = v
        else:
            var v = partial_values.unsafe_load(row * k + slot - k)
            keys[slot] = composite_key(v, partial_indices.unsafe_load(row * k + slot - k) + base, select_min)
            vals[slot] = v
        slot += MERGE_BLOCK
    barrier()
    slot = tid
    while slot < 2 * k:
        var key = keys[slot]
        var rank: Int
        var other_start: Int
        if slot < k:
            rank = slot
            other_start = k
        else:
            rank = slot - k
            other_start = 0
        for j in range(other_start, other_start + k):
            if keys[j] < key:
                rank += 1
        if rank < k:
            running_values.unsafe_store(row * k + rank, vals[slot])
            running_indices.unsafe_store(row * k + rank, UInt32(key & UInt64(4294967295)))
        slot += MERGE_BLOCK
    barrier()


def partial_topk_merge_launch(
    ctx: DeviceContext,
    running_values: MutPointer[Float32, MutAnyOrigin],
    running_indices: MutPointer[UInt32, MutAnyOrigin],
    partial_values: MutPointer[Float32, MutAnyOrigin],
    partial_indices: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, k: Int, base: Int, select_min: Bool = True,
) raises:
    """Merge one column tile's partial top-k into the running top-k, per row."""
    if rows <= 0 or rows > 2147483647 or base < 0 or base > 2147483647:
        raise Error("partial top-k merge requires positive Int32 dimensions")
    if k < 1 or k > MERGE_MAX_K:
        raise Error("partial top-k merge supports only 1 <= k <= 1024")
    ctx.enqueue_function[partial_topk_merge_kernel](
        running_values, running_indices, partial_values, partial_indices,
        Int32(k), Int32(base), Int32(select_min),
        grid_dim=(rows, 1, 1), block_dim=(MERGE_BLOCK, 1, 1),
    )
