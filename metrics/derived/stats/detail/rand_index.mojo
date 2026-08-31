# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""RAFT `cpp/include/raft/stats/detail/rand_index.cuh` (ebf9268).

THEIRS (:67-158): a 2-D grid of 16x16 blocks over (i, j), `j < i`; a thread
counts `a` (same cluster in BOTH labelings) or `b` (different in both);
`cub::BlockReduce<uint64_t>::Sum` twice, then thread (0,0) does
`atomicAdd(unsigned long long)` on `a` and `b` (:93-109); the host
returns `(a + b) / nChooseTwo` as a double (:154-157), with `nChooseTwo =
size * (size - 1) / 2` in uint64 (:154). `size < 2` returns 1.0 (:126-129).

sklearn `rand_score`: `(tp + tn) / (tp + fp + fn + tn)` from
`pair_confusion_matrix` -- the same number computed from the contingency
matrix instead of the pairs; `rand_score` of one sample (no pairs) is 1.0
too.

ALL INTEGER until the last division. Every interleaving of integer adds
gives the same bits, so no IDENTICAL arm exists; the final `double / double`
is one correctly-rounded host op.

=========================================================================
DEVIATION 652 (metrics lane, 2026-08-23): THE 64-BIT ATOMIC IS REPLACED
BY PER-BLOCK PARTIALS SUMMED ON THE HOST.
=========================================================================
THEIRS: `raft::myAtomicAdd<unsigned long long>` on the two totals.
OURS: thread (0,0) of every block WRITES its block's two counts to
`partials[2 * block + {0, 1}]` (each at most 256, an Int32) and the host
sums them into Int64. WHY: Apple's GPU has no 64-bit atomic -- `core/
block_reduce.mojo`'s DEVIATION 125 banner records the compiler's refusal
("Atomic operation is not supported for this type on Apple GPU") -- and a
32-bit atomic total overflows at `n(n-1)/2 > 2^31`, i.e. past 65,536
samples, which the metric's O(n^2) kernel can reach. A per-block partial
is the same integer as an atomic into a 64-bit slot (integer addition is
associative) and cannot overflow (a block sees at most 256 pairs). The
block-internal fold is a threadgroup Int32 atomic instead of CUB's
BlockReduce; same integer.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier


#: `static const int BLOCK_DIM_Y = 16, BLOCK_DIM_X = 16;` (:136)
comptime RAND_BLOCK_DIM_X = 16
comptime RAND_BLOCK_DIM_Y = 16


def compute_the_numerator_kernel(
    first_cluster_array: MutPointer[Int32, MutAnyOrigin],
    second_cluster_array: MutPointer[Int32, MutAnyOrigin],
    size: Int32,
    partials: MutPointer[Int32, MutAnyOrigin],
):
    """`computeTheNumerator` (:67-110), with the block totals written to
    `partials` (DEVIATION 652)."""
    var j = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    var i = Int(thread_idx.y) + Int(block_idx.y) * Int(block_dim.y)
    var n = Int(size)
    var block_counts = stack_allocation[
        2, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    if tx == 0 and ty == 0:
        block_counts[unsafe_offset = 0] = Int32(0)
        block_counts[unsafe_offset = 1] = Int32(0)
    barrier()
    var my_a = Int32(0)
    var my_b = Int32(0)
    # `if (i < size && j < size && j < i)` (:78): the unordered pairs once.
    if i < n and j < n and j < i:
        var fi = first_cluster_array.unsafe_load(i)
        var fj = first_cluster_array.unsafe_load(j)
        var si = second_cluster_array.unsafe_load(i)
        var sj = second_cluster_array.unsafe_load(j)
        if fi == fj and si == sj:
            my_a += 1
        elif fi != fj and si != sj:
            my_b += 1
    if my_a != Int32(0):
        _ = Atomic.fetch_add(block_counts.unsafe_offset(0), my_a)
    if my_b != Int32(0):
        _ = Atomic.fetch_add(block_counts.unsafe_offset(1), my_b)
    barrier()
    if tx == 0 and ty == 0:
        var blk = Int(block_idx.y) * Int(grid_dim.x) + Int(block_idx.x)
        partials.unsafe_store(2 * blk, block_counts[unsafe_offset = 0])
        partials.unsafe_store(2 * blk + 1, block_counts[unsafe_offset = 1])


def rand_index_counts(
    ctx: DeviceContext,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
) raises -> Tuple[Int64, Int64]:
    """The integer half `(a, b)`, exposed so the check gates it EXACTLY."""
    var gx = ceildiv(size, RAND_BLOCK_DIM_X)
    var gy = ceildiv(size, RAND_BLOCK_DIM_Y)
    var n_blocks = gx * gy
    var partials = ctx.enqueue_create_buffer[DType.int32](2 * n_blocks)
    ctx.enqueue_function[compute_the_numerator_kernel](
        first_cluster_array.unsafe_ptr(),
        second_cluster_array.unsafe_ptr(),
        Int32(size),
        partials.unsafe_ptr(),
        grid_dim=(gx, gy, 1),
        block_dim=(RAND_BLOCK_DIM_X, RAND_BLOCK_DIM_Y, 1),
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](2 * n_blocks)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=partials)
    ctx.synchronize()
    var a = Int64(0)
    var b = Int64(0)
    for blk in range(n_blocks):
        a += Int64(h.unsafe_ptr().unsafe_load(2 * blk))
        b += Int64(h.unsafe_ptr().unsafe_load(2 * blk + 1))
    _ = h^
    _ = partials^
    return (a, b)


def compute_rand_index(
    ctx: DeviceContext,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
) raises -> Float64:
    """`compute_rand_index(first, second, size, stream)` (:121-158)."""
    if size < 2:
        return 1.0  # (:126-129)
    var ab = rand_index_counts(
        ctx, first_cluster_array, second_cluster_array, size
    )
    var n = Int64(size)
    var n_choose_two = n * (n - 1) // 2  # (:154), integer
    # `(double)(((double)(a + b)) / (double)nChooseTwo)` (:157)
    return Float64(ab[0] + ab[1]) / Float64(n_choose_two)
