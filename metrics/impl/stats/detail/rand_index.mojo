# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RAFT `cpp/include/raft/stats/detail/rand_index.cuh` (ebf9268).

REFERENCE (:67-158): a 2-D grid of 16x16 blocks over (i, j), `j < i`; a thread
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
BY PER-BLOCK PARTIALS SUMMED INTO INT64 (ON THE DEVICE SINCE 2026-11-19).
=========================================================================
REFERENCE: `raft::myAtomicAdd<unsigned long long>` on the two totals.
HERE: thread (0,0) of every block WRITES its block's two counts to
`partials[block]` and `partials[blocks + block]` (each at most 256, an
Int32) and `device_sum_i32` sums them into Int64. WHY: Apple's GPU has no 64-bit atomic -- `core/
block_reduce.mojo`'s DEVIATION 125 banner records the compiler's refusal
("Atomic operation is not supported for this type on Apple GPU") -- and a
32-bit atomic total overflows at `n(n-1)/2 > 2^31`, i.e. past 65,536
samples, which the metric's O(n^2) kernel can reach. A per-block partial
is the same integer as an atomic into a 64-bit slot (integer addition is
associative) and cannot overflow (a block sees at most 256 pairs). The
block-internal fold is a threadgroup Int32 atomic instead of CUB's
BlockReduce; same integer.

=========================================================================
THE CONTINGENCY ROUTE (lane fx-board-modelsel-metrics, 2026-11-19).
=========================================================================
The pair grid is O(n^2): 100,000 labels are 39 million blocks, 5e9 pairs
and 312 MB of block partials (13.3 s on an L40S, of which the adjusted
index next door, which reads the same pairs from the contingency matrix,
takes 2 ms). `a` and `b` are functions of that matrix C (k x k over the
joint label range, row sums a_i, column sums b_j):

    a = sum_ij nC2(C_ij)
    b = nC2(n) - sum_i nC2(a_i) - sum_j nC2(b_j) + sum_ij nC2(C_ij)

(pairs minus those together in the first labeling, minus those together in
the second, plus those together in both, which were removed twice). Exact
Int64, so `a`, `b` and the returned double are the pair grid's, bit for bit;
sklearn computes `rand_score` this way (`pair_confusion_matrix`). O(n + k^2).

WHICH ROUTE (`rand_uses_contingency`), from cost, not from any input we
know: the matrix is k^2 Int32 cells against n^2 / 2 pairs and n^2 / 128
partial words. It is taken when the matrix is at most RAND_CM_FREE_DIM^2
cells (1 MiB: less than the pair grid's own launch at any n) or at most
the words the pair grid's partials would take (k^2 * 128 <= n^2: never
more memory than the pair route, and at least 64 times fewer cells than
pairs). A wider label range (sparse label ids, or nearly one cluster per
sample) keeps the pair grid, which needs no table. At most
RAND_PAIR_DIRECT_MAX samples the pair grid runs without asking for the
label range: it is at most 1,024 blocks, less work than the two range
launches the question costs.

The pair grid's block partials are summed ON THE DEVICE
(core/device_fold.mojo `device_sum_i32`, exact Int64) where they were
downloaded and walked by one host thread (2 * (n / 16)^2 words).
"""

from std.atomic import Atomic
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.device_fold import device_sum_i32
from metrics.impl.stats.detail.contingency_matrix import (
    contingency_matrix,
    get_input_class_cardinality,
)
from metrics.impl.stats.detail.sf_epilogue import pair_sums_device


#: `static const int BLOCK_DIM_Y = 16, BLOCK_DIM_X = 16;` (:136)
comptime RAND_BLOCK_DIM_X = 16
comptime RAND_BLOCK_DIM_Y = 16

#: A contingency matrix of at most this dimension (1 MiB of Int32 cells) is
#: always cheaper than the pair grid. Cost reasoning in the module header.
comptime RAND_CM_FREE_DIM = 512

#: At most this many samples (1,024 blocks of pairs) the pair grid runs
#: without the two label-range launches. Cost reasoning in the module header.
comptime RAND_PAIR_DIRECT_MAX = 512

#: Most block partials one `device_sum_i32` call is given (its length is an
#: Int32 on the device).
comptime RAND_SUM_SPAN = 1 << 30


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
        # `a` partials first, then `b` partials: each half is one span the
        # device sums (rand_index_counts), the whole is `a + b`.
        var n_blk = Int(grid_dim.x) * Int(grid_dim.y)
        var blk = Int(block_idx.y) * Int(grid_dim.x) + Int(block_idx.x)
        partials.unsafe_store(blk, block_counts[unsafe_offset = 0])
        partials.unsafe_store(n_blk + blk, block_counts[unsafe_offset = 1])


def _launch_pair_grid(
    ctx: DeviceContext,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    mut partials: DeviceBuffer[DType.int32],
    size: Int,
) raises:
    """The pair grid over `size` samples; `partials` holds 2 * blocks words
    (the `a` half, then the `b` half). Nothing is waited on."""
    ctx.enqueue_function[compute_the_numerator_kernel](
        first_cluster_array.unsafe_ptr(),
        second_cluster_array.unsafe_ptr(),
        Int32(size),
        partials.unsafe_ptr(),
        grid_dim=(ceildiv(size, RAND_BLOCK_DIM_X), ceildiv(size, RAND_BLOCK_DIM_Y), 1),
        block_dim=(RAND_BLOCK_DIM_X, RAND_BLOCK_DIM_Y, 1),
    )


def _sum_partials(
    ctx: DeviceContext,
    mut partials: DeviceBuffer[DType.int32],
    first: Int,
    count: Int,
) raises -> Int64:
    """The exact Int64 sum of `partials[first : first + count]`, on the
    device; one word comes back per span."""
    var total = Int64(0)
    var done = 0
    # driver-loop: over spans of the block partials (one span up to 2^30 blocks)
    while done < count:
        var take = min(count - done, RAND_SUM_SPAN)
        var view = partials.create_sub_buffer[DType.int32](first + done, take)
        total += device_sum_i32(ctx, view, take)
        done += take
    return total


def rand_index_counts(
    ctx: DeviceContext,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
) raises -> Tuple[Int64, Int64]:
    """The integer half `(a, b)` from the PAIR GRID, exposed so the check
    gates it EXACTLY."""
    var n_blocks = ceildiv(size, RAND_BLOCK_DIM_X) * ceildiv(size, RAND_BLOCK_DIM_Y)
    var partials = ctx.enqueue_create_buffer[DType.int32](2 * n_blocks)
    _launch_pair_grid(ctx, first_cluster_array, second_cluster_array, partials, size)
    var a = _sum_partials(ctx, partials, 0, n_blocks)
    var b = _sum_partials(ctx, partials, n_blocks, n_blocks)
    _ = partials^
    return (a, b)


def rand_pair_total(
    ctx: DeviceContext,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
) raises -> Int64:
    """`a + b` from the pair grid: one sum over both halves of the partials."""
    var n_blocks = ceildiv(size, RAND_BLOCK_DIM_X) * ceildiv(size, RAND_BLOCK_DIM_Y)
    var partials = ctx.enqueue_create_buffer[DType.int32](2 * n_blocks)
    _launch_pair_grid(ctx, first_cluster_array, second_cluster_array, partials, size)
    var total = _sum_partials(ctx, partials, 0, 2 * n_blocks)
    _ = partials^
    return total


def rand_uses_contingency(k: Int, size: Int) -> Bool:
    """Whether a k x k contingency matrix is the cheaper route for `size`
    samples (the module header's cost rule). `k` is the joint label range."""
    if k <= RAND_CM_FREE_DIM:
        return True
    # k <= size // 11 first: it keeps k * k * 128 inside Int64 for any Int32 range
    return k <= size // 11 and k * k * 128 <= size * size


def rand_index_counts_contingency(
    ctx: DeviceContext,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
    lower_label_range: Int32,
    upper_label_range: Int32,
) raises -> Tuple[Int64, Int64]:
    """`(a, b)` from the contingency matrix over `[lower, upper]` (the module
    header's two sums): the same integers as `rand_index_counts`."""
    var k = Int(upper_label_range) - Int(lower_label_range) + 1
    var cm = ctx.enqueue_create_buffer[DType.int32](k * k)
    contingency_matrix(
        ctx,
        first_cluster_array,
        second_cluster_array,
        size,
        cm,
        lower_label_range,
        upper_label_range,
    )
    var sums = pair_sums_device(ctx, cm, k, size)
    _ = cm^
    var n = Int64(size)
    var a = sums[0]
    var b = n * (n - 1) // 2 - sums[1] - sums[2] + sums[0]
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
    var a_plus_b = Int64(0)
    if size <= RAND_PAIR_DIRECT_MAX:
        a_plus_b = rand_pair_total(ctx, first_cluster_array, second_cluster_array, size)
    else:
        var m1 = get_input_class_cardinality(ctx, first_cluster_array, size)
        var m2 = get_input_class_cardinality(ctx, second_cluster_array, size)
        var lower = m1[0] if m1[0] < m2[0] else m2[0]
        var upper = m1[1] if m1[1] > m2[1] else m2[1]
        if rand_uses_contingency(Int(upper) - Int(lower) + 1, size):
            var ab = rand_index_counts_contingency(
                ctx, first_cluster_array, second_cluster_array, size, lower, upper
            )
            a_plus_b = ab[0] + ab[1]
        else:
            a_plus_b = rand_pair_total(ctx, first_cluster_array, second_cluster_array, size)
    var n = Int64(size)
    var n_choose_two = n * (n - 1) // 2  # (:154), integer
    # `(double)(((double)(a + b)) / (double)nChooseTwo)` (:157)
    return Float64(a_plus_b) / Float64(n_choose_two)
