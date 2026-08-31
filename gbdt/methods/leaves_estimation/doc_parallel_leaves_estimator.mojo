# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in the root DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Estimating one tree's leaves on a dataset it was not grown on.

PORT OF `catboost/cuda/methods/leaves_estimation/doc_parallel_leaves_estimator.{h,cpp}`
at CatBoost `54a8143a` -- specifically `CreateDerCalcer` (`:41-56`), the
half that turns a (structure, dataset, cursor) triple into something the
oracle can read. Transliterated. Do not improve.

## Why this exists, when `fit` already estimates leaves

Because their estimator takes TASKS, one per permutation
(`doc_parallel_boosting.h:371-385`):

    for (permutation = 0; permutation < permutationCount; ++permutation) {
        estimator.AddEstimationTask(learnTarget[permutation],
                                    dataSet.GetDataSetForPermutation(permutation),
                                    learnCursors[permutation],
                                    &iterationModels[permutation]);
    }

and every task but one is on a dataset whose rows the searcher never
partitioned. The searcher hands `fit` a bin-sorted `row_index` for free, as
a by-product of growing the tree; for the other permutations there is no
by-product, and the leaf a row falls into has to be COMPUTED from that
permutation's compressed index. That is exactly what their
`task.Model->ComputeBins(*task.DataSet, &bins)` (`:47-48`) does.

## The one substitution, and it is a real one

Their oracle takes `bins` UNSORTED and does its own partitioning inside
(`TBinOptimizedOracle`'s constructor, off the same `bins` buffer). This
port's oracle takes rows ALREADY GROUPED BY LEAF plus per-leaf offsets and
sizes, because that is the shape the searcher produces and the shape the
gather kernels were written against. So `partition_from_bins` has to build
that grouping, and it builds it ON THE HOST. DEVIATION 90.

**The learn permutation does NOT go through here.** It keeps the searcher's
own partition, so a one-permutation fit is byte for byte what it was before
this file existed. That is not an optimization: rows within a leaf are
summed in whatever order the partition holds them, and two orders give two
float sums, so routing the learn permutation through a different grouping
would move every number in the fit for no reason.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.models.kernel.add_bin_values import compute_bins_kernel
from gbdt.models.oblivious_model import BIN_SPLIT_TAKE_BIN, TBinarySplit

#: their `ComputeObliviousTreeBins`'s launch (`add_model_value.cu:193-195`)
comptime COMPUTE_BINS_BLOCK_SIZE = 256


@fieldwise_init
struct LeafPartition(Movable):
    """Rows grouped by leaf: their `bins` after the oracle has ordered it.

    `row_index` holds every row exactly once, leaf 0's rows first. `offsets`
    and `sizes` are per leaf and index into it. The oracle takes all three.
    """

    var row_index: DeviceBuffer[DType.uint32]
    var offsets: List[Int]
    var sizes: List[Int]


def compute_bins_for_model(
    ctx: DeviceContext,
    layout: CompressedIndexLayout,
    splits: List[TBinarySplit],
    depth: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    n_rows: Int,
    mut out_bins: DeviceBuffer[DType.uint32],
) raises:
    """`ComputeBinsForModel` (`models/add_oblivious_tree_model_doc_parallel.cpp:
    195-202`), which is `TComputeLeavesDocParallel` with one task.

    Packs the `depth` level records their `AddTask` packs -- the feature's
    offset, shift and mask, the split's bin, and whether the predicate is
    equality -- and launches the bins kernel once.
    """
    if depth <= 0:
        raise Error("compute_bins_for_model: depth must be positive")
    if len(splits) < depth:
        raise Error(
            "compute_bins_for_model: " + String(len(splits))
            + " splits for depth " + String(depth)
        )

    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](depth)
    var h_shift = ctx.enqueue_create_host_buffer[DType.uint32](depth)
    var h_mask = ctx.enqueue_create_host_buffer[DType.uint32](depth)
    var h_bin = ctx.enqueue_create_host_buffer[DType.uint32](depth)
    var h_eq = ctx.enqueue_create_host_buffer[DType.uint8](depth)
    for level in range(depth):
        ref cf = layout.features[Int(splits[level].feature_id)]
        h_off.unsafe_ptr().unsafe_store(level, cf.offset * UInt32(n_rows))
        h_shift.unsafe_ptr().unsafe_store(level, cf.shift)
        h_mask.unsafe_ptr().unsafe_store(level, cf.mask)
        h_bin.unsafe_ptr().unsafe_store(
            level, UInt32(Int(splits[level].bin_idx))
        )
        var take_bin = (
            Int(splits[level].split_type) == BIN_SPLIT_TAKE_BIN
        )
        h_eq.unsafe_ptr().unsafe_store(
            level, UInt8(1) if take_bin else UInt8(0)
        )

    var d_off = ctx.enqueue_create_buffer[DType.uint32](depth)
    var d_shift = ctx.enqueue_create_buffer[DType.uint32](depth)
    var d_mask = ctx.enqueue_create_buffer[DType.uint32](depth)
    var d_bin = ctx.enqueue_create_buffer[DType.uint32](depth)
    var d_eq = ctx.enqueue_create_buffer[DType.uint8](depth)
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_shift, src_ptr=h_shift.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_mask, src_ptr=h_mask.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_bin, src_ptr=h_bin.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_eq, src_ptr=h_eq.unsafe_ptr())

    var blocks = (
        n_rows + COMPUTE_BINS_BLOCK_SIZE - 1
    ) // COMPUTE_BINS_BLOCK_SIZE
    if blocks < 1:
        blocks = 1
    ctx.enqueue_function[compute_bins_kernel](
        cindex.unsafe_ptr(),
        d_off.unsafe_ptr(),
        d_shift.unsafe_ptr(),
        d_mask.unsafe_ptr(),
        d_bin.unsafe_ptr(),
        d_eq.unsafe_ptr(),
        Int32(depth),
        Int32(n_rows),
        out_bins.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(COMPUTE_BINS_BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()
    _ = d_eq^  # past the drain (step-33 race class, device side)
    _ = d_bin^  # past the drain (step-33 race class, device side)
    _ = d_mask^  # past the drain (step-33 race class, device side)
    _ = d_shift^  # past the drain (step-33 race class, device side)
    _ = d_off^  # past the drain (step-33 race class, device side)
    # past the drain (the step-33 race class: freed-at-enqueue under a
    # queued copy)
    _ = h_off^
    _ = h_shift^
    _ = h_mask^
    _ = h_bin^
    _ = h_eq^


def partition_from_bins(
    ctx: DeviceContext,
    mut bins: DeviceBuffer[DType.uint32],
    n_rows: Int,
    n_leaves: Int,
) raises -> LeafPartition:
    """Group rows by leaf. DEVIATION 90: this is a HOST counting sort.

    Their oracle partitions on the device, off the same `bins` buffer it is
    handed. Ours needs the grouping before the oracle exists, so the bins
    come back to the host, are counted into `n_leaves` buckets, and the row
    order goes out again.

    It is a STABLE counting sort -- rows keep their ascending order within a
    leaf -- because an unstable one would make the leaf sums depend on the
    scatter order, and a float sum that depends on scheduling is a number
    that changes between runs of the same fit.

    The cost is two host passes over `n_rows` plus a round trip each way,
    per permutation per tree. It is real and it is on the fixed-cost side of
    `ms/tree`; the device alternative is a radix sort by a `depth`-bit key,
    which this repository already has the pieces for and which nothing has
    measured a need for yet.
    """
    if n_leaves <= 0:
        raise Error("partition_from_bins: n_leaves must be positive")

    var h_bins = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    ctx.enqueue_copy(dst_ptr=h_bins.unsafe_ptr(), src_buf=bins)
    ctx.synchronize()

    var sizes = List[Int]()
    for _ in range(n_leaves):
        sizes.append(0)
    for r in range(n_rows):
        var b = Int(h_bins.unsafe_ptr().unsafe_load(r))
        if b < 0 or b >= n_leaves:
            raise Error(
                "partition_from_bins: row " + String(r) + " fell in leaf "
                + String(b) + " of " + String(n_leaves)
            )
        sizes[b] += 1

    var offsets = List[Int]()
    var running = 0
    for i in range(n_leaves):
        offsets.append(running)
        running += sizes[i]

    var fill = List[Int]()
    for i in range(n_leaves):
        fill.append(offsets[i])
    var h_rows = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        var b = Int(h_bins.unsafe_ptr().unsafe_load(r))
        h_rows.unsafe_ptr().unsafe_store(fill[b], UInt32(r))
        fill[b] += 1

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=h_rows.unsafe_ptr())
    ctx.synchronize()
    _ = h_rows^  # past the drain (step-33 race class)

    return LeafPartition(row_index^, offsets^, sizes^)
