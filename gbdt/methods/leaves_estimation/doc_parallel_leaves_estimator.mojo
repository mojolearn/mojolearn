# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Estimating one tree's leaves on a dataset it was not grown on.

FOLLOWS `catboost/cuda/methods/leaves_estimation/doc_parallel_leaves_estimator.{h,cpp}`
at CatBoost `54a8143a` -- specifically `CreateDerCalcer` (`:41-56`), the
half that turns a (structure, dataset, cursor) triple into something the
oracle can read. Followed statement for statement.

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
implementation's oracle takes rows ALREADY GROUPED BY LEAF plus per-leaf offsets and
sizes, because that is the shape the searcher produces and the shape the
gather kernels were written against. So `partition_from_bins` has to build
that grouping, and it builds it ON THE HOST. DEVIATION 90.

**The learn permutation does NOT go through here on the SYMMETRIC path.** It
keeps the searcher's own partition, so a one-permutation symmetric fit is byte
for byte what it was before this file existed. (Corrected 2026-09-11: the
Depthwise and Lossguide path sends EVERY permutation, the learn one included,
through `partition_from_bins`; see `fit_with_test`'s non-symmetric arm.) That is not an optimization: rows within a leaf are
summed in whatever order the partition holds them, and two orders give two
float sums, so routing the learn permutation through a different grouping
would move every number in the fit for no reason.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.sys.compile import is_defined

from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.gpu_util.copy import COPY_BLOCK, copy_u32_kernel
from gbdt.gpu_util.kernel.fill import launch_make_sequence
from gbdt.gpu_util.kernel.radix_sort import launch_radix_sort_bins
from gbdt.gpu_util.kernel.reorder_one_bit import REORDER_BLOCK

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime DEVICE_LEAF_PARTITION = is_defined["MOJOLEARN_2551_DEVICE_PARTITION"]() or (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_2551_DEVICE_PARTITION_OFF"]()
)
"""DEVIATION 2551 (2026-09-11). DEFAULT ON UNDER IDENTICAL since
2026-09-11 (DigitalOcean MI325X, taxi and Istella-S on one box, combined
over depthwise and lossguide: FLIP geomean=0.898 cells=4 quality=ok);
`-D MOJOLEARN_2551_DEVICE_PARTITION_OFF=1` restores the host partition.
FAST stays OPT-IN (`-D MOJOLEARN_2551_DEVICE_PARTITION=1`): its verdict
read NO FLIP on a taxi quality flag that is FAST's own round-to-round fit
variation, left to the orchestrator. `partition_from_bins` below
(DEVIATION 90) reads every row's leaf back to the host, counting-sorts
1M rows there on two passes, allocates two n_rows pinned buffers and one
device buffer, and uploads the row order again: once per tree per
permutation on the Depthwise and Lossguide paths. Their oracle partitions
on the device off the same `bins` buffer (the module docstring), so the
switch moves the grouping there: `DeviceLeafPartitioner.partition` is a
stable LSD radix sort of (leaf, row) over their `ReorderBins`
(`launch_radix_sort_bins`) from the identity row order, then one kernel
marks each leaf's first and one-past-last position, and the host reads
back `2 * n_leaves + 1` words. A stable sort from ascending row ids leaves
every leaf's rows ascending, which is exactly the host counting sort's
output, so `row_index`, `offsets` and `sizes` are the same integers and
the model is bitwise the default's. The buffers are the fit's pool of one.
Checks: every build compares the two partitions directly (claim 1 of
`checks/gbdt_per_round_check.mojo`); `pixi run check-gbdt-per-round` runs
the fits on the ON side and `check-gbdt-per-round-2551-host-partition` on
the OFF side."""
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


def leaf_bounds_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    bounds: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
    n_leaves_in: Int32,
):
    """DEVIATION 2551: over leaf-sorted `keys`, write each present leaf's
    first position to `bounds[leaf]` and one past its last to
    `bounds[n_leaves + leaf]`; a key at or above `n_leaves` sets
    `bounds[2 * n_leaves]`. Every slot has exactly one writer (the unique
    boundary row of its leaf), except the flag, whose writers all write 1.
    Integer only. Grid-stride, as `copy_u32_kernel`."""
    var n = Int(n_in)
    var n_leaves = Int(n_leaves_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        var k = Int(keys.unsafe_load(i))
        if k >= n_leaves:
            bounds.unsafe_store(2 * n_leaves, UInt32(1))
        else:
            if i == 0 or Int(keys.unsafe_load(i - 1)) != k:
                bounds.unsafe_store(k, UInt32(i))
            if i == n - 1 or Int(keys.unsafe_load(i + 1)) != k:
                bounds.unsafe_store(n_leaves + k, UInt32(i + 1))
        i += stride


struct DeviceLeafPartitioner(Movable):
    """DEVIATION 2551: `partition_from_bins` on the device, owned by the
    fit (pool of one, like `TEstimationWorkspace`). `bins` is the buffer the
    caller writes the tree's leaf per row into; `partition` groups by it.

    The returned `row_index` is a HANDLE onto `vals`: the next `partition`
    call rewrites it, so a caller consumes one partition (and drains) before
    asking for the next, which is what the estimation loop does
    (`_estimate_and_apply` ends on a drain)."""

    var n_rows_cap: Int
    var n_leaves_cap: Int
    var bins: DeviceBuffer[DType.uint32]
    var keys: DeviceBuffer[DType.uint32]
    var vals: DeviceBuffer[DType.uint32]
    var tkeys: DeviceBuffer[DType.uint32]
    var tvals: DeviceBuffer[DType.uint32]
    var offsets: DeviceBuffer[DType.int32]
    var bsums: DeviceBuffer[DType.int32]
    var d_bounds: DeviceBuffer[DType.uint32]
    var h_bounds: HostBuffer[DType.uint32]

    def __init__(
        out self, ctx: DeviceContext, n_rows: Int, n_leaves: Int
    ) raises:
        if n_rows <= 0 or n_leaves <= 0:
            raise Error(
                "DeviceLeafPartitioner: n_rows and n_leaves must be positive"
            )
        self.n_rows_cap = n_rows
        self.n_leaves_cap = n_leaves
        self.bins = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.keys = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.vals = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.tkeys = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.tvals = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.offsets = ctx.enqueue_create_buffer[DType.int32](n_rows)
        self.bsums = ctx.enqueue_create_buffer[DType.int32](
            (n_rows + REORDER_BLOCK - 1) // REORDER_BLOCK
        )
        self.d_bounds = ctx.enqueue_create_buffer[DType.uint32](
            2 * n_leaves + 1
        )
        self.h_bounds = ctx.enqueue_create_host_buffer[DType.uint32](
            2 * n_leaves + 1
        )
        ctx.synchronize()

    def partition(
        mut self, ctx: DeviceContext, n_rows: Int, n_leaves: Int
    ) raises -> LeafPartition:
        """Group the first `n_rows` entries of `bins` by leaf. Same result
        as `partition_from_bins(ctx, bins, n_rows, n_leaves)`, and the same
        refusal of a leaf at or above `n_leaves`."""
        if n_rows <= 0 or n_rows > self.n_rows_cap:
            raise Error(
                "DeviceLeafPartitioner: n_rows " + String(n_rows)
                + " outside capacity " + String(self.n_rows_cap)
            )
        if n_leaves <= 0 or n_leaves > self.n_leaves_cap:
            raise Error(
                "DeviceLeafPartitioner: n_leaves " + String(n_leaves)
                + " outside capacity " + String(self.n_leaves_cap)
            )
        var copy_blocks = (n_rows + COPY_BLOCK - 1) // COPY_BLOCK
        ctx.enqueue_function[copy_u32_kernel](
            self.keys.unsafe_ptr(), self.bins.unsafe_ptr(), Int32(n_rows),
            grid_dim=copy_blocks, block_dim=COPY_BLOCK,
        )
        launch_make_sequence(ctx, UInt32(0), self.vals, n_rows)
        var bits = 0
        while (1 << bits) < n_leaves:
            bits += 1
        launch_radix_sort_bins(
            ctx, n_rows, 0, bits, self.keys, self.vals, self.tkeys,
            self.tvals, self.offsets, self.bsums,
        )
        var hb = self.h_bounds.unsafe_ptr()
        for i in range(2 * n_leaves + 1):
            hb.unsafe_store(i, UInt32(0))
        ctx.enqueue_copy(dst_buf=self.d_bounds, src_ptr=hb)
        ctx.enqueue_function[leaf_bounds_kernel](
            self.keys.unsafe_ptr(), self.d_bounds.unsafe_ptr(),
            Int32(n_rows), Int32(n_leaves),
            grid_dim=copy_blocks, block_dim=COPY_BLOCK,
        )
        ctx.enqueue_copy(dst_ptr=hb, src_buf=self.d_bounds)
        ctx.synchronize()
        if hb.unsafe_load(2 * n_leaves) != UInt32(0):
            raise Error(
                "DeviceLeafPartitioner: a row fell in a leaf at or above "
                + String(n_leaves)
            )
        var sizes = List[Int]()
        var offsets = List[Int]()
        var running = 0
        for leaf in range(n_leaves):
            var first = Int(hb.unsafe_load(leaf))
            var past = Int(hb.unsafe_load(n_leaves + leaf))
            var size = past - first if past > first else 0
            offsets.append(running)
            sizes.append(size)
            running += size
        if running != n_rows:
            raise Error(
                "DeviceLeafPartitioner: leaf sizes sum to " + String(running)
                + " for " + String(n_rows) + " rows"
            )
        return LeafPartition(self.vals.copy(), offsets^, sizes^)
