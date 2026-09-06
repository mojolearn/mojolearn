# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 1902: at a split, move ONLY the row index.

FAST-ARM LAUNCHER, routed by `ridx_only_splits_for`
(`checks/kernel_matrix.mojo`). No new kernel lives here -- every kernel
this file launches is `split_points.mojo`'s, unchanged -- because the whole
deviation is a SCHEDULE: the stat planes stop being permuted at splits and
every reader gathers them through the row index instead
(`stats[row_index[pos]]`), so the split's payload drops from
`(stat_count + 1)` columns moved twice to ONE column.

THE REFERENCE DESIGN is the one both source recons converged on:

  * XGBoost's `RowPartitioner` (dmlc/xgboost,
    `src/tree/gpu_hist/row_partitioner.cuh:112-201`) permutes a `ridx`
    array only; its histogram kernel gathers the gradient through it
    (`src/tree/gpu_hist/histogram.cu:186,212`).
  * LightGBM's CUDA learner is the same shape: `data_indices_in_leaf` moves
    (~4 B/row of the split leaf, `cuda_data_partition.cu:288-334, 679-783,
    907-944`) and gradients NEVER move -- the hist kernel reads
    `cuda_gradients[data_indices_in_leaf[i]]`
    (`cuda_histogram_constructor.cu:53-55`).
  * CatBoost -- the port's source of truth everywhere else -- moves the
    stat columns too (`TSplitPointsKernel`, `split_points.cpp:64-136`).
    That is the design this deviation leaves, which is why it is a numbered
    deviation and not a port.

WHAT THIS COSTS AND BUYS. Deleted per split on the slow arm:
`2 * ceil(stat_count / 8)` launches and `stat_count * 4` bytes per row of
every splitting leaf, twice (out to scratch and back). Bought: the hist
gather kernels' stat read turns from contiguous to indirect -- through the
SAME index register they already load for the compressed-index gather, so
no new load stream, but the coalescing changes on the hot kernel. The
recons' answer is that both competitors eat that gather at every row of
every histogram and win anyway; the orchestrator's A/B is where that claim
is priced on this port (see archive/research/LOSSGUIDE.md, DEVIATION 1902 ledger).

BIT-EXACT BY CONSTRUCTION (the invariant, stated once and relied on by
every converted reader): let `D` be the stat plane in the order the
objective writes it and `ridx` the row index. The OLD schedule maintains a
permuted plane `P` with `P[stat][pos] == D[stat][ridx[pos]]` -- true at
tree start (identity index over a freshly written plane; the depth-0
DIRECT kernels already rely on exactly this) and preserved by every split,
because the split permutes `P` and `ridx` with the SAME `gather_map` and a
permutation of floats moves bytes, never re-rounds them. So a reader that
gathers `D[stat][ridx[pos]]` reads THE SAME BITS the old reader loaded at
`P[stat][pos]`, at the same iteration of the same loop on the same thread.
Same values, same accumulation order, same dither keys (they stay
position-based): the histograms, partition stats, flags, gather maps and
leaf values are byte-for-byte the pre-1902 FAST path's. The deviation is
launches and traffic, not a number -- same standing as DEVIATION 1903, and
unlike 1901, which does re-associate.

Guarded FAST-only THIS ROUND anyway, by the routing row: transcription-
exactness is an argument until the orchestrator's byte-compare A/B makes
it a measurement; the IDENTICAL column keeps the stat-moving path
byte for byte, so the merge gate's compare passes by construction.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.device_attribute import DeviceAttribute

from gbdt.methods.greedy_subsets_searcher.kernel.split_points import (
    GATHER_INPLACE_BLOCK,
    GATHER_INPLACE_SIZE,
    LEAF_COPY_BLOCK,
    copy_index_in_leaves_kernel,
    gather_index_in_leaves_kernel,
    gather_inplace_kernel,
    split_points_grid_x,
)


def launch_reorder_index_only(
    ctx: DeviceContext,
    n_leaf_slots: Int,
    max_leaf_rows: Int,
    mut leaves: DeviceBuffer[DType.uint32],
    mut part_offset: DeviceBuffer[DType.uint32],
    mut part_size: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut temp_index: DeviceBuffer[DType.uint32],
    mut gather_map: DeviceBuffer[DType.uint32],
    sm_count: Int = -1,
) raises -> Int:
    """`launch_reorder_in_leaves` with the stat columns DELETED.

    Drop-in for the index half of `TSplitPointsKernel::Run`
    (`split_points.cpp:90-98` slow arm, `:113-135` fast arm), taken by the
    non-symmetric driver when `ridx_only_splits_for` routes a build here.
    THE DISPATCH IS STILL THEIRS -- `maxLeafSize > 1024` on the largest
    SPLITTING leaf (`split_points.cpp:65`), the number the depthwise lane's
    parent-snapshot fix restored -- only the payload shrinks:

      FAST ARM (`max_leaf_rows <= 1024`): `gather_inplace_kernel` at
      `grid.x = 1` instead of `1 + statCount`. Block 0 is the row-index
      block (`blockIdx.x == 0` selects `indices`, `split_points.cu:61`);
      the stat blocks simply do not exist. ONE launch, scratch untouched.

      SLOW ARM: the index copy/gather pair alone -- their own
      "//now copy indices" block (`split_points.cpp:90`) -- and NOT the
      `2 * ceil(statCount / 8)` stat pairs before it. TWO launches,
      `temp_index` scratch only; the stat scratch (`new_stats`) is dead on
      this route and the driver may stop allocating it once the routing
      default survives its A/B.

    Returns the launch count, exactly as `launch_reorder_in_leaves` does
    and for the same reason: the caller tells the manager about each
    launch.

    THE STATS ARGUMENT IS GONE, NOT DEFAULTED. A caller that still wants
    stats moved is a caller on the other path; giving this launcher an
    optional stat plane would let one call site quietly run half of each
    schedule.

    The kernels are imported from `split_points.mojo` UNCHANGED -- the
    permutation they apply to `row_index` is byte-for-byte the one the full
    launcher applies, which is what makes the routed A/B a pure schedule
    comparison.
    """
    if n_leaf_slots <= 0:
        return 0

    if max_leaf_rows <= GATHER_INPLACE_SIZE:
        ctx.enqueue_function[gather_inplace_kernel](
            leaves.unsafe_ptr(),
            part_offset.unsafe_ptr(),
            part_size.unsafe_ptr(),
            gather_map.unsafe_ptr(),
            row_index.unsafe_ptr(),
            # The stat-plane pointer of the `blockIdx.x > 0` branch. With
            # `grid.x = 1` that branch is unreachable and the pointer is
            # never dereferenced; `temp_index` stands in because
            # `enqueue_function` refuses two mutable arguments derived from
            # ONE allocation (the aliasing note on
            # `split_and_make_sequence_kernel`), so `row_index` cannot be
            # passed twice.
            temp_index.unsafe_ptr(),
            Int32(0),
            grid_dim=(1, n_leaf_slots, 1),
            block_dim=(GATHER_INPLACE_BLOCK, 1, 1),
        )
        return 1

    # `numBlocks.x = (leavesCount > 4 ? 2 : 4) * TArchProps::SMCount()`
    # (`split_points.cu:220`, `:242`), the same occupancy target the full
    # launcher takes for the same two kernels.
    var sm = sm_count
    if sm < 0:
        sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var grid_x = split_points_grid_x(n_leaf_slots, sm)

    ctx.enqueue_function[copy_index_in_leaves_kernel](
        leaves.unsafe_ptr(),
        part_offset.unsafe_ptr(),
        part_size.unsafe_ptr(),
        row_index.unsafe_ptr(),
        temp_index.unsafe_ptr(),
        grid_dim=(grid_x, n_leaf_slots, 1),
        block_dim=(LEAF_COPY_BLOCK, 1, 1),
    )
    ctx.enqueue_function[gather_index_in_leaves_kernel](
        leaves.unsafe_ptr(),
        part_offset.unsafe_ptr(),
        part_size.unsafe_ptr(),
        temp_index.unsafe_ptr(),
        gather_map.unsafe_ptr(),
        row_index.unsafe_ptr(),
        grid_dim=(grid_x, n_leaf_slots, 1),
        block_dim=(LEAF_COPY_BLOCK, 1, 1),
    )
    return 2
