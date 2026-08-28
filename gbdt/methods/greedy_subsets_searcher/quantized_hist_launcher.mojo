"""Launcher for the quantized shared-histogram family (DEV 1911-1914).

The host half of `kernel/hist_quantized_shared.mojo`: the per-level
quantize pass, the feature-group/replica grid (DEV 1913/1914), and the
dequantizing bridge. Called by the NON-SYMMETRIC driver
(`greedy_search_helper_depthwise.mojo`) in place of
`launch_histograms_for_blocks` when -- and only when --

  * the build is FAST on a column `greedy_quantized_hist_for` claims
    (`QUANTIZED_HIST_LIVE` is comptime False under IDENTICAL, so the
    IDENTICAL column cannot reach this file at all), AND
  * the dataset's shape fits the family (`quantized_hist_shape_ok`):
    every policy block is ONE-BYTE and `stat_count == 2`. Anything else
    falls back to the standing arms AT THE CALL SITE -- refusing the
    shape to the old path, never guessing at it, the same split
    `launch_histograms_for_blocks` already makes for the fused 8-bit arm.

The shape test is a SHAPE test, not a vendor conditional: the vendor
question is the comptime kernel-matrix row, per the standing rule that
vendor divergence lives in named rows only.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_data.feature_blocks import PolicyBlock
from gbdt.gpu_data.grid_policy import POLICY_ONE_BYTE
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    DeviceBlock,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_quantized_shared import (
    QH_BLOCK,
    QH_GROUP_FEATURES,
    QH_MIN_ITEMS_PER_BLOCK,
    QH_MIN_TOTAL_BLOCKS,
    QH_STATS,
    qh_hist_gather_kernel,
    qh_hist_kernel,
    qh_write_hist_kernel,
    quantize_pair_kernel,
)
from mojo_only.kernel_matrix import (
    TARGET_COLUMN,
    greedy_quantized_hist_for,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

#: THE ONE TRUTH the driver keys its dispatch on. Comptime, so the
#: IDENTICAL build folds every consumer away and executes the pre-round
#: schedule byte for byte.
comptime QUANTIZED_HIST_LIVE = greedy_quantized_hist_for[
    TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
]()


def quantized_hist_shape_ok(
    blocks: List[PolicyBlock], stat_count: Int
) raises -> Bool:
    """Whether this fit's grid fits the quantized family: one-byte blocks
    only (the packed-pair kernels decode 4 one-byte features per cindex
    word; half-byte/binary words pack 8/32 and would read garbage bins --
    DEVIATION 1910's own semantics argument) and exactly two stat planes
    (the packed word holds two; multi-stat keeps the PASS route, the same
    refusal the fused 8-bit arm makes). Evaluated ONCE per fit, off the
    layout, before any launch."""
    if stat_count != QH_STATS:
        return False
    if len(blocks) == 0:
        return False
    for b in range(len(blocks)):
        if blocks[b].policy != POLICY_ONE_BYTE:
            return False
    return True


def qh_replicas(
    groups: Int, n_live: Int, sm_count: Int, n_rows: Int, gather: Bool
) raises -> Int:
    """DEV 1914: row-replicas per feature group, floored and capped.

    Three numbers, each cited:

      * the OCCUPANCY target is CatBoost's own
        `CeilDivide(maxActiveBlocks, numBlocks.x * numBlocks.y)` with
        `maxActiveBlocks = 2 * SMCount` and the gather arm doubled
        (`hist_one_byte.cu:291, 356` -- `replication_for`'s port,
        restated here because this family's grid has no stat axis);
      * the MIN-WORK CAP is XGBoost's `grid_size = min(occupancy_blocks,
        ceil(items / kMinItemsPerBlock))` (`histogram.cu:405-419`,
        recon a): replicas never exceed what the MEAN partition can feed
        at 8192 rows per block -- the in-kernel `active_block_count`
        then trims per-partition exactly;
      * the cap never pulls the TOTAL grid under LightGBM's
        `min_grid_dim_y_ = 160` floor
        (`cuda_histogram_constructor.hpp:152`, recon note): launch
        geometry stays device-filling on late small levels instead of
        going launch-bound.

    FAST-only by reachability (`QUANTIZED_HIST_LIVE` gates every caller),
    so the device's own `sm_count` is correct here -- there is no
    IDENTICAL arm to pin it for.
    """
    var max_active = 2 * sm_count
    if gather:
        # their gather launches divide 2x maxActiveBlocks -- an
        # indirected fetch stalls longer, so it gets more blocks to hide
        # behind (`hist_one_byte.cu:356`)
        max_active = 2 * max_active
    var base = groups * n_live
    if base < 1:
        base = 1
    var rep = (max_active + base - 1) // base
    if rep < 1:
        rep = 1

    var live = n_live
    if live < 1:
        live = 1
    var mean_rows = (n_rows + live - 1) // live
    var work_cap = (
        mean_rows + QH_MIN_ITEMS_PER_BLOCK - 1
    ) // QH_MIN_ITEMS_PER_BLOCK
    if work_cap < 1:
        work_cap = 1
    var floor_rep = (QH_MIN_TOTAL_BLOCKS + base - 1) // base
    if work_cap < floor_rep:
        work_cap = floor_rep
    if rep > work_cap:
        rep = work_cap
    return rep


def launch_quantized_histograms[ridx_stats: Bool = False](
    ctx: DeviceContext,
    mut blocks: List[DeviceBlock],
    depth: Int,
    n_live: Int,
    n_rows: Int,
    stat_count: Int,
    sm_count: Int,
    fixed_scale: MutPointer[Float32, MutAnyOrigin],
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
    mut q_stats: DeviceBuffer[DType.uint64],
    mut q_acc: DeviceBuffer[DType.int32],
    mut hist: DeviceBuffer[DType.float32],
    hist_cells_per_leaf: Int,
) raises:
    """The quantized family's whole level: quantize the pairs for the
    partitions being built (DEV 1911), one shared-histogram launch per
    one-byte policy block (DEV 1912/1913/1914), then the dequantizing
    bridge into the flat histogram. Same contract as
    `launch_histograms_for_blocks`: `ids`/`n_live` are the NON-EMPTY
    compute set and the caller must not call at all when it is empty.

    The caller has already verified `quantized_hist_shape_ok`, so every
    block here is one-byte; `fixed_scale` rides in `q_stats`'
    quantization and in the bridge's division -- the same per-tree
    `choose_scale` value every other arm reads, which is the whole
    overflow contract (`kernel/hist_quantized_shared.mojo`'s banner).

    NO scratch zero and NO separate bridge-per-block: the accumulator is
    self-cleaning (the bridge zeroes what it reads) and dense-leaf-major
    across ALL blocks -- a block writes `block_first_bin + fold_off +
    bin` into the leaf's full `hist_cells_per_leaf` span, so one bridge
    launch at the end covers every policy block's cells at once.
    """
    if stat_count != QH_STATS:
        raise Error(
            "launch_quantized_histograms is two-stat by construction;"
            " the caller's shape test admitted stat_count="
            + String(stat_count)
        )

    # ---- DEV 1911: the packed-pair plane, for exactly the compute
    # partitions. Grid-stride on x, one row of blocks per compute leaf.
    var qx = (n_rows + QH_BLOCK - 1) // QH_BLOCK
    if qx > 4 * sm_count:
        qx = 4 * sm_count
    if qx < 1:
        qx = 1
    ctx.enqueue_function[quantize_pair_kernel[ridx_stats]](
        stats.unsafe_ptr(),
        Int32(n_rows),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        ids.unsafe_ptr(),
        row_index.unsafe_ptr(),
        q_stats.unsafe_ptr(),
        fixed_scale,
        grid_dim=(qx, n_live, 1),
        block_dim=(QH_BLOCK, 1, 1),
    )

    # ---- DEV 1912/1913/1914: one launch per policy block, exactly as
    # `launch_histograms_for_blocks` walks them, advancing the flat bin
    # cursor by each block's fold count (the flat-bin-order invariant,
    # `feature_blocks.mojo`'s banner).
    var block_first_bin = 0
    for b in range(len(blocks)):
        ref blk = blocks[b]
        var base = n_rows * blk.first_column
        var line = n_rows

        var groups = (
            blk.n_features + QH_GROUP_FEATURES - 1
        ) // QH_GROUP_FEATURES
        var replicas = qh_replicas(
            groups, n_live, sm_count, n_rows, gather=(depth > 0)
        )

        if depth == 0:
            ctx.enqueue_function[qh_hist_kernel](
                blk.folds.unsafe_ptr(),
                blk.fold_off.unsafe_ptr(),
                Int32(blk.n_features),
                cindex.unsafe_ptr(),
                Int32(line),
                Int32(base),
                q_stats.unsafe_ptr(),
                p_off.unsafe_ptr(),
                p_sz.unsafe_ptr(),
                ids.unsafe_ptr(),
                q_acc.unsafe_ptr(),
                Int32(block_first_bin),
                Int32(hist_cells_per_leaf),
                grid_dim=(groups * replicas, n_live, 1),
                block_dim=(QH_BLOCK, 1, 1),
            )
        else:
            ctx.enqueue_function[qh_hist_gather_kernel](
                blk.folds.unsafe_ptr(),
                blk.fold_off.unsafe_ptr(),
                Int32(blk.n_features),
                cindex.unsafe_ptr(),
                Int32(line),
                Int32(base),
                row_index.unsafe_ptr(),
                q_stats.unsafe_ptr(),
                p_off.unsafe_ptr(),
                p_sz.unsafe_ptr(),
                ids.unsafe_ptr(),
                q_acc.unsafe_ptr(),
                Int32(block_first_bin),
                Int32(hist_cells_per_leaf),
                grid_dim=(groups * replicas, n_live, 1),
                block_dim=(QH_BLOCK, 1, 1),
            )
        block_first_bin += blk.total_folds

    # ---- the dequantizing bridge: ONE launch for the whole level's
    # built leaves, every cell stored (the flat slot must equal the
    # build's output), the accumulator zeroed where it was read.
    ctx.enqueue_function[qh_write_hist_kernel](
        ids.unsafe_ptr(),
        q_acc.unsafe_ptr(),
        fixed_scale,
        Int32(hist_cells_per_leaf),
        hist.unsafe_ptr(),
        grid_dim=((hist_cells_per_leaf + 255) // 256, n_live, stat_count),
        block_dim=(256, 1, 1),
    )
