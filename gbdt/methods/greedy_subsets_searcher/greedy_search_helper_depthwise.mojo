# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CatBoost-compatible depthwise tree growth: every live leaf selects and applies its own split."""

from gbdt.options.child_hessian import child_hessian_threshold
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import block_dim, block_idx, thread_idx
from core.device_zero import enqueue_fill

from checks.fixed_point import choose_scale
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from gbdt.gpu_lib.gpu_manager import TCudaManager
from gbdt.data.leaf_path import TLeafPath, split_leaf_path
from gbdt.data.permutation import TRandom
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.gpu_structures import CFeature
from gbdt.gpu_util.partitions_reduce import compute_partition_stats
from gbdt.gpu_util.kernel.partition_stats_gather import (
    compute_partition_stats_gather,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    CFEATURE_BYTES,
    ridx_schedule_pays,
    TTreeWorkspace,
    acc_i32_is_live,
    compute_target_std_dev,
    enqueue_leaf_iota,
    enqueue_snap_gradients,
    launch_histograms_for_blocks,
    resolve_split,
)
from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    LEAFWISE_SCORE_BLOCK_SIZE,
    compute_optimal_split_kernel,
    compute_optimal_splits_region_kernel,
)
from gbdt.methods.greedy_subsets_searcher.quantized_hist_launcher import (
    QUANTIZED_HIST_LIVE,
    QH_MODE_SKIP,
    launch_quantized_histograms,
    quantized_hist_shape_ok,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_resolve import (
    WINNER_FOLD_BLOCK_SIZE,
    WINNER_RECORD_WORDS,
    WINNER_STATUS_BIN_OUT_OF_RANGE,
    WINNER_STATUS_DEFINED,
    leaf_winner_fold_kernel,
)
from std.memory import bitcast, memcpy
from checks.soft_f64 import (
    sf64_add,
    sf64_div,
    sf64_from_f32,
    sf64_gt,
    sf64_is_nan,
    sf64_to_f32,
)
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.builtin.sort import sort
from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    choose_scale_kernel,
    copy_histograms_kernel,
    copy_histograms_vec4_kernel,
    scan_histograms_kernel,
    substract_histograms_kernel,
    substract_histograms_vec4_kernel,
    zero_histograms_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_points import (
    SPLIT_BLOCK_SIZE,
    launch_reorder_in_leaves,
    launch_stable_partition,
    split_and_make_sequence_kernel,
    split_points_grid_x,
    update_partition_stats_from_split_kernel,
    update_partitions_after_split_kernel,
)
from gbdt.gpu_util.kernel.reorder_single_pass import (
    launch_stable_partition_routed,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_points_ridx import (
    launch_reorder_index_only,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_chain_fused import (
    FUSED_CHAIN_BLOCK,
    FUSED_COPY_BLOCK,
    DW_FEAT_WORDS,
    DW_SELECT_BLOCK,
    DW_WINNER_WORDS,
    dw_select_splits_kernel,
    fused_copy_back_kernel,
    fused_flags_count_kernel,
    fused_place_scatter_kernel,
    fused_scan_update_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.dw2_level import (
    DW2_COPY_BLOCK,
    DW2_PART_BLOCK,
    DW2_SCAN_BLOCK,
    DW2_SCAN_FT,
    dw2_copy_back_kernel,
    dw2_flags_count_kernel,
    dw2_part_max_chunks,
    dw2_place_scatter_kernel,
    dw2_scan_histograms_smem_kernel,
    dw2_scan_update_kernel,
)
from checks.kernel_matrix import TARGET_COLUMN, ridx_only_splits_for
from gbdt.methods.greedy_subsets_searcher.depthwise_stage_times import (
    StageTimes,
)
from gbdt.methods.greedy_subsets_searcher.model_builder import (
    build_non_symmetric_tree,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_lossguide import (
    select_leaves_to_split_traced as lossguide_select_leaves_to_split_traced,
)
from gbdt.methods.greedy_subsets_searcher.points_subsets import (
    EHistogramsType,
    TBestSplitProperties,
    TLeaf,
)
from gbdt.methods.greedy_subsets_searcher.split_properties_helper import (
    HISTOGRAMS_CURRENT_PATH,
    HISTOGRAMS_PREVIOUS_PATH,
    HISTOGRAMS_ZEROES,
    LeafRecord,
    build_necessary_histograms,
    non_zero_leaves,
)
from gbdt.methods.greedy_subsets_searcher.structure_searcher_options import (
    TTreeStructureSearcherOptions,
)
from gbdt.methods.helpers import (
    SPLIT_VALUE_ONE,
    SPLIT_VALUE_ZERO,
    best_split_properties_less,
)
from gbdt.models.non_symmetric_tree import TNonSymmetricTree
from core.identity_trace import IdentityTrace
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)
from gbdt.options.catboost_options import (
    GROW_DEPTHWISE,
    GROW_LOSSGUIDE,
    GROW_SYMMETRIC,
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_NEWTON_L2,
)


# ============================ DEVIATION 1901 ============================
# The mode test for the existing split-cost arms. IDENTICAL retains the
# pinned reduction and host winner fold; FAST/DETERMINISTIC use the existing
# propagation/copy/fold optimizations below. The separate unchanged-leaf
# cache reduces repeated work without changing any leaf's arithmetic.
comptime SPLIT_COST_IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

comptime DEFER_HIST_COPY_1903 = not SPLIT_COST_IDENTICAL or (
    (
        has_apple_gpu_accelerator()
        or (
            is_defined["MOJOLEARN_GBDT_ID_DEFER_COPY"]()
            and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
        )
    )
    and not is_defined["MOJOLEARN_GBDT_IDENTICAL_SPLIT_COPY"]()
)
"""DEVIATION 1903's schedule (the parent-histogram copy deferred to the
plan-time pairs that need it, the zero pass over dirty slots only): FAST
everywhere, and Apple IDENTICAL since trees-apple2 (2026-09-28). Its bit
argument is schedule-only -- every slot holds the same bytes at every
read (a deferred copy copies the same untouched parent; an elided zero
leaves the tree memset's +0.0 where the kernel would write +0.0) -- so it
holds under IDENTICAL too. The other SPLIT_COST_IDENTICAL arms (1901's
propagated partition stats, which re-associate; 1904's device fold) stay
as they were. `-D MOJOLEARN_GBDT_IDENTICAL_SPLIT_COPY` restores the
split-time copy and the full zero pass on Apple IDENTICAL. Lane
gap-trees-nv: `-D MOJOLEARN_GBDT_ID_DEFER_COPY` takes the same schedule
under IDENTICAL on NVIDIA and AMD (opt-in until its A/B); fix-g1-gbdt:
that opt-in arm also turns off under `-D MOJOLEARN_IDN_ALL_OFF` (Apple's
default stays)."""

# Cache unchanged partitions under IDENTICAL without propagating histogram
# sums (which would change rounding). CatBoost updates only split children
# in TSplitPointsKernel (split_properties_helper.cpp:918-936); this implementation keeps
# its existing two-phase pinned reduction for each changed child instead.
# A split permutes only its own disjoint range. Every other leaf therefore
# keeps the same stats bytes, offset, length and per-leaf reduction result.
# Both kernels address results through the explicit leaf-id list; changing
# grid.y does not change the x-stripe or floating-point fold for any leaf.
# FULL_PARTITION_STATS restores the previous all-leaf sweep for A/B checks.
comptime INCREMENTAL_PART_STATS = (
    SPLIT_COST_IDENTICAL
    and not is_defined["MOJOLEARN_GBDT_FULL_PARTITION_STATS"]()
)
comptime REPORT_PART_STATS_WORK = is_defined["MOJOLEARN_GBDT_PART_STATS_WORK"]()

# ============================ DEVIATION 2661 ============================
# THE PER-GROUP BIT WIDTH REACHES THE NON-SYMMETRIC DRIVERS. DEVIATION 2581
# sorts a one-byte block's 4-feature groups by width and launches one kernel
# per width present, instead of paying the whole block's widest feature for
# every group; it is the IDENTICAL default in the symmetric driver
# (`greedy_search_helper.mojo`, `SYM_GROUP_WIDTH_2581`, flipped 2026-09-11 on
# geomean 0.995). The depthwise and lossguide histogram call below binds
# neither of that launcher's trailing parameters, so these two policies have
# always taken the block-widest ladder. The workspace is the SAME
# `TTreeWorkspace`, and `refresh_layout_metadata` already fills
# `width_plans` whenever 2581 is compiled in, so the plans this needs are
# built and uploaded on this path today and simply never read.
#
# WHY NO BIT CAN MOVE: the one-byte arms accumulate per-row Int32 addends
# into the same accumulator cells, and an integer sum does not depend on
# grouping. A narrower width changes WHICH launch adds a group's addends,
# never the addends or the total, and the dequantization in the writeback
# is untouched. The same argument carried 2581 in the symmetric driver.
#
# OPT-IN (`-D MOJOLEARN_2661_NONSYM_GROUP_WIDTH=1`) until its own A/B on
# both datasets says otherwise: 2581's flip was measured on the symmetric
# policy, and a switch is decided on the lanes its code reaches.
# Istella-S is where it can pay (220 features, one one-byte block of 32
# groups); taxi's two groups are both 8-bit, so there it only changes
# launcher.
comptime NONSYM_GROUP_WIDTH_2661 = is_defined[
    "MOJOLEARN_2661_NONSYM_GROUP_WIDTH"
]()

# DEVIATION 1902: whether the non-symmetric drivers move ONLY the row
# index at a split, leaving the stat planes stationary in document order
# for the life of the fit. The row is False under IDENTICAL on every
# column; the readers' gather arms are bound where this constant is
# spelled (hist build, split apply, end-of-tree sweep) and default to the
# old body everywhere else.
comptime RIDX_IDENTICAL_MAX_FEATURES = 64
"""UNUSED since lane/no-dim-idn (the cut is `ridx_schedule_pays`; its arm B
restores this same 64). See `use_ridx` in `fit_non_symmetric_tree`. A RANGE rule, not a board
row: the ridx schedule pays one gathered stat load per row per feature group
a level walks, against one stat-plane reorder per split, so it wins on
narrow layouts and loses once the groups per level grow. Measured at 16 and
220 features only: NEEDS NEIGHBOR-SHAPE VALIDATION (32, 48, 64, 65, 96, 128
features). Bit-inert either side (same digests), so the edge only picks the
faster same-order schedule."""

comptime RIDX_ONLY_SPLITS = ridx_only_splits_for[
    TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
]()

#: FAST on Apple, DEFAULT (lane apple-fast-depthwise): the per-level split
#: chain of the row-index-only schedule in four launches instead of eight
#: (`kernel/split_chain_fused.mojo` has the mapping and why it is the same
#: permutation, partitions and stats bit for bit). Taken only for a
#: Depthwise tree on the row-index-only schedule (`use_ridx`, not
#: Lossguide); the stat-moving schedule and Lossguide keep the old chain.
#: M3 A/B (plain FAST vs this + DW_NO_LEVEL_SYNC, n=2): dw-fcns-taxi
#: 14,933 -> 14,120 ms (-5.4%), auc .6322 -> .6325; dw-fcns-istella
#: 19,561 -> 19,522 ms (neutral), auc .9832 both.
#: `-D MOJOLEARN_GBDT_DW_FUSED_CHAIN_OFF` turns it off (and with it
#: DW_NO_LEVEL_SYNC, which stacks on it); the old
#: `-D MOJOLEARN_GBDT_DW_FUSED_CHAIN` is harmless.
comptime DW_FUSED_CHAIN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
# F12/depthwise M3 2026-10-06: 4 scored caller times; B/A
# 0.8646..1.1062 (mixed/regressing); existing default retained.
# Scored FAST quality 2/2 within existing bands; PASS.
# One warmup/one score; caller67d0efb29; exact cases/builds/hashes:
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/depthwise.
# Compilation/identity reused. No combined-toggle/full-board claim.
    and not is_defined["MOJOLEARN_GBDT_DW_FUSED_CHAIN_OFF"]()
)

#: FAST on Apple, opt-in, stacked on DW_FUSED_CHAIN (lane
#: apple-fast-depthwise, second pass): a Depthwise level takes ONE host wait
#: instead of two. The winner fold's records stay on the device;
#: `dw_select_splits_kernel` makes the Depthwise selection (DEFINED and
#: `Gain < 0`) and the split payload (left/right ids, bin, cell, CFeature
#: words) there, the fused chain runs over a grid of every scored leaf with
#: the split count read on the device, and the winners, the new leaf sizes
#: and the count come home in one wait. The host then replays the same
#: selection from the same records (and checks the count), so leaves,
#: paths and terminal marks are unchanged. Integer moves only: the same
#: tree bit for bit. Lossguide, `min_split_gain >= 0` and a level with
#: nothing to score keep the two-wait schedule.
#: DEFAULT with DW_FUSED_CHAIN (numbers above). `-D
#: MOJOLEARN_GBDT_DW_NO_LEVEL_SYNC_OFF` turns it off; the old `-D MOJOLEARN_GBDT_DW_NO_LEVEL_SYNC` is harmless.
comptime DW_NO_LEVEL_SYNC = DW_FUSED_CHAIN and not is_defined[
    "MOJOLEARN_GBDT_DW_NO_LEVEL_SYNC_OFF"
]()

# ---- lane apple-fast-dwgap2: FAST Apple arms --------------------------------
# (`kernel/dw2_level.mojo` has the kernels and the argument for each.)
#: The fused split chain at four rows per thread, aligned 16-byte row-index
#: and 4-byte flag accesses. Same stable partition: same tree. FAST on
#: Apple, default on (needs DW_FUSED_CHAIN, itself a FAST Apple default).
#: Default since the M3 A/B (lane/apple-fast-dwgap2 049899d81, n=2, SEP):
#: depthwise taxi 12,711 -> 11,361 ms (-10.6%), istella 17,154 -> 16,539 ms
#: (-3.6%), auc within spread. Off: `-D MOJOLEARN_GBDT_DW2_PART_VEC4_OFF`.
#: The old opt-in define `-D MOJOLEARN_GBDT_DW2_PART_VEC4` is harmless.
comptime DW2_PART_VEC4 = DW_FUSED_CHAIN and not is_defined[
    "MOJOLEARN_GBDT_DW2_PART_VEC4_OFF"
]()
#: The histogram prefix scan over a shared-memory copy of 16 features'
#: cells; the same serial fold, so the same bits. FAST on Apple, default
#: on. Default since the M3 A/B (lane/apple-fast-dwgap2 049899d81, n=2,
#: SEP): depthwise taxi 13,861 -> 13,127 ms (-5.3%), istella 17,237 ->
#: 17,014 ms (-1.3%), auc within spread. Off:
#: `-D MOJOLEARN_GBDT_DW2_SCAN_SMEM_OFF`. The old opt-in define
#: `-D MOJOLEARN_GBDT_DW2_SCAN_SMEM` is harmless.
#: lane/fam-gbdt (2026-10-04), IDN_NS_SCALE_DEVICE: IDENTICAL, every vendor,
#: default on. The Depthwise / Lossguide driver derives its fixed-point
#: scale on the device (`choose_scale_kernel`, DEVIATION 95, the kernel the
#: symmetric driver already launches) when the caller hands the magnitudes
#: buffer, so the boosting loop no longer drains once per tree to read two
#: floats back. Same bits: the kernel is the host `choose_scale` as an exact
#: integer search (its docstring), reading the same two magnitudes.
#: `-D MOJOLEARN_IDN_GBDT_NS_SCALE_DEVICE_OFF` (or the master
#: `-D MOJOLEARN_IDN_ALL_OFF`) restores the per-tree drain and host scale.
comptime IDN_NS_SCALE_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_GBDT_NS_SCALE_DEVICE_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)

comptime DW2_SCAN_SMEM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_GBDT_DW2_SCAN_SMEM_OFF"]()
)


def _dw_dev_u32(
    buf: DeviceBuffer[DType.uint32], offset: Int
) -> MutPointer[UInt32, MutAnyOrigin]:
    """A device buffer's pointer, `offset` elements in, as the kernel
    parameter type (`enqueue_fill`'s address idiom)."""
    return (
        MutPointer[UInt32, MutAnyOrigin](
            unsafe_from_address=Int(buf.unsafe_ptr())
        )
        + offset
    )


def _launch_fused_split_chain[
    GUARD: Bool = False
](
    ctx: DeviceContext,
    n_split: Int,
    n_rows: Int,
    sm_count: Int,
    stat_count: Int,
    hist_cells_per_leaf: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut temp_index: DeviceBuffer[DType.uint32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut d_left: DeviceBuffer[DType.uint32],
    mut d_right: DeviceBuffer[DType.uint32],
    mut d_win_cells: DeviceBuffer[DType.uint32],
    mut sp_feats: DeviceBuffer[DType.uint32],
    mut sp_bins: DeviceBuffer[DType.uint32],
    mut flags: DeviceBuffer[DType.uint8],
    mut chunk_zeros: DeviceBuffer[DType.uint32],
    mut chunk_offsets: DeviceBuffer[DType.uint32],
    mut leaf_zeros: DeviceBuffer[DType.uint32],
    mut slot_off: DeviceBuffer[DType.uint32],
    mut slot_sz: DeviceBuffer[DType.uint32],
    mut hist: DeviceBuffer[DType.float32],
    mut part_stats: DeviceBuffer[DType.float32],
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
) raises:
    """DW_FUSED_CHAIN's four launches. Grids as the chain they replace: the
    chunk grid is `launch_stable_partition`'s (`max_chunks` off `n_rows`,
    the workspace's stride for `chunk_zeros`, capped at the machine-sized
    `split_points_grid_x`), and the kernels stride the rest. GUARD
    (DW_NO_LEVEL_SYNC): `n_split` is the scored-leaf count, an upper bound,
    and every block past `n_split_dev[0]` returns at once."""
    if n_split <= 0:
        return
    comptime if DW2_PART_VEC4:
        # lane apple-fast-dwgap2: the same four launches at four rows per
        # thread (`kernel/dw2_level.mojo`)
        var max_chunks4 = dw2_part_max_chunks(n_rows)
        var grid_x4 = split_points_grid_x(n_split, sm_count)
        var chunk_grid4 = max_chunks4
        if grid_x4 < chunk_grid4:
            chunk_grid4 = grid_x4
        ctx.enqueue_function[dw2_flags_count_kernel[GUARD]](
            cindex.unsafe_ptr(),
            row_index.unsafe_ptr(),
            p_off.unsafe_ptr(),
            p_sz.unsafe_ptr(),
            d_left.unsafe_ptr(),
            sp_feats.unsafe_ptr().bitcast[CFeature](),
            sp_bins.unsafe_ptr(),
            flags.unsafe_ptr(),
            chunk_zeros.unsafe_ptr(),
            Int32(max_chunks4),
            n_split_dev,
            grid_dim=(chunk_grid4, n_split, 1),
            block_dim=(DW2_PART_BLOCK, 1, 1),
        )
        ctx.enqueue_function[dw2_scan_update_kernel[GUARD]](
            d_left.unsafe_ptr(),
            d_right.unsafe_ptr(),
            p_off.unsafe_ptr(),
            p_sz.unsafe_ptr(),
            chunk_zeros.unsafe_ptr(),
            chunk_offsets.unsafe_ptr(),
            leaf_zeros.unsafe_ptr(),
            slot_off.unsafe_ptr(),
            slot_sz.unsafe_ptr(),
            d_win_cells.unsafe_ptr(),
            sp_feats.unsafe_ptr().bitcast[CFeature](),
            Int32(hist_cells_per_leaf),
            Int32(stat_count),
            hist.unsafe_ptr(),
            part_stats.unsafe_ptr(),
            Int32(max_chunks4),
            n_split_dev,
            grid_dim=(1, n_split, 1),
            block_dim=(DW2_PART_BLOCK, 1, 1),
        )
        ctx.enqueue_function[dw2_place_scatter_kernel[GUARD]](
            slot_off.unsafe_ptr(),
            slot_sz.unsafe_ptr(),
            flags.unsafe_ptr(),
            chunk_offsets.unsafe_ptr(),
            leaf_zeros.unsafe_ptr(),
            row_index.unsafe_ptr(),
            temp_index.unsafe_ptr(),
            Int32(max_chunks4),
            n_split_dev,
            grid_dim=(chunk_grid4, n_split, 1),
            block_dim=(DW2_PART_BLOCK, 1, 1),
        )
        ctx.enqueue_function[dw2_copy_back_kernel[GUARD]](
            slot_off.unsafe_ptr(),
            slot_sz.unsafe_ptr(),
            temp_index.unsafe_ptr(),
            row_index.unsafe_ptr(),
            n_split_dev,
            grid_dim=(grid_x4, n_split, 1),
            block_dim=(DW2_COPY_BLOCK, 1, 1),
        )
        return
    var max_chunks = (n_rows + FUSED_CHAIN_BLOCK - 1) // FUSED_CHAIN_BLOCK
    if max_chunks < 1:
        max_chunks = 1
    var grid_x = split_points_grid_x(n_split, sm_count)
    var chunk_grid = max_chunks
    if grid_x < chunk_grid:
        chunk_grid = grid_x
    ctx.enqueue_function[fused_flags_count_kernel[GUARD]](
        cindex.unsafe_ptr(),
        row_index.unsafe_ptr(),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        d_left.unsafe_ptr(),
        sp_feats.unsafe_ptr().bitcast[CFeature](),
        sp_bins.unsafe_ptr(),
        flags.unsafe_ptr(),
        chunk_zeros.unsafe_ptr(),
        Int32(max_chunks),
        n_split_dev,
        grid_dim=(chunk_grid, n_split, 1),
        block_dim=(FUSED_CHAIN_BLOCK, 1, 1),
    )
    ctx.enqueue_function[fused_scan_update_kernel[GUARD]](
        d_left.unsafe_ptr(),
        d_right.unsafe_ptr(),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        chunk_zeros.unsafe_ptr(),
        chunk_offsets.unsafe_ptr(),
        leaf_zeros.unsafe_ptr(),
        slot_off.unsafe_ptr(),
        slot_sz.unsafe_ptr(),
        d_win_cells.unsafe_ptr(),
        sp_feats.unsafe_ptr().bitcast[CFeature](),
        Int32(hist_cells_per_leaf),
        Int32(stat_count),
        hist.unsafe_ptr(),
        part_stats.unsafe_ptr(),
        Int32(max_chunks),
        n_split_dev,
        grid_dim=(1, n_split, 1),
        block_dim=(FUSED_CHAIN_BLOCK, 1, 1),
    )
    ctx.enqueue_function[fused_place_scatter_kernel[GUARD]](
        slot_off.unsafe_ptr(),
        slot_sz.unsafe_ptr(),
        flags.unsafe_ptr(),
        chunk_offsets.unsafe_ptr(),
        leaf_zeros.unsafe_ptr(),
        row_index.unsafe_ptr(),
        temp_index.unsafe_ptr(),
        Int32(max_chunks),
        n_split_dev,
        grid_dim=(chunk_grid, n_split, 1),
        block_dim=(FUSED_CHAIN_BLOCK, 1, 1),
    )
    ctx.enqueue_function[fused_copy_back_kernel[GUARD]](
        slot_off.unsafe_ptr(),
        slot_sz.unsafe_ptr(),
        temp_index.unsafe_ptr(),
        row_index.unsafe_ptr(),
        n_split_dev,
        grid_dim=(grid_x, n_split, 1),
        block_dim=(FUSED_COPY_BLOCK, 1, 1),
    )
# ========================================================================


struct TBinFeatureTable(Copyable, Movable):
    """Every bin-feature resolved to `(feature, bin)`, once per tree.

    Their `ToSplit(FeaturesManager, props)` (`methods/helpers.cpp:164-170`)
    does this resolution on the host, one candidate at a time, out of the
    features manager. `resolve_split` already does it here -- but it is an
    O(features) WALK per call, and the depthwise host reduce resolves
    `argmax_blocks * leavesToVisit.size()` records per level where the
    symmetric one resolved ONE per level.

    So the walk is done once for every bin-feature at the top of the fit and
    the level loop indexes it. That is a change of ALGORITHM on the host and
    therefore a deviation, and it is bit-inert by
    construction: `resolve_split` is the function that fills the table, so
    the table cannot disagree with it. `checks/depthwise_check.mojo`
    claim 1 asserts the two agree cell for cell anyway, because "cannot
    disagree by construction" is exactly the sentence this repository has
    been wrong about before.

    archive/reference/HOST_AND_DEVICE.md rule one holds: `len(feature)` is the total bin
    count, which scales with FEATURES times BORDERS and never with rows.
    """

    var feature: List[Int32]
    var bin: List[Int32]
    var one_hot: List[Bool]
    var folds: List[Int32]
    """The owning feature's bin count, carried so `to_split` can apply their
    clamp (`helpers.cpp:165-170`) without a second walk of the layout."""

    def __init__(out self, layout: CompressedIndexLayout) raises:
        self.feature = List[Int32]()
        self.bin = List[Int32]()
        self.one_hot = List[Bool]()
        self.folds = List[Int32]()
        for bf in range(layout.hist_cells):  # small-loop(hist_cells: bin-feature table rows, once per tree): layout metadata resolved from feature descriptors
            var choice = resolve_split(layout, bf)
            self.feature.append(Int32(choice.feature))
            self.bin.append(Int32(choice.bin))
            self.one_hot.append(
                layout.features[choice.feature].one_hot_feature
            )
            self.folds.append(Int32(Int(layout.features[choice.feature].folds)))

    def to_split(self, bin_feature: Int) raises -> TBinarySplit:
        """Their `ToSplit` (`methods/helpers.cpp:164-170`).

        `SplitType` is `TakeBin` for a one-hot feature and `TakeGreater`
        otherwise, which is their `manager.IsCat(props.FeatureId)` test read
        off the layout instead of off the features manager -- the same
        substitution `run_tree_layout` makes at its own `ToSplit` site.
        """
        # ============ THEIR CLAMP, `helpers.cpp:157-171` ============
        #     split.BinIdx = Min<ui32>(GetBinCount(id), props.BinId);      // cat
        #     split.BinIdx = Min<ui32>(GetBorders(id).size() - 1, props.BinId);
        # under their own comment "Float arithmetic could generate empty bin
        # splits for ctrs". It was missing here while this docstring cited
        # the very lines that contain it.
        #
        # INERT ON THIS PATH BY CONSTRUCTION -- `resolve_split` derives the
        # bin as `bin_feature - first_fold_index`, which is strictly below
        # `folds` by definition, so the clamp can never bind. Implemented anyway,
        # because "inert by construction" is a claim about a DIFFERENT
        # function, and the day a bin arrives from anywhere but
        # `resolve_split` the clamp is the thing that was supposed to be
        # here. `helpers.mojo:244-262` is the canonical implementation and carries the
        # argument for the two arms' asymmetry.
        var max_bin = Int32(self.folds[bin_feature]) - 1
        if self.one_hot[bin_feature]:
            max_bin = Int32(self.folds[bin_feature])
        var bin = self.bin[bin_feature]
        if bin > max_bin:
            bin = max_bin
        return TBinarySplit(
            self.feature[bin_feature],
            bin,
            Int32(
                BIN_SPLIT_TAKE_BIN
            ) if self.one_hot[bin_feature] else Int32(BIN_SPLIT_TAKE_GREATER),
        )


# trees-apple2: the id-list arena's slots, grouped so each phase's lists sit
# in one run: plan time (copy pairs, zero, build, all, visit, subtract pair)
# and split time (split pair, winning cells).
comptime IDS_SLOT_COPY_SRC = 0
comptime IDS_SLOT_COPY_DST = 1
comptime IDS_SLOT_ZERO = 2
comptime IDS_SLOT_IDS = 3
comptime IDS_SLOT_ALL = 4
comptime IDS_SLOT_VISIT = 5
comptime IDS_SLOT_SUB_LEFT = 6
comptime IDS_SLOT_SUB_RIGHT = 7
# the split features (`CFeature` records, their `splitsFeatures`) take
# `IDS_FEAT_SLOTS` slots: one record per splitting leaf is
# `CFEATURE_BYTES / 4` words
comptime IDS_FEAT_SLOTS = CFEATURE_BYTES // 4
comptime IDS_SLOT_SP_FEATS = 8
comptime IDS_SLOT_SP_BINS = IDS_SLOT_SP_FEATS + IDS_FEAT_SLOTS
comptime IDS_SLOT_LEFT = IDS_SLOT_SP_BINS + 1
comptime IDS_SLOT_RIGHT = IDS_SLOT_SP_BINS + 2
comptime IDS_SLOT_WIN = IDS_SLOT_SP_BINS + 3
comptime IDS_SLOTS = IDS_SLOT_SP_BINS + 4

comptime ID_UPLOAD_COALESCE = not is_defined[
    "MOJOLEARN_GBDT_ID_UPLOADS_SEPARATE"
]()
"""trees-apple2 (2026-09-28): a phase's small id lists (Depthwise/Lossguide)
go to the device in ONE copy per run of adjacent arena slots instead of one
copy each. Each Metal enqueue costs ~20 us of host time (M4,
`bench/speed/metal_enqueue_cost_main.mojo`) and a Lossguide leaf split
staged six plan-time lists and four split-time ones. Same words land in
the same device buffers before the same kernels, so no bit can move.
`-D MOJOLEARN_GBDT_ID_UPLOADS_SEPARATE` copies each slot on its own (the
old count, same code otherwise)."""


def _upload_id_slots(
    ctx: DeviceContext,
    d_arena: DeviceBuffer[DType.uint32],
    h_arena: MutPointer[UInt32, MutUntrackedOrigin],
    host: List[MutPointer[UInt32, MutUntrackedOrigin]],
    max_leaves: Int,
    var slots: List[Int],
) raises:
    """Stage the named slots' host lists (`host[slot]`, `max_leaves` words
    each, the whole buffer as every former per-list copy moved) into the
    arena's staging buffer and copy each run of adjacent slots in one
    `enqueue_copy`. Only the named slots are written on the device."""
    if len(slots) == 0:
        return
    sort(slots)
    for slot in slots:
        memcpy(
            dest=h_arena.unsafe_offset(slot * max_leaves),
            src=host[slot],
            count=max_leaves,
        )
    var i = 0
    while i < len(slots):
        var j = i
        comptime if ID_UPLOAD_COALESCE:
            while j + 1 < len(slots) and slots[j + 1] == slots[j] + 1:
                j += 1
        var first = slots[i]
        var n = slots[j] - first + 1
        ctx.enqueue_copy(
            dst_buf=d_arena.create_sub_buffer[DType.uint32](
                first * max_leaves, n * max_leaves
            ),
            src_ptr=h_arena.unsafe_offset(first * max_leaves),
        )
        i = j + 1


struct TDepthwiseWorkspace(Movable):
    """The buffers `TTreeWorkspace` does not have, and only those.

    `TTreeWorkspace` (the symmetric lane's pool, `greedy_search_helper.mojo`)
    already owns every large plane: histograms, the fixed-point accumulator,
    the partitions, the stat partials, the row flags and the reorder scratch,
    the compressed-index tables, the bin-feature planes and the scale. All of
    it is shape-keyed and reused verbatim here.

    Four things it cannot serve, all of them because the SYMMETRIC level has
    one winner and one dense id list where a depthwise level has neither:

    * `region_score` / `region_bin` -- their `bestProps`, sized
      `argmaxBlockCount * numScoreBlocks` (`greedy_search_helper.cpp:441`).
      The pool's `out_score` / `out_bin` are sized `argmaxBlockCount`,
      because `numScoreBlocks` is 1 for SymmetricTree.
    * `d_visit` -- their `leafIds` (`:472`), the leaves being scored.
    * `d_left` / `d_right` -- their `leftIdsGpu` / `rightIdsGpu` (`:895-896`),
      which for a symmetric level are the dense prefix and the dense prefix
      plus half, and so needed no buffer at all.
    * `h_part_stats` -- their `ReadReduce(currentPartStats)` (`:632`), the
      once-per-tree drain that becomes the leaf values.
    """

    var max_leaves_key: Int
    var stat_count_key: Int
    var argmax_blocks_key: Int
    var final_ready: Bool
    """NS_INHERIT_PARTITION: the last fit left its leaves' row ranges in
    `final_offsets` / `final_sizes`, in the MODEL's leaf order."""
    var final_offsets: List[Int]
    var final_sizes: List[Int]

    var region_score: DeviceBuffer[DType.float32]
    var region_bin: DeviceBuffer[DType.uint32]
    var h_region_score: HostBuffer[DType.float32]
    var h_region_bin: HostBuffer[DType.uint32]
    var d_visit: DeviceBuffer[DType.uint32]
    var h_visit: HostBuffer[DType.uint32]
    var d_left: DeviceBuffer[DType.uint32]
    var h_left: HostBuffer[DType.uint32]
    var d_right: DeviceBuffer[DType.uint32]
    var h_right: HostBuffer[DType.uint32]
    var d_ids: DeviceBuffer[DType.uint32]
    var h_ids: HostBuffer[DType.uint32]
    # ============================ DEVIATION 261 ============================
    # THREE ID LISTS PER LEVEL, THREE STAGING PAIRS -- not one reused.
    # A level stages three host-built id lists before its first drain: the
    # ZERO set (`plan.compute_ids`, for `zero_histograms_kernel`), the
    # BUILD set (`non_zero`, for the histogram kernels and the scan) and
    # the ALL set (`0..len(leaves)`, for `compute_partition_stats`). They
    # used to share `h_ids`/`d_ids`: the host wrote list 2 into `h_ids`
    # while list 1's `enqueue_copy` was still QUEUED, so the device-side
    # copy could read list 2 (or 3) -- the step-33 race class
    # (`[[mojo-buffer-freed-at-last-use]]`'s host-staging sibling; the
    # symmetric lane's DEVIATION 134 is the same mechanism). Invisible to
    # every lane gate (4,096 rows, one tree, and a traced run drains at
    # every record) and found 2026-08-23 by the boosting loop's
    # run-to-run control at 20,000 rows x 24 features x 128 borders: the
    # shared-Int32 histogram arms -- whose kernels are the fast ones, so
    # the host reached the next write first -- disagreed with themselves
    # from depth 4 on, while the float arms happened to land on the right
    # side of the race. The symmetric driver never had it: its lists are
    # `ids_compute` / `dense_ids` / the drained `h_off`/`h_sz` pairs. Fix:
    # one pair per list, so no host buffer is rewritten under a queued
    # copy; the two drains their control plane already takes per level
    # (`score.read`, `split.sizes`) sit between one level's writes and
    # the next's. `d_ids`/`h_ids` stay the BUILD pair (and the end-of-tree
    # all-leaves list, which follows a drain).
    # =======================================================================
    var d_zero_ids: DeviceBuffer[DType.uint32]
    var h_zero_ids: HostBuffer[DType.uint32]
    var d_all_ids: DeviceBuffer[DType.uint32]
    var h_all_ids: HostBuffer[DType.uint32]
    var h_part_stats: HostBuffer[DType.float32]
    # DEVIATION 1901: the winning HISTOGRAM CELL of every splitting leaf,
    # staged with the split payload so the partition-stats propagation
    # kernel can read the scanned prefix it names. Its own pair, per
    # DEVIATION 261's rule -- no host buffer is rewritten under a queued
    # copy. Allocated in both numeric modes (max_leaves words), written and
    # copied only on the FAST arm.
    var d_win_cells: DeviceBuffer[DType.uint32]
    var h_win_cells: HostBuffer[DType.uint32]
    # DEVIATION 1903: the DEFERRED parent-histogram copy's pair lists --
    # source (the left child holding the parent's histogram) and destination
    # (the bigger right sibling that will be derived in place). Their own
    # staging pairs, per DEVIATION 261's rule; `h_left`/`h_right` are the
    # SPLIT stage's pairs and the deferred copy stages at PLAN time.
    # Allocated in both numeric modes, written and copied only on the FAST
    # arm.
    var d_copy_src: DeviceBuffer[DType.uint32]
    var h_copy_src: HostBuffer[DType.uint32]
    var d_copy_dst: DeviceBuffer[DType.uint32]
    var h_copy_dst: HostBuffer[DType.uint32]
    # trees-apple2 (ID_UPLOAD_COALESCE): every small id list above lives in
    # ONE device arena (`d_ids_arena`, slot `IDS_SLOT_*` of `max_leaves`
    # words each), and a phase's lists reach it through ONE staging copy
    # (`h_ids_arena`) per run of adjacent slots. The subtract pair has its
    # own slots (`d_sub_left`/`d_sub_right`), apart from the split pair, so
    # the plan-time and split-time stagings never share a host word.
    var d_ids_arena: DeviceBuffer[DType.uint32]
    var h_ids_arena: HostBuffer[DType.uint32]
    var d_sub_left: DeviceBuffer[DType.uint32]
    var d_sub_right: DeviceBuffer[DType.uint32]
    # the split's bin per splitting leaf (their `splitBins`), this
    # driver's own arena slot beside the split pair; the symmetric pool's
    # `sp_bins` is not used here
    var d_sp_bins: DeviceBuffer[DType.uint32]
    var h_sp_bins: HostBuffer[DType.uint32]
    # the split features, likewise (`IDS_FEAT_SLOTS` slots, read by the
    # kernels through a `CFeature` bitcast)
    var d_sp_feats: DeviceBuffer[DType.uint32]
    var h_sp_feats: HostBuffer[DType.uint32]
    # DEVIATION 1904 (wired): the device-resident winner fold's planes.
    # The four `TBinFeatureTable` columns (feature, clamp-raw bin, one-hot
    # flag, fold count -- each `hist_cells` long, their own staging pairs
    # per DEVIATION 261's rule) and the per-leaf winner records
    # (`WINNER_RECORD_WORDS` words each). `hist_cells` joins the pool KEY
    # for them: the table's content is layout-derived and the old key
    # (max_leaves, stat_count, argmax_blocks) cannot see a layout change
    # at equal argmax_blocks -- 256-cell granularity means two different
    # layouts can share every old key field. Allocated in both numeric
    # modes, written and copied only on the FAST arm (DEVIATION 1901's
    # precedent).
    var hist_cells_key: Int
    # ============================ DEVIATION 1911/1912 ============================
    # THE QUANTIZED FAMILY'S TWO PLANES, pool-owned like every other large
    # buffer (`kernel/hist_quantized_shared.mojo` carries the design):
    #
    #   * `d_qstats` -- the packed fixed-point gradient pairs, one UInt64
    #     per ROW (DEV 1911), rewritten per level by `quantize_pair_kernel`
    #     over exactly the partitions being built. Sized by `n_rows`, which
    #     therefore JOINS THE POOL KEY (`n_rows_key`): the old key could not
    #     see a row-count change at equal (leaves, stats, blocks, cells).
    #   * `d_qacc` -- the per-leaf Int32 accumulator the shared-histogram
    #     blocks flush into, dense-leaf-major
    #     (`max_leaves * stat_count * hist_cells`). Memset ONCE here at
    #     allocation; the bridge (`qh_write_hist_kernel`) zeroes every cell
    #     it reads, so the accumulator is self-cleaning thereafter -- the
    #     same once-per-arena discipline as DEVIATION 1903's memset proof.
    #
    # Allocated at ONE cell when the family is dead (`qh_live` False:
    # IDENTICAL builds, unclaimed columns, or a shape the family refuses),
    # the `acc_i32_is_live` precedent exactly. `qh_key` keeps a pool built
    # for one answer from being reused under the other (the shape answer is
    # layout-derived and the other key fields cannot see a policy-mix
    # change at equal hist_cells).
    # =======================================================================
    var n_rows_key: Int
    var qh_key: Bool
    var d_qstats: DeviceBuffer[DType.uint64]
    var d_qacc: DeviceBuffer[DType.int32]
    # QH_MODE_SKIP: one skip bin per feature (256 = none), set from each
    # tree's root histogram; one cell when the arm is compiled out
    var d_qskip: DeviceBuffer[DType.uint32]
    var d_bf_feature: DeviceBuffer[DType.int32]
    var h_bf_feature: HostBuffer[DType.int32]
    var d_bf_bin: DeviceBuffer[DType.int32]
    var h_bf_bin: HostBuffer[DType.int32]
    var d_bf_one_hot: DeviceBuffer[DType.uint8]
    var h_bf_one_hot: HostBuffer[DType.uint8]
    var d_bf_folds: DeviceBuffer[DType.int32]
    var h_bf_folds: HostBuffer[DType.int32]
    var d_winner: DeviceBuffer[DType.uint32]
    var h_winner: HostBuffer[DType.uint32]
    # DW_NO_LEVEL_SYNC: every feature's `CFeature` words (offset in
    # elements), uploaded once per tree, and the device split count. Sized
    # by `n_features` (a key); one record when the arm is compiled out.
    var n_features_key: Int
    var d_feat_table: DeviceBuffer[DType.uint32]
    var h_feat_table: HostBuffer[DType.uint32]
    var d_nsplit: DeviceBuffer[DType.uint32]
    var h_nsplit: HostBuffer[DType.uint32]

    def __init__(
        out self,
        ctx: DeviceContext,
        max_leaves: Int,
        stat_count: Int,
        argmax_blocks: Int,
        hist_cells: Int,
        n_rows: Int,
        qh_live: Bool,
        n_features: Int,
    ) raises:
        self.max_leaves_key = max_leaves
        self.n_features_key = n_features
        var feat_records = 1
        comptime if DW_NO_LEVEL_SYNC:
            if n_features > 1:
                feat_records = n_features
        self.d_feat_table = ctx.enqueue_create_buffer[DType.uint32](
            IDS_FEAT_SLOTS * feat_records
        )
        self.h_feat_table = ctx.enqueue_create_host_buffer[DType.uint32](
            IDS_FEAT_SLOTS * feat_records
        )
        self.d_nsplit = ctx.enqueue_create_buffer[DType.uint32](1)
        self.h_nsplit = ctx.enqueue_create_host_buffer[DType.uint32](1)
        self.final_ready = False
        self.final_offsets = List[Int]()
        self.final_sizes = List[Int]()
        self.stat_count_key = stat_count
        self.argmax_blocks_key = argmax_blocks
        self.hist_cells_key = hist_cells
        self.n_rows_key = n_rows
        self.qh_key = qh_live
        # DEVIATION 1911/1912: real planes only when the quantized family
        # can run this fit; one-cell placeholders otherwise (the
        # `acc_i32_is_live` shape). The accumulator's memset here is its
        # ONLY blanket zero -- the bridge self-cleans from then on.
        if qh_live:
            self.d_qstats = ctx.enqueue_create_buffer[DType.uint64](n_rows)
            self.d_qacc = ctx.enqueue_create_buffer[DType.int32](
                max_leaves * stat_count * hist_cells
            )
            enqueue_fill(ctx, self.d_qacc, Int32(0))
        else:
            self.d_qstats = ctx.enqueue_create_buffer[DType.uint64](1)
            self.d_qacc = ctx.enqueue_create_buffer[DType.int32](1)
        var qskip_n = 1
        comptime if QH_MODE_SKIP:
            if qh_live and n_features > 0:
                qskip_n = n_features
        self.d_qskip = ctx.enqueue_create_buffer[DType.uint32](qskip_n)
        enqueue_fill(ctx, self.d_qskip, UInt32(256))
        var records = argmax_blocks * max_leaves
        self.region_score = ctx.enqueue_create_buffer[DType.float32](records)
        self.region_bin = ctx.enqueue_create_buffer[DType.uint32](records)
        self.h_region_score = ctx.enqueue_create_host_buffer[DType.float32](
            records
        )
        self.h_region_bin = ctx.enqueue_create_host_buffer[DType.uint32](
            records
        )
        comptime assert (
            CFEATURE_BYTES % 4 == 0
        ), "the id arena holds CFeature records as whole 32-bit words"
        var arena = ctx.enqueue_create_buffer[DType.uint32](
            IDS_SLOTS * max_leaves
        )
        self.h_ids_arena = ctx.enqueue_create_host_buffer[DType.uint32](
            IDS_SLOTS * max_leaves
        )
        self.d_sub_left = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_SUB_LEFT * max_leaves, max_leaves
        )
        self.d_sub_right = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_SUB_RIGHT * max_leaves, max_leaves
        )
        self.d_visit = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_VISIT * max_leaves, max_leaves
        )
        self.d_sp_bins = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_SP_BINS * max_leaves, max_leaves
        )
        self.h_sp_bins = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.d_sp_feats = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_SP_FEATS * max_leaves, IDS_FEAT_SLOTS * max_leaves
        )
        self.h_sp_feats = ctx.enqueue_create_host_buffer[DType.uint32](
            IDS_FEAT_SLOTS * max_leaves
        )
        self.h_visit = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.d_left = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_LEFT * max_leaves, max_leaves
        )
        self.h_left = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
        self.d_right = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_RIGHT * max_leaves, max_leaves
        )
        self.h_right = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.d_ids = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_IDS * max_leaves, max_leaves
        )
        self.h_ids = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
        self.d_zero_ids = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_ZERO * max_leaves, max_leaves
        )
        self.h_zero_ids = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.d_all_ids = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_ALL * max_leaves, max_leaves
        )
        self.h_all_ids = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.h_part_stats = ctx.enqueue_create_host_buffer[DType.float32](
            max_leaves * stat_count
        )
        self.d_win_cells = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_WIN * max_leaves, max_leaves
        )
        self.h_win_cells = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.d_copy_src = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_COPY_SRC * max_leaves, max_leaves
        )
        self.h_copy_src = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.d_copy_dst = arena.create_sub_buffer[DType.uint32](
            IDS_SLOT_COPY_DST * max_leaves, max_leaves
        )
        self.h_copy_dst = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        # DEVIATION 1904 (wired): see the field block above
        self.d_bf_feature = ctx.enqueue_create_buffer[DType.int32](hist_cells)
        self.h_bf_feature = ctx.enqueue_create_host_buffer[DType.int32](
            hist_cells
        )
        self.d_bf_bin = ctx.enqueue_create_buffer[DType.int32](hist_cells)
        self.h_bf_bin = ctx.enqueue_create_host_buffer[DType.int32](
            hist_cells
        )
        self.d_bf_one_hot = ctx.enqueue_create_buffer[DType.uint8](hist_cells)
        self.h_bf_one_hot = ctx.enqueue_create_host_buffer[DType.uint8](
            hist_cells
        )
        self.d_bf_folds = ctx.enqueue_create_buffer[DType.int32](hist_cells)
        self.h_bf_folds = ctx.enqueue_create_host_buffer[DType.int32](
            hist_cells
        )
        self.d_winner = ctx.enqueue_create_buffer[DType.uint32](
            WINNER_RECORD_WORDS * max_leaves
        )
        self.h_winner = ctx.enqueue_create_host_buffer[DType.uint32](
            WINNER_RECORD_WORDS * max_leaves
        )
        self.d_ids_arena = arena^


def is_terminal_leaf(
    leaf: TLeaf, options: TTreeStructureSearcherOptions
) raises -> Bool:
    """`TGreedySearchHelper::IsTerminalLeaf` (`greedy_search_helper.cpp:691`).

        const bool checkLeafSize = Options.Policy != EGrowPolicy::SymmetricTree;
        const bool flag = (checkLeafSize && leaf.Size <= Options.MinLeafSize)
                          || leaf.Path.GetDepth() >= Options.MaxDepth;

    **The size test is `<=`, not `<`.** `min_data_in_leaf = 1` therefore
    means a one-row leaf is TERMINAL, not that a one-row leaf is permitted.
    Getting that boundary backwards grows a tree one level deeper than
    CatBoost's on every branch that reaches a single row, and no leaf count
    or row conservation can see it.

    `checkLeafSize` is written out rather than folded away even though this
    file only ever runs with Depthwise, because the constant it folds to is
    a POLICY fact and the next reader should not have to know the policy to
    read the line.
    """
    var check_leaf_size = options.policy != GROW_SYMMETRIC
    if check_leaf_size and Float64(leaf.size) <= options.min_leaf_size:
        return True
    return leaf.get_depth() >= options.max_depth


def should_terminate(
    leaves: List[TLeaf], options: TTreeStructureSearcherOptions
) raises -> Bool:
    """`ShouldTerminate` (`greedy_search_helper.cpp:678-689`).

        if (leafCount >= Options.MaxLeaves) return true;
        ... return AreAllTerminal(subsets, allLeaves);

    `AreAllTerminal` is folded into the loop below; theirs builds an
    `Iota` vector and calls the helper, which is the same test.
    """
    if len(leaves) >= options.max_leaves:
        return True
    for i in range(len(leaves)):  # small-loop(leaves: tree leaves, at most max_leaves): terminal flags tree-shape decision
        if not leaves[i].is_terminal:
            return False
    return True


def select_leaves_to_visit(leaves: List[TLeaf]) raises -> List[Int]:
    """`SelectLeavesToVisit` (`greedy_search_helper.cpp:698-711`).

        if (!leaf.IsTerminal) {
            if (leaf.BestSplit.Defined()) continue;
            leavesToVisit->push_back(leaf);
        }

    Note what the `Defined()` skip buys: a leaf whose histogram was NOT
    rebuilt this level still holds last level's best split, and is not
    re-scored. `BuildNecessaryHistograms` is what clears it, for exactly the
    leaves whose histograms it updated (`split_properties_helper.cpp:1367`).
    That coupling is the whole incremental mechanism and it is why the reset
    below lives in this file's histogram step and not next to the scoring.
    """
    var out = List[Int]()
    for leaf in range(len(leaves)):  # small-loop(leaves: tree leaves, at most max_leaves): per-level leaf id plan list
        if not leaves[leaf].is_terminal:
            if leaves[leaf].best_split.defined:
                continue
            out.append(leaf)
    return out^


#: FAST on Apple: Lossguide splits the B best leaves per iteration (the same
#: strict-< argmin, repeated, over leaves not yet taken) instead of one, so
#: a tree pays about max_leaves / B host round trips instead of max_leaves;
#: more than two new leaves are scored through the Depthwise leafwise
#: kernel. Splitting one leaf never changes another leaf's best split, so
#: the batch only differs from best-first where the leaf budget runs out.
#: M4, 1M rows, 100 trees: taxi 11.6 -> 4.25 s, Istella-S 13.8 -> 9.7 s,
#: logloss/AUC equal or better. Default 16; arms `-D
#: MOJOLEARN_GBDT_LG_BATCH2|4|8|32`; `-D MOJOLEARN_GBDT_LG_BATCH_OFF` keeps
#: one leaf per iteration.
#: OFF BY DEFAULT since 2026-09-26 (lane/apple-identical-neural): a quality
#: check against best-first (one leaf per iteration), five 300k training
#: subsets each, 300 trees, max_leaves 31, paired by subset, found the batch
#: of 16 systematically worse: Istella regression test RMSE 0.5950-0.5970 ->
#: 0.6072-0.6089 (about +2% on every subset), taxi AUC 0.6127 -> 0.6123 and
#: log-loss 0.53820 -> 0.53831 (mean). When the leaf budget binds, the batch
#: splits a different set of leaves than best-first. FAST may differ in bits
#: but not in quality, so best-first stays the FAST default;
#: `-D MOJOLEARN_GBDT_LG_BATCH=1` turns the batch on as a trial arm (fit
#: time taxi 23 -> 10.5 s, Istella ~50 -> 17 s on the M4).
comptime _LG_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_GBDT_LG_BATCH"]()
    and not is_defined["MOJOLEARN_GBDT_LG_BATCH_OFF"]()
)
comptime GBDT_LG_BATCH = 1 if not _LG_FAST_APPLE else (
    32 if is_defined["MOJOLEARN_GBDT_LG_BATCH32"]() else (
        8 if is_defined["MOJOLEARN_GBDT_LG_BATCH8"]() else (
            4 if is_defined["MOJOLEARN_GBDT_LG_BATCH4"]() else (
                2 if is_defined["MOJOLEARN_GBDT_LG_BATCH2"]() else 16
            )
        )
    )
)


#: FAST on Apple, trees-apple3: EXACT best-first in batches. The batch above
#: (GBDT_LG_BATCH) splits the B best leaves and keeps them all, which grows a
#: different tree than best-first when the leaf budget binds (its quality
#: note). This one grows the SAME tree as one leaf per iteration:
#:
#: * every round replays best-first on the host over the gains known so far
#:   (`_lg_exact_plan`: the strict-< argmin over the leaves in creation
#:   order, the min_split_gain test and the max_leaves stop, verbatim). The
#:   replay is exact up to the first leaf it wants to split that the device
#:   has not split yet;
#: * that leaf is split for certain. The replay then continues with its
#:   children unknown, and the further leaves it would split (what
#:   best-first does next when no new child beats them) are split in the
#:   same round, ahead of time;
#: * when the replay ends inside the known tree, its leaves ARE the
#:   best-first tree. A leaf split ahead of time that best-first never
#:   reached is folded back on the host: its descendants' rows are one
#:   contiguous range of the parent, so the leaf's statistics are the sum of
#:   theirs and its path is the one stored when it was split.
#:
#: A tree pays about log2(max_leaves) + max_leaves / width rounds of host
#: waits and launches instead of max_leaves - 1. Leaves split ahead of time
#: need slots, so the leaf capacity is min(2 * max_leaves, 1 << max_depth)
#: and a round never takes a slot the certain splits still to come may need.
#: THE FAST APPLE DEFAULT since lane apple-fast-trees2 (2026-10-02), with
#: NS_INHERIT_PARTITION below. M3 Ultra, board shapes (500 trees, depth 8,
#: 256 leaves), alternating A/B, FAST main vs FAST with both: Lossguide taxi
#: (4.1M rows) 78.2 -> 16.7 s, Istella-S (2.0M rows) 102.0 -> 23.1 s, held-out
#: logloss and AUC within the run-to-run spread of FAST (whose float-atomic
#: histograms vary run to run in both arms). Width 64 since aft-ab-lgw64
#: (M3, vs 32: taxi 16.8 -> 14.9 s, Istella 22.9 -> 20.7 s, quality within
#: spread; width 16 was slower, 18.1 s taxi); arms `-D
#: MOJOLEARN_GBDT_LG_EXACT_BATCH32|128`; `-D MOJOLEARN_GBDT_LG_EXACT_BATCH_OFF`
#: keeps one leaf per iteration (the A/B arm).
#:
#: IDENTICAL, every GPU (lane gap-trees-nv): the same rounds under `-D
#: MOJOLEARN_GBDT_LG_EXACT_ID`. Same bits as one leaf per iteration because
#: (1) a leaf's score reads only its own histogram and partition stats: the
#: histogram is the order-free fixed-point sum or the subtraction its sibling
#: pair fixes, the stats are the pinned per-leaf fold, both independent of
#: which other leaves share the launch, and the two-leaf and region kernels
#: run the same `_leafwise_scan_part`; (2) the replay picks the leaves
#: best-first picks, in its leaf order, so the model is the same tree; (3) a
#: leaf split ahead of time and folded back takes the partition stats it had
#: when it was scored (`lg_node_stats`, read at that iteration's score wait),
#: which are the reduction best-first's end-of-tree sweep forms over the same
#: rows in the same order, instead of the sum of its children's. Only the
#: per-level score noise (`random_strength` with a Cosine score) draws per
#: round, so a fit with noise grows one leaf per iteration.
#:
#: THE IDENTICAL DEFAULT since lane ml-gbdt (2026-10-04, roadmap T1): on for
#: every vendor and the host column together, so the four columns still
#: agree. Bits: none move against one leaf per iteration (the argument
#: above). `-D MOJOLEARN_GBDT_LG_EXACT_ID_OFF` (or the master `-D
#: MOJOLEARN_IDN_ALL_OFF`) is the A/B arm and restores one leaf per
#: iteration; the old opt-in `-D MOJOLEARN_GBDT_LG_EXACT_ID` stays harmless.
#: Memory gate (`LG_EXACT_ID_MAX_CELLS`, in the fit): the capacity doubles
#: to min(2 * max_leaves, 1 << max_depth) leaf slots; a fit whose doubled
#: slots would not fit the pool's leaf histograms, or whose histogram cells
#: would pass the Int32 offset range, keeps one leaf per iteration.
comptime LG_EXACT_ID = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    # I17 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    and not is_defined["MOJOLEARN_GBDT_LG_EXACT_ID_OFF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: LG_EXACT_ID memory gate: the most histogram cells (leaf slots x stat
#: planes x cells per leaf) a batched IDENTICAL Lossguide fit may address.
#: Int32 offsets index the histogram and fixed-point accumulator planes.
comptime LG_EXACT_ID_MAX_CELLS = 2147483647
comptime LG_EXACT_BATCH = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
# F12/lossguide M3 2026-10-06: 4 scored caller times; B/A
# 0.4561..1.0883 (mixed/regressing); existing default retained.
# Scored FAST quality 2/2 within existing bands; PASS.
# One warmup/one score; caller67d0efb29; exact cases/builds/hashes:
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/lossguide.
# Compilation/identity reused. No combined-toggle/full-board claim.
    and not is_defined["MOJOLEARN_GBDT_LG_EXACT_BATCH_OFF"]()
    and not _LG_FAST_APPLE
) or LG_EXACT_ID
#: FAST on Apple: width 128, THE DEFAULT since lane apple-fast-lgw128
#: (2026-10-03): M3 Ultra A/B aft-ab-lgw128 on main 5f5eadcde, lossguide taxi,
#: n=2: 13,148 -> 12,921 ms (-1.7%), AUC .6320 unchanged.
#: `-D MOJOLEARN_GBDT_LG_EXACT_BATCH128_OFF` is the A/B arm (back to 64); the
#: old `-D MOJOLEARN_GBDT_LG_EXACT_BATCH128` stays harmless. FAST elsewhere:
#: width 64 (arms 32|128). IDENTICAL (`LG_EXACT_ID`) keeps main's width 32
#: (arm 64). Lossguide only; depthwise never reads the width.
comptime _LG_EXACT_BATCH128 = (
    is_defined["MOJOLEARN_GBDT_LG_EXACT_BATCH128"]()
    or (
        has_apple_gpu_accelerator()
        and not is_defined["MOJOLEARN_GBDT_LG_EXACT_BATCH32"]()
    )
) and not is_defined["MOJOLEARN_GBDT_LG_EXACT_BATCH128_OFF"]()
comptime LG_EXACT_BATCH_WIDTH = (
    128 if _LG_EXACT_BATCH128 else (
        32 if is_defined["MOJOLEARN_GBDT_LG_EXACT_BATCH32"]() else 64
    )
) if GLOBAL_NUMERIC_MODE == NUMERIC_FAST else (
    64 if is_defined["MOJOLEARN_GBDT_LG_EXACT_BATCH64"]() else 32
)

#: FAST on Apple (trees-apple3): THE ESTIMATOR INHERITS THE SEARCHER'S
#: PARTITION. When the tree is grown the row index already holds every row
#: grouped by leaf: a split leaves "goes left" rows first in the parent's
#: range and the left child keeps that range's start, so the leaves lie in
#: the index in the model's own leaf order (left subtree first), and the
#: index restarts as 0..n-1 every tree and is only ever stably partitioned,
#: so rows ascend inside each leaf. That is the partition
#: `compute_non_symmetric_bins_for_model` + the device partitioner rebuild
#: from the model (one tree walk per row, a radix sort and a host wait per
#: tree), and it is what CatBoost's estimator inherits for the permutation
#: the tree was grown on. The fit records the leaves' ranges here and
#: `doc_parallel_boosting` hands them to the estimator when the fit has one
#: permutation. THE FAST APPLE DEFAULT since lane apple-fast-trees2
#: (2026-10-02): M3 Ultra, Depthwise taxi at the board shape 17.3 -> 14.9 s
#: in an alternating A/B, held-out logloss and AUC within FAST's run-to-run
#: spread. `-D MOJOLEARN_GBDT_NS_INHERIT_PARTITION_OFF` is the A/B arm.
#:
#: IDENTICAL, every GPU (lane gap-trees-nv): `-D MOJOLEARN_GBDT_NS_INHERIT_ID`.
#: The same rows in the same order per leaf as the rebuild (a stable radix
#: sort of 0..n-1 by leaf keeps rows ascending inside each leaf, as the
#: searcher's stable partitions of 0..n-1 do), at the same offsets, so the
#: estimator reduces the same values in the same order.
#:
#: THE IDENTICAL DEFAULT since lane ml-gbdt (2026-10-04, roadmap T2), every
#: vendor and the host column together. GUARD (review of T1+T2): a leaf
#: LG_EXACT_ID split ahead of time and folded back holds its descendants'
#: slots concatenated, so its rows are NOT ascending and the estimator's
#: fold order would differ from the rebuild's. Under IDENTICAL a tree with
#: any folded-back result leaf leaves the record unset and the caller
#: rebuilds the partition from the model (`NS_INHERIT_ID_GUARD` below), so
#: bits move nowhere: T1+T2 together inherit only trees where every result
#: leaf is one searcher slot, whose rows ascend. `-D
#: MOJOLEARN_GBDT_NS_INHERIT_ID_OFF` (or the master `-D
#: MOJOLEARN_IDN_ALL_OFF`) is the A/B arm; the old opt-in `-D
#: MOJOLEARN_GBDT_NS_INHERIT_ID` stays harmless.
comptime NS_INHERIT_ID = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    # I17 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    and not is_defined["MOJOLEARN_GBDT_NS_INHERIT_ID_OFF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NS_INHERIT_PARTITION = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_GBDT_NS_INHERIT_PARTITION_OFF"]()
) or NS_INHERIT_ID
#: the T1+T2 guard: IDENTICAL never inherits a tree with a folded-back leaf
comptime NS_INHERIT_ID_GUARD = NS_INHERIT_ID and LG_EXACT_ID


def _path_before(a: TLeafPath, b: TLeafPath) -> Bool:
    """Whether leaf `a` precedes leaf `b` in the model's leaf order: the
    left child (`SPLIT_VALUE_ZERO`) of their last common node first."""
    var n = len(a.directions)
    if len(b.directions) < n:
        n = len(b.directions)
    for i in range(n):  # small-loop(n: path depth, at most max_depth): leaf order from path directions
        if a.directions[i] != b.directions[i]:
            return a.directions[i] < b.directions[i]
    return len(a.directions) < len(b.directions)


comptime LG_NODE_UNKNOWN = 0
comptime LG_NODE_DEFINED = 1
comptime LG_NODE_NO_SPLIT = 2


struct TLossguideReplay(Movable):
    """What one best-first replay found (`_lg_exact_plan`)."""

    var complete: Bool
    """The replay ended inside the known tree: `final_nodes` is the
    best-first tree's leaves, in best-first leaf-id order."""
    var blocked: Bool
    """The replay reached a leaf that has no score yet."""
    var final_nodes: List[Int]
    var expand: List[Int]
    """Nodes to split this round; the first is certain."""
    var exact_picks: Int

    def __init__(out self):
        self.complete = False
        self.blocked = False
        self.final_nodes = List[Int]()
        self.expand = List[Int]()
        self.exact_picks = 0


def _lg_exact_plan(
    node_left: List[Int],
    node_right: List[Int],
    node_gain: List[Float32],
    node_state: List[Int],
    max_leaves: Int,
    min_split_gain: Float64,
    expand_limit: Int,
) raises -> TLossguideReplay:
    """Best-first (`find_best_leaf_to_split` + the min_split_gain test +
    `should_terminate`'s leaf count) replayed over the tree the device has
    grown so far. `sim[i]` is the node that best-first's leaf `i` holds: the
    left child keeps its parent's leaf id and the right child takes the next
    one, as `MakeSplit` numbers them, so the argmin's tie rule (the FIRST
    leaf wins) sees the leaves in the order best-first would.

    Exact phase: every leaf's gain is known. It ends at the first leaf the
    replay picks that the device has not split (`expand[0]`), or when
    best-first itself would stop (`complete`), or at a leaf without a score
    (`blocked`; only the check run straight after a split can see one).
    After it, the replay goes on with unknown children never picked and
    collects up to `expand_limit` leaves in all."""
    var out = TLossguideReplay()
    var sim = List[Int]()
    sim.append(0)
    var exact = True
    while True:
        if len(sim) >= max_leaves:
            if exact:
                out.complete = True
            break
        var best = -1
        var best_gain = Float32.MAX
        var unknown = False
        for i in range(len(sim)):  # small-loop(sim: replayed leaves, at most max_leaves): best-first replay over tree shape
            var n = sim[i]
            if n < 0:
                continue
            if node_state[n] == LG_NODE_UNKNOWN:
                unknown = True
                continue
            if node_state[n] != LG_NODE_DEFINED:
                continue
            if node_gain[n] < best_gain:
                best_gain = node_gain[n]
                best = i
        if exact and unknown:
            out.blocked = True
            break
        if best < 0:
            if exact:
                out.complete = True
            break
        if min_split_gain >= Float64(0):
            if not (Float64(-best_gain) > min_split_gain):
                if exact:
                    out.complete = True
                break
        var picked = sim[best]
        if node_left[picked] >= 0:
            sim[best] = node_left[picked]
            sim.append(node_right[picked])
            if exact:
                out.exact_picks += 1
        else:
            exact = False
            if len(out.expand) >= expand_limit:
                break
            out.expand.append(picked)
            sim[best] = -1
            sim.append(-1)
    if out.complete:
        for i in range(len(sim)):  # small-loop(sim: replayed leaves, at most max_leaves): final leaf id list copy
            out.final_nodes.append(sim[i])
    return out^


def _lossguide_top_b(leaves: List[TLeaf], b: Int) raises -> List[Int]:
    """GBDT_LG_BATCH: up to `b` leaves, each the strict-< argmin of the
    stored gain over the defined leaves not yet taken, returned in ascending
    id order (the multi-leaf MakeSplit numbers right children by position)."""
    var chosen = List[Int]()
    for _ in range(b):  # small-loop(b: leaves per batch, a handful): batch leaf choice for the plan
        var best = -1
        var best_gain = Float32.MAX
        for i in range(len(leaves)):  # small-loop(leaves: tree leaves, at most max_leaves): argmin of stored winner gains
            if not leaves[i].best_split.defined:
                continue
            var taken = False
            for ci in range(len(chosen)):  # small-loop(chosen: leaves already taken, at most b): batch membership test
                if chosen[ci] == i:
                    taken = True
            if taken:
                continue
            if leaves[i].best_split.gain < best_gain:
                best_gain = leaves[i].best_split.gain
                best = i
        if best < 0:
            break
        chosen.append(best)
    sort(chosen)
    return chosen^


def select_leaves_to_split(leaves: List[TLeaf]) raises -> List[Int]:
    """`SelectLeavesToSplit`'s Depthwise arm (`greedy_search_helper.cpp:359`).

        CB_ENSURE(Options.Policy == SymmetricTree || Options.Policy == Depthwise);
        for (leaf ...) if (Leaves[leaf].BestSplit.Defined()
                           && Leaves[leaf].BestSplit.Score < 0)
            leavesToSplit->push_back(leaf);

    Depthwise and SymmetricTree share this branch exactly; Lossguide takes
    the single best leaf and Region takes the shallower of the last pair,
    and both of those belong to other lanes.

    **THE TEST IS ON `Score`, NOT ON `Gain`, and for this policy they are
    THE SAME NUMBER.** `ComputeOptimalSplitsRegion` writes the gain into
    both fields -- `if (gain < bestScore) { bestScore = gain; bestIndex =
    binFeatureId; bestGain = gain; }` (`compute_scores.cu:380-384`) --
    where the oblivious kernel writes the raw score into `Score` and the
    gain into `Gain` (`:132-140`). So implementing the test as written is right
    here and would be WRONG if this branch were ever fed the oblivious
    kernel's records.

    ============ WHICH SIGN THIS FIELD IS IN, and it is THEIRS ============
    The KERNEL is sign-flipped (larger gain is better; see
    `kernel/compute_scores.mojo`). `TBestSplitProperties` IS NOT. It is a
    statement-for-statement match of their struct, its defaults are their defaults
    (`Gain = FLT_MAX` losing every comparison), and the only comparator over
    it -- `best_split_properties_less`, their `operator<` -- is keyed on
    their orientation. So `compute_optimal_splits` NEGATES the kernel's gain
    once, at the point where it builds the record, and everything downstream
    of that record is their code unaltered, this test included.

    Writing the kernel's sign into the struct and flipping the test instead
    LOOKS equivalent and is not: `best_split_properties_less` would then be
    reading a field in the opposite orientation from the one it was
    written against, and would silently select the WORST candidate on
    every cross-block reduce. This function had `gain > 0` on its first
    run and grew a one-leaf tree, which is what that mistake looks like
    from the outside: no candidate ever passes, every leaf is marked
    terminal, and the fit returns a constant.
    ======================================================================
    """
    var out = List[Int]()
    for leaf in range(len(leaves)):  # small-loop(leaves: tree leaves, at most max_leaves): leaf id plan list from stored winners
        if leaves[leaf].best_split.defined:
            # `BestSplit.Score < 0`, verbatim -- and on this kernel `Score`
            # and `Gain` are the same number (see above).
            if leaves[leaf].best_split.gain < Float32(0.0):
                out.append(leaf)
    return out^


def split_leaf(
    leaf: TLeaf, split: TBinarySplit, direction: Int
) raises -> TLeaf:
    """`SplitLeaf` (`split_properties_helper.cpp:786-798`), all five fields.

        newLeaf.Size = 0;
        newLeaf.Path = leaf.Path; newLeaf.Path.AddSplit(split, direction);
        if (leaf.HistogramsType == CurrentPath)
            newLeaf.HistogramsType = PreviousPath;
        newLeaf.BestSplit.Reset();

    **The `HistogramsType` transition is conditional and the default is
    `Zeroes`.** A child of a leaf whose histogram was current inherits
    `PreviousPath`, which is what makes it eligible for sibling subtraction
    next level -- its slot holds the PARENT's histogram, put there by the
    left child keeping the parent's slot and by `copy_histograms_kernel`
    filling the right child's. A child of a leaf whose histogram was NOT
    current falls through to `Zeroes` and is rebuilt. Collapsing the
    condition to an unconditional `PreviousPath` would subtract a histogram
    that was never written.

    `Size = 0` is theirs and is a placeholder: `RebuildLeavesSizes` fills it
    from the device right after the split kernel.
    """
    var child = TLeaf()
    child.size = 0
    child.path = split_leaf_path(leaf.path, split, direction)
    if leaf.histograms_type == EHistogramsType.CurrentPath:
        child.histograms_type = EHistogramsType.PreviousPath
    # else: stays at the constructor's Zeroes, theirs by omission
    return child^


def _as_i32(values: List[Int]) -> List[Int32]:
    """A host id list in the shape `IdentityTrace.record_list_i32` takes.

    Leaf ids and plan ids are `UInt32` on the device and `Int` on the host,
    and the trace has `record_list_i32` but no `record_list_u32`. Narrowing
    to `Int32` rather than adding a method to another lane's struct: the ids
    are small non-negative counts, the trace hashes BITS either way, and the
    dtype the record carries then honestly says what was hashed. Debug path
    only.
    """
    var out = List[Int32]()
    for i in range(len(values)):  # small-loop(values: leaf id list, at most max_leaves): trace record conversion only
        out.append(Int32(values[i]))
    return out^


def _u32_as_i32(values: List[UInt32]) -> List[Int32]:
    """Same, for the plan's `UInt32` id lists."""
    var out = List[Int32]()
    for i in range(len(values)):  # small-loop(values: plan id list, at most max_leaves): trace record conversion only
        out.append(Int32(Int(values[i])))
    return out^


def _one_i32(value: Int) -> List[Int32]:
    """A scalar as a one-element record -- their `record_list_i32` is the
    integer counterpart of `record_scalar_f32`."""
    var out = List[Int32]()
    out.append(Int32(value))
    return out^


def _leaf_records(
    leaves: List[TLeaf], parent_of: List[Int]
) raises -> List[LeafRecord]:
    """`BuildNecessaryHistograms`' view of the leaves.

    `path_id` is their `PreviousSplit(leaf.Path)` hash-map key
    (`split_properties_helper.cpp:1300`), replaced by the PARENT LEAF ID --
    the substitution `split_properties_helper.mojo` has carried since the
    symmetric lane wrote it. Two siblings share a parent id exactly when
    they share a parent path, so the equivalence classes are identical, and
    an integer key is reproducible where a hash-map walk is not.

    THE PARENT ID IS RECOVERED FROM THE PATH, not carried as a field. A
    leaf's parent is the leaf it was split from, which is the LEFT child's
    own id, and the left child keeps the parent's id (`MakeSplit`,
    `:861-862`). So the caller passes a `parent_of` list built at split
    time; this function does not guess it.
    """
    var out = List[LeafRecord]()
    for i in range(len(leaves)):  # small-loop(leaves: tree leaves, at most max_leaves): histogram plan records per leaf
        var t = HISTOGRAMS_ZEROES
        if leaves[i].histograms_type == EHistogramsType.PreviousPath:
            t = HISTOGRAMS_PREVIOUS_PATH
        elif leaves[i].histograms_type == EHistogramsType.CurrentPath:
            t = HISTOGRAMS_CURRENT_PATH
        out.append(
            LeafRecord(
                UInt32(leaves[i].size),
                t,
                parent_of[i],
                leaves[i].is_terminal,
            )
        )
    return out^


#: Threads per block of `dw_leaf_values_kernel` (one thread per leaf).
comptime DW_LEAF_VALUE_BLOCK = 64


def dw_leaf_values_kernel(
    part_stats: MutPointer[Float32, MutAnyOrigin],
    sums64: MutPointer[UInt64, MutAnyOrigin],
    use_sums64: Int32,
    n_leaves_in: Int32,
    stat_count_in: Int32,
    l2_bits: UInt64,
    eps_bits: UInt64,
    multiclass: Int32,
    out_values: MutPointer[Float32, MutAnyOrigin],
):
    """The non-symmetric searcher's own leaf values (lane cpu3-gbdt-a), the
    host tail's statements in soft-float64 (`checks/soft_f64.mojo`, IEEE
    double in integer arithmetic, the same bits on every vendor), one
    thread per final leaf:

        w = double(stats[leaf][0])        (or the replay's double sums)
        v[a] = w > 1e-20 ? float(double(stats[leaf][1 + a]) / (w + l2)) : 0
        total += double(v[a])             (in `a` order, from +0.0)
        multiclass: v[a] = float(double(v[a]) + total)

    Correctly rounded add, divide and narrowing, so the values are the old
    host loop's bit for bit and no column moves. `sums64` (doubles as bit
    patterns, `[leaf][stat]`) replaces `part_stats` when `use_sums64` is
    set: the lossguide exact replay's per-final-leaf sums."""
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if leaf >= Int(n_leaves_in):
        return
    var sc = Int(stat_count_in)
    var dim = sc - 1
    var w: UInt64
    if use_sums64 != Int32(0):
        w = sums64.unsafe_load(leaf * sc)
    else:
        w = sf64_from_f32(part_stats.unsafe_load(leaf * sc))
    var live = (not sf64_is_nan(w)) and sf64_gt(w, eps_bits)
    var denom = sf64_add(w, l2_bits)
    var total = UInt64(0)
    for a in range(dim):
        var v = Float32(0.0)
        if live:
            var s: UInt64
            if use_sums64 != Int32(0):
                s = sums64.unsafe_load(leaf * sc + 1 + a)
            else:
                s = sf64_from_f32(part_stats.unsafe_load(leaf * sc + 1 + a))
            v = sf64_to_f32(sf64_div(s, denom))
        out_values.unsafe_store(leaf * dim + a, v)
        total = sf64_add(total, sf64_from_f32(v))
    if multiclass != Int32(0):
        for a in range(dim):
            var v = out_values.unsafe_load(leaf * dim + a)
            out_values.unsafe_store(
                leaf * dim + a, sf64_to_f32(sf64_add(sf64_from_f32(v), total))
            )


def _stats_through_index_kernel(
    stats: MutPointer[Float32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    stat_count: Int32,
):
    """`dst[s * n + p] = stats[s * n + row_index[p]]`: the permuted plane a
    ridx-only build does not keep, for the identity trace only."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_rows)
    if i < n * Int(stat_count):
        var s = i // n
        var p = i - s * n
        dst[unsafe_offset=i] = stats[
            unsafe_offset = s * n + Int(row_index[unsafe_offset=p])
        ]


def fit_non_symmetric_tree[
    hist2_smem_mode: Int = 0
](
    ctx: DeviceContext,
    n_rows: Int,
    fold_counts: List[Int],
    options: TTreeStructureSearcherOptions,
    mut cindex: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut row_index: DeviceBuffer[DType.uint32],
    weight_magnitude: Float32,
    gradient_magnitude: Float32,
    mut ws: List[TTreeWorkspace],
    mut dws: List[TDepthwiseWorkspace],
    # ---- THE IDENTITY TRACE, off unless the caller enables one ----
    # `core/identity_trace.mojo`, the LOSSGUIDE lane's facility, used here
    # rather than re-implemented. When two GPU columns disagree, claim 6 of
    # `depthwise_check` says THAT and not WHERE; the trace names the stage,
    # and `tools/identity_trace_diff.py` then classifies each differing cell
    # (DENORMAL-vs-ZERO / SIGN / NAN-payload / ULP<=n / LARGE), which is the
    # diagnosis where a hash is only the location.
    #
    # This lane briefly had its own digest implementation (commit e5cef46)
    # before `core/identity_trace.mojo` became canonical. It is deleted. Two implementations of
    # one instrument is the drift surface every rule in this tree is about,
    # and theirs is a strict superset: generic over `DType` where mine was
    # four hand-written methods, `create_sub_buffer` for a short read where
    # mine needed two lengths, plus raw dumps, tag-uniqueness enforcement, a
    # format version and a gated reader.
    #
    # `IdentityTrace()` reads the environment once and is DISABLED unless
    # `MOJOLEARN_IDENTITY_TRACE` is set; `IdentityTrace.disabled()` is the
    # explicit off switch for callers that must not depend on the operator's
    # shell. Off, this costs one Bool test per stage. On, it DRAINS at every
    # record, so a traced run is not a measurement.
    mut trace: IdentityTrace,
    one_hot: List[Bool] = List[Bool](),
    approx_dim: Int = 1,
    multiclass_optimization: Bool = False,
    random_seed: UInt64 = UInt64(0),
    # ---- A TEST KNOB, and the only reason it is a parameter ----
    # "A configuration that cannot be varied inside one process cannot be
    # measured here, so make it a parameter" -- a lesson earned on this
    # box. THE CORE COUNT IS THE ONLY MACHINE-DEPENDENT INPUT this
    # algorithm has: it sizes every strided grid and it is what
    # `IDENTITY_PATHS.md` row 7 had to PIN when it turned out to be
    # feeding a float sum. Overriding it here lets one process ask the
    # cross-GPU question directly -- grow the same tree as a 10-core M4
    # and as a 108-SM A100 and compare bits -- instead of waiting for the
    # other machine. -1 means "read the device", which is every
    # non-test caller.
    sm_count_override: Int = -1,
    # ---- THE TRACE TAG PREFIX, for a fit that grows MANY trees ----
    # `core/identity_trace.mojo` requires every tag to be unique within a
    # trace, and every tag this function emits is rooted at `d<level>.`,
    # `final.` or `model.`. One tree per file (the checks, the E2 growth
    # cards) needs no prefix and passes none, so their cards are byte for
    # byte what they were; the boosting loop passes `treeNNN.` -- the
    # same `_tree_tag` the symmetric arm hands `run_tree_layout_traced` --
    # so twenty non-symmetric trees in one card stay distinguishable.
    # DEVIATION 259.
    tag_prefix: String = String(""),
    # IDN_NS_SCALE_DEVICE: the two magnitudes on the device (weight, then
    # gradient, sums of absolute values); when present the scale is derived
    # there and `weight_magnitude` / `gradient_magnitude` are not read.
    mags_dev: Optional[DeviceBuffer[DType.float32]] = None,
) raises -> TNonSymmetricTree:
    """`TGreedyTreeLikeStructureSearcher<TNonSymmetricTree>::FitImpl`.

    The reference tree, `structure_searcher_template.h:41-67`:

        TPointsSubsets subsets = searchHelper.CreateInitialSubsets(objective);
        while (true) {
            searchHelper.ComputeOptimalSplits(&subsets);
            if (!searchHelper.SplitLeaves(&subsets, &leaves, &weights, &values))
                break;
        }
        return BuildTreeLikeModel<TModel>(leaves, weights, values);

    ONE ITERATION IS ONE LEVEL under Depthwise, exactly as under
    SymmetricTree -- every non-terminal improving leaf splits at once. What
    differs is that "every leaf" is a SUBSET and each member takes its own
    split, so the iteration count is still bounded by `max_depth` but the
    leaf count is not `1 << depth`.

    `weight_magnitude` / `gradient_magnitude` are the sums of ABSOLUTE
    values, one per stat plane, and they are the safety argument for the
    fixed-point flush the `IDENTICAL` numeric mode puts in place of
    CatBoost's float atomic. Same contract, same words, as
    `run_tree_layout`: every partial sum the device forms is over a SUBSET
    of the rows, so bounding the full-dataset sum of magnitudes bounds every
    Int32 slot at every depth. See `checks/fixed_point.mojo`.

    ================= DEVIATION 352 =================
    THE PARTITION STATS ARE RECOMPUTED, NOT UPDATED IN THE SPLIT.
    Their `TSplitPointsKernel` updates `subsets->PartitionStats` inside the
    split ("Update part stats", `split_properties_helper.cpp:918`), so their
    `ComputeOptimalSplits` finds them already correct. FAST/DETERMINISTIC
    propagate child statistics from the winning histogram. IDENTICAL now
    recomputes changed children with the established pinned fold and caches
    unchanged leaves. MOJOLEARN_GBDT_FULL_PARTITION_STATS restores the former
    all-leaf sweep. Both IDENTICAL schedules use the same rows, stripe width,
    and floating-point reduction for each changed leaf (`IDENTITY_PATHS.md`
    row 7); neither substitutes histogram-derived sums.
    ===============================================
    """
    # DEVIATION 1902's schedule is a comptime row (RIDX_ONLY_SPLITS); under
    # IDENTICAL (Apple, trees-apple2) it is taken only where its gathered
    # stat loads cost less than the reorder they save. lane/no-dim-idn: that
    # is the byte rule `ridx_schedule_pays` (shared with the symmetric
    # driver), not the old 64-feature cut placed between the board widths
    # 16 and 220 (M4 Pro, steward 1790608373786: depthwise taxi 0.969,
    # Istella 1.024); -D MOJOLEARN_GBDT_RIDX_COST_RULE_OFF restores the cut.
    # FAST keeps it at every width. One decision per tree, so a tree never
    # mixes the two schedules.
    var use_ridx = RIDX_ONLY_SPLITS and (
        not SPLIT_COST_IDENTICAL or ridx_schedule_pays(len(fold_counts))
    )
    if options.policy != GROW_DEPTHWISE and options.policy != GROW_LOSSGUIDE:
        raise Error(
            "fit_non_symmetric_tree is EGrowPolicy::Depthwise or Lossguide;"
            " call run_tree_layout for SymmetricTree, and Region is"
            " unimplemented"
        )
    var lossguide = options.policy == GROW_LOSSGUIDE
    options.check()
    # LG_EXACT_BATCH, per fit: the per-round score noise (a Cosine score with
    # random_strength) would draw once per round instead of once per leaf
    # split, so such a fit keeps one leaf per iteration.
    var lg_exact = False
    comptime if LG_EXACT_BATCH:
        lg_exact = lossguide and (
            options.random_strength == Float32(0.0)
            or options.score_function == SCORE_FUNCTION_L2
            or options.score_function == SCORE_FUNCTION_NEWTON_L2
        )

    var min_child_hessian = child_hessian_threshold(options.min_child_hessian, options.policy, options.score_function)
    var stat_count = 1 + approx_dim
    var max_leaves = options.max_leaves
    var max_depth = options.max_depth
    # The symmetric pool's leaf key is what `TTreeWorkspace` stores, `1 <<
    # max_depth`'s slots; historically compared with `max_leaves`.
    var ws_leaves_key = max_leaves
    # LG_EXACT_BATCH: `max_leaves` below is the leaf CAPACITY (slots); the
    # policy reads `options.max_leaves`. `lg_room_bound` says the capacity
    # is below what `max_depth` allows, so a round must leave room for the
    # certain splits still to come.
    var lg_room_bound = False
    comptime if LG_EXACT_BATCH:
        if lg_exact:
            var depth_leaves = options.max_leaves
            if max_depth < 30:
                depth_leaves = 1 << max_depth
                # the pool is keyed on the slots it holds, so a Lossguide
                # fit whose max_leaves is not 1 << max_depth keeps its pool
                # from tree to tree
                ws_leaves_key = depth_leaves
            if depth_leaves > options.max_leaves:
                max_leaves = 2 * options.max_leaves
                if max_leaves >= depth_leaves:
                    max_leaves = depth_leaves
                else:
                    lg_room_bound = True

    var layout = build_layout(fold_counts, one_hot)
    var blocks = blocks_for(layout, n_rows)
    var hist_cells_per_leaf = layout.hist_cells
    # LG_EXACT_ID memory gate (T1): the doubled leaf capacity must fit the
    # leaf histograms the symmetric pool holds (`1 << max_depth` slots, see
    # `TTreeWorkspace`) and stay inside Int32 cell offsets; otherwise this
    # fit grows one leaf per iteration (same bits, the banner's argument).
    comptime if LG_EXACT_ID:
        if lg_exact and max_leaves > options.max_leaves:
            var lg_gate_ok = max_depth < 30 and max_leaves <= (1 << max_depth)
            if lg_gate_ok:
                var lg_cells = max_leaves * stat_count * hist_cells_per_leaf
                lg_gate_ok = lg_cells <= LG_EXACT_ID_MAX_CELLS
            if not lg_gate_ok:
                lg_exact = False
                max_leaves = options.max_leaves
                ws_leaves_key = options.max_leaves
                lg_room_bound = False
    # DEVIATION 2007a: the SM count is read off the symmetric pool below
    # (one `ctx.get_attribute` per WORKSPACE build, not per tree -- the
    # query is 1.26 ms/call on Metal, the price note in
    # `gpu_util/partitions_reduce.mojo`). The override keeps its old
    # meaning: positive replaces the machine constant, byte for byte.
    var sm_count = sm_count_override

    # `argmaxBlockCount = Min(CeilDivide(binFeatureCountPerDevice, 256), 64)`
    # (`greedy_search_helper.cpp:439`).
    var argmax_blocks = (hist_cells_per_leaf + 255) // 256
    if argmax_blocks > 64:
        argmax_blocks = 64
    if argmax_blocks < 1:
        argmax_blocks = 1

    var wide = (n_rows + 255) // 256
    if wide > 256:
        wide = 256
    if wide < 1:
        wide = 1

    # ---- THE POOLS. The symmetric lane's, unchanged, plus this lane's ----
    # DEVIATION 1892: the accumulator's liveness is part of the key (see
    # `acc_i32_is_live`'s docstring in `greedy_search_helper.mojo`).
    comptime _ACC_LIVE = acc_i32_is_live[hist2_smem_mode]()
    if (
        len(ws) == 0
        or ws[0].n_rows_key != n_rows
        or ws[0].stat_count_key != stat_count
        or ws[0].max_leaves_key != ws_leaves_key
        or ws[0].n_features_key != len(fold_counts)
        or ws[0].hist_cells_per_leaf_key != hist_cells_per_leaf
        or ws[0].acc_live_key != _ACC_LIVE
    ):
        ws.clear()
        ws.append(
            TTreeWorkspace(
                ctx, layout, blocks, n_rows, stat_count, max_depth,
                _ACC_LIVE,
            )
        )
    # DEVIATION 2007a: no override, so the pool's cached machine constant.
    if sm_count <= 0:
        sm_count = ws[0].sm_count
    # ============ DEVIATION 1911/1912: is the quantized family running? ====
    # The vendor/mode half is COMPTIME (`QUANTIZED_HIST_LIVE`, the
    # `greedy_quantized_hist_for` row -- False under IDENTICAL, so that
    # build folds every consumer away and keeps its schedule byte for
    # byte). The shape half is a per-fit SHAPE test off the layout: every
    # policy block one-byte, exactly two stat planes; anything else runs
    # the standing arms unchanged. Decided ONCE here, before the pool, so
    # buffers and dispatch cannot disagree.
    var qh_ok = False
    comptime if QUANTIZED_HIST_LIVE:
        qh_ok = quantized_hist_shape_ok(blocks, stat_count)
    # ======================================================================
    if (
        len(dws) == 0
        or dws[0].max_leaves_key != max_leaves
        or dws[0].stat_count_key != stat_count
        or dws[0].argmax_blocks_key != argmax_blocks
        # DEVIATION 1904 (wired): the bin-feature table planes are sized
        # and filled from the layout, which the three keys above cannot
        # see at equal argmax_blocks
        or dws[0].hist_cells_key != hist_cells_per_leaf
        # DEVIATION 1911: the packed-pair plane is row-sized and the
        # accumulator's liveness is fit-derived; neither is visible to
        # the keys above
        or dws[0].n_rows_key != n_rows
        or dws[0].qh_key != qh_ok
        # DW_NO_LEVEL_SYNC: the feature table is one record per feature
        or dws[0].n_features_key != len(layout.features)
    ):
        dws.clear()
        dws.append(
            TDepthwiseWorkspace(
                ctx, max_leaves, stat_count, argmax_blocks,
                hist_cells_per_leaf, n_rows, qh_ok, len(layout.features),
            )
        )

    dws[0].final_ready = False

    ref hist = ws[0].hist
    ref acc_i32 = ws[0].acc_i32
    ref block_hist = ws[0].block_hist
    ref dblocks = ws[0].dblocks
    ref p_off = ws[0].p_off
    ref p_sz = ws[0].p_sz
    ref hp_off = ws[0].hp_off
    ref hp_sz = ws[0].hp_sz
    ref h_off = ws[0].h_off
    ref h_sz = ws[0].h_sz
    ref part_stats = ws[0].part_stats
    ref stat_partials = ws[0].stat_partials
    ref flags = ws[0].flags
    ref seq = ws[0].seq
    ref gmap = ws[0].gmap
    ref sflags = ws[0].sflags
    ref new_index = ws[0].new_index
    ref new_stats = ws[0].new_stats
    ref chunk_zeros = ws[0].chunk_zeros
    ref chunk_offsets = ws[0].chunk_offsets
    ref leaf_zeros = ws[0].leaf_zeros
    ref skip = ws[0].skip
    ref bff = ws[0].bff
    ref ffw = ws[0].ffw
    ref flat_first = ws[0].flat_first
    ref flat_folds = ws[0].flat_folds
    ref flat_one_hot = ws[0].flat_one_hot
    ref sp_feats = dws[0].d_sp_feats
    ref sp_feats_h = dws[0].h_sp_feats
    ref sp_bins = dws[0].d_sp_bins
    ref sp_bins_h = dws[0].h_sp_bins
    ref dense_ids = ws[0].dense_ids

    ref region_score = dws[0].region_score
    ref region_bin = dws[0].region_bin
    ref h_region_score = dws[0].h_region_score
    ref h_region_bin = dws[0].h_region_bin
    ref d_visit = dws[0].d_visit
    ref h_visit = dws[0].h_visit
    ref d_left = dws[0].d_left
    ref h_left = dws[0].h_left
    ref d_right = dws[0].d_right
    ref h_right = dws[0].h_right
    ref d_ids = dws[0].d_ids
    ref h_ids = dws[0].h_ids
    ref d_zero_ids = dws[0].d_zero_ids
    ref h_zero_ids = dws[0].h_zero_ids
    ref d_all_ids = dws[0].d_all_ids
    ref h_all_ids = dws[0].h_all_ids
    ref h_part_stats = dws[0].h_part_stats
    ref d_win_cells = dws[0].d_win_cells
    ref h_win_cells = dws[0].h_win_cells
    ref d_copy_src = dws[0].d_copy_src
    ref h_copy_src = dws[0].h_copy_src
    ref d_copy_dst = dws[0].d_copy_dst
    ref h_copy_dst = dws[0].h_copy_dst
    ref d_ids_arena = dws[0].d_ids_arena
    ref d_sub_left = dws[0].d_sub_left
    ref d_sub_right = dws[0].d_sub_right
    var h_ids_arena_p = dws[0].h_ids_arena.unsafe_ptr().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    # host list of every arena slot, in `IDS_SLOT_*` order (the subtract and
    # split pairs both stage from `h_left`/`h_right`: the arena copy is taken
    # at upload time, into separate slots)
    var ids_host = List[MutPointer[UInt32, MutUntrackedOrigin]]()
    ids_host.append(h_copy_src.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ids_host.append(h_copy_dst.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ids_host.append(h_zero_ids.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ids_host.append(h_ids.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ids_host.append(h_all_ids.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ids_host.append(h_visit.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ids_host.append(h_left.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ids_host.append(h_right.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    for fs in range(IDS_FEAT_SLOTS):
        ids_host.append(
            sp_feats_h.unsafe_ptr()
            .unsafe_offset(fs * max_leaves)
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
    ids_host.append(sp_bins_h.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ids_host.append(h_left.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ids_host.append(h_right.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ids_host.append(h_win_cells.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ref d_bf_feature = dws[0].d_bf_feature
    ref h_bf_feature = dws[0].h_bf_feature
    ref d_bf_bin = dws[0].d_bf_bin
    ref h_bf_bin = dws[0].h_bf_bin
    ref d_bf_one_hot = dws[0].d_bf_one_hot
    ref h_bf_one_hot = dws[0].h_bf_one_hot
    ref d_bf_folds = dws[0].d_bf_folds
    ref h_bf_folds = dws[0].h_bf_folds
    ref d_winner = dws[0].d_winner
    ref h_winner = dws[0].h_winner
    # DEVIATION 1911/1912: the quantized family's planes
    ref d_qstats = dws[0].d_qstats
    ref d_qacc = dws[0].d_qacc

    var table = TBinFeatureTable(layout)

    # ============ DEVIATION 1904 (wired): the table goes to the device ====
    # The four columns the device-side winner fold resolves through --
    # the SAME `TBinFeatureTable` the host fold resolved through, uploaded
    # once per tree so record and cell cannot disagree with `to_split`.
    # Every mode (lane cpu3-gbdt-a: IDENTICAL takes the device fold too; it
    # is the host fold verbatim, so no bit moves). No drain: the staging
    # pairs are pool-owned
    # (DEVIATION 261's rule -- one pair per list, written once per tree),
    # the first `score.read` wait settles the copies, and the next tree's
    # rewrite sits behind this tree's own waits.
    for bf0 in range(hist_cells_per_leaf):
        h_bf_feature.unsafe_ptr().unsafe_store(bf0, table.feature[bf0])
        h_bf_bin.unsafe_ptr().unsafe_store(bf0, table.bin[bf0])
        h_bf_one_hot.unsafe_ptr().unsafe_store(
            bf0, UInt8(1) if table.one_hot[bf0] else UInt8(0)
        )
        h_bf_folds.unsafe_ptr().unsafe_store(bf0, table.folds[bf0])
    ctx.enqueue_copy(
        dst_buf=d_bf_feature, src_ptr=h_bf_feature.unsafe_ptr()
    )
    ctx.enqueue_copy(dst_buf=d_bf_bin, src_ptr=h_bf_bin.unsafe_ptr())
    ctx.enqueue_copy(
        dst_buf=d_bf_one_hot, src_ptr=h_bf_one_hot.unsafe_ptr()
    )
    ctx.enqueue_copy(dst_buf=d_bf_folds, src_ptr=h_bf_folds.unsafe_ptr())
    # ======================================================================

    # DW_NO_LEVEL_SYNC: whether this tree's levels take one wait, and the
    # per-feature `CFeature` table the device selection copies from (the
    # same record the host split loop packs, offset in elements). Written
    # once per tree; the previous tree's waits settled the last copy.
    ref d_feat_table = dws[0].d_feat_table
    ref h_feat_table = dws[0].h_feat_table
    ref d_nsplit = dws[0].d_nsplit
    ref h_nsplit = dws[0].h_nsplit
    var no_sync_tree = False
    comptime if DW_NO_LEVEL_SYNC:
        comptime assert (
            DW_FEAT_WORDS == IDS_FEAT_SLOTS
        ), "dw_select_splits_kernel copies CFEATURE_BYTES // 4 words"
        comptime assert (
            DW_WINNER_WORDS == WINNER_RECORD_WORDS
        ), "dw_select_splits_kernel reads the fold's record layout"
        no_sync_tree = (
            use_ridx
            and not lossguide
            and options.min_split_gain < Float64(0)
        )
        if no_sync_tree:
            var ft = h_feat_table.unsafe_ptr().bitcast[CFeature]()
            for fi in range(len(layout.features)):  # small-loop(features: layout feature descriptors, once per tree): CFeature table staging, metadata only
                var lf = layout.features[fi]
                ft[unsafe_offset=fi] = CFeature(
                    lf.offset * UInt32(n_rows),
                    lf.mask,
                    lf.shift,
                    lf.first_fold_index,
                    lf.folds,
                    lf.one_hot_feature,
                )
            ctx.enqueue_copy(
                dst_buf=d_feat_table, src_ptr=h_feat_table.unsafe_ptr()
            )

    # ============ THEIR `subsets.FeatureWeights`, WHICH WAS BEING DROPPED ===
    # `CreateInitialSubsets` writes `Options.FeatureWeights` into
    # `subsets.FeatureWeights` (`split_properties_helper.cpp:1075-1076`) and
    # the kernel multiplies every candidate's gain by
    # `binFeaturesWeights[featureId]` (`compute_scores.cu:467`).
    #
    # This driver used to hand the kernel `ws[0].ffw`, which the workspace
    # fills with 1.0 for every feature and nothing ever overwrites. A caller
    # who set `feature_weights` got them ACCEPTED AND DROPPED -- the exact
    # failure `CONTRIBUTING.md` (Evidence must be able to fail) exists to prevent, sitting under an
    # options docstring that read as though they were honored. Found by an
    # audit against their source, not by a gate.
    #
    # Empty stays empty: their `UpdateFeatureWeightsForBestSplits` leaves
    # 1.0 everywhere when there are no CTRs (`update_feature_weights.cpp
    # :14-22`), which is what the workspace already holds.
    if len(options.feature_weights) != 0:
        if len(options.feature_weights) != len(fold_counts):
            raise Error(
                String("feature_weights must be one per feature: got ")
                + String(len(options.feature_weights))
                + " for "
                + String(len(fold_counts))
                + " features"
            )
        var hfw = ctx.enqueue_create_host_buffer[DType.float32](
            len(fold_counts)
        )
        for i in range(len(fold_counts)):
            hfw.unsafe_ptr().unsafe_store(i, options.feature_weights[i])
        ctx.enqueue_copy(dst_buf=ws[0].ffw, src_ptr=hfw.unsafe_ptr())
        ctx.synchronize()
        # past the drain; `mojo-buffer-freed-at-last-use`
        _ = hfw^

    # ================= CreateInitialSubsets =========================
    # `split_properties_helper.cpp:1043-1080`: zero the partitions, write
    # the root partition over every row, zero the stats and the histograms,
    # push ONE leaf, then `RebuildLeavesSizes`.
    for i in range(max_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(0))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    enqueue_fill(ctx, hist, Float32(0.0))
    # DEVIATION 1892: same gate as the symmetric driver's per-tree memset
    # -- under a float flush with the warp-private `hist2` arm nothing
    # writes or reads `acc_i32`, so the per-tree zeroing is skipped and
    # the buffer is the pool's one-cell placeholder.
    comptime if _ACC_LIVE:
        enqueue_fill(ctx, acc_i32, Int32(0))

    # The fixed-point scale, host-derived. DEVIATION 95's device derivation
    # is not wired here: it exists to remove the boosting loop's per-tree
    # magnitudes drain, and this lane has no boosting loop yet. Same
    # function, same bits (`choose_scale` is an exact integer search).
    # lane/fam-gbdt (IDN_NS_SCALE_DEVICE): with the magnitudes buffer the
    # same derivation runs on the device, as in `run_tree_layout`.
    var fixed_scale = rebind[MutPointer[Float32, MutAnyOrigin]](
        ws[0].scale_dev.unsafe_ptr()
    )
    if mags_dev:
        ctx.enqueue_function[choose_scale_kernel](  # small-launch(n_rows: a scalar operand of the scale snap): one thread of control plane reading two magnitudes
            rebind[MutPointer[Float32, MutAnyOrigin]](
                mags_dev.value().unsafe_ptr()
            ),
            Int32(n_rows), fixed_scale,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
    else:
        var mag = Float64(weight_magnitude)
        if mag < 0.0:
            mag = -mag
        var gmag = Float64(gradient_magnitude)
        if gmag < 0.0:
            gmag = -gmag
        if gmag > mag:
            mag = gmag
        ws[0].h_scale.unsafe_ptr().unsafe_store(
            0, Float32(choose_scale(mag, n_rows))
        )
        ctx.enqueue_copy(
            dst_buf=ws[0].scale_dev, src_ptr=ws[0].h_scale.unsafe_ptr()
        )
    # lane/sym-quality: the gradient planes onto this tree's fixed-point
    # grid before the root histogram (`snap_gradients_to_scale_kernel`), as
    # the symmetric driver does; only where a histogram quantizes at all.
    comptime if _ACC_LIVE:
        enqueue_snap_gradients(ctx, stats, n_rows, stat_count, fixed_scale)

    var leaves = List[TLeaf]()
    var root = TLeaf()
    root.size = n_rows
    leaves.append(root^)
    # the sibling-pairing key of `_leaf_records`; the root has no parent and
    # is `Zeroes`, so its value is never read.
    var parent_of = List[Int]()
    parent_of.append(0)
    # DEVIATION 1901: the winning HISTOGRAM CELL per leaf, parallel to
    # `leaves`. The reduce below resolves the cell to `(feature, bin)` for
    # the split record and then FORGETS it, but the cell is the address the
    # propagation kernel needs -- the scanned prefix at the winner IS the
    # left child's stat sums. Written exactly where `update_best_split` is
    # called, so the two cannot desynchronize; -1 is "no winner stored",
    # mirroring the record's own undefined state. The bookkeeping runs in
    # BOTH numeric modes (host integers, numerically inert); only the
    # staging, the guard and the kernel that CONSUME it are FAST-gated.
    var best_cells = List[Int32]()
    best_cells.append(Int32(-1))
    # DEVIATION 1903: whether a leaf id's histogram slot has EVER been
    # written this tree. The arena is memset ONCE per tree
    # (`CreateInitialSubsets` below, LightGBM's own cadence --
    # `cuda_histogram_constructor.cpp:76-80`), leaf ids are never reused
    # within a tree, and only the zero/build/scan/subtract/copy launches
    # write a slot -- all of them over id lists this driver stages. So a
    # False here is a proof the slot still holds the tree memset's zeros,
    # and the per-level zero pass can skip it. Conservative in the other
    # direction: every id in a level's `updated` set is marked True, even
    # ones that were only zeroed. FAST-arm bookkeeping; under IDENTICAL the
    # list exists and every compute slot is zeroed exactly as before.
    var hist_slot_dirty = List[Bool]()
    hist_slot_dirty.append(False)

    # LG_EXACT_BATCH: the tree the device has grown, as nodes. A node is a
    # leaf slot's content between two splits of that slot; `lg_leaf_node`
    # names the node each leaf slot holds now. `lg_node_path` is filled when
    # a node is split (a leaf's own path is `leaves[...]`'s).
    var lg_leaf_node = List[Int]()
    var lg_node_leaf = List[Int]()
    var lg_node_left = List[Int]()
    var lg_node_right = List[Int]()
    var lg_node_gain = List[Float32]()
    var lg_node_state = List[Int]()
    var lg_node_path = List[TLeafPath]()
    var lg_final = List[Int]()
    # LG_EXACT_ID: each node's partition stats as of its score wait, the
    # stats a folded-back leaf keeps (`stat_count` per node)
    var lg_node_stats = List[Float32]()
    # Default-off IDENTICAL lifetime candidate: keep scored parent words
    # on device rather than copying the whole leaf-stat capacity each round.
    # One current snapshot plus the binary tree's bounded node arena.
    var lg_resident_stats = List[DeviceBuffer[DType.float32]]()
    var lg_resident_snapshots = 0
    # I17 2026-10-06 AMD MI325X Lossguide scoped LOSER (source 5b467815b):
    # candidate/base 1.041, 1.292, 1.083 at rows/features10000/17,10001/18,32769/9.
    # Complete 10-tree fits; logs show six resident snapshots/tree. Depthwise
    # recorded zero snapshots: those controls do not measure this candidate.
    # One same-process warmup and score; accepted identity evidence reused.
    # NVIDIA/full-workload qualification pending. Keep resident frontier OFF.
    # Evidence: overnight-ab-20261006/amd/normalized-measurements.json, I17;
    # exact snapshot counts and raw timings remain in amd/live/repairs.
    comptime if LG_EXACT_ID and is_defined["MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT"]():
        if lg_exact and max_leaves>0 and stat_count>0 and max_leaves<=(1<<22)//(3*stat_count):
            lg_resident_stats.append(ctx.enqueue_create_buffer[DType.float32]((2*max_leaves-1)*stat_count))
            lg_resident_stats.append(ctx.enqueue_create_buffer[DType.float32](max_leaves*stat_count))
            enqueue_fill(ctx,lg_resident_stats[0],Float32(0))
    comptime if LG_EXACT_BATCH:
        for _ in range(stat_count):  # small-loop(stat_count: stat planes, 1 plus classes): root node stats placeholder
            lg_node_stats.append(Float32(0.0))
        lg_leaf_node.append(0)
        lg_node_leaf.append(0)
        lg_node_left.append(-1)
        lg_node_right.append(-1)
        lg_node_gain.append(Float32.MAX)
        lg_node_state.append(LG_NODE_UNKNOWN)
        lg_node_path.append(TLeafPath())

    var part_stats_dirty = List[Bool]()
    comptime if INCREMENTAL_PART_STATS:
        part_stats_dirty.append(True)  # the root has no cached reduction
    var part_stats_rows_reduced = Int64(0)
    var part_stats_leaves_reduced = Int64(0)

    var emit_digests = trace.enabled
    var hist_live_stride = stat_count * hist_cells_per_leaf

    # ---- STAGE WALL TIMERS, off unless MOJOLEARN_STAGE_TIMES=1 ----
    # `depthwise_stage_times.mojo`. Environment read ONCE, here, per fit.
    # Enabled, every begin/end DRAINS, so a stage-timed run is triage and
    # never a benchmark -- the same standing as a traced run (identity_trace
    # rule 4). Do not enable together with the identity trace: the trace's
    # own drains would be billed to whatever stage contains them.
    var stage_times = StageTimes()

    var mgr = TCudaManager(ctx.copy(), sync_budget=-1)

    # their `TGpuAwareRandom`, drawn from ONCE PER `ComputeOptimalSplits`
    # call (`greedy_search_helper.cpp:487`), i.e. per LEVEL. Same per-tree
    # re-seed as `run_tree_layout` (DEVIATION 139).
    var level_rand = TRandom(random_seed)

    # ============ `CreateInitialSubsets`' ScoreStdDev, ONCE PER TREE ========
    # `greedy_search_helper.cpp:384-388`, verbatim:
    #
    #     if (Options.RandomStrength) {
    #         ScoreStdDev = Options.RandomStrength * ComputeTargetStdDev(target);
    #     } else {
    #         ScoreStdDev = 0;
    #     }
    #
    # and it is `ScoreStdDev` -- the PRODUCT -- that becomes the kernel's
    # `scoreStdDev` argument (`:488`), not `RandomStrength`.
    #
    # THIS WAS WRONG UNTIL 2026-08-22: `options.random_strength` went
    # straight into the kernel slot and `ComputeTargetStdDev` was never
    # called, so the noise magnitude was short by a factor of the target's
    # standard deviation. It was inert only because every caller passes 0,
    # and it was found by an audit against their source and not by any gate
    # -- which is the whole argument for reading their file against ours
    # rather than trusting green checks.
    #
    # THE TARGET IS THE BOOTSTRAPPED ONE, as in the symmetric lane: their
    # `ComputeTarget` runs `StochasticDer(bootstrapConfig, ...)` and the
    # std dev is taken from its output, so a bootstrap that scales the
    # planes scales the noise with them.
    var score_std_dev = Float32(0.0)
    if options.random_strength != Float32(0.0):
        score_std_dev = Float32(
            Float64(options.random_strength)
            * compute_target_std_dev(
                ctx, stats, n_rows, stat_count, n_rows,
                multiclass_optimization, sm_count,
            )
        )


    var result_paths = List[TLeafPath]()
    var result_weights = List[Float64]()
    var result_values = List[List[Float32]]()
    # NS_INHERIT_PARTITION: the result leaves' row ranges, handed to the
    # pool after the growth loop
    var inherit_offsets = List[Int]()
    var inherit_sizes = List[Int]()
    var inherit_ready = False

    # Their `while (true)`. The bound is OURS, and it is POLICY-SHAPED,
    # which is the whole point of this block.
    #
    # Depthwise splits every live leaf at most once per ITERATION and
    # `IsTerminalLeaf` stops at `MaxDepth`, so one iteration is one LEVEL
    # and `max_depth + 2` is unreachable unless the bookkeeping has broken.
    #
    # **LOSSGUIDE SPLITS EXACTLY ONE LEAF PER ITERATION**
    # (`greedy_search_helper.cpp:319-324`), so its iteration count is a LEAF
    # count, not a depth: `max_leaves - 1` splits grow `max_leaves` leaves,
    # at any depth. A depth-shaped bound is not conservative for it -- it is
    # WRONG, and it fires on correct trees.
    #
    # ================= HOW THIS WAS FOUND, because the shape of the
    # investigation is the lesson =================
    # The merge that gave this loop its Lossguide branch left the bound
    # alone, so every Lossguide fit needing more than `max_depth + 2` splits
    # raised. The message says "depthwise level loop", which is generic to
    # this driver, and I read it as an accusation of the DEPTHWISE ARM and
    # spent six probes eliminating options construction, fit helpers,
    # fixture reuse, claim ordering, extra live fixtures and borrowed
    # DeviceContexts -- all of which came back clean, because the depthwise
    # arm was never involved.
    #
    # The numbers had said so from the first run and I did not read them:
    # `max_depth 3 -> "5 iterations; leaves=6"`, `6 -> "8 iterations;
    # leaves=9"`, `4 -> "6 iterations; leaves=7"`. Every one is exactly
    # `max_depth + 2` iterations and exactly `iterations + 1` leaves --
    # one leaf per iteration, which IS Lossguide working correctly. A
    # diagnostic that reports its own state was telling me the answer
    # while I searched elsewhere. READ THE NUMBERS IN THE ERROR BEFORE
    # FORMING A HYPOTHESIS ABOUT WHICH COMPONENT IS WRONG.
    #
    # The message is now policy-named so the next reader is not sent to the
    # wrong arm.
    # ===============================================
    var max_iterations = max_depth + 2
    if lossguide:
        max_iterations = max_leaves + 1
    var iteration = 0
    while True:
        iteration += 1
        if iteration > max_iterations:
            raise Error(
                String("lossguide" if lossguide else "depthwise")
                + " growth loop did not terminate in "
                + String(max_iterations)
                + " iterations; leaves="
                + String(len(leaves))
                + " (a leaf is neither terminal nor improving and its"
                " histogram is being rebuilt forever)"
            )

        # DW_NO_LEVEL_SYNC: True once this level's split ran behind the
        # fold and its sizes are already home (one wait for the level)
        var level_synced = False

        # ================= ComputeOptimalSplits =====================
        # `greedy_search_helper.cpp:396`. The RNG draw is NOT here; see the
        # order block below.

        # --- SplitPropsHelper.BuildNecessaryHistograms(subsets) ---
        stage_times.begin(ctx)
        var records = _leaf_records(leaves, parent_of)
        var plan = build_necessary_histograms(records)
        var non_zero = non_zero_leaves(records, plan.compute_ids)
        stage_times.end(ctx, "host.plan")

        var d_tag = tag_prefix + String("d") + String(iteration - 1) + "."
        trace.record_list_i32(d_tag + "leaves", _one_i32(len(leaves)))
        trace.record_list_i32(
            d_tag + "plan.compute", _u32_as_i32(plan.compute_ids)
        )
        trace.record_list_i32(
            d_tag + "plan.subfrom", _u32_as_i32(plan.subtract_from)
        )
        trace.record_list_i32(
            d_tag + "plan.subwhat", _u32_as_i32(plan.subtract_what)
        )
        trace.record_list_i32(
            d_tag + "plan.nonzero", _u32_as_i32(non_zero)
        )

        # ============ trees-apple2: THE PLAN-TIME ID LISTS, STAGED ONCE ============
        # Every list below is a pure function of host state known here (the
        # plan, `hist_slot_dirty`, the leaves after their BestSplit reset,
        # `part_stats_dirty`), so it is built now, in the order the kernels
        # below consume it, and reaches the device in ONE copy per run of
        # adjacent arena slots (`_upload_id_slots`, ID_UPLOAD_COALESCE) instead
        # of one copy per list at each consumer. The kernels are unchanged
        # and read the same words; nothing on the device runs between the
        # old copy sites and the new one except kernels that do not read
        # these lists.
        var plan_slots = List[Int]()
        # DEVIATION 1903's deferred copy pairs (FAST arm)
        var n_copy = 0
        comptime if DEFER_HIST_COPY_1903:
            if len(plan.subtract_from) > 0:
                for i in range(len(plan.subtract_from)):  # small-loop(subtract_from: sibling pairs, at most max_leaves): histogram copy plan list
                    if Int(plan.subtract_from[i]) > Int(
                        plan.subtract_what[i]
                    ):
                        h_copy_src.unsafe_ptr().unsafe_store(
                            n_copy, plan.subtract_what[i]
                        )
                        h_copy_dst.unsafe_ptr().unsafe_store(
                            n_copy, plan.subtract_from[i]
                        )
                        n_copy += 1
                if n_copy > 0:
                    plan_slots.append(IDS_SLOT_COPY_SRC)
                    plan_slots.append(IDS_SLOT_COPY_DST)
        # the ZERO set (DEVIATION 1903: dirty slots only on the FAST arm)
        var zero_count = 0
        if len(plan.compute_ids) > 0:
            comptime if not DEFER_HIST_COPY_1903:
                for i in range(len(plan.compute_ids)):
                    h_zero_ids.unsafe_ptr().unsafe_store(
                        i, plan.compute_ids[i]
                    )
                zero_count = len(plan.compute_ids)
            else:
                for i in range(len(plan.compute_ids)):  # small-loop(compute_ids: leaves to build, at most max_leaves): zero-pass plan list
                    if hist_slot_dirty[Int(plan.compute_ids[i])]:
                        h_zero_ids.unsafe_ptr().unsafe_store(
                            zero_count, plan.compute_ids[i]
                        )
                        zero_count += 1
            if zero_count > 0:
                plan_slots.append(IDS_SLOT_ZERO)
        # the BUILD set
        if len(non_zero) > 0:
            for i in range(len(non_zero)):
                h_ids.unsafe_ptr().unsafe_store(i, non_zero[i])
            plan_slots.append(IDS_SLOT_IDS)
        # the SUBTRACT pairs, `from - what`
        if len(plan.subtract_from) > 0:
            for i in range(len(plan.subtract_from)):
                h_left.unsafe_ptr().unsafe_store(i, plan.subtract_from[i])
                h_right.unsafe_ptr().unsafe_store(i, plan.subtract_what[i])
            plan_slots.append(IDS_SLOT_SUB_LEFT)
            plan_slots.append(IDS_SLOT_SUB_RIGHT)

        # their `allUpdatedLeaves` loop (`:1359-1367`): computed leaves plus
        # derived big leaves become `CurrentPath`, AND THEIR BestSplit IS
        # RESET. The reset is what makes `SelectLeavesToVisit` re-score
        # exactly the leaves whose histogram moved. (Host state only; it
        # used to sit after the subtract launch, which does not read it.)
        var updated = plan.updated_ids()
        for i in range(len(updated)):  # small-loop(updated: leaves rebuilt, at most max_leaves): histogram state flags per leaf
            var id = Int(updated[i])
            leaves[id].histograms_type = EHistogramsType.CurrentPath
            leaves[id].best_split = TBestSplitProperties()
            # DEVIATION 1903: every updated slot was written (zeroed,
            # built, copied into or derived), so it can no longer skip the
            # zero pass. Marked conservatively -- a slot that was only
            # zeroed is marked too. (After the ZERO set above, as before.)
            comptime if DEFER_HIST_COPY_1903:
                hist_slot_dirty[id] = True

        # `SelectLeavesToVisit` and, when there is something to visit, the
        # ALL set of the partition-stats sweep and the VISIT list (see the
        # order block below: both are after the early return).
        var visit = select_leaves_to_visit(leaves)
        var run_part_sweep = True
        comptime if not SPLIT_COST_IDENTICAL:
            run_part_sweep = iteration == 1
        var reduce_count = 0
        if len(visit) > 0:
            if run_part_sweep:
                for i in range(len(leaves)):  # small-loop(leaves: tree leaves, at most max_leaves): partition-stats reduce id list
                    comptime if INCREMENTAL_PART_STATS:
                        if not part_stats_dirty[i]:
                            continue
                        part_stats_dirty[i] = False
                    h_all_ids.unsafe_ptr().unsafe_store(reduce_count, UInt32(i))
                    reduce_count += 1
                    comptime if REPORT_PART_STATS_WORK:
                        part_stats_rows_reduced += Int64(leaves[i].size)
                        part_stats_leaves_reduced += 1
                if reduce_count > 0:
                    plan_slots.append(IDS_SLOT_ALL)
            for i in range(len(visit)):
                h_visit.unsafe_ptr().unsafe_store(i, UInt32(visit[i]))
            plan_slots.append(IDS_SLOT_VISIT)
        _upload_id_slots(
            ctx, d_ids_arena, h_ids_arena_p, ids_host, max_leaves,
            plan_slots^,
        )

        # ============================ DEVIATION 1903 ============================
        # THE PARENT-HISTOGRAM COPY MOVES FROM EVERY SPLIT TO THE PAIRS THAT
        # NEED IT, AND THE ZERO PASS TO THE SLOTS THAT NEED IT. LightGBM's
        # learner never copies a histogram at a split and never zeroes one
        # per level: `cuda_hist_pool_` makes the larger child ALIAS the
        # parent's slot and the fresh slot arrives zero from a once-per-tree
        # memset (`cuda_data_partition.cu:825-831,895-898`;
        # `cuda_histogram_constructor.cpp:76-80` -- recon_lightgbm_cuda.md,
        # mechanism a3 / borrow 3). This implementation's slots are leaf-id-indexed and
        # every kernel from the build to the scorer addresses them by the SAME
        # id it reads the partition with, so a pointer pool proper would touch
        # `greedy_search_helper.mojo`'s launcher and `kernel/compute_scores`
        # -- the other lane's files. What lands here is the aliasing the id
        # scheme gives for free:
        #
        #   * the LEFT child keeps the parent's id (`MakeSplit`, `:861-862`),
        #     so the parent's histogram already sits in the left child's slot
        #     -- when the plan derives the LEFT sibling (`big == left`, which
        #     includes every exact-size tie), the split-time
        #     copy was writing a slot that the very next level ZEROED. Copy
        #     deleted, subtraction unchanged: `from` is the left id and its
        #     slot holds the parent's totals, as `substract_histograms`
        #     requires.
        #   * when the plan derives the RIGHT sibling (`big == right`, left
        #     strictly smaller), the parent's totals must reach the right
        #     slot before the left one is zeroed and rebuilt -- the SAME
        #     kernel, launched HERE at plan time over exactly those pairs,
        #     src = the small left id, dst = the big right id.
        #   * a fresh right child's slot still holds the tree memset's zeros
        #     (`hist_slot_dirty` above), so zeroing it is a full
        #     hist-plane write for nothing; the pass runs over the dirty
        #     compute slots only.
        #
        # Bit-inert BY CONSTRUCTION on the FAST arm it runs on: every slot
        # holds the same bytes at every read as under the old schedule (a
        # deferred copy copies the same untouched bytes; an elided zero
        # leaves the memset's zeros where the kernel would have written
        # zeros). Price deleted per split: one full-histogram copy
        # (`hist_cells * stat_count * 4` bytes each way) on tied pairs, and
        # one full-histogram zero on fresh slots, x ~63 splits per lossguide
        # tree. IDENTICAL keeps the split-time copy and the full zero pass
        # byte for byte.
        # =======================================================================
        comptime if DEFER_HIST_COPY_1903:
            if len(plan.subtract_from) > 0:
                if n_copy > 0:
                    stage_times.begin(ctx)
                    # DEVIATION 1903 / 261: their own staging pairs, staged
                    # with the plan-time lists above
                    # WIDTH DISPATCH, the same kernels the split-time copy
                    # ran, over the reduced pair list.
                    if (hist_cells_per_leaf * stat_count) % 4 == 0:
                        ctx.enqueue_function[copy_histograms_vec4_kernel](
                            d_copy_src.unsafe_ptr(),
                            d_copy_dst.unsafe_ptr(),
                            Int32(stat_count),
                            Int32(hist_cells_per_leaf),
                            hist.unsafe_ptr(),
                            grid_dim=(
                                (hist_cells_per_leaf * stat_count // 4 + 255)
                                // 256,
                                n_copy,
                                1,
                            ),
                            block_dim=(256, 1, 1),
                        )
                    else:
                        ctx.enqueue_function[copy_histograms_kernel](
                            d_copy_src.unsafe_ptr(),
                            d_copy_dst.unsafe_ptr(),
                            Int32(stat_count),
                            Int32(hist_cells_per_leaf),
                            hist.unsafe_ptr(),
                            grid_dim=(
                                (hist_cells_per_leaf * stat_count + 255)
                                // 256,
                                n_copy,
                                1,
                            ),
                            block_dim=(256, 1, 1),
                        )
                    mgr.stream_kernel()
                    stage_times.end(ctx, "hist.copy")

        if len(plan.compute_ids) > 0:
            # their `ZeroLeavesHistograms(zeroLeaves, subsets)`
            # (`:1350`), applied to EVERY compute slot rather than only the
            # empty ones. A computed slot is overwritten by the build
            # either way and an empty leaf's zeros ARE its histogram, so
            # this subsumes their split; it is the symmetric lane's choice
            # and is kept identical so the two cannot drift.
            #
            # DEVIATION 1903: on the FAST arm the list drops the slots that
            # provably still hold the tree memset's zeros (`hist_slot_dirty`
            # False) -- writing zeros over zeros is the one launch in this
            # step that can be deleted without an argument about the build.
            # (the ZERO set and its count were staged above)
            if zero_count > 0:
                stage_times.begin(ctx)
                ctx.enqueue_function[zero_histograms_kernel](
                    d_zero_ids.unsafe_ptr(),
                    Int32(hist_cells_per_leaf),
                    hist.unsafe_ptr(),
                    grid_dim=(
                        (hist_cells_per_leaf + 255) // 256,
                        zero_count,
                        stat_count,
                    ),
                    block_dim=(256, 1, 1),
                )
                mgr.stream_kernel()
                stage_times.end(ctx, "hist.zero")

        if len(non_zero) > 0:
            # their `ComputeSplitProperties(loadPolicy, nonZeroComputeLeaves,
            # subsets)` (`:1347`). `ids` must be the NON-EMPTY set and the
            # call must not happen at all when it is empty -- their
            # `if (leavesToCompute.size() == 0) { return; }` (`:1089`).
            stage_times.begin(ctx)
            # (the BUILD set was staged above)
            # ============================ DEVIATION 1911/1912 ============================
            # THE QUANTIZED SHARED-HISTOGRAM ROUTE (the recons' borrow:
            # XGBoost's one-shared-copy-per-block integer histogram +
            # LightGBM's packed pair -- `kernel/hist_quantized_shared.mojo`
            # carries the whole design). Reachable only when the comptime
            # row admits this build (`QUANTIZED_HIST_LIVE`: FAST on a
            # claimed column; IDENTICAL folds the branch away and runs the
            # standing launcher byte for byte) AND the fit's shape fits
            # (`qh_ok`: all-one-byte, two stats). The fallback is the
            # standing launcher UNCHANGED -- refusal of shape to the old
            # path, never a guess. The bridge inside
            # `launch_quantized_histograms` leaves the flat histogram in
            # exactly the layout the scan below expects, so nothing after
            # this branch knows which arm built it.
            # =======================================================================
            var quantized_built = False
            comptime if QUANTIZED_HIST_LIVE:
                if qh_ok:
                    if use_ridx:
                        launch_quantized_histograms[True](
                            ctx, dblocks, iteration - 1, len(non_zero), n_rows,
                            stat_count, sm_count, fixed_scale,
                            cindex, row_index, stats, p_off, p_sz, d_ids,
                            d_qstats, d_qacc, hist, hist_cells_per_leaf,
                            _dw_dev_u32(dws[0].d_qskip, 0), dws[0].n_features_key,
                        )
                    else:
                        launch_quantized_histograms[False](
                            ctx, dblocks, iteration - 1, len(non_zero), n_rows,
                            stat_count, sm_count, fixed_scale,
                            cindex, row_index, stats, p_off, p_sz, d_ids,
                            d_qstats, d_qacc, hist, hist_cells_per_leaf,
                            _dw_dev_u32(dws[0].d_qskip, 0), dws[0].n_features_key,
                        )
                    quantized_built = True
            if not quantized_built:
                # DEVIATION 2661 (see `NONSYM_GROUP_WIDTH_2661`): the same
                # launcher, told to take the fit's per-group width plans.
                # `level_quant` stays False, so no Int32 level plane is
                # needed and none is passed.
                comptime if NONSYM_GROUP_WIDTH_2661:
                    if use_ridx:
                        launch_histograms_for_blocks[
                            hist2_smem_mode, True, False, True
                        ](
                            ctx, dblocks, iteration - 1, len(non_zero), n_rows,
                            stat_count, max_leaves, sm_count, fixed_scale,
                            cindex, row_index, stats, p_off, p_sz, d_ids,
                            dense_ids, hist, acc_i32, block_hist,
                            hist_cells_per_leaf,
                            width_plans=ws[0].width_plans,
                        )
                    else:
                        launch_histograms_for_blocks[
                            hist2_smem_mode, False, False, True
                        ](
                            ctx, dblocks, iteration - 1, len(non_zero), n_rows,
                            stat_count, max_leaves, sm_count, fixed_scale,
                            cindex, row_index, stats, p_off, p_sz, d_ids,
                            dense_ids, hist, acc_i32, block_hist,
                            hist_cells_per_leaf,
                            width_plans=ws[0].width_plans,
                        )
                else:
                    if use_ridx:
                        launch_histograms_for_blocks[
                            hist2_smem_mode, True
                        ](
                            ctx, dblocks, iteration - 1, len(non_zero), n_rows,
                            stat_count, max_leaves, sm_count, fixed_scale,
                            cindex, row_index, stats, p_off, p_sz, d_ids,
                            dense_ids, hist, acc_i32, block_hist,
                            hist_cells_per_leaf,
                        )
                    else:
                        launch_histograms_for_blocks[
                            hist2_smem_mode, False
                        ](
                            ctx, dblocks, iteration - 1, len(non_zero), n_rows,
                            stat_count, max_leaves, sm_count, fixed_scale,
                            cindex, row_index, stats, p_off, p_sz, d_ids,
                            dense_ids, hist, acc_i32, block_hist,
                            hist_cells_per_leaf,
                        )
            mgr.stream_kernel()
            stage_times.end(ctx, "hist.build")

            # their `TScanHistogramsKernel`, over the BUILT set. A prefix
            # sum is linear, so the derived sibling needs no scan and an
            # all-zero slot scans to itself.
            stage_times.begin(ctx)
            comptime if DW2_SCAN_SMEM:
                # lane apple-fast-dwgap2: the same serial fold over a
                # shared-memory copy of 16 features per block
                ctx.enqueue_function[dw2_scan_histograms_smem_kernel](
                    d_ids.unsafe_ptr(),
                    flat_first.unsafe_ptr(),
                    flat_folds.unsafe_ptr(),
                    flat_one_hot.unsafe_ptr(),
                    Int32(len(fold_counts)),
                    Int32(hist_cells_per_leaf),
                    hist.unsafe_ptr(),
                    grid_dim=(
                        (len(fold_counts) + DW2_SCAN_FT - 1) // DW2_SCAN_FT,
                        len(non_zero),
                        stat_count,
                    ),
                    block_dim=(DW2_SCAN_BLOCK, 1, 1),
                )
            else:
                ctx.enqueue_function[scan_histograms_kernel](
                    d_ids.unsafe_ptr(),
                    flat_first.unsafe_ptr(),
                    flat_folds.unsafe_ptr(),
                    flat_one_hot.unsafe_ptr(),
                    Int32(len(fold_counts)),
                    Int32(hist_cells_per_leaf),
                    hist.unsafe_ptr(),
                    grid_dim=(
                        (len(fold_counts) + 255) // 256,
                        len(non_zero),
                        stat_count,
                    ),
                    block_dim=(256, 1, 1),
                )
            mgr.stream_kernel()
            stage_times.end(ctx, "hist.scan")
            trace.record_device(
                ctx, d_tag + "hist.scanned", hist,
                len(leaves) * hist_live_stride,
            )

        if len(plan.subtract_from) > 0:
            # their `SubstractHistograms(bigLeaves, smallLeaves, subsets)`
            # (`:1354`): `from - what`, in place, one launch for all pairs.
            stage_times.begin(ctx)
            # (the pairs were staged above, into their own arena slots)
            # WIDTH DISPATCH, the symmetric lane's, which this driver was
            # missing until 2026-08-22. The vec4 arms are not a symmetric
            # optimization -- they are a property of the BUFFER, whose
            # layout is identical under every grow policy -- and both
            # non-symmetric policies were silently taking the scalar path.
            # The copy arm's own deviation block records 11.0 -> 65.2 GB/s
            # at a depth-6 level's shape. Found by the lossguide lane
            # reading the two drivers side by side, which is the whole
            # argument for there being one driver.
            if hist_cells_per_leaf % 4 == 0:
                ctx.enqueue_function[substract_histograms_vec4_kernel](
                    d_sub_left.unsafe_ptr(),
                    d_sub_right.unsafe_ptr(),
                    Int32(hist_cells_per_leaf),
                    hist.unsafe_ptr(),
                    grid_dim=(
                        (hist_cells_per_leaf // 4 + 255) // 256,
                        len(plan.subtract_from),
                        stat_count,
                    ),
                    block_dim=(256, 1, 1),
                )
            else:
                ctx.enqueue_function[substract_histograms_kernel](
                    d_sub_left.unsafe_ptr(),
                    d_sub_right.unsafe_ptr(),
                    Int32(hist_cells_per_leaf),
                    hist.unsafe_ptr(),
                    grid_dim=(
                        (hist_cells_per_leaf + 255) // 256,
                        len(plan.subtract_from),
                        stat_count,
                    ),
                    block_dim=(256, 1, 1),
                )
            mgr.stream_kernel()
            stage_times.end(ctx, "hist.subtract")
            trace.record_device(
                ctx, d_tag + "hist.subtracted", hist,
                len(leaves) * hist_live_stride,
            )

        # (the `allUpdatedLeaves` BestSplit reset now runs above, with the
        # plan-time staging)

        # THEIR ORDER, AND IT IS NOT COSMETIC (`greedy_search_helper.cpp`):
        #
        #     SplitPropsHelper.BuildNecessaryHistograms(subsets);   :399
        #     SelectLeavesToVisit(*subsets, &leavesToVisit);        :401
        #     if (leavesToVisit.empty()) { return; }                :402-405
        #     ...
        #     AllReduceThroughMaster(subsets->CurrentPartStats(),.) :443
        #     Random.NextUniformL()                            :469/:489/:510
        #
        # Both the stats reduce AND the random draw are AFTER the early
        # return, so a level with nothing to visit does neither. This driver
        # had the reduce before the visit list and drew the seed
        # unconditionally at the top of the iteration -- so on a level that
        # visits nothing, ours burned a draw theirs does not and ran a
        # reduce theirs skips. The reduce is inert (same numbers, one extra
        # launch); the DRAW IS NOT, because it advances a stream that every
        # later level reads. Corrected 2026-08-22 from an audit against
        # their file.
        # (`visit` was selected above)
        if len(visit) > 0:
            # ============================ DEVIATION 1901 ============================
            # IDENTICAL keeps the pinned reduction, now caching unchanged
            # leaves. MOJOLEARN_GBDT_FULL_PARTITION_STATS restores the old
            # DEVIATION 352 sweep, which reads EVERY leaf on EVERY level -- O(max_leaves x
            # n_rows x stat_count) per tree, and Lossguide runs
            # `max_leaves - 1` sequential levels, so at 1M rows x 64 leaves
            # x 2 stats that is ~128M row-stat reads per tree of pure
            # recomputation (recon_lightgbm_cuda.md, mechanism e2).
            # LightGBM's learner never runs it: the child sums come off the
            # split record (`cuda_data_partition.cu:798-903`), O(1) per
            # split.
            #
            # The FAST arm keeps ONE sweep, at the first iteration, to seed
            # the root's row -- every later entry is written by
            # `update_partition_stats_from_split_kernel` in the split chain
            # below, from the parent's entry and the winner's scanned
            # histogram cell. Propagated sums re-associate against the
            # fresh reduction (the histogram-subtraction tradeoff, applied
            # to the partition stats), which is why IDENTICAL keeps the
            # sweep byte for byte. The END-OF-TREE sweep further down stays
            # in BOTH modes: leaf values always come from the exact
            # reduction.
            # =======================================================================
            # (`run_part_sweep` and the ALL set were staged above)
            if run_part_sweep:
                # their `AllReduceThroughMaster(subsets->CurrentPartStats(),
                # ...)` (`:443`) over leaves `[0, leafCount)`. See DEVIATION
                # 352.
                stage_times.begin(ctx)
                # DEVIATION 261: its own staging pair. Preserve each leaf's
                # x-stripe and reduction; only the list/grid.y gets smaller.
                if reduce_count > 0:
                    if use_ridx:
                        # the stat plane is stationary (DEVIATION 1902);
                        # the FAST arm runs this sweep only at iteration 1,
                        # where the index is the identity, the IDENTICAL
                        # arm (Apple) at every iteration
                        compute_partition_stats_gather(
                            ctx, reduce_count, n_rows, stat_count, n_rows,
                            d_all_ids, p_off, p_sz, stats, row_index,
                            stat_partials, part_stats, sm_count=sm_count,
                        )
                    else:
                        compute_partition_stats(
                            ctx, reduce_count, n_rows, stat_count, n_rows,
                            d_all_ids, p_off, p_sz, stats, stat_partials,
                            part_stats, sm_count=sm_count,
                        )
                    mgr.stream_kernel()
                stage_times.end(ctx, "partstats")
            trace.record_device(
                ctx, d_tag + "partstats", part_stats,
                len(leaves) * stat_count,
            )

            # `Random.NextUniformL()`, ONE DRAW PER LAUNCH and not per
            # iteration (`:469`, `:489`, `:510`).
            var level_seed = level_rand.next_uniform_l()

            # `numScoreBlocks = leavesToVisit.size()` (`:428-432`), and the
            # kernel is `TComputeOptimalSplitsLeafwiseKernel` (`:470-488`).
            stage_times.begin(ctx)
            # (the VISIT list was staged above)

            # ==================== POLICY BRANCH 2 OF 4 ====================
            # `greedy_search_helper.cpp:465-533`. The three policies take
            # three different kernels off ONE `numScoreBlocks`:
            #
            #   SymmetricTree   TComputeOptimalSplitsKernel        (:470)
            #   Depthwise       TComputeOptimalSplitsLeafwiseKernel(:490)
            #   Lossguide       TComputeOptimalSplitLeafwiseKernel (:513)
            #
            # and the last of those is guarded by
            # `CB_ENSURE(leavesToVisit.size() <= 2)` (`:511`), which is not
            # an assumption but a CONSEQUENCE of the Lossguide selection:
            # one split makes exactly two leaves without a `BestSplit`, and
            # `SelectLeavesToVisit` returns exactly the leaves that lack
            # one. Implemented as a raise, because the state that breaks it --
            # a leaf left undefined by a poison record -- is reachable and
            # is recorded in `checks/lossguide_policy_check.mojo` P6.
            #
            # THE RECORD LAYOUT IS THE SAME on both arms: block (x, y)
            # writes `x + y * gridDim.x`, so the host reduce below is
            # policy-independent and is NOT branched.
            if (
                lossguide
                and len(visit) > 2
                and GBDT_LG_BATCH == 1
                and not lg_exact
            ):
                raise Error(
                    String("Lossguide scored ")
                    + String(len(visit))
                    + " leaves; their CB_ENSURE allows at most 2"
                    + " (greedy_search_helper.cpp:511). A leaf left"
                    + " undefined by a poison record is the state that"
                    + " does this."
                )
            if lossguide and len(visit) <= 2:
                # their two scalars, and `numBlocks.y = partId ==
                # maybeSecondPartId ? 1 : 2` (`:570`) -- so a single-leaf
                # iteration passes the SAME id twice and launches one row.
                # Passing two equal ids with a two-row grid would score one
                # leaf twice and hand the reduce a duplicate.
                var first = Int32(visit[0])
                var second = Int32(visit[1]) if len(visit) == 2 else first
                var rows = 2 if len(visit) == 2 else 1
                if (
                    options.score_function == SCORE_FUNCTION_L2
                    or options.score_function == SCORE_FUNCTION_NEWTON_L2
                ):
                    ctx.enqueue_function[
                        compute_optimal_split_kernel[SCORE_FUNCTION_L2]
                    ](
                        skip.unsafe_ptr(),
                        Int32(hist_cells_per_leaf),
                        bff.unsafe_ptr(),
                        ffw.unsafe_ptr(),
                        hist.unsafe_ptr(),
                        part_stats.unsafe_ptr(),
                        Int32(stat_count),
                        first,
                        second,
                        Int32(1) if multiclass_optimization else Int32(0),
                        options.l2_reg,
                        Float32(0.0),
                        level_seed,
                        region_score.unsafe_ptr(),
                        region_bin.unsafe_ptr(),
                        min_child_hessian,
                        grid_dim=(argmax_blocks, rows, 1),
                        block_dim=(LEAFWISE_SCORE_BLOCK_SIZE, 1, 1),
                    )
                else:
                    ctx.enqueue_function[
                        compute_optimal_split_kernel[SCORE_FUNCTION_COSINE]
                    ](
                        skip.unsafe_ptr(),
                        Int32(hist_cells_per_leaf),
                        bff.unsafe_ptr(),
                        ffw.unsafe_ptr(),
                        hist.unsafe_ptr(),
                        part_stats.unsafe_ptr(),
                        Int32(stat_count),
                        first,
                        second,
                        Int32(1) if multiclass_optimization else Int32(0),
                        options.l2_reg,
                        score_std_dev,
                        level_seed,
                        region_score.unsafe_ptr(),
                        region_bin.unsafe_ptr(),
                        min_child_hessian,
                        grid_dim=(argmax_blocks, rows, 1),
                        block_dim=(LEAFWISE_SCORE_BLOCK_SIZE, 1, 1),
                    )
            elif (
                options.score_function == SCORE_FUNCTION_L2
                or options.score_function == SCORE_FUNCTION_NEWTON_L2
            ):
                ctx.enqueue_function[
                    compute_optimal_splits_region_kernel[SCORE_FUNCTION_L2]
                ](
                    skip.unsafe_ptr(),
                    Int32(hist_cells_per_leaf),
                    bff.unsafe_ptr(),
                    ffw.unsafe_ptr(),
                    hist.unsafe_ptr(),
                    part_stats.unsafe_ptr(),
                    Int32(stat_count),
                    d_visit.unsafe_ptr(),
                    Int32(1) if multiclass_optimization else Int32(0),
                    options.l2_reg,
                    # the L2 calcer has no noise term at all
                    # (`score_calcers.cuh:40-69`); the seed is still handed
                    # over so both arms consume the same stream.
                    Float32(0.0),
                    level_seed,
                    region_score.unsafe_ptr(),
                    region_bin.unsafe_ptr(),
                    min_child_hessian,
                    grid_dim=(argmax_blocks, len(visit), 1),
                    block_dim=(LEAFWISE_SCORE_BLOCK_SIZE, 1, 1),
                )
            else:
                ctx.enqueue_function[
                    compute_optimal_splits_region_kernel[
                        SCORE_FUNCTION_COSINE
                    ]
                ](
                    skip.unsafe_ptr(),
                    Int32(hist_cells_per_leaf),
                    bff.unsafe_ptr(),
                    ffw.unsafe_ptr(),
                    hist.unsafe_ptr(),
                    part_stats.unsafe_ptr(),
                    Int32(stat_count),
                    d_visit.unsafe_ptr(),
                    Int32(1) if multiclass_optimization else Int32(0),
                    options.l2_reg,
                    score_std_dev,
                    level_seed,
                    region_score.unsafe_ptr(),
                    region_bin.unsafe_ptr(),
                    min_child_hessian,
                    grid_dim=(argmax_blocks, len(visit), 1),
                    block_dim=(LEAFWISE_SCORE_BLOCK_SIZE, 1, 1),
                )
            mgr.stream_kernel()
            stage_times.end(ctx, "score.kernel")

            # ===== HOST WAIT ONE OF TWO: their `bestProps.Read(propsCpu)`
            # (`greedy_search_helper.cpp:517`). =====
            # lane cpu3-gbdt-a: the IDENTICAL host fold (their cross-block
            # reduce, `:520-531`, read back as `2 * argmax_blocks` records
            # per leaf and folded on the host under
            # `best_split_properties_less`) is gone from the GPU route.
            # DEVIATION 1904's device fold below is that loop VERBATIM
            # (same block order, poison skip, `ToSplit` clamp, tie rule;
            # `kernel/split_resolve.mojo`), so every mode now takes it and
            # no bit moves.
            # ============ DEVIATION 1904 (wired) ============
            # The fold moved onto the device: one block per scored leaf
            # runs the IDENTICAL arm's host reduce VERBATIM -- same sequential
            # block order, same poison skip, same ToSplit clamp, same
            # `best_split_properties_less` tie rule, incumbent keeps a
            # full tie -- so the winner records are bit-for-bit the host
            # fold's output (`kernel/split_resolve.mojo`, DEVIATION 1904
            # block). The ONE wait of this stage then brings home
            # `WINNER_RECORD_WORDS` words per leaf instead of
            # `2 * argmax_blocks` values per leaf, and the host loop
            # that follows only UNPACKS -- it resolves and compares
            # nothing. Every mode takes it (lane cpu3-gbdt-a).
            stage_times.begin(ctx)
            # the winner records this level unpacks: the per-level
            # readback (`h_winner`)
            var wrec = h_winner.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
            ctx.enqueue_function[leaf_winner_fold_kernel](
                region_score.unsafe_ptr(),
                region_bin.unsafe_ptr(),
                Int32(argmax_blocks),
                Int32(hist_cells_per_leaf),
                d_bf_feature.unsafe_ptr(),
                d_bf_bin.unsafe_ptr(),
                d_bf_one_hot.unsafe_ptr(),
                d_bf_folds.unsafe_ptr(),
                d_winner.unsafe_ptr(),
                grid_dim=(len(visit), 1, 1),
                block_dim=(WINNER_FOLD_BLOCK_SIZE, 1, 1),
            )
            mgr.stream_kernel()
            if no_sync_tree:
                # DW_NO_LEVEL_SYNC: selection, payload and the fused
                # chain go behind the fold with no wait; the winners,
                # the new sizes and the split count come home in the
                # level's ONE wait (HOST WAIT TWO is skipped below).
                level_synced = True
                ctx.enqueue_function[dw_select_splits_kernel](
                    d_winner.unsafe_ptr(),
                    d_visit.unsafe_ptr(),
                    Int32(len(visit)),
                    Int32(len(leaves)),
                    d_feat_table.unsafe_ptr(),
                    d_left.unsafe_ptr(),
                    d_right.unsafe_ptr(),
                    sp_feats.unsafe_ptr(),
                    sp_bins.unsafe_ptr(),
                    d_win_cells.unsafe_ptr(),
                    d_nsplit.unsafe_ptr(),
                    grid_dim=(
                        (len(visit) + DW_SELECT_BLOCK - 1)
                        // DW_SELECT_BLOCK,
                        1,
                        1,
                    ),
                    block_dim=(DW_SELECT_BLOCK, 1, 1),
                )
                mgr.stream_kernel()
                _launch_fused_split_chain[True](
                    ctx, len(visit), n_rows, sm_count, stat_count,
                    hist_cells_per_leaf, cindex, row_index, new_index,
                    p_off, p_sz, d_left, d_right, d_win_cells, sp_feats,
                    sp_bins, flags, chunk_zeros, chunk_offsets,
                    leaf_zeros, hp_off, hp_sz, hist, part_stats,
                    _dw_dev_u32(d_nsplit, 0),
                )
                for _ in range(4):
                    mgr.stream_kernel()
                ctx.enqueue_copy(
                    dst_ptr=h_winner.unsafe_ptr(), src_buf=d_winner
                )
                ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
                ctx.enqueue_copy(
                    dst_ptr=h_nsplit.unsafe_ptr(), src_buf=d_nsplit
                )
            else:
                ctx.enqueue_copy(
                    dst_ptr=h_winner.unsafe_ptr(), src_buf=d_winner
                )
            # LG_EXACT_ID's node stats ride this wait, as they rode
            # the host fold's
            comptime if LG_EXACT_ID:
                if lg_exact:
                    if len(lg_resident_stats)>0:
                        var snapshot=part_stats.create_sub_buffer[DType.float32](0,max_leaves*stat_count)
                        ctx.enqueue_copy(dst_buf=lg_resident_stats[1],src_buf=snapshot)
                        _ = snapshot^
                        lg_resident_snapshots+=1
                    else:
                        ctx.enqueue_copy(
                            dst_ptr=h_part_stats.unsafe_ptr(),
                            src_buf=part_stats,
                        )
            mgr.wait_complete()
            stage_times.end(ctx, "score.read")
            # the identity ladder's records are UNCHANGED: the
            # per-block score records still sit in the device
            # buffers, so a traced FAST run digests the same bytes
            # the host-fold path digested
            trace.record_device(
                ctx, d_tag + "scores.gain", region_score,
                argmax_blocks * len(visit),
            )
            trace.record_device(
                ctx, d_tag + "scores.bin", region_bin,
                argmax_blocks * len(visit),
            )
            trace.record_list_i32(d_tag + "visit", _as_i32(visit))

            # the host fold's OUTPUT, reconstructed: DEFINED rebuilds
            # the stored `TBestSplitProperties(f, bin, gain, gain)`
            # (the fold kept one number in Score and Gain); UNDEFINED
            # keeps the default record; BIN_OUT_OF_RANGE raises the
            # host fold's own diagnostic, with the leaves before it
            # updated exactly as the host loop would have left them.
            stage_times.begin(ctx)
            for i in range(len(visit)):  # small-loop(visit: leaves scored this level, at most max_leaves): unpack device winner records, no folding
                var rec = i * WINNER_RECORD_WORDS
                var status = wrec.unsafe_load(rec + 3)
                if status == WINNER_STATUS_BIN_OUT_OF_RANGE:
                    raise Error(
                        String("score kernel returned bin-feature ")
                        + String(Int(
                            wrec.unsafe_load(rec + 1)
                        ))
                        + " outside the histogram's "
                        + String(hist_cells_per_leaf)
                        + " cells"
                    )
                var best = TBestSplitProperties()
                var best_cell = Int32(-1)
                if status == WINNER_STATUS_DEFINED:
                    var w_feat = wrec.unsafe_load(rec)
                    var w_bin = wrec.unsafe_load(
                        rec + 1
                    )
                    var gain = bitcast[DType.float32](
                        wrec.unsafe_load(rec + 2)
                    )
                    best = TBestSplitProperties(
                        w_feat.cast[DType.int32](),
                        w_bin.cast[DType.int32](),
                        gain,
                        gain,
                    )
                    best_cell = wrec.unsafe_load(
                        rec + 4
                    ).cast[DType.int32]()
                leaves[visit[i]].update_best_split(best)
                # DEVIATION 1901: the same one-site store the host
                # fold makes, from record word [4]
                best_cells[visit[i]] = best_cell
            stage_times.end(ctx, "score.hostreduce")

            # Rejected leaves cannot become eligible later in this tree.
            # In Lossguide, leaving them undefined and nonterminal would
            # revisit them alongside the next two children and violate the
            # scorer's at-most-two-leaves invariant.
            if options.min_child_hessian >= 0:
                for i in range(len(visit)):  # small-loop(visit: leaves scored this level, at most max_leaves): terminal flags for rejected leaves
                    if not leaves[visit[i]].best_split.defined:
                        leaves[visit[i]].is_terminal = True

            comptime if LG_EXACT_BATCH:
                if lg_exact:
                    for i in range(len(visit)):  # small-loop(visit: leaves scored this level, at most max_leaves): replay node state from device winners
                        var vn = lg_leaf_node[visit[i]]
                        if leaves[visit[i]].best_split.defined:
                            lg_node_state[vn] = LG_NODE_DEFINED
                            lg_node_gain[vn] = leaves[visit[i]].best_split.gain
                        else:
                            lg_node_state[vn] = LG_NODE_NO_SPLIT

            # THE WINNERS, which is the last host state before the split
            # chain. A divergence that first appears here and not in
            # `scores.gain` is in the HOST REDUCE -- the sequential fold
            # under `best_split_properties_less` -- and not on the device.
            if emit_digests:
                var wf = List[Int32]()
                var wb = List[Int32]()
                var wg = List[Float32]()
                for i in range(len(visit)):  # small-loop(visit: leaves scored this level, at most max_leaves): digest record lists, trace only
                    ref bs = leaves[visit[i]].best_split
                    wf.append(bs.feature_id)
                    wb.append(bs.bin_id)
                    wg.append(bs.gain)
                trace.record_list_i32(d_tag + "best.feature", wf)
                trace.record_list_i32(d_tag + "best.bin", wb)
                trace.record_list_f32(d_tag + "best.gain", wg)

        # ===================== SplitLeaves ==========================
        # `greedy_search_helper.cpp:575`. `HaveFixedSplits` is absent: the
        # option that feeds it is refused by name (see
        # `structure_searcher_options.mojo`).
        # ==================== POLICY BRANCH 3 OF 4 ====================
        # `SelectLeavesToSplit` (`greedy_search_helper.cpp:317-361`).
        # Depthwise and SymmetricTree share one arm -- every leaf whose best
        # split IMPROVES (`Score < 0`, `:355-359`). Lossguide takes the
        # single best leaf and HAS NO SIGN TEST AT ALL (`:319-324`), so it
        # keeps splitting after every remaining split makes the objective
        # worse and is bounded only by MaxLeaves and IsTerminalLeaf.
        var to_split: List[Int]
        if lossguide:
            # the TRACED wrapper (lossguide lane's, 8426d52): records the
            # per-leaf BestSplit queue and the selected leaf on the identity
            # ladder, then delegates the decision untouched. Host records
            # only -- no drain -- so it stays outside the stage timers'
            # concern, and the trace/timer mutual exclusion covers the rest.
            comptime if GBDT_LG_BATCH > 1:
                var room = max_leaves - len(leaves)
                to_split = _lossguide_top_b(
                    leaves, GBDT_LG_BATCH if GBDT_LG_BATCH < room else room
                )
            else:
                to_split = List[Int]()
                var lg_done = False
                comptime if LG_EXACT_BATCH:
                    if lg_exact:
                        lg_done = True
                        # how many leaves this round may split: the width, the free
                        # slots, and (capacity below the depth bound) the slots the
                        # certain splits still to come may need
                        var lg_limit = LG_EXACT_BATCH_WIDTH
                        if max_leaves - len(leaves) < lg_limit:
                            lg_limit = max_leaves - len(leaves)
                        var lg_plan = _lg_exact_plan(
                            lg_node_left, lg_node_right, lg_node_gain, lg_node_state,
                            options.max_leaves, options.min_split_gain, 1,
                        )
                        if lg_plan.blocked:
                            raise Error(
                                "Lossguide exact batch: a leaf has no score at the"
                                " selection (every nonterminal leaf is scored"
                                " before it)"
                            )
                        if lg_plan.complete:
                            lg_final = lg_plan.final_nodes.copy()
                        else:
                            if lg_room_bound:
                                var lg_safe = (
                                    max_leaves + 2 - options.max_leaves
                                    - len(leaves) + lg_plan.exact_picks
                                )
                                if lg_safe < lg_limit:
                                    lg_limit = lg_safe
                            if lg_limit < 1:
                                raise Error(
                                    String("Lossguide exact batch: no slot for a")
                                    + " certain split; leaves="
                                    + String(len(leaves))
                                    + " capacity="
                                    + String(max_leaves)
                                )
                            if lg_limit > 1:
                                lg_plan = _lg_exact_plan(
                                    lg_node_left, lg_node_right, lg_node_gain,
                                    lg_node_state, options.max_leaves,
                                    options.min_split_gain, lg_limit,
                                )
                            for i in range(len(lg_plan.expand)):  # small-loop(expand: leaves the replay expands, at most max_leaves): split id plan list
                                to_split.append(lg_node_leaf[lg_plan.expand[i]])
                            # the multi-leaf MakeSplit numbers right children by
                            # position: ascending ids, as `_lossguide_top_b`
                            sort(to_split)
                if not lg_done:
                    to_split = lossguide_select_leaves_to_split_traced(
                        leaves, trace, d_tag
                    )
        else:
            to_split = select_leaves_to_split(leaves)

        # Opt-in split-gain threshold. Default -1 preserves each policy's
        # original selection, including Lossguide's non-improving splits.
        # Stored Gain is the negated improvement; equality does not split.
        if options.min_split_gain >= Float64(0):
            var accepted = List[Int]()
            for ts in range(len(to_split)):  # small-loop(to_split: leaves chosen this round, at most max_leaves): min_split_gain accept gate per leaf
                var leaf_id = to_split[ts]
                if Float64(-leaves[leaf_id].best_split.gain) > options.min_split_gain:
                    accepted.append(leaf_id)
            to_split = accepted^
            trace.record_list_i32(d_tag + "split.accepted", _as_i32(to_split))

        if level_synced:
            # DW_NO_LEVEL_SYNC: the device selected from the same records
            # the host just replayed; a count that differs is a bookkeeping
            # break, not a data condition
            var dev_n_split = Int(h_nsplit.unsafe_ptr().unsafe_load(0))
            if dev_n_split != len(to_split):
                raise Error(
                    String("DW_NO_LEVEL_SYNC: device selected ")
                    + String(dev_n_split)
                    + " splits, host selection "
                    + String(len(to_split))
                )
        if len(to_split) > 0:
            # --- MakeSplit's multi-leaf arm, `split_properties_helper
            # .cpp:845-950`. `leftId = leavesToSplit[i]` keeps the parent's
            # partition slot and `rightId = leavesCount + i` is fresh.
            stage_times.begin(ctx)
            var leaves_count = len(leaves)
            var sp_bytes = sp_feats_h.unsafe_ptr()
            # ============ THEIR REORDER DISPATCH NUMBER, RESTORED 2026-08-22 ==
            # `TSplitPointsKernel::Run` picks its arm on the LARGEST LEAF
            # BEING SPLIT -- `maxLeafSize = Max(partitionsCpuPtr[
            # cpuLeafIdsPtr[leaf]].Size, ...)` over exactly `leavesToSplit`
            # (`split_points.cpp:60-63`), fast one-launch GatherInplace when
            # `maxLeafSize <= 1024` (`:65`, `:113`). This driver passed
            # `n_rows`, so past 1024 total rows the fast arm was UNREACHABLE
            # for both non-symmetric policies -- found by the lossguide lane
            # (their 2eb2cfb), verified here against their source. The
            # symmetric driver passes its own `max_live_rows` and was never
            # wrong.
            #
            # THE MAX IS TAKEN FROM THE PARENT SNAPSHOTS INSIDE THE LOOP, not
            # from `leaves[to_split[i]].size` at the call site: by then the
            # split slot holds the LEFT CHILD, whose `size` is 0 by their own
            # `SplitLeaf` (`newLeaf.Size = 0`, `:790`) -- a call-site max
            # would be 0, always fast arm, and the inplace kernel run past
            # its shared-memory bound. Their read works because it reads the
            # PARTITION mirror, which still holds the parents' sizes; the
            # parent snapshot is this host loop's copy of the same number,
            # current since HOST WAIT TWO of the previous iteration (root:
            # set at CreateInitialSubsets).
            # ===================================================================
            var max_split_rows = 0
            for i in range(len(to_split)):  # small-loop(to_split: leaves split this round, at most max_leaves): split descriptor and id plan lists
                var left_id = to_split[i]
                var right_id = leaves_count + i
                var bs = leaves[left_id].best_split
                if not bs.defined:
                    raise Error(
                        String("Best split is undefined for leaf ")
                        + String(left_id)
                    )
                var split = TBinarySplit(
                    bs.feature_id,
                    bs.bin_id,
                    Int32(
                        BIN_SPLIT_TAKE_BIN
                    ) if layout.features[
                        Int(bs.feature_id)
                    ].one_hot_feature else Int32(BIN_SPLIT_TAKE_GREATER),
                )

                # their `splitsFeaturesBuilder.Add(DataSet.GetTCFeature(
                # splitFeature.FeatureId))` (`:875`) and
                # `splitBins.push_back(splitFeature.BinIdx)` (`:876`).
                # `CFeature` is one struct in the kernel, so the array is
                # raw bytes; see `make_split_features_buffers`.
                # ============ THE OFFSET IS IN ELEMENTS, NOT COLUMNS ============
                # `DataSet.GetTCFeature(featureId)` (`split_properties_helper
                # .cpp:875`) hands the split kernel a `TCFeature` whose
                # `Offset` is what `compressedIndex + feature.Offset +
                # loadIndex` indexes with (`split_points.cu:518`). In THIS
                # implementation's layout that is `column * n_rows`, and
                # `CompressedIndexLayout.features[f].offset` is the bare
                # COLUMN. The symmetric arm multiplies at the point it builds
                # its resolve table (`TTreeWorkspace`'s `bfr_off`,
                # `tf2.offset * UInt32(n_rows)`), and this is the same
                # multiply at this arm's equivalent point.
                #
                # PASSING THE BARE COLUMN DOES NOT CRASH AND DOES NOT LOOK
                # WRONG. It reads `cindex[column + row]` instead of
                # `cindex[column * n_rows + row]`, which is a real bin of a
                # real feature for almost every row, so the tree still grows,
                # still conserves rows, and still picks different features in
                # different leaves. The first thing that saw it was
                # `depthwise_check` claim 4 -- the apply kernel, which
                # computes the offset the other way -- at 852 rows in a bin
                # growth had put 243 in. A gate that only looked at the
                # partition would have called this green.
                var f = layout.features[Int(bs.feature_id)]
                var packed = CFeature(
                    f.offset * UInt32(n_rows),
                    f.mask,
                    f.shift,
                    f.first_fold_index,
                    f.folds,
                    f.one_hot_feature,
                )
                # `CFEATURE_BYTES` is `size_of[CFeature]()`, so indexing the
                # bitcast pointer by the slot IS their `splitsFeaturesBuilder`
                # writing element `i` of a `TCFeature` array; the byte buffer
                # exists only because Mojo device buffers are DType-shaped.
                var dst = sp_bytes.bitcast[CFeature]()
                dst[unsafe_offset=i] = packed
                sp_bins_h.unsafe_ptr().unsafe_store(i, UInt32(bs.bin_id))
                h_left.unsafe_ptr().unsafe_store(i, UInt32(left_id))
                h_right.unsafe_ptr().unsafe_store(i, UInt32(right_id))
                # DEVIATION 1901: the winner's cell rides with the split
                # payload. A defined record without a stored cell is a
                # bookkeeping break, not a data condition -- the two are
                # written at one site -- so it raises rather than
                # propagating from a wrong address.
                comptime if not SPLIT_COST_IDENTICAL:
                    if best_cells[left_id] < Int32(0):
                        raise Error(
                            String("leaf ")
                            + String(left_id)
                            + " has a defined best split but no stored"
                            " winning cell (DEVIATION 1901 bookkeeping)"
                        )
                    h_win_cells.unsafe_ptr().unsafe_store(
                        i, UInt32(Int(best_cells[left_id]))
                    )

                # `TLeaf leaf = subsets->Leaves[leftId];` -- the SNAPSHOT.
                # Both children are derived from the parent, so the parent
                # must be read before the left slot is overwritten.
                var parent = leaves[left_id].copy()
                if parent.size > max_split_rows:
                    max_split_rows = parent.size
                var left = split_leaf(parent, split, SPLIT_VALUE_ZERO)
                var right = split_leaf(parent, split, SPLIT_VALUE_ONE)
                leaves[left_id] = left^
                leaves.append(right^)
                comptime if INCREMENTAL_PART_STATS:
                    part_stats_dirty[left_id] = True
                    # Negative-control build proves that both children must
                    # invalidate the cache; never set in a shipping build.
                    part_stats_dirty.append(
                        not is_defined["MOJOLEARN_GBDT_SAB_SKIP_RIGHT_PART_STATS"]()
                    )
                # the sibling key: both children's parent is the id the
                # left child kept.
                parent_of[left_id] = left_id
                parent_of.append(left_id)
                # DEVIATION 1901: children start with no stored cell,
                # exactly as `SplitLeaf` resets their `BestSplit`.
                best_cells[left_id] = Int32(-1)
                best_cells.append(Int32(-1))
                # DEVIATION 1903: the right child's slot has never been
                # written -- fresh id, once-per-tree memset -- so it starts
                # clean. The left slot keeps its parent's True.
                hist_slot_dirty.append(False)
                comptime if LG_EXACT_BATCH:
                    if lg_exact:
                        var pn = lg_leaf_node[left_id]
                        lg_node_path[pn] = parent.path.copy()
                        comptime if LG_EXACT_ID:
                            if len(lg_resident_stats)>0:
                                var current=lg_resident_stats[1].create_sub_buffer[DType.float32](left_id*stat_count,stat_count)
                                var kept=lg_resident_stats[0].create_sub_buffer[DType.float32](pn*stat_count,stat_count)
                                ctx.enqueue_copy(dst_buf=kept,src_buf=current)
                                _ = current^; _ = kept^
                            else:
                                for st in range(stat_count):  # small-loop(stat_count: stat planes, 1 plus classes): copy of the split node's stats
                                    lg_node_stats[pn * stat_count + st] = (
                                        h_part_stats.unsafe_ptr().unsafe_load(
                                            left_id * stat_count + st
                                        )
                                    )
                        var cn = len(lg_node_leaf)
                        lg_node_left[pn] = cn
                        lg_node_right[pn] = cn + 1
                        lg_node_leaf.append(left_id)
                        lg_node_leaf.append(right_id)
                        for _ in range(2):
                            for _ in range(stat_count):  # small-loop(stat_count: stat planes, 1 plus classes): child node stats placeholders
                                lg_node_stats.append(Float32(0.0))
                            lg_node_left.append(-1)
                            lg_node_right.append(-1)
                            lg_node_gain.append(Float32.MAX)
                            lg_node_state.append(LG_NODE_UNKNOWN)
                            lg_node_path.append(TLeafPath())
                        lg_leaf_node[left_id] = cn
                        lg_leaf_node.append(cn + 1)

            var n_split = len(to_split)
            # the split bins, the split pair (and, FAST, DEVIATION 1901's
            # winning cells) in one arena copy (ID_UPLOAD_COALESCE)
            var split_slots = List[Int]()
            for fs in range(IDS_FEAT_SLOTS):
                split_slots.append(IDS_SLOT_SP_FEATS + fs)
            split_slots.append(IDS_SLOT_SP_BINS)
            split_slots.append(IDS_SLOT_LEFT)
            split_slots.append(IDS_SLOT_RIGHT)
            comptime if not SPLIT_COST_IDENTICAL:
                split_slots.append(IDS_SLOT_WIN)
            if not level_synced:
                _upload_id_slots(
                    ctx, d_ids_arena, h_ids_arena_p, ids_host, max_leaves,
                    split_slots^,
                )
            stage_times.end(ctx, "split.host")
            var fused_chain = False
            comptime if DW_FUSED_CHAIN:
                # Depthwise only: Lossguide keeps the eight-launch chain
                fused_chain = use_ridx and not lossguide
            if level_synced:
                # DW_NO_LEVEL_SYNC: the chain already ran behind the fold
                pass
            elif fused_chain:
                # DW_FUSED_CHAIN: the eight-launch chain below in four, same
                # permutation, partitions and stats (`kernel/split_chain_fused.mojo`)
                stage_times.begin(ctx)
                _launch_fused_split_chain(
                    ctx, n_split, n_rows, sm_count, stat_count,
                    hist_cells_per_leaf, cindex, row_index, new_index, p_off, p_sz,
                    d_left, d_right, d_win_cells, sp_feats, sp_bins, flags,
                    chunk_zeros, chunk_offsets, leaf_zeros, hp_off, hp_sz, hist,
                    part_stats, _dw_dev_u32(d_nsplit, 0),
                )
                for _ in range(4):
                    mgr.stream_kernel()
                stage_times.end(ctx, "split.chain.fused")
            else:
                stage_times.begin(ctx)

                # ============================ DEVIATION 1901 ============================
                # Their split's own "Update part stats"
                # (`split_properties_helper.cpp:918`), replaced by LightGBM's
                # O(1)-per-split propagation (`cuda_data_partition.cu:798-903`)
                # -- the full block is on the kernel. Enqueued FIRST in the
                # chain: it reads the parent's `part_stats` row and the
                # parent's scanned histogram, and nothing later in the chain
                # touches either, so the position is a statement of intent, not
                # an ordering need. FAST arm only; IDENTICAL's stats come from
                # the sweep above, byte for byte as before.
                # =======================================================================
                comptime if not SPLIT_COST_IDENTICAL:
                    ctx.enqueue_function[update_partition_stats_from_split_kernel](
                        d_left.unsafe_ptr(),
                        d_right.unsafe_ptr(),
                        d_win_cells.unsafe_ptr(),
                        sp_feats.unsafe_ptr().bitcast[CFeature](),
                        Int32(hist_cells_per_leaf),
                        Int32(stat_count),
                        hist.unsafe_ptr(),
                        part_stats.unsafe_ptr(),
                        grid_dim=(1, n_split, 1),
                        block_dim=(32, 1, 1),
                    )
                    mgr.stream_kernel()
                stage_times.end(ctx, "split.chain.stats")
                stage_times.begin(ctx)

                # their `TSplitPointsKernel`, whose five steps are five calls
                # here (`split_points.cpp:64-136`): flag and sequence, stable
                # partition, segmented gather of the index and every stat
                # column, copy the histogram to the new leaf, update the
                # partitions. IDENTICAL CALLS TO THE SYMMETRIC LANE'S -- only
                # the id arrays differ.
                ctx.enqueue_function[split_and_make_sequence_kernel](
                    cindex.unsafe_ptr(),
                    row_index.unsafe_ptr(),
                    p_off.unsafe_ptr(),
                    p_sz.unsafe_ptr(),
                    d_left.unsafe_ptr(),
                    sp_feats.unsafe_ptr().bitcast[CFeature](),
                    sp_bins.unsafe_ptr(),
                    flags.unsafe_ptr(),
                    seq.unsafe_ptr(),
                    grid_dim=(
                        split_points_grid_x(n_split, sm_count), n_split, 1
                    ),
                    block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
                )
                mgr.stream_kernel()
                stage_times.end(ctx, "split.chain.flags")
                stage_times.begin(ctx)

                launch_stable_partition_routed[SPLIT_COST_IDENTICAL](
                    ctx, n_split, n_rows, d_left, p_off, p_sz, flags,
                    chunk_zeros, chunk_offsets, leaf_zeros, gmap, sflags,
                    sm_count=sm_count,
                )
                mgr.stream_kernel()
                stage_times.end(ctx, "split.chain.partition")
                stage_times.begin(ctx)

                var reorder_launches = 0

                if use_ridx:
                    # DEVIATION 1902: the stat planes are stationary; the
                    # split's gather_map permutes the 4 B/row index alone.
                    reorder_launches = launch_reorder_index_only(
                        ctx, n_split, max_split_rows, d_left, p_off, p_sz,
                        row_index, new_index, gmap, sm_count=sm_count,
                    )
                else:
                    reorder_launches = launch_reorder_in_leaves(
                        ctx, n_split, wide, max_split_rows, stat_count, n_rows,
                        d_left, p_off, p_sz, stats, new_stats, row_index,
                        new_index, gmap, sm_count=sm_count,
                    )
                for _ in range(reorder_launches):
                    mgr.stream_kernel()
                stage_times.end(ctx, "split.chain.reorder")
                stage_times.begin(ctx)

                # their `CopyHistograms(leftLeaves, rightLeaves, ...)`
                # (`split_points.cpp:139-140`) -- the MULTI-leaf call, which is
                # the arm this lane mirrors. `CopyHistogram` singular at `:327`
                # is the single-leaf kernel and belongs to the lossguide lane. The left child kept the parent's
                # slot; this puts the same histogram in the right child's, so
                # both are `PreviousPath` and next level can pair them.
                # WIDTH DISPATCH, same story as the subtraction above.
                #
                # DEVIATION 1903: IDENTICAL arm only. On the FAST arm the copy
                # happens at PLAN time, and only for the pairs whose derived
                # sibling is the right child -- see the block above the zero
                # pass. Same kernels, same bytes, fewer launches.
                comptime if not DEFER_HIST_COPY_1903:
                    if (hist_cells_per_leaf * stat_count) % 4 == 0:
                        ctx.enqueue_function[copy_histograms_vec4_kernel](
                            d_left.unsafe_ptr(),
                            d_right.unsafe_ptr(),
                            Int32(stat_count),
                            Int32(hist_cells_per_leaf),
                            hist.unsafe_ptr(),
                            grid_dim=(
                                (hist_cells_per_leaf * stat_count // 4 + 255)
                                // 256,
                                n_split,
                                1,
                            ),
                            block_dim=(256, 1, 1),
                        )
                    else:
                        ctx.enqueue_function[copy_histograms_kernel](
                            d_left.unsafe_ptr(),
                            d_right.unsafe_ptr(),
                            Int32(stat_count),
                            Int32(hist_cells_per_leaf),
                            hist.unsafe_ptr(),
                            grid_dim=(
                                (hist_cells_per_leaf * stat_count + 255) // 256,
                                n_split,
                                1,
                            ),
                            block_dim=(256, 1, 1),
                        )
                    mgr.stream_kernel()

                ctx.enqueue_function[update_partitions_after_split_kernel](
                    d_left.unsafe_ptr(),
                    d_right.unsafe_ptr(),
                    Int32(n_split),
                    sflags.unsafe_ptr(),
                    p_off.unsafe_ptr(),
                    p_sz.unsafe_ptr(),
                    hp_off.unsafe_ptr(),
                    hp_sz.unsafe_ptr(),
                    grid_dim=(
                        split_points_grid_x(n_split, sm_count), n_split, 1
                    ),
                    block_dim=(512, 1, 1),
                )
                mgr.stream_kernel()
                stage_times.end(ctx, "split.chain")

            # ===== HOST WAIT TWO OF TWO: `RebuildLeavesSizes`
            # (`split_properties_helper.cpp:800-812`). Theirs reads the
            # PINNED mirror with no copy; ours copies, for the reason in
            # `gpu_util/gpu_data/partitions.mojo`'s deviation block. =====
            stage_times.begin(ctx)
            var szp = h_sz.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
            if not level_synced:
                ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
                mgr.wait_complete()
            for i in range(len(leaves)):
                leaves[i].size = Int(szp.unsafe_load(i))
            stage_times.end(ctx, "split.sizes")

            trace.record_list_i32(
                d_tag + "split.count", _one_i32(n_split)
            )
            trace.record_device(
                ctx, d_tag + "split.bins", sp_bins, n_split
            )
            trace.record_device(ctx, d_tag + "split.left", d_left, n_split)
            trace.record_device(
                ctx, d_tag + "split.right", d_right, n_split
            )
            # ============ WHY `sflags` AND `gmap` ARE NOT ON THE LADDER ==========
            # They were, for one run, and they produced a FALSE POSITIVE that
            # is worth recording because the tool is a diagnostic and a
            # diagnostic that cries wolf is worse than none.
            #
            # Both are SCRATCH sized to `n_rows`, and a level writes only the
            # rows inside the leaves it is splitting. Every other row holds
            # whatever an earlier level left there. Hashing the whole plane
            # therefore digests HISTORY, not this stage -- and the history
            # differs harmlessly between two core counts because the chunked
            # partition covers the stale regions differently.
            #
            # MEASURED 2026-08-22: the ladder named `d3.flags` as the first
            # divergence between this device's core count and 108, while
            # `depthwise_check` claim 6 said the two MODELS were bit-identical.
            # Both were right. The tag was pointing at a scratch tail.
            #
            # `row_index`, `stats` and the two partition planes are the
            # complete, live-region-only description of what the split chain
            # did, so nothing is lost by dropping the two scratch planes --
            # and the ladder stops lying.
            trace.record_device(ctx, d_tag + "rowindex", row_index, n_rows)
            if use_ridx and SPLIT_COST_IDENTICAL:
                # the plane the permuting arm would hold, gathered through
                # the index into the unused reorder scratch, so the ladder
                # records the same bytes on every column
                if trace.enabled:
                    ctx.enqueue_function[_stats_through_index_kernel](
                        stats.unsafe_ptr(),
                        row_index.unsafe_ptr(),
                        new_stats.unsafe_ptr(),
                        Int32(n_rows),
                        Int32(stat_count),
                        grid_dim=((stat_count * n_rows + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                trace.record_device(
                    ctx, d_tag + "stats", new_stats, stat_count * n_rows
                )
            else:
                trace.record_device(
                    ctx, d_tag + "stats", stats, stat_count * n_rows
                )
            trace.record_device(
                ctx, d_tag + "parts.off", p_off, len(leaves)
            )
            trace.record_device(
                ctx, d_tag + "parts.size", p_sz, len(leaves)
            )

            # `MarkTerminal(leftIds, ...)` then `MarkTerminal(rightIds, ...)`
            # (`greedy_search_helper.cpp:618-619`), AFTER the sizes are
            # rebuilt -- `IsTerminalLeaf` reads `leaf.Size`.
            for i in range(n_split):  # small-loop(n_split: leaves split this round, at most max_leaves): terminal flags tree-shape decision
                var left_id = to_split[i]
                var right_id = leaves_count + i
                leaves[left_id].is_terminal = is_terminal_leaf(
                    leaves[left_id], options
                )
                leaves[right_id].is_terminal = is_terminal_leaf(
                    leaves[right_id], options
                )
                comptime if LG_EXACT_BATCH:
                    if lg_exact:
                        # a terminal child is never scored, so it is never
                        # a candidate
                        if leaves[left_id].is_terminal:
                            lg_node_state[
                                lg_leaf_node[left_id]
                            ] = LG_NODE_NO_SPLIT
                        if leaves[right_id].is_terminal:
                            lg_node_state[
                                lg_leaf_node[right_id]
                            ] = LG_NODE_NO_SPLIT
        else:
            # `for (i ...) subsets.Leaves[i].IsTerminal = true;` (`:620-622`)
            for i in range(len(leaves)):
                leaves[i].is_terminal = True

        var terminate = False
        comptime if LG_EXACT_BATCH:
            if lg_exact:
                if len(to_split) == 0:
                    # the replay ended inside the known tree (or nothing
                    # passed min_split_gain): `lg_final` is the tree
                    terminate = True
                    if len(lg_final) == 0:
                        raise Error(
                            "Lossguide exact batch: no split and no"
                            " finished replay"
                        )
                else:
                    # best-first may be finished by this round's splits
                    # without their children's scores (the leaf budget
                    # reached, or every new child terminal)
                    var lg_check = _lg_exact_plan(
                        lg_node_left, lg_node_right, lg_node_gain,
                        lg_node_state, options.max_leaves,
                        options.min_split_gain, 0,
                    )
                    if lg_check.complete:
                        lg_final = lg_check.final_nodes.copy()
                        terminate = True
            else:
                terminate = should_terminate(leaves, options)
        else:
            terminate = should_terminate(leaves, options)

        if terminate:
            # ============== the leaf values, `:625-650` ==============
            # `numStats` is their `PartitionStats.SingleObjectSize()`.
            # The partitions moved in the split above, so the stats are
            # recomputed here (DEVIATION 352) before being read.
            stage_times.begin(ctx)
            var psp = h_part_stats.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
            enqueue_leaf_iota(ctx, d_ids, len(leaves), 0)
            # lane cpu3-gbdt-a: the leaf values are the device's
            # (`dw_leaf_values_kernel`, soft-float64, the old host tail's
            # bits). Plain trees run it behind the stats, inside the one
            # wait; the lossguide exact replay runs it on its sums below.
            var lv_dim = stat_count - 1
            # capacity: every slot leaf, or the replay's final leaves (at
            # most `max_leaves`), whichever is larger
            var lv_cap = max(len(leaves), options.max_leaves) + 1
            var lv_cells = lv_cap * max(1, lv_dim)
            var d_leaf_vals = ctx.enqueue_create_buffer[DType.float32](lv_cells)
            var h_leaf_vals = ctx.enqueue_create_host_buffer[DType.float32](
                lv_cells
            )
            var d_sums64 = ctx.enqueue_create_buffer[DType.uint64](
                lv_cap * stat_count
            )
            var lv_l2_bits = bitcast[DType.uint64](Float64(options.l2_reg))
            var lv_eps_bits = bitcast[DType.uint64](Float64(1e-20))
            var lv_mc = Int32(1) if multiclass_optimization else Int32(0)
            var lv_on_sums = False
            comptime if LG_EXACT_BATCH:
                if lg_exact:
                    lv_on_sums = True
            if use_ridx:
                # DEVIATION 1902: phase 1 gathers the stationary plane
                # through the row index; phase 2 and the chunk formula are
                # the shared kernels unchanged.
                compute_partition_stats_gather(
                    ctx, len(leaves), n_rows, stat_count, n_rows,
                    d_ids, p_off, p_sz, stats, row_index,
                    stat_partials, part_stats, sm_count=sm_count,
                )
            else:
                compute_partition_stats(
                    ctx, len(leaves), n_rows, stat_count, n_rows,
                    d_ids, p_off, p_sz, stats, stat_partials, part_stats,
                    sm_count=sm_count,
                )
            mgr.stream_kernel()
            if not lv_on_sums:
                ctx.enqueue_function[dw_leaf_values_kernel](
                    part_stats.unsafe_ptr(), d_sums64.unsafe_ptr(), Int32(0),
                    Int32(len(leaves)), Int32(stat_count),
                    lv_l2_bits, lv_eps_bits, lv_mc,
                    d_leaf_vals.unsafe_ptr(),
                    grid_dim=(
                        (len(leaves) + DW_LEAF_VALUE_BLOCK - 1)
                        // DW_LEAF_VALUE_BLOCK, 1, 1,
                    ),
                    block_dim=(DW_LEAF_VALUE_BLOCK, 1, 1),
                )
                mgr.stream_kernel()
                ctx.enqueue_copy(
                    dst_ptr=h_leaf_vals.unsafe_ptr(), src_buf=d_leaf_vals
                )
            if len(lg_resident_stats)>0:
                # All logical node slots were initialized, and only scored
                # split parents are consumed by the exact replay. This copy
                # rides the existing final leaf-value/statistics wait.
                var kept=lg_resident_stats[0].create_sub_buffer[DType.float32](0,len(lg_node_stats))
                ctx.enqueue_copy(dst_ptr=lg_node_stats.unsafe_ptr(),src_buf=kept)
                _ = kept^
            ctx.enqueue_copy(
                dst_ptr=h_part_stats.unsafe_ptr(), src_buf=part_stats
            )
            mgr.wait_complete()
            stage_times.end(ctx, "leaf.values")

            trace.record_device(
                ctx, tag_prefix + "final.partstats", part_stats,
                len(leaves) * stat_count,
            )

            var num_leaves = len(leaves)
            # LG_EXACT_BATCH: one result per leaf of the best-first tree. A
            # leaf the device split ahead of time is the sum of the leaf
            # slots below it (its rows are their rows), in Float64.
            var lg_sums = List[Float64]()
            # the rows of each result leaf (NS_INHERIT_PARTITION)
            var final_rows = List[Int]()
            # NS_INHERIT_ID_GUARD: a result leaf was split ahead of time and
            # folded back (its rows are its descendants' slots, not ascending)
            var lg_any_folded = False
            comptime if LG_EXACT_BATCH:
                if lg_exact:
                    num_leaves = len(lg_final)
                    for fi in range(num_leaves):
                        var stack = List[Int]()
                        stack.append(lg_final[fi])
                        var base = len(lg_sums)
                        for _ in range(stat_count):
                            lg_sums.append(Float64(0.0))
                        final_rows.append(0)
                        var lg_kept = False
                        comptime if LG_EXACT_ID:
                            # folded back: the stats it was scored with
                            if lg_node_left[lg_final[fi]] >= 0:
                                lg_any_folded = True
                                for st in range(stat_count):
                                    lg_sums[base + st] = Float64(
                                        lg_node_stats[
                                            lg_final[fi] * stat_count + st
                                        ]
                                    )
                                lg_kept = True
                        while len(stack) > 0:
                            var node = stack.pop()
                            if lg_node_left[node] >= 0:
                                stack.append(lg_node_right[node])
                                stack.append(lg_node_left[node])
                            else:
                                var slot = lg_node_leaf[node]
                                final_rows[fi] += leaves[slot].size
                                if lg_kept:
                                    continue
                                for st in range(stat_count):
                                    lg_sums[base + st] += Float64(
                                        psp.unsafe_load(
                                            slot * stat_count + st
                                        )
                                    )
            # lane cpu3-gbdt-a: the lossguide exact replay's per-final-leaf
            # sums (above) go to the device as double bit patterns and the
            # same kernel takes the Newton step on them; plain trees already
            # have their values home from the stats wait.
            if lv_on_sums and num_leaves > 0:
                var h_sums64 = ctx.enqueue_create_host_buffer[DType.uint64](
                    lv_cap * stat_count
                )
                for i in range(len(lg_sums)):
                    h_sums64.unsafe_ptr().unsafe_store(
                        i, bitcast[DType.uint64](lg_sums[i])
                    )
                ctx.enqueue_copy(dst_buf=d_sums64, src_ptr=h_sums64.unsafe_ptr())
                ctx.enqueue_function[dw_leaf_values_kernel](
                    part_stats.unsafe_ptr(), d_sums64.unsafe_ptr(), Int32(1),
                    Int32(num_leaves), Int32(stat_count),
                    lv_l2_bits, lv_eps_bits, lv_mc,
                    d_leaf_vals.unsafe_ptr(),
                    grid_dim=(
                        (num_leaves + DW_LEAF_VALUE_BLOCK - 1)
                        // DW_LEAF_VALUE_BLOCK, 1, 1,
                    ),
                    block_dim=(DW_LEAF_VALUE_BLOCK, 1, 1),
                )
                ctx.enqueue_copy(
                    dst_ptr=h_leaf_vals.unsafe_ptr(), src_buf=d_leaf_vals
                )
                ctx.synchronize()
                _ = h_sums64^  # past the drain (step-33 race class)
            _ = d_sums64^
            _ = d_leaf_vals^
            var hlv = h_leaf_vals.unsafe_ptr()
            # model assembly from the device's results: the weight is the
            # stats word (or the replay's double sum) widened, the values
            # are the kernel's; copies only.
            # (`:651-655` -- multiclass `float += double`, one rounding, is
            # inside the kernel; see its docstring.)
            for leaf_id in range(num_leaves):  # small-loop(num_leaves: final leaves, at most max_leaves): model assembly from device results, copies only
                var w = Float64(
                    psp.unsafe_load(
                        leaf_id * stat_count
                    )
                )
                comptime if LG_EXACT_BATCH:
                    if lg_exact:
                        w = lg_sums[leaf_id * stat_count]
                result_weights.append(w)

                var values = List[Float32]()
                for approx_id in range(lv_dim):  # small-loop(lv_dim: approx dimension, classes): copy device leaf values
                    values.append(hlv.unsafe_load(leaf_id * lv_dim + approx_id))
                result_values.append(values^)
                var lg_path_done = False
                comptime if LG_EXACT_BATCH:
                    if lg_exact:
                        var fnode = lg_final[leaf_id]
                        if lg_node_left[fnode] >= 0:
                            result_paths.append(lg_node_path[fnode].copy())
                        else:
                            result_paths.append(
                                leaves[lg_node_leaf[fnode]].path.copy()
                            )
                        lg_path_done = True
                if not lg_path_done:
                    result_paths.append(leaves[leaf_id].path.copy())
                    final_rows.append(leaves[leaf_id].size)
            _ = h_leaf_vals^  # `hlv` points into it (last-use rule)
            comptime if NS_INHERIT_PARTITION:
                # the result leaves in the model's leaf order, which is
                # their order along the row index; a total that is not
                # n_rows leaves the record unset and the caller rebuilds
                # the partition from the model
                if len(final_rows) == num_leaves:
                    var order = List[Int]()
                    for k in range(num_leaves):  # small-loop(num_leaves: final leaves, at most max_leaves): model leaf order by path
                        var at = len(order)
                        order.append(k)
                        while at > 0 and _path_before(
                            result_paths[k], result_paths[order[at - 1]]
                        ):
                            order[at] = order[at - 1]
                            at -= 1
                        order[at] = k
                    var running = 0
                    for k in range(num_leaves):  # small-loop(num_leaves: final leaves, at most max_leaves): partition record offsets for the next tree
                        inherit_offsets.append(running)
                        inherit_sizes.append(final_rows[order[k]])
                        running += final_rows[order[k]]
                    inherit_ready = running == n_rows
                comptime if NS_INHERIT_ID_GUARD:
                    if lg_any_folded:
                        inherit_ready = False
            break

    # THE MODEL ITSELF, last rung of the ladder. If every stage above
    # agrees and this does not, the divergence is in the HOST model
    # builder -- the path fold, the pre-order flatten, the bin numbering --
    # and not in anything the device did.
    if emit_digests:
        var flat_v = List[Float32]()
        for i in range(len(result_values)):  # small-loop(result_values: final leaves, at most max_leaves): digest flattening, trace only
            for j in range(len(result_values[i])):  # small-loop(result_values: approx dimension per leaf): digest flattening, trace only
                flat_v.append(result_values[i][j])
        var flat_w = List[Float32]()
        for i in range(len(result_weights)):  # small-loop(result_weights: final leaves, at most max_leaves): digest flattening, trace only
            flat_w.append(Float32(result_weights[i]))
        trace.record_list_f32(tag_prefix + "final.leafvalues", flat_v)
        trace.record_list_f32(tag_prefix + "final.leafweights", flat_w)

    # `return BuildTreeLikeModel<TModel>(leaves, leavesWeights, leavesValues)`
    comptime if NS_INHERIT_PARTITION:
        dws[0].final_offsets = inherit_offsets.copy()
        dws[0].final_sizes = inherit_sizes.copy()
        dws[0].final_ready = inherit_ready

    stage_times.begin(ctx)
    var model = build_non_symmetric_tree(
        result_paths, result_weights, result_values
    )
    stage_times.end(ctx, "model.build")
    stage_times.report(
        String("lossguide" if lossguide else "depthwise")
        + " fit: rows="
        + String(n_rows)
        + " leaves="
        + String(len(leaves))
        + " iterations="
        + String(iteration)
    )
    if emit_digests:
        var nodes = List[Int32]()
        for i in range(len(model.model_structure.nodes)):  # small-loop(nodes: model tree nodes, at most 2 x max_leaves): digest flattening, trace only
            ref n = model.model_structure.nodes[i]
            nodes.append(Int32(Int(n.feature_id)))
            nodes.append(Int32(Int(n.bin)))
            nodes.append(Int32(Int(n.left_subtree)))
            nodes.append(Int32(Int(n.right_subtree)))
        trace.record_list_i32(tag_prefix + "model.nodes", nodes)
        trace.record_list_i32(
            tag_prefix + "model.splittypes", model.model_structure.split_types
        )
        trace.record_list_f32(tag_prefix + "model.values", model.leaf_values)
        trace.record_list_i32(
            tag_prefix + "model.leaves", _one_i32(model.bin_count())
        )
    comptime if REPORT_PART_STATS_WORK:
        print("part_stats_work", options.policy, n_rows,
              part_stats_leaves_reduced, part_stats_rows_reduced)
    # Retain measured resident-snapshot diagnostics for the opt-in I17 caller.
    comptime if LG_EXACT_ID and is_defined["MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT"]():
        print("I17 frontier_resident_snapshots",lg_resident_snapshots,"retained_nodes",len(lg_node_stats)//stat_count)
    _ = lg_resident_stats^
    return model^


def fit_depthwise_tree[
    hist2_smem_mode: Int = 0
](
    ctx: DeviceContext,
    n_rows: Int,
    fold_counts: List[Int],
    options: TTreeStructureSearcherOptions,
    mut cindex: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut row_index: DeviceBuffer[DType.uint32],
    weight_magnitude: Float32,
    gradient_magnitude: Float32,
    mut ws: List[TTreeWorkspace],
    mut dws: List[TDepthwiseWorkspace],
    mut trace: IdentityTrace,
    one_hot: List[Bool] = List[Bool](),
    approx_dim: Int = 1,
    multiclass_optimization: Bool = False,
    random_seed: UInt64 = UInt64(0),
    sm_count_override: Int = -1,
    tag_prefix: String = String(""),
) raises -> TNonSymmetricTree:
    """The Depthwise name, kept so nothing that already calls it moves.

    `fit_non_symmetric_tree` is the same function; this wrapper exists
    because `checks/depthwise_check.mojo` and the digest probe were
    written against this name and a rename that moves a gate is a rename
    that costs a review.

    **The rename was not cosmetic and the merge was not either.** Two
    drivers, one per policy, was a structure the reference does not have. CatBoost has ONE
    `TGreedySearchHelper` with `if (Options.Policy == ...)` at four sites
    (`greedy_search_helper.cpp:319`, `:465`, `:355`, `:668`), and
    `CONTRIBUTING.md` (Algorithms and references) says the reference structure wins. So the two
    lanes' drivers became one function with four branches, which is both
    less code and more faithful -- and it removed the surface on which the
    Depthwise arm and the Lossguide arm could drift apart in everything
    they share, which is nearly all of it.

    It also refuses a policy this function does not implement, rather than
    growing a tree under the wrong one: `options.policy` is checked at the
    top of `fit_non_symmetric_tree`.
    """
    if options.policy != GROW_DEPTHWISE:
        raise Error(
            "fit_depthwise_tree is the Depthwise entry point; pass"
            " grow_policy=Depthwise or call fit_non_symmetric_tree"
        )
    return fit_non_symmetric_tree[hist2_smem_mode](
        ctx,
        n_rows,
        fold_counts,
        options,
        cindex,
        stats,
        row_index,
        weight_magnitude,
        gradient_magnitude,
        ws,
        dws,
        trace,
        one_hot,
        approx_dim,
        multiclass_optimization,
        random_seed,
        sm_count_override,
        tag_prefix,
    )
