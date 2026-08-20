"""One complete oblivious level, end to end, on the GPU.

PORT OF `TGreedySearchHelper::ComputeOptimalSplits` followed by `SplitLeaves`
(`catboost/cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp:398`
and `:534`) at CatBoost `54a8143a`.

Ported as a FUNCTION where theirs is a helper class, because we have no
`TPointsSubsets` object yet and the launch sequence is the part worth having
first. That is a deviation of shape, not of behavior: the order of kernels
and the data each one reads is theirs.

It lived in `mojo_only/` for one commit, which was wrong. This is a port of
their file, so it belongs beside the other ports; `mojo_only/` is for things
CatBoost never had to write at all.

**Every kernel it calls is already verified in isolation.** This is the first
code that runs them in ORDER, so a wrong answer here is in the wiring, not in
one of nine unknowns. That was the point of checking them separately.

The order is not free choice, and two of the constraints were discovered
while porting rather than read from their source:

1. `copy` must precede `subtract`, because subtract overwrites `from` in
   place with `from - what` and `from` has to already hold the parent totals.
2. `scan` must precede `score` in its OWN kernel, or the score kernel loses
   its parallel shape: it reads a leaf's cumulative sum at one bin directly,
   and a kernel that re-derived the prefix would have to walk bins in order,
   forcing the bin loop innermost and the leaf loop outward.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.device_attribute import DeviceAttribute

from mojo_only.fixed_point import choose_scale

from gbdt.gpu_lib.gpu_manager import TCudaManager
from gbdt.methods.kernel_add_model_value import (
    add_model_value_kernel,
)
from gbdt.models.oblivious_model import TBinarySplit
from gbdt.methods.greedy_subsets_searcher.split_properties_helper import (
    HISTOGRAMS_CURRENT_PATH,
    HISTOGRAMS_PREVIOUS_PATH,
    HISTOGRAMS_ZEROES,
    LeafRecord,
    build_necessary_histograms,
    non_zero_leaves,
    zero_leaves,
)
from std.sys.info import size_of
from gbdt.gpu_data.gpu_structures import CFeature
from gbdt.methods.greedy_subsets_searcher.kernel.split_resolve import (
    RESOLVE_BLOCK_SIZE,
    resolve_and_pack_kernel,
)

from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_NEWTON_L2,
)
from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    FLOAT32_MAX,
    SCORE_BLOCK_SIZE,
    compute_optimal_splits_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_binary import (
    binary_hist_gather_kernel,
    binary_hist_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    copy_histograms_kernel,
    fixed_to_float_kernel,
    scan_histograms_kernel,
    substract_histograms_kernel,
    write_reduces_histograms_kernel,
    zero_buffer_kernel,
    zero_histogram_kernel,
    zero_histograms_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
)
from gbdt.methods.leaves_estimation.leaves_estimation import (
    LEAF_BLOCK,
    compute_leaf_values_kernel,
)
from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_half_byte import (
    half_byte_hist_gather_kernel,
    half_byte_hist_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    ONE_BYTE_BLOCK_SIZE,
    one_byte_block_size,
    one_byte_hist_gather_kernel,
    one_byte_hist_kernel,
)
from gbdt.gpu_data.feature_blocks import PolicyBlock, blocks_for
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from gbdt.gpu_util.copy import (
    COPY_BLOCK,
    copy_f32_kernel,
    copy_u32_kernel,
)
from gbdt.gpu_util.partitions_reduce import (
    partition_stats_chunks,
    STATS_BLOCK,
    compute_partition_stats,
)
from mojo_only.kernel_matrix import (
    HIST_SMEM_SHARED2_I32,
    TARGET_COLUMN,
    deterministic_flush_for,
)
from mojo_only.numerics import NUMERIC_IDENTICAL
from gbdt.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    BUILD_MODE as HIST_BUILD_MODE,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_points import (
    launch_reorder_in_leaves,
    split_points_grid_x,
    SPLIT_BLOCK_SIZE,
    gather_in_leaves_kernel,
    gather_index_in_leaves_kernel,
    split_and_make_sequence_kernel,
    update_partitions_after_split_kernel,
)
from gbdt.methods.leaves_estimation.leaves_estimation import (
    LEAF_BLOCK,
    compute_leaf_values_kernel,
)
from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_half_byte import (
    half_byte_hist_gather_kernel,
    half_byte_hist_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    ONE_BYTE_BLOCK_SIZE,
    one_byte_block_size,
    one_byte_hist_gather_kernel,
    one_byte_hist_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_base import (
    HIST2_SMEM_MODE,
    hist2_block_size,
    hist2_one_byte_gather_kernel,
    hist2_one_byte_kernel,
)
from gbdt.gpu_data.feature_blocks import PolicyBlock, blocks_for
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from gbdt.gpu_util.copy import (
    COPY_BLOCK,
    copy_f32_kernel,
    copy_u32_kernel,
)
from gbdt.gpu_util.partitions_reduce import (
    STATS_BLOCK,
    compute_partition_stats,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_points import (
    PARTITION_BLOCK,
    launch_stable_partition,
)


@fieldwise_init
struct LevelResult(Copyable, Movable):
    """What one level produced, read back for checking."""

    var chosen_bin_feature: Int
    var score: Float32
    var left_size: Int
    var right_size: Int



# `TCFeature` is one struct in their kernels, not four parallel arrays, so the
# driver has to build an array of it. Mojo device buffers are DType-shaped, so
# the array is allocated as raw bytes and bitcast, which is what
# `size_of[CFeature]()` is for.
comptime CFEATURE_BYTES = size_of[CFeature]()


def make_split_features_buffers(
    ctx: DeviceContext, count: Int
) raises -> Tuple[DeviceBuffer[DType.uint8], HostBuffer[DType.uint8]]:
    """Their `splitFeaturesGpu` (`split_properties_helper.cpp:856`)."""
    var d = ctx.enqueue_create_buffer[DType.uint8](count * CFEATURE_BYTES)
    var h = ctx.enqueue_create_host_buffer[DType.uint8](
        count * CFEATURE_BYTES
    )
    return (d^, h^)


def run_one_level(
    ctx: DeviceContext,
    n_rows: Int,
    n_features: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut row_index: DeviceBuffer[DType.uint32],
    total_weight: Float32,
    total_gradient: Float32,
) raises -> LevelResult:
    """Depth 0: one leaf holding every row, split into two.

    Depth 0 is the honest place to start an end-to-end run because it needs
    no sibling subtraction (there is no sibling) and no `PreviousPath`
    bookkeeping, so it exercises the SEQUENCE without also exercising the
    level planner, which is checked separately.
    """
    var stat_count = 2  # [weight, gradient], their layout
    var n_leaves = 2  # the level's output

    # LAUNCH WIDTH. CatBoost sizes these grids as
    # `(leavesCount > 4 ? 2 : 4) * TArchProps::SMCount()`
    # (`split_points.cu:559`, `:388`, `:214`), i.e. dozens of blocks, and
    # every one of these kernels carries a grid-stride loop so the count is
    # free to choose. The driver originally passed grid_dim=(1,1,1) to all
    # three, which is correct and runs one threadgroup on the whole machine:
    # measured 0.989 ms for the index gather, which moves ONE UInt32 per row
    # and should be bandwidth-trivial. Same error the partition had.
    var wide = (n_rows + 255) // 256
    if wide > 256:
        wide = 256
    if wide < 1:
        wide = 1

    # --- partitions: one leaf, every row --------------------------------
    var p_off = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var hp_off = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var hp_sz = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    var h_off2 = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    var h_sz2 = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    h_off.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_off.unsafe_ptr().unsafe_store(1, UInt32(0))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    h_sz.unsafe_ptr().unsafe_store(1, UInt32(0))
    for i in range(n_leaves):
        h_off2.unsafe_ptr().unsafe_store(i, h_off.unsafe_ptr().unsafe_load(i))
        h_sz2.unsafe_ptr().unsafe_store(i, h_sz.unsafe_ptr().unsafe_load(i))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_off, src_ptr=h_off2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_sz, src_ptr=h_sz2.unsafe_ptr())

    var leaf0 = ctx.enqueue_create_buffer[DType.uint32](1)
    var h_leaf0 = ctx.enqueue_create_host_buffer[DType.uint32](1)
    h_leaf0.unsafe_ptr().unsafe_store(0, UInt32(0))
    ctx.enqueue_copy(dst_buf=leaf0, src_ptr=h_leaf0.unsafe_ptr())

    # --- feature descriptors: binary, one fold each ----------------------
    var folds = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var fold_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_sz = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var hfa = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var hfb = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var hfc = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var hfd = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        hfa.unsafe_ptr().unsafe_store(f, UInt32(1))
        hfb.unsafe_ptr().unsafe_store(f, UInt32(f))
        hfc.unsafe_ptr().unsafe_store(f, UInt32(0))
        hfd.unsafe_ptr().unsafe_store(f, UInt32(n_features))
    ctx.enqueue_copy(dst_buf=folds, src_ptr=hfa.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=fold_off, src_ptr=hfb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_off, src_ptr=hfc.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_sz, src_ptr=hfd.unsafe_ptr())

    # `TBinarizedFeature::OneHotFeature`, which the scan tests before it
    # prefix-sums anything (`histogram_utils.cu:395`). These are plain binary
    # features, so none of them is one-hot.
    var one_hot = ctx.enqueue_create_buffer[DType.uint8](n_features)
    var hoh = ctx.enqueue_create_host_buffer[DType.uint8](n_features)
    for f in range(n_features):
        hoh.unsafe_ptr().unsafe_store(f, UInt8(0))
    ctx.enqueue_copy(dst_buf=one_hot, src_ptr=hoh.unsafe_ptr())

    # --- histogram buffer: [leaf][stat][binFeature] ----------------------
    var hist_cells = n_leaves * stat_count * n_features
    var hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
    # their `FillBuffer(subsets.Histograms, 0.0f)` in `CreateInitialSubsets`
    # (`split_properties_helper.cpp:1061`). The histogram kernels do not
    # write a cell whose accumulator is zero, so an unfilled allocation is
    # read back as whatever the driver handed us.
    ctx.enqueue_memset(hist, Float32(0.0))
    # run_one_level is depth 0 with one block per partition, so the
    # multi-block flush never fires; the scratch satisfies the signature.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](hist_cells)
    var zero_ids = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var hz = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    for i in range(n_leaves):
        hz.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=zero_ids, src_ptr=hz.unsafe_ptr())
    ctx.synchronize()

    # 1. ZERO -------------------------------------------------------------
    # `numBlocks.x = CeilDivide(binFeatureCount, blockSize)` with blockSize
    # 256 (`histogram_utils.cu:236-238`). Grid x of 1 clears only the first
    # 256 bin-features and leaves every one after that holding the previous
    # level's values, because `fixed_to_float_kernel` deliberately does not
    # write a cell whose accumulator is zero.
    ctx.enqueue_function[zero_histograms_kernel](
        zero_ids.unsafe_ptr(),
        Int32(n_features),
        hist.unsafe_ptr(),
        grid_dim=((n_features + 255) // 256, n_leaves, stat_count),
        block_dim=(256, 1, 1),
    )

    # 2. BUILD. grid z is the stat, so both planes come from one launch.
    ctx.enqueue_function[binary_hist_kernel](
        folds.unsafe_ptr(),
        fold_off.unsafe_ptr(),
        grp_off.unsafe_ptr(),
        grp_sz.unsafe_ptr(),
        Int32(n_features),
        cindex.unsafe_ptr(),
        Int32(n_rows),
        Int32(0),
        stats.unsafe_ptr(),
        Int32(n_rows),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        leaf0.unsafe_ptr(),
        hist.unsafe_ptr(),
        acc_scratch.unsafe_ptr(),
        Float32(1.0),
        Int32(n_leaves),
        Int32(stat_count),
        grid_dim=(1, 1, stat_count),
        block_dim=(BLOCK_SIZE, 1, 1),
    )

    # 3. SCAN. Binary features are one fold, so the scan is a no-op here and
    # is issued anyway: the sequence is what is being exercised, and leaving
    # a step out because this shape does not need it is how an ordering bug
    # hides until a wider feature appears.
    # Their `ids` argument: the leaves being scanned.
    var scan_ids = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var scan_ids_h = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    for i in range(n_leaves):
        scan_ids_h.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=scan_ids, src_ptr=scan_ids_h.unsafe_ptr())
    # `numBlocks.x = CeilDivide(fCount * 32, blockSize)`
    # (`histogram_utils.cu:442`), theirs at one WARP per feature. Ours is one
    # THREAD per feature, so the same rule is `ceil(fCount / blockSize)`. The
    # kernel guards with `feature_id >= feature_count` and has no grid-stride
    # loop, so grid x of 1 silently left every feature from 256 up unscanned.
    ctx.enqueue_function[scan_histograms_kernel](
        scan_ids.unsafe_ptr(),
        fold_off.unsafe_ptr(),
        folds.unsafe_ptr(),
        one_hot.unsafe_ptr(),
        Int32(n_features),
        Int32(n_features),
        hist.unsafe_ptr(),
        grid_dim=((n_features + 255) // 256, n_leaves, stat_count),
        block_dim=(256, 1, 1),
    )

    # 4. SCORE ------------------------------------------------------------
    var part_stats = ctx.enqueue_create_buffer[DType.float32](
        n_leaves * stat_count
    )
    var hps = ctx.enqueue_create_host_buffer[DType.float32](
        n_leaves * stat_count
    )
    hps.unsafe_ptr().unsafe_store(0, total_weight)
    hps.unsafe_ptr().unsafe_store(1, total_gradient)
    hps.unsafe_ptr().unsafe_store(2, Float32(0.0))
    hps.unsafe_ptr().unsafe_store(3, Float32(0.0))
    ctx.enqueue_copy(dst_buf=part_stats, src_ptr=hps.unsafe_ptr())

    var skip = ctx.enqueue_create_buffer[DType.uint8](n_features)
    var hsk = ctx.enqueue_create_host_buffer[DType.uint8](n_features)
    for f in range(n_features):
        hsk.unsafe_ptr().unsafe_store(f, UInt8(0))
    ctx.enqueue_copy(dst_buf=skip, src_ptr=hsk.unsafe_ptr())

    var out_score = ctx.enqueue_create_buffer[DType.float32](1)
    var out_bin = ctx.enqueue_create_buffer[DType.uint32](1)
    ctx.synchronize()

    # their `TCBinFeature.FeatureId` and `binFeaturesWeights`
    # (`compute_scores.cu:136`). The weights are all 1.0 because
    # `UpdateFeatureWeights` fills 1.0 and returns early with no CTRs
    # (`update_feature_weights.cpp:14-22`) -- their value, not a stub.
    var bff = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var hbf = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var ffw = ctx.enqueue_create_buffer[DType.float32](n_features)
    var hfw = ctx.enqueue_create_host_buffer[DType.float32](n_features)
    for f in range(n_features):
        hbf.unsafe_ptr().unsafe_store(f, UInt32(f))
    for f in range(n_features):
        hfw.unsafe_ptr().unsafe_store(f, Float32(1.0))
    ctx.enqueue_copy(dst_buf=bff, src_ptr=hbf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ffw, src_ptr=hfw.unsafe_ptr())
    ctx.enqueue_function[
            compute_optimal_splits_kernel[SCORE_FUNCTION_COSINE]
        ](
        skip.unsafe_ptr(),
        Int32(n_features),
        bff.unsafe_ptr(), ffw.unsafe_ptr(),
        hist.unsafe_ptr(),
        part_stats.unsafe_ptr(),
        Int32(stat_count),
        leaf0.unsafe_ptr(),
        Int32(1),
        Float32(1.0),
        out_score.unsafe_ptr(),
        out_bin.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(SCORE_BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()

    var hos = ctx.enqueue_create_host_buffer[DType.float32](1)
    var hob = ctx.enqueue_create_host_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_ptr=hos.unsafe_ptr(), src_buf=out_score)
    ctx.enqueue_copy(dst_ptr=hob.unsafe_ptr(), src_buf=out_bin)
    ctx.synchronize()
    var best = Int(hob.unsafe_ptr().unsafe_load(0))

    # 5. SPLIT FLAGS ------------------------------------------------------
    var built1 = make_split_features_buffers(ctx, 1)
    var sp_feats = built1[0]
    var sp_feats_h = built1[1]
    var sp_bin = ctx.enqueue_create_buffer[DType.uint32](1)
    var a = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var b = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var c = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var d = ctx.enqueue_create_host_buffer[DType.uint8](1)
    var e = ctx.enqueue_create_host_buffer[DType.uint32](1)
    a.unsafe_ptr().unsafe_store(0, UInt32(0))
    b.unsafe_ptr().unsafe_store(0, UInt32(31 - best))  # binary shift
    c.unsafe_ptr().unsafe_store(0, UInt32(1))
    d.unsafe_ptr().unsafe_store(0, UInt8(0))
    e.unsafe_ptr().unsafe_store(0, UInt32(0))
    sp_feats_h.unsafe_ptr().bitcast[CFeature]()[unsafe_offset=0] = (
        CFeature(
            offset=UInt32(0),
            mask=UInt32(1),
            shift=UInt32(31 - best),
            first_fold_index=UInt32(0),
            folds=UInt32(1),
            one_hot_feature=False,
        )
    )
    ctx.enqueue_copy(dst_buf=sp_feats, src_ptr=sp_feats_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=sp_bin, src_ptr=e.unsafe_ptr())

    var flags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var seq = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    ctx.synchronize()

    ctx.enqueue_function[split_and_make_sequence_kernel](
        cindex.unsafe_ptr(),
        row_index.unsafe_ptr(),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        leaf0.unsafe_ptr(),
        sp_feats.unsafe_ptr().bitcast[CFeature](),
        sp_bin.unsafe_ptr(),
        flags.unsafe_ptr(),
        seq.unsafe_ptr(),
        grid_dim=(wide, 1, 1),
        block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
    )

    # 6. STABLE PARTITION, three phases -----------------------------------
    var gmap = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var sflags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var max_chunks = (n_rows + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    var chunk_zeros = ctx.enqueue_create_buffer[DType.uint32](max_chunks)
    var chunk_offsets = ctx.enqueue_create_buffer[DType.uint32](max_chunks)
    var leaf_zeros = ctx.enqueue_create_buffer[DType.uint32](1)
    launch_stable_partition(
        ctx,
        1,
        n_rows,
        leaf0,
        p_off,
        p_sz,
        flags,
        chunk_zeros,
        chunk_offsets,
        leaf_zeros,
        gmap,
        sflags,
    )

    # 7. GATHER the row index through the permutation ---------------------
    var new_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var new_stats = ctx.enqueue_create_buffer[DType.float32](
        stat_count * n_rows
    )
    ctx.enqueue_function[gather_index_in_leaves_kernel](
        leaf0.unsafe_ptr(),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        row_index.unsafe_ptr(),
        gmap.unsafe_ptr(),
        new_index.unsafe_ptr(),
        grid_dim=(wide, 1, 1),
        block_dim=(256, 1, 1),
    )

    # 8. UPDATE PARTITIONS ------------------------------------------------
    var left = ctx.enqueue_create_buffer[DType.uint32](1)
    var right = ctx.enqueue_create_buffer[DType.uint32](1)
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var hr = ctx.enqueue_create_host_buffer[DType.uint32](1)
    hl.unsafe_ptr().unsafe_store(0, UInt32(0))
    hr.unsafe_ptr().unsafe_store(0, UInt32(1))
    ctx.enqueue_copy(dst_buf=left, src_ptr=hl.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=right, src_ptr=hr.unsafe_ptr())

    # `hp_off` / `hp_size` are their `partsCpu` and they are ORDINARY DEVICE
    # BUFFERS, which is a deviation and not a choice. Read the DEVIATION
    # BLOCK in `gbdt/gpu_util/gpu_data/partitions.mojo` before assuming
    # otherwise. The kernel writes them (`split_points.cu:372`, `:379`) and
    # nothing on the host can address them, so the leaf size still comes back
    # through the `enqueue_copy` below.
    ctx.enqueue_function[update_partitions_after_split_kernel](
        left.unsafe_ptr(),
        right.unsafe_ptr(),
        Int32(1),
        sflags.unsafe_ptr(),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        hp_off.unsafe_ptr(),
        hp_sz.unsafe_ptr(),
        grid_dim=(wide, 1, 1),
        block_dim=(512, 1, 1),
    )

    # ONE partition read, which is their count. `RebuildLeavesSizes` reads
    # the parts once per `MakeSplit` (`split_properties_helper.cpp:803`,
    # reached at `:950`), and it is safe there because the read is stream
    # ordered behind the split kernel launched at `:920-934`.
    #
    # ORDERING HERE. Both the `left` / `right` uploads above and this copy
    # are enqueued on the same queue as the kernel between them, so the copy
    # cannot observe a `p_sz` the kernel has not written and the kernel
    # cannot observe an unwritten `left` / `right`. The single drain below is
    # what makes the copy's result readable on the host. The two
    # `ctx.synchronize()` calls that used to sit before and after the kernel
    # ordered nothing that the queue did not already order; they were three
    # drains per level where CatBoost has one.
    var osz = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    ctx.enqueue_copy(dst_ptr=osz.unsafe_ptr(), src_buf=p_sz)
    ctx.synchronize()

    return LevelResult(
        best,
        hos.unsafe_ptr().unsafe_load(0),
        Int(osz.unsafe_ptr().unsafe_load(0)),
        Int(osz.unsafe_ptr().unsafe_load(1)),
    )


def run_tree(
    ctx: DeviceContext,
    n_rows: Int,
    n_features: Int,
    max_depth: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut row_index: DeviceBuffer[DType.uint32],
    weight_magnitude: Float32,
    gradient_magnitude: Float32,
    hist_replicas: Int = 0,  # 0 means ask the matrix
) raises -> List[Int]:
    """`FitImpl`'s loop: grow a whole oblivious tree, level by level.

    PORT OF `structure_searcher_template.h:50-66` driving
    `greedy_search_helper.cpp`'s two steps. Under `SymmetricTree` one
    iteration is one LEVEL and every leaf of the level takes the SAME split
    (`greedy_search_helper.cpp:422-425`, `numScoreBlocks = 1`).

    Returns the leaf sizes at the final level, which is what a caller checks:
    they must sum to `n_rows` at every depth, since a split moves rows
    between leaves and creates none.

    **What this adds over `run_one_level` is the pair bookkeeping**: copy the
    parent into the sibling slot, build only the smaller child, derive the
    larger by subtraction. Those three kernels are individually verified and
    have never run in sequence, which is the point of this function.

    Buffers are allocated ONCE here and reused across levels, unlike
    `run_one_level`, which allocates per call and pays about 0.6 ms for it.
    The level loop below now allocates NOTHING. That sentence was already
    written when four buffers -- their `BinFeatures` and `FeatureWeights`
    plus the host staging for both -- were still being created and refilled
    inside it; `CreateInitialSubsets` builds those two once
    (`split_properties_helper.cpp:1063`, `:1075-1076`) and
    `structure_searcher_template.h:49` runs it once before the loop at
    `:55-65`, so they are hoisted to match.
    """
    var stat_count = 2
    var max_leaves = 1 << max_depth

    var wide = (n_rows + 255) // 256
    if wide > 256:
        wide = 256
    if wide < 1:
        wide = 1

    # --- persistent state, allocated once --------------------------------
    # These are the SETUP allocations and they stay as they are. CatBoost
    # allocates the same things at the same point: `CreateInitialSubsets`
    # (`split_properties_helper.cpp:1035-1080`) resets the partitions, the
    # partition stats, the histograms, the bin features and the feature
    # weights before a single level runs. What made theirs cheap was that
    # every `TCudaBuffer::Create` lands in `TStackLikeMemoryPool`
    # (`memory_provider_trait.h:14`), a slab carved into 256-byte-aligned
    # stack slices, so no allocation in a tree reaches the driver.
    #
    # We do not port that pool. Mojo's `DeviceContext` already is one: it
    # owns a "device memory pool" that a stream view shares
    # (`DeviceContext.select_stream`), it holds "cached memory buffers" until
    # the context is destroyed (`DeviceContext.__deinit__`), every allocation
    # goes through a memory manager (`MODULAR_DEBUG_DEVICE_ALLOCATOR`
    # poisons "every memory-manager allocation"), and that manager has a
    # defragmenting implementation by default on NVIDIA
    # (`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM`), which is their
    # `MemoryDefragmentation` under another name. See `gpu_lib/gpu_base.mojo`
    # and `VENDOR_LIBS.md`.
    #
    # So the work worth doing was not a second allocator underneath theirs.
    # It was getting the allocations OUT of the level loop, which is where
    # theirs are not.
    var p_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var hp_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var hp_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(0))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_sz, src_ptr=h_sz.unsafe_ptr())

    var hist_cells = max_leaves * stat_count * n_features
    var hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
    # their `FillBuffer(subsets.Histograms, 0.0f)` in `CreateInitialSubsets`
    # (`split_properties_helper.cpp:1061`). `zero_histograms_kernel` clears
    # only the leaves a level rebuilds, and the histogram kernels skip a cell
    # whose accumulator is zero, so an unfilled allocation is never fully
    # overwritten.
    ctx.enqueue_memset(hist, Float32(0.0))

    # FIXED-POINT ACCUMULATOR for replicated blocks under
    # `NUMERIC_IDENTICAL`: partial histograms from blocks sharing a partition
    # sum as Int32 through an integer atomic and are converted back
    # afterwards. Integer addition is associative, so the histogram does not
    # depend on which block lands first. `NUMERIC_FAST` leaves this buffer at
    # zero and takes CatBoost's float `atomicAdd`.
    var acc_i32 = ctx.enqueue_create_buffer[DType.int32](hist_cells)
    ctx.enqueue_memset(acc_i32, Int32(0))

    # The scale bounds every partial sum. A cell can hold at most the sum of
    # magnitudes of the plane it accumulates, and one scale serves both
    # planes, so the bound is the LARGER of the two. A leaf's rows are a
    # subset of all rows, so a scale derived from that cannot overflow at any
    # depth. `choose_scale` owns the derivation; see mojo_only/fixed_point.mojo.
    #
    # THIS WAS INLINED AND IT DRIFTED. The copy that stood here fell back to
    # `mag = 1.0` when both magnitudes were zero, which is a SCALE of
    # 268,435,455 -- the largest scale the type admits, i.e. the one that
    # overflows soonest -- where `choose_scale` returns a scale of 1.0 for
    # the same input. Inlining a safety derivation is how the safe branch and
    # the unsafe branch swapped places.
    var mag = Float64(weight_magnitude)
    if mag < 0.0:
        mag = -mag
    var gmag = Float64(gradient_magnitude)
    if gmag < 0.0:
        gmag = -gmag
    if gmag > mag:
        mag = gmag
    var fixed_scale = Float32(choose_scale(mag, n_rows))

    # REPLICATION: how many blocks share one partition. CatBoost sizes this
    # to fill the SMs (`hist_binary.cu:90-95`). A PARAMETER of this function,
    # defaulting to 1, because a configuration that cannot be varied inside
    # one process cannot be measured on this machine.
    #
    # **1 and 32 are INDISTINGUISHABLE at this shape.** Interleaved, five
    # repeats, depth 6 over 500,000 rows: median 25.755 ms at 1 replica
    # against 24.936 at 32, ranges overlapping.
    #
    # A 1.16x gap measured ACROSS RUNS said 32 was slower. It was noise:
    # this box has produced 59.7 and 44.8 ms for identical work an hour
    # apart. `mojo_only/interleaved.mojo` exists for that reason and
    # overturned the claim the first time it ran.
    #
    # THAT SHAPE HAS NOW BEEN MEASURED and replication pays enormously
    # there: the one-byte kernel at 4 features by 256 folds, 1,024 cells, is
    # **8.56x faster with 16 blocks than with 1, ranges disjoint**. So the
    # factor is not a constant, it is a function of the OUTPUT size, and
    # Passing 0 here asks THEIR formula; a positive value overrides it so the
    # two can still be interleaved.
    # THEIR formula, `hist_binary.cu:95`, not a heuristic of ours.
    # `replicas_for` used to decide this from the histogram width alone and
    # is deleted; see `replication_for` for what it got wrong. Passing a
    # positive `hist_replicas` still overrides, so the arms stay interleavable.
    # THEIR formula (`hist_binary.cu:95`), and it is computed PER LAUNCH
    # because `numBlocks.y` is the leaf count and that changes every level.
    # `replicas_for` is deleted; it keyed on histogram width alone.
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    var part_stats = ctx.enqueue_create_buffer[DType.float32](
        max_leaves * stat_count
    )
    # SIZED BY THE MACHINE CHUNK COUNT, NOT THE DATA. The kernels index
    # `(leaf * n_stats + stat) * max_chunks + chunk` with `max_chunks =
    # partition_stats_chunks(sm_count, n_stats)` -- their machine-sized
    # grid -- and this buffer was still sized by `ceil(n_rows /
    # STATS_BLOCK)` from the data-sized era. Whenever the data count was
    # SMALLER (every fit under ~5k rows on this box), the deep-level
    # writes ran past the end: leaf 15, stat 1, chunk 9 lands at slot 319
    # of a 256-float buffer at depth 6 on 4096 rows. The scribble was
    # SELF-CONSISTENT (phase 2 reads the same out-of-bounds slots), so
    # the 4096-row oracle fixtures passed over it for days; it surfaced
    # as NaN leaf values only when the allocator placed something
    # volatile after the buffer (caught by the train-api check,
    # 2026-08-21). The larger of the two counts is always safe.
    var stat_chunks = (n_rows + STATS_BLOCK - 1) // STATS_BLOCK
    var machine_chunks = partition_stats_chunks(sm_count, stat_count)
    if machine_chunks > stat_chunks:
        stat_chunks = machine_chunks
    var stat_partials = ctx.enqueue_create_buffer[DType.float32](
        max_leaves * stat_count * stat_chunks
    )
    var flags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var seq = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var gmap = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var sflags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var new_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var new_stats = ctx.enqueue_create_buffer[DType.float32](
        stat_count * n_rows
    )

    var max_chunks = (n_rows + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    var chunk_zeros = ctx.enqueue_create_buffer[DType.uint32](
        max_leaves * max_chunks
    )
    var chunk_offsets = ctx.enqueue_create_buffer[DType.uint32](
        max_leaves * max_chunks
    )
    var leaf_zeros = ctx.enqueue_create_buffer[DType.uint32](max_leaves)

    var ids_a = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var ids_b = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var ids_c = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var h_ids_a = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var h_ids_b = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var h_ids_c = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)

    # feature descriptors, constant for the fit
    var folds = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var fold_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_sz = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var q = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        q.unsafe_ptr().unsafe_store(f, UInt32(1))
    ctx.enqueue_copy(dst_buf=folds, src_ptr=q.unsafe_ptr())
    var q2 = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        q2.unsafe_ptr().unsafe_store(f, UInt32(f))
    ctx.enqueue_copy(dst_buf=fold_off, src_ptr=q2.unsafe_ptr())
    var q3 = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        q3.unsafe_ptr().unsafe_store(f, UInt32(0))
    ctx.enqueue_copy(dst_buf=grp_off, src_ptr=q3.unsafe_ptr())
    var q4 = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        q4.unsafe_ptr().unsafe_store(f, UInt32(n_features))
    ctx.enqueue_copy(dst_buf=grp_sz, src_ptr=q4.unsafe_ptr())

    # `TBinarizedFeature::OneHotFeature`, tested by the scan before it
    # prefix-sums (`histogram_utils.cu:395`). This path is uniform binary,
    # so none of them is one-hot.
    var one_hot = ctx.enqueue_create_buffer[DType.uint8](n_features)
    var q5 = ctx.enqueue_create_host_buffer[DType.uint8](n_features)
    for f in range(n_features):
        q5.unsafe_ptr().unsafe_store(f, UInt8(0))
    ctx.enqueue_copy(dst_buf=one_hot, src_ptr=q5.unsafe_ptr())

    var skip = ctx.enqueue_create_buffer[DType.uint8](n_features)
    var hsk = ctx.enqueue_create_host_buffer[DType.uint8](n_features)
    for f in range(n_features):
        hsk.unsafe_ptr().unsafe_store(f, UInt8(0))
    ctx.enqueue_copy(dst_buf=skip, src_ptr=hsk.unsafe_ptr())

    # THEIR `BinFeatures` AND `FeatureWeights`, WHICH ARE PER-TREE, NOT
    # PER-LEVEL. `CreateInitialSubsets` sets both once
    # (`split_properties_helper.cpp:1063` takes `BinFeatures` as a const view
    # of the by-blocks helper's buffer, `:1075-1076` creates and writes
    # `FeatureWeights`), and `structure_searcher_template.h:49` calls it once
    # before the level loop at `:55-65`. `ComputeOptimalSplits` then only
    # READS `subsets->BinFeatures` and `subsets->FeatureWeights`, as the two
    # arguments at `greedy_search_helper.cpp:455-456`; it allocates nothing
    # shaped like them. The one thing it does touch them with is
    # `UpdateFeatureWeightsForBestSplits` at `:447`, which returns at
    # `update_feature_weights.cpp:20-22` the moment `GetCtrsCount() == 0`.
    # With no CTRs that call is a no-op and the weights are a per-tree
    # constant, which is why hoisting them changes no value.
    #
    # These four used to be created INSIDE the depth loop and refilled with
    # the same constants at every level: `bff[f] = f` and `ffw[f] = 1.0` do
    # not depend on `depth`. That cost four allocations and two host-to-device
    # copies per level for a value that never changed.
    #
    # `bff` is their `TCBinFeature.FeatureId` column (`compute_scores.cu:136`).
    # `ffw` is all 1.0 because `UpdateFeatureWeights` fills 1.0 and returns
    # early with no CTRs (`update_feature_weights.cpp:14-22`) -- their value,
    # not a stub.
    #
    # LIFETIME. `bff` and `ffw` are read by the score kernel at every level
    # and are never written again after the two copies below, so no level can
    # observe a stale or half-written value. `hbf` and `hfw` are the host
    # staging for those copies; they stay in scope for the whole function so
    # that the asynchronous `enqueue_copy` cannot outlive its source.
    # The `ctx.synchronize()` further down settles both before the loop.
    var bff = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var hbf = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var ffw = ctx.enqueue_create_buffer[DType.float32](n_features)
    var hfw = ctx.enqueue_create_host_buffer[DType.float32](n_features)
    for f in range(n_features):
        hbf.unsafe_ptr().unsafe_store(f, UInt32(f))
    for f in range(n_features):
        hfw.unsafe_ptr().unsafe_store(f, Float32(1.0))
    ctx.enqueue_copy(dst_buf=bff, src_ptr=hbf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ffw, src_ptr=hfw.unsafe_ptr())

    var out_score = ctx.enqueue_create_buffer[DType.float32](1)
    var out_bin = ctx.enqueue_create_buffer[DType.uint32](1)
    var hos = ctx.enqueue_create_host_buffer[DType.float32](1)
    var hob = ctx.enqueue_create_host_buffer[DType.uint32](1)

    # Their `MakeSplit` packs every ui32 it has to send into ONE buffer and
    # copies once (`split_properties_helper.cpp:886-905`):
    #
    #     const TSlice binsSlice      = TSlice(0, splitBins.size());
    #     const TSlice leftIdsSlice   = TSlice(binsSlice.Right, ...);
    #     const TSlice rightIdsSlice  = TSlice(leftIdsSlice.Right, ...);
    #     auto allUi32Data = TMirrorBuffer<ui32>::Create(...);
    #     allUi32DataCpu.Write(tmp);
    #     allUi32Data.Copy(allUi32DataCpu);
    #
    # Ours holds the four ui32 fields of the split descriptor the same way.
    # It used to be four separate buffers and four separate copies followed
    # by a drain. The one-hot flag stays its own buffer because it is UInt8,
    # which is also why theirs keeps `splitFeaturesGpu` separate from
    # `allUi32Data`.
    var built_f = make_split_features_buffers(ctx, max_leaves)
    var sp_feats = built_f[0]
    var sp_feats_h = built_f[1]
    var sp5 = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var hs5 = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    ctx.synchronize()

    # Leaf values land here; the caller reads them through `tree_leaf_value`.
    var h_leaf_values = ctx.enqueue_create_host_buffer[DType.float32](
        max_leaves
    )

    var n_live = 1  # leaves at the current level

    # THE LARGEST LIVE LEAF, tracked on the host so the grids that are sized
    # per leaf are sized for the level rather than for the dataset.
    #
    # Passing `n_rows` here sizes every per-leaf grid for the biggest leaf
    # that could ever exist, so at depth d there are 2^d leaves each with a
    # full dataset's worth of chunks and nearly all of them empty. The empty
    # work grows as 2^d while the real work stays flat, which showed up as
    # marginal cost per level RISING through depth 5 (+4.9, +9.9, +15.8,
    # +19.5 ms) when a constant per-level row count says it should be flat.
    var max_live_rows = n_rows

    for depth in range(max_depth):
        # ---- 1. histograms for this level's leaves ----------------------
        # Depth 0 has one leaf and no sibling, so it is a plain build. Below
        # that every leaf of the previous level became a pair, and only the
        # SMALLER of each pair is accumulated.
        for i in range(n_live):
            h_ids_a.unsafe_ptr().unsafe_store(i, UInt32(i))
        ctx.enqueue_copy(dst_buf=ids_a, src_ptr=h_ids_a.unsafe_ptr())
        ctx.synchronize()

        # `numBlocks.x = CeilDivide(binFeatureCount, blockSize)` with
        # blockSize 256 (`histogram_utils.cu:236-238`). Grid x of 1 cleared
        # only the first 256 bin-features; everything above kept the previous
        # level's values, because `fixed_to_float_kernel` does not write a
        # cell whose accumulator is zero.
        ctx.enqueue_function[zero_histograms_kernel](
            ids_a.unsafe_ptr(),
            Int32(n_features),
            hist.unsafe_ptr(),
            grid_dim=((n_features + 255) // 256, n_live, stat_count),
            block_dim=(256, 1, 1),
        )
        # DIRECT LOADS AT DEPTH 0, GATHER BELOW IT. The compressed index is
        # never permuted, so position names row only while the index is the
        # identity. Direct loads below depth 0 read unrelated rows' bins,
        # which looks like healthy plumbing and splits nothing.
        if depth == 0:
            ctx.enqueue_function[binary_hist_kernel](
                folds.unsafe_ptr(), fold_off.unsafe_ptr(),
                grp_off.unsafe_ptr(), grp_sz.unsafe_ptr(),
                Int32(n_features), cindex.unsafe_ptr(), Int32(n_rows),
                Int32(0),
                stats.unsafe_ptr(), Int32(n_rows),
                p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids_a.unsafe_ptr(),
                hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                Int32(max_leaves), Int32(stat_count),
                grid_dim=(
                    hist_replicas if hist_replicas > 0 else replication_for(
                        1, n_live, stat_count, sm_count
                    ),
                    n_live,
                    stat_count,
                ),
                block_dim=(BLOCK_SIZE, 1, 1),
            )
        else:
            ctx.enqueue_function[binary_hist_gather_kernel](
                folds.unsafe_ptr(), fold_off.unsafe_ptr(),
                grp_off.unsafe_ptr(), grp_sz.unsafe_ptr(),
                Int32(n_features), cindex.unsafe_ptr(), Int32(n_rows),
                Int32(0),
                row_index.unsafe_ptr(),
                stats.unsafe_ptr(), Int32(n_rows),
                p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids_a.unsafe_ptr(),
                hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                Int32(max_leaves), Int32(stat_count),
                grid_dim=(
                    hist_replicas if hist_replicas > 0 else replication_for(
                        1, n_live, stat_count, sm_count, gather=True
                    ),
                    n_live,
                    stat_count,
                ),
                block_dim=(BLOCK_SIZE, 1, 1),
            )

        # Convert the replicated partials back to floats before the scan.
        ctx.enqueue_function[fixed_to_float_kernel](
            acc_i32.unsafe_ptr(),
            hist.unsafe_ptr(),
            Int32(hist_cells),
            fixed_scale,
            grid_dim=(hist_cells + 255) // 256,
            block_dim=256,
        )
        # `numBlocks.x = CeilDivide(fCount * 32, blockSize)`
        # (`histogram_utils.cu:442`), theirs at one WARP per feature. Ours is
        # one THREAD per feature, so the same rule is
        # `ceil(fCount / blockSize)`. The kernel guards with
        # `feature_id >= feature_count` and has no grid-stride loop, so grid
        # x of 1 left every feature from 256 up holding raw per-bin counts.
        ctx.enqueue_function[scan_histograms_kernel](
            ids_a.unsafe_ptr(),
            fold_off.unsafe_ptr(),
            folds.unsafe_ptr(),
            one_hot.unsafe_ptr(),
            Int32(n_features),
            Int32(n_features),
            hist.unsafe_ptr(),
            grid_dim=((n_features + 255) // 256, n_live, stat_count),
            block_dim=(256, 1, 1),
        )

        # ---- 2. per-leaf totals, ON DEVICE ------------------------------
        # `ComputePartitionStats`. The score kernel derives each leaf's right
        # side as `total - left`, so a leaf with no total scores degenerately
        # and the level re-picks the same feature. Hardcoding leaf 0's total
        # made depth 0 look correct and produced 64 leaves of which only 2
        # were non-empty at depth 6.
        compute_partition_stats(
            ctx, n_live, max_live_rows, stat_count, n_rows,
            ids_a, p_off, p_sz, stats, stat_partials, part_stats,
            sm_count=sm_count,
        )
        ctx.synchronize()

        # ---- 3. one split for the whole level ---------------------------
        # `bff` and `ffw` are their `subsets->BinFeatures` and
        # `subsets->FeatureWeights`, built once above this loop because that
        # is where `CreateInitialSubsets` builds them
        # (`split_properties_helper.cpp:1063`, `:1075-1076`).
        # `ComputeOptimalSplits` reads them and allocates neither
        # (`greedy_search_helper.cpp:455-456`). NOTHING in this loop is
        # allocated any more; every buffer the level touches was created
        # before it.
        ctx.enqueue_function[
            compute_optimal_splits_kernel[SCORE_FUNCTION_COSINE]
        ](
            skip.unsafe_ptr(),
            Int32(n_features),
            bff.unsafe_ptr(), ffw.unsafe_ptr(),
            hist.unsafe_ptr(),
            part_stats.unsafe_ptr(),
            Int32(stat_count),
            ids_a.unsafe_ptr(),
            Int32(n_live),
            Float32(1.0),
            out_score.unsafe_ptr(),
            out_bin.unsafe_ptr(),
            grid_dim=(1, 1, 1),
            block_dim=(SCORE_BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=hob.unsafe_ptr(), src_buf=out_bin)
        ctx.synchronize()
        var best = Int(hob.unsafe_ptr().unsafe_load(0))
        if best > n_features:
            best = 0

        # ---- 4. apply it to every leaf ----------------------------------
        var hfeat = sp_feats_h.unsafe_ptr().bitcast[CFeature]()
        for i in range(n_live):
            hfeat[unsafe_offset=i] = (
                CFeature(
                    offset=UInt32(0),
                    mask=UInt32(1),
                    shift=UInt32(31 - best),
                    first_fold_index=UInt32(0),
                    folds=UInt32(1),
                    one_hot_feature=False,
                )
            )
            hs5.unsafe_ptr().unsafe_store(i, UInt32(0))
        ctx.enqueue_copy(dst_buf=sp_feats, src_ptr=sp_feats_h.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=sp5, src_ptr=hs5.unsafe_ptr())
        ctx.synchronize()

        ctx.enqueue_function[split_and_make_sequence_kernel](
            cindex.unsafe_ptr(),
            row_index.unsafe_ptr(),
            p_off.unsafe_ptr(),
            p_sz.unsafe_ptr(),
            ids_a.unsafe_ptr(),
            sp_feats.unsafe_ptr().bitcast[CFeature](),
            sp5.unsafe_ptr(),
            flags.unsafe_ptr(),
            seq.unsafe_ptr(),
            grid_dim=(wide, n_live, 1),
            block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )

        launch_stable_partition(
            ctx, n_live, max_live_rows, ids_a, p_off, p_sz, flags,
            chunk_zeros, chunk_offsets, leaf_zeros, gmap, sflags,
        )

        # their `TSplitPointsKernel::Run` (`split_points.cpp:64-136`): one
        # branch on the largest splitting leaf, then either the shared-memory
        # fast path in ONE launch or `CopyInLeaves` then `GatherInLeaves` per
        # chunk of eight stat columns plus one more pair for the index. Every
        # launch is restricted to the leaf ranges, so rows outside a splitting
        # leaf are neither read nor written.
        _ = launch_reorder_in_leaves(
            ctx, n_live, wide, max_live_rows, stat_count, n_rows,
            ids_a, p_off, p_sz, stats, new_stats, row_index, new_index, gmap,
            sm_count=sm_count,
        )

        # ---- 5. one leaf becomes two ------------------------------------
        # THEIR NUMBERING (`split_properties_helper.cpp:860-861`):
        #
        #     const ui32 leftId = leavesToSplit[i];
        #     const ui32 rightId = static_cast<const ui32>(leavesCount + i);
        #
        # The left child KEEPS the parent's slot and the right child is
        # appended past the end of the level. That is why their partition
        # array never round trips: `UpdatePartitionsAfterSplitImpl` reads
        # `parts[leftLeaf]`, which still holds the parent, and writes both
        # children out of it (`split_points.cu:346-380`).
        #
        # This used to number the children 2i and 2i+1, which needs the
        # parent spread outward before the border search runs, and that
        # spread was a device-to-host read of BOTH partition planes, a host
        # loop, and a host-to-device write back, once per level.
        # `run_tree_layout` deleted it by taking their numbering; this takes
        # their numbering for the same reason. Nothing in this loop wants
        # children adjacent: every per-leaf kernel here is handed an id list
        # plus `p_off` / `p_sz` and never assumes an order over them.
        for i in range(n_live):
            h_ids_b.unsafe_ptr().unsafe_store(i, UInt32(i))
            h_ids_c.unsafe_ptr().unsafe_store(i, UInt32(n_live + i))
        ctx.enqueue_copy(dst_buf=ids_b, src_ptr=h_ids_b.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=ids_c, src_ptr=h_ids_c.unsafe_ptr())

        # `hp_off` / `hp_size` are their `partsCpu`, and ours are ORDINARY
        # DEVICE BUFFERS that no host read reaches. That is a deviation and
        # not a choice, so read the DEVIATION BLOCK in
        # `gbdt/gpu_util/gpu_data/partitions.mojo` before assuming the
        # allocation could simply be moved. The kernel writes them where
        # theirs writes `partsCpu` (`split_points.cu:372`, `:379`).
        ctx.enqueue_function[update_partitions_after_split_kernel](
            ids_b.unsafe_ptr(),
            ids_c.unsafe_ptr(),
            Int32(n_live),
            sflags.unsafe_ptr(),
            p_off.unsafe_ptr(),
            p_sz.unsafe_ptr(),
            hp_off.unsafe_ptr(),
            hp_sz.unsafe_ptr(),
            grid_dim=(wide, n_live, 1),
            block_dim=(512, 1, 1),
        )

        n_live = n_live * 2

        # ONE partition read per level, which is their count.
        # `RebuildLeavesSizes` is `currentParts.Read(partsCpu)` over the
        # whole leaf range (`split_properties_helper.cpp:800-813`) and runs
        # once per `MakeSplit`, at `:950`. O(leaves) on the host, never
        # O(rows). See HOST_AND_DEVICE.md.
        #
        # ORDERING, and it is the part that a wrong answer would not
        # announce. The two id uploads, the kernel, and this copy are all
        # enqueued on one queue in that order, so the kernel cannot read an
        # id that has not landed and the copy cannot read a `p_sz` the kernel
        # has not written. The drain below is only what makes the copy
        # READABLE on the host. It is the sole drain this path needs; the
        # three that used to sit above it ordered nothing the queue did not
        # already order, and CatBoost has one.
        ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
        ctx.synchronize()
        max_live_rows = 1
        for i in range(n_live):
            var s = Int(h_sz.unsafe_ptr().unsafe_load(i))
            if s > max_live_rows:
                max_live_rows = s

    # ---- leaf values, once the structure is final ----------------------
    # `compute_partition_stats` over the FINAL partitions, then the Newton
    # step. The stats kernel is reused rather than a second reduction
    # written: the layout it produces is exactly what the estimator reads.
    for i in range(n_live):
        h_ids_a.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=ids_a, src_ptr=h_ids_a.unsafe_ptr())
    ctx.synchronize()
    compute_partition_stats(
        ctx, n_live, max_live_rows, stat_count, n_rows,
        ids_a, p_off, p_sz, stats, stat_partials, part_stats,
        sm_count=sm_count,
    )
    var leaf_values = ctx.enqueue_create_buffer[DType.float32](max_leaves)
    ctx.enqueue_function[compute_leaf_values_kernel](
        part_stats.unsafe_ptr(),
        Int32(stat_count),
        Int32(n_live),
        Float32(1.0),
                leaf_values.unsafe_ptr(),
        grid_dim=(n_live + LEAF_BLOCK - 1) // LEAF_BLOCK,
        block_dim=LEAF_BLOCK,
    )
    ctx.synchronize()

    ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
    ctx.enqueue_copy(dst_ptr=h_leaf_values.unsafe_ptr(), src_buf=leaf_values)
    ctx.synchronize()
    var out = List[Int]()
    for i in range(n_live):
        out.append(Int(h_sz.unsafe_ptr().unsafe_load(i)))
    return out^


@fieldwise_init
struct DeviceBlock(Copyable, Movable):
    """One policy's descriptor arrays, uploaded once for the whole fit."""

    var policy: Int
    var n_features: Int
    var first_column: Int
    var total_folds: Int
    var max_folds: Int
    """Widest fold count in this block. The one-byte kernel's bit width must
    cover it and must not exceed it; see the dispatch note."""
    var folds: DeviceBuffer[DType.uint32]
    var fold_off: DeviceBuffer[DType.uint32]
    var grp_off: DeviceBuffer[DType.uint32]
    var grp_sz: DeviceBuffer[DType.uint32]


def upload_blocks(
    ctx: DeviceContext, blocks: List[PolicyBlock]
) raises -> List[DeviceBlock]:
    """Upload every policy's descriptor arrays once.

    They are constant for the fit: the layout depends on fold counts, not on
    the tree. Re-uploading per level would be a copy per level for nothing.
    """
    var out = List[DeviceBlock]()
    for b in range(len(blocks)):
        ref blk = blocks[b]
        var n = blk.count()
        var d_folds = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_fo = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_go = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_gs = ctx.enqueue_create_buffer[DType.uint32](n)
        var h1 = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var h2 = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var h3 = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var h4 = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var total = 0
        var widest = 0
        for k in range(n):
            h1.unsafe_ptr().unsafe_store(k, blk.folds[k])
            h2.unsafe_ptr().unsafe_store(k, blk.fold_offset[k])
            h3.unsafe_ptr().unsafe_store(k, blk.group_offset[k])
            h4.unsafe_ptr().unsafe_store(k, blk.group_size[k])
            total += Int(blk.folds[k])
            if Int(blk.folds[k]) > widest:
                widest = Int(blk.folds[k])
        ctx.enqueue_copy(dst_buf=d_folds, src_ptr=h1.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_fo, src_ptr=h2.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_go, src_ptr=h3.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_gs, src_ptr=h4.unsafe_ptr())
        out.append(
            DeviceBlock(
                blk.policy, n, blk.first_column, total, widest,
                d_folds^, d_fo^, d_go^, d_gs^,
            )
        )
    ctx.synchronize()
    return out^


def launch_one_byte[bits: Int, smem_mode: Int = HIST2_SMEM_MODE](
    ctx: DeviceContext,
    mut blk: DeviceBlock,
    depth: Int,
    n_live: Int,
    n_rows: Int,
    stat_count: Int,
    max_leaves: Int,
    sm_count: Int,
    line: Int,
    base: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
    mut block_hist: DeviceBuffer[DType.float32],
    mut acc_i32: DeviceBuffer[DType.int32],
    fixed_scale: Float32,
    grid_z_stats: Int = -1,
) raises:
    """One one-byte launch at a comptime bit width. Direct at depth 0,
    indexed below it.

    `grid_z_stats` is how many stat planes THIS LAUNCH covers, i.e. the
    `NumStats` argument of their `PASS(Bits, NumStats)` macro
    (`hist_one_byte.cu:283-304`). The default, `stat_count`, is their
    `PASS(8, numStats)` arm. `launch_hist2_one_byte` passes 1 for the
    odd-`numStats` prelude, their `PASS(Bits, 1)` inside `HIST2_PASS`
    (`hist_one_byte.cu:306-312`), which covers stat 0 alone while the
    kernel's `statCount` stride stays the full `stat_count`.

    OUR NAMES DO NOT MEAN THEIRS, AND THE MISMATCH IS A TRAP.

    CatBoost picks between two policies at
    `split_properties_helper.cpp:1337-1339`:

        loadPolicy = (Leaves.size() == 1 || GetStatCount() <= 2)
                        ? LoadByIndexBins : GatherBins;

    `LoadByIndexBins` reads each row's bins THROUGH `Target.Indices`, one
    indirection per load. `GatherBins` first materialises a reordered copy of
    the compressed index into `tempGatheredCompressedIndex` and then reads it
    contiguously.

    Our kernel called "gather" is the one that takes `indices`, so it is
    their **LoadByIndexBins**, not their GatherBins. We have NO
    implementation of GatherBins at all: nothing in this repository ever
    materialises a reordered compressed index, which is verifiable by
    grepping for a temp gathered index and finding none.

    That is CORRECT BY THEIR DISPATCH rather than by luck. The boosting path
    runs `stat_count = 2` (`doc_parallel_boosting.mojo:128`), so
    `GetStatCount() <= 2` is always true and CatBoost takes LoadByIndexBins
    at every depth, which is what we run at every depth below the root. The
    unported policy is one their own dispatch never selects for our
    parameters.

    So do not "fix" this by porting GatherBins, and do not read the word
    gather here as evidence that we took their gather path. The `depth == 0`
    shortcut IS ours: at the root the index is the identity, so the
    indirection is provably a no-op and the direct kernel is the same
    arithmetic with one load removed.

    THE FAILURE SHAPE THIS WAS CHECKED FOR, because a peer session found it
    four times in one round: a function of theirs ported faithfully, while
    their dispatch would never send our parameters to it. It compiles, it
    passes, its citations are real, and it is the wrong kernel. Checked here
    on 2026-08-19 and clean. `SortByFlagsInLeaf` was checked the same way,
    `part.Size > FastSortSize()` at 500000 (`split_points.cu:737, :749`), and
    our leaves are far below it, so `SortWithoutCub` is their path too.
    """
    var gz = stat_count
    if grid_z_stats >= 0:
        gz = grid_z_stats
    if depth == 0:
        ctx.enqueue_function[one_byte_hist_kernel[bits, smem_mode]](
            blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
            blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
            Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line),
            Int32(base), stats.unsafe_ptr(), Int32(n_rows),
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
            block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
            Int32(max_leaves), Int32(stat_count),
            # REPLICATION IS BACK. The one-byte kernel now has the
            # fixed-point Int32 flush that `hist_binary.mojo` has, so
            # replicated blocks sum their partials through an integer atomic
            # instead of overwriting each other with a plain store.
            #
            # It ran at grid.x = 1 for exactly as long as that flush was
            # missing, which was one commit. See UNWIRED.md for the
            # measurement that forced it: 459 of 3168 cells wrong on
            # contiguous ids at 16 replicas.
            #
            # CatBoost replicates this kernel and sums the partials with
            # `atomicAdd(dst + fold, val)` when `blockCount > 1`
            # (`hist_one_byte.cu:255`, `AddToGlobalMemory`), which is what
            # `NUMERIC_FAST` does here. `NUMERIC_IDENTICAL` sends the same
            # flush through the Int32 accumulator instead, because integer
            # addition is associative and the result then does not depend on
            # which block lands first.
            grid_dim=(
                feature_groups_for(POLICY_ONE_BYTE, Int(blk.n_features))
                * replication_for(
                    feature_groups_for(POLICY_ONE_BYTE, Int(blk.n_features)),
                    n_live, gz, sm_count,
                ),
                n_live,
                gz,
            ),
            block_dim=(one_byte_block_size[smem_mode](), 1, 1),
        )
    else:
        ctx.enqueue_function[one_byte_hist_gather_kernel[bits, smem_mode]](
            blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
            blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
            Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line),
            Int32(base), row_index.unsafe_ptr(),
            stats.unsafe_ptr(), Int32(n_rows),
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
            block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
            Int32(max_leaves), Int32(stat_count),
            # REPLICATION IS BACK. The one-byte kernel now has the
            # fixed-point Int32 flush that `hist_binary.mojo` has, so
            # replicated blocks sum their partials through an integer atomic
            # instead of overwriting each other with a plain store.
            #
            # It ran at grid.x = 1 for exactly as long as that flush was
            # missing, which was one commit. See UNWIRED.md for the
            # measurement that forced it: 459 of 3168 cells wrong on
            # contiguous ids at 16 replicas.
            #
            # CatBoost replicates this kernel and sums the partials with
            # `atomicAdd(dst + fold, val)` when `blockCount > 1`
            # (`hist_one_byte.cu:255`, `AddToGlobalMemory`), which is what
            # `NUMERIC_FAST` does here. `NUMERIC_IDENTICAL` sends the same
            # flush through the Int32 accumulator instead, because integer
            # addition is associative and the result then does not depend on
            # which block lands first.
            grid_dim=(
                feature_groups_for(POLICY_ONE_BYTE, Int(blk.n_features))
                * replication_for(
                    feature_groups_for(POLICY_ONE_BYTE, Int(blk.n_features)),
                    n_live, gz, sm_count, gather=True,
                ),
                n_live,
                gz,
            ),
            block_dim=(one_byte_block_size[smem_mode](), 1, 1),
        )



def launch_hist2_one_byte[bits: Int, smem_mode: Int = HIST2_SMEM_MODE](
    ctx: DeviceContext,
    mut blk: DeviceBlock,
    depth: Int,
    n_live: Int,
    n_rows: Int,
    stat_count: Int,
    max_leaves: Int,
    sm_count: Int,
    line: Int,
    base: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
    mut block_hist: DeviceBuffer[DType.float32],
    mut acc_i32: DeviceBuffer[DType.int32],
    fixed_scale: Float32,
) raises:
    """Their `HIST2_PASS(Bits)` macro plus `ComputeHist2OneByteBits`
    (`hist_one_byte.cu:306-312` and `hist_2_one_byte_base.cuh:155-197`
    direct / `:245-284` gather, the multi-part overloads), as one launcher.

        if (numStats % 2 != 0) {
            PASS(Bits, 1)                                   // stat 0 alone
            ComputeHist2OneByteBits<Bits, true>(...);       // pairs, shifted
        } else {
            ComputeHist2OneByteBits<Bits, false>(...);      // pairs
        }

    Our boosting path runs `stat_count = 2`, so the even arm is what runs;
    the odd arm is ported because the ladder is theirs, and the prelude goes
    through `launch_one_byte[bits]` with `grid_z_stats = 1`, which is their
    `PASS(Bits, 1)` at the same `bits`.

    THE GRID (`hist_2_one_byte_base.cuh:172-180` direct, `:259-267` gather):

        numBlocks.z = (numStats - IsOdd) / 2;               // STAT PAIRS
        numBlocks.y = partCount;
        numBlocks.x = (fCount + 3) / 4;
        numBlocks.x *= CeilDivide(maxActiveBlocks, x * y * z);

    Copied from THEIR file, not reused from the PASS family: `z` is the PAIR
    count, not the stat count, and BOTH multi-part overloads divide the
    plain `maxActiveBlocks` -- the gather arm does NOT get the PASS family's
    doubled target (`:267` against `hist_one_byte.cu:356`), so
    `replication_for` is called with `gather=False` on both arms here.
    """
    comptime BLOCK = hist2_block_size[smem_mode]()

    var is_odd = stat_count % 2
    if is_odd == 1:
        # `PASS(Bits, 1)`: the one-stat PASS-family kernel covers stat 0.
        launch_one_byte[bits, smem_mode](
            ctx, blk, depth, n_live, n_rows, stat_count, max_leaves,
            sm_count, line, base, cindex, row_index, stats, p_off,
            p_sz, ids, block_hist, acc_i32, fixed_scale, grid_z_stats=1,
        )

    var pairs = (stat_count - is_odd) // 2
    if pairs == 0:
        return

    var groups = feature_groups_for(POLICY_ONE_BYTE, Int(blk.n_features))
    var replicas = replication_for(groups, n_live, pairs, sm_count)

    if depth == 0:
        if is_odd == 1:
            ctx.enqueue_function[hist2_one_byte_kernel[bits, True, smem_mode]](
                blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
                blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
                Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line),
                Int32(base), stats.unsafe_ptr(), Int32(n_rows),
                p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
                block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                Int32(max_leaves), Int32(stat_count),
                grid_dim=(groups * replicas, n_live, pairs),
                block_dim=(BLOCK, 1, 1),
            )
        else:
            ctx.enqueue_function[hist2_one_byte_kernel[bits, False, smem_mode]](
                blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
                blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
                Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line),
                Int32(base), stats.unsafe_ptr(), Int32(n_rows),
                p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
                block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                Int32(max_leaves), Int32(stat_count),
                grid_dim=(groups * replicas, n_live, pairs),
                block_dim=(BLOCK, 1, 1),
            )
    else:
        if is_odd == 1:
            ctx.enqueue_function[hist2_one_byte_gather_kernel[bits, True, smem_mode]](
                blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
                blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
                Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line),
                Int32(base), row_index.unsafe_ptr(),
                stats.unsafe_ptr(), Int32(n_rows),
                p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
                block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                Int32(max_leaves), Int32(stat_count),
                grid_dim=(groups * replicas, n_live, pairs),
                block_dim=(BLOCK, 1, 1),
            )
        else:
            ctx.enqueue_function[hist2_one_byte_gather_kernel[bits, False, smem_mode]](
                blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
                blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
                Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line),
                Int32(base), row_index.unsafe_ptr(),
                stats.unsafe_ptr(), Int32(n_rows),
                p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
                block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                Int32(max_leaves), Int32(stat_count),
                grid_dim=(groups * replicas, n_live, pairs),
                block_dim=(BLOCK, 1, 1),
            )


def feature_groups_for(policy: Int, n_features: Int) -> Int:
    """`numBlocks.x = (fCount + 3) / 4` and its siblings.

    CatBoost's grid x carries TWO factors multiplied together
    (`hist_one_byte.cu:290-291`):

        numBlocks.x  = ceil(fCount / GroupSize);          // feature groups
        numBlocks.x *= CeilDivide(maxActiveBlocks, ...);  // row replication

    and the kernels invert it as
    `maxBlocksPerPart = gridDim.x / featureBlocks`.

    The port launched with grid x = REPLICAS ALONE, dropping the feature-group
    factor. With one group the two agree and everything works, which is every
    check this repository had. With two or more groups
    `maxBlocksPerPart` becomes `1 / 2 == 0` and the kernel divides by zero,
    and the histogram comes back EMPTY rather than partial.

    `GroupSize` is how many features share a `UInt32`, which is the policy's
    whole reason for existing: 32 binary, 8 half-byte, 4 one-byte. Kept in one
    place so it cannot drift from the kernels' own `feature_blocks`.
    """
    if policy == POLICY_BINARY:
        return (n_features + 31) // 32
    if policy == POLICY_HALF_BYTE:
        return (n_features + 7) // 8
    return (n_features + 3) // 4



def replication_for(
    groups: Int, n_live: Int, stat_count: Int, sm_count: Int,
    gather: Bool = False,
) -> Int:
    """`numBlocks.x *= CeilDivide(maxActiveBlocks, x * y * z)`.

    PORT OF the grid sizing shared by all three histogram kernels
    (`hist_binary.cu:95`, `hist_half_byte.cu:81`, `hist_one_byte.cu:291`):

        blocksPerSm     = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
        maxActiveBlocks = blocksPerSm * TArchProps::SMCount();
        numBlocks.x     = ceil(fCount / GroupSize);
        numBlocks.x    *= CeilDivide(maxActiveBlocks,
                                     numBlocks.x * numBlocks.y * numBlocks.z);

    **Replication exists to FILL THE MACHINE, and only that.** The other grid
    axes already supply blocks: `y` is the leaf count and `z` is the stat
    count. Replication makes up the shortfall between what they supply and
    what the device can run at once, so it collapses to 1 the moment there
    are enough leaves, which is every level below the first few.

    ================= WHAT THIS REPLACES =================
    `mojo_only/kernel_matrix.replicas_for` was OURS, not theirs. It chose 16
    or 1 from the HISTOGRAM WIDTH, ignoring the leaf count, the stat count
    and the device entirely. At depth 6 that asks for 16 replicas on top of
    64 leaves times 2 stats times 25 groups, which is 3200 blocks already:
    sixteen times more blocks than the machine can hold, each doing a
    sixteenth of the rows, all contending on the same atomics.

    An invented heuristic in place of a formula that was sitting in their
    source is exactly what PORTING_RULES rule 0 forbids, and it survived
    because every measurement of it was taken on an empty histogram.
    ======================================================
    """
    var blocks_per_sm = 2
    var max_active_blocks = blocks_per_sm * sm_count
    # THE GATHER ARM DOUBLES THE TARGET. Their direct-load launches divide
    # `maxActiveBlocks` (`hist_one_byte.cu:291`) but every gather launch
    # divides `2 * maxActiveBlocks` (`hist_one_byte.cu:356`,
    # `hist_half_byte.cu` and `hist_binary.cu` gather arms alike): an
    # indirected row fetch stalls longer, so it gets twice the blocks to
    # hide behind. This port used the direct-load number for both arms.
    if gather:
        max_active_blocks = 2 * max_active_blocks
    var base = groups * n_live * stat_count
    if base < 1:
        base = 1
    var rep = (max_active_blocks + base - 1) // base
    if rep < 1:
        rep = 1
    return rep


def launch_histograms_for_blocks[
    hist2_smem_mode: Int = HIST2_SMEM_MODE
](
    ctx: DeviceContext,
    mut blocks: List[DeviceBlock],
    depth: Int,
    n_live: Int,
    n_rows: Int,
    stat_count: Int,
    max_leaves: Int,
    sm_count: Int,
    fixed_scale: Float32,
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
    mut dense_ids: DeviceBuffer[DType.uint32],
    mut hist: DeviceBuffer[DType.float32],
    mut acc_i32: DeviceBuffer[DType.int32],
    mut block_hist: DeviceBuffer[DType.float32],
    hist_cells_per_leaf: Int,
    skip_bridge: Bool = False,
) raises:
    """One histogram launch per policy present, dispatching on the block.

    This is `B` histogram kernels of the `3B + 12` census, and it is where
    the port stops assuming a uniform dataset. Direct loads at depth 0 where
    the index is the identity, gather below it, exactly as the binary path
    already does.

    `bins_line_size` is passed as the block's FIRST COLUMN stride so the
    kernel's `blockIdx.x / maxBlocksPerPart` column arithmetic lands inside
    this policy's contiguous columns.

    THEIR LOOP BODY, `split_properties_helper.cpp:1147-1240`, in order:
    reset the block mapping (`:1154`), `ZeroBuffer` the scratch (`:1157`),
    launch the histogram kernel, `ReduceScatter` (`:1219`), then
    `TWriteReducesHistogramsKernel` (`:1224`). Ours is the same order with
    two differences, both stated where they happen: `fixed_to_float_kernel`
    sits between the kernel and the writeback and has no counterpart of
    theirs, and `ReduceScatter` has no counterpart of ours.

    `n_live` and `ids` are their `leavesCount` and `*leavesGpu`
    (`split_properties_helper.cpp:1096` and `:1102`), which is
    `nonZeroComputeLeaves` and NOT every leaf of the level. `ids[blockIdx.y]`
    is the leaf whose partition the kernel reads; the scratch is written at
    the dense `blockIdx.y`. Callers must therefore pass the NON-EMPTY set and
    must not call at all when it is empty, which is their
    `if (leavesToCompute.size() == 0) { return; }` (`:1089-1091`).

    `dense_ids` is no longer read. It was the id list handed to the scratch's
    old indexed zero; the scratch now takes a whole-buffer zero, which needs
    no ids at all. The argument stays because the callers of this function
    live in `mojo_only/` and their signatures are not this file's to change.
    """
    # Where each block's slice begins in the flat histogram: the running
    # total of earlier blocks' bin counts.
    var block_first_bin = 0

    # ZERO THE BLOCK SCRATCH before every block. Their writeback is guarded
    # by `if (abs(val) > 1e-20f)`, so a cell whose value is zero is NEVER
    # WRITTEN and keeps whatever the buffer held. The float `atomicAdd`
    # branch needs the same thing for a harder reason: it ACCUMULATES into
    # the cell, so a stale cell is added to rather than overwritten.
    #
    # It also means the scratch cannot be shared between blocks without
    # clearing: block 2 would inherit block 1's cells wherever its own value
    # rounds to zero.

    for b in range(len(blocks)):
        ref blk = blocks[b]
        # The policy's column BASE, and the feature-block stride within it.
        var base = n_rows * blk.first_column
        var line = n_rows

        # grid x = FEATURE GROUPS times replication, their
        # `numBlocks.x = ceil(fCount / GroupSize); numBlocks.x *= ...`
        var groups = feature_groups_for(blk.policy, blk.n_features)
        var replicas = replication_for(
            groups, n_live, stat_count, sm_count, gather=(depth > 0)
        )

        # Each block writes its own scratch, and writing straight into the
        # flat histogram is correct only when there is one block, because the
        # writeback strides by this block's `group_size` and the flat array
        # strides by the total.
        #
        # `BlockHistogramsMapping(blockId, histCount, statCount)` is
        # `blockSize.Size() * histCount * statCount`
        # (`compute_by_blocks_helper.h:74-78`), where `blockSize.Size()` is
        # this block's bin-feature count. Ours is the same product with
        # `max_leaves` in place of their `leavesCount`, because the kernels
        # are handed `max_leaves` as `leafCount` and that is what
        # `device_offset = group_offset * stat_count * leaf_count` strides
        # by. So this is exactly the span the kernels can touch.
        var block_cells = max_leaves * stat_count * blk.total_folds

        # `ZeroBuffer(blockHistograms, streamId)`
        # (`split_properties_helper.cpp:1157`), which is
        # `cudaMemsetAsync(ptr, 0, Buffer.Size() * sizeof(T))` (`:34-38`). A
        # WHOLE-BUFFER zero, sized by the block's own mapping, and it is
        # deliberate on their side: line 1155 is
        # `FillBuffer(blockHistograms, 0.0f, streamId)` COMMENTED OUT and
        # replaced by this one.
        #
        # This was `zero_histograms_kernel`, which is `ZeroHistogramsImpl`,
        # the kernel they point at the FINAL FLAT histogram and never at the
        # scratch. It happened to cover the same cells here, because
        # `blocks_for` gives every feature `group_offset = 0` and
        # `group_size = total_folds` (`gpu_data/feature_blocks.mojo`), which
        # is what CatBoost itself produces on ONE DEVICE, so `device_offset`
        # is zero and the scratch is plain `[leaf][stat][binInBlock]`. The
        # two agree today and stop agreeing the moment `group_offset` is
        # nonzero, which is the moment a second device exists. Their call is
        # the whole-buffer one; this is now their call.
        #
        # AND IT IS A MEMSET, NOT A KERNEL. Their `ZeroBuffer` is
        # `cudaMemsetAsync` (`split_properties_helper.cpp:34-38`); this port
        # launched a grid-stride kernel for it, which measured ~1.3 ms per
        # level against `ctx.enqueue_memset`'s 0.2 ms for the same 3.26M
        # cells (64 GB/s). The memset covers the WHOLE scratch rather than
        # `block_cells` of it; that is their whole-buffer semantic exactly,
        # and the cells past `block_cells` are ones no kernel writes and the
        # bridge never reads.
        ctx.enqueue_memset(block_hist, Float32(0.0))

        if blk.policy == POLICY_BINARY:
            if depth == 0:
                ctx.enqueue_function[binary_hist_kernel](
                    blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
                    blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
                    Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line), Int32(base),
                    stats.unsafe_ptr(), Int32(n_rows),
                    p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
                    block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                    Int32(max_leaves), Int32(stat_count),
                    grid_dim=(groups * replicas, n_live, stat_count),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
            else:
                ctx.enqueue_function[binary_hist_gather_kernel](
                    blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
                    blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
                    Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line), Int32(base),
                    row_index.unsafe_ptr(),
                    stats.unsafe_ptr(), Int32(n_rows),
                    p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
                    block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                    Int32(max_leaves), Int32(stat_count),
                    grid_dim=(groups * replicas, n_live, stat_count),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
        elif blk.policy == POLICY_HALF_BYTE:
            # The half-byte kernels now carry the same fixed-point Int32
            # flush the binary ones do, so replicated blocks sum their
            # partials through an integer atomic instead of overwriting each
            # other. That is what makes `* replicas` legal here: their
            # `numBlocks.x = ceil(fCount / GroupSize)` times
            # `CeilDivide(maxActiveBlocks, ...)` (`hist_half_byte.cu:81`).
            #
            # AND IT IS LEGAL ONLY WITH A BOUNDED `fixed_scale`. The integer
            # flush is exact, not forgiving: `Int32(val * fixed_scale)` wraps
            # silently once `fixed_scale` is too large for the cell. Turning
            # replication on here is what first made this path reachable from
            # `doc_parallel_boosting.fit`, which was passing zero for both
            # magnitudes -- see the WHY THIS EXISTS block there. A kernel
            # that only ever ran unreplicated hides its caller's scale bug.
            #
            # ============ THE `* replicas` WAS REMOVED AND IS BACK ============
            # It was taken out with the finding "our fixed-point Int32
            # stand-in for that atomic is WRONG in this kernel", citing the
            # depth-0 weight histogram coming back as -1.5e-08, -3.0e-08,
            # -4.5e-08, -6.0e-08 where it should read 550, 1104, 1651, 2237.
            #
            # THOSE FOUR NUMBERS ARE NOT NOISE AND THEY ARE NOT THIS KERNEL.
            # They are, to every digit given, -4, -8, -12, -16 divided by
            # 268,435,455:
            #
            #     -4  / 268435455 = -1.4901e-08
            #     -8  / 268435455 = -2.9802e-08
            #     -12 / 268435455 = -4.4703e-08
            #     -16 / 268435455 = -5.9605e-08
            #
            # 268,435,455 is `fixed_scale` when `mag` is 1.0, which is what
            # the old scale derivation returned for the `0.0, 0.0` that
            # `fit` was passing. At that scale `val * fixed_scale` is about
            # 3.7e10 for a 137-row partial, the conversion saturates at
            # INT32_MAX, and the FOUR active blocks of a 8,192-row partition
            # sum to 4 * 2147483647 = 8589934588, which as Int32 is exactly
            # -4. The prefix scan then walks it out to -8, -12, -16.
            #
            # So the kernel added its four partials correctly; the SCALE it
            # was handed was the largest the type admits. With
            # `choose_scale` bounding it by the sum of magnitudes, the
            # full-dataset sum maps to at most 2^28 - 1 and a partial is a
            # subset of it, so this cannot recur by construction.
            #
            # Measured with the bound in place: boosting 0.61 mse, better
            # than the 0.66 that stood before any of this landed.
            # `mojo_only/replicated_half_byte_check.mojo` is the standing
            # cover for this arm; it compares a REPLICATED half-byte
            # histogram against a host tally and sabotages the scale on
            # purpose so it fails rather than passes if it stops reaching
            # the flush.
            # ==================================================================
            if depth == 0:
                ctx.enqueue_function[half_byte_hist_kernel](
                    blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
                    blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
                    Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line), Int32(base),
                    stats.unsafe_ptr(), Int32(n_rows),
                    p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
                    block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                    Int32(max_leaves), Int32(stat_count),
                    grid_dim=(groups * replicas, n_live, stat_count),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
            else:
                ctx.enqueue_function[half_byte_hist_gather_kernel](
                    blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
                    blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
                    Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line), Int32(base),
                    row_index.unsafe_ptr(),
                    stats.unsafe_ptr(), Int32(n_rows),
                    p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
                    block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                    Int32(max_leaves), Int32(stat_count),
                    grid_dim=(groups * replicas, n_live, stat_count),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
        else:
            # THEIR maxBins LADDER, `ComputeHistOneByte`
            # (`hist_one_byte.cu:314-328`):
            #
            #     maxBins <= 32   HIST2_PASS(5)
            #     maxBins <= 64   HIST2_PASS(6)
            #     maxBins <= 128  HIST2_PASS(7)
            #     maxBins <= 255  PASS(8, numStats)
            #
            # Everything up to 128 bins -- their own GPU default border
            # count included -- runs the FUSED TWO-STAT `TPointHist2OneByte`
            # family; only 129-255 runs the one-stat `TPointHistOneByte`
            # PASS family. This dispatch routed ALL one-byte shapes through
            # the PASS family until 2026-08-19, a wrong-kernel-family misport
            # of exactly the shape PORTING_RULES 0b-i names: the ported
            # kernel was faithful and their dispatch never sends
            # `maxBins <= 128` to it. `mojo_only/hist2_check.mojo` covers
            # both families on the same input and fingerprints WHICH family
            # this dispatch launched.
            #
            # `bits` must MATCH the block's fold count in either family: it
            # decides the slot arithmetic and the skip value, so a width
            # wider than the data changes where bins land. A byte-level
            # probe established that for the PASS family (64-fold block at
            # 8 bits: 2 of 4 features wrong).
            if blk.max_folds <= 32:
                launch_hist2_one_byte[5, hist2_smem_mode](
                    ctx, blk, depth, n_live, n_rows, stat_count, max_leaves,
                    sm_count, line, base, cindex, row_index, stats, p_off,
                    p_sz, ids, block_hist, acc_i32, fixed_scale,
                )
            elif blk.max_folds <= 64:
                launch_hist2_one_byte[6, hist2_smem_mode](
                    ctx, blk, depth, n_live, n_rows, stat_count, max_leaves,
                    sm_count, line, base, cindex, row_index, stats, p_off,
                    p_sz, ids, block_hist, acc_i32, fixed_scale,
                )
            elif blk.max_folds <= 128:
                launch_hist2_one_byte[7, hist2_smem_mode](
                    ctx, blk, depth, n_live, n_rows, stat_count, max_leaves,
                    sm_count, line, base, cindex, row_index, stats, p_off,
                    p_sz, ids, block_hist, acc_i32, fixed_scale,
                )
            else:
                launch_one_byte[8, hist2_smem_mode](
                    ctx, blk, depth, n_live, n_rows, stat_count, max_leaves,
                    sm_count, line, base, cindex, row_index, stats, p_off,
                    p_sz, ids, block_hist, acc_i32, fixed_scale,
                )

        # THE BRIDGE. Scatter this block's slice into the flat histogram the
        # score kernel reads. `skip_bridge` leaves the result in the
        # per-block layout so a probe can read what the KERNEL wrote rather
        # than what the bridge moved, which is the only way to tell a bad
        # accumulation from a bad scatter. Without it every policy after the first lands
        # on the previous one's cells.
        if skip_bridge:
            block_first_bin += blk.total_folds
            continue

        # The fixed-point accumulator is the twin of `block_hist`, so it is
        # converted HERE, per block, before the bridge scatters the block
        # into the flat histogram. Converting it into the flat histogram
        # instead skips the bridge entirely, which is only correct when one
        # block spans everything.
        #
        # AND IT DOES NOT RUN AT ALL UNDER `NUMERIC_FAST`. The FAST build
        # flushes through CatBoost's float atomic and never writes
        # `acc_i32`, so this kernel would read 3.26M zeros and store
        # nothing -- measured at ~7 ms/tree of pure reading, with no
        # CatBoost counterpart (their words in its own docstring). The
        # branch is comptime, same truth the kernels' flush branches read,
        # so the IDENTICAL build keeps the conversion and the FAST build
        # never launches it.
        comptime _flush_is_fixed_point = deterministic_flush_for[
            TARGET_COLUMN, HIST_BUILD_MODE == NUMERIC_IDENTICAL
        ]()

        # BOTH one-byte families under `HIST_SMEM_SHARED2_I32` drain their
        # replicated flush through the Int32 accumulator EVEN UNDER FAST
        # (see the DEVIATION BLOCKs in `hist2_add_to_global_memory` and
        # `hist_one_byte.mojo`'s writeback), so the conversion must run for
        # every `POLICY_ONE_BYTE` block -- hist_2 at `maxBins <= 128` and
        # the PASS family at 129-255 alike -- and for nothing else, keeping
        # the FAST float path's measured ~7 ms/tree of accumulator reading
        # off the binary/half-byte blocks.
        var run_fixed_bridge = _flush_is_fixed_point

        @parameter
        if (
            hist2_smem_mode == HIST_SMEM_SHARED2_I32
            and not _flush_is_fixed_point
        ):
            if blk.policy == POLICY_ONE_BYTE:
                run_fixed_bridge = True

        if run_fixed_bridge:
            ctx.enqueue_function[fixed_to_float_kernel](
                acc_i32.unsafe_ptr(),
                block_hist.unsafe_ptr(),
                Int32(block_cells),
                fixed_scale,
                grid_dim=(block_cells + 255) // 256,
                block_dim=256,
            )

        # ============ THEIR `ReduceScatter` HAS NO COUNTERPART HERE ========
        # Between the histogram kernel and this writeback CatBoost runs
        #
        #     auto reducedMapping =
        #         ComputeByBlocksHelper.ReducedBlockHistogramsMapping(
        #             blockId, leavesCount, statsCount);
        #     ReduceScatter(blockHistograms, reducedMapping, false, streamId);
        #
        # (`split_properties_helper.cpp:1217-1221`). It is an INTER-DEVICE
        # all-reduce over a stripe mapping: each GPU built the histogram for
        # its own stripe of rows, and this sums the partials across GPUs and
        # scatters the result so each device owns a slice of bin-features.
        # `BlockHistogramsMapping` is the before-reduce shape and
        # `ReducedBlockHistogramsMapping` the after-reduce one
        # (`compute_by_blocks_helper.h:74-85`).
        #
        # Ours does nothing here, and that is theirs too on one device:
        #
        #     if (devCount == 1) { return *this; }
        #
        # (`cuda_lib/cuda_buffer_helpers/reduce_scatter.h:460-462`, inside
        # `TReducer::operator()`). This port is single device, so the call
        # would return immediately and the before-reduce and after-reduce
        # mappings are the same buffer. Multi-device is the point at which
        # this becomes a real gap, not a no-op, and it is where the port
        # would need `TReducer` from `gpu_lib`.
        # ==================================================================
        ctx.enqueue_function[write_reduces_histograms_kernel](
            Int32(block_first_bin),
            Int32(blk.total_folds),
            ids.unsafe_ptr(),
            block_hist.unsafe_ptr(),
            Int32(hist_cells_per_leaf),
            hist.unsafe_ptr(),
            grid_dim=((blk.total_folds + 127) // 128, n_live, stat_count),
            block_dim=(128, 1, 1),
        )
        block_first_bin += blk.total_folds


@fieldwise_init
struct SplitChoice(Copyable, Movable):
    """A winning bin-feature, resolved back to (feature, bin).

    The score kernel returns an index into the FLAT histogram, which spans
    every policy. Turning that into a split needs the owning feature and the
    bin within it, which `first_fold_index` makes an O(features) host walk.
    CatBoost does the same resolution on the host (`ToSplit` in
    `greedy_search_helper.cpp`).
    """

    var bin_feature: Int
    var feature: Int
    var bin: Int


def resolve_split(
    layout: CompressedIndexLayout, bin_feature: Int
) raises -> SplitChoice:
    """Which feature owns this bin-feature, and which of its bins is it.

    O(features) on the host, which is the right side of the boundary: it
    scales with the TREE's feature count, never with rows. See
    HOST_AND_DEVICE.md.
    """
    for i in range(len(layout.features)):
        ref f = layout.features[i]
        if f.folds == 0:
            continue
        var lo = Int(f.first_fold_index)
        var hi = lo + Int(f.folds)
        if bin_feature >= lo and bin_feature < hi:
            return SplitChoice(bin_feature, i, bin_feature - lo)
    raise Error(
        "bin-feature "
        + String(bin_feature)
        + " belongs to no feature; the histogram and the layout disagree"
    )


def run_tree_layout[
    hist2_smem_mode: Int = HIST2_SMEM_MODE
](
    ctx: DeviceContext,
    n_rows: Int,
    fold_counts: List[Int],
    max_depth: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut cursor: DeviceBuffer[DType.float32],
    weight_magnitude: Float32,
    gradient_magnitude: Float32,
    mut out_splits: List[TBinarySplit],
    mut out_leaf_values: List[Float32],
    use_subtraction: Bool = True,
    apply_to_cursor: Bool = False,
    learning_rate: Float32 = Float32(0.3),
    l2_leaf_reg: Float32 = Float32(3.0),
    sync_budget: Int = -1,
    one_hot: List[Bool] = List[Bool](),
    score_function: Int = SCORE_FUNCTION_COSINE,
) raises -> List[Int]:
    """`FitImpl` over a LAYOUT: mixed feature widths, one launch per policy.

    Same level sequence as `run_tree`, which stays as the verified uniform
    binary path. What differs is that the histogram comes from
    `launch_histograms_for_blocks` and the winning bin-feature is resolved
    through `resolve_split`, so a dataset may mix binary, half-byte and
    one-byte features.

    Returns the final leaf sizes, which must sum to `n_rows` at every depth.

    `weight_magnitude` and `gradient_magnitude` are `sum over all rows of
    abs(plane)`, ONE PER STAT PLANE, not the signed totals. They are the
    whole safety argument for the fixed-point flush `NUMERIC_IDENTICAL` puts
    in place of CatBoost's float atomic: every partial sum the device forms
    is over a SUBSET of the rows, so bounding the full-dataset sum of
    magnitudes bounds every Int32 slot at every depth. A signed total is NOT
    that bound -- gradients cancel, so it can be arbitrarily smaller than the
    largest cell -- and passing zero asks for the largest scale the type
    admits. See `mojo_only/fixed_point.mojo`.
    """
    var stat_count = 2
    var max_leaves = 1 << max_depth

    var layout = build_layout(fold_counts, one_hot)
    var blocks = blocks_for(layout, n_rows)
    var dblocks = upload_blocks(ctx, blocks)
    var hist_cells_per_leaf = layout.hist_cells

    # their `TArchProps::SMCount()`, read from the device rather than
    # guessed. `replication_for` turns it into their grid x factor.
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    var wide = (n_rows + 255) // 256
    if wide > 256:
        wide = 256
    if wide < 1:
        wide = 1

    var p_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var hp_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var hp_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(0))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_sz, src_ptr=h_sz.unsafe_ptr())

    var hist_cells = max_leaves * stat_count * hist_cells_per_leaf
    var hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
    # their `FillBuffer(subsets.Histograms, 0.0f)` in `CreateInitialSubsets`
    # (`split_properties_helper.cpp:1061`). `zero_histograms_kernel` clears
    # only `plan.compute_ids`, by design, so every other slot holds whatever
    # the allocation came with until the bridge writes it, and the bridge
    # writes only the cells the kernels produced.
    ctx.enqueue_memset(hist, Float32(0.0))

    # Scratch for the per-block layout. Sized to the LARGEST block, since the
    # blocks are written one at a time and scattered before the next runs.
    var widest_block = 1
    for b in range(len(blocks)):
        var tf = 0
        for k in range(blocks[b].count()):
            tf += Int(blocks[b].folds[k])
        if tf > widest_block:
            widest_block = tf
    var block_hist = ctx.enqueue_create_buffer[DType.float32](
        max_leaves * stat_count * widest_block
    )
    var acc_i32 = ctx.enqueue_create_buffer[DType.int32](hist_cells)
    ctx.enqueue_memset(acc_i32, Int32(0))

    # See the note in `run_tree`: one scale serves both accumulated planes,
    # so the bound is the larger of the two sums of magnitudes, and
    # `choose_scale` owns the derivation rather than a fourth copy of it.
    var mag = Float64(weight_magnitude)
    if mag < 0.0:
        mag = -mag
    var gmag = Float64(gradient_magnitude)
    if gmag < 0.0:
        gmag = -gmag
    if gmag > mag:
        mag = gmag
    var fixed_scale = Float32(choose_scale(mag, n_rows))

    var part_stats = ctx.enqueue_create_buffer[DType.float32](
        max_leaves * stat_count
    )
    # SIZED BY THE MACHINE CHUNK COUNT, NOT THE DATA. The kernels index
    # `(leaf * n_stats + stat) * max_chunks + chunk` with `max_chunks =
    # partition_stats_chunks(sm_count, n_stats)` -- their machine-sized
    # grid -- and this buffer was still sized by `ceil(n_rows /
    # STATS_BLOCK)` from the data-sized era. Whenever the data count was
    # SMALLER (every fit under ~5k rows on this box), the deep-level
    # writes ran past the end: leaf 15, stat 1, chunk 9 lands at slot 319
    # of a 256-float buffer at depth 6 on 4096 rows. The scribble was
    # SELF-CONSISTENT (phase 2 reads the same out-of-bounds slots), so
    # the 4096-row oracle fixtures passed over it for days; it surfaced
    # as NaN leaf values only when the allocator placed something
    # volatile after the buffer (caught by the train-api check,
    # 2026-08-21). The larger of the two counts is always safe.
    var stat_chunks = (n_rows + STATS_BLOCK - 1) // STATS_BLOCK
    var machine_chunks = partition_stats_chunks(sm_count, stat_count)
    if machine_chunks > stat_chunks:
        stat_chunks = machine_chunks
    var stat_partials = ctx.enqueue_create_buffer[DType.float32](
        max_leaves * stat_count * stat_chunks
    )

    var flags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var seq = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var gmap = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var sflags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var new_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var new_stats = ctx.enqueue_create_buffer[DType.float32](
        stat_count * n_rows
    )

    var max_chunks = (n_rows + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    var chunk_zeros = ctx.enqueue_create_buffer[DType.uint32](
        max_leaves * max_chunks
    )
    var chunk_offsets = ctx.enqueue_create_buffer[DType.uint32](
        max_leaves * max_chunks
    )
    var leaf_zeros = ctx.enqueue_create_buffer[DType.uint32](max_leaves)

    # `ids_a` and `ids_b` are gone from this function: their content was
    # ALWAYS the dense sequence `0..n_live-1`, which `dense_ids` already
    # holds -- their `leavesToSplit` for a symmetric tree is every live
    # leaf, and their `leftIds` keep the parents' slots. `ids_c` (their
    # `rightIds`, `leavesCount + i`) is written per level by
    # `resolve_and_pack_kernel`.
    var ids_c = ctx.enqueue_create_buffer[DType.uint32](max_leaves)

    # their `zeroLeaves` (`split_properties_helper.cpp:1344`) and the mirror
    # buffer `ZeroLeavesHistograms` writes it into (`:1391-1392`). It is a
    # buffer of its own and not a second copy of `h_ids_compute`, because
    # `enqueue_copy` is asynchronous: overwriting one host staging buffer to
    # feed two device buffers races the first copy.
    var zero_ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var h_zero_ids = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)

    # `0, 1, 2, ...`, the DENSE index list. Their histogram kernels write the
    # block scratch at `blockIdx.y`, so anything addressing that scratch by
    # position needs this rather than the leaf-id list.
    var dense_ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var h_dense = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        h_dense.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=dense_ids, src_ptr=h_dense.unsafe_ptr())

    # their weak model's leaf values, one per final leaf
    var leaf_values = ctx.enqueue_create_buffer[DType.float32](max_leaves)
    var h_leaf_values = ctx.enqueue_create_host_buffer[DType.float32](
        max_leaves
    )

    # their `computeLeaves`, `bigLeaves`, `smallLeaves`
    # (`split_properties_helper.cpp:1287-1291`)
    var ids_compute = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var h_ids_compute = ctx.enqueue_create_host_buffer[DType.uint32](
        max_leaves
    )
    var sub_from = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var sub_what = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var h_sub_from = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var h_sub_what = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)

    var skip = ctx.enqueue_create_buffer[DType.uint8](hist_cells_per_leaf)
    var hsk = ctx.enqueue_create_host_buffer[DType.uint8](
        hist_cells_per_leaf
    )
    for i in range(hist_cells_per_leaf):
        hsk.unsafe_ptr().unsafe_store(i, UInt8(0))
    ctx.enqueue_copy(dst_buf=skip, src_ptr=hsk.unsafe_ptr())

    # their `argmaxBlockCount = Min(CeilDivide(binFeatureCountPerDevice,
    # 256), 64)` (`greedy_search_helper.cpp:439`) and `bestProps` sized
    # `argmaxBlockCount * numScoreBlocks` (`:440`), `numScoreBlocks == 1`
    # for a symmetric tree. This launch ran at grid (1,1,1) -- ONE
    # threadgroup walking every candidate -- and measured 11 ms/tree of the
    # 41.7 ms fixed cost at 800k x 100 depth 6. The block decomposition
    # cannot change the winner: the tie-break (gain, then bin-feature index,
    # their `TBestSplitProperties::operator<`, `gpu_structures.h:80-94`) is
    # a total order over candidates, so this is a SCHEDULING row, not a
    # numeric one.
    var argmax_blocks = (hist_cells_per_leaf + 255) // 256
    if argmax_blocks > 64:
        argmax_blocks = 64
    if argmax_blocks < 1:
        argmax_blocks = 1
    var out_score = ctx.enqueue_create_buffer[DType.float32](argmax_blocks)
    var out_bin = ctx.enqueue_create_buffer[DType.uint32](argmax_blocks)

    # The per-level winner records `resolve_and_pack_kernel` writes and the
    # host reads back in the SAME drain as the leaf sizes -- the fold that
    # takes a level from two blocking reads to one (DEVIATION BLOCK in
    # `kernel/split_resolve.mojo`; their level reads `bestProps` and the
    # sizes separately, `greedy_search_helper.cpp:517` + `:800-813`).
    var winners_score = ctx.enqueue_create_buffer[DType.float32](max_depth)
    var winners_bf = ctx.enqueue_create_buffer[DType.uint32](max_depth)
    var h_wsc = ctx.enqueue_create_host_buffer[DType.float32](max_depth)
    var h_wbf = ctx.enqueue_create_host_buffer[DType.uint32](max_depth)

    # Their `MakeSplit` packs every ui32 it has to send into ONE buffer and
    # copies once (`split_properties_helper.cpp:886-905`):
    #
    #     const TSlice binsSlice      = TSlice(0, splitBins.size());
    #     const TSlice leftIdsSlice   = TSlice(binsSlice.Right, ...);
    #     const TSlice rightIdsSlice  = TSlice(leftIdsSlice.Right, ...);
    #     auto allUi32Data = TMirrorBuffer<ui32>::Create(...);
    #     allUi32DataCpu.Write(tmp);
    #     allUi32Data.Copy(allUi32DataCpu);
    #
    # Ours holds the four ui32 fields of the split descriptor the same way.
    # It used to be four separate buffers and four separate copies followed
    # by a drain. The one-hot flag stays its own buffer because it is UInt8,
    # which is also why theirs keeps `splitFeaturesGpu` separate from
    # `allUi32Data`.
    var built_f = make_split_features_buffers(ctx, max_leaves)
    var sp_feats = built_f[0]
    var sp_feats_h = built_f[1]
    var sp_bins = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var sp_bins_h = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)

    # The scan needs each feature's first bin and fold count over the FLAT
    # histogram, which is exactly what the layout holds.
    var flat_first = ctx.enqueue_create_buffer[DType.uint32](
        len(fold_counts)
    )
    var flat_folds = ctx.enqueue_create_buffer[DType.uint32](
        len(fold_counts)
    )
    # ...and their `TBinarizedFeature::OneHotFeature`, which the scan tests
    # before it prefix-sums anything (`histogram_utils.cu:395`). A one-hot
    # feature's bins are equality tests, so a running prefix over them is
    # meaningless. `build_layout` reports False for every feature today; the
    # flag is carried so the scan is already right when categoricals land.
    var flat_one_hot = ctx.enqueue_create_buffer[DType.uint8](
        len(fold_counts)
    )
    var hff = ctx.enqueue_create_host_buffer[DType.uint32](len(fold_counts))
    var hfd = ctx.enqueue_create_host_buffer[DType.uint32](len(fold_counts))
    var hfoh = ctx.enqueue_create_host_buffer[DType.uint8](len(fold_counts))
    for i in range(len(fold_counts)):
        hff.unsafe_ptr().unsafe_store(i, layout.features[i].first_fold_index)
        hfd.unsafe_ptr().unsafe_store(i, layout.features[i].folds)
        hfoh.unsafe_ptr().unsafe_store(
            i, UInt8(1) if layout.features[i].one_hot_feature else UInt8(0)
        )
    ctx.enqueue_copy(dst_buf=flat_first, src_ptr=hff.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=flat_folds, src_ptr=hfd.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=flat_one_hot, src_ptr=hfoh.unsafe_ptr())

    # THEIR `BinFeatures` AND `FeatureWeights`, WHICH ARE PER-TREE.
    # `bff` is the `TCBinFeature.FeatureId` column (`compute_scores.cu:136`),
    # here the flattened bin-feature to feature map the layout defines; `ffw`
    # is their `binFeaturesWeights`, all 1.0 because `UpdateFeatureWeights`
    # fills 1.0 and returns early with no CTRs
    # (`update_feature_weights.cpp:14-22`) -- their value, not a stub.
    #
    # `CreateInitialSubsets` builds both once
    # (`split_properties_helper.cpp:1063`, `:1075-1076`) and
    # `structure_searcher_template.h:49` runs it once before the level loop
    # at `:55-65`. `ComputeOptimalSplits` reads them at
    # `greedy_search_helper.cpp:455-456` and allocates neither; the one call
    # that touches them per level, `UpdateFeatureWeightsForBestSplits`
    # (`:447`), returns immediately when there are no CTRs
    # (`update_feature_weights.cpp:20-22`).
    #
    # Both used to be created INSIDE the depth loop, along with their host
    # staging, and the `bff` fill re-walked every feature and every one of
    # its folds at every level to write the same map. Neither value depends
    # on `depth`, and `layout` is fixed for the whole tree.
    #
    # LIFETIME. Nothing writes `bff` or `ffw` after the two copies below, so
    # no level can read a stale or half-written value. `hbf` and `hfw` are
    # the host staging for those copies and stay in scope for the whole
    # function, so the asynchronous copy cannot outlive its source; the
    # `ctx.synchronize()` below settles both before the first level.
    var bff = ctx.enqueue_create_buffer[DType.uint32](hist_cells_per_leaf)
    var hbf = ctx.enqueue_create_host_buffer[DType.uint32](
        hist_cells_per_leaf
    )
    var ffw = ctx.enqueue_create_buffer[DType.float32](len(fold_counts))
    var hfw = ctx.enqueue_create_host_buffer[DType.float32](len(fold_counts))
    for f in range(len(fold_counts)):
        ref lf = layout.features[f]
        for b in range(Int(lf.folds)):
            hbf.unsafe_ptr().unsafe_store(
                Int(lf.first_fold_index) + b, UInt32(f)
            )
    for f in range(len(fold_counts)):
        hfw.unsafe_ptr().unsafe_store(f, Float32(1.0))
    ctx.enqueue_copy(dst_buf=bff, src_ptr=hbf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ffw, src_ptr=hfw.unsafe_ptr())

    # THE PER-BIN-FEATURE TCFeature TABLE for `resolve_and_pack_kernel`:
    # what the host's `resolve_split` walk + `MakeSplit` pack derived per
    # level, precomputed once per tree because none of it depends on
    # `depth`. Seven parallel arrays indexed by FLAT BIN-FEATURE; `offset`
    # is pre-scaled by `n_rows` exactly as the old host pack scaled it.
    var bfr_off = ctx.enqueue_create_buffer[DType.uint32](hist_cells_per_leaf)
    var bfr_mask = ctx.enqueue_create_buffer[DType.uint32](hist_cells_per_leaf)
    var bfr_shift = ctx.enqueue_create_buffer[DType.uint32](hist_cells_per_leaf)
    var bfr_first = ctx.enqueue_create_buffer[DType.uint32](hist_cells_per_leaf)
    var bfr_folds = ctx.enqueue_create_buffer[DType.uint32](hist_cells_per_leaf)
    var bfr_oh = ctx.enqueue_create_buffer[DType.uint8](hist_cells_per_leaf)
    var bfr_bin = ctx.enqueue_create_buffer[DType.uint32](hist_cells_per_leaf)
    var hbr1 = ctx.enqueue_create_host_buffer[DType.uint32](hist_cells_per_leaf)
    var hbr2 = ctx.enqueue_create_host_buffer[DType.uint32](hist_cells_per_leaf)
    var hbr3 = ctx.enqueue_create_host_buffer[DType.uint32](hist_cells_per_leaf)
    var hbr4 = ctx.enqueue_create_host_buffer[DType.uint32](hist_cells_per_leaf)
    var hbr5 = ctx.enqueue_create_host_buffer[DType.uint32](hist_cells_per_leaf)
    var hbr6 = ctx.enqueue_create_host_buffer[DType.uint8](hist_cells_per_leaf)
    var hbr7 = ctx.enqueue_create_host_buffer[DType.uint32](hist_cells_per_leaf)
    for f in range(len(fold_counts)):
        ref tf = layout.features[f]
        for b in range(Int(tf.folds)):
            var bfx = Int(tf.first_fold_index) + b
            hbr1.unsafe_ptr().unsafe_store(bfx, tf.offset * UInt32(n_rows))
            hbr2.unsafe_ptr().unsafe_store(bfx, tf.mask)
            hbr3.unsafe_ptr().unsafe_store(bfx, tf.shift)
            hbr4.unsafe_ptr().unsafe_store(bfx, tf.first_fold_index)
            hbr5.unsafe_ptr().unsafe_store(bfx, tf.folds)
            hbr6.unsafe_ptr().unsafe_store(
                bfx, UInt8(1) if tf.one_hot_feature else UInt8(0)
            )
            hbr7.unsafe_ptr().unsafe_store(bfx, UInt32(b))
    ctx.enqueue_copy(dst_buf=bfr_off, src_ptr=hbr1.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=bfr_mask, src_ptr=hbr2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=bfr_shift, src_ptr=hbr3.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=bfr_first, src_ptr=hbr4.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=bfr_folds, src_ptr=hbr5.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=bfr_oh, src_ptr=hbr6.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=bfr_bin, src_ptr=hbr7.unsafe_ptr())

    ctx.synchronize()

    # ================================================================
    # Their `TGreedyTreeLikeStructureSearcher::FitImpl`
    # (`structure_searcher_template.h:41`):
    #
    #     TPointsSubsets subsets = searchHelper.CreateInitialSubsets(...);
    #     while (true) {
    #         searchHelper.ComputeOptimalSplits(&subsets);
    #         if (!searchHelper.SplitLeaves(&subsets, ...)) break;
    #     }
    #
    # Two phases per level. Theirs blocks the host exactly TWICE in them:
    # `bestProps.Read(propsCpu)` (`greedy_search_helper.cpp:517`) and the
    # leaf-size read in `RebuildLeavesSizes`
    # (`split_properties_helper.cpp:802`). This loop blocks ONCE: the
    # winner is resolved and packed on the device
    # (`kernel/split_resolve.mojo`, DEVIATION BLOCK -- a drain on this box
    # costs ~191 us plus the queue-empty bubble, and the covtype fixed
    # floor those drains sit in is half the tree) and rides home in the
    # same drain as the sizes, with the gates applied post-drain and a
    # one-level rollback on the rare stop.
    #
    # THE BUDGET IS NOT ARMED HERE, AND THAT IS DELIBERATE.
    #
    # `sync_budget` is OURS. CatBoost counts no drains and refuses none
    # (`gpu_single_worker.mojo`, DEVIATION BLOCK). It was written as a
    # debugging aid, because an older driver called `synchronize` nine times
    # per level and nobody could see the total.
    #
    # It was then left ARMED on this path at `2 * max_depth + 1`, which made
    # a counter into a RUNTIME FAILURE MODE OF THE LIBRARY. A fit could raise
    # "sync budget exceeded" for a reason that has nothing to do with the
    # caller's data, their parameters, or their device, and CatBoost cannot
    # fail that way because it does not count. Worse, the number is a
    # prediction about our own future code: any correct change that needs one
    # more drain turns into an exception at a call site that did nothing
    # wrong.
    #
    # A diagnostic belongs where assertions belong. The default is unbounded,
    # the counters stay because they cost nothing and are worth reading, and
    # the CHECKS pass the tight budget so the drain discipline is still
    # enforced against a fixture rather than against a user.
    # ================================================================
    var mgr = TCudaManager(ctx.copy(), sync_budget=sync_budget)

    var n_live = 1
    var max_live_rows = n_rows

    # their `subsets->Leaves`. The root holds every row and its histogram slot
    # is empty, which is `EHistogramsType::Zeroes`.
    var leaf_records = List[LeafRecord]()
    leaf_records.append(
        LeafRecord(UInt32(n_rows), HISTOGRAMS_ZEROES, 0, False)
    )

    for depth in range(max_depth):
        # ============ ComputeOptimalSplits ============
        # their `greedy_search_helper.cpp:398`. Their `leavesToSplit` for a
        # symmetric tree is every live leaf, which is `dense_ids`.

        # their `SplitPropsHelper.BuildNecessaryHistograms(subsets)`
        # (`split_properties_helper.cpp:1283`).
        #
        # THE HALVING. A level does not build `2^depth` histograms. It builds
        # the SMALLER child of each pair and derives the larger as
        # `parent - smaller`, because the larger's slot already holds the
        # parent, put there by `copy_histograms_kernel` at split time (their
        # `CopyHistogram`, `split_points.cpp:326`).
        var plan = build_necessary_histograms(leaf_records)
        var n_pairs = len(plan.subtract_from)

        # ============================================================
        # THE ZERO-LEAF SPLIT, `split_properties_helper.cpp:1342-1350`:
        #
        #     auto nonZeroComputeLeaves =
        #         NonZeroLeaves(*subsets, computeLeaves);
        #     auto zeroLeaves = ZeroLeaves(*subsets, computeLeaves);
        #     ComputeSplitProperties(loadPolicy, nonZeroComputeLeaves, subsets);
        #     ZeroLeavesHistograms(zeroLeaves, subsets);
        #
        # **AN EMPTY LEAF'S HISTOGRAM IS ALL ZEROS BY DEFINITION AND
        # CATBOOST NEVER BUILDS IT.** It zeroes the slot and moves on. This
        # loop used to hand the whole of `computeLeaves` to the histogram
        # builder, so every empty leaf bought a full accumulation over every
        # row and every feature to produce a result known in advance. At
        # depth 6 this port's own checks report 56 of 64 leaves non-empty on
        # one fixture and 46 of 64 on another.
        #
        # This is safe only because `leaf_records[i].size` is REBUILT from
        # the partition sizes at the foot of every level, which is their
        # `RebuildLeavesSizes` (`:800-812`). If it were stale, the partition
        # here would be wrong AND the smaller-child choice at `:1318` would
        # already be being made on bad data.
        # ============================================================
        var compute_nonzero = non_zero_leaves(leaf_records, plan.compute_ids)
        var compute_zero = zero_leaves(leaf_records, plan.compute_ids)
        var n_compute = len(compute_nonzero)
        var n_zero = len(compute_zero)

        # their `ComputeSplitProperties(loadPolicy, nonZeroComputeLeaves,
        # subsets)` (`:1346`), whose first statement is
        # `if (leavesToCompute.size() == 0) { return; }` (`:1089-1091`).
        if n_compute > 0:
            for i in range(n_compute):
                h_ids_compute.unsafe_ptr().unsafe_store(i, compute_nonzero[i])
            ctx.enqueue_copy(
                dst_buf=ids_compute, src_ptr=h_ids_compute.unsafe_ptr()
            )

            launch_histograms_for_blocks[hist2_smem_mode](
                ctx, dblocks, depth, n_compute, n_rows, stat_count, max_leaves,
                sm_count, fixed_scale,
                cindex, row_index, stats, p_off, p_sz, ids_compute, dense_ids,
                hist, acc_i32, block_hist, hist_cells_per_leaf,
            )
            mgr.stream_kernel()

            # their `TScanHistogramsKernel` over `*leavesGpu` (`:1262`), the
            # LAST statement of `ComputeSplitProperties` and therefore over
            # the NON-ZERO set too. It runs on the leaves JUST COMPUTED and
            # BEFORE the subtraction. A prefix sum is linear, so
            # `scan(parent) - scan(small)` is already `scan(big)` and the
            # derived child needs no further scan.
            # `numBlocks.x = CeilDivide(fCount * 32, blockSize)`
            # (`histogram_utils.cu:442`), theirs at one WARP per feature.
            # Ours is one THREAD per feature, so the same rule is
            # `ceil(fCount / blockSize)`. The kernel guards with
            # `feature_id >= feature_count` and has no grid-stride loop, so
            # grid x of 1 left every feature from 256 up unscanned, holding
            # raw per-bin counts while the score kernel read them as prefix
            # sums.
            ctx.enqueue_function[scan_histograms_kernel](
                ids_compute.unsafe_ptr(),
                flat_first.unsafe_ptr(), flat_folds.unsafe_ptr(),
                flat_one_hot.unsafe_ptr(),
                Int32(len(fold_counts)), Int32(hist_cells_per_leaf),
                hist.unsafe_ptr(),
                grid_dim=(
                    (len(fold_counts) + 255) // 256, n_compute, stat_count
                ),
                block_dim=(256, 1, 1),
            )
            mgr.stream_kernel()

        # their `ZeroLeavesHistograms(zeroLeaves, subsets)` (`:1348`), the
        # whole of `:1375-1400` including its dispatch, which is the part
        # that is easy to skip:
        #
        #     if (leaves.size() <= 2) { one TZeroHistogramKernel per leaf }
        #     else { ids.Write(leaves); one TZeroHistogramsKernel }
        #
        # The small case avoids the host id-buffer write and its copy, which
        # cost more than the extra launch at one or two ids. It runs AFTER
        # the build and BEFORE the subtraction, and the order matters in one
        # direction: an empty leaf can be the SMALLER child of a pair, and
        # `substract_histograms_kernel` reads its slot.
        if n_zero > 0:
            if n_zero <= 2:
                # `TZeroHistogramKernel`, leaf id BY VALUE, `numBlocks.y = 1`
                # (`histogram_utils.cu:268-284`).
                for i in range(n_zero):
                    ctx.enqueue_function[zero_histogram_kernel](
                        UInt32(compute_zero[i]),
                        Int32(hist_cells_per_leaf),
                        hist.unsafe_ptr(),
                        grid_dim=(
                            (hist_cells_per_leaf + 255) // 256, 1, stat_count
                        ),
                        block_dim=(256, 1, 1),
                    )
                    mgr.stream_kernel()
            else:
                # `ids.Write(leaves)` then `TZeroHistogramsKernel` (`:1391`).
                for i in range(n_zero):
                    h_zero_ids.unsafe_ptr().unsafe_store(i, compute_zero[i])
                ctx.enqueue_copy(
                    dst_buf=zero_ids, src_ptr=h_zero_ids.unsafe_ptr()
                )
                ctx.enqueue_function[zero_histograms_kernel](
                    zero_ids.unsafe_ptr(),
                    Int32(hist_cells_per_leaf),
                    hist.unsafe_ptr(),
                    grid_dim=(
                        (hist_cells_per_leaf + 255) // 256, n_zero, stat_count
                    ),
                    block_dim=(256, 1, 1),
                )
                mgr.stream_kernel()

        # their `SubstractHistograms(bigLeaves, smallLeaves, subsets)`
        # (`:1354`). `from - what`, in place, one launch for all pairs.
        if n_pairs > 0:
            for i in range(n_pairs):
                h_sub_from.unsafe_ptr().unsafe_store(i, plan.subtract_from[i])
                h_sub_what.unsafe_ptr().unsafe_store(i, plan.subtract_what[i])
            ctx.enqueue_copy(dst_buf=sub_from, src_ptr=h_sub_from.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=sub_what, src_ptr=h_sub_what.unsafe_ptr())
            ctx.enqueue_function[substract_histograms_kernel](
                sub_from.unsafe_ptr(), sub_what.unsafe_ptr(),
                Int32(hist_cells_per_leaf), hist.unsafe_ptr(),
                grid_dim=(
                    (hist_cells_per_leaf + 255) // 256, n_pairs, stat_count
                ),
                block_dim=(256, 1, 1),
            )
            mgr.stream_kernel()

        # their `allUpdatedLeaves` loop (`:1359`)
        var updated = plan.updated_ids()
        for i in range(len(updated)):
            leaf_records[Int(updated[i])].histograms_type = (
                HISTOGRAMS_CURRENT_PATH
            )

        # their `AllReduceThroughMaster(subsets->CurrentPartStats(), ...)`
        compute_partition_stats(
            ctx, n_live, max_live_rows, stat_count, n_rows,
            dense_ids, p_off, p_sz, stats, stat_partials, part_stats,
            sm_count=sm_count,
        )
        mgr.stream_kernel()

        # `bff` and `ffw` are their `subsets->BinFeatures` and
        # `subsets->FeatureWeights`, built once above this loop because that
        # is where `CreateInitialSubsets` builds them
        # (`split_properties_helper.cpp:1063`, `:1075-1076`).
        # `ComputeOptimalSplits` reads them as its two arguments at
        # `greedy_search_helper.cpp:455-456` and allocates neither. NOTHING
        # in this loop is allocated any more.
        # `switch (scoreFunction)` (`compute_scores.cu:201-219`): their
        # runtime option selects the calcer; ours selects the comptime
        # kernel arm. Cosine is their GPU default and pairs with
        # NewtonCosine onto one calcer, L2 with NewtonL2 onto the other,
        # exactly as the kernel's docstring lays out.
        if (
            score_function == SCORE_FUNCTION_L2
            or score_function == SCORE_FUNCTION_NEWTON_L2
        ):
            ctx.enqueue_function[
                compute_optimal_splits_kernel[SCORE_FUNCTION_L2]
            ](
                skip.unsafe_ptr(), Int32(hist_cells_per_leaf),
                bff.unsafe_ptr(), ffw.unsafe_ptr(),
                hist.unsafe_ptr(),
                part_stats.unsafe_ptr(), Int32(stat_count), dense_ids.unsafe_ptr(),
                Int32(n_live), l2_leaf_reg,
                out_score.unsafe_ptr(), out_bin.unsafe_ptr(),
                grid_dim=(argmax_blocks, 1, 1),
                block_dim=(SCORE_BLOCK_SIZE, 1, 1),
            )
        else:
            ctx.enqueue_function[
                compute_optimal_splits_kernel[SCORE_FUNCTION_COSINE]
            ](
                skip.unsafe_ptr(), Int32(hist_cells_per_leaf),
                bff.unsafe_ptr(), ffw.unsafe_ptr(),
                hist.unsafe_ptr(),
                part_stats.unsafe_ptr(), Int32(stat_count), dense_ids.unsafe_ptr(),
                Int32(n_live), l2_leaf_reg,
                out_score.unsafe_ptr(), out_bin.unsafe_ptr(),
                grid_dim=(argmax_blocks, 1, 1),
                block_dim=(SCORE_BLOCK_SIZE, 1, 1),
            )
        mgr.stream_kernel()

        # THE WINNER IS RESOLVED ON THE DEVICE and the split descriptors
        # are packed there too, so the split chain below is enqueued
        # WITHOUT a host read: their `bestProps.Read` + host block-winner
        # reduce (`greedy_search_helper.cpp:517-529`) + `MakeSplit`'s pack
        # and upload (`split_properties_helper.cpp:872-905`) become one
        # kernel, and this level's ONLY blocking read is the combined
        # drain below. DEVIATION BLOCK in `kernel/split_resolve.mojo`:
        # same sequential reduce order, same tie rule (higher score, tie
        # to the smaller bin-feature), same descriptors; the gates their
        # host applies BEFORE splitting are applied by ours AFTER the
        # drain, with a one-level rollback on the rare stop.
        ctx.enqueue_function[resolve_and_pack_kernel](
            out_score.unsafe_ptr(), out_bin.unsafe_ptr(),
            Int32(argmax_blocks),
            bfr_off.unsafe_ptr(), bfr_mask.unsafe_ptr(),
            bfr_shift.unsafe_ptr(), bfr_first.unsafe_ptr(),
            bfr_folds.unsafe_ptr(), bfr_oh.unsafe_ptr(),
            bfr_bin.unsafe_ptr(),
            Int32(depth), Int32(n_live),
            winners_score.unsafe_ptr(), winners_bf.unsafe_ptr(),
            sp_feats.unsafe_ptr(), sp_bins.unsafe_ptr(),
            ids_c.unsafe_ptr(),
            grid_dim=(1, 1, 1),
            block_dim=(RESOLVE_BLOCK_SIZE, 1, 1),
        )
        mgr.stream_kernel()


        # `numBlocks.x = (leavesCount > 4 ? 2 : 4) * TArchProps::SMCount()`
        # (`split_points.cu:563`): MACHINE-sized, like every strided grid in
        # their file; `wide` was data-sized and is not what their grid x
        # means.
        ctx.enqueue_function[split_and_make_sequence_kernel](
            cindex.unsafe_ptr(), row_index.unsafe_ptr(),
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), dense_ids.unsafe_ptr(),
            sp_feats.unsafe_ptr().bitcast[CFeature](), sp_bins.unsafe_ptr(),
            flags.unsafe_ptr(), seq.unsafe_ptr(),
            grid_dim=(split_points_grid_x(n_live, sm_count), n_live, 1),
            block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )
        mgr.stream_kernel()

        launch_stable_partition(
            ctx, n_live, max_live_rows, dense_ids, p_off, p_sz, flags,
            chunk_zeros, chunk_offsets, leaf_zeros, gmap, sflags,
        )
        mgr.stream_kernel()

        # their `TSplitPointsKernel::Run` (`split_points.cpp:64-136`), the
        # whole if/else in one call because it is one function of theirs.
        #
        # It branches on the largest leaf being split. At or under 1024 rows
        # every leaf is staged in shared memory and permuted in place, ONE
        # launch for the index and every stat column together
        # (`split_points.cpp:113-135`, grid width `1 + statCount` at
        # `split_points.cu:105`). Above it, `CopyInLeaves` then
        # `GatherInLeaves` per chunk of eight stat columns, then one more pair
        # for the index (`split_points.cpp:66-111`). Both arms touch only
        # `[part.Offset, part.Offset + part.Size)`, so rows outside a
        # splitting leaf are neither read nor written.
        #
        # This replaces a gather into a second full plane followed by a copy
        # of the ENTIRE `stat_count * n_rows` buffer: two defects at once. It
        # moved the whole array every level regardless of how much was
        # actually splitting, and the scratch was never initialised, so any
        # row not covered by a live leaf was copied back as garbage. That
        # survived only because this port splits every leaf unconditionally
        # and CatBoost does not (`greedy_search_helper.cpp:691-694`).
        #
        # The launch count is data dependent -- 1 on the fast path,
        # `2 * ceil(stat_count / 8) + 2` on the slow one -- so it comes back
        # from the call rather than being written out here. A hardcoded 4 was
        # counting two launches the fast path never issues.
        var reorder_launches = launch_reorder_in_leaves(
            ctx, n_live, wide, max_live_rows, stat_count, n_rows,
            dense_ids, p_off, p_sz, stats, new_stats, row_index, new_index, gmap,
            sm_count=sm_count,
        )
        for _ in range(reorder_launches):
            mgr.stream_kernel()

        # their `CopyHistogram(LeafIdToSplit, RightLeafIdAfterSplit, ...)`
        # (`split_points.cpp:326`), issued right before the partition update
        # exactly where they issue it.
        #
        # The left child kept the parent's slot so it already holds the parent
        # histogram; this puts the same histogram in the right child's slot.
        # Both children are then `PreviousPath`, which is what lets the next
        # level pair them and derive one by subtraction whichever is smaller.
        ctx.enqueue_function[copy_histograms_kernel](
            dense_ids.unsafe_ptr(), ids_c.unsafe_ptr(),
            Int32(stat_count), Int32(hist_cells_per_leaf),
            hist.unsafe_ptr(),
            grid_dim=(
                (hist_cells_per_leaf * stat_count + 255) // 256, n_live, 1
            ),
            block_dim=(256, 1, 1),
        )
        mgr.stream_kernel()

        # their `UpdatePartitionsAfterSplit` (`split_points.cu:387`), reached
        # with NO host arithmetic in front of it because the left child kept
        # the parent's slot.
        #
        # `hp_off` / `hp_size` are their `partsCpu`
        # (`split_properties_helper.h:49`), which on their side is
        # `EPtrType::CudaHost` and is READ BY THE HOST with no copy at all
        # (`split_points.cpp:56-62`, `split_points.cu:667`). Ours are
        # ordinary device buffers that nothing reads, so the kernel pays
        # their write and we collect none of their benefit. That is a
        # deviation forced by the toolchain, not a choice; the search that
        # established it is the DEVIATION BLOCK in
        # `gbdt/gpu_util/gpu_data/partitions.mojo`.
        # their `:397`, the same machine-sized expression.
        ctx.enqueue_function[update_partitions_after_split_kernel](
            dense_ids.unsafe_ptr(), ids_c.unsafe_ptr(), Int32(n_live),
            sflags.unsafe_ptr(), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            hp_off.unsafe_ptr(), hp_sz.unsafe_ptr(),
            grid_dim=(split_points_grid_x(n_live, sm_count), n_live, 1),
            block_dim=(512, 1, 1),
        )
        mgr.stream_kernel()

        # their `SplitLeaf` (`split_properties_helper.cpp:786`):
        #
        #     if (leaf.HistogramsType == EHistogramsType::CurrentPath)
        #         newLeaf.HistogramsType = EHistogramsType::PreviousPath;
        #
        # Both children of leaf `i` group on the parent's path, which is `i`.
        var prev_records = leaf_records.copy()
        leaf_records = List[LeafRecord]()
        for _ in range(2 * n_live):
            leaf_records.append(
                LeafRecord(UInt32(0), HISTOGRAMS_ZEROES, 0, False)
            )
        for i in range(n_live):
            # `use_subtraction=False` keeps every child at `Zeroes`, so the
            # planner finds no pairs and the level rebuilds all `2^depth`
            # histograms. It exists so the two can be measured INTERLEAVED
            # inside one process; this box drifts 2-3x across time windows
            # and sequential arms do not compare.
            var t = HISTOGRAMS_ZEROES
            if (
                use_subtraction
                and prev_records[i].histograms_type
                == HISTOGRAMS_CURRENT_PATH
            ):
                t = HISTOGRAMS_PREVIOUS_PATH
            leaf_records[i] = LeafRecord(UInt32(0), t, i, False)
            leaf_records[n_live + i] = LeafRecord(UInt32(0), t, i, False)

        n_live = n_live * 2
        ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
        # ...and the winner records ride home in the SAME drain, which is
        # the whole fold: two blocking reads per level become one.
        ctx.enqueue_copy(dst_ptr=h_wsc.unsafe_ptr(), src_buf=winners_score)
        ctx.enqueue_copy(dst_ptr=h_wbf.unsafe_ptr(), src_buf=winners_bf)

        # THE ONE DRAIN OF THE LEVEL. Their `RebuildLeavesSizes`
        # (`split_properties_helper.cpp:800-813`), which is also ONE blocking
        # partition read per level on their side, at `:950`. So this drain is
        # their count and not an artifact of ours.
        #
        # What IS an artifact is the transfer under it. Their `Read` moves
        # host memory to host memory, because `PartitionsCpu` is
        # `EPtrType::CudaHost` and the split kernel already wrote it
        # (`split_points.cu:372`, `:379`); ours moves device memory to host
        # memory. `hp_off` / `hp_size` are the mirror that was supposed to
        # remove that transfer and cannot, and the search behind that is the
        # DEVIATION BLOCK in `gbdt/gpu_util/gpu_data/partitions.mojo`.
        # It used to point at `gbdt/gpu_lib/NOT_PORTED.md`, which says
        # nothing about any of this.
        #
        # ORDERING. The copy above is enqueued behind
        # `update_partitions_after_split_kernel` on the same queue, so it
        # cannot read a size the kernel has not written; this drain is only
        # what makes the copy's result readable on the host. The same
        # property is what makes their stream-ordered `Read` safe.
        mgr.wait_complete()

        # ============ THE GATES, POST-DRAIN =============================
        # Their host applies these BEFORE splitting
        # (`greedy_search_helper.cpp:535-539` the sentinel ENSURE, `:360-364`
        # the `Score < 0` gate -- ours is `best_score > 0`, the sign flipped
        # as everywhere in this port -- and
        # `oblivious_tree_doc_parallel_structure_searcher.cpp:134` the
        # repeat-split stop). With the pack device-side the level's split
        # chain has already run when the host first sees the score, so a
        # stop ROLLS BACK the one speculative level: the left child kept
        # the parent's offset, so restoring `p_sz[i] = sz[i] + sz[half+i]`
        # restores the parent partitions exactly; the reorder permuted rows
        # only WITHIN parent ranges so every leaf sum the tail computes is
        # unchanged; the right-child histogram slots `copy_histograms`
        # polluted are read by nothing after the break. Gate semantics are
        # unchanged: a stopped level's split is never appended, exactly as
        # before.
        var best_score = h_wsc.unsafe_ptr().unsafe_load(depth)
        var best_bin_u = h_wbf.unsafe_ptr().unsafe_load(depth)
        if (
            best_bin_u == UInt32(0xFFFFFFFF)
            or Int(best_bin_u) >= hist_cells_per_leaf
        ):

            # their `CB_ENSURE(bestSplits[0].FeatureId != (ui32)-1, ...)`
            raise Error(
                "All splits have infinite score. Probably, numerical"
                " overflow occurs in loss function and/or split score"
                " calculation. Try increasing l2_leaf_reg, and/or"
                " decreasing learning_rate, etc."
                " [level " + String(depth) + ", live leaves "
                + String(n_live // 2) + "]"
            )
        var choice = resolve_split(layout, Int(best_bin_u))
        # improving-score gate; for an oblivious tree all leaves share one
        # BestSplit, so one test is the whole gate (see the old host-side
        # note, now in git history at 992aa86's parent).
        var stop_level = not (best_score > Float32(0.0))
        # their repeat-split stop: a repeat means no candidate improved on
        # a split whose gain is already spent.
        for i in range(len(out_splits)):
            if (
                out_splits[i].feature_id == Int32(choice.feature)
                and out_splits[i].bin_idx == Int32(choice.bin)
            ):
                stop_level = True
        if stop_level:
            var half = n_live // 2
            for i in range(half):
                var merged = (
                    h_sz.unsafe_ptr().unsafe_load(i)
                    + h_sz.unsafe_ptr().unsafe_load(half + i)
                )
                h_sz.unsafe_ptr().unsafe_store(i, merged)
            ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
            ctx.synchronize()
            leaf_records = prev_records.copy()
            n_live = half
            break
        out_splits.append(
            TBinarySplit(Int32(choice.feature), Int32(choice.bin))
        )

        # their `RebuildLeavesSizes` filling `Leaves[i].Size` (`:806`). The
        # pairing rule reads these to pick the smaller child, so a stale size
        # here silently builds the wrong one.
        max_live_rows = 1
        for i in range(n_live):
            var s = Int(h_sz.unsafe_ptr().unsafe_load(i))
            leaf_records[i].size = UInt32(s)
            if s > max_live_rows:
                max_live_rows = s

    # ================================================================
    # their `estimator.Estimate(...)` then `AppendModels(...)`
    # (`doc_parallel_boosting.h:371` and `:391`).
    #
    # Their boosting loop runs these AFTER the structure search returns, in
    # that order, and so does this. The structure is final at this point;
    # nothing below changes which rows are in which leaf.
    # ================================================================
    if apply_to_cursor:
        # The final level's stats were never computed: the loop computes them
        # BEFORE each split, and the last split has no level after it.
        # `ids_a` still holds the LAST level's ids, and `n_live` has doubled
        # since. Refill it for the final leaf count.
        compute_partition_stats(
            ctx, n_live, max_live_rows, stat_count, n_rows,
            dense_ids, p_off, p_sz, stats, stat_partials, part_stats,
            sm_count=sm_count,
        )
        mgr.stream_kernel()

        ctx.enqueue_function[compute_leaf_values_kernel](
            part_stats.unsafe_ptr(), Int32(stat_count), Int32(n_live),
            l2_leaf_reg, leaf_values.unsafe_ptr(),
            grid_dim=(n_live + LEAF_BLOCK - 1) // LEAF_BLOCK,
            block_dim=LEAF_BLOCK,
        )
        mgr.stream_kernel()

        # their `AppendModels`, with `Rescale(step)` folded into the kernel.
        ctx.enqueue_copy(
            dst_ptr=h_leaf_values.unsafe_ptr(), src_buf=leaf_values
        )

        ctx.enqueue_function[add_model_value_kernel](
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), row_index.unsafe_ptr(),
            leaf_values.unsafe_ptr(), learning_rate, cursor.unsafe_ptr(),
            grid_dim=(wide, n_live, 1), block_dim=(256, 1, 1),
        )
        mgr.stream_kernel()

    ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
    mgr.wait_complete()
    if apply_to_cursor:
        # their `result[i].AddWeakModel(model)`, the half that keeps the tree
        # rather than only its effect on the cursor.
        for i in range(n_live):
            out_leaf_values.append(h_leaf_values.unsafe_ptr().unsafe_load(i))
    var out = List[Int]()
    for i in range(n_live):
        out.append(Int(h_sz.unsafe_ptr().unsafe_load(i)))
    return out^
