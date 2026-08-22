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

from std.math import sqrt

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.device_attribute import DeviceAttribute

from mojo_only.fixed_point import choose_scale

from gbdt.gpu_lib.gpu_manager import TCudaManager
from gbdt.methods.kernel_add_model_value import (
    add_model_value_kernel,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)
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
from core.identity_trace import IdentityTrace
from gbdt.methods.greedy_subsets_searcher.depthwise_stage_times import (
    StageTimes,
)
from gbdt.gpu_data.gpu_structures import CFeature
from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    choose_scale_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_resolve import (
    RESOLVE_BLOCK_SIZE,
    resolve_and_pack_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_8bit import (
    H8_BLOCK,
    hist2_8bit_gather_kernel,
    hist2_8bit_kernel,
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
from gbdt.data.permutation import TRandom
from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    FLOAT32_MAX,
    SCORE_BLOCK_SIZE,
    TARGET_VARIANCE_BLOCK,
    compute_optimal_splits_kernel,
    compute_target_variance_kernel,
    target_variance_blocks,
)
from gbdt.targets.kernel.pointwise_targets import (
    deterministic_sum_lanes_kernel,
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
    substract_histograms_vec4_kernel,
    copy_histograms_vec4_kernel,
    write_reduces_from_fixed_kernel,
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
    partition_chunks_sm_for,
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
    update_partitions_and_plan_kernel,
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


def compute_target_std_dev(
    ctx: DeviceContext,
    mut stats: DeviceBuffer[DType.float32],
    size: Int,
    stat_count: Int,
    stat_line_size: Int,
    multiclass_optimization: Bool,
    sm_count: Int,
) raises -> Float64:
    """`ComputeTargetStdDev` (`greedy_search_helper.cpp:369-378`).

        using TKernel = NKernelHost::TComputeTargetVarianceKernel;
        auto l2Stats = TStripeBuffer<double>::Create(RepeatOnAllDevices(3));
        LaunchKernels<TKernel>(..., target.StatsToAggregate, l2Stats,
                               target.MultiLogitOptimization);
        auto l2StatsCpu = ReadReduce(l2Stats);
        //        double sum = l2StatsCpu[0];
        double sum2 = l2StatsCpu[1];
        double weight = l2StatsCpu[2];
        return sqrt(sum2 / (weight + 1e-100));

    **THE DENOMINATOR IS THE SUMMED WEIGHT, NOT THE ROW COUNT**, and the
    doc-parallel arm's `ComputeStdDev` divides the same numerator by the
    row count instead (`random_score_helper.h:14-15`). CatBoost's two arms
    therefore produce DIFFERENT noise magnitudes from the same target
    whenever the weights are not all 1 -- which under Newton, their default
    leaf estimation, is always. Both are transliterated; see the file
    docstring of `gbdt/methods/random_score_helper.mojo`.

    Lane 0 is their `sum`, computed by the kernel and commented out by
    their host. Read here and discarded for the same reason: the kernel
    writes it either way.
    """
    if size <= 0 or stat_count <= 1:
        return 0.0
    var n_blocks = target_variance_blocks(size, sm_count)
    var partials = ctx.enqueue_create_buffer[DType.float32](3 * n_blocks)
    var l2_stats = ctx.enqueue_create_buffer[DType.float32](3)
    ctx.enqueue_function[compute_target_variance_kernel](
        stats.unsafe_ptr(),
        Int32(size),
        Int32(stat_count),
        Int32(stat_line_size),
        Int32(1) if multiclass_optimization else Int32(0),
        partials.unsafe_ptr(),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(TARGET_VARIANCE_BLOCK, 1, 1),
    )
    # their `FillBuffer(aggregatedStats, 0.0, 3, stream)` before the launch
    # (`compute_scores.cu:293`) has no counterpart: nothing is accumulated
    # into `l2_stats`, the deterministic fold WRITES all three slots.
    ctx.enqueue_function[deterministic_sum_lanes_kernel[3]](
        partials.unsafe_ptr(), Int32(n_blocks), l2_stats.unsafe_ptr(),
        grid_dim=1, block_dim=256,
    )
    var h = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_buf=h, src_buf=l2_stats)
    ctx.synchronize()
    # reading `h` HERE, after the synchronize, is what keeps the staging
    # buffer alive across the copy ([[mojo-buffer-freed-at-last-use]]); the
    # two device buffers are held to the same point below.
    var sum2 = Float64(h[1])
    var weight = Float64(h[2])
    _ = partials^
    _ = l2_stats^
    return sqrt(sum2 / (weight + 1e-100))


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
    _ = hz^  # past the drain (step-33 race class)
    _ = hoh^  # past the drain (step-33 race class)
    _ = hfd^  # past the drain (step-33 race class)
    _ = hfc^  # past the drain (step-33 race class)
    _ = hfb^  # past the drain (step-33 race class)
    _ = hfa^  # past the drain (step-33 race class)
    _ = h_sz2^  # past the drain (step-33 race class)
    _ = h_sz^  # past the drain (step-33 race class)
    _ = h_off2^  # past the drain (step-33 race class)
    _ = h_off^  # past the drain (step-33 race class)
    _ = h_leaf0^  # past the drain (step-33 race class)

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
    # DEVIATION 95 turned the kernel's `fixed_scale` into a DEVICE pointer
    # and this probe-only entry kept passing the scalar -- found when
    # `pixi run probe` refused to build, the same every-known-check-
    # updated-end-to-end-command-not-run class as DEVIATION 95's own miss.
    # The VALUE stays this site's own 1.0; only where it lives changes.
    var scale_dev = ctx.enqueue_create_buffer[DType.float32](1)
    var h_scale = ctx.enqueue_create_host_buffer[DType.float32](1)
    h_scale.unsafe_ptr().unsafe_store(0, Float32(1.0))
    ctx.enqueue_copy(dst_buf=scale_dev, src_ptr=h_scale.unsafe_ptr())
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
        scale_dev.unsafe_ptr(),
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
    _ = scan_ids_h^  # past the drain (step-33 race class)
    _ = hsk^  # past the drain (step-33 race class)
    # THE SCALE PAIR MUST OUTLIVE THE LAUNCH THAT READS IT, and until this
    # line neither did. `h_scale` was LAST USED at its `enqueue_copy` and
    # `scale_dev` at the `.unsafe_ptr()` handed to `binary_hist_kernel`, so
    # Mojo could free both there -- an ENQUEUE is not a RUN, and the very
    # next allocations (`scan_ids`, `part_stats`, `hps`) come straight off
    # the same pools. That is DEVIATION 134 exactly: the failure is a
    # garbage scale read intermittently, not a crash, so `check-level`
    # passing is not evidence against it. The drain above is the first
    # point at which the copy and the kernel have certainly run.
    _ = h_scale^
    _ = scale_dev^

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
        Int32(0),  # multiclassOptimization
        Float32(1.0),
        # `scoreStdDev` / `seed`: this probe entry point never asks for the
        # `random_strength` noise. `run_tree_layout` is the training path.
        Float32(0.0),
        UInt64(0),
        out_score.unsafe_ptr(),
        out_bin.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(SCORE_BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()
    _ = hfw^  # past the drain (step-33 race class)
    _ = hbf^  # past the drain (step-33 race class)

    var hos = ctx.enqueue_create_host_buffer[DType.float32](1)
    var hob = ctx.enqueue_create_host_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_ptr=hos.unsafe_ptr(), src_buf=out_score)
    ctx.enqueue_copy(dst_ptr=hob.unsafe_ptr(), src_buf=out_bin)
    ctx.synchronize()
    # the scale buffer must outlive the hist launch's execution
    # ([[mojo-buffer-freed-at-last-use]])
    _ = scale_dev.unsafe_ptr()
    _ = h_scale.unsafe_ptr()
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
    _ = e^  # past the drain (step-33 race class)

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
    _ = hr^  # past the drain (step-33 race class)
    _ = hl^  # past the drain (step-33 race class)
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


def upload_scale(
    ctx: DeviceContext, value: Float32
) raises -> DeviceBuffer[DType.float32]:
    """One host-computed scale into a 1-float device buffer (DEVIATION
    95: the histogram kernels read the scale from device memory), for
    the checks and probes that derive their scale on the host. The
    settling drain inside makes the staging safe to drop. THE CALLER
    KEEPS THE RETURNED BUFFER ALIVE past its last enqueued kernel
    (`_ = keep^` at the end of the function): dropping the only handle
    before the launches would free the memory under them. A drain per
    call is check-tier cost; the training path never comes through
    here."""
    var d = ctx.enqueue_create_buffer[DType.float32](1)
    var h = ctx.enqueue_create_host_buffer[DType.float32](1)
    h.unsafe_ptr().unsafe_store(0, value)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^  # past the drain (step-33 race class)
    return d^


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
    # the kernels read the scale from device memory now (DEVIATION 95);
    # this checked path keeps the host derivation and uploads the value
    var scale_val = Float32(choose_scale(mag, n_rows))
    var scale_dev_local = ctx.enqueue_create_buffer[DType.float32](1)
    var h_scale_local = ctx.enqueue_create_host_buffer[DType.float32](1)
    h_scale_local.unsafe_ptr().unsafe_store(0, scale_val)
    ctx.enqueue_copy(dst_buf=scale_dev_local, src_ptr=h_scale_local.unsafe_ptr())
    var fixed_scale = scale_dev_local.unsafe_ptr()

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
    _ = q5^  # past the drain (step-33 race class)
    _ = q4^  # past the drain (step-33 race class)
    _ = q3^  # past the drain (step-33 race class)
    _ = q2^  # past the drain (step-33 race class)
    _ = q^  # past the drain (step-33 race class)
    _ = hsk^  # past the drain (step-33 race class)
    _ = h_off^  # past the drain (step-33 race class)
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
    _ = hfw^  # past the drain (step-33 race class)
    _ = hbf^  # past the drain (step-33 race class)

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
            Int32(0),  # multiclassOptimization
            Float32(1.0),
            # `scoreStdDev` / `seed`: this is the uniform-binary probe
            # path, which never asks for the `random_strength` noise.
            Float32(0.0),
            UInt64(0),
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
        _ = hs5^  # past the drain (step-33 race class)

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
        _ = h_ids_c^  # past the drain (step-33 race class)
        _ = h_ids_b^  # past the drain (step-33 race class)
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
    _ = h_ids_a^  # past the drain (step-33 race class)
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
    _ = h_leaf_values^  # past the drain (step-33 race class)
    var out = List[Int]()
    for i in range(n_live):
        out.append(Int(h_sz.unsafe_ptr().unsafe_load(i)))
    # keep the scale staging alive past every enqueued kernel
    _ = scale_dev_local^
    _ = h_scale_local^
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
    var staging_holds = List[HostBuffer[DType.uint32]]()
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
        # per-iteration staging must outlive its queued copies: park each
        # buffer in the holder until the one drain below (step-33 race
        # class -- a plain scope exit here frees while copies from THIS
        # iteration are still queued)
        staging_holds.append(h1^)
        staging_holds.append(h2^)
        staging_holds.append(h3^)
        staging_holds.append(h4^)
        out.append(
            DeviceBlock(
                blk.policy, n, blk.first_column, total, widest,
                d_folds^, d_fo^, d_go^, d_gs^,
            )
        )
    ctx.synchronize()
    _ = staging_holds^
    return out^


def launch_hist2_8bit(
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
    fixed_scale: MutPointer[Float32, MutAnyOrigin],
) raises:
    """The fused two-stat 8-bit arm (DEVIATION BLOCK in
    `kernel/hist_2_one_byte_8bit.mojo`): one launch, grid z = 1, where
    the PASS design launches z = stat_count walks. Reached only when the
    `hist_smem_mode_for` row is the shared-Int32 arm; the float columns
    dispatch `launch_one_byte[8]` exactly as CatBoost's ladder does."""
    if stat_count != 2:
        raise Error("the fused 8-bit arm is two-stat by construction")
    var groups = feature_groups_for(POLICY_ONE_BYTE, Int(blk.n_features))
    var replicas = replication_for(
        groups, n_live, 1, sm_count, gather=(depth > 0)
    )
    if depth == 0:
        ctx.enqueue_function[hist2_8bit_kernel](
            blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
            blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
            Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line),
            Int32(base), stats.unsafe_ptr(), Int32(n_rows),
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
            block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
            Int32(max_leaves), Int32(stat_count),
            grid_dim=(groups * replicas, n_live, 1),
            block_dim=(H8_BLOCK, 1, 1),
        )
    else:
        ctx.enqueue_function[hist2_8bit_gather_kernel](
            blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
            blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
            Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line),
            Int32(base), row_index.unsafe_ptr(),
            stats.unsafe_ptr(), Int32(n_rows),
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
            block_hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
            Int32(max_leaves), Int32(stat_count),
            grid_dim=(groups * replicas, n_live, 1),
            block_dim=(H8_BLOCK, 1, 1),
        )


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
    fixed_scale: MutPointer[Float32, MutAnyOrigin],
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
    fixed_scale: MutPointer[Float32, MutAnyOrigin],
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

    **AND UNDER `IDENTICAL` THE `sm_count` IS PINNED** (IDENTITY_PATHS
    row 7 class, DEVIATION 354; DEVIATIONS 252 and 353 are the
    std-dev-grid siblings). The replication factor looks like pure
    occupancy, and for the hist_2/one-byte families it is: they quantize
    PER ROW (`hist2_quantize`) and sum in Int32, so any partition of rows
    into blocks gives the same bits. The binary and half-byte families do
    NOT: their shared histograms accumulate in FLOAT and the
    deterministic flush quantizes the per-block PARTIAL
    (`Int32(val * fixed_scale)`, `hist_binary.mojo` /
    `hist_half_byte.mojo`), so `active_block_count` -- which is
    `min(f(p_size), maxBlocksPerPart)` with `maxBlocksPerPart` derived
    from THIS replication -- decides which rows form each rounded
    partial. A core count in that formula is a summation order
    (`numerics.mojo`'s opening warning), so `IDENTICAL` feeds it
    `kernel_matrix.partition_chunks_sm_for`'s pin (32 everywhere) and
    `FAST` keeps the device's count, CatBoost's behavior bit for bit.
    One pin for all three policies, because a pin only some policies
    read cannot be audited.

    THE RESIDUE THIS PIN DOES NOT CLOSE: `min_docs_per_block` inside
    those kernels multiplies in `BLOCK_SIZE`, and
    `kernel_matrix.block_size_for` is not identical-gated -- NVIDIA's
    48 KB budget yields 768 where the identity floor's 32 KB yields 512,
    so the float-family fold shape still differs cross-vendor under
    IDENTICAL until that accessor reads the frozen floor. Reported to
    the matrix's owner; this function cannot fix it from here.
    """
    var blocks_per_sm = 2
    # DEVIATION 354: pinned under IDENTICAL, device count under FAST.
    var sm = partition_chunks_sm_for[HIST_BUILD_MODE == NUMERIC_IDENTICAL](
        sm_count
    )
    var max_active_blocks = blocks_per_sm * sm
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
    fixed_scale: MutPointer[Float32, MutAnyOrigin],
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
        comptime _flush_is_fixed_point = deterministic_flush_for[
            TARGET_COLUMN, HIST_BUILD_MODE == NUMERIC_IDENTICAL
        ]()
        var run_fixed_bridge = _flush_is_fixed_point

        @parameter
        if (
            hist2_smem_mode == HIST_SMEM_SHARED2_I32
            and not _flush_is_fixed_point
        ):
            if blk.policy == POLICY_ONE_BYTE:
                run_fixed_bridge = True

        # A ONE-BYTE block on the i32 arms never touches the scratch any
        # more: its writebacks put single-block cells in the accumulator
        # too, and the fused writeback below reads the accumulator alone.
        # So the whole-buffer zero is SKIPPED for exactly those blocks --
        # their ZeroBuffer exists to give the float atomics a zeroed
        # destination, and there are none here.
        var scratch_dead = (
            run_fixed_bridge and blk.policy == POLICY_ONE_BYTE
        )
        if not scratch_dead:
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

                @parameter
                if hist2_smem_mode == HIST_SMEM_SHARED2_I32:
                    if stat_count == 2:
                        # the fused two-stat 8-bit arm: one walk over the
                        # cindex where PASS(8) makes stat_count of them
                        # (its DEVIATION BLOCK carries the shared-memory
                        # arithmetic their ladder's fallback exists to
                        # avoid)
                        launch_hist2_8bit(
                            ctx, blk, depth, n_live, n_rows, stat_count,
                            max_leaves, sm_count, line, base, cindex,
                            row_index, stats, p_off, p_sz, ids,
                            block_hist, acc_i32, fixed_scale,
                        )
                    else:
                        # MultiClass carries 1 + (K-1) stat planes and the
                        # fused arm is two-stat by construction. When it
                        # landed (8010b2f) every caller WAS two-stat;
                        # MultiClass arrived later in another lane, and the
                        # first 254-border multiclass fit hit the fused
                        # arm's guard instead of a histogram. Multi-stat
                        # shapes take the PASS route, whose shared-Int32
                        # arm walks stat pairs on the z axis exactly as
                        # their ladder does.
                        launch_one_byte[8, hist2_smem_mode](
                            ctx, blk, depth, n_live, n_rows, stat_count,
                            max_leaves, sm_count, line, base, cindex,
                            row_index, stats, p_off, p_sz, ids,
                            block_hist, acc_i32, fixed_scale,
                        )
                else:
                    launch_one_byte[8, hist2_smem_mode](
                        ctx, blk, depth, n_live, n_rows, stat_count,
                        max_leaves, sm_count, line, base, cindex,
                        row_index, stats, p_off, p_sz, ids, block_hist,
                        acc_i32, fixed_scale,
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
        # The bridge is FUSED into the writeback below when it would have
        # run: `write_reduces_from_fixed_kernel` reads the accumulator
        # directly, converts with the identical expression, and zeroes it
        # -- one cell pass and one launch fewer per block per level, and
        # the float block scratch is not touched at all on this arm. See
        # its DEVIATION BLOCK for why the bits cannot move.

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
        if scratch_dead:
            ctx.enqueue_function[write_reduces_from_fixed_kernel[False]](
                Int32(block_first_bin),
                Int32(blk.total_folds),
                ids.unsafe_ptr(),
                acc_i32.unsafe_ptr(),
                block_hist.unsafe_ptr(),
                fixed_scale,
                Int32(hist_cells_per_leaf),
                hist.unsafe_ptr(),
                grid_dim=(
                    (blk.total_folds + 127) // 128, n_live, stat_count
                ),
                block_dim=(128, 1, 1),
            )
        elif run_fixed_bridge:
            ctx.enqueue_function[write_reduces_from_fixed_kernel[True]](
                Int32(block_first_bin),
                Int32(blk.total_folds),
                ids.unsafe_ptr(),
                acc_i32.unsafe_ptr(),
                block_hist.unsafe_ptr(),
                fixed_scale,
                Int32(hist_cells_per_leaf),
                hist.unsafe_ptr(),
                grid_dim=(
                    (blk.total_folds + 127) // 128, n_live, stat_count
                ),
                block_dim=(128, 1, 1),
            )
        else:
            ctx.enqueue_function[write_reduces_histograms_kernel](
                Int32(block_first_bin),
                Int32(blk.total_folds),
                ids.unsafe_ptr(),
                block_hist.unsafe_ptr(),
                Int32(hist_cells_per_leaf),
                hist.unsafe_ptr(),
                grid_dim=(
                    (blk.total_folds + 127) // 128, n_live, stat_count
                ),
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


struct TTreeWorkspace(Movable):
    """The three large planes a tree grows in, owned by the FIT.

    ================= DEVIATION BLOCK =================
    CatBoost does not allocate per tree and neither should this. Their
    `TCudaManager` hands every buffer out of a per-device memory pool
    (`cuda_lib/memory_pool.h`), so `CreateInitialSubsets`
    (`split_properties_helper.cpp:1040-1080`) reuses the same device
    memory for tree 2 that tree 1 gave back. This port dropped the pool
    layer (`gbdt/gpu_lib/NOT_PORTED.md`) and called
    `enqueue_create_buffer` directly, which meant a fresh allocation of
    every plane for EVERY tree.

    MEASURED, 50k rows x 100 features x 254 borders, depth 6: the setup
    before the first level cost **4.69 ms of a 12.3 ms tree**, and
    allocating these three planes alone (2 x 13 MB plus the block
    scratch) was **1.9 ms** of it. The memsets that follow are 0.5 ms and
    are kept -- their `FillBuffer` in `CreateInitialSubsets` is real work,
    not allocation.

    This is a POOL OF ONE, which is all a single-device, single-stream
    port needs: `fit` holds the list across trees, and a tree reuses the
    planes when they are big enough and rebuilds them when they are not.
    An empty list means "no pool", which is what every existing caller
    passes and which restores the old allocate-per-call behaviour exactly.
    ===================================================
    """

    var hist: DeviceBuffer[DType.float32]
    var acc_i32: DeviceBuffer[DType.int32]
    var block_hist: DeviceBuffer[DType.float32]
    var hist_cells: Int
    var block_cells: Int
    # ---- the rest of the per-tree setup, hoisted 2026-08-21 ----
    # Everything below is sized by the DATASET SHAPE and either holds
    # layout-derived constants (filled once here, kernels only read) or is
    # scratch every level rewrites before reading. Re-creating it per tree
    # was measured as a covtype-scale floor term: ~20 buffer creations,
    # fifteen constant uploads and one drain per tree, none of which
    # depends on the tree.
    var n_rows_key: Int
    var stat_count_key: Int
    var max_leaves_key: Int
    var n_features_key: Int
    var hist_cells_per_leaf_key: Int
    var dblocks: List[DeviceBlock]
    var p_off: DeviceBuffer[DType.uint32]
    var p_sz: DeviceBuffer[DType.uint32]
    var hp_off: DeviceBuffer[DType.uint32]
    var hp_sz: DeviceBuffer[DType.uint32]
    var h_off: HostBuffer[DType.uint32]
    var h_sz: HostBuffer[DType.uint32]
    var part_stats: DeviceBuffer[DType.float32]
    var stat_partials: DeviceBuffer[DType.float32]
    var flags: DeviceBuffer[DType.uint8]
    var seq: DeviceBuffer[DType.uint32]
    var gmap: DeviceBuffer[DType.uint32]
    var sflags: DeviceBuffer[DType.uint8]
    var new_index: DeviceBuffer[DType.uint32]
    var new_stats: DeviceBuffer[DType.float32]
    var chunk_zeros: DeviceBuffer[DType.uint32]
    var chunk_offsets: DeviceBuffer[DType.uint32]
    var leaf_zeros: DeviceBuffer[DType.uint32]
    var ids_c: DeviceBuffer[DType.uint32]
    var zero_ids: DeviceBuffer[DType.uint32]
    var h_zero_ids: HostBuffer[DType.uint32]
    var dense_ids: DeviceBuffer[DType.uint32]
    var leaf_values: DeviceBuffer[DType.float32]
    var h_leaf_values: HostBuffer[DType.float32]
    var ids_compute: DeviceBuffer[DType.uint32]
    var h_ids_compute: HostBuffer[DType.uint32]
    var sub_from: DeviceBuffer[DType.uint32]
    var sub_what: DeviceBuffer[DType.uint32]
    var h_sub_from: HostBuffer[DType.uint32]
    var h_sub_what: HostBuffer[DType.uint32]
    var skip: DeviceBuffer[DType.uint8]
    var out_score: DeviceBuffer[DType.float32]
    var out_bin: DeviceBuffer[DType.uint32]
    var winners_score: DeviceBuffer[DType.float32]
    var winners_bf: DeviceBuffer[DType.uint32]
    var h_wsc: HostBuffer[DType.float32]
    var h_wbf: HostBuffer[DType.uint32]
    var sp_feats: DeviceBuffer[DType.uint8]
    var sp_feats_h: HostBuffer[DType.uint8]
    var sp_bins: DeviceBuffer[DType.uint32]
    var sp_bins_h: HostBuffer[DType.uint32]
    var flat_first: DeviceBuffer[DType.uint32]
    var flat_folds: DeviceBuffer[DType.uint32]
    var flat_one_hot: DeviceBuffer[DType.uint8]
    var bff: DeviceBuffer[DType.uint32]
    var ffw: DeviceBuffer[DType.float32]
    var bfr_off: DeviceBuffer[DType.uint32]
    var bfr_mask: DeviceBuffer[DType.uint32]
    var bfr_shift: DeviceBuffer[DType.uint32]
    var bfr_first: DeviceBuffer[DType.uint32]
    var bfr_folds: DeviceBuffer[DType.uint32]
    var bfr_oh: DeviceBuffer[DType.uint8]
    var bfr_bin: DeviceBuffer[DType.uint32]
    var scale_dev: DeviceBuffer[DType.float32]
    var h_scale: HostBuffer[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        layout: CompressedIndexLayout,
        blocks: List[PolicyBlock],
        n_rows: Int,
        stat_count: Int,
        max_depth: Int,
    ) raises:
        var max_leaves = 1 << max_depth
        var n_features = len(layout.features)
        var hist_cells_per_leaf = layout.hist_cells
        var hist_cells = max_leaves * stat_count * hist_cells_per_leaf
        var widest_block = 1
        for b in range(len(blocks)):
            var tf = 0
            for k in range(blocks[b].count()):
                tf += Int(blocks[b].folds[k])
            if tf > widest_block:
                widest_block = tf
        var block_cells = max_leaves * stat_count * widest_block
        var argmax_blocks = (hist_cells_per_leaf + 255) // 256
        if argmax_blocks > 64:
            argmax_blocks = 64
        if argmax_blocks < 1:
            argmax_blocks = 1
        var sm_count = ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var stat_chunks = (n_rows + STATS_BLOCK - 1) // STATS_BLOCK
        var machine_chunks = partition_stats_chunks(sm_count, stat_count)
        if machine_chunks > stat_chunks:
            stat_chunks = machine_chunks
        var max_chunks = (n_rows + PARTITION_BLOCK - 1) // PARTITION_BLOCK

        self.n_rows_key = n_rows
        self.stat_count_key = stat_count
        self.max_leaves_key = max_leaves
        self.n_features_key = n_features
        self.hist_cells_per_leaf_key = hist_cells_per_leaf
        self.hist_cells = hist_cells
        self.block_cells = block_cells
        self.hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
        self.acc_i32 = ctx.enqueue_create_buffer[DType.int32](hist_cells)
        self.block_hist = ctx.enqueue_create_buffer[DType.float32](
            block_cells
        )
        self.p_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.p_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.hp_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.hp_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.h_off = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
        self.h_sz = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
        self.part_stats = ctx.enqueue_create_buffer[DType.float32](
            max_leaves * stat_count
        )
        self.stat_partials = ctx.enqueue_create_buffer[DType.float32](
            max_leaves * stat_count * stat_chunks
        )
        self.flags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
        self.seq = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.gmap = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.sflags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
        self.new_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.new_stats = ctx.enqueue_create_buffer[DType.float32](
            stat_count * n_rows
        )
        self.chunk_zeros = ctx.enqueue_create_buffer[DType.uint32](
            max_leaves * max_chunks
        )
        self.chunk_offsets = ctx.enqueue_create_buffer[DType.uint32](
            max_leaves * max_chunks
        )
        self.leaf_zeros = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.ids_c = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.zero_ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.h_zero_ids = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.dense_ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.leaf_values = ctx.enqueue_create_buffer[DType.float32](
            max_leaves
        )
        self.h_leaf_values = ctx.enqueue_create_host_buffer[DType.float32](
            max_leaves
        )
        self.ids_compute = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.h_ids_compute = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.sub_from = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.sub_what = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.h_sub_from = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.h_sub_what = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.skip = ctx.enqueue_create_buffer[DType.uint8](
            hist_cells_per_leaf
        )
        self.out_score = ctx.enqueue_create_buffer[DType.float32](
            argmax_blocks
        )
        self.out_bin = ctx.enqueue_create_buffer[DType.uint32](argmax_blocks)
        self.winners_score = ctx.enqueue_create_buffer[DType.float32](
            max_depth
        )
        self.winners_bf = ctx.enqueue_create_buffer[DType.uint32](max_depth)
        self.h_wsc = ctx.enqueue_create_host_buffer[DType.float32](max_depth)
        self.h_wbf = ctx.enqueue_create_host_buffer[DType.uint32](max_depth)
        var built_f = make_split_features_buffers(ctx, max_leaves)
        self.sp_feats = built_f[0]
        self.sp_feats_h = built_f[1]
        self.sp_bins = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.sp_bins_h = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.flat_first = ctx.enqueue_create_buffer[DType.uint32](n_features)
        self.flat_folds = ctx.enqueue_create_buffer[DType.uint32](n_features)
        self.flat_one_hot = ctx.enqueue_create_buffer[DType.uint8](
            n_features
        )
        self.bff = ctx.enqueue_create_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        self.ffw = ctx.enqueue_create_buffer[DType.float32](n_features)
        self.bfr_off = ctx.enqueue_create_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        self.bfr_mask = ctx.enqueue_create_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        self.bfr_shift = ctx.enqueue_create_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        self.bfr_first = ctx.enqueue_create_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        self.bfr_folds = ctx.enqueue_create_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        self.bfr_oh = ctx.enqueue_create_buffer[DType.uint8](
            hist_cells_per_leaf
        )
        self.bfr_bin = ctx.enqueue_create_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        self.scale_dev = ctx.enqueue_create_buffer[DType.float32](1)
        self.h_scale = ctx.enqueue_create_host_buffer[DType.float32](1)
        self.dblocks = upload_blocks(ctx, blocks)

        # ---- constant fills: staged locally, settled by the one drain ----
        var h_dense = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
        for i in range(max_leaves):
            h_dense.unsafe_ptr().unsafe_store(i, UInt32(i))
        ctx.enqueue_copy(dst_buf=self.dense_ids, src_ptr=h_dense.unsafe_ptr())
        # a second copy of the dense sequence: the blind level plan
        # (DEVIATION 94) passes an id list and the dense list to the same
        # call, and the exclusivity checker refuses one buffer twice
        ctx.enqueue_copy(
            dst_buf=self.ids_compute, src_ptr=h_dense.unsafe_ptr()
        )
        # ...and a third: the static plan (depth 0, or subtraction off)
        # needs an id list DISTINCT from `dense_ids` for the same reason
        ctx.enqueue_copy(dst_buf=self.zero_ids, src_ptr=h_dense.unsafe_ptr())
        var hsk = ctx.enqueue_create_host_buffer[DType.uint8](
            hist_cells_per_leaf
        )
        for i in range(hist_cells_per_leaf):
            hsk.unsafe_ptr().unsafe_store(i, UInt8(0))
        ctx.enqueue_copy(dst_buf=self.skip, src_ptr=hsk.unsafe_ptr())
        var hff = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
        var hfd = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
        var hfoh = ctx.enqueue_create_host_buffer[DType.uint8](n_features)
        for i in range(n_features):
            hff.unsafe_ptr().unsafe_store(
                i, layout.features[i].first_fold_index
            )
            hfd.unsafe_ptr().unsafe_store(i, layout.features[i].folds)
            hfoh.unsafe_ptr().unsafe_store(
                i,
                UInt8(1) if layout.features[i].one_hot_feature else UInt8(0),
            )
        ctx.enqueue_copy(dst_buf=self.flat_first, src_ptr=hff.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.flat_folds, src_ptr=hfd.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.flat_one_hot, src_ptr=hfoh.unsafe_ptr())
        var hbf = ctx.enqueue_create_host_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        var hfw = ctx.enqueue_create_host_buffer[DType.float32](n_features)
        for f in range(n_features):
            ref lf = layout.features[f]
            for b in range(Int(lf.folds)):
                hbf.unsafe_ptr().unsafe_store(
                    Int(lf.first_fold_index) + b, UInt32(f)
                )
        for f in range(n_features):
            hfw.unsafe_ptr().unsafe_store(f, Float32(1.0))
        ctx.enqueue_copy(dst_buf=self.bff, src_ptr=hbf.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.ffw, src_ptr=hfw.unsafe_ptr())
        var hbr1 = ctx.enqueue_create_host_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        var hbr2 = ctx.enqueue_create_host_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        var hbr3 = ctx.enqueue_create_host_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        var hbr4 = ctx.enqueue_create_host_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        var hbr5 = ctx.enqueue_create_host_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        var hbr6 = ctx.enqueue_create_host_buffer[DType.uint8](
            hist_cells_per_leaf
        )
        var hbr7 = ctx.enqueue_create_host_buffer[DType.uint32](
            hist_cells_per_leaf
        )
        for f in range(n_features):
            ref tf2 = layout.features[f]
            for b in range(Int(tf2.folds)):
                var bfx = Int(tf2.first_fold_index) + b
                hbr1.unsafe_ptr().unsafe_store(
                    bfx, tf2.offset * UInt32(n_rows)
                )
                hbr2.unsafe_ptr().unsafe_store(bfx, tf2.mask)
                hbr3.unsafe_ptr().unsafe_store(bfx, tf2.shift)
                hbr4.unsafe_ptr().unsafe_store(bfx, tf2.first_fold_index)
                hbr5.unsafe_ptr().unsafe_store(bfx, tf2.folds)
                hbr6.unsafe_ptr().unsafe_store(
                    bfx, UInt8(1) if tf2.one_hot_feature else UInt8(0)
                )
                hbr7.unsafe_ptr().unsafe_store(bfx, UInt32(b))
        ctx.enqueue_copy(dst_buf=self.bfr_off, src_ptr=hbr1.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.bfr_mask, src_ptr=hbr2.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.bfr_shift, src_ptr=hbr3.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.bfr_first, src_ptr=hbr4.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.bfr_folds, src_ptr=hbr5.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.bfr_oh, src_ptr=hbr6.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.bfr_bin, src_ptr=hbr7.unsafe_ptr())
        ctx.synchronize()
        # past the drain, every staging buffer this ctor enqueued from:
        # their last uses were the enqueues above, which freed them under
        # queued copies (the step-33 race class -- freed host pages can
        # be reused before the queue drains, and the ctor allocates
        # between enqueue and drain)
        _ = h_dense^
        _ = hsk^
        _ = hff^
        _ = hfd^
        _ = hfoh^
        _ = hbf^
        _ = hfw^
        _ = hbr1^
        _ = hbr2^
        _ = hbr3^
        _ = hbr4^
        _ = hbr5^
        _ = hbr6^
        _ = hbr7^


def _sym_dd2(n: Int) -> String:
    """Two zero-padded digits, for depth components of trace tags."""
    if n < 10:
        return String("0") + String(n)
    return String(n)


def run_tree_layout_traced[
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
    mut out_leaf_offsets: List[Int],
    mut ws: List[TTreeWorkspace],
    mut trace: IdentityTrace,
    mut times: StageTimes,
    tree_tag: String,
    export_offsets: Bool = False,
    use_subtraction: Bool = True,
    apply_to_cursor: Bool = False,
    learning_rate: Float32 = Float32(0.3),
    l2_leaf_reg: Float32 = Float32(3.0),
    sync_budget: Int = -1,
    one_hot: List[Bool] = List[Bool](),
    score_function: Int = SCORE_FUNCTION_COSINE,
    approx_dim: Int = 1,
    multiclass_optimization: Bool = False,
    mags_dev: Optional[DeviceBuffer[DType.float32]] = None,
    # `Options.RandomStrength` AS THE SEARCHER SEES IT -- already
    # multiplied by the boosting loop's `CalcScoreModelLengthMult`
    # (`greedy_subsets_searcher.h:73-76`). Zero is CatBoost's off switch
    # and this port's default.
    random_strength: Float32 = Float32(0.0),
    # their `TGpuAwareRandom` for this tree; see DEVIATION 139 at its use.
    random_seed: UInt64 = UInt64(0),
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
    # `statCount` is `1 + point.GetColumnCount()` -- their `StochasticDer`
    # sizes `StatsToAggregate` as one weight column plus one der column
    # per APPROX DIMENSION (`pointwise_target_impl.h:186-188`, and
    # `multiclass_targets.cpp:31-42` for the multi-dimensional case, where
    # it is `1 + NumClasses` minus one for MultiClass's pinned class).
    # Every single-dimensional loss is 2 and that is the default.
    var stat_count = 1 + approx_dim
    var max_leaves = 1 << max_depth

    var layout = build_layout(fold_counts, one_hot)
    var blocks = blocks_for(layout, n_rows)
    var hist_cells_per_leaf = layout.hist_cells

    # their `TArchProps::SMCount()`, read from the device rather than
    # guessed. `replication_for` turns it into their grid x factor.
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    var wide = (n_rows + 255) // 256
    if wide > 256:
        wide = 256
    if wide < 1:
        wide = 1

    # ---- THE POOL, which since 2026-08-21 carries the WHOLE setup ----
    # (see TTreeWorkspace's deviation block). A key mismatch means a
    # different dataset shape reached this searcher; rebuild everything.
    if (
        len(ws) == 0
        or ws[0].n_rows_key != n_rows
        or ws[0].stat_count_key != stat_count
        or ws[0].max_leaves_key != max_leaves
        or ws[0].n_features_key != len(fold_counts)
        or ws[0].hist_cells_per_leaf_key != hist_cells_per_leaf
    ):
        ws.clear()
        ws.append(TTreeWorkspace(
            ctx, layout, blocks, n_rows, stat_count, max_depth
        ))
    ref hist = ws[0].hist
    ref block_hist = ws[0].block_hist
    ref acc_i32 = ws[0].acc_i32
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
    ref ids_c = ws[0].ids_c
    ref zero_ids = ws[0].zero_ids
    ref h_zero_ids = ws[0].h_zero_ids
    ref dense_ids = ws[0].dense_ids
    ref leaf_values = ws[0].leaf_values
    ref h_leaf_values = ws[0].h_leaf_values
    ref ids_compute = ws[0].ids_compute
    ref h_ids_compute = ws[0].h_ids_compute
    ref sub_from = ws[0].sub_from
    ref sub_what = ws[0].sub_what
    ref h_sub_from = ws[0].h_sub_from
    ref h_sub_what = ws[0].h_sub_what
    ref skip = ws[0].skip
    ref out_score = ws[0].out_score
    ref out_bin = ws[0].out_bin
    ref winners_score = ws[0].winners_score
    ref winners_bf = ws[0].winners_bf
    ref sp_feats = ws[0].sp_feats
    ref sp_feats_h = ws[0].sp_feats_h
    ref sp_bins = ws[0].sp_bins
    ref sp_bins_h = ws[0].sp_bins_h
    ref h_wsc = ws[0].h_wsc
    ref h_wbf = ws[0].h_wbf
    ref flat_first = ws[0].flat_first
    ref flat_folds = ws[0].flat_folds
    ref flat_one_hot = ws[0].flat_one_hot
    ref bff = ws[0].bff
    ref ffw = ws[0].ffw
    ref bfr_off = ws[0].bfr_off
    ref bfr_mask = ws[0].bfr_mask
    ref bfr_shift = ws[0].bfr_shift
    ref bfr_first = ws[0].bfr_first
    ref bfr_folds = ws[0].bfr_folds
    ref bfr_oh = ws[0].bfr_oh
    ref bfr_bin = ws[0].bfr_bin

    # `argmaxBlockCount = Min(CeilDivide(binFeatureCountPerDevice, 256),
    # 64)` (`greedy_search_helper.cpp:439`), still needed for the score
    # launches below; the buffers it sizes live in the pool.
    var argmax_blocks = (hist_cells_per_leaf + 255) // 256
    if argmax_blocks > 64:
        argmax_blocks = 64
    if argmax_blocks < 1:
        argmax_blocks = 1

    # ---- per-tree state, the only part the pool cannot carry over ----
    # their `CreateInitialSubsets`' root partition and `FillBuffer` zeroing
    for i in range(max_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(0))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_memset(hist, Float32(0.0))
    ctx.enqueue_memset(acc_i32, Int32(0))

    # See the note in `run_tree`: one scale serves both accumulated planes,
    # so the bound is the larger of the two sums of magnitudes, and
    # `choose_scale` owns the derivation rather than a fourth copy of it.
    # ================= DEVIATION 95 =================
    # THE SCALE IS DERIVED ON THE DEVICE when the caller hands the
    # magnitudes buffer: `choose_scale_kernel` reproduces the host
    # function bit-for-bit (both are the same exact integer search once
    # the snap is a power of two), and the kernels read the scale from
    # device memory -- so the boosting loop's per-tree magnitudes drain,
    # the LAST drain besides the tree's own, is gone. This whole
    # apparatus is OURS, not CatBoost's: their GPU histograms are float
    # atomicAdd, tagged non_deterministic, and need no scale at all. The
    # fixed-point accumulator is the price of the bitwise-determinism
    # thesis; this deviation makes that price one kernel launch instead
    # of a drain. Callers without a magnitudes buffer (the checks) keep
    # the host derivation and pay one async upload, no drain.
    # ================================================
    var fixed_scale = rebind[MutPointer[Float32, MutAnyOrigin]](
        ws[0].scale_dev.unsafe_ptr()
    )
    if mags_dev:
        ctx.enqueue_function[choose_scale_kernel](
            rebind[MutPointer[Float32, MutAnyOrigin]](
                mags_dev.value().unsafe_ptr()
            ),
            Int32(n_rows), fixed_scale,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
        # enqueued before `mgr` exists; the launch counter misses this
        # one launch, which is a diagnostic and not a budget
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
    # (`split_properties_helper.cpp:802`). This loop blocks NOT AT ALL:
    # the winner is resolved and packed on the device
    # (`kernel/split_resolve.mojo`, DEVIATION BLOCK), every level is
    # enqueued blind (DEVIATION 94 inside the loop), and the single drain
    # of the whole tree sits after the loop, where the gates walk the
    # winner records level by level.
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

    # ============ `CreateInitialSubsets`' ScoreStdDev ==================
    # `greedy_search_helper.cpp:383-388`, verbatim:
    #
    #     if (Options.RandomStrength) {
    #         ScoreStdDev = Options.RandomStrength * ComputeTargetStdDev(target);
    #     } else {
    #         ScoreStdDev = 0;
    #     }
    #
    # ONCE PER TREE, before any level, and `Options.RandomStrength` has
    # ALREADY been multiplied by the boosting loop's model-length
    # multiplier (`greedy_subsets_searcher.h:73-76`,
    # `options.RandomStrength *= randomStrengthMult`). That is why this
    # parameter is `random_strength` and not `random_strength * mult`: the
    # caller does the multiply, exactly where their `CreateStructureSearcher`
    # does it.
    #
    # THE TARGET IS THE BOOTSTRAPPED ONE. Their `ComputeTarget` runs
    # `StochasticDer(bootstrapConfig, ...)` and the std dev is taken from
    # its output (`greedy_search_helper.cpp:381-385`), so a bootstrap that
    # scales the planes scales the noise with them. `stats` here is
    # post-bootstrap for the same reason.
    #
    # AND ON THIS ARM THE NOISE CANCELS -- see the block in
    # `kernel/compute_scores.mojo`. It is computed, launched and paid for
    # anyway, because that is their code; what it buys is that the day the
    # two calcers stop being seeded alike, this arm is already correct.
    var score_std_dev = Float32(0.0)
    if random_strength != Float32(0.0):
        score_std_dev = Float32(
            Float64(random_strength)
            * compute_target_std_dev(
                ctx,
                stats,
                n_rows,
                stat_count,
                n_rows,
                multiclass_optimization,
                sm_count,
            )
        )

    # their `TGreedySearchHelper::Random`, a `TGpuAwareRandom` shared by the
    # whole fit and drawn from ONCE PER `ComputeOptimalSplits` call, i.e.
    # per LEVEL (`greedy_search_helper.cpp:468`, `:489`, `:510`).
    # DEVIATION 139: theirs is one stream across every tree of the fit;
    # this one is re-seeded per tree from the caller's `random_seed`,
    # because `run_tree_layout` owns the level loop and the boosting loop
    # owns the trees. Per-level draws still differ, a fit is still
    # reproducible from its seed, and the noise cancels on this arm anyway.
    var level_rand = TRandom(random_seed)

    var n_live = 1
    # grid sizing bound: with no per-level size read (DEVIATION 94) the
    # largest live leaf is unknown until the tree is done, and `n_rows`
    # bounds it at every level; the kernels stride actual leaf ranges.
    var max_live_rows = n_rows

    for depth in range(max_depth):
        # `Random.NextUniformL()`, one draw per level, BEFORE the launch
        # and regardless of which calcer the score function selects.
        var level_seed = level_rand.next_uniform_l()
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
        # ================= DEVIATION 94 =================
        # ONE DRAIN PER TREE, NOT PER LEVEL. Their host blocks every level
        # (`bestProps.Read` + `RebuildLeavesSizes`) because a CUDA
        # pinned-memory read costs ~5 us; this box's drain costs ~191 us
        # plus a queue-empty bubble, and at covtype scale those six waits
        # were the largest term of the tree. So every level is enqueued
        # BLIND -- no host read anywhere in this loop -- and the winners,
        # final sizes and gates are handled ONCE after it. What made the
        # per-level sizes load-bearing, and what replaces each use:
        #
        # * their smaller-child choice (`:1318`) picked which sibling to
        #   COMPUTE, the other derived by subtraction. That choice moved
        #   ON-DEVICE (`plan_level_kernel`, reading the `p_sz` the
        #   partition update just wrote) rather than becoming static:
        #   DEVIATION 136 measured that WHICH sibling is computed DOES
        #   change histogram bits (the fixed-point cells round through
        #   `Float32(Int(q))/fixed_scale` BEFORE the float32 subtraction),
        #   and their tie lands on the RIGHT child -- see
        #   `plan_level_kernel`'s docstring for the full correction.
        # * their `NonZeroLeaves`/`ZeroLeaves` split skipped empty
        #   leaves' builds and zeroed their slots. The compute slots are
        #   now zeroed UNCONDITIONALLY before the build (the builder
        #   writes nothing for an empty range, so the zero IS the empty
        #   leaf's histogram), which subsumes that logic without sizes.
        # * grid sizing used the level's largest leaf; it now uses
        #   `n_rows`, a safe upper bound (kernels stride leaf ranges).
        # * the gates and the split record move to the post-loop walk,
        #   with a MULTI-level rollback on the rare stop (the same
        #   left-child-keeps-parent-offset argument as the old one-level
        #   rollback, applied k times).
        # `dense_ids` holds 0..max_leaves-1, so the left children ARE its
        # prefix and the right children ARE `dense_ids + half`: the whole
        # level plan costs no host fill and no upload.
        # ================================================================
        var half = n_live // 2
        var planned = use_subtraction and depth > 0
        var n_compute = n_live
        if planned:
            n_compute = half
            # their smaller-child choice (`:1318`) was `plan_level_kernel`
            # here, reading the `p_sz` the previous level's partition
            # update just wrote; DEVIATION 210 folds that store into the
            # update's border branch (`update_partitions_and_plan_kernel`),
            # so `ids_compute`/`sub_from`/`sub_what` already hold this
            # level's plan and the launch is gone.
        var level_ids = ids_compute.unsafe_ptr()
        if not planned:
            # the static plan: the dense prefix (`zero_ids` holds a copy
            # of the dense sequence; `ids_compute` holds the planner's
            # output from some earlier level and cannot be trusted here)
            level_ids = rebind[type_of(level_ids)](
                zero_ids.unsafe_ptr()
            )

        # zero the compute slots (their `ZeroLeavesHistograms`, applied to
        # every compute slot rather than the empty ones the host no longer
        # knows about; a computed slot is overwritten either way, and an
        # empty one's zeros ARE its histogram)
        times.begin(ctx)
        ctx.enqueue_function[zero_histograms_kernel](
            level_ids,
            Int32(hist_cells_per_leaf),
            hist.unsafe_ptr(),
            grid_dim=(
                (hist_cells_per_leaf + 255) // 256, n_compute, stat_count
            ),
            block_dim=(256, 1, 1),
        )
        mgr.stream_kernel()

        if planned:
            launch_histograms_for_blocks[hist2_smem_mode](
                ctx, dblocks, depth, n_compute, n_rows, stat_count,
                max_leaves, sm_count, fixed_scale,
                cindex, row_index, stats, p_off, p_sz, ids_compute,
                dense_ids,
                hist, acc_i32, block_hist, hist_cells_per_leaf,
            )
        else:
            launch_histograms_for_blocks[hist2_smem_mode](
                ctx, dblocks, depth, n_compute, n_rows, stat_count,
                max_leaves, sm_count, fixed_scale,
                cindex, row_index, stats, p_off, p_sz, zero_ids,
                dense_ids,
                hist, acc_i32, block_hist, hist_cells_per_leaf,
            )
        mgr.stream_kernel()

        # their `TScanHistogramsKernel` (`:1262`), over the computed set;
        # a prefix sum is linear, so the derived sibling needs no scan.
        ctx.enqueue_function[scan_histograms_kernel](
            level_ids,
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

        # their `SubstractHistograms(bigLeaves, smallLeaves, subsets)`
        # (`:1354`). `from - what`, in place, one launch for all pairs:
        # from = the right children (`dense_ids + half`, whose slots hold
        # the parent via `copy_histograms`), what = the computed left.
        if planned and half > 0:
            if hist_cells_per_leaf % 4 == 0:
                ctx.enqueue_function[substract_histograms_vec4_kernel](
                    sub_from.unsafe_ptr(), sub_what.unsafe_ptr(),
                    Int32(hist_cells_per_leaf), hist.unsafe_ptr(),
                    grid_dim=(
                        (hist_cells_per_leaf // 4 + 255) // 256,
                        half, stat_count
                    ),
                    block_dim=(256, 1, 1),
                )
            else:
                ctx.enqueue_function[substract_histograms_kernel](
                    sub_from.unsafe_ptr(), sub_what.unsafe_ptr(),
                    Int32(hist_cells_per_leaf), hist.unsafe_ptr(),
                    grid_dim=(
                        (hist_cells_per_leaf + 255) // 256,
                        half, stat_count
                    ),
                    block_dim=(256, 1, 1),
                )
            mgr.stream_kernel()

        times.end(ctx, "sym.hist")

        # ---- identity checkpoint: this depth's REDUCED histograms ----
        # `hist` is `[leaf][stat][binFeature]` leaf-major, and at this
        # point slots 0..n_live-1 hold the level's SCANNED histograms
        # (computed or sibling-derived); the tail holds deeper slots'
        # stale cells, so only the live prefix is hashed (identity_trace
        # rule 3). Records drain -- a traced run is not a timing (rule 4)
        # -- and sit OUTSIDE the timed regions.
        if trace.enabled:
            trace.record_device(
                ctx,
                tree_tag + ".depth" + _sym_dd2(depth) + ".hist",
                hist,
                count=n_live * stat_count * hist_cells_per_leaf,
            )

        # their `AllReduceThroughMaster(subsets->CurrentPartStats(), ...)`
        times.begin(ctx)
        compute_partition_stats(
            ctx, n_live, max_live_rows, stat_count, n_rows,
            dense_ids, p_off, p_sz, stats, stat_partials, part_stats,
            sm_count=sm_count,
        )
        mgr.stream_kernel()
        times.end(ctx, "sym.pstats")

        # ---- identity checkpoint: the level's per-leaf totals ---------
        # `part_stats` is the REDUCED `[leaf][stat]` result the score
        # kernel reads -- the logical buffer, never the machine-sized
        # `stat_partials` scratch (identity_trace rule 3 names exactly
        # that buffer as the wrong one).
        if trace.enabled:
            trace.record_device(
                ctx,
                tree_tag + ".depth" + _sym_dd2(depth) + ".pstats",
                part_stats,
                count=n_live * stat_count,
            )
        times.begin(ctx)

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
                Int32(n_live),
                Int32(1) if multiclass_optimization else Int32(0),
                l2_leaf_reg,
                # `ScoreStdDev, Random.NextUniformL()` -- the last two
                # arguments of every `ComputeOptimalSplits*` launch
                # (`greedy_search_helper.cpp:466-468`). The L2 calcer has
                # NO noise term at all (`score_calcers.cuh:40-69`), so the
                # constant zero here is their asymmetry and not a gap; the
                # seed is still handed over so both arms consume the same
                # stream.
                Float32(0.0),
                level_seed,
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
                Int32(n_live),
                Int32(1) if multiclass_optimization else Int32(0),
                l2_leaf_reg,
                # `ScoreStdDev, Random.NextUniformL()`
                # (`greedy_search_helper.cpp:466-468`). Cosine is the only
                # calcer of the five that reads them
                # (`score_calcers.cuh:159-167`).
                score_std_dev,
                level_seed,
                out_score.unsafe_ptr(), out_bin.unsafe_ptr(),
                grid_dim=(argmax_blocks, 1, 1),
                block_dim=(SCORE_BLOCK_SIZE, 1, 1),
            )
        mgr.stream_kernel()
        times.end(ctx, "sym.score")

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
        times.begin(ctx)
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
        times.end(ctx, "sym.winner")

        times.begin(ctx)
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
            sm_count=sm_count,
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
        # WIDTH DISPATCH, see `copy_histograms_vec4_kernel`'s deviation
        # block. MEASURED 11.0 -> 65.2 GB/s at a depth-6 level's shape.
        if (hist_cells_per_leaf * stat_count) % 4 == 0:
            ctx.enqueue_function[copy_histograms_vec4_kernel](
                dense_ids.unsafe_ptr(), ids_c.unsafe_ptr(),
                Int32(stat_count), Int32(hist_cells_per_leaf),
                hist.unsafe_ptr(),
                grid_dim=(
                    (hist_cells_per_leaf * stat_count // 4 + 255) // 256,
                    n_live, 1
                ),
                block_dim=(256, 1, 1),
            )
        else:
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
        # their `:397`, the same machine-sized expression. The fused
        # variant (DEVIATION 210) also writes next level's compute plan
        # from the border thread's registers, retiring the per-level
        # `plan_level_kernel` launch above.
        ctx.enqueue_function[update_partitions_and_plan_kernel](
            dense_ids.unsafe_ptr(), ids_c.unsafe_ptr(), Int32(n_live),
            sflags.unsafe_ptr(), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            hp_off.unsafe_ptr(), hp_sz.unsafe_ptr(),
            ids_compute.unsafe_ptr(),
            sub_from.unsafe_ptr(), sub_what.unsafe_ptr(),
            grid_dim=(split_points_grid_x(n_live, sm_count), n_live, 1),
            block_dim=(512, 1, 1),
        )
        mgr.stream_kernel()
        times.end(ctx, "sym.split")

        n_live = n_live * 2

    # ---- THE ONE DRAIN OF THE TREE (DEVIATION 94) --------------------
    # Their per-level `bestProps.Read` and `RebuildLeavesSizes` reads,
    # folded into one: the winner records hold every level, the partition
    # sizes hold the final leaves, and both ride home behind everything
    # the loop enqueued.
    times.begin(ctx)
    ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
    ctx.enqueue_copy(dst_ptr=h_wsc.unsafe_ptr(), src_buf=winners_score)
    ctx.enqueue_copy(dst_ptr=h_wbf.unsafe_ptr(), src_buf=winners_bf)
    mgr.wait_complete()
    times.end(ctx, "sym.drain")

    # ---- identity checkpoint: the tree's winner records --------------
    # Every level's score and packed bin-feature id, exactly as the one
    # drain brought them home; hashed from the HOST copies, so this adds
    # no device traffic of its own.
    trace.record_host(
        tree_tag + ".winners.scores", h_wsc.unsafe_ptr(), max_depth
    )
    trace.record_host(
        tree_tag + ".winners.bf", h_wbf.unsafe_ptr(), max_depth
    )

    # ============ THE GATES, POST-TREE ================================
    # Their host applies these BEFORE each split
    # (`greedy_search_helper.cpp:535-539` the sentinel ENSURE, `:360-364`
    # the `Score < 0` gate -- ours is `best_score > 0`, the sign flipped
    # as everywhere in this port -- and
    # `oblivious_tree_doc_parallel_structure_searcher.cpp:134` the
    # repeat-split stop). The walk applies them in LEVEL ORDER, so the
    # first stop at level k discards levels k.. exactly as their loop
    # would never have grown them, and the model is unchanged. The
    # rollback is the old one-level argument applied (max_depth - k)
    # times: the left child kept the parent's offset at every level, so
    # slot i < 2^k still holds level-k leaf i's offset, pairwise-merging
    # the final sizes (max_depth - k) times restores level-k sizes
    # exactly, and the reorders permuted rows only within ancestor
    # ranges, so every leaf sum the tail computes is unchanged.
    var grown = 0
    for depth2 in range(max_depth):
        var best_score = h_wsc.unsafe_ptr().unsafe_load(depth2)
        var best_bin_u = h_wbf.unsafe_ptr().unsafe_load(depth2)
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
                " [level " + String(depth2) + ", live leaves "
                + String(1 << depth2) + "]"
            )
        var choice = resolve_split(layout, Int(best_bin_u))
        # improving-score gate; for an oblivious tree all leaves share one
        # BestSplit, so one test is the whole gate.
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
            break
        out_splits.append(
            TBinarySplit(
                Int32(choice.feature),
                Int32(choice.bin),
                Int32(
                    BIN_SPLIT_TAKE_BIN
                    if layout.features[choice.feature].one_hot_feature
                    else BIN_SPLIT_TAKE_GREATER
                ),
            )
        )
        grown += 1

    if grown < max_depth:
        # merge the final sizes pairwise once per discarded level
        var live = n_live
        for _ in range(max_depth - grown):
            var h2 = live // 2
            for i in range(h2):
                var merged = (
                    h_sz.unsafe_ptr().unsafe_load(i)
                    + h_sz.unsafe_ptr().unsafe_load(h2 + i)
                )
                h_sz.unsafe_ptr().unsafe_store(i, merged)
            live = h2
        ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
        ctx.synchronize()
        n_live = live

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
        times.begin(ctx)
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

        # MACHINE-SIZED x (the kernel strides; DEVIATION 210b's repair
        # applied to this once-per-tree launch): `wide` was data-sized
        # and paid every leaf the same block count.
        ctx.enqueue_function[add_model_value_kernel](
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), row_index.unsafe_ptr(),
            leaf_values.unsafe_ptr(), learning_rate, cursor.unsafe_ptr(),
            Int32(1), Int32(0),
            grid_dim=(split_points_grid_x(n_live, sm_count), n_live, 1),
            block_dim=(256, 1, 1),
        )
        mgr.stream_kernel()
        times.end(ctx, "sym.leaves")

    ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
    mgr.wait_complete()
    if apply_to_cursor:
        # their `result[i].AddWeakModel(model)`, the half that keeps the tree
        # rather than only its effect on the cursor.
        for i in range(n_live):
            out_leaf_values.append(h_leaf_values.unsafe_ptr().unsafe_load(i))
        # ---- identity checkpoint: this tree's leaf values ------------
        # The Simple-method leaves (the searcher's own Newton step),
        # recorded AFTER the drain that made the host copy readable. The
        # estimation path's leaves are recorded by the walker's traced
        # overload instead (`descent_helpers.mojo`).
        trace.record_host(
            tree_tag + ".leaves", h_leaf_values.unsafe_ptr(), n_live
        )
    var out = List[Int]()
    for i in range(n_live):
        out.append(Int(h_sz.unsafe_ptr().unsafe_load(i)))

    # THE OFFSETS GO OUT WITH THE SIZES, and they are the DEVICE's, not a
    # prefix sum. The level gather splits each parent's segment IN PLACE,
    # so the final partitions sit in memory in BIT-REVERSED leaf order:
    # partition half+i's rows live inside its parent's old span, not after
    # partition half+i-1's. A caller that prefix-sums the sizes builds
    # segments for the WRONG leaves -- the Logloss estimator did exactly
    # that and trained a cursor its own model could not replay
    # (logloss_train_check claim 2, 2026-08-20). Read AFTER the sizes,
    # reusing h_sz, with a queue-ordered sync of its own: `wait_complete`
    # does not order copies enqueued after it was armed, and an offsets
    # copy slid in front of the sizes read handed the caller p_off twice
    # (caught by the same check's coverage going 34x over the row count).
    # Guarded because it costs one copy and one drain per tree, which the
    # RMSE arm has no reason to pay: its cursor update happened above.
    if export_offsets:
        ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_off)
        ctx.synchronize()
        out_leaf_offsets.clear()
        for i in range(n_live):
            out_leaf_offsets.append(Int(h_sz.unsafe_ptr().unsafe_load(i)))
    return out^


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
    mut out_leaf_offsets: List[Int],
    mut ws: List[TTreeWorkspace],
    export_offsets: Bool = False,
    use_subtraction: Bool = True,
    apply_to_cursor: Bool = False,
    learning_rate: Float32 = Float32(0.3),
    l2_leaf_reg: Float32 = Float32(3.0),
    sync_budget: Int = -1,
    one_hot: List[Bool] = List[Bool](),
    score_function: Int = SCORE_FUNCTION_COSINE,
    approx_dim: Int = 1,
    multiclass_optimization: Bool = False,
    mags_dev: Optional[DeviceBuffer[DType.float32]] = None,
    random_strength: Float32 = Float32(0.0),
    random_seed: UInt64 = UInt64(0),
) raises -> List[Int]:
    """The un-instrumented entry: the exact pre-instrumentation signature,
    forwarding to `run_tree_layout_traced` with BOTH instruments off.

    The `StageTimes` is FORCE-DISABLED rather than env-constructed on
    purpose: an env-enabled timer here would pay a drain per stage per
    level for a table nobody reports (the fit-level table lives with the
    boosting loop, which calls the traced entry directly). Same for the
    trace: a per-tree `IdentityTrace()` would restart `seq` at 0 in a
    shared trace file and break the format's monotonic-seq contract.
    """
    var no_trace = IdentityTrace.disabled()
    var no_times = StageTimes()
    no_times.enabled = False
    return run_tree_layout_traced[hist2_smem_mode](
        ctx, n_rows, fold_counts, max_depth,
        cindex, stats, row_index, cursor,
        weight_magnitude, gradient_magnitude,
        out_splits, out_leaf_values, out_leaf_offsets, ws,
        no_trace, no_times, String("tree"),
        export_offsets=export_offsets,
        use_subtraction=use_subtraction,
        apply_to_cursor=apply_to_cursor,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        sync_budget=sync_budget,
        one_hot=one_hot,
        score_function=score_function,
        approx_dim=approx_dim,
        multiclass_optimization=multiclass_optimization,
        mags_dev=mags_dev,
        random_strength=random_strength,
        random_seed=random_seed,
    )
