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

from ported.gpu_lib.gpu_manager import TCudaManager
from std.sys.info import size_of
from ported.gpu_data.gpu_structures import CFeature

from ported.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.compute_scores import (
    SCORE_BLOCK_SIZE,
    compute_optimal_splits_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_binary import (
    binary_hist_gather_kernel,
    binary_hist_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    fixed_to_float_kernel,
    zero_histograms_kernel,
    write_reduces_histograms_kernel,
    scan_histograms_kernel,
    zero_histograms_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
)
from ported.methods.leaves_estimation.leaves_estimation import (
    LEAF_BLOCK,
    compute_leaf_values_kernel,
)
from mojo_only.kernel_matrix import replicas_for
from ported.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_half_byte import (
    half_byte_hist_gather_kernel,
    half_byte_hist_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    ONE_BYTE_BLOCK_SIZE,
    one_byte_hist_gather_kernel,
    one_byte_hist_kernel,
)
from ported.gpu_data.feature_blocks import PolicyBlock, blocks_for
from ported.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from ported.gpu_util.copy import (
    COPY_BLOCK,
    copy_f32_kernel,
    copy_u32_kernel,
)
from ported.gpu_util.partitions_reduce import (
    STATS_BLOCK,
    compute_partition_stats,
)
from ported.methods.greedy_subsets_searcher.kernel.split_points import (
    SPLIT_BLOCK_SIZE,
    gather_in_leaves_kernel,
    gather_index_in_leaves_kernel,
    split_and_make_sequence_kernel,
    update_partitions_after_split_kernel,
)
from ported.methods.leaves_estimation.leaves_estimation import (
    LEAF_BLOCK,
    compute_leaf_values_kernel,
)
from mojo_only.kernel_matrix import replicas_for
from ported.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_half_byte import (
    half_byte_hist_gather_kernel,
    half_byte_hist_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    ONE_BYTE_BLOCK_SIZE,
    one_byte_hist_gather_kernel,
    one_byte_hist_kernel,
)
from ported.gpu_data.feature_blocks import PolicyBlock, blocks_for
from ported.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from ported.gpu_util.copy import (
    COPY_BLOCK,
    copy_f32_kernel,
    copy_u32_kernel,
)
from ported.gpu_util.partitions_reduce import (
    STATS_BLOCK,
    compute_partition_stats,
)
from ported.methods.greedy_subsets_searcher.kernel.split_points import (
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

    # --- histogram buffer: [leaf][stat][binFeature] ----------------------
    var hist_cells = n_leaves * stat_count * n_features
    var hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
    # run_one_level is depth 0 with one block per partition, so the
    # multi-block flush never fires; the scratch satisfies the signature.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](hist_cells)
    var zero_ids = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var hz = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    for i in range(n_leaves):
        hz.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=zero_ids, src_ptr=hz.unsafe_ptr())

    # Their `FitImpl` blocks the host exactly TWICE per level:
    # `bestProps.Read(propsCpu)` (`greedy_search_helper.cpp:517`) and the
    # leaf-size read in `RebuildLeavesSizes`
    # (`split_properties_helper.cpp:800`). Everything between them is one
    # stream, and stream order is enough: measured 54 chained increments
    # behind a single drain came back exact. Depth 0 is ONE level, so the
    # budget is two and a third drain raises rather than warns.
    var mgr = TCudaManager(ctx.copy(), sync_budget=2)

    # 1. ZERO -------------------------------------------------------------
    ctx.enqueue_function[zero_histograms_kernel](
        zero_ids.unsafe_ptr(),
        Int32(n_features),
        hist.unsafe_ptr(),
        grid_dim=(1, n_leaves, stat_count),
        block_dim=(256, 1, 1),
    )
    mgr.stream_kernel()

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
    mgr.stream_kernel()

    # 3. SCAN. Binary features are one fold, so the scan is a no-op here and
    # is issued anyway: the sequence is what is being exercised, and leaving
    # a step out because this shape does not need it is how an ordering bug
    # hides until a wider feature appears.
    ctx.enqueue_function[scan_histograms_kernel](
        fold_off.unsafe_ptr(),
        folds.unsafe_ptr(),
        Int32(n_features),
        Int32(n_features),
        hist.unsafe_ptr(),
        grid_dim=(1, n_leaves, stat_count),
        block_dim=(256, 1, 1),
    )
    mgr.stream_kernel()

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

    ctx.enqueue_function[compute_optimal_splits_kernel](
        skip.unsafe_ptr(),
        Int32(n_features),
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
    mgr.stream_kernel()

    var hos = ctx.enqueue_create_host_buffer[DType.float32](1)
    var hob = ctx.enqueue_create_host_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_ptr=hos.unsafe_ptr(), src_buf=out_score)
    ctx.enqueue_copy(dst_ptr=hob.unsafe_ptr(), src_buf=out_bin)

    # DRAIN 1 of 2. Their `bestProps.Read(propsCpu)`
    # (`greedy_search_helper.cpp:517`). The host cannot build the CFeature
    # shift without the argmax, so this drain is theirs and stays.
    mgr.wait_complete()
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
    mgr.stream_kernel()

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
    mgr.stream_kernel()

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
    mgr.stream_kernel()

    # 8. UPDATE PARTITIONS ------------------------------------------------
    var left = ctx.enqueue_create_buffer[DType.uint32](1)
    var right = ctx.enqueue_create_buffer[DType.uint32](1)
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var hr = ctx.enqueue_create_host_buffer[DType.uint32](1)
    hl.unsafe_ptr().unsafe_store(0, UInt32(0))
    hr.unsafe_ptr().unsafe_store(0, UInt32(1))
    ctx.enqueue_copy(dst_buf=left, src_ptr=hl.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=right, src_ptr=hr.unsafe_ptr())

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
    mgr.stream_kernel()

    var osz = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    ctx.enqueue_copy(dst_ptr=osz.unsafe_ptr(), src_buf=p_sz)

    # DRAIN 2 of 2. Their `RebuildLeavesSizes`
    # (`split_properties_helper.cpp:800`). They read it off pinned host
    # memory; we pay a copy, which UNWIRED.md measured as the cheaper of
    # the two on this stack.
    mgr.wait_complete()

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
    total_weight: Float32,
    total_gradient: Float32,
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
    """
    var stat_count = 2
    var max_leaves = 1 << max_depth

    var wide = (n_rows + 255) // 256
    if wide > 256:
        wide = 256
    if wide < 1:
        wide = 1

    # --- persistent state, allocated once --------------------------------
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

    # FIXED-POINT ACCUMULATOR for replicated blocks. Metal has no float
    # atomic, so partial histograms from blocks sharing a partition sum as
    # Int32 through an integer atomic and are converted back afterwards.
    # Integer addition is associative, so the histogram does not depend on
    # which block lands first.
    var acc_i32 = ctx.enqueue_create_buffer[DType.int32](hist_cells)
    var zi = ctx.enqueue_create_host_buffer[DType.int32](hist_cells)
    for i in range(hist_cells):
        zi.unsafe_ptr().unsafe_store(i, Int32(0))
    ctx.enqueue_copy(dst_buf=acc_i32, src_ptr=zi.unsafe_ptr())

    # The scale bounds every partial sum. A cell can hold at most the whole
    # dataset's weight or the sum of gradient magnitudes, and both are known
    # here, so a scale derived from them cannot overflow at any depth: a
    # leaf's rows are a subset of all rows. See mojo_only/fixed_point.mojo.
    var mag = Float64(total_weight)
    var gmag = Float64(total_gradient)
    if gmag < 0.0:
        gmag = -gmag
    if gmag > mag:
        mag = gmag
    if mag <= 0.0:
        mag = 1.0
    var fixed_scale = Float32(Float64((1 << 28) - 1) / mag)

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
    # `replicas_for` in the matrix owns it. Passing 0 here asks the matrix;
    # a positive value overrides it so the two can still be interleaved.
    var replicas = hist_replicas
    if replicas <= 0:
        replicas = replicas_for(stat_count * n_features)

    var part_stats = ctx.enqueue_create_buffer[DType.float32](
        max_leaves * stat_count
    )
    var stat_chunks = (n_rows + STATS_BLOCK - 1) // STATS_BLOCK
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

    var skip = ctx.enqueue_create_buffer[DType.uint8](n_features)
    var hsk = ctx.enqueue_create_host_buffer[DType.uint8](n_features)
    for f in range(n_features):
        hsk.unsafe_ptr().unsafe_store(f, UInt8(0))
    ctx.enqueue_copy(dst_buf=skip, src_ptr=hsk.unsafe_ptr())

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

        ctx.enqueue_function[zero_histograms_kernel](
            ids_a.unsafe_ptr(),
            Int32(n_features),
            hist.unsafe_ptr(),
            grid_dim=(1, n_live, stat_count),
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
                grid_dim=(replicas, n_live, stat_count),
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
                grid_dim=(replicas, n_live, stat_count),
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
        ctx.enqueue_function[scan_histograms_kernel](
            fold_off.unsafe_ptr(),
            folds.unsafe_ptr(),
            Int32(n_features),
            Int32(n_features),
            hist.unsafe_ptr(),
            grid_dim=(1, n_live, stat_count),
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
        )
        ctx.synchronize()

        # ---- 3. one split for the whole level ---------------------------
        ctx.enqueue_function[compute_optimal_splits_kernel](
            skip.unsafe_ptr(),
            Int32(n_features),
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

        ctx.enqueue_function[gather_index_in_leaves_kernel](
            ids_a.unsafe_ptr(),
            p_off.unsafe_ptr(),
            p_sz.unsafe_ptr(),
            row_index.unsafe_ptr(),
            gmap.unsafe_ptr(),
            new_index.unsafe_ptr(),
            grid_dim=(wide, n_live, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[copy_u32_kernel](
            row_index.unsafe_ptr(),
            new_index.unsafe_ptr(),
            Int32(n_rows),
            grid_dim=wide,
            block_dim=COPY_BLOCK,
        )

        # The STAT columns are permuted too, and this kernel was ported and
        # never called until now. The histogram reads bins THROUGH the index
        # and stats BY POSITION, so leaving stats unpermuted pairs the right
        # rows' bins with the wrong rows' gradients.
        ctx.enqueue_function[gather_in_leaves_kernel](
            ids_a.unsafe_ptr(), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            stats.unsafe_ptr(), gmap.unsafe_ptr(), new_stats.unsafe_ptr(),
            Int32(stat_count), Int32(n_rows),
            grid_dim=(wide, n_live, 1), block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[copy_f32_kernel](
            stats.unsafe_ptr(),
            new_stats.unsafe_ptr(),
            Int32(stat_count * n_rows),
            grid_dim=wide,
            block_dim=COPY_BLOCK,
        )

        # ---- 5. one leaf becomes two ------------------------------------
        # Children are laid out so leaf i becomes (2i, 2i+1), which is what
        # keeps the partition ranges contiguous and ascending.
        for i in range(n_live):
            h_ids_b.unsafe_ptr().unsafe_store(i, UInt32(2 * i))
            h_ids_c.unsafe_ptr().unsafe_store(i, UInt32(2 * i + 1))
        ctx.enqueue_copy(dst_buf=ids_b, src_ptr=h_ids_b.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=ids_c, src_ptr=h_ids_c.unsafe_ptr())
        ctx.synchronize()

        # The parent's {Offset, Size} must sit at the LEFT child's index
        # before the border search runs, since it shrinks left in place.
        ctx.enqueue_copy(dst_ptr=h_off.unsafe_ptr(), src_buf=p_off)
        ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
        ctx.synchronize()
        for i in range(n_live - 1, -1, -1):
            var o = h_off.unsafe_ptr().unsafe_load(i)
            var s = h_sz.unsafe_ptr().unsafe_load(i)
            h_off.unsafe_ptr().unsafe_store(2 * i, o)
            h_sz.unsafe_ptr().unsafe_store(2 * i, s)
            h_off.unsafe_ptr().unsafe_store(2 * i + 1, o)
            h_sz.unsafe_ptr().unsafe_store(2 * i + 1, UInt32(0))
        ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
        ctx.synchronize()

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
        ctx.synchronize()

        n_live = n_live * 2

        # O(leaves) on the host, never O(rows). See HOST_AND_DEVICE.md.
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
    )
    var leaf_values = ctx.enqueue_create_buffer[DType.float32](max_leaves)
    ctx.enqueue_function[compute_leaf_values_kernel](
        part_stats.unsafe_ptr(),
        Int32(stat_count),
        Int32(n_live),
        Float32(1.0),
        Float32(1.0e6),
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


def launch_one_byte[bits: Int](
    ctx: DeviceContext,
    mut blk: DeviceBlock,
    depth: Int,
    n_live: Int,
    n_rows: Int,
    stat_count: Int,
    max_leaves: Int,
    replicas: Int,
    line: Int,
    base: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
    mut block_hist: DeviceBuffer[DType.float32],
) raises:
    """One one-byte launch at a comptime bit width. Direct at depth 0,
    gather below it."""
    if depth == 0:
        ctx.enqueue_function[one_byte_hist_kernel[bits]](
            blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
            blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
            Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line),
            Int32(base), stats.unsafe_ptr(), Int32(n_rows),
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
            block_hist.unsafe_ptr(),
            Int32(max_leaves), Int32(stat_count),
            grid_dim=(replicas, n_live, stat_count),
            block_dim=(ONE_BYTE_BLOCK_SIZE, 1, 1),
        )
    else:
        ctx.enqueue_function[one_byte_hist_gather_kernel[bits]](
            blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
            blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
            Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line),
            Int32(base), row_index.unsafe_ptr(),
            stats.unsafe_ptr(), Int32(n_rows),
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
            block_hist.unsafe_ptr(),
            Int32(max_leaves), Int32(stat_count),
            grid_dim=(replicas, n_live, stat_count),
            block_dim=(ONE_BYTE_BLOCK_SIZE, 1, 1),
        )


def launch_histograms_for_blocks(
    ctx: DeviceContext,
    mut blocks: List[DeviceBlock],
    depth: Int,
    n_live: Int,
    n_rows: Int,
    stat_count: Int,
    max_leaves: Int,
    replicas: Int,
    fixed_scale: Float32,
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
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
    """
    # Where each block's slice begins in the flat histogram: the running
    # total of earlier blocks' bin counts.
    var block_first_bin = 0

    # ZERO THE BLOCK SCRATCH before every block. Their writeback is guarded
    # by `if (abs(val) > 1e-20f)`, so a cell whose value is zero is NEVER
    # WRITTEN and keeps whatever the buffer held. CatBoost zeroes through
    # `ZeroHistograms` for the same reason; this port zeroed the FLAT
    # histogram and not the per-block scratch the kernels actually write.
    #
    # It also means the scratch cannot be shared between blocks without
    # clearing: block 2 would inherit block 1's cells wherever its own value
    # rounds to zero.

    for b in range(len(blocks)):
        ref blk = blocks[b]
        # The policy's column BASE, and the feature-block stride within it.
        var base = n_rows * blk.first_column
        var line = n_rows

        ctx.enqueue_function[zero_histograms_kernel](
            ids.unsafe_ptr(),
            Int32(blk.total_folds),
            block_hist.unsafe_ptr(),
            grid_dim=((blk.total_folds + 255) // 256, n_live, stat_count),
            block_dim=(256, 1, 1),
        )

        # Each block writes its OWN layout, [leaf][stat][bin-in-block], into
        # scratch. Writing straight into the flat histogram is correct only
        # when there is one block, because the writeback strides by this
        # block's `group_size` and the flat array strides by the total.
        var block_cells = max_leaves * stat_count * blk.total_folds

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
                    grid_dim=(replicas, n_live, stat_count),
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
                    grid_dim=(replicas, n_live, stat_count),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
        elif blk.policy == POLICY_HALF_BYTE:
            if depth == 0:
                ctx.enqueue_function[half_byte_hist_kernel](
                    blk.folds.unsafe_ptr(), blk.fold_off.unsafe_ptr(),
                    blk.grp_off.unsafe_ptr(), blk.grp_sz.unsafe_ptr(),
                    Int32(blk.n_features), cindex.unsafe_ptr(), Int32(line), Int32(base),
                    stats.unsafe_ptr(), Int32(n_rows),
                    p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids.unsafe_ptr(),
                    block_hist.unsafe_ptr(),
                    Int32(max_leaves), Int32(stat_count),
                    grid_dim=(1, n_live, stat_count),
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
                    block_hist.unsafe_ptr(),
                    Int32(max_leaves), Int32(stat_count),
                    grid_dim=(1, n_live, stat_count),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
        else:
            # One-byte is comptime-parameterized by bit width, and the width
            # must MATCH the block's fold count.
            #
            # This previously dispatched [8] for every one-byte block with a
            # comment calling the width a tuning question. A byte-level probe
            # falsified that: a 64-fold block dispatched at 8 bits returned 2
            # of 4 features with wrong counts, while every standalone check
            # passed at a width MATCHED to its folds and never mismatched.
            #
            # `bits` sets `InnerHistBitsCount = bits - 5`, which decides the
            # slot arithmetic `(bin >> 5) & mask` AND the number of passes, so
            # a width wider than the data changes where bins land.
            var ob = 5
            if blk.max_folds > 128:
                ob = 8
            elif blk.max_folds > 64:
                ob = 7
            elif blk.max_folds > 32:
                ob = 6

            if ob == 5:
                launch_one_byte[5](
                    ctx, blk, depth, n_live, n_rows, stat_count, max_leaves,
                    replicas, line, base, cindex, row_index, stats, p_off,
                    p_sz, ids, block_hist,
                )
            elif ob == 6:
                launch_one_byte[6](
                    ctx, blk, depth, n_live, n_rows, stat_count, max_leaves,
                    replicas, line, base, cindex, row_index, stats, p_off,
                    p_sz, ids, block_hist,
                )
            elif ob == 7:
                launch_one_byte[7](
                    ctx, blk, depth, n_live, n_rows, stat_count, max_leaves,
                    replicas, line, base, cindex, row_index, stats, p_off,
                    p_sz, ids, block_hist,
                )
            else:
                launch_one_byte[8](
                    ctx, blk, depth, n_live, n_rows, stat_count, max_leaves,
                    replicas, line, base, cindex, row_index, stats, p_off,
                    p_sz, ids, block_hist,
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


def run_tree_layout(
    ctx: DeviceContext,
    n_rows: Int,
    fold_counts: List[Int],
    max_depth: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut row_index: DeviceBuffer[DType.uint32],
    total_weight: Float32,
    total_gradient: Float32,
) raises -> List[Int]:
    """`FitImpl` over a LAYOUT: mixed feature widths, one launch per policy.

    Same level sequence as `run_tree`, which stays as the verified uniform
    binary path. What differs is that the histogram comes from
    `launch_histograms_for_blocks` and the winning bin-feature is resolved
    through `resolve_split`, so a dataset may mix binary, half-byte and
    one-byte features.

    Returns the final leaf sizes, which must sum to `n_rows` at every depth.
    """
    var stat_count = 2
    var max_leaves = 1 << max_depth

    var layout = build_layout(fold_counts)
    var blocks = blocks_for(layout, n_rows)
    var dblocks = upload_blocks(ctx, blocks)
    var hist_cells_per_leaf = layout.hist_cells

    var replicas = replicas_for(stat_count * hist_cells_per_leaf)

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
    var zi = ctx.enqueue_create_host_buffer[DType.int32](hist_cells)
    for i in range(hist_cells):
        zi.unsafe_ptr().unsafe_store(i, Int32(0))
    ctx.enqueue_copy(dst_buf=acc_i32, src_ptr=zi.unsafe_ptr())

    var mag = Float64(total_weight)
    var gm = Float64(total_gradient)
    if gm < 0.0:
        gm = -gm
    if gm > mag:
        mag = gm
    if mag <= 0.0:
        mag = 1.0
    var fixed_scale = Float32(Float64((1 << 28) - 1) / mag)

    var part_stats = ctx.enqueue_create_buffer[DType.float32](
        max_leaves * stat_count
    )
    var stat_chunks = (n_rows + STATS_BLOCK - 1) // STATS_BLOCK
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

    var zero_ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)

    var skip = ctx.enqueue_create_buffer[DType.uint8](hist_cells_per_leaf)
    var hsk = ctx.enqueue_create_host_buffer[DType.uint8](
        hist_cells_per_leaf
    )
    for i in range(hist_cells_per_leaf):
        hsk.unsafe_ptr().unsafe_store(i, UInt8(0))
    ctx.enqueue_copy(dst_buf=skip, src_ptr=hsk.unsafe_ptr())

    var out_score = ctx.enqueue_create_buffer[DType.float32](1)
    var out_bin = ctx.enqueue_create_buffer[DType.uint32](1)
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
    var hff = ctx.enqueue_create_host_buffer[DType.uint32](len(fold_counts))
    var hfd = ctx.enqueue_create_host_buffer[DType.uint32](len(fold_counts))
    for i in range(len(fold_counts)):
        hff.unsafe_ptr().unsafe_store(i, layout.features[i].first_fold_index)
        hfd.unsafe_ptr().unsafe_store(i, layout.features[i].folds)
    ctx.enqueue_copy(dst_buf=flat_first, src_ptr=hff.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=flat_folds, src_ptr=hfd.unsafe_ptr())
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
    # Two phases per level, and theirs blocks the host exactly TWICE in them:
    # `bestProps.Read(propsCpu)` (`greedy_search_helper.cpp:517`) and the
    # leaf-size read in `RebuildLeavesSizes`
    # (`split_properties_helper.cpp:802`). This loop now does the same two
    # and nothing else, and the budget below turns a third into an error.
    # ================================================================
    var mgr = TCudaManager(ctx.copy(), sync_budget=2 * max_depth + 1)

    var n_live = 1
    var max_live_rows = n_rows

    for depth in range(max_depth):
        # ============ ComputeOptimalSplits ============
        # their `greedy_search_helper.cpp:398`
        for i in range(n_live):
            h_ids_a.unsafe_ptr().unsafe_store(i, UInt32(i))
        ctx.enqueue_copy(dst_buf=ids_a, src_ptr=h_ids_a.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=zero_ids, src_ptr=h_ids_a.unsafe_ptr())

        ctx.enqueue_function[zero_histograms_kernel](
            zero_ids.unsafe_ptr(),
            Int32(hist_cells_per_leaf),
            hist.unsafe_ptr(),
            grid_dim=(
                (hist_cells_per_leaf + 255) // 256, n_live, stat_count
            ),
            block_dim=(256, 1, 1),
        )
        mgr.stream_kernel()

        # their `SplitPropsHelper.BuildNecessaryHistograms(subsets)`
        launch_histograms_for_blocks(
            ctx, dblocks, depth, n_live, n_rows, stat_count, max_leaves,
            replicas, fixed_scale,
            cindex, row_index, stats, p_off, p_sz, ids_a, hist, acc_i32,
            block_hist, hist_cells_per_leaf,
        )
        mgr.stream_kernel()

        ctx.enqueue_function[fixed_to_float_kernel](
            acc_i32.unsafe_ptr(), hist.unsafe_ptr(), Int32(hist_cells),
            fixed_scale,
            grid_dim=(hist_cells + 255) // 256, block_dim=256,
        )
        mgr.stream_kernel()
        ctx.enqueue_function[scan_histograms_kernel](
            flat_first.unsafe_ptr(), flat_folds.unsafe_ptr(),
            Int32(len(fold_counts)), Int32(hist_cells_per_leaf),
            hist.unsafe_ptr(),
            grid_dim=(1, n_live, stat_count), block_dim=(256, 1, 1),
        )
        mgr.stream_kernel()

        # their `AllReduceThroughMaster(subsets->CurrentPartStats(), ...)`
        compute_partition_stats(
            ctx, n_live, max_live_rows, stat_count, n_rows,
            ids_a, p_off, p_sz, stats, stat_partials, part_stats,
        )
        mgr.stream_kernel()

        ctx.enqueue_function[compute_optimal_splits_kernel](
            skip.unsafe_ptr(), Int32(hist_cells_per_leaf), hist.unsafe_ptr(),
            part_stats.unsafe_ptr(), Int32(stat_count), ids_a.unsafe_ptr(),
            Int32(n_live), Float32(3.0),
            out_score.unsafe_ptr(), out_bin.unsafe_ptr(),
            grid_dim=(1, 1, 1), block_dim=(SCORE_BLOCK_SIZE, 1, 1),
        )
        mgr.stream_kernel()
        ctx.enqueue_copy(dst_ptr=hob.unsafe_ptr(), src_buf=out_bin)

        # DRAIN 1 of 2. Their `bestProps.Read(propsCpu)`
        # (`greedy_search_helper.cpp:517`). The host cannot pick the split
        # without the argmax, so this one is theirs too and stays.
        mgr.wait_complete()

        var best_bf = Int(hob.unsafe_ptr().unsafe_load(0))
        if best_bf < 0 or best_bf >= hist_cells_per_leaf:
            best_bf = 0
        var choice = resolve_split(layout, best_bf)
        ref cf = layout.features[choice.feature]

        # ============ SplitLeaves -> MakeSplit ============
        # their `greedy_search_helper.cpp:575` and
        # `split_properties_helper.cpp:833`.
        #
        # THE LEAF NUMBERING IS THEIRS NOW:
        #
        #     const ui32 leftId  = leavesToSplit[i];
        #     const ui32 rightId = leavesCount + i;
        #
        # The left child KEEPS its parent's slot and the right child is
        # appended at the end. That is the whole reason their partition
        # array never round trips: `UpdatePartitionsAfterSplitImpl`
        # (`split_points.cu:346`) reads `parts[leftLeaf]`, which still holds
        # the parent, and writes both children from it.
        #
        # Ours used to number children 2i and 2i+1, which needs the parent
        # spread outward before the kernel can run, and that spread was a
        # host loop between a device-to-host read and a host-to-device write.
        # Adopting their numbering deletes the read, the loop and the write.
        var hfeat = sp_feats_h.unsafe_ptr().bitcast[CFeature]()
        for i in range(n_live):
            # Their `splitsFeaturesBuilder.Add(DataSet.GetTCFeature(...))`
            # and `splitBins.push_back(splitFeature.BinIdx)`
            # (`split_properties_helper.cpp:872-873`), one entry per leaf.
            # `offset` is scaled to the row stride here because our
            # compressed index is addressed in rows, not in their columns.
            hfeat[unsafe_offset=i] = (
                CFeature(
                    offset=cf.offset * UInt32(n_rows),
                    mask=cf.mask,
                    shift=cf.shift,
                    first_fold_index=cf.first_fold_index,
                    folds=cf.folds,
                    one_hot_feature=cf.one_hot_feature,
                )
            )
            sp_bins_h.unsafe_ptr().unsafe_store(i, UInt32(choice.bin))
            # their leftIds / rightIds
            h_ids_b.unsafe_ptr().unsafe_store(i, UInt32(i))
            h_ids_c.unsafe_ptr().unsafe_store(i, UInt32(n_live + i))
        ctx.enqueue_copy(dst_buf=sp_feats, src_ptr=sp_feats_h.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=sp_bins, src_ptr=sp_bins_h.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=ids_b, src_ptr=h_ids_b.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=ids_c, src_ptr=h_ids_c.unsafe_ptr())

        ctx.enqueue_function[split_and_make_sequence_kernel](
            cindex.unsafe_ptr(), row_index.unsafe_ptr(),
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids_a.unsafe_ptr(),
            sp_feats.unsafe_ptr().bitcast[CFeature](), sp_bins.unsafe_ptr(),
            flags.unsafe_ptr(), seq.unsafe_ptr(),
            grid_dim=(wide, n_live, 1),
            block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )
        mgr.stream_kernel()

        launch_stable_partition(
            ctx, n_live, max_live_rows, ids_a, p_off, p_sz, flags,
            chunk_zeros, chunk_offsets, leaf_zeros, gmap, sflags,
        )
        mgr.stream_kernel()

        ctx.enqueue_function[gather_index_in_leaves_kernel](
            ids_a.unsafe_ptr(), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            row_index.unsafe_ptr(), gmap.unsafe_ptr(), new_index.unsafe_ptr(),
            grid_dim=(wide, n_live, 1), block_dim=(256, 1, 1),
        )
        mgr.stream_kernel()
        ctx.enqueue_function[copy_u32_kernel](
            row_index.unsafe_ptr(), new_index.unsafe_ptr(), Int32(n_rows),
            grid_dim=wide, block_dim=COPY_BLOCK,
        )
        mgr.stream_kernel()
        ctx.enqueue_function[gather_in_leaves_kernel](
            ids_a.unsafe_ptr(), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            stats.unsafe_ptr(), gmap.unsafe_ptr(), new_stats.unsafe_ptr(),
            Int32(stat_count), Int32(n_rows),
            grid_dim=(wide, n_live, 1), block_dim=(256, 1, 1),
        )
        mgr.stream_kernel()
        ctx.enqueue_function[copy_f32_kernel](
            stats.unsafe_ptr(), new_stats.unsafe_ptr(),
            Int32(stat_count * n_rows),
            grid_dim=wide, block_dim=COPY_BLOCK,
        )
        mgr.stream_kernel()

        # their `UpdatePartitionsAfterSplit` (`split_points.cu:387`), reached
        # with NO host arithmetic in front of it because the left child kept
        # the parent's slot.
        ctx.enqueue_function[update_partitions_after_split_kernel](
            ids_b.unsafe_ptr(), ids_c.unsafe_ptr(), Int32(n_live),
            sflags.unsafe_ptr(), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            hp_off.unsafe_ptr(), hp_sz.unsafe_ptr(),
            grid_dim=(wide, n_live, 1), block_dim=(512, 1, 1),
        )
        mgr.stream_kernel()

        n_live = n_live * 2
        ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)

        # DRAIN 2 of 2. Their `RebuildLeavesSizes`
        # (`split_properties_helper.cpp:800`). They get it for free off
        # pinned host memory; we pay a copy. See ported/gpu_lib/NOT_PORTED.md
        # for the measurement that says the copy is the cheaper of the two
        # on this stack.
        mgr.wait_complete()

        max_live_rows = 1
        for i in range(n_live):
            var s = Int(h_sz.unsafe_ptr().unsafe_load(i))
            if s > max_live_rows:
                max_live_rows = s

    ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
    mgr.wait_complete()
    var out = List[Int]()
    for i in range(n_live):
        out.append(Int(h_sz.unsafe_ptr().unsafe_load(i)))
    return out^
