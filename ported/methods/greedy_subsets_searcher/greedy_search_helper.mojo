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

from max.gpu.host import DeviceBuffer, DeviceContext

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
    ctx.synchronize()

    # 1. ZERO -------------------------------------------------------------
    ctx.enqueue_function[zero_histograms_kernel](
        zero_ids.unsafe_ptr(),
        Int32(n_features),
        hist.unsafe_ptr(),
        grid_dim=(1, n_leaves, stat_count),
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
    ctx.enqueue_function[scan_histograms_kernel](
        fold_off.unsafe_ptr(),
        folds.unsafe_ptr(),
        Int32(n_features),
        Int32(n_features),
        hist.unsafe_ptr(),
        grid_dim=(1, n_leaves, stat_count),
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
    ctx.synchronize()

    var hos = ctx.enqueue_create_host_buffer[DType.float32](1)
    var hob = ctx.enqueue_create_host_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_ptr=hos.unsafe_ptr(), src_buf=out_score)
    ctx.enqueue_copy(dst_ptr=hob.unsafe_ptr(), src_buf=out_bin)
    ctx.synchronize()
    var best = Int(hob.unsafe_ptr().unsafe_load(0))

    # 5. SPLIT FLAGS ------------------------------------------------------
    var sp_off = ctx.enqueue_create_buffer[DType.uint32](1)
    var sp_shift = ctx.enqueue_create_buffer[DType.uint32](1)
    var sp_mask = ctx.enqueue_create_buffer[DType.uint32](1)
    var sp_hot = ctx.enqueue_create_buffer[DType.uint8](1)
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
    ctx.enqueue_copy(dst_buf=sp_off, src_ptr=a.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=sp_shift, src_ptr=b.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=sp_mask, src_ptr=c.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=sp_hot, src_ptr=d.unsafe_ptr())
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
        sp_off.unsafe_ptr(),
        sp_shift.unsafe_ptr(),
        sp_mask.unsafe_ptr(),
        sp_hot.unsafe_ptr(),
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
    ctx.synchronize()

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
    ctx.synchronize()

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
    total_weight: Float32,
    total_gradient: Float32,
    hist_replicas: Int = 1,
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
    # to fill the SMs (`hist_binary.cu:90-95`).
    #
    # **MEASURED AT 32 AND IT WAS SLOWER, so the default is 1.** It is a
    # PARAMETER so the two can be interleaved in one process rather than
    # compared across runs, which is the only comparison this box supports. Depth 8 went 70.3 ms
    # to 81.1 and depth 6 45.5 to 50.8, correctness unchanged. The reason is
    # the OUTPUT SIZE, not the input: 32 binary features is 32 bin-features,
    # so the histogram is 64 cells across two stat planes. Thirty-two replica
    # blocks of 512 threads is about 16,000 threads contending on 64 atomic
    # addresses, plus a conversion pass per level.
    #
    # CatBoost's replication pays because their histograms are thousands of
    # cells (100 features at up to 256 bins each). The rule that would make
    # this useful is bin-count-aware rather than constant, and writing one
    # without measuring the wide-feature shape would be a guess. The
    # fixed-point flush stays wired and correct so the experiment can be rerun
    # the moment there is a shape worth running it on.
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

    var sp1 = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var sp2 = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var sp3 = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var sp5 = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var sp4 = ctx.enqueue_create_buffer[DType.uint8](max_leaves)
    var hs1 = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hs2 = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hs3 = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hs5 = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hs4 = ctx.enqueue_create_host_buffer[DType.uint8](max_leaves)
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
                stats.unsafe_ptr(), Int32(n_rows),
                p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids_a.unsafe_ptr(),
                hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                Int32(max_leaves), Int32(stat_count),
                grid_dim=(hist_replicas, n_live, stat_count),
                block_dim=(BLOCK_SIZE, 1, 1),
            )
        else:
            ctx.enqueue_function[binary_hist_gather_kernel](
                folds.unsafe_ptr(), fold_off.unsafe_ptr(),
                grp_off.unsafe_ptr(), grp_sz.unsafe_ptr(),
                Int32(n_features), cindex.unsafe_ptr(), Int32(n_rows),
                row_index.unsafe_ptr(),
                stats.unsafe_ptr(), Int32(n_rows),
                p_off.unsafe_ptr(), p_sz.unsafe_ptr(), ids_a.unsafe_ptr(),
                hist.unsafe_ptr(), acc_i32.unsafe_ptr(), fixed_scale,
                Int32(max_leaves), Int32(stat_count),
                grid_dim=(hist_replicas, n_live, stat_count),
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
        for i in range(n_live):
            hs1.unsafe_ptr().unsafe_store(i, UInt32(0))
            hs2.unsafe_ptr().unsafe_store(i, UInt32(31 - best))
            hs3.unsafe_ptr().unsafe_store(i, UInt32(1))
            hs4.unsafe_ptr().unsafe_store(i, UInt8(0))
            hs5.unsafe_ptr().unsafe_store(i, UInt32(0))
        ctx.enqueue_copy(dst_buf=sp1, src_ptr=hs1.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=sp2, src_ptr=hs2.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=sp3, src_ptr=hs3.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=sp4, src_ptr=hs4.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=sp5, src_ptr=hs5.unsafe_ptr())
        ctx.synchronize()

        ctx.enqueue_function[split_and_make_sequence_kernel](
            cindex.unsafe_ptr(),
            row_index.unsafe_ptr(),
            p_off.unsafe_ptr(),
            p_sz.unsafe_ptr(),
            ids_a.unsafe_ptr(),
            sp1.unsafe_ptr(),
            sp2.unsafe_ptr(),
            sp3.unsafe_ptr(),
            sp4.unsafe_ptr(),
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
