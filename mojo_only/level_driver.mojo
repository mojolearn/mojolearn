"""One complete oblivious level, end to end, on the GPU.

NO CATBOOST COUNTERPART as a file: this is the Mojo shape of
`TGreedySearchHelper::ComputeOptimalSplits` followed by `SplitLeaves`
(`greedy_search_helper.cpp:398`, `:534`), which in their tree is host C++
spread across a helper class. Ported as a function because we have no
`TPointsSubsets` object yet and the sequence is the thing worth having first.

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
    binary_hist_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    scan_histograms_kernel,
    zero_histograms_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
)
from ported.methods.greedy_subsets_searcher.kernel.split_points import (
    SPLIT_BLOCK_SIZE,
    gather_index_in_leaves_kernel,
    split_and_make_sequence_kernel,
    update_partitions_after_split_kernel,
)
from mojo_only.stable_partition import (
    PARTITION_BLOCK,
    stable_partition_kernel,
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
        grid_dim=(1, 1, 1),
        block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
    )

    # 6. STABLE PARTITION -------------------------------------------------
    var gmap = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var sflags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    ctx.enqueue_function[stable_partition_kernel](
        leaf0.unsafe_ptr(),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        flags.unsafe_ptr(),
        gmap.unsafe_ptr(),
        sflags.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(PARTITION_BLOCK, 1, 1),
    )

    # 7. GATHER the row index through the permutation ---------------------
    var new_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    ctx.enqueue_function[gather_index_in_leaves_kernel](
        leaf0.unsafe_ptr(),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        row_index.unsafe_ptr(),
        gmap.unsafe_ptr(),
        new_index.unsafe_ptr(),
        grid_dim=(1, 1, 1),
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
        grid_dim=(1, 1, 1),
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
