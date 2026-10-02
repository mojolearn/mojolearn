"""The symmetric level loop's launches, fused (lane hr-gbdt-small).

================= DEVIATION 3110 (scheduling only) =================
A small pool is LAUNCH AND SYNC BOUND: an Apple M4 tree at 320 to 20,000
rows cost 4 to 5 ms Plain, and nearly all of it was the fixed price of the
level loop's ~15 launches per level plus two drains per tree
(bench/results/gbdt_small_defaults_2026-09-22). This file folds launches of
`run_tree_layout_traced`'s level that have NO data dependency on each other,
or whose dependency is per row on the same thread, into one launch each.

THE PER-CELL STATEMENTS AND EVERY FOLD ORDER ARE THE UNFUSED KERNELS',
TRANSCRIBED. Nothing here adds, removes or reorders a float operation:

* `sym_scan_sub_pstats_kernel` is `scan_histograms_kernel`, then
  `substract_histograms(_vec4)_kernel` on the SAME feature's cells by the
  SAME thread (the sibling cell is `ftz(from - what)` with `what` the value
  the scan just stored, `max(., 0)` on stat 0), side by side with
  `partition_stats_partial(_gather)_kernel` in blocks of their own. The
  subtraction is per cell, so which thread runs it moves no bit; the
  features tile `[0, hist_cells)` exactly (`build_layout`: each feature
  owns `[first_fold_index, + folds)`, cursor-assigned), so every cell the
  cell-parallel kernel derived is derived here. The partial blocks run the
  unfused body at the unfused chunk index, block size (`STATS_BLOCK`),
  stride and `pinned_block_sum` fold, so each partial is the same float.
* `sym_resolve_split_count_kernel` is `resolve_and_pack_kernel`'s
  sequential winner scan (every thread, as it already was within its one
  block), its pack (block (0, 0) only, the same stores), then
  `split_and_make_sequence_kernel`'s flag and sequence stores for one
  partition chunk, then `partition_count_chunks_kernel`'s count of that
  chunk -- the flag the count reads is the one this thread just stored.
  Integer work only. The winner's descriptor lives in SCALAR registers:
  binding a whole `CFeature` to a local makes the Metal backend refuse
  the metallib (split_points.mojo's DEVIATION BLOCK; it broke this
  binding's Metal build on 2026-10-02 until the local went).
* `sym_copy_update_kernel` is `copy_histograms(_vec4)_kernel` and
  `update_partitions_and_plan_kernel` in disjoint blocks: a byte copy and
  an integer border search over disjoint buffers, each a grid-stride loop
  whose result does not depend on the grid.

So a fused level writes byte for byte what the unfused level writes, and
`-D MOJOLEARN_GBDT_FUSED_LEVEL_OFF` restores the unfused schedule for the
A/B (`SYM_FUSED_LEVEL` in `greedy_search_helper.mojo`).
====================================================================
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from max.gpu.primitives.block import broadcast as block_broadcast
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier

from checks.numerics import ftz
from gbdt.gpu_data.gpu_structures import CFeature
from gbdt.gpu_util.partitions_reduce import STATS_BLOCK
from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    FLOAT32_MAX,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_points import (
    PARTITION_BLOCK,
    SPLIT_BLOCK_SIZE,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_resolve import (
    WINNER_SENTINEL,
)
from gbdt.targets.kernel.pointwise_targets import pinned_block_sum

comptime FUSED_LEVEL_BLOCK = 512
"""One block size for every fused launch. It IS `STATS_BLOCK`,
`PARTITION_BLOCK` and `SPLIT_BLOCK_SIZE` (asserted below), the three
sizes a fused block inherits a fold or a chunk from; the scan and copy
halves are per-thread or grid-stride work for which the block size moves
no bit."""


@always_inline
def _assert_block_sizes():
    comptime assert FUSED_LEVEL_BLOCK == STATS_BLOCK, (
        "fused level block must be the partition-stats block"
    )
    comptime assert FUSED_LEVEL_BLOCK == PARTITION_BLOCK, (
        "fused level block must be the partition chunk"
    )
    comptime assert FUSED_LEVEL_BLOCK == SPLIT_BLOCK_SIZE, (
        "fused level block must be the split block"
    )


def sym_scan_sub_pstats_kernel[subtract: Bool, gather: Bool](
    # ---- scan (+ subtract): `scan_histograms_kernel`'s arguments ----
    hist_ids: MutPointer[UInt32, MutAnyOrigin],
    sub_from: MutPointer[UInt32, MutAnyOrigin],
    feature_first_bin: MutPointer[UInt32, MutAnyOrigin],
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_one_hot: MutPointer[UInt8, MutAnyOrigin],
    feature_count_in: Int32,
    bin_feature_count_in: Int32,
    histogram: MutPointer[Float32, MutAnyOrigin],
    scan_blocks_in: Int32,
    n_compute_in: Int32,
    # ---- partition stats phase 1: `partition_stats_partial_kernel`'s ----
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    line_size_in: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """Grid `(scan_blocks + max_chunks, n_live, stat_count)`. Blocks with
    `x < scan_blocks` scan (and, when `subtract`, derive the sibling of)
    the compute leaf `hist_ids[y]` for `y < n_compute`; the rest are
    partition-stats chunk `x - scan_blocks` of leaf slot `y`. Grid z is the
    stat on both halves, as it was on all three unfused kernels."""
    _assert_block_sizes()
    var tid = Int(thread_idx.x)
    var stat_id = Int(block_idx.z)
    var stat_count = Int(grid_dim.z)
    var scan_blocks = Int(scan_blocks_in)
    var bx = Int(block_idx.x)

    if bx < scan_blocks:
        if Int(block_idx.y) >= Int(n_compute_in):
            return
        var feature_count = Int(feature_count_in)
        var bin_feature_count = Int(bin_feature_count_in)
        var feature_id = bx * FUSED_LEVEL_BLOCK + tid
        if feature_id >= feature_count:
            return
        var leaf_id = Int(hist_ids.unsafe_load(Int(block_idx.y)))
        var folds = Int(feature_folds.unsafe_load(feature_id))
        var first = Int(feature_first_bin.unsafe_load(feature_id))
        var base = (
            leaf_id * bin_feature_count * stat_count
            + stat_id * bin_feature_count
        ) + first
        # their `skipFeature` (`histogram_utils.cu:395`), both halves:
        # the scan's own guard, unchanged
        var scanned = not (
            feature_one_hot.unsafe_load(feature_id) != UInt8(0) or folds <= 1
        )
        if scanned:
            var running = Scalar[DType.float32](0.0)
            for i in range(folds):
                running = ftz(running + histogram.unsafe_load(base + i))
                histogram.unsafe_store(base + i, running)

        comptime if subtract:
            # `substract_histograms_kernel` over this feature's cells of
            # pair `y`: `what` IS `hist_ids[y]` (the partition update
            # stores `ids_compute[j] = sub_what[j] = small`), so the
            # `what` cell is the one this thread just scanned (or, on a
            # skipped feature, never touched).
            var from_id = Int(sub_from.unsafe_load(Int(block_idx.y)))
            var from_base = (
                from_id * bin_feature_count * stat_count
                + stat_id * bin_feature_count
            ) + first
            for i in range(folds):
                var new_val = ftz(
                    histogram.unsafe_load(from_base + i)
                    - histogram.unsafe_load(base + i)
                )
                if stat_id == 0:
                    new_val = max(new_val, Scalar[DType.float32](0.0))
                histogram.unsafe_store(from_base + i, new_val)
        return

    # ---- `partition_stats_partial_kernel` (or its gather twin), verbatim
    var line_size = Int(line_size_in)
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaves.unsafe_load(leaf_slot))
    var n_stats = stat_count
    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))
    var chunk = bx - scan_blocks
    if chunk * STATS_BLOCK >= size:
        if tid == 0:
            partials.unsafe_store(
                (leaf_slot * n_stats + stat_id) * max_chunks + chunk,
                Float32(0.0),
            )
        return
    var stride = max_chunks * STATS_BLOCK
    var v = Float32(0.0)
    var i = chunk * STATS_BLOCK + tid
    while i < size:
        comptime if gather:
            var row = Int(row_index.unsafe_load(offset + i))
            v += stats.unsafe_load(stat_id * line_size + row)
        else:
            v += stats.unsafe_load(stat_id * line_size + offset + i)
        i += stride
    var total = pinned_block_sum[STATS_BLOCK](v)
    if tid == 0:
        partials.unsafe_store(
            (leaf_slot * n_stats + stat_id) * max_chunks + chunk, total
        )


def sym_resolve_split_count_kernel(
    # ---- `resolve_and_pack_kernel`'s arguments ----
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
    argmax_blocks_in: Int32,
    bf_offset: MutPointer[UInt32, MutAnyOrigin],
    bf_mask: MutPointer[UInt32, MutAnyOrigin],
    bf_shift: MutPointer[UInt32, MutAnyOrigin],
    bf_first: MutPointer[UInt32, MutAnyOrigin],
    bf_folds: MutPointer[UInt32, MutAnyOrigin],
    bf_one_hot: MutPointer[UInt8, MutAnyOrigin],
    bf_bin: MutPointer[UInt32, MutAnyOrigin],
    depth_in: Int32,
    n_live_in: Int32,
    winners_score: MutPointer[Float32, MutAnyOrigin],
    winners_bf: MutPointer[UInt32, MutAnyOrigin],
    sp_feats: MutPointer[UInt8, MutAnyOrigin],
    sp_bins: MutPointer[UInt32, MutAnyOrigin],
    ids_c: MutPointer[UInt32, MutAnyOrigin],
    # ---- `split_and_make_sequence_kernel`'s ----
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    load_indices: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    leaf_ids: MutPointer[UInt32, MutAnyOrigin],
    split_flags: MutPointer[UInt8, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    # ---- `partition_count_chunks_kernel`'s ----
    chunk_zeros: MutPointer[UInt32, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """Grid `(chunk_grid, n_live)` at `PARTITION_BLOCK` threads: the
    partition count's grid. See the module banner."""
    _assert_block_sizes()
    var tid = Int(thread_idx.x)
    var argmax_blocks = Int(argmax_blocks_in)
    var n_live = Int(n_live_in)

    # `resolve_and_pack_kernel`'s scan, VERBATIM: sequential over blocks,
    # strict `>`, tie to the smaller bin index, on every thread.
    var best_score = -FLOAT32_MAX
    var best_bin = WINNER_SENTINEL
    for bi in range(argmax_blocks):
        var b_score = out_score.unsafe_load(bi)
        var b_bin = out_bin.unsafe_load(bi)
        var take = b_score > best_score
        if b_score == best_score and b_bin < best_bin:
            take = True
        if take:
            best_score = b_score
            best_bin = b_bin

    var bf = 0
    if best_bin != WINNER_SENTINEL:
        bf = Int(best_bin)
    # the descriptor as scalars: binding a whole `CFeature` to a local is
    # what the Metal backend refuses (split_points.mojo's DEVIATION BLOCK)
    var d_off = bf_offset.unsafe_load(bf)
    var d_mask = bf_mask.unsafe_load(bf)
    var d_shift = bf_shift.unsafe_load(bf)
    var d_first = bf_first.unsafe_load(bf)
    var d_folds = bf_folds.unsafe_load(bf)
    var d_one_hot = bf_one_hot.unsafe_load(bf) != UInt8(0)
    var split_bin = bf_bin.unsafe_load(bf)

    # the pack, by ONE block: the same stores `resolve_and_pack_kernel`
    # makes, read by later launches only (the level's own split reads the
    # registers above)
    if Int(block_idx.x) == 0 and Int(block_idx.y) == 0:
        if tid == 0:
            winners_score.unsafe_store(Int(depth_in), best_score)
            winners_bf.unsafe_store(Int(depth_in), best_bin)
        var feats = sp_feats.bitcast[CFeature]()
        var p = tid
        while p < n_live:
            feats[unsafe_offset=p] = CFeature(
                offset=d_off,
                mask=d_mask,
                shift=d_shift,
                first_fold_index=d_first,
                folds=d_folds,
                one_hot_feature=d_one_hot,
            )
            sp_bins.unsafe_store(p, split_bin)
            ids_c.unsafe_store(p, UInt32(n_live + p))
            p += FUSED_LEVEL_BLOCK

    # `split_and_make_sequence_kernel`'s per-row statements on
    # `partition_count_chunks_kernel`'s chunk walk
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaf_ids.unsafe_load(leaf_slot))
    var size = Int(ldg(part_size + leaf_id))
    var offset = Int(ldg(part_offset + leaf_id))

    var f_offset = Int(d_off)
    var shift = d_shift
    var bin_idx = UInt32(split_bin)
    var one_hot = d_one_hot
    var value = bin_idx << shift
    var mask = d_mask << shift

    var chunk = Int(block_idx.x)
    while chunk * PARTITION_BLOCK < size:
        var at = chunk * PARTITION_BLOCK + tid
        var v = Int32(0)
        if at < size:
            var load_index = Int(ldg(load_indices + (offset + at)))
            var feature_val = (
                ldg(compressed_index + (f_offset + load_index)) & mask
            )
            indices.unsafe_store(offset + at, UInt32(at))
            var goes_right = (
                feature_val == value
            ) if one_hot else (feature_val > value)
            split_flags.unsafe_store(
                offset + at, UInt8(1) if goes_right else UInt8(0)
            )
            if not goes_right:
                v = Int32(1)
        var inc = block_prefix_sum[
            block_size=PARTITION_BLOCK, exclusive=False
        ](v)
        var total = block_broadcast[block_size=PARTITION_BLOCK](
            inc, src_thread = PARTITION_BLOCK - 1
        )
        if tid == 0:
            chunk_zeros.unsafe_store(
                leaf_slot * max_chunks + chunk, UInt32(Int(total))
            )
        barrier()
        chunk += Int(grid_dim.x)




def sym_copy_update_kernel[vec4: Bool](
    # ---- `copy_histograms(_vec4)_kernel`'s ----
    num_stats_in: Int32,
    bin_features_in_hist_in: Int32,
    histograms: MutPointer[Float32, MutAnyOrigin],
    copy_blocks_in: Int32,
    # ---- `update_partitions_and_plan_kernel`'s ----
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    sorted_flags: MutPointer[UInt8, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    host_offset: MutPointer[UInt32, MutAnyOrigin],
    host_size: MutPointer[UInt32, MutAnyOrigin],
    ids_compute: MutPointer[UInt32, MutAnyOrigin],
    sub_from: MutPointer[UInt32, MutAnyOrigin],
    sub_what: MutPointer[UInt32, MutAnyOrigin],
):
    """Grid `(copy_blocks + update_blocks, n_live)`; grid y is the pair on
    both halves, as it was on both unfused kernels (left = `dense_ids`,
    right = `ids_c`)."""
    _assert_block_sizes()
    var copy_blocks = Int(copy_blocks_in)
    var bx = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var leaf_slot = Int(block_idx.y)
    var left_leaf = Int(left_leaves.unsafe_load(leaf_slot))
    var right_leaf = Int(right_leaves.unsafe_load(leaf_slot))

    if bx < copy_blocks:
        var hist_size = Int(bin_features_in_hist_in) * Int(num_stats_in)
        var src = left_leaf * hist_size
        var dst = right_leaf * hist_size
        var i = bx * FUSED_LEVEL_BLOCK + tid
        var stride = copy_blocks * FUSED_LEVEL_BLOCK
        comptime if vec4:
            var n4 = hist_size >> 2
            while i < n4:
                var off = i << 2
                (histograms + dst).store[width=4](
                    off, (histograms + src).load[width=4](off)
                )
                i += stride
        else:
            while i < hist_size:
                histograms.unsafe_store(dst + i, histograms.unsafe_load(src + i))
                i += stride
        return

    # `update_partitions_and_plan_kernel`, verbatim, at block `bx - copy`
    var offset = Int(part_offset.unsafe_load(left_leaf))
    var part_sz = Int(part_size.unsafe_load(left_leaf))
    var i = (bx - copy_blocks) * FUSED_LEVEL_BLOCK + tid
    var stride = FUSED_LEVEL_BLOCK * (Int(grid_dim.x) - copy_blocks)
    while i <= part_sz:
        var flag0 = 1
        if i < part_sz:
            flag0 = Int(ldg(sorted_flags + (offset + i)))
        var flag1 = 0
        if i != 0:
            flag1 = Int(ldg(sorted_flags + (offset + i - 1)))
        if flag0 != flag1:
            part_size.unsafe_store(left_leaf, UInt32(i))
            host_offset.unsafe_store(left_leaf, UInt32(offset))
            host_size.unsafe_store(left_leaf, UInt32(i))

            part_offset.unsafe_store(right_leaf, UInt32(offset + i))
            part_size.unsafe_store(right_leaf, UInt32(part_sz - i))
            host_offset.unsafe_store(right_leaf, UInt32(offset + i))
            host_size.unsafe_store(right_leaf, UInt32(part_sz - i))

            var left_sz = UInt32(i)
            var right_sz = UInt32(part_sz - i)
            var small = UInt32(right_leaf)
            var big = UInt32(left_leaf)
            if left_sz < right_sz:
                small = UInt32(left_leaf)
                big = UInt32(right_leaf)
            ids_compute.unsafe_store(leaf_slot, small)
            sub_from.unsafe_store(leaf_slot, big)
            sub_what.unsafe_store(leaf_slot, small)
            break
        i += stride
