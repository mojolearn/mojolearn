"""The non-symmetric split chain in four launches (FAST Apple default).

Depthwise only; off with `-D MOJOLEARN_GBDT_DW_FUSED_CHAIN_OFF` (lane
apple-fast-depthwise). The chain
`greedy_search_helper_depthwise.mojo` runs per level on the row-index-only
schedule (DEVIATION 1902) is eight launches:

    update_partition_stats_from_split   (1, n_split) x 32
    split_and_make_sequence             flags + a sequence nobody reads
    partition count / scan / place      gather_map + sorted_flags
    copy_index + gather_index           row_index -> temp -> row_index
    update_partitions_after_split       border search over sorted_flags

Here it is four, with the same row permutation, the same partitions and the
same partition stats, bit for bit:

    K1 `fused_flags_count_kernel`   the side test of every row (written to
        `flags`) and each chunk's zero count, in one pass. The sequence plane
        and the separate count pass are gone.
    K2 `fused_scan_update_kernel`   one block per split leaf: the chunk scan,
        the leaf's zero count, the partition update and the DEVIATION 1901
        stats update. The border the old update kernel searched for in
        `sorted_flags` IS the leaf's zero count (zeros sort first), so it is
        written directly. The parent's {offset, size} is snapshotted per SLOT
        first (`slot_off`, `slot_sz`), because K3 and K4 still need the
        parent's range after K2 has overwritten the left child's size.
    K3 `fused_place_scatter_kernel` the place pass, which SCATTERS the row
        index instead of writing a gather map: `tmp[off + dst] =
        row_index[off + i]` is `row_index[off + j] = tmp[off + gmap[off + j]]`
        with `gmap[off + dst] = i`, the same permutation.
    K4 `fused_copy_back_kernel`     tmp -> row_index over the split ranges.

Integer counts only: no float is re-associated anywhere in the chain, and
the stats update is DEVIATION 1901's body verbatim. So the arm is a pure
schedule change against the FAST path it replaces.
"""

from gbdt.gpu_data.gpu_structures import CFeature
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from gbdt.apple_fast_tree_experiments import AFT_N05
from std.gpu.intrinsics import ldg
from std.memory import bitcast
from max.gpu.primitives.block import broadcast as block_broadcast
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier

#: One chunk of rows per block, as the 3-launch partition (`PARTITION_BLOCK`).
comptime FUSED_CHAIN_BLOCK = 512
#: K4's copy block (`LEAF_COPY_BLOCK`).
comptime FUSED_COPY_BLOCK = 256


def fused_flags_count_kernel[GUARD: Bool = False](
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    leaf_ids: MutPointer[UInt32, MutAnyOrigin],
    split_features: MutPointer[CFeature, MutAnyOrigin],
    split_bins: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    chunk_zeros: MutPointer[UInt32, MutAnyOrigin],
    max_chunks_in: Int32,
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """K1: `split_and_make_sequence_kernel`'s side test plus
    `partition_count_chunks_kernel`. Grid (chunk stride, split leaf)."""
    comptime if GUARD:
        # DW_NO_LEVEL_SYNC: the grid covers every scored leaf; the split
        # count is on the device (`dw_select_splits_kernel`). Whole block.
        if Int(block_idx.y) >= Int(n_split_dev.unsafe_load(0)):
            return
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaf_ids.unsafe_load(leaf_slot))
    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))
    # field by field: a whole-struct CFeature load kills the Metal compiler
    # (`split_and_make_sequence_kernel`'s DEVIATION block)
    var f_offset = Int(split_features[unsafe_offset=leaf_slot].offset)
    var shift = split_features[unsafe_offset=leaf_slot].shift
    var one_hot = split_features[unsafe_offset=leaf_slot].one_hot_feature
    var value = UInt32(split_bins.unsafe_load(leaf_slot)) << shift
    var mask = split_features[unsafe_offset=leaf_slot].mask << shift
    var tid = Int(thread_idx.x)
    var chunk = Int(block_idx.x)
    while chunk * FUSED_CHAIN_BLOCK < size:
        var i = chunk * FUSED_CHAIN_BLOCK + tid
        var v = Int32(0)
        if i < size:
            var load_index = Int(ldg(row_index + (offset + i)))
            var feature_val = (
                ldg(compressed_index + (f_offset + load_index)) & mask
            )
            var goes_right = (
                feature_val == value
            ) if one_hot else (feature_val > value)
            flags.unsafe_store(
                offset + i, UInt8(1) if goes_right else UInt8(0)
            )
            if not goes_right:
                v = Int32(1)
        var inc = block_prefix_sum[
            block_size=FUSED_CHAIN_BLOCK, exclusive=False
        ](v)
        var total = block_broadcast[block_size=FUSED_CHAIN_BLOCK](
            inc, src_thread=FUSED_CHAIN_BLOCK - 1
        )
        if tid == 0:
            chunk_zeros.unsafe_store(
                leaf_slot * max_chunks + chunk, UInt32(Int(total))
            )
        barrier()
        chunk += Int(grid_dim.x)


def fused_scan_update_kernel[GUARD: Bool = False](
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    chunk_zeros: MutPointer[UInt32, MutAnyOrigin],
    chunk_zero_offsets: MutPointer[UInt32, MutAnyOrigin],
    leaf_total_zeros: MutPointer[UInt32, MutAnyOrigin],
    slot_off: MutPointer[UInt32, MutAnyOrigin],
    slot_sz: MutPointer[UInt32, MutAnyOrigin],
    win_cells: MutPointer[UInt32, MutAnyOrigin],
    split_features: MutPointer[CFeature, MutAnyOrigin],
    bin_feature_count_in: Int32,
    stat_count_in: Int32,
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    max_chunks_in: Int32,
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """K2: `partition_scan_chunks_kernel` + `update_partitions_after_split_kernel`
    + `update_partition_stats_from_split_kernel`. Grid (1, split leaf); the
    block strides over the leaf's CHUNKS, as the scan it replaces."""
    comptime if GUARD:
        # DW_NO_LEVEL_SYNC: the grid covers every scored leaf; the split
        # count is on the device (`dw_select_splits_kernel`). Whole block.
        if Int(block_idx.y) >= Int(n_split_dev.unsafe_load(0)):
            return
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var left_leaf = Int(left_leaves.unsafe_load(leaf_slot))
    var right_leaf = Int(right_leaves.unsafe_load(leaf_slot))
    var offset = Int(part_offset.unsafe_load(left_leaf))
    var size = Int(part_size.unsafe_load(left_leaf))
    var n_chunks = (size + FUSED_CHAIN_BLOCK - 1) // FUSED_CHAIN_BLOCK
    var tid = Int(thread_idx.x)

    # DEVIATION 1901's stats update, verbatim (one thread per stat; the
    # parent's entry is read before either child's is written, in-thread)
    var bin_feature_count = Int(bin_feature_count_in)
    var stat_count = Int(stat_count_in)
    var cell = Int(win_cells.unsafe_load(leaf_slot))
    var one_hot = split_features[unsafe_offset=leaf_slot].one_hot_feature
    var stat_id = tid
    while stat_id < stat_count:
        var parent = part_stats.unsafe_load(left_leaf * stat_count + stat_id)
        var cell_sum = histograms.unsafe_load(
            left_leaf * bin_feature_count * stat_count
            + stat_id * bin_feature_count
            + cell
        )
        var derived = parent - cell_sum
        if stat_id == 0:
            cell_sum = max(cell_sum, Float32(0.0))
            derived = max(derived, Float32(0.0))
        var left_sum = derived if one_hot else cell_sum
        var right_sum = cell_sum if one_hot else derived
        part_stats.unsafe_store(left_leaf * stat_count + stat_id, left_sum)
        part_stats.unsafe_store(right_leaf * stat_count + stat_id, right_sum)
        stat_id += Int(block_dim.x)

    var carry = 0
    var c = 0
    while c < n_chunks:
        var idx = c + tid
        var v = Int32(0)
        if idx < n_chunks:
            v = Int32(
                Int(chunk_zeros.unsafe_load(leaf_slot * max_chunks + idx))
            )
        var inc = block_prefix_sum[
            block_size=FUSED_CHAIN_BLOCK, exclusive=False
        ](v)
        var group_total = block_broadcast[block_size=FUSED_CHAIN_BLOCK](
            inc, src_thread=FUSED_CHAIN_BLOCK - 1
        )
        if idx < n_chunks:
            chunk_zero_offsets.unsafe_store(
                leaf_slot * max_chunks + idx,
                UInt32(carry + Int(inc) - Int(v)),
            )
        carry += Int(group_total)
        barrier()
        c += FUSED_CHAIN_BLOCK
    # every thread has read the parent's range above; only now is it moved
    barrier()
    if tid == 0:
        leaf_total_zeros.unsafe_store(leaf_slot, UInt32(carry))
        slot_off.unsafe_store(leaf_slot, UInt32(offset))
        slot_sz.unsafe_store(leaf_slot, UInt32(size))
        # the border `update_partitions_after_split_kernel` searches for:
        # the first "goes right" row of the sorted range, i.e. the zero count
        part_size.unsafe_store(left_leaf, UInt32(carry))
        part_offset.unsafe_store(right_leaf, UInt32(offset + carry))
        part_size.unsafe_store(right_leaf, UInt32(size - carry))


def fused_place_scatter_kernel[GUARD: Bool = False](
    slot_off: MutPointer[UInt32, MutAnyOrigin],
    slot_sz: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    chunk_zero_offsets: MutPointer[UInt32, MutAnyOrigin],
    leaf_total_zeros: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    temp_index: MutPointer[UInt32, MutAnyOrigin],
    max_chunks_in: Int32,
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """K3: `partition_place_kernel`'s placement, applied to the row index
    as a scatter into `temp_index`. Grid (chunk stride, split leaf)."""
    comptime if GUARD:
        # DW_NO_LEVEL_SYNC: the grid covers every scored leaf; the split
        # count is on the device (`dw_select_splits_kernel`). Whole block.
        if Int(block_idx.y) >= Int(n_split_dev.unsafe_load(0)):
            return
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var offset = Int(slot_off.unsafe_load(leaf_slot))
    var size = Int(slot_sz.unsafe_load(leaf_slot))
    var tid = Int(thread_idx.x)
    var n_zeros = Int(leaf_total_zeros.unsafe_load(leaf_slot))
    var chunk = Int(block_idx.x)
    while chunk * FUSED_CHAIN_BLOCK < size:
        var i = chunk * FUSED_CHAIN_BLOCK + tid
        var zeros_before = Int(
            chunk_zero_offsets.unsafe_load(leaf_slot * max_chunks + chunk)
        )
        var elems_before = chunk * FUSED_CHAIN_BLOCK
        if elems_before > size:
            elems_before = size
        var ones_before = elems_before - zeros_before
        var in_range = i < size
        var is_zero = Int32(0)
        if in_range:
            if flags.unsafe_load(offset + i) == UInt8(0):
                is_zero = Int32(1)
        var inc_zero = block_prefix_sum[
            block_size=FUSED_CHAIN_BLOCK, exclusive=False
        ](is_zero)
        var rank_zero = Int(inc_zero) - Int(is_zero)
        var rank_one = tid - rank_zero
        if in_range:
            var dst = 0
            if is_zero == Int32(1):
                dst = zeros_before + rank_zero
            else:
                dst = n_zeros + ones_before + rank_one
            temp_index.unsafe_store(
                offset + dst, ldg(row_index + (offset + i))
            )
        barrier()
        chunk += Int(grid_dim.x)


def fused_copy_back_kernel[GUARD: Bool = False](
    slot_off: MutPointer[UInt32, MutAnyOrigin],
    slot_sz: MutPointer[UInt32, MutAnyOrigin],
    temp_index: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    n_split_dev: MutPointer[UInt32, MutAnyOrigin],
):
    """K4: the partitioned split ranges back into the row index. Grid
    (stride, split leaf), as `copy_index_in_leaves_kernel`."""
    comptime if GUARD:
        # DW_NO_LEVEL_SYNC: the grid covers every scored leaf; the split
        # count is on the device (`dw_select_splits_kernel`). Whole block.
        if Int(block_idx.y) >= Int(n_split_dev.unsafe_load(0)):
            return
    var leaf_slot = Int(block_idx.y)
    var offset = Int(slot_off.unsafe_load(leaf_slot))
    var size = Int(slot_sz.unsafe_load(leaf_slot))
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        row_index.unsafe_store(offset + i, ldg(temp_index + (offset + i)))
        i += stride


# ---- DW_NO_LEVEL_SYNC (FAST, Apple; default) -----------------------------
#: `dw_select_splits_kernel`'s block (one thread per scored leaf).
# N05: four simdgroups select/pack leaf split descriptors per block,
# amortizing threadgroup scheduling across more independent leaves. The
# strict gain test and prefix rank defining creation order stay unchanged.
# No performance/quality evidence; source only, default OFF.
comptime DW_SELECT_BLOCK = 128 if AFT_N05 else 64
#: Words per `CFeature` record in the split payload (`CFEATURE_BYTES // 4`).
comptime DW_FEAT_WORDS = 6
#: The fold's record layout (`split_resolve.WINNER_RECORD_WORDS` and
#: `WINNER_STATUS_DEFINED`), restated so this file imports no kernel module.
comptime DW_WINNER_WORDS = 5
comptime DW_WINNER_DEFINED = UInt32(1)


def dw_select_splits_kernel(
    winner: MutPointer[UInt32, MutAnyOrigin],
    visit: MutPointer[UInt32, MutAnyOrigin],
    n_visit_in: Int32,
    leaves_count_in: Int32,
    feat_table: MutPointer[UInt32, MutAnyOrigin],
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    split_features: MutPointer[UInt32, MutAnyOrigin],
    split_bins: MutPointer[UInt32, MutAnyOrigin],
    win_cells: MutPointer[UInt32, MutAnyOrigin],
    n_split_out: MutPointer[UInt32, MutAnyOrigin],
):
    """The Depthwise `SelectLeavesToSplit` and the split payload, on the
    device. One thread per scored leaf `v` (the visit list is ascending, so
    visit order is leaf-id order, the host selection's order): selected iff
    the fold's record is DEFINED with `Gain < 0` (their `Score < 0`, and the
    two are one number on this kernel). Its slot is the count of selected
    records before it, so slot `r` gets left = the leaf, right =
    `leaves_count + r`, the bin, the winning cell, and the feature's
    `CFeature` words (`feat_table`, offset already in elements). The last
    thread writes the split count. Integer moves only; the host rebuilds
    the same lists after the level's one readback and checks the count."""
    var n_visit = Int(n_visit_in)
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if v >= n_visit:
        return
    var rank = 0
    for u in range(v):
        var ub = u * DW_WINNER_WORDS
        if winner.unsafe_load(ub + 3) == DW_WINNER_DEFINED:
            if bitcast[DType.float32](winner.unsafe_load(ub + 2)) < Float32(
                0.0
            ):
                rank += 1
    var base = v * DW_WINNER_WORDS
    var selected = False
    if winner.unsafe_load(base + 3) == DW_WINNER_DEFINED:
        selected = bitcast[DType.float32](
            winner.unsafe_load(base + 2)
        ) < Float32(0.0)
    if selected:
        left_leaves.unsafe_store(rank, visit.unsafe_load(v))
        right_leaves.unsafe_store(
            rank, UInt32(Int(leaves_count_in) + rank)
        )
        split_bins.unsafe_store(rank, winner.unsafe_load(base + 1))
        win_cells.unsafe_store(rank, winner.unsafe_load(base + 4))
        var feat = Int(winner.unsafe_load(base))
        for w in range(DW_FEAT_WORDS):
            split_features.unsafe_store(
                rank * DW_FEAT_WORDS + w,
                feat_table.unsafe_load(feat * DW_FEAT_WORDS + w),
            )
    if v == n_visit - 1:
        n_split_out.unsafe_store(0, UInt32(rank + (1 if selected else 0)))
