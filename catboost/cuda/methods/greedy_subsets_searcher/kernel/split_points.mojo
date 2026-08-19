"""Reorder each leaf's index range so its two children are contiguous.

PORT OF `catboost/cuda/methods/greedy_subsets_searcher/kernel/split_points.cu`
at CatBoost `54a8143a`. Transliterated. Do not improve.

This is what keeps `TDataPartition{Offset, Size}` true after a split. A leaf
IS a contiguous range, so splitting one means physically partitioning its
range into "goes left" followed by "goes right", stably, and then writing two
new `{Offset, Size}` records.

**What moves and what does not.** The `UInt32` index array and the stat
columns are permuted. The COMPRESSED INDEX IS NOT: it is read indirectly as
`compressedIndex[feature.Offset + loadIndices[i]]` and never moves for the
life of the fit. That is the distinction that makes this affordable and it is
worth stating loudly, because mojotrees measured physical reordering of the
BIN MATRIX at 1.535x slower on GPU and 2.6x on CPU. CatBoost reorders a
4-byte index per row, not a row of the matrix.

**Their own comments call the sort a wart**, and they are quoted here so a
future reader does not mistake it for a design to preserve:

    //TODO(noxoomo): cub sucks for this, write proper segmented version
        (`split_points.cu:657`)
    //TODO(noxoomo): for oblivious trees we have overhead for launching
      kernel per leaf
        (`split_points.cpp:53`)

So the per-leaf `cub::DeviceRadixSort::SortPairs` in a host loop, 255 of them
for a depth-8 tree, is the one part of this design its authors say is wrong.
DEVIATION (PORTING.md 4): there is no CUB in Mojo, and the sort is being used
only as a stable 1-bit partition, so that is what is written.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx


#: `split_points.cu:558`.
comptime SPLIT_BLOCK_SIZE = 512

#: `const int N = 4` at the same site. SCHEDULING row.
comptime SPLIT_UNROLL = 4


def split_and_make_sequence_kernel(
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    load_indices: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    leaf_ids: MutPointer[UInt32, MutAnyOrigin],
    split_feature_offset: MutPointer[UInt32, MutAnyOrigin],
    split_feature_shift: MutPointer[UInt32, MutAnyOrigin],
    split_feature_mask: MutPointer[UInt32, MutAnyOrigin],
    split_one_hot: MutPointer[UInt8, MutAnyOrigin],
    split_bins: MutPointer[UInt32, MutAnyOrigin],
    split_flags: MutPointer[UInt8, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
):
    """`SplitAndMakeSequenceInLeavesImpl`, copied.

    Writes, for every row of every splitting leaf, the side it lands on and
    its position within the leaf. Grid y is the LEAF, so all leaves of the
    level are one launch. The sequence written into `indices` is what the
    stable partition afterwards permutes.

    The `oneHot` branch is theirs: a one-hot feature tests equality against
    the split value while an ordered feature tests `>`. Both are kept even
    though this tree does not build categorical features yet, because
    dropping a branch is a change and the rule here is to copy.
    """
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaf_ids.unsafe_load(leaf_slot))

    var size = Int(part_size.unsafe_load(leaf_id))
    var offset = Int(part_offset.unsafe_load(leaf_id))

    var i = Int(block_idx.x) * SPLIT_BLOCK_SIZE * SPLIT_UNROLL + Int(
        thread_idx.x
    )
    if i >= size:
        return

    var f_offset = Int(split_feature_offset.unsafe_load(leaf_slot))
    var shift = UInt32(split_feature_shift.unsafe_load(leaf_slot))
    var bin_idx = UInt32(split_bins.unsafe_load(leaf_slot))
    var one_hot = split_one_hot.unsafe_load(leaf_slot) != 0

    var value = bin_idx << shift
    var mask = UInt32(split_feature_mask.unsafe_load(leaf_slot)) << shift

    var stride = SPLIT_UNROLL * SPLIT_BLOCK_SIZE * Int(grid_dim.x)

    while i < size:
        @parameter
        for k in range(SPLIT_UNROLL):
            var at = i + k * SPLIT_BLOCK_SIZE
            if at < size:
                # `loadIndices ? loadIndices[at] : at`. The port always
                # carries the array; the root seeds it with the identity.
                var load_index = Int(
                    load_indices.unsafe_load(offset + at)
                )
                var feature_val = (
                    compressed_index.unsafe_load(f_offset + load_index) & mask
                )
                # The sequence, written before the flag, exactly as they do.
                indices.unsafe_store(offset + at, UInt32(at))
                var goes_right = (
                    feature_val == value
                ) if one_hot else (feature_val > value)
                split_flags.unsafe_store(
                    offset + at, UInt8(1) if goes_right else UInt8(0)
                )
        i += stride
