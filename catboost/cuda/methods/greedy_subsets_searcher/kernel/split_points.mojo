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


def update_partitions_after_split_kernel(
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    leaf_count: Int32,
    sorted_flags: MutPointer[UInt8, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    host_offset: MutPointer[UInt32, MutAnyOrigin],
    host_size: MutPointer[UInt32, MutAnyOrigin],
):
    """`UpdatePartitionsAfterSplitImpl`, copied.

    After the stable partition has sorted a leaf's range into
    "goes left" then "goes right", this finds the BORDER between the two and
    turns one `{Offset, Size}` into two. It is how a leaf becomes two leaves
    while membership stays positional.

    Their border search, copied exactly:

        int flag0 = i < partSize ? sortedFlags[i] : 1;
        int flag1 = i ? sortedFlags[i - 1] : 0;
        if (flag0 != flag1) { ... break; }

    The sentinels are the whole trick and they are easy to lose. At `i == 0`
    the previous flag reads 0, and at `i == partSize` the current flag reads
    1, so a leaf where EVERY row goes one way still finds a border, at 0 or
    at `partSize`, and produces an empty child rather than no answer at all.
    Every thread scans a strided slice and the first to find the border
    writes both partitions; only one can, because a sorted flag array has
    exactly one transition.

    **`partsCpu` is part of the port and is NOT DONE YET.** They write the
    new partitions to device memory AND to pinned host memory in the same
    store (`split_points.cu:372`, `:379`), so the host learns every leaf's
    size with no device-to-host copy at all. That is what lets the next
    level's `build_necessary_histograms` pick the smaller sibling on the host
    without a readback in the critical path.

    The kernel here takes `host_offset` / `host_size` and writes them, so the
    device half is ported. What is missing is the driver allocating those as
    PINNED host memory rather than ordinary device buffers, which is where
    the trick actually pays. Listed in UNWIRED.md.
    """
    var leaf_slot = Int(block_idx.y)
    var left_leaf = Int(left_leaves.unsafe_load(leaf_slot))
    var right_leaf = Int(right_leaves.unsafe_load(leaf_slot))

    var offset = Int(part_offset.unsafe_load(left_leaf))
    var part_sz = Int(part_size.unsafe_load(left_leaf))

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)

    while i <= part_sz:
        var flag0 = 1
        if i < part_sz:
            flag0 = Int(sorted_flags.unsafe_load(offset + i))
        var flag1 = 0
        if i != 0:
            flag1 = Int(sorted_flags.unsafe_load(offset + i - 1))

        if flag0 != flag1:
            # The border. Left keeps the offset and shrinks to `i`; right
            # starts where left ends and takes the remainder.
            part_size.unsafe_store(left_leaf, UInt32(i))
            host_size.unsafe_store(left_leaf, UInt32(i))

            part_offset.unsafe_store(right_leaf, UInt32(offset + i))
            part_size.unsafe_store(right_leaf, UInt32(part_sz - i))
            host_offset.unsafe_store(right_leaf, UInt32(offset + i))
            host_size.unsafe_store(right_leaf, UInt32(part_sz - i))
            break
        i += stride


def gather_in_leaves_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    gather_map: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    num_stats_in: Int32,
    line_size_in: Int32,
):
    """`GatherInLeavesImpl`, copied.

    Applies a permutation WITHIN each leaf's range, to every stat column at
    once. `map[i]` is the source position for destination position `i`,
    both relative to the leaf's offset, so the permutation never crosses a
    leaf boundary and a leaf's rows stay contiguous.

    This is the second half of the reorder. `split_and_make_sequence_kernel`
    writes the flags and the identity sequence, a stable partition turns that
    sequence into a permutation with the "goes left" rows first, and this
    applies it to the payload.

    **What is permuted is the STAT COLUMNS and the index, never the binned
    matrix.** `src` here is gradients and weights, `lineSize` strides between
    stat planes. The compressed index is read indirectly through the
    permuted row ids and does not move for the life of the fit. That
    distinction is the reason CatBoost can afford a per-level reorder at all:
    mojotrees measured reordering the BIN MATRIX at 1.535x slower on GPU and
    2.6x on CPU, which is a different and far more expensive operation.

    Grid y is the LEAF, so one launch reorders every splitting leaf.
    """
    var num_stats = Int(num_stats_in)
    var line_size = Int(line_size_in)
    var leaf_id = Int(leaves.unsafe_load(Int(block_idx.y)))

    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)

    while i < size:
        var load_idx = Int(gather_map.unsafe_load(offset + i))
        for k in range(num_stats):
            dst.unsafe_store(
                offset + i + k * line_size,
                src.unsafe_load(offset + load_idx + k * line_size),
            )
        i += stride


def gather_index_in_leaves_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    src: MutPointer[UInt32, MutAnyOrigin],
    gather_map: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[UInt32, MutAnyOrigin],
):
    """The same permutation applied to the `UInt32` row-index array.

    CatBoost gets this from the same template instantiated at `ui32`
    (`GatherInLeaves<ui32>`); Mojo has no template over the element type
    here, so it is a second function. DEVIATION of expression only: identical
    arithmetic, one stat, no line stride.

    It matters as much as the stat gather. The index array is what
    `split_and_make_sequence_kernel` dereferences to reach the compressed
    index next level, so if the payload is permuted and the index is not, the
    next level reads the right rows' bins against the wrong rows' gradients.
    """
    var leaf_id = Int(leaves.unsafe_load(Int(block_idx.y)))
    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)

    while i < size:
        var load_idx = Int(gather_map.unsafe_load(offset + i))
        dst.unsafe_store(offset + i, src.unsafe_load(offset + load_idx))
        i += stride
