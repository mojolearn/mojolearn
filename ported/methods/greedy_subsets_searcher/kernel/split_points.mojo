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
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import broadcast as block_broadcast
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier


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


# =========================================================================
# DEVIATION BLOCK: everything below replaces ONE CALL in their file.
#
# `split_points.cu:658-689` calls `cub::DeviceRadixSort::SortPairs` once per
# leaf, from a host loop, to sort each leaf's range by the split flag. There
# is no CUB in Mojo, so there is no line-for-line port of that call and a
# reviewer diffing this file against theirs will find no counterpart for what
# follows. It lives HERE rather than in `mojo_only/` because it replaces a
# step of THIS module, and moving it elsewhere would leave the reorder
# incomplete in the file that owns it.
#
# It is also the one place in `ported/` allowed to be better than CatBoost,
# because there is no CatBoost code to be faithful to. Their own comments ask
# for exactly this:
#
#     //TODO(noxoomo): cub sucks for this, write proper segmented version
#     //TODO(noxoomo): for oblivious trees we have overhead for launching
#       kernel per leaf
#
# The key is ONE BIT, the split flag, so a full radix sort was never needed;
# a segmented stable partition is sufficient and is what that TODO describes.
#
# WHY THREE PHASES, AND WHAT THE FIRST VERSION GOT WRONG
# ------------------------------------------------------
# The first version ran ONE BLOCK PER LEAF, walking the leaf in chunks and
# carrying the running zero and one counts between them to keep the order
# stable. Correct, and measured 2.985 ms against the histogram's 1.327 ms at
# 500,000 rows, because at depth 0 there is ONE leaf: 256 threads, 1,954
# sequential chunks, the rest of the machine idle.
#
# That was the mirror image of CatBoost's wart rather than an escape from it.
# Theirs is too many launches where leaves are many; mine was too few blocks
# where leaves are few and huge. Depth 8 is the worst case for one, depth 0
# for the other.
#
# So: block-per-CHUNK counts, a scan of the per-chunk totals, then placement.
# Every phase is grid-parallel over (chunk, leaf), so a single 500,000-row
# leaf fills the machine and a level of 64 small leaves still runs at once.
# Stability still comes from the carry; it is precomputed instead of walked.
# =========================================================================

comptime PARTITION_BLOCK = 256


def partition_count_chunks_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    chunk_zeros: MutPointer[UInt32, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """PHASE 1: how many zeros in each chunk of each leaf.

    Grid x is the CHUNK and grid y is the leaf, so the whole machine works on
    one big leaf and on many small ones alike.
    """
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaves.unsafe_load(leaf_slot))
    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))

    var chunk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var i = chunk * PARTITION_BLOCK + tid

    var v = Int32(0)
    if i < size:
        if flags.unsafe_load(offset + i) == UInt8(0):
            v = Int32(1)
    var inc = block_prefix_sum[block_size=PARTITION_BLOCK, exclusive=False](v)
    var total = block_broadcast[block_size=PARTITION_BLOCK](
        inc, src_thread = PARTITION_BLOCK - 1
    )
    if tid == 0:
        chunk_zeros.unsafe_store(
            leaf_slot * max_chunks + chunk, UInt32(Int(total))
        )


def partition_scan_chunks_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    chunk_zeros: MutPointer[UInt32, MutAnyOrigin],
    chunk_zero_offsets: MutPointer[UInt32, MutAnyOrigin],
    leaf_total_zeros: MutPointer[UInt32, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """PHASE 2: exclusive scan of a leaf's per-chunk zero counts.

    One block per leaf, but over CHUNKS rather than rows, so the sequential
    dimension is `size / 256` instead of `size`. At 500,000 rows that is
    1,954 values scanned by 256 threads rather than 500,000, and the phase
    that has to be sequential is now three orders of magnitude smaller than
    the phases that do not.
    """
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaves.unsafe_load(leaf_slot))
    var size = Int(part_size.unsafe_load(leaf_id))
    var n_chunks = (size + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    var tid = Int(thread_idx.x)

    var carry = 0
    var c = 0
    while c < n_chunks:
        var idx = c + tid
        var v = Int32(0)
        if idx < n_chunks:
            v = Int32(Int(chunk_zeros.unsafe_load(leaf_slot * max_chunks + idx)))
        var inc = block_prefix_sum[
            block_size=PARTITION_BLOCK, exclusive=False
        ](v)
        var group_total = block_broadcast[block_size=PARTITION_BLOCK](
            inc, src_thread = PARTITION_BLOCK - 1
        )
        if idx < n_chunks:
            # exclusive within the group, plus everything before the group
            chunk_zero_offsets.unsafe_store(
                leaf_slot * max_chunks + idx,
                UInt32(carry + Int(inc) - Int(v)),
            )
        carry += Int(group_total)
        barrier()
        c += PARTITION_BLOCK

    if tid == 0:
        leaf_total_zeros.unsafe_store(leaf_slot, UInt32(carry))


def partition_place_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    chunk_zero_offsets: MutPointer[UInt32, MutAnyOrigin],
    leaf_total_zeros: MutPointer[UInt32, MutAnyOrigin],
    gather_map: MutPointer[UInt32, MutAnyOrigin],
    sorted_flags: MutPointer[UInt8, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """PHASE 3: place every row, grid-parallel again.

    A chunk's zero rows start at the number of zeros in all earlier chunks
    (`chunk_zero_offsets`), and its one rows start after ALL the leaf's zeros
    plus the number of ones in all earlier chunks, which is
    `earlier_elements - earlier_zeros` and needs no second scan.
    """
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaves.unsafe_load(leaf_slot))
    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))

    var chunk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var i = chunk * PARTITION_BLOCK + tid

    var n_zeros = Int(leaf_total_zeros.unsafe_load(leaf_slot))
    var zeros_before = Int(
        chunk_zero_offsets.unsafe_load(leaf_slot * max_chunks + chunk)
    )
    var elems_before = chunk * PARTITION_BLOCK
    if elems_before > size:
        elems_before = size
    var ones_before = elems_before - zeros_before

    var in_range = i < size
    var is_zero = Int32(0)
    if in_range:
        if flags.unsafe_load(offset + i) == UInt8(0):
            is_zero = Int32(1)

    var inc_zero = block_prefix_sum[
        block_size=PARTITION_BLOCK, exclusive=False
    ](is_zero)
    var rank_zero = Int(inc_zero) - Int(is_zero)

    var is_one = Int32(0)
    if in_range and is_zero == Int32(0):
        is_one = Int32(1)
    var inc_one = block_prefix_sum[
        block_size=PARTITION_BLOCK, exclusive=False
    ](is_one)
    var rank_one = Int(inc_one) - Int(is_one)

    if in_range:
        var dst = 0
        if is_zero == Int32(1):
            dst = zeros_before + rank_zero
        else:
            dst = n_zeros + ones_before + rank_one
        gather_map.unsafe_store(offset + dst, UInt32(i))
        sorted_flags.unsafe_store(
            offset + dst, UInt8(0) if is_zero == Int32(1) else UInt8(1)
        )


def launch_stable_partition(
    ctx: DeviceContext,
    n_leaf_slots: Int,
    max_leaf_rows: Int,
    mut leaves: DeviceBuffer[DType.uint32],
    mut part_offset: DeviceBuffer[DType.uint32],
    mut part_size: DeviceBuffer[DType.uint32],
    mut flags: DeviceBuffer[DType.uint8],
    mut chunk_zeros: DeviceBuffer[DType.uint32],
    mut chunk_offsets: DeviceBuffer[DType.uint32],
    mut leaf_zeros: DeviceBuffer[DType.uint32],
    mut gather_map: DeviceBuffer[DType.uint32],
    mut sorted_flags: DeviceBuffer[DType.uint8],
) raises:
    """The three phases in order. Callers should not launch them by hand.

    `max_leaf_rows` sizes the grid and the scratch: it is the largest leaf in
    the level, so a level of small leaves does not pay for the depth-0 shape.
    """
    var max_chunks = (max_leaf_rows + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    if max_chunks < 1:
        max_chunks = 1

    ctx.enqueue_function[partition_count_chunks_kernel](
        leaves.unsafe_ptr(),
        part_offset.unsafe_ptr(),
        part_size.unsafe_ptr(),
        flags.unsafe_ptr(),
        chunk_zeros.unsafe_ptr(),
        Int32(max_chunks),
        grid_dim=(max_chunks, n_leaf_slots, 1),
        block_dim=(PARTITION_BLOCK, 1, 1),
    )
    ctx.enqueue_function[partition_scan_chunks_kernel](
        leaves.unsafe_ptr(),
        part_size.unsafe_ptr(),
        chunk_zeros.unsafe_ptr(),
        chunk_offsets.unsafe_ptr(),
        leaf_zeros.unsafe_ptr(),
        Int32(max_chunks),
        grid_dim=(1, n_leaf_slots, 1),
        block_dim=(PARTITION_BLOCK, 1, 1),
    )
    ctx.enqueue_function[partition_place_kernel](
        leaves.unsafe_ptr(),
        part_offset.unsafe_ptr(),
        part_size.unsafe_ptr(),
        flags.unsafe_ptr(),
        chunk_offsets.unsafe_ptr(),
        leaf_zeros.unsafe_ptr(),
        gather_map.unsafe_ptr(),
        sorted_flags.unsafe_ptr(),
        Int32(max_chunks),
        grid_dim=(max_chunks, n_leaf_slots, 1),
        block_dim=(PARTITION_BLOCK, 1, 1),
    )
