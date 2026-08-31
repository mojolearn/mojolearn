# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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

from gbdt.gpu_data.gpu_structures import CFeature
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.device_attribute import DeviceAttribute
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
    split_features: MutPointer[CFeature, MutAnyOrigin],
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
    # `leafIds[blockIdx.y]` (`split_points.cu:480`) is a PLAIN load in their
    # file and stays one here.
    var leaf_id = Int(leaf_ids.unsafe_load(leaf_slot))

    # `TDataPartition part = Ldg(parts + leafId)` (`split_points.cu:482`).
    # `Ldg` is `cub::ThreadLoad<cub::LOAD_LDG>` (`kernel_helpers.cuh:180`),
    # the read-only non-coherent load; `std.gpu.intrinsics.ldg` is its Mojo
    # spelling. Their ONE struct load is two scalar loads here, for the
    # Metal reason recorded below.
    var size = Int(ldg(part_size + leaf_id))
    var offset = Int(ldg(part_offset + leaf_id))

    var i = Int(block_idx.x) * SPLIT_BLOCK_SIZE * SPLIT_UNROLL + Int(
        thread_idx.x
    )
    if i >= size:
        return

    # `const TCFeature feature = Ldg(splitFeatures + blockIdx.y);`
    #
    # Their signature is ONE `const TCFeature*`. The port used to carry the
    # four fields as four parallel arrays, which is four pointers derived
    # from one allocation, and `enqueue_function` refuses those as aliasing
    # mutable arguments. Matching their type fixes that structurally.
    #
    # ===================== DEVIATION BLOCK =====================
    # Their `Ldg` pulls the WHOLE struct into a register in one load. We read
    # field by field instead, because binding the struct to a local:
    #
    #     var feature = split_features[unsafe_offset=leaf_slot]
    #
    # makes the Metal backend die with
    #
    #     error: Metal Compiler failed to compile metallib.
    #            Please submit a bug report.
    #
    # Reproduced 2026-08-19 in a 25-line probe: per-field access through the
    # pointer compiles and runs, one whole-struct load does not, and the
    # field type is irrelevant (it fails the same with the Bool untouched).
    # Semantically identical, so nothing downstream changes.
    # ===========================================================
    var f_offset = Int(split_features[unsafe_offset=leaf_slot].offset)
    var shift = split_features[unsafe_offset=leaf_slot].shift
    var bin_idx = UInt32(split_bins.unsafe_load(leaf_slot))
    var one_hot = split_features[unsafe_offset=leaf_slot].one_hot_feature

    var value = bin_idx << shift
    var mask = split_features[unsafe_offset=leaf_slot].mask << shift

    var stride = SPLIT_UNROLL * SPLIT_BLOCK_SIZE * Int(grid_dim.x)

    while i < size:
        @parameter
        for k in range(SPLIT_UNROLL):
            var at = i + k * SPLIT_BLOCK_SIZE
            if at < size:
                # `loadIndex[k] = loadIndices ? __ldg(loadIndices + i +
                #  k * BlockSize) : i + k * BlockSize` (`split_points.cu:511`).
                # The port always carries the array; the root seeds it with
                # the identity.
                var load_index = Int(ldg(load_indices + (offset + at)))
                # `featureVal[k] = __ldg(compressedIndex + loadIndex[k]) &
                #  mask` (`split_points.cu:518`).
                var feature_val = (
                    ldg(compressed_index + (f_offset + load_index)) & mask
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
        # `Ldg(sortedFlags + i)` and `Ldg(sortedFlags + i - 1)`
        # (`split_points.cu:364-365`). The partition reads above are PLAIN in
        # their kernel (`:355-362`) and stay plain here.
        var flag0 = 1
        if i < part_sz:
            flag0 = Int(ldg(sorted_flags + (offset + i)))
        var flag1 = 0
        if i != 0:
            flag1 = Int(ldg(sorted_flags + (offset + i - 1)))

        if flag0 != flag1:
            # The border. Left keeps the offset and shrinks to `i`; right
            # starts where left ends and takes the remainder.
            #
            # Theirs stores the WHOLE struct to BOTH mirrors
            # (`split_points.cu:369-372`):
            #
            #     TDataPartition leftPart = parts[leftLeaf];
            #     leftPart.Size = i;
            #     parts[leftLeaf] = leftPart;
            #     partsCpu[leftLeaf] = leftPart;
            #
            # so `Offset` reaches the host mirror even though the split did
            # not change it. Ours is four arrays, and `host_offset[left]` was
            # simply never written: the host mirror carried whatever the
            # allocation happened to hold for the left child of every split at
            # every level. `part_offset[left]` needs no store because the
            # value read back is its own.
            part_size.unsafe_store(left_leaf, UInt32(i))
            host_offset.unsafe_store(left_leaf, UInt32(offset))
            host_size.unsafe_store(left_leaf, UInt32(i))

            part_offset.unsafe_store(right_leaf, UInt32(offset + i))
            part_size.unsafe_store(right_leaf, UInt32(part_sz - i))
            host_offset.unsafe_store(right_leaf, UInt32(offset + i))
            host_size.unsafe_store(right_leaf, UInt32(part_sz - i))
            break
        i += stride


def update_partitions_and_plan_kernel(
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    leaf_count: Int32,
    sorted_flags: MutPointer[UInt8, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    host_offset: MutPointer[UInt32, MutAnyOrigin],
    host_size: MutPointer[UInt32, MutAnyOrigin],
    ids_compute: MutPointer[UInt32, MutAnyOrigin],
    sub_from: MutPointer[UInt32, MutAnyOrigin],
    sub_what: MutPointer[UInt32, MutAnyOrigin],
):
    """`update_partitions_after_split_kernel` with `plan_level_kernel`'s
    choice folded into the border store. USED ONLY BY THE BLIND TREE LOOP.

    ================= DEVIATION 210 (scheduling only) =================
    Their next level opens with `BuildNecessaryHistograms`' smaller-child
    choice (`split_properties_helper.cpp:1318`), which DEVIATION 94 moved
    on-device as `plan_level_kernel`: one launch per level reading the
    `p_sz` this kernel just wrote. That launch is pure LAUNCH TAX on this
    box (~42 us host floor, PREP_BILL 2026-08-21 step 29), because the
    border-finding thread below already holds BOTH sizes it would read:
    the left child's size is `i` and the right child's is `part_sz - i`,
    and next level's pair `(slot j, slot half + j)` IS this launch's
    `(left_leaf, right_leaf)` at `leaf_slot == j` -- `MakeSplit` gives the
    left child the parent's dense slot and `resolve_and_pack_kernel`
    writes `ids_c[j] = n_live + j`, with `half` next level equal to
    `n_live` now. So the plan store moves into the border branch: same
    comparison, same operands, same tie rule (an exact tie computes the
    RIGHT child -- PORTING.md 136), one fewer launch per planned level.
    NO arithmetic moves: the choice is an integer compare on the same two
    numbers `plan_level_kernel` loads, and every partition store is
    byte-identical to the unfused kernel's.

    `plan_level_kernel` stays: `mojo_only/sibling_tiebreak_check.mojo`
    exercises its tie rule, and `check-plan-fusion` gates THIS kernel's
    plan words against it on the same planted partitions.

    The final level's plan words are written and never read (the loop
    ends); the buffers are `max_leaves` wide and the last level writes
    `max_leaves / 2` slots, so the dead store is in bounds. When
    `use_subtraction` is off the caller never reads `ids_compute` either
    and the store is equally dead, exactly as the unfused planner's
    output was untrusted on unplanned levels.
    ===================================================================
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

            # the plan, folded (DEVIATION 210): `plan_level_kernel`'s
            # body on the sizes this thread just stored. `left_sz <
            # right_sz` is its strict `<` on the LEFT; the tie computes
            # the right child, so `small` starts there.
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


#: `const ui32 blockSize = 256` for the in-leaf copy and the in-leaf gather
#: (`split_points.cu:217`, `:239`).
comptime LEAF_COPY_BLOCK = 256


def split_points_grid_x(n_leaf_slots: Int, sm_count: Int) -> Int:
    """`numBlocks.x` for the in-leaf copy and gather, copied.

        numBlocks.x = (leavesCount > 4 ? 2 : 4) * TArchProps::SMCount();
            (`split_points.cu:220` for `CopyInLeaves`, `:242` for
             `GatherInLeaves`; the same expression again at `:397` for
             `UpdatePartitionsAfterSplit` and `:563` for
             `SplitAndMakeSequenceInLeaves`)

    **The grid comes from the DEVICE, not from the data.** These kernels
    stride (`i += gridDim.x * blockDim.x`), so grid x does not have to cover
    the rows and covering them is not what it is for: it is an occupancy
    target. Four blocks per SM while the leaf count is small enough that
    grid y cannot fill the machine on its own, two once grid y is 5 or more
    and supplies the rest.

    ================= WHAT THIS REPLACES =================
    `min(256, ceil(n_rows / 256))`, computed on the host in
    `greedy_search_helper.mojo` and passed in as `wide`. That number is
    DEVICE-BLIND: it is the same on a 10-core M4 and on a 132-SM H100, so it
    is wrong on both sides of whichever machine it was tuned on, and it also
    ties an occupancy knob to the row count, which is the one thing the
    stride already handles.

    `replication_for` in `greedy_search_helper.mojo` ports the histogram
    kernels' shape of the same idea (`hist_binary.cu:95`), which is
    `blocksPerSm * SMCount()` divided out over the other grid axes. This is
    their OTHER shape, and it is written the way their file writes it rather
    than reduced to one rule, because they are two different expressions in
    two different files.
    ======================================================

    `TArchProps::SMCount()` is
    `ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)`, the same
    attribute `replication_for`'s caller already reads.
    """
    var per_sm = 2 if n_leaf_slots > 4 else 4
    var grid_x = per_sm * sm_count
    if grid_x < 1:
        grid_x = 1
    return grid_x

#: `const ui32 statsPerKernel = 8` (`split_properties_helper.cpp:922`,
#: `:1010`), the number of stat COLUMNS the copy/gather pair handles per
#: launch. The chunk is what bounds the temp buffer to
#: `min(8, statCount) * lineSize` floats (`split_points.cpp:10`) instead of a
#: second full stats plane.
comptime STATS_PER_KERNEL = 8


def copy_in_leaves_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    stat_base_in: Int32,
    num_stats_in: Int32,
    line_size_in: Int32,
):
    """`CopyInLeavesImpl` (`split_points.cu:21`), copied.

    First half of the write-back pair. Copies the stat columns of every
    SPLITTING leaf's range out to scratch; `gather_in_leaves_kernel` then
    permutes them back into the same cells. Rows that are not inside a
    splitting leaf are never read and never written, which is the whole point
    of the pair: traffic per level is 2x the sum of the splitting leaves'
    sizes, not 2x `n_rows`, and the scratch only has to hold `numStats`
    columns of one chunk rather than a full second stats plane.

    ===================== DEVIATION BLOCK =====================
    Their call site pre-offsets the pointer, `Statistics.GetColumn(firstStat)`
    (`split_points.cpp:74`), and passes the scratch base unoffset, so the
    kernel itself knows nothing about the chunk. A Mojo `DeviceBuffer` is
    handed to `enqueue_function` whole, so the column base travels as an
    argument instead: `stat_base` indexes the STATS side, and the scratch side
    always starts at column 0. Same addresses, same traffic.
    ===========================================================

    Note both sides stride by `lineSize`, theirs too. Their SINGLE-LEAF
    `CopyLeafImpl` (`split_points.cu:269`) writes `dst + i + k * size`, a
    COMPACTED scratch stride, because it scratches one leaf. That variant is
    not ported (there is no single-leaf split path here), and its stride must
    not be mixed into this one.
    """
    var stat_base = Int(stat_base_in)
    var num_stats = Int(num_stats_in)
    var line_size = Int(line_size_in)
    var leaf_id = Int(leaves.unsafe_load(Int(block_idx.y)))

    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)

    while i < size:
        for k in range(num_stats):
            # `WriteThrough(dst + i + k * lineSize,
            #  __ldg(src + i + k * lineSize))` (`split_points.cu:42`). The
            # LOAD half ports; the store half has no Mojo spelling, which is
            # the deviation block on `gather_inplace_kernel`.
            dst.unsafe_store(
                offset + i + k * line_size,
                ldg(src + (offset + i + (stat_base + k) * line_size)),
            )
        i += stride


def copy_index_in_leaves_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    src: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[UInt32, MutAnyOrigin],
):
    """`CopyInLeaves<ui32>` (`split_points.cpp:91`), one column.

    They reach it by instantiating the same template at `ui32` with
    `numStats = 1`, at which point `lineSize` multiplies nothing; Mojo has no
    template over the element type here, so it is a second function with the
    dead stride argument dropped. DEVIATION of expression only.
    """
    var leaf_id = Int(leaves.unsafe_load(Int(block_idx.y)))
    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)

    while i < size:
        # `__ldg(src + i + k * lineSize)` at one column (`split_points.cu:42`).
        dst.unsafe_store(offset + i, ldg(src + (offset + i)))
        i += stride


def gather_in_leaves_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    gather_map: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    stat_base_in: Int32,
    num_stats_in: Int32,
    line_size_in: Int32,
):
    """`GatherInLeavesImpl` (`split_points.cu:178`), copied.

    Applies a permutation WITHIN each leaf's range, to every stat column of
    the chunk at once. `map[i]` is the source position for destination
    position `i`, both relative to the leaf's offset, so the permutation never
    crosses a leaf boundary and a leaf's rows stay contiguous.

    Second half of the write-back pair: `src` is the scratch that
    `copy_in_leaves_kernel` just filled and `dst` is the stats plane itself,
    so the permuted rows land back IN PLACE and no row outside a splitting
    leaf is touched. Getting this backwards -- gathering into a second full
    plane and copying the whole plane back -- reads and writes every row of
    every level and returns UNINITIALISED scratch for any row whose leaf did
    not split.

    `stat_base` offsets the STATS side only; see the deviation block on
    `copy_in_leaves_kernel`. Their call site does it with
    `Statistics.GetColumn(firstStat)` (`split_points.cpp:85`).

    **What is permuted is the STAT COLUMNS and the index, never the binned
    matrix.** `src` here is gradients and weights, `lineSize` strides between
    stat planes. The compressed index is read indirectly through the
    permuted row ids and does not move for the life of the fit. That
    distinction is the reason CatBoost can afford a per-level reorder at all:
    mojotrees measured reordering the BIN MATRIX at 1.535x slower on GPU and
    2.6x on CPU, which is a different and far more expensive operation.

    Grid y is the LEAF, so one launch reorders every splitting leaf.
    """
    var stat_base = Int(stat_base_in)
    var num_stats = Int(num_stats_in)
    var line_size = Int(line_size_in)
    var leaf_id = Int(leaves.unsafe_load(Int(block_idx.y)))

    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)

    while i < size:
        # `const ui32 loadIdx = __ldg(map + i)` (`split_points.cu:198`) and
        # `__ldg(src + loadIdx + k * lineSize)` (`:202`).
        var load_idx = Int(ldg(gather_map + (offset + i)))
        for k in range(num_stats):
            dst.unsafe_store(
                offset + i + (stat_base + k) * line_size,
                ldg(src + (offset + load_idx + k * line_size)),
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
        # `__ldg(map + i)` (`split_points.cu:198`) and `__ldg(src + loadIdx)`
        # (`:202`) at one column.
        var load_idx = Int(ldg(gather_map + (offset + i)))
        dst.unsafe_store(offset + i, ldg(src + (offset + load_idx)))
        i += stride


#: `GatherInplaceLeqSize<1024>`, the `FAST_PATH(1024)` arm of their ladder
#: (`split_points.cpp:127-135`).
#:
#: The ladder names four sizes -- 12288, 6144, 3072, 1024 -- and all four are
#: explicitly instantiated at `split_points.cu:786-816`. Three of them are
#: DEAD. The whole else-branch is only entered when `maxLeafSize <= 1024`
#: (`split_points.cpp:65`), so `maxLeafSize > 6144`, `> 3072` and `> 1024` can
#: never be true inside it and `FAST_PATH(1024)` is the only reachable arm.
#: Only 1024 is ported. The other three are noise in their file, not a
#: capability we are missing.
comptime GATHER_INPLACE_SIZE = 1024

#: `const ui32 blockSize = 1024` (`split_points.cu:103`), which is also the
#: kernel's `BlockSize` template default (`split_points.cu:52`).
#:
#: `Size == BlockSize`, so their strided `for (i = tid; i < Size; i += BlockSize)`
#: runs exactly once per thread. The loop is transcribed in that form anyway,
#: because the stride is what keeps the kernel correct at ANY block size: if a
#: device refuses a 1024-wide threadgroup this constant drops on its own and
#: the body does not change.
comptime GATHER_INPLACE_BLOCK = 1024


def gather_inplace_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    gather_map: MutPointer[UInt32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[UInt32, MutAnyOrigin],
    line_size_in: Int32,
):
    """`GatherInplaceImpl` (`split_points.cu:52-91`), copied.

    THE fast path. For a leaf that fits in one threadgroup's shared memory,
    the entire copy-then-gather pair collapses into one kernel with zero
    global round trips: stage the leaf in `__shared__` (`split_points.cu:59`),
    `__syncthreads()` (`split_points.cu:85`), write it back over itself
    (`split_points.cu:90`). The scratch plane is never read and never written.

    Four launches become one for every leaf that fits, and the criterion is
    the LEAF, not the dataset: whatever `n_rows` is, the level eventually gets
    fine enough that every leaf takes this path and the levels after it are
    the many-leaves levels, where the launch count hurts most. This port
    measured about 30 ms of every 129 ms tree as fixed per-level launch
    overhead; this is aimed at it.

    Grid x is `1 + statCount` and grid y is the leaf, exactly their
    `numBlocks.x = 1 + statCount; numBlocks.y = leavesCount`
    (`split_points.cu:105-106`). Block 0 carries the row index and block
    `k + 1` carries stat column `k`, which is their `blockIdx.x == 0` branch
    (`split_points.cu:61`), so ONE launch moves both payloads.

    ===================== DEVIATION BLOCK =====================
    THE ELEMENT TYPE. Theirs stages `__shared__ char4 tmp[Size]`
    (`split_points.cu:59`) and reinterprets BOTH payloads to it:

        char4* data = blockIdx.x == 0 ? (char4*)indices
                                      : (char4*)(stats + (blockIdx.x - 1) * lineSize);
            (`split_points.cu:61`)

    `char4` is nothing but "some 4-byte word I will move without looking at
    it", which is what lets one kernel serve a `float*` and a `ui32*` at once.
    Mojo has no `char4`; the equivalent is `UInt32` plus a pointer `bitcast`,
    so both payloads arrive here as `MutPointer[UInt32]` and the driver hands
    the stats plane over as `stats.unsafe_ptr().bitcast[UInt32]()`. Four bytes
    in, four bytes out, bit for bit: a permutation never inspects the value,
    so reinterpreting floats as words is not an approximation of their trick,
    it IS their trick.

    This is why there is ONE kernel here and not a float one plus a UInt32
    one.
    ===========================================================

    ===================== DEVIATION BLOCK =====================
    `WriteThrough(data + i, tmp[i])` (`split_points.cu:90`) is
    `cub::ThreadStore<cub::STORE_WT>` (`kernel_helpers.cuh:190`), a
    write-through store that lands in L2 without dirtying L1. Every streaming
    store in this file goes through it. Mojo ships no counterpart; this is a
    plain store.

    Searched 2026-08-19, and the paths are named so the next person does not
    repeat them:

      * `max.gpu.memory.memory` (`/api/mojo/max/gpu/memory/memory/`) is the
        module whose own summary is "load/store: Memory access operations
        with cache control". It exports `load` with `read_only`,
        `cache_policy: CacheOperation` and `eviction_policy: CacheEviction`,
        and it exports NO store. The only stores in it are `multimem_st`
        (NVIDIA multimem) and the `cp_async_bulk_*` TMA family, neither of
        which is a scalar store with a cache modifier.
      * `std.gpu.intrinsics` exports `CacheOperation`, `Scope`, `ldg`,
        `threadfence` and the AMD/permlane intrinsics. No store.
      * `std.sys.intrinsics` exports `masked_store`, `compressed_store`,
        `strided_store`, `scatter` and `prefetch`. All of these are addressing
        patterns; none carries a cache or temporality hint.

    So the LOAD half of their pair ships and the STORE half does not. The
    load half is now CALLED: their `Ldg` / `__ldg` (`kernel_helpers.cuh:180`)
    is `std.gpu.intrinsics.ldg`, and every read they mark with it in this
    file and in the three histogram kernels goes through it, at their sites
    and only at their sites. `max.gpu.memory.memory.load[read_only=True]`
    is the other spelling and carries `cache_policy` and `eviction_policy`
    knobs their call does not use, so `ldg` is the closer mirror.

    The store half remains missing, and that is what this block records.

    Cost of the missing store hint: the permuted leaf is dead to this kernel
    the moment it lands, so a normal store pollutes L1 with data nothing will
    read. That is cache pressure, not a correctness difference.
    ===========================================================
    """
    # `__shared__ char4 tmp[Size]` (`split_points.cu:59`). 1024 words = 4 KB.
    var tmp = stack_allocation[
        GATHER_INPLACE_SIZE,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()

    var line_size = Int(line_size_in)

    # `char4* data = blockIdx.x == 0 ? (char4*)indices
    #                               : (char4*)(stats + (blockIdx.x - 1) * lineSize);`
    # (`split_points.cu:61`). One launch, two different allocations, selected
    # by the block's x index. Block 0 is the row index, which has no column
    # stride to apply; every other block is stat column `blockIdx.x - 1`.
    var col = Int(block_idx.x)
    var data = indices if col == 0 else stats
    var col_base = 0 if col == 0 else (col - 1) * line_size

    # `TDataPartition part = Ldg(parts + leafId)` (`split_points.cu:66`), read
    # field by field: a whole-struct load in a kernel kills the Metal
    # compiler, and our partitions are two arrays regardless. `leaf[blockIdx.y]`
    # (`:64`) is PLAIN in their file and stays plain here.
    var leaf_id = Int(leaves.unsafe_load(Int(block_idx.y)))
    var offset = Int(ldg(part_offset + leaf_id))
    var size = Int(ldg(part_size + leaf_id))

    var tid = Int(thread_idx.x)

    # `map += offset; data += offset` (`split_points.cu:75-76`). Carried as an
    # index rather than a pointer bump, which is how every other kernel in
    # this file addresses a leaf.
    var base = col_base + offset

    # `for (ui32 i = tid; i < Size; i += BlockSize) if (i < size)`
    # (`split_points.cu:78-83`). BOTH bounds are theirs: `Size` is the compile
    # time capacity of `tmp` and `size` is this leaf's actual extent, and the
    # inner guard is what lets one 1024-wide launch serve a 7-row leaf.
    for i in range(tid, GATHER_INPLACE_SIZE, GATHER_INPLACE_BLOCK):
        if i < size:
            # `const ui32 loadIdx = __ldg(map + i)` and
            # `tmp[i] = __ldg(data + loadIdx)` (`split_points.cu:80-81`).
            var load_idx = Int(ldg(gather_map + (offset + i)))
            tmp[i] = ldg(data + (base + load_idx))

    # `__syncthreads()` (`split_points.cu:85`). A THREADGROUP barrier, not
    # `syncwarp`: `tmp` is written by every thread of the block and read back
    # at a permuted index, so a producer and its consumer are in general in
    # different warps. Narrowing this to warp scope is a correctness bug, not
    # a tuning knob.
    barrier()

    # `WriteThrough(data + i, tmp[i])` (`split_points.cu:88-91`). The write
    # lands on top of the read, which is the whole point: the leaf is
    # permuted IN PLACE and global memory is touched once, not three times.
    for i in range(tid, GATHER_INPLACE_SIZE, GATHER_INPLACE_BLOCK):
        if i < size:
            data.unsafe_store(base + i, tmp[i])


def launch_reorder_in_leaves(
    ctx: DeviceContext,
    n_leaf_slots: Int,
    wide: Int,
    max_leaf_rows: Int,
    stat_count: Int,
    line_size: Int,
    mut leaves: DeviceBuffer[DType.uint32],
    mut part_offset: DeviceBuffer[DType.uint32],
    mut part_size: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut temp_stats: DeviceBuffer[DType.float32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut temp_index: DeviceBuffer[DType.uint32],
    mut gather_map: DeviceBuffer[DType.uint32],
    sm_count: Int = -1,
) raises -> Int:
    """`TSplitPointsKernel::Run` (`split_points.cpp:64-136`), the whole if/else.

    ONE function because theirs is one function. The stat columns and the row
    index are not two independent jobs with two policies; they are two arms of
    a single branch taken on a single number, and in the fast arm they are one
    kernel launch. Splitting them in two here is what made the fast path cost
    2 launches instead of their 1.

    THE PATH CHOICE IS THEIRS, and it is made on the largest leaf being split,
    `maxLeafSize` (`split_points.cpp:60-63`):

        if (maxLeafSize > 1024) { ... } else { FAST_PATH(1024) }
            (`split_points.cpp:65`, `:113`)

    SLOW PATH, `maxLeafSize > 1024` (`split_points.cpp:66-111`):

        for (firstStat = 0; firstStat < numStats; firstStat += StatsPerKernel)
            CopyInLeaves<float>(...); GatherInLeaves<float>(...);
        //now copy indices
        { CopyInLeaves<ui32>(...); GatherInLeaves(...); }

    Eight columns at a time out to scratch and straight back in place, then
    ONE more pair for the index. Every launch is restricted to the splitting
    leaves, and the scratch is `min(8, statCount) * lineSize` floats
    (`split_points.cpp:10`) rather than a second stats plane that would have
    to live for the whole fit.

    The index pass is theirs' own separate block, with `numCopies = 1` and
    `lineSize = Indices.Size()` rather than `AlignedColumnSize()`
    (`split_points.cpp:96-98`). At one column the stride multiplies nothing,
    which is why the two `ui32` kernels here carry no stride argument at all.

    FAST PATH, otherwise (`split_points.cpp:113-135`): every leaf is staged in
    shared memory and permuted in place, ONE launch for the index and all the
    stat columns together, `temp_stats` and `temp_index` untouched. Once the
    level is fine enough that no leaf exceeds 1024 rows -- which is most of
    the tree -- the copy/gather pair is not cheaper, it is absent.

    Launches per level: 1 on the fast path, `2 * ceil(statCount / 8) + 2` on
    the slow one. That count is returned, because the caller has to tell the
    manager about each launch and a hardcoded number would be wrong in one of
    the two arms.

    `temp_stats` holds chunk-local columns `0 .. n-1`; `stats` is indexed at
    absolute columns `first_stat .. first_stat + n - 1`.

    `if (leavesCount)` guards their slow launches (`split_points.cu:224`,
    `:246`) and `IsGridEmpty(numBlocks)` guards the fast one
    (`split_points.cu:108`). Their fast grid is `1 + statCount` wide
    (`split_points.cu:105`), so it is never empty for a live leaf even at
    `statCount == 0`, and the guard below is on the leaf count alone.
    """
    if n_leaf_slots <= 0:
        return 0

    # `else { FAST_PATH(1024) }` (`split_points.cpp:113`, `:134`). Their test
    # is `maxLeafSize > 1024` on the largest SPLITTING leaf, not on the row
    # count, so a 500,000-row dataset still takes this path once the level is
    # fine enough.
    if max_leaf_rows <= GATHER_INPLACE_SIZE:
        ctx.enqueue_function[gather_inplace_kernel](
            leaves.unsafe_ptr(),
            part_offset.unsafe_ptr(),
            part_size.unsafe_ptr(),
            gather_map.unsafe_ptr(),
            # `(char4*)indices` (`split_points.cu:61`). Already `ui32`, so the
            # reinterpretation is the identity and no bitcast is needed.
            row_index.unsafe_ptr(),
            # the `char4` reinterpretation, `(char4*)(stats + ...)`
            # (`split_points.cu:61`)
            stats.unsafe_ptr().bitcast[UInt32](),
            Int32(line_size),
            # `numBlocks.x = 1 + statCount` (`split_points.cu:105`): block 0
            # is the row index, block `k + 1` is stat column `k`.
            grid_dim=(1 + stat_count, n_leaf_slots, 1),
            block_dim=(GATHER_INPLACE_BLOCK, 1, 1),
        )
        return 1

    # `numBlocks.x = (leavesCount > 4 ? 2 : 4) * TArchProps::SMCount()`
    # (`split_points.cu:220` for `CopyInLeaves`, `:242` for `GatherInLeaves`).
    # Read from the DEVICE here rather than taken from the caller: `wide` is
    # `min(256, ceil(n_rows / 256))`, which is device-blind and is not what
    # their grid x means. See `split_points_grid_x`.
    #
    # Below the fast-path return, because the fast path's grid is theirs too
    # (`1 + statCount`, `split_points.cu:105`) and does not want this number;
    # the attribute query is host work and the fast path is most levels.
    #
    # `wide` is still in this signature only because the call sites live in
    # `greedy_search_helper.mojo`, which another lane owns this round. It is
    # READ BY NOTHING now and the parameter should go with the call sites.
    # Threaded in, queried only when the caller has nothing: one
    # `ctx.get_attribute` costs 1.26 ms on Metal (measured) and this runs
    # once per level. Their `TArchProps::SMCount()` is a cached static.
    var sm = sm_count
    if sm < 0:
        sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var grid_x = split_points_grid_x(n_leaf_slots, sm)

    var launches = 0
    var first_stat = 0
    while first_stat < stat_count:
        var n = stat_count - first_stat
        if n > STATS_PER_KERNEL:
            n = STATS_PER_KERNEL

        ctx.enqueue_function[copy_in_leaves_kernel](
            leaves.unsafe_ptr(),
            part_offset.unsafe_ptr(),
            part_size.unsafe_ptr(),
            stats.unsafe_ptr(),
            temp_stats.unsafe_ptr(),
            Int32(first_stat),
            Int32(n),
            Int32(line_size),
            grid_dim=(grid_x, n_leaf_slots, 1),
            block_dim=(LEAF_COPY_BLOCK, 1, 1),
        )
        ctx.enqueue_function[gather_in_leaves_kernel](
            leaves.unsafe_ptr(),
            part_offset.unsafe_ptr(),
            part_size.unsafe_ptr(),
            temp_stats.unsafe_ptr(),
            gather_map.unsafe_ptr(),
            stats.unsafe_ptr(),
            Int32(first_stat),
            Int32(n),
            Int32(line_size),
            grid_dim=(grid_x, n_leaf_slots, 1),
            block_dim=(LEAF_COPY_BLOCK, 1, 1),
        )
        launches += 2
        first_stat += STATS_PER_KERNEL

    # `//now copy indices` (`split_points.cpp:90`), their own separate block
    # after the stat loop. Theirs reuses `context.TempStorage` for it, cast to
    # `ui32` (`split_points.cpp:92`); this takes a separate `ui32` scratch
    # because Mojo buffers are typed. One column, so no chunking.
    #
    # The index matters exactly as much as the stats. It is what
    # `split_and_make_sequence_kernel` dereferences to reach the compressed
    # index next level, so permuting the payload without it reads the right
    # rows' bins against the wrong rows' gradients.
    ctx.enqueue_function[copy_index_in_leaves_kernel](
        leaves.unsafe_ptr(),
        part_offset.unsafe_ptr(),
        part_size.unsafe_ptr(),
        row_index.unsafe_ptr(),
        temp_index.unsafe_ptr(),
        grid_dim=(grid_x, n_leaf_slots, 1),
        block_dim=(LEAF_COPY_BLOCK, 1, 1),
    )
    ctx.enqueue_function[gather_index_in_leaves_kernel](
        leaves.unsafe_ptr(),
        part_offset.unsafe_ptr(),
        part_size.unsafe_ptr(),
        temp_index.unsafe_ptr(),
        gather_map.unsafe_ptr(),
        row_index.unsafe_ptr(),
        grid_dim=(grid_x, n_leaf_slots, 1),
        block_dim=(LEAF_COPY_BLOCK, 1, 1),
    )
    return launches + 2


def update_partition_stats_from_split_kernel(
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    win_cells: MutPointer[UInt32, MutAnyOrigin],
    split_features: MutPointer[CFeature, MutAnyOrigin],
    bin_feature_count_in: Int32,
    stat_count_in: Int32,
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
):
    """Both children's partition stats from the split record. FAST ARM ONLY.

    ===================== DEVIATION 1901 =====================
    THE OTHER HALF OF `TSplitPointsKernel`, REPLACED RATHER THAN PORTED.
    Their split updates `subsets->PartitionStats` inside the split ("Update
    part stats", `split_properties_helper.cpp:918`); this port never carried
    that half, and the driver instead re-reduces EVERY leaf's rows at the top
    of EVERY level (`compute_partition_stats` over `d_all_ids`,
    `greedy_search_helper_depthwise.mojo` DEVIATION 352) -- an
    O(max_leaves x n_rows x stat_count) sweep per tree where leaf-wise growth
    runs `max_leaves - 1` sequential splits. The recon against LightGBM's
    CUDA learner (recon_lightgbm_cuda.md, mechanism e2) prices that as the
    single largest asymptotic difference on the 1M-2M H100 gap: they write
    `left/right_sum_gradients/hessians/count` into the child leaf structs
    FROM THE SPLIT RECORD (`cuda_data_partition.cu:798-903`), sums their
    split finder already read off the histogram
    (`cuda_best_split_finder.cu:434-443`), and pay O(1) per split for the
    same information.

    This kernel is that mechanism on this port's planes. The split leaf's
    slot holds its SCANNED histogram (built when the leaf was scored, and a
    leaf splits only after it was scored), so the winning cell's inclusive
    prefix IS the sum over the rows that fail the `>` test -- exactly the
    rows `split_and_make_sequence_kernel` flags as "goes left":

        left  = histogram[leaf][stat][winning cell]     (ordered feature)
        right = part_stats[leaf][stat] - left

    A ONE-HOT winner inverts the pairing, because its flag test is EQUALITY
    (`feature_val == value` goes RIGHT, above) and its bins are never
    scanned (`ScanHistogramsImpl` skips one-hot features,
    `histogram_utils.cu:395-397`), so the cell holds the equal rows' own
    sums -- the RIGHT side -- and the left is the complement.

    The stat-0 clamp is `SubstractHistogramsImpl`'s, for the same reason it
    is theirs: stat 0 is the weight plane and the complement's cancellation
    can drive a derived count slightly negative, which would poison the
    leaf-value division. Gradients stay unclamped; a gradient sum is
    legitimately negative.

    PRICE, and why this is a FAST-mode arm and not the one path: the
    propagated sums are the SAME numbers only in exact arithmetic. In
    float32 they re-associate against the fresh reduction -- it is the
    histogram-subtraction tradeoff the lane already accepts, applied to the
    partition stats -- so the IDENTICAL column keeps DEVIATION 352's sweep
    byte for byte and this kernel is never launched there. The driver's
    end-of-tree sweep stays in BOTH modes, so leaf VALUES still come from
    the exact reduction; only the gains a level scores with ride on
    propagated stats.

    One thread per (split, stat): the parent's entry is read before either
    child's is written, and the left child overwrites the parent's slot
    (`MakeSplit` gives the left child the parent's id, `:861-862`), so the
    read-then-write must stay inside one thread. Control plane, not compute:
    the grid is `(1, leavesCount)` where the sweep it replaces was sized by
    `n_rows`.
    ==========================================================
    """
    var bin_feature_count = Int(bin_feature_count_in)
    var stat_count = Int(stat_count_in)
    var leaf_slot = Int(block_idx.y)
    var left_leaf = Int(left_leaves.unsafe_load(leaf_slot))
    var right_leaf = Int(right_leaves.unsafe_load(leaf_slot))
    var cell = Int(win_cells.unsafe_load(leaf_slot))
    # field read, not a whole-struct load; the Metal reason recorded on
    # `split_and_make_sequence_kernel`.
    var one_hot = split_features[unsafe_offset=leaf_slot].one_hot_feature

    var stat_id = Int(thread_idx.x)
    while stat_id < stat_count:
        var parent = part_stats.unsafe_load(left_leaf * stat_count + stat_id)
        var cell_sum = histograms.unsafe_load(
            left_leaf * bin_feature_count * stat_count
            + stat_id * bin_feature_count
            + cell
        )
        var derived = parent - cell_sum
        if stat_id == 0:
            # the weight plane's non-negativity guard, exactly where
            # `substract_histograms_kernel` applies theirs.
            cell_sum = max(cell_sum, Float32(0.0))
            derived = max(derived, Float32(0.0))
        var left_sum = derived if one_hot else cell_sum
        var right_sum = cell_sum if one_hot else derived
        part_stats.unsafe_store(left_leaf * stat_count + stat_id, left_sum)
        part_stats.unsafe_store(right_leaf * stat_count + stat_id, right_sum)
        stat_id += Int(block_dim.x)


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
# It is also the one place in `gbdt/` allowed to be better than CatBoost,
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

#: `const int blockSize = 512` with `const int N = 1`, the shape they reorder
#: at (`split_points.cu:722-723`, and again `reorder_one_bit.cu:35-36`). It
#: was 256 here, which put this file and `gpu_util/kernel/reorder_one_bit.mojo`
#: -- the same partition, ported twice -- on two different block sizes.
comptime PARTITION_BLOCK = 512


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

    var tid = Int(thread_idx.x)
    # GRID-STRIDE over chunks: with the blind level loop (DEVIATION 94)
    # the grid is machine-sized against an n_rows chunk bound, so a block
    # walks every `grid_dim.x`-th chunk of its leaf and an oversized
    # bound costs nothing. A full-coverage grid strides exactly once,
    # which is the old behaviour bit for bit. The loop condition is
    # uniform across the block, so the collectives inside stay sound.
    var chunk = Int(block_idx.x)
    while chunk * PARTITION_BLOCK < size:
        var i = chunk * PARTITION_BLOCK + tid
        var v = Int32(0)
        if i < size:
            if flags.unsafe_load(offset + i) == UInt8(0):
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

    The same subtraction settles the WITHIN-chunk rank. `in_range` is
    `i < size` and `i` is `chunk * PARTITION_BLOCK + tid`, so it is a prefix
    in `tid`: every thread below an in-range thread is itself in range, and
    the count of in-range threads before `tid` is exactly `tid`. So
    `rank_one == tid - rank_zero`, and only ONE block scan is needed. This
    used to run `block_prefix_sum` a second time over `is_one`, which is a
    full block collective with barriers bought for one subtraction.
    `ReorderOneBitImpl` (`reorder_one_bit_impl.cuh:126-177`) makes the same
    trade harder still: it has no `__syncthreads` and no collective at all,
    because its ones-prefix arrives already scanned.
    """
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaves.unsafe_load(leaf_slot))
    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))

    var tid = Int(thread_idx.x)
    var n_zeros = Int(leaf_total_zeros.unsafe_load(leaf_slot))
    # GRID-STRIDE over chunks, same shape and same reasoning as phase 1.
    var chunk = Int(block_idx.x)
    while chunk * PARTITION_BLOCK < size:
        var i = chunk * PARTITION_BLOCK + tid
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
        var rank_one = tid - rank_zero

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
        barrier()
        chunk += Int(grid_dim.x)


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
    sm_count: Int = -1,
) raises:
    """The three phases in order. Callers should not launch them by hand.

    `sm_count > 0` machine-sizes the chunk grid of phases 1 and 3 (their
    occupancy-target pattern, `split_points_grid_x`) and the kernels
    stride the rest; the blind level loop (DEVIATION 94) passes it
    because its `max_leaf_rows` is the whole-dataset bound and a
    full-coverage grid against that bound is mostly empty blocks. The
    default keeps full coverage, where the stride runs exactly once.

    `max_leaf_rows` sizes the grid and the scratch: it is the largest leaf in
    the level, so a level of small leaves does not pay for the depth-0 shape.

    The empty guard is theirs. Every launch in their file is wrapped in one --
    `if (leavesCount)` at `split_points.cu:224`, `:246`, `:401`, `:558`, and
    `if (part.Size)` around both sort paths at `:674` and `:694` -- because a
    zero-extent grid is a launch failure, not a no-op. A level with nothing to
    split reaches here once terminal leaves exist.
    """
    if n_leaf_slots <= 0:
        return

    # `const int numBlocks = (part.Size + (N * blockSize) - 1) /
    #  (N * blockSize)` (`split_points.cu:724`). FULL COVERAGE, not the
    # `(leavesCount > 4 ? 2 : 4) * SMCount()` occupancy target the copy and
    # gather kernels take: `ReorderOneBitImpl` does not stride, one block owns
    # one chunk, so grid x has to cover the rows here and `split_points_grid_x`
    # must not be used for it.
    var max_chunks = (max_leaf_rows + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    if max_chunks < 1:
        max_chunks = 1
    var chunk_grid = max_chunks
    if sm_count > 0:
        var target = split_points_grid_x(n_leaf_slots, sm_count)
        if target < chunk_grid:
            chunk_grid = target

    ctx.enqueue_function[partition_count_chunks_kernel](
        leaves.unsafe_ptr(),
        part_offset.unsafe_ptr(),
        part_size.unsafe_ptr(),
        flags.unsafe_ptr(),
        chunk_zeros.unsafe_ptr(),
        Int32(max_chunks),
        grid_dim=(chunk_grid, n_leaf_slots, 1),
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
        grid_dim=(chunk_grid, n_leaf_slots, 1),
        block_dim=(PARTITION_BLOCK, 1, 1),
    )
