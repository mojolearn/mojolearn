# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apply a stored oblivious tree to rows, by EVALUATING it.

Reference: `AddObliviousTreeImpl`, `catboost/cuda/models/kernel/add_model_value.cu:70-120`
(CatBoost `54a8143a`), which is the kernel their own `AppendModels` reaches
on the learn set as well as the test set
(`add_oblivious_tree_model_doc_parallel.cpp:191-192`).

`kernel_add_model_value.mojo` updates the LEARN cursor by reading each row's
leaf off the partition growth already produced. That is exact and free, and
it is useless for any row the tree was not grown on.

This is the other form, the one CatBoost needs for a test set and for
inference: walk the tree's splits, build the leaf index bit by bit, add the
leaf's value. It agrees with the partition form on the learn set by
construction, and `boosting_check` asserts that rather than assuming it.

## The bit order is the model

    leaf = sum over levels of (bit_l << l)

Level 0 is the LEAST significant bit. See `oblivious_model.mojo` for why the
growth numbering already produces this. Reading the bits the other way round
gives a leaf index that is a valid permutation of the right one, so every
total is preserved and every individual prediction is wrong. Conservation
cannot see it; comparing against the learn cursor can.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.apple_fast_tree_experiments import AFT_P05

#: lane/fam2-gbdt F3 (IDENTICAL, every vendor; default ON): the non-resident
#: `predict` (`gbdt/methods/doc_parallel_boosting.mojo`) launches the
#: oblivious ensemble four trees per launch (`compute_bins_and_add_four_kernel`,
#: the resident path's grouping) instead of one launch per tree. Same ordered
#: float32 adds per row, so no bit moves and the host column is untouched.
#: `-D MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF` (or `-D MOJOLEARN_IDN_ALL_OFF`)
#: restores one launch per tree.
comptime IDN_PREDICT_FOUR = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)

#: lane/fam2-gbdt F4 (IDENTICAL; CANDIDATE ARM, default OFF, enable with
#: `-D MOJOLEARN_IDN_GBDT_APPLY_WIDE`): an ensemble whose trees all share one
#: positive depth is applied `APPLY_WIDE_TREES` trees per launch
#: (`compute_bins_and_add_uniform_kernel`), each row carrying its sum in a
#: register through the trees in tree order: the same ordered float32 adds,
#: one cursor store per launch instead of one per tree. No bit moves. Time it
#: against the four-tree grouping (resident predict and `predict`). An
#: ensemble with mixed depths keeps the four-tree grouping.
comptime IDN_APPLY_WIDE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_GBDT_APPLY_WIDE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: trees per launch of the wide arm: bounded so one launch stays short (macOS
#: cuts a Metal launch that runs for seconds)
comptime APPLY_WIDE_TREES = 64


def compute_bins_and_add_kernel(
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    feature_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_shift: MutPointer[UInt32, MutAnyOrigin],
    feature_mask: MutPointer[UInt32, MutAnyOrigin],
    split_bin: MutPointer[UInt32, MutAnyOrigin],
    take_equal: MutPointer[UInt8, MutAnyOrigin],
    depth_in: Int32,
    leaf_values: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    cursor: MutPointer[Float32, MutAnyOrigin],
    dim_count_in: Int32,
    cursor_stride_in: Int32,
):
    """`AddObliviousTreeImpl`.

    Their loop, `add_model_value.cu:106-117`:

        const ui32 featureVal = __ldg((cindex + offsetsLocal[level]) + loadIdx) & mask;
        const ui32 split = (takeEqual[level] ? (featureVal == value) : featureVal > value);
        bin |= split << level;

    with `value = bins[level] << feature.Shift` and `mask = feature.Mask <<
    feature.Shift` (`:91-92`), then one grid-stride add of `leaves[bin]`.
    That is this kernel line for line; there is no fusion of two kernels
    here, because theirs is already one.

    Two shapes of theirs are not carried and neither changes a number:
    `readIndices` / `writeIndices`, which are null on this path, and the
    `__shared__` staging of the per-level masks, which is their way of
    broadcasting `depth <= 32` scalars that we pass as buffers.

    The split arrays are parallel and one entry per LEVEL, which is the whole
    of an oblivious tree's structure.

    ## THE APPROX DIMENSION IS `block_idx.y`

    Their `AddObliviousTree` takes a `TCudaBuffer` cursor whose COLUMN COUNT
    carries the dimension and adds every column
    (`add_model_value.cu`, `models/add_bin_values.h`). Ours takes a pointer
    plus a stride, so the dimension is an axis, exactly as
    `add_model_value_kernel` grew one for the same reason.

    THE TWO LAYOUTS DIFFER AND THAT IS THEIRS, and it is the same pairing
    as the estimator's: `leaf_values` is BIN-MAJOR --
    `[leaf * dimCount + dim]`, which is what `MakeEstimationResult`
    produced and what the model stores -- while the cursor is PLANE-MAJOR,
    one contiguous column per class. An implementation that read both the same way
    would predict with the classes rotated and nothing would assert.

    `dim_count_in == 1, cursor_stride_in == 0` is the single-dimensional
    path, and `block_idx.y` is 0 there, so the arithmetic is exactly what
    this kernel had before the axis existed.

    THE LEAF INDEX IS COMPUTED ONCE PER ROW AND SHARED BY EVERY DIMENSION,
    because an oblivious tree's structure does not depend on the approx:
    all `dimCount` values of a row come out of the SAME leaf. Recomputing
    it per dimension would be correct and would read the compressed index
    `dimCount` times for one answer.
    """
    var depth = Int(depth_in)
    var n_rows = Int(n_rows_in)
    var dim = Int(block_idx.y)
    var dim_count = Int(dim_count_in)
    var plane = dim * Int(cursor_stride_in)
    # One copy of the tiny tree descriptor per block instead of one global
    # load per row and level. Symmetric trees cap depth at 32.
    var meta = stack_allocation[
        5 * 32, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    if tid < depth:
        meta.unsafe_store(tid, feature_offset.unsafe_load(tid))
        meta.unsafe_store(32 + tid, feature_shift.unsafe_load(tid))
        meta.unsafe_store(64 + tid, feature_mask.unsafe_load(tid))
        meta.unsafe_store(96 + tid, split_bin.unsafe_load(tid))
        meta.unsafe_store(128 + tid, UInt32(take_equal.unsafe_load(tid)))
    barrier()
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)

    while i < n_rows:
        var leaf = 0
        for level in range(depth):
            var off = Int(meta.unsafe_load(level))
            var shift = meta.unsafe_load(32 + level)
            var mask = meta.unsafe_load(64 + level) << shift
            var value = meta.unsafe_load(96 + level) << shift
            var feature_val = compressed_index.unsafe_load(off + i) & mask
            # their `takeEqual[level] ? (featureVal == value) : (featureVal
            # > value)` (`add_model_value.cu:110`): `>` is the ordered
            # predicate (`EBinSplitType::TakeBin`), `==` the one-hot one
            # (`TakeVal`), per LEVEL exactly as their mask arrays carry it.
            var split: Bool
            if meta.unsafe_load(128 + level) != UInt32(0):
                split = feature_val == value
            else:
                split = feature_val > value
            if split:
                leaf += 1 << level
        cursor.unsafe_store(
            plane + i,
            cursor.unsafe_load(plane + i)
            + leaf_values.unsafe_load(leaf * dim_count + dim),
        )
        i += stride


def compute_bins_and_add_four_kernel(
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    feature_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_shift: MutPointer[UInt32, MutAnyOrigin],
    feature_mask: MutPointer[UInt32, MutAnyOrigin],
    split_bin: MutPointer[UInt32, MutAnyOrigin],
    take_equal: MutPointer[UInt8, MutAnyOrigin],
    depth0_in: Int32,
    depth1_in: Int32,
    depth2_in: Int32,
    depth3_in: Int32,
    tree_count_in: Int32,
    leaf_values: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    cursor: MutPointer[Float32, MutAnyOrigin],
    dim_count_in: Int32,
    cursor_stride_in: Int32,
):
    """Apply up to four consecutive oblivious trees in their original order.

    This is the resident-inference launch grouping of
    `compute_bins_and_add_kernel`: the same five descriptor planes are staged
    once per block, and each row performs the same ordered float32 additions.
    No tree or level is reordered and no reduction is introduced.
    """
    var depths = InlineArray[Int, 4](fill=0)
    depths[0] = Int(depth0_in)
    depths[1] = Int(depth1_in)
    depths[2] = Int(depth2_in)
    depths[3] = Int(depth3_in)
    var tree_count = Int(tree_count_in)
    var total_levels = 0
    for t in range(tree_count):
        total_levels += depths[t]
    var meta = stack_allocation[
        5 * 128, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    if tid < total_levels:
        meta.unsafe_store(tid, feature_offset.unsafe_load(tid))
        meta.unsafe_store(128 + tid, feature_shift.unsafe_load(tid))
        meta.unsafe_store(256 + tid, feature_mask.unsafe_load(tid))
        meta.unsafe_store(384 + tid, split_bin.unsafe_load(tid))
        meta.unsafe_store(512 + tid, UInt32(take_equal.unsafe_load(tid)))
    barrier()

    var n_rows = Int(n_rows_in)
    var dim = Int(block_idx.y)
    var dim_count = Int(dim_count_in)
    var plane = dim * Int(cursor_stride_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + tid
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < n_rows:
        var acc = cursor.unsafe_load(plane + i)
        var level_base = 0
        var leaf_base = 0
        for t in range(tree_count):
            var depth = depths[t]
            var leaf = 0
            for level in range(depth):
                var j = level_base + level
                var off = Int(meta.unsafe_load(j))
                var shift = meta.unsafe_load(128 + j)
                var mask = meta.unsafe_load(256 + j) << shift
                var value = meta.unsafe_load(384 + j) << shift
                var feature_val = compressed_index.unsafe_load(off + i) & mask
                var split: Bool
                if meta.unsafe_load(512 + j) != UInt32(0):
                    split = feature_val == value
                else:
                    split = feature_val > value
                if split:
                    leaf += 1 << level
            acc = acc + leaf_values.unsafe_load(
                leaf_base + leaf * dim_count + dim
            )
            level_base += depth
            leaf_base += (1 << depth) * dim_count
        cursor.unsafe_store(plane + i, acc)
        i += stride


def compute_bins_kernel(
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    feature_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_shift: MutPointer[UInt32, MutAnyOrigin],
    feature_mask: MutPointer[UInt32, MutAnyOrigin],
    split_bin: MutPointer[UInt32, MutAnyOrigin],
    take_equal: MutPointer[UInt8, MutAnyOrigin],
    depth_in: Int32,
    n_rows_in: Int32,
    out_bins: MutPointer[UInt32, MutAnyOrigin],
):
    """Their `ComputeObliviousTreeBinsImpl`
    (`models/kernel/add_model_value.cu:123-166`).

    THE SAME LOOP AS `compute_bins_and_add_kernel`, ending in a WRITE
    instead of an add -- and they are two kernels in CatBoost for the same
    reason they are two here (`:150-164` against `:106-118`). One is the
    apply; this one is the question "which leaf does this row fall in",
    which is what the leaves ESTIMATOR asks of a dataset the tree was not
    grown on (`doc_parallel_leaves_estimator.cpp:45-49`, where every
    estimation task computes its own bins before the oracle sees it).

    It has no approx-dimension axis and cannot have one: an oblivious
    tree's structure does not depend on the approx, so one leaf index
    serves every plane. Their `readIndices`/`writeIndices` are null on
    this path, as they are for the apply.
    """
    var depth = Int(depth_in)
    var n_rows = Int(n_rows_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)

    while i < n_rows:
        var leaf = 0
        for level in range(depth):
            var off = Int(feature_offset.unsafe_load(level))
            var shift = feature_shift.unsafe_load(level)
            var mask = feature_mask.unsafe_load(level) << shift
            var value = split_bin.unsafe_load(level) << shift
            var feature_val = compressed_index.unsafe_load(off + i) & mask
            var split: Bool
            if take_equal.unsafe_load(level) != UInt8(0):
                split = feature_val == value
            else:
                split = feature_val > value
            if split:
                leaf += 1 << level
        out_bins.unsafe_store(i, UInt32(leaf))
        i += stride


# =========================================================================
# THE NON-SYMMETRIC APPLY -- `ComputeNonSymmetricDecisionTreeBinsImpl`
# (`add_model_value.cu:353-395`)
# =========================================================================
#
# Added by the DEPTHWISE lane, 2026-08-22, at the FOOT of this file behind
# its own header. It is the same reference file as the two kernels above and
# the same job -- give every row its leaf -- for a tree whose leaves do not
# share a split list.
#
# WHY IT IS SO MUCH SIMPLER THAN THE STRUCTURE THAT PRODUCED IT. The tree
# arrives as the flat pre-order `TTreeNode` array (`TFlatTreeBuilder` in
# `greedy_subsets_searcher/model_builder.mojo`), where `left_subtree` and
# `right_subtree` are LEAF COUNTS. Walking it needs no stack and no depth
# bound at all:
#
#     bin = 0
#     loop:
#       going RIGHT:  bin += node.left_subtree     (skip the left leaves)
#                     stop if right_subtree == 1   (the right child IS a leaf)
#                     else advance by left_subtree (right child's node)
#       going LEFT:   stop if left_subtree == 1    (the left child IS a leaf)
#                     else advance by 1            (left child's node)
#
# and the accumulated `bin` is the number of leaves to the left of this row's
# leaf, which is exactly the numbering `TNonSymmetricTreeStructure.visit_bins`
# hands out. THE TWO AGREEING IS NOT ASSUMED: `checks/depthwise_check.mojo`
# walks both and compares per row.
#
# THE FEATURE ARRAY IS PARALLEL TO THE NODE ARRAY, one `TCFeature` per
# INTERNAL NODE, and both advance together -- `nodes += node.LeftSubtree;
# features += node.LeftSubtree` (`:381-382`). That is theirs and it is why
# the caller builds a per-node feature table rather than indexing a
# per-feature one; the deviation block below says what ours does instead.


def compute_non_symmetric_decision_tree_bins_kernel(
    node_offset: MutPointer[UInt32, MutAnyOrigin],
    node_mask: MutPointer[UInt32, MutAnyOrigin],
    node_shift: MutPointer[UInt32, MutAnyOrigin],
    node_one_hot: MutPointer[UInt8, MutAnyOrigin],
    node_bin: MutPointer[UInt32, MutAnyOrigin],
    node_left_subtree: MutPointer[UInt32, MutAnyOrigin],
    node_right_subtree: MutPointer[UInt32, MutAnyOrigin],
    node_count_in: Int32,
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    n_rows_in: Int32,
    out_bins: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeNonSymmetricDecisionTreeBinsImpl`, copied.

    ================= DEVIATION BLOCK =================
    THEIRS TAKES TWO STRUCT ARRAYS, `const TCFeature* features` and
    `const TTreeNode* nodes`, and walks them by POINTER ARITHMETIC. Ours
    takes seven parallel planes and walks an INDEX.

    Two reasons, both already established in this implementation and neither of them a
    preference:

    * `split_points.mojo`'s deviation block: binding a whole `CFeature` to a
      local kills the Metal backend ("Metal Compiler failed to compile
      metallib"), reproduced in a 25-line probe on 2026-08-19. Field-by-field
      access through a pointer compiles and runs. A `TTreeNode` is the same
      shape of struct and gets the same treatment rather than waiting to
      find out.
    * `enqueue_function` refuses several pointers derived from ONE
      allocation as aliasing mutable arguments, which is what a
      `bitcast`-to-struct view of a byte buffer would be.

    Semantically identical: same fields, same order, same loads. The
    `node_left_subtree` / `node_right_subtree` planes are `UInt32` where
    their `TTreeNode` fields are `ui16`, because a kernel parameter is
    Int32-shaped in Mojo; the HOST still refuses anything past 65,535 at
    `model_builder._to_ui16`, so no model can exist here that theirs could
    not hold.
    ===================================================

    `readIndices` and `writeIndices` are their two optional permutations and
    are NOT parameters here: every caller in this lane passes null for both
    (`bin = tid`, `writeIdx = tid`), exactly as their apply does on a
    doc-parallel dataset. A caller that needs them is a caller that does not
    exist yet, and an unused pointer parameter is an unreached branch.

    `nodes == nullptr` is their CONSTANT TREE -- a root that found no
    improving split. Their `bool stop = nodes == nullptr` makes the loop
    body run zero times and every row land in bin 0. `node_count == 0` is
    the same test here, and it is REACHABLE: a depthwise fit on a residual
    that is already flat produces exactly that tree.
    """
    var node_count = Int(node_count_in)
    var n_rows = Int(n_rows_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)

    while i < n_rows:
        var bin = 0
        var node = 0
        # `bool stop = nodes == nullptr;`
        var stop = node_count == 0

        while not stop:
            # `const ui32 featureVal = (__ldg(cindex + feature.Offset +
            #  loadIdx) >> feature.Shift) & feature.Mask;` (`:373`).
            #
            # NOTE THE ORDER: shift THEN mask, where the growth-side
            # `split_and_make_sequence_kernel` masks a PRE-SHIFTED value
            # (`value = bin << shift`, `mask = mask << shift`). The two are
            # the same predicate written two ways and both are theirs --
            # `split_points.cu:518` and this line. They are kept as written
            # on each side, because collapsing one into the other is the
            # kind of "obviously equivalent" edit that stops being
            # equivalent the day a shift is signed.
            var off = Int(node_offset.unsafe_load(node))
            var shift = node_shift.unsafe_load(node)
            var mask = node_mask.unsafe_load(node)
            var feature_val = (
                compressed_index.unsafe_load(off + i) >> shift
            ) & mask

            var this_bin = node_bin.unsafe_load(node)
            var split: Bool
            if node_one_hot.unsafe_load(node) != UInt8(0):
                split = feature_val == this_bin
            else:
                split = feature_val > this_bin

            var left_subtree = Int(node_left_subtree.unsafe_load(node))
            var right_subtree = Int(node_right_subtree.unsafe_load(node))

            if split:
                # `bin += node.LeftSubtree; stop = node.RightSubtree == 1;`
                # `if (!stop) { nodes += node.LeftSubtree; ... }` (`:377-383`)
                bin += left_subtree
                stop = right_subtree == 1
                if not stop:
                    node += left_subtree
            else:
                # `stop = node.LeftSubtree == 1; if (!stop) { nodes += 1; }`
                # (`:385-389`)
                stop = left_subtree == 1
                if not stop:
                    node += 1

        out_bins.unsafe_store(i, UInt32(bin))
        i += stride


def compute_bins_and_add_uniform_kernel(
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    feature_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_shift: MutPointer[UInt32, MutAnyOrigin],
    feature_mask: MutPointer[UInt32, MutAnyOrigin],
    split_bin: MutPointer[UInt32, MutAnyOrigin],
    take_equal: MutPointer[UInt8, MutAnyOrigin],
    depth_in: Int32,
    tree_count_in: Int32,
    leaf_values: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    cursor: MutPointer[Float32, MutAnyOrigin],
    dim_count_in: Int32,
    cursor_stride_in: Int32,
):
    """`IDN_APPLY_WIDE`: `tree_count` consecutive oblivious trees of ONE
    depth, applied in their original order. Tree `t`'s level records sit at
    `t * depth` in the five descriptor planes and its leaf values at
    `t * (1 << depth) * dim_count`; the planes are read from global memory
    (they are a few kilobytes and shared by every row). Each row performs
    `compute_bins_and_add_kernel`'s adds in tree order through `acc`."""
    var depth = Int(depth_in)
    var tree_count = Int(tree_count_in)
    var n_rows = Int(n_rows_in)
    var dim = Int(block_idx.y)
    var dim_count = Int(dim_count_in)
    var plane = dim * Int(cursor_stride_in)
    var leaves_per_tree = (1 << depth) * dim_count
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < n_rows:
        var acc = cursor.unsafe_load(plane + i)
        for t in range(tree_count):
            var level_base = t * depth
            var leaf = 0
            for level in range(depth):
                var j = level_base + level
                var off = Int(feature_offset.unsafe_load(j))
                var shift = feature_shift.unsafe_load(j)
                var mask = feature_mask.unsafe_load(j) << shift
                var value = split_bin.unsafe_load(j) << shift
                var feature_val = compressed_index.unsafe_load(off + i) & mask
                var split: Bool
                if take_equal.unsafe_load(j) != UInt8(0):
                    split = feature_val == value
                else:
                    split = feature_val > value
                if split:
                    leaf += 1 << level
            acc = acc + leaf_values.unsafe_load(
                t * leaves_per_tree + leaf * dim_count + dim
            )
        cursor.unsafe_store(plane + i, acc)
        i += stride


def uniform_positive_depth(depths: List[Int]) -> Int:
    """The one depth every tree shares when it is positive, else 0."""
    if len(depths) == 0:
        return 0
    var d = depths[0]
    if d < 1:
        return 0
    for t in range(1, len(depths)):  # small-loop(depths: trees): compares the per-tree depths, model metadata
        if depths[t] != d:
            return 0
    return d


def launch_oblivious_apply_wide(
    ctx: DeviceContext,
    mut cindex: DeviceBuffer[DType.uint32],
    mut d_off: DeviceBuffer[DType.uint32],
    mut d_shift: DeviceBuffer[DType.uint32],
    mut d_mask: DeviceBuffer[DType.uint32],
    mut d_bin: DeviceBuffer[DType.uint32],
    mut d_eq: DeviceBuffer[DType.uint8],
    mut d_vals: DeviceBuffer[DType.float32],
    depth: Int,
    n_trees: Int,
    n_rows: Int,
    mut cursor: DeviceBuffer[DType.float32],
    approx_dim: Int,
) raises:
    """`IDN_APPLY_WIDE`'s launches: `n_trees` trees of one positive `depth`,
    `APPLY_WIDE_TREES` per launch, back to back, no drain."""
    var wide = (n_rows + 255) // 256
    if wide > 1024:
        wide = 1024
    var t = 0
    while t < n_trees:
        var count = n_trees - t
        if count > APPLY_WIDE_TREES:
            count = APPLY_WIDE_TREES
        var lvl = t * depth
        var leaf = t * (1 << depth) * approx_dim
        ctx.enqueue_function[compute_bins_and_add_uniform_kernel](
            cindex.unsafe_ptr(),
            d_off.unsafe_ptr() + lvl,
            d_shift.unsafe_ptr() + lvl,
            d_mask.unsafe_ptr() + lvl,
            d_bin.unsafe_ptr() + lvl,
            d_eq.unsafe_ptr() + lvl,
            Int32(depth), Int32(count),
            d_vals.unsafe_ptr() + leaf,
            Int32(n_rows),
            cursor.unsafe_ptr(),
            Int32(approx_dim),
            Int32(n_rows),
            grid_dim=(wide, approx_dim, 1),
            block_dim=(256, 1, 1),
        )
        t += count


def launch_oblivious_apply_four(
    ctx: DeviceContext,
    mut cindex: DeviceBuffer[DType.uint32],
    mut d_off: DeviceBuffer[DType.uint32],
    mut d_shift: DeviceBuffer[DType.uint32],
    mut d_mask: DeviceBuffer[DType.uint32],
    mut d_bin: DeviceBuffer[DType.uint32],
    mut d_eq: DeviceBuffer[DType.uint8],
    mut d_vals: DeviceBuffer[DType.float32],
    depths: List[Int],
    n_rows: Int,
    mut cursor: DeviceBuffer[DType.float32],
    approx_dim: Int,
) raises:
    """`IDN_PREDICT_FOUR`: the packed ensemble four trees per launch
    (`compute_bins_and_add_four_kernel`), in tree order, back to back, no
    drain: the launch loop of `gbdt/resident_model.mojo::_apply`."""
    var wide = (n_rows + 255) // 256
    if wide > 1024:
        wide = 1024
    var n_trees = len(depths)
    var lvl = 0
    var leaf = 0
    var t = 0
    while t < n_trees:
        var count = n_trees - t
        if count > 4:
            count = 4
        var d0 = depths[t]
        var d1 = 0
        var d2 = 0
        var d3 = 0
        var leaves = 1 << d0
        if count > 1:
            d1 = depths[t + 1]
            leaves += 1 << d1
        if count > 2:
            d2 = depths[t + 2]
            leaves += 1 << d2
        if count > 3:
            d3 = depths[t + 3]
            leaves += 1 << d3
        # a group with no level at all keeps offset zero (no one-past-end
        # split pointer); a leading depth-0 tree needs no shift, because the
        # group's records start at `lvl` either way
        var split_offset = lvl if d0 + d1 + d2 + d3 > 0 else 0
        ctx.enqueue_function[compute_bins_and_add_four_kernel](
            cindex.unsafe_ptr(),
            d_off.unsafe_ptr() + split_offset,
            d_shift.unsafe_ptr() + split_offset,
            d_mask.unsafe_ptr() + split_offset,
            d_bin.unsafe_ptr() + split_offset,
            d_eq.unsafe_ptr() + split_offset,
            Int32(d0), Int32(d1), Int32(d2), Int32(d3), Int32(count),
            d_vals.unsafe_ptr() + leaf,
            Int32(n_rows),
            cursor.unsafe_ptr(),
            Int32(approx_dim),
            Int32(n_rows),
            grid_dim=(wide, approx_dim, 1),
            block_dim=(256, 1, 1),
        )
        lvl += d0 + d1 + d2 + d3
        leaf += leaves * approx_dim
        t += count


# ---------------------------------------------------------------------------
# lane/apple-fast-sym-feat (2026-10-03): referenced ONLY under the FAST +
# Apple guard `GBDT_PREDICT_PACKED` (`gbdt/resident_model.mojo`); IDENTICAL
# never instantiates it.
# ---------------------------------------------------------------------------

#: the most split LEVELS one staged chunk of trees holds: 5 words a level,
#: 20 KB of threadgroup memory. The host (`ResidentGbdtModel.__init__`)
#: cuts the ensemble into chunks of consecutive trees under this bound, so
#: the shared page always fits.
comptime PRED_ALL_CHUNK_LEVELS = 1024


def compute_bins_and_add_all_kernel(
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    feature_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_shift: MutPointer[UInt32, MutAnyOrigin],
    feature_mask: MutPointer[UInt32, MutAnyOrigin],
    split_bin: MutPointer[UInt32, MutAnyOrigin],
    take_equal: MutPointer[UInt8, MutAnyOrigin],
    tree_depth: MutPointer[UInt32, MutAnyOrigin],
    tree_level_start: MutPointer[UInt32, MutAnyOrigin],
    tree_leaf_start: MutPointer[UInt32, MutAnyOrigin],
    chunk_tree_start: MutPointer[UInt32, MutAnyOrigin],
    n_chunks_in: Int32,
    leaf_values: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    cursor: MutPointer[Float32, MutAnyOrigin],
    dim_count_in: Int32,
    cursor_stride_in: Int32,
):
    """Every oblivious tree of the ensemble in ONE launch, tree order kept.

    `compute_bins_and_add_kernel` per tree, fused: the ensemble is walked
    in chunks of consecutive trees (`chunk_tree_start[c] .. [c + 1])`,
    each chunk's level records (offset, shift, mask, bin, take-equal) staged
    once into threadgroup memory, and every row of the grid-stride adds the
    chunk's leaf values to its cursor word IN TREE ORDER before the next
    chunk is staged. The per-tree kernel does `cursor = cursor + leaf` once
    per tree; this does the same float32 adds in the same order, through a
    register inside a chunk and the cursor word between chunks, so the
    cursor leaves bit for bit as it did after the last per-tree launch.
    `tree_depth[t]`, `tree_level_start[t]` (index of the tree's first
    level record) and `tree_leaf_start[t]` (index of its first leaf value,
    bin-major `[leaf * dim + dim]` as the model stores them) describe the
    trees; `block_idx.y` is the cursor plane as in the per-tree kernel.
    """
    var n_rows = Int(n_rows_in)
    var dim = Int(block_idx.y)
    var dim_count = Int(dim_count_in)
    var plane = dim * Int(cursor_stride_in)
    var n_chunks = Int(n_chunks_in)
    var meta = stack_allocation[
        5 * PRED_ALL_CHUNK_LEVELS,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    for c in range(n_chunks):
        var t0 = Int(chunk_tree_start.unsafe_load(c))
        var t1 = Int(chunk_tree_start.unsafe_load(c + 1))
        var l0 = Int(tree_level_start.unsafe_load(t0))
        var l1 = Int(tree_level_start.unsafe_load(t1))
        var n_lv = l1 - l0
        var j = tid
        while j < n_lv:
            meta.unsafe_store(j, feature_offset.unsafe_load(l0 + j))
            meta.unsafe_store(
                PRED_ALL_CHUNK_LEVELS + j, feature_shift.unsafe_load(l0 + j)
            )
            meta.unsafe_store(
                2 * PRED_ALL_CHUNK_LEVELS + j, feature_mask.unsafe_load(l0 + j)
            )
            meta.unsafe_store(
                3 * PRED_ALL_CHUNK_LEVELS + j, split_bin.unsafe_load(l0 + j)
            )
            meta.unsafe_store(
                4 * PRED_ALL_CHUNK_LEVELS + j,
                UInt32(take_equal.unsafe_load(l0 + j)),
            )
            j += Int(block_dim.x)
        barrier()
        comptime if AFT_P05:
            # P05: a two-plane row tile shares every tree/split descriptor.
            # The second plane is guarded; every worker still hits both
            # shared-page barriers even when its rows are out of range.
            var i = Int(block_idx.x) * Int(block_dim.x) * 2 + tid
            while i < n_rows:
                var i1 = i + Int(block_dim.x)
                var has_second = i1 < n_rows
                var acc = cursor.unsafe_load(plane + i)
                var acc1 = Float32(0)
                if has_second:
                    acc1 = cursor.unsafe_load(plane + i1)
                for t in range(t0, t1):
                    var depth = Int(tree_depth.unsafe_load(t))
                    var lb = Int(tree_level_start.unsafe_load(t)) - l0
                    var leaf_base = Int(tree_leaf_start.unsafe_load(t))
                    var leaf = 0
                    var leaf1 = 0
                    for level in range(depth):
                        var k = lb + level
                        var off = Int(meta.unsafe_load(k))
                        var shift = meta.unsafe_load(PRED_ALL_CHUNK_LEVELS + k)
                        var mask = meta.unsafe_load(2 * PRED_ALL_CHUNK_LEVELS + k) << shift
                        var value = meta.unsafe_load(3 * PRED_ALL_CHUNK_LEVELS + k) << shift
                        var equal = meta.unsafe_load(4 * PRED_ALL_CHUNK_LEVELS + k) != UInt32(0)
                        var feature_val = compressed_index.unsafe_load(off + i) & mask
                        var take = feature_val == value if equal else feature_val > value
                        if take:
                            leaf += 1 << level
                        if has_second:
                            var feature_val1 = compressed_index.unsafe_load(off + i1) & mask
                            var take1 = feature_val1 == value if equal else feature_val1 > value
                            if take1:
                                leaf1 += 1 << level
                    acc = acc + leaf_values.unsafe_load(leaf_base + leaf * dim_count + dim)
                    if has_second:
                        acc1 = acc1 + leaf_values.unsafe_load(leaf_base + leaf1 * dim_count + dim)
                cursor.unsafe_store(plane + i, acc)
                if has_second:
                    cursor.unsafe_store(plane + i1, acc1)
                i += stride * 2
        else:
            var i = Int(block_idx.x) * Int(block_dim.x) + tid
            while i < n_rows:
                var acc = cursor.unsafe_load(plane + i)
                for t in range(t0, t1):
                    var depth = Int(tree_depth.unsafe_load(t))
                    var lb = Int(tree_level_start.unsafe_load(t)) - l0
                    var leaf_base = Int(tree_leaf_start.unsafe_load(t))
                    var leaf = 0
                    for level in range(depth):
                        var k = lb + level
                        var off = Int(meta.unsafe_load(k))
                        var shift = meta.unsafe_load(PRED_ALL_CHUNK_LEVELS + k)
                        var mask = meta.unsafe_load(2 * PRED_ALL_CHUNK_LEVELS + k) << shift
                        var value = meta.unsafe_load(3 * PRED_ALL_CHUNK_LEVELS + k) << shift
                        var feature_val = compressed_index.unsafe_load(off + i) & mask
                        var split: Bool
                        if meta.unsafe_load(4 * PRED_ALL_CHUNK_LEVELS + k) != UInt32(0):
                            split = feature_val == value
                        else:
                            split = feature_val > value
                        if split:
                            leaf += 1 << level
                    acc = acc + leaf_values.unsafe_load(
                        leaf_base + leaf * dim_count + dim
                    )
                cursor.unsafe_store(plane + i, acc)
                i += stride
        barrier()
