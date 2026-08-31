# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apply a stored oblivious tree to rows, by EVALUATING it.

PORT OF `AddObliviousTreeImpl`, `catboost/cuda/models/kernel/add_model_value.cu:70-120`
at CatBoost `54a8143a`, which is the kernel their own `AppendModels` reaches
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
    """Their `AddObliviousTreeImpl`, transcribed.

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
    one contiguous column per class. A port that read both the same way
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
            # their `takeEqual[level] ? (featureVal == value) : (featureVal
            # > value)` (`add_model_value.cu:110`): `>` is the ordered
            # predicate (`EBinSplitType::TakeBin`), `==` the one-hot one
            # (`TakeVal`), per LEVEL exactly as their mask arrays carry it.
            var split: Bool
            if take_equal.unsafe_load(level) != UInt8(0):
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
# its own header. It is the same upstream file as the two kernels above and
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
# hands out. THE TWO AGREEING IS NOT ASSUMED: `mojo_only/depthwise_check.mojo`
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

    Two reasons, both already established in this port and neither of them a
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
