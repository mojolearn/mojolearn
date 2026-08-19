"""Apply a stored oblivious tree to rows, by EVALUATING it.

PORT OF `catboost/cuda/models/kernel/add_model_value.cu` and the
`ComputeBins` half of `add_oblivious_tree_model_doc_parallel.cpp` at CatBoost
`54a8143a`.

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
    depth_in: Int32,
    leaf_values: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    cursor: MutPointer[Float32, MutAnyOrigin],
):
    """Their `ComputeBins` then `AddBinModelValue`, fused into one pass.

    Theirs writes the bin indices to a buffer and adds in a second kernel,
    because their bins are reused by other machinery. Nothing here reuses
    them yet, so the round trip through memory is skipped; the arithmetic is
    identical and the fusion is noted rather than silent.

    The split arrays are parallel and one entry per LEVEL, which is the whole
    of an oblivious tree's structure.
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
            # `> value` is the ordered-feature predicate, their
            # `EBinSplitType::TakeBin`. One-hot flips this to `==` and is not
            # reachable until categorical features land.
            if feature_val > value:
                leaf += 1 << level
        cursor.unsafe_store(
            i, cursor.unsafe_load(i) + leaf_values.unsafe_load(leaf)
        )
        i += stride
