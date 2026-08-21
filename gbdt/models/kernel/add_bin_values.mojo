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
