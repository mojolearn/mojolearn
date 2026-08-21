"""Add a grown tree's leaf values to the running predictions.

PORT OF the `AddModelValue` step of
`catboost/cuda/methods/doc_parallel_boosting.h:265` (`AppendModels`), whose
device side is CatBoost's add-model-value kernel family
(`catboost/cuda/models/kernel/add_model_value.cu`).

Their `AppendModels` runs once per boosting iteration and is what turns a
tree into a change in the cursor. `Rescale(step)` multiplies the leaf values
by the learning rate first (`doc_parallel_boosting.h:388`), which is why the
rate arrives here as a plain multiplier rather than being baked into the leaf
estimate.

================= DEVIATION BLOCK =================
CatBoost applies the model by EVALUATING it, on the learn set as much as
anywhere else. `TAddModelDocParallel<TObliviousTreeModel>::Append` calls
`AddObliviousTree` (`add_oblivious_tree_model_doc_parallel.cpp:191-192`),
which launches `AddObliviousTreeImpl` (`add_model_value.cu:70-120`): each row
reads its bin out of the compressed index at every level and builds the leaf
index bit by bit. There is NO partition-indexed path in their doc-parallel
weak learner. Their other kernel, `AddBinModelValueImpl` (`:14-53`), takes a
precomputed per-row bin array and belongs to the leaves estimator's `MoveTo`
(`pointwise_oracle.cpp:50-52`), not to `AppendModels`.

On the LEARN set during training the partition is already known: growth left
`row_index` permuted so that every row of leaf `L` sits in
`[offset[L], offset[L] + size[L])`. So the leaf assignment is read off the
partition instead of recomputed. It is exact rather than approximate and is
the same answer their evaluation produces, which `boosting_check` asserts by
comparing this cursor against `predict`'s rather than assuming it.

`add_bin_values.mojo` beside this file IS the port of `AddObliviousTreeImpl`,
so the evaluating form is not missing; what is missing is a test cursor to
point it at. Recorded so that gap is visible.
===================================================
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx


def add_model_value_kernel(
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    leaf_values: MutPointer[Float32, MutAnyOrigin],
    learning_rate: Float32,
    cursor: MutPointer[Float32, MutAnyOrigin],
    dim_count_in: Int32,
    cursor_stride_in: Int32,
):
    """Grid y is the LEAF, x strides that leaf's rows, **z is the APPROX
    DIMENSION**.

    THE TWO LAYOUTS ARE DIFFERENT AND THAT IS THEIRS. Their `shift` is
    BIN-MAJOR -- `newPoint[bin * cursorDim + dim]`
    (`pointwise_oracle.cpp:41-47`) -- while the cursor is PLANE-MAJOR,
    one contiguous column per class. So the value for (leaf, dim) is read
    at `leaf * dimCount + dim` and written into `dim * cursorStride + row`.
    Reading both as the same layout is the kind of mistake that trains a
    model with the classes rotated and no assertion fires.

    `dim_count_in == 1` with `cursor_stride_in == 0` is the single-
    dimensional path every pointwise loss takes, and it produces exactly
    the arithmetic this kernel had before the z axis existed: `block_idx.z`
    is 0, so the read is `leaf * 1 + 0` and the write offset is 0.

    `Rescale(step)` is folded in as `learning_rate` rather than mutating the
    stored leaf values, so the model keeps the UNSCALED estimate and the rate
    stays a property of the boosting loop. Their `Rescale` mutates the weak
    model in place; ours does not, because nothing here serialises a model
    yet and an unscaled leaf is the more useful thing to inspect.
    """
    var leaf = Int(block_idx.y)
    var dim = Int(block_idx.z)
    var dim_count = Int(dim_count_in)
    var plane = dim * Int(cursor_stride_in)
    var offset = Int(part_offset.unsafe_load(leaf))
    var size = Int(part_size.unsafe_load(leaf))
    var value = (
        leaf_values.unsafe_load(leaf * dim_count + dim) * learning_rate
    )

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var row = plane + Int(row_index.unsafe_load(offset + i))
        cursor.unsafe_store(row, cursor.unsafe_load(row) + value)
        i += stride
