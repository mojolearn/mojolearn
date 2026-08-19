"""Add a grown tree's leaf values to the running predictions.

PORT OF the `AddModelValue` step of
`catboost/cuda/methods/doc_parallel_boosting.h:265` (`AppendModels`), whose
device side is CatBoost's add-model-value kernel family.

Their `AppendModels` runs once per boosting iteration and is what turns a
tree into a change in the cursor. `Rescale(step)` multiplies the leaf values
by the learning rate first (`doc_parallel_boosting.h:390`), which is why the
rate arrives here as a plain multiplier rather than being baked into the leaf
estimate.

================= DEVIATION BLOCK =================
CatBoost applies the model by EVALUATING it: each row walks the tree's splits
and lands in a leaf. That is required for a test set, where no partition
exists.

On the LEARN set during training the partition is already known: growth left
`row_index` permuted so that every row of leaf `L` sits in
`[offset[L], offset[L] + size[L])`. So the leaf assignment is read off the
partition instead of recomputed, which is exact rather than approximate and
is the same answer their evaluation would produce.

The moment a test cursor exists this kernel is NOT enough and the evaluating
form has to be ported beside it. Recorded so that gap is visible.
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
):
    """Grid y is the LEAF, x strides that leaf's rows.

    `Rescale(step)` is folded in as `learning_rate` rather than mutating the
    stored leaf values, so the model keeps the UNSCALED estimate and the rate
    stays a property of the boosting loop. Their `Rescale` mutates the weak
    model in place; ours does not, because nothing here serialises a model
    yet and an unscaled leaf is the more useful thing to inspect.
    """
    var leaf = Int(block_idx.y)
    var offset = Int(part_offset.unsafe_load(leaf))
    var size = Int(part_size.unsafe_load(leaf))
    var value = leaf_values.unsafe_load(leaf) * learning_rate

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var row = Int(row_index.unsafe_load(offset + i))
        cursor.unsafe_store(row, cursor.unsafe_load(row) + value)
        i += stride
