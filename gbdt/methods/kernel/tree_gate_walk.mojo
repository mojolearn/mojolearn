# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The oblivious searcher's post-tree gate walk on the device (lane
t1-tree-builder-ops, 2026-11-08; docs/plans/EXECUTOR_T1_TREE_OPS.md, tree op
TREE_OP_GATE_WALK).

The doc-parallel searcher (`oblivious_tree_doc_parallel_structure_searcher`)
grows every level blind and records each level's winner `(feature, bin)` in
`d_winners_ids`; the gates CatBoost applies BEFORE each split are applied
after the tree, in level order:

  - a sentinel feature id (`PW_SENTINEL_ID`: every candidate scored
    non-finite) refuses the tree: status = level + 1 (the host raises by
    name with the level);
  - `structure.HasSplit(bestSplit)`: a level that repeats a split already in
    the structure ends the tree (levels from it on are discarded);
  - otherwise the split is appended with `BIN_SPLIT_TAKE_BIN` for a one-hot
    feature (word 3 of its `d_feat_table` row) and `BIN_SPLIT_TAKE_GREATER`
    otherwise.

That walk was a host loop over the drained winner records; this kernel is the
same walk, statement for statement, over the device records, writing the
structure as Int32 words (`TBinarySplit`'s fields). Integer work only, one
thread (the trip count is the tree depth, at most 16 levels: model
structure), so every vendor and the host column write the same words.

Layout of `structure` (`tree_structure_words(max_depth)` Int32 words):
  word 0  status: 0, or level + 1 of the first sentinel level
  word 1  kept depth (splits in the structure)
  word 2 + 3 k, 3 k + 1, 3 k + 2  split k: feature id, bin, split type
Word 0 is a failure word in the sense of `core/step_gate.mojo`
(`step_fail_joins_loop` reads word 0 of its buffer), so a device loop stops
on a refused tree without a host read.
"""
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.methods.kernel.pointwise_split_resolve import PW_SENTINEL_ID
from gbdt.models.oblivious_model import BIN_SPLIT_TAKE_BIN, BIN_SPLIT_TAKE_GREATER
from gbdt.models.kernel.add_bin_values import (
    compute_bins_and_add_gated_kernel,
    compute_bins_gated_kernel,
)

#: `compute_bins_kernel`'s and the apply's block (COMPUTE_BINS_BLOCK_SIZE,
#: the 256 of `_apply_last_tree_to_test`).
comptime TREE_APPLY_BLOCK = 256

comptime TREE_STRUCT_STATUS = 0
comptime TREE_STRUCT_DEPTH = 1
comptime TREE_STRUCT_SPLITS = 2
#: Words per split record: feature id, bin, split type.
comptime TREE_STRUCT_SPLIT_WORDS = 3
#: The searcher's feature table row (`PointwiseTreeWorkspace.d_feat_table`):
#: offset_elems, mask, shift, one_hot.
comptime TREE_FEAT_TABLE_WORDS = 4


def tree_structure_words(max_depth: Int) -> Int:
    """Int32 words of a structure record for trees of at most `max_depth`."""
    return TREE_STRUCT_SPLITS + TREE_STRUCT_SPLIT_WORDS * max_depth


def tree_gate_walk_kernel(
    gate: MutPointer[Int32, MutAnyOrigin],
    winners_ids: MutPointer[UInt32, MutAnyOrigin],
    feat_table: MutPointer[UInt32, MutAnyOrigin],
    max_depth_in: Int32,
    structure: MutPointer[Int32, MutAnyOrigin],
):
    """One thread: the walk above, while word 0 of `gate` is 0."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    var max_depth = Int(max_depth_in)
    var status = Int32(0)
    var kept = 0
    for level in range(max_depth):
        var fid_u = winners_ids.unsafe_load(2 * level)
        var bin_u = winners_ids.unsafe_load(2 * level + 1)
        if fid_u == PW_SENTINEL_ID:
            status = Int32(level + 1)
            break
        var fid = Int32(Int(fid_u))
        var bin = Int32(Int(bin_u))
        # `structure.HasSplit(bestSplit)`, before applying it
        var seen = False
        for k in range(kept):
            var at = TREE_STRUCT_SPLITS + TREE_STRUCT_SPLIT_WORDS * k
            if structure.unsafe_load(at) == fid and structure.unsafe_load(at + 1) == bin:
                seen = True
        if seen:
            break
        var one_hot = feat_table.unsafe_load(TREE_FEAT_TABLE_WORDS * Int(fid_u) + 3) != UInt32(0)
        var at = TREE_STRUCT_SPLITS + TREE_STRUCT_SPLIT_WORDS * kept
        structure.unsafe_store(at, fid)
        structure.unsafe_store(at + 1, bin)
        structure.unsafe_store(
            at + 2, Int32(BIN_SPLIT_TAKE_BIN) if one_hot else Int32(BIN_SPLIT_TAKE_GREATER)
        )
        kept += 1
    structure.unsafe_store(TREE_STRUCT_STATUS, status)
    structure.unsafe_store(TREE_STRUCT_DEPTH, Int32(kept))


def tree_records_kernel(
    gate: MutPointer[Int32, MutAnyOrigin],
    structure: MutPointer[Int32, MutAnyOrigin],
    feat_table: MutPointer[UInt32, MutAnyOrigin],
    max_depth_in: Int32,
    off: MutPointer[UInt32, MutAnyOrigin],
    shift: MutPointer[UInt32, MutAnyOrigin],
    mask: MutPointer[UInt32, MutAnyOrigin],
    bin: MutPointer[UInt32, MutAnyOrigin],
    take_equal: MutPointer[UInt8, MutAnyOrigin],
):
    """TREE_OP_TREE_RECORDS, one thread per level: the per-level apply
    records `compute_bins_kernel` / `compute_bins_and_add_kernel` read,
    from a structure record and a feature table built for the target's
    compressed index (`offset_elems` = column offset times that index's row
    count). The words the host packing wrote (`sym_compute_bins_pooled`,
    `_apply_last_tree_to_test`): offset, shift, mask, bin, and
    `take_equal = split type == BIN_SPLIT_TAKE_BIN`. Levels at or past the
    kept depth are left as they are (the kernels read `depth` levels)."""
    var level = Int(thread_idx.x)
    if Int(block_idx.x) != 0 or level >= Int(max_depth_in):
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    if level >= Int(structure.unsafe_load(TREE_STRUCT_DEPTH)):
        return
    var at = TREE_STRUCT_SPLITS + TREE_STRUCT_SPLIT_WORDS * level
    var fid = Int(structure.unsafe_load(at))
    var row = TREE_FEAT_TABLE_WORDS * fid
    off.unsafe_store(level, feat_table.unsafe_load(row))
    mask.unsafe_store(level, feat_table.unsafe_load(row + 1))
    shift.unsafe_store(level, feat_table.unsafe_load(row + 2))
    bin.unsafe_store(level, UInt32(Int(structure.unsafe_load(at + 1))))
    take_equal.unsafe_store(
        level,
        UInt8(1) if Int(structure.unsafe_load(at + 2)) == BIN_SPLIT_TAKE_BIN else UInt8(0),
    )


struct TreeRecords(Movable):
    """The per-level apply records of one oblivious tree, on the device:
    the five `max_depth`-word planes `compute_bins_kernel` and
    `compute_bins_and_add_kernel` read. Allocated once per fit (per target
    compressed index) and rewritten by `tree_op_tree_records` each round."""

    var max_depth: Int
    var off: DeviceBuffer[DType.uint32]
    var shift: DeviceBuffer[DType.uint32]
    var mask: DeviceBuffer[DType.uint32]
    var bin: DeviceBuffer[DType.uint32]
    var take_equal: DeviceBuffer[DType.uint8]

    def __init__(out self, ctx: DeviceContext, max_depth: Int) raises:
        var n = max_depth if max_depth > 0 else 1
        self.max_depth = max_depth
        self.off = ctx.enqueue_create_buffer[DType.uint32](n)
        self.shift = ctx.enqueue_create_buffer[DType.uint32](n)
        self.mask = ctx.enqueue_create_buffer[DType.uint32](n)
        self.bin = ctx.enqueue_create_buffer[DType.uint32](n)
        self.take_equal = ctx.enqueue_create_buffer[DType.uint8](n)


def tree_op_gate_walk(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut winners_ids: DeviceBuffer[DType.uint32],
    mut feat_table: DeviceBuffer[DType.uint32],
    max_depth: Int,
    mut structure: DeviceBuffer[DType.int32],
) raises:
    """Tree op TREE_OP_GATE_WALK (gbdt/tree_ops.mojo): the post-tree walk of
    the `2 * max_depth` winner records into `structure`
    (`tree_structure_words(max_depth)` words). Enqueue only; join the loop
    on a refused tree with `step_fail_joins_loop(ctx, words, structure, it)`."""
    if max_depth < 0 or max_depth > 32:
        raise Error("tree_op_gate_walk: max_depth " + String(max_depth) + " outside 0..32")
    ctx.enqueue_function[tree_gate_walk_kernel](  # small-launch(one thread over the tree levels): model structure
        gate.unsafe_ptr(),
        winners_ids.unsafe_ptr(),
        feat_table.unsafe_ptr(),
        Int32(max_depth),
        structure.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )


def tree_op_tree_records(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut structure: DeviceBuffer[DType.int32],
    mut feat_table: DeviceBuffer[DType.uint32],
    mut records: TreeRecords,
) raises:
    """Tree op TREE_OP_TREE_RECORDS: the apply records of the structure for
    the compressed index `feat_table` was built for. Replaces the host
    packing and the five uploads per tree. Enqueue only."""
    var md = records.max_depth if records.max_depth > 0 else 1
    ctx.enqueue_function[tree_records_kernel](  # small-launch(one thread per tree level): model structure
        gate.unsafe_ptr(),
        structure.unsafe_ptr(),
        feat_table.unsafe_ptr(),
        Int32(records.max_depth),
        records.off.unsafe_ptr(),
        records.shift.unsafe_ptr(),
        records.mask.unsafe_ptr(),
        records.bin.unsafe_ptr(),
        records.take_equal.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(md, 1, 1),
    )


def tree_op_leaf_bins(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut structure: DeviceBuffer[DType.int32],
    mut records: TreeRecords,
    mut cindex: DeviceBuffer[DType.uint32],
    n_rows: Int,
    mut out_bins: DeviceBuffer[DType.uint32],
) raises:
    """Tree op TREE_OP_LEAF_BINS: each row's leaf under the round's tree
    (`compute_bins_kernel`, the launch of `sym_compute_bins_pooled`), the
    depth read from the structure record. Enqueue only."""
    var blocks = (n_rows + TREE_APPLY_BLOCK - 1) // TREE_APPLY_BLOCK
    if blocks < 1:
        blocks = 1
    var depth_word = structure.unsafe_ptr() + TREE_STRUCT_DEPTH
    ctx.enqueue_function[compute_bins_gated_kernel](
        gate.unsafe_ptr(),
        depth_word,
        cindex.unsafe_ptr(),
        records.off.unsafe_ptr(),
        records.shift.unsafe_ptr(),
        records.mask.unsafe_ptr(),
        records.bin.unsafe_ptr(),
        records.take_equal.unsafe_ptr(),
        Int32(n_rows),
        out_bins.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(TREE_APPLY_BLOCK, 1, 1),
    )


def tree_op_apply(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut structure: DeviceBuffer[DType.int32],
    mut records: TreeRecords,
    mut cindex: DeviceBuffer[DType.uint32],
    mut leaf_values: DeviceBuffer[DType.float32],
    n_rows: Int,
    mut cursor: DeviceBuffer[DType.float32],
    approx_dim: Int,
    cursor_stride: Int,
    root_adds: Bool = False,
) raises:
    """Tree op TREE_OP_APPLY (the tree gather): `cursor[d * stride + i] +=
    leaf_values[leaf(i) * approx_dim + d]` for every row, the launch of
    `_apply_last_tree_to_test` (grid `min(ceil(n / 256), 1024)` x
    approx_dim). `leaf_values` already carry the learning rate. The cursor
    is carried state, so the add is behind the gate. `root_adds`: a depth-0
    tree adds its one leaf (the learn cursor) instead of nothing (the
    held-out cursor, as `_apply_last_tree_to_test`). Enqueue only."""
    var wide = (n_rows + TREE_APPLY_BLOCK - 1) // TREE_APPLY_BLOCK
    if wide > 1024:
        wide = 1024
    if wide < 1:
        wide = 1
    var depth_word = structure.unsafe_ptr() + TREE_STRUCT_DEPTH
    ctx.enqueue_function[compute_bins_and_add_gated_kernel](
        gate.unsafe_ptr(),
        depth_word,
        cindex.unsafe_ptr(),
        records.off.unsafe_ptr(),
        records.shift.unsafe_ptr(),
        records.mask.unsafe_ptr(),
        records.bin.unsafe_ptr(),
        records.take_equal.unsafe_ptr(),
        leaf_values.unsafe_ptr(),
        Int32(n_rows),
        cursor.unsafe_ptr(),
        Int32(approx_dim),
        Int32(cursor_stride),
        Int32(1) if root_adds else Int32(0),
        grid_dim=(wide, approx_dim, 1),
        block_dim=(TREE_APPLY_BLOCK, 1, 1),
    )
