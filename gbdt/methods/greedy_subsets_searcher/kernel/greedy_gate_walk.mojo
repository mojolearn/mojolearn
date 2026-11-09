# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The greedy SymmetricTree searcher's post-tree gates on the device (lane
t3-gbdt-greedy-on-device-loop, 2026-11-09; docs/plans/EXECUTOR_T1_TREE_OPS.md
section 8, tree op TREE_OP_GREEDY_SEARCH).

`run_tree_layout_traced` grows every level blind and records each level's
winner in `winners_score[level]` / `winners_bf[level]` (a flat bin-feature
id). Its host path drains the tree once and walks the records with
`accept_symmetric_level_winner` in level order, then rolls the partition
sizes back over the discarded levels (`merge_leaf_sizes_kernel` per level).
Under `enqueue_only=True` both run here instead, statement for statement:

`greedy_gate_walk_kernel` (one thread; the trip count is the tree depth,
model structure), per level:
  - `best_bin == 0xFFFFFFFF or best_bin >= hist_cells`: every split scored
    non-finite; status `level + 1` (the host raises CatBoost's "All splits
    have infinite score" message with the level);
  - the bin-feature owns no feature (`resolve_split`'s refusal): status
    `GREEDY_STATUS_NO_FEATURE + level + 1`;
  - `not (best_score > 0)`: the tree ends (levels from it on discarded);
  - the split repeats one already kept (`HasSplit`): the tree ends;
  - otherwise the split is appended as `(feature, bin, split type)`, the
    type `BIN_SPLIT_TAKE_BIN` for a one-hot feature, else
    `BIN_SPLIT_TAKE_GREATER`.
The resolution `bin-feature -> (feature, bin, type)` is a table built once
per fit from the searcher's layout (`greedy_bin_feature_table`), the host
walk `resolve_split` did per level.

`greedy_partition_tail_kernel` (one block): the multi-level rollback of the
partition sizes (the left child kept the parent's offset at every level, so
merging the final sizes pairwise once per discarded level restores the kept
level's sizes exactly, the host path's argument), the slots past the kept
leaves zeroed (so a consumer may launch over all `1 << max_depth` slots and
the empty ones hold no rows), the root-coverage invariant (DEVIATION 2002's
model-shape check: the sizes sum to `n_rows`; status
`GREEDY_STATUS_COVERAGE` otherwise), and the partition words in the layout
of `DeviceLeafPartitioner.d_part` (Int64 offsets `[0, nl)`, sizes
`[nl, 2 nl)`, total, out-of-range flag), so TREE_OP_LEAF_ESTIMATE and the
host-tail round read the searcher's own partition.

Integer work only: every vendor and the host column write the same words.
Both kernels are gated (word 0 of `gate`): past the loop's stop they write
nothing, the round's search scratch is recomputed for nobody.

Layout of `structure`: `gbdt/methods/kernel/tree_gate_walk.mojo`
(`tree_structure_words(max_depth)` Int32 words; word 0 status, word 1 kept
depth, then 3 words per split).
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.sync import barrier

from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.methods.kernel.tree_gate_walk import (
    TREE_STRUCT_DEPTH,
    TREE_STRUCT_SPLIT_WORDS,
    TREE_STRUCT_SPLITS,
    TREE_STRUCT_STATUS,
)
from gbdt.models.oblivious_model import BIN_SPLIT_TAKE_BIN, BIN_SPLIT_TAKE_GREATER

#: Words per bin-feature in the resolution table: feature id, bin, split type.
comptime GREEDY_BF_WORDS = 3
#: Feature id of a bin-feature no feature owns.
comptime GREEDY_BF_NO_FEATURE = UInt32(0xFFFFFFFF)
#: The winner sentinel `resolve_and_pack_kernel` leaves when no candidate
#: scored finite.
comptime GREEDY_BF_SENTINEL = UInt32(0xFFFFFFFF)
#: Status base of an unresolvable winner (`resolve_split`'s refusal):
#: `GREEDY_STATUS_NO_FEATURE + level + 1`. Below LEAF_STATUS_HOST_TAIL
#: (1 << 20) and above every depth, so the codes never collide.
comptime GREEDY_STATUS_NO_FEATURE = 1 << 16
#: Status of a final partition whose sizes do not sum to the row count.
comptime GREEDY_STATUS_COVERAGE = 1 << 17
#: Threads of the tail kernel's one block.
comptime GREEDY_TAIL_BLOCK = 256


def greedy_bin_feature_table(
    ctx: DeviceContext, layout: CompressedIndexLayout
) raises -> DeviceBuffer[DType.uint32]:
    """The `bin-feature -> (feature, bin, split type)` table of a searcher
    layout (`build_layout(fold_counts, one_hot)`, the layout
    `run_tree_layout_traced` builds), `GREEDY_BF_WORDS` words per
    bin-feature over `layout.hist_cells` bin-features. The same walk as
    `resolve_split` and `accept_symmetric_level_winner`'s split type, done
    once per fit. A bin-feature no feature owns keeps
    `GREEDY_BF_NO_FEATURE`. One upload and one drain, before the loop."""
    var cells = layout.hist_cells
    var n = cells if cells > 0 else 1
    var table = List[UInt32](length=GREEDY_BF_WORDS * n, fill=UInt32(0))
    for c in range(n):  # small-loop(hist_cells: bin-features of the layout, once per fit): the resolution table, model metadata
        table[GREEDY_BF_WORDS * c] = GREEDY_BF_NO_FEATURE
    for i in range(len(layout.features)):  # small-loop(features: layout feature descriptors, once per fit): resolve_split's walk as a table
        var f = layout.features[i]
        if f.folds == 0:
            continue
        var lo = Int(f.first_fold_index)
        var hi = lo + Int(f.folds)
        var split_type = UInt32(BIN_SPLIT_TAKE_BIN) if f.one_hot_feature else UInt32(BIN_SPLIT_TAKE_GREATER)
        for bf in range(lo, hi):  # small-loop(folds: one feature's bins, once per fit): resolution table rows
            if bf < 0 or bf >= n:
                continue
            # `resolve_split` returns the FIRST feature whose range holds the
            # bin-feature: keep an earlier owner
            if table[GREEDY_BF_WORDS * bf] != GREEDY_BF_NO_FEATURE:
                continue
            table[GREEDY_BF_WORDS * bf] = UInt32(i)
            table[GREEDY_BF_WORDS * bf + 1] = UInt32(bf - lo)
            table[GREEDY_BF_WORDS * bf + 2] = split_type
    var d = ctx.enqueue_create_buffer[DType.uint32](len(table))
    ctx.enqueue_copy(dst_buf=d, src_ptr=table.unsafe_ptr())
    ctx.synchronize()
    _ = table^  # past the drain
    return d^


def greedy_gate_walk_kernel(
    gate: MutPointer[Int32, MutAnyOrigin],
    winners_score: MutPointer[Float32, MutAnyOrigin],
    winners_bf: MutPointer[UInt32, MutAnyOrigin],
    bf_table: MutPointer[UInt32, MutAnyOrigin],
    hist_cells_in: Int32,
    max_depth_in: Int32,
    structure: MutPointer[Int32, MutAnyOrigin],
):
    """One thread: the gate walk of the module docstring, while word 0 of
    `gate` is 0."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    var max_depth = Int(max_depth_in)
    var hist_cells = Int(hist_cells_in)
    var status = Int32(0)
    var kept = 0
    for level in range(max_depth):
        var best_bin_u = winners_bf.unsafe_load(level)
        if best_bin_u == GREEDY_BF_SENTINEL or Int(best_bin_u) >= hist_cells:
            status = Int32(level + 1)
            break
        var row = GREEDY_BF_WORDS * Int(best_bin_u)
        var fid_u = bf_table.unsafe_load(row)
        if fid_u == GREEDY_BF_NO_FEATURE:
            status = Int32(GREEDY_STATUS_NO_FEATURE + level + 1)
            break
        var best_score = winners_score.unsafe_load(level)
        if not (best_score > Float32(0.0)):
            break
        var fid = Int32(Int(fid_u))
        var bin = Int32(Int(bf_table.unsafe_load(row + 1)))
        # `structure.HasSplit(bestSplit)`, before applying it
        var seen = False
        for k in range(kept):
            var at_k = TREE_STRUCT_SPLITS + TREE_STRUCT_SPLIT_WORDS * k
            if structure.unsafe_load(at_k) == fid and structure.unsafe_load(at_k + 1) == bin:
                seen = True
        if seen:
            break
        var at = TREE_STRUCT_SPLITS + TREE_STRUCT_SPLIT_WORDS * kept
        structure.unsafe_store(at, fid)
        structure.unsafe_store(at + 1, bin)
        structure.unsafe_store(at + 2, Int32(Int(bf_table.unsafe_load(row + 2))))
        kept += 1
    structure.unsafe_store(TREE_STRUCT_STATUS, status)
    structure.unsafe_store(TREE_STRUCT_DEPTH, Int32(kept))


def greedy_partition_tail_kernel(
    gate: MutPointer[Int32, MutAnyOrigin],
    structure: MutPointer[Int32, MutAnyOrigin],
    p_off: MutPointer[UInt32, MutAnyOrigin],
    p_sz: MutPointer[UInt32, MutAnyOrigin],
    max_depth_in: Int32,
    n_rows_in: Int32,
    words: MutPointer[Int64, MutAnyOrigin],
):
    """One block: the rollback, the zeroed tail, the coverage invariant and
    the partition words of the module docstring, while word 0 of `gate` is 0
    and the walk accepted the tree (status 0)."""
    if Int(block_idx.x) != 0:
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    if structure.unsafe_load(TREE_STRUCT_STATUS) != Int32(0):
        return
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var max_depth = Int(max_depth_in)
    var nl = 1 << max_depth
    var kept = Int(structure.unsafe_load(TREE_STRUCT_DEPTH))
    var live = nl
    for _ in range(max_depth - kept):
        var h2 = live // 2
        # `sizes[i] += sizes[half + i]` (`merge_leaf_sizes_kernel`'s statement)
        var i = tid
        while i < h2:
            p_sz.unsafe_store(i, p_sz.unsafe_load(i) + p_sz.unsafe_load(h2 + i))
            i += nt
        barrier()
        # the discarded half holds no leaf any more
        var j = h2 + tid
        while j < live:
            p_sz.unsafe_store(j, UInt32(0))
            j += nt
        barrier()
        live = h2
    var k = tid
    while k < nl:
        words.unsafe_store(k, Int64(Int(p_off.unsafe_load(k))))
        words.unsafe_store(nl + k, Int64(Int(p_sz.unsafe_load(k))))
        k += nt
    barrier()
    if tid == 0:
        # the root-coverage sum, in leaf order (integer: any order is exact)
        var total = 0
        for leaf in range(nl):
            total += Int(p_sz.unsafe_load(leaf))
        words.unsafe_store(2 * nl, Int64(total))
        words.unsafe_store(2 * nl + 1, Int64(0))
        if total != Int(n_rows_in):
            structure.unsafe_store(TREE_STRUCT_STATUS, Int32(GREEDY_STATUS_COVERAGE))


def greedy_gate_and_tail(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut winners_score: DeviceBuffer[DType.float32],
    mut winners_bf: DeviceBuffer[DType.uint32],
    mut bf_table: DeviceBuffer[DType.uint32],
    hist_cells: Int,
    max_depth: Int,
    n_rows: Int,
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut structure: DeviceBuffer[DType.int32],
    mut words: DeviceBuffer[DType.int64],
) raises:
    """Enqueue the walk, then the tail (module docstring). `structure` holds
    `tree_structure_words(max_depth)` words, `words` `2 * (1 << max_depth) +
    2`. Join the loop on `structure` (`step_fail_joins_loop`) after this.
    Enqueue only."""
    if max_depth < 0 or max_depth > 16:
        raise Error("greedy_gate_and_tail: max_depth " + String(max_depth) + " outside 0..16")
    if len(bf_table) < GREEDY_BF_WORDS * (hist_cells if hist_cells > 0 else 1):
        raise Error("greedy_gate_and_tail: the bin-feature table is smaller than the layout")
    if len(words) < 2 * (1 << max_depth) + 2:
        raise Error("greedy_gate_and_tail: the partition words hold fewer than 2 * leaves + 2")
    ctx.enqueue_function[greedy_gate_walk_kernel](  # small-launch(one thread over the tree levels): model structure
        gate.unsafe_ptr(),
        winners_score.unsafe_ptr(),
        winners_bf.unsafe_ptr(),
        bf_table.unsafe_ptr(),
        Int32(hist_cells),
        Int32(max_depth),
        structure.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )
    ctx.enqueue_function[greedy_partition_tail_kernel](  # small-launch(one block over the tree's leaves): the rollback is level-serial, model structure
        gate.unsafe_ptr(),
        structure.unsafe_ptr(),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        Int32(max_depth),
        Int32(n_rows),
        words.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(GREEDY_TAIL_BLOCK, 1, 1),
    )


def root_partition_kernel(
    p_off: MutPointer[UInt32, MutAnyOrigin],
    p_sz: MutPointer[UInt32, MutAnyOrigin],
    max_leaves_in: Int32,
    n_rows_in: Int32,
):
    """Their `CreateInitialSubsets`' root partition on the device: every
    offset 0, every size 0 but the root's, `n_rows`. One thread per slot.
    Replaces the per-tree host fill of `h_off` / `h_sz` and its two uploads
    (same words)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(max_leaves_in):
        return
    p_off.unsafe_store(i, UInt32(0))
    p_sz.unsafe_store(i, UInt32(Int(n_rows_in)) if i == 0 else UInt32(0))
