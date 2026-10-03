"""One host wait per TREE for the Depthwise policy (FAST, Apple; opt-in).

`-D MOJOLEARN_GBDT_DW_TREE_SYNC` (lane apple-fast-depthwise, third pass; it
implies `DW_NO_LEVEL_SYNC` and `DW_FUSED_CHAIN`). Under `DW_NO_LEVEL_SYNC` a
level still ends in one host wait: the winners, the new leaf sizes and the
split count come home, and the host builds the NEXT level's plan (which
sibling to compute, which to derive, which slots to zero, which leaves to
score) from them. Here that plan stays on the device, so the host enqueues
every level of the tree back to back and waits ONCE, after the end-of-tree
partition-stats sweep. The host then replays its own bookkeeping (leaves,
paths, terminal marks, the plan) from the per-level records that came home
in that one wait, and checks the device's lists against its own replay.

Every launch in the blind loop has a HOST grid (an upper bound per level:
at most `min(2^level, max_leaves)` leaves can be scored or split at a
level) and a DEVICE count. The kernels that already take a device count
(`dw_select_splits_kernel`, the fused chain's GUARD arm) read it; the
histogram kernels (copy, zero, build, scan, subtract) and the scorer take
ID LISTS, so the device pads each list to its cap with a DUMMY leaf slot:
one extra partition slot past the tree's (`part_size` 0, histogram all
zeros, partition stats zero) that every kernel can touch harmlessly -- a
size-0 build accumulates nothing, zeroing or scanning an all-zero slot
leaves zeros, `dummy - dummy` is zero, a copy onto itself is a no-op, and a
scored dummy writes records into slots the fold never reads. No kernel
outside this file changes.

The per-leaf state the host keeps (`TLeaf`: size, histograms type, best
split defined, terminal, depth; `hist_slot_dirty`; `parent_of`) lives in
five `n_slots`-word planes (`DW_TS_*`), and the per-level outputs the host
needs for its replay are written to LEVEL-INDEXED slices (winner records,
visit lists, plan lists, the partition sizes after the level's split, the
three counters), so nothing is overwritten before the one readback.

Integer moves only: the same selection rule (DEFINED and `Gain < 0`), the
same sibling rule (strict `<` on the LEFT child's size, tie computes the
RIGHT child, both terminal computes nothing), the same terminal rule
(`size <= min_leaf_size` or `depth >= max_depth`), the same visit rule
(not terminal, best split undefined, ascending id), the same stop rules
(`leaves >= max_leaves`, or a level that split nothing). The tree is the
same tree; only the synchronisation moves.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from gbdt.methods.greedy_subsets_searcher.kernel.split_chain_fused import (
    DW_FEAT_WORDS,
    DW_WINNER_DEFINED,
    DW_WINNER_WORDS,
)

#: One thread per leaf slot / visit slot / split slot.
comptime DW_TS_BLOCK = 64

#: The per-leaf state planes, `n_slots` words each, in one buffer.
comptime DW_TS_DEPTH = 0
comptime DW_TS_TERM = 1
comptime DW_TS_DEF = 2
comptime DW_TS_HT = 3
comptime DW_TS_DIRTY = 4
comptime DW_TS_STATE_PLANES = 5

#: `EHistogramsType`, as the host's `LeafRecord` carries it.
comptime DW_TS_HT_ZEROES = UInt32(0)
comptime DW_TS_HT_PREVIOUS = UInt32(1)
comptime DW_TS_HT_CURRENT = UInt32(2)

#: The per-level counters: leaves before the level's split, leaves scored,
#: leaves split. `DW_TS_COUNTS` words per level, `levels + 1` levels (the
#: leaf count after the last level lands in the extra triple).
comptime DW_TS_C_LEAVES = 0
comptime DW_TS_C_VISIT = 1
comptime DW_TS_C_SPLIT = 2
comptime DW_TS_COUNTS = 3

#: The per-level id lists, `max_leaves` words each, dummy-padded: the
#: deferred parent-histogram copy pairs, the zero set, the build set, the
#: subtract pairs (`from` = big, `what` = small) and the visit list. One
#: slot per split PAIR of the previous level for the first six (gaps are
#: dummies); the visit list is compacted, ascending by id.
comptime DW_TS_L_COPY_SRC = 0
comptime DW_TS_L_COPY_DST = 1
comptime DW_TS_L_ZERO = 2
comptime DW_TS_L_BUILD = 3
comptime DW_TS_L_SUB_FROM = 4
comptime DW_TS_L_SUB_WHAT = 5
comptime DW_TS_L_VISIT = 6
comptime DW_TS_LISTS = 7

#: The per-level split records the host builds the tree from: per split
#: slot the left leaf id, the feature id and the bin.
comptime DW_TS_SPLIT_WORDS = 3


def dw_ts_init_kernel(
    state: MutPointer[UInt32, MutAnyOrigin],
    n_slots_in: Int32,
    counts: MutPointer[UInt32, MutAnyOrigin],
    lists0: MutPointer[UInt32, MutAnyOrigin],
    max_leaves_in: Int32,
):
    """Per-tree reset, as the host's `CreateInitialSubsets` leaves it AFTER
    level 0's plan: one leaf (the root, slot 0), depth 0, not terminal,
    best split undefined; the root's slot is the level's one compute (its
    `Zeroes` histogram), so it is `CurrentPath` and dirty from here on;
    level 0's lists are build = {0}, visit = {0}, everything else dummy.
    Grid covers `max(n_slots, max_leaves)` threads."""
    var n_slots = Int(n_slots_in)
    var max_leaves = Int(max_leaves_in)
    var dummy = UInt32(n_slots - 1)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < n_slots:
        state.unsafe_store(DW_TS_DEPTH * n_slots + t, UInt32(0))
        state.unsafe_store(DW_TS_TERM * n_slots + t, UInt32(0))
        state.unsafe_store(DW_TS_DEF * n_slots + t, UInt32(0))
        state.unsafe_store(
            DW_TS_HT * n_slots + t,
            DW_TS_HT_CURRENT if t == 0 else DW_TS_HT_ZEROES,
        )
        state.unsafe_store(
            DW_TS_DIRTY * n_slots + t, UInt32(1) if t == 0 else UInt32(0)
        )
    if t < max_leaves:
        for l in range(DW_TS_LISTS):
            var v = dummy
            if t == 0 and (l == DW_TS_L_BUILD or l == DW_TS_L_VISIT):
                v = UInt32(0)
            lists0.unsafe_store(l * max_leaves + t, v)
    if t == 0:
        counts.unsafe_store(DW_TS_C_LEAVES, UInt32(1))
        counts.unsafe_store(DW_TS_C_VISIT, UInt32(1))
        counts.unsafe_store(DW_TS_C_SPLIT, UInt32(0))


def dw_ts_select_kernel(
    winner: MutPointer[UInt32, MutAnyOrigin],
    visit: MutPointer[UInt32, MutAnyOrigin],
    counts: MutPointer[UInt32, MutAnyOrigin],
    feat_table: MutPointer[UInt32, MutAnyOrigin],
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    split_features: MutPointer[UInt32, MutAnyOrigin],
    split_bins: MutPointer[UInt32, MutAnyOrigin],
    win_cells: MutPointer[UInt32, MutAnyOrigin],
    state: MutPointer[UInt32, MutAnyOrigin],
    n_slots_in: Int32,
    mark_undefined_terminal_in: Int32,
    split_recs: MutPointer[UInt32, MutAnyOrigin],
):
    """`dw_select_splits_kernel` with the visit count and the leaf count
    read from this level's counters, plus the host's two record-driven
    marks: `best_split.defined` for every scored leaf, and (their
    `min_child_hessian >= 0` rule) a scored leaf left undefined is
    terminal. One thread per visit slot; the grid is the level's cap and
    slots past the count return. The last scored slot writes the split
    count. Same selection, same payload, same slot order (ascending id)
    as the host's `select_leaves_to_split` + `MakeSplit`. `split_recs` is
    this level's split record slice the host builds the tree from after
    the one wait: `DW_TS_SPLIT_WORDS` words per split slot (the left
    leaf, the feature, the bin)."""
    var n_visit = Int(counts.unsafe_load(DW_TS_C_VISIT))
    var leaves_count = Int(counts.unsafe_load(DW_TS_C_LEAVES))
    var n_slots = Int(n_slots_in)
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if v >= n_visit:
        return
    var leaf = Int(visit.unsafe_load(v))
    var base = v * DW_WINNER_WORDS
    var defined = winner.unsafe_load(base + 3) == DW_WINNER_DEFINED
    state.unsafe_store(
        DW_TS_DEF * n_slots + leaf, UInt32(1) if defined else UInt32(0)
    )
    if not defined and mark_undefined_terminal_in != Int32(0):
        state.unsafe_store(DW_TS_TERM * n_slots + leaf, UInt32(1))
    var selected = False
    if defined:
        selected = bitcast[DType.float32](
            winner.unsafe_load(base + 2)
        ) < Float32(0.0)
    var rank = 0
    for u in range(v):
        var ub = u * DW_WINNER_WORDS
        if winner.unsafe_load(ub + 3) == DW_WINNER_DEFINED:
            if bitcast[DType.float32](winner.unsafe_load(ub + 2)) < Float32(
                0.0
            ):
                rank += 1
    if selected:
        left_leaves.unsafe_store(rank, UInt32(leaf))
        right_leaves.unsafe_store(rank, UInt32(leaves_count + rank))
        split_bins.unsafe_store(rank, winner.unsafe_load(base + 1))
        win_cells.unsafe_store(rank, winner.unsafe_load(base + 4))
        var feat = Int(winner.unsafe_load(base))
        for w in range(DW_FEAT_WORDS):
            split_features.unsafe_store(
                rank * DW_FEAT_WORDS + w,
                feat_table.unsafe_load(feat * DW_FEAT_WORDS + w),
            )
        split_recs.unsafe_store(rank * DW_TS_SPLIT_WORDS, UInt32(leaf))
        split_recs.unsafe_store(
            rank * DW_TS_SPLIT_WORDS + 1, winner.unsafe_load(base)
        )
        split_recs.unsafe_store(
            rank * DW_TS_SPLIT_WORDS + 2, winner.unsafe_load(base + 1)
        )
    if v == n_visit - 1:
        counts.unsafe_store(
            DW_TS_C_SPLIT, UInt32(rank + (1 if selected else 0))
        )


def dw_ts_split_state_kernel(
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    counts: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    state: MutPointer[UInt32, MutAnyOrigin],
    n_slots_in: Int32,
    min_leaf_rows_in: Int32,
    max_depth_in: Int32,
):
    """The host's `MakeSplit` + `RebuildLeavesSizes` + `MarkTerminal`
    bookkeeping for one split pair, after the fused chain wrote the two
    children's sizes. `split_leaf`: depth + 1, best split reset, the
    children of a `CurrentPath` parent are `PreviousPath` (else `Zeroes`);
    the right child's fresh slot is clean (DEVIATION 1903).
    `is_terminal_leaf`: `size <= min_leaf_size` (the host compares the
    integer size against a Float64, which is `size <= floor(min_leaf_size)`;
    `min_leaf_rows` is that floor, -1 when the test never fires) or
    `depth >= max_depth`. One thread per split slot; slots past the
    device count return."""
    var n_split = Int(counts.unsafe_load(DW_TS_C_SPLIT))
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= n_split:
        return
    var n_slots = Int(n_slots_in)
    var min_rows = Int(min_leaf_rows_in)
    var max_depth = Int(max_depth_in)
    var l = Int(left_leaves.unsafe_load(r))
    var rr = Int(right_leaves.unsafe_load(r))
    var d = Int(state.unsafe_load(DW_TS_DEPTH * n_slots + l)) + 1
    var parent_ht = state.unsafe_load(DW_TS_HT * n_slots + l)
    var child_ht = DW_TS_HT_ZEROES
    if parent_ht == DW_TS_HT_CURRENT:
        child_ht = DW_TS_HT_PREVIOUS
    var sl = Int(part_size.unsafe_load(l))
    var sr = Int(part_size.unsafe_load(rr))
    var tl = sl <= min_rows or d >= max_depth
    var tr = sr <= min_rows or d >= max_depth
    state.unsafe_store(DW_TS_DEPTH * n_slots + l, UInt32(d))
    state.unsafe_store(DW_TS_DEPTH * n_slots + rr, UInt32(d))
    state.unsafe_store(
        DW_TS_TERM * n_slots + l, UInt32(1) if tl else UInt32(0)
    )
    state.unsafe_store(
        DW_TS_TERM * n_slots + rr, UInt32(1) if tr else UInt32(0)
    )
    state.unsafe_store(DW_TS_DEF * n_slots + l, UInt32(0))
    state.unsafe_store(DW_TS_DEF * n_slots + rr, UInt32(0))
    state.unsafe_store(DW_TS_HT * n_slots + l, child_ht)
    state.unsafe_store(DW_TS_HT * n_slots + rr, child_ht)
    state.unsafe_store(DW_TS_DIRTY * n_slots + rr, UInt32(0))


def dw_ts_plan_kernel(
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    counts: MutPointer[UInt32, MutAnyOrigin],
    counts_next: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    state: MutPointer[UInt32, MutAnyOrigin],
    n_slots_in: Int32,
    lists_next: MutPointer[UInt32, MutAnyOrigin],
    max_leaves_in: Int32,
    cap_pairs_in: Int32,
):
    """`build_necessary_histograms` + `non_zero_leaves` + the plan-time
    staging of the depthwise driver, for the NEXT level, one thread per
    split pair of this level (both children are `PreviousPath`: the host
    plan's `rebuildLeaves` groups exactly these pairs by parent, in
    ascending left-id order, which is split-slot order). The rule
    (`split_properties_helper.cpp:1318-1334`): `small = right`, and
    `small = left` only if `size[left] < size[right]` (strict; a tie
    computes the right child); both terminal -> nothing. Then the level's
    lists: BUILD = small if its size is not 0; ZERO = small if its slot
    is dirty (DEVIATION 1903); SUBTRACT `big -= small`; the deferred
    parent-histogram COPY `small -> big` when big is the RIGHT child
    (`subtract_from > subtract_what`). Then their `allUpdatedLeaves`:
    both are `CurrentPath`, best split reset, slots dirty. Pair slots past
    the split count, up to the cap, hold dummies. Thread 0 writes the next
    level's leaf count."""
    var n_split = Int(counts.unsafe_load(DW_TS_C_SPLIT))
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(cap_pairs_in):
        return
    var n_slots = Int(n_slots_in)
    var max_leaves = Int(max_leaves_in)
    var dummy = UInt32(n_slots - 1)
    if r == 0:
        counts_next.unsafe_store(
            DW_TS_C_LEAVES,
            counts.unsafe_load(DW_TS_C_LEAVES) + UInt32(n_split),
        )
    var copy_src = dummy
    var copy_dst = dummy
    var zero = dummy
    var build = dummy
    var sub_from = dummy
    var sub_what = dummy
    if r < n_split:
        var l = Int(left_leaves.unsafe_load(r))
        var rr = Int(right_leaves.unsafe_load(r))
        var small = rr
        var big = l
        if part_size.unsafe_load(l) < part_size.unsafe_load(rr):
            small = l
            big = rr
        var both_terminal = (
            state.unsafe_load(DW_TS_TERM * n_slots + small) != UInt32(0)
            and state.unsafe_load(DW_TS_TERM * n_slots + big) != UInt32(0)
        )
        if not both_terminal:
            if part_size.unsafe_load(small) != UInt32(0):
                build = UInt32(small)
            if state.unsafe_load(DW_TS_DIRTY * n_slots + small) != UInt32(0):
                zero = UInt32(small)
            sub_from = UInt32(big)
            sub_what = UInt32(small)
            if big > small:
                copy_src = UInt32(small)
                copy_dst = UInt32(big)
            state.unsafe_store(DW_TS_DIRTY * n_slots + small, UInt32(1))
            state.unsafe_store(DW_TS_DIRTY * n_slots + big, UInt32(1))
            state.unsafe_store(DW_TS_DEF * n_slots + small, UInt32(0))
            state.unsafe_store(DW_TS_DEF * n_slots + big, UInt32(0))
            state.unsafe_store(DW_TS_HT * n_slots + small, DW_TS_HT_CURRENT)
            state.unsafe_store(DW_TS_HT * n_slots + big, DW_TS_HT_CURRENT)
    lists_next.unsafe_store(DW_TS_L_COPY_SRC * max_leaves + r, copy_src)
    lists_next.unsafe_store(DW_TS_L_COPY_DST * max_leaves + r, copy_dst)
    lists_next.unsafe_store(DW_TS_L_ZERO * max_leaves + r, zero)
    lists_next.unsafe_store(DW_TS_L_BUILD * max_leaves + r, build)
    lists_next.unsafe_store(DW_TS_L_SUB_FROM * max_leaves + r, sub_from)
    lists_next.unsafe_store(DW_TS_L_SUB_WHAT * max_leaves + r, sub_what)


def dw_ts_visit_kernel(
    counts: MutPointer[UInt32, MutAnyOrigin],
    counts_next: MutPointer[UInt32, MutAnyOrigin],
    state: MutPointer[UInt32, MutAnyOrigin],
    n_slots_in: Int32,
    visit_next: MutPointer[UInt32, MutAnyOrigin],
    cap_visit_in: Int32,
    max_leaves_opt_in: Int32,
):
    """`should_terminate` + `select_leaves_to_visit` for the NEXT level,
    after `dw_ts_plan_kernel`: the visit list is every leaf that is not
    terminal and has no best split, ascending by id, compacted; the rest
    of the cap holds dummies. Nothing is visited once the tree has
    stopped: `leaves >= max_leaves` (their `ShouldTerminate`), or this
    level split nothing (the host then marks every leaf terminal). One
    thread per visit slot: thread `t` ranks leaf `t` among the visitable
    leaves below it (every thread scans the same `leaves` flags, so the
    rank and the total agree with no reduction), writes it at its rank,
    and writes a dummy at slot `t` when `t` is past the total -- every
    slot of the cap is written exactly once. Thread 0 writes the next
    level's visit count and clears its split count."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(cap_visit_in):
        return
    var n_slots = Int(n_slots_in)
    var leaves = Int(counts_next.unsafe_load(DW_TS_C_LEAVES))
    var n_split = Int(counts.unsafe_load(DW_TS_C_SPLIT))
    var stop = leaves >= Int(max_leaves_opt_in) or n_split == 0
    var rank = 0
    var total = 0
    var mine = False
    if not stop:
        for id in range(leaves):
            var vis = (
                state.unsafe_load(DW_TS_TERM * n_slots + id) == UInt32(0)
                and state.unsafe_load(DW_TS_DEF * n_slots + id) == UInt32(0)
            )
            if vis:
                if id < t:
                    rank += 1
                if id == t:
                    mine = True
                total += 1
    if mine:
        visit_next.unsafe_store(rank, UInt32(t))
    if t >= total:
        visit_next.unsafe_store(t, UInt32(n_slots - 1))
    if t == 0:
        counts_next.unsafe_store(DW_TS_C_VISIT, UInt32(total))
        counts_next.unsafe_store(DW_TS_C_SPLIT, UInt32(0))


def dw_ts_snapshot_sizes_kernel(
    part_size: MutPointer[UInt32, MutAnyOrigin],
    sizes_out: MutPointer[UInt32, MutAnyOrigin],
    n_slots_in: Int32,
):
    """The partition sizes after a level's split, into that level's slice
    (the host's `RebuildLeavesSizes` reads them in the replay). One thread
    per slot."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(n_slots_in):
        return
    sizes_out.unsafe_store(t, part_size.unsafe_load(t))
