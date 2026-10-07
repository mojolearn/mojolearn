# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Bitvector (QuickScorer / RapidScorer style) apply of a NON-SYMMETRIC
ensemble, the Depthwise and Lossguide predict path.

`-D MOJOLEARN_GBDT_NS_PREDICT_BITVEC` (default off; lane/trees-predict-ideas,
2026-10-07). Off, `add_non_symmetric_trees_packed` launches two kernels per
tree (bins, then `add_bin_model_value_kernel`) and round-trips an `n_rows`
bins buffer through global memory per tree. On, one kernel per bounded tree
chunk walks rows x trees:

  * every internal node carries a leaf bitvector of `W` 64-bit words with
    ZEROS over the leaves of its not-split (left) subtree; a row ANDs in the
    vector of every node whose split predicate is true, and its exit leaf is
    the LOWEST set bit (QuickScorer, Lucchese et al. SIGIR 2015). Exact: the
    exit leaf of the incumbent walk survives every AND (no true node on its
    path holds it on the left; no node off its path holds it at all) and
    every lower leaf is cleared at the node where its path and the exit
    path diverge;
  * the nodes of a tree are evaluated in compressed-index word order (the
    feature-major order of QuickScorer), so one load of a cindex word serves
    every node of the tree that reads it; the loads are coalesced across a
    warp because every thread reads the same node at the same time;
  * the node order, the masks and the leaf ranges are computed ON THE DEVICE
    (`_bv_prep_kernel`, one block per tree, one thread per node), not on the
    host.

NO BIT MOVES. Each row's cursor takes the same float32 leaf values in the
same tree order as the incumbent: the register accumulator rounds to float32
on every add exactly as the incumbent's load-add-store does, and chunks run
in tree order on one stream. The host column is untouched.

ELIGIBILITY (cost and hardware reasoning, not a board shape): a tree needs
`ceil(leaves / 64)` mask words, held in registers per thread; the word count
is rounded up to {1, 2, 4, 8} and capped at `BV_MAX_WORDS = 8` (512 leaves,
16 32-bit registers of mask state), past which register pressure would
spill and the per-row all-node evaluation costs more than a walk. An
ensemble whose widest tree exceeds the cap, or whose trees disagree on the
approx dimension, takes the incumbent packed apply.

Exclusive with `MOJOLEARN_TREES_C50_GB_PACKED` (both replace the same
non-symmetric apply); the caller refuses the pair at compile time.
NOT TESTED ON A DEVICE — IDENTITY NOT VERIFIED — NOT MEASURED (queued).
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.bit import count_trailing_zeros
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined

#: The switch. Default off: absent define, incumbent path.
comptime GBDT_NS_BITVEC = is_defined["MOJOLEARN_GBDT_NS_PREDICT_BITVEC"]()

#: Mask words per thread at most (see the module docstring's eligibility).
comptime BV_MAX_WORDS = 8
#: One prep block per tree, one thread per internal node: a tree within the
#: word cap has at most 64 * BV_MAX_WORDS - 1 internal nodes.
comptime BV_PREP_BLOCK = 64 * BV_MAX_WORDS
#: Rows per block of the apply kernel.
comptime BV_ROWS_BLOCK = 128
#: Internal nodes evaluated per thread per launch. Bounds a launch's serial
#: per-thread work so no single launch runs long on any vendor (macOS aborts
#: multi-second Metal command buffers); a single wider tree stays whole.
comptime BV_LAUNCH_NODES = 16384


def bv_words_for(max_leaves: Int) -> Int:
    """Mask words for a tree of `max_leaves` leaves, rounded up to a power of
    two in {1, 2, 4, 8}; 0 when past the cap (ineligible)."""
    var w = 1
    while w * 64 < max_leaves:
        w *= 2
    return w if w <= BV_MAX_WORDS else 0


def _bv_prep_kernel[W: Int](
    d_off: MutPointer[UInt32, MutAnyOrigin],
    d_mask: MutPointer[UInt32, MutAnyOrigin],
    d_shift: MutPointer[UInt32, MutAnyOrigin],
    d_oh: MutPointer[UInt8, MutAnyOrigin],
    d_bin: MutPointer[UInt32, MutAnyOrigin],
    d_ls: MutPointer[UInt32, MutAnyOrigin],
    meta: MutPointer[Int32, MutAnyOrigin],
    rec: MutPointer[UInt32, MutAnyOrigin],
    masks: MutPointer[UInt64, MutAnyOrigin],
):
    """Tree `block_idx.x`, node `thread_idx.x` (preorder index `i`).

    Rank: the node's position in (cindex offset, preorder index) order, a
    total order, so the scatter is a permutation. Leaf range: walk from the
    root; at node `c` with `L = left_subtree[c]` leaves on its not-split
    side, the not-split child's internal nodes are `c+1 .. c+L-1` and the
    split child is `c+L` with its leaves starting `L` further right (the
    incumbent walk's `bin += left_subtree; node += left_subtree`). Node `i`'s
    not-split leaves are then `[b, b + left_subtree[i])`.
    """
    var t = Int(block_idx.x)
    var i = Int(thread_idx.x)
    var base = Int(meta.unsafe_load(4 * t))
    var n = Int(meta.unsafe_load(4 * t + 2))
    if i >= n:
        return
    var my_off = d_off.unsafe_load(base + i)
    var rank = 0
    for j in range(n):
        var o = d_off.unsafe_load(base + j)
        if o < my_off or (o == my_off and j < i):
            rank += 1
    var c = 0
    var b = 0
    var guard = 0
    while c != i and guard < n:
        var l = Int(d_ls.unsafe_load(base + c))
        if i < c + l:
            c += 1
        else:
            b += l
            c += l
        guard += 1
    var lo = b
    var hi = b + Int(d_ls.unsafe_load(base + i))
    var at = base + rank
    rec.unsafe_store(4 * at, my_off)
    rec.unsafe_store(4 * at + 1, d_mask.unsafe_load(base + i))
    var eq = UInt32(1) if d_oh.unsafe_load(base + i) != UInt8(0) else UInt32(0)
    rec.unsafe_store(4 * at + 2, (d_shift.unsafe_load(base + i) & 0xFFFF) | (eq << 16))
    rec.unsafe_store(4 * at + 3, d_bin.unsafe_load(base + i))
    comptime for w in range(W):
        var word = ~UInt64(0)
        var w_lo = w * 64
        for bit in range(64):
            var leaf = w_lo + bit
            if leaf >= lo and leaf < hi:
                word &= ~(UInt64(1) << UInt64(bit))
        masks.unsafe_store(W * at + w, word)


def _bv_rows_kernel[W: Int](
    rec: MutPointer[UInt32, MutAnyOrigin],
    masks: MutPointer[UInt64, MutAnyOrigin],
    meta: MutPointer[Int32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin],
    cursor: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    dim_in: Int32,
    first: Int32,
    last: Int32,
):
    """One thread per row over trees `[first, last)` in tree order."""
    var rows = Int(rows_in)
    var dim = Int(dim_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= rows:
        return
    var acc = cursor.unsafe_load(row)
    for t in range(Int(first), Int(last)):
        var base = Int(meta.unsafe_load(4 * t))
        var val_at = Int(meta.unsafe_load(4 * t + 1))
        var n = Int(meta.unsafe_load(4 * t + 2))
        var m = InlineArray[UInt64, W](fill=~UInt64(0))
        var last_off = UInt32(0xFFFFFFFF)
        var word = UInt32(0)
        var have = False
        for k in range(n):
            var r = rec.unsafe_load[width=4](4 * (base + k))
            if not have or r[0] != last_off:
                word = cindex.unsafe_load(Int(r[0]) + row)
                last_off = r[0]
                have = True
            var v = (word >> (r[2] & 0xFFFF)) & r[1]
            var split: Bool
            if (r[2] >> 16) != 0:
                split = v == r[3]
            else:
                split = v > r[3]
            if split:
                comptime for w in range(W):
                    m[w] &= masks.unsafe_load(W * (base + k) + w)
        var leaf = 0
        comptime for wr in range(W):
            comptime w = W - 1 - wr
            if m[w] != 0:
                leaf = w * 64 + Int(count_trailing_zeros(m[w]))
        if dim == 1:
            acc = acc + values.unsafe_load(val_at + leaf)
        else:
            for d in range(dim):
                var at = d * rows + row
                cursor.unsafe_store(
                    at, cursor.unsafe_load(at) + values.unsafe_load(val_at + leaf * dim + d)
                )
    if dim == 1:
        cursor.unsafe_store(row, acc)


def _bv_launch[W: Int](
    ctx: DeviceContext,
    mut d_off: DeviceBuffer[DType.uint32],
    mut d_mask: DeviceBuffer[DType.uint32],
    mut d_shift: DeviceBuffer[DType.uint32],
    mut d_oh: DeviceBuffer[DType.uint8],
    mut d_bin: DeviceBuffer[DType.uint32],
    mut d_ls: DeviceBuffer[DType.uint32],
    mut d_vals: DeviceBuffer[DType.float32],
    mut dm: DeviceBuffer[DType.int32],
    chunk_first: List[Int],
    total_slots: Int,
    n_trees: Int,
    dim: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    n_rows: Int,
    mut cursor: DeviceBuffer[DType.float32],
) raises:
    var rec = ctx.enqueue_create_buffer[DType.uint32](4 * total_slots)
    var masks = ctx.enqueue_create_buffer[DType.uint64](W * total_slots)
    ctx.enqueue_function[_bv_prep_kernel[W]](
        d_off.unsafe_ptr(), d_mask.unsafe_ptr(), d_shift.unsafe_ptr(),
        d_oh.unsafe_ptr(), d_bin.unsafe_ptr(), d_ls.unsafe_ptr(),
        dm.unsafe_ptr(), rec.unsafe_ptr(), masks.unsafe_ptr(),
        grid_dim=(n_trees, 1, 1), block_dim=(BV_PREP_BLOCK, 1, 1),
    )
    var blocks = max(1, (n_rows + BV_ROWS_BLOCK - 1) // BV_ROWS_BLOCK)
    for c in range(len(chunk_first) - 1):
        ctx.enqueue_function[_bv_rows_kernel[W]](
            rec.unsafe_ptr(), masks.unsafe_ptr(), dm.unsafe_ptr(),
            cindex.unsafe_ptr(), d_vals.unsafe_ptr(), cursor.unsafe_ptr(),
            Int32(n_rows), Int32(dim), Int32(chunk_first[c]),
            Int32(chunk_first[c + 1]),
            grid_dim=(blocks, 1, 1), block_dim=(BV_ROWS_BLOCK, 1, 1),
        )
    ctx.synchronize()
    _ = rec^
    _ = masks^


def bv_apply(
    ctx: DeviceContext,
    words: Int,
    mut d_off: DeviceBuffer[DType.uint32],
    mut d_mask: DeviceBuffer[DType.uint32],
    mut d_shift: DeviceBuffer[DType.uint32],
    mut d_oh: DeviceBuffer[DType.uint8],
    mut d_bin: DeviceBuffer[DType.uint32],
    mut d_ls: DeviceBuffer[DType.uint32],
    mut d_vals: DeviceBuffer[DType.float32],
    node_at: List[Int],
    val_at: List[Int],
    n_nodes: List[Int],
    dim: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    n_rows: Int,
    mut cursor: DeviceBuffer[DType.float32],
) raises:
    """Stage the per-tree metadata, cut the tree chunks and launch the
    `words`-wide kernels. `words` comes from `bv_words_for` (never 0 here).
    Ends with one drain."""
    var n_trees = len(node_at)
    var total_slots = 0
    var hm = ctx.enqueue_create_host_buffer[DType.int32](4 * n_trees)
    var chunk_first = List[Int]()
    chunk_first.append(0)
    var budget = 0
    for t in range(n_trees):  # small-loop(n_trees: trees): stages each tree's slab offsets and cuts launch chunks, model parameters
        hm.unsafe_ptr().unsafe_store(4 * t, Int32(node_at[t]))
        hm.unsafe_ptr().unsafe_store(4 * t + 1, Int32(val_at[t]))
        hm.unsafe_ptr().unsafe_store(4 * t + 2, Int32(n_nodes[t]))
        hm.unsafe_ptr().unsafe_store(4 * t + 3, Int32(dim))
        var cost = max(1, n_nodes[t])
        if t > chunk_first[len(chunk_first) - 1] and budget + cost > BV_LAUNCH_NODES:
            chunk_first.append(t)
            budget = 0
        budget += cost
        total_slots = node_at[t] + max(1, n_nodes[t])
    chunk_first.append(n_trees)
    var dm = ctx.enqueue_create_buffer[DType.int32](4 * n_trees)
    ctx.enqueue_copy(dst_buf=dm, src_ptr=hm.unsafe_ptr())
    if words == 1:
        _bv_launch[1](ctx, d_off, d_mask, d_shift, d_oh, d_bin, d_ls, d_vals, dm,
            chunk_first, total_slots, n_trees, dim, cindex, n_rows, cursor)
    elif words == 2:
        _bv_launch[2](ctx, d_off, d_mask, d_shift, d_oh, d_bin, d_ls, d_vals, dm,
            chunk_first, total_slots, n_trees, dim, cindex, n_rows, cursor)
    elif words == 4:
        _bv_launch[4](ctx, d_off, d_mask, d_shift, d_oh, d_bin, d_ls, d_vals, dm,
            chunk_first, total_slots, n_trees, dim, cindex, n_rows, cursor)
    elif words == 8:
        _bv_launch[8](ctx, d_off, d_mask, d_shift, d_oh, d_bin, d_ls, d_vals, dm,
            chunk_first, total_slots, n_trees, dim, cindex, n_rows, cursor)
    else:
        raise Error("bv_apply: unsupported mask word count " + String(words))
    _ = dm^
    _ = hm^
