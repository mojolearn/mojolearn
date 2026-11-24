# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ExtraTrees forest level loop's control plane ON THE DEVICE (cpu4-forest).

NOT IN THEIR SOURCE. cuML's `Builder::train` (`builder.cuh:375-389`) pops a
batch on the host, drains the stream to read the splits, and pushes the
children on the host (`NodeQueue::Push`, `:91-143`). The merged-frontier ET
forest (DEVIATION 211) did the same over one host queue per in-flight tree.
These kernels keep the whole frontier on the device, so the host enqueues K
batches per drain and reads back one fixed-size header, never a node list.
The RandomForest form (`ensemble/.../kernels/level_loop_kernels.mojo`,
`IDN_RF_DEVICE_LOOP`) lost its A/Bs and was deleted 2026-11-24; this was its
forest-batched ExtraTrees counterpart.

ONE FIFO PER GROUP. The group's trees share one device FIFO of work items
(six Int32 each: `[idx, depth, begin, count, slot, gpos]`). The host
form kept one queue per tree and filled a batch queue by queue; a batch of
the shared FIFO holds the same items in a different MIX. The mix is a
scheduling parameter that cannot move a tree (DEVIATION 211's argument:
every draw is keyed by `(seed, tree, node)`, every score cell and range is
the node's own, and the partition is range-addressed), and what CAN move a
tree is preserved exactly:
  * a tree's items leave the FIFO in the order that tree pushed them (the
    shared FIFO's per-tree subsequence IS that tree's own FIFO), and
  * a tree's node ids are assigned in that order: `left = n_nodes(tree) +
    2 * rank`, rank = valid splits of the SAME tree earlier in the batch
    (the push kernels), exactly `NodeQueue.push`'s `len(sparsetree)`.
So node ids, ranges, draws and the tree are the host queue's, bit for bit,
and the host column (`train_tree_exact`, which pushes per tree in the same
FIFO order) needs no change.

Since 2026-11-24 the FIFO pop and push kernels are deleted with the
depth-wise device loop (it lost two A/Bs; depth-wise growth runs the host
queue): the device loop runs best-first growth only, whose frontier kernels
live in the builder. The ordering argument above is the one they keep. The
device FIFO itself (its buffer, the init kernel's seed, the header's head and
tail words and the host's re-pack) went with it the same day (lane
ab-outcome-3, o3-trees): nothing read it after the depth-wise loop.

THE HOST DOES NOT KNOW THE LIVE COUNT. Every per-batch kernel is launched at
a bound the host proves from the last header (`n_bound` items, a block-map
bound); the pop CLAMPS the batch to both bounds, so a bound only ever costs
speed, never correctness. Items `[cur, n_bound)` are DUMMIES (count 0) that
own no map entry and whose split slot is stamped invalid; map entries past
the live total carry `nodeid == -1`, which every map-driven kernel leaves on.

THE NODE ARENA. Node records live in one group-wide arena in push order
(`g_nodes`: the `SparseTreeNode` itself; `g_meta`: `[slot, local id, range
begin, range count]`). At the end of the group `etl_scatter_kernel` lays the
arena out tree by tree in local-id order -- each tree's `sparsetree` exactly
as `NodeQueue.push` appended it -- for the leaf pass and one download.

PREFIX SUMS are block collectives over integers (`prefix_sum`, `sum`), with
chunk totals carried across chunks by the same block, or a second launch.
Integer adds in any order give one value, so every offset is exact on every
vendor. Atomics are integer only (`fetch_add`, `max`): order-free.

METAL: no int-to-pointer casts; the shared pages are fixed and small and each
kernel that allocates one asserts it fits (`ETL_SHARED_FITS`); cross-thread
data inside one launch goes through threadgroup memory only.
"""

from std.atomic import Atomic
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.primitives.block import sum as block_sum

from extratrees.impl.decisiontree.flatnode import SparseTreeNode
from extratrees.impl.decisiontree.batched_levelalgo.split import (
    Split,
    split_tie_salt_for,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    WorkloadInfo,
)

comptime ETL_TPB = 256
"""Threads per block of every loop kernel."""

comptime ETL_SHARED_FITS = 16384
"""The fits gate: bytes of threadgroup memory any loop kernel may take. Every
GPU this code targets (Apple included) offers at least 32 KiB per group."""

comptime ETL_HDR_WORDS = 16
comptime ETL_META_INTS = 4
comptime ETL_STAT_INTS = 4

# --- header words (`Int32`) ---------------------------------------------------
# Words 0 and 1 are unused: they were the device FIFO's head and tail
# (ETL_H_HEAD, ETL_H_TAIL), deleted 2026-11-24 with the FIFO (lane
# ab-outcome-3, o3-trees); the other words keep their numbers.
comptime ETL_H_NODES = 2
"""Node records used in the group arena."""
comptime ETL_H_CUR = 3
"""Live items of the batch in flight (`[0, cur)` of the batch arrays)."""
comptime ETL_H_NSUB = 4
"""Live items of the rescue sub-batch (DEVIATION 205)."""
comptime ETL_H_OVERFLOW = 5
"""Nonzero when a capacity or a launch bound was exceeded (host raises)."""
comptime ETL_H_ROWS = 6
"""Rows the waiting (best-first) nodes cover: every later batch's rows are a
subset, so it bounds the block maps."""
comptime ETL_H_BLOCKS = 7
"""Live entries of the block map last staged."""
comptime ETL_H_POPS = 8
"""Best-first: trees that popped a node this cycle."""
comptime ETL_H_STAT_NODES = 9
comptime ETL_H_STAT_RETRY = 10
comptime ETL_H_STAT_RESCUED = 11
comptime ETL_H_STAT_BATCHES = 12
comptime ETL_H_GCOUNT = 13
"""The group's tree count, a constant count word for per-slot batches."""
comptime ETL_H_PART_MISMATCH = 14
"""Partitioned nodes whose left-going row count (the partition's own direction
test) differed from the split's `n_left` (the search's count); the host
refuses at the next drain (lane sm120-tree-identity)."""

# --- per-slot words (`Int32`, `ETL_STAT_INTS` per tree) -------------------------
comptime ETL_ST_NODES = 0
"""`len(tree.sparsetree)`."""
comptime ETL_ST_LEAVES = 1
"""`tree.leaf_counter` (starts at 1)."""
comptime ETL_ST_DEPTH = 2
"""`tree.depth_counter`."""
comptime ETL_ST_FRONT = 3
"""Best-first: records on this tree's frontier."""


@always_inline
def etl_dummy_item() -> NodeWorkItem:
    """A dummy batch slot: count 0, so it owns no block and is never split."""
    return NodeWorkItem(Int32(0), Int32(0), InstanceRange(Int32(0), Int32(0)))


@always_inline
def etl_blocks(count: Int, tile: Int) -> Int:
    """`build_workload_info`'s `max(ceildiv(count, tile), 1)`."""
    var nb = (count + tile - 1) // tile
    if nb < 1:
        nb = 1
    return nb


def etl_init_kernel(
    hdr: MutPointer[Int32, MutAnyOrigin],
    g_nodes: MutPointer[SparseTreeNode[DType.float32], MutAnyOrigin],
    g_meta: MutPointer[Int32, MutAnyOrigin],
    slot_stat: MutPointer[Int32, MutAnyOrigin],
    b_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    b_slot: MutPointer[Int32, MutAnyOrigin],
    b_gpos: MutPointer[Int32, MutAnyOrigin],
    g_in: Int32,
    slot_rows: Int32,
    shared_base: Int32,
    root_expandable: Int32,
):
    """`NodeQueue.__init__` for every tree of the group at once (best-first
    growth, the device loop's only form).

    Tree slot `s`: root leaf record at arena slot `s` (local id 0) holding
    `slot_rows` rows from `s * slot_rows` (0 for every tree under
    `FOREST_SAB_SHARED_ROW_BASE`), its per-slot counters, and the first
    search batch's two pending entries of the tree (best-first batches hold
    two entries per tree): entry `2 s` the root's work item when the root is
    expandable (else a dummy), entry `2 s + 1` a dummy. Thread 0 writes the
    header."""
    # TOMBSTONE: the device FIFO seed (the `queue` argument, the `bestfirst`
    # flag and the depth-wise branch that wrote FIFO entry `s` and the tail
    # word) deleted 2026-11-24 by lane/ab-outcome-3 (o3-trees), the leftover of
    # MOJOLEARN_IDN_ET_DEVICE_LOOP_EARLY_EXIT; code recoverable at 9be94fd314.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/ET_DEVICE_LOOP_FIFO.ab-outcome-3.patch
    var s = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var g = Int(g_in)
    var expand = root_expandable != Int32(0)
    if s == 0:
        comptime for w in range(ETL_HDR_WORDS):
            hdr[unsafe_offset=w] = Int32(0)
        hdr[unsafe_offset=ETL_H_NODES] = g_in
        hdr[unsafe_offset=ETL_H_GCOUNT] = g_in
        hdr[unsafe_offset=ETL_H_CUR] = Int32(2) * g_in
        if expand:
            hdr[unsafe_offset=ETL_H_ROWS] = g_in * slot_rows
    if s >= g:
        return
    var base = Int32(0) if shared_base != Int32(0) else Int32(s) * slot_rows
    g_nodes[unsafe_offset=s] = SparseTreeNode[DType.float32].CreateLeafNode(
        slot_rows
    )
    g_meta[unsafe_offset = s * ETL_META_INTS + 0] = Int32(s)
    g_meta[unsafe_offset = s * ETL_META_INTS + 1] = Int32(0)
    g_meta[unsafe_offset = s * ETL_META_INTS + 2] = base
    g_meta[unsafe_offset = s * ETL_META_INTS + 3] = slot_rows
    slot_stat[unsafe_offset = s * ETL_STAT_INTS + ETL_ST_NODES] = Int32(1)
    slot_stat[unsafe_offset = s * ETL_STAT_INTS + ETL_ST_LEAVES] = Int32(1)
    slot_stat[unsafe_offset = s * ETL_STAT_INTS + ETL_ST_DEPTH] = Int32(0)
    slot_stat[unsafe_offset = s * ETL_STAT_INTS + ETL_ST_FRONT] = Int32(0)
    var root = etl_dummy_item()
    var root_gpos = Int32(-1)
    if expand:
        root = NodeWorkItem(
            Int32(0), Int32(0), InstanceRange(base, slot_rows)
        )
        root_gpos = Int32(s)
    b_items[unsafe_offset = 2 * s] = root
    b_slot[unsafe_offset = 2 * s] = Int32(s)
    b_gpos[unsafe_offset = 2 * s] = root_gpos
    b_items[unsafe_offset = 2 * s + 1] = etl_dummy_item()
    b_slot[unsafe_offset = 2 * s + 1] = Int32(s)
    b_gpos[unsafe_offset = 2 * s + 1] = Int32(-1)


# TOMBSTONE: MOJOLEARN_IDN_ET_DEVICE_LOOP_EARLY_EXIT (NOISE: NV L40S 6 cells, median 1.00, 0.99..1.06) deleted 2026-11-24 by lane/ab-outcome-2; code recoverable at ba83d4655f.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_ET_DEVICE_LOOP_EARLY_EXIT.ab-outcome-2.patch; record in docs/TOMBSTONES.md.
# (`etl_pop_kernel`: the FIFO pop of the depth-wise device loop, which only that loop ran.)


def etl_stage_kernel[
    TPB: Int
](
    hdr: MutPointer[Int32, MutAnyOrigin],
    count_word: Int32,
    src_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    src_slot: MutPointer[Int32, MutAnyOrigin],
    slot_tree: MutPointer[Int32, MutAnyOrigin],
    d_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    d_tree: MutPointer[Int32, MutAnyOrigin],
    d_tsalt: MutPointer[UInt32, MutAnyOrigin],
    d_nb: MutPointer[Int32, MutAnyOrigin],
    d_nc: MutPointer[Int32, MutAnyOrigin],
    d_blk_base: MutPointer[Int32, MutAnyOrigin],
    x_off: MutPointer[Int32, MutAnyOrigin],
    x_nb: MutPointer[Int32, MutAnyOrigin],
    x_large: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
    k: Int32,
    tile: Int32,
    part_tile: Int32,
    blocks_bound: Int32,
    scalar_tree: Int32,
    search_phase: Int32,
):
    """`stage_batch` on the device. ONE block.

    For `j < n_bound`: the work item (a dummy past the live count
    `hdr[count_word]`), its tree id (`slot_tree[slot]`; under
    `FOREST_SAB_SCALAR_TREE` the first item's for every item), its tie salt
    (`split_tie_salt_for`, the function the host staging calls), its cell
    base and count (`j * k`, `k`), its partition block base (tile
    `part_tile`, as `stage_batch` always computes it), and the block map's
    inputs for tile `tile`: offset, block count, inclusive large-node count
    (`build_workload_info`'s `n_large_nodes`). A zero-count item owns no
    block; every live item owns `max(ceildiv(count, tile), 1)`."""
    comptime assert 4 <= ETL_SHARED_FITS, "stage header page"
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        1, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()
    if tid == 0:
        sh[unsafe_offset=0] = hdr[unsafe_offset=Int(count_word)]
    barrier()
    var n = Int(sh[unsafe_offset=0])
    var tree0 = Int32(0)
    if scalar_tree != Int32(0) and n > 0:
        tree0 = slot_tree[unsafe_offset=Int(src_slot[unsafe_offset=0])]
    var run_nb = 0
    var run_large = 0
    var run_pb = 0
    var c = 0
    while c < Int(n_bound):
        var j = c + tid
        var inb = j < Int(n_bound)
        var item = etl_dummy_item()
        var slot = 0
        if inb and j < n:
            item = src_items[unsafe_offset=j]
            slot = Int(src_slot[unsafe_offset=j])
        var count = Int(item.instances.count)
        var nb = Int32(0)
        var large = Int32(0)
        var pb = Int32(0)
        if count > 0:
            nb = Int32(etl_blocks(count, Int(tile)))
            pb = Int32(etl_blocks(count, Int(part_tile)))
            if nb > Int32(1):
                large = Int32(1)
        var ex_nb = block_prefix_sum[block_size=TPB, exclusive=True](nb)
        var ex_large = block_prefix_sum[block_size=TPB, exclusive=True](large)
        var ex_pb = block_prefix_sum[block_size=TPB, exclusive=True](pb)
        var tot_nb = block_sum[block_size=TPB, broadcast=True](nb)
        var tot_large = block_sum[block_size=TPB, broadcast=True](large)
        var tot_pb = block_sum[block_size=TPB, broadcast=True](pb)
        if inb:
            var tree = slot_tree[unsafe_offset=slot]
            if scalar_tree != Int32(0):
                tree = tree0
            d_items[unsafe_offset=j] = item
            d_tree[unsafe_offset=j] = tree
            d_tsalt[unsafe_offset=j] = split_tie_salt_for(
                UInt32(Int(tree)), UInt32(Int(item.idx))
            )
            d_nb[unsafe_offset=j] = Int32(j * Int(k))
            d_nc[unsafe_offset=j] = k
            d_blk_base[unsafe_offset=j] = Int32(run_pb) + ex_pb
            x_off[unsafe_offset=j] = Int32(run_nb) + ex_nb
            x_nb[unsafe_offset=j] = nb
            x_large[unsafe_offset=j] = Int32(run_large) + ex_large + large
        run_nb += Int(tot_nb)
        run_large += Int(tot_large)
        run_pb += Int(tot_pb)
        c += TPB
    if tid == 0:
        hdr[unsafe_offset=ETL_H_BLOCKS] = Int32(run_nb)
        if run_nb > Int(blocks_bound):
            hdr[unsafe_offset=ETL_H_OVERFLOW] = Int32(2)


def etl_map_kernel(
    hdr: MutPointer[Int32, MutAnyOrigin],
    x_off: MutPointer[Int32, MutAnyOrigin],
    x_nb: MutPointer[Int32, MutAnyOrigin],
    x_large: MutPointer[Int32, MutAnyOrigin],
    d_wl: MutPointer[WorkloadInfo, MutAnyOrigin],
    n_bound: Int32,
    blocks_bound: Int32,
):
    """`build_workload_info`'s block map, one thread per entry. Entry `b`
    belongs to the LAST item whose offset is `<= b` (offsets ascend; a
    zero-block item shares the next item's offset, so it is never the last
    one at or below a live entry). Entries past the live total carry
    `nodeid == -1`, which every map-driven kernel leaves on. Overflowing
    entries (a host bound bug, flagged by the stage kernel) are clipped."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if b >= Int(blocks_bound):
        return
    var total = Int(hdr[unsafe_offset=ETL_H_BLOCKS])
    if b >= total or Int(n_bound) < 1:
        d_wl[unsafe_offset=b] = WorkloadInfo(
            Int32(-1), Int32(-1), Int32(0), Int32(0)
        )
        return
    var lo = 0
    var hi = Int(n_bound) - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if Int(x_off[unsafe_offset=mid]) <= b:
            lo = mid
        else:
            hi = mid - 1
    d_wl[unsafe_offset=b] = WorkloadInfo(
        Int32(lo),
        x_large[unsafe_offset=lo] - Int32(1),
        Int32(b - Int(x_off[unsafe_offset=lo])),
        x_nb[unsafe_offset=lo],
    )


def etl_copy_splits_kernel(
    dst: MutPointer[Split, MutAnyOrigin],
    src: MutPointer[Split, MutAnyOrigin],
    n: Int32,
):
    """`dst[i] = src[i]` for `i < n`: keeps the batch's splits while the
    rescue reuses the search workspace."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


def etl_retry_kernel[
    TPB: Int
](
    hdr: MutPointer[Int32, MutAnyOrigin],
    b_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    b_slot: MutPointer[Int32, MutAnyOrigin],
    b_splits: MutPointer[Split, MutAnyOrigin],
    nonconst: MutPointer[Int32, MutAnyOrigin],
    s_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    s_slot: MutPointer[Int32, MutAnyOrigin],
    s_idx: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
):
    """DEVIATION 205's retry list, compacted on the device. ONE block.

    The host loop's `retry = [i : any_nonconst[i] == 0 and count > 0]` in
    batch order, written as the sub-batch `s_*` (dummies past its length)
    with `s_idx` mapping each back to its batch slot. Batch slots at or past
    the live count get the invalid `Split()`, so no later kernel splits a
    dummy."""
    comptime assert 4 <= ETL_SHARED_FITS, "retry header page"
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        1, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()
    if tid == 0:
        sh[unsafe_offset=0] = hdr[unsafe_offset=ETL_H_CUR]
    barrier()
    var cur = Int(sh[unsafe_offset=0])
    var run = 0
    var c = 0
    while c < Int(n_bound):
        var j = c + tid
        var flag = Int32(0)
        if j < cur and j < Int(n_bound):
            if (
                nonconst[unsafe_offset=j] == Int32(0)
                and b_items[unsafe_offset=j].instances.count > Int32(0)
            ):
                flag = Int32(1)
        elif j < Int(n_bound):
            b_splits[unsafe_offset=j] = Split()
        var ex = Int(block_prefix_sum[block_size=TPB, exclusive=True](flag))
        var tot = Int(block_sum[block_size=TPB, broadcast=True](flag))
        if flag != Int32(0):
            var r = run + ex
            s_items[unsafe_offset=r] = b_items[unsafe_offset=j]
            s_slot[unsafe_offset=r] = b_slot[unsafe_offset=j]
            s_idx[unsafe_offset=r] = Int32(j)
        run += tot
        c += TPB
    var r2 = run + tid
    while r2 < Int(n_bound):
        s_items[unsafe_offset=r2] = etl_dummy_item()
        s_slot[unsafe_offset=r2] = Int32(0)
        s_idx[unsafe_offset=r2] = Int32(-1)
        r2 += TPB
    if tid == 0:
        hdr[unsafe_offset=ETL_H_NSUB] = Int32(run)
        hdr[unsafe_offset=ETL_H_STAT_RETRY] = hdr[
            unsafe_offset=ETL_H_STAT_RETRY
        ] + Int32(run)


def etl_merge_kernel(
    hdr: MutPointer[Int32, MutAnyOrigin],
    b_splits: MutPointer[Split, MutAnyOrigin],
    sub_splits: MutPointer[Split, MutAnyOrigin],
    pick: MutPointer[Int32, MutAnyOrigin],
    s_idx: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
):
    """`splits[retry[j]] = rescued[j]` where the pick found a column
    (`h_pick[j] >= 0`), one thread per sub-batch slot."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n_bound):
        return
    if r >= Int(hdr[unsafe_offset=ETL_H_NSUB]):
        return
    if pick[unsafe_offset=r] < Int32(0):
        return
    b_splits[unsafe_offset=Int(s_idx[unsafe_offset=r])] = sub_splits[
        unsafe_offset=r
    ]
    _ = Atomic.fetch_add(hdr.unsafe_offset(ETL_H_STAT_RESCUED), Int32(1))


# TOMBSTONE: MOJOLEARN_IDN_ET_DEVICE_LOOP_EARLY_EXIT (NOISE: NV L40S 6 cells, median 1.00, 0.99..1.06) deleted 2026-11-24 by lane/ab-outcome-2; code recoverable at ba83d4655f.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_ET_DEVICE_LOOP_EARLY_EXIT.ab-outcome-2.patch; record in docs/TOMBSTONES.md.
# (`etl_push_rank_kernel`, `etl_push_slot_kernel`, `etl_push_mark_kernel`,
# `etl_push_commit_kernel`, `etl_push_write_kernel`: the FIFO push of the
# depth-wise device loop, which only that loop ran.)


def etl_tree_base_kernel[
    TPB: Int
](
    slot_stat: MutPointer[Int32, MutAnyOrigin],
    tree_base: MutPointer[Int32, MutAnyOrigin],
    g_in: Int32,
):
    """Each tree's first node in the concatenated output (the exclusive
    prefix of `n_nodes` over the group's trees), and the total in
    `tree_base[g]`. ONE block, integer scan."""
    var tid = Int(thread_idx.x)
    var g = Int(g_in)
    var run = 0
    var c = 0
    while c < g:
        var s = c + tid
        var v = Int32(0)
        if s < g:
            v = slot_stat[unsafe_offset = s * ETL_STAT_INTS + ETL_ST_NODES]
        var ex = Int(block_prefix_sum[block_size=TPB, exclusive=True](v))
        var tot = Int(block_sum[block_size=TPB, broadcast=True](v))
        if s < g:
            tree_base[unsafe_offset=s] = Int32(run + ex)
        run += tot
        c += TPB
    if tid == 0:
        tree_base[unsafe_offset=g] = Int32(run)


def etl_scatter_kernel(
    g_nodes: MutPointer[SparseTreeNode[DType.float32], MutAnyOrigin],
    g_meta: MutPointer[Int32, MutAnyOrigin],
    tree_base: MutPointer[Int32, MutAnyOrigin],
    out_nodes: MutPointer[SparseTreeNode[DType.float32], MutAnyOrigin],
    out_ranges: MutPointer[InstanceRange, MutAnyOrigin],
    n_total: Int32,
):
    """The arena laid out tree by tree, each tree in local-id order: node
    `p` of tree slot `s` with local id `l` lands at `tree_base[s] + l`, its
    range beside it -- the `sparsetree` / `node_instances` pair the host
    queue built, concatenated as the leaf pass reads them."""
    var p = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if p >= Int(n_total):
        return
    var s = Int(g_meta[unsafe_offset = p * ETL_META_INTS + 0])
    var l = Int(g_meta[unsafe_offset = p * ETL_META_INTS + 1])
    var t = Int(tree_base[unsafe_offset=s]) + l
    out_nodes[unsafe_offset=t] = g_nodes[unsafe_offset=p]
    out_ranges[unsafe_offset=t] = InstanceRange(
        g_meta[unsafe_offset = p * ETL_META_INTS + 2],
        g_meta[unsafe_offset = p * ETL_META_INTS + 3],
    )
