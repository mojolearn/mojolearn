# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The RF/DT level loop's control plane ON THE DEVICE (fam2-forests; the
only GPU route since cpu4-forest, 2026-10-04, in IDENTICAL and FAST on every
vendor).

NOT IN THEIR SOURCE. cuML's `Builder::train` (`builder.cuh:375-389`) pops a
batch on the host, drains the stream to read the splits, retries the nodes
that found no split with fresh columns (`:452-458`), and pushes the children
on the host (`NodeQueue::Push`, `:91-143`). These kernels are the device
form of that queue, every sampling round and every `max_leaves` budget
included, so a tree runs K batches per drain and the host never holds a
node list:

  pop      `loop_pop_kernel` -> `loop_scan_offsets_kernel` ->
           `loop_workload_kernel`: take up to `max_batch_size` items from
           the head of the device FIFO into the staged `d_work_items`, and
           build `updateWorkloadInfo`'s block map (`:393-407`) from them at
           a given rows-per-block granularity. The pop does not move the
           head (the commit does), so a second pop of the same batch at
           another granularity rebuilds the items and the map identically:
           the histogram map is `HIST_WORKLOAD_GRANULARITY`-grained, the
           partition map `TPB_DEFAULT`-grained.
  retry    `loop_retry_merge_kernel` -> `loop_retry_compact_kernel` ->
           `loop_retry_stage_kernel` (+ the two map kernels): a sampling
           round's candidates land in the batch's final splits at their
           original slots, the nodes with no valid split and not pure are
           compacted in batch order into the next round's items (the
           host's `retry_items` / `retry_to_original`), and the next
           round's map is built. `loop_retry_finish_kernel` copies the
           final splits back into the split slots the partition reads.
  finalize `loop_finalize_splits_kernel`: a pure slot (DEVIATION 2502) and
           every slot past the live count become "no split".
  push     `loop_push_valid_kernel` -> `loop_push_scan_kernel` ->
           `loop_push_write_kernel` -> `loop_commit_kernel`:
           `NodeQueue::Push`, `max_leaves` budget included.
  tree     `loop_tree_kernel`: the node table as `SparseTreeNode` +
           `InstanceRange` arrays for the device leaf pass; the finished
           tree crosses to the host as two bulk copies.

THE HOST DOES NOT KNOW THE LIVE COUNT. It launches every per-batch kernel
at a bound it can prove (`n_bound` items, `blocks_bound` map entries) and
the device makes the excess inert:

  * items `[cur_n, n_bound)` are DUMMIES `NodeWorkItem(0, 0, (0, 0))`. They
    own no block of the map, so no histogram cell of theirs is written, and
    the finalize kernel stamps their split slot invalid whatever the split
    search made of an all-zero histogram.
  * map entries past the live total point at the LAST live node with
    `offset_blockid == num_blocks`. Every row kernel computes
    `range_pos = offset_blockid * TPB + tid` and guards it against the
    node's count (histogram loops, `count_local_left_kernel`, the partition
    functor's `load`/`store`, the copy-back), and `num_blocks * TPB >=
    count`, so such a block touches no row. For the scan-by-key partition
    they extend the last node's segment with invalid states, which its
    writer skips. With no live item at all the map points at dummy item 0
    (count 0).

Everything here is integer control data: node ids, counts, ranges. The node
ids are the ones the host push assigns (`left = n_nodes + 2 * rank`, rank =
number of valid splits before the item in batch order), the queue order is
push's (item order, left before right), so the feature sampler (seeded on
the node id) and every row kernel see the same inputs as on the host loop
and the tree is the same on every vendor.

PREFIX SUMS are two-level and exact: a thread sums its block's earlier
entries itself (at most `LOOP_TPB - 1` integer reads), writes the block
total, and a second launch adds the earlier blocks' totals. Integer adds in
any order give the same value.

THE ROOT (`NodeQueue`'s constructor, `builder.cuh:50-62`) is not a kernel:
the header, node 0 and queue item 0 of an expandable root are a few control
words that depend only on the sampled row count, so the builder uploads them
from one pinned buffer at the start of each tree (`loop_root_words`).

HEADER WORDS (`Int32`, `LOOP_HDR_WORDS` of them): see the `LOOP_H_*`
constants. NODE TABLE: `LOOP_NODE_INTS` Int32 per node
`[colid, left_child, instance_count, range_begin, range_count, depth]` and
two `dtype` values per node `[quesval, best_metric_val]` (written for split
nodes only). QUEUE: four Int32 per item `[idx, depth, begin, count]`.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceContext

from core.launch_clock import log_launch_ctx
from ensemble.decisiontree.batched_levelalgo.split import Split
from ensemble.flatnode import SparseTreeNode
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    WorkloadInfo,
)

comptime LOOP_TPB = 256
comptime LOOP_HDR_WORDS = 16
comptime LOOP_NODE_INTS = 6
comptime LOOP_QUEUE_INTS = 4

#: first unpopped queue item
comptime LOOP_H_HEAD = 0
#: one past the last queued item
comptime LOOP_H_TAIL = 1
#: nodes in the tree so far
comptime LOOP_H_NODES = 2
#: live items of the batch in flight
comptime LOOP_H_CUR = 3
#: 1 when a push would have exceeded the node capacity (host raises)
comptime LOOP_H_OVERFLOW = 4
#: live entries of the batch's block map
comptime LOOP_H_BLOCKS = 5
#: work items the batch in flight appends (committed by `loop_commit_kernel`)
comptime LOOP_H_PEND_APPEND = 6
#: valid splits of the batch in flight (committed likewise)
comptime LOOP_H_PEND_VALID = 7
#: items of the next sampling round (written by `loop_retry_compact_kernel`)
comptime LOOP_H_NEXT = 8
#: host slot only: the finished tree's depth (the last node's depth word)
comptime LOOP_H_DEPTH = 9


@always_inline
def _loop_blocks_of(
    queue: MutPointer[Int32, MutAnyOrigin],
    head: Int,
    j: Int,
    cur: Int,
    rows_per_block: Int,
) -> Int:
    """`max(ceildiv(count, TPB_DEFAULT), 1)` for live batch item `j`
    (`builder.cuh:399`), 0 for a dummy."""
    if j >= cur:
        return 0
    var count = Int(queue[unsafe_offset = (head + j) * LOOP_QUEUE_INTS + 3])
    var nb = (count + rows_per_block - 1) // rows_per_block
    if nb < 1:
        nb = 1
    return nb


def loop_pop_kernel(
    hdr: MutPointer[Int32, MutAnyOrigin],
    queue: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    pre: MutPointer[Int32, MutAnyOrigin],
    block_totals: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
    max_batch: Int32,
    rows_per_block: Int32,
):
    """`NodeQueue::Pop` (`:70-78`) plus the block-local half of
    `updateWorkloadInfo`'s running total. One thread per batch slot.

    `pre[i]` receives the blocks owned by this launch-block's earlier
    slots; `block_totals[b]` the launch-block's total.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_bound):
        return
    var head = Int(hdr[unsafe_offset=LOOP_H_HEAD])
    var tail = Int(hdr[unsafe_offset=LOOP_H_TAIL])
    var cur = tail - head
    if cur > Int(max_batch):
        cur = Int(max_batch)
    if cur > Int(n_bound):
        cur = Int(n_bound)
    if cur < 0:
        cur = 0
    var rpb = Int(rows_per_block)
    var blk_start = Int(block_idx.x) * Int(block_dim.x)
    var lp = 0
    var j = blk_start
    while j < i:
        lp += _loop_blocks_of(queue, head, j, cur, rpb)
        j += 1
    var own = _loop_blocks_of(queue, head, i, cur, rpb)
    pre[unsafe_offset=i] = Int32(lp)
    var blk_last = blk_start + Int(block_dim.x) - 1
    if blk_last > Int(n_bound) - 1:
        blk_last = Int(n_bound) - 1
    if i == blk_last:
        block_totals[unsafe_offset = Int(block_idx.x)] = Int32(lp + own)
    if i < cur:
        var q = (head + i) * LOOP_QUEUE_INTS
        work_items[unsafe_offset=i] = NodeWorkItem(
            Int(queue[unsafe_offset=q]),
            queue[unsafe_offset = q + 1],
            InstanceRange(
                Int(queue[unsafe_offset = q + 2]),
                Int(queue[unsafe_offset = q + 3]),
            ),
        )
    else:
        work_items[unsafe_offset=i] = NodeWorkItem(
            0, Int32(0), InstanceRange(0, 0)
        )
    if i == 0:
        hdr[unsafe_offset=LOOP_H_CUR] = Int32(cur)


def loop_scan_offsets_kernel(
    hdr: MutPointer[Int32, MutAnyOrigin],
    pre: MutPointer[Int32, MutAnyOrigin],
    block_totals: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
):
    """Second level of the prefix sum: add the earlier launch-blocks'
    totals, and publish the grand total as the live block count."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_bound):
        return
    var blk = Int(block_idx.x)
    var off = 0
    for b in range(blk):
        off += Int(block_totals[unsafe_offset=b])
    pre[unsafe_offset=i] = Int32(Int(pre[unsafe_offset=i]) + off)
    if i == Int(n_bound) - 1:
        hdr[unsafe_offset=LOOP_H_BLOCKS] = Int32(
            off + Int(block_totals[unsafe_offset=blk])
        )


def loop_workload_kernel(
    hdr: MutPointer[Int32, MutAnyOrigin],
    pre: MutPointer[Int32, MutAnyOrigin],
    workload_info: MutPointer[WorkloadInfo, MutAnyOrigin],
    blocks_bound: Int32,
    inert_scale: Int32,
):
    """`updateWorkloadInfo`'s writes (`:400-403`), one thread per map
    entry: entry `e` belongs to the node whose block range holds `e`
    (binary search on the exclusive prefix). Entries past the live total
    are made inert (module docstring): their `offset_blockid` is the last
    node's block count times `inert_scale` (rows per block / `block_dim`),
    so `offset_blockid * block_dim >= count` on a coarse histogram map as
    on the TPB-grained partition map, and no row is visited twice."""
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if e >= Int(blocks_bound):
        return
    var cur = Int(hdr[unsafe_offset=LOOP_H_CUR])
    var total = Int(hdr[unsafe_offset=LOOP_H_BLOCKS])
    if cur <= 0:
        workload_info[unsafe_offset=e] = WorkloadInfo(
            Int32(0), Int32(0), Int32(1)
        )
        return
    if e < total:
        # largest i in [0, cur) with pre[i] <= e; pre is strictly
        # increasing over live items (every node owns at least one block)
        var lo = 0
        var hi = cur - 1
        while lo < hi:
            var mid = (lo + hi + 1) // 2
            if Int(pre[unsafe_offset=mid]) <= e:
                lo = mid
            else:
                hi = mid - 1
        var first = Int(pre[unsafe_offset=lo])
        var nxt = total
        if lo + 1 < cur:
            nxt = Int(pre[unsafe_offset = lo + 1])
        workload_info[unsafe_offset=e] = WorkloadInfo(
            Int32(lo), Int32(e - first), Int32(nxt - first)
        )
    else:
        var last = cur - 1
        var nbl = total - Int(pre[unsafe_offset=last])
        workload_info[unsafe_offset=e] = WorkloadInfo(
            Int32(last), Int32(nbl * Int(inert_scale)), Int32(nbl)
        )


def loop_finalize_splits_kernel[
    dtype: DType, leaf_pure: Bool
](
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    hdr: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
):
    """`finalize_pure_splits_kernel` for a bound-sized batch: a slot past
    the live count is reset to a default `Split` (no split, not pure),
    and under `leaf_pure` a pure slot's `colid` becomes -1."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_bound):
        return
    var cur = Int(hdr[unsafe_offset=LOOP_H_CUR])
    if idx >= cur:
        splits[unsafe_offset=idx] = Split[dtype]()
        return
    comptime if leaf_pure:
        var s = splits[unsafe_offset=idx]
        if s.pure != Int32(0):
            s.colid = Int32(-1)
            splits[unsafe_offset=idx] = s


@always_inline
def _loop_valid[
    dtype: DType
](splits: MutPointer[Split[dtype], MutAnyOrigin], j: Int, cur: Int) -> Int:
    """1 when live batch item `j` holds a valid split (`:99`), else 0."""
    if j >= cur:
        return 0
    if splits[unsafe_offset=j].colid == Int32(-1):
        return 0
    return 1


@always_inline
def _loop_leaves0(hdr: MutPointer[Int32, MutAnyOrigin]) -> Int:
    """`tree->leaf_counter` at the start of the batch: 1 at the root, +1
    per split (`:111`), and every split adds two nodes."""
    return 1 + (Int(hdr[unsafe_offset=LOOP_H_NODES]) - 1) // 2


@always_inline
def _loop_push_code[
    dtype: DType
](
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    j: Int,
    cur: Int,
    rank: Int,
    max_depth: Int,
    min_samples_split: Int,
    leaves0: Int,
    max_leaves: Int,
) -> Int:
    """Batch item `j`'s contribution to `NodeQueue::Push`: bit 0 is "the
    split is taken" (valid, `:99`, and inside the leaf budget, `:101`),
    bits 1.. the number of its children that are expandable
    (`IsExpandable`, `:82-88`). `rank` is the number of valid splits
    before `j` in batch order.

    The budget: `leaf_counter` only grows, so the splits taken are exactly
    the first `max_leaves - leaves0` valid ones (their `break`), and a
    child is expandable only while `leaves0 + rank + 1 < max_leaves`, the
    counter's value after this split's `++`. `max_leaves == -1` is no
    budget."""
    if j >= cur:
        return 0
    var s = splits[unsafe_offset=j]
    if s.colid == Int32(-1):
        return 0
    if max_leaves != -1 and leaves0 + rank >= max_leaves:
        return 0
    ref item = work_items[unsafe_offset=j]
    var a = 0
    if Int(item.depth) + 1 < max_depth and (
        max_leaves == -1 or leaves0 + rank + 1 < max_leaves
    ):
        var left = Int(s.global_nLeft)
        var right = item.instances.count - left
        if left >= min_samples_split:
            a += 1
        if right >= min_samples_split:
            a += 1
    return 1 + 2 * a


def loop_push_valid_kernel[
    dtype: DType
](
    hdr: MutPointer[Int32, MutAnyOrigin],
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    pre_valid: MutPointer[Int32, MutAnyOrigin],
    bt_valid: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
):
    """Block-local prefix count of valid splits before each slot (with the
    earlier launch-blocks' totals, the slot's `rank`, which fixes its
    child ids and its place in the leaf budget)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_bound):
        return
    var cur = Int(hdr[unsafe_offset=LOOP_H_CUR])
    var blk_start = Int(block_idx.x) * Int(block_dim.x)
    var lv = 0
    var j = blk_start
    while j < i:
        lv += _loop_valid[dtype](splits, j, cur)
        j += 1
    pre_valid[unsafe_offset=i] = Int32(lv)
    var blk_last = blk_start + Int(block_dim.x) - 1
    if blk_last > Int(n_bound) - 1:
        blk_last = Int(n_bound) - 1
    if i == blk_last:
        bt_valid[unsafe_offset = Int(block_idx.x)] = Int32(
            lv + _loop_valid[dtype](splits, i, cur)
        )


def loop_push_scan_kernel[
    dtype: DType
](
    hdr: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    pre_valid: MutPointer[Int32, MutAnyOrigin],
    pre_append: MutPointer[Int32, MutAnyOrigin],
    bt_valid: MutPointer[Int32, MutAnyOrigin],
    bt_append: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
    max_depth: Int32,
    min_samples_split: Int32,
    max_leaves: Int32,
):
    """Block-local prefix count of work items appended before each slot
    (which fixes its children's queue positions). Reads the ranks
    `loop_push_valid_kernel` left."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_bound):
        return
    var cur = Int(hdr[unsafe_offset=LOOP_H_CUR])
    var l0 = _loop_leaves0(hdr)
    var md = Int(max_depth)
    var mss = Int(min_samples_split)
    var ml = Int(max_leaves)
    var blk = Int(block_idx.x)
    var off_v = 0
    for b in range(blk):
        off_v += Int(bt_valid[unsafe_offset=b])
    var blk_start = blk * Int(block_dim.x)
    var la = 0
    var j = blk_start
    while j < i:
        var rj = Int(pre_valid[unsafe_offset=j]) + off_v
        var c = _loop_push_code[dtype](
            work_items, splits, j, cur, rj, md, mss, l0, ml
        )
        la += c >> 1
        j += 1
    pre_append[unsafe_offset=i] = Int32(la)
    var blk_last = blk_start + Int(block_dim.x) - 1
    if blk_last > Int(n_bound) - 1:
        blk_last = Int(n_bound) - 1
    if i == blk_last:
        var ri = Int(pre_valid[unsafe_offset=i]) + off_v
        var own = _loop_push_code[dtype](
            work_items, splits, i, cur, ri, md, mss, l0, ml
        )
        bt_append[unsafe_offset=blk] = Int32(la + (own >> 1))


def loop_push_write_kernel[
    dtype: DType
](
    hdr: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    pre_valid: MutPointer[Int32, MutAnyOrigin],
    pre_append: MutPointer[Int32, MutAnyOrigin],
    bt_valid: MutPointer[Int32, MutAnyOrigin],
    bt_append: MutPointer[Int32, MutAnyOrigin],
    queue: MutPointer[Int32, MutAnyOrigin],
    nodes_i: MutPointer[Int32, MutAnyOrigin],
    nodes_f: MutPointer[Scalar[dtype], MutAnyOrigin],
    n_bound: Int32,
    max_depth: Int32,
    min_samples_split: Int32,
    node_capacity: Int32,
    max_leaves: Int32,
):
    """`NodeQueue::Push` (`:91-143`), one thread per batch slot, in the
    reference's field order:

      parent  overwritten as a split node whose left child id is the node
              count before this slot's children (`:105-110`);
      left    leaf with `global_nLeft` instances, range
              `(begin, local_nLeft)` (`:113-117`);
      right   leaf with `parent_count - global_nLeft` instances, range
              `(begin + local_nLeft, count - local_nLeft)` (`:122-129`);
      queue   each child appended only if expandable, left before right.

    Only the splits inside the leaf budget are taken (`_loop_push_code`);
    they are the first valid ones, so a taken split's `rank` is also its
    rank among the taken. The header's node count, head and tail are read
    here and advanced by `loop_commit_kernel` from the two pending words
    the last slot writes.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_bound):
        return
    var cur = Int(hdr[unsafe_offset=LOOP_H_CUR])
    var l0 = _loop_leaves0(hdr)
    var md = Int(max_depth)
    var mss = Int(min_samples_split)
    var ml = Int(max_leaves)
    var blk = Int(block_idx.x)
    var off_v = 0
    var off_a = 0
    for b in range(blk):
        off_v += Int(bt_valid[unsafe_offset=b])
        off_a += Int(bt_append[unsafe_offset=b])
    var rank = Int(pre_valid[unsafe_offset=i]) + off_v
    var before = Int(pre_append[unsafe_offset=i]) + off_a
    var code = _loop_push_code[dtype](
        work_items, splits, i, cur, rank, md, mss, l0, ml
    )
    if i == Int(n_bound) - 1:
        var taken = rank + _loop_valid[dtype](splits, i, cur)
        if ml != -1:
            var budget = ml - l0
            if budget < 0:
                budget = 0
            if taken > budget:
                taken = budget
        hdr[unsafe_offset=LOOP_H_PEND_VALID] = Int32(taken)
        hdr[unsafe_offset=LOOP_H_PEND_APPEND] = Int32(before + (code >> 1))
    if (code & 1) == 0:
        return
    var n0 = Int(hdr[unsafe_offset=LOOP_H_NODES])
    var tail0 = Int(hdr[unsafe_offset=LOOP_H_TAIL])
    var left_id = n0 + 2 * rank
    if left_id + 1 >= Int(node_capacity):
        hdr[unsafe_offset=LOOP_H_OVERFLOW] = Int32(1)
        return
    var s = splits[unsafe_offset=i]
    ref item = work_items[unsafe_offset=i]
    var pid = item.idx
    var depth1 = item.depth + Int32(1)
    var pbeg = item.instances.begin
    var pcnt = item.instances.count
    var gl = Int(s.global_nLeft)
    var ll = Int(s.local_nLeft)

    var pb = pid * LOOP_NODE_INTS
    nodes_i[unsafe_offset=pb] = s.colid
    nodes_i[unsafe_offset = pb + 1] = Int32(left_id)
    nodes_i[unsafe_offset = pb + 2] = Int32(pcnt)
    nodes_f[unsafe_offset = pid * 2] = s.quesval
    nodes_f[unsafe_offset = pid * 2 + 1] = s.best_metric_val

    var lb = left_id * LOOP_NODE_INTS
    nodes_i[unsafe_offset=lb] = Int32(-1)
    nodes_i[unsafe_offset = lb + 1] = Int32(-1)
    nodes_i[unsafe_offset = lb + 2] = Int32(gl)
    nodes_i[unsafe_offset = lb + 3] = Int32(pbeg)
    nodes_i[unsafe_offset = lb + 4] = Int32(ll)
    nodes_i[unsafe_offset = lb + 5] = depth1

    var rb = lb + LOOP_NODE_INTS
    nodes_i[unsafe_offset=rb] = Int32(-1)
    nodes_i[unsafe_offset = rb + 1] = Int32(-1)
    nodes_i[unsafe_offset = rb + 2] = Int32(pcnt - gl)
    nodes_i[unsafe_offset = rb + 3] = Int32(pbeg + ll)
    nodes_i[unsafe_offset = rb + 4] = Int32(pcnt - ll)
    nodes_i[unsafe_offset = rb + 5] = depth1

    # `code >> 1` counts exactly the expandable children (depth, size and
    # leaf budget), left before right.
    var pos = tail0 + before
    var leaf_ok = ml == -1 or l0 + rank + 1 < ml
    if Int(depth1) < md and leaf_ok:
        if gl >= mss:
            var ql = pos * LOOP_QUEUE_INTS
            queue[unsafe_offset=ql] = Int32(left_id)
            queue[unsafe_offset = ql + 1] = depth1
            queue[unsafe_offset = ql + 2] = Int32(pbeg)
            queue[unsafe_offset = ql + 3] = Int32(ll)
            pos += 1
        if pcnt - gl >= mss:
            var qr = pos * LOOP_QUEUE_INTS
            queue[unsafe_offset=qr] = Int32(left_id + 1)
            queue[unsafe_offset = qr + 1] = depth1
            queue[unsafe_offset = qr + 2] = Int32(pbeg + ll)
            queue[unsafe_offset = qr + 3] = Int32(pcnt - ll)


def loop_commit_kernel(hdr: MutPointer[Int32, MutAnyOrigin]):
    """Advance the queue header past the batch: three scalar adds, one
    thread. Separate from the push so no thread of the push reads a
    header word another thread of the same launch writes."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var cur = hdr[unsafe_offset=LOOP_H_CUR]
    var appended = hdr[unsafe_offset=LOOP_H_PEND_APPEND]
    var valid = hdr[unsafe_offset=LOOP_H_PEND_VALID]
    hdr[unsafe_offset=LOOP_H_HEAD] = hdr[unsafe_offset=LOOP_H_HEAD] + cur
    hdr[unsafe_offset=LOOP_H_TAIL] = (
        hdr[unsafe_offset=LOOP_H_TAIL] + appended
    )
    hdr[unsafe_offset=LOOP_H_NODES] = (
        hdr[unsafe_offset=LOOP_H_NODES] + Int32(2) * valid
    )
    hdr[unsafe_offset=LOOP_H_CUR] = Int32(0)
    hdr[unsafe_offset=LOOP_H_PEND_APPEND] = Int32(0)
    hdr[unsafe_offset=LOOP_H_PEND_VALID] = Int32(0)


# ---------------------------------------------------------------------------
# Sampling rounds (`doSplit`'s retry, `builder.cuh:452-458`) on the device
# ---------------------------------------------------------------------------


@always_inline
def _loop_retry_flag[
    dtype: DType, retry_pure: Bool
](splits: MutPointer[Split[dtype], MutAnyOrigin], j: Int, cur: Int) -> Int:
    """1 when active slot `j` goes to the next sampling round: no valid
    split and not pure (`advance_batch_replay`'s `not is_valid and not
    terminal`; under `retry_pure` a pure node is retried too)."""
    if j >= cur:
        return 0
    var s = splits[unsafe_offset=j]
    comptime if not retry_pure:
        if s.pure != Int32(0):
            return 0
    if s.colid == Int32(-1):
        return 1
    return 0


@always_inline
def _loop_items_blocks_of(
    items: MutPointer[NodeWorkItem, MutAnyOrigin],
    j: Int,
    cur: Int,
    rows_per_block: Int,
) -> Int:
    """`_loop_blocks_of` for a compacted item list."""
    if j >= cur:
        return 0
    var count = items[unsafe_offset=j].instances.count
    var nb = (count + rows_per_block - 1) // rows_per_block
    if nb < 1:
        nb = 1
    return nb


def loop_retry_merge_kernel[
    dtype: DType, retry_pure: Bool
](
    hdr: MutPointer[Int32, MutAnyOrigin],
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    final_splits: MutPointer[Split[dtype], MutAnyOrigin],
    a2o: MutPointer[Int32, MutAnyOrigin],
    rpre: MutPointer[Int32, MutAnyOrigin],
    rbt: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
    first: Int32,
):
    """A round's candidates into the batch's answer at their ORIGINAL
    slots (`st.final_splits[orig] = h[i]`), and the block-local prefix of
    the retry flags. `first` (round zero) reads the identity map."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_bound):
        return
    var cur = Int(hdr[unsafe_offset=LOOP_H_CUR])
    if i < cur:
        var orig = i
        if first == Int32(0):
            orig = Int(a2o[unsafe_offset=i])
        final_splits[unsafe_offset=orig] = splits[unsafe_offset=i]
    var blk_start = Int(block_idx.x) * Int(block_dim.x)
    var lf = 0
    var j = blk_start
    while j < i:
        lf += _loop_retry_flag[dtype, retry_pure](splits, j, cur)
        j += 1
    rpre[unsafe_offset=i] = Int32(lf)
    var blk_last = blk_start + Int(block_dim.x) - 1
    if blk_last > Int(n_bound) - 1:
        blk_last = Int(n_bound) - 1
    if i == blk_last:
        rbt[unsafe_offset = Int(block_idx.x)] = Int32(
            lf + _loop_retry_flag[dtype, retry_pure](splits, i, cur)
        )


def loop_retry_compact_kernel[
    dtype: DType, retry_pure: Bool
](
    hdr: MutPointer[Int32, MutAnyOrigin],
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    a2o: MutPointer[Int32, MutAnyOrigin],
    rpre: MutPointer[Int32, MutAnyOrigin],
    rbt: MutPointer[Int32, MutAnyOrigin],
    items_s: MutPointer[NodeWorkItem, MutAnyOrigin],
    a2o_s: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
    first: Int32,
):
    """The retried nodes, compacted in batch order (`retry_items`,
    `retry_to_original`), into the scratch lists; the count goes to
    `LOOP_H_NEXT`. Writes only scratch, so no thread reads a slot another
    thread of this launch writes."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_bound):
        return
    var cur = Int(hdr[unsafe_offset=LOOP_H_CUR])
    var blk = Int(block_idx.x)
    var off = 0
    for b in range(blk):
        off += Int(rbt[unsafe_offset=b])
    var rank = Int(rpre[unsafe_offset=i]) + off
    var flag = _loop_retry_flag[dtype, retry_pure](splits, i, cur)
    if flag != 0:
        var orig = Int32(i)
        if first == Int32(0):
            orig = a2o[unsafe_offset=i]
        items_s[unsafe_offset=rank] = work_items[unsafe_offset=i]
        a2o_s[unsafe_offset=rank] = orig
    if i == Int(n_bound) - 1:
        hdr[unsafe_offset=LOOP_H_NEXT] = Int32(rank + flag)


def loop_retry_stage_kernel(
    hdr: MutPointer[Int32, MutAnyOrigin],
    items_s: MutPointer[NodeWorkItem, MutAnyOrigin],
    a2o_s: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    a2o: MutPointer[Int32, MutAnyOrigin],
    pre: MutPointer[Int32, MutAnyOrigin],
    block_totals: MutPointer[Int32, MutAnyOrigin],
    n_bound: Int32,
    rows_per_block: Int32,
):
    """The next round's active items (`st.active_items = retry_items`):
    the compacted list into the staged `d_work_items` (dummies past it),
    its map's block-local prefix as `loop_pop_kernel` writes it, and the
    live count into `LOOP_H_CUR` for the round's kernels."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_bound):
        return
    var cur = Int(hdr[unsafe_offset=LOOP_H_NEXT])
    if cur > Int(n_bound):
        cur = Int(n_bound)
    var rpb = Int(rows_per_block)
    var blk_start = Int(block_idx.x) * Int(block_dim.x)
    var lp = 0
    var j = blk_start
    while j < i:
        lp += _loop_items_blocks_of(items_s, j, cur, rpb)
        j += 1
    var own = _loop_items_blocks_of(items_s, i, cur, rpb)
    pre[unsafe_offset=i] = Int32(lp)
    var blk_last = blk_start + Int(block_dim.x) - 1
    if blk_last > Int(n_bound) - 1:
        blk_last = Int(n_bound) - 1
    if i == blk_last:
        block_totals[unsafe_offset = Int(block_idx.x)] = Int32(lp + own)
    if i < cur:
        work_items[unsafe_offset=i] = items_s[unsafe_offset=i]
        a2o[unsafe_offset=i] = a2o_s[unsafe_offset=i]
    else:
        work_items[unsafe_offset=i] = NodeWorkItem(
            0, Int32(0), InstanceRange(0, 0)
        )
    if i == 0:
        hdr[unsafe_offset=LOOP_H_CUR] = Int32(cur)


def loop_retry_finish_kernel[
    dtype: DType
](
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    final_splits: MutPointer[Split[dtype], MutAnyOrigin],
    n_bound: Int32,
):
    """The batch's answer back into the split slots the partition and the
    push read (`st.result = st.final_splits`). Slots past the batch's
    live count carry stale words; the re-pop and `loop_finalize_splits`
    stamp them invalid."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_bound):
        return
    splits[unsafe_offset=i] = final_splits[unsafe_offset=i]


# ---------------------------------------------------------------------------
# The finished tree
# ---------------------------------------------------------------------------


def loop_tree_kernel[
    dtype: DType
](
    nodes_i: MutPointer[Int32, MutAnyOrigin],
    nodes_f: MutPointer[Scalar[dtype], MutAnyOrigin],
    tree_out: MutPointer[SparseTreeNode[dtype], MutAnyOrigin],
    ranges_out: MutPointer[InstanceRange, MutAnyOrigin],
    n_nodes: Int32,
):
    """The node table as the leaf pass's inputs, one thread per node:
    `CreateSplitNode(colid, quesval, best_metric_val, left, count)` for a
    split, `CreateLeafNode(count)` (`{0, 0, 0, -1, count}`) for a leaf,
    exactly what the host push built; and the node's instance range."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= Int(n_nodes):
        return
    var b = j * LOOP_NODE_INTS
    var colid = nodes_i[unsafe_offset=b]
    var count = nodes_i[unsafe_offset = b + 2]
    if colid != Int32(-1):
        tree_out[unsafe_offset=j] = SparseTreeNode[dtype](
            colid,
            nodes_f[unsafe_offset = 2 * j],
            nodes_f[unsafe_offset = 2 * j + 1],
            Int64(Int(nodes_i[unsafe_offset = b + 1])),
            count,
        )
    else:
        tree_out[unsafe_offset=j] = SparseTreeNode[dtype](
            Int32(0),
            Scalar[dtype](0),
            Scalar[dtype](0),
            Int64(-1),
            count,
        )
    ranges_out[unsafe_offset=j] = InstanceRange(
        Int(nodes_i[unsafe_offset = b + 3]),
        Int(nodes_i[unsafe_offset = b + 4]),
    )


# ===========================================================================
# Launchers
# ===========================================================================


def loop_root_words[
    o: MutOrigin, //
](dst: MutPointer[Int32, o], n_rows: Int):
    """Fill `LOOP_HDR_WORDS + LOOP_QUEUE_INTS + LOOP_NODE_INTS` host words
    with an expandable root's initial state, in upload order: the header
    (queue holds one item, tree holds one node), queue item 0
    `[idx 0, depth 0, begin 0, count n_rows]`, node 0 (a leaf holding every
    sampled row: `[colid -1, left -1, count, begin 0, range count, depth
    0]`)."""
    for w in range(LOOP_HDR_WORDS):
        dst[unsafe_offset=w] = Int32(0)
    dst[unsafe_offset=LOOP_H_TAIL] = Int32(1)
    dst[unsafe_offset=LOOP_H_NODES] = Int32(1)
    var q = LOOP_HDR_WORDS
    dst[unsafe_offset=q] = Int32(0)
    dst[unsafe_offset = q + 1] = Int32(0)
    dst[unsafe_offset = q + 2] = Int32(0)
    dst[unsafe_offset = q + 3] = Int32(n_rows)
    var nd = LOOP_HDR_WORDS + LOOP_QUEUE_INTS
    dst[unsafe_offset=nd] = Int32(-1)
    dst[unsafe_offset = nd + 1] = Int32(-1)
    dst[unsafe_offset = nd + 2] = Int32(n_rows)
    dst[unsafe_offset = nd + 3] = Int32(0)
    dst[unsafe_offset = nd + 4] = Int32(n_rows)
    dst[unsafe_offset = nd + 5] = Int32(0)


def loop_scan_words(max_batch: Int) -> Int:
    """Int32 words of the scan scratch: three per-slot prefix arrays and
    three per-launch-block total arrays."""
    return 3 * max_batch + 3 * ceildiv(max_batch, LOOP_TPB)


def launch_loop_pop(
    ctx: DeviceContext,
    hdr: MutPointer[Int32, MutUntrackedOrigin],
    queue: MutPointer[Int32, MutUntrackedOrigin],
    work_items: MutPointer[NodeWorkItem, MutUntrackedOrigin],
    workload_info: MutPointer[WorkloadInfo, MutUntrackedOrigin],
    scan: MutPointer[Int32, MutUntrackedOrigin],
    max_batch: Int,
    n_bound: Int,
    blocks_bound: Int,
    rows_per_block: Int,
    inert_scale: Int = 1,
) raises:
    """Pop + block map for one batch: three launches. `scan` is the
    `loop_scan_words(max_batch)` scratch. `inert_scale` is
    `rows_per_block / TPB_DEFAULT` (see `loop_workload_kernel`)."""
    var pre = scan
    var bt = scan.unsafe_offset(3 * max_batch)
    var grid = ceildiv(n_bound, LOOP_TPB)
    log_launch_ctx(ctx, "loop_pop")
    ctx.enqueue_function[loop_pop_kernel](
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        queue.unsafe_origin_cast[MutAnyOrigin](),
        work_items.unsafe_origin_cast[MutAnyOrigin](),
        pre.unsafe_origin_cast[MutAnyOrigin](),
        bt.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_bound),
        Int32(max_batch),
        Int32(rows_per_block),
        grid_dim=grid,
        block_dim=LOOP_TPB,
    )
    _launch_loop_map(
        ctx, hdr, workload_info, scan, max_batch, n_bound, blocks_bound,
        inert_scale,
    )


def _launch_loop_map(
    ctx: DeviceContext,
    hdr: MutPointer[Int32, MutUntrackedOrigin],
    workload_info: MutPointer[WorkloadInfo, MutUntrackedOrigin],
    scan: MutPointer[Int32, MutUntrackedOrigin],
    max_batch: Int,
    n_bound: Int,
    blocks_bound: Int,
    inert_scale: Int,
) raises:
    """The block map's second half (offsets + entries) from the per-slot
    block prefix a pop or a retry stage left in `scan`."""
    var pre = scan
    var bt = scan.unsafe_offset(3 * max_batch)
    var grid = ceildiv(n_bound, LOOP_TPB)
    log_launch_ctx(ctx, "loop_scan_offsets")
    ctx.enqueue_function[loop_scan_offsets_kernel](
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        pre.unsafe_origin_cast[MutAnyOrigin](),
        bt.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_bound),
        grid_dim=grid,
        block_dim=LOOP_TPB,
    )
    log_launch_ctx(ctx, "loop_workload")
    ctx.enqueue_function[loop_workload_kernel](
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        pre.unsafe_origin_cast[MutAnyOrigin](),
        workload_info.unsafe_origin_cast[MutAnyOrigin](),
        Int32(blocks_bound),
        Int32(inert_scale),
        grid_dim=ceildiv(blocks_bound, LOOP_TPB),
        block_dim=LOOP_TPB,
    )


def launch_loop_finalize[
    dtype: DType, leaf_pure: Bool
](
    ctx: DeviceContext,
    splits: MutPointer[Split[dtype], MutUntrackedOrigin],
    hdr: MutPointer[Int32, MutUntrackedOrigin],
    n_bound: Int,
) raises:
    comptime k_fin = loop_finalize_splits_kernel[dtype, leaf_pure]
    log_launch_ctx(ctx, "loop_finalize_splits")
    ctx.enqueue_function[k_fin](
        splits.unsafe_origin_cast[MutAnyOrigin](),
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_bound),
        grid_dim=ceildiv(n_bound, LOOP_TPB),
        block_dim=LOOP_TPB,
    )


def launch_loop_push[
    dtype: DType
](
    ctx: DeviceContext,
    hdr: MutPointer[Int32, MutUntrackedOrigin],
    queue: MutPointer[Int32, MutUntrackedOrigin],
    nodes_i: MutPointer[Int32, MutUntrackedOrigin],
    nodes_f: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    work_items: MutPointer[NodeWorkItem, MutUntrackedOrigin],
    splits: MutPointer[Split[dtype], MutUntrackedOrigin],
    scan: MutPointer[Int32, MutUntrackedOrigin],
    max_batch: Int,
    n_bound: Int,
    max_depth: Int,
    min_samples_split: Int,
    node_capacity: Int,
    max_leaves: Int,
) raises:
    """`NodeQueue::Push` for one batch: four launches. Uses the second
    and third per-slot arrays and block-total arrays of `scan` (the first
    of each still holds the pop's block prefix, which nothing reads after
    the map is built, but they are kept apart so a reader can inspect
    both)."""
    var nblk = ceildiv(max_batch, LOOP_TPB)
    var pre_v = scan.unsafe_offset(max_batch)
    var pre_a = scan.unsafe_offset(2 * max_batch)
    var bt_v = scan.unsafe_offset(3 * max_batch + nblk)
    var bt_a = scan.unsafe_offset(3 * max_batch + 2 * nblk)
    var grid = ceildiv(n_bound, LOOP_TPB)
    comptime k_valid = loop_push_valid_kernel[dtype]
    comptime k_scan = loop_push_scan_kernel[dtype]
    comptime k_write = loop_push_write_kernel[dtype]
    log_launch_ctx(ctx, "loop_push_valid")
    ctx.enqueue_function[k_valid](
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        splits.unsafe_origin_cast[MutAnyOrigin](),
        pre_v.unsafe_origin_cast[MutAnyOrigin](),
        bt_v.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_bound),
        grid_dim=grid,
        block_dim=LOOP_TPB,
    )
    log_launch_ctx(ctx, "loop_push_scan")
    ctx.enqueue_function[k_scan](
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        work_items.unsafe_origin_cast[MutAnyOrigin](),
        splits.unsafe_origin_cast[MutAnyOrigin](),
        pre_v.unsafe_origin_cast[MutAnyOrigin](),
        pre_a.unsafe_origin_cast[MutAnyOrigin](),
        bt_v.unsafe_origin_cast[MutAnyOrigin](),
        bt_a.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_bound),
        Int32(max_depth),
        Int32(min_samples_split),
        Int32(max_leaves),
        grid_dim=grid,
        block_dim=LOOP_TPB,
    )
    log_launch_ctx(ctx, "loop_push_write")
    ctx.enqueue_function[k_write](
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        work_items.unsafe_origin_cast[MutAnyOrigin](),
        splits.unsafe_origin_cast[MutAnyOrigin](),
        pre_v.unsafe_origin_cast[MutAnyOrigin](),
        pre_a.unsafe_origin_cast[MutAnyOrigin](),
        bt_v.unsafe_origin_cast[MutAnyOrigin](),
        bt_a.unsafe_origin_cast[MutAnyOrigin](),
        queue.unsafe_origin_cast[MutAnyOrigin](),
        nodes_i.unsafe_origin_cast[MutAnyOrigin](),
        nodes_f.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_bound),
        Int32(max_depth),
        Int32(min_samples_split),
        Int32(node_capacity),
        Int32(max_leaves),
        grid_dim=grid,
        block_dim=LOOP_TPB,
    )
    log_launch_ctx(ctx, "loop_commit")
    ctx.enqueue_function[loop_commit_kernel](
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        grid_dim=1,
        block_dim=1,
    )


def loop_retry_words(max_batch: Int) -> Int:
    """Int32 words of the retry scratch: the active-to-original map, its
    compacted twin, the per-slot flag prefix and the per-launch-block flag
    totals."""
    return 3 * max_batch + ceildiv(max_batch, LOOP_TPB)


def launch_loop_retry_merge[
    dtype: DType, retry_pure: Bool
](
    ctx: DeviceContext,
    hdr: MutPointer[Int32, MutUntrackedOrigin],
    splits: MutPointer[Split[dtype], MutUntrackedOrigin],
    final_splits: MutPointer[Split[dtype], MutUntrackedOrigin],
    retry: MutPointer[Int32, MutUntrackedOrigin],
    max_batch: Int,
    n_bound: Int,
    first: Bool,
) raises:
    """A round's candidates into the batch's final splits (one launch)."""
    var a2o = retry
    var rpre = retry.unsafe_offset(2 * max_batch)
    var rbt = retry.unsafe_offset(3 * max_batch)
    comptime k_merge = loop_retry_merge_kernel[dtype, retry_pure]
    log_launch_ctx(ctx, "loop_retry_merge")
    ctx.enqueue_function[k_merge](
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        splits.unsafe_origin_cast[MutAnyOrigin](),
        final_splits.unsafe_origin_cast[MutAnyOrigin](),
        a2o.unsafe_origin_cast[MutAnyOrigin](),
        rpre.unsafe_origin_cast[MutAnyOrigin](),
        rbt.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_bound),
        Int32(1) if first else Int32(0),
        grid_dim=ceildiv(n_bound, LOOP_TPB),
        block_dim=LOOP_TPB,
    )


def launch_loop_retry_next[
    dtype: DType, retry_pure: Bool
](
    ctx: DeviceContext,
    hdr: MutPointer[Int32, MutUntrackedOrigin],
    splits: MutPointer[Split[dtype], MutUntrackedOrigin],
    work_items: MutPointer[NodeWorkItem, MutUntrackedOrigin],
    workload_info: MutPointer[WorkloadInfo, MutUntrackedOrigin],
    items_s: MutPointer[NodeWorkItem, MutUntrackedOrigin],
    retry: MutPointer[Int32, MutUntrackedOrigin],
    scan: MutPointer[Int32, MutUntrackedOrigin],
    max_batch: Int,
    n_bound: Int,
    blocks_bound: Int,
    rows_per_block: Int,
    inert_scale: Int,
    first: Bool,
) raises:
    """After `launch_loop_retry_merge`: compact the retried nodes and
    stage them, with their block map, as the next round's batch (four
    launches). A round with nothing to retry stages zero items and every
    kernel of the next round is inert."""
    var a2o = retry
    var a2o_s = retry.unsafe_offset(max_batch)
    var rpre = retry.unsafe_offset(2 * max_batch)
    var rbt = retry.unsafe_offset(3 * max_batch)
    var grid = ceildiv(n_bound, LOOP_TPB)
    comptime k_compact = loop_retry_compact_kernel[dtype, retry_pure]
    log_launch_ctx(ctx, "loop_retry_compact")
    ctx.enqueue_function[k_compact](
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        splits.unsafe_origin_cast[MutAnyOrigin](),
        work_items.unsafe_origin_cast[MutAnyOrigin](),
        a2o.unsafe_origin_cast[MutAnyOrigin](),
        rpre.unsafe_origin_cast[MutAnyOrigin](),
        rbt.unsafe_origin_cast[MutAnyOrigin](),
        items_s.unsafe_origin_cast[MutAnyOrigin](),
        a2o_s.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_bound),
        Int32(1) if first else Int32(0),
        grid_dim=grid,
        block_dim=LOOP_TPB,
    )
    var pre = scan
    var bt = scan.unsafe_offset(3 * max_batch)
    log_launch_ctx(ctx, "loop_retry_stage")
    ctx.enqueue_function[loop_retry_stage_kernel](
        hdr.unsafe_origin_cast[MutAnyOrigin](),
        items_s.unsafe_origin_cast[MutAnyOrigin](),
        a2o_s.unsafe_origin_cast[MutAnyOrigin](),
        work_items.unsafe_origin_cast[MutAnyOrigin](),
        a2o.unsafe_origin_cast[MutAnyOrigin](),
        pre.unsafe_origin_cast[MutAnyOrigin](),
        bt.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_bound),
        Int32(rows_per_block),
        grid_dim=grid,
        block_dim=LOOP_TPB,
    )
    _launch_loop_map(
        ctx, hdr, workload_info, scan, max_batch, n_bound, blocks_bound,
        inert_scale,
    )


def launch_loop_retry_finish[
    dtype: DType
](
    ctx: DeviceContext,
    splits: MutPointer[Split[dtype], MutUntrackedOrigin],
    final_splits: MutPointer[Split[dtype], MutUntrackedOrigin],
    n_bound: Int,
) raises:
    comptime k_fin = loop_retry_finish_kernel[dtype]
    log_launch_ctx(ctx, "loop_retry_finish")
    ctx.enqueue_function[k_fin](
        splits.unsafe_origin_cast[MutAnyOrigin](),
        final_splits.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_bound),
        grid_dim=ceildiv(n_bound, LOOP_TPB),
        block_dim=LOOP_TPB,
    )


def launch_loop_tree[
    dtype: DType
](
    ctx: DeviceContext,
    nodes_i: MutPointer[Int32, MutUntrackedOrigin],
    nodes_f: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    tree_out: MutPointer[SparseTreeNode[dtype], MutUntrackedOrigin],
    ranges_out: MutPointer[InstanceRange, MutUntrackedOrigin],
    n_nodes: Int,
) raises:
    comptime k_tree = loop_tree_kernel[dtype]
    log_launch_ctx(ctx, "loop_tree")
    ctx.enqueue_function[k_tree](
        nodes_i.unsafe_origin_cast[MutAnyOrigin](),
        nodes_f.unsafe_origin_cast[MutAnyOrigin](),
        tree_out.unsafe_origin_cast[MutAnyOrigin](),
        ranges_out.unsafe_origin_cast[MutAnyOrigin](),
        Int32(n_nodes),
        grid_dim=ceildiv(n_nodes, LOOP_TPB),
        block_dim=LOOP_TPB,
    )
