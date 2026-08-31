# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does the host control plane build the tree cuML's `NodeQueue` builds?

    pixi run mojo run -I . ensemble/original/builder_check.mojo

Covers `ensemble/decisiontree/batched_levelalgo/builder.mojo` against
`builder.cuh` at rapidsai/cuml `v26.08.00` (`265b9da6`). Host only; no
kernel is launched here, because nothing in this file's scope runs on the
device.

WHY THE FIXTURES LOOK THE WAY THEY DO. A tree-shape check has a specific way
of being useless: if every node splits and every split is valid, the answer
is a perfect binary tree and ANY implementation that appends two children per
node reproduces it. That is this repository's "conservation cannot see a tree
that never splits" scar, pointed the other way -- a uniform fixture verifies
the count and nothing about the SHAPE.

So every fixture below is deliberately ragged: invalid splits mixed in with
valid ones, instance counts that trip `min_samples_split` at different
depths, a leaf budget that runs out PART WAY THROUGH a batch, and left/right
counts that are never equal, so a left/right swap is visible. Expected node
lists are written out by hand from `builder.cuh:91-143`, node by node, and
compared per FIELD -- not as a count, not as a digest.

The five arms:

  A. `IsExpandable` (`:82-88`), each of its three rejections separately.
  B. `Push` (`:91-143`) on a ragged batch: parent rewrite, child order,
     instance ranges, `leaf_counter` accounting, `depth_counter`.
  C. The `max_leaves` budget (`:101`) -- once it is spent the rest of the
     batch is abandoned. NOTE their `break` is an early exit and NOT a
     semantic: `leaf_counter` only increases, so a `continue` reaches the
     same end state and no fixture can distinguish the two spellings. The
     first version of this file claimed otherwise; a sabotage measured no
     movement and was right.
  D. The workspace and the shared-vs-global dispatch, against an independent
     transcription of `workspaceSize` (`:299-332`) and
     `computeSharedMemoryConfig` (`:522-551`).
  E. `updateWorkloadInfo` (`:393-407`) on a ragged batch, per entry.

Plus four sabotages, each in the SHIPPED code path via `builder.mojo`'s
comptime `sabotage` hook -- sabotaging a copy proves nothing about the
original -- and each with its own predicted movement.

TWO OF THE FIRST FOUR SABOTAGES FAILED, and both failures were defects in
THIS FILE rather than in the port:

  * the `break`-vs-`continue` arm measured nothing, because the two are
    genuinely equivalent (above). Replaced with one that drops the budget
    test outright.
  * the FIFO-vs-LIFO arm measured nothing, because `_build_ragged`'s batch
    has an INVALID split in it -- so both orders append the same nodes in
    the same places and the arm was blind. `_build_order_fixture` exists
    for that arm alone, with both nodes splitting and different left counts.

One remaining sabotage still predicts NO movement, deliberately: swapping
`local_nLeft` for `global_nLeft` cannot be seen on a single device, and the
arm records that as a GAP rather than reporting a pass.
"""

from std.math import ceildiv

from ensemble.decisiontree.batched_levelalgo.builder import (
    ALIGN_VALUE,
    N_BLKS_FOR_COLS,
    NodeQueue,
    SplitSummary,
    TPB_DEFAULT,
    TUNABLE_SPLIT_HISTOGRAM_DYNAMIC_SMEM_LIMIT_BYTES,
    calculate_aligned_bytes,
    compute_shared_memory_config,
    max_blocks_dimx_for,
    max_nodes,
    max_sampling_rounds_for,
    n_sampled_cols_for,
    sampled_cols_in_round,
    update_workload_info,
    workspace_layout,
)
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    WorkloadInfo,
)
from ensemble.decisiontree.decisiontree import DecisionTreeParams, GINI

comptime DT = DType.float32


def _params(
    max_depth: Int, max_leaves: Int, min_samples_split: Int, max_batch: Int
) -> DecisionTreeParams:
    return DecisionTreeParams(
        max_depth=Int32(max_depth),
        max_leaves=Int32(max_leaves),
        max_features=Float32(1.0),
        max_n_bins=Int32(128),
        min_samples_leaf=Int32(1),
        min_samples_split=Int32(min_samples_split),
        split_criterion=GINI,
        min_impurity_decrease=Float32(0.0),
        max_batch_size=Int32(max_batch),
    )


def _split(
    valid: Bool, colid: Int, quesval: Float32, n_left: Int
) -> SplitSummary[DT]:
    """A split summary as `doSplit`'s host copy would deliver it. `n_left`
    is deliberately never half the parent, so a left/right swap shows."""
    return SplitSummary[DT](
        valid,
        Int32(colid),
        quesval,
        Float32(0.5),
        Int64(n_left),
        Int64(n_left),
    )


def arm_a_is_expandable() raises -> Int:
    """`builder.cuh:82-88`, each rejection on its own."""
    var fails = 0

    # depth >= max_depth: with max_depth 0 the ROOT is not expandable, so
    # the queue starts empty and the tree is a single leaf.
    var q0 = NodeQueue[DT](_params(0, -1, 2, 16), 32, 100, Int32(1))
    if q0.has_work():
        fails += 1
        print("  arm A: max_depth 0 must leave the root unexpandable")
    if len(q0.tree.sparsetree) != 1:
        fails += 1
        print("  arm A: max_depth 0 must give a 1-node tree")

    # InstanceCount < min_samples_split
    var q1 = NodeQueue[DT](_params(8, -1, 200, 16), 32, 100, Int32(1))
    if q1.has_work():
        fails += 1
        print("  arm A: min_samples_split 200 > 100 rows must block the root")

    # max_leaves already reached: leaf_counter starts at 1, so max_leaves 1
    # blocks immediately.
    var q2 = NodeQueue[DT](_params(8, 1, 2, 16), 32, 100, Int32(1))
    if q2.has_work():
        fails += 1
        print("  arm A: max_leaves 1 must block the root (leaf_counter is 1)")

    # and the control: all three satisfied
    var q3 = NodeQueue[DT](_params(8, -1, 2, 16), 32, 100, Int32(1))
    if not q3.has_work():
        fails += 1
        print("  arm A: the permissive case must produce a work item")

    if fails == 0:
        print("  arm A OK: all three IsExpandable rejections, plus the control")
    return fails


def _build_ragged[
    sab: Int = 0
]() raises -> NodeQueue[DT, sab]:
    """One deterministic, deliberately ragged tree build.

    Root: 100 rows. Batch 1 splits it 30/70. Batch 2 gets both children:
    the left (30 rows) gets an INVALID split and must stay a leaf; the right
    (70 rows) splits 25/45. Batch 3 gets the two grandchildren; the 25-row
    one is below `min_samples_split` = 30 and never became a work item at
    all, so only the 45-row one is present, and it splits 20/25.

    Hand-computed expected tree, by `builder.cuh:91-143`:

      idx 0  split, left_child 1, count 100
      idx 1  leaf,  count 30        (invalid split in batch 2)
      idx 2  split, left_child 3, count 70
      idx 3  leaf,  count 25        (never expandable: 25 < 30)
      idx 4  split, left_child 5, count 45
      idx 5  leaf,  count 20
      idx 6  leaf,  count 25

    Ranges: 0:[0,100) 1:[0,30) 2:[30,100) 3:[30,55) 4:[55,100)
            5:[55,75) 6:[75,100)
    """
    var q = NodeQueue[DT, sab](_params(8, -1, 30, 16), 64, 100, Int32(1))

    # batch 1
    var b1 = q.pop()
    var s1 = List[SplitSummary[DT]]()
    s1.append(_split(True, 3, 1.5, 30))
    q.push(b1, s1)

    # batch 2 -- both children; the 30-row one gets an invalid split
    var b2 = q.pop()
    var s2 = List[SplitSummary[DT]]()
    for i in range(len(b2)):
        if b2[i].instances.count == 30:
            s2.append(_split(False, -1, 0.0, 0))
        else:
            s2.append(_split(True, 7, 2.5, 25))
    q.push(b2, s2)

    # batch 3 -- only the 45-row grandchild is expandable
    var b3 = q.pop()
    var s3 = List[SplitSummary[DT]]()
    for _ in range(len(b3)):
        s3.append(_split(True, 1, 3.5, 20))
    q.push(b3, s3)

    return q^


def arm_b_push() raises -> Int:
    """Per FIELD, per node, against the hand-computed tree above."""
    var q = _build_ragged()
    var fails = 0

    var want_leaf = [False, True, False, True, False, True, True]
    var want_count = [100, 30, 70, 25, 45, 20, 25]
    var want_left = [1, -1, 3, -1, 5, -1, -1]
    var want_begin = [0, 0, 30, 30, 55, 55, 75]
    var want_range = [100, 30, 70, 25, 45, 20, 25]

    if len(q.tree.sparsetree) != 7:
        fails += 1
        print(
            "  arm B: expected 7 nodes, got", len(q.tree.sparsetree),
        )
        return fails

    for i in range(7):
        var n = q.tree.sparsetree[i]
        if n.IsLeaf() != want_leaf[i]:
            fails += 1
            print("  arm B node", i, "IsLeaf", n.IsLeaf(), "want", want_leaf[i])
        if Int(n.InstanceCount()) != want_count[i]:
            fails += 1
            print(
                "  arm B node", i, "count", n.InstanceCount(),
                "want", want_count[i],
            )
        if Int(n.LeftChildId()) != want_left[i]:
            fails += 1
            print(
                "  arm B node", i, "left_child", n.LeftChildId(),
                "want", want_left[i],
            )
        var r = q.node_instances_[i]
        if r.begin != want_begin[i] or r.count != want_range[i]:
            fails += 1
            print(
                "  arm B node", i, "range [", r.begin, ",", r.count,
                ") want [", want_begin[i], ",", want_range[i], ")",
            )

    # `leaf_counter` is incremented once per SPLIT (`:111`), starting at 1.
    # Three splits happened, so 4.
    if q.tree.leaf_counter != Int32(4):
        fails += 1
        print(
            "  arm B: leaf_counter", q.tree.leaf_counter,
            "want 4 (1 initial + 1 per split, THREE splits)",
        )
    if q.tree.depth_counter != Int32(3):
        fails += 1
        print("  arm B: depth_counter", q.tree.depth_counter, "want 3")
    if q.has_work():
        fails += 1
        print("  arm B: the frontier must be empty")

    if fails == 0:
        print(
            "  arm B OK: 7 nodes, every field, every instance range, plus"
            " leaf_counter 4 and depth_counter 3"
        )
    return fails


def arm_c_max_leaves_break() raises -> Int:
    """`builder.cuh:101` is a BREAK. The rest of the batch is abandoned.

    Root 100 rows splits 40/60. Then a batch of TWO expandable children with
    `max_leaves = 3`: leaf_counter is 2 after the first split, so the first
    item splits (counter -> 3) and the SECOND is abandoned by the break.
    Result: 5 nodes, not 7.
    """
    var q = NodeQueue[DT](_params(8, 3, 2, 16), 64, 100, Int32(1))
    var b1 = q.pop()
    var s1 = List[SplitSummary[DT]]()
    s1.append(_split(True, 0, 1.0, 40))
    q.push(b1, s1)

    var b2 = q.pop()
    if len(b2) != 2:
        print("  arm C: expected a 2-item batch, got", len(b2))
        return 1
    var s2 = List[SplitSummary[DT]]()
    s2.append(_split(True, 1, 2.0, 10))
    s2.append(_split(True, 2, 3.0, 20))
    q.push(b2, s2)

    if len(q.tree.sparsetree) != 5:
        print(
            "  arm C FAILED: expected 5 nodes (the break abandons item 2),"
            " got", len(q.tree.sparsetree),
        )
        return 1
    if q.tree.leaf_counter != Int32(3):
        print("  arm C FAILED: leaf_counter", q.tree.leaf_counter, "want 3")
        return 1
    print(
        "  arm C OK: max_leaves 3 gave 5 nodes -- the second item of the"
        " batch was abandoned once the leaf budget was spent."
    )
    print(
        "    NOTE, measured not assumed: their `break` at builder.cuh:101 is"
        " an EARLY EXIT, not a semantic. leaf_counter only ever increases,"
        " so `continue` reaches the same end state and no fixture can"
        " distinguish them. This arm checks the BUDGET, not the break."
    )
    return 0


def arm_d_workspace_and_smem() raises -> Int:
    """Against an independent transcription of their two expressions."""
    var fails = 0

    # -- maxNodes (`:283-297`), across their depth-13 cliff
    var want_nodes = [1, 3, 7, 8191, 8191, 8191]
    var depths = [0, 1, 2, 13, 14, 30]
    for i in range(len(depths)):
        var got = max_nodes(Int32(depths[i]))
        if got != want_nodes[i]:
            fails += 1
            print(
                "  arm D maxNodes(", depths[i], ") =", got,
                "want", want_nodes[i],
            )
    if max_nodes(Int32(12)) != 8191:
        fails += 1
        print("  arm D: depth 12 is the last DENSE case and must be 2^13-1")

    # -- workspaceSize (`:299-332`), transcribed here a SECOND time and
    #    spelled differently, because a check that imports the formula it is
    #    checking moves with its own sabotage. (That happened to another lane
    #    this round and ran green.)
    var max_batch = 4096
    var max_n_bins = 128
    var num_outputs = 3
    var n_rows = 100000
    var n_cols = 40
    var sz_bin = 8
    var sz_split = 40
    var sz_item = 32
    var sz_wl = 12
    var sz_idx = 4

    var lay = workspace_layout(
        Int32(max_batch), Int32(max_n_bins), Int32(num_outputs),
        n_rows, n_cols, sz_bin, sz_split, sz_item, sz_wl, sz_idx,
    )

    var hist_len = max_batch * max_n_bins * N_BLKS_FOR_COLS * num_outputs
    var blocks = 1 + max_batch + n_rows // TPB_DEFAULT
    var a = ceildiv(sz_idx, ALIGN_VALUE) * ALIGN_VALUE
    a += ceildiv(sz_bin * hist_len, ALIGN_VALUE) * ALIGN_VALUE
    a += ceildiv(4 * max_batch, ALIGN_VALUE) * ALIGN_VALUE
    a += ceildiv(sz_split * max_batch, ALIGN_VALUE) * ALIGN_VALUE
    a += ceildiv(sz_item * max_batch, ALIGN_VALUE) * ALIGN_VALUE
    a += ceildiv(sz_wl * blocks, ALIGN_VALUE) * ALIGN_VALUE
    a += ceildiv(sz_idx * max_batch * n_cols, ALIGN_VALUE) * ALIGN_VALUE
    a += ceildiv(sz_idx * n_rows, ALIGN_VALUE) * ALIGN_VALUE
    if lay.device_total != a:
        fails += 1
        print("  arm D device workspace", lay.device_total, "want", a)

    var hh = ceildiv(sz_wl * blocks, ALIGN_VALUE) * ALIGN_VALUE
    hh += ceildiv(sz_split * max_batch, ALIGN_VALUE) * ALIGN_VALUE
    if lay.host_total != hh:
        fails += 1
        print("  arm D host workspace", lay.host_total, "want", hh)

    # every offset must be ALIGN_VALUE-aligned and strictly increasing
    var offs = [
        lay.n_nodes, lay.histograms, lay.mutex, lay.splits,
        lay.d_work_items, lay.workload_info, lay.column_samples,
        lay.partition_row_ids,
    ]
    for i in range(len(offs)):
        if offs[i] % ALIGN_VALUE != 0:
            fails += 1
            print("  arm D offset", i, "is not", ALIGN_VALUE, "-aligned")
        if i > 0 and offs[i] <= offs[i - 1]:
            fails += 1
            print("  arm D offsets are not strictly increasing at", i)
    if lay.partition_row_ids >= lay.device_total:
        fails += 1
        print("  arm D: the last allocation runs off the end of the buffer")

    # -- computeSharedMemoryConfig (`:522-551`), across their 16 KiB edge.
    # `available_smem` 32 KB is the Apple column's per-block limit.
    var apple = 32 * 1024
    var cdf = 1024
    # small: 128 bins x 2 classes x 8 B + 128 x 4 B + padding = 2560 + 12
    var c_small = compute_shared_memory_config(
        Int32(128), Int32(2), 8, 4, 40, apple, cdf, 32
    )
    if c_small.use_global_memory_histogram:
        fails += 1
        print("  arm D: a 2-class 128-bin histogram must take the SHARED arm")

    # large: 128 bins x 64 classes x 8 B = 65536, over both 16 KiB and 32 KB
    var c_big = compute_shared_memory_config(
        Int32(128), Int32(64), 8, 4, 40, apple, cdf, 32
    )
    if not c_big.use_global_memory_histogram:
        fails += 1
        print("  arm D: a 64-class 128-bin histogram must take the GLOBAL arm")
    if c_big.histogram_dynamic_smem_size != 0:
        fails += 1
        print("  arm D: the global arm must report 0 dynamic smem")

    # THE POINT OF THE 16 KiB ROW: find the class count where THEIR tuned
    # constant flips the branch, and confirm it flips there and not at the
    # hardware limit. With 128 bins and 8-byte bins, 16 KiB / (128*8) = 16.
    var flipped_at = -1
    for nc in range(1, 40):
        var c = compute_shared_memory_config(
            Int32(128), Int32(nc), 8, 4, 40, apple, cdf, 32
        )
        if c.use_global_memory_histogram:
            flipped_at = nc
            break
    if flipped_at != 16:
        fails += 1
        print(
            "  arm D: expected the shared->global flip at 16 classes (their"
            " 16 KiB constant), got", flipped_at,
        )
    else:
        # and it must be the CONSTANT doing it, not the hardware: raising
        # available_smem must not move the flip.
        var c2 = compute_shared_memory_config(
            Int32(128), Int32(16), 8, 4, 40, 1024 * 1024, cdf, 32
        )
        if not c2.use_global_memory_histogram:
            fails += 1
            print(
                "  arm D: with 1 MB of shared memory the flip MOVED, so the"
                " dispatch is keying on hardware and not on their constant"
            )

    # their ASSERT at `:540-541`
    var raised = False
    try:
        _ = compute_shared_memory_config(
            Int32(128), Int32(2), 8, 4, 40, 64, 1024, 32
        )
    except:
        raised = True
    if not raised:
        fails += 1
        print("  arm D: too little smem for split bookkeeping must RAISE")

    # -- n_sampled_cols_for's truncation knife-edge (`:240`)
    if n_sampled_cols_for(Float32(0.0), 40) != 1:
        fails += 1
        print("  arm D: max_features 0 must clamp to 1 column, not 0")
    if n_sampled_cols_for(Float32(1.0), 40) != 40:
        fails += 1
        print("  arm D: max_features 1.0 must give every column")
    # sqrt(40)/40 = 0.15811388 -> 6.3245 -> truncates to 6
    if n_sampled_cols_for(Float32(6.3245553) / Float32(40.0), 40) != 6:
        fails += 1
        print("  arm D: sqrt(40) must TRUNCATE to 6, not round to 6 by luck")

    # -- the round schedule (`:420-421`, `:438-440`)
    if max_sampling_rounds_for(40, 6) != 7:
        fails += 1
        print("  arm D: ceil(40/6) must be 7 rounds")
    if sampled_cols_in_round(40, 6, 6) != 4:
        fails += 1
        print("  arm D: the LAST round of 40 columns by 6 must be short (4)")
    var seen = 0
    for r in range(max_sampling_rounds_for(40, 6)):
        seen += sampled_cols_in_round(40, 6, r)
    if seen != 40:
        fails += 1
        print(
            "  arm D: the rounds must cover every column exactly once; got",
            seen, "of 40",
        )

    if fails == 0:
        print(
            "  arm D OK: maxNodes across the depth-13 cliff, both workspace"
            " totals against a second transcription, every offset aligned"
            " and in order, the shared->global flip at 16 classes and"
            " UNMOVED by a 1 MB budget, and the round schedule covering all"
            " 40 columns exactly once"
        )
    return fails


def arm_e_workload() raises -> Int:
    """`builder.cuh:393-407` on a ragged batch, per entry."""
    var items = List[NodeWorkItem]()
    # counts chosen around TPB_DEFAULT = 128: empty, one, exactly one block,
    # one over, and several blocks.
    var counts = [0, 1, 128, 129, 500]
    for i in range(len(counts)):
        items.append(NodeWorkItem(i, Int32(2), InstanceRange(0, counts[i])))

    # `update_workload_info` now writes INTO an array in place, as
    # upstream fills its pinned `h_workload_info` (`builder.cuh:401`).
    # The array is poisoned first, so "wrote exactly n entries" is
    # checked as "entry n is still poison" -- stronger than the old
    # `len(wl)` test, which could not see an over-write.
    var want_blocks = [1, 1, 1, 2, 4]  # max(ceildiv(c,128), 1)
    var want_total = 1 + 1 + 1 + 2 + 4
    var wl = List[WorkloadInfo]()
    for _ in range(want_total + 1):
        wl.append(WorkloadInfo(Int32(-1), Int32(-1), Int32(-1)))
    var n = update_workload_info(items, wl.unsafe_ptr())

    var fails = 0
    if n != want_total:
        fails += 1
        print("  arm E: n_blocks_dimx", n, "want", want_total)
    if Int(wl[want_total].nodeid) != -1:
        fails += 1
        print("  arm E: entry", want_total, "was written; expected the"
              " poison tail untouched")

    var k = 0
    for i in range(len(counts)):
        for b in range(want_blocks[i]):
            if k >= want_total:
                break
            var e = wl[k]
            if (
                Int(e.nodeid) != i
                or Int(e.offset_blockid) != b
                or Int(e.num_blocks) != want_blocks[i]
            ):
                fails += 1
                print(
                    "  arm E entry", k, "= (", e.nodeid, e.offset_blockid,
                    e.num_blocks, ") want (", i, b, want_blocks[i], ")",
                )
            k += 1
    # `[[mojo-buffer-freed-at-last-use]]`: `.unsafe_ptr()` above must not
    # be `wl`'s last use while the writes land.
    _ = wl^

    # the bound their constructor allocates must actually hold
    if max_blocks_dimx_for(Int32(len(counts)), 630) < n:
        fails += 1
        print("  arm E: max_blocks_dimx_for underestimates the real grid")

    if fails == 0:
        print(
            "  arm E OK:", want_total, "entries over a ragged batch, per"
            " entry, including the EMPTY node that still gets one block"
        )
    return fails


def _build_order_fixture[
    sab: Int = 0
]() raises -> NodeQueue[DT, sab]:
    """A tree whose shape depends on the FRONTIER ORDER.

    `_build_ragged` cannot see the order, because one of its two batch-2
    nodes has an invalid split and contributes no children -- so FIFO and
    LIFO append the same nodes in the same places. That was a real hole and
    a sabotage found it.

    Here BOTH batch-2 nodes split, with DIFFERENT left counts, so the order
    they are processed in decides which child lands at tree index 3 and
    which at index 5.
    """
    var q = NodeQueue[DT, sab](_params(8, -1, 2, 16), 64, 100, Int32(1))
    var b1 = q.pop()
    var s1 = List[SplitSummary[DT]]()
    s1.append(_split(True, 0, 1.0, 40))
    q.push(b1, s1)

    var b2 = q.pop()
    var s2 = List[SplitSummary[DT]]()
    for i in range(len(b2)):
        # left count keyed to the node's own size, so the two differ
        s2.append(_split(True, i + 1, Float32(i) + 2.0, 7 + i * 11))
    q.push(b2, s2)
    return q^


def arm_f_sabotage() raises -> Int:
    """One corruption per mechanism, in the shipped path, each predicted."""
    var fails = 0
    var base = _build_ragged()

    # (i) DROP the max_leaves budget test. Predicted: MOVES -- more nodes,
    #     because the batch keeps splitting past the budget. (The first
    #     version of this arm swapped `break` for `continue` and measured
    #     nothing; that was correct and is recorded in builder.mojo.)
    var qc = NodeQueue[DT, 1](_params(8, 3, 2, 16), 64, 100, Int32(1))
    var b1 = qc.pop()
    var s1 = List[SplitSummary[DT]]()
    s1.append(_split(True, 0, 1.0, 40))
    qc.push(b1, s1)
    var b2 = qc.pop()
    var s2 = List[SplitSummary[DT]]()
    s2.append(_split(True, 1, 2.0, 10))
    s2.append(_split(True, 2, 3.0, 20))
    qc.push(b2, s2)
    if len(qc.tree.sparsetree) == 5:
        fails += 1
        print(
            "  arm F(i) FAILED: dropping the max_leaves budget still gave 5"
            " nodes, so arm C cannot see the budget at all"
        )
    else:
        print(
            "  arm F(i) OK: dropping the max_leaves budget gave",
            len(qc.tree.sparsetree), "nodes where the budget gives 5",
        )

    # (ii) instance range from global_nLeft instead of local_nLeft.
    #      Predicted: NO movement, because on a single device the two are
    #      equal by construction. Recorded as a GAP, not a pass.
    var q2 = _build_ragged[2]()
    var moved2 = 0
    for i in range(min(len(q2.node_instances_), len(base.node_instances_))):
        if (
            q2.node_instances_[i].begin != base.node_instances_[i].begin
            or q2.node_instances_[i].count != base.node_instances_[i].count
        ):
            moved2 += 1
    if moved2 != 0:
        fails += 1
        print(
            "  arm F(ii) FAILED: swapping local_nLeft for global_nLeft moved",
            moved2,
            "ranges, but this fixture sets them equal, so something else"
            " differs between those paths",
        )
    else:
        print(
            "  arm F(ii) OK: swapping local_nLeft for global_nLeft moved 0"
            " ranges, as predicted -- RECORDED AS A GAP: nothing here can"
            " tell the two counts apart, and only a multi-rank build could."
        )

    # (iii) FIFO -> LIFO. Predicted: MOVES -- a different tree, same splits.
    #       Run on _build_order_fixture, NOT _build_ragged: the ragged one
    #       has an invalid split in the batch, so both orders append the same
    #       nodes in the same places and the arm was blind. The sabotage
    #       found that hole.
    var ord_base = _build_order_fixture()
    var q3 = _build_order_fixture[3]()
    var moved3 = 0
    if len(q3.tree.sparsetree) != len(ord_base.tree.sparsetree):
        moved3 += 1
    else:
        for i in range(len(ord_base.tree.sparsetree)):
            if (
                q3.tree.sparsetree[i].InstanceCount()
                != ord_base.tree.sparsetree[i].InstanceCount()
            ):
                moved3 += 1
    if moved3 == 0:
        fails += 1
        print(
            "  arm F(iii) FAILED: popping from the BACK produced the same"
            " tree, so this fixture cannot see the frontier order"
        )
    else:
        print(
            "  arm F(iii) OK: popping from the back moved the tree in",
            moved3, "places -- the deque order is load-bearing",
        )

    # (iv) drop `max(..., 1)` in updateWorkloadInfo. Predicted: the EMPTY
    #      node loses its block and the grid is short by exactly one.
    var items = List[NodeWorkItem]()
    var counts = [0, 1, 128, 129, 500]
    for i in range(len(counts)):
        items.append(NodeWorkItem(i, Int32(2), InstanceRange(0, counts[i])))
    var wl4 = List[WorkloadInfo]()
    for _ in range(16):
        wl4.append(WorkloadInfo(Int32(-1), Int32(-1), Int32(-1)))
    var n4 = update_workload_info[4](items, wl4.unsafe_ptr())
    _ = wl4^
    if n4 != 8:
        fails += 1
        print(
            "  arm F(iv) FAILED: dropping max(...,1) gave", n4,
            "blocks, expected exactly 8 (one fewer than 9)",
        )
    else:
        print(
            "  arm F(iv) OK: dropping max(...,1) dropped the empty node's"
            " block, 9 -> 8, so that node would never be visited"
        )

    return fails


def main() raises:
    print("builder_check: ensemble/decisiontree/batched_levelalgo/builder.mojo")
    print("  mirroring builder.cuh @ cuml v26.08.00 265b9da6")
    var fails = 0
    fails += arm_a_is_expandable()
    fails += arm_b_push()
    fails += arm_c_max_leaves_break()
    fails += arm_d_workspace_and_smem()
    fails += arm_e_workload()
    fails += arm_f_sabotage()
    if fails == 0:
        print("builder_check: ALL OK")
    else:
        raise Error("builder_check: " + String(fails) + " failure(s)")
