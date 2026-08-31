# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does DEVIATION 203's three-pass partition give the one-block kernel's answer?

    pixi run mojo run -I . extratrees/original/partition_multiblock_check.mojo

NO CUML COUNTERPART -- this is a check. It covers the four kernels of
`extratrees/derived/decisiontree/batched_levelalgo/kernels/
partition_multiblock.mojo`:

    partition_count_kernel   partition_scan_kernel
    partition_scatter_kernel partition_writeback_kernel

which replace `node_split_kernel` (their `nodeSplitKernel`,
`builder_kernels_impl.cuh:89-107`, around `partitionSamples` `:43-88`) in the
device builder. DEVIATION 203 states the reason: their grid is ONE BLOCK PER
NODE, and the root level has one node, so the root's partition is a single
threadgroup walking every row of the dataset.

WHAT IS COMPARED, AND WHY IT IS NOT `assert_equal` ON THE ARRAY
---------------------------------------------------------------
`node_split_kernel` IS STILL IN THE TREE and it is the oracle here -- both
partitions are launched on the device, on two copies of the SAME `row_ids`,
with the same work items and the same splits. But DEVIATION 203 says the two
do NOT agree slot for slot: theirs is whatever its pairwise misfit swaps
leave, this one is stable by block. So a slot-for-slot comparison would be
red on a CORRECT kernel, and the property that actually matters is the one
everything downstream reads:

  A. per node, the SET of row ids on the left is the oracle's set, and
     likewise the right -- compared as MULTISETS (a counting tally per side,
     which also catches a duplicated row), never as sequences. The number of
     slots on which the two ORDERS differ is MEASURED and printed, because
     "the order is allowed to differ" is only interesting if it actually
     does; if that number ever came back 0 this file would be comparing two
     runs of the same arrangement and arm A would be weaker than it looks.
  B. per node, every slot in `[begin, begin + n_left)` holds a row with
     `col[row] <= quesval` and every slot after it a row with `> quesval`.
     This is the DEFINITION, not a second transcription -- `partition_check`
     and `partition_leaf_kernel_check` assert the same five properties of
     the one-block form, and two transcriptions share their bugs.
  C. every slot OUTSIDE the partitioned ranges is byte-for-byte what it was,
     including every slot of a node whose split `split_not_valid` refused.
     `row_ids` is RESIDENT on the device across the whole fit; a partition
     that scribbles outside its node corrupts a sibling's rows.
  D. the whole array is still a permutation of what it was.
  E. `blk_left` and `blk_off` themselves, against a host tally, so that "the
     scan ran" is evidence and not an inference from the final answer. The
     count of nonzero cross-block offsets is asserted to be > 0: it is
     exactly the quantity `PART_MB_SAB_NO_SCAN` destroys, and if it were 0
     that arm could not move a single row.

`n_left` COMES FROM THE SPLIT, as it does in the real pipeline -- the score
pass supplies it and the partition is trusted to produce exactly that many
rows on the left. It is tallied here by a plain host loop over the node's
range through `row_ids`, which is how neither kernel computes anything.

THE FIXTURE, AND WHAT EACH PIECE IS FOR
----------------------------------------
Nine nodes with DISJOINT ranges that do NOT tile `row_ids`: there are wide
gaps between them, so a kernel that walked from the previous node's end would
touch rows belonging to no node, and (C) has something to see. Against
TPB = 128 the row counts are

    3, 127, 128, 129, 700, 96, 300, 0, 1

which is 1, 1, 1, 2, 6, 1, 3, 1 and 1 blocks -- SINGLE-BLOCK nodes, a node
that is exactly one block, a node one row over, and one node spanning SIX
blocks, which is the shape `PART_MB_SAB_SINGLE_BLOCK` and
`PART_MB_SAB_NO_SCAN` need in order to be able to move anything at all. Only
128 is a multiple of TPB; every other node has a ragged last block.

THREE of the nine are refused by `split_not_valid`, by three different
clauses: a BELOW-THRESHOLD node (the `MIN_FINITE` invalid-candidate
sentinel -- since DEVIATION 216 zero gain PASSES, sklearn's boundary, so
the sentinel is the gain refusal the fit path actually produces), a node
with NO ROWS, and a ONE-ROW node (which cannot leave `min_samples_leaf = 1`
on both sides). All three must come out untouched.

The values are HASHED and scattered, never `i % k` and never monotone: a
monotone column makes every partition a contiguous prefix, which is the one
shape that needs no movement at all. Column 1 quantizes onto nine levels
while keeping the PLACEMENT scattered, and the nodes that read it are given
a threshold sitting exactly ON a level -- `partition_check.mojo`'s scar
records that with distinct values NO ROW EVER EQUALS A THRESHOLD, the `<=`
branch is dead, and both boundary sabotages stayed green until such a column
was added. Each node's threshold is the k-th smallest of ITS OWN rows, so
`n_left` is never 0 and never `count`; the fixture asserts that rather than
hoping for it.

SABOTAGE (rule 8: one per MECHANISM, each states its prediction BEFORE it
runs, each must turn the answer RED)
-----------------------------------------------------------------------
Every arm is a kernel ARGUMENT, so all four arms run the SAME binary that
ships. A green sabotage here is a defect in THIS FIXTURE -- reported as one,
never absorbed by weakening an assertion.

  PART_MB_SAB_NO_VALID_GUARD  -- predicts (C) red and ONLY in the refused
      nodes' ranges: the guard's other side.
  PART_MB_SAB_SINGLE_BLOCK    -- predicts (A)/(D) red and ONLY inside the
      MULTI-BLOCK nodes: every block claims `offset_blockid == 0`, so the
      six blocks of the 700-row node all read and rewrite chunk 0.
  PART_MB_SAB_NO_SCAN         -- predicts (A)/(D) red and ONLY inside the
      MULTI-BLOCK nodes: with a zero cross-block prefix every block packs
      its left rows at `begin`, which is indistinguishable from a working
      scan for a ONE-block node and is exactly why arm E asserts that some
      block's offset is nonzero.

The scatter's destination is seeded with a sentinel that is not a legal row
id, so a slot NO block wrote is visible as a wild value rather than as a
plausible leftover.

A SABOTAGED ARM'S COUNTS ARE NOT REPRODUCIBLE AND THAT IS NOT DRIFT. Two of
the three sabotages make several blocks write the SAME slot, and there is no
ordering between them, so the number of moved slots wobbles by a few from run
to run (measured: 975 / 875 / 830 for the tiling arm over three runs). What is
stable, and what is asserted, is that the answer is WRONG and that every moved
slot lies inside the nodes the prediction named. The SHIPPING arm has no such
race and is bit-stable: arms A-D come back 0 and the measured order difference
comes back 1,364 every time.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    size_of,
)
from max.gpu.host import DeviceContext

from extratrees.derived.decisiontree.batched_levelalgo.split import Split
from extratrees.derived.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    WorkloadInfo,
    split_not_valid,
)
from extratrees.derived.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    build_workload_info,
    node_split_kernel,
    PART_SAB_NONE,
    PARTITION_SKIPPED,
    PARTITION_UNVISITED,
    TPB_DEFAULT,
)
from extratrees.derived.decisiontree.batched_levelalgo.kernels.partition_multiblock import (
    partition_count_kernel,
    partition_scan_kernel,
    partition_scatter_kernel,
    partition_writeback_kernel,
    PART_MB_SAB_NO_SCAN,
    PART_MB_SAB_NO_VALID_GUARD,
    PART_MB_SAB_NONE,
    PART_MB_SAB_SINGLE_BLOCK,
)


comptime TPB = TPB_DEFAULT
"""128, the width the shipping builder launches all four passes with."""

comptime N_ROWS = 4096
"""Slots in `row_ids`. The nodes below cover 1,484 of them; the rest are the
gaps property (C) watches, and they are wide enough that even a SABOTAGED
scatter cannot reach out of one node's range into another's -- a sabotage
that made two nodes race would be red for a reason nobody could name."""

comptime N_COLS = 2
"""Column 0 is hashed and essentially distinct; column 1 quantizes onto nine
levels with scattered placement, so a threshold ON a level ties many rows."""

comptime N_LEVELS = 9
comptime LEVEL_STEP: Float32 = 250.0
comptime LEVEL_BASE: Float32 = -1000.0

comptime ALT_SENTINEL: Int32 = -1
"""Seed for the scatter's destination. NOT a legal row id, so a slot that no
block wrote is a wild value rather than a plausible leftover."""

comptime MIN_IMPURITY_DECREASE: Float32 = 0.0
comptime MIN_SAMPLES_LEAF: Int32 = 1
"""The shipping defaults. Both guard clauses bite in this fixture."""


def mix32(x: UInt32) -> UInt32:
    """`partition_check.mojo`'s mixer, unchanged, so a value here can be
    reasoned about from there."""
    var h = x
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


def _vendor() -> String:
    """Which GPU family this run used. A HOST check (`has_*`), not a target
    check (`is_*`): the kernels are one GPU-agnostic source and nothing in
    them branches on the vendor."""
    comptime if has_apple_gpu_accelerator():
        return "Apple"
    elif has_nvidia_gpu_accelerator():
        return "NVIDIA"
    elif has_amd_gpu_accelerator():
        return "AMD"
    else:
        return "unknown"


def _shuffled(n: Int, seed: UInt32) -> List[Int32]:
    """A Fisher-Yates shuffle over a deterministic hash, so a slot index and
    the row id it holds are different numbers for essentially every slot."""
    var ids = List[Int32]()
    for r in range(n):
        ids.append(Int32(r))
    for i in range(n - 1, 0, -1):
        var j = Int(mix32(seed ^ UInt32(i)) % UInt32(i + 1))
        var t = ids[i]
        ids[i] = ids[j]
        ids[j] = t
    return ids^


def _feature_matrix() -> List[Float32]:
    """COLUMN MAJOR, `dataset.h:24`: `data[col * M + row]`."""
    var flat = List[Float32](length=N_COLS * N_ROWS, fill=Float32(0.0))
    for r in range(N_ROWS):
        var h0 = mix32(0xBEEF ^ UInt32(r * 2654435761))
        flat[0 * N_ROWS + r] = Float32(Int(h0 % 2000000)) / 1000.0 - 1000.0
        var h1 = mix32(0xC0DE ^ UInt32(r * 40503 + 7))
        var lvl = Int(h1 % UInt32(N_LEVELS))
        flat[1 * N_ROWS + r] = Float32(lvl) * LEVEL_STEP + LEVEL_BASE
    return flat^


@fieldwise_init
struct NodeSpec(Copyable, Movable):
    """One node of the batch, before its threshold and `n_left` are tallied.

    `kth` selects the threshold as the k-th SMALLEST of this node's own rows,
    which is what keeps `n_left` off both extremes without tuning a constant
    to the hash."""

    var begin: Int
    var count: Int
    var col: Int
    var kth: Int
    var gain: Float32
    var note: String


def _node_specs() -> List[NodeSpec]:
    """The batch. Row counts 3, 127, 128, 129, 700, 96, 300, 0, 1 against
    TPB = 128, and the begins leave gaps of hundreds of slots between the
    ranges."""
    var out = List[NodeSpec]()
    out.append(NodeSpec(5, 3, 0, 1, 1.0, "3 rows, one block, the degenerate size"))
    out.append(NodeSpec(16, 127, 0, 63, 1.0, "127 rows, ONE BLOCK SHORT"))
    out.append(NodeSpec(200, 128, 1, 42, 1.0, "128 rows, EXACTLY one block, TIED column"))
    out.append(NodeSpec(600, 129, 0, 64, 1.0, "129 rows, TWO blocks, one row over"))
    out.append(NodeSpec(1200, 700, 0, 349, 1.0, "700 rows, SIX blocks, ragged last"))
    # DEVIATION 216: zero gain now PASSES the gate (sklearn's boundary;
    # builder_check and split_check pin that side), so the gain refusal this
    # node keeps covered is the fit path's real one -- the MIN_FINITE
    # invalid-candidate sentinel.
    out.append(
        NodeSpec(
            2400, 96, 0, 47, Float32.MIN_FINITE,
            "96 rows, MIN_FINITE gain -> refused",
        )
    )
    out.append(NodeSpec(2700, 300, 1, 99, 1.0, "300 rows, THREE blocks, TIED column"))
    out.append(NodeSpec(3600, 0, 0, 0, 1.0, "0 rows -> refused (n_left < msl)"))
    out.append(NodeSpec(3700, 1, 0, 0, 1.0, "1 row -> refused (one side < msl)"))
    return out^


def _kth_smallest(
    flat: List[Float32],
    row_ids: List[Int32],
    begin: Int,
    count: Int,
    col: Int,
    k: Int,
) -> Float32:
    """The k-th smallest feature value among this node's rows, by insertion
    sort over a copy. Deliberately a different structure from anything either
    partition does."""
    if count == 0:
        return Float32(0.0)
    var vals = List[Float32]()
    for s in range(begin, begin + count):
        vals.append(flat[col * N_ROWS + Int(row_ids[s])])
    for i in range(1, len(vals)):
        var v = vals[i]
        var j = i - 1
        while j >= 0 and vals[j] > v:
            vals[j + 1] = vals[j]
            j -= 1
        vals[j + 1] = v
    return vals[k]


def _build_batch(
    flat: List[Float32], row_ids: List[Int32]
) -> Tuple[List[NodeWorkItem], List[Split]]:
    """Work items and splits, with `n_left` tallied by a plain host loop over
    the range THROUGH `row_ids` -- which is how neither kernel computes
    anything."""
    var specs = _node_specs()
    var items = List[NodeWorkItem]()
    var splits = List[Split]()
    for b in range(len(specs)):
        var s = specs[b].copy()
        var q = _kth_smallest(flat, row_ids, s.begin, s.count, s.col, s.kth)
        var n_left = 0
        for p in range(s.begin, s.begin + s.count):
            if flat[s.col * N_ROWS + Int(row_ids[p])] <= q:
                n_left += 1
        # `idx` is the node's id IN THE TREE and deliberately does not start
        # at 0: no kernel here reads it, and that is worth stating.
        items.append(
            NodeWorkItem(
                Int32(300 + b),
                Int32(5),
                InstanceRange(Int32(s.begin), Int32(s.count)),
            )
        )
        splits.append(Split(q, Int32(s.col), s.gain, Int32(n_left)))
    return (items^, splits^)


@fieldwise_init
struct MbRun(Copyable, Movable):
    """One multi-block launch's outputs, copied back."""

    var row_ids: List[Int32]
    var blk_left: List[Int32]
    var blk_off: List[Int32]


@fieldwise_init
struct Verdict(Copyable, Movable):
    """Assertions A-D, counted rather than raised on the first failure, so a
    sabotage arm can report HOW MUCH it moved."""

    var cells: Int
    var bad_a: Int
    var bad_b: Int
    var bad_c: Int
    var bad_d: Int

    def total(self) -> Int:
        return self.bad_a + self.bad_b + self.bad_c + self.bad_d


def _side_multiset_diff(
    after: List[Int32], oracle: List[Int32], lo: Int, hi: Int
) -> Int:
    """How many slots of `[lo, hi)` break the MULTISET equality between the
    two partitions. A counting tally, not a sort: it also catches a row that
    appears twice, which a set comparison would forgive."""
    var tally = List[Int](length=N_ROWS, fill=0)
    var bad = 0
    for s in range(lo, hi):
        var a = Int(after[s])
        var o = Int(oracle[s])
        if a < 0 or a >= N_ROWS:
            bad += 1
        else:
            tally[a] += 1
        if o < 0 or o >= N_ROWS:
            bad += 1
        else:
            tally[o] -= 1
    for s in range(lo, hi):
        var o = Int(oracle[s])
        if o >= 0 and o < N_ROWS and tally[o] != 0:
            bad += 1
            tally[o] = 0
        var a = Int(after[s])
        if a >= 0 and a < N_ROWS and tally[a] != 0:
            bad += 1
            tally[a] = 0
    return bad


def _verify(
    flat: List[Float32],
    before: List[Int32],
    after: List[Int32],
    oracle: List[Int32],
    items: List[NodeWorkItem],
    splits: List[Split],
    skipped: List[Bool],
) -> Verdict:
    """A, B, C and D, over the whole array."""
    var cells = 0
    var bad_a = 0
    var bad_b = 0
    var bad_c = 0
    var bad_d = 0
    var inside = List[Bool](length=N_ROWS, fill=False)

    for b in range(len(items)):
        var begin = Int(items[b].instances.begin)
        var count = Int(items[b].instances.count)
        for s in range(begin, begin + count):
            inside[s] = True

        if skipped[b]:
            # (C) a node the guard refused keeps the slots it already had.
            for s in range(begin, begin + count):
                cells += 1
                if after[s] != before[s]:
                    bad_c += 1
            continue

        var n_left = Int(splits[b].n_left)
        var col = Int(splits[b].colid)
        var q = splits[b].quesval

        # (B) the predicate, slot by slot.
        for s in range(begin, begin + count):
            cells += 1
            var rid = Int(after[s])
            if rid < 0 or rid >= N_ROWS:
                bad_b += 1
                continue
            var v = flat[col * N_ROWS + rid]
            if s < begin + n_left:
                if not (v <= q):
                    bad_b += 1
            else:
                if not (v > q):
                    bad_b += 1

        # (A) the two SIDES as multisets against the one-block kernel.
        bad_a += _side_multiset_diff(after, oracle, begin, begin + n_left)
        bad_a += _side_multiset_diff(
            after, oracle, begin + n_left, begin + count
        )
        cells += count

    # (C) nothing outside any range moved.
    for s in range(N_ROWS):
        if not inside[s]:
            cells += 1
            if after[s] != before[s]:
                bad_c += 1

    # (D) the whole array is still a permutation.
    var tally = List[Int](length=N_ROWS, fill=0)
    for s in range(N_ROWS):
        tally[Int(before[s])] += 1
        var rid = Int(after[s])
        if rid < 0 or rid >= N_ROWS:
            bad_d += 1
        else:
            tally[rid] -= 1
    for r in range(N_ROWS):
        cells += 1
        if tally[r] != 0:
            bad_d += 1

    return Verdict(cells, bad_a, bad_b, bad_c, bad_d)


def _launch_oracle(
    ctx: DeviceContext,
    flat: List[Float32],
    row_ids: List[Int32],
    items: List[NodeWorkItem],
    splits: List[Split],
) raises -> Tuple[List[Int32], List[Int32]]:
    """`node_split_kernel[TPB]`, ONE BLOCK PER NODE, on its own copy of
    `row_ids`. This is the shape DEVIATION 203 replaces and it stays in the
    tree precisely so that it can be this file's oracle.

    Returns `(row_ids, n_iters)`. `n_iters` is seeded with
    `PARTITION_UNVISITED`, which is not a legal outcome, so a node no block
    served is visible as a REACH failure and not as a plausible zero
    (DEVIATION 180)."""
    var n_items = len(items)

    var d_row_ids = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_iters = ctx.enqueue_create_buffer[DType.int32](n_items)
    var d_swaps = ctx.enqueue_create_buffer[DType.int32](n_items)
    var d_data = ctx.enqueue_create_buffer[DType.float32](N_COLS * N_ROWS)
    var d_items = ctx.enqueue_create_buffer[DType.uint8](
        n_items * size_of[NodeWorkItem]()
    )
    var d_splits = ctx.enqueue_create_buffer[DType.uint8](
        n_items * size_of[Split]()
    )

    var h_row_ids = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var h_data = ctx.enqueue_create_host_buffer[DType.float32](N_COLS * N_ROWS)
    var h_items = ctx.enqueue_create_host_buffer[DType.uint8](
        n_items * size_of[NodeWorkItem]()
    )
    var h_splits = ctx.enqueue_create_host_buffer[DType.uint8](
        n_items * size_of[Split]()
    )
    ctx.synchronize()

    for i in range(N_ROWS):
        h_row_ids.unsafe_ptr().unsafe_store(i, row_ids[i])
    for i in range(N_COLS * N_ROWS):
        h_data.unsafe_ptr().unsafe_store(i, flat[i])
    var ip = h_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
    for i in range(n_items):
        ip[unsafe_offset=i] = items[i]
    var sp = h_splits.unsafe_ptr().unsafe_bitcast[Split]()
    for i in range(n_items):
        sp[unsafe_offset=i] = splits[i]

    ctx.enqueue_copy(dst_buf=d_row_ids, src_ptr=h_row_ids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_data, src_ptr=h_data.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_items, src_ptr=h_items.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_splits, src_ptr=h_splits.unsafe_ptr())
    ctx.enqueue_memset(d_iters, PARTITION_UNVISITED)
    ctx.enqueue_memset(d_swaps, Int32(0))
    ctx.synchronize()

    ctx.enqueue_function[node_split_kernel[TPB]](
        d_row_ids.unsafe_ptr(),
        d_iters.unsafe_ptr(),
        d_swaps.unsafe_ptr(),
        d_data.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_splits.unsafe_ptr().unsafe_bitcast[Split](),
        Int32(N_ROWS),
        MIN_IMPURITY_DECREASE,
        MIN_SAMPLES_LEAF,
        PART_SAB_NONE,
        grid_dim=(n_items, 1, 1),
        block_dim=(TPB, 1, 1),
    )

    var o_row_ids = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var o_iters = ctx.enqueue_create_host_buffer[DType.int32](n_items)
    ctx.enqueue_copy(dst_buf=o_row_ids, src_buf=d_row_ids)
    ctx.enqueue_copy(dst_buf=o_iters, src_buf=d_iters)
    ctx.synchronize()

    var ids = List[Int32]()
    for i in range(N_ROWS):
        ids.append(o_row_ids.unsafe_ptr().unsafe_load(i))
    var it = List[Int32]()
    for i in range(n_items):
        it.append(o_iters.unsafe_ptr().unsafe_load(i))
    _ = d_swaps
    return (ids^, it^)


def _launch_multiblock(
    ctx: DeviceContext,
    flat: List[Float32],
    row_ids: List[Int32],
    items: List[NodeWorkItem],
    splits: List[Split],
    blk_base: List[Int32],
    n_blocks: Int,
    wl: List[WorkloadInfo],
    sabotage: Int32,
) raises -> MbRun:
    """The four passes of DEVIATION 203, in the order and with the grids
    `builder.mojo` launches them with.

    `row_ids` is re-uploaded from the pristine host copy for every arm; an
    arm that inherited the previous arm's output would be checking a
    different fixture."""
    var n_items = len(items)

    var d_row_ids = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_row_alt = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_blk_left = ctx.enqueue_create_buffer[DType.int32](n_blocks)
    var d_blk_off = ctx.enqueue_create_buffer[DType.int32](n_blocks)
    var d_blk_base = ctx.enqueue_create_buffer[DType.int32](n_items)
    var d_data = ctx.enqueue_create_buffer[DType.float32](N_COLS * N_ROWS)
    var d_items = ctx.enqueue_create_buffer[DType.uint8](
        n_items * size_of[NodeWorkItem]()
    )
    var d_splits = ctx.enqueue_create_buffer[DType.uint8](
        n_items * size_of[Split]()
    )
    var d_wl = ctx.enqueue_create_buffer[DType.uint8](
        n_blocks * size_of[WorkloadInfo]()
    )

    var h_row_ids = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var h_blk_base = ctx.enqueue_create_host_buffer[DType.int32](n_items)
    var h_data = ctx.enqueue_create_host_buffer[DType.float32](N_COLS * N_ROWS)
    var h_items = ctx.enqueue_create_host_buffer[DType.uint8](
        n_items * size_of[NodeWorkItem]()
    )
    var h_splits = ctx.enqueue_create_host_buffer[DType.uint8](
        n_items * size_of[Split]()
    )
    var h_wl = ctx.enqueue_create_host_buffer[DType.uint8](
        n_blocks * size_of[WorkloadInfo]()
    )
    ctx.synchronize()

    for i in range(N_ROWS):
        h_row_ids.unsafe_ptr().unsafe_store(i, row_ids[i])
    for i in range(n_items):
        h_blk_base.unsafe_ptr().unsafe_store(i, blk_base[i])
    for i in range(N_COLS * N_ROWS):
        h_data.unsafe_ptr().unsafe_store(i, flat[i])
    var ip = h_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
    for i in range(n_items):
        ip[unsafe_offset=i] = items[i]
    var sp = h_splits.unsafe_ptr().unsafe_bitcast[Split]()
    for i in range(n_items):
        sp[unsafe_offset=i] = splits[i]
    var wp = h_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo]()
    for i in range(n_blocks):
        wp[unsafe_offset=i] = wl[i]

    ctx.enqueue_copy(dst_buf=d_row_ids, src_ptr=h_row_ids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_blk_base, src_ptr=h_blk_base.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_data, src_ptr=h_data.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_items, src_ptr=h_items.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_splits, src_ptr=h_splits.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wl, src_ptr=h_wl.unsafe_ptr())
    # The scatter's destination, seeded with a value that is NOT a row id: a
    # slot no block wrote comes back wild instead of plausible.
    ctx.enqueue_memset(d_row_alt, ALT_SENTINEL)
    ctx.enqueue_memset(d_blk_left, Int32(0))
    ctx.enqueue_memset(d_blk_off, Int32(0))
    ctx.synchronize()

    ctx.enqueue_function[partition_count_kernel[TPB]](
        d_blk_left.unsafe_ptr(),
        d_row_ids.unsafe_ptr(),
        d_data.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        d_splits.unsafe_ptr().unsafe_bitcast[Split](),
        Int32(N_ROWS),
        MIN_IMPURITY_DECREASE,
        MIN_SAMPLES_LEAF,
        sabotage,
        grid_dim=(n_blocks, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.enqueue_function[partition_scan_kernel[TPB]](
        d_blk_off.unsafe_ptr(),
        d_blk_left.unsafe_ptr(),
        d_blk_base.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_splits.unsafe_ptr().unsafe_bitcast[Split](),
        MIN_IMPURITY_DECREASE,
        MIN_SAMPLES_LEAF,
        sabotage,
        grid_dim=(n_items, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.enqueue_function[partition_scatter_kernel[TPB]](
        d_row_alt.unsafe_ptr(),
        d_row_ids.unsafe_ptr(),
        d_blk_off.unsafe_ptr(),
        d_data.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        d_splits.unsafe_ptr().unsafe_bitcast[Split](),
        Int32(N_ROWS),
        MIN_IMPURITY_DECREASE,
        MIN_SAMPLES_LEAF,
        sabotage,
        grid_dim=(n_blocks, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.enqueue_function[partition_writeback_kernel[TPB]](
        d_row_ids.unsafe_ptr(),
        d_row_alt.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        d_splits.unsafe_ptr().unsafe_bitcast[Split](),
        MIN_IMPURITY_DECREASE,
        MIN_SAMPLES_LEAF,
        sabotage,
        grid_dim=(n_blocks, 1, 1),
        block_dim=(TPB, 1, 1),
    )

    var o_row_ids = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var o_left = ctx.enqueue_create_host_buffer[DType.int32](n_blocks)
    var o_off = ctx.enqueue_create_host_buffer[DType.int32](n_blocks)
    ctx.enqueue_copy(dst_buf=o_row_ids, src_buf=d_row_ids)
    ctx.enqueue_copy(dst_buf=o_left, src_buf=d_blk_left)
    ctx.enqueue_copy(dst_buf=o_off, src_buf=d_blk_off)
    ctx.synchronize()

    var ids = List[Int32]()
    for i in range(N_ROWS):
        ids.append(o_row_ids.unsafe_ptr().unsafe_load(i))
    var bl = List[Int32]()
    var bo = List[Int32]()
    for i in range(n_blocks):
        bl.append(o_left.unsafe_ptr().unsafe_load(i))
        bo.append(o_off.unsafe_ptr().unsafe_load(i))
    return MbRun(ids^, bl^, bo^)


def main() raises:
    comptime assert has_accelerator(), "this check must run on a GPU"

    var failures = 0
    var ctx = DeviceContext()
    print(
        "[device] accelerator present;",
        _vendor(),
        "-- BOTH partitions are enqueued on it, and every arm below runs the"
        " SAME binary",
    )

    var flat = _feature_matrix()
    var row_ids = _shuffled(N_ROWS, 0xABCD)
    var built = _build_batch(flat, row_ids)
    var items = built[0].copy()
    var splits = built[1].copy()
    var specs = _node_specs()
    var n_items = len(items)

    var plan = build_workload_info(items, TPB)
    var n_blocks = plan.n_blocks_dimx
    # Where node `i`'s blocks start in the flattened workload array.
    # `build_workload_info` lays them out contiguously in node order, so this
    # repeats the running sum it performs -- the scan pass needs that base
    # and the device cannot derive it (`builder.mojo`, deviation 203).
    var blk_base = List[Int32]()
    var acc = 0
    for i in range(n_items):
        blk_base.append(Int32(acc))
        var nb = ceildiv(Int(items[i].instances.count), TPB)
        if nb < 1:
            nb = 1
        acc += nb

    var skipped = List[Bool]()
    var n_skipped = 0
    var multi = List[Bool]()
    var n_multi = 0
    var n_single = 0
    var partitioned_slots = 0
    var tied = 0
    var degenerate = 0
    print("")
    print(
        "[fixture]",
        N_ROWS,
        "row_ids slots,",
        N_COLS,
        "columns,",
        n_items,
        "nodes with DISJOINT ranges flattened into",
        n_blocks,
        "blocks at TPB",
        TPB,
    )
    for b in range(n_items):
        var count = Int(items[b].instances.count)
        var begin = Int(items[b].instances.begin)
        var n_left = Int(splits[b].n_left)
        var sk = split_not_valid(
            splits[b],
            MIN_IMPURITY_DECREASE,
            MIN_SAMPLES_LEAF,
            items[b].instances.count,
        )
        skipped.append(sk)
        var nb = ceildiv(count, TPB)
        if nb < 1:
            nb = 1
        multi.append(nb > 1)
        if sk:
            n_skipped += 1
        else:
            partitioned_slots += count
            if nb > 1:
                n_multi += 1
            else:
                n_single += 1
            # A node whose split leaves one side empty exercises nothing.
            if n_left == 0 or n_left == count:
                degenerate += 1
        var t = 0
        for s in range(begin, begin + count):
            if flat[Int(splits[b].colid) * N_ROWS + Int(row_ids[s])] == splits[
                b
            ].quesval:
                t += 1
        tied += t
        var tag = "REFUSED by split_not_valid" if sk else "partitioned"
        print(
            "  node",
            b,
            "[",
            begin,
            "+",
            count,
            ") col",
            Int(splits[b].colid),
            "n_left",
            n_left,
            "->",
            nb,
            "block(s),",
            tag,
            "--",
            specs[b].note,
            "(",
            t,
            "rows sit ON the threshold )",
        )
    print(
        "         ",
        n_single,
        "single-block and",
        n_multi,
        "MULTI-block nodes are partitioned;",
        n_skipped,
        "are refused;",
        partitioned_slots,
        "slots are inside a partitioned range",
    )
    if n_multi < 1 or n_single < 1:
        failures += 1
        print(
            "  FIXTURE DEFECT: both the single-block and the multi-block"
            " shape must be present, or the sabotages have nothing to move"
        )
    if tied < 20:
        failures += 1
        print(
            "  FIXTURE DEFECT: too few rows sit exactly on a threshold -- the"
            " `<=` boundary rule is untested and a disagreement about it"
            " between the two partitions would be invisible"
        )
    if degenerate > 0:
        failures += 1
        print(
            "  FIXTURE DEFECT:",
            degenerate,
            "partitioned node(s) have an EMPTY side, so their partition is a"
            " no-op that any count-only comparison would pass",
        )

    # ---------------------------------------------------------------------
    # The ORACLE: their one-block-per-node kernel, on its own copy.
    # ---------------------------------------------------------------------
    var oracle_run = _launch_oracle(ctx, flat, row_ids, items, splits)
    var oracle = oracle_run[0].copy()
    var oracle_iters = oracle_run[1].copy()
    var unvisited = 0
    var oracle_guard_bad = 0
    for b in range(n_items):
        var it = Int(oracle_iters[b])
        if it == Int(PARTITION_UNVISITED):
            unvisited += 1
        elif it == Int(PARTITION_SKIPPED):
            if not skipped[b]:
                oracle_guard_bad += 1
        elif skipped[b]:
            oracle_guard_bad += 1
    print("")
    print(
        "[oracle] node_split_kernel[",
        TPB,
        "], ONE BLOCK PER NODE, grid",
        n_items,
        "-- it reports",
        unvisited,
        "unvisited node(s) and",
        oracle_guard_bad,
        "disagreement(s) with the host guard",
    )
    if unvisited != 0 or oracle_guard_bad != 0:
        failures += 1
        print(
            "  ORACLE FAILED to reach every node: comparing against it would"
            " compare against a partition that did not happen"
        )
    var self_check = _verify(
        flat, row_ids, oracle, oracle, items, splits, skipped
    )
    if self_check.total() != 0:
        failures += 1
        print(
            "  ORACLE FAILED its own definition check: B",
            self_check.bad_b,
            "C",
            self_check.bad_c,
            "D",
            self_check.bad_d,
        )
    else:
        print(
            "         the oracle satisfies the definition itself (predicate,"
            " untouched outside, permutation) --",
            self_check.cells,
            "cells",
        )

    # ---------------------------------------------------------------------
    # THE SHIPPING ARM.
    # ---------------------------------------------------------------------
    var base = _launch_multiblock(
        ctx,
        flat,
        row_ids,
        items,
        splits,
        blk_base,
        n_blocks,
        plan.info,
        PART_MB_SAB_NONE,
    )
    var v = _verify(flat, row_ids, base.row_ids, oracle, items, splits, skipped)
    print("")
    print("[arms A-D] the three-pass partition against the one-block kernel")
    if v.bad_a == 0:
        print(
            "  ARM A OK: every node's LEFT and RIGHT multiset is the oracle's,"
            " over",
            partitioned_slots,
            "partitioned slots",
        )
    else:
        failures += 1
        print("  ARM A FAILED:", v.bad_a, "multiset disagreements")
    if v.bad_b == 0:
        print(
            "  ARM B OK:",
            partitioned_slots,
            "slots satisfy the predicate -- left `<= quesval`, right `>`",
        )
    else:
        failures += 1
        print("  ARM B FAILED:", v.bad_b, "slots on the wrong side")
    if v.bad_c == 0:
        print(
            "  ARM C OK:",
            N_ROWS - partitioned_slots,
            "slots outside a partitioned range (gaps AND every slot of the",
            n_skipped,
            "refused nodes) are byte-for-byte unchanged",
        )
    else:
        failures += 1
        print("  ARM C FAILED:", v.bad_c, "slots outside the partition moved")
    if v.bad_d == 0:
        print(
            "  ARM D OK: the whole",
            N_ROWS,
            "-slot array is still a permutation of what it was",
        )
    else:
        failures += 1
        print("  ARM D FAILED:", v.bad_d, "rows dropped, duplicated or wild")

    # The ORDER is allowed to differ and is MEASURED, because arm A is only
    # interesting if the two arrangements are actually different ones.
    var order_diff = 0
    for s in range(N_ROWS):
        if base.row_ids[s] != oracle[s]:
            order_diff += 1
    print(
        "  MEASURED:",
        order_diff,
        "of",
        partitioned_slots,
        "partitioned slots hold a DIFFERENT row than the one-block kernel"
        " left there -- deviation 203's reordering, which arm A tolerates"
        " and a slot-for-slot comparison would call a bug",
    )
    if order_diff == 0:
        failures += 1
        print(
            "  FIXTURE DEFECT: the two partitions produced the IDENTICAL"
            " arrangement, so arm A never had to be a multiset comparison and"
            " this file cannot claim to have checked one"
        )

    # ---------------------------------------------------------------------
    # ARM E -- the scan's own output, so that "the scan ran" is evidence.
    # ---------------------------------------------------------------------
    var blk_bad = 0
    var nonzero_off = 0
    var scanned_blocks = 0
    for b in range(n_items):
        var begin = Int(items[b].instances.begin)
        var count = Int(items[b].instances.count)
        var col = Int(splits[b].colid)
        var q = splits[b].quesval
        var nb = ceildiv(count, TPB)
        if nb < 1:
            nb = 1
        var base_i = Int(blk_base[b])
        var running = 0
        for i in range(nb):
            var want_left = 0
            if not skipped[b]:
                for t in range(TPB):
                    var ri = i * TPB + t
                    if ri < count:
                        if flat[col * N_ROWS + Int(row_ids[begin + ri])] <= q:
                            want_left += 1
            scanned_blocks += 1
            if Int(base.blk_left[base_i + i]) != want_left:
                blk_bad += 1
            if not skipped[b]:
                if Int(base.blk_off[base_i + i]) != running:
                    blk_bad += 1
                if running > 0:
                    nonzero_off += 1
            running += want_left
    if blk_bad == 0:
        print(
            "  ARM E OK:",
            scanned_blocks,
            "per-block left counts and cross-block offsets match a host"
            " tally;",
            nonzero_off,
            "of them are NONZERO",
        )
    else:
        failures += 1
        print(
            "  ARM E FAILED:",
            blk_bad,
            "of",
            scanned_blocks,
            "blocks disagree with the host tally of blk_left / blk_off",
        )
    if nonzero_off == 0:
        failures += 1
        print(
            "  FIXTURE DEFECT: every cross-block offset is zero, so"
            " PART_MB_SAB_NO_SCAN cannot move a single row and that arm"
            " proves nothing"
        )

    # ---------------------------------------------------------------------
    # SABOTAGE -- one per mechanism, prediction stated BEFORE the count.
    # ---------------------------------------------------------------------
    print("")
    print(
        "[arm F] sabotage, one per MECHANISM, same binary, selected by a"
        " kernel argument"
    )
    print(
        "        (a sabotaged arm races -- several blocks write one slot with"
        " no ordering between them -- so the MOVED COUNT below wobbles by a"
        " few between runs while the redness and the shape do not)"
    )
    var owner = List[Int](length=N_ROWS, fill=-1)
    for b in range(n_items):
        var begin = Int(items[b].instances.begin)
        for s in range(begin, begin + Int(items[b].instances.count)):
            owner[s] = b

    var sabs = [
        PART_MB_SAB_NO_VALID_GUARD,
        PART_MB_SAB_SINGLE_BLOCK,
        PART_MB_SAB_NO_SCAN,
    ]
    var names = [
        String(
            "the VALIDITY GUARD (`split_not_valid` ignored in all four"
            " passes)"
        ),
        String(
            "the MULTI-BLOCK tiling (every block claims offset_blockid == 0"
            " in the count and the scatter)"
        ),
        String(
            "the SCAN (scatter with a zero cross-block prefix, as if every"
            " block were first in its node)"
        ),
    ]
    var predictions = [
        String(
            "arm C goes RED and the moved slots lie ONLY inside the"
            " refused nodes -- the guard's other side, invisible to a check"
            " that only looked at the partitioned nodes"
        ),
        String(
            "arms A and D go RED and the moved slots lie ONLY inside the"
            " MULTI-block nodes: their blocks all reread chunk 0, so rows are"
            " duplicated and slots are left holding the sentinel. A"
            " single-block node cannot move, because its offset_blockid is 0"
            " already"
        ),
        String(
            "arms A and D go RED and the moved slots lie ONLY inside the"
            " MULTI-block nodes: every block packs its left rows at `begin`"
            " and its right rows too far along. A one-block node cannot move,"
            " which is why arm E had to prove some offset is nonzero"
        ),
    ]
    var mechanisms = 0
    for a in range(len(sabs)):
        var run = _launch_multiblock(
            ctx,
            flat,
            row_ids,
            items,
            splits,
            blk_base,
            n_blocks,
            plan.info,
            sabs[a],
        )
        var sv = _verify(
            flat, row_ids, run.row_ids, oracle, items, splits, skipped
        )
        var moved = 0
        var moved_unpredicted = 0
        for s in range(N_ROWS):
            if run.row_ids[s] == base.row_ids[s]:
                continue
            moved += 1
            var o = owner[s]
            var allowed: Bool
            if o < 0:
                allowed = False
            elif a == 0:
                allowed = skipped[o]
            else:
                allowed = multi[o] and not skipped[o]
            if not allowed:
                moved_unpredicted += 1
        print("  -", names[a])
        print("      predicted:", predictions[a])
        if sv.total() == 0:
            failures += 1
            print(
                "      *** DEFECT IN THE CHECK ***: the answer is still"
                " RIGHT. This fixture cannot see the mechanism; the FIXTURE"
                " is what needs fixing, not the kernel."
            )
        else:
            mechanisms += 1
            print(
                "      RED as required: A",
                sv.bad_a,
                "B",
                sv.bad_b,
                "C",
                sv.bad_c,
                "D",
                sv.bad_d,
                "--",
                moved,
                "slots moved",
            )
        if moved == 0:
            failures += 1
            print("      and it moved NOTHING, which is the same defect")
        if moved_unpredicted != 0:
            failures += 1
            print(
                "      SHAPE WRONG:",
                moved_unpredicted,
                "of the",
                moved,
                "moved slots are outside the predicted set, so this arm is"
                " red for a reason the prediction did not name",
            )
        else:
            print(
                "      shape as predicted: all",
                moved,
                "moved slots lie inside the predicted nodes, and no others"
                " moved",
            )

    print("")
    if failures == 0:
        print(
            "partition_multiblock_check: PASS --",
            n_items,
            "nodes (",
            n_single,
            "single-block,",
            n_multi,
            "multi-block,",
            n_skipped,
            "refused ) over",
            n_blocks,
            "blocks;",
            partitioned_slots,
            "partitioned slots and",
            N_ROWS,
            "row_ids slots compared per arm against node_split_kernel;",
            v.cells,
            "assertion cells;",
            mechanisms,
            "mechanisms sabotaged and each turned the answer red inside the"
            " nodes it predicted",
        )
    else:
        print("partition_multiblock_check: FAIL --", failures, "arm(s) red")
        raise Error("partition_multiblock_check failed")
