# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for `gbdt/methods/oblivious_tree_fold_tasks.mojo`.

The fold layout ordered boosting grows trees on: N tasks laid out as 2N
partitions over one concatenated document array, learn at even folds and test
at odd ones.

**WHY THE PAIRING IS THE THING TO GATE.** The dynamic scorer -- ported and
gated before this file existed -- reads folds `(f, f + 1)` as
`(estimate, test)` and steps by TWO. If this layout ever put the test half
first, or interleaved tasks differently, the scorer would evaluate every
split against the wrong half of every fold. It would still produce finite
scores and a well-formed tree.

GATES:

  G1  N tasks give 2N partitions, learn at index 2k and test at 2k+1, with
      the sizes the tasks named.
  G2  the partitions TILE the document array: contiguous, no gap, no
      overlap, and covering exactly `total_indices`. Checked by walking a
      per-document ownership array, not by comparing a sum -- a layout that
      double-covers one range and skips another has the right total.
  G3  `fold_bits` is CEIL of log2, not floor. `PORTING.md` 107 records what
      floor cost when `IntLog2` was read that way, and this is the same
      function in a second place.
  G4  the SINGLE-TASK arm is one partition, fold_count 1, fold_bits 0 --
      the arm every fit in this repository takes today.
  G5  the device bins really carry 0..2N-1 in the right ranges, checked per
      document.
  G7  `create_subsets` ACCEPTS the layout: `fold_count` and `fold_bits`
      reach the subsets, `max_part_count` becomes `1 << (fold_bits +
      max_depth)` rather than `1 << max_depth`, the root level already has
      `1 << fold_bits` partitions rather than one, and the initial bins
      carry the task sizes. Gated because every one of those is a place a
      single-task assumption survives unnoticed -- a `max_part_count` that
      forgot `fold_bits` allocates a quarter of what a 2-fold tree needs and
      corrupts silently at depth.
  G6  the fold STRIPE is live at a ragged fold count. With 3 tasks the fold
      count is 6 and `PointwisePartOffsetsHelper`'s data-partition offset
      rounds the fold axis up to 8 while its histogram offset packs tight.
      At a power-of-two fold count the two coincide and nothing downstream
      could tell them apart, so this gate uses a count that is NOT a power
      of two on purpose.
"""

from max.gpu.host import DeviceContext

from gbdt.methods.kernel.split_properties_helpers import (
    PointwisePartOffsetsHelper,
)
from gbdt.methods.pointwise_optimization_subsets import (
    TL2Target,
    create_subsets,
)
from gbdt.methods.oblivious_tree_fold_tasks import (
    FoldTask,
    int_log2_ceil,
    plan_fold_layout,
    plan_single_task_layout,
    write_fold_based_initial_bins,
)


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # deliberately UNEQUAL and not powers of two
    var tasks: List[FoldTask] = [
        FoldTask(100, 50), FoldTask(213, 97), FoldTask(400, 190)
    ]
    var lay = plan_fold_layout(tasks)

    # ---------------------------------------------------------------- G1
    var bad1 = 0
    if len(lay.parts) != 2 * len(tasks):
        print(
            "FAIL G1:", len(tasks), "tasks gave", len(lay.parts),
            "partitions, expected", 2 * len(tasks),
        )
        bad1 += 1
    else:
        for k in range(len(tasks)):
            if Int(lay.parts[2 * k].size) != tasks[k].learn_size:
                print(
                    "FAIL G1: task", k, "LEARN half is",
                    lay.parts[2 * k].size, "expected", tasks[k].learn_size,
                    "-- the dynamic scorer reads the EVEN fold as the"
                    " estimate half",
                )
                bad1 += 1
            if Int(lay.parts[2 * k + 1].size) != tasks[k].test_size:
                print(
                    "FAIL G1: task", k, "TEST half is",
                    lay.parts[2 * k + 1].size, "expected",
                    tasks[k].test_size,
                )
                bad1 += 1
    failures += bad1
    if bad1 == 0:
        print(
            "  ok   G1 --", len(tasks), "tasks ->", len(lay.parts),
            "partitions, learn even and test odd",
        )

    # ---------------------------------------------------------------- G2
    # per-document ownership, NOT a sum: a layout that covers one range
    # twice and skips another has exactly the right total
    var owner = List[Int]()
    for _ in range(lay.total_indices):
        owner.append(-1)
    var overlaps = 0
    var out_of_range = 0
    for p in range(len(lay.parts)):
        ref part = lay.parts[p]
        for d in range(Int(part.offset), Int(part.offset) + Int(part.size)):
            if d < 0 or d >= lay.total_indices:
                out_of_range += 1
            elif owner[d] != -1:
                overlaps += 1
            else:
                owner[d] = p
    var uncovered = 0
    for d in range(lay.total_indices):
        if owner[d] == -1:
            uncovered += 1
    if overlaps != 0 or uncovered != 0 or out_of_range != 0:
        print(
            "FAIL G2: the partitions do not tile the document array --",
            overlaps, "documents owned twice,", uncovered, "owned by none,",
            out_of_range, "outside it",
        )
        failures += 1
    else:
        print(
            "  ok   G2 -- the", len(lay.parts), "partitions tile all",
            lay.total_indices, "documents exactly once",
        )

    # ---------------------------------------------------------------- G3
    var bad3 = 0
    var want_bits: List[Int] = [0, 0, 1, 2, 2, 3, 3, 3, 3, 4]
    for v in range(1, 10):
        if int_log2_ceil(v) != want_bits[v]:
            print(
                "FAIL G3: int_log2_ceil(", v, ") =", int_log2_ceil(v),
                "expected", want_bits[v], "-- IntLog2 is CEIL",
            )
            bad3 += 1
    if lay.fold_count != 6 or lay.fold_bits != 3:
        print(
            "FAIL G3: 3 tasks gave fold_count", lay.fold_count, "fold_bits",
            lay.fold_bits, "expected 6 and 3",
        )
        bad3 += 1
    failures += bad3
    if bad3 == 0:
        print("  ok   G3 -- fold_bits is ceil(log2(fold_count)): 6 -> 3")

    # ---------------------------------------------------------------- G4
    var single = plan_single_task_layout(1000)
    if (
        len(single.parts) != 1
        or single.fold_count != 1
        or single.fold_bits != 0
        or Int(single.parts[0].size) != 1000
    ):
        print(
            "FAIL G4: the single-task arm gave", len(single.parts),
            "partitions, fold_count", single.fold_count, "fold_bits",
            single.fold_bits,
        )
        failures += 1
    else:
        print(
            "  ok   G4 -- the single-task arm is one partition,"
            " fold_count 1, fold_bits 0 (the shipped path)",
        )

    # ---------------------------------------------------------------- G5
    var d_bins = ctx.enqueue_create_buffer[DType.uint32](lay.total_indices)
    ctx.enqueue_memset(d_bins, UInt32(0xFFFFFFFF))
    write_fold_based_initial_bins(ctx, lay, d_bins)
    var h_bins = ctx.enqueue_create_host_buffer[DType.uint32](
        lay.total_indices
    )
    ctx.enqueue_copy(dst_buf=h_bins, src_buf=d_bins)
    ctx.synchronize()
    var bad5 = 0
    for d in range(lay.total_indices):
        if Int(h_bins[d]) != owner[d]:
            if bad5 < 3:
                print(
                    "     G5 document", d, "has bin", h_bins[d],
                    "but belongs to partition", owner[d],
                )
            bad5 += 1
    if bad5 != 0:
        print(
            "FAIL G5:", bad5, "of", lay.total_indices,
            "documents carry the wrong initial bin",
        )
        failures += 1
    else:
        print(
            "  ok   G5 -- all", lay.total_indices,
            "documents carry their partition's bin, 0 to",
            len(lay.parts) - 1,
        )

    # ---------------------------------------------------------------- G6
    var helper = PointwisePartOffsetsHelper(UInt32(lay.fold_count))
    var stripe = Int(helper.data_partition_offset(1, 0))
    var tight = Int(helper.histogram_offset(1, 0))
    if stripe != 8:
        print(
            "FAIL G6: at fold_count 6 the fold STRIPE is", stripe,
            "expected 8 (1 << ceil(log2(6)))",
        )
        failures += 1
    elif stripe == tight:
        print(
            "FAIL G6: the striped and tight offsets COINCIDE at fold_count",
            lay.fold_count,
            "-- this fixture cannot tell them apart and must use a fold"
            " count that is not a power of two",
        )
        failures += 1
    else:
        print(
            "  ok   G6 -- at fold_count 6 the data-partition offset strides"
            " by 8 and the histogram offset by 6; they differ, so the"
            " stripe is live",
        )

    # ---------------------------------------------------------------- G7
    var g7_tasks: List[FoldTask] = [FoldTask(400, 200), FoldTask(300, 150)]
    var g7 = plan_fold_layout(g7_tasks)
    var n7 = g7.total_indices
    var w7 = ctx.enqueue_create_buffer[DType.float32](n7)
    var t7 = ctx.enqueue_create_buffer[DType.float32](n7)
    var h7 = ctx.enqueue_create_host_buffer[DType.float32](n7)
    for r in range(n7):
        h7.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=w7, src_ptr=h7.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=t7, src_ptr=h7.unsafe_ptr())
    ctx.synchronize()
    var src7 = TL2Target(w7^, t7^, n7)
    comptime G7_DEPTH = 4
    var subs = create_subsets(
        ctx, G7_DEPTH, src7, g7.fold_count, g7.fold_bits
    )
    var bad7 = 0
    if Int(subs.fold_count) != g7.fold_count:
        print("FAIL G7: fold_count reached the subsets as", subs.fold_count)
        bad7 += 1
    if Int(subs.fold_bits) != g7.fold_bits:
        print("FAIL G7: fold_bits reached the subsets as", subs.fold_bits)
        bad7 += 1
    var want_mpc = 1 << (g7.fold_bits + G7_DEPTH)
    if subs.max_part_count != want_mpc:
        print(
            "FAIL G7: max_part_count is", subs.max_part_count, "expected",
            want_mpc,
            "-- `1 << (FoldBits + maxDepth)`. Forgetting fold_bits"
            " allocates a fraction of what a folded tree needs and"
            " corrupts at depth rather than at allocation.",
        )
        bad7 += 1
    if subs.current_part_count() != (1 << g7.fold_bits):
        print(
            "FAIL G7: the ROOT level has", subs.current_part_count(),
            "partitions, expected", 1 << g7.fold_bits,
            "-- with folds the tree starts with one partition per fold,"
            " not one",
        )
        bad7 += 1
    write_fold_based_initial_bins(ctx, g7, subs.bins)
    var hb7 = ctx.enqueue_create_host_buffer[DType.uint32](n7)
    ctx.enqueue_copy(dst_buf=hb7, src_buf=subs.bins)
    ctx.synchronize()
    var tally = List[Int]()
    for _ in range(len(g7.parts)):
        tally.append(0)
    for r in range(n7):
        if Int(hb7[r]) < len(g7.parts):
            tally[Int(hb7[r])] += 1
    for p in range(len(g7.parts)):
        if tally[p] != Int(g7.parts[p].size):
            print(
                "FAIL G7: bin", p, "holds", tally[p], "documents, expected",
                g7.parts[p].size,
            )
            bad7 += 1
    failures += bad7
    if bad7 == 0:
        print(
            "  ok   G7 -- create_subsets takes the layout: fold_count",
            subs.fold_count, "fold_bits", subs.fold_bits,
            "max_part_count", subs.max_part_count, ", root level",
            subs.current_part_count(), "partitions, bins carry the task"
            " sizes",
        )

    _ = d_bins^
    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("fold task layout: G1-G7 pass")
