# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for ordered boosting's FOLD AXIS, from `CreateFolds` to the histogram.

    pixi run check-ordered-boosting

`check-dynamic-boosting-folds` gates the fold BOUNDARIES against a closed
form and `check-fold-tasks` gates the fold LAYOUT (2N alternating partitions,
`FoldBits` ceil, `create_subsets` accepting both). Both were green while
`fold_count` was 1 at every caller in the tree, so neither of them has ever
run a kernel with a fold axis. THIS FILE IS THE WIRING GATE: it drives the
same objects the searcher drives, at `FoldCount = 12`, and asks where every
document ENDED UP.

THE FIXTURE IS RAGGED ON PURPOSE. `create_folds(n = 600, g = 2.0,
min_fold_size = 100)` gives SIX folds, so `FoldCount` is 12 and `FoldBits` is
4: the data-partition stripe is 16 while the histogram stride is 12. At a
power-of-two fold count those two coincide and a kernel that used the wrong
one would be exact everywhere -- `PORTING.md` 91 B's fold stripe is only
observable at a ragged count, and `NEXT_TWO.md` records a whole round lost to
a fixture where "every power-of-two fold count made two offsets coincide".

EVERY PLANTED VALUE IS DISTINCT PER DOCUMENT. `weight[i]` and `target[i]` are
two different affine hashes of the position, so a partition that holds the
right NUMBER of documents but the wrong ONES fails on its sum. And O3/O4 do
not stop at sums: they compare the index array partition by partition,
element by element, in order, because a stable reorder is an ORDER claim and
a multiset comparison cannot see it ([[uniform-test-data-hides-permutation]]).

GATES

  O1  `create_folds` -> `fold_tasks_from_folds` -> `plan_fold_layout`, against
      the g = 2 closed form written here from the recurrence. Includes the
      fact that surprises everyone: the concatenated document array is LONGER
      THAN THE DATASET (1,463 positions for 600 rows), because the estimate
      slices are nested prefixes and every document appears in several folds.
  O2  `make_fold_doc_indices`, per position, identity and a real permutation.
      This is the array the compressed index is read through, and at N tasks
      a position and a document id are different numbers.
  O3  `create_fold_based_subsets` on the device: the bin of every position,
      the offset and size of every partition, the gathered stat columns per
      position, and the partition stats per partition.
  O4  `split_subsets` at `FoldBits = 4`: the fold id must survive in the LOW
      bits while the split bit lands at `CurrentDepth + FoldBits`, over two
      levels. Checked as the ORDERED index sequence of every partition
      against a host stable partition, so a reorder that keeps the right set
      in the wrong order fails.
  O5  `compute_hist2` AT FOLD COUNT 12, per histogram cell, against a host
      tally that indexes with `GetHistogramOffset(part, fold) = part *
      FoldCount + fold` while the kernel reads its partition through
      `GetDataPartitionOffset = part * 16 + fold`. THIS KERNEL HAS NEVER RUN
      WITH A FOLD AXIS IN THIS REPOSITORY.
  O6  `find_optimal_split` at fold count 12 dispatches to
      `FindOptimalSplitDynamic`, which supports THREE of the seven score
      functions. SolarL2, Cosine and NewtonCosine run; L2, NewtonL2, SatL2
      and LOOL2 must RAISE, because `pointwise_scores.cu:469` throws.
  O7  the searcher's two refusals: an unsupported score function with folds,
      and DEVIATION 126 (the calcer carrying a different fold count from the
      layout).

WHAT THIS FILE DOES NOT GATE, said plainly rather than implied by a green
tick. O5 builds `compute_hist2`'s arguments itself, so it checks the KERNEL
at a fold axis and not `PolicyScoreHelper`, which is the caller that normally
builds them and which hard-codes `1` (`PORTING.md` 115 is the same shape of
hole, found the same way). Until DEVIATION 126 is lifted no tree grows at
`fold_count > 1`, so this file gates the parts and O7 gates the refusal.
"""

from max.gpu.host import DeviceContext, HostBuffer

from gbdt.methods.dynamic_boosting_folds import (
    EBoostingType,
    IQueriesGrouping,
    TFold,
    create_folds,
    min_estimation_size,
)
from gbdt.methods.kernel.pointwise_scores import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_LOO_L2,
    SCORE_FUNCTION_NEWTON_COSINE,
    SCORE_FUNCTION_NEWTON_L2,
    SCORE_FUNCTION_SAT_L2,
    SCORE_FUNCTION_SOLAR_L2,
    find_optimal_split,
)
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.methods.oblivious_tree_doc_parallel_structure_searcher import (
    PointwiseTreeWorkspace,
    fit_oblivious_tree_structure,
)
from gbdt.methods.pointwise_kernels import compute_hist2
from gbdt.methods.histograms_helper import POLICY_HALF_BYTE
from gbdt.methods.pointwise_kernels import FoldsHistogram
from gbdt.gpu_util.kernel.transform import launch_gather_with_mask_u32
from gbdt.methods.pointwise_optimization_subsets import (
    GATHER_NO_MASK,
    PART_OFFSET,
    TOptimizationSubsets,
    PART_SIZE,
    PARTITION_RECORD,
    PARTITION_STAT_STRIDE,
    PART_STAT_SUM,
    PART_STAT_WEIGHT,
    TL2Target,
    split_subsets,
)
from gbdt.methods.oblivious_tree_fold_tasks import (
    create_fold_based_subsets,
    fold_tasks_from_folds,
    int_log2_ceil,
    make_fold_doc_indices,
    plan_fold_layout,
)

comptime N_ROWS = 600
comptime GROWTH = 2.0
comptime MIN_FOLD_SIZE = 100
comptime MAX_DEPTH = 2

comptime N_HB = 8
"""Eight half-byte features in ONE 32-bit column: the whole policy is one
nibble word, so the fixture needs no group arithmetic and the gate is about
the fold axis rather than about packing."""

comptime SM_COUNT = 8


def plant_weight(i: Int) -> Float32:
    """Distinct per position, and exactly representable: integers under
    2^24 add without rounding, so O3/O4 compare with NO tolerance."""
    return Float32(1 + (i * 37) % 251)


def plant_target(i: Int) -> Float32:
    return Float32(1 + (i * 53) % 257) - Float32(128)


def closed_form_fold_rights(n: Int, m0: Int) -> List[Int]:
    """The g = 2 recurrence solved, NOT a replay of `CreateFolds`' loop.

    With no groups `NextQueryOffsetForLine(line)` is `min(line + 1, n)`, so
    the eval right edges are `R_k = min(2^(k+1) * (m0 + 1) - 1, n)` and the
    series stops at the first `R_k == n`. `check-dynamic-boosting-folds` F3
    derives the same identity; it is repeated here because THIS file's fold
    count, fold bits, partition sizes and total index size are all functions
    of it, and a gate that took them from the port would be checking the
    port against itself.
    """
    var out = List[Int]()
    var k = 0
    while True:
        var r = (1 << (k + 1)) * (m0 + 1) - 1
        if r > n:
            r = n
        out.append(r)
        if r >= n:
            break
        k += 1
    return out^


def drain(
    ctx: DeviceContext,
    mut subsets: TOptimizationSubsets,
    mut hb_bins: HostBuffer[DType.uint32],
    mut hb_idx: HostBuffer[DType.uint32],
    mut hb_parts: HostBuffer[DType.uint32],
    mut hb_stats: HostBuffer[DType.float32],
    mut hb_gw: HostBuffer[DType.float32],
    mut hb_gt: HostBuffer[DType.float32],
) raises:
    """Module level because a nested closure cannot capture a
    `DeviceContext` (`PORTING_RULES` rule 4)."""
    ctx.enqueue_copy(dst_buf=hb_bins, src_buf=subsets.bins)
    ctx.enqueue_copy(dst_buf=hb_idx, src_buf=subsets.indices)
    ctx.enqueue_copy(dst_buf=hb_parts, src_buf=subsets.partitions)
    ctx.enqueue_copy(dst_buf=hb_stats, src_buf=subsets.partition_stats)
    ctx.enqueue_copy(dst_buf=hb_gw, src_buf=subsets.gathered_weight)
    ctx.enqueue_copy(dst_buf=hb_gt, src_buf=subsets.gathered_target)
    ctx.synchronize()


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # =============================================================== O1
    var grouping = IQueriesGrouping.without_queries(N_ROWS)
    var folds = create_folds(
        N_ROWS, GROWTH, grouping, EBoostingType.Ordered, MIN_FOLD_SIZE, 1
    )
    var m0 = min_estimation_size(N_ROWS, MIN_FOLD_SIZE) + 1
    var rights = closed_form_fold_rights(N_ROWS, m0)

    var bad = 0
    if len(folds) != len(rights):
        print(
            "FAIL O1: CreateFolds gave", len(folds),
            "folds, the closed form gives", len(rights),
        )
        bad += 1
    var want_total = 0
    for k in range(len(rights)):
        # fold k: estimate [0, L_k), eval [L_k, R_k) with L_0 = m0 and
        # L_k = R_{k-1}
        var left = m0 if k == 0 else rights[k - 1]
        want_total += left + (rights[k] - left)
        if k < len(folds):
            if (
                folds[k].estimate_samples.left != 0
                or folds[k].estimate_samples.right != left
                or folds[k].quality_evaluate_samples.left != left
                or folds[k].quality_evaluate_samples.right != rights[k]
            ):
                print(
                    "FAIL O1: fold", k, "is [0,",
                    folds[k].estimate_samples.right, ") + [",
                    folds[k].quality_evaluate_samples.left, ",",
                    folds[k].quality_evaluate_samples.right,
                    "), closed form says [0,", left, ") + [", left, ",",
                    rights[k], ")",
                )
                bad += 1

    var tasks = fold_tasks_from_folds(folds)
    var lay = plan_fold_layout(tasks)

    # THE INDEPENDENT PARTITION TABLE, and O1 is where it earns its keep.
    # Every gate below sizes its spans from THIS, not from `lay.parts`.
    # Taking them from `lay.parts` makes the whole file self-referential:
    # the first sabotage run against it -- `plan_fold_layout` writing the
    # TEST half before the LEARN half, which is the exact failure
    # `PORTING.md` 119 warns about -- moved NOTHING, because both sides of
    # every comparison came from the swapped layout.
    var want_off = List[Int]()
    var want_size = List[Int]()
    var cur0 = 0
    for k in range(len(folds)):
        var lz = (
            folds[k].estimate_samples.right - folds[k].estimate_samples.left
        )
        var tz = (
            folds[k].quality_evaluate_samples.right
            - folds[k].quality_evaluate_samples.left
        )
        # LEARN at 2k, TEST at 2k+1 -- the pairing the dynamic scorer reads
        # as `(estimate, test)` and steps by two
        want_off.append(cur0)
        want_size.append(lz)
        cur0 += lz
        want_off.append(cur0)
        want_size.append(tz)
        cur0 += tz
    if len(lay.parts) == len(want_off):
        for p in range(len(want_off)):
            if (
                Int(lay.parts[p].offset) != want_off[p]
                or Int(lay.parts[p].size) != want_size[p]
            ):
                print(
                    "FAIL O1: partition", p, "is (", lay.parts[p].offset,
                    ",", lay.parts[p].size, ") but fold", p // 2,
                    "'s", "LEARN" if p % 2 == 0 else "TEST",
                    "half is (", want_off[p], ",", want_size[p], ")",
                )
                bad += 1
    if lay.fold_count != 2 * len(folds):
        print("FAIL O1: FoldCount", lay.fold_count, "for", len(folds), "folds")
        bad += 1
    if lay.fold_bits != int_log2_ceil(lay.fold_count):
        print("FAIL O1: FoldBits", lay.fold_bits, "is not IntLog2 ceil")
        bad += 1
    # THE SAME CLAIM WITHOUT `int_log2_ceil`, in shifts. The line above
    # compares the layout against the function that produced it, so a FLOOR
    # `IntLog2` agrees with itself: sabotaging `int_log2_ceil` to floor left
    # this gate green and the check went on to die three sections later on
    # an out-of-range partition. Ceil means `1 << FoldBits` covers the count
    # and `1 << (FoldBits - 1)` does not.
    if lay.fold_count > (1 << lay.fold_bits) or (
        lay.fold_bits > 0 and lay.fold_count <= (1 << (lay.fold_bits - 1))
    ):
        print(
            "FAIL O1: FoldBits", lay.fold_bits, "is not the CEIL of log2(",
            lay.fold_count, "): the stripe is", 1 << lay.fold_bits,
        )
        bad += 1
    if lay.total_indices != want_total:
        print(
            "FAIL O1: total_indices", lay.total_indices, "want", want_total
        )
        bad += 1
    if lay.total_indices <= N_ROWS:
        print(
            "FAIL O1: the concatenated array is", lay.total_indices,
            "for", N_ROWS, "rows -- the estimate slices are nested"
            " prefixes and it MUST be longer",
        )
        bad += 1
    # the stripe must be WIDER than the count, or O5 proves nothing
    var stripe = 1 << lay.fold_bits
    if stripe == lay.fold_count:
        print(
            "FAIL O1: fold count", lay.fold_count, "is a power of two, so",
            "the data-partition stripe and the histogram stride coincide",
            "and this fixture cannot tell them apart",
        )
        bad += 1
    if bad != 0:
        failures += 1
    else:
        print(
            "  ok   O1 --", len(folds), "folds, FoldCount", lay.fold_count,
            "FoldBits", lay.fold_bits, "stripe", stripe, ", ",
            lay.total_indices, "positions over", N_ROWS, "rows",
        )

    var n_docs = lay.total_indices
    var fold_count = lay.fold_count
    var fold_bits = lay.fold_bits

    # =============================================================== O2
    var ids = make_fold_doc_indices(folds)
    bad = 0
    if len(ids) != n_docs:
        print("FAIL O2: MakeDocIndices gave", len(ids), "want", n_docs)
        bad += 1
    else:
        var at = 0
        for k in range(len(folds)):
            for p in range(
                folds[k].estimate_samples.left,
                folds[k].estimate_samples.right,
            ):
                if ids[at] != UInt32(p):
                    bad += 1
                at += 1
            for p in range(
                folds[k].quality_evaluate_samples.left,
                folds[k].quality_evaluate_samples.right,
            ):
                if ids[at] != UInt32(p):
                    bad += 1
                at += 1
    # a REAL permutation, so "identity" is not what is being verified
    var perm = List[UInt32]()
    for r in range(N_ROWS):
        perm.append(UInt32((r * 421 + 17) % N_ROWS))
    var ids_p = make_fold_doc_indices(folds, perm)
    var moved = 0
    for i in range(len(ids_p)):
        if ids_p[i] != ids[i]:
            moved += 1
    var at2 = 0
    for k in range(len(folds)):
        for p in range(
            folds[k].estimate_samples.left, folds[k].estimate_samples.right
        ):
            if ids_p[at2] != perm[p]:
                bad += 1
            at2 += 1
        for p in range(
            folds[k].quality_evaluate_samples.left,
            folds[k].quality_evaluate_samples.right,
        ):
            if ids_p[at2] != perm[p]:
                bad += 1
            at2 += 1
    if moved < n_docs // 2:
        print(
            "FAIL O2: the permutation moved only", moved, "of", n_docs,
            "ids -- the non-identity arm is inert",
        )
        bad += 1
    if bad != 0:
        print("FAIL O2:", bad, "positions wrong")
        failures += 1
    else:
        print(
            "  ok   O2 --", n_docs, "doc ids, identity and permuted"
            " (", moved, "moved )",
        )

    # =============================================================== O3
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n_docs)
    var h_t = ctx.enqueue_create_host_buffer[DType.float32](n_docs)
    for i in range(n_docs):
        h_w[i] = plant_weight(i)
        h_t[i] = plant_target(i)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n_docs)
    var d_t = ctx.enqueue_create_buffer[DType.float32](n_docs)
    ctx.enqueue_copy(dst_buf=d_w, src_buf=h_w)
    ctx.enqueue_copy(dst_buf=d_t, src_buf=h_t)
    ctx.synchronize()
    # A HOST buffer handed to `enqueue_copy` is LAST USED at the enqueue, so
    # Mojo may free it before the copy runs
    # ([[mojo-buffer-freed-at-last-use]]); a read after `synchronize` keeps
    # it alive across the queue. Without these two lines this gate reported
    # every gathered stat column and every histogram cell as 0.0, on some
    # runs and not others.
    _ = h_w[0]
    _ = h_t[0]

    var target = TL2Target(d_w^, d_t^, n_docs)
    var subsets = create_fold_based_subsets(ctx, MAX_DEPTH, target, lay)

    var max_parts = 1 << (fold_bits + MAX_DEPTH)
    var hb_bins = ctx.enqueue_create_host_buffer[DType.uint32](n_docs)
    var hb_idx = ctx.enqueue_create_host_buffer[DType.uint32](n_docs)
    var hb_parts = ctx.enqueue_create_host_buffer[DType.uint32](
        max_parts * PARTITION_RECORD
    )
    var hb_stats = ctx.enqueue_create_host_buffer[DType.float32](
        max_parts * PARTITION_STAT_STRIDE
    )
    var hb_gw = ctx.enqueue_create_host_buffer[DType.float32](n_docs)
    var hb_gt = ctx.enqueue_create_host_buffer[DType.float32](n_docs)

    drain(
        ctx, subsets, hb_bins, hb_idx, hb_parts, hb_stats, hb_gw, hb_gt
    )

    bad = 0
    var cursor = 0
    for p in range(fold_count):
        var size = want_size[p]
        var off = Int(hb_parts[p * PARTITION_RECORD + PART_OFFSET])
        var got_size = Int(hb_parts[p * PARTITION_RECORD + PART_SIZE])
        if off != cursor or got_size != size:
            print(
                "FAIL O3: partition", p, "is (", off, ",", got_size,
                ") want (", cursor, ",", size, ")",
            )
            bad += 1
        var sw = Float32(0.0)
        var st = Float32(0.0)
        for i in range(cursor, cursor + size):
            if hb_bins[i] != UInt32(p):
                bad += 1
            if hb_idx[i] != UInt32(i):
                bad += 1
            if hb_gw[i] != plant_weight(i) or hb_gt[i] != plant_target(i):
                bad += 1
            sw += plant_weight(i)
            st += plant_target(i)
        var gw = hb_stats[p * PARTITION_STAT_STRIDE + PART_STAT_WEIGHT]
        var gt = hb_stats[p * PARTITION_STAT_STRIDE + PART_STAT_SUM]
        if gw != sw or gt != st:
            print(
                "FAIL O3: partition", p, "stats (", gw, ",", gt,
                ") want (", sw, ",", st, ")",
            )
            bad += 1
        cursor += size
    if cursor != n_docs:
        print("FAIL O3: partitions cover", cursor, "of", n_docs)
        bad += 1
    if bad != 0:
        print("FAIL O3:", bad, "cells wrong")
        failures += 1
    else:
        print(
            "  ok   O3 --", fold_count, "partitions,", n_docs,
            "positions: bin, index, both gathered columns, offset, size,"
            " weight and sum, all per cell",
        )

    # =============================================================== O5
    # (before O4, because O4 splits the subsets and O5 wants depth 0)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var h_ci = ctx.enqueue_create_host_buffer[DType.uint32](N_ROWS)
    var hb_folds_host = List[UInt32]()
    for j in range(N_HB):
        hb_folds_host.append(UInt32(16))
    var hb_first = List[UInt32]()
    var c = UInt32(0)
    for j in range(N_HB):
        hb_first.append(c)
        c += hb_folds_host[j]
    var hist_line = Int(c)
    for r in range(N_ROWS):
        var word = UInt32(0)
        for k in range(N_HB):
            var b = (r * (5 + 2 * k) + 3 * k) % 16
            # `Shift(j) = 28 - 4 * localId`, high nibble first
            word |= UInt32(b) << UInt32(28 - 4 * k)
        h_ci[r] = word
    ctx.enqueue_copy(dst_buf=d_ci, src_buf=h_ci)
    ctx.synchronize()
    _ = h_ci[0]

    var d_off = ctx.enqueue_create_buffer[DType.uint32](N_HB)
    var d_first = ctx.enqueue_create_buffer[DType.uint32](N_HB)
    var d_folds = ctx.enqueue_create_buffer[DType.uint32](N_HB)
    var d_oh = ctx.enqueue_create_buffer[DType.uint8](N_HB)
    var off_host = List[UInt32]()
    var oh_host = List[UInt8]()
    for _ in range(N_HB):
        off_host.append(UInt32(0))
        oh_host.append(UInt8(0))
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=off_host.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_first, src_ptr=hb_first.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_folds, src_ptr=hb_folds_host.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_oh, src_ptr=oh_host.unsafe_ptr())

    var d_docs = ctx.enqueue_create_buffer[DType.uint32](n_docs)
    ctx.enqueue_copy(dst_buf=d_docs, src_ptr=ids.unsafe_ptr())
    ctx.synchronize()

    var hist_cells = (1 << MAX_DEPTH) * fold_count * hist_line * 2
    var d_hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
    var h_hist = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)
    ctx.enqueue_memset(d_hist, Float32(0.0))

    compute_hist2(
        ctx, POLICY_HALF_BYTE,
        d_off.unsafe_ptr(), d_first.unsafe_ptr(), d_folds.unsafe_ptr(),
        d_oh.unsafe_ptr(), N_HB, 0, hist_line,
        d_ci.unsafe_ptr(),
        subsets.gathered_target.unsafe_ptr(),
        subsets.gathered_weight.unsafe_ptr(),
        d_docs.unsafe_ptr(),
        n_docs,
        subsets.partitions.unsafe_ptr(),
        1,
        fold_count,
        d_hist.unsafe_ptr(),
        hist_line,
        True,
        FoldsHistogram(),
        SM_COUNT,
        Float32(1.0),
    )
    ctx.enqueue_copy(dst_buf=h_hist, src_buf=d_hist)
    ctx.synchronize()

    # host tally: part 0, folds 0..FoldCount-1, indexed with the TIGHT
    # histogram stride while the kernel found its span through the ROUNDED
    # data-partition stripe
    var want = List[Float32]()
    for _ in range(fold_count * hist_line * 2):
        want.append(Float32(0.0))
    var cur = 0
    for f in range(fold_count):
        var size = want_size[f]
        for i in range(cur, cur + size):
            var row = Int(ids[i])
            var word = h_ci[row]
            for k in range(N_HB):
                var b = Int((word >> UInt32(28 - 4 * k)) & 15)
                var at = (f * hist_line + Int(hb_first[k]) + b) * 2
                want[at + 0] += plant_weight(i)
                want[at + 1] += plant_target(i)
        cur += size
    # `ScanPointwiseHistograms`: a running prefix per (feature, stat) over
    # the feature's own folds, per fold slice
    for f in range(fold_count):
        for k in range(N_HB):
            var base = (f * hist_line + Int(hb_first[k])) * 2
            for s in range(2):
                var run = Float32(0.0)
                for b in range(16):
                    run += want[base + b * 2 + s]
                    want[base + b * 2 + s] = run

    bad = 0
    var first_bad = -1
    for i in range(fold_count * hist_line * 2):
        if h_hist[i] != want[i]:
            bad += 1
            if first_bad < 0:
                first_bad = i
    if bad != 0:
        print(
            "FAIL O5:", bad, "of", fold_count * hist_line * 2,
            "histogram cells wrong at fold count", fold_count,
            "-- first at", first_bad, "got", h_hist[first_bad],
            "want", want[first_bad],
        )
        failures += 1
    else:
        print(
            "  ok   O5 --", fold_count * hist_line * 2,
            "histogram cells exact at FoldCount", fold_count,
            "( stripe", stripe, ", stride", fold_count, "), scanned",
        )

    # =============================================================== O6
    var d_bf = ctx.enqueue_create_buffer[DType.uint32](3 * hist_line)
    var bf_host = List[UInt32]()
    for k in range(N_HB):
        for b in range(16):
            bf_host.append(UInt32(k))
            bf_host.append(UInt32(b))
            bf_host.append(UInt32(0))
    ctx.enqueue_copy(dst_buf=d_bf, src_ptr=bf_host.unsafe_ptr())
    var d_cw = ctx.enqueue_create_buffer[DType.float32](hist_line)
    var d_bw = ctx.enqueue_create_buffer[DType.float32](hist_line)
    var ones = List[Float32]()
    for _ in range(hist_line):
        ones.append(Float32(1.0))
    ctx.enqueue_copy(dst_buf=d_cw, src_ptr=ones.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_bw, src_ptr=ones.unsafe_ptr())
    var d_rid = ctx.enqueue_create_buffer[DType.uint32](2)
    var d_rsc = ctx.enqueue_create_buffer[DType.float32](2)
    # DEVIATION 207: the scorers read scoreBeforeSplit from the device
    var d_sb = ctx.enqueue_create_buffer[DType.float32](1)
    d_sb.enqueue_fill(Float32(0.0))
    ctx.synchronize()

    var supported: List[Int] = [
        SCORE_FUNCTION_SOLAR_L2,
        SCORE_FUNCTION_COSINE,
        SCORE_FUNCTION_NEWTON_COSINE,
    ]
    var refused: List[Int] = [
        SCORE_FUNCTION_L2,
        SCORE_FUNCTION_NEWTON_L2,
        SCORE_FUNCTION_SAT_L2,
        SCORE_FUNCTION_LOO_L2,
    ]
    bad = 0
    for s in range(len(supported)):
        try:
            find_optimal_split(
                ctx, d_bf, hist_line, d_cw, d_bw, hist_line, d_hist,
                subsets.partition_stats, 1, fold_count, d_sb,
                d_rid, d_rsc, 1, supported[s], Float32(1.0), Float32(1.0),
                Float32(0.0), False, Float32(0.0), 0, False,
            )
            ctx.synchronize()
        except e:
            print(
                "FAIL O6: score function", supported[s],
                "has a dynamic kernel upstream but raised:", e,
            )
            bad += 1
    for s in range(len(refused)):
        var raised = False
        try:
            find_optimal_split(
                ctx, d_bf, hist_line, d_cw, d_bw, hist_line, d_hist,
                subsets.partition_stats, 1, fold_count, d_sb,
                d_rid, d_rsc, 1, refused[s], Float32(1.0), Float32(1.0),
                Float32(0.0), False, Float32(0.0), 0, False,
            )
            ctx.synchronize()
        except:
            raised = True
        if not raised:
            print(
                "FAIL O6: score function", refused[s],
                "has NO dynamic kernel upstream"
                " (`pointwise_scores.cu:469` throws) and did not raise at"
                " fold count", fold_count,
            )
            bad += 1
        # and the SAME function at fold count 1 must be accepted, or the
        # refusal above is just a broken score function rather than the
        # ordered-boosting limit
        try:
            find_optimal_split(
                ctx, d_bf, hist_line, d_cw, d_bw, hist_line, d_hist,
                subsets.partition_stats, 1, 1, d_sb,
                d_rid, d_rsc, 1, refused[s], Float32(1.0), Float32(1.0),
                Float32(0.0), False, Float32(0.0), 0, False,
            )
            ctx.synchronize()
        except e:
            print(
                "FAIL O6: score function", refused[s],
                "was refused at fold count 1 too, so O6 proves nothing"
                " about the DYNAMIC arm:", e,
            )
            bad += 1
    # keep the score buffer alive past the last enqueue's synchronize
    # ([[mojo-buffer-freed-at-last-use]])
    _ = d_sb.unsafe_ptr()
    if bad != 0:
        failures += 1
    else:
        print(
            "  ok   O6 -- 3 of 7 score functions run at FoldCount",
            fold_count, ", the other 4 raise and all 4 still run at 1",
        )

    # =============================================================== O4
    # `split_subsets` at FoldBits = 4: the split bit lands at
    # `CurrentDepth + FoldBits` and the fold id keeps bits [0, FoldBits).
    # Feature 3 of the half-byte word, split at bin 7.
    var bin_idx = UInt32(7)
    comptime SPLIT_FEATURE = 3
    var bin_idx_1 = UInt32(6)
    comptime SPLIT_FEATURE_1 = 5
    """TWO DIFFERENT FEATURES, so all four depth-2 leaves are populated.
    Splitting twice on one feature at one bin makes the second bit equal
    the first, leaves 1 and 2 come out empty, and O5b's stripe-versus-
    stride comparison then only has leaf 3 to work with."""

    # the host model starts where O3 left off: identity order, bin = fold id
    var host_idx = List[UInt32]()
    var host_bin = List[UInt32]()
    for f in range(fold_count):
        for _ in range(want_size[f]):
            host_bin.append(UInt32(f))
    for i in range(n_docs):
        host_idx.append(UInt32(i))

    bad = 0
    var moved_docs = 0
    var d_obs = ctx.enqueue_create_buffer[DType.uint32](n_docs)
    for level in range(2):
        # `Gather(groupedByBinObservations, observations, subsets.Indices)`
        # (`oblivious_tree_doc_parallel_structure_searcher.cpp:65`), and it
        # is NOT optional past depth 0. Passing the unpermuted doc array
        # here splits on a DIFFERENT document at every position, and the
        # symptom is a level that partitions into quarters instead of
        # halves while every partition offset still tiles: the second split
        # bit disagrees with the first even though the predicate is
        # identical. That is how this line was found.
        var idx_for_gather = subsets.indices.copy()
        launch_gather_with_mask_u32(
            ctx, d_obs, d_docs, idx_for_gather, n_docs, GATHER_NO_MASK
        )
        var docs2 = d_obs.copy()
        var lvl_feature = SPLIT_FEATURE if level == 0 else SPLIT_FEATURE_1
        var lvl_bin = bin_idx if level == 0 else bin_idx_1
        split_subsets(
            ctx, target, d_ci, docs2,
            UInt32(0), UInt32(15), UInt32(28 - 4 * lvl_feature), False,
            lvl_bin, subsets,
        )
        drain(
            ctx, subsets, hb_bins, hb_idx, hb_parts, hb_stats, hb_gw,
            hb_gt,
        )

        var bit_pos = UInt32(level + fold_bits)
        var nb = List[UInt32]()
        for i in range(n_docs):
            var doc = Int(ids[Int(host_idx[i])])
            var v = (
                h_ci[doc] >> UInt32(28 - 4 * lvl_feature)
            ) & UInt32(15)
            var bit = UInt32(1) if v > lvl_bin else UInt32(0)
            nb.append(host_bin[i] | (bit << bit_pos))
        # ReorderBins is a STABLE one-bit sort: zeros first, order kept
        var sb = List[UInt32]()
        var si = List[UInt32]()
        for want_bit in range(2):
            for i in range(n_docs):
                if ((nb[i] >> bit_pos) & 1) == UInt32(want_bit):
                    sb.append(nb[i])
                    si.append(host_idx[i])
        host_bin = sb^
        host_idx = si^

        var wrong = 0
        for i in range(n_docs):
            if hb_bins[i] != host_bin[i] or hb_idx[i] != host_idx[i]:
                wrong += 1
        if wrong != 0:
            print(
                "FAIL O4: level", level, "--", wrong, "of", n_docs,
                "positions have the wrong (bin, index) pair",
            )
            bad += 1
    for i in range(n_docs):
        if hb_idx[i] != UInt32(i):
            moved_docs += 1
    if moved_docs == 0:
        print(
            "FAIL O4: two splits reordered NOTHING -- this fixture's split"
            " bit is constant and the gate is inert",
        )
        bad += 1

    # partitions and stats, per partition, over 1 << (2 + FoldBits) slots
    var live = 1 << (2 + fold_bits)
    var pos = 0
    var nonempty = 0
    for p in range(live):
        var want_size = 0
        for i in range(n_docs):
            if host_bin[i] == UInt32(p):
                want_size += 1
        var off = Int(hb_parts[p * PARTITION_RECORD + PART_OFFSET])
        var got_size = Int(hb_parts[p * PARTITION_RECORD + PART_SIZE])
        if got_size != want_size:
            print(
                "FAIL O4: partition", p, "size", got_size, "want", want_size
            )
            bad += 1
        if want_size > 0:
            nonempty += 1
            if off != pos:
                print("FAIL O4: partition", p, "offset", off, "want", pos)
                bad += 1
        var sw = Float32(0.0)
        var st = Float32(0.0)
        for i in range(pos, pos + want_size):
            sw += plant_weight(Int(host_idx[i]))
            st += plant_target(Int(host_idx[i]))
        var gw = hb_stats[p * PARTITION_STAT_STRIDE + PART_STAT_WEIGHT]
        var gt = hb_stats[p * PARTITION_STAT_STRIDE + PART_STAT_SUM]
        if want_size > 0 and (gw != sw or gt != st):
            print(
                "FAIL O4: partition", p, "stats (", gw, ",", gt,
                ") want (", sw, ",", st, ")",
            )
            bad += 1
        pos += want_size

    # THE FOLD ID SURVIVES IN THE LOW BITS. Checked against the position's
    # ORIGINAL fold, recovered from the layout, so a split that overwrote
    # the fold bits instead of adding above them fails here even when every
    # partition size is right.
    var fold_of_pos = List[Int]()
    for f in range(fold_count):
        for _ in range(want_size[f]):
            fold_of_pos.append(f)
    var fold_wrong = 0
    for i in range(n_docs):
        if (Int(hb_bins[i]) & (stripe - 1)) != fold_of_pos[Int(host_idx[i])]:
            fold_wrong += 1
    if fold_wrong != 0:
        print(
            "FAIL O4:", fold_wrong, "positions lost their fold id from the"
            " low", fold_bits, "bits",
        )
        bad += 1

    if bad != 0:
        failures += 1
    else:
        print(
            "  ok   O4 -- two splits at FoldBits", fold_bits, ":", n_docs,
            "positions in order,", nonempty, "non-empty partitions of",
            live, ", fold id intact (", moved_docs, "positions moved )",
        )

    # ============================================================== O5b
    # THE STRIPE, WHICH O5 CANNOT SEE. At `part_count == 1` both offsets
    # reduce to `fold`: `GetDataPartitionOffset(0, f) = 0 * 16 + f` and
    # `GetHistogramOffset(0, f) = 0 * 12 + f` are the same number, so O5 is
    # exact whichever one the kernel uses. The two only separate at
    # `leaf > 0`, so this repeats the histogram at DEPTH 2 -- four leaves,
    # twelve folds -- where the partition array is strided by 16 and the
    # histogram by 12. `NEXT_TWO.md`'s "reached but inert" list has an entry
    # for exactly this shape and this gate would have joined it.
    var idx_after = subsets.indices.copy()
    launch_gather_with_mask_u32(
        ctx, d_obs, d_docs, idx_after, n_docs, GATHER_NO_MASK
    )
    ctx.enqueue_memset(d_hist, Float32(0.0))
    ctx.synchronize()

    var deep_parts = 1 << MAX_DEPTH
    compute_hist2(
        ctx, POLICY_HALF_BYTE,
        d_off.unsafe_ptr(), d_first.unsafe_ptr(), d_folds.unsafe_ptr(),
        d_oh.unsafe_ptr(), N_HB, 0, hist_line,
        d_ci.unsafe_ptr(),
        subsets.gathered_target.unsafe_ptr(),
        subsets.gathered_weight.unsafe_ptr(),
        d_obs.unsafe_ptr(),
        n_docs,
        subsets.partitions.unsafe_ptr(),
        deep_parts,
        fold_count,
        d_hist.unsafe_ptr(),
        hist_line,
        True,
        FoldsHistogram(),
        SM_COUNT,
        Float32(1.0),
    )
    ctx.enqueue_copy(dst_buf=h_hist, src_buf=d_hist)
    ctx.synchronize()

    var want2 = List[Float32]()
    for _ in range(deep_parts * fold_count * hist_line * 2):
        want2.append(Float32(0.0))
    # EVERY SPAN HERE IS HOST-DERIVED. Reading the offsets back out of
    # `subsets.Partitions` would make the model share the port's answer,
    # and a fold axis that landed in the wrong partition would agree with
    # itself. `host_bin` is the independent model O4 already checked.
    for i in range(n_docs):
        var pid = Int(host_bin[i])
        # THE STRIPE: the fold id is the low `FoldBits` of the bin and the
        # leaf is what sits above it, so the partition array is strided by
        # `1 << FoldBits`
        var leaf = pid >> fold_bits
        var f = pid & (stripe - 1)
        var row = Int(ids[Int(host_idx[i])])
        var word = h_ci[row]
        for k in range(N_HB):
            var b = Int((word >> UInt32(28 - 4 * k)) & 15)
            # THE STRIDE: the histogram is packed TIGHT at FoldCount
            var at = (
                (leaf * fold_count + f) * hist_line + Int(hb_first[k]) + b
            ) * 2
            want2[at + 0] += plant_weight(Int(host_idx[i]))
            want2[at + 1] += plant_target(Int(host_idx[i]))
    for slice_id in range(deep_parts * fold_count):
        for k in range(N_HB):
            var base = (slice_id * hist_line + Int(hb_first[k])) * 2
            for st2 in range(2):
                var run = Float32(0.0)
                for b in range(16):
                    run += want2[base + b * 2 + st2]
                    want2[base + b * 2 + st2] = run

    bad = 0
    first_bad = -1
    for i in range(deep_parts * fold_count * hist_line * 2):
        if h_hist[i] != want2[i]:
            bad += 1
            if first_bad < 0:
                first_bad = i
    if bad != 0:
        print(
            "FAIL O5b:", bad, "of", deep_parts * fold_count * hist_line * 2,
            "cells wrong at depth", MAX_DEPTH, "x FoldCount", fold_count,
            "-- first at", first_bad, "got", h_hist[first_bad], "want",
            want2[first_bad],
        )
        failures += 1
    else:
        print(
            "  ok   O5b --", deep_parts * fold_count * hist_line * 2,
            "cells exact at", deep_parts, "leaves x", fold_count,
            "folds: partition stride", stripe, ", histogram stride",
            fold_count,
        )

    # =============================================================== O7
    # THE SEARCHER'S TWO REFUSALS. Both are fold-gated, so both are checked
    # WITH folds and the score-function one is checked WITHOUT them too --
    # a guard that fires on every call would pass the first half of this
    # gate and be a different bug (`PORTING_RULES` rule 8).
    var fc_list = List[Int]()
    for _ in range(N_HB):
        fc_list.append(16)
    var lay2 = build_layout(fc_list)
    var d_ci2 = ctx.enqueue_create_buffer[DType.uint32](
        lay2.columns * N_ROWS
    )
    ctx.enqueue_memset(d_ci2, UInt32(0))
    ctx.synchronize()

    bad = 0
    var msg_a = String("")
    try:
        var w1 = ctx.enqueue_create_buffer[DType.float32](n_docs)
        var t1 = ctx.enqueue_create_buffer[DType.float32](n_docs)
        ctx.enqueue_memset(w1, Float32(1.0))
        ctx.enqueue_memset(t1, Float32(1.0))
        ctx.synchronize()
        var pool_a = List[PointwiseTreeWorkspace]()
        _ = fit_oblivious_tree_structure(
            ctx, lay2, N_ROWS, MAX_DEPTH, d_ci2, w1^, t1^, SM_COUNT,
            Float32(1.0), SCORE_FUNCTION_L2, pool_a, folds=folds,
        )
    except e:
        msg_a = String(e)
    if msg_a.find("ordered boosting") < 0:
        print(
            "FAIL O7: L2 with folds did not raise the ordered-boosting"
            " refusal; got:", msg_a,
        )
        bad += 1

    # O7b, REWRITTEN 2026-09-03. This arm used to assert that SolarL2 with
    # folds RAISED "DEVIATION 126": the calcer was constructed without a
    # fold count, so its helpers were built at 1 while the layout was built
    # at the fold count, and the consistency check refused. The call site
    # now passes `fold_count`, so the arm asserts what it was always meant
    # to: A FOLD-BASED TREE GROWS. A supported score function plus folds is
    # no longer a refusal.
    var msg_b = String("")
    var nsplits_b = -1
    try:
        var w2 = ctx.enqueue_create_buffer[DType.float32](n_docs)
        var t2 = ctx.enqueue_create_buffer[DType.float32](n_docs)
        ctx.enqueue_memset(w2, Float32(1.0))
        ctx.enqueue_memset(t2, Float32(1.0))
        ctx.synchronize()
        var pool_b = List[PointwiseTreeWorkspace]()
        var sp_b = fit_oblivious_tree_structure(
            ctx, lay2, N_ROWS, MAX_DEPTH, d_ci2, w2^, t2^, SM_COUNT,
            Float32(1.0), SCORE_FUNCTION_SOLAR_L2, pool_b, folds=folds,
        )
        nsplits_b = len(sp_b)
    except e:
        msg_b = String(e)
    if msg_b != "":
        print(
            "FAIL O7: SolarL2 with folds must now GROW A TREE, not raise;"
            " got:", msg_b,
        )
        bad += 1
    elif nsplits_b < 1:
        print("FAIL O7: SolarL2 with folds grew no splits at all")
        bad += 1

    # the CONTROL: no folds, and the SAME score function L2 must get past
    # the ordered-boosting guard and past DEVIATION 126
    var msg_c = String("")
    var nsplits_c = -1
    try:
        var w3 = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
        var t3 = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
        ctx.enqueue_memset(w3, Float32(1.0))
        ctx.enqueue_memset(t3, Float32(1.0))
        ctx.synchronize()
        var pool_c = List[PointwiseTreeWorkspace]()
        var sp_c = fit_oblivious_tree_structure(
            ctx, lay2, N_ROWS, MAX_DEPTH, d_ci2, w3^, t3^, SM_COUNT,
            Float32(1.0), SCORE_FUNCTION_L2, pool_c,
        )
        nsplits_c = len(sp_c)
    except e:
        msg_c = String(e)
    if msg_c.find("ordered boosting") >= 0 or msg_c.find("DEVIATION 126") >= 0:
        print(
            "FAIL O7: the single-task arm hit a FOLD guard, so neither"
            " guard is fold-gated; got:", msg_c,
        )
        bad += 1
    if bad != 0:
        failures += 1
    else:
        print(
            "  ok   O7 -- the 4-of-7 score refusal fires with folds and not"
            " without; A FOLD-BASED TREE GROWS at FoldCount 12 (",
            nsplits_b, "splits, against", nsplits_c,
            "for the single-task control at the same depth ). The split"
            " counts are NOT required to match: the fold arm scores each"
            " split on a different partition of the data, which is the"
            " point of ordered scoring.",
        )

    if failures == 0:
        print("ordered boosting: OK")
    else:
        raise Error(String(failures) + " ordered-boosting gate(s) failed")
