# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The half-byte histogram at TWO FEATURE GROUPS, per cell, with sabotage.

WHAT THIS FILE ASSERTED BEFORE, AND WHY THAT ASSERTION WAS NEVER TRUE
---------------------------------------------------------------------
It was a diagnostic, written to answer "why does the boosting dataset always
pick bin-feature 0", and it was committed in the same commit that answered
the question: grid x carried replication ALONE, so
`maxBlocksPerPart = gridDim.x / featureBlocks` came out `1 / 2 == 0` and any
policy past one `UInt32` column returned an EMPTY histogram. That question is
closed. What was left behind failed at commits that were known good, and the
cause was in the FIXTURE, not in the kernel:

- ONE host staging buffer, `ho`, was handed to FOUR `enqueue_copy` calls and
  then MUTATED between them, `ho[0] = n_rows` for `p_sz`. Those copies are
  asynchronous, so `p_off[0]`, `ids[0]` and `dense[0]` were never established
  as zero; the partition could begin at row `n_rows` and the writeback could
  address leaf `n_rows`. The check then demanded a histogram it had never
  asked the device to build, which is the sense in which it asserted
  something that was never true. RESUME states the rule it broke outright:
  ONE HOST STAGING BUFFER PER `enqueue_copy`.
- The weight plane was the constant 1.0, so every cell of stat 0 was a COUNT,
  and the gradient was a function of three of the sixteen features, so across
  the other thirteen every cell carried the same expected value. That is the
  shape RESUME warns about in capitals: uniform bins reported 0 wrong of 512
  on a kernel that hashed, scattered bins showed wrong in 490 of 512. A cell
  whose neighbour has the same expected value verifies the total and nothing
  about placement.
- Bins were planted `x % folds[f]`, which never produces the TOP bucket
  `bin == Folds`, so the writeback's `fold < features[fid].Folds`
  (`hist_half_byte.cu:36`) had no reader.
- It had no sabotage arm, so it could not show it reached anything.
- Its second half asserted nothing at all: it computed a host argmax of the
  score, printed it, and returned. `hist_check.check_scores` covers the score
  kernel against a hand calculation, so that half is deleted rather than
  repaired.

WHAT IT ASSERTS NOW, AND WHERE CATBOOST GUARANTEES IT
-----------------------------------------------------
`TPointHistHalfByte::AddToGlobalMemory` (`hist_half_byte.cu:27-53`) writes
into cell `fold` of feature `fid` the sum of one stat over the documents of
THIS partition whose bin is `fold`, for `fold < features[fid].Folds` (`:36`),
taking the value from `Histogram[fid + 8 * fold]` (`:43`). That is a
statement about ONE CELL, so the oracle is a host tally of the same sum,
compared cell by cell, and never a conserved total.

THE SHAPE, AND WHY THIS SHAPE AND NOT ANOTHER
---------------------------------------------
Sixteen features at fifteen folds, which is `boosting_check`'s exact dataset.
Fifteen is `policy_max_folds(HALF_BYTE)`, so all sixteen land in the
half-byte policy, and `FeaturesPerInt` is eight, so the layout takes TWO
compressed-index columns and the launch takes TWO feature groups:
`numBlocks.x = (fCount + 7) / 8` (`hist_half_byte.cu:80`). Group `g` reads
column `g` through `bins += binsLineSize * (blockIdx.x / maxBlocksPerPart)`
(`compute_hist_loop_one_stat.cuh:492`). ONE group is what every check in this
repository used before the column bug, and one group is exactly the case
where the dropped feature-group factor happened to agree.

TWO ARMS, AND THE SECOND IS THE POINT
-------------------------------------
1. The launch, per cell against a host tally. Bins are hashed per row and per
   feature and BOTH stat planes are hashed per row, so a cell's expected
   value differs from its neighbour's on every feature and stat 0 is not a
   count.
2. SABOTAGE: the same launch with every compressed-index column past the
   first ZEROED, which sends every feature of group 1 to bin 0 and leaves
   group 0's bytes untouched. Two things are then required, and both are per
   cell:
     a. group 1's cells MUST MOVE. If they do not, group 1 is never read,
        arm 1 verified half the dataset while claiming all of it, and its
        agreement is vacuous -- which is precisely the state the column bug
        lived in, undetected, through every check in this tree.
     b. group 0's cells must NOT move, bit for bit. At one block per
        partition the reduction order is fixed, so the two launches are
        bit-identical wherever their input is, and any drift there means a
        group is reading across its own column boundary.
   The sabotaged output is then compared against a SECOND host tally, so this
   file says WHERE the histogram moved and not merely that it did.

The one-block store (`hist_half_byte.cu:49`) is what this file exercises;
`checks/replicated_half_byte_check.mojo` is the standing cover for the
`blockCount > 1` flush at `:46-47`.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.fixed_point import choose_scale

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import (
    POLICY_HALF_BYTE,
    policy_for_fold_count,
)
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    upload_scale,
    DeviceBlock,
    launch_histograms_for_blocks,
    upload_blocks,
)


def check_boosting_histogram(
    n_rows: Int = 8192, n_features: Int = 16, n_folds: Int = 15
) raises:
    var ctx = DeviceContext()
    var stat_count = 2

    var folds = List[Int]()
    for _ in range(n_features):
        folds.append(n_folds)
    for f in range(n_features):
        if policy_for_fold_count(folds[f]) != POLICY_HALF_BYTE:
            raise Error(
                "feature " + String(f) + " did not land in the half-byte"
                " policy; this check would be testing a different kernel"
            )

    var lay = build_layout(folds)
    var blocks = blocks_for(lay, n_rows)
    if len(blocks) != 1:
        raise Error(
            "expected exactly one policy block, got " + String(len(blocks))
        )
    var dblocks = upload_blocks(ctx, blocks)
    var n_bf = lay.hist_cells

    # REACH, decided before anything runs. `numBlocks.x = (fCount + 7) / 8`
    # (`hist_half_byte.cu:80`), restated here rather than imported from the
    # thing under test: a check that reads its own expectation out of the
    # code it is checking agrees with that code's bugs.
    var groups = (n_features + 7) // 8
    if groups < 2:
        raise Error(
            "this launch has " + String(groups) + " feature group, and the"
            " multi-group grid is the whole subject of this file. Raise"
            " n_features above 8"
        )
    if lay.columns < 2:
        raise Error(
            "the layout took " + String(lay.columns) + " compressed-index"
            " column, so the sabotage arm has no second column to zero"
        )
    print(
        "  feature groups", groups, " compressed-index columns", lay.columns,
        " flat bin-features", n_bf,
    )

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * lay.columns)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows * lay.columns)
    for i in range(n_rows * lay.columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var host_bin = List[List[Int]]()
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        var col = List[Int]()
        ref cf = lay.features[f]
        for r in range(n_rows):
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            # `folds + 1` bin values, so the TOP bucket exists and is the one
            # the writeback's `fold < features[fid].Folds`
            # (`hist_half_byte.cu:36`) drops. The version of this file that
            # planted `x % folds[f]` never produced it, so that guard had no
            # reader at all.
            var v = Int(x % UInt32(folds[f] + 1))
            col.append(v)
            hb.unsafe_ptr().unsafe_store(r, UInt8(v))
        host_bin.append(col^)
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            bins.unsafe_ptr(), Int32(n_rows), cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    # ---- the stat planes, hashed on BOTH ---------------------------------
    # Plane 0 carries weights near 1 rather than exactly 1. A constant weight
    # plane makes every cell of stat 0 a COUNT, and a count is the one thing
    # a wrong-placement bug still gets right whenever the totals match. The
    # gradient no longer derives from three of the sixteen features either,
    # which left thirteen features with a flat expected profile.
    var stats = ctx.enqueue_create_buffer[DType.float32](stat_count * n_rows)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](
        stat_count * n_rows
    )
    var host_w = List[Float64]()
    var host_g = List[Float64]()
    var w_mag = Float64(0.0)
    var g_mag = Float64(0.0)
    for r in range(n_rows):
        var x = UInt32(r * 2246822519 + 0x9E3779B9)
        x ^= x << 13
        x ^= x >> 17
        x ^= x << 5
        var w = 0.5 + Float64(Int(x % UInt32(1000))) / 1000.0
        var g = Float64(Int(x % UInt32(2001))) / 1000.0 - 1.0
        host_w.append(w)
        host_g.append(g)
        hs.unsafe_ptr().unsafe_store(r, Float32(w))
        hs.unsafe_ptr().unsafe_store(n_rows + r, Float32(g))
        w_mag += w
        if g < 0.0:
            g_mag += -g
        else:
            g_mag += g
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    # `choose_scale`'s contract is a sum of MAGNITUDES that bounds every
    # partial the device forms, and one scale serves both planes, so the
    # bound is the larger. At one block per partition the Int32 flush is not
    # taken at all -- `hist_half_byte.cu:46-49` takes the plain store when
    # `blockCount == 1` -- but a scale that would still be legal if it were
    # taken is the only kind worth passing: an unbounded one wrapped that
    # accumulator in silence once already.
    var mag = w_mag
    if g_mag > mag:
        mag = g_mag
    var scale = choose_scale(mag)
    var scale_keep = upload_scale(ctx, Float32(scale))
    var fixed_scale = rebind[MutPointer[Float32, MutAnyOrigin]](
        scale_keep.unsafe_ptr()
    )
    print("  sum of magnitudes: weights", w_mag, " gradients", g_mag)
    print("  fixed_scale", scale)

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())

    # ---- the partitions --------------------------------------------------
    # TWO live partitions, so a partition OFFSET is exercised and not only a
    # size, and `partition.Offset` is the field the old fixture could not
    # establish.
    #
    # ONE HOST STAGING BUFFER PER `enqueue_copy`. These copies are
    # asynchronous. The version of this file that shared a single `ho` across
    # four of them and rewrote `ho[0]` in between is the reason it failed at
    # commits that were known good: `p_off`, `ids` and `dense` raced against
    # the store meant only for `p_sz`.
    var max_leaves = 4
    var off = List[Int]()
    var siz = List[Int]()
    off.append(0)
    siz.append((n_rows * 5) // 8)
    off.append(siz[0])
    siz.append(n_rows - siz[0])
    var n_live = len(off)

    var p_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var dense = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var h_ids = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var h_dense = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_ids.unsafe_ptr().unsafe_store(i, UInt32(i))
        h_dense.unsafe_ptr().unsafe_store(i, UInt32(i))
    for i in range(n_live):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(off[i]))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(siz[i]))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ids, src_ptr=h_ids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dense, src_ptr=h_dense.unsafe_ptr())

    var hist_cells = max_leaves * stat_count * n_bf
    var hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
    var acc = ctx.enqueue_create_buffer[DType.int32](hist_cells)
    var zf = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)
    var zi = ctx.enqueue_create_host_buffer[DType.int32](hist_cells)
    for i in range(hist_cells):
        zf.unsafe_ptr().unsafe_store(i, Float32(0.0))
        zi.unsafe_ptr().unsafe_store(i, Int32(0))

    var widest = 1
    for b in range(len(blocks)):
        var tf = 0
        for k in range(blocks[b].count()):
            tf += Int(blocks[b].folds[k])
        if tf > widest:
            widest = tf
    var block_hist = ctx.enqueue_create_buffer[DType.float32](
        max_leaves * stat_count * widest
    )
    ctx.synchronize()

    # ---- the two host tallies, once --------------------------------------
    # `[leaf][stat][flat bin]`, the layout the bridge leaves behind.
    #
    # `tally_mag` is the SUM OF MAGNITUDES of the same cell and is what the
    # tolerance is built from. The gradient plane is signed and cancels, so a
    # cell's VALUE can sit near zero while the Float32 error of summing it
    # does not; a relative tolerance keyed on the value would then be tighter
    # than the arithmetic can deliver and would fail on correct output.
    #
    # `tally_bad` is the same sum with every feature whose compressed-index
    # column is past the first forced to bin 0, which is exactly what zeroing
    # those columns does to it. It is a full second EXPECTATION rather than a
    # delta, so the sabotage arm is checked per cell like the first one and
    # this file can say WHERE the histogram moved.
    var tally = List[List[Float64]]()
    var tally_mag = List[List[Float64]]()
    var tally_bad = List[List[Float64]]()
    var tally_bad_mag = List[List[Float64]]()
    for k in range(n_live):
        var t = List[Float64]()
        var tm = List[Float64]()
        var tb = List[Float64]()
        var tbm = List[Float64]()
        for _ in range(stat_count * n_bf):
            t.append(0.0)
            tm.append(0.0)
            tb.append(0.0)
            tbm.append(0.0)
        for r in range(off[k], off[k] + siz[k]):
            var wr = host_w[r]
            var gr = host_g[r]
            var gm = gr
            if gm < 0.0:
                gm = -gm
            for f in range(n_features):
                var base = Int(lay.features[f].first_fold_index)
                var b = host_bin[f][r]
                # the writeback's `fold < features[fid].Folds`
                # (`hist_half_byte.cu:36`): the top bucket has no
                # bin-feature and is dropped.
                if b < folds[f]:
                    t[base + b] += wr
                    tm[base + b] += wr
                    t[n_bf + base + b] += gr
                    tm[n_bf + base + b] += gm
                # SABOTAGE: a zeroed column reads bin 0 for every feature it
                # packs, and `build_layout` packs `FeaturesPerInt` of them
                # per column in policy order.
                var bb = b
                if Int(lay.features[f].offset) > 0:
                    bb = 0
                if bb < folds[f]:
                    tb[base + bb] += wr
                    tbm[base + bb] += wr
                    tb[n_bf + base + bb] += gr
                    tbm[n_bf + base + bb] += gm
        tally.append(t^)
        tally_mag.append(tm^)
        tally_bad.append(tb^)
        tally_bad_mag.append(tbm^)

    # One quantum of the fixed point per contributing block plus one for the
    # host's own rounding. At one block the flush is not even taken, so this
    # is slack, not a fudge: a cell that landed in the wrong slot misses by
    # the whole cell.
    var quantum_tol = Float64(3.0) / scale

    # ---- arm 1, the launch as built --------------------------------------
    var good = run_hist_arm(
        ctx, dblocks, n_live, n_rows, stat_count, max_leaves, 1, fixed_scale,
        cindex, row_index, stats, p_off, p_sz, ids, dense,
        hist, acc, block_hist, n_bf, zf, zi,
    )
    var wrong_good = compare_cells(
        good, tally, tally_mag, n_live, stat_count, n_bf, quantum_tol,
        String("as built"),
    )

    # ---- arm 2, SABOTAGE: zero every column past the first ---------------
    # Read the compressed index back rather than rebuilding it on the host,
    # so what is sabotaged is the bytes the kernel actually reads and not a
    # host reimplementation of the packing.
    var ci_good = ctx.enqueue_create_host_buffer[DType.uint32](
        n_rows * lay.columns
    )
    ctx.enqueue_copy(dst_ptr=ci_good.unsafe_ptr(), src_buf=cindex)
    ctx.synchronize()
    var ci_bad = ctx.enqueue_create_host_buffer[DType.uint32](
        n_rows * lay.columns
    )
    for i in range(n_rows * lay.columns):
        if i < n_rows:
            ci_bad.unsafe_ptr().unsafe_store(
                i, ci_good.unsafe_ptr().unsafe_load(i)
            )
        else:
            ci_bad.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=ci_bad.unsafe_ptr())
    ctx.synchronize()

    var broken = run_hist_arm(
        ctx, dblocks, n_live, n_rows, stat_count, max_leaves, 1, fixed_scale,
        cindex, row_index, stats, p_off, p_sz, ids, dense,
        hist, acc, block_hist, n_bf, zf, zi,
    )
    var wrong_broken = compare_cells(
        broken, tally_bad, tally_bad_mag, n_live, stat_count, n_bf,
        quantum_tol,
        String("columns past the first zeroed (SABOTAGE)"),
    )

    # Which cells moved, split by the feature group that owns them. Exact
    # inequality is the right test: one block per partition fixes the
    # reduction order, so the two launches are bit-identical wherever their
    # input is.
    var moved_g0 = 0
    var moved_g1 = 0
    for k in range(n_live):
        for f in range(n_features):
            var base = Int(lay.features[f].first_fold_index)
            for s in range(stat_count):
                for b in range(folds[f]):
                    var c = k * stat_count * n_bf + s * n_bf + base + b
                    var a = good.unsafe_ptr().unsafe_load(c)
                    var d = broken.unsafe_ptr().unsafe_load(c)
                    if a != d:
                        if Int(lay.features[f].offset) > 0:
                            moved_g1 += 1
                        else:
                            moved_g0 += 1
    print(
        "    cells moved by the sabotage: column 0", moved_g0,
        " columns past the first", moved_g1,
    )

    if wrong_good != 0:
        raise Error(
            String("the half-byte histogram disagrees with the host tally in ")
            + String(wrong_good)
            + " cells of "
            + String(n_live * stat_count * n_bf)
            + ". `AddToGlobalMemory` (`hist_half_byte.cu:27-53`) must leave"
            " in each cell the sum of that stat over this partition's"
            " documents whose bin is that fold"
        )
    if moved_g1 == 0:
        raise Error(
            "THE SABOTAGE ARM DID NOT MOVE. Zeroing every compressed-index"
            " column past the first sends every feature of group 1 to bin 0,"
            " so its cells MUST change. That they did not means the second"
            " group is never read: `bins += binsLineSize * (blockIdx.x /"
            " maxBlocksPerPart)` (`compute_hist_loop_one_stat.cuh:492`) is"
            " not reaching the second column, and arm 1 verified half the"
            " dataset while claiming all of it"
        )
    if moved_g0 != 0:
        raise Error(
            String("the sabotage moved ") + String(moved_g0)
            + " cells belonging to features in column 0, which was not"
            " touched. At one block per partition the two launches are"
            " bit-identical wherever their input is, so a group reading"
            " across its own column boundary is the only way this happens"
        )
    if wrong_broken != 0:
        raise Error(
            String("the sabotaged histogram moved, but not to where zeroing")
            + " the later columns puts it: " + String(wrong_broken)
            + " cells disagree with the second host tally. The launch does"
            " reach the second column, so the fault is in the"
            " group-to-column or the fold-offset arithmetic"
        )

    print(
        "  half-byte histogram matches the host tally at", groups,
        "feature groups, and the sabotage moves", moved_g1,
        "cells past column 0 and none inside it, so both groups are reached",
    )
    _ = scale_keep^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


def run_hist_arm(
    ctx: DeviceContext,
    mut dblocks: List[DeviceBlock],
    n_live: Int,
    n_rows: Int,
    stat_count: Int,
    max_leaves: Int,
    sm_count: Int,
    fixed_scale: MutPointer[Float32, MutAnyOrigin],
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
    mut dense_ids: DeviceBuffer[DType.uint32],
    mut hist: DeviceBuffer[DType.float32],
    mut acc: DeviceBuffer[DType.int32],
    mut block_hist: DeviceBuffer[DType.float32],
    hist_cells_per_leaf: Int,
    zf: HostBuffer[DType.float32],
    zi: HostBuffer[DType.int32],
) raises -> HostBuffer[DType.float32]:
    """One launch, read back to the host.

    Both scratch planes are re-zeroed first. The writeback is guarded by
    `if (abs(val) > 1e-20f)` (`hist_half_byte.cu:45`), so a cell whose value
    is zero is NEVER WRITTEN and keeps whatever the buffer held; an arm that
    leaves a cell at zero would otherwise inherit the previous arm's, and the
    sabotage arm zeroes bins for a living.

    `depth = 0` selects the DIRECT load
    (`compute_hist_loop_one_stat.cuh:490-494`), which is the overload whose
    `bins += binsLineSize * (blockIdx.x / maxBlocksPerPart)` this file
    sabotages. The gather overload does the same thing at `:557-558` through
    `CompressedIndexOffset`.
    """
    ctx.enqueue_copy(dst_buf=hist, src_ptr=zf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=acc, src_ptr=zi.unsafe_ptr())
    ctx.synchronize()

    launch_histograms_for_blocks(
        ctx, dblocks, 0, n_live, n_rows, stat_count, max_leaves,
        sm_count, fixed_scale,
        cindex, row_index, stats, p_off, p_sz, ids, dense_ids,
        hist, acc, block_hist, hist_cells_per_leaf,
    )
    ctx.synchronize()

    var cells = max_leaves * stat_count * hist_cells_per_leaf
    var out = ctx.enqueue_create_host_buffer[DType.float32](cells)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()
    return out^


def compare_cells(
    got: HostBuffer[DType.float32],
    tally: List[List[Float64]],
    tally_mag: List[List[Float64]],
    n_live: Int,
    stat_count: Int,
    hist_cells: Int,
    quantum_tol: Float64,
    name: String,
) raises -> Int:
    """Cell by cell against a host tally, keyed on the cell's magnitude.

    Never a conserved total. RESUME records what a total is worth here: the
    same kernel at the same parameters reported 0 wrong of 512 on uniform
    bins and 490 wrong of 512 on scattered hashed ones.
    """
    var wrong = 0
    var worst = Float64(0.0)
    var worst_cell = -1
    for k in range(n_live):
        for c in range(stat_count * hist_cells):
            var want = tally[k][c]
            var have = Float64(
                got.unsafe_ptr().unsafe_load(k * stat_count * hist_cells + c)
            )
            var d = have - want
            if d < 0.0:
                d = -d
            var tol = quantum_tol + 1.0e-4 * tally_mag[k][c]
            if d > tol:
                wrong += 1
                if d > worst:
                    worst = d
                    worst_cell = k * stat_count * hist_cells + c
    print(
        "    arm", name, ":", wrong, "wrong of",
        n_live * stat_count * hist_cells, " worst |delta|", worst,
        "at cell", worst_cell,
    )
    return wrong


def main() raises:
    # STANDALONE DRIVER, the same call `probe_main.mojo` makes under
    # "half-byte histogram at TWO FEATURE GROUPS vs a host tally (GPU)".
    # Defaults only: 8192 rows x 16 half-byte features at 15 folds is the
    # shape that takes TWO compressed-index columns, which is the whole
    # point of the fixture -- a smaller feature count would run one group
    # and check nothing this file exists for.
    print("half-byte histogram at TWO FEATURE GROUPS vs a host tally (GPU):")
    check_boosting_histogram()
