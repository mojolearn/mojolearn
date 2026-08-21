"""A REPLICATED half-byte histogram against a host tally.

THE GAP THIS FILLS, stated as the commit that found it stated it: "EVERY
half-byte check runs at `grid_dim=(1,1,1)`, so the multi-block path of that
kernel has never been executed by any test." `mixed_hist_probe` covers 16
half-byte features but at one block per partition; `permuted_ids_check`
covers a non-identity leaf list, also at one block. The arm that only runs
when `activeBlockCount > 1` -- the fixed-point Int32 flush that stands in for
CatBoost's `atomicAdd` (`hist_half_byte.cu:45-51`) -- had no reader.

What went through the gap: `doc_parallel_boosting.fit` passed `0.0` for both
stat magnitudes, the scale derivation answered 268,435,455 for that, and
every replicated flush saturated `Int32` and wrapped. The depth-0 weight
histogram came back as -1.5e-08, -3.0e-08, -4.5e-08, -6.0e-08, which is
-4, -8, -12, -16 over 268,435,455: four blocks each contributing INT32_MAX,
summed to 8,589,934,588, read back as -4, then walked out by the prefix scan.

THREE ARMS, AND THE THIRD IS THE POINT
--------------------------------------
1. ONE BLOCK. `sm_count = 1` collapses `replication_for` to 1, so the
   writeback takes the plain float store. This is the arm every existing
   check already runs, and it is here as the control.
2. MANY BLOCKS. `sm_count` large enough to force eight replicas, so the
   partitions large enough for it take the Int32 flush. Must agree with the
   same host tally, within the quantum the fixed point costs.
3. SABOTAGE. The same many-block launch with a deliberately unbounded scale.
   Its expectation FOLLOWS THE BUILD, because the two flushes fail in
   opposite directions. Where the flush is fixed point the arm must come
   back WRONG, since nothing else here separates a reached Int32 flush from
   an unreached one. Where the flush is CatBoost's float `atomicAdd` the
   scale is dead input and the arm must move NOTHING, which is the positive
   statement that the Int32 path is gone rather than merely quiet; reach
   then rests on arm 2, and arm 2 earns it, because a plain store in place
   of the atomic keeps one block's partial and drops three.

   A check that cannot fail on a broken kernel is a check that passed a
   broken kernel. A check that fails on a CORRECT kernel because it asserts
   dead code is live is the same defect wearing the other sign, and this
   file shipped that version for exactly one commit.

The bins are HASHED per row and per feature and the stats are hashed per
row, so a cell's expected value differs from its neighbour's. A uniform
plant verifies the total and nothing about placement.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from mojo_only.fixed_point import choose_scale

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
from gbdt.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
    BUILD_MODE,
    TARGET_COLUMN,
)
from mojo_only.kernel_matrix import deterministic_flush_for
from mojo_only.numerics import NUMERIC_IDENTICAL

#: WHICH FLUSH THIS BUILD ACTUALLY COMPILED, read from the same expression the
#: kernel branches on (`hist_half_byte.mojo`, both writeback sites). Arm 3 has
#: to know it, because the two flushes fail in OPPOSITE directions and a
#: sabotage aimed at the wrong one proves nothing.
comptime FLUSH_IS_FIXED_POINT = deterministic_flush_for[
    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
]()


#: `THist::BlockLoadSize(ECIndexLoadType::Direct)`, i.e.
#: `LoadSize * BlockSize * Unroll` = `4 * BLOCK_SIZE * 1`
#: (`compute_hist_loop_one_stat.cuh:45`). It is what decides
#: `activeBlockCount`, so the check has to know it to know whether it
#: replicated anything.
comptime MIN_DOCS_PER_BLOCK = 4 * BLOCK_SIZE

#: The scale a broken caller produced: `(1 << 28) - 1` divided by a magnitude
#: of 1.0. Arm 3 hands this to the kernel on purpose.
comptime UNBOUNDED_SCALE = Float32(Float64((1 << 28) - 1))


def check_replicated_half_byte() raises:
    var ctx = DeviceContext()

    # Sixteen features at 15 folds. 15 is `policy_max_folds(HALF_BYTE)`, so
    # every one lands in the half-byte policy and this check launches that
    # kernel and no other, at the same shape `boosting_check` trains on.
    var folds = List[Int]()
    for _ in range(16):
        folds.append(15)
    var n_features = len(folds)
    for f in range(n_features):
        if policy_for_fold_count(folds[f]) != POLICY_HALF_BYTE:
            raise Error(
                "feature " + String(f) + " did not land in the half-byte"
                " policy; this check would be testing a different kernel"
            )

    var stat_count = 2
    var max_leaves = 4

    # Three partitions, sized so the level mixes block counts: one that
    # replicates four ways, one that replicates two ways, and one that
    # cannot replicate at all. The last is a control INSIDE the same launch:
    # it takes the plain store while its neighbours take the atomic.
    var off = List[Int]()
    var siz = List[Int]()
    off.append(0)
    siz.append(4 * MIN_DOCS_PER_BLOCK)
    off.append(4 * MIN_DOCS_PER_BLOCK)
    siz.append(MIN_DOCS_PER_BLOCK + 1)
    off.append(5 * MIN_DOCS_PER_BLOCK + 1)
    siz.append(MIN_DOCS_PER_BLOCK - 1)
    var n_live = len(off)
    var n_rows = off[n_live - 1] + siz[n_live - 1]

    var lay = build_layout(folds)
    var blocks = blocks_for(lay, n_rows)
    if len(blocks) != 1:
        raise Error(
            "expected exactly one policy block, got " + String(len(blocks))
        )
    var dblocks = upload_blocks(ctx, blocks)

    # ---- the compressed index, hashed ----------------------------------
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
            # the writeback's `fold < features[fid].Folds` guard drops. A
            # plant that never produces it cannot see that guard.
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

    # ---- the stat planes, hashed too -----------------------------------
    # Plane 0 carries weights near 1 rather than exactly 1: a constant weight
    # plane makes every cell of stat 0 a COUNT, and a count is the one thing
    # a wrong-placement bug still gets right when the totals happen to match.
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

    # The contract `choose_scale` is specified against: sum of MAGNITUDES per
    # plane, and one scale serves both, so the bound is the larger.
    var mag = w_mag
    if g_mag > mag:
        mag = g_mag
    var scale = choose_scale(mag)
    var scale_keep = upload_scale(ctx, Float32(scale))
    var unbounded_keep = upload_scale(ctx, UNBOUNDED_SCALE)
    var fixed_scale = rebind[MutPointer[Float32, MutAnyOrigin]](
        scale_keep.unsafe_ptr()
    )
    print("  sum of magnitudes: weights", w_mag, " gradients", g_mag)
    print("  fixed_scale", scale, " quantum", 1.0 / scale)

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())

    var p_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hzz = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        ho.unsafe_ptr().unsafe_store(i, UInt32(0))
        hzz.unsafe_ptr().unsafe_store(i, UInt32(0))
    for i in range(n_live):
        ho.unsafe_ptr().unsafe_store(i, UInt32(off[i]))
        hzz.unsafe_ptr().unsafe_store(i, UInt32(siz[i]))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=hzz.unsafe_ptr())

    var ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var dense_ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var hid = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        hid.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=ids, src_ptr=hid.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dense_ids, src_ptr=hid.unsafe_ptr())

    var hist_cells = max_leaves * stat_count * lay.hist_cells
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

    # ---- the host tally, once ------------------------------------------
    # `[leaf][stat][flat bin]`, the layout the bridge leaves behind.
    #
    # `tally_mag` is the SUM OF MAGNITUDES of the same cell, and it is what
    # the tolerance is built from. The gradient plane is signed and cancels,
    # so a cell's VALUE can be near zero while the Float32 error of summing
    # it is not; a relative tolerance keyed on the value would then be
    # tighter than the arithmetic can deliver, and the check would fail on
    # correct output. Keyed on the magnitude it tracks the real error.
    var tally = List[List[Float64]]()
    var tally_mag = List[List[Float64]]()
    for k in range(n_live):
        var t = List[Float64]()
        var tm = List[Float64]()
        for _ in range(stat_count * lay.hist_cells):
            t.append(0.0)
            tm.append(0.0)
        for r in range(off[k], off[k] + siz[k]):
            for f in range(n_features):
                var b = host_bin[f][r]
                # the writeback's `fold < features[fid].Folds`: the top
                # bucket has no bin-feature and is dropped.
                if b < folds[f]:
                    var cell = Int(lay.features[f].first_fold_index) + b
                    t[cell] += host_w[r]
                    tm[cell] += host_w[r]
                    t[lay.hist_cells + cell] += host_g[r]
                    if host_g[r] < 0.0:
                        tm[lay.hist_cells + cell] += -host_g[r]
                    else:
                        tm[lay.hist_cells + cell] += host_g[r]
        tally.append(t^)
        tally_mag.append(tm^)

    # ---- REACH, computed before anything runs --------------------------
    # `activeBlockCount = min(ceil(size / minDocsPerBlock), maxBlocksPerPart)`
    # (`compute_hist_loop_one_stat.cuh`, the direct overload). If this is 1
    # everywhere then arm 2 is arm 1 and the whole file is decoration.
    var groups = (n_features + 7) // 8
    var many_sm = 48
    var many_rep = replicas_for_check(groups, n_live, stat_count, many_sm)
    var one_rep = replicas_for_check(groups, n_live, stat_count, 1)
    var max_active = 1
    for k in range(n_live):
        var a = (siz[k] + MIN_DOCS_PER_BLOCK - 1) // MIN_DOCS_PER_BLOCK
        if a > many_rep:
            a = many_rep
        print("    leaf", k, "size", siz[k], "-> active blocks", a)
        if a > max_active:
            max_active = a
    print("  replicas: one-block arm", one_rep, " many-block arm", many_rep)
    if one_rep != 1:
        raise Error(
            "the one-block arm asked for " + String(one_rep)
            + " replicas; it is not a control"
        )
    if max_active < 2:
        raise Error(
            "no partition in this level replicates, so the Int32 flush is"
            " never reached and this check cannot see the bug it exists for."
            " Raise the partition sizes or the replica count"
        )

    # A truncation per block plus one for the host's own rounding. The
    # fixed-point path is EXACT in its addition and inexact only in the
    # `Int32(val * scale)` conversion, so the error is bounded by the number
    # of contributing blocks, not by the number of rows.
    var quantum_tol = Float64(max_active + 2) / scale

    var one = run_arm(
        ctx, dblocks, n_live, n_rows, stat_count, max_leaves, 1, fixed_scale,
        cindex, row_index, stats, p_off, p_sz, ids, dense_ids,
        hist, acc, block_hist, lay.hist_cells, zf, zi,
    )
    var many = run_arm(
        ctx, dblocks, n_live, n_rows, stat_count, max_leaves, many_sm,
        fixed_scale,
        cindex, row_index, stats, p_off, p_sz, ids, dense_ids,
        hist, acc, block_hist, lay.hist_cells, zf, zi,
    )
    var broken = run_arm(
        ctx, dblocks, n_live, n_rows, stat_count, max_leaves, many_sm,
        rebind[MutPointer[Float32, MutAnyOrigin]](unbounded_keep.unsafe_ptr()),
        cindex, row_index, stats, p_off, p_sz, ids, dense_ids,
        hist, acc, block_hist, lay.hist_cells, zf, zi,
    )

    var wrong_one = compare(
        one, tally, tally_mag, n_live, stat_count, lay.hist_cells,
        quantum_tol,
        String("one block"),
    )
    var wrong_many = compare(
        many, tally, tally_mag, n_live, stat_count, lay.hist_cells,
        quantum_tol,
        String("many blocks"),
    )
    var wrong_broken = compare(
        broken, tally, tally_mag, n_live, stat_count, lay.hist_cells,
        quantum_tol,
        String("many blocks, unbounded scale (SABOTAGE)"),
    )

    if wrong_one != 0:
        raise Error(
            String("the ONE-BLOCK half-byte histogram is already wrong in ")
            + String(wrong_one)
            + " cells, so nothing below it means anything. Fix the kernel"
            " before reading the replicated arms"
        )
    if wrong_many != 0:
        raise Error(
            String("the REPLICATED half-byte histogram is wrong in ")
            + String(wrong_many)
            + " cells while the one-block arm is exact. The Int32 flush that"
            " stands in for their `atomicAdd` (`hist_half_byte.cu:45-51`) is"
            " not summing the partials correctly, or the scale handed to it"
            " does not bound them"
        )
    # ================ ARM 3 DEPENDS ON WHICH FLUSH COMPILED ================
    # `fixed_scale` only reaches arithmetic on the Int32 path. Under the float
    # atomic it is loaded and never used, so an unbounded value is inert BY
    # CONSTRUCTION and "the sabotage moved nothing" is the CORRECT result
    # rather than a missing reader. Asserting movement in that build asserts
    # that dead code is live, which is how this check failed the moment the
    # flush row was restored to CatBoost's `atomicAdd`.
    #
    # So each build gets the sabotage that can actually bite it:
    #
    #   FIXED POINT  the unbounded scale MUST move cells. Nothing else in
    #                this file distinguishes a reached Int32 flush from an
    #                unreached one.
    #   FLOAT ATOMIC the unbounded scale must move NOTHING, which is the
    #                positive statement that the Int32 path is genuinely gone
    #                and not merely quiet. Reach then comes from arm 2 on its
    #                own, and arm 2 is a real reach test here: with four
    #                blocks landing on one leaf's cells, a plain store in
    #                place of `atomicAdd` keeps ONE block's partial and drops
    #                the rest, so arm 2 fails. That is not hypothetical. It
    #                is the defect that was sitting in `hist_one_byte.mojo`
    #                at both writeback sites, unreachable only because this
    #                row was pinned, and it surfaced the instant it was not.
    # =======================================================================
    @parameter
    if FLUSH_IS_FIXED_POINT:
        if wrong_broken == 0:
            raise Error(
                "THE SABOTAGE ARM PASSED. An unbounded `fixed_scale` must"
                " overflow the Int32 accumulator and produce wrong cells;"
                " that it did not means the replicated launch never took the"
                " Int32 path at all, and the agreement between the other two"
                " arms is vacuous. This check is not reaching what it claims"
                " to check"
            )
        print(
            "  replicated half-byte histogram matches the host tally, and"
            " the sabotage arm moves", wrong_broken,
            "cells, so the Int32 flush is reached",
        )
    else:
        if wrong_broken != 0:
            raise Error(
                String("the unbounded `fixed_scale` moved ")
                + String(wrong_broken)
                + " cells in a build whose flush is CatBoost's float"
                " `atomicAdd` (`hist_half_byte.cu:45-51`). Nothing should"
                " read `fixed_scale` on that path, so a value that changes"
                " the answer means the Int32 accumulator is still live"
                " somewhere it should not be"
            )
        print(
            "  replicated half-byte histogram matches the host tally on"
            " CatBoost's float atomic, and an unbounded `fixed_scale` moves"
            " nothing, so the Int32 path is gone rather than quiet"
        )
    _ = scale_keep^
    _ = unbounded_keep^


def replicas_for_check(
    groups: Int, n_live: Int, stat_count: Int, sm_count: Int
) -> Int:
    """`replication_for`, restated so the check can predict its own grid.

    Deliberately a SECOND copy and not an import: this file's job is to say
    what the launch should do, and reading the answer out of the thing under
    test is how a check agrees with a bug. It is four lines and their
    formula (`hist_half_byte.cu:81`).
    """
    var base = groups * n_live * stat_count
    if base < 1:
        base = 1
    var rep = (2 * sm_count + base - 1) // base
    if rep < 1:
        rep = 1
    return rep


def run_arm(
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
    """One launch at a given `sm_count` and scale, read back to the host.

    Both scratch planes are re-zeroed first. `fixed_to_float_kernel` clears
    the accumulator as it converts, but only where it found a non-zero, so an
    arm that leaves a cell at zero would otherwise inherit the previous arm's
    -- which is precisely the class of bug this file is about.

    `depth = 0` selects the DIRECT load. The gather variant shares the
    writeback verbatim, so the flush is covered either way; if that ever
    stops being true this takes a `depth` argument.
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


def compare(
    got: HostBuffer[DType.float32],
    tally: List[List[Float64]],
    tally_mag: List[List[Float64]],
    n_live: Int,
    stat_count: Int,
    hist_cells: Int,
    quantum_tol: Float64,
    name: String,
) raises -> Int:
    """Cell-by-cell against the host tally, with the fixed point's own bound.

    The tolerance is a QUANTUM COUNT plus a relative term, not a fudge
    factor: the Int32 path loses at most one truncation per contributing
    block, and Float32 accumulation over a leaf's rows carries a relative
    error. A wrapped accumulator misses by the whole cell, so nothing about
    this tolerance makes the failure it was written for survivable.
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
    # "REPLICATED half-byte histogram vs a host tally (GPU)". Its third arm
    # sabotages the fixed-point scale and requires the result to move.
    print("REPLICATED half-byte histogram vs a host tally (GPU):")
    check_replicated_half_byte()
