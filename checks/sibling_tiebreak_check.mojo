# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SMALLER-SIBLING TIE-BREAK, and what an exact size tie costs.

CatBoost's `BuildNecessaryHistograms` picks ONE of each sibling pair to
COMPUTE and derives the other by subtracting from the parent
(`split_properties_helper.cpp:1318-1324`):

    if (firstLeaf.Size < secondLeaf.Size) { smallLeafId = ids[0]; bigLeafId = ids[1]; }
    else                                  { smallLeafId = ids[1]; bigLeafId = ids[0]; }

`ids` is filled in ASCENDING leaf index (`:1300`) and `MakeSplit` gives the
LEFT child the existing (lower) id and the RIGHT child the appended (higher)
one (`:861-862`, `:976-977`). So `ids[0]` is the LEFT child, `ids[1]` is the
RIGHT child, the comparison is a STRICT `<` on the LEFT, and **on an exact
size tie the `else` branch fires and the RIGHT child is the one computed.**

This port had it inverted from the day the planner was written (`409a16c`,
2026-08-19, host; carried onto the device by DEVIATION 94 at `2bbe6af`,
2026-08-21): a tie kept the LEFT child as the computed one. PORTING.md 136.

WHAT THIS FILE CHECKS, and why each part is here.

1. `check_planner_matches_theirs` -- the shipped device planner
   (`kernel/split_resolve.plan_level_kernel`) on four crafted sibling pairs,
   two of them exact ties. THE SABOTAGE is `plan_level_inverted_kernel`
   below: the pre-fix comparison, verbatim, run on the same input. It must
   disagree on EXACTLY the tied pairs and agree everywhere else. A gate no
   sabotage moves is decorative.

2. `measure_subtraction_blast_radius` -- the arithmetic question the old
   comment answered by reasoning and got wrong. The claim was that the Int32
   fixed-point accumulator makes the sibling subtraction EXACT, so which
   sibling is computed "cannot change a histogram bit". It does not follow.
   The accumulator is Int32 and exact, but `write_reduces_from_fixed_kernel`
   converts it with `Float32(Int(q)) / fixed_scale` BEFORE the subtraction,
   and `substract_histograms_kernel` then works in float32.
   `choose_scale` targets 2^30, so `q` routinely exceeds float32's
   exact-integer range of 2^24 and `Float32(Int(q))` ROUNDS. Once parent and
   child are each rounded, `parent_f - left_f` need not equal `right_f`.
   This function measures how often, on the real kernels, with a real scale.

3. `measure_fit` -- a 24-tree fit on a fixture whose sibling sizes tie at
   every binary split, printing the tree fingerprint and COUNTING the ties
   rather than assuming them. Run this file before and after the tie-break
   flip and diff the fingerprint: that is the "did a chosen split change"
   measurement.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import fit
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    launch_histograms_for_blocks,
    upload_blocks,
    upload_scale,
)
from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    substract_histograms_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_resolve import (
    RESOLVE_BLOCK_SIZE,
    plan_level_kernel,
)
from gbdt.models.oblivious_model import BIN_SPLIT_TAKE_BIN, TAdditiveModel
from checks.fixed_point import choose_scale


def plan_level_inverted_kernel(
    part_size: MutPointer[UInt32, MutAnyOrigin],
    half_in: Int32,
    ids_compute: MutPointer[UInt32, MutAnyOrigin],
    sub_from: MutPointer[UInt32, MutAnyOrigin],
    sub_what: MutPointer[UInt32, MutAnyOrigin],
):
    """THE SABOTAGE: `plan_level_kernel` as it stood before PORTING.md 136.

    Byte for byte the old body. `small` starts at the LEFT child and only
    moves when the right is STRICTLY smaller, so a tie keeps the left --
    the inversion of their `:1318`. Nothing in the product calls this; it
    exists so the assertion above it can be shown to fail on demand.
    """
    var half = Int(half_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < half:
        var left_sz = part_size.unsafe_load(i)
        var right_sz = part_size.unsafe_load(half + i)
        var small = i
        var big = half + i
        if right_sz < left_sz:
            small = half + i
            big = i
        ids_compute.unsafe_store(i, UInt32(small))
        sub_from.unsafe_store(i, UInt32(big))
        sub_what.unsafe_store(i, UInt32(small))


def run_planner(
    ctx: DeviceContext,
    sizes: List[Int],
    half: Int,
    inverted: Bool,
) raises -> List[Int]:
    """Both planners behind one call, returning `compute` then `from`."""
    var n = len(sizes)
    var d_sz = ctx.enqueue_create_buffer[DType.uint32](n)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](n)
    for i in range(n):
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(sizes[i]))
    ctx.enqueue_copy(dst_buf=d_sz, src_ptr=h_sz.unsafe_ptr())

    var d_c = ctx.enqueue_create_buffer[DType.uint32](half)
    var d_f = ctx.enqueue_create_buffer[DType.uint32](half)
    var d_w = ctx.enqueue_create_buffer[DType.uint32](half)
    ctx.enqueue_memset(d_c, UInt32(0xDEAD))
    ctx.enqueue_memset(d_f, UInt32(0xDEAD))
    ctx.enqueue_memset(d_w, UInt32(0xDEAD))
    ctx.synchronize()

    var grid = (half + RESOLVE_BLOCK_SIZE - 1) // RESOLVE_BLOCK_SIZE
    if inverted:
        ctx.enqueue_function[plan_level_inverted_kernel](
            d_sz.unsafe_ptr(), Int32(half),
            d_c.unsafe_ptr(), d_f.unsafe_ptr(), d_w.unsafe_ptr(),
            grid_dim=(grid, 1, 1),
            block_dim=(RESOLVE_BLOCK_SIZE, 1, 1),
        )
    else:
        ctx.enqueue_function[plan_level_kernel](
            d_sz.unsafe_ptr(), Int32(half),
            d_c.unsafe_ptr(), d_f.unsafe_ptr(), d_w.unsafe_ptr(),
            grid_dim=(grid, 1, 1),
            block_dim=(RESOLVE_BLOCK_SIZE, 1, 1),
        )
    ctx.synchronize()

    var hc = ctx.enqueue_create_host_buffer[DType.uint32](half)
    var hf = ctx.enqueue_create_host_buffer[DType.uint32](half)
    var hw = ctx.enqueue_create_host_buffer[DType.uint32](half)
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=d_c)
    ctx.enqueue_copy(dst_ptr=hf.unsafe_ptr(), src_buf=d_f)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=d_w)
    ctx.synchronize()

    var out = List[Int]()
    for i in range(half):
        out.append(Int(hc.unsafe_ptr().unsafe_load(i)))
    for i in range(half):
        out.append(Int(hf.unsafe_ptr().unsafe_load(i)))
    for i in range(half):
        out.append(Int(hw.unsafe_ptr().unsafe_load(i)))
    _ = h_sz^
    _ = hc^
    _ = hf^
    _ = hw^
    _ = d_sz^
    _ = d_c^
    _ = d_f^
    _ = d_w^
    return out^


def check_planner_matches_theirs() raises:
    """`:1318` on four pairs, two of them EXACT TIES."""
    print("PLANNER TIE-BREAK, against split_properties_helper.cpp:1318")
    var ctx = DeviceContext()
    var half = 4
    var sizes = List[Int]()
    # left sizes                       # right sizes
    sizes.append(100)   # pair 0 TIE
    sizes.append(50)    # pair 1 left smaller
    sizes.append(100)   # pair 2 right smaller
    sizes.append(0)     # pair 3 TIE at zero
    sizes.append(100)
    sizes.append(100)
    sizes.append(50)
    sizes.append(0)

    # THEIRS, by hand: small = left iff left < right, else right.
    var want_small = List[Int]()
    var want_big = List[Int]()
    var is_tie = List[Bool]()
    for i in range(half):
        var l = sizes[i]
        var r = sizes[half + i]
        is_tie.append(l == r)
        if l < r:
            want_small.append(i)
            want_big.append(half + i)
        else:
            want_small.append(half + i)
            want_big.append(i)

    var got = run_planner(ctx, sizes, half, False)
    var bad = run_planner(ctx, sizes, half, True)

    var wrong = 0
    var sabotage_moved = 0
    var sabotage_moved_on_tie = 0
    for i in range(half):
        var g_small = got[i]
        var g_big = got[half + i]
        var g_what = got[2 * half + i]
        print(
            "  pair", i,
            " left", sizes[i], " right", sizes[half + i],
            " -> compute", g_small, " subtract-from", g_big,
            " (theirs: compute", want_small[i], ")",
            " TIE" if is_tie[i] else "",
        )
        if g_small != want_small[i] or g_big != want_big[i]:
            wrong += 1
        if g_what != g_small:
            wrong += 1
        if bad[i] != g_small:
            sabotage_moved += 1
            if is_tie[i]:
                sabotage_moved_on_tie += 1

    print("  pairs disagreeing with CatBoost:", wrong, "of", half)
    print(
        "  SABOTAGE (the pre-136 comparison) moved", sabotage_moved,
        "pairs,", sabotage_moved_on_tie, "of them ties; ties present:",
        2,
    )
    if wrong != 0:
        raise Error(
            String("the smaller-sibling tie-break disagrees with CatBoost on ")
            + String(wrong)
            + " pairs; on an exact tie THEY compute the RIGHT child"
            " (split_properties_helper.cpp:1318-1324)"
        )
    if sabotage_moved != 2 or sabotage_moved_on_tie != 2:
        raise Error(
            String("the sabotage did not move what it must: expected it to")
            + " flip exactly the 2 tied pairs, it flipped "
            + String(sabotage_moved)
            + " pairs ("
            + String(sabotage_moved_on_tie)
            + " tied). A gate no sabotage moves is decorative."
        )
    print("  the planner computes the RIGHT child on a tie, as theirs does")
    print("  and the inverted comparison fails this check on exactly the ties")


# ---------------------------------------------------------------------------
# THE FIXTURE. Its whole purpose is to make sibling sizes tie EXACTLY.
#
# `CELLS` binary columns hold the bits of `row >> LOG_R`, so the dataset is a
# full factorial design with `R` replicate rows per cell: every leaf a chain
# of binary splits can reach holds the same number of rows, and every further
# binary split halves it exactly. The wide columns are a function of
# `row & (R - 1)` alone, so they are independent of every binary column and
# of every leaf those columns can carve out; they are there to make the
# histogram WIDE (a one-byte policy block of 254 folds) rather than to carry
# signal.
# ---------------------------------------------------------------------------

comptime FIX_BITS = 8
comptime FIX_LOG_R = 4
comptime FIX_R = 1 << FIX_LOG_R
comptime FIX_ROWS = (1 << FIX_BITS) * FIX_R
comptime FIX_WIDE = 4
comptime FIX_WIDE_FOLDS = 254


def fixture_folds() -> List[Int]:
    var folds = List[Int]()
    for _ in range(FIX_BITS):
        folds.append(1)
    for _ in range(FIX_WIDE):
        folds.append(FIX_WIDE_FOLDS)
    return folds^


def fixture_bin(f: Int, r: Int) -> Int:
    """The compressed-index value of column `f` at row `r`."""
    if f < FIX_BITS:
        return (r >> (FIX_LOG_R + f)) & 1
    # A wide column: 16 distinct bins spread over 0..253, a function of the
    # within-cell index only.
    var k = r & (FIX_R - 1)
    var g = f - FIX_BITS
    var v = ((k * 7 + g * 3) % FIX_R) * 15 + g
    return v


def fixture_target(r: Int) -> Float32:
    """Signal on the binary columns, a hashed wobble on top, and a positive
    mean -- the mean is what keeps a histogram cell's accumulator large, and
    a cell that never leaves float32's exact-integer range cannot show the
    defect this file measures."""
    var y = Float64(2.0)
    for f in range(FIX_BITS):
        var b = (r >> (FIX_LOG_R + f)) & 1
        if b == 1:
            y += Float64(f + 1) * 0.11
    var x = UInt32(r * 2654435761 + 0x2545F491)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    y += (Float64(Int(x % UInt32(2048))) / 2048.0 - 0.5) * 0.4
    return Float32(y)


def build_fixture_cindex(
    ctx: DeviceContext,
    mut cindex: DeviceBuffer[DType.uint32],
    lay: CompressedIndexLayout,
) raises:
    var n_features = FIX_BITS + FIX_WIDE
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](FIX_ROWS)
    var bins = ctx.enqueue_create_buffer[DType.uint8](FIX_ROWS)
    for f in range(n_features):
        ref cf = lay.features[f]
        for r in range(FIX_ROWS):
            hb.unsafe_ptr().unsafe_store(r, UInt8(fixture_bin(f, r)))
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * FIX_ROWS), cf.mask, cf.shift,
            bins.unsafe_ptr(), Int32(FIX_ROWS), cindex.unsafe_ptr(),
            grid_dim=(FIX_ROWS + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()
    _ = hb^
    _ = bins^


def ulp_gap(a: Float32, b: Float32) -> Int:
    """Distance in representable float32 steps. Both operands here are
    finite and same-signed in practice; a sign straddle is counted through
    zero rather than as a bit difference."""
    var ia = Int(bitcast[DType.int32](a))
    var ib = Int(bitcast[DType.int32](b))
    if ia < 0:
        ia = -2147483648 - ia
    if ib < 0:
        ib = -2147483648 - ib
    var d = ia - ib
    if d < 0:
        d = -d
    return d


def measure_subtraction_blast_radius() raises:
    """DOES SWAPPING THE COMPUTED SIBLING CHANGE A HISTOGRAM BIT?

    One parent and its two children, all three built by the SHIPPED
    histogram path at a scale the shipped path would choose, then the
    SHIPPED `substract_histograms_kernel` run both ways:

        derived RIGHT = parent - computed LEFT     (what this port did)
        derived LEFT  = parent - computed RIGHT    (what CatBoost does)

    and each compared BIT FOR BIT against the sibling actually built.
    """
    print("SUBTRACTION BLAST RADIUS, on the shipped kernels")
    var ctx = DeviceContext()
    var stat_count = 2
    var max_leaves = 4
    var folds = fixture_folds()
    var n_features = len(folds)

    var lay = build_layout(folds)
    var blocks = blocks_for(lay, FIX_ROWS)
    var dblocks = upload_blocks(ctx, blocks)

    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        FIX_ROWS * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    ctx.synchronize()
    build_fixture_cindex(ctx, cindex, lay)

    # stat 0 = weights (all 1), stat 1 = gradients (the first round's
    # residuals against a zero cursor, which is the target itself).
    var stats = ctx.enqueue_create_buffer[DType.float32](
        stat_count * FIX_ROWS
    )
    var hs = ctx.enqueue_create_host_buffer[DType.float32](
        stat_count * FIX_ROWS
    )
    var wmag = Float64(0.0)
    var gmag = Float64(0.0)
    for r in range(FIX_ROWS):
        var g = fixture_target(r)
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(FIX_ROWS + r, g)
        wmag += 1.0
        gmag += Float64(g) if g >= Float32(0.0) else -Float64(g)
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    var row_index = ctx.enqueue_create_buffer[DType.uint32](FIX_ROWS)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](FIX_ROWS)
    for r in range(FIX_ROWS):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())

    # THE TIE, made of real rows: leaf 0 is the parent (every row), leaf 1
    # the left half and leaf 2 the right half, which on this factorial
    # layout is exactly the top binary column's split. |left| == |right|.
    var off = List[Int]()
    var siz = List[Int]()
    off.append(0)
    siz.append(FIX_ROWS)
    off.append(0)
    siz.append(FIX_ROWS // 2)
    off.append(FIX_ROWS // 2)
    siz.append(FIX_ROWS - FIX_ROWS // 2)
    off.append(0)
    siz.append(0)

    var p_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hz = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        ho.unsafe_ptr().unsafe_store(i, UInt32(off[i]))
        hz.unsafe_ptr().unsafe_store(i, UInt32(siz[i]))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=hz.unsafe_ptr())

    var n_live = 3
    var ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var dense_ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var hid = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        hid.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=ids, src_ptr=hid.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dense_ids, src_ptr=hid.unsafe_ptr())

    var cells = lay.hist_cells
    var hist_cells = max_leaves * stat_count * cells
    var hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
    var acc = ctx.enqueue_create_buffer[DType.int32](hist_cells)
    ctx.enqueue_memset(hist, Float32(0.0))
    ctx.enqueue_memset(acc, Int32(0))

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

    # The scale the boosting loop would choose for this round:
    # `choose_scale(max(sum|w|, sum|g|), n_rows)` (`run_tree_layout`).
    var mag = wmag
    if gmag > mag:
        mag = gmag
    var scale = Float32(choose_scale(mag, FIX_ROWS))
    var scale_keep = upload_scale(ctx, scale)
    var scale_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        scale_keep.unsafe_ptr()
    )

    # depth 1 so the GATHER load path runs, which is the path every level
    # the planner plans for actually takes.
    launch_histograms_for_blocks(
        ctx, dblocks, 1, n_live, FIX_ROWS, stat_count, max_leaves,
        10, scale_ptr, cindex, row_index, stats, p_off, p_sz, ids,
        dense_ids, hist, acc, block_hist, cells,
    )
    ctx.synchronize()

    var base = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)
    ctx.enqueue_copy(dst_ptr=base.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()

    var from_ids = ctx.enqueue_create_buffer[DType.uint32](1)
    var what_ids = ctx.enqueue_create_buffer[DType.uint32](1)
    var h_from = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var h_what = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var work = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)
    var derived = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)

    # Their `CopyHistogram` puts the parent in BOTH children's slots, so
    # either sibling may be the one subtracted from. Slot 3 stands in for
    # whichever slot the derived child owns.
    var diff_cells = List[Int]()
    var worst_ulp = List[Int]()
    var max_q = Float64(0.0)
    for arm in range(2):
        var computed = 1 + arm          # 1 = left computed, 2 = right
        var other = 2 - arm
        for i in range(hist_cells):
            work.unsafe_ptr().unsafe_store(
                i, base.unsafe_ptr().unsafe_load(i)
            )
        # slot 3 <- the parent, cell for cell, both stats
        for s in range(stat_count):
            for c in range(cells):
                work.unsafe_ptr().unsafe_store(
                    3 * stat_count * cells + s * cells + c,
                    base.unsafe_ptr().unsafe_load(s * cells + c),
                )
        ctx.enqueue_copy(dst_buf=hist, src_ptr=work.unsafe_ptr())
        h_from.unsafe_ptr().unsafe_store(0, UInt32(3))
        h_what.unsafe_ptr().unsafe_store(0, UInt32(computed))
        ctx.enqueue_copy(dst_buf=from_ids, src_ptr=h_from.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=what_ids, src_ptr=h_what.unsafe_ptr())
        ctx.synchronize()

        ctx.enqueue_function[substract_histograms_kernel](
            from_ids.unsafe_ptr(), what_ids.unsafe_ptr(),
            Int32(cells), hist.unsafe_ptr(),
            grid_dim=((cells + 255) // 256, 1, stat_count),
            block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=derived.unsafe_ptr(), src_buf=hist)
        ctx.synchronize()

        var differ = 0
        var worst = 0
        for s in range(stat_count):
            for c in range(cells):
                var got = derived.unsafe_ptr().unsafe_load(
                    3 * stat_count * cells + s * cells + c
                )
                var want = base.unsafe_ptr().unsafe_load(
                    other * stat_count * cells + s * cells + c
                )
                var pq = Float64(
                    base.unsafe_ptr().unsafe_load(s * cells + c)
                ) * Float64(scale)
                if pq < 0.0:
                    pq = -pq
                if pq > max_q:
                    max_q = pq
                if bitcast[DType.uint32](got) != bitcast[DType.uint32](want):
                    differ += 1
                    var u = ulp_gap(got, want)
                    if u > worst:
                        worst = u
        diff_cells.append(differ)
        worst_ulp.append(worst)
        print(
            "  computed", "LEFT " if arm == 0 else "RIGHT",
            "-> derived", "RIGHT" if arm == 0 else "LEFT ",
            ": cells differing bit-for-bit from the built sibling",
            differ, "of", stat_count * cells,
            " worst ulp", worst,
        )

    print(
        "  fixed_scale", scale, " largest |accumulator cell|", max_q,
        " float32 exact-integer limit 16777216.0",
    )
    var moved = diff_cells[0] + diff_cells[1]
    print(
        "  HISTOGRAM CELLS THE TIE-BREAK CHOICE MOVES:", moved,
        "of", 2 * stat_count * cells, "compared",
    )
    _ = scale_keep^
    _ = hs^
    _ = hi^
    _ = ho^
    _ = hz^
    _ = hid^
    _ = h_from^
    _ = h_what^
    _ = work^
    _ = derived^
    _ = base^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


def measure_fit(n_trees: Int, depth: Int, pointwise: Bool) raises:
    """A REAL FIT on the tie-forcing fixture, with the ties COUNTED.

    Prints a fingerprint of every tree. Run this file on both sides of the
    tie-break flip and diff the fingerprints: an identical fingerprint means
    no chosen split and no leaf value moved.
    """
    var ctx = DeviceContext()
    var folds = fixture_folds()
    var n_features = len(folds)
    var lay = build_layout(folds)

    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        FIX_ROWS * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    ctx.synchronize()
    build_fixture_cindex(ctx, cindex, lay)

    var targets = ctx.enqueue_create_buffer[DType.float32](FIX_ROWS)
    var weights = ctx.enqueue_create_buffer[DType.float32](FIX_ROWS)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](FIX_ROWS)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](FIX_ROWS)
    for r in range(FIX_ROWS):
        ht.unsafe_ptr().unsafe_store(r, fixture_target(r))
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var model = TAdditiveModel()
    var losses = fit(
        model, ctx, FIX_ROWS, folds, depth, cindex, targets, weights, False,
        n_trees, Float32(0.3), Float32(3.0), True,
        use_pointwise_searcher=pointwise,
    )

    var name = String("pointwise") if pointwise else String("greedy-subsets")
    print(
        "FIT on the tie-forcing fixture [", name, "]:",
        FIX_ROWS, "rows,", n_features, "features, depth", depth, ",",
        model.size(), "trees",
    )
    print("  final loss", losses[len(losses) - 1])

    # THE TIE COUNT, from the fit's OWN splits rather than from the claim
    # that a factorial design must tie. Leaf index bit d is the decision at
    # depth d, which is what makes the planner's pair (i, i + 2^d).
    var pairs = 0
    var ties = 0
    var empty_pairs = 0
    for t in range(model.size()):
        ref sp = model.weak_models[t].structure.splits
        var d_max = len(sp)
        # row -> leaf, one level at a time
        var leaf = List[Int]()
        for _ in range(FIX_ROWS):
            leaf.append(0)
        for d in range(d_max):
            var f = Int(sp[d].feature_id)
            var b = Int(sp[d].bin_idx)
            var take_bin = Int(sp[d].split_type) == BIN_SPLIT_TAKE_BIN
            var half = 1 << d
            var size = List[Int]()
            for _ in range(1 << (d + 1)):
                size.append(0)
            for r in range(FIX_ROWS):
                var v = fixture_bin(f, r)
                var goes_right: Bool
                if take_bin:
                    goes_right = v == b
                else:
                    goes_right = v > b
                if goes_right:
                    leaf[r] = leaf[r] + half
                size[leaf[r]] = size[leaf[r]] + 1
            # The planner runs at the NEXT level over these pairs, so a
            # split at depth d contributes 2^d pairs -- except the last,
            # whose children no level ever plans for.
            if d < d_max - 1:
                for i in range(half):
                    pairs += 1
                    if size[i] == size[half + i]:
                        ties += 1
                        if size[i] == 0:
                            empty_pairs += 1
        var line = String("  tree ") + String(t) + " "
        for d in range(d_max):
            line += (
                String("(")
                + String(Int(sp[d].feature_id))
                + (
                    String("==")
                    if Int(sp[d].split_type) == BIN_SPLIT_TAKE_BIN
                    else String(">")
                )
                + String(Int(sp[d].bin_idx))
                + String(")")
            )
        ref lv = model.weak_models[t].leaf_values
        var acc = UInt64(1469598103934665603)
        for i in range(len(lv)):
            acc ^= UInt64(Int(bitcast[DType.uint32](lv[i])))
            acc *= UInt64(1099511628211)
        line += String(" leaves#") + hex(acc)
        print(line)

    print(
        "  SIBLING PAIRS THE PLANNER SAW:", pairs,
        " EXACT SIZE TIES:", ties,
        " (of which both-empty:", empty_pairs, ")",
    )
    if pairs == 0:
        raise Error("no sibling pair was planned; the fixture is degenerate")
    if ties == 0:
        raise Error(
            "this fixture forced NO exact sibling tie, so it measures"
            " nothing about the tie-break; the factorial construction is"
            " broken"
        )


def main() raises:
    # THE MEASUREMENTS RUN FIRST and the gate last, deliberately: the gate
    # RAISES, and a gate that raises before the numbers are printed turns a
    # measurement run into a stack trace.
    measure_subtraction_blast_radius()
    print()
    measure_fit(24, 6, False)
    print()
    measure_fit(24, 6, True)
    print()
    check_planner_matches_theirs()
