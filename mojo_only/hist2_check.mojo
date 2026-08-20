"""The hist_2 two-stat one-byte family against the PASS family, cell for cell.

WHAT THIS PROVES. CatBoost's one-byte dispatch sends every `maxBins <= 128`
shape to the fused two-stat `TPointHist2OneByte` family
(`hist_one_byte.cu:314-328`) and only 129-255 to the one-stat
`TPointHistOneByte` PASS family this repository ported first. The two
families are independent code with different shared-memory layouts, different
sync disciplines and different reductions, and they must produce THE SAME
histogram. This check runs both on one dataset PER BIT VARIANT -- bits 5
(`max_folds = 25`), bits 6 (`max_folds = 60`) and bits 7 (`max_folds = 100`,
the arm that includes CatBoost's own GPU default border count) -- and demands
EXACT, cell-for-cell agreement -- with each other and with a host tally --
on the direct arm at depth 0 and the gather arm at depth 1, with a permuted
row index on the gather arm so the indirection is load-bearing. Enumerating
the variants is PORTING_RULES 8: `bits` selects a kernel, so the checks
enumerate it.

WHY EXACTNESS IS AVAILABLE HERE, IN EITHER BUILD MODE. Every stat value is
an INTEGER (hashed, scattered, but integral), every cell's sum of magnitudes
stays far below 2^24, and `fixed_scale` is a power of two. Integer-valued
Float32 sums below 2^24 are exact under ANY summation order, float atomics
included, and `hist2_quantize(val, 64.0, u)` of such a value is exact too
for any dither u < 1 (an integral input has fraction zero, and the
quantizer compares the fraction rather than adding u into the value, which
is what makes that claim exact rather than approximate). So under
`NUMERIC_FAST` (CatBoost's float atomic) and `NUMERIC_IDENTICAL` (the Int32
flush) alike, both families must land on the identical bits, and the compare
below is `!=`, not a tolerance. Hashed per-row values, never uniform:
a uniform plant verifies the total and nothing about placement, and it has
passed broken kernels in this repository twice.

REACH, per PORTING_RULES 7 and 8, because agreement between two arms is
vacuous if the dispatch quietly ran the same kernel twice:

1. FAMILY FINGERPRINT. The two families READ an out-of-contract bin
   differently, from their own source: PASS drops any `bin >= (1 << Bits)`
   (`(bin >> Bits) == 0`, `hist_one_byte.cu:86`), while hist_2 drops ONLY
   the skip mark `bin == (1 << Bits)` and folds everything else through
   `bin & ((1 << Bits) - 1)` (`hist_2_one_byte_5bit.cu:61-63`, `_6bit.cu:69`,
   `_7bit.cu:62-65`). A planted `bin = (1 << Bits) + 1` (33 / 65 / 129)
   therefore lands in fold 1 under hist_2 and lands NOWHERE under PASS. The
   check plants 64 such rows, runs BOTH families on the planted index, and
   requires the dispatch's output to differ from the PASS family's output at
   exactly the predicted cells. A dispatch still routing to the PASS family
   cannot pass this: its two outputs would be identical.

2. STAT SABOTAGE. One row's gradient in the multi-block leaf and one in the
   single-block leaf are poisoned by +4096, and the dispatch's output must
   move at exactly the predicted (leaf, stat 1, feature-bin) cells by
   exactly 4096 -- one leaf per flush branch (`blockCount > 1` atomic /
   single-block store, `hist_2_one_byte_base.cuh:135-141`), so reach is
   per-branch.

The level mixes block counts on purpose: leaf 0 replicates in every arm and
leaf 2 cannot replicate in any, so the atomic and the plain-store flush are
both exercised INSIDE each arm, for both families AND both accumulation
modes (each arm's `BlockLoadSize` is derived from its own block size, so the
reach is predicted per arm below rather than assumed shared).

3. ACCUMULATION MODES, per the `hist_smem_mode_for` matrix row
   (PORTING_RULES 8: the row selects a kernel variant, so the checks
   enumerate it). BOTH modes -- CatBoost's warp-private float
   (`HIST_SMEM_WARP_PRIVATE_F32`) and the Apple/bit-identical 2-warp-shared
   Int32 (`HIST_SMEM_SHARED2_I32`) -- run in this one binary through
   `launch_hist2_one_byte[bits, mode]`, on the direct and gather arms, and
   must agree EXACTLY with each other, with the host tally and with the
   dispatch: the integer arm quantizes `Int32(val * 64.0)` of
   integer-valued Float32, which is exact, so `!=` still applies.
   Reach per mode is proven by a SCALE FINGERPRINT: a run with
   `fixed_scale` poisoned to 2^30 must move the Int32 arm (its shared-
   memory quantization wraps) and must NOT move the float arm under the
   float flush (nothing on that path reads the scale). The DISPATCH arm is
   run with the same poisoned scale and must move exactly iff the matrix
   row (or the integer flush) says the build quantizes -- that is the
   expectation FOLLOWING THE BUILD, and it is what proves the dispatch
   launched the mode the row selects. The stat sabotage of REACH 2 is also
   repeated per mode, so both flush branches are covered inside each mode.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from ported.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from ported.gpu_data.feature_blocks import blocks_for
from ported.gpu_data.grid_policy import (
    POLICY_ONE_BYTE,
    policy_for_fold_count,
)
from ported.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from mojo_only.kernel_matrix import (
    HIST_SMEM_SHARED2_I32,
    HIST_SMEM_WARP_PRIVATE_F32,
    TARGET_COLUMN,
    deterministic_flush_for,
)
from mojo_only.numerics import NUMERIC_IDENTICAL
from ported.methods.greedy_subsets_searcher.greedy_search_helper import (
    DeviceBlock,
    launch_hist2_one_byte,
    launch_histograms_for_blocks,
    launch_one_byte,
    upload_blocks,
)
from ported.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    fixed_to_float_kernel,
    write_reduces_histograms_kernel,
    zero_buffer_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_base import (
    BUILD_MODE as HIST2_BUILD_MODE,
    HIST2_MIN_DOCS_PER_BLOCK,
    HIST2_SMEM_MODE,
    hist2_min_docs,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    ONE_BYTE_BLOCK_SIZE,
    one_byte_block_size,
)

#: `TPointHistOneByte::BlockLoadSize(Direct)` = `LoadSize * BlockSize *
#: Unroll` = `4 * 256 * 4` (`hist_one_byte.cu:52-54` with the modern arch
#: arms; constants in `hist_one_byte.mojo`). What decides the PASS family's
#: `activeBlockCount`, restated here so the check predicts its own reach.
comptime PASS_MIN_DOCS_PER_BLOCK = 4 * ONE_BYTE_BLOCK_SIZE * 4

#: The PASS family's shared-Int32 arm doubles the block, so its
#: `BlockLoadSize` doubles with it; the bits-8 fixture below sizes its
#: partitions from the larger of the two, exactly as the hist_2 fixture
#: does with `MIN_DOCS_I32`.
comptime PASS_MIN_DOCS_I32 = 4 * one_byte_block_size[
    HIST_SMEM_SHARED2_I32
]() * 4

comptime N_FEATURES = 8
comptime STAT_COUNT = 2
comptime MAX_LEAVES = 4
comptime SM_COUNT = 48

#: Power of two, so `Int32(val * FIXED_SCALE)` of an integer-valued Float32
#: is exact and the fixed-point arm quantizes NOTHING. See the module
#: docstring's exactness argument.
comptime FIXED_SCALE = Float32(64.0)

#: How many rows carry the family-fingerprint bin. The bin itself is per
#: config: the first value past each width's skip mark (33 / 65 / 129),
#: which hist_2 folds to fold 1 and PASS drops outright.
comptime FINGERPRINT_ROWS = 64

#: Integral sabotage, big enough that a moved cell cannot be mistaken for
#: noise, small enough that every sum stays exact (PORTING.md 20).
comptime POISON = 4096

#: The per-mode `BlockLoadSize`s. The Int32 mode's doubled block doubles its
#: value, so the partition sizes below are derived from the LARGER of the
#: two and reach is predicted per mode.
comptime MIN_DOCS_F32 = hist2_min_docs[HIST_SMEM_WARP_PRIVATE_F32]()
comptime MIN_DOCS_I32 = hist2_min_docs[HIST_SMEM_SHARED2_I32]()

#: The scale fingerprint: 2^30 wraps any cell holding more than a couple of
#: units of stat, so a mode that quantizes in shared memory cannot return
#: its good-run answer, and a mode that never reads the scale cannot move.
comptime POISON_SCALE = Float32(1073741824.0)

#: Whether the FLOAT-accumulation arm reads `fixed_scale` anywhere: only in
#: its writeback, and only under the integer flush. The scale-fingerprint
#: expectations follow the build through this and `HIST2_SMEM_MODE`.
comptime FLUSH_IS_FIXED = deterministic_flush_for[
    TARGET_COLUMN, HIST2_BUILD_MODE == NUMERIC_IDENTICAL
]()


def hash_bin(r: Int, f: Int, folds: Int) -> Int:
    """The hashed bin of row `r`, feature `f`: scattered, never uniform."""
    var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return Int(x % UInt32(folds + 1))


def hash_w(r: Int) -> Int:
    """Integer weight in [1, 7]."""
    var x = UInt32(r * 2246822519 + 0x9E3779B9)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return 1 + Int(x % UInt32(7))


def hash_g(r: Int) -> Int:
    """Integer gradient in [-7, 7]: signed, so a placement bug cannot hide
    behind a count."""
    var x = UInt32(r * 3266489917 + 0x85EBCA6B)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return Int(x % UInt32(15)) - 7


def pass_cell_of(bin: Int, bits: Int, folds: Int) -> Int:
    """Which fold the PASS family counts `bin` into, or -1 for none:
    `(bin >> Bits) == 0` (`hist_one_byte.cu:86`), then the writeback's
    `fold < Folds` guard."""
    if (bin >> bits) != 0:
        return -1
    if bin >= folds:
        return -1
    return bin


def hist2_cell_of(bin: Int, bits: Int, folds: Int) -> Int:
    """Which fold the hist_2 family counts `bin` into, or -1 for none:
    `pass = bin != (1 << bits)` -- exactly the skip mark, nothing wider --
    and the slot keeps `bin & ((1 << bits) - 1)`
    (`hist_2_one_byte_5bit.cu:61-63`, `_6bit.cu:63-69`, `_7bit.cu:61-65`),
    then the writeback's `fold < Folds` guard."""
    if bin == (1 << bits):
        return -1
    var cell = bin & ((1 << bits) - 1)
    if cell >= folds:
        return -1
    return cell


def host_tally(
    bins: List[List[Int]],
    w: List[Int],
    g: List[Int],
    off: List[Int],
    siz: List[Int],
    perm: List[Int],
    use_perm: Bool,
    hist2_family: Bool,
    lay: CompressedIndexLayout,
    bits: Int,
    folds: Int,
) raises -> List[Float64]:
    """The exact expected flat histogram, `[leaf][stat][flat bin]`, under one
    family's bin semantics. Gather semantics per
    `compute_hist_loop_two_stats.cuh:134-137`: the BIN comes through the
    index, the STAT stays positional."""
    var cells = lay.hist_cells
    var out = List[Float64]()
    for _ in range(MAX_LEAVES * STAT_COUNT * cells):
        out.append(0.0)
    var n_live = len(off)
    for k in range(n_live):
        for pos in range(off[k], off[k] + siz[k]):
            var brow = pos
            if use_perm:
                brow = perm[pos]
            for f in range(N_FEATURES):
                var b = bins[f][brow]
                var cell = -1
                if hist2_family:
                    cell = hist2_cell_of(b, bits, folds)
                else:
                    cell = pass_cell_of(b, bits, folds)
                if cell < 0:
                    continue
                var flat = Int(lay.features[f].first_fold_index) + cell
                out[k * STAT_COUNT * cells + flat] += Float64(w[pos])
                out[k * STAT_COUNT * cells + cells + flat] += Float64(g[pos])
    return out^


def diff_cells(
    got: HostBuffer[DType.float32],
    other: HostBuffer[DType.float32],
    total: Int,
) raises -> List[Int]:
    """Every flat index where two arms disagree. Exact `!=`; the module
    docstring is the argument for why no tolerance belongs here."""
    var out = List[Int]()
    for i in range(total):
        if got.unsafe_ptr().unsafe_load(i) != other.unsafe_ptr().unsafe_load(
            i
        ):
            out.append(i)
    return out^


def compare_exact(
    got: HostBuffer[DType.float32],
    want: List[Float64],
    name: String,
) raises -> Int:
    var wrong = 0
    var first = -1
    for i in range(len(want)):
        if Float64(got.unsafe_ptr().unsafe_load(i)) != want[i]:
            wrong += 1
            if first < 0:
                first = i
    print(
        "    arm", name, ":", wrong, "wrong of", len(want),
        " first wrong cell", first,
    )
    return wrong


def check_hist2_one_byte[bits: Int](fold_count: Int) raises:
    var ctx = DeviceContext()
    var fingerprint_bin = (1 << bits) + 1
    print(
        "  == bits", bits, ": max_folds", fold_count,
        " fingerprint bin", fingerprint_bin, "==",
    )

    # Eight features, every one in the one-byte policy; `max_folds <=
    # (1 << bits)` routes the dispatch to hist_2 at exactly `bits`.
    if fold_count > (1 << bits) or fold_count <= (1 << bits) // 2:
        raise Error("fold_count does not select bits "
                    + String(bits) + "; fixture bug")
    var folds = List[Int]()
    for _ in range(N_FEATURES):
        folds.append(fold_count)
    for f in range(N_FEATURES):
        if policy_for_fold_count(folds[f]) != POLICY_ONE_BYTE:
            raise Error(
                "feature " + String(f) + " did not land in the one-byte"
                " policy; this check would be testing a different kernel"
            )

    # Three partitions sized so the level mixes block counts IN EVERY ARM.
    # The float mode's BlockLoadSize matches the PASS family's, asserted;
    # the Int32 mode's doubled block doubles its own, so leaf 0 is sized by
    # the LARGER of the two (multi-block everywhere) and leaf 2 one short of
    # the SMALLER (single-block everywhere).
    if MIN_DOCS_F32 != PASS_MIN_DOCS_PER_BLOCK:
        raise Error(
            "the float-mode hist_2 BlockLoadSize diverged from the PASS"
            " family's (" + String(MIN_DOCS_F32) + " vs "
            + String(PASS_MIN_DOCS_PER_BLOCK)
            + "); re-derive this check's partition sizes"
        )
    comptime MIN_MAX = (
        MIN_DOCS_I32 if MIN_DOCS_I32 > MIN_DOCS_F32 else MIN_DOCS_F32
    )
    var off = List[Int]()
    var siz = List[Int]()
    off.append(0)
    siz.append(4 * MIN_MAX)
    off.append(4 * MIN_MAX)
    siz.append(MIN_DOCS_F32 + 1)
    off.append(4 * MIN_MAX + MIN_DOCS_F32 + 1)
    siz.append(MIN_DOCS_F32 - 1)
    var n_live = len(off)
    var n_rows = off[n_live - 1] + siz[n_live - 1]

    var lay = build_layout(folds)
    var blocks = blocks_for(lay, n_rows)
    if len(blocks) != 1:
        raise Error(
            "expected exactly one policy block, got " + String(len(blocks))
        )
    var dblocks = upload_blocks(ctx, blocks)
    if dblocks[0].max_folds != fold_count:
        raise Error("max_folds is not the fold count; dispatch would move")

    # ---- REACH, computed before anything runs ---------------------------
    var groups = (N_FEATURES + 3) // 4
    var h2_rep = predicted_replicas(groups, n_live, STAT_COUNT // 2)
    var pass_rep = predicted_replicas(groups, n_live, STAT_COUNT)
    var mode_min_docs = List[Int]()
    mode_min_docs.append(MIN_DOCS_F32)
    mode_min_docs.append(MIN_DOCS_I32)
    for m in range(len(mode_min_docs)):
        var min_docs = mode_min_docs[m]
        var max_active = 1
        var min_active = 1 << 30
        for k in range(n_live):
            var a = (siz[k] + min_docs - 1) // min_docs
            if a > h2_rep:
                a = h2_rep
            print(
                "    mode", m, "leaf", k, "size", siz[k],
                "-> active blocks", a,
            )
            if a > max_active:
                max_active = a
            if a < min_active:
                min_active = a
        if max_active < 2:
            raise Error(
                "no partition replicates in mode " + String(m) + ", so its"
                " multi-block flush is never reached; raise the partition"
                " sizes"
            )
        if min_active > 1:
            raise Error(
                "every partition replicates in mode " + String(m) + ", so"
                " its single-block store is never reached; shrink a"
                " partition"
            )
    print(
        "  replicas: hist_2 grid", h2_rep, "(pairs axis 1), PASS grid",
        pass_rep, "(stats axis 2)",
    )

    # ---- the two compressed indexes: base and fingerprint ---------------
    var host_bin = List[List[Int]]()
    for f in range(N_FEATURES):
        var col = List[Int]()
        for r in range(n_rows):
            col.append(hash_bin(r, f, fold_count))
        host_bin.append(col^)

    var fp_bin = List[List[Int]]()
    for f in range(N_FEATURES):
        var col = List[Int]()
        for r in range(n_rows):
            var b = host_bin[f][r]
            # The plant: feature 0's first 64 rows carry the bin the two
            # families read differently. Rows 0..63 sit in leaf 0, the
            # four-block leaf, so the fingerprint ALSO proves the hist_2
            # atomic flush branch is reached.
            if f == 0 and r < FINGERPRINT_ROWS:
                b = fingerprint_bin
            col.append(b)
        fp_bin.append(col^)

    var cindex_base = build_cindex(ctx, lay, host_bin, n_rows)
    var cindex_fp = build_cindex(ctx, lay, fp_bin, n_rows)

    # ---- the stat planes: hashed INTEGERS -------------------------------
    var host_w = List[Int]()
    var host_g = List[Int]()
    for r in range(n_rows):
        host_w.append(hash_w(r))
        host_g.append(hash_g(r))

    var stats = ctx.enqueue_create_buffer[DType.float32](STAT_COUNT * n_rows)
    upload_stats(ctx, stats, host_w, host_g, n_rows)

    # ---- index, partitions, ids ------------------------------------------
    var perm = List[Int]()
    for pos in range(n_rows):
        perm.append((pos * 7919 + 13) % n_rows)

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())

    var p_off = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](MAX_LEAVES)
    var hz = ctx.enqueue_create_host_buffer[DType.uint32](MAX_LEAVES)
    for i in range(MAX_LEAVES):
        ho.unsafe_ptr().unsafe_store(i, UInt32(0))
        hz.unsafe_ptr().unsafe_store(i, UInt32(0))
    for i in range(n_live):
        ho.unsafe_ptr().unsafe_store(i, UInt32(off[i]))
        hz.unsafe_ptr().unsafe_store(i, UInt32(siz[i]))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=hz.unsafe_ptr())

    var ids = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var dense_ids = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var hid = ctx.enqueue_create_host_buffer[DType.uint32](MAX_LEAVES)
    for i in range(MAX_LEAVES):
        hid.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=ids, src_ptr=hid.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dense_ids, src_ptr=hid.unsafe_ptr())

    var cells = lay.hist_cells
    var total = MAX_LEAVES * STAT_COUNT * cells
    var hist = ctx.enqueue_create_buffer[DType.float32](total)
    var acc = ctx.enqueue_create_buffer[DType.int32](total)
    var block_hist = ctx.enqueue_create_buffer[DType.float32](total)
    var zf = ctx.enqueue_create_host_buffer[DType.float32](total)
    var zi = ctx.enqueue_create_host_buffer[DType.int32](total)
    for i in range(total):
        zf.unsafe_ptr().unsafe_store(i, Float32(0.0))
        zi.unsafe_ptr().unsafe_store(i, Int32(0))
    ctx.synchronize()

    # ---- expected tallies, all exact -------------------------------------
    var noperm = List[Int]()
    var want_base = host_tally(
        host_bin, host_w, host_g, off, siz, noperm, False, True, lay, bits, fold_count
    )
    var want_base_pass = host_tally(
        host_bin, host_w, host_g, off, siz, noperm, False, False, lay, bits, fold_count
    )
    var want_perm = host_tally(
        host_bin, host_w, host_g, off, siz, perm, True, True, lay, bits, fold_count
    )
    var want_perm_pass = host_tally(
        host_bin, host_w, host_g, off, siz, perm, True, False, lay, bits, fold_count
    )
    var want_fp_h2 = host_tally(
        fp_bin, host_w, host_g, off, siz, noperm, False, True, lay, bits, fold_count
    )
    var want_fp_pass = host_tally(
        fp_bin, host_w, host_g, off, siz, noperm, False, False, lay, bits, fold_count
    )

    # In-contract data must be family-invariant, or the cross-check below
    # asserts two different numbers agree.
    for i in range(total):
        if want_base[i] != want_base_pass[i]:
            raise Error("the two families' EXPECTED tallies differ on"
                        " in-contract data; the fixture is wrong")

    # The fingerprint's own prediction: where the two families' expectations
    # differ on the planted index. Must be nonempty and confined to feature
    # 0's fold 1 in leaf 0.
    var predicted_fp = List[Int]()
    for i in range(total):
        if want_fp_h2[i] != want_fp_pass[i]:
            predicted_fp.append(i)
    if len(predicted_fp) == 0:
        raise Error("the fingerprint plants nothing; the check cannot tell"
                    " the families apart")
    var f0_cell = Int(lay.features[0].first_fold_index) + (
        fingerprint_bin & ((1 << bits) - 1)
    )
    for i in range(len(predicted_fp)):
        var idx = predicted_fp[i]
        var leaf = idx // (STAT_COUNT * cells)
        var flat = idx % cells
        if leaf != 0 or flat != f0_cell:
            raise Error("the fingerprint prediction leaked outside leaf 0"
                        " fold " + String(f0_cell) + "; fixture bug")
    print(
        "  fingerprint: bin", fingerprint_bin, "on", FINGERPRINT_ROWS,
        "rows -> the families must differ at", len(predicted_fp),
        "cells, flat bin", f0_cell, "of leaf 0",
    )

    # ---- the arms ---------------------------------------------------------
    print("  direct arms (depth 0):")
    var a_pass = run_pass_arm[bits](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    var b_h2 = run_dispatch_arm(
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, dense_ids, hist, acc, block_hist, cells, zf, zi,
    )
    var wrong = compare_exact(a_pass, want_base_pass, String("PASS direct"))
    wrong += compare_exact(b_h2, want_base, String("dispatch direct"))
    var cross_dir = diff_cells(b_h2, a_pass, total)
    print("    families cross-agree on", total, "cells:",
          len(cross_dir) == 0)
    if wrong != 0 or len(cross_dir) != 0:
        raise Error(
            "the hist_2 and PASS one-byte families disagree (or miss the"
            " host tally) on the DIRECT arm; the two-stat port is wrong"
        )

    # gather arms read the PERMUTED index; direct arms above already ran on
    # the identity, so the overwrite cannot retroactively weaken them.
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(perm[r]))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()

    print("  gather arms (depth 1, permuted index):")
    var c_pass = run_pass_arm[bits](
        ctx, dblocks, 1, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    var d_h2 = run_dispatch_arm(
        ctx, dblocks, 1, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, dense_ids, hist, acc, block_hist, cells, zf, zi,
    )
    wrong = compare_exact(c_pass, want_perm_pass, String("PASS gather"))
    wrong += compare_exact(d_h2, want_perm, String("dispatch gather"))
    var cross_gat = diff_cells(d_h2, c_pass, total)
    print("    families cross-agree on", total, "cells:",
          len(cross_gat) == 0)
    if wrong != 0 or len(cross_gat) != 0:
        raise Error(
            "the hist_2 and PASS one-byte families disagree (or miss the"
            " host tally) on the GATHER arm; the two-stat port is wrong"
        )
    # The permutation must have been load-bearing, or depth 1 was depth 0.
    var perm_moved = diff_cells(d_h2, b_h2, total)
    if len(perm_moved) == 0:
        raise Error(
            "the permuted gather arm equals the direct arm, so the index"
            " indirection was never exercised"
        )
    print("    the permutation moved", len(perm_moved),
          "cells, so the gather indirection is load-bearing")

    # restore the identity index for the remaining direct arms
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()

    # ---- REACH 1: the family fingerprint ---------------------------------
    print("  family fingerprint (planted bin", fingerprint_bin, "):")
    var e_h2_fp = run_dispatch_arm(
        ctx, dblocks, 0, n_live, n_rows, cindex_fp, row_index, stats,
        p_off, p_sz, ids, dense_ids, hist, acc, block_hist, cells, zf, zi,
    )
    var f_pass_fp = run_pass_arm[bits](
        ctx, dblocks, 0, n_live, n_rows, cindex_fp, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    wrong = compare_exact(e_h2_fp, want_fp_h2, String("dispatch fingerprint"))
    wrong += compare_exact(f_pass_fp, want_fp_pass, String("PASS fingerprint"))
    if wrong != 0:
        raise Error("an arm missed its own family's expected tally on the"
                    " planted index")
    var fp_diff = diff_cells(e_h2_fp, f_pass_fp, total)
    if len(fp_diff) != len(predicted_fp):
        raise Error(
            "THE DISPATCH DID NOT LAUNCH THE hist_2 FAMILY: the planted bin"
            " moved " + String(len(fp_diff)) + " cells between the dispatch"
            " and the PASS family where their sources predict "
            + String(len(predicted_fp))
            + ". If it moved zero, the dispatch is still routing"
            " maxBins <= 128 to the PASS family"
        )
    for i in range(len(fp_diff)):
        if fp_diff[i] != predicted_fp[i]:
            raise Error("the fingerprint moved an unpredicted cell "
                        + String(fp_diff[i]))
    print(
        "    the dispatch's kernel reads bin", fingerprint_bin,
        "as fold 1 and the PASS family drops it, at exactly the",
        len(predicted_fp),
        "predicted cells: the dispatch runs the hist_2 family",
    )

    # ---- REACH 2: stat sabotage, one row per flush branch -----------------
    var poison_rows = List[Int]()
    poison_rows.append(off[0] + 37)  # leaf 0: four blocks, atomic flush
    poison_rows.append(off[2] + 100)  # leaf 2: one block, plain store
    var g_poison = List[Int]()
    for r in range(n_rows):
        g_poison.append(host_g[r])
    for i in range(len(poison_rows)):
        g_poison[poison_rows[i]] += POISON
    var want_poison = host_tally(
        host_bin, host_w, g_poison, off, siz, noperm, False, True, lay,
        bits, fold_count
    )
    var predicted_moves = List[Int]()
    for i in range(total):
        if want_poison[i] != want_base[i]:
            predicted_moves.append(i)
    print(
        "  sabotage: +", POISON, "on the gradient of rows",
        poison_rows[0], "(leaf 0, atomic) and", poison_rows[1],
        "(leaf 2, store) ->", len(predicted_moves), "cells must move",
    )
    if len(predicted_moves) == 0:
        raise Error("the sabotage predicts nothing; fixture bug")

    upload_stats_poisoned(ctx, stats, host_w, g_poison, n_rows)
    var g_h2_poison = run_dispatch_arm(
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, dense_ids, hist, acc, block_hist, cells, zf, zi,
    )
    wrong = compare_exact(g_h2_poison, want_poison, String("dispatch poisoned"))
    if wrong != 0:
        raise Error("the poisoned arm missed its expected tally")
    var moved = diff_cells(g_h2_poison, b_h2, total)
    if len(moved) != len(predicted_moves):
        raise Error(
            "the sabotage moved " + String(len(moved)) + " cells where "
            + String(len(predicted_moves)) + " were predicted; the hist_2"
            " path is not reading what this check thinks it reads"
        )
    for i in range(len(moved)):
        if moved[i] != predicted_moves[i]:
            raise Error("the sabotage moved an unpredicted cell "
                        + String(moved[i]))
    # per-branch reach: the predicted set must span BOTH leaves
    var saw_leaf0 = False
    var saw_leaf2 = False
    for i in range(len(predicted_moves)):
        var leaf = predicted_moves[i] // (STAT_COUNT * cells)
        if leaf == 0:
            saw_leaf0 = True
        if leaf == 2:
            saw_leaf2 = True
    if not (saw_leaf0 and saw_leaf2):
        raise Error("the sabotage did not span both flush branches")
    print(
        "    every predicted cell moved and nothing else did, in the"
        " four-block leaf AND the one-block leaf: both flush branches of"
        " the hist_2 writeback are reached"
    )

    # ---- REACH 3: BOTH ACCUMULATION MODES, exact, with a per-mode reach
    # fingerprint. Everything below runs in this one binary; the matrix row
    # only decides which mode the DISPATCH launches, and the last block
    # proves it launched that one.
    upload_stats(ctx, stats, host_w, host_g, n_rows)
    print(
        "  accumulation modes (hist_smem_mode_for row; the build's dispatch"
        " mode is", HIST2_SMEM_MODE, "):",
    )
    var m_f32 = run_mode_arm[bits, HIST_SMEM_WARP_PRIVATE_F32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    var m_i32 = run_mode_arm[bits, HIST_SMEM_SHARED2_I32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    wrong = compare_exact(m_f32, want_base, String("warp-private f32 direct"))
    wrong += compare_exact(
        m_i32, want_base, String("2-warp-shared i32 direct")
    )
    var cross_mode = diff_cells(m_i32, m_f32, total)
    var cross_disp = diff_cells(m_i32, b_h2, total)
    print(
        "    modes cross-agree on", total, "cells:",
        len(cross_mode) == 0, " and match the dispatch:",
        len(cross_disp) == 0,
    )
    if wrong != 0 or len(cross_mode) != 0 or len(cross_disp) != 0:
        raise Error(
            "the two hist_2 accumulation modes disagree (or miss the host"
            " tally) on the DIRECT arm; the shared-Int32 variant is wrong"
        )

    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(perm[r]))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()
    var mg_f32 = run_mode_arm[bits, HIST_SMEM_WARP_PRIVATE_F32](
        ctx, dblocks, 1, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    var mg_i32 = run_mode_arm[bits, HIST_SMEM_SHARED2_I32](
        ctx, dblocks, 1, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    wrong = compare_exact(mg_f32, want_perm, String("warp-private f32 gather"))
    wrong += compare_exact(
        mg_i32, want_perm, String("2-warp-shared i32 gather")
    )
    var cross_mode_g = diff_cells(mg_i32, mg_f32, total)
    print(
        "    gather modes cross-agree on", total, "cells:",
        len(cross_mode_g) == 0,
    )
    if wrong != 0 or len(cross_mode_g) != 0:
        raise Error(
            "the two hist_2 accumulation modes disagree (or miss the host"
            " tally) on the GATHER arm; the shared-Int32 variant is wrong"
        )
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()

    # THE SCALE FINGERPRINT: where does the add land? A poisoned
    # `fixed_scale` (2^30) wraps any arm that QUANTIZES -- the Int32 mode in
    # shared memory always, the float mode only in its writeback and only
    # under the integer flush -- and cannot touch an arm that never reads
    # the scale. Sabotage per branch, PORTING_RULES 7.
    var sf_i32 = run_mode_arm[bits, HIST_SMEM_SHARED2_I32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
        scale=POISON_SCALE,
    )
    var i32_moved = diff_cells(sf_i32, m_i32, total)
    if len(i32_moved) == 0:
        raise Error(
            "a wrecked fixed_scale did not move the shared-Int32 arm, so"
            " that arm is NOT quantizing in shared memory: the i32 mode's"
            " adds are not landing where this check thinks"
        )
    var sf_f32 = run_mode_arm[bits, HIST_SMEM_WARP_PRIVATE_F32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
        scale=POISON_SCALE,
    )
    var f32_moved = diff_cells(sf_f32, m_f32, total)

    @parameter
    if FLUSH_IS_FIXED:
        if len(f32_moved) == 0:
            raise Error(
                "under the integer flush the float-accumulation arm must"
                " quantize in its writeback, and a wrecked scale did not"
                " move it"
            )
    else:
        if len(f32_moved) != 0:
            raise Error(
                "the float-accumulation arm moved under a wrecked"
                " fixed_scale, but nothing on its float-flush path reads"
                " the scale; it is quantizing somewhere it must not"
            )
    var sf_disp = run_dispatch_arm(
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, dense_ids, hist, acc, block_hist, cells, zf, zi,
        scale=POISON_SCALE,
    )
    var disp_moved = diff_cells(sf_disp, b_h2, total)
    comptime DISPATCH_QUANTIZES = (
        HIST2_SMEM_MODE == HIST_SMEM_SHARED2_I32 or FLUSH_IS_FIXED
    )

    @parameter
    if DISPATCH_QUANTIZES:
        if len(disp_moved) == 0:
            raise Error(
                "THE DISPATCH DID NOT LAUNCH THE MODE THE MATRIX ROW"
                " SELECTS: hist_smem_mode_for says this build quantizes,"
                " and a wrecked fixed_scale did not move the dispatch arm"
            )
    else:
        if len(disp_moved) != 0:
            raise Error(
                "THE DISPATCH DID NOT LAUNCH THE MODE THE MATRIX ROW"
                " SELECTS: hist_smem_mode_for says this build takes the"
                " float path, which never reads the scale, and the"
                " dispatch arm moved under a wrecked one"
            )
    print(
        "    scale fingerprint: wrecked scale moved i32 at",
        len(i32_moved), "cells, f32 at", len(f32_moved),
        "(flush fixed:", FLUSH_IS_FIXED, "), dispatch at", len(disp_moved),
        "-> the dispatch runs the matrix row's mode",
    )

    # stat sabotage PER MODE: both flush branches inside each mode.
    upload_stats_poisoned(ctx, stats, host_w, g_poison, n_rows)
    var mp_f32 = run_mode_arm[bits, HIST_SMEM_WARP_PRIVATE_F32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    var mp_i32 = run_mode_arm[bits, HIST_SMEM_SHARED2_I32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    wrong = compare_exact(mp_f32, want_poison, String("f32 mode poisoned"))
    wrong += compare_exact(mp_i32, want_poison, String("i32 mode poisoned"))
    if wrong != 0:
        raise Error("a mode arm missed the poisoned tally")
    var moved_f32_mode = diff_cells(mp_f32, m_f32, total)
    var moved_i32_mode = diff_cells(mp_i32, m_i32, total)
    if len(moved_f32_mode) != len(predicted_moves) or len(
        moved_i32_mode
    ) != len(predicted_moves):
        raise Error(
            "the per-mode stat sabotage moved "
            + String(len(moved_f32_mode)) + " (f32) / "
            + String(len(moved_i32_mode)) + " (i32) cells where "
            + String(len(predicted_moves)) + " were predicted"
        )
    for i in range(len(predicted_moves)):
        if moved_f32_mode[i] != predicted_moves[i] or moved_i32_mode[
            i
        ] != predicted_moves[i]:
            raise Error("a mode arm's sabotage moved an unpredicted cell")
    print(
        "    per-mode stat sabotage: every predicted cell moved and nothing"
        " else did, in the multi-block leaf AND the one-block leaf, in BOTH"
        " modes"
    )

    print(
        "  hist_2 (maxBins <= 128, hist_2_one_byte_base.cuh) and PASS"
        " (hist_one_byte.cu) agree EXACTLY, cell for cell, direct and"
        " gather, in BOTH accumulation modes, and the dispatch demonstrably"
        " runs hist_2 in the matrix row's mode"
    )


def predicted_replicas(groups: Int, n_live: Int, z: Int) -> Int:
    """`CeilDivide(blocksPerSm * SMCount, x * y * z)` restated so the check
    predicts its own grid, deliberately NOT imported from the thing under
    test (`hist_2_one_byte_base.cuh:176-180`; blocksPerSm is 2 above
    Kepler)."""
    var base = groups * n_live * z
    if base < 1:
        base = 1
    var rep = (2 * SM_COUNT + base - 1) // base
    if rep < 1:
        rep = 1
    return rep


def build_cindex(
    ctx: DeviceContext,
    lay: CompressedIndexLayout,
    bins: List[List[Int]],
    n_rows: Int,
) raises -> DeviceBuffer[DType.uint32]:
    """One compressed index from one bin table, through the same binarize
    kernel the fit uses."""
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        n_rows * lay.columns
    )
    var z = ctx.enqueue_create_host_buffer[DType.uint32](
        n_rows * lay.columns
    )
    for i in range(n_rows * lay.columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var dev_bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(N_FEATURES):
        ref cf = lay.features[f]
        for r in range(n_rows):
            hb.unsafe_ptr().unsafe_store(r, UInt8(bins[f][r]))
        ctx.enqueue_copy(dst_buf=dev_bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            dev_bins.unsafe_ptr(), Int32(n_rows), cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()
    return cindex^


def upload_stats(
    ctx: DeviceContext,
    mut stats: DeviceBuffer[DType.float32],
    w: List[Int],
    g: List[Int],
    n_rows: Int,
) raises:
    var hs = ctx.enqueue_create_host_buffer[DType.float32](
        STAT_COUNT * n_rows
    )
    for r in range(n_rows):
        hs.unsafe_ptr().unsafe_store(r, Float32(w[r]))
        hs.unsafe_ptr().unsafe_store(n_rows + r, Float32(g[r]))
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())
    ctx.synchronize()


def upload_stats_poisoned(
    ctx: DeviceContext,
    mut stats: DeviceBuffer[DType.float32],
    w: List[Int],
    g: List[Int],
    n_rows: Int,
) raises:
    upload_stats(ctx, stats, w, g, n_rows)


def run_dispatch_arm(
    ctx: DeviceContext,
    mut dblocks: List[DeviceBlock],
    depth: Int,
    n_live: Int,
    n_rows: Int,
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
    scale: Float32 = FIXED_SCALE,
) raises -> HostBuffer[DType.float32]:
    """The REAL one-byte launch path, dispatch included: what a fit runs.
    `scale` exists for the scale fingerprint of REACH 3."""
    ctx.enqueue_copy(dst_buf=hist, src_ptr=zf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=acc, src_ptr=zi.unsafe_ptr())
    ctx.synchronize()

    launch_histograms_for_blocks(
        ctx, dblocks, depth, n_live, n_rows, STAT_COUNT, MAX_LEAVES,
        SM_COUNT, scale,
        cindex, row_index, stats, p_off, p_sz, ids, dense_ids,
        hist, acc, block_hist, hist_cells_per_leaf,
    )
    ctx.synchronize()

    var total = MAX_LEAVES * STAT_COUNT * hist_cells_per_leaf
    var out = ctx.enqueue_create_host_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()
    return out^


def run_mode_arm[
    bits: Int, smem_mode: Int
](
    ctx: DeviceContext,
    mut dblocks: List[DeviceBlock],
    depth: Int,
    n_live: Int,
    n_rows: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
    mut hist: DeviceBuffer[DType.float32],
    mut acc: DeviceBuffer[DType.int32],
    mut block_hist: DeviceBuffer[DType.float32],
    hist_cells_per_leaf: Int,
    zf: HostBuffer[DType.float32],
    zi: HostBuffer[DType.int32],
    scale: Float32 = FIXED_SCALE,
) raises -> HostBuffer[DType.float32]:
    """ONE accumulation mode of the hist_2 family, enqueued directly through
    `launch_hist2_one_byte[bits, smem_mode]`, bypassing the dispatch's
    default, plus the same bridge the helper runs: zero the scratch, kernel,
    `fixed_to_float`, `write_reduces`. The `fixed_to_float` pass is a no-op
    for an arm that never wrote the accumulator (it stores only nonzero
    cells), so one bridge serves both modes. `scale` exists for the scale
    fingerprint of REACH 3."""
    ctx.enqueue_copy(dst_buf=hist, src_ptr=zf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=acc, src_ptr=zi.unsafe_ptr())
    ctx.synchronize()

    ref blk = dblocks[0]
    var total_folds = blk.total_folds
    var block_cells = MAX_LEAVES * STAT_COUNT * total_folds
    ctx.enqueue_function[zero_buffer_kernel](
        block_hist.unsafe_ptr(),
        Int32(block_cells),
        grid_dim=(block_cells + 255) // 256,
        block_dim=256,
    )

    launch_hist2_one_byte[bits, smem_mode](
        ctx, dblocks[0], depth, n_live, n_rows, STAT_COUNT, MAX_LEAVES,
        SM_COUNT, n_rows, n_rows * blk.first_column,
        cindex, row_index, stats, p_off, p_sz, ids,
        block_hist, acc, scale,
    )

    ctx.enqueue_function[fixed_to_float_kernel](
        acc.unsafe_ptr(),
        block_hist.unsafe_ptr(),
        Int32(block_cells),
        scale,
        grid_dim=(block_cells + 255) // 256,
        block_dim=256,
    )
    ctx.enqueue_function[write_reduces_histograms_kernel](
        Int32(0),
        Int32(total_folds),
        ids.unsafe_ptr(),
        block_hist.unsafe_ptr(),
        Int32(hist_cells_per_leaf),
        hist.unsafe_ptr(),
        grid_dim=((total_folds + 127) // 128, n_live, STAT_COUNT),
        block_dim=(128, 1, 1),
    )
    ctx.synchronize()

    var total = MAX_LEAVES * STAT_COUNT * hist_cells_per_leaf
    var out = ctx.enqueue_create_host_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()
    return out^


def run_pass_arm[
    bits: Int, smem_mode: Int = HIST_SMEM_WARP_PRIVATE_F32
](
    ctx: DeviceContext,
    mut dblocks: List[DeviceBlock],
    depth: Int,
    n_live: Int,
    n_rows: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
    mut hist: DeviceBuffer[DType.float32],
    mut acc: DeviceBuffer[DType.int32],
    mut block_hist: DeviceBuffer[DType.float32],
    hist_cells_per_leaf: Int,
    zf: HostBuffer[DType.float32],
    zi: HostBuffer[DType.int32],
    scale: Float32 = FIXED_SCALE,
) raises -> HostBuffer[DType.float32]:
    """The PASS family (`TPointHistOneByte` at the same `bits`) enqueued
    DIRECTLY,
    bypassing the dispatch, plus the same bridge the helper runs: zero the
    scratch, kernel, `fixed_to_float`, `write_reduces`. This is what the
    dispatch used to run for these parameters, kept reachable here as the
    reference arm. `smem_mode` selects the accumulation arm (the
    `hist_smem_mode_for` row applies to this family too) and `scale` exists
    for the scale fingerprint."""
    ctx.enqueue_copy(dst_buf=hist, src_ptr=zf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=acc, src_ptr=zi.unsafe_ptr())
    ctx.synchronize()

    ref blk = dblocks[0]
    var total_folds = blk.total_folds
    var block_cells = MAX_LEAVES * STAT_COUNT * total_folds
    ctx.enqueue_function[zero_buffer_kernel](
        block_hist.unsafe_ptr(),
        Int32(block_cells),
        grid_dim=(block_cells + 255) // 256,
        block_dim=256,
    )

    launch_one_byte[bits, smem_mode](
        ctx, dblocks[0], depth, n_live, n_rows, STAT_COUNT, MAX_LEAVES,
        SM_COUNT, n_rows, n_rows * blk.first_column,
        cindex, row_index, stats, p_off, p_sz, ids,
        block_hist, acc, scale,
    )

    ctx.enqueue_function[fixed_to_float_kernel](
        acc.unsafe_ptr(),
        block_hist.unsafe_ptr(),
        Int32(block_cells),
        scale,
        grid_dim=(block_cells + 255) // 256,
        block_dim=256,
    )
    ctx.enqueue_function[write_reduces_histograms_kernel](
        Int32(0),
        Int32(total_folds),
        ids.unsafe_ptr(),
        block_hist.unsafe_ptr(),
        Int32(hist_cells_per_leaf),
        hist.unsafe_ptr(),
        grid_dim=((total_folds + 127) // 128, n_live, STAT_COUNT),
        block_dim=(128, 1, 1),
    )
    ctx.synchronize()

    var total = MAX_LEAVES * STAT_COUNT * hist_cells_per_leaf
    var out = ctx.enqueue_create_host_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()
    return out^


def check_pass_family_modes[bits: Int](fold_count: Int) raises:
    """The PASS family's OWN two accumulation modes at 129-255 bins, exact.

    CatBoost's ladder sends `maxBins <= 128` to hist_2, so everything above
    128 runs `TPointHistOneByte` PASS(8) -- the family the hist_2 sections
    of this check use only as a reference arm, at widths where no hist_2
    kernel exists to cross it against. What replaces the family cross-check
    here: BOTH PASS modes against the SAME exact host tally, against each
    other, and against the real dispatch; the scale fingerprint proving
    which arm quantizes (and that the dispatch launched the matrix row's
    mode); and the per-branch stat sabotage. The family fingerprint has no
    bits-8 form -- a u8 bin cannot hold `(1 << 8) + 1` -- and nothing is
    lost: it existed to prove the DISPATCH picked a family, and above 128
    folds there is only one family the dispatch can pick, which the scale
    fingerprint pins to the mode as well."""
    var ctx = DeviceContext()
    print("  == bits", bits, ": max_folds", fold_count, " (PASS family,"
          " both accumulation modes) ==")
    if fold_count > (1 << bits) or fold_count <= (1 << bits) // 2:
        raise Error("fold_count does not select bits "
                    + String(bits) + "; fixture bug")
    var folds = List[Int]()
    for _ in range(N_FEATURES):
        folds.append(fold_count)
    for f in range(N_FEATURES):
        if policy_for_fold_count(folds[f]) != POLICY_ONE_BYTE:
            raise Error("feature " + String(f) + " did not land in the"
                        " one-byte policy")

    # Partition sizes from the LARGER BlockLoadSize, so leaf 0 replicates
    # in both modes and leaf 2 replicates in neither.
    comptime MIN_MAX = (
        PASS_MIN_DOCS_I32
        if PASS_MIN_DOCS_I32 > PASS_MIN_DOCS_PER_BLOCK
        else PASS_MIN_DOCS_PER_BLOCK
    )
    var off = List[Int]()
    var siz = List[Int]()
    off.append(0)
    siz.append(4 * MIN_MAX)
    off.append(4 * MIN_MAX)
    siz.append(PASS_MIN_DOCS_PER_BLOCK + 1)
    off.append(4 * MIN_MAX + PASS_MIN_DOCS_PER_BLOCK + 1)
    siz.append(PASS_MIN_DOCS_PER_BLOCK - 1)
    var n_live = len(off)
    var n_rows = off[n_live - 1] + siz[n_live - 1]

    var lay = build_layout(folds)
    var blocks = blocks_for(lay, n_rows)
    if len(blocks) != 1:
        raise Error("expected exactly one policy block, got "
                    + String(len(blocks)))
    var dblocks = upload_blocks(ctx, blocks)
    if dblocks[0].max_folds != fold_count:
        raise Error("max_folds is not the fold count; dispatch would move")

    # ---- REACH, per mode, computed before anything runs -----------------
    var groups = (N_FEATURES + 3) // 4
    var pass_rep = predicted_replicas(groups, n_live, STAT_COUNT)
    var mode_min_docs = List[Int]()
    mode_min_docs.append(PASS_MIN_DOCS_PER_BLOCK)
    mode_min_docs.append(PASS_MIN_DOCS_I32)
    for m in range(len(mode_min_docs)):
        var min_docs = mode_min_docs[m]
        var max_active = 1
        var min_active = 1 << 30
        for k in range(n_live):
            var a = (siz[k] + min_docs - 1) // min_docs
            if a > pass_rep:
                a = pass_rep
            if a > max_active:
                max_active = a
            if a < min_active:
                min_active = a
        if max_active < 2:
            raise Error("no partition replicates in PASS mode "
                        + String(m) + "; raise the partition sizes")
        if min_active > 1:
            raise Error("every partition replicates in PASS mode "
                        + String(m) + "; shrink a partition")
    print("    replicas: PASS grid", pass_rep, ", rows", n_rows)

    # ---- data, index, partitions ----------------------------------------
    var host_bin = List[List[Int]]()
    for f in range(N_FEATURES):
        var col = List[Int]()
        for r in range(n_rows):
            col.append(hash_bin(r, f, fold_count))
        host_bin.append(col^)
    var cindex_base = build_cindex(ctx, lay, host_bin, n_rows)

    var host_w = List[Int]()
    var host_g = List[Int]()
    for r in range(n_rows):
        host_w.append(hash_w(r))
        host_g.append(hash_g(r))
    var stats = ctx.enqueue_create_buffer[DType.float32](STAT_COUNT * n_rows)
    upload_stats(ctx, stats, host_w, host_g, n_rows)

    var perm = List[Int]()
    for pos in range(n_rows):
        perm.append((pos * 7919 + 13) % n_rows)

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())

    var p_off = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](MAX_LEAVES)
    var hz = ctx.enqueue_create_host_buffer[DType.uint32](MAX_LEAVES)
    for i in range(MAX_LEAVES):
        ho.unsafe_ptr().unsafe_store(i, UInt32(0))
        hz.unsafe_ptr().unsafe_store(i, UInt32(0))
    for i in range(n_live):
        ho.unsafe_ptr().unsafe_store(i, UInt32(off[i]))
        hz.unsafe_ptr().unsafe_store(i, UInt32(siz[i]))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=hz.unsafe_ptr())

    var ids = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var dense_ids = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var hid = ctx.enqueue_create_host_buffer[DType.uint32](MAX_LEAVES)
    for i in range(MAX_LEAVES):
        hid.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=ids, src_ptr=hid.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dense_ids, src_ptr=hid.unsafe_ptr())

    var cells = lay.hist_cells
    var total = MAX_LEAVES * STAT_COUNT * cells
    var hist = ctx.enqueue_create_buffer[DType.float32](total)
    var acc = ctx.enqueue_create_buffer[DType.int32](total)
    var block_hist = ctx.enqueue_create_buffer[DType.float32](total)
    var zf = ctx.enqueue_create_host_buffer[DType.float32](total)
    var zi = ctx.enqueue_create_host_buffer[DType.int32](total)
    for i in range(total):
        zf.unsafe_ptr().unsafe_store(i, Float32(0.0))
        zi.unsafe_ptr().unsafe_store(i, Int32(0))
    ctx.synchronize()

    # ---- exact expectations ----------------------------------------------
    var noperm = List[Int]()
    var want_base = host_tally(
        host_bin, host_w, host_g, off, siz, noperm, False, False, lay,
        bits, fold_count
    )
    var want_perm = host_tally(
        host_bin, host_w, host_g, off, siz, perm, True, False, lay,
        bits, fold_count
    )

    # ---- direct arms, both modes, plus the real dispatch ------------------
    print("    direct arms (depth 0):")
    var p_f32 = run_pass_arm[bits, HIST_SMEM_WARP_PRIVATE_F32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    var p_i32 = run_pass_arm[bits, HIST_SMEM_SHARED2_I32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    var b_disp = run_dispatch_arm(
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, dense_ids, hist, acc, block_hist, cells, zf, zi,
    )
    var wrong = compare_exact(
        p_f32, want_base, String("PASS warp-private f32 direct")
    )
    wrong += compare_exact(
        p_i32, want_base, String("PASS 2-warp-shared i32 direct")
    )
    wrong += compare_exact(b_disp, want_base, String("dispatch direct"))
    var cross = diff_cells(p_i32, p_f32, total)
    print("      modes cross-agree on", total, "cells:", len(cross) == 0)
    if wrong != 0 or len(cross) != 0:
        raise Error("the PASS family's two accumulation modes disagree (or"
                    " miss the host tally) on the DIRECT arm")

    # ---- gather arms, both modes ------------------------------------------
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(perm[r]))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()
    print("    gather arms (depth 1, permuted index):")
    var g_f32 = run_pass_arm[bits, HIST_SMEM_WARP_PRIVATE_F32](
        ctx, dblocks, 1, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    var g_i32 = run_pass_arm[bits, HIST_SMEM_SHARED2_I32](
        ctx, dblocks, 1, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    wrong = compare_exact(
        g_f32, want_perm, String("PASS f32 gather")
    )
    wrong += compare_exact(
        g_i32, want_perm, String("PASS i32 gather")
    )
    var cross_g = diff_cells(g_i32, g_f32, total)
    print("      gather modes cross-agree on", total, "cells:",
          len(cross_g) == 0)
    if wrong != 0 or len(cross_g) != 0:
        raise Error("the PASS family's two accumulation modes disagree (or"
                    " miss the host tally) on the GATHER arm")
    var perm_moved = diff_cells(g_i32, p_i32, total)
    if len(perm_moved) == 0:
        raise Error("the permuted gather arm equals the direct arm, so the"
                    " index indirection was never exercised")
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()

    # ---- the scale fingerprint: which arm quantizes, and what the
    # dispatch launched ------------------------------------------------------
    var sf_i32 = run_pass_arm[bits, HIST_SMEM_SHARED2_I32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
        scale=POISON_SCALE,
    )
    var i32_moved = diff_cells(sf_i32, p_i32, total)
    if len(i32_moved) == 0:
        raise Error("a wrecked fixed_scale did not move the PASS"
                    " shared-Int32 arm, so it is NOT quantizing in shared"
                    " memory")
    var sf_f32 = run_pass_arm[bits, HIST_SMEM_WARP_PRIVATE_F32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
        scale=POISON_SCALE,
    )
    var f32_moved = diff_cells(sf_f32, p_f32, total)

    @parameter
    if FLUSH_IS_FIXED:
        if len(f32_moved) == 0:
            raise Error("under the integer flush the PASS float arm must"
                        " quantize in its writeback, and a wrecked scale"
                        " did not move it")
    else:
        if len(f32_moved) != 0:
            raise Error("the PASS float arm moved under a wrecked"
                        " fixed_scale, but nothing on its float-flush path"
                        " reads the scale")
    var sf_disp = run_dispatch_arm(
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, dense_ids, hist, acc, block_hist, cells, zf, zi,
        scale=POISON_SCALE,
    )
    var disp_moved = diff_cells(sf_disp, b_disp, total)
    comptime DISPATCH_QUANTIZES = (
        HIST2_SMEM_MODE == HIST_SMEM_SHARED2_I32 or FLUSH_IS_FIXED
    )

    @parameter
    if DISPATCH_QUANTIZES:
        if len(disp_moved) == 0:
            raise Error("THE DISPATCH DID NOT LAUNCH THE MODE THE MATRIX"
                        " ROW SELECTS at 129-255 bins: the row says this"
                        " build quantizes and the dispatch arm did not"
                        " move")
    else:
        if len(disp_moved) != 0:
            raise Error("THE DISPATCH DID NOT LAUNCH THE MODE THE MATRIX"
                        " ROW SELECTS at 129-255 bins: the float path never"
                        " reads the scale and the dispatch arm moved")
    print(
        "      scale fingerprint: wrecked scale moved i32 at",
        len(i32_moved), "cells, f32 at", len(f32_moved),
        "(flush fixed:", FLUSH_IS_FIXED, "), dispatch at", len(disp_moved),
        "-> the dispatch runs the matrix row's mode",
    )

    # ---- stat sabotage per mode, both flush branches ----------------------
    var poison_rows = List[Int]()
    poison_rows.append(off[0] + 37)
    poison_rows.append(off[2] + 100)
    var g_poison = List[Int]()
    for r in range(n_rows):
        g_poison.append(host_g[r])
    for i in range(len(poison_rows)):
        g_poison[poison_rows[i]] += POISON
    var want_poison = host_tally(
        host_bin, host_w, g_poison, off, siz, noperm, False, False, lay,
        bits, fold_count
    )
    var predicted_moves = List[Int]()
    for i in range(total):
        if want_poison[i] != want_base[i]:
            predicted_moves.append(i)
    if len(predicted_moves) == 0:
        raise Error("the sabotage predicts nothing; fixture bug")
    var saw_leaf0 = False
    var saw_leaf2 = False
    for i in range(len(predicted_moves)):
        var leaf = predicted_moves[i] // (STAT_COUNT * cells)
        if leaf == 0:
            saw_leaf0 = True
        if leaf == 2:
            saw_leaf2 = True
    if not (saw_leaf0 and saw_leaf2):
        raise Error("the sabotage did not span both flush branches")

    upload_stats_poisoned(ctx, stats, host_w, g_poison, n_rows)
    var mp_f32 = run_pass_arm[bits, HIST_SMEM_WARP_PRIVATE_F32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    var mp_i32 = run_pass_arm[bits, HIST_SMEM_SHARED2_I32](
        ctx, dblocks, 0, n_live, n_rows, cindex_base, row_index, stats,
        p_off, p_sz, ids, hist, acc, block_hist, cells, zf, zi,
    )
    wrong = compare_exact(mp_f32, want_poison, String("PASS f32 poisoned"))
    wrong += compare_exact(mp_i32, want_poison, String("PASS i32 poisoned"))
    if wrong != 0:
        raise Error("a PASS mode arm missed the poisoned tally")
    var moved_f = diff_cells(mp_f32, p_f32, total)
    var moved_i = diff_cells(mp_i32, p_i32, total)
    if len(moved_f) != len(predicted_moves) or len(moved_i) != len(
        predicted_moves
    ):
        raise Error("the per-mode sabotage moved " + String(len(moved_f))
                    + " (f32) / " + String(len(moved_i)) + " (i32) cells"
                    " where " + String(len(predicted_moves))
                    + " were predicted")
    for i in range(len(predicted_moves)):
        if moved_f[i] != predicted_moves[i] or moved_i[
            i
        ] != predicted_moves[i]:
            raise Error("a PASS mode arm's sabotage moved an unpredicted"
                        " cell")
    upload_stats(ctx, stats, host_w, host_g, n_rows)
    print(
        "      per-mode stat sabotage: every predicted cell moved and"
        " nothing else did, in the multi-block leaf AND the one-block"
        " leaf, in BOTH modes"
    )


def main() raises:
    print("hist_2 one-byte family check: bits 5 / 6 / 7 against the PASS"
          " family")
    # One max_folds per bit variant of their ladder (`hist_one_byte.cu:
    # 314-322`): 25 -> HIST2_PASS(5), 60 -> HIST2_PASS(6), 100 ->
    # HIST2_PASS(7). One-byte policy needs > 15 folds, so 25 is real data
    # shape, not a degenerate one.
    check_hist2_one_byte[5](25)
    check_hist2_one_byte[6](60)
    check_hist2_one_byte[7](100)
    # 129-255 bins never reach hist_2 -- their ladder's PASS(8) arm -- so
    # the coverage there is the PASS family's own two accumulation modes.
    check_pass_family_modes[8](200)
    print("check-hist2: PASS")
