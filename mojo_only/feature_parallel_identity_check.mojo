"""RUNG 2's GATE: the two POINTWISE searchers, one fixture, one tree, to the bit.

`PORTING.md` 91 A says the feature-parallel and doc-parallel data layouts
build a bit-identical compressed index at device count 1. 91 B says the two
searchers share their whole downstream stack. `NEXT_TWO.md` turns those into
a prediction: at `FoldBits == 0` and one device
`TFeatureParallelObliviousTreeSearcher` and `TDocParallelObliviousTreeSearcher`
are the same program, so one must reproduce the other TO THE BIT.

This file is that prediction, run.

WHAT IS ACTUALLY INDEPENDENT HERE, WHICH IS LESS THAN RUNG 1's GATE
-------------------------------------------------------------------
`check-pointwise-vs-greedy` compares two searchers that share only the
compressed index. This one compares two searchers that share almost
everything: the histogram family, the scorer, `TakeBest`, `ReorderBins`,
`UpdateSubsetsStats`, the stopping rule. **The only thing that differs is
the way a chosen split becomes the next level's bins**, and that is exactly
what this gate is for:

    doc-parallel   UpdateBinFromCompressedIndex   1 kernel, reads the
                                                  compressed index at the
                                                  gathered position
    feature-par.   WriteCompressedSplit           3 kernels, through
                   UpdateBinFromCompressedBits    `docBins` and a bit-packed
                   UpdateFoldBins                 intermediate

Three kernels, a 64-keys-per-word interleaved compression layout, a
document-ordered array and a gather, against one kernel. Saying so plainly:
this gate proves that chain, and it inherits everything else from rung 1's
gates rather than re-proving it.

THE FOUR GATES
--------------
    1  IDENTITY      same splits, same order, same feature/bin/split_type
    2  LEAF IDS      `docBins` -- the feature-parallel arm's per-document
                     leaf assignment -- against a HOST recomputation from
                     the fixture's own bins. PER DOCUMENT, not a total.
    3  COMPRESSION   the `readIndices` arm of `WriteCompressedSplit`, which
                     rung 2 never takes, run at the identity permutation and
                     required to agree with the `nullptr` arm bit for bit
                     (`PORTING_RULES.md` 8: reach is per-branch)
    4  CONTROL       a fixture the feature-parallel searcher MUST split
                     differently on. Without it gate 1 passes for a searcher
                     that returns a constant.

GATE 2 IS THE ONE THAT CANNOT BE FAKED BY AGREEMENT
---------------------------------------------------
Gates 1 and 4 are differential; if both ports shared a misreading they would
still agree. Gate 2 is not: it recomputes each document's leaf id on the
HOST from `host_bins` and the returned split list, and compares every
document's leaf id one at a time. A compression layout that packs the right multiset of
bits into the wrong words gives every leaf the right SIZE and the wrong
MEMBERS, and only a per-document comparison sees it
([[uniform-test-data-hides-permutation]]).

THE SABOTAGE TABLE, taken by EDITING the port and re-running
------------------------------------------------------------
There is no sabotage switch in the shipped files (`PORTING_RULES.md` 8: a
switch that outlives its measurement is a defect). Each row was produced by
making the edit, running this check at BOTH row counts, and reverting.
`R` is red, `.` is still green.

    #  edit                                              1  2  3  4
    -  ------------------------------------------------  -  -  -  -
    1  `feature_offset` left as the COLUMN index,        R  R  .  .
       dropping the `* n_rows` element conversion
    3  `TBinUpdater`'s OR made a STORE                    .  R  .  .
       (`bins[slot] = entry << depth`)
    4  `CompressBlock`'s bit position read as `id`        R  R  .  .
       instead of `KEYS_PER_STORAGE - id - 1`
    5  `add_split` reading the depth AFTER the push       R  R  .  .
    6  the block stride zeroed in BOTH new kernels        R* R* .  .
    7  `UpdateFoldBins`'s `loadBit` given as              R  .  .  .
       `CurrentDepth + 1`
    8  `compressed_split_size` sized "tight" as           CRASH
       `ceil(n / 64)` instead of `numBlocks * 128`
    9  the `readIndices` arm reading `indices[offset]`    .  .  R* .
       without the block base
   10  `subsets.gathered_target` flattened to 1.0         R  .  .  R
       immediately before `submit_compute`
    2  `split_subsets_mirror` given `subsets.indices`     REFUSED BY THE
       as `doc_map` instead of `observation_indices`      COMPILER

    R* = red at 16,434 rows and GREEN at 4,000. See the two-row-count
         section below; this is the whole reason it is there.

FOUR RESULTS IN THAT TABLE ARE WORTH READING TWICE.

**Sabotage 3 does not change the tree.** Turning `TBinUpdater`'s OR into a
STORE leaves bit `CurrentDepth` of `docBins` correct -- it only clears the
bits around it -- and `UpdateFoldBins` reads exactly that one bit. So
`subsets.Bins` is untouched, every split is identical, and gate 1 stays
green while the model's leaf assignment is wrong for 3 documents in 4. GATE
2 IS THE ONLY THING IN THIS FILE THAT SEES IT, and that is why it compares
per document rather than checking a total.

**Sabotage 7 does not move gate 2**, for the mirror-image reason: `docBins`
is written correctly and only `subsets.Bins` reads the wrong bit, so the
tree changes and the per-document recomputation -- which is taken against
the tree the searcher returned -- still agrees. The two gates cover
different halves of the same chain and neither is redundant.

**Sabotage 4 does NOT redden gate 3**, which an earlier draft of this
docstring asserted it would. Gate 3 compares the two `WriteCompressedSplit`
arms AGAINST EACH OTHER, and both go through the same `CompressBlock`; a
defect in the shared half cancels. Sabotage 9 is what makes gate 3 non
vacuous, and it reddens gate 3 ALONE.

**Sabotage 2 is refused by the Mojo compiler**, not by this check:

    error: aliasing values passed mutably to 'doc_map' argument and passed
    mutably to 'subsets' argument in 'split_subsets_mirror' call

The same rule that DEVIATION 97.2 records as a toolchain wall -- two views
of one buffer cannot both reach a call -- makes the most obvious way to get
`docMap` wrong unwritable. Recorded because a defect the toolchain cannot
express is coverage this file does not have to provide.

THE FIXTURE is `check-pointwise-vs-greedy`'s, deliberately, so a divergence
cannot be a fixture difference: ten features spanning binary, half-byte and
the 5/6/7-bit one-byte ranges, signal on the SIXTH one-byte feature so a
`group_offset` regression is visible (`PORTING.md` 107), weights 1 and
integer gradients so every float32 sum is exact.

IT IS RUN AT TWO ROW COUNTS AND THE SECOND ONE IS THE POINT
-----------------------------------------------------------
A compression block is `KeysPerBlock() == 64 * 128 == 8192` documents. At
rung 1's 4,000 rows there is exactly ONE block, `blockIdx.x` is always 0,
and every `+ KEYS_PER_COMPRESS_BLOCK * block` term in both new kernels is
identically zero -- the block striding, the partial-block guards and the
`indices`-versus-`compressedIndex` choice of which pointer advances are all
invisible. That is `PORTING.md` 107's rule verbatim: a fixture must exercise
the SECOND of anything the code groups.

    4,000  rows   1 block,  block 0 only
    16,434 rows   3 blocks, the last holding 50 keys -- so `tid < srcSize`
                  in `CompressBlock` (`compression_helper.cuh:105`) and
                  `dstOffset < dstSize` in `DecompressBlock` (`:126`) both
                  cut, and 78 of the last block's 128 threads write nothing

MEASURED (sabotage 6 above): with the row count at 4,000 only, zeroing the
block stride in both new kernels moves NOTHING -- all four gates stay green.
At 16,434 it reddens gates 1 and 2, with the first wrong document at row
8,193, which is the first document of block 1. Sabotage 9 behaves the same
way on gate 3. **Two of the ten sabotages in the table are invisible at
rung 1's row count**, so the second size is not thoroughness, it is the
only reason those two rows are not in the "moved nothing" paragraph.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.oblivious_tree_bin_builder import (
    compressed_split_size,
    create_compressed_split,
)
from gbdt.methods.oblivious_tree_doc_parallel_structure_searcher import (
    PointwiseTreeWorkspace,
    fit_oblivious_tree_structure,
)
from gbdt.methods.oblivious_tree_structure_searcher import (
    fit_feature_parallel_oblivious_tree_structure,
)
from gbdt.models.oblivious_model import BIN_SPLIT_TAKE_GREATER
from gbdt.options.catboost_options import SCORE_FUNCTION_COSINE

comptime MAX_DEPTH = 4


def main() raises:
    var ctx = DeviceContext()
    var failures = 0
    # ONE compression block, then THREE with a 50-key tail. See the module
    # docstring: at 4,000 rows `blockIdx.x` is always 0.
    failures += run_case(ctx, 4000)
    failures += run_case(ctx, 16434)
    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("rung 2: identity holds at BOTH row counts")


def run_case(ctx: DeviceContext, n_rows: Int) raises -> Int:
    var N_ROWS = n_rows
    print("")
    print("=========== ", N_ROWS, "rows,",
          (N_ROWS + 8191) // 8192, "compression block(s), last holding",
          N_ROWS - 8192 * ((N_ROWS + 8191) // 8192 - 1), "keys ===========")

    # ---- the fixture, `pointwise_vs_greedy_check.mojo`'s exactly ---------
    var folds: List[Int] = [1, 1, 12, 9, 20, 32, 48, 100, 64, 127]
    var n_features = len(folds)
    var lay = build_layout(folds)

    var cindex = ctx.enqueue_create_buffer[DType.uint32](N_ROWS * lay.columns)
    ctx.enqueue_memset(cindex, UInt32(0))
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](N_ROWS)
    var bins8 = ctx.enqueue_create_buffer[DType.uint8](N_ROWS)
    var host_bins = List[List[Int]]()
    for f in range(n_features):
        ref cf = lay.features[f]
        var col = List[Int]()
        for r in range(N_ROWS):
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            var b = Int(x % UInt32(folds[f] + 1))
            col.append(b)
            hb.unsafe_ptr().unsafe_store(r, UInt8(b))
        host_bins.append(col^)
        ctx.enqueue_copy(dst_buf=bins8, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * N_ROWS),
            cf.mask,
            cf.shift,
            bins8.unsafe_ptr(),
            Int32(N_ROWS),
            cindex.unsafe_ptr(),
            grid_dim=(N_ROWS + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    # signal at three depths; feature 9 is the SIXTH one-byte feature and
    # lives in the second cindex column of its policy (`PORTING.md` 107)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](2 * N_ROWS)
    for r in range(N_ROWS):
        var g = Float64(0.0)
        if host_bins[9][r] > 60:
            g += 8.0
        if host_bins[2][r] > 6:
            g += 3.0
        if host_bins[8][r] > 30:
            g += 1.0
        g -= 6.0
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(N_ROWS + r, Float32(g))

    var failures = 0

    # ---- arm A: the DOC-PARALLEL searcher (rung 1) -----------------------
    var a_w = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var a_g = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    ctx.enqueue_copy(dst_buf=a_w, src_ptr=hs.unsafe_ptr())
    ctx.enqueue_copy(
        dst_buf=a_g, src_ptr=hs.unsafe_ptr().unsafe_offset(N_ROWS)
    )
    ctx.synchronize()
    print("arm A: TDocParallelObliviousTreeSearcher (UpdateBinFromCompressedIndex)")
    var dp_pool = List[PointwiseTreeWorkspace]()
    var dp_splits = fit_oblivious_tree_structure(
        ctx, lay, N_ROWS, MAX_DEPTH, cindex, a_w^, a_g^, 10, Float32(1.0),
        SCORE_FUNCTION_COSINE, dp_pool,
    )

    # ---- arm B: the FEATURE-PARALLEL searcher (rung 2) -------------------
    var b_w = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var b_g = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    ctx.enqueue_copy(dst_buf=b_w, src_ptr=hs.unsafe_ptr())
    ctx.enqueue_copy(
        dst_buf=b_g, src_ptr=hs.unsafe_ptr().unsafe_offset(N_ROWS)
    )
    ctx.synchronize()
    print(
        "arm B: TFeatureParallelObliviousTreeSearcher"
        " (WriteCompressedSplit -> UpdateBinFromCompressedBits ->"
        " UpdateFoldBins)"
    )
    var fp = fit_feature_parallel_oblivious_tree_structure(
        ctx, lay, N_ROWS, MAX_DEPTH, cindex, b_w^, b_g^, 10, Float32(1.0),
        SCORE_FUNCTION_COSINE,
    )
    var fp_splits = fp[0].copy()
    var fp_doc_bins = fp[1]

    # ---- GATE 1: the identity ------------------------------------------
    print("doc-parallel     :", len(dp_splits), "splits")
    for i in range(len(dp_splits)):
        print(
            "   depth", i, "feature", dp_splits[i].feature_id,
            "bin", dp_splits[i].bin_idx,
            "type", dp_splits[i].split_type,
        )
    print("feature-parallel :", len(fp_splits), "splits")
    for i in range(len(fp_splits)):
        print(
            "   depth", i, "feature", fp_splits[i].feature_id,
            "bin", fp_splits[i].bin_idx,
            "type", fp_splits[i].split_type,
        )

    if len(dp_splits) != len(fp_splits):
        print(
            "FAIL gate 1: the two searchers grew trees of DIFFERENT DEPTH --",
            len(dp_splits), "against", len(fp_splits),
        )
        failures += 1
    else:
        var differ = 0
        for i in range(len(fp_splits)):
            if (
                dp_splits[i].feature_id != fp_splits[i].feature_id
                or dp_splits[i].bin_idx != fp_splits[i].bin_idx
                or dp_splits[i].split_type != fp_splits[i].split_type
            ):
                print(
                    "   MISMATCH at depth", i, ": doc-parallel feature",
                    dp_splits[i].feature_id, "bin", dp_splits[i].bin_idx,
                    "; feature-parallel feature", fp_splits[i].feature_id,
                    "bin", fp_splits[i].bin_idx,
                )
                differ += 1
        if differ != 0:
            print("FAIL gate 1:", differ, "of", len(fp_splits), "splits differ")
            failures += 1
        else:
            print("gate 1 OK: identity over", len(fp_splits), "splits")

    if len(fp_splits) == 0:
        print("FAIL: the feature-parallel searcher grew NO splits at all")
        failures += 1

    # ---- GATE 2: docBins, per document, against the host ----------------
    # their `CacheBinsForModel(..., std::move(docBins))` (`:300-304`).
    # `docBins[d]` bit `k` is document `d`'s side of split `k`, so the whole
    # word is the leaf id. Recompute it from `host_bins` and the split list
    # and compare all 4,000, not their histogram.
    var h_doc_bins = ctx.enqueue_create_host_buffer[DType.uint32](N_ROWS)
    ctx.enqueue_copy(dst_buf=h_doc_bins, src_buf=fp_doc_bins)
    ctx.synchronize()

    var wrong = 0
    var first_wrong = -1
    var occupancy = List[Int]()
    for _ in range(1 << MAX_DEPTH):
        occupancy.append(0)
    for r in range(N_ROWS):
        var expect = UInt32(0)
        for k in range(len(fp_splits)):
            ref s = fp_splits[k]
            var b = host_bins[Int(s.feature_id)][r]
            var bit = UInt32(0)
            if s.split_type == Int32(BIN_SPLIT_TAKE_GREATER):
                if b > Int(s.bin_idx):
                    bit = UInt32(1)
            else:
                if b == Int(s.bin_idx):
                    bit = UInt32(1)
            expect |= bit << UInt32(k)
        var got = h_doc_bins[r]
        if Int(expect) < len(occupancy):
            occupancy[Int(expect)] += 1
        if got != expect:
            wrong += 1
            if first_wrong < 0:
                first_wrong = r
    if wrong != 0:
        print(
            "FAIL gate 2:", wrong, "of", N_ROWS,
            "documents have the wrong leaf id in docBins; first at row",
            first_wrong, "-- got", h_doc_bins[first_wrong],
        )
        failures += 1
    else:
        print(
            "gate 2 OK:", N_ROWS,
            "documents' leaf ids match the host recomputation, per document",
        )
    # a fixture where every document lands in one leaf would pass gate 2
    # for a searcher that never splits anything; say what the spread is.
    var occupied = 0
    for i in range(len(occupancy)):
        if occupancy[i] > 0:
            occupied += 1
    print("        leaves occupied:", occupied, "of", len(occupancy))
    if occupied < 2:
        print("FAIL gate 2: docBins puts every document in ONE leaf")
        failures += 1

    # ---- GATE 3: the readIndices arm of WriteCompressedSplit ------------
    # `PORTING_RULES.md` 8 -- reach is PER BRANCH. `TSplitHelper::
    # GetCompressedBits` passes `nullptr` for an ordinary feature and
    # `&DataSet.GetInverseIndices()` for a permutation-dependent one; rung 2
    # has no CTR columns, so the second arm never runs here. Run it at the
    # identity permutation, where it must produce the same bits.
    if len(fp_splits) > 0:
        ref s0 = fp_splits[0]
        ref cf0 = lay.features[Int(s0.feature_id)]
        var packed = compressed_split_size(N_ROWS)
        var bits_null = ctx.enqueue_create_buffer[DType.uint64](packed)
        var bits_idx = ctx.enqueue_create_buffer[DType.uint64](packed)
        ctx.enqueue_memset(bits_null, UInt64(0))
        ctx.enqueue_memset(bits_idx, UInt64(0))
        var ident = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
        var h_ident = ctx.enqueue_create_host_buffer[DType.uint32](N_ROWS)
        for r in range(N_ROWS):
            h_ident.unsafe_ptr().unsafe_store(r, UInt32(r))
        ctx.enqueue_copy(dst_buf=ident, src_ptr=h_ident.unsafe_ptr())
        ctx.synchronize()

        var off0 = UInt32(Int(cf0.offset) * N_ROWS)
        create_compressed_split(
            ctx, cindex, ident, False, N_ROWS, off0, cf0.mask, cf0.shift,
            False, UInt32(s0.bin_idx), bits_null,
        )
        create_compressed_split(
            ctx, cindex, ident, True, N_ROWS, off0, cf0.mask, cf0.shift,
            False, UInt32(s0.bin_idx), bits_idx,
        )
        var hn = ctx.enqueue_create_host_buffer[DType.uint64](packed)
        var hi = ctx.enqueue_create_host_buffer[DType.uint64](packed)
        ctx.enqueue_copy(dst_buf=hn, src_buf=bits_null)
        ctx.enqueue_copy(dst_buf=hi, src_buf=bits_idx)
        ctx.synchronize()

        var bits_differ = 0
        var set_bits = 0
        for w in range(packed):
            if hn[w] != hi[w]:
                bits_differ += 1
            var v = hn[w]
            while v != UInt64(0):
                set_bits += 1
                v &= v - UInt64(1)
        if bits_differ != 0:
            print(
                "FAIL gate 3:", bits_differ, "of", packed,
                "packed words differ between the nullptr and readIndices"
                " arms at the identity permutation",
            )
            failures += 1
        else:
            print(
                "gate 3 OK: both WriteCompressedSplit arms agree over",
                packed, "packed words,", set_bits, "documents on the right",
            )
        # an all-zero pack agrees with an all-zero pack; the split has to
        # actually separate somebody.
        if set_bits == 0 or set_bits == N_ROWS:
            print(
                "FAIL gate 3: the packed split is CONSTANT (", set_bits,
                "of", N_ROWS, "), so the agreement is vacuous",
            )
            failures += 1

    # ---- GATE 4: the control, which MUST differ -------------------------
    # Gate 1 passes trivially for a searcher that ignores its input. Feed
    # arm B a target whose signal is on a DIFFERENT feature and require the
    # tree to change.
    var hc = ctx.enqueue_create_host_buffer[DType.float32](2 * N_ROWS)
    for r in range(N_ROWS):
        var g = Float64(0.0)
        # feature 3 (half-byte, 9 folds) carries everything now, and none of
        # the three original signal features appears at all
        if host_bins[3][r] > 4:
            g += 8.0
        if host_bins[4][r] > 10:
            g += 3.0
        g -= 4.0
        hc.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hc.unsafe_ptr().unsafe_store(N_ROWS + r, Float32(g))
    var c_w = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var c_g = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    ctx.enqueue_copy(dst_buf=c_w, src_ptr=hc.unsafe_ptr())
    ctx.enqueue_copy(
        dst_buf=c_g, src_ptr=hc.unsafe_ptr().unsafe_offset(N_ROWS)
    )
    ctx.synchronize()
    var ctl = fit_feature_parallel_oblivious_tree_structure(
        ctx, lay, N_ROWS, MAX_DEPTH, cindex, c_w^, c_g^, 10, Float32(1.0),
        SCORE_FUNCTION_COSINE,
    )
    var ctl_splits = ctl[0].copy()
    print("control (signal moved to features 3 and 4):", len(ctl_splits), "splits")
    for i in range(len(ctl_splits)):
        print(
            "   depth", i, "feature", ctl_splits[i].feature_id,
            "bin", ctl_splits[i].bin_idx,
        )
    var control_same = len(ctl_splits) == len(fp_splits)
    if control_same:
        for i in range(len(ctl_splits)):
            if (
                ctl_splits[i].feature_id != fp_splits[i].feature_id
                or ctl_splits[i].bin_idx != fp_splits[i].bin_idx
            ):
                control_same = False
    if control_same:
        print(
            "FAIL gate 4: the CONTROL grew the SAME tree as the identity"
            " arm. Gate 1 is not measuring anything -- the searcher is not"
            " reading its target."
        )
        failures += 1
    else:
        print("gate 4 OK: the control tree differs, so gate 1 can fail")

    if failures == 0:
        print(
            "  OK at", N_ROWS, "rows:"
            " TFeatureParallelObliviousTreeSearcher reproduces"
            " TDocParallelObliviousTreeSearcher to the bit at FoldBits == 0,",
            len(fp_splits), "splits and", N_ROWS, "leaf ids",
        )
    return failures
