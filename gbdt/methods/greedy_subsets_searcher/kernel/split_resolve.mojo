# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in the root DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Device-side split resolution: the block-winner reduce and the split-
descriptor pack, on the device, so a level blocks the host ONCE.

================= DEVIATION BLOCK =================
NO CATBOOST COUNTERPART AS A KERNEL. Their level blocks the host TWICE:
`bestProps.Read(propsCpu)` (`greedy_search_helper.cpp:517`) hands the
argmax to the host, which reduces the block winners (`:520-529`), packs
the split descriptors (`split_properties_helper.cpp:872-905`) and uploads
them; then `RebuildLeavesSizes` blocks again (`:800-813`). On CUDA a
drain is cheap and their design tolerates two per level. On THIS box a
drain (submit + wait round trip) measures ~191 us plus the queue-empty
bubble behind it, and the fixed per-tree floor those drains sit in is
9.3-11.5 ms of a 19.7 ms covtype tree (scratchpad `fixedfloor_probe`,
2026-08-20) -- more than CatBoost's entire covtype CPU tree.

So the winner reduce and the pack move onto the device, the split chain
is enqueued behind them WITHOUT a host read, and the winner rides home
in the SAME drain that already reads the leaf sizes. Two blocking reads
per level become one. This is SCHEDULING, not arithmetic: the reduce
below reproduces the host loop's exact sequential order and tie rule
(higher score wins, tie to the smaller bin-feature; poison records of
score -FLOAT32_MAX / bin 0xFFFFFFFF never beat a real one), so the
winner is bit-for-bit the winner the host loop picked, and the oracles
-- which compare every split to CatBoost's -- gate that claim per tree.

THE GATES STAY ON THE HOST. Their score gate (`Score < 0`, ours
`best_score > 0`) and repeat-split rule decide whether a level's split
is APPLIED, and with the pack device-side the split chain has already
run by the time the host sees the score. The host therefore applies the
gates AFTER the combined drain and, on the rare stop, ROLLS BACK the
one speculative level: the left child kept the parent's offset
(`UpdatePartitionsAfterSplitImpl` reads `parts[leftLeaf]`), so the
parent partition is exactly (off[i], sz[i] + sz[n_live + i]) and one
small `p_sz` upload restores it; the reorder permuted rows only WITHIN
parent ranges, so every leaf sum the tail computes is unchanged; the
polluted right-child histogram slots are read by nothing after the
break. `run_tree` (the uniform checked path) keeps their two-drain
shape untouched.
===================================================

The bin-feature table the pack reads is built ONCE per tree by
`fill_bin_feature_table` from the same layout the host's `resolve_split`
walks: seven parallel arrays indexed by FLAT BIN-FEATURE, carrying the
owning feature's `TCFeature` fields (offset pre-scaled by `n_rows`, as
the split kernels expect), and the bin within the feature.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast

from gbdt.gpu_data.gpu_structures import CFeature
from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    FLOAT32_MAX,
)

comptime RESOLVE_BLOCK_SIZE = 64

comptime WINNER_SENTINEL = UInt32(0xFFFFFFFF)

# ---- DEVIATION 1904: the leaf-winner fold's record layout ---------------
# One record per scored leaf, `WINNER_RECORD_WORDS` UInt32 words:
#   [0] feature id   (Int32 bits; the sentinel -1 when UNDEFINED)
#   [1] bin id       (Int32 bits, `to_split`-clamped) -- or, under
#                    BIN_OUT_OF_RANGE, the offending raw bin-feature
#   [2] gain         (Float32 bits, THEIR sign: the kernel's gain negated,
#                    exactly what the host fold stored)
#   [3] status       (one of the three constants below)
#   [4] winning CELL (the raw flat bin-feature of the winner, un-clamped;
#                    0xFFFFFFFF when UNDEFINED). DEVIATION 1901's
#                    `best_cell`: the partition-stats propagation kernel
#                    addresses the scanned histogram by CELL and
#                    `to_split` discards it, so the fold carries it out
#                    beside the resolved record exactly as the host fold
#                    kept it beside `best`.
# Small enough that the whole per-iteration readback is
# `WINNER_RECORD_WORDS * leaves_scored * 4` bytes -- 40 bytes at the
# lossguide cap of two leaves, the same order as LightGBM's 8-int
# per-split readback (`cuda_best_split_finder.cu:2161-2191`).
comptime WINNER_RECORD_WORDS = 5

comptime WINNER_STATUS_UNDEFINED = UInt32(0)
comptime WINNER_STATUS_DEFINED = UInt32(1)
comptime WINNER_STATUS_BIN_OUT_OF_RANGE = UInt32(2)

comptime WINNER_FOLD_BLOCK_SIZE = 32


def resolve_and_pack_kernel(
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
    argmax_blocks_in: Int32,
    bf_offset: MutPointer[UInt32, MutAnyOrigin],
    bf_mask: MutPointer[UInt32, MutAnyOrigin],
    bf_shift: MutPointer[UInt32, MutAnyOrigin],
    bf_first: MutPointer[UInt32, MutAnyOrigin],
    bf_folds: MutPointer[UInt32, MutAnyOrigin],
    bf_one_hot: MutPointer[UInt8, MutAnyOrigin],
    bf_bin: MutPointer[UInt32, MutAnyOrigin],
    depth_in: Int32,
    n_live_in: Int32,
    winners_score: MutPointer[Float32, MutAnyOrigin],
    winners_bf: MutPointer[UInt32, MutAnyOrigin],
    sp_feats: MutPointer[UInt8, MutAnyOrigin],
    sp_bins: MutPointer[UInt32, MutAnyOrigin],
    ids_c: MutPointer[UInt32, MutAnyOrigin],
):
    """One block. Every thread redundantly scans the <= 64 block-winner
    records (cheaper than a sync), thread 0 records the level's winner,
    and threads 0..n_live-1 pack the per-leaf split descriptors their
    `MakeSplit` builds on the host.

    The scan is the host loop VERBATIM: sequential over blocks, strict
    `>`, tie to the smaller bin index. Sequential order on every thread
    means every thread lands on the same winner with no reduction-order
    freedom to differ across vendors.
    """
    var tid = Int(thread_idx.x)
    var argmax_blocks = Int(argmax_blocks_in)
    var n_live = Int(n_live_in)

    var best_score = -FLOAT32_MAX
    var best_bin = WINNER_SENTINEL
    for bi in range(argmax_blocks):
        var b_score = out_score.unsafe_load(bi)
        var b_bin = out_bin.unsafe_load(bi)
        var take = b_score > best_score
        if b_score == best_score and b_bin < best_bin:
            take = True
        if take:
            best_score = b_score
            best_bin = b_bin

    if tid == 0:
        winners_score.unsafe_store(Int(depth_in), best_score)
        winners_bf.unsafe_store(Int(depth_in), best_bin)

    # The sentinel case (every score infinite) packs bin-feature 0 so the
    # speculative chain reads a defined descriptor; the host sees the
    # sentinel in `winners_bf` after the drain and raises exactly the
    # diagnostic it raised before, so nothing downstream of a sentinel is
    # ever read.
    var bf = 0
    if best_bin != WINNER_SENTINEL:
        bf = Int(best_bin)

    var feats = sp_feats.bitcast[CFeature]()
    var i = tid
    while i < n_live:
        feats[unsafe_offset=i] = CFeature(
            offset=bf_offset.unsafe_load(bf),
            mask=bf_mask.unsafe_load(bf),
            shift=bf_shift.unsafe_load(bf),
            first_fold_index=bf_first.unsafe_load(bf),
            folds=bf_folds.unsafe_load(bf),
            one_hot_feature=bf_one_hot.unsafe_load(bf) != UInt8(0),
        )
        sp_bins.unsafe_store(i, bf_bin.unsafe_load(bf))
        # their `rightIds`: `leavesCount + i`. The left ids are the dense
        # sequence the caller already holds.
        ids_c.unsafe_store(i, UInt32(n_live + i))
        i += RESOLVE_BLOCK_SIZE


def plan_level_kernel(
    part_size: MutPointer[UInt32, MutAnyOrigin],
    half_in: Int32,
    ids_compute: MutPointer[UInt32, MutAnyOrigin],
    sub_from: MutPointer[UInt32, MutAnyOrigin],
    sub_what: MutPointer[UInt32, MutAnyOrigin],
):
    """`BuildNecessaryHistograms`' smaller-child choice, on the device.

    Their `:1318` rule over each sibling pair (left = slot i, right =
    slot half + i, siblings by construction of the split chain): the
    SMALLER child is computed and the larger derived by subtraction.

        if (firstLeaf.Size < secondLeaf.Size) { smallLeafId = ids[0]; ... }
        else                                  { smallLeafId = ids[1]; ... }

    `ids` is filled in ASCENDING leaf index (`:1300`) and `MakeSplit`
    gives the LEFT child the existing (lower) id and the RIGHT child the
    appended one (`:861-862`, `:976-977`), so `ids[0]` is the LEFT child
    and the strict `<` is on the LEFT. **ON AN EXACT TIE THE `else`
    BRANCH FIRES AND THE RIGHT CHILD IS COMPUTED**, which is why the
    condition below is `left_sz < right_sz` and `small` starts on the
    right. PORTING.md 136: it was the other way round from `409a16c`
    until 2026-08-21.

    The host used to make this choice from a per-level size read;
    DEVIATION 94 enqueues levels blind, so the choice moves here, reading
    the same `p_sz` the partition-update kernel just wrote.

    AND THE CHOICE DOES CHANGE HISTOGRAM BITS, MEASURED. The claim that
    used to sit here -- that the Int32 fixed-point accumulator makes
    sibling subtraction exact, so which sibling is computed cannot move a
    bit -- is FALSE, and it is false for a reason the accumulator cannot
    fix: `write_reduces_from_fixed_kernel` converts each cell with
    `Float32(Int(q)) / fixed_scale` BEFORE the subtraction, and
    `substract_histograms_kernel` then works in float32. `choose_scale`
    targets 2^30, so `q` sits far above float32's exact-integer range of
    2^24 and `Float32(Int(q))` ROUNDS; parent and child are each rounded
    separately and `parent_f - left_f` need not equal `right_f`.
    Measured on this port's own kernels at a 4096-row fixture with
    `fixed_scale` 65536 and cells up to 5.3e8: 19 of 2048 derived cells
    differ from the built sibling one way, 24 of 2048 the other, worst
    gap 3 ulp (`original/sibling_tiebreak_check.mojo`). What survives of
    the old paragraph is only the second half: making THEIR choice keeps
    the computed set the smaller half, which is the point of the halving.
    """
    var half = Int(half_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < half:
        var left_sz = part_size.unsafe_load(i)
        var right_sz = part_size.unsafe_load(half + i)
        # their `else` branch is the DEFAULT, so the tie lands on the right
        var small = half + i
        var big = i
        if left_sz < right_sz:
            small = i
            big = half + i
        ids_compute.unsafe_store(i, UInt32(small))
        sub_from.unsafe_store(i, UInt32(big))
        sub_what.unsafe_store(i, UInt32(small))


def leaf_winner_fold_kernel(
    region_score: MutPointer[Float32, MutAnyOrigin],
    region_bin: MutPointer[UInt32, MutAnyOrigin],
    argmax_blocks_in: Int32,
    hist_cells_in: Int32,
    bf_feature: MutPointer[Int32, MutAnyOrigin],
    bf_bin: MutPointer[Int32, MutAnyOrigin],
    bf_one_hot: MutPointer[UInt8, MutAnyOrigin],
    bf_folds: MutPointer[Int32, MutAnyOrigin],
    winner_records: MutPointer[UInt32, MutAnyOrigin],
):
    """The depthwise/lossguide host reduce, on the device: one block per
    scored leaf folds that leaf's `argmax_blocks` block-winner records to
    ONE winner record, so the driver reads back `WINNER_RECORD_WORDS`
    words per leaf instead of `2 * argmax_blocks` and never folds on the
    host.

    ================= DEVIATION 1904 =================
    DEVICE-RESIDENT BEST-SPLIT SELECTION. The non-symmetric drivers block
    the host at `score.read` -- two `argmax_blocks * leaves`-sized copies
    plus a `wait_complete` -- and then fold sequentially on the host under
    `best_split_properties_less`
    (`greedy_search_helper_depthwise.mojo:1357-1470`,
    `find_best_leaf_to_split` downstream of it). That shape is CatBoost's
    own (`bestProps.Read(propsCpu)`, `greedy_search_helper.cpp:517-529`);
    LightGBM's CUDA learner keeps the same fold ON DEVICE and reads back
    only the winner (`SyncBestSplitForLeafKernel` /
    `FindBestFromAllSplitsKernel`, `cuda_best_split_finder.cu:1920-2191`,
    8 ints per split). This kernel is that borrow: SCHEDULING, not
    arithmetic. The blocking boundary it removes is one of the two host
    waits per split the lossguide lane pays 63 times per tree.

    ORDER-DETERMINISTIC BY TRANSCRIPTION, NOT BY REDUCTION DESIGN. The
    fold below is the host loop VERBATIM, in the host loop's order --
    sequential over blocks `b` ascending, one incumbent record -- so
    there is no reduction-order freedom to differ across vendors or from
    the host fold it replaces. Every clause is the host's, in the host's
    sequence:

    * POISON SKIP FIRST: a record whose bin is 0xFFFFFFFF is skipped, and
      a leaf whose every record is poison ends UNDEFINED -- the corrected
      2026-08-22 semantics (their argmax poison vs default-ctor mismatch
      is outcome-equivalent to the skip; see the block at
      `greedy_search_helper_depthwise.mojo:1409-1436`).
    * BOUNDS SECOND: a non-poison bin-feature at or past `hist_cells`
      stops the fold and marks BIN_OUT_OF_RANGE with the offending value
      in the bin slot; the host raises the exact diagnostic the host fold
      raised, from the record instead of from the loop.
    * RESOLVE THIRD: flat bin-feature -> (feature, bin) through the
      caller-uploaded `TBinFeatureTable` columns, WITH their `ToSplit`
      clamp (`helpers.cpp:157-171`: cat bins clamp to `folds`, float bins
      to `folds - 1`) -- the host fold resolves through `table.to_split`
      and the comparator reads the CLAMPED bin, so the clamp must sit
      before the compare here too.
    * COMPARE LAST: `best_split_properties_less(cand, best)` transcribed
      (`helpers.mojo:112-141`, their `operator<`,
      `gpu_structures.h:80-93`): gain ascending under THEIR sign (the
      score kernel's gain negated), feature id as ui32, bin id as ui32,
      all strict -- so a full tie keeps the INCUMBENT, i.e. the earlier
      block, exactly as `if (less(cand, best)) best = cand` does.

    THE RECORD IS THE HOST'S STATE, BIT FOR BIT. Status DEFINED
    reconstructs `TBestSplitProperties(feature, bin, gain, gain)` (the
    fold stored the same number in Score and Gain, which is what
    `find_best_leaf_to_split`'s argmin reads); status UNDEFINED
    reconstructs the default record (feature -1, bin 0, gain Float32.MAX,
    defined False). Nothing downstream can tell the folds apart.

    WIRED 2026-08-27 in `greedy_search_helper_depthwise.mojo` (the merged
    non-symmetric driver -- the lossguide arm flows through the SAME fold
    site; `greedy_search_helper_lossguide.mojo` contains no fold, and its
    `find_best_leaf_to_split` argmin over the host leaf queue stays host
    by design: O(leaves), zero device traffic). The wiring is exactly the
    four steps this block specified: (a) the four `TBinFeatureTable`
    columns upload once per tree from workspace-owned staging pairs;
    (b) this kernel launches behind the score kernel with grid x = the
    scored-leaf count; (c) the two block-record copies became one
    `WINNER_RECORD_WORDS * leaves` copy under the SAME single
    `wait_complete`; (d) the unpack raises on BIN_OUT_OF_RANGE with the
    host fold's message and feeds `update_best_split` / `best_cells`
    (DEVIATION 1901's cell rides out in record word [4]). ALL OF IT UNDER
    THE FAST COMPTIME BRANCH (`SPLIT_COST_IDENTICAL`, the DEVIATION-1876
    pattern): IDENTICAL keeps the host fold byte-for-byte even though the
    winner is provably the same, because IDENTICAL's contract is the code
    path, not just the bits.
    ==================================================
    """
    if Int(thread_idx.x) != 0:
        return
    var row = Int(block_idx.x)
    var argmax_blocks = Int(argmax_blocks_in)
    var hist_cells = Int(hist_cells_in)

    # the host fold's seed: a default `TBestSplitProperties()` -- feature
    # sentinel, bin 0, gain Float32.MAX on THEIR lower-is-better sign
    var best_feature = Int32(-1)
    var best_bin = Int32(0)
    var best_gain = FLOAT32_MAX
    var status = WINNER_STATUS_UNDEFINED
    var bad_bf = UInt32(0)
    # DEVIATION 1901's `best_cell`, carried beside the record exactly as
    # the host fold carried it beside `best`
    var best_cell = WINNER_SENTINEL

    for b in range(argmax_blocks):
        var slot = row * argmax_blocks + b
        var bf = region_bin.unsafe_load(slot)
        if bf == WINNER_SENTINEL:
            # their poison record; skipping it is outcome-equivalent to
            # their compare-against-poison (see the deviation block)
            continue
        if Int(bf) >= hist_cells:
            # the host fold raises here and abandons the level; the
            # record carries the offending bin-feature out instead
            status = WINNER_STATUS_BIN_OUT_OF_RANGE
            bad_bf = bf
            break
        var our_gain = region_score.unsafe_load(slot)
        # THEIR sign, restored exactly where the host fold restored it
        var cand_gain = -our_gain
        var cand_feature = bf_feature.unsafe_load(Int(bf))
        var cand_bin = bf_bin.unsafe_load(Int(bf))
        # `ToSplit`'s clamp (`helpers.cpp:157-171`), before the compare,
        # because the host comparator saw the clamped bin
        var max_bin = bf_folds.unsafe_load(Int(bf)) - Int32(1)
        if bf_one_hot.unsafe_load(Int(bf)) != UInt8(0):
            max_bin = bf_folds.unsafe_load(Int(bf))
        if cand_bin > max_bin:
            cand_bin = max_bin
        # `best_split_properties_less(cand, best)`, transcribed: strict
        # everywhere, feature and bin compared as ui32 so the -1 sentinel
        # loses every gain tie instead of winning it
        var take = False
        if cand_gain < best_gain:
            take = True
        elif cand_gain == best_gain:
            var f1 = cand_feature.cast[DType.uint32]()
            var f2 = best_feature.cast[DType.uint32]()
            if f1 < f2:
                take = True
            elif f1 == f2:
                take = (
                    cand_bin.cast[DType.uint32]()
                    < best_bin.cast[DType.uint32]()
                )
        if take:
            best_feature = cand_feature
            best_bin = cand_bin
            best_gain = cand_gain
            best_cell = bf
            status = WINNER_STATUS_DEFINED

    var base = row * WINNER_RECORD_WORDS
    winner_records.unsafe_store(base + 0, best_feature.cast[DType.uint32]())
    if status == WINNER_STATUS_BIN_OUT_OF_RANGE:
        winner_records.unsafe_store(base + 1, bad_bf)
    else:
        winner_records.unsafe_store(base + 1, best_bin.cast[DType.uint32]())
    winner_records.unsafe_store(base + 2, bitcast[DType.uint32](best_gain))
    winner_records.unsafe_store(base + 3, status)
    winner_records.unsafe_store(base + 4, best_cell)
