# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The pointwise winner fold and pack, on the device -- DEVIATION 207.

NO CATBOOST COUNTERPART AS KERNELS. Upstream resolves each level's winner
on the HOST: `TFindBestSplitsHelper::ReadOptimalSplit`
(`histograms_helper.h:248-252`) reads the per-block records back and folds
them with `TakeBest`, `TScoresCalcerOnCompressedDataSet::ReadOptimalSplit`
(`pointwise_scores_calcer.h:94-105`) folds the helpers, and the searcher
(`oblivious_tree_doc_parallel_structure_searcher.cpp:113-134`) consumes the
winner -- a blocking read per level, cheap on their pinned memory (~5 us)
and ~191 us plus a queue-empty bubble on this box (the greedy census,
DEVIATION 94). So the level loop here is enqueued BLIND: the same two
nested folds run as one-thread kernels, the winner record lands in a
per-level device array, and the host reads ALL levels once per tree.

The greedy family made the identical move in
`greedy_subsets_searcher/kernel/split_resolve.mojo`; this file is its
pointwise sibling and copies the DISCIPLINE (same sequential order, same
tie rules as the host code it replaces), not its code -- the record
formats differ.

WHAT `take_best` MEANS, BIT FOR BIT (`gbdt/methods/helpers.mojo`):
`first < second ? first : second` under a comparator ordering by ascending
`Gain`, then `FeatureId` AS `ui32` (so the `-1` sentinel LOSES every tie),
then `BinId` as `ui32`, all strict -- a FULL tie falls through to
`second`. The two host folds pass their arguments in OPPOSITE orders and
both are theirs: the per-block and per-helper folds put the CHALLENGER
first (ties keep the incumbent), the searcher's level fold puts the
INCUMBENT first (a tying challenger replaces it). The kernels here
replicate the first two; the third has nothing to fold against (the
searcher's incumbent is the default record, which loses to anything
defined and ties only another default).
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from gbdt.gpu_data.apple_fast_trees_experiments import AFT_G03

from gbdt.methods.kernel.sym_fast import SYM_RESOLVE_BLOCK
from gbdt.apple_fast_classical import AFCL_RESOLVE_RECORDS
from gbdt.methods.pointwise_optimization_subsets import SPLIT_BLOCK_SIZE

comptime PW_SENTINEL_ID = UInt32(0xFFFFFFFF)
"""`(ui32)-1`, `TBestSplitProperties::FeatureId`'s default
(`gpu_structures.h:64`)."""

comptime FLOAT32_MAX = Float32(3.4028234663852886e38)
"""`Score` and `Gain` defaults (`points_subsets.mojo:33-38`), the same
bits as `pointwise_scores.mojo:282` and `Float32.MAX` -- the sentinel
record must compare EQUAL to the host default under the gain key."""


@always_inline
def _record_less(
    gain_a: Float32,
    fid_a: UInt32,
    bin_a: UInt32,
    gain_b: Float32,
    fid_b: UInt32,
    bin_b: UInt32,
) -> Bool:
    """`TBestSplitProperties::operator<` (`gpu_structures.h:80-93`),
    the same three-key strict order `best_split_properties_less` carries on
    the host -- gain, then feature id as `ui32`, then bin as `ui32`."""
    if gain_a < gain_b:
        return True
    elif gain_a == gain_b:
        if fid_a < fid_b:
            return True
        elif fid_a == fid_b:
            return bin_a < bin_b
        else:
            return False
    else:
        return False


def pw_fold_winner_kernel(
    result_ids: MutPointer[UInt32, MutAnyOrigin],
    result_scores: MutPointer[Float32, MutAnyOrigin],
    block_count_in: Int32,
    is_first_in: Int32,
    best_ids: MutPointer[UInt32, MutAnyOrigin],
    best_scores: MutPointer[Float32, MutAnyOrigin],
):
    """One helper's blocks folded, then folded into the running incumbent.

    LITERALLY the host nesting, not a flattened equivalent: the helper's
    blocks fold into a LOCAL record seeded with the sentinel
    (`PolicyScoreHelper.read_optimal_split`'s loop, challenger first, tie
    keeps the incumbent -- so the EARLIEST block wins a tie), and the local
    record then folds ONCE into the global slot (`TakeBest(helper->Read(),
    best)`, ties keep the EARLIER policy). The direct fold happens to be
    equivalent here -- a comparator tie implies identical records because
    `Gain` is monotone in `Score` per (feature, bin) and a feature lives in
    exactly one policy -- but CONTRIBUTING.md (Algorithms and references) says implementation the branch, not
    the reachability argument.

    ONE THREAD. The work is at most 32 records; a parallel reduction would
    buy nothing and cost the sequential-order guarantee the tie rule needs.

    `is_first` seeds the global slot with the default record first, which
    is also the whole story of the empty-calcer call (`block_count == 0`):
    the slot ends up holding the sentinel, exactly what the host fold
    returns when no helper has features.
    """
    if Int(thread_idx.x) != 0:
        return

    var block_count = Int(block_count_in)

    if is_first_in != Int32(0):
        best_ids.unsafe_store(0, PW_SENTINEL_ID)
        best_ids.unsafe_store(1, UInt32(0))
        best_scores.unsafe_store(0, FLOAT32_MAX)
        best_scores.unsafe_store(1, FLOAT32_MAX)

    # the helper-local fold, seeded with the default record
    var loc_fid = PW_SENTINEL_ID
    var loc_bin = UInt32(0)
    var loc_score = FLOAT32_MAX
    var loc_gain = FLOAT32_MAX
    for b in range(block_count):
        var c_fid = result_ids.unsafe_load(2 * b)
        var c_bin = result_ids.unsafe_load(2 * b + 1)
        var c_score = result_scores.unsafe_load(2 * b)
        var c_gain = result_scores.unsafe_load(2 * b + 1)
        # `take_best(challenger, incumbent)`: challenger wins only if
        # strictly less
        if _record_less(c_gain, c_fid, c_bin, loc_gain, loc_fid, loc_bin):
            loc_fid = c_fid
            loc_bin = c_bin
            loc_score = c_score
            loc_gain = c_gain

    if block_count > 0:
        # `TakeBest(helper->Read(), best)`: the local record challenges the
        # global incumbent once
        var g_fid = best_ids.unsafe_load(0)
        var g_bin = best_ids.unsafe_load(1)
        var g_gain = best_scores.unsafe_load(1)
        if _record_less(loc_gain, loc_fid, loc_bin, g_gain, g_fid, g_bin):
            best_ids.unsafe_store(0, loc_fid)
            best_ids.unsafe_store(1, loc_bin)
            best_scores.unsafe_store(0, loc_score)
            best_scores.unsafe_store(1, loc_gain)


def pw_pack_winner_kernel(
    best_ids: MutPointer[UInt32, MutAnyOrigin],
    best_scores: MutPointer[Float32, MutAnyOrigin],
    depth_in: Int32,
    winners_ids: MutPointer[UInt32, MutAnyOrigin],
    winners_scores: MutPointer[Float32, MutAnyOrigin],
    score_before: MutPointer[Float32, MutAnyOrigin],
    feat_table: MutPointer[UInt32, MutAnyOrigin],
    n_features_in: Int32,
    split_desc: MutPointer[UInt32, MutAnyOrigin],
):
    """The level's winner into the tree record, the next level's score, and
    the split descriptor -- everything the host loop used to do with the
    record between the read and the split.

    * `winners_*[2*depth .. 2*depth+1]`: the record, for the ONE post-tree
      drain (the searcher's gates and `structure` appends move there).
    * `score_before[0] = Score`: the searcher's
      `score_before_split = best.score`, read by the NEXT level's score
      kernels (their loop-carried host float).
    * `split_desc`: `(offset_elems, mask, shift, one_hot, bin)`, the five
      scalars the searcher passed to `split_subsets` from
      `layout.features[fid]` / `one_hot[fid]` -- here read from
      `feat_table` (4 words per feature, same order, uploaded once per
      workspace from the same two host tables).

    A SENTINEL WINNER PACKS FEATURE 0. When every candidate scored
    non-finite the record holds `(ui32)-1`, the host loop RAISES, and the
    levels this blind loop still has in flight must not index the table at
    -1: they run a well-formed (feature 0, bin 0) split whose output the
    post-tree walk never reads -- it raises at this level's record, their
    message, before touching any later level.
    """
    if Int(thread_idx.x) != 0:
        return

    var fid = best_ids.unsafe_load(0)
    var bin = best_ids.unsafe_load(1)
    var score = best_scores.unsafe_load(0)
    var gain = best_scores.unsafe_load(1)
    var depth = Int(depth_in)

    winners_ids.unsafe_store(2 * depth, fid)
    winners_ids.unsafe_store(2 * depth + 1, bin)
    winners_scores.unsafe_store(2 * depth, score)
    winners_scores.unsafe_store(2 * depth + 1, gain)

    # NOT DECORATION, AND NOT CURRENTLY RANKING-VISIBLE EITHER: with
    # `binFeaturesWeights` hard-coded to ones (the implementation's state), `gain =
    # (score - scoreBeforeSplit) * w` shifts every candidate by the same
    # constant and the argmin cannot move -- a sabotage that skips this
    # store passes every fit gate (PREP_BILL step 27). It is carried
    # because THEIRS carries it and becomes load-bearing the moment
    # per-feature weights are implemented; its plumbing is held by the S1
    # per-cell gain values (score_before = -3.25 there) and by
    # `checks/pointwise_resolve_check.mojo` reading this word back.
    score_before.unsafe_store(0, score)

    var fid_c = Int(fid)
    if fid >= UInt32(n_features_in):
        fid_c = 0
    split_desc.unsafe_store(0, feat_table.unsafe_load(4 * fid_c))
    split_desc.unsafe_store(1, feat_table.unsafe_load(4 * fid_c + 1))
    split_desc.unsafe_store(2, feat_table.unsafe_load(4 * fid_c + 2))
    split_desc.unsafe_store(3, feat_table.unsafe_load(4 * fid_c + 3))
    split_desc.unsafe_store(4, bin)


def pw_seed_sentinel_kernel(
    best_ids: MutPointer[UInt32, MutAnyOrigin],
    best_scores: MutPointer[Float32, MutAnyOrigin],
):
    """The default record into the winner slot -- what the host fold
    returns when no helper has features. Its own kernel because the fold
    kernel's `is_first` arm cannot serve: Mojo refuses the same buffer
    passed as both source and destination at the call (the DEVIATION 97.2
    aliasing rule)."""
    if Int(thread_idx.x) != 0:
        return
    best_ids.unsafe_store(0, PW_SENTINEL_ID)
    best_ids.unsafe_store(1, UInt32(0))
    best_scores.unsafe_store(0, FLOAT32_MAX)
    best_scores.unsafe_store(1, FLOAT32_MAX)


def launch_pw_seed_sentinel(
    ctx: DeviceContext,
    mut best_ids: DeviceBuffer[DType.uint32],
    mut best_scores: DeviceBuffer[DType.float32],
) raises:
    ctx.enqueue_function[pw_seed_sentinel_kernel](
        best_ids.unsafe_ptr(),
        best_scores.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )


def launch_pw_fold_winner(
    ctx: DeviceContext,
    mut result_ids: DeviceBuffer[DType.uint32],
    mut result_scores: DeviceBuffer[DType.float32],
    block_count: Int,
    is_first: Bool,
    mut best_ids: DeviceBuffer[DType.uint32],
    mut best_scores: DeviceBuffer[DType.float32],
) raises:
    ctx.enqueue_function[pw_fold_winner_kernel](
        result_ids.unsafe_ptr(),
        result_scores.unsafe_ptr(),
        Int32(block_count),
        Int32(1) if is_first else Int32(0),
        best_ids.unsafe_ptr(),
        best_scores.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )


def launch_pw_pack_winner(
    ctx: DeviceContext,
    mut best_ids: DeviceBuffer[DType.uint32],
    mut best_scores: DeviceBuffer[DType.float32],
    depth: Int,
    mut winners_ids: DeviceBuffer[DType.uint32],
    mut winners_scores: DeviceBuffer[DType.float32],
    mut score_before: DeviceBuffer[DType.float32],
    mut feat_table: DeviceBuffer[DType.uint32],
    n_features: Int,
    mut split_desc: DeviceBuffer[DType.uint32],
) raises:
    ctx.enqueue_function[pw_pack_winner_kernel](
        best_ids.unsafe_ptr(),
        best_scores.unsafe_ptr(),
        Int32(depth),
        winners_ids.unsafe_ptr(),
        winners_scores.unsafe_ptr(),
        score_before.unsafe_ptr(),
        feat_table.unsafe_ptr(),
        Int32(n_features),
        split_desc.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )


# ================= DEVIATION 3111 (scheduling only) =================
# THE POINTWISE LEVEL'S WINNER, PACK AND BIN UPDATE IN ONE LAUNCH (lane
# hr-gbdt-small). The Ordered structure search enqueued, per level, one
# one-thread `pw_fold_winner_kernel` per policy helper (or the sentinel
# seed), the one-thread `pw_pack_winner_kernel`, then
# `update_bins_from_desc_kernel` reading the descriptor the pack wrote: two
# to four launches whose only work is a fold of at most 32 records per
# helper and a descriptor copy. Here every thread of the bin update runs
# the SAME nested fold (`_fold_helper`, the per-block fold then the
# per-helper fold, the chain's order and tie rules: challenger first, ties
# keep the incumbent), block (0, 0) thread 0 makes the chain's stores
# (`best_*`, `winners_*`, `score_before`, `split_desc`), and every thread
# takes the descriptor from its registers. Integer and compare work only;
# the bins written are the unfused kernel's. `-D
# MOJOLEARN_GBDT_FUSED_LEVEL_OFF` restores the chain.
# ====================================================================
comptime PW_FUSED_LEVEL = not is_defined["MOJOLEARN_GBDT_FUSED_LEVEL_OFF"]()
# the searcher's fused launch alone, for bisecting
comptime PW_FUSED_SEARCH = PW_FUSED_LEVEL and not is_defined[
    "MOJOLEARN_GBDT_FUSED_PW_OFF"
]()


@always_inline
def _fold_winner_kern(
    result_ids: MutPointer[UInt32, MutAnyOrigin],
    result_scores: MutPointer[Float32, MutAnyOrigin],
    block_count: Int,
    mut g_fid: UInt32,
    mut g_bin: UInt32,
    mut g_score: Float32,
    mut g_gain: Float32,
):
    """`pw_fold_winner_kernel`'s body after its seed, in registers."""
    var loc_fid = PW_SENTINEL_ID
    var loc_bin = UInt32(0)
    var loc_score = FLOAT32_MAX
    var loc_gain = FLOAT32_MAX
    for b in range(block_count):
        var c_fid = result_ids.unsafe_load(2 * b)
        var c_bin = result_ids.unsafe_load(2 * b + 1)
        var c_score = result_scores.unsafe_load(2 * b)
        var c_gain = result_scores.unsafe_load(2 * b + 1)
        if _record_less(c_gain, c_fid, c_bin, loc_gain, loc_fid, loc_bin):
            loc_fid = c_fid
            loc_bin = c_bin
            loc_score = c_score
            loc_gain = c_gain
    if block_count > 0:
        if _record_less(loc_gain, loc_fid, loc_bin, g_gain, g_fid, g_bin):
            g_fid = loc_fid
            g_bin = loc_bin
            g_score = loc_score
            g_gain = loc_gain


@always_inline
def _fold_block(
    r0_ids: MutPointer[UInt32, MutAnyOrigin],
    r0_scores: MutPointer[Float32, MutAnyOrigin],
    n0: Int,
    r1_ids: MutPointer[UInt32, MutAnyOrigin],
    r1_scores: MutPointer[Float32, MutAnyOrigin],
    n1: Int,
    r2_ids: MutPointer[UInt32, MutAnyOrigin],
    r2_scores: MutPointer[Float32, MutAnyOrigin],
    n2: Int,
    mut g_fid: UInt32,
    mut g_bin: UInt32,
    mut g_score: Float32,
    mut g_gain: Float32,
):
    """lane/apple-fast-sym-hist, `-D MOJOLEARN_SYM_RESOLVE_BLOCK` (FAST +
    Apple only): the three `_fold_helper` calls as ONE fold per
    threadgroup. Threads stride the concatenated record list (helper 0,
    then 1, then 2), each folding from the sentinel under `_record_less`,
    then a shared-memory tree reduce under the same order. `_record_less`
    is a strict total order on (gain, feature, bin) and a tie means
    identical records (a feature lives in one policy; the sentinel and the
    `bf[0]`-at-FLOAT32_MAX fillers tie only themselves), so every fold
    order returns the same record as the sequential one. Every thread
    reads the result from slot 0. Requires `block_dim.x ==
    SPLIT_BLOCK_SIZE`, a power of two, which `pw_resolve_pack_bins_kernel`'s
    launch guarantees."""
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)
    var total = n0 + n1 + n2

    var loc_fid = PW_SENTINEL_ID
    var loc_bin = UInt32(0)
    var loc_score = FLOAT32_MAX
    var loc_gain = FLOAT32_MAX
    # AFCL-T05 changes only the assignment of the same complete candidate
    # list to lanes. Four adjacent records improve per-lane metadata reuse;
    # tails are guarded before any record load.
    var first = tid * AFCL_RESOLVE_RECORDS
    while first < total:
        for record in range(AFCL_RESOLVE_RECORDS):
            var r = first + record
            if r >= total:
                break
            var c_fid: UInt32
            var c_bin: UInt32
            var c_score: Float32
            var c_gain: Float32
            if r < n0:
                c_fid = r0_ids.unsafe_load(2 * r)
                c_bin = r0_ids.unsafe_load(2 * r + 1)
                c_score = r0_scores.unsafe_load(2 * r)
                c_gain = r0_scores.unsafe_load(2 * r + 1)
            elif r < n0 + n1:
                var q = r - n0
                c_fid = r1_ids.unsafe_load(2 * q)
                c_bin = r1_ids.unsafe_load(2 * q + 1)
                c_score = r1_scores.unsafe_load(2 * q)
                c_gain = r1_scores.unsafe_load(2 * q + 1)
            else:
                var q = r - n0 - n1
                c_fid = r2_ids.unsafe_load(2 * q)
                c_bin = r2_ids.unsafe_load(2 * q + 1)
                c_score = r2_scores.unsafe_load(2 * q)
                c_gain = r2_scores.unsafe_load(2 * q + 1)
            if _record_less(c_gain, c_fid, c_bin, loc_gain, loc_fid, loc_bin):
                loc_fid = c_fid
                loc_bin = c_bin
                loc_score = c_score
                loc_gain = c_gain
        first += threads * AFCL_RESOLVE_RECORDS

    var s_fid = stack_allocation[
        SPLIT_BLOCK_SIZE,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_bin = stack_allocation[
        SPLIT_BLOCK_SIZE,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_score = stack_allocation[
        SPLIT_BLOCK_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_gain = stack_allocation[
        SPLIT_BLOCK_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    s_fid[unsafe_offset=tid] = loc_fid
    s_bin[unsafe_offset=tid] = loc_bin
    s_score[unsafe_offset=tid] = loc_score
    s_gain[unsafe_offset=tid] = loc_gain
    barrier()

    var s = threads >> 1
    while s > 0:
        if tid < s:
            if _record_less(
                s_gain[unsafe_offset=tid + s],
                s_fid[unsafe_offset=tid + s],
                s_bin[unsafe_offset=tid + s],
                s_gain[unsafe_offset=tid],
                s_fid[unsafe_offset=tid],
                s_bin[unsafe_offset=tid],
            ):
                s_fid[unsafe_offset=tid] = s_fid[unsafe_offset=tid + s]
                s_bin[unsafe_offset=tid] = s_bin[unsafe_offset=tid + s]
                s_score[unsafe_offset=tid] = s_score[unsafe_offset=tid + s]
                s_gain[unsafe_offset=tid] = s_gain[unsafe_offset=tid + s]
        barrier()
        s >>= 1

    g_fid = s_fid[unsafe_offset=0]
    g_bin = s_bin[unsafe_offset=0]
    g_score = s_score[unsafe_offset=0]
    g_gain = s_gain[unsafe_offset=0]


def pw_resolve_pack_bins_kernel(
    r0_ids: MutPointer[UInt32, MutAnyOrigin],
    r0_scores: MutPointer[Float32, MutAnyOrigin],
    n0_in: Int32,
    r1_ids: MutPointer[UInt32, MutAnyOrigin],
    r1_scores: MutPointer[Float32, MutAnyOrigin],
    n1_in: Int32,
    r2_ids: MutPointer[UInt32, MutAnyOrigin],
    r2_scores: MutPointer[Float32, MutAnyOrigin],
    n2_in: Int32,
    best_ids: MutPointer[UInt32, MutAnyOrigin],
    best_scores: MutPointer[Float32, MutAnyOrigin],
    depth_in: Int32,
    winners_ids: MutPointer[UInt32, MutAnyOrigin],
    winners_scores: MutPointer[Float32, MutAnyOrigin],
    score_before: MutPointer[Float32, MutAnyOrigin],
    feat_table: MutPointer[UInt32, MutAnyOrigin],
    n_features_in: Int32,
    split_desc: MutPointer[UInt32, MutAnyOrigin],
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    bin_depth: UInt32,
    bins: MutPointer[UInt32, MutAnyOrigin],
):
    """DEVIATION 3111. Helpers absent from the calcer (or empty) come in
    with a count of 0, which folds nothing, exactly as the chain skips
    them; the global record starts as the sentinel the chain's first fold
    (or `pw_seed_sentinel_kernel`) seeds. Grid and block are
    `update_bins_from_desc_kernel`'s."""
    var g_fid = PW_SENTINEL_ID
    var g_bin = UInt32(0)
    var g_score = FLOAT32_MAX
    var g_gain = FLOAT32_MAX
    comptime if AFT_G03:
        # G03: preserve the original serial record order, but only lane 0
        # performs it. Broadcast costs one barrier and sixteen shared bytes,
        # avoiding both per-row folds and the older block-tree reduction.
        # Uncompiled/unverified/unmeasured; original tie comparison retained.
        var ids = stack_allocation[
            2, UInt32, address_space=AddressSpace.SHARED
        ]()
        var scores = stack_allocation[
            2, Float32, address_space=AddressSpace.SHARED
        ]()
        if Int(thread_idx.x) == 0:
            _fold_winner_kern(r0_ids, r0_scores, Int(n0_in), g_fid, g_bin, g_score, g_gain)
            _fold_winner_kern(r1_ids, r1_scores, Int(n1_in), g_fid, g_bin, g_score, g_gain)
            _fold_winner_kern(r2_ids, r2_scores, Int(n2_in), g_fid, g_bin, g_score, g_gain)
            ids[unsafe_offset=0] = g_fid
            ids[unsafe_offset=1] = g_bin
            scores[unsafe_offset=0] = g_score
            scores[unsafe_offset=1] = g_gain
        barrier()
        g_fid = ids[unsafe_offset=0]
        g_bin = ids[unsafe_offset=1]
        g_score = scores[unsafe_offset=0]
        g_gain = scores[unsafe_offset=1]
    elif SYM_RESOLVE_BLOCK:
        _fold_block(
            r0_ids, r0_scores, Int(n0_in),
            r1_ids, r1_scores, Int(n1_in),
            r2_ids, r2_scores, Int(n2_in),
            g_fid, g_bin, g_score, g_gain,
        )
    else:
        _fold_winner_kern(r0_ids, r0_scores, Int(n0_in), g_fid, g_bin, g_score, g_gain)
        _fold_winner_kern(r1_ids, r1_scores, Int(n1_in), g_fid, g_bin, g_score, g_gain)
        _fold_winner_kern(r2_ids, r2_scores, Int(n2_in), g_fid, g_bin, g_score, g_gain)

    # `pw_pack_winner_kernel`'s descriptor, in registers
    var fid_c = Int(g_fid)
    if g_fid >= UInt32(n_features_in):
        fid_c = 0
    var d_off = feat_table.unsafe_load(4 * fid_c)
    var d_mask = feat_table.unsafe_load(4 * fid_c + 1)
    var d_shift = feat_table.unsafe_load(4 * fid_c + 2)
    var d_one_hot = feat_table.unsafe_load(4 * fid_c + 3)

    if (
        Int(block_idx.x) == 0
        and Int(block_idx.y) == 0
        and Int(thread_idx.x) == 0
    ):
        best_ids.unsafe_store(0, g_fid)
        best_ids.unsafe_store(1, g_bin)
        best_scores.unsafe_store(0, g_score)
        best_scores.unsafe_store(1, g_gain)
        var depth = Int(depth_in)
        winners_ids.unsafe_store(2 * depth, g_fid)
        winners_ids.unsafe_store(2 * depth + 1, g_bin)
        winners_scores.unsafe_store(2 * depth, g_score)
        winners_scores.unsafe_store(2 * depth + 1, g_gain)
        score_before.unsafe_store(0, g_score)
        split_desc.unsafe_store(0, d_off)
        split_desc.unsafe_store(1, d_mask)
        split_desc.unsafe_store(2, d_shift)
        split_desc.unsafe_store(3, d_one_hot)
        split_desc.unsafe_store(4, g_bin)

    # `update_bins_from_desc_kernel`, the descriptor from registers
    var size = Int(size_in)
    var f_offset = Int(d_off)
    var feature_shift = d_shift
    var value = g_bin << feature_shift
    var mask = d_mask << feature_shift
    var one_hot = d_one_hot != UInt32(0)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < size:
        var idx = Int(indices.unsafe_load(i))
        var feature_val = compressed_index.unsafe_load(f_offset + idx) & mask
        var goes_right: Bool
        if one_hot:
            goes_right = feature_val == value
        else:
            goes_right = feature_val > value
        if goes_right:
            bins.unsafe_store(i, bins.unsafe_load(i) | (UInt32(1) << bin_depth))
        i += stride
