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
from std.gpu import thread_idx
from std.memory import stack_allocation

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


# ---------------------------------------------------------------------------
# lane/apple-fast-trees-symmetric: the cross-block fold as one block
# (FAST + Apple only, `SYM_DEVICE_LEVEL` in pointwise_scores.mojo)
# ---------------------------------------------------------------------------

#: threads of `pw_fold_winner_block_kernel`
comptime PW_FOLD_BLOCK = 256


def pw_fold_winner_block_kernel[
    block_size: Int
](
    result_ids: MutPointer[UInt32, MutAnyOrigin],
    result_scores: MutPointer[Float32, MutAnyOrigin],
    block_count_in: Int32,
    is_first_in: Int32,
    best_ids: MutPointer[UInt32, MutAnyOrigin],
    best_scores: MutPointer[Float32, MutAnyOrigin],
):
    """`pw_fold_winner_kernel` for an UNCAPPED record count (the node-tiled
    score pass writes one record per 128 bin features: ~440 on istella),
    as one block instead of one thread.

    THE SAME ANSWER AS THE SEQUENTIAL FOLD. `_record_less` is a strict
    total order on (gain, feature, bin), so the minimum is the same
    whichever order the records are compared in, with one exception: a
    NaN gain, which the sequential fold never takes (`less` is false both
    ways). To keep that, each thread first folds its strided share
    SEQUENTIALLY from the sentinel (so a NaN record never becomes a
    thread's local), and only then do the locals meet in the shared tree,
    where every record is finite or the sentinel. The nesting is the
    original's: the helper's records fold into a LOCAL seeded with the
    sentinel, and the local folds ONCE into the global slot
    (`TakeBest(helper->Read(), best)`, ties keep the earlier policy).
    Thread 0 does the `is_first` seeding and that final fold, after the
    tree, which is the same global write order as the one-thread kernel.
    """
    var tid = Int(thread_idx.x)
    var block_count = Int(block_count_in)

    var loc_fid = PW_SENTINEL_ID
    var loc_bin = UInt32(0)
    var loc_score = FLOAT32_MAX
    var loc_gain = FLOAT32_MAX
    var b = tid
    while b < block_count:
        var c_fid = result_ids.unsafe_load(2 * b)
        var c_bin = result_ids.unsafe_load(2 * b + 1)
        var c_score = result_scores.unsafe_load(2 * b)
        var c_gain = result_scores.unsafe_load(2 * b + 1)
        if _record_less(c_gain, c_fid, c_bin, loc_gain, loc_fid, loc_bin):
            loc_fid = c_fid
            loc_bin = c_bin
            loc_score = c_score
            loc_gain = c_gain
        b += block_size

    var s_fid = stack_allocation[
        block_size,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_bin = stack_allocation[
        block_size,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_score = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_gain = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    s_fid[unsafe_offset=tid] = loc_fid
    s_bin[unsafe_offset=tid] = loc_bin
    s_score[unsafe_offset=tid] = loc_score
    s_gain[unsafe_offset=tid] = loc_gain
    barrier()

    # the barrier is outside the `if`, as in `_block_argmin_and_store`
    var s = block_size >> 1
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

    if tid == 0:
        if is_first_in != Int32(0):
            best_ids.unsafe_store(0, PW_SENTINEL_ID)
            best_ids.unsafe_store(1, UInt32(0))
            best_scores.unsafe_store(0, FLOAT32_MAX)
            best_scores.unsafe_store(1, FLOAT32_MAX)
        if block_count > 0:
            var l_fid = s_fid[unsafe_offset=0]
            var l_bin = s_bin[unsafe_offset=0]
            var l_score = s_score[unsafe_offset=0]
            var l_gain = s_gain[unsafe_offset=0]
            var g_fid = best_ids.unsafe_load(0)
            var g_bin = best_ids.unsafe_load(1)
            var g_gain = best_scores.unsafe_load(1)
            if _record_less(l_gain, l_fid, l_bin, g_gain, g_fid, g_bin):
                best_ids.unsafe_store(0, l_fid)
                best_ids.unsafe_store(1, l_bin)
                best_scores.unsafe_store(0, l_score)
                best_scores.unsafe_store(1, l_gain)


def launch_pw_fold_winner_block(
    ctx: DeviceContext,
    mut result_ids: DeviceBuffer[DType.uint32],
    mut result_scores: DeviceBuffer[DType.float32],
    block_count: Int,
    is_first: Bool,
    mut best_ids: DeviceBuffer[DType.uint32],
    mut best_scores: DeviceBuffer[DType.float32],
) raises:
    """`launch_pw_fold_winner`'s signature, one block of `PW_FOLD_BLOCK`."""
    ctx.enqueue_function[pw_fold_winner_block_kernel[PW_FOLD_BLOCK]](
        result_ids.unsafe_ptr(),
        result_scores.unsafe_ptr(),
        Int32(block_count),
        Int32(1) if is_first else Int32(0),
        best_ids.unsafe_ptr(),
        best_scores.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(PW_FOLD_BLOCK, 1, 1),
    )
