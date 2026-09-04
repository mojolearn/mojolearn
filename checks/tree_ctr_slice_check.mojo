# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Focused gate for the first executable combination-CTR vertical slice."""

from gbdt.models.tensor_ctr_value_table import (
    build_feature_freq_tensor_table,
    build_borders_split_tensor_table,
    build_split_feature_freq_tensor_table,
    join_tensor_hash,
    insert_staged_tensor_candidate_device,
    materialize_tensor_candidate,
    parse_feature_freq_tensor_table,
    persist_ranked_tensor_winners,
    persist_synchronized_tensor_path,
    persist_winning_tensor_candidate,
    regenerate_feature_freq_after_winner,
    split_tensor_hash,
    stage_tensor_candidate_host,
    TStagedTensorCandidate,
    TTensorCtrRegistry,
    value_for_split_tensor_row,
    next_tensor_base_from_winner,
)
from max.gpu.host import DeviceContext
from gbdt.ctrs.ctr_binarization import (
    BORDER_SELECTION_UNIFORM,
    TBinarizationOptions,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)
from gbdt.gpu_data.compressed_index_builder import pack_quantized_columns_host
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    accept_symmetric_level_winner,
    TSynchronizedSymmetricLevelState,
    TTreeWorkspace,
    run_synchronized_symmetric_level,
    run_tree_layout,
)
from gbdt.gpu_data.feature_blocks import blocks_for
from core.identity_trace import IdentityTrace
from gbdt.methods.greedy_subsets_searcher.depthwise_stage_times import StageTimes
from gbdt.options.catboost_options import SCORE_FUNCTION_COSINE


def main() raises:
    var ctx = DeviceContext()
    # The model format must cover the full UInt64 hash domain. A direct
    # decimal parse through Int64 cannot represent this value.
    var high_hash = UInt64(0xFEDCBA9876543210)
    var words = split_tensor_hash(high_hash)
    var joined = join_tensor_hash(String(words[0]), String(words[1]))
    if joined != high_hash:
        raise Error("high-bit tensor hash does not round trip through u32 words")

    # column-major: two categorical columns and one irrelevant numeric one
    var x: List[Float32] = [
        0, 0, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 0,
        9, 8, 7, 6, 5, 4,
    ]
    var sources: List[Int] = [0, 1]
    var table = build_feature_freq_tensor_table(x, 6, 3, sources.copy())
    # keys (0,0),(0,1),(1,0),(1,1) have counts 2,1,1,2
    var want: List[Float32] = [
        Float32(2.0 / 7.0), Float32(1.0 / 7.0),
        Float32(1.0 / 7.0), Float32(2.0 / 7.0),
        Float32(2.0 / 7.0), Float32(2.0 / 7.0),
    ]
    for r in range(6):
        if table.value_for_row(x, 6, r) != want[r]:
            raise Error("combination FeatureFreq value mismatch at row " + String(r))

    var text = table.to_text()
    var loaded = parse_feature_freq_tensor_table(text)
    if loaded.to_text() != text:
        raise Error("combination CTR model record is not canonical on round trip")
    for r in range(6):
        if loaded.value_for_row(x, 6, r) != table.value_for_row(x, 6, r):
            raise Error("combination CTR apply changed after serialization")

    # Internal feature manager/cache: exact re-registration reuses a stable
    # model-column id and apply materializes that column after the raw inputs.
    var registry = TTensorCtrRegistry(3)
    var tensor_column = registry.register(loaded.copy())
    var same_column = registry.register(loaded.copy())
    if tensor_column != 3 or same_column != tensor_column:
        raise Error("tensor registry did not deduplicate to a stable column")
    var expanded = registry.expand_for_apply(x, 6, 3)
    if len(expanded) != 24:
        raise Error("tensor apply plan emitted the wrong model shape")
    for r in range(6):
        if expanded[3 * 6 + r] != want[r]:
            raise Error("registered tensor model column mismatch")

    # Same tensor identity with different learned state is not a second
    # feature and must not silently overwrite the cache entry.
    var conflicting = loaded.copy()
    conflicting.counts[0] += 1
    var refused = False
    try:
        _ = registry.register(conflicting^)
    except:
        refused = True
    if not refused:
        raise Error("tensor registry accepted conflicting learned state")

    # An unseen but in-range pair takes count zero and the prior-only value.
    var unseen = x.copy()
    unseen[0] = Float32(0.0)
    unseen[6] = Float32(1.0)
    if loaded.value_for_row(unseen, 6, 0) != Float32(1.0 / 7.0):
        # (0,1) is seen once; make the assertion explicit rather than vague.
        raise Error("combination CTR tensor key order changed")

    # Dynamic split history: cross the same categorical tensor with the
    # canonical predicate `quantized feature 0 > 0`, then prove that the
    # predicate and its counts survive model text and registry application.
    var cindex: List[UInt32] = [0, 0, 1, 1, 1, 0]
    var history: List[TBinarySplit] = [
        TBinarySplit(Int32(0), Int32(0), Int32(BIN_SPLIT_TAKE_GREATER))
    ]
    var split_table = build_split_feature_freq_tensor_table(
        x, cindex, 6, 3, sources.copy(), history^
    )
    var split_text = split_table.to_text()
    var split_loaded = parse_feature_freq_tensor_table(split_text)
    if split_loaded.to_text() != split_text:
        raise Error("split-history tensor is not canonical on round trip")
    for r in range(6):
        if value_for_split_tensor_row(
            split_loaded, x, cindex, 6, r
        ) != want[r]:
            raise Error("split-history FeatureFreq mismatch at row " + String(r))
    var split_registry = TTensorCtrRegistry(3)
    _ = split_registry.register(split_loaded^)
    var split_expanded = split_registry.expand_for_apply_with_bins(
        x, cindex, 6, 3
    )
    var apply_borders = List[List[Float32]]()
    var raw_split_border: List[Float32] = [Float32(0.5)]
    apply_borders.append(raw_split_border^)
    apply_borders.append(List[Float32]())
    apply_borders.append(List[Float32]())
    apply_borders.append(List[Float32]())
    var model_expanded = split_registry.expand_for_model_apply(
        x, 6, apply_borders^
    )
    for r in range(6):
        if split_expanded[18 + r] != want[r]:
            raise Error("split-history registry apply mismatch")
        if model_expanded[18 + r] != split_expanded[18 + r]:
            raise Error("tensor model apply did not reconstruct split bins")

    # Borders is ordered while fitting and full-pool at apply. The first
    # row of every tensor key sees only the prior; repeated keys see only
    # earlier rows in `order`. The serialized table carries the final
    # per-key target histograms, not those prefix values.
    var target: List[UInt8] = [0, 1, 1, 0, 1, 1]
    var order: List[UInt32] = [0, 1, 2, 3, 4, 5]
    var history2: List[TBinarySplit] = [
        TBinarySplit(Int32(0), Int32(0), Int32(BIN_SPLIT_TAKE_GREATER))
    ]
    var borders_fit = build_borders_split_tensor_table(
        x, cindex, target, order, 6, 3, sources.copy(), history2^,
        2, 0, Float32(0.5), Float32(1.0),
    )
    var learn_want: List[Float32] = [
        Float32(0.5), Float32(0.5), Float32(0.5),
        Float32(0.5), Float32(0.25), Float32(0.25),
    ]
    for r in range(6):
        if borders_fit.learn_values[r] != learn_want[r]:
            raise Error("ordered Borders prefix value mismatch")
    var borders_text = borders_fit.table.to_text()
    var borders_table = parse_feature_freq_tensor_table(borders_text)
    var apply_want: List[Float32] = [
        Float32(0.5), Float32(0.75), Float32(0.75),
        Float32(0.5), Float32(0.5), Float32(0.5),
    ]
    for r in range(6):
        if value_for_split_tensor_row(
            borders_table, x, cindex, 6, r
        ) != apply_want[r]:
            raise Error("Borders full-pool apply value mismatch")

    # Candidate seam consumed by compressed-index/ranking code: ordered
    # learn values, the CTR's own grid, and its quantized bins stay bound
    # to the same tensor/model table.
    var grid = TBinarizationOptions(BORDER_SELECTION_UNIFORM, 3)
    var candidate = materialize_tensor_candidate(
        borders_fit.table.copy(), borders_fit.learn_values.copy(), grid
    )
    if candidate.tensor_hash != borders_fit.table.tensor_hash or (
        len(candidate.bins) != 6 or len(candidate.borders) == 0
    ):
        raise Error("tensor CTR candidate lost its identity or shape")
    for r in range(6):
        if Int(candidate.bins[r]) > len(candidate.borders):
            raise Error("tensor CTR candidate emitted an invalid bin")
    if candidate.bins[0] != candidate.bins[1] or (
        candidate.bins[4] != candidate.bins[5]
    ):
        raise Error("equal ordered CTR values did not quantize equally")
    if candidate.bins[0] <= candidate.bins[4]:
        raise Error("tensor CTR candidate grid reversed value ordering")

    # Real compressed-index seam: appending the tensor can shift policy
    # blocks, so rebuild through `build_layout` rather than OR-ing into an
    # assumed trailing word. Decode from the resulting CFeature exactly as
    # the histogram kernels do and recover every candidate bin.
    var quantized = List[List[UInt32]]()
    quantized.append(cindex.copy())
    quantized.append(candidate.bins.copy())
    var candidate_folds = len(candidate.borders)
    var folds: List[Int] = [1, candidate_folds]
    var packed = pack_quantized_columns_host(quantized, folds)
    ref candidate_feature = packed.layout.features[1]
    if Int(candidate_feature.first_fold_index) != 1:
        raise Error("tensor candidate does not occupy the expected rank cells")
    for r in range(6):
        var word = packed.words[Int(candidate_feature.offset) * 6 + r]
        var decoded = (word >> candidate_feature.shift) & (
            candidate_feature.mask
        )
        if decoded != candidate.bins[r]:
            raise Error("compressed tensor candidate bin mismatch")

    # The reusable synchronized-driver boundary applies exactly the same
    # host gates and split resolution as the baseline post-tree walk.
    var accepted_splits = List[TBinarySplit]()
    if not accept_symmetric_level_winner(
        packed.layout, Float32(1.0), candidate_feature.first_fold_index,
        packed.layout.hist_cells, 0, 1, accepted_splits,
    ):
        raise Error("one-level winner boundary rejected an improving split")
    if len(accepted_splits) != 1 or accepted_splits[0].feature_id != Int32(1):
        raise Error("one-level winner boundary resolved the wrong feature")
    if accept_symmetric_level_winner(
        packed.layout, Float32(1.0), candidate_feature.first_fold_index,
        packed.layout.hist_cells, 1, 2, accepted_splits,
    ):
        raise Error("one-level winner boundary accepted a repeated split")

    # Candidate feature ids are ephemeral until ranking chooses one. Stage
    # two candidates at the same next id, persist only the selected Borders
    # candidate, and prove the losing FeatureFreq table never enters model
    # apply state.
    var freq_candidate = materialize_tensor_candidate(
        loaded.copy(), want.copy(), grid
    )
    var base_columns = List[List[UInt32]]()
    base_columns.append(cindex.copy())
    var base_folds: List[Int] = [1]
    var staged_freq = stage_tensor_candidate_host(
        base_columns, base_folds, List[Bool](), freq_candidate^
    )
    var staged_borders = stage_tensor_candidate_host(
        base_columns, base_folds, List[Bool](), candidate^
    )
    if staged_freq.feature_id != 1 or staged_borders.feature_id != 1:
        raise Error("dynamic tensor candidates did not share the next feature id")
    var winner_registry = TTensorCtrRegistry(1)
    var winning_column = persist_winning_tensor_candidate(
        staged_borders, winner_registry
    )
    if winning_column != 1 or len(winner_registry.features) != 1:
        raise Error("winning tensor was not persisted at its ranked feature id")
    if winner_registry.features[0].table.tensor_hash != (
        staged_borders.candidate.tensor_hash
    ):
        raise Error("model registry persisted the losing tensor candidate")

    # Device insertion starts from the extended layout with only this
    # candidate's bitfield cleared. The production OR-writer must restore
    # exactly the host reference words without moving neighbouring features.
    ref staged_cf = staged_borders.compressed.layout.features[
        staged_borders.feature_id
    ]
    var shifted_mask = staged_cf.mask << staged_cf.shift
    var base_words = staged_borders.compressed.words.copy()
    for r in range(6):
        var at = Int(staged_cf.offset) * 6 + r
        base_words[at] &= ~shifted_mask
    var device_words = insert_staged_tensor_candidate_device(
        ctx, staged_borders, base_words^
    )
    var host_words = ctx.enqueue_create_host_buffer[DType.uint32](
        len(staged_borders.compressed.words)
    )
    ctx.enqueue_copy(dst_ptr=host_words.unsafe_ptr(), src_buf=device_words)
    ctx.synchronize()
    for i in range(len(staged_borders.compressed.words)):
        if host_words[i] != staged_borders.compressed.words[i]:
            raise Error("device tensor insertion differs from host packing")
    _ = host_words^
    _ = device_words^

    # Invoke the real symmetric histogram/scoring loop. The tensor carries
    # the dominant signal and must win level one; a weaker independent base
    # signal leaves legitimate gain for level two. A tensor-only target would
    # make the first split pure and correctly stop at the acceptance gate.
    var secondary: List[UInt32] = [0, 1, 0, 1, 1, 0]
    var zero_columns = List[List[UInt32]]()
    zero_columns.append(secondary.copy())
    var dynamic_stage = stage_tensor_candidate_host(
        zero_columns, base_folds, List[Bool](),
        staged_borders.candidate.copy(),
    )
    var dynamic_base = dynamic_stage.compressed.words.copy()
    ref dynamic_cf = dynamic_stage.compressed.layout.features[
        dynamic_stage.feature_id
    ]
    var dynamic_mask = dynamic_cf.mask << dynamic_cf.shift
    for r in range(6):
        var at = Int(dynamic_cf.offset) * 6 + r
        dynamic_base[at] &= ~dynamic_mask
    var dynamic_device = insert_staged_tensor_candidate_device(
        ctx, dynamic_stage, dynamic_base^
    )
    var base_device = ctx.enqueue_create_buffer[DType.uint32](6)
    var h_base = ctx.enqueue_create_host_buffer[DType.uint32](6)
    for r in range(6):
        h_base.unsafe_ptr().unsafe_store(r, secondary[r])
    ctx.enqueue_copy(dst_buf=base_device, src_ptr=h_base.unsafe_ptr())
    var stats = ctx.enqueue_create_buffer[DType.float32](12)
    var h_stats = ctx.enqueue_create_host_buffer[DType.float32](12)
    var grad_mag = Float32(0.0)
    for r in range(6):
        h_stats.unsafe_ptr().unsafe_store(r, Float32(1.0))
        var tensor_signal = Float32(-4.0) if dynamic_stage.candidate.bins[r] == (
            UInt32(0)
        ) else Float32(4.0)
        var base_signal = Float32(-1.0) if secondary[r] == UInt32(0) else (
            Float32(1.0)
        )
        var g = tensor_signal + base_signal
        h_stats.unsafe_ptr().unsafe_store(6 + r, g)
        grad_mag += -g if g < Float32(0.0) else g
    ctx.enqueue_copy(dst_buf=stats, src_ptr=h_stats.unsafe_ptr())
    var rows = ctx.enqueue_create_buffer[DType.uint32](6)
    var h_rows = ctx.enqueue_create_host_buffer[DType.uint32](6)
    for r in range(6):
        h_rows.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=rows, src_ptr=h_rows.unsafe_ptr())
    var cursor = ctx.enqueue_create_buffer[DType.float32](6)
    ctx.enqueue_memset(cursor, Float32(0.0))
    var splits = List[TBinarySplit]()
    var leaves = List[Float32]()
    var offsets = List[Int]()
    var workspace = List[TTreeWorkspace]()
    var dynamic_folds: List[Int] = [1, len(dynamic_stage.candidate.borders)]
    _ = run_tree_layout(
        ctx, 6, base_folds, 2, base_device, stats, rows, cursor,
        Float32(6.0), grad_mag, splits, leaves, offsets, workspace,
        dynamic_cindex=Optional(dynamic_device.copy()),
        dynamic_fold_counts=dynamic_folds,
    )
    if len(splits) != 2 or Int(splits[0].feature_id) != (
        dynamic_stage.feature_id
    ):
        raise Error("baseline did not rank the staged tensor before base signal")

    # Pin the dynamic slot at this CTR configuration's maximum requested
    # border count. Reserving all 255 UInt8 values would expose hundreds of
    # empty, equivalent thresholds to scoring; the configured grid is the
    # tight upper bound shared by every per-level regeneration.
    var pinned_initial = stage_tensor_candidate_host(
        zero_columns, base_folds, List[Bool](),
        dynamic_stage.candidate.copy(), fold_capacity=grid.border_count,
    )
    var pinned_base = pinned_initial.compressed.words.copy()
    ref pinned_cf = pinned_initial.compressed.layout.features[
        pinned_initial.feature_id
    ]
    var pinned_mask = pinned_cf.mask << pinned_cf.shift
    for r in range(6):
        var at = Int(pinned_cf.offset) * 6 + r
        pinned_base[at] &= ~pinned_mask
    var pinned_device = insert_staged_tensor_candidate_device(
        ctx, pinned_initial, pinned_base.copy()
    )
    var reference_device = insert_staged_tensor_candidate_device(
        ctx, pinned_initial, pinned_base^
    )

    # Opt-in synchronized driver, initially over the same planted signal.
    # The baseline monolith above remains the independent first-level oracle.
    var sync_rows = ctx.enqueue_create_buffer[DType.uint32](6)
    var sync_stats = ctx.enqueue_create_buffer[DType.float32](12)
    ctx.enqueue_copy(dst_buf=sync_rows, src_ptr=h_rows.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=sync_stats, src_ptr=h_stats.unsafe_ptr())
    var sync_layout = pinned_initial.compressed.layout.copy()
    var sync_blocks = blocks_for(sync_layout, 6)
    var sync_state = TSynchronizedSymmetricLevelState(
        ctx, sync_layout^, sync_blocks^, pinned_device^, 6, 2, 2, False
    )
    sync_state.initialize_tree(ctx, Float32(6.0), grad_mag)
    var reference_rows = ctx.enqueue_create_buffer[DType.uint32](6)
    var reference_stats = ctx.enqueue_create_buffer[DType.float32](12)
    ctx.enqueue_copy(dst_buf=reference_rows, src_ptr=h_rows.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=reference_stats, src_ptr=h_stats.unsafe_ptr())
    var reference_layout = pinned_initial.compressed.layout.copy()
    var reference_blocks = blocks_for(reference_layout, 6)
    var reference_state = TSynchronizedSymmetricLevelState(
        ctx, reference_layout^, reference_blocks^, reference_device^,
        6, 2, 2, False,
    )
    reference_state.initialize_tree(ctx, Float32(6.0), grad_mag)
    var no_trace = IdentityTrace.disabled()
    var no_times = StageTimes()
    no_times.enabled = False
    var sync_splits = List[TBinarySplit]()
    var reference_splits = List[TBinarySplit]()
    if not run_synchronized_symmetric_level(
        sync_state, ctx, sync_rows, sync_stats, sync_splits,
        SCORE_FUNCTION_COSINE, False, Float32(3.0), Float32(0.0),
        UInt64(0), True, no_trace, no_times, "tensor_sync",
    ):
        raise Error("synchronized tensor driver rejected its planted winner")
    if not run_synchronized_symmetric_level(
        reference_state, ctx, reference_rows, reference_stats,
        reference_splits, SCORE_FUNCTION_COSINE, False, Float32(3.0),
        Float32(0.0), UInt64(0), False, no_trace, no_times,
        "tensor_sync_reference",
    ):
        raise Error("full-histogram tensor reference rejected level one")
    ctx.synchronize()
    if len(sync_splits) != 1 or sync_splits[0].feature_id != splits[0].feature_id:
        raise Error("synchronized tensor driver disagrees with baseline winner")
    if reference_splits[0].feature_id != sync_splits[0].feature_id or (
        reference_splits[0].bin_idx != sync_splits[0].bin_idx
    ):
        raise Error(
            "full-histogram reference disagrees at level one: sync feature="
            + String(sync_splits[0].feature_id) + " bin="
            + String(sync_splits[0].bin_idx) + " reference feature="
            + String(reference_splits[0].feature_id) + " bin="
            + String(reference_splits[0].bin_idx)
        )
    if sync_state.level != 1 or sync_state.live_leaves != 2:
        raise Error("synchronized tensor driver did not advance level state")
    if sync_state.last_pre_score_accounting != 3:
        raise Error("first synchronized level changed baseline pre-score schedule")

    # Persist from the real ranked split list, rather than from the candidate
    # batch. This is the model-state boundary that prevents losing candidates
    # from leaking into inference state.
    var ranked_registry = TTensorCtrRegistry(dynamic_stage.feature_id)
    var ranked_candidates = List[TStagedTensorCandidate]()
    ranked_candidates.append(dynamic_stage.copy())
    var persisted = persist_ranked_tensor_winners(
        splits.copy(), ranked_candidates, ranked_registry
    )
    if len(persisted) != 1 or persisted[0] != dynamic_stage.feature_id:
        raise Error("ranked tensor winner was not persisted")
    if len(ranked_registry.features) != 1 or (
        ranked_registry.features[0].table.tensor_hash
        != dynamic_stage.candidate.table.tensor_hash
    ):
        raise Error("persisted tensor winner does not match ranked candidate")

    # The selected split deterministically advances the tensor projection for
    # candidate regeneration at the next tree level.
    var next_tensor = next_tensor_base_from_winner(dynamic_stage, splits[0])
    if len(next_tensor.splits) != len(
        dynamic_stage.candidate.table.splits
    ) + 1:
        raise Error("winning tensor split did not advance split history")
    if next_tensor.get_hash() == dynamic_stage.candidate.table.tensor_hash:
        raise Error("next-level tensor identity did not change")

    # Regenerate the next candidate from the selected dynamic column itself.
    # The extended column-major index is the host mirror of the layout that
    # the real ranker consumed above.
    var extended_cindex = cindex.copy()
    for r in range(6):
        extended_cindex.append(dynamic_stage.candidate.bins[r])
    var regenerated = regenerate_feature_freq_after_winner(
        dynamic_stage, splits[0], x, extended_cindex, 6, 3, grid
    )
    if regenerated.tensor_hash != next_tensor.get_hash() or (
        len(regenerated.table.splits) != len(next_tensor.splits)
    ):
        raise Error("next-level tensor candidate was not regenerated")
    for r in range(6):
        if regenerated.values[r] != value_for_split_tensor_row(
            regenerated.table, x, extended_cindex, 6, r
        ):
            raise Error("regenerated tensor candidate value mismatch")

    # A synchronized dynamic driver pins the ephemeral column capacity, so
    # regeneration rewrites only its bits and never invalidates histogram or
    # workspace sizing established before the tree loop.
    var pinned_next = stage_tensor_candidate_host(
        zero_columns, base_folds, List[Bool](), regenerated^,
        fold_capacity=grid.border_count,
    )
    ref initial_cf = pinned_initial.compressed.layout.features[
        pinned_initial.feature_id
    ]
    ref next_cf = pinned_next.compressed.layout.features[pinned_next.feature_id]
    if (
        pinned_initial.feature_id != pinned_next.feature_id
        or initial_cf.offset != next_cf.offset
        or initial_cf.mask != next_cf.mask
        or initial_cf.shift != next_cf.shift
        or initial_cf.first_fold_index != next_cf.first_fold_index
        or initial_cf.folds != next_cf.folds
    ):
        raise Error("pinned tensor regeneration changed the search layout")

    # Replace the accepted level's active column only after its reorder has
    # drained, then execute level two. The replacement invalidates parent
    # histograms, so this level must rebuild all live leaves; subtracting a
    # new-column child from an old-column parent is mathematically invalid.
    var replacement_base = pinned_next.compressed.words.copy()
    ref replacement_cf = pinned_next.compressed.layout.features[
        pinned_next.feature_id
    ]
    var replacement_mask = replacement_cf.mask << replacement_cf.shift
    for r in range(6):
        var at = Int(replacement_cf.offset) * 6 + r
        replacement_base[at] &= ~replacement_mask
    var replacement_device = insert_staged_tensor_candidate_device(
        ctx, pinned_next, replacement_base.copy()
    )
    var reference_replacement_device = insert_staged_tensor_candidate_device(
        ctx, pinned_next, replacement_base^
    )
    sync_state.replace_active_cindex(ctx, replacement_device^)
    reference_state.replace_active_cindex(ctx, reference_replacement_device^)
    if not run_synchronized_symmetric_level(
        sync_state, ctx, sync_rows, sync_stats, sync_splits,
        SCORE_FUNCTION_COSINE, False, Float32(3.0), Float32(0.0),
        UInt64(1), True, no_trace, no_times, "tensor_sync",
    ):
        var rejected_rows = sync_state.drain_live_row_count(ctx)
        raise Error(
            "synchronized tensor driver rejected level-two winner: score="
            + String(sync_state.last_winner_score) + " bin="
            + String(sync_state.last_winner_bin) + " rows="
            + String(rejected_rows) + " accounting="
            + String(sync_state.last_pre_score_accounting)
        )
    if not run_synchronized_symmetric_level(
        reference_state, ctx, reference_rows, reference_stats,
        reference_splits, SCORE_FUNCTION_COSINE, False, Float32(3.0),
        Float32(0.0), UInt64(1), False, no_trace, no_times,
        "tensor_sync_reference",
    ):
        var rejected_rows = reference_state.drain_live_row_count(ctx)
        raise Error(
            "full-histogram tensor reference rejected level two: score="
            + String(reference_state.last_winner_score) + " bin="
            + String(reference_state.last_winner_bin) + " rows="
            + String(rejected_rows) + " accounting="
            + String(reference_state.last_pre_score_accounting)
        )
    if len(sync_splits) != 2 or sync_state.level != 2 or (
        sync_state.live_leaves != 4
    ):
        raise Error("synchronized tensor driver did not complete two levels")
    if sync_splits[1].feature_id != reference_splits[1].feature_id or (
        sync_splits[1].bin_idx != reference_splits[1].bin_idx
    ):
        raise Error("synchronized winner disagrees with full-histogram reference")
    if sync_state.last_pre_score_accounting != 3:
        raise Error("replacement level did not rebuild full histograms")
    if reference_state.last_pre_score_accounting != 3:
        raise Error("level-two reference did not use the full histogram path")

    # Production persistence cannot serialize the synchronized driver's
    # reused ephemeral slot directly. Remap its level-ordered candidates to
    # distinct stable columns, including split-history references.
    var path_splits = List[TBinarySplit]()
    path_splits.append(splits[0])
    var second_bin = (Int(splits[0].bin_idx) + 1) % grid.border_count
    path_splits.append(TBinarySplit(
        Int32(dynamic_stage.feature_id), Int32(second_bin),
        Int32(BIN_SPLIT_TAKE_GREATER),
    ))
    var path_candidates = List[TStagedTensorCandidate]()
    path_candidates.append(pinned_initial.copy())
    path_candidates.append(pinned_next.copy())
    var path_registry = TTensorCtrRegistry(2)
    var path_columns = persist_synchronized_tensor_path(
        path_splits, path_candidates^, path_registry
    )
    if len(path_columns) != 2 or path_columns[0] != 2 or path_columns[1] != 3:
        raise Error("synchronized tensor path did not allocate stable columns")
    if path_splits[0].feature_id != Int32(2) or (
        path_splits[1].feature_id != Int32(3)
    ):
        raise Error("synchronized tensor winners were not remapped")
    var found_stable_parent = False
    for s in range(len(path_registry.features[1].table.splits)):
        if path_registry.features[1].table.splits[s].feature_id == Int32(2):
            found_stable_parent = True
    if not found_stable_parent:
        raise Error("tensor split history retained its ephemeral parent id")
    ref sync_workspace = sync_state.workspace[0]
    ctx.enqueue_copy(
        dst_ptr=sync_workspace.h_sz.unsafe_ptr(), src_buf=sync_workspace.p_sz
    )
    ctx.synchronize()
    var conserved_rows = 0
    for leaf in range(sync_state.live_leaves):
        conserved_rows += Int(sync_workspace.h_sz.unsafe_ptr().unsafe_load(leaf))
    if conserved_rows != 6:
        raise Error("synchronized tensor driver lost rows across two levels")
    print("tree CTR slice: deterministic pair FeatureFreq + text round trip PASS")
