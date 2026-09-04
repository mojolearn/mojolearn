# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Focused gate for the first executable combination-CTR vertical slice."""

from gbdt.models.tensor_ctr_value_table import (
    build_feature_freq_tensor_table,
    build_borders_split_tensor_table,
    build_split_feature_freq_tensor_table,
    join_tensor_hash,
    materialize_tensor_candidate,
    parse_feature_freq_tensor_table,
    persist_winning_tensor_candidate,
    split_tensor_hash,
    stage_tensor_candidate_host,
    TTensorCtrRegistry,
    value_for_split_tensor_row,
)
from gbdt.ctrs.ctr_binarization import (
    BORDER_SELECTION_UNIFORM,
    TBinarizationOptions,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)
from gbdt.gpu_data.compressed_index_builder import pack_quantized_columns_host


def main() raises:
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
    for r in range(6):
        if split_expanded[18 + r] != want[r]:
            raise Error("split-history registry apply mismatch")

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
    print("tree CTR slice: deterministic pair FeatureFreq + text round trip PASS")
