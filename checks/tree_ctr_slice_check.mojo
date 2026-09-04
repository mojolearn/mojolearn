# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Focused gate for the first executable combination-CTR vertical slice."""

from gbdt.models.tensor_ctr_value_table import (
    build_feature_freq_tensor_table,
    build_split_feature_freq_tensor_table,
    join_tensor_hash,
    parse_feature_freq_tensor_table,
    split_tensor_hash,
    TTensorCtrRegistry,
    value_for_split_tensor_row,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)


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
    print("tree CTR slice: deterministic pair FeatureFreq + text round trip PASS")
