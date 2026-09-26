# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ExperimentalTwoLevelFeatureFreq fit on the host, a SECOND spelling of
`gbdt/estimator.mojo::gbdt_fit_two_level_feature_freq` on the
gbdt-feature-freq lane (lane/cpu-training-gbdt-ordered, 2026-09-15).

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu`, a `DeviceContext` or a
module that defines a kernel. The imports are the `checks/numerics` seams,
`checks/fixed_point.choose_scale`, the GPU-free host modules the device fit
ITSELF runs on the host (`dense_category_code`, `compute_ctr_borders`,
`pack_quantized_columns_host`, `blocks_for`), and the symmetric oracle
(`gbdt/host/gbdt_oracle.mojo`), whose restatements of the greedy searcher's
histogram, scan, partition-stat and Cosine kernels this fit reaches
unchanged. The host helpers of `gbdt/models/tensor_ctr_value_table.mojo`
are RESTATED below, because that module imports `max.gpu.host` for its
device staging entry.

THE CONFIGURATION THIS COVERS, by name (tools/identity_break.py
`gbdt-feature-freq`: `sources=[0, 1]`, `random_state=7`, the defaults
`learning_rate=0.03`, `l2_leaf_reg=3.0`, on `_coded(X)`). The binding refuses
by name: `sample_weight` (the weighted arm is not measured), a feature whose
histogram policy is BinaryFeatures (a numeric column whose Uniform-3 grid
collapses to one border; the symmetric oracle does not restate that policy),
and a tree whose level winner is the FeatureFreq tensor column itself (the
registry, its canonical tensor hash and the tensor apply are not restated).

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT

  1. `gbdt_fit_two_level_feature_freq` (`gbdt/estimator.mojo:106-235`): the
     source columns become dense one-hot codes (`folds = max_code + 1`,
     borders `code + 0.5`), every other column is binned on its Uniform-3
     grid (`compute_ctr_borders`, the bin is the count of borders strictly
     below the value), the column-major `flat_bins`.
  2. `build_feature_freq_tensor_table` and `value_for_key`
     (`gbdt/models/tensor_ctr_value_table.mojo:167-232`, `:113-137`): the
     sorted sources, the mixed-radix key, the integer counts, and the one
     Float32 division `(count + 0) / (n_rows + 1)`; then
     `materialize_tensor_candidate` (`:480-504`) and
     `stage_tensor_candidate_host` (`:373-417`) at the pinned fold capacity
     3, packed by `pack_quantized_columns_host`.
  3. `fit_two_level_feature_freq_tree` (`gbdt/methods/doc_parallel_boosting.
     mojo:443-535`): plane 0 the unit weight, plane 1 `1.0 * y`, the Float32
     row-order magnitudes and `choose_scale(max(weight, gradient), n_rows)`
     (`initialize_tree`, `greedy_search_helper.mojo:4192-4232`).
  4. `run_sequential_two_level_feature_freq_tree`
     (`greedy_search_helper.mojo:4621-4719`) and, per level,
     `run_synchronized_symmetric_level` (`:4566-4605`):
       - `enqueue_pre_score`: the histograms of every live leaf (the first
         level has one; the second rebuilds both, because
         `replace_active_cindex` sets `cindex_changed_since_histogram`, so no
         subtraction runs), the symmetric oracle's `_half_byte_block` and
         `_one_byte_block`; `scan_histograms_kernel`; `compute_partition_
         stats` at the pinned 32 chunks (`_partition_stat`).
       - `enqueue_score`: `compute_optimal_splits_kernel[COSINE]` with no
         noise, skip all zero and feature weight 1 (`_cosine_gain`), the
         block argmax and `resolve_and_pack_kernel` (largest gain, ties to
         the smaller bin-feature).
       - `accept_symmetric_level_winner` (`:3835-3880`): the sentinel raise,
         `score > 0`, the repeat check, TakeBin for a one-hot feature.
       - `enqueue_post_winner`: the stable split of every live leaf, zeros
         then ones, the rows and both stat planes moved with them, the right
         child at `n_live + i`.
     Between the levels, `stage_next_feature_freq_after_winner`
     (`tensor_ctr_value_table.mojo:1126-1184`): the split-history table
     over the level-one winner's bit (`_split_bit`, `:235-247`), its values,
     borders, bins, and the repacked words.
  5. The leaves: `two_level_weighted_leaf_value` (`doc_parallel_boosting.
     mojo:424-440`), the Float32 fold in final row order, then
     `learning_rate * total / (total_weight + l2)`.
  6. `model_text` (`gbdt/models/model_text.mojo:374-670`) for the one
     oblivious tree: `features n n`, `type cat` for the sources, the
     trailing `split_type take_bin` on a one-hot split, no losses, no bias.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1`
(`GBDT_ORACLE_HOST_SABOTAGE`) adds 1.0 to the l2 regularizer of the leaf
value, so every nonzero leaf moves and every fixture's predictions and model
move with it.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the gbdt-feature-freq lane is the measurement.
"""
from std.math import isfinite

from checks.fixed_point import choose_scale
from checks.numerics import ftz
from gbdt.ctrs.ctr_binarization import (
    BORDER_SELECTION_UNIFORM,
    TBinarizationOptions,
    compute_ctr_borders,
)
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    HostCompressedIndex,
    pack_quantized_columns_host,
)
from gbdt.gpu_data.feature_blocks import PolicyBlock, blocks_for
from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from gbdt.host.gbdt_oracle import (
    GBDT_FLOAT32_MAX,
    GBDT_ORACLE_HOST_SABOTAGE,
    GBDT_SENTINEL,
    _cosine_gain,
    _half_byte_block,
    _one_byte_block,
    _partition_stat,
    gbdt_f32_token,
)
from gbdt.models.ctr_value_table import dense_category_code


#: `TBinarizationOptions(BORDER_SELECTION_UNIFORM, 3)`
#: (`gbdt/estimator.mojo:194-197`, `:215`) and the pinned fold capacity
#: `grid.border_count` (`:218-219`).
comptime GBDT_FF_GRID_BORDERS = 3
#: `BIN_SPLIT_TAKE_BIN` and `BIN_SPLIT_TAKE_GREATER`
#: (`gbdt/models/oblivious_model.mojo:36-37`).
comptime GBDT_FF_TAKE_BIN = 0
comptime GBDT_FF_TAKE_GREATER = 1


def _refuse_ff(what: String) raises:
    raise Error(
        "no CPU implementation of"
        " _mojolearn_gbdt.gbdt_fit_two_level_feature_freq for " + what
        + "; the gbdt host binding trains the gbdt-feature-freq lane only"
        " (unit weights, half-byte and one-byte policies, an ordinary-feature"
        " winner at both levels), see gbdt/host/gbdt_oracle_feature_freq.mojo"
    )


# ===========================================================================
# THE TENSOR COLUMN (`tensor_ctr_value_table.mojo`, host helpers restated)
# ===========================================================================


@fieldwise_init
struct _FFTable(Movable):
    """`TFeatureFreqTensorTable` for FeatureFreq (no target classes, prior
    0 / 1): the sorted sources, their cardinalities, the canonical split
    history (feature, bin, type) and the counts."""

    var sources: List[Int]
    var cards: List[Int]
    var split_feature: List[Int]
    var split_bin: List[Int]
    var split_type: List[Int]
    var counts: List[Int]


def _code_key(
    t: _FFTable, x_colmajor: List[Float32], n_rows: Int, row: Int
) raises -> Int:
    """`key_for_row` (`:94-104`)."""
    var key = 0
    for i in range(len(t.sources)):
        var f = t.sources[i]
        var code = dense_category_code(x_colmajor[f * n_rows + row], f, row)
        if code >= t.cards[i]:
            return -1
        key = key * t.cards[i] + code
    return key


def _split_bit(
    feature: Int, bin: Int, split_type: Int, flat: List[UInt32], n_rows: Int,
    row: Int,
) raises -> Int:
    """`_split_bit` (`:235-247`) over the flat quantized columns."""
    if feature < 0 or bin < 0 or feature * n_rows + row >= len(flat):
        raise Error("split-history tensor references an invalid quantized column")
    var value = Int(flat[feature * n_rows + row])
    if split_type == GBDT_FF_TAKE_BIN:
        return 1 if value == bin else 0
    if split_type == GBDT_FF_TAKE_GREATER:
        return 1 if value > bin else 0
    raise Error("split-history tensor has an unknown split type")


def _ff_table(
    x_colmajor: List[Float32],
    n_rows: Int,
    n_features: Int,
    var sources: List[Int],
    split_feature: List[Int],
    split_bin: List[Int],
    split_type: List[Int],
    flat: List[UInt32],
) raises -> _FFTable:
    """`build_feature_freq_tensor_table` (`:167-232`) and, with a split,
    `build_split_feature_freq_tensor_table` (`:250-292`): the insertion sort
    of the sources, the dense-code checks, the mixed-radix counts, then one
    bit per split appended to the key."""
    if n_rows <= 0:
        raise Error("a combination CTR needs at least one training row")
    if len(sources) < 2:
        raise Error("a combination CTR needs at least two source features")
    for i in range(1, len(sources)):
        var j = i
        while j > 0 and sources[j] < sources[j - 1]:
            var tmp = sources[j]
            sources[j] = sources[j - 1]
            sources[j - 1] = tmp
            j -= 1
    var cards = List[Int]()
    var entry_count = 1
    for i in range(len(sources)):
        var f = sources[i]
        if f < 0 or f >= n_features:
            raise Error("combination CTR source feature is out of range")
        if i > 0 and sources[i - 1] == f:
            raise Error("combination CTR source features must be unique")
        var max_code = 0
        var seen = List[Bool](length=1, fill=False)
        for r in range(n_rows):
            var code = dense_category_code(x_colmajor[f * n_rows + r], f, r)
            if code > max_code:
                max_code = code
                seen.resize(max_code + 1, False)
            seen[code] = True
        for code in range(max_code + 1):
            if not seen[code]:
                raise Error("combination CTR source is not densely coded")
        var card = max_code + 1
        if entry_count > 10000000 // card:
            raise Error("combination CTR dense table exceeds 10,000,000 entries")
        entry_count *= card
        cards.append(card)
    var entries = entry_count
    for _ in range(len(split_feature)):
        if entries > 10000000 // 2:
            raise Error("split-history tensor exceeds 10,000,000 entries")
        entries *= 2
    var t = _FFTable(
        sources^, cards^, split_feature.copy(), split_bin.copy(),
        split_type.copy(), List[Int](length=entries, fill=0),
    )
    for r in range(n_rows):
        var key = 0
        for i in range(len(t.sources)):
            var f = t.sources[i]
            var code = dense_category_code(x_colmajor[f * n_rows + r], f, r)
            key = key * t.cards[i] + code
        for s in range(len(t.split_feature)):
            key = 2 * key + _split_bit(
                t.split_feature[s], t.split_bin[s], t.split_type[s], flat,
                n_rows, r,
            )
        t.counts[key] += 1
    return t^


def _ff_values(
    t: _FFTable, x_colmajor: List[Float32], n_rows: Int, flat: List[UInt32]
) raises -> List[Float32]:
    """`value_for_row` / `value_for_split_tensor_row` (`:106-137`,
    `:295-308`): FeatureFreq with prior 0 / 1, one Float32 division."""
    var values = List[Float32](length=n_rows, fill=Float32(0.0))
    for row in range(n_rows):
        var key = _code_key(t, x_colmajor, n_rows, row)
        if key >= 0:
            for s in range(len(t.split_feature)):
                key = 2 * key + _split_bit(
                    t.split_feature[s], t.split_bin[s], t.split_type[s], flat,
                    n_rows, row,
                )
        var count = 0
        if key >= 0 and key < len(t.counts):
            count = t.counts[key]
        values[row] = (Float32(count) + Float32(0.0)) / (
            Float32(n_rows) + Float32(1.0)
        )
    return values^


@fieldwise_init
struct _FFStaged(Movable):
    var borders: List[Float32]
    var bins: List[UInt32]
    var compressed: HostCompressedIndex


def _ff_stage(
    values: List[Float32],
    base_columns: List[List[UInt32]],
    base_folds: List[Int],
    base_one_hot: List[Bool],
) raises -> _FFStaged:
    """`materialize_tensor_candidate` (`:480-504`) then
    `stage_tensor_candidate_host` (`:373-417`) at the pinned capacity."""
    if len(values) == 0:
        raise Error("cannot rank an empty tensor CTR column")
    var borders = compute_ctr_borders(
        values, TBinarizationOptions(BORDER_SELECTION_UNIFORM, GBDT_FF_GRID_BORDERS)
    )
    if len(borders) > 255:
        raise Error("tensor CTR candidate exceeds one-byte fold capacity")
    var bins = List[UInt32](length=len(values), fill=UInt32(0))
    for r in range(len(values)):
        var bin = 0
        for b in range(len(borders)):
            if values[r] > borders[b]:
                bin += 1
        bins[r] = UInt32(bin)
    if len(borders) == 0:
        raise Error("tensor candidate has no rankable border")
    if GBDT_FF_GRID_BORDERS < len(borders):
        raise Error("tensor candidate exceeds its pinned fold capacity")
    var feature_id = len(base_columns)
    var columns = List[List[UInt32]]()
    for i in range(len(base_columns)):
        columns.append(base_columns[i].copy())
    columns.append(bins.copy())
    var folds = base_folds.copy()
    folds.append(GBDT_FF_GRID_BORDERS)
    var one_hot = base_one_hot.copy()
    one_hot.append(False)
    var compressed = pack_quantized_columns_host(columns^, folds^, one_hot^)
    if compressed.layout.features[feature_id].one_hot_feature:
        raise Error("a tensor CTR was registered as a one-hot feature")
    return _FFStaged(borders^, bins^, compressed^)


# ===========================================================================
# ONE SYNCHRONIZED LEVEL
# ===========================================================================


@fieldwise_init
struct _FFWinner(ImplicitlyCopyable, Movable):
    var feature: Int
    var bin: Int
    var split_type: Int


def _ff_level(
    layout: CompressedIndexLayout,
    words: List[UInt32],
    n_rows: Int,
    level: Int,
    mut row_index: List[Int],
    mut stats: List[Float32],
    mut p_off: List[Int],
    mut p_sz: List[Int],
    fixed_scale: Float32,
    l2_leaf_reg: Float32,
    mut splits: List[_FFWinner],
) raises:
    """`run_synchronized_symmetric_level` for one level with no subtraction:
    every live leaf's histograms, the scan, the partition stats, the score,
    the winner and its gates, then the split of every live leaf."""
    var n_live = 1 << level
    var n_features = len(layout.features)
    var hist_cells = layout.hist_cells
    var blocks = blocks_for(layout, n_rows)
    for b in range(len(blocks)):
        if blocks[b].policy == POLICY_BINARY:
            _refuse_ff(
                "a feature with exactly one border (the BinaryFeatures"
                " histogram policy, feature "
                + String(blocks[b].feature_ids[0]) + ")"
            )
    var compute = List[Int]()
    for j in range(n_live):
        compute.append(j)
    var hist = List[Float32](length=n_live * 2 * hist_cells, fill=Float32(0.0))
    var block_first_bin = 0
    for b in range(len(blocks)):
        ref blk = blocks[b]
        var total = 0
        for k in range(blk.count()):
            total += Int(blk.folds[k])
        if blk.policy == POLICY_HALF_BYTE:
            _half_byte_block(
                blk, block_first_bin, hist_cells, compute, level,
                p_off, p_sz, row_index, stats, words, n_rows, fixed_scale,
                hist,
            )
        elif blk.policy == POLICY_ONE_BYTE:
            _one_byte_block(
                blk, block_first_bin, hist_cells, compute, p_off, p_sz,
                row_index, stats, words, layout, n_rows, fixed_scale, hist,
            )
        block_first_bin += total

    # `scan_histograms_kernel` over the computed leaves
    for j in range(n_live):
        for z in range(2):
            for f in range(n_features):
                ref cf = layout.features[f]
                var folds = Int(cf.folds)
                if cf.one_hot_feature or folds <= 1:
                    continue
                var base = j * 2 * hist_cells + z * hist_cells + Int(cf.first_fold_index)
                var running = Float32(0.0)
                for i in range(folds):
                    running = ftz(running + hist[base + i])
                    hist[base + i] = running

    var part_stats = List[Float32](length=2 * n_live, fill=Float32(0.0))
    for i in range(n_live):
        part_stats[2 * i] = _partition_stat(stats, n_rows, 0, p_off[i], p_sz[i])
        part_stats[2 * i + 1] = _partition_stat(stats, n_rows, 1, p_off[i], p_sz[i])

    var best_gain = -GBDT_FLOAT32_MAX
    var best_bin = GBDT_SENTINEL
    for bf in range(hist_cells):
        var gain = _cosine_gain(hist, hist_cells, part_stats, n_live, bf, l2_leaf_reg)
        if gain > best_gain:
            best_gain = gain
            best_bin = UInt32(bf)
    var best_score = ftz(best_gain) if best_bin != GBDT_SENTINEL else ftz(-GBDT_FLOAT32_MAX)

    # `accept_symmetric_level_winner` (`greedy_search_helper.mojo:3835-3880`)
    if best_bin == GBDT_SENTINEL or Int(best_bin) >= hist_cells:
        raise Error(
            "All splits have infinite score. Probably, numerical"
            " overflow occurs in loss function and/or split score"
            " calculation. Try increasing l2_leaf_reg, and/or"
            " decreasing learning_rate, etc."
            " [level " + String(level) + ", live leaves "
            + String(n_live) + "]"
        )
    var feature = -1
    var fbin = 0
    for i in range(n_features):
        ref lf = layout.features[i]
        if lf.folds == 0:
            continue
        var lo = Int(lf.first_fold_index)
        if Int(best_bin) >= lo and Int(best_bin) < lo + Int(lf.folds):
            feature = i
            fbin = Int(best_bin) - lo
            break
    if feature < 0:
        raise Error(
            "bin-feature " + String(best_bin)
            + " belongs to no feature; the histogram and the layout disagree"
        )
    var rejected = not (best_score > Float32(0.0))
    for i in range(len(splits)):
        if splits[i].feature == feature and splits[i].bin == fbin:
            rejected = True
    if rejected:
        if level == 0:
            raise Error("sequential tensor tree rejected level-one winner")
        raise Error("sequential tensor tree rejected level-two winner")
    ref sfeat = layout.features[feature]
    splits.append(_FFWinner(
        feature, fbin,
        GBDT_FF_TAKE_BIN if sfeat.one_hot_feature else GBDT_FF_TAKE_GREATER,
    ))

    # `enqueue_post_winner`: split flags, the stable partition, the reorder
    # of rows and both stat planes, the partition update
    var new_rows = row_index.copy()
    var new_stats = stats.copy()
    for i in range(n_live):
        var off = p_off[i]
        var sz = p_sz[i]
        var zeros = List[Int]()
        var ones = List[Int]()
        for k in range(sz):
            var row = row_index[off + k]
            var word = words[Int(sfeat.offset) * n_rows + row]
            var feature_val = word & (sfeat.mask << sfeat.shift)
            var value = UInt32(fbin) << sfeat.shift
            var goes_right: Bool
            if sfeat.one_hot_feature:
                goes_right = feature_val == value
            else:
                goes_right = feature_val > value
            if goes_right:
                ones.append(k)
            else:
                zeros.append(k)
        var dst = 0
        for k in range(len(zeros)):
            var src = off + zeros[k]
            new_rows[off + dst] = row_index[src]
            new_stats[off + dst] = stats[src]
            new_stats[n_rows + off + dst] = stats[n_rows + src]
            dst += 1
        for k in range(len(ones)):
            var src = off + ones[k]
            new_rows[off + dst] = row_index[src]
            new_stats[off + dst] = stats[src]
            new_stats[n_rows + off + dst] = stats[n_rows + src]
            dst += 1
        var left_sz = len(zeros)
        p_sz[i] = left_sz
        p_off[n_live + i] = off + left_sz
        p_sz[n_live + i] = sz - left_sz
    row_index = new_rows^
    stats = new_stats^


# ===========================================================================
# THE FIT
# ===========================================================================


def gbdt_feature_freq_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    sources_in: List[Int],
    learning_rate: Float32,
    l2_leaf_reg: Float32,
) raises -> String:
    """`gbdt_fit_two_level_feature_freq` on the covered configuration; returns
    the model text. See the module docstring for what mirrors what."""
    if n_rows < 1 or n_features < 2 or len(sources_in) < 2:
        raise Error("two-level FeatureFreq fit needs rows and two sources")
    if len(x_colmajor) != n_rows * n_features or len(y) != n_rows:
        raise Error("two-level FeatureFreq input shape mismatch")
    for r in range(n_rows):
        if not isfinite(y[r]):
            raise Error("two-level FeatureFreq target is not finite")
    var source_ids = List[Int]()
    var seen_source = List[Bool](length=n_features, fill=False)
    for i in range(len(sources_in)):
        var source = sources_in[i]
        if source < 0 or source >= n_features or seen_source[source]:
            raise Error("two-level FeatureFreq source is invalid or duplicated")
        seen_source[source] = True
        source_ids.append(source)

    # ---- 1. the base columns (`estimator.mojo:163-209`) ----
    var columns = List[List[UInt32]]()
    var folds = List[Int]()
    var one_hot = List[Bool]()
    var borders = List[List[Float32]]()
    var flat = List[UInt32]()
    for f in range(n_features):
        var col = List[UInt32]()
        var feature_borders = List[Float32]()
        if seen_source[f]:
            var max_code = -1
            var seen_codes = List[Bool]()
            for r in range(n_rows):
                var code = dense_category_code(x_colmajor[f * n_rows + r], f, r)
                if code > max_code:
                    max_code = code
                    seen_codes.resize(max_code + 1, False)
                seen_codes[code] = True
                col.append(UInt32(code))
            if max_code < 1:
                raise Error("two-level FeatureFreq source column is constant")
            for code in range(max_code + 1):
                if not seen_codes[code]:
                    raise Error("two-level FeatureFreq categories must be dense")
            folds.append(max_code + 1)
            one_hot.append(True)
            for code in range(max_code):
                feature_borders.append(Float32(code) + Float32(0.5))
        else:
            var numeric_values = List[Float32]()
            var numeric_changes = False
            for r in range(n_rows):
                var value = x_colmajor[f * n_rows + r]
                if not isfinite(value):
                    raise Error("two-level FeatureFreq numeric column is not finite")
                if r > 0 and value != numeric_values[0]:
                    numeric_changes = True
                numeric_values.append(value)
            if not numeric_changes:
                raise Error("two-level FeatureFreq numeric column is constant")
            feature_borders = compute_ctr_borders(
                numeric_values,
                TBinarizationOptions(BORDER_SELECTION_UNIFORM, GBDT_FF_GRID_BORDERS),
            )
            for r in range(n_rows):
                var bin = 0
                for b in range(len(feature_borders)):
                    if numeric_values[r] > feature_borders[b]:
                        bin += 1
                col.append(UInt32(bin))
            folds.append(len(feature_borders))
            one_hot.append(False)
        for r in range(n_rows):
            flat.append(col[r])
        columns.append(col^)
        borders.append(feature_borders^)

    # ---- 2. the level-one tensor column ----
    var no_splits = List[Int]()
    var table = _ff_table(
        x_colmajor, n_rows, n_features, source_ids.copy(), no_splits,
        no_splits, no_splits, flat,
    )
    var values = _ff_values(table, x_colmajor, n_rows, flat)
    var staged = _ff_stage(values, columns, folds, one_hot)
    var tensor_feature = n_features

    # ---- 3. the stat planes and the scale ----
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var grad_mag = Float32(0.0)
    var weight_mag = Float32(0.0)
    for r in range(n_rows):
        var weight = Float32(1.0)
        var weighted_target = weight * y[r]
        if not isfinite(weighted_target):
            raise Error("two-level tensor fit weighted target is not finite")
        stats[r] = weight
        stats[n_rows + r] = weighted_target
        weight_mag += weight
        grad_mag += -weighted_target if weighted_target < Float32(0.0) else weighted_target
    if not isfinite(weight_mag) or not isfinite(grad_mag):
        raise Error("two-level tensor fit weighted statistics overflow")
    var magnitude = Float64(weight_mag)
    if magnitude < 0.0:
        magnitude = -magnitude
    var gradient = Float64(grad_mag)
    if gradient < 0.0:
        gradient = -gradient
    if gradient > magnitude:
        magnitude = gradient
    var fixed_scale = Float32(choose_scale(magnitude, n_rows))

    # ---- 4. the two synchronized levels ----
    var row_index = List[Int](length=n_rows, fill=0)
    for r in range(n_rows):
        row_index[r] = r
    var p_off = List[Int](length=4, fill=0)
    var p_sz = List[Int](length=4, fill=0)
    p_sz[0] = n_rows
    var splits = List[_FFWinner]()
    var layout = staged.compressed.layout.copy()
    _ff_level(
        layout, staged.compressed.words, n_rows, 0, row_index, stats, p_off,
        p_sz, fixed_scale, l2_leaf_reg, splits,
    )
    if splits[0].feature == tensor_feature:
        _refuse_ff("a level-one winner on the FeatureFreq tensor column")

    # `stage_next_feature_freq_after_winner`: the flat columns extended by
    # the level-one candidate's bins, the one-split history table
    var extended = flat.copy()
    for r in range(n_rows):
        extended.append(staged.bins[r])
    var s_feature = List[Int]()
    var s_bin = List[Int]()
    var s_type = List[Int]()
    s_feature.append(splits[0].feature)
    s_bin.append(splits[0].bin)
    s_type.append(splits[0].split_type)
    var table2 = _ff_table(
        x_colmajor, n_rows, n_features, source_ids.copy(), s_feature, s_bin,
        s_type, extended,
    )
    var values2 = _ff_values(table2, x_colmajor, n_rows, extended)
    var staged2 = _ff_stage(values2, columns, folds, one_hot)
    if staged2.compressed.layout.hist_cells != layout.hist_cells:
        raise Error("generated level-two tensor changed pinned layout")
    _ff_level(
        staged2.compressed.layout, staged2.compressed.words, n_rows, 1,
        row_index, stats, p_off, p_sz, fixed_scale, l2_leaf_reg, splits,
    )
    if splits[1].feature == tensor_feature:
        _refuse_ff("a level-two winner on the FeatureFreq tensor column")

    # ---- 5. the leaves (`two_level_weighted_leaf_value`) ----
    var l2 = l2_leaf_reg
    comptime if GBDT_ORACLE_HOST_SABOTAGE:
        l2 = l2 + Float32(1.0)
    var leaves = List[Float32]()
    for leaf in range(4):
        var total = Float32(0.0)
        var total_weight = Float32(0.0)
        for i in range(p_sz[leaf]):
            var row = row_index[p_off[leaf] + i]
            var weight = Float32(1.0)
            total += weight * y[row]
            total_weight += weight
        if total_weight > Float32(0.0):
            leaves.append(learning_rate * total / (total_weight + l2))
        else:
            leaves.append(Float32(0.0))

    # ---- 6. the model text ----
    var out = String("")
    out += "# mojolearn model. One record per line, keyword first.\n"
    out += "# Every float is <decimal>/<IEEE-754 bits in hex>; the BITS are\n"
    out += "# what is loaded, because this toolchain's decimal formatter\n"
    out += "# loses one ULP on ~0.46% of float32 values (measured).\n"
    out += "# Format and CTR seam: gbdt/models/model_text.mojo.\n"
    out += String("format ") + String("mojolearn-model") + " " + String(2) + "\n"
    out += String("features ") + String(n_features) + " " + String(n_features) + "\n"
    out += String("trees ") + String(1) + "\n"
    out += String("losses ") + String(0) + "\n"
    for f in range(n_features):
        var kind = String("cat") if one_hot[f] else String("float")
        var line = (
            String("feature ") + String(f) + " folds " + String(folds[f])
            + " one_hot " + String(1 if one_hot[f] else 0)
            + " type " + kind + " nan as_is borders " + String(len(borders[f]))
        )
        for b in range(len(borders[f])):
            line += " " + gbdt_f32_token(borders[f][b])
        out += line + "\n"
    out += String("tree 0 depth 2 dim 1 weights 0\n")
    for level in range(2):
        var line = (
            String("split 0 ") + String(level) + " "
            + String(splits[level].feature) + " " + String(splits[level].bin)
        )
        if splits[level].split_type == GBDT_FF_TAKE_BIN:
            line += " split_type take_bin"
        out += line + "\n"
    for i in range(4):
        out += String("leaf 0 ") + String(i) + " " + gbdt_f32_token(leaves[i]) + "\n"
    return out^
