# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Executable first slice of combination CTR storage and application.

This implements permutation-independent FeatureFreq over a tensor of two or
more dense categorical features.  The tensor identity uses the canonical
`TFeatureTensor` hash; row keys use deterministic mixed-radix packing.  The
same source-feature order and radices are serialized with the counts, so the
apply side cannot guess a training-time projection.

This is intentionally not admitted by `train(max_ctr_complexity > 1)` yet.
That surface also promises history-dependent split tensors and Borders CTRs;
its refusal remains until the structure searcher can mint those tensors.
"""

from gbdt.methods.batch_feature_tensor_builder import TFeatureTensor
from gbdt.models.ctr_value_table import dense_category_code
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)
from gbdt.ctrs.ctr_binarization import (
    TBinarizationOptions,
    compute_ctr_borders,
)


def split_tensor_hash(hash: UInt64) -> Tuple[UInt32, UInt32]:
    """Unsigned halves for a text format whose integer parser is signed."""
    return (
        (hash >> 32).cast[DType.uint32](),
        hash.cast[DType.uint32](),
    )


def join_tensor_hash(hi_text: String, lo_text: String) raises -> UInt64:
    """Parse two decimal u32 words without routing through signed Int64 hash."""
    var hi = Int(hi_text)
    var lo = Int(lo_text)
    if hi < 0 or hi > 4294967295 or lo < 0 or lo > 4294967295:
        raise Error("tensor hash halves must each fit UInt32")
    return (UInt64(from_int=hi) << 32) | UInt64(from_int=lo)


struct TFeatureFreqTensorTable(Copyable, Movable):
    var tensor_hash: UInt64
    var source_features: List[Int]
    var cardinalities: List[Int]
    var splits: List[TBinarySplit]
    var target_classes_count: Int
    var target_border_idx: Int
    var prior_num: Float32
    var prior_denom: Float32
    var denominator: Int
    var counts: List[Int]

    def __init__(
        out self,
        tensor_hash: UInt64,
        var source_features: List[Int],
        var cardinalities: List[Int],
        var splits: List[TBinarySplit],
        target_classes_count: Int,
        target_border_idx: Int,
        prior_num: Float32,
        prior_denom: Float32,
        denominator: Int,
        var counts: List[Int],
    ):
        self.tensor_hash = tensor_hash
        self.source_features = source_features^
        self.cardinalities = cardinalities^
        self.splits = splits^
        self.target_classes_count = target_classes_count
        self.target_border_idx = target_border_idx
        self.prior_num = prior_num
        self.prior_denom = prior_denom
        self.denominator = denominator
        self.counts = counts^

    def key_for_row(
        self, x_colmajor: List[Float32], n_rows: Int, row: Int
    ) raises -> Int:
        var key = 0
        for i in range(len(self.source_features)):
            var f = self.source_features[i]
            var code = dense_category_code(x_colmajor[f * n_rows + row], f, row)
            if code >= self.cardinalities[i]:
                return -1  # their NotFoundIndex / empty-value arm
            key = key * self.cardinalities[i] + code
        return key

    def value_for_row(
        self, x_colmajor: List[Float32], n_rows: Int, row: Int
    ) raises -> Float32:
        if len(self.splits) != 0:
            raise Error("split-history tensor apply requires quantized columns")
        var key = self.key_for_row(x_colmajor, n_rows, row)
        return self.value_for_key(key)

    def value_for_key(self, key: Int) raises -> Float32:
        if self.target_classes_count == 0:
            var count = 0
            if key >= 0 and key < len(self.counts):
                count = self.counts[key]
            return (Float32(count) + self.prior_num) / (
                Float32(self.denominator) + self.prior_denom
            )
        if self.target_classes_count < 2 or self.target_border_idx < 0 or (
            self.target_border_idx >= self.target_classes_count - 1
        ):
            raise Error("invalid Borders tensor target-class metadata")
        if key < 0:
            return self.prior_num / self.prior_denom
        var off = key * self.target_classes_count
        if off + self.target_classes_count > len(self.counts):
            return self.prior_num / self.prior_denom
        var total = 0
        var good = 0
        for cls in range(self.target_classes_count):
            var count = self.counts[off + cls]
            total += count
            if cls > self.target_border_idx:
                good += count
        return (Float32(good) + self.prior_num) / (
            Float32(total) + self.prior_denom
        )

    def to_text(self) -> String:
        """Canonical one-line model record; counts follow tensor metadata."""
        var hash_words = split_tensor_hash(self.tensor_hash)
        var out = String("feature_freq_tensor 2 hash_hi ") + String(hash_words[0])
        out += " hash_lo " + String(hash_words[1])
        out += " sources " + String(len(self.source_features))
        for i in range(len(self.source_features)):
            out += " " + String(self.source_features[i])
        out += " cardinalities " + String(len(self.cardinalities))
        for i in range(len(self.cardinalities)):
            out += " " + String(self.cardinalities[i])
        out += " splits " + String(len(self.splits))
        for i in range(len(self.splits)):
            out += " " + String(Int(self.splits[i].feature_id))
            out += " " + String(Int(self.splits[i].bin_idx))
            out += " " + String(Int(self.splits[i].split_type))
        out += " classes " + String(self.target_classes_count)
        out += " target_border " + String(self.target_border_idx)
        out += " prior " + String(self.prior_num) + " " + String(self.prior_denom)
        out += " denominator " + String(self.denominator)
        out += " counts " + String(len(self.counts))
        for i in range(len(self.counts)):
            out += " " + String(self.counts[i])
        return out


def build_feature_freq_tensor_table(
    x_colmajor: List[Float32],
    n_rows: Int,
    n_features: Int,
    var source_features: List[Int],
    prior_num: Float32 = Float32(0.0),
    prior_denom: Float32 = Float32(1.0),
) raises -> TFeatureFreqTensorTable:
    """Fit a deterministic dense combination-frequency table."""
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    if n_rows <= 0:
        raise Error("a combination CTR needs at least one training row")
    if len(source_features) < 2:
        raise Error("a combination CTR needs at least two source features")
    # `TFeatureTensor` canonicalizes categorical ids. Canonicalize the row
    # key in the same order, or [1, 0] and [0, 1] would share a tensor hash
    # while assigning different mixed-radix keys.
    for i in range(1, len(source_features)):
        var j = i
        while j > 0 and source_features[j] < source_features[j - 1]:
            var tmp = source_features[j]
            source_features[j] = source_features[j - 1]
            source_features[j - 1] = tmp
            j -= 1
    var tensor = TFeatureTensor()
    var cards = List[Int]()
    var entry_count = 1
    for i in range(len(source_features)):
        var f = source_features[i]
        if f < 0 or f >= n_features:
            raise Error("combination CTR source feature is out of range")
        if i > 0 and source_features[i - 1] == f:
            raise Error("combination CTR source features must be unique")
        tensor.add_cat_feature(UInt32(f))
        var max_code = 0
        var seen = List[Bool]()
        seen.resize(1, False)
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

    var counts = List[Int]()
    counts.resize(entry_count, 0)
    for r in range(n_rows):
        var key = 0
        for i in range(len(source_features)):
            var f = source_features[i]
            var code = dense_category_code(x_colmajor[f * n_rows + r], f, r)
            key = key * cards[i] + code
        counts[key] += 1
    return TFeatureFreqTensorTable(
        tensor.get_hash(), source_features^, cards^, List[TBinarySplit](), 0, -1,
        prior_num, prior_denom, n_rows, counts^,
    )


def _split_bit(
    split: TBinarySplit, cindex: List[UInt32], n_rows: Int, row: Int
) raises -> Int:
    var feature = Int(split.feature_id)
    var bin = Int(split.bin_idx)
    if feature < 0 or bin < 0 or feature * n_rows + row >= len(cindex):
        raise Error("split-history tensor references an invalid quantized column")
    var value = Int(cindex[feature * n_rows + row])
    if Int(split.split_type) == BIN_SPLIT_TAKE_BIN:
        return 1 if value == bin else 0
    if Int(split.split_type) == BIN_SPLIT_TAKE_GREATER:
        return 1 if value > bin else 0
    raise Error("split-history tensor has an unknown split type")


def build_split_feature_freq_tensor_table(
    x_colmajor: List[Float32],
    cindex: List[UInt32],
    n_rows: Int,
    n_features: Int,
    var source_features: List[Int],
    var splits: List[TBinarySplit],
    prior_num: Float32 = Float32(0.0),
    prior_denom: Float32 = Float32(1.0),
) raises -> TFeatureFreqTensorTable:
    """Fit FeatureFreq on categorical ids crossed with split history."""
    if len(splits) == 0:
        raise Error("split-history tensor needs at least one binary split")
    var base = build_feature_freq_tensor_table(
        x_colmajor, n_rows, n_features, source_features^,
        prior_num, prior_denom,
    )
    var tensor = TFeatureTensor()
    for i in range(len(base.source_features)):
        tensor.add_cat_feature(UInt32(base.source_features[i]))
    tensor.add_binary_splits(splits^)
    var canonical_splits = tensor.get_splits()
    if len(canonical_splits) > 30:
        raise Error("split-history tensor supports at most 30 unique splits")
    var entries = len(base.counts)
    for _ in range(len(canonical_splits)):
        if entries > 10000000 // 2:
            raise Error("split-history tensor exceeds 10,000,000 entries")
        entries *= 2
    var counts = List[Int]()
    counts.resize(entries, 0)
    for row in range(n_rows):
        var key = base.key_for_row(x_colmajor, n_rows, row)
        for i in range(len(canonical_splits)):
            key = 2 * key + _split_bit(
                canonical_splits[i], cindex, n_rows, row
            )
        counts[key] += 1
    return TFeatureFreqTensorTable(
        tensor.get_hash(), base.source_features.copy(),
        base.cardinalities.copy(), canonical_splits^, 0, -1,
        prior_num, prior_denom, n_rows, counts^,
    )


def value_for_split_tensor_row(
    table: TFeatureFreqTensorTable,
    x_colmajor: List[Float32],
    cindex: List[UInt32],
    n_rows: Int,
    row: Int,
) raises -> Float32:
    """Apply a serialized split-history tensor to one quantized row."""
    var key = table.key_for_row(x_colmajor, n_rows, row)
    if key < 0:
        return table.value_for_key(-1)
    for i in range(len(table.splits)):
        key = 2 * key + _split_bit(table.splits[i], cindex, n_rows, row)
    return table.value_for_key(key)


struct TBordersSplitTensorFit(Copyable, Movable):
    """Ordered learn column plus the full-pool table used by model apply."""

    var learn_values: List[Float32]
    var table: TFeatureFreqTensorTable

    def __init__(
        out self, var learn_values: List[Float32],
        var table: TFeatureFreqTensorTable,
    ):
        self.learn_values = learn_values^
        self.table = table^


struct TTensorCtrCandidate(Copyable, Movable):
    """A dynamic tensor column at the compressed-index/ranking boundary.

    `values` are the learn statistic (ordered for Borders), `borders` are
    selected with the CTR description's own grid, and `bins` are exactly
    what the existing compressed-index builder and histogram ranker consume.
    Keeping the table beside them binds model apply to the candidate identity.
    """

    var tensor_hash: UInt64
    var table: TFeatureFreqTensorTable
    var values: List[Float32]
    var borders: List[Float32]
    var bins: List[UInt32]

    def __init__(
        out self,
        tensor_hash: UInt64,
        var table: TFeatureFreqTensorTable,
        var values: List[Float32],
        var borders: List[Float32],
        var bins: List[UInt32],
    ):
        self.tensor_hash = tensor_hash
        self.table = table^
        self.values = values^
        self.borders = borders^
        self.bins = bins^


def materialize_tensor_candidate(
    var table: TFeatureFreqTensorTable,
    var learn_values: List[Float32],
    grid: TBinarizationOptions,
) raises -> TTensorCtrCandidate:
    """Turn a fitted tensor statistic into a rankable quantized feature."""
    if len(learn_values) == 0:
        raise Error("cannot rank an empty tensor CTR column")
    var borders = compute_ctr_borders(learn_values, grid)
    if len(borders) > 255:
        raise Error("tensor CTR candidate exceeds one-byte fold capacity")
    var bins = List[UInt32]()
    bins.resize(len(learn_values), UInt32(0))
    for r in range(len(learn_values)):
        var bin = 0
        # Same strict comparison as `binarize_float_feature_kernel`: the
        # bin is the number of borders strictly below the feature value.
        for b in range(len(borders)):
            if learn_values[r] > borders[b]:
                bin += 1
        bins[r] = UInt32(bin)
    var hash = table.tensor_hash
    return TTensorCtrCandidate(
        hash, table^, learn_values^, borders^, bins^
    )


def build_borders_split_tensor_table(
    x_colmajor: List[Float32],
    cindex: List[UInt32],
    binarized_target: List[UInt8],
    order: List[UInt32],
    n_rows: Int,
    n_features: Int,
    var source_features: List[Int],
    var splits: List[TBinarySplit],
    target_classes_count: Int,
    target_border_idx: Int,
    prior_num: Float32,
    prior_denom: Float32,
) raises -> TBordersSplitTensorFit:
    """Fit an ordered Borders CTR and its full-pool apply histogram."""
    if len(binarized_target) != n_rows or len(order) != n_rows:
        raise Error("Borders tensor target/order size mismatch")
    if target_classes_count < 2 or target_classes_count > 256 or (
        target_border_idx < 0 or target_border_idx >= target_classes_count - 1
    ):
        raise Error("invalid Borders tensor target-class metadata")
    if prior_denom == Float32(0.0):
        raise Error("Borders tensor prior denominator must be non-zero")
    var shape = build_split_feature_freq_tensor_table(
        x_colmajor, cindex, n_rows, n_features,
        source_features^, splits^, prior_num, prior_denom,
    )
    var key_count = len(shape.counts)
    if key_count > 10000000 // target_classes_count:
        raise Error("Borders tensor histogram exceeds 10,000,000 counts")
    var histogram = List[Int]()
    histogram.resize(key_count * target_classes_count, 0)
    var prefix = List[Int]()
    prefix.resize(key_count * target_classes_count, 0)
    var learn = List[Float32]()
    learn.resize(n_rows, Float32(0.0))
    var seen_rows = List[Bool]()
    seen_rows.resize(n_rows, False)
    for pos in range(n_rows):
        var row = Int(order[pos])
        if row < 0 or row >= n_rows or seen_rows[row]:
            raise Error("Borders tensor order is not a permutation")
        seen_rows[row] = True
        var key = shape.key_for_row(x_colmajor, n_rows, row)
        for i in range(len(shape.splits)):
            key = 2 * key + _split_bit(shape.splits[i], cindex, n_rows, row)
        var cls = Int(binarized_target[row])
        if cls < 0 or cls >= target_classes_count:
            raise Error("Borders tensor target class is out of range")
        var total = 0
        var good = 0
        for k in range(target_classes_count):
            var count = prefix[key * target_classes_count + k]
            total += count
            if k > target_border_idx:
                good += count
        learn[row] = (Float32(good) + prior_num) / (
            Float32(total) + prior_denom
        )
        prefix[key * target_classes_count + cls] += 1
        histogram[key * target_classes_count + cls] += 1
    var table = TFeatureFreqTensorTable(
        shape.tensor_hash, shape.source_features.copy(),
        shape.cardinalities.copy(), shape.splits.copy(),
        target_classes_count, target_border_idx,
        prior_num, prior_denom, 0, histogram^,
    )
    return TBordersSplitTensorFit(learn^, table^)


def parse_feature_freq_tensor_table(line: String) raises -> TFeatureFreqTensorTable:
    """Read the exact record emitted by `to_text`, refusing drift."""
    var t = line.split(" ")
    var p = 0
    if String(t[p]) != "feature_freq_tensor":
        raise Error("expected feature_freq_tensor record")
    p += 1
    if Int(String(t[p])) != 2:
        raise Error("unsupported feature_freq_tensor version")
    p += 1
    if String(t[p]) != "hash_hi":
        raise Error("expected tensor hash_hi")
    p += 1
    var hash_hi = String(t[p])
    p += 1
    if String(t[p]) != "hash_lo":
        raise Error("expected tensor hash_lo")
    p += 1
    var hash_lo = String(t[p])
    p += 1
    var hash = join_tensor_hash(hash_hi, hash_lo)
    if String(t[p]) != "sources":
        raise Error("expected tensor sources")
    p += 1
    var ns = Int(String(t[p]))
    p += 1
    var sources = List[Int]()
    for _ in range(ns):
        sources.append(Int(String(t[p])))
        p += 1
    if String(t[p]) != "cardinalities":
        raise Error("expected tensor cardinalities")
    p += 1
    var nc = Int(String(t[p]))
    p += 1
    if nc != ns:
        raise Error("tensor source/cardinality count mismatch")
    var cards = List[Int]()
    var product = 1
    for _ in range(nc):
        var card = Int(String(t[p]))
        p += 1
        if card < 1 or product > 10000000 // card:
            raise Error("invalid tensor cardinality product")
        product *= card
        cards.append(card)
    if String(t[p]) != "splits":
        raise Error("expected tensor splits")
    p += 1
    var n_splits = Int(String(t[p]))
    p += 1
    if n_splits < 0 or n_splits > 30:
        raise Error("invalid tensor split count")
    var splits = List[TBinarySplit]()
    for _ in range(n_splits):
        var feature = Int(String(t[p]))
        p += 1
        var bin = Int(String(t[p]))
        p += 1
        var kind = Int(String(t[p]))
        p += 1
        if feature < 0 or feature > 2147483647 or bin < 0 or (
            bin > 2147483647
        ) or (
            kind != BIN_SPLIT_TAKE_BIN and kind != BIN_SPLIT_TAKE_GREATER
        ):
            raise Error("invalid tensor split record")
        splits.append(TBinarySplit(Int32(feature), Int32(bin), Int32(kind)))
        if product > 10000000 // 2:
            raise Error("tensor split table exceeds 10,000,000 entries")
        product *= 2
    if String(t[p]) != "classes":
        raise Error("expected tensor target classes")
    p += 1
    var classes = Int(String(t[p]))
    p += 1
    if classes != 0 and (classes < 2 or classes > 256):
        raise Error("tensor target classes must be zero or at least two")
    if String(t[p]) != "target_border":
        raise Error("expected tensor target border")
    p += 1
    var target_border = Int(String(t[p]))
    p += 1
    if (classes == 0 and target_border != -1) or (
        classes > 0 and (target_border < 0 or target_border >= classes - 1)
    ):
        raise Error("invalid tensor target border")
    if String(t[p]) != "prior":
        raise Error("expected tensor prior")
    p += 1
    var pn = Float32(Float64(String(t[p])))
    p += 1
    var pd = Float32(Float64(String(t[p])))
    p += 1
    if classes > 0 and pd == Float32(0.0):
        raise Error("Borders tensor prior denominator must be non-zero")
    if String(t[p]) != "denominator":
        raise Error("expected tensor denominator")
    p += 1
    var denom = Int(String(t[p]))
    p += 1
    if denom < 0:
        raise Error("tensor denominator must be non-negative")
    if String(t[p]) != "counts":
        raise Error("expected tensor counts")
    p += 1
    var ncounts = Int(String(t[p]))
    p += 1
    var count_width = classes if classes > 0 else 1
    if product > 10000000 // count_width:
        raise Error("tensor table exceeds 10,000,000 counts")
    var expected_counts = product * count_width
    if ncounts != expected_counts:
        raise Error("tensor count length does not match cardinalities")
    var counts = List[Int]()
    var total = 0
    for _ in range(ncounts):
        var count = Int(String(t[p]))
        p += 1
        if count < 0:
            raise Error("tensor counts must be non-negative")
        total += count
        counts.append(count)
    if classes == 0 and total != denom:
        raise Error("tensor counts do not sum to denominator")
    if classes > 0 and denom != 0:
        raise Error("Borders tensor denominator must be zero")
    if p != len(t):
        raise Error("trailing fields in feature_freq_tensor record")
    var tensor = TFeatureTensor()
    for i in range(len(sources)):
        if sources[i] < 0:
            raise Error("tensor source feature must be non-negative")
        if i > 0 and sources[i - 1] >= sources[i]:
            raise Error("tensor sources are not canonical and unique")
        tensor.add_cat_feature(UInt32(sources[i]))
    tensor.add_binary_splits(splits.copy())
    if tensor.get_hash() != hash:
        raise Error("feature tensor hash does not match its sources")
    if len(tensor.splits) != len(splits):
        raise Error("tensor splits are not canonical and unique")
    for i in range(len(splits)):
        if (
            tensor.splits[i].feature_id != splits[i].feature_id
            or tensor.splits[i].bin_idx != splits[i].bin_idx
            or tensor.splits[i].split_type != splits[i].split_type
        ):
            raise Error("tensor splits are not in canonical order")
    return TFeatureFreqTensorTable(
        hash, sources^, cards^, splits^, classes, target_border,
        pn, pd, denom, counts^
    )


def _same_projection(a: TFeatureFreqTensorTable, b: TFeatureFreqTensorTable) -> Bool:
    if a.tensor_hash != b.tensor_hash:
        return False
    if len(a.source_features) != len(b.source_features):
        return False
    if len(a.cardinalities) != len(b.cardinalities):
        return False
    if len(a.splits) != len(b.splits):
        return False
    for i in range(len(a.source_features)):
        if a.source_features[i] != b.source_features[i]:
            return False
        if a.cardinalities[i] != b.cardinalities[i]:
            return False
    for i in range(len(a.splits)):
        if (
            a.splits[i].feature_id != b.splits[i].feature_id
            or a.splits[i].bin_idx != b.splits[i].bin_idx
            or a.splits[i].split_type != b.splits[i].split_type
        ):
            return False
    return True


def _same_table(a: TFeatureFreqTensorTable, b: TFeatureFreqTensorTable) -> Bool:
    if not _same_projection(a, b):
        return False
    if (
        a.prior_num != b.prior_num
        or a.prior_denom != b.prior_denom
        or a.denominator != b.denominator
        or a.target_classes_count != b.target_classes_count
        or a.target_border_idx != b.target_border_idx
        or len(a.counts) != len(b.counts)
    ):
        return False
    for i in range(len(a.counts)):
        if a.counts[i] != b.counts[i]:
            return False
    return True


struct TTensorCtrFeature(Copyable, Movable):
    """One registered tensor feature and its stable model-column id."""

    var model_column: Int
    var table: TFeatureFreqTensorTable

    def __init__(
        out self, model_column: Int, var table: TFeatureFreqTensorTable
    ):
        self.model_column = model_column
        self.table = table^


struct TTensorCtrRegistry(Copyable, Movable):
    """Internal tensor→feature map and apply-time table cache.

    CatBoost's feature manager keys this map by tensor identity. Hashes only
    select candidates here: canonical source/cardinality metadata resolves a
    collision before an existing model column is reused.
    """

    var first_model_column: Int
    var features: List[TTensorCtrFeature]

    def __init__(out self, first_model_column: Int) raises:
        if first_model_column < 0:
            raise Error("first tensor CTR model column must be non-negative")
        self.first_model_column = first_model_column
        self.features = List[TTensorCtrFeature]()

    def register(
        mut self, var table: TFeatureFreqTensorTable
    ) raises -> Int:
        """Return the stable model column, deduplicating an exact table."""
        for i in range(len(self.features)):
            ref old = self.features[i]
            if old.table.tensor_hash != table.tensor_hash:
                continue
            if not _same_projection(old.table, table):
                # A 64-bit collision must not alias two observable features.
                continue
            if not _same_table(old.table, table):
                raise Error(
                    "the same feature tensor was registered with different"
                    " counts, priors, or denominator"
                )
            return old.model_column
        var column = self.first_model_column + len(self.features)
        self.features.append(TTensorCtrFeature(column, table^))
        return column

    def expand_for_apply(
        self,
        x_raw: List[Float32],
        n_rows: Int,
        n_raw_features: Int,
    ) raises -> List[Float32]:
        """Raw columns followed by registered tensors in model-column order."""
        if self.first_model_column != n_raw_features:
            raise Error(
                "tensor CTR apply plan starts at model column "
                + String(self.first_model_column) + " but carries "
                + String(n_raw_features) + " raw columns"
            )
        if len(x_raw) != n_rows * n_raw_features:
            raise Error("tensor CTR apply raw shape mismatch")
        var out = x_raw.copy()
        for i in range(len(self.features)):
            ref feature = self.features[i]
            if feature.model_column != n_raw_features + i:
                raise Error("tensor CTR registry model columns are not contiguous")
            for r in range(n_rows):
                out.append(feature.table.value_for_row(x_raw, n_rows, r))
        return out^

    def expand_for_apply_with_bins(
        self,
        x_raw: List[Float32],
        cindex: List[UInt32],
        n_rows: Int,
        n_raw_features: Int,
    ) raises -> List[Float32]:
        """Apply both categorical-only and split-history tensor columns."""
        if self.first_model_column != n_raw_features:
            raise Error("tensor CTR apply plan/raw column mismatch")
        if len(x_raw) != n_rows * n_raw_features:
            raise Error("tensor CTR apply raw shape mismatch")
        var out = x_raw.copy()
        for i in range(len(self.features)):
            ref feature = self.features[i]
            if feature.model_column != n_raw_features + i:
                raise Error("tensor CTR registry model columns are not contiguous")
            for r in range(n_rows):
                if len(feature.table.splits) == 0:
                    out.append(feature.table.value_for_row(x_raw, n_rows, r))
                else:
                    out.append(value_for_split_tensor_row(
                        feature.table, x_raw, cindex, n_rows, r
                    ))
        return out^
