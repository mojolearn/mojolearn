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
from gbdt.gpu_data.compressed_index_builder import (
    HostCompressedIndex,
    pack_quantized_columns_host,
)
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
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
        out += " prior_bits " + String(bitcast[DType.uint32](self.prior_num))
        out += " " + String(bitcast[DType.uint32](self.prior_denom))
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


struct TStagedTensorCandidate(Copyable, Movable):
    """Ephemeral candidate registered in a structure-search layout."""

    var feature_id: Int
    var candidate: TTensorCtrCandidate
    var compressed: HostCompressedIndex

    def __init__(
        out self,
        feature_id: Int,
        var candidate: TTensorCtrCandidate,
        var compressed: HostCompressedIndex,
    ):
        self.feature_id = feature_id
        self.candidate = candidate^
        self.compressed = compressed^


def stage_tensor_candidate_host(
    base_columns: List[List[UInt32]],
    base_fold_counts: List[Int],
    base_one_hot: List[Bool],
    var candidate: TTensorCtrCandidate,
    fold_capacity: Int = 0,
) raises -> TStagedTensorCandidate:
    """Assign a dynamic tensor the next feature id and pack it for ranking.

    A positive `fold_capacity` pins the dynamic feature's search layout
    across per-level regeneration. Empty folds carry zero statistics, while
    the real candidate bins remain bounded by its learned borders. This is
    the layout invariant needed by the synchronized tensor-CTR driver: it
    can replace column bits without reallocating histogram/workspace state.
    """
    if len(base_columns) != len(base_fold_counts):
        raise Error("tensor candidate base columns/folds mismatch")
    if len(base_one_hot) != 0 and len(base_one_hot) != len(base_columns):
        raise Error("tensor candidate base one-hot flags mismatch")
    if candidate.tensor_hash != candidate.table.tensor_hash:
        raise Error("tensor candidate identity disagrees with its apply table")
    if len(candidate.borders) == 0:
        raise Error("tensor candidate has no rankable border")
    var dynamic_folds = len(candidate.borders)
    if fold_capacity > 0:
        if fold_capacity < dynamic_folds:
            raise Error("tensor candidate exceeds its pinned fold capacity")
        if fold_capacity > 255:
            raise Error("tensor candidate fold capacity exceeds UInt8 storage")
        dynamic_folds = fold_capacity
    var feature_id = len(base_columns)
    var columns = List[List[UInt32]]()
    for i in range(len(base_columns)):
        columns.append(base_columns[i].copy())
    columns.append(candidate.bins.copy())
    var folds = base_fold_counts.copy()
    folds.append(dynamic_folds)
    var one_hot = List[Bool]()
    if len(base_one_hot) != 0:
        one_hot = base_one_hot.copy()
        one_hot.append(False)
    var compressed = pack_quantized_columns_host(columns^, folds^, one_hot^)
    if compressed.layout.features[feature_id].one_hot_feature:
        raise Error("a tensor CTR was registered as a one-hot feature")
    return TStagedTensorCandidate(feature_id, candidate^, compressed^)


def insert_staged_tensor_candidate_device(
    ctx: DeviceContext,
    staged: TStagedTensorCandidate,
    base_words: List[UInt32],
) raises -> DeviceBuffer[DType.uint32]:
    """Insert one staged tensor into a base index already in staged layout.

    This is the exact launch the symmetric searcher needs after repacking its
    standing columns under the extended layout. Candidate bits must be zero;
    the kernel ORs them in and preserves every neighbouring packed feature.
    The staging buffers are drained here because they are local owners.
    """
    if len(base_words) != len(staged.compressed.words):
        raise Error("staged tensor base compressed-index size mismatch")
    ref cf = staged.compressed.layout.features[staged.feature_id]
    var shifted_mask = cf.mask << cf.shift
    var n_rows = len(staged.candidate.bins)
    for r in range(n_rows):
        var at = Int(cf.offset) * n_rows + r
        if (base_words[at] & shifted_mask) != UInt32(0):
            raise Error("staged tensor destination bits are not zero")
        if staged.candidate.bins[r] > UInt32(255):
            raise Error("staged tensor bin exceeds UInt8 writer input")
    var h_words = ctx.enqueue_create_host_buffer[DType.uint32](len(base_words))
    for i in range(len(base_words)):
        h_words.unsafe_ptr().unsafe_store(i, base_words[i])
    var d_words = ctx.enqueue_create_buffer[DType.uint32](len(base_words))
    ctx.enqueue_copy(dst_buf=d_words, src_ptr=h_words.unsafe_ptr())
    var h_bins = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    for r in range(n_rows):
        h_bins.unsafe_ptr().unsafe_store(r, UInt8(staged.candidate.bins[r]))
    var d_bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    ctx.enqueue_copy(dst_buf=d_bins, src_ptr=h_bins.unsafe_ptr())
    ctx.enqueue_function[write_compressed_index_kernel](
        Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
        d_bins.unsafe_ptr(), Int32(n_rows), d_words.unsafe_ptr(),
        grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
        block_dim=WRITE_BLOCK_SIZE,
    )
    ctx.synchronize()
    _ = h_words^
    _ = h_bins^
    _ = d_bins^
    return d_words^


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
    if String(t[p]) != "prior_bits":
        raise Error("expected tensor prior bits")
    p += 1
    var pn_word = Int(String(t[p]))
    p += 1
    var pd_word = Int(String(t[p]))
    p += 1
    if pn_word < 0 or pn_word > 4294967295 or pd_word < 0 or (
        pd_word > 4294967295
    ):
        raise Error("tensor prior bit pattern exceeds UInt32")
    var pn = bitcast[DType.float32](UInt32(pn_word))
    var pd = bitcast[DType.float32](UInt32(pd_word))
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

    def expand_for_model_apply(
        self,
        x_raw: List[Float32],
        n_rows: Int,
        model_borders: List[List[Float32]],
        one_hot: List[Bool] = List[Bool](),
    ) raises -> List[Float32]:
        """Reconstruct tensor columns and their split-history bins in order.

        Split tensors may name an earlier tensor model column, so apply is
        necessarily sequential. Each completed column is quantized against
        the model's own borders before the next table is evaluated. This is
        the host counterpart of the device cindex ultimately used by tree
        prediction; it does not invent a second grid.
        """
        var n_raw_features = self.first_model_column
        var n_model_features = n_raw_features + len(self.features)
        if len(model_borders) != n_model_features:
            raise Error("tensor CTR registry/model border count mismatch")
        if len(one_hot) != 0 and len(one_hot) != n_model_features:
            raise Error("tensor CTR registry/model one-hot count mismatch")
        if len(x_raw) != n_rows * n_raw_features:
            raise Error("tensor CTR model apply raw shape mismatch")

        var expanded = x_raw.copy()
        var bins = List[UInt32]()
        bins.resize(n_rows * n_model_features, UInt32(0))
        for f in range(n_raw_features):
            var categorical = len(one_hot) != 0 and one_hot[f]
            for r in range(n_rows):
                var value = x_raw[f * n_rows + r]
                if value != value:
                    raise Error("tensor CTR split history cannot quantize NaN")
                var bin = 0
                if categorical:
                    bin = dense_category_code(value, f, r)
                else:
                    for b in range(len(model_borders[f])):
                        if value > model_borders[f][b]:
                            bin += 1
                bins[f * n_rows + r] = UInt32(bin)

        for i in range(len(self.features)):
            ref feature = self.features[i]
            var column = n_raw_features + i
            if feature.model_column != column:
                raise Error("tensor CTR registry model columns are not contiguous")
            for r in range(n_rows):
                var value: Float32
                if len(feature.table.splits) == 0:
                    value = feature.table.value_for_row(x_raw, n_rows, r)
                else:
                    value = value_for_split_tensor_row(
                        feature.table, x_raw, bins, n_rows, r
                    )
                expanded.append(value)
                var bin = 0
                for b in range(len(model_borders[column])):
                    if value > model_borders[column][b]:
                        bin += 1
                bins[column * n_rows + r] = UInt32(bin)
        return expanded^


def persist_winning_tensor_candidate(
    staged: TStagedTensorCandidate, mut registry: TTensorCtrRegistry
) raises -> Int:
    """Move only a selected search candidate into model apply state."""
    if staged.feature_id != registry.first_model_column + len(
        registry.features
    ):
        raise Error(
            "winning tensor feature id is not the registry's next model column"
        )
    return registry.register(staged.candidate.table.copy())


def persist_ranked_tensor_winners(
    ranked_splits: List[TBinarySplit],
    staged_candidates: List[TStagedTensorCandidate],
    mut registry: TTensorCtrRegistry,
) raises -> List[Int]:
    """Persist tensor tables selected by the real structure-search output."""
    # One feature id must identify one candidate. Search batches that reuse
    # an id are valid only serially; handing such a batch here loses which
    # table the score belonged to and is therefore refused.
    for i in range(len(staged_candidates)):
        for j in range(i):
            if staged_candidates[i].feature_id == staged_candidates[j].feature_id:
                raise Error("ranked tensor candidate feature ids are ambiguous")
    var columns = List[Int]()
    for level in range(len(ranked_splits)):
        var feature_id = Int(ranked_splits[level].feature_id)
        for i in range(len(staged_candidates)):
            if staged_candidates[i].feature_id != feature_id:
                continue
            var column = persist_winning_tensor_candidate(
                staged_candidates[i], registry
            )
            columns.append(column)
            break
    return columns^


def next_tensor_base_from_winner(
    staged: TStagedTensorCandidate, winning_split: TBinarySplit
) raises -> TFeatureTensor:
    """Canonical tensor tracker state used to generate the next level."""
    if Int(winning_split.feature_id) != staged.feature_id:
        raise Error("winning split does not name the staged tensor candidate")
    var tensor = TFeatureTensor()
    for i in range(len(staged.candidate.table.source_features)):
        tensor.add_cat_feature(UInt32(
            staged.candidate.table.source_features[i]
        ))
    tensor.add_binary_splits(staged.candidate.table.splits.copy())
    var before = tensor.get_hash()
    tensor.add_binary_split(winning_split)
    if tensor.get_hash() == before:
        raise Error("winning split did not advance tensor tracker state")
    return tensor^


def regenerate_feature_freq_after_winner(
    staged: TStagedTensorCandidate,
    winning_split: TBinarySplit,
    x_colmajor: List[Float32],
    extended_cindex: List[UInt32],
    n_rows: Int,
    n_features: Int,
    grid: TBinarizationOptions,
) raises -> TTensorCtrCandidate:
    """Build the next-level FeatureFreq candidate from an actual winner.

    `extended_cindex` must include the ephemeral winning tensor column. This
    makes regeneration consume precisely the bins that structure search
    ranked, rather than reconstructing a similar-looking split off-path.
    """
    var tensor = next_tensor_base_from_winner(staged, winning_split)
    var sources = List[Int]()
    for i in range(len(tensor.cat_features)):
        sources.append(Int(tensor.cat_features[i]))
    var table = build_split_feature_freq_tensor_table(
        x_colmajor, extended_cindex, n_rows, n_features,
        sources^, tensor.splits.copy(),
        staged.candidate.table.prior_num,
        staged.candidate.table.prior_denom,
    )
    var values = List[Float32]()
    values.resize(n_rows, Float32(0.0))
    for row in range(n_rows):
        values[row] = value_for_split_tensor_row(
            table, x_colmajor, extended_cindex, n_rows, row
        )
    return materialize_tensor_candidate(table^, values^, grid)
