# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The apply half of a tensor (combination) FeatureFreq or Borders CTR,
with no device import, so a host binding compiles the same code the GPU
predict runs (lane/inference-gbdt-ctr-tables, 2026-09-15).

`gbdt/models/tensor_ctr_value_table.mojo` imports `max.gpu.host`, the
compressed-index writer kernel and the tensor builder, so a binding built
with no accelerator target cannot import it. Everything a SAVED tensor CTR
model needs at predict time is here instead, and that module's
`TFeatureFreqTensorTable.key_for_row`, `value_for_key`, `_split_bit` and
`TTensorCtrRegistry.expand_for_model_apply` call these functions, so the
GPU predict and the forest host binding run one body:

- `tensor_key_for_row`: the mixed-radix key over the dense codes of the
  source features; a code at or past its cardinality is the reference's
  NotFoundIndex, key -1.
- `tensor_value_for_key`: FeatureFreq `(count + prior_num) / (denominator +
  prior_denom)`, a missing key counting zero; Borders the good-over-total
  histogram form, a missing key the prior alone.
- `tensor_split_bit`: one split-history bit off the quantized columns.
- `expand_tensor_ctr_columns`: raw columns in, raw columns followed by the
  tensor columns out, each tensor column quantized against the model's own
  borders before the next table (a split history may name an earlier tensor
  column).

The tensor hash is not read at apply time; the text reader checks it.
"""

from gbdt.models.ctr_value_table import dense_category_code


#: `BIN_SPLIT_TAKE_BIN` and `BIN_SPLIT_TAKE_GREATER`
#: (`gbdt/models/oblivious_model.mojo`), restated because that module is not
#: imported here; `tensor_ctr_value_table.mojo` asserts they agree.
comptime TENSOR_SPLIT_TAKE_BIN = 0
comptime TENSOR_SPLIT_TAKE_GREATER = 1


@fieldwise_init
struct TTensorCtrApplyTable(Copyable, Movable):
    """What one serialized tensor table carries that apply reads: the
    canonical sources and their cardinalities, the split history as three
    parallel lists (feature, bin, split type), the target-class axis, the
    priors, the denominator and the counts."""

    var source_features: List[Int]
    var cardinalities: List[Int]
    var split_features: List[Int]
    var split_bins: List[Int]
    var split_types: List[Int]
    var target_classes_count: Int
    var target_border_idx: Int
    var prior_num: Float32
    var prior_denom: Float32
    var denominator: Int
    var counts: List[Int]


def tensor_key_for_row(
    source_features: List[Int],
    cardinalities: List[Int],
    x_colmajor: List[Float32],
    n_rows: Int,
    row: Int,
) raises -> Int:
    var key = 0
    for i in range(len(source_features)):
        var f = source_features[i]
        var code = dense_category_code(x_colmajor[f * n_rows + row], f, row)
        if code >= cardinalities[i]:
            return -1  # their NotFoundIndex / empty-value arm
        key = key * cardinalities[i] + code
    return key


def tensor_value_for_key(
    counts: List[Int],
    target_classes_count: Int,
    target_border_idx: Int,
    prior_num: Float32,
    prior_denom: Float32,
    denominator: Int,
    key: Int,
) raises -> Float32:
    if target_classes_count == 0:
        var count = 0
        if key >= 0 and key < len(counts):
            count = counts[key]
        return (Float32(count) + prior_num) / (
            Float32(denominator) + prior_denom
        )
    if target_classes_count < 2 or target_border_idx < 0 or (
        target_border_idx >= target_classes_count - 1
    ):
        raise Error("invalid Borders tensor target-class metadata")
    if key < 0:
        return prior_num / prior_denom
    var off = key * target_classes_count
    if off + target_classes_count > len(counts):
        return prior_num / prior_denom
    var total = 0
    var good = 0
    for cls in range(target_classes_count):
        var count = counts[off + cls]
        total += count
        if cls > target_border_idx:
            good += count
    return (Float32(good) + prior_num) / (Float32(total) + prior_denom)


def tensor_split_bit(
    feature: Int, bin: Int, split_type: Int, cindex: List[UInt32], n_rows: Int,
    row: Int,
) raises -> Int:
    if feature < 0 or bin < 0 or feature * n_rows + row >= len(cindex):
        raise Error("split-history tensor references an invalid quantized column")
    var value = Int(cindex[feature * n_rows + row])
    if split_type == TENSOR_SPLIT_TAKE_BIN:
        return 1 if value == bin else 0
    if split_type == TENSOR_SPLIT_TAKE_GREATER:
        return 1 if value > bin else 0
    raise Error("split-history tensor has an unknown split type")


def tensor_value_for_row(
    table: TTensorCtrApplyTable,
    x_colmajor: List[Float32],
    cindex: List[UInt32],
    n_rows: Int,
    row: Int,
) raises -> Float32:
    """`value_for_row` for a table with no split history, and
    `value_for_split_tensor_row` for one with it."""
    var key = tensor_key_for_row(
        table.source_features, table.cardinalities, x_colmajor, n_rows, row
    )
    if len(table.split_features) != 0 and key >= 0:
        for i in range(len(table.split_features)):
            key = 2 * key + tensor_split_bit(
                table.split_features[i], table.split_bins[i],
                table.split_types[i], cindex, n_rows, row,
            )
    return tensor_value_for_key(
        table.counts, table.target_classes_count, table.target_border_idx,
        table.prior_num, table.prior_denom, table.denominator, key,
    )


def expand_tensor_ctr_columns(
    tables: List[TTensorCtrApplyTable],
    x_raw: List[Float32],
    n_rows: Int,
    model_borders: List[List[Float32]],
    one_hot: List[Bool],
) raises -> List[Float32]:
    """Reconstruct tensor columns and their split-history bins in order.

    The tables are the model's tensor columns in model-column order, the
    first at `len(model_borders) - len(tables)`, which is the raw input
    column count. Split tensors may name an earlier tensor model column, so
    apply is necessarily sequential. Each completed column is quantized
    against the model's own borders before the next table is evaluated.
    This is the host counterpart of the device cindex ultimately used by
    tree prediction; it does not invent a second grid.
    """
    var n_model_features = len(model_borders)
    var n_raw_features = n_model_features - len(tables)
    if n_raw_features < 0:
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

    for i in range(len(tables)):
        ref table = tables[i]
        var column = n_raw_features + i
        for r in range(n_rows):
            var value = tensor_value_for_row(table, x_raw, bins, n_rows, r)
            expanded.append(value)
            var bin = 0
            for b in range(len(model_borders[column])):
                if value > model_borders[column][b]:
                    bin += 1
            bins[column * n_rows + r] = UInt32(bin)
    return expanded^
