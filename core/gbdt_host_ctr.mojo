# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CTR and tensor CTR columns of a saved GBDT model, on the host, for a box
with no GPU (lane/inference-gbdt-ctr-tables, 2026-09-15).

HOST ONLY, and NO RESTATEMENT. `predict_floats` (`gbdt/train.mojo`) turns
raw input columns into model columns before it quantizes: a model with CTR
tables through `expand_raw_columns` (`gbdt/models/ctr_value_table.mojo`, the
reference `TStaticCtrProvider::CalcCtrs` step), a model with tensor CTRs
through `TTensorCtrRegistry.expand_for_model_apply`, whose body is
`expand_tensor_ctr_columns` (`gbdt/models/tensor_ctr_apply.mojo`). This file
rebuilds the same tables from the flat arrays `python/mojolearn/_gbdt_host.py`
parses out of the model text and calls those two functions; the quantize,
walk and leaf sum that follow are `core/gbdt_host_predict.mojo`'s.

Raw categorical values reach both functions as float32 dense codes and go
through `dense_category_code`, the fit's own check (a NaN, a negative, an
oversized or a non-integer code raises by name; an integer the learn pool
never saw takes the table's empty value). This format keys a table by the
dense code, not by a hash of the raw value (deviation 56), so the code IS
what the fit hashed.

THE CTR SABOTAGE ARM. `-D MOJOLEARN_GBDT_CTR_HOST_SABOTAGE=1` rotates every
table's counts by one category (category `c` reads the statistics of
`c + 1`, the last reads the first's), so every looked-up CTR value is some
other category's; a CTR identity cell that does not move under it does not
measure the lookup.
"""
from std.sys.compile import is_defined

from gbdt.models.ctr_value_table import TCtrValueTable, expand_raw_columns
from gbdt.models.tensor_ctr_apply import (
    TTensorCtrApplyTable,
    expand_tensor_ctr_columns,
)


comptime GBDT_CTR_HOST_SABOTAGE = is_defined["MOJOLEARN_GBDT_CTR_HOST_SABOTAGE"]()

#: fields per CTR table in `ctr_ints`: column, source, type, denom, classes,
#: target_border, counts offset, counts length
comptime CTR_INTS_PER_TABLE = 8
#: fields per CTR table in `ctr_floats`: prior_num, prior_denom, shift, scale
comptime CTR_FLOATS_PER_TABLE = 4
#: fields per tensor table in `tensor_floats`: prior_num, prior_denom
comptime TENSOR_FLOATS_PER_TABLE = 2


def _counts(counts: List[Int32], offset: Int, length: Int, width: Int) raises -> List[Int]:
    if offset < 0 or length < 0 or offset + length > len(counts):
        raise Error("gbdt host ctr: a table's counts run past the counts array")
    var out = List[Int](capacity=length)
    for i in range(length):
        out.append(Int(counts[offset + i]))
    comptime if GBDT_CTR_HOST_SABOTAGE:
        var w = width if width > 0 else 1
        if length >= 2 * w and length % w == 0:
            var rotated = List[Int](capacity=length)
            for i in range(length):
                rotated.append(out[(i + w) % length])
            return rotated^
    return out^


def _take(ints: List[Int32], mut p: Int) raises -> Int:
    """The next value of the tensor table stream, bounds-checked."""
    if p < 0 or p >= len(ints):
        raise Error("gbdt host ctr: the tensor table array is short")
    var v = Int(ints[p])
    p += 1
    return v


def gbdt_host_expand_ctr(
    x_raw: List[Float32],
    n_rows: Int,
    n_columns: Int,
    ctr_ints: List[Int32],
    ctr_floats: List[Float32],
    n_ctr_tables: Int,
    tensor_ints: List[Int32],
    tensor_floats: List[Float32],
    n_tensor_tables: Int,
    counts: List[Int32],
    border_offsets: List[Int32],
    borders: List[Float32],
    one_hot: List[Int32],
) raises -> List[Float32]:
    """Raw columns (column-major) in, the model's `n_columns` columns
    (column-major) out. One of `n_ctr_tables` and `n_tensor_tables` is zero:
    `predict_floats` refuses a model carrying both ("combined simple-CTR and
    tensor-CTR model apply needs a composed column plan and is not wired
    yet"), and so does this.

    `tensor_ints` holds, per table: n_sources, the sources, their
    cardinalities, n_splits, (feature, bin, split type) per split, classes,
    target_border, denominator, counts offset, counts length."""
    if n_ctr_tables != 0 and n_tensor_tables != 0:
        raise Error(
            "combined simple-CTR and tensor-CTR model apply needs a composed"
            " column plan and is not wired yet"
        )
    if n_ctr_tables < 0 or n_tensor_tables < 0 or n_rows <= 0 or n_columns <= 0:
        raise Error("gbdt host ctr: counts must be non-negative and shapes positive")
    if len(ctr_ints) < n_ctr_tables * CTR_INTS_PER_TABLE or (
        len(ctr_floats) < n_ctr_tables * CTR_FLOATS_PER_TABLE
    ):
        raise Error("gbdt host ctr: the CTR table arrays are short")
    if n_tensor_tables == 0:
        var tables = List[TCtrValueTable]()
        for t in range(n_ctr_tables):
            var at = t * CTR_INTS_PER_TABLE
            var fl = t * CTR_FLOATS_PER_TABLE
            var classes = Int(ctr_ints[at + 4])
            tables.append(
                TCtrValueTable(
                    Int(ctr_ints[at]), Int(ctr_ints[at + 1]), Int(ctr_ints[at + 2]),
                    ctr_floats[fl], ctr_floats[fl + 1], ctr_floats[fl + 2],
                    ctr_floats[fl + 3], Int(ctr_ints[at + 3]), classes,
                    Int(ctr_ints[at + 5]),
                    _counts(counts, Int(ctr_ints[at + 6]), Int(ctr_ints[at + 7]), classes),
                )
            )
        return expand_raw_columns(tables, n_columns, x_raw, n_rows)

    if len(tensor_floats) < n_tensor_tables * TENSOR_FLOATS_PER_TABLE:
        raise Error("gbdt host ctr: the tensor prior array is short")
    if len(border_offsets) != n_columns + 1 or len(one_hot) != n_columns:
        raise Error("gbdt host ctr: border offsets or one-hot flags do not match the columns")
    var model_borders = List[List[Float32]](capacity=n_columns)
    var flags = List[Bool](capacity=n_columns)
    for c in range(n_columns):
        var lo = Int(border_offsets[c])
        var hi = Int(border_offsets[c + 1])
        if lo < 0 or hi < lo or hi > len(borders):
            raise Error("gbdt host ctr: border offsets are not a prefix scan")
        var bs = List[Float32](capacity=hi - lo)
        for b in range(lo, hi):
            bs.append(borders[b])
        model_borders.append(bs^)
        flags.append(one_hot[c] != 0)
    var tensors = List[TTensorCtrApplyTable]()
    var p = 0
    for t in range(n_tensor_tables):
        var n_sources = _take(tensor_ints, p)
        if n_sources < 1:
            raise Error("gbdt host ctr: a tensor table has no source")
        var sources = List[Int]()
        for _ in range(n_sources):
            sources.append(_take(tensor_ints, p))
        var cards = List[Int]()
        for _ in range(n_sources):
            cards.append(_take(tensor_ints, p))
        var n_splits = _take(tensor_ints, p)
        if n_splits < 0:
            raise Error("gbdt host ctr: a tensor table has a negative split count")
        var sf = List[Int]()
        var sb = List[Int]()
        var st = List[Int]()
        for _ in range(n_splits):
            sf.append(_take(tensor_ints, p))
            sb.append(_take(tensor_ints, p))
            st.append(_take(tensor_ints, p))
        var classes = _take(tensor_ints, p)
        var target_border = _take(tensor_ints, p)
        var denominator = _take(tensor_ints, p)
        var offset = _take(tensor_ints, p)
        var length = _take(tensor_ints, p)
        tensors.append(
            TTensorCtrApplyTable(
                sources^, cards^, sf^, sb^, st^, classes, target_border,
                tensor_floats[t * TENSOR_FLOATS_PER_TABLE],
                tensor_floats[t * TENSOR_FLOATS_PER_TABLE + 1], denominator,
                _counts(counts, offset, length, classes),
            )
        )
    if p != len(tensor_ints):
        raise Error("gbdt host ctr: trailing values in the tensor table array")
    return expand_tensor_ctr_columns(tensors, x_raw, n_rows, model_borders, flags)
