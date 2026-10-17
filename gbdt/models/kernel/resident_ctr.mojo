# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The resident GBDT predict's CTR expansion on the device (lane
gbdt-host-loops-12, 2026-10-17).

`gbdt/resident_model.mojo::predict_into` used to transpose a row-major input
on the host and run the CTR expansion there (`ctr_value_table.
expand_raw_columns` for the simple tables, `tensor_ctr_apply.
expand_tensor_ctr_columns` for the tensor ones): host loops over every row
of the GPU predict. Here the raw input goes up as the caller handed it (row-
or column-major) and the expansion runs in kernels that write the model
columns, column-major, straight into the resident model's `d_raw`, which the
staging kernel reads as before:

  * `ctr_simple_expand_kernel`: one thread per (model column, row). A column
    without a table copies its raw column; a CTR column takes its table's
    value for the row's dense category code (`TCtrValueTable.value_for`,
    `calc`, `empty_value`, arm for arm).
  * `ctr_raw_bins_kernel` + `ctr_tensor_column_kernel` (tensor models): the
    raw columns copied and quantized against the model's own borders
    (categorical columns: the code is the bin), then ONE LAUNCH PER TENSOR
    TABLE in model-column order, one thread per row: the mixed-radix key,
    the split-history bits off the bins of earlier columns, the table's
    value, and that column's bins for the tables after it. The launches are
    sequential because a split history may name an earlier tensor column
    (the host's order); the rows are parallel.

THE SAME BITS AS THE HOST. Every float32 operation of the host statements
(`(count + prior_num) / (total + prior_denom)`, `(ctr + shift) * scale`, the
int-to-float conversions) is computed as the correctly rounded soft-float64
result narrowed once (`checks/soft_f64.mojo`), which is the correctly rounded
float32 result of that one operation on every vendor: no contraction, no
approximate division, subnormals honored. The comparisons and the integer
key arithmetic are exact. The host column (`gbdt/train.mojo::predict_floats`)
keeps the host functions and computes the same words.

THE REFUSALS. A value that is not a dense category code (or, on a tensor
model, any NaN) is recorded with `Atomic.min` into `bad[0]` (the lowest
feature) and `bad[1 + f]` (that feature's lowest row); after the drain the
host re-raises the host function's own message for that one value
(`dense_category_code`). The host scanned feature-major in row order, so the
first refusal is the same cell, except that a tensor table's source feature
that is not one-hot is checked with the others rather than after them.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import isfinite
from std.memory import memcpy
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.soft_f64 import (
    sf64_add,
    sf64_div,
    sf64_from_f32,
    sf64_from_int,
    sf64_mul,
    sf64_to_f32,
)
from gbdt.ctrs.ctr import CTR_BORDERS, CTR_COUNTER, CTR_FEATURE_FREQ
from gbdt.models.ctr_value_table import (
    TCtrValueTable,
    column_plan,
    ctr_type_name,
    dense_category_code,
)
from gbdt.models.tensor_ctr_apply import (
    TENSOR_SPLIT_TAKE_BIN,
    TENSOR_SPLIT_TAKE_GREATER,
    TTensorCtrApplyTable,
)

#: the expansion kernels' block
comptime CTR_BLOCK = 256
#: `bad[0]` / `bad[1 + f]` before any refusal
comptime CTR_NO_BAD = Int32(0x7FFFFFFF)
#: the resident model's border slab (`resident_model.BORDER_SLAB`): one
#: float32 slab per model column, `[count, border_0, ...]`
comptime CTR_BORDER_SLAB = 256
#: words per simple table in `d_meta_i` / `d_meta_f`
comptime SIMPLE_I = 6
comptime SIMPLE_F = 4
#: words per tensor table in `d_meta_i` / `d_meta_f`
comptime TENSOR_I = 9
comptime TENSOR_F = 2


# ===========================================================================
# DEVICE STATEMENTS
# ===========================================================================


@always_inline
def _f32(w: UInt64) -> Float32:
    return sf64_to_f32(w)


@always_inline
def _int_to_f32(i: Int) -> Float32:
    """`Float32(i)`: the integer as a double (exact below 2^53) narrowed
    once, the correctly rounded conversion."""
    return _f32(sf64_from_int(i))


@always_inline
def _add32(a: Float32, b: Float32) -> Float32:
    return _f32(sf64_add(sf64_from_f32(a), sf64_from_f32(b)))


@always_inline
def _div32(a: Float32, b: Float32) -> Float32:
    return _f32(sf64_div(sf64_from_f32(a), sf64_from_f32(b)))


@always_inline
def _mul32(a: Float32, b: Float32) -> Float32:
    return _f32(sf64_mul(sf64_from_f32(a), sf64_from_f32(b)))


@always_inline
def _code_ok(v: Float32) -> Bool:
    """`dense_category_code`'s three tests: finite, in `[0, 2^32)`, and an
    exact integer."""
    if not isfinite(v):
        return False
    if v < Float32(0.0) or v >= Float32(4294967296.0):
        return False
    return _int_to_f32(Int(v)) == v


@always_inline
def _flag_bad(bad: MutPointer[Int32, MutAnyOrigin], f: Int, r: Int):
    _ = Atomic.min(bad, Int32(f))
    _ = Atomic.min(bad + 1 + f, Int32(r))


@always_inline
def _raw_at(
    x: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    n_raw: Int,
    row_major: Int32,
    f: Int,
    r: Int,
) -> Float32:
    if row_major != Int32(0):
        return x[r * n_raw + f]
    return x[f * n + r]


@always_inline
def _calc(
    count_in_class: Float32,
    total_count: Float32,
    prior_num: Float32,
    prior_denom: Float32,
    shift: Float32,
    scale: Float32,
) -> Float32:
    """`TCtrValueTable.calc` (`TModelCtr::Calc`), one rounding per float32
    operation and no contraction (the host's `@no_inline` boundary)."""
    var ctr = _div32(
        _add32(count_in_class, prior_num), _add32(total_count, prior_denom)
    )
    return _mul32(_add32(ctr, shift), scale)


@always_inline
def _simple_value(  # device-helper: one row's CTR value inside ctr_simple_expand_kernel
    meta_i: MutPointer[Int64, MutAnyOrigin],
    meta_f: MutPointer[Float32, MutAnyOrigin],
    counts: MutPointer[Int64, MutAnyOrigin],
    t: Int,
    code: Int,
) -> Float32:
    """`TCtrValueTable.value_for(code)` (the metadata was validated when the
    plan was built, so no arm raises here)."""
    var ctr_type = Int(meta_i[t * SIMPLE_I + 0])
    var classes = Int(meta_i[t * SIMPLE_I + 1])
    var border_idx = Int(meta_i[t * SIMPLE_I + 2])
    var off = Int(meta_i[t * SIMPLE_I + 3])
    var length = Int(meta_i[t * SIMPLE_I + 4])
    var denom = Int(meta_i[t * SIMPLE_I + 5])
    var pn = meta_f[t * SIMPLE_F + 0]
    var pd = meta_f[t * SIMPLE_F + 1]
    var sh = meta_f[t * SIMPLE_F + 2]
    var sc = meta_f[t * SIMPLE_F + 3]
    if ctr_type == CTR_FEATURE_FREQ or ctr_type == CTR_COUNTER:
        if code < 0 or code >= length:
            return _calc(Float32(0.0), _int_to_f32(denom), pn, pd, sh, sc)
        return _calc(
            _int_to_f32(Int(counts[off + code])), _int_to_f32(denom),
            pn, pd, sh, sc,
        )
    # CTR_BORDERS
    var cats = length // classes
    if code < 0 or code >= cats:
        return _calc(Float32(0.0), Float32(0.0), pn, pd, sh, sc)
    var base = off + code * classes
    if classes > 2:
        var total = 0
        var good = 0
        for class_id in range(border_idx + 1):  # small-loop(border_idx: target classes of one Borders table): device statement, one row's histogram
            total += Int(counts[base + class_id])
        for class_id in range(border_idx + 1, classes):  # small-loop(classes: target classes of one Borders table): device statement, one row's histogram
            good += Int(counts[base + class_id])
        total += good
        return _calc(_int_to_f32(good), _int_to_f32(total), pn, pd, sh, sc)
    var c0 = Int(counts[base])
    var c1 = Int(counts[base + 1])
    return _calc(_int_to_f32(c1), _int_to_f32(c0 + c1), pn, pd, sh, sc)


@always_inline
def _bin_of(  # device-helper: one value's bin against one column's borders inside the expansion kernels
    borders: MutPointer[Float32, MutAnyOrigin], column: Int, v: Float32
) -> UInt32:
    """`for b in borders[column]: if v > b: bin += 1`."""
    var slab = column * CTR_BORDER_SLAB
    var count = Int(borders[slab])
    var bin = 0
    for b in range(count):  # small-loop(count: borders of one model column, under 256): device statement, one value's bin
        if v > borders[slab + 1 + b]:
            bin += 1
    return UInt32(bin)


# ===========================================================================
# KERNELS
# ===========================================================================


def ctr_simple_expand_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_raw_in: Int32,
    row_major: Int32,
    n_cols_in: Int32,
    col_plan: MutPointer[Int64, MutAnyOrigin],
    meta_i: MutPointer[Int64, MutAnyOrigin],
    meta_f: MutPointer[Float32, MutAnyOrigin],
    counts: MutPointer[Int64, MutAnyOrigin],
    out: MutPointer[Float32, MutAnyOrigin],
    bad: MutPointer[Int32, MutAnyOrigin],
):
    """`expand_raw_columns`: one thread per (model column, row),
    grid-stride; `col_plan[2 c]` is the column's raw source, `col_plan[2 c +
    1]` its table (-1: copied through). `out` is column-major."""
    var n = Int(n_rows_in)
    var n_raw = Int(n_raw_in)
    var total = Int(n_cols_in) * n
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < total:
        var c = i // n
        var r = i - c * n
        var src = Int(col_plan[2 * c])
        var t = Int(col_plan[2 * c + 1])
        var v = _raw_at(x, n, n_raw, row_major, src, r)
        if t < 0:
            out[i] = v
        elif not _code_ok(v):
            _flag_bad(bad, src, r)
            out[i] = Float32(0.0)
        else:
            out[i] = _simple_value(meta_i, meta_f, counts, t, Int(v))
        i += stride


def ctr_raw_bins_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_raw_in: Int32,
    row_major: Int32,
    one_hot: MutPointer[Int32, MutAnyOrigin],
    borders: MutPointer[Float32, MutAnyOrigin],
    out: MutPointer[Float32, MutAnyOrigin],
    bins: MutPointer[UInt32, MutAnyOrigin],
    bad: MutPointer[Int32, MutAnyOrigin],
):
    """`expand_tensor_ctr_columns`' first pass: every raw column copied to
    `out` (column-major) and quantized into `bins` (a categorical column's
    code is its bin), one thread per (raw column, row), grid-stride. A NaN
    anywhere, or a categorical value that is not a code, is a refusal."""
    var n = Int(n_rows_in)
    var n_raw = Int(n_raw_in)
    var total = n_raw * n
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < total:
        var f = i // n
        var r = i - f * n
        var v = _raw_at(x, n, n_raw, row_major, f, r)
        out[i] = v
        var bin = UInt32(0)
        if v != v:
            _flag_bad(bad, f, r)
        elif one_hot[f] != Int32(0):
            if _code_ok(v):
                bin = UInt32(Int(v))
            else:
                _flag_bad(bad, f, r)
        else:
            bin = _bin_of(borders, f, v)
        bins[i] = bin
        i += stride


def ctr_tensor_column_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_raw_in: Int32,
    row_major: Int32,
    t_in: Int32,
    meta_i: MutPointer[Int64, MutAnyOrigin],
    meta_f: MutPointer[Float32, MutAnyOrigin],
    aux: MutPointer[Int64, MutAnyOrigin],
    counts: MutPointer[Int64, MutAnyOrigin],
    borders: MutPointer[Float32, MutAnyOrigin],
    column_in: Int32,
    out: MutPointer[Float32, MutAnyOrigin],
    bins: MutPointer[UInt32, MutAnyOrigin],
    bad: MutPointer[Int32, MutAnyOrigin],
):
    """Tensor table `t` (model column `column`), one thread per row,
    grid-stride: `tensor_value_for_row` (the mixed-radix key over the
    sources' codes, -1 at a code past its cardinality; the split-history
    bits off `bins`; `tensor_value_for_key`), the value into `out` and its
    bin into `bins`."""
    var n = Int(n_rows_in)
    var n_raw = Int(n_raw_in)
    var t = Int(t_in)
    var column = Int(column_in)
    var n_src = Int(meta_i[t * TENSOR_I + 0])
    var src_off = Int(meta_i[t * TENSOR_I + 1])
    var n_split = Int(meta_i[t * TENSOR_I + 2])
    var split_off = Int(meta_i[t * TENSOR_I + 3])
    var classes = Int(meta_i[t * TENSOR_I + 4])
    var border_idx = Int(meta_i[t * TENSOR_I + 5])
    var off = Int(meta_i[t * TENSOR_I + 6])
    var length = Int(meta_i[t * TENSOR_I + 7])
    var denom = Int(meta_i[t * TENSOR_I + 8])
    var pn = meta_f[t * TENSOR_F + 0]
    var pd = meta_f[t * TENSOR_F + 1]
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while r < n:
        # `tensor_key_for_row`
        var key = 0
        for s in range(n_src):
            var f = Int(aux[src_off + 2 * s])
            var card = Int(aux[src_off + 2 * s + 1])
            var v = _raw_at(x, n, n_raw, row_major, f, r)
            if not _code_ok(v):
                _flag_bad(bad, f, r)
                key = -1
                break
            var code = Int(v)
            if code >= card:
                key = -1  # their NotFoundIndex / empty-value arm
                break
            key = key * card + code
        # the split history (`tensor_value_for_row`)
        if n_split != 0 and key >= 0:
            for s in range(n_split):
                var sf = Int(aux[split_off + 3 * s])
                var sb = Int(aux[split_off + 3 * s + 1])
                var st = Int(aux[split_off + 3 * s + 2])
                var q = Int(bins[sf * n + r])
                var bit = 0
                if st == TENSOR_SPLIT_TAKE_BIN:
                    bit = 1 if q == sb else 0
                else:
                    bit = 1 if q > sb else 0
                key = 2 * key + bit
        # `tensor_value_for_key`
        var value: Float32
        if classes == 0:
            var count = 0
            if key >= 0 and key < length:
                count = Int(counts[off + key])
            value = _div32(
                _add32(_int_to_f32(count), pn), _add32(_int_to_f32(denom), pd)
            )
        elif key < 0 or key * classes + classes > length:
            value = _div32(pn, pd)
        else:
            var base = off + key * classes
            var total = 0
            var good = 0
            for cls in range(classes):
                var cnt = Int(counts[base + cls])
                total += cnt
                if cls > border_idx:
                    good += cnt
            value = _div32(
                _add32(_int_to_f32(good), pn), _add32(_int_to_f32(total), pd)
            )
        out[column * n + r] = value
        bins[column * n + r] = _bin_of(borders, column, value)
        r += stride


# ===========================================================================
# THE PLAN (built once per resident model) AND THE CALL
# ===========================================================================


struct ResidentCtrPlan(Movable):
    """The model's CTR tables flattened for the expansion kernels, plus the
    per-call buffers of the current row count."""

    var tensor: Bool
    var n_raw: Int
    var n_cols: Int
    var n_tables: Int
    #: simple: `[src, table]` per model column; tensor: the sources'
    #: `[feature, cardinality]` pairs, then the splits' `[feature, bin,
    #: type]` triples
    var d_aux: DeviceBuffer[DType.int64]
    var d_meta_i: DeviceBuffer[DType.int64]
    var d_meta_f: DeviceBuffer[DType.float32]
    var d_counts: DeviceBuffer[DType.int64]
    var d_one_hot: DeviceBuffer[DType.int32]
    var d_bad: DeviceBuffer[DType.int32]
    var h_bad: HostBuffer[DType.int32]
    var rows: Int
    var d_in: Optional[DeviceBuffer[DType.float32]]
    var d_bins: Optional[DeviceBuffer[DType.uint32]]

    def __init__(
        out self,
        ctx: DeviceContext,
        ctr_tables: List[TCtrValueTable],
        tensor_tables: List[TTensorCtrApplyTable],
        n_columns: Int,
        one_hot: List[Bool],
        n_raw: Int,
    ) raises:
        var tensor = len(tensor_tables) != 0
        self.tensor = tensor
        self.n_raw = n_raw
        self.n_cols = n_columns
        self.rows = 0
        self.d_in = None
        self.d_bins = None
        var aux = List[Int]()
        var mi = List[Int]()
        var mf = List[Float32]()
        var counts = List[Int]()
        if tensor:
            self.n_tables = len(tensor_tables)
            if n_raw + len(tensor_tables) != n_columns:
                raise Error("tensor CTR registry/model border count mismatch")
            if len(one_hot) != 0 and len(one_hot) != n_columns:
                raise Error("tensor CTR registry/model one-hot count mismatch")
            # driver-loop: over the model's tensor CTR tables (model structure)
            for t in range(len(tensor_tables)):  # small-loop(tensor_tables: tensor CTR model columns): packs model metadata once per model
                ref tab = tensor_tables[t]
                var classes = tab.target_classes_count
                if classes != 0 and (
                    classes < 2
                    or tab.target_border_idx < 0
                    or tab.target_border_idx >= classes - 1
                ):
                    raise Error("invalid Borders tensor target-class metadata")
                mi.append(len(tab.source_features))
                mi.append(len(aux))
                # driver-loop: over one tensor table's source features
                for s in range(len(tab.source_features)):  # small-loop(source_features: one tensor table's source features): packs model metadata once per model
                    var f = tab.source_features[s]
                    if f < 0 or f >= n_raw:
                        raise Error(
                            "tensor CTR source feature " + String(f)
                            + " is outside the " + String(n_raw)
                            + " raw input columns"
                        )
                    aux.append(f)
                    aux.append(tab.cardinalities[s])
                mi.append(len(tab.split_features))
                mi.append(len(aux))
                # driver-loop: over one tensor table's split history
                for s in range(len(tab.split_features)):  # small-loop(split_features: one tensor table's split history, bounded by tree depth): packs model metadata once per model
                    var sf = tab.split_features[s]
                    var sb = tab.split_bins[s]
                    var st = tab.split_types[s]
                    if sf < 0 or sb < 0 or sf >= n_columns:
                        raise Error(
                            "split-history tensor references an invalid"
                            " quantized column"
                        )
                    if st != TENSOR_SPLIT_TAKE_BIN and st != TENSOR_SPLIT_TAKE_GREATER:
                        raise Error("split-history tensor has an unknown split type")
                    aux.append(sf)
                    aux.append(sb)
                    aux.append(st)
                mi.append(classes)
                mi.append(tab.target_border_idx)
                mi.append(len(counts))
                mi.append(len(tab.counts))
                mi.append(tab.denominator)
                mf.append(tab.prior_num)
                mf.append(tab.prior_denom)
                counts.extend(tab.counts.copy())
        else:
            self.n_tables = len(ctr_tables)
            var plan = column_plan(ctr_tables, n_columns)
            if plan.n_input_features != n_raw:
                raise Error(
                    "expected " + String(plan.n_input_features)
                    + " raw input columns, the model reports " + String(n_raw)
                )
            # driver-loop: over the model columns (model structure)
            for c in range(n_columns):  # small-loop(n_columns: model columns): packs the column plan once per model
                aux.append(plan.source_of_column[c])
                aux.append(plan.table_of_column[c])
            # driver-loop: over the model's CTR tables (model structure)
            for t in range(len(ctr_tables)):  # small-loop(ctr_tables: CTR model columns): packs model metadata once per model
                ref tab = ctr_tables[t]
                if tab.ctr_type == CTR_BORDERS:
                    var classes = tab.target_classes_count
                    if classes < 2:
                        raise Error(
                            "a Borders table needs at least two target classes and"
                            " has " + String(classes)
                            + "; their TargetClassesCount is Borders.size() + 1"
                            " (libs/model/target_classifier.h:32-34)"
                        )
                    if tab.target_border_idx < 0 or (
                        tab.target_border_idx >= classes
                    ):
                        raise Error(
                            "TargetBorderIdx " + String(tab.target_border_idx)
                            + " is outside the " + String(classes)
                            + " target classes this table carries"
                        )
                elif tab.ctr_type != CTR_FEATURE_FREQ and tab.ctr_type != CTR_COUNTER:
                    raise Error(
                        "no apply-time table arithmetic is implemented for ctr type "
                        + ctr_type_name(tab.ctr_type)
                        + "; TCatFeatureParams.check() admits only Borders and"
                        " FeatureFreq (catboost_options.mojo), so a table of any other"
                        " type is a corrupt file rather than a missing feature"
                    )
                mi.append(tab.ctr_type)
                mi.append(tab.target_classes_count)
                mi.append(tab.target_border_idx)
                mi.append(len(counts))
                mi.append(len(tab.counts))
                mi.append(tab.counter_denominator)
                mf.append(tab.prior_num)
                mf.append(tab.prior_denom)
                mf.append(tab.shift)
                mf.append(tab.scale)
                counts.extend(tab.counts.copy())
        var oh = List[Int32](length=max(n_columns, 1), fill=Int32(0))
        # driver-loop: over the model columns (their one-hot flags)
        for c in range(len(one_hot)):  # small-loop(one_hot: model columns): packs the one-hot flags once per model
            if one_hot[c]:
                oh[c] = Int32(1)
        self.d_aux = _upload_i64(ctx, aux)
        self.d_meta_i = _upload_i64(ctx, mi)
        self.d_meta_f = _upload_f32(ctx, mf)
        self.d_counts = _upload_i64(ctx, counts)
        var h_oh = ctx.enqueue_create_host_buffer[DType.int32](len(oh))
        memcpy(dest=h_oh.unsafe_ptr(), src=oh.unsafe_ptr(), count=len(oh))
        self.d_one_hot = ctx.enqueue_create_buffer[DType.int32](len(oh))
        ctx.enqueue_copy(dst_buf=self.d_one_hot, src_ptr=h_oh.unsafe_ptr())
        self.d_bad = ctx.enqueue_create_buffer[DType.int32](1 + max(n_raw, 1))
        self.h_bad = ctx.enqueue_create_host_buffer[DType.int32](1 + max(n_raw, 1))
        ctx.synchronize()
        _ = h_oh^

    def __deinit__(deinit self):
        # every buffer before the context it was created on (DEVIATION
        # 1946): the owner releases the plan before its context
        _ = self.d_bins^
        _ = self.d_in^
        _ = self.h_bad^
        _ = self.d_bad^
        _ = self.d_one_hot^
        _ = self.d_counts^
        _ = self.d_meta_f^
        _ = self.d_meta_i^
        _ = self.d_aux^


def _upload_i64(ctx: DeviceContext, values: List[Int]) raises -> DeviceBuffer[DType.int64]:
    """One bulk copy of a host list of model metadata (at least one word)."""
    var n = max(len(values), 1)
    var h = ctx.enqueue_create_host_buffer[DType.int64](n)
    var d = ctx.enqueue_create_buffer[DType.int64](n)
    ctx.synchronize()
    h.unsafe_ptr().unsafe_store(0, Int64(0))
    if len(values) > 0:
        memcpy(
            dest=h.unsafe_ptr(),
            src=values.unsafe_ptr().bitcast[Int64](),
            count=len(values),
        )
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^
    return d^


def _upload_f32(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var n = max(len(values), 1)
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    h.unsafe_ptr().unsafe_store(0, Float32(0.0))
    if len(values) > 0:
        memcpy(dest=h.unsafe_ptr(), src=values.unsafe_ptr(), count=len(values))
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^
    return d^


def _grid(total: Int) -> Int:
    var g = (total + CTR_BLOCK - 1) // CTR_BLOCK
    if g < 1:
        g = 1
    return min(g, 65535)


def enqueue_resident_ctr_expand(
    ctx: DeviceContext,
    mut plan: ResidentCtrPlan,
    x: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    row_major: Bool,
    mut d_borders: DeviceBuffer[DType.float32],
    mut d_out: DeviceBuffer[DType.float32],
) raises:
    """The raw input up in ONE copy (as handed over), the expansion into
    `d_out` (`n_cols * n_rows`, column-major), and the refusal words' copy
    home, ENQUEUED. `resident_ctr_raise_if_bad` reads them after the
    caller's drain."""
    if plan.rows != n_rows:
        plan.d_bins = None
        plan.d_in = None
        plan.rows = 0
        plan.d_in = ctx.enqueue_create_buffer[DType.float32](
            max(n_rows * plan.n_raw, 1)
        )
        if plan.tensor:
            plan.d_bins = ctx.enqueue_create_buffer[DType.uint32](
                max(n_rows * plan.n_cols, 1)
            )
        plan.rows = n_rows
    plan.d_bad.enqueue_fill(CTR_NO_BAD)
    var rm = Int32(1) if row_major else Int32(0)
    if n_rows * plan.n_raw > 0:
        ctx.enqueue_copy(
            dst_buf=plan.d_in.value(),
            src_ptr=rebind[UnsafePointer[Float32, MutAnyOrigin]](x),
        )
    var d_x = plan.d_in.value().unsafe_ptr()
    if not plan.tensor:
        ctx.enqueue_function[ctr_simple_expand_kernel](
            d_x, Int32(n_rows), Int32(plan.n_raw), rm, Int32(plan.n_cols),
            plan.d_aux.unsafe_ptr(), plan.d_meta_i.unsafe_ptr(),
            plan.d_meta_f.unsafe_ptr(), plan.d_counts.unsafe_ptr(),
            d_out.unsafe_ptr(), plan.d_bad.unsafe_ptr(),
            grid_dim=_grid(n_rows * plan.n_cols), block_dim=CTR_BLOCK,
        )
    else:
        # the host's `bins.resize(n_rows * n_model_features, 0)`: a split
        # history naming a column not yet written reads 0
        plan.d_bins.value().enqueue_fill(UInt32(0))
        ctx.enqueue_function[ctr_raw_bins_kernel](
            d_x, Int32(n_rows), Int32(plan.n_raw), rm,
            plan.d_one_hot.unsafe_ptr(), d_borders.unsafe_ptr(),
            d_out.unsafe_ptr(), plan.d_bins.value().unsafe_ptr(),
            plan.d_bad.unsafe_ptr(),
            grid_dim=_grid(n_rows * plan.n_raw), block_dim=CTR_BLOCK,
        )
        # driver-loop: over the tensor CTR tables in model-column order (a split history may name an earlier one)
        for t in range(plan.n_tables):
            ctx.enqueue_function[ctr_tensor_column_kernel](
                d_x, Int32(n_rows), Int32(plan.n_raw), rm, Int32(t),
                plan.d_meta_i.unsafe_ptr(), plan.d_meta_f.unsafe_ptr(),
                plan.d_aux.unsafe_ptr(), plan.d_counts.unsafe_ptr(),
                d_borders.unsafe_ptr(), Int32(plan.n_raw + t),
                d_out.unsafe_ptr(), plan.d_bins.value().unsafe_ptr(),
                plan.d_bad.unsafe_ptr(),
                grid_dim=_grid(n_rows), block_dim=CTR_BLOCK,
            )
    ctx.enqueue_copy(dst_ptr=plan.h_bad.unsafe_ptr(), src_buf=plan.d_bad)


def resident_ctr_raise_if_bad(
    plan: ResidentCtrPlan,
    x: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    row_major: Bool,
) raises:
    """After the drain: when a refusal was recorded, re-raise the host
    function's own message for that one input value."""
    var f = Int(plan.h_bad.unsafe_ptr().unsafe_load(0))
    if f == Int(CTR_NO_BAD):
        return
    var r = Int(plan.h_bad.unsafe_ptr().unsafe_load(1 + f))
    var v: Float32
    if row_major:
        v = x.unsafe_load(r * plan.n_raw + f)
    else:
        v = x.unsafe_load(f * n_rows + r)
    if plan.tensor and v != v:
        raise Error("tensor CTR split history cannot quantize NaN")
    _ = dense_category_code(v, f, r)
    raise Error(
        "categorical feature " + String(f) + " row " + String(r)
        + " is not a dense category code"
    )
