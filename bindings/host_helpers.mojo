# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The base binding's host helpers, for a binary that is not the base binding.

DEVIATION 2614 put three of these (`all_finite_f32`, `all_finite_f64`,
`cast_f64_to_f32`) into the byte LM host binding so a CPU-only install can
convert its inputs through the same native helper and not a Python copy of
it. The forest host binding needs those three and the four the label helpers
resolve (`argmax_rows_f32`, `argmax_rows_f64`, `gather_i64`, `gather_f64`,
`python/mojolearn/_labels.py`), because `HostForest.predict` on a classifier
is `decode_labels(classes_, argmax_rows(vote))` exactly as the GPU classes
compute it, and on a CPU-only install the base binding that exports them is a
stub.

Workstream E (lane/cpu-training-e, 2026-09-14) added the three centering
helpers of `linear_model.py` here; lane hr-small-passes (2026-10-02)
deleted them (the estimators binding's `lm_col_sums`, `lm_center` and
`lm_scale_rows` replace them on every tier).

Every body here is the base binding's, `bindings/_mojolearn.mojo`, copied
verbatim (the three DEVIATION 2614 ones as the byte LM host binding carries
them). `bindings/_mojolearn.mojo` and `bindings/_mojolearn_byte_lm_host.mojo`
are not changed and keep their own copies, so neither certified artifact
moves. A change to one of these bodies belongs in all three places.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import f32_ptr, f64_ptr
from core.host_helpers_impl import (
    all_finite_f32,
    all_finite_f64,
    argmax_last_rows_f32,
    argmax_rows_f32,
    argmax_rows_f64,
    cast_f64_to_f32,
    gather_f64,
    gather_i64,
    gather_rows_bytes,
    probability_rows_f32,
)


def _i64_ptr(addr: Int) raises -> MutPointer[Int64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int64 buffer address")
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)


def all_finite_f32_binding(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """1 if every one of the `n` float32 values at `addr` is finite, else 0."""
    var p = f32_ptr(Int(py=addr))
    var count = Int(py=n)
    if count < 0:
        raise Error("all_finite_f32: n must be non-negative, got " + String(count))
    var ok: Int = 1
    with GILReleased(Python()):
        ok = all_finite_f32(p, count)
    return PythonObject(ok)


def all_finite_f64_binding(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """`all_finite_f32_binding` over float64 values."""
    var p = f64_ptr(Int(py=addr))
    var count = Int(py=n)
    if count < 0:
        raise Error("all_finite_f64: n must be non-negative, got " + String(count))
    var ok: Int = 1
    with GILReleased(Python()):
        ok = all_finite_f64(p, count)
    return PythonObject(ok)


def cast_f64_to_f32_binding(src_addr: PythonObject, dst_addr: PythonObject,
                            n: PythonObject) raises -> PythonObject:
    """Write `Float32(src[i])` to `dst[i]` for `i` in `[0, n)`. Returns 0."""
    var count = Int(py=n)
    if count < 0:
        raise Error("cast_f64_to_f32: n must be non-negative, got " + String(count))
    if count == 0:
        return PythonObject(0)
    var sp = f64_ptr(Int(py=src_addr))
    var dp = f32_ptr(Int(py=dst_addr))
    with GILReleased(Python()):
        cast_f64_to_f32(sp, dp, count)
    return PythonObject(0)


def gather_i64_binding(
    table_addr: PythonObject, n_table: PythonObject, codes_addr: PythonObject,
    n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`dst[i] = table[codes[i]]` over int64 tables (DEVIATION 2500): the
    decode half of label encoding, `classes_[code]` per predicted row.
    A code outside `[0, n_table)` raises before any write."""
    var count = Int(py=n)
    var nt = Int(py=n_table)
    if count < 0 or nt < 1:
        raise Error("gather_i64: n must be non-negative and the table non-empty")
    if count == 0:
        return PythonObject(0)
    if Int(py=table_addr) == 0 or Int(py=codes_addr) == 0 or Int(py=dst_addr) == 0:
        raise Error("gather_i64: null buffer address")
    var tp = _i64_ptr(Int(py=table_addr))
    var cp = _i64_ptr(Int(py=codes_addr))
    var dp = _i64_ptr(Int(py=dst_addr))
    var bad = False
    with GILReleased(Python()):
        bad = gather_i64(tp, nt, cp, count, dp)
    if bad:
        raise Error("gather_i64: code out of range")
    return PythonObject(0)


def gather_f64_binding(
    table_addr: PythonObject, n_table: PythonObject, codes_addr: PythonObject,
    n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`gather_i64_binding` over a float64 table (int64 codes)."""
    var count = Int(py=n)
    var nt = Int(py=n_table)
    if count < 0 or nt < 1:
        raise Error("gather_f64: n must be non-negative and the table non-empty")
    if count == 0:
        return PythonObject(0)
    if Int(py=table_addr) == 0 or Int(py=codes_addr) == 0 or Int(py=dst_addr) == 0:
        raise Error("gather_f64: null buffer address")
    var tp = f64_ptr(Int(py=table_addr))
    var cp = _i64_ptr(Int(py=codes_addr))
    var dp = f64_ptr(Int(py=dst_addr))
    var bad = False
    with GILReleased(Python()):
        bad = gather_f64(tp, nt, cp, count, dp)
    if bad:
        raise Error("gather_f64: code out of range")
    return PythonObject(0)


def gather_rows_bytes_binding(
    src_addr: PythonObject, dst_addr: PythonObject, indices_addr: PythonObject,
    source_rows: PythonObject, output_rows: PythonObject, row_bytes: PythonObject,
) raises -> PythonObject:
    """`dst[r] = src[indices[r]]`, one fixed-width row of bytes at a time
    (`python/mojolearn/model_selection.py::_take_rows`, the fold rows of
    `cross_val_score`; the cross-val lane, lane/cpu-training-misc,
    2026-09-15). A byte copy: no arithmetic, so the fold rows are the same
    bytes on every column. Every index is checked before any write."""
    var ns = Int(py=source_rows)
    var no = Int(py=output_rows)
    var width = Int(py=row_bytes)
    if ns < 0 or no < 0 or width < 0:
        raise Error("gather_rows_bytes: dimensions must be non-negative")
    if no == 0 or width == 0:
        return PythonObject(0)
    if Int(py=src_addr) == 0 or Int(py=dst_addr) == 0 or Int(py=indices_addr) == 0:
        raise Error("gather_rows_bytes: null buffer address")
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=src_addr))
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=dst_addr))
    var idx = _i64_ptr(Int(py=indices_addr))
    var invalid = False
    with GILReleased(Python()):
        invalid = gather_rows_bytes(src, dst, idx, ns, no, width)
    if invalid:
        raise Error("gather_rows_bytes: row index out of bounds")
    return PythonObject(0)


def argmax_rows_f32_binding(
    scores_addr: PythonObject, n_rows: PythonObject, n_cols: PythonObject,
    dst_addr: PythonObject,
) raises -> PythonObject:
    """Row-wise first-max-wins argmax over a C-order [n_rows, n_cols] float32
    block into int64 codes (DEVIATION 2500), the rule of
    `_labels.argmax_rows`: strictly greater replaces, so ties keep the
    lowest column, and a NaN never replaces (every comparison with it is
    false), so a row of NaN answers column 0 as the Python loop did."""
    var rows = Int(py=n_rows)
    var cols = Int(py=n_cols)
    if rows < 0 or cols < 1:
        raise Error("argmax_rows_f32: n_rows must be non-negative and n_cols positive")
    if rows == 0:
        return PythonObject(0)
    var sp = f32_ptr(Int(py=scores_addr))
    if Int(py=dst_addr) == 0:
        raise Error("argmax_rows_f32: null buffer address")
    var dp = _i64_ptr(Int(py=dst_addr))
    with GILReleased(Python()):
        argmax_rows_f32(sp, rows, cols, dp)
    return PythonObject(0)


def argmax_rows_f64_binding(
    scores_addr: PythonObject, n_rows: PythonObject, n_cols: PythonObject,
    dst_addr: PythonObject,
) raises -> PythonObject:
    """`argmax_rows_f32_binding` over float64 scores."""
    var rows = Int(py=n_rows)
    var cols = Int(py=n_cols)
    if rows < 0 or cols < 1:
        raise Error("argmax_rows_f64: n_rows must be non-negative and n_cols positive")
    if rows == 0:
        return PythonObject(0)
    var sp = f64_ptr(Int(py=scores_addr))
    if Int(py=dst_addr) == 0:
        raise Error("argmax_rows_f64: null buffer address")
    var dp = _i64_ptr(Int(py=dst_addr))
    with GILReleased(Python()):
        argmax_rows_f64(sp, rows, cols, dp)
    return PythonObject(0)


def argmax_last_rows_f32_binding(
    logits_addr: PythonObject, dims: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """dims = [b, l, v]: the first-max argmax (`argmax_rows_f32_binding`'s
    rule: strictly greater replaces, ties keep the lowest column, a NaN never
    replaces) of the LAST position of each row of C-order float32 logits
    `[b, l, v]` into int64 `dst[b]`. The host column of the base binding's
    device `argmax_last_rows_f32` (greedy next-token choice of the byte LM
    and `models.causal_lm`), so a CPU-only install serves it natively."""
    if len(dims) != 3:
        raise Error("argmax_last_rows_f32: dims [b, l, v]")
    var b = Int(py=dims[0])
    var l = Int(py=dims[1])
    var v = Int(py=dims[2])
    if b < 0 or l < 1 or v < 1:
        raise Error("argmax_last_rows_f32: b must be non-negative, l and v positive")
    if b == 0:
        return PythonObject(0)
    if Int(py=logits_addr) == 0 or Int(py=dst_addr) == 0:
        raise Error("argmax_last_rows_f32: null buffer address")
    var sp = f32_ptr(Int(py=logits_addr))
    var dp = _i64_ptr(Int(py=dst_addr))
    with GILReleased(Python()):
        argmax_last_rows_f32(sp, b, l, v, dp)
    return PythonObject(0)


# ---------------------------------------------------------------------------
# The three centering helpers of `linear_model.py` (workstream E): the base
# binding's definitions on the calling thread.
# ---------------------------------------------------------------------------


def probability_rows_f32_binding(
    src_addr: PythonObject, dst_addr: PythonObject, rows: PythonObject,
    cols: PythonObject, binary: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn.mojo::probability_rows_f32_binding`, restated for
    the core host binding (the metrics-classification lane, 2026-09-14): the
    log loss's probability check. Returns 1 for a non-finite value, 2 for a
    value outside [0, 1], 3 for a multiclass row whose Float64 sum is more
    than sqrt(Float32 epsilon) from one, else 0; a binary column is packed
    into `dst` as `[1 - p, p]` rows."""
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    var bin = Int(py=binary)
    if nr < 0 or nc <= 0 or (bin != 0 and bin != 1):
        raise Error("probability_rows_f32: invalid dimensions or binary flag")
    if bin == 1 and nc != 1:
        raise Error("probability_rows_f32: binary input must have one column")
    if nr == 0:
        return PythonObject(0)
    var src = f32_ptr(Int(py=src_addr))
    # Multiclass validation never reads or writes dst.
    var dst = src
    if bin == 1:
        dst = f32_ptr(Int(py=dst_addr))
    var combined = UInt32(0)
    with GILReleased(Python()):
        combined = probability_rows_f32(src, dst, nr, nc, bin)
    if (combined & UInt32(1)) != 0:
        return PythonObject(1)
    if (combined & UInt32(2)) != 0:
        return PythonObject(2)
    if (combined & UInt32(4)) != 0:
        return PythonObject(3)
    return PythonObject(0)
