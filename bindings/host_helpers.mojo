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

Workstream E (lane/cpu-training-e, 2026-09-14) adds the three centering
helpers `python/mojolearn/linear_model.py` reaches through `_native` on an
OLS or ridge fit (`column_mean_f64`, `center_columns_f32`, `scale_rows_f32`,
DEVIATIONS 2324, 2441, 2442): on the seven-runner CPU identity gate the ols
and ridge lanes REFUSED at `_mojolearn.column_mean_f64` (run 34869406147)
because the core host binding, which stands in for the base binding on a
CPU-only install, did not carry them. Their bodies are the base binding's
definitions spelled on the calling thread: the base binding splits columns
or rows over the host pool above 2^20 cells (DEVIATION 2632) without
splitting, merging or reordering any column's chain or any cell's
operation, so the serial loop here is the same additions in the same
order and the same bits by construction.

Every body here is the base binding's, `bindings/_mojolearn.mojo`, copied
verbatim (the three DEVIATION 2614 ones as the byte LM host binding carries
them). `bindings/_mojolearn.mojo` and `bindings/_mojolearn_byte_lm_host.mojo`
are not changed and keep their own copies, so neither certified artifact
moves. A change to one of these bodies belongs in all three places.
"""
from std.math import isfinite
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from max.algorithm import sync_parallelize

from bindings.hostptr import f32_ptr, f64_ptr
from core.host_predict_threads import (
    host_list_ptr_u32,
    host_predict_chunk,
    host_predict_task_count,
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
        for i in range(count):
            if not isfinite(p.unsafe_load(i)):
                ok = 0
                break
    return PythonObject(ok)


def all_finite_f64_binding(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """`all_finite_f32_binding` over float64 values."""
    var p = f64_ptr(Int(py=addr))
    var count = Int(py=n)
    if count < 0:
        raise Error("all_finite_f64: n must be non-negative, got " + String(count))
    var ok: Int = 1
    with GILReleased(Python()):
        for i in range(count):
            if not isfinite(p.unsafe_load(i)):
                ok = 0
                break
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
        var i = 0
        var body = count - (count % 8)
        while i < body:
            var v = sp.unsafe_load[width=8](i)
            dp.unsafe_store[width=8](i, v.cast[DType.float32]())
            i += 8
        while i < count:
            dp.unsafe_store(i, sp.unsafe_load(i).cast[DType.float32]())
            i += 1
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
        for i in range(count):
            var c = Int(cp.unsafe_load(i))
            if c < 0 or c >= nt:
                bad = True
                break
        if not bad:
            for i in range(count):
                dp.unsafe_store(i, tp.unsafe_load(Int(cp.unsafe_load(i))))
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
        for i in range(count):
            var c = Int(cp.unsafe_load(i))
            if c < 0 or c >= nt:
                bad = True
                break
        if not bad:
            for i in range(count):
                dp.unsafe_store(i, tp.unsafe_load(Int(cp.unsafe_load(i))))
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
        # Validate all indices before any output mutation.
        for r in range(no):
            var index = Int(idx.unsafe_load(r))
            if index < 0 or index >= ns:
                invalid = True
                break
        if not invalid:
            for r in range(no):
                var index = Int(idx.unsafe_load(r))
                memcpy(dest=dst + r * width, src=src + index * width, count=width)
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
        for r in range(rows):
            var base = r * cols
            var best = 0
            var best_value = sp.unsafe_load(base)
            for c in range(1, cols):
                var value = sp.unsafe_load(base + c)
                if value > best_value:
                    best = c
                    best_value = value
            dp.unsafe_store(r, Int64(best))
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
        for r in range(rows):
            var base = r * cols
            var best = 0
            var best_value = sp.unsafe_load(base)
            for c in range(1, cols):
                var value = sp.unsafe_load(base + c)
                if value > best_value:
                    best = c
                    best_value = value
            dp.unsafe_store(r, Int64(best))
    return PythonObject(0)


# ---------------------------------------------------------------------------
# The three centering helpers of `linear_model.py` (workstream E): the base
# binding's definitions on the calling thread.
# ---------------------------------------------------------------------------


def column_mean_f64_binding(
    x_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn.mojo::column_mean_f64_binding`, DEVIATION 2324:
    per column, one binary64 round-to-nearest-even addition of the widened
    float32 element onto the running total, row by row, then one division
    by `rows`. No pairwise tree, no lane split, no Kahan term. Returns 0."""
    var xp = f32_ptr(Int(py=x_addr))
    var op = f64_ptr(Int(py=out_addr))
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr <= 0:
        raise Error(
            "column_mean_f64: rows must be positive, got " + String(nr)
        )
    if nc <= 0:
        raise Error(
            "column_mean_f64: cols must be positive, got " + String(nc)
        )
    with GILReleased(Python()):
        var acc = List[Float64](length=nc, fill=Float64(0.0))
        for r in range(nr):
            var row = r * nc
            for c in range(nc):
                acc[c] = acc[c] + Float64(xp.unsafe_load(row + c))
        for c in range(nc):
            op.unsafe_store(c, acc[c] / Float64(nr))
    return PythonObject(0)


def center_columns_f32_binding(
    x_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
    mean_addr: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn.mojo::center_columns_f32_binding`, DEVIATION
    2441: `out[r, c] = Float32(Float64(x[r, c]) - mean[c])`, `mean` a
    float64 buffer the caller already narrowed to float32 values. `out`
    may alias `x`. Returns 0."""
    var xp = f32_ptr(Int(py=x_addr))
    var mp = f64_ptr(Int(py=mean_addr))
    var op = f32_ptr(Int(py=out_addr))
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr < 0 or nc < 0:
        raise Error(
            "center_columns_f32: rows and cols must be non-negative, got "
            + String(nr) + " x " + String(nc)
        )
    with GILReleased(Python()):
        for r in range(nr):
            for c in range(nc):
                var d = Float64(xp.unsafe_load(r * nc + c)) - mp.unsafe_load(c)
                op.unsafe_store(r * nc + c, Float32(d))
    return PythonObject(0)


def scale_rows_f32_binding(
    x_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
    w_addr: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn.mojo::scale_rows_f32_binding`, DEVIATION 2442:
    `out[r, c] = Float32(Float64(x[r, c]) * Float64(w[r]))`, `w` float32.
    `out` may alias `x`. Returns 0."""
    var xp = f32_ptr(Int(py=x_addr))
    var wp = f32_ptr(Int(py=w_addr))
    var op = f32_ptr(Int(py=out_addr))
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr < 0 or nc < 0:
        raise Error(
            "scale_rows_f32: rows and cols must be non-negative, got "
            + String(nr) + " x " + String(nc)
        )
    with GILReleased(Python()):
        for r in range(nr):
            var w = Float64(wp.unsafe_load(r))
            for c in range(nc):
                var p = Float64(xp.unsafe_load(r * nc + c)) * w
                op.unsafe_store(r * nc + c, Float32(p))
    return PythonObject(0)


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
        var tasks = host_predict_task_count(nr)
        if nr * nc < 32768:
            tasks = 1
        var chunk = host_predict_chunk(nr, tasks)
        var flags = List[UInt32](length=tasks, fill=UInt32(0))
        var fp = host_list_ptr_u32(flags)

        def _rows(task: Int) {imm src, imm dst, imm nr, imm nc, imm bin, imm chunk, imm fp}:
            var lo = task * chunk
            var hi = min(lo + chunk, nr)
            var bits = UInt32(0)
            for r in range(lo, hi):
                var total = Float64(0)
                for c in range(nc):
                    var p = src.unsafe_load(r * nc + c)
                    if not isfinite(p):
                        bits |= UInt32(1)
                    if p < 0 or p > 1:
                        bits |= UInt32(2)
                    total += Float64(p)
                    if bin == 1:
                        dst.unsafe_store(2 * r, Float32(1) - p)
                        dst.unsafe_store(2 * r + 1, p)
                if bin == 0:
                    var error = total - Float64(1)
                    if abs(error) > Float64(0.00034526697709225118):
                        bits |= UInt32(4)
            fp.unsafe_store(task, bits)

        if tasks == 1:
            _rows(0)
        else:
            sync_parallelize(_rows, tasks)
        for task in range(tasks):
            combined |= flags[task]
    if (combined & UInt32(1)) != 0:
        return PythonObject(1)
    if (combined & UInt32(2)) != 0:
        return PythonObject(2)
    if (combined & UInt32(4)) != 0:
        return PythonObject(3)
    return PythonObject(0)
