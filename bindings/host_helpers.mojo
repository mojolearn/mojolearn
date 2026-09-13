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

Every body here is the base binding's, `bindings/_mojolearn.mojo`, copied
verbatim (the three DEVIATION 2614 ones as the byte LM host binding carries
them). `bindings/_mojolearn.mojo` and `bindings/_mojolearn_byte_lm_host.mojo`
are not changed and keep their own copies, so neither certified artifact
moves. A change to one of these bodies belongs in all three places.
"""
from std.math import isfinite
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import f32_ptr, f64_ptr


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
