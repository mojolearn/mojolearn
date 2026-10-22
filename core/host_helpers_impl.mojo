# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed kernels of `bindings/host_helpers.mojo`.

The binding keeps the Python door (argument unpacking, the checks that
raise, GILReleased); the loops live here with no Python import, word for
word as they were in the binding (lane thin-core, 2026-10-21: a pure move,
no bit change).
"""
from std.math import isfinite
from std.memory import memcpy

from core.host_parallel import host_parallelize
from core.host_predict_threads import (
    host_list_ptr_u32,
    host_predict_chunk,
    host_predict_task_count,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]


def all_finite_f32(p: _F32P, count: Int) -> Int:
    """1 if every one of the `count` float32 values at `p` is finite, else 0."""
    var ok: Int = 1
    for i in range(count):
        if not isfinite(p.unsafe_load(i)):
            ok = 0
            break
    return ok


def all_finite_f64(p: _F64P, count: Int) -> Int:
    """`all_finite_f32` over float64 values."""
    var ok: Int = 1
    for i in range(count):
        if not isfinite(p.unsafe_load(i)):
            ok = 0
            break
    return ok


def cast_f64_to_f32(sp: _F64P, dp: _F32P, count: Int):
    """Write `Float32(sp[i])` to `dp[i]` for `i` in `[0, count)`."""
    var i = 0
    var body = count - (count % 8)
    while i < body:
        var v = sp.unsafe_load[width=8](i)
        dp.unsafe_store[width=8](i, v.cast[DType.float32]())
        i += 8
    while i < count:
        dp.unsafe_store(i, sp.unsafe_load(i).cast[DType.float32]())
        i += 1


def gather_i64(tp: _I64P, nt: Int, cp: _I64P, count: Int, dp: _I64P) -> Bool:
    """`dp[i] = tp[cp[i]]`; True (and no write) when a code is outside `[0, nt)`."""
    var bad = False
    for i in range(count):
        var c = Int(cp.unsafe_load(i))
        if c < 0 or c >= nt:
            bad = True
            break
    if not bad:
        for i in range(count):
            dp.unsafe_store(i, tp.unsafe_load(Int(cp.unsafe_load(i))))
    return bad


def gather_f64(tp: _F64P, nt: Int, cp: _I64P, count: Int, dp: _F64P) -> Bool:
    """`gather_i64` over a float64 table (int64 codes)."""
    var bad = False
    for i in range(count):
        var c = Int(cp.unsafe_load(i))
        if c < 0 or c >= nt:
            bad = True
            break
    if not bad:
        for i in range(count):
            dp.unsafe_store(i, tp.unsafe_load(Int(cp.unsafe_load(i))))
    return bad


def gather_rows_bytes(
    src: _U8P, dst: _U8P, idx: _I64P, ns: Int, no: Int, width: Int
) -> Bool:
    """`dst[r] = src[idx[r]]` by rows of `width` bytes; True (and no write)
    when an index is outside `[0, ns)`."""
    var invalid = False
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
    return invalid


def argmax_rows_f32(sp: _F32P, rows: Int, cols: Int, dp: _I64P):
    """Row-wise first-max-wins argmax of a C-order [rows, cols] block."""
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


def argmax_rows_f64(sp: _F64P, rows: Int, cols: Int, dp: _I64P):
    """`argmax_rows_f32` over float64 scores."""
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


def argmax_last_rows_f32(sp: _F32P, b: Int, l: Int, v: Int, dp: _I64P):
    """First-max argmax of the last position of each row of [b, l, v] logits."""
    for r in range(b):
        var base = (r * l + l - 1) * v
        var best = 0
        var best_value = sp.unsafe_load(base)
        for c in range(1, v):
            var value = sp.unsafe_load(base + c)
            if value > best_value:
                best = c
                best_value = value
        dp.unsafe_store(r, Int64(best))

