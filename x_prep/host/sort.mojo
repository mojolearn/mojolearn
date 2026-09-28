# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host's `sort_cols` (op 0 of x_prep/units.mojo), lane prep-cpu
2026-09-28: the SAME OUTPUT WORDS as the device's heap sort, faster.

Why the words cannot move. `sort_cols_unit` (x_prep/prims.mojo) writes
column c's values, flushed (and canonicalised when asked), ordered by
`key` (x_prep/common.mojo, DEVIATION 5402). `key` is one-to-one on every
non-NaN word and sends every NaN word to 0xFFFFFFFF. So the non-NaN prefix
of ANY correct sort by `key` is one sequence of words, whatever algorithm
produced it, and the NaN suffix is n_nan copies of one word whenever the
column holds a single NaN word (always, with `canon`). This path sorts the
32-bit keys themselves (std `sort` on UInt32, a total order on integers)
and writes each key's word back (`unkey`, the inverse of `key` off NaN).

When a column holds two or more DIFFERENT NaN words without `canon`, their
order is the heap sort's own, so that column runs the device's unit
(`sort_cols_unit`) unchanged. The check x_prep/seams/host_sort_check.mojo
holds this path to the heap sort, word for word, on columns with -0.0/+0.0,
subnormals, infinities, ties, one NaN word and mixed NaN payloads.
"""
from std.builtin.sort import sort
from std.memory import bitcast
from checks.numerics import ftz
from x_prep.common import FP, IP, p, raw, canon, key
from x_prep.prims import sort_cols_unit

comptime NAN_KEY = UInt32(0xFFFFFFFF)


@always_inline
def unkey(k: UInt32) -> Float32:
    """The word `key` sent to k (k != NAN_KEY)."""
    if (k & UInt32(0x80000000)) != UInt32(0):
        return bitcast[DType.float32](k & UInt32(0x7FFFFFFF))
    return bitcast[DType.float32](~k)


def sort_cols_host_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, S, canon]; t = column: `sort_cols_unit`'s words."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var S = p(q, 3)
    var do_canon = p(q, 4) != 0
    var c = t
    var keys = List[UInt32](capacity=n)
    var nan_word = UInt32(0)
    var nan_count = 0
    for i in range(n):
        var v = ftz(raw(f, X + i * d + c))
        if do_canon:
            v = canon(v)
        var k = key(v)
        if k == NAN_KEY:
            var w = bitcast[DType.uint32](v)
            if nan_count == 0:
                nan_word = w
            elif w != nan_word:
                sort_cols_unit(t, f, q)
                return
            nan_count += 1
        keys.append(k)
    sort(keys)
    var m = n - nan_count
    var out = S + c * n
    for i in range(m):
        f.unsafe_store(out + i, unkey(keys[i]))
    var nw = bitcast[DType.float32](nan_word)
    for i in range(m, n):
        f.unsafe_store(out + i, nw)
