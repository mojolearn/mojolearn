# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Class counts, class weights and per-row class weights in native code
(lane cpu2-l10-linear, 2026-10-04).

The linear lane's Python layer counted the classes (`collections.Counter`,
a `counts[c] += w` loop), formed scikit-learn's 'balanced' weights and
multiplied every row's sample weight by its class's weight in Python lists:
data work on a GPU route (SGDClassifier, RidgeClassifier and
LogisticRegressionCV with class_weight, LogisticRegressionCV's StratifiedKFold
size check). One door now does it, `x_linear_class_prep`, on the device on a
GPU binding (`x_linear/class_prep_device.mojo`, a grid per step) and in a
host loop on the CPU binding, from the per-thread functions below, so every
column computes the same words in the same order:

  1. parts   (row block b, class c): the rows of block b (FOLD_BLOCK rows,
             ascending) whose code is c: their count (Int) and, when the
             counts are weighted, the fold `fa(acc, w_i)` from 0.
  2. counts  class c: the block parts folded in ascending block order (the
             integer count exactly; the weighted count by `fa` from 0).
  3. weights class c ('balanced'): total / (k * count_c), 0 for an empty
             class. Unweighted: i2f(n) / i2f(k * count_c) (the integers
             exact). Weighted: total = the class counts folded ascending by
             `fa`, then fd(total, fm(i2f(k), count_c)). Also the largest
             unweighted class count (one word Python reads).
  4. rows    row i: fm(w_i, cw[code_i]), w_i the sample weight (1 without).

The bits are not the old Python float64 ones (old bits do not matter); every
column computes these. This file has no device import: both bindings read it.
"""

from x_linear.ops import FP, IP, fa, fm, fd, ld, st, ldi, sti, i2f
from x_linear.tops import FOLD_BLOCK, fold_blocks

comptime CP_BALANCED = 1
"""dims[2]: the class weights are 'balanced' (written into cw); else cw is
the caller's (a dict's weights), read only."""


@always_inline
def cp_part_count(codes: IP, n: Int, b: Int, c: Int) -> Int:
    """Step 1, the count: rows of block b with code c."""
    var lo = b * FOLD_BLOCK
    var hi = min(lo + FOLD_BLOCK, n)
    var cnt = 0
    for i in range(lo, hi):
        if ldi(codes, i) == c:
            cnt += 1
    return cnt


@always_inline
def cp_part_wsum(codes: IP, w: FP, n: Int, b: Int, c: Int) -> Float32:
    """Step 1, the weighted count: fa(acc, w_i) over block b's rows of code
    c, ascending, from 0."""
    var lo = b * FOLD_BLOCK
    var hi = min(lo + FOLD_BLOCK, n)
    var acc = Float32(0)
    for i in range(lo, hi):
        if ldi(codes, i) == c:
            acc = fa(acc, ld(w, i))
    return acc


@always_inline
def cp_class_count(icnt_parts: IP, nb: Int, k: Int, c: Int) -> Int:
    """Step 2, the count: class c's block parts (layout b * k + c) summed."""
    var s = 0
    for b in range(nb):
        s += ldi(icnt_parts, b * k + c)
    return s


@always_inline
def cp_class_wsum(wcnt_parts: FP, nb: Int, k: Int, c: Int) -> Float32:
    """Step 2, the weighted count: class c's block parts folded ascending."""
    var acc = Float32(0)
    for b in range(nb):
        acc = fa(acc, ld(wcnt_parts, b * k + c))
    return acc


@always_inline
def cp_balanced_weight(icnt: IP, wcnt: FP, weighted: Bool, n: Int, k: Int, c: Int) -> Float32:
    """Step 3: class c's 'balanced' weight (see the header)."""
    if weighted:
        var cnt = ld(wcnt, c)
        if cnt == Float32(0):
            return Float32(0)
        var total = Float32(0)
        for j in range(k):
            total = fa(total, ld(wcnt, j))
        return fd(total, fm(i2f(k), cnt))
    var ic = ldi(icnt, c)
    if ic == 0:
        return Float32(0)
    return fd(i2f(n), i2f(k * ic))


@always_inline
def cp_largest(icnt: IP, k: Int) -> Int:
    """Step 3: the largest unweighted class count."""
    var m = 0
    for c in range(k):
        m = max(m, ldi(icnt, c))
    return m


@always_inline
def cp_row_weight(codes: IP, sw: FP, has_sw: Bool, cw: FP, i: Int) -> Float32:
    """Step 4: row i's weight, its sample weight times its class's weight."""
    var w = ld(sw, i) if has_sw else Float32(1)
    return fm(w, ld(cw, ldi(codes, i)))


def class_prep_host(codes: IP, sw: FP, has_sw: Bool, cw: FP, n: Int, k: Int, balanced: Bool,
                    weighted: Bool, rows_out: FP, has_rows: Bool) -> Int:
    """The CPU column: steps 1-4 in host loops over the same functions.
    Returns the largest unweighted class count."""
    var nb = fold_blocks(n)
    var ip = List[Int32](length=max(nb * k, 1), fill=Int32(0))
    var wp = List[Float32](length=max(nb * k, 1), fill=Float32(0))
    var ic = List[Int32](length=max(k, 1), fill=Int32(0))
    var wc = List[Float32](length=max(k, 1), fill=Float32(0))
    var ipp = IP(unsafe_from_address=Int(ip.unsafe_ptr()))
    var wpp = FP(unsafe_from_address=Int(wp.unsafe_ptr()))
    var icp = IP(unsafe_from_address=Int(ic.unsafe_ptr()))
    var wcp = FP(unsafe_from_address=Int(wc.unsafe_ptr()))
    for b in range(nb):
        for c in range(k):
            sti(ipp, b * k + c, cp_part_count(codes, n, b, c))
            if weighted:
                st(wpp, b * k + c, cp_part_wsum(codes, sw, n, b, c))
    for c in range(k):
        sti(icp, c, cp_class_count(ipp, nb, k, c))
        if weighted:
            st(wcp, c, cp_class_wsum(wpp, nb, k, c))
    if balanced:
        for c in range(k):
            st(cw, c, cp_balanced_weight(icp, wcp, weighted, n, k, c))
    var largest = cp_largest(icp, k)
    if has_rows:
        for i in range(n):
            st(rows_out, i, cp_row_weight(codes, sw, has_sw, cw, i))
    _ = ip^
    _ = wp^
    _ = ic^
    _ = wc^
    return largest
