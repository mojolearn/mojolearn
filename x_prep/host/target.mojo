# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host's `te_enc` (op 21 of x_prep/units.mojo), lane prep-cpu
2026-09-28: `te_enc_unit`'s words (x_prep/target.mojo), one pass over the
rows per (fold, feature, target column) instead of one per category.

The device's unit for category `cat` walks every row ascending and folds the
target of each row whose code is `cat` (and whose fold is not `fi`) into its
sum, then (the "auto" encoding) the squared deviations the same way. This
path walks the rows ascending ONCE and folds each row into ITS category's
accumulators: every category still sees exactly its own rows, in ascending
order, through the same `add` / `sub` / `mul`, so every sum, count and
squared deviation is the device's word, and `te_value` (shared) finishes
each category as the unit does. A code outside [0, CMAX) matches no
category on the device (the unknown code -1), and is skipped here. Work per
(fold, feature, target column): O(n + CMAX), where the device's is
O(n * CMAX).
"""
from x_prep.common import FP, IP, p, ld, st, sti
from x_prep.prims import add, sub, mul, div
from x_prep.target import te_value, te_enc_lt_fold
from x_prep.idn_fold import IDN_TE_ENC_TREE


@always_inline
def te_enc_host_groups(total: Int, q: IP) -> Int:
    """The number of (fold, feature, target column) groups of a te_enc stage
    of `total` units."""
    var cmax = p(q, 6)
    if cmax <= 0:
        return 0
    return total // cmax


def te_enc_host_group(g: Int, f: FP, q: IP):
    """q as `te_enc_unit`; g = (fi*d + j)*T + tt: every category's unit
    ((fi*d + j)*CMAX + cat)*T + tt of that group."""
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var cmax = p(q, 6)
    var tt = g % T
    var fj = g // T
    var j = fj % d
    var fi = fj // d
    var ncat = Int(ld(f, p(q, 7) + j))
    if ncat > cmax:
        ncat = cmax
    if ncat <= 0:
        return
    var ymean = ld(f, p(q, 8) + 2 * (fi * T + tt))
    var yvar = ld(f, p(q, 8) + 2 * (fi * T + tt) + 1)
    var smooth = ld(f, p(q, 9))
    comptime if IDN_TE_ENC_TREE:
        te_enc_host_group_lt(fj, fi, j, tt, ncat, ymean, yvar, smooth, f, q)
        return
    var s = List[Float32](length=ncat, fill=Float32(0))
    var cnt = List[Int](length=ncat, fill=0)
    for i in range(n):
        if Int(ld(f, p(q, 5) + i)) == fi:
            continue
        var code = Int(ld(f, p(q, 0) + i * d + j))
        if code < 0 or code >= ncat:
            continue
        s[code] = add(s[code], ld(f, p(q, 3) + i * T + tt))
        cnt[code] += 1
    var mean = List[Float32](length=ncat, fill=Float32(0))
    var ssd = List[Float32](length=ncat, fill=Float32(0))
    if smooth < Float32(0):
        for cat in range(ncat):
            if cnt[cat] > 0:
                mean[cat] = div(s[cat], Float32(cnt[cat]))
        for i in range(n):
            if Int(ld(f, p(q, 5) + i)) == fi:
                continue
            var code = Int(ld(f, p(q, 0) + i * d + j))
            if code < 0 or code >= ncat:
                continue
            var e = sub(ld(f, p(q, 3) + i * T + tt), mean[code])
            ssd[code] = add(ssd[code], mul(e, e))
    for cat in range(ncat):
        st(f, p(q, 10) + (fj * cmax + cat) * T + tt,
           te_value(ymean, yvar, smooth, s[cat], cnt[cat], mean[cat], ssd[cat]))


def te_enc_host_group_lt(fj: Int, fi: Int, j: Int, tt: Int, ncat: Int, ymean: Float32, yvar: Float32,
                         smooth: Float32, f: FP, q: IP):
    """IDN_TE_ENC_TREE (lane ml-prep-nb): the group's categories in the
    lane-tree order (x_prep/idn_fold.mojo). A row's rank among its category's
    rows (all folds, ascending) is its position in a counting-sort bucket
    built here (the device's `te_bucket`: O(n + NCAT) words), and
    `te_enc_lt_fold` (shared with the unit) folds each bucket, so every
    category's sum, count, mean and squared deviation is the device's word."""
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var cmax = p(q, 6)
    var start = List[Int](length=ncat + 1, fill=0)
    for i in range(n):
        var code = Int(ld(f, p(q, 0) + i * d + j))
        if code >= 0 and code < ncat:
            start[code + 1] += 1
    for c in range(ncat):
        start[c + 1] += start[c]
    var put = List[Int](length=ncat, fill=0)
    # the bucket's row indices as int words (te_enc_lt_fold reads them with ldi)
    var rows = List[Float32](length=max(start[ncat], 1), fill=Float32(0))
    var rp = FP(unsafe_from_address=Int(rows.unsafe_ptr()))
    for i in range(n):
        var code = Int(ld(f, p(q, 0) + i * d + j))
        if code >= 0 and code < ncat:
            sti(rp, start[code] + put[code], i)
            put[code] += 1
    for cat in range(ncat):
        var s = Float32(0)
        var cnt = 0
        var mean = Float32(0)
        var ssd = Float32(0)
        te_enc_lt_fold(f, rp, -1, 0, 0, p(q, 3), T, tt, p(q, 5), start[cat], start[cat + 1], fi, smooth, s, cnt,
                       mean, ssd)
        st(f, p(q, 10) + (fj * cmax + cat) * T + tt, te_value(ymean, yvar, smooth, s, cnt, mean, ssd))
    # rows backs rp: keep it alive past the last fold
    _ = rows^
