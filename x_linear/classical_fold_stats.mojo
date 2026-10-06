# SPDX-License-Identifier: Apache-2.0
"""C13 disjoint-fold centered sufficient statistics, shared host/device cells.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.

Each fold owns one count, d+1 means, and an augmented centered Gram.
Training means and moments combine retained fold IDs ascending; no
full-minus-held-out subtraction and no held-out value enters a training sum.
"""
from x_linear.ops import FP, ld, st, fa, fs, fm, fd, fmad, i2f


def fold_stat_words(d: Int) -> Int:
    return 1 + d + 1 + (d + 1) * (d + 1)


def fold_value(x: FP, y: FP, row: Int, col: Int, d: Int) -> Float32:
    return ld(y, row) if col == d else ld(x, row * d + col)


def fold_mean_cell(x: FP, y: FP, n: Int, d: Int, f: Int, col: Int, fi: Bool, cache: FP):
    var base = f * fold_stat_words(d)
    var count = 0
    var acc = Float32(0)
    for row in range(n):
        if Int(ld(y, n + row)) == f:
            count += 1
            if fi:
                acc = fa(acc, fold_value(x, y, row, col, d))
    st(cache, base + 1 + col, fd(acc, i2f(count)) if count > 0 and fi else Float32(0))
    if col == 0:
        st(cache, base, i2f(count))


def fold_gram_cell(x: FP, y: FP, n: Int, d: Int, f: Int, i: Int, j: Int, cache: FP):
    var base = f * fold_stat_words(d)
    var mi = ld(cache, base + 1 + i)
    var mj = ld(cache, base + 1 + j)
    var acc = Float32(0)
    for row in range(n):
        if Int(ld(y, n + row)) == f:
            acc = fmad(fs(fold_value(x, y, row, i, d), mi), fs(fold_value(x, y, row, j, d), mj), acc)
    st(cache, base + d + 2 + i * (d + 1) + j, acc)
    st(cache, base + d + 2 + j * (d + 1) + i, acc)


def retained_count(cache: FP, d: Int, folds: Int, held: Int) -> Float32:
    var count = Float32(0)
    for f in range(folds):
        if f != held:
            count = fa(count, ld(cache, f * fold_stat_words(d)))
    return count


def retained_mean(cache: FP, d: Int, folds: Int, held: Int, col: Int) -> Float32:
    var acc = Float32(0)
    for f in range(folds):
        if f != held:
            var base = f * fold_stat_words(d)
            acc = fmad(ld(cache, base), ld(cache, base + 1 + col), acc)
    var count = retained_count(cache, d, folds, held)
    return fd(acc, count) if count > 0 else Float32(0)


def retained_cross(cache: FP, d: Int, folds: Int, held: Int, i: Int, j: Int) -> Float32:
    var mi = retained_mean(cache, d, folds, held, i)
    var mj = retained_mean(cache, d, folds, held, j)
    var acc = Float32(0)
    for f in range(folds):
        if f != held:
            var base = f * fold_stat_words(d)
            var shift = fm(ld(cache, base), fs(ld(cache, base + 1 + i), mi))
            var value = fmad(shift, fs(ld(cache, base + 1 + j), mj), ld(cache, base + d + 2 + i * (d + 1) + j))
            acc = fa(acc, value)
    return acc


def fold_prep_from_cache(cache: FP, d: Int, folds: Int, held: Int, fw: FP, xm: Int, gg: Int, q: Int, sc: Int):
    for i in range(d):
        st(fw, xm + i, retained_mean(cache, d, folds, held, i))
        st(fw, q + i, retained_cross(cache, d, folds, held, i, d))
        for j in range(i, d):
            var v = retained_cross(cache, d, folds, held, i, j)
            st(fw, gg + i * d + j, v)
            st(fw, gg + j * d + i, v)
    st(fw, sc, retained_mean(cache, d, folds, held, d))
    st(fw, sc + 1, retained_cross(cache, d, folds, held, d, d))
    st(fw, sc + 2, retained_count(cache, d, folds, held))


@always_inline
def _fold_two_sum(mut value: Float32, mut carry: Float32, word: Float32):
    var total = fa(value, word)
    var bp = fs(total, value)
    var ap = fs(total, bp)
    carry = fa(carry, fa(fs(value, ap), fs(word, bp)))
    value = total


def kfold_mean_cell(x: FP, y: FP, n: Int, d: Int, folds: Int, f: Int, col: Int, fi: Bool, cache: FP):
    """RidgeCV's contiguous folds use compensated within-fold sums on every column."""
    var lo = f * (n // folds) + min(f, n % folds)
    var hi = lo + n // folds + (1 if f < n % folds else 0)
    var base = f * fold_stat_words(d)
    var acc = Float32(0)
    var cc = Float32(0)
    if fi:
        for row in range(lo, hi):
            _fold_two_sum(acc, cc, fold_value(x, y, row, col, d))
    st(cache, base + 1 + col, fd(fa(acc, cc), i2f(hi - lo)) if fi and hi > lo else Float32(0))
    if col == 0:
        st(cache, base, i2f(hi - lo))


def kfold_gram_cell(x: FP, y: FP, n: Int, d: Int, folds: Int, f: Int, i: Int, j: Int, cache: FP):
    var lo = f * (n // folds) + min(f, n % folds)
    var hi = lo + n // folds + (1 if f < n % folds else 0)
    var base = f * fold_stat_words(d)
    var mi = ld(cache, base + 1 + i)
    var mj = ld(cache, base + 1 + j)
    var acc = Float32(0)
    var cc = Float32(0)
    for row in range(lo, hi):
        _fold_two_sum(acc, cc, fm(fs(fold_value(x, y, row, i, d), mi), fs(fold_value(x, y, row, j, d), mj)))
    var value = fa(acc, cc)
    st(cache, base + d + 2 + i * (d + 1) + j, value)
    st(cache, base + d + 2 + j * (d + 1) + i, value)


def kfold_combine_unit(u: Int, cache: FP, d: Int, folds: Int, held: Int, xm: FP, g: FP, xty: FP):
    if u < d + 1:
        st(xm, u, retained_mean(cache, d, folds, held, u))
    elif u < d + 1 + d * d:
        var cell = u - d - 1
        var i = cell // d
        var j = cell % d
        if j >= i:
            var value = retained_cross(cache, d, folds, held, i, j)
            st(g, i * d + j, value)
            st(g, j * d + i, value)
    else:
        var j = u - d - 1 - d * d
        if j < d:
            st(xty, j, retained_cross(cache, d, folds, held, j, d))
