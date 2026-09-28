# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The model_selection splitters' one random draw: a permutation of 0..n-1.

DEVIATION 6108 (THE RNG MAPPING). scikit-learn draws its shuffles from
numpy's Mersenne Twister (`rng.permutation`, `rng.shuffle`, `rng.choice`),
whose stream no GPU reproduces. Here every permutation is the SORT of the
row indices by a counter-based key, `splitmix_pair(i, salt)` from
checks/fixture_rng.mojo, ties (none in practice: the key is a bijection of
i for a fixed salt) broken by the index. The permutation is a pure function
of (n, salt) on every column; the Python side derives `salt` from the
caller's `random_state` and the draw's position in the splitter's sequence.
A heapsort gives the same answer as any correct sort of a strict order.
"""
from x_metrics.common import FP, IP, p, ldi, sti
from checks.fixture_rng import splitmix_pair


@always_inline
def _salt(q: IP) -> Int:
    var lo = UInt64(UInt32(p(q, 2)))
    var hi = UInt64(UInt32(p(q, 3)))
    return Int((hi << 32) | lo)


@always_inline
def _before(salt: Int, i: Int, j: Int) -> Bool:
    var ki = splitmix_pair(i, salt)
    var kj = splitmix_pair(j, salt)
    return ki < kj or (ki == kj and i < j)


def _sift(f: FP, O: Int, salt: Int, start: Int, n: Int):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= n:
            return
        if child + 1 < n and _before(salt, ldi(f, O + child), ldi(f, O + child + 1)):
            child += 1
        if _before(salt, ldi(f, O + root), ldi(f, O + child)):
            var tmp = ldi(f, O + root)
            sti(f, O + root, ldi(f, O + child))
            sti(f, O + child, tmp)
            root = child
        else:
            return


def permute_unit(t: Int, f: FP, q: IP):
    """q = [n, OUT, salt_lo, salt_hi]; t = 0. OUT[0..n) = the permutation."""
    if t != 0:
        return
    var n = p(q, 0)
    var O = p(q, 1)
    var salt = _salt(q)
    for i in range(n):
        sti(f, O + i, i)
    if n < 2:
        return
    var start = n // 2 - 1
    while start >= 0:
        _sift(f, O, salt, start, n)
        start -= 1
    var end = n - 1
    while end > 0:
        var tmp = ldi(f, O)
        sti(f, O, ldi(f, O + end))
        sti(f, O + end, tmp)
        _sift(f, O, salt, 0, end)
        end -= 1


@always_inline
def fold_of_position(i: Int, n: Int, K: Int) -> Int:
    """KFold's contiguous folds of a row order: the first n % K folds hold
    n // K + 1 positions, the rest n // K (lane metrics-apple2)."""
    var qn = n // K
    var r = n - qn * K
    var big = r * (qn + 1)
    if i < big:
        return i // (qn + 1)
    return r + (i - big) // qn


@always_inline
def st_row64(f: FP, at: Int, row: Int):
    """A row index as a little-endian Int64 in two words (low, high = 0):
    the Python side reads the words straight into an array('q')."""
    sti(f, at, row)
    sti(f, at + 1, 0)


def fold_rows_unit(t: Int, f: FP, q: IP):
    """q = [n, K, CODE, ORD, OUT, SZ]; t = 0 (lane metrics-apple2). Every
    fold's (test, train) rows of a K-fold split, as Int64 words.

    ORD >= 0: CODE[ORD[i]] = the contiguous fold of position i first
    (KFold over the row order ORD); ORD < 0: CODE holds each row's fold
    (StratifiedKFold's test_folds). A code outside [0, K) is a train row of
    every fold. Then for each fold f: SZ[f] = its test count c, and
    OUT[2nf ..) = its test rows ascending, then its train rows ascending,
    each row two words (`st_row64`). The planner replaces this one walk
    with fr_scatter / fr_cnt / fr_off / fr_fill (x_metrics/par.mojo): a
    stable partition's output is unique, so the words are these."""
    if t != 0:
        return
    var n = p(q, 0)
    var K = p(q, 1)
    var CODE = p(q, 2)
    var ORD = p(q, 3)
    var OUT = p(q, 4)
    var SZ = p(q, 5)
    if ORD >= 0:
        for i in range(n):
            sti(f, CODE + ldi(f, ORD + i), fold_of_position(i, n, K))
    for fo in range(K):
        var c = 0
        for i in range(n):
            if ldi(f, CODE + i) == fo:
                c += 1
        sti(f, SZ + fo, c)
        var base = OUT + 2 * n * fo
        var a = 0
        var b = c
        for i in range(n):
            if ldi(f, CODE + i) == fo:
                st_row64(f, base + 2 * a, i)
                a += 1
            else:
                st_row64(f, base + 2 * b, i)
                b += 1


def rows64_unit(t: Int, f: FP, q: IP):
    """q = [IN, OUT]; t = row: the Int32 word IN[t] as the Int64 words
    OUT[2t], OUT[2t + 1] (`st_row64`; lane metrics-apple2), so a
    permutation reaches Python as an array('q') without a Python int per
    row. One unit per row; nothing to plan."""
    st_row64(f, p(q, 1) + 2 * t, ldi(f, p(q, 0) + t))
