# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TargetEncoder's cross-fit fold assignment on the host, in integer
arithmetic (lane prep-apple2): the SAME rows -> folds as
python/mojolearn/_expansion_prep.py `_kfold_assignment` and
`_stratified_assignment` (splitmix64 Fisher-Yates, numpy KFold's split and
StratifiedKFold's `_make_test_folds`), which spent 0.4 s of a 1M row
fit_transform in the Python loop. No float is involved, so no bit can move;
the Python spelling stays the reference (and the route of a binding without
these entries)."""

comptime I32P = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _splitmix64(mut state: UInt64) -> UInt64:
    state = state + UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def kfold_folds(n: Int, n_folds: Int, seed: UInt64, shuffle: Bool, fold: I32P):
    """`_kfold_assignment`: fold[r] for every row r."""
    var perm = List[Int32](capacity=n)
    for i in range(n):
        perm.append(Int32(i))
    var state = seed
    if shuffle:
        for i in range(n - 1, 0, -1):
            var z = _splitmix64(state)
            var j = Int(z % UInt64(i + 1))
            var t = perm[i]
            perm[i] = perm[j]
            perm[j] = t
    var start = 0
    for k in range(n_folds):
        var size = n // n_folds + (1 if k < n % n_folds else 0)
        for r in range(start, start + size):
            fold[Int(perm[r])] = Int32(k)
        start += size


def strat_folds(codes: I32P, n: Int, ncls: Int, n_folds: Int, seed: UInt64, shuffle: Bool, fold: I32P) -> Int:
    """`_stratified_assignment` for labels given as codes in [0, ncls): fold[r]
    for every row r. Returns 0, or -1 when every class has fewer rows than
    n_folds (the caller raises the reference's error)."""
    # classes renumbered by first appearance
    var first = List[Int](length=ncls, fill=-1)
    var enc = List[Int](capacity=n)
    var K = 0
    for i in range(n):
        var c = Int(codes[i])
        if first[c] < 0:
            first[c] = K
            K += 1
        enc.append(first[c])
    var counts = List[Int](length=K, fill=0)
    for i in range(n):
        counts[enc[i]] += 1
    var all_small = True
    for k in range(K):
        if not (n_folds > counts[k]):
            all_small = False
    if all_small:
        return -1
    # alloc[f][e]: the positions p = f, f + n_folds, .. of sorted(enc) holding e
    var alloc = List[Int](length=n_folds * K, fill=0)
    var p = 0
    for e in range(K):
        for _ in range(counts[e]):
            alloc[(p % n_folds) * K + e] += 1
            p += 1
    # each class's rows in row order (a stable counting sort)
    var start = List[Int](length=K + 1, fill=0)
    for e in range(K):
        start[e + 1] = start[e] + counts[e]
    var at = start.copy()
    var rows = List[Int](length=n, fill=0)
    for i in range(n):
        rows[at[enc[i]]] = i
        at[enc[i]] += 1
    var state = seed
    var block = List[Int32](capacity=n)
    for k in range(K):
        block.clear()
        for f in range(n_folds):
            for _ in range(alloc[f * K + k]):
                block.append(Int32(f))
        if shuffle:
            for i in range(len(block) - 1, 0, -1):
                var z = _splitmix64(state)
                var j = Int(z % UInt64(i + 1))
                var t = block[i]
                block[i] = block[j]
                block[j] = t
        for r in range(len(block)):
            fold[rows[start[k] + r]] = block[r]
    return 0
