# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Items for the neighbors lane's two host-loop replacements (cpu-gpu-cleanup
w2-pyglue, 2026-10-02), in the `items.mojo` contract: one item is one device
thread's work, the host driver runs the same items in the same order, so the
CPU column and every GPU column store the same bits.

NearestCentroid's manhattan centroids (`nc_median`): the per-class,
per-feature median, which `_expansion_neighbors.py` took by a Python sort of
every class's members per feature. Here a SEGMENTED SORT on the device: for
every feature, the row ids padded to a power of two `p` are sorted by the key
(class code, value, row) with a bitonic network (`nc_median_step_item`,
one launch per network step, one item per compare pair of every feature),
which puts each class's rows contiguous and in value order at the class's
offset `start[c]`; then one item per (class, feature) reads the middle one
or two values. The row tie-break makes every key distinct, so the sorted
order is unique and no launch shape can move it. Even counts give the mean
of the two middle values as `a/2 + b/2` (halving is exact above the FTZ
range), which cannot overflow.

OneClassSVM's sample weights (`pos_compact`): the rows with a positive
value, in row order, and their values, by a flag count per XN_FOLD_BLOCK
block, an exclusive scan of the block counts, and a per-block emit at the
block's scanned offset (the lane's blocked-fold shape). Integers and copies.

Nothing here imports a GPU module, so the CPU-only host binding compiles it.
"""
from checks.numerics import ftz
from x_neighbors.items import XN_FOLD_BLOCK

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _nc_less(a: Int, b: Int, x: FP, lab: IP, n: Int, d: Int, f: Int) -> Bool:
    """Key order (class, value, row); the padding row id n sorts last."""
    if a == n:
        return False
    if b == n:
        return True
    var la = lab.unsafe_load(a)
    var lb = lab.unsafe_load(b)
    if la != lb:
        return la < lb
    var va = x.unsafe_load(a * d + f)
    var vb = x.unsafe_load(b * d + f)
    if va != vb:
        return va < vb
    return a < b


def nc_median_init_item(
    t: Int, x: FP, lab: IP, start: IP, cent: FP, perm: IP,
    n: Int, d: Int, n_classes: Int, p: Int, n_steps: Int,
):
    """Stage 1, t = f * p + i: perm[f, i] = i, the padding slots n."""
    var i = t % p
    perm.unsafe_store(t, Int32(i if i < n else n))


def nc_median_step_item(
    t: Int, j: Int, x: FP, lab: IP, start: IP, cent: FP, perm: IP,
    n: Int, d: Int, n_classes: Int, p: Int, n_steps: Int,
):
    """Stage 2, launched for network step j = 0 .. n_steps - 1 (merge size
    2^kk for kk = 1, 2, ..., then strides 2^(kk-1) down to 1), one item per
    compare pair t = f * (p / 2) + q of every feature f."""
    var kk = 1
    var rem = j
    while rem >= kk:
        rem -= kk
        kk += 1
    var jj = kk - 1 - rem
    var stride = 1 << jj
    var half = p // 2
    var f = t // half
    var q = t - f * half
    var lo = ((q >> jj) << (jj + 1)) | (q & (stride - 1))
    var hi = lo + stride
    var up = (lo & (1 << kk)) == 0
    var base = f * p
    var a = Int(perm.unsafe_load(base + lo))
    var b = Int(perm.unsafe_load(base + hi))
    var swap = _nc_less(b, a, x, lab, n, d, f) if up else _nc_less(a, b, x, lab, n, d, f)
    if swap:
        perm.unsafe_store(base + lo, Int32(b))
        perm.unsafe_store(base + hi, Int32(a))


def nc_median_pick_item(
    t: Int, x: FP, lab: IP, start: IP, cent: FP, perm: IP,
    n: Int, d: Int, n_classes: Int, p: Int, n_steps: Int,
):
    """Stage 3, t = c * d + f: the median of class c's sorted values of
    feature f, at perm[f, start[c] : start[c + 1]]."""
    var c = t // d
    var f = t - c * d
    var s0 = Int(start.unsafe_load(c))
    var m = Int(start.unsafe_load(c + 1)) - s0
    var base = f * p + s0
    if m % 2 == 1:
        cent.unsafe_store(t, x.unsafe_load(Int(perm.unsafe_load(base + m // 2)) * d + f))
        return
    var a = x.unsafe_load(Int(perm.unsafe_load(base + m // 2 - 1)) * d + f)
    var b = x.unsafe_load(Int(perm.unsafe_load(base + m // 2)) * d + f)
    cent.unsafe_store(t, ftz(ftz(a * Float32(0.5)) + ftz(b * Float32(0.5))))


def pos_count_item(t: Int, w: FP, rows: IP, vals: FP, info: IP, part: IP, n: Int):
    """Stage 1, one item per XN_FOLD_BLOCK block t: its count of w > 0."""
    var lo = t * XN_FOLD_BLOCK
    var hi = min(lo + XN_FOLD_BLOCK, n)
    var c = Int32(0)
    for i in range(lo, hi):
        if w.unsafe_load(i) > Float32(0):
            c += 1
    part.unsafe_store(t, c)


def pos_scan_item(t: Int, w: FP, rows: IP, vals: FP, info: IP, part: IP, n: Int):
    """Stage 2, one item: the exclusive scan of the block counts (one per
    XN_FOLD_BLOCK rows) in place; the total to info[0]."""
    var nb = (n + XN_FOLD_BLOCK - 1) // XN_FOLD_BLOCK
    var run = Int32(0)
    for b in range(nb):
        var c = part.unsafe_load(b)
        part.unsafe_store(b, run)
        run += c
    info.unsafe_store(0, run)


def pos_emit_item(t: Int, w: FP, rows: IP, vals: FP, info: IP, part: IP, n: Int):
    """Stage 3, one item per block t: its rows with w > 0, in row order, at
    the block's scanned offset; their values beside them."""
    var lo = t * XN_FOLD_BLOCK
    var hi = min(lo + XN_FOLD_BLOCK, n)
    var at = Int(part.unsafe_load(t))
    for i in range(lo, hi):
        var v = w.unsafe_load(i)
        if v > Float32(0):
            rows.unsafe_store(at, Int32(i))
            vals.unsafe_store(at, v)
            at += 1
