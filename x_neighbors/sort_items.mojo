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
from checks.soft_f64 import sf64_add, sf64_div, sf64_from_f32, sf64_to_f32
from std.memory import bitcast
from std.python import PythonObject
from std.sys.compile import is_defined
from x_neighbors.items import XN_FOLD_BLOCK

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _nc_key(v: Float32) -> Int32:
    """The value's order as an integer: ascending float order on the word's
    bits, both zeros one key (as a float compare). An integer compare,
    because a float compare on the Apple GPU reads a subnormal operand as
    zero, so subnormals tied there and sorted by row while every other
    target sorted them by value (x-neighbors-nearest-centroid/denormal
    manhattan, the 0.8.36 reference recording)."""
    var b = bitcast[DType.int32](v)
    if (b & Int32(0x7FFFFFFF)) == 0:
        return Int32(0)
    return b ^ Int32(0x7FFFFFFF) if b < 0 else b


@always_inline
def _nc_less(a: Int, b: Int, x: FP, lab: IP, n: Int, d: Int, f: Int) -> Bool:
    """Key order (class, value, row); the padding row id n sorts last. The
    value compares by `_nc_key`, never as floats."""
    if a == n:
        return False
    if b == n:
        return True
    var la = lab.unsafe_load(a)
    var lb = lab.unsafe_load(b)
    if la != lb:
        return la < lb
    var va = _nc_key(x.unsafe_load(a * d + f))
    var vb = _nc_key(x.unsafe_load(b * d + f))
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


# ---- NearestCentroid's median within-class std (lane apple-fast-purity) ----
# `fit` took `stats.tolist()`, a Python sort of the d stds and the median on
# the host. Here the d values padded with NaN to a power of two `p` are
# sorted ascending by a bitonic network (non-NaN values ascending, NaN last,
# so a NaN std lands past index d - 1 just as it fails Python's `v == 0`);
# one item then reads the middle one or two values: the mean of two is
# Python's `(a + b) / 2.0` in binary64 rounded once to float32 (soft binary64,
# checks/soft_f64.mojo, the same words on every column). res = [median,
# 1.0 if every std is zero else 0.0].


@always_inline
def _ms_less(a: Float32, b: Float32) -> Bool:
    if a != a:
        return False
    if b != b:
        return True
    return a < b


def nc_med_std_init_item(t: Int, std: FP, res: FP, key: FP, d: Int, p: Int, n_steps: Int):
    """Stage 1, t < p: key[t] = std[t], NaN past d."""
    if t < d:
        key.unsafe_store(t, std.unsafe_load(t))
    else:
        key.unsafe_store(t, bitcast[DType.float32](UInt32(0x7FC00000)))


def nc_med_std_step_item(t: Int, j: Int, std: FP, res: FP, key: FP, d: Int, p: Int, n_steps: Int):
    """Stage 2, network step j (`nc_median_step_item`'s schedule), one item
    per compare pair t < p / 2."""
    var kk = 1
    var rem = j
    while rem >= kk:
        rem -= kk
        kk += 1
    var jj = kk - 1 - rem
    var stride = 1 << jj
    var lo = ((t >> jj) << (jj + 1)) | (t & (stride - 1))
    var hi = lo + stride
    var up = (lo & (1 << kk)) == 0
    var a = key.unsafe_load(lo)
    var b = key.unsafe_load(hi)
    var swap = _ms_less(b, a) if up else _ms_less(a, b)
    if swap:
        key.unsafe_store(lo, b)
        key.unsafe_store(hi, a)


def nc_med_std_pick_item(t: Int, std: FP, res: FP, key: FP, d: Int, p: Int, n_steps: Int):
    """Stage 3, ONE item of constant work: the median and the all-zero flag."""
    var med: Float32
    if d % 2 == 1:
        med = key.unsafe_load(d // 2)
    else:
        var s = sf64_add(sf64_from_f32(key.unsafe_load(d // 2 - 1)), sf64_from_f32(key.unsafe_load(d // 2)))
        med = sf64_to_f32(sf64_div(s, UInt64(0x4000000000000000)))
    res.unsafe_store(0, med)
    var z = key.unsafe_load(0) == Float32(0) and key.unsafe_load(d - 1) == Float32(0)
    res.unsafe_store(1, Float32(1) if z else Float32(0))


def purity_flags_binding() raises -> PythonObject:
    """lane apple-fast-purity switches the Python layer reads
    (`x_neighbors_purity_flags`): bit 1 the per-row argmax on the device
    (off under -D MOJOLEARN_PURITY_3_OFF), bit 2 NearestCentroid's median
    std on the device (off under -D MOJOLEARN_PURITY_6_OFF)."""
    var f = 0
    comptime if not is_defined["MOJOLEARN_PURITY_3_OFF"]():
        f |= 1
    comptime if not is_defined["MOJOLEARN_PURITY_6_OFF"]():
        f |= 2
    return PythonObject(f)
