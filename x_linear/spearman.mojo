# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The sign of Spearman's rho in native code (lane cpu2-l10-linear, 2026-10-04).

IsotonicRegression(increasing='auto') asks for the sign of Spearman's rho of
(x, y) (scikit-learn's `check_increasing`). The Python layer ranked both
columns (a Python sort, average ranks for ties) and summed the centered
products in float64: data work on a GPU route. `x_linear_spearman_sign`
now answers it, on the device on a GPU binding (x_linear/spearman_device.mojo:
the isotonic block radix sort, group starts by a scan, a grid of block sums)
and on the CPU binding (`spearman_sign_host`).

The answer is EXACT on every column, so no fold order can move it: with
R_i = 2 * (average rank of row i) = (first + last sorted position of its tie
group), an integer in [0, 2n - 2], and the mean of R exactly n - 1,

    rho's sign = sign( sum_i (Rx_i - (n - 1)) * (Ry_i - (n - 1)) ),

each product an Int64 (|.| < 4 n^2), each FOLD_BLOCK block's sum an Int64,
and the blocks' sum carried in two Int64 words (`sp_fold_sign`). Ties are
equal float32 values, -0 and +0 one value (`sp_key`); a NaN sorts by its bits
and is harmless here: the isotonic fit that follows refuses it.
This file has no device import: both bindings read it.
"""

from std.memory import bitcast
from x_linear.ops import FP, IP, ld, ldi, sti
from x_linear.tops import FOLD_BLOCK, fold_blocks

comptime I64P = MutPointer[Int64, MutAnyOrigin]


@always_inline
def sp_key(v: Float32) -> Int32:
    """An order-preserving 32-bit key (as unsigned bits) of a float32; -0
    and +0 one key (the isotonic fit's `_iso_key`)."""
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x7FFFFFFF)) == UInt32(0):
        b = UInt32(0)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return bitcast[DType.int32](~b)
    return bitcast[DType.int32](b | UInt32(0x80000000))


@always_inline
def sp_block_sum(rx: IP, ry: IP, n: Int, b: Int) -> Int64:
    """Block b's sum of (Rx_i - (n - 1)) * (Ry_i - (n - 1))."""
    var lo = b * FOLD_BLOCK
    var hi = min(lo + FOLD_BLOCK, n)
    var m = Int64(n - 1)
    var acc = Int64(0)
    for i in range(lo, hi):
        acc += (Int64(ldi(rx, i)) - m) * (Int64(ldi(ry, i)) - m)
    return acc


@always_inline
def sp_fold_sign(parts: I64P, nb: Int) -> Int:
    """The sign (-1, 0, 1) of the sum of the nb block sums, exact: each part
    split into its high word (arithmetic shift) and its low 32 bits."""
    var hi = Int64(0)
    var lo = Int64(0)
    for b in range(nb):
        var p = parts.unsafe_load(b)
        hi += p >> 32
        lo += p & Int64(0xFFFFFFFF)
    hi += lo >> 32
    lo = lo & Int64(0xFFFFFFFF)
    if hi > 0:
        return 1
    if hi < 0:
        return -1
    return 1 if lo > 0 else 0


def _sp_ranks_host(v: FP, n: Int, rank2: IP):
    """rank2[i] = first + last sorted position of row i's tie group: an LSD
    radix sort of the keys (four 8-bit passes, stable), then the groups."""
    var key = List[UInt32](length=n, fill=UInt32(0))
    var a = List[Int32](length=n, fill=Int32(0))
    var t = List[Int32](length=n, fill=Int32(0))
    for i in range(n):
        key[i] = bitcast[DType.uint32](sp_key(ld(v, i)))
        a[i] = Int32(i)
    for pas in range(4):
        var shift = UInt32(8 * pas)
        var cnt = List[Int](length=257, fill=0)
        for i in range(n):
            cnt[Int((key[Int(a[i])] >> shift) & UInt32(255)) + 1] += 1
        for d in range(256):
            cnt[d + 1] += cnt[d]
        for i in range(n):
            var d = Int((key[Int(a[i])] >> shift) & UInt32(255))
            t[cnt[d]] = a[i]
            cnt[d] += 1
        for i in range(n):
            a[i] = t[i]
    var s = 0
    while s < n:
        var e = s
        while e + 1 < n and key[Int(a[e + 1])] == key[Int(a[s])]:
            e += 1
        for p in range(s, e + 1):
            sti(rank2, Int(a[p]), s + e)
        s = e + 1


def spearman_sign_host(x: FP, y: FP, n: Int) -> Int:
    """The CPU column: the same exact sign (see the header)."""
    var rx = List[Int32](length=max(n, 1), fill=Int32(0))
    var ry = List[Int32](length=max(n, 1), fill=Int32(0))
    var rxp = IP(unsafe_from_address=Int(rx.unsafe_ptr()))
    var ryp = IP(unsafe_from_address=Int(ry.unsafe_ptr()))
    _sp_ranks_host(x, n, rxp)
    _sp_ranks_host(y, n, ryp)
    var nb = fold_blocks(n)
    var parts = List[Int64](length=max(nb, 1), fill=Int64(0))
    var pp = I64P(unsafe_from_address=Int(parts.unsafe_ptr()))
    for b in range(nb):
        pp.unsafe_store(b, sp_block_sum(rxp, ryp, n, b))
    var sign = sp_fold_sign(pp, nb)
    _ = rx^
    _ = ry^
    _ = parts^
    return sign
