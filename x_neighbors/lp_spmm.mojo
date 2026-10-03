# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Label propagation's sparse graph product (lane neighbors-apple2's
kernel; since lane/cgr-kernel the graph's nonzeros are found on the device
and the product reads them row-padded). A skipped term is fma(+-0, x, acc)
with x finite, whose product is a zero and whose sum is acc unchanged,
because acc starts at +0.0 and an fma returns -0.0 only from (-0) + (-0),
so acc is never -0.0: exact against the dense product while x is finite,
which `lp_nonfinite_kernel` tests every iteration."""
from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from checks.numerics import ftz, identical_mul_add
from x_neighbors.items import FP, IP, matmul_item


# ---- lane/cgr-kernel: G's nonzeros found on the device (row-padded, columns
# ascending) and the finiteness test on the device, so the fit loop never
# reads the graph or the distributions on the host. Before, the host
# scanned G (n x n) for its CSR and downloaded the distributions every
# iteration for the stopping sum and the finiteness test.


def lp_rowcount_kernel(g: FP, n_: Int64, cnt: IP, stats: IP):
    """Row i's nonzero count into cnt[i]; stats[0] += it, stats[1] =
    max(stats[1], it) (integer atomics: order-free)."""
    var n = Int(n_)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var c = 0
    for j in range(n):
        if g.unsafe_load(i * n + j) != Float32(0):
            c += 1
    cnt.unsafe_store(i, Int32(c))
    _ = Atomic.fetch_add[ordering = Ordering.RELAXED](stats, Int32(c))
    _ = Atomic[DType.int32].max(stats + 1, Int32(c))


def lp_ell_fill_kernel(g: FP, n_: Int64, maxk_: Int64, cols: IP, vals: FP):
    """Row i's nonzeros, columns ascending, at cols / vals [i * maxk ...]."""
    var n = Int(n_)
    var maxk = Int(maxk_)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var e = 0
    for j in range(n):
        var v = g.unsafe_load(i * n + j)
        if v != Float32(0):
            cols.unsafe_store(i * maxk + e, Int32(j))
            vals.unsafe_store(i * maxk + e, v)
            e += 1


def lp_nonfinite_kernel(x: FP, count_: Int64, flag: IP):
    """flag[0] = 1 when any x is Inf or NaN (the caller zeroes it first)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(count_):
        var bits = bitcast[DType.uint32](x.unsafe_load(t)) & UInt32(0x7F800000)
        if bits == UInt32(0x7F800000):
            flag.unsafe_store(0, Int32(1))


def lp_prod_kernel(
    flag: IP, cnt: IP, cols: IP, vals: FP, maxk_: Int64, g: FP, x: FP, res: FP, n_: Int64, c_: Int64,
):
    """G x: over G's nonzeros (the sparse fma chain, exact when x is
    finite), or `matmul_item` over every column when flag[0] says x holds
    an Inf or NaN. The same bits as the dense product either way."""
    var n = Int(n_)
    var c = Int(c_)
    var maxk = Int(maxk_)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n * c:
        return
    if flag.unsafe_load(0) != Int32(0):
        matmul_item(t, g, x, res, n, n, c)
        return
    var i = t // c
    var j = t - i * c
    var acc = Float32(0)
    for e in range(Int(cnt.unsafe_load(i))):
        var q = i * maxk + e
        acc = ftz(identical_mul_add(ftz(vals.unsafe_load(q)), ftz(x.unsafe_load(Int(cols.unsafe_load(q)) * c + j)), acc))
    res.unsafe_store(t, acc)
