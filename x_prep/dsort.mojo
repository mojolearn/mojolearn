# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device form of the `sort_cols` stage (lane prep-apple, 2026-09-28).

`sort_cols` (x_prep/prims.mojo) sorts every column of X by the lane's word
order. Its unit is one thread per column running a heapsort: at 1M rows that
is d GPU threads, each walking ~40M dependent loads, the slowest stage of every
estimator that sorts (RobustScaler, QuantileTransformer, KBinsDiscretizer,
SimpleImputer median / most_frequent, the encoders' categories,
SplineTransformer quantile knots).

A sort has ONE answer under a total order on the words, whatever algorithm
reaches it. The order here is `word_order` (x_prep/common.mojo): `key` first
(DEVIATION 5402), the raw bits second. On a non-NaN word `key` is a bijection
of the bits, so the second word only orders NaN words of different payload
among themselves (the tail no unit reads: every reader stops at the first NaN
or reads the first CNT entries). The host heapsort orders by the same pair,
so the arena is equal word for word, the NaN tail included.

The device therefore runs a bitonic sort instead of the heapsort: the column
is padded to a power of two N >= TILE with the word 0xFFFFFFFF (a NaN whose
order pair is the largest there is, so a pad ties only with that same word
and the first n words of the sorted column are the n sorted inputs), the
steps with a stride below TILE / 2 run inside one threadgroup in shared
memory (TILE words, 4 KB), the longer strides one global pass each. No
arithmetic, only comparisons and moves of words, so there is no reduction
order to keep: the result is the heapsort's, bit for bit, by construction.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz
from x_prep.common import FP, canon, word_order

comptime UP = MutPointer[UInt32, MutAnyOrigin]
#: words per threadgroup tile (a power of two) and threads per group
comptime TILE = 1024
comptime TG = TILE // 2
comptime PAD = UInt32(0xFFFFFFFF)


@always_inline
def _cx(w: UP, i: Int, l: Int, up: Bool):
    """compare-exchange w[i], w[l] (i < l) toward ascending when `up`."""
    var a = w[i]
    var b = w[l]
    var a_first = word_order(a) <= word_order(b)
    if a_first != up:
        w[i] = b
        w[l] = a


def sort_load_kernel(f: FP, w: UP, X: Int32, n: Int32, d: Int32, cn: Int32, big_n: Int32, total: Int32):
    """t = c * N + i: w[t] = the flushed (and with cn, canonical) word of
    X[i, c], or PAD past the column's n rows (the unit's own load: ftz, then
    canon)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var bn = Int(big_n)
    var c = t // bn
    var i = t - c * bn
    if i < Int(n):
        var v = ftz(f.unsafe_load(Int(X) + i * Int(d) + c))
        if cn != Int32(0):
            v = canon(v)
        w[t] = bitcast[DType.uint32](v)
    else:
        w[t] = PAD


def sort_store_kernel(f: FP, w: UP, S: Int32, n: Int32, big_n: Int32, total: Int32):
    """t = c * n + i: S[c*n + i] = w[c*N + i], moved as bits."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var nn = Int(n)
    var c = t // nn
    var i = t - c * nn
    var fu = f.bitcast[UInt32]()
    fu[Int(S) + t] = w[c * Int(big_n) + i]


def sort_global_kernel(w: UP, big_n: Int32, k: Int32, j: Int32, total: Int32):
    """One bitonic step of stride j inside merge size k, every column: t =
    c * N/2 + r, the pair (i, i + j) with i = 2j*(r // j) + r % j."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var half = Int(big_n) // 2
    var c = t // half
    var r = t - c * half
    var jj = Int(j)
    var i = 2 * jj * (r // jj) + (r % jj)
    var base = c * Int(big_n)
    _cx(w, base + i, base + i + jj, (i & Int(k)) == 0)


def sort_tile_kernel(w: UP, big_n: Int32, k_lo: Int32, k_hi: Int32, j_top: Int32):
    """Every step with stride j <= j_top for merge sizes k in [k_lo, k_hi]
    (powers of two), on one TILE of one column in shared memory: block b
    holds words [b*TILE, (b+1)*TILE) of the flat (column, row) array (N is a
    multiple of TILE, so a tile never straddles two columns and the row index
    is the flat index mod N). k_lo = 2, k_hi = TILE, j_top = TILE/2 is the
    whole sort of each tile; k_lo = k_hi = K, j_top = TILE/2 finishes merge
    size K after its global steps."""
    var s = stack_allocation[TILE, Scalar[DType.uint32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var g0 = Int(block_idx.x) * TILE
    var row0 = g0 % Int(big_n)
    s[tid] = w[g0 + tid]
    s[tid + TG] = w[g0 + tid + TG]
    barrier()
    var k = Int(k_lo)
    while k <= Int(k_hi):
        var jj = min(k // 2, Int(j_top))
        while jj >= 1:
            var i = 2 * jj * (tid // jj) + (tid % jj)
            var a = s[i]
            var b = s[i + jj]
            if (word_order(a) <= word_order(b)) != (((row0 + i) & k) == 0):
                s[i] = b
                s[i + jj] = a
            barrier()
            jj //= 2
        k *= 2
    w[g0 + tid] = s[tid]
    w[g0 + tid + TG] = s[tid + TG]


def _blocks(total: Int, bs: Int) -> Int:
    return (total + bs - 1) // bs


def sort_scratch_words(n: Int, cols: Int) -> Int:
    """The scratch words `sort_cols_device` needs for `cols` columns of n
    rows: cols * N."""
    var big_n = TILE
    while big_n < n:
        big_n *= 2
    return max(cols, 0) * big_n


def sort_cols_device(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint32],
                     cols: Int, X: Int, n: Int, d: Int, S: Int, cn: Int) raises:
    """Enqueue the sort of columns 0 .. cols-1 of X[n, d] into S[c*n : c*n+n]
    (the unit's layout; `cols` is the stage's unit count) on the device arena
    `f`, through the scratch `w` of at least sort_scratch_words(n, cols) words."""
    if n <= 0 or cols <= 0:
        return
    var f = df.unsafe_ptr()
    var w = dw.unsafe_ptr()
    var big_n = TILE
    while big_n < n:
        big_n *= 2
    var tot = cols * big_n
    if tot > 2 ** 31 - 1:
        raise Error("x_prep: sort_cols too large for the device sort")
    comptime BS = 256
    ctx.enqueue_function[sort_load_kernel](
        f, w, Int32(X), Int32(n), Int32(d), Int32(cn), Int32(big_n), Int32(tot),
        grid_dim=_blocks(tot, BS), block_dim=BS,
    )
    var tiles = tot // TILE
    ctx.enqueue_function[sort_tile_kernel](
        w, Int32(big_n), Int32(2), Int32(TILE), Int32(TG), grid_dim=tiles, block_dim=TG,
    )
    var pairs = tot // 2
    var k = 2 * TILE
    while k <= big_n:
        var j = k // 2
        while j >= TILE:
            ctx.enqueue_function[sort_global_kernel](
                w, Int32(big_n), Int32(k), Int32(j), Int32(pairs), grid_dim=_blocks(pairs, BS), block_dim=BS,
            )
            j //= 2
        ctx.enqueue_function[sort_tile_kernel](
            w, Int32(big_n), Int32(k), Int32(k), Int32(TG), grid_dim=tiles, block_dim=TG,
        )
        k *= 2
    var outs = cols * n
    ctx.enqueue_function[sort_store_kernel](
        f, w, Int32(S), Int32(n), Int32(big_n), Int32(outs), grid_dim=_blocks(outs, BS), block_dim=BS,
    )
