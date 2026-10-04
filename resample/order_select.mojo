# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""K12 (IDENTICAL, lane ml-cluster-nbrs 2026-10-04): the bootstrap quantile
of each replicate WITHOUT materializing the replicates or sorting them.

Today the quantile arm writes `n_replicates x n` drawn values
(`materialize_resample_kernel`), sorts every segment
(`core/segmented_sort.mojo`) and reads two ranks (`order_stat_kernel`).
Here one block per replicate finds those two ranks by bisection on the
sorting keys: the key of a draw is `float_to_sortable` of the same flushed
word `materialize_resample_kernel` writes, and the j-th order statistic of the
replicate is the smallest key K with #{draws with key <= K} >= j + 1. Each
bisection step regenerates the replicate's draws (`draw_row_index` is a pure
function of (key, r, i)) and counts with an integer block sum, so every
count is exact and order-free. The order statistic of a multiset has one
answer, so the two words, the interpolation (`quantile_interpolate`, the same
three lines) and theta are the words of the sort path on every vendor; the
host column is untouched.

Shared page: tpb Int32 (at most 1 KB at tpb = 256). No atomics.
`-D MOJOLEARN_IDN_BOOT_SELECT_OFF` (or MOJOLEARN_IDN_ALL_OFF) restores the
materialize + sort path. The trimmed mean keeps the sort (its pinned fold
reads every kept rank in order).
"""
from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from core.segmented_sort import float_to_sortable, sortable_to_float
from metrics.checks.pinned_sum import canonicalize_nan
from resample.checks.index_map import draw_row_index, key_join
from resample.checks.statistics import (
    quantile_interpolate,
    quantile_lower_index,
    quantile_position,
)

comptime _SI32 = UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
comptime _SU32 = UnsafePointer[UInt32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]

comptime IDN_BOOT_SELECT = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_BOOT_SELECT_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


@always_inline
def _draw_key(
    x: MutPointer[Float32, MutAnyOrigin], key: UInt64, r: Int, i: Int, n_rows: Int32, n_features: Int, col: Int
) -> UInt32:
    """The sorting key of draw i of replicate r: `materialize_resample_kernel`'s
    flushed word, twiddled as `segmented_sort_keys_f32` twiddles it."""
    var row = Int(draw_row_index(key, r, i, n_rows))
    var v = ftz(x.unsafe_load(row * n_features + col))
    return float_to_sortable(bitcast[DType.uint32](v))


@always_inline
def _block_total[tpb: Int](red: _SI32, tid: Int, v: Int32) -> Int32:
    """The integer sum of every thread's v, returned to every thread (exact:
    Int32 adds in any order). Two barriers, so `red` is free on return."""
    red[tid] = v
    barrier()
    var step = tpb // 2
    while step > 0:
        if tid < step:
            red[tid] = red[tid] + red[tid + step]
        barrier()
        step //= 2
    var t = red[0]
    barrier()
    return t


@always_inline
def _block_min[tpb: Int](red: _SU32, tid: Int, v: UInt32) -> UInt32:
    """The minimum of every thread's v, returned to every thread (order-free)."""
    red[tid] = v
    barrier()
    var step = tpb // 2
    while step > 0:
        if tid < step:
            red[tid] = min(red[tid], red[tid + step])
        barrier()
        step //= 2
    var t = red[0]
    barrier()
    return t


@always_inline
def _count_le[tpb: Int](
    x: MutPointer[Float32, MutAnyOrigin], key: UInt64, r: Int, n: Int, n_rows: Int32, n_features: Int, col: Int,
    bound: UInt32, red: _SI32, tid: Int,
) -> Int:
    var c = Int32(0)
    for i in range(tid, n, tpb):
        if _draw_key(x, key, r, i, n_rows, n_features, col) <= bound:
            c += 1
    return Int(_block_total[tpb](red, tid, c))


@always_inline
def _select_key[tpb: Int](
    x: MutPointer[Float32, MutAnyOrigin], key: UInt64, r: Int, n: Int, n_rows: Int32, n_features: Int, col: Int,
    j: Int, red: _SI32, tid: Int,
) -> UInt32:
    """The smallest key K with #{draws <= K} >= j + 1: the key of rank j."""
    var lo = UInt32(0)
    var hi = UInt32(0xFFFFFFFF)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if _count_le[tpb](x, key, r, n, n_rows, n_features, col, mid, red, tid) >= j + 1:
            hi = mid
        else:
            lo = mid + 1
    return lo


def boot_quantile_select_kernel[tpb: Int](
    theta: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    r_first_in: Int32,
    n_replicates_in: Int32,
    n_in: Int32,
    n_rows_in: Int32,
    n_features_in: Int32,
    col_in: Int32,
    q: Float32,
):
    """theta[rr] = `order_stat_kernel[STAT_QUANTILE]`'s value for replicate
    r_first + rr, from the two order statistics found by bisection."""
    var rr = Int(block_idx.x)
    if rr >= Int(n_replicates_in):
        return
    var tid = Int(thread_idx.x)
    var red = stack_allocation[tpb, Int32, address_space = AddressSpace.SHARED]()
    var redu = stack_allocation[tpb, UInt32, address_space = AddressSpace.SHARED]()
    var key = key_join(lo_bits, hi_bits)
    var r = Int(r_first_in) + rr
    var m = Int(n_in)
    var nf = Int(n_features_in)
    var col = Int(col_in)
    var v: Float32
    if m == 1:
        v = bitcast[DType.float32](sortable_to_float(_draw_key(x, key, r, 0, n_rows_in, nf, col)))
    else:
        var h = quantile_position(m, q)
        var lo = quantile_lower_index(h, m)
        var frac = ftz(h - Float32(lo))
        var hi = lo + 1
        if hi > m - 1:
            hi = m - 1
        var k_lo = _select_key[tpb](x, key, r, m, n_rows_in, nf, col, lo, red, tid)
        var k_hi = k_lo
        if hi != lo:
            # rank hi has key k_lo when more than hi draws are <= k_lo,
            # else it is the smallest drawn key above k_lo
            var c = _count_le[tpb](x, key, r, m, n_rows_in, nf, col, k_lo, red, tid)
            if c < hi + 1:
                var best = UInt32(0xFFFFFFFF)
                for i in range(tid, m, tpb):
                    var kk = _draw_key(x, key, r, i, n_rows_in, nf, col)
                    if kk > k_lo and kk < best:
                        best = kk
                k_hi = _block_min[tpb](redu, tid, best)
        v = quantile_interpolate(
            bitcast[DType.float32](sortable_to_float(k_lo)),
            bitcast[DType.float32](sortable_to_float(k_hi)),
            frac,
        )
    if tid == 0:
        theta.unsafe_store(rr, canonicalize_nan(v))
