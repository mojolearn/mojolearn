# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The bound-and-compact selector (DEVIATION 3060) against an exhaustive host
order statistic AND against the small-k selector, byte for byte, with both
of its paths seen to run.

Rows of every case:
  0  hashed, nearly distinct values            (the fast path at every length)
  1  the special values in a cycle of nine     (signed zeros, subnormals,
                                                infinities, NaN payloads)
  2  one value everywhere                      (every tie broken by column)
  3  the row's smallest values all in the columns of ONE thread
                                               (a hiding thread: the flagged launch)
  4  descending values                         (the answer sits in the tail)
  5  five distinct values, hashed              (heavy duplicates)

The host oracle scans the whole row for its next composite key; it shares no
list, bound or reduction with either device selector. A case passes when the
bound-and-compact output equals the oracle and equals the small-k selector's
output in every (distance bits, index) cell. The flags are read back: row 3
must be flagged wherever a thread owns more than the list depth of the
row's k smallest, and row 0 must not be flagged at 65,536 columns, so a pass
covers both launches. Build with
`-D MOJOLEARN_KNN_SELECTOR_BOUND_SABOTAGE=1` (or `..._FALLBACK_SABOTAGE=1`)
and this check must FAIL; that is its control.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.select_radix_identical import composite_key
from neighbors.checks.select_smallk_identical_candidate import smallk_select_launch
from neighbors.checks.knn_selector_bound_compact import (
    SBC_DEPTH,
    bound_compact_select_launch,
)

comptime ROWS = 6


def splitmix64(x: UInt64) -> UInt64:
    var z = x &+ UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> UInt64(30))) &* UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) &* UInt64(0x94D049BB133111EB)
    return z ^ (z >> UInt64(31))


def cell_bits(row: Int, col: Int, length: Int) -> UInt32:
    var special: List[UInt32] = [0, 2147483648, 1, 2147483649, 2139095040, 4286578688, 2143289345, 2143289346, 4290772993]
    var h = splitmix64(UInt64(row * 1000003 + col))
    if row == 0:
        # positive normal floats in [1, 2): 23 hashed mantissa bits
        return UInt32(1065353216) + UInt32(h & UInt64(8388607))
    if row == 1:
        return special[col % len(special)]
    if row == 2:
        return UInt32(1065353216)
    if row == 3:
        # thread 7's columns ascend from 1.0; every other column is 4.0 or more
        if col % 256 == 7:
            return UInt32(1065353216) + UInt32(col // 256)
        return UInt32(1082130432) + UInt32(h & UInt64(65535))
    if row == 4:
        return UInt32(1065353216) + UInt32(length - col)
    return UInt32(1065353216) + UInt32(h % UInt64(5))


def check_case(length: Int, k: Int, select_min: Bool) raises -> Int:
    """Returns the number of flagged rows."""
    with DeviceContext() as ctx:
        var host = ctx.enqueue_create_host_buffer[DType.float32](ROWS * length)
        var new_d = ctx.enqueue_create_host_buffer[DType.float32](ROWS * k)
        var new_i = ctx.enqueue_create_host_buffer[DType.uint32](ROWS * k)
        var old_d = ctx.enqueue_create_host_buffer[DType.float32](ROWS * k)
        var old_i = ctx.enqueue_create_host_buffer[DType.uint32](ROWS * k)
        var host_flags = ctx.enqueue_create_host_buffer[DType.uint32](ROWS)
        ctx.synchronize()
        for row in range(ROWS):
            for col in range(length):
                host.unsafe_ptr().unsafe_store(row * length + col, bitcast[DType.float32](cell_bits(row, col, length)))
        var values = ctx.enqueue_create_buffer[DType.float32](ROWS * length)
        var distances = ctx.enqueue_create_buffer[DType.float32](ROWS * k)
        var indices = ctx.enqueue_create_buffer[DType.uint32](ROWS * k)
        var distances_old = ctx.enqueue_create_buffer[DType.float32](ROWS * k)
        var indices_old = ctx.enqueue_create_buffer[DType.uint32](ROWS * k)
        var flags = ctx.enqueue_create_buffer[DType.uint32](ROWS)
        ctx.enqueue_copy(dst_buf=values, src_ptr=host.unsafe_ptr())
        bound_compact_select_launch(
            ctx, values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            distances.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            indices.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            flags, ROWS, length, k, select_min,
        )
        smallk_select_launch(
            ctx, values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            distances_old.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            indices_old.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            ROWS, length, k, select_min,
        )
        ctx.enqueue_copy(dst_ptr=new_d.unsafe_ptr(), src_buf=distances)
        ctx.enqueue_copy(dst_ptr=new_i.unsafe_ptr(), src_buf=indices)
        ctx.enqueue_copy(dst_ptr=old_d.unsafe_ptr(), src_buf=distances_old)
        ctx.enqueue_copy(dst_ptr=old_i.unsafe_ptr(), src_buf=indices_old)
        ctx.enqueue_copy(dst_ptr=host_flags.unsafe_ptr(), src_buf=flags)
        ctx.synchronize()
        var flagged = 0
        for row in range(ROWS):
            var flag = Int(host_flags.unsafe_ptr().unsafe_load(row))
            if flag != 0 and flag != 1:
                raise Error("bound-compact check: a flag is neither 0 nor 1")
            flagged += flag
            var previous = UInt64(0)
            for rank in range(k):
                var best = UInt64(18446744073709551615)
                var selected = -1
                for col in range(length):
                    var key = composite_key(host.unsafe_ptr().unsafe_load(row * length + col), UInt32(col), select_min)
                    if (rank == 0 or key > previous) and key < best:
                        best = key
                        selected = col
                var cell = row * k + rank
                var got_i = new_i.unsafe_ptr().unsafe_load(cell)
                var got_d = bitcast[DType.uint32](new_d.unsafe_ptr().unsafe_load(cell))
                if selected < 0 or got_i != UInt32(selected):
                    raise Error(
                        "bound-compact index differs from the host rank: length " + String(length) + " k " + String(k)
                        + " row " + String(row) + " rank " + String(rank) + " got " + String(got_i) + " want " + String(selected)
                    )
                if got_d != bitcast[DType.uint32](host.unsafe_ptr().unsafe_load(row * length + selected)):
                    raise Error("bound-compact distance bits differ from the host input")
                if got_i != old_i.unsafe_ptr().unsafe_load(cell) or got_d != bitcast[DType.uint32](old_d.unsafe_ptr().unsafe_load(cell)):
                    raise Error("bound-compact output differs from the small-k selector")
                previous = best
            # Reach. Row 3's k smallest are thread 7's first columns whenever
            # that thread owns at least k of them; more than the list depth
            # of them below the bound is a hiding thread.
            if row == 3 and select_min and k > SBC_DEPTH and length >= 256 * k and flag != 1:
                raise Error("bound-compact check: the one-thread row was not flagged")
            if row == 0 and length == 65536 and flag != 0:
                raise Error("bound-compact check: the hashed row was flagged at 65,536 columns")
        print("SELECTOR_BOUND_CASE_PASS", length, k, select_min, "cells", ROWS * k, "flagged_rows", flagged)
        _ = host^
        _ = values^
        _ = flags^
        return flagged


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("bound-compact selector check requires IDENTICAL")
    var lengths: List[Int] = [64, 65, 255, 256, 257, 2047, 2048, 4096, 6784, 16384, 65536, 65537]
    var counts: List[Int] = [1, 8, 16, 17, 32, 33, 64]
    var cases = 0
    var flagged = 0
    for length in lengths:
        for k in counts:
            if k <= length:
                flagged += check_case(length, k, True)
                flagged += check_case(length, k, False)
                cases += 2
    if flagged == 0:
        raise Error("bound-compact check: no case reached the flagged launch")
    print("KNN BOUND-COMPACT SELECTOR PASS", "cases", cases, "flagged_rows", flagged, "depth", SBC_DEPTH)
