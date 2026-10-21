# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host passes of the base binding (`bindings/_mojolearn.mojo`): the
float64 narrowing, the tiled cast-and-transpose, the nonzero count and fill,
the finiteness replays and the probability-row check, moved here with no
Python import (lane thin-core, 2026-10-21: a pure move, no bit change)."""
from std.math import isfinite


comptime _CAST_WIDTH = 8
"""SIMD lanes per step of the flat cast: 8 float64 in, 8 float32 out. A
tail shorter than this is finished one element at a time through the same
`cast`, so the width never changes a bit of the answer."""

comptime _TILE_ROWS = 128
comptime _TILE_COLS = 64
"""The fused cast-and-transpose walks the source in `_TILE_ROWS` x
`_TILE_COLS` tiles: 64 KB of float64 in, 32 KB of float32 out, sized to
stay in a performance core's L1 so the strided side of the transpose hits
cache instead of memory. That is DEVIATION 1887's row-tile arm, which the
pure-Python converter does not carry, moved to where it can be fast. The
tile shape changes only the order in which independent elements are
written; each element's value is the same single cast."""


def cast_f64_to_f32_host(sp: MutPointer[Float64, MutUntrackedOrigin], dp: MutPointer[Float32, MutUntrackedOrigin], count: Int):
    """`dp[i] = Float32(sp[i])` for i in [0, count): _CAST_WIDTH lanes per
    step, then the tail one element at a time through the same cast."""
    var i = 0
    var body = count - (count % _CAST_WIDTH)
    while i < body:
        var v = sp.unsafe_load[width=_CAST_WIDTH](i)
        dp.unsafe_store[width=_CAST_WIDTH](i, v.cast[DType.float32]())
        i += _CAST_WIDTH
    while i < count:
        dp.unsafe_store(i, sp.unsafe_load(i).cast[DType.float32]())
        i += 1


def tiled_transpose_to_f32[
    S: DType
](
    sp: MutPointer[Scalar[S], MutUntrackedOrigin],
    dp: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int,
    nc: Int,
):
    """`dp[c * nr + r] = Float32(sp[r * nc + c])` over `_TILE_ROWS` x
    `_TILE_COLS` tiles. The one loop behind DEVIATION 2471 (float64 in)
    and 2472 (float32 in, where the cast is the identity and compiles
    away). Caller holds valid, non-overlapping, non-empty buffers and has
    released the GIL."""
    var r0 = 0
    while r0 < nr:
        var r1 = min(r0 + _TILE_ROWS, nr)
        var c0 = 0
        while c0 < nc:
            var c1 = min(c0 + _TILE_COLS, nc)
            # Inside the tile: one column at a time, so the writes are
            # contiguous and the strided reads stay within the tile's
            # rows, which the previous column just pulled into cache.
            for c in range(c0, c1):
                var dbase = c * nr
                for r in range(r0, r1):
                    dp.unsafe_store(
                        dbase + r,
                        sp.unsafe_load(r * nc + c).cast[DType.float32](),
                    )
            c0 = c1
        r0 = r1


def nonzero_f64_count_host(sp: MutPointer[Float64, MutUntrackedOrigin], count: Int) -> Int:
    """How many of the count float64 values at sp pass `v != 0.0`."""
    var nz = 0
    for i in range(count):
        if sp.unsafe_load(i) != 0.0:
            nz += 1
    return nz


def nonzero_f64_fill_host(
    sp: MutPointer[Float64, MutUntrackedOrigin], nr: Int, nc: Int, cap: Int, rp: MutPointer[Int32, MutUntrackedOrigin], cp: MutPointer[Int32, MutUntrackedOrigin], vp: MutPointer[Float32, MutUntrackedOrigin],
) -> Int:
    """COO triples of the nr x nc float64 matrix in row-major scan order;
    returns the number written, or -1 when more than cap entries are nonzero
    (the fill stops at the first one past cap)."""
    var k = 0
    var overflow = False
    for r in range(nr):
        var base = r * nc
        for c in range(nc):
            var v = sp.unsafe_load(base + c)
            if v != 0.0:
                if k >= cap:
                    overflow = True
                    break
                rp.unsafe_store(k, Int32(r))
                cp.unsafe_store(k, Int32(c))
                vp.unsafe_store(k, v.cast[DType.float32]())
                k += 1
        if overflow:
            break
    if overflow:
        return -1
    return k


def all_finite_host_f32(p: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Int:
    """Host replay of the device finiteness scan: reached only from a
    HOTPATH_SABOTAGE build (the tests' sabotage switch) or `n == 0`."""
    for i in range(count):
        if not isfinite(p.unsafe_load(i)):
            return 0
    return 1


def all_finite_host_f64(p: MutPointer[Float64, MutUntrackedOrigin], count: Int) -> Int:
    """`all_finite_host_f32` over float64 values."""
    for i in range(count):
        if not isfinite(p.unsafe_load(i)):
            return 0
    return 1


@fieldwise_init
struct ProbRowFlags(Copyable, Movable):
    """What `probability_rows_f32_host` found: a non-finite value, a value
    outside [0, 1], a multiclass row whose sum is off by more than the
    tolerance."""
    var nonfinite: Bool
    var outside: Bool
    var bad_sum: Bool


def probability_rows_f32_host(src: MutPointer[Float32, MutUntrackedOrigin], dst: MutPointer[Float32, MutUntrackedOrigin], nr: Int, nc: Int, bin: Int) -> ProbRowFlags:
    """Check nr x nc probability rows; when bin == 1 also write the two-column
    (1 - p, p) rows to dst."""
    var nonfinite = False
    var outside = False
    var bad_sum = False
    for r in range(nr):
        var total = Float64(0)
        for c in range(nc):
            var p = src.unsafe_load(r * nc + c)
            nonfinite = nonfinite or not isfinite(p)
            outside = outside or p < 0 or p > 1
            total += Float64(p)
            if bin == 1:
                dst.unsafe_store(2 * r, Float32(1) - p)
                dst.unsafe_store(2 * r + 1, p)
        # Existing NumPy validation used sqrt(Float32 epsilon), itself
        # rounded to Float32, then widened for the Float64 comparison.
        if bin == 0:
            var error = total - Float64(1)
            bad_sum = bad_sum or abs(error) > Float64(0.00034526697709225118)
    return ProbRowFlags(nonfinite, outside, bad_sum)
