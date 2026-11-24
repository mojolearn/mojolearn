# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host passes of the base binding (`bindings/_mojolearn.mojo`): the
float64 narrowing, the nonzero count and fill,
the finiteness replays and the probability-row check, moved here with no
Python import (lane thin-core, 2026-10-21: a pure move, no bit change)."""
from std.math import isfinite


comptime _CAST_WIDTH = 8
"""SIMD lanes per step of the flat cast: 8 float64 in, 8 float32 out. A
tail shorter than this is finished one element at a time through the same
`cast`, so the width never changes a bit of the answer."""


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


# TOMBSTONE: tiled_transpose_to_f32 and its _TILE_ROWS / _TILE_COLS (only the base transpose exports called it) deleted 2026-11-23 on lane/sab-host-missed-cls (no product caller); code recoverable at 4519097d38.


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

