# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Plumbing of the decomp lane's seam checks (pass 2, lane/algos-decomp): the
planted fixture, pointers onto host lists, bit comparisons, the VACUOUS guard
and the equality assertion. Host code, no GPU import."""
from std.memory import bitcast

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.cells import F32Ptr, I32Ptr


def lcg_unit(mut state: UInt64) -> Float32:
    """A 64-bit LCG (Knuth MMIX constants), the top 24 bits as [0, 1)."""
    state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Float32(UInt32(state >> 40)) * Float32(5.9604644775390625e-08)


def seam_fixture(n: Int, d: Int, seed: UInt64) -> List[Float32]:
    """Rows of mixed scale with the pathologies every seam reads planted:
    columns spanning 1e-3 .. 1e4 with a +1e8/-1e8 pair every fourth column
    (so a fold's order moves bits), -0.0 and a subnormal in row 0, an
    all-zero column (d - 1), row 1 == row 2 (an exact tie) and row 3 ==
    -row 4."""
    var st = seed * UInt64(2654435761) + UInt64(1)
    var x = List[Float32](capacity=n * d)
    for r in range(n):
        for f in range(d):
            var u = lcg_unit(st) * Float32(2) - Float32(1)
            var scale = Float32(1)
            if f % 4 == 1:
                scale = Float32(10000)
            elif f % 4 == 2:
                scale = Float32(0.001)
            elif f % 4 == 3:
                scale = Float32(1e8) if r % 2 == 0 else Float32(-1e8)
            x.append(u * scale)
    for r in range(n):
        x[r * d + d - 1] = Float32(0)
    x[0] = Float32(-0.0)
    if d > 2:
        x[2] = bitcast[DType.float32](UInt32(0x00000003))
    if n > 4:
        for f in range(d):
            x[2 * d + f] = x[1 * d + f]
            x[4 * d + f] = -x[3 * d + f]
    return x^


def positive_fixture(n: Int, d: Int, seed: UInt64) -> List[Float32]:
    """|seam_fixture| scaled into (0, 4], every tenth entry exactly 0."""
    var st = seed * UInt64(40503) + UInt64(7)
    var x = List[Float32](capacity=n * d)
    for t in range(n * d):
        var u = lcg_unit(st) * Float32(4)
        x.append(Float32(0) if t % 10 == 3 else u)
    return x^


def ptr(values: List[Float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def iptr(values: List[Int32]) -> I32Ptr:
    return I32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def zeros(n: Int) -> List[Float32]:
    return List[Float32](length=n if n > 0 else 1, fill=Float32(0))


def bits_f32(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def count_diff_f32(a: List[Float32], b: List[Float32]) -> Int:
    var c = 0
    var n = min(len(a), len(b))
    for t in range(n):
        if bits_f32(a[t]) != bits_f32(b[t]):
            c += 1
    return c + abs(len(a) - len(b))


def count_diff_i32(a: List[Int32], b: List[Int32]) -> Int:
    var c = 0
    var n = min(len(a), len(b))
    for t in range(n):
        if a[t] != b[t]:
            c += 1
    return c + abs(len(a) - len(b))


def require_separates(seam: String, differing: Int) raises:
    """THE FIXTURE MUST SEPARATE the pinned spelling from the unpinned one
    before the equality below means anything."""
    if differing == 0:
        raise Error("VACUOUS " + seam + ": the fixture does not separate the pinned spelling from the alternative")
    print("  " + seam + ": fixture separates (" + String(differing) + " cells differ between the spellings)")


def same(seam: String, differing: Int) raises:
    """Device (or host) == oracle, bit for bit, under IDENTICAL; FAST reports."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if differing != 0:
            raise Error("SEAM " + seam + ": != oracle in " + String(differing) + " cells")
        print("  " + seam + ": == oracle, bit for bit")
    else:
        print("  " + seam + ": FAST, " + String(differing) + " cells differ from the oracle (no claim)")
