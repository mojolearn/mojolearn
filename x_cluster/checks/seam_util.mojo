# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Shared plumbing of the cluster lane's seam checks (pass 2,
lane/algos-cluster): the planted fixture, bit comparisons, the VACUOUS guard
and the equality assertion. Host code."""
from std.memory import bitcast

from x_cluster.bodies import SplitMix64


def seam_fixture(n: Int, d: Int, seed: UInt64) -> List[Float32]:
    """Rows of mixed scale with the pathologies every seam reads planted:
    columns spanning 1e-3 .. 1e3 (so a fold's order moves low bits), a -0.0
    and subnormals in row 0, an all-zero column (d - 1), row 1 == row 2 (an
    exact tie) and row 3 == -row 4."""
    var rng = SplitMix64(seed)
    var x = List[Float32](capacity=n * d)
    for r in range(n):
        for f in range(d):
            var u = Float32(rng.unit()) * Float32(2) - Float32(1)
            var scale = Float32(1)
            if f % 3 == 1:
                scale = Float32(1000)
            elif f % 3 == 2:
                scale = Float32(0.001)
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


def bits_f32(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def count_diff_f32(a: List[Float32], b: List[Float32]) -> Int:
    var c = 0
    for t in range(len(a)):
        if bits_f32(a[t]) != bits_f32(b[t]):
            c += 1
    return c + abs(len(a) - len(b))


def count_diff_i32(a: List[Int32], b: List[Int32]) -> Int:
    var c = 0
    for t in range(len(a)):
        if a[t] != b[t]:
            c += 1
    return c + abs(len(a) - len(b))


def require_separates(seam: String, differing: Int) raises:
    """THE FIXTURE MUST SEPARATE the pinned spelling from the unpinned one
    before the equality below means anything."""
    if differing == 0:
        raise Error("VACUOUS " + seam + ": the fixture does not separate the pinned spelling from the alternative")
    print("  " + seam + ": fixture separates (" + String(differing) + " cells differ between the spellings)")


def require_equal(seam: String, differing: Int) raises:
    if differing != 0:
        raise Error("SEAM " + seam + ": device != oracle in " + String(differing) + " cells")
    print("  " + seam + ": device == oracle, bit for bit")
