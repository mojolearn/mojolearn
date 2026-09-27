# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Shared plumbing of the neighbors lane's seam checks (pass 2): the planted
fixture, bit comparisons, the VACUOUS guard, the equality assertion and the
address of a List for the drivers' address contract. Host code."""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


struct Lcg(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 6364136223846793005 + 1442695040888963407

    def unit(mut self) -> Float32:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float32(Int((self.s >> 40) & 0xFFFFFF)) / Float32(16777216)


def seam_fixture(n: Int, d: Int, seed: UInt64) -> List[Float32]:
    """Rows of mixed scale with the pathologies the seams read: columns at
    1, 1e3 and 1e-3 (a fold's order moves low bits), -0.0 and a subnormal in
    row 0, an all-zero last column, row 2 == row 1 (an exact tie) and
    row 4 == -row 3."""
    var rng = Lcg(seed)
    var x = List[Float32](capacity=n * d)
    for _ in range(n):
        for f in range(d):
            var u = rng.unit() * Float32(2) - Float32(1)
            var scale = Float32(1)
            if f % 3 == 1:
                scale = Float32(1000)
            elif f % 3 == 2:
                scale = Float32(0.001)
            x.append(u * scale)
    for row in range(n):
        x[row * d + d - 1] = Float32(0)
    x[0] = Float32(-0.0)
    if d > 2:
        x[2] = bitcast[DType.float32](UInt32(0x00000003))
    if n > 4:
        for f in range(d):
            x[2 * d + f] = x[1 * d + f]
            x[4 * d + f] = -x[3 * d + f]
    return x^


def positive_fixture(n: Int, d: Int, seed: UInt64) -> List[Float32]:
    """Non-negative rows (chi-squared inputs) with exact zeros planted."""
    var x = seam_fixture(n, d, seed)
    for i in range(len(x)):
        x[i] = abs(x[i])
    return x^


def fa(l: List[Float32]) -> Int:
    """The address of a List for an op call. THE CALLER KEEPS THE LIST ALIVE
    past the call (`_ = l^` after its last use): Mojo frees a value at its
    last use, and taking an address is not a use of the storage."""
    return Int(l.unsafe_ptr())


def ia(l: List[Int32]) -> Int:
    return Int(l.unsafe_ptr())


def zf(n: Int) -> List[Float32]:
    return List[Float32](length=n if n > 0 else 1, fill=Float32(0))


def zi(n: Int) -> List[Int32]:
    return List[Int32](length=n if n > 0 else 1, fill=Int32(0))


def count_diff_f32(a: List[Float32], b: List[Float32]) -> Int:
    var c = 0
    for t in range(min(len(a), len(b))):
        if bitcast[DType.uint32](a[t]) != bitcast[DType.uint32](b[t]):
            c += 1
    return c + abs(len(a) - len(b))


def count_diff_i32(a: List[Int32], b: List[Int32]) -> Int:
    var c = 0
    for t in range(min(len(a), len(b))):
        if a[t] != b[t]:
            c += 1
    return c + abs(len(a) - len(b))


def require_separates(seam: String, differing: Int) raises:
    """THE FIXTURE MUST SEPARATE the pinned spelling from the alternative
    before the equality means anything (CONTRIBUTING.md "Numerical changes")."""
    if differing == 0:
        raise Error("VACUOUS " + seam + ": the fixture does not separate the pinned spelling from the alternative")
    print("  " + seam + ": fixture separates (" + String(differing) + " cells differ between the spellings)")


def same(seam: String, differing: Int) raises:
    """IDENTICAL: equality with the oracle, bit for bit. FAST: reported, no claim."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if differing != 0:
            raise Error("SEAM " + seam + ": != oracle in " + String(differing) + " cells")
        print("  " + seam + ": == oracle, bit for bit")
    else:
        print("  " + seam + ": FAST, " + String(differing) + " cells differ from the oracle (no claim)")
