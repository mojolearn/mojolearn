# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's seam oracles (DEVIATIONS 5400-5409, IDENTITY_PATHS rows
140-149): each seam restated as plain host code, in BOTH spellings -- the
pinned one the units ship and the unpinned one a port would write -- so
x_prep/seams/prep_check.mojo can first show a fixture separates them and
then hold the shipped unit to the pinned one bit for bit."""
from std.math import fma
from std.memory import bitcast
from checks.numerics import ftz, identical_mul, identical_div


def seq_sum(v: List[Float32]) -> Float32:
    """5400 pinned: ascending, one rounding per add, operands flushed."""
    var s = Float32(0)
    for i in range(len(v)):
        s = ftz(ftz(s) + ftz(v[i]))
    return s


def first_max(v: List[Float32]) -> Int:
    """5404 pinned."""
    var b = 0
    for i in range(1, len(v)):
        if v[i] > v[b]:
            b = i
    return b


def last_max(v: List[Float32]) -> Int:
    """5404 unpinned."""
    var b = 0
    for i in range(1, len(v)):
        if v[i] >= v[b]:
            b = i
    return b


def splitmix(v: UInt64) -> UInt64:
    var z = v + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def top24(seed: Int, t: Int) -> Int:
    """5406 pinned: the first uniform's 24-bit integer is the top 24 bits."""
    return Int(splitmix(UInt64(seed) * UInt64(0x100000000) + UInt64(2 * t)) >> 40)


def raw_max(v: List[Float32]) -> Float32:
    """5408 unpinned."""
    var m = v[0]
    for i in range(1, len(v)):
        if v[i] > m:
            m = v[i]
    return m

