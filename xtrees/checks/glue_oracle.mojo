# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The trees lane's seams restated as plain host code, in the spelling each
seam PINS (xtrees/ops.mojo, DEVIATIONS 5600-5605), and beside each the
spelling it does NOT use, so `glue_check.mojo` can show a fixture separates
the two before it trusts the fixture.

Written from the reference and the DEVIATION text, not by copying
xtrees/ops.mojo: an index is `draw mod n`; a fold is sequential in index
order; a product meeting an add is rounded before the add; exp is the pinned
polynomial; a tie goes to the lower index; a zero row is uniform."""
from std.math import fma, exp
from checks.numerics import portable_exp64, pinned_mul_f64


def splitmix(z_in: UInt64) -> UInt64:
    var z = z_in
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def counter_draw(seed: Int, stream: Int, k: Int) -> UInt64:
    """DEVIATION 5600: draw k of (seed, stream) is a pure function of the three."""
    comptime G: UInt64 = 0x9E3779B97F4A7C15
    var base = splitmix(splitmix(UInt64(seed) + G) + UInt64(stream) * G + 1)
    return splitmix(base + UInt64(k + 1) * G)


def index_mod(r: UInt64, n: Int) -> Int:
    """The pinned index mapping: r mod n."""
    return Int(r % UInt64(n))


def index_mulshift(r: UInt64, n: Int) -> Int:
    """The UNPINNED alternative (Lemire's multiply-shift over 32 high bits)."""
    return Int(((r >> 32) * UInt64(n)) >> 32)


def pinned_acc(acc: Float64, w: Float64, x: Float64) -> Float64:
    """DEVIATION 5601: the product rounds, then the add."""
    return acc + pinned_mul_f64(w, x)


def fused_acc(acc: Float64, w: Float64, x: Float64) -> Float64:
    """The contracted spelling a build may pick for `acc + w * x`."""
    return fma(w, x, acc)


def seq_sum(v: List[Float64]) -> Float64:
    """DEVIATION 5602: sequential in index order."""
    var s: Float64 = 0.0
    for i in range(len(v)):
        s = s + v[i]
    return s


def pair_sum(v: List[Float64]) -> Float64:
    """The pairwise spelling (numpy's), for the separation test."""
    if len(v) == 1:
        return v[0]
    var half = len(v) // 2
    var a = List[Float64]()
    var b = List[Float64]()
    for i in range(len(v)):
        if i < half:
            a.append(v[i])
        else:
            b.append(v[i])
    return pair_sum(a) + pair_sum(b)


def pinned_exp(x: Float64) -> Float64:
    """DEVIATION 5603: the pinned binary64 polynomial."""
    return portable_exp64(x)


def libm_exp(x: Float64) -> Float64:
    return exp(x)


def first_max(v: List[Float64]) -> Int:
    """DEVIATION 5604: the lower index wins a tie."""
    var best = 0
    for i in range(1, len(v)):
        if v[i] > v[best]:
            best = i
    return best


def last_max(v: List[Float64]) -> Int:
    var best = 0
    for i in range(1, len(v)):
        if v[i] >= v[best]:
            best = i
    return best


def normalized(v: List[Float64]) -> List[Float64]:
    """DEVIATION 5605: a zero row is uniform 1/k, never 0/0."""
    var s: Float64 = 0.0
    for i in range(len(v)):
        s = s + v[i]
    var out = List[Float64]()
    for i in range(len(v)):
        out.append(v[i] / s if s > 0.0 else 1.0 / Float64(len(v)))
    return out^
