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


def rev_sum(v: List[Float32]) -> Float32:
    """5400 unpinned: the same adds in descending order."""
    var s = Float32(0)
    for i in range(len(v)):
        s = ftz(s + v[len(v) - 1 - i])
    return s


def pinned_dot(a: List[Float32], b: List[Float32]) -> Float32:
    """5401 pinned: each product rounded before its add."""
    var s = Float32(0)
    for i in range(len(a)):
        s = ftz(ftz(s) + ftz(identical_mul(ftz(a[i]), ftz(b[i]))))
    return s


def fused_dot(a: List[Float32], b: List[Float32]) -> Float32:
    """5401 unpinned: acc = fma(a, b, acc)."""
    var s = Float32(0)
    for i in range(len(a)):
        s = fma(a[i], b[i], s)
    return s


def total_key(x: Float32) -> UInt32:
    """5402 pinned: the sort key (negatives below positives, -0.0 below
    +0.0, every NaN last)."""
    if x != x:
        return UInt32(0xFFFFFFFF)
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


def key_sorted(v: List[Float32]) -> List[Float32]:
    """5402 pinned: a stable insertion sort by total_key (the heap sort's
    output words are the same multiset in the same key order; equal keys are
    equal words except NaN, which the fixture keeps to one word)."""
    var out = v.copy()
    for i in range(1, len(out)):
        var x = out[i]
        var j = i - 1
        while j >= 0 and total_key(out[j]) > total_key(x):
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = x
    return out^


def value_sorted(v: List[Float32]) -> List[Float32]:
    """5402 unpinned: the same insertion sort comparing values with `<`."""
    var out = v.copy()
    for i in range(1, len(out)):
        var x = out[i]
        var j = i - 1
        while j >= 0 and x < out[j]:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = x
    return out^


def guarded_mean(s: Float32, cnt: Int) -> Float32:
    """5403 pinned: an empty column's mean is 0, never 0/0."""
    if cnt == 0:
        return Float32(0)
    return ftz(identical_div(s, Float32(cnt)))


def raw_mean(s: Float32, cnt: Int) -> Float32:
    """5403 unpinned: s / cnt, a NaN (with the vendor's payload) when empty."""
    return identical_div(s, Float32(cnt))


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


def largest_positive(col: List[Float32]) -> Bool:
    """5405 pinned: the largest-magnitude component (first on a tie) is > 0."""
    var big = 0
    for i in range(1, len(col)):
        if abs(col[i]) > abs(col[big]):
            big = i
    return col[big] > Float32(0)


def first_positive(col: List[Float32]) -> Bool:
    """5405 unpinned: the first component is >= 0."""
    return col[0] >= Float32(0)


def splitmix(v: UInt64) -> UInt64:
    var z = v + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def top24(seed: Int, t: Int) -> Int:
    """5406 pinned: the first uniform's 24-bit integer is the top 24 bits."""
    return Int(splitmix(UInt64(seed) * UInt64(0x100000000) + UInt64(2 * t)) >> 40)


def low24(seed: Int, t: Int) -> Int:
    """5406 unpinned: the low 24 bits."""
    return Int(splitmix(UInt64(seed) * UInt64(0x100000000) + UInt64(2 * t)) & UInt64(0xFFFFFF))


def count_strict(v: List[Float32], x: Float32, r: Float32) -> Int:
    """5407 pinned: points within nextafter(r, 0) of x: dist < r (== 0 at r == 0)."""
    var c = 0
    for i in range(len(v)):
        var dist = abs(v[i] - x)
        if (r > Float32(0) and dist < r) or (r == Float32(0) and dist == Float32(0)):
            c += 1
    return c


def count_closed(v: List[Float32], x: Float32, r: Float32) -> Int:
    """5407 unpinned: dist <= r."""
    var c = 0
    for i in range(len(v)):
        if abs(v[i] - x) <= r:
            c += 1
    return c


def flushed_max(v: List[Float32]) -> Float32:
    """5408 pinned: operands through ftz before the compare."""
    var m = ftz(v[0])
    for i in range(1, len(v)):
        if ftz(v[i]) > m:
            m = ftz(v[i])
    return m


def raw_max(v: List[Float32]) -> Float32:
    """5408 unpinned."""
    var m = v[0]
    for i in range(1, len(v)):
        if v[i] > m:
            m = v[i]
    return m


def numpy_lerp(a: Float32, b: Float32, g: Float32) -> Float32:
    """5409 pinned: numpy `_lerp`, the upper spelling from g >= 0.5."""
    var diff = ftz(b - a)
    if g >= Float32(0.5):
        return ftz(b - ftz(identical_mul(diff, ftz(Float32(1) - g))))
    return ftz(a + ftz(identical_mul(diff, g)))


def naive_lerp(a: Float32, b: Float32, g: Float32) -> Float32:
    """5409 unpinned: a + (b - a) * g everywhere."""
    return ftz(a + ftz(identical_mul(ftz(b - a), g)))
