# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The metrics lane's seam oracles (DEVIATIONS 6100-6108): each seam's PINNED
spelling restated as plain host code, beside the UNPINNED spelling a fixture
must separate it from. Written without importing the units they judge."""
from std.memory import bitcast
from std.math import fma
from checks.fixture_rng import splitmix_pair
from checks.numerics import pinned_mul_f32


def seq_sum(v: List[Float32]) -> Float32:
    var s = Float32(0)
    for x in v:
        s = s + x
    return s


def pair_sum(v: List[Float32], leaf: Int) -> Float32:
    """DEVIATION 6100 restated: leaves of `leaf` summed in order, then merged
    as a binary counter merges carries (older on the left), the partial leaf
    and the stack folded from the newest upward."""
    var stack = List[Float32]()
    var count = 0
    var acc = Float32(0)
    var k = 0
    for x in v:
        acc = acc + x
        k += 1
        if k == leaf:
            var y = acc
            var c = count
            while (c & 1) == 1:
                y = stack.pop() + y
                c >>= 1
            stack.append(y)
            count += 1
            acc = Float32(0)
            k = 0
    var i = len(stack) - 1
    while i >= 0:
        acc = stack[i] + acc
        i -= 1
    return acc


def total_key(x: Float32) -> UInt32:
    if x != x:
        return UInt32(0xFFFFFFFF)
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


def argsort_key(v: List[Float32]) -> List[Int32]:
    """Ascending by the total order, ties by row (insertion sort)."""
    var out = List[Int32]()
    for i in range(len(v)):
        out.append(Int32(i))
    for i in range(1, len(v)):
        var j = i
        while j > 0:
            var a = Int(out[j - 1])
            var b = Int(out[j])
            if total_key(v[b]) < total_key(v[a]):
                var t = out[j - 1]
                out[j - 1] = out[j]
                out[j] = t
                j -= 1
            else:
                break
    return out^


def argsort_value(v: List[Float32]) -> List[Int32]:
    """Ascending by VALUE (-0.0 == +0.0), ties by row: the unpinned spelling."""
    var out = List[Int32]()
    for i in range(len(v)):
        out.append(Int32(i))
    for i in range(1, len(v)):
        var j = i
        while j > 0 and v[Int(out[j])] < v[Int(out[j - 1])]:
            var t = out[j - 1]
            out[j - 1] = out[j]
            out[j] = t
            j -= 1
    return out^


def pinned_dot_add(acc: Float32, a: Float32, b: Float32) -> Float32:
    """DEVIATION 6105: the product rounded, then the add."""
    var p = pinned_mul_f32(a, b)
    return acc + p


def fused_dot_add(acc: Float32, a: Float32, b: Float32) -> Float32:
    return fma(a, b, acc)


def seq_prefix(w: List[Float32]) -> List[Float32]:
    """DEVIATION 6107: the CDF as a sequential ascending prefix."""
    var out = List[Float32]()
    var s = Float32(0)
    for x in w:
        s = s + x
        out.append(s)
    return out^


def refold_prefix(w: List[Float32]) -> List[Float32]:
    """The unpinned spelling: each prefix re-summed from the newest term back."""
    var out = List[Float32]()
    for i in range(len(w)):
        var s = Float32(0)
        var j = i
        while j >= 0:
            s = s + w[j]
            j -= 1
        out.append(s)
    return out^


def key_permutation(n: Int, salt: Int) -> List[Int32]:
    """DEVIATION 6108 restated: rows sorted by splitmix_pair(i, salt), then i."""
    var out = List[Int32]()
    for i in range(n):
        out.append(Int32(i))
    for i in range(1, n):
        var j = i
        while j > 0:
            var a = Int(out[j - 1])
            var b = Int(out[j])
            var ka = splitmix_pair(a, salt)
            var kb = splitmix_pair(b, salt)
            if kb < ka or (kb == ka and b < a):
                var t = out[j - 1]
                out[j - 1] = out[j]
                out[j] = t
                j -= 1
            else:
                break
    return out^


def fisher_yates(n: Int, salt: Int) -> List[Int32]:
    """The unpinned spelling of a shuffle from the same stream."""
    var out = List[Int32]()
    for i in range(n):
        out.append(Int32(i))
    var i = n - 1
    while i > 0:
        var j = Int(splitmix_pair(i, salt) % UInt64(i + 1))
        var t = out[i]
        out[i] = out[j]
        out[j] = t
        i -= 1
    return out^
