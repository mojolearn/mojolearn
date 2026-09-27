# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fixtures and comparisons shared by the ann lane's seam checks."""

from std.memory import bitcast


def hash_u(i: Int, salt: Int) -> UInt64:
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt) * UInt64(0xD1B54A32D192ED03)
    h = h ^ (h >> 33)
    h = h * UInt64(0xFF51AFD7ED558CCD)
    return h ^ (h >> 29)


def fixture_wide(n: Int, dim: Int) -> List[Float32]:
    """Signed values over six decades per column: fold orders separate."""
    var x = List[Float32](capacity=n * dim)
    for i in range(n):
        for c in range(dim):
            var u = Int(hash_u(i * dim + c, 7) % UInt64(20001)) - 10000
            var scale = Float32(1.0)
            for _ in range(c % 7):
                scale = scale * Float32(10.0)
            x.append(Float32(u) * Float32(0.0001) * scale * Float32(0.001))
    return x^


def fixture_ties(n: Int, dim: Int) -> List[Float32]:
    """Values in {0, 1, 2, 3}: exact distance ties everywhere."""
    var x = List[Float32](capacity=n * dim)
    for i in range(n):
        for c in range(dim):
            x.append(Float32(Int(hash_u(i * dim + c, 11) % UInt64(4))))
    return x^


def same_f32(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def same_i32(a: List[Int32], b: List[Int32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def report(name: String, ok: Bool, mut failed: Int):
    print("  ", "OK  " if ok else "FAIL", name)
    if not ok:
        failed += 1


