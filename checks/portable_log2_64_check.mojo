# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-only portable_log2_64 admission: 2^18 deterministic hashed positive
finite doubles across every exponent band, including subnormals; exact powers
of two across the full binary64 range; branch boundaries and special values.
Each finite result must be within 2 ulp of platform libm log2 via FFI.
The FNV-1a hash records our arithmetic independently of the reference. Matching
hashes on separately executed hosts, not this one run, establish host agreement.

    pixi run check-portable-log2-64
"""
from std.ffi import external_call

from checks.numerics import portable_log2_64


def _splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _bits(x: Float64) -> UInt64:
    from std.memory import bitcast

    return bitcast[DType.uint64](x)


def log(x: Float64) -> Float64:
    return external_call["log2", Float64](x)


def _ulps(a: Float64, b: Float64) -> Int:
    var ia = Int(_bits(a))
    var ib = Int(_bits(b))
    var d = ia - ib
    return d if d >= 0 else -d


def main() raises:
    from std.memory import bitcast

    var n = 1 << 18
    var h = UInt64(0xCBF29CE484222325)
    var worst = 0
    var worst_x = Float64(0.0)
    var over2 = 0
    for i in range(n):
        var r = _splitmix(UInt64(i) + UInt64(0xA5A5))
        # Normal exponents 1..2046; force subnormal draws as well.
        var ebits = UInt64(1) + (r >> 12) % UInt64(2046)
        if i % 97 == 0:
            ebits = UInt64(0)  # a denormal
        var mant = r & UInt64(0x000FFFFFFFFFFFFF)
        var x = bitcast[DType.float64]((ebits << 52) | mant)
        if x <= Float64(0.0) or x != x:
            continue
        var got = portable_log2_64(x)
        var want = log(x)
        var u = _ulps(got, want)
        if u > worst:
            worst = u
            worst_x = x
        if u > 2:
            over2 += 1
        var gb = _bits(got)
        for b in range(8):
            h = (h ^ ((gb >> UInt64(8 * b)) & UInt64(0xFF))) * UInt64(0x100000001B3)
    # the edges, exact
    if portable_log2_64(1.0) != 0.0:
        raise Error("portable_log2_64(1) != 0")
    if portable_log2_64(bitcast[DType.float64](UInt64(0x7FF0000000000000))) != bitcast[DType.float64](UInt64(0x7FF0000000000000)):
        raise Error("portable_log2_64(+inf) != +inf")
    if _bits(portable_log2_64(0.0)) != UInt64(0xFFF0000000000000):
        raise Error("portable_log2_64(0) != -inf")
    if portable_log2_64(-1.0) == portable_log2_64(-1.0):
        raise Error("portable_log2_64(-1) is not NaN")
    if _bits(portable_log2_64(bitcast[DType.float64](UInt64(0x8000000000000000)))) != UInt64(0xFFF0000000000000):
        raise Error("portable_log2_64(-0) != -inf")
    var nonfinite: List[UInt64] = [0x7FF8000000001234, 0xFFF0000000000000]
    for nb in nonfinite:
        var nv = portable_log2_64(bitcast[DType.float64](nb))
        if nv == nv:
            raise Error("portable_log2_64(NaN/-inf) is not NaN")
    for k in range(-1074, 1024):
        var p = bitcast[DType.float64](UInt64(k + 1023) << 52) if k > -1023 else bitcast[DType.float64](UInt64(1) << UInt64(k + 1074))
        var got = portable_log2_64(p)
        if _bits(got) != _bits(Float64(k)):
            raise Error("portable_log2_64(2^" + String(k) + ") is not exact")
    # Adjacent words around one, sqrt(1/2), normal/subnormal transition,
    # and each exponent-arm boundary. Include near-overflow explicitly.
    var boundaries: List[UInt64] = [
        1, 0x000FFFFFFFFFFFFF, 0x0010000000000000,
        0x3FE6A09E667F3BCC, 0x3FF0000000000000,
        0x3FC0000000000000, 0x3FD0000000000000,
        0x4000000000000000, 0x4010000000000000,
        0x7FEFFFFFFFFFFFFF,
    ]
    var edge_count = 0
    for center in boundaries:
        for offset in range(-4, 5):
            var word = Int(center) + offset
            if word <= 0 or UInt64(word) >= UInt64(0x7FF0000000000000):
                continue
            var x = bitcast[DType.float64](UInt64(word))
            var u = _ulps(portable_log2_64(x), log(x))
            if u > 2:
                raise Error("portable_log2_64 boundary at bits " + String(word) + " off by " + String(u) + " ulp")
            edge_count += 1
    print("portable_log2_64: 2098 powers exact; boundary inputs", edge_count)
    print(
        "portable_log2_64:", n, "hashed doubles; worst", worst, "ulp from this host's"
        " libm at x =", worst_x, ";", over2, "beyond 2 ulp; edges exact"
    )
    print("portable_log2_64 device-independent hash:", h)
    if over2 != 0:
        raise Error("portable_log2_64: " + String(over2) + " results beyond 2 ulp of the host libm")
    print("portable log2_64 check OK")
