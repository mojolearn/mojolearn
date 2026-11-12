# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of johnson_lindenstrauss_min_dim (bindings/jl_native.mojo): the
exact correctly rounded cube and the pinned log of the bound. Typed Mojo
only, no Python import (tools/python_free_dirs.sh); moved unchanged, so the
bits are the same (lane rs-decomposition, 2026-11-12)."""
from std.memory import bitcast
from x_metrics.epilogue import portable_log_c


comptime _LIMB = UInt64(0xFFFFFFFF)


def _limbs_mul(a: List[UInt64], b: List[UInt64]) -> List[UInt64]:
    """The exact product of two little-endian base-2^32 integers."""
    var out = List[UInt64](length=len(a) + len(b), fill=UInt64(0))
    for i in range(len(a)):  # driver-loop: over the limbs of a 53-bit power (at most five)
        var carry = UInt64(0)
        for j in range(len(b)):  # driver-loop: over the limbs of a 53-bit mantissa (two)
            var t = a[i] * b[j] + out[i + j] + carry
            out[i + j] = t & _LIMB
            carry = t >> 32
        var k = i + len(b)
        while carry != 0:  # driver-loop: over the carry's limbs
            var t = out[k] + carry
            out[k] = t & _LIMB
            carry = t >> 32
            k += 1
    return out^


def _bit(a: List[UInt64], i: Int) -> Bool:
    return ((a[i // 32] >> UInt64(i % 32)) & UInt64(1)) != 0


def _bit_length(a: List[UInt64]) -> Int:
    var k = len(a) - 1
    while k >= 0 and a[k] == 0:  # driver-loop: over the limbs of an exact power
        k -= 1
    if k < 0:
        return 0
    var top = a[k]
    var bits = 0
    while top != 0:  # driver-loop: over the bits of one limb
        top >>= 1
        bits += 1
    return k * 32 + bits


def _cube_rounded(x: Float64) -> Float64:
    """x^3 correctly rounded to nearest-even for a finite positive x whose
    cube is normal (`_portable_math.powi(x, 3)`); x = m 2^e, the exact m^3
    rounded once to 53 bits."""
    var bits = bitcast[DType.uint64](x)
    var field = Int((bits >> 52) & UInt64(0x7FF))
    if field == 0 or field < 1023 - 300:
        # A subnormal or tiny eps: eps^3 / 3 is below half an ulp of
        # eps^2 / 2 by far more than any rounding of the cube, so the
        # bound's denominator is the same for every rounding of it.
        return x * x * x
    var m = (bits & UInt64(0xFFFFFFFFFFFFF)) | UInt64(0x10000000000000)
    var e = field - 1075
    var mm: List[UInt64] = [m & _LIMB, m >> 32]
    var power = _limbs_mul(_limbs_mul(mm, mm), mm)
    var length = _bit_length(power)
    var shift = length - 53
    var q = UInt64(0)
    for i in range(53):  # driver-loop: over the 53 kept bits of the exact cube
        if _bit(power, shift + i):
            q |= UInt64(1) << UInt64(i)
    var half = _bit(power, shift - 1)
    var sticky = False
    for i in range(shift - 1):  # driver-loop: over the discarded bits of the exact cube
        if _bit(power, i):
            sticky = True
            break
    if half and (sticky or (q & UInt64(1)) != 0):
        q += 1
        if q == (UInt64(1) << 53):
            q >>= 1
            shift += 1
    # q in [2^52, 2^53): the normal binary64 q 2^(shift + 3e), built from its fields
    var biased = shift + 3 * e + 52 + 1023
    return bitcast[DType.float64]((UInt64(biased) << 52) | (q & UInt64(0xFFFFFFFFFFFFF)))


def _pinned_log(x: Float64) -> Float64:
    # `_portable_math.log` of a positive x: the special values as
    # `special_log`, finite values through the pinned polynomial.
    if x - x != 0.0:
        return x  # NaN, +inf
    return portable_log_c(x)
