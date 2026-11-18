# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of johnson_lindenstrauss_min_dim (bindings/jl_native.mojo): the
exact correctly rounded cube and the pinned log of the bound. Typed Mojo
only, no Python import (tools/python_free_dirs.sh); moved unchanged, so the
bits are the same (lane rs-decomposition, 2026-11-12)."""
from std.memory import bitcast
from x_metrics.epilogue import portable_log_c


comptime _LIMB = UInt64(0xFFFFFFFF)
#: Limbs (base 2^32) of the exact cube of a 53-bit mantissa: 159 bits.
comptime _CUBE_LIMBS = 6

# Perf (lane fb12-rows, 2026-11-17; algos/jl-min-dim, one public call per grid
# point): the exact cube is built in fixed stack limbs. Before: three heap
# Lists per call (`_limbs_mul` of List[UInt64], deleted here) and
# bounds-checked List reads in the bit loops. The same schoolbook product
# and the same rounding, so the same bits.


def _cube_limbs(m: UInt64) -> Array[UInt64, _CUBE_LIMBS]:
    """The exact m^3 of a mantissa m < 2^53 as little-endian base-2^32
    limbs (the square, then the square times m; each row's carry lands in
    the next untouched limb)."""
    var a = Array[UInt64, 2](fill=UInt64(0))
    a[0] = m & _LIMB
    a[1] = m >> 32
    var square = Array[UInt64, 4](fill=UInt64(0))
    for i in range(2):  # driver-loop: over the two limbs of a 53-bit mantissa
        var carry = UInt64(0)
        for j in range(2):  # driver-loop: over the two limbs of a 53-bit mantissa
            var t = a[i] * a[j] + square[i + j] + carry
            square[i + j] = t & _LIMB
            carry = t >> 32
        square[i + 2] = carry
    var cube = Array[UInt64, _CUBE_LIMBS](fill=UInt64(0))
    for i in range(4):  # driver-loop: over the four limbs of the square
        var carry = UInt64(0)
        for j in range(2):  # driver-loop: over the two limbs of a 53-bit mantissa
            var t = square[i] * a[j] + cube[i + j] + carry
            cube[i + j] = t & _LIMB
            carry = t >> 32
        cube[i + 2] = carry
    return cube^


@always_inline
def _bit(a: Array[UInt64, _CUBE_LIMBS], i: Int) -> Bool:
    return ((a[i // 32] >> UInt64(i % 32)) & UInt64(1)) != 0


def _bit_length(a: Array[UInt64, _CUBE_LIMBS]) -> Int:
    var k = _CUBE_LIMBS - 1
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
    var power = _cube_limbs(m)
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
