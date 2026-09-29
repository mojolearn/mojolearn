# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host bindings that serve the int15 profile
# (python/mojolearn/host_surface.py names which); product, not only a check.
"""THE SEAMS OF `mojolearn.identical.gemm.int15i64.v1`.

Lane lane/lowbit-int15, 2026-09-29, DEVIATIONS 2965 to 2971. Contract
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`, the section headed THE FIFTEEN-BIT
PROFILE, clauses W-1 to W-7. Answers `gemm/host/gemm_int15_oracle.mojo`;
device `gemm/checks/gemm_int15.mojo`; gates `gemm/checks/gemm_int15_check.mojo`.

A FILE OF ITS OWN, ON PURPOSE. Every binding of the package imports
`checks/numerics.mojo`, so one line added there changes the source every one
of them is built from. These seams are the low-bit block of that file one
step wider, and they are built from its helpers (`ftz`, `identical_mul`,
`pow2_f32`, `f32_exponent`, `f32_round_half_even`) without changing one
character of them, so `fp32.v1`, `bf16f32.v1` and `int8i32.v1` are
compiled from the bytes they were compiled from before this lane.

THE CODES ARE THE QUALITY LANE'S. `bench/lowbit_quality/arith.py`, kind
`int15`, is the arithmetic whose held-out perplexity was measured, and the
two functions `int15_row_exponent` and `quantize_int15_value` are that rule
and nothing else. `gemm/checks/gemm_int15_sim_check.mojo` compares the host
oracle with that simulation's exported vectors bit for bit.

Every seam here is a bit-level construction, for the reason the low-bit
block gives: no conversion trusts a backend's `cast`.
"""

from std.memory import bitcast

from checks.numerics import (
    f32_exponent,
    f32_round_half_even,
    ftz,
    identical_mul,
    pow2_f32,
)

#: The quantizer scales a row so its largest magnitude lands in
#: `[8192, 16384)`: one exponent below the fifteen-bit ceiling, which keeps
#: `rne(x * 2^-e)` at most 16384 and the clamp a one-sided event. It is
#: `INT8_TARGET_EXPONENT` plus seven, the seven bits of the second piece.
comptime INT15_TARGET_EXPONENT = 13

#: The largest code magnitude. The clamp is symmetric so `-q` is always a
#: code, and a code is `hi * 128 + lo` with both pieces int8 (clause W-3).
comptime INT15_CODE_MAX = 16383

#: Bits of the low piece. `lo = c mod 2^7` lies in `[0, 127]`.
comptime INT15_PIECE_BITS = 7


def int15_row_exponent(absmax: Float32) -> Int:
    """DEVIATION 2965: the power-of-two scale of a row, clause W-1.
    `e = floor(log2 absmax) - 13`, so `absmax * 2^-e` is in `[8192, 16384)`.
    An all-zero row (absmax `0`, after the flush) takes `e = 0`. The range
    of `e` over every float32 row is `[-139, 115]`: the smallest normal has
    exponent -126 and the infinity's field reads 128."""
    var a = ftz(absmax)
    if a == Float32(0.0) or a != a:
        return 0
    return f32_exponent(a) - INT15_TARGET_EXPONENT


def quantize_int15_value(x: Float32, e: Int) -> Int16:
    """DEVIATION 2966: `q = clamp(rne(ftz(ftz(x) * 2^-e)), -16383, 16383)`,
    clause W-2. `quantize_int8_value` one step wider, with ONE line more.

    THE LINE MORE. When the row's absmax is below `2^-114` the exponent is
    below -127 and `2^-e` is the infinity (`pow2_f32`), so a zero of that
    row scales to `0 * inf`, a NaN. The simulation that defines the codes
    sends a NaN to the code 0 AFTER the scaling (`nan_to_num`), so this
    function tests the scaled value as well as the input. Converting a NaN
    to an integer is not defined on any backend, and the test is what keeps
    that conversion from ever being asked for. The nonzero values of such a
    row scale to an infinity and clamp to the largest code, as the
    simulation's do.

    `f32_round_half_even` is exact for magnitudes up to `2^22`; a scaled
    finite value is below `2^14`, and an infinity passes through it
    unchanged and is clamped."""
    if x != x:
        return Int16(0)
    var s = ftz(identical_mul(ftz(x), pow2_f32(-e)))
    if s != s:
        return Int16(0)
    var r = f32_round_half_even(s)
    if r > Float32(16383.0):
        r = Float32(16383.0)
    if r < Float32(-16383.0):
        r = Float32(-16383.0)
    return Int16(Int(r))


def int15_piece_lo(c: Int16) -> Int8:
    """DEVIATION 2967: the low piece, clause W-3. `c mod 128` as a FLOOR
    modulus, in `[0, 127]` for a negative code too: the low seven bits of
    the two's complement word. A mask, never a division."""
    return Int8(Int32(c) & Int32(127))


def int15_piece_hi(c: Int16) -> Int8:
    """DEVIATION 2967: the high piece, clause W-3. `floor(c / 128)`, an
    arithmetic shift, never a division. It lies in `[-128, 127]`: a code in
    `[-16383, -16257]` has `hi = -128` and `lo = c + 16384` in `[1, 127]`,
    so the piece reaches the one int8 value whose negation is not an int8.
    Nothing negates a piece."""
    return Int8(Int32(c) >> Int32(7))


def int15_recombine(hh: Int32, mid: Int32, ll: Int32) -> Int64:
    """DEVIATION 2969: the four piece sums back to the sum of code
    products, clause W-5. With `a = ah * 128 + al` and `b = bh * 128 + bl`,
    `sum(a * b) = HH * 2^14 + (HL + LH) * 2^7 + LL`, and `mid` is `HL + LH`
    (clause W-4 says why they share one Int32). Two shifts and two
    additions in Int64, every one exact: `|HH * 2^14| <= 2^44`,
    `|mid * 2^7| < 2^38`, `LL < 2^30`. A shift of a negative Int64 is the
    multiplication by the power of two, in two's complement."""
    return (Int64(hh) << Int64(14)) + (Int64(mid) << Int64(7)) + Int64(ll)


def i64_to_f32_pinned(v: Int64) -> Float32:
    """DEVIATION 2970: the Int64 sum as a float32, round to nearest even,
    clause W-6. The wider sibling of `i32_to_f32_pinned`.

    BELOW `2^48`: two exact conversions (a 24-bit high part and a 24-bit
    low part, each an integer a float32 holds exactly, each converted from
    an Int32 so no 64-bit conversion is asked of any backend), one exact
    scaling of the high part by `2^24`, and ONE float32 addition. The exact
    sum of the two addends is the integer itself, so the addition's
    rounding is the conversion's rounding, and that one is IEEE's. A code
    generator that fuses the scaling into the addition changes nothing: the
    product is exact, so the fused result is the rounded sum of the same
    two numbers. Every sum the profile admits is below `2^44` (clause W-4).

    AT AND ABOVE `2^48`, so the seam is total over Int64: the low sixteen
    bits are dropped and a STICKY bit is set in their place when any was
    set. The reduced magnitude is below `2^48` and at least `2^32`, so its
    rounding position is at bit 8 or above, the sticky bit lies below it,
    and the rounding of the reduced magnitude decides exactly as the
    rounding of the whole would. The result is then scaled by `2^16`,
    exactly. The magnitude is taken in UInt64 so the most negative Int64
    has one."""
    var neg = v < Int64(0)
    var mag = bitcast[DType.uint64](v)
    if neg:
        mag = UInt64(0) - mag
    var reduced = False
    if mag >= (UInt64(1) << UInt64(48)):
        var lost = mag & UInt64(0xFFFF)
        mag = mag >> UInt64(16)
        if lost != UInt64(0):
            mag = mag | UInt64(1)
        reduced = True
    var hi = Float32(Int32(mag >> UInt64(24)))
    var lo = Float32(Int32(mag & UInt64(0xFFFFFF)))
    var r = hi * Float32(16777216.0) + lo
    if reduced:
        r = r * Float32(65536.0)
    if neg:
        return -r
    return r


def dequant_int15_pinned(acc: Int64, e_sum: Int) -> Float32:
    """DEVIATION 2971: the dequantization seam, clauses W-6 and W-7.
    `ftz(acc_f32 * 2^(ea + eb))`: `dequant_int8_pinned` with the wider
    conversion. `ea + eb` lies in `[-278, 230]`; above 127 the scale is the
    infinity and below -126 it is `+0.0` (`pow2_f32`), so a sum of zero
    under an infinite scale is a NaN, as it is in the simulation."""
    return ftz(identical_mul(i64_to_f32_pinned(acc), pow2_f32(e_sum)))


def dequant_int15_code(c: Int16, e: Int) -> Float32:
    """`ftz(code * 2^e)`: the float32 a stored code stands for. A code is
    below `2^24`, so its conversion is exact on any spelling; it goes
    through the pinned seam so there is one spelling."""
    return dequant_int15_pinned(Int64(c), e)
