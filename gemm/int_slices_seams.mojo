# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SEAMS OF THE RESEARCH PROFILE `mojolearn.identical.gemm.fp32.v2-int`
(prototype, lane/gemm-int-slices, 2026-10-28; switch `-D MOJOLEARN_IDN_GEMM_INT_SLICES`).

PROTOTYPE, NOT A SHIPPED PROFILE. The contract is `docs/plans/GEMM_INT_SLICES.md`.
Device plans: `gemm/int_slices.mojo`. Host column: `gemm/host/gemm_int_slices_oracle.mojo`.
Gate: `gemm/checks/gemm_int_slices_check.mojo`.

A file of its own so that no binding which imports `checks/numerics.mojo` or
`checks/numerics_int15.mojo` is built from a changed source (the int15 file's
reason). Every function here is a plain host-and-device function; the device
plans, the flat plan and the host oracle call the SAME functions, so the bits
of a cell are a function of the integers alone.

THE CONSTRUCTION (exact integer slicing, Ozaki style)
-----------------------------------------------------
S1 Row exponent. For every row of op(A) and every column of op(B) along `k`,
   `e = floor(log2 absmax) - 27`, read from the exponent field of the absmax
   after the denormal flush (a maximum: the same float under every grouping;
   a NaN never wins). An all-zero row takes `e = 0`.
S2 Code. `q = rne(ftz(x) * 2^-e)` as an integer, `|q| <= 2^28 - 1`. The
   scaling is done on the exponent field (exact; a result below the normal
   range is a signed zero, which `rne` sends to 0 as it would the exact
   value, since it is below 1/2). The rounding is ties-to-even with the
   `2^23` magic constant for `|v| < 2^23`; at and above `2^23` a float32 is
   already an integer. NaN codes to 0; an infinity clamps to `+-(2^28 - 1)`.
   The largest magnitude of a row lands in `[2^27, 2^28)`: its 24-bit
   significand is carried EXACTLY (28 >= 24), and so is every value within
   `2^4` of it. A value further below its row maximum keeps the bits above
   `2^e` only: this is the one precision change of the profile.
S3 Halves and planes. `hi = q >> 14` (arithmetic, in `[-16384, 16383]`),
   `lo = q & 0x3FFF` (in `[0, 16383]`), `q = hi * 2^14 + lo`. Each half is
   an int15-style pair of int8 planes: `h = half >> 7`, `l = half & 127`
   (so the high plane of `hi` is any int8, every other plane is in
   `[0, 127]`, which is the operand domain `INT8_PIECES_MAX_K` is stated
   for in `gemm/checks/gemm_int8_mma_tuned.mojo`).
S4 Products. Four half products per cell, each an exact integer:
   `P_hh = sum hi_a hi_b`, `P_hl = sum hi_a lo_b`, `P_lh = sum lo_a hi_b`,
   `P_ll = sum lo_a lo_b`. On a column with an integer matrix unit each is
   the tuned four-plane kernel's three Int32 sums recombined by
   `int15_recombine` (exact for `k <= 65536`); on every other column and in
   the host oracle each is an Int64 sum of Int32 products, `p` ascending.
   An integer sum is the same integer under every order, tile, k-split and
   vendor, so the four Int64 values of a cell are the same on every column.
S5 Recombination. `T = P_hh 2^28 + (P_hl + P_lh) 2^14 + P_ll`, the exact
   integer `sum q_a q_b`, held as `(H, L)` with `T = H 2^28 + L`,
   `L in [0, 2^28)` (`int_slices_recombine`), all in Int64 with no overflow
   (bounds at the function).
S6 Conversion. ONE round-to-nearest-even of `|T|` to a 24-bit significand
   (`int_slices_t_to_f32`, through `i64_to_f32_pinned` with a sticky bit).
S7 Scale. The rounded value times `2^(ea + eb)` on the exponent field,
   exact while the result is normal; below the normal range a signed zero
   (the IDENTICAL denormal policy), above it the signed infinity.

WHAT THE CELL IS. `C_ij = RN_f32( sum_p q_a(i,p) q_b(j,p) * 2^(ea_i + eb_j) )`
with one rounding, i.e. the correctly rounded value of the exact dot product
of the CODED operands. Relative to fp32.v1 (one FMA chain per leaf, a
balanced tree over leaves) this is a BITS change (different fold, one
rounding instead of k) AND a PRECISION change on the operand side (S2:
values more than 2^4 below their row maximum lose low bits). The plan doc
states the error bound and that the quality metric must be checked.
"""

from std.memory import bitcast

from checks.numerics_int15 import i64_to_f32_pinned, int15_recombine
from gemm.contract import OP_NT, OP_TN

#: The row maximum lands in `[2^27, 2^28)` (clause S1).
comptime INT_SLICES_TARGET_EXPONENT = 27

#: The largest code magnitude, `2^28 - 1` (clause S2).
comptime INT_SLICES_CODE_MAX = 268435455

#: The largest `k` the profile admits. Derivation (clause S4): the tuned
#: four-plane kernel's three Int32 sums are exact to `k = 65536` for the
#: plane domain of S3 (`INT8_PIECES_MAX_K`); the Int64 flat sums are exact
#: far beyond (`|hi_a hi_b| <= 2^28`, `2^28 * 2^16 = 2^44`); and S5's bounds
#: below hold for `k <= 2^16`. A bound on `k` from the arithmetic only: no
#: shape rule.
comptime INT_SLICES_MAX_K = 65536

comptime _TWO23 = Float32(8388608.0)
comptime _TWO28_F = Float32(268435456.0)
comptime _MASK28 = Int64(268435455)
comptime _TWO28_I = Int64(268435456)
comptime _TWO34_I = Int64(17179869184)


def int_slices_scale_pow2(x: Float32, s: Int) -> Float32:
    """`x * 2^s` on the exponent field (clauses S2 and S7). Exact while the
    result is normal. A zero or subnormal `x` is its signed zero (the flush);
    a result whose exponent falls below the normal range is the signed zero;
    above it, the signed infinity. An infinity or NaN `x` is returned as is.
    No float multiply, so no backend's FMA, contraction or subnormal mode
    can reach it."""
    var bits = bitcast[DType.uint32](x)
    var sign = bits & UInt32(0x80000000)
    var field = Int((bits >> UInt32(23)) & UInt32(0xFF))
    if field == 0:
        return bitcast[DType.float32](sign)
    if field == 255:
        return x
    var nf = field + s
    if nf >= 255:
        return bitcast[DType.float32](sign | UInt32(0x7F800000))
    if nf <= 0:
        return bitcast[DType.float32](sign)
    return bitcast[DType.float32](sign | (UInt32(nf) << UInt32(23)) | (bits & UInt32(0x7FFFFF)))


def int_slices_abs_ftz(x: Float32) -> Float32:
    """`|ftz(x)|` from the bits: a subnormal is 0, a NaN stays a NaN (and
    never wins a `>` maximum)."""
    var bits = bitcast[DType.uint32](x) & UInt32(0x7FFFFFFF)
    if (bits >> UInt32(23)) == UInt32(0):
        return Float32(0.0)
    return bitcast[DType.float32](bits)


def int_slices_row_exponent(absmax: Float32) -> Int:
    """Clause S1. `absmax` is the row's maximum of `int_slices_abs_ftz`.
    `e = floor(log2 absmax) - 27`; an all-zero row takes 0. An infinity's
    field reads 128, so its row takes 101 and its finite values code to
    small integers (outside the prototype's admitted inputs, but defined)."""
    if absmax == Float32(0.0) or absmax != absmax:
        return 0
    var field = Int((bitcast[DType.uint32](absmax) >> UInt32(23)) & UInt32(0xFF))
    return field - 127 - INT_SLICES_TARGET_EXPONENT


def int_slices_rne(v: Float32) -> Float32:
    """Round to the nearest integer, ties to even, for every float32: the
    `2^23` magic constant below `2^23` (the sum lies in `[2^23, 2^24)`,
    whose spacing is 1, so the one IEEE addition is the rounding), and the
    value itself at and above (a float32 there is an integer)."""
    if v >= _TWO23 or v <= -_TWO23:
        return v
    if v >= Float32(0.0):
        return (v + _TWO23) - _TWO23
    return (v - _TWO23) + _TWO23


def int_slices_code(x: Float32, e: Int) -> Int32:
    """Clause S2: `q = clamp(rne(ftz(x) * 2^-e), -(2^28 - 1), 2^28 - 1)`."""
    if x != x:
        return Int32(0)
    var s = int_slices_scale_pow2(x, -e)
    var r = int_slices_rne(s)
    if r >= _TWO28_F:
        return Int32(INT_SLICES_CODE_MAX)
    if r <= -_TWO28_F:
        return Int32(-INT_SLICES_CODE_MAX)
    return Int32(Int(r))


def int_slices_half_hi(q: Int32) -> Int32:
    """Clause S3: `q >> 14`, arithmetic, in `[-16384, 16383]`."""
    return q >> Int32(14)


def int_slices_half_lo(q: Int32) -> Int32:
    """Clause S3: `q & 0x3FFF`, in `[0, 16383]`."""
    return q & Int32(0x3FFF)


def int_slices_plane_h(half: Int32) -> Int8:
    """Clause S3: the high int8 plane of a half, `half >> 7`."""
    return Int8(half >> Int32(7))


def int_slices_plane_l(half: Int32) -> Int8:
    """Clause S3: the low int8 plane of a half, `half & 127`, in `[0, 127]`."""
    return Int8(half & Int32(127))


def int_slices_half_from_planes(h: Int8, l: Int8) -> Int32:
    """The half back from its planes: `h * 128 + l`, exact."""
    return Int32(h) * Int32(128) + Int32(l)


def int_slices_product_from_sums(hh: Int32, mid: Int32, ll: Int32) -> Int64:
    """One half product of S4 from the four-plane kernel's three exact Int32
    sums: `int15_recombine`, unchanged (two shifts and two Int64 adds)."""
    return int15_recombine(hh, mid, ll)


def int_slices_recombine(p_hh: Int64, p_hl: Int64, p_lh: Int64, p_ll: Int64) -> Tuple[Int64, Int64]:
    """Clause S5: `T = p_hh 2^28 + (p_hl + p_lh) 2^14 + p_ll` as `(H, L)`
    with `T = H 2^28 + L` and `L in [0, 2^28)`.

    BOUNDS for `k <= 2^16`: `|hi| <= 2^14`, `lo < 2^14`, so
    `|p_hh| <= 2^44`, `|p_hl|, |p_lh| < 2^44`, `0 <= p_ll < 2^44`;
    `low = (p_hl + p_lh) 2^14 + p_ll` is below `2^60` in magnitude; the
    carry `low >> 28` (floor) is below `2^32`, so `H = p_hh + carry` is below
    `2^45`. Every step is an exact Int64 operation."""
    var low = ((p_hl + p_lh) << Int64(14)) + p_ll
    var carry = low >> Int64(28)
    var l = low & _MASK28
    return (p_hh + carry, l)


def int_slices_t_to_f32(h: Int64, l: Int64) -> Float32:
    """Clause S6: `T = h 2^28 + l` (`l in [0, 2^28)`) rounded once to float32,
    nearest even. The magnitude is taken as `(mh, ml)`, `ml in [0, 2^28)`.
    Below `2^62` (`mh < 2^34`) it is one Int64 and `i64_to_f32_pinned` is the
    rounding. Above, the low 14 bits fold into a sticky bit: the reduced
    value is at least `2^48`, so its rounding position is at bit 24 or above
    and the sticky bit (bit 0) lies below it; `i64_to_f32_pinned`'s own
    sticky fold (16 bits at and above `2^48`) keeps the same property; the
    rounded value is then scaled by `2^14`, exactly (it is below `2^77`)."""
    var neg = h < Int64(0)
    var mh: Int64
    var ml: Int64
    if not neg:
        mh = h
        ml = l
    elif l == Int64(0):
        mh = -h
        ml = Int64(0)
    else:
        mh = -h - Int64(1)
        ml = _TWO28_I - l
    var f: Float32
    if mh < _TWO34_I:
        f = i64_to_f32_pinned((mh << Int64(28)) | ml)
    else:
        var v = (mh << Int64(14)) | (ml >> Int64(14))
        if (ml & Int64(0x3FFF)) != Int64(0):
            v = v | Int64(1)
        f = int_slices_scale_pow2(i64_to_f32_pinned(v), 14)
    if neg:
        return -f
    return f


def int_slices_cell(p_hh: Int64, p_hl: Int64, p_lh: Int64, p_ll: Int64, e_sum: Int) -> Float32:
    """Clauses S5 to S7: the four exact half products of a cell to its
    float32. A zero `T` is `+0.0`."""
    var hl = int_slices_recombine(p_hh, p_hl, p_lh, p_ll)
    return int_slices_scale_pow2(int_slices_t_to_f32(hl[0], hl[1]), e_sum)

def int_slices_strides(op: Int, m: Int, n: Int, k: Int) -> Tuple[Int, Int, Int, Int]:
    """`(a_row_stride, a_k_stride, b_row_stride, b_k_stride)`: element `(r, p)`
    of a "row along k" of op(A) (r < m) or of op(B) (r < n) sits at
    `r * row_stride + p * k_stride` of the stored row-major operand
    (contract section 0.1's three orientations)."""
    var ars = k
    var aps = 1
    if op == OP_TN:
        ars = 1
        aps = m
    var brs = 1
    var bps = n
    if op == OP_NT:
        brs = k
        bps = 1
    return (ars, aps, brs, bps)
