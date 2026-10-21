# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fifteen-bit profile's gates: the seams, the bounds with a planted
case at each boundary, oracle agreement on the device, plan agreement,
planted worst cases, batch invariance, and the sabotage that shows each can
fail.

    pixi run check-gemm-int15                  every gate passes
    pixi run check-gemm-int15-force-flat       the dispatchers pinned off the unit; every gate passes
    pixi run check-gemm-int15-sabotage         MUST FAIL the device oracle gates
    pixi run check-gemm-int15-host-sabotage    MUST FAIL the host and the device oracle gates
    pixi run check-gemm-int15-piece-sabotage   MUST FAIL the split gate and the planted worst cases

Profile `mojolearn.identical.gemm.int15i64.v1`. Contract
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`, the section headed THE FIFTEEN-BIT
PROFILE; kernels `gemm/checks/gemm_int15.mojo`; answers
`gemm/host/gemm_int15_oracle.mojo`; seams `checks/numerics_int15.mojo`.
Lane lane/lowbit-int15, 2026-09-29.

EVERY PRODUCT GATE PRINTS A DIGEST LINE, `DIGEST <case> <plan> <hex>`: a
64-bit FNV-1a of the output words, NaN cells as one word. The host
oracle's line is `plan=oracle`. Equal lines across boxes are the
cross-vendor and cross-generation comparison; `tools/lowbit_int15/
digests.py` reads them.

THE MATRIX-UNIT PLAN runs only on a column whose `lib_int8_matrix_unit_for`
row is True; on any other column the plan gates print that it did not run,
which is not a pass.

MAIN RUNS EVERY GATE AND REPORTS EVERY VERDICT before it raises, as
`gemm_lowbit_check.mojo` does and for the same reason.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.sys import has_accelerator

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN, column_name, lib_int8_matrix_unit_for
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    numeric_mode_name,
)
from checks.numerics_int15 import (
    INT15_CODE_MAX,
    dequant_int15_pinned,
    i64_to_f32_pinned,
    int15_piece_hi,
    int15_piece_lo,
    int15_recombine,
)
from gemm.checks.gemm_int8_mma import INT8_MMA_UNSTATED_LOADS, mma_operands_aligned
from gemm.checks.gemm_int15_apple import (
    INT15_APPLE_FORM_FOUR,
    INT15_APPLE_FORM_TWO,
    INT15_APPLE_GEOMETRY_ROW,
    INT15_APPLE_GEOMETRY_WIDE,
    identical_gemm_int15_apple_with_geometry,
    int15_apple_sabotage_name,
)
from gemm.checks.gemm_int15 import (
    Int15QuantWorkspace,
    Int15Workspace,
    dequantize_planes_int15_device,
    dequantize_rows_int15_device,
    identical_gemm_int15_from_f32,
    int15_quant_chunks,
    quantize_cols_int15_device,
    quantize_int15_parallel_device,
    quantize_planes_int15_parallel_device,
    identical_gemm_int15_flat_into,
    identical_gemm_int15_into,
    identical_gemm_int15_mma_into,
    identical_gemm_int15_pieces_into,
    identical_gemm_int15_planes_into,
    int15_plan_dispatch_name,
    int15_sabotage_name,
    quantize_rows_int15_device,
    split_int15_device,
)
from gemm.contract import INT15_MAX_K
from gemm.host.gemm_int15_oracle import (
    INT15_MID_BOUND_K,
    INT15_PIECE_BOUND_K,
    Int15Rows,
    dequantize_rows_int15,
    join_int15,
    quantize_cols_int15,
    gemm_int15_oracle,
    gemm_int15_oracle_cell,
    gemm_int15_pieces_oracle,
    int15_dot_cell,
    int15_dot_cell_pieces,
    quantize_rows_int15,
    split_int15,
)
from gemm.contract import GEMM_ORACLE_HOST_SABOTAGE

comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime POISON = Float32(-987654.0)
comptime HAS_UNIT = lib_int8_matrix_unit_for[TARGET_COLUMN]()
comptime IS_APPLE = TARGET_COLUMN == COLUMN_APPLE

comptime PLAN_FLAT = 0
comptime PLAN_PIECES = 1
comptime PLAN_MMA = 2
comptime PLAN_DISPATCH_CODES = 3
comptime PLAN_DISPATCH_PLANES = 4
comptime PLAN_APPLE_WIDE = 5
comptime PLAN_APPLE_ROW = 6
comptime PLAN_APPLE4_WIDE = 7
comptime PLAN_APPLE4_ROW = 8
comptime PLAN_COUNT = 9


def _plan_name(plan: Int) -> String:
    if plan == PLAN_FLAT:
        return String("flat")
    if plan == PLAN_PIECES:
        return String("pieces")
    if plan == PLAN_MMA:
        return String("mma")
    if plan == PLAN_DISPATCH_CODES:
        return String("dispatch-codes")
    if plan == PLAN_APPLE_WIDE:
        return String("apple-two-wide")
    if plan == PLAN_APPLE_ROW:
        return String("apple-two-row")
    if plan == PLAN_APPLE4_WIDE:
        return String("apple-four-wide")
    if plan == PLAN_APPLE4_ROW:
        return String("apple-four-row")
    return String("dispatch-planes")


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _show(x: Float32) -> String:
    return String(x) + "/0x" + hex(_bits(x))


def _is_nan(x: Float32) -> Bool:
    return x != x


def _same(a: Float32, b: Float32) -> Bool:
    """The contract's cell comparison: the same word, a NaN equal to a NaN."""
    if _is_nan(a) and _is_nan(b):
        return True
    return _bits(a) == _bits(b)


def _digest(v: List[Float32]) -> String:
    """FNV-1a, 64 bit, over the output words low byte first; every NaN
    enters as the word 0x7FC00000."""
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(v)):
        var w = _bits(v[i])
        if _is_nan(v[i]):
            w = UInt32(0x7FC00000)
        for b in range(4):
            var byte = UInt64((w >> UInt32(8 * b)) & UInt32(0xFF))
            h = (h ^ byte) * UInt64(0x100000001B3)
    return hex(h)


def _hash64(i: Int, salt: Int) -> UInt64:
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(
        0xBF58476D1CE4E5B9
    )
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _val(i: Int, salt: Int) -> Float32:
    """`gemm_lowbit_check.mojo::_val` with a 20-bit significand: fifteen-bit
    codes keep fourteen magnitude bits, so a 13-bit significand would reach
    the top binade of a row without ever being rounded."""
    var h = _hash64(i, salt)
    var mant = Float32(1.0) + Float32(Int(h & UInt64(0xFFFFF))) / Float32(1048576.0)
    var e = Int((h >> UInt64(21)) & UInt64(7)) - 4
    var scale = Float32(1.0)
    if e >= 0:
        for _ in range(e):
            scale = scale * Float32(2.0)
    else:
        for _ in range(-e):
            scale = scale * Float32(0.5)
    var v = mant * scale
    if (h >> UInt64(28)) & UInt64(1) == UInt64(1):
        return -v
    return v


def _fill(n_elems: Int, salt: Int) -> List[Float32]:
    var v = List[Float32]()
    for i in range(n_elems):
        v.append(_val(i, salt))
    return v^


def _pow2(e: Int) -> Float32:
    """`2^e` by repeated exact doubling or halving: the check's own
    spelling, not the seam's."""
    var s = Float32(1.0)
    if e >= 0:
        for _ in range(e):
            s = s * Float32(2.0)
    else:
        for _ in range(-e):
            s = s * Float32(0.5)
    return s


# ===========================================================================
# THE SEAMS, ON THE HOST
# ===========================================================================


def _i64_to_f32_reference_bits(v: Int64) -> UInt32:
    """The correctly rounded float32 of an Int64 as its WORD, in integer
    arithmetic only: the leading bit, the 24 kept bits, the remainder
    against the half, ties to the even kept word. No float operation
    appears, so it shares no step with `i64_to_f32_pinned`."""
    if v == Int64(0):
        return UInt32(0)
    var sign = UInt32(0)
    var mag = bitcast[DType.uint64](v)
    if v < Int64(0):
        sign = UInt32(0x80000000)
        mag = UInt64(0) - mag
    var p = 63
    while ((mag >> UInt64(p)) & UInt64(1)) == UInt64(0):
        p -= 1
    var mant: UInt64
    if p <= 23:
        mant = mag << UInt64(23 - p)
    else:
        var shift = p - 23
        mant = mag >> UInt64(shift)
        var rem = mag & ((UInt64(1) << UInt64(shift)) - UInt64(1))
        var half = UInt64(1) << UInt64(shift - 1)
        if rem > half or (rem == half and (mant & UInt64(1)) == UInt64(1)):
            mant = mant + UInt64(1)
        if mant == (UInt64(1) << UInt64(24)):
            mant = mant >> UInt64(1)
            p += 1
    var field = UInt32(p + 127)
    return sign | (field << UInt32(23)) | (UInt32(mant & UInt64(0x7FFFFF)))


def _needs_rounding(v: Int64) -> Bool:
    """Whether the integer has more than 24 significant bits, so its
    float32 is a rounded value. Integer arithmetic only."""
    if v == Int64(0):
        return False
    var mag = bitcast[DType.uint64](v)
    if v < Int64(0):
        mag = UInt64(0) - mag
    while (mag & UInt64(1)) == UInt64(0):
        mag = mag >> UInt64(1)
    return mag >= (UInt64(1) << UInt64(24))


def _planted_sums() -> List[Int64]:
    """The rounding boundaries of clause W-6, both signs of each: exact
    values, ties toward each parity, one above and one below a tie, the
    largest sum the profile admits and its neighbors, the reduced branch
    (at and above `2^48`) and the two ends of Int64."""
    var base: List[Int64] = [
        0,
        1,
        16383,
        16777215,
        16777216,
        16777217,
        16777218,
        16777219,
        16777220,
        33554434,
        33554435,
        1099511627776,
        1099511693312,
        1099511693311,
        1099511693313,
        1099511824384,
        1099511824383,
        1099511824385,
        17590038626304,
        17590038626303,
        17590038626305,
        17592186044415,
        17592186044416,
        281474976710655,
        281474976710656,
        281474993487872,
        281474993487871,
        281474993487873,
        281475010265088,
        281475027042304,
        281475027042305,
        9223371487098961920,
        9223371761976868863,
        9223371761976868864,
        9223371761976868865,
        9223372036854775807,
    ]
    var out = List[Int64]()
    for i in range(len(base)):
        out.append(base[i])
        out.append(-base[i])
    # The most negative Int64, whose magnitude no Int64 holds.
    out.append(-Int64(9223372036854775807) - Int64(1))
    return out^


def check_int15_pieces_cover_every_code() raises:
    """W-3 (a): every code in `[-16383, 16383]` splits into a low piece in
    `[0, 127]` and a high piece in `[-128, 127]` with `hi * 128 + lo` the
    code; the high piece is -128 on exactly the 127 codes from -16383 to
    -16257, and +127 is reached."""
    var at_min = 0
    var at_max = 0
    for ci in range(-INT15_CODE_MAX, INT15_CODE_MAX + 1):
        var c = Int16(ci)
        var lo = Int(int15_piece_lo(c))
        var hi = Int(int15_piece_hi(c))
        if lo < 0 or lo > 127:
            raise Error("code " + String(ci) + ": low piece " + String(lo) + " outside [0, 127]")
        if hi < -128 or hi > 127:
            raise Error("code " + String(ci) + ": high piece " + String(hi) + " outside [-128, 127]")
        if hi * 128 + lo != ci:
            raise Error("code " + String(ci) + ": hi * 128 + lo = " + String(hi * 128 + lo))
        if hi == -128:
            at_min += 1
            if ci < -16383 or ci > -16257:
                raise Error("code " + String(ci) + " has the high piece -128")
        if hi == 127 and lo == 127:
            at_max += 1
    if at_min != 127:
        raise Error("expected 127 codes with the high piece -128, saw " + String(at_min))
    if at_max != 1:
        raise Error("the code 16383 did not split into (127, 127)")
    print("   ok 32767 codes split; 127 of them carry the high piece -128")


def check_int15_sum_to_float_is_correctly_rounded() raises:
    """W-6 (g): `i64_to_f32_pinned` against the integer-only spelling at
    every planted boundary, and the fixture shown to separate round to
    nearest even from truncation and from round half away."""
    var sums = _planted_sums()
    var inexact = 0
    for i in range(len(sums)):
        var v = sums[i]
        var got = _bits(i64_to_f32_pinned(v))
        var want = _i64_to_f32_reference_bits(v)
        if got != want:
            raise Error(
                "i64_to_f32_pinned(" + String(v) + ") = 0x" + hex(got)
                + ", the correctly rounded word is 0x" + hex(want)
            )
        if _needs_rounding(v):
            inexact += 1
    if inexact < 20:
        raise Error("the planted sums hold only " + String(inexact) + " inexact values; they are not evidence of a rounding")
    # Separation: 2^24 + 1 is a tie whose even neighbor is BELOW, 2^24 + 3 a
    # tie whose even neighbor is ABOVE. Truncation gets the second wrong,
    # round half away gets the first wrong.
    if _bits(i64_to_f32_pinned(Int64(16777217))) != _bits(Float32(16777216.0)):
        raise Error("the tie 2^24 + 1 did not round to the even neighbor below")
    if _bits(i64_to_f32_pinned(Int64(16777219))) != _bits(Float32(16777220.0)):
        raise Error("the tie 2^24 + 3 did not round to the even neighbor above")
    # The backend's own conversion, recorded and not trusted.
    var native_differs = 0
    for i in range(len(sums)):
        if _bits(Float32(sums[i])) != _bits(i64_to_f32_pinned(sums[i])):
            native_differs += 1
    print(
        "   ok " + String(len(sums)) + " planted sums, " + String(inexact)
        + " of them inexact; the host's own conversion differs on "
        + String(native_differs)
    )


def check_int15_quantizer_is_the_rule() raises:
    """W-1, W-2: codes within `[-16383, 16383]`; the dequantized error
    within one step; the ties of round to nearest even; the clamp; a row of
    zeros; a NaN; an infinity; a row below `2^-114`."""
    var x = _fill(64 * 96, 7)
    var qr = quantize_rows_int15(x, 64, 96)
    var y = dequantize_rows_int15(qr)
    var rounded = 0
    for r in range(64):
        var step = _pow2(Int(qr.e[r]))
        for c in range(96):
            var q = Int(qr.q[r * 96 + c])
            if q < -16383 or q > 16383:
                raise Error("code out of range: " + String(q))
            var err = y[r * 96 + c] - x[r * 96 + c]
            if err < Float32(0.0):
                err = -err
            if err > step + step * Float32(1e-6):
                raise Error("dequantized error " + _show(err) + " exceeds a step " + _show(step))
            if err > Float32(0.0):
                rounded += 1
    if rounded < 1000:
        raise Error("only " + String(rounded) + " of 6144 fixture values were rounded; the fixture does not exercise the rounding")
    # One row whose absmax is 8192 (exponent 13, so e = 0): the codes are
    # the rounded values themselves.
    var row: List[Float32] = [8192.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.4999, -0.0, 16383.0]
    var want: List[Int] = [8192, 0, 2, 2, 0, -2, -2, 3, 0, 16383]
    var rq = quantize_rows_int15(row, 1, len(row))
    if rq.e[0] != Int32(0):
        raise Error("absmax 16383 took exponent " + String(rq.e[0]) + ", want 0")
    for i in range(len(row)):
        if Int(rq.q[i]) != want[i]:
            raise Error("code of " + _show(row[i]) + " is " + String(Int(rq.q[i])) + ", want " + String(want[i]))
    # The clamp: 16383.75 rounds to 16384 and is brought to 16383.
    var top: List[Float32] = [16383.75, -16383.75, 16383.5, 16382.5]
    var tq = quantize_rows_int15(top, 1, 4)
    if tq.e[0] != Int32(0) or Int(tq.q[0]) != 16383 or Int(tq.q[1]) != -16383 or Int(tq.q[2]) != 16383 or Int(tq.q[3]) != 16382:
        raise Error("the clamp or the tie at the top of the range is wrong: " + String(Int(tq.q[0])) + " " + String(Int(tq.q[1])) + " " + String(Int(tq.q[2])) + " " + String(Int(tq.q[3])))
    # A row of zeros takes exponent 0 and the code 0.
    var zeros: List[Float32] = [0.0, -0.0, 0.0, 0.0]
    var zq = quantize_rows_int15(zeros, 1, 4)
    if zq.e[0] != Int32(0):
        raise Error("all-zero row took exponent " + String(zq.e[0]))
    for i in range(4):
        if Int(zq.q[i]) != 0:
            raise Error("a zero took the code " + String(Int(zq.q[i])))
    # A NaN codes to 0 and never wins the absmax; an infinity wins it.
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var inf = bitcast[DType.float32](UInt32(0x7F800000))
    var odd: List[Float32] = [nan, 3.0, -inf, 1.0e30]
    var oq = quantize_rows_int15(odd, 1, 4)
    if oq.e[0] != Int32(115):
        raise Error("a row holding an infinity took exponent " + String(oq.e[0]) + ", want 115")
    if Int(oq.q[0]) != 0 or Int(oq.q[1]) != 0 or Int(oq.q[2]) != -16383:
        raise Error("NaN, 3.0 and -inf coded to " + String(Int(oq.q[0])) + " " + String(Int(oq.q[1])) + " " + String(Int(oq.q[2])))
    comptime if IDENTICAL_BUILD:
        # Below 2^-114 the scale 2^-e is the infinity: a nonzero value takes
        # the largest code and a zero takes 0 (the simulation's nan_to_num).
        var tiny: List[Float32] = [_pow2(-120), 0.0, -_pow2(-125), _pow2(-126)]
        var sq = quantize_rows_int15(tiny, 1, 4)
        if sq.e[0] != Int32(-133):
            raise Error("a row of absmax 2^-120 took exponent " + String(sq.e[0]) + ", want -133")
        if Int(sq.q[0]) != 16383 or Int(sq.q[1]) != 0 or Int(sq.q[2]) != -16383 or Int(sq.q[3]) != 16383:
            raise Error("the row below 2^-114 coded to " + String(Int(sq.q[0])) + " " + String(Int(sq.q[1])) + " " + String(Int(sq.q[2])) + " " + String(Int(sq.q[3])))
    print("   ok " + String(rounded) + " of 6144 fixture values rounded, every error within a step")


def _const_codes(count: Int, value: Int) -> List[Int16]:
    var v = List[Int16]()
    for _ in range(count):
        v.append(Int16(value))
    return v^


def _zeros_i32(count: Int) -> List[Int32]:
    var v = List[Int32]()
    for _ in range(count):
        v.append(Int32(0))
    return v^


def _wrapped_equals_exact(code: Int, k: Int) -> Bool:
    """One cell of `k` products `code * code`: whether the three wrapping
    Int32 accumulators recombine to the exact Int64 sum."""
    var q = _const_codes(k, code)
    var pl = split_int15(q)
    var exact = int15_dot_cell(q, q, 0, 0, k)
    var pieces = int15_dot_cell_pieces(pl.hi, pl.lo, pl.hi, pl.lo, 0, 0, k)
    return exact == pieces


def check_int15_bounds_are_where_the_contract_says() raises:
    """Section W, THE BOUNDS, (b) to (f): a planted case AT each bound that
    must be exact and one ABOVE it that must not be. A bound with no case
    above it would be a number nobody tested."""
    # (b) one step of a unit: 32 products, each at the largest magnitude a
    # piece product has, 128 * 128. The sum of the step is 2^19.
    var step_q = _const_codes(32, -16383)
    var step_p = split_int15(step_q)
    var step_hh = Int32(0)
    for p in range(32):
        step_hh = step_hh + Int32(step_p.hi[p]) * Int32(step_p.hi[p])
    if step_hh != Int32(524288):
        raise Error("one unit step of the largest piece products sums to " + String(step_hh) + ", want 2^19")
    if not _wrapped_equals_exact(-16383, 32):
        raise Error("one unit step is not exact")
    # (c) one piece sum. The code -16383 is (hi, lo) = (-128, 1): every HH
    # term is 16384, the largest a piece product can be.
    if not _wrapped_equals_exact(-16383, INT15_PIECE_BOUND_K):
        raise Error("HH wrapped at k = " + String(INT15_PIECE_BOUND_K) + ", the bound of clause W-4 (c)")
    if _wrapped_equals_exact(-16383, INT15_PIECE_BOUND_K + 1):
        raise Error("HH did not wrap at k = " + String(INT15_PIECE_BOUND_K + 1) + ": the bound of clause W-4 (c) is not tight")
    # (d) HL + LH in one Int32. The code -16257 is (-128, 127): every HL
    # term and every LH term is -16256, the pair -32512.
    if not _wrapped_equals_exact(-16257, INT15_MID_BOUND_K):
        raise Error("HL + LH wrapped at k = " + String(INT15_MID_BOUND_K) + ", the bound of clause W-4 (d)")
    if _wrapped_equals_exact(-16257, INT15_MID_BOUND_K + 1):
        raise Error("HL + LH did not wrap at k = " + String(INT15_MID_BOUND_K + 1) + ": the bound of clause W-4 (d) is not tight")
    # The profile's bound is below both, and exact on both planted codes.
    var max_k = INT15_MAX_K
    var smallest_bound = INT15_MID_BOUND_K
    if INT15_PIECE_BOUND_K < smallest_bound:
        smallest_bound = INT15_PIECE_BOUND_K
    if max_k > smallest_bound:
        raise Error("INT15_MAX_K " + String(max_k) + " is above the smallest derived bound " + String(smallest_bound))
    if max_k * 2 <= smallest_bound:
        raise Error("INT15_MAX_K " + String(max_k) + " is not the largest power of two at or below " + String(smallest_bound))
    if not _wrapped_equals_exact(-16257, INT15_MAX_K) or not _wrapped_equals_exact(-16383, INT15_MAX_K) or not _wrapped_equals_exact(16383, INT15_MAX_K):
        raise Error("a planted code is not exact at INT15_MAX_K")
    # (e) the recombination at the ends of each accumulator.
    var hh_max = Int32(1073741824)  # 16384 * 65536
    var mid_min = Int32(-2130706432)  # -32512 * 65536
    var ll_max = Int32(1057030144)  # 16129 * 65536
    var r = int15_recombine(hh_max, mid_min, ll_max)
    var want = Int64(1073741824) * Int64(16384) + Int64(-2130706432) * Int64(128) + Int64(1057030144)
    if r != want:
        raise Error("recombination at the accumulators' ends: " + String(r) + ", want " + String(want))
    if int15_recombine(Int32(-1), Int32(-1), Int32(-1)) != Int64(-16384 - 128 - 1):
        raise Error("recombination of (-1, -1, -1) is " + String(int15_recombine(Int32(-1), Int32(-1), Int32(-1))))
    # The largest sum: every code 16383, k = 65536.
    var largest = Int64(16383) * Int64(16383) * Int64(INT15_MAX_K)
    if largest >= (Int64(1) << Int64(44)):
        raise Error("the largest admitted sum is not below 2^44")
    # A k above the bound is refused by name.
    var refused = False
    try:
        var one = _const_codes(1, 1)
        var e0 = _zeros_i32(1)
        _ = gemm_int15_oracle(one, e0, one, e0, 1, 1, INT15_MAX_K + 1)
    except e:
        refused = True
    if not refused:
        raise Error("the oracle accepted k = " + String(INT15_MAX_K + 1))
    comptime if IDENTICAL_BUILD:
        # (f) the scale exponent at both ends of the finite range and one
        # past each.
        var inf = bitcast[DType.float32](UInt32(0x7F800000))
        if _bits(dequant_int15_pinned(Int64(1), 127)) != UInt32(0x7F000000):
            raise Error("1 * 2^127 is " + _show(dequant_int15_pinned(Int64(1), 127)))
        if _bits(dequant_int15_pinned(Int64(1), 128)) != _bits(inf):
            raise Error("1 * 2^128 is not the infinity")
        if _bits(dequant_int15_pinned(Int64(-3), 230)) != _bits(-inf):
            raise Error("-3 * 2^230 is not the negative infinity")
        if _bits(dequant_int15_pinned(Int64(1), -126)) != UInt32(0x00800000):
            raise Error("1 * 2^-126 is " + _show(dequant_int15_pinned(Int64(1), -126)))
        if _bits(dequant_int15_pinned(Int64(8192), -127)) != UInt32(0):
            raise Error("8192 * 2^-127 is " + _show(dequant_int15_pinned(Int64(8192), -127)) + ", want +0.0: the contract has no subnormal scale")
        if _bits(dequant_int15_pinned(Int64(-8192), -278)) != UInt32(0x80000000):
            raise Error("-8192 * 2^-278 is " + _show(dequant_int15_pinned(Int64(-8192), -278)) + ", want -0.0")
        if not _is_nan(dequant_int15_pinned(Int64(0), 128)):
            raise Error("0 * 2^128 is not a NaN")
        if _bits(dequant_int15_pinned(Int64(0), -127)) != UInt32(0):
            raise Error("0 * 2^-127 is not +0.0")
        # The largest sum is 2^44 - 2^31 + 2^16 and rounds to 2^44 - 2^31:
        # finite under 2^84, the infinity under 2^85.
        if _bits(dequant_int15_pinned(largest, 0)) != _bits(Float32(17590038560768.0)):
            raise Error("the largest sum converts to " + _show(dequant_int15_pinned(largest, 0)) + ", want 2^44 - 2^31")
        if _bits(dequant_int15_pinned(largest, 84)) == _bits(inf):
            raise Error("the largest sum times 2^84 overflowed")
        if _bits(dequant_int15_pinned(largest, 85)) != _bits(inf):
            raise Error("the largest sum times 2^85 is not the infinity")
    print(
        "   ok bounds: piece sum exact at k = " + String(INT15_PIECE_BOUND_K)
        + " and wrapped above; HL + LH exact at k = " + String(INT15_MID_BOUND_K)
        + " and wrapped above; the profile stops at " + String(INT15_MAX_K)
    )


#: The shapes: `gemm_lowbit_check.mojo`'s OP_NT shapes (decode rows, odd
#: extents, k across 127, 128, 129, 256, 1000, one wide output) and its
#: ragged matrix-unit shapes (every k off the unit's k-tile of 32, m and n
#: off the 16-wide warp tile and the 32-wide block tile).
comptime SHAPE_COUNT = 15


def _shape(i: Int) -> Tuple[Int, Int, Int]:
    if i == 0:
        return (1, 32, 32)
    if i == 1:
        return (1, 64, 128)
    if i == 2:
        return (2, 96, 127)
    if i == 3:
        return (7, 33, 129)
    if i == 4:
        return (16, 64, 1000)
    if i == 5:
        return (129, 129, 256)
    if i == 6:
        return (1, 4096, 512)
    if i == 7:
        return (3, 5, 17)
    if i == 8:
        return (17, 33, 31)
    if i == 9:
        return (33, 17, 33)
    if i == 10:
        return (1, 47, 100)
    if i == 11:
        return (50, 70, 1000)
    if i == 12:
        return (13, 21, 4097)
    if i == 13:
        return (2, 16, 32)
    return (100, 3, 64)


def _tag(m: Int, n: Int, k: Int) -> String:
    return String(m) + "x" + String(n) + "x" + String(k)


def check_int15_pieces_oracle_matches_oracle() raises:
    """HOST GATE: the sum formed from the four piece products is the sum
    of code products, on every shape. No device is asked. The host value
    arm reaches `gemm_int15_oracle` and not the pieces spelling, so a host
    sabotage build fails HERE, on a box that has no GPU at all."""
    for s in range(SHAPE_COUNT):
        var sh = _shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var qa = quantize_rows_int15(_fill(m * k, 211 + s), m, k)
        var qb = quantize_rows_int15(_fill(n * k, 223 + s), n, k)
        var want = gemm_int15_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
        var got = gemm_int15_pieces_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
        _first_diff(got, want, "host pieces " + _tag(m, n, k))
    print("   ok the pieces spelling equals the definition on " + String(SHAPE_COUNT) + " shapes")


# ===========================================================================
# THE DEVICE
# ===========================================================================


def _upload[dt: DType](ctx: DeviceContext, h: List[Scalar[dt]]) raises -> DeviceBuffer[dt]:
    var n = len(h)
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[dt](n)
    var hb = ctx.enqueue_create_host_buffer[dt](n)
    ctx.synchronize()
    for i in range(len(h)):
        hb.unsafe_ptr().unsafe_store(i, h[i])
    ctx.enqueue_copy(dst_buf=d, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    _ = hb
    return d^


def _download[dt: DType](ctx: DeviceContext, mut d: DeviceBuffer[dt], count: Int) raises -> List[Scalar[dt]]:
    var hb = ctx.enqueue_create_host_buffer[dt](count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    var out = List[Scalar[dt]]()
    for i in range(count):
        out.append(hb.unsafe_ptr().unsafe_load(i))
    _ = hb
    return out^


def _poisoned(ctx: DeviceContext, count: Int) raises -> DeviceBuffer[DType.float32]:
    var h = List[Float32]()
    for _ in range(count):
        h.append(POISON)
    return _upload[DType.float32](ctx, h)


def _download_cells(ctx: DeviceContext, mut d: DeviceBuffer[DType.float32], count: Int, tag: String) raises -> List[Float32]:
    var out = _download[DType.float32](ctx, d, count)
    for i in range(count):
        if _bits(out[i]) == _bits(POISON):
            raise Error("POISON SURVIVED at cell " + String(i) + " of " + tag)
    return out^


def _first_diff(got: List[Float32], want: List[Float32], tag: String) raises:
    var bad = 0
    var first = -1
    for i in range(len(want)):
        if not _same(got[i], want[i]):
            bad += 1
            if first < 0:
                first = i
    if bad > 0:
        raise Error(
            tag + ": " + String(bad) + " of " + String(len(want))
            + " cells differ; first at " + String(first) + " got "
            + _show(got[first]) + " want " + _show(want[first])
        )


def _run_plan(
    ctx: DeviceContext, qa: Int15Rows, qb: Int15Rows, m: Int, n: Int, k: Int, plan: Int, tag: String
) raises -> List[Float32]:
    """One plan on host codes. The planes are split ON THE DEVICE, the way
    a caller's are, so the split's defect arm reaches every plan that reads
    planes; the device split has its own gate against the host's."""
    var dea = _upload[DType.int32](ctx, qa.e)
    var deb = _upload[DType.int32](ctx, qb.e)
    var dc = _poisoned(ctx, m * n)
    var dqa = _upload[DType.int16](ctx, qa.q)
    var dqb = _upload[DType.int16](ctx, qb.q)
    if plan == PLAN_FLAT or plan == PLAN_DISPATCH_CODES:
        var work = Int15Workspace(ctx)
        if plan == PLAN_FLAT:
            identical_gemm_int15_flat_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
        else:
            identical_gemm_int15_into(ctx, dc, dqa, dea, dqb, deb, work, m, n, k)
        ctx.synchronize()
        _ = work^
    else:
        var dah = ctx.enqueue_create_buffer[DType.int8](m * k)
        var dal = ctx.enqueue_create_buffer[DType.int8](m * k)
        var dbh = ctx.enqueue_create_buffer[DType.int8](n * k)
        var dbl = ctx.enqueue_create_buffer[DType.int8](n * k)
        split_int15_device(ctx, dah, dal, dqa, m * k)
        split_int15_device(ctx, dbh, dbl, dqb, n * k)
        if plan == PLAN_PIECES:
            identical_gemm_int15_pieces_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k)
        elif plan == PLAN_MMA:
            identical_gemm_int15_mma_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k)
        elif plan == PLAN_APPLE_WIDE:
            identical_gemm_int15_apple_with_geometry(
                ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k, INT15_APPLE_GEOMETRY_WIDE, INT15_APPLE_FORM_TWO
            )
        elif plan == PLAN_APPLE_ROW:
            identical_gemm_int15_apple_with_geometry(
                ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k, INT15_APPLE_GEOMETRY_ROW, INT15_APPLE_FORM_TWO
            )
        elif plan == PLAN_APPLE4_WIDE:
            identical_gemm_int15_apple_with_geometry(
                ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k, INT15_APPLE_GEOMETRY_WIDE, INT15_APPLE_FORM_FOUR
            )
        elif plan == PLAN_APPLE4_ROW:
            identical_gemm_int15_apple_with_geometry(
                ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k, INT15_APPLE_GEOMETRY_ROW, INT15_APPLE_FORM_FOUR
            )
        else:
            identical_gemm_int15_planes_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k)
        ctx.synchronize()
        _ = dah
        _ = dal
        _ = dbh
        _ = dbl
    var out = _download_cells(ctx, dc, m * n, tag + " " + _plan_name(plan))
    _ = dqa
    _ = dqb
    _ = dea
    _ = deb
    _ = dc
    return out^


def _every_plan_equals(
    ctx: DeviceContext, qa: Int15Rows, qb: Int15Rows, m: Int, n: Int, k: Int, name: String
) raises:
    """The oracle, then every plan this column has, each against the
    oracle; a digest line for each. Every plan runs before the first
    difference is raised, so a log names every plan a defect reaches."""
    var want = gemm_int15_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
    print("   DIGEST " + name + " oracle " + _digest(want))
    var failures = String("")
    for plan in range(PLAN_COUNT):
        if plan == PLAN_MMA:
            comptime if not HAS_UNIT:
                continue
        if plan >= PLAN_APPLE_WIDE:
            comptime if not IS_APPLE:
                continue
        var got = _run_plan(ctx, qa, qb, m, n, k, plan, name)
        print("   DIGEST " + name + " " + _plan_name(plan) + " " + _digest(got))
        try:
            _first_diff(got, want, name + " (" + _plan_name(plan) + " vs oracle)")
        except e:
            if failures.byte_length() > 0:
                failures += "; "
            failures += String(e)
    if failures.byte_length() > 0:
        raise Error(failures)


def int15_seam_probe_kernel(
    pinned: MutPointer[Float32, MutAnyOrigin],
    native: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Int64, MutAnyOrigin],
    n_in: Int32,
):
    """One planted sum per thread through the pinned seam, and through the
    backend's own int64 conversion beside it."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var x = v.unsafe_load(i)
    pinned.unsafe_store(i, i64_to_f32_pinned(x))
    native.unsafe_store(i, Float32(x))


def int15_recombine_probe_kernel(
    dst: MutPointer[Int64, MutAnyOrigin],
    hh: MutPointer[Int32, MutAnyOrigin],
    mid: MutPointer[Int32, MutAnyOrigin],
    ll: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """One recombination per thread, the Int64 itself stored: what the
    device's 64-bit shifts and additions give, before any float."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, int15_recombine(hh.unsafe_load(i), mid.unsafe_load(i), ll.unsafe_load(i)))


def check_int15_device_integers_match_host(ctx: DeviceContext) raises:
    """DEVICE GATE: Int64 arithmetic in a kernel gives the host's bits. The
    recombination's Int64 at the accumulators' ends and on a spread of
    triples; the pinned conversion at every planted sum; the backend's own
    conversion recorded beside it."""
    var hh = List[Int32]()
    var mid = List[Int32]()
    var ll = List[Int32]()
    var ends_hh: List[Int32] = [1073741824, -1065353216, 0, -1, 2147483647, -2147483647]
    var ends_mid: List[Int32] = [-2130706432, 2114060288, 0, -1, 2147483647, -2147483647]
    var ends_ll: List[Int32] = [1057030144, 0, 65536, -1, 2147483647, 1]
    for a in range(len(ends_hh)):
        for b in range(len(ends_mid)):
            for c in range(len(ends_ll)):
                hh.append(ends_hh[a])
                mid.append(ends_mid[b])
                ll.append(ends_ll[c])
    for i in range(4096):
        hh.append(Int32(Int(_hash64(i, 301) & UInt64(0x7FFFFFFF)) - 1073741824))
        mid.append(Int32(Int(_hash64(i, 302) & UInt64(0xFFFFFFFF)) - 2147483648))
        ll.append(Int32(Int(_hash64(i, 303) & UInt64(0x3FFFFFFF))))
    var count = len(hh)
    var dhh = _upload[DType.int32](ctx, hh)
    var dmid = _upload[DType.int32](ctx, mid)
    var dll = _upload[DType.int32](ctx, ll)
    var dout = ctx.enqueue_create_buffer[DType.int64](count)
    ctx.enqueue_function[int15_recombine_probe_kernel](
        dout.unsafe_ptr(),
        dhh.unsafe_ptr(),
        dmid.unsafe_ptr(),
        dll.unsafe_ptr(),
        Int32(count),
        grid_dim=((count + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()
    var got = _download[DType.int64](ctx, dout, count)
    for i in range(count):
        var want = int15_recombine(hh[i], mid[i], ll[i])
        if got[i] != want:
            raise Error(
                "device recombination of (" + String(hh[i]) + ", " + String(mid[i]) + ", " + String(ll[i])
                + ") is " + String(got[i]) + ", the host's is " + String(want)
            )
    _ = dhh
    _ = dmid
    _ = dll
    _ = dout
    var sums = _planted_sums()
    # and the sums a product can produce: k copies of one code product
    for i in range(2048):
        sums.append(Int64(Int(_hash64(i, 307) % UInt64(35180077252608)) - 17590038626304))
    var ns = len(sums)
    var dv = _upload[DType.int64](ctx, sums)
    var dp = _poisoned(ctx, ns)
    var dn = _poisoned(ctx, ns)
    ctx.enqueue_function[int15_seam_probe_kernel](
        dp.unsafe_ptr(),
        dn.unsafe_ptr(),
        dv.unsafe_ptr(),
        Int32(ns),
        grid_dim=((ns + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()
    var pinned = _download_cells(ctx, dp, ns, "seam probe")
    var native = _download[DType.float32](ctx, dn, ns)
    var native_differs = 0
    for i in range(ns):
        var want = _i64_to_f32_reference_bits(sums[i])
        if _bits(pinned[i]) != want:
            raise Error(
                "device i64_to_f32_pinned(" + String(sums[i]) + ") = 0x" + hex(_bits(pinned[i]))
                + ", the correctly rounded word is 0x" + hex(want)
            )
        if _bits(native[i]) != want:
            native_differs += 1
    _ = dv
    _ = dp
    _ = dn
    # Recorded, never trusted: a line of its own so a job's summary carries it.
    print(
        "   NATIVE " + column_name(TARGET_COLUMN)
        + ": the column's own int64 to float32 conversion differs from the correctly rounded word on "
        + String(native_differs) + " of " + String(ns) + " planted sums"
    )
    print("   ok " + String(count) + " recombinations and " + String(ns) + " conversions equal the host")


def check_int15_device_conversions_match_host(ctx: DeviceContext) raises:
    """DEVICE GATE: the quantizer's codes and exponents, the dequantized
    image and the split planes are the host's, on the fixture rows and on
    the planted rows (zeros, a NaN, an infinity, a row below `2^-114`, the
    clamp, every code of the range for the split)."""
    var rows = 40
    var cols = 257
    var x = _fill(rows * cols, 401)
    # planted rows over the first six
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var inf = bitcast[DType.float32](UInt32(0x7F800000))
    for c in range(cols):
        x[0 * cols + c] = Float32(0.0)
        x[1 * cols + c] = _pow2(-120) if c % 3 == 0 else (Float32(0.0) if c % 3 == 1 else -_pow2(-124))
        x[2 * cols + c] = Float32(16383.75) if c % 2 == 0 else Float32(-16382.5)
        x[3 * cols + c] = Float32(c) * Float32(0.5) - Float32(60.0)
    x[3 * cols] = Float32(8192.0)
    x[4 * cols + 5] = nan
    x[5 * cols + 7] = -inf
    # Row 6 has its largest value in the LAST chunk of the parallel
    # quantizer's absmax, a chunk of one value: a schedule that drops the
    # ragged tail takes another exponent here. Column 256 of every row is
    # that chunk; a column's own tail is planted the same way.
    if int15_quant_chunks(cols) != 2 or int15_quant_chunks(rows) != 1:
        raise Error("the fixture no longer has a ragged last chunk; move the planted row with the chunk size")
    x[6 * cols + cols - 1] = Float32(1000.0)
    var want = quantize_rows_int15(x, rows, cols)
    if Int(want.e[6]) != -4:
        raise Error("the planted row 6 took exponent " + String(want.e[6]) + ", want -4 (absmax 1000)")
    var dx = _upload[DType.float32](ctx, x)
    var dq = ctx.enqueue_create_buffer[DType.int16](rows * cols)
    var de = ctx.enqueue_create_buffer[DType.int32](rows)
    var dy = _poisoned(ctx, rows * cols)
    quantize_rows_int15_device(ctx, dq, de, dx, rows, cols)
    dequantize_rows_int15_device(ctx, dy, dq, de, rows, cols)
    ctx.synchronize()
    var got_q = _download[DType.int16](ctx, dq, rows * cols)
    var got_e = _download[DType.int32](ctx, de, rows)
    var got_y = _download_cells(ctx, dy, rows * cols, "dequantize")
    for r in range(rows):
        if got_e[r] != want.e[r]:
            raise Error("device exponent of row " + String(r) + " is " + String(got_e[r]) + ", the host's is " + String(want.e[r]))
    for i in range(rows * cols):
        if got_q[i] != want.q[i]:
            raise Error("device code at " + String(i) + " (row " + String(i // cols) + ") is " + String(Int(got_q[i])) + ", the host's is " + String(Int(want.q[i])))
    _first_diff(got_y, dequantize_rows_int15(want), "dequantized image")
    # THE TRANSPOSING QUANTIZER: the codes of the COLUMNS of the same
    # matrix, written as the rows of its transpose. The host's are the
    # row quantizer's on the transposed values, spelled here and not
    # taken from `quantize_cols_int15`, which is then held to it too.
    var xt = List[Float32]()
    for c in range(cols):
        for r in range(rows):
            xt.append(x[r * cols + c])
    var want_t = quantize_rows_int15(xt, cols, rows)
    var host_t = quantize_cols_int15(x, rows, cols)
    var dqt = ctx.enqueue_create_buffer[DType.int16](rows * cols)
    var det = ctx.enqueue_create_buffer[DType.int32](cols)
    quantize_cols_int15_device(ctx, dqt, det, dx, rows, cols)
    ctx.synchronize()
    var got_qt = _download[DType.int16](ctx, dqt, rows * cols)
    var got_et = _download[DType.int32](ctx, det, cols)
    for r in range(cols):
        if got_et[r] != want_t.e[r] or host_t.e[r] != want_t.e[r]:
            raise Error("transposing quantizer: exponent of column " + String(r) + " is " + String(got_et[r]) + " on the device and " + String(host_t.e[r]) + " on the host, want " + String(want_t.e[r]))
    for i in range(rows * cols):
        if got_qt[i] != want_t.q[i] or host_t.q[i] != want_t.q[i]:
            raise Error("transposing quantizer: code at " + String(i) + " is " + String(Int(got_qt[i])) + " on the device and " + String(Int(host_t.q[i])) + " on the host, want " + String(Int(want_t.q[i])))
    # THE PARALLEL SCHEDULE, to codes and straight to planes, the operand
    # stored either way: the same codes, exponents and planes.
    var quant = Int15QuantWorkspace(ctx)
    var want_p = split_int15(want.q)
    var want_tp = split_int15(want_t.q)
    comptime way_count = 2
    for way in range(way_count):
        var transposed = way == 1
        var r_out = cols if transposed else rows
        var c_out = rows if transposed else cols
        var dpq = ctx.enqueue_create_buffer[DType.int16](rows * cols)
        var dpe = ctx.enqueue_create_buffer[DType.int32](r_out)
        var dph = ctx.enqueue_create_buffer[DType.int8](rows * cols)
        var dpl = ctx.enqueue_create_buffer[DType.int8](rows * cols)
        var dpe2 = ctx.enqueue_create_buffer[DType.int32](r_out)
        # The transposed way reads `x` as the transpose of the matrix it
        # quantizes, so its codes are the transposing quantizer's.
        quantize_int15_parallel_device(ctx, dpq, dpe, dx, quant, r_out, c_out, transposed)
        quantize_planes_int15_parallel_device(ctx, dph, dpl, dpe2, dx, quant, r_out, c_out, transposed)
        ctx.synchronize()
        var got_pq = _download[DType.int16](ctx, dpq, rows * cols)
        var got_pe = _download[DType.int32](ctx, dpe, r_out)
        var got_ph = _download[DType.int8](ctx, dph, rows * cols)
        var got_pl = _download[DType.int8](ctx, dpl, rows * cols)
        var got_pe2 = _download[DType.int32](ctx, dpe2, r_out)
        var what = String("parallel quantizer (transposed)") if transposed else String("parallel quantizer")
        for r in range(r_out):
            var we = want_t.e[r] if transposed else want.e[r]
            if got_pe[r] != we or got_pe2[r] != we:
                raise Error(what + ": exponent of row " + String(r) + " is " + String(got_pe[r]) + " (codes) and " + String(got_pe2[r]) + " (planes), want " + String(we))
        for i in range(rows * cols):
            var wq = want_t.q[i] if transposed else want.q[i]
            var wh = want_tp.hi[i] if transposed else want_p.hi[i]
            var wl = want_tp.lo[i] if transposed else want_p.lo[i]
            if got_pq[i] != wq:
                raise Error(what + ": code at " + String(i) + " is " + String(Int(got_pq[i])) + ", want " + String(Int(wq)))
            if got_ph[i] != wh or got_pl[i] != wl:
                raise Error(what + ": planes at " + String(i) + " are (" + String(Int(got_ph[i])) + ", " + String(Int(got_pl[i])) + "), want (" + String(Int(wh)) + ", " + String(Int(wl)) + ")")
        if not transposed:
            # The planes read back as the codes they were split from, and
            # the image dequantized FROM THE PLANES is the image
            # dequantized from the codes.
            var joined = join_int15(got_ph, got_pl)
            for i in range(rows * cols):
                if joined[i] != want.q[i]:
                    raise Error("planes joined at " + String(i) + " give " + String(Int(joined[i])) + ", the code is " + String(Int(want.q[i])))
            var dyp = _poisoned(ctx, rows * cols)
            dequantize_planes_int15_device(ctx, dyp, dph, dpl, dpe2, rows, cols)
            ctx.synchronize()
            var got_yp = _download_cells(ctx, dyp, rows * cols, "dequantize from planes")
            _first_diff(got_yp, dequantize_rows_int15(want), "image dequantized from planes")
            _ = dyp
        _ = dpq
        _ = dpe
        _ = dph
        _ = dpl
        _ = dpe2
    _ = quant^
    _ = dqt
    _ = det
    _ = dx
    _ = dq
    _ = de
    _ = dy
    # the split, on every code of the range
    var all = List[Int16]()
    for ci in range(-INT15_CODE_MAX, INT15_CODE_MAX + 1):
        all.append(Int16(ci))
    var count = len(all)
    var planes = split_int15(all)
    var dall = _upload[DType.int16](ctx, all)
    var dhi = ctx.enqueue_create_buffer[DType.int8](count)
    var dlo = ctx.enqueue_create_buffer[DType.int8](count)
    split_int15_device(ctx, dhi, dlo, dall, count)
    ctx.synchronize()
    var got_hi = _download[DType.int8](ctx, dhi, count)
    var got_lo = _download[DType.int8](ctx, dlo, count)
    for i in range(count):
        if got_hi[i] != planes.hi[i] or got_lo[i] != planes.lo[i]:
            raise Error(
                "device split of the code " + String(Int(all[i])) + " is (" + String(Int(got_hi[i])) + ", "
                + String(Int(got_lo[i])) + "), the host's is (" + String(Int(planes.hi[i])) + ", " + String(Int(planes.lo[i])) + ")"
            )
    _ = dall
    _ = dhi
    _ = dlo
    print("   ok " + String(rows * cols) + " codes, " + String(rows) + " exponents, the image and " + String(count) + " splits equal the host")


def check_int15_device_matches_oracle(ctx: DeviceContext) raises:
    """GATE: from float32 operands, the device quantizer, the device split
    and the dispatched product are the host oracle's, bit for bit, on every
    shape. This is the path a caller takes."""
    for s in range(SHAPE_COUNT):
        var sh = _shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var ha = _fill(m * k, 83 + s)
        var hb = _fill(n * k, 97 + s)
        var qa = quantize_rows_int15(ha, m, k)
        var qb = quantize_rows_int15(hb, n, k)
        var want = gemm_int15_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
        var tag = "int15 " + _tag(m, n, k)
        var da = _upload[DType.float32](ctx, ha)
        var db = _upload[DType.float32](ctx, hb)
        var dqa = ctx.enqueue_create_buffer[DType.int16](m * k)
        var dea = ctx.enqueue_create_buffer[DType.int32](m)
        var dqb = ctx.enqueue_create_buffer[DType.int16](n * k)
        var deb = ctx.enqueue_create_buffer[DType.int32](n)
        var dc = _poisoned(ctx, m * n)
        var work = Int15Workspace(ctx)
        quantize_rows_int15_device(ctx, dqa, dea, da, m, k)
        quantize_rows_int15_device(ctx, dqb, deb, db, n, k)
        identical_gemm_int15_into(ctx, dc, dqa, dea, dqb, deb, work, m, n, k)
        ctx.synchronize()
        var got = _download_cells(ctx, dc, m * n, tag)
        print("   DIGEST path-" + _tag(m, n, k) + " oracle " + _digest(want))
        print("   DIGEST path-" + _tag(m, n, k) + " device " + _digest(got))
        _first_diff(got, want, tag)
        # the one-call form: the parallel quantizer and the dispatched plan
        var dc1 = _poisoned(ctx, m * n)
        identical_gemm_int15_from_f32(ctx, dc1, da, db, m, n, k)
        var got1 = _download_cells(ctx, dc1, m * n, tag + " from_f32")
        print("   DIGEST path-" + _tag(m, n, k) + " from-f32 " + _digest(got1))
        _first_diff(got1, want, tag + " (from_f32)")
        _ = dc1
        _ = da
        _ = db
        _ = dqa
        _ = dea
        _ = dqb
        _ = deb
        _ = dc
        _ = work^
        print("   ok " + tag + "  c[0]=" + _show(got[0]))


def _row_scaled(rows: Int, k: Int, salt: Int, modulus: Int, shift: Int, sign: Int) -> List[Float32]:
    """`_fill`, row `r` scaled by `2^(sign * ((r mod modulus) - shift))`: an
    operand whose rows carry different exponents."""
    var v = _fill(rows * k, salt)
    for r in range(rows):
        var s = _pow2(sign * ((r % modulus) - shift))
        for p in range(k):
            v[r * k + p] = v[r * k + p] * s
    return v^


def _distinct_exponents(e: List[Int32]) -> Int:
    var seen = List[Int32]()
    for i in range(len(e)):
        var found = False
        for j in range(len(seen)):
            if seen[j] == e[i]:
                found = True
                break
        if not found:
            seen.append(e[i])
    return len(seen)


def check_int15_row_scales(ctx: DeviceContext) raises:
    """GATE (clause W-7, the scale `2^(ea[i] + eb[j])`): every plan on
    operands whose rows carry DIFFERENT exponents, so a cell scaled by any
    exponent but its own row's and its own column's is seen. `_fill` gives
    nearly every row of an operand the same exponent (the top binade of
    eight is reached in almost every row), so no other gate can see an
    exponent read at the wrong index (run 6's finding on the tuned gate,
    2026-09-29). The fixture is held to at least 5 distinct exponents per
    operand before any plan runs, so the gate cannot pass by being blind.
    Two geometries: one below the Apple float unit's dispatch threshold and
    one above the unit plans' tile, ragged."""
    for g in range(2):
        var m = 37
        var n = 41
        var k = 300
        if g == 1:
            m = 70
            n = 67
            k = 1000
        var qa = quantize_rows_int15(_row_scaled(m, k, 311 + g, 7, 3, 1), m, k)
        var qb = quantize_rows_int15(_row_scaled(n, k, 313 + g, 5, 2, -1), n, k)
        var da = _distinct_exponents(qa.e)
        var db = _distinct_exponents(qb.e)
        if da < 5 or db < 5:
            raise Error(
                "the fixture is blind: " + String(da) + " distinct row exponents in A and "
                + String(db) + " in B, 5 each required"
            )
        _every_plan_equals(ctx, qa, qb, m, n, k, "row-scales-" + _tag(m, n, k))
    print("   ok every plan equals the oracle on operands whose rows carry 5 or more exponents")


def check_int15_plans_agree(ctx: DeviceContext) raises:
    """GATE (W-8): every plan this column has returns the oracle's bits on
    every shape, the ragged ones included, so every plan returns every
    other plan's."""
    for s in range(SHAPE_COUNT):
        var sh = _shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var qa = quantize_rows_int15(_fill(m * k, 131 + s), m, k)
        var qb = quantize_rows_int15(_fill(n * k, 149 + s), n, k)
        _every_plan_equals(ctx, qa, qb, m, n, k, "plans-" + _tag(m, n, k))
    comptime if HAS_UNIT:
        print("   ok flat, pieces, mma and both dispatchers equal the oracle on " + String(SHAPE_COUNT) + " shapes")
    else:
        print(
            "   ok flat, pieces and both dispatchers equal the oracle on " + String(SHAPE_COUNT)
            + " shapes; THE MMA PLAN DID NOT RUN: column " + column_name(TARGET_COLUMN) + " has no int8 matrix unit"
        )


def _planted_rows(rows: Int, k: Int, kind: Int, e: Int) -> Int15Rows:
    """Codes written directly, never through the quantizer.
      0  every code +16383
      1  every code -16383            (hi, lo) = (-128, 1): HH at its largest
      2  every code -16257            (hi, lo) = (-128, 127): HL + LH at its most negative
      3  +16383 on the first half of k, -16383 on the second
      4  a row of zeros first, then +16383
      5  codes walking the whole range, a different phase per row
      6  every code +16257            (hi, lo) = (127, 1)
      7  a row of zeros first, then the code 1 at p = 0 and zeros after
      8  the code -1 at p = 0 and zeros after
      9  +16383, but +16382 where p is a multiple of 16: against kind 0
         every product by the high piece is ODD but one in sixteen, so the
         sum of any sixteen consecutive steps is an odd integer above 2^24,
         which no float32 holds, while the sum of any eight is below it"""
    var q = List[Int16]()
    for r in range(rows):
        for p in range(k):
            var c: Int
            if kind == 0:
                c = 16383
            elif kind == 1:
                c = -16383
            elif kind == 2:
                c = -16257
            elif kind == 3:
                c = 16383 if p < k // 2 else -16383
            elif kind == 4:
                c = 0 if r == 0 else 16383
            elif kind == 5:
                c = ((p * 2731 + r * 977) % 32767) - 16383
            elif kind == 6:
                c = 16257
            elif kind == 7:
                c = 1 if (p == 0 and r > 0) else 0
            elif kind == 9:
                c = 16382 if p % 16 == 0 else 16383
            else:
                c = -1 if p == 0 else 0
            q.append(Int16(c))
    var ex = List[Int32]()
    for _ in range(rows):
        ex.append(Int32(e))
    return Int15Rows(q^, ex^, rows, k)


def _expect_cell(got: Float32, want: Float32, what: String) raises:
    if not _same(got, want):
        raise Error(what + ": the oracle gives " + _show(got) + ", the closed form is " + _show(want))


def check_int15_planted_worst_cases(ctx: DeviceContext) raises:
    """GATE: the cases a random fixture does not reach, each against the
    oracle on every plan, and the oracle itself against a closed form so
    the comparison is not the oracle against its own mistake.

    Every code at +16383 or -16383; the high piece at -128 on both sides;
    `HL + LH` at its most negative; cancelling halves; a row of zeros; the
    codes walking the whole range; each at ragged shapes and at the largest
    admitted `k`; and the scale exponent at both ends."""
    var cases = 0
    var failures = String("")
    comptime geom_count = 4
    for geom in range(geom_count):
        var m = 3
        var n = 5
        var k = 17
        if geom == 1:
            m = 17
            n = 33
            k = 1000
        elif geom == 2:
            m = 1
            n = 40
            k = 4097
        elif geom == 3:
            m = 2
            n = 3
            k = INT15_MAX_K
        comptime left_count = 7
        for ka in range(left_count):
            for kb in range(left_count):
                # The largest k runs the pairs that load an accumulator.
                if geom == 3 and not ((ka < 3 and kb < 3) or (ka == 3 and kb == 0)):
                    continue
                # The middle shapes skip the mixed pairs of the walk.
                if (geom == 1 or geom == 2) and (ka == 5) != (kb == 5):
                    continue
                var qa = _planted_rows(m, k, ka, 0)
                var qb = _planted_rows(n, k, kb, 0)
                var name = "planted-" + String(ka) + "." + String(kb) + "-" + _tag(m, n, k)
                # the closed forms of the constant pairs
                if ka < 3 and kb < 3:
                    var ca = 16383 if ka == 0 else (-16383 if ka == 1 else -16257)
                    var cb = 16383 if kb == 0 else (-16383 if kb == 1 else -16257)
                    var closed = i64_to_f32_pinned(Int64(ca) * Int64(cb) * Int64(k))
                    var o = gemm_int15_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
                    comptime if not GEMM_ORACLE_HOST_SABOTAGE:
                        _expect_cell(o[m * n - 1], closed, name)
                if ka == 3 and kb == 0 and k % 2 == 0:
                    var o = gemm_int15_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
                    comptime if not GEMM_ORACLE_HOST_SABOTAGE:
                        _expect_cell(o[0], Float32(0.0), name + " (cancelling halves)")
                try:
                    _every_plan_equals(ctx, qa, qb, m, n, k, name)
                except e:
                    if failures.byte_length() > 0:
                        failures += "; "
                    failures += String(e)
                cases += 1
        # The float unit's own worst case (clause W-12): an odd sum above
        # 2^24 across two steps of the unit, none inside one.
        var qa9 = _planted_rows(m, k, 9, 0)
        var qb9 = _planted_rows(n, k, 0, 0)
        try:
            _every_plan_equals(ctx, qa9, qb9, m, n, k, "planted-9.0-" + _tag(m, n, k))
        except e:
            if failures.byte_length() > 0:
                failures += "; "
            failures += String(e)
        cases += 1
    # the scale exponent at both ends (clause W-7), on every plan
    comptime scale_count = 6
    for s in range(scale_count):
        var ea = 115
        var eb = 12
        if s == 1:
            ea = 115
            eb = 13
        elif s == 2:
            ea = -139
            eb = 13
        elif s == 3:
            ea = -139
            eb = 12
        elif s == 4:
            ea = 115
            eb = 115
        elif s == 5:
            ea = -139
            eb = -139
        # Every sum is 0 (row 0 of A is zeros) or -1, so the cell is the
        # scale itself: -2^127, the infinity, -2^-126, -0.0; and under an
        # infinite scale the cells of row 0 are 0 * inf, a NaN.
        var qa = _planted_rows(3, 33, 7, ea)
        var qb = _planted_rows(5, 33, 8, eb)
        var name = "scale-" + String(ea + eb)
        var o = gemm_int15_oracle(qa.q, qa.e, qb.q, qb.e, 3, 5, 33)
        comptime if IDENTICAL_BUILD and not GEMM_ORACLE_HOST_SABOTAGE:
            var want_cell = -_pow2(127)
            if s == 1 or s == 4:
                want_cell = -bitcast[DType.float32](UInt32(0x7F800000))
            elif s == 2:
                want_cell = -_pow2(-126)
            elif s == 3 or s == 5:
                want_cell = Float32(-0.0)
            _expect_cell(o[14], want_cell, name)
            if s == 1 or s == 4:
                if not _is_nan(o[0]):
                    raise Error(name + ": a sum of zero under an infinite scale is " + _show(o[0]) + ", want a NaN")
        try:
            _every_plan_equals(ctx, qa, qb, 3, 5, 33, name)
        except e:
            if failures.byte_length() > 0:
                failures += "; "
            failures += String(e)
        cases += 1
    # the largest sum under the largest scale that keeps it finite, and one above
    for top in range(2):
        var qa = _planted_rows(2, INT15_MAX_K, 0, 84 + top)
        var qb = _planted_rows(3, INT15_MAX_K, 0, 0)
        try:
            _every_plan_equals(ctx, qa, qb, 2, 3, INT15_MAX_K, "largest-sum-scale-" + String(84 + top))
        except e:
            if failures.byte_length() > 0:
                failures += "; "
            failures += String(e)
        cases += 1
    if failures.byte_length() > 0:
        raise Error(failures)
    print("   ok " + String(cases) + " planted cases, every plan equal to the oracle")


def check_int15_large_product_is_written_whole(ctx: DeviceContext) raises:
    """GATE (DEVIATION 2979, clause W-14): THE ROW THAT FAILED on the M2
    Pro, 512 x 4096 x 14336, on every plan this column has.

    A launch that holds an Apple GPU for seconds is aborted by the system
    and leaves its output partly written, and the wait returns as if it had
    finished. No shape of the other gates is large enough to reach that.
    Here every plan's output is poisoned first and read back whole, so a
    cell no launch wrote is seen; every plan's digest must be one digest;
    and the host oracle is held to it on cells sampled across the whole
    output, the first and the last included (the whole product is 3e10
    steps on the host, the sample is a million).

    The operands are the walk over the whole range of codes, written
    directly. The planes are split on the device."""
    var m = 512
    var n = 4096
    var k = 14336
    var qa = _planted_rows(m, k, 5, -3)
    var qb = _planted_rows(n, k, 5, 2)
    var dea = _upload[DType.int32](ctx, qa.e)
    var deb = _upload[DType.int32](ctx, qb.e)
    var dqa = _upload[DType.int16](ctx, qa.q)
    var dqb = _upload[DType.int16](ctx, qb.q)
    var dah = ctx.enqueue_create_buffer[DType.int8](m * k)
    var dal = ctx.enqueue_create_buffer[DType.int8](m * k)
    var dbh = ctx.enqueue_create_buffer[DType.int8](n * k)
    var dbl = ctx.enqueue_create_buffer[DType.int8](n * k)
    var work = Int15Workspace(ctx)
    split_int15_device(ctx, dah, dal, dqa, m * k)
    split_int15_device(ctx, dbh, dbl, dqb, n * k)
    ctx.synchronize()
    var reference = String("")
    var failures = String("")
    var ran = 0
    for plan in range(PLAN_COUNT):
        if plan == PLAN_MMA:
            comptime if not HAS_UNIT:
                continue
        if plan >= PLAN_APPLE_WIDE:
            comptime if not IS_APPLE:
                continue
        # The ROW geometry owns 8 rows of output a block; at 512 rows it is
        # the WIDE plan's work in eight times the blocks. Once is enough.
        if plan == PLAN_APPLE_ROW or plan == PLAN_APPLE4_ROW:
            continue
        var dc = _poisoned(ctx, m * n)
        if plan == PLAN_FLAT:
            identical_gemm_int15_flat_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
        elif plan == PLAN_DISPATCH_CODES:
            identical_gemm_int15_into(ctx, dc, dqa, dea, dqb, deb, work, m, n, k)
        elif plan == PLAN_PIECES:
            identical_gemm_int15_pieces_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k)
        elif plan == PLAN_MMA:
            identical_gemm_int15_mma_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k)
        elif plan == PLAN_APPLE_WIDE:
            identical_gemm_int15_apple_with_geometry(
                ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k, INT15_APPLE_GEOMETRY_WIDE, INT15_APPLE_FORM_TWO
            )
        elif plan == PLAN_APPLE4_WIDE:
            identical_gemm_int15_apple_with_geometry(
                ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k, INT15_APPLE_GEOMETRY_WIDE, INT15_APPLE_FORM_FOUR
            )
        else:
            identical_gemm_int15_planes_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k)
        ctx.synchronize()
        var name = "large-" + _tag(m, n, k)
        try:
            var got = _download_cells(ctx, dc, m * n, name + " " + _plan_name(plan))
            var d = _digest(got)
            print("   DIGEST " + name + " " + _plan_name(plan) + " " + d)
            if reference.byte_length() == 0:
                reference = d
                # the host oracle on sampled cells, against the first plan
                var samples = 64
                for t in range(samples):
                    var cell = (t * (m * n - 1)) // (samples - 1)
                    var i = cell // n
                    var j = cell - i * n
                    var want = gemm_int15_oracle_cell(qa.q, qa.e, qb.q, qb.e, i, j, k)
                    if not _same(got[cell], want):
                        raise Error(
                            name + " (" + _plan_name(plan) + "): cell " + String(cell) + " is "
                            + _show(got[cell]) + ", the host oracle has " + _show(want)
                        )
            elif d != reference:
                raise Error(name + ": " + _plan_name(plan) + " printed " + d + ", the first plan " + reference)
        except e:
            if failures.byte_length() > 0:
                failures += "; "
            failures += String(e)
        ran += 1
        _ = dc
    _ = dea
    _ = deb
    _ = dqa
    _ = dqb
    _ = dah
    _ = dal
    _ = dbh
    _ = dbl
    _ = work^
    if failures.byte_length() > 0:
        raise Error(failures)
    print("   ok " + String(ran) + " plans wrote every cell of " + _tag(m, n, k) + " and printed one digest; 64 sampled cells equal the host oracle")


def check_int15_is_batch_invariant(ctx: DeviceContext) raises:
    """GATE: a row's codes, exponent and products do not depend on the other
    rows in the call."""
    var m = 5
    var n = 40
    var k = 300
    var ha = _fill(m * k, 101)
    var hb = _fill(n * k, 103)
    var qb = quantize_rows_int15(hb, n, k)
    var dqb = _upload[DType.int16](ctx, qb.q)
    var deb = _upload[DType.int32](ctx, qb.e)
    var da = _upload[DType.float32](ctx, ha)
    var dqa = ctx.enqueue_create_buffer[DType.int16](m * k)
    var dea = ctx.enqueue_create_buffer[DType.int32](m)
    var dc = _poisoned(ctx, m * n)
    var work = Int15Workspace(ctx)
    quantize_rows_int15_device(ctx, dqa, dea, da, m, k)
    identical_gemm_int15_into(ctx, dc, dqa, dea, dqb, deb, work, m, n, k)
    ctx.synchronize()
    var batch = _download_cells(ctx, dc, m * n, "int15 batch")
    for i in range(m):
        var row = List[Float32]()
        for p in range(k):
            row.append(ha[i * k + p])
        var dr = _upload[DType.float32](ctx, row)
        var dqr = ctx.enqueue_create_buffer[DType.int16](k)
        var der = ctx.enqueue_create_buffer[DType.int32](1)
        var dcr = _poisoned(ctx, n)
        quantize_rows_int15_device(ctx, dqr, der, dr, 1, k)
        identical_gemm_int15_into(ctx, dcr, dqr, der, dqb, deb, work, 1, n, k)
        ctx.synchronize()
        var one = _download_cells(ctx, dcr, n, "int15 row")
        for j in range(n):
            if _bits(one[j]) != _bits(batch[i * n + j]):
                raise Error("int15 row " + String(i) + " col " + String(j) + " differs between the batch and the single-row call")
        _ = dr
        _ = dqr
        _ = der
        _ = dcr
    _ = da
    _ = dqa
    _ = dea
    _ = dqb
    _ = deb
    _ = dc
    _ = work^
    print("   ok " + String(m) + " int15 rows agree with their single-row calls")


def check_int15_unit_loads_state_their_alignment(ctx: DeviceContext) raises:
    """GATE (DEVIATION 2975), unit columns only: the fragment loads of the
    matrix-unit plan state an alignment only when the launch finds the
    bases of the planes aligned, so a box whose allocator returned
    unaligned bases would run every gate through the OLD loads and the
    stated ones would be reached by nothing. This gate allocates planes the
    way every caller does, at sizes from one code to the training rows',
    and requires every base to be a multiple of 8. Its pass is what makes
    the other gates' passes a statement about the stated loads."""
    var sizes: List[Int] = [1, 17, 4096, 65536, 2097152]
    var seen = 0
    for i in range(len(sizes)):
        var a = ctx.enqueue_create_buffer[DType.int8](sizes[i])
        var b = ctx.enqueue_create_buffer[DType.int8](sizes[i])
        ctx.synchronize()
        if not mma_operands_aligned(Int(a.unsafe_ptr()), Int(b.unsafe_ptr())):
            raise Error(
                "two int8 buffers of " + String(sizes[i]) + " codes have the bases "
                + String(Int(a.unsafe_ptr())) + " and " + String(Int(b.unsafe_ptr()))
                + ", not both multiples of 8: the stated-alignment loads are not reached on this box"
            )
        seen += 2
        _ = a
        _ = b
    var work = Int15Workspace(ctx)
    work.ensure(ctx, 12345, 54321)
    ctx.synchronize()
    if not mma_operands_aligned(Int(work.ah.unsafe_ptr()), Int(work.al.unsafe_ptr())) or not mma_operands_aligned(Int(work.bh.unsafe_ptr()), Int(work.bl.unsafe_ptr())):
        raise Error("a workspace plane has a base that is not a multiple of 8")
    seen += 4
    _ = work^
    comptime if INT8_MMA_UNSTATED_LOADS:
        print("   ok " + String(seen) + " bases are multiples of 8; THIS BUILD STATES NO ALIGNMENT (MOJOLEARN_INT8_MMA_UNSTATED_LOADS)")
    else:
        print("   ok " + String(seen) + " bases are multiples of 8; the unit plan's fragment loads state their alignment")


def check_int15_device_refuses_above_max_k(ctx: DeviceContext) raises:
    """GATE: every launch refuses `k = INT15_MAX_K + 1` by name, before it
    reads a buffer. The operands are as long as the shape the calls pass
    (`1 x k` each), so a launch that failed to refuse reads inside its
    buffers and the gate fails on the count, never on memory outside an
    allocation (the MI325X finding, 2026-09-29, job 1790657862351)."""
    var k = INT15_MAX_K + 1
    var one16 = List[Int16](length=k, fill=Int16(1))
    var one8 = List[Int8](length=k, fill=Int8(1))
    var e0: List[Int32] = [0]
    var dqa = _upload[DType.int16](ctx, one16)
    var dqb = _upload[DType.int16](ctx, one16)
    var dah = _upload[DType.int8](ctx, one8)
    var dal = _upload[DType.int8](ctx, one8)
    var dbh = _upload[DType.int8](ctx, one8)
    var dbl = _upload[DType.int8](ctx, one8)
    var dea = _upload[DType.int32](ctx, e0)
    var deb = _upload[DType.int32](ctx, e0)
    var dc = _poisoned(ctx, 1)
    var work = Int15Workspace(ctx)
    var refused = 0
    try:
        identical_gemm_int15_flat_into(ctx, dc, dqa, dea, dqb, deb, 1, 1, k)
    except e:
        refused += 1
    try:
        identical_gemm_int15_pieces_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, 1, 1, k)
    except e:
        refused += 1
    try:
        identical_gemm_int15_planes_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, 1, 1, k)
    except e:
        refused += 1
    try:
        identical_gemm_int15_into(ctx, dc, dqa, dea, dqb, deb, work, 1, 1, k)
    except e:
        refused += 1
    ctx.synchronize()
    _ = dqa
    _ = dqb
    _ = dah
    _ = dal
    _ = dbh
    _ = dbl
    _ = dea
    _ = deb
    _ = dc
    _ = work^
    if refused != 4:
        raise Error("only " + String(refused) + " of 4 launches refused k = " + String(k))
    print("   ok 4 launches refuse k = " + String(k))


def _gate(name: String, mut ran: Int, mut failed: Int, e: String):
    ran += 1
    if e.byte_length() > 0:
        failed += 1
        print("!! GATE FAILED: " + name)
        print("   " + e)
    else:
        print("ok " + name)


def main() raises:
    print(
        "== gemm/checks/gemm_int15_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + int15_sabotage_name()
        + "  apple unit sabotage: " + int15_apple_sabotage_name()
        + "  host sabotage: " + String(GEMM_ORACLE_HOST_SABOTAGE) + " =="
    )
    print("   profile: mojolearn.identical.gemm.int15i64.v1")
    print("   contract: gemm/IDENTICAL_LOWBIT_CONTRACT.md (THE FIFTEEN-BIT PROFILE)")
    print("   column: " + column_name(TARGET_COLUMN) + "  int15 dispatch: " + int15_plan_dispatch_name())
    var ran = 0
    var failed = 0
    try:
        check_int15_pieces_cover_every_code()
        _gate(String("check_int15_pieces_cover_every_code"), ran, failed, String(""))
    except e:
        _gate(String("check_int15_pieces_cover_every_code"), ran, failed, String(e))
    try:
        check_int15_sum_to_float_is_correctly_rounded()
        _gate(String("check_int15_sum_to_float_is_correctly_rounded"), ran, failed, String(""))
    except e:
        _gate(String("check_int15_sum_to_float_is_correctly_rounded"), ran, failed, String(e))
    try:
        check_int15_quantizer_is_the_rule()
        _gate(String("check_int15_quantizer_is_the_rule"), ran, failed, String(""))
    except e:
        _gate(String("check_int15_quantizer_is_the_rule"), ran, failed, String(e))
    try:
        check_int15_bounds_are_where_the_contract_says()
        _gate(String("check_int15_bounds_are_where_the_contract_says"), ran, failed, String(""))
    except e:
        _gate(String("check_int15_bounds_are_where_the_contract_says"), ran, failed, String(e))
    try:
        check_int15_pieces_oracle_matches_oracle()
        _gate(String("check_int15_pieces_oracle_matches_oracle"), ran, failed, String(""))
    except e:
        _gate(String("check_int15_pieces_oracle_matches_oracle"), ran, failed, String(e))
    comptime if not has_accelerator():
        print("   no accelerator: the device gates did not run")
    else:
        var ctx = DeviceContext()
        try:
            check_int15_device_integers_match_host(ctx)
            _gate(String("check_int15_device_integers_match_host"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_device_integers_match_host"), ran, failed, String(e))
        try:
            check_int15_device_conversions_match_host(ctx)
            _gate(String("check_int15_device_conversions_match_host"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_device_conversions_match_host"), ran, failed, String(e))
        try:
            check_int15_device_matches_oracle(ctx)
            _gate(String("check_int15_device_matches_oracle"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_device_matches_oracle"), ran, failed, String(e))
        try:
            check_int15_plans_agree(ctx)
            _gate(String("check_int15_plans_agree"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_plans_agree"), ran, failed, String(e))
        try:
            check_int15_row_scales(ctx)
            _gate(String("check_int15_row_scales"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_row_scales"), ran, failed, String(e))
        try:
            check_int15_planted_worst_cases(ctx)
            _gate(String("check_int15_planted_worst_cases"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_planted_worst_cases"), ran, failed, String(e))
        try:
            check_int15_large_product_is_written_whole(ctx)
            _gate(String("check_int15_large_product_is_written_whole"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_large_product_is_written_whole"), ran, failed, String(e))
        try:
            check_int15_is_batch_invariant(ctx)
            _gate(String("check_int15_is_batch_invariant"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_is_batch_invariant"), ran, failed, String(e))
        comptime if HAS_UNIT:
            try:
                check_int15_unit_loads_state_their_alignment(ctx)
                _gate(String("check_int15_unit_loads_state_their_alignment"), ran, failed, String(""))
            except e:
                _gate(String("check_int15_unit_loads_state_their_alignment"), ran, failed, String(e))
        try:
            check_int15_device_refuses_above_max_k(ctx)
            _gate(String("check_int15_device_refuses_above_max_k"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_device_refuses_above_max_k"), ran, failed, String(e))
        _ = ctx^
    print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
    if failed > 0:
        raise Error(String(failed) + " gate(s) failed")
