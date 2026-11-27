# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The one float64 -> float32 admission rule (Andrew, 2026-11-27):
round-to-nearest-even, NaN stays NaN, an infinity stays that infinity, and a
finite value outside float32's range is refused naming its column (the
smallest such column; the view's row when the cast is transposed), never
turned into an infinity. core/admitted_input.mojo `admitted_cast_f32`,
`admitted_cast_f32_words`; core/admitted_input_nd.mojo `admitted_cast_nd_f32`."""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases, require
from core.cast_in import CAST_KIND_FLOAT
from core.admitted_input import (
    admitted_cast_f32, admitted_cast_f32_words, ADMITTED_ALL_FINITE, ADMITTED_NONFINITE,
)
from core.admitted_input_nd import admitted_cast_nd_f32

comptime ADMITTED_F64_NARROW_FIXTURES = 9
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]

comptime _TIE = UInt64(0x3FF0000010000000)       # 1 + 2^-24: ties to even 1.0
comptime _TENTH = UInt64(0x3FB999999999999A)     # 0.1 -> 0x3DCCCCCD
comptime _KEPT = UInt64(0x47EFFFFFEFFFFFFF)      # below max + half ulp -> 0x7F7FFFFF
comptime _HALF_UP = UInt64(0x47EFFFFFF0000000)   # max + half ulp: ties to even -> infinity
comptime _NEG_BIG = UInt64(0xC7EFFFFFF0000000)   # its negative
comptime _QNAN = UInt64(0x7FF8000000000000)
comptime _INF = UInt64(0x7FF0000000000000)
comptime _ONE = UInt64(0x3FF0000000000000)


def _status(words: List[UInt64], rows: Int, cols: Int, mut dst: List[Float32], transposed: Bool = False) -> Int:
    """The word status of the C-order rows x cols float64 matrix `words`
    (`transposed`: the cols x rows view of it, as an F-order admission reads)."""
    var addr = Int(words.unsafe_ptr())
    var out = _F32P(unsafe_from_address=Int(dst.unsafe_ptr()))
    if transposed:
        return admitted_cast_f32_words(addr, 8, cols * 8, cols, rows, 8, CAST_KIND_FLOAT, out, True)
    return admitted_cast_f32_words(addr, cols * 8, 8, rows, cols, 8, CAST_KIND_FLOAT, out)


def _bits(dst: List[Float32], i: Int) -> UInt32:
    return bitcast[DType.uint32](dst[i])


def fixture(unit: Int) raises:
    if unit == 0:
        # round-to-nearest-even, once
        var words: List[UInt64] = [_TIE, _TENTH, _ONE]
        var dst = List[Float32](length=3, fill=0)
        require(_status(words, 1, 3, dst) == ADMITTED_ALL_FINITE, "finite input is finite")
        require(_bits(dst, 0) == UInt32(0x3F800000) and _bits(dst, 1) == UInt32(0x3DCCCCCD)
                and _bits(dst, 2) == UInt32(0x3F800000), "round-to-nearest-even words")
    elif unit == 1:
        # a value that rounds to float32's maximum is kept
        var words: List[UInt64] = [_KEPT, _ONE]
        var dst = List[Float32](length=2, fill=0)
        require(_status(words, 2, 1, dst) == ADMITTED_ALL_FINITE, "a value rounding to the maximum is in range")
        require(_bits(dst, 0) == UInt32(0x7F7FFFFF), "rounds to float32's maximum")
    elif unit == 2 or unit == 3:
        # 3 x 4: out-of-range values at (1, 2) and (2, 3); the smallest column
        # is 2, the smallest row 1 (an F-order view names the caller's column,
        # which is the view's row)
        var words = List[UInt64](length=12, fill=_ONE)
        words[1 * 4 + 2] = _HALF_UP
        words[2 * 4 + 3] = _NEG_BIG
        var dst = List[Float32](length=12, fill=0)
        if unit == 2:
            require(_status(words, 3, 4, dst) == 2, "smallest out-of-range column")
            var refused = False
            try:
                _ = admitted_cast_f32(Int(words.unsafe_ptr()), 32, 8, 3, 4, 8, CAST_KIND_FLOAT,
                                      _F32P(unsafe_from_address=Int(dst.unsafe_ptr())))
            except error:
                refused = String(error).find("ValueError: column 2 holds a float64 value") == 0
            require(refused, "the raising form refuses naming column 2")
            _ = words^
            _ = dst^
        else:
            # transposed: the caller's matrix is the 4 x 3 transpose; its
            # out-of-range columns are 1 and 2, the view's rows
            require(_status(words, 3, 4, dst, True) == 1, "transposed view names the caller's column")
    elif unit == 4:
        var words: List[UInt64] = [_QNAN, _ONE]
        var dst = List[Float32](length=2, fill=0)
        require(_status(words, 1, 2, dst) == ADMITTED_NONFINITE, "a NaN is non-finite, not out of range")
        require(_bits(dst, 0) == UInt32(0x7FC00000), "NaN stays NaN")
    elif unit == 5:
        var words: List[UInt64] = [_ONE, _INF | (UInt64(1) << 63)]
        var dst = List[Float32](length=2, fill=0)
        require(_status(words, 1, 2, dst) == ADMITTED_NONFINITE, "an infinity is non-finite, not out of range")
        require(_bits(dst, 1) == UInt32(0xFF800000), "an infinity stays that infinity")
    elif unit == 6:
        # rank 3: the column is the last axis
        var words = List[UInt64](length=12, fill=_ONE)
        words[1 * 6 + 0 * 3 + 1] = _HALF_UP
        var dst = List[Float32](length=12, fill=0)
        var dims: List[Int] = [2, 2, 3]
        var steps: List[Int] = [48, 24, 8]
        var refused = False
        try:
            _ = admitted_cast_nd_f32(Int(words.unsafe_ptr()), dims, steps, 8, CAST_KIND_FLOAT,
                                     _F32P(unsafe_from_address=Int(dst.unsafe_ptr())))
        except error:
            refused = String(error).find("ValueError: column 1 holds a float64 value") == 0
        require(refused, "the N-d cast refuses naming its last-axis column")
        _ = words^
        _ = dst^
    elif unit == 7:
        # float32 words cast as they are (no rounding, never refused)
        var words: List[Float32] = [1.5, -0.25, 3.0e38]
        var dst = List[Float32](length=3, fill=0)
        var status = admitted_cast_f32_words(Int(words.unsafe_ptr()), 12, 4, 1, 3, 4, CAST_KIND_FLOAT,
                                             _F32P(unsafe_from_address=Int(dst.unsafe_ptr())))
        require(status == ADMITTED_ALL_FINITE and dst[0] == 1.5 and dst[1] == -0.25 and dst[2] == words[2],
                "float32 words are copied unchanged")
        _ = words^
    else:
        # many rows split over tasks: the verdict is the minimum over tasks
        var rows = 6000
        var words = List[UInt64](length=rows * 3, fill=_TENTH)
        words[4000 * 3 + 0] = _HALF_UP
        words[10 * 3 + 2] = _NEG_BIG
        var dst = List[Float32](length=rows * 3, fill=0)
        require(_status(words, rows, 3, dst) == 0, "the smallest column over every task")


def test_admitted_f64_narrow() raises:
    check_cases("admitted_f64_narrow", ADMITTED_F64_NARROW_FIXTURES, fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
