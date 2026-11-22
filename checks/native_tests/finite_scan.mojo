# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
# Copyright 2026 Andrew Hendel. Part of Mojolearn.
"""Direct native finiteness checks migrated from the Python position sweep.

The original 4096 x 64 shape, six word positions, NaN/+Inf/-Inf and four
layouts remain: contiguous f32, Fortran f32, contiguous f64 and row-strided
f32. Each of the four layouts also has an all-finite fixture. Expected
verdicts come from the fixture labels, independently of the scan algorithm.

Independent fixtures use host_parallelize with task-local storage. The
large native scans retain their own production work splitting; no thread
limit, scan override, Python object, Python import or estimator is involved.
Each fixture needs at most about 2 MiB of input storage.

The general view scans (`host_all_finite_view`, `host_all_finite_nd`) add six
layouts on the same grid: C order in the other byte order, row-padded float64,
column-strided float64, a rank-3 view with a padded middle axis, float16 and
Fortran order; at most about 4 MiB per fixture.

The three predicates (lane fx-host-scans): all finite, any NaN, any infinity,
the fused copy with the finite test, the first-index walk and the List
entries, each fed a quiet NaN, a signalling NaN, a negative NaN, +inf, -inf
at six positions and a clean buffer with signed zeros and subnormals, at an
odd length above the split's grain floor and at a short odd length that stays
on the calling thread, both at an offset inside NaN padding.
"""
from std.memory import bitcast
from core.host_finite_scan import (
    host_all_finite_f32,
    host_all_finite_f64,
    host_all_finite_f32_strided,
    host_all_finite_view,
    host_all_finite_nd,
    host_all_finite_list,
    host_any_nan_f32,
    host_any_nan_list,
    host_any_nan_unit,
    host_any_inf_f32,
    host_first_nonfinite_f32,
    host_copy_f32_finite,
    host_copy_f32_any_nan,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases

comptime ROWS = 4096
comptime COLS = 64
comptime WORDS = ROWS * COLS
comptime PER_LAYOUT = 19
comptime FIXTURES = 4 * PER_LAYOUT
#: The general view scans: 6 layouts, each with 6 positions x 2 non-finite
#: words and one all-finite fixture.
comptime GENERAL_LAYOUTS = 6
comptime PER_GENERAL = 13
comptime GENERAL_FIXTURES = GENERAL_LAYOUTS * PER_GENERAL
#: Bytes before the view, so its words sit at an odd address.
comptime GENERAL_LEAD = 9
#: The predicate fixtures: 2 lengths, each 6 positions x 5 special words and
#: one clean fixture.
comptime PRED_KINDS = 5
comptime PER_PRED = 6 * PRED_KINDS + 1
comptime PRED_FIXTURES = 2 * PER_PRED


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _position(index: Int) -> Int:
    var positions: List[Int] = [0, 1, WORDS // 3, WORDS // 2 - 1, WORDS // 2, WORDS - 1]
    return positions[index]


def _bad32(kind: Int) -> Float32:
    var words: List[UInt32] = [0x7FC00000, 0x7F800000, 0xFF800000]
    return bitcast[DType.float32](words[kind])


def _bad64(kind: Int) -> Float64:
    var words: List[UInt64] = [0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000]
    return bitcast[DType.float64](words[kind])


def _scan32(layout: Int, variant: Int) raises:
    var storage_words = WORDS * 2 if layout == 3 else WORDS
    # NaN padding and endpoint canaries must be ignored by a finite scan.
    var storage = List[Float32](length=storage_words + 2, fill=_bad32(0))
    var row_stride = COLS
    var col_stride = 1
    if layout == 1:
        row_stride = 1
        col_stride = ROWS
    elif layout == 3:
        row_stride = 2 * COLS
    # One fixture-local table includes signed zeros, signed subnormals and
    # largest finite values. Integer construction preserves subnormal bits.
    var finite_words: List[UInt32] = [
        0, 0x80000000, 1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF,
        0x3FA00000, 0xBFC00000,
    ]
    # Setup is local to this independently scheduled fixture.
    for r in range(ROWS):
        for c in range(COLS):
            storage[1 + r * row_stride + c * col_stride] = bitcast[DType.float32](
                finite_words[(r * COLS + c) % len(finite_words)]
            )
    var bad_index = -1
    var bad_word = Float32(0)
    if variant < 18:
        var position = _position(variant // 3)
        bad_index = 1 + (position // COLS) * row_stride + (position % COLS) * col_stride
        bad_word = _bad32(variant % 3)
        storage[bad_index] = bad_word
    var ptr = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var finite = False
    if layout == 0:
        finite = host_all_finite_f32(ptr + 1, WORDS)
    else:
        finite = host_all_finite_f32_strided(Int(ptr + 1), row_stride * 4, col_stride * 4, ROWS, COLS)
    _require(finite == (variant == 18), "f32 layout/position verdict")
    _require(bitcast[DType.uint32](storage[0]) == 0x7FC00000, "leading canary changed")
    _require(bitcast[DType.uint32](storage[storage_words + 1]) == 0x7FC00000, "trailing canary changed")
    if bad_index >= 0:
        _require(bitcast[DType.uint32](storage[bad_index]) == bitcast[DType.uint32](bad_word),
                 "scan changed the nonfinite input word")
    _ = storage


def _scan64(variant: Int) raises:
    var storage = List[Float64](length=WORDS + 2, fill=_bad64(0))
    var finite_words: List[UInt64] = [
        0, 0x8000000000000000, 1, 0x8000000000000001,
        0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF,
        0x3FF4000000000000, 0xBFF8000000000000,
    ]
    for i in range(WORDS):
        storage[i + 1] = bitcast[DType.float64](finite_words[i % len(finite_words)])
    var position = -1
    var bad_word = Float64(0)
    if variant < 18:
        position = _position(variant // 3)
        bad_word = _bad64(variant % 3)
        storage[position + 1] = bad_word
    var ptr = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    _require(host_all_finite_f64(ptr + 1, WORDS) == (variant == 18), "f64 position verdict")
    _require(bitcast[DType.uint64](storage[0]) == 0x7FF8000000000000, "leading f64 canary changed")
    _require(bitcast[DType.uint64](storage[WORDS + 1]) == 0x7FF8000000000000, "trailing f64 canary changed")
    if position >= 0:
        _require(bitcast[DType.uint64](storage[position + 1]) == bitcast[DType.uint64](bad_word),
                 "scan changed the nonfinite f64 word")
    _ = storage


def _general_width(layout: Int) -> Int:
    if layout == 1 or layout == 2:
        return 8
    if layout == 4:
        return 2
    return 4


def _general_offset(layout: Int, r: Int, c: Int) -> Int:
    """The byte offset of logical element (r, c) of the ROWS x COLS grid."""
    if layout == 0:
        return (r * COLS + c) * 4  # C order, the other byte order
    if layout == 1:
        return r * 2 * COLS * 8 + c * 8  # float64 rows with row padding
    if layout == 2:
        return r * COLS * 16 + c * 16  # float64, every second word
    if layout == 3:
        return (r // 4) * (8 * COLS * 4) + (r % 4) * (2 * COLS * 4) + c * 4  # rank 3, padded middle axis
    if layout == 4:
        return (r * COLS + c) * 2  # float16, C order
    return (r + c * ROWS) * 4  # Fortran order


def _general_put(mut storage: List[UInt8], at: Int, value: UInt64, width: Int, native: Bool):
    for b in range(width):
        var slot = b if native else width - 1 - b
        storage[at + slot] = UInt8((value >> UInt64(8 * b)) & UInt64(0xFF))


def _scan_general(layout: Int, variant: Int) raises:
    """One fixture of the general view scans (`host_all_finite_view`,
    `host_all_finite_nd`). The storage is all 0xFF bytes (a NaN at every
    width and byte order), so padding the view skips must be ignored; the
    view's own words are finite except the one the variant plants."""
    var width = _general_width(layout)
    var native = layout != 0
    var span = _general_offset(layout, ROWS - 1, COLS - 1) + width
    var storage = List[UInt8](length=GENERAL_LEAD + span + 8, fill=UInt8(0xFF))
    var finite32: List[UInt64] = [0, 0x80000001, 0x7F7FFFFF, 0x3FA00000]
    var finite64: List[UInt64] = [0, 0x8000000000000001, 0x7FEFFFFFFFFFFFFF, 0x3FF4000000000000]
    var finite16: List[UInt64] = [0, 0x8001, 0x7BFF, 0x3C00]
    var bad32: List[UInt64] = [0x7F800000, 0xFFC00000]
    var bad64: List[UInt64] = [0x7FF0000000000000, 0xFFF8000000000000]
    var bad16: List[UInt64] = [0x7C00, 0xFE00]
    for r in range(ROWS):
        for c in range(COLS):
            var which = (r * COLS + c) % 4
            var word = finite32[which]
            if width == 8:
                word = finite64[which]
            elif width == 2:
                word = finite16[which]
            _general_put(storage, GENERAL_LEAD + _general_offset(layout, r, c), word, width, native)
    if variant < 12:
        var position = _position(variant // 2)
        var bad = bad32[variant % 2]
        if width == 8:
            bad = bad64[variant % 2]
        elif width == 2:
            bad = bad16[variant % 2]
        _general_put(storage, GENERAL_LEAD + _general_offset(layout, position // COLS, position % COLS),
                     bad, width, native)
    var ptr = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var addr = Int(ptr + GENERAL_LEAD)
    var finite = False
    if layout == 0:
        var shape: List[Int] = [ROWS, COLS]
        var strides: List[Int] = [COLS * 4, 4]
        finite = host_all_finite_nd(addr, shape, strides, 4, False)
    elif layout == 1:
        finite = host_all_finite_view(addr, 2 * COLS * 8, 8, ROWS, COLS, 8, True)
    elif layout == 2:
        finite = host_all_finite_view(addr, COLS * 16, 16, ROWS, COLS, 8, True)
    elif layout == 3:
        var shape: List[Int] = [ROWS // 4, 4, COLS]
        var strides: List[Int] = [8 * COLS * 4, 2 * COLS * 4, 4]
        finite = host_all_finite_nd(addr, shape, strides, 4, True)
    elif layout == 4:
        finite = host_all_finite_view(addr, COLS * 2, 2, ROWS, COLS, 2, True)
    else:
        var shape: List[Int] = [ROWS, COLS]
        var strides: List[Int] = [4, ROWS * 4]
        finite = host_all_finite_nd(addr, shape, strides, 4, True)
    _require(finite == (variant == 12), "general view layout/position verdict")
    _ = storage


def _pred_word(kind: Int) -> UInt32:
    # quiet NaN, signalling NaN, negative NaN, +inf, -inf
    var words: List[UInt32] = [0x7FC00000, 0x7F800001, 0xFFC00000, 0x7F800000, 0xFF800000]
    return words[kind]


def _scan_predicates(size_index: Int, variant: Int) raises:
    # An odd length above the grain floor (the split path, with a scalar
    # tail in the last task) and a short odd one (the calling thread).
    var n = WORDS - 3 if size_index == 0 else 37
    var lead = 1 if size_index == 0 else 3
    # NaN padding around the scanned words must not be read.
    var storage = List[Float32](length=n + lead + 2, fill=_bad32(0))
    var staged = List[Float32](length=n + 2, fill=Float32(0))
    var exact = List[Float32](length=n, fill=Float32(0))
    var finite_words: List[UInt32] = [
        0, 0x80000000, 1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF,
        0x3FA00000, 0xBFC00000, 0x007FFFFF, 0x807FFFFF,
    ]
    for i in range(n):
        storage[lead + i] = bitcast[DType.float32](finite_words[i % len(finite_words)])
    var bad = -1
    var kind = -1
    if variant < PER_PRED - 1:
        kind = variant % PRED_KINDS
        var positions: List[Int] = [0, 1, n // 3, n // 2 - 1, n // 2, n - 1]
        bad = positions[variant // PRED_KINDS]
        storage[lead + bad] = bitcast[DType.float32](_pred_word(kind))
    for i in range(n):
        exact[i] = storage[lead + i]
    var is_nan = kind >= 0 and kind < 3
    var is_inf = kind >= 3
    var ptr = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]() + lead
    _require(host_all_finite_f32(ptr, n) == (bad < 0), "all-finite verdict")
    _require(host_any_nan_f32(ptr, n) == is_nan, "any-NaN verdict")
    _require(host_any_inf_f32(ptr, n) == is_inf, "any-infinity verdict")
    _require(host_first_nonfinite_f32(ptr, n) == bad, "first non-finite index")
    _require(host_all_finite_list(exact) == (bad < 0), "all-finite List verdict")
    _require(host_any_nan_list(exact) == is_nan, "any-NaN List verdict")
    _require(host_any_nan_unit(exact) == is_nan, "any-NaN unit verdict")
    var dst = staged.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]() + 1
    _require(host_copy_f32_finite(Int(ptr), Int(dst), n) == (bad < 0), "fused copy verdict")
    for i in range(n):
        _require(bitcast[DType.uint32](staged[1 + i]) == bitcast[DType.uint32](storage[lead + i]),
                 "fused copy changed a word")
    _require(bitcast[DType.uint32](staged[0]) == 0 and bitcast[DType.uint32](staged[n + 1]) == 0,
             "fused copy wrote outside its range")
    # the copy with the any-NaN verdict (GBDT host fit): same range, same words
    for i in range(n + 2):
        staged[i] = bitcast[DType.float32](UInt32(0))
    _require(host_copy_f32_any_nan(Int(ptr), Int(dst), n) == is_nan, "fused copy any-NaN verdict")
    for i in range(n):
        _require(bitcast[DType.uint32](staged[1 + i]) == bitcast[DType.uint32](storage[lead + i]),
                 "fused NaN copy changed a word")
    _require(bitcast[DType.uint32](staged[0]) == 0 and bitcast[DType.uint32](staged[n + 1]) == 0,
             "fused NaN copy wrote outside its range")
    if bad >= 0:
        _require(bitcast[DType.uint32](storage[lead + bad]) == _pred_word(kind), "scan changed the special input word")
    _ = storage
    _ = staged
    _ = exact


def _layout_case(unit: Int) raises:
    var layout = unit // PER_LAYOUT
    var variant = unit % PER_LAYOUT
    if layout == 2:
        _scan64(variant)
    else:
        _scan32(layout, variant)


def _general_case(unit: Int) raises:
    _scan_general(unit // PER_GENERAL, unit % PER_GENERAL)


def _predicate_case(unit: Int) raises:
    _scan_predicates(unit // PER_PRED, unit % PER_PRED)


def test_layouts() raises:
    check_cases("layout", FIXTURES, _layout_case)


def test_general_layouts() raises:
    check_cases("general layout", GENERAL_FIXTURES, _general_case)


def test_predicates() raises:
    check_cases("predicate", PRED_FIXTURES, _predicate_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
