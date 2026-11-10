# SPDX-License-Identifier: Apache-2.0
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
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_finite_scan import (
    host_all_finite_f32,
    host_all_finite_f64,
    host_all_finite_f32_strided,
)

comptime ROWS = 4096
comptime COLS = 64
comptime WORDS = ROWS * COLS
comptime PER_LAYOUT = 19
comptime FIXTURES = 4 * PER_LAYOUT


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


def main() raises:
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        var layout = unit // PER_LAYOUT
        var variant = unit % PER_LAYOUT
        try:
            if layout == 2:
                _scan64(variant)
            else:
                _scan32(layout, variant)
        except error:
            print("FAIL finite_scan layout", layout, "variant", variant, String(error))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "finite_scan failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS finite_scan:", FIXTURES, "native fixtures, 262144 words, 4 layouts, parallel")
