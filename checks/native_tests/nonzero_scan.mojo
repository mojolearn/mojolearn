# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn.
"""Native nonzero count/fill correctness, migrated from test_native_nonzero.py.

Sixty independent fixture/implementation pairs run through host_parallelize, using its host-wide
worker pool and preserving the caller's floating-point environment. The scalar
oracle within a fixture intentionally emits one ordered COO stream. It uses an
integer IEEE-754 narrowing implementation, not the product's float cast.

No Python objects, imports, binding calls, device identity records or timings.
The old 8 shapes x 3 seeds remain, with a portable integer fixture generator
replacing Python's Random. All planted bit patterns and midpoint cases remain.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.core_host_impl import nonzero_f64_count, nonzero_f64_fill
from core.mojolearn_shell_impl import nonzero_f64_count_host, nonzero_f64_fill_host


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _rounded_shift(word: UInt64, shift: Int) -> UInt64:
    if shift >= 64:
        return 0
    var q = word >> UInt64(shift)
    var remainder = word & ((UInt64(1) << UInt64(shift)) - 1)
    var half = UInt64(1) << UInt64(shift - 1)
    return q + UInt64(remainder > half or (remainder == half and (q & 1) != 0))


def _narrow_bits(value: Float64) -> UInt32:
    """Independent round-to-nearest-even binary64 -> binary32, including NaN."""
    var u = bitcast[DType.uint64](value)
    var sign = UInt32(u >> 32) & UInt32(0x80000000)
    var exp = Int((u >> 52) & UInt64(0x7FF))
    var fraction = u & UInt64(0xFFFFFFFFFFFFF)
    if exp == 0x7FF:
        if fraction == 0:
            return sign | UInt32(0x7F800000)
        return sign | UInt32(0x7FC00000) | UInt32(fraction >> 29)
    if exp == 0:
        return sign
    var exponent = exp - 1023
    if exponent > 127:
        return sign | UInt32(0x7F800000)
    var significand = fraction | UInt64(0x10000000000000)
    if exponent < -126:
        if exponent < -150:
            return sign
        return sign | UInt32(_rounded_shift(significand, -exponent - 97))
    var rounded = _rounded_shift(significand, 29)
    if rounded == UInt64(0x1000000):
        rounded >>= 1
        exponent += 1
    if exponent > 127:
        return sign | UInt32(0x7F800000)
    return sign | (UInt32(exponent + 127) << 23) | (UInt32(rounded) & UInt32(0x7FFFFF))


def _planted() -> List[Float64]:
    # Exact binary64 words of the former Python fixture, including signed
    # zeros/NaNs, subnormal inputs, binary32 underflow and finite overflow.
    var words: List[UInt64] = [
        0x0, 0x8000000000000000, 0x3FF0000000000000, 0xBFF0000000000000,
        0x4004000000000000, 0x01A56E1FC2F8F359, 0x81A56E1FC2F8F359, 0x1,
        0x3696D601AD376AB9, 0x369FF868BF4D956A, 0x380B38FB9DAA78E4,
        0x3810000000000000, 0x47EFFFFFE0000000, 0x47F074F8C4D3CD7B,
        0xC7F074F8C4D3CD7B, 0x7E37E43C8800759C, 0x7FF0000000000000,
        0xFFF0000000000000, 0x7FF8000000000000, 0xFFF8000000000000,
        0x3FB999999999999A, 0x3FC999999999999A, 0x3FD3333333333333,
        0x3FD5555555555555, 0x400921FB54442D18, 0x4005BF0A8B145769,
    ]
    var values = List[Float64]()
    for word in words:
        values.append(bitcast[DType.float64](word))
    var mantissas: List[UInt32] = [0x800001, 0x800003, 0x800005, 0xFFFFFD, 0xFFFFFF]
    for mantissa in mantissas:
        var low = Float64(bitcast[DType.float32](mantissa))
        var high = Float64(bitcast[DType.float32](mantissa + 1))
        var midpoint = (low + high) / Float64(2)
        values.append(midpoint)
        values.append(-midpoint)
    return values^


def _fixture(fixture_id: Int) -> Tuple[Int, List[Float64]]:
    var planted = _planted()
    if fixture_id < 24:
        var sizes: List[Int] = [1, 2, 3, 7, 16, 33, 64, 129]
        var n = sizes[fixture_id // 3]
        var seed = (fixture_id % 3) * 1000 + n
        var values = List[Float64]()
        var state = UInt64(seed + 1)
        for i in range(n * n):
            state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            var word = state ^ (state >> 29)
            if word % 10 < 6:
                values.append(bitcast[DType.float64]((word & 1) << 63))
            elif word % 10 < 8:
                values.append(planted[Int((word >> 8) % UInt64(len(planted)))])
            else:
                values.append(Float64(Int((word >> 12) % 200001) - 100000) / Float64(10000))
        return n, values^
    var n = len(planted)
    if fixture_id == 25:
        n = 10
    elif fixture_id == 26:
        n = 5
    elif fixture_id == 27:
        n = 3
    elif fixture_id == 28:
        n = 4
    var values = List[Float64](length=n * n, fill=Float64(0))
    if fixture_id == 24:
        for i in range(n):
            values[i] = planted[i]
            values[i * n + i] = planted[i]
    elif fixture_id == 25:
        for i in range(n):
            values[i * n + i] = planted[26 + i]
    elif fixture_id == 27:
        for i in range(len(values)):
            values[i] = bitcast[DType.float64](UInt64(0x8000000000000000))
    elif fixture_id == 28:
        for i in range(len(values)):
            values[i] = Float64(1)
    return n, values^


def _check(mut source: List[Float64], n: Int, ties: Bool, arm: Int) raises:
    var expected_rows = List[Int32]()
    var expected_cols = List[Int32]()
    var expected_values = List[UInt32]()
    # Membership is checked from binary64 bits, independently of the product
    # floating-point comparison: both zeros are absent; every NaN is present.
    for i in range(len(source)):
        if (bitcast[DType.uint64](source[i]) & UInt64(0x7FFFFFFFFFFFFFFF)) != 0:
            expected_rows.append(Int32(i // n))
            expected_cols.append(Int32(i % n))
            expected_values.append(_narrow_bits(source[i]))
    var count = len(expected_rows)
    var sp = source.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    if arm == 0:
        _require(nonzero_f64_count(sp, len(source)) == count, "core count")
    else:
        _require(nonzero_f64_count_host(sp, len(source)) == count, "base count")
    var rows = List[Int32](length=count + 2, fill=Int32(-777))
    var cols = rows.copy()
    var values = List[Float32](length=count + 2, fill=Float32(77))
    var rp = rows.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var cp = cols.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var vp = values.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var got = 0
    if arm == 0:
        got = nonzero_f64_fill(sp, n, n, rp + 1, cp + 1, vp + 1, count)
    else:
        got = nonzero_f64_fill_host(sp, n, n, count, rp + 1, cp + 1, vp + 1)
    _require(got == count, "fill count")
    for i in range(count):
        _require(rows[i + 1] == expected_rows[i], "row order")
        _require(cols[i + 1] == expected_cols[i], "column order")
        var bits = bitcast[DType.uint32](values[i + 1])
        _require(bits == expected_values[i], "binary32 value")
        if ties:
            _require((bits & 1) == 0, "midpoint must round to even")
    _require(rows[0] == -777 and rows[count + 1] == -777, "row canary")
    _require(cols[0] == -777 and cols[count + 1] == -777, "column canary")
    _require(values[0] == 77 and values[count + 1] == 77, "value canary")
    _ = source


def _capacity(arm: Int) raises:
    var source: List[Float64] = [1, 0, 2, 3]
    var rows = List[Int32](length=3, fill=Int32(-777))
    var cols = rows.copy()
    var values = List[Float32](length=3, fill=Float32(77))
    var sp = source.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var rp = rows.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var cp = cols.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var vp = values.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    if arm == 0:
        var refused = False
        try:
            _ = nonzero_f64_fill(sp, 2, 2, rp + 1, cp + 1, vp + 1, 1)
        except error:
            refused = String(error).find("more than 1 nonzero") >= 0
        _require(refused, "core fill must refuse insufficient capacity")
    else:
        _require(nonzero_f64_fill_host(sp, 2, 2, 1, rp + 1, cp + 1, vp + 1) == -1,
                 "base fill must refuse insufficient capacity")
    _require(rows[0] == -777 and rows[2] == -777, "capacity row canary")
    _require(cols[0] == -777 and cols[2] == -777, "capacity column canary")
    _require(values[0] == 77 and values[2] == 77, "capacity value canary")
    if arm == 0:
        _require(nonzero_f64_fill(sp, 0, 5, rp, cp, vp, 1) == 0, "empty core span")
    else:
        _require(nonzero_f64_fill_host(sp, 0, 5, 1, rp, cp, vp) == 0, "empty base span")
    _ = source
    _ = rows
    _ = cols
    _ = values


def main() raises:
    var failed = List[Int](length=60, fill=0)
    var fp = failed.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        var fixture_id = unit // 2
        var arm = unit % 2
        try:
            if fixture_id == 29:
                _capacity(arm)
            else:
                var fixture = _fixture(fixture_id)
                var n = fixture[0]
                var values = fixture[1].copy()
                _check(values, n, fixture_id == 25, arm)
        except error:
            print("FAIL nonzero_scan fixture_id", fixture_id, "arm", arm, String(error))
            fp.unsafe_store(unit, 1)
    host_parallelize(task, 60)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "nonzero_scan failed " + String(failures) + " of 60 native fixture/implementation pairs")
    print("PASS nonzero_scan: 30 native fixtures x 2 typed scan implementations, parallel")
