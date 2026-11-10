# SPDX-License-Identifier: Apache-2.0
"""Independent weight conversion fixtures formerly checked with NumPy.

Keep the original 40x32 and 32x24 shapes/uniform range using a native fixture
generator. The former unused special-row helper now supplies a third executed
fixture. BF16 expectations use quotient/remainder rounding; INT8 expectations
use the independent Float64 ties-even oracle in int8_linalg_cases.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from checks.native_tests.bf16_conversion_cases import expected_narrow, expected_widen
from checks.native_tests.int8_linalg_cases import require

comptime LOWBIT_WEIGHT_FIXTURES = 3


def shape(fixture_id: Int) -> Tuple[Int, Int]:
    if fixture_id == 0:
        return (40, 32)
    if fixture_id == 1:
        return (32, 24)
    return (7, 4)


def values(fixture_id: Int) -> List[Float32]:
    if fixture_id == 2:
        var special: List[Float32] = [
            0.0, -0.0, 0.0, 0.0,
            1e-40, -1e-40, 1.0, -1.0,
            0.5, 1.5, 2.5, -2.5,
            1e-30, 2e-30, -3e-30, 4e-30,
            3e38, -1e38, 1.0, 2.0,
            0.1, -0.2, 0.3, -0.4,
            65504.0, 0.000030517578125, -0.000030517578125, 100.0,
        ]
        return special^
    var dims = shape(fixture_id)
    var rows = dims[0]
    var cols = dims[1]
    var source = List[Float32](length=rows * cols, fill=0)
    var sp = source.unsafe_ptr()
    def row_task(row: Int) {imm sp, imm cols, imm fixture_id}:
        for col in range(cols):
            var h = UInt64(row * cols + col + 1) * UInt64(0x9E3779B97F4A7C15) + UInt64(1 if fixture_id == 0 else 3)
            h ^= h >> 30
            h *= UInt64(0xBF58476D1CE4E5B9)
            h ^= h >> 27
            sp.unsafe_store(row * cols + col, Float32(Int(h & 0xFFFFFF) - 8388608) * 0.00000001490116119384765625)
    host_parallelize(row_task, rows)
    return source^


def check_bf16(source: List[Float32], bits: List[UInt16], restored: List[Float32], rows: Int, cols: Int) raises:
    var sp = source.unsafe_ptr()
    var bp = bits.unsafe_ptr()
    var rp = restored.unsafe_ptr()
    var failures = List[Int](length=rows, fill=0)
    var fp = failures.unsafe_ptr()
    def row_task(row: Int) {imm sp, imm bp, imm rp, imm fp, imm cols}:
        for col in range(cols):
            var i = row * cols + col
            var expected = expected_narrow(bitcast[DType.uint32](sp.unsafe_load(i)))
            if bp.unsafe_load(i) != expected or bitcast[DType.uint32](rp.unsafe_load(i)) != expected_widen(expected):
                fp.unsafe_store(row, 1)
    host_parallelize(row_task, rows)
    for row in range(rows):
        require(failures[row] == 0, "packed BF16 weight row " + String(row))
    _ = source
    _ = bits
    _ = restored
