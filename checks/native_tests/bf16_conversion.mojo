# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Independent integer oracle for the native BF16 conversion contract.

All 65,536 upper halves are narrowed with discarded tails zero, below tie,
at tie and above tie. An additional 37 x 53 sample retains the former
Python test's shape and 13-bit-significand/eight-binade distribution, using
a native deterministic generator rather than importing NumPy's RNG.
Every raw BF16 word is widened too, including NaNs and subnormals.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, u16_ptr
from decomposition.linalg_host_impl import to_bf16_host_run, from_bf16_host_run
from max.gpu.host import DeviceContext
from gemm.checks.gemm_lowbit import bf16_narrow, bf16_widen
from std.sys.compile import is_defined
from checks.native_tests.kit import download_host_into, upload_host_into
from checks.native_tests.lowbit_oracles import expected_narrow, expected_widen


comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()

def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


comptime BF16_NARROW_EXHAUSTIVE = 65536 * 4
comptime BF16_NARROW_COUNT = BF16_NARROW_EXHAUSTIVE + 37 * 53
comptime BF16_WIDEN_COUNT = 65536
comptime BF16_CHUNK = 1024
comptime BF16_NARROW_UNITS = (BF16_NARROW_COUNT + BF16_CHUNK - 1) // BF16_CHUNK
comptime BF16_WIDEN_UNITS = BF16_WIDEN_COUNT // BF16_CHUNK
comptime BF16_UNITS = BF16_NARROW_UNITS + BF16_WIDEN_UNITS


def input_word(index: Int) -> UInt32:
    if index < BF16_NARROW_EXHAUSTIVE:
        var tail_id = index % 4
        var tail = UInt32(0)
        if tail_id == 1:
            tail = 0x7FFF
        elif tail_id == 2:
            tail = 0x8000
        elif tail_id == 3:
            tail = 0xFFFF
        return UInt32(index // 4) << 16 | tail
    var h = UInt64(index - BF16_NARROW_EXHAUSTIVE + 1) * UInt64(0x9E3779B97F4A7C15)
    h ^= h >> 30
    h *= UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 27
    var mantissa = UInt32(h & 0xFFF) << 11
    var exponent = (UInt32((h >> 12) & 7) + 123) << 23
    var sign = UInt32((h >> 19) & 1) << 31
    return sign | exponent | mantissa


# ---- The host column entry (was bf16_conversion_cpu.mojo) ----
# Parallel typed host conversion checks with independent integer expectations.
def check_unit(unit: Int) raises:
    if unit < BF16_NARROW_UNITS:
        var start = unit * BF16_CHUNK
        var count = min(BF16_CHUNK, BF16_NARROW_COUNT - start)
        var source = List[Float32](length=count, fill=0)
        var narrowed = List[UInt16](length=count, fill=0)
        var widened = List[Float32](length=count, fill=0)
        for j in range(count):
            source[j] = bitcast[DType.float32](input_word(start + j))
        to_bf16_host_run(u16_ptr(Int(narrowed.unsafe_ptr())), Int(source.unsafe_ptr()), count)
        from_bf16_host_run(f32_ptr(Int(widened.unsafe_ptr())), Int(narrowed.unsafe_ptr()), count)
        for j in range(count):
            var expected = expected_narrow(input_word(start + j))
            require(narrowed[j] == expected, "host BF16 narrowing word " + String(start + j))
            require(bitcast[DType.uint32](widened[j]) == expected_widen(expected), "host BF16 roundtrip word " + String(start + j))
        _ = source
        _ = narrowed
        _ = widened
    else:
        var start = (unit - BF16_NARROW_UNITS) * BF16_CHUNK
        var source = List[UInt16](length=BF16_CHUNK, fill=0)
        var widened = List[Float32](length=BF16_CHUNK, fill=0)
        for j in range(BF16_CHUNK):
            source[j] = UInt16(start + j)
        from_bf16_host_run(f32_ptr(Int(widened.unsafe_ptr())), Int(source.unsafe_ptr()), BF16_CHUNK)
        for j in range(BF16_CHUNK):
            require(bitcast[DType.uint32](widened[j]) == expected_widen(UInt16(start + j)), "host raw BF16 widening word " + String(start + j))
        _ = source
        _ = widened


# The host column: parallel typed host conversion checks with independent
# integer expectations.
def test_host_units() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("bf16_conversion host", BF16_UNITS, check_unit)


# The device column: direct GPU conversion kernels; parallel host preparation
# and integer oracle. All fixtures share one packed upload and the existing
# parallel element kernels. Narrowing must precede roundtrip widening.
def test_device_conversion() raises:
    comptime if not CPU_COLUMN_BUILD:
        var ctx = DeviceContext()
        var source = ctx.enqueue_create_host_buffer[DType.float32](BF16_NARROW_COUNT)
        var raw = ctx.enqueue_create_host_buffer[DType.uint16](BF16_WIDEN_COUNT)
        ctx.synchronize()
        var sp = source.unsafe_ptr()
        var rp = raw.unsafe_ptr()
        def prepare(unit: Int) {imm sp, imm rp}:
            if unit < BF16_NARROW_UNITS:
                var start = unit * BF16_CHUNK
                for j in range(start, min(start + BF16_CHUNK, BF16_NARROW_COUNT)):
                    sp.unsafe_store(j, bitcast[DType.float32](input_word(j)))
            else:
                var start = (unit - BF16_NARROW_UNITS) * BF16_CHUNK
                for j in range(start, start + BF16_CHUNK):
                    rp.unsafe_store(j, UInt16(j))
        host_parallelize(prepare, BF16_UNITS)
        var dx = ctx.enqueue_create_buffer[DType.float32](BF16_NARROW_COUNT)
        var dn = ctx.enqueue_create_buffer[DType.uint16](BF16_NARROW_COUNT)
        var dw = ctx.enqueue_create_buffer[DType.float32](BF16_NARROW_COUNT)
        var dr = ctx.enqueue_create_buffer[DType.uint16](BF16_WIDEN_COUNT)
        var drw = ctx.enqueue_create_buffer[DType.float32](BF16_WIDEN_COUNT)
        upload_host_into[DType.float32](ctx, dx, source)
        upload_host_into[DType.uint16](ctx, dr, raw)
        bf16_narrow(ctx, dn, dx, BF16_NARROW_COUNT)
        bf16_widen(ctx, dw, dn, BF16_NARROW_COUNT)
        bf16_widen(ctx, drw, dr, BF16_WIDEN_COUNT)
        var narrowed = ctx.enqueue_create_host_buffer[DType.uint16](BF16_NARROW_COUNT)
        var widened = ctx.enqueue_create_host_buffer[DType.float32](BF16_NARROW_COUNT)
        var raw_widened = ctx.enqueue_create_host_buffer[DType.float32](BF16_WIDEN_COUNT)
        ctx.synchronize()
        download_host_into[DType.uint16](ctx, dn, narrowed)
        download_host_into[DType.float32](ctx, dw, widened)
        download_host_into[DType.float32](ctx, drw, raw_widened)
        ctx.synchronize()
        var np = narrowed.unsafe_ptr()
        var wp = widened.unsafe_ptr()
        var rwp = raw_widened.unsafe_ptr()
        def check(unit: Int) raises {imm np, imm wp, imm rwp}:
            if unit < BF16_NARROW_UNITS:
                var start = unit * BF16_CHUNK
                for j in range(start, min(start + BF16_CHUNK, BF16_NARROW_COUNT)):
                    var expected = expected_narrow(input_word(j))
                    if np.unsafe_load(j) != expected or bitcast[DType.uint32](wp.unsafe_load(j)) != expected_widen(expected):
                        raise Error("bf16 narrow/widen word mismatch at index " + String(j))
            else:
                var start = (unit - BF16_NARROW_UNITS) * BF16_CHUNK
                for j in range(start, start + BF16_CHUNK):
                    if bitcast[DType.uint32](rwp.unsafe_load(j)) != expected_widen(UInt16(j)):
                        raise Error("bf16 widen word mismatch at half " + String(j))
        check_cases("bf16_conversion device", BF16_UNITS, check)
        # Keep pinned host allocations and device storage alive through the last
        # queued copy and every unsafe-pointer read by the joined host tasks.
        _ = source
        _ = raw
        _ = narrowed
        _ = widened
        _ = raw_widened
        _ = dx
        _ = dn
        _ = dw
        _ = dr
        _ = drw


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    # A column's tests are compiled empty on the other column and reported SKIP there.
    comptime if CPU_COLUMN_BUILD:
        suite.skip[test_device_conversion]()
    else:
        suite.skip[test_host_units]()
    suite^.run()
