# SPDX-License-Identifier: Apache-2.0
"""Independent integer oracle for the native BF16 conversion contract.

All 65,536 upper halves are narrowed with discarded tails zero, below tie,
at tie and above tie. An additional 37 x 53 sample retains the former
Python test's shape and 13-bit-significand/eight-binade distribution, using
a native deterministic generator rather than importing NumPy's RNG.
Every raw BF16 word is widened too, including NaNs and subnormals.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, u16_ptr
from decomposition.linalg_host_impl import to_bf16_host_run, from_bf16_host_run
from max.gpu.host import DeviceContext
from gemm.checks.gemm_lowbit import bf16_narrow, bf16_widen
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


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


def expected_narrow(word: UInt32) -> UInt16:
    # Integer quotient/remainder rounding, independent of the product's
    # add-bias implementation. Native contract flushes binary32 subnormals
    # before rounding and preserves/quietens the high NaN payload bits.
    var exponent = (word >> 23) & 255
    var fraction = word & 0x007FFFFF
    if exponent == 0:
        return UInt16((word >> 16) & 0x8000)
    var upper = word >> 16
    if exponent == 255 and fraction != 0:
        return UInt16(upper | 0x0040)
    var remainder = word & 0xFFFF
    if remainder > 0x8000 or (remainder == 0x8000 and (upper & 1) != 0):
        upper += 1
    return UInt16(upper)


def expected_widen(word: UInt16) -> UInt32:
    return UInt32(word) << 16


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


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(BF16_UNITS)
    def task(unit: Int) {imm cases}:
        try:
            check_unit(unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, BF16_UNITS)


# ---- The device column entry (was bf16_conversion_gpu.mojo) ----
# Direct GPU conversion kernels; parallel host preparation and integer oracle.
#
# All fixtures share one packed upload and the existing parallel element
# kernels. Narrowing must precede roundtrip widening. No Python or hashes.
def _run_gpu(mut report: Report) raises:
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
    report.parallel(prepare, BF16_UNITS)
    var dx = ctx.enqueue_create_buffer[DType.float32](BF16_NARROW_COUNT)
    var dn = ctx.enqueue_create_buffer[DType.uint16](BF16_NARROW_COUNT)
    var dw = ctx.enqueue_create_buffer[DType.float32](BF16_NARROW_COUNT)
    var dr = ctx.enqueue_create_buffer[DType.uint16](BF16_WIDEN_COUNT)
    var drw = ctx.enqueue_create_buffer[DType.float32](BF16_WIDEN_COUNT)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=source.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dr, src_ptr=raw.unsafe_ptr())
    bf16_narrow(ctx, dn, dx, BF16_NARROW_COUNT)
    bf16_widen(ctx, dw, dn, BF16_NARROW_COUNT)
    bf16_widen(ctx, drw, dr, BF16_WIDEN_COUNT)
    var narrowed = ctx.enqueue_create_host_buffer[DType.uint16](BF16_NARROW_COUNT)
    var widened = ctx.enqueue_create_host_buffer[DType.float32](BF16_NARROW_COUNT)
    var raw_widened = ctx.enqueue_create_host_buffer[DType.float32](BF16_WIDEN_COUNT)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=narrowed.unsafe_ptr(), src_buf=dn)
    ctx.enqueue_copy(dst_ptr=widened.unsafe_ptr(), src_buf=dw)
    ctx.enqueue_copy(dst_ptr=raw_widened.unsafe_ptr(), src_buf=drw)
    ctx.synchronize()
    var np = narrowed.unsafe_ptr()
    var wp = widened.unsafe_ptr()
    var rwp = raw_widened.unsafe_ptr()
    var cases = report.cases(BF16_UNITS)
    def check(unit: Int) {imm np, imm wp, imm rwp, imm cases}:
        if unit < BF16_NARROW_UNITS:
            var start = unit * BF16_CHUNK
            for j in range(start, min(start + BF16_CHUNK, BF16_NARROW_COUNT)):
                var expected = expected_narrow(input_word(j))
                if np.unsafe_load(j) != expected or bitcast[DType.uint32](wp.unsafe_load(j)) != expected_widen(expected):
                    cases.fail(unit, error)
                    return
        else:
            var start = (unit - BF16_NARROW_UNITS) * BF16_CHUNK
            for j in range(start, start + BF16_CHUNK):
                if bitcast[DType.uint32](rwp.unsafe_load(j)) != expected_widen(UInt16(j)):
                    cases.fail(unit, error)
                    return
    report.parallel(check, BF16_UNITS)
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


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("bf16_conversion", cpu=True, gpu=True, serial="", run=run_cases)
