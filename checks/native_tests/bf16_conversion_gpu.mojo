# SPDX-License-Identifier: Apache-2.0
"""Direct GPU conversion kernels; parallel host preparation and integer oracle.

All fixtures share one packed upload and the existing parallel element
kernels. Narrowing must precede roundtrip widening. No Python or hashes.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.native_tests.bf16_conversion_cases import (
    BF16_NARROW_COUNT, BF16_WIDEN_COUNT, BF16_CHUNK, BF16_NARROW_UNITS,
    BF16_UNITS, input_word, expected_narrow, expected_widen, require,
)
from core.host_parallel import host_parallelize
from gemm.checks.gemm_lowbit import bf16_narrow, bf16_widen


def main() raises:
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
    var failures = List[Int](length=BF16_UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    def check(unit: Int) {imm np, imm wp, imm rwp, imm fp}:
        if unit < BF16_NARROW_UNITS:
            var start = unit * BF16_CHUNK
            for j in range(start, min(start + BF16_CHUNK, BF16_NARROW_COUNT)):
                var expected = expected_narrow(input_word(j))
                if np.unsafe_load(j) != expected or bitcast[DType.uint32](wp.unsafe_load(j)) != expected_widen(expected):
                    print("FAIL bf16_conversion_gpu narrow/roundtrip word", j)
                    fp.unsafe_store(unit, 1)
                    return
        else:
            var start = (unit - BF16_NARROW_UNITS) * BF16_CHUNK
            for j in range(start, start + BF16_CHUNK):
                if bitcast[DType.uint32](rwp.unsafe_load(j)) != expected_widen(UInt16(j)):
                    print("FAIL bf16_conversion_gpu raw widening word", j)
                    fp.unsafe_store(unit, 1)
                    return
    host_parallelize(check, BF16_UNITS)
    var failed = 0
    for value in failures:
        failed += value
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
    require(failed == 0, "bf16_conversion_gpu failed " + String(failed) + " units")
    print("PASS bf16_conversion_gpu:", BF16_NARROW_COUNT, "narrow/roundtrip words and", BF16_WIDEN_COUNT, "raw widening words; parallel element kernels and host checks")
