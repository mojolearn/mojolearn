# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2512: `core/device_zero.mojo` zeroes EXACTLY the span it is
given. Every byte inside is 0 afterwards, every byte outside keeps its
0xA5 fill, over offsets and lengths that exercise the aligned body, the
byte tail, the unaligned-start arm and the empty span.

DEVIATION 2560/2561: `enqueue_fill` (the gbdt fill helper) writes the
same bytes as `ctx.enqueue_memset` AND as a host oracle that stores the
value n times, for float32/int32/uint32/uint8, values including -0.0, a
NaN payload and a denormal, on poisoned buffers, at lengths that reach the
byte tail and the grid-stride path past the block cap. Rule 8, one named
check per side of the switch:

    pixi run check-device-zero          # kernel arm (default)
    pixi run check-device-zero-memset   # -D MOJOLEARN_2560_MEMSET_FILL=1
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.sys.compile import is_defined
from std.sys.info import size_of
from core.device_zero import enqueue_zero_bytes, enqueue_fill

comptime MEMSET_ARM = is_defined["MOJOLEARN_2560_MEMSET_FILL"]()


def check_span(ctx: DeviceContext, total: Int, offset: Int, nbytes: Int) raises -> Bool:
    var d = ctx.enqueue_create_buffer[DType.uint8](total)
    var h = ctx.enqueue_create_host_buffer[DType.uint8](total)
    for i in range(total):
        h[i] = UInt8(0xA5)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    var p = d.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    enqueue_zero_bytes(ctx, p + offset, nbytes)
    ctx.enqueue_copy(dst_buf=h, src_buf=d)
    ctx.synchronize()
    var bad = 0
    for i in range(total):
        var inside = i >= offset and i < offset + nbytes
        var want = UInt8(0) if inside else UInt8(0xA5)
        if h[i] != want:
            bad += 1
    if bad != 0:
        print("FAIL total", total, "offset", offset, "nbytes", nbytes, "bad bytes", bad)
    _ = d^
    _ = h^
    return bad == 0


def check_fill[dt: DType](
    ctx: DeviceContext, n: Int, value: Scalar[dt], label: String
) raises -> Bool:
    """`enqueue_fill` vs `enqueue_memset` vs a host store loop, byte for
    byte, over `n` poisoned elements."""
    var nbytes = n * size_of[Scalar[dt]]()
    var d_fill = ctx.enqueue_create_buffer[dt](n)
    var d_ref = ctx.enqueue_create_buffer[dt](n)
    var h_fill = ctx.enqueue_create_host_buffer[dt](n)
    var h_ref = ctx.enqueue_create_host_buffer[dt](n)
    var h_oracle = ctx.enqueue_create_host_buffer[dt](n)
    var fp = h_fill.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(nbytes):
        fp[unsafe_offset=i] = UInt8(0xA5)
    ctx.enqueue_copy(dst_buf=d_fill, src_buf=h_fill)
    ctx.enqueue_copy(dst_buf=d_ref, src_buf=h_fill)
    enqueue_fill(ctx, d_fill, value)
    ctx.enqueue_memset(d_ref, value)
    ctx.enqueue_copy(dst_buf=h_fill, src_buf=d_fill)
    ctx.enqueue_copy(dst_buf=h_ref, src_buf=d_ref)
    ctx.synchronize()
    var op = h_oracle.unsafe_ptr()
    for i in range(n):
        op.unsafe_store(i, value)
    var rp = h_ref.unsafe_ptr().unsafe_bitcast[UInt8]()
    var orp = h_oracle.unsafe_ptr().unsafe_bitcast[UInt8]()
    var bad_ref = 0
    var bad_oracle = 0
    for i in range(nbytes):
        if fp[unsafe_offset=i] != rp[unsafe_offset=i]:
            bad_ref += 1
        if fp[unsafe_offset=i] != orp[unsafe_offset=i]:
            bad_oracle += 1
    if bad_ref != 0 or bad_oracle != 0:
        print(
            "FAIL fill", label, "n", n, "bytes != memset", bad_ref,
            "bytes != host oracle", bad_oracle,
        )
    _ = d_fill^
    _ = d_ref^
    _ = h_fill^
    _ = h_ref^
    _ = h_oracle^
    return bad_ref == 0 and bad_oracle == 0


def main() raises:
    var ctx = DeviceContext()
    var ok = True
    var offsets = [0, 4, 16, 1, 3, 32]
    var lengths = [0, 1, 3, 4, 15, 16, 17, 31, 33, 4096, 4101, 1 << 20, (1 << 20) + 7]
    var n_cases = 0
    for o in range(len(offsets)):
        for l in range(len(lengths)):
            var total = offsets[o] + lengths[l] + 64
            ok = check_span(ctx, total, offsets[o], lengths[l]) and ok
            n_cases += 1
    # a large aligned span past the block cap (grid-stride path)
    ok = check_span(ctx, 64 << 20, 16, (64 << 20) - 48) and ok
    n_cases += 1
    print("device_zero_check spans:", n_cases)

    comptime if MEMSET_ARM:
        print("device_zero_check fill arm: memset (DEVIATION 2560 opt-out)")
    else:
        print("device_zero_check fill arm: kernel (DEVIATION 2512/2561)")
    # 524,289 and 3,000,001 elements pass the 2048 x 256 thread cap on both
    # the fill kernel and the 16-byte zero kernel (grid-stride arm).
    var ns = [1, 3, 4, 5, 15, 16, 17, 4096, 4101, 524289, 3000001]
    var f32 = List[Float32]()
    f32.append(Float32(0.0))
    f32.append(bitcast[DType.float32](UInt32(0x80000000)))  # -0.0
    f32.append(Float32(1.0))
    f32.append(Float32(0.3))
    f32.append(bitcast[DType.float32](UInt32(0x7FC00001)))  # NaN payload
    f32.append(bitcast[DType.float32](UInt32(1)))           # denormal
    var i32 = List[Int32]()
    i32.append(Int32(0))
    i32.append(Int32(-1))
    i32.append(Int32(0x12345678))
    var u32 = List[UInt32]()
    u32.append(UInt32(0))
    u32.append(UInt32(7))
    u32.append(UInt32(0xDEADBEEF))
    var u8 = List[UInt8]()
    u8.append(UInt8(0))
    u8.append(UInt8(9))
    var fills = 0
    for k in range(len(ns)):
        var n = ns[k]
        for v in range(len(f32)):
            ok = check_fill[DType.float32](ctx, n, f32[v], String("float32 #") + String(v)) and ok
            fills += 1
        for v in range(len(i32)):
            ok = check_fill[DType.int32](ctx, n, i32[v], String("int32 #") + String(v)) and ok
            fills += 1
        for v in range(len(u32)):
            ok = check_fill[DType.uint32](ctx, n, u32[v], String("uint32 #") + String(v)) and ok
            fills += 1
        for v in range(len(u8)):
            ok = check_fill[DType.uint8](ctx, n, u8[v], String("uint8 #") + String(v)) and ok
            fills += 1
    print("device_zero_check fills:", fills)
    if not ok:
        raise Error("device_zero_check FAILED")
    print("device_zero_check PASS,", n_cases, "spans,", fills, "fills")
