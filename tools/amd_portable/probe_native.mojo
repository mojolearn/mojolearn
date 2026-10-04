# AMD portable-path probe 2 (runs on the GPU): launch pkernels natively
# through Mojo's own embedded gfx942 code object and print every output word
# in hex, one kernel per "=== <kernel>" section. hip_load.cpp prints the same
# format for a code object built from the emitted IR; the two must be equal.
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import bitcast
from pkernels import k_muladd, k_pinned, k_math, k_dot

comptime N = 4096
comptime PER_LANE = 37


# Integer-only input generator (no host float arithmetic, so no host
# contraction can enter): sign from bit 31, exponent near 1, 23 mantissa bits.
def gen(i: Int, salt: UInt32) -> Float32:
    var u = (UInt32(i) * UInt32(2654435761)) ^ salt
    u = u ^ (u >> 15)
    u = u * UInt32(2246822519)
    u = u ^ (u >> 13)
    var e = UInt32(0x3E800000) + ((u >> 23) & UInt32(3)) * UInt32(0x00800000)
    var bits = (u & UInt32(0x80000000)) | e | (u & UInt32(0x007FFFFF))
    return bitcast[DType.float32](bits)


def dump(ctx: DeviceContext, name: String, buf: DeviceBuffer[DType.float32], count: Int) raises:
    var h = ctx.enqueue_create_host_buffer[DType.float32](count)
    ctx.enqueue_copy(h, buf)
    ctx.synchronize()
    print("===", name)
    for i in range(count):
        print(hex(bitcast[DType.uint32](h[i])))


def main() raises:
    var ctx = DeviceContext()
    var total = N * PER_LANE
    var ha = ctx.enqueue_create_host_buffer[DType.float32](total)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](total)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](total)
    ctx.synchronize()
    for i in range(total):
        ha[i] = gen(i, UInt32(0x1234567))
        hb[i] = gen(i, UInt32(0x89ABCDE))
        hc[i] = gen(i, UInt32(0x5555AAA))
    var a = ctx.enqueue_create_buffer[DType.float32](total)
    var b = ctx.enqueue_create_buffer[DType.float32](total)
    var c = ctx.enqueue_create_buffer[DType.float32](total)
    var o = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(a, ha)
    ctx.enqueue_copy(b, hb)
    ctx.enqueue_copy(c, hc)
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var cp = c.unsafe_ptr()
    var op = o.unsafe_ptr()
    ctx.enqueue_function[k_muladd](op, ap, bp, cp, Int64(N), grid_dim=(N // 64, 1, 1), block_dim=(64, 1, 1))
    dump(ctx, "k_muladd", o, N)
    ctx.enqueue_function[k_pinned](op, ap, bp, cp, Int64(N), grid_dim=(N // 64, 1, 1), block_dim=(64, 1, 1))
    dump(ctx, "k_pinned", o, N)
    ctx.enqueue_function[k_math](op, ap, bp, cp, Int64(N), grid_dim=(N // 64, 1, 1), block_dim=(64, 1, 1))
    dump(ctx, "k_math", o, N)
    ctx.enqueue_function[k_dot](op, ap, bp, Int64(PER_LANE), grid_dim=(N // 64, 1, 1), block_dim=(64, 1, 1))
    dump(ctx, "k_dot", o, N // 64)
