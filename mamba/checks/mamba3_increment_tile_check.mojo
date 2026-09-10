# SPDX-License-Identifier: Apache-2.0
"""Compare complete S20 increment buffers for scalar, shared-V and tiled plans."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from mamba.impl.mamba_ssm.modules.mamba3_transfer import m3_upload, m3_download
from mamba.impl.mamba_ssm.ops.mamba3_siso import (
    m3_state_increment_kernel, m3_state_increment_shared_v_kernel,
    m3_state_increment_tiled_kernel,
)


def run_case(ctx: DeviceContext, b: Int, t: Int, nh: Int, q: Int, adverse: Bool) raises -> Int:
    var nc = (t + q - 1) // q
    var k = List[Float32]()
    var v = List[Float32]()
    var decay = List[Float32]()
    var words: List[UInt32] = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x3F7FFFFF, 0xBF7FFFFF, 0x3F800001, 0xBF800001, 0x3F800000, 0xBF800000, 0x7E800000, 0xFE800000]
    for i in range(b * t * nh * 128):
        var value = Float32((i * 17 + i // 128) % 127 - 63) / Float32(128.0)
        if adverse: value = bitcast[DType.float32](words[(i * 7 + i // 128) % len(words)])
        k.append(value)
    for i in range(b * t * nh * 64):
        var value = Float32((i * 11 + i // 64) % 113 - 56) / Float32(256.0)
        if adverse: value = bitcast[DType.float32](words[(i * 3 + i // 64) % 14])
        v.append(value)
    for i in range(b * nh * nc * (q + 1)):
        decay.append(Float32((i * 7) % 31 + 1) / Float32(32.0))
    var kd = m3_upload(ctx, k)
    var vd = m3_upload(ctx, v)
    var dd = m3_upload(ctx, decay)
    var count = b * nc * nh * 64 * 128
    var a = ctx.enqueue_create_buffer[DType.float32](count)
    var c = ctx.enqueue_create_buffer[DType.float32](count)
    var z = ctx.enqueue_create_buffer[DType.float32](count)
    ctx.enqueue_function[m3_state_increment_kernel](a.unsafe_ptr(), kd.unsafe_ptr(), vd.unsafe_ptr(), dd.unsafe_ptr(), Int32(b), Int32(t), Int32(nh), Int32(nc), Int32(q), grid_dim=((count + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    ctx.enqueue_function[m3_state_increment_shared_v_kernel](c.unsafe_ptr(), kd.unsafe_ptr(), vd.unsafe_ptr(), dd.unsafe_ptr(), Int32(b), Int32(t), Int32(nh), Int32(nc), Int32(q), grid_dim=((count + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    ctx.enqueue_function[m3_state_increment_tiled_kernel](z.unsafe_ptr(), kd.unsafe_ptr(), vd.unsafe_ptr(), dd.unsafe_ptr(), Int32(b), Int32(t), Int32(nh), Int32(nc), Int32(q), grid_dim=(b * nc * nh * 32, 1, 1), block_dim=(256, 1, 1))
    ctx.synchronize()
    var ah = m3_download(ctx, a, count)
    var ch = m3_download(ctx, c, count)
    var zh = m3_download(ctx, z, count)
    for i in range(count):
        var expected = bitcast[DType.uint32](ah[i])
        if bitcast[DType.uint32](ch[i]) != expected or bitcast[DType.uint32](zh[i]) != expected:
            raise Error("increment tile mismatch at " + String(i) + " t=" + String(t) + " q=" + String(q) + " adverse=" + String(adverse))
    print("M3_INCREMENT_TILE_CASE_PASS", b, t, nh, q, adverse, count)
    _ = kd^
    _ = vd^
    _ = dd^
    _ = a^
    _ = c^
    _ = z^
    return count


def main() raises:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("requires IDENTICAL")
    var ctx = DeviceContext()
    var lengths: List[Int] = [1, 31, 32, 33, 63, 64, 65, 129]
    var chunks: List[Int] = [32, 64]
    var total = 0
    for q in chunks:
        for t in lengths:
            total += run_case(ctx, 2, t, 3, q, False)
            total += run_case(ctx, 2, t, 3, q, True)
    print("M3_INCREMENT_TILE_PASS", total, "three-plan output cells")
