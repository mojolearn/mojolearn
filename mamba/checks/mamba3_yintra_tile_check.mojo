# SPDX-License-Identifier: Apache-2.0
"""Full S16 scalar/tiled identity, causal exclusions and guarded output extent."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from mamba.impl.modules.mamba3_transfer import m3_upload, m3_download
from mamba.impl.ops.mamba3_siso import m3_yintra_kernel, m3_yintra_tiled_kernel


def run_case(ctx: DeviceContext, l: Int, q0: Int, q: Int, adverse: Bool) raises -> Int:
    comptime b = 2
    comptime nh = 3
    var tw = q0 + l
    var nc = (tw + q - 1) // q
    var qk = List[Float32]()
    var seg = List[Float32]()
    var v = List[Float32]()
    var words: List[UInt32] = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x3F7FFFFF, 0xBF7FFFFF, 0x3F800001, 0xBF800001, 0x3F800000, 0xBF800000, 0x7E800000, 0xFE800000]
    var poison = bitcast[DType.float32](UInt32(0x7FC01234))
    for i in range(b * nc * nh * q * q):
        var a = Float32((i * 17 + i // q) % 127 - 63) / Float32(128.0)
        var s = Float32((i * 7) % 31 + 1) / Float32(32.0)
        if adverse:
            a = bitcast[DType.float32](words[(i * 7 + i // q) % len(words)])
            s = bitcast[DType.float32](words[(i * 3 + i // q) % 14])
        # Neither spelling may evaluate diagonal/future coefficients.
        if i % q >= (i // q) % q:
            a = poison
            s = poison
        qk.append(a)
        seg.append(s)
    for i in range(b * tw * nh * 64):
        var value = Float32((i * 11 + i // 64) % 113 - 56) / Float32(256.0)
        if adverse: value = bitcast[DType.float32](words[(i * 3 + i // 64) % 14])
        v.append(value)
    var qd = m3_upload(ctx, qk)
    var sd = m3_upload(ctx, seg)
    var vd = m3_upload(ctx, v)
    var count = b * l * nh * 64
    var initial = List[Float32]()
    for i in range(count + 17): initial.append(poison)
    var a = m3_upload(ctx, initial)
    var z = m3_upload(ctx, initial)
    ctx.enqueue_function[m3_yintra_kernel](a.unsafe_ptr(), qd.unsafe_ptr(), sd.unsafe_ptr(), vd.unsafe_ptr(), Int32(b), Int32(l), Int32(q0), Int32(nh), Int32(nc), Int32(q), grid_dim=((count + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    ctx.enqueue_function[m3_yintra_tiled_kernel](z.unsafe_ptr(), qd.unsafe_ptr(), sd.unsafe_ptr(), vd.unsafe_ptr(), Int32(b), Int32(l), Int32(q0), Int32(nh), Int32(nc), Int32(q), grid_dim=(b * nh * ((tw + 7) // 8) * 2, 1, 1), block_dim=(256, 1, 1))
    ctx.synchronize()
    var ah = m3_download(ctx, a, count + 17)
    var zh = m3_download(ctx, z, count + 17)
    for i in range(count + 17):
        var expected = bitcast[DType.uint32](ah[i])
        if bitcast[DType.uint32](zh[i]) != expected:
            raise Error("yintra tile mismatch at " + String(i) + " l=" + String(l) + " q0=" + String(q0) + " q=" + String(q))
        if i >= count and expected != UInt32(0x7FC01234):
            raise Error("yintra output extent overwritten")
    print("M3_YINTRA_TILE_CASE_PASS", l, q0, q, adverse, count)
    _ = qd^
    _ = sd^
    _ = vd^
    _ = a^
    _ = z^
    return count


def main() raises:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("requires IDENTICAL")
    var ctx = DeviceContext()
    var lengths: List[Int] = [1, 8, 9, 64, 65]
    var offsets: List[Int] = [0, 7, 31, 63, 64]
    var chunks: List[Int] = [32, 64]
    var total = 0
    for q in chunks:
        for q0 in offsets:
            for l in lengths:
                total += run_case(ctx, l, q0, q, False)
                total += run_case(ctx, l, q0, q, True)
    print("M3_YINTRA_TILE_PASS", total, "output cells plus guarded suffixes")
