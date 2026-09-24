# SPDX-License-Identifier: Apache-2.0
"""lane/amd-step-time, 2026-09-24, the second MFMA probe. Three questions a
matrix-core IDENTICAL GEMM step needs answered on the device:

1. LAYOUT of `v_mfma_f32_32x32x1f32` (two 32x32 blocks, K = 1): for lane l
   and accumulator register r, which A lane (row) and which B lane (column)
   produced D[l][r]. Read off two launches: A = lane id, B = 1 (D = the A
   lane) and A = 1, B = lane id (D = the B lane), C = 0.
   Prints `MFMA_LAYOUT lane=l r=r a_lane=.. b_lane=..` for every (l, r) as a
   compact table.
2. MODE: with the wave's f32 FP_DENORM field set to each of 0..3, is the
   MFMA still `fma_rn(a, b, c)` unflushed (as measured at 2 by the first
   probe), and does a plain `x * one` (one = 1.0 from a kernel argument, so
   the compiler cannot fold it; scalar and 2-wide) return `ftz(x)` exactly,
   signed zero included, on the subnormal words?
3. Does `v_cmp_class` (the ftz spelling) still see a subnormal as subnormal
   under each mode?
Lines: `MFMA2_MODE v=.. mfma_vs_fma=.. mul_vs_ftz=.. pkmul_vs_ftz=.. class_ok=..`.
"""
from std.gpu import block_idx, thread_idx
from std.memory import bitcast
from std.sys import llvm_intrinsic
from std.math import fma
from max.gpu.host import DeviceContext
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_AMD, column_name
from transformer.impl.llama.modeling_llama import _upload, _download, _zeros


def _sw_ftz(x: Float32) -> Float32:
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x7F800000)) == UInt32(0) and (b & UInt32(0x007FFFFF)) != UInt32(0):
        return bitcast[DType.float32](b & UInt32(0x80000000))
    return x


def layout_kernel(outp: MutPointer[Float32, MutAnyOrigin], which: Int32):
    var lane = Int(thread_idx.x)
    var a = Float32(lane) if which == 0 else Float32(1.0)
    var b = Float32(1.0) if which == 0 else Float32(lane)
    var c = SIMD[DType.float32, 32](0.0)
    var d = c
    comptime if TARGET_COLUMN == COLUMN_AMD:
        d = llvm_intrinsic["llvm.amdgcn.mfma.f32.32x32x1f32", SIMD[DType.float32, 32]](
            a, b, c, Int32(0), Int32(0), Int32(0)
        )
    comptime for r in range(32):
        outp.unsafe_store(lane * 32 + r, d[r])


def mode_kernel(
    outp: MutPointer[Float32, MutAnyOrigin],
    words: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
    mode_in: Int32,
    one: Float32,
):
    """Wave w: a = words[w mod N], b = words[(w div N) mod N] (flushed),
    c[r] = words[(lane*32 + r + w) mod N] flushed. Stores per element:
    the MFMA d, `d * one` (scalar), and the 2-wide product of (d[r], d[r^1])
    by (one, one); and per lane the class test of a raw subnormal word."""
    comptime if TARGET_COLUMN == COLUMN_AMD:
        if mode_in == 0:
            llvm_intrinsic["llvm.amdgcn.s.setreg", NoneType](Int32(0x0901), Int32(0))
        elif mode_in == 1:
            llvm_intrinsic["llvm.amdgcn.s.setreg", NoneType](Int32(0x0901), Int32(1))
        elif mode_in == 2:
            llvm_intrinsic["llvm.amdgcn.s.setreg", NoneType](Int32(0x0901), Int32(2))
        # 3: the default (allow both), nothing set
    var count = Int(count_in)
    var w = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var a = _sw_ftz(words.unsafe_load(w % count))
    var b = _sw_ftz(words.unsafe_load((w // count) % count))
    var c = SIMD[DType.float32, 32](0.0)
    comptime for r in range(32):
        c[r] = _sw_ftz(words.unsafe_load((lane * 32 + r + w) % count))
    var d = c
    comptime if TARGET_COLUMN == COLUMN_AMD:
        d = llvm_intrinsic["llvm.amdgcn.mfma.f32.32x32x1f32", SIMD[DType.float32, 32]](
            a, b, c, Int32(0), Int32(0), Int32(0)
        )
    var base = (w * 64 + lane) * 32 * 4
    comptime for r2 in range(0, 32, 2):
        var pair = SIMD[DType.float32, 2](d[r2], d[r2 + 1])
        var pm = pair * SIMD[DType.float32, 2](one, one)
        outp.unsafe_store(base + 4 * r2, d[r2])
        outp.unsafe_store(base + 4 * r2 + 1, d[r2] * one)
        outp.unsafe_store(base + 4 * r2 + 2, pm[0])
        outp.unsafe_store(base + 4 * r2 + 4, d[r2 + 1])
        outp.unsafe_store(base + 4 * r2 + 5, d[r2 + 1] * one)
        outp.unsafe_store(base + 4 * r2 + 6, pm[1])
        # the class test of a raw subnormal operand word (0x00400000)
        var sub = bitcast[DType.float32](UInt32(0x00400000) | UInt32(lane & 1) << UInt32(31))
        comptime if TARGET_COLUMN == COLUMN_AMD:
            var cls = llvm_intrinsic["llvm.amdgcn.class.f32", Bool, has_side_effect=False](sub, Int32(0x90))
            outp.unsafe_store(base + 4 * r2 + 3, Float32(1.0) if cls else Float32(0.0))
            outp.unsafe_store(base + 4 * r2 + 7, Float32(1.0) if cls else Float32(0.0))


def main() raises:
    print("MFMA2_PROBE column=" + column_name(TARGET_COLUMN))
    var ctx = DeviceContext()
    var lay = _zeros(ctx, 64 * 32)
    var rows = List[Float32]()
    var cols = List[Float32]()
    for which in range(2):
        ctx.enqueue_function[layout_kernel](lay.unsafe_ptr(), Int32(which), grid_dim=(1, 1, 1), block_dim=(64, 1, 1))
        ctx.synchronize()
        var h = _download(ctx, lay, 64 * 32)
        if which == 0:
            rows = h.copy()
        else:
            cols = h.copy()
    for l in range(64):
        var line = String("MFMA_LAYOUT lane=") + String(l) + " a_lanes="
        for r in range(32):
            line += String(Int(rows[l * 32 + r])) + ("," if r < 31 else "")
        line += " b_lanes="
        for r in range(32):
            line += String(Int(cols[l * 32 + r])) + ("," if r < 31 else "")
        print(line)

    var words: List[UInt32] = [
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x00800001, 0x80800001,
        0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
        0x3f800001, 0xbf800001, 0x3f7fffff, 0xbf7fffff,
        0x7f7fffff, 0xff7fffff, 0x4b800001, 0xcb800001,
        0x1f000000, 0x9f000001, 0x20800003, 0xa0ffffff, 0x1f7fffff, 0x21000000,
        0x00c00000, 0x80a00001, 0x1c800001, 0x21800001, 0x9c800000, 0x21800000,
    ]
    var seed = UInt32(0x9A718BCD)
    for _ in range(30):
        seed = seed * UInt32(1664525) + UInt32(1013904223)
        words.append(seed & UInt32(0xFEFFFFFF))
    var values = List[Float32]()
    for w in words:
        values.append(bitcast[DType.float32](w))
    var count = len(values)
    var waves = count * count
    var n = waves * 64 * 32
    var inputs = _upload(ctx, values)
    for mode in range(4):
        var outb = _zeros(ctx, 4 * n)
        ctx.enqueue_function[mode_kernel](
            outb.unsafe_ptr(), inputs.unsafe_ptr(), Int32(count), Int32(mode), Float32(1.0),
            grid_dim=(waves, 1, 1), block_dim=(64, 1, 1),
        )
        ctx.synchronize()
        var h = _download(ctx, outb, 4 * n)
        var mfma_vs_fma = 0
        var mul_vs_ftz = 0
        var pk_vs_ftz = 0
        var class_bad = 0
        var subnormal_d = 0
        for i in range(n):
            var wv = i // (64 * 32)
            var lane = (i // 32) % 64
            var r = i % 32
            var a = _sw_ftz(values[wv % count])
            var b = _sw_ftz(values[(wv // count) % count])
            var c = _sw_ftz(values[(lane * 32 + r + wv) % count])
            var f = fma(a, b, c)  # host fma: IEEE, subnormals kept
            var d = bitcast[DType.uint32](h[4 * i])
            if d != bitcast[DType.uint32](f):
                mfma_vs_fma += 1
            var t = bitcast[DType.uint32](_sw_ftz(f))
            if (d & UInt32(0x7F800000)) == UInt32(0) and (d & UInt32(0x007FFFFF)) != UInt32(0):
                subnormal_d += 1
            if bitcast[DType.uint32](h[4 * i + 1]) != t:
                mul_vs_ftz += 1
            if bitcast[DType.uint32](h[4 * i + 2]) != t:
                pk_vs_ftz += 1
            if h[4 * i + 3] != Float32(1.0):
                class_bad += 1
        print("MFMA2_MODE v=" + String(mode) + " n=" + String(n) + " subnormal_mfma_results=" + String(subnormal_d)
              + " mfma_vs_fma=" + String(mfma_vs_fma) + " mul_vs_ftz=" + String(mul_vs_ftz)
              + " pkmul_vs_ftz=" + String(pk_vs_ftz) + " class_not_subnormal=" + String(class_bad))
    print("MFMA2_DONE")
