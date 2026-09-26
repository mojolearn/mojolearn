# SPDX-License-Identifier: Apache-2.0
"""lane/amd-step-time-2, 2026-09-25, the third MFMA probe: the attention
matrix-core step uses `v_mfma_f32_16x16x1f32` (four 16x16 blocks, K = 1).
Before any attention chain may use it, the same two facts the GEMM's
32x32x1 step rests on (amd_mfma_probe2.mojo) are measured for it:

1. under the wave's MODE f32 FP_DENORM field at 2 and at 3, every output
   equals the host's `fma(a, b, c)` (one rounding, subnormal results kept);
2. under MODE 2 the VALU product by `one` (scalar and 2-wide) returns
   `ftz(d)` exactly, signed zero included.

The words are probe2's list (zeros, subnormals, the smallest normals, words
around 1, the largest finite, and 30 hashed words); a = words[w mod N],
b = words[(w div N) mod N], c[r] = words[(lane*16 + r + w) mod N], all
flushed as staging flushes them. Lines:
`MFMA3_MODE v=.. n=.. subnormal_mfma_results=.. mfma_vs_fma=.. mul_vs_ftz=.. pkmul_vs_ftz=..`
Pass: at v=2, mfma_vs_fma=0, mul_vs_ftz=0, pkmul_vs_ftz=0 and
subnormal_mfma_results > 0 (the words do reach the flush).
"""
from max.gpu import block_idx, thread_idx
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


def mode16_kernel(
    outp: MutPointer[Float32, MutAnyOrigin],
    words: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
    mode_in: Int32,
    one: Float32,
):
    comptime if TARGET_COLUMN == COLUMN_AMD:
        if mode_in == 2:
            llvm_intrinsic["llvm.amdgcn.s.setreg", NoneType](Int32(0x0901), Int32(2))
        # 3: the default (allow both), nothing set
    var count = Int(count_in)
    var w = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var a = _sw_ftz(words.unsafe_load(w % count))
    var b = _sw_ftz(words.unsafe_load((w // count) % count))
    var c = SIMD[DType.float32, 16](0.0)
    comptime for r in range(16):
        c[r] = _sw_ftz(words.unsafe_load((lane * 16 + r + w) % count))
    var d = c
    comptime if TARGET_COLUMN == COLUMN_AMD:
        d = llvm_intrinsic["llvm.amdgcn.mfma.f32.16x16x1f32", SIMD[DType.float32, 16]](
            a, b, c, Int32(0), Int32(0), Int32(0)
        )
    var base = (w * 64 + lane) * 16 * 3
    var ones = SIMD[DType.float32, 16](one)
    var pk = d * ones
    comptime for r in range(16):
        outp.unsafe_store(base + 3 * r, d[r])
        outp.unsafe_store(base + 3 * r + 1, d[r] * one)
        outp.unsafe_store(base + 3 * r + 2, pk[r])


def main() raises:
    print("MFMA3_PROBE column=" + column_name(TARGET_COLUMN))
    var ctx = DeviceContext()
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
    var n = waves * 64 * 16
    var inputs = _upload(ctx, values)
    for mode in range(2, 4):
        var outb = _zeros(ctx, 3 * n)
        ctx.enqueue_function[mode16_kernel](
            outb.unsafe_ptr(), inputs.unsafe_ptr(), Int32(count), Int32(mode), Float32(1.0),
            grid_dim=(waves, 1, 1), block_dim=(64, 1, 1),
        )
        ctx.synchronize()
        var h = _download(ctx, outb, 3 * n)
        var mfma_vs_fma = 0
        var mul_vs_ftz = 0
        var pk_vs_ftz = 0
        var subnormal_d = 0
        for i in range(n):
            var wv = i // (64 * 16)
            var lane = (i // 16) % 64
            var r = i % 16
            var a = _sw_ftz(values[wv % count])
            var b = _sw_ftz(values[(wv // count) % count])
            var c = _sw_ftz(values[(lane * 16 + r + wv) % count])
            var f = fma(a, b, c)
            var d = bitcast[DType.uint32](h[3 * i])
            if d != bitcast[DType.uint32](f):
                mfma_vs_fma += 1
            if (d & UInt32(0x7F800000)) == UInt32(0) and (d & UInt32(0x007FFFFF)) != UInt32(0):
                subnormal_d += 1
            var t = bitcast[DType.uint32](_sw_ftz(f))
            if bitcast[DType.uint32](h[3 * i + 1]) != t:
                mul_vs_ftz += 1
            if bitcast[DType.uint32](h[3 * i + 2]) != t:
                pk_vs_ftz += 1
        print("MFMA3_MODE v=" + String(mode) + " n=" + String(n) + " subnormal_mfma_results=" + String(subnormal_d)
              + " mfma_vs_fma=" + String(mfma_vs_fma) + " mul_vs_ftz=" + String(mul_vs_ftz)
              + " pkmul_vs_ftz=" + String(pk_vs_ftz))
    print("MFMA3_DONE")
