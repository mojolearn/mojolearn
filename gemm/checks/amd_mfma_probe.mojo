# SPDX-License-Identifier: Apache-2.0
"""lane/amd-step-time, 2026-09-24: is one gfx942 FP32 matrix-core step,
`v_mfma_f32_32x32x1f32` (K = 1: every output is ONE product plus its
accumulator), the same function as the contract's FMA, and what does it do
at the subnormal boundary with the wave's MODE register at its default and
with f32 output flush set?

If an MFMA K=1 step returns exactly `fma_rn(a, b, c)` for every element, a
matrix-core GEMM can run the contract's chain (one product per step, `p`
ascending, one accumulator per cell) and only the flush remains to be proven.

The probe needs no output layout: every lane of a wave holds the SAME `a`
and the SAME `b`, so every one of the 32 x 32 x 2 outputs is
`a * b + c[lane][r]` whatever the (i, j) mapping; the accumulators vary per
lane and register. Wave `w` takes a = words[w mod N], b = words[(w div N)
mod N], c = words[(lane * 32 + r + w) mod N] (N words, the seam probe's set
plus tiny normals). For each output the device also computes the VALU
`fma(a, b, c)` and the contract `ftz(fma(a, b, c))`; the host counts the
outputs where the MFMA differs from each, per mode:
  MFMA_PROBE mode=ieee   n=... vs_fma=... vs_rtf=...
  MFMA_PROBE mode=ftzout n=... vs_fma=... vs_rtf=... (MODE.FP_DENORM f32 =
                          allow input, flush output, set in its own kernel)
and prints the first differences as hex.
"""
from std.gpu import block_idx, thread_idx
from std.memory import bitcast
from std.sys import llvm_intrinsic
from std.math import fma
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_AMD, column_name
from transformer.impl.llama.modeling_llama import _upload, _download, _zeros


def _sw_ftz(x: Float32) -> Float32:
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x7F800000)) == UInt32(0) and (b & UInt32(0x007FFFFF)) != UInt32(0):
        return bitcast[DType.float32](b & UInt32(0x80000000))
    return x


def mfma_kernel(
    outp: MutPointer[Float32, MutAnyOrigin],
    words: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
    mode_in: Int32,
):
    """Block = one wave of 64. Writes, per (wave, lane, r): the MFMA
    output, the VALU fma and the contract step, three floats."""
    comptime if TARGET_COLUMN == COLUMN_AMD:
        if mode_in == 1:
            # hwreg(HW_REG_MODE, 4, 2): f32 FP_DENORM = 2 (allow input
            # denormals, flush output denormals).
            llvm_intrinsic["llvm.amdgcn.s.setreg", NoneType](Int32(0x0901), Int32(2))
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
    var base = (w * 64 + lane) * 32 * 3
    comptime for r2 in range(32):
        var f = fma(a, b, c[r2])
        outp.unsafe_store(base + 3 * r2, d[r2])
        outp.unsafe_store(base + 3 * r2 + 1, f)
        outp.unsafe_store(base + 3 * r2 + 2, _sw_ftz(f))


def _hex(w: UInt32) -> String:
    var digits: List[String] = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"
    ]
    var s = String("")
    for k in range(8):
        var nib = Int((w >> UInt32(28 - 4 * k)) & UInt32(0xF))
        s += digits[nib]
    return s


def main() raises:
    print("MFMA_PROBE column=" + column_name(TARGET_COLUMN))
    var words: List[UInt32] = [
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x00800001, 0x80800001,
        0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
        0x3f800001, 0xbf800001, 0x3f7fffff, 0xbf7fffff,
        0x7f7fffff, 0xff7fffff, 0x4b800001, 0xcb800001,
        0x1f000000, 0x9f000001, 0x20800003, 0xa0ffffff, 0x1f7fffff, 0x21000000,
        0x00c00000, 0x80a00001,
    ]
    var seed = UInt32(0x9A718BCD)
    for _ in range(35):
        seed = seed * UInt32(1664525) + UInt32(1013904223)
        words.append(seed & UInt32(0xFEFFFFFF))
    var values = List[Float32]()
    for w in words:
        values.append(bitcast[DType.float32](w))
    var count = len(values)
    var waves = count * count
    var n = waves * 64 * 32
    var ctx = DeviceContext()
    var inputs = _upload(ctx, values)
    # The references (VALU fma, contract step) come from the mode-0 launch,
    # which runs with the default MODE; the mode-1 kernel sets the flush for
    # its whole wave, VALU included, so its own references are not used.
    var refv = List[Float32]()
    for mode in range(2):
        var outb = _zeros(ctx, 3 * n)
        ctx.enqueue_function[mfma_kernel](
            outb.unsafe_ptr(), inputs.unsafe_ptr(), Int32(count), Int32(mode),
            grid_dim=(waves, 1, 1), block_dim=(64, 1, 1),
        )
        ctx.synchronize()
        var h = _download(ctx, outb, 3 * n)
        if mode == 0:
            refv = h.copy()
        var vs_fma = 0
        var vs_rtf = 0
        var shown = 0
        for i in range(n):
            var d = bitcast[DType.uint32](h[3 * i])
            var f = bitcast[DType.uint32](refv[3 * i + 1])
            var t = bitcast[DType.uint32](refv[3 * i + 2])
            if d != f:
                vs_fma += 1
            if d != t:
                vs_rtf += 1
                if shown < 12:
                    print("MFMA_DIFF mode=" + String(mode) + " i=" + String(i) + " mfma=" + _hex(d)
                          + " fma=" + _hex(f) + " rtf=" + _hex(t))
                    shown += 1
        print("MFMA_PROBE mode=" + ("ieee" if mode == 0 else "ftzout") + " n=" + String(n)
              + " vs_fma=" + String(vs_fma) + " vs_rtf=" + String(vs_rtf))
    print("MFMA_DONE")
