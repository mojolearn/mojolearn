# SPDX-License-Identifier: Apache-2.0
"""DEVIATION 2701: what each column's NATIVE FMA does at the smallest-normal
boundary, against the shipped per-step seam, on the 262,144 adversarial
triples of `transformer/checks/attention_fma_boundary_check.mojo`.

THE QUESTION. The
identity contract's seam is round-then-flush, `ftz(fma_rn(a, b, acc))`. On
NVIDIA that is two issued instructions per product step (`fma.rn` then
`mul.rn.ftz`), which halves the reachable fp32 peak (brief 3.2), because the
single `fma.rn.ftz` flushes BEFORE rounding at the boundary a=0x3f7fffff,
b=0x00800000, acc=+0 (it returns 0 where round-then-flush returns
0x00800000). The kNN audit of 2026-09-09 found Apple's native FMA does the
SAME pre-round flush (bench/results/knn/2026-09-09-selector-final/apple/BOUNDARY.md),
so on Apple the shipped GEMM seam `ftz(fma(...))` is already flush-before-
round at that boundary and does not implement the contract there. If every
column's native FMA flushes before rounding, or can be made to, a contract
whose seam IS the native instruction costs ONE instruction per step on
every column and closes the Apple gap. This probe measures each column's
native FMA; it changes no shipped line.

WHAT IT PRINTS, per triple set, five lanes:
  shipped   `_tuned_step(ftz(a), ftz(b), ftz(acc))`, the column's shipped seam
  fma       `identical_mul_add(...)`, the column's native FMA, NO flush after
  hwftz     NVIDIA: `llvm.nvvm.fma.rn.ftz.f`; other columns: the `fma` lane again
  swrtf     `ftz(identical_mul_add(...))`, the software round-then-flush spelling
  class     post-round AMD class flush; software spelling on other columns
  nativefix `rtf_fix(fma(...))` with NO software flush (lane/apple-seam-repair,
            2026-09-18): on Apple the native FMA's own flush plus the exact
            zero repair, the candidate cheaper spelling; on NVIDIA and AMD
            `rtf_fix` is the identity, so this lane is `fma` again (`none`)
The closed wave-mode experiment is not launched. `class_shipped` reports
whether this build enables the class spelling in the production seam.
and for each lane an FNV-1a 64-bit hash over the 262,144 result words in
triple order, plus pairwise mismatch counts and the first differing triples
as hex. `tools/gemm_seam_probe_reference.py` recomputes the three candidate
semantics EXACTLY on the host (rational arithmetic, RN-even to binary32) and
names which one each lane's hash equals: round-then-flush (the contract),
flush-before-round, or no flush. The device never sees the reference.

Triple order: `i` in `[0, 64^3)`, a = words[i mod 64], b = words[(i div 64)
mod 64], acc = words[i div 4096], all three flushed before the seam (the
production path flushes operands at staging, 5a/5b, and the accumulator is
flushed by the previous step). Same words, same order, same flushes as the
NVIDIA gate, so its PASS on H100 and L40S is this probe's `shipped == swrtf`
line on the NVIDIA column.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast
from std.sys import llvm_intrinsic
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_AMD, column_name
from checks.rtf_seam import rtf_fix, RTF_REPAIR
from gemm.checks.gemm_identical import _tuned_step, _ftz_class, TUNED_HW_FTZ_FMA, TUNED_CLASS_FLUSH
from transformer.impl.llama.modeling_llama import _upload, _download, _zeros

comptime LANES = 6

#: DEVIATION 2701, extended 2026-09-17 (lane `lane/gemm-next`, brief section
#: 14.5 point 2 and section 20). A FIFTH lane, in its own kernel and its own
#: buffer, asks the one question that decides whether AMD's eight-instruction
#: seam can become one: **when the wave's MODE register is told to flush f32
#: denormals, does AMD's FMA flush BEFORE or AFTER rounding?** If after
#: (`rtf`), a single `v_fma_f32` computes the contract and the seven `ftz`
#: instructions in the loop go away. If before (`fbr`, which is what Apple's
#: FMA does), the arm is WRONG at the 315 boundary triples and is a defect.
#:
#: It is a SEPARATE KERNEL on purpose. `s_setreg` changes the mode for the
#: rest of the wave, so setting it inside `seam_kernel` would silently change
#: the four lanes computed after it. Nothing here touches those four.
comptime MODE_LANE = TARGET_COLUMN == COLUMN_AMD


def mode_seam_kernel(
    results: MutPointer[Float32, MutAnyOrigin],
    words: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
):
    comptime if MODE_LANE:
        # hwreg(HW_REG_MODE = 1, offset = 4, width = 2), the f32 FP_DENORM
        # field: 1 | (4 << 6) | ((2 - 1) << 11) = 2305 = 0x0901. Value 0 is
        # "flush f32 denormal inputs and outputs". Read back from the emitted
        # gfx942 GCN as `s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0`
        # (bench/results/e1g/2026-09-17_163900-apple-m4-amd-seam-instruction-count).
        # `llvm.amdgcn.s.setreg.imm32.b32` is the ISA mnemonic, not an LLVM
        # intrinsic name, and is rejected by name.
        llvm_intrinsic["llvm.amdgcn.s.setreg", NoneType](Int32(0x0901), Int32(0))
    var count = Int(count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= count * count * count:
        return
    var a = ftz(words.unsafe_load(i % count))
    var b = ftz(words.unsafe_load((i // count) % count))
    var acc = ftz(words.unsafe_load(i // (count * count)))
    # The NATIVE FMA ALONE. On a column where the mode was not set (every
    # column but AMD) this is the `fma` lane again, which is how the line
    # stays readable everywhere and is why the printed name says so.
    results.unsafe_store(i, identical_mul_add(a, b, acc))


@always_inline
def _native_ftz_fma(a: Float32, b: Float32, c: Float32) -> Float32:
    comptime if TUNED_HW_FTZ_FMA:
        return llvm_intrinsic[
            "llvm.nvvm.fma.rn.ftz.f", Float32, has_side_effect=False
        ](a, b, c)
    return identical_mul_add(a, b, c)


def seam_kernel(
    results: MutPointer[Float32, MutAnyOrigin],
    words: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
):
    var count = Int(count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= count * count * count:
        return
    var a = ftz(words.unsafe_load(i % count))
    var b = ftz(words.unsafe_load((i // count) % count))
    var acc = ftz(words.unsafe_load(i // (count * count)))
    results.unsafe_store(LANES * i + 0, _tuned_step(a, b, acc))
    results.unsafe_store(LANES * i + 1, identical_mul_add(a, b, acc))
    results.unsafe_store(LANES * i + 2, _native_ftz_fma(a, b, acc))
    results.unsafe_store(LANES * i + 3, ftz(identical_mul_add(a, b, acc)))
    var class_value = _ftz_class(identical_mul_add(a, b, acc))
    comptime if is_defined["MOJOLEARN_CLASS_PROBE_SABOTAGE"]():
        if i == 400:
            class_value = bitcast[DType.float32](bitcast[DType.uint32](class_value) ^ UInt32(1))
    results.unsafe_store(LANES * i + 4, class_value)
    var native = identical_mul_add(a, b, acc)
    results.unsafe_store(LANES * i + 5, rtf_fix(a, b, acc, native))


def _hex(w: UInt32) -> String:
    var digits: List[String] = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"
    ]
    var s = String("")
    for k in range(8):
        var nib = Int((w >> UInt32(28 - 4 * k)) & UInt32(0xF))
        s += digits[nib]
    return s


def _fnv1a(actual: List[Float32], n: Int, lane: Int, stride: Int = LANES) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(n):
        var w = bitcast[DType.uint32](actual[stride * i + lane])
        for k in range(4):
            h = h ^ UInt64((w >> UInt32(8 * k)) & UInt32(0xFF))
            h = h * UInt64(0x100000001B3)
    return h


def _hex64(h: UInt64) -> String:
    return _hex(UInt32(h >> UInt64(32))) + _hex(UInt32(h & UInt64(0xFFFFFFFF)))


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "gemm_seam_probe: build with -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    var words: List[UInt32] = [
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x00800001, 0x80800001,
        0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
        0x3f800001, 0xbf800001, 0x3f7fffff, 0xbf7fffff,
        0x7f7fffff, 0xff7fffff, 0x4b800001, 0xcb800001,
    ]
    var seed = UInt32(0x9A718BCD)
    for _ in range(42):
        seed = seed * UInt32(1664525) + UInt32(1013904223)
        words.append(seed & UInt32(0xFEFFFFFF))
    var values = List[Float32]()
    var wline = String("SEAM_WORDS")
    for i in range(len(words)):
        values.append(bitcast[DType.float32](words[i]))
        wline += " " + _hex(words[i])
    print("SEAM_PROBE column=" + column_name(TARGET_COLUMN) + " hwftz=" + String(TUNED_HW_FTZ_FMA)
          + " class_shipped=" + String(TUNED_CLASS_FLUSH)
          + " rtf_repair=" + String(RTF_REPAIR)
          + " lanes=shipped,fma,hwftz,swrtf,class,nativefix")
    print(wline)
    var ctx = DeviceContext()
    var inputs = _upload(ctx, values)
    var count = len(values)
    var n = count * count * count
    var result = _zeros(ctx, LANES * n)
    ctx.enqueue_function[seam_kernel](
        result.unsafe_ptr(), inputs.unsafe_ptr(), Int32(count),
        grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()
    var actual = _download(ctx, result, LANES * n)
    print("SEAM_N " + String(n))
    var names: List[String] = ["shipped", "fma", "hwftz", "swrtf", "class", "nativefix"]
    for lane in range(LANES):
        print("SEAM_HASH lane=" + names[lane] + " fnv1a64=" + _hex64(_fnv1a(actual, n, lane)))

    # The closed wave-mode experiment is deliberately not launched here.
    # The literal boundary triple: a=0x3f7fffff (index 16), b=0x00800000 (6), acc=+0 (0).
    var boundary = 16 + count * 6
    var bl = String("SEAM_BOUNDARY a=3f7fffff b=00800000 acc=00000000")
    for lane in range(LANES):
        bl += " " + names[lane] + "=" + _hex(bitcast[DType.uint32](actual[LANES * boundary + lane]))
    print(bl)
    # Pairwise mismatches, the first 24 of each as hex triples.
    var pairs_a: List[Int] = [0, 1, 0, 1, 0, 0]
    var pairs_b: List[Int] = [3, 2, 1, 3, 4, 5]
    for p in range(len(pairs_a)):
        var la = pairs_a[p]
        var lb = pairs_b[p]
        var diff = 0
        var shown = 0
        for i in range(n):
            var wa = bitcast[DType.uint32](actual[LANES * i + la])
            var wb = bitcast[DType.uint32](actual[LANES * i + lb])
            if wa != wb:
                if shown < 24:
                    print(
                        "SEAM_DIFF " + names[la] + "/" + names[lb] + " i=" + String(i)
                        + " a=" + _hex(words[i % count]) + " b=" + _hex(words[(i // count) % count])
                        + " acc=" + _hex(words[i // (count * count)])
                        + " " + names[la] + "=" + _hex(wa) + " " + names[lb] + "=" + _hex(wb)
                    )
                    shown += 1
                diff += 1
        print("SEAM_MISMATCH " + names[la] + "/" + names[lb] + " count=" + String(diff))
    print("SEAM_DONE")
