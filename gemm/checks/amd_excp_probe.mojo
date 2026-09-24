# SPDX-License-Identifier: Apache-2.0
"""lane/amd-step-time, 2026-09-24: does the gfx942 wave's sticky exception
status (TRAPSTS.EXCP) record every FMA that CONSUMES a subnormal, so that a
GEMM leaf can run bare FMAs and still prove it computed the contract's
round-then-flush chain?

THE ARGUMENT BEING TESTED. The contract step is `ftz(fma(a, b, acc))` with
`a`, `b` already flushed and `acc` the previous flushed step. A bare
`fma(a, b, acc)` chain differs from it only after some step's rounded result
is subnormal (a normal or zero result is its own flush). Such a result is
either the leaf's LAST step, whose value the leaf partial flushes anyway
(`ftz(acc)`, 5d, so the stored partial is equal), or it is the `acc` input of
the next FMA. So if no FMA of the chain consumed a subnormal input, the bare
chain's flushed partial equals the contract's, bit for bit. The hardware
bit that would say "some FMA consumed a subnormal" is TRAPSTS.EXCP[1]
(input denormal), which the ISA documents as sticky per wave and accumulated
whether or not the exception is enabled. This probe measures, on the device,
whether that is true; it changes no shipped line.

WHAT IT PRINTS
  EXCP_CASE name=... result=<hex> excp=<9 bits hex>   one line per case
  EXCP_CHAIN lane=scalar|packed triples=N flagged=F clear_mismatch=X
      over the 262,144 triples of gemm_seam_probe's word set, each followed by
      a second step: the bare two-step chain, flushed at the end, against the
      contract's two-step chain. `clear_mismatch` counts triples whose wave
      flag was CLEAR and whose bits DIFFER: the argument needs it to be 0.
  EXCP_DONE
One active lane per wave (lane 0 of a 64-thread block), so the wave's flag
belongs to one triple.

Build (on an AMD box, column amd):
  pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_TARGET_COLUMN=amd \
      -I . gemm/checks/amd_excp_probe.mojo -o /tmp/amd_excp_probe
"""
from std.gpu import block_idx, thread_idx
from std.memory import bitcast
from std.sys import llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.math import fma
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_AMD, column_name
from gemm.checks.gemm_identical import _tuned_step, _ftz_class
from transformer.impl.llama.modeling_llama import _upload, _download, _zeros

#: hwreg(HW_REG_TRAPSTS = 3, offset 0, width 9): 3 | (0 << 6) | (8 << 11).
comptime TRAPSTS_EXCP = 3 | (8 << 11)


@always_inline
def excp_clear():
    comptime if TARGET_COLUMN == COLUMN_AMD:
        llvm_intrinsic["llvm.amdgcn.s.setreg", NoneType](Int32(TRAPSTS_EXCP), Int32(0))


@always_inline
def excp_clear_dep(x: Float32) -> Float32:
    """Clear the EXCP bits and return `x` through the same asm statement, so
    every FP use of the result is ordered after the clear."""
    return inlined_assembly[
        "s_setreg_imm32_b32 hwreg(HW_REG_TRAPSTS, 0, 9), 0\n\ts_nop 7\n\tv_mov_b32 $0, $1",
        Float32, constraints="=v,v", has_side_effect=True,
    ](x)


@always_inline
def excp_read_dep(x: Float32) -> Int32:
    """Read TRAPSTS (all 32 bits) after `x` is computed (x is an input)."""
    return inlined_assembly[
        "s_nop 7\n\ts_nop 7\n\ts_getreg_b32 $0, hwreg(HW_REG_TRAPSTS)\n\t; $1",
        Int32, constraints="=s,v", has_side_effect=True,
    ](x)


@always_inline
def mode_read_dep(x: Float32) -> Int32:
    return inlined_assembly[
        "s_getreg_b32 $0, hwreg(HW_REG_MODE)\n\t; $1",
        Int32, constraints="=s,v", has_side_effect=True,
    ](x)


@always_inline
def excp_read() -> Int32:
    comptime if TARGET_COLUMN == COLUMN_AMD:
        return llvm_intrinsic["llvm.amdgcn.s.getreg", Int32](Int32(TRAPSTS_EXCP))
    return Int32(-1)


def case_kernel(
    vals: MutPointer[Float32, MutAnyOrigin],
    bits: MutPointer[Int32, MutAnyOrigin],
    words: MutPointer[Float32, MutAnyOrigin],
    n_cases: Int32,
):
    """Case `c` = block `c`: words[3c], words[3c+1], words[3c+2] = a, b, acc
    (UNflushed here on purpose). Mode by `c mod 6`: 0 scalar fma, 1 packed
    fma (both halves the same triple), 2 software ftz of acc, 3 class flush
    of acc, 4 the shipped seam `_tuned_step`, 5 nothing (control)."""
    var c = Int(block_idx.x)
    if c >= Int(n_cases) or thread_idx.x != 0:
        return
    var a = words.unsafe_load(3 * c)
    var b = words.unsafe_load(3 * c + 1)
    var acc = words.unsafe_load(3 * c + 2)
    var mode = c % 6
    acc = excp_clear_dep(acc)
    a = excp_clear_dep(a)
    var r = Float32(0.0)
    if mode == 0:
        r = identical_mul_add(a, b, acc)
    elif mode == 1:
        var pa = SIMD[DType.float32, 2](a, a)
        var pb = SIMD[DType.float32, 2](b, b)
        var pc = SIMD[DType.float32, 2](acc, acc)
        var pr = fma(pa, pb, pc)
        r = pr[0]
        vals.unsafe_store(2 * c + 1, pr[1])
    elif mode == 2:
        r = ftz(acc)
    elif mode == 3:
        r = _ftz_class(acc)
    elif mode == 4:
        r = _tuned_step(a, b, acc)
    else:
        r = acc
    vals.unsafe_store(2 * c, r)
    bits.unsafe_store(c, excp_read_dep(r))
    if c == 0:
        bits.unsafe_store(n_cases, mode_read_dep(r))


def chain_kernel(
    outp: MutPointer[Float32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    words: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
    packed: Int32,
):
    """Triple `i` = block `i`, lane 0 only. Bare chain x1 = fma(a, b, acc),
    x2 = fma(a2, b2, x1), stored flushed; then the wave's EXCP; then the
    contract chain y2 = step(a2, b2, step(a, b, acc)) stored beside it."""
    var count = Int(count_in)
    var i = Int(block_idx.x)
    if i >= count * count * count or thread_idx.x != 0:
        return
    var a = ftz(words.unsafe_load(i % count))
    var b = ftz(words.unsafe_load((i // count) % count))
    var acc = ftz(words.unsafe_load(i // (count * count)))
    var a2 = ftz(words.unsafe_load((i * 7 + 3) % count))
    var b2 = ftz(words.unsafe_load((i * 13 + 5) % count))
    acc = excp_clear_dep(acc)
    a = excp_clear_dep(a)
    var x2 = Float32(0.0)
    if packed != 0:
        var p1 = fma(SIMD[DType.float32, 2](a, b), SIMD[DType.float32, 2](b, a), SIMD[DType.float32, 2](acc, acc))
        var p2 = fma(SIMD[DType.float32, 2](a2, b2), SIMD[DType.float32, 2](b2, a2), p1)
        x2 = p2[0]
        outp.unsafe_store(3 * i + 2, p2[1])
    else:
        var x1 = identical_mul_add(a, b, acc)
        x2 = identical_mul_add(a2, b2, x1)
    outp.unsafe_store(3 * i, x2)
    flags.unsafe_store(i, excp_read_dep(x2))
    var y2 = _tuned_step(a2, b2, _tuned_step(a, b, acc))
    outp.unsafe_store(3 * i + 1, y2)


def _hex(w: UInt32) -> String:
    var digits: List[String] = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"
    ]
    var s = String("")
    for k in range(8):
        var nib = Int((w >> UInt32(28 - 4 * k)) & UInt32(0xF))
        s += digits[nib]
    return s


def _ftz_word(w: UInt32) -> UInt32:
    if (w & UInt32(0x7F800000)) == UInt32(0) and (w & UInt32(0x007FFFFF)) != UInt32(0):
        return w & UInt32(0x80000000)
    return w


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "amd_excp_probe: build with -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    print("EXCP_PROBE column=" + column_name(TARGET_COLUMN))
    var ctx = DeviceContext()

    # ---- the single cases -------------------------------------------------
    var names: List[String] = [
        "normal", "sub_acc", "sub_a", "sub_result", "sub_result_exact", "boundary", "zero_acc", "huge",
    ]
    var triples: List[UInt32] = [
        0x3f800000, 0x40000000, 0x3f800000,  # 1*2+1
        0x3f800000, 0x3f800000, 0x00000001,  # acc subnormal
        0x00400000, 0x3f800000, 0x3f800000,  # a subnormal
        0x1c800001, 0x21800001, 0x00000000,  # product ~2^-130, inexact subnormal
        0x1c800000, 0x21800000, 0x00000000,  # product 2^-130 exactly
        0x3f7fffff, 0x00800000, 0x00000000,  # the rtf/fbr boundary triple
        0x3f800000, 0x3f800000, 0x00000000,  # acc +0
        0x7f000000, 0x7f000000, 0x00000000,  # overflow
    ]
    var n_cases = len(names) * 6
    var cw = List[Float32]()
    for t in range(len(names)):
        for _m in range(6):
            cw.append(bitcast[DType.float32](triples[3 * t]))
            cw.append(bitcast[DType.float32](triples[3 * t + 1]))
            cw.append(bitcast[DType.float32](triples[3 * t + 2]))
    var cin = _upload(ctx, cw)
    var cvals = _zeros(ctx, 2 * n_cases)
    var cbits = ctx.enqueue_create_buffer[DType.int32](n_cases + 1)
    ctx.enqueue_function[case_kernel](
        cvals.unsafe_ptr(), cbits.unsafe_ptr(), cin.unsafe_ptr(), Int32(n_cases),
        grid_dim=(n_cases, 1, 1), block_dim=(64, 1, 1),
    )
    ctx.synchronize()
    var hv = _download(ctx, cvals, 2 * n_cases)
    var hb = ctx.enqueue_create_host_buffer[DType.int32](n_cases + 1)
    ctx.enqueue_copy(dst_buf=hb, src_buf=cbits)
    ctx.synchronize()
    print("EXCP_MODE_REG " + _hex(UInt32(hb.unsafe_ptr()[n_cases])))
    var modes: List[String] = ["fma", "pk_fma", "sw_ftz", "class_ftz", "shipped_step", "none"]
    for c in range(n_cases):
        var line = String("EXCP_CASE name=") + names[c // 6] + " mode=" + modes[c % 6]
        line += " result=" + _hex(bitcast[DType.uint32](hv[2 * c]))
        if c % 6 == 1:
            line += " result_hi=" + _hex(bitcast[DType.uint32](hv[2 * c + 1]))
        line += " excp=" + _hex(UInt32(hb.unsafe_ptr()[c]))
        print(line)

    # ---- the chains over the seam probe's words ----------------------------
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
    # Tiny normals, so products and cancellations land near 2^-126 often.
    var tiny: List[UInt32] = [0x1f000000, 0x9f000001, 0x20800003, 0xa0ffffff, 0x1f7fffff, 0x21000000]
    for t in tiny:
        words.append(t)
    var values = List[Float32]()
    for w in words:
        values.append(bitcast[DType.float32](w))
    var count = len(values)
    var n = count * count * count
    var inputs = _upload(ctx, values)
    for packed in range(2):
        var outb = _zeros(ctx, 3 * n)
        var flg = ctx.enqueue_create_buffer[DType.int32](n)
        ctx.enqueue_function[chain_kernel](
            outb.unsafe_ptr(), flg.unsafe_ptr(), inputs.unsafe_ptr(), Int32(count), Int32(packed),
            grid_dim=(n, 1, 1), block_dim=(64, 1, 1),
        )
        ctx.synchronize()
        var ho = _download(ctx, outb, 3 * n)
        var hf = ctx.enqueue_create_host_buffer[DType.int32](n)
        ctx.enqueue_copy(dst_buf=hf, src_buf=flg)
        ctx.synchronize()
        var flagged = 0
        var clear_mismatch = 0
        var flagged_mismatch = 0
        var pk_half_mismatch = 0
        var shown = 0
        for i in range(n):
            var f = UInt32(hf.unsafe_ptr()[i])
            var set_ = (f & UInt32(0x2)) != UInt32(0)  # input denormal
            var x = _ftz_word(bitcast[DType.uint32](ho[3 * i]))
            var y = bitcast[DType.uint32](ho[3 * i + 1])
            if packed == 1 and bitcast[DType.uint32](ho[3 * i]) != bitcast[DType.uint32](ho[3 * i + 2]):
                # a*b and b*a are the same exact product; the halves must agree
                pk_half_mismatch += 1
            if set_:
                flagged += 1
                if x != y:
                    flagged_mismatch += 1
            elif x != y:
                clear_mismatch += 1
                if shown < 16:
                    print("EXCP_CLEAR_MISMATCH i=" + String(i) + " fast=" + _hex(x) + " exact=" + _hex(y)
                          + " excp=" + _hex(f))
                    shown += 1
        print("EXCP_CHAIN lane=" + ("packed" if packed == 1 else "scalar") + " triples=" + String(n)
              + " flagged=" + String(flagged) + " flagged_mismatch=" + String(flagged_mismatch)
              + " clear_mismatch=" + String(clear_mismatch)
              + " packed_half_mismatch=" + String(pk_half_mismatch))
    print("EXCP_DONE")
