# SPDX-License-Identifier: Apache-2.0
"""Require round-then-flush at the register distance tile's exact FMA seam.

The first four results round to the smallest normal magnitude. NVIDIA's
fma.rn.ftz instead produces zero, so this is a necessary gate even when all
ordinary distance fixtures agree (their epilogues can hide tiny differences).

lane fix-nv-mul-rn-via-fma (2026-11-14): on an NVIDIA build the gate also
compares the schedulable spellings the kernels now use against the PTX asm
they replace, bit for bit: `fma.rn.f32(a, b, -0.0)` against `mul.rn.f32`
over every ordered pair of 24 boundary words (zeros, smallest subnormals,
largest subnormal, smallest normal, the 1-ulp neighbours of one, the
largest finite, infinities, a NaN, scale words), `fma.rn.f64(a, b, -0.0)`
against `mul.rn.f64` over 12 float64 words, and `fma.rn.ftz.f32(x, 1, -0)`
against `mul.rn.ftz.f32 x, 1` on the 24 words. Two NaN results pass
whatever their payloads (the contract refuses payloads, row 39). It also
probes that the pinned product is not fused into the add after it:
`pinned_mul(1+2^-12, 1+2^-12) - (1+2^-11)` is +0 when the product is
rounded first and 2^-24 when it is fused. Other columns print SKIP for
this part. The full 2^32-word check of the flush is tools/probe_ftz_hw.mojo.
"""
from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.sys.info import is_nvidia_gpu
from checks.kernel_matrix import COLUMN_NVIDIA, TARGET_COLUMN
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    nv_flush_by_one_kernel,
    nv_mul_rn_f32,
    nv_mul_rn_f64,
    nv_mul_rn_ftz_f32,
    nv_mul_rn_kernel_f32,
    nv_mul_rn_kernel_f64,
    pinned_mul_f32,
)
from neighbors.checks.pinned_distance_tile import _rt_load, _rt_step


def boundary_kernel(words: MutPointer[UInt32, MutAnyOrigin], output: MutPointer[UInt32, MutAnyOrigin]):
    var row = Int(thread_idx.x)
    if row >= 8:
        return
    var a = bitcast[DType.float32](words.unsafe_load(row * 4))
    var b = bitcast[DType.float32](words.unsafe_load(row * 4 + 1))
    var c = bitcast[DType.float32](words.unsafe_load(row * 4 + 2))
    output.unsafe_store(row, bitcast[DType.uint32](_rt_step(_rt_load(a), _rt_load(b), ftz(c))))


comptime MUL_WORDS = 24
comptime MUL_PAIRS = MUL_WORDS * MUL_WORDS
comptime MUL64_WORDS = 12
comptime MUL64_PAIRS = MUL64_WORDS * MUL64_WORDS


def mul_rn_kernel(
    words: MutPointer[UInt32, MutAnyOrigin],
    words64: MutPointer[UInt64, MutAnyOrigin],
    probe: MutPointer[UInt32, MutAnyOrigin],
    got: MutPointer[UInt32, MutAnyOrigin],
    want: MutPointer[UInt32, MutAnyOrigin],
    got64: MutPointer[UInt64, MutAnyOrigin],
    want64: MutPointer[UInt64, MutAnyOrigin],
    flush_got: MutPointer[UInt32, MutAnyOrigin],
    flush_want: MutPointer[UInt32, MutAnyOrigin],
    fused: MutPointer[UInt32, MutAnyOrigin],
):
    """Thread t: pair t of the f32 words, pair t of the f64 words, word t's flush; thread 0 the fusion probe."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    comptime if is_nvidia_gpu():
        if t < MUL_PAIRS:
            var a = bitcast[DType.float32](words.unsafe_load(t // MUL_WORDS))
            var b = bitcast[DType.float32](words.unsafe_load(t % MUL_WORDS))
            got.unsafe_store(t, bitcast[DType.uint32](nv_mul_rn_kernel_f32(a, b)))
            want.unsafe_store(t, bitcast[DType.uint32](nv_mul_rn_f32(a, b)))
        if t < MUL64_PAIRS:
            var a64 = bitcast[DType.float64](words64.unsafe_load(t // MUL64_WORDS))
            var b64 = bitcast[DType.float64](words64.unsafe_load(t % MUL64_WORDS))
            got64.unsafe_store(t, bitcast[DType.uint64](nv_mul_rn_kernel_f64(a64, b64)))
            want64.unsafe_store(t, bitcast[DType.uint64](nv_mul_rn_f64(a64, b64)))
        if t < MUL_WORDS:
            var x = bitcast[DType.float32](words.unsafe_load(t))
            flush_got.unsafe_store(t, bitcast[DType.uint32](nv_flush_by_one_kernel(x)))
            flush_want.unsafe_store(t, bitcast[DType.uint32](nv_mul_rn_ftz_f32(x, Float32(1.0))))
        if t == 0:
            var p = bitcast[DType.float32](probe.unsafe_load(0))
            var c = bitcast[DType.float32](probe.unsafe_load(1))
            fused.unsafe_store(0, bitcast[DType.uint32](pinned_mul_f32(p, p) + c))


def _is_nan32(w: UInt32) -> Bool:
    return (w & UInt32(0x7F800000)) == UInt32(0x7F800000) and (w & UInt32(0x007FFFFF)) != UInt32(0)


def _is_nan64(w: UInt64) -> Bool:
    return (w & UInt64(0x7FF0000000000000)) == UInt64(0x7FF0000000000000) and (
        w & UInt64(0x000FFFFFFFFFFFFF)
    ) != UInt64(0)


def run_mul_rn_checks(ctx: DeviceContext) raises:
    """The schedulable NVIDIA product and flush against the asm they replace; see the module docstring."""
    comptime if TARGET_COLUMN != COLUMN_NVIDIA:
        print("NV MUL RN VIA FMA SKIP: not the NVIDIA column")
        return
    var words: List[UInt32] = [
        0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x3f7fffff, 0xbf7fffff, 0x3f800000, 0xbf800000,
        0x3f800001, 0x3f7ffffe, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7fc00000, 0x1f800000, 0x20000000, 0x5f800000, 0x3fc00000, 0x34000000,
    ]
    var words64: List[UInt64] = [
        0x0000000000000000, 0x8000000000000000, 0x0000000000000001, 0x000fffffffffffff,
        0x0010000000000000, 0x3fefffffffffffff, 0xbfefffffffffffff, 0x3ff0000000000001,
        0x7fefffffffffffff, 0x7ff0000000000000, 0x1ff0000000000000, 0x7ff8000000000000,
    ]
    # 1 + 2^-12 squared is 1 + 2^-11 + 2^-24, a tie that rounds to 1 + 2^-11.
    var probe_words: List[UInt32] = [0x3f800800, 0xbf801000]
    var h_words = ctx.enqueue_create_host_buffer[DType.uint32](MUL_WORDS)
    var h_words64 = ctx.enqueue_create_host_buffer[DType.uint64](MUL64_WORDS)
    var h_probe = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h_got = ctx.enqueue_create_host_buffer[DType.uint32](MUL_PAIRS)
    var h_want = ctx.enqueue_create_host_buffer[DType.uint32](MUL_PAIRS)
    var h_got64 = ctx.enqueue_create_host_buffer[DType.uint64](MUL64_PAIRS)
    var h_want64 = ctx.enqueue_create_host_buffer[DType.uint64](MUL64_PAIRS)
    var h_flush_got = ctx.enqueue_create_host_buffer[DType.uint32](MUL_WORDS)
    var h_flush_want = ctx.enqueue_create_host_buffer[DType.uint32](MUL_WORDS)
    var h_fused = ctx.enqueue_create_host_buffer[DType.uint32](1)
    ctx.synchronize()
    for i in range(MUL_WORDS):
        h_words.unsafe_ptr().unsafe_store(i, words[i])
    for i in range(MUL64_WORDS):
        h_words64.unsafe_ptr().unsafe_store(i, words64[i])
    for i in range(2):
        h_probe.unsafe_ptr().unsafe_store(i, probe_words[i])
    var d_words = ctx.enqueue_create_buffer[DType.uint32](MUL_WORDS)
    var d_words64 = ctx.enqueue_create_buffer[DType.uint64](MUL64_WORDS)
    var d_probe = ctx.enqueue_create_buffer[DType.uint32](2)
    var d_got = ctx.enqueue_create_buffer[DType.uint32](MUL_PAIRS)
    var d_want = ctx.enqueue_create_buffer[DType.uint32](MUL_PAIRS)
    var d_got64 = ctx.enqueue_create_buffer[DType.uint64](MUL64_PAIRS)
    var d_want64 = ctx.enqueue_create_buffer[DType.uint64](MUL64_PAIRS)
    var d_flush_got = ctx.enqueue_create_buffer[DType.uint32](MUL_WORDS)
    var d_flush_want = ctx.enqueue_create_buffer[DType.uint32](MUL_WORDS)
    var d_fused = ctx.enqueue_create_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_buf=d_words, src_ptr=h_words.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_words64, src_ptr=h_words64.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_probe, src_ptr=h_probe.unsafe_ptr())
    ctx.enqueue_function[mul_rn_kernel](
        d_words.unsafe_ptr(), d_words64.unsafe_ptr(), d_probe.unsafe_ptr(),
        d_got.unsafe_ptr(), d_want.unsafe_ptr(), d_got64.unsafe_ptr(), d_want64.unsafe_ptr(),
        d_flush_got.unsafe_ptr(), d_flush_want.unsafe_ptr(), d_fused.unsafe_ptr(),
        grid_dim=((MUL_PAIRS + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=h_got.unsafe_ptr(), src_buf=d_got)
    ctx.enqueue_copy(dst_ptr=h_want.unsafe_ptr(), src_buf=d_want)
    ctx.enqueue_copy(dst_ptr=h_got64.unsafe_ptr(), src_buf=d_got64)
    ctx.enqueue_copy(dst_ptr=h_want64.unsafe_ptr(), src_buf=d_want64)
    ctx.enqueue_copy(dst_ptr=h_flush_got.unsafe_ptr(), src_buf=d_flush_got)
    ctx.enqueue_copy(dst_ptr=h_flush_want.unsafe_ptr(), src_buf=d_flush_want)
    ctx.enqueue_copy(dst_ptr=h_fused.unsafe_ptr(), src_buf=d_fused)
    ctx.synchronize()
    var bad = 0
    var nan_pairs = 0
    for t in range(MUL_PAIRS):
        var g = h_got.unsafe_ptr().unsafe_load(t)
        var w = h_want.unsafe_ptr().unsafe_load(t)
        if g == w:
            continue
        if _is_nan32(g) and _is_nan32(w):
            nan_pairs += 1
            continue
        bad += 1
        print("NV_MUL_RN_F32_FAIL", hex(words[t // MUL_WORDS]), hex(words[t % MUL_WORDS]), "fma", hex(g), "mul", hex(w))
    for t in range(MUL64_PAIRS):
        var g = h_got64.unsafe_ptr().unsafe_load(t)
        var w = h_want64.unsafe_ptr().unsafe_load(t)
        if g == w:
            continue
        if _is_nan64(g) and _is_nan64(w):
            nan_pairs += 1
            continue
        bad += 1
        print("NV_MUL_RN_F64_FAIL", hex(words64[t // MUL64_WORDS]), hex(words64[t % MUL64_WORDS]), "fma", hex(g), "mul", hex(w))
    for t in range(MUL_WORDS):
        var g = h_flush_got.unsafe_ptr().unsafe_load(t)
        var w = h_flush_want.unsafe_ptr().unsafe_load(t)
        if g == w:
            continue
        if _is_nan32(g) and _is_nan32(w):
            nan_pairs += 1
            continue
        bad += 1
        print("NV_FLUSH_BY_ONE_FAIL", hex(words[t]), "fma.ftz", hex(g), "mul.ftz", hex(w))
    var fused_word = h_fused.unsafe_ptr().unsafe_load(0)
    if fused_word != UInt32(0):
        bad += 1
        print("NV_PINNED_MUL_FUSED_FAIL", hex(fused_word), "want 0x0 (0x33800000 = fused)")
    if bad != 0:
        raise Error("schedulable NVIDIA product/flush differs from the PTX asm")
    print(
        "NV MUL RN VIA FMA PASS", "f32 pairs", MUL_PAIRS, "f64 pairs", MUL64_PAIRS,
        "flush words", MUL_WORDS, "nan pairs (payload not judged)", nan_pairs, "fusion probe clean",
    )
    _ = h_words^
    _ = h_words64^
    _ = h_probe^
    _ = d_words^
    _ = d_words64^
    _ = d_probe^
    _ = d_got^
    _ = d_want^
    _ = d_got64^
    _ = d_want64^
    _ = d_flush_got^
    _ = d_flush_want^
    _ = d_fused^
    _ = h_got^
    _ = h_want^
    _ = h_got64^
    _ = h_want64^
    _ = h_flush_got^
    _ = h_flush_want^
    _ = h_fused^


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("FMA boundary gate requires IDENTICAL")
    var fixtures: List[UInt32] = [
        0x3f7fffff, 0x00800000, 0, 0x00800000,
        0xbf7fffff, 0x00800000, 0, 0x80800000,
        0x3f7fffff, 0x80800000, 0, 0x80800000,
        0xbf7fffff, 0x80800000, 0, 0x00800000,
        0x3f7ffffe, 0x00800000, 0, 0,
        0xbf7ffffe, 0x00800000, 0, 0x80000000,
        0x007fffff, 0x40000000, 0, 0,
        0x3f800000, 0x00800000, 0x80800000, 0,
    ]
    with DeviceContext() as ctx:
        var host = ctx.enqueue_create_host_buffer[DType.uint32](32)
        var actual = ctx.enqueue_create_host_buffer[DType.uint32](8)
        ctx.synchronize()
        for i in range(32):
            host.unsafe_ptr().unsafe_store(i, fixtures[i])
        var words = ctx.enqueue_create_buffer[DType.uint32](32)
        var output = ctx.enqueue_create_buffer[DType.uint32](8)
        ctx.enqueue_copy(dst_buf=words, src_ptr=host.unsafe_ptr())
        ctx.enqueue_function[boundary_kernel](words.unsafe_ptr(), output.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(32, 1, 1))
        ctx.enqueue_copy(dst_ptr=actual.unsafe_ptr(), src_buf=output)
        ctx.synchronize()
        for row in range(8):
            var got = actual.unsafe_ptr().unsafe_load(row)
            if got != fixtures[row * 4 + 3]:
                print("DISTANCE_FMA_BOUNDARY_FAIL", row, got, fixtures[row * 4 + 3])
                raise Error("distance FMA violated round-then-flush contract")
        print("KNN DISTANCE FMA BOUNDARY PASS", "cases", 8)
        run_mul_rn_checks(ctx)
        _ = host^
        _ = words^
