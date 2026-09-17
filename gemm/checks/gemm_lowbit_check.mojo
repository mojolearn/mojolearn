# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The low-bit profiles' gates: the seams, oracle agreement on the device,
plan agreement, batch invariance, and the sabotage that shows each can fail.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_lowbit_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_SABOTAGE=1 -I . gemm/checks/gemm_lowbit_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_HOST_SABOTAGE=1 -I . gemm/checks/gemm_lowbit_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_FORCE_FLAT=1 -I . gemm/checks/gemm_lowbit_check.mojo

The first must pass every gate; the second and third must FAIL the device
gates (the sabotage report at the end says which did); the fourth pins the
int8 dispatcher to the flat plan and must pass every gate too. Contract
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`; kernels `gemm/checks/gemm_lowbit.mojo`
and `gemm/checks/gemm_int8_mma.mojo`; answers
`gemm/host/gemm_lowbit_oracle.mojo`.

THE int8 MATRIX-UNIT GATE (DEVIATION 2910, contract L-9).
`check_int8_mma_matches_flat` runs the flat plan and the MMA plan on every
OP_NT shape of `_shape` and on the ragged shapes of `_mma_shape` (k = 17,
31, 33, 100, 1000, 4097 with m, n off the 16 and 32 tile edges) and
requires the same bits, and on the ragged shapes also the oracle's bits. It
runs only on a column whose `lib_int8_matrix_unit_for` row is True; on any
other column main prints that it did not run, which is not a pass.
    RUN OWED: pixi run check-gemm-lowbit                (H100 sm_90a; MI300X/MI325X gfx942)
    RUN OWED: pixi run check-gemm-lowbit-sabotage       (must FAIL on both boxes, naming check_int8_mma_matches_flat)
    RUN OWED: pixi run check-gemm-lowbit-host-sabotage  (must FAIL on both boxes)
    RUN OWED: pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_FORCE_FLAT=1 -I . gemm/checks/gemm_lowbit_check.mojo
`tools/lowbit_mma_leg.sh` is the body that runs the four on a rented box.

MAIN RUNS EVERY GATE AND REPORTS EVERY VERDICT before it raises, as
`gemm_device_check.mojo` does and for the same reason: under a sabotage
build the useful evidence is which gates a defect reaches.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.sys import has_accelerator

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    bf16_bits_to_f32,
    f32_round_half_even,
    f32_to_bf16_bits_rne,
    i32_to_f32_pinned,
    numeric_mode_name,
    quantize_int8_value,
)
from checks.kernel_matrix import TARGET_COLUMN, column_name, lib_int8_matrix_unit_for
from gemm.checks.gemm_int8_mma import identical_gemm_int8_mma_into
from gemm.checks.gemm_lowbit import (
    BF16W_FUSED_MAX_CELLS,
    LowbitWorkspace,
    bf16_narrow,
    bf16_widen,
    dequantize_rows_int8_device,
    identical_gemm_bf16w_fused_into,
    identical_gemm_bf16w_into,
    identical_gemm_bf16w_widen_into,
    identical_gemm_int8_flat_into,
    identical_gemm_int8_into,
    int8_plan_dispatch_name,
    lowbit_sabotage_name,
    quantize_rows_int8_device,
)
from gemm.host.gemm_lowbit_oracle import (
    Int8Rows,
    dequantize_rows_int8,
    gemm_bf16_oracle,
    gemm_int8_oracle,
    narrow_bf16,
    quantize_rows_int8,
    widen_bf16,
)
from gemm.host.gemm_oracle import (
    GEMM_ORACLE_HOST_SABOTAGE,
    OP_NN,
    OP_NT,
    OP_TN,
    gemm_oracle,
    op_name,
)

comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime POISON = Float32(-987654.0)


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _show(x: Float32) -> String:
    return String(x) + "/0x" + hex(_bits(x))


def _hash64(i: Int, salt: Int) -> UInt64:
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(
        0xBF58476D1CE4E5B9
    )
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _val(i: Int, salt: Int) -> Float32:
    """`gemm_device_check.mojo::_val`: a 13-bit significand spread over eight
    binades, so every product is inexact and every order matters."""
    var h = _hash64(i, salt)
    var mant = Float32(1.0) + Float32(Int(h & UInt64(0xFFF))) / Float32(4096.0)
    var e = Int((h >> UInt64(13)) & UInt64(7)) - 4
    var scale = Float32(1.0)
    if e >= 0:
        for _ in range(e):
            scale = scale * Float32(2.0)
    else:
        for _ in range(-e):
            scale = scale * Float32(0.5)
    var v = mant * scale
    if (h >> UInt64(20)) & UInt64(1) == UInt64(1):
        return -v
    return v


def _fill(n_elems: Int, salt: Int) -> List[Float32]:
    var v = List[Float32]()
    for i in range(n_elems):
        v.append(_val(i, salt))
    return v^


def _fill_bf16(n_elems: Int, salt: Int) -> List[UInt16]:
    """bf16 bits of the same generator: a 13-bit significand does not fit
    in bf16's 7, so the narrowing rounds and the widened values are NOT the
    float32 values; that is the point of having a separate fixture."""
    var v = List[UInt16]()
    for i in range(n_elems):
        v.append(f32_to_bf16_bits_rne(_val(i, salt)))
    return v^


def _a_elems(op: Int, m: Int, n: Int, k: Int) -> Int:
    return m * k


def _b_elems(op: Int, m: Int, n: Int, k: Int) -> Int:
    return n * k


# ===========================================================================
# THE SEAMS, ON THE HOST
# ===========================================================================


def check_bf16_narrowing_is_round_to_nearest_even() raises:
    """L-2 against an explicit spelling, at the tie and off it."""
    # 1 + 2^-8 is exactly halfway between the bf16 neighbours 1 and 1+2^-7:
    # ties to even keeps 1.0 (mantissa ...0), not 1+2^-7 (mantissa ...1).
    var tie_down = Float32(1.0) + Float32(1.0) / Float32(256.0)
    if f32_to_bf16_bits_rne(tie_down) != UInt16(0x3F80):
        raise Error("tie at 1 + 2^-8 did not round to even (1.0): got 0x" + hex(UInt32(f32_to_bf16_bits_rne(tie_down))))
    # 1 + 3 * 2^-8 is halfway between 1+2^-7 (odd) and 1+2^-6 (even): up.
    var tie_up = Float32(1.0) + Float32(3.0) / Float32(256.0)
    if f32_to_bf16_bits_rne(tie_up) != UInt16(0x3F82):
        raise Error("tie at 1 + 3*2^-8 did not round to even (1 + 2^-6): got 0x" + hex(UInt32(f32_to_bf16_bits_rne(tie_up))))
    # A round-toward-zero spelling differs at both ties; that is the
    # separation proof that the fixture can tell the two apart.
    var rtz_up = UInt16(_bits(tie_up) >> UInt32(16))
    if rtz_up == f32_to_bf16_bits_rne(tie_up):
        raise Error("the tie fixture does not separate RNE from RTZ; it is not evidence")
    # Widening is the exact inverse on every NORMAL bf16 value. A bf16
    # subnormal widens to a float32 subnormal, which L-2 flushes before it
    # narrows, so it comes back as the signed zero; that is the contract,
    # and the check asserts it rather than skipping it.
    var flushed = 0
    for i in range(65536):
        var b = UInt16(i)
        var w = bf16_bits_to_f32(b)
        if w != w:
            continue
        var back = f32_to_bf16_bits_rne(w)
        if (b & UInt16(0x7F80)) == UInt16(0) and (b & UInt16(0x007F)) != UInt16(0):
            comptime if IDENTICAL_BUILD:
                if back != (b & UInt16(0x8000)):
                    raise Error("bf16 subnormal 0x" + hex(UInt32(b)) + " did not narrow to the signed zero")
                flushed += 1
            continue
        if back != b:
            raise Error("widen then narrow moved bf16 0x" + hex(UInt32(b)))
    comptime if IDENTICAL_BUILD:
        if flushed != 254:
            raise Error("expected 254 bf16 subnormals to flush, saw " + String(flushed))
    # NaN narrows to a quiet NaN, never to an infinity.
    var nan = bitcast[DType.float32](UInt32(0x7F800001))
    var nb = f32_to_bf16_bits_rne(nan)
    if (nb & UInt16(0x7F80)) != UInt16(0x7F80) or (nb & UInt16(0x007F)) == UInt16(0):
        raise Error("NaN narrowed to a non-NaN: 0x" + hex(UInt32(nb)))


def check_integer_seams_match_explicit_spellings() raises:
    """L-4 and L-5: the magic-number rounding against an integer spelling,
    and the two-part int-to-float against the backend's own conversion."""
    for i in range(-4096, 4097):
        var half = Float32(i) / Float32(2.0)
        var got = f32_round_half_even(half)
        # explicit: floor, then the tie rule
        var lo = Float32(i // 2)
        var want = lo
        if i % 2 != 0:
            # exactly halfway: pick the even of lo and lo + 1
            var lo_int = i // 2
            want = lo if lo_int % 2 == 0 else lo + Float32(1.0)
        if _bits(got) != _bits(want):
            raise Error("rne(" + _show(half) + ") = " + _show(got) + ", want " + _show(want))
    var probes: List[Int32] = [0, 1, -1, 16777215, 16777216, 16777217, -16777217, 33554435, 2147483647, -2147483647, 1234567891]
    for idx in range(len(probes)):
        var v = probes[idx]
        var got = i32_to_f32_pinned(v)
        var want = Float32(v)
        if _bits(got) != _bits(want):
            raise Error("i32_to_f32_pinned(" + String(v) + ") = " + _show(got) + " differs from the backend conversion " + _show(want))


def check_quantizer_bounds() raises:
    """L-3, L-4: every code within [-127, 127]; dequantized error within
    one step of `2^e`; an all-zero row takes exponent 0."""
    var x = _fill(64 * 96, 7)
    var qr = quantize_rows_int8(x, 64, 96)
    var y = dequantize_rows_int8(qr)
    for r in range(64):
        var step = Float32(1.0)
        var e = Int(qr.e[r])
        if e >= 0:
            for _ in range(e):
                step = step * Float32(2.0)
        else:
            for _ in range(-e):
                step = step * Float32(0.5)
        for c in range(96):
            var q = Int(qr.q[r * 96 + c])
            if q < -127 or q > 127:
                raise Error("code out of range: " + String(q))
            var err = y[r * 96 + c] - x[r * 96 + c]
            if err < Float32(0.0):
                err = -err
            # One step, not half: a value in (127.5, 128) times 2^-e rounds
            # to 128 and the symmetric clamp brings it to 127 (contract L-4).
            if err > step + step * Float32(1e-6):
                raise Error("dequantized error " + _show(err) + " exceeds a step " + _show(step))
    var zeros = List[Float32]()
    for _ in range(8):
        zeros.append(Float32(0.0))
    var zq = quantize_rows_int8(zeros, 1, 8)
    if zq.e[0] != Int32(0):
        raise Error("all-zero row took exponent " + String(zq.e[0]))


# ===========================================================================
# THE DEVICE
# ===========================================================================


def _upload_f32(ctx: DeviceContext, h: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var n = len(h)
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(len(h)):
        hb.unsafe_ptr().unsafe_store(i, h[i])
    ctx.enqueue_copy(dst_buf=d, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    _ = hb
    return d^


def _upload_u16(ctx: DeviceContext, h: List[UInt16]) raises -> DeviceBuffer[DType.uint16]:
    var n = len(h)
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.uint16](n)
    var hb = ctx.enqueue_create_host_buffer[DType.uint16](n)
    ctx.synchronize()
    for i in range(len(h)):
        hb.unsafe_ptr().unsafe_store(i, h[i])
    ctx.enqueue_copy(dst_buf=d, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    _ = hb
    return d^


def _upload_i8(ctx: DeviceContext, h: List[Int8]) raises -> DeviceBuffer[DType.int8]:
    var n = len(h)
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.int8](n)
    var hb = ctx.enqueue_create_host_buffer[DType.int8](n)
    ctx.synchronize()
    for i in range(len(h)):
        hb.unsafe_ptr().unsafe_store(i, h[i])
    ctx.enqueue_copy(dst_buf=d, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    _ = hb
    return d^


def _upload_i32(ctx: DeviceContext, h: List[Int32]) raises -> DeviceBuffer[DType.int32]:
    var n = len(h)
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    var hb = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.synchronize()
    for i in range(len(h)):
        hb.unsafe_ptr().unsafe_store(i, h[i])
    ctx.enqueue_copy(dst_buf=d, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    _ = hb
    return d^


def _poisoned(ctx: DeviceContext, count: Int) raises -> DeviceBuffer[DType.float32]:
    var h = List[Float32]()
    for _ in range(count):
        h.append(POISON)
    return _upload_f32(ctx, h)


def _download_f32(ctx: DeviceContext, mut d: DeviceBuffer[DType.float32], count: Int, tag: String) raises -> List[Float32]:
    var hb = ctx.enqueue_create_host_buffer[DType.float32](count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(count):
        var v = hb.unsafe_ptr().unsafe_load(i)
        if _bits(v) == _bits(POISON):
            raise Error("POISON SURVIVED at cell " + String(i) + " of " + tag)
        out.append(v)
    _ = hb
    return out^


def _download_u16(ctx: DeviceContext, mut d: DeviceBuffer[DType.uint16], count: Int) raises -> List[UInt16]:
    var hb = ctx.enqueue_create_host_buffer[DType.uint16](count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    var out = List[UInt16]()
    for i in range(count):
        out.append(hb.unsafe_ptr().unsafe_load(i))
    _ = hb
    return out^


def _download_i8(ctx: DeviceContext, mut d: DeviceBuffer[DType.int8], count: Int) raises -> List[Int8]:
    var hb = ctx.enqueue_create_host_buffer[DType.int8](count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    var out = List[Int8]()
    for i in range(count):
        out.append(hb.unsafe_ptr().unsafe_load(i))
    _ = hb
    return out^


def _download_i32(ctx: DeviceContext, mut d: DeviceBuffer[DType.int32], count: Int) raises -> List[Int32]:
    var hb = ctx.enqueue_create_host_buffer[DType.int32](count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    var out = List[Int32]()
    for i in range(count):
        out.append(hb.unsafe_ptr().unsafe_load(i))
    _ = hb
    return out^


def _first_diff(got: List[Float32], want: List[Float32], tag: String) raises:
    var bad = 0
    var first = -1
    for i in range(len(want)):
        if _bits(got[i]) != _bits(want[i]):
            bad += 1
            if first < 0:
                first = i
    if bad > 0:
        raise Error(
            tag + ": " + String(bad) + " of " + String(len(want))
            + " cells differ; first at " + String(first) + " got "
            + _show(got[first]) + " want " + _show(want[first])
        )


#: The shapes: decode rows (m = 1, 2), odd extents, k across the leaf rule's
#: boundaries (127, 128, 129, 256, 1000), and one output above
#: BF16W_FUSED_MAX_CELLS so the dispatcher's widen plan runs.
comptime SHAPE_COUNT = 9


def _shape(i: Int) -> Tuple[Int, Int, Int, Int]:
    if i == 0:
        return (1, 32, 32, OP_NT)
    if i == 1:
        return (1, 64, 128, OP_NT)
    if i == 2:
        return (2, 96, 127, OP_NT)
    if i == 3:
        return (7, 33, 129, OP_NT)
    if i == 4:
        return (16, 64, 1000, OP_NT)
    if i == 5:
        return (5, 9, 256, OP_NN)
    if i == 6:
        return (9, 5, 300, OP_TN)
    if i == 7:
        return (129, 129, 256, OP_NT)
    return (1, 4096, 512, OP_NT)


def _run_bf16(
    ctx: DeviceContext, ha: List[Float32], hb: List[UInt16], m: Int, n: Int, k: Int, op: Int, plan: Int, tag: String
) raises -> List[Float32]:
    """plan 0 = dispatcher, 1 = fused, 2 = widen."""
    var da = _upload_f32(ctx, ha)
    var db = _upload_u16(ctx, hb)
    var dc = _poisoned(ctx, m * n)
    var work = LowbitWorkspace(ctx)
    if plan == 1:
        identical_gemm_bf16w_fused_into(ctx, dc, da, db, m, n, k, op)
    elif plan == 2:
        identical_gemm_bf16w_widen_into(ctx, dc, da, db, work, m, n, k, op)
    else:
        identical_gemm_bf16w_into(ctx, dc, da, db, work, m, n, k, op)
    ctx.synchronize()
    var out = _download_f32(ctx, dc, m * n, tag)
    _ = da
    _ = db
    _ = dc
    _ = work^
    return out^


def check_bf16_device_matches_oracle(ctx: DeviceContext) raises:
    """GATE: every shape, the dispatcher's answer is `gemm_bf16_oracle`'s."""
    for s in range(SHAPE_COUNT):
        var sh = _shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var op = sh[3]
        var ha = _fill(_a_elems(op, m, n, k), 11 + s)
        var hb = _fill_bf16(_b_elems(op, m, n, k), 23 + s)
        var want = gemm_bf16_oracle(ha, hb, op, m, n, k)
        var tag = "bf16 " + op_name(op) + " " + String(m) + "x" + String(n) + "x" + String(k)
        var got = _run_bf16(ctx, ha, hb, m, n, k, op, 0, tag)
        _first_diff(got, want, tag)
        print("   ok " + tag + "  c[0]=" + _show(got[0]))


def check_bf16_plans_agree(ctx: DeviceContext) raises:
    """GATE (L-8): the fused plan and the widen plan return the same bits on
    every shape, whichever the dispatcher would have picked."""
    for s in range(SHAPE_COUNT):
        var sh = _shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var op = sh[3]
        var ha = _fill(_a_elems(op, m, n, k), 31 + s)
        var hb = _fill_bf16(_b_elems(op, m, n, k), 47 + s)
        var tag = "plans " + op_name(op) + " " + String(m) + "x" + String(n) + "x" + String(k)
        var fused = _run_bf16(ctx, ha, hb, m, n, k, op, 1, tag + " fused")
        var widen = _run_bf16(ctx, ha, hb, m, n, k, op, 2, tag + " widen")
        _first_diff(fused, widen, tag)
    print("   ok fused == widen on " + String(SHAPE_COUNT) + " shapes; fused cap " + String(BF16W_FUSED_MAX_CELLS) + " cells")


def check_bf16_is_batch_invariant(ctx: DeviceContext) raises:
    """GATE: row `i` of an `m`-row call equals the one-row call on row `i`
    alone, at the decode shape. The fused plan is one thread per cell and
    the widen plan is the fp32 profile's, so a failure here is a defect and
    not a measurement."""
    var m = 6
    var n = 96
    var k = 257
    var ha = _fill(m * k, 61)
    var hb = _fill_bf16(n * k, 67)
    var batch = _run_bf16(ctx, ha, hb, m, n, k, OP_NT, 0, "batch")
    for i in range(m):
        var row = List[Float32]()
        for p in range(k):
            row.append(ha[i * k + p])
        var one = _run_bf16(ctx, row, hb, 1, n, k, OP_NT, 0, "row " + String(i))
        for j in range(n):
            if _bits(one[j]) != _bits(batch[i * n + j]):
                raise Error("row " + String(i) + " col " + String(j) + " differs between the batch and the single-row call: " + _show(batch[i * n + j]) + " vs " + _show(one[j]))
    print("   ok " + String(m) + " rows agree with their single-row calls")


def check_bf16_device_conversions_match_host(ctx: DeviceContext) raises:
    """GATE: the device widen and narrow kernels are the host functions."""
    var count = 4099
    var x = _fill(count, 71)
    var dx = _upload_f32(ctx, x)
    var dn = ctx.enqueue_create_buffer[DType.uint16](count)
    var dw = _poisoned(ctx, count)
    bf16_narrow(ctx, dn, dx, count)
    bf16_widen(ctx, dw, dn, count)
    ctx.synchronize()
    var narrowed = _download_u16(ctx, dn, count)
    var widened = _download_f32(ctx, dw, count, "widen")
    var want_n = narrow_bf16(x)
    var want_w = widen_bf16(want_n)
    for i in range(count):
        if narrowed[i] != want_n[i]:
            raise Error("device narrow differs at " + String(i) + ": 0x" + hex(UInt32(narrowed[i])) + " vs 0x" + hex(UInt32(want_n[i])))
        if _bits(widened[i]) != _bits(want_w[i]):
            raise Error("device widen differs at " + String(i))
    _ = dx
    _ = dn
    _ = dw
    print("   ok narrow and widen agree with the host on " + String(count) + " values")


def check_int8_device_matches_oracle(ctx: DeviceContext) raises:
    """GATE: the device quantizer and the device product are the host
    oracle's, bit for bit, at every OP_NT shape."""
    for s in range(SHAPE_COUNT):
        var sh = _shape(s)
        if sh[3] != OP_NT:
            continue
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var ha = _fill(m * k, 83 + s)
        var hb = _fill(n * k, 97 + s)
        var qa = quantize_rows_int8(ha, m, k)
        var qb = quantize_rows_int8(hb, n, k)
        var want = gemm_int8_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
        var tag = "int8 " + String(m) + "x" + String(n) + "x" + String(k)
        var da = _upload_f32(ctx, ha)
        var db = _upload_f32(ctx, hb)
        var dqa = ctx.enqueue_create_buffer[DType.int8](m * k)
        var dea = ctx.enqueue_create_buffer[DType.int32](m)
        var dqb = ctx.enqueue_create_buffer[DType.int8](n * k)
        var deb = ctx.enqueue_create_buffer[DType.int32](n)
        var dc = _poisoned(ctx, m * n)
        quantize_rows_int8_device(ctx, dqa, dea, da, m, k)
        quantize_rows_int8_device(ctx, dqb, deb, db, n, k)
        identical_gemm_int8_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
        ctx.synchronize()
        var got_qa = _download_i8(ctx, dqa, m * k)
        var got_ea = _download_i32(ctx, dea, m)
        for i in range(m * k):
            if got_qa[i] != qa.q[i]:
                raise Error(tag + ": device code differs at " + String(i) + ": " + String(Int(got_qa[i])) + " vs " + String(Int(qa.q[i])))
        for i in range(m):
            if got_ea[i] != qa.e[i]:
                raise Error(tag + ": device exponent differs at row " + String(i))
        var got = _download_f32(ctx, dc, m * n, tag)
        _first_diff(got, want, tag)
        # the dequantized weight image, the int8w block formats' operand
        var dy = _poisoned(ctx, n * k)
        dequantize_rows_int8_device(ctx, dy, dqb, deb, n, k)
        ctx.synchronize()
        var got_y = _download_f32(ctx, dy, n * k, tag + " dequant")
        _first_diff(got_y, dequantize_rows_int8(qb), tag + " dequant")
        _ = da
        _ = db
        _ = dqa
        _ = dea
        _ = dqb
        _ = deb
        _ = dc
        _ = dy
        print("   ok " + tag + "  c[0]=" + _show(got[0]))


#: The ragged shapes of the matrix-unit gate: every k off the unit's k-tile
#: of 32 (17, 31, 33, 100, 1000, 4097) and m, n off the 16-wide warp tile
#: and the 32-wide block tile, so the zero-code padding and the store mask
#: are exercised on both edges; one decode row; one k that is a multiple of
#: 32 with a single warp tile (2 x 16 x 32) so the aligned vector load path
#: runs.
comptime MMA_SHAPE_COUNT = 8


def _mma_shape(i: Int) -> Tuple[Int, Int, Int]:
    if i == 0:
        return (3, 5, 17)
    if i == 1:
        return (17, 33, 31)
    if i == 2:
        return (33, 17, 33)
    if i == 3:
        return (1, 47, 100)
    if i == 4:
        return (50, 70, 1000)
    if i == 5:
        return (13, 21, 4097)
    if i == 6:
        return (2, 16, 32)
    return (100, 3, 64)


def _run_int8_plans(
    ctx: DeviceContext, qa: Int8Rows, qb: Int8Rows, m: Int, n: Int, k: Int, tag: String
) raises -> Tuple[List[Float32], List[Float32]]:
    """Both int8 plans on the same codes: (flat, mma)."""
    var dqa = _upload_i8(ctx, qa.q)
    var dea = _upload_i32(ctx, qa.e)
    var dqb = _upload_i8(ctx, qb.q)
    var deb = _upload_i32(ctx, qb.e)
    var dflat = _poisoned(ctx, m * n)
    var dmma = _poisoned(ctx, m * n)
    identical_gemm_int8_flat_into(ctx, dflat, dqa, dea, dqb, deb, m, n, k)
    identical_gemm_int8_mma_into(ctx, dmma, dqa, dea, dqb, deb, m, n, k)
    ctx.synchronize()
    var flat = _download_f32(ctx, dflat, m * n, tag + " flat")
    var mma = _download_f32(ctx, dmma, m * n, tag + " mma")
    _ = dqa
    _ = dea
    _ = dqb
    _ = deb
    _ = dflat
    _ = dmma
    return (flat^, mma^)


def check_int8_mma_matches_flat(ctx: DeviceContext) raises:
    """GATE (L-9, DEVIATION 2910): the matrix-unit plan and the flat plan
    return the same bits on every OP_NT shape of the file and on the
    ragged shapes, and on the ragged shapes both equal the oracle. The
    oracle comparison is what a sabotage build fails here: the value arm
    reaches both plans alike, so plan-versus-plan alone would pass it."""
    var shapes = 0
    for s in range(SHAPE_COUNT):
        var sh = _shape(s)
        if sh[3] != OP_NT:
            continue
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var qa = quantize_rows_int8(_fill(m * k, 131 + s), m, k)
        var qb = quantize_rows_int8(_fill(n * k, 149 + s), n, k)
        var tag = "int8 mma " + String(m) + "x" + String(n) + "x" + String(k)
        var both = _run_int8_plans(ctx, qa, qb, m, n, k, tag)
        _first_diff(both[1], both[0], tag + " (mma vs flat)")
        shapes += 1
    for s in range(MMA_SHAPE_COUNT):
        var sh = _mma_shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var qa = quantize_rows_int8(_fill(m * k, 167 + s), m, k)
        var qb = quantize_rows_int8(_fill(n * k, 181 + s), n, k)
        var want = gemm_int8_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
        var tag = "int8 mma ragged " + String(m) + "x" + String(n) + "x" + String(k)
        var both = _run_int8_plans(ctx, qa, qb, m, n, k, tag)
        _first_diff(both[1], both[0], tag + " (mma vs flat)")
        _first_diff(both[1], want, tag + " (mma vs oracle)")
        _first_diff(both[0], want, tag + " (flat vs oracle)")
        shapes += 1
        print("   ok " + tag + "  c[0]=" + _show(both[1][0]))
    print("   ok mma == flat on " + String(shapes) + " shapes, oracle on " + String(MMA_SHAPE_COUNT))


def check_int8_is_batch_invariant(ctx: DeviceContext) raises:
    """GATE: a row's codes, exponent and products do not depend on the other
    rows in the call."""
    var m = 5
    var n = 40
    var k = 300
    var ha = _fill(m * k, 101)
    var hb = _fill(n * k, 103)
    var qb = quantize_rows_int8(hb, n, k)
    var dqb = _upload_i8(ctx, qb.q)
    var deb = _upload_i32(ctx, qb.e)
    var da = _upload_f32(ctx, ha)
    var dqa = ctx.enqueue_create_buffer[DType.int8](m * k)
    var dea = ctx.enqueue_create_buffer[DType.int32](m)
    var dc = _poisoned(ctx, m * n)
    quantize_rows_int8_device(ctx, dqa, dea, da, m, k)
    identical_gemm_int8_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
    ctx.synchronize()
    var batch = _download_f32(ctx, dc, m * n, "int8 batch")
    for i in range(m):
        var row = List[Float32]()
        for p in range(k):
            row.append(ha[i * k + p])
        var dr = _upload_f32(ctx, row)
        var dqr = ctx.enqueue_create_buffer[DType.int8](k)
        var der = ctx.enqueue_create_buffer[DType.int32](1)
        var dcr = _poisoned(ctx, n)
        quantize_rows_int8_device(ctx, dqr, der, dr, 1, k)
        identical_gemm_int8_into(ctx, dcr, dqr, der, dqb, deb, 1, n, k)
        ctx.synchronize()
        var one = _download_f32(ctx, dcr, n, "int8 row")
        for j in range(n):
            if _bits(one[j]) != _bits(batch[i * n + j]):
                raise Error("int8 row " + String(i) + " col " + String(j) + " differs between the batch and the single-row call")
        _ = dr
        _ = dqr
        _ = der
        _ = dcr
    _ = da
    _ = dqa
    _ = dea
    _ = dqb
    _ = deb
    _ = dc
    print("   ok " + String(m) + " int8 rows agree with their single-row calls")


def _gate(name: String, mut ran: Int, mut failed: Int, e: String):
    ran += 1
    if e.byte_length() > 0:
        failed += 1
        print("!! GATE FAILED: " + name)
        print("   " + e)
    else:
        print("ok " + name)


def main() raises:
    print(
        "== gemm/checks/gemm_lowbit_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + lowbit_sabotage_name()
        + "  host sabotage: " + String(GEMM_ORACLE_HOST_SABOTAGE) + " =="
    )
    print("   profiles: mojolearn.identical.gemm.bf16f32.v1, mojolearn.identical.gemm.int8i32.v1")
    print("   contract: gemm/IDENTICAL_LOWBIT_CONTRACT.md")
    print("   column: " + column_name(TARGET_COLUMN) + "  int8 dispatch: " + int8_plan_dispatch_name())
    var ran = 0
    var failed = 0
    try:
        check_bf16_narrowing_is_round_to_nearest_even()
        _gate(String("check_bf16_narrowing_is_round_to_nearest_even"), ran, failed, String(""))
    except e:
        _gate(String("check_bf16_narrowing_is_round_to_nearest_even"), ran, failed, String(e))
    try:
        check_integer_seams_match_explicit_spellings()
        _gate(String("check_integer_seams_match_explicit_spellings"), ran, failed, String(""))
    except e:
        _gate(String("check_integer_seams_match_explicit_spellings"), ran, failed, String(e))
    try:
        check_quantizer_bounds()
        _gate(String("check_quantizer_bounds"), ran, failed, String(""))
    except e:
        _gate(String("check_quantizer_bounds"), ran, failed, String(e))
    comptime if not has_accelerator():
        print("   no accelerator: the device gates did not run")
    else:
        var ctx = DeviceContext()
        try:
            check_bf16_device_conversions_match_host(ctx)
            _gate(String("check_bf16_device_conversions_match_host"), ran, failed, String(""))
        except e:
            _gate(String("check_bf16_device_conversions_match_host"), ran, failed, String(e))
        try:
            check_bf16_device_matches_oracle(ctx)
            _gate(String("check_bf16_device_matches_oracle"), ran, failed, String(""))
        except e:
            _gate(String("check_bf16_device_matches_oracle"), ran, failed, String(e))
        try:
            check_bf16_plans_agree(ctx)
            _gate(String("check_bf16_plans_agree"), ran, failed, String(""))
        except e:
            _gate(String("check_bf16_plans_agree"), ran, failed, String(e))
        try:
            check_bf16_is_batch_invariant(ctx)
            _gate(String("check_bf16_is_batch_invariant"), ran, failed, String(""))
        except e:
            _gate(String("check_bf16_is_batch_invariant"), ran, failed, String(e))
        try:
            check_int8_device_matches_oracle(ctx)
            _gate(String("check_int8_device_matches_oracle"), ran, failed, String(""))
        except e:
            _gate(String("check_int8_device_matches_oracle"), ran, failed, String(e))
        try:
            check_int8_is_batch_invariant(ctx)
            _gate(String("check_int8_is_batch_invariant"), ran, failed, String(""))
        except e:
            _gate(String("check_int8_is_batch_invariant"), ran, failed, String(e))
        comptime if lib_int8_matrix_unit_for[TARGET_COLUMN]():
            try:
                check_int8_mma_matches_flat(ctx)
                _gate(String("check_int8_mma_matches_flat"), ran, failed, String(""))
            except e:
                _gate(String("check_int8_mma_matches_flat"), ran, failed, String(e))
        else:
            # Not a pass: the gate did not run. The column has no integer
            # matrix unit, so the MMA plan cannot be launched here.
            print(
                "   check_int8_mma_matches_flat DID NOT RUN: column "
                + column_name(TARGET_COLUMN)
                + " has no int8 matrix unit (RUN OWED on an H100 and an MI300X/MI325X, tools/lowbit_mma_leg.sh)"
            )
    print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
    if failed > 0:
        raise Error(String(failed) + " gate(s) failed")
