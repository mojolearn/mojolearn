# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host helpers of the GEMM step arms (DEVIATION 2543).

The twelve GEMM calls of the byte LM training step at the target shape
(`docs/lanes/BRIEF_gemm_step_2026-09-11.md` section 2), and the fill,
poison, readback, compare and median helpers that
`gemm/checks/gemm_step_arms_check.mojo` and `bench/gemm_step_price_main.mojo`
share.

The table is DERIVED, not written down: each linear layer's forward is
`OP_NT` at `(tokens, out_features, in_features)`, and its two backward
calls come from `gemm_backward_a_call` and `gemm_backward_b_call`, the
functions the step's backward routes through. So the table cannot hold a
second opinion about a backward shape.

No kernel lives here. The arm kernel, its launcher and the selector are in
`gemm/checks/gemm_identical.mojo` beside the hook in `identical_gemm_into`
that dispatches them (a separate module would need a circular import).
"""
from std.memory import bitcast
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from gemm.checks.gemm_backward import gemm_backward_a_call, gemm_backward_b_call
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN

#: `TARGET_SHAPE` in `tools/lm_step_memory_probe.py`: 12 layers, DM 768,
#: FF 2048, V 50257, L 2048, batch 1.
comptime GEMM_STEP_TOKENS = 2048
comptime GEMM_STEP_DM = 768
comptime GEMM_STEP_FF = 2048
comptime GEMM_STEP_VOCAB = 50257
comptime GEMM_STEP_LAYERS = 12
comptime GEMM_STEP_LM_CALLS = 12

#: Written into `C` before every launch; a cell still holding it was never
#: written (the device check's value).
comptime GEMM_STEP_POISON = Float32(-987654.0)


def gemm_step_lm_call_name(i: Int) -> String:
    """`proj_fwd`, `proj_dA`, `proj_dB`, then `gateup_*`, `down_*`, `head_*`."""
    var group = i // 3
    var role = i % 3
    var layer = String("proj")
    if group == 1:
        layer = String("gateup")
    elif group == 2:
        layer = String("down")
    elif group == 3:
        layer = String("head")
    if role == 0:
        return layer + "_fwd"
    if role == 1:
        return layer + "_dA"
    return layer + "_dB"


def gemm_step_lm_call(i: Int) raises -> Tuple[Int, Int, Int, Int, Int]:
    """`(op, m, n, k, calls per step)` for LM call `i`.

    q, k, v, o projections: 4 per layer; gate and up: 2 per layer; down: 1
    per layer; the head: 1 per step. Forward `OP_NT` at `(tokens, out, in)`;
    dA and dB through the backward module's own shape functions.
    """
    if i < 0 or i >= GEMM_STEP_LM_CALLS:
        raise Error("gemm_step_lm_call: no LM call " + String(i))
    var group = i // 3
    var role = i % 3
    var out_f = GEMM_STEP_DM
    var in_f = GEMM_STEP_DM
    var per = 4 * GEMM_STEP_LAYERS
    if group == 1:
        out_f = GEMM_STEP_FF
        per = 2 * GEMM_STEP_LAYERS
    elif group == 2:
        in_f = GEMM_STEP_FF
        per = GEMM_STEP_LAYERS
    elif group == 3:
        out_f = GEMM_STEP_VOCAB
        per = 1
    var m = GEMM_STEP_TOKENS
    if role == 0:
        return (OP_NT, m, out_f, in_f, per)
    if role == 1:
        var ca = gemm_backward_a_call(OP_NT, m, out_f, in_f)
        return (ca[0], ca[1], ca[2], ca[3], per)
    var cb = gemm_backward_b_call(OP_NT, m, out_f, in_f)
    return (cb[0], cb[1], cb[2], cb[3], per)


#: DEVIATION 2593: control calls the price harness adds under
#: `MOJOLEARN_GEMM_STEP_CONTROLS=1` (never weighted into the STEP line). Each
#: separates explanations of brief docs/lanes/BRIEF_gemm_long_k_2026-09-11.md
#: section 3.2 at a 128x128 shipped plan: TN at 36 blocks and a short k (E3,
#: E4), NT and NN at proj_dB's 36 blocks and long k (E2), 132 against 143
#: blocks (rounds against a proportional rate, and the column's block
#: parallelism), and 64 blocks at a long k.
comptime GEMM_STEP_CONTROL_CALLS = 6


def gemm_step_control_call_name(i: Int) -> String:
    if i == 0:
        return String("ctl_tn_768x768x768")
    if i == 1:
        return String("ctl_nt_768x768x2048")
    if i == 2:
        return String("ctl_nn_768x768x2048")
    if i == 3:
        return String("ctl_nt_1536x1408x768")
    if i == 4:
        return String("ctl_nt_1664x1408x768")
    return String("ctl_nt_1024x1024x2048")


def gemm_step_control_call(i: Int) raises -> Tuple[Int, Int, Int, Int, Int]:
    """`(op, m, n, k, 0)` for control call `i`: zero calls per step."""
    if i == 0:
        return (OP_TN, 768, 768, 768, 0)
    if i == 1:
        return (OP_NT, 768, 768, 2048, 0)
    if i == 2:
        return (OP_NN, 768, 768, 2048, 0)
    if i == 3:
        return (OP_NT, 1536, 1408, 768, 0)
    if i == 4:
        return (OP_NT, 1664, 1408, 768, 0)
    if i == 5:
        return (OP_NT, 1024, 1024, 2048, 0)
    raise Error("gemm_step_control_call: no control call " + String(i))


def gemm_step_operand_counts(m: Int, n: Int, k: Int) -> Tuple[Int, Int]:
    """`A` holds `m k` floats and `B` holds `n k` in all three orientations
    (contract section 3: `m x k` or `k x m`, `k x n` or `n x k`)."""
    return (m * k, n * k)


def _mix(i: Int, salt: Int) -> UInt64:
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(
        0xBF58476D1CE4E5B9
    )
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _value(i: Int, salt: Int) -> Float32:
    """The device check's generator: a 13-bit significand spread over eight
    binades (`2^-4 .. 2^3`), signed, so every product is inexact and the
    order of additions matters."""
    var h = _mix(i, salt)
    var mant = Float32(1.0) + Float32(Int(h & UInt64(0xFFF))) / Float32(4096.0)
    var scales = SIMD[DType.float32, 8](0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    var v = mant * scales[Int((h >> UInt64(13)) & UInt64(7))]
    if (h >> UInt64(20)) & UInt64(1) == UInt64(1):
        return -v
    return v


def gemm_step_fill(
    ctx: DeviceContext,
    mut d: DeviceBuffer[DType.float32],
    count: Int,
    salt: Int,
    subnormal: Bool,
) raises:
    """Fill `count` operand floats. `subnormal` scales every value by
    `2^-64`, so every PRODUCT lands below the smallest normal and the leaf
    runs in the range seams 5a to 5c exist for (an adversarial kind: a
    correctness fixture, never a timing input)."""
    if count < 1:
        return
    var scale = Float32(1.0)
    if subnormal:
        for _ in range(64):
            scale = scale * Float32(0.5)
    var h = ctx.enqueue_create_host_buffer[DType.float32](count)
    ctx.synchronize()
    var p = h.unsafe_ptr()
    for i in range(count):
        p.unsafe_store(i, _value(i, salt) * scale)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h


def gemm_step_poison(
    ctx: DeviceContext,
    mut d: DeviceBuffer[DType.float32],
    mut h: HostBuffer[DType.float32],
    count: Int,
) raises:
    """Poison `count` cells of `h`, copy them over `d`, wait."""
    var p = h.unsafe_ptr()
    for i in range(count):
        p.unsafe_store(i, GEMM_STEP_POISON)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()


def gemm_step_readback(
    ctx: DeviceContext,
    mut d: DeviceBuffer[DType.float32],
    mut h: HostBuffer[DType.float32],
) raises:
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d)
    ctx.synchronize()


def gemm_step_compare(
    got: HostBuffer[DType.float32],
    expected: HostBuffer[DType.float32],
    count: Int,
) -> Tuple[Int, Int, Int]:
    """`(cells whose bits differ, cells of got still poisoned, first
    differing cell or -1)`. Bits, never float equality: `-0.0 == +0.0`."""
    var gp = got.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    var poison_bits = bitcast[DType.uint32](GEMM_STEP_POISON)
    var moved = 0
    var poison = 0
    var first = -1
    for i in range(count):
        var g = bitcast[DType.uint32](gp.unsafe_load(i))
        if g == poison_bits:
            poison += 1
        if g != bitcast[DType.uint32](ep.unsafe_load(i)):
            if first < 0:
                first = i
            moved += 1
    return (moved, poison, first)


def gemm_step_poison_left(h: HostBuffer[DType.float32], count: Int) -> Int:
    var p = h.unsafe_ptr()
    var poison_bits = bitcast[DType.uint32](GEMM_STEP_POISON)
    var left = 0
    for i in range(count):
        if bitcast[DType.uint32](p.unsafe_load(i)) == poison_bits:
            left += 1
    return left


def gemm_step_digest(h: HostBuffer[DType.float32], count: Int) -> UInt64:
    """FNV-1a64 over the raw output bits, as `gemm_tuned_probe.mojo`."""
    var p = h.unsafe_ptr()
    var acc = UInt64(0xCBF29CE484222325)
    for i in range(count):
        var b = UInt64(bitcast[DType.uint32](p.unsafe_load(i)))
        for s in range(4):
            acc = (acc ^ ((b >> UInt64(8 * s)) & UInt64(0xFF))) * UInt64(0x100000001B3)
    return acc


def gemm_step_env_int(name: String, dflt: Int) raises -> Int:
    """An integer knob. Unset is the default; anything unparsable RAISES (a
    typo in a leg's environment is not a default)."""
    var s = String(getenv(name))
    if s.byte_length() == 0:
        return dflt
    try:
        return Int(s)
    except:
        raise Error(name + "='" + s + "' is not an integer")


def gemm_step_selected(spec: String, name: String) -> Bool:
    """Empty `spec` selects everything; otherwise a comma list of names."""
    if spec.byte_length() == 0:
        return True
    return (String(",") + spec + ",").find(String(",") + name + ",") >= 0


def gemm_step_median_ms(samples: List[Int]) -> Float64:
    """Median of nanosecond samples, in milliseconds (insertion sort on a
    copy; a handful of samples)."""
    var n = len(samples)
    if n == 0:
        return Float64(0.0)
    var s = List[Int]()
    for i in range(n):
        s.append(samples[i])
    for i in range(1, n):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    if n % 2 == 1:
        return Float64(s[n // 2]) / 1.0e6
    return (Float64(s[n // 2 - 1]) + Float64(s[n // 2])) / 2.0e6
