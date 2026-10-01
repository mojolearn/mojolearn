# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exhaustive check of the lane-wise square root, and sampled checks of the
lane-wise division and of the Adam element update (lane neural-pass8).

`core/host_lanes.mojo::sqrt_lanes` respells `portable_sqrtf`
(checks/numerics.mojo) lane by lane; `div_lanes` respells `portable_divf`;
`training/optimizer_host_rows.mojo::adam_lanes` respells
`adam_element_oracle` statement by statement through them. Two spellings of
one arithmetic can part at a single input (a candidate residual tie, a
compiler contracting one spelling and not the other), so:

  - `sqrt_lanes` is compared with `identical_sqrt`, by bits, on EVERY one of
    the 2^32 Float32 bit patterns, as `byte_lm_host_exp_check.mojo` does for
    the exponential;
  - `div_lanes` is compared with `identical_div` on 2^26 pairs drawn from a
    fixed generator over every binade, both signs, zeros, subnormals,
    infinities and NaNs (one correctly rounded division under the same two
    flushes; the pairs are a sample, the statement is the same);
  - `adam_lanes` is compared with `adam_element_oracle` on 2^24 element
    quadruples drawn the same way, at three steps and two configurations
    (AdamW with decay, Adam without).

The sqrt work splits into 3 contiguous ranges of bit patterns on 3 threads.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/host_lanes_sqrt_check.mojo

Raises (nonzero exit) on any mismatch, after printing the first few.
"""
from std.memory import bitcast

from checks.numerics import identical_div, identical_sqrt
from core.host_lanes import F32V, HOST_FW, U32V, div_lanes, sqrt_lanes
from core.host_parallel import host_parallelize
from training.checks.optimizer_oracle import (
    OPT_ADAM,
    OPT_ADAMW,
    OptimizerConfig,
    adam_element_oracle,
    step_scalars,
)
from training.optimizer_host_rows import adam_lanes

comptime SQRT_CHECK_TASKS = 3
comptime DIV_SAMPLES = 1 << 26
comptime ADAM_SAMPLES = 1 << 24


def _next(mut state: UInt64) -> UInt64:
    state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return state


def _draw_bits(mut state: UInt64) -> UInt32:
    """A Float32 bit pattern: one in sixteen a special (zero, subnormal,
    infinity, NaN, either sign), the rest a value in a random binade."""
    var r = _next(state)
    var kind = Int((r >> 60) & UInt64(15))
    var low = UInt32(Int(r & UInt64(0xFFFFFFFF)))
    if kind == 0:
        var which = Int((r >> 56) & UInt64(7))
        if which == 0:
            return UInt32(0)
        if which == 1:
            return UInt32(0x80000000)
        if which == 2:
            return low & UInt32(0x007FFFFF)  # subnormal
        if which == 3:
            return (low & UInt32(0x007FFFFF)) | UInt32(0x80000000)
        if which == 4:
            return UInt32(0x7F800000)
        if which == 5:
            return UInt32(0xFF800000)
        if which == 6:
            return UInt32(0x7FC00000) | (low & UInt32(0x003FFFFF))
        return UInt32(0x7F800001) | (low & UInt32(0x003FFFFF))
    # a normal value: random sign, exponent over the whole range, mantissa
    var expo = UInt32(Int((r >> 32) & UInt64(0xFF)))
    if expo == UInt32(0):
        expo = UInt32(1)
    if expo == UInt32(255):
        expo = UInt32(254)
    return (low & UInt32(0x807FFFFF)) | (expo << UInt32(23))


def _draw_near(mut state: UInt64, scale: Float32) -> UInt32:
    """A finite value near `scale` in magnitude (an optimizer operand)."""
    var r = _next(state)
    var u = Float32(Int((r >> 33) & UInt64(0xFFFFFF))) / Float32(16777216.0)
    var sign = Float32(1.0) if (r & UInt64(1)) == UInt64(0) else Float32(-1.0)
    var e = Int((r >> 40) & UInt64(31)) - 16
    var mag = scale * (u + Float32(0.5))
    var p = Float32(1.0)
    if e > 0:
        for _ in range(e):
            p = p * Float32(2.0)
    else:
        for _ in range(-e):
            p = p * Float32(0.5)
    return bitcast[DType.uint32](sign * mag * p)


def main() raises:
    # ---- 1. sqrt_lanes over every bit pattern ------------------------------
    var total = 4294967296
    var chunk = (total + SQRT_CHECK_TASKS - 1) // SQRT_CHECK_TASKS
    chunk = ((chunk + HOST_FW - 1) // HOST_FW) * HOST_FW
    var bad = List[Int](length=SQRT_CHECK_TASKS, fill=0)
    var first_bits = List[Int](length=SQRT_CHECK_TASKS, fill=-1)
    var bp = bad.unsafe_ptr()
    var fp = first_bits.unsafe_ptr()

    def _task(c: Int) {imm bp, imm fp, imm chunk, imm total}:
        var lo = c * chunk
        var hi = lo + chunk
        if hi > total:
            hi = total
        var iota = U32V(0)
        for lane in range(HOST_FW):
            iota[lane] = UInt32(lane)
        var count = 0
        var first = -1
        var u = lo
        while u < hi:
            var xv = bitcast[DType.float32](U32V(UInt32(u)) + iota)
            var sv = sqrt_lanes(xv)
            for lane in range(HOST_FW):
                if bitcast[DType.uint32](sv[lane]) != bitcast[DType.uint32](identical_sqrt(xv[lane])):
                    count += 1
                    if first < 0:
                        first = u + lane
            u += HOST_FW
        bp.unsafe_store(c, count)
        fp.unsafe_store(c, first)

    host_parallelize(_task, SQRT_CHECK_TASKS)
    var sqrt_total = 0
    for c in range(SQRT_CHECK_TASKS):
        sqrt_total += bad[c]
        if first_bits[c] >= 0:
            var x = bitcast[DType.float32](UInt32(first_bits[c]))
            print("  first sqrt mismatch in range", c, "at bits", hex(first_bits[c]), "x =", x,
                  "scalar", identical_sqrt(x), "lanes", sqrt_lanes(F32V(x))[0])
    print("sqrt bit patterns checked:", total, "lanes of width", HOST_FW, "mismatches:", sqrt_total)

    # ---- 2. div_lanes on sampled pairs ---------------------------------------
    var state = UInt64(0x9E3779B97F4A7C15)
    var div_bad = 0
    var shown = 0
    var i = 0
    while i < DIV_SAMPLES:
        var av = F32V(0.0)
        var bv = F32V(0.0)
        for lane in range(HOST_FW):
            av[lane] = bitcast[DType.float32](_draw_bits(state))
            bv[lane] = bitcast[DType.float32](_draw_bits(state))
        var qv = div_lanes(av, bv)
        for lane in range(HOST_FW):
            if bitcast[DType.uint32](qv[lane]) != bitcast[DType.uint32](identical_div(av[lane], bv[lane])):
                div_bad += 1
                if shown < 4:
                    print("  div mismatch a", av[lane], "b", bv[lane], "scalar", identical_div(av[lane], bv[lane]), "lanes", qv[lane])
                    shown += 1
        i += HOST_FW
    print("div pairs checked:", DIV_SAMPLES, "mismatches:", div_bad)

    # ---- 3. adam_lanes on sampled element quadruples --------------------------
    var adam_bad = 0
    shown = 0
    var cfgs = List[OptimizerConfig]()
    cfgs.append(OptimizerConfig(OPT_ADAMW, Float32(1e-3), Float32(0.9), Float32(0.95), Float32(1e-8),
                                Float32(0.1), Float32(0.0), Float32(0.0), False, Float32(0.0)))
    cfgs.append(OptimizerConfig(OPT_ADAM, Float32(3e-4), Float32(0.9), Float32(0.999), Float32(1e-8),
                                Float32(0.0), Float32(0.0), Float32(0.0), False, Float32(0.0)))
    var steps: List[Int] = [1, 2, 1000]
    for ci in range(len(cfgs)):
        var cfg = cfgs[ci].copy()
        for si in range(len(steps)):
            var sc = step_scalars(cfg, steps[si])
            var k = 0
            while k < ADAM_SAMPLES // (len(cfgs) * len(steps)):
                var pv = F32V(0.0)
                var gv = F32V(0.0)
                var mv = F32V(0.0)
                var vv = F32V(0.0)
                for lane in range(HOST_FW):
                    var r = _next(state)
                    if (r & UInt64(63)) == UInt64(0):
                        pv[lane] = bitcast[DType.float32](_draw_bits(state) & UInt32(0x7FFFFFFF) if (r & UInt64(64)) != UInt64(0) else _draw_bits(state))
                        gv[lane] = bitcast[DType.float32](_draw_bits(state))
                        mv[lane] = bitcast[DType.float32](_draw_bits(state))
                        vv[lane] = bitcast[DType.float32](_draw_bits(state) & UInt32(0x7FFFFFFF))
                    else:
                        pv[lane] = bitcast[DType.float32](_draw_near(state, Float32(0.05)))
                        gv[lane] = bitcast[DType.float32](_draw_near(state, Float32(1e-3)))
                        mv[lane] = bitcast[DType.float32](_draw_near(state, Float32(1e-3)))
                        vv[lane] = bitcast[DType.float32](_draw_near(state, Float32(1e-6)) & UInt32(0x7FFFFFFF))
                var out = adam_lanes(pv, gv, mv, vv, cfg, sc)
                for lane in range(HOST_FW):
                    var e = adam_element_oracle(pv[lane], gv[lane], mv[lane], vv[lane], cfg, sc)
                    if (bitcast[DType.uint32](out[0][lane]) != bitcast[DType.uint32](e.p)
                            or bitcast[DType.uint32](out[1][lane]) != bitcast[DType.uint32](e.m)
                            or bitcast[DType.uint32](out[2][lane]) != bitcast[DType.uint32](e.v)):
                        adam_bad += 1
                        if shown < 4:
                            print("  adam mismatch cfg", ci, "t", steps[si], "p", pv[lane], "g", gv[lane], "m", mv[lane], "v", vv[lane],
                                  "scalar", e.p, e.m, e.v, "lanes", out[0][lane], out[1][lane], out[2][lane])
                            shown += 1
                k += HOST_FW
    print("adam quadruples checked:", ADAM_SAMPLES, "mismatches:", adam_bad)

    if sqrt_total != 0 or div_bad != 0 or adam_bad != 0:
        raise Error("lane-wise sqrt / div / adam differ from the scalar seams")
    print("PASS")
