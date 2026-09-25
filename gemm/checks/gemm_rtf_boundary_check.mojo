# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The whole device GEMM, not one seam, at the smallest-normal boundary
(lane `lane/apple-seam-repair`, 2026-09-18).

`gemm_seam_probe.mojo` proves the per-step seam on 262,144 triples. The tuned
kernel on Apple does not run that seam: it runs the fast step and an exact
BLOCK ADMISSION (`TUNED_BLOCK_ADMIT`, gemm_identical.mojo) and recomputes a
block exactly when admission fails. This check drives the device GEMM on
matrices built from the probe's 64 adversarial words, so products land in the
window `[2^-126 - 2^-150, 2^-126)`, and compares every cell's bits against
the host oracle `gemm_oracle` (the host FMA rounds once and the oracle flushes
after: round-then-flush, the contract). A build without the repair
(`-D MOJOLEARN_NO_ZERO_FMA_REPAIR=1`) must FAIL here on Apple; that is how
this check is shown able to fail.

Cases: every plan below at k = 1 (all 64 x 64 word pairs with acc = +0, the
eight literal boundary cells among them), then pseudo-random word draws at
several k (multi-window, ragged tiles), then an ADMITTED control (ordinary
magnitudes: every block takes the fast path) that must also match. Each line
prints an FNV-1a 64 hash of the device output so columns can be compared.

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \\
        gemm/checks/gemm_rtf_boundary_check.mojo -o <bin>
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.rtf_seam import RTF_REPAIR
from gemm.checks.gemm_identical import (
    PLAN_FLAT,
    PLAN_SPLIT_128_8X8,
    PLAN_TUNED_128_8X8,
    PLAN_TUNED_64_4X4,
    TUNED_BLOCK_ADMIT,
    gemm_plan_name,
    identical_gemm_into,
    identical_gemm_with_plan,
    identical_gemm_workspace_floats,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NN, gemm_oracle


def _words() -> List[UInt32]:
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
    return words^


def _mix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> UInt64(31))


def _mix32(x: UInt32) -> UInt32:
    var h = x * UInt32(0x9E3779B1)
    h = h ^ (h >> UInt32(15))
    h = h * UInt32(0x85EBCA77)
    h = h ^ (h >> UInt32(13))
    h = h * UInt32(0xC2B2AE3D)
    return h ^ (h >> UInt32(16))


def _kind_word(i: Int, seed: UInt32, kind: Int) -> Float32:
    """The operand kinds of `bench/gemm_excp_ab_main.mojo::fill_kernel` (the
    AMD and NVIDIA admission A/B), filled on the host at small shapes:
    0 ordinary, 2 mixed, 3 skew's A, 4 border, 5 sparse."""
    var h = _mix32(UInt32(i) ^ _mix32(seed))
    var h2 = _mix32(h ^ UInt32(0x5BD1E995))
    var sign = h & UInt32(0x80000000)
    var mant = h & UInt32(0x007FFFFF)
    var e = UInt32(119) + (h2 % UInt32(9))
    if kind == 2:
        if (h2 >> UInt32(26)) == UInt32(0):
            e = UInt32(17) + (h2 % UInt32(10))
    elif kind == 3:
        e = UInt32(2) + (h2 % UInt32(11))
    elif kind == 4:
        e = UInt32(87) + (h2 % UInt32(7))
    elif kind == 5:
        if (h2 >> UInt32(20)) == UInt32(0):
            e = UInt32(17) + (h2 % UInt32(10))
    return bitcast[DType.float32](sign | (e << UInt32(23)) | mant)


def _fnv(v: List[Float32]) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(v)):
        var w = bitcast[DType.uint32](v[i])
        for k in range(4):
            h = h ^ UInt64((w >> UInt32(8 * k)) & UInt32(0xFF))
            h = h * UInt64(0x100000001B3)
    return h


def _hex(w: UInt64, nib: Int) -> String:
    var d = "0123456789abcdef"
    var s = String("")
    for k in range(nib):
        var x = Int((w >> UInt64(4 * (nib - 1 - k))) & UInt64(0xF))
        s += d[byte=x]
    return s


def _run(
    ctx: DeviceContext, ha: List[Float32], hb: List[Float32],
    m: Int, n: Int, k: Int, plan: Int,
) raises -> List[Float32]:
    var da = ctx.enqueue_create_buffer[DType.float32](max(len(ha), 1))
    var db = ctx.enqueue_create_buffer[DType.float32](max(len(hb), 1))
    var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
    var nws = identical_gemm_workspace_max_floats(m, n, k)
    if plan >= 0:
        nws = max(nws, identical_gemm_workspace_floats(m, n, k, plan))
    var dw = ctx.enqueue_create_buffer[DType.float32](max(nws, 1))
    with da.map_to_host() as h:
        for i in range(len(ha)):
            h[i] = ha[i]
    with db.map_to_host() as h:
        for i in range(len(hb)):
            h[i] = hb[i]
    with dc.map_to_host() as h:
        for i in range(m * n):
            h[i] = bitcast[DType.float32](UInt32(0x7FC0DEAD))
    if plan < 0:
        identical_gemm_into(ctx, dc, da, db, dw, m, n, k, OP_NN)
    else:
        identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, OP_NN, plan)
    ctx.synchronize()
    var out = List[Float32]()
    with dc.map_to_host() as h:
        for i in range(m * n):
            out.append(h[i])
    return out^


def _case(
    ctx: DeviceContext, label: String, ha: List[Float32], hb: List[Float32],
    m: Int, n: Int, k: Int, mut fails: Int,
) raises:
    var want = gemm_oracle(ha, hb, OP_NN, m, n, k)
    var plans: List[Int] = [-1, PLAN_FLAT, PLAN_TUNED_64_4X4, PLAN_TUNED_128_8X8, PLAN_SPLIT_128_8X8]
    for pi in range(len(plans)):
        var plan = plans[pi]
        var got = _run(ctx, ha, hb, m, n, k, plan)
        var bad = 0
        var first = -1
        for c in range(m * n):
            if bitcast[DType.uint32](got[c]) != bitcast[DType.uint32](want[c]):
                bad += 1
                if first < 0:
                    first = c
        var name = String("dispatch") if plan < 0 else gemm_plan_name(plan)
        var line = (
            ("RTFGEMM OK   " if bad == 0 else "RTFGEMM FAIL ") + label
            + " " + String(m) + "x" + String(n) + "x" + String(k)
            + " plan=" + name + " mismatches=" + String(bad)
            + " device_fnv=" + _hex(_fnv(got), 16) + " oracle_fnv=" + _hex(_fnv(want), 16)
        )
        if bad > 0:
            fails += 1
            line += (
                " first=(" + String(first // n) + "," + String(first % n) + ") device="
                + _hex(UInt64(bitcast[DType.uint32](got[first])), 8) + " oracle="
                + _hex(UInt64(bitcast[DType.uint32](want[first])), 8)
            )
        print(line)


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "gemm_rtf_boundary_check: build with -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    print(
        "RTFGEMM column=" + column_name(TARGET_COLUMN) + " rtf_repair=" + String(RTF_REPAIR)
        + " block_admit=" + String(TUNED_BLOCK_ADMIT)
    )
    var ctx = DeviceContext()
    var w = _words()
    var fails = 0
    # k = 1: all 64 x 64 pairs, acc = +0 (the probe's acc=0 slice).
    var a1 = List[Float32]()
    var b1 = List[Float32]()
    for i in range(64):
        a1.append(bitcast[DType.float32](w[i]))
        b1.append(bitcast[DType.float32](w[i]))
    _case(ctx, "pairs", a1, b1, 64, 64, 1, fails)
    # Random draws from the adversarial words; accumulators vary.
    var shapes: List[Int] = [130, 130, 2, 130, 131, 3, 129, 200, 17, 260, 130, 40, 64, 64, 300]
    for s in range(len(shapes) // 3):
        var m = shapes[3 * s]
        var n = shapes[3 * s + 1]
        var k = shapes[3 * s + 2]
        var ha = List[Float32]()
        var hb = List[Float32]()
        for i in range(m * k):
            ha.append(bitcast[DType.float32](w[Int(_mix(UInt64(i * 7 + s)) % 64)]))
        for i in range(k * n):
            hb.append(bitcast[DType.float32](w[Int(_mix(UInt64(i * 13 + 1000 + s)) % 64)]))
        _case(ctx, "draw" + String(s), ha, hb, m, n, k, fails)
    # ADMITTED control: magnitudes in [0.5, 2), every block on the fast path.
    var m2 = 256
    var n2 = 256
    var k2 = 96
    var ha2 = List[Float32]()
    var hb2 = List[Float32]()
    for i in range(m2 * k2):
        ha2.append(bitcast[DType.float32](UInt32(0x3F000000) | UInt32(_mix(UInt64(i)) & 0x80FFFFFF)))
    for i in range(k2 * n2):
        hb2.append(bitcast[DType.float32](UInt32(0x3F000000) | UInt32(_mix(UInt64(i + 77777)) & 0x80FFFFFF)))
    _case(ctx, "admitted", ha2, hb2, m2, n2, k2, fails)
    # MIXED (lane/apple-identical-gemm): the block admission holds (A's
    # smallest nonzero words near 2^-97 (exponent field 30 or 31), B in [0.5, 2): exponent fields sum
    # to at least 156, above 151) but every fifth 16-deep window of A holds
    # the small words, so those windows fail the window admission (sum below
    # 174) and the rest of their leaf runs the shipped step, while the other
    # leaves and windows are admitted. Signs and mantissas vary.
    var m3 = 192
    var n3 = 160
    var k3 = 640
    var ha3 = List[Float32]()
    var hb3 = List[Float32]()
    for i in range(m3 * k3):
        var p = i % k3
        var hi = UInt32(0x0F000000) if (p // 16) % 5 == 2 else UInt32(0x3F000000)
        ha3.append(bitcast[DType.float32](hi | UInt32(_mix(UInt64(i + 555)) & 0x80FFFFFF)))
    for i in range(k3 * n3):
        hb3.append(bitcast[DType.float32](UInt32(0x3F000000) | UInt32(_mix(UInt64(i + 999)) & 0x80FFFFFF)))
    _case(ctx, "mixed", ha3, hb3, m3, n3, k3, fails)
    # The admission A/B's adversarial operand kinds (`bench/gemm_excp_ab_main.mojo`)
    # at a small shape with the production k = 768: mixed (one word in 64
    # near 2^-105), skew (A in [2^-125, 2^-114), B ordinary: subnormal
    # products), border (both in [2^-40, 2^-33): exponent sums 174..186,
    # the window admission's bound), sparse (one word in 4096 near 2^-105).
    var kind_names: List[String] = ["kmixed", "kskew", "kborder", "ksparse"]
    var kind_ids: List[Int] = [2, 3, 4, 5]
    var m4 = 160
    var n4 = 192
    var k4 = 768
    for kd in range(len(kind_ids)):
        var kid = kind_ids[kd]
        var ha4 = List[Float32]()
        var hb4 = List[Float32]()
        for i in range(m4 * k4):
            ha4.append(_kind_word(i, UInt32(1000 + 7 * kd), kid))
        for i in range(k4 * n4):
            hb4.append(_kind_word(i, UInt32(2000 + 7 * kd), 0 if kid == 3 else kid))
        _case(ctx, kind_names[kd], ha4, hb4, m4, n4, k4, fails)
    print("RTFGEMM DONE fails=" + String(fails))
    if fails > 0:
        raise Error("gemm_rtf_boundary_check: " + String(fails) + " plan/case pairs differ from the oracle")
