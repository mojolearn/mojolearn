# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE GATE of `gemm_int15_heads.mojo` (lane/lowbit-default, 2026-09-29):
every (batch, head) score product in one launch.

    pixi run check-gemm-int15-heads                     MUST pass
    pixi run check-gemm-int15-heads-sabotage            MUST fail (every stored value flipped)
    pixi run check-gemm-int15-heads-epilogue-sabotage   MUST fail (the high sum from the middle slot)
    pixi run check-gemm-int15-heads-exponent-sabotage   MUST fail (a key's exponent read at the row index)
    pixi run check-gemm-int15-heads-host-sabotage       MUST fail (the oracle's value flipped)

Per case, on operands whose ROWS CARRY DIFFERENT EXPONENTS (at least five
distinct per operand, else the case refuses as blind), with n_rep > 1
where the case says so:
  1. the device's ONE quantizer launch over every head's rows equals the
     host rule `quantize_rows_int15` (planes and exponents), for the query
     rows and for the key rows: quantizing all heads at once is quantizing
     each head, because a row's code is a function of that row alone;
  2. the one-launch product equals the normative oracle
     (`gemm_int15_oracle`) of each (batch, head) product, bit for bit;
  3. it equals the reference plan (`identical_gemm_int15_planes_into`) run
     once per (batch, head) on the same planes, bit for bit.
Every output is poisoned first and read back whole.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.sys import has_accelerator

from gemm.checks.gemm_int15 import (
    Int15QuantWorkspace,
    identical_gemm_int15_planes_into,
    quantize_planes_int15_parallel_device,
)
from gemm.checks.gemm_int15_heads import INT15_HEADS_MAX_L, identical_gemm_int15_heads_into
from gemm.host.gemm_int15_oracle import gemm_int15_oracle, quantize_rows_int15, split_int15

comptime POISON = Float32(-987654.0)


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32, 1](x)


def _hash64(i: Int, salt: Int) -> UInt64:
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _pow2(e: Int) -> Float32:
    var s = Float32(1.0)
    if e >= 0:
        for _ in range(e):
            s = s * Float32(2.0)
    else:
        for _ in range(-e):
            s = s * Float32(0.5)
    return s


def _rows(rows: Int, k: Int, salt: Int, modulus: Int, shift: Int, sign: Int) -> List[Float32]:
    """Values with a 20-bit significand, row `r` scaled by
    `2^(sign * ((r mod modulus) - shift))`: rows of different exponents."""
    var v = List[Float32]()
    for r in range(rows):
        var s = _pow2(sign * ((r % modulus) - shift))
        for p in range(k):
            var h = _hash64(r * k + p, salt)
            var x = (Float32(1.0) + Float32(Int(h & UInt64(0xFFFFF))) / Float32(1048576.0)) * _pow2(
                Int((h >> UInt64(21)) & UInt64(3)) - 2
            )
            if (h >> UInt64(28)) & UInt64(1) == UInt64(1):
                x = -x
            v.append(x * s)
    return v^


def _distinct(e: List[Int32]) -> Int:
    var seen = List[Int32]()
    for i in range(len(e)):
        var found = False
        for j in range(len(seen)):
            if seen[j] == e[i]:
                found = True
                break
        if not found:
            seen.append(e[i])
    return len(seen)


def _upload[dt: DType](ctx: DeviceContext, h: List[Scalar[dt]]) raises -> DeviceBuffer[dt]:
    var n = len(h)
    var d = ctx.enqueue_create_buffer[dt](n)
    var hb = ctx.enqueue_create_host_buffer[dt](n)
    ctx.synchronize()
    for i in range(n):
        hb.unsafe_ptr().unsafe_store(i, h[i])
    ctx.enqueue_copy(dst_buf=d, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    _ = hb
    return d^


def _download[dt: DType](ctx: DeviceContext, mut d: DeviceBuffer[dt], count: Int) raises -> List[Scalar[dt]]:
    var hb = ctx.enqueue_create_host_buffer[dt](count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    var out = List[Scalar[dt]]()
    for i in range(count):
        out.append(hb.unsafe_ptr().unsafe_load(i))
    _ = hb
    return out^


def _poisoned(ctx: DeviceContext, count: Int) raises -> DeviceBuffer[DType.float32]:
    var h = List[Float32]()
    for _ in range(count):
        h.append(POISON)
    return _upload[DType.float32](ctx, h)


def _compare(got: List[Float32], want: List[Float32], tag: String) raises:
    var bad = 0
    var first = -1
    for i in range(len(want)):
        if _bits(got[i]) == _bits(POISON):
            raise Error(tag + ": POISON SURVIVED at cell " + String(i))
        if _bits(got[i]) != _bits(want[i]):
            bad += 1
            if first < 0:
                first = i
    if bad > 0:
        raise Error(
            tag + ": " + String(bad) + " of " + String(len(want)) + " cells differ; first at "
            + String(first) + " got " + String(got[first]) + " want " + String(want[first])
        )


def _same_i[dt: DType](got: List[Scalar[dt]], want: List[Scalar[dt]], tag: String) raises:
    for i in range(len(want)):
        if got[i] != want[i]:
            raise Error(tag + ": differs at " + String(i))


def check_case(ctx: DeviceContext, b: Int, nh: Int, nkv: Int, l: Int, s: Int, k: Int, salt: Int) raises:
    var tag = String("b") + String(b) + ".nh" + String(nh) + ".nkv" + String(nkv) + ".l" + String(l) + ".s" + String(s) + ".k" + String(k)
    var ra = b * nh * l
    var rb = b * nkv * s
    var fa = _rows(ra, k, salt, 7, 3, 1)
    var fb = _rows(rb, k, salt + 17, 5, 2, -1)
    var qa = quantize_rows_int15(fa, ra, k)
    var qb = quantize_rows_int15(fb, rb, k)
    if _distinct(qa.e) < 5 or _distinct(qb.e) < 5:
        raise Error(tag + ": the fixture is blind (fewer than 5 distinct row exponents)")
    var pa = split_int15(qa.q)
    var pb = split_int15(qb.q)
    # 1. one quantizer launch over every head's rows == the host rule
    var dfa = _upload[DType.float32](ctx, fa)
    var dfb = _upload[DType.float32](ctx, fb)
    var ah = ctx.enqueue_create_buffer[DType.int8](ra * k)
    var al = ctx.enqueue_create_buffer[DType.int8](ra * k)
    var ea = ctx.enqueue_create_buffer[DType.int32](ra)
    var bh = ctx.enqueue_create_buffer[DType.int8](rb * k)
    var bl = ctx.enqueue_create_buffer[DType.int8](rb * k)
    var eb = ctx.enqueue_create_buffer[DType.int32](rb)
    var quant = Int15QuantWorkspace(ctx)
    quantize_planes_int15_parallel_device(ctx, ah, al, ea, dfa, quant, ra, k, False)
    quantize_planes_int15_parallel_device(ctx, bh, bl, eb, dfb, quant, rb, k, False)
    ctx.synchronize()
    _same_i[DType.int8](_download[DType.int8](ctx, ah, ra * k), pa.hi, tag + " query hi")
    _same_i[DType.int8](_download[DType.int8](ctx, al, ra * k), pa.lo, tag + " query lo")
    _same_i[DType.int32](_download[DType.int32](ctx, ea, ra), qa.e, tag + " query exponents")
    _same_i[DType.int8](_download[DType.int8](ctx, bh, rb * k), pb.hi, tag + " key hi")
    _same_i[DType.int8](_download[DType.int8](ctx, bl, rb * k), pb.lo, tag + " key lo")
    _same_i[DType.int32](_download[DType.int32](ctx, eb, rb), qb.e, tag + " key exponents")
    # the oracle, one (batch, head) product at a time
    var n_rep = nh // nkv
    var want = List[Float32]()
    for bb in range(b):
        for h in range(nh):
            var kvh = h // n_rep
            var a0 = (bb * nh + h) * l
            var b0 = (bb * nkv + kvh) * s
            var sa = List[Int16]()
            var sea = List[Int32]()
            for r in range(l):
                sea.append(qa.e[a0 + r])
                for p in range(k):
                    sa.append(qa.q[(a0 + r) * k + p])
            var sb = List[Int16]()
            var seb = List[Int32]()
            for r in range(s):
                seb.append(qb.e[b0 + r])
                for p in range(k):
                    sb.append(qb.q[(b0 + r) * k + p])
            var blk = gemm_int15_oracle(sa, sea, sb, seb, l, s, k)
            for i in range(len(blk)):
                want.append(blk[i])
    var cells = b * nh * l * s
    # 2. the one-launch product == the oracle
    var c = _poisoned(ctx, cells)
    identical_gemm_int15_heads_into(ctx, c, ah, al, ea, bh, bl, eb, b, nh, nkv, l, s, k)
    ctx.synchronize()
    var got = _download[DType.float32](ctx, c, cells)
    _compare(got, want, tag + " heads vs oracle")
    # 3. == the reference plan, once per (batch, head), on the same planes
    var ref_c = _poisoned(ctx, cells)
    for bb in range(b):
        for h in range(nh):
            var kvh = h // n_rep
            var a0 = (bb * nh + h) * l
            var b0 = (bb * nkv + kvh) * s
            var sub_c = ref_c.create_sub_buffer[DType.float32]((bb * nh + h) * l * s, l * s)
            var sub_ah = ah.create_sub_buffer[DType.int8](a0 * k, l * k)
            var sub_al = al.create_sub_buffer[DType.int8](a0 * k, l * k)
            var sub_ea = ea.create_sub_buffer[DType.int32](a0, l)
            var sub_bh = bh.create_sub_buffer[DType.int8](b0 * k, s * k)
            var sub_bl = bl.create_sub_buffer[DType.int8](b0 * k, s * k)
            var sub_eb = eb.create_sub_buffer[DType.int32](b0, s)
            identical_gemm_int15_planes_into(ctx, sub_c, sub_ah, sub_al, sub_ea, sub_bh, sub_bl, sub_eb, l, s, k)
            ctx.synchronize()
    var ref_got = _download[DType.float32](ctx, ref_c, cells)
    _compare(got, ref_got, tag + " heads vs the reference plan per head")
    print("   ok " + tag + ": one quantizer launch == host rule; one product launch == oracle == reference plan per head (" + String(cells) + " cells)")


def check_refusals(ctx: DeviceContext) raises:
    var c = ctx.enqueue_create_buffer[DType.float32](1)
    var i8 = ctx.enqueue_create_buffer[DType.int8](1)
    var i8b = ctx.enqueue_create_buffer[DType.int8](1)
    var i8c = ctx.enqueue_create_buffer[DType.int8](1)
    var i8d = ctx.enqueue_create_buffer[DType.int8](1)
    var e1 = ctx.enqueue_create_buffer[DType.int32](1)
    var e2 = ctx.enqueue_create_buffer[DType.int32](1)
    var refused = 0
    try:
        identical_gemm_int15_heads_into(ctx, c, i8, i8b, e1, i8c, i8d, e2, 1, 3, 2, 1, 1, 1)
    except:
        refused += 1
    try:
        identical_gemm_int15_heads_into(ctx, c, i8, i8b, e1, i8c, i8d, e2, 1, 1, 1, INT15_HEADS_MAX_L + 1, 1, 1)
    except:
        refused += 1
    try:
        identical_gemm_int15_heads_into(ctx, c, i8, i8b, e1, i8c, i8d, e2, 1, 1, 1, 1, 2, 1)
    except:
        refused += 1
    if refused != 3:
        raise Error("refusals: " + String(refused) + " of 3 (n_heads not a multiple, l too large, buffer short)")
    print("   ok the entry refuses a non-multiple n_rep, l above the decode rows and a short buffer by name")


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator: the heads gate needs a device")
        return
    else:
        var ctx = DeviceContext()
        var failed = 0
        var ran = 0
        # (b, nh, nkv, l, s, k): SmolLM2's decode step; GQA with ragged keys;
        # several rows per head; MHA; the widest l
        var cases: List[List[Int]] = [
            [1, 15, 5, 1, 37, 64],
            [2, 4, 2, 3, 19, 24],
            [2, 6, 3, 16, 70, 64],
            [2, 4, 4, 1, 9, 16],
            [3, 4, 1, 7, 13, 40],
        ]
        for i in range(len(cases)):
            ran += 1
            try:
                check_case(ctx, cases[i][0], cases[i][1], cases[i][2], cases[i][3], cases[i][4], cases[i][5], 101 + i)
            except e:
                failed += 1
                print("FAIL " + String(e))
        ran += 1
        try:
            check_refusals(ctx)
        except e:
            failed += 1
            print("FAIL " + String(e))
        print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
        if failed > 0:
            raise Error(String(failed) + " gate(s) failed")
