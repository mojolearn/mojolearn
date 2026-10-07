# SPDX-License-Identifier: Apache-2.0
"""NN-OZ accuracy and identity check: Ozaki slices S = 3..6 against the
incumbent pinned fp32 GEMM and a float64 host reference.

Host arithmetic here is a verification-only oracle (`# cpu-route:` below),
outside the GPU runtime and any scored interval. No timing is emitted.

Per case it prints
  OZAKI_ACC fixture op m n k S=<s> nerr=<x> rerr=<x> inc_nerr=<x> inc_rerr=<x> digest=<h>
where nerr is max |C - expect| / sum_p |a||b| over the cells (normwise, the
fp32 GEMM error bound's own scale) and rerr is max |C - expect| / |expect| over
cells with |expect| >= 2^-10 * sum|a||b|. The digest is FNV-1a over the Ozaki
output bits: NVIDIA, AMD and the host column must print the same digest for
the same line (the IDENTICAL claim). The last line,
  OZAKI_CHECK S4=<worst nerr ratio vs incumbent> S5=.. S6=.. exact_ints=PASS|FAIL
summarizes. Build: gemm/experiments/native_build.py --vendor nvidia|amd|host
--mode identical.
"""
from std.math import abs
from std.memory import bitcast
from max.gpu.host import DeviceContext, HostBuffer
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.checks.gemm_step_arms import _mix, gemm_step_digest
from gemm.contract import OP_NN, OP_NT, OP_TN
from gemm.experiments.neural_ozaki import neural_ozaki_into, ozaki_admits, ozaki_workspace_floats


def _full(i: Int, salt: Int, fixture: Int) -> Float32:
    """0: full 24-bit significands over 2^-4..2^3, signed. 1: the same over
    2^-20..2^3 (wide dynamic range inside a row). 2: ReLU-like, half zeros.
    3: small integers |x| < 64 (every product and sum exact)."""
    var h = _mix(i, salt)
    var sign = UInt32((h >> UInt64(40)) & UInt64(1)) << 31
    if fixture == 3:
        var v = Float32(Int((h >> UInt64(8)) & UInt64(63)))
        return -v if sign != UInt32(0) else v
    var span = 24 if fixture == 1 else 8
    var e = UInt32(127 - (20 if fixture == 1 else 4)) + UInt32(Int((h >> UInt64(24)) & UInt64(0xFFFF)) % span)
    var bits = sign | (e << 23) | UInt32(h & UInt64(0x7FFFFF))
    if fixture == 2:
        if (h >> UInt64(41)) & UInt64(1) == UInt64(1):
            return Float32(0.0)
        bits = bits & UInt32(0x7FFFFFFF)
    return bitcast[DType.float32](bits)


def _a_at(al: List[Float32], op: Int, m: Int, k: Int, i: Int, p: Int) -> Float64:
    return Float64(al[p * m + i]) if op == OP_TN else Float64(al[i * k + p])


def _b_at(bl: List[Float32], op: Int, n: Int, k: Int, j: Int, p: Int) -> Float64:
    return Float64(bl[j * k + p]) if op == OP_NT else Float64(bl[p * n + j])


def _errors(ch: HostBuffer[DType.float32], expect: List[Float64], sab: List[Float64], count: Int) -> Tuple[Float64, Float64]:
    var nerr = Float64(0)
    var rerr = Float64(0)
    for c in range(count):
        var got = Float64(ch[c])
        var d = abs(got - expect[c])
        if got != got:
            d = Float64(1e300)
        if sab[c] > 0:
            nerr = max(nerr, d / sab[c])
            if abs(expect[c]) >= sab[c] * 0.0009765625:
                rerr = max(rerr, d / abs(expect[c]))
        elif d != 0:
            nerr = Float64(1e300)
    return (nerr, rerr)


def main() raises:
    var ctx = DeviceContext()
    # (m, n, k, op): small ragged, square, a long-k weight gradient, a
    # projection; plus a k that is not a multiple of 32.
    var shapes: List[Int] = [7, 9, 127, OP_NN, 7, 9, 127, OP_NT, 7, 9, 127, OP_TN,
                             64, 64, 1000, OP_NN, 256, 256, 4096, OP_NT,
                             8, 9, 86528, OP_TN, 16, 72, 25088, OP_TN, 4096, 256, 256, OP_NT]
    var worst = List[Float64]()
    for _ in range(7):
        worst.append(Float64(0))
    var exact_ok = True
    for si in range(len(shapes) // 4):
        var m = shapes[4 * si]
        var n = shapes[4 * si + 1]
        var k = shapes[4 * si + 2]
        var op = shapes[4 * si + 3]
        var ah = ctx.enqueue_create_host_buffer[DType.float32](m * k)
        var bh = ctx.enqueue_create_host_buffer[DType.float32](n * k)
        var ch = ctx.enqueue_create_host_buffer[DType.float32](m * n)
        var a = ctx.enqueue_create_buffer[DType.float32](m * k)
        var b = ctx.enqueue_create_buffer[DType.float32](n * k)
        var c = ctx.enqueue_create_buffer[DType.float32](m * n)
        var wsn = identical_gemm_workspace_max_floats(m, n, k)
        wsn = max(wsn, ozaki_workspace_floats[3](m, n, k))
        wsn = max(wsn, ozaki_workspace_floats[6](m, n, k))
        var ws = ctx.enqueue_create_buffer[DType.float32](wsn)
        ctx.synchronize()
        for fixture in range(4):
            var al = List[Float32]()
            var bl = List[Float32]()
            for x in range(m * k):
                var v = _full(x, 17 + si, fixture)
                ah[x] = v
                al.append(v)
            for x in range(n * k):
                var v = _full(x, 91 + si, fixture)
                bh[x] = v
                bl.append(v)
            ctx.enqueue_copy(dst_buf=a, src_buf=ah)
            ctx.enqueue_copy(dst_buf=b, src_buf=bh)
            ctx.synchronize()
            # cpu-route: qualification-only float64 reference and |a||b| scale.
            var expect = List[Float64]()
            var sab = List[Float64]()
            for i in range(m):
                for j in range(n):
                    var s = Float64(0)
                    var t = Float64(0)
                    for p in range(k):
                        var x = _a_at(al, op, m, k, i, p)
                        var y = _b_at(bl, op, n, k, j, p)
                        s += x * y
                        t += abs(x * y)
                    expect.append(s)
                    sab.append(t)
            identical_gemm_into(ctx, c, a, b, ws, m, n, k, op)
            ctx.enqueue_copy(dst_ptr=ch.unsafe_ptr(), src_buf=c)
            ctx.synchronize()
            var inc = _errors(ch, expect, sab, m * n)
            comptime for S in range(3, 7):
                var line = ("OZAKI_ACC fixture=" + String(fixture) + " op=" + String(op) + " m=" + String(m)
                    + " n=" + String(n) + " k=" + String(k) + " S=" + String(S))
                if not ozaki_admits[S](m, n, k):
                    print(line + " SKIP k_above_exact_bound")
                else:
                    _ = neural_ozaki_into[S](ctx, c, a, b, ws, m, n, k, op)
                    ctx.enqueue_copy(dst_ptr=ch.unsafe_ptr(), src_buf=c)
                    ctx.synchronize()
                    var oz = _errors(ch, expect, sab, m * n)
                    if fixture == 3:
                        # Small integers: the exact sum, rounded once, is the answer.
                        for x in range(m * n):
                            if bitcast[DType.uint32](ch[x]) != bitcast[DType.uint32](Float32(expect[x])):
                                exact_ok = False
                    if fixture != 3:
                        worst[S] = max(worst[S], oz[0] / max(inc[0], Float64(5.9604644775390625e-08)))
                    print(line + " nerr=" + String(oz[0]) + " rerr=" + String(oz[1]) + " inc_nerr=" + String(inc[0])
                        + " inc_rerr=" + String(inc[1]) + " digest=" + hex(gemm_step_digest(ch, m * n)))
    print("OZAKI_CHECK S3=" + String(worst[3]) + " S4=" + String(worst[4]) + " S5=" + String(worst[5])
        + " S6=" + String(worst[6]) + " exact_ints=" + ("PASS" if exact_ok else "FAIL")
        + " (worst nerr / max(incumbent nerr, 2^-24))")
