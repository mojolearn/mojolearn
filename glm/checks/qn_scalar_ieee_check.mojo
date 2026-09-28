# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fused L-BFGS two-loop's scalar arithmetic against the host's IEEE words.

lane/linear-apple (2026-09-28): `qn_util.lbfgs_two_loop_kernel` computes
`alpha = dot / yhist`, `beta = dot / yhist` and `alpha - beta` on the device,
where the host used to. Its bits are the host's only if
`dense.ieee_div_f32` and `dense.ieee_sub_f32` return the host's `/` and `-`
words, including where the result or an operand is SUBNORMAL (a flushing
GPU ALU returns a signed zero there, and the two helpers recompute those
cases in integers). This driver runs both helpers in a kernel over pairs
built to land in every class (normal, subnormal result, underflow to zero,
subnormal operands, near-equal operands, a big operand against a subnormal
one) and compares each word with the host operation. It also counts how many
RAW hardware words differed from the host, which says whether this GPU
flushes at all.

    pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 glm/checks/qn_scalar_ieee_check.mojo

Prints `PASS qn scalar ieee: ...` or `FAIL ...` (exit 1).
"""
from std.memory import bitcast
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from glm.impl.qn.simple_mat.dense import ieee_div_f32, ieee_sub_f32


def scalar_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    res: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Res[4i..4i+3] = ieee_div, ieee_sub, raw a/b, raw a-b."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var x = a.unsafe_load(i)
        var y = b.unsafe_load(i)
        res.unsafe_store(4 * i, ieee_div_f32(x, y))
        res.unsafe_store(4 * i + 1, ieee_sub_f32(x, y))
        res.unsafe_store(4 * i + 2, x / y)
        res.unsafe_store(4 * i + 3, x - y)


struct Rng:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s


def _f(sign: UInt64, e: UInt64, frac: UInt64) -> Float32:
    var w = ((sign & 1) << 31) | ((e & 0xFF) << 23) | (frac & 0x7FFFFF)
    return bitcast[DType.float32](w.cast[DType.uint32]())


def _is_sub(x: Float32) -> Bool:
    var w = bitcast[DType.uint32](x)
    return (w & UInt32(0x7F800000)) == UInt32(0) and (w & UInt32(0x7FFFFF)) != UInt32(0)


def main() raises:
    var r = Rng(0x9E3779B97F4A7C15)
    var A = List[Float32]()
    var B = List[Float32]()
    var div_ok = List[Bool]()  # division is only claimed for non-subnormal operands
    # 1. division into the subnormal / underflow range: a small normal, b big
    for _ in range(40000):
        var ea = 1 + r.next() % 40
        var eb = 100 + r.next() % 100
        A.append(_f(r.next(), ea, r.next()))
        B.append(_f(r.next(), eb, r.next()))
        div_ok.append(True)
    # 2. division, ordinary normal range
    for _ in range(20000):
        A.append(_f(r.next(), 60 + r.next() % 130, r.next()))
        B.append(_f(r.next(), 60 + r.next() % 130, r.next()))
        div_ok.append(True)
    # 3. subtraction of tiny operands, subnormals included (fields 0..40)
    for _ in range(40000):
        A.append(_f(r.next(), r.next() % 41, r.next()))
        B.append(_f(r.next(), r.next() % 41, r.next()))
        div_ok.append(False)
    # 4. near-equal operands around the normal/subnormal boundary
    for _ in range(40000):
        var e = r.next() % 6
        var fr = r.next()
        var d = r.next() % 9
        A.append(_f(0, e, fr))
        B.append(_f(0, e, fr + d - 4))
        div_ok.append(e != 0)
    # 5. a big operand (fields 25..40) against a subnormal or zero one
    for _ in range(20000):
        A.append(_f(r.next(), 25 + r.next() % 16, r.next()))
        B.append(_f(r.next(), 0, r.next() % 3 * (r.next() & 0x7FFFFF)))
        div_ok.append(False)
    var n = len(A)
    var ctx = DeviceContext()
    var da = ctx.enqueue_create_buffer[DType.float32](n)
    var db = ctx.enqueue_create_buffer[DType.float32](n)
    var dout = ctx.enqueue_create_buffer[DType.float32](4 * n)
    ctx.enqueue_copy(dst_buf=da, src_ptr=A.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=db, src_ptr=B.unsafe_ptr())
    ctx.enqueue_function[scalar_kernel](
        da.unsafe_ptr(), db.unsafe_ptr(), dout.unsafe_ptr(), Int32(n),
        grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    var h = List[Float32](length=4 * n, fill=Float32(0.0))
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=dout)
    ctx.synchronize()
    var bad_div = 0
    var bad_sub = 0
    var raw_div = 0
    var raw_sub = 0
    var sub_results = 0
    var checked_div = 0
    for i in range(n):
        var x = A[i]
        var y = B[i]
        var q = x / y
        var d = x - y
        var bq = bitcast[DType.uint32](q)
        var bd = bitcast[DType.uint32](d)
        if _is_sub(q) or _is_sub(d):
            sub_results += 1
        if div_ok[i] and not _is_sub(x) and not _is_sub(y) and y != Float32(0.0):
            checked_div += 1
            if bitcast[DType.uint32](h[4 * i]) != bq:
                if bad_div < 5:
                    print("  div", x, "/", y, "host", q, "device", h[4 * i])
                bad_div += 1
            if bitcast[DType.uint32](h[4 * i + 2]) != bq:
                raw_div += 1
        if bitcast[DType.uint32](h[4 * i + 1]) != bd:
            if bad_sub < 5:
                print("  sub", x, "-", y, "host", d, "device", h[4 * i + 1])
            bad_sub += 1
        if bitcast[DType.uint32](h[4 * i + 3]) != bd:
            raw_sub += 1
    print(
        "pairs", n, "div checked", checked_div, "subnormal host results", sub_results,
        "| raw hardware words != host: div", raw_div, "sub", raw_sub,
    )
    _ = da^
    _ = db^
    _ = dout^
    if bad_div != 0 or bad_sub != 0:
        print("FAIL qn scalar ieee: div", bad_div, "sub", bad_sub, "words differ from the host")
        raise Error("qn scalar ieee check failed")
    print("PASS qn scalar ieee: every ieee_div_f32 / ieee_sub_f32 word equals the host's")
