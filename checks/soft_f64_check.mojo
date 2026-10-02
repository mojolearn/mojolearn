"""Gate for `checks/soft_f64.mojo` (lane hr2-gbdt-host).

HOST: every soft operation against the host's hardware double over random
bit patterns (all exponents, subnormals, zeros, infinities) plus
link-range values: add, sub, mul, div, fma, floor, the f32 narrowing, and
`sf64_exp` / `sf64_log` against `portable_exp64` / `portable_log64`.
DEVICE: the same operand set through one kernel on the GPU, compared word
for word with the host soft result (`same bits on every column`).
Prints `SOFT_F64 host_mismatch=<n> device_mismatch=<n> status=PASS|FAIL`.
`-D SOFT_F64_SABOTAGE=1` flips one rounding so the gate must FAIL.
"""
from std.math import floor, fma
from std.memory import bitcast
from std.sys import has_accelerator
from std.sys.compile import is_defined
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from checks.numerics import portable_exp64, portable_log64
from checks.soft_f64 import (
    sf64_add,
    sf64_div,
    sf64_exp,
    sf64_floor,
    sf64_fma,
    sf64_log,
    sf64_mul,
    sf64_sub,
    sf64_to_f32,
)

comptime N_OPS = 9


struct _Rng:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> UInt64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        var z = self.s
        z = (z ^ (z >> 33)) * 0xFF51AFD7ED558CCD
        z = (z ^ (z >> 33)) * 0xC4CEB9FE1A85EC53
        return z ^ (z >> 33)


def _operand(mut r: _Rng) -> UInt64:
    var k = r.next() % 8
    var v = r.next()
    if k == 0:
        return v  # any word
    if k == 1:
        return v & UInt64(0x800FFFFFFFFFFFFF)  # subnormal / zero
    if k == 2:
        # exponents near 1.0
        var e = UInt64(1023 - 30) + (r.next() % 60)
        return (v & UInt64(0x800FFFFFFFFFFFFF)) | (e << 52)
    if k == 3:
        var e = r.next() % 4
        return (v & UInt64(0x800FFFFFFFFFFFFF)) | (UInt64(2045 + e % 3) << 52)
    if k == 4:
        # few mantissa bits (ties, exact results)
        var e = UInt64(1000) + (r.next() % 48)
        return (v & UInt64(0x800FF00000000000)) | (e << 52)
    if k == 5:
        # link range: |x| < 800
        var f = Float64(Int(v % 1600000)) / 1000.0 - 800.0
        return bitcast[DType.uint64](f)
    if k == 6:
        var e = UInt64(1) + (r.next() % 3)
        return (v & UInt64(0x800FFFFFFFFFFFFF)) | (e << 52)
    var specials = SIMD[DType.uint64, 8](
        0, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000,
        0x3FF0000000000000, 0xBFF0000000000000, 0x0000000000000001,
        0x7FEFFFFFFFFFFFFF,
    )
    return specials[Int(v % 8)]


@always_inline
def _soft(op: Int, a: UInt64, b: UInt64, c: UInt64) -> UInt64:
    if op == 0:
        return sf64_add(a, b)
    if op == 1:
        return sf64_sub(a, b)
    if op == 2:
        return sf64_mul(a, b)
    if op == 3:
        return sf64_div(a, b)
    if op == 4:
        return sf64_fma(a, b, c)
    if op == 5:
        return sf64_floor(a)
    if op == 6:
        return UInt64(bitcast[DType.uint32](sf64_to_f32(a)))
    if op == 7:
        return sf64_exp(a)
    return sf64_log(a)


def _native(op: Int, a: UInt64, b: UInt64, c: UInt64) -> UInt64:
    var x = bitcast[DType.float64](a)
    var y = bitcast[DType.float64](b)
    var z = bitcast[DType.float64](c)
    var r: Float64
    if op == 0:
        r = x + y
    elif op == 1:
        r = x - y
    elif op == 2:
        r = x * y
    elif op == 3:
        r = x / y
    elif op == 4:
        r = fma(x, y, z)
    elif op == 5:
        r = floor(x)
    elif op == 6:
        return UInt64(bitcast[DType.uint32](Float32(x)))
    elif op == 7:
        r = portable_exp64(x)
    else:
        r = portable_log64(x)
    return bitcast[DType.uint64](r)


@always_inline
def _canon(op: Int, w: UInt64) -> UInt64:
    """Every NaN to one word (the soft library returns the canonical one)."""
    if op == 6:
        if (w & 0x7FFFFFFF) > 0x7F800000:
            return UInt64(0x7FC00000)
        return w
    if (w & UInt64(0x7FFFFFFFFFFFFFFF)) > UInt64(0x7FF0000000000000):
        return UInt64(0x7FF8000000000000)
    return w


def _soft_kernel(
    ops: MutPointer[Int32, MutAnyOrigin],
    a: MutPointer[UInt64, MutAnyOrigin],
    b: MutPointer[UInt64, MutAnyOrigin],
    c: MutPointer[UInt64, MutAnyOrigin],
    dst: MutPointer[UInt64, MutAnyOrigin],
    n: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst[i] = _soft(Int(ops[i]), a[i], b[i], c[i])


def main() raises:
    comptime PER_OP = 200000
    var n = PER_OP * N_OPS
    var r = _Rng(20261002)
    var ops = List[Int32](capacity=n)
    var av = List[UInt64](capacity=n)
    var bv = List[UInt64](capacity=n)
    var cv = List[UInt64](capacity=n)
    for op in range(N_OPS):
        for _ in range(PER_OP):
            ops.append(Int32(op))
            av.append(_operand(r))
            bv.append(_operand(r))
            cv.append(_operand(r))
    var host = List[UInt64](capacity=n)
    var host_bad = 0
    var shown = 0
    for i in range(n):
        var op = Int(ops[i])
        var s = _soft(op, av[i], bv[i], cv[i])
        comptime if is_defined["SOFT_F64_SABOTAGE"]():
            if i == 7:
                s ^= 1
        host.append(s)
        var want = _native(op, av[i], bv[i], cv[i])
        if _canon(op, s) != _canon(op, want):
            host_bad += 1
            if shown < 12:
                shown += 1
                print("MISMATCH op", op, "a", hex(av[i]), "b", hex(bv[i]),
                      "c", hex(cv[i]), "soft", hex(s), "native", hex(want))
    var dev_bad = -1
    comptime if has_accelerator():
        var ctx = DeviceContext()
        var d_ops = ctx.enqueue_create_buffer[DType.int32](n)
        var d_a = ctx.enqueue_create_buffer[DType.uint64](n)
        var d_b = ctx.enqueue_create_buffer[DType.uint64](n)
        var d_c = ctx.enqueue_create_buffer[DType.uint64](n)
        var d_o = ctx.enqueue_create_buffer[DType.uint64](n)
        ctx.enqueue_copy(dst_buf=d_ops, src_ptr=ops.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_a, src_ptr=av.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_b, src_ptr=bv.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_c, src_ptr=cv.unsafe_ptr())
        ctx.enqueue_function[_soft_kernel](
            d_ops.unsafe_ptr(), d_a.unsafe_ptr(), d_b.unsafe_ptr(),
            d_c.unsafe_ptr(), d_o.unsafe_ptr(), Int32(n),
            grid_dim=(n + 255) // 256, block_dim=256,
        )
        var got = List[UInt64](length=n, fill=0)
        ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=d_o)
        ctx.synchronize()
        dev_bad = 0
        for i in range(n):
            if got[i] != host[i]:
                dev_bad += 1
                if dev_bad <= 5:
                    print("DEVICE DIFF op", ops[i], "a", hex(av[i]), "host",
                          hex(host[i]), "device", hex(got[i]))
    var ok = host_bad == 0 and dev_bad <= 0
    print(
        "SOFT_F64 host_mismatch=" + String(host_bad) + " device_mismatch="
        + String(dev_bad) + " status=" + ("PASS" if ok else "FAIL")
    )
