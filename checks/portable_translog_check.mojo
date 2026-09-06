# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTITY_PATHS row 12's gate: the portable transcendentals.

Three claims, three arms:

1. ACCURACY -- `portable_expf`/`portable_logf` against a float64 stdlib
   reference over 2^20 HASHED inputs each (uniform data hides permutation;
   hashed patterns are house rule): max error <= 4 ulp on the normal
   domain, with the saturation contracts (inf above 88.722835, zero below
   -87.33655, flush at the underflow edge) asserted exactly.

2. DEVICE == HOST -- the same functions run in a device kernel over the
   same inputs must produce the SAME BITS the host loop produced. The
   portable pair is basic ops and fma only, each correctly rounded on
   every measured backend (`check-ieee-arith`), so any mismatch here is a
   backend that broke the premise -- run this FIRST on every new column.
   The printed device hash is the cross-vendor certificate line: it must
   be THE SAME NUMBER on Apple, NVIDIA and AMD.

3. REACH (`sabotage` argv) -- the device arm recomputes on inputs nudged
   by one ulp-scale factor while the host arm does not; the compare must
   then FAIL on nearly every lane. A sabotage run that reports zero
   mismatches means the compare is blind and the gate is decoration.
   Sabotage PASSES by detecting mismatches.

Run: `pixi run check-portable-translog` (append `sabotage` for arm 3).
"""

from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp, log
from std.memory import bitcast
from std.sys import argv

from checks.numerics import portable_expf, portable_logf

comptime N = 1 << 20
comptime BLOCK = 256


def translog_kernel(
    xs_exp: MutPointer[Float32, MutAnyOrigin],
    xs_log: MutPointer[Float32, MutAnyOrigin],
    out_exp: MutPointer[Float32, MutAnyOrigin],
    out_log: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    sabotage_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var xe = xs_exp.unsafe_load(i)
        var xl = xs_log.unsafe_load(i)
        if sabotage_in != Int32(0):
            # the host arm computes UNNUDGED; zero mismatches after this
            # would prove the compare is blind
            xe = xe * Float32(1.0000001)
            xl = xl * Float32(1.0000001)
        out_exp.unsafe_store(i, portable_expf(xe))
        out_log.unsafe_store(i, portable_logf(xl))


def _splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _bits(x: Float32) -> UInt32:
    return rebind[UInt32](x.to_bits())


def _ord(b: UInt32) -> Int64:
    # monotone map of float bits to a total order, so ulp distance is a
    # subtraction; both zeros land adjacent, which is close enough here
    if b & UInt32(0x80000000):
        return Int64(0xFFFFFFFF) - Int64(b & UInt32(0x7FFFFFFF))
    return Int64(0x100000000) + Int64(b)


def _ulp_dist(a: Float32, b: Float32) -> Int64:
    var d = _ord(_bits(a)) - _ord(_bits(b))
    if d < Int64(0):
        return -d
    return d


def main() raises:
    var sabotage = False
    for a in argv():
        if a == "sabotage":
            sabotage = True

    var ctx = DeviceContext()

    var h_xe = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_xl = ctx.enqueue_create_host_buffer[DType.float32](N)
    var pxe = h_xe.unsafe_ptr()
    var pxl = h_xl.unsafe_ptr()

    # exp: hashed reals spanning past both saturation bounds; the first
    # slots pin the edges by hand
    pxe.unsafe_store(0, Float32(0.0))
    pxe.unsafe_store(1, Float32(1.0))
    pxe.unsafe_store(2, Float32(-1.0))
    pxe.unsafe_store(3, Float32(88.9))
    pxe.unsafe_store(4, Float32(-88.0))
    pxe.unsafe_store(5, Float32(88.722))
    pxe.unsafe_store(6, Float32(-87.33655))
    pxe.unsafe_store(7, Float32(-87.336))
    # log: positive normals with hashed mantissa AND hashed exponent; the
    # first slots pin the fold boundary and both domain ends
    pxl.unsafe_store(0, Float32(1.0))
    pxl.unsafe_store(1, Float32(0.5))
    pxl.unsafe_store(2, Float32(2.0))
    pxl.unsafe_store(3, bitcast[DType.float32](UInt32(0x00800000)))
    pxl.unsafe_store(4, bitcast[DType.float32](UInt32(0x7F7FFFFF)))
    pxl.unsafe_store(5, bitcast[DType.float32](UInt32(0x3F800001)))
    pxl.unsafe_store(6, bitcast[DType.float32](UInt32(0x3F7FFFFF)))
    pxl.unsafe_store(7, Float32(3.0))
    for i in range(8, N):
        var h1 = _splitmix(UInt64(2 * i + 1))
        var h2 = _splitmix(UInt64(2 * i + 2))
        var u = Float64(h1 & UInt64(0xFFFFFFFF)) / Float64(4294967296.0)
        pxe.unsafe_store(i, Float32(-92.0 + 184.0 * u))
        var mant = UInt32(h2 & UInt64(0x007FFFFF))
        var ef = UInt32(1) + UInt32((h2 >> 23) % UInt64(254))
        pxl.unsafe_store(i, bitcast[DType.float32](mant | (ef << 23)))

    # ---- arm 1: accuracy on the host, portable vs float64 stdlib -------
    var host_exp = ctx.enqueue_create_host_buffer[DType.float32](N)
    var host_log = ctx.enqueue_create_host_buffer[DType.float32](N)
    var phe = host_exp.unsafe_ptr()
    var phl = host_log.unsafe_ptr()
    var max_ulp_exp = Int64(0)
    var max_ulp_log = Int64(0)
    for i in range(N):
        var xe = pxe.unsafe_load(i)
        var xl = pxl.unsafe_load(i)
        var ye = portable_expf(xe)
        var yl = portable_logf(xl)
        phe.unsafe_store(i, ye)
        phl.unsafe_store(i, yl)

        if xe > Float32(88.722835):
            if _bits(ye) != UInt32(0x7F800000):
                raise Error("exp overflow contract broken at lane " + String(i))
        elif xe < Float32(-87.33655):
            if _bits(ye) != UInt32(0):
                raise Error("exp underflow contract broken at lane " + String(i))
        else:
            var ref_e = Float32(exp(Float64(xe)))
            if abs(ref_e) < Float32(1.1754943508222875e-38):
                if _bits(ye) != UInt32(0):
                    raise Error("exp edge flush broken at lane " + String(i))
            else:
                var d = _ulp_dist(ye, ref_e)
                if d > max_ulp_exp:
                    max_ulp_exp = d

        var refl = Float32(log(Float64(xl)))
        var dl = _ulp_dist(yl, refl)
        if dl > max_ulp_log:
            max_ulp_log = dl

    print("portable_expf max ulp vs float64 ref:", max_ulp_exp)
    print("portable_logf max ulp vs float64 ref:", max_ulp_log)
    if max_ulp_exp > Int64(4) or max_ulp_log > Int64(4):
        raise Error("accuracy bound (4 ulp) exceeded")

    # ---- arm 2: the device arm must reproduce the host bits ------------
    var d_xe = ctx.enqueue_create_buffer[DType.float32](N)
    var d_xl = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=d_xe, src_ptr=h_xe.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_xl, src_ptr=h_xl.unsafe_ptr())
    var o_exp = ctx.enqueue_create_buffer[DType.float32](N)
    var o_log = ctx.enqueue_create_buffer[DType.float32](N)

    ctx.enqueue_function[translog_kernel](
        d_xe.unsafe_ptr(), d_xl.unsafe_ptr(),
        o_exp.unsafe_ptr(), o_log.unsafe_ptr(),
        Int32(N), Int32(1) if sabotage else Int32(0),
        grid_dim=(N + BLOCK - 1) // BLOCK,
        block_dim=(BLOCK, 1, 1),
    )

    var r_exp = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_log = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_ptr=r_exp.unsafe_ptr(), src_buf=o_exp)
    ctx.enqueue_copy(dst_ptr=r_log.unsafe_ptr(), src_buf=o_log)
    ctx.synchronize()

    var qe = r_exp.unsafe_ptr()
    var ql = r_log.unsafe_ptr()
    var mismatches = 0
    var fnv = UInt64(0xCBF29CE484222325)
    for i in range(N):
        var be = _bits(qe.unsafe_load(i))
        var bl = _bits(ql.unsafe_load(i))
        if be != _bits(phe.unsafe_load(i)):
            mismatches += 1
        if bl != _bits(phl.unsafe_load(i)):
            mismatches += 1
        fnv = (fnv ^ UInt64(be)) * UInt64(0x100000001B3)
        fnv = (fnv ^ UInt64(bl)) * UInt64(0x100000001B3)

    print("translog device hash =", fnv)

    if sabotage:
        if mismatches == 0:
            raise Error(
                "SABOTAGE arm found ZERO mismatches: the compare is blind"
            )
        print(
            "sabotage detected as it must be:", mismatches,
            "device/host mismatches -- the compare has teeth",
        )
    else:
        if mismatches != 0:
            raise Error(
                "device bits differ from host bits on "
                + String(mismatches)
                + " lanes: this backend broke the basic-ops premise"
            )
        print("device == host on every lane; both arms OK")

    # keepalives past the last pointer reads
    _ = h_xe.unsafe_ptr()
    _ = h_xl.unsafe_ptr()
    _ = host_exp.unsafe_ptr()
    _ = host_log.unsafe_ptr()
    _ = r_exp.unsafe_ptr()
    _ = r_log.unsafe_ptr()
