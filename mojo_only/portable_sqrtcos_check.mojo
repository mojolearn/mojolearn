# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTITY_PATHS row 10 (sqrt) and row 12 (cos, pow): the second gate of
portable device arithmetic, the sibling of `check-portable-translog`.

WHY IT EXISTS: `check-ieee-arith` on the NVIDIA H100 (E2, 2026-08-23)
measured `std.math.sqrt` NOT correctly rounded -- 180,714 of 2^20 hashed
patterns one ulp off, 176,577 of them normal -- where Metal and HIP are
exact, so every Cosine split score (`score / sqrt(denum_sqr)`) carried a
different last bit on that column alone. `portable_sqrtf` is the closure;
`portable_cosf` and `portable_powf` close the last unrouted device
transcendentals on the tree paths (Box-Muller's cos in random_gen, the
Bayesian bootstrap's temperature power).

Three arms, the same three as the translog gate:

1. ACCURACY -- sqrt must match the float64 reference rounded to float32
   EXACTLY (0 mismatches: correctly rounded by construction), cos within
   2 ulp on |x| < 2*pi (Box-Muller's range) and 8 across |x| < 8192, pow
   within 64 ulp on (x in (1e-6, 50), p in (0.05, 4)) -- a float32
   composition's inherent amplification; the special-value contracts
   asserted exactly.
2. DEVICE == HOST -- the device kernel reproduces the host bits lane for
   lane; the printed device hash must be THE SAME NUMBER on every vendor
   column. Run FIRST on any new backend, beside check-portable-translog.
3. REACH (`sabotage` argv) -- device inputs nudged, compare must fail.

Run: `pixi run check-portable-sqrtcos` (append `sabotage` for arm 3).
"""

from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import cos, exp, log, sqrt
from std.memory import bitcast
from std.sys import argv

from mojo_only.numerics import portable_cosf, portable_powf, portable_sqrtf

comptime N = 1 << 20
comptime BLOCK = 256


def sqrtcos_kernel(
    xs_sqrt: MutPointer[Float32, MutAnyOrigin],
    xs_cos: MutPointer[Float32, MutAnyOrigin],
    xs_pow: MutPointer[Float32, MutAnyOrigin],
    ps_pow: MutPointer[Float32, MutAnyOrigin],
    out_sqrt: MutPointer[Float32, MutAnyOrigin],
    out_cos: MutPointer[Float32, MutAnyOrigin],
    out_pow: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    sabotage_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var xs = xs_sqrt.unsafe_load(i)
        var xc = xs_cos.unsafe_load(i)
        var xp = xs_pow.unsafe_load(i)
        var pp = ps_pow.unsafe_load(i)
        if sabotage_in != Int32(0):
            xs = xs * Float32(1.0000001)
            xc = xc * Float32(1.0000001)
            xp = xp * Float32(1.0000001)
        out_sqrt.unsafe_store(i, portable_sqrtf(xs))
        out_cos.unsafe_store(i, portable_cosf(xc))
        out_pow.unsafe_store(i, portable_powf(xp, pp))


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

    var h_xs = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_xc = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_xp = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_pp = ctx.enqueue_create_host_buffer[DType.float32](N)
    var pxs = h_xs.unsafe_ptr()
    var pxc = h_xc.unsafe_ptr()
    var pxp = h_xp.unsafe_ptr()
    var ppp = h_pp.unsafe_ptr()

    # sqrt: hashed mantissa AND exponent over the whole normal range, with
    # the edges, perfect squares and the largest finite pinned by hand
    pxs.unsafe_store(0, Float32(0.0))
    pxs.unsafe_store(1, Float32(1.0))
    pxs.unsafe_store(2, Float32(4.0))
    pxs.unsafe_store(3, Float32(2.0))
    pxs.unsafe_store(4, bitcast[DType.float32](UInt32(0x00800000)))
    pxs.unsafe_store(5, bitcast[DType.float32](UInt32(0x7F7FFFFF)))
    pxs.unsafe_store(6, Float32(0.25))
    pxs.unsafe_store(7, Float32(1e-10))
    # cos: |x| < 8192 hashed, the quadrant edges pinned
    pxc.unsafe_store(0, Float32(0.0))
    pxc.unsafe_store(1, Float32(1.5707963267948966))
    pxc.unsafe_store(2, Float32(3.141592653589793))
    pxc.unsafe_store(3, Float32(6.283185307179586))
    pxc.unsafe_store(4, Float32(-0.7853981633974483))
    pxc.unsafe_store(5, Float32(2.356194490192345))
    pxc.unsafe_store(6, Float32(0.5))
    pxc.unsafe_store(7, Float32(8191.0))
    # pow: x in (1e-6, 50) -- Bayesian's -log(u) lives in (0, 46) -- and
    # p in (0.05, 4), the temperature's realistic range
    pxp.unsafe_store(0, Float32(1.0))
    ppp.unsafe_store(0, Float32(2.0))
    pxp.unsafe_store(1, Float32(2.0))
    ppp.unsafe_store(1, Float32(0.5))
    pxp.unsafe_store(2, Float32(10.0))
    ppp.unsafe_store(2, Float32(3.0))
    pxp.unsafe_store(3, Float32(0.001))
    ppp.unsafe_store(3, Float32(1.0))
    pxp.unsafe_store(4, Float32(46.0))
    ppp.unsafe_store(4, Float32(4.0))
    pxp.unsafe_store(5, Float32(1e-6))
    ppp.unsafe_store(5, Float32(0.05))
    pxp.unsafe_store(6, Float32(3.0))
    ppp.unsafe_store(6, Float32(0.0))
    pxp.unsafe_store(7, Float32(0.0))
    ppp.unsafe_store(7, Float32(1.5))
    for i in range(8, N):
        var h1 = _splitmix(UInt64(3 * i + 1))
        var h2 = _splitmix(UInt64(3 * i + 2))
        var h3 = _splitmix(UInt64(3 * i + 3))
        var mant = UInt32(h1 & UInt64(0x007FFFFF))
        var ef = UInt32(1) + UInt32((h1 >> 23) % UInt64(254))
        pxs.unsafe_store(i, bitcast[DType.float32](mant | (ef << 23)))
        var u = Float64(h2 & UInt64(0xFFFFFFFF)) / Float64(4294967296.0)
        # half the lanes in Box-Muller's own range, half across the domain
        if (h2 >> 40) & UInt64(1) == UInt64(0):
            pxc.unsafe_store(i, Float32(6.283185307179586 * u))
        else:
            pxc.unsafe_store(i, Float32(-8192.0 + 16384.0 * u))
        var v = Float64(h3 & UInt64(0xFFFFFFFF)) / Float64(4294967296.0)
        var w = Float64((h3 >> 32) & UInt64(0xFFFFFFFF)) / Float64(4294967296.0)
        pxp.unsafe_store(i, Float32(1e-6 + 50.0 * v))
        ppp.unsafe_store(i, Float32(0.05 + 3.95 * w))

    # ---- arm 1: accuracy on the host ----------------------------------
    var host_s = ctx.enqueue_create_host_buffer[DType.float32](N)
    var host_c = ctx.enqueue_create_host_buffer[DType.float32](N)
    var host_p = ctx.enqueue_create_host_buffer[DType.float32](N)
    var phs = host_s.unsafe_ptr()
    var phc = host_c.unsafe_ptr()
    var php = host_p.unsafe_ptr()
    var sqrt_mismatch = 0
    var max_ulp_cos = Int64(0)
    var max_ulp_cos_bm = Int64(0)  # Box-Muller's range, |x| < 2*pi
    var max_ulp_pow = Int64(0)
    for i in range(N):
        var xs = pxs.unsafe_load(i)
        var ys = portable_sqrtf(xs)
        phs.unsafe_store(i, ys)
        var refs = Float32(sqrt(Float64(xs)))
        if _bits(ys) != _bits(refs):
            sqrt_mismatch += 1
            if sqrt_mismatch <= 3:
                print("  sqrt mismatch: x=", _bits(xs), " portable=", _bits(ys),
                      " ref=", _bits(refs))
        var xc = pxc.unsafe_load(i)
        var yc = portable_cosf(xc)
        phc.unsafe_store(i, yc)
        var refc = Float32(cos(Float64(xc)))
        var dc = _ulp_dist(yc, refc)
        if dc > max_ulp_cos:
            max_ulp_cos = dc
        if abs(xc) < Float32(6.2831855) and dc > max_ulp_cos_bm:
            max_ulp_cos_bm = dc
        var xp = pxp.unsafe_load(i)
        var pp = ppp.unsafe_load(i)
        var yp = portable_powf(xp, pp)
        php.unsafe_store(i, yp)
        if xp > Float32(0.0):
            var refp = Float32(exp(Float64(pp) * log(Float64(xp))))
            var dp = _ulp_dist(yp, refp)
            if dp > max_ulp_pow:
                max_ulp_pow = dp
    # special values
    if _bits(portable_sqrtf(Float32(-1.0))) & UInt32(0x7FC00000) != UInt32(0x7FC00000):
        raise Error("sqrt(-1) is not NaN")
    if _bits(portable_sqrtf(bitcast[DType.float32](UInt32(0x7F800000)))) != UInt32(0x7F800000):
        raise Error("sqrt(+inf) is not +inf")
    if _bits(portable_sqrtf(bitcast[DType.float32](UInt32(0x00400000)))) != UInt32(0):
        raise Error("sqrt(subnormal) did not flush to +0")
    if portable_powf(Float32(0.0), Float32(2.0)) != Float32(0.0):
        raise Error("pow(0, 2) != 0")
    if portable_powf(Float32(7.0), Float32(0.0)) != Float32(1.0):
        raise Error("pow(7, 0) != 1")

    print("portable_sqrtf mismatches vs correctly-rounded float64 ref:", sqrt_mismatch, "of", N)
    print("portable_cosf max ulp vs float64 ref: |x|<2pi", max_ulp_cos_bm,
          " |x|<8192", max_ulp_cos)
    print("portable_powf max ulp vs float64 exp(p*log x):", max_ulp_pow)
    if sqrt_mismatch != 0:
        raise Error("portable_sqrtf is not correctly rounded on every lane")
    if max_ulp_cos_bm > Int64(2) or max_ulp_cos > Int64(8):
        raise Error("cos accuracy bound (2 ulp on |x|<2pi, 8 across) exceeded")
    # pow is a float32 composition: exp amplifies log's last bit by
    # |p*log x| (up to ~4*17 here), so tens of ulp is what ANY float32
    # powf built this way carries; identity, not accuracy, is the property
    if max_ulp_pow > Int64(64):
        raise Error("pow accuracy bound (64 ulp) exceeded")

    # ---- arm 2: the device arm must reproduce the host bits ------------
    var d_xs = ctx.enqueue_create_buffer[DType.float32](N)
    var d_xc = ctx.enqueue_create_buffer[DType.float32](N)
    var d_xp = ctx.enqueue_create_buffer[DType.float32](N)
    var d_pp = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=d_xs, src_ptr=h_xs.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_xc, src_ptr=h_xc.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_xp, src_ptr=h_xp.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_pp, src_ptr=h_pp.unsafe_ptr())
    var o_s = ctx.enqueue_create_buffer[DType.float32](N)
    var o_c = ctx.enqueue_create_buffer[DType.float32](N)
    var o_p = ctx.enqueue_create_buffer[DType.float32](N)

    ctx.enqueue_function[sqrtcos_kernel](
        d_xs.unsafe_ptr(), d_xc.unsafe_ptr(), d_xp.unsafe_ptr(), d_pp.unsafe_ptr(),
        o_s.unsafe_ptr(), o_c.unsafe_ptr(), o_p.unsafe_ptr(),
        Int32(N), Int32(1) if sabotage else Int32(0),
        grid_dim=(N + BLOCK - 1) // BLOCK,
        block_dim=(BLOCK, 1, 1),
    )

    var r_s = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_c = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_p = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_ptr=r_s.unsafe_ptr(), src_buf=o_s)
    ctx.enqueue_copy(dst_ptr=r_c.unsafe_ptr(), src_buf=o_c)
    ctx.enqueue_copy(dst_ptr=r_p.unsafe_ptr(), src_buf=o_p)
    ctx.synchronize()

    var qs = r_s.unsafe_ptr()
    var qc = r_c.unsafe_ptr()
    var qp = r_p.unsafe_ptr()
    var mismatches = 0
    var fnv = UInt64(0xCBF29CE484222325)
    for i in range(N):
        var bs = _bits(qs.unsafe_load(i))
        var bc = _bits(qc.unsafe_load(i))
        var bp = _bits(qp.unsafe_load(i))
        if bs != _bits(phs.unsafe_load(i)):
            mismatches += 1
        if bc != _bits(phc.unsafe_load(i)):
            mismatches += 1
        if bp != _bits(php.unsafe_load(i)):
            mismatches += 1
        fnv = (fnv ^ UInt64(bs)) * UInt64(0x100000001B3)
        fnv = (fnv ^ UInt64(bc)) * UInt64(0x100000001B3)
        fnv = (fnv ^ UInt64(bp)) * UInt64(0x100000001B3)

    print("sqrtcos device hash =", fnv)

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
        print("device == host on every lane; all arms OK")

    _ = h_xs.unsafe_ptr()
    _ = h_xc.unsafe_ptr()
    _ = h_xp.unsafe_ptr()
    _ = h_pp.unsafe_ptr()
    _ = host_s.unsafe_ptr()
    _ = host_c.unsafe_ptr()
    _ = host_p.unsafe_ptr()
    _ = r_s.unsafe_ptr()
    _ = r_c.unsafe_ptr()
    _ = r_p.unsafe_ptr()
