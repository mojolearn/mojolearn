# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTITY_PATHS row 220 (DEVIATION 5410): the normal CDF and its inverse,
`identical_ndtr` / `identical_ndtri` in checks/numerics.mojo, the certificate
the way `check-division` certifies `portable_divf`.

    tools/with_identical_mode.sh pixi run mojo run -I . checks/ndtri_check.mojo
    (tools/with_identical_mode.sh pixi run check-ndtri; a FAST build is refused:
    its ops are the stdlib's, which differ by vendor by design)

## 2^20 hashed inputs per function, in classes (lane i's class is a function
## of i, so every column sees the same inputs)

ndtri, p:  0 uniform (0, 1) on the 2^-24 grid   1 the central branch
           |p - 1/2| <= 0.425                   2 the lower tail, p in
           [2^-125, 1/2) (exponent hashed)       3 the upper tail, 1 - k 2^-24
           4 the branch edges (|q| = 0.425 and r = 5 +- a few ulps)
           5 raw hashed words (NaN, inf, negatives, above 1, subnormals)
ndtr, x:   0 uniform [-6, 6]   1 [-1, 1]   2 the tails [-14, -5] and [5, 14]
           3 ndtri of class-0 p (the round trip)   4 raw hashed words

## What is judged

  A. HOST == DEVICE, bit for bit, on every lane of both functions (the seam's
     identity). The two FNV-1a hashes are THE CERTIFICATE LINES, the same
     number on every column (NaN is canonical by construction, not by this
     check). Without an accelerator the host half runs alone.
  B. ACCURACY against a float64 oracle: AS 241 PPND16 refined by two Newton
     steps on `0.5 erfc(-x / sqrt 2)` (host libm float64, an oracle only,
     never shipped), per class: the largest ulp distance of ndtri in (0, 1)
     and the largest absolute error of ndtr. Bounded: ndtri <= NDTRI_ULP_BOUND
     ulps everywhere (near p = 1/2 too: q = p - 1/2 is exact there and
     PPND7's relative accuracy carries), ndtr <= NDTR_ABS_BOUND absolute. A wrong coefficient,
     a swapped branch or a lost term leaves these bounds by orders of
     magnitude (the sabotage arm x_prep/seams/sabotage/seam_5410_ndtri.patch
     moves one PPND7 coefficient by one part in 10^3 and FAILS here).
  C. THE SEPARATING FIXTURE, exact words: ndtri(0.5) = +0, ndtri(0) = -inf,
     ndtri(1) = +inf, ndtri(NaN), ndtri(-0.25), ndtri(1.5) = the canonical NaN,
     ndtr(+-inf) = 1 / 0, ndtr(0) = 0.5, and the symmetry
     ndtri(1 - p) = -ndtri(p) on the 2^-24 grid (a sign or branch fault breaks
     it where the accuracy bound might not).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import erfc, exp, log, sqrt
from std.memory import bitcast
from std.os import getenv
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, identical_ndtr, identical_ndtri, numeric_mode_name
from core.identity_trace import IdentityTrace

comptime N = 1 << 20
comptime BLOCK = 256
comptime NDTRI_ULP_BOUND = 8
comptime NDTR_ABS_BOUND = Float64(1.5e-7)


def ndtri_kernel(
    p: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    out_q: MutPointer[Float32, MutAnyOrigin],
    out_c: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        out_q.unsafe_store(i, identical_ndtri(p.unsafe_load(i)))
        out_c.unsafe_store(i, identical_ndtr(x.unsafe_load(i)))


def _splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _b(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _f(b: UInt32) -> Float32:
    return bitcast[DType.float32](b)


def _u24(h: UInt64) -> Float32:
    """(k + 1/2) 2^-24 for the top 24 bits k: exact, in (0, 1)."""
    return (Float32(Int(h >> 40)) + Float32(0.5)) * Float32(5.9604644775390625e-08)


def _ord(b: UInt32) -> Int64:
    if (b & UInt32(0x80000000)) != UInt32(0):
        return -Int64(b & UInt32(0x7FFFFFFF))
    return Int64(b)


def _ulps(a: Float32, b: Float32) -> Int64:
    return abs(_ord(_b(a)) - _ord(_b(b)))


def _fnv(h: UInt64, w: UInt32) -> UInt64:
    var x = h
    for k in range(4):
        x = (x ^ UInt64((w >> UInt32(8 * k)) & UInt32(0xFF))) * UInt64(0x100000001B3)
    return x


# ---------------------------------------------------------------- the oracle
def _ppnd16(p: Float64) -> Float64:
    """AS 241 PPND16 (Wichura 1988), float64, host only."""
    var q = p - 0.5
    if abs(q) <= 0.425:
        var r = 0.180625 - q * q
        var num = (((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r
                    + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
                    + 133.14166789178437745) * r + 3.387132872796366608)
        var den = (((((((5226.495278852545461 * r + 28729.085735721942674) * r + 39307.89580009271061) * r
                    + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r
                    + 42.313330701600911252) * r + 1.0)
        return q * num / den
    var r = p if q < 0.0 else 1.0 - p
    r = sqrt(-log(r))
    var v: Float64
    if r <= 5.0:
        r -= 1.6
        var num = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
                    + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r
                    + 4.6303378461565452959) * r + 1.42343711074968357734)
        var den = (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r
                    + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r
                    + 2.05319162663775882187) * r + 1.0)
        v = num / den
    else:
        r -= 5.0
        var num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r
                    + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r
                    + 5.4637849111641143699) * r + 6.6579046435011037772)
        var den = (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r
                    + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
                    + 0.59983220655588793769) * r + 1.0)
        v = num / den
    return -v if q < 0.0 else v


def _ndtr64(x: Float64) -> Float64:
    return 0.5 * erfc(-x * 0.70710678118654752440)


def _oracle_ndtri(p: Float64) -> Float64:
    """PPND16 on the smaller tail, two Newton steps on ndtr, the sign by
    symmetry (1 - p is exact for a float32 p >= 1/2)."""
    if p > 0.5:
        return -_oracle_ndtri(1.0 - p)
    var q = _ppnd16(p)
    for _ in range(2):
        var dens = exp(-0.5 * q * q) * 0.39894228040143267794
        if dens > 0.0:
            q = q - (_ndtr64(q) - p) / dens
    return q


def _gen_p(i: Int) -> Float32:
    var h = _splitmix(UInt64(2 * i + 1))
    var k = i % 6
    if k == 0:
        return _u24(h)
    if k == 1:
        return Float32(0.075) + Float32(0.85) * _u24(h)
    if k == 2:
        var e = 2 + Int((h >> 8) % UInt64(124))  # 2^-126 .. 2^-3
        var m = UInt32(h & UInt64(0x007FFFFF))
        return _f((UInt32(127 - e) << 23) | m)
    if k == 3:
        var kk = 1 + Int((h >> 20) % UInt64(1 << 22))
        return Float32(1.0) - Float32(kk) * Float32(5.9604644775390625e-08)
    if k == 4:
        var edges: List[Float32] = [Float32(0.075), Float32(0.925), Float32(1.3887943864964021e-11)]
        var base = edges[Int(h % UInt64(3))]
        var step = Int((h >> 8) % UInt64(17)) - 8
        return _f(UInt32(Int(_b(base)) + step))
    return _f(UInt32(h & UInt64(0xFFFFFFFF)))


def _gen_x(i: Int, p0: Float32) -> Float32:
    var h = _splitmix(UInt64(2 * i + 2))
    var k = i % 5
    var u = _u24(h)
    if k == 0:
        return Float32(12.0) * u - Float32(6.0)
    if k == 1:
        return Float32(2.0) * u - Float32(1.0)
    if k == 2:
        var t = Float32(5.0) + Float32(9.0) * u
        return -t if (h & UInt64(1)) == UInt64(1) else t
    if k == 3:
        return identical_ndtri(p0)
    return _f(UInt32(h & UInt64(0xFFFFFFFF)))


def _require(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL ndtri check: " + what)


def main() raises:
    print("ndtri check (DEVIATION 5410, IDENTITY_PATHS row 220); build mode", numeric_mode_name())
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("ndtri check: build under NUMERIC_IDENTICAL (tools/with_identical_mode.sh); this build is "
                    + numeric_mode_name())
    var card = IdentityTrace.to_path(getenv("MOJOLEARN_NDTRI_CARD", "/tmp/ndtri_check.card"))
    card.header("ndtri / ndtr seam (DEVIATION 5410)")

    # ---- C: the separating fixture, exact words --------------------------
    var qnan = UInt32(0x7FC00000)
    _require(_b(identical_ndtri(Float32(0.5))) == UInt32(0), "ndtri(0.5) is not +0")
    _require(_b(identical_ndtri(Float32(0.0))) == UInt32(0xFF800000), "ndtri(0) is not -inf")
    _require(_b(identical_ndtri(Float32(1.0))) == UInt32(0x7F800000), "ndtri(1) is not +inf")
    _require(_b(identical_ndtri(_f(UInt32(0x7FA00001)))) == qnan, "ndtri(NaN) is not the canonical NaN")
    _require(_b(identical_ndtri(Float32(-0.25))) == qnan, "ndtri(-0.25) is not the canonical NaN")
    _require(_b(identical_ndtri(Float32(1.5))) == qnan, "ndtri(1.5) is not the canonical NaN")
    _require(_b(identical_ndtr(_f(UInt32(0x7F800000)))) == _b(Float32(1.0)), "ndtr(+inf) is not 1")
    _require(_b(identical_ndtr(_f(UInt32(0xFF800000)))) == UInt32(0), "ndtr(-inf) is not +0")
    _require(_b(identical_ndtr(Float32(0.0))) == _b(Float32(0.5)), "ndtr(0) is not 0.5")
    _require(_b(identical_ndtr(_f(UInt32(0xFFC12345)))) == qnan, "ndtr(NaN) is not the canonical NaN")
    var asym = 0
    for k in range(1, 1 << 23, 97):
        var p = Float32(k) * Float32(5.9604644775390625e-08)
        if _b(identical_ndtri(Float32(1.0) - p)) != _b(-identical_ndtri(p)):
            asym += 1
    _require(asym == 0, "ndtri(1 - p) != -ndtri(p) on " + String(asym) + " grid points")
    print("PASS C: the fixture's exact words (0.5, 0, 1, NaN, out of range, +-inf) and the symmetry on the 2^-24 grid")

    # ---- the inputs and the host run -------------------------------------
    var hp = List[Float32](capacity=N)
    var hx = List[Float32](capacity=N)
    for i in range(N):
        var pv = _gen_p(i)
        hp.append(pv)
        hx.append(_gen_x(i, _u24(_splitmix(UInt64(2 * i + 1)))))
    var host_q = List[Float32](capacity=N)
    var host_c = List[Float32](capacity=N)
    var fq = UInt64(0xCBF29CE484222325)
    var fc = UInt64(0xCBF29CE484222325)
    for i in range(N):
        var q = identical_ndtri(hp[i])
        var c = identical_ndtr(hx[i])
        host_q.append(q)
        host_c.append(c)
        fq = _fnv(fq, _b(q))
        fc = _fnv(fc, _b(c))

    # ---- B: accuracy against the float64 oracle --------------------------
    var worst_ulp = List[Int64](length=6, fill=Int64(0))
    var worst_abs = List[Float64](length=5, fill=Float64(0))
    var judged = 0
    for i in range(N):
        var pv = hp[i]
        var bits = _b(pv)
        # judged: normal p strictly inside (0, 1)
        if pv > Float32(0) and pv < Float32(1) and (bits & UInt32(0x7F800000)) != UInt32(0):
            var o = _oracle_ndtri(Float64(pv))
            var got = host_q[i]
            judged += 1
            var u = _ulps(got, Float32(o))
            if u > worst_ulp[i % 6]:
                worst_ulp[i % 6] = u
            _require(u <= NDTRI_ULP_BOUND, "ndtri(" + String(pv) + ") = " + String(got) + ", oracle "
                     + String(o) + ": " + String(u) + " ulps > " + String(NDTRI_ULP_BOUND))
        var xv = hx[i]
        if xv == xv and abs(xv) < Float32(3.0e38):
            var e = abs(Float64(host_c[i]) - _ndtr64(Float64(xv)))
            if e > worst_abs[i % 5]:
                worst_abs[i % 5] = e
            _require(e <= NDTR_ABS_BOUND, "ndtr(" + String(xv) + ") = " + String(host_c[i]) + " is "
                     + String(e) + " from the oracle")
    print("PASS B: ndtri on", judged, "normal p in (0, 1): worst ulps per class (uniform, central, lower tail,"
          " upper tail, edges, raw):", worst_ulp[0], worst_ulp[1], worst_ulp[2], worst_ulp[3], worst_ulp[4],
          worst_ulp[5], "(bound", NDTRI_ULP_BOUND, ")")
    print("PASS B: ndtr worst absolute error per class (uniform, [-1,1], tails, round trip, raw):", worst_abs[0],
          worst_abs[1], worst_abs[2], worst_abs[3], worst_abs[4], "(bound", NDTR_ABS_BOUND, ")")

    # ---- A: host == device -----------------------------------------------
    comptime if has_accelerator():
        var ctx = DeviceContext()
        var dp = ctx.enqueue_create_buffer[DType.float32](N)
        var dx = ctx.enqueue_create_buffer[DType.float32](N)
        var dq = ctx.enqueue_create_buffer[DType.float32](N)
        var dc = ctx.enqueue_create_buffer[DType.float32](N)
        ctx.enqueue_copy(dst_buf=dp, src_ptr=hp.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
        ctx.enqueue_function[ndtri_kernel](
            dp.unsafe_ptr(), dx.unsafe_ptr(), dq.unsafe_ptr(), dc.unsafe_ptr(), Int32(N),
            grid_dim=(N + BLOCK - 1) // BLOCK,
            block_dim=(BLOCK, 1, 1),
        )
        var rq = List[Float32](length=N, fill=Float32(0))
        var rc = List[Float32](length=N, fill=Float32(0))
        ctx.enqueue_copy(dst_ptr=rq.unsafe_ptr(), src_buf=dq)
        ctx.enqueue_copy(dst_ptr=rc.unsafe_ptr(), src_buf=dc)
        ctx.synchronize()
        var mq = 0
        var mc = 0
        for i in range(N):
            if _b(rq[i]) != _b(host_q[i]):
                mq += 1
                if mq <= 4:
                    print("  ndtri DEVICE != HOST: p=", hex(_b(hp[i])), " device", hex(_b(rq[i])), " host",
                          hex(_b(host_q[i])))
            if _b(rc[i]) != _b(host_c[i]):
                mc += 1
                if mc <= 4:
                    print("  ndtr DEVICE != HOST: x=", hex(_b(hx[i])), " device", hex(_b(rc[i])), " host",
                          hex(_b(host_c[i])))
        _ = dp^
        _ = dx^
        _ = dq^
        _ = dc^
        _ = ctx^
        _require(mq == 0 and mc == 0, "device != host on " + String(mq) + " ndtri and " + String(mc) + " ndtr lanes")
        print("PASS A: device == host on all", N, "lanes of both functions")
    else:
        print("A: no accelerator on this box: the host half only")

    var words: List[Float32] = [
        identical_ndtri(Float32(0.025)), identical_ndtri(Float32(0.975)), identical_ndtri(Float32(1.0e-6)),
        identical_ndtr(Float32(-1.959964)), identical_ndtr(Float32(1.0)),
    ]
    card.record_list_f32("5410.ndtri_ndtr.fixture", words)
    print("CERTIFICATE ndtri fnv1a", hex(fq), "ndtr fnv1a", hex(fc), "(", N, "lanes each; the same on every column)")
    print("PASS ndtri check")
