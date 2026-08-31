# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTITY_PATHS rows 50-54: the neural-block primitives (DEVIATIONS
741-745) -- rsqrt, log1p, sigmoid, silu, softplus -- the third gate of
portable device arithmetic, sibling of `check-portable-translog` and
`check-portable-sqrtcos`.

Three arms, the same three as the other two gates, over 2^20 hashed
inputs PER FUNCTION:

1. ACCURACY, host -- each function against a float64 reference (libm's
   double `log1p`/`exp` through FFI, `sqrt` in hardware; NOT
   `std.math.log` on the host, which `portable_log64_check.mojo` measured
   as an approximation; the activations' inner exp is the double exp
   rounded once to float32 and flushed, `_expf_ideal`, because a float32
   composition saturates where a double does not), ulp distance on the
   finite domain, bounds:
   rsqrt <= 1 ulp (pin (a) is two correctly rounded ops, so it sits within
   one ulp of the correctly rounded answer), log1p <= 2, sigmoid/silu/
   softplus <= 4 (float32 compositions of two certified functions and one
   division). Every planted special value (row 39: signed zeros, inf,
   NaN, the saturation edges, the softplus guard boundary) asserted by
   bit.
2. DEVICE == HOST -- one kernel evaluates all five on the same inputs; the
   device bits must equal the host bits on every lane (NaN compared as
   NaN, the H100 payload lesson). The printed `nn device hash` is the
   cross-vendor certificate line (NaN canonicalized to 0x7FC00000 before
   hashing so a payload cannot move it): THE SAME NUMBER on Apple, NVIDIA
   and AMD, or the column broke the basic-ops premise. Per-function
   hashes are printed too, so a leg can name which function moved.
3. REACH (`sabotage` argv) -- the device arm computes the STDLIB spelling
   of each function instead (the vendor's rsqrt intrinsic, `log(1+x)`,
   `exp`), the host arm stays portable; the compare must then FAIL on
   many lanes per function. A sabotage run that reports zero mismatches
   for any function means that function's compare is blind.

Run: `pixi run check-portable-nn` (append `sabotage` for arm 3).
"""

from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp, log, rsqrt, sqrt
from std.memory import bitcast
from std.sys import argv
from std.ffi import external_call

from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    portable_log1pf,
    portable_rsqrtf,
    portable_sigmoidf,
    portable_siluf,
    portable_softplusf,
    portable_sqrtf,
    numeric_mode_name,
)

comptime N = 1 << 20
comptime BLOCK = 256
comptime NF = 5  # rsqrt, log1p, sigmoid, silu, softplus


def nn_kernel(
    xs_rsqrt: MutPointer[Float32, MutAnyOrigin],
    xs_log1p: MutPointer[Float32, MutAnyOrigin],
    xs_act: MutPointer[Float32, MutAnyOrigin],
    out_rsqrt: MutPointer[Float32, MutAnyOrigin],
    out_log1p: MutPointer[Float32, MutAnyOrigin],
    out_sigmoid: MutPointer[Float32, MutAnyOrigin],
    out_silu: MutPointer[Float32, MutAnyOrigin],
    out_softplus: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    sabotage_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var xr = xs_rsqrt.unsafe_load(i)
        var xl = xs_log1p.unsafe_load(i)
        var xa = xs_act.unsafe_load(i)
        if sabotage_in != Int32(0):
            # ARM 3: the vendor's own spellings -- the host stays portable
            out_rsqrt.unsafe_store(i, rsqrt(xr))
            # not std.math.log1p: it lowers through float64 and Metal
            # refuses it (see identical_log1p); Triton 3's spelling
            out_log1p.unsafe_store(i, log(Float32(1.0) + xl))
            var e = exp(-xa)
            out_sigmoid.unsafe_store(i, Float32(1.0) / (Float32(1.0) + e))
            out_silu.unsafe_store(i, xa / (Float32(1.0) + e))
            if xa <= Float32(20.0):
                out_softplus.unsafe_store(i, log(exp(xa) + Float32(1.0)))
            else:
                out_softplus.unsafe_store(i, xa)
        else:
            out_rsqrt.unsafe_store(i, portable_rsqrtf(xr))
            out_log1p.unsafe_store(i, portable_log1pf(xl))
            out_sigmoid.unsafe_store(i, portable_sigmoidf(xa))
            out_silu.unsafe_store(i, portable_siluf(xa))
            out_softplus.unsafe_store(i, portable_softplusf(xa))


def _splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _bits(x: Float32) -> UInt32:
    return rebind[UInt32](x.to_bits())


def _f(b: UInt32) -> Float32:
    return bitcast[DType.float32](b)


def _is_nan(b: UInt32) -> Bool:
    return (b & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000)


def _canon(b: UInt32) -> UInt32:
    """NaN payloads are the vendor's (Metal 0x7fc00000, H100 0x7fffffff):
    canonicalize before hashing and comparing so a payload is not a
    mismatch and cannot move the certificate."""
    if _is_nan(b):
        return UInt32(0x7FC00000)
    return b


def _same(a: UInt32, b: UInt32) -> Bool:
    return _canon(a) == _canon(b)


def _ord(b: UInt32) -> Int64:
    if b & UInt32(0x80000000):
        return Int64(0xFFFFFFFF) - Int64(b & UInt32(0x7FFFFFFF))
    return Int64(0x100000000) + Int64(b)


def _ulp_dist(a: Float32, b: Float32) -> Int64:
    var d = _ord(_bits(a)) - _ord(_bits(b))
    if d < Int64(0):
        return -d
    return d


def _finite(b: UInt32) -> Bool:
    return (b & UInt32(0x7F800000)) != UInt32(0x7F800000)


# ---- the float64 references: libm through FFI (see the module docstring)
def _log1p64(x: Float64) -> Float64:
    return external_call["log1p", Float64](x)


def _exp64(x: Float64) -> Float64:
    return external_call["exp", Float64](x)


def _expf_ideal(x: Float64) -> Float64:
    """The reference's inner exp for the three activations: libm's double
    exp ROUNDED ONCE TO FLOAT32 and row-10 flushed (so above 88.72284 it
    is +inf and below -87.33654 it is 0, exactly as any float32 `expf`
    saturates), carried back to double for the exact rest of the
    composition. A float64 exp that does not overflow where float32 must
    would call silu(-88.7) = -2.6e-37 "right" when every float32 silu --
    the reference kernel's `expf` included -- returns -0.0 there."""
    var v = Float32(external_call["exp", Float64](x))
    if abs(v) < Float32(1.1754943508222875e-38):
        v = Float32(0.0)
    return Float64(v)


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _expect(name: String, got: Float32, want_bits: UInt32) raises:
    if not _same(_bits(got), want_bits):
        raise Error(
            name + ": planted value wrong, got " + hex(_bits(got))
            + " want " + hex(want_bits)
        )


def main() raises:
    var sabotage = False
    for a in argv():
        if a == "sabotage":
            sabotage = True

    print("portable nn check; build mode", _mode_name(),
          "(the portable_* functions do not depend on it)")

    # ---- PLANTED SPECIAL VALUES (row 39), asserted by bit on the host --
    comptime PINF = UInt32(0x7F800000)
    comptime NINF = UInt32(0xFF800000)
    comptime QNAN = UInt32(0x7FC00000)
    comptime PZ = UInt32(0x00000000)
    comptime NZ = UInt32(0x80000000)
    comptime ONE = UInt32(0x3F800000)
    comptime HALF = UInt32(0x3F000000)
    var sub = _f(UInt32(0x00400000))  # a subnormal
    var nsub = _f(UInt32(0x80400000))  # a negative subnormal
    # rsqrt
    _expect("rsqrt(NaN)", portable_rsqrtf(_f(QNAN)), QNAN)
    _expect("rsqrt(-1)", portable_rsqrtf(Float32(-1.0)), QNAN)
    _expect("rsqrt(-inf)", portable_rsqrtf(_f(NINF)), QNAN)
    _expect("rsqrt(+0)", portable_rsqrtf(_f(PZ)), PINF)
    _expect("rsqrt(-0)", portable_rsqrtf(_f(NZ)), PINF)  # sqrt flushes -0 to +0
    _expect("rsqrt(subnormal)", portable_rsqrtf(sub), PINF)
    _expect("rsqrt(-subnormal)", portable_rsqrtf(nsub), PINF)  # flushed BEFORE the sign test
    _expect("sqrt(-subnormal)", portable_sqrtf(nsub), PZ)
    _expect("rsqrt(+inf)", portable_rsqrtf(_f(PINF)), PZ)
    _expect("rsqrt(1)", portable_rsqrtf(Float32(1.0)), ONE)
    _expect("rsqrt(4)", portable_rsqrtf(Float32(4.0)), HALF)
    _expect("rsqrt(0.25)", portable_rsqrtf(Float32(0.25)), _bits(Float32(2.0)))
    # log1p
    _expect("log1p(NaN)", portable_log1pf(_f(QNAN)), QNAN)
    _expect("log1p(+0)", portable_log1pf(_f(PZ)), PZ)
    _expect("log1p(-0)", portable_log1pf(_f(NZ)), NZ)
    _expect("log1p(subnormal)", portable_log1pf(sub), PZ)
    _expect("log1p(-subnormal)", portable_log1pf(nsub), NZ)
    _expect("log1p(+inf)", portable_log1pf(_f(PINF)), PINF)
    _expect("log1p(-inf)", portable_log1pf(_f(NINF)), QNAN)
    _expect("log1p(-1)", portable_log1pf(Float32(-1.0)), NINF)
    _expect("log1p(-1.5)", portable_log1pf(Float32(-1.5)), QNAN)
    # both sides of each branch point land in the polynomial / log arm
    # and agree with libm within the bound (checked in the sweep below;
    # the four boundary inputs are pinned into the sweep's first lanes)
    # sigmoid
    _expect("sigmoid(NaN)", portable_sigmoidf(_f(QNAN)), QNAN)
    _expect("sigmoid(+0)", portable_sigmoidf(_f(PZ)), HALF)
    _expect("sigmoid(-0)", portable_sigmoidf(_f(NZ)), HALF)
    _expect("sigmoid(-inf)", portable_sigmoidf(_f(NINF)), PZ)
    _expect("sigmoid(+inf)", portable_sigmoidf(_f(PINF)), ONE)
    _expect("sigmoid(-100)", portable_sigmoidf(Float32(-100.0)), PZ)
    _expect("sigmoid(-89)", portable_sigmoidf(Float32(-89.0)), PZ)  # exp overflow
    _expect("sigmoid(-88)", portable_sigmoidf(Float32(-88.0)), PZ)  # subnormal quotient, flushed
    _expect("sigmoid(17)", portable_sigmoidf(Float32(17.0)), ONE)
    _expect("sigmoid(100)", portable_sigmoidf(Float32(100.0)), ONE)
    # silu
    _expect("silu(NaN)", portable_siluf(_f(QNAN)), QNAN)
    _expect("silu(+0)", portable_siluf(_f(PZ)), PZ)
    _expect("silu(-0)", portable_siluf(_f(NZ)), NZ)
    _expect("silu(+inf)", portable_siluf(_f(PINF)), PINF)
    _expect("silu(-inf)", portable_siluf(_f(NINF)), QNAN)  # -inf/+inf, the reference's answer
    _expect("silu(-100)", portable_siluf(Float32(-100.0)), NZ)  # x/+inf, signed
    _expect("silu(-89)", portable_siluf(Float32(-89.0)), NZ)
    _expect("silu(subnormal)", portable_siluf(sub), PZ)
    _expect("silu(-subnormal)", portable_siluf(nsub), NZ)
    _expect("silu(100)", portable_siluf(Float32(100.0)), _bits(Float32(100.0)))
    # softplus: the guard boundary both sides
    _expect("softplus(NaN)", portable_softplusf(_f(QNAN)), QNAN)
    _expect("softplus(-inf)", portable_softplusf(_f(NINF)), PZ)
    _expect("softplus(+inf)", portable_softplusf(_f(PINF)), PINF)
    _expect("softplus(-100)", portable_softplusf(Float32(-100.0)), PZ)
    var twenty = Float32(20.0)
    var above20 = _f(_bits(twenty) + UInt32(1))
    # the arm taken at exactly 20 is log1p(exp(20)): assert it by
    # recomputing that arm explicitly. In float32 the two arms AGREE at
    # 20.0 (log1p(exp(x)) = x + exp(-x), below half an ulp of x from ~17
    # up), so the guard's REACH is not at 20 -- it is where exp(x)
    # overflows (x > 88.72: the identity arm returns x, a missing guard
    # returns log1p(+inf) = +inf), which softplus(100) below plants.
    from mojo_only.numerics import portable_expf
    if _bits(portable_softplusf(twenty)) != _bits(portable_log1pf(portable_expf(twenty))):
        raise Error("softplus(20.0) did not take the log1p(exp) arm")
    var sp20 = portable_softplusf(twenty)
    print("softplus(20.0) =", sp20, "(", hex(_bits(sp20)), "); identity arm would be", hex(_bits(twenty)))
    _expect("softplus(next above 20) returns itself",
            portable_softplusf(above20), _bits(above20))
    _expect("softplus(100)", portable_softplusf(Float32(100.0)), _bits(Float32(100.0)))
    print("planted special values: all as stated")

    var ctx = DeviceContext()

    # ---- INPUTS -----------------------------------------------------------
    var h_xr = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_xl = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_xa = ctx.enqueue_create_host_buffer[DType.float32](N)
    var pxr = h_xr.unsafe_ptr()
    var pxl = h_xl.unsafe_ptr()
    var pxa = h_xa.unsafe_ptr()
    # rsqrt: the first 16 lanes pinned, then RAW HASHED 32-bit patterns
    # (so denormals, zeros, infs, NaNs and negatives all arrive at their
    # natural density, ~1/256 each for the exponent-0 and exponent-255
    # bands, half negative)
    pxr.unsafe_store(0, Float32(0.0))
    pxr.unsafe_store(1, _f(NZ))
    pxr.unsafe_store(2, Float32(1.0))
    pxr.unsafe_store(3, Float32(4.0))
    pxr.unsafe_store(4, _f(PINF))
    pxr.unsafe_store(5, _f(NINF))
    pxr.unsafe_store(6, _f(QNAN))
    pxr.unsafe_store(7, _f(UInt32(0x00800000)))  # FLT_MIN
    pxr.unsafe_store(8, _f(UInt32(0x007FFFFF)))  # largest subnormal
    pxr.unsafe_store(9, _f(UInt32(0x7F7FFFFF)))  # FLT_MAX
    pxr.unsafe_store(10, Float32(-1.0))
    pxr.unsafe_store(11, Float32(2.0))
    pxr.unsafe_store(12, Float32(0.25))
    pxr.unsafe_store(13, Float32(1e-10))
    pxr.unsafe_store(14, _f(UInt32(0x0D800000)))  # 2^-100
    pxr.unsafe_store(15, Float32(3.0))
    # log1p: pinned edges incl. the two branch points' neighbours, then
    # half the lanes hashed log-uniform magnitudes in (2^-30, 2^6) both
    # signs (the domain that matters, x > -1), half raw patterns
    pxl.unsafe_store(0, Float32(0.0))
    pxl.unsafe_store(1, _f(NZ))
    pxl.unsafe_store(2, Float32(-1.0))
    pxl.unsafe_store(3, Float32(-1.5))
    pxl.unsafe_store(4, _f(PINF))
    pxl.unsafe_store(5, _f(QNAN))
    pxl.unsafe_store(6, _f(UInt32(0xBE95F61A)))  # lower branch point
    pxl.unsafe_store(7, _f(UInt32(0xBE95F61A) + UInt32(1)))  # just below it (more negative)
    pxl.unsafe_store(8, _f(UInt32(0xBE95F61A) - UInt32(1)))  # just above it
    pxl.unsafe_store(9, _f(UInt32(0x3ED413CD)))  # upper branch point
    pxl.unsafe_store(10, _f(UInt32(0x3ED413CD) + UInt32(1)))
    pxl.unsafe_store(11, _f(UInt32(0x3ED413CD) - UInt32(1)))
    pxl.unsafe_store(12, Float32(1.0))
    pxl.unsafe_store(13, Float32(-0.99999994))
    pxl.unsafe_store(14, Float32(1e-20))
    pxl.unsafe_store(15, Float32(1e30))
    # activations: pinned edges, then 3/4 hashed reals in [-120, 120]
    # (past both saturation edges and the softplus guard), 1/4 raw patterns
    pxa.unsafe_store(0, Float32(0.0))
    pxa.unsafe_store(1, _f(NZ))
    pxa.unsafe_store(2, Float32(20.0))
    pxa.unsafe_store(3, above20)
    pxa.unsafe_store(4, _f(_bits(twenty) - UInt32(1)))
    pxa.unsafe_store(5, _f(PINF))
    pxa.unsafe_store(6, _f(NINF))
    pxa.unsafe_store(7, _f(QNAN))
    pxa.unsafe_store(8, Float32(-88.0))
    pxa.unsafe_store(9, Float32(-89.0))
    pxa.unsafe_store(10, Float32(-87.0))
    pxa.unsafe_store(11, Float32(88.8))
    pxa.unsafe_store(12, Float32(17.0))
    pxa.unsafe_store(13, Float32(-17.0))
    pxa.unsafe_store(14, _f(UInt32(0x00400000)))
    pxa.unsafe_store(15, _f(UInt32(0x80400000)))
    for i in range(16, N):
        var h1 = _splitmix(UInt64(3 * i + 1))
        var h2 = _splitmix(UInt64(3 * i + 2))
        var h3 = _splitmix(UInt64(3 * i + 3))
        pxr.unsafe_store(i, _f(UInt32(h1 & UInt64(0xFFFFFFFF))))
        if (h2 >> 40) & UInt64(1) == UInt64(0):
            # log-uniform magnitude 2^-30 .. 2^6, hashed mantissa, sign
            var ex = Int((h2 >> 41) % UInt64(36)) - 30
            var mant = UInt32(h2 & UInt64(0x007FFFFF))
            var eb = UInt32(ex + 127) << 23
            var sgn = UInt32(0x80000000) if ((h2 >> 32) & UInt64(1)) == UInt64(1) else UInt32(0)
            pxl.unsafe_store(i, _f(mant | eb | sgn))
        else:
            pxl.unsafe_store(i, _f(UInt32(h2 & UInt64(0xFFFFFFFF))))
        if (h3 >> 40) & UInt64(3) != UInt64(0):
            var u = Float64(h3 & UInt64(0xFFFFFFFF)) / Float64(4294967296.0)
            pxa.unsafe_store(i, Float32(-120.0 + 240.0 * u))
        else:
            pxa.unsafe_store(i, _f(UInt32(h3 & UInt64(0xFFFFFFFF))))

    # ---- arm 1: accuracy on the host ---------------------------------
    var hr = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hl = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hsg = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hsl = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hsp = ctx.enqueue_create_host_buffer[DType.float32](N)
    var phr = hr.unsafe_ptr()
    var phl = hl.unsafe_ptr()
    var phsg = hsg.unsafe_ptr()
    var phsl = hsl.unsafe_ptr()
    var phsp = hsp.unsafe_ptr()
    var max_ulp_r = Int64(0)
    var exact_r = 0
    var n_r = 0
    var max_ulp_l = Int64(0)
    var max_ulp_l_poly = Int64(0)
    var n_l = 0
    var n_l_poly = 0
    var max_ulp_sg = Int64(0)
    var max_ulp_sl = Int64(0)
    var max_ulp_sp = Int64(0)
    var n_a = 0
    var worst_l = Float32(0.0)
    var worst_sg = Float32(0.0)
    var worst_sl = Float32(0.0)
    var worst_sp = Float32(0.0)
    for i in range(N):
        var xr = pxr.unsafe_load(i)
        var yr = portable_rsqrtf(xr)
        phr.unsafe_store(i, yr)
        if _finite(_bits(xr)) and xr > Float32(0.0) and _bits(xr) >= UInt32(0x00800000):
            # positive normal: the float64 1/sqrt rounded once (two double
            # roundings, then one to float: within an ulp of the correctly
            # rounded rsqrt; a reference for DISTANCE, not an oracle)
            var refr = Float32(Float64(1.0) / sqrt(Float64(xr)))
            var d = _ulp_dist(yr, refr)
            n_r += 1
            if d == Int64(0):
                exact_r += 1
            if d > max_ulp_r:
                max_ulp_r = d
        var xl = pxl.unsafe_load(i)
        var yl = portable_log1pf(xl)
        phl.unsafe_store(i, yl)
        if _finite(_bits(xl)) and xl > Float32(-1.0) and _bits(xl) & UInt32(0x7F800000) != UInt32(0):
            var refl = Float32(_log1p64(Float64(xl)))
            var d = _ulp_dist(yl, refl)
            n_l += 1
            if d > max_ulp_l:
                max_ulp_l = d
                worst_l = xl
            if xl >= _f(UInt32(0xBE95F61A)) and xl <= _f(UInt32(0x3ED413CD)):
                n_l_poly += 1
                if d > max_ulp_l_poly:
                    max_ulp_l_poly = d
        var xa = pxa.unsafe_load(i)
        var ysg = portable_sigmoidf(xa)
        var ysl = portable_siluf(xa)
        var ysp = portable_softplusf(xa)
        phsg.unsafe_store(i, ysg)
        phsl.unsafe_store(i, ysl)
        phsp.unsafe_store(i, ysp)
        if _finite(_bits(xa)) and _bits(xa) & UInt32(0x7F800000) != UInt32(0):
            n_a += 1
            var x64 = Float64(xa)
            var e64 = _expf_ideal(-x64)
            var rsg = Float32(Float64(1.0) / (Float64(1.0) + e64))
            var rsl = Float32(x64 / (Float64(1.0) + e64))
            var rsp: Float32
            if xa <= Float32(20.0):
                rsp = Float32(_log1p64(_expf_ideal(x64)))
            else:
                rsp = xa
            # a subnormal float64->float32 reference is flushed like the
            # portable result (row 10 is baked in on purpose)
            if abs(rsg) < Float32(1.1754943508222875e-38):
                rsg = Float32(0.0)
            if abs(rsl) < Float32(1.1754943508222875e-38):
                rsl = Float32(-0.0) if rsl < Float32(0.0) else Float32(0.0)
            if abs(rsp) < Float32(1.1754943508222875e-38):
                rsp = Float32(0.0)
            var dsg = _ulp_dist(ysg, rsg)
            var dsl = _ulp_dist(ysl, rsl)
            var dsp = _ulp_dist(ysp, rsp)
            # signed-zero results compare by bit (ulp distance of +0/-0 is 1
            # in the order map; exempt exact-zero references)
            if rsg == Float32(0.0) and ysg == Float32(0.0):
                dsg = Int64(0)
            if rsl == Float32(0.0) and ysl == Float32(0.0):
                dsl = Int64(0)
            if rsp == Float32(0.0) and ysp == Float32(0.0):
                dsp = Int64(0)
            if dsg > max_ulp_sg:
                max_ulp_sg = dsg
                worst_sg = xa
            if dsl > max_ulp_sl:
                max_ulp_sl = dsl
                worst_sl = xa
            if dsp > max_ulp_sp:
                max_ulp_sp = dsp
                worst_sp = xa

    print("portable_rsqrtf vs float64 1/sqrt (positive normals,", n_r, "lanes): max",
          max_ulp_r, "ulp; exact on", exact_r)
    print("portable_log1pf vs libm log1p (finite x > -1,", n_l, "lanes): max",
          max_ulp_l, "ulp (worst x =", worst_l, "); polynomial branch", n_l_poly,
          "lanes max", max_ulp_l_poly, "ulp")
    print("portable_sigmoidf vs float64: max", max_ulp_sg, "ulp (worst x =", worst_sg, ")")
    print("portable_siluf vs float64: max", max_ulp_sl, "ulp (worst x =", worst_sl, ")")
    print("portable_softplusf vs float64: max", max_ulp_sp, "ulp (worst x =", worst_sp, ")")
    if max_ulp_r > Int64(1):
        raise Error("rsqrt pin (a) further than 1 ulp from the float64 1/sqrt")
    if max_ulp_l > Int64(2):
        raise Error("log1p accuracy bound (2 ulp) exceeded")
    if max_ulp_sg > Int64(4) or max_ulp_sl > Int64(4) or max_ulp_sp > Int64(4):
        raise Error("sigmoid/silu/softplus accuracy bound (4 ulp) exceeded")

    # ---- arm 2: the device arm must reproduce the host bits ------------
    var d_xr = ctx.enqueue_create_buffer[DType.float32](N)
    var d_xl = ctx.enqueue_create_buffer[DType.float32](N)
    var d_xa = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=d_xr, src_ptr=h_xr.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_xl, src_ptr=h_xl.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_xa, src_ptr=h_xa.unsafe_ptr())
    var o_r = ctx.enqueue_create_buffer[DType.float32](N)
    var o_l = ctx.enqueue_create_buffer[DType.float32](N)
    var o_sg = ctx.enqueue_create_buffer[DType.float32](N)
    var o_sl = ctx.enqueue_create_buffer[DType.float32](N)
    var o_sp = ctx.enqueue_create_buffer[DType.float32](N)

    ctx.enqueue_function[nn_kernel](
        d_xr.unsafe_ptr(), d_xl.unsafe_ptr(), d_xa.unsafe_ptr(),
        o_r.unsafe_ptr(), o_l.unsafe_ptr(), o_sg.unsafe_ptr(),
        o_sl.unsafe_ptr(), o_sp.unsafe_ptr(),
        Int32(N), Int32(1) if sabotage else Int32(0),
        grid_dim=(N + BLOCK - 1) // BLOCK,
        block_dim=(BLOCK, 1, 1),
    )

    var r_r = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_l = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_sg = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_sl = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_sp = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_ptr=r_r.unsafe_ptr(), src_buf=o_r)
    ctx.enqueue_copy(dst_ptr=r_l.unsafe_ptr(), src_buf=o_l)
    ctx.enqueue_copy(dst_ptr=r_sg.unsafe_ptr(), src_buf=o_sg)
    ctx.enqueue_copy(dst_ptr=r_sl.unsafe_ptr(), src_buf=o_sl)
    ctx.enqueue_copy(dst_ptr=r_sp.unsafe_ptr(), src_buf=o_sp)
    ctx.synchronize()

    var qr = r_r.unsafe_ptr()
    var ql = r_l.unsafe_ptr()
    var qsg = r_sg.unsafe_ptr()
    var qsl = r_sl.unsafe_ptr()
    var qsp = r_sp.unsafe_ptr()
    var mm = List[Int]()
    var hashes = List[UInt64]()
    var fnv = UInt64(0xCBF29CE484222325)
    var shown: Int
    # function-major: each function's lanes in order, then the next
    for k in range(NF):
        var dp = qr
        var hp = phr
        if k == 1:
            dp = ql
            hp = phl
        elif k == 2:
            dp = qsg
            hp = phsg
        elif k == 3:
            dp = qsl
            hp = phsl
        elif k == 4:
            dp = qsp
            hp = phsp
        var bad = 0
        var hk = UInt64(0xCBF29CE484222325)
        shown = 0
        for i in range(N):
            var db = _bits(dp.unsafe_load(i))
            var hb = _bits(hp.unsafe_load(i))
            if not _same(db, hb):
                bad += 1
                if shown < 3 and not sabotage:
                    print("  mismatch fn", k, "lane", i, "device", hex(db),
                          "host", hex(hb))
                    shown += 1
            var c = UInt64(_canon(db))
            hk = (hk ^ c) * UInt64(0x100000001B3)
            fnv = (fnv ^ c) * UInt64(0x100000001B3)
        mm.append(bad)
        hashes.append(hk)

    print("nn device hash =", fnv)
    print("  per function: rsqrt", hashes[0], " log1p", hashes[1],
          " sigmoid", hashes[2], " silu", hashes[3], " softplus", hashes[4])
    print("  device/host mismatches: rsqrt", mm[0], " log1p", mm[1],
          " sigmoid", mm[2], " silu", mm[3], " softplus", mm[4])

    if sabotage:
        var blind = 0
        for k in range(NF):
            if mm[k] == 0:
                blind += 1
        if blind != 0:
            raise Error(
                "SABOTAGE arm found ZERO mismatches for "
                + String(blind)
                + " function(s): that compare is blind"
            )
        print("sabotage detected as it must be on every function -- the compare has teeth")
    else:
        var total = 0
        for k in range(NF):
            total += mm[k]
        if total != 0:
            raise Error(
                "device bits differ from host bits on "
                + String(total)
                + " lanes: this backend broke the basic-ops premise"
            )
        print("device == host on every lane; all arms OK")

    _ = h_xr.unsafe_ptr()
    _ = h_xl.unsafe_ptr()
    _ = h_xa.unsafe_ptr()
    _ = hr.unsafe_ptr()
    _ = hl.unsafe_ptr()
    _ = hsg.unsafe_ptr()
    _ = hsl.unsafe_ptr()
    _ = hsp.unsafe_ptr()
    _ = r_r.unsafe_ptr()
    _ = r_l.unsafe_ptr()
    _ = r_sg.unsafe_ptr()
    _ = r_sl.unsafe_ptr()
    _ = r_sp.unsafe_ptr()
