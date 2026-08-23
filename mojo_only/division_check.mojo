"""IDENTITY_PATHS row 49 (DEVIATION 740): DIVISION, characterized per
class on this backend, and the `portable_divf`/`identical_div` seam's
certificate. Row 10's open clause was "division is still correctly rounded
on normals everywhere measured" -- true of `check-ieee-arith`'s patterns
on three vendors, and the same sentence about sqrt was false once
(DEVIATION 258). This gate is the shape of `check-ieee-arith`'s div arm
with the classes a Mamba block's divisions actually reach spelled out,
and a hash a leg re-prints.

    pixi run check-division            (append `sabotage` for the reach arm)

## 2^20 hashed (a, b) pairs in 13 CLASSES (lane i's class is a function of
## i, so every column sees the same pairs), each judged BIT FOR BIT

     0  normal / normal                     (4/16 of the lanes)
     1  subnormal a / normal b
     2  normal a / subnormal b
     3  subnormal / subnormal
     4  +-0 / b
     5  a / +-0
     6  +-inf in either operand
     7  NaN in either operand
     8  quotient at or over FLT_MAX (|a| >= 2^64, |b| <= 2^-64)
     9  quotient is SUBNORMAL (|a| in [2^-70,2^-60), |b| in [2^67,2^77))
    10  quotient underflows to zero (|a| < 2^-90, |b| > 2^90)
    11  exact quotients (b a power of two, a hashed normal)
    12  raw hashed 32-bit pairs (whatever falls out)

## The references, and why they are exact

HOST FP32 `a / b` is the correctly rounded IEEE quotient on this host
(arm64 fdiv is IEEE, subnormals honored: probed). It is cross-checked on
EVERY lane against `Float32(Float64(a) / Float64(b))`, which is ALSO
correctly rounded: double rounding is innocuous for division when the
wide format carries p >= 2q + 2 bits (53 >= 50, the classical bound) --
so the worry that "float64 then float32 is not correctly rounded" does
not apply to division (it would to a composed op). The two agree on
every lane or this check raises before judging the device.

THE FTZ MODEL (row 10): operands flushed to signed zero, one correctly
rounded division, result flushed. `portable_divf` IS this model, on the
host and on the device.

## What is judged

  A. device raw `a / b` per class: matches IEEE / matches the FTZ model /
     NEITHER. Neither is a division defect (a fast reciprocal) and FAILS.
     A FTZ-model match is row 10's policy, not a defect, and is reported.
  B. device `portable_divf(a, b)` == host `portable_divf(a, b)` on every
     lane (the seam's identity; FAILS on any mismatch). Its hash is THE
     CERTIFICATE LINE, the same number on every column.
  C. device `identical_div(a, b)` == host: under IDENTICAL must be 0 (it
     is B); under FAST RECORDED (host honors subnormals, an FTZ device
     does not, so the subnormal classes differ there by construction).
  D. RECORDED, NOT USED (DEVIATION 746): the hardware `rsqrt` intrinsic
     on |a| (positive normals) against a float64 1/sqrt and against
     `portable_rsqrtf`; and the stdlib `1 / sqrt(|a|)`. Mismatch counts,
     max ulp, a per-vendor hash. Nothing here is a pass/fail.
  E. a raw-division fingerprint hash (device `a / b` bits, NaN
     canonicalized): per-vendor, NOT expected to agree across columns
     (an FTZ column and a denormal-honoring one differ on classes 1-3, 9).

## Reach (`sabotage` argv)

The device divides a dividend nudged by one part in 2^23 while the host
does not; the B compare must then fail on the lanes whose quotient is
finite, nonzero and normal (~290k of 2^20 on Apple). The flush
INSIDE `portable_divf` is bit-inert on an FTZ column and its reach can
only be seen on a denormal-honoring leg (where arm A's class 1-3 counts
are the evidence); this file prints the device raw-vs-portable mismatch
count so that leg can read it off.
"""

from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import rsqrt, sqrt
from std.memory import bitcast
from std.sys import argv

from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    identical_div,
    portable_divf,
    portable_rsqrtf,
)

comptime N = 1 << 20
comptime BLOCK = 256
comptime NCLASS = 13


def div_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    out_raw: MutPointer[Float32, MutAnyOrigin],
    out_port: MutPointer[Float32, MutAnyOrigin],
    out_ident: MutPointer[Float32, MutAnyOrigin],
    out_rsqrt_hw: MutPointer[Float32, MutAnyOrigin],
    out_rsqrt_div: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    sabotage_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var av = a.unsafe_load(i)
        var bv = b.unsafe_load(i)
        out_raw.unsafe_store(i, av / bv)
        var ap = av
        if sabotage_in != Int32(0):
            ap = av * Float32(1.00000012)
        out_port.unsafe_store(i, portable_divf(ap, bv))
        out_ident.unsafe_store(i, identical_div(av, bv))
        var x = abs(av)
        out_rsqrt_hw.unsafe_store(i, rsqrt(x))
        out_rsqrt_div.unsafe_store(i, Float32(1.0) / sqrt(x))


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
    if _is_nan(b):
        return UInt32(0x7FC00000)
    return b


def _same(a: UInt32, b: UInt32) -> Bool:
    return _canon(a) == _canon(b)


def _ftz(x: Float32) -> Float32:
    if abs(x) < Float32(1.1754943508222875e-38) and x != Float32(0.0):
        if x < Float32(0.0):
            return Float32(-0.0)
        return Float32(0.0)
    return x


def _ord(b: UInt32) -> Int64:
    if b & UInt32(0x80000000):
        return Int64(0xFFFFFFFF) - Int64(b & UInt32(0x7FFFFFFF))
    return Int64(0x100000000) + Int64(b)


def _ulp_dist(a: Float32, b: Float32) -> Int64:
    var d = _ord(_bits(a)) - _ord(_bits(b))
    if d < Int64(0):
        return -d
    return d


def _normal(h: UInt64, lo_e: Int, hi_e: Int) -> UInt32:
    """A normal float with hashed sign and mantissa and an unbiased
    exponent hashed from [lo_e, hi_e]."""
    var mant = UInt32(h & UInt64(0x007FFFFF))
    var span = UInt64(hi_e - lo_e + 1)
    var e = lo_e + Int((h >> 23) % span)
    var sgn = UInt32(0x80000000) if ((h >> 60) & UInt64(1)) == UInt64(1) else UInt32(0)
    return sgn | (UInt32(e + 127) << 23) | mant


def _subnormal(h: UInt64) -> UInt32:
    var mant = UInt32(h & UInt64(0x007FFFFF))
    if mant == UInt32(0):
        mant = UInt32(1)
    var sgn = UInt32(0x80000000) if ((h >> 60) & UInt64(1)) == UInt64(1) else UInt32(0)
    return sgn | mant


def _class_name(k: Int) -> String:
    if k == 0:
        return "normal/normal            "
    if k == 1:
        return "subnormal a / normal b   "
    if k == 2:
        return "normal a / subnormal b   "
    if k == 3:
        return "subnormal / subnormal    "
    if k == 4:
        return "+-0 / b                  "
    if k == 5:
        return "a / +-0                  "
    if k == 6:
        return "inf operand              "
    if k == 7:
        return "NaN operand              "
    if k == 8:
        return "quotient at/over FLT_MAX "
    if k == 9:
        return "quotient subnormal       "
    if k == 10:
        return "quotient underflows to 0 "
    if k == 11:
        return "exact (b power of two)   "
    return "raw hashed pairs         "


def _mode_name() -> String:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return "IDENTICAL"
    return "FAST"


def main() raises:
    var sabotage = False
    for a in argv():
        if a == "sabotage":
            sabotage = True

    print("division check; build mode", _mode_name())

    var ctx = DeviceContext()
    var ha = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](N)
    var pa = ha.unsafe_ptr()
    var pb = hb.unsafe_ptr()
    var cls = List[Int]()

    for i in range(N):
        var h1 = _splitmix(UInt64(2 * i + 1))
        var h2 = _splitmix(UInt64(2 * i + 2))
        var r = i % 16
        var k: Int
        if r < 4:
            k = 0
        else:
            k = r - 3  # 1..12
        cls.append(k)
        var ba: UInt32
        var bb: UInt32
        if k == 0:
            ba = _normal(h1, -126, 127)
            bb = _normal(h2, -126, 127)
        elif k == 1:
            ba = _subnormal(h1)
            bb = _normal(h2, -126, 127)
        elif k == 2:
            ba = _normal(h1, -126, 127)
            bb = _subnormal(h2)
        elif k == 3:
            ba = _subnormal(h1)
            bb = _subnormal(h2)
        elif k == 4:
            ba = UInt32(0x80000000) if (h1 & UInt64(1)) == UInt64(1) else UInt32(0)
            bb = _normal(h2, -126, 127)
            if (h2 >> 61) & UInt64(1) == UInt64(1):
                bb = _subnormal(h2)  # a zero over a subnormal too
        elif k == 5:
            ba = _normal(h1, -126, 127)
            if (h1 >> 61) & UInt64(1) == UInt64(1):
                ba = _subnormal(h1)
            bb = UInt32(0x80000000) if (h2 & UInt64(1)) == UInt64(1) else UInt32(0)
        elif k == 6:
            var which = (h1 >> 61) & UInt64(3)
            ba = _normal(h1, -126, 127)
            bb = _normal(h2, -126, 127)
            var inf = UInt32(0x7F800000) | (UInt32(0x80000000) if (h2 >> 62) & UInt64(1) == UInt64(1) else UInt32(0))
            if which == UInt64(0):
                ba = inf
            elif which == UInt64(1):
                bb = inf
            else:
                ba = inf
                bb = inf ^ (UInt32(0x80000000) if (h1 >> 59) & UInt64(1) == UInt64(1) else UInt32(0))
        elif k == 7:
            var which = (h1 >> 61) & UInt64(3)
            ba = _normal(h1, -126, 127)
            bb = _normal(h2, -126, 127)
            # a hashed payload: exponent all ones, nonzero mantissa
            var nan = UInt32(0x7F800000) | UInt32(h2 & UInt64(0x007FFFFF)) | UInt32(0x00400000)
            if which == UInt64(0):
                ba = nan
            elif which == UInt64(1):
                bb = nan
            else:
                ba = nan
                bb = nan ^ UInt32(0x80000000)
        elif k == 8:
            ba = _normal(h1, 64, 127)
            bb = _normal(h2, -126, -64)
        elif k == 9:
            # ma/mb in (0.5, 2) so the quotient exponent is ea-eb-1..ea-eb,
            # kept inside [-147, -127]: every quotient is a subnormal
            ba = _normal(h1, -70, -61)
            bb = _normal(h2, 67, 76)
        elif k == 10:
            ba = _normal(h1, -126, -91)
            bb = _normal(h2, 91, 127)
        elif k == 11:
            ba = _normal(h1, -126, 127)
            bb = _normal(h2, -126, 127) & UInt32(0xFF800000)  # mantissa cleared
        else:
            ba = UInt32(h1 & UInt64(0xFFFFFFFF))
            bb = UInt32(h2 & UInt64(0xFFFFFFFF))
        pa.unsafe_store(i, _f(ba))
        pb.unsafe_store(i, _f(bb))

    # ---- the host references, cross-checked ---------------------------
    var h_ieee = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_port = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_ident = ctx.enqueue_create_host_buffer[DType.float32](N)
    var p_ieee = h_ieee.unsafe_ptr()
    var p_port = h_port.unsafe_ptr()
    var p_ident = h_ident.unsafe_ptr()
    var ref_disagree = 0
    for i in range(N):
        var av = pa.unsafe_load(i)
        var bv = pb.unsafe_load(i)
        var q32 = av / bv
        var q64 = Float32(Float64(av) / Float64(bv))
        if not _same(_bits(q32), _bits(q64)):
            ref_disagree += 1
            if ref_disagree <= 4:
                print("  HOST REFERENCES DISAGREE: a=", hex(_bits(av)), " b=",
                      hex(_bits(bv)), " fp32 div=", hex(_bits(q32)),
                      " f64 route=", hex(_bits(q64)))
        p_ieee.unsafe_store(i, q32)
        p_port.unsafe_store(i, portable_divf(av, bv))
        p_ident.unsafe_store(i, identical_div(av, bv))
    print("host fp32 a/b vs Float32(Float64(a)/Float64(b)):", ref_disagree,
          "disagreements of", N, "(both correctly rounded; must be 0)")
    if ref_disagree != 0:
        raise Error("the two correctly rounded host references disagree: host arithmetic is not IEEE here")
    # the FTZ model must be what portable_divf computes on the host
    var model_off = 0
    for i in range(N):
        var av = pa.unsafe_load(i)
        var bv = pb.unsafe_load(i)
        var m = _ftz(_ftz(av) / _ftz(bv))
        if not _same(_bits(m), _bits(p_port.unsafe_load(i))):
            model_off += 1
    if model_off != 0:
        raise Error("portable_divf on the host is not the FTZ model on " + String(model_off) + " lanes")

    # ---- the device -----------------------------------------------------
    var da = ctx.enqueue_create_buffer[DType.float32](N)
    var db = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=da, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=db, src_ptr=hb.unsafe_ptr())
    var o_raw = ctx.enqueue_create_buffer[DType.float32](N)
    var o_port = ctx.enqueue_create_buffer[DType.float32](N)
    var o_ident = ctx.enqueue_create_buffer[DType.float32](N)
    var o_rhw = ctx.enqueue_create_buffer[DType.float32](N)
    var o_rdv = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_function[div_kernel](
        da.unsafe_ptr(), db.unsafe_ptr(),
        o_raw.unsafe_ptr(), o_port.unsafe_ptr(), o_ident.unsafe_ptr(),
        o_rhw.unsafe_ptr(), o_rdv.unsafe_ptr(),
        Int32(N), Int32(1) if sabotage else Int32(0),
        grid_dim=(N + BLOCK - 1) // BLOCK,
        block_dim=(BLOCK, 1, 1),
    )
    var r_raw = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_port = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_ident = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_rhw = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_rdv = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_ptr=r_raw.unsafe_ptr(), src_buf=o_raw)
    ctx.enqueue_copy(dst_ptr=r_port.unsafe_ptr(), src_buf=o_port)
    ctx.enqueue_copy(dst_ptr=r_ident.unsafe_ptr(), src_buf=o_ident)
    ctx.enqueue_copy(dst_ptr=r_rhw.unsafe_ptr(), src_buf=o_rhw)
    ctx.enqueue_copy(dst_ptr=r_rdv.unsafe_ptr(), src_buf=o_rdv)
    ctx.synchronize()
    var q_raw = r_raw.unsafe_ptr()
    var q_port = r_port.unsafe_ptr()
    var q_ident = r_ident.unsafe_ptr()
    var q_rhw = r_rhw.unsafe_ptr()
    var q_rdv = r_rdv.unsafe_ptr()

    # ---- A: raw division per class ----------------------------------------
    var n_cls = List[Int]()
    var ieee_cls = List[Int]()
    var ftz_cls = List[Int]()
    var neither_cls = List[Int]()
    for _ in range(NCLASS):
        n_cls.append(0)
        ieee_cls.append(0)
        ftz_cls.append(0)
        neither_cls.append(0)
    var shown = 0
    var raw_vs_port = 0  # the flush's visibility on this column
    var port_mm = 0
    var ident_mm = 0
    var model_divergent = 0  # lanes where the IEEE quotient and the FTZ model differ
    var ident_mm_outside_model = 0  # identical_div mismatches NOT explained by the flush
    var fnv_port = UInt64(0xCBF29CE484222325)
    var fnv_raw = UInt64(0xCBF29CE484222325)
    var fnv_rhw = UInt64(0xCBF29CE484222325)
    for i in range(N):
        var k = cls[i]
        n_cls[k] += 1
        var draw = _bits(q_raw.unsafe_load(i))
        var ieee = _bits(p_ieee.unsafe_load(i))
        var model = _bits(p_port.unsafe_load(i))
        if _same(draw, ieee):
            ieee_cls[k] += 1
        elif _same(draw, model):
            ftz_cls[k] += 1
        else:
            neither_cls[k] += 1
            if shown < 6:
                print("  NEITHER: class", k, " a=", hex(_bits(pa.unsafe_load(i))),
                      " b=", hex(_bits(pb.unsafe_load(i))), " device=", hex(draw),
                      " ieee=", hex(ieee), " ftz-model=", hex(model))
                shown += 1
        var dport = _bits(q_port.unsafe_load(i))
        if not _same(dport, model):
            port_mm += 1
        if not _same(draw, dport) and not sabotage:
            raw_vs_port += 1
        var dident = _bits(q_ident.unsafe_load(i))
        var flush_differs = not _same(ieee, model)
        if flush_differs:
            model_divergent += 1
        if not _same(dident, _bits(p_ident.unsafe_load(i))):
            ident_mm += 1
            if not flush_differs:
                ident_mm_outside_model += 1
        fnv_port = (fnv_port ^ UInt64(_canon(dport))) * UInt64(0x100000001B3)
        fnv_raw = (fnv_raw ^ UInt64(_canon(draw))) * UInt64(0x100000001B3)
        fnv_rhw = (fnv_rhw ^ UInt64(_canon(_bits(q_rhw.unsafe_load(i))))) * UInt64(0x100000001B3)

    print("A. device a / b per class (", N, "pairs ): class | n | == IEEE | == FTZ model | NEITHER")
    var total_neither = 0
    for k in range(NCLASS):
        print("   ", k, _class_name(k), n_cls[k], " ", ieee_cls[k], " ", ftz_cls[k], " ", neither_cls[k])
        total_neither += neither_cls[k]
    print("   device raw a/b vs device portable_divf differ on", raw_vs_port,
          "lanes (0 = the flush is bit-inert on this column; nonzero = a denormal-honoring column, aligned)")
    print("B. portable_divf device vs host mismatches:", port_mm,
          " certificate hash =", fnv_port)
    print("C. identical_div device vs host mismatches (", _mode_name(), "):", ident_mm,
          " -- lanes where IEEE and the flush model differ:", model_divergent,
          "; mismatches NOT explained by the flush:", ident_mm_outside_model)
    print("E. raw-division per-vendor fingerprint hash =", fnv_raw)

    # ---- D: the rsqrt intrinsic, RECORDED ----------------------------------
    var n_pos = 0
    var rhw_wrong = 0
    var rhw_vs_pin = 0
    var rdv_wrong = 0
    var rdv_vs_pin = 0
    var rhw_max = Int64(0)
    var rdv_max = Int64(0)
    for i in range(N):
        var x = abs(pa.unsafe_load(i))
        var xb = _bits(x)
        if xb >= UInt32(0x00800000) and xb < UInt32(0x7F800000):
            n_pos += 1
            var refx = Float32(Float64(1.0) / sqrt(Float64(x)))
            var pin = portable_rsqrtf(x)
            var hw = q_rhw.unsafe_load(i)
            var dv = q_rdv.unsafe_load(i)
            var d1 = _ulp_dist(hw, refx)
            var d2 = _ulp_dist(dv, refx)
            if d1 != Int64(0):
                rhw_wrong += 1
            if d2 != Int64(0):
                rdv_wrong += 1
            if d1 > rhw_max:
                rhw_max = d1
            if d2 > rdv_max:
                rdv_max = d2
            if _bits(hw) != _bits(pin):
                rhw_vs_pin += 1
            if _bits(dv) != _bits(pin):
                rdv_vs_pin += 1
    print("D. RECORDED, not used (", n_pos, "positive normals ): hardware rsqrt intrinsic off the float64 1/sqrt on",
          rhw_wrong, "lanes, max", rhw_max, "ulp; off the pin portable_rsqrtf on", rhw_vs_pin,
          "lanes; hash", fnv_rhw)
    print("   stdlib 1/sqrt(x) on the device off the float64 1/sqrt on", rdv_wrong,
          "lanes, max", rdv_max, "ulp; off the pin on", rdv_vs_pin, "lanes")

    # ---- verdicts -----------------------------------------------------------
    if total_neither != 0:
        raise Error("device division matched NEITHER the IEEE quotient nor the FTZ model on "
                    + String(total_neither) + " lanes: a fast reciprocal or a third arithmetic -- "
                    "identical_div needs the Newton-refined arm on this column")
    if sabotage:
        if port_mm == 0:
            raise Error("SABOTAGE arm found ZERO portable_divf mismatches: the compare is blind")
        print("sabotage detected as it must be:", port_mm, "portable_divf device/host mismatches -- the compare has teeth")
    else:
        if port_mm != 0:
            raise Error("portable_divf device bits differ from host bits on " + String(port_mm)
                        + " lanes: the flush model does not describe this column")
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            if ident_mm != 0:
                raise Error("identical_div device != host under IDENTICAL on " + String(ident_mm) + " lanes")
        if ident_mm_outside_model != 0:
            raise Error("identical_div device != host on " + String(ident_mm_outside_model)
                        + " lanes the flush model does not explain")
        var ftz_total = 0
        for k in range(NCLASS):
            ftz_total += ftz_cls[k]
        if ftz_total == 0:
            print("division check OK: correctly rounded on EVERY class, subnormals honored (a denormal-honoring column); portable_divf aligns it to the FTZ columns")
        else:
            print("division check OK: correctly rounded on normals, every subnormal-touching divergence (", ftz_total,
                  ") is row 10's flush model bit for bit -- record this column as IEEE+FTZ for division")

    _ = ha.unsafe_ptr()
    _ = hb.unsafe_ptr()
    _ = h_ieee.unsafe_ptr()
    _ = h_port.unsafe_ptr()
    _ = h_ident.unsafe_ptr()
    _ = r_raw.unsafe_ptr()
    _ = r_port.unsafe_ptr()
    _ = r_ident.unsafe_ptr()
    _ = r_rhw.unsafe_ptr()
    _ = r_rdv.unsafe_ptr()
