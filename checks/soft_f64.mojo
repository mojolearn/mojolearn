"""Software binary64 over UInt64 bit patterns (lane hr2-gbdt-host, 2026-10-02).

WHY. The GBDT probability links (sigmoid, softmax, the Logloss pair) and the
GreedyLogSum border scores are double-precision arithmetic, and the Apple
GPU has no float64 (`mojolearn-hardware-limits`). To run them on the device
on EVERY vendor with the same bits, they are spelled here as integer
arithmetic on the IEEE-754 binary64 encoding: correctly rounded
(round-to-nearest-even) add, sub, mul, div and fused multiply-add, plus
floor, comparisons and the conversions they need. The construction is
Berkeley SoftFloat 3's (`s_addMagsF64.c`, `s_subMagsF64.c`, `f64_mul.c`,
`f64_div.c` by restoring division, `s_roundPackToF64.c`,
`s_normRoundPackToF64.c`, `f64_to_f32.c`), so every result is the IEEE
result a hardware double unit returns in round-to-nearest with subnormals
honored. No product is ever contracted (there is no float product to
contract), so the NVIDIA, AMD, Apple and host columns compute the same word
by construction: one code path, integer instructions only.

NaN: every NaN result is the canonical quiet NaN `SF64_NAN` (payloads are
refused by the identity contract, IDENTITY_PATHS row 39).

`sf64_exp` and `sf64_log` are `checks/numerics.mojo`'s `portable_exp64` and
`portable_log64`, statement for statement, over these operations.
"""

from std.memory import bitcast

comptime SF64_SIGN = UInt64(0x8000000000000000)
comptime SF64_FRAC = UInt64(0x000FFFFFFFFFFFFF)
comptime SF64_HIDDEN = UInt64(0x0010000000000000)
comptime SF64_NAN = UInt64(0x7FF8000000000000)
comptime SF64_INF = UInt64(0x7FF0000000000000)
comptime SF64_ZERO = UInt64(0)
comptime SF64_ONE = UInt64(0x3FF0000000000000)

comptime _LOG2E = UInt64(0x3FF71547652B82FE)  # 1.4426950408889634
comptime _HALF = UInt64(0x3FE0000000000000)  # 0.5
comptime _TWO = UInt64(0x4000000000000000)  # 2.0
comptime _NC1 = UInt64(0xBFE62E4000000000)  # -0.693145751953125
comptime _NC2 = UInt64(0xBEB7F7D1CF79ABCA)  # -1.4286068203094173e-06
comptime _P0 = UInt64(0x3F2089CDD5E44BE8)
comptime _P1 = UInt64(0x3F9F06D10CCA2C7E)
comptime _P2 = UInt64(0x3FF0000000000000)
comptime _Q0 = UInt64(0x3EC92EB6BC365FA0)
comptime _Q1 = UInt64(0x3F64AE39B508B6C0)
comptime _Q2 = UInt64(0x3FCD17099887E074)
comptime _Q3 = UInt64(0x4000000000000000)
comptime _EXP_HI = UInt64(0x40862E42FEFA39EF)  # 709.782712893384
comptime _EXP_LO = UInt64(0xC086232BDD7ABCD2)  # -708.3964185322641
comptime _SQRTH = UInt64(0x3FE6A09E667F3BCD)
comptime _L_R0 = UInt64(0xBFE9443DDC6C0E84)
comptime _L_R1 = UInt64(0x403062FC73027B6B)
comptime _L_R2 = UInt64(0xC050090611222A20)
comptime _L_S0 = UInt64(0xC041D60D43EC6D0A)
comptime _L_S1 = UInt64(0x40738180112AE40E)
comptime _L_S2 = UInt64(0xC0880D8919B33F3B)
comptime _L_C2 = UInt64(0xBF2BD0105C610CA8)  # -2.121944400546905827679e-4
comptime _L_C1 = UInt64(0x3FE6300000000000)  # 0.693359375
comptime _L_P0 = UInt64(0x3F1AB4C293C31BB0)
comptime _L_P1 = UInt64(0x3FDFD6F53F5652F2)
comptime _L_P2 = UInt64(0x4012D2BAED926911)
comptime _L_P3 = UInt64(0x402CFF72C63EEB2E)
comptime _L_P4 = UInt64(0x4031EFD6924BC84D)
comptime _L_P5 = UInt64(0x401ED5637D7EDCF8)
comptime _L_Q0 = UInt64(0x40269320AE97EF8E)
comptime _L_Q1 = UInt64(0x40469D2C4E19C033)
comptime _L_Q2 = UInt64(0x4054BF33A326BDBD)
comptime _L_Q3 = UInt64(0x4051C9E2EB5EAE21)
comptime _L_Q4 = UInt64(0x4037200A9E1F25B2)
comptime _NHALF = UInt64(0xBFE0000000000000)
comptime _NONE = UInt64(0xBFF0000000000000)
comptime _TWO54 = UInt64(0x4350000000000000)


@always_inline
def _clz64(x: UInt64) -> Int:
    """Leading zeros, by shifts only (no ctlz intrinsic, so every backend
    takes the same instructions)."""
    if x == 0:
        return 64
    var n = 0
    var v = x
    if (v & UInt64(0xFFFFFFFF00000000)) == 0:
        n += 32
        v <<= 32
    if (v & UInt64(0xFFFF000000000000)) == 0:
        n += 16
        v <<= 16
    if (v & UInt64(0xFF00000000000000)) == 0:
        n += 8
        v <<= 8
    if (v & UInt64(0xF000000000000000)) == 0:
        n += 4
        v <<= 4
    if (v & UInt64(0xC000000000000000)) == 0:
        n += 2
        v <<= 2
    if (v & UInt64(0x8000000000000000)) == 0:
        n += 1
    return n


@always_inline
def _exp_of(a: UInt64) -> Int:
    return Int((a >> 52) & UInt64(0x7FF))


@always_inline
def _pack(sign: UInt64, exp: Int, sig: UInt64) -> UInt64:
    """`packToF64UI`: an ADDITION, so a significand carry bumps the exponent."""
    return (sign << 63) + (UInt64(exp) << 52) + sig


@always_inline
def _jam(a: UInt64, dist: Int) -> UInt64:
    """`softfloat_shiftRightJam64`."""
    if dist <= 0:
        return a
    if dist < 63:
        var lost = a << UInt64(64 - dist)
        return (a >> UInt64(dist)) | (UInt64(1) if lost != 0 else UInt64(0))
    return UInt64(1) if a != 0 else UInt64(0)


def _round_pack(sign: UInt64, exp_in: Int, sig_in: UInt64) -> UInt64:
    """`softfloat_roundPackToF64`, near-even. `sig_in` has its leading bit
    at 62 and ten rounding bits below the 53-bit result."""
    var exp = exp_in
    var sig = sig_in
    var round_bits = sig & UInt64(0x3FF)
    if exp < 0:
        sig = _jam(sig, -exp)
        exp = 0
        round_bits = sig & UInt64(0x3FF)
    elif exp > 0x7FD or (
        exp == 0x7FD and sig + UInt64(0x200) >= UInt64(0x8000000000000000)
    ):
        return _pack(sign, 0x7FF, 0)
    sig = (sig + UInt64(0x200)) >> 10
    if round_bits == UInt64(0x200):
        sig &= ~UInt64(1)
    if sig == 0:
        exp = 0
    return _pack(sign, exp, sig)


def _norm_round_pack(sign: UInt64, exp_in: Int, sig: UInt64) -> UInt64:
    """`softfloat_normRoundPackToF64`."""
    var sd = _clz64(sig) - 1
    var exp = exp_in - sd
    if sd >= 10 and exp >= 0 and exp < 0x7FD:
        return _pack(sign, exp if sig != 0 else 0, sig << UInt64(sd - 10))
    return _round_pack(sign, exp, sig << UInt64(sd))


@always_inline
def sf64_is_nan(a: UInt64) -> Bool:
    return (a & ~SF64_SIGN) > SF64_INF


@always_inline
def sf64_neg(a: UInt64) -> UInt64:
    return a ^ SF64_SIGN


def _add_mags(a: UInt64, b: UInt64, sign: UInt64) -> UInt64:
    var ea = _exp_of(a)
    var sa = a & SF64_FRAC
    var eb = _exp_of(b)
    var sb = b & SF64_FRAC
    var d = ea - eb
    var ez: Int
    var sz: UInt64
    if d == 0:
        if ea == 0:
            return a + sb
        if ea == 0x7FF:
            if (sa | sb) != 0:
                return SF64_NAN
            return a
        ez = ea
        sz = (UInt64(0x0020000000000000) + sa + sb) << 9
        return _round_pack(sign, ez, sz)
    sa <<= 9
    sb <<= 9
    if d < 0:
        if eb == 0x7FF:
            if sb != 0:
                return SF64_NAN
            return _pack(sign, 0x7FF, 0)
        ez = eb
        if ea != 0:
            sa += UInt64(0x2000000000000000)
        else:
            sa <<= 1
        sa = _jam(sa, -d)
    else:
        if ea == 0x7FF:
            if sa != 0:
                return SF64_NAN
            return a
        ez = ea
        if eb != 0:
            sb += UInt64(0x2000000000000000)
        else:
            sb <<= 1
        sb = _jam(sb, d)
    sz = UInt64(0x2000000000000000) + sa + sb
    if sz < UInt64(0x4000000000000000):
        ez -= 1
        sz <<= 1
    return _round_pack(sign, ez, sz)


def _sub_mags(a: UInt64, b: UInt64, sign_in: UInt64) -> UInt64:
    var sign = sign_in
    var ea = _exp_of(a)
    var sa = a & SF64_FRAC
    var eb = _exp_of(b)
    var sb = b & SF64_FRAC
    var d = ea - eb
    if d == 0:
        if ea == 0x7FF:
            return SF64_NAN
        if sa == sb:
            return SF64_ZERO
        var e = ea
        if e != 0:
            e -= 1
        var u: UInt64
        if sa < sb:
            sign ^= 1
            u = sb - sa
        else:
            u = sa - sb
        var sd = _clz64(u) - 11
        var ez = e - sd
        if ez < 0:
            sd = e
            ez = 0
        return _pack(sign, ez, u << UInt64(sd))
    sa <<= 10
    sb <<= 10
    var ez: Int
    var sz: UInt64
    if d < 0:
        sign ^= 1
        if eb == 0x7FF:
            if sb != 0:
                return SF64_NAN
            return _pack(sign, 0x7FF, 0)
        if ea != 0:
            sa += UInt64(0x4000000000000000)
        else:
            sa += sa
        sa = _jam(sa, -d)
        sb |= UInt64(0x4000000000000000)
        ez = eb
        sz = sb - sa
    else:
        if ea == 0x7FF:
            if sa != 0:
                return SF64_NAN
            return a
        if eb != 0:
            sb += UInt64(0x4000000000000000)
        else:
            sb += sb
        sb = _jam(sb, d)
        sa |= UInt64(0x4000000000000000)
        ez = ea
        sz = sa - sb
    return _norm_round_pack(sign, ez - 1, sz)


def sf64_add(a: UInt64, b: UInt64) -> UInt64:
    """`a + b`, correctly rounded."""
    var sa = a >> 63
    if sa == (b >> 63):
        return _add_mags(a, b, sa)
    return _sub_mags(a, b, sa)


def sf64_sub(a: UInt64, b: UInt64) -> UInt64:
    """`a - b`, correctly rounded."""
    var sa = a >> 63
    if sa == (b >> 63):
        return _sub_mags(a, b, sa)
    return _add_mags(a, b, sa)


@always_inline
def _mul64to128(a: UInt64, b: UInt64) -> Tuple[UInt64, UInt64]:
    """(high, low) of the exact 128-bit product, by 32-bit limbs."""
    comptime M32 = UInt64(0xFFFFFFFF)
    var a0 = a & M32
    var a1 = a >> 32
    var b0 = b & M32
    var b1 = b >> 32
    var p00 = a0 * b0
    var p01 = a0 * b1
    var p10 = a1 * b0
    var p11 = a1 * b1
    var mid = (p00 >> 32) + (p01 & M32) + (p10 & M32)
    var lo = (p00 & M32) | (mid << 32)
    var hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32)
    return (hi, lo)


@always_inline
def _norm_sub_exp(sig: UInt64) -> Int:
    return 1 - (_clz64(sig) - 11)


@always_inline
def _norm_sub_sig(sig: UInt64) -> UInt64:
    return sig << UInt64(_clz64(sig) - 11)


def sf64_mul(a: UInt64, b: UInt64) -> UInt64:
    """`a * b`, correctly rounded (`f64_mul.c`)."""
    var sign = (a ^ b) >> 63
    var ea = _exp_of(a)
    var sa = a & SF64_FRAC
    var eb = _exp_of(b)
    var sb = b & SF64_FRAC
    if ea == 0x7FF:
        if sa != 0 or (eb == 0x7FF and sb != 0):
            return SF64_NAN
        if eb == 0 and sb == 0:
            return SF64_NAN
        return _pack(sign, 0x7FF, 0)
    if eb == 0x7FF:
        if sb != 0:
            return SF64_NAN
        if ea == 0 and sa == 0:
            return SF64_NAN
        return _pack(sign, 0x7FF, 0)
    if ea == 0:
        if sa == 0:
            return sign << 63
        ea = _norm_sub_exp(sa)
        sa = _norm_sub_sig(sa)
    if eb == 0:
        if sb == 0:
            return sign << 63
        eb = _norm_sub_exp(sb)
        sb = _norm_sub_sig(sb)
    var ez = ea + eb - 0x3FF
    sa = (sa | SF64_HIDDEN) << 10
    sb = (sb | SF64_HIDDEN) << 11
    var p = _mul64to128(sa, sb)
    var sz = p[0] | (UInt64(1) if p[1] != 0 else UInt64(0))
    if sz < UInt64(0x4000000000000000):
        ez -= 1
        sz <<= 1
    return _round_pack(sign, ez, sz)


def sf64_div(a: UInt64, b: UInt64) -> UInt64:
    """`a / b`, correctly rounded (restoring division, 63 quotient bits plus
    a sticky bit)."""
    var sign = (a ^ b) >> 63
    var ea = _exp_of(a)
    var sa = a & SF64_FRAC
    var eb = _exp_of(b)
    var sb = b & SF64_FRAC
    if ea == 0x7FF:
        if sa != 0 or eb == 0x7FF:
            return SF64_NAN
        return _pack(sign, 0x7FF, 0)
    if eb == 0x7FF:
        if sb != 0:
            return SF64_NAN
        return sign << 63
    if eb == 0:
        if sb == 0:
            if ea == 0 and sa == 0:
                return SF64_NAN
            return _pack(sign, 0x7FF, 0)
        eb = _norm_sub_exp(sb)
        sb = _norm_sub_sig(sb)
    if ea == 0:
        if sa == 0:
            return sign << 63
        ea = _norm_sub_exp(sa)
        sa = _norm_sub_sig(sa)
    var ez = ea - eb + 0x3FE
    sa |= SF64_HIDDEN
    sb |= SF64_HIDDEN
    if sa < sb:
        ez -= 1
        sa <<= 1
    var rem = sa
    var q = UInt64(0)
    for _ in range(63):
        q <<= 1
        if rem >= sb:
            rem -= sb
            q |= 1
        rem <<= 1
    if rem != 0:
        q |= 1
    return _round_pack(sign, ez, q)


@always_inline
def _shr128_jam(h: UInt64, l: UInt64, d: Int) -> Tuple[UInt64, UInt64]:
    if d <= 0:
        return (h, l)
    if d < 64:
        var sticky = (l << UInt64(64 - d)) != 0
        var nl = (l >> UInt64(d)) | (h << UInt64(64 - d))
        if sticky:
            nl |= 1
        return (h >> UInt64(d), nl)
    if d < 128:
        var nl = h >> UInt64(d - 64) if d > 64 else h
        var sticky = l != 0
        if d > 64 and (h << UInt64(128 - d)) != 0:
            sticky = True
        if sticky:
            nl |= 1
        return (UInt64(0), nl)
    return (UInt64(0), UInt64(1) if (h | l) != 0 else UInt64(0))


def sf64_fma(a: UInt64, b: UInt64, c: UInt64) -> UInt64:
    """`a * b + c` with ONE rounding: the exact product (106 bits) and `c`
    aligned in 128 bits with a sticky bit, summed, rounded once."""
    var ea = _exp_of(a)
    var sa = a & SF64_FRAC
    var eb = _exp_of(b)
    var sb = b & SF64_FRAC
    var ec = _exp_of(c)
    var sc = c & SF64_FRAC
    var sign_p = (a ^ b) >> 63
    var sign_c = c >> 63
    if sf64_is_nan(a) or sf64_is_nan(b) or sf64_is_nan(c):
        return SF64_NAN
    var a_zero = ea == 0 and sa == 0
    var b_zero = eb == 0 and sb == 0
    if ea == 0x7FF or eb == 0x7FF:
        if a_zero or b_zero:
            return SF64_NAN
        if ec == 0x7FF and sign_c != sign_p:
            return SF64_NAN
        return _pack(sign_p, 0x7FF, 0)
    if ec == 0x7FF:
        return c
    if a_zero or b_zero:
        return sf64_add(sign_p << 63, c)
    if ea == 0:
        ea = _norm_sub_exp(sa)
        sa = _norm_sub_sig(sa)
    if eb == 0:
        eb = _norm_sub_exp(sb)
        sb = _norm_sub_sig(sb)
    var p = _mul64to128(sa | SF64_HIDDEN, sb | SF64_HIDDEN)
    # product in [2^104, 2^106): lift by 20 so its leading bit is at 124/125
    var ph = (p[0] << 20) | (p[1] >> 44)
    var pl = p[1] << 20
    var ep = ea + eb - 2150 - 20  # exponent of bit 0
    var h: UInt64
    var l: UInt64
    var e: Int
    var sign: UInt64
    if ec == 0 and sc == 0:
        h = ph
        l = pl
        e = ep
        sign = sign_p
    else:
        if ec == 0:
            ec = _norm_sub_exp(sc)
            sc = _norm_sub_sig(sc)
        var ch = (sc | SF64_HIDDEN) << 8  # mc * 2^72
        var cl = UInt64(0)
        var ecl = ec - 1075 - 72
        if ep >= ecl:
            var t = _shr128_jam(ch, cl, ep - ecl)
            ch = t[0]
            cl = t[1]
            e = ep
        else:
            var t2 = _shr128_jam(ph, pl, ecl - ep)
            ph = t2[0]
            pl = t2[1]
            e = ecl
        if sign_p == sign_c:
            l = pl + cl
            h = ph + ch + (UInt64(1) if l < pl else UInt64(0))
            sign = sign_p
        else:
            var p_big = ph > ch or (ph == ch and pl >= cl)
            if ph == ch and pl == cl:
                return SF64_ZERO
            if p_big:
                l = pl - cl
                h = ph - ch - (UInt64(1) if pl < cl else UInt64(0))
                sign = sign_p
            else:
                l = cl - pl
                h = ch - ph - (UInt64(1) if cl < pl else UInt64(0))
                sign = sign_c
    var lead: Int
    if h != 0:
        lead = 127 - _clz64(h)
    else:
        lead = 63 - _clz64(l)
    var sig: UInt64
    if lead >= 62:
        sig = _shr128_jam(h, l, lead - 62)[1]
    else:
        sig = l << UInt64(62 - lead)
    return _round_pack(sign, e + lead + 1022, sig)


def sf64_lt(a: UInt64, b: UInt64) -> Bool:
    """`a < b` for non-NaN operands (+0 == -0)."""
    if ((a | b) & ~SF64_SIGN) == 0:
        return False
    var sa = a >> 63
    var sb = b >> 63
    if sa != sb:
        return sa == 1
    if sa == 0:
        return a < b
    return a > b


@always_inline
def sf64_gt(a: UInt64, b: UInt64) -> Bool:
    return sf64_lt(b, a)


def sf64_floor(a: UInt64) -> UInt64:
    var ea = _exp_of(a)
    if ea == 0x7FF:
        return SF64_NAN if sf64_is_nan(a) else a
    var e = ea - 1023
    if e < 0:
        if (a & ~SF64_SIGN) == 0:
            return a
        return _NONE if (a >> 63) != 0 else SF64_ZERO
    if e >= 52:
        return a
    var mask = SF64_FRAC >> UInt64(e)
    if (a & mask) == 0:
        return a
    var r = a
    if (a >> 63) != 0:
        r = a + mask
    return r & ~mask


def sf64_to_int(a: UInt64) -> Int:
    """Truncation toward zero of a finite value with |a| < 2^62."""
    var e = _exp_of(a) - 1023
    if e < 0:
        return 0
    var m = (a & SF64_FRAC) | SF64_HIDDEN
    var v: UInt64
    if e >= 52:
        v = m << UInt64(e - 52)
    else:
        v = m >> UInt64(52 - e)
    if (a >> 63) != 0:
        return -Int(v)
    return Int(v)


def sf64_from_int(i: Int) -> UInt64:
    """Exact for |i| < 2^53."""
    if i == 0:
        return SF64_ZERO
    var s = UInt64(1) if i < 0 else UInt64(0)
    var u = UInt64(-i) if i < 0 else UInt64(i)
    var sd = _clz64(u) - 11
    return _pack(s, 1074 - sd, u << UInt64(sd))


def sf64_from_f32(x: Float32) -> UInt64:
    """The exact widening."""
    var b = UInt64(bitcast[DType.uint32](x))
    var s = b >> 31
    var e = Int((b >> 23) & UInt64(0xFF))
    var f = b & UInt64(0x7FFFFF)
    if e == 0xFF:
        if f != 0:
            return SF64_NAN
        return _pack(s, 0x7FF, 0)
    if e == 0:
        if f == 0:
            return s << 63
        var lead = 63 - _clz64(f)  # 0..22
        var frac = (f << UInt64(52 - lead)) & SF64_FRAC
        return (s << 63) | (UInt64(lead - 149 + 1023) << 52) | frac
    return (s << 63) | (UInt64(e + 896) << 52) | (f << 29)


def sf64_to_f32(a: UInt64) -> Float32:
    """Round-to-nearest-even narrowing (`f64_to_f32.c`)."""
    var s = a >> 63
    var e = _exp_of(a)
    var f = a & SF64_FRAC
    if e == 0x7FF:
        if f != 0:
            return bitcast[DType.float32](UInt32(0x7FC00000))
        return bitcast[DType.float32](UInt32((s << 31) | UInt64(0x7F800000)))
    var f30 = _jam(f, 22)
    if e == 0 and f30 == 0:
        return bitcast[DType.float32](UInt32(s << 31))
    var exp = e - 0x381
    var sig = f30 | UInt64(0x40000000)
    var round_bits = sig & UInt64(0x7F)
    if exp < 0:
        sig = _jam(sig, -exp)
        exp = 0
        round_bits = sig & UInt64(0x7F)
    elif exp > 0xFD or (exp == 0xFD and sig + UInt64(0x40) >= UInt64(0x80000000)):
        return bitcast[DType.float32](UInt32((s << 31) | UInt64(0x7F800000)))
    sig = (sig + UInt64(0x40)) >> 7
    if round_bits == UInt64(0x40):
        sig &= ~UInt64(1)
    if sig == 0:
        exp = 0
    return bitcast[DType.float32](UInt32((s << 31) + (UInt64(exp) << 23) + sig))


def sf64_sqrt(a: UInt64) -> UInt64:
    """`sqrt(a)`, correctly rounded (round-to-nearest-even), as IEEE-754
    requires of a hardware double `sqrt` (lane fix-g1-gbdt, 2026-10-04; NEW,
    no existing function changed). Special cases: NaN -> `SF64_NAN`;
    `+-0` -> itself; a negative nonzero -> `SF64_NAN`; `+inf` -> `+inf`.

    Digit-by-digit (restoring) square root. The finite positive input is
    `m * 2^(E-52)` with `m` normalized to 53 bits; an odd `E` moves one bit
    into `m` so `E'` is even. The radicand `R = m' << 56` (110 bits) gives a
    55-bit root `r = floor(sqrt(R))` in `[2^54, 2^55)`: 53 result bits, one
    round bit, one more, plus the remainder as the sticky bit, so
    `_round_pack` rounds once and correctly (a square root is never exactly
    halfway). `R`'s low 56 bits are zero, so the two radicand bits taken at
    step `i` are `m'` bits `53-2i` and `52-2i` while `i <= 26`, else 0. The
    remainder stays under `2^59`, inside one word."""
    var ea = _exp_of(a)
    var sa = a & SF64_FRAC
    if ea == 0x7FF:
        if sa != 0:
            return SF64_NAN
        if (a >> 63) != 0:
            return SF64_NAN
        return a
    if (a & ~SF64_SIGN) == 0:
        return a
    if (a >> 63) != 0:
        return SF64_NAN
    if ea == 0:
        ea = _norm_sub_exp(sa)
        sa = _norm_sub_sig(sa)
    var m = sa | SF64_HIDDEN
    var e_unb = ea - 1023
    if (e_unb & 1) != 0:
        m <<= 1
        e_unb -= 1
    var root = UInt64(0)
    var rem = UInt64(0)
    for i in range(55):
        var pair = UInt64(0)
        if i <= 26:
            pair = (m >> UInt64(52 - 2 * i)) & UInt64(3)
        rem = (rem << 2) | pair
        var trial = (root << 2) | UInt64(1)
        if rem >= trial:
            rem -= trial
            root = (root << 1) | UInt64(1)
        else:
            root = root << 1
    var sig = root << 8
    if rem != 0:
        sig |= UInt64(1)
    # value = sig * 2^(exp - 1084) and sqrt = r * 2^(E'/2 - 54), so
    # exp = 1022 + E'/2 (E' even; `//` is exact here)
    return _round_pack(UInt64(0), 1022 + e_unb // 2, sig)


@always_inline
def sf64_ftz(a: UInt64) -> UInt64:
    """A subnormal becomes its signed zero (the FTZ+DAZ host pool's
    behavior on a result, applied explicitly so every column agrees)."""
    if _exp_of(a) == 0:
        return a & SF64_SIGN
    return a


def sf64_exp(x: UInt64) -> UInt64:
    """`portable_exp64`, statement for statement."""
    if sf64_is_nan(x):
        return SF64_NAN
    if sf64_gt(x, _EXP_HI):
        return SF64_INF
    if sf64_lt(x, _EXP_LO):
        return SF64_ZERO
    var k = sf64_floor(sf64_fma(x, _LOG2E, _HALF))
    var r = sf64_fma(k, _NC1, x)
    r = sf64_fma(k, _NC2, r)
    var xx = sf64_mul(r, r)
    var px = sf64_fma(_P0, xx, _P1)
    px = sf64_fma(px, xx, _P2)
    px = sf64_mul(px, r)
    var qx = sf64_fma(_Q0, xx, _Q1)
    qx = sf64_fma(qx, xx, _Q2)
    qx = sf64_fma(qx, xx, _Q3)
    var y = sf64_div(px, sf64_sub(qx, px))
    y = sf64_fma(_TWO, y, SF64_ONE)
    var ki = sf64_to_int(k)
    var k1 = ki >> 1
    var k2 = ki - k1
    y = sf64_mul(y, UInt64(k1 + 1023) << 52)
    y = sf64_mul(y, UInt64(k2 + 1023) << 52)
    return y


def sf64_log(x_in: UInt64) -> UInt64:
    """`portable_log64`, statement for statement."""
    var x = x_in
    if sf64_is_nan(x):
        return SF64_NAN
    if (x & ~SF64_SIGN) == 0:
        return SF64_INF | SF64_SIGN
    if (x >> 63) != 0:
        return SF64_NAN
    if x == SF64_INF:
        return x
    var e = 0
    if _exp_of(x) == 0:
        x = sf64_mul(x, _TWO54)
        e = -54
    e += _exp_of(x) - 1022
    var m = (x & SF64_FRAC) | UInt64(0x3FE0000000000000)
    var z: UInt64
    var y: UInt64
    var xm: UInt64
    if e > 2 or e < -2:
        if sf64_lt(m, _SQRTH):
            e -= 1
            z = sf64_sub(m, _HALF)
            y = sf64_fma(_HALF, z, _HALF)
        else:
            z = sf64_sub(m, _HALF)
            z = sf64_sub(z, _HALF)
            y = sf64_fma(_HALF, m, _HALF)
        xm = sf64_div(z, y)
        z = sf64_mul(xm, xm)
        var r = sf64_fma(_L_R0, z, _L_R1)
        r = sf64_fma(r, z, _L_R2)
        var sq = sf64_add(z, _L_S0)
        sq = sf64_fma(sq, z, _L_S1)
        sq = sf64_fma(sq, z, _L_S2)
        z = sf64_mul(xm, sf64_div(sf64_mul(z, r), sq))
        var ye = sf64_from_int(e)
        z = sf64_fma(ye, _L_C2, z)
        z = sf64_add(z, xm)
        z = sf64_fma(ye, _L_C1, z)
        return z
    if sf64_lt(m, _SQRTH):
        e -= 1
        xm = sf64_fma(_TWO, m, _NONE)
    else:
        xm = sf64_sub(m, SF64_ONE)
    z = sf64_mul(xm, xm)
    var pp = sf64_fma(_L_P0, xm, _L_P1)
    pp = sf64_fma(pp, xm, _L_P2)
    pp = sf64_fma(pp, xm, _L_P3)
    pp = sf64_fma(pp, xm, _L_P4)
    pp = sf64_fma(pp, xm, _L_P5)
    var qq = sf64_add(xm, _L_Q0)
    qq = sf64_fma(qq, xm, _L_Q1)
    qq = sf64_fma(qq, xm, _L_Q2)
    qq = sf64_fma(qq, xm, _L_Q3)
    qq = sf64_fma(qq, xm, _L_Q4)
    y = sf64_mul(xm, sf64_div(sf64_mul(z, pp), qq))
    var ye2 = sf64_from_int(e)
    y = sf64_fma(ye2, _L_C2, y)
    y = sf64_fma(z, _NHALF, y)
    z = sf64_add(xm, y)
    z = sf64_fma(ye2, _L_C1, z)
    return z


# ---- the GBDT probability links, one row-element each -------------------


def sf64_sigmoid_f64(raw: UInt64) -> UInt64:
    """`1 / (1 + exp(-raw))` in double."""
    var e = sf64_exp(sf64_neg(raw))
    return sf64_div(SF64_ONE, sf64_add(SF64_ONE, e))


def sf64_sigmoid_f32(raw: Float32) -> UInt64:
    """`sf64_sigmoid_f64` over the exact widening of `raw`."""
    return sf64_sigmoid_f64(sf64_from_f32(raw))
