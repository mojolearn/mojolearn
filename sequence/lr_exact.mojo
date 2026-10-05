# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The learning-rate schedules' EXACT route in Mojo (lane py-runtime round
3, owner decision: a Mojo big rational, sequence/bigrat.mojo). It was
Python's `fractions.Fraction` arithmetic in `_training_impl` (`_f32_round`,
`_cos_taylor`, `_cos_pi_interval`, `_decide_f32`, the schedules'
`_lr_at_slow`) and `_x_sequence_sched` (OneCycleLR's `_lr_at_slow`,
StepLR / ExponentialLR's `_pow_value` enclosures).

THE SAME DEFINITION, THE SAME ANSWERS. A schedule's value is the float32
nearest an exact rational (warmup, linear, OneCycle linear, base gamma^e),
or, for the cosine forms, the float32 both ends of Python's Taylor interval
round to: the SAME rational interval (pi between the same 60-digit decimal
bounds, the same Taylor partial sums and remainder bound, the same 24, 32,
48, 64 term escalation and the same refusal when the 64-term interval
still straddles a float32 boundary). Every value here is an exact
rational, so the answers are Python's bit for bit. Python's faster
enclosures (OneCycle's fixed point, the gamma^e P-bit intervals) only ever
returned that same float32 when they decided; here the gamma^e enclosure
widens until it decides, and the exact power answers the rest, as
Python's last fallback did.

Host scalar code on every column: a schedule value is one number per
optimizer step (the CPU-only route by nature, DEVIATION 5540)."""
from std.math import isfinite
from std.memory import bitcast

from sequence.bigrat import (
    F32Round, Int_, Nat, Rat, nadd, ncmp, nfrom_decimal, nlow_zero, nmul, npow, nshl, nshr, nsub,
    nat_f32, rabs, radd, rat_f32, rat_of_f64, rcmp, rdiv, rmul, rneg, rsub, zadd,
)

comptime _PI_DIGITS = "3141592653589793238462643383279502884197169399375105820974944"

#: statuses returned to Python
comptime LR_OK = 0
comptime LR_OVERFLOW = 1        # _f32_round's OverflowError
comptime LR_STRADDLE = 2        # _decide_f32's ArithmeticError
comptime LR_BAD_P = 3           # _cos_pi_interval's ValueError


@fieldwise_init
struct LrOut(Copyable, Movable):
    var value: Float32
    var status: Int


def _pi_bounds() raises -> Tuple[Rat, Rat]:
    var digits = nfrom_decimal(_PI_DIGITS)
    var den = npow(Nat.of(10), 60)
    var lo = Rat(Int_(False, digits.copy()), den.copy())
    var hi = Rat(Int_(False, nadd(digits, Nat.of(1))), den^)
    return (lo^, hi^)


def _cos_taylor(x: Rat, terms: Int) -> Tuple[Rat, Rat]:
    """sum_{k<terms} (-1)^k x^(2k) / (2k)! and the next term's magnitude
    x^(2 terms) / (2 terms)! (`_training_impl._cos_taylor`), over one common
    denominator x_d^(2(terms-1)) (2(terms-1))! (exact, no reduction)."""
    var K = terms
    var a = nmul(x.n.m, x.n.m)          # x_n^2
    var b = nmul(x.d, x.d)              # x_d^2
    # powers a^k, b^k for k < K
    var ap = List[Nat]()
    var bp = List[Nat]()
    ap.append(Nat.of(1))
    bp.append(Nat.of(1))
    for k in range(1, K):
        ap.append(nmul(ap[k - 1], a))
        bp.append(nmul(bp[k - 1], b))
    # F_k = (2(K-1))! / (2k)!, descending from F_{K-1} = 1
    var F = List[Nat](length=K, fill=Nat.of(1))
    var k = K - 2
    while k >= 0:
        F[k] = nmul(F[k + 1], Nat.of(UInt64((2 * k + 1) * (2 * k + 2))))
        k -= 1
    var num = Int_.of(0)
    for j in range(K):
        var term = nmul(nmul(ap[j], bp[K - 1 - j]), F[j])
        num = zadd(num, Int_((j & 1) == 1, term^))
    # (2(K-1))! for the common denominator
    var fact = Nat.of(1)
    for v in range(1, 2 * (K - 1) + 1):
        fact = nmul(fact, Nat.of(UInt64(v)))
    var den = nmul(bp[K - 1], fact)
    var total = Rat(num^, den^)
    # next term magnitude: a^K / (b^K (2K)!)
    var factK = nmul(nmul(fact, Nat.of(UInt64(2 * K - 1))), Nat.of(UInt64(2 * K)))
    var rem = Rat(Int_(False, nmul(ap[K - 1], a)), nmul(nmul(bp[K - 1], b), factK))
    return (total^, rem^)


def cos_pi_interval(p: Rat, terms: Int, mut lo: Rat, mut hi: Rat) raises -> Int:
    """`_training_impl._cos_pi_interval`: [lo, hi] enclosing cos(pi p) for a
    rational p in [0, 1]; exact at 0, 1/3, 1/2, 2/3, 1. Returns LR_BAD_P
    outside [0, 1], else LR_OK."""
    if p.sign() < 0 or rcmp(p, Rat.of(1)) > 0:
        return LR_BAD_P
    if p.sign() == 0:
        lo = Rat.of(1)
        hi = Rat.of(1)
        return LR_OK
    if rcmp(p, Rat.frac(1, 3)) == 0:
        lo = Rat.frac(1, 2)
        hi = Rat.frac(1, 2)
        return LR_OK
    if rcmp(p, Rat.frac(1, 2)) == 0:
        lo = Rat.of(0)
        hi = Rat.of(0)
        return LR_OK
    if rcmp(p, Rat.frac(2, 3)) == 0:
        lo = Rat.frac(-1, 2)
        hi = Rat.frac(-1, 2)
        return LR_OK
    if rcmp(p, Rat.of(1)) == 0:
        lo = Rat.of(-1)
        hi = Rat.of(-1)
        return LR_OK
    var q = p.copy()
    var flip = rcmp(q, Rat.frac(1, 2)) > 0
    if flip:
        q = rsub(Rat.of(1), q)
    var pis = _pi_bounds()
    var x_lo = rmul(pis[0], q)
    var x_hi = rmul(pis[1], q)
    var th = _cos_taylor(x_hi, terms)
    var tl = _cos_taylor(x_lo, terms)
    var l = rsub(th[0], th[1])
    var h = radd(tl[0], tl[1])
    if flip:
        lo = rneg(h)
        hi = rneg(l)
    else:
        lo = l^
        hi = h^
    return LR_OK


def _decide_cos(a: Rat, b: Rat, p: Rat) raises -> LrOut:
    """`_decide_f32` of the interval a + (b - a)... in the two cosine forms:
    the value b_ + w (c + 1) for c in Python's Taylor interval, here given
    as base `a` and weight `b` (value = a + b (1 + c)); the ends ordered,
    rounded, kept when equal; 24, 32, 48, 64 terms."""
    for li in range(4):
        var clo = Rat.of(0)
        var chi = Rat.of(0)
        var st = cos_pi_interval(p, 24 if li == 0 else (32 if li == 1 else (48 if li == 2 else 64)), clo, chi)
        if st != LR_OK:
            return LrOut(Float32(0), st)
        var x = radd(a, rmul(b, radd(Rat.of(1), clo)))
        var y = radd(a, rmul(b, radd(Rat.of(1), chi)))
        if rcmp(x, y) > 0:
            var t = x^
            x = y^
            y = t^
        var fa = rat_f32(x)
        if fa.overflow:
            return LrOut(Float32(0), LR_OVERFLOW)
        var fb = rat_f32(y)
        if fb.overflow:
            return LrOut(Float32(0), LR_OVERFLOW)
        if fa.value == fb.value:
            return LrOut(fa.value, LR_OK)
    return LrOut(Float32(0), LR_STRADDLE)


def _round(q: Rat) -> LrOut:
    var r = rat_f32(q)
    if r.overflow:
        return LrOut(Float32(0), LR_OVERFLOW)
    return LrOut(r.value, LR_OK)


def schedule_exact(kind: Int, peak_f: Float64, lo_f: Float64, w: Int, total: Int, t: Int) raises -> LrOut:
    """`_Schedule._lr_at_slow(t)` (t >= 1; total < 0: none): kind 0 constant,
    1 linear, 2 cosine."""
    var peak = rat_of_f64(peak_f)
    var lo = rat_of_f64(lo_f)
    if t <= w:
        return _round(rmul(peak, Rat.frac(t, w)))
    if total < 0:
        return _round(peak)
    if t >= total:
        return _round(lo)
    var p = Rat.frac(t - w, total - w)
    if kind == 1:
        return _round(radd(peak, rmul(rsub(lo, peak), p)))
    if kind == 2:
        # lo + (peak - lo) (1 + c) / 2
        return _decide_cos(lo, rdiv(rsub(peak, lo), Rat.of(2)), p)
    return _round(peak)


def onecycle_exact(linear: Bool, three: Bool, max_lr: Float64, div: Float64, fdiv: Float64,
                   e1: Float64, e2: Float64, total_steps: Int, step: Int) raises -> LrOut:
    """`OneCycleLR._lr_at_slow` for torch's step (t - 1 <= total_steps):
    the phases (end, a, b) built as Python builds them, the linear value's
    exact rounding, the cosine value's interval decision."""
    var mx = rat_of_f64(max_lr)
    var init = rdiv(mx, rat_of_f64(div))
    var low = rdiv(init, rat_of_f64(fdiv))
    var ends = List[Rat]()
    var aa = List[Rat]()
    var bb = List[Rat]()
    ends.append(rat_of_f64(e1))
    aa.append(init.copy())
    bb.append(mx.copy())
    if three:
        ends.append(rat_of_f64(e2))
        aa.append(mx.copy())
        bb.append(init.copy())
        ends.append(Rat.of(total_steps - 1))
        aa.append(init.copy())
        bb.append(low.copy())
    else:
        ends.append(Rat.of(total_steps - 1))
        aa.append(mx.copy())
        bb.append(low.copy())
    var start = Rat.of(0)
    var s = Rat.of(step)
    var last = len(ends) - 1
    for i in range(len(ends)):  # small-loop(ends: the at most three OneCycle phases): phase membership
        if rcmp(s, ends[i]) <= 0 or i == last:
            if rcmp(ends[i], start) == 0:
                return _round(bb[i])
            var pct = rdiv(rsub(s, start), rsub(ends[i], start))
            if linear:
                return _round(radd(rmul(rsub(bb[i], aa[i]), pct), aa[i]))
            if pct.sign() <= 0:
                return _round(aa[i])
            if rcmp(pct, Rat.of(1)) == 0:
                return _round(bb[i])
            if rcmp(pct, Rat.of(1)) > 0:
                pct = rsub(Rat.of(2), pct)
            # b + (a - b) / 2 (c + 1)
            return _decide_cos(bb[i], rdiv(rsub(aa[i], bb[i]), Rat.of(2)), pct)
        start = ends[i].copy()
    return LrOut(Float32(0), LR_BAD_P)


# --------------------------------------------------------------- base gamma^e


def _mag(x: Float64) -> Tuple[Nat, Int]:
    """|x| = M 2^E exactly, M odd (x finite, nonzero)."""
    var bits = bitcast[DType.uint64](x)
    var ex = Int((bits >> 52) & 0x7FF)
    var man = bits & UInt64(0xFFFFFFFFFFFFF)
    var e: Int
    if ex == 0:
        e = -1074
    else:
        man |= UInt64(1) << 52
        e = ex - 1075
    while (man & 1) == 0:
        man >>= 1
        e += 1
    return (Nat.of(man), e)


def _trim(mut lo: Nat, mut hi: Nat, mut X: Int, P: Int):
    """[lo, hi] 2^X narrowed to P bits: lo down, hi up."""
    var s = hi.bits() - P
    if s > 0:
        lo = nshr(lo, s)
        var up = not nlow_zero(hi, s)
        hi = nshr(hi, s)
        if up:
            hi = nadd(hi, Nat.of(1))
        X += s


def _pow_iv(M: Nat, e: Int, P: Int, mut lo: Nat, mut hi: Nat, mut X: Int):
    """[lo, hi] 2^X enclosing M^e (`_x_sequence_sched._pow_iv`); P <= 0:
    exact (no trimming)."""
    lo = Nat.of(1)
    hi = Nat.of(1)
    X = 0
    var blo = M.copy()
    var bhi = M.copy()
    var bX = 0
    var k = e
    while k > 0:
        if (k & 1) == 1:
            lo = nmul(lo, blo)
            hi = nmul(hi, bhi)
            X += bX
            if P > 0:
                _trim(lo, hi, X, P)
        k >>= 1
        if k > 0:
            blo = nmul(blo, blo)
            bhi = nmul(bhi, bhi)
            bX = 2 * bX
            if P > 0:
                _trim(blo, bhi, bX, P)


def _dyadic_f32(M: Nat, E: Int) -> F32Round:
    """The float32 nearest M 2^E (M > 0), `_f32_round`'s rule."""
    if E >= 0:
        return nat_f32(nshl(M, E), Nat.of(1), False)
    return nat_f32(M, nshl(Nat.of(1), -E), False)


def pow_exact(base: Float64, gamma: Float64, e: Int) raises -> LrOut:
    """The float32 nearest base gamma^e (`_PowSched._pow_value`): the P-bit
    enclosure widened (128, 512, 2048, ... bits) until both ends round
    alike, then the exact power; the sign from base and gamma."""
    if base == 0.0 or (gamma == 0.0 and e > 0):
        return LrOut(Float32(0), LR_OK)
    if e == 0:
        return _round(rat_of_f64(base))
    var neg = (base < 0.0) != (gamma < 0.0 and (e & 1) == 1)
    var mb = _mag(base)
    var mg = _mag(gamma)
    var E0 = mb[1] + mg[1] * e
    var P = 128
    while True:
        var exact = P > mg[0].bits() * e + 64
        var lo = Nat.zero()
        var hi = Nat.zero()
        var X = 0
        _pow_iv(mg[0], e, 0 if exact else P, lo, hi, X)
        var a = _dyadic_f32(nmul(mb[0], lo), E0 + X)
        var b = _dyadic_f32(nmul(mb[0], hi), E0 + X)
        if exact:
            if a.overflow:
                return LrOut(Float32(0), LR_OVERFLOW)
            var v = a.value
            return LrOut(-v if neg and v != 0 else v, LR_OK)
        if not a.overflow and not b.overflow and a.value == b.value:
            var v = a.value
            return LrOut(-v if neg and v != 0 else v, LR_OK)
        P *= 4
