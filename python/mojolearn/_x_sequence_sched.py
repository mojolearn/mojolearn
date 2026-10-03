# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Learning-rate schedules of `torch.optim.lr_scheduler`: StepLR,
ExponentialLR and OneCycleLR, as closed forms of the step index. THE
CONTRACT (DEVIATION 5540): `lr_at(t)` is the exact rational value of the
closed form (the cosine of OneCycleLR as `_training_impl`'s rational
enclosure decides it) rounded ONCE to float32, ties to even, flushed to +0.0
below the smallest normal, so a schedule's bits are the same on every host.
`lr_at(t)` takes the ONE-BASED optimizer step, the convention of mojolearn's
schedules: `lr_at(1)` is torch's lr before its first `scheduler.step()`
(last_epoch 0), `lr_at(t)` its lr after t - 1 steps.

HOW (lane py-sequence): every value is first ENCLOSED in a fixed-width
integer interval [lo, hi] 2^E (directed rounding: lo rounded down, hi up, at
every product), a few machine words wide, and the float32 is returned only
when BOTH ends round to the same float32. Rounding with flush is monotone,
so the exact value, which lies inside, rounds to that float32 too: the bits
equal the exact-rational evaluation's by construction. When the ends differ
(the exact value sits on or within about 2^-100 of a float32 rounding
boundary) the exact `Fraction` evaluation of the old implementation decides
it (`_exact_lr_at`, kept as the reference). gamma^e is carried
incrementally across calls (one product per step), so ExponentialLR is O(1)
per step instead of the exact O(e) bits; OneCycleLR's cosine is a
fixed-point Taylor series with its Lagrange remainder instead of 24 to 64
rational terms. `sequence/checks/sched_check.py` compares every bit with an
independent 70-digit oracle and with `_exact_lr_at` over dense step ranges.

Pass one as `lr_schedule=` to the lane's optimizers (RMSprop, Adagrad, Lion,
Adamax, NAdam, LAMB) and recurrent estimators (LSTM/GRU/RNN), which set
`lr = schedule.lr_at(t)` before step t.

Not carried (sequence/NOT_IMPLEMENTED.tsv): OneCycleLR's momentum cycling
(cycle_momentum, base_momentum, max_momentum); per-group learning rates;
`last_epoch` resumption (evaluate `lr_at` at the step you resume from)."""
from . import _portable_math as _math
from fractions import Fraction

from ._training_impl import (_F64_EPS, _LrTable, _PI_HI, _PI_LO, _cos_pi_interval, _cos_pi_run,
                             _decide_f32, _f32_round)

_F32_MIN_NORMAL_EXP = -126
_F32_MAX_EXP = 127
#: working width of the fast enclosures, in bits
_P = 128
#: fixed-point fraction bits of OneCycleLR's cosine
_F = 160
_PI_LO_FX = (_PI_LO.numerator << _F) // _PI_LO.denominator            # <= pi 2^F
_PI_HI_FX = -((-(_PI_HI.numerator << _F)) // _PI_HI.denominator)      # >= pi 2^F
_EXACT_COS_POINTS = (Fraction(1, 3), Fraction(1, 2), Fraction(2, 3))


def _q(x, name):
    x = float(x)
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return Fraction(x)


# ------------------------------------------------------------ fixed-width enclosures
def _dy_f32(M, E):
    """The float32 nearest to M 2^E (M > 0 an int), ties to even, flushed to
    0.0 below the smallest normal: `_training_impl._f32_round`'s rounding of
    that exact dyadic. None where `_f32_round` would raise (overflow)."""
    sh = M.bit_length() - 24
    if sh > 0:
        m = M >> sh
        rem = M & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and (m & 1) == 1):
            m += 1
    else:
        m = M << (-sh)
    e = E + sh
    if m == (1 << 24):
        m = 1 << 23
        e += 1
    if e + 23 < _F32_MIN_NORMAL_EXP:
        return 0.0
    if e + 23 > _F32_MAX_EXP:
        return None
    return _math.ldexp(float(m), e)


def _iv_f32(lo, hi, E):
    """The float32 every value of [lo, hi] 2^E rounds to (lo <= hi ints of one
    sign, possibly negative), or None when the ends round apart or the
    interval touches zero."""
    if lo > 0:
        a, b = _dy_f32(lo, E), _dy_f32(hi, E)
        sign = 1.0
    elif hi < 0:
        a, b = _dy_f32(-hi, E), _dy_f32(-lo, E)
        sign = -1.0
    else:
        return None
    if a is None or a != b:
        return None
    return sign * a if a != 0.0 else 0.0


def _trim(lo, hi, X, P):
    """[lo, hi] 2^X narrowed to P bits: lo rounded down, hi up."""
    s = hi.bit_length() - P
    if s > 0:
        lo >>= s
        hi = -((-hi) >> s)
        X += s
    return lo, hi, X


def _mag(x):
    """(M, E) with |x| = M 2^E exactly, x a finite nonzero float."""
    m, e = _math.frexp(abs(x))
    return int(m * 9007199254740992.0), e - 53


def _pow_iv(M, e, P):
    """[lo, hi] 2^X enclosing M^e (M > 0, e >= 0), P-bit ends."""
    lo = hi = 1
    X = 0
    blo = bhi = M
    bX = 0
    while True:
        if e & 1:
            lo, hi, X = _trim(lo * blo, hi * bhi, X + bX, P)
        e >>= 1
        if not e:
            return lo, hi, X
        blo, bhi, bX = _trim(blo * blo, bhi * bhi, 2 * bX, P)


class _GammaPow:
    """gamma^e enclosures, carried from the last exponent asked for (one
    product per step on a forward walk; square-and-multiply otherwise)."""

    def __init__(self, gamma):
        self.M, self.Eg = _mag(gamma)
        self.e, self.lo, self.hi, self.X = 0, 1, 1, 0

    def at(self, e):
        d = e - self.e
        if 0 <= d <= 64:
            lo, hi, X, M = self.lo, self.hi, self.X, self.M
            for _ in range(d):
                lo, hi, X = _trim(lo * M, hi * M, X, _P)
        else:
            lo, hi, X = _pow_iv(self.M, e, _P)
        self.e, self.lo, self.hi, self.X = e, lo, hi, X
        return lo, hi, X + self.Eg * e


def _fx(q, F):
    """floor and ceil of the rational q times 2^F."""
    n, d = q.numerator << F, q.denominator
    return n // d, -((-n) // d)


def _cos_fx(x):
    """[lo, hi] 2^-_F enclosing cos(x 2^-_F), x >= 0 an int: the Taylor
    series with directed rounding and the Lagrange remainder x^(2k)/(2k)!."""
    F = _F
    xx = x * x
    x2lo, x2hi = xx >> F, -((-xx) >> F)
    lo = hi = tlo = thi = 1 << F
    k = 0
    while True:
        k += 1
        dd = ((2 * k - 1) * (2 * k)) << F
        tlo = (tlo * x2lo) // dd
        thi = -((-(thi * x2hi)) // dd)
        if thi <= 16:
            return lo - thi, hi + thi
        if k & 1:
            lo, hi = lo - thi, hi - tlo
        else:
            lo, hi = lo + tlo, hi + thi


# ------------------------------------------------------------------ schedules
class _Sched:
    def _t(self, t):
        t = int(t)
        if t < 1:
            raise ValueError("schedule step t is ONE-BASED")
        return t - 1           # torch's last_epoch

    def bits_at(self, t):
        import struct
        return struct.unpack("<I", struct.pack("<f", self.lr_at(t)))[0]

    def lrs(self, n):
        """lr_at(1) .. lr_at(n) as a list."""
        return [self.lr_at(t) for t in range(1, int(n) + 1)]


class _PowSched(_Sched):
    """base_lr gamma^E(t), E(t) the schedule's exponent."""

    def _init_pow(self):
        self._pow = None
        self._last = (None, None)
        b, g = self.base_lr, self.gamma
        if _math.isfinite(b) and _math.isfinite(g) and b != 0.0 and g != 0.0:
            self._pow = _GammaPow(g)
            self._Mb, self._Eb = _mag(b)

    def _pow_value(self, e):
        if self._last[0] == e:
            return self._last[1]
        v = None
        if self._pow is not None and e > 0:
            neg = (self.base_lr < 0) != (self.gamma < 0 and (e & 1) == 1)
            lo, hi, X = self._pow.at(e)
            if neg:
                lo, hi = -hi, -lo
            v = _iv_f32(self._Mb * lo, self._Mb * hi, self._Eb + X)
            if v is None:
                lo, hi, X = _pow_iv(self._pow.M, e, 4 * _P)
                X += self._pow.Eg * e
                if neg:
                    lo, hi = -hi, -lo
                v = _iv_f32(self._Mb * lo, self._Mb * hi, self._Eb + X)
        if v is None:
            v = _f32_round(_q(self.base_lr, "base_lr") * _q(self.gamma, "gamma") ** e)
        self._last = (e, v)
        return v


class StepLR(_PowSched):
    """lr = base_lr gamma^(epoch // step_size) (torch's closed form)."""

    def __init__(self, base_lr, step_size, gamma=0.1):
        if int(step_size) < 1:
            raise ValueError("StepLR: step_size must be >= 1")
        self.base_lr, self.step_size, self.gamma = float(base_lr), int(step_size), float(gamma)
        self._init_pow()

    def lr_at(self, t):
        return self._pow_value(self._t(t) // self.step_size)

    def _exact_lr_at(self, t):
        e = self._t(t)
        return _f32_round(_q(self.base_lr, "base_lr") * _q(self.gamma, "gamma") ** (e // self.step_size))


class ExponentialLR(_PowSched):
    """lr = base_lr gamma^epoch."""

    def __init__(self, base_lr, gamma):
        self.base_lr, self.gamma = float(base_lr), float(gamma)
        self._init_pow()

    def lr_at(self, t):
        return self._pow_value(self._t(t))

    def _exact_lr_at(self, t):
        return _f32_round(_q(self.base_lr, "base_lr") * _q(self.gamma, "gamma") ** self._t(t))


class OneCycleLR(_LrTable, _Sched):
    """torch's OneCycleLR learning rate: from max_lr / div_factor up to
    max_lr over pct_start of total_steps, then down to
    initial_lr / final_div_factor (or the three-phase form), by cosine or
    linear annealing."""

    def __init__(self, max_lr, total_steps, pct_start=0.3, anneal_strategy="cos", div_factor=25.0,
                 final_div_factor=1e4, three_phase=False):
        if int(total_steps) < 1:
            raise ValueError("OneCycleLR: total_steps must be >= 1")
        if not 0.0 <= float(pct_start) <= 1.0:
            raise ValueError("OneCycleLR: pct_start must lie in [0, 1]")
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError("OneCycleLR: anneal_strategy must be 'cos' or 'linear'")
        self.max_lr, self.total_steps = float(max_lr), int(total_steps)
        self.pct_start, self.anneal_strategy = float(pct_start), anneal_strategy
        self.div_factor, self.final_div_factor, self.three_phase = float(div_factor), float(final_div_factor), bool(three_phase)
        mx = _q(self.max_lr, "max_lr")
        init = mx / _q(self.div_factor, "div_factor")
        low = init / _q(self.final_div_factor, "final_div_factor")
        # torch builds the phase ends in float64: float(pct_start * total_steps) - 1
        p1 = Fraction(float(self.pct_start * self.total_steps) - 1)
        if self.three_phase:
            p2 = Fraction(float(2 * self.pct_start * self.total_steps) - 2)
            self._phases = [(p1, init, mx), (p2, mx, init), (Fraction(self.total_steps - 1), init, low)]
        else:
            self._phases = [(p1, init, mx), (Fraction(self.total_steps - 1), mx, low)]
        self._fixed = {}
        # lane gap-train-utils: the block table of `_training_impl._LrTable`
        # over steps 1 .. total_steps + 1 (no constant tail; beyond it the
        # exact route refuses, as torch does)
        self._table_setup(self.total_steps + 2, None)

    def _fast_values(self, t0, t1):
        """binary64 values and error bounds of steps t0 .. t1 - 1 (zeros where
        the exact route answers: a phase's two ends, its extra step, an empty
        phase, a step angle over 1/2)."""
        n = t1 - t0
        vs = [0.0] * n
        es = [0.0] * n
        s0 = t0 - 1                 # torch's step of index 0
        s1 = s0 + n - 1
        start = Fraction(0)
        last = len(self._phases) - 1
        for i, (end, a, b) in enumerate(self._phases):
            if end != start:
                # the phase's steps (`lr_at`'s membership), strictly inside
                # (start, end)
                lo = s0 if i == 0 else max(s0, start.numerator // start.denominator + 1)
                hi = s1 if i == last else min(s1, end.numerator // end.denominator)
                lo = max(lo, start.numerator // start.denominator + 1)
                hi = min(hi, -((-end.numerator) // end.denominator) - 1)
                if lo <= hi:
                    self._fill(vs, es, lo - s0, lo, hi - lo + 1, float(start), float(end),
                               float(a), float(b))
            start = end
        return vs, es

    def _fill(self, vs, es, at, st0, count, start_f, end_f, af, bf):
        den = end_f - start_f
        scale = 16.0 * _F64_EPS
        if self.anneal_strategy == "linear":
            # (st - start) / den, (b - a), the product, + a: under 2^-53 of
            # |a| + |b| + |v| each; bound 2^-48 of that sum
            d = bf - af
            for k in range(count):
                v = d * ((st0 + k - start_f) / den) + af
                vs[at + k] = v
                es[at + k] = scale * (abs(af) + abs(bf) + abs(v))
            return
        got = _cos_pi_run(st0 - start_f, den, count)
        if got is None:
            return
        cs, ce = got
        hd = abs(af - bf)
        for k in range(count):
            # torch's form: end + (start - end) / 2 (cos + 1)
            v = bf + (af - bf) / 2.0 * (cs[k] + 1.0)
            vs[at + k] = v
            es[at + k] = hd * ce[k] + scale * (abs(af) + abs(bf) + abs(v))

    def _phase_fx(self, i, a, b):
        """Per phase fixed-point constants: F2 and the floor/ceil of b and of
        (a - b) / 2 (cos) or b - a (linear) times 2^F2; None when a or b is 0
        (the exact path decides those)."""
        if i not in self._fixed:
            fx = None
            if a != 0 and b != 0:
                m = min(abs(a), abs(b))
                F2 = 160 - (m.numerator.bit_length() - m.denominator.bit_length())
                if F2 > 0:
                    d = (a - b) / 2 if self.anneal_strategy == "cos" else b - a
                    fx = (F2, _fx(b, F2), _fx(a, F2), _fx(d, F2))
            self._fixed[i] = fx
        return self._fixed[i]

    def _lr_at_slow(self, t):
        step = self._t(t)
        if step > self.total_steps:
            raise ValueError(f"OneCycleLR: step {t} is beyond total_steps {self.total_steps} + 1 (torch refuses it too)")
        start = Fraction(0)
        for i, (end, a, b) in enumerate(self._phases):
            if step <= end or i == len(self._phases) - 1:
                if end == start:
                    return _f32_round(b)
                pct = (Fraction(step) - start) / (end - start)
                fx = self._phase_fx(i, a, b)
                if self.anneal_strategy == "linear":
                    if fx is not None and pct >= 0:
                        F2, _, (alo, ahi), (dlo, dhi) = fx
                        pn, pd = pct.numerator, pct.denominator
                        v = _iv_f32(alo + (dlo * pn) // pd, ahi - ((-(dhi * pn)) // pd), -F2)
                        if v is not None:
                            return v
                    return _f32_round((b - a) * pct + a)
                if pct <= 0:
                    return _f32_round(a)
                if pct == 1:
                    return _f32_round(b)
                if pct > 1:
                    # torch's one extra step past the last phase end: cos is
                    # even about pi, cos(pi p) = cos(pi (2 - p))
                    pct = 2 - pct
                if fx is not None and pct < 1 and pct not in _EXACT_COS_POINTS:
                    v = self._cos_fast(fx, pct)
                    if v is not None:
                        return v

                def interval(terms, a=a, b=b, pct=pct):
                    c_lo, c_hi = _cos_pi_interval(pct, terms)
                    x = b + (a - b) / 2 * (c_lo + 1)
                    y = b + (a - b) / 2 * (c_hi + 1)
                    return (x, y) if x <= y else (y, x)
                return _decide_f32(interval)
            start = end

    @staticmethod
    def _cos_fast(fx, p):
        """b + (a - b)/2 (cos(pi p) + 1) for p in (0, 1), enclosed in fixed
        point; None when the ends round apart."""
        F2, (blo, bhi), _, (dlo, dhi) = fx
        flip = p > Fraction(1, 2)
        if flip:
            p = 1 - p
        pn, pd = p.numerator, p.denominator
        x_lo = (_PI_LO_FX * pn) // pd
        x_hi = -((-(_PI_HI_FX * pn)) // pd)
        c_lo, c_hi = _cos_fx(x_lo)            # encloses cos(x_lo)
        c_lo -= x_hi - x_lo                   # |cos'| <= 1 over [x_lo, x_hi]
        if flip:
            c_lo, c_hi = -c_hi, -c_lo
        one = 1 << _F
        c_lo += one
        c_hi += one
        prods = (dlo * c_lo, dlo * c_hi, dhi * c_lo, dhi * c_hi)
        lo = blo + (min(prods) >> _F)
        hi = bhi - ((-max(prods)) >> _F)
        return _iv_f32(lo, hi, -F2)

    def _exact_lr_at(self, t):
        """The previous, all-Fraction evaluation: the reference the fast path
        is checked against."""
        step = self._t(t)
        if step > self.total_steps:
            raise ValueError(f"OneCycleLR: step {t} is beyond total_steps {self.total_steps} + 1 (torch refuses it too)")
        start = Fraction(0)
        for i, (end, a, b) in enumerate(self._phases):
            if step <= end or i == len(self._phases) - 1:
                if end == start:
                    return _f32_round(b)
                pct = (Fraction(step) - start) / (end - start)
                if self.anneal_strategy == "linear":
                    return _f32_round((b - a) * pct + a)
                if pct <= 0:
                    return _f32_round(a)
                if pct == 1:
                    return _f32_round(b)
                if pct > 1:
                    pct = 2 - pct

                def interval(terms, a=a, b=b, pct=pct):
                    c_lo, c_hi = _cos_pi_interval(pct, terms)
                    x = b + (a - b) / 2 * (c_lo + 1)
                    y = b + (a - b) / 2 * (c_hi + 1)
                    return (x, y) if x <= y else (y, x)
                return _decide_f32(interval)
            start = end
