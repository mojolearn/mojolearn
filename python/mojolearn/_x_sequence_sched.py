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
import os as _os
from fractions import Fraction

import array as _array

from ._training_impl import _F64_EPS, _LrTable, _lr_buffers, _lr_native, _lr_status

_F32_MIN_NORMAL_EXP = -126
_F32_MAX_EXP = 127
#: steps of one StepLR / ExponentialLR block filled natively per lr_at miss
_TAB_BLOCK = 8192


def _q(x, name):
    x = float(x)
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return Fraction(x)


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
        """lr_at(1) .. lr_at(n) as a list, from the schedule's native blocks
        (`lr_values`: every schedule here has one)."""
        return self.lr_values(n).tolist()


class _PowSched(_Sched):
    """base_lr gamma^E(t), E(t) the schedule's exponent: the float32 nearest
    the exact value, in Mojo (`lr_pow_values`, sequence/lr_exact.mojo: a
    P-bit enclosure widened until both ends round alike, then the exact
    power; lane py-runtime round 3, it was Python integer enclosures and a
    Fraction power), served from blocks of _TAB_BLOCK steps."""

    _step = 1

    def _init_pow(self):
        self._tab, self._tab0 = [], 1

    def _block(self, t0, n):
        """(float32 values of steps t0 .. t0 + n - 1, index of the first
        overflowing step or None)."""
        _q(self.base_lr, "base_lr")
        _q(self.gamma, "gamma")
        out = _array.array("f", bytes(4 * n))
        st, at = _lr_native("lr_pow_values")([self.base_lr, self.gamma], [t0 - 1, n, self._step],
                                             out.buffer_info()[0])
        st, at = int(st), int(at)
        if st:
            del out[at:]
            return out, at
        return out, None

    def lr_at(self, t):
        t = self._t(t) + 1                 # validates the ONE-BASED step
        i = t - self._tab0
        if not 0 <= i < len(self._tab):
            out, _ = self._block(t, _TAB_BLOCK)
            self._tab, self._tab0, i = out.tolist(), t, 0
            if not self._tab:
                _lr_status(1)              # this step's value overflows float32
        return self._tab[i]

    def lr_values(self, n):
        """lr_at(1) .. lr_at(n) as an array('f'), natively."""
        out, bad = self._block(1, int(n))
        if bad is not None:
            _lr_status(1)
        return out


class StepLR(_PowSched):
    """lr = base_lr gamma^(epoch // step_size) (torch's closed form)."""

    def __init__(self, base_lr, step_size, gamma=0.1):
        if int(step_size) < 1:
            raise ValueError("StepLR: step_size must be >= 1")
        self.base_lr, self.step_size, self.gamma = float(base_lr), int(step_size), float(gamma)
        self._step = self.step_size
        self._init_pow()


class ExponentialLR(_PowSched):
    """lr = base_lr gamma^epoch."""

    def __init__(self, base_lr, gamma):
        self.base_lr, self.gamma = float(base_lr), float(gamma)
        self._init_pow()


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
        # the phase ends as torch builds them in float64, for the exact route
        self._e1f = float(self.pct_start * self.total_steps) - 1
        self._e2f = float(2 * self.pct_start * self.total_steps) - 2
        # lane gap-train-utils: the block table of `_training_impl._LrTable`
        # over steps 1 .. total_steps + 1 (no constant tail; beyond it the
        # exact route refuses, as torch does)
        self._table_setup(self.total_steps + 2, None)

    def _fast_values(self, t0, t1):
        """binary64 values and error bounds of steps t0 .. t1 - 1 (zeros where
        the exact route answers: a phase's two ends, its extra step, an empty
        phase, a step angle over 1/2)."""
        n = t1 - t0
        vs, es = _lr_buffers(n)
        s0 = t0 - 1                 # torch's step of index 0
        s1 = s0 + n - 1
        start = Fraction(0)
        last = len(self._phases) - 1
        for i, (end, a, b) in enumerate(self._phases):  # glue: the at most three OneCycle phases
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
        """One phase's binary64 values and bounds into vs / es from `at`, in
        Mojo (`lr_onecycle_fill`, bindings/lr_table_helpers.mojo; lane
        py-runtime round 2): torch's linear form d ((st - start) / den) + a
        or its cosine form end + (start - end) / 2 (cos + 1), with the same
        error bounds as before."""
        _lr_native("lr_onecycle_fill")(
            [1 if self.anneal_strategy == "linear" else 0, int(at), int(st0), int(count),
             float(start_f), float(end_f), float(af), float(bf)],
            vs.buffer_info()[0], es.buffer_info()[0])

    def _exact_params(self):
        return ([1 if self.anneal_strategy == "linear" else 0, 1 if self.three_phase else 0, self.total_steps],
                [self.max_lr, self.div_factor, self.final_div_factor, self._e1f, self._e2f],
                "lr_onecycle_exact_block")

    def _lr_at_slow(self, t):
        """The exact route of one step in Mojo big rationals
        (`lr_onecycle_exact`, sequence/lr_exact.mojo; lane py-runtime round
        3, it was Fraction arithmetic and a fixed-point cosine here)."""
        step = self._t(t)
        if step > self.total_steps:
            raise ValueError(f"OneCycleLR: step {t} is beyond total_steps {self.total_steps} + 1 (torch refuses it too)")
        ip, fp, _ = self._exact_params()
        st, v = _lr_native("lr_onecycle_exact")(ip + [step], fp)
        _lr_status(st)
        return float(v)

