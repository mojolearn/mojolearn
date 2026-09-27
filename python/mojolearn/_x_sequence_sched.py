# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Learning-rate schedules of `torch.optim.lr_scheduler`: StepLR,
ExponentialLR and OneCycleLR, as closed forms of the step index, evaluated
in exact rational arithmetic and rounded once to float32 (the cosine through
`_training_impl`'s rational enclosure), so a schedule's bits are the same on
every host. `lr_at(t)` takes the ONE-BASED optimizer step, the convention of
mojolearn's schedules: `lr_at(1)` is torch's lr before its first
`scheduler.step()` (last_epoch 0), `lr_at(t)` its lr after t - 1 steps.

Pass one as `lr_schedule=` to the lane's optimizers (RMSprop, Adagrad, Lion,
Adamax, NAdam, LAMB) and recurrent estimators (LSTM/GRU/RNN), which set
`lr = schedule.lr_at(t)` before step t.

Not carried (sequence/NOT_IMPLEMENTED.tsv): OneCycleLR's momentum cycling
(cycle_momentum, base_momentum, max_momentum); per-group learning rates;
`last_epoch` resumption (evaluate `lr_at` at the step you resume from)."""
from fractions import Fraction

from ._training_impl import _cos_pi_interval, _decide_f32, _f32_round


def _q(x, name):
    x = float(x)
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return Fraction(x)


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


class StepLR(_Sched):
    """lr = base_lr gamma^(epoch // step_size) (torch's closed form)."""

    def __init__(self, base_lr, step_size, gamma=0.1):
        if int(step_size) < 1:
            raise ValueError("StepLR: step_size must be >= 1")
        self.base_lr, self.step_size, self.gamma = float(base_lr), int(step_size), float(gamma)

    def lr_at(self, t):
        e = self._t(t)
        return _f32_round(_q(self.base_lr, "base_lr") * _q(self.gamma, "gamma") ** (e // self.step_size))


class ExponentialLR(_Sched):
    """lr = base_lr gamma^epoch."""

    def __init__(self, base_lr, gamma):
        self.base_lr, self.gamma = float(base_lr), float(gamma)

    def lr_at(self, t):
        return _f32_round(_q(self.base_lr, "base_lr") * _q(self.gamma, "gamma") ** self._t(t))


class OneCycleLR(_Sched):
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

    def lr_at(self, t):
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
                    # torch's one extra step past the last phase end: cos is
                    # even about pi, cos(pi p) = cos(pi (2 - p))
                    pct = 2 - pct

                def interval(terms, a=a, b=b, pct=pct):
                    c_lo, c_hi = _cos_pi_interval(pct, terms)
                    x = b + (a - b) / 2 * (c_lo + 1)
                    y = b + (a - b) / 2 * (c_hi + 1)
                    return (x, y) if x <= y else (y, x)
                return _decide_f32(interval)
            start = end
