# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SEAM 5540: the LR schedulers' ONE rounding (python/mojolearn/_x_sequence_sched.py).

    tools/with_identical_mode.sh python -u sequence/checks/sched_check.py

StepLR, ExponentialLR and OneCycleLR (linear and cos) are closed forms of the
step, evaluated exactly and rounded ONCE to float32 (ties to even), so their
bits are the same on every host. This driver restates each schedule from
torch.optim.lr_scheduler's formulas with its OWN arithmetic (Decimal at 70
digits, pi by Machin, cos by its Taylor series; the float32 neighbour picked
by distance, not by the module's rounding helper) and requires:
  1. the fixture SEPARATES the pinned spelling from the alternative (the
     float64 closed form rounded to float32, what torch computes and what a
     float schedule would carry), else VACUOUS;
  2. every lr_at bit pattern == the oracle's.
The sabotage arm (sequence/checks/sabotage/seam_5540_sched_exact_round.patch)
evaluates StepLR and ExponentialLR in float64 first; the driver must FAIL.
The CPU and device columns share these host-side values (the optimizers take
lr as a host scalar), so there is one column."""
import struct
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from mojolearn._x_sequence_sched import ExponentialLR, OneCycleLR, StepLR  # noqa: E402

getcontext().prec = 70


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32_of_bits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def nearest_f32(d):
    """The float32 nearest to the Decimal (or Fraction) d, ties to even,
    found by comparing the distances of the neighbours of a first guess.
    Raises when d is within 1e-50 (relative) of a midpoint, where a 70-digit
    value could not decide."""
    q = Fraction(d) if not isinstance(d, Fraction) else d
    if q == 0:
        return 0.0
    guess = struct.unpack("<f", struct.pack("<f", float(q)))[0]
    b = f32_bits(guess)
    cands = [f32_of_bits(x) for x in (b - 1, b, b + 1) if 0 <= x < 0xFFFFFFFF]
    cands.sort(key=lambda c: (abs(Fraction(c) - q), f32_bits(c) & 1))
    best, second = cands[0], cands[1]
    gap = abs(abs(Fraction(best) - q) - abs(Fraction(second) - q))
    if gap < abs(q) * Fraction(1, 10 ** 50):
        raise SystemExit(f"UNDECIDABLE fixture: {float(q)!r} sits on a float32 midpoint")
    return best


def dec(q):
    return Decimal(q.numerator) / Decimal(q.denominator)


def pi_dec():
    def atan_inv(n):
        x = Decimal(1) / n
        x2 = x * x
        s, term, k = Decimal(0), x, 0
        while term != 0:
            s += term / (2 * k + 1) if k % 2 == 0 else -term / (2 * k + 1)
            term *= x2
            k += 1
            if k > 400:
                break
        return s
    return 16 * atan_inv(5) - 4 * atan_inv(239)


PI = pi_dec()


def cos_dec(x):
    s, term, k = Decimal(1), Decimal(1), 0
    x2 = x * x
    while True:
        term = -term * x2 / ((2 * k + 1) * (2 * k + 2))
        k += 1
        if abs(term) < Decimal(10) ** -68:
            break
        s += term
    return s


# ---------------------------------------------------------------- oracles
def o_step(base, step_size, gamma, t, alt):
    e = t - 1
    if alt:
        return struct.unpack("<f", struct.pack("<f", base * gamma ** (e // step_size)))[0]
    return nearest_f32(Fraction(base) * Fraction(gamma) ** (e // step_size))


def o_exp(base, gamma, t, alt):
    if alt:
        return struct.unpack("<f", struct.pack("<f", base * gamma ** (t - 1)))[0]
    return nearest_f32(Fraction(base) * Fraction(gamma) ** (t - 1))


def o_onecycle(max_lr, total, pct_start, strat, div, fdiv, t, alt):
    """torch OneCycleLR (two phases): the phase ends float(pct_start *
    total) - 1 and total - 1; annealing from start to end over pct."""
    step = t - 1
    mx = Fraction(max_lr)
    init = mx / Fraction(div)
    low = init / Fraction(fdiv)
    phases = [(Fraction(float(pct_start * total) - 1), init, mx), (Fraction(total - 1), mx, low)]
    start = Fraction(0)
    for i, (end, a, b) in enumerate(phases):
        if step <= end or i == len(phases) - 1:
            pct = (Fraction(step) - start) / (end - start)
            if alt:
                pf, af, bf = float(pct), float(a), float(b)
                if strat == "linear":
                    v = (bf - af) * pf + af
                else:
                    import math
                    v = bf + (af - bf) / 2.0 * (math.cos(math.pi * pf) + 1)
                return struct.unpack("<f", struct.pack("<f", v))[0]
            if strat == "linear":
                return nearest_f32((b - a) * pct + a)
            if pct > 1:
                pct = 2 - pct
            c = cos_dec(PI * dec(pct))
            return nearest_f32(dec(b) + dec(a - b) / 2 * (c + 1))
        start = end
    raise AssertionError


def main():
    cases = []   # (name, schedule, oracle(t, alt), steps)
    for base, ss, g in ((0.1, 3, 0.1), (0.037, 2, 0.93), (1e-3, 5, 0.77)):
        cases.append((f"StepLR({base}, {ss}, {g})", StepLR(base, ss, g),
                      lambda t, alt, base=base, ss=ss, g=g: o_step(base, ss, g, t, alt), 60))
    for base, g in ((0.1, 0.9), (0.01, 0.97), (3e-4, 0.999)):
        cases.append((f"ExponentialLR({base}, {g})", ExponentialLR(base, g),
                      lambda t, alt, base=base, g=g: o_exp(base, g, t, alt), 80))
    for strat in ("linear", "cos"):
        for mx, total, pct in ((0.1, 97, 0.3), (0.003, 50, 0.25)):
            s = OneCycleLR(mx, total, pct_start=pct, anneal_strategy=strat)
            cases.append((f"OneCycleLR({mx}, {total}, {pct}, {strat})", s,
                          lambda t, alt, mx=mx, total=total, pct=pct, strat=strat:
                          o_onecycle(mx, total, pct, strat, 25.0, 1e4, t, alt), total))
    separated = 0
    wrong = 0
    for name, sched, oracle, steps in cases:
        sep = bad = 0
        for t in range(1, steps + 1):
            want = oracle(t, False)
            if f32_bits(want) != f32_bits(oracle(t, True)):
                sep += 1
            if f32_bits(sched.lr_at(t)) != f32_bits(want):
                bad += 1
                if bad <= 3:
                    print(f"  {name} step {t}: lr_at {sched.lr_at(t)!r} oracle {want!r}")
        print(f"  {name}: {steps} steps, fixture separates {sep}, differ from the oracle {bad}")
        separated += sep
        wrong += bad
    if separated == 0:
        print("VACUOUS 5540_sched_exact_round: the fixture does not separate the pinned spelling from the alternative")
        return 1
    if wrong:
        print(f"FAIL 5540_sched_exact_round: {wrong} learning rates differ from the oracle")
        return 1
    print(f"PASS 5540_sched_exact_round ({separated} separating steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
