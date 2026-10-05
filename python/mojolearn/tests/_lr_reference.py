# SPDX-License-Identifier: Apache-2.0
"""Reference float32 rounding of an exact rational for the schedule tests
(lane py-runtime round 3: the runtime moved to sequence/lr_exact.mojo; this
is the Python definition it replaced, for the tests to compare with)."""
import math
from fractions import Fraction

_F32_MIN_NORMAL_EXP = -126
_F32_MAX_EXP = 127


def _f32_round(q):
    """The float32 nearest to the exact rational `q`, ties to even, flushed
    to +0.0 below the smallest normal (the identical tier's ftz), as a
    Python float holding exactly that float32 value."""
    q = Fraction(q)
    if q == 0:
        return 0.0
    sign = -1.0 if q < 0 else 1.0
    q = abs(q)
    num, den = q.numerator, q.denominator
    e = num.bit_length() - den.bit_length() - 24

    def scaled(exp):
        if exp >= 0:
            return Fraction(num, den * (1 << exp))
        return Fraction(num * (1 << (-exp)), den)

    while scaled(e) >= (1 << 24):
        e += 1
    while scaled(e) < (1 << 23):
        e -= 1
    sc = scaled(e)
    m = sc.numerator // sc.denominator
    rem = sc - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (m & 1) == 1):
        m += 1
    if m == (1 << 24):
        m = 1 << 23
        e += 1
    if e + 23 < _F32_MIN_NORMAL_EXP:
        return 0.0
    if e + 23 > _F32_MAX_EXP:
        raise OverflowError("mojolearn: schedule value overflows float32")
    return math.ldexp(sign * float(m), e)


