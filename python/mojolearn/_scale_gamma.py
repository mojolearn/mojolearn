# SPDX-License-Identifier: Apache-2.0
"""scikit-learn's `gamma='scale'` for the kernel estimators (SVC, SVR,
RBFSampler), resolved exactly on the host (DEVIATION 870)."""
from fractions import Fraction

from ._portable_math import isfinite


def scale_gamma(values, n_features):
    """scikit-learn's `gamma='scale'`, `1 / (n_features * X.var())` (1.0 when
    the variance is 0), CORRECTLY ROUNDED from the exact variance of `values`
    (the cells of X, each a finite float32 or float64).

    The population variance is formed exactly in integers, (N * S2 - S1^2)
    / N^2 with S1 and S2 the exact sums of x and x * x at a common 2^-1074
    scale (x * x of a float32 is exact in binary64), and the reciprocal is
    rounded ONCE to binary64. No fold order exists to differ, so every host
    and every column reads the same gamma bits; scikit-learn's float32
    `X.var()` differs from it only by that reduction's rounding.
    """
    n = 0
    s1 = 0
    s2 = 0
    for value in values:
        value = float(value)
        if not isfinite(value):
            raise ValueError("gamma='scale' needs finite input: X holds a NaN or an infinity")
        numerator, denominator = value.as_integer_ratio()
        shift = denominator.bit_length() - 1
        s1 += numerator << (1074 - shift)
        s2 += (numerator * numerator) << (2 * (1074 - shift))
        n += 1
    if n == 0:
        raise ValueError("gamma='scale' needs at least one cell of X")
    # var * N^2 = (N * S2 - S1^2) * 2^-2148, both terms at scale 2^-2148
    spread = n * s2 - s1 * s1
    if spread == 0:
        return 1.0
    return float(Fraction(n * n << 2148, int(n_features) * spread))
