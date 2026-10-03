# SPDX-License-Identifier: Apache-2.0
"""scikit-learn's `gamma='scale'` for the kernel estimators (SVC, SVR,
RBFSampler), resolved exactly on the host (DEVIATION 870)."""
from fractions import Fraction

from ._portable_math import isfinite

#: lane/apple-fast-py2mojo-linear: `scale_gamma_limbs` word layout
#: (svm/impl/scale_gamma_limbs.mojo): S1 at scale 2^-149 in 9 base-2^32
#: limbs, S2 at scale 2^-298 in 18, then the count of non-finite cells
_SG_L1, _SG_L2 = 9, 18
_SG_SLOTS = _SG_L1 + _SG_L2 + 1
_PY2MOJO_SCALE_GAMMA = 1


def py2mojo_linear_flags(binding):
    """The bound binary's lane/apple-fast-py2mojo-linear switches (0 for a
    binary without the entry or built with -D MOJOLEARN_PY2MOJO_linear_OFF,
    which is the old Python path)."""
    fn = getattr(binding, "py2mojo_linear_flags", None)
    if fn is None:
        return 0
    return int(fn())


def scale_gamma_x(binding, x, n_features):
    """`scale_gamma` of the float32 Array `x`: the exact sums from the
    binding's `scale_gamma_limbs` (the device grid, or the host column on a
    CPU-only install), the one rounding here. The same rational as
    `scale_gamma`, so the same bits."""
    if x.dtype != "<f4":
        raise TypeError("gamma='scale' reads the float32 X the kernel estimators fit on")
    if not py2mojo_linear_flags(binding) & _PY2MOJO_SCALE_GAMMA:
        # lane pyglue-sweep (Oct 3): no Python pass over the cells any more
        raise RuntimeError(
            "mojolearn: this binding has no scale_gamma_limbs (an older binary or a "
            "-D MOJOLEARN_PY2MOJO_linear_OFF build); rebuild it"
        )
    from ._buffer import addr, addr_ro, empty

    n = int(x.size)
    if n == 0:
        raise ValueError("gamma='scale' needs at least one cell of X")
    words = empty((_SG_SLOTS,), "<i8")
    binding.scale_gamma_limbs(addr_ro(x, name="X"), n, addr(words, name="scale_gamma limbs"))
    w = words.tolist()
    if w[_SG_L1 + _SG_L2] != 0:
        raise ValueError("gamma='scale' needs finite input: X holds a NaN or an infinity")
    s1 = 0
    for i in range(_SG_L1):  # glue: assembles nine fixed limb words
        s1 += int(w[i]) << (32 * i)
    s2 = 0
    for i in range(_SG_L2):  # glue: assembles eighteen fixed limb words
        s2 += int(w[_SG_L1 + i]) << (32 * i)
    # var * N^2 = (N * S2 - S1^2) * 2^-298
    spread = n * s2 - s1 * s1
    if spread == 0:
        return 1.0
    return float(Fraction(n * n << 298, int(n_features) * spread))


def scale_gamma(values, n_features):
    """THE REFERENCE DEFINITION (tests only since lane pyglue-sweep, Oct 3:
    no runtime path calls it; `scale_gamma_x` is the binding's). scikit-learn's `gamma='scale'`, `1 / (n_features * X.var())` (1.0 when
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
