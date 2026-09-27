# SPDX-License-Identifier: Apache-2.0
"""Host math with repository-owned arithmetic and no import of platform math.

The native logarithm/exponential functions use the same pinned binary64
polynomials as checks/numerics.mojo. They are portable approximations, not a
claim of correctly rounded general transcendentals. exp follows that existing
primitive's flush-to-zero policy below the smallest normal output. sqrt uses
the correctly rounded CPU instruction on the wheel's supported baselines.
Integer, classification, sign and scaling operations below require no libm.
"""
import ctypes
import operator
from pathlib import Path
import struct
import sys

inf = float("inf")
nan = float("nan")
_lib = None


def _bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _float(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _native(name, x):
    global _lib
    if _lib is None:
        root = Path(__file__).resolve().parent
        path = (root / ".dylibs/libMojolearnMath.dylib" if sys.platform == "darwin"
                else root / ".libs/libMojolearnMath.so")
        lib = ctypes.CDLL(str(path))
        for operation in ("sqrt", "log", "log2", "log10", "exp"):
            fn = getattr(lib, "mojolearn_" + operation)
            fn.argtypes = [ctypes.c_double]
            fn.restype = ctypes.c_double
        _lib = lib
    return getattr(_lib, "mojolearn_" + name)(x)


def isfinite(x):
    return (_bits(x) & 0x7ff0000000000000) != 0x7ff0000000000000


def isinf(x):
    return (_bits(x) & 0x7fffffffffffffff) == 0x7ff0000000000000


def isnan(x):
    return (_bits(x) & 0x7fffffffffffffff) > 0x7ff0000000000000


def copysign(x, y):
    return _float((_bits(x) & 0x7fffffffffffffff) | (_bits(y) & 0x8000000000000000))


def sqrt(x):
    x = float(x)
    if x < 0.0:
        raise ValueError("math domain error")
    return _native("sqrt", x)


def log(x, base=None):
    x = float(x)
    if x <= 0.0:
        raise ValueError("math domain error")
    result = _native("log", x)
    return result if base is None else result / log(base)


def log2(x):
    x = float(x)
    if x <= 0.0:
        raise ValueError("math domain error")
    return _native("log2", x)


def exp(x):
    x = float(x)
    result = _native("exp", x)
    if isinf(result) and isfinite(x):
        raise OverflowError("math range error")
    return result


def floor(x):
    if isinstance(x, int):
        return int(x)
    if not isinstance(x, float) and hasattr(type(x), "__floor__"):
        return x.__floor__()
    x = float(x)
    value = int(x)
    return value - (x < value)


def ceil(x):
    if isinstance(x, int):
        return int(x)
    if not isinstance(x, float) and hasattr(type(x), "__ceil__"):
        return x.__ceil__()
    x = float(x)
    value = int(x)
    return value + (x > value)


def prod(values, *, start=1):
    for value in values:
        start *= value
    return start


def _scaled_integer(value, exponent):
    """Round value * 2**exponent once to binary64, nearest/even."""
    sign = 0x8000000000000000 if value < 0 else 0
    value = abs(value)
    if not value:
        return _float(sign)
    top = value.bit_length() - 1 + exponent
    if top > 1023:
        raise OverflowError("math range error")
    if top < -1075:
        return _float(sign)
    shift = max(value.bit_length() - 53, -1074 - exponent, 0)
    if shift:
        rounded, remainder = divmod(value, 1 << shift)
        halfway = 1 << (shift - 1)
        value = rounded + (remainder > halfway or (remainder == halfway and rounded & 1))
        exponent += shift
    if not value:
        return _float(sign)
    top = value.bit_length() - 1 + exponent
    if top > 1023:
        raise OverflowError("math range error")
    if top < -1022:
        return _float(sign | (value << (exponent + 1074)))
    width = value.bit_length()
    mantissa = value << (53 - width) if width <= 53 else value >> (width - 53)
    return _float(sign | ((top + 1023) << 52) | (mantissa & 0xfffffffffffff))


def ldexp(x, exponent):
    x = float(x)
    exponent = operator.index(exponent)
    if x == 0.0 or not isfinite(x):
        return x
    numerator, denominator = x.as_integer_ratio()
    return _scaled_integer(numerator, exponent - (denominator.bit_length() - 1))


def fsum(values):
    """Exact finite sum followed by one nearest/even binary64 rounding."""
    total = 0
    positive_inf = negative_inf = False
    nan_value = None
    for value in values:
        value = float(value)
        if isnan(value):
            nan_value = value
        elif isinf(value):
            positive_inf |= value > 0
            negative_inf |= value < 0
        else:
            numerator, denominator = value.as_integer_ratio()
            total += numerator << (1074 - (denominator.bit_length() - 1))
    if positive_inf and negative_inf:
        raise ValueError("-inf + inf in fsum")
    if nan_value is not None:
        return nan_value
    if positive_inf or negative_inf:
        return inf if positive_inf else -inf
    return _scaled_integer(total, -1074)


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
    from fractions import Fraction
    return float(Fraction(n * n << 2148, int(n_features) * spread))
