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
import math as _cmath
import operator
from os import environ as _environ
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


# The predicates below take a float fast path (lane py-shared): for a Python
# float, `x - x` is +0.0 exactly when x is finite (inf - inf and NaN - NaN are
# NaN), and `x != x` exactly when x is a NaN. These are IEEE comparisons, not
# libm, so they give the bit test's answer on every host at about a fifth of
# its cost. Anything else (an int, a NumPy scalar, a Fraction) keeps the bit
# test, which converts it with float() first, as before.
def isfinite(x):
    if type(x) is float:
        return x - x == 0.0
    return (_bits(x) & 0x7ff0000000000000) != 0x7ff0000000000000


def isinf(x):
    if type(x) is float:
        return x == x and x - x != 0.0
    return (_bits(x) & 0x7fffffffffffffff) == 0x7ff0000000000000


def isnan(x):
    if type(x) is float:
        return x != x
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
    """Exact finite sum followed by one nearest/even binary64 rounding.

    Fast path (lane py-shared; the argument `_expansion_metrics._fsum` made
    first): CPython's compiled `math.fsum` keeps Shewchuk's exact partials and
    rounds the exact sum once to nearest/even, so whenever its result is
    finite every term was finite and it is this function's result bit for
    bit. A zero result is +0.0 here, so a zero goes out as +0.0. A NaN or
    infinite result, an intermediate overflow, or a term `math.fsum` refuses
    (a string float() accepts) goes to the exact sum below, which decides it.
    The reference arm `MOJOLEARN_HOTPATH=python` keeps the exact sum always.
    About 4 ns per term instead of about 500 ns."""
    vals = values if type(values) is list else list(values)
    if _environ.get("MOJOLEARN_HOTPATH", "").strip().lower() != "python":
        try:
            s = _cmath.fsum(vals)
        except (OverflowError, ValueError, TypeError):
            return _fsum_exact(vals)
        if s - s == 0.0:
            return s if s != 0.0 else 0.0
    return _fsum_exact(vals)


def _fsum_exact(values):
    """The exact portable sum (the reference `fsum` keeps)."""
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
