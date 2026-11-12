# SPDX-License-Identifier: Apache-2.0
"""Host math with repository-owned transcendental arithmetic.

The native logarithm/exponential functions use the same pinned binary64
polynomials as checks/numerics.mojo. They are portable approximations, not a
claim of correctly rounded general transcendentals. exp follows that existing
primitive's flush-to-zero policy below the smallest normal output. sqrt uses
the correctly rounded CPU instruction on the wheel's supported baselines.
Integer, classification, sign and scaling operations below require no libm.
The finite compensated-sum fast path uses CPython's arithmetic implementation,
with the exact integer sum retained for exceptional cases and as its oracle.
"""
import array
import ctypes
import decimal
import functools
import operator
from pathlib import Path
import struct
import sys

inf = float("inf")
nan = float("nan")
pi = float.fromhex("0x1.921fb54442d18p+1")
_lib = None


def _bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _float(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _load():
    global _lib
    if _lib is None:
        root = Path(__file__).resolve().parent
        path = (root / ".dylibs/libMojolearnMath.dylib" if sys.platform == "darwin"
                else root / ".libs/libMojolearnMath.so")
        lib = ctypes.CDLL(str(path))
        for operation in ("sqrt", "log", "log2", "log10", "exp"):  # glue: binds five library symbols
            fn = getattr(lib, "mojolearn_" + operation)
            fn.argtypes = [ctypes.c_double]
            fn.restype = ctypes.c_double
        vector = lib.mojolearn_exp_f64
        vector.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
        vector.restype = ctypes.c_long
        _lib = lib
    return _lib


def _native(name, x):
    return getattr(_load(), "mojolearn_" + name)(x)


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
    return functools.reduce(operator.mul, values, start)  # glue: product of a shape's dimensions (every caller passes a shape tuple)


def fsum(values):
    """Exact finite sum followed by one nearest/even binary64 rounding, with
    math.fsum's NaN and infinity rules (a zero sum is +0.0). Lane py-runtime
    round 3: the core helper `fsum_f64` (bindings/array_helpers_pack.mojo,
    Shewchuk's partials in Mojo); the `math.fsum` fast path and the Python
    big-integer fold are gone. Callers pass k-sized lists (estimator
    weights, class priors)."""
    from ._buffer import _native
    src = array.array("d", values)
    out = array.array("d", [0.0])
    if int(_native("fsum_f64")(src.buffer_info()[0], len(src), out.buffer_info()[0])):
        if out[0]:  # glue: the binding's out-of-range marker (+-inf); -inf + inf leaves 0.0
            raise OverflowError("math range error")
        raise ValueError("-inf + inf in fsum")
    return out[0]


# No result cache: an lru_cache keyed on (x, n) by equality served +0.0 for
# -0.0 (equal, same hash) and vice versa, so the bits depended on call order.


_POW_CONTEXT = decimal.Context(prec=50, rounding=decimal.ROUND_HALF_EVEN, Emax=999999, Emin=-999999)


# The scalar helpers no product path called (isclose, comb, frexp, ldexp,
# powi, powr, nsum, exp_array and the standard normal erfc / normal_cdf /
# normal_inv_cdf) are deleted (lane pyout-runtime-boundary, 2026-11-11). Their
# native definitions are the ones the product runs: bindings/normal_dist_helpers_pack.mojo
# (erfc64, normal_cdf64, normal_inv_cdf64), bindings/jl_native_pack.mojo (the exact
# cube), bindings/array_helpers_pack.mojo (nsum_f64) and the binding's own powers.


_SQRT2 = 1.4142135623730951  # sqrt(2.0), correctly rounded


