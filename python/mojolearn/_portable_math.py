# SPDX-License-Identifier: Apache-2.0
"""Host math with repository-owned arithmetic and no import of platform math.

The native logarithm/exponential functions use the same pinned binary64
polynomials as checks/numerics.mojo. They are portable approximations, not a
claim of correctly rounded general transcendentals. exp follows that existing
primitive's flush-to-zero policy below the smallest normal output. sqrt uses
the correctly rounded CPU instruction on the wheel's supported baselines.
Integer, classification, sign and scaling operations below require no libm.
"""
import array
import ctypes
import decimal
import functools
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


def _load():
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
        vector = getattr(lib, "mojolearn_exp_f64", None)
        if vector is not None:
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


def exp_array(values):
    """`exp` of every value, as an array('d'): the scalar `exp`'s bits (one
    C loop over the same pinned function, `mojolearn_exp_f64`), with its
    OverflowError. For the O(n) links of the Python front door, which paid
    one ctypes call per element (or called the platform exp) before."""
    src = array.array("d", values)
    n = len(src)
    out = array.array("d", bytes(8 * n))
    vector = getattr(_load(), "mojolearn_exp_f64", None) if n else None
    if vector is None:
        for i in range(n):
            out[i] = exp(src[i])
        return out
    if vector(src.buffer_info()[0], n, out.buffer_info()[0]):
        raise OverflowError("math range error")
    return out


def nsum(values):
    """CPython 3.12+'s builtin `sum` over floats, spelled out: `0 + x0`
    (int start, so -0.0 becomes +0.0), then Neumaier's compensated sum in
    iteration order, the compensation added once at the end when it is
    nonzero and finite. Python 3.10 and 3.11 `sum` is the plain left-to-right
    fold, so the builtin's bits depend on the interpreter; this twin gives
    3.12+'s bits on every supported Python. An empty input gives 0.0."""
    it = iter(values)
    for first in it:
        total = 0.0 + float(first)
        break
    else:
        return 0.0
    c = 0.0
    for x in it:
        x = float(x)
        t = total + x
        if abs(total) >= abs(x):
            c += (total - t) + x
        else:
            c += (x - t) + total
        total = t
    if c and isfinite(c):
        total += c
    return total


def _cut(value, shift, bits, up):
    """value * 2**shift cut to at most `bits` significant bits (+1 when
    rounding up), toward zero or away from it: (value', shift')."""
    extra = value.bit_length() - bits
    if extra <= 0:
        return value, shift
    kept = value >> extra
    if up and (kept << extra) != value:
        kept += 1
    return kept, shift + extra


def _pow_bound(m, n, bits, up):
    """A bound on m**n (m, n positive integers) as (v, s) with v * 2**s below
    (up=False) or above (up=True) it: square-and-multiply, every product cut
    to `bits` bits in the bound's direction."""
    acc, acc_s = 1, 0
    base, base_s = m, 0
    while n:
        if n & 1:
            acc, acc_s = _cut(acc * base, acc_s + base_s, bits, up)
        n >>= 1
        if n:
            base, base_s = _cut(base * base, 2 * base_s, bits, up)
    return acc, acc_s


@functools.lru_cache(maxsize=256)
def powi(x, n):
    """`x ** n` for a float x and an integer n >= 0, CORRECTLY ROUNDED
    (nearest/even), without the platform pow that `x ** n` calls. The exact
    power lies between two truncated square-and-multiply bounds; when both
    round to the same binary64 that is the answer, else the precision
    doubles (and ends exact). Zero, infinite and NaN bases are exact IEEE
    special cases. Overflow raises OverflowError, as `x ** n` does."""
    x = float(x)
    n = operator.index(n)
    if n < 0:
        raise ValueError("powi: n must be >= 0")
    if n == 0:
        return 1.0
    if x == 0.0 or not isfinite(x):
        return x ** n  # exact special values, no rounding
    num, den = x.as_integer_ratio()
    negative = num < 0 and n & 1
    num = abs(num)
    scale = -(den.bit_length() - 1) * n
    bits = 128
    while True:
        if bits >= num.bit_length() * n + 2:
            result = _scaled_integer(num ** n, scale)
            break
        lo, lo_s = _pow_bound(num, n, bits, False)
        hi, hi_s = _pow_bound(num, n, bits, True)
        a = _scaled_integer(lo, lo_s + scale)
        try:
            b = _scaled_integer(hi, hi_s + scale)
        except OverflowError:
            b = inf
        if a == b:
            result = a
            break
        bits *= 2
    return -result if negative else result


_POW_CONTEXT = decimal.Context(prec=50, rounding=decimal.ROUND_HALF_EVEN, Emax=999999, Emin=-999999)


def powr(x, y):
    """`x ** y` for a positive finite float x and a finite float y, without
    the platform pow: Python's decimal module (libmpdec, integer arithmetic,
    the same result on every host) at 50 significant digits, then one
    correctly rounded conversion to binary64. Correctly rounded unless the
    exact power lies within about 1e-49 (relative) of a binary64 halfway
    point; either way the same bits on every host."""
    x, y = float(x), float(y)
    if not (x > 0.0 and isfinite(x) and isfinite(y)):
        raise ValueError("powr: x must be positive and finite, y finite")
    value = _POW_CONTEXT.power(decimal.Decimal(x), decimal.Decimal(y))
    result = float(value)
    if isinf(result):
        raise OverflowError("math range error")
    return result


# The standard normal distribution: statistics.NormalDist's formulas on the
# pinned log / exp above, so a draw is the same on every host. NormalDist.cdf
# calls the platform erf, and NormalDist.inv_cdf the platform log (or the C
# `_statistics` accelerator, which a compiler may contract into FMAs).

_ERX = 8.45062911510467529297e-01
_PP = (1.28379167095512558561e-01, -3.25042107247001499370e-01, -2.84817495755985104766e-02,
       -5.77027029648944159157e-03, -2.37630166566501626084e-05)
_QQ = (3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03,
       1.32494738004321644526e-04, -3.96022827877536812320e-06)
_PA = (-2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01,
       3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02,
       -2.16637559486879084300e-03)
_QA = (1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02,
       1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.19844998467991074170e-02)
_RA = (-9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e+01,
       -6.23753324503260060396e+01, -1.62396669462573470355e+02, -1.84605092906711035994e+02,
       -8.12874355063065934246e+01, -9.81432934416914548592e+00)
_SA = (1.96512716674392571292e+01, 1.37657754143519042600e+02, 4.34565877475229228821e+02,
       6.45387271733267880336e+02, 4.29008140027567833386e+02, 1.08635005541779435134e+02,
       6.57024977031928170135e+00, -6.04244152148580987438e-02)
_RB = (-9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e+01,
       -1.60636384855821916062e+02, -6.37566443368389627722e+02, -1.02509513161107724954e+03,
       -4.83519191608651397019e+02)
_SB = (3.03380607434824582924e+01, 3.25792512996573918826e+02, 1.53672958608443695994e+03,
       3.19985821950859553908e+03, 2.55305040643316442583e+03, 4.74528541206955367215e+02,
       -2.24409524465858183362e+01)


def _horner(c, z):
    """c[0] + z * (c[1] + z * (... + z * c[-1])), innermost first."""
    r = c[-1]
    for v in reversed(c[:-1]):
        r = v + z * r
    return r


def erfc(x):
    """The complementary error function: fdlibm's `s_erf.c` erfc (Sun
    Microsystems, 1993) in binary64: its intervals (chosen on the high
    word, including its 0x4006DB6D tail split), its rational approximations
    in its Horner order, its split exp, on the pinned exp above. No
    platform erfc, so the same bits on every host."""
    x = float(x)
    if isnan(x):
        return x
    hx = _bits(x) >> 32
    negative = hx >> 31
    ix = hx & 0x7fffffff
    if ix >= 0x7ff00000:                     # erfc(+inf) = 0, erfc(-inf) = 2
        return 2.0 if negative else 0.0
    if ix < 0x3feb0000:                      # |x| < 0.84375
        if ix < 0x3c700000:                  # |x| < 2**-56
            return 1.0 - x
        z = x * x
        y = _horner(_PP, z) / _horner((1.0,) + _QQ, z)
        if negative or ix < 0x3fd00000:      # x < 1/4
            return 1.0 - (x + x * y)
        r = x * y
        r += (x - 0.5)
        return 0.5 - r
    if ix < 0x3ff40000:                      # 0.84375 <= |x| < 1.25
        s = abs(x) - 1.0
        p = _horner(_PA, s)
        q = _horner((1.0,) + _QA, s)
        if not negative:
            return (1.0 - _ERX) - p / q
        return 1.0 + (_ERX + p / q)
    if ix >= 0x403c0000:                     # |x| >= 28
        return 2.0 if negative else 0.0
    ax = abs(x)
    s = 1.0 / (ax * ax)
    if ix < 0x4006DB6D:                      # |x| < 1 / 0.35
        r, q = _horner(_RA, s), _horner((1.0,) + _SA, s)
    else:
        if negative and ix >= 0x40180000:    # x <= -6: two - tiny rounds to two
            return 2.0
        r, q = _horner(_RB, s), _horner((1.0,) + _SB, s)
    z = _float(_bits(ax) & 0xffffffff00000000)
    t = exp(-z * z - 0.5625) * exp((z - ax) * (z + ax) + r / q)
    return 2.0 - t / ax if negative else t / ax


_SQRT2 = 1.4142135623730951  # sqrt(2.0), correctly rounded


def normal_cdf(x):
    """statistics.NormalDist().cdf(x) as Python 3.14 spells it,
    0.5 * erfc((mu - x) / (sigma * sqrt 2)) with mu 0.0 and sigma 1.0, on
    the pinned erfc (NormalDist calls the platform erfc; Python 3.10 and 3.11
    spell it 0.5 * (1 + erf(...)), other bits)."""
    return 0.5 * erfc((0.0 - float(x)) / (1.0 * _SQRT2))


def normal_inv_cdf(p):
    """statistics.NormalDist().inv_cdf(p) for 0 < p < 1: Wichura's AS241
    (PPND16), exactly `statistics._normal_dist_inv_cdf`'s operations with mu
    0.0 and sigma 1.0, on the pinned log and the correctly rounded sqrt."""
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("normal_inv_cdf: p must be in (0.0, 1.0)")
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((2.5090809287301226727e+3 * r +
                     3.3430575583588128105e+4) * r +
                     6.7265770927008700853e+4) * r +
                     4.5921953931549871457e+4) * r +
                     1.3731693765509461125e+4) * r +
                     1.9715909503065514427e+3) * r +
                     1.3314166789178437745e+2) * r +
                     3.3871328727963666080e+0) * q
        den = (((((((5.2264952788528545610e+3 * r +
                     2.8729085735721942674e+4) * r +
                     3.9307895800092710610e+4) * r +
                     2.1213794301586595867e+4) * r +
                     5.3941960214247511077e+3) * r +
                     6.8718700749205790830e+2) * r +
                     4.2313330701600911252e+1) * r +
                     1.0)
        return 0.0 + ((num / den) * 1.0)
    r = p if q <= 0.0 else 1.0 - p
    r = sqrt(-log(r))
    if r <= 5.0:
        r = r - 1.6
        num = (((((((7.74545014278341407640e-4 * r +
                     2.27238449892691845833e-2) * r +
                     2.41780725177450611770e-1) * r +
                     1.27045825245236838258e+0) * r +
                     3.64784832476320460504e+0) * r +
                     5.76949722146069140550e+0) * r +
                     4.63033784615654529590e+0) * r +
                     1.42343711074968357734e+0)
        den = (((((((1.05075007164441684324e-9 * r +
                     5.47593808499534494600e-4) * r +
                     1.51986665636164571966e-2) * r +
                     1.48103976427480074590e-1) * r +
                     6.89767334985100004550e-1) * r +
                     1.67638483018380384940e+0) * r +
                     2.05319162663775882187e+0) * r +
                     1.0)
    else:
        r = r - 5.0
        num = (((((((2.01033439929228813265e-7 * r +
                     2.71155556874348757815e-5) * r +
                     1.24266094738807843860e-3) * r +
                     2.65321895265761230930e-2) * r +
                     2.96560571828504891230e-1) * r +
                     1.78482653991729133580e+0) * r +
                     5.46378491116411436990e+0) * r +
                     6.65790464350110377720e+0)
        den = (((((((2.04426310338993978564e-15 * r +
                     1.42151175831644588870e-7) * r +
                     1.84631831751005468180e-5) * r +
                     7.86869131145613259100e-4) * r +
                     1.48753612908506148525e-2) * r +
                     1.36929880922735805310e-1) * r +
                     5.99832206555887937690e-1) * r +
                     1.0)
    x = num / den
    if q < 0.0:
        x = -x
    return 0.0 + (x * 1.0)
