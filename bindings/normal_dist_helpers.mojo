# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The standard normal distribution in Mojo (lane py-runtime round 3):
`_portable_math.erfc` (fdlibm's s_erf.c erfc), `normal_cdf`,
`normal_inv_cdf` (Wichura's AS241) and IterativeImputer's
truncated-normal posterior draw (`_truncnorm_host`), which ran per missing
entry in Python.

SAME BITS AS THE PYTHON THEY REPLACE, by construction: the same binary64
operations in the same order, every product that meets an addition spelled
`pinned_mul_f64` (Python never fuses; Mojo's default fp-mode would), the
pinned exp and log (`checks/numerics.mojo` portable_exp64 / portable_log64,
which packaging/portable_math/portable_math.c, the Python side's library,
translates) and the correctly rounded sqrt. The draw stream is the same
splitmix64 words, consumed only by the entries that draw.

The draw runs on the host column on every install: the predictions come
from the user's estimator (IterativeImputer's user-estimator route, a
CPU-only input step), and it is float64 transcendental work Apple's GPU
cannot do."""
from std.math import sqrt
from std.memory import bitcast
from std.python import Python, PythonObject

from checks.numerics import pinned_mul_f64, portable_exp64, portable_log64

comptime _ERX: Float64 = 8.45062911510467529297e-01
comptime _SQRT2: Float64 = 1.4142135623730951


@always_inline
def _m(a: Float64, b: Float64) -> Float64:
    return pinned_mul_f64(a, b)


def _h5(c0: Float64, c1: Float64, c2: Float64, c3: Float64, c4: Float64, z: Float64) -> Float64:
    """c0 + z (c1 + z (c2 + z (c3 + z c4))), innermost first (`_horner`)."""
    var r = c4
    r = c3 + _m(z, r)
    r = c2 + _m(z, r)
    r = c1 + _m(z, r)
    return c0 + _m(z, r)


def _h6(c0: Float64, c1: Float64, c2: Float64, c3: Float64, c4: Float64, c5: Float64, z: Float64) -> Float64:
    var r = c5
    r = c4 + _m(z, r)
    r = c3 + _m(z, r)
    r = c2 + _m(z, r)
    r = c1 + _m(z, r)
    return c0 + _m(z, r)


def _h7(c0: Float64, c1: Float64, c2: Float64, c3: Float64, c4: Float64, c5: Float64, c6: Float64,
        z: Float64) -> Float64:
    var r = c6
    r = c5 + _m(z, r)
    r = c4 + _m(z, r)
    r = c3 + _m(z, r)
    r = c2 + _m(z, r)
    r = c1 + _m(z, r)
    return c0 + _m(z, r)


def _h8(c0: Float64, c1: Float64, c2: Float64, c3: Float64, c4: Float64, c5: Float64, c6: Float64,
        c7: Float64, z: Float64) -> Float64:
    var r = c7
    r = c6 + _m(z, r)
    r = c5 + _m(z, r)
    r = c4 + _m(z, r)
    r = c3 + _m(z, r)
    r = c2 + _m(z, r)
    r = c1 + _m(z, r)
    return c0 + _m(z, r)


def _h9(c0: Float64, c1: Float64, c2: Float64, c3: Float64, c4: Float64, c5: Float64, c6: Float64,
        c7: Float64, c8: Float64, z: Float64) -> Float64:
    var r = c8
    r = c7 + _m(z, r)
    r = c6 + _m(z, r)
    r = c5 + _m(z, r)
    r = c4 + _m(z, r)
    r = c3 + _m(z, r)
    r = c2 + _m(z, r)
    r = c1 + _m(z, r)
    return c0 + _m(z, r)


def erfc64(x: Float64) -> Float64:
    """`_portable_math.erfc`: fdlibm's erfc in binary64 on the pinned exp."""
    if x != x:
        return x
    var hx = Int((bitcast[DType.uint64](x) >> 32) & 0xFFFFFFFF)
    var negative = (hx >> 31) != 0
    var ix = hx & 0x7FFFFFFF
    if ix >= 0x7FF00000:
        return 2.0 if negative else 0.0
    if ix < 0x3FEB0000:
        if ix < 0x3C700000:
            return 1.0 - x
        var z = _m(x, x)
        var y = _h5(1.28379167095512558561e-01, -3.25042107247001499370e-01, -2.84817495755985104766e-02,
                    -5.77027029648944159157e-03, -2.37630166566501626084e-05, z) / _h6(
            1.0, 3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03,
            1.32494738004321644526e-04, -3.96022827877536812320e-06, z)
        if negative or ix < 0x3FD00000:
            return 1.0 - (x + _m(x, y))
        var r = _m(x, y)
        r += (x - 0.5)
        return 0.5 - r
    if ix < 0x3FF40000:
        var s = abs(x) - 1.0
        var p = _h7(-2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01,
                    3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02,
                    -2.16637559486879084300e-03, s)
        var q = _h7(1.0, 1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02,
                    1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.19844998467991074170e-02, s)
        if not negative:
            return (1.0 - _ERX) - p / q
        return 1.0 + (_ERX + p / q)
    if ix >= 0x403C0000:
        return 2.0 if negative else 0.0
    var ax = abs(x)
    var s = 1.0 / _m(ax, ax)
    var r: Float64
    var q: Float64
    if ix < 0x4006DB6D:
        r = _h8(-9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e+01,
                -6.23753324503260060396e+01, -1.62396669462573470355e+02, -1.84605092906711035994e+02,
                -8.12874355063065934246e+01, -9.81432934416914548592e+00, s)
        q = _h9(1.0, 1.96512716674392571292e+01, 1.37657754143519042600e+02, 4.34565877475229228821e+02,
                6.45387271733267880336e+02, 4.29008140027567833386e+02, 1.08635005541779435134e+02,
                6.57024977031928170135e+00, -6.04244152148580987438e-02, s)
    else:
        if negative and ix >= 0x40180000:
            return 2.0
        r = _h7(-9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e+01,
                -1.60636384855821916062e+02, -6.37566443368389627722e+02, -1.02509513161107724954e+03,
                -4.83519191608651397019e+02, s)
        q = _h8(1.0, 3.03380607434824582924e+01, 3.25792512996573918826e+02, 1.53672958608443695994e+03,
                3.19985821950859553908e+03, 2.55305040643316442583e+03, 4.74528541206955367215e+02,
                -2.24409524465858183362e+01, s)
    var z = bitcast[DType.float64](bitcast[DType.uint64](ax) & UInt64(0xFFFFFFFF00000000))
    var t = _m(portable_exp64(_m(-z, z) - 0.5625), portable_exp64(_m(z - ax, z + ax) + r / q))
    return 2.0 - t / ax if negative else t / ax


def normal_cdf64(x: Float64) -> Float64:
    """`_portable_math.normal_cdf`: 0.5 erfc((0 - x) / sqrt 2)."""
    return _m(0.5, erfc64((0.0 - x) / _m(1.0, _SQRT2)))


def normal_inv_cdf64(p: Float64) -> Float64:
    """`_portable_math.normal_inv_cdf` (AS241) for 0 < p < 1."""
    var q = p - 0.5
    if abs(q) <= 0.425:
        var r = 0.180625 - _m(q, q)
        var num = _m(_h8(3.3871328727963666080e+0, 1.3314166789178437745e+2, 1.9715909503065514427e+3,
                         1.3731693765509461125e+4, 4.5921953931549871457e+4, 6.7265770927008700853e+4,
                         3.3430575583588128105e+4, 2.5090809287301226727e+3, r), q)
        var den = _h8(1.0, 4.2313330701600911252e+1, 6.8718700749205790830e+2, 5.3941960214247511077e+3,
                      2.1213794301586595867e+4, 3.9307895800092710610e+4, 2.8729085735721942674e+4,
                      5.2264952788528545610e+3, r)
        return 0.0 + _m(num / den, 1.0)
    var r = p if q <= 0.0 else 1.0 - p
    r = sqrt(-portable_log64(r))
    var num: Float64
    var den: Float64
    if r <= 5.0:
        r = r - 1.6
        num = _h8(1.42343711074968357734e+0, 4.63033784615654529590e+0, 5.76949722146069140550e+0,
                  3.64784832476320460504e+0, 1.27045825245236838258e+0, 2.41780725177450611770e-1,
                  2.27238449892691845833e-2, 7.74545014278341407640e-4, r)
        den = _h8(1.0, 2.05319162663775882187e+0, 1.67638483018380384940e+0, 6.89767334985100004550e-1,
                  1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4,
                  1.05075007164441684324e-9, r)
    else:
        r = r - 5.0
        num = _h8(6.65790464350110377720e+0, 5.46378491116411436990e+0, 1.78482653991729133580e+0,
                  2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
                  2.71155556874348757815e-5, 2.01033439929228813265e-7, r)
        den = _h8(1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
                  7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7,
                  2.04426310338993978564e-15, r)
    var x = num / den
    if q < 0.0:
        x = -x
    return 0.0 + _m(x, 1.0)


@always_inline
def _splitmix64(mut s: UInt64) -> UInt64:
    s += UInt64(0x9E3779B97F4A7C15)
    var z = s
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@always_inline
def _pmax(a: Float64, b: Float64) -> Float64:
    """Python's max(a, b): b only when b > a."""
    return b if b > a else a


@always_inline
def _pmin(a: Float64, b: Float64) -> Float64:
    """Python's min(a, b): b only when b < a."""
    return b if b < a else a


def truncnorm_draws_binding(
    mus_addr: PythonObject, sig_addr: PythonObject, m: PythonObject, bounds: PythonObject,
    state: PythonObject, out_addr: PythonObject,
) raises -> PythonObject:
    """IterativeImputer's `_truncnorm_host` for m entries: mu beyond a bound
    -> the bound, sigma <= 0 (or NaN) -> mu, else the inversion of the
    truncated normal at a 53-bit splitmix64 uniform (only drawing entries
    advance the stream). bounds = [lo, hi] (+-inf allowed); state = the
    stream's 64-bit word as [high 32 bits, low 32 bits]. Writes m float64
    values; returns the new state as [high, low]."""
    var n = Int(py=m)
    var lo = Float64(py=bounds[0])
    var hi = Float64(py=bounds[1])
    var s = (UInt64(Int(py=state[0])) << 32) | UInt64(Int(py=state[1]))
    var mp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=mus_addr))
    var sp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=sig_addr))
    var op = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=out_addr))
    var neg_inf = bitcast[DType.float64](UInt64(0xFFF0000000000000))
    var pos_inf = bitcast[DType.float64](UInt64(0x7FF0000000000000))
    for i in range(n):  # small-loop(n: missing entries of one feature): the user-estimator route's per-entry draw (CPU-only input step)
        var mu = mp[i]
        var sigma = sp[i]
        var v: Float64
        if mu < lo:
            v = lo
        elif mu > hi:
            v = hi
        elif not (sigma > 0.0):
            v = mu
        else:
            var pa = 0.0 if lo == neg_inf else normal_cdf64((lo - mu) / sigma)
            var pb = 1.0 if hi == pos_inf else normal_cdf64((hi - mu) / sigma)
            var w = _splitmix64(s)
            var u = _m(Float64(w >> 11) + 0.5, 1.1102230246251565e-16)
            var pu = pa + _m(u, pb - pa)
            if pu <= 0.0:
                v = lo
            elif pu >= 1.0:
                v = hi
            else:
                v = _pmin(_pmax(mu + _m(sigma, normal_inv_cdf64(pu)), lo), hi)
        op[i] = v
    var out = Python.list()
    out.append(PythonObject(Int(s >> 32)))
    out.append(PythonObject(Int(s & UInt64(0xFFFFFFFF))))
    return out
