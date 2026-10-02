# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2832's float64 class probability as software binary64
(`checks/soft_f64.mojo`), so the device computes it on EVERY vendor, the
Apple GPU (no float64) included (cpu-gpu-cleanup c-gp-kernel, 2026-10-02:
the GPU binding downloaded the latent mean and variance and ran
`gpc_proba` on host threads).

`gpc_pi_star_sf64` is `gaussian_process/host/gpc_steps.mojo::gpc_pi_star`
statement for statement, and `gpc_erf_sf64` is its `gpc_erf64`: every
operation there is one IEEE binary64 operation in round-to-nearest (the
products through `pinned_mul_f64`, the fused steps through an explicit
`fma`, `sqrt` the correctly rounded square root, the exponential
`identical_exp64` = `portable_exp64`, whose `sf64_exp` twin is in
soft_f64.mojo), and soft_f64's add, sub, mul, div and fma are those
operations bit for bit. The one operation soft_f64 lacks is the square
root; `sf64_sqrt` below is correctly rounded by construction (the
digit-by-digit integer root of the 110-bit radicand, its remainder in 64
bits, then round-to-nearest-even on the guard bit and the sticky
remainder), so it is the IEEE `sqrt` word as well.

The inputs are flushed (`ftz`, the host column's float32 rule) before the
exact widening, on both columns.

`gpc_ovr_combine_row` is the one-vs-rest normalization and argmax of
DEVIATION 2833 (`multiclass.py:523-562`): the k unnormalized class
probabilities summed ascending from +0.0, divided by that sum unless it is
zero, and the first strictly largest unnormalized value's index. k is the
class count (a k-sized loop per row, one row per thread on the device).

GPU-free: soft_f64 and the float32 seams only, so the host column compiles
the same source.
"""

from std.memory import bitcast

from checks.numerics import ftz
from checks.soft_f64 import (
    SF64_FRAC,
    SF64_HIDDEN,
    SF64_INF,
    SF64_NAN,
    SF64_ONE,
    SF64_SIGN,
    SF64_ZERO,
    _clz64,
    sf64_add,
    sf64_div,
    sf64_exp,
    sf64_fma,
    sf64_from_f32,
    sf64_from_int,
    sf64_gt,
    sf64_is_nan,
    sf64_lt,
    sf64_mul,
    sf64_neg,
    sf64_sub,
)

#: DEVIATION 2832: `_gpc.py:30-33` LAMBDAS and COEFS, by their float64 bits
#: (the same words as gpc_steps.mojo's GPC_*_BITS).
comptime _LAMBDA0 = UInt64(0x3FDA3D70A3D70A3D)
comptime _LAMBDA1 = UInt64(0x3FD999999999999A)
comptime _LAMBDA2 = UInt64(0x3FD7AE147AE147AE)
comptime _LAMBDA3 = UInt64(0x3FDC28F5C28F5C29)
comptime _LAMBDA4 = UInt64(0x3FD8F5C28F5C28F6)
comptime _COEF0 = UInt64(0xC09CFB49210A3BC3)
comptime _COEF1 = UInt64(0x40AB79CC416651C4)
comptime _COEF2 = UInt64(0x406BA96415285B3E)
comptime _COEF3 = UInt64(0x406003F190EC4BEE)
comptime _COEF4 = UInt64(0xC09F69FA1685A876)
comptime _HALF_COEF_SUM = UInt64(0x3FDFFFFFFAA1A800)
comptime _PI64 = UInt64(0x400921FB54442D18)
comptime _SQRT_PI64 = UInt64(0x3FFC5BF891B4EF6A)
comptime _TWO_OVER_SQRT_PI64 = UInt64(0x3FF20DD750429B6D)
comptime _HALF64 = UInt64(0x3FE0000000000000)
comptime _TWO64 = UInt64(0x4000000000000000)
comptime _THREE64 = UInt64(0x4008000000000000)
comptime _SIX64 = UInt64(0x4018000000000000)

comptime GPC_ERF_SERIES_TERMS_SF = 80
comptime GPC_ERFC_CF_LEVELS_SF = 40


def sf64_sqrt(a: UInt64) -> UInt64:
    """The correctly rounded binary64 square root (round-to-nearest-even).
    NaN and negative non-zero inputs give the canonical NaN, +-0 and +inf
    return themselves."""
    if sf64_is_nan(a):
        return SF64_NAN
    if (a & ~SF64_SIGN) == 0:
        return a
    if (a >> 63) != 0:
        return SF64_NAN
    if a == SF64_INF:
        return a
    var e = Int((a >> 52) & UInt64(0x7FF))
    var sig: UInt64
    var eu: Int
    if e == 0:
        # subnormal: normalize the fraction so its leading bit is bit 52
        var f = a & SF64_FRAC
        var s = _clz64(f) - 11
        sig = f << UInt64(s)
        eu = 1 - 1023 - s
    else:
        sig = (a & SF64_FRAC) | SF64_HIDDEN
        eu = e - 1023
    # value = sig * 2^(eu - 52), sig in [2^52, 2^53); make eu even
    if (eu & 1) != 0:
        sig = sig << 1
        eu -= 1
    # sqrt(sig * 2^56) in [2^54, 2^55): 55 root bits from 55 radicand pairs.
    # Pair p (bits 2p+1, 2p of sig * 2^56) is sig's bits 2p-55, 2p-56 for
    # p >= 28 and zero below. The remainder stays below 2 * root + 1 < 2^56.
    var root = UInt64(0)
    var rem = UInt64(0)
    for k in range(55):
        var p = 54 - k
        var pair = UInt64(0)
        if p >= 28:
            pair = (sig >> UInt64(2 * p - 56)) & UInt64(3)
        rem = (rem << 2) | pair
        var trial = (root << 2) | UInt64(1)
        if rem >= trial:
            rem = rem - trial
            root = (root << 1) | UInt64(1)
        else:
            root = root << 1
    # root: bit 54 the leading one, 52 fraction bits, a guard and an extra bit
    var m = root >> 2
    var guard = (root >> 1) & UInt64(1)
    var below = (root & UInt64(1)) | (UInt64(1) if rem != 0 else UInt64(0))
    if guard != 0 and (below != 0 or (m & UInt64(1)) != 0):
        m += 1
    var er = (eu >> 1) + 1023
    # an ADDITION, so a significand carry to 2^53 bumps the exponent
    return (UInt64(er - 1) << 52) + m


def gpc_erf_sf64(x: UInt64) -> UInt64:
    """`gpc_steps.mojo::gpc_erf64`, statement for statement."""
    if sf64_is_nan(x):
        return SF64_NAN
    var ax = x
    var negative = False
    if sf64_lt(x, SF64_ZERO):
        ax = sf64_neg(x)
        negative = True
    var r = SF64_ONE
    if sf64_lt(ax, _THREE64):
        var x2 = sf64_mul(ax, ax)
        var neg_x2 = sf64_neg(x2)
        var t = ax
        var s = ax
        for n in range(1, GPC_ERF_SERIES_TERMS_SF):
            var tn = sf64_mul(t, neg_x2)
            t = sf64_div(tn, sf64_from_int(n))
            var q = sf64_div(t, sf64_from_int(2 * n + 1))
            s = sf64_add(s, q)
        r = sf64_mul(s, _TWO_OVER_SQRT_PI64)
    elif sf64_lt(ax, _SIX64):
        var t = ax
        for kk in range(GPC_ERFC_CF_LEVELS_SF):
            var k = GPC_ERFC_CF_LEVELS_SF - kk
            var h = sf64_mul(sf64_from_int(k), _HALF64)
            var q = sf64_div(h, t)
            t = sf64_add(ax, q)
        var e = sf64_exp(sf64_neg(sf64_mul(ax, ax)))
        var den = sf64_mul(_SQRT_PI64, t)
        var c = sf64_div(e, den)
        r = sf64_sub(SF64_ONE, c)
    if negative:
        return sf64_neg(r)
    return r


@always_inline
def _lambda_sf(k: Int) -> UInt64:
    if k == 0:
        return _LAMBDA0
    if k == 1:
        return _LAMBDA1
    if k == 2:
        return _LAMBDA2
    if k == 3:
        return _LAMBDA3
    return _LAMBDA4


@always_inline
def _coef_sf(k: Int) -> UInt64:
    if k == 0:
        return _COEF0
    if k == 1:
        return _COEF1
    if k == 2:
        return _COEF2
    if k == 3:
        return _COEF3
    return _COEF4


def gpc_pi_star_sf64(mean: Float32, variance: Float32) -> UInt64:
    """`gpc_steps.mojo::gpc_pi_star` (`_gpc.py:320-327`), statement for
    statement, as the float64 word."""
    var mu = sf64_from_f32(ftz(mean))
    var va = sf64_from_f32(ftz(variance))
    var acc = SF64_ZERO
    for k in range(5):
        var lam = _lambda_sf(k)
        var gamma = sf64_mul(lam, mu)
        var integral = SF64_ZERO
        # `va > 0.0` is False for a NaN variance (sf64_gt wants non-NaN)
        if not sf64_is_nan(va) and sf64_gt(va, SF64_ZERO):
            var alpha = sf64_div(SF64_ONE, sf64_mul(_TWO64, va))
            var ratio = sf64_div(alpha, sf64_fma(lam, lam, alpha))
            var arg = sf64_mul(gamma, sf64_sqrt(ratio))
            var num = sf64_mul(sf64_sqrt(sf64_div(_PI64, alpha)), gpc_erf_sf64(arg))
            var inner = sf64_mul(sf64_mul(va, _TWO64), _PI64)
            var den = sf64_mul(_TWO64, sf64_sqrt(inner))
            integral = sf64_div(num, den)
        else:
            integral = sf64_mul(_HALF64, gpc_erf_sf64(gamma))
        acc = sf64_fma(_coef_sf(k), integral, acc)
    return sf64_add(acc, _HALF_COEF_SUM)


@always_inline
def gpc_ovr_combine_row(
    cols: MutPointer[UInt64, MutAnyOrigin],
    dst: MutPointer[UInt64, MutAnyOrigin],
    codes: MutPointer[Int32, MutAnyOrigin],
    t: Int,
    n: Int,
    k: Int,
):
    """Row `t` of DEVIATION 2833's one-vs-rest combine. `cols` holds the k
    unnormalized class-1 probabilities class-major (`cols[c * n + t]`),
    `dst` the normalized row row-major (`dst[t * k + c]`), `codes[t]` the
    first index of the strictly largest unnormalized value (NumPy's argmax
    tie rule; a NaN never wins a `>`)."""
    var total = SF64_ZERO
    var best = cols.unsafe_load(t)
    var best_k = 0
    for c in range(k):
        var v = cols.unsafe_load(c * n + t)
        total = sf64_add(total, v)
        if c > 0 and not sf64_is_nan(v) and not sf64_is_nan(best) and sf64_gt(v, best):
            best = v
            best_k = c
    var nonzero = (total & ~SF64_SIGN) != 0
    for c in range(k):
        var v = cols.unsafe_load(c * n + t)
        dst.unsafe_store(t * k + c, sf64_div(v, total) if nonzero else v)
    codes.unsafe_store(t, Int32(best_k))
