# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""VAR(p) by ordinary least squares, one regression per equation sharing one
design, as statsmodels `statsmodels/tsa/vector_ar/var_model.py`
(`VAR._estimate_var`, `util.get_var_endog`): the design row of time t is
[1 (trend 'c'), y_{t-1}, ..., y_{t-p}], the lag-1 block first.

The solve is ours (statsmodels calls LAPACK's lstsq in float64): every design
column is scaled by a power of two chosen from its largest magnitude (exact:
it moves only the exponent), the normal equations Z'Z B = Z'Y are formed by
the lane's fixed-order GEMM, and one thread factors Z'Z by Cholesky and solves
every equation's column; the coefficients are scaled back by the same powers
of two. A non-positive pivot (a rank-deficient design) is reported through a
status word, never as a NaN.
"""
from std.memory import bitcast
from checks.soft_f64 import sf64_from_f32, sf64_sub, sf64_to_f32

from sequence.ops import FP, Args, add, fma3, gemm_dot, ld, mul, st, sub
from checks.numerics import ftz, identical_div, identical_sqrt


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def op_var_design(t: Int, a: Args):
    """Row r = t // m, column c = t % m of Z [R, m] and, for c < K, of the
    target Ys [R, K]. p0 y [n, K], p1 Z, p2 Ys; i0 K, i1 p, i2 k_trend, i3 m."""
    var K = a.i0
    var p = a.i1
    var kt = a.i2
    var m = a.i3
    var r = t // m
    var c = t - r * m
    var v: Float32
    if c < kt:
        v = Float32(1.0)
    else:
        var j = c - kt
        var lag = j // K + 1
        var var_ = j - (lag - 1) * K
        v = ld(a.p0, (p + r - lag) * K + var_)
    st(a.p1, t, v)
    if c < K:
        st(a.p2, r * K + c, ld(a.p0, (p + r) * K + c))


def pow2_scale(maxabs: Float32) -> Float32:
    """2^-e with e the unbiased exponent of maxabs (1 for 0 or a subnormal):
    maxabs * scale lies in [1, 2)."""
    var bits = bitcast[DType.uint32](maxabs)
    var e = Int((bits >> 23) & 0xFF)
    if e == 0:
        return Float32(1.0)
    var se = 254 - e   # biased exponent of 2^-(e-127)
    if se < 1:
        return Float32(1.0)
    return bitcast[DType.float32](UInt32(se) << 23)


def op_colscale(t: Int, a: Args):
    """Column t of p0 [R, m] scaled by a power of two into [1, 2) at its
    largest magnitude; p1[t] = the scale. i0 R, i1 m."""
    var R = a.i0
    var m = a.i1
    var mx = Float32(0.0)
    for r in range(R):
        var v = abs(ld(a.p0, r * m + t))
        if v > mx:
            mx = v
    var s = pow2_scale(mx)
    for r in range(R):
        st(a.p0, r * m + t, mul(ld(a.p0, r * m + t), s))
    st(a.p1, t, s)


def op_cholsolve(t: Int, a: Args):
    """One thread: G [m, m] (in place, lower Cholesky factor) and B [m, K]
    (in place, the solution of G X = B). p0 G, p1 B, p2 status [1]
    (0 ok, 1 + column of the first non-positive pivot); i0 m, i1 K."""
    var m = a.i0
    var K = a.i1
    for j in range(m):
        var d = ld(a.p0, j * m + j)
        for k in range(j):
            var l = ld(a.p0, j * m + k)
            d = sub(d, mul(l, l))
        if not (d > Float32(0.0)):
            st(a.p2, 0, Float32(1 + j))
            return
        var ljj = ftz(identical_sqrt(d))
        st(a.p0, j * m + j, ljj)
        for i in range(j + 1, m):
            var v = ld(a.p0, i * m + j)
            for k in range(j):
                v = sub(v, mul(ld(a.p0, i * m + k), ld(a.p0, j * m + k)))
            st(a.p0, i * m + j, div(v, ljj))
    for c in range(K):
        # forward: L z = b
        for i in range(m):
            var v = ld(a.p1, i * K + c)
            for k in range(i):
                v = sub(v, mul(ld(a.p0, i * m + k), ld(a.p1, k * K + c)))
            st(a.p1, i * K + c, div(v, ld(a.p0, i * m + i)))
        # backward: L' x = z
        var i = m - 1
        while i >= 0:
            var v = ld(a.p1, i * K + c)
            for k in range(i + 1, m):
                v = sub(v, mul(ld(a.p0, k * m + i), ld(a.p1, k * K + c)))
            st(a.p1, i * K + c, div(v, ld(a.p0, i * m + i)))
            i -= 1
    st(a.p2, 0, Float32(0.0))


def op_rowscale(t: Int, a: Args):
    """p0[r, :] *= p1[r] for element t of p0 [m, K]; i1 K."""
    var r = t // a.i1
    st(a.p0, t, mul(ld(a.p0, t), ld(a.p1, r)))


def op_var_forecast(t: Int, a: Args):
    """One thread: p2 [h, K] = the VAR recursion from the last p rows of
    p0 [p, K] under params p1 [m, K] (statsmodels' layout: trend row first,
    then lag 1's K rows, ...). i0 K, i1 p, i2 k_trend, i3 h. The sum for
    y_t[j] runs trend, then lag 1 variables 0..K-1, then lag 2, ..."""
    var K = a.i0
    var p = a.i1
    var kt = a.i2
    var h = a.i3
    for s in range(h):
        for j in range(K):
            var acc = Float32(0.0)
            if kt == 1:
                acc = ld(a.p1, j)
            for lag in range(1, p + 1):
                for v in range(K):
                    var idx = s - lag
                    var x: Float32
                    if idx >= 0:
                        x = ld(a.p2, idx * K + v)
                    else:
                        x = ld(a.p0, (p + idx) * K + v)
                    acc = fma3(ld(a.p1, (kt + (lag - 1) * K + v) * K + j), x, acc)
            st(a.p2, s * K + j, acc)


def op_sub(t: Int, a: Args):
    """p2[t] = p0[t] - p1[t]."""
    st(a.p2, t, sub(ld(a.p0, t), ld(a.p1, t)))


def op_var_fitted(t: Int, a: Args):
    """Public fittedvalues[t] = original_y[t] - returned_residual[t].

    Keep the former NumPy binary32 subtraction graph, not the earlier GEMM
    prediction (subtracting a rounded residual can yield another last bit).
    Raw loads/stores preserve subnormal input/results. Existing integer
    soft-f64 arithmetic plus RNE narrowing implements binary32 subtraction
    on GPU and host without a target FTZ choice or a host fallback. NaNs
    follow the existing canonical-NaN soft-f64 contract.
    p0 original y after its lag rows, p1 residual, p2 output.
    """
    var y = sf64_from_f32(a.p0.unsafe_load(t))
    var residual = sf64_from_f32(a.p1.unsafe_load(t))
    a.p2.unsafe_store(t, sf64_to_f32(sf64_sub(y, residual)))


def op_scale(t: Int, a: Args):
    """p0[t] *= f0."""
    st(a.p0, t, mul(ld(a.p0, t), a.f0))


# ---- lane/apple-fast-tsa2 (-D MOJOLEARN_TSA2_VAR; FAST + Apple only).
# The two epilogues of the fit fused into the products that feed them, so
# the fit queues two launches fewer and keeps no F buffer. Each cell's chain
# is the unfused pair's: `gemm_dot` from zero, then the same `sub` or `mul`.


def op_var_resid(t: Int, a: Args):
    """Cell t of Rs [R, K] = Ys[r, c] - sum_k Z[r, k] Bm[k, c] (k ascending,
    the GEMM chain, then one subtraction: `op_gemm` into F followed by
    `op_sub`). p0 Z [R, m], p1 Bm [m, K], p2 Ys [R, K], p3 Rs [R, K];
    i0 K, i1 m."""
    var K = a.i0
    var m = a.i1
    var r = t // K
    var c = t - r * K
    var f = gemm_dot(a.p0, r * m, 1, a.p1, c, K, m, Float32(0.0))
    st(a.p3, t, sub(ld(a.p2, t), ftz(f)))


def op_var_sigma(t: Int, a: Args):
    """Cell t of S [K, K] = f0 * sum_r Rs[r, i] Rs[r, j] (r ascending, the
    GEMM chain, then one product: `op_gemm` followed by `op_scale`).
    p0 Rs [R, K], p1 S [K, K]; i0 K, i1 R; f0 the 1 / (R - m) factor."""
    var K = a.i0
    var R = a.i1
    var i = t // K
    var j = t - i * K
    var s = gemm_dot(a.p0, i, K, a.p0, j, K, R, Float32(0.0))
    st(a.p1, t, mul(ftz(s), a.f0))
