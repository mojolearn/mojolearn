# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the gp CPU host binding and into the host side of the GPU binding; product, not only a check.
"""The host arithmetic of Gaussian process CLASSIFICATION, shared by the
GPU path (`gaussian_process/classifier.mojo`) and the CPU host oracle
(`gaussian_process/host/gpc_oracle.mojo`). GPU-free: it imports only the
`checks/numerics.mojo` seams, so both binaries compile the same source for
every step that is not a kernel matrix, a factorization, a solve or a
matrix-vector product.

THE REFERENCE. scikit-learn 1.9.0 `sklearn/gaussian_process/_gpc.py`, the
Laplace approximation of Rasmussen and Williams (GPML) Algorithms 3.1 and
3.2 with the logistic likelihood:

    _gpc.py:443-499  _posterior_mode (Algorithm 3.1), the Newton loop
      :467        pi = expit(f)
      :468        W = pi * (1 - pi)
      :470-472    W_sr = sqrt(W); B = I + W_sr K W_sr
      :473        L = cholesky(B, lower=True)
      :475        b = W f + (y - pi)
      :477        a = b - W_sr cho_solve(L, W_sr K b)
      :479        f = K a
      :483-487    lml = -0.5 a.f - sum log1p(exp(-(2y - 1) f)) - sum log diag L
      :491-493    stop when lml - previous < 1e-10, else keep lml
    _gpc.py:434-441  latent_mean_and_variance (Algorithm 3.2 lines 4-6)
    _gpc.py:320-329  predict_proba: the logistic sigmoid integrated against
                     the latent Gaussian through five error functions
                     (LAMBDAS and COEFS, _gpc.py:30-33)

DEVIATION 2830, THE STOP RULE. The reference's iteration count is data
dependent, and DEVIATION 1766 named that as the reason classification was
not implemented. The count is deterministic across devices when the value
the test reads is itself identical, and here it is: `lml` is a float32
built from identical seams, a pinned Cholesky and the pinned gemm, so every
column reads the same bits and takes the same branch. The rule is the
reference's rule at float32: stop when `ftz(lml - previous)` is below the
float32 nearest 1e-10 (bits `0x2EDBE6FF`), a strict `<`, so a NaN
difference does not stop the loop (the reference's `nan < 1e-10` is False
too); at most `max_iter_predict` iterations; `previous` starts at float32
negative infinity. The reported likelihood is the reference's: the last
value kept before the stop (so the one BEFORE the iteration that failed to
improve), while `pi`, `W_sr` and `L` come from the last iteration run.
`n_iter_` reports how many ran, so the count is on the identity card.

DEVIATION 2831, THE PINNED FLOAT32 ORDERS. The reference is float64 NumPy;
this lane is float32 with every order written down:
  - `B_ij = ftz((W_sr_i * W_sr_j) * K_ij)`, the diagonal `ftz(1 + that)`.
    The reference spells `(W_sr_i * K_ij) * W_sr_j` (`_gpc.py:471-472`);
    multiplying the two weights first makes `B` symmetric BY BITS whenever
    `K` is, which the Cholesky door checks.
  - `W_sr K b` is `W_sr_i * (K b)_i`, with `K b` the pinned gemm at `OP_TN`
    (`K` is symmetric by bits, DEVIATION 1758's argument).
  - `log1p(exp(z))` is `identical_softplus`, which returns `z` above 20. In
    float32 that is the same number: `log1p(exp(z)) - z < 2.1e-9` there,
    below half the float32 spacing at 20.
  - `sum log diag L` is `0.5 * logdet`, the factor's own log-determinant
    (DEVIATION 1757), never a second fold.
  - `lml = ftz(ftz(t1 - t2) - t3)`, the reference's left-to-right order;
    `a.f`, the softplus sum and the latent variance fold are serial
    ascending chains through `identical_mul_add`.

DEVIATION 2832, THE PROBABILITY IS FLOAT64 ON THE HOST. `COEFS` reach 3517 in
magnitude and cancel to a probability, so a float32 expansion would lose
about 1e-4 absolute. The five-term expansion therefore runs in float64 from
the float32 latent mean and variance (exact widenings), in the reference's
spelling (`sqrt(pi / alpha) * erf(gamma * sqrt(alpha / (alpha + lambda^2)))
/ (2 sqrt(var * 2 pi))`, the five terms summed ascending, then
`0.5 * sum(COEFS)` added), with every constant pinned by its float64 bits.
There is no float64 erf in `checks/numerics.mojo` (its `identical_erf` is
float32 Cephes, DEVIATION 822), so `gpc_erf64` below is new: the Maclaurin
series (79 terms) below |x| = 3, the Laplace continued fraction for erfc
(40 levels, evaluated bottom-up) from 3 to 6, and +-1 from 6 on, every
product through `fma(a, b, -0.0)` and the exponential through
`identical_exp64`. Its largest error against CPython's `math.erf` over
[-7, 7] on a 200001-point grid is 8.9e-14. A latent variance that is not
positive (zero in the limit of a training point with an exact fit) takes
the variance-to-zero limit of the same expression, `0.5 * erf(gamma)`, where
the reference divides zero by zero.
"""

from std.math import fma, sqrt
from std.memory import bitcast

from max.algorithm import sync_parallelize

from checks.numerics import (
    ftz,
    identical_exp64,
    identical_mul,
    identical_mul_add,
    identical_sigmoid,
    identical_softplus,
    identical_sqrt,
)
from core.host_predict_threads import (
    host_predict_chunk,
    host_predict_task_count,
)

#: DEVIATION 2830: float32(1e-10), the reference's tolerance at this width.
comptime GPC_LML_TOL_BITS: UInt32 = 0x2EDBE6FF
comptime GPC_NEG_INF32_BITS: UInt32 = 0xFF800000

#: DEVIATION 2832: `_gpc.py:30-33` LAMBDAS and COEFS, by their float64 bits.
comptime GPC_LAMBDA0_BITS: UInt64 = 0x3FDA3D70A3D70A3D
comptime GPC_LAMBDA1_BITS: UInt64 = 0x3FD999999999999A
comptime GPC_LAMBDA2_BITS: UInt64 = 0x3FD7AE147AE147AE
comptime GPC_LAMBDA3_BITS: UInt64 = 0x3FDC28F5C28F5C29
comptime GPC_LAMBDA4_BITS: UInt64 = 0x3FD8F5C28F5C28F6
comptime GPC_COEF0_BITS: UInt64 = 0xC09CFB49210A3BC3
comptime GPC_COEF1_BITS: UInt64 = 0x40AB79CC416651C4
comptime GPC_COEF2_BITS: UInt64 = 0x406BA96415285B3E
comptime GPC_COEF3_BITS: UInt64 = 0x406003F190EC4BEE
comptime GPC_COEF4_BITS: UInt64 = 0xC09F69FA1685A876
#: `0.5 * COEFS.sum()` as NumPy computes it (ascending), 0.49999999500016656.
comptime GPC_HALF_COEF_SUM_BITS: UInt64 = 0x3FDFFFFFFAA1A800
comptime GPC_PI64_BITS: UInt64 = 0x400921FB54442D18
comptime GPC_SQRT_PI64_BITS: UInt64 = 0x3FFC5BF891B4EF6A
comptime GPC_TWO_OVER_SQRT_PI64_BITS: UInt64 = 0x3FF20DD750429B6D

comptime GPC_ERF_SERIES_TERMS = 80
comptime GPC_ERFC_CF_LEVELS = 40


@fieldwise_init
struct GPCBinaryFit(Movable):
    """One binary Laplace fit: sklearn's `L_`, `pi_`, `W_sr_` and
    `log_marginal_likelihood_value_`, plus the iteration count (DEVIATION
    2830) and the Cholesky panel width that ran."""

    var l: List[Float32]
    var pi: List[Float32]
    var wsr: List[Float32]
    var lml: Float32
    var n_iter: Int
    var nb: Int


@fieldwise_init
struct GPCLatent(Movable):
    """`latent_mean_and_variance` at the query rows. `variance` is empty
    when only the mean was asked for (`predict`)."""

    var mean: List[Float32]
    var variance: List[Float32]


@fieldwise_init
struct GPCWeights(Movable):
    var pi: List[Float32]
    var w: List[Float32]
    var wsr: List[Float32]


def gpc_lml_tol() -> Float32:
    return bitcast[DType.float32](GPC_LML_TOL_BITS)


def gpc_neg_inf32() -> Float32:
    return bitcast[DType.float32](GPC_NEG_INF32_BITS)


# ===========================================================================
# VALIDATION (host, before any launch)
# ===========================================================================


def gpc_validate_labels(y: List[Float32], n_train: Int) raises:
    """The binary targets: `n_train` values, each exactly 0 or 1, both
    present. The Python surface encodes labels (and one-vs-rest columns)
    before this; a single class is refused by name there with the
    reference's sentence (`_gpc.py:198-203`), and again here."""
    if len(y) != n_train:
        raise Error(
            "gpc_fit_host: y holds "
            + String(len(y))
            + " values for "
            + String(n_train)
            + " training rows"
        )
    var zeros = 0
    var ones = 0
    for i in range(n_train):
        var v = y[i]
        if v == Float32(0.0):
            zeros += 1
        elif v == Float32(1.0):
            ones += 1
        else:
            raise Error(
                "gpc_fit_host: the binary target at index "
                + String(i)
                + " is not 0 or 1; the surface encodes the classes before"
                " the binding is reached, so this boundary disagrees with it"
            )
    if zeros == 0 or ones == 0:
        raise Error(
            "gpc_fit_host: a binary Laplace fit requires 2 classes; got 1"
            " class. scikit-learn refuses the same data (_gpc.py:207-212)"
        )


def gpc_validate_max_iter(max_iter_predict: Int) raises:
    if max_iter_predict < 1:
        raise Error(
            "gpc_fit_host: max_iter_predict must be at least 1, got "
            + String(max_iter_predict)
            + " (scikit-learn's Interval(Integral, 1, None))"
        )


# ===========================================================================
# THE NEWTON STEP'S HOST ARITHMETIC (_gpc.py:467-493)
# ===========================================================================


def gpc_weights(f: List[Float32]) -> GPCWeights:
    """`pi = expit(f)`, `W = pi (1 - pi)`, `W_sr = sqrt(W)`."""
    var n = len(f)
    var pi = List[Float32](capacity=n)
    var w = List[Float32](capacity=n)
    var wsr = List[Float32](capacity=n)
    for i in range(n):
        var p = ftz(identical_sigmoid(ftz(f[i])))
        var one_minus = ftz(Float32(1.0) - p)
        var wi = ftz(identical_mul(p, one_minus))
        pi.append(p)
        w.append(wi)
        wsr.append(ftz(identical_sqrt(wi)))
    return GPCWeights(pi^, w^, wsr^)


def gpc_b_matrix(k: List[Float32], wsr: List[Float32], n: Int) -> List[Float32]:
    """`B = I + W_sr K W_sr`, row-major, the weights multiplied first
    (DEVIATION 2831)."""
    var b = List[Float32](capacity=n * n)
    for i in range(n):
        for j in range(n):
            var ww = ftz(identical_mul(ftz(wsr[i]), ftz(wsr[j])))
            var cell = ftz(identical_mul(ww, ftz(k[i * n + j])))
            if i == j:
                cell = ftz(Float32(1.0) + cell)
            b.append(cell)
    return b^


def gpc_newton_rhs(
    w: List[Float32], f: List[Float32], y: List[Float32], pi: List[Float32]
) -> List[Float32]:
    """`b = W f + (y - pi)`."""
    var n = len(f)
    var out = List[Float32](capacity=n)
    for i in range(n):
        var wf = ftz(identical_mul(ftz(w[i]), ftz(f[i])))
        var r = ftz(ftz(y[i]) - ftz(pi[i]))
        out.append(ftz(wf + r))
    return out^


def gpc_scale(wsr: List[Float32], v: List[Float32]) -> List[Float32]:
    """`W_sr_i * v_i`."""
    var n = len(v)
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(ftz(identical_mul(ftz(wsr[i]), ftz(v[i]))))
    return out^


def gpc_a_vector(
    b: List[Float32], wsr: List[Float32], x: List[Float32]
) -> List[Float32]:
    """`a = b - W_sr x`, where `x = cho_solve(L, W_sr K b)`."""
    var n = len(b)
    var out = List[Float32](capacity=n)
    for i in range(n):
        var s = ftz(identical_mul(ftz(wsr[i]), ftz(x[i])))
        out.append(ftz(ftz(b[i]) - s))
    return out^


def gpc_lml(
    a: List[Float32], f: List[Float32], y: List[Float32], logdet_b: Float32
) -> Float32:
    """`-0.5 a.f - sum log1p(exp(-(2y - 1) f)) - sum log diag L`
    (`_gpc.py:483-487`) in DEVIATION 2831's order."""
    var n = len(a)
    var dot = Float32(0.0)
    for i in range(n):
        dot = ftz(identical_mul_add(ftz(a[i]), ftz(f[i]), dot))
    var t1 = ftz(identical_mul(Float32(-0.5), dot))
    var t2 = Float32(0.0)
    for i in range(n):
        # -(2y - 1) f is -f for y = 1 and +f for y = 0, an exact negation.
        var z = ftz(f[i])
        if y[i] == Float32(1.0):
            z = -z
        t2 = ftz(t2 + ftz(identical_softplus(z)))
    var t3 = ftz(identical_mul(Float32(0.5), ftz(logdet_b)))
    return ftz(ftz(t1 - t2) - t3)


def gpc_stop(lml: Float32, previous: Float32) -> Bool:
    """DEVIATION 2830's test, `lml - previous < 1e-10` at float32."""
    return ftz(lml - previous) < gpc_lml_tol()


# ===========================================================================
# PREDICTION'S HOST ARITHMETIC (_gpc.py:287-329, :434-441)
# ===========================================================================


def gpc_residual(y: List[Float32], pi: List[Float32]) -> List[Float32]:
    """`y_train_ - pi_`, the right-hand side of the latent mean."""
    var n = len(y)
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(ftz(ftz(y[i]) - ftz(pi[i])))
    return out^


def gpc_scale_rows(
    kcross: List[Float32], wsr: List[Float32], n_train: Int, n_star: Int
) -> List[Float32]:
    """`W_sr[:, None] * K_star` over the `n_train x n_star` cross-covariance
    (DEVIATION 1758's orientation), the right-hand sides of `solve(L, .)`."""
    var out = List[Float32](capacity=n_train * n_star)
    for i in range(n_train):
        var s = ftz(wsr[i])
        for t in range(n_star):
            out.append(ftz(identical_mul(s, ftz(kcross[i * n_star + t]))))
    return out^


def gpc_latent_var(
    v: List[Float32], n_train: Int, n_star: Int, kss: Float32
) -> List[Float32]:
    """`kernel.diag(X) - einsum("ij,ij->j", v, v)` (`_gpc.py:439`), the fold
    over `i` ascending. NOT clamped: the reference does not clamp, and a
    non-positive value reaches `gpc_pi_star`'s limit arm instead."""
    var out = List[Float32](capacity=n_star)
    for t in range(n_star):
        var acc = Float32(0.0)
        for i in range(n_train):
            var vv = ftz(v[i * n_star + t])
            acc = ftz(identical_mul_add(vv, vv, acc))
        out.append(ftz(ftz(kss) - acc))
    return out^


def _mul64(a: Float64, b: Float64) -> Float64:
    """A float64 product no code generator may contract into a neighbor's
    add (IDENTITY_PATHS row 9's pin, `identical_mul`'s float64 twin)."""
    return fma(a, b, Float64(-0.0))


def gpc_erf64(x: Float64) -> Float64:
    """DEVIATION 2832's float64 erf. See this file's header."""
    if x != x:
        return x
    var ax = x
    var negative = False
    if x < Float64(0.0):
        ax = -x
        negative = True
    var r = Float64(1.0)
    if ax < Float64(3.0):
        var x2 = _mul64(ax, ax)
        var neg_x2 = -x2
        var t = ax
        var s = ax
        for n in range(1, GPC_ERF_SERIES_TERMS):
            var tn = _mul64(t, neg_x2)
            t = tn / Float64(n)
            var q = t / Float64(2 * n + 1)
            s = s + q
        r = _mul64(s, bitcast[DType.float64](GPC_TWO_OVER_SQRT_PI64_BITS))
    elif ax < Float64(6.0):
        var t = ax
        for kk in range(GPC_ERFC_CF_LEVELS):
            var k = GPC_ERFC_CF_LEVELS - kk
            var h = _mul64(Float64(k), Float64(0.5))
            var q = h / t
            t = ax + q
        var e = identical_exp64(-_mul64(ax, ax))
        var den = _mul64(bitcast[DType.float64](GPC_SQRT_PI64_BITS), t)
        var c = e / den
        r = Float64(1.0) - c
    if negative:
        return -r
    return r


def _lambda64(k: Int) -> Float64:
    if k == 0:
        return bitcast[DType.float64](GPC_LAMBDA0_BITS)
    if k == 1:
        return bitcast[DType.float64](GPC_LAMBDA1_BITS)
    if k == 2:
        return bitcast[DType.float64](GPC_LAMBDA2_BITS)
    if k == 3:
        return bitcast[DType.float64](GPC_LAMBDA3_BITS)
    return bitcast[DType.float64](GPC_LAMBDA4_BITS)


def _coef64(k: Int) -> Float64:
    if k == 0:
        return bitcast[DType.float64](GPC_COEF0_BITS)
    if k == 1:
        return bitcast[DType.float64](GPC_COEF1_BITS)
    if k == 2:
        return bitcast[DType.float64](GPC_COEF2_BITS)
    if k == 3:
        return bitcast[DType.float64](GPC_COEF3_BITS)
    return bitcast[DType.float64](GPC_COEF4_BITS)


def gpc_pi_star(mean: Float32, variance: Float32) -> Float64:
    """The probability of class 1 at one query row, `_gpc.py:320-327`."""
    var mu = Float64(mean)
    var va = Float64(variance)
    var pi64 = bitcast[DType.float64](GPC_PI64_BITS)
    var acc = Float64(0.0)
    for k in range(5):
        var lam = _lambda64(k)
        var gamma = _mul64(lam, mu)
        var integral = Float64(0.0)
        if va > Float64(0.0):
            var alpha = Float64(1.0) / _mul64(Float64(2.0), va)
            var lam2 = _mul64(lam, lam)
            var ratio = alpha / (alpha + lam2)
            var arg = _mul64(gamma, sqrt(ratio))
            var num = _mul64(sqrt(pi64 / alpha), gpc_erf64(arg))
            var inner = _mul64(_mul64(va, Float64(2.0)), pi64)
            var den = _mul64(Float64(2.0), sqrt(inner))
            integral = num / den
        else:
            integral = _mul64(Float64(0.5), gpc_erf64(gamma))
        var term = _mul64(_coef64(k), integral)
        acc = acc + term
    return acc + bitcast[DType.float64](GPC_HALF_COEF_SUM_BITS)


def gpc_proba(mean: List[Float32], variance: List[Float32]) raises -> List[Float64]:
    """`pi_star` per query row (class 1); the surface builds `1 - pi_star`."""
    if len(mean) != len(variance):
        raise Error(
            "gpc_proba: the latent mean holds "
            + String(len(mean))
            + " rows and the variance "
            + String(len(variance))
        )
    var n = len(mean)
    var out = List[Float64](length=n, fill=Float64(0.0))
    var tasks = host_predict_task_count(n)
    # The probability expansion is expensive, but a pool join still dominates
    # very small prediction batches.
    if n < 128:
        tasks = 1
    var chunk = host_predict_chunk(n, tasks)
    var op = out.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()

    def _rows(task: Int) {imm mean, imm variance, imm n, imm chunk, imm op}:
        var lo = task * chunk
        var hi = min(lo + chunk, n)
        for t in range(lo, hi):
            op.unsafe_store(t, gpc_pi_star(mean[t], variance[t]))

    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)
    return out^
