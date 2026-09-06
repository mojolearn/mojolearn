# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""References for the `fit` gates. NOT A PORT, and deliberately NOT a copy
of the device spelling.

`archive/research/arima/SABOTAGES.md` opens by saying that this lane's expected values are
OUR OWN TALLY, so an arm that does not bite proves nothing about the gate.
For `fit` we can do better than a tally, and everything in this file exists
to make the `fit` gates independent of how the device spells things:

    normal_eq_residual_f64      asks whether `estimate_x0`'s answer SOLVES
                                the least-squares problem, in Float64, at
                                the device's own x. A solver with the wrong
                                association is still a solver; a solver with
                                the wrong ANSWER is not, and this is the
                                only question that separates them.
    build_ls_system_f64         rebuilds the design matrix from its
                                DEFINITION (lags of y, an intercept column,
                                lags of an AR pre-fit's residual) rather
                                than from the device's scratch. A mistake
                                here makes the residual LARGE, never small,
                                so it cannot produce a false pass.
    grad_f64_central            a Float64 CENTRAL difference, which is the
                                reference DEVIATION 687's step `h` is chosen
                                against. The existing gate only ever asked
                                whether the device gradient equalled a host
                                replay of the same forward difference: two
                                spellings of the same possibly-wrong number.
    jones_transform_host_f64    so the reference above spans the TRANSFORM.
                                `kalman_host_f64`'s docstring says it starts
                                DOWNSTREAM of the transform, on the same
                                Float32 transformed parameters the device
                                uses, which is right for isolating the
                                filter and wrong for a gradient in the
                                unconstrained coordinates the optimizer
                                actually moves.
    test_invparams_host         BOTH associations, side by side, so
                                `check_invparams_contraction_is_visible` can
                                assert the seam is observable before any
                                sabotage claims to move it.

THE REFERENCE GRADIENT'S OWN ERROR BUDGET, stated because a reference
without one is an assertion. `kalman_host_f64` takes `ARIMAParamsHost`,
which is Float32, so the Float64 transformed parameters are NARROWED before
the Float64 filter runs. That quantization is about `ulp(theta) = 6e-8`
relative, and over a central-difference step of `1e-3` it contributes about
`6e-5` relative error to the reference gradient. What the reference is used
to measure is a Float32 forward-difference error of order `1e-3`, so the
reference is roughly twenty times sharper than the thing it measures. That
is enough, and it is not comfortable. STRENGTHENING IT IS EXACTLY
`arima/README.md`'s OWED item 5: give `kalman_host_f64` a Float64 parameter
type so the whole reference is Float64 end to end. This file is the second
caller that wants it.
"""

from std.math import atanh, tanh

from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParamsHost
from arima.checks.kalman_oracle import kalman_host_f64
from tsa.impl.timeSeries.arima_helpers import prepare_data_host


comptime JONES_CLAMP_F64 = 0.9999
comptime MIN_SIGMA2_F64 = 1.0e-6


# ---------------------------------------------------------------------------
# the Jones transform in Float64
# ---------------------------------------------------------------------------


def jones_transform_f64(
    params: List[Float64], batch_size: Int, parameter: Int, is_ar: Bool, is_inv: Bool
) raises -> List[Float64]:
    """`jones_transform.cuh:36-90` in Float64 with the stdlib `tanh` /
    `atanh`, which is what THEIRS runs (`raft::tanh` on `double`). Neither
    DEVIATION 675's `identical_exp` identity nor its `identical_log` one
    appears here; that is the point."""
    var out = List[Float64]()
    for model in range(batch_size):
        var tmp = List[Float64]()
        var mine = List[Float64]()
        for i in range(parameter):
            var v = params[model * parameter + i]
            tmp.append(v)
            mine.append(v)
        if is_inv:
            var sign = 1.0 if is_ar else -1.0
            var j = parameter - 1
            while j > 0:
                var a = mine[j]
                var den = 1.0 - a * a
                for k in range(j):
                    tmp[k] = (mine[k] + sign * (a * mine[j - k - 1])) / den
                for it in range(j):
                    mine[it] = tmp[it]
                j -= 1
            for i in range(parameter):
                mine[i] = 2.0 * atanh(mine[i])
        else:
            for i in range(parameter):
                tmp[i] = tanh(tmp[i] * 0.5)
                mine[i] = tmp[i]
            var sign = -1.0 if is_ar else 1.0
            for j in range(1, parameter):
                var a = mine[j]
                for k in range(j):
                    tmp[k] = tmp[k] + sign * (a * mine[j - k - 1])
                for it in range(j):
                    mine[it] = tmp[it]
            for i in range(parameter):
                var v = mine[i]
                mine[i] = max(min(v, JONES_CLAMP_F64), -JONES_CLAMP_F64)
        for i in range(parameter):
            out.append(mine[i])
    return out^


def _f64(v: List[Float32]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(v)):
        out.append(Float64(v[i]))
    return out^


def _f32(v: List[Float64]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(Float32(v[i]))
    return out^


def batched_jones_transform_f64(
    order: ARIMAOrder, batch_size: Int, is_inv: Bool, ph: ARIMAParamsHost
) raises -> ARIMAParamsHost:
    """`batched_jones_transform` in Float64, narrowed back to Float32 on the
    way out because `kalman_host_f64` takes `ARIMAParamsHost`. The
    narrowing is the reference's whole error budget; see the banner."""
    var ar = ph.ar.copy()
    var ma = ph.ma.copy()
    var sar = ph.sar.copy()
    var sma = ph.sma.copy()
    if order.p != 0:
        ar = _f32(jones_transform_f64(_f64(ph.ar), batch_size, order.p, True, is_inv))
    if order.q != 0:
        ma = _f32(jones_transform_f64(_f64(ph.ma), batch_size, order.q, False, is_inv))
    if order.P != 0:
        sar = _f32(jones_transform_f64(_f64(ph.sar), batch_size, order.P, True, is_inv))
    if order.Q != 0:
        sma = _f32(jones_transform_f64(_f64(ph.sma), batch_size, order.Q, False, is_inv))
    var sigma2 = List[Float32]()
    for i in range(batch_size):
        sigma2.append(Float32(max(Float64(ph.sigma2[i]), MIN_SIGMA2_F64)))
    return ARIMAParamsHost(
        mu=ph.mu.copy(), ar=ar^, ma=ma^, sar=sar^, sma=sma^, sigma2=sigma2^
    )


# ---------------------------------------------------------------------------
# the objective and its Float64 gradient
# ---------------------------------------------------------------------------


def _one_zero() -> List[Float32]:
    """The `max(1, n)` allocation `ARIMAParams` makes for an absent kind, so
    an absent kind still has a list to hand to `kalman_host_f64`."""
    var out = List[Float32]()
    out.append(Float32(0.0))
    return out^


def objective_f64(
    y: List[Float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    x: List[Float64],
) raises -> List[Float64]:
    """`-loglike(jones(x)) / (n_obs - 1)`, per series, with the JONES
    TRANSFORM in Float64 and the filter in Float64 (`kalman_host_f64`).

    The differencing is `prepare_data_host` and the filter runs on the
    differenced series with `d = D = 0`, which is what `batched_fit` does;
    the `n_obs - 1` scale uses the ORIGINAL length, which is what
    `arima.pyx:910` does."""
    var N = order.complexity()
    var ar = List[Float64]()
    var ma = List[Float64]()
    var sar = List[Float64]()
    var sma = List[Float64]()
    var mu = List[Float32]()
    var sigma2 = List[Float32]()
    for b in range(batch_size):
        var o = b * N
        if order.k != 0:
            mu.append(Float32(x[o]))
            o += 1
        for i in range(order.p):
            ar.append(x[o + i])
        o += order.p
        for i in range(order.q):
            ma.append(x[o + i])
        o += order.q
        for i in range(order.P):
            sar.append(x[o + i])
        o += order.P
        for i in range(order.Q):
            sma.append(x[o + i])
        o += order.Q
        sigma2.append(Float32(max(x[o], MIN_SIGMA2_F64)))
    var t_ar = _one_zero()
    var t_ma = _one_zero()
    var t_sar = _one_zero()
    var t_sma = _one_zero()
    if order.p != 0:
        t_ar = _f32(jones_transform_f64(ar, batch_size, order.p, True, False))
    if order.q != 0:
        t_ma = _f32(jones_transform_f64(ma, batch_size, order.q, False, False))
    if order.P != 0:
        t_sar = _f32(jones_transform_f64(sar, batch_size, order.P, True, False))
    if order.Q != 0:
        t_sma = _f32(jones_transform_f64(sma, batch_size, order.Q, False, False))
    if order.k == 0:
        mu.append(Float32(0.0))
    var tp = ARIMAParamsHost(
        mu=mu^, ar=t_ar^, ma=t_ma^, sar=t_sar^, sma=t_sma^, sigma2=sigma2^
    )
    var y_kf = y.copy()
    var n_obs_kf = n_obs
    var order_kf = order
    if order.need_diff():
        y_kf = prepare_data_host(y, batch_size, n_obs, order.d, order.D, order.s)
        n_obs_kf = n_obs - order.n_diff()
        order_kf = order.without_diff()
    var f = kalman_host_f64(y_kf, batch_size, n_obs_kf, order_kf, tp, 0)
    var out = List[Float64]()
    var scale = Float64(n_obs - 1)
    for b in range(batch_size):
        out.append(-f.loglike[b] / scale)
    return out^


def grad_f64_central(
    y: List[Float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    x: List[Float64],
    h64: Float64,
) raises -> List[Float64]:
    """A Float64 CENTRAL difference of `objective_f64`, `O(h^2)` where the
    device's forward difference is `O(h)`. This is the reference DEVIATION
    687 is chosen against; nothing in this lane compared the gradient to
    anything but another spelling of itself before it existed."""
    var N = order.complexity()
    var out = List[Float64]()
    for _ in range(N * batch_size):
        out.append(0.0)
    for i in range(N):
        var xp = x.copy()
        var xm = x.copy()
        for b in range(batch_size):
            xp[N * b + i] = x[N * b + i] + h64
            xm[N * b + i] = x[N * b + i] - h64
        var fp = objective_f64(y, batch_size, n_obs, order, xp)
        var fm = objective_f64(y, batch_size, n_obs, order, xm)
        for b in range(batch_size):
            out[N * b + i] = (fp[b] - fm[b]) / (2.0 * h64)
    return out^


# ---------------------------------------------------------------------------
# the least-squares system, rebuilt from its definition in Float64
# ---------------------------------------------------------------------------


def solve_normal_f64(a: List[Float64], m: Int, n: Int, b: List[Float64]) raises -> List[Float64]:
    """`min |A x - b|` in Float64 through the normal equations and Gaussian
    elimination with partial pivoting.

    THE NORMAL EQUATIONS ARE FINE HERE AND ARE NOT FINE ON THE DEVICE, and
    the difference is the whole of DEVIATION 678's accuracy argument.
    Float64 carries 15.9 digits, so squaring a condition number of `1e4`
    leaves 7 digits of the answer; Float32 carries 7.2 and the same squaring
    leaves none. This routine is used only to build a reference and only in
    Float64."""
    var g = List[Float64]()
    for _ in range(n * n):
        g.append(0.0)
    var r = List[Float64]()
    for _ in range(n):
        r.append(0.0)
    for i in range(n):
        for j in range(n):
            var acc = 0.0
            for t in range(m):
                acc += a[t + i * m] * a[t + j * m]
            g[i + j * n] = acc
        var accb = 0.0
        for t in range(m):
            accb += a[t + i * m] * b[t]
        r[i] = accb
    # Gaussian elimination with partial pivoting on [g | r]
    for col in range(n):
        var best = col
        var bmag = abs(g[col + col * n])
        for i in range(col + 1, n):
            var mg = abs(g[i + col * n])
            if mg > bmag:
                bmag = mg
                best = i
        if bmag == 0.0:
            raise Error("solve_normal_f64: singular normal equations at column " + String(col))
        if best != col:
            for c in range(n):
                var t0 = g[col + c * n]
                g[col + c * n] = g[best + c * n]
                g[best + c * n] = t0
            var t1 = r[col]
            r[col] = r[best]
            r[best] = t1
        for i in range(col + 1, n):
            var f = g[i + col * n] / g[col + col * n]
            for c in range(col, n):
                g[i + c * n] = g[i + c * n] - f * g[col + c * n]
            r[i] = r[i] - f * r[col]
    var x = List[Float64]()
    for _ in range(n):
        x.append(0.0)
    var i = n - 1
    while i >= 0:
        var acc = r[i]
        for c in range(i + 1, n):
            acc -= g[i + c * n] * x[c]
        x[i] = acc / g[i + i * n]
        i -= 1
    return x^


@fieldwise_init
struct LsSystem(Movable):
    """One series' `(A, b)` in Float64, column-major, `m x n`."""

    var a: List[Float64]
    var b: List[Float64]
    var m: Int
    var n: Int


def build_ls_system_f64(
    yd: List[Float32],
    bid: Int,
    n_obs_d: Int,
    p: Int,
    q: Int,
    s: Int,
    k: Int,
) raises -> LsSystem:
    """`bm_ls_ar_res` and `bm_arma_fit` for one series, rebuilt from their
    DEFINITION (`batched_arima.cu:699-771`) rather than from the device's
    scratch.

    NOTE WHERE THIS IS EXACT AND WHERE IT IS NOT. With `q == 0` the matrix
    is an intercept column and lags of `y`, every entry of which is a
    Float32 value of `y` widened exactly, so `A_f64 == A_f32` BIT FOR BIT
    and the residual gate is exact. With `q > 0` the last `q` columns are
    lags of an AR pre-fit's residual, and this reference computes that
    pre-fit in Float64 where the device computes it in Float32, so the two
    matrices differ by about `1e-7` relative and the gate's bound has to
    admit it. `check_x0_solves_the_normal_equations` runs both arms and
    holds the tight bound only on the exact one."""
    var ps = p * s
    var qs = q * s
    var p_ar = ps if ps > 2 * qs else 2 * qs
    var r_ls = (p_ar + qs) if (p_ar + qs) > ps else ps
    var m1 = n_obs_d - r_ls
    var ncols = p + q + k
    var yb = bid * n_obs_d
    var a = List[Float64]()
    for _ in range(m1 * ncols):
        a.append(0.0)
    var b = List[Float64]()
    for i in range(m1):
        b.append(Float64(yd[yb + r_ls + i]))
    if k != 0:
        for i in range(m1):
            a[i] = 1.0
    var ar_offset = r_ls - ps
    for lag in range(p):
        var src = yb + ar_offset + s * (p - lag - 1)
        for i in range(m1):
            a[m1 * k + lag * m1 + i] = Float64(yd[src + i])
    if q != 0:
        var m2 = n_obs_d - p_ar
        var pre = List[Float64]()
        for _ in range(m2 * p_ar):
            pre.append(0.0)
        for lag in range(p_ar):
            var src = yb + (p_ar - lag - 1)
            for i in range(m2):
                pre[lag * m2 + i] = Float64(yd[src + i])
        var tgt = List[Float64]()
        for i in range(m2):
            tgt.append(Float64(yd[yb + p_ar + i]))
        var coef = solve_normal_f64(pre, m2, p_ar, tgt)
        var resid = List[Float64]()
        for i in range(m2):
            var acc = tgt[i]
            for c in range(p_ar):
                acc -= pre[c * m2 + i] * coef[c]
            resid.append(acc)
        var res_offset = r_ls - p_ar - qs
        for lag in range(q):
            var src = res_offset + s * (q - lag - 1)
            for i in range(m1):
                a[m1 * (k + p) + lag * m1 + i] = resid[src + i]
    return LsSystem(a=a^, b=b^, m=m1, n=ncols)


def normal_eq_residual_f64(sys: LsSystem, x: List[Float64]) raises -> Float64:
    """`|A'(A x - b)|_inf`, scaled by `|A|_inf_col * |b|_inf` so the number
    is dimensionless.

    THIS IS THE GATE THAT DOES NOT CARE HOW THE SOLVER IS SPELLED. `x` is a
    least-squares solution exactly when `A'(A x - b) = 0`; a QR with the
    wrong association, a different pivot rule, or a different fold order all
    still satisfy it to rounding. Only a WRONG ANSWER fails."""
    var m = sys.m
    var n = sys.n
    var res = List[Float64]()
    var bmax = 0.0
    for i in range(m):
        var acc = -sys.b[i]
        if abs(sys.b[i]) > bmax:
            bmax = abs(sys.b[i])
        for c in range(n):
            acc += sys.a[c * m + i] * x[c]
        res.append(acc)
    var amax = 0.0
    for c in range(n):
        var col = 0.0
        for i in range(m):
            col += abs(sys.a[c * m + i])
        if col > amax:
            amax = col
    var worst = 0.0
    for c in range(n):
        var acc = 0.0
        for i in range(m):
            acc += sys.a[c * m + i] * res[i]
        if abs(acc) > worst:
            worst = abs(acc)
    var denom = amax * bmax
    if denom == 0.0:
        return worst
    return worst / denom


# ---------------------------------------------------------------------------
# test_invparams, BOTH associations
# ---------------------------------------------------------------------------


def test_invparams_host(
    params: List[Float32], base: Int, pq: Int, is_ar: Bool, wrong_association: Bool
) raises -> Bool:
    """`test_invparams` (`batched_arima.cu:634-657`) on the host, in Float32,
    spelled TWICE.

    `wrong_association = False` is theirs: `coef * a * x` is `(coef*a) * x`,
    `coef*a` is exact, the surviving product feeds the add, ONE rounding.

    `wrong_association = True` is `invtransform`'s tree, `sign * (a * x)`,
    which rounds `a * x` first: TWO roundings. It is HERE, in the oracle,
    rather than only in a sabotage patch, because
    `check_invparams_contraction_is_visible` has to prove the two differ on
    this fixture BEFORE any sabotage can claim to have moved something. That
    is the shape `check_jones_contraction_is_visible` already uses for the
    same seam one file over."""
    from checks.numerics import ftz, identical_mul_add

    var new_params = List[Float32]()
    var tmp = List[Float32]()
    for i in range(pq):
        var v = ftz(params[base + i])
        new_params.append(v)
        tmp.append(v)
    var j = pq - 1
    while j > 0:
        var a = new_params[j]
        var den = ftz(identical_mul_add(-a, a, Float32(1.0)))
        var sign = Float32(1.0) if is_ar else Float32(-1.0)
        var coef_a = a if is_ar else ftz(-a)
        for kk in range(j):
            var num: Float32
            if wrong_association:
                var prod = ftz(a * new_params[j - kk - 1])
                num = ftz(identical_mul_add(sign, prod, new_params[kk]))
            else:
                num = ftz(
                    identical_mul_add(coef_a, new_params[j - kk - 1], new_params[kk])
                )
            tmp[kk] = ftz(num / den)
        for it in range(j):
            new_params[it] = tmp[it]
        j -= 1
    var result = True
    for i in range(pq):
        var v = new_params[i]
        result = result and not (v <= Float32(-1.0) or v >= Float32(1.0))
    return result


# ---------------------------------------------------------------------------
# the Float32 host replay of DEVIATION 678's QR
# ---------------------------------------------------------------------------


def householder_qr_solve_host(
    mut a: List[Float32], m: Int, n: Int, mut b: List[Float32]
) raises -> Int:
    """The serial host replay of
    `arima/impl/linalg/batched/least_squares.mojo::householder_qr_solve`,
    through the SAME numeric helpers, so under IDENTICAL the device result
    must equal this bit for bit.

    RE-SPELLED HERE RATHER THAN IMPORTED, which is `kalman_oracle.mojo`'s
    rule and the reason it is a rule: a sabotage of the device spelling must
    not be able to move the oracle with it. Two people wrote the same
    arithmetic twice on purpose."""
    from checks.numerics import ftz, identical_mul_add, identical_sqrt

    var rdiag = List[Float32]()
    for _ in range(n):
        rdiag.append(Float32(0.0))
    for j in range(n):
        var sigma = Float32(0.0)
        for i in range(j, m):
            var v = ftz(a[i + j * m])
            sigma = ftz(identical_mul_add(v, v, sigma))
        if sigma == Float32(0.0):
            return j + 1
        var normx = ftz(identical_sqrt(sigma))
        var ajj = ftz(a[j + j * m])
        var s = Float32(-1.0) if ajj >= Float32(0.0) else Float32(1.0)
        var r_jj = ftz(s * normx)
        var u1 = ftz(ajj - r_jj)
        if u1 == Float32(0.0):
            return j + 1
        for i in range(j + 1, m):
            a[i + j * m] = ftz(ftz(a[i + j * m]) / u1)
        var tau = ftz(ftz(ftz(-s) * u1) / normx)
        rdiag[j] = r_jj
        for c in range(j + 1, n):
            var acc = ftz(a[j + c * m])
            for i in range(j + 1, m):
                acc = ftz(identical_mul_add(ftz(a[i + j * m]), ftz(a[i + c * m]), acc))
            var td = ftz(tau * acc)
            a[j + c * m] = ftz(a[j + c * m] - td)
            for i in range(j + 1, m):
                a[i + c * m] = ftz(
                    identical_mul_add(-td, ftz(a[i + j * m]), ftz(a[i + c * m]))
                )
        var accb = ftz(b[j])
        for i in range(j + 1, m):
            accb = ftz(identical_mul_add(ftz(a[i + j * m]), ftz(b[i]), accb))
        var tdb = ftz(tau * accb)
        b[j] = ftz(b[j] - tdb)
        for i in range(j + 1, m):
            b[i] = ftz(identical_mul_add(-tdb, ftz(a[i + j * m]), ftz(b[i])))
    var rmax = Float32(0.0)
    for j in range(n):
        var v = abs(rdiag[j])
        if v > rmax:
            rmax = v
    if rmax == Float32(0.0):
        return 1
    from arima.impl.linalg.batched.least_squares import LS_RANK_TOL

    for j in range(n):
        if abs(rdiag[j]) <= ftz(LS_RANK_TOL * rmax):
            return j + 1
    var i = n - 1
    while i >= 0:
        var acc = ftz(b[i])
        for c in range(i + 1, n):
            acc = ftz(identical_mul_add(-ftz(a[i + c * m]), ftz(b[c]), acc))
        b[i] = ftz(acc / rdiag[i])
        i -= 1
    return 0


def normal_equations_solve_f32(a: List[Float32], m: Int, n: Int, b: List[Float32]) raises -> List[Float32]:
    """The route DEVIATION 678 REJECTED, in Float32, so the rejection is a
    MEASUREMENT and not an argument: form `A'A` and `A'b`, then Cholesky and
    two triangular solves.

    `check_qr_beats_normal_equations_on_ill_conditioning` runs this beside
    the QR on a lag matrix with a root near the unit circle and compares
    both against the Float64 answer. If this route ever wins, DEVIATION
    678's accuracy argument is wrong and the deviation should be revisited.

    A non-positive pivot RAISES here rather than returning a NaN, because
    that outcome IS the finding: the Gram of a well-posed problem went
    indefinite in Float32."""
    from checks.numerics import ftz, identical_mul_add, identical_sqrt

    var g = List[Float32]()
    for _ in range(n * n):
        g.append(Float32(0.0))
    var r = List[Float32]()
    for _ in range(n):
        r.append(Float32(0.0))
    for i in range(n):
        for j in range(n):
            var acc = Float32(0.0)
            for t in range(m):
                acc = ftz(identical_mul_add(a[t + i * m], a[t + j * m], acc))
            g[i + j * n] = acc
        var accb = Float32(0.0)
        for t in range(m):
            accb = ftz(identical_mul_add(a[t + i * m], b[t], accb))
        r[i] = accb
    # Cholesky, lower
    for j in range(n):
        var d = ftz(g[j + j * n])
        for c in range(j):
            var l = ftz(g[j + c * n])
            d = ftz(identical_mul_add(-l, l, d))
        if not (d > Float32(0.0)):
            raise Error(
                "normal_equations_solve_f32: the Gram matrix is not positive"
                " definite in Float32 at column " + String(j)
                + " -- which is the outcome DEVIATION 678 predicts for a"
                " near-unit-root design"
            )
        var djj = ftz(identical_sqrt(d))
        g[j + j * n] = djj
        for i in range(j + 1, n):
            var acc = ftz(g[i + j * n])
            for c in range(j):
                acc = ftz(identical_mul_add(-ftz(g[i + c * n]), ftz(g[j + c * n]), acc))
            g[i + j * n] = ftz(acc / djj)
    var yv = List[Float32]()
    for _ in range(n):
        yv.append(Float32(0.0))
    for i in range(n):
        var acc = ftz(r[i])
        for c in range(i):
            acc = ftz(identical_mul_add(-ftz(g[i + c * n]), yv[c], acc))
        yv[i] = ftz(acc / ftz(g[i + i * n]))
    var x = List[Float32]()
    for _ in range(n):
        x.append(Float32(0.0))
    var i2 = n - 1
    while i2 >= 0:
        var acc = ftz(yv[i2])
        for c in range(i2 + 1, n):
            acc = ftz(identical_mul_add(-ftz(g[c + i2 * n]), x[c], acc))
        x[i2] = ftz(acc / ftz(g[i2 + i2 * n]))
        i2 -= 1
    return x^
