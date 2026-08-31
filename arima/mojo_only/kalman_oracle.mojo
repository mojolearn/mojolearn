# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host oracles for the batched Kalman filter. NOT A PORT.

`kalman_host_f32` replays `init_batched_kalman_matrices_kernel`,
`kalman_init_state_kernel` and `batched_kalman_loop_kernel`
(`arima/ported/arima/batched_kalman.mojo`) serially on the host, one series
at a time, ascending, in Float32 through the SAME numeric helpers
(`identical_mul_add`, `ftz`, `identical_log`), so under IDENTICAL the
device buffers must equal these lists BIT FOR BIT. The tiny dense
contractions (`_mv`/`_mm`, the LU solve) are re-spelled here rather than
imported from the device file, so a sabotage of the device spelling cannot
move the oracle with it; the pure scalar functions that are shared
(`param_to_poly`, `reduced_polynomial`, `reduced_poly_indices`,
`jones_transform_host`, the `lu_inverse_host` family) are shared because
they are host functions already, spelled separately from their kernels.

`kalman_host_f64` is the tolerance reference (DEVIATION 670): the same
structure in Float64 with the stdlib arithmetic, so the Float32 error of
the whole pipeline is MEASURED, never assumed.
"""

from std.math import log
from std.memory import bitcast

from arima.ported.timeSeries.arima_helpers import (
    param_to_poly,
    reduced_poly_indices,
    reduced_polynomial,
)
from arima.ported.tsa.arima_common import ARIMAOrder, ARIMAParamsHost
from arima.ported.linalg.batched.matrix import (
    kron_minus_identity_host,
    lu_inverse_host,
    matvec_serial_host,
)
from mojo_only.numerics import ftz, identical_log, identical_mul_add


comptime ORACLE_LOG_2PI = Float32(1.8378770664093453)
comptime ORACLE_KAPPA = Float32(1e6)


@fieldwise_init
struct KalmanHostStages(Movable):
    """Every stage the device writes, in the device's layout."""

    var Z: List[Float32]
    var R: List[Float32]
    var T: List[Float32]
    var RQ: List[Float32]
    var RQR: List[Float32]
    var P0: List[Float32]
    var alpha0: List[Float32]
    var pred: List[Float32]
    var vs: List[Float32]
    var loglike: List[Float32]
    var fc: List[Float32]
    var rd: Int
    var r: Int
    var n_diff: Int


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def init_matrices_host(
    order: ARIMAOrder, batch_size: Int, params: ARIMAParamsHost
) -> Tuple[List[Float32], List[Float32], List[Float32]]:
    """`init_batched_kalman_matrices_kernel`, serial per series."""
    var p = order.p
    var d = order.d
    var q = order.q
    var P = order.P
    var D = order.D
    var Q = order.Q
    var s = order.s
    var n_diff = order.n_diff()
    var n_phi = order.n_phi()
    var n_theta = order.n_theta()
    var r = order.r()
    var rd = order.rd()
    var Zl = _zeros(rd * batch_size)
    var Rl = _zeros(rd * batch_size)
    var Tl = _zeros(rd * rd * batch_size)
    for bid in range(batch_size):
        var zb = bid * rd
        var tb = bid * rd * rd
        for i in range(d):
            Zl[zb + i] = Float32(1.0)
        for i in range(1, D + 1):
            Zl[zb + d + i * s - 1] = Float32(1.0)
        Zl[zb + n_diff] = Float32(1.0)
        Rl[zb + n_diff] = Float32(1.0)
        for i in range(n_theta):
            var ix = reduced_poly_indices(i + 1, s)
            var c0 = param_to_poly(False, params.ma[bid * q + ix[0] - 1] if (ix[0] != 0 and ix[0] <= q) else Float32(0.0), ix[0], q)
            var c1 = param_to_poly(False, params.sma[bid * Q + ix[1] - 1] if (ix[1] != 0 and ix[1] <= Q) else Float32(0.0), ix[1], Q)
            Rl[zb + n_diff + i + 1] = reduced_polynomial(False, c0, c1)
        for i in range(d):
            for j in range(i, d):
                Tl[tb + j * rd + i] = Float32(1.0)
        for id_ in range(d):
            Tl[tb + n_diff * rd + id_] = Float32(1.0)
            for iD in range(1, D + 1):
                Tl[tb + (d + s * iD - 1) * rd + id_] = Float32(1.0)
        for iD in range(D):
            var offset = d + iD * s
            for i in range(s - 1):
                Tl[tb + (offset + i) * rd + offset + i + 1] = Float32(1.0)
            Tl[tb + (offset + s - 1) * rd + offset] = Float32(1.0)
            Tl[tb + n_diff * rd + offset] = Float32(1.0)
        if D == 2:
            Tl[tb + (n_diff - 1) * rd + d] = Float32(1.0)
        for i in range(n_phi):
            var ix = reduced_poly_indices(i + 1, s)
            var c0 = param_to_poly(True, params.ar[bid * p + ix[0] - 1] if (ix[0] != 0 and ix[0] <= p) else Float32(0.0), ix[0], p)
            var c1 = param_to_poly(True, params.sar[bid * P + ix[1] - 1] if (ix[1] != 0 and ix[1] <= P) else Float32(0.0), ix[1], P)
            Tl[tb + n_diff * (rd + 1) + i] = reduced_polynomial(True, c0, c1)
        for i in range(r - 1):
            Tl[tb + (n_diff + i + 1) * rd + n_diff + i] = Float32(1.0)
        if rd == 2 and p == 2:
            var t1 = ftz(Tl[tb + 1])
            if abs(ftz(t1 + Float32(1.0))) < Float32(0.01):
                Tl[tb + 1] = Float32(-0.99)
    return (Zl^, Rl^, Tl^)


def init_state_host(
    order: ARIMAOrder,
    batch_size: Int,
    Rl: List[Float32],
    Tl: List[Float32],
    sigma2: List[Float32],
    mu: List[Float32],
) raises -> Tuple[List[Float32], List[Float32], List[Float32], List[Float32]]:
    """`kalman_init_state_kernel`, serial per series: (RQ, RQR, P0, alpha0)."""
    var rd = order.rd()
    var r = order.r()
    var n_diff = order.n_diff()
    var rd2 = rd * rd
    var r2 = r * r
    var RQ = _zeros(rd * batch_size)
    var RQR = _zeros(rd2 * batch_size)
    var P0 = _zeros(rd2 * batch_size)
    var alpha0 = _zeros(rd * batch_size)
    for bid in range(batch_size):
        var vb = bid * rd
        var mb = bid * rd2
        var s2 = ftz(sigma2[bid])
        for i in range(rd):
            RQ[vb + i] = ftz(ftz(Rl[vb + i]) * s2)
        for j in range(rd):
            var rj = ftz(Rl[vb + j])
            for i in range(rd):
                RQR[mb + i + j * rd] = ftz(ftz(RQ[vb + i]) * rj)
        for i in range(rd2):
            P0[mb + i] = Float32(0.0)
        for i in range(n_diff):
            P0[mb + (rd + 1) * i] = ORACLE_KAPPA
        var imaa = kron_minus_identity_host(Tl, mb, rd, n_diff, r)
        var inv = lu_inverse_host(imaa, r2)
        # the device stores vecq[i + j*r] = ftz(RQR[i+n_diff, j+n_diff]);
        # appending in (j outer, i inner) order lands each value at the
        # same index j*r + i == i + j*r
        var vq = List[Float32]()
        for j in range(r):
            for i in range(r):
                vq.append(ftz(RQR[mb + (i + n_diff) + (j + n_diff) * rd]))
        var x = matvec_serial_host(inv, r2, vq)
        for j in range(r):
            for i in range(r):
                P0[mb + (i + n_diff) + (j + n_diff) * rd] = x[i + j * r]
        if order.k != 0:
            var imt = _zeros(r2)
            for j in range(r):
                for i in range(r):
                    var delta = Float32(1.0) if i == j else Float32(0.0)
                    var tij = ftz(Tl[mb + (i + n_diff) + (j + n_diff) * rd])
                    imt[i + j * r] = ftz(delta - tij)
            if r == 1:
                var v = imt[0]
                if abs(v) < Float32(1e-3):
                    var neg = (bitcast[DType.uint32](v) >> 31) != 0
                    imt[0] = Float32(-1e-3) if neg else Float32(1e-3)
            var imt_inv = lu_inverse_host(imt, r)
            var muv = ftz(mu[bid])
            for i in range(n_diff):
                alpha0[vb + i] = Float32(0.0)
            for i in range(r):
                alpha0[vb + i + n_diff] = ftz(ftz(imt_inv[i]) * muv)
        else:
            for i in range(rd):
                alpha0[vb + i] = Float32(0.0)
    return (RQ^, RQR^, P0^, alpha0^)


def _mv_host(n: Int, alpha: Float32, a: List[Float32], v: List[Float32], mut out_v: List[Float32]):
    for i in range(n):
        var acc = Float32(0.0)
        for j in range(n):
            acc = ftz(identical_mul_add(a[i + j * n], v[j], acc))
        out_v[i] = ftz(alpha * acc)


def _mm_host(n: Int, a: List[Float32], b: List[Float32], bT: Bool, mut out_v: List[Float32]):
    for i in range(n):
        for j in range(n):
            var acc = Float32(0.0)
            for k in range(n):
                var bkj = b[j + k * n] if bT else b[k + j * n]
                acc = ftz(identical_mul_add(a[i + k * n], bkj, acc))
            out_v[i + j * n] = acc


def _numerical_stability_host(n: Int, mut a: List[Float32]):
    for i in range(n - 1):
        for j in range(i + 1, n):
            var s = ftz(a[j * n + i] + a[i * n + j])
            var new_val = ftz(Float32(0.5) * s)
            a[j * n + i] = new_val
            a[i * n + j] = new_val
    for i in range(n):
        a[i * n + i] = abs(a[i * n + i])


def kalman_loop_host(
    ys: List[Float32],
    Tl: List[Float32],
    Zl: List[Float32],
    RQR: List[Float32],
    P0: List[Float32],
    alpha0: List[Float32],
    mu_in: List[Float32],
    rd: Int,
    nobs: Int,
    batch_size: Int,
    intercept: Int,
    n_diff: Int,
    fc_steps: Int,
) -> Tuple[List[Float32], List[Float32], List[Float32], List[Float32]]:
    """`batched_kalman_loop_kernel`, serial per series: (pred, vs, loglike,
    fc)."""
    var rd2 = rd * rd
    var pred_l = _zeros(nobs * batch_size)
    var vs_l = _zeros(nobs * batch_size)
    var ll_l = _zeros(batch_size)
    var fc_l = _zeros(fc_steps * batch_size if fc_steps > 0 else 0)
    for bid in range(batch_size):
        var l_RQR = _zeros(rd2)
        var l_T = _zeros(rd2)
        var l_Z = _zeros(rd)
        var l_P = _zeros(rd2)
        var l_alpha = _zeros(rd)
        var l_K = _zeros(rd)
        var l_tmp = _zeros(rd2)
        var l_TP = _zeros(rd2)
        var l_v = _zeros(rd)
        var b_rd = bid * rd
        var b_rd2 = bid * rd2
        for i in range(rd2):
            l_RQR[i] = ftz(RQR[b_rd2 + i])
            l_T[i] = ftz(Tl[b_rd2 + i])
            l_P[i] = ftz(P0[b_rd2 + i])
        for i in range(rd):
            if n_diff > 0:
                l_Z[i] = ftz(Zl[b_rd + i])
            l_alpha[i] = ftz(alpha0[b_rd + i])
        var b_sum_logFs = Float32(0.0)
        var b_ll_s2 = Float32(0.0)
        var n_obs_ll = 0
        var b_ys = bid * nobs
        var mu = ftz(mu_in[bid]) if intercept != 0 else Float32(0.0)
        for it in range(nobs):
            var pred = Float32(0.0)
            if n_diff == 0:
                pred = ftz(pred + l_alpha[0])
            else:
                for i in range(rd):
                    pred = ftz(identical_mul_add(l_alpha[i], l_Z[i], pred))
            pred_l[b_ys + it] = pred
            var yt = ftz(ys[b_ys + it])
            var vs_it = ftz(yt - pred)
            vs_l[b_ys + it] = vs_it
            var _Fs = Float32(0.0)
            if n_diff == 0:
                _Fs = l_P[0]
            else:
                for i in range(rd):
                    for j in range(rd):
                        var t0 = ftz(l_P[j * rd + i] * l_Z[i])
                        _Fs = ftz(identical_mul_add(t0, l_Z[j], _Fs))
            if it >= n_diff:
                if _Fs > Float32(0.0):
                    b_sum_logFs = ftz(b_sum_logFs + ftz(identical_log(_Fs)))
                    var v2 = ftz(vs_it * vs_it)
                    b_ll_s2 = ftz(b_ll_s2 + ftz(v2 / _Fs))
                n_obs_ll += 1
            _mm_host(rd, l_T, l_P, False, l_TP)
            var _1_Fs = ftz(Float32(1.0) / _Fs)
            if n_diff == 0:
                for i in range(rd):
                    l_K[i] = ftz(_1_Fs * l_TP[i])
            else:
                _mv_host(rd, _1_Fs, l_TP, l_Z, l_K)
            _mv_host(rd, Float32(1.0), l_T, l_alpha, l_v)
            for i in range(rd):
                l_alpha[i] = ftz(identical_mul_add(l_K[i], vs_it, l_v[i]))
            l_alpha[n_diff] = ftz(l_alpha[n_diff] + mu)
            for i in range(rd2):
                l_tmp[i] = l_T[i]
            if n_diff == 0:
                for i in range(rd):
                    l_tmp[i] = ftz(l_tmp[i] - l_K[i])
            else:
                for i in range(rd):
                    for j in range(rd):
                        l_tmp[j * rd + i] = ftz(identical_mul_add(-l_K[i], l_Z[j], l_tmp[j * rd + i]))
            _mm_host(rd, l_TP, l_tmp, True, l_P)
            for i in range(rd2):
                l_P[i] = ftz(l_P[i] + l_RQR[i])
            _numerical_stability_host(rd, l_P)
        var n_obs_ll_f = Float32(n_obs_ll)
        b_ll_s2 = ftz(b_ll_s2 / n_obs_ll_f)
        var inner = ftz(b_ll_s2 + ORACLE_LOG_2PI)
        var tot = ftz(identical_mul_add(n_obs_ll_f, inner, b_sum_logFs))
        ll_l[bid] = ftz(Float32(-0.5) * tot)
        var b_fc = bid * fc_steps
        for it in range(fc_steps):
            var pred = Float32(0.0)
            if n_diff == 0:
                pred = ftz(pred + l_alpha[0])
            else:
                for i in range(rd):
                    pred = ftz(identical_mul_add(l_alpha[i], l_Z[i], pred))
            fc_l[b_fc + it] = pred
            _mv_host(rd, Float32(1.0), l_T, l_alpha, l_v)
            for i in range(rd):
                l_alpha[i] = l_v[i]
            l_alpha[n_diff] = ftz(l_alpha[n_diff] + mu)
    return (pred_l^, vs_l^, ll_l^, fc_l^)


def kalman_host_f32(
    y: List[Float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    t_params: ARIMAParamsHost,
    fc_steps: Int,
) raises -> KalmanHostStages:
    """The whole `batched_kalman_filter` on TRANSFORMED parameters."""
    var zrt = init_matrices_host(order, batch_size, t_params)
    var st = init_state_host(order, batch_size, zrt[1], zrt[2], t_params.sigma2, t_params.mu)
    var loop = kalman_loop_host(
        y, zrt[2], zrt[0], st[1], st[2], st[3], t_params.mu,
        order.rd(), n_obs, batch_size, order.k, order.n_diff(), fc_steps,
    )
    return KalmanHostStages(
        Z=zrt[0].copy(), R=zrt[1].copy(), T=zrt[2].copy(),
        RQ=st[0].copy(), RQR=st[1].copy(), P0=st[2].copy(), alpha0=st[3].copy(),
        pred=loop[0].copy(), vs=loop[1].copy(), loglike=loop[2].copy(), fc=loop[3].copy(),
        rd=order.rd(), r=order.r(), n_diff=order.n_diff(),
    )


# ---------------------------------------------------------------------------
# Float64 reference (tolerance sanity; DEVIATION 670)
# ---------------------------------------------------------------------------


def _zeros64(n: Int) -> List[Float64]:
    var out = List[Float64]()
    for _ in range(n):
        out.append(0.0)
    return out^


def lu_inverse_host64(a_in: List[Float64], n: Int) raises -> List[Float64]:
    var a = a_in.copy()
    var piv = List[Int]()
    for j in range(n):
        var best = j
        var best_mag = abs(a[j + j * n])
        for i in range(j + 1, n):
            if abs(a[i + j * n]) > best_mag:
                best_mag = abs(a[i + j * n])
                best = i
        piv.append(best)
        if best != j:
            for c in range(n):
                var t0 = a[j + c * n]
                a[j + c * n] = a[best + c * n]
                a[best + c * n] = t0
        var pivot = a[j + j * n]
        if pivot == 0.0:
            raise Error("lu_inverse_host64: zero pivot at column " + String(j))
        for i in range(j + 1, n):
            var l = a[i + j * n] / pivot
            a[i + j * n] = l
            for c in range(j + 1, n):
                a[i + c * n] -= l * a[j + c * n]
    var inv = _zeros64(n * n)
    for col in range(n):
        for i in range(n):
            inv[i + col * n] = 1.0 if i == col else 0.0
        for j in range(n):
            var pj = piv[j]
            if pj != j:
                var t0 = inv[j + col * n]
                inv[j + col * n] = inv[pj + col * n]
                inv[pj + col * n] = t0
        for i in range(n):
            var acc = inv[i + col * n]
            for k in range(i):
                acc -= a[i + k * n] * inv[k + col * n]
            inv[i + col * n] = acc
        var i = n - 1
        while i >= 0:
            var acc = inv[i + col * n]
            for k in range(i + 1, n):
                acc -= a[i + k * n] * inv[k + col * n]
            inv[i + col * n] = acc / a[i + i * n]
            i -= 1
    return inv^


@fieldwise_init
struct KalmanHostF64(Movable):
    var P0: List[Float64]
    var loglike: List[Float64]


def kalman_host_f64(
    y: List[Float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    t_params: ARIMAParamsHost,
    fc_steps: Int,
) raises -> KalmanHostF64:
    """The same pipeline in Float64 with plain arithmetic, on the SAME
    Float32 transformed parameters (isolates the filter's precision)."""
    var p = order.p
    var q = order.q
    var P = order.P
    var Q = order.Q
    var s = order.s
    var n_diff = order.n_diff()
    var r = order.r()
    var rd = order.rd()
    var rd2 = rd * rd
    var r2 = r * r
    var P0_out = _zeros64(rd2 * batch_size)
    var ll_out = _zeros64(batch_size)
    for bid in range(batch_size):
        # matrices
        var Zl = _zeros64(rd)
        var Rl = _zeros64(rd)
        var Tl = _zeros64(rd2)
        for i in range(order.d):
            Zl[i] = 1.0
        for i in range(1, order.D + 1):
            Zl[order.d + i * s - 1] = 1.0
        Zl[n_diff] = 1.0
        Rl[n_diff] = 1.0
        for i in range(order.n_theta()):
            var ix = reduced_poly_indices(i + 1, s)
            var c0 = Float64(param_to_poly(False, t_params.ma[bid * q + ix[0] - 1] if (ix[0] != 0 and ix[0] <= q) else Float32(0.0), ix[0], q))
            var c1 = Float64(param_to_poly(False, t_params.sma[bid * Q + ix[1] - 1] if (ix[1] != 0 and ix[1] <= Q) else Float32(0.0), ix[1], Q))
            Rl[n_diff + i + 1] = c0 * c1
        for i in range(order.d):
            for j in range(i, order.d):
                Tl[j * rd + i] = 1.0
        for id_ in range(order.d):
            Tl[n_diff * rd + id_] = 1.0
            for iD in range(1, order.D + 1):
                Tl[(order.d + s * iD - 1) * rd + id_] = 1.0
        for iD in range(order.D):
            var offset = order.d + iD * s
            for i in range(s - 1):
                Tl[(offset + i) * rd + offset + i + 1] = 1.0
            Tl[(offset + s - 1) * rd + offset] = 1.0
            Tl[n_diff * rd + offset] = 1.0
        if order.D == 2:
            Tl[(n_diff - 1) * rd + order.d] = 1.0
        for i in range(order.n_phi()):
            var ix = reduced_poly_indices(i + 1, s)
            var c0 = Float64(param_to_poly(True, t_params.ar[bid * p + ix[0] - 1] if (ix[0] != 0 and ix[0] <= p) else Float32(0.0), ix[0], p))
            var c1 = Float64(param_to_poly(True, t_params.sar[bid * P + ix[1] - 1] if (ix[1] != 0 and ix[1] <= P) else Float32(0.0), ix[1], P))
            Tl[n_diff * (rd + 1) + i] = -(c0 * c1)
        for i in range(r - 1):
            Tl[(n_diff + i + 1) * rd + n_diff + i] = 1.0
        if rd == 2 and p == 2:
            if abs(Tl[1] + 1.0) < 0.01:
                Tl[1] = -0.99
        # state
        var s2 = Float64(t_params.sigma2[bid])
        var RQR64 = _zeros64(rd2)
        for j in range(rd):
            for i in range(rd):
                RQR64[i + j * rd] = Rl[i] * s2 * Rl[j]
        var P64 = _zeros64(rd2)
        for i in range(n_diff):
            P64[(rd + 1) * i] = 1e6
        var imaa = _zeros64(r2 * r2)
        for ia in range(r):
            for ja in range(r):
                var av = -Tl[(ia + n_diff) + (ja + n_diff) * rd]
                for ib in range(r):
                    for jb in range(r):
                        var i_ab = ia * r + ib
                        var j_ab = ja * r + jb
                        var v = av * Tl[(ib + n_diff) + (jb + n_diff) * rd]
                        if i_ab == j_ab:
                            v += 1.0
                        imaa[i_ab + j_ab * r2] = v
        var inv = lu_inverse_host64(imaa, r2)
        var vq = _zeros64(r2)
        for j in range(r):
            for i in range(r):
                vq[i + j * r] = RQR64[(i + n_diff) + (j + n_diff) * rd]
        for i in range(r2):
            var acc = 0.0
            for k in range(r2):
                acc += inv[i + k * r2] * vq[k]
            var ii = i % r
            var jj = i // r
            P64[(ii + n_diff) + (jj + n_diff) * rd] = acc
        for i in range(rd2):
            P0_out[bid * rd2 + i] = P64[i]
        var alpha64 = _zeros64(rd)
        var mu64 = 0.0
        if order.k != 0:
            mu64 = Float64(t_params.mu[bid])
            var imt = _zeros64(r2)
            for j in range(r):
                for i in range(r):
                    var delta = 1.0 if i == j else 0.0
                    imt[i + j * r] = delta - Tl[(i + n_diff) + (j + n_diff) * rd]
            if r == 1 and abs(imt[0]) < 1e-3:
                imt[0] = -1e-3 if imt[0] < 0.0 else 1e-3
            var imt_inv = lu_inverse_host64(imt, r)
            for i in range(r):
                alpha64[i + n_diff] = imt_inv[i] * mu64
        # loop
        var sum_logFs = 0.0
        var ll_s2 = 0.0
        var n_obs_ll = 0
        for it in range(n_obs):
            var pred = 0.0
            if n_diff == 0:
                pred = alpha64[0]
            else:
                for i in range(rd):
                    pred += alpha64[i] * Zl[i]
            var vs_it = Float64(y[bid * n_obs + it]) - pred
            var F = 0.0
            if n_diff == 0:
                F = P64[0]
            else:
                for i in range(rd):
                    for j in range(rd):
                        F += P64[j * rd + i] * Zl[i] * Zl[j]
            if it >= n_diff:
                if F > 0.0:
                    sum_logFs += log(F)
                    ll_s2 += vs_it * vs_it / F
                n_obs_ll += 1
            var TP = _zeros64(rd2)
            for i in range(rd):
                for j in range(rd):
                    var acc = 0.0
                    for k in range(rd):
                        acc += Tl[i + k * rd] * P64[k + j * rd]
                    TP[i + j * rd] = acc
            var K = _zeros64(rd)
            if n_diff == 0:
                for i in range(rd):
                    K[i] = TP[i] / F
            else:
                for i in range(rd):
                    var acc = 0.0
                    for j in range(rd):
                        acc += TP[i + j * rd] * Zl[j]
                    K[i] = acc / F
            var av = _zeros64(rd)
            for i in range(rd):
                var acc = 0.0
                for j in range(rd):
                    acc += Tl[i + j * rd] * alpha64[j]
                av[i] = acc
            for i in range(rd):
                alpha64[i] = av[i] + K[i] * vs_it
            alpha64[n_diff] += mu64
            var L = _zeros64(rd2)
            for i in range(rd2):
                L[i] = Tl[i]
            if n_diff == 0:
                for i in range(rd):
                    L[i] -= K[i]
            else:
                for i in range(rd):
                    for j in range(rd):
                        L[j * rd + i] -= K[i] * Zl[j]
            var newP = _zeros64(rd2)
            for i in range(rd):
                for j in range(rd):
                    var acc = 0.0
                    for k in range(rd):
                        acc += TP[i + k * rd] * L[j + k * rd]
                    newP[i + j * rd] = acc + RQR64[i + j * rd]
            for i in range(rd - 1):
                for j in range(i + 1, rd):
                    var sym = 0.5 * (newP[j * rd + i] + newP[i * rd + j])
                    newP[j * rd + i] = sym
                    newP[i * rd + j] = sym
            for i in range(rd):
                newP[i * rd + i] = abs(newP[i * rd + i])
            P64 = newP^
        var nll = Float64(n_obs_ll)
        ll_out[bid] = -0.5 * (sum_logFs + nll * (ll_s2 / nll + 1.8378770664093453))
    return KalmanHostF64(P0=P0_out^, loglike=ll_out^)


# ---------------------------------------------------------------------------
# predict's host replay (NOT A PORT: the oracle for `in_sample_prediction_
# kernel` + `copy_forecast_kernel`, which the audit of 2026-08-23 found had
# no oracle at all -- `predict` was the one entry point no gate could see)
# ---------------------------------------------------------------------------


def in_sample_prediction_host(
    y: List[Float32],
    pred: List[Float32],
    batch_size: Int,
    n_obs: Int,
    n_obs_kf: Int,
    start: Int,
    predict_ld: Int,
    res_offset: Int,
    p_start: Int,
    p_end: Int,
    dD: Int,
    period1: Int,
    period2: Int,
    mut y_p: List[Float32],
):
    """`in_sample_prediction_kernel` serially, one series at a time. The
    sentinel is the same CONSTANT bit pattern (DEVIATION 676), never a
    computed NaN, so the oracle and the device agree on the payload."""
    var nan_bits = UInt32(0x7FC00000)
    var nan_val = bitcast[DType.float32](nan_bits)
    for bid in range(batch_size):
        for i in range(res_offset - start):
            y_p[bid * predict_ld + i] = nan_val
        for i in range(p_start, p_end):
            var v: Float32
            if dD == 0:
                v = ftz(pred[bid * n_obs + i])
            elif dD == 1:
                var a = ftz(y[bid * n_obs + i - period1])
                var b = ftz(pred[bid * n_obs_kf + i - res_offset])
                v = ftz(a + b)
            else:
                var a = ftz(y[bid * n_obs + i - period1])
                var b = ftz(y[bid * n_obs + i - period2])
                var c = ftz(y[bid * n_obs + i - period1 - period2])
                var pr = ftz(pred[bid * n_obs_kf + i - res_offset])
                var t0 = ftz(a + b)
                var t1 = ftz(t0 - c)
                v = ftz(t1 + pr)
            y_p[bid * predict_ld + i - start] = v


def copy_forecast_host(
    fc: List[Float32], batch_size: Int, num_steps: Int, predict_ld: Int,
    n_obs_minus_start: Int, mut y_p: List[Float32],
):
    """`copy_forecast_kernel` serially."""
    for bid in range(batch_size):
        for i in range(num_steps):
            y_p[bid * predict_ld + n_obs_minus_start + i] = fc[num_steps * bid + i]
