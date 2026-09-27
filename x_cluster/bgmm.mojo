# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BayesianGaussianMixture, full covariance (lane/algos-cluster). Reference:
scikit-learn `sklearn/mixture/_bayesian_mixture.py` (the variational
updates, `_estimate_log_weights`, `_estimate_log_prob`,
`_compute_lower_bound`, `_set_parameters`), `_base.py::fit_predict` (the EM
loop, `abs(change) < tol`, the best of `n_init`) and `_gaussian_mixture.py`
(`_estimate_gaussian_parameters`, `_compute_precision_cholesky`).

THE n-SIZED WORK IS THE DEVICE'S, float32: the Mahalanobis squares against
the upper-triangular precision Cholesky factors (`bodies.gauss_q_cell`), the
row log-sum-exp of the E-step (`bodies.resp_row`, the portable exp and log),
and the M-step moments (`nk_cell`, `xk_cell`, `cov_cell`), every fold over the
rows ascending. THE k-SIZED WORK IS HOST FLOAT64, one source in both
bindings: the Wishart and Dirichlet(-process) updates, the d x d Cholesky and
its triangular inverse, digamma and log-gamma (series on the portable
`identical_log64`), the lower bound. Every host product that feeds an add is
`identical_mul64`. The k-means start is this library's KMeans through
`ClusterOps.kmeans`. Only covariance_type='full' (NOT_IMPLEMENTED.tsv)."""
from std.math import sqrt

from checks.numerics import identical_log64, identical_mul64
from cluster.impl.kmeans_params import INIT_KMEANS_PLUS_PLUS
from x_cluster.bodies import SplitMix64
from x_cluster.ops import ClusterOps

comptime LOG2 = 0.6931471805599453
comptime LOG_2PI = 1.8378770664093453


def _ln(x: Float64) -> Float64:
    return identical_log64(x)


def digamma64(x_in: Float64) -> Float64:
    """psi(x), x > 0: the recurrence up to x >= 6, then the asymptotic series."""
    var x = x_in
    var acc = Float64(0)
    while x < 6.0:
        acc = acc - 1.0 / x
        x = x + 1.0
    var inv = 1.0 / x
    var inv2 = identical_mul64(inv, inv)
    var series = identical_mul64(
        inv2,
        1.0 / 12.0 - identical_mul64(inv2, 1.0 / 120.0 - identical_mul64(inv2, 1.0 / 252.0 - identical_mul64(inv2, 1.0 / 240.0 - identical_mul64(inv2, 1.0 / 132.0)))),
    )
    return acc + _ln(x) - identical_mul64(0.5, inv) - series


def lgamma64(x_in: Float64) -> Float64:
    """log Gamma(x), x > 0: shift to x >= 7 (one product, one log), then Stirling."""
    var x = x_in
    var prod = Float64(1)
    while x < 7.0:
        prod = identical_mul64(prod, x)
        x = x + 1.0
    var inv = 1.0 / x
    var inv2 = identical_mul64(inv, inv)
    var series = identical_mul64(
        inv,
        1.0 / 12.0 - identical_mul64(inv2, 1.0 / 360.0 - identical_mul64(inv2, 1.0 / 1260.0 - identical_mul64(inv2, 1.0 / 1680.0))),
    )
    return identical_mul64(x - 0.5, _ln(x)) - x + 0.5 * LOG_2PI + series - _ln(prod)


struct BgmmState(Copyable, Movable):
    var nk: List[Float64]
    var wc0: List[Float64]
    var wc1: List[Float64]
    var mean_prec: List[Float64]
    var means: List[Float64]
    var dof: List[Float64]
    var cov: List[Float64]
    var pchol: List[Float64]

    def __init__(out self):
        self.nk = List[Float64]()
        self.wc0 = List[Float64]()
        self.wc1 = List[Float64]()
        self.mean_prec = List[Float64]()
        self.means = List[Float64]()
        self.dof = List[Float64]()
        self.cov = List[Float64]()
        self.pchol = List[Float64]()


@fieldwise_init
struct BgmmPriors(Copyable, Movable):
    var kc: Int
    var d: Int
    var dp: Bool
    var wcp: Float64
    var mpp: Float64
    var mean_prior: List[Float64]
    var dofp: Float64
    var cov_prior: List[Float64]


def _precision_cholesky(cov: List[Float64], kc: Int, d: Int) raises -> List[Float64]:
    """Per component: L = cholesky(cov) (lower), P = inv(L)^T (upper)."""
    var out = List[Float64](length=kc * d * d, fill=0)
    for k in range(kc):
        var o = k * d * d
        var l = List[Float64](length=d * d, fill=0)
        for j in range(d):
            var s = cov[o + j * d + j]
            for q in range(j):
                s = s - identical_mul64(l[j * d + q], l[j * d + q])
            if not (s > 0):
                raise Error("Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number of components, increase reg_covar, or scale the input data.")
            var ljj = sqrt(s)
            l[j * d + j] = ljj
            for i in range(j + 1, d):
                var t = cov[o + i * d + j]
                for q in range(j):
                    t = t - identical_mul64(l[i * d + q], l[j * d + q])
                l[i * d + j] = t / ljj
        # Y = inv(L) by forward substitution, column by column; P = Y^T
        for c in range(d):
            for i in range(c, d):
                var t = Float64(1) if i == c else Float64(0)
                for q in range(c, i):
                    t = t - identical_mul64(l[i * d + q], out[o + c * d + q])
                out[o + c * d + i] = t / l[i * d + i]
    return out^


def _m_step_host(
    pr: BgmmPriors, reg_nk: List[Float32], xk32: List[Float32], sk32: List[Float32], mut st: BgmmState
) raises:
    var kc = pr.kc
    var d = pr.d
    st.nk = List[Float64](capacity=kc)
    for k in range(kc):
        st.nk.append(Float64(reg_nk[k]))
    # weights
    st.wc0 = List[Float64](capacity=kc)
    st.wc1 = List[Float64](capacity=kc)
    if pr.dp:
        var tail = List[Float64](length=kc, fill=0)
        var acc = Float64(0)
        for q in range(kc):
            var k = kc - 1 - q
            tail[k] = acc
            acc = acc + st.nk[k]
        for k in range(kc):
            st.wc0.append(1.0 + st.nk[k])
            st.wc1.append(pr.wcp + tail[k])
    else:
        for k in range(kc):
            st.wc0.append(pr.wcp + st.nk[k])
    # means
    st.mean_prec = List[Float64](capacity=kc)
    st.means = List[Float64](capacity=kc * d)
    for k in range(kc):
        var mp = pr.mpp + st.nk[k]
        st.mean_prec.append(mp)
        for a in range(d):
            st.means.append(
                (identical_mul64(pr.mpp, pr.mean_prior[a]) + identical_mul64(st.nk[k], Float64(xk32[k * d + a]))) / mp
            )
    # Wishart, full
    st.dof = List[Float64](capacity=kc)
    st.cov = List[Float64](length=kc * d * d, fill=0)
    for k in range(kc):
        var dof = pr.dofp + st.nk[k]
        st.dof.append(dof)
        var coef = identical_mul64(st.nk[k], pr.mpp) / st.mean_prec[k]
        for a in range(d):
            var da = Float64(xk32[k * d + a]) - pr.mean_prior[a]
            for b in range(d):
                var db = Float64(xk32[k * d + b]) - pr.mean_prior[b]
                var v = pr.cov_prior[a * d + b] + identical_mul64(st.nk[k], Float64(sk32[k * d * d + a * d + b]))
                v = v + identical_mul64(coef, identical_mul64(da, db))
                st.cov[k * d * d + a * d + b] = v / dof
    st.pchol = _precision_cholesky(st.cov, kc, d)


def _log_weights(pr: BgmmPriors, st: BgmmState) -> List[Float64]:
    var kc = pr.kc
    var out = List[Float64](capacity=kc)
    if pr.dp:
        var run = Float64(0)
        for k in range(kc):
            var ds = digamma64(st.wc0[k] + st.wc1[k])
            out.append(digamma64(st.wc0[k]) - ds + run)
            run = run + (digamma64(st.wc1[k]) - ds)
    else:
        var tot = Float64(0)
        for k in range(kc):
            tot = tot + st.wc0[k]
        var dt = digamma64(tot)
        for k in range(kc):
            out.append(digamma64(st.wc0[k]) - dt)
    return out^


def _log_det_pchol(st: BgmmState, kc: Int, d: Int) -> List[Float64]:
    var out = List[Float64](capacity=kc)
    for k in range(kc):
        var acc = Float64(0)
        for j in range(d):
            acc = acc + _ln(st.pchol[k * d * d + j * d + j])
        out.append(acc)
    return out^


def bgmm_constants(pr: BgmmPriors, st: BgmmState) -> List[Float32]:
    """c_k with wlp[i, k] = c_k - q_ik / 2 (`_estimate_weighted_log_prob`)."""
    var kc = pr.kc
    var d = pr.d
    var fd = Float64(d)
    var lw = _log_weights(pr, st)
    var ld = _log_det_pchol(st, kc, d)
    var out = List[Float32](capacity=kc)
    for k in range(kc):
        var log_lambda = identical_mul64(fd, LOG2)
        for j in range(d):
            log_lambda = log_lambda + digamma64(identical_mul64(0.5, st.dof[k] - Float64(j)))
        var c = identical_mul64(-0.5, identical_mul64(fd, LOG_2PI)) + ld[k]
        c = c - identical_mul64(identical_mul64(0.5, fd), _ln(st.dof[k]))
        c = c + identical_mul64(0.5, log_lambda - fd / st.mean_prec[k])
        c = c + lw[k]
        out.append(Float32(c))
    return out^


def _lower_bound(pr: BgmmPriors, st: BgmmState, resp: List[Float32], lr: List[Float32], n: Int) -> Float64:
    var kc = pr.kc
    var d = pr.d
    var fd = Float64(d)
    var ent = Float64(0)
    for t in range(n * kc):
        ent = ent + identical_mul64(Float64(resp[t]), Float64(lr[t]))
    var ld = _log_det_pchol(st, kc, d)
    var log_wishart = Float64(0)
    for k in range(kc):
        var ldpc = ld[k] - identical_mul64(identical_mul64(0.5, fd), _ln(st.dof[k]))
        var g = Float64(0)
        for j in range(d):
            g = g + lgamma64(identical_mul64(0.5, st.dof[k] - Float64(j)))
        var term = identical_mul64(st.dof[k], ldpc) + identical_mul64(identical_mul64(st.dof[k], fd), identical_mul64(0.5, LOG2))
        log_wishart = log_wishart - (term + g)
    var log_norm_weight = Float64(0)
    if pr.dp:
        for k in range(kc):
            var betaln = lgamma64(st.wc0[k]) + lgamma64(st.wc1[k]) - lgamma64(st.wc0[k] + st.wc1[k])
            log_norm_weight = log_norm_weight - betaln
    else:
        var tot = Float64(0)
        var sg = Float64(0)
        for k in range(kc):
            tot = tot + st.wc0[k]
            sg = sg + lgamma64(st.wc0[k])
        log_norm_weight = lgamma64(tot) - sg
    var slog = Float64(0)
    for k in range(kc):
        slog = slog + _ln(st.mean_prec[k])
    return -ent - log_wishart - log_norm_weight - identical_mul64(identical_mul64(0.5, fd), slog)


def bgmm_weights(pr: BgmmPriors, st: BgmmState) -> List[Float64]:
    var kc = pr.kc
    var w = List[Float64](capacity=kc)
    var tot = Float64(0)
    if pr.dp:
        var run = Float64(1)
        for k in range(kc):
            var s = st.wc0[k] + st.wc1[k]
            w.append(identical_mul64(st.wc0[k] / s, run))
            run = identical_mul64(run, st.wc1[k] / s)
    else:
        for k in range(kc):
            w.append(st.wc0[k])
    for k in range(kc):
        tot = tot + w[k]
    for k in range(kc):
        w[k] = w[k] / tot
    return w^


def _f32(v: List[Float64]) -> List[Float32]:
    var out = List[Float32](capacity=len(v))
    for t in v:
        out.append(Float32(t))
    return out^


@fieldwise_init
struct BgmmFit(Copyable, Movable):
    var lower_bound: Float64
    var n_iter: Int
    var converged: Bool


def bgmm_fit[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, pr: BgmmPriors, reg: Float32, tol: Float64,
    max_iter: Int, n_init: Int, init_random: Bool, seed: UInt64,
    mut best: BgmmState, mut labels: List[Int32],
) raises -> BgmmFit:
    var kc = pr.kc
    if kc < 1 or kc > n:
        raise Error("Expected n_samples >= n_components but got n_components = " + String(kc) + ", n_samples = " + String(n))
    var rng = SplitMix64(seed)
    var xs = ops.put(x)
    var rs = ops.zeros(n * kc)
    var qs = ops.zeros(n * kc)
    var lpn = ops.zeros(n)
    var nks = ops.zeros(kc)
    var xks = ops.zeros(kc * d)
    var sks = ops.zeros(kc * d * d)
    var ms = ops.zeros(kc * d)
    var ps = ops.zeros(kc * d * d)
    var cs = ops.zeros(kc)
    var max_lb = Float64(0)
    var have_best = False
    var best_iter = 0
    var converged_best = False
    for _init in range(n_init):
        # the start: one-hot k-means labels or normalized uniforms
        var resp0 = List[Float32](length=n * kc, fill=Float32(0))
        if init_random:
            for i in range(n):
                var row = List[Float64](capacity=kc)
                var s = Float64(0)
                for _k in range(kc):
                    var u = rng.unit()
                    row.append(u)
                    s = s + u
                for k in range(kc):
                    resp0[i * kc + k] = Float32(row[k] / s)
        else:
            var c = List[Float32]()
            var l = List[Int32]()
            _ = ops.kmeans(x, n, d, kc, 300, 1e-4, rng.next() >> 1, 1, INIT_KMEANS_PLUS_PLUS, c, l)
            for i in range(n):
                resp0[i * kc + Int(l[i])] = Float32(1)
        ops.set(rs, resp0)
        ops.moments(rs, xs, n, d, kc, reg, nks, xks, sks)
        var st = BgmmState()
        _m_step_host(pr, ops.get(nks, kc), ops.get(xks, kc * d), ops.get(sks, kc * d * d), st)
        var lb = Float64(0)
        var have_lb = False
        var converged = False
        var n_iter = 0
        for it in range(1, max_iter + 1):
            n_iter = it
            var prev = lb
            # E-step
            ops.set(ms, _f32(st.means))
            ops.set(ps, _f32(st.pchol))
            ops.set(cs, bgmm_constants(pr, st))
            ops.gauss_q(xs, n, d, ms, ps, kc, qs)
            ops.resp(qs, cs, n, kc, lpn)
            ops.exp(qs, rs, n * kc)
            # M-step
            ops.moments(rs, xs, n, d, kc, reg, nks, xks, sks)
            _m_step_host(pr, ops.get(nks, kc), ops.get(xks, kc * d), ops.get(sks, kc * d * d), st)
            lb = _lower_bound(pr, st, ops.get(rs, n * kc), ops.get(qs, n * kc), n)
            if have_lb:
                var change = lb - prev
                if abs(change) < tol:
                    converged = True
                    break
            have_lb = True
        if not have_best or lb > max_lb:
            max_lb = lb
            have_best = True
            best = st.copy()
            best_iter = n_iter
            converged_best = converged
    # the final E-step on the best parameters: the labels
    ops.set(ms, _f32(best.means))
    ops.set(ps, _f32(best.pchol))
    ops.set(cs, bgmm_constants(pr, best))
    ops.gauss_q(xs, n, d, ms, ps, kc, qs)
    ops.resp(qs, cs, n, kc, lpn)
    var lr = ops.get(qs, n * kc)
    labels = List[Int32](capacity=n)
    for i in range(n):
        var bk = 0
        for k in range(1, kc):
            if lr[i * kc + k] > lr[i * kc + bk]:
                bk = k
        labels.append(Int32(bk))
    return BgmmFit(max_lb, best_iter, converged_best)


def bgmm_score[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, kc: Int, means: List[Float32], pchol: List[Float32],
    c: List[Float32], mut log_resp: List[Float32], mut lpn_out: List[Float32],
) raises:
    var xs = ops.put(x)
    var ms = ops.put(means)
    var ps = ops.put(pchol)
    var cs = ops.put(c)
    var qs = ops.zeros(n * kc)
    var lpn = ops.zeros(n)
    ops.gauss_q(xs, n, d, ms, ps, kc, qs)
    ops.resp(qs, cs, n, kc, lpn)
    log_resp = ops.get(qs, n * kc)
    lpn_out = ops.get(lpn, n)
