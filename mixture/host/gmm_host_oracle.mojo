# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""`GaussianMixture` on the HOST: `mixture/estimator.mojo`'s one-shot fit and
scoring entries restated without a device (CPU training for the workstream D
estimators, 2026-09-15).

WHAT THIS IS. `gaussian_mixture_fit` and the four scoring entries run their
arithmetic in `mixture/checks/estep.mojo` and `mstep.mojo` device kernels.
This file is a SECOND spelling of those kernels and of the EM driver, in the
driver's order, with no `DeviceContext`, no card and no launch geometry.
Every kernel it restates is one thread per output cell, row or component with
its fold inside that thread, so the host loop over the same index reproduces
it:

    the initial responsibilities
                        estimator.mojo::gmm_initial_resp: the k-means one-hot
                        through cluster/host/kmeans_oracle.mojo::
                        host_kmeans_fit (the core host binding's k-means, at
                        kmeans_fit's defaults: k-means||, L2 expanded, 300
                        iterations, tol 1e-4, one init), or the
                        position-mapped Philox uniforms (DEVIATION 1733),
                        which are host code on the device path too
    the log of them     estimator.mojo::_safe_log
    resp                mstep.mojo::resp_exp_kernel
    nk                  nk_kernel, rows ascending, plus 10 eps
    the denominator     Float32(n) at initialization, nk_total_kernel after
    weights             weights_kernel, identical_div and identical_log
    means               identical_gemm_into at OP_TN, then means_divide_kernel
    covariances         center_scale_kernel, OP_TN, cov_finish_kernel
    precision Cholesky  gmm_precision_cholesky: add_jitter at +0.0,
                        potrf_lower (cholesky/host/chol_oracle.mojo::
                        chol_host_factor_lower, no validation, as on the
                        device), chol_logdet times -0.5, trsm_lower against
                        the identity, the transpose
    the E-step          estep.mojo::gmm_e_step: X . P and mu . P at OP_NN,
                        mahal_kernel, weighted_log_prob_kernel,
                        logsumexp_kernel (the positional row max and
                        DEVIATION 1727's all -inf row), log_resp_kernel,
                        meanll_kernel
    convergence         gmm_convergence_change (DEVIATION 1747) and
                        gmm_converged, on the host on both paths
    predict             argmax_kernel over the weighted log probabilities
    score, bic, aic     estimator.mojo's host folds

WHAT IS REFUSED, BY NAME, AS ON THE DEVICE: covariance types other than
'full', init methods other than 'kmeans' and 'random', n_components below 1
or above n_samples, a negative max_iter, a NaN, negative or infinite tol or
reg_covar, non-finite X, and a collapsed component (DEVIATION 1723).

THE SABOTAGE. No arm of its own. Under `-D MOJOLEARN_HOST_SABOTAGE=1` every
GEMM leaf walks descending (`gemm/host/gemm_oracle.mojo`), which moves the
means, the covariances, every Cholesky trailing update and the E-step
products, and the k-means host oracle's quantized accumulation moves the
initialization.
"""

from std.memory import bitcast

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
    identical_mul_add,
)
from cholesky.host.chol_oracle import chol_host_factor_lower, chol_host_trsm_lower
from cluster.host.kmeans_oracle import (
    INIT_KMEANS_PLUS_PLUS,
    METRIC_L2_EXPANDED,
    host_kmeans_fit,
)
from core.philox import philox4x32_10
from gemm.host.identical_gemm import OP_NN, OP_TN, gemm_oracle

comptime GMMH_COV_FULL = 0
comptime GMMH_INIT_KMEANS = 0
comptime GMMH_INIT_RANDOM = 1

#: `estep.mojo`'s constants, by their bits.
comptime GMMH_LOG_2PI_BITS: UInt32 = 0x3FEB3F8E
comptime GMMH_TEN_EPS_BITS: UInt32 = 0x35A00000
comptime GMMH_NEG_INF_BITS: UInt32 = 0xFF800000
comptime GMMH_POS_INF_BITS: UInt32 = 0x7F800000
#: `estimator.mojo::GMM_TWO_POW_M24_BITS`.
comptime GMMH_TWO_POW_M24_BITS: UInt32 = 0x33800000


def _neg_inf() -> Float32:
    return bitcast[DType.float32](GMMH_NEG_INF_BITS)


def _pos_inf() -> Float32:
    return bitcast[DType.float32](GMMH_POS_INF_BITS)


def gmmh_hex32(v: Float32) -> String:
    """`estimator.mojo::gmm_hex32`."""
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def gmmh_covariance_type_from_name(name: String) raises -> Int:
    """`covariance_type_from_name`: 'full', or a refusal by name."""
    if name == "full":
        return GMMH_COV_FULL
    if name == "tied" or name == "diag" or name == "spherical":
        raise Error(
            "GaussianMixture: covariance_type='"
            + name
            + "' is NOT IMPLEMENTED. Only 'full' is implemented here;"
            " mixture/NOT_IMPLEMENTED.tsv carries it"
        )
    raise Error(
        "GaussianMixture: covariance_type='"
        + name
        + "' is not one of scikit-learn's four ('full', 'tied', 'diag',"
        " 'spherical'). Only 'full' is implemented here"
    )


def gmmh_init_params_from_name(name: String) raises -> Int:
    """`init_params_from_name`: 'kmeans' or 'random', or a refusal by name."""
    if name == "kmeans":
        return GMMH_INIT_KMEANS
    if name == "random":
        return GMMH_INIT_RANDOM
    if name == "k-means++" or name == "random_from_data":
        raise Error(
            "GaussianMixture: init_params='"
            + name
            + "' is NOT IMPLEMENTED. 'kmeans' and 'random' are implemented"
            " here; mixture/NOT_IMPLEMENTED.tsv carries it"
        )
    raise Error(
        "GaussianMixture: init_params='"
        + name
        + "' is not one of scikit-learn's four ('kmeans', 'k-means++',"
        " 'random', 'random_from_data'). 'kmeans' and 'random' are"
        " implemented here"
    )


@fieldwise_init
struct GmmHostParams(Copyable, ImplicitlyCopyable, Movable):
    """`GmmParams`, field for field."""

    var n_components: Int
    var covariance_type: Int
    var tol: Float32
    var reg_covar: Float32
    var max_iter: Int
    var init_params: Int
    var random_state: UInt64


@fieldwise_init
struct GmmHostModel(Movable):
    """`GaussianMixtureModel`, field for field."""

    var n_components: Int
    var n_features: Int
    var weights: List[Float32]
    var means: List[Float32]
    var covariances: List[Float32]
    var precisions_cholesky: List[Float32]
    var log_det_chol: List[Float32]
    var n_iter: Int
    var converged: Bool
    var lower_bound: Float32


def gmmh_validate_params(params: GmmHostParams, n_samples: Int) raises:
    """`gmm_validate_params`, in its order and words."""
    if params.covariance_type != GMMH_COV_FULL:
        raise Error(
            "GaussianMixture: only covariance_type='full' is implemented"
            " (got type id "
            + String(params.covariance_type)
            + ")"
        )
    if (
        params.init_params != GMMH_INIT_KMEANS
        and params.init_params != GMMH_INIT_RANDOM
    ):
        raise Error(
            "GaussianMixture: init_params id "
            + String(params.init_params)
            + " is not implemented; only 'kmeans' and 'random' are"
        )
    if params.n_components < 1:
        raise Error(
            "GaussianMixture: n_components must be at least 1, got "
            + String(params.n_components)
        )
    if params.n_components > n_samples:
        raise Error(
            "GaussianMixture: n_components="
            + String(params.n_components)
            + " exceeds n_samples="
            + String(n_samples)
            + ". Every component would need at least one point to have a"
            " covariance at all, so the fit is guaranteed to collapse"
            " (DEVIATION 1723)"
        )
    if params.max_iter < 0:
        raise Error(
            "GaussianMixture: max_iter must be at least 0, got "
            + String(params.max_iter)
        )
    if params.tol != params.tol:
        raise Error("GaussianMixture: tol is NaN; refused by name")
    if params.tol < Float32(0.0):
        raise Error(
            "GaussianMixture: tol must be non-negative, got "
            + gmmh_hex32(params.tol)
        )
    if params.reg_covar != params.reg_covar:
        raise Error("GaussianMixture: reg_covar is NaN; refused by name")
    if params.reg_covar < Float32(0.0):
        raise Error(
            "GaussianMixture: reg_covar must be non-negative, got "
            + gmmh_hex32(params.reg_covar)
        )
    var big = _pos_inf()
    if params.reg_covar == big:
        raise Error("GaussianMixture: reg_covar is +inf; refused by name")
    if params.tol == big:
        raise Error("GaussianMixture: tol is +inf; refused by name")


def gmmh_validate_data(x: List[Float32], n_samples: Int, n_features: Int) raises:
    """`gmm_validate_data`, in its order and words."""
    if n_samples < 1:
        raise Error(
            "GaussianMixture: X must have at least one row, got "
            + String(n_samples)
        )
    if n_features < 1:
        raise Error(
            "GaussianMixture: X must have at least one column, got "
            + String(n_features)
        )
    if len(x) != n_samples * n_features:
        raise Error(
            "GaussianMixture: X holds "
            + String(len(x))
            + " floats, "
            + String(n_samples)
            + " x "
            + String(n_features)
            + " needs "
            + String(n_samples * n_features)
        )
    for i in range(len(x)):
        var v = x[i]
        if v != v:
            raise Error(
                "GaussianMixture: X contains NaN at row "
                + String(i // n_features)
                + " column "
                + String(i % n_features)
                + "; refused by name before any upload (DEVIATION 1738)"
            )
        var u = bitcast[DType.uint32](v) & UInt32(0x7FFFFFFF)
        if u == UInt32(0x7F800000):
            raise Error(
                "GaussianMixture: X contains an infinity at row "
                + String(i // n_features)
                + " column "
                + String(i % n_features)
                + " ("
                + gmmh_hex32(v)
                + "); refused by name before any upload (DEVIATION 1738)"
            )


def _safe_log(v: Float32) -> Float32:
    """`estimator.mojo::_safe_log`."""
    if v == Float32(0.0):
        return _neg_inf()
    return ftz(identical_log(ftz(v)))


def gmmh_initial_resp(
    x: List[Float32], n: Int, d: Int, params: GmmHostParams
) raises -> List[Float32]:
    """`gmm_initial_resp`: the one-hot k-means labels or the normalized
    position-mapped uniforms."""
    var ncomp = params.n_components
    var resp = List[Float32]()
    if params.init_params == GMMH_INIT_KMEANS:
        var centroids = List[Float32](length=ncomp * d, fill=Float32(0.0))
        var labels = List[UInt32](length=n, fill=UInt32(0))
        var no_weights = List[Float32]()
        var r = host_kmeans_fit(
            x, n, d, ncomp, centroids, labels, no_weights, 0,
            300, Float64(1.0e-4), params.random_state, 1,
            INIT_KMEANS_PLUS_PLUS, METRIC_L2_EXPANDED,
        )
        _ = r
        for i in range(n):
            var lab = Int(labels[i])
            for k in range(ncomp):
                resp.append(Float32(1.0) if k == lab else Float32(0.0))
        return resp^

    var key = SIMD[DType.uint32, 2](
        UInt32(params.random_state & 0xFFFFFFFF),
        UInt32((params.random_state >> 32) & 0xFFFFFFFF),
    )
    for i in range(n):
        var row = List[Float32]()
        var s = Float32(0.0)
        for k in range(ncomp):
            var ctr = SIMD[DType.uint32, 4](
                UInt32(i & 0xFFFFFFFF),
                UInt32((i >> 32) & 0xFFFFFFFF),
                UInt32(k),
                UInt32(0),
            )
            var draw = philox4x32_10(ctr, key)
            var u = Float32(Int(draw[0] >> UInt32(8))) * bitcast[
                DType.float32
            ](GMMH_TWO_POW_M24_BITS)
            row.append(u)
            s = ftz(s + u)
        if not (s > Float32(0.0)):
            raise Error(
                "GaussianMixture: init_params='random' produced a row of"
                " responsibilities summing to "
                + gmmh_hex32(s)
                + " at row "
                + String(i)
                + ", which has no normalizer; refused by name"
            )
        for k in range(ncomp):
            resp.append(ftz(row[k] / s))
    return resp^


@fieldwise_init
struct GmmHostMStep(Movable):
    var weights: List[Float32]
    var log_weights: List[Float32]
    var means: List[Float32]
    var covariances: List[Float32]


def gmmh_m_step(
    x: List[Float32],
    logresp: List[Float32],
    n: Int,
    d: Int,
    ncomp: Int,
    reg_covar: Float32,
    divide_weights_by_n: Bool,
) -> GmmHostMStep:
    """`gmm_m_step` at GMM_SAB_NONE."""
    var dd = d * d
    var resp = List[Float32](length=n * ncomp, fill=Float32(0.0))
    for idx in range(n * ncomp):
        resp[idx] = ftz(identical_exp(ftz(logresp[idx])))

    var ten_eps = bitcast[DType.float32](GMMH_TEN_EPS_BITS)
    var nk = List[Float32](length=ncomp, fill=Float32(0.0))
    for k in range(ncomp):
        var acc = Float32(0.0)
        for i in range(n):
            acc = ftz(acc + ftz(resp[i * ncomp + k]))
        nk[k] = ftz(acc + ten_eps)

    var denom: Float32
    if divide_weights_by_n:
        denom = Float32(n)
    else:
        var acc = Float32(0.0)
        for k in range(ncomp):
            acc = ftz(acc + ftz(nk[k]))
        denom = acc

    var weights = List[Float32](length=ncomp, fill=Float32(0.0))
    var log_weights = List[Float32](length=ncomp, fill=Float32(0.0))
    for k in range(ncomp):
        var w = ftz(identical_div(ftz(nk[k]), ftz(denom)))
        weights[k] = w
        log_weights[k] = ftz(identical_log(w))

    var raw = gemm_oracle(resp, x, OP_TN, ncomp, d, n)
    var means = List[Float32](length=ncomp * d, fill=Float32(0.0))
    for idx in range(ncomp * d):
        var k = idx // d
        means[idx] = ftz(identical_div(ftz(raw[idx]), ftz(nk[k])))

    var cov = List[Float32](length=ncomp * dd, fill=Float32(0.0))
    var diff = List[Float32](length=n * d, fill=Float32(0.0))
    var scaled = List[Float32](length=n * d, fill=Float32(0.0))
    for kc in range(ncomp):
        for idx in range(n * d):
            var i = idx // d
            var j = idx % d
            var dv = ftz(ftz(x[idx]) - ftz(means[kc * d + j]))
            diff[idx] = dv
            var r = ftz(resp[i * ncomp + kc])
            scaled[idx] = ftz(identical_mul(r, dv))
        var rc = gemm_oracle(scaled, diff, OP_TN, d, d, n)
        for idx in range(dd):
            var a = idx // d
            var b = idx % d
            var v = ftz(rc[idx])
            v = ftz(identical_div(v, ftz(nk[kc])))
            if a == b:
                v = ftz(v + reg_covar)
            cov[kc * dd + idx] = v
    return GmmHostMStep(weights^, log_weights^, means^, cov^)


@fieldwise_init
struct GmmHostPrecision(Movable):
    var info: Int
    var failed_component: Int
    var precisions: List[Float32]
    var log_det_chol: List[Float32]


def gmmh_precision_cholesky(
    cov: List[Float32], d: Int, ncomp: Int
) raises -> GmmHostPrecision:
    """`gmm_precision_cholesky` at GMM_SAB_NONE: components ascending, the
    first failure stops the step."""
    var dd = d * d
    var prec = List[Float32](length=ncomp * dd, fill=Float32(0.0))
    var logdet = List[Float32](length=ncomp, fill=Float32(0.0))
    for kc in range(ncomp):
        var work = List[Float32](capacity=dd)
        for i in range(dd):
            work.append(cov[kc * dd + i])
        var f = chol_host_factor_lower(work, d, Float32(0.0))
        if f.info != 0:
            return GmmHostPrecision(f.info, kc, prec^, logdet^)
        logdet[kc] = ftz(identical_mul(Float32(-0.5), f.logdet))
        var ident = List[Float32](length=dd, fill=Float32(0.0))
        for i in range(d):
            ident[i * d + i] = Float32(1.0)
        chol_host_trsm_lower(f.l, ident, d, d)
        for i in range(d):
            for j in range(d):
                prec[kc * dd + i * d + j] = ident[j * d + i]
    return GmmHostPrecision(0, -1, prec^, logdet^)


@fieldwise_init
struct GmmHostEStep(Movable):
    var wlp: List[Float32]
    var lse: List[Float32]
    var logresp: List[Float32]
    var meanll: Float32


def gmmh_e_step(
    x: List[Float32],
    means: List[Float32],
    prec: List[Float32],
    log_det_chol: List[Float32],
    log_weights: List[Float32],
    n: Int,
    d: Int,
    ncomp: Int,
    output_level: Int = 3,
) -> GmmHostEStep:
    """`gmm_e_step` at GMM_SAB_NONE.

    ``output_level`` stops after weighted log probability (0), logsumexp
    (1), or log responsibilities (2).  Training's default 3 also computes
    mean likelihood.  This changes no arithmetic in any requested output;
    the host scoring APIs no longer construct values they discard.
    """
    var dd = d * d
    var mahal = List[Float32](length=n * ncomp, fill=Float32(0.0))
    for kc in range(ncomp):
        var pk = List[Float32](capacity=dd)
        for i in range(dd):
            pk.append(prec[kc * dd + i])
        var muk = List[Float32](capacity=d)
        for j in range(d):
            muk.append(means[kc * d + j])
        var y = gemm_oracle(x, pk, OP_NN, n, d, d)
        var murow = gemm_oracle(muk, pk, OP_NN, 1, d, d)
        for i in range(n):
            var acc = Float32(0.0)
            for j in range(d):
                var t = ftz(ftz(y[i * d + j]) - ftz(murow[j]))
                acc = ftz(identical_mul_add(t, t, acc))
            mahal[i * ncomp + kc] = acc

    var d_log_2pi = ftz(
        identical_mul(Float32(d), bitcast[DType.float32](GMMH_LOG_2PI_BITS))
    )
    var wlp = List[Float32](length=n * ncomp, fill=Float32(0.0))
    for idx in range(n * ncomp):
        var k = idx % ncomp
        var m = ftz(mahal[idx])
        var inner = ftz(d_log_2pi + m)
        var half = ftz(identical_mul(Float32(-0.5), inner))
        var lp = ftz(half + ftz(log_det_chol[k]))
        wlp[idx] = ftz(lp + ftz(log_weights[k]))

    if output_level == 0:
        return GmmHostEStep(
            wlp^, List[Float32](), List[Float32](), Float32(0.0)
        )

    var lse = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        var base = i * ncomp
        var max_exp = wlp[base]
        for k in range(1, ncomp):
            var v = wlp[base + k]
            if v > max_exp:
                max_exp = v
        if max_exp == _neg_inf():
            lse[i] = max_exp
            continue
        var s = Float32(0.0)
        for k in range(ncomp):
            s = ftz(s + ftz(identical_exp(ftz(wlp[base + k] - max_exp))))
        lse[i] = ftz(identical_log(s) + max_exp)

    if output_level == 1:
        return GmmHostEStep(wlp^, lse^, List[Float32](), Float32(0.0))

    var logresp = List[Float32](length=n * ncomp, fill=Float32(0.0))
    for idx in range(n * ncomp):
        var i = idx // ncomp
        logresp[idx] = ftz(ftz(wlp[idx]) - ftz(lse[i]))

    if output_level == 2:
        return GmmHostEStep(wlp^, lse^, logresp^, Float32(0.0))

    var acc = Float32(0.0)
    for i in range(n):
        acc = ftz(acc + ftz(lse[i]))
    var meanll = ftz(identical_div(acc, Float32(n)))
    return GmmHostEStep(wlp^, lse^, logresp^, meanll)


def _collapse_message(it: Int, info: Int, comp: Int, reg_covar: Float32) -> String:
    """`estimator.mojo::_collapse_message`."""
    var where = String("initialization")
    if it > 0:
        where = String("EM iteration ") + String(it)
    return (
        "GaussianMixture: fitting the mixture model failed because some"
        " components have ill-defined empirical covariance (for instance"
        " caused by singleton or collapsed samples). Try to decrease the"
        " number of components, increase reg_covar, or scale the input"
        " data. FAILED AT "
        + where
        + ", COMPONENT "
        + String(comp)
        + ", LAPACK info="
        + String(info)
        + ". reg_covar was "
        + gmmh_hex32(reg_covar)
        + ". THE COMPONENT IS NOT RESET AND THE FIT IS NOT CONTINUED"
        " (DEVIATION 1723)"
    )


def gmmh_fit(
    x: List[Float32], n: Int, d: Int, params: GmmHostParams
) raises -> GmmHostModel:
    """`gaussian_mixture_fit`, in scikit-learn's order: validate, initialize
    (resp, moments, precision Cholesky), then EM until the change is below
    tol or max_iter."""
    gmmh_validate_data(x, n, d)
    gmmh_validate_params(params, n)
    var ncomp = params.n_components

    var resp0 = gmmh_initial_resp(x, n, d, params)
    var loginit = List[Float32](capacity=n * ncomp)
    for i in range(n * ncomp):
        loginit.append(_safe_log(resp0[i]))

    var ms = gmmh_m_step(x, loginit, n, d, ncomp, params.reg_covar, True)
    var pre = gmmh_precision_cholesky(ms.covariances, d, ncomp)
    if pre.info != 0:
        raise Error(
            _collapse_message(0, pre.info, pre.failed_component, params.reg_covar)
        )

    var lower_bound = _neg_inf()
    var n_iter = 0
    var converged = False
    for it in range(1, params.max_iter + 1):
        var prev = lower_bound
        var es = gmmh_e_step(
            x, ms.means, pre.precisions, pre.log_det_chol, ms.log_weights,
            n, d, ncomp,
        )
        ms = gmmh_m_step(x, es.logresp, n, d, ncomp, params.reg_covar, False)
        pre = gmmh_precision_cholesky(ms.covariances, d, ncomp)
        if pre.info != 0:
            raise Error(
                _collapse_message(it, pre.info, pre.failed_component, params.reg_covar)
            )
        lower_bound = es.meanll
        # gmm_convergence_change (DEVIATION 1747) and gmm_converged
        var change: Float32
        if prev == _neg_inf():
            change = _pos_inf()
        else:
            change = ftz(lower_bound - prev)
        n_iter = it
        var a = change
        if a < Float32(0.0):
            a = -a
        if a < params.tol:
            converged = True
            break

    return GmmHostModel(
        ncomp, d, ms.weights.copy(), ms.means.copy(), ms.covariances.copy(),
        pre.precisions.copy(), pre.log_det_chol.copy(), n_iter, converged,
        lower_bound,
    )


def _score_e_step(
    weights: List[Float32],
    means: List[Float32],
    prec: List[Float32],
    log_det_chol: List[Float32],
    ncomp: Int,
    d: Int,
    x: List[Float32],
    n: Int,
    output_level: Int,
) raises -> GmmHostEStep:
    """The scoring entries' E-step: the log weights through `_safe_log`."""
    gmmh_validate_data(x, n, d)
    var lw = List[Float32](capacity=ncomp)
    for k in range(ncomp):
        lw.append(_safe_log(weights[k]))
    return gmmh_e_step(
        x, means, prec, log_det_chol, lw, n, d, ncomp, output_level
    )


def gmmh_score_samples(
    weights: List[Float32],
    means: List[Float32],
    prec: List[Float32],
    log_det_chol: List[Float32],
    ncomp: Int,
    d: Int,
    x: List[Float32],
    n: Int,
) raises -> List[Float32]:
    """`gaussian_mixture_score_samples`: the logsumexp per row."""
    var es = _score_e_step(
        weights, means, prec, log_det_chol, ncomp, d, x, n, 1
    )
    return es.lse.copy()


def gmmh_predict_proba(
    weights: List[Float32],
    means: List[Float32],
    prec: List[Float32],
    log_det_chol: List[Float32],
    ncomp: Int,
    d: Int,
    x: List[Float32],
    n: Int,
) raises -> List[Float32]:
    """`gaussian_mixture_predict_proba`: `exp(log_resp)` through
    `_exp_resp`."""
    var es = _score_e_step(
        weights, means, prec, log_det_chol, ncomp, d, x, n, 2
    )
    var out = List[Float32](capacity=n * ncomp)
    for i in range(n * ncomp):
        out.append(ftz(identical_exp(ftz(es.logresp[i]))))
    return out^


def gmmh_predict(
    weights: List[Float32],
    means: List[Float32],
    prec: List[Float32],
    log_det_chol: List[Float32],
    ncomp: Int,
    d: Int,
    x: List[Float32],
    n: Int,
) raises -> List[Int32]:
    """`gaussian_mixture_predict`: `argmax_kernel` over the weighted log
    probabilities, the lowest index on a tie."""
    var es = _score_e_step(
        weights, means, prec, log_det_chol, ncomp, d, x, n, 0
    )
    var out = List[Int32](capacity=n)
    for i in range(n):
        var base = i * ncomp
        var best = es.wlp[base]
        var best_k = 0
        for k in range(1, ncomp):
            var v = es.wlp[base + k]
            if v > best:
                best = v
                best_k = k
        out.append(Int32(best_k))
    return out^


def gmmh_n_parameters(d: Int, ncomp: Int) -> Int:
    """`gmm_n_parameters`."""
    var cov_params = ncomp * d * (d + 1) // 2
    var mean_params = d * ncomp
    return cov_params + mean_params + ncomp - 1


@fieldwise_init
struct GmmHostScores(Copyable, Movable):
    var score: Float32
    var bic: Float32
    var aic: Float32


def gmmh_score_bic_aic(
    weights: List[Float32],
    means: List[Float32],
    prec: List[Float32],
    log_det_chol: List[Float32],
    ncomp: Int,
    d: Int,
    x: List[Float32],
    n: Int,
) raises -> GmmHostScores:
    """`gaussian_mixture_score`, `_bic` and `_aic`: each recomputes the score
    on the device path, and the score is a pure function of its inputs, so
    one computation serves all three."""
    var s = gmmh_score_samples(weights, means, prec, log_det_chol, ncomp, d, x, n)
    var acc = Float32(0.0)
    for i in range(n):
        acc = ftz(acc + ftz(s[i]))
    var sc = ftz(identical_div(acc, Float32(n)))
    var p = gmmh_n_parameters(d, ncomp)
    var a = ftz(
        identical_mul(Float32(-2.0), ftz(identical_mul(sc, Float32(n))))
    )
    var b = ftz(identical_mul(Float32(p), ftz(identical_log(Float32(n)))))
    var bic = ftz(a + b)
    var aic = ftz(a + ftz(identical_mul(Float32(2.0), Float32(p))))
    return GmmHostScores(sc, bic, aic)
