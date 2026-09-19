# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""Binary Gaussian process classification on the HOST, the second spelling
of `gaussian_process/classifier.mojo` in the device path's order, so the gp
CPU host binding (`bindings/_mojolearn_gp_host.mojo`) serves
`GaussianProcessClassifier` with no GPU.

THE STAGES, EACH WITH THE DEVICE LINE IT RESTATES

    validation          classifier.mojo gpc_fit_binary_host, same order
    K = k(X, X)         gpr_oracle.mojo gpr_host_kernel_matrix, is_self True
                        (kernels.mojo gp_kernel_matrix on the device)
    the Newton step     gpc_steps.mojo, the same source on both paths
    L L^T = B           chol_oracle.mojo chol_host_potrf at jitter +0.0
                        (cholesky_factor_host)
    K b, K a            gemm_oracle(k, v, OP_TN, n, 1, n)
                        (identical_gemm_into at OP_TN)
    cho_solve           chol_host_solve (cholesky_solve_host)
    K_* = k(Xtr, X*)    gpr_host_kernel_matrix, is_self False
    mean                gemm_oracle(kcross, y - pi, OP_TN, n_star, 1, n_train)
    v = L^-1 W_sr K_*   chol_host_trsm_lower in place (trsm_lower)

THE SABOTAGE is the gp family's: `-D MOJOLEARN_HOST_SABOTAGE=1` walks the
scaled distance's feature axis descending in `gpr_host_kernel_matrix` and the
gemm oracle's leaf descending, so every fit and prediction here moves.
"""

from cholesky.host.chol_oracle import (
    chol_host_potrf,
    chol_host_solve,
    chol_host_trsm_lower,
)
from gaussian_process.host.gpc_steps import (
    GPCBinaryFit,
    GPCLatent,
    gpc_a_vector,
    gpc_b_matrix,
    gpc_latent_var,
    gpc_lml,
    gpc_neg_inf32,
    gpc_newton_rhs,
    gpc_residual,
    gpc_scale,
    gpc_scale_rows,
    gpc_stop,
    gpc_validate_labels,
    gpc_validate_max_iter,
    gpc_weights,
)
from gaussian_process.host.gpr_oracle import (
    GPHostKernelSpec,
    gpr_host_kernel_diag,
    gpr_host_kernel_matrix,
    gpr_host_validate_data,
    gpr_host_validate_kernel,
)
from gemm.host.identical_gemm import OP_TN, gemm_oracle


def gpc_host_fit(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    spec: GPHostKernelSpec,
    max_iter_predict: Int,
) raises -> GPCBinaryFit:
    """`classifier.mojo::gpc_fit_binary_host` on the host."""
    gpr_host_validate_data(x, n_train, n_features, String("X"))
    gpc_validate_labels(y, n_train)
    gpr_host_validate_kernel(spec, n_features)
    gpc_validate_max_iter(max_iter_predict)

    var k = gpr_host_kernel_matrix(x, n_train, x, n_train, n_features, spec, True)
    var f = List[Float32](capacity=n_train)
    for _i in range(n_train):
        f.append(Float32(0.0))
    var previous = gpc_neg_inf32()
    var n_iter = 0
    var last_pi = List[Float32]()
    var last_wsr = List[Float32]()
    var last_l = List[Float32]()
    var nb = 0
    for it in range(max_iter_predict):
        var wt = gpc_weights(f)
        var bmat = gpc_b_matrix(k, wt.wsr, n_train)
        var factor = chol_host_potrf(bmat, n_train, Float32(0.0))
        if factor.info != 0:
            raise Error(
                "gpc_fit_host: the factorization of B = I + W_sr K W_sr failed"
                " at Newton iteration "
                + String(it + 1)
                + " (info="
                + String(factor.info)
                + "). B's eigenvalues are at least 1 for any finite kernel"
                " matrix, so this means a non-finite latent value"
            )
        var bvec = gpc_newton_rhs(wt.w, f, y, wt.pi)
        var kb = gemm_oracle(k, bvec, OP_TN, n_train, 1, n_train)
        var c = gpc_scale(wt.wsr, kb)
        var xs = chol_host_solve(factor, c, 1)
        var a = gpc_a_vector(bvec, wt.wsr, xs)
        f = gemm_oracle(k, a, OP_TN, n_train, 1, n_train)
        var lml = gpc_lml(a, f, y, factor.logdet)
        n_iter = it + 1
        last_pi = wt.pi.copy()
        last_wsr = wt.wsr.copy()
        last_l = factor.l.copy()
        nb = factor.nb
        if gpc_stop(lml, previous):
            break
        previous = lml
    return GPCBinaryFit(last_l^, last_pi^, last_wsr^, previous, n_iter, nb)


def gpc_host_predict(
    x_train: List[Float32],
    y: List[Float32],
    pi: List[Float32],
    wsr: List[Float32],
    l: List[Float32],
    n_train: Int,
    n_features: Int,
    spec: GPHostKernelSpec,
    x_star: List[Float32],
    n_star: Int,
    want_variance: Bool,
) raises -> GPCLatent:
    """`classifier.mojo::gpc_predict_binary_host` on the host."""
    if n_star <= 0:
        raise Error(
            "gpc_predict_host: n_star must be positive, got " + String(n_star)
        )
    gpr_host_validate_data(x_train, n_train, n_features, String("X_train"))
    gpr_host_validate_data(x_star, n_star, n_features, String("X_star"))
    gpr_host_validate_kernel(spec, n_features)
    gpc_validate_labels(y, n_train)
    if len(pi) != n_train or len(wsr) != n_train or len(l) != n_train * n_train:
        raise Error(
            "gpc_predict_host: the fitted arrays hold pi "
            + String(len(pi))
            + ", W_sr "
            + String(len(wsr))
            + " and L "
            + String(len(l))
            + " values; n_train "
            + String(n_train)
            + " needs n_train, n_train and n_train^2"
        )
    var kss = gpr_host_kernel_diag(spec)
    var r = gpc_residual(y, pi)
    var kcross = gpr_host_kernel_matrix(
        x_train, n_train, x_star, n_star, n_features, spec, False
    )
    var mean = gemm_oracle(kcross, r, OP_TN, n_star, 1, n_train)
    var variance = List[Float32]()
    if want_variance:
        var v = gpc_scale_rows(kcross, wsr, n_train, n_star)
        chol_host_trsm_lower(l, v, n_train, n_star)
        variance = gpc_latent_var(v, n_train, n_star, kss)
    return GPCLatent(mean^, variance^)
