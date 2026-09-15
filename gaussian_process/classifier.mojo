# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surface for binary Gaussian process CLASSIFICATION on the
GPU: the Laplace approximation of scikit-learn 1.9.0 `_gpc.py`
(`_BinaryGaussianProcessClassifierLaplace`), what `bindings/_mojolearn_gp.mojo`
calls and `python/mojolearn/_gpc_impl.py` exposes as
`mojolearn.GaussianProcessClassifier`. One-vs-rest past two classes is the
surface's loop over this entry (DEVIATION 2833).

WHAT RUNS WHERE. The device runs what the regressor's device path runs, and
nothing vendor specific: the kernel matrix and the cross-covariance
(`gaussian_process/checks/kernels.mojo::gp_kernel_matrix`), the Cholesky
factor and solve of `B` (`cholesky/estimator.mojo`), the two
matrix-vector products per Newton iteration and the latent mean
(`gemm/checks/gemm_identical.mojo::identical_gemm_into` at `OP_TN`), and the
triangular solve for the latent variance (`cholesky/checks/trsm.mojo::
trsm_lower`). The per-row steps (the sigmoid, the weights, `B`'s cells, the
Newton right-hand side, the likelihood and the stop test, the variance fold
and the float64 probability) are `gaussian_process/host/gpc_steps.mojo`, the
same source the CPU host oracle compiles. DEVIATIONS 2830 (the stop rule),
2831 (the pinned float32 orders) and 2832 (the float64 probability and its
erf) are written there.

NO OPTIMIZER. As for the regressor (DEVIATION 1761), the kernel is the
kernel passed in; `optimizer`, `n_restarts_optimizer` and `warm_start` are
refused by name in Python before a binding is reached. DEVIATION 1766, the
refusal of classification itself, is closed by DEVIATION 2830.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from cholesky.checks.trsm import CHOL_SOLVE_TPB, trsm_lower
from cholesky.estimator import cholesky_factor_host, cholesky_solve_host
from core.identity_trace import IdentityTrace
from gaussian_process.checks.gp_sabotage import GP_SAB_NONE
from gaussian_process.checks.kernels import (
    GP_ELEM_TPB,
    GPKernelSpec,
    gp_kernel_diag,
    gp_kernel_matrix,
    gp_kernel_stack_floats,
    gp_validate_kernel,
)
from gaussian_process.estimator import (
    _download,
    _length_scale_table,
    _upload,
    gp_validate_data,
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
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_TN


def _gpc_kernel_self(
    x: List[Float32], n_train: Int, n_features: Int, kernel: GPKernelSpec
) raises -> List[Float32]:
    """`K = kernel(X)` on the device (`_gpc.py:261`), is_self True, so a
    WhiteKernel adds its noise to the diagonal as in the regressor."""
    var trace = IdentityTrace()
    var ctx = DeviceContext()
    var dx = _upload(ctx, x)
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dk = ctx.enqueue_create_buffer[DType.float32](n_train * n_train)
    var dstack = ctx.enqueue_create_buffer[DType.float32](
        gp_kernel_stack_floats(n_train, n_train)
    )
    ctx.synchronize()
    gp_kernel_matrix(
        ctx,
        dk,
        dx,
        dx,
        dls,
        dstack,
        n_train,
        n_train,
        n_features,
        kernel,
        True,
        trace,
        "gpc.kernel",
        GP_ELEM_TPB,
        GP_SAB_NONE,
    )
    var k_host = _download(ctx, dk, n_train * n_train)
    _ = dx^
    _ = dls^
    _ = dk^
    _ = dstack^
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^
    return k_host^


def _gpc_matvec(k: List[Float32], v: List[Float32], n: Int) raises -> List[Float32]:
    """`K v` through the pinned gemm at `OP_TN` (`K` is symmetric by bits,
    so `K^T v` is `K v`), the host oracle's `gemm_oracle(k, v, OP_TN, n, 1,
    n)` on the device."""
    var ctx = DeviceContext()
    var dk = _upload(ctx, k)
    var dv = _upload(ctx, v)
    var dc = ctx.enqueue_create_buffer[DType.float32](n)
    var dws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n, 1, n)
    )
    ctx.synchronize()
    identical_gemm_into(ctx, dc, dk, dv, dws, n, 1, n, OP_TN)
    var out = _download(ctx, dc, n)
    _ = dk^
    _ = dv^
    _ = dc^
    _ = dws^
    _ = ctx^
    return out^


def gpc_fit_binary_host(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    kernel: GPKernelSpec,
    max_iter_predict: Int,
) raises -> GPCBinaryFit:
    """`_BinaryGaussianProcessClassifierLaplace(kernel, optimizer=None).fit`
    with `y` already encoded to 0 and 1 (`_gpc.py:171-267`): the kernel
    matrix, then `_posterior_mode` from `f = 0` (`_gpc.py:461`, warm_start
    refused). Refuses by name: non-finite X, a kernel the constructors
    refuse, a target outside {0, 1} or a single class, max_iter_predict
    below 1, and a factorization of `B` that fails (it cannot for finite
    inputs, `B`'s eigenvalues are at least 1)."""
    gp_validate_data(x, n_train, n_features, String("X"))
    gpc_validate_labels(y, n_train)
    gp_validate_kernel(kernel, n_features)
    gpc_validate_max_iter(max_iter_predict)

    var k = _gpc_kernel_self(x, n_train, n_features, kernel)
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
        var factor = cholesky_factor_host(bmat, n_train, Float32(0.0))
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
        var kb = _gpc_matvec(k, bvec, n_train)
        var c = gpc_scale(wt.wsr, kb)
        var xs = cholesky_solve_host(factor, c, 1)
        var a = gpc_a_vector(bvec, wt.wsr, xs)
        f = _gpc_matvec(k, a, n_train)
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


def gpc_validate_model(
    y: List[Float32],
    pi: List[Float32],
    wsr: List[Float32],
    l: List[Float32],
    n_train: Int,
) raises:
    """The saved arrays a prediction reads, sized against `n_train`."""
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


def gpc_predict_binary_host(
    x_train: List[Float32],
    y: List[Float32],
    pi: List[Float32],
    wsr: List[Float32],
    l: List[Float32],
    n_train: Int,
    n_features: Int,
    kernel: GPKernelSpec,
    x_star: List[Float32],
    n_star: Int,
    want_variance: Bool,
) raises -> GPCLatent:
    """`latent_mean_and_variance(X_star)` (`_gpc.py:434-441`); the mean
    alone when `want_variance` is False (`predict`, `_gpc.py:287-290`).

        K_star = kernel(X_train, X_star)          n_train x n_star
        mean   = K_star^T (y - pi)                gemm at OP_TN
        v      = solve(L, W_sr[:, None] K_star)   trsm_lower in place
        var    = kernel.diag(X_star) - sum_i v_i^2
    """
    if n_star <= 0:
        raise Error(
            "gpc_predict_host: n_star must be positive, got " + String(n_star)
        )
    gp_validate_data(x_train, n_train, n_features, String("X_train"))
    gp_validate_data(x_star, n_star, n_features, String("X_star"))
    gp_validate_kernel(kernel, n_features)
    gpc_validate_model(y, pi, wsr, l, n_train)
    var kss = gp_kernel_diag(kernel)
    var r = gpc_residual(y, pi)

    var trace = IdentityTrace()
    var ctx = DeviceContext()
    var dx = _upload(ctx, x_train)
    var dxs = _upload(ctx, x_star)
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dr = _upload(ctx, r)
    var dkc = ctx.enqueue_create_buffer[DType.float32](n_train * n_star)
    var dstack = ctx.enqueue_create_buffer[DType.float32](
        gp_kernel_stack_floats(n_train, n_star)
    )
    var dmean = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_star, 1, n_train)
    )
    ctx.synchronize()
    gp_kernel_matrix(
        ctx,
        dkc,
        dx,
        dxs,
        dls,
        dstack,
        n_train,
        n_star,
        n_features,
        kernel,
        False,
        trace,
        "gpc.kcross",
        GP_ELEM_TPB,
        GP_SAB_NONE,
    )
    identical_gemm_into(ctx, dmean, dkc, dr, dws, n_star, 1, n_train, OP_TN)
    var mean = _download(ctx, dmean, n_star)
    var variance = List[Float32]()
    if want_variance:
        var kc = _download(ctx, dkc, n_train * n_star)
        var scaled = gpc_scale_rows(kc, wsr, n_train, n_star)
        var dv = _upload(ctx, scaled)
        var dl = _upload(ctx, l)
        trsm_lower(ctx, dl, dv, n_train, n_star, trace, "gpc.v", CHOL_SOLVE_TPB)
        var v = _download(ctx, dv, n_train * n_star)
        variance = gpc_latent_var(v, n_train, n_star, kss)
        _ = dv^
        _ = dl^
    _ = dx^
    _ = dxs^
    _ = dls^
    _ = dr^
    _ = dkc^
    _ = dstack^
    _ = dmean^
    _ = dws^
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^
    return GPCLatent(mean^, variance^)
