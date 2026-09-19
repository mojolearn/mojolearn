# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""KernelRidge, Nystroem and RBFSampler on the HOST: the one-shot entries of
`kernel_methods/estimator.mojo` restated without a device (CPU training for
the workstream D estimators, 2026-09-15).

WHAT THIS IS. `kernel_ridge_fit_host`, `nystroem_fit_host`,
`rbf_sampler_fit_host` and their predict and transform run their arithmetic
in device kernels. This file is a SECOND spelling of the driver, in the
driver's order, with no `DeviceContext`, no trace and no launch geometry, so
the kernel_methods CPU host binding serves the three estimators on a
CPU-only install. Every stage names the device line it mirrors:

    the kernel matrix       kernel_methods/checks/kernel_matrix.mojo::
                            km_kernel_matrix at LINEAR and RBF (plus polynomial,
                            sigmoid and ascending L1 Laplacian epilogues): the squared row
                            norms (svm row_norm_l2sq_kernel, an ascending
                            ftz(identical_mul_add(v, v, acc)) chain), the
                            product at OP_NT through the gemm profile's
                            normative answer, then svm
                            rbf_kernel_expanded_kernel's epilogue in its
                            association
    the ridge               kernel_ridge.mojo::add_ridge_diag_kernel,
                            ftz(ftz(K_ii) + alpha)
    potrf and potrs         cholesky/host/chol_oracle.mojo::
                            chol_host_factor_lower and chol_host_solve (the
                            gp host binding's factorization, without the
                            door's validation), at jitter +0.0, which
                            is DEVIATION 1660's state: the ridge is already in
                            the matrix
    the KRR prediction      identical_gemm_into at OP_NN, `K(X, X_fit) . dual`
    the basis rows          random_features.mojo::km_basis_indices, CALLED
                            (a host pass on the device path too)
    the Jacobi              decomposition/host/pca_oracle.mojo::
                            host_jacobi_eigh at JACOBI_SWEEPS and JACOBI_TOL,
                            the PCA host lane's replay of jacobi_eigh_kernel
    the sign flip           pca_oracle.mojo::host_sign_flip (sign_flip_kernel)
    the order and the clip  estimator.mojo::_eigen_order_f32,
                            _singular_value_f32 and _eigen_clip_f32, restated
    U / sqrt(S)             estimator.mojo::scale_columns_kernel,
                            identical_div per cell
    the normalization       identical_gemm_into at OP_NT with the column
                            signed Q when an eigenvalue is negative
    the Nystroem embedding  the cross kernel, then OP_NT against the
                            normalization (DEVIATION 1674's transpose)
    the random features     random_features.mojo::km_random_weights_host and
                            km_random_offsets_host: the SAME scalar functions
                            random_weights_kernel and random_offsets_kernel
                            call per cell, so the draws are integer arithmetic
                            into one pinned transform on both sides
    the RBF feature map     identical_gemm_into at OP_NN, then
                            feature_map_epilogue_kernel's add, cos, multiply

WHAT IS REFUSED, BY NAME, AS ON THE DEVICE: non-finite or mis-shaped inputs,
a non-positive gamma where the kernel reads one, a NaN or negative alpha, a
ridged kernel matrix that does not factor (DEVIATION 1662), an unconverged
Jacobi, n_components above n_samples or above KM_MAX_BASIS_POOL rows, and a
non-positive gamma or n_components for the sampler. Precomputed kernels remain refused, matching the device surface. Polynomial
degree is passed through and bounded to the same 0..32 interval as the device.

THE SABOTAGE. Laplacian reverses its L1 feature chain. Under
`-D MOJOLEARN_HOST_SABOTAGE=1` every GEMM leaf walks descending
(`gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`), which moves every
kernel matrix, the Cholesky trailing update, the normalization, the
embedding and the random feature map.
"""

from std.memory import bitcast
from std.math import abs
from std.sys.compile import is_defined

from checks.numerics import (
    ftz,
    identical_cos,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
    identical_tanh,
)
from cholesky.host.chol_oracle import chol_host_factor_lower, chol_host_solve
from decomposition.host.pca_oracle import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    host_jacobi_eigh,
    host_sign_flip,
)
from gemm.host.identical_gemm import OP_NN, OP_NT, gemm_oracle
from kernel_methods.checks.random_features import (
    km_basis_indices,
    km_feature_scale,
    km_random_offsets_host,
    km_random_weights_host,
    km_weight_sigma,
)

#: `KM_KERNEL_*` (`kernel_methods/checks/kernel_matrix.mojo`), which are
#: svm's `KERNEL_*` codes plus the lane's laplacian at 5.
comptime KMH_KERNEL_LINEAR = 0
comptime KMH_KERNEL_POLYNOMIAL = 1
comptime KMH_KERNEL_RBF = 2
comptime KMH_KERNEL_SIGMOID = 3
comptime KMH_KERNEL_PRECOMPUTED = 4
comptime KMH_KERNEL_LAPLACIAN = 5
# Same bound as impl/distance/kernel_matrices.mojo, without importing GPU code.
comptime KMH_MAX_DEGREE = 32


def kmh_kernel_name(kernel: Int) -> String:
    """`km_kernel_name`."""
    if kernel == KMH_KERNEL_LINEAR:
        return String("linear")
    if kernel == KMH_KERNEL_POLYNOMIAL:
        return String("polynomial")
    if kernel == KMH_KERNEL_RBF:
        return String("rbf")
    if kernel == KMH_KERNEL_SIGMOID:
        return String("sigmoid")
    if kernel == KMH_KERNEL_PRECOMPUTED:
        return String("precomputed")
    if kernel == KMH_KERNEL_LAPLACIAN:
        return String("laplacian")
    return String("unknown")


def kmh_validate_matrix(
    values: List[Float32], n_rows: Int, n_cols: Int, what: String
) raises:
    """`km_validate_matrix`: shape and finiteness, by name, with the
    offending flat index (DEVIATION 1686)."""
    if n_rows <= 0 or n_cols <= 0:
        raise Error(
            what
            + ": need positive dimensions, got "
            + String(n_rows)
            + " x "
            + String(n_cols)
        )
    if len(values) != n_rows * n_cols:
        raise Error(
            what
            + " holds "
            + String(len(values))
            + " floats, "
            + String(n_rows)
            + " x "
            + String(n_cols)
            + " needs "
            + String(n_rows * n_cols)
        )
    for i in range(len(values)):
        var v = values[i]
        if v != v:
            raise Error(
                what + ": NaN at flat index " + String(i)
                + "; refused by name (DEVIATION 1686)"
            )
        if v > Float32(3.4028234663852886e38) or v < Float32(
            -3.4028234663852886e38
        ):
            raise Error(
                what + ": infinity at flat index " + String(i)
                + "; refused by name (DEVIATION 1686)"
            )


def kmh_validate_kernel(
    kernel: Int, degree: Int, gamma: Float64, coef0: Float64, what: String
) raises:
    """The device's five kernel kinds and polynomial degree contract."""
    if kernel == KMH_KERNEL_PRECOMPUTED:
        raise Error(what + ": kernel='precomputed' is refused by name (DEVIATION 1683)")
    if kernel < 0 or kernel > KMH_KERNEL_LAPLACIAN:
        raise Error(what + ": unknown kernel value " + String(kernel))
    if kernel == KMH_KERNEL_POLYNOMIAL:
        if degree < 0 or degree > KMH_MAX_DEGREE:
            raise Error(what + ": polynomial degree must be between 0 and " + String(KMH_MAX_DEGREE))
    if gamma != gamma:
        raise Error(what + ": gamma is NaN")
    if coef0 != coef0:
        raise Error(what + ": coef0 is NaN")
    if kernel != KMH_KERNEL_LINEAR and not (gamma > 0.0):
        raise Error(
            what
            + ": the " + kmh_kernel_name(kernel) + " kernel needs a POSITIVE gamma; got a value that is"
            " not greater than zero"
        )


def kmh_row_norms(x: List[Float32], n_rows: Int, k: Int) -> List[Float32]:
    """`svm row_norm_l2sq_kernel`: one ascending chain per row."""
    var out = List[Float32]()
    for i in range(n_rows):
        var acc = Float32(0.0)
        for c in range(k):
            var v = ftz(x[i * k + c])
            acc = ftz(identical_mul_add(v, v, acc))
        out.append(ftz(acc))
    return out^


def kmh_kernel_matrix(
    kernel: Int,
    degree: Int,
    gamma: Float64,
    coef0: Float64,
    a: List[Float32],
    b: List[Float32],
    m: Int,
    n: Int,
    k: Int,
) raises -> List[Float32]:
    """Replay the five device kernels, preserving each cell's operation order.

    Polynomial/tanh mirror impl/distance/kernel_matrices.mojo; Laplacian
    mirrors the ascending L1 chain and checks/kernel_matrix.mojo epilogue.
    """
    kmh_validate_kernel(kernel, degree, gamma, coef0, "kernel matrix")
    if kernel == KMH_KERNEL_LAPLACIAN:
        var out = List[Float32]()
        var gain = Float32(-gamma)
        for i in range(m):
            for j in range(n):
                var acc = Float32(0.0)
                for pos in range(k):
                    var c = pos
                    comptime if is_defined["MOJOLEARN_HOST_SABOTAGE"]():
                        c = k - 1 - pos
                    acc = ftz(acc + abs(ftz(ftz(a[i * k + c]) - ftz(b[j * k + c]))))
                out.append(ftz(identical_exp(ftz(identical_mul(gain, acc)))))
        return out^
    var dot = gemm_oracle(a, b, OP_NT, m, n, k)
    if kernel == KMH_KERNEL_LINEAR:
        return dot^
    if kernel == KMH_KERNEL_RBF:
        var na = kmh_row_norms(a, m, k)
        var nb = kmh_row_norms(b, n, k)
        var gain = Float32(gamma)
        for i in range(m):
            for j in range(n):
                var t = i * n + j
                var s = ftz(
                    ftz(ftz(na[i]) + ftz(nb[j])) - ftz(Float32(2.0) * ftz(dot[t]))
                )
                var e = ftz((-gain) * s)
                dot[t] = ftz(identical_exp(e))
        return dot^
    var gain = Float32(gamma)
    var offset = Float32(coef0)
    for t in range(m * n):
        var base = ftz(identical_mul_add(gain, ftz(dot[t]), offset))
        if kernel == KMH_KERNEL_SIGMOID:
            dot[t] = ftz(identical_tanh(base))
        else:
            var acc = Float32(1.0)
            for _ in range(degree):
                acc = ftz(identical_mul(acc, base))
            dot[t] = acc
    return dot^


# ===========================================================================
# KernelRidge
# ===========================================================================


def kmh_kernel_ridge_fit(
    x: List[Float32],
    y: List[Float32],
    n: Int,
    d: Int,
    t: Int,
    kernel: Int,
    degree: Int,
    gamma: Float64,
    coef0: Float64,
    alpha: Float32,
) raises -> List[Float32]:
    """`kernel_ridge_fit_host`: form `K`, ridge it, factor it, solve it.
    Returns `dual_coef_`, `n x t` row-major; the fit refuses a non-zero
    `info` by name (DEVIATION 1662)."""
    kmh_validate_matrix(x, n, d, "kernel_ridge X")
    kmh_validate_matrix(y, n, t, "kernel_ridge y")
    kmh_validate_kernel(kernel, degree, gamma, coef0, "kernel_ridge")
    if alpha != alpha:
        raise Error("kernel_ridge_fit_host: alpha is NaN; refused by name")
    if alpha < Float32(0.0):
        raise Error(
            "kernel_ridge_fit_host: alpha must be non-negative, got a"
            " negative value. scikit-learn's own parameter constraint is"
            " Interval(Real, 0, None, closed='left'). DEVIATION 1686"
        )
    var k = kmh_kernel_matrix(kernel, degree, gamma, coef0, x, x, n, n, d)
    # add_ridge_diag_kernel
    for i in range(n):
        var dv = ftz(k[i * n + i])
        k[i * n + i] = ftz(dv + alpha)
    # potrf_lower at the pinned width with no jitter of its own
    # (kernel_ridge_alpha_is_the_ridge) and no validation, as the device
    # solve calls it; the host factorization's ridge step at +0.0 leaves
    # every positive diagonal entry's bits as they are.
    var f = chol_host_factor_lower(k, n, Float32(0.0))
    if f.info != 0:
        raise Error(
            "kernel_ridge_fit_host: the ridged kernel matrix K + alpha I is"
            " NOT positive definite (info="
            + String(f.info)
            + ", the leading minor of order "
            + String(f.info)
            + " failed). kernel="
            + kmh_kernel_name(kernel)
            + ", n_samples="
            + String(n)
            + ". THE CLOSURE IS alpha: raise it (DEVIATION 1662)"
        )
    var dual = chol_host_solve(f, y, t)
    _ = f^
    return dual^


def kmh_kernel_ridge_predict(
    x_fit: List[Float32],
    dual: List[Float32],
    n: Int,
    d: Int,
    t: Int,
    kernel: Int,
    degree: Int,
    gamma: Float64,
    coef0: Float64,
    x_new: List[Float32],
    q: Int,
) raises -> List[Float32]:
    """`kernel_ridge_predict_host`: `K(X, X_fit) . dual` at OP_NN
    (DEVIATION 1680). `q x t` row-major."""
    kmh_validate_matrix(x_new, q, d, "predict X")
    kmh_validate_kernel(kernel, degree, gamma, coef0, "kernel_ridge")
    var k = kmh_kernel_matrix(kernel, degree, gamma, coef0, x_new, x_fit, q, n, d)
    return gemm_oracle(k, dual, OP_NN, q, t, n)


# ===========================================================================
# Nystroem
# ===========================================================================


@fieldwise_init
struct KmhNystroem(Movable):
    """`NystroemModel`'s arrays and sweep count."""

    var components: List[Float32]
    var indices: List[Int32]
    var normalization: List[Float32]
    var eigenvalues: List[Float32]
    var eigenvectors: List[Float32]
    var sweeps: Int


def _kmh_singular_value(lam: Float32) -> Float32:
    """`estimator.mojo::_singular_value_f32`: `|lambda|` by bits."""
    return bitcast[DType.float32](
        bitcast[DType.uint32](lam) & UInt32(0x7FFFFFFF)
    )


def _kmh_eigen_order(values: List[Float32], q: Int) -> List[Int]:
    """`estimator.mojo::_eigen_order_f32` at KMSAB_NONE: a selection sort,
    value descending, the lower index kept on a tie."""
    var used = List[Bool]()
    for _ in range(q):
        used.append(False)
    var order = List[Int]()
    for _ in range(q):
        var best = -1
        for c in range(q):
            if used[c]:
                continue
            if best < 0:
                best = c
                continue
            if values[c] > values[best]:
                best = c
        used[best] = True
        order.append(best)
    return order^


def kmh_nystroem_fit(
    x: List[Float32],
    n: Int,
    d: Int,
    kernel: Int,
    degree: Int,
    gamma: Float64,
    coef0: Float64,
    q: Int,
    seed: UInt64,
) raises -> KmhNystroem:
    """`nystroem_fit_host`, step for step."""
    kmh_validate_matrix(x, n, d, "nystroem X")
    kmh_validate_kernel(kernel, degree, gamma, coef0, "nystroem")

    var basis = km_basis_indices(seed, n, q)
    var comp = List[Float32]()
    for c in range(q):
        var srow = Int(basis[c])
        for f in range(d):
            comp.append(x[srow * d + f])

    var raw = kmh_kernel_matrix(kernel, degree, gamma, coef0, comp, comp, q, q, d)
    var jac = host_jacobi_eigh(raw, q, JACOBI_SWEEPS, Float32(JACOBI_TOL))
    var vecs = jac.vectors.copy()
    host_sign_flip(vecs, q)
    if not jac.converged:
        raise Error(
            "nystroem_fit_host: the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n_components = "
            + String(q)
            + "; ||offdiag(A)||_F / ||A||_F is still "
            + String(jac.rel)
            + " against a tolerance of "
            + String(JACOBI_TOL)
        )
    var sweeps = jac.executed

    var values_raw = List[Float32]()
    var mags = List[Float32]()
    for c in range(q):
        values_raw.append(raw[c * q + c])
        mags.append(_kmh_singular_value(raw[c * q + c]))
    var order = _kmh_eigen_order(mags, q)

    var clip = Float32(1e-12)
    var values = List[Float32]()
    var sqrt_s = List[Float32]()
    var any_negative = False
    var vecs_ord = List[Float32](length=q * q, fill=Float32(0.0))
    var vt_ord = List[Float32](length=q * q, fill=Float32(0.0))
    for c in range(q):
        var src = order[c]
        var s = mags[src]
        if s < clip:
            s = clip
        values.append(s)
        sqrt_s.append(ftz(identical_sqrt(s)))
        var negative = values_raw[src] < Float32(0.0)
        if negative:
            any_negative = True
        for f in range(q):
            var e = vecs[f * q + src]
            vecs_ord[f * q + c] = e
            vt_ord[f * q + c] = -e if negative else e

    # scale_columns_kernel: a divide per cell, never a reciprocal.
    var z = List[Float32](length=q * q, fill=Float32(0.0))
    for tcell in range(q * q):
        var kcol = tcell % q
        z[tcell] = ftz(identical_div(ftz(vecs_ord[tcell]), ftz(sqrt_s[kcol])))

    var norm: List[Float32]
    if any_negative:
        norm = gemm_oracle(z, vt_ord, OP_NT, q, q, q)
    else:
        norm = gemm_oracle(z, vecs_ord, OP_NT, q, q, q)

    _ = raw^
    _ = jac^
    return KmhNystroem(comp^, basis^, norm^, values^, vecs_ord^, sweeps)


def kmh_nystroem_transform(
    components: List[Float32],
    normalization: List[Float32],
    q: Int,
    d: Int,
    kernel: Int,
    degree: Int,
    gamma: Float64,
    coef0: Float64,
    x: List[Float32],
    m: Int,
) raises -> List[Float32]:
    """`nystroem_transform_host`: `K(X, components) @ normalization.T`
    (DEVIATION 1674). `m x q` row-major."""
    kmh_validate_matrix(x, m, d, "nystroem transform X")
    kmh_validate_kernel(kernel, degree, gamma, coef0, "nystroem")
    var k = kmh_kernel_matrix(kernel, degree, gamma, coef0, x, components, m, q, d)
    return gemm_oracle(k, normalization, OP_NT, m, q, q)


# ===========================================================================
# RBFSampler
# ===========================================================================


@fieldwise_init
struct KmhRBFSampler(Movable):
    var weights: List[Float32]
    var offset: List[Float32]
    var sigma: Float32
    var scale: Float32


def kmh_rbf_sampler_fit(
    d: Int, q: Int, gamma: Float32, seed: UInt64
) raises -> KmhRBFSampler:
    """`rbf_sampler_fit_host`: the draws depend on `n_features`,
    `n_components` and the seed alone."""
    if d <= 0:
        raise Error(
            "rbf_sampler_fit_host: n_features must be positive, got "
            + String(d)
        )
    if q <= 0:
        raise Error(
            "rbf_sampler_fit_host: n_components must be positive, got "
            + String(q)
            + ". scikit-learn's constraint is Interval(Integral, 1, None,"
            " closed='left'). DEVIATION 1686"
        )
    if gamma != gamma or not (gamma > Float32(0.0)):
        raise Error(
            "rbf_sampler_fit_host: gamma must be POSITIVE; got a value that"
            " is not greater than zero. DEVIATION 1686"
        )
    var sigma = km_weight_sigma(gamma)
    var scale = km_feature_scale(q)
    var w = km_random_weights_host(seed, d, q, sigma)
    var b = km_random_offsets_host(seed, q)
    return KmhRBFSampler(w^, b^, sigma, scale)


def kmh_rbf_sampler_transform(
    weights: List[Float32],
    offset: List[Float32],
    d: Int,
    q: Int,
    scale: Float32,
    x: List[Float32],
    m: Int,
) raises -> List[Float32]:
    """`rbf_sampler_transform_host`: the dot at OP_NN, then
    `feature_map_epilogue_kernel`'s add, cos and multiply in their order."""
    kmh_validate_matrix(x, m, d, "rbf_sampler transform X")
    var p = gemm_oracle(x, weights, OP_NN, m, q, d)
    for t in range(m * q):
        var j = t % q
        var pv = ftz(p[t])
        var shifted = ftz(pv + ftz(offset[j]))
        p[t] = ftz(identical_mul(identical_cos(shifted), scale))
    return p^
