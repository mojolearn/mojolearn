# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Internal CPU verifier: compiled into the gp host binding for identity checks; not a public CPU surface.
"""`GaussianProcessRegressor.sample_y` on the HOST (2026-09-15), the
internal CPU verifier of `gaussian_process/estimator.mojo::gpr_sample_y_host`.

A separate file from `gpr_oracle.mojo` because the stream imports
`core/philox.mojo`, and `gpr_oracle.mojo`'s import list is pinned GPU-free
(test_cpu_training_gp). Every step is the device line's host spelling,
DEVIATION 2793 in `gaussian_process/checks/sample_y.mojo`:

    K_* = k(Xtr, X*)    gpr_host_kernel_matrix, is_self False
    mean                gemm_oracle(kcross, dual, OP_TN, n_star, 1, n_train)
    V = L^-1 K_*        chol_host_trsm_lower in place
    V^T V               gemm_oracle(V, V, OP_TN, n_star, n_star, n_train)
    K** = k(X*, X*)     gpr_host_kernel_matrix, is_self True
    C, mirrored         gp_sample_y_covariance (shared host code)
    L_C                 chol_host_potrf(C, n_star, 2^-20)
    Z                   gp_sample_y_normals (shared host code)
    L_C Z               gemm_oracle(L_C, Z, OP_NN, n_star, n_samples, n_star)
    y                   gp_sample_y_add_mean (shared host code)

Public CPU access to `sample_y` from a saved model is not here; it belongs
to lane/inference-neighbors-density. The sabotage build moves this path
through `gpr_oracle.mojo`'s descending feature walk (both kernel matrices)
and `gemm_oracle`'s own leaf.
"""

from cholesky.host.chol_oracle import (
    chol_host_jitter_pinned,
    chol_host_potrf,
    chol_host_trsm_lower,
)
from gaussian_process.checks.sample_y import (
    gp_sample_y_add_mean,
    gp_sample_y_check_factor,
    gp_sample_y_covariance,
    gp_sample_y_normals,
    gp_sample_y_validate,
)
from gaussian_process.host.gpr_oracle import (
    GPHostKernelSpec,
    gpr_host_kernel_matrix,
    gpr_host_validate_data,
)
from gemm.host.identical_gemm import OP_NN, OP_TN, gemm_oracle


def gpr_host_sample_y(
    x_train: List[Float32],
    l: List[Float32],
    dual: List[Float32],
    n_train: Int,
    n_features: Int,
    spec: GPHostKernelSpec,
    info: Int,
    x_star: List[Float32],
    n_star: Int,
    n_samples: Int,
    seed: UInt64,
) raises -> List[Float32]:
    """`n_star x n_samples` float32 row-major draws, the device method's
    bytes."""
    gp_sample_y_validate(info, n_star, n_samples)
    gpr_host_validate_data(x_star, n_star, n_features, String("X_star"))
    var kcross = gpr_host_kernel_matrix(
        x_train, n_train, x_star, n_star, n_features, spec, False
    )
    var mean = gemm_oracle(kcross, dual, OP_TN, n_star, 1, n_train)
    chol_host_trsm_lower(l, kcross, n_train, n_star)
    var vtv = gemm_oracle(kcross, kcross, OP_TN, n_star, n_star, n_train)
    var kss = gpr_host_kernel_matrix(
        x_star, n_star, x_star, n_star, n_features, spec, True
    )
    var cov = gp_sample_y_covariance(kss, vtv, n_star)
    var factor = chol_host_potrf(cov, n_star, chol_host_jitter_pinned())
    gp_sample_y_check_factor(factor.info, n_star)
    var z = gp_sample_y_normals(n_star, n_samples, seed)
    var lz = gemm_oracle(factor.l, z, OP_NN, n_star, n_samples, n_star)
    var y = gp_sample_y_add_mean(mean, lz, n_star, n_samples)
    _ = kcross^
    _ = vtv^
    _ = kss^
    _ = cov^
    _ = factor^
    _ = z^
    _ = lz^
    return y^
