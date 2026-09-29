# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 5110, REVISED 2026-09-29: the M-step means and covariances fold
the SAMPLE axis through the identical GEMM, as `mixture/` does (its DEVIATION
1729), not one row-ascending float32 chain per cell.

WHY. A cell of the covariance is a sum of n row terms. One float32 chain over
n rows loses the small directions of a nearly singular covariance: on
Istella-S (100,000 x 200 fit rows, few-valued columns) the chain made the
covariance fail its Cholesky where the GEMM fold did not, with everything
else equal: our GaussianMixture through this lane's chain refused at
reg_covar 1e-4 and 1e-3, through the GEMM fold it fitted at both (m2pro,
2026-09-29, tools/gmm_istella_probe.py GMM_PROBE_DIAG), and our
BayesianGaussianMixture refused at 3e-3 where scikit-learn's fitted
(scikit-learn forms these moments in float64 for float32 input). The GEMM's
fold shape is the gemm lane's, already certified across vendors; this lane
inherits it rather than inventing one.

THE ARITHMETIC, `mixture/host/gmm_host_oracle.mojo`'s and the device kernels'
(`mixture/checks/mstep.mojo` `means_divide_kernel`, `center_scale_kernel`,
`cov_finish_kernel`), in their order:

    means   = (resp^T . X) / nk                    OP_TN, k-axis n
    diff_k  = X - means[k];  scaled_k = resp[:, k] * diff_k
    cov_k   = (scaled_k^T . diff_k) / nk[k], + reg on the diagonal after

`nk` keeps its own row-ascending chain (`bodies.nk_cell`, 5110's first part).
This module is the host column's and the host oracle's one spelling;
`host_gemm_oracle` equals the device's `identical_gemm_into` bit for bit."""
from checks.numerics import ftz, identical_div, identical_mul
from cluster.host.host_gemm_cells import host_gemm_oracle
from gemm.host.identical_gemm import OP_TN


def gemm_fold_means(
    resp: List[Float32], x: List[Float32], nk: List[Float32], n: Int, d: Int, kc: Int
) -> List[Float32]:
    """means (kc x d) = (resp^T . X) / nk."""
    var raw = host_gemm_oracle(resp, x, OP_TN, kc, d, n)
    var means = List[Float32](length=kc * d, fill=Float32(0))
    for idx in range(kc * d):
        var k = idx // d
        means[idx] = ftz(identical_div(ftz(raw[idx]), ftz(nk[k])))
    return means^


def gemm_fold_cov(
    resp: List[Float32], x: List[Float32], means: List[Float32], nk: List[Float32], n: Int, d: Int, kc: Int,
    reg: Float32,
) -> List[Float32]:
    """cov (kc x d x d): per component (scaled^T . diff) / nk, + reg on the diagonal."""
    var dd = d * d
    var cov = List[Float32](length=kc * dd, fill=Float32(0))
    var diff = List[Float32](length=n * d, fill=Float32(0))
    var scaled = List[Float32](length=n * d, fill=Float32(0))
    for k in range(kc):
        for i in range(n):
            var r = ftz(resp[i * kc + k])
            for j in range(d):
                var idx = i * d + j
                var dv = ftz(ftz(x[idx]) - ftz(means[k * d + j]))
                diff[idx] = dv
                scaled[idx] = ftz(identical_mul(r, dv))
        var rc = host_gemm_oracle(scaled, diff, OP_TN, d, d, n)
        for idx in range(dd):
            var v = ftz(rc[idx])
            v = ftz(identical_div(v, ftz(nk[k])))
            if idx // d == idx % d:
                v = ftz(v + reg)
            cov[k * dd + idx] = v
    return cov^
