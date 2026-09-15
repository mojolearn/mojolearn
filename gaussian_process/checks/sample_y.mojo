# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`GaussianProcessRegressor.sample_y(X, n_samples, random_state)`
(2026-09-15): the stream, the covariance assembly and the refusals that the
device estimator (`gaussian_process/estimator.mojo::gpr_sample_y_host`) and
the host verifier (`gaussian_process/host/sample_y_oracle.mojo`) share.
Host code only; nothing here launches a kernel.

Reference (semantics): scikit-learn `GaussianProcessRegressor.sample_y`
(`sklearn/gaussian_process/_gpr.py`): `y_mean, y_cov = predict(X,
return_cov=True)`, then `rng.multivariate_normal(y_mean, y_cov,
n_samples).T`, shape `(n_star, n_samples)` for one target. With
`normalize_y` both the mean and the covariance are un-normalized first.

DEVIATION 2793: THE POSTERIOR DRAW. Each step is named with the pieces it
reuses (DEVIATION 2792's rule: no refactor of an existing factor or kernel).

  mean     `K_trans @ alpha_`, `predict`'s own: the cross-covariance
           `k(X_train, X)` stored `n_train x n_star` (DEVIATION 1758), then
           the identical GEMM at `OP_TN`.
  V        `L^-1 K_trans^T`, `predict`'s own `trsm_lower` in place.
  V^T V    the identical GEMM at `OP_TN` with `A = B = V`
           (`m = n = n_star`, `k = n_train`), the full matrix whose diagonal
           `predict` folds (DEVIATION 1759 is lifted for this call only).
  K**      `k(X, X)` through `gp_kernel_matrix` with `is_self` TRUE, so a
           WhiteKernel adds its noise on the diagonal, as scikit-learn's
           `kernel_(X)` does in `predict(return_cov=True)`.
  C        on the HOST, the lower triangle `C[i, j] = ftz(ftz(K**[i, j]) -
           ftz(VtV[i, j]))` for `j <= i`, row ascending, then MIRRORED to the
           upper triangle, so `C` is symmetric by construction and the
           Cholesky profile's symmetry refusal cannot fire on a last-bit
           GEMM asymmetry. No clamp: scikit-learn's `predict` clamps only the
           variance vector, never `y_cov`.
  L_C      the repository's identical Cholesky of `C` with the profile's
           pinned jitter `2^-20` (`chol_jitter_pinned`, DEVIATION 1637),
           panel order `potrf_lower`. scikit-learn's `multivariate_normal`
           takes an SVD with no jitter; the jitter and the triangular factor
           are the deviation. A failed factor (`info != 0`) is REFUSED BY
           NAME, never a partial draw (DEVIATION 1634's rule).
  Z        `n_star x n_samples` standard normals, DEVIATION 2791's
           construction with its own tag: key = `random_state` as UInt64
           (low word, high word); draw column `s` (one sample vector),
           entry `i` (one test point) reads Box-Muller pair `q = i // 2` from
           `philox4x32_10` with `ctr = (s low, s high, q, GP_SAMPLE_Y_TAG)`,
           word 0 then word 1 mapped `Float32(w >> 8) * 2^-24`, word 0
           through `km_guard_unit`, then `km_boxmuller_pair(u1, u2, 1, 0)`;
           `i` even takes the first value, `i` odd the second. A sample
           vector's entries therefore do not depend on `n_samples`.
  L_C Z    the identical GEMM at `OP_NN` (`m = n_star`, `n = n_samples`,
           `k = n_star`), the factor's strict upper triangle read as its
           stored `+0.0`.
  y        on the HOST, `ftz(ftz(mean[i]) + ftz((L_C Z)[i, s]))`.

THE SOLVE ORDER, in one line: kernel cross, mean, trsm, V^T V, K**, C on
the host, factor, Z on the host, L_C Z, mean add on the host. The mean is
taken before the trsm because the trsm overwrites the cross-covariance.

`normalize_y` (python/mojolearn/_gp_impl.py): the draw above is made in the
normalized scale, then every value goes through `predict`'s own
un-normalization, `ftz(round32(ftz(round32(std * v)) + y_mean))`. That is
`y_mean + std * mean + std * L_C z`, so the covariance is `std^2 C`, and the
jitter stays `2^-20` in the normalized scale.

The draws are the same bits on every vendor, on the CPU verifier and at
every launch. They are NOT scikit-learn's bits: the stream, the factor and
the float32 width all differ, so `sample_y` is MEANING-COMPATIBLE (the same
distribution, the same shape) and BIT-DIFFERENT from the reference.
"""

from std.memory import bitcast

from checks.numerics import ftz
from core.philox import philox4x32_10
from kernel_methods.impl.random.rng_device import km_boxmuller_pair, km_guard_unit


#: `ctr[3]` of the sample_y normals, ASCII "GPSY". DEVIATION 2793. Distinct
#: from GaussianMixture.sample's "COMP" and "SAMP" and from DEVIATION 1733's 0.
comptime GP_SAMPLE_Y_TAG: UInt32 = 0x47505359
#: `2^-24` as float32 bits, DEVIATION 1733's scale.
comptime GP_SAMPLE_Y_TWO_POW_M24_BITS: UInt32 = 0x33800000


@always_inline
def gp_sample_y_key(seed: UInt64) -> SIMD[DType.uint32, 2]:
    return SIMD[DType.uint32, 2](
        UInt32(seed & 0xFFFFFFFF), UInt32((seed >> 32) & 0xFFFFFFFF)
    )


@always_inline
def gp_sample_y_unit(w: UInt32) -> Float32:
    """`Float32(w >> 8) * 2^-24`: exact, a 24-bit integer times a power of
    two."""
    return Float32(Int(w >> UInt32(8))) * bitcast[DType.float32](
        GP_SAMPLE_Y_TWO_POW_M24_BITS
    )


def gp_sample_y_normal(key: SIMD[DType.uint32, 2], s: Int, i: Int) -> Float32:
    """The standard normal of sample `s`, test point `i` (DEVIATION 2793)."""
    var ctr = SIMD[DType.uint32, 4](
        UInt32(s & 0xFFFFFFFF),
        UInt32((s >> 32) & 0xFFFFFFFF),
        UInt32((i // 2) & 0xFFFFFFFF),
        GP_SAMPLE_Y_TAG,
    )
    var draw = philox4x32_10(ctr, key)
    var u1 = km_guard_unit(gp_sample_y_unit(draw[0]))
    var u2 = gp_sample_y_unit(draw[1])
    var pair = km_boxmuller_pair(u1, u2, Float32(1.0), Float32(0.0))
    if i % 2 == 0:
        return pair[0]
    return pair[1]


def gp_sample_y_normals(n_star: Int, n_samples: Int, seed: UInt64) -> List[Float32]:
    """`Z`, `n_star x n_samples` row-major: `Z[i, s]`."""
    var key = gp_sample_y_key(seed)
    var z = List[Float32](length=n_star * n_samples, fill=Float32(0.0))
    for i in range(n_star):
        for s in range(n_samples):
            z[i * n_samples + s] = ftz(gp_sample_y_normal(key, s, i))
    return z^


def gp_sample_y_validate(info: Int, n_star: Int, n_samples: Int) raises:
    """The refusals before any launch, in this order on both paths."""
    if info != 0:
        raise Error(
            "gpr_sample_y: refusing to sample from a FAILED fit (info="
            + String(info)
            + "). The factor's columns from "
            + String(info - 1)
            + " onward are unfinished. DEVIATION 1634"
        )
    if n_star <= 0:
        raise Error(
            "gpr_sample_y: X must have at least one row, got " + String(n_star)
        )
    if n_samples < 1:
        raise Error(
            "gpr_sample_y: n_samples must be at least 1, got "
            + String(n_samples)
        )


def gp_sample_y_covariance(
    kss: List[Float32], vtv: List[Float32], n: Int
) -> List[Float32]:
    """`C = K** - V^T V`, the lower triangle row ascending, mirrored to the
    upper (DEVIATION 2793)."""
    var c = List[Float32](length=n * n, fill=Float32(0.0))
    for i in range(n):
        for j in range(i + 1):
            var v = ftz(ftz(kss[i * n + j]) - ftz(vtv[i * n + j]))
            c[i * n + j] = v
            c[j * n + i] = v
    return c^


def gp_sample_y_check_factor(info: Int, n: Int) raises:
    """A posterior covariance that does not factor is refused by name."""
    if info != 0:
        raise Error(
            "gpr_sample_y: the posterior covariance of the "
            + String(n)
            + " query rows did not factor (info="
            + String(info)
            + ": the leading minor of order "
            + String(info)
            + " of K(X, X) - V^T V + 2^-20 I is not positive definite)."
            " There is no partial draw to return. The usual cause is query"
            " rows that repeat training rows or each other under a kernel"
            " with no WhiteKernel term; add one, or sample fewer, distinct"
            " rows. DEVIATION 2793"
        )


def gp_sample_y_add_mean(
    mean: List[Float32], lz: List[Float32], n_star: Int, n_samples: Int
) -> List[Float32]:
    """`y[i, s] = ftz(ftz(mean[i]) + ftz((L_C Z)[i, s]))`."""
    var y = List[Float32](length=n_star * n_samples, fill=Float32(0.0))
    for i in range(n_star):
        for s in range(n_samples):
            y[i * n_samples + s] = ftz(ftz(mean[i]) + ftz(lz[i * n_samples + s]))
    return y^
