# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Classical inference on the host, for a box with no GPU (the classical host
inference lane, 2026-09-13; brief
docs/lanes/BRIEF_forest_host_inference_2026-09-13.md, "Classical lanes").

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
and the GPU bindings do not import this file. It exists because the five
inference entries of `bindings/_mojolearn_estimators.mojo` that
LinearRegression, Ridge, TruncatedSVD, LogisticRegression and PCA call
(`ols_predict`, `tsvd_transform`, `qn_decision_function`, `qn_sigmoid`,
`pca_transform`) all reach kernels in files that import `max.gpu.host` at
module level (`core/gemm.mojo:8`, `core/column_stats.mojo`,
`glm/impl/qn/glm_base.mojo`, `glm/estimator.mojo:37-38`), while the
arithmetic each kernel performs is a handful of `checks/numerics.mojo`
calls, which is GPU-free.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS. Every function below names the
kernel it mirrors and keeps its statements in its order:

  `host_pinned_cell`       `pinned_gemm_nt_kernel`, `core/gemm.mojo:33-62`,
                           one output cell; `pinned_gemv_n_kernel`,
                           `core/gemm.mojo:93-113`, is the same loop over one
                           row, so `gemm_nt` (`:137`, which routes `n == 1`
                           to `gemv_n`, `:322`) and `gemv_n` are one cell
                           function here. Under IDENTICAL, `gemm_nt` launches
                           the pinned kernel unless the build defines
                           MOJOLEARN_537_GEMM_IDENT_SWAP, which no build script
                           does (`git grep 537_GEMM_IDENT_SWAP` finds only
                           bench/speed/gemm_speed_main.mojo's opt-in).
  `host_gemm_nt`           the kernel's grid, every cell in row-major order.
  `host_ols_predict`       `ols_predict_host`, `glm/estimator.mojo:192-231`:
                           the gemv, then `_add_scalar_kernel` (`:53-74`)
                           only when the intercept is not exactly 0.0.
  `host_qn_decision`       `linear_fwd` at C == 1, `glm/impl/qn/glm_base.mojo:
                           198-244`: the gemv over the first D entries of W,
                           then `add_bias_kernel` (`:121-134`) when
                           fit_intercept, whose bias read is NOT flushed.
  `host_qn_sigmoid`        `qn_sigmoid_host`, `glm/estimator.mojo:384-410`,
                           which already runs on the host and is relocated
                           here because its file imports `std.gpu`.
  `host_pca_transform`     `pca_transform`, `decomposition/impl/linalg/detail/
                           pca.mojo:319-355`: `shift_columns_kernel` with
                           sign -1.0 (`core/column_stats.mojo:153-204`), then
                           `gemm_nt`. The restoring shift (+1.0) is applied to
                           the device copy of X, which `pca_transform_host`
                           never reads back, so it changes no output and is
                           not restated.
  `host_tsvd_transform`    `tsvd_transform_host`, `decomposition/estimator.mojo:
                           265-281`: one `gemm_nt`.

The restatement is a prediction until measured. tools/classical_host_gate.py
is the measurement, and the brief records what it has shown.
"""
from std.sys.compile import is_defined

from checks.numerics import ftz, identical_exp64, identical_mul_add


#: The gate's negative control, the phase 1 host bindings' define
#: (`gemm/checks/gemm_oracle.mojo:90`, `kde/checks/kde_oracle.mojo:86`), so
#: one -D MOJOLEARN_HOST_SABOTAGE=1 build sabotages every host arithmetic in
#: `_mojolearn_estimators_host`. A build with it walks every dot product's
#: k loop DESCENDING, which is wrong on purpose (a serial float32 fold in
#: the other order is a different bit pattern on almost every row), and
#: the gate must say so. Read back by `estimators_host_sabotage`.
comptime CLASSICAL_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def host_pinned_cell(
    x: List[Float32], x_off: Int, y: List[Float32], y_off: Int, k: Int,
) -> Float32:
    """One cell of `pinned_gemm_nt_kernel` (`core/gemm.mojo:52-62`):

        var acc = Float32(0.0)
        for p in range(k):
            acc = ftz(identical_mul_add(ftz(x[i*k+p]), ftz(y[j*k+p]), acc))
        z[cell] = ftz(Float32(0.0) + ftz(acc))

    `x_off` is `i * k`, `y_off` is `j * k`. The final `0.0 + acc` turns a
    `-0.0` accumulator into `+0.0`, and is kept for that reason."""
    var acc = Float32(0.0)
    comptime if CLASSICAL_HOST_SABOTAGE:
        # THE SABOTAGE ARM: the same fold, walked DESCENDING. Wrong on
        # purpose; see CLASSICAL_HOST_SABOTAGE.
        for q in range(k):
            var p = k - 1 - q
            acc = ftz(identical_mul_add(ftz(x[x_off + p]), ftz(y[y_off + p]), acc))
        return ftz(Float32(0.0) + ftz(acc))
    for p in range(k):
        acc = ftz(identical_mul_add(ftz(x[x_off + p]), ftz(y[y_off + p]), acc))
    return ftz(Float32(0.0) + ftz(acc))


def host_gemm_nt(
    x: List[Float32], y: List[Float32], m: Int, n: Int, k: Int,
) -> List[Float32]:
    """`z[m x n] = x[m x k] . y[n x k]^T`, `gemm_nt` under IDENTICAL
    (`core/gemm.mojo:137-160`): `pinned_gemm_nt_kernel` over `m * n` cells,
    `i = cell // n`, `j = cell % n`; `n == 1` is `pinned_gemv_n_kernel`, the
    same fold over the one column."""
    var z = List[Float32](length=m * n, fill=Float32(0.0))
    for cell in range(m * n):
        var i = cell // n
        var j = cell % n
        z[cell] = host_pinned_cell(x, i * k, y, j * k, k)
    return z^


def host_ols_predict(
    x: List[Float32], coef: List[Float32], n_rows: Int, n_features: Int,
    intercept: Float32,
) -> List[Float32]:
    """`ols_predict_host` (`glm/estimator.mojo:192-231`): `gemv_n`, then
    `_add_scalar_kernel` (`:53-74`, `dst[i] = ftz(ftz(dst[i]) + ftz(value))`)
    only `if intercept != Float32(0.0)`, the host compare that file audits
    under DEVIATION 527."""
    var out = host_gemm_nt(x, coef, n_rows, 1, n_features)
    if intercept != Float32(0.0):
        for i in range(n_rows):
            var v = ftz(out[i])
            out[i] = ftz(v + ftz(intercept))
    return out^


def host_qn_decision(
    x: List[Float32], w: List[Float32], n_rows: Int, n_features: Int,
    fit_intercept: Bool,
) -> List[Float32]:
    """`qn_decision_function_host` (`glm/estimator.mojo:353-381`) at the
    binary logistic shape, `n_targets = 1`: `linear_fwd`'s C == 1 arm
    (`glm/impl/qn/glm_base.mojo:230-244`), `gemv_n` over `w[0:D]` (the
    `w_head` copy), then `add_bias_kernel` (`:121-134`) when
    `fit_intercept`: `z[i] = ftz(z[i] + b)` with `b = w[D]` read as it is,
    NOT flushed (that kernel flushes the sum only)."""
    var w_head = List[Float32](length=n_features, fill=Float32(0.0))
    for j in range(n_features):
        w_head[j] = w[j]
    var z = host_gemm_nt(x, w_head, n_rows, 1, n_features)
    if fit_intercept:
        var b = w[n_features]
        for i in range(n_rows):
            z[i] = ftz(z[i] + b)
    return z^


def host_qn_sigmoid(
    scores: List[Float32], n_rows: Int,
) -> List[Float64]:
    """`qn_sigmoid_host` (`glm/estimator.mojo:384-410`), DEVIATION 549:
    `p = 1 / (1 + identical_exp64(-Float64(z)))` in float64, `1 - p` for
    class 0, `p` for class 1, `(n_rows, 2)` row-major."""
    var out = List[Float64](length=2 * n_rows, fill=Float64(0.0))
    for i in range(n_rows):
        var z = Float64(scores[i])
        var p = 1.0 / (1.0 + identical_exp64(-z))
        out[2 * i] = 1.0 - p
        out[2 * i + 1] = p
    return out^


def host_center(
    x: List[Float32], mu: List[Float32], n_rows: Int, n_cols: Int,
    sign: Float32,
) -> List[Float32]:
    """`shift_columns_kernel` (`core/column_stats.mojo:153-204`), one local
    per flush as that kernel spells it:

        var xv = ftz(x[idx]); var mv = ftz(mu[col])
        x[idx] = ftz(xv + sign * mv)

    Returns the shifted copy; the kernel shifts the device copy in place."""
    var out = List[Float32](length=n_rows * n_cols, fill=Float32(0.0))
    for idx in range(n_rows * n_cols):
        var col = idx % n_cols
        var xv = ftz(x[idx])
        var mv = ftz(mu[col])
        out[idx] = ftz(xv + sign * mv)
    return out^


def host_pca_transform(
    x: List[Float32], mu: List[Float32], components: List[Float32],
    n_rows: Int, n_cols: Int, n_components: Int,
) -> List[Float32]:
    """`pca_transform` (`decomposition/impl/linalg/detail/pca.mojo:319-355`):
    center with sign `-1.0`, then `gemm_nt(out, x, components, n_rows,
    n_components, n_cols)`."""
    var centered = host_center(x, mu, n_rows, n_cols, Float32(-1.0))
    return host_gemm_nt(centered, components, n_rows, n_components, n_cols)


def host_tsvd_transform(
    x: List[Float32], components: List[Float32],
    n_rows: Int, n_cols: Int, n_components: Int,
) -> List[Float32]:
    """`tsvd_transform_host` (`decomposition/estimator.mojo:265-281`): one
    `gemm_nt(out, x, components, n_rows, n_components, n_features)`."""
    return host_gemm_nt(x, components, n_rows, n_components, n_cols)
