# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""OLS and ridge TRAINING on the host, for a box with no GPU (workstream E,
the lanes ols and ridge, 2026-09-14).

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu` or a `DeviceContext`,
and no GPU binding imports this file. The Gram and the Jacobi are the PCA
host restatement (`decomposition/host/pca_oracle.mojo`, the same two
stages `lstsq_eig` and `svd_eig` reach on the device), the small dense
products are `core/classical_host_predict.mojo::host_gemm_nt` (the pinned
`gemm_nt` and `gemv_n` cell), and every elementwise kernel is spelled a
second time here from `core/column_stats.mojo` and
`glm/impl/matrix/math.mojo`, line for line.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS.

  `host_xty`               `xty_kernel`, `core/column_stats.mojo:207`: one
                           STATS_TPB block per column, lane `t` chains rows
                           `t, t + STATS_TPB, ...` through
                           `identical_mul_add` with no flush inside, the
                           halving tree of `pinned_block_sum`, the total
                           flushed and stored.
  `host_equilibration_scale`
                           `ols_equilibration_scale`, `glm/impl/linalg/
                           detail/lstsq.mojo` (DEVIATION 2620): integer
                           arithmetic on the bits, copied.
  `host_pinv_threshold`    `ols_pinv_threshold` (DEVIATION 2621):
                           `Float32(Float64(n) * eps32 * Float64(max_abs))`.
  `host_lstsq_eig`         `lstsq_eig_traced`, `lstsq.mojo:269`, in its
                           order: `gemm_tn` (the split-K Gram), `xty`, the
                           diagonal, the equilibration scales, `G <- S G S`
                           by `row_vector_binary_mult_kernel` then
                           `matrix_vector_binary_mult_kernel` (`ftz(a * b)`
                           each, row scale first), `Ab <- S Ab`
                           (`vector_binary_mult_kernel`), the Jacobi at the
                           device's settings with the convergence refusal
                           in its words, `max|lam|` under a strict `>`, the
                           threshold, `divide_columns_by_nonzero_kernel`
                           (`ftz(q / lam)` where `lam > thresh or lam <
                           -thresh`, else 0), `inv <- QS Q^T` by `gemm_nt`,
                           `w <- inv Ab` by `gemv_n`, `w <- S w`.
  `host_ols_fit`           `ols_fit_host` then `ols_fit_weighted_traced`,
                           `glm/impl/ols.mojo:298`, unweighted: the guards
                           in their words, the dispatch (`n_cols >
                           n_rows` is `lstsq_min_norm`, which this file
                           does NOT restate and refuses BY NAME; `n_cols ==
                           1` and every other shape is `lstsq_eig`).
  `host_svd_eig`           `svd_eig_traced`, `glm/impl/linalg/detail/svd.mojo:
                           89`: the Gram, the Jacobi, the descending
                           selection sort of indices under a strict `>`
                           (`_descending_order`), the gathered basis and
                           its transpose, `seq_root_kernel` with
                           `set_neg_zero` (`a < 0 -> 0`, else
                           `ftz(identical_sqrt(ftz(a * 1.0)))`), `U <- A V`
                           by `gemm_nt(u, a, vt)`, then
                           `matrix_vector_binary_div_skip_zero_kernel` at
                           `return_zero = 0` (`|s| < 1e-10` leaves the
                           column as is, else `ftz(u / s)`).
  `host_ridge_solve`       `ridge_solve_traced`, `glm/impl/ridge.mojo:69`:
                           `set_small_values_zero_kernel` (`a <= thres and
                           -a <= thres -> 0`, a NaN kept), `power_kernel`
                           (`sa = ftz(1.0 * a)`, `ftz(sa * a)`),
                           `add_scalar_kernel` (`ftz(s + alpha)`),
                           `matrix_vector_binary_div_skip_zero_kernel` at
                           `return_zero = 1` on the row vector `S`, `V[:, j]
                           *= S[j]` (`ftz(v * s)`), `U^T b` by `xty`, `w <-
                           V S_nnz` by `gemv_n`.
  `host_ridge_fit`         `ridge_fit_host` then `ridge_fit_traced` and
                           `ridge_eig_traced`, `ridge.mojo:189, 240`: the
                           guards and the dispatch in their words.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` reaches this file
through the PCA oracle's Gram reduce (chunks walked descending) and
`host_gemm_nt`'s cell (the k loop walked descending), so every
coefficient moves; nothing here needs an arm of its own. Read back by
`estimators_host_sabotage`.

The restatement is a prediction until measured. The CPU identity gate
(`tools/identity_break.py --diff <3 GPU columns> <cpu json> --lanes
ols,ridge --require-columns 4`) is the measurement.
"""
from std.memory import bitcast

from checks.numerics import ftz, identical_mul_add, identical_sqrt
from core.classical_host_predict import host_gemm_nt
from decomposition.host.pca_oracle import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    STATS_TPB,
    host_gemm_tn,
    host_halving_sum,
    host_jacobi_eigh,
)


#: `OLS_PINV_EPS32`, `numpy.finfo(numpy.float32).eps`, 2^-23 exactly.
comptime OLS_PINV_EPS32 = Float64(1.1920928955078125e-07)

#: `RIDGE_SMALL_THRESH` (`ridge.mojo:63`) and the `1e-10` of
#: `matrix_vector_binary_div_skip_zero_kernel`, one literal each there.
comptime RIDGE_SMALL_THRESH = Float32(1.0e-10)
comptime DIV_SKIP_ZERO_THRESH = Float32(1.0e-10)


def host_xty(
    x: List[Float32], y: List[Float32], n_rows: Int, n_cols: Int,
) -> List[Float32]:
    """`xty_kernel`: `A^T b`, one STATS_TPB block per column."""
    var out = List[Float32](length=n_cols, fill=Float32(0.0))
    for col in range(n_cols):
        var partials = List[Float32](length=STATS_TPB, fill=Float32(0.0))
        for t in range(STATS_TPB):
            var acc = Float32(0.0)
            var r = t
            while r < n_rows:
                acc = identical_mul_add(x[r * n_cols + col], y[r], acc)
                r += STATS_TPB
            partials[t] = acc
        out[col] = ftz(host_halving_sum(partials))
    return out^


def host_equilibration_scale(diag: Float32) -> Float32:
    """`ols_equilibration_scale`, DEVIATION 2620, copied."""
    var bits = bitcast[DType.uint32](diag)
    var field = Int((bits >> 23) & UInt32(0xFF))
    var frac = bits & UInt32(0x7FFFFF)
    if (bits >> 31) != UInt32(0) or field == 255:
        return Float32(1.0)
    if field == 0 and frac == UInt32(0):
        return Float32(1.0)
    var e = field - 127
    if field == 0:
        var width = 0
        var f = frac
        while f != UInt32(0):
            f = f >> 1
            width += 1
        e = (width - 1) - 149
    var k = (e + 1) // 2 if e >= 0 else -((-e) // 2)
    return bitcast[DType.float32](UInt32(127 - k) << 23)


def host_pinv_threshold(max_abs_eig: Float32, n: Int) -> Float32:
    """`ols_pinv_threshold`, DEVIATION 2621."""
    return Float32(Float64(n) * OLS_PINV_EPS32 * Float64(max_abs_eig))


def _diagonal(a: List[Float32], n: Int) -> List[Float32]:
    """`diagonal_to_vector_kernel`: a copy, no arithmetic."""
    var out = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        out[i] = a[i * n + i]
    return out^


def host_lstsq_eig(
    a: List[Float32], b: List[Float32], n_rows: Int, n_cols: Int,
) raises -> List[Float32]:
    """`lstsq_eig_traced` without the DeviceContext: `w = inv(A^T A) A^T b`
    through the equilibrated pseudo-inverse."""
    # covA <- A^T A, Ab <- A^T b.
    var cov = host_gemm_tn(a, n_cols, n_rows)
    var ab = host_xty(a, b, n_rows, n_cols)

    # Step 2b: S from the diagonal's bits; G <- S G S (rows first, then
    # columns, each `ftz(a * b)`); Ab <- S Ab.
    var diag0 = _diagonal(cov, n_cols)
    var scale = List[Float32](length=n_cols, fill=Float32(0.0))
    for i in range(n_cols):
        scale[i] = host_equilibration_scale(diag0[i])
    var cells = n_cols * n_cols
    for idx in range(cells):
        cov[idx] = ftz(cov[idx] * scale[idx // n_cols])
    for idx in range(cells):
        cov[idx] = ftz(cov[idx] * scale[idx % n_cols])
    for i in range(n_cols):
        ab[i] = ftz(ab[i] * scale[i])

    # Q S Q* <- covA.
    var jac = host_jacobi_eigh(cov, n_cols, JACOBI_SWEEPS, Float32(JACOBI_TOL))
    var q = jac.vectors.copy()
    var s_vec = _diagonal(cov, n_cols)
    if not jac.converged:
        raise Error(
            "lstsq_eig: the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps on the "
            + String(n_cols)
            + " x "
            + String(n_cols)
            + " Gram matrix; ||offdiag(A^T A)||_F / ||A^T A||_F is still "
            + String(jac.rel)
            + ". A rank-deficient or badly scaled design produces this."
        )

    # Step 3b: the relative cutoff.
    var max_abs = Float32(0.0)
    for i in range(n_cols):
        var mag = abs(s_vec[i])
        if mag > max_abs:
            max_abs = mag
    var thresh = host_pinv_threshold(max_abs, n_cols)

    # QS <- Q invS, with DivideByNonZero.
    var qs = List[Float32](length=cells, fill=Float32(0.0))
    for idx in range(cells):
        var col = idx % n_cols
        var lam = s_vec[col]
        if lam > thresh or lam < -thresh:
            qs[idx] = ftz(q[idx] / lam)
        else:
            qs[idx] = Float32(0.0)

    # inv <- QS Q^T, then w <- inv Ab, then w <- S w.
    var inv = host_gemm_nt(qs, q, n_cols, n_cols, n_cols)
    var w = host_gemm_nt(inv, ab, n_cols, 1, n_cols)
    for i in range(n_cols):
        w[i] = ftz(w[i] * scale[i])
    return w^


def host_ols_fit(
    a: List[Float32], b: List[Float32], n_rows: Int, n_cols: Int,
) raises -> List[Float32]:
    """`ols_fit_host` through `ols_fit_weighted_traced`'s guards and
    dispatch, unweighted (the Python layer applies sample weights on the
    host before the call, `linear_model.py`), `fit_intercept = False`."""
    if n_cols <= 0:
        raise Error("olsFit: number of columns cannot be less than one")
    if n_rows <= 1:
        raise Error("olsFit: number of rows cannot be less than two")
    if n_cols > n_rows:
        raise Error(
            "olsFit: n_cols > n_rows selects lstsq_min_norm (DEVIATION 550),"
            " which has no CPU restatement yet; the host OLS fit covers the"
            " tall shape (lstsq_eig) only."
        )
    return host_lstsq_eig(a, b, n_rows, n_cols)


@fieldwise_init
struct SVDHostResult(Movable):
    """`svd_eig`'s three outputs: `s` (n_cols, descending), `u` (n_rows x
    n_cols) and `v` (n_cols x n_cols, right singular vectors as COLUMNS)."""

    var s: List[Float32]
    var u: List[Float32]
    var v: List[Float32]


def host_svd_eig(
    a: List[Float32], n_rows: Int, n_cols: Int,
) raises -> SVDHostResult:
    """`svd_eig_traced` with `gen_left_vec = True`."""
    var cov = host_gemm_tn(a, n_cols, n_rows)
    var jac = host_jacobi_eigh(cov, n_cols, JACOBI_SWEEPS, Float32(JACOBI_TOL))
    var v_raw = jac.vectors.copy()
    var s_raw = _diagonal(cov, n_cols)
    if not jac.converged:
        raise Error(
            "svdEig: the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps on the "
            + String(n_cols)
            + " x "
            + String(n_cols)
            + " Gram matrix; ||offdiag||_F / ||.||_F is still "
            + String(jac.rel)
            + ". eigDC aborts here too (raft eig.cuh:149)."
        )

    # `_descending_order`: a selection sort of indices, strict `>`.
    var order = List[Int]()
    for i in range(n_cols):
        order.append(i)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            var vj = s_raw[order[j]]
            var vi = s_raw[order[i]]
            if vj > vi:
                var t = order[i]
                order[i] = order[j]
                order[j] = t

    # `gather_columns_kernel` and `gather_vector_kernel`: data movement.
    var cells = n_cols * n_cols
    var v = List[Float32](length=cells, fill=Float32(0.0))
    var vt = List[Float32](length=cells, fill=Float32(0.0))
    for idx in range(cells):
        var p = idx // n_cols
        var j = idx % n_cols
        var val = v_raw[p * n_cols + order[j]]
        v[p * n_cols + j] = val
        vt[j * n_cols + p] = val
    var s = List[Float32](length=n_cols, fill=Float32(0.0))
    for i in range(n_cols):
        s[i] = s_raw[order[i]]

    # `seq_root_kernel(s, 1.0, set_neg_zero = 1)`.
    for i in range(n_cols):
        var av = s[i]
        if av < Float32(0.0):
            s[i] = Float32(0.0)
        else:
            var p = ftz(av * Float32(1.0))
            s[i] = ftz(identical_sqrt(p))

    # U <- A V by `gemm_nt(u, a, vt, n_rows, n_cols, n_cols)`, then
    # `matrix_vector_binary_div_skip_zero_kernel` at return_zero = 0.
    var u = host_gemm_nt(a, vt, n_rows, n_cols, n_cols)
    for idx in range(n_rows * n_cols):
        var bv = s[idx % n_cols]
        if abs(bv) < DIV_SKIP_ZERO_THRESH:
            pass
        else:
            u[idx] = ftz(u[idx] / bv)
    return SVDHostResult(s^, u^, v^)


def host_ridge_solve(
    mut s: List[Float32],
    mut v: List[Float32],
    u: List[Float32],
    n_rows: Int,
    n_cols: Int,
    b: List[Float32],
    alpha: Float32,
) -> List[Float32]:
    """`ridge_solve_traced`: `s` and `v` are consumed in place, as theirs."""
    # setSmallValuesZero(S, thres)
    for i in range(n_cols):
        var av = s[i]
        if av <= RIDGE_SMALL_THRESH and -av <= RIDGE_SMALL_THRESH:
            s[i] = Float32(0.0)
    # S_nnz = power(S, 1.0); S_nnz += alpha
    var s_nnz = List[Float32](length=n_cols, fill=Float32(0.0))
    for i in range(n_cols):
        var av = s[i]
        var sa = ftz(Float32(1.0) * av)
        s_nnz[i] = ftz(sa * av)
    for i in range(n_cols):
        s_nnz[i] = ftz(s_nnz[i] + alpha)
    # matrixVectorBinaryDivSkipZero(S, S_nnz, 1, n_cols, return_zero = TRUE)
    for i in range(n_cols):
        var bv = s_nnz[i]
        if abs(bv) < DIV_SKIP_ZERO_THRESH:
            s[i] = Float32(0.0)
        else:
            s[i] = ftz(s[i] / bv)
    # matrixVectorBinaryMult(V, S): column j times S[j]
    for idx in range(n_cols * n_cols):
        v[idx] = ftz(v[idx] * s[idx % n_cols])
    # S_nnz <- U^T b (xty), w <- V S_nnz (gemv)
    var utb = host_xty(u, b, n_rows, n_cols)
    return host_gemm_nt(v, utb, n_cols, 1, n_cols)


def host_ridge_fit(
    a: List[Float32], b: List[Float32], n_rows: Int, n_cols: Int, alpha: Float32,
) raises -> List[Float32]:
    """`ridge_fit_host` through `ridge_fit_traced`'s guards and dispatch at
    `RIDGE_ALGO_EIG`, `fit_intercept = False`, then `ridge_eig_traced`."""
    if n_cols <= 0:
        raise Error("ridgeFit: number of columns cannot be less than one")
    if n_rows <= 1:
        raise Error("ridgeFit: number of rows cannot be less than two")
    if alpha < Float32(0.0):
        raise Error("ridgeFit: alpha must be non-negative")
    if n_cols == 1:
        raise Error(
            "ridgeFit: n_cols == 1 selects ridgeSVD (ridge.cuh:210),"
            " which is NOT IMPLEMENTED (raft svdQR is cuSOLVER gesvd, no"
            " equivalent). cuML's Python layer forces the same switch"
            " (ridge.pyx:355). See glm/NOT_IMPLEMENTED.tsv"
        )
    # ridgeEig's two ASSERTs.
    if n_cols <= 1:
        raise Error("ridgeEig: number of columns cannot be less than two")
    if n_rows <= 1:
        raise Error("ridgeEig: number of rows cannot be less than two")
    var svd = host_svd_eig(a, n_rows, n_cols)
    var s = svd.s.copy()
    var v = svd.v.copy()
    return host_ridge_solve(s, v, svd.u, n_rows, n_cols, b, alpha)
