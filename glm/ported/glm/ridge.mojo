# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`ridgeFit`: l2-regularized least squares, cuML's `eig` solver.

PORT OF `cuml/cpp/src/glm/ridge.cuh` at cuML `00094f7`: `ridgeSolve`,
`ridgeEig`, `ridgeFit`. Partial. Do not improve.

WHAT THEIR `eig` SOLVER ACTUALLY IS, because the name undersells it
---------------------------------------------------------------------
`ridgeEig` (`ridge.cuh:109-135`) is NOT "add alpha to the eigenvalues of
A^T A and reuse lstsqEig", which is what `glm/UNPORTED.tsv` used to say it
was ("cheap once wanted"). It is

    svdEig(A) -> S, U, V        an SVD through the eigendecomposition of A^T A
                                (`raft/linalg/detail/svd.cuh`, ported beside
                                this file as `glm/ported/linalg/detail/svd.mojo`)
    ridgeSolve(S, V, U, b, alpha) -> w

and `ridgeSolve` (`ridge.cuh:41-77`) implements, in their own comment,

    w = V * inv(S^2 + alpha*I) * S * U^T * b

in SIX device ops, each a RAFT matrix primitive with its own threshold rule:

    setSmallValuesZero(S, thres=1e-10)            S <= 1e-10 (abs) -> 0
    S_nnz = S;  power(S_nnz)                      S_nnz = S^2
    addScalar(S_nnz, alpha)                       S_nnz = S^2 + alpha
    matrixVectorBinaryDivSkipZero(S, S_nnz, return_zero=TRUE)
                                                  S = S / S_nnz, 0 where |S_nnz| < 1e-10
    matrixVectorBinaryMult(V, S)                  V[:, j] *= S[j]
    S_nnz = U^T b                                 gemm, CUBLAS_OP_T
    w = V S_nnz                                   gemm

Algebraically `V diag(s/(s^2+a)) S^-1 V^T A^T b = (A^T A + a I)^-1 A^T b`,
the closed form; the route through U costs an extra `A V` pass (O(n d^2),
the same order as the Gram) plus `U^T b` (O(n d)), and it is THEIR route, so
it is the one ported. The shortcut would be a different program with a
different rounding at every step and a card nobody could align with theirs.

WHY THE ORDER OF OPS MATTERS FOR THE BITS: `U` is `A V / S` with `|S| <
1e-10` columns left UNDIVIDED (`svdEig`'s flag), then `U^T b` is a pinned
fold over rows (row 29, `xty_kernel`), then `V S_nnz` is a pinned gemv
(row 28). The two thresholds are ABSOLUTE on a SINGULAR VALUE -- which
scales with the data -- the identical deviation `OLS_NONZERO_THRESH`
records, carried rather than fixed (COPY-DO-NOT-IMPROVE) and gated.

THE DISPATCH, `ridge.cuh:210-218`, copied including its guard:

    if (algo == 0 || n_cols == 1) ridgeSVD      -> svdQR, NOT PORTED, raises by name
    else if (algo == 1)           ridgeEig      -> this file
    else                          ASSERT(false)

`ridgeEig` additionally ASSERTs `n_cols > 1` (`ridge.cuh:122`), which the
`n_cols == 1` arm of the dispatch already routed away. cuML's Python
`Ridge(solver='auto')` maps to `'eig'` = algo 1 (`ridge.pyx:304,312`), so the
Python default IS this arm, and the one-column case is the same refusal
`LinearRegression` carries (`ridge.pyx:355-359` switches to svd and says so).

NOT PORTED, refused by name: `fit_intercept`/`normalize` at this layer
(`preProcessData`/`postProcessData`, `preprocess.cuh`; the Python surface
centers on the host exactly as `LinearRegression` does, DEVIATION 517's
note), `sample_weight` (`ridge.cuh:197-208, 220-231`, a sqrt-scaling and its
inverse), `algo == 0` (`ridgeSVD` -> `raft::linalg::svdQR`, cuSOLVER gesvd,
no equivalent), and `n_alpha > 1` (their signature carries an array and
reads `alpha[0]` only: `ridge.cuh:67`).
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import STATS_TPB, xty_kernel
from core.gemm import gemv_n
from core.identity_trace import IdentityTrace
from glm.ported.linalg.detail.svd import svd_eig_traced
from glm.ported.matrix.math import (
    MATRIX_ELEM_TPB,
    add_scalar_kernel,
    matrix_vector_binary_div_skip_zero_kernel,
    matrix_vector_binary_mult_kernel,
    power_kernel,
    set_small_values_zero_kernel,
)


# `ridge.cuh:210-214`, their ids.
comptime RIDGE_ALGO_SVD = 0
comptime RIDGE_ALGO_EIG = 1

#: `ridgeSolve`'s `thres` (`ridge.cuh:61`), hoisted so the one place it is
#: written is the one place a check reads. ABSOLUTE, on a singular value;
#: see the module docstring and `glm/UNPORTED.tsv`.
comptime RIDGE_SMALL_THRESH = Float32(1.0e-10)


def _elem_grid(n: Int) -> Int:
    return (n + MATRIX_ELEM_TPB - 1) // MATRIX_ELEM_TPB


def ridge_solve_traced(
    ctx: DeviceContext,
    mut s: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut b: DeviceBuffer[DType.float32],
    alpha: Float32,
    mut w: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
) raises:
    """`ridgeSolve(handle, S, V, U, n_rows, n_cols, b, alpha, n_alpha, w)`.

    `s`, `v` are CONSUMED (overwritten in place exactly as theirs are);
    `u` is read. `alpha` is their `alpha[0]`.
    """
    var s_nnz = ctx.enqueue_create_buffer[DType.float32](n_cols)
    ctx.synchronize()

    # raft::matrix::setSmallValuesZero(S, n_cols, stream, thres)
    ctx.enqueue_function[set_small_values_zero_kernel](
        s.unsafe_ptr(), Int32(n_cols), RIDGE_SMALL_THRESH,
        grid_dim=(_elem_grid(n_cols), 1, 1), block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    # raft::copy(S_nnz, S); raft::matrix::power(S_nnz)  -- S_nnz = 1 * S * S
    ctx.enqueue_function[power_kernel](
        s_nnz.unsafe_ptr(), s.unsafe_ptr(), Int32(n_cols), Float32(1.0),
        grid_dim=(_elem_grid(n_cols), 1, 1), block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    # raft::linalg::addScalar(S_nnz, S_nnz, alpha[0])
    ctx.enqueue_function[add_scalar_kernel](
        s_nnz.unsafe_ptr(), Int32(n_cols), alpha,
        grid_dim=(_elem_grid(n_cols), 1, 1), block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    # matrixVectorBinaryDivSkipZero<false, true>(S, S_nnz, 1, n_cols, stream, TRUE)
    ctx.enqueue_function[matrix_vector_binary_div_skip_zero_kernel](
        s.unsafe_ptr(), s_nnz.unsafe_ptr(), Int32(1), Int32(n_cols), Int32(1),
        grid_dim=(_elem_grid(n_cols), 1, 1), block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, "ridge.solve.S_over", s, n_cols)
    _record_nnz(ctx, trace, s, n_cols)
    # matrixVectorBinaryMult<false, true>(V, S, n_cols, n_cols)
    ctx.enqueue_function[matrix_vector_binary_mult_kernel](
        v.unsafe_ptr(), s.unsafe_ptr(), Int32(n_cols), Int32(n_cols),
        grid_dim=(_elem_grid(n_cols * n_cols), 1, 1),
        block_dim=(MATRIX_ELEM_TPB, 1, 1),
    )
    # gemm(U, n_rows, n_cols, b, S_nnz, n_cols, 1, CUBLAS_OP_T, CUBLAS_OP_N):
    # S_nnz <- U^T b. A fold over rows, one block per column, pinned
    # (row 29's `xty_kernel`, the same kernel OLS's `A^T b` runs).
    ctx.enqueue_function[xty_kernel](
        s_nnz.unsafe_ptr(), u.unsafe_ptr(), b.unsafe_ptr(),
        Int32(n_rows), Int32(n_cols),
        grid_dim=(n_cols, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, "ridge.solve.Utb", s_nnz, n_cols)
    # gemm(V, n_cols, n_cols, S_nnz, w, n_cols, 1, N, N): w <- V S_nnz.
    # A matrix against one vector: the gemv (row 28).
    gemv_n(ctx, w, v, s_nnz, n_cols, n_cols)
    ctx.synchronize()
    _ = s_nnz^


def _record_nnz(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut s: DeviceBuffer[DType.float32],
    n_cols: Int,
) raises:
    """The card's INTEGER stage: how many singular directions survived the
    two thresholds (`S_over[j] != 0`). The same role as `ols.step4.rank`:
    a cross-vendor card that first differs here differs about how many
    directions the model HAS, which the differ reads before any float."""
    if not trace.enabled:
        return
    var hs = ctx.enqueue_create_host_buffer[DType.float32](n_cols)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=s)
    ctx.synchronize()
    var kept = 0
    for i in range(n_cols):
        if hs.unsafe_ptr().unsafe_load(i) != Float32(0.0):
            kept += 1
    var one = List[Int32]()
    one.append(Int32(kept))
    trace.record_list_i32("ridge.solve.nnz", one)
    _ = hs^


def ridge_eig_traced(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut b: DeviceBuffer[DType.float32],
    alpha: Float32,
    mut w: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
) raises:
    """`ridgeEig(handle, A, n_rows, n_cols, b, alpha, n_alpha, w)`,
    `ridge.cuh:109-135`: its two ASSERTs, `svdEig` with `gen_left_vec =
    true`, then `ridgeSolve`."""
    if n_cols <= 1:
        raise Error("ridgeEig: number of columns cannot be less than two")
    if n_rows <= 1:
        raise Error("ridgeEig: number of rows cannot be less than two")
    var s = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var v = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var u = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    ctx.synchronize()
    svd_eig_traced(ctx, a, n_rows, n_cols, s, u, v, True, trace, "ridge.svd")
    ridge_solve_traced(ctx, s, v, u, n_rows, n_cols, b, alpha, w, trace)
    ctx.synchronize()
    _ = s^
    _ = v^
    _ = u^


def ridge_fit(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    alpha: Float32,
    algo: Int = RIDGE_ALGO_EIG,
    fit_intercept: Bool = False,
    normalize: Bool = False,
) raises:
    """`ridgeFit`, untraced. One implementation of the guards: a DISABLED
    trace into `ridge_fit_traced`."""
    var off = IdentityTrace.disabled()
    ridge_fit_traced(
        ctx, a, b, w, n_rows, n_cols, alpha, off, algo, fit_intercept, normalize
    )


def ridge_fit_traced(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    alpha: Float32,
    mut trace: IdentityTrace,
    algo: Int = RIDGE_ALGO_EIG,
    fit_intercept: Bool = False,
    normalize: Bool = False,
) raises:
    """`ridgeFit(handle, input, n_rows, n_cols, labels, alpha, n_alpha, coef,
    intercept, fit_intercept, normalize, algo, sample_weight)`, their guards
    and their dispatch in their order, with the stage card.

    `algo` defaults to `RIDGE_ALGO_EIG` (theirs defaults the C++ door to 0,
    SVD, `ridge.cuh:166`; the Python door maps `'auto'` to eig, `ridge.pyx:
    304`). Same reasoning as `ols_fit`: defaulting to an arm that always
    raises makes the entry useless, and the Python door is the one a user
    comes in by. The intercept is `0` on the arm this ports (`ridge.cuh:247`).
    """
    # `ridge.cuh:173-174`
    if n_cols <= 0:
        raise Error("ridgeFit: number of columns cannot be less than one")
    if n_rows <= 1:
        raise Error("ridgeFit: number of rows cannot be less than two")
    if fit_intercept:
        raise Error(
            "ridgeFit: fit_intercept is not ported at this layer. It needs"
            " preProcessData and postProcessData from cuml glm/preprocess.cuh;"
            " python/mojolearn/linear_model.py centers on the host instead"
            " (DEVIATION 517's note). See glm/UNPORTED.tsv"
        )
    if normalize:
        raise Error(
            "ridgeFit: normalize is not ported, and theirs is only reachable"
            " with fit_intercept; see glm/UNPORTED.tsv"
        )
    if alpha < Float32(0.0):
        # `ridge.pyx:296-297`: "alpha must be non-negative". Their C++ layer
        # does not check; the Python layer a user reaches does.
        raise Error("ridgeFit: alpha must be non-negative")

    trace.record_device[DType.float32](ctx, "ridge.input.A", a, n_rows * n_cols)
    trace.record_device[DType.float32](ctx, "ridge.input.b", b, n_rows)
    trace.record_scalar_f32("ridge.input.alpha", alpha)

    # THE DISPATCH. `ridge.cuh:210-218`, copied including the `n_cols == 1`
    # override.
    if algo == RIDGE_ALGO_SVD or n_cols == 1:
        if n_cols == 1:
            raise Error(
                "ridgeFit: n_cols == 1 selects ridgeSVD (ridge.cuh:210),"
                " which is NOT PORTED (raft svdQR is cuSOLVER gesvd, no"
                " equivalent). cuML's Python layer forces the same switch"
                " (ridge.pyx:355). See glm/UNPORTED.tsv"
            )
        raise Error(
            "ridgeFit: algo 0 is ridgeSVD (raft::linalg::svdQR), which is"
            " NOT PORTED; solver='eig' is the ported arm. See glm/UNPORTED.tsv"
        )
    elif algo == RIDGE_ALGO_EIG:
        ridge_eig_traced(ctx, a, n_rows, n_cols, b, alpha, w, trace)
    else:
        raise Error("ridgeFit: no algorithm with this id has been implemented")
    trace.record_device[DType.float32](ctx, "ridge.coef", w, n_cols)
