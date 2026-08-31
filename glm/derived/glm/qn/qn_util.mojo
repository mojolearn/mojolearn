# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""`LBFGSParam`, the return codes, `check_convergence`, `lbfgs_search_dir`.

PORT OF `cuml/cpp/src/glm/qn/qn_util.cuh` at cuML `00094f7`. Partial:
`project_orth`, `get_pseudo_grad`, `op_project`, `op_pseudo_grad` (OWL-QN's
operators) are not ported. Do not improve.

THE HOST SCALARS ARE FLOAT32 AND EVERY ONE OF THEM STEERS A BRANCH
-----------------------------------------------------------------
`T = float` on the Python float32 path, so `ys`, `yy`, `alpha[j]`, `beta`,
`step`, `fmag` are HOST Float32 and the comparisons below -- the skipping
test `ys <= eps * yy`, the convergence test `gnorm <= epsilon * fmag`, the
insufficient-change test -- are branches on them. The inputs to every one
are device reductions that `simple_mat/dense.mojo` pins (DEVIATION 547);
the host arithmetic between them is single IEEE operations (`/`, `*`, `-`,
`<=`, `max`, `abs`), which round the same on every host, plus one
contraction candidate handled in `qn_linesearch.mojo`. So under IDENTICAL
the branch sequence -- and therefore the ITERATION COUNT and the L-BFGS
history -- is a function of the inputs alone, and the count is recorded on
the card (`qn.n_iter`) as the certificate's integer stage.

`std::numeric_limits<T>::epsilon()` for float is `2^-23`; written as the
literal below and not derived.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from glm.derived.glm.qn.simple_mat.dense import (
    ax,
    ax_inplace,
    axpy_inplace,
    dot,
    squared_norm,
)
from glm.derived.linear_model.qn import QNParams


# `LINE_SEARCH_ALGORITHM`, `qn_util.cuh:30-35`
comptime LBFGS_LS_BT_ARMIJO = 1
comptime LBFGS_LS_BT = 2
comptime LBFGS_LS_BT_WOLFE = 2
comptime LBFGS_LS_BT_STRONG_WOLFE = 3

# `LINE_SEARCH_RETCODE`, `:37-44`
comptime LS_SUCCESS = 0
comptime LS_INVALID_STEP_MIN = 1
comptime LS_INVALID_STEP_MAX = 2
comptime LS_MAX_ITERS_REACHED = 3
comptime LS_INVALID_DIR = 4
comptime LS_INVALID_STEP = 5

# `OPT_RETCODE`, `:46-52`
comptime OPT_SUCCESS = 0
comptime OPT_NUMERIC_ERROR = 1
comptime OPT_LS_FAILED = 2
comptime OPT_MAX_ITERS_REACHED = 3
comptime OPT_INVALID_ARGS = 4

#: `std::numeric_limits<float>::epsilon()`
comptime FLOAT_EPSILON = Float32(1.1920928955078125e-7)


@fieldwise_init
struct LBFGSParam(ImplicitlyCopyable, Copyable, Movable):
    """`LBFGSParam<T>` (`qn_util.cuh:54-122`) with T = float."""

    var m: Int
    var epsilon: Float32
    var past: Int
    var delta: Float32
    var max_iterations: Int
    var linesearch: Int
    var max_linesearch: Int
    var min_step: Float32
    var max_step: Float32
    var ftol: Float32
    var wolfe: Float32
    var ls_dec: Float32
    var ls_inc: Float32

    @staticmethod
    def defaults() -> Self:
        """The default constructor, `qn_util.cuh:71-86`. NOTE `linesearch =
        LBFGS_LS_BT_ARMIJO`: cuML's shipped line search is backtracking
        Armijo; the Wolfe arms of `ls_success` exist and are not reached
        from the Python door, which never sets this field."""
        return Self(
            6, Float32(1e-5), 0, Float32(0.0), 0, LBFGS_LS_BT_ARMIJO, 20,
            Float32(1e-20), Float32(1e20), Float32(1e-4), Float32(0.9),
            Float32(0.5), Float32(2.1),
        )

    @staticmethod
    def from_params(pams: QNParams) -> Self:
        """`explicit LBFGSParam(const qn_params&)`, `qn_util.cuh:88-98`:
        `m = lbfgs_memory`, `epsilon = grad_tol`, `past = change_tol > 0 ?
        10 : 0`, `delta = change_tol`, `max_iterations = max_iter`,
        `max_linesearch = linesearch_max_iter`, `ftol = change_tol > 0 ?
        change_tol * 0.1 : 1e-4`. The products are double then narrowed,
        as `T(pams.change_tol * 0.1)` is."""
        var p = Self.defaults()
        p.m = pams.lbfgs_memory
        p.epsilon = Float32(pams.grad_tol)
        p.past = 10 if pams.change_tol > 0.0 else 0
        p.delta = Float32(pams.change_tol)
        p.max_iterations = pams.max_iter
        p.max_linesearch = pams.linesearch_max_iter
        p.ftol = Float32(pams.change_tol * 0.1) if pams.change_tol > 0.0 else Float32(1e-4)
        return p^

    def check_param(self) -> Int:
        """`check_param`, `qn_util.cuh:100-121`: 0 if valid, else the 1-based
        index of the first failing test."""
        var ret = 1
        if self.m <= 0:
            return ret
        ret += 1
        if self.epsilon <= Float32(0.0):
            return ret
        ret += 1
        if self.past < 0:
            return ret
        ret += 1
        if self.delta < Float32(0.0):
            return ret
        ret += 1
        if self.max_iterations < 0:
            return ret
        ret += 1
        if self.linesearch < LBFGS_LS_BT_ARMIJO or self.linesearch > LBFGS_LS_BT_STRONG_WOLFE:
            return ret
        ret += 1
        if self.max_linesearch <= 0:
            return ret
        ret += 1
        if self.min_step < Float32(0.0):
            return ret
        ret += 1
        if self.max_step < self.min_step:
            return ret
        ret += 1
        if self.ftol <= Float32(0.0) or self.ftol >= Float32(0.5):
            return ret
        ret += 1
        if self.wolfe <= self.ftol or self.wolfe >= Float32(1.0):
            return ret
        ret += 1
        return 0


def check_convergence(
    param: LBFGSParam,
    k: Int,
    fx: Float32,
    gnorm: Float32,
    mut fx_hist: List[Float32],
) -> Bool:
    """`check_convergence`, `qn_util.cuh:147-169`."""
    var fmag = max(fx, param.epsilon)
    if gnorm <= param.epsilon * fmag:
        return True
    if param.past > 0:
        if k >= param.past and abs(fx_hist[k % param.past] - fx) <= param.delta * fmag:
            return True
        fx_hist[k % param.past] = fx
    return False


def lbfgs_search_dir(
    ctx: DeviceContext,
    param: LBFGSParam,
    mut n_vec: Int,
    end_prev: Int,
    mut S: List[DeviceBuffer[DType.float32]],
    mut Y: List[DeviceBuffer[DType.float32]],
    mut g: DeviceBuffer[DType.float32],
    mut drt: DeviceBuffer[DType.float32],
    mut yhist: List[Float32],
    mut alpha: List[Float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Int:
    """`lbfgs_search_dir`, `qn_util.cuh:176-241`: `drt = -H g` by the
    two-loop recursion over the `S`, `Y` history. `svec`/`yvec` are
    `S[end_prev]`/`Y[end_prev]`, which the caller just wrote
    (`col_ref(S, svec, end)`). Returns the new `end`."""
    var end = end_prev
    var ys = dot(ctx, S[end], Y[end], n, scalar)
    var yy = squared_norm(ctx, Y[end], n, scalar)
    # Skipping test (`:190-206`): the Hessian is ~0, keep the direction.
    if ys <= FLOAT_EPSILON * yy:
        return end
    n_vec += 1
    yhist[end] = ys

    ax(ctx, drt, Float32(-1.0), g, n)
    var bound = min(param.m, n_vec)
    end = (end + 1) % param.m
    var j = end
    for _ in range(bound):
        j = (j + param.m - 1) % param.m
        alpha[j] = dot(ctx, S[j], drt, n, scalar) / yhist[j]
        axpy_inplace(ctx, drt, -alpha[j], Y[j], n)

    ax_inplace(ctx, drt, ys / yy, n)

    for _ in range(bound):
        var beta = dot(ctx, Y[j], drt, n, scalar) / yhist[j]
        axpy_inplace(ctx, drt, alpha[j] - beta, S[j], n)
        j = (j + 1) % param.m

    return end
