# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`LBFGSParam`, the return codes, `check_convergence`, `lbfgs_search_dir`.

PORT OF `cuml/cpp/src/glm/qn/qn_util.cuh` at cuML `00094f7`. WHOLE FILE
since 2026-09-01: `project_orth`, `get_pseudo_grad`, `op_project` and
`op_pseudo_grad` -- OWL-QN's four operators -- are ported at the bottom,
with the two device kernels that apply them elementwise and their host
wrappers. Do not improve.

WHY THE KERNELS ARE HERE AND NOT IN `simple_mat/dense.mojo`. Upstream these
are functors passed to `SimpleVec::assign_binary`, so the arithmetic lives
in `qn_util.cuh` and the loop lives in `dense.hpp`. Here the loop is a
launch, and a launch that calls `get_pseudo_grad` has to be in a module that
can see it. `qn_util.mojo` already imports `simple_mat/dense.mojo`, so
putting the kernels in `dense.mojo` would close an import cycle. They are
placed beside the operator they apply, which is also where a reader looking
for `op_pseudo_grad` will look.

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

from std.gpu import block_dim, block_idx, thread_idx

from glm.impl.glm.qn.simple_mat.dense import (
    VEC_ELEM_TPB,
    ax,
    ax_inplace,
    axpy_inplace,
    copy_vec,
    dot,
    squared_norm,
)
from glm.impl.linear_model.qn import QNParams
from checks.numerics import ftz


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


# ===========================================================================
# OWL-QN'S OPERATORS (`qn_util.cuh:131-134`, `:237-261`)
# ===========================================================================
#
# The l1 solver never differentiates `|w|`. It works with a PSEUDO-GRADIENT,
# which is the subgradient of the l1 term chosen to point downhill, and it
# keeps every step inside the orthant the current point is in by projecting.
# Those are the two operators below and they are the whole of what makes
# OWL-QN different from L-BFGS. DEVIATION 552.
#
# BOTH ARE DISCRETE FUNCTIONS OF A FLOAT, which is IDENTITY_PATHS row 32's
# class and not a rounding: `project_orth` returns exactly 0 or exactly `x`
# on a sign test, and `get_pseudo_grad` picks one of four expressions on
# `x != 0`, `dmins > 0`, `dplus < 0`. A last-bit disagreement at one of
# those boundaries does not move a coefficient's fifth decimal, it moves
# WHICH ORTHANT the iterate is in and therefore how many entries of the
# solution are exactly zero -- the sparsity pattern, which is the thing an
# l1 fit is asked for. That is why every operand is flushed through `ftz`
# before it is compared and why the l1 arm's identity claim is about a
# SPARSITY PATTERN as well as about bits.


@always_inline
def project_orth(x: Float32, y: Float32) -> Float32:
    """`project_orth(x, y)`, `qn_util.cuh:131-134`: `x * y <= 0 ? 0 : x`.

    Note `<=`, so a product of exactly zero projects to zero, and note that
    a NaN product fails the test and returns `x` unchanged, which is theirs.
    """
    return Float32(0.0) if ftz(x * y) <= Float32(0.0) else x


@always_inline
def get_pseudo_grad(x: Float32, dlossx: Float32, c: Float32) -> Float32:
    """`get_pseudo_grad(x, dlossx, C)`, `qn_util.cuh:236-245`.

        if (x != 0) return dlossx + sgn(x) * C;
        dplus = dlossx + C; dmins = dlossx - C;
        if (dmins > 0) return dmins;
        if (dplus < 0) return dplus;
        return 0;

    `raft::sgn` returns an **int** (`raft/core/math.hpp:706-710`,
    `(T(0) < val) - (val < T(0))`), so `sgn(x) * C` is an exact +C or -C and
    the add is one rounding. It is written that way here rather than as a
    copysign because `sgn` of a NaN is 0 on their side, which makes the
    first branch `dlossx + 0`, and a copysign would not.
    """
    if x != Float32(0.0):
        var sgn = (
            Float32(1.0) if Float32(0.0) < x
            else (Float32(-1.0) if x < Float32(0.0) else Float32(0.0))
        )
        return ftz(dlossx + ftz(sgn * c))
    var dplus = ftz(dlossx + c)
    var dmins = ftz(dlossx - c)
    if dmins > Float32(0.0):
        return dmins
    if dplus < Float32(0.0):
        return dplus
    return Float32(0.0)


def pseudo_grad_kernel(
    pseudo: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    l1: Float32,
):
    """`op_pseudo_grad` under `assign_binary` (`qn_util.cuh:255-261`,
    `dense.hpp`): `pseudo[i] = get_pseudo_grad(x[i], grad[i], l1)`.

    One thread per entry, no fold, so the block width is SCHEDULING."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        pseudo.unsafe_store(
            i, ftz(get_pseudo_grad(x.unsafe_load(i), grad.unsafe_load(i), l1))
        )


def project_neg_kernel(
    drt: MutPointer[Float32, MutAnyOrigin],
    pseudo: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`drt.assign_binary(drt, pseudo, op_project(-1.0))`
    (`qn_solvers.cuh:393`, `qn_util.cuh:247-253`):
    `drt[i] = project_orth(drt[i], -1 * pseudo[i])`.

    In place on `drt`, which is what their call site is (`drt` is both the
    first operand and the destination). The `-1 *` is exact."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var y = ftz(Float32(-1.0) * pseudo.unsafe_load(i))
        drt.unsafe_store(i, project_orth(drt.unsafe_load(i), y))


def update_pseudo(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    l1: Float32,
    pg_limit: Int,
    mut pseudo: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`update_pseudo`, `qn_solvers.cuh:245-259`.

        if (grad.len > pg_limit) { pseudo = grad; mask(pg_limit) = op(x, grad) }
        else                     { pseudo = op(x, grad) over all n }

    THE BRANCH IS NOT AN OPTIMIZATION, IT IS WHERE THE BIAS ESCAPES THE
    PENALTY. `pg_limit` is `loss.D * loss.C` (`qn_solvers.cuh:447`), the
    weight block, while `n` is `n_param = (D + fit_intercept) * C`. With an
    intercept the two differ by one entry per class, and that entry gets the
    RAW loss gradient copied straight through rather than a pseudo-gradient
    -- i.e. the intercept is not l1-penalized, matching
    `glm_regularizer.cuh:45`, where it is not l2-penalized either.
    """
    if n > pg_limit:
        copy_vec(ctx, pseudo, grad)
        ctx.enqueue_function[pseudo_grad_kernel](
            pseudo.unsafe_ptr(), x.unsafe_ptr(), grad.unsafe_ptr(),
            Int32(pg_limit), l1,
            grid_dim=((pg_limit + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
            block_dim=(VEC_ELEM_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[pseudo_grad_kernel](
            pseudo.unsafe_ptr(), x.unsafe_ptr(), grad.unsafe_ptr(),
            Int32(n), l1,
            grid_dim=((n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
            block_dim=(VEC_ELEM_TPB, 1, 1),
        )
    ctx.synchronize()


def project_direction(
    ctx: DeviceContext,
    mut drt: DeviceBuffer[DType.float32],
    mut pseudo: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """"Project drt onto orthant of -pseudog", `qn_solvers.cuh:392-393`."""
    ctx.enqueue_function[project_neg_kernel](
        drt.unsafe_ptr(), pseudo.unsafe_ptr(), Int32(n),
        grid_dim=((n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
        block_dim=(VEC_ELEM_TPB, 1, 1),
    )
    ctx.synchronize()
