# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`ls_success` and `ls_backtrack`: the backtracking line search.

PORT OF `cuml/cpp/src/glm/qn/qn_linesearch.cuh` at cuML `00094f7`. WHOLE
FILE since 2026-09-01: `LSProjectedStep` and `ls_backtrack_projected`,
OWL-QN's projected line search, are at the bottom (DEVIATION 552). Do not
improve.

THE ONE CONTRACTION CANDIDATE ON THE HOST. `ls_success`'s Armijo test
(`qn_linesearch.cuh:62`) is

    if (fx > fx_init + step * dg_test)

a multiply-add in HOST code, where a C++ host compiler may or may not
contract it (`-ffp-contract` is the host compiler's default, not nvcc's)
and Mojo's host codegen has been seen contracting across expressions
(`checks/numerics.mojo`). It decides whether a step is accepted. Under
IDENTICAL it is `identical_mul_add(step, dg_test, fx_init)` -- `fma`, one
rounding, the same on every host; under FAST the naive spelling. The other
scalars here (`ftol * dg_init`, `step *= width`, `wolfe * dg_init`) are
single operations.

`ls_backtrack`'s default path is ARMIJO (`LBFGSParam::linesearch`), so the
Wolfe `dot(grad, drt)` is ported and not reached from the Python surface.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from glm.impl.glm.qn.glm_base import GLMWithData
from glm.impl.glm.qn.glm_linear import nrm1
from glm.impl.glm.qn.qn_util import (
    LBFGS_LS_BT_ARMIJO,
    LBFGS_LS_BT_WOLFE,
    LBFGSParam,
    LS_INVALID_DIR,
    LS_INVALID_STEP,
    LS_INVALID_STEP_MAX,
    LS_INVALID_STEP_MIN,
    LS_MAX_ITERS_REACHED,
    LS_SUCCESS,
    project_orth,
)
from glm.impl.glm.qn.simple_mat.dense import VEC_ELEM_TPB, axpy, dot
from checks.numerics import ftz, identical_mul_add


def ls_success(
    ctx: DeviceContext,
    param: LBFGSParam,
    fx_init: Float32,
    dg_init: Float32,
    fx: Float32,
    dg_test: Float32,
    step: Float32,
    mut grad: DeviceBuffer[DType.float32],
    mut drt: DeviceBuffer[DType.float32],
    n: Int,
    mut width: Float32,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Bool:
    """`ls_success`, `qn_linesearch.cuh:49-86`."""
    if fx > identical_mul_add(step, dg_test, fx_init):
        width = param.ls_dec
    else:
        # Armijo condition is met
        if param.linesearch == LBFGS_LS_BT_ARMIJO:
            return True
        var dg = dot(ctx, grad, drt, n, scalar)
        if dg < param.wolfe * dg_init:
            width = param.ls_inc
        else:
            # Regular Wolfe condition is met
            if param.linesearch == LBFGS_LS_BT_WOLFE:
                return True
            if dg > -param.wolfe * dg_init:
                width = param.ls_dec
            else:
                # Strong Wolfe condition is met
                return True
    return False


def ls_backtrack(
    ctx: DeviceContext,
    param: LBFGSParam,
    mut f: GLMWithData,
    mut fx: Float32,
    mut x: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut step: Float32,
    mut drt: DeviceBuffer[DType.float32],
    mut xp: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
    mut ls_iters: Int,
) raises -> Int:
    """`ls_backtrack`, `qn_linesearch.cuh:109-146`. `ls_iters` reports how
    many candidates were evaluated (for the card)."""
    if step <= Float32(0.0):
        return LS_INVALID_STEP
    var fx_init = fx
    var dg_init = dot(ctx, grad, drt, n, scalar)
    if dg_init > Float32(0.0):
        return LS_INVALID_DIR
    var dg_test = param.ftol * dg_init
    var width = Float32(0.0)
    ls_iters = 0
    for _ in range(param.max_linesearch):
        # x_{k+1} = x_k + step * d_k
        axpy(ctx, x, step, drt, xp, n)
        fx = f.evaluate(ctx, x, grad)
        ls_iters += 1
        if ls_success(
            ctx, param, fx_init, dg_init, fx, dg_test, step, grad, drt, n,
            width, scalar,
        ):
            return LS_SUCCESS
        if step < param.min_step:
            return LS_INVALID_STEP_MIN
        if step > param.max_step:
            return LS_INVALID_STEP_MAX
        step *= width
    return LS_MAX_ITERS_REACHED


# ===========================================================================
# OWL-QN'S PROJECTED LINE SEARCH (`qn_linesearch.cuh:17-39`, `:148-197`)
# ===========================================================================
#
# DEVIATION 552. Two things differ from `ls_backtrack` above and both are
# about staying inside one orthant:
#
#   1. THE STEP IS PROJECTED. `x = proj_orth(xp + step * drt, xi)` where
#      `xi = xp == 0 ? -pg : xp` -- the reference orthant is the current
#      point's, or, for a coordinate sitting exactly at zero, the one the
#      pseudo-gradient wants to move into. A coordinate that would cross
#      zero is CLAMPED TO ZERO instead. That clamp is where an l1 fit's
#      exact zeros come from; without it the iterate would step over the
#      kink in `|w|` and the sparsity would never appear.
#   2. THE ARMIJO TEST USES THE PSEUDO-GRADIENT. `dg_init` is
#      `dot(pseudo_grad, drt)`, not `dot(grad, drt)`, because `f_wrap`'s
#      value includes the l1 term while `grad` is the gradient of the LOSS
#      ALONE (`qn_solvers.cuh:322-323` says so in a comment). Comparing a
#      value that includes the penalty against a slope that does not is the
#      obvious way to get this wrong.
#
# `ls_success` is reused unchanged: upstream calls the SAME function and
# passes `pseudo_grad` in its `grad` parameter (`qn_linesearch.cuh:186-187`).


def owlqn_objective(
    ctx: DeviceContext,
    mut f: GLMWithData,
    mut x: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    l1_penalty: Float32,
    pg_limit: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Float32:
    """`min_owlqn`'s `f_wrap` lambda, `qn_solvers.cuh:308-313`:

        T tmp = f(x, grad, dev_scalar, stream);
        SimpleVec<T> mask(x.data, pg_limit);
        return tmp + l1_penalty * nrm1(mask, dev_scalar, stream);

    **The value carries the l1 term and `grad` does not.** That asymmetry is
    deliberate upstream and is commented there ("fx is loss+regularizer,
    grad is grad of loss only", `:322-323`); the pseudo-gradient is what
    stands in for the missing piece.

    `nrm1` is taken over the FIRST `pg_limit` entries, the weight block, so
    the intercept is not penalized. `nrm1` already takes a length, so the
    mask is that argument and no sub-buffer is made.

    It lives in this module and not in `qn_solvers.mojo` because Mojo has no
    closure to hand across files and `ls_backtrack_projected` below is its
    other caller. Two roundings on the host, both flushed (row 10).
    """
    var tmp = f.evaluate(ctx, x, grad)
    var pen = nrm1(ctx, x, pg_limit, scalar)
    return ftz(tmp + ftz(l1_penalty * pen))


def projected_step_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    drt: MutPointer[Float32, MutAnyOrigin],
    pgrad: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    step: Float32,
):
    """`LSProjectedStep::op_pstep` under `assign_ternary`
    (`qn_linesearch.cuh:20-39`):

        xi = xp == 0 ? -pg : xp
        x  = project_orth(xp + step * drt, xi)

    `xp + step * drt` is row 9's contraction exactly -- the same expression
    `axpy_kernel` pins -- so it goes through `identical_mul_add`. One thread
    per entry, no fold."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var xpi = xp.unsafe_load(i)
        var xi = (
            ftz(-pgrad.unsafe_load(i)) if xpi == Float32(0.0) else xpi
        )
        var moved = ftz(identical_mul_add(step, drt.unsafe_load(i), xpi))
        x.unsafe_store(i, project_orth(moved, xi))


def projected_step(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    step: Float32,
    mut drt: DeviceBuffer[DType.float32],
    mut xp: DeviceBuffer[DType.float32],
    mut pgrad: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`LSProjectedStep::operator()(step, x, drt, xp, pgrad, stream)`."""
    ctx.enqueue_function[projected_step_kernel](
        x.unsafe_ptr(), xp.unsafe_ptr(), drt.unsafe_ptr(),
        pgrad.unsafe_ptr(), Int32(n), step,
        grid_dim=((n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
        block_dim=(VEC_ELEM_TPB, 1, 1),
    )


def ls_backtrack_projected(
    ctx: DeviceContext,
    param: LBFGSParam,
    mut f: GLMWithData,
    mut fx: Float32,
    mut x: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut pseudo_grad: DeviceBuffer[DType.float32],
    mut step: Float32,
    mut drt: DeviceBuffer[DType.float32],
    mut xp: DeviceBuffer[DType.float32],
    l1_penalty: Float32,
    pg_limit: Int,
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
    mut ls_iters: Int,
) raises -> Int:
    """`ls_backtrack_projected`, `qn_linesearch.cuh:148-197`. `ls_iters`
    reports how many candidates were evaluated (for the card), as
    `ls_backtrack` does."""
    if step <= Float32(0.0):
        return LS_INVALID_STEP
    var fx_init = fx
    # `dot(pseudo_grad, drt)`, NOT `dot(grad, drt)`. See the banner.
    var dg_init = dot(ctx, pseudo_grad, drt, n, scalar)
    if dg_init > Float32(0.0):
        return LS_INVALID_DIR
    var dg_test = param.ftol * dg_init
    var width = Float32(0.0)
    ls_iters = 0
    for _ in range(param.max_linesearch):
        # x_{k+1} = proj_orth(x_k + step * d_k)
        projected_step(ctx, x, step, drt, xp, pseudo_grad, n)
        # evaluates fx WITH the l1 term, but only grad of the loss term
        fx = owlqn_objective(ctx, f, x, grad, l1_penalty, pg_limit, scalar)
        ls_iters += 1
        if ls_success(
            ctx, param, fx_init, dg_init, fx, dg_test, step, pseudo_grad,
            drt, n, width, scalar,
        ):
            return LS_SUCCESS
        if step < param.min_step:
            return LS_INVALID_STEP_MIN
        if step > param.max_step:
            return LS_INVALID_STEP_MAX
        step *= width
    return LS_MAX_ITERS_REACHED
