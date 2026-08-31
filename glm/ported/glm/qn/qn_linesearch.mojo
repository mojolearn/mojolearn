# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`ls_success` and `ls_backtrack`: the backtracking line search.

PORT OF `cuml/cpp/src/glm/qn/qn_linesearch.cuh` at cuML `00094f7`. Partial:
`LSProjectedStep` and `ls_backtrack_projected` (OWL-QN's) are not ported.
Do not improve.

THE ONE CONTRACTION CANDIDATE ON THE HOST. `ls_success`'s Armijo test
(`qn_linesearch.cuh:62`) is

    if (fx > fx_init + step * dg_test)

a multiply-add in HOST code, where a C++ host compiler may or may not
contract it (`-ffp-contract` is the host compiler's default, not nvcc's)
and Mojo's host codegen has been seen contracting across expressions
(`mojo_only/numerics.mojo`). It decides whether a step is accepted. Under
IDENTICAL it is `identical_mul_add(step, dg_test, fx_init)` -- `fma`, one
rounding, the same on every host; under FAST the naive spelling. The other
scalars here (`ftol * dg_init`, `step *= width`, `wolfe * dg_init`) are
single operations.

`ls_backtrack`'s default path is ARMIJO (`LBFGSParam::linesearch`), so the
Wolfe `dot(grad, drt)` is ported and not reached from the Python surface.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from glm.ported.glm.qn.glm_base import GLMWithData
from glm.ported.glm.qn.qn_util import (
    LBFGS_LS_BT_ARMIJO,
    LBFGS_LS_BT_WOLFE,
    LBFGSParam,
    LS_INVALID_DIR,
    LS_INVALID_STEP,
    LS_INVALID_STEP_MAX,
    LS_INVALID_STEP_MIN,
    LS_MAX_ITERS_REACHED,
    LS_SUCCESS,
)
from glm.ported.glm.qn.simple_mat.dense import axpy, dot
from mojo_only.numerics import identical_mul_add


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
