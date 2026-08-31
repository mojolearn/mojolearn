# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`min_lbfgs`, `update_and_check`, `qn_minimize`: the L-BFGS driver.

PORT OF `cuml/cpp/src/glm/qn/qn_solvers.cuh` at cuML `00094f7`. Partial:
`min_owlqn` and `update_pseudo` (the l1 solver) are NOT ported and
`qn_minimize` RAISES by name on `l1 != 0` where theirs dispatches to them.
Do not improve.

THE LOOP, `min_lbfgs` (`qn_solvers.cuh:136-227`), in their order:

    fx = f(x, grad); gnorm = nrmMax(grad)
    converged at x0?                 -> OPT_SUCCESS, k = 0
    drt = -grad; step = 1 / nrm2(drt); fxp = fx
    for k = 1 .. max_iterations:
        xp = x; gradp = grad; fxp = fx
        lsret = ls_backtrack(...)    updates x, fx, grad, step
        gnorm = nrmMax(grad)
        update_and_check(...)        may stop (success / ls failed / numeric)
                                     and on a failed step RESTORES x, grad, fx
        S[end] = x - xp; Y[end] = grad - gradp
        end = lbfgs_search_dir(...)  drt = -H grad
        step = 1
    -> OPT_MAX_ITERS_REACHED

Every branch is on a host Float32 the ported reductions produced; see
`qn_util.mojo`'s header for why that makes the ITERATION COUNT part of the
certificate. The card records, per iteration, `qn.iterNNNN.loss`,
`qn.iterNNNN.grad` and `qn.iterNNNN.ls` (the line-search return code and
its evaluation count, two integers), and at the end `qn.n_iter` and
`qn.retcode`. DEVIATION 548.

`update_and_check` (`:73-133`) is copied including its three-way verdict
on a non-critical line-search failure: `isLsInDoubt` accepts a step whose
objective did not grow past `fxp + ftol`, and stops if it did not improve
either. Their `CUML_LOG_*` calls are dropped; nothing else is.
"""

from std.math import isinf, isnan
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from glm.derived.glm.qn.glm_base import GLMWithData
from glm.derived.glm.qn.qn_linesearch import ls_backtrack
from glm.derived.glm.qn.qn_util import (
    LBFGSParam,
    LS_INVALID_STEP_MIN,
    LS_MAX_ITERS_REACHED,
    LS_SUCCESS,
    OPT_LS_FAILED,
    OPT_MAX_ITERS_REACHED,
    OPT_NUMERIC_ERROR,
    OPT_SUCCESS,
    check_convergence,
    lbfgs_search_dir,
)
from glm.derived.glm.qn.simple_mat.dense import ax, axpy, copy_vec, nrm2


def _iter_tag(k: Int) -> String:
    """`qn.iterNNNN`, zero padded so the tags sort and align."""
    var s = String(k)
    while s.byte_length() < 4:
        s = "0" + s
    return "qn.iter" + s


def update_and_check(
    ctx: DeviceContext,
    param: LBFGSParam,
    iter: Int,
    lsret: Int,
    mut fx: Float32,
    fxp: Float32,
    gnorm: Float32,
    mut x: DeviceBuffer[DType.float32],
    mut xp: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut gradp: DeviceBuffer[DType.float32],
    mut fx_hist: List[Float32],
    mut outcode: Int,
) raises -> Bool:
    """`update_and_check`, `qn_solvers.cuh:73-133`. Returns `stop`."""
    var stop = False
    var converged = False
    var is_ls_valid = (not isnan(fx)) and (not isinf(fx))
    var is_ls_non_critical = lsret == LS_INVALID_STEP_MIN or lsret == LS_MAX_ITERS_REACHED
    var is_ls_in_doubt = is_ls_valid and fx <= fxp + param.ftol and is_ls_non_critical
    var is_ls_success = lsret == LS_SUCCESS or is_ls_in_doubt

    if is_ls_valid:
        converged = check_convergence(param, iter, fx, gnorm, fx_hist)

    if (not is_ls_success) and (not converged):
        outcode = OPT_LS_FAILED
        stop = True
    elif not is_ls_valid:
        outcode = OPT_NUMERIC_ERROR
        stop = True
    elif converged:
        outcode = OPT_SUCCESS
        stop = True
    elif is_ls_in_doubt and fx + param.ftol >= fxp:
        outcode = OPT_LS_FAILED
        stop = True

    # if linesearch wasn't successful, undo the update.
    if (not is_ls_success) or (not is_ls_valid):
        fx = fxp
        copy_vec(ctx, x, xp)
        copy_vec(ctx, grad, gradp)
    return stop


def min_lbfgs(
    ctx: DeviceContext,
    param: LBFGSParam,
    mut f: GLMWithData,
    mut x: DeviceBuffer[DType.float32],
    mut fx: Float32,
    mut k: Int,
    n: Int,
    mut trace: IdentityTrace,
) raises -> Int:
    """`min_lbfgs`, `qn_solvers.cuh:136-227`. `x` holds the initial point
    and the result; `fx` the final objective; `k` the iteration count.
    Returns the `OPT_RETCODE`."""
    if param.check_param() != 0:
        raise Error(
            "L-BFGS: invalid parameter (check_param code "
            + String(param.check_param()) + ")"
        )
    # SETUP WORKSPACE (`:146-164`): S, Y as `param.m` column buffers each.
    var S = List[DeviceBuffer[DType.float32]]()
    var Y = List[DeviceBuffer[DType.float32]]()
    for _ in range(param.m):
        S.append(ctx.enqueue_create_buffer[DType.float32](n))
        Y.append(ctx.enqueue_create_buffer[DType.float32](n))
    var xp = ctx.enqueue_create_buffer[DType.float32](n)
    var grad = ctx.enqueue_create_buffer[DType.float32](n)
    var gradp = ctx.enqueue_create_buffer[DType.float32](n)
    var drt = ctx.enqueue_create_buffer[DType.float32](n)
    var scalar = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()

    var ys = List[Float32]()
    var alpha = List[Float32]()
    for _ in range(param.m):
        ys.append(Float32(0.0))
        alpha.append(Float32(0.0))
    var fx_hist = List[Float32]()
    for _ in range(param.past if param.past > 0 else 0):
        fx_hist.append(Float32(0.0))

    k = 0
    # Evaluate function and compute gradient
    fx = f.evaluate(ctx, x, grad)
    var gnorm = f.grad_norm(ctx, grad)
    trace.record_scalar_f32("qn.init.loss", fx)
    trace.record_device[DType.float32](ctx, "qn.init.grad", grad, n)
    if param.past > 0:
        fx_hist[0] = fx

    # Early exit if the initial x is already a minimizer
    if check_convergence(param, k, fx, gnorm, fx_hist):
        _release(S, Y, xp, grad, gradp, drt, scalar)
        return OPT_SUCCESS

    # Initial direction
    ax(ctx, drt, Float32(-1.0), grad, n)
    # Initial step
    var step = Float32(1.0) / nrm2(ctx, drt, n, scalar)
    var fxp = fx

    k = 1
    var end = 0
    var n_vec = 0
    var retcode = OPT_MAX_ITERS_REACHED
    var lsret = LS_SUCCESS
    var ls_iters = 0
    while k <= param.max_iterations:
        # Save the current x and gradient
        copy_vec(ctx, xp, x)
        copy_vec(ctx, gradp, grad)
        fxp = fx

        # Line search to update x, fx and gradient
        lsret = ls_backtrack(
            ctx, param, f, fx, x, grad, step, drt, xp, n, scalar, ls_iters
        )
        gnorm = f.grad_norm(ctx, grad)

        var stop = update_and_check(
            ctx, param, k, lsret, fx, fxp, gnorm, x, xp, grad, gradp,
            fx_hist, retcode,
        )
        if trace.enabled:
            var tag = _iter_tag(k)
            trace.record_scalar_f32(tag + ".loss", fx)
            trace.record_device[DType.float32](ctx, tag + ".grad", grad, n)
            var ls = List[Int32]()
            ls.append(Int32(lsret))
            ls.append(Int32(ls_iters))
            trace.record_list_i32(tag + ".ls", ls)
        if stop:
            _release(S, Y, xp, grad, gradp, drt, scalar)
            return retcode

        # Update s and y: s_{k+1} = x_{k+1} - x_k, y_{k+1} = g_{k+1} - g_k
        axpy(ctx, S[end], Float32(-1.0), xp, x, n)
        axpy(ctx, Y[end], Float32(-1.0), gradp, grad, n)
        # drt <- -H * g
        end = lbfgs_search_dir(
            ctx, param, n_vec, end, S, Y, grad, drt, ys, alpha, n, scalar
        )
        # step = 1.0 as initial guess
        step = Float32(1.0)
        k += 1
    _release(S, Y, xp, grad, gradp, drt, scalar)
    return OPT_MAX_ITERS_REACHED


def _release(
    mut S: List[DeviceBuffer[DType.float32]],
    mut Y: List[DeviceBuffer[DType.float32]],
    mut xp: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut gradp: DeviceBuffer[DType.float32],
    mut drt: DeviceBuffer[DType.float32],
    mut scalar: DeviceBuffer[DType.float32],
):
    """`[[mojo-buffer-freed-at-last-use]]`: a use of every workspace buffer
    AFTER the last launch that reads it, so none is freed under a kernel."""
    _ = len(S)
    _ = len(Y)
    _ = len(xp)
    _ = len(grad)
    _ = len(gradp)
    _ = len(drt)
    _ = len(scalar)


def qn_minimize(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut fx: Float32,
    mut num_iters: Int,
    mut loss: GLMWithData,
    l1: Float32,
    opt_param: LBFGSParam,
    n: Int,
    mut trace: IdentityTrace,
) raises -> Int:
    """`qn_minimize`, `qn_solvers.cuh:417-474`: L-BFGS when `l1 == 0`,
    OWL-QN otherwise. The OWL-QN arm RAISES by name here."""
    if l1 != Float32(0.0):
        raise Error(
            "qn_minimize: l1 != 0 selects min_owlqn (qn_solvers.cuh:446),"
            " which is NOT PORTED: penalty='l1' and 'elasticnet' are refused."
            " See glm/NOT_IMPLEMENTED.tsv"
        )
    var ret = min_lbfgs(ctx, opt_param, loss, x, fx, num_iters, n, trace)
    # `:466-471`: "Maximum iterations reached before solver is converged"
    # is a WARNING upstream, not an error; the retcode carries it.
    return ret
