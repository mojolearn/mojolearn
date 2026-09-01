# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`min_lbfgs`, `update_and_check`, `qn_minimize`: the L-BFGS driver.

PORT OF `cuml/cpp/src/glm/qn/qn_solvers.cuh` at cuML `00094f7`. WHOLE FILE
since 2026-09-01: `min_owlqn` is at the bottom and `qn_minimize` dispatches
to it on `l1 != 0` exactly as theirs does (DEVIATION 552). `update_pseudo`
is in `qn_util.mojo`, beside the operator it applies. Do not improve.

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
from glm.impl.glm.qn.glm_base import GLMWithData
from glm.impl.glm.qn.qn_linesearch import (
    ls_backtrack,
    ls_backtrack_projected,
    owlqn_objective,
)
from glm.impl.glm.qn.qn_util import (
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
    project_direction,
    update_pseudo,
)
from glm.impl.glm.qn.simple_mat.dense import ax, axpy, copy_vec, nrm2


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


# ===========================================================================
# OWL-QN (`qn_solvers.cuh:261-400`), THE l1 SOLVER -- DEVIATION 552
# ===========================================================================
#
# WHY IT IS A DIFFERENT SOLVER AND NOT A DIFFERENT PENALTY. L-BFGS builds a
# quadratic model out of gradient differences, and `|w|` has no gradient at
# zero -- which is precisely the point every l1 solution sits on. OWL-QN
# keeps the L-BFGS machinery and changes three things:
#
#   the objective    carries `l1 * ||w_{0:pg_limit}||_1` in its VALUE
#   the gradient     is replaced by a PSEUDO-gradient wherever it is used
#                    to choose a direction (`qn_util.mojo::get_pseudo_grad`)
#   the step         is projected back into the starting orthant
#                    (`qn_linesearch.mojo::ls_backtrack_projected`)
#
# and leaves `S`, `Y`, the two-loop recursion, `update_and_check` and
# `check_convergence` untouched. Reading the two loops side by side is the
# fastest way to see that: `min_lbfgs` above and `min_owlqn` below differ at
# five lines and every one of them is marked.
#
# THE CERTIFICATE GAINS A DISCRETE STAGE HERE. Every branch that decides
# whether a coefficient is exactly zero is a float comparison with a
# DISCRETE output (IDENTITY_PATHS row 32), so under IDENTICAL the claim this
# arm makes is not only "the coefficients agree bit for bit" but "the SET OF
# ZERO COEFFICIENTS agrees", and the card's `qn.iterNNNN.grad` /
# `qn.n_iter` stages are what a differ reads to locate a disagreement.


def _release_owlqn(
    mut S: List[DeviceBuffer[DType.float32]],
    mut Y: List[DeviceBuffer[DType.float32]],
    mut xp: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut gradp: DeviceBuffer[DType.float32],
    mut drt: DeviceBuffer[DType.float32],
    mut pseudo: DeviceBuffer[DType.float32],
    mut scalar: DeviceBuffer[DType.float32],
):
    """`[[mojo-buffer-freed-at-last-use]]`: `_release` with the extra
    `pseudo` workspace, so none of the eight is freed under a kernel."""
    _ = len(S)
    _ = len(Y)
    _ = len(xp)
    _ = len(grad)
    _ = len(gradp)
    _ = len(drt)
    _ = len(pseudo)
    _ = len(scalar)


def min_owlqn(
    ctx: DeviceContext,
    param: LBFGSParam,
    mut f: GLMWithData,
    l1_penalty: Float32,
    pg_limit: Int,
    mut x: DeviceBuffer[DType.float32],
    mut fx: Float32,
    mut k: Int,
    n: Int,
    mut trace: IdentityTrace,
) raises -> Int:
    """`min_owlqn`, `qn_solvers.cuh:261-400`. `x` holds the initial point and
    the result; `fx` the final objective INCLUDING the l1 term; `k` the
    iteration count. Returns the `OPT_RETCODE`.

    The five lines that are not `min_lbfgs`'s are marked OWL-QN in the body.
    """
    if param.check_param() != 0:
        raise Error(
            "OWL-QN: invalid parameter (check_param code "
            + String(param.check_param()) + ")"
        )
    # `:276`, their ASSERT, copied including both halves.
    if not (pg_limit <= n and pg_limit > 0):
        raise Error(
            "OWL-QN: Invalid pseudo grad limit parameter (pg_limit "
            + String(pg_limit) + ", n " + String(n) + ")"
        )
    # SETUP WORKSPACE (`:279-297`): `min_lbfgs`'s, plus `pseudo`.
    var S = List[DeviceBuffer[DType.float32]]()
    var Y = List[DeviceBuffer[DType.float32]]()
    for _ in range(param.m):
        S.append(ctx.enqueue_create_buffer[DType.float32](n))
        Y.append(ctx.enqueue_create_buffer[DType.float32](n))
    var xp = ctx.enqueue_create_buffer[DType.float32](n)
    var grad = ctx.enqueue_create_buffer[DType.float32](n)
    var gradp = ctx.enqueue_create_buffer[DType.float32](n)
    var drt = ctx.enqueue_create_buffer[DType.float32](n)
    var pseudo = ctx.enqueue_create_buffer[DType.float32](n)  # OWL-QN
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
    # OWL-QN: `f_wrap`, not `f`. The value carries the l1 term; `grad` comes
    # back as the gradient of the LOSS ONLY (`:321-323`).
    fx = owlqn_objective(ctx, f, x, grad, l1_penalty, pg_limit, scalar)
    var gnorm = f.grad_norm(ctx, grad)
    trace.record_scalar_f32("qn.init.loss", fx)
    trace.record_device[DType.float32](ctx, "qn.init.grad", grad, n)
    # OWL-QN: build the pseudo-gradient without overwriting `grad`, which is
    # what `S`/`Y` are built from (`:325-327`).
    update_pseudo(ctx, x, grad, l1_penalty, pg_limit, pseudo, n)
    trace.record_device[DType.float32](ctx, "qn.init.pseudo", pseudo, n)
    if param.past > 0:
        fx_hist[0] = fx

    # Early exit if the initial x is already a minimizer
    if check_convergence(param, k, fx, gnorm, fx_hist):
        _release_owlqn(S, Y, xp, grad, gradp, drt, pseudo, scalar)
        return OPT_SUCCESS

    # Initial direction. OWL-QN: from the PSEUDO gradient (`:338`).
    ax(ctx, drt, Float32(-1.0), pseudo, n)
    # Initial step. OWL-QN: `1 / max(1, nrm2(drt))` (`:344`), where
    # `min_lbfgs` has no `max` (`:180`). Copied, including the difference.
    var d_nrm = nrm2(ctx, drt, n, scalar)
    var step = Float32(1.0) / (d_nrm if d_nrm > Float32(1.0) else Float32(1.0))
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

        # OWL-QN: the PROJECTED line search (`:357-358`).
        lsret = ls_backtrack_projected(
            ctx, param, f, fx, x, grad, pseudo, step, drt, xp, l1_penalty,
            pg_limit, n, scalar, ls_iters,
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
            _release_owlqn(S, Y, xp, grad, gradp, drt, pseudo, scalar)
            return retcode

        # OWL-QN: recompute the pseudo-gradient at the new point (`:380`).
        update_pseudo(ctx, x, grad, l1_penalty, pg_limit, pseudo, n)

        # Update s and y: s_{k+1} = x_{k+1} - x_k, y_{k+1} = g_{k+1} - g_k.
        # NOTE `grad`, not `pseudo`: the quasi-Newton model is built from the
        # LOSS gradient, which is why `update_pseudo` above is careful not to
        # overwrite it.
        axpy(ctx, S[end], Float32(-1.0), xp, x, n)
        axpy(ctx, Y[end], Float32(-1.0), gradp, grad, n)
        # OWL-QN: `drt <- -H * pseudo`, the PSEUDO gradient (`:388-390`).
        end = lbfgs_search_dir(
            ctx, param, n_vec, end, S, Y, pseudo, drt, ys, alpha, n, scalar
        )
        # OWL-QN: project the direction onto the orthant of -pseudo (`:393`).
        project_direction(ctx, drt, pseudo, n)

        # step = 1.0 as initial guess
        step = Float32(1.0)
        k += 1
    _release_owlqn(S, Y, xp, grad, gradp, drt, pseudo, scalar)
    return OPT_MAX_ITERS_REACHED


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
    """`qn_minimize`, `qn_solvers.cuh:406-460`: L-BFGS when `l1 == 0`,
    OWL-QN otherwise. Their dispatch, both arms.

    The test is `l1 == 0.0` and it is EXACT, theirs (`:420`). A penalty of
    1e-40 takes the OWL-QN arm, which is what a user asking for l1 means and
    is not a tolerance this port gets to choose.

    `pg_limit` is `loss.D * loss.C` (`:447`), the weight block without the
    bias; see `qn_util.mojo::update_pseudo` for why that is where the
    intercept escapes the penalty.
    """
    var ret = OPT_MAX_ITERS_REACHED
    if l1 == Float32(0.0):
        ret = min_lbfgs(ctx, opt_param, loss, x, fx, num_iters, n, trace)
    else:
        ret = min_owlqn(
            ctx, opt_param, loss, l1, loss.dims.D * loss.dims.C, x, fx,
            num_iters, n, trace,
        )
    # `:452-457`: "Maximum iterations reached before solver is converged"
    # is a WARNING upstream, not an error; the retcode carries it.
    return ret
