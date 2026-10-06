# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`ls_success` and `ls_backtrack`: the backtracking line search.

Reference: `cuml/cpp/src/glm/qn/qn_linesearch.cuh` (cuML `00094f7`). WHOLE
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
Wolfe `dot(grad, drt)` is implemented and not reached from the Python surface.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from glm.impl.qn.glm_base import GLMWithData, QN_FAST_LS_BATCH, QNF_LS_K
from glm.impl.qn.glm_linear import nrm1
from glm.impl.qn.qn_util import (
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
from glm.impl.qn.simple_mat.dense import VEC_ELEM_TPB, axpy, copy_vec, dot, dot_kernel, read_scalars
from core.column_stats import STATS_TPB
from checks.numerics import ftz, identical_mul_add,GLOBAL_NUMERIC_MODE,NUMERIC_IDENTICAL


def _dg_init_enqueue(
    ctx: DeviceContext,
    mut u: DeviceBuffer[DType.float32],
    mut drt: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
    mut stage: HostBuffer[DType.float32],
    dg_ready: Bool,
) raises:
    """`dot(u, drt)` (`dense.dot`'s launch and value) into `scalar` word 0
    and the copy of words 0..3 into `stage`, ENQUEUED (lane/linear-apple):
    the first candidate's evaluate synchronizes, so dg_init, and the search
    direction's verdict in words 2..3 (`qn_util.lbfgs_search_dir_enqueue`),
    come home with the candidate's loss behind ONE synchronize. The dot is
    enqueued before the step overwrites anything it reads."""
    if not dg_ready:  # else the direction's launch already wrote it
        ctx.enqueue_function[dot_kernel](  # small-launch(n: parameter count n_param, the coefficient vector): an L-BFGS vector phase over d-sized data, never rows
            scalar.unsafe_ptr(), u.unsafe_ptr(), drt.unsafe_ptr(), Int32(n),
            grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
    var sub = scalar.create_sub_buffer[DType.float32](0, 4)
    ctx.enqueue_copy(dst_ptr=stage.unsafe_ptr(), src_buf=sub)
    _ = sub^


def _undo_candidate(
    ctx: DeviceContext,
    mut f: GLMWithData,
    mut x: DeviceBuffer[DType.float32],
    mut xp: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut gradp: DeviceBuffer[DType.float32],
) raises:
    """A positive dg_init: the reference returned LS_INVALID_DIR BEFORE the
    first candidate, so the speculative one is undone. `x` and `grad` go
    back to `xp` / `gradp`, which the solver copied from them at the top of
    this iteration (so they are the words the reference left); `fx` was
    never assigned; the evaluation is uncounted and its gradient norm
    forgotten, so `grad_norm` reduces the restored `grad`."""
    copy_vec(ctx, x, xp)
    copy_vec(ctx, grad, gradp)
    f.n_evals -= 1
    f.gnorm_at = 0


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
    mut stage: HostBuffer[DType.float32],
    mut fresh: Bool,
    mut gradp: DeviceBuffer[DType.float32],
    dg_ready: Bool,
) raises -> Int:
    """`ls_backtrack`, `qn_linesearch.cuh:109-146`. `ls_iters` reports how
    many candidates were evaluated (for the card)."""
    if step <= Float32(0.0):
        return LS_INVALID_STEP
    # lane/apple-fast-linsvr: FAST on Apple under QN_FAST_LS_BATCH (default; -D MOJOLEARN_LSVR_LINESEARCH_BATCH_OFF reverts),
    # Armijo only (the Wolfe arms need a gradient dot per candidate)
    comptime if QN_FAST_LS_BATCH:
        if param.linesearch == LBFGS_LS_BT_ARMIJO and f.ls_batch_applies():
            return ls_backtrack_batched(
                ctx, param, f, fx, x, grad, step, drt, xp, n, scalar, ls_iters,
                stage, fresh, gradp, dg_ready,
            )
    comptime if GLOBAL_NUMERIC_MODE==NUMERIC_IDENTICAL and is_defined["MOJOLEARN_IDN_QN_EXACT_TRIALS"]():
        if param.linesearch==LBFGS_LS_BT_ARMIJO and param.max_linesearch>0 and n>0 and n<=(16*1024*1024-96)//32:
            return ls_backtrack_exact_trials(ctx,param,f,fx,x,grad,step,drt,xp,n,scalar,ls_iters,stage,fresh,gradp,dg_ready)
    return ls_backtrack_sequential(ctx,param,f,fx,x,grad,step,drt,xp,n,scalar,ls_iters,stage,fresh,gradp,dg_ready)


def ls_backtrack_sequential(ctx: DeviceContext,param: LBFGSParam,mut f: GLMWithData,
    mut fx: Float32,mut x: DeviceBuffer[DType.float32],mut grad: DeviceBuffer[DType.float32],
    mut step: Float32,mut drt: DeviceBuffer[DType.float32],mut xp: DeviceBuffer[DType.float32],
    n: Int,mut scalar: DeviceBuffer[DType.float32],mut ls_iters: Int,
    mut stage: HostBuffer[DType.float32],mut fresh: Bool,mut gradp: DeviceBuffer[DType.float32],
    dg_ready: Bool) raises -> Int:
    """The unchanged canonical control, callable by focused schedule fixtures."""
    if step<=Float32(0):
        return LS_INVALID_STEP
    var fx_init = fx
    # lane/linear-apple: dg_init's dot is enqueued and read home with the
    # first candidate's evaluate (one synchronize for both); see
    # `_dg_init_enqueue`. A positive dg_init undoes the speculative step.
    _dg_init_enqueue(ctx, grad, drt, n, scalar, stage, dg_ready)
    var dg_init = Float32(0.0)
    var dg_test = Float32(0.0)
    var first = True
    var width = Float32(0.0)
    ls_iters = 0
    for _ in range(param.max_linesearch):
        # x_{k+1} = x_k + step * d_k
        axpy(ctx, x, step, drt, xp, n)
        var fx_new = f.evaluate(ctx, x, grad)
        if first:
            first = False
            fresh = True
            dg_init = stage.unsafe_ptr().unsafe_load(0)
            if dg_init > Float32(0.0):
                _undo_candidate(ctx, f, x, xp, grad, gradp)
                return LS_INVALID_DIR
            dg_test = param.ftol * dg_init
        fx = fx_new
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


def ls_backtrack_exact_trials(ctx: DeviceContext,param: LBFGSParam,mut f: GLMWithData,
    mut fx: Float32,mut x: DeviceBuffer[DType.float32],mut grad: DeviceBuffer[DType.float32],
    mut step: Float32,mut drt: DeviceBuffer[DType.float32],mut xp: DeviceBuffer[DType.float32],
    n: Int,mut scalar: DeviceBuffer[DType.float32],mut ls_iters: Int,
    mut stage: HostBuffer[DType.float32],mut fresh: Bool,mut gradp: DeviceBuffer[DType.float32],
    dg_ready: Bool) raises -> Int:
    """Bounded independent full trial-vector evaluations, one scalar drain.
    Each w_c = xp + step_c*drt uses the original axpy and full evaluator;
    unlike FAST's linearized-score batch, no objective spelling changes.
    Physical work may exceed the logical first-acceptable evaluations and
    is recorded separately. Every rejection and min/max test is walked in
    the original order; all later trial words are discarded on acceptance.
    """
    var count = min(4,param.max_linesearch)
    var before = f.n_evals
    ls_iters=0
    var initial = fx
    _dg_init_enqueue(ctx,grad,drt,n,scalar,stage,dg_ready)
    var trial_x = List[DeviceBuffer[DType.float32]]()
    var trial_g = List[DeviceBuffer[DType.float32]]()
    var steps = List[Float32]()
    var results = ctx.enqueue_create_buffer[DType.float32](count*3)
    var host = ctx.enqueue_create_host_buffer[DType.float32](count*3)
    var next = step
    for c in range(4):  # four admitted independent trial descriptors
        if c>=count:
            continue
        trial_x.append(ctx.enqueue_create_buffer[DType.float32](n))
        trial_g.append(ctx.enqueue_create_buffer[DType.float32](n))
        steps.append(next)
        axpy(ctx,trial_x[c],next,drt,xp,n)
        _ = f.evaluate_pen(ctx,trial_x[c],trial_g[c],0,False)
        var source = f.slots.create_sub_buffer[DType.float32](0,3)
        var target = results.create_sub_buffer[DType.float32](c*3,3)
        ctx.enqueue_copy(dst_buf=target,src_buf=source)
        _ = source^; _ = target^
        next*=param.ls_dec
    ctx.enqueue_copy(dst_buf=host,src_buf=results)
    ctx.synchronize()
    f.speculative_evals+=count
    fresh=True
    var dg = stage[0]
    if dg>Float32(0):
        f.n_evals=before+1
        _undo_candidate(ctx,f,x,xp,grad,gradp)
        _ = trial_x^; _ = trial_g^; _ = results^
        return LS_INVALID_DIR
    var test = param.ftol*dg
    var selected = count-1
    var ret = LS_MAX_ITERS_REACHED
    var decided = False
    ls_iters=0
    for c in range(4):  # four admitted independent trial descriptors
        if c>=count:
            continue
        step=steps[c]
        fx=host[c*3] if f.l2==Float32(0) else ftz(host[c*3]+host[c*3+1])
        ls_iters+=1
        selected=c
        if not (fx>identical_mul_add(step,test,initial)):
            ret=LS_SUCCESS
            decided=True
            break
        if step<param.min_step:
            ret=LS_INVALID_STEP_MIN
            decided=True
            break
        if step>param.max_step:
            ret=LS_INVALID_STEP_MAX
            decided=True
            break
    # Observable logical evaluation count and accepted gradient-norm cache
    # match the sequential solver. The separate diagnostic counts all work.
    f.n_evals=before+ls_iters
    copy_vec(ctx,x,trial_x[selected])
    copy_vec(ctx,grad,trial_g[selected])
    f.gnorm_raw=host[selected*3+2]
    f.gnorm_at=Int(grad.unsafe_ptr())
    for slot in range(3):
        f.stage[slot]=host[selected*3+slot]
    var selected_slots=results.create_sub_buffer[DType.float32](selected*3,3)
    var active_slots=f.slots.create_sub_buffer[DType.float32](0,3)
    ctx.enqueue_copy(dst_buf=active_slots,src_buf=selected_slots)
    _ = selected_slots^; _ = active_slots^
    ctx.synchronize()  # selected words copied before temporary buffers die
    _ = trial_x^; _ = trial_g^; _ = results^
    if decided:
        return ret
    step*=param.ls_dec
    var width=Float32(0)
    for remaining in range(param.max_linesearch-ls_iters):
        axpy(ctx,x,step,drt,xp,n)
        fx=f.evaluate(ctx,x,grad)
        ls_iters+=1
        if ls_success(ctx,param,initial,dg,fx,test,step,grad,drt,n,width,scalar):
            return LS_SUCCESS
        if step<param.min_step:
            return LS_INVALID_STEP_MIN
        if step>param.max_step:
            return LS_INVALID_STEP_MAX
        step*=width
    return LS_MAX_ITERS_REACHED


def ls_backtrack_batched(
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
    mut stage: HostBuffer[DType.float32],
    mut fresh: Bool,
    mut gradp: DeviceBuffer[DType.float32],
    dg_ready: Bool,
) raises -> Int:
    """`ls_backtrack` with the first QNF_LS_K candidates priced by ONE fused
    pass (QN_FAST_LS_BATCH, `GLMWithData.evaluate_batch`): candidate 0 at
    `step` is evaluated in full (its loss, gradient and norm, as `evaluate`
    would), and the objectives at `step * ls_dec^c`, c = 1 .. K - 1, come
    home behind the same synchronize. The host then walks the candidates
    exactly as the loop in `ls_backtrack` does (Armijo on `fx_c`, the
    min_step / max_step tests, `ls_iters`, `step *= ls_dec`), without a
    launch or a synchronize per rejected step. Whichever candidate the walk
    stops on is then materialized: `x = xp + step_c drt` and one full
    `evaluate` for its gradient and its authoritative `fx` (candidate 0
    needs nothing more). If all K are rejected with budget left, the walk
    continues one candidate at a time as before. The decision for c >= 1 is
    made on the batch's `fx_c` (z_c formed as `x . xp + step_c (x . drt)`),
    which can differ from the full evaluation at that point in the last
    bits; FAST promises no bits."""
    comptime if not QN_FAST_LS_BATCH:
        raise Error("qn: ls_backtrack_batched is compiled under FAST + Apple (QN_FAST_LS_BATCH) only")
    else:
        var fx_init = fx
        _dg_init_enqueue(ctx, grad, drt, n, scalar, stage, dg_ready)
        # candidate 0 and the K - 1 next steps, one pass, one synchronize
        axpy(ctx, x, step, drt, xp, n)
        var fx0 = f.evaluate_batch(ctx, x, grad, xp, drt, step, param.ls_dec)
        fresh = True
        var dg_init = stage.unsafe_ptr().unsafe_load(0)
        if dg_init > Float32(0.0):
            _undo_candidate(ctx, f, x, xp, grad, gradp)
            return LS_INVALID_DIR
        var dg_test = param.ftol * dg_init
        ls_iters = 0
        var c = 0
        var last = 0
        var step_last = step
        var ret = LS_MAX_ITERS_REACHED
        var decided = False
        while c < QNF_LS_K and ls_iters < param.max_linesearch:
            fx = fx0 if c == 0 else f.batch_fx(c)
            ls_iters += 1
            last = c
            step_last = step
            if not (fx > identical_mul_add(step, dg_test, fx_init)):
                ret = LS_SUCCESS
                decided = True
                break
            if step < param.min_step:
                ret = LS_INVALID_STEP_MIN
                decided = True
                break
            if step > param.max_step:
                ret = LS_INVALID_STEP_MAX
                decided = True
                break
            step *= param.ls_dec
            c += 1
        if not decided and ls_iters < param.max_linesearch:
            # every batch candidate rejected, budget left: the plain walk on
            for _ in range(param.max_linesearch - ls_iters):
                axpy(ctx, x, step, drt, xp, n)
                fx = f.evaluate(ctx, x, grad)
                ls_iters += 1
                var width = Float32(0.0)
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
        if last != 0:
            # the walk stopped on a priced candidate: its point, gradient and
            # authoritative objective, one evaluation
            axpy(ctx, x, step_last, drt, xp, n)
            fx = f.evaluate(ctx, x, grad)
        return ret


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
# `ls_success` is reused unchanged: the reference calls the SAME function and
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
    deliberate in the reference and is commented there ("fx is loss+regularizer,
    grad is grad of loss only", `:322-323`); the pseudo-gradient is what
    stands in for the missing piece.

    `nrm1` is taken over the FIRST `pg_limit` entries, the weight block, so
    the intercept is not penalized. `nrm1` already takes a length, so the
    mask is that argument and no sub-buffer is made.

    It lives in this module and not in `qn_solvers.mojo` because Mojo has no
    closure to hand across files and `ls_backtrack_projected` below is its
    other caller. Two roundings on the host, both flushed (row 10).
    """
    # lane/linear-apple: the l1 norm comes home with the loss (one
    # synchronize); the same kernel on the same `x`, after the same launches.
    var tmp = f.evaluate_pen(ctx, x, grad, pg_limit)
    var pen = f.last_pen
    _ = len(scalar)
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
    mut stage: HostBuffer[DType.float32],
    mut fresh: Bool,
    mut gradp: DeviceBuffer[DType.float32],
    dg_ready: Bool,
) raises -> Int:
    """`ls_backtrack_projected`, `qn_linesearch.cuh:148-197`. `ls_iters`
    reports how many candidates were evaluated (for the card), as
    `ls_backtrack` does."""
    if step <= Float32(0.0):
        return LS_INVALID_STEP
    var fx_init = fx
    # `dot(pseudo_grad, drt)`, NOT `dot(grad, drt)`. See the banner.
    # lane/linear-apple: read home with the first candidate (see ls_backtrack).
    _dg_init_enqueue(ctx, pseudo_grad, drt, n, scalar, stage, dg_ready)
    var dg_init = Float32(0.0)
    var dg_test = Float32(0.0)
    var first = True
    var width = Float32(0.0)
    ls_iters = 0
    for _ in range(param.max_linesearch):
        # x_{k+1} = proj_orth(x_k + step * d_k)
        projected_step(ctx, x, step, drt, xp, pseudo_grad, n)
        # evaluates fx WITH the l1 term, but only grad of the loss term
        var fx_new = owlqn_objective(ctx, f, x, grad, l1_penalty, pg_limit, scalar)
        if first:
            first = False
            fresh = True
            dg_init = stage.unsafe_ptr().unsafe_load(0)
            if dg_init > Float32(0.0):
                _undo_candidate(ctx, f, x, xp, grad, gradp)
                return LS_INVALID_DIR
            dg_test = param.ftol * dg_init
        fx = fx_new
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
