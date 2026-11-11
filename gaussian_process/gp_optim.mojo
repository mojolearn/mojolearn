# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Kernel hyperparameter optimization ON THE DEVICE (cgr4-device-optim-gp,
2026-10-03): DEVIATION 2881's projected L-BFGS (`gp_optim_items.mojo`) with
its state resident in device buffers, evaluating the likelihood and its
gradient in place (`gpr_lml_grad_host`'s kernels, in its order) at the
parameter table the optimizer writes from theta. One call runs every start;
the only word that comes home is the run sequencer's done word, once per
GP_OPT_EVAL_CHECK evaluations (lane n8-gp-krr-smo: the Cholesky's `info`
stays on the device, cholesky/chol_ops.mojo). Before, the state machine was
Python (`_gp_optimizer.py`) and every evaluation rebuilt the kernel on the
host, uploaded X and y, and downloaded the likelihood and the gradient.

The kernel matrix and its gradient are `kernel_gradient.mojo::
gp_kernel_matrix_grad`'s launches, in its order, except that a CONST or
WHITE leaf reads its value from the device parameter table
(`gp_const_dev_kernel` / `gp_white_dev_kernel`: the same `ftz(value)` per
cell) and the length scales are read from the device table as they always
were."""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz
from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    add_jitter,
    chol_validate_jitter,
)
from cholesky.chol_ops import (
    chol_op_begin,
    chol_op_cho_solve,
    chol_op_factor,
    chol_op_logdet,
    chol_op_logdet_scratch_floats,
)
from core.device_zero import enqueue_fill
from core.identity_trace import IdentityTrace
from gaussian_process.checks.kernels import (
    GP_ELEM_TPB,
    GP_K_CONST,
    GP_K_PROD,
    GP_K_RBF,
    GP_K_SUM,
    GP_K_WHITE,
    GPKernelSpec,
    GP_IDN_PRESCALE,
    gp_combine_kernel,
    gp_copy_kernel,
    gp_kernel_stack_floats,
    gp_matern_kernel,
    gp_matern_nu_selector,
    gp_prescale_kernel,
    gp_rbf_kernel,
    gp_sqrt3,
    gp_sqrt5,
    gp_validate_kernel,
)
from gaussian_process.checks.kernel_gradient import (
    GP_GRAD_MATERN05,
    GP_GRAD_RBF,
    gp_ls_grad_kernel,
    gp_ls_grad_pre_kernel,
)
from gaussian_process.estimator import (
    GP_GRAD_TPB,
    GP_YDOT_TPB,
    _family_ctx,
    _gp,
    _length_scale_table,
    _upload,
    gp_eye_kernel,
    gp_grad_fin_kernel,
    gp_grad_part_kernel,
    gp_log_marginal_likelihood_value,
    gp_validate_alpha,
    gp_validate_data,
    gp_validate_targets,
    gpr_ydot_fin_kernel,
    gpr_ydot_part_kernel,
)
from gaussian_process.gp_grad_items import gp_free_count, gp_grad_blocks
from gaussian_process.gpc_items import gpc_fold_blocks
from gaussian_process.host.gp_theta import gp_grad_half
from gaussian_process.gp_optim_items import (
    GP_OPT_MAX_ITER,
    GP_OPT_MAX_LS,
    GP_OPT_SI_BEST,
    GP_OPT_SI_LEN,
    GP_OPT_SEQ_DONE,
    GP_OPT_SEQ_LEN,
    GP_OPT_SI_STOP,
    GP_OPT_TPB,
    GP_STOP_RUNNING,
    gp_opt_advance_item,
    gp_opt_final_item,
    gp_opt_init_item,
    gp_opt_st_len,
    gp_opt_step_item,
    gp_opt_theta_map,
)

comptime _P = MutPointer[Float32, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]

#: Evaluations enqueued between two reads of the done word. An evaluation
#: costs a Cholesky (n^3 / 3) and an n-RHS solve; past the last stop at most
#: GP_OPT_EVAL_CHECK - 1 evaluations run their kernel build and gradient
#: products with the factor and solves gated off. Eight keeps that waste
#: below one evaluation's factor at every n while cutting the drains per fit
#: eightfold (a size-free rule: the same at every shape).
comptime GP_OPT_EVAL_CHECK = 8


@fieldwise_init
struct GPOptResult(Movable):
    """The optimizer's answer: `theta` and `values` (the float32
    hyperparameters that run there) of the winning run, theta order;
    `runs` = n_iter, n_eval, stop, f per run; `best` the winning run."""

    var theta: List[Float32]
    var values: List[Float32]
    var runs: List[Float32]
    var best: Int


@always_inline
def _gi(buf: DeviceBuffer[DType.int32]) -> _I:
    return _I(unsafe_from_address=Int(buf.unsafe_ptr()))


def gp_const_dev_kernel(out_k: _P, mn_in: Int32, par: _P, node: Int32):
    """`gp_const_kernel` with the value read from the parameter table."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(mn_in):
        out_k.unsafe_store(t, ftz(par.unsafe_load(Int(node))))


def gp_white_dev_kernel(out_k: _P, n_in: Int32, par: _P, node: Int32):
    """`gp_white_kernel(is_self = 1, row start 0)` with the noise read from
    the parameter table."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n * n:
        return
    var i = t // n
    var j = t - i * n
    if i == j:
        out_k.unsafe_store(t, ftz(par.unsafe_load(Int(node))))
    else:
        out_k.unsafe_store(t, Float32(0.0))


def gp_opt_init_kernel(
    st: _P, si: _I, tmap: _I, bnd: _P, dpar: _P, dls: _P, t: Int32, run: Int32, s0: UInt32, s1: UInt32
):
    gp_opt_init_item(Int(thread_idx.x), GP_OPT_TPB, Int(t), st, si, tmap, bnd, dpar, dls, Int(run), s0, s1)


def gp_opt_step_kernel(
    st: _P, si: _I, tmap: _I, dpar: _P, dls: _P, ydot: _P, ldet: _P, n: Int32, lml: _P, graw: _P,
    t: Int32, info: _I,
):
    """The optimizer's step after an evaluation. `info` is the factor's
    device word (cholesky/chol_ops.mojo), read here instead of on the host.
    Thread 0 first combines the likelihood's three scalar terms
    (`gp_log_marginal_likelihood_value(y^T alpha_, log|K| at ldet[0], n)`,
    the fit's own) into `lml`; thread 0 is also the only reader of it in the
    step. Once the run's stop word is set the step changes nothing, so the
    evaluations enqueued past a run's stop (up to the next done-word read)
    leave the optimizer's state as the stopping step left it; every thread
    reads the word before the item's first barrier."""
    if si.unsafe_load(GP_OPT_SI_STOP) != Int32(GP_STOP_RUNNING):
        return
    var code = Int(info.unsafe_load(0))
    if Int(thread_idx.x) == 0 and code == 0:
        lml.unsafe_store(0, gp_log_marginal_likelihood_value(ydot.unsafe_load(0), ldet.unsafe_load(0), Int(n)))
    gp_opt_step_item(Int(thread_idx.x), GP_OPT_TPB, Int(t), st, si, tmap, dpar, dls, lml, graw, code)


# TOMBSTONE: gp_opt_run_end_kernel and the per-evaluation host loop of gpr_optimize_device (superseded by the device run sequencer) deleted 2026-11-09 by lane/n8-gp-krr-smo; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/N8_GP_OPT_DEVICE_RUNS.patch
def gp_opt_advance_kernel(
    st: _P, si: _I, tmap: _I, bnd: _P, dpar: _P, dls: _P, rec: _P, seq: _I, t: Int32, n_runs: Int32,
    s0: UInt32, s1: UInt32,
):
    """`gp_opt_advance_item`: a stopped run's record and the next run's
    start (or the done word), on the device."""
    gp_opt_advance_item(
        Int(thread_idx.x), GP_OPT_TPB, Int(t), st, si, tmap, bnd, dpar, dls, rec, seq, Int(n_runs), s0, s1,
    )


def gp_opt_final_kernel(st: _P, tmap: _I, dpar: _P, dls: _P, res: _P, t: Int32):
    gp_opt_final_item(Int(thread_idx.x), GP_OPT_TPB, Int(t), st, tmap, dpar, dls, res)


def gp_kernel_matrix_grad_dev(
    ctx: DeviceContext,
    mut kout: DeviceBuffer[DType.float32],
    x_input: DeviceBuffer[DType.float32],
    mut dpar: DeviceBuffer[DType.float32],
    mut dls: DeviceBuffer[DType.float32],
    mut stack: DeviceBuffer[DType.float32],
    mut dgrad: DeviceBuffer[DType.float32],
    n: Int,
    d: Int,
    spec: GPKernelSpec,
    free: List[Int32],
    xs_input: DeviceBuffer[DType.float32],
    elem_tpb: Int = GP_ELEM_TPB,
) raises:
    """`gp_kernel_matrix_grad`'s walk and launches, CONST and WHITE values
    from the device parameter table `dpar` (one float per postfix node).
    ASYNCHRONOUS. `xs_input` is the caller's n x d scratch for the scaled
    copy of X (GP_IDN_PRESCALE; rewritten per RBF / Matern leaf, unread
    when the define is off)."""
    var x = x_input.create_sub_buffer[DType.float32](0, len(x_input))
    var x2 = x_input.create_sub_buffer[DType.float32](0, len(x_input))
    var xs = xs_input.create_sub_buffer[DType.float32](0, len(xs_input))
    var xs2 = xs_input.create_sub_buffer[DType.float32](0, len(xs_input))
    var cells = n * n
    var grid = (cells + elem_tpb - 1) // elem_tpb
    var sqrt3 = gp_sqrt3()
    var sqrt5 = gp_sqrt5()
    var pp = _gp(dpar)
    var sp = 0
    var gi = 0
    var first = List[Int]()
    var count = List[Int]()
    for t in range(len(spec.kinds)):
        var kind = Int(spec.kinds[t])
        if kind == GP_K_SUM or kind == GP_K_PROD:
            var lhs = stack.create_sub_buffer[DType.float32]((sp - 2) * cells, cells)
            var rhs = stack.create_sub_buffer[DType.float32]((sp - 1) * cells, cells)
            var fb = first.pop()
            var cb = count.pop()
            var fa = first.pop()
            var ca = count.pop()
            if kind == GP_K_PROD:
                for q in range(ca):
                    var gsub = dgrad.create_sub_buffer[DType.float32]((fa + q) * cells, cells)
                    ctx.enqueue_function[gp_combine_kernel](
                        gsub.unsafe_ptr(), rhs.unsafe_ptr(), Int32(cells), Int32(1),
                        grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                    )
                    _ = gsub^
                for q in range(cb):
                    var gsub = dgrad.create_sub_buffer[DType.float32]((fb + q) * cells, cells)
                    ctx.enqueue_function[gp_combine_kernel](
                        gsub.unsafe_ptr(), lhs.unsafe_ptr(), Int32(cells), Int32(1),
                        grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                    )
                    _ = gsub^
            ctx.enqueue_function[gp_combine_kernel](
                lhs.unsafe_ptr(), rhs.unsafe_ptr(), Int32(cells),
                Int32(1) if kind == GP_K_PROD else Int32(0),
                grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
            )
            _ = lhs^
            _ = rhs^
            first.append(fa)
            count.append(ca + cb)
            sp -= 1
            continue

        var slot = stack.create_sub_buffer[DType.float32](sp * cells, cells)
        var is_free = Int(free[t]) != 0
        first.append(gi)
        if kind == GP_K_CONST:
            ctx.enqueue_function[gp_const_dev_kernel](
                _gp(slot), Int32(cells), pp, Int32(t),
                grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
            )
            if is_free:
                var gsub = dgrad.create_sub_buffer[DType.float32](gi * cells, cells)
                ctx.enqueue_function[gp_const_dev_kernel](
                    _gp(gsub), Int32(cells), pp, Int32(t),
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
                _ = gsub^
                gi += 1
                count.append(1)
            else:
                count.append(0)
        elif kind == GP_K_WHITE:
            ctx.enqueue_function[gp_white_dev_kernel](
                _gp(slot), Int32(n), pp, Int32(t),
                grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
            )
            if is_free:
                var gsub = dgrad.create_sub_buffer[DType.float32](gi * cells, cells)
                ctx.enqueue_function[gp_white_dev_kernel](
                    _gp(gsub), Int32(n), pp, Int32(t),
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
                _ = gsub^
                gi += 1
                count.append(1)
            else:
                count.append(0)
        else:
            var ln = Int(spec.ls_len[t])
            var lsview = dls.create_sub_buffer[DType.float32](Int(spec.ls_off[t]), ln)
            var form = GP_GRAD_RBF
            # GP_IDN_PRESCALE: X / length_scale once per leaf (the table the
            # optimizer's step wrote), then the cells fold the quotients
            var ls_arg = spec.ls_len[t]
            var xa = x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var xb = x2.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            comptime if GP_IDN_PRESCALE:
                ctx.enqueue_function[gp_prescale_kernel](
                    xs.unsafe_ptr(), x.unsafe_ptr(), lsview.unsafe_ptr(),
                    Int32(n), Int32(d), spec.ls_len[t],
                    grid_dim=((n * d + elem_tpb - 1) // elem_tpb, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
                ls_arg = Int32(0)
                xa = xs.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
                xb = xs2.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            if kind == GP_K_RBF:
                ctx.enqueue_function[gp_rbf_kernel](
                    slot.unsafe_ptr(), xa, xb, lsview.unsafe_ptr(),
                    Int32(n), Int32(n), Int32(d), ls_arg,
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
            else:
                var nu_sel = gp_matern_nu_selector(spec.params[t])
                form = GP_GRAD_MATERN05 + nu_sel
                ctx.enqueue_function[gp_matern_kernel](
                    slot.unsafe_ptr(), xa, xb, lsview.unsafe_ptr(),
                    Int32(n), Int32(n), Int32(d), ls_arg, Int32(nu_sel), sqrt3, sqrt5,
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
            if is_free:
                var gsub = dgrad.create_sub_buffer[DType.float32](gi * cells, ln * cells)
                comptime if GP_IDN_PRESCALE:
                    ctx.enqueue_function[gp_ls_grad_pre_kernel](
                        gsub.unsafe_ptr(), slot.unsafe_ptr(), xs.unsafe_ptr(),
                        Int32(n), Int32(d), Int32(ln), Int32(form), sqrt3, sqrt5,
                        grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                    )
                else:
                    ctx.enqueue_function[gp_ls_grad_kernel](
                        gsub.unsafe_ptr(), slot.unsafe_ptr(), x.unsafe_ptr(), lsview.unsafe_ptr(),
                        Int32(n), Int32(d), Int32(ln), Int32(form), sqrt3, sqrt5,
                        grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                    )
                _ = gsub^
                gi += ln
                count.append(ln)
            else:
                count.append(0)
            _ = lsview^
        _ = slot^
        sp += 1

    var root = stack.create_sub_buffer[DType.float32](0, cells)
    ctx.enqueue_function[gp_copy_kernel](
        kout.unsafe_ptr(), root.unsafe_ptr(), Int32(cells),
        grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
    )
    _ = root^
    _ = x^
    _ = x2^
    _ = xs^
    _ = xs2^


def gpr_optimize_device(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    kernel: GPKernelSpec,
    free: List[Int32],
    bounds: List[Float32],
    alpha: Float32,
    n_restarts: Int,
    seed_lo: UInt32,
    seed_hi: UInt32,
) raises -> GPOptResult:
    """DEVIATION 2881 from the kernel's theta and `n_restarts` Philox starts,
    every run on the device. `bounds` holds (lo, hi) per theta entry, the
    hyperparameter values (not their logs)."""
    gp_validate_data(x, n_train, n_features, String("X"))
    gp_validate_targets(y, n_train)
    gp_validate_kernel(kernel, n_features)
    gp_validate_alpha(alpha)
    chol_validate_jitter(alpha)
    var n = n_train
    var cells = n * n
    var nt = gp_free_count(kernel.kinds, kernel.ls_len, free)
    if nt < 1:
        raise Error("gpr_optimize: the kernel has no free hyperparameter")
    if len(bounds) != 2 * nt:
        raise Error(
            "gpr_optimize: bounds holds " + String(len(bounds)) + " values, the kernel's "
            + String(nt) + " free hyperparameters need " + String(2 * nt)
        )
    if n_restarts < 0:
        raise Error("gpr_optimize: n_restarts cannot be negative")
    var tmap = gp_opt_theta_map(kernel.kinds, kernel.ls_off, kernel.ls_len, free)
    var n_runs = 1 + n_restarts

    var ctx = _family_ctx()
    var dx = _upload(ctx, x)
    var dy = _upload(ctx, y)
    var dpar = _upload(ctx, kernel.params)
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dbnd = _upload(ctx, bounds)
    var dtmap = ctx.enqueue_create_buffer[DType.int32](nt)
    var htmap = ctx.enqueue_create_host_buffer[DType.int32](nt)
    for i in range(nt):
        htmap.unsafe_ptr().unsafe_store(i, tmap[i])
    ctx.enqueue_copy(dst_buf=dtmap, src_ptr=htmap.unsafe_ptr())
    var dk = ctx.enqueue_create_buffer[DType.float32](cells)
    var dstack = ctx.enqueue_create_buffer[DType.float32](gp_kernel_stack_floats(n, n))
    var dgrad = ctx.enqueue_create_buffer[DType.float32](max(nt * cells, 1))
    # GP_IDN_PRESCALE's scaled copy of X, one for the whole run
    var dxs = ctx.enqueue_create_buffer[DType.float32](max(n * n_features, 1))
    # the factor's info word, the log-determinant's scratch and its scalar
    # (cholesky/chol_ops.mojo: enqueued and read-free on every vendor)
    var dinfo = ctx.enqueue_create_buffer[DType.int32](1)
    var dldscr = ctx.enqueue_create_buffer[DType.float32](chol_op_logdet_scratch_floats(n))
    var dldet = ctx.enqueue_create_buffer[DType.float32](1)
    var ddual = ctx.enqueue_create_buffer[DType.float32](n)
    var dkinv = ctx.enqueue_create_buffer[DType.float32](cells)
    var nb = gp_grad_blocks(n)
    var dgpart = ctx.enqueue_create_buffer[DType.float32](max(nt * nb, 1))
    var dgraw = ctx.enqueue_create_buffer[DType.float32](nt)
    var nbf = gpc_fold_blocks(n)
    var dypart = ctx.enqueue_create_buffer[DType.float32](max(nbf, 1))
    var dyd = ctx.enqueue_create_buffer[DType.float32](1)
    var dlml = ctx.enqueue_create_buffer[DType.float32](1)
    var dst = ctx.enqueue_create_buffer[DType.float32](gp_opt_st_len(nt))
    var dsi = ctx.enqueue_create_buffer[DType.int32](GP_OPT_SI_LEN)
    var drec = ctx.enqueue_create_buffer[DType.float32](n_runs * 4)
    var dout = ctx.enqueue_create_buffer[DType.float32](2 * nt)
    # the run sequencer (gp_opt_advance_item): done word, current run
    var dseq = ctx.enqueue_create_buffer[DType.int32](GP_OPT_SEQ_LEN)
    enqueue_fill(ctx, dst, Float32(0.0))
    enqueue_fill(ctx, dseq, Int32(0))
    var hdone = ctx.enqueue_create_host_buffer[DType.int32](1)
    var done_word = dseq.create_sub_buffer[DType.int32](GP_OPT_SEQ_DONE, 1)
    ctx.synchronize()
    _ = htmap^

    # Every run on the device (lane n8-gp-krr-smo, 2026-11-09). Before, the
    # host read the factor's info once per panel and the stop word once per
    # evaluation (evals x (2 + ceil(n/nb)) drains per fit). Now each
    # evaluation is enqueued read-free: the Cholesky's info stays in `dinfo`
    # and gates the factor's later launches, the solves and the
    # log-determinant (CHOL_DEFER_INFO on every vendor); the step reads it on
    # the device; `gp_opt_advance_kernel` closes a stopped run and starts
    # the next. The host reads the done word once per GP_OPT_EVAL_CHECK
    # evaluations. Evaluations enqueued past the last stop find the factor
    # gated (`chol_op_begin` on the done word: nothing factored or solved)
    # and the step and the sequencer inert, so the optimizer's state, the
    # records and the result are the per-evaluation loop's.
    var eval_cap = n_runs * (GP_OPT_MAX_ITER * (GP_OPT_MAX_LS + 1) + 2)
    ctx.enqueue_function[gp_opt_init_kernel](  # small-launch(nt: hyperparameters): one block over the kernel hyperparameters sets a start, never the rows
        _gp(dst), _gi(dsi), _gi(dtmap), _gp(dbnd), _gp(dpar), _gp(dls),
        Int32(nt), Int32(0), seed_lo, seed_hi,
        grid_dim=1, block_dim=GP_OPT_TPB,
    )
    var evals = 0
    while True:  # driver-loop: over done-word reads, one per GP_OPT_EVAL_CHECK evaluations, bounded by eval_cap
        if evals >= eval_cap:
            raise Error("gpr_optimize: the optimizer passed its evaluation cap without a stop word")
        for _e in range(GP_OPT_EVAL_CHECK):  # driver-loop: over the evaluations enqueued between two done-word reads
            evals += 1
            # the likelihood and its gradient at the parameter table, in place
            gp_kernel_matrix_grad_dev(ctx, dk, dx, dpar, dls, dstack, dgrad, n, n_features, kernel, free, dxs)
            add_jitter(ctx, dk, n, alpha, CHOL_ELEM_TPB)
            chol_op_begin(ctx, dseq, dinfo)
            chol_op_factor(ctx, dk, dinfo, n)
            chol_op_logdet(ctx, dk, dinfo, dldscr, dldet, n)
            ctx.enqueue_copy(dst_buf=ddual, src_buf=dy)
            chol_op_cho_solve(ctx, dk, ddual, dinfo, n, 1)
            if nbf > 0:
                ctx.enqueue_function[gpr_ydot_part_kernel](
                    _gp(dy), _gp(ddual), Int32(n), _gp(dypart), Int32(nbf),
                    grid_dim=(nbf + GP_YDOT_TPB - 1) // GP_YDOT_TPB, block_dim=GP_YDOT_TPB,
                )
            ctx.enqueue_function[gpr_ydot_fin_kernel](_gp(dypart), Int32(nbf), _gp(dyd), grid_dim=1, block_dim=1)
            enqueue_fill(ctx, dkinv, Float32(0.0))
            ctx.enqueue_function[gp_eye_kernel](
                _gp(dkinv), Int32(n), grid_dim=(n + GP_YDOT_TPB - 1) // GP_YDOT_TPB, block_dim=GP_YDOT_TPB
            )
            chol_op_cho_solve(ctx, dk, dkinv, dinfo, n, n)
            ctx.enqueue_function[gp_grad_part_kernel](
                _gp(ddual), _gp(dkinv), _gp(dgrad), Int32(n), Int32(nb), Int32(nt * nb), _gp(dgpart),
                grid_dim=(nt * nb + GP_GRAD_TPB - 1) // GP_GRAD_TPB, block_dim=GP_GRAD_TPB,
            )
            ctx.enqueue_function[gp_grad_fin_kernel](
                _gp(dgpart), Int32(nb), Int32(nt), gp_grad_half(), _gp(dgraw),
                grid_dim=(nt + GP_GRAD_TPB - 1) // GP_GRAD_TPB, block_dim=GP_GRAD_TPB,
            )
            ctx.enqueue_function[gp_opt_step_kernel](  # small-launch(nt: hyperparameters): one block strides the theta-sized optimizer stages; n is a constant of the likelihood
                _gp(dst), _gi(dsi), _gi(dtmap), _gp(dpar), _gp(dls), _gp(dyd), _gp(dldet), Int32(n),
                _gp(dlml), _gp(dgraw), Int32(nt), _gi(dinfo),
                grid_dim=1, block_dim=GP_OPT_TPB,
            )
            ctx.enqueue_function[gp_opt_advance_kernel](  # small-launch(nt: hyperparameters): one block closes a stopped run and starts the next over the theta entries only
                _gp(dst), _gi(dsi), _gi(dtmap), _gp(dbnd), _gp(dpar), _gp(dls), _gp(drec), _gi(dseq),
                Int32(nt), Int32(n_runs), seed_lo, seed_hi,
                grid_dim=1, block_dim=GP_OPT_TPB,
            )
        # the one word home per chunk: the done word
        ctx.enqueue_copy(dst_ptr=hdone.unsafe_ptr(), src_buf=done_word)
        ctx.synchronize()
        if Int(hdone.unsafe_ptr().unsafe_load(0)) != 0:
            break
    ctx.enqueue_function[gp_opt_final_kernel](  # small-launch(nt: hyperparameters): one block writes the winning theta, a few entries only
        _gp(dst), _gi(dtmap), _gp(dpar), _gp(dls), _gp(dout), Int32(nt), grid_dim=1, block_dim=GP_OPT_TPB,
    )
    # the results land straight in their lists (no host unpack loops)
    var theta = List[Float32](length=nt, fill=Float32(0.0))
    var values = List[Float32](length=nt, fill=Float32(0.0))
    var runs = List[Float32](length=n_runs * 4, fill=Float32(0.0))
    var hsi = ctx.enqueue_create_host_buffer[DType.int32](GP_OPT_SI_LEN)
    var out_theta = dout.create_sub_buffer[DType.float32](0, nt)
    var out_values = dout.create_sub_buffer[DType.float32](nt, nt)
    ctx.enqueue_copy(dst_ptr=runs.unsafe_ptr(), src_buf=drec)
    ctx.enqueue_copy(dst_ptr=theta.unsafe_ptr(), src_buf=out_theta)
    ctx.enqueue_copy(dst_ptr=values.unsafe_ptr(), src_buf=out_values)
    ctx.enqueue_copy(dst_ptr=hsi.unsafe_ptr(), src_buf=dsi)
    ctx.synchronize()
    var best = Int(hsi.unsafe_ptr().unsafe_load(GP_OPT_SI_BEST))
    _ = out_theta^
    _ = out_values^
    _ = hsi^
    _ = hdone^
    _ = done_word^
    _ = dx^
    _ = dy^
    _ = dpar^
    _ = dls^
    _ = dbnd^
    _ = dtmap^
    _ = dk^
    _ = dstack^
    _ = dgrad^
    _ = dxs^
    _ = dinfo^
    _ = dldscr^
    _ = dldet^
    _ = ddual^
    _ = dkinv^
    _ = dgpart^
    _ = dgraw^
    _ = dypart^
    _ = dyd^
    _ = dlml^
    _ = dst^
    _ = dsi^
    _ = drec^
    _ = dout^
    _ = dseq^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return GPOptResult(theta^, values^, runs^, best)
