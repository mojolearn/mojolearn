# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Kernel hyperparameter optimization ON THE DEVICE (cgr4-device-optim-gp,
2026-10-03): DEVIATION 2881's projected L-BFGS (`gp_optim_items.mojo`) with
its state resident in device buffers, evaluating the likelihood and its
gradient in place (`gpr_lml_grad_host`'s kernels, in its order) at the
parameter table the optimizer writes from theta. One call runs every start;
the only words that come home during a run are the Cholesky's `info` and
the optimizer's stop word, once per evaluation. Before, the state machine was
Python (`_gp_optimizer.py`) and every evaluation rebuilt the kernel on the
host, uploaded X and y, and downloaded the likelihood and the gradient.

The kernel matrix and its gradient are `kernel_gradient.mojo::
gp_kernel_matrix_grad`'s launches, in its order, except that a CONST or
WHITE leaf reads its value from the device parameter table
(`gp_const_dev_kernel` / `gp_white_dev_kernel`: the same `ftz(value)` per
cell) and the length scales are read from the device table as they always
were."""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz
from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    CHOL_PANEL_TPB,
    add_jitter,
    chol_default_nb_hint,
    chol_nb_for,
    CHOL_NB_PINNED,
    chol_validate_jitter,
    chol_workspace_floats,
    logdet_kernel,
    potrf_lower,
)
from cholesky.impl.matrix.detail.matrix import copy_vector_from_matrix_diagonal_kernel
from cholesky.checks.trsm import CHOL_SOLVE_TPB, cho_solve
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
    gp_combine_kernel,
    gp_copy_kernel,
    gp_kernel_stack_floats,
    gp_matern_kernel,
    gp_matern_nu_selector,
    gp_rbf_kernel,
    gp_sqrt3,
    gp_sqrt5,
    gp_validate_kernel,
)
from gaussian_process.checks.kernel_gradient import GP_GRAD_MATERN05, GP_GRAD_RBF, gp_ls_grad_kernel
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
    GP_OPT_SI_STOP,
    GP_OPT_TPB,
    gp_opt_final_item,
    gp_opt_init_item,
    gp_opt_run_end_item,
    gp_opt_st_len,
    gp_opt_step_item,
    gp_opt_theta_map,
)

comptime _P = MutPointer[Float32, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]


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


def gp_lml_dev_kernel(ydot: _P, work: _P, n: Int32, dst: _P):
    """`gp_log_marginal_likelihood_value(y^T alpha_, log|K|, n)`, one scalar
    (the likelihood's three terms; not a loop)."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        dst.unsafe_store(0, gp_log_marginal_likelihood_value(ydot.unsafe_load(0), work.unsafe_load(Int(n)), Int(n)))


def gp_opt_init_kernel(
    st: _P, si: _I, tmap: _I, bnd: _P, dpar: _P, dls: _P, t: Int32, run: Int32, s0: UInt32, s1: UInt32
):
    gp_opt_init_item(Int(thread_idx.x), GP_OPT_TPB, Int(t), st, si, tmap, bnd, dpar, dls, Int(run), s0, s1)


def gp_opt_step_kernel(st: _P, si: _I, tmap: _I, dpar: _P, dls: _P, lml: _P, graw: _P, t: Int32, info: Int32):
    gp_opt_step_item(Int(thread_idx.x), GP_OPT_TPB, Int(t), st, si, tmap, dpar, dls, lml, graw, Int(info))


def gp_opt_run_end_kernel(st: _P, si: _I, rec: _P, t: Int32, run: Int32):
    gp_opt_run_end_item(Int(thread_idx.x), GP_OPT_TPB, Int(t), st, si, rec, Int(run))


def gp_opt_final_kernel(st: _P, tmap: _I, dpar: _P, dls: _P, out: _P, t: Int32):
    gp_opt_final_item(Int(thread_idx.x), GP_OPT_TPB, Int(t), st, tmap, dpar, dls, out)


def gp_kernel_matrix_grad_dev(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    x_input: DeviceBuffer[DType.float32],
    mut dpar: DeviceBuffer[DType.float32],
    mut dls: DeviceBuffer[DType.float32],
    mut stack: DeviceBuffer[DType.float32],
    mut dgrad: DeviceBuffer[DType.float32],
    n: Int,
    d: Int,
    spec: GPKernelSpec,
    free: List[Int32],
    elem_tpb: Int = GP_ELEM_TPB,
) raises:
    """`gp_kernel_matrix_grad`'s walk and launches, CONST and WHITE values
    from the device parameter table `dpar` (one float per postfix node).
    ASYNCHRONOUS."""
    var x = x_input.create_sub_buffer[DType.float32](0, len(x_input))
    var x2 = x_input.create_sub_buffer[DType.float32](0, len(x_input))
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
            if kind == GP_K_RBF:
                ctx.enqueue_function[gp_rbf_kernel](
                    slot.unsafe_ptr(), x.unsafe_ptr(), x2.unsafe_ptr(), lsview.unsafe_ptr(),
                    Int32(n), Int32(n), Int32(d), spec.ls_len[t],
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
            else:
                var nu_sel = gp_matern_nu_selector(spec.params[t])
                form = GP_GRAD_MATERN05 + nu_sel
                ctx.enqueue_function[gp_matern_kernel](
                    slot.unsafe_ptr(), x.unsafe_ptr(), x2.unsafe_ptr(), lsview.unsafe_ptr(),
                    Int32(n), Int32(n), Int32(d), spec.ls_len[t], Int32(nu_sel), sqrt3, sqrt5,
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
            if is_free:
                var gsub = dgrad.create_sub_buffer[DType.float32](gi * cells, ln * cells)
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
        out.unsafe_ptr(), root.unsafe_ptr(), Int32(cells),
        grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
    )
    _ = root^
    _ = x^
    _ = x2^


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
    var nb_pin = chol_nb_for(n, CHOL_NB_PINNED)
    var ws = ctx.enqueue_create_buffer[DType.float32](chol_workspace_floats(n, nb_pin))
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1)
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
    enqueue_fill(ctx, dst, Float32(0.0))
    var hstop = ctx.enqueue_create_host_buffer[DType.int32](1)
    var stop_word = dsi.create_sub_buffer[DType.int32](GP_OPT_SI_STOP, 1)
    ctx.synchronize()
    _ = htmap^

    var ctrace = IdentityTrace()
    var eval_cap = GP_OPT_MAX_ITER * (GP_OPT_MAX_LS + 1) + 2
    for run in range(n_runs):
        ctx.enqueue_function[gp_opt_init_kernel](
            _gp(dst), _gi(dsi), _gi(dtmap), _gp(dbnd), _gp(dpar), _gp(dls),
            Int32(nt), Int32(run), seed_lo, seed_hi,
            grid_dim=1, block_dim=GP_OPT_TPB,
        )
        var evals = 0
        while True:
            evals += 1
            if evals > eval_cap:
                raise Error("gpr_optimize: the optimizer passed its evaluation cap without a stop word")
            # the likelihood and its gradient at the parameter table, in place
            gp_kernel_matrix_grad_dev(ctx, dk, dx, dpar, dls, dstack, dgrad, n, n_features, kernel, free)
            add_jitter(ctx, dk, n, alpha, CHOL_ELEM_TPB)
            var chol = potrf_lower(ctx, dk, ws, n, ctrace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB)
            var info = chol.info
            if info == 0:
                var diag = dwork.create_sub_buffer[DType.float32](0, n)
                var scalar = dwork.create_sub_buffer[DType.float32](n, 1)
                ctx.enqueue_function[copy_vector_from_matrix_diagonal_kernel](
                    diag.unsafe_ptr(), dk.unsafe_ptr(), Int32(n), Int32(n),
                    grid_dim=((n + CHOL_ELEM_TPB - 1) // CHOL_ELEM_TPB, 1, 1), block_dim=(CHOL_ELEM_TPB, 1, 1),
                )
                ctx.enqueue_function[logdet_kernel](
                    diag.unsafe_ptr(), scalar.unsafe_ptr(), Int32(n), grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
                )
                _ = diag^
                _ = scalar^
                ctx.enqueue_copy(dst_buf=ddual, src_buf=dy)
                cho_solve(ctx, dk, ddual, n, 1, ctrace, CHOL_SOLVE_TPB)
                if nbf > 0:
                    ctx.enqueue_function[gpr_ydot_part_kernel](
                        _gp(dy), _gp(ddual), Int32(n), _gp(dypart), Int32(nbf),
                        grid_dim=(nbf + GP_YDOT_TPB - 1) // GP_YDOT_TPB, block_dim=GP_YDOT_TPB,
                    )
                ctx.enqueue_function[gpr_ydot_fin_kernel](_gp(dypart), Int32(nbf), _gp(dyd), grid_dim=1, block_dim=1)
                ctx.enqueue_function[gp_lml_dev_kernel](_gp(dyd), _gp(dwork), Int32(n), _gp(dlml), grid_dim=1, block_dim=1)
                enqueue_fill(ctx, dkinv, Float32(0.0))
                ctx.enqueue_function[gp_eye_kernel](
                    _gp(dkinv), Int32(n), grid_dim=(n + GP_YDOT_TPB - 1) // GP_YDOT_TPB, block_dim=GP_YDOT_TPB
                )
                cho_solve(ctx, dk, dkinv, n, n, ctrace, CHOL_SOLVE_TPB)
                ctx.enqueue_function[gp_grad_part_kernel](
                    _gp(ddual), _gp(dkinv), _gp(dgrad), Int32(n), Int32(nb), Int32(nt * nb), _gp(dgpart),
                    grid_dim=(nt * nb + GP_GRAD_TPB - 1) // GP_GRAD_TPB, block_dim=GP_GRAD_TPB,
                )
                ctx.enqueue_function[gp_grad_fin_kernel](
                    _gp(dgpart), Int32(nb), Int32(nt), gp_grad_half(), _gp(dgraw),
                    grid_dim=(nt + GP_GRAD_TPB - 1) // GP_GRAD_TPB, block_dim=GP_GRAD_TPB,
                )
            ctx.enqueue_function[gp_opt_step_kernel](
                _gp(dst), _gi(dsi), _gi(dtmap), _gp(dpar), _gp(dls), _gp(dlml), _gp(dgraw),
                Int32(nt), Int32(info),
                grid_dim=1, block_dim=GP_OPT_TPB,
            )
            # the one word home per evaluation: the stop word
            ctx.enqueue_copy(dst_ptr=hstop.unsafe_ptr(), src_buf=stop_word)
            ctx.synchronize()
            if Int(hstop.unsafe_ptr().unsafe_load(0)) != 0:
                break
        ctx.enqueue_function[gp_opt_run_end_kernel](
            _gp(dst), _gi(dsi), _gp(drec), Int32(nt), Int32(run), grid_dim=1, block_dim=GP_OPT_TPB,
        )
    ctx.enqueue_function[gp_opt_final_kernel](
        _gp(dst), _gi(dtmap), _gp(dpar), _gp(dls), _gp(dout), Int32(nt), grid_dim=1, block_dim=GP_OPT_TPB,
    )
    var hrec = ctx.enqueue_create_host_buffer[DType.float32](n_runs * 4)
    var hout = ctx.enqueue_create_host_buffer[DType.float32](2 * nt)
    var hsi = ctx.enqueue_create_host_buffer[DType.int32](GP_OPT_SI_LEN)
    ctx.enqueue_copy(dst_ptr=hrec.unsafe_ptr(), src_buf=drec)
    ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=dout)
    ctx.enqueue_copy(dst_ptr=hsi.unsafe_ptr(), src_buf=dsi)
    ctx.synchronize()
    var theta = List[Float32](length=nt, fill=Float32(0.0))
    var values = List[Float32](length=nt, fill=Float32(0.0))
    for i in range(nt):
        theta[i] = hout.unsafe_ptr().unsafe_load(i)
        values[i] = hout.unsafe_ptr().unsafe_load(nt + i)
    var runs = List[Float32](length=n_runs * 4, fill=Float32(0.0))
    for i in range(n_runs * 4):
        runs[i] = hrec.unsafe_ptr().unsafe_load(i)
    var best = Int(hsi.unsafe_ptr().unsafe_load(GP_OPT_SI_BEST))
    _ = hrec^
    _ = hout^
    _ = hsi^
    _ = hstop^
    _ = stop_word^
    _ = dx^
    _ = dy^
    _ = dpar^
    _ = dls^
    _ = dbnd^
    _ = dtmap^
    _ = dk^
    _ = dstack^
    _ = dgrad^
    _ = ws^
    _ = dwork^
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
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return GPOptResult(theta^, values^, runs^, best)
