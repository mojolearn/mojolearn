# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ARIMA fit's batched objective, `-loglike / (n_obs - 1)` and its
forward-difference gradient for the WHOLE BATCH, enqueued only: the
sequential evaluation (`eval_batch_device`, moved here from
`batched_fit.mojo` by lane n10-dictl-arima-loops, 2026-11-09, so the one
solver loop in `fast_lbfgs_async.mojo` can take it) and `arima_eval`, which
picks the held stacked workspace (`fast_eval_ws.mojo`) when the fit has one.
Both write the same bits (`batched_kalman.mojo::eval_ws_fits`). Nothing here
reads device memory back or drains the stream: the solver's loop body is
these launches, and its one host read is the "any series running" word every
`ASYNC_READ_EVERY` evaluations."""

from max.gpu.host import DeviceBuffer, DeviceContext

from arima.impl.batched_arima import (
    ARIMA_FAST_BATCH_GRAD,
    grad_kernel,
    loglike_ws_packed,
    perturb_kernel,
    reset_param_kernel,
)
from arima.impl.fast_eval_ws import FastEvalWS
from arima.impl.lbfgs_device import LBFGS_TPB, arima_eval_finish_kernel, arima_mark_infeasible_kernel
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams


def eval_batch_device(
    ctx: DeviceContext,
    mut d_y_kf: DeviceBuffer[DType.float32],
    mut d_exog_kf: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs_kf: Int,
    order_kf: ARIMAOrder,
    mut d_x: DeviceBuffer[DType.float32],
    mut d_grad: DeviceBuffer[DType.float32],
    mut d_x_pert: DeviceBuffer[DType.float32],
    mut scratch: ARIMAParams,
    h: Float32,
    scale: Float32,
    mut d_f: DeviceBuffer[DType.float32],
    mut d_g: DeviceBuffer[DType.float32],
    mut d_bad: DeviceBuffer[DType.int32],
) raises:
    """`fit_helper`'s `f` and `gf` (`arima.pyx:905-919`) in ONE call, at the
    candidates already in `d_x`, with every result left on the device:

        f(x)  = -loglike(x, trans=True) / (n_obs - 1)          -> d_f
        gf(x) = -loglike_grad(x, h, trans=True) / (n_obs - 1)  -> d_g

    `scale` is `n_obs - 1` with the ORIGINAL `n_obs`, not the differenced
    one; theirs is `self.n_obs` and the differencing has already happened by
    the time the loglike is called. The base log-likelihood is taken from
    the gradient's own base evaluation (one Kalman pass saved, no bit
    moved). An INFEASIBLE candidate (a Kalman refusal code or a -inf
    log-likelihood at the base or at any forward-difference point) is
    `f = +inf` with a zero gradient, both constants, marked by a kernel:
    the line search's Armijo test fails and halves the step (cuML's NaN
    does the same). The host column (`arima_oracle._eval_batch`) writes the
    same values.

    Nothing is read back here (cpu-gpu-cleanup n-seq, 2026-10-02: the old
    `eval_batch` uploaded the candidates from host lists and brought the
    log-likelihoods and the gradient down every evaluation). Nothing waits
    either (lane n10-dictl-arima-loops, 2026-11-09: the sequential form
    drained the stream once per forward-difference point and once at the
    end): every workspace is released in stream order after its readers.
    The stacked arm (Apple, `ARIMA_FAST_BATCH_GRAD`) evaluates all of the
    points in one batch."""
    var n = order_kf.complexity()
    var bs = batch_size
    var grid = (bs + LBFGS_TPB - 1) // LBFGS_TPB
    ctx.enqueue_memset(d_bad, Int32(0))
    var fut = ctx.enqueue_create_buffer[DType.float32](1)
    comptime if ARIMA_FAST_BATCH_GRAD:
        if order_kf.n_exog == 0:
            # ONE stacked evaluation over (n + 1) x batch members: member m's
            # parameters perturbed in parameter m - 1 (`batched_arima.mojo::
            # _batched_loglike_grad_stacked`, without its read-back).
            var m1 = n + 1
            var eb = m1 * bs
            var nb_y = bs * n_obs_kf
            var nb_x = bs * n
            var y_ext = ctx.enqueue_create_buffer[DType.float32](eb * n_obs_kf)
            var x_ext = ctx.enqueue_create_buffer[DType.float32](eb * n)
            for mm in range(m1):
                ctx.enqueue_copy(
                    dst_buf=y_ext.create_sub_buffer[DType.float32](mm * nb_y, nb_y),
                    src_buf=d_y_kf.create_sub_buffer[DType.float32](0, nb_y),
                )
                ctx.enqueue_copy(
                    dst_buf=x_ext.create_sub_buffer[DType.float32](mm * nb_x, nb_x),
                    src_buf=d_x.create_sub_buffer[DType.float32](0, nb_x),
                )
            for i in range(n):
                var blk = x_ext.unsafe_ptr().unsafe_offset((i + 1) * nb_x)
                var blk_src = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(blk))
                ctx.enqueue_function[perturb_kernel](
                    blk, blk_src, Int32(bs), Int32(n), Int32(i), h,
                    grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
                )
            var p_ext = ARIMAParams(ctx, order_kf, eb)
            var r = loglike_ws_packed(
                ctx, y_ext, d_exog_kf, fut, eb, n_obs_kf, order_kf, x_ext, p_ext
            )
            ctx.enqueue_function[arima_mark_infeasible_kernel](
                d_bad.unsafe_ptr(), r.ws.loglike.unsafe_ptr(), r.ws.info_init.unsafe_ptr(),
                r.ws.info_loop.unsafe_ptr(), Int32(bs), Int32(m1),
                grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
            )
            for i in range(n):
                ctx.enqueue_function[grad_kernel](
                    d_grad.unsafe_ptr(),
                    r.ws.loglike.unsafe_ptr().unsafe_offset((i + 1) * bs),
                    MutPointer[Float32, MutAnyOrigin](
                        unsafe_from_address=Int(r.ws.loglike.unsafe_ptr())
                    ),
                    Int32(bs), Int32(n), Int32(i), h,
                    grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
                )
            # the caller's scratch ends equal to d_x, as the sequential form leaves it
            ctx.enqueue_copy(
                dst_buf=d_x_pert.create_sub_buffer[DType.float32](0, nb_x),
                src_buf=d_x.create_sub_buffer[DType.float32](0, nb_x),
            )
            ctx.enqueue_function[arima_eval_finish_kernel](
                d_f.unsafe_ptr(), d_g.unsafe_ptr(), r.ws.loglike.unsafe_ptr(),
                d_grad.unsafe_ptr(), d_bad.unsafe_ptr(), Int32(bs), Int32(n), scale,
                grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
            )
            # lane n10-dictl-arima-loops: no drain; the stream releases the
            # workspaces after their readers ran (stream-ordered frees, as
            # `DevExec._rr_batch_enq` relies on)
            _ = y_ext^
            _ = x_ext^
            _ = p_ext^
            _ = r^
            _ = fut^
            return
    # THE SEQUENTIAL FORM (`batched_loglike_grad_x`): the base, then one
    # forward-difference point per parameter, the perturbation reset by a
    # COPY (a `+ 0.0` would turn a `-0.0` parameter into `+0.0`).
    ctx.enqueue_copy(
        dst_buf=d_x_pert.create_sub_buffer[DType.float32](0, n * bs),
        src_buf=d_x.create_sub_buffer[DType.float32](0, n * bs),
    )
    var base = loglike_ws_packed(
        ctx, d_y_kf, d_exog_kf, fut, bs, n_obs_kf, order_kf, d_x, scratch
    )
    ctx.enqueue_function[arima_mark_infeasible_kernel](
        d_bad.unsafe_ptr(), base.ws.loglike.unsafe_ptr(), base.ws.info_init.unsafe_ptr(),
        base.ws.info_loop.unsafe_ptr(), Int32(bs), Int32(1),
        grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
    )
    for i in range(n):
        ctx.enqueue_function[perturb_kernel](
            d_x_pert.unsafe_ptr(), d_x.unsafe_ptr(), Int32(bs), Int32(n), Int32(i), h,
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )
        var pert = loglike_ws_packed(
            ctx, d_y_kf, d_exog_kf, fut, bs, n_obs_kf, order_kf, d_x_pert, scratch
        )
        ctx.enqueue_function[arima_mark_infeasible_kernel](
            d_bad.unsafe_ptr(), pert.ws.loglike.unsafe_ptr(), pert.ws.info_init.unsafe_ptr(),
            pert.ws.info_loop.unsafe_ptr(), Int32(bs), Int32(1),
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )
        ctx.enqueue_function[grad_kernel](
            d_grad.unsafe_ptr(), pert.ws.loglike.unsafe_ptr(), base.ws.loglike.unsafe_ptr(),
            Int32(bs), Int32(n), Int32(i), h,
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )
        ctx.enqueue_function[reset_param_kernel](
            d_x_pert.unsafe_ptr(), d_x.unsafe_ptr(), Int32(bs), Int32(n), Int32(i),
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )
        # this point's workspace goes back in stream order, after its
        # readers (lane n10-dictl-arima-loops: was one drain per point)
        _ = pert^
    ctx.enqueue_function[arima_eval_finish_kernel](
        d_f.unsafe_ptr(), d_g.unsafe_ptr(), base.ws.loglike.unsafe_ptr(),
        d_grad.unsafe_ptr(), d_bad.unsafe_ptr(), Int32(bs), Int32(n), scale,
        grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
    )
    # released in stream order (lane n10-dictl-arima-loops: was a drain)
    _ = base^
    _ = fut^


def arima_eval(
    ctx: DeviceContext,
    mut ews: Optional[FastEvalWS],
    mut d_y_kf: DeviceBuffer[DType.float32],
    mut d_exog_kf: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs_kf: Int,
    order_kf: ARIMAOrder,
    mut d_x: DeviceBuffer[DType.float32],
    mut d_grad: DeviceBuffer[DType.float32],
    mut d_x_pert: DeviceBuffer[DType.float32],
    mut scratch: ARIMAParams,
    h: Float32,
    scale: Float32,
    mut d_f: DeviceBuffer[DType.float32],
    mut d_g: DeviceBuffer[DType.float32],
    mut d_bad: DeviceBuffer[DType.int32],
) raises:
    """One batched evaluation at the candidates in `d_x`: the held stacked
    workspace when the fit has one (`ews`), else `eval_batch_device`. Same
    outputs, same bits, nothing read back."""
    if ews:
        ews.value().eval(ctx, order_kf, h, scale, d_x, d_grad, d_x_pert, d_f, d_g, d_bad)
        return
    eval_batch_device(
        ctx, d_y_kf, d_exog_kf, batch_size, n_obs_kf, order_kf, d_x, d_grad, d_x_pert,
        scratch, h, scale, d_f, d_g, d_bad,
    )
