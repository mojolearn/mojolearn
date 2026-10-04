# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Opt-in nonseasonal AutoARIMA order fits batched by Kalman dimension.

Orders keep separate parameter vectors, Jones transforms and GPU L-BFGS
state. Their gradient members share a contiguous Kalman workspace and one
filter launch per dimension, improving occupancy for the small board batch.
Output rows retain input grid order, so Python's first-minimum IC tie rule
is unchanged. No order pruning or approximation is performed.
"""
from std.math import isfinite
from max.gpu.host import DeviceContext, DeviceBuffer
from core.neural_context import process_ctx
from arima.estimator import _DEVCTX_SLOT, _upload_f32, _loglike_at, _write_list_f32
from arima.impl.batched_arima import _refuse_non_finite
from arima.impl.batched_fit import ARIMA_FIT_H, arima_fit_params, _download, _upload
from arima.impl.batched_kalman import KalmanWorkspace, fast_kalman_init_into, _launch_loop_ll_only
from arima.impl.estimate_x0 import estimate_x0_x
from arima.impl.fast_eval_ws import FastEvalWS
from arima.impl.fast_lbfgs_async import ASYNC_READ_EVERY
from arima.impl.fast_order_state import ARIMA_ORDER_BATCH, OrderOptimizer
from arima.impl.fast_slab import slab_begin, slab_end, slab_f32
from arima.impl.timeSeries.arima_helpers import batched_jones_transform
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams, pack, unpack, validate_order
from tsa.impl.timeSeries.arima_helpers import prepare_data


def _initial_x(ctx: DeviceContext, mut y: DeviceBuffer[DType.float32],
               mut exog: DeviceBuffer[DType.float32], bs: Int, nobs: Int,
               order: ARIMAOrder) raises -> List[Float32]:
    """Original fit's estimate_x0 and inverse Jones transform, unchanged."""
    var params = ARIMAParams(ctx, order, bs)
    var info = ctx.enqueue_create_buffer[DType.int32](bs)
    var start = estimate_x0_x(ctx, params, y, exog, bs, nobs, order, info)
    var inv = ARIMAParams(ctx, order, bs)
    batched_jones_transform(ctx, order, bs, True, params, inv)
    var x = ctx.enqueue_create_buffer[DType.float32](order.complexity() * bs)
    pack(ctx, inv, order, bs, x)
    ctx.synchronize()
    var x0 = _download(ctx, x, order.complexity() * bs)
    for i in range(len(x0)):
        if not isfinite(x0[i]):
            raise Error("AutoARIMA: non-finite initial parameter at index " + String(i))
    _ = start^
    _ = params^
    _ = inv^
    _ = info^
    _ = x^
    return x0^


def order_search_loglike(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    orders: List[ARIMAOrder], bs: Int, nobs: Int, maxiter: Int,
) raises -> Int:
    comptime if not ARIMA_ORDER_BATCH:
        raise Error("AutoARIMA grouped orders were not compiled")
    else:
        if bs < 1 or nobs < 3 or maxiter < 1 or len(orders) < 1:
            raise Error("AutoARIMA grouped orders: invalid shape or iteration count")
        var d = orders[0].d
        for i in range(len(orders)):
            var o = orders[i]
            validate_order(o)
            if o.d != d or o.P != 0 or o.D != 0 or o.Q != 0 or o.s != 0 or o.n_exog != 0 or o.r() > 4:
                raise Error("AutoARIMA grouped orders require a nonseasonal same-d grid with r <= 4")
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var y = _upload_f32(ctx, y_ptr, bs * nobs)
        _refuse_non_finite(ctx, y, bs * nobs, "y")
        var exog = ctx.enqueue_create_buffer[DType.float32](1)
        var nkf = nobs - d
        var ykf = ctx.enqueue_create_buffer[DType.float32](bs * nkf)
        if d > 0:
            prepare_data(ctx, ykf, y, bs, nobs, d, 0, 0)
        else:
            ctx.enqueue_copy(dst_buf=ykf, src_buf=y)
        ctx.synchronize()
        var hp = arima_fit_params(maxiter)
        # Grouping never changes the output index. k is uniform within a
        # filter launch, as are rd and observation count after differencing.
        for rd in range(1, 5):
            for k in range(2):
                var ids = List[Int]()
                var members = 0
                for i in range(len(orders)):
                    if orders[i].r() == rd and orders[i].k == k:
                        ids.append(i)
                        members += bs * (orders[i].complexity() + 1)
                if len(ids) == 0:
                    continue
                # ARIMA_SLAB (opt-in): this group's buffers are views of
                # the slab; an empty mark (no window) in other builds.
                var slab_mark = slab_begin()
                var group_order = ARIMAOrder(rd, 0, 0, 0, 0, 0, 0, 1, 0)
                var group_ws = KalmanWorkspace(ctx, group_order, members, nkf, 0)
                var group_y = slab_f32(ctx, members * nkf)
                var group_params = ARIMAParams(ctx, group_order, members)
                var evals = List[FastEvalWS]()
                var states = List[OrderOptimizer]()
                var offset = 0
                for j in range(len(ids)):
                    var order = orders[ids[j]]
                    var okf = order.without_diff()
                    var x0 = _initial_x(ctx, y, exog, bs, nobs, order)
                    var ew = FastEvalWS(ctx, ykf, bs, nkf, okf,
                                        group_ws, group_y, group_params.mu, offset)
                    var state = OrderOptimizer(ctx, ew, bs, Float32(nobs - 1), okf,
                                               x0, hp, ARIMA_FIT_H)
                    offset += ew.eb
                    evals.append(ew^)
                    states.append(state^)
                var rounds = 0
                var max_rounds = hp.max_iterations * max(1, hp.max_linesearch) + 1
                while rounds < max_rounds:
                    if rounds % ASYNC_READ_EVERY == 0:
                        for j in range(len(states)):
                            states[j].enqueue_poll(ctx)
                        ctx.synchronize()
                        var running = False
                        for j in range(len(states)):
                            running = running or states[j].running()
                        if not running:
                            break
                    for j in range(len(states)):
                        ref state = states[j]
                        ref ew = evals[j]
                        ew.prepare(ctx, state.order, state.h, state.cand, state.bad)
                        fast_kalman_init_into(ctx, ew.t_params, state.order, ew.eb, ew.ws)
                    # The actual joint GPU work: all gradient members of
                    # all same-rd orders in a single specialized filter.
                    _launch_loop_ll_only(ctx, group_y, group_params, group_ws,
                                         rd, nkf, members, k, 0, 32)
                    for j in range(len(states)):
                        ref state = states[j]
                        ref ew = evals[j]
                        ew.finish(ctx, state.h, state.scale, state.cand, state.d_grad,
                                  state.d_x_pert, state.f_fxc, state.gradc, state.bad)
                        state.advance(ctx)
                    rounds += 1
                for j in range(len(states)):
                    var order = orders[ids[j]]
                    var result = states[j].result(ctx)
                    var x = ctx.enqueue_create_buffer[DType.float32](bs * order.complexity())
                    _upload(ctx, x, result.x)
                    var raw = ARIMAParams(ctx, order, bs)
                    var fitted = ARIMAParams(ctx, order, bs)
                    unpack(ctx, raw, order, bs, x)
                    batched_jones_transform(ctx, order, bs, False, raw, fitted)
                    # Reevaluate the fitted likelihood exactly as public
                    # ARIMA.fit does; never recover it by rescaling fx.
                    var ll = _loglike_at(ctx, y, exog, bs, nobs, order, fitted)
                    _write_list_f32(out_ptr, ll, ids[j] * bs)
                    _ = result^
                    _ = raw^
                    _ = fitted^
                    _ = x^
                ctx.synchronize()
                _ = states^
                _ = evals^
                _ = group_params^
                _ = group_y^
                _ = group_ws^
                slab_end(ctx, slab_mark)
        _ = ykf^
        _ = exog^
        _ = y^
        _ = ctx^
        return len(orders) * bs
