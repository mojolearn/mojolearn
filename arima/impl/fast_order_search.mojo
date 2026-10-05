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
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext, DeviceBuffer
from core.neural_context import process_ctx
from arima.estimator import _DEVCTX_SLOT, _upload_f32, _loglike_at, _write_list_f32, _write_list_i32
from arima.impl.batched_arima import _refuse_non_finite
from arima.impl.batched_fit import ARIMA_FIT_H, arima_fit_params, _download, _upload
from arima.impl.batched_kalman import KalmanWorkspace, fast_kalman_init_into, _launch_loop_ll_only
from arima.impl.estimate_x0 import estimate_x0_x
from arima.impl.fast_eval_ws import FastEvalWS
from arima.impl.fast_lbfgs_async import ASYNC_READ_EVERY
from arima.impl.fast_order_state import ARIMA_ORDER_BATCH, OrderOptimizer
from arima.impl.timeSeries.arima_helpers import batched_jones_transform
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams, pack, unpack, validate_order
from tsa.impl.timeSeries.arima_helpers import prepare_data
from glm.impl.qn.qn_util import LBFGSParam

#: MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT (FAST+Apple via ARIMA_ORDER_BATCH,
#: default off, READY-AB). The grouped search ran its Kalman-dimension groups
#: (rd = 1..4) one after another, each to its own convergence, so the
#: search's wall time was the SUM of four optimizer loops, each paying its
#: own polls and drain waits. Every group's optimizer is per-series and
#: independent, so all groups now advance in ONE round loop: each round
#: enqueues every live group's prepare / filter / finish / step, one poll
#: covers all groups, and a group stops being enqueued once its own poll
#: shows no running series. Per-group rounds, polls and stopping are the
#: ones the sequential loop had, so every order's likelihood and optimum
#: are unchanged; only the overlap differs (rounds = max over groups, not sum).
comptime ARIMA_FAST_GROUPS_CONCURRENT = (
    ARIMA_ORDER_BATCH and is_defined["MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT"]()
)

#: MOJOLEARN_ARIMA_FAST_SEARCH_REUSE (FAST+Apple via ARIMA_ORDER_BATCH,
#: default off, READY-AB). The search already runs every candidate order to
#: the fit's own optimizer (same estimate_x0 start, same per-series device
#: L-BFGS, same h, same scale); AutoARIMA.fit then refitted every chosen
#: order from scratch, the same work again. With this switch the search also
#: returns each order's fitted parameters, optimum x, start x0, fx, n_iter and
#: retcode (arima_order_search's optional fit buffers), and AutoARIMA.fit
#: reuses them when the fit's maxiter equals the search's (the only case where
#: the refit is the same optimization). Python side: _x_sequence_autoarima.py.
comptime ARIMA_FAST_SEARCH_REUSE = (
    ARIMA_ORDER_BATCH and is_defined["MOJOLEARN_ARIMA_FAST_SEARCH_REUSE"]()
)


def order_fit_f32_offset(orders: List[ARIMAOrder], i: Int, bs: Int) -> Int:
    """Float32 offset of order `i`'s fit block: per order, params, x, x0
    (bs * N each) then fx (bs), N = complexity()."""
    var off = 0
    for j in range(i):
        off += bs * (3 * orders[j].complexity() + 1)
    return off


def _finish_order(
    ctx: DeviceContext, mut y: DeviceBuffer[DType.float32],
    mut exog: DeviceBuffer[DType.float32], bs: Int, nobs: Int,
    order: ARIMAOrder, mut state: OrderOptimizer, x0: List[Float32],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin], row: Int,
    want_fit: Bool,
    fit_f32: Optional[MutPointer[Float32, MutUntrackedOrigin]],
    fit_i32: Optional[MutPointer[Int32, MutUntrackedOrigin]],
    f_off: Int,
) raises:
    """Order `row`'s fitted log-likelihood into `out_ptr` (as public
    ARIMA.fit evaluates it) and, when `want_fit`, its fit block."""
    var result = state.result(ctx)
    var n = bs * order.complexity()
    var x = ctx.enqueue_create_buffer[DType.float32](n)
    _upload(ctx, x, result.x)
    var raw = ARIMAParams(ctx, order, bs)
    var fitted = ARIMAParams(ctx, order, bs)
    unpack(ctx, raw, order, bs, x)
    batched_jones_transform(ctx, order, bs, False, raw, fitted)
    # Reevaluate the fitted likelihood exactly as public
    # ARIMA.fit does; never recover it by rescaling fx.
    var ll = _loglike_at(ctx, y, exog, bs, nobs, order, fitted)
    _write_list_f32(out_ptr, ll, row * bs)
    if want_fit:
        # batched_fit_x's t_x: the fitted parameters, packed
        var t_x_buf = ctx.enqueue_create_buffer[DType.float32](n)
        pack(ctx, fitted, order, bs, t_x_buf)
        var t_x = _download(ctx, t_x_buf, n)
        var fp = fit_f32.value()
        _write_list_f32(fp, t_x, f_off)
        _write_list_f32(fp, result.x, f_off + n)
        _write_list_f32(fp, x0, f_off + 2 * n)
        _write_list_f32(fp, result.fx, f_off + 3 * n)
        var ip = fit_i32.value()
        _write_list_i32(ip, result.n_iter, row * 2 * bs)
        _write_list_i32(ip, result.retcode, row * 2 * bs + bs)
        _ = t_x_buf^
    _ = result^
    _ = raw^
    _ = fitted^
    _ = x^


struct _OrderGroup(Movable):
    """One Kalman-dimension group's shared filter buffers (concurrent search)."""
    var rd: Int
    var k: Int
    var members: Int
    var live: Bool
    var ws: KalmanWorkspace
    var y: DeviceBuffer[DType.float32]
    var params: ARIMAParams

    def __init__(out self, ctx: DeviceContext, rd: Int, k: Int, members: Int, nkf: Int) raises:
        var group_order = ARIMAOrder(rd, 0, 0, 0, 0, 0, 0, 1, 0)
        self.rd = rd
        self.k = k
        self.members = members
        self.live = True
        self.ws = KalmanWorkspace(ctx, group_order, members, nkf, 0)
        self.y = ctx.enqueue_create_buffer[DType.float32](members * nkf)
        self.params = ARIMAParams(ctx, group_order, members)


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
    return order_search_fit(y_ptr, out_ptr, orders, bs, nobs, maxiter, False, None, None)


def order_search_fit(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    orders: List[ARIMAOrder], bs: Int, nobs: Int, maxiter: Int,
    want_fit: Bool,
    fit_f32: Optional[MutPointer[Float32, MutUntrackedOrigin]],
    fit_i32: Optional[MutPointer[Int32, MutUntrackedOrigin]],
) raises -> Int:
    """`order_search_loglike`; with `want_fit` (ARIMA_FAST_SEARCH_REUSE) it
    also writes every order's fit block: f32 at order_fit_f32_offset (params,
    x, x0, fx) and i32 at row * 2 * bs (n_iter, retcode)."""
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
        if want_fit and (not fit_f32 or not fit_i32):
            raise Error("AutoARIMA grouped orders: fit output requested without buffers")
        comptime if ARIMA_FAST_GROUPS_CONCURRENT:
            _search_concurrent(ctx, y, exog, ykf, orders, bs, nobs, nkf, hp,
                               out_ptr, want_fit, fit_f32, fit_i32)
            _ = ykf^
            _ = exog^
            _ = y^
            _ = ctx^
            return len(orders) * bs
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
                var group_order = ARIMAOrder(rd, 0, 0, 0, 0, 0, 0, 1, 0)
                var group_ws = KalmanWorkspace(ctx, group_order, members, nkf, 0)
                var group_y = ctx.enqueue_create_buffer[DType.float32](members * nkf)
                var group_params = ARIMAParams(ctx, group_order, members)
                var evals = List[FastEvalWS]()
                var states = List[OrderOptimizer]()
                var x0s = List[List[Float32]]()
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
                    x0s.append(x0^)
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
                    _finish_order(ctx, y, exog, bs, nobs, orders[ids[j]], states[j], x0s[j],
                                  out_ptr, ids[j], want_fit, fit_f32, fit_i32,
                                  order_fit_f32_offset(orders, ids[j], bs))
                ctx.synchronize()
                _ = states^
                _ = evals^
                _ = group_params^
                _ = group_y^
                _ = group_ws^
        _ = ykf^
        _ = exog^
        _ = y^
        _ = ctx^
        return len(orders) * bs


def _search_concurrent(
    ctx: DeviceContext, mut y: DeviceBuffer[DType.float32],
    mut exog: DeviceBuffer[DType.float32], mut ykf: DeviceBuffer[DType.float32],
    orders: List[ARIMAOrder], bs: Int, nobs: Int, nkf: Int, hp: LBFGSParam,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin], want_fit: Bool,
    fit_f32: Optional[MutPointer[Float32, MutUntrackedOrigin]],
    fit_i32: Optional[MutPointer[Int32, MutUntrackedOrigin]],
) raises:
    """ARIMA_FAST_GROUPS_CONCURRENT: the sequential group loop of
    `order_search_fit`, with every (rd, k) group advanced in one round loop.
    Each group keeps its own filter launch, poll verdict and stop round."""
    var groups = List[_OrderGroup]()
    var evals = List[FastEvalWS]()
    var states = List[OrderOptimizer]()
    var x0s = List[List[Float32]]()
    var owner = List[Int]()     # the group of each state
    var row = List[Int]()       # the grid row (output index) of each state
    for rd in range(1, 5):
        for k in range(2):
            var members = 0
            for i in range(len(orders)):
                if orders[i].r() == rd and orders[i].k == k:
                    members += bs * (orders[i].complexity() + 1)
            if members == 0:
                continue
            groups.append(_OrderGroup(ctx, rd, k, members, nkf))
            var g = len(groups) - 1
            var offset = 0
            for i in range(len(orders)):
                if orders[i].r() != rd or orders[i].k != k:
                    continue
                var order = orders[i]
                var okf = order.without_diff()
                var x0 = _initial_x(ctx, y, exog, bs, nobs, order)
                ref grp = groups[g]
                var ew = FastEvalWS(ctx, ykf, bs, nkf, okf,
                                    grp.ws, grp.y, grp.params.mu, offset)
                var state = OrderOptimizer(ctx, ew, bs, Float32(nobs - 1), okf,
                                           x0, hp, ARIMA_FIT_H)
                offset += ew.eb
                evals.append(ew^)
                states.append(state^)
                x0s.append(x0^)
                owner.append(g)
                row.append(i)
    var rounds = 0
    var max_rounds = hp.max_iterations * max(1, hp.max_linesearch) + 1
    while rounds < max_rounds:
        if rounds % ASYNC_READ_EVERY == 0:
            # one wait for every live group's poll
            for j in range(len(states)):
                if groups[owner[j]].live:
                    states[j].enqueue_poll(ctx)
            ctx.synchronize()
            for g in range(len(groups)):
                if not groups[g].live:
                    continue
                var running = False
                for j in range(len(states)):
                    if owner[j] == g:
                        running = running or states[j].running()
                groups[g].live = running
            var any_live = False
            for g in range(len(groups)):
                any_live = any_live or groups[g].live
            if not any_live:
                break
        for j in range(len(states)):
            if not groups[owner[j]].live:
                continue
            ref state = states[j]
            ref ew = evals[j]
            ew.prepare(ctx, state.order, state.h, state.cand, state.bad)
            fast_kalman_init_into(ctx, ew.t_params, state.order, ew.eb, ew.ws)
        for g in range(len(groups)):
            ref grp = groups[g]
            if not grp.live:
                continue
            _launch_loop_ll_only(ctx, grp.y, grp.params, grp.ws,
                                 grp.rd, nkf, grp.members, grp.k, 0, 32)
        for j in range(len(states)):
            if not groups[owner[j]].live:
                continue
            ref state = states[j]
            ref ew = evals[j]
            ew.finish(ctx, state.h, state.scale, state.cand, state.d_grad,
                      state.d_x_pert, state.f_fxc, state.gradc, state.bad)
            state.advance(ctx)
        rounds += 1
    for j in range(len(states)):
        _finish_order(ctx, y, exog, bs, nobs, orders[row[j]], states[j], x0s[j],
                      out_ptr, row[j], want_fit, fit_f32, fit_i32,
                      order_fit_f32_offset(orders, row[j], bs))
    ctx.synchronize()
    _ = states^
    _ = evals^
    _ = groups^
