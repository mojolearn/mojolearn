# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Opt-in AutoARIMA final fit of every chosen order in ONE device call
(`-D MOJOLEARN_ARIMA_FIT_GROUPS`, FAST + Apple, lane/apple-fast-w2-ts).

THE CAUSE. `AutoARIMA.fit` refits each chosen (order, series subset) with
`maxiter = 1000` through one `ARIMA.fit` per chosen order, one after the
other. On the board (64 series, d in {0, 1}, p, q in 0..3) that is up to 32
separate solves of a few series each, every one a GPU L-BFGS whose rounds
each launch a latency-bound Kalman filter (one thread per member, ~1,400
serial steps) over a handful of threads. The search phase (16 orders,
`maxiter = 20`, already grouped by `fast_order_search.mojo`) is small next
to it: the refit's wall time is the SUM over chosen orders of their round
counts times one filter latency.

THE CHANGE. Every chosen order keeps its own series, parameter vector,
`estimate_x0` start, Jones transforms and `OrderOptimizer` state machine,
exactly as the single-order fit (`order_min_lbfgs`) runs them. Orders that
share (d, r, k) share one contiguous Kalman workspace and ONE filter launch
per round (`fast_order_search.mojo`'s grouping, which the ARIMA_ORDER_BATCH
FULL_PASS showed member-for-member byte-identical to the single launch).
All groups advance in ONE round loop, one poll every ASYNC_READ_EVERY
rounds, and an order whose poll reads "not running" leaves the loop at
exactly the round where `order_min_lbfgs` breaks. A group whose orders are
all done stops launching. So each order sees the same evaluation/advance
sequence as its own fit, and its x, fx, n_iter and retcode are the single
fit's; wall time becomes about the slowest order's rounds times the number
of live groups, not the sum over orders.

The tail per order is `batched_fit_x`'s steps 4-5 and
`arima_fit_ptr_host`'s output writes: forward Jones, pack, the fitted
log-likelihood re-evaluated by `_loglike_at` (never rescaled from fx).

Nonseasonal, no exog, r <= 4 only (the grouped filter's instantiations);
the Python side keeps the per-order path for anything else and for
identity tracing. Nothing here is compiled into a build without the define.
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext, DeviceBuffer
from core.neural_context import process_ctx
from arima.estimator import _DEVCTX_SLOT, _upload_f32, _loglike_at, _write_list_f32, _write_list_i32
from arima.impl.batched_arima import _refuse_non_finite
from arima.impl.batched_fit import ARIMA_FIT_H, arima_fit_params, _download, _upload
from arima.impl.batched_kalman import KalmanWorkspace, fast_kalman_init_into, _launch_loop_ll_only
from arima.impl.fast_eval_ws import FastEvalWS
from arima.impl.fast_lbfgs_async import ASYNC_READ_EVERY
from arima.impl.fast_order_search import _initial_x
from arima.impl.fast_order_state import ARIMA_ORDER_BATCH, OrderOptimizer
from arima.impl.timeSeries.arima_helpers import batched_jones_transform
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams, pack, unpack, validate_order
from tsa.impl.timeSeries.arima_helpers import prepare_data

# Candidate (opt-in), lane/apple-fast-w2-ts, 2026-10-04: not yet measured.
# Expected bits: identical to the per-order refit (same state machine per
# order, grouped filter already proven member-identical). Quality gate:
# tools/arima_fitgroups_quality.sh (byte-identical fitted/forecast arrays).
comptime ARIMA_FIT_GROUPS = ARIMA_ORDER_BATCH and is_defined["MOJOLEARN_ARIMA_FIT_GROUPS"]()


def _f32_at(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("arima_fit_orders: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_at(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("arima_fit_orders: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def fit_orders_grouped(
    addrs: List[Int], orders: List[ARIMAOrder], sizes: List[Int], nobs: Int, maxiter: Int,
) raises -> Int:
    """Fit order j on its own `sizes[j]` series. `addrs` holds six host
    addresses per order, in this order: y (sizes[j] * nobs float32, series
    contiguous), params / x / x0 (sizes[j] * N_j float32 each, the packings
    of `arima_fit_ptr_host`), stats (2 * sizes[j] float32: loglike, fx),
    flags (2 * sizes[j] int32: n_iter, retcode). Returns the total number of
    fitted parameters written, sum_j sizes[j] * N_j."""
    comptime if not ARIMA_FIT_GROUPS:
        raise Error("AutoARIMA grouped refit was not compiled (MOJOLEARN_ARIMA_FIT_GROUPS)")
    else:
        var n_ord = len(orders)
        if n_ord < 1 or len(sizes) != n_ord or len(addrs) != 6 * n_ord:
            raise Error("arima_fit_orders: expected six addresses and one size per order")
        if nobs < 3 or maxiter < 1:
            raise Error("arima_fit_orders: invalid observation or iteration count")
        for j in range(n_ord):
            var o = orders[j]
            validate_order(o)
            if sizes[j] < 1:
                raise Error("arima_fit_orders: every order needs at least one series")
            if o.d < 0 or o.d > 2 or o.P != 0 or o.D != 0 or o.Q != 0 or o.s != 0 or o.n_exog != 0 or o.r() > 4:
                raise Error("arima_fit_orders: nonseasonal orders without exog, r <= 4, only")
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var hp = arima_fit_params(maxiter)
        var exog = ctx.enqueue_create_buffer[DType.float32](1)
        # Per order: the series, once differenced (batched_fit_x step 3) and
        # the start (steps 1-2), in input order.
        var ys = List[DeviceBuffer[DType.float32]]()
        var ykfs = List[DeviceBuffer[DType.float32]]()
        var x0s = List[List[Float32]]()
        for j in range(n_ord):
            var o = orders[j]
            var bs = sizes[j]
            var y = _upload_f32(ctx, _f32_at(addrs[6 * j]), bs * nobs)
            _refuse_non_finite(ctx, y, bs * nobs, "y")
            var nkf = nobs - o.d
            var ykf = ctx.enqueue_create_buffer[DType.float32](bs * nkf)
            if o.d > 0:
                prepare_data(ctx, ykf, y, bs, nobs, o.d, 0, 0)
            else:
                ctx.enqueue_copy(dst_buf=ykf, src_buf=y)
            ctx.synchronize()
            var x0 = _initial_x(ctx, y, exog, bs, nobs, o)
            ys.append(y^)
            ykfs.append(ykf^)
            x0s.append(x0^)
        # Groups by (d, r, k): one shared workspace and filter launch each.
        # `slot[j]` is order j's index in the flat evals/states lists, which
        # are filled group by group; `g_first`/`g_end` bound each group.
        var evals = List[FastEvalWS]()
        var states = List[OrderOptimizer]()
        var slot_order = List[Int]()
        var g_ws = List[KalmanWorkspace]()
        var g_y = List[DeviceBuffer[DType.float32]]()
        var g_params = List[ARIMAParams]()
        var g_rd = List[Int]()
        var g_k = List[Int]()
        var g_nkf = List[Int]()
        var g_members = List[Int]()
        var g_first = List[Int]()
        var g_end = List[Int]()
        for d in range(3):
            for rd in range(1, 5):
                for k in range(2):
                    var ids = List[Int]()
                    var members = 0
                    for j in range(n_ord):
                        if orders[j].d == d and orders[j].r() == rd and orders[j].k == k:
                            ids.append(j)
                            members += sizes[j] * (orders[j].complexity() + 1)
                    if len(ids) == 0:
                        continue
                    var nkf = nobs - d
                    var group_order = ARIMAOrder(rd, 0, 0, 0, 0, 0, 0, 1, 0)
                    var group_ws = KalmanWorkspace(ctx, group_order, members, nkf, 0)
                    var group_y = ctx.enqueue_create_buffer[DType.float32](members * nkf)
                    var group_params = ARIMAParams(ctx, group_order, members)
                    g_first.append(len(states))
                    var offset = 0
                    for t in range(len(ids)):
                        var j = ids[t]
                        var okf = orders[j].without_diff()
                        var ew = FastEvalWS(ctx, ykfs[j], sizes[j], nkf, okf,
                                            group_ws, group_y, group_params.mu, offset)
                        var state = OrderOptimizer(ctx, ew, sizes[j], Float32(nobs - 1), okf,
                                                   x0s[j], hp, ARIMA_FIT_H)
                        offset += ew.eb
                        evals.append(ew^)
                        states.append(state^)
                        slot_order.append(j)
                    g_end.append(len(states))
                    g_ws.append(group_ws^)
                    g_y.append(group_y^)
                    g_params.append(group_params^)
                    g_rd.append(rd)
                    g_k.append(k)
                    g_nkf.append(nkf)
                    g_members.append(members)
        var n_groups = len(g_first)
        var live = List[Bool](length=len(states), fill=True)
        var g_live = List[Bool](length=n_groups, fill=True)
        var rounds = 0
        var max_rounds = hp.max_iterations * max(1, hp.max_linesearch) + 1
        while rounds < max_rounds:
            if rounds % ASYNC_READ_EVERY == 0:
                for i in range(len(states)):
                    if live[i]:
                        states[i].enqueue_poll(ctx)
                ctx.synchronize()
                var any_live = False
                for g in range(n_groups):
                    var gl = False
                    for i in range(g_first[g], g_end[g]):
                        # the round `order_min_lbfgs` breaks at, per order
                        if live[i] and not states[i].running():
                            live[i] = False
                        gl = gl or live[i]
                    g_live[g] = gl
                    any_live = any_live or gl
                if not any_live:
                    break
            for g in range(n_groups):
                if not g_live[g]:
                    continue
                for i in range(g_first[g], g_end[g]):
                    if live[i]:
                        ref state = states[i]
                        ref ew = evals[i]
                        ew.prepare(ctx, state.order, state.h, state.cand, state.bad)
                        fast_kalman_init_into(ctx, ew.t_params, state.order, ew.eb, ew.ws)
                # Members of finished orders ride along on stale inputs; every
                # member is its own thread, so no live member reads them.
                _launch_loop_ll_only(ctx, g_y[g], g_params[g], g_ws[g],
                                     g_rd[g], g_nkf[g], g_members[g], g_k[g], 0, 32)
                for i in range(g_first[g], g_end[g]):
                    if live[i]:
                        ref state = states[i]
                        ref ew = evals[i]
                        ew.finish(ctx, state.h, state.scale, state.cand, state.d_grad,
                                  state.d_x_pert, state.f_fxc, state.gradc, state.bad)
                        state.advance(ctx)
            rounds += 1
        # batched_fit_x steps 4-5 and arima_fit_ptr_host's writes, per order
        var written = 0
        for i in range(len(states)):
            var j = slot_order[i]
            var order = orders[j]
            var bs = sizes[j]
            var N = order.complexity()
            var result = states[i].result(ctx)
            var d_x = ctx.enqueue_create_buffer[DType.float32](N * bs)
            _upload(ctx, d_x, result.x)
            var raw = ARIMAParams(ctx, order, bs)
            var fitted = ARIMAParams(ctx, order, bs)
            unpack(ctx, raw, order, bs, d_x)
            batched_jones_transform(ctx, order, bs, False, raw, fitted)
            var d_t_x = ctx.enqueue_create_buffer[DType.float32](N * bs)
            pack(ctx, fitted, order, bs, d_t_x)
            ctx.synchronize()
            var t_x = _download(ctx, d_t_x, N * bs)
            var ll = _loglike_at(ctx, ys[j], exog, bs, nobs, order, fitted)
            _write_list_f32(_f32_at(addrs[6 * j + 1]), t_x, 0)
            _write_list_f32(_f32_at(addrs[6 * j + 2]), result.x, 0)
            _write_list_f32(_f32_at(addrs[6 * j + 3]), x0s[j], 0)
            _write_list_f32(_f32_at(addrs[6 * j + 4]), ll, 0)
            _write_list_f32(_f32_at(addrs[6 * j + 4]), result.fx, bs)
            _write_list_i32(_i32_at(addrs[6 * j + 5]), result.n_iter, 0)
            _write_list_i32(_i32_at(addrs[6 * j + 5]), result.retcode, bs)
            written += N * bs
            _ = result^
            _ = raw^
            _ = fitted^
            _ = d_x^
            _ = d_t_x^
        ctx.synchronize()
        _ = states^
        _ = evals^
        _ = g_params^
        _ = g_y^
        _ = g_ws^
        _ = ykfs^
        _ = ys^
        _ = exog^
        _ = ctx^
        return written
