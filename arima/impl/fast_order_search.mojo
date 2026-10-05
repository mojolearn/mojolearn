# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Opt-in nonseasonal AutoARIMA order fits batched by Kalman dimension.

Orders keep separate parameter vectors, Jones transforms and GPU L-BFGS
state. Their gradient members share a contiguous Kalman workspace and one
filter launch per dimension, improving occupancy for the small board batch.
Output rows retain input grid order, so Python's first-minimum IC tie rule
is unchanged. No order pruning or approximation is performed.

lane/fam2-timeseries (2026-10-04), IDENTICAL on every vendor, each with its
own `_OFF` (all off under MOJOLEARN_IDN_ALL_OFF; FAST builds unchanged):

ARIMA_ORDER_CAP (`-D MOJOLEARN_IDN_ARIMA_ORDER_CAP_OFF=1`). The group's held
workspace was unbounded under IDENTICAL (members = batch x sum over the
group's orders of N + 1, four float32 arrays of members x n_obs). A group is
now split, in grid order, into chunks whose members x n_obs stay under
`eval_ws_fits`' bound; a search holding ONE order that alone exceeds it
returns 0 before any device work, and the caller takes the per-order fits
(whose evaluation has its own bound). Chunking moves no bit: every order
keeps its own parameters and every member is its own filter thread.

ARIMA_ORDER_DEVICE (`-D MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF=1`). The search
stays on the device between the upload and one final download: each order's
starting point goes from `pack` straight into its optimizer (was a download,
a host finite walk and a re-upload per order; the finite test is a device
flag read with the first poll), and the log-likelihood at each optimum is
ONE more grouped evaluation at `x` (member 0 of each order: the same unpack,
Jones transform and filter recurrence `_loglike_at` ran one order at a time
after downloading and re-uploading `x`), marked -inf on a refusal code by
`order_ll_kernel`.

ARIMA_ORDER_SEASONAL (`-D MOJOLEARN_IDN_ARIMA_ORDER_SEASONAL_OFF=1`, needs
ORDER_DEVICE). Seasonal grids (P, Q, one D, one s) take the grouped search:
orders are grouped by r = max(p + s P, q + s Q + 1) and k after one shared
differencing (d, D, s), r <= 5 as `validate_order` allows.

ARIMA_ORDER_IC_DEVICE (`-D MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF=1`, needs
ORDER_DEVICE). The information criterion and the first-minimum order choice
run on the device (`order_ic_argmin_kernel`): ic = fma(-2, loglike, penalty)
in float32, one rounding (the product is exact, so no contraction choice
exists), `np.argmin`'s rule. Only the chosen index and its criterion cross
back. BITS: the criterion was float64 on the host from the float32
log-likelihood; it is now that value rounded once to float32, on every
vendor, and the host column takes the same formula
(`bindings/hotpath_helpers.mojo::ic_running_min_f32_binding`).
"""
from std.math import inf, isfinite
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext, DeviceBuffer
from core.neural_context import process_ctx
from arima.estimator import _DEVCTX_SLOT, _upload_f32, _write_list_f32, _write_list_i32
from arima.impl.batched_arima import _refuse_non_finite, canonical_nan
from arima.impl.batched_fit import ARIMA_FIT_H, arima_fit_params
from arima.impl.batched_kalman import (
    KalmanWorkspace, eval_ws_fits, fast_kalman_init_into, _launch_loop_ll_only,
)
from arima.impl.estimate_x0 import estimate_x0_x
from arima.impl.fast_eval_ws import FastEvalWS
from arima.impl.fast_lbfgs_async import ASYNC_READ_EVERY
from arima.impl.fast_order_state import ARIMA_ORDER_BATCH, OrderOptimizer
from arima.impl.timeSeries.arima_helpers import batched_jones_transform
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams, pack, validate_order
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from tsa.impl.timeSeries.arima_helpers import prepare_data


comptime _ORDER_IDN = (
    ARIMA_ORDER_BATCH and GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime ARIMA_ORDER_CAP = _ORDER_IDN and not is_defined["MOJOLEARN_IDN_ARIMA_ORDER_CAP_OFF"]()
comptime ARIMA_ORDER_DEVICE = _ORDER_IDN and not is_defined["MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF"]()
comptime ARIMA_ORDER_SEASONAL = (
    ARIMA_ORDER_DEVICE and not is_defined["MOJOLEARN_IDN_ARIMA_ORDER_SEASONAL_OFF"]()
)
comptime ARIMA_ORDER_IC_DEVICE = (
    ARIMA_ORDER_DEVICE and not is_defined["MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF"]()
)
comptime ORDER_TPB = 128


def order_search_caps() -> Int:
    """What this build's grouped search does, for the Python glue: bit 0 the
    device search entry (`order_search_device`), bit 1 seasonal grids, bit 2
    the criterion and order choice on the device (float32 criterion)."""
    var caps = 0
    comptime if ARIMA_ORDER_DEVICE:
        caps |= 1
    comptime if ARIMA_ORDER_SEASONAL:
        caps |= 2
    comptime if ARIMA_ORDER_IC_DEVICE:
        caps |= 4
    return caps


def order_x0_flag_kernel(
    flag: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """flag[0] = 1 when any packed starting parameter is non-finite. Every
    such thread writes the same word; the flag starts 0."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    if not isfinite(x.unsafe_load(i)):
        flag.unsafe_store(0, Int32(1))


def order_ll_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    ll: MutPointer[Float32, MutAnyOrigin],
    info0: MutPointer[Int32, MutAnyOrigin],
    info1: MutPointer[Int32, MutAnyOrigin],
    bs_in: Int32,
    row_in: Int32,
):
    """One thread per series: row `row` of the (order, series) table takes
    the base member's log-likelihood, or the constant -inf when either Kalman
    refusal code is set (`_mark_infeasible`'s rule on member 0)."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var bs = Int(bs_in)
    if b >= bs:
        return
    var v = ll.unsafe_load(b)
    if info0.unsafe_load(b) != Int32(0) or info1.unsafe_load(b) != Int32(0):
        v = -inf[DType.float32]()
    output.unsafe_store(Int(row_in) * bs + b, v)


def order_ic_argmin_kernel(
    best: MutPointer[Int32, MutAnyOrigin],
    best_ic: MutPointer[Float32, MutAnyOrigin],
    ll: MutPointer[Float32, MutAnyOrigin],
    pen: MutPointer[Float32, MutAnyOrigin],
    bs_in: Int32,
    n_orders_in: Int32,
):
    """One thread per series: ic = fma(-2, loglike, penalty) for every order
    in grid order and the running first minimum, `np.argmin`'s rule (a NaN
    is the minimum, the first NaN wins), as `ic_running_min_f64` folds it on
    the host. A NaN criterion is stored as the canonical NaN."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var bs = Int(bs_in)
    if b >= bs:
        return
    var cur = Float32(0.0)
    var at = Int32(0)
    for t in range(Int(n_orders_in)):
        var v = ftz(identical_mul_add(Float32(-2.0), ftz(ll.unsafe_load(t * bs + b)), pen.unsafe_load(t)))
        if t == 0:
            cur = v
        else:
            var take = False
            if cur == cur:
                take = (v != v) or v < cur
            if take:
                cur = v
                at = Int32(t)
    if cur != cur:
        cur = canonical_nan()
    best.unsafe_store(b, at)
    best_ic.unsafe_store(b, cur)


def _initial_x_device(ctx: DeviceContext, mut y: DeviceBuffer[DType.float32],
                      mut exog: DeviceBuffer[DType.float32], bs: Int, nobs: Int,
                      order: ARIMAOrder,
                      mut flag: DeviceBuffer[DType.int32]) raises -> DeviceBuffer[DType.float32]:
    """`_initial_x` with the packed starting point left on the device and
    the finite test raised into `flag` by a kernel (read by the caller with
    its first poll). One wait, so the temporaries outlive their launches."""
    var params = ARIMAParams(ctx, order, bs)
    var info = ctx.enqueue_create_buffer[DType.int32](bs)
    var start = estimate_x0_x(ctx, params, y, exog, bs, nobs, order, info)
    var inv = ARIMAParams(ctx, order, bs)
    batched_jones_transform(ctx, order, bs, True, params, inv)
    var n = order.complexity() * bs
    var x = ctx.enqueue_create_buffer[DType.float32](max(1, n))
    pack(ctx, inv, order, bs, x)
    ctx.enqueue_function[order_x0_flag_kernel](
        flag.unsafe_ptr(), x.unsafe_ptr(), Int32(n),
        grid_dim=((n + ORDER_TPB - 1) // ORDER_TPB, 1, 1), block_dim=(ORDER_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = start^
    _ = params^
    _ = inv^
    _ = info^
    return x^


def _order_chunks(orders: List[ARIMAOrder], r: Int, k: Int, bs: Int,
                  nkf: Int) -> List[List[Int]]:
    """The grid positions of the orders with state dimension `r` and
    intercept flag `k`, in grid order, split into chunks whose stacked
    members fit the held-workspace bound (ARIMA_ORDER_CAP; one chunk
    otherwise). An order always lands in a chunk, alone if need be: the
    caller has already refused a search whose single order exceeds the
    bound."""
    var chunks = List[List[Int]]()
    var cur = List[Int]()
    var members = 0
    for i in range(len(orders)):  # small-loop(orders: the AutoARIMA order grid, a few dozen entries): plan entries, no series data
        if orders[i].r() != r or orders[i].k != k:
            continue
        var add = bs * (orders[i].complexity() + 1)
        var split = False
        comptime if ARIMA_ORDER_CAP:
            split = len(cur) > 0 and not eval_ws_fits(members + add, nkf)
        if split:
            chunks.append(cur.copy())
            cur = List[Int]()
            members = 0
        cur.append(i)
        members += add
    if len(cur) > 0:
        chunks.append(cur.copy())
    return chunks^


def _order_search_core(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    orders: List[ARIMAOrder], bs: Int, nobs: Int, maxiter: Int,
    pen: List[Float32], want_ic: Bool,
    mut ll_out: List[Float32], mut best_out: List[Int32], mut ic_out: List[Float32],
) raises -> Int:
    """The grouped search. Returns 0 before any device work when the grid is
    not one this build groups (the caller takes the per-order fits), else
    the count of results: `len(orders) * bs` log-likelihoods in `ll_out`
    (grid order), or with `want_ic` (ARIMA_ORDER_IC_DEVICE) `bs` chosen grid
    positions in `best_out` and their criteria in `ic_out`."""
    if bs < 1 or nobs < 3 or maxiter < 1 or len(orders) < 1:
        raise Error("AutoARIMA grouped orders: invalid shape or iteration count")
    var first = orders[0]
    var seasonal = False
    comptime if ARIMA_ORDER_SEASONAL:
        seasonal = True
    var device = False
    comptime if ARIMA_ORDER_DEVICE:
        device = True
    var nkf = nobs - first.n_diff()
    if nkf < 3:
        return 0
    for i in range(len(orders)):  # small-loop(orders: the AutoARIMA order grid, a few dozen entries): plan entries, no series data
        var o = orders[i]
        validate_order(o)
        if o.n_exog != 0 or o.d != first.d or o.D != first.D:
            raise Error("AutoARIMA grouped orders require one d, one D and no exogenous regressors")
        if seasonal:
            # one differencing for the whole grid: with D > 0 every order
            # carries the period; r <= 5 is `validate_order`'s bound
            if o.n_diff() != first.n_diff():
                return 0
        elif o.P != 0 or o.D != 0 or o.Q != 0 or o.s != 0 or o.r() > 4:
            if device:
                return 0
            raise Error("AutoARIMA grouped orders require a nonseasonal same-d grid with r <= 4")
        comptime if ARIMA_ORDER_CAP:
            if not eval_ws_fits(bs * (o.complexity() + 1), nkf):
                return 0
    if want_ic and (not device or len(pen) != len(orders)):
        raise Error("AutoARIMA grouped orders: the device criterion needs one penalty per order")
    var n_orders = len(orders)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var y = _upload_f32(ctx, y_ptr, bs * nobs)
    _refuse_non_finite(ctx, y, bs * nobs, "y")
    var exog = ctx.enqueue_create_buffer[DType.float32](1)
    var ykf = ctx.enqueue_create_buffer[DType.float32](bs * nkf)
    if first.need_diff():
        prepare_data(ctx, ykf, y, bs, nobs, first.d, first.D, first.s)
    else:
        ctx.enqueue_copy(dst_buf=ykf, src_buf=y)
    # the (order, series) log-likelihood table, grid order (ORDER_DEVICE)
    var ll_all = ctx.enqueue_create_buffer[DType.float32](n_orders * bs)
    var x0_flag = ctx.enqueue_create_buffer[DType.int32](1)
    var x0_flag_host = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_memset(x0_flag, Int32(0))
    ctx.synchronize()
    x0_flag_host.unsafe_ptr().unsafe_store(0, Int32(0))
    var hp = arima_fit_params(maxiter)
    var ll_grid = (bs + ORDER_TPB - 1) // ORDER_TPB
    # Grouping never changes the output index. k is uniform within a
    # filter launch, as are rd and observation count after differencing.
    for rd in range(1, 6):
        for k in range(2):
            var chunks = _order_chunks(orders, rd, k, bs, nkf)
            for c in range(len(chunks)):
                var ids = chunks[c].copy()
                var members = 0
                for j in range(len(ids)):  # small-loop(ids: orders in one chunk of the grid): member-count plan arithmetic, no data
                    members += bs * (orders[ids[j]].complexity() + 1)
                var group_order = ARIMAOrder(rd, 0, 0, 0, 0, 0, 0, 1, 0)
                var group_ws = KalmanWorkspace(ctx, group_order, members, nkf, 0)
                var group_y = ctx.enqueue_create_buffer[DType.float32](members * nkf)
                var group_params = ARIMAParams(ctx, group_order, members)
                var evals = List[FastEvalWS]()
                var states = List[OrderOptimizer]()
                var offset = 0
                for j in range(len(ids)):
                    var order = orders[ids[j]]
                    var okf = order.without_diff()
                    # lane cpu3-seq (2026-10-04): every build takes the
                    # device start (was a download, a host finite walk and a
                    # re-upload per order outside ARIMA_ORDER_DEVICE)
                    var x_dev = _initial_x_device(ctx, y, exog, bs, nobs, order, x0_flag)
                    var ew = FastEvalWS(ctx, ykf, bs, nkf, okf,
                                        group_ws, group_y, group_params.mu, offset)
                    var state = OrderOptimizer(ctx, ew, bs, Float32(nobs - 1), okf,
                                               x_dev^, hp, ARIMA_FIT_H)
                    offset += ew.eb
                    evals.append(ew^)
                    states.append(state^)
                ctx.enqueue_copy(dst_ptr=x0_flag_host.unsafe_ptr(), src_buf=x0_flag)
                var rounds = 0
                var max_rounds = hp.max_iterations * max(1, hp.max_linesearch) + 1
                while rounds < max_rounds:
                    if rounds % ASYNC_READ_EVERY == 0:
                        for j in range(len(states)):
                            states[j].enqueue_poll(ctx)
                        ctx.synchronize()
                        if x0_flag_host.unsafe_ptr().unsafe_load(0) != Int32(0):
                            raise Error("AutoARIMA: non-finite initial parameter")
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
                # lane cpu3-seq (2026-10-04): every build re-evaluates on the
                # device (was, outside ARIMA_ORDER_DEVICE, a download, a
                # re-upload and a host copy of each order's log-likelihoods)
                # The fitted log-likelihood, re-evaluated as ARIMA.fit
                # does (never recovered by rescaling fx): one more
                # grouped evaluation AT x; member 0 of each order is the
                # unperturbed point.
                for j in range(len(states)):
                    ref state = states[j]
                    ref ew = evals[j]
                    ew.prepare(ctx, state.order, state.h, state.x, state.bad)
                    fast_kalman_init_into(ctx, ew.t_params, state.order, ew.eb, ew.ws)
                _launch_loop_ll_only(ctx, group_y, group_params, group_ws,
                                     rd, nkf, members, k, 0, 32)
                for j in range(len(states)):
                    ref ew = evals[j]
                    ctx.enqueue_function[order_ll_kernel](
                        ll_all.unsafe_ptr(), ew.ws.loglike.unsafe_ptr(),
                        ew.ws.info_init.unsafe_ptr(), ew.ws.info_loop.unsafe_ptr(),
                        Int32(bs), Int32(ids[j]),
                        grid_dim=(ll_grid, 1, 1), block_dim=(ORDER_TPB, 1, 1),
                    )
                ctx.synchronize()
                _ = states^
                _ = evals^
                _ = group_params^
                _ = group_y^
                _ = group_ws^
    var written = n_orders * bs
    # lane cpu3-seq (2026-10-04): the results land straight in the caller's
    # lists, one copy each (no download-and-walk); `want_ic` is refused above
    # in a build without ARIMA_ORDER_DEVICE
    if want_ic:
        var d_pen = ctx.enqueue_create_buffer[DType.float32](n_orders)
        var h_pen = pen.copy()
        ctx.enqueue_copy(dst_buf=d_pen, src_ptr=h_pen.unsafe_ptr())
        var d_best = ctx.enqueue_create_buffer[DType.int32](bs)
        var d_ic = ctx.enqueue_create_buffer[DType.float32](bs)
        ctx.enqueue_function[order_ic_argmin_kernel](
            d_best.unsafe_ptr(), d_ic.unsafe_ptr(), ll_all.unsafe_ptr(), d_pen.unsafe_ptr(),
            Int32(bs), Int32(n_orders),
            grid_dim=(ll_grid, 1, 1), block_dim=(ORDER_TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=best_out.unsafe_ptr(), src_buf=d_best)
        ctx.enqueue_copy(dst_ptr=ic_out.unsafe_ptr(), src_buf=d_ic)
        ctx.synchronize()
        _ = h_pen^
        _ = d_pen^
        _ = d_best^
        _ = d_ic^
        written = bs
    else:
        ctx.enqueue_copy(dst_ptr=ll_out.unsafe_ptr(), src_buf=ll_all)
        ctx.synchronize()
    _ = x0_flag_host^
    _ = x0_flag^
    _ = ll_all^
    _ = ykf^
    _ = exog^
    _ = y^
    _ = ctx^
    return written


def order_search_loglike(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    orders: List[ARIMAOrder], bs: Int, nobs: Int, maxiter: Int,
) raises -> Int:
    """Every (order, series) fitted log-likelihood into `out_ptr`, grid
    order. Returns the count written; 0 (nothing written) when the grid is
    one this build does not group (ARIMA_ORDER_CAP / ARIMA_ORDER_DEVICE)."""
    comptime if not ARIMA_ORDER_BATCH:
        raise Error("AutoARIMA grouped orders were not compiled")
    else:
        var n = len(orders) * bs
        var ll = List[Float32](length=n, fill=Float32(0.0))
        var best = List[Int32](length=1, fill=Int32(0))
        var ic = List[Float32](length=1, fill=Float32(0.0))
        var pen = List[Float32]()
        var written = _order_search_core(y_ptr, orders, bs, nobs, maxiter, pen, False, ll, best, ic)
        if written > 0:
            _write_list_f32(out_ptr, ll, 0)
        return written


def order_search_device(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    best_ptr: MutPointer[Int32, MutUntrackedOrigin],
    orders: List[ARIMAOrder], pen: List[Float32],
    bs: Int, nobs: Int, maxiter: Int, want_ic: Bool,
) raises -> Int:
    """lane/fam2-timeseries: the grouped search for a grid of full orders
    (seasonal under ARIMA_ORDER_SEASONAL). `want_ic` (ARIMA_ORDER_IC_DEVICE
    builds): `best_ptr` takes `bs` chosen grid positions and `out_ptr` their
    `bs` float32 criteria, `pen` holding one penalty per order; otherwise
    `out_ptr` takes `len(orders) * bs` log-likelihoods. Returns the count
    written to `out_ptr`, 0 when the caller must take the per-order fits."""
    comptime if not ARIMA_ORDER_DEVICE:
        raise Error("AutoARIMA device order search was not compiled")
    else:
        var n = len(orders) * bs
        var ll = List[Float32](length=(1 if want_ic else n), fill=Float32(0.0))
        var best = List[Int32](length=bs, fill=Int32(0))
        var ic = List[Float32](length=bs, fill=Float32(0.0))
        var written = _order_search_core(y_ptr, orders, bs, nobs, maxiter, pen, want_ic, ll, best, ic)
        if written > 0:
            if want_ic:
                _write_list_f32(out_ptr, ic, 0)
                _write_list_i32(best_ptr, best, 0)
            else:
                _write_list_f32(out_ptr, ll, 0)
        return written
