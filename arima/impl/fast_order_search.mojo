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
(`bindings/hotpath_helpers.mojo::ic_running_min_f32_host_binding`).
"""
from std.math import inf, isfinite
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from core.neural_context import process_ctx
from arima.estimator import _DEVCTX_SLOT, _upload_f32, _write_list_f32, _write_list_i32
from arima.impl.batched_arima import _refuse_non_finite, canonical_nan
from arima.impl.batched_fit import ARIMA_FIT_H, arima_fit_params
from arima.impl.batched_kalman import (
    KalmanWorkspace, eval_ws_fits, fast_kalman_init_into, _launch_loop_ll_only,
)
from arima.impl.estimate_x0 import estimate_x0_x
from arima.impl.fast_eval_ws import FastEvalWS
from arima.impl.fast_arima_quality import ARIMA_FAST_ROOT_CHECK, enqueue_root_check, quality_mode
from arima.impl.fast_lbfgs_async import ASYNC_READ_EVERY, F_FX, I_ACTIVE, I_NITER, I_RETCODE
from arima.impl.fast_order_state import ARIMA_ORDER_BATCH, OrderOptimizer
from arima.impl.timeSeries.arima_helpers import batched_jones_transform
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams, pack, unpack, validate_order
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz, identical_mul_add
from tsa.impl.timeSeries.arima_helpers import prepare_data
from glm.impl.qn.qn_util import LBFGSParam


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

#: MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT (FAST+Apple via ARIMA_ORDER_BATCH,
#: default off, READY-AB). The grouped search ran its Kalman-dimension groups
#: one after another, each to its own convergence, so the search's wall time
#: was the SUM of the groups' optimizer loops, each paying its own polls and
#: drain waits. Every group's optimizer is per-series and independent, so all
#: groups now advance in ONE round loop (`_search_tasks`): each round enqueues
#: every live group's prepare / filter / finish / step, one poll covers all
#: groups, and a group stops being enqueued once its own poll shows no running
#: series. Per-group rounds, polls and stopping are the ones the sequential
#: loop had, so every order's optimum is unchanged; only the overlap differs
#: (rounds = max over groups, not sum). Merge 2026-10-05: rebuilt on main's
#: device search (device start + finite flag, chunked groups, the final
#: grouped re-evaluation into the device log-likelihood table).
#: GROUPS_CONCURRENT OUTCOME (M3 afc_ab_def, full board size, 1 run per arm,
#: 2026-10-05, verdicts batch 6): autoarima +14-16% slower. DROPPED: stays
#: off (opt-in only).
comptime ARIMA_FAST_GROUPS_CONCURRENT = (
    ARIMA_ORDER_BATCH and is_defined["MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT"]()
)

#: MOJOLEARN_ARIMA_FAST_SEARCH_REUSE (FAST+Apple via ARIMA_ORDER_BATCH; the
#: default since verdicts batch 6, rollback -D MOJOLEARN_ARIMA_FAST_SEARCH_REUSE_OFF). The search already runs every candidate order to
#: the fit's own optimizer (same estimate_x0 start, same per-series device
#: L-BFGS, same h, same scale); AutoARIMA.fit then refitted every chosen
#: order from scratch, the same work again. With this switch the search also
#: returns each order's fitted parameters, optimum x, start x0, fx, n_iter and
#: retcode (`order_search_fit`'s fit buffers, copied device to caller by
#: `_enqueue_fit_block`), and AutoARIMA.fit reuses them when the fit's maxiter
#: equals the search's (the only case where the refit is the same
#: optimization). Python side: _x_sequence_autoarima.py.
#: SEARCH_REUSE OUTCOME (M3 afc_ab_def, full board size, 1 run per arm,
#: 2026-10-05, rab16-arimareuse): autoarima synthetic 18229.7 -> 13688.1 ms,
#: taxi-hourly 16386.5 -> 9872.2 ms, forecast_rmse identical. KEEP: the
#: OFF again since 2026-10-05 (opt-in -D MOJOLEARN_ARIMA_FAST_SEARCH_REUSE): the reuse path adopts the search's fit without the
#: device-written aic/bic (ARIMA._adopt_fit got no `ics`), so FAST AutoARIMA failed on main (NameError, then TypeError on M3 rab23/rab24).
#: Computing aic/bic in Python is not allowed (no-host-routes). Lane arima-ics (2026-10-05): ARIMA._adopt_fit now gets them from the
#: plain fit's device kernel (`arima_ic_kernel` via `arima_ic_from_loglike`) over the adopted log-likelihood; still opt-in pending the M3 A/B.
#: The bundle with GROUPS/D_CONCURRENT (rab16-arimaall) was slower than this
#: alone, so only this one is on.
comptime ARIMA_FAST_SEARCH_REUSE = (
    ARIMA_ORDER_BATCH and is_defined["MOJOLEARN_ARIMA_FAST_SEARCH_REUSE"]()
)

#: MOJOLEARN_ARIMA_FAST_D_CONCURRENT (needs GROUPS_CONCURRENT; default off,
#: READY-AB). AutoARIMA searched each KPSS d group in its own native call,
#: one after another (python/mojolearn/_x_sequence_autoarima.py's d loop), so
#: the d = 0 and d = 1 grids' optimizer loops ran back to back. With this
#: switch `order_search_multi` takes every d group at once and all their
#: groups share one concurrent round loop (`_search_tasks`); each group's
#: rounds and stopping are unchanged, so every order's optimum is.
#: D_CONCURRENT OUTCOME (M3 afc_ab_def, full board size, 1 run per arm,
#: 2026-10-05, verdicts batch 6): autoarima +5-7% slower. DROPPED: stays off
#: (opt-in only).
comptime ARIMA_FAST_D_CONCURRENT = (
    ARIMA_FAST_GROUPS_CONCURRENT and is_defined["MOJOLEARN_ARIMA_FAST_D_CONCURRENT"]()
)


# TOMBSTONE: MOJOLEARN_ARIMA_FAST_CSS_SEARCH (DROPPED-quality) and MOJOLEARN_ARIMA_FAST_STEPWISE (DROPPED-slower) deleted
# 2026-10-09 on lane/owed-deletions-D1 (order_search_css, order_search_stepwise, _run_orders, the sw_* kernels); code
# recoverable at b639a2bd2. rab26: CSS -45%/-31% ms but forecast_rmse 2.465 -> 3.485 / 75.71 -> 76.06; STEPWISE +52%/+103% ms.
# Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_CSS_SEARCH.patch; record in docs/TOMBSTONES.md.


def fast_search_mode() -> Int:
    """This build's FAST search switches, for the Python glue: bits 0 and 1
    (ARIMA_FAST_CSS_SEARCH, ARIMA_FAST_STEPWISE) are always 0 since their
    deletion (2026-10-09, lane/owed-deletions-D1), bit 2 the grouped final
    fit (`order_search_multi` with fit output) for chosen orders, bits 3..5
    `fast_arima_quality.quality_mode` (CONST_BOTH, ROOT_CHECK, KPSS_D)."""
    var mode = 0
    comptime if ARIMA_FAST_D_CONCURRENT:
        mode |= 4
    # lane/apple-fast-arima-quality: bit 3 ARIMA_FAST_CONST_BOTH, bit 4
    # ARIMA_FAST_ROOT_CHECK, bit 5 ARIMA_FAST_KPSS_D (fast_arima_quality.mojo)
    mode |= quality_mode()
    return mode


def order_fit_f32_offset(orders: List[ARIMAOrder], i: Int, bs: Int) -> Int:
    """Float32 offset of order `i`'s fit block: per order, params, x, x0
    (bs * N each) then fx (bs), N = complexity()."""
    var off = 0
    for j in range(i):  # small-loop(i: a position in the AutoARIMA order grid, a few dozen entries): plan offsets, no series data
        off += bs * (3 * orders[j].complexity() + 1)
    return off


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


def _enqueue_fit_block(
    ctx: DeviceContext, mut state: OrderOptimizer, order: ARIMAOrder, bs: Int,
    mut x0: DeviceBuffer[DType.float32], fit_f32: Int, fit_i32: Int,
    f_off: Int, row: Int,
) raises:
    """ARIMA_FAST_SEARCH_REUSE: order `row`'s fit block, copied from the
    device straight into the caller's arrays: f32 at `f_off` the packed
    fitted parameters (batched_fit_x's t_x: unpack, forward Jones, pack on
    the device), the optimum x, the start x0, then fx; i32 at row * 2 * bs
    n_iter then retcode. One wait, so the temporaries outlive their copies."""
    var n = bs * order.complexity()
    var fp = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=fit_f32)
    var ip = MutPointer[Int32, MutAnyOrigin](unsafe_from_address=fit_i32)
    var raw = ARIMAParams(ctx, order, bs)
    var fitted = ARIMAParams(ctx, order, bs)
    var t_x = ctx.enqueue_create_buffer[DType.float32](max(1, n))
    unpack(ctx, raw, order, bs, state.x)
    batched_jones_transform(ctx, order, bs, False, raw, fitted)
    pack(ctx, fitted, order, bs, t_x)
    var s_tx = t_x.create_sub_buffer[DType.float32](0, max(1, n))
    var s_x = state.x.create_sub_buffer[DType.float32](0, max(1, n))
    var s_x0 = x0.create_sub_buffer[DType.float32](0, max(1, n))
    if n > 0:
        ctx.enqueue_copy(dst_ptr=fp + f_off, src_buf=s_tx)
        ctx.enqueue_copy(dst_ptr=fp + f_off + n, src_buf=s_x)
        ctx.enqueue_copy(dst_ptr=fp + f_off + 2 * n, src_buf=s_x0)
    var s_fx = state.fst.create_sub_buffer[DType.float32](F_FX * bs, bs)
    var s_ni = state.ist.create_sub_buffer[DType.int32](I_NITER * bs, bs)
    var s_rc = state.ist.create_sub_buffer[DType.int32](I_RETCODE * bs, bs)
    ctx.enqueue_copy(dst_ptr=fp + f_off + 3 * n, src_buf=s_fx)
    ctx.enqueue_copy(dst_ptr=ip + row * 2 * bs, src_buf=s_ni)
    ctx.enqueue_copy(dst_ptr=ip + row * 2 * bs + bs, src_buf=s_rc)
    ctx.synchronize()
    _ = s_tx^
    _ = s_x^
    _ = s_x0^
    _ = s_fx^
    _ = s_ni^
    _ = s_rc^
    _ = t_x^
    _ = raw^
    _ = fitted^


def _keep_x0(ctx: DeviceContext, want_fit: Bool, n: Int,
             mut x_dev: DeviceBuffer[DType.float32]) raises -> DeviceBuffer[DType.float32]:
    """A device copy of the packed start for the fit block (want_fit), else a
    one-word placeholder; the start itself moves into the optimizer."""
    if not want_fit:
        return ctx.enqueue_create_buffer[DType.float32](1)
    var keep = ctx.enqueue_create_buffer[DType.float32](max(1, n))
    ctx.enqueue_copy(dst_buf=keep, src_buf=x_dev)
    return keep^


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


struct _SearchTask(Movable):
    """One same-differencing order grid on its own series (concurrent
    search): its series, differenced series, device log-likelihood table
    (grid order), start finite flag and, with `want_fit`, its fit buffers'
    addresses."""
    var y: DeviceBuffer[DType.float32]
    var exog: DeviceBuffer[DType.float32]
    var ykf: DeviceBuffer[DType.float32]
    var ll_all: DeviceBuffer[DType.float32]
    var x0_flag: DeviceBuffer[DType.int32]
    var x0_flag_host: HostBuffer[DType.int32]
    var orders: List[ARIMAOrder]
    var bs: Int
    var nobs: Int
    var nkf: Int
    var want_fit: Bool
    var fit_f32: Int
    var fit_i32: Int
    var root_check: Bool
    """ARIMA_FAST_ROOT_CHECK: reject this task's candidates whose fitted AR
    or MA polynomial has a root of modulus < 1.01 (a search; False for the
    final exact refit of chosen orders)."""

    def __init__(out self, var y: DeviceBuffer[DType.float32], var exog: DeviceBuffer[DType.float32],
                 var ykf: DeviceBuffer[DType.float32], var ll_all: DeviceBuffer[DType.float32],
                 var x0_flag: DeviceBuffer[DType.int32], var x0_flag_host: HostBuffer[DType.int32],
                 orders: List[ARIMAOrder], bs: Int, nobs: Int, nkf: Int,
                 want_fit: Bool, fit_f32: Int, fit_i32: Int, root_check: Bool = True):
        self.y = y^
        self.exog = exog^
        self.ykf = ykf^
        self.ll_all = ll_all^
        self.x0_flag = x0_flag^
        self.x0_flag_host = x0_flag_host^
        self.orders = orders.copy()
        self.bs = bs
        self.nobs = nobs
        self.nkf = nkf
        self.want_fit = want_fit
        self.fit_f32 = fit_f32
        self.fit_i32 = fit_i32
        self.root_check = root_check


def _search_tasks(ctx: DeviceContext, mut tasks: List[_SearchTask], hp: LBFGSParam) raises:
    """ARIMA_FAST_GROUPS_CONCURRENT: `_order_search_core`'s group loop with
    every (task, r, k, chunk) group advanced in one round loop. Each group
    keeps its own filter launch, poll verdict and stop round, so every
    order's optimum is the sequential loop's. The fitted log-likelihoods land
    in each task's `ll_all` by the same grouped re-evaluation at x and
    `order_ll_kernel`; with a task's `want_fit` its fit blocks are written
    too. Several tasks (ARIMA_FAST_D_CONCURRENT: AutoARIMA's d groups) share
    the loop. Leaves the queue drained."""
    var groups = List[_OrderGroup]()
    var gtask = List[Int]()     # the task of each group
    var evals = List[FastEvalWS]()
    var states = List[OrderOptimizer]()
    var x0s = List[DeviceBuffer[DType.float32]]()
    var owner = List[Int]()     # the group of each state
    var row = List[Int]()       # the grid row (output index) of each state
    for t in range(len(tasks)):  # small-loop(tasks: AutoARIMA's d groups, at most a few): plan entries, no series data
        ref tk = tasks[t]
        for rd in range(1, 6):
            for k in range(2):
                var chunks = _order_chunks(tk.orders, rd, k, tk.bs, tk.nkf)
                for c in range(len(chunks)):  # small-loop(chunks: groups of the order grid): plan entries, no series data
                    var ids = chunks[c].copy()
                    var members = 0
                    for j in range(len(ids)):  # small-loop(ids: orders in one chunk of the grid): member-count plan arithmetic, no data
                        members += tk.bs * (tk.orders[ids[j]].complexity() + 1)
                    groups.append(_OrderGroup(ctx, rd, k, members, tk.nkf))
                    gtask.append(t)
                    var g = len(groups) - 1
                    var offset = 0
                    for j in range(len(ids)):  # small-loop(ids: orders in one chunk of the grid): one optimizer per order, device work
                        var order = tk.orders[ids[j]]
                        var okf = order.without_diff()
                        var x_dev = _initial_x_device(ctx, tk.y, tk.exog, tk.bs, tk.nobs, order, tk.x0_flag)
                        x0s.append(_keep_x0(ctx, tk.want_fit, tk.bs * order.complexity(), x_dev))
                        ref grp = groups[g]
                        var ew = FastEvalWS(ctx, tk.ykf, tk.bs, tk.nkf, okf,
                                            grp.ws, grp.y, grp.params.mu, offset)
                        var state = OrderOptimizer(ctx, ew, tk.bs, Float32(tk.nobs - 1), okf,
                                                   x_dev^, hp, ARIMA_FIT_H)
                        offset += ew.eb
                        evals.append(ew^)
                        states.append(state^)
                        owner.append(g)
                        row.append(ids[j])
        ctx.enqueue_copy(dst_ptr=tk.x0_flag_host.unsafe_ptr(), src_buf=tk.x0_flag)
    var rounds = 0
    var max_rounds = hp.max_iterations * max(1, hp.max_linesearch) + 1
    while rounds < max_rounds:
        if rounds % ASYNC_READ_EVERY == 0:
            # one wait for every live group's poll
            for j in range(len(states)):  # small-loop(states: one optimizer per order): enqueues, no series data
                if groups[owner[j]].live:
                    states[j].enqueue_poll(ctx)
            ctx.synchronize()
            for t in range(len(tasks)):  # small-loop(tasks: AutoARIMA's d groups): one flag word each
                if tasks[t].x0_flag_host.unsafe_ptr().unsafe_load(0) != Int32(0):
                    raise Error("AutoARIMA: non-finite initial parameter")
            for g in range(len(groups)):  # small-loop(groups: Kalman-dimension groups of the grid): poll verdicts, no series data
                if not groups[g].live:
                    continue
                var running = False
                for j in range(len(states)):  # small-loop(states: one optimizer per order): poll verdicts, no series data
                    if owner[j] == g:
                        running = running or states[j].running()
                groups[g].live = running
            var any_live = False
            for g in range(len(groups)):  # small-loop(groups: Kalman-dimension groups of the grid): poll verdicts
                any_live = any_live or groups[g].live
            if not any_live:
                break
        for j in range(len(states)):  # small-loop(states: one optimizer per order): enqueues, no series data
            if not groups[owner[j]].live:
                continue
            ref state = states[j]
            ref ew = evals[j]
            ew.prepare(ctx, state.order, state.h, state.cand, state.bad)
            fast_kalman_init_into(ctx, ew.t_params, state.order, ew.eb, ew.ws)
        for g in range(len(groups)):  # small-loop(groups: Kalman-dimension groups of the grid): one filter launch each
            ref grp = groups[g]
            if not grp.live:
                continue
            _launch_loop_ll_only(ctx, grp.y, grp.params, grp.ws,
                                 grp.rd, tasks[gtask[g]].nkf, grp.members, grp.k, 0, 32)
        for j in range(len(states)):  # small-loop(states: one optimizer per order): enqueues, no series data
            if not groups[owner[j]].live:
                continue
            ref state = states[j]
            ref ew = evals[j]
            ew.finish(ctx, state.h, state.scale, state.cand, state.d_grad,
                      state.d_x_pert, state.f_fxc, state.gradc, state.bad)
            state.advance(ctx)
        rounds += 1
    # The fitted log-likelihood, re-evaluated as ARIMA.fit does (never
    # recovered by rescaling fx): one more grouped evaluation AT x per group,
    # member 0 of each order the unperturbed point (main's final pass).
    for j in range(len(states)):  # small-loop(states: one optimizer per order): enqueues, no series data
        ref state = states[j]
        ref ew = evals[j]
        ew.prepare(ctx, state.order, state.h, state.x, state.bad)
        fast_kalman_init_into(ctx, ew.t_params, state.order, ew.eb, ew.ws)
    for g in range(len(groups)):  # small-loop(groups: Kalman-dimension groups of the grid): one filter launch each
        ref grp = groups[g]
        _launch_loop_ll_only(ctx, grp.y, grp.params, grp.ws,
                             grp.rd, tasks[gtask[g]].nkf, grp.members, grp.k, 0, 32)
    for j in range(len(states)):  # small-loop(states: one optimizer per order): one launch each
        ref tk = tasks[gtask[owner[j]]]
        ref ew = evals[j]
        ctx.enqueue_function[order_ll_kernel](
            tk.ll_all.unsafe_ptr(), ew.ws.loglike.unsafe_ptr(),
            ew.ws.info_init.unsafe_ptr(), ew.ws.info_loop.unsafe_ptr(),
            Int32(tk.bs), Int32(row[j]),
            grid_dim=((tk.bs + ORDER_TPB - 1) // ORDER_TPB, 1, 1), block_dim=(ORDER_TPB, 1, 1),
        )
    # ARIMA_FAST_ROOT_CHECK: each candidate's fitted roots, its row -inf
    # where a root has modulus < 1.01 (statsforecast `myarima`)
    var root_keep = List[ARIMAParams]()
    comptime if ARIMA_FAST_ROOT_CHECK:
        for j in range(len(states)):  # small-loop(states: one optimizer per order): enqueues, no series data
            ref tk = tasks[gtask[owner[j]]]
            if tk.root_check:
                ref state = states[j]
                enqueue_root_check(ctx, state.order, tk.bs, state.x, tk.ll_all, row[j], root_keep)
    ctx.synchronize()
    _ = root_keep^
    for j in range(len(states)):  # small-loop(states: one optimizer per order): device-to-caller copies
        ref tk = tasks[gtask[owner[j]]]
        if tk.want_fit:
            _enqueue_fit_block(ctx, states[j], tk.orders[row[j]], tk.bs, x0s[j],
                               tk.fit_f32, tk.fit_i32,
                               order_fit_f32_offset(tk.orders, row[j], tk.bs), row[j])
    _ = x0s^
    _ = states^
    _ = evals^
    _ = groups^


def _order_search_core(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    orders: List[ARIMAOrder], bs: Int, nobs: Int, maxiter: Int,
    pen: List[Float32], want_ic: Bool,
    mut ll_out: List[Float32], mut best_out: List[Int32], mut ic_out: List[Float32],
    want_fit: Bool = False, fit_f32: Int = 0, fit_i32: Int = 0,
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
    if want_fit and (want_ic or fit_f32 == 0 or fit_i32 == 0):
        raise Error("AutoARIMA grouped orders: fit output needs its buffers and no device criterion")
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
    comptime if ARIMA_FAST_GROUPS_CONCURRENT:
        if not want_ic:
            var tasks = List[_SearchTask]()
            tasks.append(_SearchTask(y^, exog^, ykf^, ll_all^, x0_flag^, x0_flag_host^,
                                     orders, bs, nobs, nkf, want_fit, fit_f32, fit_i32))
            _search_tasks(ctx, tasks, hp)
            ctx.enqueue_copy(dst_ptr=ll_out.unsafe_ptr(), src_buf=tasks[0].ll_all)
            ctx.synchronize()
            _ = tasks^
            _ = ctx^
            return n_orders * bs
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
                var x0s = List[DeviceBuffer[DType.float32]]()
                var offset = 0
                for j in range(len(ids)):
                    var order = orders[ids[j]]
                    var okf = order.without_diff()
                    # lane cpu3-seq (2026-10-04): every build takes the
                    # device start (was a download, a host finite walk and a
                    # re-upload per order outside ARIMA_ORDER_DEVICE)
                    var x_dev = _initial_x_device(ctx, y, exog, bs, nobs, order, x0_flag)
                    # ARIMA_FAST_SEARCH_REUSE: the start, kept for the fit block
                    x0s.append(_keep_x0(ctx, want_fit, bs * order.complexity(), x_dev))
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
                # ARIMA_FAST_ROOT_CHECK: statsforecast `myarima`'s root rule
                var root_keep = List[ARIMAParams]()
                comptime if ARIMA_FAST_ROOT_CHECK:
                    for j in range(len(states)):  # small-loop(states: one optimizer per order): enqueues, no series data
                        ref state = states[j]
                        enqueue_root_check(ctx, state.order, bs, state.x, ll_all, ids[j], root_keep)
                ctx.synchronize()
                _ = root_keep^
                if want_fit:
                    for j in range(len(states)):  # small-loop(states: one optimizer per order): device-to-caller copies
                        _enqueue_fit_block(ctx, states[j], orders[ids[j]], bs, x0s[j], fit_f32, fit_i32,
                                           order_fit_f32_offset(orders, ids[j], bs), ids[j])
                _ = x0s^
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


def order_search_fit(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    orders: List[ARIMAOrder], bs: Int, nobs: Int, maxiter: Int,
    fit_f32: Int, fit_i32: Int,
) raises -> Int:
    """ARIMA_FAST_SEARCH_REUSE: `order_search_loglike` that also writes every
    order's fit block: f32 at `fit_f32` + order_fit_f32_offset (params, x,
    x0, fx) and i32 at `fit_i32` + row * 2 * bs (n_iter, retcode). Returns
    the count written to `out_ptr`, 0 (nothing written) as
    `order_search_loglike`."""
    comptime if not ARIMA_FAST_SEARCH_REUSE:
        raise Error("AutoARIMA search fit output was not compiled")
    else:
        var n = len(orders) * bs
        var ll = List[Float32](length=n, fill=Float32(0.0))
        var best = List[Int32](length=1, fill=Int32(0))
        var ic = List[Float32](length=1, fill=Float32(0.0))
        var pen = List[Float32]()
        var written = _order_search_core(y_ptr, orders, bs, nobs, maxiter, pen, False, ll, best, ic,
                                         True, fit_f32, fit_i32)
        if written > 0:
            _write_list_f32(out_ptr, ll, 0)
        return written


def order_search_multi(
    y_ptrs: List[Int], out_ptrs: List[Int], fit_f32s: List[Int], fit_i32s: List[Int],
    task_orders: List[List[ARIMAOrder]], bss: List[Int], nobs: Int, maxiter: Int,
    want_fit: Bool, root_check: Bool = True,
) raises -> Int:
    """ARIMA_FAST_D_CONCURRENT: `order_search_loglike` (or, with `want_fit`,
    `order_search_fit`) for several same-d nonseasonal grids, one per task:
    its own series y_ptrs[t] (bss[t] x nobs), its own output out_ptrs[t] and
    fit buffers fit_f32s[t] / fit_i32s[t]; every group of every task in ONE
    concurrent round loop (`_search_tasks`). `root_check`
    (ARIMA_FAST_ROOT_CHECK builds): reject candidates by statsforecast's root
    rule; the final exact refit of chosen orders passes False. Returns the
    total number of log-likelihoods written."""
    comptime if not ARIMA_FAST_D_CONCURRENT:
        raise Error("AutoARIMA concurrent d groups were not compiled")
    else:
        var nt = len(y_ptrs)
        if nt < 1 or len(out_ptrs) != nt or len(task_orders) != nt or len(bss) != nt:
            raise Error("AutoARIMA concurrent d groups: inconsistent task lists")
        if want_fit and (len(fit_f32s) != nt or len(fit_i32s) != nt):
            raise Error("AutoARIMA concurrent d groups: fit output requested without buffers")
        if nobs < 3 or maxiter < 1:
            raise Error("AutoARIMA concurrent d groups: invalid shape or iteration count")
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var tasks = List[_SearchTask]()
        var total = 0
        for t in range(nt):  # small-loop(tasks: AutoARIMA's d groups, at most a few): uploads, no series compute
            ref orders = task_orders[t]
            var bs = bss[t]
            if bs < 1 or len(orders) < 1:
                raise Error("AutoARIMA concurrent d groups: empty task")
            var d = orders[0].d
            for i in range(len(orders)):  # small-loop(orders: the AutoARIMA order grid): plan validation, no series data
                var o = orders[i]
                validate_order(o)
                if o.d != d or o.P != 0 or o.D != 0 or o.Q != 0 or o.s != 0 or o.n_exog != 0 or o.r() > 4:
                    raise Error("AutoARIMA grouped orders require a nonseasonal same-d grid with r <= 4")
            var nkf = nobs - d
            if nkf < 3:
                raise Error("AutoARIMA concurrent d groups: too few observations after differencing")
            var yp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=y_ptrs[t])
            var y = _upload_f32(ctx, yp, bs * nobs)
            _refuse_non_finite(ctx, y, bs * nobs, "y")
            var exog = ctx.enqueue_create_buffer[DType.float32](1)
            var ykf = ctx.enqueue_create_buffer[DType.float32](bs * nkf)
            if d > 0:
                prepare_data(ctx, ykf, y, bs, nobs, d, 0, 0)
            else:
                ctx.enqueue_copy(dst_buf=ykf, src_buf=y)
            var ll_all = ctx.enqueue_create_buffer[DType.float32](len(orders) * bs)
            var x0_flag = ctx.enqueue_create_buffer[DType.int32](1)
            var x0_flag_host = ctx.enqueue_create_host_buffer[DType.int32](1)
            ctx.enqueue_memset(x0_flag, Int32(0))
            var ff = fit_f32s[t] if want_fit else 0
            var fi = fit_i32s[t] if want_fit else 0
            tasks.append(_SearchTask(y^, exog^, ykf^, ll_all^, x0_flag^, x0_flag_host^,
                                     orders, bs, nobs, nkf, want_fit, ff, fi, root_check))
            total += len(orders) * bs
        ctx.synchronize()
        for t in range(nt):  # small-loop(tasks: AutoARIMA's d groups): one flag word each
            tasks[t].x0_flag_host.unsafe_ptr().unsafe_store(0, Int32(0))
        _search_tasks(ctx, tasks, arima_fit_params(maxiter))
        for t in range(nt):  # small-loop(tasks: AutoARIMA's d groups): one device-to-caller copy each
            ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=out_ptrs[t]),
                             src_buf=tasks[t].ll_all)
        ctx.synchronize()
        _ = tasks^
        _ = ctx^
        return total


# TOMBSTONE: MOJOLEARN_ARIMA_FAST_CSS_SEARCH (DROPPED-quality) and MOJOLEARN_ARIMA_FAST_STEPWISE (DROPPED-slower) deleted
# 2026-10-09 on lane/owed-deletions-D1 (order_search_css, order_search_stepwise, _run_orders, the sw_* kernels); code
# recoverable at b639a2bd2. rab26: CSS -45%/-31% ms but forecast_rmse 2.465 -> 3.485 / 75.71 -> 76.06; STEPWISE +52%/+103% ms.
# Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_CSS_SEARCH.patch; record in docs/TOMBSTONES.md.
