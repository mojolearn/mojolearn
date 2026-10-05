# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-tsa (2026-10-02): the FAST ARIMA fit's evaluation
workspace, allocated ONCE per solve (`-D MOJOLEARN_ARIMA_FAST_EVAL_WS=1`).

THE CAUSE. `eval_batch_device`'s stacked arm (`arima/impl/batched_fit.mojo`,
the evaluation `batched_min_lbfgs` makes at every candidate point of the
device L-BFGS, hundreds of times per fit) allocates at every call: `y_ext`
((N + 1) x batch_size series, the input replicated N + 1 times through
N + 1 device copies), `x_ext` (N + 1 copies of the candidates), `p_ext`
and `loglike_ws_packed`'s `t_params` (7 buffers each) and a
`KalmanWorkspace` (26 buffers), then issues N `perturb_kernel` launches, N
`grad_kernel` launches, the card's `P0` / `alpha0` copies, and WAITS once
so the workspace can be released. About 45 buffer creations, 2 (N + 1)
copies and 2N + 5 launches per evaluation on a batch whose Kalman pass is
320 threads of 2,000 steps (the board's 64 ARMA(1,1) series): the
evaluation is allocator and launch time, not filter time.

THE CHANGE. `FastEvalWS` holds every buffer for the solve; `eval` issues
ONE stacking kernel (`ew_stack_kernel`: member m of series b is the
candidate, member m >= 1 perturbed at parameter m - 1 with
`perturb_kernel`'s statement), `unpack`, the Jones transform,
`fast_kalman_into` (the filter's three launches into the held workspace),
then main's `arima_mark_infeasible_kernel`, ONE gradient kernel
(`grad_kernel`'s statement over every (parameter, series)), the
`d_x_pert = d_x` copy and `arima_eval_finish_kernel`, exactly the stacked
arm's tail. Nothing is read back and nothing waits: the results stay in the
optimizer's device buffers as the stacked arm leaves them. The per-element
arithmetic is `perturb_kernel`'s, `grad_kernel`'s and the filter's, so the
log-likelihood and gradient bits are the stacked arm's; the batch
composition and launch geometry are unchanged.

FAST on Apple only; nothing here is launched or instantiated in any other
build (`eval`'s body exists only under KALMAN_FAST_EVAL_WS)."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import inf, isinf
from std.sys.compile import is_defined

from arima.impl.batched_kalman import (
    KALMAN_FAST_EVAL_WS,
    KalmanWorkspace,
    fast_kalman_into,
)
from arima.impl.lbfgs_device import (
    LBFGS_TPB,
    arima_eval_finish_kernel,
    arima_mark_infeasible_kernel,
)
from arima.impl.timeSeries.arima_helpers import batched_jones_transform
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams, unpack, validate_order
from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_log, identical_mul_add,
)

comptime EW_TPB = 128
# FAST + Apple default after M3 gap26-arima-tail-{synthetic,taxi-hourly}:
# 14498.225 -> 13569.230 ms and 24231.028 -> 22705.348 ms (one run/arm).
# Forecast RMSE/digests unchanged; gap26-arima-quality-fixed checks 44
# selected-order, parameter, likelihood and forecast arrays unchanged.
# MOJOLEARN_ARIMA_FUSED_EVAL_TAIL_OFF restores the separate launches.
# See docs/apple-fast/EXPERIMENTS.md; IDENTICAL/other vendors unchanged.
# IDENTICAL on every vendor since lane/fam-timeseries (2026-10-04), with the
# held workspace: one tail launch per evaluation instead of four, the same
# finite differences and FTZ sites (`ew_finish_kernel`).
# -D MOJOLEARN_IDN_ARIMA_FUSED_TAIL_OFF=1 restores the separate launches
# under IDENTICAL.
comptime ARIMA_FUSED_EVAL_TAIL = (
    KALMAN_FAST_EVAL_WS and not is_defined["MOJOLEARN_ARIMA_FUSED_EVAL_TAIL_OFF"]()
    and not (
        GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        and is_defined["MOJOLEARN_IDN_ARIMA_FUSED_TAIL_OFF"]()
    )
)


def ew_finish_kernel(
    f_out: MutPointer[Float32, MutAnyOrigin],
    g_out: MutPointer[Float32, MutAnyOrigin],
    g_raw: MutPointer[Float32, MutAnyOrigin],
    x_pert: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    ll: MutPointer[Float32, MutAnyOrigin],
    info0: MutPointer[Int32, MutAnyOrigin],
    info1: MutPointer[Int32, MutAnyOrigin],
    bad: MutPointer[Int32, MutAnyOrigin],
    batch_in: Int32, n_in: Int32, h: Float32, scale: Float32,
):
    """One thread per independent series, same finite differences and FTZ
    sites as mark_infeasible + ew_grad + eval_finish. Also preserves both
    scratch outputs (including raw gradients on infeasible series)."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var bs = Int(batch_in)
    if b >= bs:
        return
    var n = Int(n_in)
    var invalid = False
    for member in range(n + 1):
        var j = member * bs + b
        var v = ll[j]
        if info0[j] != 0 or info1[j] != 0 or (isinf(v) and v < Float32(0.0)):
            invalid = True
    bad[b] = Int32(1) if invalid else Int32(0)
    if invalid:
        f_out[b] = inf[DType.float32]()
    else:
        f_out[b] = ftz(identical_div(ftz(-ll[b]), scale))
    for i in range(n):
        var diff = ftz(ftz(ll[(i + 1) * bs + b]) - ftz(ll[b]))
        var raw = ftz(diff / h)
        g_raw[b * n + i] = raw
        x_pert[b * n + i] = x[b * n + i]
        if invalid:
            g_out[b * n + i] = Float32(0.0)
        else:
            g_out[b * n + i] = ftz(identical_div(ftz(-raw), scale))



def ew_stack_kernel(
    x_ext: MutPointer[Float32, MutAnyOrigin],
    d_x: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    N_in: Int32,
    h: Float32,
):
    """One thread per (member m, series b): block m of `x_ext` is `d_x`;
    for m >= 1 parameter m - 1 is `ftz(ftz(x) + h)` (`perturb_kernel`,
    `batched_arima.mojo`), the others the copy."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var B = Int(batch_size_in)
    var N = Int(N_in)
    if t >= (N + 1) * B:
        return
    var m = t // B
    var b = t - m * B
    var src = N * b
    var dst = m * (N * B) + N * b
    for i in range(N):
        var v = d_x.unsafe_load(src + i)
        if i == m - 1:
            v = ftz(ftz(v) + h)
        x_ext.unsafe_store(dst + i, v)


def ew_grad_kernel(
    d_grad: MutPointer[Float32, MutAnyOrigin],
    d_ll: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    N_in: Int32,
    h: Float32,
):
    """One thread per (parameter i, series b): `grad_kernel`'s
    `(ll_pert - ll_base) / h` with member i + 1's log-likelihood."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var B = Int(batch_size_in)
    var N = Int(N_in)
    if t >= N * B:
        return
    var i = t // B
    var b = t - i * B
    var diff = ftz(ftz(d_ll.unsafe_load((i + 1) * B + b)) - ftz(d_ll.unsafe_load(b)))
    d_grad.unsafe_store(N * b + i, ftz(diff / h))


def ew_obs_intercept_kernel(
    d_exog: MutPointer[Float32, MutAnyOrigin],
    d_beta: MutPointer[Float32, MutAnyOrigin],
    d_obs: MutPointer[Float32, MutAnyOrigin],
    nb_in: Int32,
    eb_in: Int32,
    n_in: Int32,
    n_exog_in: Int32,
):
    """lane/fam2-timeseries: the stacked evaluation's observation intercept,
    one thread per (member, step) cell. Member `bid` of the `eb` stacked
    members is series `bid % nb`, whose regressors it reads (the series'
    exog is NOT replicated); its beta is its own (perturbed) one.
    `obs_intercept_kernel`'s statement (`batched_kalman.mojo`): the ascending
    fma from 0 over the regressors, so each cell's bits are the sequential
    pass's."""
    var t_all = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    var eb = Int(eb_in)
    if t_all >= eb * n:
        return
    var nb = Int(nb_in)
    var n_exog = Int(n_exog_in)
    var bid = t_all // n
    var t = t_all - bid * n
    var b = bid % nb
    var xb = b * n_exog * n
    var bb = bid * n_exog
    var acc = Float32(0.0)
    for i in range(n_exog):
        var xv = ftz(d_exog.unsafe_load(xb + i * n + t))
        var bv = ftz(d_beta.unsafe_load(bb + i))
        acc = ftz(identical_mul_add(xv, bv, acc))
    d_obs.unsafe_store(bid * n + t, acc)


#: lane/apple-fast-arima-sf (2026-10-04): the conditional-sum-of-squares
#: objective for MOJOLEARN_ARIMA_FAST_CSS_SEARCH (`fast_order_search.mojo`).
#: The widest AR / MA order the CSS recursion holds in registers (the grouped
#: search's nonseasonal grids have p, q <= 4).
comptime CSS_PQ_MAX = 8
comptime CSS_LOG_2PI = Float32(1.8378770664093453)


def css_ll_kernel(
    ll: MutPointer[Float32, MutAnyOrigin],
    info0: MutPointer[Int32, MutAnyOrigin],
    info1: MutPointer[Int32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    ar: MutPointer[Float32, MutAnyOrigin],
    ma: MutPointer[Float32, MutAnyOrigin],
    mask: MutPointer[Int32, MutAnyOrigin],
    nb_in: Int32, eb_in: Int32, n_in: Int32, p_in: Int32, q_in: Int32, k_in: Int32,
    has_mask_in: Int32,
):
    """ARIMA_FAST_CSS_SEARCH: one thread per stacked member (member m of
    series b at m * nb + b, `ew_stack_kernel`'s layout). The conditional sum
    of squares of R's `arima(method = "CSS")` (`ARIMA_CSS`, the objective
    statsforecast's `auto_arima_f` scores candidates with when
    `approximation` is on): on the differenced series w (`y`, nb x n),
    e_t = w_t - c - sum_i phi_i w_{t-i} - sum_j theta_j e_{t-j} for t >= p,
    e_t = 0 before (ncond = p after differencing), ssq = sum e_t^2 over the
    nu = n - p summed steps, sigma2 = ssq / nu. c is this ARIMA's intercept
    (the Kalman filter's state constant), the reparametrization of R's mean
    form, so the minimizing ARMA coefficients are the same.

    Written as a log-likelihood on the criterion's scale:
    ll = -0.5 n (log(2 pi) + 1 + log(ssq / nu)), n = the differenced length
    (statsforecast's `nstar`), so -2 ll + penalty is statsforecast's CSS
    criterion `offset + nstar log(sigma2) + penalty` up to a per-series
    constant (the offset), which no order choice reads. A non-finite or
    non-positive ssq is refused through `info1` (the evaluation's
    infeasible mark). `mask` (with `has_mask_in`): series whose word is 0
    are skipped (ll 0, the optimizer has retired them)."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var eb = Int(eb_in)
    if j >= eb:
        return
    var nb = Int(nb_in)
    var n = Int(n_in)
    var p = Int(p_in)
    var q = Int(q_in)
    var b = j % nb
    info0.unsafe_store(j, Int32(0))
    info1.unsafe_store(j, Int32(0))
    if has_mask_in != 0 and mask.unsafe_load(b) == Int32(0):
        ll.unsafe_store(j, Float32(0.0))
        return
    var c = mu.unsafe_load(j) if k_in != 0 else Float32(0.0)
    var phi = InlineArray[Float32, CSS_PQ_MAX](fill=Float32(0.0))
    var theta = InlineArray[Float32, CSS_PQ_MAX](fill=Float32(0.0))
    var e = InlineArray[Float32, CSS_PQ_MAX](fill=Float32(0.0))
    for i in range(p):
        phi[i] = ar.unsafe_load(p * j + i)
    for i in range(q):
        theta[i] = ma.unsafe_load(q * j + i)
    var yb = b * n
    var ssq = Float32(0.0)
    for t in range(p, n):
        var v = y.unsafe_load(yb + t) - c
        for i in range(p):
            v -= phi[i] * y.unsafe_load(yb + t - 1 - i)
        for i in range(q):
            v -= theta[i] * e[i]
        # e[i] holds e_{t-1-i}: shift, newest first
        var i2 = q - 1
        while i2 > 0:
            e[i2] = e[i2 - 1]
            i2 -= 1
        e[0] = v
        ssq += v * v
    var nu = n - p
    if nu < 1 or not (ssq > Float32(0.0)) or isinf(ssq):
        info1.unsafe_store(j, Int32(1))
        ll.unsafe_store(j, -inf[DType.float32]())
        return
    var s2 = ssq / Float32(nu)
    ll.unsafe_store(j, Float32(-0.5) * Float32(n) * (CSS_LOG_2PI + Float32(1.0) + identical_log(s2)))


struct FastEvalWS(Movable):
    """Every buffer of one stacked evaluation, sized for `nb` series. The
    optimizer holds one in an `Optional` that is `None` when the switch is
    off (`training/byte_lm_logits.mojo`'s `Optional[ByteLogitsScratch]`
    shape), so nothing here is allocated in any other build."""

    var nb: Int
    var n_obs: Int
    var N: Int
    var eb: Int
    var n_exog: Int
    """0, or the regressors per series once `attach_exog` ran
    (lane/fam2-timeseries, ARIMA_EVAL_WS_EXOG)."""
    var exog: DeviceBuffer[DType.float32]
    """The `nb` series' differenced regressors (a view of the caller's
    buffer, `[b * n_exog * n_obs + i * n_obs + t]`); one float when
    `n_exog == 0`."""
    var y_ext: DeviceBuffer[DType.float32]
    var x_ext: DeviceBuffer[DType.float32]
    var p_ext: ARIMAParams
    var t_params: ARIMAParams
    var ws: KalmanWorkspace
    var css: Bool
    """ARIMA_FAST_CSS_SEARCH: the evaluation is the conditional sum of
    squares (`css_ll_kernel`) on `y_ext` (the `nb` differenced series, NOT
    replicated) instead of the Kalman filter; set only by the CSS
    constructor."""
    var css_mask: DeviceBuffer[DType.int32]
    """The CSS evaluation's per-series skip words (`set_css_mask`); one
    word when no mask is set."""
    var has_css_mask: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        mut d_y: DeviceBuffer[DType.float32],
        nb: Int,
        n_obs: Int,
        order: ARIMAOrder,
    ) raises:
        """`d_y` holds the `nb` series (contiguous, `nb * n_obs`); it is
        replicated N + 1 times here, once per solve."""
        var N = order.complexity()
        var M1 = N + 1
        var eb = M1 * nb
        var nb_y = nb * n_obs
        var y_ext = ctx.enqueue_create_buffer[DType.float32](max(1, eb * n_obs))
        for m in range(M1):
            ctx.enqueue_copy(
                dst_buf=y_ext.create_sub_buffer[DType.float32](m * nb_y, nb_y),
                src_buf=d_y.create_sub_buffer[DType.float32](0, nb_y),
            )
        var x_ext = ctx.enqueue_create_buffer[DType.float32](max(1, eb * N))
        var p_ext = ARIMAParams(ctx, order, eb)
        var t_params = ARIMAParams(ctx, order, eb)
        var ws = KalmanWorkspace(ctx, order, eb, n_obs, 0)
        ctx.synchronize()
        self.nb = nb
        self.n_obs = n_obs
        self.N = N
        self.eb = eb
        self.n_exog = 0
        self.exog = ctx.enqueue_create_buffer[DType.float32](1)
        self.y_ext = y_ext^
        self.x_ext = x_ext^
        self.p_ext = p_ext^
        self.t_params = t_params^
        self.ws = ws^
        self.css = False
        self.css_mask = ctx.enqueue_create_buffer[DType.int32](1)
        self.has_css_mask = False

    def __init__(
        out self,
        ctx: DeviceContext,
        mut d_y: DeviceBuffer[DType.float32],
        nb: Int,
        n_obs: Int,
        order: ARIMAOrder,
        parent_ws: KalmanWorkspace,
        parent_y: DeviceBuffer[DType.float32],
        parent_mu: DeviceBuffer[DType.float32],
        offset: Int,
    ) raises:
        """`d_y` holds the `nb` series (contiguous, `nb * n_obs`); it is
        replicated N + 1 times here, once per solve."""
        var N = order.complexity()
        var M1 = N + 1
        var eb = M1 * nb
        var nb_y = nb * n_obs
        var y_ext = parent_y.create_sub_buffer[DType.float32](offset * n_obs, eb * n_obs)
        for m in range(M1):
            ctx.enqueue_copy(
                dst_buf=y_ext.create_sub_buffer[DType.float32](m * nb_y, nb_y),
                src_buf=d_y.create_sub_buffer[DType.float32](0, nb_y),
            )
        var x_ext = ctx.enqueue_create_buffer[DType.float32](max(1, eb * N))
        var p_ext = ARIMAParams(ctx, order, eb)
        var t_params = ARIMAParams(ctx, order, eb)
        var ws = KalmanWorkspace(parent_ws, order, offset, eb, n_obs)
        t_params.mu = parent_mu.create_sub_buffer[DType.float32](offset, eb)
        ctx.synchronize()
        self.nb = nb
        self.n_obs = n_obs
        self.N = N
        self.eb = eb
        self.n_exog = 0
        self.exog = ctx.enqueue_create_buffer[DType.float32](1)
        self.y_ext = y_ext^
        self.x_ext = x_ext^
        self.p_ext = p_ext^
        self.t_params = t_params^
        self.ws = ws^
        self.css = False
        self.css_mask = ctx.enqueue_create_buffer[DType.int32](1)
        self.has_css_mask = False

    def __init__(
        out self,
        ctx: DeviceContext,
        nb: Int,
        n_obs: Int,
        order: ARIMAOrder,
        mut d_ykf: DeviceBuffer[DType.float32],
    ) raises:
        """ARIMA_FAST_CSS_SEARCH: the CSS evaluation's buffers. `d_ykf` holds
        the `nb` DIFFERENCED series (`nb * n_obs`, `n_obs` the differenced
        length); `order` is the order without differencing. The CSS kernel
        reads each series once per member, so nothing is replicated and the
        Kalman workspace is a one-step placeholder (only `loglike` and the
        two refusal words are written)."""
        var N = order.complexity()
        var eb = (N + 1) * nb
        var x_ext = ctx.enqueue_create_buffer[DType.float32](max(1, eb * N))
        var p_ext = ARIMAParams(ctx, order, eb)
        var t_params = ARIMAParams(ctx, order, eb)
        var ws = KalmanWorkspace(ctx, order, eb, 1, 0)
        ctx.synchronize()
        self.nb = nb
        self.n_obs = n_obs
        self.N = N
        self.eb = eb
        self.n_exog = 0
        self.exog = ctx.enqueue_create_buffer[DType.float32](1)
        self.y_ext = d_ykf.create_sub_buffer[DType.float32](0, max(1, nb * n_obs))
        self.x_ext = x_ext^
        self.p_ext = p_ext^
        self.t_params = t_params^
        self.ws = ws^
        self.css = True
        self.css_mask = ctx.enqueue_create_buffer[DType.int32](1)
        self.has_css_mask = False

    def set_css_mask(mut self, mask: DeviceBuffer[DType.int32], offset: Int) raises:
        """ARIMA_FAST_STEPWISE over CSS: series whose word in `mask[offset :
        offset + nb]` is 0 are skipped by the CSS kernel."""
        self.css_mask = mask.create_sub_buffer[DType.int32](offset, self.nb)
        self.has_css_mask = True

    def css_into(mut self, ctx: DeviceContext, order: ARIMAOrder) raises:
        """The CSS objective of every stacked member into `ws.loglike` (and
        0 / the refusal into `ws.info_init` / `ws.info_loop`), from the
        transformed parameters `prepare` wrote. One launch, no wait."""
        var g = (self.eb + EW_TPB - 1) // EW_TPB
        ctx.enqueue_function[css_ll_kernel](
            self.ws.loglike.unsafe_ptr(), self.ws.info_init.unsafe_ptr(), self.ws.info_loop.unsafe_ptr(),
            self.y_ext.unsafe_ptr(), self.t_params.mu.unsafe_ptr(), self.t_params.ar.unsafe_ptr(),
            self.t_params.ma.unsafe_ptr(), self.css_mask.unsafe_ptr(),
            Int32(self.nb), Int32(self.eb), Int32(self.n_obs), Int32(order.p), Int32(order.q),
            Int32(order.k), Int32(1) if self.has_css_mask else Int32(0),
            grid_dim=(g, 1, 1), block_dim=(EW_TPB, 1, 1),
        )

    def loglike_at(mut self, ctx: DeviceContext, order: ARIMAOrder, h: Float32,
                   mut d_x: DeviceBuffer[DType.float32],
                   mut d_bad: DeviceBuffer[DType.int32]) raises:
        """Every member's objective at `d_x` (member 0 the unperturbed point)
        into `ws.loglike`: CSS or the Kalman filter, as `eval` takes it."""
        self.prepare(ctx, order, h, d_x, d_bad)
        if self.css:
            self.css_into(ctx, order)
        else:
            fast_kalman_into(ctx, self.y_ext, self.t_params, order, self.eb, self.n_obs, self.ws, 32,
                             1 if self.n_exog > 0 else 0)

    def attach_exog(mut self, d_exog: DeviceBuffer[DType.float32], n_exog: Int) raises:
        """The fit's differenced regressors (`nb * n_exog * n_obs` floats),
        for an order with `n_exog != 0`. The workspace was built from that
        order, so `ws.obs` is `eb * n_obs` long."""
        if n_exog > 0:
            self.exog = d_exog.create_sub_buffer[DType.float32](0, self.nb * n_exog * self.n_obs)
            self.n_exog = n_exog

    def eval(
        mut self,
        ctx: DeviceContext,
        order: ARIMAOrder,
        h: Float32,
        scale: Float32,
        mut d_x: DeviceBuffer[DType.float32],
        mut d_grad: DeviceBuffer[DType.float32],
        mut d_x_pert: DeviceBuffer[DType.float32],
        mut d_f: DeviceBuffer[DType.float32],
        mut d_g: DeviceBuffer[DType.float32],
        mut d_bad: DeviceBuffer[DType.int32],
    ) raises:
        """`eval_batch_device`'s stacked arm at the candidates in `d_x`, on
        the held buffers: `d_f` / `d_g` / `d_bad` are written as that arm
        writes them, `d_x_pert` ends equal to `d_x`, nothing is read back
        and nothing waits."""
        comptime if not KALMAN_FAST_EVAL_WS:
            raise Error("FastEvalWS.eval: not compiled in this build (MOJOLEARN_ARIMA_FAST_EVAL_WS)")
        else:
            self.loglike_at(ctx, order, h, d_x, d_bad)
            self.finish(ctx, h, scale, d_x, d_grad, d_x_pert, d_f, d_g, d_bad)

    def prepare(mut self, ctx: DeviceContext, order: ARIMAOrder, h: Float32,
                mut d_x: DeviceBuffer[DType.float32],
                mut d_bad: DeviceBuffer[DType.int32]) raises:
        var nb = self.nb
        var N = self.N
        var eb = self.eb
        var nb_x = nb * N
        var grid = (nb + LBFGS_TPB - 1) // LBFGS_TPB
        comptime if not ARIMA_FUSED_EVAL_TAIL:
            ctx.enqueue_memset(d_bad, Int32(0))
        var g1 = (eb + EW_TPB - 1) // EW_TPB
        ctx.enqueue_function[ew_stack_kernel](
            self.x_ext.unsafe_ptr(), d_x.unsafe_ptr(), Int32(nb), Int32(N), h,
            grid_dim=(g1, 1, 1), block_dim=(EW_TPB, 1, 1),
        )
        unpack(ctx, self.p_ext, order, eb, self.x_ext)
        validate_order(order)
        batched_jones_transform(ctx, order, eb, False, self.p_ext, self.t_params)
        if self.n_exog > 0:
            # every member's x_t beta, before the filter reads `ws.obs`
            var cells = eb * self.n_obs
            ctx.enqueue_function[ew_obs_intercept_kernel](
                self.exog.unsafe_ptr(), self.t_params.beta.unsafe_ptr(), self.ws.obs.unsafe_ptr(),
                Int32(nb), Int32(eb), Int32(self.n_obs), Int32(self.n_exog),
                grid_dim=((cells + EW_TPB - 1) // EW_TPB, 1, 1), block_dim=(EW_TPB, 1, 1),
            )

    def finish(mut self, ctx: DeviceContext, h: Float32, scale: Float32,
               mut d_x: DeviceBuffer[DType.float32],
               mut d_grad: DeviceBuffer[DType.float32],
               mut d_x_pert: DeviceBuffer[DType.float32],
               mut d_f: DeviceBuffer[DType.float32],
               mut d_g: DeviceBuffer[DType.float32],
               mut d_bad: DeviceBuffer[DType.int32]) raises:
        var nb = self.nb
        var N = self.N
        var nb_x = nb * N
        var grid = (nb + LBFGS_TPB - 1) // LBFGS_TPB
        # Shared by single-order and grouped-order fits: both arms retain
        # current main's accepted fused tail, independently of ORDER_BATCH.
        comptime if ARIMA_FUSED_EVAL_TAIL:
            ctx.enqueue_function[ew_finish_kernel](
                d_f.unsafe_ptr(), d_g.unsafe_ptr(), d_grad.unsafe_ptr(),
                d_x_pert.unsafe_ptr(), d_x.unsafe_ptr(), self.ws.loglike.unsafe_ptr(),
                self.ws.info_init.unsafe_ptr(), self.ws.info_loop.unsafe_ptr(),
                d_bad.unsafe_ptr(), Int32(nb), Int32(N), h, scale,
                grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
            )
            return
        ctx.enqueue_function[arima_mark_infeasible_kernel](
            d_bad.unsafe_ptr(), self.ws.loglike.unsafe_ptr(), self.ws.info_init.unsafe_ptr(),
            self.ws.info_loop.unsafe_ptr(), Int32(nb), Int32(N + 1),
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )
        var g2 = (nb_x + EW_TPB - 1) // EW_TPB
        ctx.enqueue_function[ew_grad_kernel](
            d_grad.unsafe_ptr(), self.ws.loglike.unsafe_ptr(), Int32(nb), Int32(N), h,
            grid_dim=(g2, 1, 1), block_dim=(EW_TPB, 1, 1),
        )
        # the caller's scratch ends equal to d_x, as the sequential form leaves it
        ctx.enqueue_copy(
            dst_buf=d_x_pert.create_sub_buffer[DType.float32](0, nb_x),
            src_buf=d_x.create_sub_buffer[DType.float32](0, nb_x),
        )
        ctx.enqueue_function[arima_eval_finish_kernel](
            d_f.unsafe_ptr(), d_g.unsafe_ptr(), self.ws.loglike.unsafe_ptr(),
            d_grad.unsafe_ptr(), d_bad.unsafe_ptr(), Int32(nb), Int32(N), scale,
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )
