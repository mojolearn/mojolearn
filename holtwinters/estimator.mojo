"""Host-list surface for Holt-Winters: what `bindings/` will call.

**NOT YET WIRED** into `bindings/` or `python/` (not this lane's
directories; COMMON_BRIEF). The README's HAND-OFF names the Python surface;
this is the entry it should reach, shaped like `kde/estimator.mojo`.

`holtwinters_fit_host` takes the `batch_size x n` series-major data as a
host list (each series contiguous, cuML's numpy convention `(ts_num, n)`),
validates exactly as `holtwinters.pyx` does (`holtwinters_validate_params`,
by name) plus DEVIATION 664's finiteness / positivity rules, uploads, runs
`ML::HoltWinters::fit` (`holtwinters/ported/holtwinters/holtwinters.mojo`)
with the given trace, and returns every output as host lists.
`holtwinters_forecast_host` re-uploads the fitted components and runs
`ML::HoltWinters::forecast` (the one-shot form; a bindings layer that
keeps the `DeviceBuffer`s can call the ported entries directly).
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from holtwinters.ported.holtwinters.holtwinters import buffer_size, fit, forecast
from holtwinters.ported.holtwinters.internal.hw_utils import HW_OPTIM_TPB
from holtwinters.ported.holtwinters.runner import (
    HW_DEFAULT_EPS,
    HW_DEFAULT_TRACE_ITERS,
    holtwinters_validate_data,
    holtwinters_validate_params,
)
from holtwinters.ported.tsa.holtwinters_params import seasonal_from_name


struct HWFit(Movable):
    """Everything `fit` produced, on the host. `level`/`trend`/`season` are
    time-major (`components_len = (n - frequency) * batch_size`; series `s`
    at step `i` is `[s + i * batch_size]`); `iter_trace` is
    `[(iter * 3 + k) * batch_size + s]` for `iter < trace_iters`."""

    var n: Int
    var batch_size: Int
    var frequency: Int
    var seasonal: Int
    var level: List[Float32]
    var trend: List[Float32]
    var season: List[Float32]
    var sse: List[Float32]
    var alpha: List[Float32]
    var beta: List[Float32]
    var gamma: List[Float32]
    var criterion: List[Int32]
    var niter: List[Int32]
    #: DEVIATION 699's packed branch record, one per series.
    var decisions: List[Int32]
    var iter_trace: List[Float32]
    var trace_iters: Int

    def __init__(out self, n: Int, batch_size: Int, frequency: Int, seasonal: Int, trace_iters: Int):
        self.n = n
        self.batch_size = batch_size
        self.frequency = frequency
        self.seasonal = seasonal
        self.level = List[Float32]()
        self.trend = List[Float32]()
        self.season = List[Float32]()
        self.sse = List[Float32]()
        self.alpha = List[Float32]()
        self.beta = List[Float32]()
        self.gamma = List[Float32]()
        self.criterion = List[Int32]()
        self.niter = List[Int32]()
        self.decisions = List[Int32]()
        self.iter_trace = List[Float32]()
        self.trace_iters = trace_iters


def upload_f32(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n if n > 0 else 1)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def download_f32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view^
    var out = List[Float32]()
    out.reserve(n)
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def download_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int32]:
    var host = ctx.enqueue_create_host_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        var view = buf.create_sub_buffer[DType.int32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view^
    var out = List[Int32]()
    out.reserve(n)
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def holtwinters_fit_host_traced(
    ctx: DeviceContext,
    data: List[Float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    start_periods: Int,
    seasonal: String,
    eps: Float32,
    mut trace: IdentityTrace,
    trace_iters: Int = HW_DEFAULT_TRACE_ITERS,
    tpb_decomp: Int = -1,
    tpb_optim: Int = HW_OPTIM_TPB,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = Float32(0.0),
) raises -> HWFit:
    """`ExponentialSmoothing(endog, seasonal, seasonal_periods=frequency,
    start_periods, ts_num=batch_size, eps).fit()` with an explicit trace
    and the scheduling / padding knobs the gates vary."""
    var st = seasonal_from_name(seasonal)
    holtwinters_validate_params(n, batch_size, frequency, start_periods, eps)
    holtwinters_validate_data(data, n, batch_size, st)
    var sizes = buffer_size(n, batch_size, frequency)
    var ddata = upload_f32(ctx, data)
    var level_d = ctx.enqueue_create_buffer[DType.float32](sizes.components_len + scratch_pad)
    var trend_d = ctx.enqueue_create_buffer[DType.float32](sizes.components_len + scratch_pad)
    var season_d = ctx.enqueue_create_buffer[DType.float32](sizes.components_len + scratch_pad)
    var error_d = ctx.enqueue_create_buffer[DType.float32](batch_size + scratch_pad)
    var alpha_d = ctx.enqueue_create_buffer[DType.float32](batch_size + scratch_pad)
    var beta_d = ctx.enqueue_create_buffer[DType.float32](batch_size + scratch_pad)
    var gamma_d = ctx.enqueue_create_buffer[DType.float32](batch_size + scratch_pad)
    var criterion_d = ctx.enqueue_create_buffer[DType.int32](batch_size + scratch_pad)
    var niter_d = ctx.enqueue_create_buffer[DType.int32](batch_size + scratch_pad)
    var decisions_d = ctx.enqueue_create_buffer[DType.int32](batch_size + scratch_pad)
    var n_trace = trace_iters * 3 * batch_size
    var iter_trace_d = ctx.enqueue_create_buffer[DType.float32]((n_trace if n_trace > 0 else 1) + scratch_pad)
    level_d.enqueue_fill(scratch_poison)
    trend_d.enqueue_fill(scratch_poison)
    season_d.enqueue_fill(scratch_poison)
    error_d.enqueue_fill(scratch_poison)
    iter_trace_d.enqueue_fill(scratch_poison)
    criterion_d.enqueue_fill(Int32(-7))
    niter_d.enqueue_fill(Int32(-7))
    decisions_d.enqueue_fill(Int32(-7))
    ctx.synchronize()
    trace.header(
        "holtwinters: n=" + String(n) + " batch_size=" + String(batch_size) + " frequency="
        + String(frequency) + " start_periods=" + String(start_periods) + " seasonal=" + seasonal
        + " eps=" + String(eps) + " trace_iters=" + String(trace_iters)
    )
    fit(
        ctx, n, batch_size, frequency, start_periods, st, eps, ddata,
        level_d, trend_d, season_d, error_d, alpha_d, beta_d, gamma_d,
        criterion_d, niter_d, decisions_d, iter_trace_d, trace, trace_iters,
        tpb_decomp, tpb_optim, scratch_pad, scratch_poison,
    )
    var out = HWFit(n, batch_size, frequency, st, trace_iters)
    out.level = download_f32(ctx, level_d, sizes.components_len)
    out.trend = download_f32(ctx, trend_d, sizes.components_len)
    out.season = download_f32(ctx, season_d, sizes.components_len)
    out.sse = download_f32(ctx, error_d, batch_size)
    out.alpha = download_f32(ctx, alpha_d, batch_size)
    out.beta = download_f32(ctx, beta_d, batch_size)
    out.gamma = download_f32(ctx, gamma_d, batch_size)
    out.criterion = download_i32(ctx, criterion_d, batch_size)
    out.niter = download_i32(ctx, niter_d, batch_size)
    out.decisions = download_i32(ctx, decisions_d, batch_size)
    out.iter_trace = download_f32(ctx, iter_trace_d, n_trace)
    _ = ddata^
    _ = level_d^
    _ = trend_d^
    _ = season_d^
    _ = error_d^
    _ = alpha_d^
    _ = beta_d^
    _ = gamma_d^
    _ = criterion_d^
    _ = niter_d^
    _ = decisions_d^
    _ = iter_trace_d^
    return out^


def holtwinters_fit_host(
    data: List[Float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    start_periods: Int = 2,
    seasonal: String = "additive",
    eps: Float32 = HW_DEFAULT_EPS,
) raises -> HWFit:
    """The bindings entry: the environment's trace (`MOJOLEARN_IDENTITY_TRACE`),
    their block widths, no padding."""
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    return holtwinters_fit_host_traced(
        ctx, data, n, batch_size, frequency, start_periods, seasonal, eps, trace
    )


def holtwinters_forecast_host_traced(
    ctx: DeviceContext, fitted: HWFit, h: Int, mut trace: IdentityTrace, tpb: Int = -1
) raises -> List[Float32]:
    """`.forecast(h)`: `h * batch_size` values, time-major (`[s + i *
    batch_size]`)."""
    if h <= 0:
        raise Error("h must be > 0. Currently: " + String(h))
    var level_d = upload_f32(ctx, fitted.level)
    var trend_d = upload_f32(ctx, fitted.trend)
    var season_d = upload_f32(ctx, fitted.season)
    var fc_d = ctx.enqueue_create_buffer[DType.float32](h * fitted.batch_size)
    ctx.synchronize()
    forecast(
        ctx, fitted.n, fitted.batch_size, fitted.frequency, h, fitted.seasonal,
        level_d, trend_d, season_d, fc_d, trace, tpb,
    )
    var out = download_f32(ctx, fc_d, h * fitted.batch_size)
    _ = level_d^
    _ = trend_d^
    _ = season_d^
    _ = fc_d^
    return out^


def holtwinters_forecast_host(fitted: HWFit, h: Int) raises -> List[Float32]:
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    return holtwinters_forecast_host_traced(ctx, fitted, h, trace)
