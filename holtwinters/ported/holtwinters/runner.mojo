"""cuML `cpp/src/holtwinters/runner.cuh` (v26.08.00).

`HWTranspose`, `HoltWintersBufferSize`, `HoltWintersDecompose` (the
seasonal arm), `HoltWintersEval`, `HoltWintersOptim` (their defaults),
`HoltWintersForecast`, `HoltWintersFitHelper`, `HoltWintersForecastHelper`,
plus the validation that cuML's Python wrapper performs
(`holtwinters.pyx:162-282`) and DEVIATION 664's finiteness refusals.

THE CARD (`core/identity_trace.mojo`, every float stage through DEVIATION
661's canonicalizing record): `hw.input` (the transposed dataset, n x
batch), `hw.decomp.trend` / `hw.decomp.season` (the smoothed trend and the
residual over the first `start_periods` seasons), `hw.start.level`,
`hw.start.trend`, `hw.start.season`, `hw.opt.iterNNN.params` (NNN < the
largest iteration count in the batch, capped at `trace_iters`),
`hw.opt.niter` (i32), `hw.opt.criterion` (i32), `hw.params` (bounded
alpha | beta | gamma), `hw.sse`, `hw.level`, `hw.trend`, `hw.season`;
`HoltWintersForecastHelper` adds `hw.forecast`.

============ DEVIATION 661 (2026-08-23): THE CARD RECORDS ONE NaN PAYLOAD
============ (solver/mojo_only/record_canon.mojo, DEVIATION 612's helper,
============ IMPORTED) ======================================================
WHAT THEIRS DOES: no card. Their optimizer can compute NaN (DEVIATION 662's
header lists the routes; cuml#888) and their multiplicative eval divides by
`clevel`, which a legal positive series with a negative fitted trend can
drive through zero.
WHAT OURS DOES: every float stage is hashed through a COPY whose NaNs are
rewritten to `0x7FC00000`; the live buffers are untouched; the host oracle's
card goes through `canon_nan_list`; `hw_check` compares device and oracle
cells after `canon_nan_f32`.
WHY: IDENTITY_PATHS row 39 FACT 2 -- a computed NaN's payload is the
vendor's (Apple 0x7fc00000, NVIDIA 0x7fffffff, AMD 0xffc00000), so a raw
hash of a stage holding one is a fingerprint, not an identity, while no
NaN payload steers any downstream bit here (every consumer is a compare,
payload-blind, or yields NaN again; `bound_device(NaN)` is +0.0 by
DEVIATION 663 on every vendor).
MEASURED: `hw_check::check_hw_zero_series_keeps_start` with the
NO_ZERO_DIR_GUARD sabotage: the device writes its NaN and the card reads
0x7fc00000; the solver lane's gate measured the mechanism itself.
============================================================================

============ DEVIATION 664 (2026-08-23): NON-FINITE INPUT, AND NON-POSITIVE
============ INPUT UNDER MULTIPLICATIVE, ARE REFUSED BY NAME ================
WHAT THEIRS DOES (`holtwinters.pyx:236-241`): `check_array(...,
ensure_all_finite=False)`; the C++ has no guard. A NaN or inf anywhere
reaches every stage; a zero or negative value under MULTIPLICATIVE divides
(`season = ts / trend`, `/= mean`, `pts / stmp_eps`, `pts / clevel`) and a
zero trend window makes `0/0`.
WHAT OURS DOES: `holtwinters_validate_data` raises naming the series, the
position and the value BEFORE any upload; under MULTIPLICATIVE every value
must be > 0 (statsmodels' ExponentialSmoothing raises the same way:
"endog must be strictly positive when using multiplicative trend or
seasonal components").
WHY: every stage is recorded and a computed NaN carries the vendor's
payload (row 39). SSE OVERFLOW is NOT guarded: `sum diff^2` past FLT_MAX
is `+inf`, the same bits everywhere; the optimizer's `inf - inf` then
canonicalizes through DEVIATION 661 (README names it).
MEASURED: `hw_check::check_hw_refusals`.
============================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from holtwinters.ported.holtwinters.internal.hw_decompose import stl_decomposition_gpu
from holtwinters.ported.holtwinters.internal.hw_eval import HW_WRITE_ALL, holtwinters_eval_gpu
from holtwinters.ported.holtwinters.internal.hw_forecast import holtwinters_forecast_gpu
from holtwinters.ported.holtwinters.internal.hw_optim import holtwinters_optim_gpu
from holtwinters.ported.holtwinters.internal.hw_utils import (
    HW_OPTIM_TPB,
    get_threads_per_block,
    hw_is_finite,
)
from holtwinters.ported.tsa.holtwinters_params import (
    OptimParams,
    SEASONAL_ADDITIVE,
    SEASONAL_MULTIPLICATIVE,
)
from solver.mojo_only.record_canon import record_device_canon

#: `HoltWintersFitHelper`'s initial alpha/beta/gamma (`runner.cuh:334-336`)
comptime HW_ALPHA0 = Float32(0.4)
comptime HW_BETA0 = Float32(0.3)
comptime HW_GAMMA0 = Float32(0.3)
#: the Python default `eps=2.24e-3` (`holtwinters.pyx:164`), as (float)2.24e-3
comptime HW_DEFAULT_EPS = Float32(Float64(2.24e-3))
#: card cap on per-iteration stages (DEVIATION 665)
comptime HW_DEFAULT_TRACE_ITERS = 64


def default_optim_params(epsilon: Float32) -> OptimParams:
    """`HoltWintersOptim`'s defaults (`runner.cuh:210-219`): no caller of
    the fit path passes an `OptimParams`, so the override block (`:221-240`)
    never runs and is not ported (UNPORTED.tsv)."""
    return OptimParams(
        eps=epsilon,
        min_param_diff=Float32(Float64(1e-8)),
        min_error_diff=Float32(Float64(1e-8)),
        min_grad_norm=Float32(Float64(1e-4)),
        bfgs_iter_limit=1000,
        linesearch_iter_limit=100,
        linesearch_tau=Float32(0.5),
        linesearch_c=Float32(0.8),
        linesearch_step_size=Float32(-1.0),
    )


@fieldwise_init
struct HWBufferSizes(Copyable, Movable, ImplicitlyCopyable):
    var start_leveltrend_len: Int
    var start_season_len: Int
    var components_len: Int
    var error_len: Int
    var leveltrend_coef_shift: Int
    var season_coef_shift: Int


def holtwinters_buffer_size(
    n: Int, batch_size: Int, frequency: Int, use_beta: Bool, use_gamma: Bool
) raises -> HWBufferSizes:
    """`HoltWintersBufferSize` (`runner.cuh:53-96`), `w_len = frequency`
    when `use_gamma`. Their int64 overflow trap is Mojo's native Int."""
    var w_len: Int
    if use_gamma:
        w_len = frequency
    elif use_beta:
        w_len = 2
    else:
        w_len = 1
    var n_minus = n - w_len
    if n_minus < 1:
        raise Error("HoltWintersBufferSize: n - w_len = " + String(n_minus) + " < 1")
    return HWBufferSizes(
        start_leveltrend_len=batch_size,
        start_season_len=frequency * batch_size if use_gamma else 0,
        components_len=n_minus * batch_size,
        error_len=batch_size,
        leveltrend_coef_shift=(n_minus - 1) * batch_size,
        season_coef_shift=(n_minus - frequency) * batch_size if use_gamma else 0,
    )


def hw_transpose_kernel(
    data_in: MutPointer[Float32, MutAnyOrigin],
    data_out: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    n_in: Int32,
):
    """`HWTranspose` (`runner.cuh:27-37`): RAFT `transpose` (a cuBLAS
    `geam`, CLOSED) of the `batch_size x n` row-major input into the
    `n x batch_size` time-major dataset the kernels read (`ts[tid + t *
    batch_size]`). A pure permutation: no arithmetic, so one thread per
    cell is bit-identical to whatever the library does."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var batch_size = Int(batch_size_in)
    var n = Int(n_in)
    if idx < batch_size * n:
        var s = idx // n
        var t = idx % n
        data_out.unsafe_store(s + t * batch_size, data_in.unsafe_load(idx))


def hw_transpose(
    ctx: DeviceContext,
    mut data_in: DeviceBuffer[DType.float32],
    batch_size: Int,
    n: Int,
    mut data_out: DeviceBuffer[DType.float32],
    tpb: Int,
) raises:
    var cells = batch_size * n
    ctx.enqueue_function[hw_transpose_kernel](
        data_in.unsafe_ptr(), data_out.unsafe_ptr(), Int32(batch_size), Int32(n),
        grid_dim=((cells + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()


def holtwinters_validate_params(
    n: Int, batch_size: Int, frequency: Int, start_periods: Int, eps: Float32
) raises:
    """`holtwinters.pyx:172-220, 275-282` exactly, by name."""
    if batch_size <= 0:
        raise Error("ts_num: Must state at least 1 series. Given: " + String(batch_size))
    if frequency < 2:
        raise Error("seasonal_periods: Frequency must be >= 2. Given: " + String(frequency))
    if start_periods < 2:
        raise Error("start_periods: Start Periods must be >= 2. Given: " + String(start_periods))
    if frequency < start_periods:
        raise Error(
            "seasonal_periods (" + String(frequency) + ") cannot be less than start_periods ("
            + String(start_periods) + ")."
        )
    if not (eps > Float32(0.0)):
        raise Error("eps: Epsilon must be positive. Given: " + String(eps))
    if n <= 0:
        raise Error("Time series must contain at least 1 value. Given: " + String(n))
    if n < start_periods * frequency:
        raise Error(
            "Length of time series (" + String(n) + ") must be at least freq*start_periods ("
            + String(start_periods * frequency) + ")."
        )


def holtwinters_validate_data(
    data: List[Float32], n: Int, batch_size: Int, seasonal: Int
) raises:
    """DEVIATION 664: finite everywhere; > 0 under MULTIPLICATIVE. `data` is
    series-major (`batch_size x n`, each series contiguous)."""
    if len(data) != n * batch_size:
        raise Error(
            "endog: expected " + String(n * batch_size) + " values (ts_num=" + String(batch_size)
            + " x n=" + String(n) + "), got " + String(len(data))
        )
    for s in range(batch_size):
        for t in range(n):
            var v = data[s * n + t]
            if not hw_is_finite(v):
                raise Error(
                    "endog[series " + String(s) + ", t=" + String(t) + "]=" + String(v)
                    + " is not finite (DEVIATION 664)"
                )
            if seasonal == SEASONAL_MULTIPLICATIVE and not (v > Float32(0.0)):
                raise Error(
                    "endog[series " + String(s) + ", t=" + String(t) + "]=" + String(v)
                    + " must be strictly positive under seasonal='multiplicative'"
                    " (DEVIATION 664)"
                )


def _iter_stage_tag(iter: Int) -> String:
    var s = String(iter)
    while s.byte_length() < 3:
        s = "0" + s
    return "hw.opt.iter" + s + ".params"


def holtwinters_fit_helper(
    ctx: DeviceContext,
    mut data: DeviceBuffer[DType.float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    start_periods: Int,
    seasonal: Int,
    epsilon: Float32,
    mut level_d: DeviceBuffer[DType.float32],
    mut trend_d: DeviceBuffer[DType.float32],
    mut season_d: DeviceBuffer[DType.float32],
    mut error_d: DeviceBuffer[DType.float32],
    mut alpha_d: DeviceBuffer[DType.float32],
    mut beta_d: DeviceBuffer[DType.float32],
    mut gamma_d: DeviceBuffer[DType.float32],
    mut criterion_d: DeviceBuffer[DType.int32],
    mut niter_d: DeviceBuffer[DType.int32],
    mut iter_trace_d: DeviceBuffer[DType.float32],
    trace_iters: Int,
    mut trace: IdentityTrace,
    tpb_decomp: Int = -1,
    tpb_optim: Int = HW_OPTIM_TPB,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = Float32(0.0),
) raises:
    """`HoltWintersFitHelper` (`runner.cuh:320-428`): transpose, decompose,
    optimize (BFGS, all three parameters), with the card. `data` is the
    `batch_size x n` row-major input (series-major). Outputs: `level_d`,
    `trend_d`, `season_d` hold `components_len = (n - frequency) *
    batch_size` each (time-major), `error_d` the SSE per series,
    `alpha_d/beta_d/gamma_d` the bounded parameters, `criterion_d`/
    `niter_d` (DEVIATION 665), `iter_trace_d` the first `trace_iters`
    iterations' parameters (`3 * batch_size` per iteration). `tpb_decomp <
    0` means their `GET_THREADS_PER_BLOCK(batch_size)`; `tpb_optim` their
    128; both are scheduling knobs the gates vary. `scratch_pad` /
    `scratch_poison` over-allocate and pre-fill every scratch buffer."""
    var additive = seasonal == SEASONAL_ADDITIVE
    var sizes = holtwinters_buffer_size(n, batch_size, frequency, True, True)
    var tpb_d = tpb_decomp if tpb_decomp > 0 else get_threads_per_block(batch_size)
    if len(level_d) < sizes.components_len or len(trend_d) < sizes.components_len or len(season_d) < sizes.components_len:
        raise Error("holtwinters_fit_helper: level/trend/season must hold components_len = " + String(sizes.components_len))
    if len(error_d) < batch_size or len(alpha_d) < batch_size or len(beta_d) < batch_size or len(gamma_d) < batch_size or len(criterion_d) < batch_size or len(niter_d) < batch_size:
        raise Error("holtwinters_fit_helper: per-series outputs must hold batch_size")
    if trace_iters > 0 and len(iter_trace_d) < trace_iters * 3 * batch_size:
        raise Error("holtwinters_fit_helper: iter_trace_d must hold trace_iters * 3 * batch_size")

    # initial values for alpha, beta and gamma
    alpha_d.enqueue_fill(HW_ALPHA0)
    beta_d.enqueue_fill(HW_BETA0)
    gamma_d.enqueue_fill(HW_GAMMA0)
    var dataset_d = ctx.enqueue_create_buffer[DType.float32](batch_size * n + scratch_pad)
    dataset_d.enqueue_fill(scratch_poison)
    var level_seed_d = ctx.enqueue_create_buffer[DType.float32](sizes.start_leveltrend_len + scratch_pad)
    var trend_seed_d = ctx.enqueue_create_buffer[DType.float32](sizes.start_leveltrend_len + scratch_pad)
    var start_season_d = ctx.enqueue_create_buffer[DType.float32](sizes.start_season_len + scratch_pad)
    level_seed_d.enqueue_fill(scratch_poison)
    trend_seed_d.enqueue_fill(scratch_poison)
    start_season_d.enqueue_fill(scratch_poison)
    var pseason_d = ctx.enqueue_create_buffer[DType.float32](batch_size * frequency + scratch_pad)
    pseason_d.enqueue_fill(scratch_poison)
    var xhat_dummy = ctx.enqueue_create_buffer[DType.float32](1)
    if trace_iters > 0:
        # DEVIATION 665: a series that stops early leaves its later slots
        # at this fill, so the stage is a function of the fit alone.
        var view = iter_trace_d.create_sub_buffer[DType.float32](0, trace_iters * 3 * batch_size)
        view.enqueue_fill(Float32(0.0))
        _ = view^
    ctx.synchronize()
    var canon_scratch_n = batch_size * n
    if sizes.components_len > canon_scratch_n:
        canon_scratch_n = sizes.components_len
    if trace_iters * 3 * batch_size > canon_scratch_n:
        canon_scratch_n = trace_iters * 3 * batch_size
    var canon_scratch = ctx.enqueue_create_buffer[DType.float32](canon_scratch_n)
    ctx.synchronize()

    # Step 1: transpose the dataset (ML expects col major dataset)
    hw_transpose(ctx, data, batch_size, n, dataset_d, tpb_d)
    record_device_canon(ctx, trace, "hw.input", dataset_d, batch_size * n, canon_scratch)

    # Step 2: Decompose dataset to get seed for level, trend and seasonal values
    var decomp_trend = ctx.enqueue_create_buffer[DType.float32](1)
    var decomp_season = ctx.enqueue_create_buffer[DType.float32](1)
    stl_decomposition_gpu(
        ctx, dataset_d, n, batch_size, frequency, start_periods,
        level_seed_d, trend_seed_d, start_season_d, seasonal, tpb_d,
        scratch_pad, scratch_poison, decomp_trend, decomp_season,
    )
    var filter_size = (frequency // 2) * 2 + 1
    var trend_len = start_periods * frequency - (filter_size - 1)
    record_device_canon(ctx, trace, "hw.decomp.trend", decomp_trend, trend_len * batch_size, canon_scratch)
    record_device_canon(ctx, trace, "hw.decomp.season", decomp_season, trend_len * batch_size, canon_scratch)
    record_device_canon(ctx, trace, "hw.start.level", level_seed_d, batch_size, canon_scratch)
    record_device_canon(ctx, trace, "hw.start.trend", trend_seed_d, batch_size, canon_scratch)
    record_device_canon(ctx, trace, "hw.start.season", start_season_d, frequency * batch_size, canon_scratch)

    # Step 3: Find optimal alpha, beta and gamma values (seasonal HW)
    var p = default_optim_params(epsilon)
    holtwinters_optim_gpu(
        ctx, dataset_d, n, batch_size, frequency,
        level_seed_d, trend_seed_d, start_season_d,
        alpha_d, True, beta_d, True, gamma_d, True,
        level_d, trend_d, season_d, xhat_dummy, error_d,
        HW_WRITE_ALL ^ 8, True,  # level, trend, season; xhat is their nullptr
        criterion_d, niter_d, iter_trace_d, trace_iters,
        additive, True, True,
        p.eps, p.min_param_diff, p.min_error_diff, p.min_grad_norm,
        p.bfgs_iter_limit, p.linesearch_iter_limit,
        p.linesearch_tau, p.linesearch_c, p.linesearch_step_size,
        pseason_d, tpb_optim,
    )
    if trace.enabled:
        # the per-iteration stages, up to the largest niter in the batch
        var nh = ctx.enqueue_create_host_buffer[DType.int32](batch_size)
        ctx.enqueue_copy(dst_ptr=nh.unsafe_ptr(), src_buf=niter_d)
        ctx.synchronize()
        var max_iter = 0
        for s in range(batch_size):
            var v = Int(nh.unsafe_ptr().unsafe_load(s))
            if v > max_iter:
                max_iter = v
        _ = nh^
        var shown = max_iter if max_iter < trace_iters else trace_iters
        if max_iter > trace_iters:
            trace.header(
                "hw.opt: largest niter " + String(max_iter) + " exceeds trace_iters "
                + String(trace_iters) + "; per-iteration stages truncated"
            )
        for it in range(shown):
            var view = iter_trace_d.create_sub_buffer[DType.float32](it * 3 * batch_size, 3 * batch_size)
            record_device_canon(ctx, trace, _iter_stage_tag(it), view, 3 * batch_size, canon_scratch)
            _ = view^
        trace.record_device[DType.int32](ctx, "hw.opt.niter", niter_d, batch_size)
        trace.record_device[DType.int32](ctx, "hw.opt.criterion", criterion_d, batch_size)
        # hw.params: alpha | beta | gamma
        var params = ctx.enqueue_create_buffer[DType.float32](3 * batch_size)
        var pa = params.create_sub_buffer[DType.float32](0, batch_size)
        var pb = params.create_sub_buffer[DType.float32](batch_size, batch_size)
        var pg = params.create_sub_buffer[DType.float32](2 * batch_size, batch_size)
        var sa = alpha_d.create_sub_buffer[DType.float32](0, batch_size)
        var sb = beta_d.create_sub_buffer[DType.float32](0, batch_size)
        var sg = gamma_d.create_sub_buffer[DType.float32](0, batch_size)
        ctx.enqueue_copy(dst_buf=pa, src_buf=sa)
        ctx.enqueue_copy(dst_buf=pb, src_buf=sb)
        ctx.enqueue_copy(dst_buf=pg, src_buf=sg)
        ctx.synchronize()
        record_device_canon(ctx, trace, "hw.params", params, 3 * batch_size, canon_scratch)
        _ = pa^
        _ = pb^
        _ = pg^
        _ = sa^
        _ = sb^
        _ = sg^
        _ = params^
        record_device_canon(ctx, trace, "hw.sse", error_d, batch_size, canon_scratch)
        record_device_canon(ctx, trace, "hw.level", level_d, sizes.components_len, canon_scratch)
        record_device_canon(ctx, trace, "hw.trend", trend_d, sizes.components_len, canon_scratch)
        record_device_canon(ctx, trace, "hw.season", season_d, sizes.components_len, canon_scratch)
    ctx.synchronize()
    _ = dataset_d^
    _ = level_seed_d^
    _ = trend_seed_d^
    _ = start_season_d^
    _ = pseason_d^
    _ = xhat_dummy^
    _ = canon_scratch^
    _ = decomp_trend^
    _ = decomp_season^


def holtwinters_eval(
    ctx: DeviceContext,
    mut ts_time_major: DeviceBuffer[DType.float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    mut start_level: DeviceBuffer[DType.float32],
    mut start_trend: DeviceBuffer[DType.float32],
    mut start_season: DeviceBuffer[DType.float32],
    mut alpha: DeviceBuffer[DType.float32],
    mut beta: DeviceBuffer[DType.float32],
    mut gamma: DeviceBuffer[DType.float32],
    mut level: DeviceBuffer[DType.float32],
    mut trend: DeviceBuffer[DType.float32],
    mut season: DeviceBuffer[DType.float32],
    mut error: DeviceBuffer[DType.float32],
    seasonal: Int,
    tpb: Int = -1,
) raises:
    """`HoltWintersEval` (`runner.cuh:144-195`), the seasonal arm (all three
    parameters present): the recurrence at GIVEN parameters. `ts` is
    time-major (already transposed). Used by the gates to plant parameters
    at the clamp."""
    if frequency < 2:
        raise Error("HoltWintersEval: start_season != nullptr && frequency < 2")
    var tpb_ = tpb if tpb > 0 else get_threads_per_block(batch_size)
    var pseason = ctx.enqueue_create_buffer[DType.float32](batch_size * frequency)
    var xhat = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    holtwinters_eval_gpu(
        ctx, ts_time_major, n, batch_size, frequency, start_level, start_trend, start_season,
        alpha, beta, gamma, level, trend, season, xhat, error,
        HW_WRITE_ALL ^ 8, True, True, True, seasonal == SEASONAL_ADDITIVE, pseason, tpb_,
    )
    _ = pseason^
    _ = xhat^


def holtwinters_optim(
    ctx: DeviceContext,
    mut ts_time_major: DeviceBuffer[DType.float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    mut start_level: DeviceBuffer[DType.float32],
    mut start_trend: DeviceBuffer[DType.float32],
    mut start_season: DeviceBuffer[DType.float32],
    mut alpha: DeviceBuffer[DType.float32],
    mut beta: DeviceBuffer[DType.float32],
    mut gamma: DeviceBuffer[DType.float32],
    mut level: DeviceBuffer[DType.float32],
    mut trend: DeviceBuffer[DType.float32],
    mut season: DeviceBuffer[DType.float32],
    mut error: DeviceBuffer[DType.float32],
    mut optim_result: DeviceBuffer[DType.int32],
    mut niter: DeviceBuffer[DType.int32],
    mut iter_trace: DeviceBuffer[DType.float32],
    trace_iters: Int,
    epsilon: Float32,
    seasonal: Int,
    tpb: Int = HW_OPTIM_TPB,
) raises:
    """`HoltWintersOptim` (`runner.cuh:201-299`), the seasonal BFGS arm
    with their defaults: optimize all three parameters FROM THE GIVEN
    `alpha/beta/gamma` (theirs: whatever the caller uploaded; the fit helper
    uploads 0.4/0.3/0.3). `ts` is time-major. The gates use this to plant
    start parameters (-0.0, NaN) at the recorded clamp."""
    if frequency < 2:
        raise Error("HoltWintersOptim: start_season && frequency < 2")
    var pseason = ctx.enqueue_create_buffer[DType.float32](batch_size * frequency)
    var xhat = ctx.enqueue_create_buffer[DType.float32](1)
    if trace_iters > 0:
        var view = iter_trace.create_sub_buffer[DType.float32](0, trace_iters * 3 * batch_size)
        view.enqueue_fill(Float32(0.0))
        _ = view^
    ctx.synchronize()
    var p = default_optim_params(epsilon)
    holtwinters_optim_gpu(
        ctx, ts_time_major, n, batch_size, frequency, start_level, start_trend, start_season,
        alpha, True, beta, True, gamma, True,
        level, trend, season, xhat, error, HW_WRITE_ALL ^ 8, True,
        optim_result, niter, iter_trace, trace_iters,
        seasonal == SEASONAL_ADDITIVE, True, True,
        p.eps, p.min_param_diff, p.min_error_diff, p.min_grad_norm,
        p.bfgs_iter_limit, p.linesearch_iter_limit,
        p.linesearch_tau, p.linesearch_c, p.linesearch_step_size,
        pseason, tpb,
    )
    _ = pseason^
    _ = xhat^


def holtwinters_forecast_helper(
    ctx: DeviceContext,
    n: Int,
    batch_size: Int,
    frequency: Int,
    h: Int,
    seasonal: Int,
    mut level_d: DeviceBuffer[DType.float32],
    mut trend_d: DeviceBuffer[DType.float32],
    mut season_d: DeviceBuffer[DType.float32],
    mut forecast_d: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
    tpb: Int = -1,
) raises:
    """`HoltWintersForecastHelper` (`runner.cuh:431-460`): the seasonal
    forecast from the LAST fitted row of level/trend and the last
    `frequency` rows of season. `forecast_d` holds `h * batch_size`
    (time-major). Card: `hw.forecast`."""
    if h <= 0:
        raise Error("h must be > 0. Currently: " + String(h))
    var sizes = holtwinters_buffer_size(n, batch_size, frequency, True, True)
    if len(level_d) < sizes.components_len or len(trend_d) < sizes.components_len or len(season_d) < sizes.components_len:
        raise Error("holtwinters_forecast_helper: level/trend/season must hold components_len")
    if len(forecast_d) < h * batch_size:
        raise Error("holtwinters_forecast_helper: forecast_d must hold h * batch_size")
    var tpb_ = tpb if tpb > 0 else get_threads_per_block(batch_size)
    holtwinters_forecast_gpu(
        ctx, forecast_d, h, batch_size, frequency,
        level_d, sizes.leveltrend_coef_shift,
        trend_d, sizes.leveltrend_coef_shift,
        season_d, sizes.season_coef_shift,
        seasonal == SEASONAL_ADDITIVE, tpb_,
    )
    if trace.enabled:
        var scratch = ctx.enqueue_create_buffer[DType.float32](h * batch_size)
        ctx.synchronize()
        record_device_canon(ctx, trace, "hw.forecast", forecast_d, h * batch_size, scratch)
        _ = scratch^
