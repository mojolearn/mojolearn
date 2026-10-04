# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host surface for Holt-Winters: what `bindings/` calls.

WIRED 2026-08-24 (DEVIATIONS 900-909). `bindings/_mojolearn_tsa.mojo` calls
`holtwinters_fit_ptr` / `holtwinters_forecast_ptr` at the bottom of this
file, and `python/mojolearn/_tsa_impl.py::ExponentialSmoothing` calls those.
The list-shaped entries in the middle of the file are what the gates and the
pointer entries both go through, so there is one fit path and not two.

The sentence this replaced said NOT YET WIRED, which was true when it was
written and stopped being true here.

`holtwinters_fit_host` takes the `batch_size x n` series-major data as a
host list (each series contiguous, cuML's numpy convention `(ts_num, n)`),
validates exactly as `holtwinters.pyx` does (`holtwinters_validate_params`,
by name) plus DEVIATION 664's finiteness / positivity rules, uploads, runs
`ML::HoltWinters::fit` (`holtwinters/impl/holtwinters.mojo`)
with the given trace, and returns every output as host lists.
`holtwinters_forecast_host` re-uploads the fitted components and runs
`ML::HoltWinters::forecast` (the one-shot form; a bindings layer that
keeps the `DeviceBuffer`s can call the implemented entries directly).
"""

# DEVIATION 2486: bulk host staging; stream/lifetime boundaries unchanged.
from bindings.hostptr import copy_f32
from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL
from checks.numerics import ftz
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from std.memory import bitcast
from std.sys.compile import is_defined

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoTsaContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoTsaContextFast"


from core.identity_trace import IdentityTrace
from holtwinters.impl.holtwinters import buffer_size, fit, forecast
from holtwinters.impl.internal.hw_estimate import HW_INIT_ESTIMATED, HW_INIT_HEURISTIC
from holtwinters.impl.internal.hw_utils import HW_OPTIM_TPB
from holtwinters.impl.runner import (
    HW_DEFAULT_EPS,
    HW_DEFAULT_TRACE_ITERS,
    holtwinters_validate_data,
    holtwinters_validate_params,
)
from holtwinters.impl.tsa.holtwinters_params import (
    SEASONAL_ADDITIVE,
    SEASONAL_MULTIPLICATIVE,
    seasonal_from_name,
)

#: lane/fam-timeseries (2026-10-04), IDENTICAL on every vendor: the binding's
#: fit (`holtwinters_fit_ptr`) uploads from the caller's address, checks the
#: input with a device kernel (one flag word) and brings every output down
#: into the caller's buffers with ONE wait, instead of a host copy of the
#: input into a list, a host walk over every value, eleven downloads each
#: with its own wait (the 64-iteration trace among them, which the binding
#: never hands out) and element-by-element copies out. The fit itself is
#: `fit(...)` with the same arguments, so no bit moves. An input the host
#: walk would refuse, and a run with the identity trace on, take the old
#: sequence (which raises the refusal by name, series and time).
#: `-D MOJOLEARN_IDN_HW_FIT_DIRECT_OFF=1` (or MOJOLEARN_IDN_ALL_OFF)
#: restores the old sequence. FAST is unchanged.
comptime HW_FIT_DIRECT = _DEVCTX_MODE == _DEVCTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_HW_FIT_DIRECT_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime HW_FLAG_TPB = 256
#: lane/fam-timeseries (2026-10-04), IDENTICAL on every vendor: the binding's
#: forecast (`holtwinters_forecast_ptr`) uploads the three component blocks
#: from the caller's packed buffer and downloads into the caller's output in
#: bulk, instead of three element-by-element host copies of every component,
#: three uploads each with its own wait and an element-by-element copy out.
#: The forecast is `forecast(...)` with the same arguments; no bit moves.
#: `-D MOJOLEARN_IDN_HW_FORECAST_DIRECT_OFF=1` (or MOJOLEARN_IDN_ALL_OFF)
#: restores the old sequence, which a traced run and `h <= 0` still take.
comptime HW_FORECAST_DIRECT = _DEVCTX_MODE == _DEVCTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_HW_FORECAST_DIRECT_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


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
    copy_f32(values.unsafe_ptr(), host.unsafe_ptr(), n)
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
    init_method: Int = HW_INIT_HEURISTIC,
) raises -> HWFit:
    """`ExponentialSmoothing(endog, seasonal, seasonal_periods=frequency,
    start_periods, ts_num=batch_size, eps).fit()` with an explicit trace
    and the scheduling / padding knobs the gates vary. `init_method` is
    `hw_estimate.mojo`'s code (HEURISTIC, cuML's fit, unless named)."""
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
        + (" initialization=estimated" if init_method == HW_INIT_ESTIMATED else "")
    )
    fit(
        ctx, n, batch_size, frequency, start_periods, st, eps, ddata,
        level_d, trend_d, season_d, error_d, alpha_d, beta_d, gamma_d,
        criterion_d, niter_d, decisions_d, iter_trace_d, trace, trace_iters,
        tpb_decomp, tpb_optim, scratch_pad, scratch_poison, init_method,
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
    init_method: Int = HW_INIT_HEURISTIC,
) raises -> HWFit:
    """The bindings entry: the environment's trace (`MOJOLEARN_IDENTITY_TRACE`),
    their block widths, no padding."""
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var trace = IdentityTrace()
    return holtwinters_fit_host_traced(
        ctx, data, n, batch_size, frequency, start_periods, seasonal, eps, trace,
        init_method=init_method,
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
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var trace = IdentityTrace()
    return holtwinters_forecast_host_traced(ctx, fitted, h, trace)


# =============================================================================
# THE POINTER SURFACE bindings/_mojolearn_tsa.mojo CALLS
# =============================================================================
# ADDED 2026-08-24 with the Python surface (DEVIATIONS 900-909). The two
# entries above take and return host `List`s, which is the right shape for a
# gate and the wrong shape for a CPython binding: a binding wants to hold the
# GIL released across the whole device call, and moving a `HWFit` across that
# boundary buys nothing. These two take the caller's numpy addresses directly,
# in the same style as `decomposition/estimator.mojo::pca_fit_host`, and every
# byte of arithmetic still goes through `holtwinters_fit_host` /
# `holtwinters_forecast_host` above, so there is one fit path and not two.
#
# THE PACKED BUFFERS, AND WHY THEY ARE PACKED. `PythonModuleBuilder.
# def_function` infers its signature from arity and gives out around nine
# arguments, so `fit` cannot hand back eight separate addresses. The three
# component series share one float32 buffer and the four per-series scalars
# share another. Each layout is written out below and in the SAME WORDS in
# `bindings/_mojolearn_tsa.mojo` and `python/mojolearn/_tsa_impl.py`. A
# silently reordered slice is a wrong answer rather than a failure, which is
# why the three copies of the order are meant to be diffable by eye.


def hw_input_flag_kernel(
    flag: MutPointer[UInt32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int64,
    positive_in: Int32,
):
    """flag[0] = 1 when any input value is one `holtwinters_validate_data`
    refuses: non-finite, or (multiplicative) not strictly positive. Every
    such thread writes the same word; the flag starts 0."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(cells_in):
        return
    var v = y.unsafe_load(i)
    var refused = not isfinite(v)
    if positive_in != 0 and not (v > Float32(0.0)):
        refused = True
    if refused:
        flag.unsafe_store(0, UInt32(1))


def _holtwinters_fit_direct(
    data_ptr: MutPointer[Float32, MutUntrackedOrigin],
    comps_ptr: MutPointer[Float32, MutUntrackedOrigin],
    stats_ptr: MutPointer[Float32, MutUntrackedOrigin],
    flags_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n: Int,
    batch_size: Int,
    frequency: Int,
    start_periods: Int,
    seasonal: String,
    eps: Float32,
    init_method: Int,
) raises -> Int:
    """HW_FIT_DIRECT (lane/fam-timeseries): `holtwinters_fit_ptr`'s answer
    with the input and the outputs moved in bulk and one wait after the fit.
    Returns `components_len`, or -1 with nothing written when the caller
    must take the old sequence (the identity trace is on, or the input holds
    a value the host walk refuses by name). The caller has checked
    `batch_size`, `n` and their product."""
    var trace = IdentityTrace()
    if trace.enabled:
        return -1
    # the old sequence's refusals, in its order
    var st = seasonal_from_name(seasonal)
    holtwinters_validate_params(n, batch_size, frequency, start_periods, eps)
    var cells = batch_size * n
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var sizes = buffer_size(n, batch_size, frequency)
    var ddata = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.enqueue_copy(dst_buf=ddata, src_ptr=data_ptr)
    var flag_d = ctx.enqueue_create_buffer[DType.uint32](1)
    var flag_h = ctx.enqueue_create_host_buffer[DType.uint32](1)
    flag_d.enqueue_fill(UInt32(0))
    ctx.enqueue_function[hw_input_flag_kernel](
        flag_d.unsafe_ptr(), ddata.unsafe_ptr(), Int64(cells),
        Int32(1 if st == SEASONAL_MULTIPLICATIVE else 0),
        grid_dim=(cells + HW_FLAG_TPB - 1) // HW_FLAG_TPB,
        block_dim=HW_FLAG_TPB,
    )
    ctx.enqueue_copy(dst_ptr=flag_h.unsafe_ptr(), src_buf=flag_d)
    ctx.synchronize()
    var refused = flag_h.unsafe_ptr().unsafe_load(0) != UInt32(0)
    _ = flag_h^
    _ = flag_d^
    if refused:
        _ = ddata^
        _ = ctx^
        return -1
    var cl = sizes.components_len
    var n_trace = HW_DEFAULT_TRACE_ITERS * 3 * batch_size
    var level_d = ctx.enqueue_create_buffer[DType.float32](cl)
    var trend_d = ctx.enqueue_create_buffer[DType.float32](cl)
    var season_d = ctx.enqueue_create_buffer[DType.float32](cl)
    var error_d = ctx.enqueue_create_buffer[DType.float32](batch_size)
    var alpha_d = ctx.enqueue_create_buffer[DType.float32](batch_size)
    var beta_d = ctx.enqueue_create_buffer[DType.float32](batch_size)
    var gamma_d = ctx.enqueue_create_buffer[DType.float32](batch_size)
    var criterion_d = ctx.enqueue_create_buffer[DType.int32](batch_size)
    var niter_d = ctx.enqueue_create_buffer[DType.int32](batch_size)
    var decisions_d = ctx.enqueue_create_buffer[DType.int32](batch_size)
    var iter_trace_d = ctx.enqueue_create_buffer[DType.float32](n_trace if n_trace > 0 else 1)
    # `holtwinters_fit_host_traced`'s fills (no padding, poison 0.0)
    level_d.enqueue_fill(Float32(0.0))
    trend_d.enqueue_fill(Float32(0.0))
    season_d.enqueue_fill(Float32(0.0))
    error_d.enqueue_fill(Float32(0.0))
    iter_trace_d.enqueue_fill(Float32(0.0))
    criterion_d.enqueue_fill(Int32(-7))
    niter_d.enqueue_fill(Int32(-7))
    decisions_d.enqueue_fill(Int32(-7))
    ctx.synchronize()
    fit(
        ctx, n, batch_size, frequency, start_periods, st, eps, ddata,
        level_d, trend_d, season_d, error_d, alpha_d, beta_d, gamma_d,
        criterion_d, niter_d, decisions_d, iter_trace_d, trace, HW_DEFAULT_TRACE_ITERS,
        -1, HW_OPTIM_TPB, 0, Float32(0.0), init_method,
    )
    # every output the binding hands out, in `holtwinters_fit_ptr`'s packed
    # order, one wait
    ctx.enqueue_copy(dst_ptr=comps_ptr, src_buf=level_d)
    ctx.enqueue_copy(dst_ptr=comps_ptr + cl, src_buf=trend_d)
    ctx.enqueue_copy(dst_ptr=comps_ptr + 2 * cl, src_buf=season_d)
    ctx.enqueue_copy(dst_ptr=stats_ptr, src_buf=error_d)
    ctx.enqueue_copy(dst_ptr=stats_ptr + batch_size, src_buf=alpha_d)
    ctx.enqueue_copy(dst_ptr=stats_ptr + 2 * batch_size, src_buf=beta_d)
    ctx.enqueue_copy(dst_ptr=stats_ptr + 3 * batch_size, src_buf=gamma_d)
    ctx.enqueue_copy(dst_ptr=flags_ptr, src_buf=niter_d)
    ctx.enqueue_copy(dst_ptr=flags_ptr + batch_size, src_buf=criterion_d)
    ctx.synchronize()
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
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return cl


def holtwinters_fit_ptr(
    data_ptr: MutPointer[Float32, MutUntrackedOrigin],
    comps_ptr: MutPointer[Float32, MutUntrackedOrigin],
    stats_ptr: MutPointer[Float32, MutUntrackedOrigin],
    flags_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n: Int,
    batch_size: Int,
    frequency: Int,
    start_periods: Int,
    seasonal: String,
    eps: Float32,
    init_method: Int = HW_INIT_HEURISTIC,
) raises -> Int:
    """`ExponentialSmoothing(endog, seasonal, seasonal_periods=frequency,
    start_periods, ts_num=batch_size, eps).fit()`. Returns `components_len
    = (n - frequency) * batch_size`.

    `data_ptr` reads `batch_size * n` float32, SERIES-MAJOR: series `s`
    occupies `[s * n, (s + 1) * n)`. That is `holtwinters.pyx::_check_dims`
    on a numpy input, which takes an `(ts_num, n)` array and `ravel()`s it
    in C order.

    `comps_ptr` is written with `3 * components_len` float32, in this exact
    order (the same words in `bindings/_mojolearn_tsa.mojo` and
    `python/mojolearn/_tsa_impl.py`):

        [0 * components_len, 1 * components_len)   level
        [1 * components_len, 2 * components_len)   trend
        [2 * components_len, 3 * components_len)   season

    Each block is TIME-MAJOR: series `s` at step `i` is at `[s + i *
    batch_size]`. That is `holtwinters.pyx:341-344`'s `reshape((ts_num,
    num_rows), order='F')`, and it is a TRANSPOSE of the input's layout.

    `stats_ptr` is written with `4 * batch_size` float32, in this exact
    order:

        [0 * batch_size, 1 * batch_size)   sse    (their `SSE`)
        [1 * batch_size, 2 * batch_size)   alpha
        [2 * batch_size, 3 * batch_size)   beta
        [3 * batch_size, 4 * batch_size)   gamma

    `flags_ptr` is written with `2 * batch_size` int32, in this exact
    order:

        [0 * batch_size, 1 * batch_size)   niter      (DEVIATION 665)
        [1 * batch_size, 2 * batch_size)   criterion  (OptimCriterion,
                                                       DEVIATION 665)

    `sse` is cuML's; `alpha`/`beta`/`gamma`/`niter`/`criterion` are OURS.
    cuML's `ExponentialSmoothing` returns none of the last five: the
    parameters live in device scratch their pyx never reads back, and
    `optim_result` is written only in the arm their fit does not take
    (`runner.cuh` / `hw_optim.cuh:801`, `holtwinters/NOT_IMPLEMENTED.tsv`).
    DEVIATION 699's `decisions` mask is NOT handed out here: it is a card
    instrument, it means nothing without the sabotage table beside it, and
    a published integer nobody can read is a liability.
    """
    # DEVIATION 931, 2026-08-24. THESE TWO GUARDS MUST PRECEDE THE READ LOOP
    # AND THE COMMENT THAT STOOD HERE SAID THEY DID NOT NEED TO.
    #
    # It read: "Validation is `holtwinters_fit_host`'s, by name, and it runs
    # before any upload; nothing is duplicated here." That is true of the
    # UPLOAD and false of the HOST READ below it. `holtwinters_validate_params`
    # runs inside `holtwinters_fit_host`, which is called AFTER this loop has
    # already dereferenced `data_ptr` `batch_size * n` times. A caller passing
    # a bad `n` or `batch_size` therefore reads out of bounds before any
    # validation sees the values, and `data_ptr` is a RAW ADDRESS crossing
    # from Python, so the caller supplying it is a user, not this tree.
    #
    # The sibling entry `holtwinters_forecast_ptr` already got this right and
    # guards before its own read. This is that pattern, and nothing more: the
    # FULL validation is still `holtwinters_fit_host`'s and is not duplicated.
    # These guard exactly the extents this loop dereferences.
    if batch_size < 1:
        raise Error(
            "holtwinters fit: batch_size must be >= 1 (batch_size="
            + String(batch_size) + "); the input read is batch_size * n cells"
        )
    if n < 1:
        raise Error(
            "holtwinters fit: n must be >= 1 (n=" + String(n)
            + "); the input read is batch_size * n cells"
        )
    var _cells = batch_size * n
    if _cells < 0 or _cells // batch_size != n:
        raise Error(
            "holtwinters fit: batch_size * n overflowed (batch_size="
            + String(batch_size) + ", n=" + String(n) + ")"
        )
    comptime if HW_FIT_DIRECT:
        # IDENTICAL on every vendor (lane/fam-timeseries): bulk transfers,
        # a device input check and one wait after the fit; -1 hands the call
        # to the sequence below (trace on, or a refusal it raises by name).
        var direct = _holtwinters_fit_direct(
            data_ptr, comps_ptr, stats_ptr, flags_ptr, n, batch_size, frequency,
            start_periods, seasonal, eps, init_method,
        )
        if direct >= 0:
            return direct
    var data = List[Float32]()
    data.reserve(_cells)
    for i in range(_cells):
        data.append(data_ptr.unsafe_load(i))
    var fitted = holtwinters_fit_host(
        data, n, batch_size, frequency, start_periods, seasonal, eps, init_method
    )
    var components_len = len(fitted.level)
    for i in range(components_len):
        comps_ptr.unsafe_store(i, fitted.level[i])
        comps_ptr.unsafe_store(components_len + i, fitted.trend[i])
        comps_ptr.unsafe_store(2 * components_len + i, fitted.season[i])
    for b in range(batch_size):
        stats_ptr.unsafe_store(b, fitted.sse[b])
        stats_ptr.unsafe_store(batch_size + b, fitted.alpha[b])
        stats_ptr.unsafe_store(2 * batch_size + b, fitted.beta[b])
        stats_ptr.unsafe_store(3 * batch_size + b, fitted.gamma[b])
        flags_ptr.unsafe_store(b, fitted.niter[b])
        flags_ptr.unsafe_store(batch_size + b, fitted.criterion[b])
    _ = fitted^
    return components_len


def holtwinters_forecast_ptr(
    comps_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    batch_size: Int,
    frequency: Int,
    seasonal: String,
    h: Int,
) raises -> Int:
    """`.forecast(h)`. Returns `h * batch_size`.

    `comps_ptr` reads `3 * components_len` float32 in the SAME packed order
    `holtwinters_fit_ptr` wrote them: level, then trend, then season, each
    `components_len = (n - frequency) * batch_size` and each time-major.

    `out_ptr` is written with `h * batch_size` float32, TIME-MAJOR: series
    `s` at step `i` is at `[s + i * batch_size]`. That is
    `holtwinters.pyx:386-388`'s `(ts_num, h)` array with `order="F"`.

    `n` and `frequency` are the FIT's, not the forecast's; they are what
    `HoltWintersForecastHelper` needs to find the last component of each
    series (`runner.cuh:414-458`).
    """
    var st = seasonal_from_name(seasonal)
    if n <= frequency:
        raise Error(
            "holtwinters forecast: n (" + String(n) + ") must exceed frequency ("
            + String(frequency) + "); there would be no fitted components"
        )
    if batch_size < 1:
        raise Error(
            "holtwinters forecast: batch_size must be >= 1 (batch_size="
            + String(batch_size) + ")"
        )
    var components_len = (n - frequency) * batch_size
    comptime if HW_FORECAST_DIRECT:
        var direct_trace = IdentityTrace()
        if h > 0 and not direct_trace.enabled:
            var ctx = process_ctx[_DEVCTX_SLOT]()
            var level_d = ctx.enqueue_create_buffer[DType.float32](components_len)
            var trend_d = ctx.enqueue_create_buffer[DType.float32](components_len)
            var season_d = ctx.enqueue_create_buffer[DType.float32](components_len)
            var fc_d = ctx.enqueue_create_buffer[DType.float32](h * batch_size)
            ctx.enqueue_copy(dst_buf=level_d, src_ptr=comps_ptr)
            ctx.enqueue_copy(dst_buf=trend_d, src_ptr=comps_ptr + components_len)
            ctx.enqueue_copy(dst_buf=season_d, src_ptr=comps_ptr + 2 * components_len)
            forecast(
                ctx, n, batch_size, frequency, h, st,
                level_d, trend_d, season_d, fc_d, direct_trace, -1,
            )
            ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=fc_d)
            ctx.synchronize()
            _ = level_d^
            _ = trend_d^
            _ = season_d^
            _ = fc_d^
            # DEVIATION 1946: the context dies LAST, after every value built on it.
            _ = ctx^
            return h * batch_size
    var fitted = HWFit(n, batch_size, frequency, st, 0)
    fitted.level.reserve(components_len)
    fitted.trend.reserve(components_len)
    fitted.season.reserve(components_len)
    for i in range(components_len):
        fitted.level.append(comps_ptr.unsafe_load(i))
        fitted.trend.append(comps_ptr.unsafe_load(components_len + i))
        fitted.season.append(comps_ptr.unsafe_load(2 * components_len + i))
    # `h <= 0` is refused inside `holtwinters_forecast_host_traced`, by name,
    # in their words ("h must be > 0. Currently: ...").
    var fc = holtwinters_forecast_host(fitted, h)
    for i in range(h * batch_size):
        out_ptr.unsafe_store(i, fc[i])
    _ = fitted^
    return h * batch_size


# =============================================================================
# THE IN-SAMPLE PREDICTION ON THE DEVICE (cpu-gpu-cleanup n-seq, 2026-10-02)
# =============================================================================
# `holtwinters/host/hw_predict.mojo::hw_predict_in_sample_ptr`'s arithmetic,
# one GPU thread per output cell (step k, series s): the same three loads,
# the same two flushed operations, the canonical quiet NaN where
# `t < 2 * frequency`. Each cell is independent, so the device and the host
# column (the CPU-only install's binding) write the same bits. The GPU
# binding registers `holtwinters_predict` from here; it used to call the host
# walk.

comptime HW_PREDICT_TPB = 256


def hw_predict_in_sample_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    comps: MutPointer[Float32, MutAnyOrigin],
    components_len_in: Int64,
    bs_in: Int64,
    f_in: Int64,
    start_in: Int64,
    ld_in: Int64,
    additive_in: Int32,
):
    var bs = Int(bs_in)
    var total = Int(ld_in) * bs
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= total:
        return
    var f = Int(f_in)
    var cl = Int(components_len_in)
    var k = idx // bs
    var s = idx - k * bs
    var t = Int(start_in) + k
    if t < 2 * f:
        dst.unsafe_store(idx, bitcast[DType.float32](UInt32(0x7FC00000)))
        return
    var i = t - f
    var leveltrend = ftz(
        comps.unsafe_load(s + (i - 1) * bs) + comps.unsafe_load(cl + s + (i - 1) * bs)
    )
    var stmp = comps.unsafe_load(2 * cl + s + (i - f) * bs)
    if additive_in != 0:
        dst.unsafe_store(idx, ftz(leveltrend + stmp))
    else:
        dst.unsafe_store(idx, ftz(leveltrend * stmp))


def holtwinters_predict_ptr(
    comps_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    batch_size: Int,
    frequency: Int,
    seasonal: String,
    start: Int,
    end: Int,
) raises -> Int:
    """The in-sample one-step predictions at times `[start, end)` on the
    device. Returns `(end - start) * batch_size`. `comps_ptr` is the packed
    components `holtwinters_fit_ptr` wrote; `out_ptr` is written TIME-MAJOR.
    The guards are the host entry's (`bindings/holtwinters_host_predict.mojo`),
    in its words and order."""
    var st = seasonal_from_name(seasonal)
    if n <= frequency:
        raise Error(
            "holtwinters predict: n (" + String(n) + ") must exceed frequency ("
            + String(frequency) + "); there would be no fitted components"
        )
    if batch_size < 1:
        raise Error(
            "holtwinters predict: batch_size must be >= 1 (batch_size="
            + String(batch_size) + ")"
        )
    if start < 0 or end <= start:
        raise Error(
            "holtwinters predict: need 0 <= start < end (start="
            + String(start) + ", end=" + String(end) + ")"
        )
    if end > n:
        raise Error(
            "holtwinters predict: the in-sample prediction ends at n (end="
            + String(end) + ", n=" + String(n) + "); later times are the forecast's"
        )
    var components_len = (n - frequency) * batch_size
    var ld = end - start
    var total = ld * batch_size
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var comps_d = ctx.enqueue_create_buffer[DType.float32](3 * components_len)
    var out_d = ctx.enqueue_create_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_buf=comps_d, src_ptr=comps_ptr)
    ctx.enqueue_function[hw_predict_in_sample_kernel](
        out_d.unsafe_ptr(), comps_d.unsafe_ptr(),
        Int64(components_len), Int64(batch_size), Int64(frequency),
        Int64(start), Int64(ld), Int32(1 if st == SEASONAL_ADDITIVE else 0),
        grid_dim=(total + HW_PREDICT_TPB - 1) // HW_PREDICT_TPB,
        block_dim=HW_PREDICT_TPB,
    )
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out_d)
    ctx.synchronize()
    _ = comps_d^
    _ = out_d^
    return total
