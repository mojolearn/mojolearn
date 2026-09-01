# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surface for the batched ARIMA lane, what `bindings/` calls.

Shaped like `tsa/estimator.mojo` and `svm/estimator.mojo`, raw host pointers
in and out, one `DeviceContext` per call destroyed with it, no pointer
retained past the call. The binding is `bindings/_mojolearn_arima.mojo` and
the wrapper is `python/mojolearn/_arima_impl.py`.

THREE ENTRY POINTS, AND THEY ARE THE WHOLE PUBLIC DOOR

    arima_fit_ptr_host       `ARIMA.fit` (`arima.pyx:860-958`) with
                             `method = "ml"`, `start_params = None`,
                             `simple_differencing = True`, no exog. Runs
                             `estimate_x0`, the inverse Jones transform,
                             the own-written batched L-BFGS, the forward
                             transform and the unpack, then evaluates the
                             log-likelihood once more AT the fitted point
                             so the caller has a number the optimizer's
                             rescaled objective does not give it.
    arima_predict_ptr_host   `ARIMA.predict(start, end)` (`arima.pyx:615`)
                             with `level = None` and no exog.
    arima_forecast_ptr_host  `ARIMA.forecast(nsteps)` (`arima.pyx:770`),
                             which upstream is `predict(n_obs, n_obs +
                             nsteps)` and is that here too.

EVERY OUTPUT SIZE IS A FUNCTION OF `(batch_size, n_obs, order, n_steps)`
ALONE, so the Python caller allocates before it calls and nothing here has
to hand a length back through a second call. Written out because the
question was asked of this design directly and the answer is not obviously
yes for a fit.

    fit       params  N * batch_size,  N = p + q + P + Q + k + 1
              x       N * batch_size
              x0      N * batch_size
              stats   2 * batch_size   float32
              flags   2 * batch_size   int32
    predict   out     (end - start) * batch_size
    forecast  out     n_steps * batch_size

`N` is `ARIMAOrder::complexity()` with `n_exog = 0`, and `n_exog` is refused
by name one line into every entry, so the Python side computes the same `N`
from the order tuple it was constructed with. Nothing here is sized by an
answer. Contrast `svm/estimator.mojo`, where `n_support` is not known until
the solve finishes and the caller has to allocate the worst case; ARIMA has
no such quantity.

DEVIATION 990: THE FITTED MODEL CROSSES BACK TO THE HOST AND IS UPLOADED
AGAIN AT EVERY PREDICT. The ARIMA sibling of DEVIATION 873. This boundary
retains no device pointer, so `fit` writes the packed transformed parameter
vector into the caller's array and `predict` and `forecast` take that array
back and `unpack` it onto a fresh device. Upstream keeps `ARIMAParams` on
the device between calls. Nothing numeric changes, because the packed vector
is float32 on both sides and the round trip is a copy; what changes is one
upload of `N * batch_size` floats per predict, which is small beside the
Kalman pass that follows it.

DEVIATION 992: `method` CROSSES AS AN INTEGER AND IS REFUSED HERE, NOT IN
PYTHON. `conditional_sum_of_squares` and `sum_of_squares_kernel`
(`batched_arima.cu:271-391`) are NOT PORTED and `arima/NOT_IMPLEMENTED.tsv`
says so, so CSS and CSS-ML have to be refused somewhere. Refusing them in
the Python wrapper would put the only copy of that policy where no Mojo gate
can see it and would make the refusal unreachable from any other caller of
this file. `_refuse_method` below is the refusal, it runs before the
`DeviceContext` exists, and the wrapper passes the code through untouched.

WHAT THIS SURFACE DOES NOT ADD. Every other refusal already exists one layer
down and is raised there by name. `arima/impl/tsa/arima_common.mojo::
validate_order` refuses exog (`n_exog != 0`), `rd > 8`, `r > 5`, `p, q, P, Q
> 8`, `d + D > 2`, a seasonal order with `s < 2` and an order with no
parameters at all; `arima/impl/arima/batched_arima.mojo::_refuse_non_finite`
refuses a non-finite series with its flat index; `predict` refuses `start <
0` and `end <= start`. `validate_order` is called HERE as well, before any
device work, so a refused parameter costs nothing, and it is called again
underneath, which keeps the guarantee independent of this file.

CROSS-VENDOR STATUS. The lane behind this surface is bit-identical on three
vendors at `221aa141`, 139 card records (`arima/arima_main.mojo`'s header
carries the evidence). THAT CARD IS THE KALMAN FILTER AND ITS STAGES, NOT
THIS FILE. The fit landed after it and the card was re-emitted byte
identical, which says the fit moved no stage the card records; it does not
say `arima_fit_ptr_host` has been run on a second vendor, because it has
not. A three-vendor run THROUGH this door is OWED.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace

from arima.impl.arima.batched_arima import batched_diff, batched_loglike, predict
from arima.impl.arima.batched_fit import batched_fit
from arima.impl.tsa.arima_common import (
    ARIMAOrder,
    ARIMAParams,
    unpack,
    validate_order,
)


# ---------------------------------------------------------------------------
# the log-likelihood method, as an integer (DEVIATION 992)
# ---------------------------------------------------------------------------
# cuML's `LoglikeMethod` is `{MLE, CSS}` and `arima.pyx:944` accepts the
# three strings "ml", "css" and "css-ml". These are the codes the Python
# wrapper sends, in the same order that file lists them.

comptime ARIMA_METHOD_MLE = 0
comptime ARIMA_METHOD_CSS = 1
comptime ARIMA_METHOD_CSS_ML = 2


def _refuse_method(method: Int) raises:
    """Only MLE is offered, and the other two are refused BY NAME.

    Not by clamping and not by a warning. `arima.pyx:947-950` DOES silently
    force `method` to "ml" when the series has missing values, and that arm
    is not copied: it is a downgrade the caller cannot see, and the missing
    observation path it belongs to is itself unported."""
    if method == ARIMA_METHOD_MLE:
        return
    if method == ARIMA_METHOD_CSS:
        raise Error(
            "ARIMA: method='css' is not ported; the conditional sum of"
            " squares log-likelihood (batched_arima.cu:271-391,"
            " conditional_sum_of_squares and sum_of_squares_kernel) and its"
            " `truncate` parameter have no port. Only MLE is offered;"
            " refused by name (arima/NOT_IMPLEMENTED.tsv)"
        )
    if method == ARIMA_METHOD_CSS_ML:
        raise Error(
            "ARIMA: method='css-ml' is not ported; it starts the maximum"
            " likelihood fit from the CSS optimum, and the CSS"
            " log-likelihood (batched_arima.cu:271-391) has no port. Only"
            " MLE is offered; refused by name (arima/NOT_IMPLEMENTED.tsv)"
        )
    raise Error(
        "ARIMA: method code " + String(method) + " is not one of MLE ("
        + String(ARIMA_METHOD_MLE) + "), CSS (" + String(ARIMA_METHOD_CSS)
        + ") or CSS-ML (" + String(ARIMA_METHOD_CSS_ML) + ")"
    )


# ---------------------------------------------------------------------------
# the plumbing this layer owns
# ---------------------------------------------------------------------------


def _order(
    p: Int, d: Int, q: Int, P: Int, D: Int, Q: Int, s: Int, k: Int, n_exog: Int
) raises -> ARIMAOrder:
    """The nine integers as an `ARIMAOrder`, validated before anything opens
    a device.

    `validate_order` is where `n_exog != 0` is refused, and that refusal is
    the one this whole surface most depends on: `ARIMAParams` carries no
    `beta` anywhere in the lane, so an accepted exog would not be a wrong
    answer, it would be a read of memory nobody wrote."""
    var order = ARIMAOrder(p, d, q, P, D, Q, s, k, n_exog)
    validate_order(order)
    return order


def _refuse_shape(batch_size: Int, n_obs: Int, who: String) raises:
    """The two things only this layer can see. A null address is refused in
    `bindings/_mojolearn_arima.mojo` before it reaches here."""
    if batch_size < 1:
        raise Error(
            who + ": batch_size must be >= 1 (batch_size=" + String(batch_size) + ")"
        )
    if n_obs < 2:
        raise Error(
            who + ": n_obs must be at least 2 (n_obs=" + String(n_obs) + ")"
        )


def _upload_f32(
    ctx: DeviceContext, ptr: MutPointer[Float32, MutUntrackedOrigin], n: Int
) raises -> DeviceBuffer[DType.float32]:
    """Copy `n` floats from the caller's memory onto the device.

    The host staging buffer is kept live past the `synchronize` on purpose.
    Mojo frees a buffer at its LAST USE and `.unsafe_ptr()` is a use, so
    without the trailing `_ = host^` the staging allocation can be gone
    before the copy it feeds has run (mojo-buffer-freed-at-last-use). Copied
    from `tsa/estimator.mojo`, which learned it the same way."""
    var count = n if n > 0 else 1
    var buf = ctx.enqueue_create_buffer[DType.float32](count)
    var host = ctx.enqueue_create_host_buffer[DType.float32](count)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, ptr.unsafe_load(i))
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _download_f32(
    ctx: DeviceContext,
    buf: DeviceBuffer[DType.float32],
    ptr: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
) raises:
    """Copy the whole of `buf` back into the caller's memory. `buf` must be
    exactly `n` floats long; every buffer handed here is allocated at that
    length by the routine that produced it."""
    if n <= 0:
        return
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    for i in range(n):
        ptr.unsafe_store(i, host.unsafe_ptr().unsafe_load(i))
    _ = host^


def _write_list_f32(
    ptr: MutPointer[Float32, MutUntrackedOrigin], values: List[Float32], offset: Int
) raises:
    for i in range(len(values)):
        ptr.unsafe_store(offset + i, values[i])


def _write_list_i32(
    ptr: MutPointer[Int32, MutUntrackedOrigin], values: List[Int32], offset: Int
) raises:
    for i in range(len(values)):
        ptr.unsafe_store(offset + i, values[i])


def _loglike_at(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    mut params: ARIMAParams,
) raises -> List[Float32]:
    """The log-likelihood of `params` on `y`, one value per series.

    THE OPTIMIZER'S `fx` IS NOT THIS NUMBER. `eval_batch` minimizes
    `-loglike / (n_obs - 1)`, so recovering the log-likelihood from it costs
    a float32 negate and a float32 multiply whose rounding nobody has
    measured. One more Kalman pass at the point the fit returned is cheaper
    to reason about than that arithmetic, and it is what cuML's own
    `information_criterion` does (`batched_arima.cu:592-618` calls
    `batched_loglike` again with `trans = false` rather than reusing
    anything from the fit).

    `trans = false`, because `params` are the FITTED parameters and are
    already forward transformed, which is the same argument `predict` makes
    at `batched_arima.cu:175`. `check_finite = false`, because
    `batched_fit` already refused a non-finite series once and nothing has
    written the buffer since."""
    if order.need_diff():
        var n_kf = n_obs - order.n_diff()
        var y_kf = ctx.enqueue_create_buffer[DType.float32](n_kf * batch_size)
        batched_diff(ctx, y_kf, y, batch_size, n_obs, order)
        ctx.synchronize()
        var lld = batched_loglike(
            ctx, y_kf, batch_size, n_kf, order.without_diff(), params,
            False, 0, 32, False,
        )
        var got = lld.loglike.copy()
        _ = lld^
        _ = y_kf^
        return got^
    var ll = batched_loglike(
        ctx, y, batch_size, n_obs, order, params, False, 0, 32, False
    )
    var out = ll.loglike.copy()
    _ = ll^
    return out^


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def arima_fit_ptr_host(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    params_ptr: MutPointer[Float32, MutUntrackedOrigin],
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    x0_ptr: MutPointer[Float32, MutUntrackedOrigin],
    stats_ptr: MutPointer[Float32, MutUntrackedOrigin],
    flags_ptr: MutPointer[Int32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    p: Int,
    d: Int,
    q: Int,
    P: Int,
    D: Int,
    Q: Int,
    s: Int,
    k: Int,
    n_exog: Int,
    method: Int,
    max_iterations: Int,
) raises -> Int:
    """`ARIMA(order, seasonal_order, fit_intercept=k).fit(y)`, one shot.
    Returns `N * batch_size`, the number of float32 written to
    `params_ptr`, with `N = p + q + P + Q + k + 1`.

    `y_ptr` reads `batch_size * n_obs` float32 with each SERIES CONTIGUOUS,
    series `b` at `[b * n_obs, (b + 1) * n_obs)`. That is the layout every
    kernel in the lane indexes (`matrix.cuh:74-76`) and it is what cuML's
    `(n_obs, batch_size)` Fortran-order array is in flat bytes. The Python
    wrapper takes `(batch_size, n_obs)` and so hands this over with no
    transpose; it says so, and says why it differs from cuML's Python shape.

    `params_ptr` is written with `N * batch_size` float32, THE FITTED MODEL,
    packed per series in this exact order (the same words in
    `bindings/_mojolearn_arima.mojo` and `python/mojolearn/_arima_impl.py`,
    and it is `ARIMAParams::pack`'s order at
    `arima/impl/tsa/arima_common.mojo`):

        series b occupies [b * N, (b + 1) * N)
            mu       k values     (absent when k == 0)
            ar       p values
            ma       q values
            sar      P values
            sma      Q values
            sigma2   1 value

    These are FORWARD TRANSFORMED, which is what `predict` must be handed,
    and they are the array `arima_predict_ptr_host` and
    `arima_forecast_ptr_host` take back (DEVIATION 990).

    `x_ptr` is written with `N * batch_size` float32, the UNCONSTRAINED
    optimum in the coordinates the optimizer works in, same packing.
    `x0_ptr` is written with `N * batch_size` float32, the starting point
    `estimate_x0` produced, same packing. Neither is on cuML's Python
    surface. They are here because a fit that goes wrong is nearly always a
    fit that started wrong, and `estimate_x0` is the half of this lane with
    no upstream oracle.

    `stats_ptr` is written with `2 * batch_size` float32, in this exact
    order:

        [0 * batch_size, 1 * batch_size)   loglike  (at the fitted point)
        [1 * batch_size, 2 * batch_size)   fx       (the objective the
                                                     optimizer minimized,
                                                     -loglike / (n_obs - 1))

    `flags_ptr` is written with `2 * batch_size` int32, in this exact order:

        [0 * batch_size, 1 * batch_size)   n_iter
        [1 * batch_size, 2 * batch_size)   retcode  (0 is OPT_SUCCESS)

    THERE IS NO AIC OR BIC HERE. `information_criterion`
    (`batched_arima.cu:592-618`) and its AIC / AICc / BIC arms are NOT
    PORTED (`arima/NOT_IMPLEMENTED.tsv`). What that routine does beyond the
    log-likelihood is one `raft::stats::information_criterion_batched`
    unary op, `ic_base - 2 * loglike`, and the Python wrapper computes that
    on the host in float64 and says so on the class (DEVIATION 991). A
    device kernel here would be a kernel with no gate.
    """
    _refuse_method(method)
    var order = _order(p, d, q, P, D, Q, s, k, n_exog)
    _refuse_shape(batch_size, n_obs, "arima_fit")
    if max_iterations < 1:
        raise Error(
            "arima_fit: max_iterations must be >= 1 (max_iterations="
            + String(max_iterations) + ")"
        )
    var N = order.complexity()

    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "arima_fit_ptr_host: batch_size=" + String(batch_size)
        + " n_obs=" + String(n_obs)
        + " order=(" + String(p) + "," + String(d) + "," + String(q) + ")"
        + " seasonal=(" + String(P) + "," + String(D) + "," + String(Q)
        + "," + String(s) + ")"
        + " k=" + String(k) + " max_iterations=" + String(max_iterations)
    )
    var y = _upload_f32(ctx, y_ptr, batch_size * n_obs)
    var params = ARIMAParams(ctx, order, batch_size)
    var r = batched_fit(
        ctx, y, batch_size, n_obs, order, params, trace, max_iterations
    )
    var loglike = _loglike_at(ctx, y, batch_size, n_obs, order, params)

    _write_list_f32(params_ptr, r.t_x, 0)
    _write_list_f32(x_ptr, r.x, 0)
    _write_list_f32(x0_ptr, r.x0, 0)
    _write_list_f32(stats_ptr, loglike, 0)
    _write_list_f32(stats_ptr, r.fx, batch_size)
    _write_list_i32(flags_ptr, r.n_iter, 0)
    _write_list_i32(flags_ptr, r.retcode, batch_size)

    _ = r^
    _ = params^
    _ = y^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return N * batch_size


# ---------------------------------------------------------------------------
# predict and forecast
# ---------------------------------------------------------------------------


def _predict_into(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    params_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    start: Int,
    end: Int,
    order: ARIMAOrder,
    who: String,
) raises -> Int:
    """The body both public prediction entries share, so `forecast` cannot
    drift from `predict` by an edit to one of them."""
    _refuse_shape(batch_size, n_obs, who)
    if start < 0 or end <= start:
        raise Error(
            who + ": need 0 <= start < end (start=" + String(start)
            + ", end=" + String(end) + ")"
        )
    if start > n_obs:
        raise Error(
            who + ": there can't be a gap between the data and the"
            " prediction (start=" + String(start) + ", n_obs="
            + String(n_obs) + ")"
        )
    var N = order.complexity()
    var predict_ld = end - start

    var ctx = DeviceContext()
    var y = _upload_f32(ctx, y_ptr, batch_size * n_obs)
    var xin = _upload_f32(ctx, params_ptr, N * batch_size)
    var params = ARIMAParams(ctx, order, batch_size)
    unpack(ctx, params, order, batch_size, xin)
    ctx.synchronize()
    # `pre_diff = true` is cuML's `simple_differencing = True`, the arm this
    # lane carries and the arm the fit optimized in. Predictions before
    # `d + s*D` are undefined under it and come back as the canonical NaN
    # (DEVIATION 676), which is `arima.pyx:672-674`'s warning made into a
    # value.
    var res = predict(
        ctx, y, batch_size, n_obs, start, end, order, params, True
    )
    _download_f32(ctx, res.y_p, out_ptr, predict_ld * batch_size)

    _ = res^
    _ = params^
    _ = xin^
    _ = y^
    _ = ctx^
    return predict_ld * batch_size


def arima_predict_ptr_host(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    params_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    start: Int,
    end: Int,
    p: Int,
    d: Int,
    q: Int,
    P: Int,
    D: Int,
    Q: Int,
    s: Int,
    k: Int,
    n_exog: Int,
) raises -> Int:
    """`ARIMA.predict(start, end)` with `level = None` and no exog
    (`arima.pyx:615-766`). Returns `(end - start) * batch_size`.

    `end` IS EXCLUDED, which is cuML's convention and is stated in their own
    docstring ("Index where to end the predictions, excluded"). It is NOT
    statsmodels', where `end` is the last index returned. The Python wrapper
    repeats this in three places because it is the one thing on that surface
    a caller can silently get wrong by one.

    `y_ptr` reads `batch_size * n_obs` float32, series contiguous.
    `params_ptr` reads `N * batch_size` float32 in `arima_fit_ptr_host`'s
    packed order, the FITTED, already transformed parameters, so
    `batched_loglike` runs with `trans = false` exactly as
    `batched_arima.cu:175` does.

    `out_ptr` is written with `(end - start) * batch_size` float32, SERIES
    MAJOR: series `b` at step `i` is `[b * (end - start) + i]`. THAT IS A
    TRANSPOSE OF cuML's OUTPUT, whose `y_p` is `(end - start, batch_size)`;
    it matches this lane's own kernel (`in_sample_prediction_kernel` writes
    `d_y_p[bid * ld + i]`) and it matches the input layout, so a caller
    holding `(batch_size, n_obs)` gets `(batch_size, end - start)` back and
    never transposes anything.
    """
    var order = _order(p, d, q, P, D, Q, s, k, n_exog)
    return _predict_into(
        y_ptr, params_ptr, out_ptr, batch_size, n_obs, start, end, order,
        "arima_predict",
    )


def arima_forecast_ptr_host(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    params_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    n_steps: Int,
    p: Int,
    d: Int,
    q: Int,
    P: Int,
    D: Int,
    Q: Int,
    s: Int,
    k: Int,
    n_exog: Int,
) raises -> Int:
    """`ARIMA.forecast(nsteps)` (`arima.pyx:770-838`), which upstream is
    literally `self.predict(self.n_obs, self.n_obs + nsteps, ...)` and is
    that here too. Returns `n_steps * batch_size`.

    `out_ptr` is written with `n_steps * batch_size` float32, series major,
    series `b` at step `i` at `[b * n_steps + i]`. There is no in-sample
    block in it: `start == n_obs`, so `predict`'s in-sample kernel does not
    launch at all and every value comes from the Kalman forecast, undiffed
    by `finalize_forecast` when `d + D > 0`.
    """
    if n_steps < 1:
        raise Error(
            "arima_forecast: n_steps must be >= 1 (n_steps="
            + String(n_steps) + ")"
        )
    var order = _order(p, d, q, P, D, Q, s, k, n_exog)
    return _predict_into(
        y_ptr, params_ptr, out_ptr, batch_size, n_obs, n_obs, n_obs + n_steps,
        order, "arima_forecast",
    )
