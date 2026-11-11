# SPDX-License-Identifier: Apache-2.0
"""RUN half of ProphetForecaster (`bindings/xseq_prophet_native.mojo` is the
UNPACK/PACK boundary; lane doorsplit-cnn-xseq, 2026-11-11;
docs/plans/PYTHON_AT_THE_BOUNDARY.md section 2.2).

Typed values only: the day, feature, changepoint, fit and predict glue calls
(`sequence/xseq_door_run.mojo` `xseq_run_glue`) with their argument words,
the changepoint plan arithmetic, the automatic seasonality periods, and the
model's refusals in its own texts. The boundary snapshots the model's
fields, admits and allocates every buffer (addresses arrive as words), keeps
them alive, releases the GIL around each glue call here, and publishes the
fitted state.
"""
from std.math import floor
from core.xseq_entry_table import (
    XS_PROPHET_DAYS, XS_PROPHET_FEATURES, XS_PROPHET_CHANGEPOINTS, XS_PROPHET_FIT, XS_PROPHET_PREDICT,
    XS_PROPHET_PREDICT_DAYS,
)
from core.entry_table import EntryRef
from sequence.xseq_door_run import xseq_run_glue

comptime _P = "ProphetForecaster: "


@fieldwise_init
struct ProphetFitRequest(Copyable, Movable):
    """The fit entry's buffers (addresses) and words."""
    var y: Int                  # float32 (B, N)
    var t_scaled: Int           # float32 (N,)
    var phases: Int             # float32 (N, max(ns, 1))
    var orders: Int             # float32 Fourier orders
    var holidays: Int           # float32 (N, max(nh, 1))
    var changepoints: Int       # float32 (S,)
    var priors: Int             # float32 prior scales (max(K, 1),)
    var params: Int             # float32 (B, 3 + S + K), written
    var info: Int               # float32 (B, 4), written
    var batch: Int
    var n: Int
    var n_seasons: Int
    var nh: Int
    var k: Int
    var s: Int
    var multiplicative: Bool
    var max_iter: Int
    var changepoint_prior_scale: Float64


@fieldwise_init
struct ProphetPredictRequest(Copyable, Movable):
    """The predict entries' buffers (addresses) and words. `fused` builds
    the features from the days on the executor (`days`, `periods`, `start`,
    `t_scale`); otherwise `t_scaled` and `phases` are the feature call's."""
    var fused: Bool
    var days: Int               # float64 (M,), fused
    var periods: Int            # float64 seasonal periods, fused
    var t_scaled: Int           # float32 (M,), not fused
    var phases: Int             # float32 (M, max(ns, 1)), not fused
    var params: Int
    var info: Int
    var orders: Int
    var holidays: Int
    var changepoints: Int
    var yhat: Int               # float32 (B, M), written
    var trend: Int              # float32 (B, M), written
    var batch: Int
    var m: Int
    var n_seasons: Int
    var nh: Int
    var k: Int
    var n_changepoints: Int
    var multiplicative: Bool
    var start: Float64
    var t_scale: Float64


# ---------------------------------------------------------------- days and features

def prophet_run_days(entry: EntryRef, src: Int, days: Int, info: Int, n: Int, datetime: Bool) raises:
    """`_days(t)`: days float64 (n,) and info [status, span, least spacing]
    from float64 values or datetime64 nanoseconds. The boundary skips the
    call for an empty `t`."""
    var addresses: List[Int] = [src, days, info]
    var words: List[Int] = [n, 1 if datetime else 0]
    _ = xseq_run_glue(entry, XS_PROPHET_DAYS, addresses, words, List[Float64]())


def prophet_run_check_days(status: Int) raises:
    """The day conversion's status word: 1 a NaN, 2 not ascending."""
    if status == 1:
        raise Error("ValueError: " + _P + "t must not hold NaN")
    if status == 2:
        raise Error("ValueError: " + _P + "t must be sorted ascending")


def prophet_run_features(entry: EntryRef, days: Int, periods: Int, t_scaled: Int, phases: Int, n: Int,
                         n_seasons: Int, start: Float64, t_scale: Float64) raises:
    """`_features(days)`: scaled times float32 (n,) and seasonal phases
    float32 (n, max(n_seasons, 1))."""
    var addresses: List[Int] = [days, periods, t_scaled, phases]
    var words: List[Int] = [n, n_seasons]
    var floats: List[Float64] = [start, t_scale]
    _ = xseq_run_glue(entry, XS_PROPHET_FEATURES, addresses, words, floats)


def prophet_run_auto_period(name: String) -> Float64:
    """The period of an automatic seasonality: yearly 365.25 days, weekly 7,
    daily 1."""
    return 365.25 if name == "yearly" else (7.0 if name == "weekly" else 1.0)


def prophet_run_scale(span: Float64) -> Float64:
    """`t_scale_`: the day span, or 1 for a single instant."""
    return span if span > 0 else 1.0


# ---------------------------------------------------------------- fit

def prophet_run_check_y_rank(ndim: Int) raises:
    if ndim != 1 and ndim != 2:
        raise Error("ValueError: ProphetForecaster: y must be a series or a batch of series")


def prophet_run_check_lengths(n: Int, n_days: Int) raises:
    if n != n_days or n < 2:
        raise Error("ValueError: " + _P + "y and t lengths differ, or fewer than 2 points")


def prophet_run_check_finite(finite: Bool) raises:
    if not finite:
        raise Error("ValueError: " + _P + "y must be finite")


def prophet_run_changepoint_plan(n: Int, changepoint_range: Float64, n_changepoints: Int) -> List[Int]:
    """[hist, n_cp] of automatic changepoints: floor(n * changepoint_range)
    (one IEEE product, floored exactly) history points, and at most one less
    than that many changepoints."""
    var hist = Int(floor(Float64(n) * changepoint_range))
    var n_cp = max(min(n_changepoints, hist - 1), 0)
    return [hist, n_cp]


def prophet_run_changepoint_count(n_user: Int, n_cp: Int) -> Int:
    """Words of the changepoint buffer: the user's count, else the planned
    one, at least one."""
    return max(n_user, max(n_cp, 1))


def prophet_run_changepoints(entry: EntryRef, days: Int, user: Int, changepoints: Int, n: Int, n_user: Int,
                             hist: Int, n_cp: Int, start: Float64, t_scale: Float64) raises -> Int:
    """The changepoint plan (scaled float32 times); `n_user` -1 when the
    model has no user changepoints. The answer is their count, negative when
    a user changepoint falls outside the training data."""
    # cpu-route: changepoint plan over a handful of changepoints
    var addresses: List[Int] = [days, user, changepoints]
    var words: List[Int] = [n, n_user, hist, n_cp]
    var floats: List[Float64] = [start, t_scale]
    return xseq_run_glue(entry, XS_PROPHET_CHANGEPOINTS, addresses, words, floats)


def prophet_run_check_changepoints(count: Int) raises:
    if count < 0:
        raise Error("ValueError: Changepoints must fall within training data.")


def prophet_run_parameter_count(n_changepoints: Int, k: Int) -> Int:
    """Parameters per series: growth, offset, noise, one per changepoint,
    one per seasonal and holiday column."""
    return 3 + n_changepoints + k


def prophet_run_fit(entry: EntryRef, request: ProphetFitRequest) raises:
    """The MAP fit of every series (`XS_PROPHET_FIT`)."""
    var addresses: List[Int] = [request.y, request.t_scaled, request.phases, request.orders, request.holidays,
                                request.changepoints, request.priors, request.params, request.info]
    var words: List[Int] = [request.batch, request.n, request.n_seasons, request.nh, request.k, request.s,
                            1 if request.multiplicative else 0, request.max_iter]
    var floats: List[Float64] = [request.changepoint_prior_scale]
    _ = xseq_run_glue(entry, XS_PROPHET_FIT, addresses, words, floats)


# ---------------------------------------------------------------- predict

def prophet_run_check_stored_dtype(dtype_number: Int) raises:
    """A fitted buffer's dtype word: 11 is float32."""
    if dtype_number != 11:
        raise Error("TypeError: ProphetForecaster fitted buffers require float32 storage")


def prophet_run_check_ranks(params_ndim: Int, info_ndim: Int, changepoints_ndim: Int) raises:
    if params_ndim != 2 or info_ndim != 2 or changepoints_ndim != 1:
        raise Error("ValueError: ProphetForecaster fitted buffer ranks disagree")


def prophet_run_check_holidays_given(nh: Int, given: Bool) raises:
    if nh != 0 and not given:
        raise Error("ValueError: " + _P + "the model has holiday columns; pass them for t")


def prophet_run_check_order(order: Int) raises:
    if order < 0:
        raise Error("ValueError: ProphetForecaster Fourier orders must be nonnegative")


def prophet_run_check_schema(nh: Int, k: Int, feature_count: Int) raises:
    """The fitted feature schema: 2 * order columns per seasonality plus the
    holiday columns make the fitted K. `k` is not read when `nh` is
    negative."""
    if nh < 0 or k < 0 or feature_count + nh != k:
        raise Error("ValueError: ProphetForecaster seasonalities changed after fit; fit the model again")


def prophet_run_check_info_shape(rows: Int, cols: Int, batch: Int) raises:
    if rows != batch or cols != 4:
        raise Error("ValueError: ProphetForecaster fitted diagnostics shape disagrees")


def prophet_run_check_params_shape(cols: Int, n_changepoints: Int, k: Int) raises:
    if cols != prophet_run_parameter_count(n_changepoints, k):
        raise Error("ValueError: ProphetForecaster fitted parameter shape disagrees")


def prophet_run_check_holiday_columns(nh: Int, columns: Int) raises:
    if nh != 0 and columns != nh:
        raise Error("ValueError: ProphetForecaster holiday column count differs from fit")


def prophet_run_predict(entry: EntryRef, request: ProphetPredictRequest) raises:
    """The forecast and trend of every series: `XS_PROPHET_PREDICT_DAYS`
    (features built on the executor) when fused, `XS_PROPHET_PREDICT` from
    the feature call's buffers otherwise."""
    var words: List[Int] = [request.batch, request.m, request.n_seasons, request.nh, request.k,
                            request.n_changepoints, 1 if request.multiplicative else 0]
    if request.fused:
        var fused_addresses: List[Int] = [request.days, request.periods, request.params, request.info,
                                          request.orders, request.holidays, request.changepoints, request.yhat,
                                          request.trend]
        var floats: List[Float64] = [request.start, request.t_scale]
        _ = xseq_run_glue(entry, XS_PROPHET_PREDICT_DAYS, fused_addresses, words, floats)
    else:
        var addresses: List[Int] = [request.params, request.info, request.t_scaled, request.phases, request.orders,
                                    request.holidays, request.changepoints, request.yhat, request.trend]
        _ = xseq_run_glue(entry, XS_PROPHET_PREDICT, addresses, words, List[Float64]())
