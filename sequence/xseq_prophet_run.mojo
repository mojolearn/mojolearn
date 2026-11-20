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


# ---------------------------------------------------------------- typed fit preparation
# Lane py-tail (2026-11-15): the fit's input preparation as one typed entry.
# Before, the door interleaved the day conversion, the refusals, the
# automatic seasonalities, the feature and changepoint glue calls and the
# prior/order vectors with Python list and array work. Now the door unpacks
# the model's settings and the t/y/holiday buffers into the records below,
# calls `prophet_run_prepare_fit` and `prophet_run_fit_plan`, and publishes
# the plan. The glue calls, their argument words and the refusal order are
# the door's (days, float64 y, y rank, lengths, finite y, changepoints, then
# the door's holiday admission, then the fit): same entries, same bits.

comptime PROPHET_AUTO_AUTO = 0      # the string 'auto'
comptime PROPHET_AUTO_ON = 1        # True
comptime PROPHET_AUTO_OFF = 2       # False
comptime PROPHET_AUTO_ORDER = 3     # an explicit Fourier order


@fieldwise_init
struct ProphetSeason(Copyable, Movable):
    """One seasonality record: name, period in days, Fourier order, prior scale."""
    var name: String
    var period: Float64
    var fourier_order: Int
    var prior_scale: Float64


@fieldwise_init
struct ProphetAuto(Copyable, Movable):
    """One automatic seasonality setting (`yearly_seasonality`, ...): `kind`
    is a PROPHET_AUTO_* word; `order` is read for PROPHET_AUTO_ORDER."""
    var kind: Int
    var order: Int


@fieldwise_init
struct ProphetTimes(Copyable, Movable):
    """A time column: `src` is a contiguous float64 (n,) buffer, or int64
    nanoseconds when `datetime`."""
    var src: Int
    var n: Int
    var datetime: Bool


@fieldwise_init
struct ProphetFitInput(Copyable, Movable):
    """The fit's typed inputs. `n` is the series length (0 when the rank is
    refused); `y_finite` is read only after the rank and length checks."""
    var t: ProphetTimes
    var y_float64: Bool
    var y_rank: Int
    var n: Int
    var y_finite: Bool
    var seasons: List[ProphetSeason]
    var yearly: ProphetAuto
    var weekly: ProphetAuto
    var daily: ProphetAuto
    var seasonality_prior_scale: Float64
    var has_user_changepoints: Bool
    var user_changepoints: ProphetTimes
    var changepoint_range: Float64
    var n_changepoints: Int


@fieldwise_init
struct ProphetFitPlan(Movable):
    """The prepared fit: every seasonality in publication order (the last
    `n_added` are the automatic additions), the days, start and scale, the
    feature buffers and the changepoint buffer (its first `s` entries are the
    plan)."""
    var seasons: List[ProphetSeason]
    var n_added: Int
    var days: List[Float64]
    var start: Float64
    var t_scale: Float64
    var t_scaled: List[Float32]
    var phases: List[Float32]
    var changepoints: List[Float32]
    var s: Int


def _prophet_days_into(entry: EntryRef, times: ProphetTimes, mut days: List[Float64], mut info: List[Float64]) raises:
    """`_days(t)` into native buffers; an empty `t` makes no call (info stays 0)."""
    days = List[Float64](length=times.n, fill=0.0)
    info = List[Float64](length=3, fill=0.0)
    if times.n:
        prophet_run_days(entry, times.src, Int(days.unsafe_ptr()), Int(info.unsafe_ptr()), times.n, times.datetime)
        _ = len(days)  # keep both buffers alive through the glue call above
        _ = len(info)


def _prophet_auto_season(mut seasons: List[ProphetSeason], name: String, setting: ProphetAuto, default_order: Int,
                         ok: Bool, prior_scale: Float64) -> Bool:
    """`_auto(name, arg, default_order, ok)`: True when it adds the season."""
    for i in range(len(seasons)):  # driver-loop: over the model's seasonalities (a name already present wins)
        if seasons[i].name == name:
            return False
    var use: Bool
    var order: Int
    if setting.kind == PROPHET_AUTO_AUTO:
        use = ok
        order = default_order
    elif setting.kind == PROPHET_AUTO_ON:
        use = True
        order = default_order
    elif setting.kind == PROPHET_AUTO_OFF:
        use = False
        order = 0
    else:
        use = True
        order = setting.order
    if use:
        seasons.append(ProphetSeason(name, prophet_run_auto_period(name), order, prior_scale))
    return use


def prophet_run_prepare_fit(entry: EntryRef, fit_input: ProphetFitInput) raises -> ProphetFitPlan:
    """Days, refusals, automatic seasonalities, features and changepoints of
    one fit, in the door's former order. The caller keeps the `t`, `y` and
    user changepoint buffers alive and has released the GIL."""
    var days = List[Float64]()
    var info = List[Float64]()
    _prophet_days_into(entry, fit_input.t, days, info)
    prophet_run_check_days(Int(info[0]))
    if fit_input.y_float64:
        raise Error("TypeError: " + _P + "float64 y is refused; pass float32")
    prophet_run_check_y_rank(fit_input.y_rank)
    prophet_run_check_lengths(fit_input.n, len(days))
    prophet_run_check_finite(fit_input.y_finite)
    var span = info[1]
    var dmin = info[2]
    var seasons = fit_input.seasons.copy()
    var added = 0
    if _prophet_auto_season(seasons, "yearly", fit_input.yearly, 10, span >= 730, fit_input.seasonality_prior_scale):
        added += 1
    if _prophet_auto_season(seasons, "weekly", fit_input.weekly, 3, span >= 14 and dmin < 7,
                            fit_input.seasonality_prior_scale):
        added += 1
    if _prophet_auto_season(seasons, "daily", fit_input.daily, 4, span >= 2 and dmin < 1, fit_input.seasonality_prior_scale):
        added += 1
    var start = days[0]
    var t_scale = prophet_run_scale(span)
    var ns = len(seasons)
    var periods = List[Float64](length=max(ns, 1), fill=1.0)
    for i in range(ns):  # driver-loop: over the seasonalities (one period each)
        periods[i] = seasons[i].period
    var t_scaled = List[Float32](length=fit_input.n, fill=0.0)
    var phases = List[Float32](length=fit_input.n * max(ns, 1), fill=0.0)
    prophet_run_features(entry, Int(days.unsafe_ptr()), Int(periods.unsafe_ptr()), Int(t_scaled.unsafe_ptr()),
                         Int(phases.unsafe_ptr()), fit_input.n, ns, start, t_scale)
    _ = len(periods)  # keep the periods alive through the glue call above
    var user = List[Float64](length=1, fill=0.0)
    var n_user = -1
    var hist = 0
    var n_cp = 0
    if fit_input.has_user_changepoints:
        var user_info = List[Float64]()
        _prophet_days_into(entry, fit_input.user_changepoints, user, user_info)
        n_user = len(user)
    else:
        var plan = prophet_run_changepoint_plan(fit_input.n, fit_input.changepoint_range, fit_input.n_changepoints)
        hist = plan[0]
        n_cp = plan[1]
    var changepoints = List[Float32](length=prophet_run_changepoint_count(n_user, n_cp), fill=0.0)
    var s = prophet_run_changepoints(entry, Int(days.unsafe_ptr()), Int(user.unsafe_ptr()),
                                     Int(changepoints.unsafe_ptr()), fit_input.n, n_user, hist, n_cp, start, t_scale)
    _ = len(user)  # keep the user changepoints alive through the glue call above
    prophet_run_check_changepoints(s)
    return ProphetFitPlan(seasons^, added, days^, start, t_scale, t_scaled^, phases^, changepoints^, s)


def prophet_run_prior_count(plan: ProphetFitPlan, nh: Int) -> Int:
    """K: two parameters per Fourier order of every seasonality, one per
    holiday column."""
    var k = nh
    for i in range(len(plan.seasons)):  # driver-loop: over the seasonalities
        k += max(2 * plan.seasons[i].fourier_order, 0)
    return k


def prophet_run_fit_plan(entry: EntryRef, plan: ProphetFitPlan, y: Int, batch: Int, holidays: Int, nh: Int,
                         holidays_prior_scale: Float64, params: Int, info: Int, multiplicative: Bool, max_iter: Int,
                         changepoint_prior_scale: Float64) raises:
    """The MAP fit of every series from a prepared plan: the Fourier orders
    (one 0 placeholder without seasonalities) and the prior scale of every
    Fourier parameter, then of every holiday column (one 1.0 placeholder when
    K is 0), rounded to float32 as the door's arrays were. `params` is
    (batch, 3 + s + K) and `info` (batch, 4), float32, allocated by the
    caller, which keeps `y`, `holidays`, `params` and `info` alive and has
    released the GIL."""
    var ns = len(plan.seasons)
    var orders = List[Float32](length=max(ns, 1), fill=0.0)
    var priors = List[Float32]()
    for i in range(ns):  # driver-loop: over the seasonalities
        orders[i] = Float32(plan.seasons[i].fourier_order)
        for _ in range(2 * plan.seasons[i].fourier_order):  # driver-loop: over one seasonality's Fourier parameters (model structure)
            priors.append(Float32(plan.seasons[i].prior_scale))
    for _ in range(nh):  # driver-loop: over the holiday columns (model structure)
        priors.append(Float32(holidays_prior_scale))
    var k = len(priors)
    if k == 0:
        priors.append(Float32(1.0))
    var request = ProphetFitRequest(
        y, Int(plan.t_scaled.unsafe_ptr()), Int(plan.phases.unsafe_ptr()), Int(orders.unsafe_ptr()), holidays,
        Int(plan.changepoints.unsafe_ptr()), Int(priors.unsafe_ptr()), params, info, batch, len(plan.days), ns, nh,
        k, plan.s, multiplicative, max_iter, changepoint_prior_scale)
    prophet_run_fit(entry, request)
    _ = len(orders)  # keep the orders and priors alive through the glue call above
    _ = len(priors)


def prophet_run_copy_changepoints(plan: ProphetFitPlan, dst: Int) raises:
    """`changepoints_t_`: the plan's first `s` changepoints into the caller's
    float32 (s,) buffer."""
    if plan.s <= 0:
        return
    if dst == 0:
        raise Error("ProphetForecaster: null changepoint buffer")
    var target = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst)
    for i in range(plan.s):  # driver-loop: over the changepoints (model structure, at most n_changepoints)
        target[i] = plan.changepoints[i]
