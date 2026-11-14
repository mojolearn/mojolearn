# SPDX-License-Identifier: Apache-2.0
"""Holt-Winters `initialization_method`: the estimated initial states
(ported from python/mojolearn/tests/test_holtwinters_initialization.py
`test_estimated_is_the_default_and_refits_to_the_same_bits` and
`test_estimated_forecasts_at_least_as_well_on_a_clean_series`).

Every fit is the host fit body the reference training context runs
(core/mojolearn_tsa_host_impl.mojo::holtwinters_fit_host_into, start_periods
2, eps 2.24e-3, the defaults) and every forecast the host forecast from the
packed state (holtwinters/host/hw_predict.mojo::hw_forecast_from_state_ptr).

REFIT units (additive and multiplicative, the original 2 x 96 series of
`_series(np, 96, 2, 12)`, period 12): the estimated method fitted twice
gives the same words for the forecast(24), level, trend, season, sse,
alpha, beta, gamma, n_iter and criterion (the original's sha256 digest of
exactly those bytes), the heuristic method gives different words; for
both methods the level is (2, 96 - 12), forecast(12) is finite and alpha,
beta and gamma lie in [0, 1]; the estimated criterion is 0, 1 or 2.

CLEAN units (a noiseless trend + season, 72 + 12 points, period 12, one
series): the estimated fit forecasts the 12 held-out points with an RMSE
at most the heuristic fit's times 1.001 plus 1e-6, and below 0.05 times
the standard deviation of all 84 points.

The default-method surface, the method codes, the saved method, and the
0.8.13 digest pins stay in the Python file. Both series are closed forms
evaluated in Float64 and rounded to Float32 here. Importable; no `main()`.
"""
from std.math import cos, isfinite, pi, sin, sqrt
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, i32_ptr
from core.mojolearn_tsa_host_impl import holtwinters_fit_host_into
from holtwinters.host.hw_predict import hw_forecast_from_state_ptr
from holtwinters.impl.internal.hw_estimate import HW_INIT_ESTIMATED, HW_INIT_HEURISTIC

#: REFIT: 2 seasonal types x (estimated, estimated again, heuristic)
comptime HW_REFIT_UNITS = 6
#: CLEAN: 2 seasonal types x (estimated, heuristic)
comptime HW_CLEAN_UNITS = 4
comptime HW_INIT_UNITS = HW_REFIT_UNITS + HW_CLEAN_UNITS
#: words per unit record: components, scalars, flag bits, forecast(24), forecast(12)
comptime HW_RECORD = 640

comptime REFIT_N = 96
comptime REFIT_B = 2
comptime CLEAN_N = 72
comptime CLEAN_H = 12
comptime HW_PERIOD = 12


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def unit_additive(unit: Int) -> Bool:
    if unit < HW_REFIT_UNITS:
        return unit // 3 == 0
    return (unit - HW_REFIT_UNITS) // 2 == 0


def unit_method(unit: Int) -> Int:
    if unit < HW_REFIT_UNITS:
        return HW_INIT_HEURISTIC if unit % 3 == 2 else HW_INIT_ESTIMATED
    return HW_INIT_HEURISTIC if (unit - HW_REFIT_UNITS) % 2 == 1 else HW_INIT_ESTIMATED


def refit_series() -> List[Float32]:
    """Series-major `REFIT_B x REFIT_N`, the original `_series(np, 96, 2, 12)`."""
    var count = REFIT_B * REFIT_N
    var result = List[Float32](unsafe_uninit_length=count)
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm rp}:
        var s = Float64(index // REFIT_N)
        var t = Float64(index % REFIT_N)
        var value = (20.0 + (s + 1.0) * 3.0 * sin(2.0 * pi * t / Float64(HW_PERIOD) + s) + 0.05 * t
                     + 0.5 * cos(0.37 * t * (s + 1.0)))
        rp.unsafe_store(index, Float32(value))
    host_parallelize(cell, count)
    return result^


def clean_series(additive: Bool) -> List[Float32]:
    """The original noiseless trend + season over `CLEAN_N + CLEAN_H` points."""
    var count = CLEAN_N + CLEAN_H
    var result = List[Float32](unsafe_uninit_length=count)
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm rp, imm additive}:
        var t = Float64(index)
        var f = Float64(HW_PERIOD)
        var season = 3.0 * sin(2.0 * pi * t / f) + cos(4.0 * pi * t / f)
        var level = 40.0 + 0.3 * t
        var value = level + season if additive else level * (1.0 + 0.05 * season)
        rp.unsafe_store(index, Float32(value))
    host_parallelize(cell, count)
    return result^


def fit_unit(unit: Int, record: MutPointer[Float32, MutUntrackedOrigin]) raises -> Int:
    """Fit one unit and write its record; returns its component plane."""
    var additive = unit_additive(unit)
    var refit = unit < HW_REFIT_UNITS
    var batch = REFIT_B if refit else 1
    var n = REFIT_N if refit else CLEAN_N
    var y = refit_series() if refit else clean_series(additive)
    var name = String("additive") if additive else String("multiplicative")
    var plane = (n - HW_PERIOD) * batch
    var components = List[Float32](length=3 * plane, fill=Float32(0))
    var scalars = List[Float32](length=4 * batch, fill=Float32(0))
    var flags = List[Int32](length=2 * batch, fill=Int32(0))
    var cp = f32_ptr(Int(components.unsafe_ptr()))
    var written = holtwinters_fit_host_into(Int(y.unsafe_ptr()), n, batch, HW_PERIOD, 2, Float32(2.24e-3),
        unit_method(unit), name, cp, f32_ptr(Int(scalars.unsafe_ptr())), i32_ptr(Int(flags.unsafe_ptr())))
    require(written == plane, name + ": the level is not (batch, n - period), unit " + String(unit))
    var long = hw_forecast_from_state_ptr(cp, plane, n, batch, HW_PERIOD, additive, 24)
    var short = hw_forecast_from_state_ptr(cp, plane, n, batch, HW_PERIOD, additive, 12)
    var at = 0
    for i in range(3 * plane):  # small-loop(one unit's record, inside its own task)
        record[at] = components[i]
        at += 1
    for i in range(4 * batch):  # small-loop(one unit's record)
        record[at] = scalars[i]
        at += 1
    for i in range(2 * batch):  # small-loop(one unit's record)
        record[at] = bitcast[DType.float32](flags[i])
        at += 1
    for i in range(len(long)):  # small-loop(one unit's record)
        record[at] = long[i]
        at += 1
    for i in range(len(short)):  # small-loop(one unit's record)
        record[at] = short[i]
        at += 1
    require(at <= HW_RECORD, "Holt-Winters unit record overflow")
    if refit:
        for i in range(len(short)):  # small-loop(forecast(12) of one unit)
            require(isfinite(short[i]), name + ": forecast(12) is not finite")
        for s in range(batch):  # small-loop(series of one unit)
            for k in range(1, 4):  # small-loop(alpha, beta, gamma)
                var v = scalars[k * batch + s]
                require(v >= 0 and v <= 1, name + ": a smoothing parameter is outside [0, 1]")
        if unit_method(unit) == HW_INIT_ESTIMATED:
            for s in range(batch):  # small-loop(series of one unit)
                var c = flags[batch + s]
                require(c == 0 or c == 1 or c == 2, name + ": criterion_ is not 0, 1 or 2")
    _ = y^
    _ = components^
    _ = scalars^
    _ = flags^
    return plane


def clean_rmse(record: MutPointer[Float32, MutUntrackedOrigin], plane: Int, y: List[Float32]) -> Float64:
    """RMSE of the forecast(12) words of a CLEAN record against y[n:]."""
    var start = 3 * plane + 4 + 2 + 24  # one series: 4 scalars, 2 flags, forecast(24)
    var total = Float64(0)
    for i in range(CLEAN_H):  # small-loop(12 held-out points)
        var d = Float64(record[start + i]) - Float64(y[CLEAN_N + i])
        total += d * d
    return sqrt(total / Float64(CLEAN_H))


def series_std(y: List[Float32]) -> Float64:
    var mean = Float64(0)
    for v in y:  # small-loop(84 points)
        mean += Float64(v)
    mean /= Float64(len(y))
    var total = Float64(0)
    for v in y:  # small-loop(84 points)
        total += (Float64(v) - mean) * (Float64(v) - mean)
    return sqrt(total / Float64(len(y)))


def same_record(records: List[Float32], a: Int, b: Int) -> Bool:
    for i in range(HW_RECORD):  # small-loop(one record pair)
        if bitcast[DType.uint32](records[a * HW_RECORD + i]) != bitcast[DType.uint32](records[b * HW_RECORD + i]):
            return False
    return True


def check_units(records: List[Float32], planes: List[Int]) raises:
    """The cross-unit assertions once every unit is fitted."""
    for seasonal in range(2):  # small-loop(two seasonal types)
        var name = String("additive") if seasonal == 0 else String("multiplicative")
        var first = 3 * seasonal
        require(same_record(records, first, first + 1), name + ": the estimated refit moved a word")
        require(not same_record(records, first, first + 2), name + ": estimated and heuristic fits are the same words")
        var est = HW_REFIT_UNITS + 2 * seasonal
        var y = clean_series(seasonal == 0)
        var base = Int(records.unsafe_ptr())
        var rmse_est = clean_rmse(f32_ptr(base + 4 * est * HW_RECORD), planes[est], y)
        var rmse_heu = clean_rmse(f32_ptr(base + 4 * (est + 1) * HW_RECORD), planes[est + 1], y)
        require(rmse_est <= rmse_heu * 1.001 + 1e-6,
                name + ": the estimated forecast RMSE " + String(rmse_est) + " exceeds the heuristic's " + String(rmse_heu))
        require(rmse_est < 0.05 * series_std(y), name + ": the estimated forecast RMSE " + String(rmse_est) + " is not small")
        _ = y^
