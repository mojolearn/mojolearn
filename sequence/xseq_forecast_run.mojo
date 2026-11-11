# SPDX-License-Identifier: Apache-2.0
"""RUN half of the theta, Croston, ETS and GARCH forecasters
(`bindings/xseq_forecast_native.mojo` is the UNPACK/PACK boundary; lane
doorsplit-cnn-xseq, 2026-11-11; docs/plans/PYTHON_AT_THE_BOUNDARY.md
section 2.2).

Typed values only: the model checks in the forecasters' own words and
refusal texts, the setting masks and argument words of each glue operation,
the selected-model code words of a theta fit, the stored-forecast word copy,
and the glue entry calls (`sequence/xseq_door_run.mojo` `xseq_run_glue`).
The boundary reads the model's fields, admits and allocates every buffer
(addresses arrive as words), keeps them alive, releases the GIL around each
glue call here, and writes the fitted state back. VAR and STL run through
the forecast matrix owner (`providers/python/forecast_matrix_pyapi.mojo`),
not this module.
"""
from std.math import floor
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef
from core.base_helpers_calls import base_reduce_stat
from core.xseq_entry_table import (
    XS_THETA, XS_THETA_FORECAST, XS_CROSTON, XS_CROSTON_FORECAST, XS_ETS, XS_GARCH,
)
from sequence.xseq_door_run import xseq_run_glue

comptime ForecastF32P = MutPointer[Float32, MutUntrackedOrigin]
comptime ForecastU8P = MutPointer[UInt8, MutUntrackedOrigin]


# ---------------------------------------------------------------- shared words

def forecast_run_letter(text: String, index: Int) raises -> String:
    """The one-byte letter of a model spec (`"AAN"[1]`)."""
    return String(text[byte=index:index + 1])


def forecast_run_index(letters: String, letter: String) -> Int:
    return letters.find(letter)


def forecast_run_copy_f32_words(dst: Int, src: Int, count: Int):
    """`count` float32 words from `src` to `dst` (the stored-forecast copy)."""
    if count:
        unsafe_memcpy(dest=ForecastU8P(unsafe_from_address=dst), src=ForecastU8P(unsafe_from_address=src),
                      count=count * 4)


# ---------------------------------------------------------------- theta

def forecast_run_theta_check_n(n: Int) raises:
    if n <= 3:
        raise Error("Theta: each series needs more than 3 observations")


def forecast_run_theta_mask(has_initial: Bool, has_alpha: Bool, has_theta: Bool) -> Int:
    """The fixed-setting mask: 1 initial level, 2 alpha, 4 theta."""
    return (1 if has_initial else 0) | (2 if has_alpha else 0) | (4 if has_theta else 0)


def forecast_run_theta(entry: EntryRef, y: Int, forecast: Int, info: Int, batch: Int, n: Int, h: Int,
                       season: Int, code: Int, additive: Bool, mask: Int, settings: List[Float64]) raises:
    """The whole theta fit and forecast (`XS_THETA`); `settings` holds the
    initial level, alpha and theta (zero when free). The caller keeps every
    addressed buffer alive and has released the GIL."""
    var addresses: List[Int] = [y, forecast, info]
    var words: List[Int] = [batch, n, h, season, code, 1 if additive else 0, mask]
    _ = xseq_run_glue(entry, XS_THETA, addresses, words, settings)


def forecast_run_theta_forecast(entry: EntryRef, y: Int, info: Int, forecast: Int, batch: Int, n: Int, h: Int,
                                season: Int) raises:
    """The fit's forecast-only tail at another horizon (`XS_THETA_FORECAST`),
    driven by the fit's own info rows."""
    var addresses: List[Int] = [y, info, forecast]
    var words: List[Int] = [batch, n, h, season]
    _ = xseq_run_glue(entry, XS_THETA_FORECAST, addresses, words, List[Float64]())


def forecast_run_theta_codes(info: Int, batch: Int) -> List[Int]:
    """The selected model of each series: info column 4 (of 8) truncated
    toward zero, as `astype(np.int64)`; the boundary maps each code to its
    connected name (Python indexing, its own IndexError)."""
    var words = ForecastF32P(unsafe_from_address=info)
    var codes = List[Int](capacity=batch)
    for row in range(batch):  # driver-loop: over the fitted series (one selected model code each)
        codes.append(Int(Int64(words[row * 8 + 4])))
    return codes^


# ---------------------------------------------------------------- Croston

def forecast_run_croston_check_n(n: Int) raises:
    if n < 1:
        raise Error("Croston: y must hold at least one observation per series")


def forecast_run_croston_check_h(h: Int) raises:
    if h < 0:
        raise Error("Croston: h must be >= 0")


def forecast_run_croston(entry: EntryRef, y: Int, mean: Int, batch: Int, n: Int, variant: Int) raises:
    """Each series' Croston mean (`XS_CROSTON`)."""
    var addresses: List[Int] = [y, mean]
    var words: List[Int] = [batch, n, variant]
    _ = xseq_run_glue(entry, XS_CROSTON, addresses, words, List[Float64]())


def forecast_run_croston_forecast(entry: EntryRef, mean: Int, forecast: Int, batch: Int, h: Int) raises:
    """Every forecast row its series' mean (`XS_CROSTON_FORECAST`)."""
    var addresses: List[Int] = [mean, forecast]
    var words: List[Int] = [batch, h]
    _ = xseq_run_glue(entry, XS_CROSTON_FORECAST, addresses, words, List[Float64]())


# ---------------------------------------------------------------- ETS

def forecast_run_ets_check_n(n: Int) raises:
    if n < 4:
        raise Error("ETS: each series needs at least 4 observations")


def forecast_run_ets_season(spec: String, n: Int, m: Int) raises -> String:
    """The season letter the fit keeps: the reference drops the season when
    a series is not longer than one period (ets_f)."""
    var s = forecast_run_letter(spec, 2)
    if s != "N" and n <= m:
        s = "N"
    return s


def forecast_run_ets_needs_positive(spec: String, season: String) raises -> Bool:
    """A multiplicative error or season needs every value > 0."""
    return forecast_run_letter(spec, 0) == "M" or season == "M"


def forecast_run_min_above_zero(base: EntryRef, y: Int, size: Int) raises -> Bool:
    """A first-wins minimum (`reduce_stat` 0) above zero; the boundary has
    already checked every value finite."""
    var low = base_reduce_stat(base, y, 0, size, 0)
    return low.f > 0.0


def forecast_run_ets_check_positive(positive: Bool) raises:
    if not positive:
        raise Error("ETS: Inappropriate model for data with negative or zero values")


def _forecast_run_ets_trend(spec: String) raises -> Int:
    return 1 if forecast_run_letter(spec, 1) != "N" else 0


def forecast_run_ets_check_tiny(spec: String, season: String, n: Int, damped: Bool) raises:
    """The tiny-dataset refusal of an ETS fit: n above the parameter count
    plus 4."""
    var trend = _forecast_run_ets_trend(spec)
    var seasonal = 1 if season != "N" else 0
    var npars = 2 + 2 * trend + 2 * seasonal + (1 if damped else 0)
    if n <= npars + 4:
        raise Error("NotImplementedError: ETS: tiny datasets (n <= parameter count + 4) are not implemented")


def forecast_run_ets_check_counts(spec: String, season: String, n: Int, m: Int, damped: Bool, alpha_free: Bool,
                                  beta_free: Bool, gamma_free: Bool, phi_free: Bool) raises:
    """Too few observations for the free parameters (the reference needs
    n > parameters + 1). A `*_free` flag is the setting left to the fit;
    beta counts only with a trend, gamma only with a season, phi only when
    damped."""
    var trend = _forecast_run_ets_trend(spec)
    var seasonal = 1 if season != "N" else 0
    var n_smooth = (1 if alpha_free else 0) + (1 if trend == 1 and beta_free else 0)
    n_smooth += (1 if seasonal == 1 and gamma_free else 0) + (1 if damped and phi_free else 0)
    var n_par = n_smooth + 1 + trend + (m - 1 if seasonal == 1 else 0)
    if n_par >= n - 1:
        raise Error("ETS: " + String(n_par) + " parameters for " + String(n)
                    + " observations (the reference needs n > parameters + 1)")


def forecast_run_ets_mask(has_alpha: Bool, has_beta: Bool, has_phi: Bool, has_gamma: Bool) -> Int:
    """The fixed-setting mask: 1 alpha, 2 beta, 4 phi, 8 gamma."""
    return (1 if has_alpha else 0) | (2 if has_beta else 0) | (4 if has_phi else 0) | (8 if has_gamma else 0)


def forecast_run_ets_words(batch: Int, n: Int, h: Int, spec: String, damped: Bool, mask: Int, season: String,
                           m: Int) raises -> List[Int]:
    """The ETS glue words: batch, n, h, error (A/M), trend (N/A), damped,
    mask, season (N/A/M), period. The FAST stop's iteration word, when set,
    is appended by the caller."""
    var words: List[Int] = [batch, n, h, forecast_run_index("AM", forecast_run_letter(spec, 0)),
                            forecast_run_index("NA", forecast_run_letter(spec, 1)), 1 if damped else 0, mask,
                            forecast_run_index("NAM", season), m]
    return words^


def forecast_run_ets(entry: EntryRef, y: Int, forecast: Int, info: Int, states: Int, words: List[Int],
                     floats: List[Float64]) raises:
    """The ETS fit and forecast (`XS_ETS`)."""
    var addresses: List[Int] = [y, forecast, info, states]
    _ = xseq_run_glue(entry, XS_ETS, addresses, words, floats)


# ---------------------------------------------------------------- GARCH

def forecast_run_round_half_even(x: Float64) raises -> Int:
    """Python's `round(x)` of a float as its integer: ties to even, its
    refusals of NaN and infinity. Plain arithmetic on one scalar word."""
    if x != x:
        raise Error("ValueError: cannot convert float NaN to integer")
    if x > 9.2e18 or x < -9.2e18:
        raise Error("OverflowError: round() of the GARCH stall drop exceeds the integer word")
    var f = floor(x)
    var whole = Int(f)
    var d = x - f
    if d > 0.5:
        return whole + 1
    if d < 0.5:
        return whole
    return whole + (whole & 1)


def forecast_run_garch_check_n(n: Int) raises:
    if n < 10:
        raise Error("GARCH: each series needs at least 10 observations")


def forecast_run_garch_check_finite(finite: Bool) raises:
    if not finite:
        raise Error("GARCH: y holds a non-finite value")


def forecast_run_garch_horizon(h: Int) -> Int:
    """The fit's stored forecast horizon: at least one step."""
    return max(1, h)


def forecast_run_garch_words(batch: Int, n: Int, h: Int, p: Int, o: Int, q: Int, constant: Bool) -> List[Int]:
    var words: List[Int] = [batch, n, h, p, o, q, 1 if constant else 0]
    return words^


def forecast_run_garch_stall(mut words: List[Int], iterations: Int, drop: Float64) raises:
    """The FAST stop's two words: the iteration cap and the relative drop in
    parts per 1e9 (Python's round of it, ties to even)."""
    words.append(iterations)
    words.append(forecast_run_round_half_even(drop * 1e9))


def forecast_run_garch(entry: EntryRef, y: Int, params: Int, info: Int, sigma: Int, forecast: Int,
                       words: List[Int]) raises:
    """The GARCH fit and forecast (`XS_GARCH`)."""
    var addresses: List[Int] = [y, params, info, sigma, forecast]
    _ = xseq_run_glue(entry, XS_GARCH, addresses, words, List[Float64]())
