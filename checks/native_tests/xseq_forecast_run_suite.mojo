# SPDX-License-Identifier: Apache-2.0
"""The theta/Croston/ETS/GARCH forecaster RUN half
(`sequence/xseq_forecast_run.mojo`), reached without Python (lane
doorsplit-cnn-xseq, 2026-11-11). One case per group of RUN functions:
literal words and the forecasters' own refusal texts. The glue entry calls
need a published sequence table and are not called here.

The cases are independent units and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from sequence.xseq_forecast_run import (
    forecast_run_letter, forecast_run_index, forecast_run_copy_f32_words, forecast_run_theta_check_n,
    forecast_run_theta_mask, forecast_run_theta_codes, forecast_run_croston_check_n, forecast_run_croston_check_h,
    forecast_run_ets_check_n, forecast_run_ets_season, forecast_run_ets_needs_positive,
    forecast_run_ets_check_positive, forecast_run_ets_check_tiny, forecast_run_ets_check_counts,
    forecast_run_ets_mask, forecast_run_ets_words, forecast_run_round_half_even, forecast_run_garch_check_n,
    forecast_run_garch_check_finite, forecast_run_garch_horizon, forecast_run_garch_words, forecast_run_garch_stall,
)

comptime XSEQ_FORECAST_RUN_CASES = 13


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses(problem: Error, text: String) raises:
    require(String(problem) == text, "refusal text: got '" + String(problem) + "', expected '" + text + "'")


def run_case(unit: Int) raises:
    if unit == 0:
        require(forecast_run_letter("MAN", 1) == "A" and forecast_run_letter("MAN", 2) == "N", "spec letters")
        require(forecast_run_index("NAM", "M") == 2 and forecast_run_index("AM", "A") == 0, "letter index")
    elif unit == 1:
        var src: List[Float32] = [1.5, 2.5, 3.5]
        var dst = List[Float32](length=3, fill=0.0)
        forecast_run_copy_f32_words(Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 0)
        require(dst[0] == 0.0, "zero-word copy writes nothing")
        forecast_run_copy_f32_words(Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 3)
        require(dst[0] == 1.5 and dst[2] == 3.5, "copied words")
        _ = src
        _ = dst
    elif unit == 2:
        forecast_run_theta_check_n(4)
        try:
            forecast_run_theta_check_n(3)
            require(False, "three observations accepted")
        except problem:
            _refuses(problem, "Theta: each series needs more than 3 observations")
        require(forecast_run_theta_mask(True, False, True) == 5 and forecast_run_theta_mask(False, True, False) == 2,
                "theta mask")
    elif unit == 3:
        var info = List[Float32](length=16, fill=0.0)
        info[4] = 2.9
        info[12] = 1.0
        var codes = forecast_run_theta_codes(Int(info.unsafe_ptr()), 2)
        _ = info
        require(len(codes) == 2 and codes[0] == 2 and codes[1] == 1, "theta codes truncate toward zero")
    elif unit == 4:
        forecast_run_croston_check_n(1)
        forecast_run_croston_check_h(0)
        try:
            forecast_run_croston_check_n(0)
            require(False, "empty series accepted")
        except problem:
            _refuses(problem, "Croston: y must hold at least one observation per series")
        try:
            forecast_run_croston_check_h(-1)
            require(False, "negative horizon accepted")
        except problem:
            _refuses(problem, "Croston: h must be >= 0")
    elif unit == 5:
        try:
            forecast_run_ets_check_n(3)
            require(False, "three observations accepted")
        except problem:
            _refuses(problem, "ETS: each series needs at least 4 observations")
        require(forecast_run_ets_season("AAA", 4, 4) == "N", "season dropped at n <= m")
        require(forecast_run_ets_season("AAA", 5, 4) == "A", "season kept")
        require(forecast_run_ets_season("AAN", 3, 4) == "N", "no season")
    elif unit == 6:
        require(forecast_run_ets_needs_positive("MNN", "N") and forecast_run_ets_needs_positive("ANN", "M"),
                "multiplicative needs positive")
        require(not forecast_run_ets_needs_positive("ANN", "N"), "additive needs nothing")
        forecast_run_ets_check_positive(True)
        try:
            forecast_run_ets_check_positive(False)
            require(False, "non-positive data accepted")
        except problem:
            _refuses(problem, "ETS: Inappropriate model for data with negative or zero values")
    elif unit == 7:
        forecast_run_ets_check_tiny("AAN", "N", 9, False)
        try:
            forecast_run_ets_check_tiny("AAN", "N", 8, False)
            require(False, "tiny dataset accepted")
        except problem:
            _refuses(problem, "NotImplementedError: ETS: tiny datasets (n <= parameter count + 4) are not implemented")
    elif unit == 8:
        forecast_run_ets_check_counts("ANN", "N", 4, 1, False, True, True, True, True)
        try:
            forecast_run_ets_check_counts("ANN", "N", 3, 1, False, True, True, True, True)
            require(False, "too few observations accepted")
        except problem:
            _refuses(problem, "ETS: 2 parameters for 3 observations (the reference needs n > parameters + 1)")
        forecast_run_ets_check_counts("AAA", "A", 11, 4, True, True, True, True, True)
        try:
            forecast_run_ets_check_counts("AAA", "A", 10, 4, True, True, True, True, True)
            require(False, "seasonal parameter count accepted")
        except problem:
            _refuses(problem, "ETS: 9 parameters for 10 observations (the reference needs n > parameters + 1)")
    elif unit == 9:
        require(forecast_run_ets_mask(False, True, True, False) == 6 and forecast_run_ets_mask(True, False, False, True) == 9,
                "ets mask")
        var words = forecast_run_ets_words(2, 10, 3, "MAN", True, 5, "M", 4)
        var expected: List[Int] = [2, 10, 3, 1, 1, 1, 5, 2, 4]
        require(len(words) == len(expected), "ets word count")
        for i in range(len(expected)):  # driver-loop: over nine literal words
            require(words[i] == expected[i], "ets word " + String(i))
    elif unit == 10:
        require(forecast_run_round_half_even(2.5) == 2 and forecast_run_round_half_even(3.5) == 4, "ties to even")
        require(forecast_run_round_half_even(-2.5) == -2 and forecast_run_round_half_even(2.6) == 3, "round words")
        require(forecast_run_round_half_even(2.4) == 2, "round down")
        try:
            _ = forecast_run_round_half_even(Float64(1e19))
            require(False, "overflow accepted")
        except problem:
            _refuses(problem, "OverflowError: round() of the GARCH stall drop exceeds the integer word")
    elif unit == 11:
        forecast_run_garch_check_n(10)
        forecast_run_garch_check_finite(True)
        try:
            forecast_run_garch_check_n(9)
            require(False, "nine observations accepted")
        except problem:
            _refuses(problem, "GARCH: each series needs at least 10 observations")
        try:
            forecast_run_garch_check_finite(False)
            require(False, "non-finite series accepted")
        except problem:
            _refuses(problem, "GARCH: y holds a non-finite value")
        require(forecast_run_garch_horizon(0) == 1 and forecast_run_garch_horizon(5) == 5, "garch horizon")
    else:
        var words = forecast_run_garch_words(2, 20, 3, 1, 0, 1, True)
        require(len(words) == 7 and words[6] == 1 and words[3] == 1, "garch words")
        forecast_run_garch_stall(words, 50, 0.25)
        require(len(words) == 9 and words[7] == 50 and words[8] == 250000000, "garch stall words")
        var nan = Float64(0.0) / Float64(0.0)
        try:
            forecast_run_garch_stall(words, 50, nan)
            require(False, "NaN drop accepted")
        except problem:
            _refuses(problem, "ValueError: cannot convert float NaN to integer")


def run_checks() raises:
    var failures = List[Int](length=XSEQ_FORECAST_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL xseq forecast run case", unit, String(problem))
    host_parallelize(task, XSEQ_FORECAST_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "xseq forecast run cases failed: " + String(failed))
    print("PASS xseq forecast run", XSEQ_FORECAST_RUN_CASES)
