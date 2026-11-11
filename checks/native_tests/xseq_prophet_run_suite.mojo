# SPDX-License-Identifier: Apache-2.0
"""The ProphetForecaster RUN half (`sequence/xseq_prophet_run.mojo`),
reached without Python (lane doorsplit-cnn-xseq, 2026-11-11). One case per
group of RUN functions: literal words and the model's own refusal texts.
The glue entry calls need a published sequence table and are not called
here.

The cases are independent units and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from sequence.xseq_prophet_run import (
    prophet_run_check_days, prophet_run_auto_period, prophet_run_scale, prophet_run_check_y_rank,
    prophet_run_check_lengths, prophet_run_check_finite, prophet_run_changepoint_plan, prophet_run_changepoint_count,
    prophet_run_check_changepoints, prophet_run_parameter_count, prophet_run_check_stored_dtype,
    prophet_run_check_ranks, prophet_run_check_holidays_given, prophet_run_check_order, prophet_run_check_schema,
    prophet_run_check_info_shape, prophet_run_check_params_shape, prophet_run_check_holiday_columns,
)

comptime XSEQ_PROPHET_RUN_CASES = 9


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses(problem: Error, text: String) raises:
    require(String(problem) == text, "refusal text: got '" + String(problem) + "', expected '" + text + "'")


def case(unit: Int) raises:
    if unit == 0:
        prophet_run_check_days(0)
        try:
            prophet_run_check_days(1)
            require(False, "NaN time accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster: t must not hold NaN")
        try:
            prophet_run_check_days(2)
            require(False, "unsorted times accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster: t must be sorted ascending")
    elif unit == 1:
        require(prophet_run_auto_period("yearly") == 365.25 and prophet_run_auto_period("weekly") == 7.0
                and prophet_run_auto_period("daily") == 1.0, "automatic periods")
        require(prophet_run_scale(30.0) == 30.0 and prophet_run_scale(0.0) == 1.0, "time scale")
    elif unit == 2:
        prophet_run_check_y_rank(1)
        prophet_run_check_lengths(5, 5)
        prophet_run_check_finite(True)
        try:
            prophet_run_check_y_rank(3)
            require(False, "rank-three y accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster: y must be a series or a batch of series")
        try:
            prophet_run_check_lengths(1, 1)
            require(False, "one point accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster: y and t lengths differ, or fewer than 2 points")
        try:
            prophet_run_check_finite(False)
            require(False, "non-finite y accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster: y must be finite")
    elif unit == 3:
        var a = prophet_run_changepoint_plan(100, 0.8, 25)
        require(a[0] == 80 and a[1] == 25, "changepoint plan")
        var b = prophet_run_changepoint_plan(10, 0.8, 25)
        require(b[0] == 8 and b[1] == 7, "changepoints capped by history")
        var c = prophet_run_changepoint_plan(1, 0.5, 25)
        require(c[0] == 0 and c[1] == 0, "no history, no changepoints")
        require(prophet_run_changepoint_count(-1, 0) == 1 and prophet_run_changepoint_count(3, 0) == 3
                and prophet_run_changepoint_count(-1, 7) == 7, "changepoint buffer words")
    elif unit == 4:
        prophet_run_check_changepoints(0)
        try:
            prophet_run_check_changepoints(-1)
            require(False, "outside changepoint accepted")
        except problem:
            _refuses(problem, "ValueError: Changepoints must fall within training data.")
        require(prophet_run_parameter_count(4, 6) == 13, "parameter count")
    elif unit == 5:
        prophet_run_check_stored_dtype(11)
        prophet_run_check_ranks(2, 2, 1)
        try:
            prophet_run_check_stored_dtype(12)
            require(False, "float64 fitted buffer accepted")
        except problem:
            _refuses(problem, "TypeError: ProphetForecaster fitted buffers require float32 storage")
        try:
            prophet_run_check_ranks(2, 1, 1)
            require(False, "rank-one diagnostics accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster fitted buffer ranks disagree")
    elif unit == 6:
        prophet_run_check_holidays_given(0, False)
        prophet_run_check_holidays_given(2, True)
        prophet_run_check_order(0)
        try:
            prophet_run_check_holidays_given(2, False)
            require(False, "missing holiday columns accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster: the model has holiday columns; pass them for t")
        try:
            prophet_run_check_order(-1)
            require(False, "negative Fourier order accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster Fourier orders must be nonnegative")
    elif unit == 7:
        prophet_run_check_schema(2, 8, 6)
        var schema = "ValueError: ProphetForecaster seasonalities changed after fit; fit the model again"
        try:
            prophet_run_check_schema(-1, 0, 0)
            require(False, "negative holiday count accepted")
        except problem:
            _refuses(problem, schema)
        try:
            prophet_run_check_schema(0, 5, 4)
            require(False, "changed seasonalities accepted")
        except problem:
            _refuses(problem, schema)
    else:
        prophet_run_check_info_shape(3, 4, 3)
        prophet_run_check_params_shape(13, 4, 6)
        prophet_run_check_holiday_columns(0, 0)
        prophet_run_check_holiday_columns(2, 2)
        try:
            prophet_run_check_info_shape(3, 5, 3)
            require(False, "diagnostics shape accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster fitted diagnostics shape disagrees")
        try:
            prophet_run_check_params_shape(12, 4, 6)
            require(False, "parameter shape accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster fitted parameter shape disagrees")
        try:
            prophet_run_check_holiday_columns(2, 3)
            require(False, "holiday column count accepted")
        except problem:
            _refuses(problem, "ValueError: ProphetForecaster holiday column count differs from fit")


def run_checks() raises:
    var failures = List[Int](length=XSEQ_PROPHET_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL xseq prophet run case", unit, String(problem))
    host_parallelize(task, XSEQ_PROPHET_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "xseq prophet run cases failed: " + String(failed))
    print("PASS xseq prophet run", XSEQ_PROPHET_RUN_CASES)
