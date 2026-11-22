# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""DampedETS, ETS(A, Ad, N): the damped forecast flattens and a batch fit
equals the single-series fits (ported from
python/mojolearn/tests/test_x_sequence_ets.py
`test_damped_flattens_and_batch`).

The original two 80-point series (`np.random.default_rng(0)`, a trend plus
noise and a level plus noise) are NumPy PCG64 draws, stored as raw float32
words in `fixtures/tsa/ets-y.f32` by `tools/tsa_metrics_native_fixtures.py`.
The fit and forecast are the x_sequence glue's ETS operation with the
words `DampedETS()` sends (`sequence/xseq_forecast_run.mojo`
`forecast_run_ets_words`: error A, trend A, damped, nothing fixed, no
season, period 1; no fixed alpha, beta, phi or gamma), horizon 60:

- the increments of series 0's forecast shrink: d[k] <= d[k - 1] + 1e-4
  for every k >= 1, d = diff(forecast[0]) in float32;
- the batch-of-two forecast equals, value for value, each series fitted
  alone.

`test_against_statsforecast` (lane py2mojo-A, 2026-11-22; ported from
`test_against_statsforecast`): the batch-of-two fit with horizon 8 against
statsforecast AutoETS(model="AAN", damped=True) 8-step means of each series,
stored as binary64 words in `fixtures/tsa_ets_statsforecast/` by
`tools/py2mojo_native_fixtures_b8.py` (statsforecast 2.1.1), to rtol 5e-3
(assert_allclose's default atol 0).
"""
from std.math import isnan
from checks.native_tests.tsa_metrics_fixture_words import TSA_FIXTURE_DIR, fixture_f32
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from core.xseq_entry_table import XS_ETS
from sequence.exec import HostExec
from sequence.xseq_glue import xseq_glue_run
from sequence.xseq_glue_device import xseq_glue_device_run
from std.testing import TestSuite, assert_true as require
from checks.native_tests.testsuite_cases import check_cases
from std.sys.compile import is_defined


comptime ETS_N = 80
comptime ETS_B = 2
comptime ETS_H = 60
#: unit 0 fits both series; unit 1 + i fits series i alone
comptime ETS_UNITS = 1 + ETS_B
#: forecast words: the batch (ETS_B * ETS_H), then one row per single fit
comptime ETS_FORECAST_WORDS = 2 * ETS_B * ETS_H


trait EtsRun:
    @staticmethod
    def run(addresses: List[Int], words: List[Int], floats: List[Float64]) raises -> Int:
        """The glue's XS_ETS operation on this column; returns B * h."""
        ...


def ets_series() raises -> List[Float32]:
    """`_Y()`: series-major `ETS_B x ETS_N` float32."""
    return fixture_f32(TSA_FIXTURE_DIR + "ets-y.f32", ETS_B * ETS_N)


def ets_words(batch: Int) -> List[Int]:
    """batch, n, h, error A, trend A, damped, mask 0, season N, period 1."""
    var words: List[Int] = [batch, ETS_N, ETS_H, 0, 1, 1, 0, 0, 1]
    return words^


def ets_forecast_offset(unit: Int) -> Int:
    return 0 if unit == 0 else (ETS_B + unit - 1) * ETS_H


def ets_unit[R: EtsRun](unit: Int, y_address: Int, forecasts_address: Int) raises:
    """Fit and forecast one unit, writing its forecast into its own words."""
    var batch = ETS_B if unit == 0 else 1
    var series = 0 if unit == 0 else unit - 1
    var info = List[Float32](length=batch * 10, fill=Float32(0))
    var states = List[Float32](length=batch, fill=Float32(0))
    var addresses: List[Int] = [y_address + 4 * series * ETS_N, forecasts_address + 4 * ets_forecast_offset(unit),
                                Int(info.unsafe_ptr()), Int(states.unsafe_ptr())]
    var floats: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    require(R.run(addresses, ets_words(batch), floats) == batch * ETS_H, "ETS forecast count, unit " + String(unit))
    _ = info^
    _ = states^


def _same_value(a: Float32, b: Float32) -> Bool:
    return a == b or (isnan(a) and isnan(b))


def check_forecasts(forecasts: List[Float32]) raises:
    """The original assertions over the three units' forecasts."""
    require(len(forecasts) == ETS_FORECAST_WORDS, "ETS forecast words")
    for k in range(1, ETS_H - 1):  # small-loop(59 increments of one forecast row)
        var step = forecasts[k + 1] - forecasts[k]
        var before = forecasts[k] - forecasts[k - 1]
        require(step <= before + Float32(1e-4),
                "damped ETS forecast increment " + String(k) + " grew: " + String(step) + " > " + String(before))
    for series in range(ETS_B):  # small-loop(two series)
        var alone = ets_forecast_offset(1 + series)
        for i in range(ETS_H):  # small-loop(one forecast row)
            require(_same_value(forecasts[series * ETS_H + i], forecasts[alone + i]),
                    "ETS batch forecast differs from series " + String(series) + " fitted alone at step " + String(i))


# ---- The host column (was tsa_sequence_ets_cpu_suite.mojo) ----
# DampedETS forecast shape and batch composition on the host column
# (`sequence/xseq_glue.mojo` on `HostExec`, as the host image runs it).
#
# The three fits (the batch, then each series alone) are independent units
# and run across host cores, each with its own executor and its own forecast
# words; the comparisons follow once all three are written.
struct HostEts(EtsRun):
    @staticmethod
    def run(addresses: List[Int], words: List[Int], floats: List[Float64]) raises -> Int:
        var ex = HostExec()
        return xseq_glue_run(ex, XS_ETS, addresses, words, floats)


# ---- The device column (was tsa_sequence_ets_gpu_suite.mojo) ----
# DampedETS forecast shape and batch composition on the device column
# (`sequence/xseq_glue_device.mojo`: the ETS team where it applies, as the
# device image runs it).
#
# The three fits share the process's one sequence device context, so they
# are submitted in order; each fit's kernels keep their own parallelism
# (one series per block).
struct DeviceEts(EtsRun):
    @staticmethod
    def run(addresses: List[Int], words: List[Int], floats: List[Float64]) raises -> Int:
        return xseq_glue_device_run(XS_ETS, addresses, words, floats)


def _run_gpu() raises:
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():  # discovery instantiates every module function
        var y = ets_series()
        var forecasts = List[Float32](length=ETS_FORECAST_WORDS, fill=Float32(0))
        var y_address = Int(y.unsafe_ptr())
        var f_address = Int(forecasts.unsafe_ptr())
        for unit in range(ETS_UNITS):  # driver-loop: over the three fits on the one device context
            ets_unit[DeviceEts](unit, y_address, f_address)
        check_forecasts(forecasts)
        _ = y^
        print("PASS tsa_sequence_ets_gpu:", ETS_UNITS, "DampedETS fits")


def _run_cpu() raises:
    var y = ets_series()
    var forecasts = List[Float32](length=ETS_FORECAST_WORDS, fill=Float32(0))
    var y_address = Int(y.unsafe_ptr())
    var f_address = Int(forecasts.unsafe_ptr())
    def unit_fit(unit: Int) raises {imm y_address, imm f_address}:
        ets_unit[HostEts](unit, y_address, f_address)
    check_cases("tsa_sequence_ets", ETS_UNITS, unit_fit)
    check_forecasts(forecasts)
    _ = y^


comptime ETS_SF_H = 8
comptime ETS_SF_DIR = "checks/native_tests/fixtures/tsa_ets_statsforecast/"


def ets_statsforecast[R: EtsRun]() raises:
    """`DampedETS().fit(Y).predict(8)`: both series in one fit, horizon 8, against the stored AutoETS means."""
    var y = ets_series()
    var forecasts = List[Float32](length=ETS_B * ETS_SF_H, fill=Float32(0))
    var info = List[Float32](length=ETS_B * 10, fill=Float32(0))
    var states = List[Float32](length=ETS_B, fill=Float32(0))
    var addresses: List[Int] = [Int(y.unsafe_ptr()), Int(forecasts.unsafe_ptr()), Int(info.unsafe_ptr()),
                                Int(states.unsafe_ptr())]
    var words: List[Int] = [ETS_B, ETS_N, ETS_SF_H, 0, 1, 1, 0, 0, 1]
    var floats: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    require(R.run(addresses, words, floats) == ETS_B * ETS_SF_H, "ETS forecast count, horizon 8")
    var want = fixture_f64(ETS_SF_DIR + "ets-sf-mean.f64", ETS_B * ETS_SF_H)
    for i in range(ETS_B * ETS_SF_H):  # small-loop(oracle: two 8-step forecasts)
        var got = Float64(forecasts[i])
        require(abs(got - want[i]) <= 5e-3 * abs(want[i]), "ETS series " + String(i // ETS_SF_H) + " step "
                + String(i % ETS_SF_H) + ": " + String(forecasts[i]) + ", statsforecast " + String(want[i]))
    _ = y^
    _ = forecasts^
    _ = info^
    _ = states^


def test_against_statsforecast() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        ets_statsforecast[HostEts]()
    else:
        ets_statsforecast[DeviceEts]()


def test_fits_then_forecasts() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        _run_cpu()
    else:
        _run_gpu()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
