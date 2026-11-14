# SPDX-License-Identifier: Apache-2.0
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

The comparison against statsforecast's AutoETS stays in the Python file
(an external-library oracle). Importable; no `main()`.
"""
from std.math import isnan
from checks.native_tests.tsa_metrics_fixture_words import TSA_FIXTURE_DIR, fixture_f32

comptime ETS_N = 80
comptime ETS_B = 2
comptime ETS_H = 60
#: unit 0 fits both series; unit 1 + i fits series i alone
comptime ETS_UNITS = 1 + ETS_B
#: forecast words: the batch (ETS_B * ETS_H), then one row per single fit
comptime ETS_FORECAST_WORDS = 2 * ETS_B * ETS_H


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


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
