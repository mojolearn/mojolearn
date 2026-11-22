# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""DynamicTheta and OptimizedTheta batch forecasts equal each series fitted
alone, moved from test_public_verifier_gap_methods.py
`test_theta_variant_batch_forecasts_preserve_single_series` (2026-11-15).

Two series of 48 in float32: 20 + 0.1 t and 30 + sin(0.4 t) (each product,
sum and sine in float32, as the test built them with NumPy; the relation
does not depend on the last bit of the sine). The public door's call
(`forecast_run_theta`, bindings/xseq_forecast_native_door.mojo) runs the
same `sequence/ops_impl.mojo::theta_impl` words: model DSTM (2) or OTM (1),
multiplicative decomposition, alpha fixed at 0.3 (mask 2, settings
[0, 0.3, 0]), horizon 7.

  fixture = model x season_length: 0 DSTM m 1, 1 DSTM m 12, 2 OTM m 1,
  3 OTM m 12. The batch fit's forecast row of each series equals that series
  fitted alone, bit for bit, and every batch forecast is finite.
"""
from std.math import isfinite, sin
from std.memory import bitcast
from sequence.exec import Exec, HostExec
from sequence.ops_impl import theta_impl
from sequence.exec_device import DeviceExec
from std.testing import TestSuite, assert_true as require
from checks.native_tests.testsuite_cases import check_cases
from std.sys.compile import is_defined


comptime THV_FIXTURES = 4
comptime THV_N = 48
comptime THV_H = 7


def thv_model(unit: Int) -> Int:
    return 2 if unit < 2 else 1  # DSTM, then OTM (`_x_sequence_theta._MODELS`)


def thv_season(unit: Int) -> Int:
    return 12 if unit % 2 == 1 else 1


def thv_series() -> List[Float32]:
    var y = List[Float32](length=2 * THV_N, fill=0.0)
    for t in range(THV_N):  # small-loop(fixture: 96 test values)
        var time = Float32(t)
        y[t] = Float32(20.0) + time * Float32(0.1)
        y[THV_N + t] = Float32(30.0) + sin(time * Float32(0.4))
    return y^


def thv_fit[E: Exec](mut ex: E, y: List[Float32], first: Int, batch: Int, unit: Int,
                     mut forecast: List[Float32], mut info: List[Float32]) raises:
    var addresses: List[Int] = [Int(y.unsafe_ptr()) + 4 * first * THV_N, Int(forecast.unsafe_ptr()),
                                Int(info.unsafe_ptr())]
    var words: List[Int] = [batch, THV_N, THV_H, thv_season(unit), thv_model(unit), 0, 2]
    var settings: List[Float64] = [0.0, 0.3, 0.0]
    require(theta_impl(ex, addresses, words, settings) == batch * THV_H, "theta result count")


def run_variant_fixture[E: Exec](mut ex: E, unit: Int) raises:
    var tag = (String("DSTM") if thv_model(unit) == 2 else String("OTM")) + " season " + String(thv_season(unit))
    var y = thv_series()
    var batch_forecast = List[Float32](length=2 * THV_H, fill=-7.0)
    var batch_info = List[Float32](length=2 * 8, fill=-7.0)
    thv_fit(ex, y, 0, 2, unit, batch_forecast, batch_info)
    for i in range(2 * THV_H):  # fourteen forecast words
        require(isfinite(batch_forecast[i]), tag + ": batch forecast word " + String(i) + " is not finite")
    for row in range(2):  # the two series
        var single_forecast = List[Float32](length=THV_H, fill=-7.0)
        var single_info = List[Float32](length=8, fill=-7.0)
        thv_fit(ex, y, row, 1, unit, single_forecast, single_info)
        for j in range(THV_H):  # seven forecast words
            require(bitcast[DType.uint32](batch_forecast[row * THV_H + j]) == bitcast[DType.uint32](single_forecast[j]),
                    tag + ": batch row " + String(row) + " step " + String(j) + " is "
                    + String(batch_forecast[row * THV_H + j]) + ", the single fit " + String(single_forecast[j]))
        _ = single_forecast^
        _ = single_info^
    _ = y^
    _ = batch_forecast^
    _ = batch_info^


def _host_fixture(unit: Int) raises:
    var ex = HostExec()
    run_variant_fixture(ex, unit)


def _cpu_cases() raises:
    check_cases("sequence_theta_variants", THV_FIXTURES, _host_fixture)


def _gpu_cases() raises:
    var ex = DeviceExec()
    var failed = String("")
    var count = 0
    for unit in range(THV_FIXTURES):  # driver-loop: over fixtures on the one device
        try:
            run_variant_fixture(ex, unit)
        except error:
            count += 1
            failed += "\n      sequence_theta_variants row " + String(unit) + ": " + String(error)
    if count > 0:
        raise Error("sequence_theta_variants: " + String(count) + " of " + String(THV_FIXTURES) + " cases failed" + failed)


def test_cases() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        _cpu_cases()
    else:
        _gpu_cases()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
