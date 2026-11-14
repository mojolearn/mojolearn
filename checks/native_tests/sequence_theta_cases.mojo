# SPDX-License-Identifier: Apache-2.0
"""AutoTheta's own identities (ported from
python/mojolearn/tests/test_x_sequence_theta.py
`test_batch_equals_single_and_seasonality`).

Two series of 96: 30 + 5 sin(2 pi t / 12) + 0.1 t + z and a random walk
50 + cumsum(z'), z and z' hashed normals (the original drew them from
NumPy; neither relation depends on the stream). AutoTheta (model -1,
multiplicative decomposition, nothing fixed) with season_length 12 through
`sequence/ops_impl.mojo::theta_impl`, horizon 6:

- the batch fit's forecast row of each series equals that series fitted
  alone, bit for bit (fixture = series);
- the seasonal series was decomposed: info[0, 5] == 1.

The statsforecast comparison stays in the Python file (an external oracle).
"""
from std.math import sin
from std.memory import bitcast
from core.host_parallel import host_parallelize
from sequence.exec import Exec
from sequence.ops_impl import theta_impl
from checks.native_tests.sequence_port_common import require, normal

comptime THETA_FIXTURES = 2
comptime TH_N = 96
comptime TH_H = 6
comptime TH_M = 12


def theta_series() -> List[Float32]:
    var y = List[Float32](length=2 * TH_N, fill=0.0)
    var yp = y.unsafe_ptr()
    def seasonal(t: Int) {imm yp}:
        var tt = Float64(t)
        yp.unsafe_store(t, Float32(30.0 + 5.0 * sin(6.283185307179586 * tt / 12.0) + 0.1 * tt + normal(0x7E7A0, t)))
    host_parallelize(seasonal, TH_N)
    var walk = 0.0
    for t in range(TH_N):  # a cumulative sum: serial by nature
        walk += normal(0x7E7A1, t)
        y[TH_N + t] = Float32(walk + 50.0)
    return y^


def theta_fit[E: Exec](mut ex: E, y: List[Float32], first: Int, batch: Int, mut forecast: List[Float32],
                       mut info: List[Float32]) raises:
    var addresses: List[Int] = [Int(y.unsafe_ptr()) + 4 * first * TH_N, Int(forecast.unsafe_ptr()),
                                Int(info.unsafe_ptr())]
    var words: List[Int] = [batch, TH_N, TH_H, TH_M, -1, 0, 0]
    var settings = List[Float64](length=3, fill=0.0)
    require(theta_impl(ex, addresses, words, settings) == batch * TH_H, "AutoTheta result count")


def run_theta_fixture[E: Exec](mut ex: E, unit: Int) raises:
    var y = theta_series()
    var batch_forecast = List[Float32](length=2 * TH_H, fill=-7.0)
    var batch_info = List[Float32](length=2 * 8, fill=-7.0)
    theta_fit(ex, y, 0, 2, batch_forecast, batch_info)
    if unit == 0:
        require(batch_info[5] == 1.0, "AutoTheta did not decompose the seasonal series (info[0, 5] = "
                + String(batch_info[5]) + ")")
    var single_forecast = List[Float32](length=TH_H, fill=-7.0)
    var single_info = List[Float32](length=8, fill=-7.0)
    theta_fit(ex, y, unit, 1, single_forecast, single_info)
    for j in range(TH_H):  # six forecast words
        require(bitcast[DType.uint32](batch_forecast[unit * TH_H + j]) == bitcast[DType.uint32](single_forecast[j]),
                "AutoTheta batch row " + String(unit) + " step " + String(j) + " differs from the single fit")
    _ = y^
    _ = batch_forecast^
    _ = batch_info^
    _ = single_forecast^
    _ = single_info^
