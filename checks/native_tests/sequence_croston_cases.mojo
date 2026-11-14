# SPDX-License-Identifier: Apache-2.0
"""Croston classic and SBA against a float64 restatement of statsforecast's
`_croston_classic` / `_croston_sba` (ported from
python/mojolearn/tests/test_x_sequence_croston.py `test_classic_and_sba`).

Four intermittent series of 60 (about 30 percent non-zero demand of 1..8,
hashed; the original drew them from NumPy, and its reference never depended
on the stream), the last one all zeros. Each series' mean from
`sequence/ops_impl.mojo::croston_impl` and its two-step flat forecast from
`croston_forecast_impl` must equal d / itv (classic) or 0.95 d / itv (SBA)
within rtol 1e-5, where d and itv are simple exponential smoothing (alpha
0.1) of the non-zero demands and of their intervals; the all-zero series
forecasts its last value. One fixture per variant.
"""
from sequence.exec import Exec
from sequence.ops_impl import croston_impl, croston_forecast_impl
from core.host_parallel import host_parallelize
from checks.native_tests.sequence_port_common import require, allclose, u01

comptime CROSTON_FIXTURES = 2
comptime CR_B = 4
comptime CR_N = 60
comptime CR_H = 2
comptime CR_SALT = 0x5EC0C1


def croston_series() -> List[Float32]:
    var y = List[Float32](length=CR_B * CR_N, fill=0.0)
    var yp = y.unsafe_ptr()
    def cell(i: Int) {imm yp}:
        if i // CR_N != CR_B - 1 and u01(CR_SALT, i, 0) < 0.3:
            yp.unsafe_store(i, Float32(1 + Int(u01(CR_SALT, i, 1) * 8.0)))
    host_parallelize(cell, CR_B * CR_N)
    return y^


def ses(x: List[Float64], a: Float64) -> Float64:
    """The original `_ses`: f = x[0], then a v + (1 - a) f over x[:-1]."""
    var f = x[0]
    for k in range(len(x) - 1):  # a recursion: serial by nature
        f = a * x[k] + (1.0 - a) * f
    return a * x[len(x) - 1] + (1.0 - a) * f if len(x) > 1 else x[0]


def run_croston_fixture[E: Exec](mut ex: E, unit: Int) raises:
    var variant = 0 if unit == 0 else 2
    var factor = 1.0 if unit == 0 else 0.95
    var name = String("classic") if unit == 0 else String("SBA")
    var y = croston_series()
    var mean = List[Float32](length=CR_B, fill=-1.0)
    var forecast = List[Float32](length=CR_B * CR_H, fill=-1.0)
    require(croston_impl(ex, [Int(y.unsafe_ptr()), Int(mean.unsafe_ptr())], [CR_B, CR_N, variant]) == CR_B,
            name + ": Croston result count")
    require(croston_forecast_impl(ex, [Int(mean.unsafe_ptr()), Int(forecast.unsafe_ptr())], [CR_B, CR_H])
            == CR_B * CR_H, name + ": Croston forecast count")
    # 0 pass, 1 all-zero series not flat at its last value, 2 fewer than two
    # demands (a vacuous fixture), 3 forecast off the reference
    var failures = List[Int](length=CR_B, fill=0)
    var fp = failures.unsafe_ptr()
    var yp = y.unsafe_ptr()
    var op = forecast.unsafe_ptr()
    def series(i: Int) {imm fp, imm yp, imm op, imm factor}:
        var demands = List[Float64]()
        var intervals = List[Float64]()
        var previous = 0
        for t in range(CR_N):  # one series' demand times, in order
            var v = Float64(yp[i * CR_N + t])
            if v != 0.0:
                demands.append(v)
                intervals.append(Float64(t + 1 - previous))
                previous = t + 1
        if len(demands) == 0:
            if op[i * CR_H] != yp[i * CR_N + CR_N - 1] or op[i * CR_H + 1] != yp[i * CR_N + CR_N - 1]:
                fp.unsafe_store(i, 1)
            return
        if len(demands) < 2:
            fp.unsafe_store(i, 2)
            return
        var want = factor * ses(demands, 0.1) / ses(intervals, 0.1)
        for j in range(CR_H):
            if not allclose(Float64(op[i * CR_H + j]), want, 1e-5, 0.0):
                fp.unsafe_store(i, 3)
    host_parallelize(series, CR_B)
    for i in range(CR_B):
        require(failures[i] != 1, name + ": the all-zero series does not forecast its last value")
        require(failures[i] != 2, name + ": fixture series " + String(i) + " has fewer than two demands")
        require(failures[i] != 3, name + ": series " + String(i) + " forecast is not factor * d / itv (rtol 1e-5)")
    _ = y^
    _ = mean^
    _ = forecast^
