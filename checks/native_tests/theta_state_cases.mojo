# SPDX-License-Identifier: Apache-2.0
"""Native Theta fit-state/horizon contracts with independent closed forms.

Six exact fixtures use constant series (all four models) and a linear trend
(the two static models), fixed alpha=1 and theta=2. The constant forecast is
its level; a static linear fit with slope 2 forecasts last_y + j + 1. Two
additional automatic seasonal fits retain the original public regression's
3x60 values, period 7 and horizons 7/14, for both decomposition forms.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from sequence.exec import Exec
from sequence.ops_impl import theta_impl, theta_forecast_impl

comptime THETA_STATE_FIXTURES = 8
comptime THETA_GUARD = Float32(723.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct ThetaBuffer(Movable):
    var cells: List[Float32]
    var count: Int

    def __init__(out self, count: Int):
        self.count = count
        self.cells = List[Float32](length=count + 2, fill=THETA_GUARD)

    def address(self) -> Int:
        return Int(self.cells.unsafe_ptr()) + 4

    def guards(self) raises:
        require(self.cells[0] == THETA_GUARD and self.cells[self.count + 1] == THETA_GUARD,
            "Theta wrote outside output bounds")


def same_words(actual: ThetaBuffer, expected: ThetaBuffer, label: String) raises:
    require(actual.count == expected.count, label + ": buffer extent differs")
    actual.guards()
    expected.guards()
    var failures = List[Int](length=actual.count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=actual.address())
    var ep = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=expected.address())
    def cell(index: Int) {imm fp, imm ap, imm ep}:
        if bitcast[DType.uint32](ap[index]) != bitcast[DType.uint32](ep[index]):
            fp.unsafe_store(index, 1)
    host_parallelize(cell, actual.count)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, label + ": changed " + String(failed) + " words")
    _ = actual
    _ = expected


def expected_forecast(actual: ThetaBuffer, fixture_id: Int, batch: Int, horizon: Int) raises:
    actual.guards()
    if fixture_id >= 6:
        return  # Seasonal auto fit is the original full-fit/tail parity contract.
    var failures = List[Int](length=batch * horizon, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=actual.address())
    def cell(index: Int) {imm fp, imm ap, imm fixture_id, imm horizon}:
        var series = index // horizon
        var step = index % horizon
        # Independent closed-form expectations; no product state/fit helper.
        var expected = Float32(20 + 4 * series) if fixture_id < 4 else Float32(19 + 4 * series + step)
        if bitcast[DType.uint32](ap[index]) != bitcast[DType.uint32](expected):
            fp.unsafe_store(index, 1)
    host_parallelize(cell, batch * horizon)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "Theta independent closed form failed " + String(failed) + " cells")
    _ = actual


def run_theta_state_fixture[E: Exec](mut ex: E, fixture_id: Int) raises:
    var seasonal = fixture_id >= 6
    var n = 60 if seasonal else 8
    var batch = 3
    var period = 7 if seasonal else 1
    var model = -1 if seasonal else (fixture_id if fixture_id < 4 else fixture_id - 4)
    var decomposition = 1 if fixture_id == 7 else 0
    var fixed = 0 if seasonal else 7
    var observations = ThetaBuffer(batch * n)
    var op = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=observations.address())
    def input_cell(index: Int) {imm op, imm n, imm fixture_id}:
        var series = index // n
        var time = index % n
        var value = Float32(10 + index % 7) if fixture_id >= 6 else (
            Float32(20 + 4 * series) if fixture_id < 4 else Float32(4 + 4 * series + 2 * time))
        op[index] = value
    host_parallelize(input_cell, batch * n)
    var preserved = ThetaBuffer(batch * n)
    preserved.cells = observations.cells.copy()
    var info = ThetaBuffer(batch * 8)
    var short_forecast = ThetaBuffer(batch * 7)
    var fit_addresses: List[Int] = [observations.address(), short_forecast.address(), info.address()]
    var fit_ints: List[Int] = [batch, n, 7, period, model, decomposition, fixed]
    var fixed_values: List[Float64] = [20, 1, 2]
    require(theta_impl(ex, fit_addresses, fit_ints, fixed_values) == batch * 7, "Theta short-fit result count")
    expected_forecast(short_forecast, fixture_id, batch, 7)
    info.guards()
    var saved_info = ThetaBuffer(batch * 8)
    saved_info.cells = info.cells.copy()
    if not seasonal:
        for series in range(batch):  # Small model metadata, not sample arithmetic.
            require(info.cells[1 + series * 8 + 1] == 1 and info.cells[1 + series * 8 + 2] == 2 and
                info.cells[1 + series * 8 + 4] == Float32(model) and info.cells[1 + series * 8 + 5] == 0,
                "fixed native Theta metadata differs")
    # These horizons depend on one fitted state; independent fixtures above run
    # in parallel, but a fixture's state transition order is intentional.
    for horizon in [14, 5]:
        var tail = ThetaBuffer(batch * horizon)
        require(theta_forecast_impl(ex, [observations.address(), info.address(), tail.address()],
            [batch, n, horizon, period]) == batch * horizon, "Theta tail result count")
        expected_forecast(tail, fixture_id, batch, horizon)
        same_words(info, saved_info, "tail mutated borrowed fit metadata")
        same_words(observations, preserved, "tail mutated borrowed observations")
        var fresh = ThetaBuffer(batch * horizon)
        var fresh_info = ThetaBuffer(batch * 8)
        require(theta_impl(ex, [observations.address(), fresh.address(), fresh_info.address()],
            [batch, n, horizon, period, model, decomposition, fixed], fixed_values) == batch * horizon,
            "Theta full-fit result count")
        expected_forecast(fresh, fixture_id, batch, horizon)
        same_words(tail, fresh, "forecast tail differs from full fit at changed horizon")
        same_words(fresh_info, saved_info, "fit state depends on prediction horizon")
        _ = tail
        _ = fresh
        _ = fresh_info
    same_words(observations, preserved, "full fit mutated borrowed observations")
    short_forecast.guards()
    _ = info
    _ = saved_info
    _ = observations
    _ = preserved
    _ = short_forecast
