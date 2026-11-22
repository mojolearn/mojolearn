# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Native Prophet state queries and exact piecewise-linear expectations.

Fixture0 retains the original full default fit (120 days, weekly order3,
25 changepoints,2000 iterations) and repeat130-day prediction. Five given
states have independent Float64 trend/holiday equations on dyadic values;
fit repeat and fused/separate feature parity are not independent oracles.
"""
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from std.math import sin, isfinite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from sequence.exec_trait import Exec
from sequence.prophet_prep import prophet_features, prophet_changepoints
from sequence.ops_impl import prophet_predict_impl, prophet_predict_days_impl, prophet_fit_impl
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.fit_team_impl import prophet_fit_team_impl
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef, entry_ref_word
from core.xseq_entry_table import XS_OP_COUNT
from sequence.xseq_glue_host import x_sequence_glue_host_v1
from sequence.xseq_prophet_run import (
    PROPHET_AUTO_AUTO, ProphetAuto, ProphetFitInput, ProphetSeason, ProphetTimes, prophet_run_fit_plan,
    prophet_run_prepare_fit,
)


# The column this binary was built for (`-D MOJOLEARN_COLUMN_CPU=1` builds the host column).
comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


comptime PROPHET_STATE_FIXTURES = 6
comptime PROPHET_GUARD = Float32(12345.5)


struct ProphetWords(Movable):
    var words: List[Float32]
    var count: Int

    def __init__(out self, count: Int):
        self.count = count
        self.words = List[Float32](length=count + 2, fill=Float32(0))
        self.words[0] = PROPHET_GUARD
        self.words[count + 1] = PROPHET_GUARD

    def address(self) -> Int:
        return Int(self.words.unsafe_ptr()) + 4

    def guards(self) raises:
        require(self.words[0] == PROPHET_GUARD and self.words[self.count + 1] == PROPHET_GUARD,
            "Prophet wrote outside a native buffer")


struct ProphetDays(Movable):
    var words: List[Float64]
    var count: Int

    def __init__(out self, count: Int, offset: Int):
        self.count = count
        self.words = List[Float64](length=count + 2, fill=Float64(99999.25))
        var p = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(self.words.unsafe_ptr()))
        def cell(index: Int) {imm p, imm offset}:
            p[index + 1] = Float64(index + offset)
        host_parallelize(cell, count)
        _ = self.words

    def address(self) -> Int:
        return Int(self.words.unsafe_ptr()) + 8

    def guards(self) raises:
        require(self.words[0] == 99999.25 and self.words[self.count + 1] == 99999.25,
            "Prophet wrote outside day input")


def same_words(actual: ProphetWords, expected: List[Float32], label: String) raises:
    actual.guards()
    require(len(expected) == actual.count + 2, label + ": extent differs")
    var failures = List[Int](length=actual.count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=actual.address())
    var ep = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(expected.unsafe_ptr()) + 4)
    def cell(index: Int) {imm fp, imm ap, imm ep}:
        if bitcast[DType.uint32](ap[index]) != bitcast[DType.uint32](ep[index]):
            fp.unsafe_store(index, 1)
    host_parallelize(cell, actual.count)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, label + ": " + String(failed) + " raw words differ")
    _ = actual
    _ = expected


def holiday(index: Int, column: Int) -> Float32:
    if column == 0:
        return Float32(1) if index % 3 == 0 else Float32(0)
    return Float32(1) if index % 5 < 2 else Float32(0)


struct ProphetState(Movable):
    var fixture_id: Int
    var batch: Int
    var ns: Int
    var nh: Int
    var features: Int
    var changes: Int
    var multiplicative: Int
    var scale: Float64
    var periods: List[Float64]
    var orders: ProphetWords
    var changepoints: ProphetWords
    var params: ProphetWords
    var info: ProphetWords

    def __init__(out self, fixture_id: Int):
        self.fixture_id = fixture_id
        self.batch = 1 if fixture_id == 0 else 3
        self.ns = 1 if fixture_id == 0 else 0
        self.nh = 2 if fixture_id >= 4 else 0
        self.features = 6 if fixture_id == 0 else self.nh
        self.changes = 25 if fixture_id == 0 else (2 if fixture_id >= 3 else 0)
        self.multiplicative = 1 if fixture_id == 5 else 0
        self.scale = Float64(119) if fixture_id == 0 else Float64(8)
        self.periods = [Float64(7)]
        self.orders = ProphetWords(1)
        self.orders.words[1] = Float32(3) if fixture_id == 0 else Float32(0)
        self.changepoints = ProphetWords(max(1, self.changes))
        if fixture_id >= 3:
            self.changepoints.words[1] = 0.25
            self.changepoints.words[2] = 0.75
        var width = 3 + self.changes + self.features
        self.params = ProphetWords(self.batch * width)
        self.info = ProphetWords(self.batch * 4)
        if fixture_id != 0:
            var pp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.params.address())
            var ip = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.info.address())
            var changes = self.changes
            def parameter(series: Int) {imm pp, imm ip, imm width, imm changes, imm fixture_id}:
                pp[series * width] = Float32(0) if fixture_id == 1 else Float32(series + 1) * Float32(0.25)
                pp[series * width + 1] = Float32(2 + series)
                if changes:
                    pp[series * width + 2] = 0.5
                    pp[series * width + 3] = -0.25
                pp[series * width + 2 + changes] = 1  # Positive observation scale parameter.
                if fixture_id >= 4:
                    pp[series * width + 3 + changes] = 0.5
                    pp[series * width + 4 + changes] = -0.25
                ip[series * 4] = Float32(2 << series)  # Public y_scale, independent of sigma.
            host_parallelize(parameter, self.batch)

    def ints(self, count: Int) -> List[Int]:
        return [self.batch, count, self.ns, self.nh, self.features, self.changes, self.multiplicative]

    def guards(self) raises:
        self.orders.guards()
        self.changepoints.guards()
        self.params.guards()
        self.info.guards()


struct ProphetFitInputs(Movable):
    var days: ProphetDays
    var y: ProphetWords
    var scaled: ProphetWords
    var phases: ProphetWords
    var holidays: ProphetWords
    var sigmas: ProphetWords
    var user: List[Float64]

    def __init__(out self):
        self.days = ProphetDays(120, 0)
        self.y = ProphetWords(120)
        self.scaled = ProphetWords(120)
        self.phases = ProphetWords(120)
        self.holidays = ProphetWords(120)
        self.sigmas = ProphetWords(6)
        self.user = [Float64(0)]
        var yp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.y.address())
        def cell(index: Int) {imm yp}:
            var t = Float64(index)
            yp[index] = Float32(sin(t / 7) + t / 50)
        host_parallelize(cell, 120)
        var sp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.sigmas.address())
        def prior(index: Int) {imm sp}:
            sp[index] = 10
        host_parallelize(prior, 6)

    def addresses(self, state: ProphetState) -> List[Int]:
        return [self.y.address(), self.scaled.address(), self.phases.address(), state.orders.address(),
            self.holidays.address(), state.changepoints.address(), self.sigmas.address(), state.params.address(), state.info.address()]

    def guards(self) raises:
        self.days.guards()
        self.y.guards()
        self.scaled.guards()
        self.phases.guards()
        self.holidays.guards()
        self.sigmas.guards()


def prepare_fit[E: Exec](mut ex: E, inputs: ProphetFitInputs, state: ProphetState) raises:
    var feature_addresses: List[Int] = [inputs.days.address(), Int(state.periods.unsafe_ptr()), inputs.scaled.address(), inputs.phases.address()]
    require(prophet_features(ex, feature_addresses, 120, 1, 0, 119) == 0, "Prophet fit feature count")
    var change_addresses: List[Int] = [inputs.days.address(), Int(inputs.user.unsafe_ptr()), state.changepoints.address()]
    require(prophet_changepoints(change_addresses, 120, -1, 96, 25, 0, 119) == 25, "Prophet default changepoint count")
    inputs.guards()
    state.guards()
    _ = inputs
    _ = state


def check_equations(state: ProphetState, yhat: ProphetWords, trend: ProphetWords, count: Int) raises:
    yhat.guards()
    trend.guards()
    var failures = List[Int](length=yhat.count, fill=0)
    var fp = failures.unsafe_ptr()
    var yp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=yhat.address())
    var tp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=trend.address())
    var fixture_id = state.fixture_id
    def cell(index: Int) {imm fp, imm yp, imm tp, imm count, imm fixture_id}:
        if fixture_id == 0:
            if not isfinite(yp[index]) or not isfinite(tp[index]):
                fp.unsafe_store(index, 1)
            return  # Full fit has repeat parity, not a closed-form fit oracle.
        var series = index // count
        var column = index % count
        var time = Float64(column - 2) / 8
        var slope = Float64(0) if fixture_id == 1 else Float64(series + 1) * Float64(0.25)
        var expected_trend = Float64(2 + series) + slope * time
        if fixture_id >= 3:
            expected_trend += Float64(0.5) * max(time - Float64(0.25), Float64(0))
            expected_trend -= Float64(0.25) * max(time - Float64(0.75), Float64(0))
        var season = Float64(0)
        if fixture_id >= 4:
            season = Float64(0.5) * Float64(holiday(column, 0)) - Float64(0.25) * Float64(holiday(column, 1))
        var expected_y = expected_trend * (1 + season) if fixture_id == 5 else expected_trend + season
        var yscale = Float64(2 << series)
        if bitcast[DType.uint32](yp[index]) != bitcast[DType.uint32](Float32(expected_y * yscale)):
            fp.unsafe_store(index, 1)
        if bitcast[DType.uint32](tp[index]) != bitcast[DType.uint32](Float32(expected_trend * yscale)):
            fp.unsafe_store(index, 1)
    host_parallelize(cell, yhat.count)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "Prophet independent equations/nonfinite fit failed in " + String(failed) + " cells")
    _ = state
    _ = yhat
    _ = trend


def run_queries[E: Exec](mut ex: E, state: ProphetState) raises:
    var saved_params = state.params.words.copy()
    var saved_info = state.info.words.copy()
    var saved_orders = state.orders.words.copy()
    var saved_changes = state.changepoints.words.copy()
    for query_id in range(2):  # Dependent predictions from one fixed fitted state.
        var count = 130 + query_id * 7 if state.fixture_id == 0 else 17 + query_id * 6
        var days = ProphetDays(count, 0 if state.fixture_id == 0 else -2)
        var holidays = ProphetWords(count * max(state.nh, 1))
        if state.nh:
            var hp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=holidays.address())
            def cell(index: Int) {imm hp}:
                hp[index] = holiday(index // 2, index % 2)
            host_parallelize(cell, count * 2)
        var saved_holidays = holidays.words.copy()
        var first = ProphetWords(state.batch * count)
        var trend = ProphetWords(first.count)
        var repeated = ProphetWords(first.count)
        var repeated_trend = ProphetWords(first.count)
        var ip = state.ints(count)
        var fp: List[Float64] = [0, state.scale]
        var addrs: List[Int] = [days.address(), Int(state.periods.unsafe_ptr()), state.params.address(), state.info.address(),
            state.orders.address(), holidays.address(), state.changepoints.address(), first.address(), trend.address()]
        require(prophet_predict_days_impl(ex, addrs, ip, fp) == first.count, "Prophet fused prediction count")
        check_equations(state, first, trend, count)
        addrs[7] = repeated.address()
        addrs[8] = repeated_trend.address()
        require(prophet_predict_days_impl(ex, addrs, ip, fp) == repeated.count, "Prophet repeated prediction count")
        same_words(repeated, first.words, "Prophet repeat prediction")
        same_words(repeated_trend, trend.words, "Prophet repeat trend")
        var scaled = ProphetWords(count)
        var phases = ProphetWords(count * max(state.ns, 1))
        var feature_addrs: List[Int] = [days.address(), Int(state.periods.unsafe_ptr()), scaled.address(), phases.address()]
        require(prophet_features(ex, feature_addrs, count, state.ns, 0, state.scale) == 0, "Prophet query feature count")
        var plain_addrs: List[Int] = [state.params.address(), state.info.address(), scaled.address(), phases.address(),
            state.orders.address(), holidays.address(), state.changepoints.address(), repeated.address(), repeated_trend.address()]
        require(prophet_predict_impl(ex, plain_addrs, ip) == repeated.count, "Prophet separate prediction count")
        same_words(repeated, first.words, "Prophet fused/separate feature prediction")
        same_words(repeated_trend, trend.words, "Prophet fused/separate feature trend")
        same_words(state.params, saved_params, "Prophet prediction changed fitted parameters")
        same_words(state.info, saved_info, "Prophet prediction changed fitted statistics")
        same_words(state.orders, saved_orders, "Prophet prediction changed Fourier orders")
        same_words(state.changepoints, saved_changes, "Prophet prediction changed changepoints")
        same_words(holidays, saved_holidays, "Prophet prediction changed holiday input")
        days.guards()
        scaled.guards()
        phases.guards()
        state.guards()
        _ = days
        _ = holidays
        _ = first
        _ = trend
        _ = repeated
        _ = repeated_trend
        _ = scaled
        _ = phases
    _ = state


# ---- The host column (was prophet_state_cpu_suite.mojo) ----
# Importable native Prophet CPU fitted-state checks.
def fixture(fixture_id: Int) raises:
    var ex = HostExec()
    var state = ProphetState(fixture_id)
    if fixture_id == 0:
        var inputs = ProphetFitInputs()
        prepare_fit(ex, inputs, state)
        var addresses = inputs.addresses(state)
        var ip = state.ints(120)
        ip.append(2000)
        var fp: List[Float64] = [0.05]
        require(prophet_fit_impl(ex, addresses, ip, fp) == 1, "Prophet full default fit count")
        inputs.guards()
        state.guards()
        _ = inputs
    run_queries(ex, state)
    _ = state


def same_prefix(actual: List[Float32], expected: ProphetWords, label: String) raises:
    """The first `expected.count` words of a plan buffer, raw, against a guarded buffer."""
    expected.guards()
    require(len(actual) >= expected.count, label + ": extent differs")
    var failures = List[Int](length=expected.count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(actual.unsafe_ptr()))
    var ep = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=expected.address())
    def cell(index: Int) {imm fp, imm ap, imm ep}:
        if bitcast[DType.uint32](ap[index]) != bitcast[DType.uint32](ep[index]):
            fp.unsafe_store(index, 1)
    host_parallelize(cell, expected.count)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, label + ": " + String(failed) + " raw words differ")
    _ = actual
    _ = expected


def typed_entry_fit() raises:
    """The door's typed fit entry (sequence/xseq_prophet_run.mojo `prophet_run_prepare_fit` then
    `prophet_run_fit_plan`) through the host image's glue entry, word for word against fixture 0's
    direct feature, changepoint and fit calls on the same inputs and host executor: weekly order 3 is
    the automatic seasonality of 120 daily points, 25 changepoints over floor(120 * 0.8) = 96 points,
    prior scale 10 for its 6 Fourier parameters, no holidays."""
    var ex = HostExec()
    var state = ProphetState(0)
    var inputs = ProphetFitInputs()
    prepare_fit(ex, inputs, state)
    var addresses = inputs.addresses(state)
    var ip = state.ints(120)
    ip.append(2000)
    var fp: List[Float64] = [0.05]
    require(prophet_fit_impl(ex, addresses, ip, fp) == 1, "Prophet full default fit count")
    var entry = EntryRef(x_sequence_glue_host_v1, entry_ref_word(Int(GLOBAL_NUMERIC_MODE), 0, XS_OP_COUNT))
    var fit_input = ProphetFitInput(ProphetTimes(inputs.days.address(), 120, False), False, 1, 120, True,
        List[ProphetSeason](), ProphetAuto(PROPHET_AUTO_AUTO, 0), ProphetAuto(PROPHET_AUTO_AUTO, 0),
        ProphetAuto(PROPHET_AUTO_AUTO, 0), 10.0, False, ProphetTimes(0, 0, False), 0.8, 25)
    var plan = prophet_run_prepare_fit(entry, fit_input)
    require(len(plan.seasons) == 1 and plan.n_added == 1 and plan.seasons[0].name == "weekly"
            and plan.seasons[0].fourier_order == 3, "Prophet typed entry: automatic seasonalities")
    require(plan.s == 25 and plan.start == 0 and plan.t_scale == 119, "Prophet typed entry: plan words")
    same_prefix(plan.t_scaled, inputs.scaled, "Prophet typed entry scaled times")
    same_prefix(plan.phases, inputs.phases, "Prophet typed entry phases")
    same_prefix(plan.changepoints, state.changepoints, "Prophet typed entry changepoints")
    var params = ProphetWords(state.params.count)
    var info = ProphetWords(state.info.count)
    prophet_run_fit_plan(entry, plan, inputs.y.address(), 1, inputs.holidays.address(), 0, 10.0, params.address(),
                         info.address(), False, 2000, 0.05)
    same_words(params, state.params.words, "Prophet typed entry fitted parameters")
    same_words(info, state.info.words, "Prophet typed entry fitted statistics")
    inputs.guards()
    state.guards()
    _ = inputs
    _ = state
    _ = plan
    _ = params
    _ = info


# ---- The device column (was prophet_state_gpu_suite.mojo) ----
# Importable native Prophet GPU fitted-state checks.
def fixture_gpu(fixture_id: Int) raises:
    # Device column only: every module function is instantiated by TestSuite discovery, so the host build
    # must not reach the device route.
    comptime if not CPU_COLUMN_BUILD:
        var ex = DeviceExec()
        var state = ProphetState(fixture_id)
        if fixture_id == 0:
            var inputs = ProphetFitInputs()
            prepare_fit(ex, inputs, state)
            var addresses = inputs.addresses(state)
            var ip = state.ints(120)
            ip.append(2000)
            var fp: List[Float64] = [0.05]
            require(prophet_fit_team_impl(ex, addresses, ip, fp) == 1, "Prophet full default fit count")
            inputs.guards()
            state.guards()
            _ = inputs
        run_queries(ex, state)
        _ = state


def test_state() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("state", PROPHET_STATE_FIXTURES, fixture)
    else:
        # Shared sequence DeviceContext: independent states submitted in order; every production fit/prediction
        # uses its existing parallel device kernels.
        for fixture_id in range(PROPHET_STATE_FIXTURES):  # driver-loop: ordered on the shared sequence context
            try:
                fixture_gpu(fixture_id)
            except problem:
                raise Error("state row " + String(fixture_id) + ": " + String(problem))


def test_typed_entry_fit() raises:
    comptime if CPU_COLUMN_BUILD:
        typed_entry_fit()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
