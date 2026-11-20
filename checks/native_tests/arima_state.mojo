# SPDX-License-Identifier: Apache-2.0
"""Native ARIMA fitted-state checks and exact dyadic prediction expectations.

Original MLE/maxiter5/order111 and order100+one-regressor settings are retained
in two fitted cases. Six separately supplied states have independent AR(1)
conditional-mean/random-walk equations, with zero MA coefficient in the111
closed forms. Fitted-case resident/per-call equality is ownership parity, not
an independent numerical oracle. No Python runtime or old hashes are used.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from arima.host.arima_host_entries import _order, _forecast_into, _predict_in_sample_into, arima_host_fit_into
from arima.estimator import arima_fit_ptr_host, arima_predict_ptr_host, arima_forecast_ptr_host
from arima.resident_state import (
    arima_resident_prepare,
    arima_resident_count,
    arima_resident_release,
    arima_predict_resident,
)
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime ARIMA_STATE_FIXTURES = 8
comptime ARIMA_STATE_QUERIES = 4
comptime ARIMA_GUARD = Float32(31415.5)


struct ArimaWords(Movable):
    var words: List[Float32]
    var count: Int

    def __init__(out self, count: Int):
        self.count = count
        self.words = List[Float32](length=count + 2, fill=ARIMA_GUARD)

    def address(self) -> Int:
        return Int(self.words.unsafe_ptr()) + 4

    def guards(self) raises:
        require(self.words[0] == ARIMA_GUARD and self.words[self.count + 1] == ARIMA_GUARD,
            "ARIMA wrote outside a native buffer")

    def fill(mut self, value: Float32):
        var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(self.words.unsafe_ptr()))
        var count = self.count
        def cell(index: Int) {imm p, imm value}:
            p[index + 1] = value
        host_parallelize(cell, count)
        _ = self.words


def same_words(actual: ArimaWords, expected: List[Float32], label: String) raises:
    actual.guards()
    require(len(expected) == actual.count + 2, label + ": extent differs")
    var failure = List[Int](length=actual.count, fill=0)
    var fp = failure.unsafe_ptr()
    var ap = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=actual.address())
    var ep = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(expected.unsafe_ptr()) + 4)
    def cell(index: Int) {imm fp, imm ap, imm ep}:
        if bitcast[DType.uint32](ap[index]) != bitcast[DType.uint32](ep[index]):
            fp.unsafe_store(index, 1)
    host_parallelize(cell, actual.count)
    var failed = 0
    for value in failure:
        failed += value
    require(failed == 0, label + ": " + String(failed) + " raw words differ")
    _ = actual
    _ = expected


def observation(index: Int, fixture_id: Int) -> Float32:
    # Native deterministic dyadic data replaces fixture-only NumPy RNG. Keep
    # distinct seed labels7/11 and a nontrivial signed distribution per series.
    var seed = 11 if fixture_id == 1 or fixture_id == 4 or fixture_id == 5 or fixture_id == 7 else 7
    var word = UInt32(index + 1) * UInt32(747796405) + UInt32(seed) * UInt32(2891336453)
    word = ((word >> 16) ^ word) * UInt32(2246822519)
    return Float32(Int(word & 1023) - 512) * Float32(0.0078125)


def regressor(series: Int, time: Int) -> Float32:
    return Float32(series * 7 + time % 11 - 5) * Float32(0.125)


struct ArimaState(Movable):
    var fixture_id: Int
    var batch: Int
    var n: Int
    var d: Int
    var q: Int
    var k: Int
    var exog_count: Int
    var complexity: Int
    var y: ArimaWords
    var exog: ArimaWords
    var params: ArimaWords
    var x: ArimaWords
    var x0: ArimaWords
    var stats: ArimaWords
    var flags: List[Int32]

    def __init__(out self, fixture_id: Int):
        self.fixture_id = fixture_id
        var exogenous = fixture_id == 1 or fixture_id == 4 or fixture_id == 5 or fixture_id == 7
        self.batch = 2 if exogenous else 3
        self.n = 64 if exogenous else 80
        self.d = 0 if exogenous else 1
        self.q = 0 if exogenous else 1
        self.k = 1 if exogenous else 0
        self.exog_count = 1 if exogenous else 0
        self.complexity = 4 if exogenous else 3
        self.y = ArimaWords(self.batch * self.n)
        self.exog = ArimaWords(self.batch * self.n if exogenous else 0)
        self.params = ArimaWords(self.batch * self.complexity)
        self.x = ArimaWords(self.batch * self.complexity)
        self.x0 = ArimaWords(self.batch * self.complexity)
        self.stats = ArimaWords(self.batch * 2)
        self.flags = List[Int32](length=self.batch * 2 + 2, fill=Int32(12345))
        var yp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.y.address())
        var ep = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.exog.address())
        var n = self.n
        def cell(index: Int) {imm yp, imm ep, imm n, imm exogenous, imm fixture_id}:
            yp[index] = observation(index, fixture_id)
            if exogenous:
                ep[index] = regressor(index // n, index % n)
        host_parallelize(cell, self.batch * self.n)
        if fixture_id >= 2:
            var pp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.params.address())
            var width = self.complexity
            def parameter(series: Int) {imm pp, imm width, imm exogenous, imm fixture_id}:
                var phi = Float32(0.5)
                if fixture_id == 3 or fixture_id == 4:
                    phi = 0
                elif fixture_id == 6 or fixture_id == 7:
                    phi = -0.5
                if exogenous:
                    pp[series * width] = Float32(2) if fixture_id == 4 else (Float32(0.75) if fixture_id == 7 else Float32(1))
                    pp[series * width + 1] = 0.5
                    pp[series * width + 2] = phi
                    pp[series * width + 3] = 1
                else:
                    pp[series * width] = phi
                    pp[series * width + 1] = 0  # MA coefficient; order remains111.
                    pp[series * width + 2] = 1
            host_parallelize(parameter, self.batch)

    def exog_address(self) -> Int:
        return self.exog.address() if self.exog_count else 0

    def guards(self) raises:
        self.y.guards()
        self.exog.guards()
        self.params.guards()
        self.x.guards()
        self.x0.guards()
        self.stats.guards()
        require(self.flags[0] == 12345 and self.flags[self.batch * 2 + 1] == 12345, "ARIMA fit flag bounds")


trait ArimaCalls:
    @staticmethod
    def fit(state: ArimaState) raises -> Int:
        ...

    @staticmethod
    def predict(state: ArimaState, future: Int, target: Int, start: Int, end: Int) raises -> Int:
        ...

    @staticmethod
    def forecast(state: ArimaState, future: Int, target: Int, steps: Int) raises -> Int:
        ...


def query(state: ArimaState, query_id: Int) -> Tuple[Int, Int, Bool]:
    if query_id == 0:
        return (state.n, state.n + (5 if state.exog_count else 6), True)
    if query_id == 1:
        return (state.n, state.n + (8 if state.exog_count else 9), True)
    if query_id == 2:
        return (10, 70 if state.exog_count else 90, False)
    return (0, 3, False)


def future_words(state: ArimaState, end: Int) -> ArimaWords:
    var steps = max(end - state.n, 0)
    var result = ArimaWords(state.batch * steps if state.exog_count else 0)
    if state.exog_count:
        var fp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=result.address())
        var n = state.n
        def cell(index: Int) {imm fp, imm n, imm steps}:
            fp[index] = regressor(index // steps, n + index % steps)
        host_parallelize(cell, state.batch * steps)
    return result^


def check_closed_form(state: ArimaState, target: ArimaWords, start: Int, end: Int) raises:
    target.guards()
    if state.fixture_id < 2:
        return
    var length = end - start
    var failure = List[Int](length=target.count, fill=0)
    var fp = failure.unsafe_ptr()
    var dst = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=target.address())
    var yp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=state.y.address())
    var pp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=state.params.address())
    var n = state.n
    var width = state.complexity
    var exogenous = state.exog_count != 0
    def cell(index: Int) {imm fp, imm dst, imm yp, imm pp, imm n, imm width, imm exogenous, imm start, imm length}:
        var series = index // length
        var time = start + index % length
        var expected = Float64(0)
        if exogenous:
            var mu = Float64(pp[series * width])
            var beta = Float64(pp[series * width + 1])
            var phi = Float64(pp[series * width + 2])
            if time == 0:
                expected = mu / (1 - phi)
            elif time < n:
                expected = mu + phi * (Float64(yp[series * n + time - 1]) - beta * Float64(regressor(series, time - 1)))
            else:
                expected = Float64(yp[series * n + n - 1]) - beta * Float64(regressor(series, n - 1))
                for step in range(time - n + 1):  # Dependent mathematical AR recurrence in Float64.
                    expected = mu + phi * expected
            expected += beta * Float64(regressor(series, time))
        elif time == 0:
            if bitcast[DType.uint32](dst[index]) != 0x7FC00000:
                fp.unsafe_store(index, 1)
            return
        elif time < n:
            var phi = Float64(pp[series * width])
            expected = Float64(yp[series * n + time - 1])
            if time > 1:
                expected += phi * (Float64(yp[series * n + time - 1]) - Float64(yp[series * n + time - 2]))
        else:
            var phi = Float64(pp[series * width])
            expected = Float64(yp[series * n + n - 1])
            var difference = Float64(yp[series * n + n - 1]) - Float64(yp[series * n + n - 2])
            for step in range(time - n + 1):  # Geometric decay of the last observed first difference.
                difference *= phi
                expected += difference
        if bitcast[DType.uint32](dst[index]) != bitcast[DType.uint32](Float32(expected)):
            fp.unsafe_store(index, 1)
    host_parallelize(cell, target.count)
    var failed = 0
    for value in failure:
        failed += value
    require(failed == 0, "ARIMA independent closed form differs in " + String(failed) + " cells")
    _ = state
    _ = target


def prepare_state[K: ArimaCalls](fixture_id: Int) raises -> ArimaState:
    var state = ArimaState(fixture_id)
    if fixture_id < 2:
        require(K.fit(state) == state.batch * state.complexity, "ARIMA native MLE fit result count")
    state.guards()
    return state^


def per_call[K: ArimaCalls](state: ArimaState, future: ArimaWords, target: ArimaWords, start: Int, end: Int, forecast: Bool) raises:
    var fa = future.address() if future.count else 0
    var count = K.forecast(state, fa, target.address(), end - start) if forecast else K.predict(state, fa, target.address(), start, end)
    require(count == state.batch * (end - start), "ARIMA query result count")
    target.guards()
    future.guards()
    check_closed_form(state, target, start, end)
    _ = state
    _ = future
    _ = target


# ---- The host column (was arima_state_cpu_suite.mojo) ----
# Importable native CPU ARIMA state suite; private state per parallel fixture.
struct HostArima(ArimaCalls):
    @staticmethod
    def fit(state: ArimaState) raises -> Int:
        return arima_host_fit_into(state.y.address(), state.exog_address(), f32_ptr(state.params.address()),
            f32_ptr(state.x.address()), f32_ptr(state.x0.address()), f32_ptr(state.stats.address()),
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(state.flags.unsafe_ptr()) + 4),
            state.batch, state.n, 1, state.d, state.q, 0, 0, 0, 0, state.k, state.exog_count, 5, 0, 0, 0)

    @staticmethod
    def predict(state: ArimaState, future: Int, target: Int, start: Int, end: Int) raises -> Int:
        var order = _order(1, state.d, state.q, 0, 0, 0, 0, state.k, state.exog_count)
        if start == state.n:
            return _forecast_into(state.y.address(), state.exog_address(), future, state.params.address(),
                f32_ptr(target), state.batch, state.n, end - start, order)
        return _predict_in_sample_into(state.y.address(), state.exog_address(), future, state.params.address(),
            f32_ptr(target), state.batch, state.n, start, end, order)

    @staticmethod
    def forecast(state: ArimaState, future: Int, target: Int, steps: Int) raises -> Int:
        var order = _order(1, state.d, state.q, 0, 0, 0, 0, state.k, state.exog_count)
        return _forecast_into(state.y.address(), state.exog_address(), future, state.params.address(),
            f32_ptr(target), state.batch, state.n, steps, order)


def fixture(fixture_id: Int) raises:
    var state = prepare_state[HostArima](fixture_id)
    var saved_y = state.y.words.copy()
    var saved_exog = state.exog.words.copy()
    var saved_params = state.params.words.copy()
    for query_id in range(ARIMA_STATE_QUERIES):  # Dependent queries over one fitted state.
        var bounds = query(state, query_id)
        var future = future_words(state, bounds[1])
        var saved_future = future.words.copy()
        var result = ArimaWords(state.batch * (bounds[1] - bounds[0]))
        per_call[HostArima](state, future, result, bounds[0], bounds[1], bounds[2])
        var repeated = ArimaWords(result.count)
        # Predict(n,n+h) and forecast(h) share an end-exclusive public contract.
        require(HostArima.predict(state, future.address() if future.count else 0, repeated.address(),
            bounds[0], bounds[1]) == result.count, "host ARIMA repeated query count")
        same_words(repeated, result.words, "host ARIMA repeated fitted-state query")
        same_words(state.y, saved_y, "host ARIMA changed observations")
        same_words(state.exog, saved_exog, "host ARIMA changed fitted regressors")
        same_words(state.params, saved_params, "host ARIMA changed fitted parameters")
        same_words(future, saved_future, "host ARIMA changed future regressors")
        _ = future
        _ = result
        _ = repeated
    state.guards()
    _ = state


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(ARIMA_STATE_FIXTURES)
    def task(fixture_id: Int) {imm cases}:
        try:
            fixture(fixture_id)
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, ARIMA_STATE_FIXTURES)


# ---- The device column (was arima_state_gpu_suite.mojo) ----
# Importable ARIMA resident owner suite; ordered shared GPU context submission.
struct DeviceArima(ArimaCalls):
    @staticmethod
    def fit(state: ArimaState) raises -> Int:
        return arima_fit_ptr_host(f32_ptr(state.y.address()), state.exog_address(), f32_ptr(state.params.address()),
            f32_ptr(state.x.address()), f32_ptr(state.x0.address()), f32_ptr(state.stats.address()),
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(state.flags.unsafe_ptr()) + 4),
            state.batch, state.n, 1, state.d, state.q, 0, 0, 0, 0, state.k, state.exog_count, 0, 5)

    @staticmethod
    def predict(state: ArimaState, future: Int, target: Int, start: Int, end: Int) raises -> Int:
        return arima_predict_ptr_host(f32_ptr(state.y.address()), state.exog_address(), future,
            f32_ptr(state.params.address()), f32_ptr(target), state.batch, state.n, start, end,
            1, state.d, state.q, 0, 0, 0, 0, state.k, state.exog_count)

    @staticmethod
    def forecast(state: ArimaState, future: Int, target: Int, steps: Int) raises -> Int:
        return arima_forecast_ptr_host(f32_ptr(state.y.address()), state.exog_address(), future,
            f32_ptr(state.params.address()), f32_ptr(target), state.batch, state.n, steps,
            1, state.d, state.q, 0, 0, 0, 0, state.k, state.exog_count)


def fixture_gpu(fixture_id: Int) raises:
    var state = prepare_state[DeviceArima](fixture_id)
    var saved_y = state.y.words.copy()
    var saved_exog = state.exog.words.copy()
    var saved_params = state.params.words.copy()
    var count_before = arima_resident_count()
    var handle = arima_resident_prepare(f32_ptr(state.y.address()), state.exog_address(), f32_ptr(state.params.address()),
        state.batch, state.n, 1, state.d, state.q, 0, 0, 0, 0, state.k, state.exog_count)
    var expected = List[ArimaWords]()
    try:
        require(arima_resident_count() == count_before + 1, "ARIMA prepare did not acquire exactly one owner")
        for query_id in range(ARIMA_STATE_QUERIES):  # Dependent queries over one resident snapshot.
            var bounds = query(state, query_id)
            var future = future_words(state, bounds[1])
            var saved_future = future.words.copy()
            var result = ArimaWords(state.batch * (bounds[1] - bounds[0]))
            per_call[DeviceArima](state, future, result, bounds[0], bounds[1], bounds[2])
            var resident = ArimaWords(result.count)
            var written = arima_predict_resident(handle, future.address() if future.count else 0, f32_ptr(resident.address()),
                bounds[0], bounds[1] - bounds[0] if bounds[2] else bounds[1], bounds[2], "arima_forecast" if bounds[2] else "arima_predict")
            require(written == result.count, "resident ARIMA query count")
            same_words(resident, result.words, "resident/per-call ARIMA fitted-state parity")
            same_words(state.y, saved_y, "GPU ARIMA changed observations")
            same_words(state.exog, saved_exog, "GPU ARIMA changed fitted regressors")
            same_words(state.params, saved_params, "GPU ARIMA changed fitted parameters")
            same_words(future, saved_future, "GPU ARIMA changed future regressors")
            require(arima_resident_count() == count_before + 1, "changing horizon allocated another ARIMA owner")
            expected.append(result^)
            _ = resident
            _ = future
        # Resident state owns its upload; mutating caller buffers cannot change
        # any later answer, including the fitted regressor and parameter owners.
        state.y.fill(-99)
        state.exog.fill(-98)
        state.params.fill(-97)
        for query_id in range(ARIMA_STATE_QUERIES):
            var bounds = query(state, query_id)
            var future = future_words(state, bounds[1])
            var result = ArimaWords(expected[query_id].count)
            require(arima_predict_resident(handle, future.address() if future.count else 0, f32_ptr(result.address()),
                bounds[0], bounds[1] - bounds[0] if bounds[2] else bounds[1], bounds[2], "arima_predict") == result.count,
                "retained ARIMA query count after source mutation")
            same_words(result, expected[query_id].words, "resident ARIMA borrowed mutated caller state")
            _ = result
            _ = future
        state.guards()
    finally:
        arima_resident_release(handle)
    arima_resident_release(handle)
    require(arima_resident_count() == count_before, "ARIMA owner release/double release leaked state")
    var target = ArimaWords(state.batch * 3)
    var untouched = target.words.copy()
    var refused = False
    try:
        _ = arima_predict_resident(handle, 0, f32_ptr(target.address()), 0, 3, False, "arima_predict")
    except error:
        refused = True
        require(String(error) == "unknown or released resident ARIMA handle", "released ARIMA refusal changed")
    require(refused, "released ARIMA owner remained queryable")
    same_words(target, untouched, "released ARIMA query wrote output")
    _ = state
    _ = target
    _ = expected


def _run_gpu(mut report: Report) raises:
    var cases = report.cases(ARIMA_STATE_FIXTURES)
    for fixture_id in range(ARIMA_STATE_FIXTURES):
        try:
            fixture_gpu(fixture_id)
        except error:
            cases.fail(fixture_id, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("arima_state", cpu=True, gpu=True, serial="", run=run_cases)
