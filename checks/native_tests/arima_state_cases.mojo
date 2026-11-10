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

comptime ARIMA_STATE_FIXTURES = 8
comptime ARIMA_STATE_QUERIES = 4
comptime ARIMA_GUARD = Float32(31415.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


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
