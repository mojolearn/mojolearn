# SPDX-License-Identifier: Apache-2.0
"""Importable native CPU ARIMA state suite; private state per parallel fixture."""
from core.hostptr import f32_ptr
from core.host_parallel import host_parallelize
from arima.host.arima_host_entries import _order, _forecast_into, _predict_in_sample_into, arima_host_fit_into
from checks.native_tests.arima_state_cases import (
    ARIMA_STATE_FIXTURES, ARIMA_STATE_QUERIES, ArimaCalls, ArimaState, ArimaWords,
    require, same_words, query, future_words, prepare_state, per_call,
)


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


def run_checks() raises:
    var failures = List[Int](length=ARIMA_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            fixture(fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL arima_state_cpu fixture", fixture_id, String(error))
    host_parallelize(task, ARIMA_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "arima_state_cpu failed " + String(failed) + " fixtures")
    print("PASS arima_state_cpu:", ARIMA_STATE_FIXTURES, "parallel fitted-state fixtures x", ARIMA_STATE_QUERIES, "queries")
