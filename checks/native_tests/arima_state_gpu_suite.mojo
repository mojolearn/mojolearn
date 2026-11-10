# SPDX-License-Identifier: Apache-2.0
"""Importable ARIMA resident owner suite; ordered shared GPU context submission."""
from core.hostptr import f32_ptr
from arima.estimator import arima_fit_ptr_host, arima_predict_ptr_host, arima_forecast_ptr_host
from arima.resident_state import arima_resident_prepare, arima_resident_count, arima_resident_release, arima_predict_resident
from checks.native_tests.arima_state_cases import (
    ARIMA_STATE_FIXTURES, ARIMA_STATE_QUERIES, ArimaCalls, ArimaState, ArimaWords,
    require, same_words, query, future_words, prepare_state, per_call,
)


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


def fixture(fixture_id: Int) raises:
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


def run_checks() raises:
    var failed = 0
    for fixture_id in range(ARIMA_STATE_FIXTURES):
        try:
            fixture(fixture_id)
        except error:
            failed += 1
            print("FAIL arima_state_gpu fixture", fixture_id, String(error))
    require(failed == 0, "arima_state_gpu failed " + String(failed) + " fixtures")
    print("PASS arima_state_gpu:", ARIMA_STATE_FIXTURES, "resident owners x", ARIMA_STATE_QUERIES, "queries")
