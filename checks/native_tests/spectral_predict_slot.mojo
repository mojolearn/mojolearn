# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""SpectralClustering.predict over rows already on the device (lane vdf-scope-cluster8: spectral/impl/spectral_predict.mojo
`spectral_predict_run`'s slot form with `knn_search_slot_queries`, spectral/impl/spectral_predict_resident.mojo
`spectral_resident_prepare_slot`, the SPECTRAL_CAP_DEV_QUERY prepare / query_resident words of
core/spectral_native_execution.mojo, the member's THE FITTED-ROWS SLOT in core/spectral_member.mojo).

Claims (device column; the host image has no store and refuses the slot words), one fitted state (the dataset
route, nearest-neighbors affinity, prediction data kept):
  case 0: training rows in a metrics-store slot, queries in a second slot == the per-call host-input predict
          (`spectral_predict_host`) == the host rule (`host_spectral_predict`, the CPU column's spelling): every
          label and every float32 bit of the Nystrom embedding.
  case 1: training rows in a slot, queries as host words (one upload) == the same two references.
  case 2: the host-rows resident handle (`spectral_resident_prepare`) == the same references (the old resident
          path, unchanged).
Data: a SplitMix64 stream (uniform [0, 4)), 60 training rows x 3, 17 queries x 3; 3 clusters, 3 components,
n_init 2, n_neighbors 6, seed 11, tolerance 1e-5.
Sabotage: in spectral/impl/spectral_predict.mojo pass `n_neighbors - 1` to `knn_search_slot_queries` -> cases 0 and 1
fail; drop the device-to-device copy in `spectral_resident_prepare_slot` (read the slot directly, then free it before
the predict) -> case 0 fails on a store with the spare pool.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from core.metrics_store import METRICS_STORE, metrics_store_ctx, metrics_store_buffer
from spectral.estimator import spectral_fit_predict_dataset_host_keep, spectral_predict_host
from spectral.host.spectral_predict_host import host_spectral_predict
from spectral.impl.spectral_predict_common import SpectralPredictionState, SpectralPrediction
from spectral.impl.spectral_predict_resident import (
    spectral_resident_prepare, spectral_resident_prepare_slot, spectral_resident_predict, spectral_resident_release,
)

comptime _N = 60
comptime _Q = 17
comptime _D = 3
comptime _C = 3
comptime _K = 3
comptime _NB = 6


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _rows(mut state: UInt64, n: Int) -> List[Float32]:
    var x = List[Float32](capacity=n * _D)
    for _ in range(n * _D):  # small-loop(fixture words)
        x.append(Float32(Float64(_next(state) >> 11) * (1.0 / 9007199254740992.0) * 4.0))
    return x^


def _same(a: SpectralPrediction, b: SpectralPrediction, what: String) raises:
    require(len(a.labels) == len(b.labels) and len(a.embedding) == len(b.embedding), what + ": output lengths")
    for i in range(len(a.labels)):  # small-loop(compared labels)
        require(a.labels[i] == b.labels[i], what + ": label differs at " + String(i))
    for i in range(len(a.embedding)):  # small-loop(compared words)
        require(bitcast[DType.uint32](a.embedding[i]) == bitcast[DType.uint32](b.embedding[i]),
                what + ": embedding bits differ at " + String(i))


def _device_case(case: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var seed = UInt64(20261124)
        var train = _rows(seed, _N)
        var queries = _rows(seed, _Q)
        var labels = List[Int32]()
        var embedding = List[Float32]()
        var state = SpectralPredictionState()
        var cols = spectral_fit_predict_dataset_host_keep(train, _N, _D, _C, _K, 2, _NB, Float32(1e-5), UInt64(11),
                                                          labels, embedding, state, True)
        require(cols == _K, "fit column count")
        var per_call = spectral_predict_host(queries, train, _N, _Q, _D, _K, _C, _NB, 0, state)
        var rule = host_spectral_predict(queries, train, _N, _Q, _D, _K, _C, _NB, 0, state)
        _same(per_call, rule, "per-call device == host rule")
        var ctx = metrics_store_ctx()
        var store = METRICS_STORE.get_or_create_ptr()
        var t_slot = store[].put(ctx, Int(train.unsafe_ptr()), _N * _D)
        var q_slot = store[].put(ctx, Int(queries.unsafe_ptr()), _Q * _D)
        var handle = -1
        var failure = String("")
        try:
            if case == 2:
                var own = train.copy()
                handle = spectral_resident_prepare(own^, _N, _D, _K, _C, 0, state)
                var got = spectral_resident_predict(handle, queries, _N, _Q, _D, _K, _C, _NB, 0)
                _same(got, per_call, "host-rows resident == per-call")
            else:
                handle = spectral_resident_prepare_slot(metrics_store_buffer(t_slot, _N * _D), _N, _D, _K, _C, 0,
                                                        state)
                # The handle holds its own copy: the training slot may go before the predict.
                store[].free(ctx, t_slot)
                t_slot = -1
                if case == 0:
                    var qbuf = metrics_store_buffer(q_slot, _Q * _D)
                    var got = spectral_resident_predict(handle, List[Float32](), _N, _Q, _D, _K, _C, _NB, 0,
                                                        Optional(qbuf.copy()))
                    _ = qbuf^
                    _same(got, per_call, "slot training + slot queries == per-call")
                    _same(got, rule, "slot training + slot queries == host rule")
                else:
                    var got = spectral_resident_predict(handle, queries, _N, _Q, _D, _K, _C, _NB, 0)
                    _same(got, per_call, "slot training + host queries == per-call")
                    _same(got, rule, "slot training + host queries == host rule")
        except error:
            failure = String(error)
        if handle >= 0:
            spectral_resident_release(handle)
        if t_slot >= 0:
            store[].free(ctx, t_slot)
        store[].free(ctx, q_slot)
        _ = train^
        _ = queries^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _device_case(case)

        check_cases_serial("spectral_predict_slot device", 3, body)


def main() raises:
    var suite = TestSuite()
    comptime if not CPU_COLUMN_BUILD:
        suite.test[test_device_fixtures]()
    suite^.run()
