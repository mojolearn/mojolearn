# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""DBSCAN predict over query rows already on the device (lane vdf-scope-cluster6: DBSCAN_RESIDENT_QUERY_STORE,
core/dbscan_native_device.mojo `resident_store`, core/labeled_reference_resident.mojo `src_rows`; the member's
`_predict_rows` on a model-selection fold's test rows taken into the estimators store).

Claims (device column; the host provider has no store and refuses the form):
  case 0 (euclidean), case 1 (manhattan): the labels of the store form over a slot holding the queries == the
  labels of the resident host-pointer form over the same rows (int32 words, eps 0.9, the threshold on);
  case 2 (cosine): the store form refuses (its unit rows are the host's), so the member keeps the gathered form.
Data: 4 reference rows (two per label), 12 query rows of a SplitMix64 stream (uniform [0, 4)), 3 features.
Sabotage: in core/labeled_reference_resident.mojo copy `n_queries * d - 1` words of `src_rows` -> case 0 fails.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from core.dbscan_native_api import DBSCANReferenceRequest, DBSCANResidentQuery
from core.dbscan_native_device import DBSCANDeviceKernels
from core.estimators_store import ESTIMATORS_STORE, estimators_store_ctx

comptime _D = 3
comptime _Q = 12


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _device_case(case: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var metric = 0 if case == 0 else (1 if case == 1 else 2)
        var refs: List[Float32] = [0.5, 0.5, 0.5, 0.9, 0.7, 0.4, 3.0, 3.2, 2.9, 3.4, 3.1, 3.3]
        var keys: List[Int32] = [0, 1, 2, 3]
        var labels: List[Int32] = [0, 0, 1, 1]
        var state = UInt64(20261124)
        var queries = List[Float32](capacity=_Q * _D)
        for _ in range(_Q * _D):  # small-loop(fixture words)
            queries.append(Float32(Float64(_next(state) >> 11) * (1.0 / 9007199254740992.0) * 4.0))
        var handle = DBSCANDeviceKernels.prepare(DBSCANReferenceRequest(Int(refs.unsafe_ptr()),
            Int(keys.unsafe_ptr()), Int(labels.unsafe_ptr()), 4, _D, metric))
        var ctx = estimators_store_ctx()
        var store = ESTIMATORS_STORE.get_or_create_ptr()
        var slot = store[].put(ctx, Int(queries.unsafe_ptr()), _Q * _D)
        var failure = String("")
        try:
            var host = List[Int32](length=_Q, fill=-7)
            var resident = List[Int32](length=_Q, fill=-7)
            if metric != 2:
                _ = DBSCANDeviceKernels.resident(DBSCANResidentQuery(Int(queries.unsafe_ptr()), Int(host.unsafe_ptr()),
                    handle, _Q, _D, metric, True, 0.9))
                _ = DBSCANDeviceKernels.resident_store(DBSCANResidentQuery(slot, Int(resident.unsafe_ptr()), handle,
                    _Q, _D, metric, True, 0.9))
                for i in range(_Q):  # small-loop(compared labels)
                    require(host[i] == resident[i], "label " + String(i) + " differs")
            else:
                var refused = False
                try:
                    _ = DBSCANDeviceKernels.resident_store(DBSCANResidentQuery(slot, Int(resident.unsafe_ptr()),
                        handle, _Q, _D, metric, True, 0.9))
                    _ = resident^
                except error:
                    refused = String(error).find("cosine") >= 0
                require(refused, "the cosine store form refuses")
        except error:
            failure = String(error)
        store[].free(ctx, slot)
        _ = DBSCANDeviceKernels.release(handle)
        _ = refs^
        _ = keys^
        _ = labels^
        _ = queries^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _device_case(case)

        check_cases_serial("dbscan_resident_store_query device", 3, body)


def main() raises:
    var suite = TestSuite()
    comptime if not CPU_COLUMN_BUILD:
        suite.test[test_device_fixtures]()
    suite^.run()
