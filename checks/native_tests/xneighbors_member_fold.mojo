# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD X WORD of the XNeighbors member's NearestCentroid (lane vdf-scope-cluster3,
core/xneighbors_member.mojo `_nc_fold_op`; x_neighbors/device_ops.mojo `_buf`).

A model-selection fold's rows taken ON THE DEVICE into a slot of the x_prep store (what `member_fold_x_open`
does) and handed to the NearestCentroid RUN as a negative X word must give the SAME BYTES as the same rows
gathered on the host and handed as host words: every x_neighbors device op reads its X upload from the slot
(one device copy) instead of the host. Claims, per column:
  device: fit over the fold word == fit over host-gathered rows (centroids, std, deviations, priors bits);
          predict, decision and predict_proba over the test fold's word == over its host-gathered rows (codes and
          float32 bits), after a fit over host rows.
  host:   the host take of the fold (`member_fold_take_f32`, row chunks across the cores) == a serial gather, and
          the NearestCentroid RUN over it on the host entry fits (the CPU column of the verify digest).
Data: a SplitMix64 stream (uniform [0, 4)), 90 rows x 3; class codes i % 3; train fold the rows with i % 5 != 0
(in order, then reversed halves), test fold the rest.
Sabotage: in x_neighbors/device_ops.mojo `_buf` copy from the slot at word 1 -> device fit bits differ.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from max.gpu.host import DeviceContext
from core.member_fold_x import member_fold_take_f32
from core.fold_x_device import FOLD_X_STORE, fold_x_word
from x_neighbors.native_descriptor_host import xn_native_v1 as xn_host_v1
from x_neighbors.native_descriptor_device import xn_native_v1 as xn_device_v1
from core.neighbors_native_driver import XnDriver
from core.nearest_centroid_plan import NcLabelCodes, NcFitPlan
from neighbors.nearest_centroid_run import (
    NcFitRequest, NcQueryRequest, NC_PRIORS_UNIFORM, nc_fit_count_run, nc_fit_priors_run, nc_fit_run,
    nc_labels_run, nc_decision_run, nc_proba_run,
)

comptime _ROWS = 90
comptime _D = 3
comptime _CLASSES = 3


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _x() -> List[Float32]:
    var state = UInt64(20261124)
    var out = List[Float32](capacity=_ROWS * _D)
    for _ in range(_ROWS * _D):  # small-loop(fixture words)
        out.append(Float32(Float64(_next(state) >> 11) * (1.0 / 9007199254740992.0) * 4.0))
    return out^


def _codes() -> List[Int32]:
    var out = List[Int32](capacity=_ROWS)
    for i in range(_ROWS):  # small-loop(fixture rows)
        out.append(Int32(i % _CLASSES))
    return out^


def _index(train: Bool) -> List[Int64]:
    var rows = List[Int64]()
    for i in range(_ROWS):  # small-loop(fixture rows)
        if (i % 5 != 0) == train:
            rows.append(Int64(i))
    # Not in base order: the second half first (a shuffled fold's shape).
    var n = len(rows)
    var out = List[Int64](capacity=n)
    for i in range(n // 2, n):  # small-loop(fixture rows)
        out.append(rows[i])
    for i in range(n // 2):  # small-loop(fixture rows)
        out.append(rows[i])
    return out^


def _serial(x: List[Float32], index: List[Int64], width: Int) -> List[Float32]:
    var out = List[Float32](capacity=len(index) * width)
    for r in range(len(index)):  # small-loop(oracle rows)
        for c in range(width):  # small-loop(oracle row words)
            out.append(x[Int(index[r]) * width + c])
    return out^


def _take_codes(codes: List[Int32], index: List[Int64]) -> List[Int32]:
    var out = List[Int32](capacity=len(index))
    for r in range(len(index)):  # small-loop(oracle rows)
        out.append(codes[Int(index[r])])
    return out^


struct _Fit(Movable):
    var prior: List[Float64]
    var std: List[Float32]
    var centroids: List[Float32]
    var deviations: List[Float32]

    def __init__(out self):
        self.prior = List[Float64](length=_CLASSES, fill=0.0)
        self.std = List[Float32](length=_D, fill=0.0)
        self.centroids = List[Float32](length=_CLASSES * _D, fill=0.0)
        self.deviations = List[Float32](length=_CLASSES * _D, fill=0.0)


def _fit(driver: XnDriver, x_word: Int, labels: Int, rows: Int) raises -> _Fit:
    """The member's `_nc_fit` steps (uniform priors, euclidean, no shrink)."""
    var fit = _Fit()
    var request = NcFitRequest(x_word, labels, rows, _D, _CLASSES, True, NC_PRIORS_UNIFORM, 0.0)
    request.prior = Int(fit.prior.unsafe_ptr())
    request.stats = Int(fit.std.unsafe_ptr())
    request.centroids = Int(fit.centroids.unsafe_ptr())
    request.deviations = Int(fit.deviations.unsafe_ptr())
    var plan = NcFitPlan(rows, _D, _CLASSES)
    nc_fit_count_run(driver, request, plan)
    nc_fit_priors_run(request)
    var outcome = nc_fit_run(driver, request, plan)
    require(outcome.status == 0, "the fit refused finite rows")
    _ = plan^
    return fit^


def _same_f32(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths")
    for i in range(len(a)):  # small-loop(compared words)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": bits differ at " + String(i))


def _same_fit(a: _Fit, b: _Fit) raises:
    _same_f32(a.std, b.std, "std")
    _same_f32(a.centroids, b.centroids, "centroids")
    _same_f32(a.deviations, b.deviations, "deviations")
    for i in range(_CLASSES):  # small-loop(priors)
        require(a.prior[i].to_bits() == b.prior[i].to_bits(), "priors")


def _host_case(case: Int) raises:
    comptime if CPU_COLUMN_BUILD:
        var x = _x()
        var codes = _codes()
        var index = _index(True)
        var n = len(index)
        var taken = List[Float32](length=n * _D, fill=0.0)
        member_fold_take_f32(Int(x.unsafe_ptr()), Int(index.unsafe_ptr()), n, _D, Int(taken.unsafe_ptr()))
        _same_f32(taken, _serial(x, index, _D), "host take of the fold")
        var labels = _take_codes(codes, index)
        var fit = _fit(XnDriver(xn_host_v1), Int(taken.unsafe_ptr()), Int(labels.unsafe_ptr()), n)
        _ = fit^
        _ = x^
        _ = codes^
        _ = labels^


def _device_case(case: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var driver = XnDriver(xn_device_v1)
        var x = _x()
        var codes = _codes()
        var train = _index(True)
        var test = _index(False)
        var nt = len(train)
        var nq = len(test)
        var train_rows = _serial(x, train, _D)
        var test_rows = _serial(x, test, _D)
        var labels = _take_codes(codes, train)
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), _ROWS * _D)
        var train_slot = store[].take_rows(store_ctx, base, _D, Int(train.unsafe_ptr()), nt)
        var test_slot = store[].take_rows(store_ctx, base, _D, Int(test.unsafe_ptr()), nq)
        var failure = String("")
        try:
            var host_fit = _fit(driver, Int(train_rows.unsafe_ptr()), Int(labels.unsafe_ptr()), nt)
            if case == 0:
                var fold_fit = _fit(driver, fold_x_word(train_slot), Int(labels.unsafe_ptr()), nt)
                _same_fit(fold_fit, host_fit)
            else:
                var queries: List[Int] = [fold_x_word(test_slot), Int(test_rows.unsafe_ptr())]
                var got_codes = List[List[Int32]]()
                var got_decision = List[List[Float32]]()
                var got_proba = List[List[Float32]]()
                for q in range(2):  # small-loop(the fold word, then the host rows)
                    var request = NcQueryRequest(queries[q], Int(host_fit.centroids.unsafe_ptr()),
                        Int(host_fit.std.unsafe_ptr()), Int(host_fit.prior.unsafe_ptr()), nq, _D, _CLASSES,
                        True, True, False)
                    var out_codes = NcLabelCodes(nq)
                    require(nc_labels_run(driver, request, out_codes) == 0, "predict refused finite rows")
                    got_codes.append(out_codes.values.copy())
                    var decision = List[Float32](length=nq * _CLASSES, fill=0.0)
                    require(nc_decision_run(driver, request, Int(decision.unsafe_ptr())) == 0,
                            "decision refused finite rows")
                    got_decision.append(decision^)
                    var proba = List[Float32](length=nq * _CLASSES, fill=0.0)
                    require(nc_proba_run(driver, request, Int(proba.unsafe_ptr()), False) == 0,
                            "predict_proba refused finite rows")
                    got_proba.append(proba^)
                for i in range(nq):  # small-loop(test rows)
                    require(got_codes[0][i] == got_codes[1][i], "predicted codes")
                _same_f32(got_decision[0], got_decision[1], "decision")
                _same_f32(got_proba[0], got_proba[1], "predict_proba")
                _ = queries^
        except error:
            failure = String(error)
        store[].free(store_ctx, test_slot)
        store[].free(store_ctx, train_slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = codes^
        _ = train^
        _ = test^
        _ = train_rows^
        _ = test_rows^
        _ = labels^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _host_case(case)

        check_cases_serial("xneighbors_member_fold host", 1, body)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _device_case(case)

        check_cases_serial("xneighbors_member_fold device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
