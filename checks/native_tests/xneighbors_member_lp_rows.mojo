# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""LabelPropagation / LabelSpreading over a model-selection fold's rows on the device (lane vdf-scope-cluster6,
core/xneighbors_member.mojo `_rows_fold_op` for the LP kinds: THE FOLD X WORD and THE FITTED-ROWS SLOT).

The member's fold fit hands `lp_fit_run` the fold slot as a negative X word and keeps that slot as the fitted
`X_`; every later query hands `lp_query_run` the query fold's slot word and the fitted slot word. Claims, per column:
  device: (case 0, LabelPropagation kNN graph, k = 7; case 1, LabelSpreading RBF kernel, gamma 20, the
          operation chain) the fit over the fold word == the fit over the host-gathered rows (label
          distributions bits, transduction codes, n_iter_); predict's codes and predict_proba's float32 words
          of the test fold, with the query AND the fitted rows as slot words == with both as host rows.
  host:   the same fits over the host take of the fold on the host entry (the CPU column of the verify digest),
          and the take == a serial gather.
Data: a SplitMix64 stream (uniform [0, 4)), 90 rows x 3; codes i % 3, the rows with i % 7 == 0 unlabeled (code
3, the marker word); train fold the rows with i % 5 != 0 (second half first), test fold the rest.
Sabotage: in x_neighbors/device_ops.mojo `_buf` pass `count - 1` to `fold_x_fill` for a negative address -> the
fold-fit claim fails (the last fitted word is uninitialized).
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from max.gpu.host import DeviceContext
from core.member_fold_x import member_fold_take_f32, member_fold_take_words
from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_word
from x_neighbors.native_descriptor_host import xn_native_v1 as xn_host_v1
from x_neighbors.native_descriptor_device import xn_native_v1 as xn_device_v1
from core.neighbors_native_driver import XnDriver
from neighbors.label_propagation_run import LpFitRequest, LpQueryRequest, lp_fit_run, lp_query_run
from neighbors.neighbors_semi_run import xn_semi_knn_k, xn_lp_static_weight
from neighbors.neighbors_graph_run import xn_f32_word

comptime _ROWS = 90
comptime _D = 3
comptime _C = 3
comptime _MARKER = 3
comptime _RBF = 2   # python/mojolearn/_expansion_neighbors.py `_KERNELS["rbf"]`


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
        out.append(Int32(_MARKER if i % 7 == 0 else i % _C))
    return out^


def _index(train: Bool) -> List[Int64]:
    var rows = List[Int64]()
    for i in range(_ROWS):  # small-loop(fixture rows)
        if (i % 5 != 0) == train:
            rows.append(Int64(i))
    var n = len(rows)
    var out = List[Int64](capacity=n)
    for i in range(n // 2, n):  # small-loop(fixture rows)
        out.append(rows[i])
    for i in range(n // 2):  # small-loop(fixture rows)
        out.append(rows[i])
    return out^


def _serial(x: List[Float32], index: List[Int64]) -> List[Float32]:
    var out = List[Float32](capacity=len(index) * _D)
    for r in range(len(index)):  # small-loop(oracle rows)
        for c in range(_D):  # small-loop(oracle row words)
            out.append(x[Int(index[r]) * _D + c])
    return out^


def _serial_codes(codes: List[Int32], index: List[Int64]) -> List[Int32]:
    var out = List[Int32](capacity=len(index))
    for r in range(len(index)):  # small-loop(oracle rows)
        out.append(codes[Int(index[r])])
    return out^


struct _Fit(Movable):
    var dist: List[Float32]
    var codes: List[Int32]
    var iterations: Int
    var k: Int

    def __init__(out self, rows: Int):
        self.dist = List[Float32](length=rows * _C, fill=0.0)
        self.codes = List[Int32](length=rows, fill=0)
        self.iterations = 0
        self.k = 0


def _fit(driver: XnDriver, x_word: Int, codes: List[Int32], rows: Int, unit: Int) raises -> _Fit:
    """The member's `_lp_fit` RUN with the defaults' words (every arm off)."""
    var spreading = unit == 1
    var knn = unit == 0
    var fit = _Fit(rows)
    fit.k = xn_semi_knn_k(7, rows) if knn else 0
    var weight = xn_lp_static_weight(0.2) if spreading else 1.0
    var alpha = xn_f32_word(0.2) if spreading else 0.0
    var request = LpFitRequest(x_word, Int(codes.unsafe_ptr()), Int(fit.dist.unsafe_ptr()),
        Int(fit.codes.unsafe_ptr()), rows, _D, _C, _MARKER, spreading, knn, fit.k, 0 if knn else _RBF,
        0.0 if knn else 20.0, False, weight, alpha, 0.001, 30 if spreading else 1000, False, False)
    var outcome = lp_fit_run(driver, request)
    require(outcome.status == 0, "the fit refused finite rows")
    fit.iterations = outcome.iterations
    _ = codes
    return fit^


def _query(driver: XnDriver, query: Int, fit_x: Int, fit: _Fit, nq: Int, nt: Int, unit: Int,
           labels: Bool) raises -> List[Float32]:
    """predict (codes as float32 words) or predict_proba of the member's `_lp_request`."""
    var knn = unit == 0
    var proba = List[Float32](length=nq * _C, fill=0.0)
    var predicted = List[Int32](length=nq, fill=0)
    var request = LpQueryRequest(query, fit_x, Int(fit.dist.unsafe_ptr()), 0 if labels else Int(proba.unsafe_ptr()),
        Int(predicted.unsafe_ptr()) if labels else 0, nq, nt, _D, _C, knn, fit.k, 0 if knn else _RBF,
        0.0 if knn else 20.0, False, labels, False, False)
    require(lp_query_run(driver, request) == 0, "the query refused finite rows")
    if not labels:
        return proba^
    var out = List[Float32](capacity=nq)
    for i in range(nq):  # small-loop(compared codes)
        out.append(Float32(predicted[i]))
    return out^


def _same_f32(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths")
    for i in range(len(a)):  # small-loop(compared words)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": bits differ at " + String(i))


def _same_fit(a: _Fit, b: _Fit) raises:
    _same_f32(a.dist, b.dist, "label distributions")
    require(len(a.codes) == len(b.codes), "transduction lengths")
    for i in range(len(a.codes)):  # small-loop(compared codes)
        require(a.codes[i] == b.codes[i], "transduction differs at " + String(i))
    require(a.iterations == b.iterations, "n_iter_")


def _host_case(unit: Int) raises:
    comptime if CPU_COLUMN_BUILD:
        var x = _x()
        var codes = _codes()
        var index = _index(True)
        var n = len(index)
        var taken = List[Float32](length=n * _D, fill=0.0)
        member_fold_take_f32(Int(x.unsafe_ptr()), Int(index.unsafe_ptr()), n, _D, Int(taken.unsafe_ptr()))
        _same_f32(taken, _serial(x, index), "host take of the fold")
        var fold_codes = List[Int32](length=n, fill=0)
        member_fold_take_words(Int(codes.unsafe_ptr()), Int(index.unsafe_ptr()), n, 1, Int(fold_codes.unsafe_ptr()), 4)
        var want = _serial_codes(codes, index)
        for i in range(n):  # small-loop(compared codes)
            require(fold_codes[i] == want[i], "host take of the fold's codes")
        var fit = _fit(XnDriver(xn_host_v1), Int(taken.unsafe_ptr()), fold_codes, n, unit)
        _ = fit^
        _ = x^
        _ = codes^
        _ = taken^


def _device_case(unit: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var driver = XnDriver(xn_device_v1)
        var x = _x()
        var codes = _codes()
        var train = _index(True)
        var test = _index(False)
        var nt = len(train)
        var nq = len(test)
        var train_rows = _serial(x, train)
        var test_rows = _serial(x, test)
        var train_codes = _serial_codes(codes, train)
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), _ROWS * _D)
        var train_slot = store[].take_rows(store_ctx, base, _D, Int(train.unsafe_ptr()), nt)
        var test_slot = store[].take_rows(store_ctx, base, _D, Int(test.unsafe_ptr()), nq)
        var failure = String("")
        try:
            var host_fit = _fit(driver, Int(train_rows.unsafe_ptr()), train_codes, nt, unit)
            var fold_fit = _fit(driver, fold_x_word(train_slot), train_codes, nt, unit)
            _same_fit(fold_fit, host_fit)
            for which in range(2):  # small-loop(predict, then predict_proba)
                var labels = which == 0
                var resident = _query(driver, fold_x_word(test_slot), fold_x_word(train_slot), fold_fit, nq, nt,
                                      unit, labels)
                var host = _query(driver, Int(test_rows.unsafe_ptr()), Int(train_rows.unsafe_ptr()), host_fit,
                                  nq, nt, unit, labels)
                _same_f32(resident, host, "query over the resident fitted rows")
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
        _ = train_codes^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(unit: Int) raises:
            _host_case(unit)

        check_cases_serial("xneighbors_member_lp_rows host", 2, body)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(unit: Int) raises:
            _device_case(unit)

        check_cases_serial("xneighbors_member_lp_rows device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
