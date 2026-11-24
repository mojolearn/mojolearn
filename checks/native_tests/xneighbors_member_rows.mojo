# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FITTED-ROWS SLOT of the XNeighbors member (lane vdf-scope-cluster5, core/xneighbors_member.mojo
`_rows_fold_op`, `_rows_dev_download`; x_neighbors/device_ops.mojo `_buf`, x_neighbors/iter_device.mojo
`op_knn_sq_tiled`).

A LocalOutlierFactor fold fit keeps its fold's rows resident in the x_prep store slot it took them into, and every
later query reads the fitted rows from that slot (a negative X word). Claims, per column:
  device: (case 0, fused tiled k-NN; case 1, the old-items `knn_sq`) the fit over the fold word == the fit over
          the host-gathered rows (root distances, densities, scores bits, offset bits); score_samples and
          decision_function of the test fold, with the query AND the fitted rows as slot words == with both as
          host rows (float32 bits); OneClassSVM's support-vector gather over the fold word == the host rows'
          words; the export's download of the resident rows (`take_rows` with the identity
          index over the slot word) == the host-gathered rows (bits).
  host:   the host take of the fold == a serial gather, and the same LocalOutlierFactor RUN over it on the host
          entry (the CPU column of the verify digest).
Data: a SplitMix64 stream (uniform [0, 4)), 90 rows x 3; train fold the rows with i % 5 != 0 (second half first),
test fold the rest; k = 7, contamination 0.1 (not auto).
Sabotage: in core/xneighbors_member.mojo `_rows_dev_download` pass `rows - 1` as n_out -> the export claim fails.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from max.gpu.host import DeviceContext
from core.member_fold_x import member_fold_take_f32
from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_word
from x_neighbors.native_descriptor_host import xn_native_v1 as xn_host_v1
from x_neighbors.native_descriptor_device import xn_native_v1 as xn_device_v1
from core.neighbors_native_driver import XnDriver
from neighbors.lof_run import LofFitRequest, LofQueryRequest, lof_fit_run, lof_query_run
from neighbors.neighbors_local_run import LOF_RESULT_SCORES, LOF_RESULT_DECISION

comptime _ROWS = 90
comptime _D = 3
comptime _K = 7


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


struct _Fit(Movable):
    var distances: List[Float32]
    var densities: List[Float32]
    var scores: List[Float32]
    var offset: Float64

    def __init__(out self, rows: Int):
        self.distances = List[Float32](length=rows * _K, fill=0.0)
        self.densities = List[Float32](length=rows, fill=0.0)
        self.scores = List[Float32](length=rows, fill=0.0)
        self.offset = 0.0


def _fit(driver: XnDriver, x_word: Int, rows: Int, old_items: Bool) raises -> _Fit:
    """The member's `_lof_fit` RUN (contamination 0.1)."""
    var fit = _Fit(rows)
    var request = LofFitRequest(x_word, rows, _D, _K, False, old_items, False, 0.1)
    request.distances = Int(fit.distances.unsafe_ptr())
    request.densities = Int(fit.densities.unsafe_ptr())
    request.scores = Int(fit.scores.unsafe_ptr())
    var outcome = lof_fit_run(driver, request)
    require(outcome.status == 0, "the fit refused finite rows")
    fit.offset = outcome.offset
    return fit^


def _query(driver: XnDriver, query: Int, fit_x: Int, fit: _Fit, nq: Int, nt: Int, old_items: Bool,
           kind: Int) raises -> List[Float32]:
    """The member's `_lof_query` RUN (score_samples or decision_function)."""
    var out = List[Float32](length=nq, fill=0.0)
    var request = LofQueryRequest(query, fit_x, Int(fit.distances.unsafe_ptr()), Int(fit.densities.unsafe_ptr()),
        nq, nt, _D, _K, nt, False, old_items, kind, 0.0 if kind == LOF_RESULT_SCORES else -fit.offset)
    request.dst = Int(out.unsafe_ptr())
    require(lof_query_run(driver, request) == 0, "the query refused finite rows")
    return out^


def _same_f32(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths")
    for i in range(len(a)):  # small-loop(compared words)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": bits differ at " + String(i))


def _same_fit(a: _Fit, b: _Fit) raises:
    _same_f32(a.distances, b.distances, "root distances")
    _same_f32(a.densities, b.densities, "densities")
    _same_f32(a.scores, b.scores, "scores")
    require(a.offset.to_bits() == b.offset.to_bits(), "offset bits")


def _host_case(case: Int) raises:
    comptime if CPU_COLUMN_BUILD:
        var x = _x()
        var index = _index(True)
        var n = len(index)
        var taken = List[Float32](length=n * _D, fill=0.0)
        member_fold_take_f32(Int(x.unsafe_ptr()), Int(index.unsafe_ptr()), n, _D, Int(taken.unsafe_ptr()))
        _same_f32(taken, _serial(x, index), "host take of the fold")
        var fit = _fit(XnDriver(xn_host_v1), Int(taken.unsafe_ptr()), n, case == 1)
        _ = fit^
        _ = x^
        _ = taken^


def _device_case(case: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var driver = XnDriver(xn_device_v1)
        var old_items = case == 1
        var x = _x()
        var train = _index(True)
        var test = _index(False)
        var nt = len(train)
        var nq = len(test)
        var train_rows = _serial(x, train)
        var test_rows = _serial(x, test)
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), _ROWS * _D)
        var train_slot = store[].take_rows(store_ctx, base, _D, Int(train.unsafe_ptr()), nt)
        var test_slot = store[].take_rows(store_ctx, base, _D, Int(test.unsafe_ptr()), nq)
        var failure = String("")
        try:
            var host_fit = _fit(driver, Int(train_rows.unsafe_ptr()), nt, old_items)
            var fold_fit = _fit(driver, fold_x_word(train_slot), nt, old_items)
            _same_fit(fold_fit, host_fit)
            for kind in range(2):  # small-loop(score_samples, then decision_function)
                var which = LOF_RESULT_SCORES if kind == 0 else LOF_RESULT_DECISION
                var resident = _query(driver, fold_x_word(test_slot), fold_x_word(train_slot), fold_fit, nq, nt,
                                      old_items, which)
                var host = _query(driver, Int(test_rows.unsafe_ptr()), Int(train_rows.unsafe_ptr()), host_fit,
                                  nq, nt, old_items, which)
                _same_f32(resident, host, "query over the resident fitted rows")
            # The export's download: `take_rows` with the identity index over the slot word.
            var identity = List[Int32](capacity=nt)
            for r in range(nt):  # small-loop(fixture rows)
                identity.append(Int32(r))
            var down = List[Float32](length=nt * _D, fill=0.0)
            var a: Array[Int, 3] = [fold_x_word(train_slot), Int(identity.unsafe_ptr()), Int(down.unsafe_ptr())]
            var p: Array[Int, 3] = [nt, _D, nt]
            var f: Array[Float64, 0] = []
            _ = driver.run("take_rows", a, p, f)
            _same_f32(down, train_rows, "export download of the resident rows")
            _ = identity^
            # OneClassSVM's support vectors gathered on the device from the fold slot (`oc_take_rows_into`'s op):
            # rows 5, 0, 11 of the fold, out of order.
            var support: List[Int32] = [5, 0, 11]
            var vectors = List[Float32](length=3 * _D, fill=0.0)
            var sa: Array[Int, 3] = [fold_x_word(train_slot), Int(support.unsafe_ptr()), Int(vectors.unsafe_ptr())]
            var sp: Array[Int, 3] = [3, _D, nt]
            _ = driver.run("take_rows", sa, sp, f)
            var want = List[Float32](capacity=3 * _D)
            for r in range(3):  # small-loop(support rows)
                for c in range(_D):  # small-loop(row words)
                    want.append(train_rows[Int(support[r]) * _D + c])
            _same_f32(vectors, want, "support vectors gathered from the fold slot")
            _ = support^
        except error:
            failure = String(error)
        store[].free(store_ctx, test_slot)
        store[].free(store_ctx, train_slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = train^
        _ = test^
        _ = train_rows^
        _ = test_rows^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _host_case(case)

        check_cases_serial("xneighbors_member_rows host", 2, body)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _device_case(case)

        check_cases_serial("xneighbors_member_rows device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
