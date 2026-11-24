# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The KNNImputer over a model-selection fold's rows on the device (lane vdf-scope-cluster6,
core/xneighbors_member.mojo `_rows_fold_op` for the imputer kind; neighbors/knn_imputer_run.mojo, the cell ops
starting their result from the uploaded X, x_neighbors/nan_cells_device.mojo THE FOLD X WORD).

Claims, per column:
  device: (case 0, the tiled cell op; case 1, the old-items `knn_impute_cells`; case 2, the missing number 0.5
          masked, add_indicator) the fit's column masks over the fold word == over the host-gathered rows; the
          transform of the test fold with the query AND the fitted rows as slot words == with both as host rows
          (float32 bits of the whole result, indicator columns included); a test fold with no missing cell
          (`imp_copy_rows`' `take_rows` read) == the host rows.
  host:   the same fit and transform over the host take of the fold on the host entry (the CPU column), where
          the cell op now copies x into its result itself: the transform's present cells == x's bits.
Data: a SplitMix64 stream (uniform [0, 4)), 90 rows x 4, the cells (r * 7 + c) % 11 == 0 NaN (case 2: 0.5);
train fold the rows with i % 5 != 0 (second half first), test fold the rest; n_neighbors 5, uniform weights.
Sabotage: in x_neighbors/iter_device.mojo `op_knn_impute_tiled` drop the device copy of x into res -> the
present cells of the device transform are zeros and the claims fail.
"""
from std.testing import TestSuite
from std.memory import bitcast
from std.math import nan, isnan
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from max.gpu.host import DeviceContext
from core.member_fold_x import member_fold_take_f32
from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_word
from x_neighbors.native_descriptor_host import xn_native_v1 as xn_host_v1
from x_neighbors.native_descriptor_device import xn_native_v1 as xn_device_v1
from core.neighbors_native_driver import XnDriver
from neighbors.knn_imputer_run import ImpFitRequest, ImpTransformRequest, imp_fit_run, imp_transform_run
from neighbors.neighbors_local_run import imp_keep_plan, ImpKeepPlan

comptime _ROWS = 90
comptime _D = 4
comptime _K = 5


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _x(which: Int) -> List[Float32]:
    var state = UInt64(20261124)
    var out = List[Float32](capacity=_ROWS * _D)
    for i in range(_ROWS * _D):  # small-loop(fixture words)
        var v = Float32(Float64(_next(state) >> 11) * (1.0 / 9007199254740992.0) * 4.0)
        var r = i // _D
        var c = i % _D
        if (r * 7 + c) % 11 == 0:
            v = Float32(0.5) if which == 2 else nan[DType.float32]()
        out.append(v)
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
    var plan: ImpKeepPlan
    var missing: List[Int]
    var masked: List[Float32]
    var rows_word: Int

    def __init__(out self):
        self.plan = imp_keep_plan(List[Bool]())
        self.missing = List[Int]()
        self.masked = List[Float32]()
        self.rows_word = 0


def _fit(driver: XnDriver, x_word: Int, rows: Int, which: Int) raises -> _Fit:
    """The member's `_im_fit` RUN (the missing number NaN, or 0.5 masked in case 2)."""
    var fit = _Fit()
    var mask = which == 2
    if mask:
        fit.masked = List[Float32](length=rows * _D, fill=0.0)
    var columns = imp_fit_run(driver, ImpFitRequest(x_word, Int(fit.masked.unsafe_ptr()) if mask else 0, rows, _D,
                                                    0.5 if mask else nan[DType.float64](), False))
    fit.plan = imp_keep_plan(columns.valid)
    fit.missing = columns.missing.copy()
    fit.rows_word = Int(fit.masked.unsafe_ptr()) if mask else x_word
    return fit^


def _transform(driver: XnDriver, query: Int, fit: _Fit, nq: Int, nt: Int, which: Int) raises -> List[Float32]:
    var indicator = fit.missing.copy() if which == 2 else List[Int]()
    var width = _D + len(indicator)
    var out = List[Float32](length=max(nq * width, 1), fill=0.0)
    var request = ImpTransformRequest(query, fit.rows_word, Int(out.unsafe_ptr()), nq, nt, _D, _K, False, which == 2,
                                      0.5 if which == 2 else nan[DType.float64](), False, which == 1, True)
    imp_transform_run(driver, request, fit.plan, indicator)
    return out^


def _same_f32(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths")
    for i in range(len(a)):  # small-loop(compared words)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": bits differ at " + String(i))


def _host_case(which: Int) raises:
    comptime if CPU_COLUMN_BUILD:
        var driver = XnDriver(xn_host_v1)
        var x = _x(which)
        var train = _index(True)
        var test = _index(False)
        var nt = len(train)
        var nq = len(test)
        var taken = List[Float32](length=nt * _D, fill=0.0)
        member_fold_take_f32(Int(x.unsafe_ptr()), Int(train.unsafe_ptr()), nt, _D, Int(taken.unsafe_ptr()))
        _same_f32(taken, _serial(x, train), "host take of the fold")
        var query = _serial(x, test)
        var fit = _fit(driver, Int(taken.unsafe_ptr()), nt, which)
        var out = _transform(driver, Int(query.unsafe_ptr()), fit, nq, nt, which)
        if which != 2:
            # The cell op's own copy: every present cell is x's word.
            for i in range(nq * _D):  # small-loop(compared words)
                if not isnan(query[i]):
                    require(bitcast[DType.uint32](out[i]) == bitcast[DType.uint32](query[i]), "a present cell moved")
        _ = fit^
        _ = x^
        _ = taken^


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var driver = XnDriver(xn_device_v1)
        var x = _x(which)
        var train = _index(True)
        var test = _index(False)
        var nt = len(train)
        var nq = len(test)
        var train_rows = _serial(x, train)
        var test_rows = _serial(x, test)
        # a test fold with no missing cell: rows 1, 2, 4 (no (r * 7 + c) % 11 == 0 cell among them)
        var clean: List[Int64] = [1, 2, 4]
        var clean_rows = _serial(x, clean)
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), _ROWS * _D)
        var train_slot = store[].take_rows(store_ctx, base, _D, Int(train.unsafe_ptr()), nt)
        var test_slot = store[].take_rows(store_ctx, base, _D, Int(test.unsafe_ptr()), nq)
        var clean_slot = store[].take_rows(store_ctx, base, _D, Int(clean.unsafe_ptr()), 3)
        var failure = String("")
        try:
            var host_fit = _fit(driver, Int(train_rows.unsafe_ptr()), nt, which)
            var fold_fit = _fit(driver, fold_x_word(train_slot), nt, which)
            require(len(host_fit.missing) == len(fold_fit.missing), "missing columns")
            for i in range(len(host_fit.missing)):  # small-loop(missing columns)
                require(host_fit.missing[i] == fold_fit.missing[i], "missing column " + String(i))
            require(len(host_fit.plan.keep) == len(fold_fit.plan.keep), "kept columns")
            if which == 2:
                _same_f32(fold_fit.masked, host_fit.masked, "the masked fit rows")
            var resident = _transform(driver, fold_x_word(test_slot), fold_fit, nq, nt, which)
            var host = _transform(driver, Int(test_rows.unsafe_ptr()), host_fit, nq, nt, which)
            _same_f32(resident, host, "transform over the resident rows")
            var clean_resident = _transform(driver, fold_x_word(clean_slot), fold_fit, 3, nt, which)
            var clean_host = _transform(driver, Int(clean_rows.unsafe_ptr()), host_fit, 3, nt, which)
            _same_f32(clean_resident, clean_host, "transform of a fold with no missing cell")
        except error:
            failure = String(error)
        store[].free(store_ctx, clean_slot)
        store[].free(store_ctx, test_slot)
        store[].free(store_ctx, train_slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = train^
        _ = test^
        _ = clean^
        _ = train_rows^
        _ = test_rows^
        _ = clean_rows^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _host_case(which)

        check_cases_serial("xneighbors_member_im_rows host", 3, body)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("xneighbors_member_im_rows device", 3, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
