# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""Device-side forest member pieces of lane vdf-scope-trees3.

- XF_SHAPE_Y_CODES on the ExtraTrees GPU image: a classifier fit whose y is
  int32 class codes (`upload_dataset_codes_i32`, the range checked on the
  device) writes the same trees, byte for byte, as the fit on the float32
  codes of the same rows (`upload_dataset_labels_f32`): the plain row-major
  fit and the device-store rows fit (XF_FIT_ROWS), unweighted and weighted.
- XF_RESIDENT_ROWS mode 2 on both GPU images: the count the device returns
  for the store-taken rows equals the host count of the host-gathered rows'
  resident labels equal to the true codes.
- The device pool's rows input (`PooledForest.predict_rows_into`,
  `predict_labels_rows_into`, one owner here: the same-device copy of
  `contribution_rows`): votes, labels and the score equal the pool's host-row
  predictions of the same rows byte for byte. Two physical devices (the
  device-to-device and AMD host-staged copies) need a two-GPU box.

The CPU column is the host images' fit / walk of the same gathered rows,
checked equal by the release's one verification (identical mode); here each
device form is compared with the host-input form on the same image.

Small fixture: 96 rows x 5 features, a fold of 61 rows with repeats.
"""
from std.testing import TestSuite
from checks.native_tests.kit import upload_whole_into
from checks.native_tests.testsuite_cases import require as _require
from core.entry_table import entry_ref_unresolved
from bindings.forest_config_run import forest_request
from xtrees.trees_forest_run import ForestFit
from ensemble.instruments import StageTimes
from checks.native_tests.bm_trees_et_fits import EtSpec, et_config
from extratrees.mojolearn_trees_run import _et_classifier_fit_run, ET_STORE, et_store_ctx
from core.mojolearn_trees_host_impl import et_forest_lists
from core.mojolearn_rf_run import RF_STORE, rf_store_ctx
from core.forest_inference_run import (
    forest_prepare_gpu_native, forest_predict_resident_labels_gpu_native, forest_resident_rows_native,
)
from core.forest_inference_model import resident_release
from core.forest_inference_pool import PooledForest

comptime _N = 96
comptime _D = 5
comptime _K = 3


def _x() -> List[Float32]:
    var x = List[Float32](length=_N * _D, fill=0.0)
    for i in range(_N):  # fixture rows
        for c in range(_D):  # fixture columns
            x[i * _D + c] = Float32(((i * 37 + c * 11) % 53) - 20) * Float32(0.25)
    return x^


def _fold() -> List[Int64]:
    var rows = List[Int64]()
    for i in range(61):  # fixture fold: a stride with repeats
        rows.append(Int64((i * 7 + 3) % _N))
    return rows^


def _gather(x: List[Float32], fold: List[Int64]) -> List[Float32]:
    var n = len(fold)
    var xg = List[Float32](length=n * _D, fill=0.0)
    for i in range(n):  # fixture gather
        var r = Int(fold[i])
        for c in range(_D):  # columns
            xg[i * _D + c] = x[r * _D + c]
    return xg^


def _same_codes[ROWS: Bool](weighted: Bool) raises:
    """The ET classifier on int32 codes == on float32 codes (same rows)."""
    var x = _x()
    var fold = _fold()
    var n = len(fold)
    var xg = _gather(x, fold)
    var yf = List[Float32](length=n, fill=0.0)
    var yi = List[Int32](length=n, fill=0)
    var wg = List[Float32](length=n, fill=0.0)
    for i in range(n):  # fixture class codes (both forms) and weights of the fold rows
        var r = Int(fold[i])
        yi[i] = Int32((r * 5) % _K)
        yf[i] = Float32(yi[i])
        wg[i] = Float32(1 + (r % 3))
    var spec = EtSpec(True, 7, 1234)
    var fit = ForestFit(True, rowmajor_ok=True)
    var x_word = Int(fold.unsafe_ptr()) if ROWS else Int(xg.unsafe_ptr())
    var floats = forest_request(et_config(spec), entry_ref_unresolved(), n, _D, _K, x_word, Int(yf.unsafe_ptr()), fit, False)
    var codes = forest_request(et_config(spec), entry_ref_unresolved(), n, _D, _K, x_word, Int(yi.unsafe_ptr()), fit, False)
    codes.y_codes = True
    if weighted:
        floats.weights = Int(wg.unsafe_ptr())
        codes.weights = Int(wg.unsafe_ptr())
    floats.gil_free = True
    codes.gil_free = True
    var store = ET_STORE.get_or_create_ptr()
    var ctx = et_store_ctx()
    var sid = -1
    comptime if ROWS:
        sid = store[].put(ctx, Int(x.unsafe_ptr()), _N * _D)
        floats.session = sid
        codes.session = sid
    var o1 = List[Int64]()
    var c1 = List[Int64]()
    var q1 = List[Float64]()
    var l1 = List[Int64]()
    var v1 = List[Float64]()
    var o2 = List[Int64]()
    var c2 = List[Int64]()
    var q2 = List[Float64]()
    var l2 = List[Int64]()
    var v2 = List[Float64]()
    var times = StageTimes.disabled()
    var start = 0
    try:
        comptime if ROWS:
            var a = _et_classifier_fit_run[False, True](floats, times, start)
            var b = _et_classifier_fit_run[False, True](codes, times, start)
            et_forest_lists(a, o1, c1, q1, l1, v1)
            et_forest_lists(b, o2, c2, q2, l2, v2)
        else:
            var a = _et_classifier_fit_run[True, False](floats, times, start)
            var b = _et_classifier_fit_run[True, False](codes, times, start)
            et_forest_lists(a, o1, c1, q1, l1, v1)
            et_forest_lists(b, o2, c2, q2, l2, v2)
    finally:
        if sid >= 0:
            store[].free(ctx, sid)
    _require(len(o1) > 1, "int32-code ET fit: no trees")
    _require(o1 == o2 and c1 == c2 and l1 == l2, "int32-code ET fit: tree structure differs from the float-code fit")
    _require(q1 == q2 and v1 == v2, "int32-code ET fit: thresholds or leaves differ from the float-code fit")
    _ = xg^
    _ = x^
    _ = fold^
    _ = yf^
    _ = yi^
    _ = wg^


def _truth(n: Int) -> List[Int32]:
    var y = List[Int32](length=n, fill=0)
    for i in range(n):  # fixture true codes: about half match any one class
        y[i] = Int32((i * 3) % 2)
    return y^


def _host_hits(codes: List[Int32], truth: List[Int32]) -> Int:
    var hits = 0
    for i in range(len(codes)):  # fixture count on the host (the oracle)
        if codes[i] == truth[i]:
            hits += 1
    return hits


def _same_score[RF_INPUT: Bool]() raises:
    """XF_RESIDENT_ROWS mode 2 == the host count over the host-gathered
    rows' resident labels."""
    var offsets: List[Int32] = [0, 3, 6]
    var columns: List[Int32] = [1, 0, 0, 3, 0, 0]
    var thresholds: List[Float32] = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
    var left: List[Int32] = [1, -1, -1, 1, -1, -1]
    var leaves: List[Float32] = [0.0, 0.0, 0.75, 0.25, 0.125, 0.875, 0.0, 0.0, 0.5, 0.5, 0.375, 0.625]
    var model: List[Int] = [Int(offsets.unsafe_ptr()), Int(columns.unsafe_ptr()), Int(thresholds.unsafe_ptr()),
                            Int(left.unsafe_ptr()), Int(leaves.unsafe_ptr())]
    var outputs = 2
    var handle = forest_prepare_gpu_native[RF_INPUT](model, _D, 2, outputs, -1)
    var x = _x()
    var fold = _fold()
    var n = len(fold)
    var xg = _gather(x, fold)
    var la = List[Int32](length=n, fill=0)
    var truth = _truth(n)
    var ctx = rf_store_ctx() if RF_INPUT else et_store_ctx()
    var store = RF_STORE.get_or_create_ptr()
    comptime if not RF_INPUT:
        store = ET_STORE.get_or_create_ptr()
    var sid = store[].put(ctx, Int(x.unsafe_ptr()), _N * _D)
    var hits = -1
    try:
        _ = forest_predict_resident_labels_gpu_native[RF_INPUT](handle, Int(xg.unsafe_ptr()),
                                                                Int(la.unsafe_ptr()), n, _D, outputs)
        hits = forest_resident_rows_native[RF_INPUT](store[], ctx, 2, handle, sid,
                                                     Int(fold.unsafe_ptr()), Int(truth.unsafe_ptr()), n, _D, outputs)
    finally:
        store[].free(ctx, sid)
        resident_release[RF_INPUT](handle)
    var expected = _host_hits(la, truth)
    _require(expected > 0 and expected < n, "device-rows score fixture: the truth must split the rows")
    _require(hits == expected, "device-rows score: the device count differs from the host count")
    _ = offsets^
    _ = columns^
    _ = thresholds^
    _ = left^
    _ = leaves^
    _ = x^
    _ = xg^
    _ = fold^


def _same_pool[RF_INPUT: Bool]() raises:
    """The device pool on device-resident rows == on host rows (one owner)."""
    var offsets: List[Int32] = [0, 3, 6]
    var columns: List[Int32] = [1, 0, 0, 3, 0, 0]
    var thresholds: List[Float32] = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
    var left: List[Int32] = [1, -1, -1, 1, -1, -1]
    var leaves: List[Float32] = [0.0, 0.0, 0.75, 0.25, 0.125, 0.875, 0.0, 0.0, 0.5, 0.5, 0.375, 0.625]
    var outputs = 2
    var pool = PooledForest(offsets, columns, thresholds, left, leaves, _D, outputs, 1)
    var x = _x()
    var fold = _fold()
    var n = len(fold)
    var xg = _gather(x, fold)
    var truth = _truth(n)
    var va = List[Float32](length=n * outputs, fill=0.0)
    var vb = List[Float32](length=n * outputs, fill=-1.0)
    var la = List[Int32](length=n, fill=0)
    var lb = List[Int32](length=n, fill=-1)
    var ctx = rf_store_ctx() if RF_INPUT else et_store_ctx()
    var dx = ctx.enqueue_create_buffer[DType.float32](n * _D)
    upload_whole_into(ctx, dx, xg)
    var xp = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(xg.unsafe_ptr()))
    pool.predict_into[RF_INPUT](xp, MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(va.unsafe_ptr())), n)
    pool.predict_labels_into[RF_INPUT](xp, MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(la.unsafe_ptr())), n)
    pool.predict_rows_into[RF_INPUT](ctx, dx.create_sub_buffer[DType.float32](0, n * _D),
                                     MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(vb.unsafe_ptr())), n)
    var wrote = pool.predict_labels_rows_into[RF_INPUT](ctx, dx.create_sub_buffer[DType.float32](0, n * _D),
        MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(lb.unsafe_ptr())), n)
    var hits = pool.predict_labels_rows_into[RF_INPUT](ctx, dx.create_sub_buffer[DType.float32](0, n * _D),
        MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(truth.unsafe_ptr())), n, True)
    _ = dx^  # the predictions above read the device rows (synchronous)
    _require(wrote == n, "pool device-rows labels: rows written")
    _require(va == vb, "pool device-rows vote differs from the host rows' vote")
    _require(la == lb, "pool device-rows labels differ from the host rows' labels")
    _require(hits == _host_hits(la, truth), "pool device-rows score differs from the host count")
    _ = pool^
    _ = offsets^
    _ = columns^
    _ = thresholds^
    _ = left^
    _ = leaves^
    _ = x^
    _ = xg^
    _ = fold^


def test_et_codes_plain() raises:
    _same_codes[False](False)


def test_et_codes_plain_weighted() raises:
    _same_codes[False](True)


def test_et_codes_rows() raises:
    _same_codes[True](False)


def test_et_codes_rows_weighted() raises:
    _same_codes[True](True)


def test_rf_rows_score() raises:
    _same_score[True]()


def test_et_rows_score() raises:
    _same_score[False]()


def test_rf_pool_rows() raises:
    _same_pool[True]()


def test_et_pool_rows() raises:
    _same_pool[False]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
