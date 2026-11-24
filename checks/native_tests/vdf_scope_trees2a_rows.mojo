# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""Device-store rows for the forest images (lane vdf-scope-trees2a).

- XF_FIT_ROWS on the ExtraTrees GPU image: a fit on rows TAKEN ON THE DEVICE
  out of the image's store (`_et_*_fit_run[ROWS]`, `x_dev_rows`) writes the
  same trees, byte for byte, as the plain row-major fit (XF_FIT, ROWMAJOR) on
  the same rows gathered on the host: classifier, regressor, weighted.
- XF_RESIDENT_ROWS on both GPU images: a resident snapshot's votes and labels
  of rows taken on the device out of the image's store
  (`forest_resident_rows_native`) equal, byte for byte, its votes and labels
  of the same rows gathered on the host (XF_RESIDENT_PREDICT /
  XF_RESIDENT_LABELS).

The CPU column is the host images' fit / walk of the same gathered rows,
checked equal by the release's one verification (identical mode); here the
device-rows form is compared with the host-input form on the same image.

Small fixture: 96 rows x 5 features, a fold of 61 rows with repeats.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require as _require
from core.entry_table import entry_ref_unresolved
from bindings.forest_config_run import forest_request
from xtrees.trees_forest_run import ForestFit
from ensemble.instruments import StageTimes
from checks.native_tests.bm_trees_et_fits import EtSpec, et_config
from extratrees.mojolearn_trees_run import (
    _et_classifier_fit_run, _et_regressor_fit_run, ET_STORE, et_store_ctx,
)
from core.mojolearn_trees_host_impl import et_forest_lists
from core.mojolearn_rf_run import RF_STORE, rf_store_ctx
from core.forest_inference_run import (
    forest_prepare_gpu_native, forest_predict_resident_into_gpu_native,
    forest_predict_resident_labels_gpu_native, forest_resident_rows_native,
)
from core.forest_inference_model import resident_release

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


def _same_et[cls: Bool](weighted: Bool) raises:
    var x = _x()
    var fold = _fold()
    var n = len(fold)
    var xg = _gather(x, fold)
    var yg = List[Float32](length=n, fill=0.0)
    var wg = List[Float32](length=n, fill=0.0)
    for i in range(n):  # fixture labels (float32 codes for the classifier) and weights of the fold rows
        var r = Int(fold[i])
        yg[i] = Float32((r * 5) % _K) if cls else Float32(r % 17) * Float32(0.5)
        wg[i] = Float32(1 + (r % 3))
    var classes = _K if cls else 0
    var spec = EtSpec(cls, 7, 1234)
    var fit = ForestFit(cls, rowmajor_ok=True)
    var plain = forest_request(et_config(spec), entry_ref_unresolved(), n, _D, classes,
                               Int(xg.unsafe_ptr()), Int(yg.unsafe_ptr()), fit, False)
    var rows = forest_request(et_config(spec), entry_ref_unresolved(), n, _D, classes,
                              Int(fold.unsafe_ptr()), Int(yg.unsafe_ptr()), fit, False)
    if weighted:
        plain.weights = Int(wg.unsafe_ptr())
        rows.weights = Int(wg.unsafe_ptr())
    plain.gil_free = True
    rows.gil_free = True
    # The whole X, row-major, in the image's store (as the scope cache puts it).
    var store = ET_STORE.get_or_create_ptr()
    var ctx = et_store_ctx()
    rows.session = store[].put(ctx, Int(x.unsafe_ptr()), _N * _D)
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
        comptime if cls:
            var a = _et_classifier_fit_run[True, False](plain, times, start)
            var b = _et_classifier_fit_run[False, True](rows, times, start)
            et_forest_lists(a, o1, c1, q1, l1, v1)
            et_forest_lists(b, o2, c2, q2, l2, v2)
        else:
            var a = _et_regressor_fit_run[True, False](plain, times, start)
            var b = _et_regressor_fit_run[False, True](rows, times, start)
            et_forest_lists(a, o1, c1, q1, l1, v1)
            et_forest_lists(b, o2, c2, q2, l2, v2)
    finally:
        store[].free(ctx, rows.session)
    _require(len(o1) > 1, "device-rows ET fit: no trees")
    _require(o1 == o2 and c1 == c2 and l1 == l2, "device-rows ET fit: tree structure differs from the plain fit")
    _require(q1 == q2 and v1 == v2, "device-rows ET fit: thresholds or leaves differ from the plain fit")
    _ = xg^
    _ = x^
    _ = fold^
    _ = yg^
    _ = wg^


def _same_resident[RF_INPUT: Bool]() raises:
    """A two-tree, two-output snapshot (local children, right = left + 1,
    leaf left = -1): votes and labels of the fold's rows, device rows
    against host-gathered rows."""
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
    var va = List[Float32](length=n * outputs, fill=0.0)
    var vb = List[Float32](length=n * outputs, fill=-1.0)
    var la = List[Int32](length=n, fill=0)
    var lb = List[Int32](length=n, fill=-1)
    var ctx = rf_store_ctx() if RF_INPUT else et_store_ctx()
    var store = RF_STORE.get_or_create_ptr()
    comptime if not RF_INPUT:
        store = ET_STORE.get_or_create_ptr()
    var sid = store[].put(ctx, Int(x.unsafe_ptr()), _N * _D)
    try:
        _ = forest_predict_resident_into_gpu_native[RF_INPUT, True](handle, Int(xg.unsafe_ptr()),
                                                                    Int(va.unsafe_ptr()), n, _D, outputs)
        _ = forest_predict_resident_labels_gpu_native[RF_INPUT](handle, Int(xg.unsafe_ptr()),
                                                                Int(la.unsafe_ptr()), n, _D, outputs)
        var wrote = forest_resident_rows_native[RF_INPUT](store[], ctx, False, handle, sid,
                                                          Int(fold.unsafe_ptr()), Int(vb.unsafe_ptr()), n, _D, outputs)
        _require(wrote == n, "device-rows resident vote: rows written")
        wrote = forest_resident_rows_native[RF_INPUT](store[], ctx, True, handle, sid,
                                                      Int(fold.unsafe_ptr()), Int(lb.unsafe_ptr()), n, _D, outputs)
        _require(wrote == n, "device-rows resident labels: rows written")
    finally:
        store[].free(ctx, sid)
        resident_release[RF_INPUT](handle)
    _require(va == vb, "device-rows resident vote differs from the host-gathered rows' vote")
    _require(la == lb, "device-rows resident labels differ from the host-gathered rows' labels")
    _ = offsets^
    _ = columns^
    _ = thresholds^
    _ = left^
    _ = leaves^
    _ = x^
    _ = xg^
    _ = fold^


def test_et_classifier_rows() raises:
    _same_et[True](False)


def test_et_classifier_rows_weighted() raises:
    _same_et[True](True)


def test_et_regressor_rows() raises:
    _same_et[False](False)


def test_et_regressor_rows_weighted() raises:
    _same_et[False](True)


def test_rf_resident_rows() raises:
    _same_resident[True]()


def test_et_resident_rows() raises:
    _same_resident[False]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
