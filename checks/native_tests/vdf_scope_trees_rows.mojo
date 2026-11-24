# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""XF_FIT_ROWS (lane vdf-scope-trees): a RandomForest fit on rows TAKEN ON
THE DEVICE out of the image's store (`rf_store_rows_x`, the ROWS form of the
fit bodies) writes the same trees, byte for byte, as the plain row-major fit
(XF_FIT, ROWMAJOR) on the same rows gathered on the host: classifier and
regressor, unweighted, weighted, with importances and with out-of-bag
outputs. The CPU column is the host image's fit of the same gathered rows,
checked equal by the release's one verification (identical mode); here the
two GPU forms are compared with each other.

Small fixture: 96 rows x 5 features, 7 trees, a fold of 61 rows with
repeats; Gini / MSE.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require as _require
from core.forest_fit_request import ForestFitArgs
from core.mojolearn_rf_run import (
    _rf_classifier_fit_forest, _rf_regressor_fit_forest, RF_STORE, rf_store_ctx,
)
from core.mojolearn_rf_impl import rf_flatten_trees
from ensemble.instruments import StageTimes

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


def _params(rows: Int, classes: Int) -> List[Int]:
    # n_rows, n_cols, n_classes, trees, depth, leaves, (fraction), bins,
    # min leaf, min split, (decrease), bootstrap, (max_samples), seed, streams, batch
    return [rows, _D, classes, 7, 6, -1, 0, 32, 1, 2, 0, 1, 0, 1234, 4, 4096]


def _floats(rows: Int, classes: Int) -> List[Float64]:
    var p = _params(rows, classes)
    var f = List[Float64](length=len(p), fill=0.0)
    for k in range(len(p)):  # 16 slots
        f[k] = Float64(p[k])
    f[6] = 0.6
    f[10] = 0.0
    f[12] = 1.0
    return f^


def _args(x: Int, y: Int, rows: Int, classes: Int, criterion: Int) raises -> ForestFitArgs:
    var p = _params(rows, classes)
    return ForestFitArgs(x, y, p^, _floats(rows, classes), List[UInt8](length=16, fill=0), criterion)


def _same_forest[cls: Bool](weighted: Bool, importance: Bool, oob: Bool) raises:
    var x = _x()
    var fold = _fold()
    var n = len(fold)
    # Host gather of the fold (the plain fit's input).
    var xg = List[Float32](length=n * _D, fill=0.0)
    var yc = List[Int32](length=_N, fill=0)
    var yr = List[Float32](length=_N, fill=0.0)
    var w = List[Float32](length=_N, fill=1.0)
    for i in range(_N):  # fixture labels and weights
        yc[i] = Int32((i * 5) % _K)
        yr[i] = Float32(i % 17) * Float32(0.5)
        w[i] = Float32(1 + (i % 3))
    var ycg = List[Int32](length=n, fill=0)
    var yrg = List[Float32](length=n, fill=0.0)
    var wg = List[Float32](length=n, fill=0.0)
    for i in range(n):  # fixture gather
        var r = Int(fold[i])
        for c in range(_D):  # columns
            xg[i * _D + c] = x[r * _D + c]
        ycg[i] = yc[r]
        yrg[i] = yr[r]
        wg[i] = w[r]
    var classes = _K if cls else 0
    var y_addr = Int(ycg.unsafe_ptr()) if cls else Int(yrg.unsafe_ptr())
    var imp_a = List[Float32](length=_D, fill=0.0)
    var imp_b = List[Float32](length=_D, fill=0.0)
    var oob_a = List[Float64](length=n * max(classes, 1), fill=0.0)
    var oob_b = List[Float64](length=n * max(classes, 1), fill=0.0)
    var sc_a = List[Float64](length=1, fill=0.0)
    var sc_b = List[Float64](length=1, fill=0.0)
    var crit = 0 if cls else 2
    var plain = _args(Int(xg.unsafe_ptr()), y_addr, n, classes, crit)
    var rows = _args(Int(fold.unsafe_ptr()), y_addr, n, classes, crit)
    if weighted:
        plain.weights = Int(wg.unsafe_ptr())
        rows.weights = Int(wg.unsafe_ptr())
    if importance:
        plain.importance = Int(imp_a.unsafe_ptr())
        rows.importance = Int(imp_b.unsafe_ptr())
    if oob:
        plain.oob = Int(oob_a.unsafe_ptr())
        plain.score = Int(sc_a.unsafe_ptr())
        rows.oob = Int(oob_b.unsafe_ptr())
        rows.score = Int(sc_b.unsafe_ptr())
    plain.gil_free = True
    rows.gil_free = True
    # The whole X, row-major, in the image's store (as the scope cache puts it).
    var store = RF_STORE.get_or_create_ptr()
    var ctx = rf_store_ctx()
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
    var bt = StageTimes()
    try:
        comptime if cls:
            var a = _rf_classifier_fit_forest[True, False](plain, bt)
            var b = _rf_classifier_fit_forest[False, True](rows, bt)
            rf_flatten_trees(a.trees, o1, c1, q1, l1, v1)
            rf_flatten_trees(b.trees, o2, c2, q2, l2, v2)
        else:
            var a = _rf_regressor_fit_forest[True, False](plain, bt)
            var b = _rf_regressor_fit_forest[False, True](rows, bt)
            rf_flatten_trees(a.trees, o1, c1, q1, l1, v1)
            rf_flatten_trees(b.trees, o2, c2, q2, l2, v2)
    finally:
        store[].free(ctx, rows.session)
    _require(o1 == o2 and c1 == c2 and l1 == l2, "device-rows fit: tree structure differs from the plain fit")
    _require(q1 == q2 and v1 == v2, "device-rows fit: thresholds or leaves differ from the plain fit")
    _require(imp_a == imp_b, "device-rows fit: importances differ")
    _require(oob_a == oob_b and sc_a == sc_b, "device-rows fit: out-of-bag outputs differ")
    _ = xg^
    _ = x^
    _ = fold^
    _ = ycg^
    _ = yrg^
    _ = wg^


def test_classifier_rows() raises:
    _same_forest[True](False, False, False)


def test_classifier_rows_weighted_importances() raises:
    _same_forest[True](True, True, False)


def test_classifier_rows_oob() raises:
    _same_forest[True](False, False, True)


def test_regressor_rows() raises:
    _same_forest[False](False, False, False)


def test_regressor_rows_weighted() raises:
    _same_forest[False](True, False, False)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
