# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""DecisionTreeClassifier and DecisionTreeRegressor (splitter 'best') host fits against scikit-learn's held-out
scores (lane py2mojo-C, 2026-11-22; ported from python/mojolearn/tests/test_x_trees_sanity.py
`test_decision_tree_classifier` and `test_decision_tree_regressor`).

The inputs are the original `_clf()` / `_reg()` splits (make_classification / make_regression, 1050 training and
450 held-out rows, 10 features) and the references scikit-learn's accuracy and R^2 of its own trees, stored by
`tools/py2mojo_native_fixtures_b3.py` in `fixtures/trees_sk/`. The fits and predictions are the host image's
(`py2mojo_forest_fits.mojo`). Assertions are the original's: accuracy at least scikit-learn's minus 0.05, depth at
most 6, every probability row summing to 1 (rtol 1e-5), the class-0 recall of the fit weighted 5:1 toward class 0
at least the unweighted fit's minus 0.02; the regressor's R^2 at least scikit-learn's minus 0.05. The two tests are
one case table of two independent fits each running its own parallel builder.

Not converted here (they stay in Python; docs/plans/PY2MOJO_MAP.tsv names the reason per row): the random
splitter and ExtraTrees fits (their host fit body is only in the binding module bindings/_mojolearn_trees_host.mojo)
and the meta-estimators (Bagging, AdaBoost, Voting, Stacking, MultiOutput, OneVsRest, CalibratedClassifierCV,
RandomTreesEmbedding: each composes its members in a door over Python objects).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_forest_fits import (
    RfSpec, rf_fit, predict, argmax_rows, tree_depth_leaves, accuracy, r2,
)

comptime SK_DIR = "checks/native_tests/fixtures/trees_sk/"
comptime NA = 1050
comptime NB = 450
comptime D = 10


def _recall0(pred: List[Int], want: List[Int32]) -> Float64:
    var hit = 0
    var total = 0
    for i in range(len(want)):  # small-loop(oracle: one held-out block)
        if want[i] == 0:
            total += 1
            if pred[i] == 0:
                hit += 1
    return Float64(hit) / Float64(total)


def decision_tree_case(unit: Int) raises:
    var scores = fixture_f64(SK_DIR + "dt-scores.f64", 2)
    if unit == 0:
        var xa = fixture_f32(SK_DIR + "clf-xa.f32", NA * D)
        var xb = fixture_f32(SK_DIR + "clf-xb.f32", NB * D)
        var ya = fixture_i32(SK_DIR + "clf-ya.i32", NA)
        var yb = fixture_i32(SK_DIR + "clf-yb.i32", NB)
        var spec = RfSpec.decision_tree(True, 6)
        var ours = rf_fit(spec, xa, NA, D, ya, List[Float32](), 3, List[Float32]())
        var p = predict(ours, xb, NB)
        var pred = argmax_rows(p, NB, 3)
        var a = accuracy(pred, yb)
        require(a >= scores[0] - 0.05, "decision tree accuracy " + String(a) + ", scikit-learn " + String(scores[0]))
        var shape = tree_depth_leaves(ours, 0)
        require(shape[0] <= 6, "decision tree depth " + String(shape[0]))
        for r in range(NB):  # small-loop(oracle: the probability rows)
            var s = Float64(p[3 * r]) + Float64(p[3 * r + 1]) + Float64(p[3 * r + 2])
            require(abs(s - 1.0) <= 1e-5, "probability row " + String(r) + " sums to " + String(s))
        var w = List[Float32](length=NA, fill=Float32(1))
        for i in range(NA):  # small-loop(np.where(ya == 0, 5.0, 1.0))
            if ya[i] == 0:
                w[i] = Float32(5)
        var weighted = rf_fit(spec, xa, NA, D, ya, List[Float32](), 3, w)
        var wpred = argmax_rows(predict(weighted, xb, NB), NB, 3)
        var rw = _recall0(wpred, yb)
        var ru = _recall0(pred, yb)
        require(rw >= ru - 0.02, "class-0 recall weighted " + String(rw) + ", unweighted " + String(ru))
        _ = xa^
        _ = xb^
        return
    var xa = fixture_f32(SK_DIR + "reg-xa.f32", NA * D)
    var xb = fixture_f32(SK_DIR + "reg-xb.f32", NB * D)
    var ya = fixture_f32(SK_DIR + "reg-ya.f32", NA)
    var yb = fixture_f32(SK_DIR + "reg-yb.f32", NB)
    var ours = rf_fit(RfSpec.decision_tree(False, 8), xa, NA, D, List[Int32](), ya, 0, List[Float32]())
    var got = r2(predict(ours, xb, NB), yb)
    require(got >= scores[1] - 0.05, "decision tree R^2 " + String(got) + ", scikit-learn " + String(scores[1]))
    _ = xa^
    _ = xb^


def test_decision_trees() raises:
    check_cases("decision_trees", 2, decision_tree_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
