# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""DecisionTree*(splitter='random') and the ExtraTreesRegressor deviance criteria, host fits against
scikit-learn's held-out scores (lane bm-trees, 2026-11-22; ported from python/mojolearn/tests/test_x_trees_sanity.py
`test_decision_tree_random_splitter` and `test_extratrees_deviance_criteria`).

The inputs are the original `_clf()` / `_reg()` splits (1050 training and 450 held-out rows, 10 features, in
`fixtures/trees_sk/`) and, in `fixtures/trees_bm/` (`tools/py2mojo_native_fixtures_bm_trees.py`), the exp-transformed
targets of the deviance test and scikit-learn's scores. The fits and predictions are the host ExtraTrees image's
typed RUN (`bm_trees_et_fits.mojo` over core/mojolearn_et_host_run.mojo). Assertions are the original's: the
random-splitter classifier's accuracy at least scikit-learn's minus 0.08; the random-splitter regressor at most 20
leaves and its R^2 at least scikit-learn's minus 0.1; the Poisson forest's R^2 at least scikit-learn's minus 0.08, the
gamma and inverse-Gaussian forests' at least scikit-learn's Poisson R^2 minus 0.15, and a gamma fit on a target with
negative values refused (the door's ValueError words). The six units are one case table of independent fits, each
running its own parallel builder.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_forest_fits import argmax_rows, tree_depth_leaves, accuracy, r2
from checks.native_tests.bm_trees_et_fits import (
    EtSpec, et_fit, et_predict, ET_POISSON, ET_GAMMA, ET_INVERSE_GAUSSIAN,
)

comptime SK_DIR = "checks/native_tests/fixtures/trees_sk/"
comptime BM_DIR = "checks/native_tests/fixtures/trees_bm/"
comptime NA = 1050
comptime NB = 450
comptime D = 10
comptime ET_CASES = 6


def _codes(y: List[Int32]) -> List[Float32]:
    var out = List[Float32](capacity=len(y))
    for v in y:  # small-loop(fixture staging: `codes.astype('<f4')`)
        out.append(Float32(v))
    return out^


def et_case(unit: Int) raises:
    if unit == 0:
        var scores = fixture_f64(BM_DIR + "dt-random-scores.f64", 2)
        var xa = fixture_f32(SK_DIR + "clf-xa.f32", NA * D)
        var xb = fixture_f32(SK_DIR + "clf-xb.f32", NB * D)
        var ya = fixture_i32(SK_DIR + "clf-ya.i32", NA)
        var yb = fixture_i32(SK_DIR + "clf-yb.i32", NB)
        var ours = et_fit(EtSpec.random_tree(True, 10, -1, 0), xa, NA, D, _codes(ya), 3)
        var a = accuracy(argmax_rows(et_predict(ours, xb, NB), NB, 3), yb)
        require(a >= scores[0] - 0.08, "random-splitter accuracy " + String(a) + ", scikit-learn " + String(scores[0]))
        return
    var xa = fixture_f32(SK_DIR + "reg-xa.f32", NA * D)
    var xb = fixture_f32(SK_DIR + "reg-xb.f32", NB * D)
    if unit == 1:
        var scores = fixture_f64(BM_DIR + "dt-random-scores.f64", 2)
        var ya = fixture_f32(SK_DIR + "reg-ya.f32", NA)
        var yb = fixture_f32(SK_DIR + "reg-yb.f32", NB)
        var ours = et_fit(EtSpec.random_tree(False, -1, 20, 0), xa, NA, D, ya, 0)
        var shape = tree_depth_leaves(ours, 0)
        require(shape[1] <= 20, "random-splitter leaves " + String(shape[1]))
        var got = r2(et_predict(ours, xb, NB), yb)
        require(got >= scores[1] - 0.1, "random-splitter R^2 " + String(got) + ", scikit-learn " + String(scores[1]))
        return
    var reference = fixture_f64(BM_DIR + "et-poisson-r2.f64", 1)[0]
    var ya = fixture_f32(BM_DIR + "pos-ya.f32", NA)
    var yb = fixture_f32(BM_DIR + "pos-yb.f32", NB)
    var spec = EtSpec(False, 40, 0)
    spec.max_depth = 10
    if unit == 5:
        spec.criterion = ET_GAMMA
        var shifted = List[Float32](capacity=NA)
        for v in ya:  # small-loop(fixture staging: ya - 10.0)
            shifted.append(v - Float32(10.0))
        try:
            _ = et_fit(spec, xa, NA, D, shifted, 0)
        except problem:
            require("criterion='gamma' requires y > 0" in String(problem), "unexpected refusal: " + String(problem))
            return
        raise Error("a gamma fit on a target with negative values did not refuse")
    var tol = 0.08
    if unit == 2:
        spec.criterion = ET_POISSON
    elif unit == 3:
        spec.criterion = ET_GAMMA
        tol = 0.15
    else:
        spec.criterion = ET_INVERSE_GAUSSIAN
        tol = 0.15
    var got = r2(et_predict(et_fit(spec, xa, NA, D, ya, 0), xb, NB), yb)
    require(got >= reference - tol, "criterion " + String(spec.criterion) + " R^2 " + String(got) + ", scikit-learn poisson "
            + String(reference))


def test_et_quality() raises:
    check_cases("et_quality", ET_CASES, et_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
