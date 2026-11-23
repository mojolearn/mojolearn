# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""RandomTreesEmbedding and DART host fits against the original Python checks (lane bm-rest-rte, 2026-11-23;
ported from python/mojolearn/tests/test_x_trees_sanity.py `test_random_trees_embedding`, `test_dart` and
`test_dart_options`).

The inputs are the original `_clf()` / `_reg()` splits (make_classification / make_regression, 1050 training and
450 held-out rows, 10 features): the three-class and regression splits in `fixtures/trees_sk/`
(`tools/py2mojo_native_fixtures_b3.py`), the two-class split and LightGBM's held-out scores of its own DART fits
with the same settings in `fixtures/trees_rte/` (`tools/py2mojo_native_fixtures_bm_rest_rte.py`). The fits are the
product's typed RUN through the host images (`bm_rest_boost_fits.mojo`). Assertions are the originals':

- the embedding (n_estimators=30, max_depth=5, random_state=0): every row of the training and held-out one-hot
  has exactly 30 ones (one leaf per tree), every cell is 0 or 1, and the one-hot is n x n_output_features_ wide.
  The downstream LogisticRegression accuracy stays in Python (docs/plans/PY2MOJO_MAP.tsv).
- DART: the regressor's R^2 at least LightGBM's minus 0.08, the binary classifier's accuracy at least LightGBM's
  minus 0.05; with options, the three-class accuracy at least LightGBM's minus 0.06 and every probability row
  summing to 1 (numpy allclose: atol 1e-8, rtol 1e-5), the regressor with L1 and sampling at least LightGBM's minus
  0.08, and with max_delta_step=40 at least LightGBM's minus 0.15.

The fits are independent case-table rows, run across the host cores; each keeps its own parallel builders.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_forest_fits import accuracy
from checks.native_tests.bm_rest_boost_fits import (
    publish_host_images, rte_spec, rte_fit, rte_transform, DartSpec, dart_fit, dart_raw, dart_labels, dart_proba,
    r2_f64,
)

comptime SK_DIR = "checks/native_tests/fixtures/trees_sk/"
comptime RTE_DIR = "checks/native_tests/fixtures/trees_rte/"
comptime NA = 1050
comptime NB = 450
comptime D = 10
#: the embedding's tree count (the original's n_estimators=30)
comptime RTE_TREES = 30


def _codes(y: List[Int32]) -> List[Float32]:
    """The door's float32 class codes of labels 0 .. K-1 (`trees_encode_labels`, then the cast)."""
    var out = List[Float32](length=len(y), fill=Float32(0))
    for i in range(len(y)):  # small-loop(fixture staging: one label block)
        out[i] = Float32(y[i])
    return out^


def _one_hot_rows(e: List[Float64], rows: Int, width: Int, what: String) raises:
    require(len(e) == rows * width, what + ": " + String(len(e)) + " cells for " + String(rows) + " x "
            + String(width))
    for r in range(rows):  # small-loop(oracle: the rows of one embedding)
        var total = 0.0
        for c in range(width):  # small-loop(the cells of one row)
            var v = e[r * width + c]
            require(v == 0.0 or v == 1.0, what + " row " + String(r) + " column " + String(c) + ": " + String(v))
            total += v
        require(total == Float64(RTE_TREES), what + " row " + String(r) + " sums to " + String(total)
                + ", want " + String(RTE_TREES))


def embedding_case(unit: Int) raises:
    var xa = fixture_f32(SK_DIR + "clf-xa.f32", NA * D)
    var xb = fixture_f32(SK_DIR + "clf-xb.f32", NB * D)
    var emb = rte_fit(rte_spec(RTE_TREES, 5, 0), xa, NA, D)
    require(emb.forest.trees == RTE_TREES, "embedding trees " + String(emb.forest.trees))
    require(emb.width >= RTE_TREES, "embedding width " + String(emb.width))
    if unit == 0:
        _one_hot_rows(rte_transform(emb, xa, NA), NA, emb.width, "transform(Xa)")
    else:
        _one_hot_rows(rte_transform(emb, xb, NB), NB, emb.width, "transform(Xb)")
    _ = xa^
    _ = xb^


def dart_case(unit: Int) raises:
    """Units 0-1: `test_dart`; 2-4: `test_dart_options`, in the original's order."""
    var scores = fixture_f64(RTE_DIR + "dart-scores.f64", 5)
    if unit == 0 or unit == 3 or unit == 4:
        var xa = fixture_f32(SK_DIR + "reg-xa.f32", NA * D)
        var xb = fixture_f32(SK_DIR + "reg-xb.f32", NB * D)
        var ya = fixture_f32(SK_DIR + "reg-ya.f32", NA)
        var yb = fixture_f32(SK_DIR + "reg-yb.f32", NB)
        var s = DartSpec(60, 0)
        var slack = 0.08
        var ref_index = 0
        var what = String("DARTRegressor(n_estimators=60, learning_rate=0.1)")
        if unit == 3:
            s.reg_alpha = 2.0
            s.colsample_bytree = 0.6
            s.subsample = 0.7
            s.subsample_freq = 2
            ref_index = 3
            what = "DARTRegressor(reg_alpha=2.0, colsample_bytree=0.6, subsample=0.7, subsample_freq=2)"
        elif unit == 4:
            s.max_delta_step = 40.0
            slack = 0.15
            ref_index = 4
            what = "DARTRegressor(max_delta_step=40.0)"
        var m = dart_fit(s, xa, NA, D, ya, 1, 0)
        var got = r2_f64(dart_raw(m, xb, NB), yb)
        require(got >= scores[ref_index] - slack, what + " R^2 " + String(got) + ", LightGBM "
                + String(scores[ref_index]))
        _ = xa^
        _ = xb^
        return
    if unit == 1:
        var xa = fixture_f32(RTE_DIR + "clf2-xa.f32", NA * D)
        var xb = fixture_f32(RTE_DIR + "clf2-xb.f32", NB * D)
        var ya = fixture_i32(RTE_DIR + "clf2-ya.i32", NA)
        var yb = fixture_i32(RTE_DIR + "clf2-yb.i32", NB)
        var m = dart_fit(DartSpec(60, 0), xa, NA, D, _codes(ya), 1, 1)
        var got = accuracy(dart_labels(m, xb, NB), yb)
        require(got >= scores[1] - 0.05, "DARTClassifier(n_estimators=60) binary accuracy " + String(got)
                + ", LightGBM " + String(scores[1]))
        _ = xa^
        _ = xb^
        return
    var xa = fixture_f32(SK_DIR + "clf-xa.f32", NA * D)
    var xb = fixture_f32(SK_DIR + "clf-xb.f32", NB * D)
    var ya = fixture_i32(SK_DIR + "clf-ya.i32", NA)
    var yb = fixture_i32(SK_DIR + "clf-yb.i32", NB)
    var s = DartSpec(60, 0)
    s.reg_alpha = 0.5
    s.max_delta_step = 0.7
    s.colsample_bytree = 0.8
    s.subsample = 0.8
    s.subsample_freq = 1
    var m = dart_fit(s, xa, NA, D, _codes(ya), 3, 2)
    var got = accuracy(dart_labels(m, xb, NB), yb)
    require(got >= scores[2] - 0.06, "DARTClassifier(options) three-class accuracy " + String(got) + ", LightGBM "
            + String(scores[2]))
    var p = dart_proba(m, xb, NB)
    require(len(p) == NB * 3, "predict_proba cells " + String(len(p)))
    for r in range(NB):  # small-loop(oracle: the probability rows)
        var total = p[3 * r] + p[3 * r + 1] + p[3 * r + 2]
        require(abs(total - 1.0) <= 1e-8 + 1e-5, "probability row " + String(r) + " sums to " + String(total))
    _ = xa^
    _ = xb^


def test_random_trees_embedding() raises:
    publish_host_images()
    check_cases("random_trees_embedding", 2, embedding_case)


def _dart_unit(unit: Int) raises:
    dart_case(unit)


def _dart_options_unit(unit: Int) raises:
    dart_case(unit + 2)


def test_dart() raises:
    publish_host_images()
    check_cases("dart", 2, _dart_unit)


def test_dart_options() raises:
    publish_host_images()
    check_cases("dart_options", 3, _dart_options_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
