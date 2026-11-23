# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""TreeExplainer on a host DART regressor is additive (lane bm-rest-rte, 2026-11-23; ported from
python/mojolearn/tests/test_meta_members_native.py `test_dart_tree_explainer_is_additive`).

The inputs are the original's: X = default_rng(2).random((150, 4), float32), y = X[:, 0] - X[:, 2] in float32
(`fixtures/trees_rte/shap-*`, `tools/py2mojo_native_fixtures_bm_rest_rte.py`). The model is
DARTRegressor(n_estimators=DART_TREES, random_state=0) (the original's 10; check-trim-2, 2026-11-23: the DART fit is
this file's time, and additivity holds for any member count, so 5 rounds, with drops possible from round 1, prove it), fitted by the product's typed RUN through the host images
(`bm_rest_boost_fits.mojo` `dart_fit`). The explainer is what `TreeExplainer(d, data=X[:60])` builds for a DART
model (bindings/trees_explainer_native_door.mojo `_flat` and `_construct`): the members' node words joined into one
flat forest with the members' leaf values as its leaves, the tree coefficients as float32 tree scales, the starting
score as the float32 initial expected value, the cover of the 60 background rows (`shap_prepare` of
xtrees/shap_host.mojo); `shap_values(X[:20])` is `tree_shap_values`. The assertion is the original's: each row's
values summed plus the expected value equal the model's prediction (numpy assert_allclose rtol 1e-5, atol 1e-3).
The 20 explained rows are independent oracle rows across the host cores.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, require
from checks.native_tests.bm_rest_boost_fits import publish_host_images, DartSpec, dart_fit, dart_raw
from bindings.trees_dart_rounds_run import DART_M_OFF_START, DART_M_NODE_START, DART_M_VAL_START
from xtrees.shap import shap_path_width
from xtrees.shap_host import shap_prepare, tree_shap_values

comptime RTE_DIR = "checks/native_tests/fixtures/trees_rte/"
comptime N = 150
comptime D = 4
comptime BACKGROUND = 60
comptime ROWS = 20
comptime DART_TREES = 5


def test_dart_tree_explainer_is_additive() raises:
    publish_host_images()
    var x = fixture_f32(RTE_DIR + "shap-x.f32", N * D)
    var y = fixture_f32(RTE_DIR + "shap-y.f32", N)
    var m = dart_fit(DartSpec(DART_TREES, 0), x, N, D, y, 1, 0)
    var trees = m.trees()
    require(trees == DART_TREES, "DART trees " + String(trees))
    # `_flat`: each member's node words in tree order, its leaf values as the leaves (k = 1: no spread)
    var offsets = List[Int32](length=trees + 1, fill=Int32(0))
    var total = 0
    for t in range(trees):  # small-loop(the flat forest's offsets, trees-sized)
        total += m.size(t)
        offsets[t + 1] = Int32(total)
    var colid = List[Int32](length=total, fill=Int32(0))
    var quesval = List[Float32](length=total, fill=Float32(0))
    var left = List[Int32](length=total, fill=Int32(0))
    var leaves = List[Float32](length=total, fill=Float32(0))
    var tscale = List[Float32](length=trees, fill=Float32(0))
    for t in range(trees):  # small-loop(oracle staging: one copy of each member's nodes, as `_copy_into`)
        var at = Int(offsets[t])
        var node = m.rounds.word(t, DART_M_NODE_START) + Int(m.rounds.offsets[m.rounds.word(t, DART_M_OFF_START)])
        var val = m.rounds.word(t, DART_M_VAL_START)
        for i in range(m.size(t)):  # small-loop(the nodes of one member tree)
            colid[at + i] = m.rounds.colid[node + i]
            quesval[at + i] = m.rounds.quesval[node + i]
            left[at + i] = m.rounds.left[node + i]
            leaves[at + i] = m.rounds.values[val + i]
        # `_snapshot(tree_coefs_, "<f4")`: the float64 coefficient as float32
        tscale[t] = Float32(m.rounds.coefs[t])
    var model: List[Int] = [Int(offsets.unsafe_ptr()), Int(colid.unsafe_ptr()), Int(quesval.unsafe_ptr()),
                            Int(left.unsafe_ptr()), Int(leaves.unsafe_ptr())]
    var cover = List[Int32](length=total, fill=Int32(0))
    # the initial expected value: `init_score_` as float32; `shap_prepare` adds every tree's share
    var ev = List[Float32](length=1, fill=Float32(m.starts[0]))
    var meta = List[Int32](length=3, fill=Int32(0))
    shap_prepare(model, Int(tscale.unsafe_ptr()), Int(x.unsafe_ptr()), Int(cover.unsafe_ptr()), Int(ev.unsafe_ptr()),
                 Int(meta.unsafe_ptr()), BACKGROUND, D, trees, 1, total)
    var width = shap_path_width(Int(meta[1]), Int(meta[0]))
    require(width > 0, "shap path width")
    var phi = List[Float32](length=ROWS * D, fill=Float32(0))
    tree_shap_values(model, Int(tscale.unsafe_ptr()), Int(cover.unsafe_ptr()), Int(x.unsafe_ptr()),
                     Int(phi.unsafe_ptr()), ROWS, D, trees, 1, total, Int(meta[0]), width)
    var pred = dart_raw(m, x, ROWS)
    var expected = Float64(ev[0])
    var pp = phi.unsafe_ptr()
    var qp = pred.unsafe_ptr()

    def row(r: Int) raises {imm pp, imm qp, imm expected}:
        var got = expected
        for j in range(D):  # small-loop(oracle: the features of one row)
            got += Float64(pp[r * D + j])
        var want = qp[r]
        require(abs(got - want) <= 1e-3 + 1e-5 * abs(want), "row " + String(r) + ": phi sum + expected "
                + String(got) + ", predict " + String(want))

    check_cases("dart_tree_explainer_additive", ROWS, row)
    _ = x^
    _ = y^
    _ = phi^
    _ = pred^
    _ = cover^
    _ = tscale^
    _ = ev^
    _ = meta^
    _ = model^
    _ = offsets^
    _ = colid^
    _ = quesval^
    _ = left^
    _ = leaves^
    _ = m^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
