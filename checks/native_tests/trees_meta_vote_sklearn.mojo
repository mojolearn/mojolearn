# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""python/mojolearn/tests/test_x_trees_sanity.py `test_voting` through the typed vote RUN
(xtrees/trees_meta_vote_run.mojo; lane bm-meta-2, 2026-11-22; the Python test stays): decision-tree members
fitted by the host image (checks/native_tests/py2mojo_forest_fits.mojo) on the `_clf()` / `_reg()` splits
(`fixtures/trees_sk/`), folded by the vote RUN, against scikit-learn's own VotingClassifier accuracy and weighted
VotingRegressor R^2 (`fixtures/meta_vote/sk-scores.f64`, tools/py2mojo_native_fixtures_bm_meta.py): the soft vote
of depths 4 and 8 at least scikit-learn's accuracy minus 0.05, the hard vote of depths 4, 8 and 6 at least minus
0.08, the regressor (weights 1, 2) at least scikit-learn's R^2 minus 0.05.

A file of its own, not a second test of trees_meta_vote.mojo: in one process after that file's cases the process
dies at exit in the runtime's thread teardown (lane bm-meta finding; ~/mojolearn-evidence/bm-meta-2-asyncrt-exit/).
"""
from std.testing import TestSuite
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32, fixture_i32
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_forest_fits import RfSpec, rf_fit, predict, argmax_rows, require, accuracy, r2
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef
from core.trees_entry_table import TreesEntryKey
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import xt_ref
from xtrees.trees_meta_vote_run import VoteMember, VoteRows, VOTE_ALL_ANSWERED, vote_codes_run, vote_soft_run, vote_regress_run


def _glue() raises -> EntryRef:
    return xt_ref(TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


comptime SK_DIR = "checks/native_tests/fixtures/trees_sk/"
comptime NA = 1050
comptime NB = 450
comptime SD = 10


def _argmax_f64(acc: List[Float64], rows: Int, k: Int) -> List[Int]:
    var out = List[Int](length=rows, fill=0)
    for i in range(rows):  # small-loop(oracle: the first maximum of each voted row)
        var best = 0
        for c in range(1, k):  # small-loop(the classes of one row)
            if acc[i * k + c] > acc[i * k + best]:
                best = c
        out[i] = best
    return out^


def test_voting_sklearn() raises:
    """test_x_trees_sanity.py::test_voting (module note)."""
    publish_x_trees_glue("cpu")
    var scores = fixture_f64("checks/native_tests/fixtures/meta_vote/sk-scores.f64", 2)
    var xa = fixture_f32(SK_DIR + "clf-xa.f32", NA * SD)
    var xb = fixture_f32(SK_DIR + "clf-xb.f32", NB * SD)
    var ya = fixture_i32(SK_DIR + "clf-ya.i32", NA)
    var yb = fixture_i32(SK_DIR + "clf-yb.i32", NB)
    var classes: List[Int32] = [0, 1, 2]
    var depths: List[Int] = [4, 8, 6]
    var proba = List[List[Float32]]()
    var codes = List[List[Int32]]()
    for t in range(3):  # small-loop(the three members; each fit splits its own work over the cores)
        var f = rf_fit(RfSpec.decision_tree(True, depths[t]), xa, NA, SD, ya, List[Float32](), 3, List[Float32]())
        var p = predict(f, xb, NB)
        var best = argmax_rows(p, NB, 3)
        var c = List[Int32](length=NB, fill=Int32(0))
        for i in range(NB):  # small-loop(one member's codes)
            c[i] = Int32(best[i])
        proba.append(p^)
        codes.append(c^)
    var rows = VoteRows(0, NB, SD)
    var cols = Int(classes.unsafe_ptr())
    var soft: List[VoteMember] = [VoteMember(Int(proba[0].unsafe_ptr()), 3, cols, 1.0),
                                  VoteMember(Int(proba[1].unsafe_ptr()), 3, cols, 1.0)]
    var acc = List[Float64](length=NB * 3, fill=0.0)
    require(vote_soft_run(_glue(), rows, soft, 3, Int(acc.unsafe_ptr())) == VOTE_ALL_ANSWERED, "soft vote refused")
    var a = accuracy(_argmax_f64(acc, NB, 3), yb)
    require(a >= scores[0] - 0.05, "soft vote accuracy " + String(a) + ", scikit-learn " + String(scores[0]))
    var hard: List[VoteMember] = [VoteMember(Int(codes[0].unsafe_ptr()), 1, 0, 1.0),
                                  VoteMember(Int(codes[1].unsafe_ptr()), 1, 0, 1.0),
                                  VoteMember(Int(codes[2].unsafe_ptr()), 1, 0, 1.0)]
    var counts = List[Float64](length=NB * 3, fill=0.0)
    var got = List[Int32](length=NB, fill=Int32(-1))
    require(vote_codes_run(_glue(), rows, hard, 3, False, Int(counts.unsafe_ptr()), Int(got.unsafe_ptr()))
            == VOTE_ALL_ANSWERED, "hard vote refused")
    var hard_pred = List[Int](length=NB, fill=0)
    for i in range(NB):  # small-loop(the voted codes)
        hard_pred[i] = Int(got[i])
    var h = accuracy(hard_pred, yb)
    require(h >= scores[0] - 0.08, "hard vote accuracy " + String(h) + ", scikit-learn soft " + String(scores[0]))
    var ra = fixture_f32(SK_DIR + "reg-xa.f32", NA * SD)
    var rb = fixture_f32(SK_DIR + "reg-xb.f32", NB * SD)
    var ta = fixture_f32(SK_DIR + "reg-ya.f32", NA)
    var tb = fixture_f32(SK_DIR + "reg-yb.f32", NB)
    var shallow = predict(rf_fit(RfSpec.decision_tree(False, 4), ra, NA, SD, List[Int32](), ta, 0, List[Float32]()),
                          rb, NB)
    var deep = predict(rf_fit(RfSpec.decision_tree(False, 8), ra, NA, SD, List[Int32](), ta, 0, List[Float32]()),
                       rb, NB)
    var reg: List[VoteMember] = [VoteMember(Int(shallow.unsafe_ptr()), 1, 0, 1.0),
                                 VoteMember(Int(deep.unsafe_ptr()), 1, 0, 2.0)]
    var mean = List[Float64](length=NB, fill=0.0)
    require(vote_regress_run(_glue(), rows, reg, Int(mean.unsafe_ptr())) == VOTE_ALL_ANSWERED, "regress vote refused")
    var pred = List[Float32](length=NB, fill=Float32(0))
    for i in range(NB):  # small-loop(the voted predictions, float32 as the estimator returns them)
        pred[i] = Float32(mean[i])
    var q = r2(pred, tb)
    require(q >= scores[1] - 0.05, "regress vote R^2 " + String(q) + ", scikit-learn " + String(scores[1]))
    _ = len(classes)
    _ = len(proba)
    _ = len(codes)
    _ = len(shallow)
    _ = len(deep)
    _ = len(xa)
    _ = len(ra)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
