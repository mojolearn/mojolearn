# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The typed vote RUN (xtrees/trees_meta_vote_run.mojo) of VotingClassifier / VotingRegressor, reached natively
(lane bm-meta, 2026-11-22). Replaces the numeric part of
python/mojolearn/tests/test_meta_members_native.py test_hard_vote_of_forest_members_matches_their_predictions and
test_soft_vote_is_the_weighted_mean_of_member_probabilities (the Python tests stay).

The members are the tests' own: a random forest (4 trees, depth 3, seed 0) and a decision tree (depth 2), fitted
by the host image (checks/native_tests/py2mojo_forest_fits.mojo) on `_clf()` (X from
`tools/py2mojo_native_fixtures_bm_meta.py`, `fixtures/meta_vote/`; y recomputed from X as the test computes it).
A forest member answers through its door in the product, so RUN reads each member's block as the door hands it.
The vote is checked against the fold the Python test writes: hard, the first argmax of the one-hot counts of the
members' codes; soft, the weighted sum of the members' probabilities at their class columns over the sum of the
weights. A native member whose call refuses is reported by index before anything is written. The independent
checks run in order after one fit of the members (see `test_vote_run`).

`test_voting_sklearn` (lane bm-meta-2) is python/mojolearn/tests/test_x_trees_sanity.py `test_voting` (the Python
test stays): decision-tree members fitted by the host image on the `_clf()` / `_reg()` splits
(`fixtures/trees_sk/`), folded by the vote RUN, against scikit-learn's own VotingClassifier accuracy and weighted
VotingRegressor R^2 (`fixtures/meta_vote/sk-scores.f64`): the soft vote of depths 4 and 8 at least scikit-learn's
accuracy minus 0.05, the hard vote of depths 4, 8 and 6 at least minus 0.08, the regressor (weights 1, 2) at least
scikit-learn's R^2 minus 0.05.
"""
from std.math import min, max
from std.testing import TestSuite
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32
from checks.native_tests.py2mojo_forest_fits import RfSpec, rf_fit, predict, argmax_rows, require, accuracy, r2
from checks.native_tests.tsa_metrics_fixture_words import fixture_i32
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef
from core.member_abi import MemberPlan, MEMBER_OP_PREDICT, MEMBER_I32
from core.trees_entry_table import TreesEntryKey
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import xt_ref
from xtrees.trees_meta_vote_run import (
    VoteMember, VoteRows, VOTE_ALL_ANSWERED, vote_codes_run, vote_soft_run, vote_regress_run, vote_transform_run,
)

comptime N = 90
comptime D = 5
comptime K = 3
comptime X_PATH = "checks/native_tests/fixtures/meta_vote/clf-x.f32"
comptime CASES = 5


def _glue() raises -> EntryRef:
    """The glue entry `test_vote_run` published before the cases."""
    return xt_ref(TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


def _codes(x: List[Float32]) -> List[Int32]:
    """`_clf`: y = (X[:, 0] * k) as int, clipped to 0 .. k-1."""
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: the labels of one block)
        y[i] = Int32(min(max(Int(x[i * D] * Float32(K)), 0), K - 1))
    return y^


def _member(spec: RfSpec, x: List[Float32], y: List[Int32], mut proba: List[List[Float32]],
            mut codes: List[List[Int32]]) raises:
    var f = rf_fit(spec, x, N, D, y, List[Float32](), K, List[Float32]())
    require(f.outputs == K, "member outputs " + String(f.outputs))
    var p = predict(f, x, N)
    var best = argmax_rows(p, N, K)
    var c = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(one member's codes)
        c[i] = Int32(best[i])
    proba.append(p^)
    codes.append(c^)


struct Members(Movable):
    """The two fitted members' probabilities (n x k, float32) and their codes (the first argmax)."""
    var proba: List[List[Float32]]
    var codes: List[List[Int32]]
    var classes: List[Int32]

    def __init__(out self) raises:
        var x = fixture_f32(X_PATH, N * D)
        var y = _codes(x)
        var forest = RfSpec(True, 4, 0)
        forest.max_depth = 3
        var tree = RfSpec.decision_tree(True, 2)
        self.proba = List[List[Float32]]()
        self.codes = List[List[Int32]]()
        _member(forest, x, y, self.proba, self.codes)
        _member(tree, x, y, self.proba, self.codes)
        self.classes = [0, 1, 2]
        _ = x^

    def rows(self) -> VoteRows:
        # The members' blocks are their doors' answers: RUN reads no X for them.
        return VoteRows(0, N, D)


def check_hard_vote(m: Members) raises:
    """test_hard_vote_of_forest_members_matches_their_predictions."""
    var members: List[VoteMember] = [VoteMember(Int(m.codes[0].unsafe_ptr()), 1, 0, 1.0),
                                     VoteMember(Int(m.codes[1].unsafe_ptr()), 1, 0, 1.0)]
    var acc = List[Float64](length=N * K, fill=0.0)
    var got = List[Int32](length=N, fill=Int32(-1))
    var refused = vote_codes_run(_glue(), m.rows(), members, K, False, Int(acc.unsafe_ptr()), Int(got.unsafe_ptr()))
    require(refused == VOTE_ALL_ANSWERED, "hard vote refused member " + String(refused))
    for i in range(N):  # small-loop(oracle: the test's one-hot count and first argmax)
        var votes: List[Float64] = [0.0, 0.0, 0.0]
        for j in range(2):  # small-loop(the members)
            votes[Int(m.codes[j][i])] += 1.0
        var best = 0
        for c in range(1, K):  # small-loop(the classes of one row)
            if votes[c] > votes[best]:
                best = c
        require(Int(got[i]) == best, "hard vote row " + String(i) + ": " + String(got[i]) + " != " + String(best))


def check_soft_vote(m: Members) raises:
    """test_soft_vote_is_the_weighted_mean_of_member_probabilities (weights 2, 1), and the soft predict codes."""
    var cols = Int(m.classes.unsafe_ptr())
    var members: List[VoteMember] = [VoteMember(Int(m.proba[0].unsafe_ptr()), K, cols, 2.0),
                                     VoteMember(Int(m.proba[1].unsafe_ptr()), K, cols, 1.0)]
    var acc = List[Float64](length=N * K, fill=0.0)
    var refused = vote_soft_run(_glue(), m.rows(), members, K, Int(acc.unsafe_ptr()))
    require(refused == VOTE_ALL_ANSWERED, "soft vote refused member " + String(refused))
    var top = List[Float64](length=N * K, fill=0.0)
    var codes = List[Int32](length=N, fill=Int32(-1))
    _ = vote_codes_run(_glue(), m.rows(), members, K, True, Int(top.unsafe_ptr()), Int(codes.unsafe_ptr()))
    for i in range(N):  # small-loop(oracle: the test's weighted sum over the weight total)
        var best = 0
        for c in range(K):  # small-loop(the classes of one row)
            var want = (2.0 * Float64(m.proba[0][i * K + c]) + 1.0 * Float64(m.proba[1][i * K + c])) / 3.0
            require(abs(acc[i * K + c] - want) <= 1e-6, "soft vote cell " + String(i) + "," + String(c) + ": " +
                    String(acc[i * K + c]) + " != " + String(want))
            if acc[i * K + c] > acc[i * K + best]:
                best = c
        require(Int(codes[i]) == best, "soft predict row " + String(i))


def check_regress_vote(m: Members) raises:
    """VotingRegressor.predict: the weighted mean of the members' predictions (weights 2, 1, 1)."""
    var a = List[Float32](length=N, fill=Float32(0))
    var b = List[Float32](length=N, fill=Float32(0))
    var c = List[Float32](length=N, fill=Float32(0))
    for i in range(N):  # small-loop(fixture: three prediction columns from the members' probabilities)
        a[i] = m.proba[0][i * K]
        b[i] = m.proba[1][i * K + 1]
        c[i] = Float32(i) * Float32(0.25)
    var members: List[VoteMember] = [VoteMember(Int(a.unsafe_ptr()), 1, 0, 2.0), VoteMember(Int(b.unsafe_ptr()), 1, 0, 1.0),
                                     VoteMember(Int(c.unsafe_ptr()), 1, 0, 1.0)]
    var acc = List[Float64](length=N, fill=0.0)
    var refused = vote_regress_run(_glue(), m.rows(), members, Int(acc.unsafe_ptr()))
    require(refused == VOTE_ALL_ANSWERED, "regress vote refused member " + String(refused))
    for i in range(N):  # small-loop(oracle)
        var want = (2.0 * Float64(a[i]) + Float64(b[i]) + Float64(c[i])) / 4.0
        require(abs(acc[i] - want) <= 1e-12, "regress row " + String(i) + ": " + String(acc[i]) + " != " + String(want))


def check_transform(m: Members) raises:
    """Voting*.transform: hard, the members' outputs side by side; soft, each member's probabilities at its
    classes, rounded to float32, in its own k columns."""
    var a = List[Float32](length=N, fill=Float32(0))
    var b = List[Float32](length=N, fill=Float32(0))
    for i in range(N):  # small-loop(fixture: two members' codes as float32 predictions)
        a[i] = Float32(m.codes[0][i])
        b[i] = Float32(m.codes[1][i])
    var hard: List[VoteMember] = [VoteMember(Int(a.unsafe_ptr()), 1, 0, 1.0), VoteMember(Int(b.unsafe_ptr()), 1, 0, 1.0)]
    var out = List[Float64](length=N * 2, fill=0.0)
    require(vote_transform_run(_glue(), m.rows(), hard, 1, False, 2, Int(out.unsafe_ptr())) == VOTE_ALL_ANSWERED,
            "hard transform refused")
    for i in range(N):  # small-loop(oracle)
        require(out[2 * i] == Float64(a[i]) and out[2 * i + 1] == Float64(b[i]), "hard transform row " + String(i))
    var cols = Int(m.classes.unsafe_ptr())
    var soft: List[VoteMember] = [VoteMember(Int(m.proba[0].unsafe_ptr()), K, cols, 2.0),
                                  VoteMember(Int(m.proba[1].unsafe_ptr()), K, cols, 1.0)]
    var wide = List[Float64](length=N * 2 * K, fill=0.0)
    require(vote_transform_run(_glue(), m.rows(), soft, K, True, 2 * K, Int(wide.unsafe_ptr())) == VOTE_ALL_ANSWERED,
            "soft transform refused")
    for i in range(N):  # small-loop(oracle: the weights do not scale a transform)
        for j in range(2):  # small-loop(the members)
            for c in range(K):  # small-loop(the classes)
                require(wide[i * 2 * K + j * K + c] == Float64(m.proba[j][i * K + c]),
                        "soft transform cell " + String(i) + "," + String(j) + "," + String(c))


def check_refusal_index(m: Members) raises:
    """A native member whose call refuses (here: a plan with no published member entry) is reported by its
    index, and no vote is written: every member answers before any vote is cast."""
    var plan = MemberPlan()
    var members: List[VoteMember] = [VoteMember(Int(m.codes[0].unsafe_ptr()), 1, 0, 1.0),
                                     VoteMember(Int(UnsafePointer(to=plan)), 0, MEMBER_OP_PREDICT, 1, MEMBER_I32, 0,
                                                1.0)]
    var acc = List[Float64](length=N * K, fill=0.0)
    var got = List[Int32](length=N, fill=Int32(-7))
    var refused = vote_codes_run(_glue(), m.rows(), members, K, False, Int(acc.unsafe_ptr()), Int(got.unsafe_ptr()))
    require(refused == 1, "refused index " + String(refused))
    for i in range(N * K):  # small-loop(oracle: nothing written)
        require(acc[i] == 0.0, "a vote was cast before every member answered")
    for i in range(N):  # small-loop(oracle)
        require(got[i] == Int32(-7), "codes written after a refusal")
    _ = plan^


def test_vote_run() raises:
    # The publication _mojolearn_x_trees_host makes at module init, once before the parallel cases.
    publish_x_trees_glue("cpu")
    # The members are fitted once (each fit splits its own work over the cores); the cases read them.
    var members = Members()
    var at = UnsafePointer(to=members)
    def run_case(which: Int) raises {imm at}:
        if which == 0:
            check_hard_vote(at[])
        elif which == 1:
            check_soft_vote(at[])
        elif which == 2:
            check_regress_vote(at[])
        elif which == 3:
            check_transform(at[])
        else:
            check_refusal_index(at[])
    # SERIAL, and why: as a check_cases table (one host worker per case) every case passes and the process then
    # dies at exit (SIGSEGV in libAsyncRTRuntimeGlobals' thread-local teardown, __nptl_deallocate_tsd, on a case
    # worker; cpc1 2026-11-22, lane bm-meta: rc 139 with 0 failed cases, rc 0 when the same cases run in order).
    # The glue ops a case calls start runtime work on the worker thread. Each case is a few milliseconds of
    # folds over fitted members; the fits above use every core. Restore the table when that teardown is fixed.
    for which in range(CASES):  # small-loop(five millisecond-scale cases; see the note above)
        run_case(which)
    _ = members^


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
