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
checks run as one case table across the host cores after one fit of the members.
"""
from std.math import min, max
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, require
from checks.native_tests.py2mojo_forest_fits import RfSpec, rf_fit, predict, argmax_rows
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
    _ = len(acc)  # Mojo frees a local at its last named use: the fold wrote `acc` through its address


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
    _ = len(top)  # Mojo frees a local at its last named use: the fold wrote `top` through its address


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
    # One case table across the host cores. It once ran in order: as a table the process died at exit
    # (SIGSEGV in the runtime allocator's thread-cache teardown on a case worker, lane bm-meta). The cause was
    # this file's (lane bm-meta-2, 2026-11-22): `check_hard_vote`'s `acc` and `check_soft_vote`'s `top` were last
    # named in the argument that took their address, so Mojo freed them before the fold wrote into them; the
    # writes corrupted the worker's free lists. Both are named after their use now.
    check_cases("vote", CASES, run_case)
    _ = members^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
