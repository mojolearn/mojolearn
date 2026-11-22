# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The typed member-set RUN (core/member_set_run.mojo, lane bm-meta-2, 2026-11-22): the fit of a set of native
members that VotingClassifier / VotingRegressor, MultiOutput*, OneVsRest, Bagging* and Stacking* run, reached
from Mojo with no Python.

The members are k-NN members (core/neighbors_member.mojo), whose fit needs no kernel table: a fitted k-NN state
IS its fit rows, so the two blocks it exports (the points, the targets) are exactly what the RUN handed it. Each
case builds a `MemberSetRun` as the door's UNPACK would (one plan per member, the kind's own answers) and checks:

- draws: bootstrap-style row draws with repeats and a column draw: every member's fitted rows are the drawn rows
  of X at the drawn columns and the drawn targets, both when the members take their turns on this thread (the
  k-NN kind's own word, `MEMBER_GIL_DOOR`) and when they run on host workers;
- own block: a member over a block of its own reads that block, not the set's X;
- class cover: a classifier member whose drawn rows lack a class, or whose codes hold a class outside the
  ensemble's, fails the whole set before any fit (no state is left) with the words the door raises;
- classifier draws: a covered classifier set fits and each member holds its drawn codes;
- failure: the first failing member's words in member order are raised and every state is released.

The cases are independent and run as one case table across the host cores.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from core.device_route import DeviceRoute
from core.entry_table import EntryRef
from core.neighbors_member import neighbors_member_v2
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_kind_capabilities,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_EXPORT, MEMBER_BLOCK_FAMILY, MEMBER_F32, MEMBER_I32, MEMBER_GIL_FREE,
    MEMBER_GIL_DOOR, MEMBER_BORROWS_FIT, MEMBER_CAN_TEST_ROWS,
)
from core.member_set_run import (
    MemberSetRun, member_set_fit, member_set_run_release, MEMBER_SET_LACKING, MEMBER_SET_WORKER,
    MEMBER_SET_TURN_RELEASED, MEMBER_SET_TURN_HELD,
)

comptime N = 40
comptime D = 4
comptime K = 3
comptime CASES = 6


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _x() -> List[Float32]:
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture: one 40 x 4 block)
        for j in range(D):
            x[i * D + j] = Float32((i * 13 + j * 7) % 17) * 0.25
    return x^


def _targets() -> List[Float32]:
    var y = List[Float32](length=N, fill=0.0)
    for i in range(N):  # small-loop(fixture: the regression targets)
        y[i] = Float32(i) * 0.5 - 3.0
    return y^


def _codes() -> List[Int32]:
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: the class codes)
        y[i] = Int32(i % K)
    return y^


def _plan(classifier: Bool) raises -> MemberPlan:
    """A k-NN member (one neighbor, euclidean, uniform) on the CPU route, its family mask asked as the door's
    `member_resolve_family` asks it."""
    var plan = MemberPlan()
    plan.route = DeviceRoute(0)
    plan.member = MemberRef(EntryRef(neighbors_member_v2, 0), 0)
    var probe = MemberCall(DeviceRoute(0))
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params = [0.0 if classifier else 1.0, 1.0, 1.0, 2.0, 0.0, 0.0]
    plan.classes = K if classifier else 0
    return plan^


def _add(mut run: MemberSetRun, classifier: Bool, target: Int, weights: Int, row_draw: Int, row_count: Int,
         column_draw: Int, column_count: Int, workers: Bool) raises:
    """One member, with its kind's three answers as `member_set_admit` reads them (`workers`: run it on a host
    worker whatever its kind says, the mode a `MEMBER_GIL_FREE` kind takes)."""
    var plan = _plan(classifier)
    var mask = member_kind_capabilities(plan)
    var tests = UInt8(1) if (mask & MEMBER_CAN_TEST_ROWS) != 0 else UInt8(0)
    var borrows = UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0)
    var gil = MEMBER_SET_TURN_HELD
    if workers or (mask & MEMBER_GIL_FREE) != 0:
        gil = MEMBER_SET_WORKER
    elif (mask & MEMBER_GIL_DOOR) != 0:
        gil = MEMBER_SET_TURN_RELEASED
    run.add(plan^, target, weights, row_draw, row_count, column_draw, column_count, tests, borrows, gil)


def _block(run: MemberSetRun, u: Int, block: Int) raises -> MemberMatrix:
    var call = MemberCall(DeviceRoute(0), run.states[u])
    call.block = block
    _ = member_call(run.plans[u].member, MEMBER_OP_EXPORT, call)
    return call.dst


def _f32(view: MemberMatrix, i: Int) -> Float32:
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=view.data)[i]


def _i32(view: MemberMatrix, i: Int) -> Int32:
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=view.data)[i]


def _draw(u: Int, count: Int) -> List[Int32]:
    """Member `u`'s bootstrap-style draw: `count` rows with repeats."""
    var rows = List[Int32](length=count, fill=Int32(0))
    for i in range(count):  # small-loop(fixture: one member's draw)
        rows[i] = Int32((u * 7 + i * 11 + (i * i) % 5) % N)
    return rows^


def _check_fitted(run: MemberSetRun, u: Int, x: List[Float32], y_f: List[Float32], y_c: List[Int32],
                  rows: List[Int32], cols: List[Int32], label: String) raises:
    """Member `u` holds the drawn rows (all rows when `rows` is empty) at the drawn columns (all when empty) and
    their targets (codes when `y_c` is given)."""
    require(run.states[u] != 0, label + ": member " + String(u) + " holds no state")
    var points = _block(run, u, MEMBER_BLOCK_FAMILY)
    var targets = _block(run, u, MEMBER_BLOCK_FAMILY + 1)
    var n = len(rows) if len(rows) > 0 else N
    var width = len(cols) if len(cols) > 0 else D
    require(points.rows == n and points.cols == width, label + ": member " + String(u) + " points "
            + String(points.rows) + " x " + String(points.cols) + ", want " + String(n) + " x " + String(width))
    for i in range(n):  # small-loop(oracle: one member's fitted rows)
        var r = Int(rows[i]) if len(rows) > 0 else i
        for j in range(width):
            var c = Int(cols[j]) if len(cols) > 0 else j
            var got = _f32(points, i * width + j)
            require(got == x[r * D + c], label + ": member " + String(u) + " point " + String(i) + "," + String(j)
                    + " " + String(got) + " != X[" + String(r) + "," + String(c) + "] " + String(x[r * D + c]))
        if len(y_c) > 0:
            require(_i32(targets, i) == y_c[r], label + ": member " + String(u) + " code " + String(i))
        else:
            require(_f32(targets, i) == y_f[r], label + ": member " + String(u) + " target " + String(i))


def _no_state(run: MemberSetRun, label: String) raises:
    for u in range(len(run.states)):  # small-loop(oracle: every member's state word)
        require(run.states[u] == 0, label + ": member " + String(u) + " still holds a state")


def case_draws(workers: Bool) raises:
    """Six regressor members with row draws (repeats), every odd one also a column draw."""
    var label = String("draws (workers)") if workers else String("draws (turns)")
    var x = _x()
    var y = _targets()
    var cols: List[Int32] = [3, 1]
    var draws = List[List[Int32]]()
    var run = MemberSetRun(Int(x.unsafe_ptr()), N, D, MEMBER_F32, MEMBER_I32)
    for u in range(6):  # small-loop(fixture: the members)
        draws.append(_draw(u, 25 + u))
    for u in range(6):  # small-loop(fixture: the members)
        var odd = u % 2 == 1
        _add(run, False, Int(y.unsafe_ptr()), 0, Int(draws[u].unsafe_ptr()), len(draws[u]),
             Int(cols.unsafe_ptr()) if odd else 0, 2 if odd else 0, workers)
    try:
        member_set_fit(run)
        if workers:
            require(len(run.parallel) == 6 and run.workers >= 1, label + ": " + String(len(run.parallel))
                    + " worker members, " + String(run.workers) + " workers")
        else:
            require(len(run.parallel) == 0, label + ": a k-NN member takes its turn on the calling thread")
        for u in range(6):  # small-loop(oracle: the members)
            if u % 2 == 1:
                _check_fitted(run, u, x, y, List[Int32](), draws[u], cols, label)
            else:
                _check_fitted(run, u, x, y, List[Int32](), draws[u], List[Int32](), label)
    finally:
        member_set_run_release(run)
        _ = len(x)
        _ = len(y)
        _ = len(cols)
        _ = len(draws)
    _no_state(run, label)


def case_own_block() raises:
    """Member 0 over the set's X, member 1 over a 10-row block of its own (rows 30 .. 39 of X, reversed)."""
    var x = _x()
    var y = _targets()
    var own = List[Float32](length=10 * D, fill=0.0)
    var own_y = List[Float32](length=10, fill=0.0)
    var rows = List[Int32](length=10, fill=Int32(0))
    for i in range(10):  # small-loop(fixture: the block)
        rows[i] = Int32(39 - i)
        own_y[i] = y[39 - i]
        for j in range(D):
            own[i * D + j] = x[(39 - i) * D + j]
    var run = MemberSetRun(Int(x.unsafe_ptr()), N, D, MEMBER_F32, MEMBER_I32)
    _add(run, False, Int(y.unsafe_ptr()), 0, 0, 0, 0, 0, False)
    _add(run, False, Int(own_y.unsafe_ptr()), 0, 0, 0, 0, 0, False)
    run.own_block(Int(own.unsafe_ptr()), 10)
    try:
        member_set_fit(run)
        _check_fitted(run, 0, x, y, List[Int32](), List[Int32](), List[Int32](), "own block")
        _check_fitted(run, 1, x, y, List[Int32](), rows, List[Int32](), "own block")
    finally:
        member_set_run_release(run)
        _ = len(x)
        _ = len(y)
        _ = len(own)
        _ = len(own_y)


def case_cover() raises:
    """A drawn member without class 2 fails the set before any fit; so do codes outside the ensemble's classes."""
    var x = _x()
    var c = _codes()
    var lacking = List[Int32]()
    for i in range(N):  # small-loop(fixture: the rows whose code is not 2)
        if c[i] != Int32(2):
            lacking.append(Int32(i))
    for which in range(2):  # small-loop(two refusals: a class lacking, a code out of range)
        var run = MemberSetRun(Int(x.unsafe_ptr()), N, D, MEMBER_I32, MEMBER_I32)
        _add(run, True, Int(c.unsafe_ptr()), 0, 0, 0, 0, 0, which == 1)
        if which == 0:
            _add(run, True, Int(c.unsafe_ptr()), 0, Int(lacking.unsafe_ptr()), len(lacking), 0, 0, False)
        else:
            run.plans[0].classes = 2  # codes 0 .. 2 against an ensemble of two classes
        var raised = String("")
        try:
            member_set_fit(run)
        except problem:
            raised = String(problem)
        member_set_run_release(run)
        require(raised == MEMBER_SET_LACKING, "cover " + String(which) + ": raised '" + raised + "'")
        _no_state(run, "cover " + String(which))
    _ = len(x)
    _ = len(c)
    _ = len(lacking)


def case_classifier_draws() raises:
    """Four classifier members, each draw holding every class: each member holds its drawn codes."""
    var x = _x()
    var c = _codes()
    var draws = List[List[Int32]]()
    for u in range(4):  # small-loop(fixture: the members)
        draws.append(_draw(u + 3, 30))
    var run = MemberSetRun(Int(x.unsafe_ptr()), N, D, MEMBER_I32, MEMBER_I32)
    for u in range(4):  # small-loop(fixture: the members)
        _add(run, True, Int(c.unsafe_ptr()), 0, Int(draws[u].unsafe_ptr()), len(draws[u]), 0, 0, u % 2 == 0)
    try:
        member_set_fit(run)
        for u in range(4):  # small-loop(oracle: the members)
            _check_fitted(run, u, x, List[Float32](), c, draws[u], List[Int32](), "classifier draws")
    finally:
        member_set_run_release(run)
        _ = len(x)
        _ = len(c)
        _ = len(draws)


def case_failure() raises:
    """Members 1 and 2 are given sample weights a k-NN member refuses: the set raises member 1's words and keeps
    no state (member 0 fitted and was released)."""
    var x = _x()
    var y = _targets()
    var w = List[Float32](length=N, fill=1.0)
    var run = MemberSetRun(Int(x.unsafe_ptr()), N, D, MEMBER_F32, MEMBER_I32)
    _add(run, False, Int(y.unsafe_ptr()), 0, 0, 0, 0, 0, False)
    _add(run, False, Int(y.unsafe_ptr()), Int(w.unsafe_ptr()), 0, 0, 0, 0, False)
    _add(run, False, Int(y.unsafe_ptr()), Int(w.unsafe_ptr()), 0, 0, 0, 0, True)
    var raised = String("")
    try:
        member_set_fit(run)
    except problem:
        raised = String(problem)
    member_set_run_release(run)
    require(raised.find("fits no sample_weight") >= 0, "failure: raised '" + raised + "'")
    require(run.failed[1] != UInt8(0) and run.failed[2] != UInt8(0) and run.failed[0] == UInt8(0),
            "failure: the failing members are 1 and 2")
    _no_state(run, "failure")
    _ = len(x)
    _ = len(y)
    _ = len(w)


def run_case(which: Int) raises:
    if which == 0:
        case_draws(False)
    elif which == 1:
        case_draws(True)
    elif which == 2:
        case_own_block()
    elif which == 3:
        case_cover()
    elif which == 4:
        case_classifier_draws()
    else:
        case_failure()


def test_member_set_run() raises:
    check_cases("member_set_run", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
