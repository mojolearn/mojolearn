# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""CalibratedClassifierCV.fit's typed RUN (xtrees/trees_calibrated_fit_run.mojo `calibrated_fit_run`, lane mf-cal,
2026-11-23) over NATIVE members, in a process with no Python.

Every case builds the member set the door's UNPACK (bindings/trees_public_meta_door.mojo `_cal_native`) builds:
one unit per cv fold over X with the fold's train rows as its row draw and the ensemble's int32 codes (plus the
full-X unit for ensemble=False), then runs the RUN twice, with one fold worker (`cap` 1) and with four, and
requires:

- the two runs equal word for word (the folds are independent units: the parallel fit changes no bit);
- every calibrator word equal to the CURRENT DOOR's (`fixtures/mf_cal/`, written by
  tools/py2mojo_native_fixtures_mf_cal.py against the base build 4d76a9e16e): Platt (a, b), the isotonic knot
  count and knots, in pair order then class column.

Configurations: {LogisticRegression (core/logistic_member.mojo, host QN entry), LinearSVC
(core/linear_svm_member.mojo, host LinearSVC entry)} x {sigmoid, isotonic} x {k = 2, k = 3} x ensemble
{True, False}: 16 cases; both members score through decision_function, as their estimators have one.

A fold that lacks a class (lane mf-cal2, `cal_fold_classes_run`): cases 16 to 18 (`y4gap`, k = 4, fold 0's train
rows lack class 3) fit fold 0's member natively on the three classes it holds, re-coded, and scatter its three
decision columns at those classes; every calibrator word equals the door's, which fitted that fold through the
member's fit door (fixtures `logreg-sigmoid-k4gap-e1`, `logreg-sigmoid-k4gap-e0`, `linsvc-isotonic-k4gap-e1`,
written against the round-1 base d3a8864541, whose door units fitted that fold). Case 19 (`y3gap`, k = 3, fold 0's
train rows hold classes 0 and 1): the two-class member's ONE decision column goes to class 1's column alone (the
door read two columns of it before mf-cal2, so no door words exist: the out-of-fold block's fold-0 rows must be zero
in columns 0 and 2 and not all zero in column 1), serial equal to parallel. Case 20 (`y2gap`, k = 2, fold 0's
train rows hold class 0 only; lane mf-cal3): fold 0's member is fitted with ONE class, as scikit-learn's
`_fit_classifier_calibrator_pair` fits the fold's estimator on whatever its train rows hold, and LogisticRegression's
fit refuses it in its fit door's words (core/logistic_model_state.mojo `logistic_fit_target_count`, the one body
both the door and the member call): the RUN answers CAL_FIT_FAILED with fold 0 the first failing member and its words
exactly "mojolearn LogisticRegression: y has one class", the other folds' words empty, every state released; case
21 is the same with four fold workers (the same first failure, the same words).

The member words are the admitted words of LogisticRegression(C=1.0, max_iter=100, linesearch_max_iter=20,
tol=1e-4) (providers/python/member_logistic_boundary_door.mojo) and LinearSVC(C=1.0)
(providers/python/member_linear_svm_boundary_door.mojo), the estimators the generator fits.

Fail-proof edits (each must fail the named cases): in `cal_calibrators_run` use `cls = j` for c = 1 (every k = 2
case); pass `plan.test_rows[(f + 1) % folds]` as `te` in `calibrated_fit_scores_run` (every case); in
`cal_fold_classes_run` leave `run.plans[f].classes` at k (cases 16 to 18: the fold's member is fitted with four
classes); in `_cal_fold_scores` drop `class_at += 4` (case 19: column 0 is written); in `cal_fold_classes_run`
skip a fold that holds one class (`if held < 2: continue`: cases 20 and 21, fold 0 is fitted with two classes and
answers).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.linear_svm_member import linear_svm_member_v2
from core.linear_svc_native_api import LINEAR_SVC_NATIVE_COUNT
from core.member_abi import (
    MemberCall, MemberPlan, MemberRef, member_call, member_kind_capabilities, MEMBER_OP_CAPABILITIES,
    MEMBER_OP_DECISION, MEMBER_I32, MEMBER_CAN_TEST_ROWS, MEMBER_BORROWS_FIT, MEMBER_GIL_FREE, MEMBER_GIL_DOOR,
)
from core.member_set_run import (
    MemberSetRun, member_set_run_release, member_set_first_failure, MEMBER_SET_WORKER, MEMBER_SET_TURN_RELEASED, MEMBER_SET_TURN_HELD,
)
from core.trees_entry_table import TreesEntryKey
from glm.host.logistic_native_host import qn_host_native_v1
from glm.host.linear_svc_native_host import linear_svc_host_v1
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import xt_ref
from xtrees.trees_calibrated_fit_run import CalFitPlan, CalFitOut, CAL_ALL_ANSWERED, CAL_FIT_FAILED, cal_columns, calibrated_fit_run

comptime N = 60
comptime D = 3
comptime FOLDS = 3
comptime CASES = 22
comptime DIR = "checks/native_tests/fixtures/mf_cal/"


def _glue() raises -> EntryRef:
    return xt_ref(TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


def _plan(linsvc: Bool, k: Int) raises -> MemberPlan:
    """The member plan the family's admit builds for the generator's estimator, on the CPU image's entry."""
    var plan = MemberPlan()
    if linsvc:
        var route = DeviceRoute(DEVICE_CPU)
        route.mode = Int(GLOBAL_NUMERIC_MODE)
        route.vendor = 0
        route.entry = EntryRef(linear_svc_host_v1, entry_ref_word(route.mode, 0, LINEAR_SVC_NATIVE_COUNT))
        route.flags = 3  # LINEAR_SVC_ROUTE_ROWS | LINEAR_SVC_ROUTE_MULTICLASS (the admit requires both)
        plan.route = route
        plan.member = MemberRef(EntryRef(linear_svm_member_v2, 0), 0)
        # kind, fit_intercept, max_iter, linesearch_max_iter, lbfgs_memory, penalty_normalized, loss
        # (squared_hinge), l1, l2 = 1/C, tol, epsilon
        plan.params = [0.0, 1.0, 1000.0, 100.0, 5.0, 1.0, 4.0, 0.0, 1.0, 1e-4, 0.0]
    else:
        plan.route = logistic_route_pack(LogisticRoute(DEVICE_CPU, Int(GLOBAL_NUMERIC_MODE), 0, False, False,
                                                       QnNativeDriver(qn_host_native_v1)))
        plan.member = MemberRef(EntryRef(logistic_member_v2, 0), 0)
        # kind, fit_intercept, l1, l2, max_iter, linesearch_max_iter, lbfgs_memory, penalty_normalized, tol,
        # change tolerance (tol * 0.01)
        plan.params = [0.0, 1.0, 0.0, 1.0, 100.0, 20.0, 5.0, 1.0, 1e-4, 1e-6]
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.classes = k
    return plan^


def _folds() -> List[List[Int32]]:
    """[train 0, test 0, train 1, test 1, ...]: fold f's test rows are i % 3 == f (the generator's cv)."""
    var out = List[List[Int32]]()
    for f in range(FOLDS):  # small-loop(fixture)
        var tr = List[Int32]()
        var te = List[Int32]()
        for i in range(N):  # small-loop(fixture)
            if i % FOLDS == f:
                te.append(Int32(i))
            else:
                tr.append(Int32(i))
        out.append(tr^)
        out.append(te^)
    return out^


def _run(linsvc: Bool, sigmoid: Bool, k: Int, ensemble: Bool, cap: Int, x: List[Float32], y: List[Int32],
         folds: List[List[Int32]], mut res: CalFitOut) raises -> Int:
    """The door's UNPACK as typed values, then the RUN. The states are released before returning."""
    var failed = -1
    var words = List[String]()
    return _run_words(linsvc, sigmoid, k, ensemble, cap, x, y, folds, res, failed, words)


def _run_words(linsvc: Bool, sigmoid: Bool, k: Int, ensemble: Bool, cap: Int, x: List[Float32], y: List[Int32],
               folds: List[List[Int32]], mut res: CalFitOut, mut failed: Int, mut words: List[String]) raises -> Int:
    """`_run`, also giving the first failing member (`member_set_first_failure`, -1: none) and every member's
    recorded words."""
    var plan = _plan(linsvc, k)
    var mask = member_kind_capabilities(plan)
    var tests = UInt8(1) if (mask & MEMBER_CAN_TEST_ROWS) != 0 else UInt8(0)
    var borrows = UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0)
    var gil = MEMBER_SET_TURN_HELD
    if (mask & MEMBER_GIL_FREE) != 0:
        gil = MEMBER_SET_WORKER
    elif (mask & MEMBER_GIL_DOOR) != 0:
        gil = MEMBER_SET_TURN_RELEASED
    var run = MemberSetRun(Int(x.unsafe_ptr()), N, D, MEMBER_I32, MEMBER_I32)
    run.cap = cap
    var units = FOLDS if ensemble else FOLDS + 1
    for u in range(units):  # small-loop(fixture: the door's admission per unit)
        var draw = 0
        var count = 0
        if u < FOLDS:
            draw = Int(folds[2 * u].unsafe_ptr())
            count = len(folds[2 * u])
        run.add(plan.copy(), Int(y.unsafe_ptr()), 0, draw, count, 0, 0, tests, borrows, gil)
    var cal = CalFitPlan(Int(x.unsafe_ptr()), N, D, Int(y.unsafe_ptr()), k, MEMBER_OP_DECISION, sigmoid, ensemble,
                         False)
    for f in range(FOLDS):  # small-loop(fixture)
        cal.fold(Int(folds[2 * f + 1].unsafe_ptr()), len(folds[2 * f + 1]))
    var answered = CAL_ALL_ANSWERED
    try:
        answered = calibrated_fit_run(_glue(), run, cal, res)
    finally:
        member_set_run_release(run)
    failed = member_set_first_failure(run)
    for u in range(len(run.errors)):  # small-loop(oracle: every member's words)
        words.append(run.errors[u].copy())
    _ = len(x)  # read through the fits and the RUN above
    _ = len(y)
    _ = len(folds)
    _ = cal.folds()  # its re-coded targets were read through the fits above
    return answered


def _same(a: CalFitOut, b: CalFitOut, what: String) raises:
    require(a.calibrators() == b.calibrators(), what + ": calibrator counts differ")
    for j in range(a.calibrators()):  # small-loop(oracle: every word)
        require(a.isotonic[j] == b.isotonic[j] and a.a[j] == b.a[j] and a.b[j] == b.b[j] and a.m[j] == b.m[j],
                what + ": calibrator " + String(j) + " differs")
        for i in range(a.m[j]):  # small-loop(oracle: every knot)
            require(a.knots_x[j][i] == b.knots_x[j][i] and a.knots_y[j][i] == b.knots_y[j][i],
                    what + ": calibrator " + String(j) + " knot " + String(i) + " differs")
    require(len(a.scores) == len(b.scores), what + ": out-of-fold lengths differ")
    for i in range(len(a.scores)):  # small-loop(oracle: every word)
        require(a.scores[i] == b.scores[i], what + ": out-of-fold score " + String(i) + " differs")


def _door(res: CalFitOut, name: String) raises:
    """Every calibrator word against the door's (fixture), in pair order then class column."""
    var ab = fixture_f64(DIR + name + "-ab.f64", -1)
    var ms = fixture_i32(DIR + name + "-m.i32", -1)
    require(len(ms) == res.calibrators(), name + ": " + String(res.calibrators()) + " calibrators, the door wrote "
            + String(len(ms)))
    var knots = List[Float64]()
    var total = 0
    for j in range(len(ms)):  # small-loop(oracle)
        total += Int(ms[j])
    if total > 0:
        knots = fixture_f64(DIR + name + "-knots.f64", 2 * total)
    var at = 0
    for j in range(len(ms)):  # small-loop(oracle: every calibrator)
        var m = Int(ms[j])
        require(res.m[j] == m, name + ": calibrator " + String(j) + " has " + String(res.m[j]) + " knots, the door "
                + String(m))
        if m == 0:
            require(not res.isotonic[j] and res.a[j] == ab[2 * j] and res.b[j] == ab[2 * j + 1],
                    name + ": Platt calibrator " + String(j) + " (" + String(res.a[j]) + ", " + String(res.b[j])
                    + ") != the door's (" + String(ab[2 * j]) + ", " + String(ab[2 * j + 1]) + ")")
        else:
            for i in range(m):  # small-loop(oracle: every knot)
                require(res.knots_x[j][i] == knots[at + i] and res.knots_y[j][i] == knots[at + m + i],
                        name + ": isotonic calibrator " + String(j) + " knot " + String(i) + " != the door's")
            at += 2 * m


def check_config(which: Int) raises:
    var linsvc = (which // 8) % 2 == 1
    var sigmoid = (which // 4) % 2 == 0
    var k = 2 if (which // 2) % 2 == 0 else 3
    var ensemble = which % 2 == 0
    var name = (String("linsvc") if linsvc else String("logreg")) + "-" + (String("sigmoid") if sigmoid
                else String("isotonic")) + "-k" + String(k) + "-e" + (String("1") if ensemble else String("0"))
    var x = fixture_f32(DIR + "x.f32", N * D)
    var y = fixture_i32(DIR + ("y2.i32" if k == 2 else "y3.i32"), N)
    var folds = _folds()
    var serial = CalFitOut()
    var parallel = CalFitOut()
    require(_run(linsvc, sigmoid, k, ensemble, 1, x, y, folds, serial) == CAL_ALL_ANSWERED, name + ": a member refused")
    require(_run(linsvc, sigmoid, k, ensemble, 4, x, y, folds, parallel) == CAL_ALL_ANSWERED,
            name + ": a member refused (parallel)")
    require(ensemble or serial.final_count == cal_columns(k),
            name + ": " + String(serial.final_count) + " final calibrators")
    _same(serial, parallel, name + " serial vs parallel")
    _door(serial, name)


def check_lacking_fold(which: Int) raises:
    """Cases 16 to 18: fold 0's train rows lack class 3 of four; fitted natively on the three it holds, every
    calibrator word the door's."""
    var linsvc = which == 18
    var sigmoid = which != 18
    var ensemble = which != 17
    var name = (String("linsvc") if linsvc else String("logreg")) + "-" + (String("sigmoid") if sigmoid
                else String("isotonic")) + "-k4gap-e" + (String("1") if ensemble else String("0"))
    var x = fixture_f32(DIR + "x.f32", N * D)
    var y = fixture_i32(DIR + "y4gap.i32", N)
    var folds = _folds()
    var serial = CalFitOut()
    var parallel = CalFitOut()
    require(_run(linsvc, sigmoid, 4, ensemble, 1, x, y, folds, serial) == CAL_ALL_ANSWERED, name + ": a member refused")
    require(_run(linsvc, sigmoid, 4, ensemble, 4, x, y, folds, parallel) == CAL_ALL_ANSWERED,
            name + ": a member refused (parallel)")
    require(ensemble or serial.final_count == cal_columns(k),
            name + ": " + String(serial.final_count) + " final calibrators")
    _same(serial, parallel, name + " serial vs parallel")
    _door(serial, name)


def check_two_class_fold() raises:
    """Case 19: fold 0's train rows hold classes 0 and 1 of three; its member's one decision column is class 1's."""
    var x = fixture_f32(DIR + "x.f32", N * D)
    var y = fixture_i32(DIR + "y3gap.i32", N)
    var folds = _folds()
    var serial = CalFitOut()
    var parallel = CalFitOut()
    require(_run(False, True, 3, False, 1, x, y, folds, serial) == CAL_ALL_ANSWERED, "gap3: a member refused")
    require(_run(False, True, 3, False, 4, x, y, folds, parallel) == CAL_ALL_ANSWERED,
            "gap3: a member refused (parallel)")
    require(serial.final_count == 3, "gap3: " + String(serial.final_count) + " final calibrators")
    _same(serial, parallel, "gap3 serial vs parallel")
    var written = False
    for i in range(0, N, FOLDS):  # small-loop(oracle: fold 0's test rows)
        require(serial.scores[3 * i] == 0.0 and serial.scores[3 * i + 2] == 0.0,
                "gap3: row " + String(i) + " has a score outside class 1's column")
        if serial.scores[3 * i + 1] != 0.0:
            written = True
    require(written, "gap3: fold 0's decision was not placed in class 1's column")


def check_one_class_fold(cap: Int) raises:
    """Cases 20 (one fold worker) and 21 (four): fold 0's train rows hold one class of two; its member is fitted
    with one class and LogisticRegression's fit refuses it in its door's words; nothing is published."""
    var x = fixture_f32(DIR + "x.f32", N * D)
    var y = fixture_i32(DIR + "y2gap.i32", N)
    var folds = _folds()
    var res = CalFitOut()
    var failed = -1
    var words = List[String]()
    var name = "gap2 cap " + String(cap)
    require(_run_words(False, True, 2, False, cap, x, y, folds, res, failed, words) == CAL_FIT_FAILED,
            name + ": the one-class fold was fitted")
    require(failed == 0, name + ": the first failing member is " + String(failed) + ", not fold 0")
    require(words[0] == "mojolearn LogisticRegression: y has one class",
            name + ": fold 0 raised '" + words[0] + "', not its fit door's words")
    for u in range(1, len(words)):  # small-loop(oracle: the covered folds and the full-X member)
        require(len(words[u]) == 0, name + ": member " + String(u) + " failed: " + words[u])
    require(res.calibrators() == 0, name + ": calibrators were fitted after a failed fit")


def test_trees_calibrated_fit() raises:
    # The publication _mojolearn_x_trees_host makes at module init, once before the parallel cases.
    publish_x_trees_glue("cpu")
    def run_case(which: Int) raises:
        if which < 16:
            check_config(which)
        elif which < 19:
            check_lacking_fold(which)
        elif which == 19:
            check_two_class_fold()
        else:
            check_one_class_fold(1 if which == 20 else 4)
    check_cases("trees_calibrated_fit", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
