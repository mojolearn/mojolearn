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

A fold that lacks a class (lane mf-cal2, `cal_fold_classes_run`; placed by THE PLACEMENT RULE since lane mf-cal3b,
`cal_fold_place`, read from scikit-learn 1.9's calibration.py and `_enforce_prediction_order`): cases 16 to 18
(`y4gap`, k = 4, fold 0's train rows lack class 3) fit fold 0's member natively on the three classes it holds,
re-coded, and place its three decision columns at those classes. ensemble=True (16, 18): fold 0 has THREE
calibrators (scikit-learn's zip of pos_class_indices with the score columns), each word equal to the door's first
three for that fold (fixtures `logreg-sigmoid-k4gap-e1`, `linsvc-isotonic-k4gap-e1`, written against the round-1 base
d3a8864541, whose door fitted a fourth for the lacking class: skipped), every other calibrator the door's.
ensemble=False (17): fold 0's test rows hold float32's lowest word in class 3's column of the out-of-fold block (the
default `_enforce_prediction_order` gives a decision_function's lacking class); the final calibrators of classes 0 to
2 are the door's (`logreg-sigmoid-k4gap-e0`), class 3's is skipped (fitted over the filled column now). Case 19
(`y3gap`, k = 3, fold 0's train rows hold classes 0 and 1, ensemble=False, decision_function): scikit-learn's
refusal "Only 2 class/es in training fold, but 3 in overall dataset. ..." raised by the RUN, nothing fitted kept.
Case 22: the same folds with ensemble=True: fold 0 has ONE calibrator, fitted on its member's one decision column
against class 0 (oracle: the member fitted alone on fold 0's two classes, its decision placed by hand, the
calibrator glue over that column); folds 1 and 2 three each; serial equal to parallel. Case 20 (`y2gap`, k = 2,
fold 0's train rows hold class 0 only; lane mf-cal3): fold 0's member is fitted with ONE class, as scikit-learn's
`_fit_classifier_calibrator_pair` fits the fold's estimator on whatever its train rows hold, and LogisticRegression's
fit refuses it in its fit door's words (core/logistic_model_state.mojo `logistic_fit_target_count`, the one body
both the door and the member call): the RUN answers CAL_FIT_FAILED with fold 0 the first failing member and its words
exactly "mojolearn LogisticRegression: y has one class", the other folds' words empty, every state released; case
21 is the same with four fold workers (the same first failure, the same words). Case 23: `cal_fold_place` branch by
branch (placements, calibrated columns, fills, scikit-learn's refusal words). Case 24: THE PREDICT RULE
(xtrees/calibrated_apply.mojo `_cal_column`) on the host program: a pair whose member holds classes 1 and 2 of three,
placed at [2, 1], one calibrator, writes class 1's column only.

The member words are the admitted words of LogisticRegression(C=1.0, max_iter=100, linesearch_max_iter=20,
tol=1e-4) (providers/python/member_logistic_boundary_door.mojo) and LinearSVC(C=1.0)
(providers/python/member_linear_svm_boundary_door.mojo), the estimators the generator fits.

Fail-proof edits (each must fail the named cases): in `cal_calibrators_run` use `cls = j` for c = 1 (every k = 2
case); pass `plan.test_rows[(f + 1) % folds]` as `te` in `calibrated_fit_scores_run` (every case); in
`cal_fold_classes_run` leave `run.plans[f].classes` at k (cases 16 to 18: the fold's member is fitted with four
classes); in `cal_fold_classes_run` skip a fold that holds one class (`if held < 2: continue`: cases 20 and 21, fold 0
is fitted with two classes and answers); in `cal_fold_place` calibrate every held class of a two-class fold, or
place its one column at `held[1]` (cases 22 and 23); drop the `got.fill` line (cases 17 and 23); fit `range(c)`
calibrators whatever `which` says in `cal_calibrators_run` (cases 16, 18, 22); in `_cal_column` return `j` (case 24).
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
    MemberCall, MemberPlan, MemberRef, member_call, member_call_of, member_outputs_of, member_kind_capabilities,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_DECISION, MEMBER_OP_PREDICT_PROBA, MEMBER_I32, MEMBER_CAN_TEST_ROWS,
    MEMBER_BORROWS_FIT, MEMBER_GIL_FREE, MEMBER_GIL_DOOR,
)
from core.member_set_run import (
    MemberSetRun, member_set_run_release, member_set_first_failure, member_set_fit, MEMBER_SET_WORKER,
    MEMBER_SET_TURN_RELEASED, MEMBER_SET_TURN_HELD,
)
from core.trees_entry_table import TreesEntryKey
from glm.host.logistic_native_host import qn_host_native_v1
from glm.host.linear_svc_native_host import linear_svc_host_v1
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import xt_ref
from xtrees.trees_calibrated_fit_run import (
    CalFitPlan, CalFitOut, CalPlace, CAL_ALL_ANSWERED, CAL_FIT_FAILED, cal_columns, calibrated_fit_run, cal_fold_place,
    cal_calibrators_run,
)
from xtrees.trees_meta_vote_run import VoteMember, VoteRows, member_f32_block
from xtrees.calibrated_apply import CalPair, CAL_SIGMOID, calibrated_proba_host

comptime N = 60
comptime D = 3
comptime FOLDS = 3
comptime CASES = 25
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
    var cal = CalFitPlan(Int(x.unsafe_ptr()), N, D, Int(y.unsafe_ptr()), k, MEMBER_OP_DECISION, sigmoid, ensemble)
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


def _door(res: CalFitOut, name: String, skip: Int = -1, kept: Bool = False) raises:
    """Every calibrator word against the door's (fixture), in pair order then class column. `skip` >= 0: the
    door's calibrator `skip` has no counterpart (lane mf-cal3b: a fold that lacks a class has no calibrator for
    it, as scikit-learn's `_fit_calibrator` fits none; the door's fixture, written before, fitted one), or with
    `kept` it is there but fitted over other words (not compared)."""
    var ab = fixture_f64(DIR + name + "-ab.f64", -1)
    var ms = fixture_i32(DIR + name + "-m.i32", -1)
    var dropped = 1 if skip >= 0 and not kept else 0
    require(len(ms) - dropped == res.calibrators(), name + ": " + String(res.calibrators())
            + " calibrators, the door wrote " + String(len(ms)) + " (" + String(dropped) + " dropped)")
    var knots = List[Float64]()
    var total = 0
    for j in range(len(ms)):  # small-loop(oracle)
        total += Int(ms[j])
    if total > 0:
        knots = fixture_f64(DIR + name + "-knots.f64", 2 * total)
    var at = 0
    for j in range(len(ms)):  # small-loop(oracle: every calibrator)
        var m = Int(ms[j])
        if j == skip:
            at += 2 * m
            continue
        var r = j - 1 if (skip >= 0 and j > skip and not kept) else j
        require(res.m[r] == m, name + ": calibrator " + String(r) + " has " + String(res.m[r]) + " knots, the door "
                + String(m))
        if m == 0:
            require(not res.isotonic[r] and res.a[r] == ab[2 * j] and res.b[r] == ab[2 * j + 1],
                    name + ": Platt calibrator " + String(r) + " (" + String(res.a[r]) + ", " + String(res.b[r])
                    + ") != the door's (" + String(ab[2 * j]) + ", " + String(ab[2 * j + 1]) + ")")
        else:
            for i in range(m):  # small-loop(oracle: every knot)
                require(res.knots_x[r][i] == knots[at + i] and res.knots_y[r][i] == knots[at + m + i],
                        name + ": isotonic calibrator " + String(r) + " knot " + String(i) + " != the door's")
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
    require(ensemble or serial.final_count == cal_columns(4),
            name + ": " + String(serial.final_count) + " final calibrators")
    _same(serial, parallel, name + " serial vs parallel")
    if ensemble:
        # fold 0 holds classes 0, 1, 2: three calibrators (scikit-learn's zip), the door's first three
        require(serial.count[0] == 3 and serial.count[1] == 4 and serial.count[2] == 4,
                name + ": fold calibrator counts " + String(serial.count[0]) + ", " + String(serial.count[1]) + ", "
                + String(serial.count[2]))
        _door(serial, name, 3)
        return
    # ensemble=False: fold 0's rows hold float32's lowest word in class 3's column (`_enforce_prediction_order`);
    # the final calibrators of classes 0 .. 2 are the door's, class 3's is fitted over the filled column
    for i in range(0, N, FOLDS):  # small-loop(oracle: fold 0's test rows)
        require(serial.scores[4 * i + 3] == Float64(Float32.MIN_FINITE),
                name + ": row " + String(i) + " class 3 holds " + String(serial.scores[4 * i + 3]))
    _door(serial, name, 3, True)


def check_two_class_fold() raises:
    """Case 19: fold 0's train rows hold classes 0 and 1 of three, ensemble=False, decision_function: scikit-learn's
    `_enforce_prediction_order` refusal of an imbalanced decision fold, in its words, from the RUN."""
    var x = fixture_f32(DIR + "x.f32", N * D)
    var y = fixture_i32(DIR + "y3gap.i32", N)
    var folds = _folds()
    var serial = CalFitOut()
    var words = String("")
    try:
        _ = _run(False, True, 3, False, 1, x, y, folds, serial)
    except error:
        words = String(error)
    require(words == "ValueError: Only 2 class/es in training fold, but 3 in overall dataset. This is not supported "
            + "for decision_function with imbalanced folds. To fix this, use a cross-validation technique resulting in "
            + "properly stratified folds", "gap3: raised '" + words + "'")
    require(serial.calibrators() == 0, "gap3: calibrators were fitted before the refusal")


def check_two_class_fold_ensemble() raises:
    """Case 22: the same folds, ensemble=True: fold 0's ONE calibrator is fitted on its member's one decision column
    against class 0 (scikit-learn's zip of pos_class_indices [0, 1] with one score column). Oracle: fold 0's member
    fitted alone on its train rows with two classes (their codes are 0 and 1 already), its decision over fold 0's
    test rows placed at column 0 by hand, `cal_calibrators_run` over column 0 only."""
    var x = fixture_f32(DIR + "x.f32", N * D)
    var y = fixture_i32(DIR + "y3gap.i32", N)
    var folds = _folds()
    var serial = CalFitOut()
    var parallel = CalFitOut()
    require(_run(False, True, 3, True, 1, x, y, folds, serial) == CAL_ALL_ANSWERED, "gap3e: a member refused")
    require(_run(False, True, 3, True, 4, x, y, folds, parallel) == CAL_ALL_ANSWERED,
            "gap3e: a member refused (parallel)")
    _same(serial, parallel, "gap3e serial vs parallel")
    require(serial.count[0] == 1 and serial.count[1] == 3 and serial.count[2] == 3,
            "gap3e: fold calibrator counts " + String(serial.count[0]) + ", " + String(serial.count[1]) + ", "
            + String(serial.count[2]))
    var plan = _plan(False, 2)
    var mask = member_kind_capabilities(plan)
    var gil = MEMBER_SET_WORKER if (mask & MEMBER_GIL_FREE) != 0 else MEMBER_SET_TURN_HELD
    var run = MemberSetRun(Int(x.unsafe_ptr()), N, D, MEMBER_I32, MEMBER_I32)
    run.cap = 1
    run.add(plan.copy(), Int(y.unsafe_ptr()), 0, Int(folds[0].unsafe_ptr()), len(folds[0]), 0, 0,
            UInt8(1) if (mask & MEMBER_CAN_TEST_ROWS) != 0 else UInt8(0),
            UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0), gil)
    var expected = CalFitOut()
    try:
        member_set_fit(run)
        var nte = len(folds[1])
        var xte = List[Float32](length=nte * D, fill=0.0)
        var codes_te = List[Int32](length=nte, fill=Int32(0))
        for i in range(nte):  # small-loop(oracle: fold 0's test rows)
            var row = Int(folds[1][i])
            codes_te[i] = y[row]
            for c in range(D):  # small-loop(oracle)
                xte[i * D + c] = x[row * D + c]
        var call = member_call_of(run.plans[0], run.states[0])
        var cols = member_outputs_of(run.plans[0], call, MEMBER_OP_DECISION)
        require(cols == 1, "gap3e: the two-class member answered " + String(cols) + " decision columns")
        var keep = List[List[Float32]]()
        var member = VoteMember(Int(run.plans.unsafe_ptr()), run.states[0], MEMBER_OP_DECISION, cols, call.out_dtype,
                                0, 1.0)
        var blk = member_f32_block(member, VoteRows(Int(xte.unsafe_ptr()), nte, D), keep)
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=blk)
        var s = List[Float64](length=nte * 3, fill=0.0)
        for i in range(nte):  # small-loop(oracle: column 0, the first held class)
            s[3 * i] = Float64(src[i])
        var which: List[Int] = [0]
        cal_calibrators_run(_glue(), Int(s.unsafe_ptr()), nte, 3, Int(codes_te.unsafe_ptr()), True, which, expected)
        _ = len(keep)
        _ = len(xte)
        _ = len(s)
        _ = len(codes_te)
    finally:
        member_set_run_release(run)
    var j = serial.first[0]
    require(expected.calibrators() == 1 and serial.a[j] == expected.a[0] and serial.b[j] == expected.b[0],
            "gap3e: fold 0's calibrator (" + String(serial.a[j]) + ", " + String(serial.b[j]) + ") != the oracle's ("
            + String(expected.a[0]) + ", " + String(expected.b[0]) + ")")
    _ = len(x)
    _ = len(y)
    _ = len(folds)


def _place_raises(k: Int, var held: List[Int32], operation: Int, cols: Int, ensemble: Bool, rows: Int,
                  words: String) raises:
    var got = String("")
    try:
        _ = cal_fold_place(k, held, operation, cols, ensemble, rows)
    except error:
        got = String(error)
    require(got == words, "placement k " + String(k) + ": raised '" + got + "', not '" + words + "'")


def _place_is(k: Int, var held: List[Int32], operation: Int, cols: Int, ensemble: Bool, var place: List[Int32],
              var calibrate: List[Int], pick: Int, fill: Float64) raises:
    var got = cal_fold_place(k, held, operation, cols, ensemble, 5)
    var what = "placement k " + String(k) + " held " + String(len(held)) + " cols " + String(cols)
    require(got.pick == pick and got.fill == fill, what + ": pick " + String(got.pick) + " fill " + String(got.fill))
    require(len(got.place) == len(place) and len(got.calibrate) == len(calibrate), what + ": lengths differ")
    for i in range(len(place)):  # small-loop(oracle)
        require(got.place[i] == place[i], what + ": place " + String(i) + " is " + String(got.place[i]))
    for i in range(len(calibrate)):  # small-loop(oracle)
        require(got.calibrate[i] == calibrate[i], what + ": calibrate " + String(i) + " is " + String(got.calibrate[i]))


def _c(*values: Int) -> List[Int32]:
    """A list of int32 class codes."""
    var out = List[Int32]()
    for v in values:  # small-loop(oracle)
        out.append(Int32(v))
    return out^


def _n(*values: Int) -> List[Int]:
    """A list of column numbers."""
    var out = List[Int]()
    for v in values:  # small-loop(oracle)
        out.append(v)
    return out^


def check_placement_rule() raises:
    """Case 23: THE PLACEMENT RULE (`cal_fold_place`) against scikit-learn 1.9's calibration.py and
    `_enforce_prediction_order`, branch by branch: placements, calibrated columns, fills and refusal words."""
    var D_ = MEMBER_OP_DECISION
    var P_ = MEMBER_OP_PREDICT_PROBA
    var lo = Float64(Float32.MIN_FINITE)
    # every class: as before
    _place_is(2, _c(0, 1), P_, 2, True, _c(), _n(0), 1, 0.0)
    _place_is(2, _c(0, 1), D_, 1, False, _c(), _n(0), 0, 0.0)
    _place_is(3, _c(0, 1, 2), D_, 3, True, _c(0, 1, 2), _n(0, 1, 2), 0, 0.0)
    # ensemble=True, a fold that lacks a class
    _place_is(3, _c(0, 2), D_, 1, True, _c(0), _n(0), 0, 0.0)
    _place_is(3, _c(0, 2), P_, 2, True, _c(2, 0), _n(0), 0, 0.0)
    _place_is(4, _c(0, 1, 3), P_, 3, True, _c(0, 1, 3), _n(0, 1, 3), 0, 0.0)
    _place_raises(3, _c(1), P_, 1, True, 7, "ValueError: Got predict_proba of shape (7, 1), but need classifier with "
                  + "two classes.")
    _place_raises(2, _c(1), P_, 1, True, 4, "ValueError: Got predict_proba of shape (4, 1), but need classifier with "
                  + "two classes.")
    # ensemble=False, a fold that lacks a class
    _place_is(2, _c(0), P_, 1, False, _c(), _n(), -1, 0.0)
    _place_is(2, _c(1), P_, 1, False, _c(), _n(), 0, 0.0)
    _place_is(4, _c(0, 1, 3), P_, 3, False, _c(0, 1, 3), _n(), 0, 0.0)
    _place_is(4, _c(0, 1, 3), D_, 3, False, _c(0, 1, 3), _n(), 0, lo)
    _place_raises(2, _c(0), D_, 1, False, 5, "ValueError: Only 1 class/es in training fold, but 2 in overall dataset. "
                  + "This is not supported for decision_function with imbalanced folds. To fix this, use a "
                  + "cross-validation technique resulting in properly stratified folds")
    _place_raises(4, _c(0, 1, 3), D_, 2, False, 5, "ValueError: Output shape (5, 2) of decision_function does not match "
                  + "number of classes (3) in fold. Irregular decision_function outputs are not currently supported "
                  + "by cross_val_predict")


def check_predict_rule() raises:
    """Case 24: THE PREDICT RULE (xtrees/calibrated_apply.mojo `_cal_column`) on the host program: a k = 3 pair whose
    member holds classes 1 and 2 answers two probability columns, placed at [2, 1] with ONE calibrator, which reads
    and writes class 1's column (the second class's probability against the first class: scikit-learn's
    `_CalibratedClassifier.predict_proba`); the row normalization then makes that column 1 and the others 0."""
    var n = 4
    var block: List[Float32] = [0.3, 0.7, 0.6, 0.4, 0.1, 0.9, 0.8, 0.2]
    var cols: List[Int32] = [2, 1]
    var pairs = List[CalPair]()
    var kinds: List[Int] = [CAL_SIGMOID]
    var a: List[Float64] = [-2.0]
    var b: List[Float64] = [0.5]
    var kx: List[Int] = [0]
    var ky: List[Int] = [0]
    var m: List[Int] = [0]
    pairs.append(CalPair(Int(block.unsafe_ptr()), 2, 0, Int(cols.unsafe_ptr()), 2, kinds^, a^, b^, kx^, ky^, m^))
    var acc = List[Float64](length=n * 3, fill=-1.0)
    calibrated_proba_host(pairs, n, 3, acc.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    for r in range(n):  # small-loop(oracle: every row)
        require(acc[3 * r] == 0.0 and acc[3 * r + 1] == 1.0 and acc[3 * r + 2] == 0.0,
                "predict rule: row " + String(r) + " is (" + String(acc[3 * r]) + ", " + String(acc[3 * r + 1]) + ", "
                + String(acc[3 * r + 2]) + ")")
    _ = len(block)
    _ = len(cols)


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
        require(words[u].byte_length() == 0, name + ": member " + String(u) + " failed: " + words[u])
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
        elif which < 22:
            check_one_class_fold(1 if which == 20 else 4)
        elif which == 22:
            check_two_class_fold_ensemble()
        elif which == 23:
            check_placement_rule()
        else:
            check_predict_rule()
    check_cases("trees_calibrated_fit", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
