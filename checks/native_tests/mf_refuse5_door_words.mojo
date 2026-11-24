# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE DOOR'S REFUSAL (lane mf-refuse5; core/ensemble_member.mojo `meta_plan_refuse`), in a process with no Python.

A META admission that meets a constructor value its door refuses (bindings/trees_meta_validate_unpack.mojo, the
check the door and the admission both call) mints a plan entry carrying the door's words; the META member's FIT
raises exactly those words, where the door raises them. Each case mints such an entry by hand with the door's
words (literal here, copied from the one door function named) and checks the member against them:

1. VotingClassifier, `meta_voting_validate`'s "voting must be 'hard' or 'soft'", refused first: a plan with no
   members; NEW succeeds, CAPABILITIES answers FIT, PREDICT_PROBA and the classifier bit, FIT raises the words;
2. StackingRegressor, `meta_stacking_validate`'s stack_method words: CAPABILITIES has no classifier bit, FIT
   raises the words;
3. CalibratedClassifierCV, `meta_calibrated_validate`'s method words: FIT raises them;
4. BaggingClassifier, `boost_bagging_validate`'s "Out of bag estimation only available if bootstrap=True": FIT
   raises them;
5. AdaBoostRegressor, `boost_check_learning_rate`'s "learning_rate must be > 0": FIT raises them;
6. VotingClassifier, `meta_voting_weights`' count words raised LATE (after the nested fits): a soft vote over two
   LogisticRegression members whose entry carries the words with `late` True raises them from FIT; the same
   entry without words fits (the control: the members and the vote are whole).

The member-vs-door equality over real estimators (each refusal's words from the member path and from the door)
is tools/mf_del5_digests.py section refuse5.

FAIL-PROOF EDITS (each must make the named case fail): drop the `_refused_first` raise at the top of `_fit`
(cases 1-5: FIT reaches the kind's body with no members and raises other words); drop `mask |=
MEMBER_CAN_CLASSIFIER` in `_refused_mask` (case 1); drop the `refusal_late` raise after the kind's fit in `_fit`
(case 6: FIT succeeds).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef
from core.ensemble_member import (
    ensemble_member_v2, meta_plan_mint, meta_plan_mint_with, meta_plan_release, meta_plan_refuse,
    META_VOTE_CLASSIFIER, META_STACK_REGRESSOR, META_CALIBRATED, META_BAG_CLASSIFIER, META_ADA_REGRESSOR,
    META_PARAM_COUNT,
)
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_new, member_free,
    member_kind_capabilities, MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_I32, MEMBER_F32,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_CLASSIFIER,
)
from core.member_families import MEMBER_FAMILY_ENSEMBLE
from glm.host.logistic_native_host import qn_host_native_v1
from xtrees.glue_runs import publish_x_trees_glue

comptime N = 30
comptime D = 3
comptime K = 3
comptime CASES = 6


def _x() -> List[Float32]:
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        for j in range(D):  # small-loop(fixture)
            x[i * D + j] = Float32((i * 7 + j * 5) % 11) * 0.25 - 1.0 + Float32(i % 3) * 0.5 * Float32(j + 1)
    return x^


def _codes() -> List[Int32]:
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: every class present)
        y[i] = Int32((i + i // 5) % K)
    return y^


def _logistic(l2: Float64) raises -> MemberPlan:
    """A LogisticRegression member plan (intercept, l2 `l2`, lbfgs) on the CPU QN entry."""
    var plan = MemberPlan()
    plan.route = logistic_route_pack(LogisticRoute(DEVICE_CPU, Int(GLOBAL_NUMERIC_MODE), 0, False, False,
                                                   QnNativeDriver(qn_host_native_v1)))
    plan.member = MemberRef(EntryRef(logistic_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    # LogisticRegression()'s own words as providers/python/member_logistic_boundary_door.mojo admits them:
    # max_iter 1000, linesearch 50, lbfgs_memory 5, penalty_normalized 1 (was 100, 20, 10, 0: the old words,
    # an unnormalized penalty N times the estimator's; lane mf-iterimp-logit, w3-cleanup).
    plan.params = [0.0, 1.0, 0.0, l2, 1000.0, 50.0, 5.0, 1.0, 1e-4, 1e-6]
    plan.classes = K
    return plan^


def _meta_plan(kind: Int, handle: Int, flag: Float64) raises -> MemberPlan:
    """The META plan over a minted entry (as the admission's `_finish` writes it)."""
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    plan.route = route
    plan.member = MemberRef(EntryRef(ensemble_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.family = MEMBER_FAMILY_ENSEMBLE
    plan.params = [Float64(kind), Float64(handle), flag, 0.0, 0.0, 0.0]
    require(len(plan.params) == META_PARAM_COUNT, "META parameter count")
    plan.classes = K
    return plan^


def _fit_words(plan: MemberPlan, classifier: Bool) raises -> String:
    """FIT over the fixture: "" when it fits (the state freed), else the words it raised."""
    var x = _x()
    var codes = _codes()
    var targets = List[Float32](length=N, fill=0.0)
    for i in range(N):  # small-loop(fixture: a regressor's float32 targets)
        targets[i] = Float32(codes[i])
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    if classifier:
        call.y = MemberMatrix(Int(codes.unsafe_ptr()), N, 1, MEMBER_I32)
        call.classes = K
    else:
        call.y = MemberMatrix(Int(targets.unsafe_ptr()), N, 1, MEMBER_F32)
    var words = String("")
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    except e:
        words = String(e)
    member_free(plan, call)
    _ = len(x)  # keep the fixture alive past the fit (AGENTS.md part 6)
    _ = len(codes)
    _ = len(targets)
    return words^


def _refused_first(kind: Int, words: String, classifier: Bool) raises:
    """A plan entry with no members carrying the door's `words` (`_refuse_first`): FIT raises them exactly."""
    var handle = meta_plan_mint_with(List[MemberPlan](), List[Float64](), List[Int](), List[Int](), List[Float64](),
                                     List[List[Int32]](), List[List[Int32]]())
    meta_plan_refuse(handle, words.copy(), False)
    var plan = _meta_plan(kind, handle, 0.0)
    var mask = member_kind_capabilities(plan)
    require((mask & MEMBER_CAN_FIT) != 0, "a refused plan admits a fit")
    require(((mask & MEMBER_CAN_CLASSIFIER) != 0) == classifier, "a refused plan keeps its kind's classifier bit")
    if classifier:
        require((mask & MEMBER_CAN_PREDICT_PROBA) != 0, "a refused classifier plan admits predict_proba")
    var got = _fit_words(plan, classifier)
    require(got == words, "FIT raised '" + got + "', the door's words are '" + words + "'")
    # A second state of the same entry raises them too (the words are the entry's, copied by NEW).
    require(_fit_words(plan, classifier) == words, "a second state raises the door's words")
    meta_plan_release(handle)


def check_voting_first() raises:
    _refused_first(META_VOTE_CLASSIFIER, String("voting must be 'hard' or 'soft'"), True)


def check_stacking_first() raises:
    _refused_first(META_STACK_REGRESSOR,
                   String("stack_method must be auto, predict_proba, decision_function or predict"), False)


def check_calibrated_first() raises:
    _refused_first(META_CALIBRATED, String("method must be 'sigmoid' or 'isotonic'"), True)


def check_bagging_first() raises:
    _refused_first(META_BAG_CLASSIFIER, String("Out of bag estimation only available if bootstrap=True"), True)


def check_ada_first() raises:
    _refused_first(META_ADA_REGRESSOR, String("learning_rate must be > 0"), False)


def check_voting_late() raises:
    var words = String("weights has 3 entries, there are 2 estimators")
    var nested: List[MemberPlan] = [_logistic(1.0), _logistic(0.5)]
    var masks = List[Int]()
    for u in range(len(nested)):  # small-loop(fixture: each nested kind's mask, as the admission asks it)
        masks.append(member_kind_capabilities(nested[u]))
    var control_nested: List[MemberPlan] = [_logistic(1.0), _logistic(0.5)]
    var control_masks = masks.copy()
    var handle = meta_plan_mint(nested^, [1.0, 1.0], List[Int](), masks^)
    meta_plan_refuse(handle, words.copy(), True)
    var got = _fit_words(_meta_plan(META_VOTE_CLASSIFIER, handle, 1.0), True)
    require(got == words, "a late refusal: FIT raised '" + got + "', the door's words are '" + words + "'")
    meta_plan_release(handle)
    var control = meta_plan_mint(control_nested^, [1.0, 1.0], List[Int](), control_masks^)
    var fitted = _fit_words(_meta_plan(META_VOTE_CLASSIFIER, control, 1.0), True)
    require(fitted == "", "the same entry without words fits, got '" + fitted + "'")
    meta_plan_release(control)


def test_mf_refuse5_door_words() raises:
    publish_x_trees_glue("cpu")
    def run_case(which: Int) raises:
        if which == 0:
            check_voting_first()
        elif which == 1:
            check_stacking_first()
        elif which == 2:
            check_calibrated_first()
        elif which == 3:
            check_bagging_first()
        elif which == 4:
            check_ada_first()
        else:
            check_voting_late()
    check_cases("mf_refuse5_door_words", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
