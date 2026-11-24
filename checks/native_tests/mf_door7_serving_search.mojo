# SPDX-License-Identifier: Apache-2.0
"""THE SERVING SEARCH PLAN (lane mf-door7, 2026-11-23): a fitted GridSearchCV held for predict only
(providers/python/member_ensemble_boundary_door.mojo `_admit_search_serving`: one nested plan, the fitted best's;
the scoring=None head with cv=None's 5 folds; META_PARAM_SERVING set) and its import (`_search_state`: the best
imported as candidate 0). Before, a search whose scorer, splitter, candidates or device the FIT admission did not
take answered False and SHAP evaluated its parent through the model's door (`_DoorModel`, deleted).

Cases (LogisticRegression best, C = 0.1, fitted directly on the fixture; N = 60 rows, D = 3, K = 3 classes):
0. NEW of the serving plan, IMPORT (META_BLOCK_NESTED: the fitted best; META_BLOCK_SEARCH: index 0; DONE):
   predict_proba == the best's own predict_proba, word for word (the kind forwards to its best).
1. The same state's predict (codes) == the best's own predict.
2. The serving plan's capability mask (`member_kind_capabilities`) has PREDICT, PREDICT_PROBA and CLASSIFIER
   (what a SHAP or meta caller asks of it), as the best's mask has them.
3. The import with the best's ORIGINAL index (1, as `best_index_` reads for the second of three candidates)
   is refused ("best candidate is not one of its candidates"): why `_search_state` writes 0 for a serving plan.

Fail-proof edits (each makes the named case fail):
- core/ensemble_member.mojo, the import of META_BLOCK_SEARCH: leave `slot[].plans` empty instead of
  `[slot[].base_plans[best]]` (cases 0, 1: the forwarded best has no plan).
- drop `msel_scorer_group_words` from `_serving_words` (cases 0-2: NEW refuses "a search member needs ...").
- write folds 0 in `_serving_plan` (as the first draft of `_admit_search_serving` did; cases 0-2: NEW refuses
  "cv >= 2").
- in case 3 write index 0 (case 3: the import is accepted).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef
from core.ensemble_member import (
    ensemble_member_v2, meta_plan_mint_with, meta_plan_release, META_SEARCH_CLASSIFIER, META_BLOCK_NESTED,
    META_BLOCK_SEARCH, META_PARAM_COUNT, META_PARAM_SERVING,
)
from core.msel_scorer_run import MselScorerOptions, msel_scorer_group_words
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    member_kind_capabilities, MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA,
    MEMBER_OP_IMPORT, MEMBER_OP_NEW, MEMBER_IMPORT_DONE, MEMBER_F64, MEMBER_I32, MEMBER_I64,
    MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_CLASSIFIER,
)
from core.member_families import MEMBER_FAMILY_ENSEMBLE
from glm.host.logistic_native_host import qn_host_native_v1
from xtrees.glue_runs import publish_x_trees_glue

comptime N = 60
comptime D = 3
comptime K = 3
comptime CASES = 4


def _x() -> List[Float32]:
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        for j in range(D):  # small-loop(fixture)
            x[i * D + j] = Float32((i * 7 + j * 5) % 11) * 0.25 - 1.0 + Float32(i % 3) * 0.5 * Float32(j + 1)
    return x^


def _codes() -> List[Int32]:
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: every class has N / 3 rows)
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


def _free(plan: MemberPlan, state: Int):
    var call = MemberCall(plan.route, state)
    member_free(plan, call)


def _fit(plan: MemberPlan, x: List[Float32], y: List[Int32]) raises -> Int:
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(y.unsafe_ptr()), N, 1, MEMBER_I32)
    call.classes = K
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    except e:
        member_free(plan, call)
        raise e^
    return call.state


def _serving_words() -> List[Float64]:
    """`_admit_search_serving`'s settings words: the scoring=None head (a grid, refit True, count 0) and its one
    scorer group."""
    var words: List[Float64] = [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]
    msel_scorer_group_words(words, 0, -1, MselScorerOptions(), List[Int32]())
    return words^


struct _Serving(Movable):
    """A serving search plan and the plan entry it names (released by `close`, as the admission's capsule)."""
    var plan: MemberPlan
    var handle: Int

    def __init__(out self, var plan: MemberPlan, handle: Int):
        self.plan = plan^
        self.handle = handle

    def close(mut self):
        meta_plan_release(self.handle)
        self.handle = 0


def _serving_plan(best: MemberPlan) raises -> _Serving:
    """`_admit_search_serving` + `_finish`: one nested plan (the best's), its mask, cv=None's 5 folds, the
    serving word."""
    var nested: List[MemberPlan] = [best.copy()]
    var masks: List[Int] = [member_kind_capabilities(best)]
    var handle = meta_plan_mint_with(nested^, List[Float64](), List[Int](), masks^, _serving_words(),
                                     List[List[Int32]](), List[List[Int32]]())
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    plan.route = route
    plan.member = MemberRef(EntryRef(ensemble_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.family = MEMBER_FAMILY_ENSEMBLE
    plan.params = [Float64(META_SEARCH_CLASSIFIER), Float64(handle), 0.0, 5.0, 0.0, 0.0]
    plan.params[META_PARAM_SERVING] = 1.0
    require(len(plan.params) == META_PARAM_COUNT, "META parameter count")
    plan.classes = K
    return _Serving(plan^, handle)


def _imported(serving: _Serving, best_state: Int, index: Int) raises -> Int:
    """`_search_state`'s calls: NEW, the fitted best as the one nested state, the head words at `index`, DONE.
    The best's state belongs to the new state once the nested block is in."""
    var call = member_call_of(serving.plan)
    _ = member_call(serving.plan.member, MEMBER_OP_NEW, call)
    var handles: List[Int64] = [Int64(best_state)]
    try:
        var block = MemberCall(serving.plan.route, call.state)
        block.block = META_BLOCK_NESTED
        block.x = MemberMatrix(Int(handles.unsafe_ptr()), 1, 1, MEMBER_I64)
        _ = member_call(serving.plan.member, MEMBER_OP_IMPORT, block)
        _ = len(handles)  # read by the call above
        var words: List[Float64] = [Float64(index), 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0]
        var search = MemberCall(serving.plan.route, call.state)
        search.block = META_BLOCK_SEARCH
        search.x = MemberMatrix(Int(words.unsafe_ptr()), 1, len(words), MEMBER_F64)
        _ = member_call(serving.plan.member, MEMBER_OP_IMPORT, search)
        _ = len(words)  # copied in by the call above
        var last = MemberCall(serving.plan.route, call.state)
        last.flags = MEMBER_IMPORT_DONE
        last.classes = K
        _ = member_call(serving.plan.member, MEMBER_OP_IMPORT, last)
    except e:
        member_free(serving.plan, call)
        raise e^
    return call.state


def _proba(plan: MemberPlan, state: Int, x: List[Float32]) raises -> List[Float64]:
    var got = List[Float64](length=N * K, fill=-7.0)
    var call = member_call_of(plan, state)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.dst = MemberMatrix(Int(got.unsafe_ptr()), N, K, MEMBER_F64)
    _ = member_call(plan.member, MEMBER_OP_PREDICT_PROBA, call)
    _ = len(x)  # read by the call above
    return got^


def _labels(plan: MemberPlan, state: Int, x: List[Float32]) raises -> List[Int64]:
    var got = List[Int64](length=N, fill=-1)
    var call = member_call_of(plan, state)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.dst = MemberMatrix(Int(got.unsafe_ptr()), N, 1, MEMBER_I64)
    _ = member_call(plan.member, MEMBER_OP_PREDICT, call)
    _ = len(x)  # read by the call above
    return got^


def check_case(which: Int) raises:
    var x = _x()
    var y = _codes()
    var best = _logistic(0.1)
    var serving = _serving_plan(best)
    if which == 2:
        var mask = member_kind_capabilities(serving.plan)
        var need = MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_CLASSIFIER
        require((mask & need) == need, "the serving search's mask has predict, predict_proba and classifier")
        serving.close()
        return
    var fitted = _fit(best, x, y)
    if which == 3:
        # The best's own index among three candidates: the serving plan holds one.
        var refused = False
        try:
            var state = _imported(serving, fitted, 1)
            _free(serving.plan, state)
        except e:
            refused = "not one of its candidates" in String(e)
        require(refused, "an import at index 1 of a one-candidate serving plan is refused")
        serving.close()
        _ = y^
        _ = x^
        return
    # Its own outputs before the import takes the state.
    var want_p = _proba(best, fitted, x)
    var want_c = _labels(best, fitted, x)
    var state = _imported(serving, fitted, 0)
    if which == 0:
        var got = _proba(serving.plan, state, x)
        for i in range(N * K):  # small-loop(check: every word)
            require(got[i] == want_p[i], "word " + String(i) + ": served " + String(got[i]) + " != best "
                    + String(want_p[i]))
    else:
        var got = _labels(serving.plan, state, x)
        for i in range(N):  # small-loop(check: every code)
            require(got[i] == want_c[i], "code " + String(i) + ": served " + String(got[i]) + " != best "
                    + String(want_c[i]))
    _free(serving.plan, state)
    serving.close()
    _ = y^
    _ = x^


def test_mf_door7_serving_search() raises:
    # The publication _mojolearn_x_trees_host makes at module init (the META entry's glue), once.
    publish_x_trees_glue("cpu")
    def run_case(which: Int) raises:
        check_case(which)
    check_cases("mf_door7_serving_search", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
