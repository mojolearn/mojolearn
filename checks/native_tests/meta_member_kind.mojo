# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE META MEMBER KIND (core/ensemble_member.mojo; lane mf-meta3, 2026-11-23), in a process with no Python.

Every case builds META plans by hand (the plan store `meta_plan_mint`, the entry `ensemble_member_v2`, the CPU
x_trees glue the host image publishes) over LogisticRegression members (core/logistic_member.mojo, kind 0, on the
CPU QN entry: MEMBER_GIL_FREE), and checks the META member against the DOOR'S WORDS, i.e. the same typed RUNs of
xtrees/trees_meta_vote_run.mojo the door runs, over the members' own answers:

1. soft VotingClassifier, k = 3: META fit + predict_proba / predict == `vote_soft_run` / `vote_codes_run` over
   two members fitted directly on the same rows and codes (door blocks: their own answers, float64 rounded to
   float32 as the door's admission rounds them);
2. hard VotingClassifier, k = 3: META predict == `vote_codes_run` over the members' own codes;
3. StackingClassifier, k = 3, cv = 3, predict_proba stacking: META predict_proba == the final member's own
   predict_proba over the meta features `stack_place_run` builds from the META's exported full members
   (META_BLOCK_NESTED), rounded to float32 (the door's `_stacking_meta32` + final call); widths 3 + 3 exported;
4. NESTED IN A MEMBER SET, parallel == serial: four soft-vote META units fitted as ONE member set
   (core/member_set_run.mojo `member_set_fit`; the META kind inherits MEMBER_GIL_FREE, so the units run on host
   workers and each runs its own nested set inside) and again with the worker bound 1: every unit's
   predict_proba equals case 1's words;
5. OneVsRestClassifier (lane mf-meta3b, unit kind META_OVR), k = 3: META predict / predict_proba ==
   `ovr_assemble_run` over three members fitted directly on indicators computed here (their decision columns for
   predict, their code-1 probability columns for predict_proba);
6. OneVsRest META units nested in a member set, parallel == serial (as case 4);
7. CalibratedClassifierCV (META_CALIBRATED), cv = 3, sigmoid, ensemble=True, decision base: META predict_proba ==
   `calibrated_fit_run` + `calibrated_proba_run` run here over the same stratified folds (the door's RUN words),
   predict == their row argmax;
8. StackingClassifier with stack_method='predict': META predict_proba == the final member over the exported
   members' codes as float32 meta features (widths 1 + 1);
9. StackingClassifier over a user's index list: it fits (widths 3 + 3), and a list whose test sides are not a
   partition raises the door's `_splits` words exactly;
10. NESTED TWO DEEP (lane mf-meta3c): a soft vote over [case 1's soft vote META, a LogisticRegression], weights
   1.0 and 1.0: META predict_proba / predict == `vote_soft_run` / `vote_codes_run` over the inner META's own
   predict_proba (fitted directly, float64 rounded to float32 as the door's admission) and the logistic member's;
   the inner META runs as one more member of the outer one's set.
11. THE ADABOOST KIND (lane mf-ada4), AdaBoostClassifier over a DecisionTreeClassifier(max_depth=1) base, k = 3,
   5 rounds, seed 3: META fit + estimator_weights_ / estimator_errors_ (META_BLOCK_ADA), predict, predict_proba and
   decision_function == the door's words: checks/native_tests/mf_ada_fits.mojo `ada_fit` (the door's
   `_ada_native` plan over `rf_config`, the host session when the image serves one) and `ada_classifier_run` over
   the members' own host codes (ADA_CODES / ADA_PROBA / ADA_DECISION);
12. AdaBoostRegressor over DecisionTreeRegressor(max_depth=3), loss 'square' (code 1), 5 rounds: weights, errors and
   predict (float32) == `ada_fit` + `ada_predict_values`;
13. AdaBoostClassifier over a RandomForestClassifier(n_estimators=3, max_depth=2) base (the MEMBER ROUNDS): weights,
   errors and predict of the training X == `ada_member_fit` (the door's `_ada_members` plan);
14. AdaBoost IMPORT: case 11's fitted rounds copied block by block into new tree member states and imported with
   its weights into a NEW META state: predict_proba equal to the fitted state's.
15. THE SEARCH KIND (lane mf-search4): GridSearchCV over three LogisticRegression candidates, cv = 3: the META's
   META_BLOCK_SEARCH words == `msel_search_run`'s tables, predict_proba == the best's refit state's;
16. THE MULTI-OUTPUT KIND over THE 2-D y: MultiOutputClassifier, outputs of 3 and 2 classes: predict == each
   output's member fitted directly; score == the counted share of rows with every output right.
FAIL-PROOF EDITS, lane mf-search4: `slot[].plans = [slot[].base_plans[0].copy()]` in `_fit_search` (case 15: the
wrong candidate's plan serves the best's state, or the header's best differs); `dst[i] = src[i * outputs]` in
`_fit_multi`'s `_column` (case 16: the second output fits the first column); `hit += 1.0` without the
`all_equal` test in `_score_multi` (case 16).
The regressor kinds (VotingRegressor `vote_regress_run`, StackingRegressor, BaggingRegressor) and the Bagging kinds
(their base is a tree, forest or extra-trees member; this process publishes only the logistic entry) run through
tools/mf_meta3_digests.py (this file's members are classifiers).

FAIL-PROOF EDITS, lane mf-meta3b: `at.append(1 if decision else 0)` in `_predict_ovr` (case 5: predict reads the
wrong column); `_add(slot, run, u, k, ...)` for the units in `_fit_ovr` (cases 5, 6: two-class units told k
classes); `slot[].pair_first.append(res.first[f] + 1)` in `_fit_cal` (case 7: the wrong calibrators); drop the
`held[c]` mapping and write `Float32(c + 1)` in `_label_block` (case 8); skip the partition loop in `_user_folds`
(case 9: no refusal).

FAIL-PROOF EDIT, lane mf-meta3c: write `slot[].proba_dtype[u] = MEMBER_F32` in `_nested_outputs` (case 10: the
inner META's float64 probabilities are read as float32 words).

FAIL-PROOF EDITS, lane mf-ada4: write `finish = ADA_DECISION` for MEMBER_OP_PREDICT_PROBA in `_predict_ada`
(case 11: predict_proba is the margin); drop `plan.loss = Int(words[META_ADA_LOSS])` in `_fit_ada` (case 12: the
linear loss's weights); drop `plan.member_route = True` in `_fit_ada` (case 13: the RF base fits as session rounds
and refuses); read `words[2 + na + i]` into `ada_alphas` in `_import_ada` (case 14: the errors weight the vote).

FAIL-PROOF EDITS (each must make a case fail): drop `_zero_dst` in `_predict_vote` (cases 1, 4: the sum starts
from stale words); use `run.states[v + 1]` in `_keep_states` (cases 1-4: a member's state is another's, or 0);
skip the passthrough branch flag test by forcing `slot[].flag = True` in `_new` for a stack (case 3: widths and
final input differ); change `vote_weight_sum` to ignore one weight (case 1, weights 1.0 and 2.0).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef
from core.ensemble_member import (
    ensemble_member_v2, meta_plan_mint, meta_plan_mint_with, meta_plan_release, META_VOTE_CLASSIFIER,
    META_STACK_CLASSIFIER, META_OVR, META_CALIBRATED, META_PARAM_COUNT, META_BLOCK_NESTED, META_BLOCK_WIDTHS,
    META_ADA_CLASSIFIER, META_ADA_REGRESSOR, META_BLOCK_ADA, META_SEARCH_CLASSIFIER, META_MULTI_CLASSIFIER,
    META_BLOCK_SEARCH, META_SEARCH_WORDS, META_SEARCH_HEAD,
)
from bindings.arena_prog_run import MetricRoute
from core.entry_table import entry_ref_unresolved
from core.member_abi import MEMBER_OP_SCORE
from core.msel_search_run import MselSearchPlan, msel_search_run
from core.entry_table import entry_ref_gil_free
from core.forest_calls import xf_ref, xf_caps
from core.forest_entry_table import ForestEntryKey, FOREST_TABLE_RF
from core.forest_fit_request import XF_CAP_SESSION
from core.member_abi import MEMBER_OP_IMPORT, MEMBER_IMPORT_DONE, MEMBER_BLOCK_BY_INDEX, MEMBER_OP_NEW
from core.member_families import MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST
from checks.native_tests.bm_rest_boost_fits import publish_host_images, trees_key
from checks.native_tests.mf_ada_fits import (
    AdaSpec, AdaModel, ada_fit, ada_predict_codes, ada_predict_values, publish_host_members, member_plan_of,
    ada_member_fit,
)
from checks.native_tests.py2mojo_forest_fits import RfSpec, rf_config, predict, argmax_rows, RF_SQUARED
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    member_kind_capabilities, MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_OUTPUTS, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_EXPORT, MEMBER_OP_DECISION, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64,
    MEMBER_GIL_FREE, MEMBER_CAN_PREDICT_PROBA, MEMBER_BORROWS_FIT,
)
from core.member_families import MEMBER_FAMILY_ENSEMBLE
from core.member_set_run import MemberSetRun, member_set_fit, member_set_run_release, MEMBER_SET_WORKER
from core.trees_entry_table import TreesEntryKey
from glm.host.logistic_native_host import qn_host_native_v1
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import xt_ref, xt_device_folds
from xtrees.trees_calibrated_fit_run import CalFitPlan, CalFitOut, calibrated_fit_run, CAL_ALL_ANSWERED
from xtrees.trees_meta_vote_run import (
    VoteMember, VoteRows, VOTE_ALL_ANSWERED, vote_soft_run, vote_codes_run, stack_place_run, ovr_assemble_run,
    calibrated_proba_run, CalPair, CalFit, STACK_PROBA, STACK_PREDICT, ada_classifier_run, ADA_PROBA, ADA_DECISION,
)

comptime N = 60
comptime D = 3
comptime K = 3
comptime CASES = 16
comptime UNITS = 4

comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


def _glue() raises -> EntryRef:
    return xt_ref(TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


def _x() -> List[Float32]:
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        for j in range(D):  # small-loop(fixture)
            x[i * D + j] = Float32((i * 7 + j * 5) % 11) * 0.25 - 1.0 + Float32(i % 3) * 0.5 * Float32(j + 1)
    return x^


def _codes() -> List[Int32]:
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: every class has N / 3 rows, so 3 stratified folds hold every class)
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
    plan.params = [0.0, 1.0, 0.0, l2, 100.0, 20.0, 10.0, 0.0, 1e-4, 1e-6]
    plan.classes = K
    return plan^


struct _Meta(Movable):
    """A META plan and the plan entry it names (released by `close`, as the admission's capsule does)."""
    var plan: MemberPlan
    var handle: Int

    def __init__(out self, var plan: MemberPlan, handle: Int):
        self.plan = plan^
        self.handle = handle

    def close(mut self):
        meta_plan_release(self.handle)
        self.handle = 0


def _meta(kind: Int, flag: Float64, folds: Int, var nested: List[MemberPlan], var weights: List[Float64],
          var stack_kinds: List[Int], cap: Int) raises -> _Meta:
    var masks = List[Int]()
    for u in range(len(nested)):  # small-loop(fixture: each nested kind's mask, as the admission asks it)
        masks.append(member_kind_capabilities(nested[u]))
    var handle = meta_plan_mint(nested^, weights^, stack_kinds^, masks^)
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    plan.route = route
    plan.member = MemberRef(EntryRef(ensemble_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.family = MEMBER_FAMILY_ENSEMBLE
    plan.params = [Float64(kind), Float64(handle), flag, Float64(folds), Float64(cap), 0.0]
    require(len(plan.params) == META_PARAM_COUNT, "META parameter count")
    plan.classes = K
    return _Meta(plan^, handle)


def _fit_member(plan: MemberPlan, x: List[Float32], y: List[Int32]) raises -> Int:
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


def _free(plan: MemberPlan, state: Int):
    var call = MemberCall(plan.route, state)
    member_free(plan, call)


def _proba32(plan: MemberPlan, state: Int, x: Int, n: Int, d: Int) raises -> List[Float32]:
    """A member's predict_proba (n x K), float64 rounded word by word to float32 (the door's admission)."""
    var call = member_call_of(plan, state)
    call.flags = MEMBER_OP_PREDICT_PROBA
    _ = member_call(plan.member, MEMBER_OP_OUTPUTS, call)
    var words = List[Float32](length=n * K, fill=0.0)
    call.x = MemberMatrix(x, n, d)
    if call.out_dtype == MEMBER_F64:
        var wide = List[Float64](length=n * K, fill=0.0)
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), n, K, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_PREDICT_PROBA, call)
        for i in range(n * K):  # small-loop(oracle: the door's float32 admission)
            words[i] = wide[i].cast[DType.float32]()
        _ = wide^
    else:
        call.dst = MemberMatrix(Int(words.unsafe_ptr()), n, K, MEMBER_F32)
        _ = member_call(plan.member, MEMBER_OP_PREDICT_PROBA, call)
    return words^


def _codes32(plan: MemberPlan, state: Int, x: Int) raises -> List[Int32]:
    var wide = List[Int64](length=N, fill=0)
    var call = member_call_of(plan, state)
    call.x = MemberMatrix(x, N, D)
    call.dst = MemberMatrix(Int(wide.unsafe_ptr()), N, 1, MEMBER_I64)
    _ = member_call(plan.member, MEMBER_OP_PREDICT, call)
    var codes = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(oracle: the door's int32 cast)
        codes[i] = Int32(wide[i])
    _ = wide^
    return codes^


def _meta_fit(meta: _Meta, x: List[Float32], y: List[Int32]) raises -> Int:
    return _fit_member(meta.plan, x, y)


def _meta_proba(meta: _Meta, state: Int, x: List[Float32]) raises -> List[Float64]:
    var got = List[Float64](length=N * K, fill=-7.0)
    var call = member_call_of(meta.plan, state)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.dst = MemberMatrix(Int(got.unsafe_ptr()), N, K, MEMBER_F64)
    _ = member_call(meta.plan.member, MEMBER_OP_PREDICT_PROBA, call)
    return got^


def _meta_codes(meta: _Meta, state: Int, x: List[Float32]) raises -> List[Int32]:
    var got = List[Int32](length=N, fill=Int32(-1))
    var call = member_call_of(meta.plan, state)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.dst = MemberMatrix(Int(got.unsafe_ptr()), N, 1, MEMBER_I32)
    _ = member_call(meta.plan.member, MEMBER_OP_PREDICT, call)
    return got^


def _same(a: List[Float64], b: List[Float64], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(oracle: every word)
        require(a[i] == b[i], what + " word " + String(i) + ": META " + String(a[i]) + " != door " + String(b[i]))


def _same_i32(a: List[Int32], b: List[Int32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(oracle: every code)
        require(a[i] == b[i], what + " code " + String(i) + ": META " + String(a[i]) + " != door " + String(b[i]))


def _iota() -> List[Int32]:
    var cols = List[Int32](length=K, fill=Int32(0))
    for c in range(K):  # small-loop(fixture)
        cols[c] = Int32(c)
    return cols^


def _vote_door(soft: Bool, x: List[Float32], y: List[Int32], mut proba: List[Float64],
               mut codes: List[Int32]) raises:
    """The door's words of a two-member vote (weights 1.0, 2.0): the members fitted directly, their own blocks
    folded by the vote RUNs."""
    var a = _logistic(1.0)
    var b = _logistic(0.25)
    var sa = _fit_member(a, x, y)
    var sb = _fit_member(b, x, y)
    var cols = _iota()
    var rows = VoteRows(Int(x.unsafe_ptr()), N, D)
    var acc = List[Float64](length=N * K, fill=0.0)
    if soft:
        var pa = _proba32(a, sa, Int(x.unsafe_ptr()), N, D)
        var pb = _proba32(b, sb, Int(x.unsafe_ptr()), N, D)
        var door: List[VoteMember] = [VoteMember(Int(pa.unsafe_ptr()), K, Int(cols.unsafe_ptr()), 1.0),
                                      VoteMember(Int(pb.unsafe_ptr()), K, Int(cols.unsafe_ptr()), 2.0)]
        _ = vote_soft_run(_glue(), rows, door, K, Int(proba.unsafe_ptr()))
        var acc2 = List[Float64](length=N * K, fill=0.0)
        _ = vote_codes_run(_glue(), rows, door, K, True, Int(acc2.unsafe_ptr()), Int(codes.unsafe_ptr()))
        _ = acc2^
        _ = pa^
        _ = pb^
    else:
        var ca = _codes32(a, sa, Int(x.unsafe_ptr()))
        var cb = _codes32(b, sb, Int(x.unsafe_ptr()))
        var door: List[VoteMember] = [VoteMember(Int(ca.unsafe_ptr()), 1, 0, 1.0),
                                      VoteMember(Int(cb.unsafe_ptr()), 1, 0, 2.0)]
        _ = vote_codes_run(_glue(), rows, door, K, False, Int(acc.unsafe_ptr()), Int(codes.unsafe_ptr()))
        _ = ca^
        _ = cb^
    _ = acc^
    _ = cols^
    _free(a, sa)
    _free(b, sb)


def _vote_meta(soft: Bool, cap: Int) raises -> _Meta:
    var nested: List[MemberPlan] = [_logistic(1.0), _logistic(0.25)]
    var weights: List[Float64] = [1.0, 2.0]
    return _meta(META_VOTE_CLASSIFIER, 1.0 if soft else 0.0, 0, nested^, weights^, List[Int](), cap)


def check_soft_vote() raises:
    var x = _x()
    var y = _codes()
    var want = List[Float64](length=N * K, fill=0.0)
    var want_codes = List[Int32](length=N, fill=Int32(0))
    _vote_door(True, x, y, want, want_codes)
    var meta = _vote_meta(True, 0)
    require((member_kind_capabilities(meta.plan) & MEMBER_GIL_FREE) != 0, "META kind lost MEMBER_GIL_FREE")
    require((member_kind_capabilities(meta.plan) & MEMBER_CAN_PREDICT_PROBA) != 0, "soft vote has no predict_proba")
    var state = _meta_fit(meta, x, y)
    _same(_meta_proba(meta, state, x), want, "soft vote predict_proba")
    _same_i32(_meta_codes(meta, state, x), want_codes, "soft vote predict")
    _free(meta.plan, state)
    meta.close()
    _ = x^
    _ = y^


def check_hard_vote() raises:
    var x = _x()
    var y = _codes()
    var unused = List[Float64](length=N * K, fill=0.0)
    var want_codes = List[Int32](length=N, fill=Int32(0))
    _vote_door(False, x, y, unused, want_codes)
    var meta = _vote_meta(False, 0)
    require((member_kind_capabilities(meta.plan) & MEMBER_CAN_PREDICT_PROBA) == 0, "hard vote has predict_proba")
    var state = _meta_fit(meta, x, y)
    _same_i32(_meta_codes(meta, state, x), want_codes, "hard vote predict")
    _free(meta.plan, state)
    meta.close()
    _ = x^
    _ = y^


def check_stack() raises:
    var x = _x()
    var y = _codes()
    var nested: List[MemberPlan] = [_logistic(1.0), _logistic(0.25), _logistic(1.0)]
    var kinds: List[Int] = [STACK_PROBA, STACK_PROBA]
    var meta = _meta(META_STACK_CLASSIFIER, 0.0, 3, nested^, List[Float64](), kinds^, 0)
    var state = _meta_fit(meta, x, y)
    var got = _meta_proba(meta, state, x)
    # The door's words over the META's own fitted members (export: they are the state's).
    var call = MemberCall(meta.plan.route, state)
    call.block = META_BLOCK_NESTED
    _ = member_call(meta.plan.member, MEMBER_OP_EXPORT, call)
    require(call.dst.cols == 3 and call.dst.dtype == MEMBER_I64, "nested block: three int64 handles")
    var handles = _I64P(unsafe_from_address=call.dst.data)
    var s0 = Int(handles[0])
    var s1 = Int(handles[1])
    var s2 = Int(handles[2])
    var w = MemberCall(meta.plan.route, state)
    w.block = META_BLOCK_WIDTHS
    _ = member_call(meta.plan.member, MEMBER_OP_EXPORT, w)
    var widths = _I64P(unsafe_from_address=w.dst.data)
    require(w.dst.cols == 2 and widths[0] == 3 and widths[1] == 3, "stacking widths 3 + 3")
    var a = _logistic(1.0)
    var b = _logistic(0.25)
    var f = _logistic(1.0)
    var pa = _proba32(a, s0, Int(x.unsafe_ptr()), N, D)
    var pb = _proba32(b, s1, Int(x.unsafe_ptr()), N, D)
    var cols = _iota()
    var door: List[VoteMember] = [VoteMember(Int(pa.unsafe_ptr()), K, Int(cols.unsafe_ptr()), 1.0),
                                  VoteMember(Int(pb.unsafe_ptr()), K, Int(cols.unsafe_ptr()), 1.0)]
    var place = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: the row numbers)
        place[i] = Int32(i)
    var meta64 = List[Float64](length=N * 6, fill=0.0)
    var counts: List[Int] = [K, K]
    var stack_kinds: List[Int] = [STACK_PROBA, STACK_PROBA]
    var placed = List[Int]()
    require(stack_place_run(_glue(), VoteRows(Int(x.unsafe_ptr()), N, D), door, stack_kinds, counts, K, False,
                            Int(meta64.unsafe_ptr()), N, 6, Int(place.unsafe_ptr()), 0, placed, 0)
            == VOTE_ALL_ANSWERED, "a door block refused")
    var meta32 = List[Float32](length=N * 6, fill=0.0)
    for i in range(N * 6):  # small-loop(oracle: the door's float32 admission of the meta features)
        meta32[i] = meta64[i].cast[DType.float32]()
    var want32 = List[Float64](length=N * K, fill=0.0)
    var fc = member_call_of(f, s2)
    fc.flags = MEMBER_OP_PREDICT_PROBA
    _ = member_call(f.member, MEMBER_OP_OUTPUTS, fc)
    require(fc.out_dtype == MEMBER_F64, "the logistic final writes float64 probabilities (this oracle reads them)")
    fc.x = MemberMatrix(Int(meta32.unsafe_ptr()), N, 6)
    fc.dst = MemberMatrix(Int(want32.unsafe_ptr()), N, K, MEMBER_F64)
    _ = member_call(f.member, MEMBER_OP_PREDICT_PROBA, fc)
    _same(got, want32, "stacking predict_proba")
    _ = meta32^
    _ = meta64^
    _ = place^
    _ = pa^
    _ = pb^
    _ = cols^
    _free(meta.plan, state)   # frees s0, s1, s2 with it
    meta.close()
    _ = x^
    _ = y^


def check_nested_parallel() raises:
    var x = _x()
    var y = _codes()
    var want = List[Float64](length=N * K, fill=0.0)
    var want_codes = List[Int32](length=N, fill=Int32(0))
    _vote_door(True, x, y, want, want_codes)
    var meta = _vote_meta(True, 0)
    var mask = member_kind_capabilities(meta.plan)
    for cap in range(2):  # small-loop(two runs: workers (cap 0) and serial (cap 1))
        var run = MemberSetRun(Int(x.unsafe_ptr()), N, D, MEMBER_I32, MEMBER_I32)
        run.cap = cap
        for _ in range(UNITS):  # small-loop(fixture: the units)
            run.add(meta.plan.copy(), Int(y.unsafe_ptr()), 0, 0, 0, 0, 0, UInt8(0), UInt8(1),
                    MEMBER_SET_WORKER if (mask & MEMBER_GIL_FREE) != 0 else UInt8(0))
        member_set_fit(run)
        for u in range(UNITS):  # small-loop(every unit's answer)
            _same(_meta_proba(meta, run.states[u], x), want,
                  "nested unit " + String(u) + (" serial" if cap == 1 else " parallel"))
        member_set_run_release(run)
    meta.close()
    _ = want_codes^
    _ = x^
    _ = y^


# ---------------------------------------------------------------- the unit kinds and the stacking cases (lane mf-meta3b)


def _meta_with(kind: Int, folds: Int, var nested: List[MemberPlan], var words: List[Float64],
               var fold_train: List[List[Int32]], var fold_test: List[List[Int32]], var stack_kinds: List[Int],
               cap: Int) raises -> _Meta:
    """A META plan with the kind's settings words and a stack's user folds (the boundary's `_finish`)."""
    var masks = List[Int]()
    for u in range(len(nested)):  # small-loop(fixture: each nested kind's mask, as the admission asks it)
        masks.append(member_kind_capabilities(nested[u]))
    var handle = meta_plan_mint_with(nested^, List[Float64](), stack_kinds^, masks^, words^, fold_train^, fold_test^)
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    plan.route = route
    plan.member = MemberRef(EntryRef(ensemble_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.family = MEMBER_FAMILY_ENSEMBLE
    plan.params = [Float64(kind), Float64(handle), 0.0, Float64(folds), Float64(cap), 0.0]
    plan.classes = K
    return _Meta(plan^, handle)


def _fit_k(plan: MemberPlan, x: List[Float32], y: List[Int32], k: Int) raises -> Int:
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(y.unsafe_ptr()), N, 1, MEMBER_I32)
    call.classes = k
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    except e:
        member_free(plan, call)
        raise e^
    return call.state


def _block32(plan: MemberPlan, state: Int, operation: Int, x: Int, n: Int, d: Int) raises -> List[Float32]:
    """A member's `operation` block over n rows, float64 rounded to float32 word by word (the door's admission);
    its width from OUTPUTS."""
    var call = member_call_of(plan, state)
    call.flags = operation
    _ = member_call(plan.member, MEMBER_OP_OUTPUTS, call)
    var cols = call.out_cols
    var words = List[Float32](length=max(n * cols, 1), fill=0.0)
    call.x = MemberMatrix(x, n, d)
    if call.out_dtype == MEMBER_F64:
        var wide = List[Float64](length=max(n * cols, 1), fill=0.0)
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), n, cols, MEMBER_F64)
        _ = member_call(plan.member, operation, call)
        for i in range(n * cols):  # small-loop(oracle: the door's float32 admission)
            words[i] = wide[i].cast[DType.float32]()
        _ = wide^
    else:
        call.dst = MemberMatrix(Int(words.unsafe_ptr()), n, cols, MEMBER_F32)
        _ = member_call(plan.member, operation, call)
    return words^


def _cols_of(plan: MemberPlan, state: Int, operation: Int) raises -> Int:
    var call = member_call_of(plan, state)
    call.flags = operation
    _ = member_call(plan.member, MEMBER_OP_OUTPUTS, call)
    return call.out_cols


def _ovr_door(x: List[Float32], y: List[Int32], mut proba: List[Float64], mut codes: List[Int32]) raises:
    """The door's words of a k = 3 OneVsRest over LogisticRegression: one member per class fitted directly on its
    0/1 indicator (two classes), then `ovr_assemble_run` over their decision columns (predict: the base has
    decision_function) and over their code-1 probability columns (predict_proba)."""
    var plans = List[MemberPlan]()
    var states = List[Int]()
    var targets = List[List[Int32]]()
    for j in range(K):  # small-loop(fixture: one member per class)
        var yj = List[Int32](length=N, fill=Int32(0))
        for i in range(N):  # small-loop(fixture: the indicator, computed here independently)
            yj[i] = Int32(1) if Int(y[i]) == j else Int32(0)
        var plan = _logistic(1.0)
        plan.classes = 2
        states.append(_fit_k(plan, x, yj, 2))
        plans.append(plan^)
        targets.append(yj^)
    var rows = VoteRows(Int(x.unsafe_ptr()), N, D)
    var dec = List[List[Float32]]()
    var pro = List[List[Float32]]()
    for j in range(K):  # small-loop(fixture: the members' own blocks)
        dec.append(_block32(plans[j], states[j], MEMBER_OP_DECISION, Int(x.unsafe_ptr()), N, D))
        pro.append(_block32(plans[j], states[j], MEMBER_OP_PREDICT_PROBA, Int(x.unsafe_ptr()), N, D))
    var by_dec = List[VoteMember]()
    var by_pro = List[VoteMember]()
    for j in range(K):  # small-loop(fixture)
        by_dec.append(VoteMember(Int(dec[j].unsafe_ptr()), _cols_of(plans[j], states[j], MEMBER_OP_DECISION), 0, 1.0))
        by_pro.append(VoteMember(Int(pro[j].unsafe_ptr()), 2, 0, 1.0))
    var zeros: List[Int] = [0, 0, 0]
    var ones: List[Int] = [1, 1, 1]
    require(ovr_assemble_run(_glue(), rows, by_dec, zeros, K, 2, Int(codes.unsafe_ptr())) == VOTE_ALL_ANSWERED,
            "door decision blocks refused")
    require(ovr_assemble_run(_glue(), rows, by_pro, ones, K, 1, Int(proba.unsafe_ptr())) == VOTE_ALL_ANSWERED,
            "door probability blocks refused")
    _ = dec^
    _ = pro^
    _ = targets^
    for j in range(K):  # small-loop(fixture)
        _free(plans[j], states[j])


def _ovr_meta(cap: Int) raises -> _Meta:
    var nested: List[MemberPlan] = [_logistic(1.0)]
    var words: List[Float64] = [1.0]  # META_OVR_DECISION: LogisticRegression has decision_function
    return _meta_with(META_OVR, 0, nested^, words^, List[List[Int32]](), List[List[Int32]](), List[Int](), cap)


def check_ovr() raises:
    var x = _x()
    var y = _codes()
    var want = List[Float64](length=N * K, fill=0.0)
    var want_codes = List[Int32](length=N, fill=Int32(0))
    _ovr_door(x, y, want, want_codes)
    var meta = _ovr_meta(0)
    require((member_kind_capabilities(meta.plan) & MEMBER_GIL_FREE) != 0, "OVR kind lost MEMBER_GIL_FREE")
    var state = _meta_fit(meta, x, y)
    _same(_meta_proba(meta, state, x), want, "OneVsRest predict_proba")
    _same_i32(_meta_codes(meta, state, x), want_codes, "OneVsRest predict")
    var call = MemberCall(meta.plan.route, state)
    call.block = META_BLOCK_NESTED
    _ = member_call(meta.plan.member, MEMBER_OP_EXPORT, call)
    require(call.dst.cols == K, "OneVsRest exports one unit per class")
    _free(meta.plan, state)
    meta.close()
    _ = x^
    _ = y^


def check_ovr_parallel() raises:
    """Four OneVsRest META units fitted as one member set on host workers and again serially: every unit's
    predict_proba equals the door's words."""
    var x = _x()
    var y = _codes()
    var want = List[Float64](length=N * K, fill=0.0)
    var want_codes = List[Int32](length=N, fill=Int32(0))
    _ovr_door(x, y, want, want_codes)
    var meta = _ovr_meta(0)
    var mask = member_kind_capabilities(meta.plan)
    for cap in range(2):  # small-loop(two runs: workers (cap 0) and serial (cap 1))
        var run = MemberSetRun(Int(x.unsafe_ptr()), N, D, MEMBER_I32, MEMBER_I32)
        run.cap = cap
        for _ in range(UNITS):  # small-loop(fixture: the units)
            run.add(meta.plan.copy(), Int(y.unsafe_ptr()), 0, 0, 0, 0, 0, UInt8(0), UInt8(1),
                    MEMBER_SET_WORKER if (mask & MEMBER_GIL_FREE) != 0 else UInt8(0))
        member_set_fit(run)
        for u in range(UNITS):  # small-loop(every unit's answer)
            _same(_meta_proba(meta, run.states[u], x), want,
                  "nested OneVsRest unit " + String(u) + (" serial" if cap == 1 else " parallel"))
        member_set_run_release(run)
    meta.close()
    _ = want_codes^
    _ = x^
    _ = y^


def check_calibrated() raises:
    """CalibratedClassifierCV, cv = 3, sigmoid, ensemble=True, decision_function base: META predict_proba and
    predict == the door's RUN words: the same stratified folds, one unit per fold as one set,
    `calibrated_fit_run`, then `calibrated_proba_run` over the run's own states and calibrators."""
    var x = _x()
    var y = _codes()
    var nested: List[MemberPlan] = [_logistic(1.0)]
    var words: List[Float64] = [1.0, 1.0, 1.0]  # sigmoid, ensemble, decision
    var meta = _meta_with(META_CALIBRATED, 3, nested^, words^, List[List[Int32]](), List[List[Int32]](), List[Int](), 0)
    var state = _meta_fit(meta, x, y)
    var got = _meta_proba(meta, state, x)
    var got_codes = _meta_codes(meta, state, x)
    # The door's RUN words.
    var counts = List[Int32](length=4, fill=Int32(0))
    var rows = List[Int32](length=3 * N, fill=Int32(0))
    require(xt_device_folds(_glue(), Int(y.unsafe_ptr()), Int(rows.unsafe_ptr()), Int(counts.unsafe_ptr()), N, 3, K)
            == 0, "stratified folds")
    var base = _logistic(1.0)
    var mask = member_kind_capabilities(base)
    var run = MemberSetRun(Int(x.unsafe_ptr()), N, D, MEMBER_I32, MEMBER_I32)
    for f in range(3):  # small-loop(fixture: the folds)
        var c = Int(counts[f])
        run.add(base.copy(), Int(y.unsafe_ptr()), 0, Int(rows.unsafe_ptr()) + 4 * (f * N), N - c, 0, 0, UInt8(0),
                UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0),
                MEMBER_SET_WORKER if (mask & MEMBER_GIL_FREE) != 0 else UInt8(0))
    var plan = CalFitPlan(Int(x.unsafe_ptr()), N, D, Int(y.unsafe_ptr()), K, MEMBER_OP_DECISION, True, True, False)
    for f in range(3):  # small-loop(fixture: the folds' test rows)
        var c = Int(counts[f])
        plan.fold(Int(rows.unsafe_ptr()) + 4 * (f * N + N - c), c)
    var res = CalFitOut()
    require(calibrated_fit_run(_glue(), run, plan, res) == CAL_ALL_ANSWERED, "the door's calibrated RUN failed")
    var cols = _iota()
    var members = List[VoteMember]()
    var pairs = List[CalPair]()
    var cals = List[CalFit]()
    for f in range(3):  # small-loop(fixture: the pairs)
        var c = _cols_of(run.plans[f], run.states[f], MEMBER_OP_DECISION)
        var probe = member_call_of(run.plans[f], run.states[f])
        probe.flags = MEMBER_OP_DECISION
        _ = member_call(run.plans[f].member, MEMBER_OP_OUTPUTS, probe)
        members.append(VoteMember(Int(run.plans.unsafe_ptr() + f), run.states[f], MEMBER_OP_DECISION, c,
                                  probe.out_dtype, Int(cols.unsafe_ptr()), 1.0))
        pairs.append(CalPair(0, Int(cols.unsafe_ptr()), K, len(cals), res.count[f]))
        for j in range(res.first[f], res.first[f] + res.count[f]):  # small-loop(fixture: the pair's calibrators)
            cals.append(CalFit(res.isotonic[j], res.a[j], res.b[j], Int(res.knots_x[j].unsafe_ptr()),
                               Int(res.knots_y[j].unsafe_ptr()), res.m[j]))
    var want = List[Float64](length=N * K, fill=0.0)
    require(calibrated_proba_run(_glue(), VoteRows(Int(x.unsafe_ptr()), N, D), members, pairs, cals, K,
                                 Int(want.unsafe_ptr())) == VOTE_ALL_ANSWERED, "a door pair refused")
    _same(got, want, "CalibratedClassifierCV predict_proba")
    for i in range(N):  # small-loop(oracle: the row argmax, first maximum)
        var best = 0
        for c in range(1, K):  # small-loop(oracle)
            if want[i * K + c] > want[i * K + best]:
                best = c
        require(Int(got_codes[i]) == best, "CalibratedClassifierCV predict row " + String(i))
    member_set_run_release(run)
    _ = plan.folds()
    _ = res.calibrators()
    _ = cols^
    _ = rows^
    _ = counts^
    _free(meta.plan, state)
    meta.close()
    _ = x^
    _ = y^


def check_stack_predict_labels() raises:
    """StackingClassifier with stack_method='predict' (lane mf-meta3b): META predict_proba == the final member's
    own predict_proba over the members' predicted codes as float32 meta features (widths 1 + 1)."""
    var x = _x()
    var y = _codes()
    var nested: List[MemberPlan] = [_logistic(1.0), _logistic(0.25), _logistic(1.0)]
    var kinds: List[Int] = [STACK_PREDICT, STACK_PREDICT]
    var meta = _meta_with(META_STACK_CLASSIFIER, 3, nested^, List[Float64](), List[List[Int32]](),
                          List[List[Int32]](), kinds^, 0)
    var state = _meta_fit(meta, x, y)
    var got = _meta_proba(meta, state, x)
    var call = MemberCall(meta.plan.route, state)
    call.block = META_BLOCK_NESTED
    _ = member_call(meta.plan.member, MEMBER_OP_EXPORT, call)
    var handles = _I64P(unsafe_from_address=call.dst.data)
    var s0 = Int(handles[0])
    var s1 = Int(handles[1])
    var s2 = Int(handles[2])
    var w = MemberCall(meta.plan.route, state)
    w.block = META_BLOCK_WIDTHS
    _ = member_call(meta.plan.member, MEMBER_OP_EXPORT, w)
    var widths = _I64P(unsafe_from_address=w.dst.data)
    require(w.dst.cols == 2 and widths[0] == 1 and widths[1] == 1, "stacking predict widths 1 + 1")
    var ca = _codes32(_logistic(1.0), s0, Int(x.unsafe_ptr()))
    var cb = _codes32(_logistic(0.25), s1, Int(x.unsafe_ptr()))
    var meta32 = List[Float32](length=N * 2, fill=0.0)
    for i in range(N):  # small-loop(oracle: the codes as float32 meta features, member after member)
        meta32[i * 2] = Float32(Int(ca[i]))
        meta32[i * 2 + 1] = Float32(Int(cb[i]))
    var f = _logistic(1.0)
    var want = List[Float64](length=N * K, fill=0.0)
    var fc = member_call_of(f, s2)
    fc.x = MemberMatrix(Int(meta32.unsafe_ptr()), N, 2)
    fc.dst = MemberMatrix(Int(want.unsafe_ptr()), N, K, MEMBER_F64)
    _ = member_call(f.member, MEMBER_OP_PREDICT_PROBA, fc)
    _same(got, want, "stacking predict-label predict_proba")
    _ = meta32^
    _ = ca^
    _ = cb^
    _free(meta.plan, state)
    meta.close()
    _ = x^
    _ = y^


def _user_split(partition: Bool, mut fold_train: List[List[Int32]], mut fold_test: List[List[Int32]]):
    """Three folds: test rows i % 3 == f, train the rest; `partition` False drops row 0 from every test side."""
    for f in range(3):  # small-loop(fixture: the folds)
        var tr = List[Int32]()
        var te = List[Int32]()
        for i in range(N):  # small-loop(fixture)
            if i % 3 == f:
                if partition or i != 0:
                    te.append(Int32(i))
            else:
                tr.append(Int32(i))
        fold_train.append(tr^)
        fold_test.append(te^)


def check_stack_user_folds() raises:
    """StackingClassifier over a user's index list (lane mf-meta3b): it fits (widths 3 + 3) and a list that is not
    a partition is refused in the door's `_splits` words."""
    var x = _x()
    var y = _codes()
    var tr = List[List[Int32]]()
    var te = List[List[Int32]]()
    _user_split(True, tr, te)
    var nested: List[MemberPlan] = [_logistic(1.0), _logistic(0.25), _logistic(1.0)]
    var kinds: List[Int] = [STACK_PROBA, STACK_PROBA]
    var meta = _meta_with(META_STACK_CLASSIFIER, 3, nested^, List[Float64](), tr^, te^, kinds^, 0)
    var state = _meta_fit(meta, x, y)
    var w = MemberCall(meta.plan.route, state)
    w.block = META_BLOCK_WIDTHS
    _ = member_call(meta.plan.member, MEMBER_OP_EXPORT, w)
    var widths = _I64P(unsafe_from_address=w.dst.data)
    require(w.dst.cols == 2 and widths[0] == 3 and widths[1] == 3, "user-fold stacking widths 3 + 3")
    _free(meta.plan, state)
    meta.close()
    var tr2 = List[List[Int32]]()
    var te2 = List[List[Int32]]()
    _user_split(False, tr2, te2)
    var nested2: List[MemberPlan] = [_logistic(1.0), _logistic(0.25), _logistic(1.0)]
    var kinds2: List[Int] = [STACK_PROBA, STACK_PROBA]
    var bad = _meta_with(META_STACK_CLASSIFIER, 3, nested2^, List[Float64](), tr2^, te2^, kinds2^, 0)
    var words = String("")
    try:
        var s = _meta_fit(bad, x, y)
        _free(bad.plan, s)
    except e:
        words = String(e)
    require(words == "cross_val_predict only works for partitions: every row must be in exactly one test set",
            "user folds that are not a partition: got '" + words + "'")
    bad.close()
    _ = x^
    _ = y^


def check_nested_two_deep() raises:
    var x = _x()
    var y = _codes()
    var inner = _vote_meta(True, 0)
    var c = _logistic(0.5)
    # the door's words: the inner META and the logistic member fitted directly, their blocks voted
    var si = _meta_fit(inner, x, y)
    var sc = _fit_member(c, x, y)
    var wide = _meta_proba(inner, si, x)
    var pa = List[Float32](length=N * K, fill=0.0)
    for i in range(N * K):  # small-loop(oracle: the door's float32 admission of a float64 block)
        pa[i] = wide[i].cast[DType.float32]()
    var pc = _proba32(c, sc, Int(x.unsafe_ptr()), N, D)
    var cols = _iota()
    var rows = VoteRows(Int(x.unsafe_ptr()), N, D)
    var door: List[VoteMember] = [VoteMember(Int(pa.unsafe_ptr()), K, Int(cols.unsafe_ptr()), 1.0),
                                  VoteMember(Int(pc.unsafe_ptr()), K, Int(cols.unsafe_ptr()), 1.0)]
    var want = List[Float64](length=N * K, fill=0.0)
    var want_codes = List[Int32](length=N, fill=Int32(0))
    _ = vote_soft_run(_glue(), rows, door, K, Int(want.unsafe_ptr()))
    var acc = List[Float64](length=N * K, fill=0.0)
    _ = vote_codes_run(_glue(), rows, door, K, True, Int(acc.unsafe_ptr()), Int(want_codes.unsafe_ptr()))
    _ = acc^
    _ = pa^
    _ = pc^
    _ = cols^
    _free(inner.plan, si)
    _free(c, sc)
    # the META of a META
    var nested: List[MemberPlan] = [inner.plan.copy(), _logistic(0.5)]
    var weights: List[Float64] = [1.0, 1.0]
    var outer = _meta(META_VOTE_CLASSIFIER, 1.0, 0, nested^, weights^, List[Int](), 0)
    require((member_kind_capabilities(outer.plan) & MEMBER_CAN_PREDICT_PROBA) != 0, "nested soft vote has no proba")
    var state = _meta_fit(outer, x, y)
    _same(_meta_proba(outer, state, x), want, "nested vote predict_proba")
    _same_i32(_meta_codes(outer, state, x), want_codes, "nested vote predict")
    _free(outer.plan, state)
    outer.close()
    inner.close()  # the inner plan entry is read by every NEW of the inner META (the outer fit's) up to here
    _ = x^
    _ = y^


# ------------------------------------------------------------ THE ADABOOST KIND (lane mf-ada4)


def _targets(x: List[Float32]) -> List[Float32]:
    var y = List[Float32](length=N, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        y[i] = x[i * D] * 2.0 + x[i * D + 1] - Float32(i % 4) * 0.5
    return y^


def _session_served() raises -> Bool:
    """The host RF image serves the data session (the kind and the door's `_member_session` open one)."""
    return (xf_caps(entry_ref_gil_free(xf_ref(ForestEntryKey(FOREST_TABLE_RF, trees_key())))) & XF_CAP_SESSION) != 0


def _ada_meta(kind: Int, var base: MemberPlan, rounds: Int, seed: Int, loss: Int, native: Bool) raises -> _Meta:
    """An AdaBoost META plan as the boundary's `_admit_ada` mints it: the base's forest member plan and the
    settings words (n_estimators, seed, learning_rate 1.0, loss, the session route word, no table sharing, the
    squared criterion code, no target check, not a random-splitter tree; no class-weight tail)."""
    var words: List[Float64] = [Float64(rounds), Float64(seed), 1.0, Float64(loss), 1.0 if native else 0.0, 0.0,
                                Float64(RF_SQUARED), -1.0, -2.0, -2.0, -2.0, 0.0]
    var masks: List[Int] = [member_kind_capabilities(base)]
    var plans: List[MemberPlan] = [base^]
    var handle = meta_plan_mint_with(plans^, List[Float64](), List[Int](), masks^, words^, List[List[Int32]](),
                                     List[List[Int32]]())
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    plan.route = route
    plan.member = MemberRef(EntryRef(ensemble_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.family = MEMBER_FAMILY_ENSEMBLE
    plan.params = [Float64(kind), Float64(handle), 0.0, 0.0, 0.0, 0.0]
    plan.classes = K if kind == META_ADA_CLASSIFIER else 0
    return _Meta(plan^, handle)


def _ada_words(meta: _Meta, state: Int) raises -> List[Float64]:
    var call = MemberCall(meta.plan.route, state)
    call.block = META_BLOCK_ADA
    _ = member_call(meta.plan.member, MEMBER_OP_EXPORT, call)
    var out = List[Float64]()
    var w = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    for i in range(call.dst.rows * call.dst.cols):  # small-loop(the kept rounds' words)
        out.append(w[i])
    return out^


def _want_words(alphas: List[Float64], errors: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    out.append(Float64(len(alphas)))
    for i in range(len(alphas)):  # small-loop(oracle)
        out.append(alphas[i])
    out.append(Float64(len(errors)))
    for i in range(len(errors)):  # small-loop(oracle)
        out.append(errors[i])
    return out^


def _ada_door(m: AdaModel, x: List[Float32], finish: Int, width: Int) raises -> List[Float64]:
    """The door's predict_proba / decision_function: each member's host codes, `ada_classifier_run`."""
    var blocks = List[List[Int32]]()
    var addrs = List[Int]()
    for j in range(len(m.members)):  # small-loop(oracle: the members' host codes)
        var c = argmax_rows(predict(m.members[j], x, N), N, m.k)
        var words = List[Int32](length=N, fill=Int32(0))
        for i in range(N):  # small-loop(oracle)
            words[i] = Int32(c[i])
        blocks.append(words^)
    for j in range(len(blocks)):  # small-loop(oracle)
        addrs.append(Int(blocks[j].unsafe_ptr()))
    var acc = List[Float64](length=N * m.k, fill=0.0)
    var out = List[Float64](length=N * width, fill=0.0)
    ada_classifier_run(_glue(), addrs, m.alphas, N, m.k, Int(acc.unsafe_ptr()), finish, Int(out.unsafe_ptr()))
    _ = len(blocks)
    if m.k > 2:
        return acc^  # more than two classes: the finish wrote the block in place
    return out^


def _meta_decision(meta: _Meta, state: Int, x: List[Float32]) raises -> List[Float64]:
    var got = List[Float64](length=N * K, fill=-7.0)
    var call = member_call_of(meta.plan, state)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.dst = MemberMatrix(Int(got.unsafe_ptr()), N, K, MEMBER_F64)
    _ = member_call(meta.plan.member, MEMBER_OP_DECISION, call)
    return got^


def check_ada_classifier() raises:
    var x = _x()
    var y = _codes()
    var s = AdaSpec(True, 5, 3)
    var door = ada_fit(s, x, N, D, y, K, List[Float32](), _session_served())
    var meta = _ada_meta(META_ADA_CLASSIFIER, member_plan_of(rf_config(RfSpec.decision_tree(True, 1)),
                                                             MEMBER_FAMILY_TREE, True), 5, 3, 0, True)
    var state = _meta_fit(meta, x, y)
    _same(_ada_words(meta, state), _want_words(door.alphas, door.errors), "AdaBoost weights and errors")
    _same_i32(_meta_codes(meta, state, x), ada_predict_codes(door, x, N), "AdaBoost predict")
    _same(_meta_proba(meta, state, x), _ada_door(door, x, ADA_PROBA, K), "AdaBoost predict_proba")
    _same(_meta_decision(meta, state, x), _ada_door(door, x, ADA_DECISION, K), "AdaBoost decision_function")
    _free(meta.plan, state)
    meta.close()
    _ = x^
    _ = y^


def check_ada_regressor() raises:
    var x = _x()
    var t = _targets(x)
    var s = AdaSpec(False, 5, 3)
    s.loss = 1
    var door = ada_fit(s, x, N, D, List[Int32](), 0, t, _session_served())
    var meta = _ada_meta(META_ADA_REGRESSOR, member_plan_of(rf_config(RfSpec.decision_tree(False, 3)),
                                                            MEMBER_FAMILY_TREE, False), 5, 3, 1, True)
    var call = member_new(meta.plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(t.unsafe_ptr()), N, 1, MEMBER_F32)
    _ = member_call(meta.plan.member, MEMBER_OP_FIT, call)
    var state = call.state
    _same(_ada_words(meta, state), _want_words(door.alphas, door.errors), "AdaBoostRegressor weights and errors")
    var got = List[Float32](length=N, fill=-7.0)
    var p = member_call_of(meta.plan, state)
    p.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    p.dst = MemberMatrix(Int(got.unsafe_ptr()), N, 1, MEMBER_F32)
    _ = member_call(meta.plan.member, MEMBER_OP_PREDICT, p)
    var want = ada_predict_values(door, x, N)
    for i in range(N):  # small-loop(oracle: every value)
        require(got[i] == want[i], "AdaBoostRegressor predict row " + String(i) + ": META " + String(got[i])
                + " != door " + String(want[i]))
    _free(meta.plan, state)
    meta.close()
    _ = x^
    _ = t^


def check_ada_member_route() raises:
    var x = _x()
    var y = _codes()
    var s = AdaSpec(True, 4, 1)
    var r = RfSpec(True, 3, 0)
    r.max_depth = 2
    var door = ada_member_fit(s, rf_config(r), MEMBER_FAMILY_FOREST, False, x, N, D, y, K, List[Float32](),
                              List[Float32]())
    var meta = _ada_meta(META_ADA_CLASSIFIER, member_plan_of(rf_config(r), MEMBER_FAMILY_FOREST, True), 4, 1, 0, False)
    var state = _meta_fit(meta, x, y)
    _same(_ada_words(meta, state), _want_words(door.alphas, door.errors), "AdaBoost(RF) weights and errors")
    _same_i32(_meta_codes(meta, state, x), door.codes, "AdaBoost(RF) predict")
    _free(meta.plan, state)
    meta.close()
    _ = x^
    _ = y^


def _copy_forest_state(plan: MemberPlan, state: Int) raises -> Int:
    """A new state of `plan` holding `state`'s six fitted blocks (EXPORT by index, IMPORT, the last DONE)."""
    var probe = MemberCall(plan.route, state)
    probe.block = 0
    probe.flags = MEMBER_BLOCK_BY_INDEX
    _ = member_call(plan.member, MEMBER_OP_EXPORT, probe)
    var blocks = probe.out_cols
    var copy = member_new(plan)
    for i in range(blocks):  # small-loop(oracle: one member's fitted blocks)
        var view = MemberCall(plan.route, state)
        view.block = i
        view.flags = MEMBER_BLOCK_BY_INDEX
        _ = member_call(plan.member, MEMBER_OP_EXPORT, view)
        copy.block = view.block
        copy.flags = MEMBER_IMPORT_DONE if i == blocks - 1 else 0
        copy.x = view.dst
        _ = member_call(plan.member, MEMBER_OP_IMPORT, copy)
    return copy.state


def check_ada_import() raises:
    var x = _x()
    var y = _codes()
    var base = member_plan_of(rf_config(RfSpec.decision_tree(True, 1)), MEMBER_FAMILY_TREE, True)
    var meta = _ada_meta(META_ADA_CLASSIFIER, base.copy(), 5, 3, 0, True)
    var state = _meta_fit(meta, x, y)
    var want = _meta_proba(meta, state, x)
    var nested = MemberCall(meta.plan.route, state)
    nested.block = META_BLOCK_NESTED
    _ = member_call(meta.plan.member, MEMBER_OP_EXPORT, nested)
    var handles = List[Int64]()
    var hw = _I64P(unsafe_from_address=nested.dst.data)
    for u in range(nested.dst.rows * nested.dst.cols):  # small-loop(oracle: the rounds' states copied)
        handles.append(Int64(_copy_forest_state(base, Int(hw[u]))))
    var words = _ada_words(meta, state)
    var call = member_call_of(meta.plan)
    _ = member_call(meta.plan.member, MEMBER_OP_NEW, call)
    var block = MemberCall(meta.plan.route, call.state)
    block.block = META_BLOCK_NESTED
    block.x = MemberMatrix(Int(handles.unsafe_ptr()), 1, len(handles), MEMBER_I64)
    _ = member_call(meta.plan.member, MEMBER_OP_IMPORT, block)
    var aw = MemberCall(meta.plan.route, call.state)
    aw.block = META_BLOCK_ADA
    aw.x = MemberMatrix(Int(words.unsafe_ptr()), 1, len(words), MEMBER_F64)
    _ = member_call(meta.plan.member, MEMBER_OP_IMPORT, aw)
    var last = MemberCall(meta.plan.route, call.state)
    last.block = 0
    last.flags = MEMBER_IMPORT_DONE
    last.classes = K
    _ = member_call(meta.plan.member, MEMBER_OP_IMPORT, last)
    _same(_meta_proba(meta, call.state, x), want, "AdaBoost imported predict_proba")
    _ = len(handles)
    _ = len(words)
    _free(meta.plan, call.state)
    _free(meta.plan, state)
    meta.close()
    _ = x^
    _ = y^


# ---------------------------------------------------------------- the search and multi-output kinds (lane mf-search4)


def _search_plans() raises -> List[MemberPlan]:
    var out: List[MemberPlan] = [_logistic(1.0), _logistic(0.1), _logistic(10.0)]
    return out^


def check_search() raises:
    """15. GridSearchCV (META_SEARCH_CLASSIFIER) over three LogisticRegression candidates, cv = 3, scoring None:
    the META's search words (META_BLOCK_SEARCH) == `msel_search_run` over the same candidates (whose words
    checks/native_tests/msel_search_run.mojo proves against the door's), and its predict_proba == the RUN's refit
    state's."""
    var x = _x()
    var y = _codes()
    # The kind's settings words (core/ensemble_member.mojo META_SEARCH_*, lane mf-cluster5: the 9 words the
    # kind requires since mf-search4b): scoring None, error_score='raise', no train scores, a grid (seed -1),
    # refit=True, refit scorer 0, one scorer (count 0), then its one positive word (-1: none).
    var words: List[Float64] = [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0]
    require(len(words) == META_SEARCH_WORDS + 1, "fixture: the search kind's settings words")
    var meta = _meta_with(META_SEARCH_CLASSIFIER, 3, _search_plans(), words^, List[List[Int32]](),
                          List[List[Int32]](), List[Int](), 0)
    var state = _meta_fit(meta, x, y)
    var masks = List[Int]()
    var plans = _search_plans()
    for c in range(len(plans)):  # small-loop(fixture: each candidate kind's mask)
        masks.append(member_kind_capabilities(plans[c]))
    var plan = MselSearchPlan(plans.copy(), masks^, Int(x.unsafe_ptr()), N, D, Int(y.unsafe_ptr()), K, 3, _glue(),
                              MetricRoute(entry_ref_unresolved().call, False, False))
    var res = msel_search_run(plan)
    var view = member_call_of(meta.plan, state)
    view.block = META_BLOCK_SEARCH
    _ = member_call(meta.plan.member, MEMBER_OP_EXPORT, view)
    var got = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=view.dst.data)
    require(Int(got[0]) == res.best and Int(got[1]) == res.n_splits and got[2] == res.best_score
            and Int(got[3]) == len(plans), "the search header words")
    for c in range(len(plans)):  # small-loop(check: candidates)
        for f in range(res.n_splits):  # small-loop(check: folds)
            # the tables follow the fitted header (META_SEARCH_HEAD words; it was 6 before mf-search4b)
            require(got[META_SEARCH_HEAD + c * res.n_splits + f] == res.test[c][f], "a split score")
    var want = List[Float64](length=N * K, fill=0.0)
    var q = member_call_of(plans[res.best], res.best_state)
    q.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    q.dst = MemberMatrix(Int(want.unsafe_ptr()), N, K, MEMBER_F64)
    _ = member_call(plans[res.best].member, MEMBER_OP_PREDICT_PROBA, q)
    _same(_meta_proba(meta, state, x), want, "the search's predict_proba (its best estimator's)")
    _free(plans[res.best], res.best_state)
    _free(meta.plan, state)
    meta.close()
    _ = x^
    _ = y^


def check_multi_output() raises:
    """16. MultiOutputClassifier (META_MULTI_CLASSIFIER) over LogisticRegression, two outputs with 3 and 2
    classes (THE 2-D y: rows x 2 int32 codes, `class_counts` [3, 2]): predict (rows x 2 int32) == each output's
    member fitted directly on its column with its count; score == the share of rows whose both outputs match,
    counted here."""
    var x = _x()
    var y = _codes()
    var y2 = List[Int32](length=N * 2, fill=Int32(0))
    var y1 = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: the second output's two classes)
        y1[i] = Int32((i // 2) % 2)
        y2[2 * i] = y[i]
        y2[2 * i + 1] = y1[i]
    var counts: List[Int] = [K, 2]
    var meta = _meta_with(META_MULTI_CLASSIFIER, 0, [_logistic(1.0)], [0.0], List[List[Int32]](),
                          List[List[Int32]](), List[Int](), 0)
    var call = member_new(meta.plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(y2.unsafe_ptr()), N, 2, MEMBER_I32)
    call.class_counts = Int(counts.unsafe_ptr())
    _ = member_call(meta.plan.member, MEMBER_OP_FIT, call)
    var state = call.state
    var got = List[Int32](length=N * 2, fill=Int32(-1))
    var p = member_call_of(meta.plan, state)
    p.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    p.dst = MemberMatrix(Int(got.unsafe_ptr()), N, 2, MEMBER_I32)
    _ = member_call(meta.plan.member, MEMBER_OP_PREDICT, p)
    var base = _logistic(1.0)
    var s0 = _fit_k(base, x, y, K)
    var s1 = _fit_k(base, x, y1, 2)
    var w0 = _codes32(base, s0, Int(x.unsafe_ptr()))
    var w1 = _codes32(base, s1, Int(x.unsafe_ptr()))
    var hits = 0
    for i in range(N):  # small-loop(check)
        require(got[2 * i] == w0[i] and got[2 * i + 1] == w1[i], "an output's predicted code")
        if w0[i] == y[i] and w1[i] == y1[i]:
            hits += 1
    var sc = member_call_of(meta.plan, state)
    sc.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    sc.y = MemberMatrix(Int(y2.unsafe_ptr()), N, 2, MEMBER_I32)
    _ = member_call(meta.plan.member, MEMBER_OP_SCORE, sc)
    require(sc.value == Float64(hits) / Float64(N), "MultiOutputClassifier.score")
    _free(base, s0)
    _free(base, s1)
    _free(meta.plan, state)
    meta.close()
    _ = counts^
    _ = y2^
    _ = x^
    _ = y^


def test_meta_member_kind() raises:
    # The publication _mojolearn_x_trees_host makes at module init (the META entry's glue), once.
    publish_x_trees_glue("cpu")
    # The AdaBoost cases' forest images and member entries (lane mf-ada4).
    publish_host_images()
    publish_host_members()
    def run_case(which: Int) raises:
        if which == 0:
            check_soft_vote()
        elif which == 1:
            check_hard_vote()
        elif which == 2:
            check_stack()
        elif which == 3:
            check_nested_parallel()
        elif which == 4:
            check_ovr()
        elif which == 5:
            check_ovr_parallel()
        elif which == 6:
            check_calibrated()
        elif which == 7:
            check_stack_predict_labels()
        elif which == 8:
            check_stack_user_folds()
        elif which == 9:
            check_nested_two_deep()
        elif which == 10:
            check_ada_classifier()
        elif which == 11:
            check_ada_regressor()
        elif which == 12:
            check_ada_member_route()
        elif which == 13:
            check_ada_import()
        elif which == 14:
            check_search()
        else:
            check_multi_output()
    check_cases("meta_member_kind", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
