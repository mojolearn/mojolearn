# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE RFE KIND (core/ensemble_member.mojo META_RFE_CLASSIFIER, lane mf-meta5) against the RFE RUN it runs
whole (bindings/prep_select_run.mojo `rfe_fit_run`, THE ONE BODY RFE's door runs), on the host images: the
x_trees glue (the META entry's), the host x_prep transport (the rounds' programs) and the discriminant member
(LinearDiscriminantAnalysis, solver 'svd', the base). Inputs: the fixed rows of the shaprfe fixture pack
(checks/native_tests/fixtures/prep_sel_mf/, lda-k3: 300 x 8, three classes), as prep_feature_selection's
`test_rfe_lda` reads them.

    test_rfe_kind_equals_run   n_features_to_select 3, step 2: the kind's META_BLOCK_RFE words (support,
                               ranking, kept columns) equal the RUN's; its predict / predict_proba / transform
                               equal the RUN's final state's on the kept columns; its export round trip
                               (META_BLOCK_NESTED + META_BLOCK_RFE into a new state) predicts the same codes

FAIL-PROOF (each edit makes the case fail): in `_rfe_block` gather `slot[].rfe_columns` shifted by one (or
skip the gather) -> predict / transform; in `_rfe_view` write the ranking before the support -> the words; in
`_import_rfe` skip the kept columns -> the round trip refuses; (lane mf-meta5b) in `_export_forward` read
`words[1]` as the index (or drop `call[].dst = inner.dst`) -> the forwarded coef_ words; drop its range check
-> the past-the-end refusal.

    test_rfe_kind_over_meta_base   (lane mf-del5b) THE RFE KIND over a META base, as a dotted
                               importance_getter 'estimator_.coef_' admits it: the outer RFE's nested member
                               is an inner RFE kind (n_features_to_select 8 >= d: no rounds, so its estimator_
                               is LDA on every column it is given) and the outer plan entry carries THE
                               FORWARDED EXPORT's hops [0, MEMBER_BLOCK_COEF] (`meta_plan_rfe`). The outer
                               kind's META_BLOCK_RFE words equal the leaf RUN's (squared coef_ over the same
                               blocks) and the RUN over the META base with the same hops (RFE's door's form)

FAIL-PROOF (lane mf-del5b): in `_fit_rfe` drop `plan.hops = slot[].rfe_hops.copy()` (or in NEW drop the
`rfe_hops` copy) -> the forwarded export reads no path and the outer fit refuses; in `meta_plan_rfe` drop
`entry[].rfe_hops = hops^` -> the same.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require, fixture_f32, fixture_i32
from checks.native_tests.prep_cls_host import da_host_mode
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.ensemble_member import (
    ensemble_member_v2, meta_plan_mint_with, meta_plan_rfe, meta_plan_release, META_RFE_CLASSIFIER,
    META_RFE_WORDS, META_BLOCK_NESTED, META_BLOCK_RFE, META_PARAM_COUNT, META_BLOCK_FORWARD,
)
from core.member_abi import (
    MemberCall, MemberPlan, MemberRef, MemberMatrix, member_call, member_call_of, member_new, member_free,
    member_kind_capabilities, MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA,
    MEMBER_OP_TRANSFORM, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_OP_OUTPUTS, MEMBER_IMPORT_DONE,
    MEMBER_I32, MEMBER_I64, MEMBER_F32, MEMBER_F64, MEMBER_BLOCK_COEF,
)
from core.member_families import MEMBER_FAMILY_DISCRIMINANT, MEMBER_FAMILY_ENSEMBLE
from core.discriminant_member import discriminant_member_v2
from bindings.prep_classify_prog_run import ClsRunPlan
from bindings.prep_select_run import RfeFitPlan, rfe_fit_run, rfe_member_source, RFE_SELECT_COUNT
from x_prep.transport_host import x_prep_transport_host_v1
from x_prep.transport_abi import XP_OP_COUNT
from xtrees.glue_runs import publish_x_trees_glue

comptime DIR = "checks/native_tests/fixtures/prep_sel_mf/"
comptime N = 300
comptime D = 8
comptime K = 3
comptime NSEL = 3
comptime STEP = 2


def _xprep_route() -> DeviceRoute:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(x_prep_transport_host_v1, entry_ref_word(mode, 0, XP_OP_COUNT))
    route.flags = 0
    return route


def _lda() raises -> MemberPlan:
    """LinearDiscriminantAnalysis(solver='svd') as the door's admission writes it on the identical host image
    (prep_feature_selection `da_params(0, 0)`)."""
    var mode = da_host_mode(True)
    var plan = MemberPlan()
    plan.route = _xprep_route()
    plan.family = MEMBER_FAMILY_DISCRIMINANT
    plan.kind = 0
    plan.member = MemberRef(EntryRef(discriminant_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params = [0.0, 0.0, 0.0, -1.0, 1e-4, 0.0, -1.0, Float64(1 if mode.stats_blocked else 0),
                   Float64(1 if mode.class_onepass else 0), Float64(mode.gram), 0.0, 0.0, 0.0]
    plan.classes = K
    return plan^


def _words(block: Int, squared: Bool) -> List[Float64]:
    """THE RFE KIND's settings words for n_features_to_select NSEL and step STEP (the boundary's)."""
    return _words_n(NSEL, block, squared)


def _words_n(nsel: Int, block: Int, squared: Bool) -> List[Float64]:
    """THE RFE KIND's settings words for n_features_to_select `nsel` and step STEP (the boundary's)."""
    var w: List[Float64] = [Float64(RFE_SELECT_COUNT), 0.0, Float64(nsel), 0.0, 0.0, Float64(STEP), Float64(block),
                            1.0 if squared else 0.0, 0.0, 1.0, 0.0]
    return w^


def _rfe_meta(handle: Int) raises -> MemberPlan:
    """An RFE classifier kind's META plan over plan entry `handle` (the boundary's minted plan)."""
    var meta = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    meta.route = route
    meta.member = MemberRef(EntryRef(ensemble_member_v2, 0), 0)
    var probe = MemberCall(meta.route)
    meta.member.capabilities = member_call(meta.member, MEMBER_OP_CAPABILITIES, probe)
    meta.family = MEMBER_FAMILY_ENSEMBLE
    meta.params = [Float64(META_RFE_CLASSIFIER), Float64(handle), 0.0, 0.0, 0.0, 0.0]
    meta.classes = K
    return meta^


def _rfe_kind_words(meta: MemberPlan, state: Int) raises -> List[Float64]:
    """The fitted state's META_BLOCK_RFE words, copied out of its view."""
    var view = member_call_of(meta, state)
    view.block = META_BLOCK_RFE
    _ = member_call(meta.member, MEMBER_OP_EXPORT, view)
    var words = List[Float64](length=view.dst.rows * view.dst.cols, fill=0.0)
    var src = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=view.dst.data)
    for i in range(len(words)):  # small-loop(the kind's fitted words)
        words[i] = src[i]
    return words^


def _predict(plan: MemberPlan, state: Int, x: Int, n: Int, d: Int) raises -> List[Int32]:
    """predict in the element type the state answers (int32 or int64 codes), as int32."""
    var call = member_call_of(plan, state)
    call.flags = MEMBER_OP_PREDICT
    _ = member_call(plan.member, MEMBER_OP_OUTPUTS, call)
    var got = List[Int32](length=n, fill=Int32(-1))
    call.x = MemberMatrix(x, n, d)
    if call.out_dtype == MEMBER_I64:
        var wide = List[Int64](length=n, fill=Int64(-1))
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), n, 1, MEMBER_I64)
        _ = member_call(plan.member, MEMBER_OP_PREDICT, call)
        for i in range(n):  # small-loop(fixture: the int64 codes narrowed)
            got[i] = Int32(wide[i])
        _ = len(wide)
    else:
        call.dst = MemberMatrix(Int(got.unsafe_ptr()), n, 1, MEMBER_I32)
        _ = member_call(plan.member, MEMBER_OP_PREDICT, call)
    return got^


def _proba(plan: MemberPlan, state: Int, x: Int, n: Int, d: Int) raises -> List[Float64]:
    var call = member_call_of(plan, state)
    call.flags = MEMBER_OP_PREDICT_PROBA
    _ = member_call(plan.member, MEMBER_OP_OUTPUTS, call)
    var got = List[Float64](length=n * K, fill=-7.0)
    call.x = MemberMatrix(x, n, d)
    if call.out_dtype == MEMBER_F64:
        call.dst = MemberMatrix(Int(got.unsafe_ptr()), n, K, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_PREDICT_PROBA, call)
    else:
        var narrow = List[Float32](length=n * K, fill=0.0)
        call.dst = MemberMatrix(Int(narrow.unsafe_ptr()), n, K, MEMBER_F32)
        _ = member_call(plan.member, MEMBER_OP_PREDICT_PROBA, call)
        for i in range(n * K):  # small-loop(fixture: the float32 answer widened)
            got[i] = Float64(narrow[i])
        _ = len(narrow)
    return got^


def test_rfe_kind_equals_run() raises:
    publish_x_trees_glue("cpu")
    var x = fixture_f32(DIR + "lda-k3-x.f32", N * D)
    var codes = fixture_i32(DIR + "lda-k3-y.i32", N)
    var base = _lda()
    var mask = member_kind_capabilities(base)
    var source = rfe_member_source(MEMBER_FAMILY_DISCRIMINANT, mask, "auto", "")
    var run = ClsRunPlan(_xprep_route().entry, False, False, False, False, False)
    # The RUN, as RFE's door runs it.
    var request = RfeFitPlan(base.copy(), run, True, Int(x.unsafe_ptr()), N, D, Int(codes.unsafe_ptr()), MEMBER_I32)
    request.nsel = NSEL
    request.step = STEP
    request.block = source.block
    request.squared = source.squared
    var want = rfe_fit_run(request)
    # THE RFE KIND, as the boundary's admission mints it.
    var nested: List[MemberPlan] = [base.copy()]
    var masks: List[Int] = [mask]
    var handle = meta_plan_mint_with(nested^, List[Float64](), List[Int](), masks^, _words(source.block, source.squared),
                                     List[List[Int32]](), List[List[Int32]]())
    meta_plan_rfe(handle, run, String(""), List[Float64]())
    var meta = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    meta.route = route
    meta.member = MemberRef(EntryRef(ensemble_member_v2, 0), 0)
    var probe = MemberCall(meta.route)
    meta.member.capabilities = member_call(meta.member, MEMBER_OP_CAPABILITIES, probe)
    meta.family = MEMBER_FAMILY_ENSEMBLE
    meta.params = [Float64(META_RFE_CLASSIFIER), Float64(handle), 0.0, 0.0, 0.0, 0.0]
    require(len(meta.params) == META_PARAM_COUNT, "META parameter count")
    meta.classes = K
    var fitted = member_new(meta)
    var restored = 0
    var nested_copy = 0
    try:
        fitted.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
        fitted.y = MemberMatrix(Int(codes.unsafe_ptr()), N, 1, MEMBER_I32)
        fitted.classes = K
        _ = member_call(meta.member, MEMBER_OP_FIT, fitted)
        var view = member_call_of(meta, fitted.state)
        view.block = META_BLOCK_RFE
        _ = member_call(meta.member, MEMBER_OP_EXPORT, view)
        var words = List[Float64](length=view.dst.rows * view.dst.cols, fill=0.0)
        var src = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=view.dst.data)
        for i in range(len(words)):  # small-loop(the kind's fitted words)
            words[i] = src[i]
        require(len(words) == 2 + 2 * D + NSEL and Int(words[0]) == D and Int(words[1]) == NSEL,
                "RFE kind: META_BLOCK_RFE is [d, k, support, ranking, columns]")
        for j in range(D):  # small-loop(support and ranking, exact)
            require(Float32(words[2 + j]) == want.support[j], "RFE kind: support_[" + String(j) + "]")
            require(Float32(words[2 + D + j]) == want.ranking[j], "RFE kind: ranking_[" + String(j) + "]")
        for j in range(NSEL):  # small-loop(the kept columns)
            require(Int(words[2 + 2 * D + j]) == Int(want.columns[j]), "RFE kind: kept column " + String(j))
        # The RUN's final state on the kept columns: the kind's predictions are its.
        var kept = List[Float32](length=N * NSEL, fill=0.0)
        for i in range(N):  # small-loop(fixture: the kept columns gathered)
            for j in range(NSEL):  # small-loop(fixture)
                kept[i * NSEL + j] = x[i * D + Int(want.columns[j])]
        var want_codes = _predict(want.final_plan, want.final_state, Int(kept.unsafe_ptr()), N, NSEL)
        var got_codes = _predict(meta, fitted.state, Int(x.unsafe_ptr()), N, D)
        for i in range(N):  # small-loop(exact codes)
            require(got_codes[i] == want_codes[i], "RFE kind: predict code " + String(i))
        var want_p = _proba(want.final_plan, want.final_state, Int(kept.unsafe_ptr()), N, NSEL)
        var got_p = _proba(meta, fitted.state, Int(x.unsafe_ptr()), N, D)
        for i in range(N * K):  # small-loop(exact words)
            require(got_p[i] == want_p[i], "RFE kind: predict_proba word " + String(i))
        var block = List[Float32](length=N * NSEL, fill=-7.0)
        var t = member_call_of(meta, fitted.state)
        t.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
        t.dst = MemberMatrix(Int(block.unsafe_ptr()), N, NSEL, MEMBER_F32)
        _ = member_call(meta.member, MEMBER_OP_TRANSFORM, t)
        for i in range(N * NSEL):  # small-loop(exact words)
            require(block[i] == kept[i], "RFE kind: transform word " + String(i))
        # THE FORWARDED EXPORT (lane mf-meta5b; a dotted getter 'estimator_.coef_'): the final estimator's
        # coefficient block through the RFE state equals the RUN's final state's own export, word for word.
        var own = member_call_of(want.final_plan, want.final_state)
        own.block = MEMBER_BLOCK_COEF
        _ = member_call(want.final_plan.member, MEMBER_OP_EXPORT, own)
        var hops: List[Float64] = [0.0, Float64(MEMBER_BLOCK_COEF)]
        var fwd = member_call_of(meta, fitted.state)
        fwd.block = META_BLOCK_FORWARD
        fwd.params = Int(hops.unsafe_ptr())
        fwd.param_count = len(hops)
        _ = member_call(meta.member, MEMBER_OP_EXPORT, fwd)
        _ = len(hops)
        require(fwd.dst.rows == own.dst.rows and fwd.dst.cols == own.dst.cols and fwd.dst.cols == NSEL,
                "RFE kind: the forwarded coef_ block has the final estimator's shape")
        var a = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=fwd.dst.data)
        var e = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=own.dst.data)
        for i in range(fwd.dst.rows * fwd.dst.cols):  # small-loop(exact words)
            require(a[i] == e[i], "RFE kind: forwarded coef_ word " + String(i))
        # A path step the state does not hold is refused in the ABI's words.
        var bad: List[Float64] = [3.0, Float64(MEMBER_BLOCK_COEF)]
        var wrong = member_call_of(meta, fitted.state)
        wrong.block = META_BLOCK_FORWARD
        wrong.params = Int(bad.unsafe_ptr())
        wrong.param_count = len(bad)
        var refused = False
        try:
            _ = member_call(meta.member, MEMBER_OP_EXPORT, wrong)
        except error:
            refused = "no nested member 3" in String(error)
        _ = len(bad)
        require(refused, "RFE kind: a forwarded export past its nested members is refused")
        # THE ROUND TRIP: the final estimator's state handed to a new RFE state with the fitted words.
        var nested_view = member_call_of(meta, fitted.state)
        nested_view.block = META_BLOCK_NESTED
        _ = member_call(meta.member, MEMBER_OP_EXPORT, nested_view)
        var final_handle = Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=nested_view.dst.data)[0])
        # A second final state for the new RFE state to own (the first stays the fitted one's).
        var final_plan = want.final_plan.copy()
        var second = member_new(final_plan)
        second.x = MemberMatrix(Int(kept.unsafe_ptr()), N, NSEL)
        second.y = MemberMatrix(Int(codes.unsafe_ptr()), N, 1, MEMBER_I32)
        second.classes = K
        _ = member_call(final_plan.member, MEMBER_OP_FIT, second)
        nested_copy = second.state
        require(final_handle != 0, "RFE kind: no final estimator state")
        var fresh = member_new(meta)
        restored = fresh.state
        var handles: List[Int64] = [Int64(nested_copy)]
        var imp = member_call_of(meta, restored)
        imp.block = META_BLOCK_NESTED
        imp.x = MemberMatrix(Int(handles.unsafe_ptr()), 1, 1, MEMBER_I64)
        _ = member_call(meta.member, MEMBER_OP_IMPORT, imp)
        nested_copy = 0  # the restored state owns it now
        imp.block = META_BLOCK_RFE
        imp.x = MemberMatrix(Int(words.unsafe_ptr()), 1, len(words), MEMBER_F64)
        _ = member_call(meta.member, MEMBER_OP_IMPORT, imp)
        imp.block = 0
        imp.flags = MEMBER_IMPORT_DONE
        imp.classes = K
        _ = member_call(meta.member, MEMBER_OP_IMPORT, imp)
        var again = _predict(meta, restored, Int(x.unsafe_ptr()), N, D)
        for i in range(N):  # small-loop(exact codes)
            require(again[i] == want_codes[i], "RFE kind round trip: predict code " + String(i))
        _ = len(handles)
        _ = len(kept)
        _ = len(block)
        _ = len(words)
    finally:
        member_free(meta, fitted)
        if restored != 0:
            var r = MemberCall(meta.route, restored)
            member_free(meta, r)
        if nested_copy != 0:
            var c = MemberCall(want.final_plan.route, nested_copy)
            member_free(want.final_plan, c)
        var w = MemberCall(want.final_plan.route, want.final_state)
        member_free(want.final_plan, w)
        meta_plan_release(handle)
    _ = len(x)
    _ = len(codes)


def test_rfe_kind_over_meta_base() raises:
    publish_x_trees_glue("cpu")
    var x = fixture_f32(DIR + "lda-k3-x.f32", N * D)
    var codes = fixture_i32(DIR + "lda-k3-y.i32", N)
    var base = _lda()
    var mask = member_kind_capabilities(base)
    var source = rfe_member_source(MEMBER_FAMILY_DISCRIMINANT, mask, "auto", "")
    var run = ClsRunPlan(_xprep_route().entry, False, False, False, False, False)
    # The leaf RUN (RFE(LDA) with 'auto'): the support and ranking the forwarded path must reproduce.
    var request = RfeFitPlan(base.copy(), run, True, Int(x.unsafe_ptr()), N, D, Int(codes.unsafe_ptr()), MEMBER_I32)
    request.nsel = NSEL
    request.step = STEP
    request.block = source.block
    request.squared = source.squared
    var want = rfe_fit_run(request)
    # The inner RFE kind (n_features_to_select 8: no round; estimator_ is LDA on every given column).
    var inner_nested: List[MemberPlan] = [base.copy()]
    var inner_masks: List[Int] = [mask]
    var inner_handle = meta_plan_mint_with(inner_nested^, List[Float64](), List[Int](), inner_masks^,
                                           _words_n(D, source.block, source.squared), List[List[Int32]](),
                                           List[List[Int32]]())
    meta_plan_rfe(inner_handle, run, String(""), List[Float64]())
    var inner = _rfe_meta(inner_handle)
    var inner_mask = member_kind_capabilities(inner)
    # The outer RFE kind over it, as `_admit_rfe` mints it for 'estimator_.coef_'.
    var hops: List[Float64] = [0.0, Float64(MEMBER_BLOCK_COEF)]
    var outer_nested: List[MemberPlan] = [inner.copy()]
    var outer_masks: List[Int] = [inner_mask]
    var outer_handle = meta_plan_mint_with(outer_nested^, List[Float64](), List[Int](), outer_masks^,
                                           _words(META_BLOCK_FORWARD, True), List[List[Int32]](),
                                           List[List[Int32]]())
    meta_plan_rfe(outer_handle, run, String(""), hops.copy())
    var outer = _rfe_meta(outer_handle)
    # The RUN over the META base with the same hops (RFE's door's `_rfe_native` form).
    var door = RfeFitPlan(inner.copy(), run, True, Int(x.unsafe_ptr()), N, D, Int(codes.unsafe_ptr()), MEMBER_I32)
    door.nsel = NSEL
    door.step = STEP
    door.block = META_BLOCK_FORWARD
    door.squared = True
    door.hops = hops.copy()
    var via_door = rfe_fit_run(door)
    var fitted = member_new(outer)
    try:
        fitted.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
        fitted.y = MemberMatrix(Int(codes.unsafe_ptr()), N, 1, MEMBER_I32)
        fitted.classes = K
        _ = member_call(outer.member, MEMBER_OP_FIT, fitted)
        var words = _rfe_kind_words(outer, fitted.state)
        require(len(words) == 2 + 2 * D + NSEL and Int(words[0]) == D and Int(words[1]) == NSEL,
                "RFE kind over META: META_BLOCK_RFE is [d, k, support, ranking, columns]")
        for j in range(D):  # small-loop(support and ranking, exact)
            require(Float32(words[2 + j]) == want.support[j], "RFE kind over META: support_[" + String(j) + "]")
            require(Float32(words[2 + D + j]) == want.ranking[j], "RFE kind over META: ranking_[" + String(j) + "]")
            require(via_door.support[j] == want.support[j], "RFE RUN over META: support_[" + String(j) + "]")
            require(via_door.ranking[j] == want.ranking[j], "RFE RUN over META: ranking_[" + String(j) + "]")
        for j in range(NSEL):  # small-loop(the kept columns)
            require(Int(words[2 + 2 * D + j]) == Int(want.columns[j]), "RFE kind over META: kept column " + String(j))
    finally:
        member_free(outer, fitted)
        var w = MemberCall(want.final_plan.route, want.final_state)
        member_free(want.final_plan, w)
        var v = MemberCall(via_door.final_plan.route, via_door.final_state)
        member_free(via_door.final_plan, v)
        meta_plan_release(outer_handle)
        meta_plan_release(inner_handle)
    _ = len(hops)
    _ = len(x)
    _ = len(codes)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
