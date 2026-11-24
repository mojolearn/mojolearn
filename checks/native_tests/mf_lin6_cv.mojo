# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-lin6: the x_linear CV member routes that answered False before (every cv the door's `_xl_cv_splits`
splits is a spec now) and the LinearSVC member's binary codes in an image without the PY2MOJO rows bit, in a
process with no Python. Oracles: the public door's own calls (the same plan functions with the same words, the
door's decision-then-threshold body), or the integer-cv member over the same folds.

 1. LassoCV (x_linear member) over an EMPTY index list (a LISTED spec with no pair): the fit raises the door's
    `_xl_cv_splits` words "mojolearn: cv produced no (train, test) split".
 2. LassoCV over the index list of KFold(3)'s three contiguous folds of 60 rows == the integer cv=3 member: coef,
    scalars and mse_path_ word for word.
 3. KFold(n_splits=1) taken by the strict reader: the spec's sides raise exactly the words of the door's call
    (`msel_plan_kfold(route, n, 1, ...)`, which `msel_kfold_split` makes), and LassoCV over that spec raises them.
 4. TimeSeriesSplit with an unreadable max_train_size (kept as words by the strict reader): a plan with splits
    raises those words; a plan the door refuses first (more folds than rows) raises the plan's own words instead.
 5. RepeatedKFold with n_repeats=0: no split, and LassoCV over it raises "mojolearn: cv produced no (train, test)
    split" (the door returns no pair, then `_xl_cv_splits` refuses).
 6. LinearSVC's binary class codes: the member's QN_NATIVE_BINARY query == the door's OFF-image route
    (QN_NATIVE_DECISION, then the labels threshold strict `> 0`, below 0, above 1: `threshold_labels_i64`, the
    host body of `labels_threshold`), including rows whose score is exactly 0 (code 0 on both).

FAIL-PROOF EDITS (each must make the named case fail): in core/xlinear_member.mojo `_cv_sides` drop the
`sides.splits() == 0` refusal (cases 1 and 5); in bindings/msel_split_spec_run.mojo `msel_spec_sides` LISTED branch
append `spec.listed_test[s]` as the train side (case 2); in `msel_plan_kfold` return before the `k < 2` refusal
(case 3); in the TIME_SERIES branch of `msel_spec_sides` raise `spec.max_train_refusal` before the plan (case 4,
second half) or drop the raise (case 4, first half); in core/classical_host_predict.mojo
`host_qn_predict_binary_into` use `z[i] >= 0` (case 6).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_unresolved, entry_ref_word
from core.msel_entry_table import MselEntryKey
from core.msel_helpers_host import publish_msel_helpers_host
from core.msel_helpers_calls import msel_helpers_ref
from core.base_entry_table import BaseEntryKey
from core.base_helpers_host import publish_base_helpers_host
from core.base_helpers_calls import base_helpers_ref
from core.hotpath_impl_rows import threshold_labels_i64
from x_metrics.descriptor_host import metrics_program_host_v1
from bindings.msel_split_plan_run import MselSeed, MselSplitRoute, msel_plan_kfold, msel_plan_time_series
from bindings.msel_split_spec_run import (
    MselSplitSpec, MselGroupCodes, msel_spec_sides, MSEL_SPEC_KFOLD, MSEL_SPEC_TIME_SERIES, MSEL_SPEC_REPEATED_KFOLD,
    MSEL_SPEC_LISTED, msel_spec_mint, msel_spec_release,
)
from core.logistic_native_api import QnNativeDriver, QnQueryRequest, QN_NATIVE_BINARY, QN_NATIVE_DECISION
from glm.host.logistic_native_host import qn_host_native_v1
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_EXPORT, MEMBER_F32, MEMBER_F64, MEMBER_BLOCK_COEF,
    MEMBER_BLOCK_SCALARS,
)
from core.member_families import MEMBER_FAMILY_XLINEAR
from core.xlinear_member import xlinear_member_v2, XL_MEMBER_BLOCK_MSE_PATH
from x_linear.ops_entry_host import x_linear_ops_host_v1

comptime CASES = 6
comptime N = 60
comptime D = 3
comptime _NO_SPLIT = "mojolearn: cv produced no (train, test) split"


def _route() raises -> MselSplitRoute:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    return MselSplitRoute(msel_helpers_ref(MselEntryKey(mode, 0)),
                          metrics_program_host_v1, False, False)


def _spec(kind: Int) raises -> MselSplitSpec:
    var spec = MselSplitSpec()
    spec.kind = kind
    spec.route = _route()
    spec.rows_helpers = base_helpers_ref(BaseEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))
    return spec^


def _x() -> List[Float32]:
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        for j in range(D):  # small-loop(fixture)
            x[i * D + j] = Float32((i * 7 + j * 5) % 11) * 0.25 - 1.0 + Float32(i % 3) * 0.5 * Float32(j + 1)
    return x^


def _targets(x: List[Float32]) -> List[Float32]:
    var t = List[Float32](length=N, fill=0.0)
    for i in range(N):  # small-loop(fixture: a linear target of two columns)
        t[i] = x[i * D] * 1.5 - x[i * D + 2] * 0.5 + 0.25
    return t^


def _lasso_plan(folds_word: Float64) raises -> MemberPlan:
    """LassoCV(cv=..., n_alphas=5) as the x_linear provider admits it: [11, fit_intercept, max_iter, eps, tol,
    positive, folds, G, explicit, L, l1_ratio 1.0]."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(x_linear_ops_host_v1, entry_ref_word(mode, 0, 0))
    plan.family = MEMBER_FAMILY_XLINEAR
    plan.member = MemberRef(EntryRef(xlinear_member_v2, 0), 0)
    plan.kind = 11
    plan.params = [11.0, 1.0, 1000.0, 1e-3, 1e-4, 0.0, folds_word, 5.0, 0.0, 1.0, 1.0]
    plan.route = route
    var probe = MemberCall(route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    return plan^


def _lasso_words(plan: MemberPlan, x: List[Float32], t: List[Float32]) raises -> List[Float64]:
    """One fit; its COEF, SCALARS and MSE_PATH words in order."""
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(t.unsafe_ptr()), N, 1, MEMBER_F32)
    var out = List[Float64]()
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
        var blocks: List[Int] = [MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, XL_MEMBER_BLOCK_MSE_PATH]
        for b in range(3):  # small-loop(oracle: three blocks)
            var e = member_call_of(plan, call.state)
            e.block = blocks[b]
            e.flags = 0
            _ = member_call(plan.member, MEMBER_OP_EXPORT, e)
            var count = e.dst.rows * e.dst.cols
            out.append(Float64(e.dst.rows))
            out.append(Float64(e.dst.cols))
            for i in range(count):  # small-loop(oracle: one block's words)
                if e.dst.dtype == MEMBER_F64:
                    out.append(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=e.dst.data)[i])
                else:
                    out.append(Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=e.dst.data)[i]))
    finally:
        member_free(plan, call)
    _ = len(x)
    _ = len(t)
    return out^


def _lasso_refusal(spec: MselSplitSpec) raises -> String:
    """LassoCV over `spec`: the words its fit raises ("" when it fits)."""
    var x = _x()
    var t = _targets(x)
    var handle = msel_spec_mint(spec)
    var words = String("")
    try:
        _ = _lasso_words(_lasso_plan(-Float64(handle)), x, t)
    except problem:
        words = String(problem)
    msel_spec_release(handle)
    _ = len(x)
    _ = len(t)
    return words^


def _same_words(a: List[Float64], b: List[Float64], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(oracle: every word)
        require(a[i] == b[i], what + " word " + String(i) + ": " + String(a[i]) + " != " + String(b[i]))


def _refused(problem: String, words: String, what: String) raises:
    require(problem == words, what + ": got '" + problem + "', expected '" + words + "'")


# ------------------------------------------------------------ the cases


def check_listed_empty() raises:
    var spec = _spec(MSEL_SPEC_LISTED)
    _refused(_lasso_refusal(spec), _NO_SPLIT, "LassoCV over an empty index list")


def check_listed_folds() raises:
    var spec = _spec(MSEL_SPEC_LISTED)
    for f in range(3):  # small-loop(oracle: KFold(3)'s contiguous folds of 60 rows, 20 each)
        var tr = List[Int64]()
        var te = List[Int64]()
        for i in range(N):  # small-loop(oracle)
            if i // 20 == f:
                te.append(Int64(i))
            else:
                tr.append(Int64(i))
        spec.listed_train.append(tr^)
        spec.listed_test.append(te^)
    var x = _x()
    var t = _targets(x)
    var handle = msel_spec_mint(spec)
    var got = List[Float64]()
    try:
        got = _lasso_words(_lasso_plan(-Float64(handle)), x, t)
    finally:
        msel_spec_release(handle)
    var want = _lasso_words(_lasso_plan(3.0), x, t)
    _same_words(got, want, "LassoCV over KFold(3)'s index list vs cv=3")
    _ = len(x)
    _ = len(t)


def check_kfold_one() raises:
    var route = _route()
    var door = String("")
    try:
        _ = msel_plan_kfold(route, 7, 1, False, MselSeed(0, False), False)
    except problem:
        door = String(problem)
    require(door.byte_length() > 0, "the door's KFold(1) plan refuses")
    var spec = _spec(MSEL_SPEC_KFOLD)
    spec.k = 1
    var got = String("")
    try:
        _ = msel_spec_sides(spec, 7, 0, 0, MselGroupCodes())
    except problem:
        got = String(problem)
    _refused(got, door, "KFold(1) spec sides vs the door's plan")
    var door60 = String("")
    try:
        _ = msel_plan_kfold(route, N, 1, False, MselSeed(0, False), False)
    except problem:
        door60 = String(problem)
    _refused(_lasso_refusal(spec), door60, "LassoCV over KFold(1)")


def check_time_series_unread() raises:
    var spec = _spec(MSEL_SPEC_TIME_SERIES)
    spec.k = 2
    spec.max_train_unread = True
    spec.max_train_refusal = String("TypeError: unreadable max_train_size words")
    var got = String("")
    try:
        _ = msel_spec_sides(spec, 9, 0, 0, MselGroupCodes())
    except problem:
        got = String(problem)
    _refused(got, "TypeError: unreadable max_train_size words", "TimeSeriesSplit with splits")
    spec.k = 5
    var door = String("")
    try:
        _ = msel_plan_time_series(2, 5, 0, False, 0, 0)
    except problem:
        door = String(problem)
    require(door.byte_length() > 0, "the door's TimeSeriesSplit(5) plan over 2 rows refuses")
    var first = String("")
    try:
        _ = msel_spec_sides(spec, 2, 0, 0, MselGroupCodes())
    except problem:
        first = String(problem)
    _refused(first, door, "TimeSeriesSplit: the plan's refusal comes first")


def check_repeated_zero() raises:
    var spec = _spec(MSEL_SPEC_REPEATED_KFOLD)
    spec.k = 0
    spec.repeats = 0
    spec.seed = MselSeed(11, False)
    var sides = msel_spec_sides(spec, N, 0, 0, MselGroupCodes())
    require(sides.splits() == 0, "RepeatedKFold(n_repeats=0): no split")
    _refused(_lasso_refusal(spec), _NO_SPLIT, "LassoCV over RepeatedKFold(n_repeats=0)")


def check_linear_svc_binary() raises:
    comptime R = 7
    comptime F = 2
    # rows 0 and 4 score exactly 0 (zero row, zero intercept): code 0 on both routes
    var x: List[Float32] = [0.0, 0.0, 1.0, -1.0, -2.0, 0.5, 0.25, 0.75, 0.0, 0.0, -0.5, 3.0, 2.0, -2.0]
    var w: List[Float32] = [0.75, -1.25, 0.0]  # two coefficients, then the intercept word
    var driver = QnNativeDriver(qn_host_native_v1)
    var codes = List[Int64](length=R, fill=Int64(-9))
    driver.query(QN_NATIVE_BINARY, QnQueryRequest(Int(x.unsafe_ptr()), Int(w.unsafe_ptr()), Int(codes.unsafe_ptr()),
        0, R, F, 1, True, False))
    var scores = List[Float32](length=R, fill=Float32(-9.0))
    driver.query(QN_NATIVE_DECISION, QnQueryRequest(Int(x.unsafe_ptr()), Int(w.unsafe_ptr()),
        Int(scores.unsafe_ptr()), 0, R, F, 1, True, False))
    var door = List[Int64](length=R, fill=Int64(-9))
    _ = threshold_labels_i64(Int(scores.unsafe_ptr()), 0, R, 0.0, True, Int64(0), Int64(1), Int(door.unsafe_ptr()))
    for i in range(R):  # small-loop(oracle: every row)
        require(codes[i] == door[i], "LinearSVC binary row " + String(i) + ": member " + String(codes[i])
                + ", door " + String(door[i]))
    require(codes[0] == 0 and codes[4] == 0, "a zero score is code 0 (strict > 0)")
    _ = len(x)
    _ = len(w)
    _ = len(scores)
    _ = len(door)
    _ = len(codes)


def test_mf_lin6_cv() raises:
    # The publications the host images make at module init (a repeat publishes the same pointer).
    publish_msel_helpers_host()
    publish_base_helpers_host()

    def run_case(which: Int) raises:
        if which == 0:
            check_listed_empty()
        elif which == 1:
            check_listed_folds()
        elif which == 2:
            check_kfold_one()
        elif which == 3:
            check_time_series_unread()
        elif which == 4:
            check_repeated_zero()
        else:
            check_linear_svc_binary()

    check_cases("mf_lin6_cv", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
