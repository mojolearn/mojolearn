# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Class weights of classifier MEMBERS (core/member_class_weight.mojo, lane mf-adm3, 2026-11-23), in a process
with no Python.

The doors weight a classifier's fit with per-row weights (LogisticRegression, LinearSVC: sample weights, or
ones, times each row's class weight, bindings/logistic_native_door.mojo `_row_weights`,
bindings/svc_linear_boundary_unpack.mojo `_qn_row_weights`; RidgeClassifier: the x_linear class step's rows,
bindings/x_linear_sgd_unpack.mojo `xl_fit_ridge_classifier_unpack`). A member now takes class_weight as a tail of
its NEW words and makes the same weights in its FIT. The oracle is exact and needs no fixture: the same member
fitted WITHOUT the tail on the door's row weights, computed here by hand from the door's formulas
(scikit-learn's balanced n / (k * count_c) in binary64, narrowed to float32 once, times the row's weight in
float32), must give the same fitted words, bit for bit.

Cases (one row each, run across the host cores by `check_cases`):
  0 counts: `member_cw_counts` over 1000 int32, int64 and float32 codes equals a serial count; a code outside
    the classes is refused in the doors' words.  Fail-proof: drop the per-task sum (`counts[c] = counts[c] +`)
    or count into `pp[c]` instead of `pp[worker * k + c]`.
  1 balanced rows: `member_cw_rows` equals the hand rows, with and without sample weights.  Fail-proof: drop the
    `s *` product or narrow the ratio in float64 order (`Float32(n) / Float32(k * count)`).
  2 every class weight 1 (a dict of ones): the sample weights' own address, or n ones.  Fail-proof: return a copy.
  3 refusals: a pending dict (no labels) and an absent class raise their words; a K that is not the fit's class
    count is refused.  Fail-proof: drop the `params[start + 2]` test in `member_cw_dict_ready`.
  4 the fold subset: a 3-class table narrowed to classes {0, 2}: two entries, their weights; a named class 1 the
    fold lacks marks the tail absent, an unnamed one does not.  Fail-proof: skip the absent marking.
  5 the keyed tail: keys {1.0: 2.0, 3.0: 0.5} over a step's classes [1, 2, 3] give weights [2, 1, 0.5] and named
    [1, 0, 1]; a key 4.0 no class holds marks it absent.  Fail-proof: compare the key with Float32(label).
  6 the float32 table: `member_cw_table_f32` narrows each weight once.
  7 LogisticRegression member: balanced tail == explicit door rows, without and with sample weights; a dict tail
    == explicit rows.  Fail-proof: pass `sample` instead of the rows address from `member_cw_rows` in
    core/logistic_member.mojo `_fit`.
  8 LinearSVC member: the same two oracles.  Fail-proof: the same edit in core/linear_svm_member.mojo `_fit`.
  9 RidgeClassifier member (x_linear kind 9): a dict tail == the plain weighted fit on the class step's rows
    (sample weight x float32 class weight).  Fail-proof: drop `has_sw = 1` in the class-weight branch of
    core/xlinear_member.mojo `_cls_fit`.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.linear_svc_native_api import LINEAR_SVC_NATIVE_COUNT
from core.linear_svm_member import linear_svm_member_v2
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_EXPORT, MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT,
    MEMBER_F32, MEMBER_I32, MEMBER_I64,
)
from core.member_class_weight import (
    MEMBER_CW_BALANCED, MEMBER_CW_DICT, MEMBER_CW_KEYED, MEMBER_CW_HEAD, member_cw_counts, member_cw_rows,
    member_cw_balanced_words, member_cw_pending_words, member_cw_table_words, member_cw_keyed_words,
    member_cw_subset, member_cw_from_values, member_cw_table_f32, member_cw_dict_ready,
)
from core.member_families import MEMBER_FAMILY_LOGISTIC, MEMBER_FAMILY_LINEAR_SVM, MEMBER_FAMILY_XLINEAR
from core.xlinear_member import xlinear_member_v2
from glm.host.linear_svc_native_host import linear_svc_host_v1
from glm.host.logistic_native_host import qn_host_native_v1
from x_linear.ops_entry_host import x_linear_ops_host_v1

comptime N = 60
comptime D = 3
comptime CASES = 10
comptime ABSENT = "class_weight contains labels absent from y"


def _x() -> List[Float32]:
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        for j in range(D):  # small-loop(fixture)
            x[i * D + j] = Float32((i * 7 + j * 5) % 11) * 0.25 - 1.0 + Float32(i % 3) * 0.5 * Float32(j + 1)
    return x^


def _codes(k: Int) -> List[Int32]:
    """Unbalanced codes: class c holds about (c + 1) parts of the rows."""
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture)
        var r = i % (k * (k + 1) // 2)
        var c = 0
        var edge = 1
        while r >= edge:  # small-loop(fixture)
            c += 1
            edge += c + 1
        y[i] = Int32(c)
    return y^


def _sample() -> List[Float32]:
    var w = List[Float32](length=N, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        w[i] = Float32(0.5) + Float32(i % 4) * Float32(0.25)
    return w^


def _hand_balanced(codes: List[Int32], k: Int) -> List[Float32]:
    """The doors' 'balanced' class words: n / (k * count_c) in binary64, narrowed to float32 once."""
    var counts = List[Int](length=k, fill=0)
    for i in range(len(codes)):  # small-loop(oracle)
        counts[Int(codes[i])] += 1
    var words = List[Float32](length=k, fill=0.0)
    for c in range(k):  # small-loop(oracle)
        words[c] = Float32(Float64(len(codes)) / Float64(k * counts[c]))
    return words^


def _hand_rows(codes: List[Int32], words: List[Float32], sample: List[Float32]) -> List[Float32]:
    """Each row's weight (its sample weight, or 1) times its class word, one float32 product."""
    var rows = List[Float32](length=len(codes), fill=0.0)
    for i in range(len(codes)):  # small-loop(oracle)
        var s = sample[i] if len(sample) > 0 else Float32(1.0)
        rows[i] = s * words[Int(codes[i])]
    return rows^


def _same(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(oracle: every word)
        require(a[i] == b[i], what + " word " + String(i) + ": " + String(a[i]) + " != " + String(b[i]))


def _raises(what: String, words: String, params: List[Float64], start: Int, classes: Int) raises:
    var said = String("")
    try:
        member_cw_dict_ready(params, start, classes, ABSENT)
    except e:
        said = String(e)
    require(words in said, what + ": expected '" + words + "', got '" + said + "'")


def check_counts() raises:
    var n = 1000
    var k = 7
    var i32 = List[Int32](length=n, fill=Int32(0))
    var i64 = List[Int64](length=n, fill=Int64(0))
    var f32 = List[Float32](length=n, fill=0.0)
    var serial = List[Int64](length=k, fill=Int64(0))
    for i in range(n):  # small-loop(fixture)
        var c = (i * 13 + i // 7) % k
        i32[i] = Int32(c)
        i64[i] = Int64(c)
        f32[i] = Float32(c)
        serial[c] += 1
    var a = member_cw_counts(Int(i32.unsafe_ptr()), MEMBER_I32, n, k)
    var b = member_cw_counts(Int(i64.unsafe_ptr()), MEMBER_I64, n, k)
    var c = member_cw_counts(Int(f32.unsafe_ptr()), MEMBER_F32, n, k)
    for j in range(k):  # small-loop(oracle)
        require(a[j] == serial[j] and b[j] == serial[j] and c[j] == serial[j], "count of class " + String(j))
    i32[n - 1] = Int32(k)
    var said = String("")
    try:
        _ = member_cw_counts(Int(i32.unsafe_ptr()), MEMBER_I32, n, k)
    except e:
        said = String(e)
    require("a class code outside [0, n_classes)" in said, "the out-of-range code refusal: " + said)
    _ = i32^
    _ = i64^
    _ = f32^


def check_balanced_rows() raises:
    var k = 3
    var codes = _codes(k)
    var hand = _hand_balanced(codes, k)
    var params: List[Float64] = [0.0]
    member_cw_balanced_words(params)
    var rows = List[Float32]()
    var at = member_cw_rows(params, 1, Int(codes.unsafe_ptr()), MEMBER_I32, N, k, 0, ABSENT, rows)
    require(at == Int(rows.unsafe_ptr()), "balanced rows: the rows' own address")
    _same(rows, _hand_rows(codes, hand, List[Float32]()), "balanced rows")
    var sample = _sample()
    var weighted = List[Float32]()
    _ = member_cw_rows(params, 1, Int(codes.unsafe_ptr()), MEMBER_I32, N, k, Int(sample.unsafe_ptr()), ABSENT,
        weighted)
    _same(weighted, _hand_rows(codes, hand, sample), "balanced rows x sample weights")
    _ = codes^
    _ = sample^


def check_all_one() raises:
    var k = 2
    var codes = _codes(k)
    var params: List[Float64] = [0.0]
    member_cw_pending_words(params)
    var values: List[Float64] = [1.0, 1.0]
    var named: List[Bool] = [True, False]
    member_cw_table_words(params, 1, values, named)
    var sample = _sample()
    var rows = List[Float32]()
    var at = member_cw_rows(params, 1, Int(codes.unsafe_ptr()), MEMBER_I32, N, k, Int(sample.unsafe_ptr()), ABSENT,
        rows)
    require(at == Int(sample.unsafe_ptr()), "every class weight 1: the sample weights themselves")
    var ones = List[Float32]()
    var at2 = member_cw_rows(params, 1, Int(codes.unsafe_ptr()), MEMBER_I32, N, k, 0, ABSENT, ones)
    require(at2 == Int(ones.unsafe_ptr()) and len(ones) == N, "every class weight 1, no sample weights: n ones")
    for i in range(N):  # small-loop(oracle)
        require(ones[i] == Float32(1.0), "a one")
    _ = codes^
    _ = sample^


def check_refusals() raises:
    var pending: List[Float64] = [0.0]
    member_cw_pending_words(pending)
    _raises("pending", "hands over its labels", pending, 1, 2)
    var table: List[Float64] = [0.0]
    member_cw_pending_words(table)
    var values: List[Float64] = [2.0, 1.0]
    var named: List[Bool] = [True, True]
    member_cw_table_words(table, 1, values, named)
    _raises("class count", "entries for 3 classes", table, 1, 3)
    table[1 + 2] = 1.0  # absent
    _raises("absent", ABSENT, table, 1, 2)


def _plan_logistic(k: Int) raises -> MemberPlan:
    var plan = MemberPlan()
    plan.family = MEMBER_FAMILY_LOGISTIC
    plan.kind = 0
    plan.route = logistic_route_pack(LogisticRoute(DEVICE_CPU, Int(GLOBAL_NUMERIC_MODE), 0, False, False,
                                                   QnNativeDriver(qn_host_native_v1)))
    plan.member = MemberRef(EntryRef(logistic_member_v2, 0), 0)
    plan.params = [0.0, 1.0, 0.0, 1.0, 100.0, 20.0, 5.0, 1.0, 1e-4, 1e-6]
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.classes = k
    return plan^


def check_subset() raises:
    var plan = _plan_logistic(3)
    member_cw_pending_words(plan.params)
    var values: List[Float64] = [2.0, 3.0, 0.5]
    var named: List[Bool] = [True, False, True]
    member_cw_table_words(plan.params, 10, values, named)
    var held: List[Int32] = [Int32(0), Int32(2)]
    var narrowed = plan.copy()
    member_cw_subset(narrowed, held)
    require(Int(narrowed.params[11]) == 2 and narrowed.params[12] == 0.0, "two entries, not absent")
    require(narrowed.params[13] == 2.0 and narrowed.params[15] == 0.5, "the held classes' weights")
    var lacking = plan.copy()
    lacking.params[10 + MEMBER_CW_HEAD + 2 * 1 + 1] = 1.0  # class 1 named
    member_cw_subset(lacking, held)
    require(lacking.params[12] == 1.0, "a named class the fold lacks marks the tail absent")


def check_keyed() raises:
    var plan = _plan_logistic(3)
    member_cw_pending_words(plan.params)
    var keys: List[Float64] = [1.0, 3.0]
    var weights: List[Float64] = [2.0, 0.5]
    member_cw_keyed_words(plan.params, 10, keys, weights)
    require(Int(plan.params[10]) == MEMBER_CW_KEYED, "the keyed tail")
    var classes: List[Float32] = [Float32(1.0), Float32(2.0), Float32(3.0)]
    var step = plan.copy()
    member_cw_from_values(step, classes)
    require(Int(step.params[10]) == MEMBER_CW_DICT and Int(step.params[11]) == 3 and step.params[12] == 0.0,
            "the step's table")
    require(step.params[13] == 2.0 and step.params[14] == 1.0, "class 1.0")
    require(step.params[15] == 1.0 and step.params[16] == 0.0, "class 2.0 (not named)")
    require(step.params[17] == 0.5 and step.params[18] == 1.0, "class 3.0")
    var other: List[Float64] = [1.0, 4.0]
    var unmatched = plan.copy()
    member_cw_keyed_words(unmatched.params, 10, other, weights)
    member_cw_from_values(unmatched, classes)
    require(unmatched.params[12] == 1.0, "a key no class holds marks the tail absent")


def check_table_f32() raises:
    var params: List[Float64] = [0.0]
    member_cw_pending_words(params)
    var values: List[Float64] = [0.1, 1.0 / 3.0]
    var named: List[Bool] = [True, True]
    member_cw_table_words(params, 1, values, named)
    var t = member_cw_table_f32(params, 1, 2, ABSENT)
    require(t[0] == Float32(0.1) and t[1] == Float32(1.0 / 3.0), "each weight narrowed once")


def _fit_words(plan: MemberPlan, codes: List[Int32], k: Int, weights: List[Float32]) raises -> List[Float32]:
    """The member's fitted coef then intercept words."""
    var x = _x()
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(codes.unsafe_ptr()), N, 1, MEMBER_I32)
    call.classes = k
    if len(weights) > 0:
        call.weights = MemberMatrix(Int(weights.unsafe_ptr()), N, 1)
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    except e:
        member_free(plan, call)
        raise e^
    var out = List[Float32]()
    var blocks: List[Int] = [MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT]
    for b in range(2):  # small-loop(two blocks)
        var ex = member_call_of(plan, call.state)
        ex.block = blocks[b]
        _ = member_call(plan.member, MEMBER_OP_EXPORT, ex)
        var view = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=ex.dst.data)
        for i in range(ex.dst.rows * ex.dst.cols):  # small-loop(oracle: every fitted word)
            out.append(view[i])
    member_free(plan, call)
    _ = x^
    _ = codes
    _ = weights
    return out^


def _weighted_pair(plain: MemberPlan, start: Int, what: String) raises:
    """`plain` with a balanced tail == `plain` on the hand rows; the same with sample weights; a dict tail ==
    `plain` on its hand rows."""
    var k = 3
    var codes = _codes(k)
    var hand = _hand_balanced(codes, k)
    var balanced = plain.copy()
    member_cw_balanced_words(balanced.params)
    _same(_fit_words(balanced, codes, k, List[Float32]()), _fit_words(plain, codes, k, _hand_rows(codes, hand,
          List[Float32]())), what + " balanced")
    var sample = _sample()
    _same(_fit_words(balanced, codes, k, sample), _fit_words(plain, codes, k, _hand_rows(codes, hand, sample)),
          what + " balanced x sample weights")
    var keyed = plain.copy()
    member_cw_pending_words(keyed.params)
    var values: List[Float64] = [0.25, 4.0, 1.0]
    var named: List[Bool] = [True, True, False]
    member_cw_table_words(keyed.params, start, values, named)
    var words: List[Float32] = [Float32(0.25), Float32(4.0), Float32(1.0)]
    _same(_fit_words(keyed, codes, k, List[Float32]()), _fit_words(plain, codes, k, _hand_rows(codes, words,
          List[Float32]())), what + " dict")
    _ = codes^
    _ = sample^


def check_logistic() raises:
    _weighted_pair(_plan_logistic(3), 10, "LogisticRegression")


def check_linear_svc() raises:
    var plan = MemberPlan()
    plan.family = MEMBER_FAMILY_LINEAR_SVM
    plan.kind = 0
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    route.entry = EntryRef(linear_svc_host_v1, entry_ref_word(route.mode, 0, LINEAR_SVC_NATIVE_COUNT))
    route.flags = 3
    plan.route = route
    plan.member = MemberRef(EntryRef(linear_svm_member_v2, 0), 0)
    plan.params = [0.0, 1.0, 1000.0, 100.0, 5.0, 1.0, 4.0, 0.0, 1.0, 1e-4, 0.0]
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.classes = 3
    _weighted_pair(plan, 11, "LinearSVC")


def check_ridge_classifier() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var plan = MemberPlan()
    plan.family = MEMBER_FAMILY_XLINEAR
    plan.kind = 9
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(x_linear_ops_host_v1, entry_ref_word(mode, 0, 0))
    plan.route = route
    plan.member = MemberRef(EntryRef(xlinear_member_v2, 0), 0)
    plan.params = [9.0, 1.0, 1.0]
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    var k = 3
    plan.classes = k
    var codes = _codes(k)
    var sample = _sample()
    var keyed = plan.copy()
    member_cw_pending_words(keyed.params)
    var values: List[Float64] = [0.25, 4.0, 1.0]
    var named: List[Bool] = [True, True, False]
    member_cw_table_words(keyed.params, 3, values, named)
    var words: List[Float32] = [Float32(0.25), Float32(4.0), Float32(1.0)]
    _same(_fit_words(keyed, codes, k, sample), _fit_words(plan, codes, k, _hand_rows(codes, words, sample)),
          "RidgeClassifier dict x sample weights")
    _ = codes^
    _ = sample^


def test_member_class_weight() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_counts()
        elif which == 1:
            check_balanced_rows()
        elif which == 2:
            check_all_one()
        elif which == 3:
            check_refusals()
        elif which == 4:
            check_subset()
        elif which == 5:
            check_keyed()
        elif which == 6:
            check_table_f32()
        elif which == 7:
            check_logistic()
        elif which == 8:
            check_linear_svc()
        else:
            check_ridge_classifier()
    check_cases("member_class_weight", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
