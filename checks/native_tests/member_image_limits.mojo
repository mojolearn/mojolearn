# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The member image limits lane mf-img3 closed (2026-11-23), in a process with no Python.

Cases (one row each, run across the host cores by `check_cases`):
  0 the host QN class-code query (glm/host/logistic_native_host.mojo QN_NATIVE_MULTICLASS, new): a
    LinearSVC member on the host LinearSVC entry with the rows bit only (`route.flags` 1, what the admit now
    takes), k = 3: its predicted codes equal the first-max argmax of its own decision scores, computed here
    serially (the door's former host route: the decision query, then `labels_argmax_rows`). Fail-proof: pass
    `cols - 1` for `cols` to `argmax_rows_f32` in `_host_argmax_rows` (rows whose top class is 2 differ), or
    return before `_host_argmax_rows` (every code 0).
  1 the same for a LogisticRegression member on the host QN entry, k = 3 (core/logistic_member.mojo `_codes`
    no longer has its decision + argmax fallback). Fail-proof: as case 0.
  2 the Gbdt feature-list tail (core/gbdt_member.mojo `_feature_tail`): `[kind, config..., cat..., hot...,
    C, H]` gives the config's word count and both lists; a tail longer than the words is refused. Fail-proof:
    read C from the last word.
  3 the one-hot / categorical flags (`_feature_flags`): bit 0 per cat_features index, bit 1 per
    one_hot_features index, one zero word with neither list, and the fit door's range refusal word for word
    ("mojolearn: cat_features index 5 out of range for 4 features"). Fail-proof: test `at > cols`.
  4 the group sizes (`_group_sizes` over `call.groups`): int64 ids [7, 7, 3, 3, 3, 9] give runs [2, 3, 1];
    int32 ids the same; ids [1, 2, 1] raise the fit door's split-group words ("mojolearn: group Ids are not
    consecutive: the rows of group 1 are split into more than one run (row 2); ..."); a length that is not the
    rows' raises `gbdt_check_group_rows`'s words. Fail-proof: reset `last` only on a new run.
  5 the category map (`_category_fit`, `_category_apply`): numeric categories of a column-major block keyed
    from their Float32 words (5.0 and 5 are one category, NaN is one category): the fitted codes equal the
    ranks of the distinct keys, the stored map names the column, and a query value the fit never saw takes the
    unseen code k. The oracle is the door's own body sequence (gbdt/data/category_map.mojo, as
    bindings/gbdt_category_unpack.mojo `gbdt_category_fit` / `gbdt_category_apply` run it). Fail-proof: key
    the column with `category_key_f64` of a widened value in `_category_fit`.
"""
from std.math import nan
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.host_helpers_impl import argmax_rows_f32
from core.linear_svc_native_api import LINEAR_SVC_NATIVE_COUNT
from core.linear_svm_member import linear_svm_member_v2
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_DECISION,
    MEMBER_F32, MEMBER_I32, MEMBER_I64,
)
from core.member_families import MEMBER_FAMILY_LOGISTIC, MEMBER_FAMILY_LINEAR_SVM
from core.gbdt_member import (
    GbdtMemberSlot, _feature_tail, _feature_flags, _group_sizes, _category_fit, _category_apply,
)
from gbdt.data.category_map import (
    CategoryMaps, CategoryKeyPtr, CategoryF32Ptr, category_keys_from_f32, category_block_columns,
    category_block_feature,
)
from glm.host.linear_svc_native_host import linear_svc_host_v1
from glm.host.logistic_native_host import qn_host_native_v1

comptime N = 60
comptime D = 3
comptime K = 3
comptime CASES = 6
comptime _SlotP = MutPointer[GbdtMemberSlot, MutUntrackedOrigin]


def _x() -> List[Float32]:
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        for j in range(D):  # small-loop(fixture)
            x[i * D + j] = Float32((i * 7 + j * 5) % 11) * 0.25 - 1.0 + Float32(i % 3) * 0.5 * Float32(j + 1)
    return x^


def _codes() -> List[Int32]:
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture)
        y[i] = Int32(i % K)
    return y^


def _plan_linear_svc() raises -> MemberPlan:
    var plan = MemberPlan()
    plan.family = MEMBER_FAMILY_LINEAR_SVM
    plan.kind = 0
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    route.entry = EntryRef(linear_svc_host_v1, entry_ref_word(route.mode, 0, LINEAR_SVC_NATIVE_COUNT))
    route.flags = 1  # LINEAR_SVC_ROUTE_ROWS only: the admit no longer asks for the multiclass bit
    plan.route = route
    plan.member = MemberRef(EntryRef(linear_svm_member_v2, 0), 0)
    plan.params = [0.0, 1.0, 1000.0, 100.0, 5.0, 1.0, 4.0, 0.0, 1.0, 1e-4, 0.0]
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.classes = K
    return plan^


def _plan_logistic() raises -> MemberPlan:
    var plan = MemberPlan()
    plan.family = MEMBER_FAMILY_LOGISTIC
    plan.kind = 0
    # multiclass False: the member no longer reads the word (every image answers the class-code query).
    plan.route = logistic_route_pack(LogisticRoute(DEVICE_CPU, Int(GLOBAL_NUMERIC_MODE), 0, False, False,
                                                   QnNativeDriver(qn_host_native_v1)))
    plan.member = MemberRef(EntryRef(logistic_member_v2, 0), 0)
    plan.params = [0.0, 1.0, 0.0, 1.0, 100.0, 20.0, 5.0, 1.0, 1e-4, 1e-6]
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.classes = K
    return plan^


def _codes_equal_argmax(plan: MemberPlan, wide_codes: Bool, what: String) raises:
    """Fit on the fixture; predict codes; decision scores; the codes must be the serial first-max argmax."""
    var x = _x()
    var y = _codes()
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(y.unsafe_ptr()), N, 1, MEMBER_I32)
    call.classes = K
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    except e:
        member_free(plan, call)
        raise e^
    var scores = List[Float32](length=N * K, fill=0.0)
    var dec = member_call_of(plan, call.state)
    dec.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    dec.dst = MemberMatrix(Int(scores.unsafe_ptr()), N, K)
    _ = member_call(plan.member, MEMBER_OP_DECISION, dec)
    var expected = List[Int64](length=N, fill=0)
    argmax_rows_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(scores.unsafe_ptr())), N, K,
                    MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(expected.unsafe_ptr())))
    var got = List[Int64](length=N, fill=0)
    var narrow = List[Int32](length=N, fill=Int32(0))
    var pred = member_call_of(plan, call.state)
    pred.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    if wide_codes:
        pred.dst = MemberMatrix(Int(got.unsafe_ptr()), N, 1, MEMBER_I64)
    else:
        pred.dst = MemberMatrix(Int(narrow.unsafe_ptr()), N, 1, MEMBER_I32)
    _ = member_call(plan.member, MEMBER_OP_PREDICT, pred)
    var top_two = 0
    for i in range(N):  # small-loop(oracle: every row)
        var code = got[i] if wide_codes else Int64(narrow[i])
        require(code == expected[i], what + " row " + String(i) + ": code " + String(code) + " != argmax "
                + String(expected[i]))
        if expected[i] == 2:
            top_two += 1
    require(top_two > 0, what + ": the fixture must have rows whose top class is 2")
    member_free(plan, call)
    _ = x^
    _ = y^
    _ = scores^


def check_linear_svc_codes() raises:
    _codes_equal_argmax(_plan_linear_svc(), False, "LinearSVC host class codes")


def check_logistic_codes() raises:
    _codes_equal_argmax(_plan_logistic(), True, "LogisticRegression host class codes")


def check_feature_tail() raises:
    # kind, three config words, cat [2, 0], hot [1], C = 2, H = 1
    var words: List[Float64] = [1.0, 9.0, 8.0, 7.0, 2.0, 0.0, 1.0, 2.0, 1.0]
    var cat = List[Int]()
    var hot = List[Int]()
    var head = _feature_tail(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(words.unsafe_ptr())),
                             len(words), cat, hot)
    require(head == 4, "the config ends before the tail, got " + String(head))
    require(len(cat) == 2 and cat[0] == 2 and cat[1] == 0, "cat_features in the estimator's order")
    require(len(hot) == 1 and hot[0] == 1, "one_hot_features")
    var bad: List[Float64] = [1.0, 9.0, 5.0, 0.0]
    var said = String("")
    try:
        var c2 = List[Int]()
        var h2 = List[Int]()
        _ = _feature_tail(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(bad.unsafe_ptr())),
                          len(bad), c2, h2)
    except e:
        said = String(e)
    require("feature-list tail does not fit" in said, "a tail longer than the words is refused, got '" + said + "'")
    _ = words^
    _ = bad^


def check_feature_flags() raises:
    var slot = GbdtMemberSlot()
    var p = _SlotP(unsafe_from_address=Int(UnsafePointer(to=slot)))
    var none = _feature_flags(p, 4)
    require(len(none) == 1 and none[0] == 0, "neither list: one zero word")
    p[].cat_given = [2, 0]
    p[].hot_given = [2, 3]
    var flags = _feature_flags(p, 4)
    require(len(flags) == 4, "one word per feature")
    require(flags[0] == 1 and flags[1] == 0 and flags[2] == 3 and flags[3] == 2, "bit 0 cat, bit 1 one-hot")
    p[].cat_given = [5]
    var said = String("")
    try:
        _ = _feature_flags(p, 4)
    except e:
        said = String(e)
    require(said == "mojolearn: cat_features index 5 out of range for 4 features",
            "the door's range refusal, got '" + said + "'")
    _ = slot^


def check_group_sizes() raises:
    var ids64: List[Int64] = [7, 7, 3, 3, 3, 9]
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    call.groups = MemberMatrix(Int(ids64.unsafe_ptr()), 6, 1, MEMBER_I64)
    var cp = MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=call)))
    var sizes = _group_sizes(cp, 6)
    require(len(sizes) == 3 and sizes[0] == 2 and sizes[1] == 3 and sizes[2] == 1, "int64 runs [2, 3, 1]")
    var ids32: List[Int32] = [7, 7, 3, 3, 3, 9]
    cp[].groups = MemberMatrix(Int(ids32.unsafe_ptr()), 6, 1, MEMBER_I32)
    var sizes32 = _group_sizes(cp, 6)
    require(len(sizes32) == 3 and sizes32[1] == 3, "int32 runs [2, 3, 1]")
    var split: List[Int64] = [1, 2, 1]
    cp[].groups = MemberMatrix(Int(split.unsafe_ptr()), 3, 1, MEMBER_I64)
    var said = String("")
    try:
        _ = _group_sizes(cp, 3)
    except e:
        said = String(e)
    require(said == ("mojolearn: group Ids are not consecutive: the rows of group 1 are split into more than one "
                     + "run (row 2); CatBoost requires every group's rows to be contiguous"),
            "the door's split-group words, got '" + said + "'")
    said = String("")
    try:
        _ = _group_sizes(cp, 4)
    except e:
        said = String(e)
    require(said == "mojolearn: Length of group_id=3 and length of data=4 are different.",
            "the door's length words, got '" + said + "'")
    _ = ids64^
    _ = ids32^
    _ = split^
    _ = call^


def check_category_map() raises:
    var rows = 8
    var cols = 2
    # column-major: column 0 numeric, column 1 categorical (5 and 5.0 one category, NaN one category)
    var block: List[Float32] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                5.0, 2.0, 5.0, nan[DType.float32](), 2.0, -1.5, nan[DType.float32](), 5.0]
    var oracle = block.copy()
    var columns: List[Int] = [1]
    var map = _category_fit(columns, Int(block.unsafe_ptr()), rows)
    # The door's body sequence on the oracle copy.
    var maps = CategoryMaps()
    var keys = List[UInt64](length=rows, fill=0)
    var column = CategoryF32Ptr(unsafe_from_address=Int(oracle.unsafe_ptr()) + 4 * rows)
    category_keys_from_f32(column, rows, CategoryKeyPtr(unsafe_from_address=Int(keys.unsafe_ptr())))
    maps.fit_column(1, CategoryKeyPtr(unsafe_from_address=Int(keys.unsafe_ptr())), rows, column)
    var words = List[UInt64](length=maps.block_words(), fill=0)
    maps.write_block(CategoryKeyPtr(unsafe_from_address=Int(words.unsafe_ptr())))
    require(len(map) == len(words), "the stored map's word count")
    for i in range(len(words)):  # small-loop(oracle: every map word)
        require(map[i] == words[i], "map word " + String(i))
    for i in range(rows * cols):  # small-loop(oracle: every block word)
        var same = (block[i] == oracle[i]) or (block[i] != block[i] and oracle[i] != oracle[i])
        require(same, "block word " + String(i) + ": " + String(block[i]) + " != " + String(oracle[i]))
    require(block[0] == Float32(0.1), "the numeric column is untouched")
    require(block[rows] == block[rows + 2] and block[rows] == block[rows + 7], "5.0 is one category")
    require(block[rows + 3] == block[rows + 6], "NaN is one category")
    var mp = CategoryKeyPtr(unsafe_from_address=Int(map.unsafe_ptr()))
    require(category_block_columns(mp, len(map), cols) == 1 and category_block_feature(mp, 0) == 1,
            "the map names feature 1")
    # A query with a value the fit never saw (7.0) takes the unseen code k = 4.
    var slot = GbdtMemberSlot()
    var p = _SlotP(unsafe_from_address=Int(UnsafePointer(to=slot)))
    p[].features = cols
    p[].cat_map = map.copy()
    var query: List[Float32] = [0.0, 0.0, 2.0, 7.0]
    _category_apply(p, Int(query.unsafe_ptr()), 2)
    require(query[2] == block[rows + 1], "a fitted value takes its fitted code")
    require(query[3] == Float32(4.0), "an unseen value takes the unseen code k, got " + String(query[3]))
    _ = slot^
    _ = block^
    _ = oracle^
    _ = keys^
    _ = query^


def test_member_image_limits() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_linear_svc_codes()
        elif which == 1:
            check_logistic_codes()
        elif which == 2:
            check_feature_tail()
        elif which == 3:
            check_feature_flags()
        elif which == 4:
            check_group_sizes()
        else:
            check_category_map()
    check_cases("member_image_limits", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
