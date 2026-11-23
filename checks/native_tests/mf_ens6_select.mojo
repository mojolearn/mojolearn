# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-ens6: SelectKBest's new member settings (core/select_member.mojo kind 1) in a process with no Python,
on the host x_prep transport. Oracles: the select door's own chain (`mi_host`, the body of
bindings/prep_select_native_door.mojo `prep_mutual_info_call` on the typed builders, as
checks/native_tests/prep_feature_selection.mojo runs it) and the door's k-range words.

 1. mutual_info_classif (score word 4), k=2, on the bm-sel fixtures (500 x 4, three classes): the member's scores_
    block == the door chain's scores word for word (same program, seed 0, n_neighbors 3, every column
    continuous), and its support keeps exactly the two best columns (a later column wins a tie).
 2. mutual_info_regression (score word 5), k=2: the same against the regression chain.
 3. f_classif with k = -1 (not 'all'): the fit raises the door's "mojolearn: k should be 0 <= k <= n_features = 4;
    got -1" after the scores (before, admission answered False and the door raised it).
 4. mutual_info_classif with k 'all' (word [5] = 1, word [2] = 0): every column kept; and k = 5 > d raises the
    door's range words with d = 4.

FAIL-PROOF EDITS (each must make the named case fail): in core/select_member.mojo `_fit_kbest_mi` pass seed 1 to
`MiProgram` (cases 1, 2), or call `score_topk_stage` with k - 1 (cases 1, 2); in `_fit_kbest` compute `k_ok` as
`keep_all or k <= d` (case 3); in `_fit_kbest_mi` read `keep_all` from word [2] < 0 (case 4).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32, fixture_i32
from checks.native_tests.prep_enc_host import enc_host_run
from checks.native_tests.prep_cls_host import codes_words
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_prep.blocked import IDN_STATS_BLOCKED, IDN_CLASS_ONEPASS
from x_prep.select_blocked import IDN_SELECT_BLOCKED
from x_prep.transport_host import x_prep_transport_host_v1
from x_prep.transport_abi import XP_OP_COUNT
from bindings.prep_encode_prog_run import EncProgram
from bindings.prep_encode_cats_run import CatBlock
from bindings.prep_select_run import MiProgram, mi_class_count_stage, mi_int_words, mi_class_check, mi_stage
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_EXPORT, MEMBER_I32, MEMBER_F32,
)
from core.member_families import MEMBER_FAMILY_SELECT
from core.select_member import (
    select_member_v2, SELECT_KIND_KBEST, SELECT_KBEST_PARAM_COUNT, SELECT_BLOCK_SCORES, SELECT_BLOCK_SUPPORT,
    SELECT_SCORE_F_CLASSIF, SELECT_SCORE_MI_CLASSIF, SELECT_SCORE_MI_REGRESSION,
)

comptime CASES = 4
comptime N = 500
comptime D = 4
comptime K = 3
comptime BM_DIR = "checks/native_tests/fixtures/prep_sel_bm/"


def _fam() -> Int:
    return (1 if IDN_STATS_BLOCKED else 0) | (2 if IDN_CLASS_ONEPASS else 0) | (4 if IDN_SELECT_BLOCKED else 0)


def _plan(score: Int, k: Int, every: Bool) raises -> MemberPlan:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.second = EntryRef(x_prep_transport_host_v1, entry_ref_word(mode, 0, XP_OP_COUNT))
    route.has_second = True
    route.flags = 0
    var plan = MemberPlan()
    plan.route = route
    plan.family = MEMBER_FAMILY_SELECT
    plan.kind = SELECT_KIND_KBEST
    plan.member = MemberRef(EntryRef(select_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    var words = List[Float64](length=SELECT_KBEST_PARAM_COUNT, fill=0.0)
    words[0] = Float64(SELECT_KIND_KBEST)
    words[1] = Float64(score)
    words[2] = Float64(k)
    words[3] = Float64(_fam())
    words[4] = 1.0
    words[5] = 1.0 if every else 0.0
    words[6] = 0.0
    plan.params = words^
    plan.classes = K
    return plan^


def _oracle(x: List[Float32], codes: List[Int32], yr: List[Float32], discrete_target: Bool) raises -> List[Float32]:
    """`prep_mutual_info_call(X, y, 'auto', 3, None)` on the host column: every column continuous, seed 0."""
    var counts = List[Int]()
    var owners = List[List[Float32]]()
    if discrete_target:
        var pc = EncProgram(True)
        var tot = mi_class_count_stage(pc, 0, N, N, K)
        var held = List[List[Float32]]()
        held.append(codes_words(codes))
        var counted = enc_host_run(pc, held)
        counts = mi_int_words(Int(counted.unsafe_ptr()), tot, K)
        _ = counted^  # the arena lives through the read
        mi_class_check(counts, True)
        owners.append(codes_words(codes))
    else:
        owners.append(yr.copy())
    owners.append(x.copy())
    var cont = List[Int]()
    for j in range(D):  # small-loop(fixture: every column continuous)
        cont.append(j)
    var prog = EncProgram(True)
    var staged = mi_stage(prog, MiProgram(N, 3, 0, discrete_target, False, (_fam() & 1) != 0, cont^, List[Int](),
        1, 0, N, counts^, -1, CatBlock(-1, 0, 0, -1, 0), List[Int]()))
    var arena = enc_host_run(prog, owners)
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(arena.unsafe_ptr()))
    var scores = List[Float32](length=D, fill=0)
    for j in range(D):  # small-loop(oracle: the d scores)
        scores[j] = src[staged.outc + j]
    _ = arena^  # the arena lives through the read
    return scores^


struct _Fitted(Movable):
    var scores: List[Float32]
    var support: List[Float32]

    def __init__(out self):
        self.scores = List[Float32]()
        self.support = List[Float32]()


def _fit(plan: MemberPlan, x: List[Float32], codes: List[Int32], yr: List[Float32], classif: Bool) raises -> _Fitted:
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    if classif:
        call.y = MemberMatrix(Int(codes.unsafe_ptr()), N, 1, MEMBER_I32)
        call.classes = K
    else:
        call.y = MemberMatrix(Int(yr.unsafe_ptr()), N, 1, MEMBER_F32)
    var out = _Fitted()
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
        var blocks: List[Int] = [SELECT_BLOCK_SCORES, SELECT_BLOCK_SUPPORT]
        for b in range(2):  # small-loop(oracle: two blocks)
            var e = member_call_of(plan, call.state)
            e.block = blocks[b]
            e.flags = 0
            _ = member_call(plan.member, MEMBER_OP_EXPORT, e)
            var words = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=e.dst.data)
            var got = List[Float32](length=e.dst.cols, fill=0)
            for j in range(e.dst.cols):  # small-loop(oracle: one block's words)
                got[j] = words[j]
            if b == 0:
                out.scores = got^
            else:
                out.support = got^
    finally:
        member_free(plan, call)
    _ = len(x)
    _ = len(codes)
    _ = len(yr)
    return out^


def _refusal(plan: MemberPlan, x: List[Float32], codes: List[Int32], yr: List[Float32]) raises -> String:
    var words = String("")
    try:
        _ = _fit(plan, x, codes, yr, True)
    except problem:
        words = String(problem)
    return words^


def _top_two(scores: List[Float32], support: List[Float32], label: String) raises:
    """Exactly two kept, each kept score >= every dropped one (a tie: the later column kept)."""
    var kept = 0
    for j in range(D):  # small-loop(oracle: the mask)
        if support[j] != 0:
            kept += 1
    require(kept == 2, label + ": " + String(kept) + " columns kept, expected 2")
    for a in range(D):  # small-loop(oracle: kept vs dropped)
        for c in range(D):  # small-loop(oracle: kept vs dropped)
            if support[a] != 0 and support[c] == 0:
                require(scores[a] > scores[c] or (scores[a] == scores[c] and a > c),
                        label + ": column " + String(a) + " kept over a better column " + String(c))


def check_mi(regression: Bool) raises:
    var label = String("mutual_info_regression") if regression else String("mutual_info_classif")
    var x = fixture_f32(BM_DIR + "mi-x.f32", N * D)
    var y = fixture_i32(BM_DIR + "mi-y.i32", N)
    var yr = fixture_f32(BM_DIR + "mi-yr.f32", N)
    var want = _oracle(x, y, yr, not regression)
    var got = _fit(_plan(SELECT_SCORE_MI_REGRESSION if regression else SELECT_SCORE_MI_CLASSIF, 2, False), x, y, yr,
                   not regression)
    require(len(got.scores) == D, label + ": scores_ width")
    for j in range(D):  # small-loop(oracle: word for word)
        require(got.scores[j] == want[j], label + " score " + String(j) + ": " + String(got.scores[j]) + " != "
                + String(want[j]))
    _top_two(got.scores, got.support, label)


def check_negative_k() raises:
    var x = fixture_f32(BM_DIR + "mi-x.f32", N * D)
    var y = fixture_i32(BM_DIR + "mi-y.i32", N)
    var yr = fixture_f32(BM_DIR + "mi-yr.f32", N)
    var words = _refusal(_plan(SELECT_SCORE_F_CLASSIF, -1, False), x, y, yr)
    var expected = String("mojolearn: k should be 0 <= k <= n_features = 4; got -1")
    require(words == expected, "f_classif k=-1: got '" + words + "'")


def check_all_and_range() raises:
    var x = fixture_f32(BM_DIR + "mi-x.f32", N * D)
    var y = fixture_i32(BM_DIR + "mi-y.i32", N)
    var yr = fixture_f32(BM_DIR + "mi-yr.f32", N)
    var got = _fit(_plan(SELECT_SCORE_MI_CLASSIF, 0, True), x, y, yr, True)
    for j in range(D):  # small-loop(oracle: every column kept)
        require(got.support[j] != 0, "k='all': column " + String(j) + " dropped")
    var words = _refusal(_plan(SELECT_SCORE_MI_CLASSIF, 5, False), x, y, yr)
    require(words == "mojolearn: k should be 0 <= k <= n_features = 4; got 5", "mutual_info k=5: got '" + words + "'")


def test_mf_ens6_select() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_mi(False)
        elif which == 1:
            check_mi(True)
        elif which == 2:
            check_negative_k()
        else:
            check_all_and_range()

    check_cases("mf_ens6_select", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
