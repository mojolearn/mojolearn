# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-ens6: the NEW words of the member settings this lane added, minted and refused by the members'
own entries in a process with no Python (no image needed: MEMBER_OP_NEW reads only the words).

 1. The recurrent member (core/sequence_member.mojo): 30 words with a built-in schedule (word [20] = 1, class
    index [21] = 3 StepLR, its LrSpec in [22 .. 29]) mint a state; a class index 9 is refused ("schedule words
    are out of range"); the old 20-word form is refused ("takes thirty parameters").
 2. The ETS member (core/forecast_seq_member.mojo kind 1): 14 words with the FAST stop (word [11] = 1, iterations
    [12], drop [13]) mint a state; a stop word 2 is refused; the old 11-word form is refused ("takes fourteen
    parameters").
 3. SelectKBest (core/select_member.mojo kind 1): 7 words with score_func 5 (mutual_info_regression) mint a
    state; score_func 6 is refused ("unknown SelectKBest score_func code"); the old 5-word form is refused
    ("takes seven parameters").

FAIL-PROOF EDITS (each must make the named case fail): in core/sequence_member.mojo `_new` drop the schedule
range check (case 1); in core/forecast_seq_member.mojo `_new` drop `w[11]` from the ETS range check (case 2); in
core/select_member.mojo `_select_member` MEMBER_OP_NEW accept `source[1] > 5.0` (case 3).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef
from core.member_abi import MemberCall, MemberPlan, MemberRef, member_new, member_free
from core.member_families import MEMBER_FAMILY_SEQUENCE, MEMBER_FAMILY_FORECAST_SEQ, MEMBER_FAMILY_SELECT
from core.sequence_member import sequence_member_v2
from core.forecast_seq_member import forecast_seq_member_v2
from core.select_member import select_member_v2

comptime CASES = 3


def _plan(family: Int, which: Int, var words: List[Float64]) -> MemberPlan:
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    var plan = MemberPlan()
    plan.route = route
    plan.family = family
    plan.kind = Int(words[0])
    if which == 0:
        plan.member = MemberRef(EntryRef(sequence_member_v2, 0), 0)
    elif which == 1:
        plan.member = MemberRef(EntryRef(forecast_seq_member_v2, 0), 0)
    else:
        plan.member = MemberRef(EntryRef(select_member_v2, 0), 0)
    plan.params = words^
    return plan^


def _mints(plan: MemberPlan, label: String) raises:
    var call = member_new(plan)
    require(call.state != 0, label + ": no state minted")
    member_free(plan, call)


def _refused(plan: MemberPlan, part: String, label: String) raises:
    var words = String("")
    try:
        var call = member_new(plan)
        member_free(plan, call)
    except problem:
        words = String(problem)
    require(words.find(part) >= 0, label + ": got '" + words + "', expected '" + part + "'")


def _rnn(schedule_class: Float64, count: Int) -> List[Float64]:
    # regressor, LSTM, hidden 4, one layer, batch 8, 2 epochs, no shuffle, lr 0.01, chunk 64, optimizer words 0,
    # seed 7, not fresh; then StepLR(base_lr 0.1, step_size 2, gamma 0.5): LrSpec(20, 2, 0, 0, 0.1, 0.5, 0, 0).
    var w: List[Float64] = [0.0, 2.0, 4.0, 1.0, 8.0, 2.0, 0.0, 0.01, 64.0, 0.0, 0.0, 0.9, 0.999, 1e-8, 0.0, 0.0,
                            0.0, 7.0, 0.0, 0.0, 1.0, schedule_class, 20.0, 2.0, 0.0, 0.0, 0.1, 0.5, 0.0, 0.0]
    w.resize(count, 0.0)
    return w^


def _ets(stop: Float64, count: Int) -> List[Float64]:
    # ETS kind 1, season 1, A, A, N, damped, no fixed setting, then the stop (100 iterations, drop 1e-4).
    var w: List[Float64] = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, stop, 100.0, 1e-4]
    w.resize(count, 0.0)
    return w^


def _kbest(score: Float64, count: Int) -> List[Float64]:
    var w: List[Float64] = [1.0, score, 2.0, 0.0, 1.0, 0.0, 0.0]
    w.resize(count, 0.0)
    return w^


def check_sequence() raises:
    _mints(_plan(MEMBER_FAMILY_SEQUENCE, 0, _rnn(3.0, 30)), "recurrent with StepLR")
    _refused(_plan(MEMBER_FAMILY_SEQUENCE, 0, _rnn(9.0, 30)), "schedule words are out of range", "class 9")
    _refused(_plan(MEMBER_FAMILY_SEQUENCE, 0, _rnn(3.0, 20)), "takes thirty parameters", "20 words")


def check_ets() raises:
    _mints(_plan(MEMBER_FAMILY_FORECAST_SEQ, 1, _ets(1.0, 14)), "ETS with the FAST stop")
    _refused(_plan(MEMBER_FAMILY_FORECAST_SEQ, 1, _ets(2.0, 14)), "out of range", "stop word 2")
    _refused(_plan(MEMBER_FAMILY_FORECAST_SEQ, 1, _ets(0.0, 11)), "takes fourteen parameters", "11 words")


def check_kbest() raises:
    _mints(_plan(MEMBER_FAMILY_SELECT, 2, _kbest(5.0, 7)), "SelectKBest mutual_info_regression")
    _refused(_plan(MEMBER_FAMILY_SELECT, 2, _kbest(6.0, 7)), "unknown SelectKBest score_func code", "score 6")
    _refused(_plan(MEMBER_FAMILY_SELECT, 2, _kbest(0.0, 5)), "takes seven parameters", "5 words")


def test_mf_ens6_words() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_sequence()
        elif which == 1:
            check_ets()
        else:
            check_kbest()

    check_cases("mf_ens6_words", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
