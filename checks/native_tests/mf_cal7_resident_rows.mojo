# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE RESIDENT ROWS of CalibratedClassifierCV.fit's typed RUN (lane mf-cal7, 2026-11-23;
xtrees/trees_calibrated_fit_run.mojo module note, xtrees/trees_meta_vote_run.mojo `VoteRows`), in a process with no
Python.

The DeviceRows route (GaussianNB, LDA, QDA on an image that gathers on the device) now scores each fold through ONE
member call over the WHOLE X plus the fold's int32 test rows (`VoteRows` THE INDEXED ROWS, `member_f32_block`), not
over a host gather of those rows. The host column has no naive Bayes image that gathers, so the call path is checked
with the host member that reads a row index on this column: the Prep member (core/prep_member.mojo, its
QuantileTransformer kind, MEMBER_CAN_ROW_INDEX on the host transport), whose TRANSFORM is a per-row function of its
fitted quantiles, so the indexed answer must equal the gathered answer word for word.

    case 0   member_f32_block over THE INDEXED ROWS (no scope, no pin) == member_f32_block over the host gather of
             the same rows, for three folds of a 40 x 3 X (repeats and an out-of-order fold included); the pin
             byte is never written (no pin offered)
    case 1   CalFitPlan's pin bookkeeping: `resident` sets the route words, `fold` adds one `took` byte per fold,
             `pins_taken` counts the bytes the RUN wrote

Fail-proof edits (each must fail the named case): in `_vote_rows_point` drop the `call.row_index = ...` line (case
0: the member reads the first rows of the base, or refuses a base of the wrong row count); in `_vote_rows_point` pass
`rows.n` as the base's row count (case 0: the member refuses or reads out of the base); in `_vote_rows_call` write the
pin byte whatever `member_pin_settle` answers (case 0: the byte reads 1); in `CalFitPlan.fold` drop the `took.append`
(case 1); in `pins_taken` count the zero bytes (case 1).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from checks.native_tests.py2mojo_host_members import HostMember, MEMBER_PREP, host_idn_fam, host_route, same_bits
from core.member_abi import MemberPlan, MEMBER_OP_TRANSFORM, MEMBER_OP_PREDICT_PROBA, MEMBER_F32, MEMBER_SCOPE_NONE
from xtrees.trees_meta_vote_run import VoteMember, VoteRows, member_f32_block
from xtrees.trees_calibrated_fit_run import CalFitPlan

comptime N = 40
comptime D = 3


def _x() -> List[Float32]:
    # A deterministic X with ties and spread (no fixture: the oracle is the member's own gathered answer).
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture: one 40 x 3 block)
        for j in range(D):
            x[i * D + j] = Float32(((i * (j + 3) * 7) % 23) - 11) * Float32(0.25) + Float32(j)
    return x^


def _fold(f: Int) -> List[Int32]:
    # Three folds: every third row; rows in reverse order; a repeated row and the last row.
    if f == 0:
        var rows = List[Int32]()
        for i in range(0, N, 3):  # small-loop(fixture: fold rows)
            rows.append(Int32(i))
        return rows^
    if f == 1:
        var rows = List[Int32]()
        for i in range(N - 1, N // 2, -1):  # small-loop(fixture: fold rows)
            rows.append(Int32(i))
        return rows^
    var rows: List[Int32] = [Int32(5), Int32(5), Int32(N - 1), Int32(0), Int32(17)]
    return rows^


def _gather(x: List[Float32], rows: List[Int32]) -> List[Float32]:
    var out = List[Float32](length=len(rows) * D, fill=0.0)
    for i in range(len(rows)):  # small-loop(oracle: the host gather of one fold)
        for j in range(D):
            out[i * D + j] = x[Int(rows[i]) * D + j]
    return out^


def _indexed_case() raises:
    var params: List[Float64] = [5.0, 0.0, 10.0, 10000.0, 0.0, 0.0, Float64(host_idn_fam()), 1.0]
    var plan = MemberPlan()
    plan.params = params.copy()
    plan.route = host_route()
    var member = HostMember(MEMBER_PREP, params^)
    try:
        var x = _x()
        member.fit(x, N, D)
        plan.member = member.member
        var cols = member.width(MEMBER_OP_TRANSFORM)
        require(cols == D, "QuantileTransformer member answers " + String(cols) + " columns, expected " + String(D))
        var plan_at = Int(UnsafePointer(to=plan))
        var vm = VoteMember(plan_at, member.state, MEMBER_OP_TRANSFORM, cols, MEMBER_F32, 0, 1.0)
        for f in range(3):  # small-loop(fixture: the three folds)
            var rows = _fold(f)
            var nte = len(rows)
            var block = _gather(x, rows)
            var keep = List[List[Float32]]()
            var gathered_at = member_f32_block(vm, VoteRows(Int(block.unsafe_ptr()), nte, D), keep)
            var took = List[UInt8](length=1, fill=UInt8(0))
            var indexed_at = member_f32_block(
                vm, VoteRows(Int(x.unsafe_ptr()), N, D, Int(rows.unsafe_ptr()), nte, MEMBER_SCOPE_NONE, 0,
                             Int(took.unsafe_ptr())), keep)
            var gathered = List[Float32](length=nte * cols, fill=0.0)
            var indexed = List[Float32](length=nte * cols, fill=0.0)
            var g = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=gathered_at)
            var h = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=indexed_at)
            for i in range(nte * cols):  # small-loop(oracle: copy of two answers)
                gathered[i] = g[i]
                indexed[i] = h[i]
            same_bits(indexed, gathered, "fold " + String(f) + ": indexed rows vs gathered rows")
            require(took[0] == UInt8(0), "fold " + String(f) + ": the pin byte was written with no pin offered")
            _ = len(keep)  # the answers live in `keep` to here
            _ = len(block)
            _ = len(rows)
        _ = len(x)
        _ = len(plan.params)  # the plan is read through its address above
    finally:
        member.free()


def _pins_case() raises:
    var plan = CalFitPlan(0, 10, 2, 0, 3, MEMBER_OP_PREDICT_PROBA, True, True, False)
    var rows = List[Int32](length=4, fill=Int32(0))
    for _ in range(3):  # small-loop(fixture: three folds)
        plan.fold(Int(rows.unsafe_ptr()), 4)
    require(len(plan.took) == 3, "one took byte per fold: " + String(len(plan.took)))
    require(not plan.indexed and plan.pins_taken() == 0, "a new plan reads gathered rows and took nothing")
    plan.resident(7, 99)
    require(plan.indexed and plan.scope == 7 and plan.pin == 99, "resident() sets the route words")
    plan.took[0] = UInt8(1)
    plan.took[2] = UInt8(1)
    require(plan.pins_taken() == 2, "pins_taken counts the written bytes: " + String(plan.pins_taken()))
    _ = len(rows)


def _case(unit: Int) raises:
    if unit == 0:
        _indexed_case()
    else:
        _pins_case()


def test_mf_cal7_resident_rows() raises:
    check_cases_serial("mf_cal7_resident_rows", 2, _case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
