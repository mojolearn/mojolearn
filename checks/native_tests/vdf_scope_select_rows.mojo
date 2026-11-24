# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE RESIDENT ROUTE of the Select members (lane vdf-scope-neural; core/select_member.mojo `_gathers`, `_fold_y`),
in a process with no Python, on the host x_prep transport.

A search hands a member that answers MEMBER_CAN_ROW_INDEX the WHOLE X (and the whole target) plus the fold's int32
row numbers instead of a host-gathered block. The device column takes the rows on the device
(core/prep_member.mojo `_x_record`, `x_prep_dev_take_rows`: a word copy); this column takes them on the host through
the same record rule. Both must equal the member's answer on the gathered block word for word, which is what the
CPU column computes for the gathered route.

    case 0   VarianceThreshold: fit over THE INDEXED ROWS == fit over the gathered rows (variances_, support), and
             transform over the indexed rows == transform over the gathered rows, for three folds (repeats and an
             out-of-order fold included)
    case 1   SelectKBest f_classif (k=2): fit over the indexed rows with the WHOLE y == fit over the gathered X and y
             (scores_, pvalues_, support); an out-of-range row is refused in the IndexError words
    case 2   capabilities: the host route (no device put / gather) answers neither MEMBER_CAN_ROW_INDEX nor
             MEMBER_CAN_SCOPE; a route with the four gathering words answers both

Fail-proof edits (each must fail the named case): in `_fit_variance` read `call.x.rows` for n (case 0: the program
reads the base's rows or the arena is the wrong size); in `_fit_kbest` hand `call.y.data` instead of the fold's
words (case 1); in `_fold_y` drop the range test (case 1); in `_gathers` drop the PREP_ROUTE_TAKE_ROWS test (case 2
still passes: add a route with only that word cleared to see it fail).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from checks.native_tests.py2mojo_host_members import host_idn_fam, same_bits
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_prep.transport_host import x_prep_transport_host_v1
from x_prep.transport_abi import XP_OP_COUNT
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_EXPORT, MEMBER_OP_TRANSFORM, MEMBER_I32, MEMBER_F32,
    MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE, MEMBER_BLOCK_BY_INDEX,
)
from core.member_families import MEMBER_FAMILY_SELECT
from core.prep_member import PREP_ROUTE_OUT, PREP_ROUTE_RANGES, PREP_ROUTE_DEV_PUT, PREP_ROUTE_TAKE_ROWS
from core.select_member import (
    select_member_v2, SELECT_KIND_VARIANCE, SELECT_VARIANCE_PARAM_COUNT, SELECT_KIND_KBEST, SELECT_KBEST_PARAM_COUNT,
    SELECT_SCORE_F_CLASSIF,
)

comptime N = 40
comptime D = 4
comptime K = 3


def _route(flags: Int) -> DeviceRoute:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.second = EntryRef(x_prep_transport_host_v1, entry_ref_word(mode, 0, XP_OP_COUNT))
    route.has_second = True
    route.flags = flags
    return route


def _plan(var words: List[Float64], kind: Int) raises -> MemberPlan:
    var plan = MemberPlan()
    plan.route = _route(0)
    plan.family = MEMBER_FAMILY_SELECT
    plan.kind = kind
    plan.member = MemberRef(EntryRef(select_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params = words^
    plan.classes = K
    return plan^


def _variance_plan() raises -> MemberPlan:
    var words = List[Float64](length=SELECT_VARIANCE_PARAM_COUNT, fill=0.0)
    words[0] = Float64(SELECT_KIND_VARIANCE)
    words[1] = 0.5
    words[2] = Float64(host_idn_fam())
    words[3] = 1.0
    return _plan(words^, SELECT_KIND_VARIANCE)


def _kbest_plan() raises -> MemberPlan:
    var words = List[Float64](length=SELECT_KBEST_PARAM_COUNT, fill=0.0)
    words[0] = Float64(SELECT_KIND_KBEST)
    words[1] = Float64(SELECT_SCORE_F_CLASSIF)
    words[2] = 2.0
    words[3] = Float64(host_idn_fam())
    words[4] = 1.0
    return _plan(words^, SELECT_KIND_KBEST)


def _x() -> List[Float32]:
    # A deterministic X with spread, ties and one near-constant column (no fixture: the oracle is the member's own
    # gathered answer).
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture: one 40 x 4 block)
        for j in range(D):
            x[i * D + j] = Float32(((i * (j + 3) * 7) % 23) - 11) * Float32(0.25) + Float32(j)
        x[i * D + 3] = Float32(1.0) + Float32(i % 2) * Float32(0.125)
    return x^


def _codes() -> List[Int32]:
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: class codes)
        y[i] = Int32((i * 5 + i // 7) % K)
    return y^


def _fold(f: Int) -> List[Int32]:
    # Every third row; rows in reverse order; repeated rows with the last row (each holds every class).
    var rows = List[Int32]()
    if f == 0:
        for i in range(0, N, 3):  # small-loop(fixture: fold rows)
            rows.append(Int32(i))
    elif f == 1:
        for i in range(N - 1, N // 2, -1):  # small-loop(fixture: fold rows)
            rows.append(Int32(i))
    else:
        var some: List[Int32] = [Int32(5), Int32(5), Int32(N - 1), Int32(0), Int32(17), Int32(2), Int32(9), Int32(1)]
        rows = some^
    return rows^


def _take_f32(x: List[Float32], rows: List[Int32], width: Int) -> List[Float32]:
    var out = List[Float32](length=len(rows) * width, fill=0.0)
    for i in range(len(rows)):  # small-loop(oracle: the host gather of one fold)
        for j in range(width):
            out[i * width + j] = x[Int(rows[i]) * width + j]
    return out^


def _take_i32(y: List[Int32], rows: List[Int32]) -> List[Int32]:
    var out = List[Int32](length=len(rows), fill=Int32(0))
    for i in range(len(rows)):  # small-loop(oracle: the host gather of one fold's labels)
        out[i] = y[Int(rows[i])]
    return out^


def _block(plan: MemberPlan, state: Int, index: Int) raises -> List[Float32]:
    var e = member_call_of(plan, state)
    e.block = index
    e.flags = MEMBER_BLOCK_BY_INDEX
    _ = member_call(plan.member, MEMBER_OP_EXPORT, e)
    var words = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=e.dst.data)
    var got = List[Float32](length=e.dst.rows * e.dst.cols, fill=0.0)
    for j in range(len(got)):  # small-loop(oracle: one block's words)
        got[j] = words[j]
    return got^


def _fit_blocks(plan: MemberPlan, x: List[Float32], x_rows: List[Int32], y: List[Int32], with_y: Bool,
                blocks: Int, apply: List[Float32]) raises -> List[List[Float32]]:
    """Fit over `x` (whole, with the fold's `x_rows` when given), export `blocks` blocks (1 ..), then (when `apply`
    is not empty) transform `apply`'s fold the same way. One list per block, then the transform."""
    var n_base = len(x) // D
    var call = member_new(plan)
    var out = List[List[Float32]]()
    try:
        call.x = MemberMatrix(Int(x.unsafe_ptr()), n_base, D)
        if len(x_rows) > 0:
            call.row_index = MemberMatrix(Int(x_rows.unsafe_ptr()), len(x_rows), 1, MEMBER_I32)
        if with_y:
            call.y = MemberMatrix(Int(y.unsafe_ptr()), len(y), 1, MEMBER_I32)
            call.classes = K
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
        for b in range(1, blocks + 1):  # small-loop(oracle: the fitted blocks)
            out.append(_block(plan, call.state, b))
        if len(apply) > 0:
            var support = out[len(out) - 1].copy()
            var kept = 0
            for j in range(D):  # small-loop(oracle: kept columns)
                if support[j] != 0:
                    kept += 1
            var n = len(x_rows) if len(x_rows) > 0 else n_base
            var dst = List[Float32](length=max(n * kept, 1), fill=0.0)
            var t = member_call_of(plan, call.state)
            t.x = MemberMatrix(Int(apply.unsafe_ptr()), len(apply) // D, D)
            if len(x_rows) > 0:
                t.row_index = MemberMatrix(Int(x_rows.unsafe_ptr()), len(x_rows), 1, MEMBER_I32)
            t.dst = MemberMatrix(Int(dst.unsafe_ptr()), n, kept)
            _ = member_call(plan.member, MEMBER_OP_TRANSFORM, t)
            out.append(dst^)
    finally:
        member_free(plan, call)
    _ = len(x)
    _ = len(x_rows)
    _ = len(y)
    _ = len(apply)
    return out^


def _variance_case() raises:
    var plan = _variance_plan()
    var x = _x()
    var none = List[Int32]()
    for f in range(3):  # small-loop(fixture: three folds)
        var rows = _fold(f)
        var block = _take_f32(x, rows, D)
        var gathered = _fit_blocks(plan, block, none, none, False, 2, block)
        var indexed = _fit_blocks(plan, x, rows, none, False, 2, x)
        for b in range(3):  # small-loop(oracle: variances, support, transform)
            same_bits(indexed[b], gathered[b], "VarianceThreshold fold " + String(f) + " block " + String(b))


def _kbest_case() raises:
    var plan = _kbest_plan()
    var x = _x()
    var y = _codes()
    var none = List[Float32]()
    var empty = List[Int32]()
    for f in range(3):  # small-loop(fixture: three folds)
        var rows = _fold(f)
        var block = _take_f32(x, rows, D)
        var labels = _take_i32(y, rows)
        var gathered = _fit_blocks(plan, block, empty, labels, True, 3, none)
        var indexed = _fit_blocks(plan, x, rows, y, True, 3, none)
        for b in range(3):  # small-loop(oracle: scores, support, pvalues)
            same_bits(indexed[b], gathered[b], "SelectKBest fold " + String(f) + " block " + String(b))
    var bad: List[Int32] = [Int32(0), Int32(N)]
    var words = String("")
    try:
        _ = _fit_blocks(plan, x, bad, y, True, 1, none)
    except problem:
        words = String(problem)
    require(words.startswith("IndexError:"), "an out-of-range fold row: got '" + words + "'")


def _capability_case() raises:
    var member = MemberRef(EntryRef(select_member_v2, 0), 0)
    var host = MemberCall(_route(0))
    var mask = member_call(member, MEMBER_OP_CAPABILITIES, host)
    require((mask & (MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE)) == 0, "the host route answers no row index / scope")
    var device = MemberCall(_route(PREP_ROUTE_OUT | PREP_ROUTE_RANGES | PREP_ROUTE_DEV_PUT | PREP_ROUTE_TAKE_ROWS))
    mask = member_call(member, MEMBER_OP_CAPABILITIES, device)
    require((mask & MEMBER_CAN_ROW_INDEX) != 0 and (mask & MEMBER_CAN_SCOPE) != 0,
            "a gathering route answers MEMBER_CAN_ROW_INDEX and MEMBER_CAN_SCOPE")
    var no_take = MemberCall(_route(PREP_ROUTE_OUT | PREP_ROUTE_RANGES | PREP_ROUTE_DEV_PUT))
    mask = member_call(member, MEMBER_OP_CAPABILITIES, no_take)
    require((mask & (MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE)) == 0, "a route without the device take answers neither")


def _case(unit: Int) raises:
    if unit == 0:
        _variance_case()
    elif unit == 1:
        _kbest_case()
    else:
        _capability_case()


def test_vdf_scope_select_rows() raises:
    check_cases_serial("vdf_scope_select_rows", 3, _case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
