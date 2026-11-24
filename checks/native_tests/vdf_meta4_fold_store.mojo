# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD FROM THE STORE and CalibratedClassifierCV's TWO INDEXES (lane vdf-meta4, 2026-11-24;
xtrees/trees_meta_vote_run.mojo `VoteRows.member_rows` / `took_each`, xtrees/trees_calibrated_fit_run.mojo
`CalFitPlan.in_base` / `fold_in`, bindings/trees_public_meta_door.mojo `_vote_run_indexed`, `_cal_native`,
`_stack_place_indexed`), in a process with no Python.

An opted-in meta estimator's predict over a model-selection fold (`DeviceRows`) hands every row-index member the
base, the fold's rows, the scope and ONE PIN REFERENCE PER MEMBER CALL (each member's answer in its own byte, so
two members' caches that both take the pin each own one; before, one reference backed every call of a stacking
fold and two takers released it twice). CalibratedClassifierCV.fit over a fold reads its cv splits composed inside
the fold (base rows: what the member READS) while its out-of-fold scores are PLACED, and its test codes read, at
the fold positions. The device-row path must give the bytes of the host-row path (the fold gathered on the host,
as the admission did before), which is the CPU column's input.

Host column (CPU_COLUMN_BUILD):
    case 0   `VoteRows.member_rows`: without an index, the plain rows; indexed with `took_each` False, the one
             answer byte for every member (the nested meta member's `pin_kept`); with `took_each` True, member j's
             byte at `took + j`; and the pin accounting the doors make from those bytes (references taken = one
             per call, returned = calls - bytes set)
    case 1   CalFitPlan's TWO INDEXES: `fold` keeps one index (`x_test_rows` == `test_rows`, `x_rows` == n,
             `draw_codes` == codes); `in_base` + `fold_in` keep them apart; the codes a unit's composed train draw
             reads out of `draw_codes` (the fold codes put at base rows) == the fold's own codes at those
             positions, and the base rows a fold member reads at `x_test_rows` == the fold block's rows at
             `test_rows` (bitwise), 40000 base rows

Device column (the same bytes as the host-row path):
    case 0   the vote predict FROM THE STORE: the fold's rows taken on the device out of the base put once, then
             the prediction entry body == the same entry over the fold gathered on the host
    case 1   the calibrated fold scores: a cv split's test rows composed inside the fold and taken on the device,
             scored and PLACED at the split's fold positions of an n-row out-of-fold block == the host route (the
             fold gathered, then the split's rows, scored, placed at the same positions), bit for bit

FAIL-PROOF EDITS (each must fail the named case): in `member_rows` drop the `took + j` (host 0); in `fold` append
`rows + 4` to `x_test_rows` (host 1); in `fold_in` swap `place` and `read` (host 1, device 1 through the oracle).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from core.meta_fold_rows import meta_rows_index, meta_rows_compose, meta_rows_take_u32
from core.member_abi import MEMBER_OP_PREDICT_PROBA
from xtrees.trees_meta_vote_run import VoteRows
from xtrees.trees_calibrated_fit_run import CalFitPlan
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from core.p3_linear_impl import ols_predict_device
from core.fold_x_device import FOLD_X_STORE, fold_x_word

comptime NB = 40000
comptime FD = 4
comptime DN = 900


def _outer(base_rows: Int) -> List[Int64]:
    """The fold: every third row from the top down, then a repeat of row 7 and the last row."""
    var rows = List[Int64]()
    for i in range(base_rows - 1, -1, -3):  # small-loop(fixture: fold rows)
        rows.append(Int64(i))
    rows.append(Int64(7))
    rows.append(Int64(7))
    rows.append(Int64(base_rows - 1))
    return rows^


def _split(n: Int) -> List[Int32]:
    """A cv split's test positions in the fold: 2, 5, 8, ... then 1 (out of order)."""
    var rows = List[Int32]()
    for i in range(2, n, 3):  # small-loop(fixture: split positions)
        rows.append(Int32(i))
    rows.append(Int32(1))
    return rows^


def _x(n: Int, d: Int) -> List[Float32]:
    var x = List[Float32](length=n * d, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for j in range(d):  # small-loop(fixture columns)
            x[i * d + j] = Float32(((i * (j + 5) * 11) % 29) - 13) * Float32(0.125) + Float32(j)
    return x^


def _index(outer: List[Int64], base_rows: Int) raises -> Tuple[List[Int32], List[Int64]]:
    var n = len(outer)
    var draw = List[Int32](length=n, fill=Int32(-5))
    var inv = List[Int64](length=base_rows, fill=Int64(-9))
    meta_rows_index(Int(outer.unsafe_ptr()), n, base_rows, Int(draw.unsafe_ptr()), Int(inv.unsafe_ptr()))
    return (draw^, inv^)


def _host_case(case_index: Int) raises:
    if case_index == 0:
        var plain = VoteRows(4096, 10, 3).member_rows(2)
        require(plain.index == 0 and plain.x == 4096 and plain.n == 10 and plain.d == 3, "plain rows changed")
        var took = List[UInt8](length=3, fill=UInt8(0))
        var at = Int(took.unsafe_ptr())
        var one = VoteRows(4096, 50, 3, 8192, 10, 7, 99, at)
        for j in range(3):  # small-loop(fixture: three member calls)
            var r = one.member_rows(j)
            require(r.took == at and r.index == 8192 and r.n == 10 and r.base_rows == 50 and r.scope == 7
                    and r.pin == 99, "a one-byte member call does not read the one byte")
        var each = VoteRows(4096, 50, 3, 8192, 10, 7, 99, at)
        each.took_each = True
        for j in range(3):  # small-loop(fixture: three member calls, two caches take the pin)
            var r = each.member_rows(j)
            require(r.took == at + j, "member " + String(j) + " does not answer in its own byte")
            if j != 1:
                MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=r.took)[] = UInt8(1)
        var taken = 0
        for j in range(3):  # small-loop(oracle: the doors' count)
            if took[j] != UInt8(0):
                taken += 1
        require(taken == 2 and 3 - taken == 1, "two takers must own two of three references, one returned")
        var none = VoteRows(4096, 50, 3, 8192, 10, 7, 0, 0)
        none.took_each = True
        require(none.member_rows(2).took == 0, "no pin offered: no answer byte")
        _ = len(took)
        return
    # case 1: CalFitPlan's two indexes
    var single = CalFitPlan(1000, 12, FD, 2000, 3, MEMBER_OP_PREDICT_PROBA, True, True)
    single.fold(3000, 4)
    require(single.x_rows == 12 and single.draw_codes == 2000, "one index: x_rows / draw_codes moved")
    require(single.x_test_rows[0] == 3000 and single.test_rows[0] == 3000, "one index: the two lists differ")
    var outer = _outer(NB)
    var n = len(outer)
    var got = _index(outer, NB)
    var draw = got[0].copy()
    var inv = got[1].copy()
    var codes = List[Int32](length=n, fill=Int32(0))
    for i in range(n):  # small-loop(fixture: the fold's own codes, a function of the base row)
        codes[i] = Int32(Int(outer[i]) % 3)
    var draw_codes = List[Int32](length=NB, fill=Int32(-7))
    meta_rows_take_u32(Int(codes.unsafe_ptr()), Int(inv.unsafe_ptr()), NB, Int(draw_codes.unsafe_ptr()))
    _ = inv^
    var te = _split(n)
    var m = len(te)
    var te_x = List[Int32](length=m, fill=Int32(-1))
    meta_rows_compose(Int(te.unsafe_ptr()), m, Int(draw.unsafe_ptr()), Int(te_x.unsafe_ptr()))
    var plan = CalFitPlan(Int(draw.unsafe_ptr()), n, FD, Int(codes.unsafe_ptr()), 3, MEMBER_OP_PREDICT_PROBA, True,
                          False)
    plan.in_base(NB, Int(draw_codes.unsafe_ptr()))
    plan.fold_in(Int(te.unsafe_ptr()), Int(te_x.unsafe_ptr()), m)
    require(plan.x_rows == NB and plan.n == n and plan.draw_codes == Int(draw_codes.unsafe_ptr()),
            "in_base: the base words are not the plan's")
    require(plan.test_rows[0] == Int(te.unsafe_ptr()) and plan.x_test_rows[0] == Int(te_x.unsafe_ptr())
            and plan.test_counts[0] == m and plan.folds() == 1, "fold_in: placement and read rows mixed")
    var x = _x(NB, FD)
    for i in range(m):  # small-loop(oracle: what a fold member reads vs the fold block, and its codes)
        var p = Int(te[i])
        var b = Int(te_x[i])
        require(b == Int(outer[p]), "composed test row " + String(i) + " is not the fold's row")
        require(draw_codes[b] == codes[p], "the code a draw reads at row " + String(i) + " differs")
        for c in range(FD):  # small-loop(oracle gather)
            require(x[b * FD + c].to_bits() == x[Int(outer[p]) * FD + c].to_bits(), "read row " + String(i)
                    + " differs from the fold block's")
    _ = len(codes)
    _ = len(draw)


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(case_index: Int) raises:
            _host_case(case_index)

        check_cases_serial("vdf_meta4_fold_store host", 2, body)


def _same(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(word compare)
        require(a[i].to_bits() == b[i].to_bits(), label + ": word " + String(i) + " differs")


def _device_case(case_index: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        comptime _CTX = "MojolearnEstimatorsContextIdenticalV3"
        var x = _x(DN, FD)
        var outer = _outer(DN)
        var n = len(outer)
        var got = _index(outer, DN)
        var draw = got[0].copy()
        # The rows the members read: the whole fold (case 0, a vote predict) or a split composed inside it
        # (case 1, a calibrated fold's test rows); `place` the fold positions their outputs go to.
        var place = List[Int32]()
        if case_index == 0:
            for i in range(n):  # small-loop(fixture: every fold position)
                place.append(Int32(i))
        else:
            place = _split(n)
        var m = len(place)
        var read = List[Int32](length=m, fill=Int32(0))
        meta_rows_compose(Int(place.unsafe_ptr()), m, Int(draw.unsafe_ptr()), Int(read.unsafe_ptr()))
        var read64 = List[Int64](length=m, fill=Int64(0))
        for i in range(m):  # small-loop(fixture: the store reads int64 row numbers)
            read64[i] = Int64(read[i])
        # The host-row path: the fold gathered out of the base (the admission), then the rows at `place`.
        var host_x = List[Float32](length=m * FD, fill=0.0)
        for r in range(m):  # small-loop(oracle: the former fold gather, then the split's rows)
            var p = Int(place[r])
            for c in range(FD):  # small-loop(oracle gather)
                host_x[r * FD + c] = x[Int(outer[p]) * FD + c]
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), DN * FD)
        var slot = store[].take_rows(store_ctx, base, FD, Int(read64.unsafe_ptr()), m)
        var ctx = process_ctx[_CTX]()
        var failure = String("")
        try:
            var coef: List[Float32] = [Float32(0.5), Float32(-1.25), Float32(1.75), Float32(-0.375)]
            var out_a = List[Float32](length=m, fill=0.0)
            var out_b = List[Float32](length=m, fill=0.0)
            var ra = ols_predict_device(ctx,
                MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=fold_x_word(slot)),
                MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef.unsafe_ptr())),
                Int(out_a.unsafe_ptr()), m, FD, Float32(-0.5), True)
            var rb = ols_predict_device(ctx,
                MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(host_x.unsafe_ptr())),
                MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef.unsafe_ptr())),
                Int(out_b.unsafe_ptr()), m, FD, Float32(-0.5), True)
            _ = coef^
            require(ra == rb and ra == 0, "finite status differs")
            # THE PLACEMENT (case 1: the out-of-fold block, n rows, at the split's fold positions).
            var block_a = List[Float32](length=n, fill=0.0)
            var block_b = List[Float32](length=n, fill=0.0)
            for r in range(m):  # small-loop(fixture: the scatter at fold positions, both routes)
                block_a[Int(place[r])] = out_a[r]
                block_b[Int(place[r])] = out_b[r]
            _same(out_a, out_b, "scores")
            _same(block_a, block_b, "out-of-fold block")
        except error:
            failure = String(error)
        store[].free(store_ctx, slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = read64^
        _ = host_x^
        _ = draw^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(case_index: Int) raises:
            _device_case(case_index)

        check_cases_serial("vdf_meta4_fold_store device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
