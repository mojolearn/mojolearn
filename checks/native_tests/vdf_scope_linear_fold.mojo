# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD X WORD of the linear member group (lane vdf-scope-linear, 2026-11-24; core/fold_x_device.mojo,
core/member_fold_x.mojo, core/linear_member.mojo `_fold_op`), in a process with no Python.

Host column (CPU_COLUMN_BUILD):
    case 0   `member_fold_take_f32` over three folds of a 40000 x 3 matrix (every third row; the rows in reverse,
             crossing three task chunks; a repeated row and the last row) == a plain serial gather, word for word
    case 1   `member_fold_index`: an int32 index is widened to the same int64 numbers; a row number outside X
             (-1, then n) is refused in the member's words; a two-column index is refused
    case 2   the linear member's capabilities on a route with no image (the host column): no
             MEMBER_CAN_ROW_INDEX and no MEMBER_CAN_SCOPE (the host column keeps its host path)

Device column:
    case 0   the Gram Ridge fit over a fold slot (X put once into the fold store, the fold's rows taken on the
             device, `linear_gram_fit_host` handed `fold_x_word(slot)`) == the same fit over the host gather of
             those rows: coef, column means and the binary64 y mean bit for bit
    case 1   the prediction entry body (`ols_predict_device`) over the fold slot == over the host gather
The host-gather fit is the estimator's own host-input route, whose device and CPU columns the release's one
verification compares; equal bytes here put the fold route in that same column.

FAIL-PROOF EDITS (each must fail the named case): in `member_fold_take_f32` read `i[r] + 1` (host 0); in
`_index_check` test `v > base_rows` (host 1); in core/linear_member.mojo `_route_slot_x` return True (host 2); in
`fold_x_fill` copy from word 1 of the slot (device 0 and 1); in `fold_x_word` drop the `+ 1` (device 0: slot -0
reads as a host address).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_I32, MEMBER_I64, MEMBER_OP_CAPABILITIES, MEMBER_CAN_ROW_INDEX,
    MEMBER_CAN_SCOPE, MemberRef, member_call,
)
from core.entry_table import EntryRef
from core.device_route import DeviceRoute
from core.member_fold_x import MemberFoldX, member_fold_take_f32, member_fold_index
from core.linear_member import linear_member_v2
from max.gpu.host import DeviceContext
from core.gram_multi_gpu import gram_plan_current
from core.neural_context import process_ctx
from glm.impl.gram_solve import linear_gram_fit_host
from core.p3_linear_impl import ols_predict_device
from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_word

comptime N = 40000
comptime D = 3
comptime FN = 300
comptime FD = 4


def _x(n: Int, d: Int) -> List[Float32]:
    var x = List[Float32](length=n * d, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for j in range(d):  # small-loop(fixture columns)
            x[i * d + j] = Float32(((i * (j + 3) * 7) % 23) - 11) * Float32(0.25) + Float32(j)
    return x^


def _fold(f: Int, n: Int) -> List[Int64]:
    var rows = List[Int64]()
    if f == 0:
        for i in range(0, n, 3):  # small-loop(fixture: fold rows)
            rows.append(Int64(i))
    elif f == 1:
        for i in range(n - 1, n // 3, -1):  # small-loop(fixture: fold rows)
            rows.append(Int64(i))
    else:
        rows = [Int64(5), Int64(5), Int64(n - 1), Int64(0), Int64(17)]
    return rows^


def _serial(x: List[Float32], index: List[Int64], d: Int) -> List[Float32]:
    var out = List[Float32](length=len(index) * d, fill=0.0)
    for r in range(len(index)):  # small-loop(oracle gather)
        for c in range(d):  # small-loop(oracle gather)
            out[r * d + c] = x[Int(index[r]) * d + c]
    return out^


def _same(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(word compare)
        require(a[i].to_bits() == b[i].to_bits(), label + ": word " + String(i) + " differs")


def _host_case(which: Int) raises:
    var x = _x(N, D)
    if which == 0:
        for f in range(3):  # small-loop(the three folds)
            var index = _fold(f, N)
            var got = List[Float32](length=len(index) * D, fill=0.0)
            member_fold_take_f32(Int(x.unsafe_ptr()), Int(index.unsafe_ptr()), len(index), D, Int(got.unsafe_ptr()))
            _same(got, _serial(x, index, D), "fold " + String(f))
        return
    if which == 1:
        var narrow: List[Int32] = [Int32(7), Int32(0), Int32(N - 1)]
        var call = MemberCall(DeviceRoute(0))
        call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
        call.row_index = MemberMatrix(Int(narrow.unsafe_ptr()), 3, 1, MEMBER_I32)
        var fold = MemberFoldX()
        member_fold_index(MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=call))), fold, "Ridge")
        # `fold.index` points into `fold.wide`, which this value owns: read the words while `fold` lives and name
        # it again after (a local is freed at its last named use; the untracked pointer does not keep it alive)
        var wide = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=fold.index)
        var w0 = wide[0]
        var w1 = wide[1]
        var w2 = wide[2]
        var shape_ok = fold.rows == 3 and fold.cols == D
        _ = fold^
        require(shape_ok and w0 == 7 and w1 == 0 and w2 == N - 1, "int32 index not widened to the same numbers")
        for bad in range(2):  # small-loop(the two out-of-range numbers)
            var outside: List[Int64] = [Int64(1), Int64(-1 if bad == 0 else N)]
            call.row_index = MemberMatrix(Int(outside.unsafe_ptr()), 2, 1, MEMBER_I64)
            var refused = False
            try:
                var other = MemberFoldX()
                member_fold_index(MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=call))), other, "Ridge")
            except error:
                refused = String(error).find("outside X") >= 0
            require(refused, "a row number outside X was not refused (" + String(bad) + ")")
            _ = outside^
        var two: List[Int64] = [Int64(0), Int64(1)]
        call.row_index = MemberMatrix(Int(two.unsafe_ptr()), 1, 2, MEMBER_I64)
        var refused = False
        try:
            var other = MemberFoldX()
            member_fold_index(MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=call))), other, "Ridge")
        except error:
            refused = String(error).find("row index") >= 0
        require(refused, "a two-column index was not refused")
        _ = narrow^
        _ = two^
        _ = x^
        return
    # case 2: the family's union on a route with no image
    var probe = MemberCall(DeviceRoute(0))
    var mask = member_call(MemberRef(EntryRef(linear_member_v2, 0), 0), MEMBER_OP_CAPABILITIES, probe)
    require((mask & (MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE)) == 0,
            "the host column answered the fold bits without an image that reads them")


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _host_case(which)

        check_cases("vdf_scope_linear_fold host", 3, body)


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        comptime _CTX = "MojolearnEstimatorsContextIdenticalV3"
        var x = _x(FN, FD)
        var y = List[Float32](length=FN, fill=0.0)
        for i in range(FN):  # small-loop(fixture targets)
            y[i] = Float32((i * 13) % 17) * Float32(0.5) - Float32(3.0)
        var index = _fold(1, FN)
        var n = len(index)
        var gathered = _serial(x, index, FD)
        var y_fold = List[Float32](length=n, fill=0.0)
        for r in range(n):  # small-loop(oracle targets)
            y_fold[r] = y[Int(index[r])]
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), FN * FD)
        var slot = store[].take_rows(store_ctx, base, FD, Int(index.unsafe_ptr()), n)
        var ctx = process_ctx[_CTX]()
        var plan = gram_plan_current()
        var failure = String("")
        try:
            if which == 0:
                var coef_a = List[Float32](length=FD, fill=0.0)
                var mean_a = List[Float32](length=FD, fill=0.0)
                var ym_a = List[Float64](length=1, fill=0.0)
                var coef_b = List[Float32](length=FD, fill=0.0)
                var mean_b = List[Float32](length=FD, fill=0.0)
                var ym_b = List[Float64](length=1, fill=0.0)
                var sa = linear_gram_fit_host(plan, ctx,
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=fold_x_word(slot)),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(y_fold.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef_a.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(mean_a.unsafe_ptr())),
                    MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(ym_a.unsafe_ptr())),
                    n, FD, Float32(0.5), True)
                var sb = linear_gram_fit_host(plan, ctx,
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(gathered.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(y_fold.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef_b.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(mean_b.unsafe_ptr())),
                    MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(ym_b.unsafe_ptr())),
                    n, FD, Float32(0.5), True)
                require(sa == sb, "Gram status differs")
                _same(coef_a, coef_b, "coef")
                _same(mean_a, mean_b, "column means")
                require(ym_a[0].to_bits() == ym_b[0].to_bits(), "y mean differs")
            else:
                var coef: List[Float32] = [Float32(0.25), Float32(-1.5), Float32(2.0), Float32(0.125)]
                var out_a = List[Float32](length=n, fill=0.0)
                var out_b = List[Float32](length=n, fill=0.0)
                var ra = ols_predict_device(ctx,
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=fold_x_word(slot)),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef.unsafe_ptr())),
                    Int(out_a.unsafe_ptr()), n, FD, Float32(0.75), True)
                var rb = ols_predict_device(ctx,
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(gathered.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef.unsafe_ptr())),
                    Int(out_b.unsafe_ptr()), n, FD, Float32(0.75), True)
                _ = coef^
                require(ra == rb and ra == 0, "finite status differs")
                _same(out_a, out_b, "predictions")
        except error:
            failure = String(error)
        store[].free(store_ctx, slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = y^
        _ = index^
        _ = gathered^
        _ = y_fold^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("vdf_scope_linear_fold device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
