# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE PER-INPUT PINS (lane vdf-scope-pins, 2026-11-24; core/member_abi.mojo `MemberCall.y_pin`,
core/member_fold_x.mojo `member_fold_column_open`), in a process with no Python.

A member reading y / sample_weight from their pins takes the fold's rows of the scope-resident column ON THE
DEVICE (`take_rows` of the x_prep store, `row_words` 4-byte words a row) and hands the entry the slot's word; the
oracle is the host gather of the same rows (the path every member took before) through the same entry.

Host column (CPU_COLUMN_BUILD):
    case 0   `member_call_of` hands the plan's y / weights tokens to the call; `member_y_pin_settle` and
             `member_weights_pin_settle` answer 1 only when the member wrote the kept word, and clear both words;
             `member_pins_offered` sees each of the three tokens
    case 1   the solver host column refuses a fold y word and a fold sample_weight word in CD_FIT, in words

Device column:
    case 0   the fold's rows of a float32 y column (1 word a row) and of a float64 weight column (2 words a row)
             taken on the device == the host gather, word for word
    case 1   cd_fit_host (weighted, row_major) over fold X, fold y and fold weights slots == over the host
             gathers: coef, intercept and n_iter bits

FAIL-PROOF EDITS (each must fail the named case): in `member_call_of` drop `call.y_pin = plan.y_pin` (host 0); in
`member_y_pin_settle` answer 1 for any nonzero `y_pin` (host 0); delete the fold y / weights refusal in
solver/ops_entry_host.mojo (host 1); in solver/estimator.mojo put back `ctx.enqueue_copy(dst_buf=y, src_ptr=y_ptr)`
(device 1: a negative address) or read the weights' slot from word 1 (device 1).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.member_abi import (
    MemberPlan, MemberCall, member_call_of, member_y_pin_settle, member_weights_pin_settle, member_pins_offered,
)
from core.native_descriptor_abi import NativeDescriptorError
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from core.fold_x_device import FOLD_X_STORE, fold_x_word
from solver.estimator import cd_fit_host
from solver.ops_entry_device import SOLVER_DEVCTX_SLOT
from solver.ops_entry_host import solver_ops_host_v1
from core.solver_entry_table import SV_OP_CD_FIT

comptime FN = 240
comptime FD = 3
comptime _P = MutPointer[Float32, MutUntrackedOrigin]


def _p(values: List[Float32]) -> _P:
    return _P(unsafe_from_address=Int(values.unsafe_ptr()))


def _same(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(word compare)
        require(a[i].to_bits() == b[i].to_bits(), label + ": word " + String(i) + " differs")


def _host_case(case: Int) raises:
    if case == 0:
        var plan = MemberPlan()
        plan.pin = 11
        plan.y_pin = 22
        plan.weights_pin = 33
        var call = member_call_of(plan)
        require(call.pin == 11 and call.y_pin == 22 and call.weights_pin == 33, "the plan's tokens reach the call")
        require(member_pins_offered(call), "three tokens offered")
        # no member wrote a kept word: nothing taken, the words cleared
        require(member_y_pin_settle(call) == 0 and call.y_pin == 0 and call.y_pin_kept == 0, "y not taken")
        call.weights_pin_kept = 1
        require(member_weights_pin_settle(call) == 1, "weights taken")
        require(call.weights_pin == 0 and call.weights_pin_kept == 0, "weights words cleared")
        call.pin = 0
        require(not member_pins_offered(call), "nothing offered after the settles")
        call.y_pin_kept = 1  # a kept word without an offered token is no take
        require(member_y_pin_settle(call) == 0, "no token, no take")
        return
    # case 1: the solver host column refuses a fold y word and a fold weights word
    var sink = NativeDescriptorError()
    var result = 0
    var x = List[Float32](length=8, fill=1.0)
    var out = List[Float32](length=8, fill=0.0)
    var p: List[Int] = [4, 2, 1, 10, 0, 1, 1]
    var f: List[Float64] = [0.1, 0.5, 1e-4]
    var ay: List[Int] = [Int(x.unsafe_ptr()), fold_x_word(0), Int(out.unsafe_ptr()), Int(out.unsafe_ptr()),
                         Int(x.unsafe_ptr())]
    var rc = solver_ops_host_v1(SV_OP_CD_FIT, Int(ay.unsafe_ptr()), 5, Int(p.unsafe_ptr()), 7, Int(f.unsafe_ptr()),
        3, Int(UnsafePointer(to=result)), 0, sink)
    require(rc != 0, "the solver host column read a fold y word")
    var aw: List[Int] = [Int(x.unsafe_ptr()), Int(x.unsafe_ptr()), Int(out.unsafe_ptr()), Int(out.unsafe_ptr()),
                         fold_x_word(0)]
    rc = solver_ops_host_v1(SV_OP_CD_FIT, Int(aw.unsafe_ptr()), 5, Int(p.unsafe_ptr()), 7, Int(f.unsafe_ptr()),
        3, Int(UnsafePointer(to=result)), 0, sink)
    require(rc != 0, "the solver host column read a fold sample_weight word")
    _ = x^
    _ = out^
    _ = p^
    _ = f^
    _ = ay^
    _ = aw^


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _host_case(case)

        check_cases("vdf_scope_pins host", 2, body)


def _device_case(case: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var x = List[Float32](length=FN * FD, fill=0.0)
        var y = List[Float32](length=FN, fill=0.0)
        var w = List[Float32](length=FN, fill=0.0)
        var w64 = List[Float64](length=FN, fill=0.0)
        for i in range(FN):  # small-loop(fixture rows)
            for j in range(FD):  # small-loop(fixture columns)
                x[i * FD + j] = Float32(((i * (j + 5) * 3) % 19) - 9) * Float32(0.25) + Float32(j)
            y[i] = Float32((i * 11) % 13) * Float32(0.5) - Float32(2.0)
            w[i] = Float32(0.5) + Float32(i % 7) * Float32(0.125)
            w64[i] = Float64(i) * 0.0625 - 3.0
        var index = List[Int64]()
        for i in range(FN - 1, FN // 4, -2):  # small-loop(fixture: fold rows, reversed, crossing the base)
            index.append(Int64(i))
        index.append(Int64(5))
        index.append(Int64(5))  # a repeat (a bootstrap)
        var n = len(index)
        var gx = List[Float32](length=n * FD, fill=0.0)
        var gy = List[Float32](length=n, fill=0.0)
        var gw = List[Float32](length=n, fill=0.0)
        for r in range(n):  # small-loop(oracle gather)
            for c in range(FD):  # small-loop(oracle gather)
                gx[r * FD + c] = x[Int(index[r]) * FD + c]
            gy[r] = y[Int(index[r])]
            gw[r] = w[Int(index[r])]
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var ids = List[Int]()
        var failure = String("")
        try:
            if case == 0:
                var by = store[].put(store_ctx, Int(y.unsafe_ptr()), FN)
                ids.append(by)
                var bw = store[].put(store_ctx, Int(w64.unsafe_ptr()), 2 * FN)  # float64: two words a row
                ids.append(bw)
                var sy = store[].take_rows(store_ctx, by, 1, Int(index.unsafe_ptr()), n)
                ids.append(sy)
                var sw = store[].take_rows(store_ctx, bw, 2, Int(index.unsafe_ptr()), n)
                ids.append(sw)
                var ry = List[Float32](length=n, fill=0.0)
                var rw = List[Float64](length=n, fill=0.0)
                store[].read(store_ctx, sy, Int(ry.unsafe_ptr()), n)
                store[].read(store_ctx, sw, Int(rw.unsafe_ptr()), 2 * n)
                _same(ry, gy, "fold y rows")
                for r in range(n):  # small-loop(oracle compare)
                    require(rw[r].to_bits() == w64[Int(index[r])].to_bits(), "fold float64 weight row " + String(r))
            else:
                var bx = store[].put(store_ctx, Int(x.unsafe_ptr()), FN * FD)
                ids.append(bx)
                var by = store[].put(store_ctx, Int(y.unsafe_ptr()), FN)
                ids.append(by)
                var bw = store[].put(store_ctx, Int(w.unsafe_ptr()), FN)
                ids.append(bw)
                var sx = store[].take_rows(store_ctx, bx, FD, Int(index.unsafe_ptr()), n)
                ids.append(sx)
                var sy = store[].take_rows(store_ctx, by, 1, Int(index.unsafe_ptr()), n)
                ids.append(sy)
                var sw = store[].take_rows(store_ctx, bw, 1, Int(index.unsafe_ptr()), n)
                ids.append(sw)
                var ctx = process_ctx[SOLVER_DEVCTX_SLOT]()
                var ca = List[Float32](length=FD, fill=0.0)
                var cb = List[Float32](length=FD, fill=0.0)
                var ia = List[Float32](length=1, fill=0.0)
                var ib = List[Float32](length=1, fill=0.0)
                var na = cd_fit_host(ctx, _P(unsafe_from_address=fold_x_word(sx)), _P(unsafe_from_address=fold_x_word(sy)),
                    _p(ca), _p(ia), n, FD, True, 200, Float32(0.05), Float32(0.5), Float32(1e-4), False, True,
                    _P(unsafe_from_address=fold_x_word(sw)), True)
                var nb = cd_fit_host(ctx, _p(gx), _p(gy), _p(cb), _p(ib), n, FD, True, 200, Float32(0.05),
                    Float32(0.5), Float32(1e-4), False, True, _p(gw), True)
                require(na == nb, "n_iter differs")
                _same(ca, cb, "weighted cd coef")
                _same(ia, ib, "weighted cd intercept")
        except error:
            failure = String(error)
        for i in range(len(ids)):  # small-loop(the case's slots)
            try:
                store[].free(store_ctx, ids[i])
            except:
                pass
        _ = x^
        _ = y^
        _ = w^
        _ = w64^
        _ = index^
        _ = gx^
        _ = gy^
        _ = gw^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _device_case(case)

        check_cases_serial("vdf_scope_pins device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
