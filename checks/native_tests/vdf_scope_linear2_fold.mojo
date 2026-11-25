# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD X WORD in the rest of the linear member group (lane vdf-scope-linear2, 2026-11-24;
core/fold_x_device.mojo, core/member_fold_x.mojo), in a process with no Python.

Each device case puts a 300 x 4 X once into the fold store, takes one fold's rows (reversed, crossing the base)
on the device and hands the entry `fold_x_word(slot)`; the same entry over the host gather of the same rows is the
oracle (the estimator's own host-input route, whose device and CPU columns the release's one verification
compares). Equal bytes put the fold route in that column.

Host column (CPU_COLUMN_BUILD):
    case 0   `member_fold_take_into` of int32 class codes and of float64 weights == a serial gather, word for word,
             and the view keeps its dtype
    case 1   the host operations entries refuse a fold X word in words: solver (cd) and x_linear

Device column:
    case 0   cd_fit_host (row_major, the device transpose) over the slot == over the gather: coef, intercept, n_iter
    case 1   cd_predict_host over the slot (C order, transposed on the device) == over the host F-order gather
    case 2   x_linear decision_device over the slot == over the gather
    case 3   qn_fit_host (binary logistic) over the slot == over the gather: coef and info words   [owed by
             lane vdf-scope-linear: LogisticRegression, QNRegressor]
    case 4   qn_fit_ovr_host (logistic one-vs-rest, the OvR entry LinearSVC shares) over the slot == over the gather            [owed]
    case 5   qn_decision_function_device over the slot == over the gather (the QN queries)               [owed]
    case 6   DeviceQueryX over the slot == over the gather: the uploaded words and the finite word (the KRR,
             SVM and GPR query and KRR fit uploads)
    case 7   x_decomp DevExec.ols_tsqr_factor (LinearRegression's one-entry TSQR) over the slot == over the gather:
             R, the column means and the y mean bits

FAIL-PROOF EDITS (each must fail the named case): in `member_fold_take_words` read `i[r] + 1` (host 0); delete the
`a[0] < 0` refusal in solver/ops_entry_host.mojo (host 1); in `fold_x_fill` copy from word 1 of the slot (device
0-7); in `cd_predict_host` skip the transpose of a fold slot (device 1); in core/device_query_x.mojo put back the
host `enqueue_copy` (device 6: a negative address).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.member_abi import MemberMatrix, MEMBER_I32, MEMBER_F64
from core.member_fold_x import MemberFoldX, member_fold_take_into
from core.native_descriptor_abi import NativeDescriptorError
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from core.fold_x_device import FOLD_X_STORE, fold_x_word
from solver.estimator import cd_fit_host, cd_predict_host
from solver.ops_entry_device import SOLVER_DEVCTX_SLOT
from solver.ops_entry_host import solver_ops_host_v1
from core.solver_entry_table import SV_OP_CD_PREDICT
from x_linear.ops_entry_host import x_linear_ops_host_v1
from core.linear_entry_table_ext import XL_OP_DECISION
from x_linear.device import decision_device
from x_linear.ops import FP
from glm.estimator import qn_fit_host, qn_fit_ovr_host, qn_decision_function_device
from glm.impl.linear_model.qn import QN_LOSS_LOGISTIC
from core.device_query_x import DeviceQueryX, QueryFold
from x_decomp.device import DevExec


def _drop_error(context: Int, address: Int, count: Int) abi("C") -> NoneType:
    """The refusal's words are not read here (the return code says it refused)."""
    _ = context


comptime FN = 300
comptime FD = 4
comptime _CTX = "MojolearnEstimatorsContextIdenticalV3"
comptime _P = MutPointer[Float32, MutUntrackedOrigin]


def _x(n: Int, d: Int) -> List[Float32]:
    var x = List[Float32](length=n * d, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for j in range(d):  # small-loop(fixture columns)
            x[i * d + j] = Float32(((i * (j + 3) * 7) % 23) - 11) * Float32(0.25) + Float32(j)
    return x^


def _fold(n: Int) -> List[Int64]:
    var rows = List[Int64]()
    for i in range(n - 1, n // 3, -1):  # small-loop(fixture: fold rows, reversed)
        rows.append(Int64(i))
    return rows^


def _same(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(word compare)
        require(a[i].to_bits() == b[i].to_bits(), label + ": word " + String(i) + " differs")


def _p(values: List[Float32]) -> _P:
    return _P(unsafe_from_address=Int(values.unsafe_ptr()))


def _host_case(which: Int) raises:
    if which == 0:
        var codes = List[Int32](length=50, fill=Int32(0))
        var weights = List[Float64](length=50, fill=0.0)
        for i in range(50):  # small-loop(fixture)
            codes[i] = Int32((i * 7) % 5)
            weights[i] = Float64(i) * 0.125 - 1.0
        var index: List[Int64] = [Int64(49), Int64(3), Int64(3), Int64(0), Int64(17)]
        var fold = MemberFoldX()
        fold.index = Int(index.unsafe_ptr())
        fold.rows = len(index)
        var store_c = List[UInt64]()
        var store_w = List[UInt64]()
        var vc = member_fold_take_into(MemberMatrix(Int(codes.unsafe_ptr()), 50, 1, MEMBER_I32), fold, store_c)
        var vw = member_fold_take_into(MemberMatrix(Int(weights.unsafe_ptr()), 50, 1, MEMBER_F64), fold, store_w)
        require(vc.dtype == MEMBER_I32 and vw.dtype == MEMBER_F64 and vc.rows == 5 and vw.rows == 5, "view shape or dtype")
        var gc = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=vc.data)
        var gw = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=vw.data)
        for r in range(5):  # small-loop(oracle compare)
            require(gc[r] == codes[Int(index[r])], "code row " + String(r))
            require(gw[r].to_bits() == weights[Int(index[r])].to_bits(), "weight row " + String(r))
        _ = store_c^
        _ = store_w^
        _ = index^
        return
    # case 1: the host columns refuse a fold word in words
    var sink: NativeDescriptorError = _drop_error
    var result = 0
    var dst = List[Float32](length=4, fill=0.0)
    var a: List[Int] = [fold_x_word(0), Int(dst.unsafe_ptr()), Int(dst.unsafe_ptr())]
    var pa: List[Int] = [2, 2]
    var fa: List[Float64] = [0.0]
    var rc = solver_ops_host_v1(SV_OP_CD_PREDICT, Int(a.unsafe_ptr()), 3, Int(pa.unsafe_ptr()), 2,
        Int(fa.unsafe_ptr()), 1, Int(UnsafePointer(to=result)), 0, sink)
    require(rc != 0, "the solver host column read a fold X word")
    var pd: List[Int] = [2, 2, 1, 0, 0]
    rc = x_linear_ops_host_v1(XL_OP_DECISION, Int(a.unsafe_ptr()), 3, Int(pd.unsafe_ptr()), 5, 0, 0,
        Int(UnsafePointer(to=result)), 0, sink)
    require(rc != 0, "the x_linear host column read a fold X word")
    _ = dst^
    _ = a^
    _ = pa^
    _ = fa^
    _ = pd^


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _host_case(which)

        check_cases("vdf_scope_linear2_fold host", 2, body)


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var x = _x(FN, FD)
        var y = List[Float32](length=FN, fill=0.0)
        for i in range(FN):  # small-loop(fixture targets)
            y[i] = Float32((i * 13) % 17) * Float32(0.5) - Float32(3.0)
        var index = _fold(FN)
        var n = len(index)
        var gathered = List[Float32](length=n * FD, fill=0.0)
        var y_fold = List[Float32](length=n, fill=0.0)
        var labels = List[Float32](length=n, fill=0.0)
        var codes = List[Int32](length=n, fill=Int32(0))
        for r in range(n):  # small-loop(oracle gather)
            for c in range(FD):  # small-loop(oracle gather)
                gathered[r * FD + c] = x[Int(index[r]) * FD + c]
            y_fold[r] = y[Int(index[r])]
            labels[r] = Float32(1.0) if y_fold[r] > 0.0 else Float32(0.0)
            codes[r] = Int32(Int(index[r]) % 3)
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), FN * FD)
        var slot = store[].take_rows(store_ctx, base, FD, Int(index.unsafe_ptr()), n)
        var word = _P(unsafe_from_address=fold_x_word(slot))
        var failure = String("")
        try:
            if which == 0 or which == 1:
                var ctx = process_ctx[SOLVER_DEVCTX_SLOT]()
                if which == 0:
                    var ca = List[Float32](length=FD, fill=0.0)
                    var cb = List[Float32](length=FD, fill=0.0)
                    var ia = List[Float32](length=1, fill=0.0)
                    var ib = List[Float32](length=1, fill=0.0)
                    var na = cd_fit_host(ctx, word, _p(y_fold), _p(ca), _p(ia), n, FD, True, 200, Float32(0.05),
                        Float32(0.5), Float32(1e-4), False, False, _p(y_fold), True)
                    var nb = cd_fit_host(ctx, _p(gathered), _p(y_fold), _p(cb), _p(ib), n, FD, True, 200,
                        Float32(0.05), Float32(0.5), Float32(1e-4), False, False, _p(y_fold), True)
                    require(na == nb, "n_iter differs")
                    _same(ca, cb, "cd coef")
                    _same(ia, ib, "cd intercept")
                else:
                    var coef: List[Float32] = [Float32(0.25), Float32(-1.5), Float32(2.0), Float32(0.125)]
                    var f_order = List[Float32](length=n * FD, fill=0.0)
                    for r in range(n):  # small-loop(oracle transpose)
                        for c in range(FD):  # small-loop(oracle transpose)
                            f_order[c * n + r] = gathered[r * FD + c]
                    var oa = List[Float32](length=n, fill=0.0)
                    var ob = List[Float32](length=n, fill=0.0)
                    cd_predict_host(ctx, word, _p(coef), _p(oa), n, FD, Float32(0.75))
                    cd_predict_host(ctx, _p(f_order), _p(coef), _p(ob), n, FD, Float32(0.75))
                    _same(oa, ob, "cd predictions")
            elif which == 2:
                var wb: List[Float32] = [Float32(0.5), Float32(-0.25), Float32(1.5), Float32(2.0), Float32(-0.125)]
                var oa = List[Float32](length=n, fill=0.0)
                var ob = List[Float32](length=n, fill=0.0)
                decision_device(FP(unsafe_from_address=Int(word)), FP(unsafe_from_address=Int(wb.unsafe_ptr())), n,
                    FD, 1, 0, FP(unsafe_from_address=Int(oa.unsafe_ptr())))
                decision_device(FP(unsafe_from_address=Int(gathered.unsafe_ptr())),
                    FP(unsafe_from_address=Int(wb.unsafe_ptr())), n, FD, 1, 0, FP(unsafe_from_address=Int(ob.unsafe_ptr())))
                _ = wb^
                _same(oa, ob, "x_linear decisions")
            elif which == 3 or which == 4:
                var ctx = process_ctx[_CTX]()
                var ca = List[Float32](length=64, fill=0.0)
                var cb = List[Float32](length=64, fill=0.0)
                var ia = List[Float32](length=64, fill=0.0)
                var ib = List[Float32](length=64, fill=0.0)
                if which == 3:
                    qn_fit_host(ctx, word, _p(labels), _p(ca), _p(ia), n, FD, 2, 0.0, 1.0 / Float64(n), 1e-4, 1e-5,
                        100, 20, 5, True, True, False, _p(labels))
                    qn_fit_host(ctx, _p(gathered), _p(labels), _p(cb), _p(ib), n, FD, 2, 0.0, 1.0 / Float64(n), 1e-4,
                        1e-5, 100, 20, 5, True, True, False, _p(labels))
                else:
                    var cp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(codes.unsafe_ptr()))
                    qn_fit_ovr_host(ctx, word, cp, _p(ca), _p(ia), n, FD, 0, 3, 0.0, 1.0 / Float64(n), 1e-4, 1e-5,
                        100, 20, 5, True, True, QN_LOSS_LOGISTIC, False, _p(labels))
                    qn_fit_ovr_host(ctx, _p(gathered), cp, _p(cb), _p(ib), n, FD, 0, 3, 0.0, 1.0 / Float64(n), 1e-4,
                        1e-5, 100, 20, 5, True, True, QN_LOSS_LOGISTIC, False, _p(labels))
                _same(ca, cb, "QN coef")
                _same(ia, ib, "QN info")
            elif which == 5:
                var ctx = process_ctx[_CTX]()
                var coef: List[Float32] = [Float32(0.25), Float32(-1.5), Float32(2.0), Float32(0.125), Float32(0.5)]
                var sa = ctx.enqueue_create_buffer[DType.float32](n)
                var sb = ctx.enqueue_create_buffer[DType.float32](n)
                qn_decision_function_device(ctx, word, _p(coef), sa, n, FD, True, 2)
                qn_decision_function_device(ctx, _p(gathered), _p(coef), sb, n, FD, True, 2)
                var ha = List[Float32](length=n, fill=0.0)
                var hb = List[Float32](length=n, fill=0.0)
                ctx.enqueue_copy(dst_ptr=ha.unsafe_ptr(), src_buf=sa)
                ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=sb)
                ctx.synchronize()
                _same(ha, hb, "QN decisions")
                _ = sa^
                _ = sb^
            elif which == 6:
                var ctx = process_ctx[_CTX]()
                var qa = DeviceQueryX(ctx, fold_x_word(slot), n, FD, QueryFold.none())
                var qb = DeviceQueryX(ctx, Int(gathered.unsafe_ptr()), n, FD, QueryFold.none())
                var ha = List[Float32](length=n * FD, fill=0.0)
                var hb = List[Float32](length=n * FD, fill=0.0)
                ctx.enqueue_copy(dst_ptr=ha.unsafe_ptr(), src_buf=qa.x)
                ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=qb.x)
                ctx.synchronize()
                _same(ha, hb, "query X words")
                require(qa.finite_word() == qb.finite_word() and qa.finite_word() == 0, "finite word differs")
                _ = qa^
                _ = qb^
            else:
                var ra = List[Float32](length=(FD + 1) * (FD + 1), fill=0.0)
                var rb = List[Float32](length=(FD + 1) * (FD + 1), fill=0.0)
                var ma = List[Float32](length=FD, fill=0.0)
                var mb = List[Float32](length=FD, fill=0.0)
                var ya = List[UInt64](length=1, fill=UInt64(0))
                var yb = List[UInt64](length=1, fill=UInt64(0))
                DevExec.ols_tsqr_factor(word, _p(y_fold), _p(ra), _p(ma),
                    MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=Int(ya.unsafe_ptr())), n, FD)
                DevExec.ols_tsqr_factor(_p(gathered), _p(y_fold), _p(rb), _p(mb),
                    MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=Int(yb.unsafe_ptr())), n, FD)
                _same(ra, rb, "TSQR R")
                _same(ma, mb, "column means")
                require(ya[0] == yb[0], "y mean differs")
        except error:
            failure = String(error)
        store[].free(store_ctx, slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = y^
        _ = index^
        _ = gathered^
        _ = y_fold^
        _ = labels^
        _ = codes^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("vdf_scope_linear2_fold device", 8, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
