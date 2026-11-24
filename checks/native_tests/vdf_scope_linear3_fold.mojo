# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD X WORD on the routes lane vdf-scope-linear3 moved to the device (2026-11-24;
core/fold_x_device.mojo, core/member_fold_x.mojo), in a process with no Python.

Each device case puts a 300 x 4 X (and its y and weights) once into the fold store, takes one fold's rows
(reversed, crossing the base) on the device and hands the entry `fold_x_word(slot)`; the same entry over the host
gather of the same rows is the oracle (the host-input route, whose device and CPU columns the release's one
verification compares). Equal bytes put the fold route in that column.

Host column (CPU_COLUMN_BUILD):
    case 0   the host columns refuse a fold word in words: the solver's cd fit for a y word and a weights word,
             x_decomp HostExec.tsqr_factor for an X word

Device column:
    case 0   cd_fit_host, WEIGHTED, with X, y and the weights all fold slots == over the host gathers: coef,
             intercept, n_iter (core/cd_member.mojo `member_fold_column_open`)
    case 1   x_linear_finite_device (the FAST build's scan) over a finite slot passes; over a slot holding a NaN
             it raises the binding's wording
    case 2   ols_prep_device (weighted means, the LF_PREP centering and row scaling) over the slot == over the
             gather: work X (a device slot for the fold, lane vdf-scope-linear4, read back), work y, both means
    case 3   DevExec.tsqr_factor (nrhs 1: LinearRegression's uncentered XD_TSQR_R) over the slot == over the gather
    case 4   device_buffer_any_negative: True on the raw fold (it has negative cells), False on its absolute
             values with -0.0 cells kept, True again with one negative cell
             (the chi2 kernels' x >= 0 test of KernelRidge and Nystroem)
    case 5   scale_gamma_limbs_device (gamma='scale') over the slot == over the gather: every limb word

Owed (need the x_prep transport entry of a loaded image, so a member-level GPU run): RidgeCV's split take
(x_linear/door_run.mojo `_xl_split_take`), the SVC pair / Platt takes and `_support_from_pairs`
(svm/svc_kernel_fit_run.mojo), the KRR and Nystroem entries' `krr_chi2_fold_check` / `nys_chi2_fold_check`.

FAIL-PROOF EDITS (each must fail the named case): in solver/estimator.mojo put back
`enqueue_copy(dst_buf=y, src_ptr=y_ptr)` (device 0: a negative address); delete the `a[1] < 0` refusal in
solver/ops_entry_host.mojo (host 0); drop the `Int(p) < 0` branch of `x_linear_finite_device` (device 1); in
glm/impl/center_device.mojo put back the host `enqueue_copy` of X (device 2); `_up` for `_up_fold_x` in
DevExec.tsqr_factor (device 3); test `bits & 0x80000000` alone in `_negative_u32_kernel` (device 4, the -0.0
fixture); drop the fold branch of `scale_gamma_limbs_device` (device 5).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.native_descriptor_abi import NativeDescriptorError
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from core.fold_x_device import FOLD_X_STORE, fold_x_word
from core.fold_x_word import fold_x_slot
from solver.estimator import cd_fit_host
from solver.ops_entry_device import SOLVER_DEVCTX_SLOT
from solver.ops_entry_host import solver_ops_host_v1
from core.solver_entry_table import SV_OP_CD_FIT
from x_linear.x_linear_input_impl import x_linear_finite_device
from x_linear.ops import FP
from glm.impl.ols_prep_device import ols_prep_device, OLS_PREP_WEIGHTED
from x_decomp.device import DevExec
from x_decomp.host import HostExec
from core.input_device import device_buffer_any_negative
from svm.impl.scale_gamma_device import scale_gamma_limbs_device
from svm.impl.scale_gamma_limbs import SG_SLOTS

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


def _host_case(case: Int) raises:
    var sink = NativeDescriptorError()
    var result = 0
    var buf = List[Float32](length=16, fill=0.0)
    var host = Int(buf.unsafe_ptr())
    var pi: List[Int] = [4, 2, 1, 10, 0, 1, 1]
    var pf: List[Float64] = [0.1, 0.5, 1e-4]
    # a y fold word, then a weights fold word (has_sw on)
    var ay: List[Int] = [host, fold_x_word(0), host, host, host]
    var rc = solver_ops_host_v1(SV_OP_CD_FIT, Int(ay.unsafe_ptr()), 5, Int(pi.unsafe_ptr()), 7,
        Int(pf.unsafe_ptr()), 3, Int(UnsafePointer(to=result)), 0, sink)
    require(rc != 0, "the solver host column read a fold y word")
    var aw: List[Int] = [host, host, host, host, fold_x_word(1)]
    rc = solver_ops_host_v1(SV_OP_CD_FIT, Int(aw.unsafe_ptr()), 5, Int(pi.unsafe_ptr()), 7,
        Int(pf.unsafe_ptr()), 3, Int(UnsafePointer(to=result)), 0, sink)
    require(rc != 0, "the solver host column read a fold weights word")
    var refused = False
    try:
        _ = HostExec.tsqr_factor(_P(unsafe_from_address=fold_x_word(0)), _p(buf), _p(buf), 4, 1, 1, False)
    except:
        refused = True
    require(refused, "x_decomp's host tsqr read a fold X word")
    _ = buf^
    _ = ay^
    _ = aw^
    _ = pi^
    _ = pf^


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _host_case(case)

        check_cases("vdf_scope_linear3_fold host", 1, body)


def _device_case(case: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var x = _x(FN, FD)
        var y = List[Float32](length=FN, fill=0.0)
        var w = List[Float32](length=FN, fill=0.0)
        for i in range(FN):  # small-loop(fixture targets and weights)
            y[i] = Float32((i * 13) % 17) * Float32(0.5) - Float32(3.0)
            w[i] = Float32(1 + (i * 5) % 7) * Float32(0.25)
        if case == 1:
            x[(FN - 2) * FD + 1] = Float32(0.0) / Float32(0.0)  # inside the fold (row FN - 2)
        if case == 4:
            for j in range(FD):  # small-loop(fixture: -0.0 cells)
                x[(FN - 3) * FD + j] = Float32(-0.0)
        var index = _fold(FN)
        var n = len(index)
        var gathered = List[Float32](length=n * FD, fill=0.0)
        var y_fold = List[Float32](length=n, fill=0.0)
        var w_fold = List[Float32](length=n, fill=0.0)
        for r in range(n):  # small-loop(oracle gather)
            for c in range(FD):  # small-loop(oracle gather)
                gathered[r * FD + c] = x[Int(index[r]) * FD + c]
            y_fold[r] = y[Int(index[r])]
            w_fold[r] = w[Int(index[r])]
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), FN * FD)
        var base_y = store[].put(store_ctx, Int(y.unsafe_ptr()), FN)
        var base_w = store[].put(store_ctx, Int(w.unsafe_ptr()), FN)
        var slot = store[].take_rows(store_ctx, base, FD, Int(index.unsafe_ptr()), n)
        var slot_y = store[].take_rows(store_ctx, base_y, 1, Int(index.unsafe_ptr()), n)
        var slot_w = store[].take_rows(store_ctx, base_w, 1, Int(index.unsafe_ptr()), n)
        var word = _P(unsafe_from_address=fold_x_word(slot))
        var failure = String("")
        try:
            if case == 0:
                var ctx = process_ctx[SOLVER_DEVCTX_SLOT]()
                var ca = List[Float32](length=FD, fill=0.0)
                var cb = List[Float32](length=FD, fill=0.0)
                var ia = List[Float32](length=1, fill=0.0)
                var ib = List[Float32](length=1, fill=0.0)
                var na = cd_fit_host(ctx, word, _P(unsafe_from_address=fold_x_word(slot_y)), _p(ca), _p(ia), n, FD,
                    True, 200, Float32(0.05), Float32(0.5), Float32(1e-4), False, True,
                    _P(unsafe_from_address=fold_x_word(slot_w)), True)
                var nb = cd_fit_host(ctx, _p(gathered), _p(y_fold), _p(cb), _p(ib), n, FD, True, 200,
                    Float32(0.05), Float32(0.5), Float32(1e-4), False, True, _p(w_fold), True)
                require(na == nb, "n_iter differs")
                _same(ca, cb, "weighted cd coef")
                _same(ia, ib, "weighted cd intercept")
            elif case == 1:
                var raised = String("")
                try:
                    x_linear_finite_device(FP(unsafe_from_address=Int(word)), n * FD, "X")
                except error:
                    raised = String(error)
                require(raised == "mojolearn: X contains NaN or infinity", "the NaN fold slot passed: " + raised)
                x_linear_finite_device(FP(unsafe_from_address=Int(gathered.unsafe_ptr())), FD, "X")  # row 0 finite
            elif case == 2:
                var ctx = process_ctx[_CTX]()
                # Lane vdf-scope-linear4: a fold X's prepared X stays a device slot; work_x
                # receives its fold word (one Int64), read back here for the compare.
                var box = List[Int64](length=1, fill=Int64(0))
                var wxa = List[Float32](length=n * FD, fill=0.0)
                var wxb = List[Float32](length=n * FD, fill=0.0)
                var wya = List[Float32](length=n, fill=0.0)
                var wyb = List[Float32](length=n, fill=0.0)
                var m64a = List[Float64](length=FD, fill=0.0)
                var m64b = List[Float64](length=FD, fill=0.0)
                var m32a = List[Float32](length=FD, fill=0.0)
                var m32b = List[Float32](length=FD, fill=0.0)
                var y64a = List[Float64](length=1, fill=0.0)
                var y64b = List[Float64](length=1, fill=0.0)
                var y32a = List[Float32](length=1, fill=0.0)
                var y32b = List[Float32](length=1, fill=0.0)
                ols_prep_device(ctx, Int(word), Int(y_fold.unsafe_ptr()), Int(w_fold.unsafe_ptr()),
                    Int(box.unsafe_ptr()), Int(wya.unsafe_ptr()), Int(m64a.unsafe_ptr()), Int(m32a.unsafe_ptr()),
                    Int(y64a.unsafe_ptr()), Int(y32a.unsafe_ptr()), n, FD, True, OLS_PREP_WEIGHTED)
                require(Int(box[0]) < 0, "the fold prep left no prepared X slot")
                store[].read(ctx, fold_x_slot(Int(box[0])), Int(wxa.unsafe_ptr()), n * FD)
                store[].free(ctx, fold_x_slot(Int(box[0])))
                ols_prep_device(ctx, Int(gathered.unsafe_ptr()), Int(y_fold.unsafe_ptr()), Int(w_fold.unsafe_ptr()),
                    Int(wxb.unsafe_ptr()), Int(wyb.unsafe_ptr()), Int(m64b.unsafe_ptr()), Int(m32b.unsafe_ptr()),
                    Int(y64b.unsafe_ptr()), Int(y32b.unsafe_ptr()), n, FD, True, OLS_PREP_WEIGHTED)
                _same(wxa, wxb, "prepared X")
                _same(wya, wyb, "prepared y")
                _same(m32a, m32b, "float32 means")
                for j in range(FD):  # small-loop(binary64 means)
                    require(m64a[j].to_bits() == m64b[j].to_bits(), "binary64 mean " + String(j))
                require(y64a[0].to_bits() == y64b[0].to_bits(), "y mean differs")
            elif case == 3:
                var ra = List[Float32](length=(FD + 1) * (FD + 1), fill=0.0)
                var rb = List[Float32](length=(FD + 1) * (FD + 1), fill=0.0)
                _ = DevExec.tsqr_factor(word, _p(y_fold), _p(ra), n, FD, 1, False)
                _ = DevExec.tsqr_factor(_p(gathered), _p(y_fold), _p(rb), n, FD, 1, False)
                _same(ra, rb, "uncentered TSQR R")
            elif case == 4:
                var ctx = process_ctx[_CTX]()
                var dg = ctx.enqueue_create_buffer[DType.float32](n * FD)
                ctx.enqueue_copy(dst_buf=dg, src_ptr=_p(gathered))
                require(device_buffer_any_negative(ctx, dg, n * FD), "the raw fold (negative cells) passed")
                var shifted = List[Float32](length=n * FD, fill=0.0)
                for i in range(n * FD):  # small-loop(fixture: the fold made non-negative, -0.0 kept)
                    var v = gathered[i]
                    shifted[i] = v if v.to_bits() == Float32(-0.0).to_bits() else abs(v)
                ctx.enqueue_copy(dst_buf=dg, src_ptr=_p(shifted))
                require(not device_buffer_any_negative(ctx, dg, n * FD), "a non-negative fold (with -0.0) flagged")
                shifted[n * FD - 1] = Float32(-0.5)
                ctx.enqueue_copy(dst_buf=dg, src_ptr=_p(shifted))
                require(device_buffer_any_negative(ctx, dg, n * FD), "a negative cell passed")
                _ = dg^
                _ = shifted^
            else:
                var ctx = process_ctx[_CTX]()
                var la = List[Int64](length=SG_SLOTS, fill=Int64(0))
                var lb = List[Int64](length=SG_SLOTS, fill=Int64(0))
                scale_gamma_limbs_device(ctx, Int(word), n * FD, Int(la.unsafe_ptr()))
                scale_gamma_limbs_device(ctx, Int(gathered.unsafe_ptr()), n * FD, Int(lb.unsafe_ptr()))
                for k in range(SG_SLOTS):  # small-loop(limb words)
                    require(la[k] == lb[k], "limb " + String(k) + " differs")
        except error:
            failure = String(error)
        store[].free(store_ctx, slot)
        store[].free(store_ctx, slot_y)
        store[].free(store_ctx, slot_w)
        store[].free(store_ctx, base)
        store[].free(store_ctx, base_y)
        store[].free(store_ctx, base_w)
        _ = x^
        _ = y^
        _ = w^
        _ = index^
        _ = gathered^
        _ = y_fold^
        _ = w_fold^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _device_case(case)

        check_cases_serial("vdf_scope_linear3_fold device", 6, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
