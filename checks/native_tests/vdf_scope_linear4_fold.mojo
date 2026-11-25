# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD X WORD on the routes lane vdf-scope-linear4 moved to the device (2026-11-24;
core/fold_x_device.mojo, core/member_fold_x.mojo), in a process with no Python.

Each device case puts a 300 x 4 X (and its y) once into the fold store, takes one fold's rows (reversed, crossing
the base) on the device and hands the entry the slot's word; the same entry over the host gather of the same rows
is the oracle (the host-input route, whose device and CPU columns the release's one verification compares). Equal
bytes put the fold route in that column.

Host column (CPU_COLUMN_BUILD):
    case 0   the host columns refuse a fold X word in words: the linear fit entry (LF_PREP) and the
             weighted / one-column Ridge solve

Device column:
    case 0   ols_prep_device, unweighted (separate means, LinearRegression's LF_PREP centering): the prepared X
             of the fold stays a device slot (its word in work_x) == the host gather's prepared X, y and means
    case 1   ridge_fit_device_v1 (Ridge's weighted / one-column / several-target solve) over the X slot ==
             over the gather, weighted, two targets: every result word
    case 2   ols_predict_device_targets (a multi-target Ridge member's predict) over the slot == over the gather
    case 3   member_r2_score_slot (a cd fold score, y in a slot) == member_r2_score over the host gather of y
             (the same `_Reg` program, same entry): the same float64 bits
    case 4   the limbs entry's non-finite word over a slot (RBFSampler's fold fit test): 0 on the finite fold,
             nonzero with one NaN inside the fold

FAIL-PROOF EDITS (each must fail the named case): drop the `dst < 0` branch of `center_device`
(glm/impl/center_device.mojo; device 0, a negative host address); put back `enqueue_copy(dst_buf=x, src_ptr=x_ptr)`
in `ols_predict_device_targets` (device 2); fill the y range from the host arena in `member_r2_score_slot`
(device 3: zeros, a different R^2); delete the `a[0] < 0` refusal of glm/host/linear_fit_native.mojo (host 0).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_word, fold_x_slot
from core.linear_fit_types import LF_PREP, ridge_fit_result_words
from glm.host.linear_fit_native import linear_fit_host_v1
from x_linear.ridge_descriptor_host import ridge_fit_host_v1
from x_linear.ridge_descriptor_device import ridge_fit_device_v1
from glm.impl.ols_prep_device import ols_prep_device, OLS_PREP_SEPARATE
from core.p3_linear_impl import ols_predict_device_targets
from core.entry_table import EntryRef, entry_ref_word
from core.linear_score_run import member_r2_score, member_r2_score_slot
from x_metrics.descriptor_device import metrics_program_device_v1
from x_metrics.descriptor_schema import MP_MODE, MP_VENDOR, MP_OPERATION_COUNT
from svm.impl.scale_gamma_device import scale_gamma_limbs_device
from svm.impl.scale_gamma_limbs import SG_SLOTS
from core.scale_gamma_exact import SGX_BAD


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
    var sink: NativeDescriptorError = _drop_error
    var result = 0
    var buf = List[Float32](length=64, fill=0.0)
    var host = Int(buf.unsafe_ptr())
    var prep: List[Int] = [fold_x_word(0), host, 0, host, host, host, host, host, host]
    var pi: List[Int] = [4, 2, 1, 0]
    var rc = linear_fit_host_v1(LF_PREP, Int(prep.unsafe_ptr()), 9, Int(pi.unsafe_ptr()), 4, 0, 0,
        Int(UnsafePointer(to=result)), 0, sink)
    require(rc != 0, "the host linear fit column read a fold X word")
    var ridge: List[Int] = [fold_x_word(0), host, host]
    var ri: List[Int] = [4, 2, 1, 0]
    var rf: List[Float64] = [1.0]
    rc = ridge_fit_host_v1(0, Int(ridge.unsafe_ptr()), 3, Int(ri.unsafe_ptr()), 4, Int(rf.unsafe_ptr()), 1,
        Int(UnsafePointer(to=result)), 0, sink)
    require(rc != 0, "the host Ridge solve column read a fold X word")
    _ = buf^
    _ = prep^
    _ = pi^
    _ = ridge^
    _ = ri^
    _ = rf^


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _host_case(which)

        check_cases("vdf_scope_linear4_fold host", 1, body)


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var x = _x(FN, FD)
        var y = List[Float32](length=FN, fill=0.0)
        for i in range(FN):  # small-loop(fixture targets)
            y[i] = Float32((i * 13) % 17) * Float32(0.5) - Float32(3.0)
        if which == 4:
            x[(FN - 2) * FD + 1] = Float32(0.0) / Float32(0.0)  # inside the fold (row FN - 2)
        var index = _fold(FN)
        var n = len(index)
        var gathered = List[Float32](length=n * FD, fill=0.0)
        var y_fold = List[Float32](length=n, fill=0.0)
        for r in range(n):  # small-loop(oracle gather)
            for c in range(FD):  # small-loop(oracle gather)
                gathered[r * FD + c] = x[Int(index[r]) * FD + c]
            y_fold[r] = y[Int(index[r])]
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), FN * FD)
        var base_y = store[].put(store_ctx, Int(y.unsafe_ptr()), FN)
        var slot = store[].take_rows(store_ctx, base, FD, Int(index.unsafe_ptr()), n)
        var slot_y = store[].take_rows(store_ctx, base_y, 1, Int(index.unsafe_ptr()), n)
        var word = fold_x_word(slot)
        var failure = String("")
        try:
            if which == 0:
                var ctx = process_ctx[_CTX]()
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
                ols_prep_device(ctx, word, Int(y_fold.unsafe_ptr()), 0, Int(box.unsafe_ptr()), Int(wya.unsafe_ptr()),
                    Int(m64a.unsafe_ptr()), Int(m32a.unsafe_ptr()), Int(y64a.unsafe_ptr()), Int(y32a.unsafe_ptr()),
                    n, FD, True, OLS_PREP_SEPARATE)
                _ = y32a^
                ols_prep_device(ctx, Int(gathered.unsafe_ptr()), Int(y_fold.unsafe_ptr()), 0, Int(wxb.unsafe_ptr()),
                    Int(wyb.unsafe_ptr()), Int(m64b.unsafe_ptr()), Int(m32b.unsafe_ptr()), Int(y64b.unsafe_ptr()),
                    Int(y32b.unsafe_ptr()), n, FD, True, OLS_PREP_SEPARATE)
                _ = y32b^
                require(Int(box[0]) < 0, "the fold prep left no prepared X slot")
                store[].read(ctx, fold_x_slot(Int(box[0])), Int(wxa.unsafe_ptr()), n * FD)
                store[].free(ctx, fold_x_slot(Int(box[0])))
                _same(wxa, wxb, "prepared X")
                _same(wya, wyb, "prepared y")
                _same(m32a, m32b, "float32 means")
                for j in range(FD):  # small-loop(binary64 means)
                    require(m64a[j].to_bits() == m64b[j].to_bits(), "binary64 mean " + String(j))
                require(y64a[0].to_bits() == y64b[0].to_bits(), "y mean differs")
            elif which == 1:
                # Two targets and the weights after them (rows x (T + 1) words).
                comptime T = 2
                var yw = List[Float32](length=n * (T + 1), fill=0.0)
                for r in range(n):  # small-loop(fixture targets and weights)
                    for t in range(T):  # small-loop(fixture targets)
                        yw[r * T + t] = y_fold[r] * Float32(t + 1)
                    yw[n * T + r] = Float32(1 + (r * 5) % 7) * Float32(0.25)
                var width = ridge_fit_result_words(T, FD)
                var va = List[Float32](length=width, fill=0.0)
                var vb = List[Float32](length=width, fill=0.0)
                var sink: NativeDescriptorError = _drop_error
                var result = 0
                var pi: List[Int] = [n, FD, 1, 1, T]
                var pf: List[Float64] = [0.5]
                var aa: List[Int] = [word, Int(yw.unsafe_ptr()), Int(va.unsafe_ptr())]
                var ab: List[Int] = [Int(gathered.unsafe_ptr()), Int(yw.unsafe_ptr()), Int(vb.unsafe_ptr())]
                var rc = ridge_fit_device_v1(0, Int(aa.unsafe_ptr()), 3, Int(pi.unsafe_ptr()), 5,
                    Int(pf.unsafe_ptr()), 1, Int(UnsafePointer(to=result)), 0, sink)
                require(rc == 0, "the weighted Ridge solve refused the fold slot")
                rc = ridge_fit_device_v1(0, Int(ab.unsafe_ptr()), 3, Int(pi.unsafe_ptr()), 5,
                    Int(pf.unsafe_ptr()), 1, Int(UnsafePointer(to=result)), 0, sink)
                require(rc == 0, "the weighted Ridge solve refused the gather")
                _same(va, vb, "weighted two-target Ridge words")
                _ = aa^
                _ = ab^
                _ = pi^
                _ = pf^
                _ = yw^
            elif which == 2:
                var ctx = process_ctx[_CTX]()
                comptime T = 3
                var coef = List[Float32](length=T * FD, fill=0.0)
                for k in range(T * FD):  # small-loop(fixture coefficients)
                    coef[k] = Float32(k % 5) * Float32(0.375) - Float32(0.5)
                var icpt = List[Float64](length=T, fill=0.0)
                for t in range(T):  # small-loop(fixture intercepts)
                    icpt[t] = Float64(t) * 0.25 - 0.125
                var pa = List[Float32](length=n * T, fill=0.0)
                var pb = List[Float32](length=n * T, fill=0.0)
                var ip = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(icpt.unsafe_ptr()))
                _ = ols_predict_device_targets(ctx, _P(unsafe_from_address=word), _p(coef), ip,
                    Int(pa.unsafe_ptr()), n, FD, T, False)
                _ = ols_predict_device_targets(ctx, _p(gathered), _p(coef), ip, Int(pb.unsafe_ptr()), n, FD, T, False)
                _same(pa, pb, "multi-target predictions")
                _ = coef^
                _ = icpt^
            elif which == 3:
                var mode = native_descriptor_call(metrics_program_device_v1, MP_MODE, [], [], [])
                var vendor = native_descriptor_call(metrics_program_device_v1, MP_VENDOR, [], [], [])
                var metrics = EntryRef(metrics_program_device_v1, entry_ref_word(mode, vendor, MP_OPERATION_COUNT))
                var pred = List[Float32](length=n, fill=0.0)
                for r in range(n):  # small-loop(fixture predictions)
                    pred[r] = y_fold[r] * Float32(0.875) + Float32(0.25)
                var a = member_r2_score_slot(metrics, slot_y, Int(pred.unsafe_ptr()), n)
                var b = member_r2_score(metrics, Int(y_fold.unsafe_ptr()), Int(pred.unsafe_ptr()), n)
                require(a.to_bits() == b.to_bits(), "the slot R^2 differs: " + String(a) + " vs " + String(b))
                _ = pred^
            else:
                var ctx = process_ctx[_CTX]()
                var la = List[Int64](length=SG_SLOTS, fill=Int64(0))
                scale_gamma_limbs_device(ctx, word, n * FD, Int(la.unsafe_ptr()))
                require(la[SGX_BAD] != 0, "the NaN fold slot reads as finite")
                # the fold's first row (row FN - 1) holds no NaN
                var row = store[].take_rows(store_ctx, base, FD, Int(index.unsafe_ptr()), 1)
                var lb = List[Int64](length=SG_SLOTS, fill=Int64(0))
                scale_gamma_limbs_device(ctx, fold_x_word(row), FD, Int(lb.unsafe_ptr()))
                store[].free(store_ctx, row)
                require(lb[SGX_BAD] == 0, "a finite fold row reads as non-finite")
        except error:
            failure = String(error)
        store[].free(store_ctx, slot)
        store[].free(store_ctx, slot_y)
        store[].free(store_ctx, base)
        store[].free(store_ctx, base_y)
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

        check_cases_serial("vdf_scope_linear4_fold device", 5, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
