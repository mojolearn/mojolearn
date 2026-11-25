# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD X WORD through the Gaussian process entries (lane vdf-scope-gp, 2026-11-24;
core/fold_x_device.mojo, core/member_fold_x.mojo, core/gp_member.mojo `_fold_op`), in a process with no Python.

Each device case puts a 40 x 4 X once into the fold store (the x_prep transport, as a member's scope does),
takes one fold's rows (reversed, crossing the base) on the device and hands the GP provider entry the slot's
word; the same entry over the host gather of the same rows is the oracle (the host-input route, whose device
and CPU columns the release's one verification compares). Equal bytes put the fold route in that column.

Host column (CPU_COLUMN_BUILD):
    case 0   the host provider refuses a fold X word in words: gpr_fit, gpr_optimize, gpc_fit

Device column:
    case 0   gpr_fit (pointer route) over the slot == over the gather: L_, alpha_ and the five scalars
    case 1   gpr_query, resident route, X_train_ AND the query a slot (a fold-fitted state's lease uploads the
             slot on the device, core/fitted_resident_store.mojo) == host words: mean and std
    case 2   gpr_optimize over the slot == over the gather: theta, values, runs
    case 3   gpc_fit_all (one binary fit) over the slot == over the gather, then gpc_query_all through a
             lease with X_train_ a slot == host words: L_, pi_, W_sr_, y_train_, scalars, probabilities
    case 4   a NaN inside the fold: gpr_fit, gpr_optimize and gpc_fit_all refuse in the members' wording
    case 5   the x_prep transport's read (XP_DEV_READ, the member's export of a kept X_train_) == the gather

FAIL-PROOF EDITS (each must fail the named case): put back `enqueue_copy(dst_buf=dx, src_ptr=x_addr)` in
`gpr_fit_ptr_device` (device 0: a negative host address); drop the `srcs[i] < 0` branch of
core/fitted_resident_store.mojo `_upload_all` (device 1); pass `x_word` 0 in core/gp_native_device.mojo
`gpr_optimize` (device 2: an empty X); drop the `x_addr < 0` test in `gpc_fit_all_device` (device 4);
delete the XP_DEV_READ branch of x_prep/transport_device.mojo (device 5); delete `_host_x` in
core/gp_native_host.mojo (host 0).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.fold_x_word import fold_x_word
from core.gp_native_api import (
    GPRFitRequest, GPRQueryRequest, GPROptimizeRequest, GPCFitRequest, GPCFitAllRequest, GPCQueryAllRequest,
    GP_ROUTE_RESIDENT,
)
from core.gp_native_storage import GPKernelStorage
from core.gp_native_host import GPHostKernels
from core.gp_native_device import GPDeviceKernels
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from x_prep.transport_device import x_prep_transport_device_v1
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_READ

comptime FN = 40
comptime FD = 4


def _kernel() -> GPKernelStorage:
    """1.0 * RBF(length_scale=[0.8, 1.2, 2.0, 1.5]) in postfix (const, rbf, product)."""
    var value = GPKernelStorage()
    value.kinds = [0, 2, 5]
    value.parameters = [1.0, 0.0, 0.0]
    value.lengths = [0, 4, 0]
    value.scales = [0.8, 1.2, 2.0, 1.5]
    value.finish()
    return value^


def _x(n: Int, d: Int) -> List[Float32]:
    var x = List[Float32](length=n * d, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for j in range(d):  # small-loop(fixture columns)
            x[i * d + j] = Float32(((i * (j + 3) * 7) % 23) - 11) * Float32(0.125) + Float32(j) * Float32(0.5)
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


def _same64(a: List[Float64], b: List[Float64], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(word compare)
        require(a[i].to_bits() == b[i].to_bits(), label + ": word " + String(i) + " differs")


def _a(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _a64(values: List[Float64]) -> Int:
    return Int(values.unsafe_ptr())


def _host_case(which: Int) raises:
    comptime if CPU_COLUMN_BUILD:
        var kernel = _kernel()
        var buf = List[Float32](length=64, fill=0.0)
        var wide = List[Float64](length=64, fill=0.0)
        var host = _a(buf)
        var word = fold_x_word(0)
        var refused = 0
        try:
            _ = GPHostKernels.gpr_fit(GPRFitRequest(word, host, host, host, _a64(wide), 4, FD, kernel.view(), 1e-6,
                                                    False))
        except:
            refused += 1
        try:
            _ = GPHostKernels.gpr_optimize(GPROptimizeRequest(word, host, host, host, _a64(wide), _a64(wide),
                _a64(wide), 4, FD, kernel.view(), 1e-6, 0, 0, 0))
        except:
            refused += 1
        try:
            _ = GPHostKernels.gpc_fit(GPCFitRequest(word, host, host, host, host, _a64(wide), 4, FD, kernel.view(), 5,
                                                    False, 0, 0))
        except:
            refused += 1
        require(refused == 3, "the host GP column read a fold X word (" + String(3 - refused) + " of 3 entries)")
        _ = buf^
        _ = wide^
        _ = kernel^


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _host_case(which)

        check_cases("vdf_scope_gp_fold host", 1, body)


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var x = _x(FN, FD)
        if which == 4:
            x[(FN - 2) * FD + 1] = Float32(0.0) / Float32(0.0)  # inside the fold (row FN - 2)
        var y = List[Float32](length=FN, fill=0.0)
        var codes = List[Int32](length=FN, fill=Int32(0))
        for i in range(FN):  # small-loop(fixture targets and class codes)
            y[i] = Float32((i * 13) % 17) * Float32(0.25) - Float32(2.0)
            codes[i] = Int32(1) if (i * 7) % 3 == 0 else Int32(0)
        var index = _fold(FN)
        var n = len(index)
        var gathered = List[Float32](length=n * FD, fill=0.0)
        var y_fold = List[Float32](length=n, fill=0.0)
        var codes_fold = List[Int32](length=n, fill=Int32(0))
        for r in range(n):  # small-loop(oracle gather)
            for c in range(FD):  # small-loop(oracle gather)
                gathered[r * FD + c] = x[Int(index[r]) * FD + c]
            y_fold[r] = y[Int(index[r])]
            codes_fold[r] = codes[Int(index[r])]
        var store: NativeDescriptorCall = x_prep_transport_device_v1
        var base = native_descriptor_call(store, XP_DEV_PUT, [_a(x)], [FN * FD], [])
        var slot = native_descriptor_call(store, XP_DEV_TAKE_ROWS, [Int(index.unsafe_ptr())], [base, FD, n], [])
        var word = fold_x_word(slot)
        var kernel = _kernel()
        var failure = String("")
        try:
            if which == 0 or which == 1:
                var la = List[Float32](length=n * n, fill=0.0)
                var lb = List[Float32](length=n * n, fill=0.0)
                var da = List[Float32](length=n, fill=0.0)
                var db = List[Float32](length=n, fill=0.0)
                var sa = List[Float64](length=5, fill=0.0)
                var sb = List[Float64](length=5, fill=0.0)
                var ia = GPDeviceKernels.gpr_fit(GPRFitRequest(word, _a(y_fold), _a(la), _a(da), _a64(sa), n, FD,
                                                               kernel.view(), 1e-6, True))
                var ib = GPDeviceKernels.gpr_fit(GPRFitRequest(_a(gathered), _a(y_fold), _a(lb), _a(db), _a64(sb), n,
                                                               FD, kernel.view(), 1e-6, True))
                require(ia == ib, "the fold fit's info differs")
                _same(la, lb, "L_")
                _same(da, db, "alpha_")
                _same64(sa, sb, "fit scalars")
                if which == 1:
                    var q = n // 2
                    var qrows = List[Int64]()
                    for r in range(q):  # small-loop(fixture: query rows)
                        qrows.append(Int64((r * 5) % FN))
                    var xq = List[Float32](length=q * FD, fill=0.0)
                    for r in range(q):  # small-loop(oracle gather of the query)
                        for c in range(FD):  # small-loop(oracle gather of the query)
                            xq[r * FD + c] = x[Int(qrows[r]) * FD + c]
                    var qslot = native_descriptor_call(store, XP_DEV_TAKE_ROWS, [Int(qrows.unsafe_ptr())],
                                                       [base, FD, q], [])
                    var ma = List[Float32](length=q, fill=0.0)
                    var mb = List[Float32](length=q, fill=0.0)
                    var va = List[Float32](length=q, fill=0.0)
                    var vb = List[Float32](length=q, fill=0.0)
                    var ta = List[Float32](length=q, fill=0.0)
                    var tb = List[Float32](length=q, fill=0.0)
                    var ca = List[Int32](length=q, fill=Int32(0))
                    var cb = List[Int32](length=q, fill=Int32(0))
                    var lease_a = GPDeviceKernels.lease_open()
                    var lease_b = GPDeviceKernels.lease_open()
                    var qa = GPDeviceKernels.gpr_query(GPRQueryRequest(word, _a(la), _a(da), fold_x_word(qslot),
                        _a(ma), _a(va), _a(ta), Int(ca.unsafe_ptr()), n, FD, q, kernel.view(), ia, True, False, 1.0,
                        0.0, GP_ROUTE_RESIDENT, lease_a, -1, 0, -1))
                    _ = ca^
                    _ = da^
                    _ = la^
                    _ = va^
                    var qb = GPDeviceKernels.gpr_query(GPRQueryRequest(_a(gathered), _a(lb), _a(db), _a(xq),
                        _a(mb), _a(vb), _a(tb), Int(cb.unsafe_ptr()), n, FD, q, kernel.view(), ib, True, False, 1.0,
                        0.0, GP_ROUTE_RESIDENT, lease_b, -1, 0, -1))
                    _ = cb^
                    _ = db^
                    _ = lb^
                    _ = vb^
                    GPDeviceKernels.lease_close(lease_a)
                    GPDeviceKernels.lease_close(lease_b)
                    _ = native_descriptor_call(store, XP_DEV_FREE, [], [qslot], [])
                    require(qa.finite == 0 and qb.finite == 0, "a finite query was flagged")
                    _same(ma, mb, "predictive mean")
                    _same(ta, tb, "predictive std")
                    _ = qrows^
                    _ = xq^
            elif which == 2:
                var flags: List[Int32] = [0, 1, 0]
                var bounds: List[Float32] = [0.05, 20.0, 0.05, 20.0, 0.05, 20.0, 0.05, 20.0]
                var view = kernel.view()
                var ta = List[Float64](length=4, fill=0.0)
                var tb = List[Float64](length=4, fill=0.0)
                var va = List[Float64](length=4, fill=0.0)
                var vb = List[Float64](length=4, fill=0.0)
                var ra = List[Float64](length=8, fill=0.0)
                var rb = List[Float64](length=8, fill=0.0)
                var ba = GPDeviceKernels.gpr_optimize(GPROptimizeRequest(word, _a(y_fold), Int(flags.unsafe_ptr()),
                    _a(bounds), _a64(ta), _a64(va), _a64(ra), n, FD, view.copy(), 1e-6, 1, 7, 0))
                var bb = GPDeviceKernels.gpr_optimize(GPROptimizeRequest(_a(gathered), _a(y_fold),
                    Int(flags.unsafe_ptr()), _a(bounds), _a64(tb), _a64(vb), _a64(rb), n, FD, view, 1e-6, 1, 7, 0))
                require(ba == bb, "the optimizer's best run differs")
                _same64(ta, tb, "theta")
                _same64(va, vb, "optimized values")
                _same64(ra, rb, "optimizer runs")
                _ = flags^
                _ = bounds^
            elif which == 3:
                var columns: List[Int] = [1]
                var ya = List[Float32](length=n, fill=0.0)
                var yb = List[Float32](length=n, fill=0.0)
                var la = List[Float32](length=n * n, fill=0.0)
                var lb = List[Float32](length=n * n, fill=0.0)
                var pa = List[Float32](length=n, fill=0.0)
                var pb = List[Float32](length=n, fill=0.0)
                var wa = List[Float32](length=n, fill=0.0)
                var wb = List[Float32](length=n, fill=0.0)
                var sa = List[Float64](length=3, fill=0.0)
                var sb = List[Float64](length=3, fill=0.0)
                var tya: List[Int] = [_a(ya)]
                var tyb: List[Int] = [_a(yb)]
                var tla: List[Int] = [_a(la)]
                var tlb: List[Int] = [_a(lb)]
                var tpa: List[Int] = [_a(pa)]
                var tpb: List[Int] = [_a(pb)]
                var twa: List[Int] = [_a(wa)]
                var twb: List[Int] = [_a(wb)]
                var tsa: List[Int] = [_a64(sa)]
                var tsb: List[Int] = [_a64(sb)]
                _ = GPDeviceKernels.gpc_fit_all(GPCFitAllRequest(word, Int(codes_fold.unsafe_ptr()),
                    Int(columns.unsafe_ptr()), Int(tya.unsafe_ptr()), Int(tla.unsafe_ptr()), Int(tpa.unsafe_ptr()),
                    Int(twa.unsafe_ptr()), Int(tsa.unsafe_ptr()), n, FD, kernel.view(), 20, 1))
                _ = GPDeviceKernels.gpc_fit_all(GPCFitAllRequest(_a(gathered), Int(codes_fold.unsafe_ptr()),
                    Int(columns.unsafe_ptr()), Int(tyb.unsafe_ptr()), Int(tlb.unsafe_ptr()), Int(tpb.unsafe_ptr()),
                    Int(twb.unsafe_ptr()), Int(tsb.unsafe_ptr()), n, FD, kernel.view(), 20, 1))
                _same(ya, yb, "y_train_")
                _same(la, lb, "L_")
                _same(pa, pb, "pi_")
                _same(wa, wb, "W_sr_")
                _same64(sa, sb, "fit scalars")
                var q = 9
                var xq = _x(q, FD)
                var oa = List[Float64](length=2 * q, fill=0.0)
                var ob = List[Float64](length=2 * q, fill=0.0)
                var lease_a = GPDeviceKernels.lease_open()
                var lease_b = GPDeviceKernels.lease_open()
                var qa = GPDeviceKernels.gpc_query_all(GPCQueryAllRequest(word, _a(xq), _a64(oa),
                    Int(tya.unsafe_ptr()), Int(tpa.unsafe_ptr()), Int(twa.unsafe_ptr()), Int(tla.unsafe_ptr()),
                    n, FD, q, kernel.view(), 2, 1, lease_a, -1, 0, -1))
                _ = ya^
                _ = la^
                _ = pa^
                _ = wa^
                var qb = GPDeviceKernels.gpc_query_all(GPCQueryAllRequest(_a(gathered), _a(xq), _a64(ob),
                    Int(tyb.unsafe_ptr()), Int(tpb.unsafe_ptr()), Int(twb.unsafe_ptr()), Int(tlb.unsafe_ptr()),
                    n, FD, q, kernel.view(), 2, 1, lease_b, -1, 0, -1))
                _ = yb^
                _ = lb^
                _ = pb^
                _ = wb^
                GPDeviceKernels.lease_close(lease_a)
                GPDeviceKernels.lease_close(lease_b)
                require(qa.finite == 0 and qb.finite == 0, "a finite query was flagged")
                _same64(oa, ob, "class probabilities")
                _ = columns^
                _ = xq^
                _ = tya^
                _ = tyb^
                _ = tla^
                _ = tlb^
                _ = tpa^
                _ = tpb^
                _ = twa^
                _ = twb^
                _ = tsa^
                _ = tsb^
            elif which == 4:
                var refused = 0
                var la = List[Float32](length=n * n, fill=0.0)
                var da = List[Float32](length=n, fill=0.0)
                var sa = List[Float64](length=8, fill=0.0)
                try:
                    _ = GPDeviceKernels.gpr_fit(GPRFitRequest(word, _a(y_fold), _a(la), _a(da), _a64(sa), n, FD,
                                                              kernel.view(), 1e-6, True))
                except error:
                    if "Input X contains NaN or infinity (GaussianProcessRegressor)." in String(error):
                        refused += 1
                var flags: List[Int32] = [0, 1, 0]
                var bounds: List[Float32] = [0.05, 20.0, 0.05, 20.0, 0.05, 20.0, 0.05, 20.0]
                try:
                    _ = GPDeviceKernels.gpr_optimize(GPROptimizeRequest(word, _a(y_fold), Int(flags.unsafe_ptr()),
                        _a(bounds), _a64(sa), _a64(sa), _a64(sa), n, FD, kernel.view(), 1e-6, 0, 7, 0))
                except error:
                    if "Input X contains NaN or infinity (GaussianProcessRegressor)." in String(error):
                        refused += 1
                var columns: List[Int] = [1]
                var t32: List[Int] = [_a(da)]
                var tl: List[Int] = [_a(la)]
                var t64: List[Int] = [_a64(sa)]
                try:
                    _ = GPDeviceKernels.gpc_fit_all(GPCFitAllRequest(word, Int(codes_fold.unsafe_ptr()),
                        Int(columns.unsafe_ptr()), Int(t32.unsafe_ptr()), Int(tl.unsafe_ptr()), Int(t32.unsafe_ptr()),
                        Int(t32.unsafe_ptr()), Int(t64.unsafe_ptr()), n, FD, kernel.view(), 20, 1))
                except error:
                    if "Input X contains NaN or infinity (GaussianProcessClassifier)." in String(error):
                        refused += 1
                require(refused == 3, "a NaN fold reached a GP fit (" + String(3 - refused) + " of 3 entries)")
                _ = flags^
                _ = bounds^
                _ = columns^
                _ = t32^
                _ = tl^
                _ = t64^
                _ = da^
                _ = la^
                _ = sa^
            else:
                var back = List[Float32](length=n * FD, fill=0.0)
                _ = native_descriptor_call(store, XP_DEV_READ, [_a(back)], [slot, n * FD], [])
                _same(back, gathered, "the read of a fold slot")
        except error:
            failure = String(error)
        _ = native_descriptor_call(store, XP_DEV_FREE, [], [slot], [])
        _ = native_descriptor_call(store, XP_DEV_FREE, [], [base], [])
        _ = x^
        _ = y^
        _ = codes^
        _ = index^
        _ = gathered^
        _ = y_fold^
        _ = codes_fold^
        _ = kernel^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("vdf_scope_gp_fold device", 6, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
