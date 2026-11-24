# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE FOLD X WORD through the Gaussian process LIST routes (lane vdf-tail-a, 2026-11-24): the `fast` GPU routes
(GPR_IDN_PTR / GPC_IDN_OVR are identical-only, so a `fast` member fits through `gpr_fit_host` and the classifier's
per-column `gpc_fit_binary_host`, and the classifier queries through `gpc_predict_binary_host`), now reading a
model-selection fold's rows from a fold store slot on the device (gaussian_process/estimator.mojo `_x_device`, one
device copy of the same bytes the host gather produces). In a process with no Python; both numeric modes (the list
routes exist in every GPU image).

Each case puts a 40 x 4 X once into the fold store (the x_prep transport, as a member's scope does), takes one
fold's rows (reversed, crossing the base) on the device and hands the GP device entry the slot's word on the LIST
route; the same entry over the host gather of the same rows is the oracle (the host-input route, whose device and
CPU columns the release's one verification compares). Equal bytes put the fold route in that column.

    case 0   gpr_fit, list route (pointer_route False), X a slot == the gather: L_, alpha_ and the five scalars
    case 1   gpr_query, LIST route, X_train_ a slot (a fold-fitted state) AND the query a slot == host words: mean,
             variance, std, clamp flags
    case 2   gpr_covariance and gpr_sample over the two slots == host words: mean, covariance, draws
    case 3   gpr_gradient over the slot == over the gather: the gradient and (info, lml)
    case 4   gpc_fit, list route (coded targets, column 1) over the slot == over the gather (L_, pi_, W_sr_, y01,
             scalars), then gpc_query over the X_train_ slot and a query slot == host words (mean, variance,
             probability)
    case 5   a NaN inside the fold: the list gpr_fit, gpr_gradient and gpc_fit, and a list gpr_query and gpc_query
             with the NaN rows as the query, refuse in the members' wording

FAIL-PROOF EDITS (each must fail the named case): in `_x_device` copy `n - 1` words (cases 0-4); return before
the finite test (case 5); in core/gp_native_device.mojo pass `_x_word` 0 to `_gpr_fit_run` (case 0: an empty X
list); drop the `xt_word` argument of `gpc_predict_binary_host`'s call (case 4).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from core.fold_x_word import fold_x_word
from core.gp_native_api import (
    GPRFitRequest, GPRQueryRequest, GPRCovarianceRequest, GPRSampleRequest, GPRGradientRequest, GPCFitRequest,
    GPCQueryRequest, GP_ROUTE_LIST,
)
from core.gp_native_storage import GPKernelStorage
from core.gp_native_device import GPDeviceKernels
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from x_prep.transport_device import x_prep_transport_device_v1
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS

comptime FN = 40
comptime FD = 4
comptime CASES = 6


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


def _query_rows(q: Int) -> List[Int64]:
    var rows = List[Int64]()
    for r in range(q):  # small-loop(fixture: query rows, crossing the fold)
        rows.append(Int64((r * 5 + 3) % FN))
    return rows^


def _gather(x: List[Float32], index: List[Int64]) -> List[Float32]:
    var out = List[Float32](length=len(index) * FD, fill=0.0)
    for r in range(len(index)):  # small-loop(oracle gather)
        for c in range(FD):  # small-loop(oracle gather)
            out[r * FD + c] = x[Int(index[r]) * FD + c]
    return out^


def _same(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(word compare)
        require(a[i].to_bits() == b[i].to_bits(), label + ": word " + String(i) + " differs")


def _same64(a: List[Float64], b: List[Float64], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(word compare)
        require(a[i].to_bits() == b[i].to_bits(), label + ": word " + String(i) + " differs")


def _same32(a: List[Int32], b: List[Int32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(word compare)
        require(a[i] == b[i], label + ": word " + String(i) + " differs")


def _a(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _a64(values: List[Float64]) -> Int:
    return Int(values.unsafe_ptr())


def _refused(error: Error, est: String) -> Bool:
    return ("Input X contains NaN or infinity (" + est + ").") in String(error)


struct _Fit(Movable):
    var l: List[Float32]
    var dual: List[Float32]
    var scalars: List[Float64]
    var info: Int

    def __init__(out self, n: Int):
        self.l = List[Float32](length=n * n, fill=0.0)
        self.dual = List[Float32](length=n, fill=0.0)
        self.scalars = List[Float64](length=5, fill=0.0)
        self.info = 0


def _gpr_fit(x_addr: Int, y: List[Float32], n: Int, kernel: GPKernelStorage) raises -> _Fit:
    var fit = _Fit(n)
    fit.info = GPDeviceKernels.gpr_fit(GPRFitRequest(x_addr, _a(y), _a(fit.l), _a(fit.dual), _a64(fit.scalars), n,
                                                     FD, kernel.view(), 1e-6, False))
    return fit^


def _device_case(which: Int) raises:
    var x = _x(FN, FD)
    if which == 5:
        x[(FN - 2) * FD + 1] = Float32(0.0) / Float32(0.0)  # inside the fold (row FN - 2) and the query (row 38)
    var y = List[Float32](length=FN, fill=0.0)
    var codes = List[Int32](length=FN, fill=Int32(0))
    for i in range(FN):  # small-loop(fixture targets and class codes)
        y[i] = Float32((i * 13) % 17) * Float32(0.25) - Float32(2.0)
        codes[i] = Int32(1) if (i * 7) % 3 == 0 else Int32(0)
    var index = _fold(FN)
    var n = len(index)
    var gathered = _gather(x, index)
    var y_fold = List[Float32](length=n, fill=0.0)
    var codes_fold = List[Int32](length=n, fill=Int32(0))
    for r in range(n):  # small-loop(oracle gather of the targets)
        y_fold[r] = y[Int(index[r])]
        codes_fold[r] = codes[Int(index[r])]
    var q = 9
    var qrows = _query_rows(q)
    if which == 5:
        qrows[0] = Int64(FN - 2)
    var xq = _gather(x, qrows)
    var store: NativeDescriptorCall = x_prep_transport_device_v1
    var base = native_descriptor_call(store, XP_DEV_PUT, [_a(x)], [FN * FD], [])
    var slot = native_descriptor_call(store, XP_DEV_TAKE_ROWS, [Int(index.unsafe_ptr())], [base, FD, n], [])
    var qslot = native_descriptor_call(store, XP_DEV_TAKE_ROWS, [Int(qrows.unsafe_ptr())], [base, FD, q], [])
    var word = fold_x_word(slot)
    var qword = fold_x_word(qslot)
    var kernel = _kernel()
    var failure = String("")
    try:
        if which >= 0 and which <= 2:
            var fa = _gpr_fit(word, y_fold, n, kernel)
            var fb = _gpr_fit(_a(gathered), y_fold, n, kernel)
            require(fa.info == fb.info and fa.info == 0, "the fold fit's info differs or the fit failed")
            _same(fa.l, fb.l, "L_")
            _same(fa.dual, fb.dual, "alpha_")
            _same64(fa.scalars, fb.scalars, "fit scalars")
            if which == 1:
                var ma = List[Float32](length=q, fill=0.0)
                var mb = List[Float32](length=q, fill=0.0)
                var va = List[Float32](length=q, fill=0.0)
                var vb = List[Float32](length=q, fill=0.0)
                var ta = List[Float32](length=q, fill=0.0)
                var tb = List[Float32](length=q, fill=0.0)
                var ca = List[Int32](length=q, fill=Int32(0))
                var cb = List[Int32](length=q, fill=Int32(0))
                var qa = GPDeviceKernels.gpr_query(GPRQueryRequest(word, _a(fa.l), _a(fa.dual), qword, _a(ma),
                    _a(va), _a(ta), Int(ca.unsafe_ptr()), n, FD, q, kernel.view(), fa.info, True, False, 1.0, 0.0,
                    GP_ROUTE_LIST, 0, -1, 0, -1))
                var qb = GPDeviceKernels.gpr_query(GPRQueryRequest(_a(gathered), _a(fb.l), _a(fb.dual), _a(xq),
                    _a(mb), _a(vb), _a(tb), Int(cb.unsafe_ptr()), n, FD, q, kernel.view(), fb.info, True, False, 1.0,
                    0.0, GP_ROUTE_LIST, 0, -1, 0, -1))
                require(qa.value == qb.value, "the clamp counts differ")
                _same(ma, mb, "predictive mean")
                _same(va, vb, "predictive variance")
                _same(ta, tb, "predictive std")
                _same32(ca, cb, "clamp flags")
            elif which == 2:
                var ma = List[Float32](length=q, fill=0.0)
                var mb = List[Float32](length=q, fill=0.0)
                var ka = List[Float32](length=q * q, fill=0.0)
                var kb = List[Float32](length=q * q, fill=0.0)
                _ = GPDeviceKernels.gpr_covariance(GPRCovarianceRequest(word, _a(fa.l), _a(fa.dual), qword, _a(ma),
                    _a(ka), n, FD, q, kernel.view(), fa.info, False, 1.0, 0.0))
                _ = GPDeviceKernels.gpr_covariance(GPRCovarianceRequest(_a(gathered), _a(fb.l), _a(fb.dual), _a(xq),
                    _a(mb), _a(kb), n, FD, q, kernel.view(), fb.info, False, 1.0, 0.0))
                _same(ma, mb, "posterior mean")
                _same(ka, kb, "posterior covariance")
                var s = 3
                var da = List[Float32](length=q * s, fill=0.0)
                var db = List[Float32](length=q * s, fill=0.0)
                _ = GPDeviceKernels.gpr_sample(GPRSampleRequest(word, _a(fa.l), _a(fa.dual), qword, _a(da), n, FD, q,
                    kernel.view(), fa.info, s, 20261124, 0, False, 1.0, 0.0))
                _ = GPDeviceKernels.gpr_sample(GPRSampleRequest(_a(gathered), _a(fb.l), _a(fb.dual), _a(xq), _a(db),
                    n, FD, q, kernel.view(), fb.info, s, 20261124, 0, False, 1.0, 0.0))
                _same(da, db, "posterior draws")
        elif which == 3:
            var flags: List[Int32] = [0, 1, 0]
            var ga = List[Float64](length=4, fill=0.0)
            var gb = List[Float64](length=4, fill=0.0)
            var sa = List[Float64](length=2, fill=0.0)
            var sb = List[Float64](length=2, fill=0.0)
            var ia = GPDeviceKernels.gpr_gradient(GPRGradientRequest(word, _a(y_fold), Int(flags.unsafe_ptr()),
                _a64(ga), _a64(sa), n, FD, kernel.view(), 1e-6))
            var ib = GPDeviceKernels.gpr_gradient(GPRGradientRequest(_a(gathered), _a(y_fold),
                Int(flags.unsafe_ptr()), _a64(gb), _a64(sb), n, FD, kernel.view(), 1e-6))
            require(ia == ib, "the gradient's info differs")
            _same64(ga, gb, "the likelihood gradient")
            _same64(sa, sb, "(info, lml)")
            _ = flags^
        elif which == 4:
            var la = List[Float32](length=n * n, fill=0.0)
            var lb = List[Float32](length=n * n, fill=0.0)
            var pa = List[Float32](length=n, fill=0.0)
            var pb = List[Float32](length=n, fill=0.0)
            var wa = List[Float32](length=n, fill=0.0)
            var wb = List[Float32](length=n, fill=0.0)
            var ya = List[Float32](length=n, fill=0.0)
            var yb = List[Float32](length=n, fill=0.0)
            var sa = List[Float64](length=3, fill=0.0)
            var sb = List[Float64](length=3, fill=0.0)
            var ita = GPDeviceKernels.gpc_fit(GPCFitRequest(word, Int(codes_fold.unsafe_ptr()), _a(la), _a(pa),
                _a(wa), _a64(sa), n, FD, kernel.view(), 20, True, 1, _a(ya)))
            var itb = GPDeviceKernels.gpc_fit(GPCFitRequest(_a(gathered), Int(codes_fold.unsafe_ptr()), _a(lb),
                _a(pb), _a(wb), _a64(sb), n, FD, kernel.view(), 20, True, 1, _a(yb)))
            require(ita == itb, "the Newton iteration counts differ")
            _same(la, lb, "L_")
            _same(pa, pb, "pi_")
            _same(wa, wb, "W_sr_")
            _same(ya, yb, "y_train_ (0 / 1)")
            _same64(sa, sb, "fit scalars")
            var ma = List[Float32](length=q, fill=0.0)
            var mb = List[Float32](length=q, fill=0.0)
            var va = List[Float32](length=q, fill=0.0)
            var vb = List[Float32](length=q, fill=0.0)
            var ra = List[Float64](length=q, fill=0.0)
            var rb = List[Float64](length=q, fill=0.0)
            _ = GPDeviceKernels.gpc_query(GPCQueryRequest(word, _a(ya), _a(pa), _a(wa), _a(la), qword, _a(ma),
                _a(va), _a64(ra), n, FD, q, kernel.view(), True, 0, 0))
            _ = GPDeviceKernels.gpc_query(GPCQueryRequest(_a(gathered), _a(yb), _a(pb), _a(wb), _a(lb), _a(xq),
                _a(mb), _a(vb), _a64(rb), n, FD, q, kernel.view(), True, 0, 0))
            _same(ma, mb, "latent mean")
            _same(va, vb, "latent variance")
            _same64(ra, rb, "class probability")
        else:
            var refused = 0
            var fit = _Fit(n)
            try:
                _ = GPDeviceKernels.gpr_fit(GPRFitRequest(word, _a(y_fold), _a(fit.l), _a(fit.dual),
                    _a64(fit.scalars), n, FD, kernel.view(), 1e-6, False))
            except error:
                if _refused(error, "GaussianProcessRegressor"):
                    refused += 1
            var flags: List[Int32] = [0, 1, 0]
            var g = List[Float64](length=4, fill=0.0)
            try:
                _ = GPDeviceKernels.gpr_gradient(GPRGradientRequest(word, _a(y_fold), Int(flags.unsafe_ptr()),
                    _a64(g), _a64(fit.scalars), n, FD, kernel.view(), 1e-6))
            except error:
                if _refused(error, "GaussianProcessRegressor"):
                    refused += 1
            var y01 = List[Float32](length=n, fill=0.0)
            try:
                _ = GPDeviceKernels.gpc_fit(GPCFitRequest(word, Int(codes_fold.unsafe_ptr()), _a(fit.l),
                    _a(fit.dual), _a(fit.dual), _a64(fit.scalars), n, FD, kernel.view(), 20, True, 1, _a(y01)))
            except error:
                if _refused(error, "GaussianProcessClassifier"):
                    refused += 1
            # The queries: a finite fit over the host gather of the training rows without the NaN row, the NaN
            # rows as the query slot.
            var clean = _gather(_x(FN, FD), index)
            var ok = _gpr_fit(_a(clean), y_fold, n, kernel)
            var m = List[Float32](length=q, fill=0.0)
            var v = List[Float32](length=q, fill=0.0)
            var t = List[Float32](length=q, fill=0.0)
            var c = List[Int32](length=q, fill=Int32(0))
            try:
                _ = GPDeviceKernels.gpr_query(GPRQueryRequest(_a(clean), _a(ok.l), _a(ok.dual), qword, _a(m), _a(v),
                    _a(t), Int(c.unsafe_ptr()), n, FD, q, kernel.view(), ok.info, True, False, 1.0, 0.0,
                    GP_ROUTE_LIST, 0, -1, 0, -1))
            except error:
                if _refused(error, "GaussianProcessRegressor"):
                    refused += 1
            var r = List[Float64](length=q, fill=0.0)
            for i in range(n):  # small-loop(fixture: two-class 0 / 1 targets for the query's model checks)
                y01[i] = Float32(codes_fold[i])
            try:
                _ = GPDeviceKernels.gpc_query(GPCQueryRequest(_a(clean), _a(y01), _a(ok.dual), _a(ok.dual),
                    _a(ok.l), qword, _a(m), _a(v), _a64(r), n, FD, q, kernel.view(), True, 0, 0))
            except error:
                if _refused(error, "GaussianProcessClassifier"):
                    refused += 1
            require(refused == 5, "a NaN fold reached a GP list entry (" + String(5 - refused) + " of 5 entries)")
            _ = flags^
            _ = g^
            _ = y01^
            _ = clean^
            _ = r^
    except error:
        failure = String(error)
    _ = native_descriptor_call(store, XP_DEV_FREE, [], [qslot], [])
    _ = native_descriptor_call(store, XP_DEV_FREE, [], [slot], [])
    _ = native_descriptor_call(store, XP_DEV_FREE, [], [base], [])
    _ = x^
    _ = y^
    _ = codes^
    _ = index^
    _ = qrows^
    _ = gathered^
    _ = xq^
    _ = y_fold^
    _ = codes_fold^
    _ = kernel^
    if failure.byte_length() > 0:
        raise Error(failure)


def test_vdf_tail_a_gp_list() raises:
    check_cases_serial("vdf_tail_a_gp_list", CASES, _device_case)


def main() raises:
    var suite = TestSuite()
    suite.test[test_vdf_tail_a_gp_list]()
    suite^.run()
