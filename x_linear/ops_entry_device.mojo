# SPDX-License-Identifier: Apache-2.0
"""The device column of the x_linear operations table
(core/linear_entry_table_ext.mojo) and the native bodies the GPU binding's
Python exports (bindings/_mojolearn_x_linear.mojo) share with it.

Each body is the former Python export's body after its arguments were
unpacked: the same checks, the same kernels and the same words. The table
entry is called with the GIL already released by its caller; the Python
exports release it around the same bodies.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_zero import enqueue_fill
from core.msel_device_x import msel_fold_entry_context, msel_device_x_f32
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.py2mojo_rows import rows_call, rows_device
from core.shared_slot import SharedSlot
from core.linear_entry_table_ext import (
    XL_OP_FLAGS, XL_OP_FIT, XL_OP_FIT_FOLD, XL_OP_DECISION, XL_OP_DECISION_CODES,
    XL_OP_DECISION_ROWS, XL_OP_HUBER_OUT, XL_OP_CLASS_PREP, XL_OP_ROWS, XL_OP_SPEARMAN,
    XL_OP_BEST_ALPHA, XL_HAS_FIT_FOLD, XL_FIT_HEADER, XL_FOLD_HEADER,
    xl_ops_check_spans, xl_ops_int32, xl_ops_float32,
)
from x_linear.class_prep_device import class_prep_device
from x_linear.device import (
    fit_device, ridge_resident, decision_device, decision_codes_device, decision_rows_device,
    decision_huber_out_device, SGD_IDN_DEV_FINITE, linear_ctx,
)
from x_linear.dispatch import isotonic_abi_check, ALGO_SGD, ALGO_RIDGE
from x_linear.finite_device import XLIN_IDN_DEV_FINITE
from x_linear.ops import FP, IP
from x_linear.ridgecv import ridge_best_alpha
from x_linear.spearman_device import spearman_sign_device
from x_linear.x_linear_input_impl import x_linear_finite_device


def _fp(addr: Int) raises -> FP:
    if addr == 0:
        raise Error("x_linear: null float32 buffer address")
    return FP(unsafe_from_address=addr)


def xl_fit_device_run(a: Int, x_addr: Int, y_addr: Int, n: Int, d: Int, n_x: Int, n_y: Int, n_out: Int,
                      n_fw: Int, n_iw: Int, ipl: List[Int32], fpl: List[Float32], out_addr: Int) raises -> Int:
    """`x_linear_fit`: one fit route into the flat float32 result."""
    if n <= 0 or d < 0 or n_out <= 0:
        raise Error("x_linear: positive dimensions required")
    isotonic_abi_check(a, n, n_y, n_out, n_fw, n_iw, len(ipl))
    var x = _fp(x_addr)
    var y = _fp(y_addr)
    var out = _fp(out_addr)
    # lane/idn-sgd-multiblock: the SGD grids test the uploaded words on the
    # device (x_linear/device.mojo SGD_IDN_DEV_FINITE; the same error)
    # lane/fam-linear: every other route does the same on the X it uploads
    # (x_linear/finite_device.mojo XLIN_IDN_DEV_FINITE; the same error)
    # The fallback finite scans (builds without the device-side checks) are
    # native device work over host pointers, before the fit, with the same
    # refusals.
    var host_scan = not ((SGD_IDN_DEV_FINITE and a == ALGO_SGD) or (XLIN_IDN_DEV_FINITE and a != ALGO_SGD))
    if host_scan:
        x_linear_finite_device(x, n_x, "X")
        x_linear_finite_device(y, n_y, "y")
    fit_device(a, x, n_x, y, n_y, n, d, ipl, fpl, n_out, n_fw, n_iw, out)
    return n_out


def xl_fit_fold_device_run(a: Int, base_id: Int, base_rows: Int, index_id: Int, y_addr: Int, n: Int, d: Int,
                           n_x: Int, n_y: Int, n_out: Int, ipl: List[Int32], fpl: List[Float32],
                           out_addr: Int) raises -> Tuple[Int, Int]:
    """`x_linear_fit_fold` (lane a8c-device-x-linear-prep-rfe; IDENTICAL GPU
    builds): the Ridge fit over a model-selection fold resident on the device,
    gathered straight into the fit's X buffer with the finite and range tests
    in the same pass. Returns (finite word, range word); a nonzero word means
    the result is not to be read and the door refuses."""
    if a != ALGO_RIDGE:
        raise Error("x_linear_fit_fold: only the Ridge fit (algo 8) reads a device fold; algo " + String(a))
    if n <= 0 or d <= 0 or n_out <= 0:
        raise Error("x_linear: positive dimensions required")
    if n_x != n * d:
        raise Error("x_linear_fit_fold: dims[2] must be n * d (the fold's X words)")
    var y = _fp(y_addr)
    var out = _fp(out_addr)
    var ctx = msel_fold_entry_context()
    var dx = ctx.enqueue_create_buffer[DType.float32](n * d)
    var dst = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill[DType.int32](ctx, dst, Int32(0))
    msel_device_x_f32(ctx, base_id, base_rows, d, index_id, n, dx, dst.unsafe_ptr(), 0)
    var words = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_buf=words, src_buf=dst)
    # the gather tested X (n_x_test 0); y is tested by the fit as before.
    # ridge_resident synchronizes once at its end, which delivers words.
    ridge_resident(ctx.copy(), dx, 0, y, n_y, n, d, ipl, fpl, n_out, out)
    var finite = Int(words.unsafe_ptr().unsafe_load(0))
    var in_range = Int(words.unsafe_ptr().unsafe_load(1))
    _ = words^
    _ = dst^
    _ = dx^
    return (finite, in_range)


def xl_decision_device_run(x_addr: Int, wb_addr: Int, n: Int, d: Int, k: Int, link: Int, out_addr: Int,
                           status: Int) raises -> Int:
    if n < 0 or d < 0 or k <= 0:
        raise Error("x_linear: positive dimensions required")
    decision_device(_fp(x_addr), _fp(wb_addr), n, d, k, link, _fp(out_addr), status)
    return n * k


def xl_decision_codes_device_run(x_addr: Int, wb_addr: Int, n: Int, d: Int, k: Int, link: Int, strict: Int,
                                 below: Int, above: Int, out_addr: Int, status: Int) raises -> Int:
    if n < 0 or d < 0 or k <= 0:
        raise Error("x_linear: positive dimensions required")
    decision_codes_device(_fp(x_addr), _fp(wb_addr), n, d, k, link, strict, below, above,
                          IP(unsafe_from_address=out_addr), status)
    return n


def xl_decision_rows_device_run(x_addr: Int, wb_addr: Int, n: Int, d: Int, k: Int, link: Int, rows_mode: Int,
                                out_addr: Int, status: Int) raises -> Int:
    if n < 0 or d < 0 or k <= 0:
        raise Error("x_linear: positive dimensions required")
    decision_rows_device(_fp(x_addr), _fp(wb_addr), n, d, k, link, rows_mode, _fp(out_addr), status)
    return n


def xl_huber_out_device_run(x_addr: Int, wb_addr: Int, n: Int, d: Int, link: Int, y_addr: Int, thr: Float64,
                            flags_addr: Int) raises -> Int:
    if n < 0 or d < 0:
        raise Error("x_linear: positive dimensions required")
    var y = _fp(y_addr)
    var x = _fp(x_addr)
    var wb = _fp(wb_addr)
    if flags_addr == 0:
        raise Error("x_linear: null flag buffer address")
    var flags = MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=flags_addr)
    decision_huber_out_device(x, wb, n, d, link, y, thr, flags)
    return n


def xl_class_prep_device_run(ca: Int, sa: Int, wa: Int, oa: Int, n: Int, k: Int, balanced: Bool,
                             weighted: Bool) raises -> Int:
    """Class counts, 'balanced' class weights and per-row class weights on
    the device (x_linear/class_prep.mojo). Returns the largest unweighted
    class count."""
    if n <= 0 or k < 1 or ca == 0 or wa == 0:
        raise Error("x_linear class prep: n > 0, k > 0, codes and class weights required")
    if weighted and sa == 0:
        raise Error("x_linear class prep: weighted counts need sample weights")
    return class_prep_device(
        linear_ctx(), IP(unsafe_from_address=ca), FP(unsafe_from_address=sa if sa != 0 else wa), sa != 0,
        FP(unsafe_from_address=wa), n, k, balanced, weighted, FP(unsafe_from_address=oa if oa != 0 else wa),
        oa != 0)


def xl_spearman_device_run(xa: Int, ya: Int, n: Int) raises -> Int:
    if n <= 0 or xa == 0 or ya == 0:
        raise Error("x_linear spearman: n > 0 and two buffers required")
    return spearman_sign_device(linear_ctx(), FP(unsafe_from_address=xa), FP(unsafe_from_address=ya), n)


def xl_best_alpha_run(scores_addr: Int, n: Int) raises -> Int:
    """RidgeCV's best alpha index over the n mean CV scores; -1 if all NaN."""
    if n <= 0:
        return -1
    return ridge_best_alpha(FP(unsafe_from_address=scores_addr), n)


def x_linear_ops_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        xl_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == XL_OP_FLAGS:
            comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
                r[] = XL_HAS_FIT_FOLD
            else:
                r[] = 0
        elif operation == XL_OP_FIT:
            var ipl = xl_ops_int32(pp, XL_FIT_HEADER, p[8])
            var fpl = xl_ops_float32(ff, p[9])
            r[] = xl_fit_device_run(p[0], a[0], a[1], p[1], p[2], p[3], p[4], p[5], p[6], p[7], ipl, fpl, a[2])
        elif operation == XL_OP_FIT_FOLD:
            # lane a8c: the fold store is the IDENTICAL base binding's, so only
            # an IDENTICAL GPU build reads a device fold (XL_HAS_FIT_FOLD).
            comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
                var ipl = xl_ops_int32(pp, XL_FOLD_HEADER, p[11])
                var fpl = xl_ops_float32(ff, p[12])
                var words = xl_fit_fold_device_run(p[0], p[1], p[2], p[3], a[0], p[4], p[5], p[6], p[7], p[8],
                                                   ipl, fpl, a[1])
                var w = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[2])
                w[0] = words[0]
                w[1] = words[1]
                r[] = p[8]
            else:
                raise Error("x_linear_fit_fold: only an identical GPU build reads a device fold")
        elif operation == XL_OP_DECISION:
            r[] = xl_decision_device_run(a[0], a[1], p[0], p[1], p[2], p[3], a[2], p[4])
        elif operation == XL_OP_DECISION_CODES:
            r[] = xl_decision_codes_device_run(a[0], a[1], p[0], p[1], p[2], p[3], p[4], p[5], p[6], a[2], p[7])
        elif operation == XL_OP_DECISION_ROWS:
            r[] = xl_decision_rows_device_run(a[0], a[1], p[0], p[1], p[2], p[3], p[4], a[2], p[5])
        elif operation == XL_OP_HUBER_OUT:
            r[] = xl_huber_out_device_run(a[0], a[1], p[0], p[1], p[2], a[2], f[0], a[3])
        elif operation == XL_OP_CLASS_PREP:
            r[] = xl_class_prep_device_run(a[0], a[1], a[2], a[3], p[0], p[1], p[2] != 0, p[3] != 0)
        elif operation == XL_OP_ROWS:
            var call = rows_call(p[0], p[1], p[2], a[0], a[1], a[2], f[0])
            rows_device(linear_ctx(), call)
            r[] = call.n
        elif operation == XL_OP_SPEARMAN:
            r[] = xl_spearman_device_run(a[0], a[1], p[0])
        else:
            r[] = xl_best_alpha_run(a[0], p[0])
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XLinearOpsDeviceAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(x_linear_ops_device_v1)


comptime _OPS_SLOT = "MojolearnXLinearOpsDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXLinearOpsDeviceFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime XLINEAR_OPS_DEVICE_API = SharedSlot[XLinearOpsDeviceAPI, _OPS_SLOT]
