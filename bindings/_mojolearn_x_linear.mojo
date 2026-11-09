# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE'S GPU BINDING (lane/algos-linear, 2026-09-27).

x_linear_fit(algo, x_addr, y_addr, dims, ip, fp, out_addr)
    dims = [n, d, n_x, n_y, n_out, n_fw, n_iw, n_ip, n_fp]
x_linear_decision(x_addr, wb_addr, dims, out_addr)
    dims = [n, d, k, link]; wb is k rows of (w_0..w_{d-1}, b)
The host binding (_mojolearn_x_linear_host.mojo) exports the same names with
the same address contract and runs the same x_linear/ source on the CPU."""
from core.experiment_guards_linear import LINEAR_GUARDS_OK
from bindings.device_plan_binding import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from providers.python.core_py2mojo_rows_pyapi import py2mojo_rows_device_binding
from core.py2mojo_linear import py2mojo_linear_flags
from x_linear.spearman_device import spearman_sign_device
from x_linear.device import linear_ctx as _p2m_ctx
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from std.sys.info import has_apple_gpu_accelerator
from std.sys.compile import is_defined
from x_linear.ops import FP, IP
from x_linear.ridgecv import ridge_best_alpha
from x_linear.device import fit_device, ridge_resident, decision_device, decision_codes_device, decision_rows_device, decision_huber_out_device, SGD_IDN_DEV_FINITE
from x_linear.dispatch import isotonic_abi_check, ALGO_SGD, ALGO_RIDGE
from core.device_zero import enqueue_fill
from core.msel_device_x import msel_device_x_context, msel_device_x_f32
from checks.numerics import NUMERIC_IDENTICAL
from x_linear.finite_device import XLIN_IDN_DEV_FINITE
from x_linear.class_prep_device import class_prep_device
from x_linear.x_linear_input_impl import x_linear_finite_device as _finite


def _fp(addr: Int) raises -> FP:
    if addr == 0:
        raise Error("x_linear: null float32 buffer address")
    return FP(unsafe_from_address=addr)


def best_alpha_binding(scores_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """Lane py-runtime round 2: RidgeCV's best alpha index over the n mean
    CV scores (`ridge_best_alpha`, x_linear/ridgecv.mojo); -1 if all NaN."""
    var na = Int(py=n)
    if na <= 0:
        return PythonObject(-1)
    return PythonObject(ridge_best_alpha(FP(unsafe_from_address=Int(py=scores_addr)), na))


def fit_binding(algo: PythonObject, x_addr: PythonObject, y_addr: PythonObject, dims: PythonObject,
                ip: PythonObject, fp: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var n_x = Int(py=dims[2])
    var n_y = Int(py=dims[3])
    var n_out = Int(py=dims[4])
    var n_fw = Int(py=dims[5])
    var n_iw = Int(py=dims[6])
    var n_ip = Int(py=dims[7])
    var n_fp = Int(py=dims[8])
    if n <= 0 or d < 0 or n_out <= 0:
        raise Error("x_linear: positive dimensions required")
    isotonic_abi_check(Int(py=algo), n, n_y, n_out, n_fw, n_iw, n_ip)
    var ipl = List[Int32](capacity=n_ip)
    for i in range(n_ip):  # small-loop(n_ip: integer hyperparameters of the route): Python parameter list, not data
        ipl.append(Int32(Int(py=ip[i])))
    var fpl = List[Float32](capacity=n_fp)
    for i in range(n_fp):  # small-loop(n_fp: float hyperparameters of the route): Python parameter list, not data
        fpl.append(Float32(Float64(py=fp[i])))
    var a = Int(py=algo)
    var x = _fp(Int(py=x_addr))
    var y = _fp(Int(py=y_addr))
    var out = _fp(Int(py=out_addr))
    # lane/idn-sgd-multiblock: the SGD grids test the uploaded words on the
    # device (x_linear/device.mojo SGD_IDN_DEV_FINITE; the same error)
    # lane/fam-linear: every other route does the same on the X it uploads
    # (x_linear/finite_device.mojo XLIN_IDN_DEV_FINITE; the same error)
    # The fallback finite scans (builds without the device-side checks) are
    # native device work over host pointers: they run with the GIL released,
    # before the fit, with the same refusals.
    var host_scan = not ((SGD_IDN_DEV_FINITE and a == ALGO_SGD) or (XLIN_IDN_DEV_FINITE and a != ALGO_SGD))
    with GILReleased(Python()):
        if host_scan:
            _finite(x, n_x, "X")
            _finite(y, n_y, "y")
        fit_device(a, x, n_x, y, n_y, n, d, ipl, fpl, n_out, n_fw, n_iw, out)
    return PythonObject(n_out)


def fit_fold_binding(algo: PythonObject, ids: PythonObject, y_addr: PythonObject, dims: PythonObject,
                     ip: PythonObject, fp: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """`x_linear_fit` over a model-selection fold resident on the device
    (lane a8c-device-x-linear-prep-rfe; IDENTICAL GPU builds): `ids` is
    [base store id, base rows, index store id] in model selection's store
    (bindings/device_rows_door.mojo door_rows_store); the fold's n x d rows
    are gathered on the device straight into the fit's X buffer, the finite
    and range tests in the same pass (core/msel_device_x.mojo), and the fit
    runs on the store's context (`msel_device_x_context()`): no host gather
    and no upload of X. `dims`, `ip`, `fp` and the result are
    `x_linear_fit`'s. Ridge (algo 8: Ridge with weights, RidgeClassifier)
    only; any other algo is refused by name (its door gathers host rows).
    Returns [n_out, finite word, range word]: a nonzero word means the
    result is not to be read and the door refuses."""
    var a = Int(py=algo)
    if a != ALGO_RIDGE:
        raise Error("x_linear_fit_fold: only the Ridge fit (algo 8) reads a device fold; algo " + String(a))
    if len(ids) != 3:
        raise Error("x_linear_fit_fold: ids must hold 3 values")
    var base_id = Int(py=ids[0])
    var base_rows = Int(py=ids[1])
    var index_id = Int(py=ids[2])
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var n_y = Int(py=dims[3])
    var n_out = Int(py=dims[4])
    var n_ip = Int(py=dims[7])
    var n_fp = Int(py=dims[8])
    if n <= 0 or d <= 0 or n_out <= 0:
        raise Error("x_linear: positive dimensions required")
    if Int(py=dims[2]) != n * d:
        raise Error("x_linear_fit_fold: dims[2] must be n * d (the fold's X words)")
    var ipl = List[Int32](capacity=n_ip)
    for i in range(n_ip):  # small-loop(n_ip: integer hyperparameters of the route): Python parameter list, not data
        ipl.append(Int32(Int(py=ip[i])))
    var fpl = List[Float32](capacity=n_fp)
    for i in range(n_fp):  # small-loop(n_fp: float hyperparameters of the route): Python parameter list, not data
        fpl.append(Float32(Float64(py=fp[i])))
    var y = _fp(Int(py=y_addr))
    var out = _fp(Int(py=out_addr))
    var finite = Int32(0)
    var in_range = Int32(0)
    with GILReleased(Python()):
        var ctx = msel_device_x_context()
        var dx = ctx.enqueue_create_buffer[DType.float32](n * d)
        var dst = ctx.enqueue_create_buffer[DType.int32](2)
        enqueue_fill[DType.int32](ctx, dst, Int32(0))
        msel_device_x_f32(ctx, base_id, base_rows, d, index_id, n, dx, dst.unsafe_ptr(), 0)
        var words = ctx.enqueue_create_host_buffer[DType.int32](2)
        ctx.enqueue_copy(dst_buf=words, src_buf=dst)
        # the gather tested X (n_x_test 0); y is tested by the fit as before.
        # ridge_resident synchronizes once at its end, which delivers words.
        ridge_resident(ctx.copy(), dx, 0, y, n_y, n, d, ipl, fpl, n_out, out)
        finite = words.unsafe_ptr().unsafe_load(0)
        in_range = words.unsafe_ptr().unsafe_load(1)
        _ = words^
        _ = dst^
        _ = dx^
    return Python.list(n_out, Int(finite), Int(in_range))


def decision_binding(x_addr: PythonObject, wb_addr: PythonObject, dims: PythonObject,
                     out_addr: PythonObject) raises -> PythonObject:
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var k = Int(py=dims[2])
    var link = Int(py=dims[3])
    if n < 0 or d < 0 or k <= 0:
        raise Error("x_linear: positive dimensions required")
    # dims[4] (optional): the host address of the deferred finite status
    # word of X (lane a1b; core/admitted_finite_status.mojo), 0 or absent
    # when the door tested X.
    var status = Int(py=dims[4]) if len(dims) > 4 else 0
    var x = _fp(Int(py=x_addr))
    var wb = _fp(Int(py=wb_addr))
    var out = _fp(Int(py=out_addr))
    with GILReleased(Python()):
        decision_device(x, wb, n, d, k, link, out, status)
    return PythonObject(n * k)


def decision_codes_binding(x_addr: PythonObject, wb_addr: PythonObject, dims: PythonObject,
                           out_addr: PythonObject) raises -> PythonObject:
    """Each row's class code (int32) of link(X W^T + b): dims = [n, d, k,
    link, strict, below, above] (k == 1: the threshold at 0; k > 1: the
    argmax), on the device (lane pyglue-numeric)."""
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var k = Int(py=dims[2])
    var link = Int(py=dims[3])
    var strict = Int(py=dims[4])
    var below = Int(py=dims[5])
    var above = Int(py=dims[6])
    if n < 0 or d < 0 or k <= 0:
        raise Error("x_linear: positive dimensions required")
    # dims[7] (optional): the deferred finite status word's host address.
    var status = Int(py=dims[7]) if len(dims) > 7 else 0
    var x = _fp(Int(py=x_addr))
    var wb = _fp(Int(py=wb_addr))
    var out = IP(unsafe_from_address=Int(py=out_addr))
    with GILReleased(Python()):
        decision_codes_device(x, wb, n, d, k, link, strict, below, above, out, status)
    return PythonObject(n)


# TOMBSTONE: x_linear_glm_ydom, x_linear_cls1_flags (dead native entries: constant / switch-word probes nothing reads) deleted 2026-10-26 by 44fc3b5bb; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_linear.patch


def decision_rows_binding(x_addr: PythonObject, wb_addr: PythonObject, dims: PythonObject,
                          out_addr: PythonObject) raises -> PythonObject:
    """predict_proba in one call: link(X W^T + b) then `py2mojo_rows` mode
    `rows_mode` (2 SGD_PROBA, 3 LRCV_PROBA) on the device; dims = [n, d, k,
    link, rows_mode]; out float32 n x max(k, 2). Returns n."""
    # ne2-g-linear (N362): the scores stay on the device between the two steps.
    if len(dims) != 5 and len(dims) != 6:
        raise Error("x_linear_decision_rows: dims must contain n, d, k, link, rows_mode (and the status address)")
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var k = Int(py=dims[2])
    var link = Int(py=dims[3])
    var mode = Int(py=dims[4])
    var status = Int(py=dims[5]) if len(dims) > 5 else 0
    if n < 0 or d < 0 or k <= 0:
        raise Error("x_linear: positive dimensions required")
    var x = _fp(Int(py=x_addr))
    var wb = _fp(Int(py=wb_addr))
    var out = _fp(Int(py=out_addr))
    with GILReleased(Python()):
        decision_rows_device(x, wb, n, d, k, link, mode, out, status)
    return PythonObject(n)


def decision_huber_out_binding(x_addr: PythonObject, wb_addr: PythonObject, dims: PythonObject,
                               out_addr: PythonObject) raises -> PythonObject:
    """HuberRegressor.outliers_ in one call: the one-target decision on the
    training rows, then |y - pred| > thr (`py2mojo_rows` ROWS_HUBER_OUT) on
    the device; dims = [n, d, link, y_addr, thr]; out uint8 n. Returns n."""
    # ne2-g-linear (N302): the prediction stays on the device.
    if len(dims) != 5:
        raise Error("x_linear_decision_huber_out: dims must contain n, d, link, y address, thr")
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var link = Int(py=dims[2])
    if n < 0 or d < 0:
        raise Error("x_linear: positive dimensions required")
    var y = _fp(Int(py=dims[3]))
    var thr = Float64(py=dims[4])
    var x = _fp(Int(py=x_addr))
    var wb = _fp(Int(py=wb_addr))
    var oa = Int(py=out_addr)
    if oa == 0:
        raise Error("x_linear: null flag buffer address")
    var flags = MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=oa)
    with GILReleased(Python()):
        decision_huber_out_device(x, wb, n, d, link, y, thr, flags)
    return PythonObject(n)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))



def py2mojo_rows_binding(mode: PythonObject, src_addr: PythonObject, dst_addr: PythonObject,
                         params: PythonObject) raises -> PythonObject:
    """lane/apple-fast-py2mojo-linear: per-row probability glue on the
    device (`core/py2mojo_rows.mojo`)."""
    return py2mojo_rows_device_binding(_p2m_ctx(), mode, src_addr, dst_addr, params)


def py2mojo_linear_flags_binding() raises -> PythonObject:
    return PythonObject(py2mojo_linear_flags())


def spearman_sign_binding(x_addr: PythonObject, y_addr: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    """The sign (-1, 0, 1) of Spearman's rho of n float32 x and y, exact
    (x_linear/spearman.mojo, lane cpu2-l10-linear)."""
    var n = Int(py=n_obj)
    var xa = Int(py=x_addr)
    var ya = Int(py=y_addr)
    if n <= 0 or xa == 0 or ya == 0:
        raise Error("x_linear spearman: n > 0 and two buffers required")
    var sign = 0
    with GILReleased(Python()):
        sign = spearman_sign_device(_p2m_ctx(), FP(unsafe_from_address=xa), FP(unsafe_from_address=ya), n)
    return PythonObject(sign)


def class_prep_binding(codes_addr: PythonObject, sw_addr: PythonObject, cw_addr: PythonObject,
                       dims: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """Class counts, 'balanced' class weights and per-row class weights on
    the device (x_linear/class_prep.mojo, lane cpu2-l10-linear). codes: n
    int32 class codes in [0, k); sw: n float32 sample weights or 0; cw: k
    float32 (written when balanced, else read); out: n float32 row weights
    or 0. dims = [n, k, balanced, weighted]. Returns the largest unweighted
    class count."""
    var n = Int(py=dims[0])
    var k = Int(py=dims[1])
    var balanced = Int(py=dims[2]) != 0
    var weighted = Int(py=dims[3]) != 0
    var ca = Int(py=codes_addr)
    var sa = Int(py=sw_addr)
    var wa = Int(py=cw_addr)
    var oa = Int(py=out_addr)
    if n <= 0 or k < 1 or ca == 0 or wa == 0:
        raise Error("x_linear class prep: n > 0, k > 0, codes and class weights required")
    if weighted and sa == 0:
        raise Error("x_linear class prep: weighted counts need sample weights")
    var largest = 0
    with GILReleased(Python()):
        largest = class_prep_device(
            _p2m_ctx(), IP(unsafe_from_address=ca), FP(unsafe_from_address=sa if sa != 0 else wa), sa != 0,
            FP(unsafe_from_address=wa), n, k, balanced, weighted, FP(unsafe_from_address=oa if oa != 0 else wa),
            oa != 0)
    return PythonObject(largest)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_linear.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_linear() -> PythonObject:
    comptime assert LINEAR_GUARDS_OK, "invalid integrated experiment configuration (linear guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_x_linear")
        m.def_function[fit_binding]("x_linear_fit")
        # lane a8c: the fold store is the IDENTICAL base binding's, so only
        # an IDENTICAL GPU build reads a device fold
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            m.def_function[fit_fold_binding]("x_linear_fit_fold")
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
        m.def_function[best_alpha_binding]("x_linear_best_alpha")
        # TOMBSTONE: x_linear_glm_ydom, x_linear_cls1_flags (dead native entries: constant / switch-word probes nothing reads) deleted 2026-10-26 by 44fc3b5bb; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_linear.patch
        m.def_function[class_prep_binding]("x_linear_class_prep")
        m.def_function[spearman_sign_binding]("x_linear_spearman_sign")
        m.def_function[decision_binding]("x_linear_decision")
        m.def_function[decision_codes_binding]("x_linear_decision_codes")
        m.def_function[decision_rows_binding]("x_linear_decision_rows")
        m.def_function[decision_huber_out_binding]("x_linear_decision_huber_out")
        m.def_function[numeric_mode_binding]("x_linear_numeric_mode")
        # TOMBSTONE: x_linear_glm_ydom, x_linear_cls1_flags (dead native entries: constant / switch-word probes nothing reads) deleted 2026-10-26 by 44fc3b5bb; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_linear.patch
        m.def_function[vendor_binding]("x_linear_vendor")
        m.def_function[py2mojo_rows_binding]("py2mojo_rows")
        m.def_function[py2mojo_linear_flags_binding]("py2mojo_linear_flags")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_linear: ", e))
