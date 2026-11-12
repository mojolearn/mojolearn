# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE'S GPU BINDING (lane/algos-linear, 2026-09-27).

x_linear_fit(algo, x_addr, y_addr, dims, ip, fp, out_addr)
    dims = [n, d, n_x, n_y, n_out, n_fw, n_iw, n_ip, n_fp]
x_linear_decision(x_addr, wb_addr, dims, out_addr)
    dims = [n, d, k, link]; wb is k rows of (w_0..w_{d-1}, b)
The host binding (_mojolearn_x_linear_host.mojo) exports the same names with
the same address contract and runs the same x_linear/ source on the CPU."""
from x_linear.ridge_descriptor_device import RIDGE_FIT_DEVICE_API
from x_linear.x_linear_binding_run import _publish_linear_entry_table
from x_linear.ops_entry_device import (
    xl_fit_device_run, xl_fit_fold_device_run,
    xl_decision_device_run, xl_decision_codes_device_run, xl_decision_rows_device_run, xl_huber_out_device_run,
    xl_class_prep_device_run, xl_spearman_device_run, xl_best_alpha_run,
)
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from core.experiment_guards_linear import LINEAR_GUARDS_OK
from bindings.device_plan_binding import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from providers.python.core_py2mojo_rows_pyapi import py2mojo_rows_device_binding
from core.py2mojo_linear import py2mojo_linear_flags
from x_linear.device import linear_ctx as _p2m_ctx
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from std.sys.compile import is_defined
from checks.numerics import NUMERIC_IDENTICAL


def best_alpha_binding(scores_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """Lane py-runtime round 2: RidgeCV's best alpha index over the n mean
    CV scores (`ridge_best_alpha`, x_linear/ridgecv.mojo); -1 if all NaN.
    The native owner calls XL_OP_BEST_ALPHA; this export remains for the
    Python surface (_surface_linear.py)."""
    return PythonObject(xl_best_alpha_run(Int(py=scores_addr), Int(py=n)))


def ridge_fit_native_api_v1_binding() raises -> PythonObject:
    var api = RIDGE_FIT_DEVICE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def _int32_words(values: PythonObject, count: Int) raises -> List[Int32]:
    var words = List[Int32](capacity=count)
    for i in range(count):  # small-loop(n_ip: integer hyperparameters of the route): Python parameter list, not data
        words.append(Int32(Int(py=values[i])))
    return words^


def _float32_words(values: PythonObject, count: Int) raises -> List[Float32]:
    var words = List[Float32](capacity=count)
    for i in range(count):  # small-loop(n_fp: float hyperparameters of the route): Python parameter list, not data
        words.append(Float32(Float64(py=values[i])))
    return words^


def fit_binding(algo: PythonObject, x_addr: PythonObject, y_addr: PythonObject, dims: PythonObject,
                ip: PythonObject, fp: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    # The Python door of XL_OP_FIT (x_linear/ops_entry_device.mojo), kept for
    # the Python surface (_surface_linear.py, _expansion_linear.py); the
    # native owner bindings/x_linear_native.mojo calls the table.
    var ipl = _int32_words(ip, Int(py=dims[7]))
    var fpl = _float32_words(fp, Int(py=dims[8]))
    var a = Int(py=algo)
    var xa = Int(py=x_addr)
    var ya = Int(py=y_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var n_x = Int(py=dims[2])
    var n_y = Int(py=dims[3])
    var n_out = Int(py=dims[4])
    var n_fw = Int(py=dims[5])
    var n_iw = Int(py=dims[6])
    with GILReleased(Python()):
        n_out = xl_fit_device_run(a, xa, ya, n, d, n_x, n_y, n_out, n_fw, n_iw, ipl, fpl, oa)
    _ = ipl^
    _ = fpl^
    return PythonObject(n_out)


def fit_fold_binding(algo: PythonObject, ids: PythonObject, y_addr: PythonObject, dims: PythonObject,
                     ip: PythonObject, fp: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """`x_linear_fit` over a model-selection fold resident on the device
    (XL_OP_FIT_FOLD, x_linear/ops_entry_device.mojo). `ids` is [base store
    id, base rows, index store id]. Returns [n_out, finite word, range word]."""
    if len(ids) != 3:
        raise Error("x_linear_fit_fold: ids must hold 3 values")
    var ipl = _int32_words(ip, Int(py=dims[7]))
    var fpl = _float32_words(fp, Int(py=dims[8]))
    var a = Int(py=algo)
    var base_id = Int(py=ids[0])
    var base_rows = Int(py=ids[1])
    var index_id = Int(py=ids[2])
    var ya = Int(py=y_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var n_x = Int(py=dims[2])
    var n_y = Int(py=dims[3])
    var n_out = Int(py=dims[4])
    var finite = 0
    var in_range = 0
    with GILReleased(Python()):
        var words = xl_fit_fold_device_run(a, base_id, base_rows, index_id, ya, n, d, n_x, n_y, n_out, ipl, fpl, oa)
        finite = words[0]
        in_range = words[1]
    _ = ipl^
    _ = fpl^
    return Python.list(n_out, finite, in_range)


def decision_binding(x_addr: PythonObject, wb_addr: PythonObject, dims: PythonObject,
                     out_addr: PythonObject) raises -> PythonObject:
    # dims[4] (optional): the host address of the deferred finite status
    # word of X (lane a1b; core/admitted_finite_status.mojo), 0 or absent
    # when the door tested X.
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var k = Int(py=dims[2])
    var link = Int(py=dims[3])
    var status = Int(py=dims[4]) if len(dims) > 4 else 0
    var xa = Int(py=x_addr)
    var wa = Int(py=wb_addr)
    var oa = Int(py=out_addr)
    var cells = 0
    with GILReleased(Python()):
        cells = xl_decision_device_run(xa, wa, n, d, k, link, oa, status)
    return PythonObject(cells)


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
    # dims[7] (optional): the deferred finite status word's host address.
    var status = Int(py=dims[7]) if len(dims) > 7 else 0
    var xa = Int(py=x_addr)
    var wa = Int(py=wb_addr)
    var oa = Int(py=out_addr)
    var rows = 0
    with GILReleased(Python()):
        rows = xl_decision_codes_device_run(xa, wa, n, d, k, link, strict, below, above, oa, status)
    return PythonObject(rows)


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
    var xa = Int(py=x_addr)
    var wa = Int(py=wb_addr)
    var oa = Int(py=out_addr)
    var rows = 0
    with GILReleased(Python()):
        rows = xl_decision_rows_device_run(xa, wa, n, d, k, link, mode, oa, status)
    return PythonObject(rows)


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
    var ya = Int(py=dims[3])
    var thr = Float64(py=dims[4])
    var xa = Int(py=x_addr)
    var wa = Int(py=wb_addr)
    var oa = Int(py=out_addr)
    var rows = 0
    with GILReleased(Python()):
        rows = xl_huber_out_device_run(xa, wa, n, d, link, ya, thr, oa)
    return PythonObject(rows)


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
    var sign = 0
    with GILReleased(Python()):
        sign = xl_spearman_device_run(xa, ya, n)
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
    var largest = 0
    with GILReleased(Python()):
        largest = xl_class_prep_device_run(ca, sa, wa, oa, n, k, balanced, weighted)
    return PythonObject(largest)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_linear.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).


def build__mojolearn_x_linear() -> PythonObject:
    comptime assert LINEAR_GUARDS_OK, "invalid integrated experiment configuration (linear guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_x_linear")
        m.def_function[fit_binding]("x_linear_fit")
        m.def_function[ridge_fit_native_api_v1_binding]("ridge_fit_native_api_v1")
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
        var published_module = m.finalize()
        _publish_linear_entry_table()
        return published_module
    except e:
        abort(String("failed to create _mojolearn_x_linear: ", e))
