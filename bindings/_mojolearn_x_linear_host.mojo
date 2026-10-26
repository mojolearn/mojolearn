# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_linear` (lane/algos-linear, 2026-09-27).
HOST ONLY: the GPU binding's export names and address contract, running the
same x_linear/ source (`fit_dispatch`, `decision_one`) directly on the CPU."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from core.py2mojo_rows_pyapi import py2mojo_rows_host_binding
from svm.host.scale_gamma_host_pyapi import py2mojo_linear_flags_binding
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_linear.spearman import spearman_sign_host
from x_linear.ops import FP, IP, X_LINEAR_HOST_SABOTAGE
from x_linear.ridgecv import ridge_best_alpha
from x_linear.class_prep import class_prep_host
from x_linear.x_linear_host_impl import x_linear_fit_host, x_linear_decision_host, x_linear_decision_codes_host


# TOMBSTONE: x_linear_glm_ydom, x_linear_cls1_flags (dead native entries: constant / switch-word probes nothing reads) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_linear.patch


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
        sign = spearman_sign_host(FP(unsafe_from_address=xa), FP(unsafe_from_address=ya), n)
    return PythonObject(sign)


def class_prep_binding(codes_addr: PythonObject, sw_addr: PythonObject, cw_addr: PythonObject,
                       dims: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """The GPU binding's `x_linear_class_prep` on the CPU (x_linear/class_prep.mojo):
    the same contract, the same words."""
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
        largest = class_prep_host(
            IP(unsafe_from_address=ca), FP(unsafe_from_address=sa if sa != 0 else wa), sa != 0,
            FP(unsafe_from_address=wa), n, k, balanced, weighted, FP(unsafe_from_address=oa if oa != 0 else wa),
            oa != 0)
    return PythonObject(largest)


def best_alpha_binding(scores_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """Lane py-runtime round 2: RidgeCV's best alpha index over the n mean
    CV scores (`ridge_best_alpha`, x_linear/ridgecv.mojo); -1 if all NaN."""
    var na = Int(py=n)
    if na <= 0:
        return PythonObject(-1)
    return PythonObject(ridge_best_alpha(FP(unsafe_from_address=Int(py=scores_addr)), na))


def fit_binding(algo: PythonObject, x_addr: PythonObject, y_addr: PythonObject, dims: PythonObject,
                ip: PythonObject, fp: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    var dl = List[Int](capacity=9)
    for i in range(9):  # small-loop(the nine dims words)
        dl.append(Int(py=dims[i]))
    var n_ip = dl[7]
    var n_fp = dl[8]
    var ipl = List[Int32](capacity=max(n_ip, 1))
    for i in range(n_ip):  # small-loop(the integer parameter words)
        ipl.append(Int32(Int(py=ip[i])))
    var fpl = List[Float32](capacity=max(n_fp, 1))
    for i in range(n_fp):  # small-loop(the float parameter words)
        fpl.append(Float32(Float64(py=fp[i])))
    var a = Int(py=algo)
    var xa = Int(py=x_addr)
    var ya = Int(py=y_addr)
    var oa = Int(py=out_addr)
    var n_out = 0
    with GILReleased(Python()):
        n_out = x_linear_fit_host(a, xa, ya, dl, ipl, fpl, oa)
    _ = ipl^
    _ = fpl^
    return PythonObject(n_out)


def decision_binding(x_addr: PythonObject, wb_addr: PythonObject, dims: PythonObject,
                     out_addr: PythonObject) raises -> PythonObject:
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var k = Int(py=dims[2])
    var link = Int(py=dims[3])
    var xa = Int(py=x_addr)
    var wa = Int(py=wb_addr)
    var oa = Int(py=out_addr)
    var cells = 0
    with GILReleased(Python()):
        cells = x_linear_decision_host(xa, wa, n, d, k, link, oa)
    return PythonObject(cells)


def decision_codes_binding(x_addr: PythonObject, wb_addr: PythonObject, dims: PythonObject,
                           out_addr: PythonObject) raises -> PythonObject:
    """The GPU binding's `x_linear_decision_codes` on the host: the same
    `decision_one` cells, then `decision_code_row` per row."""
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var k = Int(py=dims[2])
    var link = Int(py=dims[3])
    var strict = Int(py=dims[4])
    var below = Int(py=dims[5])
    var above = Int(py=dims[6])
    var xa = Int(py=x_addr)
    var wa = Int(py=wb_addr)
    var oa = Int(py=out_addr)
    var rows = 0
    with GILReleased(Python()):
        rows = x_linear_decision_codes_host(xa, wa, n, d, k, link, strict, below, above, oa)
    return PythonObject(rows)


def x_linear_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_linear_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_linear_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_linear host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_linear_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_LINEAR_HOST_SABOTAGE)


def x_linear_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def x_linear_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_linear_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_linear_host() -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_linear_host")
        m.def_function[x_linear_host_numeric_mode_binding]("x_linear_host_numeric_mode")
        m.def_function[x_linear_host_vendor_binding]("x_linear_host_vendor")
        m.def_function[x_linear_host_column_binding]("x_linear_host_column")
        m.def_function[x_linear_host_sabotage_binding]("x_linear_host_sabotage")
        m.def_function[fit_binding]("x_linear_fit")
        m.def_function[best_alpha_binding]("x_linear_best_alpha")
        # TOMBSTONE: x_linear_glm_ydom, x_linear_cls1_flags (dead native entries: constant / switch-word probes nothing reads) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_linear.patch
        m.def_function[class_prep_binding]("x_linear_class_prep")
        m.def_function[spearman_sign_binding]("x_linear_spearman_sign")
        m.def_function[decision_binding]("x_linear_decision")
        m.def_function[decision_codes_binding]("x_linear_decision_codes")
        m.def_function[x_linear_numeric_mode_binding]("x_linear_numeric_mode")
        m.def_function[x_linear_vendor_binding]("x_linear_vendor")
        m.def_function[py2mojo_rows_host_binding]("py2mojo_rows")
        m.def_function[py2mojo_linear_flags_binding]("py2mojo_linear_flags")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_linear_host: ", e))
