# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_linear` (lane/algos-linear, 2026-09-27).
HOST ONLY: the GPU binding's export names and address contract, running the
same x_linear/ source (`fit_dispatch`, `decision_one`) directly on the CPU."""
from std.os import abort
from experiments.classical_identical_ideas.linear_controls import C13_LOGCV_WEIGHTS, ENETCV_FOLD_BLOCKS
from x_linear.enetcv_blocks import fb_host_words
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from core.py2mojo_rows import py2mojo_rows_host_binding
from svm.host.scale_gamma_host import py2mojo_linear_flags_binding
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_linear.spearman import spearman_sign_host
from x_linear.ops import FP, IP, X_LINEAR_HOST_SABOTAGE
from x_linear.ridgecv import ridge_best_alpha
from x_linear.dispatch import fit_dispatch, decision_one, decision_code_row, team_rows, team_own, ALGO_ISOTONIC, ALGO_LOGCV, ALGO_RIDGE, isotonic_abi_check
from x_linear.dispatch import ALGO_GLM
from x_linear.glm_ydom import XLIN_GLM_DEV_YDOM, GLM_YDOM_REFUSED, glm_ydom_host
from x_linear.logcv import logcv_fold_ids
from x_linear.class_prep import class_prep_host
from x_linear.isotonic_host import isotonic_fit_host
from x_linear.team import team_work, solo


def _fp(addr: Int) raises -> FP:
    if addr == 0:
        raise Error("x_linear: null float32 buffer address")
    return FP(unsafe_from_address=addr)


def _finite(p: FP, count: Int, name: String) raises:
    """The input check both columns run before a fit: NaN or infinity is refused by name."""
    for i in range(count):
        var v = p.unsafe_load(i)
        if not (v == v) or v > Float32(3.4028234e38) or v < Float32(-3.4028234e38):
            raise Error(String("mojolearn: ", name, " contains NaN or infinity"))


def glm_ydom_binding() raises -> PythonObject:
    """1: a GLM fit checks its targets' range itself (x_linear/glm_ydom.mojo)."""
    return PythonObject(1)


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
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var n_out = Int(py=dims[4])
    var n_fw = Int(py=dims[5])
    var n_iw = Int(py=dims[6])
    var n_ip = Int(py=dims[7])
    var n_fp = Int(py=dims[8])
    if n <= 0 or d < 0 or n_out <= 0:
        raise Error("x_linear: positive dimensions required")
    isotonic_abi_check(Int(py=algo), n, Int(py=dims[3]), n_out, n_fw, n_iw, n_ip)
    var ipl = List[Int32](capacity=max(n_ip, 1))
    for i in range(n_ip):
        ipl.append(Int32(Int(py=ip[i])))
    var fpl = List[Float32](capacity=max(n_fp, 1))
    for i in range(n_fp):
        fpl.append(Float32(Float64(py=fp[i])))
    comptime if ENETCV_FOLD_BLOCKS:
        if Int(py=algo) == 9 and n_ip >= 4:
            n_fw += fb_host_words(d, Int(ipl[3]))
    comptime if C13_LOGCV_WEIGHTS:
        if Int(py=algo) == ALGO_LOGCV:
            n_fw += Int(ipl[4]) + 1
    var fw = List[Float32](length=max(n_fw, 1), fill=Float32(0))
    var iw = List[Int32](length=max(n_iw, 1), fill=Int32(0))
    var bufs = team_rows(Int(py=algo), IP(unsafe_from_address=Int(ipl.unsafe_ptr())))
    var own = team_own(Int(py=algo), d)
    var tw = List[Float32](length=team_work(n, bufs, own), fill=Float32(0))
    var a = Int(py=algo)
    var x = _fp(Int(py=x_addr))
    var y = _fp(Int(py=y_addr))
    var out = _fp(Int(py=out_addr))
    _finite(x, Int(py=dims[2]), "X")
    _finite(y, Int(py=dims[3]), "y")
    for i in range(n_out):
        out.unsafe_store(i, Float32(0))
    # lane fam2-linear: the GLM targets' range, the device grid's rule
    # (x_linear/glm_ydom.mojo); a refused fit returns -1 in the converged word
    comptime if XLIN_GLM_DEV_YDOM:
        if a == ALGO_GLM and n_fp > 0 and n_out >= d + 3 and glm_ydom_host(y, n, fpl[0]):
            out.unsafe_store(d + 2, GLM_YDOM_REFUSED)
            return PythonObject(n_out)
    # LogisticRegressionCV: the caller's fold ids are zeros; its StratifiedKFold
    # ids are built here from the labels (cgr-linear), into a copy of y
    var ycopy = List[Float32]()
    if a == ALGO_LOGCV:
        var ny = Int(py=dims[3])
        ycopy = List[Float32](length=max(ny, 1), fill=Float32(0))
        for i in range(ny):
            ycopy[i] = y.unsafe_load(i)
        y = FP(unsafe_from_address=Int(ycopy.unsafe_ptr()))
        logcv_fold_ids(y, n, IP(unsafe_from_address=Int(ipl.unsafe_ptr())))
    # RidgeClassifier: ip[4] == 1 means y holds the n int32 class codes (then
    # the n weights when ip[3]); the +-1 targets as the device builds them
    # (`c1_codes_targets_kernel`), the weights after them
    if a == ALGO_RIDGE and n_ip > 4 and Int(ipl[4]) == 1:
        var t_c = Int(ipl[0])
        var has_sw = Int(ipl[3]) != 0
        ycopy = List[Float32](length=n * t_c + (n if has_sw else 0), fill=Float32(0))
        var yi = IP(unsafe_from_address=Int(y))
        for i in range(n):
            var c = Int(yi.unsafe_load(i))
            if t_c == 1:
                ycopy[i] = Float32(1) if c == 1 else Float32(-1)
            else:
                for t in range(t_c):
                    ycopy[i * t_c + t] = Float32(1) if c == t else Float32(-1)
        if has_sw:
            for i in range(n):
                ycopy[n * t_c + i] = y.unsafe_load(n + i)
        y = FP(unsafe_from_address=Int(ycopy.unsafe_ptr()))
    with GILReleased(Python()):
        if a == ALGO_ISOTONIC:
            isotonic_fit_host(
                x, y, n, d,
                IP(unsafe_from_address=Int(ipl.unsafe_ptr())), FP(unsafe_from_address=Int(fpl.unsafe_ptr())),
                out, FP(unsafe_from_address=Int(fw.unsafe_ptr())), IP(unsafe_from_address=Int(iw.unsafe_ptr())),
            )
        else:
            fit_dispatch(
                solo(FP(unsafe_from_address=Int(tw.unsafe_ptr())), n, bufs, own), a, x, y, n, d,
                IP(unsafe_from_address=Int(ipl.unsafe_ptr())), FP(unsafe_from_address=Int(fpl.unsafe_ptr())),
                out, FP(unsafe_from_address=Int(fw.unsafe_ptr())), IP(unsafe_from_address=Int(iw.unsafe_ptr())),
            )
    _ = ipl^
    _ = fpl^
    _ = fw^
    _ = iw^
    _ = tw^
    _ = ycopy^
    return PythonObject(n_out)


def decision_binding(x_addr: PythonObject, wb_addr: PythonObject, dims: PythonObject,
                     out_addr: PythonObject) raises -> PythonObject:
    var n = Int(py=dims[0])
    var d = Int(py=dims[1])
    var k = Int(py=dims[2])
    var link = Int(py=dims[3])
    if n < 0 or d < 0 or k <= 0:
        raise Error("x_linear: positive dimensions required")
    var x = _fp(Int(py=x_addr))
    var wb = _fp(Int(py=wb_addr))
    var out = _fp(Int(py=out_addr))
    with GILReleased(Python()):
        for t in range(n * k):
            out.unsafe_store(t, decision_one(x, t // k, d, wb, t % k, link))
    return PythonObject(n * k)


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
    if n < 0 or d < 0 or k <= 0:
        raise Error("x_linear: positive dimensions required")
    var x = _fp(Int(py=x_addr))
    var wb = _fp(Int(py=wb_addr))
    var out = IP(unsafe_from_address=Int(py=out_addr))
    with GILReleased(Python()):
        var s = List[Float32](length=max(n * k, 1), fill=Float32(0))
        var sp = s.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        for t in range(n * k):
            sp.unsafe_store(t, decision_one(x, t // k, d, wb, t % k, link))
        for i in range(n):
            out.unsafe_store(i, decision_code_row(sp, i, k, strict, below, above))
        _ = s^
    return PythonObject(n)


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


@export
def PyInit__mojolearn_x_linear_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_linear_host")
        m.def_function[x_linear_host_numeric_mode_binding]("x_linear_host_numeric_mode")
        m.def_function[x_linear_host_vendor_binding]("x_linear_host_vendor")
        m.def_function[x_linear_host_column_binding]("x_linear_host_column")
        m.def_function[x_linear_host_sabotage_binding]("x_linear_host_sabotage")
        m.def_function[fit_binding]("x_linear_fit")
        m.def_function[best_alpha_binding]("x_linear_best_alpha")
        comptime if XLIN_GLM_DEV_YDOM:
            m.def_function[glm_ydom_binding]("x_linear_glm_ydom")
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
