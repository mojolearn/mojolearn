# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_linear` (lane/algos-linear, 2026-09-27).
HOST ONLY: the GPU binding's export names and address contract, running the
same x_linear/ source (`fit_dispatch`, `decision_one`) directly on the CPU."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from core.py2mojo_rows import py2mojo_rows_host_binding
from svm.host.scale_gamma_host import py2mojo_linear_flags_binding
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_linear.ops import FP, IP, X_LINEAR_HOST_SABOTAGE
from x_linear.dispatch import fit_dispatch, decision_one, team_rows, team_own, ALGO_ISOTONIC, ALGO_LOGCV
from x_linear.logcv import logcv_fold_ids
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
    var ipl = List[Int32](capacity=max(n_ip, 1))
    for i in range(n_ip):
        ipl.append(Int32(Int(py=ip[i])))
    var fpl = List[Float32](capacity=max(n_fp, 1))
    for i in range(n_fp):
        fpl.append(Float32(Float64(py=fp[i])))
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
        m.def_function[decision_binding]("x_linear_decision")
        m.def_function[x_linear_numeric_mode_binding]("x_linear_numeric_mode")
        m.def_function[x_linear_vendor_binding]("x_linear_vendor")
        m.def_function[py2mojo_rows_host_binding]("py2mojo_rows")
        m.def_function[py2mojo_linear_flags_binding]("py2mojo_linear_flags")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_linear_host: ", e))
