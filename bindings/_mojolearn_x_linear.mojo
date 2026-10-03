# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE'S GPU BINDING (lane/algos-linear, 2026-09-27).

x_linear_fit(algo, x_addr, y_addr, dims, ip, fp, out_addr)
    dims = [n, d, n_x, n_y, n_out, n_fw, n_iw, n_ip, n_fp]
x_linear_decision(x_addr, wb_addr, dims, out_addr)
    dims = [n, d, k, link]; wb is k rows of (w_0..w_{d-1}, b)
The host binding (_mojolearn_x_linear_host.mojo) exports the same names with
the same address contract and runs the same x_linear/ source on the CPU."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from core.py2mojo_rows import py2mojo_rows_device_binding
from core.py2mojo_linear import py2mojo_linear_flags
from x_linear.device import linear_ctx as _p2m_ctx
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator
from std.sys.compile import is_defined
from x_linear.ops import FP, IP
from x_linear.device import fit_device, decision_device, decision_codes_device
from x_linear.dispatch import isotonic_abi_check
from x_linear.cls1_fast import cls1_flags
from x_linear.dispatch import ALGO_ISOTONIC
from x_linear.isotonic_fast import isotonic_fast


def _fp(addr: Int) raises -> FP:
    if addr == 0:
        raise Error("x_linear: null float32 buffer address")
    return FP(unsafe_from_address=addr)


def _finite(p: FP, count: Int, name: String) raises:
    """The input check both columns run before a fit: NaN or infinity is refused by name."""
    comptime if has_apple_gpu_accelerator() and is_defined["MOJOLEARN_X_LINEAR_FINITE_SIMD"]():
        # lane/linear-apple3 (WIP, opt-in): sixteen values at a time. v - v is 0 for a
        # finite v and NaN for an infinity or a NaN, so the running sum of
        # v - v is NaN exactly when some value is not finite: the scalar
        # test's verdict (the scalar walk was 16M branches on the host
        # before a fit of 1M x 16 could start).
        comptime W = 16
        var acc = SIMD[DType.float32, W](0)
        var i = 0
        while i + W <= count:
            var v = p.unsafe_load[width=W](i)
            acc = acc + (v - v)
            i += W
        var s = acc.reduce_add()
        while i < count:
            var v = p.unsafe_load(i)
            s = s + (v - v)
            i += 1
        if not (s == s):
            raise Error(String("mojolearn: ", name, " contains NaN or infinity"))
        return
    for i in range(count):
        var v = p.unsafe_load(i)
        if not (v == v) or v > Float32(3.4028234e38) or v < Float32(-3.4028234e38):
            raise Error(String("mojolearn: ", name, " contains NaN or infinity"))


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
    for i in range(n_ip):
        ipl.append(Int32(Int(py=ip[i])))
    var fpl = List[Float32](capacity=n_fp)
    for i in range(n_fp):
        fpl.append(Float32(Float64(py=fp[i])))
    var a = Int(py=algo)
    var x = _fp(Int(py=x_addr))
    var y = _fp(Int(py=y_addr))
    var out = _fp(Int(py=out_addr))
    _finite(x, Int(py=dims[2]), "X")
    _finite(y, Int(py=dims[3]), "y")
    with GILReleased(Python()):
        # FAST on Apple DEFAULT (lane/apple-fast-isotonic-knn): the isotonic
        # fit on the grid (x_linear/isotonic_fast.mojo) instead of the lead
        # thread of one block (x_linear/isotonic.mojo:129); the team fit when
        # it declines. M2 A/B ik-iso-pair-istella-b: PAR alone timed out,
        # PAR + PAIRMERGE 165.5 ms, r2 .188. -D MOJOLEARN_ISOTONIC_FAST_PAR_OFF
        # reverts.
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_ISOTONIC_FAST_PAR_OFF"]():
            if a == ALGO_ISOTONIC:
                var ctx = _p2m_ctx()
                if isotonic_fast(ctx, x, n_x, y, n_y, n, d, ipl, fpl, n_out, out):
                    return PythonObject(n_out)
        fit_device(a, x, n_x, y, n_y, n, d, ipl, fpl, n_out, n_fw, n_iw, out)
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
        decision_device(x, wb, n, d, k, link, out)
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
    var x = _fp(Int(py=x_addr))
    var wb = _fp(Int(py=wb_addr))
    var out = IP(unsafe_from_address=Int(py=out_addr))
    with GILReleased(Python()):
        decision_codes_device(x, wb, n, d, k, link, strict, below, above, out)
    return PythonObject(n)


def cls1_flags_binding() raises -> PythonObject:
    return PythonObject(cls1_flags())


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


@export
def PyInit__mojolearn_x_linear() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_linear")
        m.def_function[fit_binding]("x_linear_fit")
        m.def_function[decision_binding]("x_linear_decision")
        m.def_function[decision_codes_binding]("x_linear_decision_codes")
        m.def_function[numeric_mode_binding]("x_linear_numeric_mode")
        m.def_function[cls1_flags_binding]("x_linear_cls1_flags")
        m.def_function[vendor_binding]("x_linear_vendor")
        m.def_function[py2mojo_rows_binding]("py2mojo_rows")
        m.def_function[py2mojo_linear_flags_binding]("py2mojo_linear_flags")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_linear: ", e))
