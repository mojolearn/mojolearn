# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE'S GPU BINDING (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md, lane 8).
Host addresses in, host addresses out; the work is x_cnn/device.mojo. The CPU
twin is bindings/_mojolearn_x_cnn_host.mojo, same names, same contract."""
from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32, copy_f32
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cnn.ops import CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, conv_params
from x_cnn.ops import PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW, pool_params
from x_cnn.device import conv2d_forward_device as conv2d_forward_impl
from x_cnn.device import conv2d_backward_device as conv2d_backward_impl
from x_cnn.device import gemm_device as gemm_impl
from x_cnn.device import maxpool2d_forward_device as maxpool2d_forward_impl
from x_cnn.device import maxpool2d_backward_device as maxpool2d_backward_impl
from x_cnn.device import avgpool2d_forward_device as avgpool2d_forward_impl
from x_cnn.device import avgpool2d_backward_device as avgpool2d_backward_impl


def _ints(params: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for k in range(Int(py=len(params))):
        out.append(Int(py=params[k]))
    return out^


def gemm_binding(a_addr: PythonObject, b_addr: PythonObject, c_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var op = Int(py=params[3])
    if m <= 0 or n <= 0 or k <= 0 or op < 0 or op > 2:
        raise Error("x_cnn gemm: positive m, n, k and op in {0, 1, 2} required")
    var a = read_f32(Int(py=a_addr), m * k)
    var b = read_f32(Int(py=b_addr), n * k)
    var output = f32_ptr(Int(py=c_addr))
    with GILReleased(Python()):
        var c = gemm_impl(a, b, m, n, k, op)
        copy_f32(c.unsafe_ptr(), output, m * n)
    return PythonObject(m * n)


def conv2d_forward_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    var prm = conv_params(_ints(params))
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var x = read_f32(Int(py=x_addr), N * C * Int(prm[CP_H]) * Int(prm[CP_W]))
    var w = read_f32(Int(py=w_addr), OC * ckk)
    var b = read_f32(Int(py=b_addr), OC)
    var total = N * OC * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var output = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var y = conv2d_forward_impl(x, w, b, prm)
        copy_f32(y.unsafe_ptr(), output, total)
    return PythonObject(total)


def conv2d_backward_binding(
    x_addr: PythonObject, w_addr: PythonObject, dout_addr: PythonObject, dx_addr: PythonObject,
    dw_addr: PythonObject, db_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    var prm = conv_params(_ints(params))
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var nx = N * C * Int(prm[CP_H]) * Int(prm[CP_W])
    var x = read_f32(Int(py=x_addr), nx)
    var w = read_f32(Int(py=w_addr), OC * ckk)
    var dout = read_f32(Int(py=dout_addr), N * OC * Int(prm[CP_OH]) * Int(prm[CP_OW]))
    var pdx = f32_ptr(Int(py=dx_addr))
    var pdw = f32_ptr(Int(py=dw_addr))
    var pdb = f32_ptr(Int(py=db_addr))
    with GILReleased(Python()):
        var r = conv2d_backward_impl(x, w, dout, prm)
        var base = r.unsafe_ptr()
        copy_f32(base, pdx, nx)
        copy_f32(base + nx, pdw, OC * ckk)
        copy_f32(base + nx + OC * ckk, pdb, OC)
    return PythonObject(nx)


def conv_shape_binding(params: PythonObject) raises -> PythonObject:
    var prm = conv_params(_ints(params))
    return Python.tuple(Int(prm[CP_OH]), Int(prm[CP_OW]))


def _pool_prm(params: PythonObject) raises -> List[Int32]:
    return pool_params(_ints(params))


def _pool_counts(prm: List[Int32]) -> Tuple[Int, Int]:
    var nc = Int(prm[PP_N]) * Int(prm[PP_C])
    return (nc * Int(prm[PP_H]) * Int(prm[PP_W]), nc * Int(prm[PP_OH]) * Int(prm[PP_OW]))


def pool_shape_binding(params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    return Python.tuple(Int(prm[PP_OH]), Int(prm[PP_OW]))


def maxpool2d_forward_binding(x_addr: PythonObject, out_addr: PythonObject, idx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    var c = _pool_counts(prm)
    var x = read_f32(Int(py=x_addr), c[0])
    var po = f32_ptr(Int(py=out_addr))
    var pi = i32_ptr(Int(py=idx_addr))
    with GILReleased(Python()):
        var idx = List[Int32]()
        var y = maxpool2d_forward_impl(x, prm, idx)
        copy_f32(y.unsafe_ptr(), po, c[1])
        for k in range(c[1]):
            pi.unsafe_store(k, idx[k])
    return PythonObject(c[1])


def maxpool2d_backward_binding(dout_addr: PythonObject, idx_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    var c = _pool_counts(prm)
    var dout = read_f32(Int(py=dout_addr), c[1])
    var idx = read_i32(Int(py=idx_addr), c[1])
    var pd = f32_ptr(Int(py=dx_addr))
    with GILReleased(Python()):
        var g = maxpool2d_backward_impl(dout, idx, prm)
        copy_f32(g.unsafe_ptr(), pd, c[0])
    return PythonObject(c[0])


def avgpool2d_forward_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    var c = _pool_counts(prm)
    var x = read_f32(Int(py=x_addr), c[0])
    var po = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var y = avgpool2d_forward_impl(x, prm)
        copy_f32(y.unsafe_ptr(), po, c[1])
    return PythonObject(c[1])


def avgpool2d_backward_binding(dout_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    var c = _pool_counts(prm)
    var dout = read_f32(Int(py=dout_addr), c[1])
    var pd = f32_ptr(Int(py=dx_addr))
    with GILReleased(Python()):
        var g = avgpool2d_backward_impl(dout, prm)
        copy_f32(g.unsafe_ptr(), pd, c[0])
    return PythonObject(c[0])


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_x_cnn() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_cnn")
        m.def_function[gemm_binding]("x_cnn_gemm")
        m.def_function[conv2d_forward_binding]("x_cnn_conv2d_forward")
        m.def_function[conv2d_backward_binding]("x_cnn_conv2d_backward")
        m.def_function[conv_shape_binding]("x_cnn_conv_shape")
        m.def_function[pool_shape_binding]("x_cnn_pool_shape")
        m.def_function[maxpool2d_forward_binding]("x_cnn_maxpool2d_forward")
        m.def_function[maxpool2d_backward_binding]("x_cnn_maxpool2d_backward")
        m.def_function[avgpool2d_forward_binding]("x_cnn_avgpool2d_forward")
        m.def_function[avgpool2d_backward_binding]("x_cnn_avgpool2d_backward")
        m.def_function[numeric_mode_binding]("x_cnn_numeric_mode")
        m.def_function[vendor_binding]("x_cnn_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cnn: ", e))
