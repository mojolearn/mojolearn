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
from x_cnn.device import spmm_device as spmm_impl
from x_cnn.device import gcn_norm_device as gcn_norm_impl
from x_cnn.device import dropout2d_device as dropout2d_impl
from x_cnn.device import mul_device as mul_impl
from x_cnn.device import batchnorm_forward_device as batchnorm_forward_impl
from x_cnn.device import batchnorm_backward_device as batchnorm_backward_impl
from x_cnn.device import relu_forward_device as relu_forward_impl
from x_cnn.device import relu_backward_device as relu_backward_impl
from x_cnn.device import add_device as add_impl
from x_cnn.device import linear_forward_device as linear_forward_impl
from x_cnn.device import linear_backward_device as linear_backward_impl
from x_cnn.device import softmax_xent_device as softmax_xent_impl
from x_cnn.device import sgd_device as sgd_impl
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


def _count(params: PythonObject) raises -> Int:
    var n = Int(py=params[0])
    if n <= 0:
        raise Error("x_cnn: a positive element count is required")
    return n


def relu_forward_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var x = read_f32(Int(py=x_addr), n)
    var po = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var y = relu_forward_impl(x)
        copy_f32(y.unsafe_ptr(), po, n)
    return PythonObject(n)


def relu_backward_binding(x_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var x = read_f32(Int(py=x_addr), n)
    var g = read_f32(Int(py=g_addr), n)
    var po = f32_ptr(Int(py=dx_addr))
    with GILReleased(Python()):
        var y = relu_backward_impl(x, g)
        copy_f32(y.unsafe_ptr(), po, n)
    return PythonObject(n)


def add_binding(a_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var a = read_f32(Int(py=a_addr), n)
    var b = read_f32(Int(py=b_addr), n)
    var po = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var y = add_impl(a, b)
        copy_f32(y.unsafe_ptr(), po, n)
    return PythonObject(n)


def _lin(params: PythonObject) raises -> Tuple[Int, Int, Int]:
    var n = Int(py=params[0])
    var d_in = Int(py=params[1])
    var d_out = Int(py=params[2])
    if n <= 0 or d_in <= 0 or d_out <= 0:
        raise Error("x_cnn linear: positive rows, in and out features required")
    return (n, d_in, d_out)


def linear_forward_binding(x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var s = _lin(params)
    var x = read_f32(Int(py=x_addr), s[0] * s[1])
    var w = read_f32(Int(py=w_addr), s[2] * s[1])
    var b = read_f32(Int(py=b_addr), s[2])
    var po = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var y = linear_forward_impl(x, w, b, s[0], s[1], s[2])
        copy_f32(y.unsafe_ptr(), po, s[0] * s[2])
    return PythonObject(s[0] * s[2])


def linear_backward_binding(
    x_addr: PythonObject, w_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject,
    dw_addr: PythonObject, db_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    var s = _lin(params)
    var n = s[0]
    var d_in = s[1]
    var d_out = s[2]
    var x = read_f32(Int(py=x_addr), n * d_in)
    var w = read_f32(Int(py=w_addr), d_out * d_in)
    var g = read_f32(Int(py=g_addr), n * d_out)
    var pdx = f32_ptr(Int(py=dx_addr))
    var pdw = f32_ptr(Int(py=dw_addr))
    var pdb = f32_ptr(Int(py=db_addr))
    with GILReleased(Python()):
        var r = linear_backward_impl(x, w, g, n, d_in, d_out)
        var base = r.unsafe_ptr()
        copy_f32(base, pdx, n * d_in)
        copy_f32(base + n * d_in, pdw, d_out * d_in)
        copy_f32(base + n * d_in + d_out * d_in, pdb, d_out)
    return PythonObject(n)


def softmax_xent_binding(
    logits_addr: PythonObject, labels_addr: PythonObject, grad_addr: PythonObject, proba_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The mean cross entropy (a float; 0 when every label is < 0)."""
    var n = Int(py=params[0])
    var k = Int(py=params[1])
    if n <= 0 or k <= 0:
        raise Error("x_cnn softmax: positive rows and classes required")
    var logits = read_f32(Int(py=logits_addr), n * k)
    var labels = read_i32(Int(py=labels_addr), n)
    for i in range(n):
        if Int(labels[i]) >= k:
            raise Error("x_cnn softmax: a label is not a class index")
    var pg = f32_ptr(Int(py=grad_addr))
    var pp = f32_ptr(Int(py=proba_addr))
    var loss = Float32(0)
    with GILReleased(Python()):
        var r = softmax_xent_impl(logits, labels, n, k)
        var base = r.unsafe_ptr()
        copy_f32(base, pg, n * k)
        copy_f32(base + n * k, pp, n * k)
        loss = r[2 * n * k]
    return PythonObject(Float64(loss))


def sgd_binding(w_addr: PythonObject, g_addr: PythonObject, v_addr: PythonObject, params: PythonObject, hyper: PythonObject) raises -> PythonObject:
    """In place: w and the momentum buffer v. hyper = [lr, momentum, weight_decay]."""
    var n = _count(params)
    var h = List[Float32]()
    for k in range(3):
        h.append(Float32(Float64(py=hyper[k])))
    var w = read_f32(Int(py=w_addr), n)
    var g = read_f32(Int(py=g_addr), n)
    var v = read_f32(Int(py=v_addr), n)
    var pw = f32_ptr(Int(py=w_addr))
    var pv = f32_ptr(Int(py=v_addr))
    with GILReleased(Python()):
        var r = sgd_impl(w, g, v, h)
        copy_f32(r.unsafe_ptr(), pw, n)
        copy_f32(r.unsafe_ptr() + n, pv, n)
    return PythonObject(n)


def _bn_prm(params: PythonObject) raises -> List[Int32]:
    var N = Int(py=params[0])
    var C = Int(py=params[1])
    var HW = Int(py=params[2])
    if N <= 0 or C <= 0 or HW <= 0:
        raise Error("x_cnn batchnorm: positive N, C and H*W required")
    var out: List[Int32] = [Int32(N), Int32(C), Int32(HW)]
    return out^


def batchnorm_forward_binding(
    x_addr: PythonObject, y_addr: PythonObject, running_addr: PythonObject, aux_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """params = [N, C, HW, training]; running (2C) and aux (2 + 7C) update in place."""
    var prm = _bn_prm(params)
    var training = Int(py=params[3]) != 0
    var N = Int(prm[0]); var C = Int(prm[1]); var HW = Int(prm[2])
    if training and N * HW < 2:
        raise Error("x_cnn batchnorm: expected more than 1 value per channel when training")
    var total = N * C * HW
    var x = read_f32(Int(py=x_addr), total)
    var running = read_f32(Int(py=running_addr), 2 * C)
    var aux = read_f32(Int(py=aux_addr), 2 + 7 * C)
    var pyo = f32_ptr(Int(py=y_addr))
    var pr = f32_ptr(Int(py=running_addr))
    var pa = f32_ptr(Int(py=aux_addr))
    with GILReleased(Python()):
        var r = batchnorm_forward_impl(x, running, aux, prm, training)
        copy_f32(r.unsafe_ptr(), pyo, total)
        copy_f32(r.unsafe_ptr() + total, pr, 2 * C)
        copy_f32(r.unsafe_ptr() + total + 2 * C, pa, 2 + 7 * C)
    return PythonObject(total)


def batchnorm_backward_binding(
    x_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject, aux_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [N, C, HW, training]; aux updates in place (sum_g, sum_gx)."""
    var prm = _bn_prm(params)
    var training = Int(py=params[3]) != 0
    var C = Int(prm[1])
    var total = Int(prm[0]) * C * Int(prm[2])
    var x = read_f32(Int(py=x_addr), total)
    var g = read_f32(Int(py=g_addr), total)
    var aux = read_f32(Int(py=aux_addr), 2 + 7 * C)
    var pd = f32_ptr(Int(py=dx_addr))
    var pa = f32_ptr(Int(py=aux_addr))
    with GILReleased(Python()):
        var r = batchnorm_backward_impl(x, g, aux, prm, training)
        copy_f32(r.unsafe_ptr(), pd, total)
        copy_f32(r.unsafe_ptr() + total, pa, 2 + 7 * C)
    return PythonObject(total)


def dropout2d_binding(x_addr: PythonObject, y_addr: PythonObject, mask_addr: PythonObject, params: PythonObject, drop_p: PythonObject) raises -> PythonObject:
    """params = [N, C, HW, seed_lo, seed_hi, thresh_hi, thresh_lo] (the seed
    words and the threshold halves as non-negative ints)."""
    var N = Int(py=params[0])
    var C = Int(py=params[1])
    var HW = Int(py=params[2])
    if N <= 0 or C <= 0 or HW <= 0:
        raise Error("x_cnn dropout2d: positive N, C and H*W required")
    var prm = List[Int32]()
    for k in range(7):
        var v = Int(py=params[k])
        if v < 0:
            raise Error("x_cnn dropout2d: negative parameter")
        prm.append(Int32(Int64(v) - Int64(4294967296)) if v > 2147483647 else Int32(v))
    var h: List[Float32] = [Float32(Float64(py=drop_p))]
    var total = N * C * HW
    var x = read_f32(Int(py=x_addr), total)
    var pyo = f32_ptr(Int(py=y_addr))
    var pm = f32_ptr(Int(py=mask_addr))
    with GILReleased(Python()):
        var r = dropout2d_impl(x, prm, h)
        copy_f32(r.unsafe_ptr(), pyo, total)
        copy_f32(r.unsafe_ptr() + total, pm, total)
    return PythonObject(total)


def mul_binding(a_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var a = read_f32(Int(py=a_addr), n)
    var b = read_f32(Int(py=b_addr), n)
    var po = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var y = mul_impl(a, b)
        copy_f32(y.unsafe_ptr(), po, n)
    return PythonObject(n)


def _csr(csr_addr: PythonObject, params: PythonObject) raises -> Tuple[List[Int32], List[Int32]]:
    """Validate a CSR block [rowptr | col | row] against params [n, F, nnz, mode]."""
    var n = Int(py=params[0])
    var F = Int(py=params[1])
    var nnz = Int(py=params[2])
    var mode = Int(py=params[3])
    if n <= 0 or F <= 0 or nnz < 0 or mode < 0 or mode > 2:
        raise Error("x_cnn spmm: positive n and F, nnz >= 0, mode in {0, 1, 2}")
    var csr = read_i32(Int(py=csr_addr), n + 1 + 2 * nnz)
    if Int(csr[0]) != 0 or Int(csr[n]) != nnz:
        raise Error("x_cnn spmm: rowptr must start at 0 and end at nnz")
    for r in range(n):
        if csr[r + 1] < csr[r]:
            raise Error("x_cnn spmm: rowptr must be non-decreasing")
    for e in range(nnz):
        var c = Int(csr[n + 1 + e])
        var rr = Int(csr[n + 1 + nnz + e])
        if c < 0 or c >= n or rr < 0 or rr >= n:
            raise Error("x_cnn spmm: a column or row index is out of range")
    var prm: List[Int32] = [Int32(n), Int32(F), Int32(nnz), Int32(mode)]
    return (csr^, prm^)


def spmm_binding(vals_addr: PythonObject, h_addr: PythonObject, out_addr: PythonObject, csr_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """out (n x F) = the CSR fold of h (n x F); see x_cnn/ops.mojo spmm_at."""
    var t = _csr(csr_addr, params)
    var n = Int(t[1][0])
    var F = Int(t[1][1])
    var nnz = Int(t[1][2])
    var vals = read_f32(Int(py=vals_addr), nnz if nnz > 0 else 1) if nnz > 0 else List[Float32](length=1, fill=Float32(0))
    var h = read_f32(Int(py=h_addr), n * F)
    var po = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var y = spmm_impl(vals, h, t[0], t[1])
        copy_f32(y.unsafe_ptr(), po, n * F)
    return PythonObject(n * F)


def gcn_norm_binding(w_addr: PythonObject, vals_addr: PythonObject, csr_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """vals[e] = PyG gcn_norm of the CSR (rows = targets, self loops included)."""
    var t = _csr(csr_addr, params)
    var nnz = Int(t[1][2])
    if nnz <= 0:
        raise Error("x_cnn gcn_norm: at least one edge (the self loops) is required")
    var w = read_f32(Int(py=w_addr), nnz)
    var pv = f32_ptr(Int(py=vals_addr))
    with GILReleased(Python()):
        var y = gcn_norm_impl(w, t[0], t[1])
        copy_f32(y.unsafe_ptr(), pv, nnz)
    return PythonObject(nnz)


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
        m.def_function[relu_forward_binding]("x_cnn_relu_forward")
        m.def_function[relu_backward_binding]("x_cnn_relu_backward")
        m.def_function[add_binding]("x_cnn_add")
        m.def_function[linear_forward_binding]("x_cnn_linear_forward")
        m.def_function[linear_backward_binding]("x_cnn_linear_backward")
        m.def_function[softmax_xent_binding]("x_cnn_softmax_xent")
        m.def_function[sgd_binding]("x_cnn_sgd")
        m.def_function[batchnorm_forward_binding]("x_cnn_batchnorm_forward")
        m.def_function[batchnorm_backward_binding]("x_cnn_batchnorm_backward")
        m.def_function[dropout2d_binding]("x_cnn_dropout2d")
        m.def_function[mul_binding]("x_cnn_mul")
        m.def_function[spmm_binding]("x_cnn_spmm")
        m.def_function[gcn_norm_binding]("x_cnn_gcn_norm")
        m.def_function[numeric_mode_binding]("x_cnn_numeric_mode")
        m.def_function[vendor_binding]("x_cnn_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cnn: ", e))
