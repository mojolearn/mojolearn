# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_cnn` (the CNN expansion lane). HOST ONLY;
the GPU binding's export names and address contract, the work is
x_cnn/host/ops_host.mojo."""
from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32, copy_f32
from std.os import abort
from std.memory import alloc, memcpy, memset_zero
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cnn.ops import CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, conv_params
from x_cnn.ops import PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW, pool_params
from x_cnn.host.ops_host import X_CNN_HOST_SABOTAGE
from x_cnn.host.ops_host import conv2d_forward_host as conv2d_forward_impl
from x_cnn.host.ops_host import conv2d_backward_host as conv2d_backward_impl
from x_cnn.host.ops_host import gemm_host as gemm_impl
from x_cnn.host.ops_host import adam_host as adam_impl
from x_cnn.host.ops_host import graph_op_host as graph_op_impl
from x_cnn.host.ops_host import adaptive_pool_host as adaptive_pool_impl
from x_cnn.host.ops_host import pad2d_forward_host as pad2d_forward_impl
from x_cnn.host.ops_host import pad2d_backward_host as pad2d_backward_impl
from x_cnn.host.ops_host import spmm_host as spmm_impl
from x_cnn.host.ops_host import gcn_norm_host as gcn_norm_impl
from x_cnn.host.ops_host import dropout2d_host as dropout2d_impl
from x_cnn.host.ops_host import mul_host as mul_impl
from x_cnn.host.ops_host import batchnorm_forward_host as batchnorm_forward_impl
from x_cnn.host.ops_host import batchnorm_backward_host as batchnorm_backward_impl
from x_cnn.host.ops_host import relu_forward_host as relu_forward_impl
from x_cnn.host.ops_host import relu_backward_host as relu_backward_impl
from x_cnn.host.ops_host import add_host as add_impl
from x_cnn.host.ops_host import linear_forward_host as linear_forward_impl
from x_cnn.host.ops_host import linear_backward_host as linear_backward_impl
from x_cnn.host.ops_host import softmax_xent_host as softmax_xent_impl
from x_cnn.host.ops_host import sgd_host as sgd_impl
from x_cnn.host.ops_host import maxpool2d_forward_host as maxpool2d_forward_impl
from x_cnn.host.ops_host import maxpool2d_backward_host as maxpool2d_backward_impl
from x_cnn.host.ops_host import avgpool2d_forward_host as avgpool2d_forward_impl
from x_cnn.host.ops_host import avgpool2d_backward_host as avgpool2d_backward_impl


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

def _block_prms(conv_params_obj: PythonObject, pool_params_obj: PythonObject) raises -> Tuple[List[Int32], List[Int32], Bool]:
    """The conv block's two parameter blocks; an empty pool list is no pool."""
    var cprm = conv_params(_ints(conv_params_obj))
    var pool = Int(py=len(pool_params_obj)) > 0
    var pprm = pool_params(_ints(pool_params_obj)) if pool else List[Int32]()
    if pool:
        if (Int(pprm[PP_N]) != Int(cprm[CP_N]) or Int(pprm[PP_C]) != Int(cprm[CP_OC])
                or Int(pprm[PP_H]) != Int(cprm[CP_OH]) or Int(pprm[PP_W]) != Int(cprm[CP_OW])):
            raise Error("x_cnn conv block: the pool's input is not the conv's output")
    return (cprm^, pprm^, pool)


def conv_block_forward_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, idx_addr: PythonObject,
    conv_prm: PythonObject, pool_prm: PythonObject,
) raises -> PythonObject:
    """Conv2d -> ReLU -> MaxPool2d (CNNClassifier's block): the three host
    layer entries in sequence, the GPU binding's fused entry's twin."""
    var t = _block_prms(conv_prm, pool_prm)
    var cprm = t[0].copy()
    var pprm = t[1].copy()
    var N = Int(cprm[CP_N]); var C = Int(cprm[CP_C]); var OC = Int(cprm[CP_OC])
    var ckk = C * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var x = read_f32(Int(py=x_addr), N * C * Int(cprm[CP_H]) * Int(cprm[CP_W]))
    var w = read_f32(Int(py=w_addr), OC * ckk)
    var b = read_f32(Int(py=b_addr), OC)
    var po = f32_ptr(Int(py=out_addr))
    var pi = i32_ptr(Int(py=idx_addr))
    with GILReleased(Python()):
        var r = relu_forward_impl(conv2d_forward_impl(x, w, b, cprm))
        if t[2]:
            var idx = List[Int32]()
            var y = maxpool2d_forward_impl(r, pprm, idx)
            var no = _pool_counts(pprm)[1]
            copy_f32(y.unsafe_ptr(), po, no)
            for k in range(no):
                pi.unsafe_store(k, idx[k])
        else:
            copy_f32(r.unsafe_ptr(), po, len(r))
    return PythonObject(0)


def conv_block_backward_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, g_addr: PythonObject, idx_addr: PythonObject,
    outs: PythonObject, conv_prm: PythonObject, pool_prm: PythonObject,
) raises -> PythonObject:
    """The block's backward from its output gradient g: outs = [dx address
    (0: not wanted, the first block's), dW address, db address]."""
    var t = _block_prms(conv_prm, pool_prm)
    var dx_addr = outs[0]
    var dw_addr = outs[1]
    var db_addr = outs[2]
    var need_dx = Int(py=dx_addr) != 0
    var cprm = t[0].copy()
    var pprm = t[1].copy()
    var want = need_dx
    var N = Int(cprm[CP_N]); var C = Int(cprm[CP_C]); var OC = Int(cprm[CP_OC])
    var ckk = C * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var nx = N * C * Int(cprm[CP_H]) * Int(cprm[CP_W])
    var ny = N * OC * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var no = _pool_counts(pprm)[1] if t[2] else ny
    var x = read_f32(Int(py=x_addr), nx)
    var w = read_f32(Int(py=w_addr), OC * ckk)
    var b = read_f32(Int(py=b_addr), OC)
    var g = read_f32(Int(py=g_addr), no)
    var idx = read_i32(Int(py=idx_addr), no) if t[2] else List[Int32]()
    var pdx = f32_ptr(Int(py=dx_addr)) if want else f32_ptr(Int(py=db_addr))
    var pdw = f32_ptr(Int(py=dw_addr))
    var pdb = f32_ptr(Int(py=db_addr))
    with GILReleased(Python()):
        var yconv = conv2d_forward_impl(x, w, b, cprm)
        var gr = maxpool2d_backward_impl(g, idx, pprm) if t[2] else g.copy()
        var gy = relu_backward_impl(yconv, gr)
        var r = conv2d_backward_impl(x, w, gy, cprm)
        var base = r.unsafe_ptr()
        if want:
            copy_f32(base, pdx, nx)
        copy_f32(base + nx, pdw, OC * ckk)
        copy_f32(base + nx + OC * ckk, pdb, OC)
    return PythonObject(0)


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
    """In place: w and the momentum buffer v. hyper = [lr, momentum,
    weight_decay, dampening, nesterov (0/1), first step (0/1)]; the last
    three default to 0."""
    var n = _count(params)
    var h = List[Float32]()
    var nh = Int(py=len(hyper))
    if nh < 3 or nh > 6:
        raise Error("x_cnn sgd: hyper is [lr, momentum, weight_decay(, dampening, nesterov, first)]")
    for k in range(6):
        h.append(Float32(Float64(py=hyper[k])) if k < nh else Float32(0))
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
    return _csr_ints(csr_addr, Int(py=params[0]), Int(py=params[1]), Int(py=params[2]), Int(py=params[3]))


def _csr_ints(csr_addr: PythonObject, n: Int, F: Int, nnz: Int, mode: Int) raises -> Tuple[List[Int32], List[Int32]]:
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


def _pad_prm(params: PythonObject) raises -> List[Int32]:
    var out = List[Int32]()
    for k in range(9):
        var v = Int(py=params[k])
        if v < 0:
            raise Error("x_cnn pad: negative parameter")
        out.append(Int32(v))
    var N = Int(out[0]); var C = Int(out[1]); var H = Int(out[2]); var W = Int(out[3])
    if N <= 0 or C <= 0 or H <= 0 or W <= 0 or Int(out[8]) > 3:
        raise Error("x_cnn pad: positive N, C, H, W and a mode in 0..3 required")
    if Int(out[8]) == 1 and (Int(out[4]) >= H or Int(out[5]) >= H or Int(out[6]) >= W or Int(out[7]) >= W):
        raise Error("x_cnn pad: reflect padding must be smaller than the input dimension")
    if Int(out[8]) == 3 and (Int(out[4]) > H or Int(out[5]) > H or Int(out[6]) > W or Int(out[7]) > W):
        raise Error("x_cnn pad: circular padding can wrap around at most once")
    return out^


def pad2d_forward_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [N, C, H, W, top, bottom, left, right, mode (0 zeros, 1 reflect, 2 replicate, 3 circular)]."""
    var prm = _pad_prm(params)
    var nc = Int(prm[0]) * Int(prm[1])
    var nx = nc * Int(prm[2]) * Int(prm[3])
    var no = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    var x = read_f32(Int(py=x_addr), nx)
    var po = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var y = pad2d_forward_impl(x, prm)
        copy_f32(y.unsafe_ptr(), po, no)
    return PythonObject(no)


def pad2d_backward_binding(g_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pad_prm(params)
    var nc = Int(prm[0]) * Int(prm[1])
    var nx = nc * Int(prm[2]) * Int(prm[3])
    var ng = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    var g = read_f32(Int(py=g_addr), ng)
    var po = f32_ptr(Int(py=dx_addr))
    with GILReleased(Python()):
        var y = pad2d_backward_impl(g, prm)
        copy_f32(y.unsafe_ptr(), po, nx)
    return PythonObject(nx)


def adaptive_pool_binding(in_addr: PythonObject, out_addr: PythonObject, idx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [N, C, H, W, OH, OW, kind]: kind 0 avg forward (in x, out y),
    1 avg backward (in g, out dx), 2 max forward (in x, out y, idx written),
    3 max backward (in g, out dx, idx read)."""
    var prm = List[Int32]()
    for k in range(6):
        var v = Int(py=params[k])
        if v <= 0:
            raise Error("x_cnn adaptive pool: positive N, C, H, W, OH, OW required")
        prm.append(Int32(v))
    var kind = Int(py=params[6])
    if kind < 0 or kind > 3:
        raise Error("x_cnn adaptive pool: kind in 0..3")
    var nc = Int(prm[0]) * Int(prm[1])
    var nin = nc * Int(prm[2]) * Int(prm[3])
    var nout = nc * Int(prm[4]) * Int(prm[5])
    var n_read = nin if kind == 0 or kind == 2 else nout
    var n_write = nout if kind == 0 or kind == 2 else nin
    var src = read_f32(Int(py=in_addr), n_read)
    var idx = read_i32(Int(py=idx_addr), nout) if kind == 3 else List[Int32](length=nout, fill=Int32(0))
    var po = f32_ptr(Int(py=out_addr))
    var pi = i32_ptr(Int(py=idx_addr))
    with GILReleased(Python()):
        var idx_out = List[Int32]()
        var y = adaptive_pool_impl(src, idx, prm, kind, idx_out)
        copy_f32(y.unsafe_ptr(), po, n_write)
        if kind == 2:
            for k in range(nout):
                pi.unsafe_store(k, idx_out[k])
    return PythonObject(n_write)


def graph_op_binding(
    a_addr: PythonObject, b_addr: PythonObject, aux_addr: PythonObject, out_addr: PythonObject, csr_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """params = [n, F, nnz, kind]: kind 0 SAGE max forward (a = h; aux 2nF
    written), 1 its backward (a = h, b = g, aux read; the TRANSPOSED csr),
    2 row L2 normalize forward (a = x; aux n written), 3 its backward
    (a = y, b = g, aux read). A kind with no graph passes nnz 0 and a
    rowptr of zeros."""
    var kind = Int(py=params[3])
    if kind < 0 or kind > 3:
        raise Error("x_cnn graph op: kind in 0..3")
    var n = Int(py=params[0])
    var F = Int(py=params[1])
    var nnz = Int(py=params[2])
    var t = _csr_ints(csr_addr, n, F, nnz, 0)
    var naux = 2 * n * F if kind < 2 else n
    var a = read_f32(Int(py=a_addr), n * F)
    var b = read_f32(Int(py=b_addr), n * F) if kind == 1 or kind == 3 else a.copy()
    var aux = read_f32(Int(py=aux_addr), naux)
    var po = f32_ptr(Int(py=out_addr))
    var pa = f32_ptr(Int(py=aux_addr))
    with GILReleased(Python()):
        var r = graph_op_impl(a, b, aux, t[0], t[1], kind)
        copy_f32(r.unsafe_ptr(), po, n * F)
        copy_f32(r.unsafe_ptr() + n * F, pa, naux)
    _ = nnz
    return PythonObject(n * F)


def adam_binding(w_addr: PythonObject, g_addr: PythonObject, mv_addr: PythonObject, params: PythonObject, hyper: PythonObject) raises -> PythonObject:
    """In place: w and mv = [m (n) | v (n)]. hyper = [step_size, 1 - beta1,
    beta2, 1 - beta2, eps, sqrt(bias_correction2), weight_decay, adamw (0/1),
    1 - lr * weight_decay] (x_cnn/ops.mojo adam_at)."""
    var n = _count(params)
    if Int(py=len(hyper)) != 9:
        raise Error("x_cnn adam: hyper has 9 entries")
    var h = List[Float32]()
    for k in range(9):
        h.append(Float32(Float64(py=hyper[k])))
    var w = read_f32(Int(py=w_addr), n)
    var g = read_f32(Int(py=g_addr), n)
    var mv = read_f32(Int(py=mv_addr), 2 * n)
    var pw = f32_ptr(Int(py=w_addr))
    var pm = f32_ptr(Int(py=mv_addr))
    with GILReleased(Python()):
        var r = adam_impl(w, g, mv, h)
        copy_f32(r.unsafe_ptr(), pw, n)
        copy_f32(r.unsafe_ptr() + n, pm, 2 * n)
    return PythonObject(n)


def x_cnn_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_cnn_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_cnn_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_cnn host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_cnn_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_CNN_HOST_SABOTAGE)


# DEVIATION 5718: resident arrays. On the CPU a resident array is a host
# allocation and its handle is its address, so the `_r` entries are the
# ordinary entries (the GPU binding's `_r` entries read device addresses).


def res_alloc_binding(n: PythonObject) raises -> PythonObject:
    var nn = Int(py=n)
    var p = alloc[Float32](nn if nn > 0 else 1)
    memset_zero(p, nn if nn > 0 else 1)
    return PythonObject(Int(p))


def res_free_binding(h: PythonObject) raises -> PythonObject:
    f32_ptr(Int(py=h)).free()
    return PythonObject(0)


def res_upload_binding(h: PythonObject, src_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """n 4-byte words (float32 or int32) from a host array into the resident array."""
    var nn = Int(py=n)
    if nn > 0:
        memcpy(dest=f32_ptr(Int(py=h)), src=f32_ptr(Int(py=src_addr)), count=nn)
    return PythonObject(nn)


def res_download_binding(h: PythonObject, dst_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    var nn = Int(py=n)
    if nn > 0:
        memcpy(dest=f32_ptr(Int(py=dst_addr)), src=f32_ptr(Int(py=h)), count=nn)
    return PythonObject(nn)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


@export
def PyInit__mojolearn_x_cnn_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_cnn_host")
        m.def_function[x_cnn_host_numeric_mode_binding]("x_cnn_host_numeric_mode")
        m.def_function[x_cnn_host_vendor_binding]("x_cnn_host_vendor")
        m.def_function[x_cnn_host_column_binding]("x_cnn_host_column")
        m.def_function[x_cnn_host_sabotage_binding]("x_cnn_host_sabotage")
        m.def_function[gemm_binding]("x_cnn_gemm")
        m.def_function[conv2d_forward_binding]("x_cnn_conv2d_forward")
        m.def_function[conv2d_backward_binding]("x_cnn_conv2d_backward")
        m.def_function[conv_shape_binding]("x_cnn_conv_shape")
        m.def_function[conv_block_forward_binding]("x_cnn_conv_block_forward")
        m.def_function[conv_block_backward_binding]("x_cnn_conv_block_backward")
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
        m.def_function[adam_binding]("x_cnn_adam")
        m.def_function[batchnorm_forward_binding]("x_cnn_batchnorm_forward")
        m.def_function[batchnorm_backward_binding]("x_cnn_batchnorm_backward")
        m.def_function[dropout2d_binding]("x_cnn_dropout2d")
        m.def_function[mul_binding]("x_cnn_mul")
        m.def_function[spmm_binding]("x_cnn_spmm")
        m.def_function[gcn_norm_binding]("x_cnn_gcn_norm")
        m.def_function[pad2d_forward_binding]("x_cnn_pad2d_forward")
        m.def_function[pad2d_backward_binding]("x_cnn_pad2d_backward")
        m.def_function[adaptive_pool_binding]("x_cnn_adaptive_pool")
        m.def_function[graph_op_binding]("x_cnn_graph_op")
        m.def_function[numeric_mode_binding]("x_cnn_numeric_mode")
        m.def_function[vendor_binding]("x_cnn_vendor")
        m.def_function[res_alloc_binding]("x_cnn_res_alloc")
        m.def_function[res_free_binding]("x_cnn_res_free")
        m.def_function[res_upload_binding]("x_cnn_res_upload")
        m.def_function[res_download_binding]("x_cnn_res_download")
        m.def_function[conv_block_forward_binding]("x_cnn_conv_block_forward_r")
        m.def_function[conv_block_backward_binding]("x_cnn_conv_block_backward_r")
        m.def_function[linear_forward_binding]("x_cnn_linear_forward_r")
        m.def_function[linear_backward_binding]("x_cnn_linear_backward_r")
        m.def_function[softmax_xent_binding]("x_cnn_softmax_xent_r")
        m.def_function[sgd_binding]("x_cnn_sgd_r")
        m.def_function[adam_binding]("x_cnn_adam_r")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cnn_host: ", e))
