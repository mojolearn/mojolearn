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
from x_cnn.ops import FP, IP
from x_cnn.device import (
    gemm_into, conv2d_forward_into, conv2d_backward_into, maxpool2d_forward_into, maxpool2d_backward_into,
    avgpool2d_forward_into, avgpool2d_backward_into, relu_forward_into, relu_backward_into, add_into, mul_into,
    linear_forward_into, linear_backward_into, softmax_xent_into, sgd_into, adam_into,
    batchnorm_forward_into, batchnorm_backward_into, dropout2d_into, spmm_into, pad2d_forward_into,
    pad2d_backward_into, conv_block_forward_into, conv_block_backward_into,
    res_alloc, res_free, res_upload, res_download, res_gather,
)
from x_cnn.device import graph_op_device as graph_op_impl
from x_cnn.device import adaptive_pool_device as adaptive_pool_impl
from x_cnn.device import gcn_norm_device as gcn_norm_impl


def _fp(addr: PythonObject) raises -> FP:
    """The caller's float32 array, by address (DEVIATION 5716: the device
    entries copy straight to and from it)."""
    return f32_ptr(Int(py=addr)).unsafe_origin_cast[MutAnyOrigin]()


def _ip(addr: PythonObject) raises -> IP:
    return i32_ptr(Int(py=addr)).unsafe_origin_cast[MutAnyOrigin]()


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
    var a = _fp(a_addr)
    var b = _fp(b_addr)
    var c = _fp(c_addr)
    with GILReleased(Python()):
        gemm_into(a, b, c, m, n, k, op)
    return PythonObject(m * n)


def conv2d_forward_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    var prm = conv_params(_ints(params))
    var total = Int(prm[CP_N]) * Int(prm[CP_OC]) * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var x = _fp(x_addr)
    var w = _fp(w_addr)
    var b = _fp(b_addr)
    var out = _fp(out_addr)
    with GILReleased(Python()):
        conv2d_forward_into(x, w, b, prm, out)
    return PythonObject(total)


def conv2d_backward_binding(
    x_addr: PythonObject, w_addr: PythonObject, dout_addr: PythonObject, dx_addr: PythonObject,
    dw_addr: PythonObject, db_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    var prm = conv_params(_ints(params))
    var nx = Int(prm[CP_N]) * Int(prm[CP_C]) * Int(prm[CP_H]) * Int(prm[CP_W])
    var x = _fp(x_addr)
    var w = _fp(w_addr)
    var dout = _fp(dout_addr)
    var pdx = _fp(dx_addr)
    var pdw = _fp(dw_addr)
    var pdb = _fp(db_addr)
    with GILReleased(Python()):
        conv2d_backward_into(x, w, dout, prm, pdx, pdw, pdb)
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


def conv_block_forward_binding[resident: Bool = False](
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, idx_addr: PythonObject,
    conv_prm: PythonObject, pool_prm: PythonObject,
) raises -> PythonObject:
    """Conv2d -> ReLU -> MaxPool2d (CNNClassifier's block) in one call; out
    and idx are the pool's (idx unused without a pool)."""
    var t = _block_prms(conv_prm, pool_prm)
    var x = _fp(x_addr)
    var w = _fp(w_addr)
    var b = _fp(b_addr)
    var po = _fp(out_addr)
    var pi = _ip(idx_addr)
    with GILReleased(Python()):
        conv_block_forward_into[resident](x, w, b, t[0], t[1], t[2], po, pi)
    return PythonObject(0)


def _saved(saved: PythonObject) raises -> Tuple[Int, Int]:
    """[cols address, conv output address] of a resident block, or [] for none."""
    if Int(py=len(saved)) == 0:
        return (0, 0)
    if Int(py=len(saved)) != 2:
        raise Error("x_cnn conv block: saved is [] or [cols, conv output]")
    return (Int(py=saved[0]), Int(py=saved[1]))


def conv_block_forward_r_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, idx_addr: PythonObject,
    conv_prm: PythonObject, pool_prm: PythonObject, saved: PythonObject,
) raises -> PythonObject:
    """The resident block forward; `saved` = [cols, conv output] resident
    arrays the backward reads (DEVIATION 5718), or []."""
    var t = _block_prms(conv_prm, pool_prm)
    var sv = _saved(saved)
    var x = _fp(x_addr)
    var w = _fp(w_addr)
    var b = _fp(b_addr)
    var po = _fp(out_addr)
    var pi = _ip(idx_addr)
    with GILReleased(Python()):
        conv_block_forward_into[True](x, w, b, t[0], t[1], t[2], po, pi, sv[0], sv[1])
    return PythonObject(0)


def conv_block_backward_binding[resident: Bool = False](
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
    var want = need_dx
    var x = _fp(x_addr)
    var w = _fp(w_addr)
    var b = _fp(b_addr)
    var g = _fp(g_addr)
    var pi = _ip(idx_addr)
    var pdx = _fp(dx_addr) if want else _fp(db_addr)
    var pdw = _fp(dw_addr)
    var pdb = _fp(db_addr)
    with GILReleased(Python()):
        conv_block_backward_into[resident](x, w, b, g, pi, t[0], t[1], t[2], want, pdx, pdw, pdb)
    return PythonObject(0)


def conv_block_backward_r_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, g_addr: PythonObject, idx_addr: PythonObject,
    outs: PythonObject, conv_prm: PythonObject, pool_prm: PythonObject,
) raises -> PythonObject:
    """The resident block backward: outs = [dx (0: none), dW, db] or [dx, dW,
    db, cols, conv output] with the forward's saved arrays."""
    var t = _block_prms(conv_prm, pool_prm)
    var sv = (Int(py=outs[3]), Int(py=outs[4])) if Int(py=len(outs)) == 5 else (0, 0)
    var dx_addr = outs[0]
    var want = Int(py=dx_addr) != 0
    var x = _fp(x_addr)
    var w = _fp(w_addr)
    var b = _fp(b_addr)
    var g = _fp(g_addr)
    var pi = _ip(idx_addr)
    var pdx = _fp(dx_addr) if want else _fp(outs[2])
    var pdw = _fp(outs[1])
    var pdb = _fp(outs[2])
    with GILReleased(Python()):
        conv_block_backward_into[True](x, w, b, g, pi, t[0], t[1], t[2], want, pdx, pdw, pdb, sv[0], sv[1])
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
    var x = _fp(x_addr)
    var po = _fp(out_addr)
    var pi = _ip(idx_addr)
    with GILReleased(Python()):
        maxpool2d_forward_into(x, prm, po, pi)
    return PythonObject(c[1])


def maxpool2d_backward_binding(dout_addr: PythonObject, idx_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    var c = _pool_counts(prm)
    var dout = _fp(dout_addr)
    var idx = _ip(idx_addr)
    var pd = _fp(dx_addr)
    with GILReleased(Python()):
        maxpool2d_backward_into(dout, idx, prm, pd)
    return PythonObject(c[0])


def avgpool2d_forward_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    var c = _pool_counts(prm)
    var x = _fp(x_addr)
    var po = _fp(out_addr)
    with GILReleased(Python()):
        avgpool2d_forward_into(x, prm, po)
    return PythonObject(c[1])


def avgpool2d_backward_binding(dout_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    var c = _pool_counts(prm)
    var dout = _fp(dout_addr)
    var pd = _fp(dx_addr)
    with GILReleased(Python()):
        avgpool2d_backward_into(dout, prm, pd)
    return PythonObject(c[0])


def _count(params: PythonObject) raises -> Int:
    var n = Int(py=params[0])
    if n <= 0:
        raise Error("x_cnn: a positive element count is required")
    return n


def relu_forward_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var x = _fp(x_addr)
    var po = _fp(out_addr)
    with GILReleased(Python()):
        relu_forward_into(x, n, po)
    return PythonObject(n)


def relu_backward_binding(x_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var x = _fp(x_addr)
    var g = _fp(g_addr)
    var po = _fp(dx_addr)
    with GILReleased(Python()):
        relu_backward_into(x, g, n, po)
    return PythonObject(n)


def add_binding(a_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var a = _fp(a_addr)
    var b = _fp(b_addr)
    var po = _fp(out_addr)
    with GILReleased(Python()):
        add_into(a, b, n, po)
    return PythonObject(n)


def _lin(params: PythonObject) raises -> Tuple[Int, Int, Int]:
    var n = Int(py=params[0])
    var d_in = Int(py=params[1])
    var d_out = Int(py=params[2])
    if n <= 0 or d_in <= 0 or d_out <= 0:
        raise Error("x_cnn linear: positive rows, in and out features required")
    return (n, d_in, d_out)


def linear_forward_binding[resident: Bool = False](x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var s = _lin(params)
    var x = _fp(x_addr)
    var w = _fp(w_addr)
    var b = _fp(b_addr)
    var po = _fp(out_addr)
    with GILReleased(Python()):
        linear_forward_into[resident](x, w, b, s[0], s[1], s[2], po)
    return PythonObject(s[0] * s[2])


def linear_backward_binding[resident: Bool = False](
    x_addr: PythonObject, w_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject,
    dw_addr: PythonObject, db_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    var s = _lin(params)
    var x = _fp(x_addr)
    var w = _fp(w_addr)
    var g = _fp(g_addr)
    var pdx = _fp(dx_addr)
    var pdw = _fp(dw_addr)
    var pdb = _fp(db_addr)
    with GILReleased(Python()):
        linear_backward_into[resident](x, w, g, s[0], s[1], s[2], pdx, pdw, pdb)
    return PythonObject(s[0])


def softmax_xent_binding[resident: Bool = False](
    logits_addr: PythonObject, labels_addr: PythonObject, grad_addr: PythonObject, proba_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The mean cross entropy (a float; 0 when every label is < 0)."""
    var n = Int(py=params[0])
    var k = Int(py=params[1])
    if n <= 0 or k <= 0:
        raise Error("x_cnn softmax: positive rows and classes required")
    var py_labels = _ip(labels_addr)
    comptime if not resident:  # a resident label array is on the device; its caller checked it
        for i in range(n):
            if Int(py_labels[i]) >= k:
                raise Error("x_cnn softmax: a label is not a class index")
    var logits = _fp(logits_addr)
    var pg = _fp(grad_addr)
    var pp = _fp(proba_addr)
    var loss = Float32(0)
    with GILReleased(Python()):
        loss = softmax_xent_into[resident](logits, py_labels, n, k, pg, pp)
    return PythonObject(Float64(loss))


def sgd_binding[resident: Bool = False](w_addr: PythonObject, g_addr: PythonObject, v_addr: PythonObject, params: PythonObject, hyper: PythonObject) raises -> PythonObject:
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
    var pw = _fp(w_addr)
    var g = _fp(g_addr)
    var pv = _fp(v_addr)
    with GILReleased(Python()):
        sgd_into[resident](pw, g, pv, h, n)
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
    var x = _fp(x_addr)
    var pyo = _fp(y_addr)
    var pr = _fp(running_addr)
    var pa = _fp(aux_addr)
    with GILReleased(Python()):
        batchnorm_forward_into(x, pr, pa, prm, training, pyo)
    return PythonObject(total)


def batchnorm_backward_binding(
    x_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject, aux_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [N, C, HW, training]; aux updates in place (sum_g, sum_gx)."""
    var prm = _bn_prm(params)
    var training = Int(py=params[3]) != 0
    var total = Int(prm[0]) * Int(prm[1]) * Int(prm[2])
    var x = _fp(x_addr)
    var g = _fp(g_addr)
    var pd = _fp(dx_addr)
    var pa = _fp(aux_addr)
    with GILReleased(Python()):
        batchnorm_backward_into(x, g, pa, prm, training, pd)
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
    var x = _fp(x_addr)
    var pyo = _fp(y_addr)
    var pm = _fp(mask_addr)
    with GILReleased(Python()):
        dropout2d_into(x, total, prm, h, pyo, pm)
    return PythonObject(total)


def mul_binding(a_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var a = _fp(a_addr)
    var b = _fp(b_addr)
    var po = _fp(out_addr)
    with GILReleased(Python()):
        mul_into(a, b, n, po)
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
    var vals = _fp(vals_addr) if nnz > 0 else _fp(h_addr)
    var h = _fp(h_addr)
    var po = _fp(out_addr)
    with GILReleased(Python()):
        spmm_into(vals, nnz, h, t[0], t[1], po)
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
    var no = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    var x = _fp(x_addr)
    var po = _fp(out_addr)
    with GILReleased(Python()):
        pad2d_forward_into(x, prm, po)
    return PythonObject(no)


def pad2d_backward_binding(g_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pad_prm(params)
    var nx = Int(prm[0]) * Int(prm[1]) * Int(prm[2]) * Int(prm[3])
    var g = _fp(g_addr)
    var po = _fp(dx_addr)
    with GILReleased(Python()):
        pad2d_backward_into(g, prm, po)
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


def adam_binding[resident: Bool = False](w_addr: PythonObject, g_addr: PythonObject, mv_addr: PythonObject, params: PythonObject, hyper: PythonObject) raises -> PythonObject:
    """In place: w and mv = [m (n) | v (n)]. hyper = [step_size, 1 - beta1,
    beta2, 1 - beta2, eps, sqrt(bias_correction2), weight_decay, adamw (0/1),
    1 - lr * weight_decay] (x_cnn/ops.mojo adam_at)."""
    var n = _count(params)
    if Int(py=len(hyper)) != 9:
        raise Error("x_cnn adam: hyper has 9 entries")
    var h = List[Float32]()
    for k in range(9):
        h.append(Float32(Float64(py=hyper[k])))
    var pw = _fp(w_addr)
    var g = _fp(g_addr)
    var pm = _fp(mv_addr)
    with GILReleased(Python()):
        adam_into[resident](pw, g, pm, h, n)
    return PythonObject(n)


# DEVIATION 5718: resident arrays. A handle is a device address the caller
# keeps between the `_r` entries (the same entries, reading and writing those
# addresses instead of uploading and downloading host arrays).


def res_alloc_binding(n: PythonObject) raises -> PythonObject:
    return PythonObject(res_alloc(Int(py=n)))


def res_free_binding(h: PythonObject) raises -> PythonObject:
    res_free(Int(py=h))
    return PythonObject(0)


def res_upload_binding(h: PythonObject, src_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """n 4-byte words (float32 or int32) from a host array into the resident array."""
    var src = _fp(src_addr)
    var nn = Int(py=n)
    var hh = Int(py=h)
    with GILReleased(Python()):
        res_upload(hh, src, nn)
    return PythonObject(nn)


def res_gather_binding(dst: PythonObject, src: PythonObject, rows_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """Resident dst rows = resident src rows[r] (int32 host indices); params = [n, row words]."""
    var n = Int(py=params[0])
    var row = Int(py=params[1])
    var d = Int(py=dst)
    var sr = Int(py=src)
    var rp = _ip(rows_addr)
    with GILReleased(Python()):
        res_gather(d, sr, rp, n, row)
    return PythonObject(n)


def res_download_binding(h: PythonObject, dst_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    var dst = _fp(dst_addr)
    var nn = Int(py=n)
    var hh = Int(py=h)
    with GILReleased(Python()):
        res_download(hh, dst, nn)
    return PythonObject(nn)


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
        m.def_function[conv_block_forward_binding[False]]("x_cnn_conv_block_forward")
        m.def_function[conv_block_backward_binding[False]]("x_cnn_conv_block_backward")
        m.def_function[pool_shape_binding]("x_cnn_pool_shape")
        m.def_function[maxpool2d_forward_binding]("x_cnn_maxpool2d_forward")
        m.def_function[maxpool2d_backward_binding]("x_cnn_maxpool2d_backward")
        m.def_function[avgpool2d_forward_binding]("x_cnn_avgpool2d_forward")
        m.def_function[avgpool2d_backward_binding]("x_cnn_avgpool2d_backward")
        m.def_function[relu_forward_binding]("x_cnn_relu_forward")
        m.def_function[relu_backward_binding]("x_cnn_relu_backward")
        m.def_function[add_binding]("x_cnn_add")
        m.def_function[linear_forward_binding[False]]("x_cnn_linear_forward")
        m.def_function[linear_backward_binding[False]]("x_cnn_linear_backward")
        m.def_function[softmax_xent_binding[False]]("x_cnn_softmax_xent")
        m.def_function[sgd_binding[False]]("x_cnn_sgd")
        m.def_function[adam_binding[False]]("x_cnn_adam")
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
        m.def_function[conv_block_forward_r_binding]("x_cnn_conv_block_forward_r")
        m.def_function[conv_block_backward_r_binding]("x_cnn_conv_block_backward_r")
        m.def_function[res_gather_binding]("x_cnn_res_gather")
        m.def_function[linear_forward_binding[True]]("x_cnn_linear_forward_r")
        m.def_function[linear_backward_binding[True]]("x_cnn_linear_backward_r")
        m.def_function[softmax_xent_binding[True]]("x_cnn_softmax_xent_r")
        m.def_function[sgd_binding[True]]("x_cnn_sgd_r")
        m.def_function[adam_binding[True]]("x_cnn_adam_r")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cnn: ", e))
