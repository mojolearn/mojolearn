# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_cnn` (the CNN expansion lane). HOST ONLY;
the GPU binding's export names and address contract, the work is
x_cnn/host/ops_host.mojo."""
from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32, copy_f32
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.memory import alloc, memcpy, memset_zero
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cnn.ops import CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, conv_params
from x_cnn.ops import PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW, pool_params
from x_cnn.host.ops_host import X_CNN_HOST_SABOTAGE
from x_cnn.ops import FP, IP
from x_cnn.host.gemm_host import gemm_host_into
from x_cnn.host.ops_host import conv2d_forward_into, conv2d_backward_into
from x_cnn.host.ops_host import conv_block_forward_into, conv_block_backward_into
from x_cnn.host.ops_host import linear_forward_into, linear_backward_into
from x_cnn.host.ops_host import maxpool2d_forward_into, maxpool2d_backward_into
from x_cnn.host.ops_host import avgpool2d_forward_into, avgpool2d_backward_into
from x_cnn.host.ops_host import map2_into, softmax_xent_into, sgd_into, adam_into
from x_cnn.host.ops_host import batchnorm_forward_into, batchnorm_backward_into, dropout2d_into
from x_cnn.ops import relu_fwd_at, relu_bwd_at, add_at, mul_at
from x_cnn.host.ops_host import graph_op_host as graph_op_impl
from x_cnn.host.ops_host import adaptive_pool_host as adaptive_pool_impl
from x_cnn.host.ops_host import pad2d_forward_host as pad2d_forward_impl
from x_cnn.host.ops_host import pad2d_backward_host as pad2d_backward_impl
from x_cnn.host.ops_host import spmm_host as spmm_impl
from x_cnn.host.ops_host import gcn_norm_host as gcn_norm_impl


def fp(addr: PythonObject) raises -> FP:
    """The caller's float32 array at `addr`, read and written in place
    (DEVIATION 5719: the address-in/address-out entries)."""
    var a = Int(py=addr)
    if a == 0:
        raise Error("x_cnn: null float32 buffer address")
    return FP(unsafe_from_address=a)


def ip(addr: PythonObject) raises -> IP:
    var a = Int(py=addr)
    if a == 0:
        raise Error("x_cnn: null int32 buffer address")
    return IP(unsafe_from_address=a)


@always_inline
def out_f32(p: FP, n: Int):
    """THE CPU COLUMN'S OUTPUT SEAM for the address-out entries: every
    float32 word such an entry returns is in p[0:n] when this runs. In
    production it does nothing. The end-to-end sabotage arm
    (x_cnn/checks/sabotage/e2e_host_output_bit.patch) flips every word's low
    bit here, and in `copy_f32` for the entries that still copy a result."""
    _ = p
    _ = n


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
    var pa = fp(a_addr)
    var pb = fp(b_addr)
    var pc = fp(c_addr)
    with GILReleased(Python()):
        gemm_host_into(pa, pb, pc, op, m, n, k)
        out_f32(pc, m * n)
    return PythonObject(m * n)


def conv2d_forward_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    var prm = conv_params(_ints(params))
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var total = N * OC * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var px = fp(x_addr)
    var pw = fp(w_addr)
    var pb = fp(b_addr)
    var po = fp(out_addr)
    _ = C
    _ = ckk
    with GILReleased(Python()):
        conv2d_forward_into(px, pw, pb, po, prm)
        out_f32(po, total)
    return PythonObject(total)


def conv2d_backward_binding(
    x_addr: PythonObject, w_addr: PythonObject, dout_addr: PythonObject, dx_addr: PythonObject,
    dw_addr: PythonObject, db_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    var prm = conv_params(_ints(params))
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var nx = N * C * Int(prm[CP_H]) * Int(prm[CP_W])
    var px = fp(x_addr)
    var pw = fp(w_addr)
    var pd = fp(dout_addr)
    var pdx = fp(dx_addr)
    var pdw = fp(dw_addr)
    var pdb = fp(db_addr)
    with GILReleased(Python()):
        conv2d_backward_into(px, pw, pd, pdx, pdw, pdb, prm, True)
        out_f32(pdx, nx)
        out_f32(pdw, OC * ckk)
        out_f32(pdb, OC)
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
    var pool = t[2]
    var ny = Int(cprm[CP_N]) * Int(cprm[CP_OC]) * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var no = _pool_counts(pprm)[1] if pool else ny
    var px = fp(x_addr)
    var pw = fp(w_addr)
    var pb = fp(b_addr)
    var po = fp(out_addr)
    var pi = ip(idx_addr)
    with GILReleased(Python()):
        conv_block_forward_into(px, pw, pb, po, pi, cprm, pprm, pool)
        out_f32(po, no)
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
    var pool = t[2]
    var N = Int(cprm[CP_N]); var C = Int(cprm[CP_C]); var OC = Int(cprm[CP_OC])
    var ckk = C * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var nx = N * C * Int(cprm[CP_H]) * Int(cprm[CP_W])
    var px = fp(x_addr)
    var pw = fp(w_addr)
    var pb = fp(b_addr)
    var pg = fp(g_addr)
    var noidx = List[Int32](length=1, fill=Int32(0))
    var pi = ip(idx_addr) if pool else noidx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pdw = fp(dw_addr)
    var pdb = fp(db_addr)
    var pdx = fp(dx_addr) if need_dx else pdb
    with GILReleased(Python()):
        conv_block_backward_into(px, pw, pb, pg, pi, pdx, pdw, pdb, cprm, pprm, pool, need_dx)
        if need_dx:
            out_f32(pdx, nx)
        out_f32(pdw, OC * ckk)
        out_f32(pdb, OC)
    _ = noidx^
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
    var px = fp(x_addr)
    var po = fp(out_addr)
    var pi = ip(idx_addr)
    with GILReleased(Python()):
        maxpool2d_forward_into(px, po, pi, prm)
        out_f32(po, c[1])
    return PythonObject(c[1])


def maxpool2d_backward_binding(dout_addr: PythonObject, idx_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    var c = _pool_counts(prm)
    var pg = fp(dout_addr)
    var pi = ip(idx_addr)
    var pd = fp(dx_addr)
    with GILReleased(Python()):
        maxpool2d_backward_into(pg, pi, pd, prm)
        out_f32(pd, c[0])
    return PythonObject(c[0])


def avgpool2d_forward_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    var c = _pool_counts(prm)
    var px = fp(x_addr)
    var po = fp(out_addr)
    with GILReleased(Python()):
        avgpool2d_forward_into(px, po, prm)
        out_f32(po, c[1])
    return PythonObject(c[1])


def avgpool2d_backward_binding(dout_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    var c = _pool_counts(prm)
    var pg = fp(dout_addr)
    var pd = fp(dx_addr)
    with GILReleased(Python()):
        avgpool2d_backward_into(pg, pd, prm)
        out_f32(pd, c[0])
    return PythonObject(c[0])


def _count(params: PythonObject) raises -> Int:
    var n = Int(py=params[0])
    if n <= 0:
        raise Error("x_cnn: a positive element count is required")
    return n


def relu_forward_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var px = fp(x_addr)
    var po = fp(out_addr)
    with GILReleased(Python()):
        map2_into[relu_fwd_at](px, px, po, n)
        out_f32(po, n)
    return PythonObject(n)


def relu_backward_binding(x_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var px = fp(x_addr)
    var pg = fp(g_addr)
    var po = fp(dx_addr)
    with GILReleased(Python()):
        map2_into[relu_bwd_at](px, pg, po, n)
        out_f32(po, n)
    return PythonObject(n)


def add_binding(a_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var pa = fp(a_addr)
    var pb = fp(b_addr)
    var po = fp(out_addr)
    with GILReleased(Python()):
        map2_into[add_at](pa, pb, po, n)
        out_f32(po, n)
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
    var px = fp(x_addr)
    var pw = fp(w_addr)
    var pb = fp(b_addr)
    var po = fp(out_addr)
    with GILReleased(Python()):
        linear_forward_into(px, pw, pb, po, s[0], s[1], s[2])
        out_f32(po, s[0] * s[2])
    return PythonObject(s[0] * s[2])


def linear_backward_binding(
    x_addr: PythonObject, w_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject,
    dw_addr: PythonObject, db_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    var s = _lin(params)
    var n = s[0]
    var d_in = s[1]
    var d_out = s[2]
    var px = fp(x_addr)
    var pw = fp(w_addr)
    var pg = fp(g_addr)
    var pdx = fp(dx_addr)
    var pdw = fp(dw_addr)
    var pdb = fp(db_addr)
    with GILReleased(Python()):
        linear_backward_into(px, pw, pg, pdx, pdw, pdb, n, d_in, d_out)
        out_f32(pdx, n * d_in)
        out_f32(pdw, d_out * d_in)
        out_f32(pdb, d_out)
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
    var pl = fp(logits_addr)
    var py_ = ip(labels_addr)
    for i in range(n):
        if Int(py_[i]) >= k:
            raise Error("x_cnn softmax: a label is not a class index")
    var pg = fp(grad_addr)
    var pp = fp(proba_addr)
    var loss = Float32(0)
    with GILReleased(Python()):
        loss = softmax_xent_into(pl, py_, pg, pp, n, k)
        out_f32(pg, n * k)
        out_f32(pp, n * k)
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
    var pw = fp(w_addr)
    var pg = fp(g_addr)
    var pv = fp(v_addr)
    with GILReleased(Python()):
        sgd_into(pw, pg, pv, h, n)
        out_f32(pw, n)
        out_f32(pv, n)
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
    var px = fp(x_addr)
    var pyo = fp(y_addr)
    var pr = fp(running_addr)
    var pa = fp(aux_addr)
    with GILReleased(Python()):
        batchnorm_forward_into(px, pr, pa, pyo, prm, training)
        out_f32(pyo, total)
        out_f32(pr, 2 * C)
        out_f32(pa, 2 + 7 * C)
    return PythonObject(total)


def batchnorm_backward_binding(
    x_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject, aux_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [N, C, HW, training]; aux updates in place (sum_g, sum_gx)."""
    var prm = _bn_prm(params)
    var training = Int(py=params[3]) != 0
    var C = Int(prm[1])
    var total = Int(prm[0]) * C * Int(prm[2])
    var px = fp(x_addr)
    var pg = fp(g_addr)
    var pd = fp(dx_addr)
    var pa = fp(aux_addr)
    with GILReleased(Python()):
        batchnorm_backward_into(px, pg, pa, pd, prm, training)
        out_f32(pd, total)
        out_f32(pa, 2 + 7 * C)
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
    var px = fp(x_addr)
    var pyo = fp(y_addr)
    var pm = fp(mask_addr)
    with GILReleased(Python()):
        dropout2d_into(px, pyo, pm, prm, h, total)
        out_f32(pyo, total)
        out_f32(pm, total)
    return PythonObject(total)


def mul_binding(a_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = _count(params)
    var pa = fp(a_addr)
    var pb = fp(b_addr)
    var po = fp(out_addr)
    with GILReleased(Python()):
        map2_into[mul_at](pa, pb, po, n)
        out_f32(po, n)
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
    var pw = fp(w_addr)
    var pg = fp(g_addr)
    var pm = fp(mv_addr)
    with GILReleased(Python()):
        adam_into(pw, pg, pm, h, n)
        out_f32(pw, n)
        out_f32(pm, 2 * n)
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


def conv_block_forward_r_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, idx_addr: PythonObject,
    conv_prm: PythonObject, pool_prm: PythonObject, saved: PythonObject,
) raises -> PythonObject:
    """The GPU binding's resident block forward: `saved` only spares the
    device a recomputation, so the CPU twin runs its ordinary entry."""
    return conv_block_forward_binding(x_addr, w_addr, b_addr, out_addr, idx_addr, conv_prm, pool_prm)


def conv_block_backward_r_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, g_addr: PythonObject, idx_addr: PythonObject,
    outs: PythonObject, conv_prm: PythonObject, pool_prm: PythonObject,
) raises -> PythonObject:
    """outs may carry the GPU binding's two saved arrays after [dx, dW, db]; the CPU twin recomputes."""
    return conv_block_backward_binding(x_addr, w_addr, b_addr, g_addr, idx_addr, outs, conv_prm, pool_prm)


def res_gather_binding(dst: PythonObject, src: PythonObject, rows_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """dst rows = src rows[r] (int32 host indices), `row` 4-byte words each."""
    var n = Int(py=params[0])
    var row = Int(py=params[1])
    var d = f32_ptr(Int(py=dst))
    var sr = f32_ptr(Int(py=src))
    var rp = i32_ptr(Int(py=rows_addr))
    for r in range(n):
        memcpy(dest=d + r * row, src=sr + Int(rp[r]) * row, count=row)
    return PythonObject(n)


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
        m.def_function[conv_block_forward_r_binding]("x_cnn_conv_block_forward_r")
        m.def_function[conv_block_backward_r_binding]("x_cnn_conv_block_backward_r")
        m.def_function[res_gather_binding]("x_cnn_res_gather")
        m.def_function[linear_forward_binding]("x_cnn_linear_forward_r")
        m.def_function[linear_backward_binding]("x_cnn_linear_backward_r")
        m.def_function[softmax_xent_binding]("x_cnn_softmax_xent_r")
        m.def_function[sgd_binding]("x_cnn_sgd_r")
        m.def_function[adam_binding]("x_cnn_adam_r")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cnn_host: ", e))
