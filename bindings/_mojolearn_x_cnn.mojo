# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE'S GPU BINDING (algorithm expansion lane 8).
Host addresses in, host addresses out; the work is x_cnn/device.mojo. The CPU
twin is bindings/_mojolearn_x_cnn_host.mojo, same names, same contract."""
from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32, read_i32, copy_f32
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cnn.ops import NN14_BOUNDED_IM2COL

from x_cnn.ops import CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, conv_params
from x_cnn.ops import PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW, pool_params
from x_cnn.ops import FP, IP
from x_cnn.device import (
    gemm_into, conv2d_forward_into, conv2d_backward_into, maxpool2d_forward_into, maxpool2d_backward_into,
    avgpool2d_forward_into, avgpool2d_backward_into, relu_forward_into, relu_backward_into, add_into, mul_into,
    linear_forward_into, linear_backward_into, softmax_xent_into, sgd_into, adam_into,
    batchnorm_forward_into, batchnorm_backward_into, dropout2d_into, spmm_into, pad2d_forward_into,
    pad2d_backward_into, conv_block_forward_into, conv_block_backward_into,
    res_alloc, res_free, res_upload, res_download, res_argmax, res_gather, res_gather_pair, opt_many_resident,
    gemm_m, conv2d_forward_m, conv2d_backward_m, maxpool2d_forward_m, maxpool2d_backward_m,
    avgpool2d_forward_m, avgpool2d_backward_m, map2_m, linear_forward_m, linear_backward_m,
    batchnorm_forward_m, batchnorm_backward_m, dropout2d_m, spmm_m,
)
from x_cnn.ops import relu_fwd_at, relu_bwd_at, add_at, mul_at, bias_rows_at
from x_cnn.device import graph_op_device as graph_op_impl
from x_cnn.device import adaptive_pool_device as adaptive_pool_impl
from x_cnn.device import gcn_norm_device as gcn_norm_impl
from x_cnn.device import csr_build_device as csr_build_impl
from x_cnn.device import gcn_loops_device
# lane fam-neural (2026-10-04): the `_m` forms of the entries that had none
from x_cnn.device import idn_flags, adaptive_pool_m, graph_op_m, gcn_norm_m, pad2d_m, chan_copy_m
# lane fam2-neural (2026-10-04): the device epoch (order, Adam scalars and losses on the device)
from x_cnn.ops import idn2_flags, epoch_key, adam_hyper_base, AH_ROW, neural_tape_budget_bytes, neural_numerical_profile
from x_cnn.device import (
    epoch_rows_download, res_gather_pair_perm, adam_hyper_resident, adam_hyper_download, opt_many_resident_h,
    softmax_xent_res_loss,
)
from x_cnn.device import cnn_ctx
from core.device_zero import enqueue_fill
from std.gpu import block_dim, block_idx, thread_idx


# ------------------------------------------------ cpu3-bindings: input checks on the device
# The host entries' label and CSR refusals, as one bulk upload, one check
# launch and two flag words read back (the host walks they replace read
# every label, rowptr entry and edge on the CPU). Same predicates, same
# messages in the same priority; every writer of a flag stores the same 1.
comptime _CHK_TPB = 256


def _label_bad_kernel(labels: IP, n: Int32, k: Int32, flag: IP):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        if labels.unsafe_load(i) >= k:
            flag.unsafe_store(0, Int32(1))


def _csr_bad_kernel(csr: IP, n: Int32, nnz: Int32, flag: IP):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        if csr.unsafe_load(i + 1) < csr.unsafe_load(i):
            flag.unsafe_store(0, Int32(1))
    if i < Int(nnz):
        var c = csr.unsafe_load(Int(n) + 1 + i)
        var r = csr.unsafe_load(Int(n) + 1 + Int(nnz) + i)
        if c < Int32(0) or c >= n or r < Int32(0) or r >= n:
            flag.unsafe_store(1, Int32(1))


def _device_int_check(src: IP, words: Int, n: Int, m: Int, csr: Bool) raises -> Tuple[Int32, Int32]:
    """`words` int32 of the caller's `src` checked on the device: the
    label check (`m` = classes) or the CSR check (`m` = nnz). Returns the
    two flag words."""
    var ctx = cnn_ctx()
    var d = ctx.enqueue_create_buffer[DType.int32](max(words, 1))
    if words > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=src)
    var flag = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, flag, Int32(0))
    var span = max(n, m) if csr else n
    if span > 0:
        if csr:
            ctx.enqueue_function[_csr_bad_kernel](
                d.unsafe_ptr(), Int32(n), Int32(m), flag.unsafe_ptr(),
                grid_dim=((span + _CHK_TPB - 1) // _CHK_TPB, 1, 1), block_dim=(_CHK_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[_label_bad_kernel](
                d.unsafe_ptr(), Int32(n), Int32(m), flag.unsafe_ptr(),
                grid_dim=((span + _CHK_TPB - 1) // _CHK_TPB, 1, 1), block_dim=(_CHK_TPB, 1, 1),
            )
    var host = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_buf=host, src_buf=flag)
    ctx.synchronize()
    var f0 = host.unsafe_ptr().unsafe_load(0)
    var f1 = host.unsafe_ptr().unsafe_load(1)
    _ = d^
    _ = flag^
    _ = host^
    return (f0, f1)


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
        if _device_int_check(py_labels, n, n, k, False)[0] != Int32(0):
            raise Error("x_cnn softmax: a label is not a class index")
    var logits = _fp(logits_addr)
    var pg = _fp(grad_addr)
    var pp = _fp(proba_addr)
    var loss = Float32(0)
    with GILReleased(Python()):
        loss = softmax_xent_into[resident](logits, py_labels, n, k, pg, pp)
    return PythonObject(Float64(loss))


def _is_list(o: PythonObject) raises -> Bool:
    var bi = Python.import_module("builtins")
    return Bool(py=bi.isinstance(o, bi.list))


def _many(ws_: PythonObject, gs: PythonObject, bs: PythonObject, params: PythonObject) raises -> Tuple[List[Int], List[Int], List[Int], List[Int]]:
    """lane/cnn-apple2: the resident optimizer entries' list form, one
    handle per parameter in each list and params = their element counts."""
    var k = Int(py=len(params))
    if Int(py=len(ws_)) != k or Int(py=len(gs)) != k or Int(py=len(bs)) != k:
        raise Error("x_cnn optimizer: one handle per parameter in each list and one count each")
    var a = List[Int]()
    var b = List[Int]()
    var c = List[Int]()
    var n = List[Int]()
    for j in range(k):  # small-loop(k: parameter tensors of the model): one handle triple per tensor, not data
        a.append(Int(py=ws_[j]))
        b.append(Int(py=gs[j]))
        c.append(Int(py=bs[j]))
        var nj = Int(py=params[j])
        if nj <= 0:
            raise Error("x_cnn: a positive element count is required")
        n.append(nj)
    return (a^, b^, c^, n^)


def sgd_binding[resident: Bool = False](w_addr: PythonObject, g_addr: PythonObject, v_addr: PythonObject, params: PythonObject, hyper: PythonObject) raises -> PythonObject:
    """In place: w and the momentum buffer v. hyper = [lr, momentum,
    weight_decay, dampening, nesterov (0/1), first step (0/1)]; the last
    three default to 0. Resident, w/g/v may be LISTS of handles (one per
    parameter, params = their counts): every parameter in one entry."""
    var h = List[Float32]()
    var nh = Int(py=len(hyper))
    if nh < 3 or nh > 6:
        raise Error("x_cnn sgd: hyper is [lr, momentum, weight_decay(, dampening, nesterov, first)]")
    for k in range(6):
        h.append(Float32(Float64(py=hyper[k])) if k < nh else Float32(0))
    comptime if resident:
        if _is_list(w_addr):
            var t = _many(w_addr, g_addr, v_addr, params)
            with GILReleased(Python()):
                opt_many_resident[False](t[0], t[1], t[2], t[3], h)
            return PythonObject(len(t[3]))
    var n = _count(params)
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
    if n <= 0 or F <= 0 or nnz < 0 or mode < 0 or mode > 3:
        raise Error("x_cnn spmm: positive n and F, nnz >= 0, mode in {0, 1, 2, 3}")
    var csr = read_i32(Int(py=csr_addr), n + 1 + 2 * nnz)
    if Int(csr[0]) != 0 or Int(csr[n]) != nnz:
        raise Error("x_cnn spmm: rowptr must start at 0 and end at nnz")
    var bad = _device_int_check(
        csr.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), len(csr), n, nnz, True
    )
    if bad[0] != Int32(0):
        raise Error("x_cnn spmm: rowptr must be non-decreasing")
    if bad[1] != Int32(0):
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
    1 - lr * weight_decay] (x_cnn/ops.mojo adam_at). Resident, w/g/mv may
    be LISTS of handles (one per parameter, params = their counts)."""
    if Int(py=len(hyper)) != 9:
        raise Error("x_cnn adam: hyper has 9 entries")
    var h = List[Float32]()
    for k in range(9):
        h.append(Float32(Float64(py=hyper[k])))
    comptime if resident:
        if _is_list(w_addr):
            var t = _many(w_addr, g_addr, mv_addr, params)
            with GILReleased(Python()):
                opt_many_resident[True](t[0], t[1], t[2], t[3], h)
            return PythonObject(len(t[3]))
    var n = _count(params)
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
    """Resident dst rows = resident src rows[r] (int32 host indices); params = [n, row words].
    lane/cnn-apple2: dst and src may be two-handle LISTS with params = [n,
    row words, row2 words] (the batch and its labels in one entry)."""
    var n = Int(py=params[0])
    if _is_list(dst):
        if Int(py=len(dst)) != 2 or Int(py=len(src)) != 2 or Int(py=len(params)) != 3:
            raise Error("x_cnn res_gather: the pair form is [dst, dst2], [src, src2], [n, row, row2]")
        var d0 = Int(py=dst[0]); var d1 = Int(py=dst[1])
        var s0 = Int(py=src[0]); var s1 = Int(py=src[1])
        var r0 = Int(py=params[1]); var r1 = Int(py=params[2])
        var rq = _ip(rows_addr)
        with GILReleased(Python()):
            res_gather_pair(d0, s0, r0, d1, s1, r1, rq, n)
        return PythonObject(n)
    var row = Int(py=params[1])
    var d = Int(py=dst)
    var sr = Int(py=src)
    var rp = _ip(rows_addr)
    with GILReleased(Python()):
        res_gather(d, sr, rp, n, row)
    return PythonObject(n)


def res_argmax_binding(h: PythonObject, dst_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """dst (int32, n) = each row's first largest column of the resident
    n x k block h (params [n, k]), computed on the device; only the labels
    come down (lane pyglue-numeric: the classifier's numpy argmax)."""
    var n = Int(py=params[0])
    var k = Int(py=params[1])
    var hh = Int(py=h)
    var dst = _ip(dst_addr)
    with GILReleased(Python()):
        res_argmax(hh, dst, n, k)
    return PythonObject(n)


def res_download_binding(h: PythonObject, dst_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    var dst = _fp(dst_addr)
    var nn = Int(py=n)
    var hh = Int(py=h)
    with GILReleased(Python()):
        res_download(hh, dst, nn)
    return PythonObject(nn)


# ------------------------------------------------------- mixed residency
# lane gap-neural-overhead2 (2026-10-02): the `_m` entries (x_cnn/device.mojo
# "mixed residency"). `addrs` lists the entry's arrays in the order its
# docstring names; bit k of `dev` says addrs[k] is a resident device address
# (`x_cnn_res_alloc`'s) instead of a host one. The same parameter checks as
# the host entries; the same device bodies (the host entries are these with
# `dev = 0`).


def _addrs(addrs: PythonObject, k: Int) raises -> List[Int]:
    if Int(py=len(addrs)) != k:
        raise Error("x_cnn: this entry takes " + String(k) + " addresses")
    var out = List[Int]()
    for i in range(k):  # small-loop(k: arrays of one entry, at most nine): reads address list, not data
        out.append(Int(py=addrs[i]))
    return out^


def gemm_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [A, B, C]; params = [m, n, k, op]."""
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var op = Int(py=params[3])
    if m <= 0 or n <= 0 or k <= 0 or op < 0 or op > 2:
        raise Error("x_cnn gemm: positive m, n, k and op in {0, 1, 2} required")
    var a = _addrs(addrs, 3)
    var d = Int(py=dev)
    with GILReleased(Python()):
        gemm_m(a, d, m, n, k, op)
    return PythonObject(m * n)


def conv2d_forward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [x, w, bias, out]."""
    var prm = conv_params(_ints(params))
    var a = _addrs(addrs, 4)
    var d = Int(py=dev)
    with GILReleased(Python()):
        conv2d_forward_m(a, d, prm)
    return PythonObject(Int(prm[CP_N]) * Int(prm[CP_OC]) * Int(prm[CP_OH]) * Int(prm[CP_OW]))


def conv2d_backward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [x, w, dout, dx, dW, db]."""
    var prm = conv_params(_ints(params))
    var a = _addrs(addrs, 6)
    var d = Int(py=dev)
    with GILReleased(Python()):
        conv2d_backward_m(a, d, prm)
    return PythonObject(Int(prm[CP_N]) * Int(prm[CP_C]) * Int(prm[CP_H]) * Int(prm[CP_W]))


def maxpool2d_forward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [x, out, idx (int32)]."""
    var prm = _pool_prm(params)
    var a = _addrs(addrs, 3)
    var d = Int(py=dev)
    with GILReleased(Python()):
        maxpool2d_forward_m(a, d, prm)
    return PythonObject(_pool_counts(prm)[1])


def maxpool2d_backward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [dout, idx (int32), dx]."""
    var prm = _pool_prm(params)
    var a = _addrs(addrs, 3)
    var d = Int(py=dev)
    with GILReleased(Python()):
        maxpool2d_backward_m(a, d, prm)
    return PythonObject(_pool_counts(prm)[0])


def avgpool2d_forward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [x, out]."""
    var prm = _pool_prm(params)
    var a = _addrs(addrs, 2)
    var d = Int(py=dev)
    with GILReleased(Python()):
        avgpool2d_forward_m(a, d, prm)
    return PythonObject(_pool_counts(prm)[1])


def avgpool2d_backward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [dout, dx]."""
    var prm = _pool_prm(params)
    var a = _addrs(addrs, 2)
    var d = Int(py=dev)
    with GILReleased(Python()):
        avgpool2d_backward_m(a, d, prm)
    return PythonObject(_pool_counts(prm)[0])


def map2_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [a, b, out]; params = [kind, n] or [4, n, cols]: kind 0 ReLU
    forward (a = x; b unused), 1 ReLU backward (a = x, b = g), 2 add, 3
    multiply (`x_cnn_add`'s and `x_cnn_mul`'s element functions), 4 the
    row bias add (a = y (n = rows * cols), b = bias (cols); `bias_rows_at`,
    the linear forward's)."""
    var kind = Int(py=params[0])
    var n = Int(py=params[1])
    if n <= 0 or kind < 0 or kind > 4:
        raise Error("x_cnn map2: positive n and kind in 0..4")
    var a = _addrs(addrs, 3)
    var d = Int(py=dev)
    if kind == 4:
        var cols = Int(py=params[2])
        if cols <= 0 or n % cols != 0:
            raise Error("x_cnn map2: the row bias needs cols dividing n")
        var prm: List[Int32] = [Int32(n // cols), Int32(0), Int32(cols)]
        with GILReleased(Python()):
            map2_m[bias_rows_at](a, d, n, cols, n, prm)
        return PythonObject(n)
    var prm: List[Int32] = [0, 0, 0]
    with GILReleased(Python()):
        if kind == 0:
            map2_m[relu_fwd_at](a, d & ~2, n, 0, n, prm)
        elif kind == 1:
            map2_m[relu_bwd_at](a, d, n, n, n, prm)
        elif kind == 2:
            map2_m[add_at](a, d, n, n, n, prm)
        else:
            map2_m[mul_at](a, d, n, n, n, prm)
    return PythonObject(n)


def linear_forward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [x, w, b, y]."""
    var s = _lin(params)
    var a = _addrs(addrs, 4)
    var d = Int(py=dev)
    with GILReleased(Python()):
        linear_forward_m(a, d, s[0], s[1], s[2])
    return PythonObject(s[0] * s[2])


def linear_backward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [x, w, g, dx, dW, db]."""
    var s = _lin(params)
    var a = _addrs(addrs, 6)
    var d = Int(py=dev)
    with GILReleased(Python()):
        linear_backward_m(a, d, s[0], s[1], s[2])
    return PythonObject(s[0])


def batchnorm_forward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [x, y, running, aux] (the host entry's order); params = [N, C, HW, training]."""
    var prm = _bn_prm(params)
    var training = Int(py=params[3]) != 0
    if training and Int(prm[0]) * Int(prm[2]) < 2:
        raise Error("x_cnn batchnorm: expected more than 1 value per channel when training")
    var h = _addrs(addrs, 4)
    var a: List[Int] = [h[0], h[2], h[3], h[1]]
    var d0 = Int(py=dev)
    # host order bits (x, y, running, aux) to the body's (x, running, aux, y)
    var d = (d0 & 1) | (((d0 >> 2) & 1) << 1) | (((d0 >> 3) & 1) << 2) | (((d0 >> 1) & 1) << 3)
    with GILReleased(Python()):
        batchnorm_forward_m(a, d, prm, training)
    return PythonObject(Int(prm[0]) * Int(prm[1]) * Int(prm[2]))


def batchnorm_backward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [x, g, dx, aux] (the host entry's order); params = [N, C, HW, training]."""
    var prm = _bn_prm(params)
    var training = Int(py=params[3]) != 0
    var h = _addrs(addrs, 4)
    var a: List[Int] = [h[0], h[1], h[3], h[2]]
    var d0 = Int(py=dev)
    var d = (d0 & 3) | (((d0 >> 3) & 1) << 2) | (((d0 >> 2) & 1) << 3)
    with GILReleased(Python()):
        batchnorm_backward_m(a, d, prm, training)
    return PythonObject(Int(prm[0]) * Int(prm[1]) * Int(prm[2]))


def dropout2d_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject, drop_p: PythonObject) raises -> PythonObject:
    """addrs = [x, y, mask]; params as `x_cnn_dropout2d`'s."""
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
    var a = _addrs(addrs, 3)
    var d = Int(py=dev)
    var total = N * C * HW
    with GILReleased(Python()):
        dropout2d_m(a, d, total, prm, h)
    return PythonObject(total)


def csr_upload_binding(csr_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """A resident copy of a CSR block [rowptr | col | row] (params [n, F,
    nnz, mode]), checked ONCE here as `x_cnn_spmm` checks it on every call;
    its handle (free with `x_cnn_res_free`)."""
    var t = _csr(csr_addr, params)
    var csr = t[0].copy()
    var words = len(csr)
    var h = res_alloc(words)
    var src = FP(unsafe_from_address=Int(csr.unsafe_ptr()))
    with GILReleased(Python()):
        res_upload(h, src, words)
    _ = csr^
    return PythonObject(h)


def csr_build_binding(rows_addr: PythonObject, cols_addr: PythonObject, csr_addr: PythonObject, order_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """The CSR view of an edge list, on the device (x_cnn/device.mojo
    csr_build_device): `rows`, `cols` int32 [nnz] in [0, n); `csr` int32
    [n + 1 + 2 * nnz] gets [rowptr | col | row] in ascending (row, col) order,
    ties in edge order; `order` int32 [nnz] the edge ids in that order.
    params = [n, nnz]. Returns n + 1 + 2 * nnz."""
    var n = Int(py=params[0])
    var nnz = Int(py=params[1])
    if n <= 0 or nnz < 0:
        raise Error("x_cnn csr_build: positive n and nnz >= 0 required")
    var pc = _ip(csr_addr)
    var rows = _ip(rows_addr) if nnz > 0 else pc
    var cols = _ip(cols_addr) if nnz > 0 else pc
    var order = _ip(order_addr) if nnz > 0 else pc
    with GILReleased(Python()):
        csr_build_impl(rows, cols, nnz, n, pc, order)
    return PythonObject(n + 1 + 2 * nnz)


def gcn_loops_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """GCN's remaining self loops on the device (lane fix-n1-lm-neural,
    x_cnn/device.mojo gcn_loops_device): addrs = [src, dst (int32 [nnz]),
    w (float32 [nnz]), src_out, dst_out (int32 [nnz + n]), w_out (float32
    [nnz + n])]; params = [n, nnz, improved]. Returns K: the outputs' first
    K + n slots are the edge list."""
    var n = Int(py=params[0])
    var nnz = Int(py=params[1])
    var fill = Float32(2.0) if Int(py=params[2]) != 0 else Float32(1.0)
    if n <= 0 or nnz < 0 or Int(py=len(addrs)) != 6:
        raise Error("x_cnn gcn_loops: addrs [src, dst, w, src_out, dst_out, w_out], positive n and nnz >= 0 required")
    var so = _ip(addrs[3])
    var dso = _ip(addrs[4])
    var wo = _fp(addrs[5])
    var src = _ip(addrs[0]) if nnz > 0 else so
    var dst = _ip(addrs[1]) if nnz > 0 else so
    var w = _fp(addrs[2]) if nnz > 0 else wo
    var k = 0
    with GILReleased(Python()):
        k = gcn_loops_device(src, dst, w, nnz, n, fill, so, dso, wo)
    return PythonObject(k)


def spmm_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [vals, h, csr, out]; params = [n, F, nnz, mode]. A resident
    csr (bit 2) is a `x_cnn_csr_upload` handle, checked when it was made; a
    host one is checked here as `x_cnn_spmm` checks it."""
    var n = Int(py=params[0])
    var F = Int(py=params[1])
    var nnz = Int(py=params[2])
    var mode = Int(py=params[3])
    var a = _addrs(addrs, 4)
    var d = Int(py=dev)
    if (d >> 2) & 1 == 0:
        _ = _csr(addrs[2], params)
    elif n <= 0 or F <= 0 or nnz < 0 or mode < 0 or mode > 3:
        raise Error("x_cnn spmm: positive n and F, nnz >= 0, mode in {0, 1, 2, 3}")
    var prm: List[Int32] = [Int32(n), Int32(F), Int32(nnz), Int32(mode)]
    if nnz <= 0:
        # the host entry's: no values are read, h stands in for them
        a[0] = a[1]
        d = (d & ~1) | ((d >> 1) & 1)
    with GILReleased(Python()):
        spmm_m(a, d, nnz, n + 1 + 2 * nnz, prm)
    return PythonObject(n * F)


# ------------------------------------------------- lane fam-neural `_m` forms
# lane fam-neural (2026-10-04): x_cnn/device.mojo "lane fam-neural: `_m`
# forms". New entries only; no existing entry changes. The same parameter
# checks as the host entries they stand beside. `x_cnn_idn_flags` reports
# which of the lane's switches this build has on (the glue uses an entry
# only when its bit is set; the CPU twin has no such entry, so its glue
# keeps the host entries).


def idn_flags_binding() raises -> PythonObject:
    return PythonObject(idn_flags())


def adaptive_pool_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [in, out, idx (int32)]; params = [N, C, H, W, OH, OW, kind]
    as `x_cnn_adaptive_pool`'s. The average kinds (0, 1) never touch idx."""
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
    var a = _addrs(addrs, 3)
    var d = Int(py=dev)
    with GILReleased(Python()):
        adaptive_pool_m(a, d, prm, kind)
    return PythonObject(nout if kind == 0 or kind == 2 else nin)


def graph_op_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [a, b, aux, out, csr]; params = [n, F, nnz, kind] as
    `x_cnn_graph_op`'s. A resident csr (bit 4) is a `x_cnn_csr_upload`
    handle, checked when it was made; a host one is checked here. b is read
    only by the backward kinds (1, 3), csr only by the SAGE kinds (0, 1)."""
    var kind = Int(py=params[3])
    if kind < 0 or kind > 3:
        raise Error("x_cnn graph op: kind in 0..3")
    var n = Int(py=params[0])
    var F = Int(py=params[1])
    var nnz = Int(py=params[2])
    var a = _addrs(addrs, 5)
    var d = Int(py=dev)
    if kind < 2 and (d >> 4) & 1 == 0:
        _ = _csr_ints(addrs[4], n, F, nnz, 0)
    elif n <= 0 or F <= 0 or nnz < 0:
        raise Error("x_cnn graph op: positive n and F, nnz >= 0")
    var prm: List[Int32] = [Int32(n), Int32(F), Int32(nnz), Int32(0)]
    with GILReleased(Python()):
        graph_op_m(a, d, prm, kind)
    return PythonObject(n * F)


def gcn_norm_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [w, vals, csr]; params = [n, F, nnz, mode] as `x_cnn_gcn_norm`'s."""
    var n = Int(py=params[0])
    var F = Int(py=params[1])
    var nnz = Int(py=params[2])
    var mode = Int(py=params[3])
    var a = _addrs(addrs, 3)
    var d = Int(py=dev)
    if (d >> 2) & 1 == 0:
        _ = _csr(addrs[2], params)
    elif n <= 0 or F <= 0 or nnz < 0 or mode < 0 or mode > 3:
        raise Error("x_cnn spmm: positive n and F, nnz >= 0, mode in {0, 1, 2, 3}")
    if nnz <= 0:
        raise Error("x_cnn gcn_norm: at least one edge (the self loops) is required")
    var prm: List[Int32] = [Int32(n), Int32(F), Int32(nnz), Int32(mode)]
    with GILReleased(Python()):
        gcn_norm_m(a, d, prm)
    return PythonObject(nnz)


def pad2d_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [in, out]; params = `x_cnn_pad2d_forward`'s nine and a tenth,
    0 the forward (in x, out padded) or 1 the backward (in the padded
    gradient, out dx)."""
    var prm = _pad_prm(params)
    var back = Int(py=params[9]) != 0
    var nc = Int(prm[0]) * Int(prm[1])
    var nx = nc * Int(prm[2]) * Int(prm[3])
    var no = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    var a = _addrs(addrs, 2)
    var d = Int(py=dev)
    with GILReleased(Python()):
        pad2d_m(a, d, prm, back)
    return PythonObject(nx if back else no)


def chan_copy_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [src, dst]; params = [N, C, HW, cg, c0, place]: place 0 copies
    channels [c0, c0 + cg) of the (N, C, HW) src into the (N, cg, HW) dst;
    place 1 copies the (N, cg, HW) src into those channels of the resident
    (N, C, HW) dst."""
    var N = Int(py=params[0])
    var C = Int(py=params[1])
    var HW = Int(py=params[2])
    var cg = Int(py=params[3])
    var c0 = Int(py=params[4])
    var place = Int(py=params[5]) != 0
    if N <= 0 or C <= 0 or HW <= 0 or cg <= 0 or c0 < 0 or c0 + cg > C:
        raise Error("x_cnn chan copy: positive N, C, HW, cg and a group inside [0, C)")
    var prm: List[Int32] = [Int32(N), Int32(C), Int32(HW), Int32(cg), Int32(c0)]
    var a = _addrs(addrs, 2)
    var d = Int(py=dev)
    with GILReleased(Python()):
        chan_copy_m(a, d, prm, place)
    return PythonObject(N * cg * HW)


# ------------------------------------------------------------ the fit epoch
# lane/py-misc (2026-09-28, audit rank 10): CNNClassifier.fit's steps looped
# HERE instead of in Python. `x_cnn_fit_epoch_r` runs a run of trainer steps
# on the resident arrays: per step the pair gather of the step's rows, each
# block's resident forward, the head, the softmax, the head's backward, each
# block's backward (last to first) and the list-form optimizer, which are
# the very calls `_expansion_cnn.CNNClassifier.fit` makes one binding call
# at a time (the same `_into[True]` functions, the same arguments, the same
# order, each still ending in its own wait). The step's loss is written as
# the Float64 of the Float32 the softmax entry returns, which is what the
# Python loop stored. The optimizer's hyper rows (Adam's step scalars in
# double, SGD's first-step flag) come from Python, computed as before. No
# kernel, operand or order changes: the bits cannot move.


@always_inline
def _at(a: Int) -> FP:
    return FP(unsafe_from_address=a)


@always_inline
def _at_i(a: Int) -> IP:
    return IP(unsafe_from_address=a)


def _epoch_plan(
    obj: PythonObject, nb: Int, mut cs: List[List[List[Int32]]], mut ps: List[List[List[Int32]]],
    mut pools: List[List[Bool]],
) raises:
    """Appends one plan: [[conv params, pool params] per block] -> the blocks' parameter blocks."""
    if Int(py=len(obj)) != nb:
        raise Error("x_cnn fit epoch: one [conv, pool] plan per block")
    var c = List[List[Int32]]()
    var p = List[List[Int32]]()
    var o = List[Bool]()
    for j in range(nb):  # small-loop(nb: conv blocks of the network): one plan entry per block, not data
        var t = _block_prms(obj[j][0], obj[j][1])
        c.append(t[0].copy())
        p.append(t[1].copy())
        o.append(t[2])
    cs.append(c^)
    ps.append(p^)
    pools.append(o^)


def fit_epoch_r_binding(
    spec: PythonObject, rows_addr: PythonObject, hyper_addr: PythonObject, losses_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """params = [n, batch, adam (0/1)]: ceil(n / batch) steps over host int32
    rows[0:n] (the epoch's order), step t on rows[t*batch : t*batch + m];
    hyper = float64 rows of 9 (Adam) or 6 (SGD) per step; losses = float64 per
    step. spec: blocks [[w, b, gw, gb, out, idx, gout, cols, conv out] per
    block], head [w, b, gw, gb], a [x, y, logits, glog, proba, ghead], opt
    [params, grads, buffers, counts], data [X, y, row words], dims [flat, k],
    plan_full / plan_last [[conv params, pool params] per block] at the batch
    and at the last step's rows."""
    var n = Int(py=params[0])
    var batch = Int(py=params[1])
    var adam = Int(py=params[2]) != 0
    if n <= 0 or batch <= 0:
        raise Error("x_cnn fit epoch: positive rows and batch required")
    var nh = 9 if adam else 6
    var blocks = spec["blocks"]
    var nb = Int(py=len(blocks))
    var bh = List[List[Int]]()
    for j in range(nb):  # small-loop(nb: conv blocks of the network): one handle list per block, not data
        var h = _ints(blocks[j])
        if len(h) != 9:
            raise Error("x_cnn fit epoch: a block is [w, b, gw, gb, out, idx, gout, cols, conv out]")
        bh.append(h^)
    var head = _ints(spec["head"])
    var a = _ints(spec["a"])
    var data = _ints(spec["data"])
    var dims = _ints(spec["dims"])
    var opt = spec["opt"]
    var t = _many(opt[0], opt[1], opt[2], opt[3])
    # plan 0 at the batch, plan 1 at the last step's rows
    var cs = List[List[List[Int32]]]()
    var ps = List[List[List[Int32]]]()
    var pools = List[List[Bool]]()
    _epoch_plan(spec["plan_full"], nb, cs, ps, pools)
    _epoch_plan(spec["plan_last"], nb, cs, ps, pools)
    var flat = dims[0]
    var k = dims[1]
    var steps = (n + batch - 1) // batch
    var rows = _ip(rows_addr)
    var hyp = f64_ptr(Int(py=hyper_addr))
    var losses = f64_ptr(Int(py=losses_addr))
    with GILReleased(Python()):
        for st in range(steps):  # small-loop(steps: optimizer steps of one epoch): orchestration that launches device work per step
            var s0 = st * batch
            var m = min(batch, n - s0)
            var q = 0 if m == batch else 1
            res_gather_pair(a[0], data[0], data[2], a[1], data[1], 1, rows + s0, m)
            var src = a[0]
            for j in range(nb):
                conv_block_forward_into[True](
                    _at(src), _at(bh[j][0]), _at(bh[j][1]), cs[q][j], ps[q][j], pools[q][j], _at(bh[j][4]),
                    _at_i(bh[j][5]), bh[j][7], bh[j][8],
                )
                src = bh[j][4]
            linear_forward_into[True](_at(src), _at(head[0]), _at(head[1]), m, flat, k, _at(a[2]))
            var loss = softmax_xent_into[True](_at(a[2]), _at_i(a[1]), m, k, _at(a[3]), _at(a[4]))
            losses[st] = Float64(loss)
            var glast = bh[nb - 1][6] if nb > 0 else a[5]
            linear_backward_into[True](
                _at(src), _at(head[0]), _at(a[3]), m, flat, k, _at(glast), _at(head[2]), _at(head[3])
            )
            for jj in range(nb):
                var j = nb - 1 - jj
                var bsrc = bh[j - 1][4] if j > 0 else a[0]
                var dx = bh[j - 1][6] if j > 0 else 0
                var want = dx != 0
                conv_block_backward_into[True](
                    _at(bsrc), _at(bh[j][0]), _at(bh[j][1]), _at(bh[j][6]), _at_i(bh[j][5]), cs[q][j], ps[q][j],
                    pools[q][j], want, _at(dx) if want else _at(bh[j][3]), _at(bh[j][2]), _at(bh[j][3]), bh[j][7],
                    bh[j][8],
                )
            var h = List[Float32]()
            for e in range(nh):  # small-loop(nh: optimizer scalars of one step, six or nine): builds one launch parameter list, not data
                h.append(Float32(hyp[st * nh + e]))
            if adam:
                opt_many_resident[True](t[0], t[1], t[2], t[3], h)
            else:
                opt_many_resident[False](t[0], t[1], t[2], t[3], h)
    return PythonObject(steps)


# ------------------------------------------------ lane fam2-neural: the device epoch
# `x_cnn_fit_epoch_d` is `x_cnn_fit_epoch_r` with nothing computed on the
# host and nothing crossing the bus inside the epoch: each step's rows come
# from `epoch_rows_at` on the device (was the CPU's Fisher-Yates order,
# uploaded per step), Adam's step scalars from `adam_hyper_at` into a
# resident block (was `adam_hyper_f64` on the CPU, uploaded per step), and
# each step's mean loss stays in a resident block (`blk_fold_at`; was n row
# losses down per step and a host fold) that comes down once at the end.
# The forward, backward and optimizer launches are the `_r` entry's.


def idn2_flags_binding() raises -> PythonObject:
    return PythonObject(idn2_flags())


def neural_tape_budget_binding() raises -> PythonObject:
    return PythonObject(neural_tape_budget_bytes())


def neural_numerical_profile_binding() raises -> PythonObject:
    return PythonObject(neural_numerical_profile())


def _seed64(lo: PythonObject, hi: PythonObject) raises -> UInt64:
    return (UInt64(Int(py=hi)) << UInt64(32)) | UInt64(Int(py=lo))


def epoch_rows_binding(dst_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """Host int32 dst[0:n] = epoch `epoch`'s row order. params = [n, epoch,
    shuffle (0/1), seed low 32 bits, seed high 32 bits]."""
    var n = Int(py=params[0])
    if n < 1 or n >= 2147483647:
        raise Error("x_cnn epoch rows: 1 <= n < 2^31 - 1")
    var key = epoch_key(_seed64(params[3], params[4]), Int(py=params[1]))
    var shuffle = Int(py=params[2]) != 0
    var dst = _ip(dst_addr)
    with GILReleased(Python()):
        epoch_rows_download(dst, n, 0, n, shuffle, key)
    return PythonObject(n)


def _adam_base(fparams: PythonObject) raises -> List[Float32]:
    if Int(py=len(fparams)) != 6:
        raise Error("x_cnn adam hyper: fparams is [lr, beta1, beta2, eps, weight_decay, decoupled]")
    return adam_hyper_base(
        Float64(py=fparams[0]), Float64(py=fparams[1]), Float64(py=fparams[2]), Float64(py=fparams[3]),
        Float64(py=fparams[4]), Float64(py=fparams[5]),
    )


def adam_hyper_d_binding(dst_addr: PythonObject, params: PythonObject, fparams: PythonObject) raises -> PythonObject:
    """Host float64 dst[0 : 9 nsteps] = `adam_at`'s hyper blocks of 1-based
    steps step0 .. step0 + nsteps - 1, computed on the device in float32
    pairs (`adam_hyper_at`) and widened. params = [step0, nsteps]; fparams
    = [lr, beta1, beta2, eps, weight_decay, decoupled]."""
    var step0 = Int(py=params[0])
    var nsteps = Int(py=params[1])
    if step0 < 1 or nsteps < 0:
        raise Error("x_cnn adam hyper: step0 >= 1, nsteps >= 0")
    var base = _adam_base(fparams)
    var out = List[Float32](length=nsteps * AH_ROW + 1, fill=Float32(0))
    var po = out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    with GILReleased(Python()):
        adam_hyper_download(base, step0, nsteps, po)
    var dst = f64_ptr(Int(py=dst_addr))
    for e in range(nsteps * AH_ROW):  # small-loop(nsteps: optimizer steps, AH_ROW scalars each): widens per step optimizer scalars, not row data
        dst[e] = Float64(out[e])
    _ = base^
    _ = out^
    return PythonObject(nsteps)


def fit_epoch_d_binding(
    spec: PythonObject, losses_addr: PythonObject, params: PythonObject, fparams: PythonObject,
) raises -> PythonObject:
    """params = [n, batch, adam (0/1), steps already taken, epoch, shuffle
    (0/1), seed low 32 bits, seed high 32 bits]; fparams = Adam's [lr,
    beta1, beta2, eps, weight_decay, decoupled] or SGD's [lr, momentum,
    weight_decay, dampening, nesterov, 0]; losses = float64 per step; spec
    as `x_cnn_fit_epoch_r`."""
    var n = Int(py=params[0])
    var batch = Int(py=params[1])
    var adam = Int(py=params[2]) != 0
    var done = Int(py=params[3])
    var epoch = Int(py=params[4])
    var shuffle = Int(py=params[5]) != 0
    if n <= 0 or batch <= 0 or n >= 2147483647 or done < 0:
        raise Error("x_cnn fit epoch: positive rows and batch required")
    var key = epoch_key(_seed64(params[6], params[7]), epoch)
    var blocks = spec["blocks"]
    var nb = Int(py=len(blocks))
    var bh = List[List[Int]]()
    for j in range(nb):  # small-loop(nb: conv blocks of the network): one handle list per block, not data
        var h = _ints(blocks[j])
        if len(h) != 9:
            raise Error("x_cnn fit epoch: a block is [w, b, gw, gb, out, idx, gout, cols, conv out]")
        bh.append(h^)
    var head = _ints(spec["head"])
    var a = _ints(spec["a"])
    var data = _ints(spec["data"])
    var dims = _ints(spec["dims"])
    var opt = spec["opt"]
    var t = _many(opt[0], opt[1], opt[2], opt[3])
    var cs = List[List[List[Int32]]]()
    var ps = List[List[List[Int32]]]()
    var pools = List[List[Bool]]()
    _epoch_plan(spec["plan_full"], nb, cs, ps, pools)
    _epoch_plan(spec["plan_last"], nb, cs, ps, pools)
    var flat = dims[0]
    var k = dims[1]
    var steps = (n + batch - 1) // batch
    var losses = f64_ptr(Int(py=losses_addr))
    # the hyper rows: Adam's from the device kernel; SGD's two rows (the
    # first step's, flag 1, then every later step's), words only
    var base = List[Float32]()
    if adam:
        base = _adam_base(fparams)
    else:
        if Int(py=len(fparams)) != 6:
            raise Error("x_cnn fit epoch: SGD's fparams is [lr, momentum, weight_decay, dampening, nesterov, 0]")
        for r in range(2):
            for e in range(5):
                base.append(Float32(Float64(py=fparams[e])))
            base.append(Float32(1) if r == 0 else Float32(0))
    var l32 = List[Float32](length=steps, fill=Float32(0))
    with GILReleased(Python()):
        var hbuf = res_alloc(steps * AH_ROW if adam else 12)
        var lbuf = res_alloc(steps)
        if adam:
            adam_hyper_resident(base, done + 1, steps, hbuf)
        else:
            res_upload(hbuf, base.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 12)
        for st in range(steps):
            var s0 = st * batch
            var m = min(batch, n - s0)
            var q = 0 if m == batch else 1
            res_gather_pair_perm(a[0], data[0], data[2], a[1], data[1], 1, n, s0, m, shuffle, key)
            var src = a[0]
            for j in range(nb):
                conv_block_forward_into[True](
                    _at(src), _at(bh[j][0]), _at(bh[j][1]), cs[q][j], ps[q][j], pools[q][j], _at(bh[j][4]),
                    _at_i(bh[j][5]), bh[j][7], bh[j][8],
                )
                src = bh[j][4]
            linear_forward_into[True](_at(src), _at(head[0]), _at(head[1]), m, flat, k, _at(a[2]))
            softmax_xent_res_loss(_at(a[2]), _at_i(a[1]), m, k, _at(a[3]), _at(a[4]), lbuf, st)
            var glast = bh[nb - 1][6] if nb > 0 else a[5]
            linear_backward_into[True](
                _at(src), _at(head[0]), _at(a[3]), m, flat, k, _at(glast), _at(head[2]), _at(head[3])
            )
            for jj in range(nb):
                var j = nb - 1 - jj
                var bsrc = bh[j - 1][4] if j > 0 else a[0]
                var dx = bh[j - 1][6] if j > 0 else 0
                var want = dx != 0
                conv_block_backward_into[True](
                    _at(bsrc), _at(bh[j][0]), _at(bh[j][1]), _at(bh[j][6]), _at_i(bh[j][5]), cs[q][j], ps[q][j],
                    pools[q][j], want, _at(dx) if want else _at(bh[j][3]), _at(bh[j][2]), _at(bh[j][3]), bh[j][7],
                    bh[j][8],
                )
            if adam:
                opt_many_resident_h[True](t[0], t[1], t[2], t[3], hbuf + 4 * AH_ROW * st)
            else:
                opt_many_resident_h[False](t[0], t[1], t[2], t[3], hbuf + (0 if done + st == 0 else 24))
        res_download(lbuf, l32.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), steps)
        res_free(hbuf)
        res_free(lbuf)
    for st in range(steps):  # small-loop(steps: optimizer steps of one epoch): widens one loss scalar per step, not row data
        losses[st] = Float64(l32[st])
    _ = base^
    _ = l32^
    return PythonObject(steps)


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
        m.def_function[nn14_bounded_im2col_binding]("x_cnn_bounded_im2col")
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
        m.def_function[res_argmax_binding]("x_cnn_res_argmax")
        m.def_function[conv_block_forward_r_binding]("x_cnn_conv_block_forward_r")
        m.def_function[conv_block_backward_r_binding]("x_cnn_conv_block_backward_r")
        m.def_function[res_gather_binding]("x_cnn_res_gather")
        m.def_function[linear_forward_binding[True]]("x_cnn_linear_forward_r")
        m.def_function[linear_backward_binding[True]]("x_cnn_linear_backward_r")
        m.def_function[softmax_xent_binding[True]]("x_cnn_softmax_xent_r")
        m.def_function[sgd_binding[True]]("x_cnn_sgd_r")
        m.def_function[adam_binding[True]]("x_cnn_adam_r")
        m.def_function[fit_epoch_r_binding]("x_cnn_fit_epoch_r")
        m.def_function[gemm_m_binding]("x_cnn_gemm_m")
        m.def_function[conv2d_forward_m_binding]("x_cnn_conv2d_forward_m")
        m.def_function[conv2d_backward_m_binding]("x_cnn_conv2d_backward_m")
        m.def_function[maxpool2d_forward_m_binding]("x_cnn_maxpool2d_forward_m")
        m.def_function[maxpool2d_backward_m_binding]("x_cnn_maxpool2d_backward_m")
        m.def_function[avgpool2d_forward_m_binding]("x_cnn_avgpool2d_forward_m")
        m.def_function[avgpool2d_backward_m_binding]("x_cnn_avgpool2d_backward_m")
        m.def_function[map2_m_binding]("x_cnn_map2_m")
        m.def_function[linear_forward_m_binding]("x_cnn_linear_forward_m")
        m.def_function[linear_backward_m_binding]("x_cnn_linear_backward_m")
        m.def_function[batchnorm_forward_m_binding]("x_cnn_batchnorm_forward_m")
        m.def_function[batchnorm_backward_m_binding]("x_cnn_batchnorm_backward_m")
        m.def_function[dropout2d_m_binding]("x_cnn_dropout2d_m")
        m.def_function[csr_upload_binding]("x_cnn_csr_upload")
        m.def_function[csr_build_binding]("x_cnn_csr_build")
        m.def_function[gcn_loops_binding]("x_cnn_gcn_loops")
        m.def_function[idn_flags_binding]("x_cnn_idn_flags")
        m.def_function[idn2_flags_binding]("x_cnn_idn2_flags")
        m.def_function[neural_tape_budget_binding]("x_cnn_neural_tape_budget_bytes")
        m.def_function[neural_numerical_profile_binding]("x_cnn_numerical_profile")
        m.def_function[epoch_rows_binding]("x_cnn_epoch_rows")
        m.def_function[adam_hyper_d_binding]("x_cnn_adam_hyper_d")
        m.def_function[fit_epoch_d_binding]("x_cnn_fit_epoch_d")
        m.def_function[adaptive_pool_m_binding]("x_cnn_adaptive_pool_m")
        m.def_function[graph_op_m_binding]("x_cnn_graph_op_m")
        m.def_function[gcn_norm_m_binding]("x_cnn_gcn_norm_m")
        m.def_function[pad2d_m_binding]("x_cnn_pad2d_m")
        m.def_function[chan_copy_m_binding]("x_cnn_chan_copy_m")
        m.def_function[spmm_m_binding]("x_cnn_spmm_m")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cnn: ", e))


def nn14_bounded_im2col_binding() -> PythonObject:
    """Source-arm query for saved-buffer allocation, never data-dependent."""
    return PythonObject(1 if NN14_BOUNDED_IM2COL else 0)
