# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_cnn` (the CNN expansion lane). HOST ONLY;
the GPU binding's export names and address contract, the work is
x_cnn/host/ops_host.mojo."""
from core.experiment_guards_neural import NEURAL_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type
from bindings.py_cpython_table import cpython_table

comptime _TPFLAGS_LIST_SUBCLASS = UInt64(1) << 25  # Py_TPFLAGS_LIST_SUBCLASS
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cnn.ops import NN14_BOUNDED_IM2COL

from x_cnn.ops import CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, conv_params
from x_cnn.ops import PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW, pool_params
from x_cnn.host.ops_host import X_CNN_HOST_SABOTAGE
from x_cnn.ops import FP, IP
from x_cnn.host.mojolearn_x_cnn_host_run import (
    conv_block_forward_run,
    conv_block_backward_run,
    softmax_xent_run,
    sgd_list_run,
    sgd_run,
    adam_list_run,
    adam_run,
)
# The typed bodies (lane thin-trees-metrics, 2026-10-21), moved word for word
# out of this shell; `out_f32` (the CPU column's output seam) lives there too
# and is called from x_cnn/host/mojolearn_x_cnn_host_run.mojo.
# TOMBSTONE: the host fit-epoch helpers (epoch_rows_fill, adam_rows, widen_f32_into, widen_f32_list_into, sgd_hyper_rows, fit_epoch_r_into) deleted 2026-11-09 by lane/x8-cnn-on-executor; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/X8_CNN_HAND_EPOCH.patch
from x_cnn.host.binding_entries_impl import (
    pool_counts as _pool_counts,
    check_labels_below,
)
# lane calltable-xcnn (2026-11-11): the operations the CNN layer doors call
# are typed runs in x_cnn/host/entry_runs_host.mojo (the graph builders'
# former providers/python/x_cnn_host_ops_host_pyapi.mojo folded in), published
# at module init as this image's native entry (core/xcnn_entry_table.mojo);
# their Python entries below only unpack into plain words.
from x_cnn.host.entry_runs_host import x_cnn_host_run, publish_x_cnn_host_entries
from core.xcnn_entry_table import (
    XC_RES_ALLOC, XC_RES_FREE, XC_RES_UPLOAD, XC_RES_DOWNLOAD, XC_GEMM, XC_CONV2D_FORWARD, XC_CONV2D_BACKWARD,
    XC_MAXPOOL2D_FORWARD, XC_MAXPOOL2D_BACKWARD, XC_AVGPOOL2D_FORWARD, XC_AVGPOOL2D_BACKWARD, XC_RELU_FORWARD,
    XC_RELU_BACKWARD, XC_ADD, XC_MUL, XC_LINEAR_FORWARD, XC_LINEAR_BACKWARD, XC_BATCHNORM_FORWARD,
    XC_BATCHNORM_BACKWARD, XC_DROPOUT2D, XC_SPMM, XC_GCN_NORM, XC_PAD2D_FORWARD, XC_PAD2D_BACKWARD,
    XC_ADAPTIVE_POOL, XC_GRAPH_OP, XC_CSR_BUILD, XC_PERM_GATHER, XC_VALS_T, XC_GCN_LOOPS, XC_GCN_BACKWARD_T,
    XC_SAGE_BACKWARD_T,
)
from x_cnn.ops import idn2_flags, neural_tape_budget_bytes, neural_numerical_profile


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


def _ints(params: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for k in range(Int(py=len(params))):  # small-loop(the caller's int parameter words)
        out.append(Int(py=params[k]))
    return out^

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


# TOMBSTONE: x_cnn_conv_block_forward / _backward (non-resident) and the host x_cnn_numeric_mode alias (dead native entries: every caller uses the _r entries / x_cnn_host_numeric_mode) deleted 2026-10-26 by 352103272; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_cnn.patch


def _block_forward(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, idx_addr: PythonObject,
    conv_prm: PythonObject, pool_prm: PythonObject, kcols: Int, ky: Int,
) raises -> PythonObject:
    """kcols, ky: the fit's saved arrays (x's im2col matrix, the conv output
    before the ReLU) or 0, 0."""
    var t = _block_prms(conv_prm, pool_prm)
    var cprm = t[0].copy()
    var pprm = t[1].copy()
    var pool = t[2]
    var ny = Int(cprm[CP_N]) * Int(cprm[CP_OC]) * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var no = _pool_counts(pprm)[1] if pool else ny
    var keep = kcols != 0 and ky != 0
    var px = fp(x_addr)
    var pw = fp(w_addr)
    var pb = fp(b_addr)
    var po = fp(out_addr)
    var noidx = List[Int32](length=1, fill=Int32(0))
    var pi = ip(idx_addr) if pool else noidx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pk = FP(unsafe_from_address=kcols) if keep else po
    var py_ = FP(unsafe_from_address=ky) if keep else po
    with GILReleased(Python()):
        conv_block_forward_run(px, pw, pb, po, pi, cprm, pprm, pool, keep, pk, py_, no)
    _ = noidx^
    return PythonObject(0)


def conv_block_backward_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, g_addr: PythonObject, idx_addr: PythonObject,
    outs: PythonObject, conv_prm: PythonObject, pool_prm: PythonObject,
) raises -> PythonObject:
    """The block's backward from its output gradient g: outs = [dx address
    (0: not wanted, the first block's), dW address, db address], or those
    and the forward's two saved arrays [cols, conv output]."""
    var t = _block_prms(conv_prm, pool_prm)
    var nouts = Int(py=len(outs))
    if nouts != 3 and nouts != 5:
        raise Error("x_cnn conv block backward: outs is [dx, dW, db] or [dx, dW, db, cols, conv output]")
    var kcols = Int(py=outs[3]) if nouts == 5 else 0
    var ky = Int(py=outs[4]) if nouts == 5 else 0
    var keep = kcols != 0 and ky != 0
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
    _ = dw_addr  # keep the Python object alive through the native call above
    var pdb = fp(db_addr)
    _ = db_addr  # keep the Python object alive through the native call above
    var pdx = fp(dx_addr) if need_dx else pdb
    _ = db_addr  # keep the Python object alive through the native call above
    _ = dx_addr  # keep the Python object alive through the native call above
    var pk = FP(unsafe_from_address=kcols) if keep else pdb
    _ = db_addr  # keep the Python object alive through the native call above
    var py_ = FP(unsafe_from_address=ky) if keep else pdb
    _ = db_addr  # keep the Python object alive through the native call above
    with GILReleased(Python()):
        conv_block_backward_run(px, pw, pb, pg, pi, pdx, pdw, pdb, cprm, pprm, pool, need_dx, keep, pk, py_, nx, OC, ckk)
        _ = db_addr  # keep the Python object alive through the native call above
        _ = dw_addr  # keep the Python object alive through the native call above
        _ = dx_addr  # keep the Python object alive through the native call above
    _ = noidx^
    return PythonObject(0)


def conv_shape_binding(params: PythonObject) raises -> PythonObject:
    var prm = conv_params(_ints(params))
    return Python.tuple(Int(prm[CP_OH]), Int(prm[CP_OW]))


def _pool_prm(params: PythonObject) raises -> List[Int32]:
    return pool_params(_ints(params))


def pool_shape_binding(params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    return Python.tuple(Int(prm[PP_OH]), Int(prm[PP_OW]))

def _count(params: PythonObject) raises -> Int:
    var n = Int(py=params[0])
    if n <= 0:
        raise Error("x_cnn: a positive element count is required")
    return n

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
    check_labels_below(py_, n, k)
    var pg = fp(grad_addr)
    var pp = fp(proba_addr)
    var loss = Float32(0)
    with GILReleased(Python()):
        loss = softmax_xent_run(pl, py_, pg, pp, n, k)
    return PythonObject(Float64(loss))


def sgd_binding(w_addr: PythonObject, g_addr: PythonObject, v_addr: PythonObject, params: PythonObject, hyper: PythonObject) raises -> PythonObject:
    """In place: w and the momentum buffer v. hyper = [lr, momentum,
    weight_decay, dampening, nesterov (0/1), first step (0/1)]; the last
    three default to 0."""
    var h = List[Float32]()
    var nh = Int(py=len(hyper))
    if nh < 3 or nh > 6:
        raise Error("x_cnn sgd: hyper is [lr, momentum, weight_decay(, dampening, nesterov, first)]")
    for k in range(6):  # small-loop(the six sgd hyper words)
        h.append(Float32(Float64(py=hyper[k])) if k < nh else Float32(0))
    if _is_list(w_addr):
        # lane/cnn-apple2: the list form (one handle per parameter), the
        # per-parameter entry in the caller's order
        var cnt = _list_counts(w_addr, g_addr, v_addr, params)
        # p3-cnn-sequence-tsa: the handle lists are read once, then one
        # GIL release covers every parameter (it was one per parameter)
        var qws = List[FP]()
        var qgs = List[FP]()
        var qvs = List[FP]()
        for j in range(len(cnt)):  # driver-loop: over the model's parameters
            qws.append(fp(w_addr[j]))
            qgs.append(fp(g_addr[j]))
            qvs.append(fp(v_addr[j]))
        with GILReleased(Python()):
            sgd_list_run(qws, qgs, qvs, h, cnt)
        return PythonObject(len(cnt))
    var n = _count(params)
    var pw = fp(w_addr)
    var pg = fp(g_addr)
    var pv = fp(v_addr)
    with GILReleased(Python()):
        sgd_run(pw, pg, pv, h, n)
    return PythonObject(n)

def adam_binding(w_addr: PythonObject, g_addr: PythonObject, mv_addr: PythonObject, params: PythonObject, hyper: PythonObject) raises -> PythonObject:
    """In place: w and mv = [m (n) | v (n)]. hyper = [step_size, 1 - beta1,
    beta2, 1 - beta2, eps, sqrt(bias_correction2), weight_decay, adamw (0/1),
    1 - lr * weight_decay] (x_cnn/ops.mojo adam_at)."""
    if Int(py=len(hyper)) != 9:
        raise Error("x_cnn adam: hyper has 9 entries")
    var h = List[Float32]()
    for k in range(9):  # small-loop(the nine adam hyper words)
        h.append(Float32(Float64(py=hyper[k])))
    if _is_list(w_addr):
        # lane/cnn-apple2: the list form (one handle per parameter)
        var cnt = _list_counts(w_addr, g_addr, mv_addr, params)
        # p3-cnn-sequence-tsa: handle lists read once, one GIL release
        var qws = List[FP]()
        var qgs = List[FP]()
        var qms = List[FP]()
        for j in range(len(cnt)):  # driver-loop: over the model's parameters
            qws.append(fp(w_addr[j]))
            qgs.append(fp(g_addr[j]))
            qms.append(fp(mv_addr[j]))
        with GILReleased(Python()):
            adam_list_run(qws, qgs, qms, h, cnt)
        return PythonObject(len(cnt))
    var n = _count(params)
    var pw = fp(w_addr)
    var pg = fp(g_addr)
    var pm = fp(mv_addr)
    with GILReleased(Python()):
        adam_run(pw, pg, pm, h, n)
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

# TOMBSTONE: x_cnn_res_argmax (CNNClassifier.predict's device argmax; the classifier predicts on the x_sequence program executor since lane x8) deleted 2026-11-09 by lane/x8b-cnn-resident-predict; code recoverable at 7c9d9d5575.
# Restore: git apply experiments/removed/X8B_XCNN_RES_ARGMAX.patch

def conv_block_forward_r_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, idx_addr: PythonObject,
    conv_prm: PythonObject, pool_prm: PythonObject, saved: PythonObject,
) raises -> PythonObject:
    """The resident block forward; `saved` = [cols, conv output] (host
    allocations here) that the backward reads, or []."""
    var ns = Int(py=len(saved))
    if ns != 0 and ns != 2:
        raise Error("x_cnn conv block: saved is [] or [cols, conv output]")
    var kcols = Int(py=saved[0]) if ns == 2 else 0
    var ky = Int(py=saved[1]) if ns == 2 else 0
    return _block_forward(x_addr, w_addr, b_addr, out_addr, idx_addr, conv_prm, pool_prm, kcols, ky)


def conv_block_backward_r_binding(
    x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, g_addr: PythonObject, idx_addr: PythonObject,
    outs: PythonObject, conv_prm: PythonObject, pool_prm: PythonObject,
) raises -> PythonObject:
    """outs = [dx, dW, db] or [dx, dW, db, cols, conv output] with the forward's saved arrays."""
    return conv_block_backward_binding(x_addr, w_addr, b_addr, g_addr, idx_addr, outs, conv_prm, pool_prm)


def _is_list(o: PythonObject) raises -> Bool:
    """`isinstance(o, list)` as CPython's PyList_Check: the type's
    Py_TPFLAGS_LIST_SUBCLASS bit (p3-cnn-sequence-tsa: it ran a builtins
    module lookup and two Python calls per optimizer step and per gather)."""
    var cls = py_type(o)
    return (cpython_table()[].PyType_GetFlags(cls._obj_ptr) & _TPFLAGS_LIST_SUBCLASS) != 0


def _list_counts(a: PythonObject, b: PythonObject, c: PythonObject, params: PythonObject) raises -> List[Int]:
    var k = Int(py=len(params))
    if Int(py=len(a)) != k or Int(py=len(b)) != k or Int(py=len(c)) != k:
        raise Error("x_cnn optimizer: one handle per parameter in each list and one count each")
    var out = List[Int]()
    for j in range(k):  # small-loop(one count word per parameter)
        var nj = Int(py=params[j])
        if nj <= 0:
            raise Error("x_cnn: a positive element count is required")
        out.append(nj)
    return out^


# TOMBSTONE: x_cnn_res_gather (the CPU entry; unused: no caller since lane x8, only python/mojolearn/tests/test_x_cnn_repeat.py) deleted 2026-11-09 by lane/m1-followups-misc; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/XCNN_RES_GATHER.patch


def idn2_flags_binding() raises -> PythonObject:
    return PythonObject(idn2_flags())


def neural_tape_budget_binding() raises -> PythonObject:
    return PythonObject(neural_tape_budget_bytes())


def neural_numerical_profile_binding() raises -> PythonObject:
    return PythonObject(neural_numerical_profile())


def vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


# ------------------------------------------------ the table operations' Python entries
# lane calltable-xcnn (2026-11-11): UNPACK only. Each entry turns its Python
# arguments into the plain words of `core/xcnn_entry_table.mojo` and runs
# `x_cnn_run` (the body moved to x_cnn/host/entry_runs_host.mojo) with the GIL
# released. They stay for the binding's Python surface (python/mojolearn/
# _surface_cnn.py, the tools and tests that call them by name); the CNN layer
# doors call the published native entry instead.


def _w(value: PythonObject) raises -> Int:
    """An address or handle word; None is 0 (a run refuses a null it reads)."""
    if value is None:
        return 0
    return Int(py=value)


def _ws(values: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for k in range(Int(py=len(values))):  # small-loop(k: arrays of one entry): reads address list, not data
        out.append(_w(values[k]))
    return out^


def _py_run(op: Int, a: List[Int], p: List[Int]) raises -> PythonObject:
    var f = List[Float64]()
    var r = 0
    with GILReleased(Python()):
        r = x_cnn_host_run(op, a, p, f)
    return PythonObject(r)


def _py_run_f(op: Int, a: List[Int], p: List[Int], drop_p: PythonObject) raises -> PythonObject:
    var f: List[Float64] = [Float64(py=drop_p)]
    var r = 0
    with GILReleased(Python()):
        r = x_cnn_host_run(op, a, p, f)
    return PythonObject(r)


def _m_words(dev: PythonObject, params: PythonObject) raises -> List[Int]:
    """A `_m` entry's integers: the residency word, then the parameters."""
    var p: List[Int] = [Int(py=dev)]
    for k in range(Int(py=len(params))):  # small-loop(the caller's int parameter words)
        p.append(Int(py=params[k]))
    return p^


def gemm_binding(a_addr: PythonObject, b_addr: PythonObject, c_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_GEMM, [_w(a_addr), _w(b_addr), _w(c_addr)], _ints(params))


def conv2d_forward_binding(x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject,
                           params: PythonObject) raises -> PythonObject:
    return _py_run(XC_CONV2D_FORWARD, [_w(x_addr), _w(w_addr), _w(b_addr), _w(out_addr)], _ints(params))


def conv2d_backward_binding(x_addr: PythonObject, w_addr: PythonObject, dout_addr: PythonObject, dx_addr: PythonObject,
                            dw_addr: PythonObject, db_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_CONV2D_BACKWARD, [_w(x_addr), _w(w_addr), _w(dout_addr), _w(dx_addr), _w(dw_addr), _w(db_addr)],
                   _ints(params))


def maxpool2d_forward_binding(x_addr: PythonObject, out_addr: PythonObject, idx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_MAXPOOL2D_FORWARD, [_w(x_addr), _w(out_addr), _w(idx_addr)], _ints(params))


def maxpool2d_backward_binding(dout_addr: PythonObject, idx_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_MAXPOOL2D_BACKWARD, [_w(dout_addr), _w(idx_addr), _w(dx_addr)], _ints(params))


def avgpool2d_forward_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_AVGPOOL2D_FORWARD, [_w(x_addr), _w(out_addr)], _ints(params))


def avgpool2d_backward_binding(dout_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_AVGPOOL2D_BACKWARD, [_w(dout_addr), _w(dx_addr)], _ints(params))


def relu_forward_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_RELU_FORWARD, [_w(x_addr), _w(out_addr)], _ints(params))


def relu_backward_binding(x_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_RELU_BACKWARD, [_w(x_addr), _w(g_addr), _w(dx_addr)], _ints(params))


def add_binding(a_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_ADD, [_w(a_addr), _w(b_addr), _w(out_addr)], _ints(params))


def mul_binding(a_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_MUL, [_w(a_addr), _w(b_addr), _w(out_addr)], _ints(params))


def linear_forward_binding(x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject,
                      params: PythonObject) raises -> PythonObject:
    return _py_run(XC_LINEAR_FORWARD, [_w(x_addr), _w(w_addr), _w(b_addr), _w(out_addr)], _ints(params))


def linear_backward_binding(x_addr: PythonObject, w_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject,
                       dw_addr: PythonObject, db_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_LINEAR_BACKWARD, [_w(x_addr), _w(w_addr), _w(g_addr), _w(dx_addr), _w(dw_addr), _w(db_addr)],
                   _ints(params))


def batchnorm_forward_binding(x_addr: PythonObject, y_addr: PythonObject, running_addr: PythonObject,
                              aux_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_BATCHNORM_FORWARD, [_w(x_addr), _w(y_addr), _w(running_addr), _w(aux_addr)], _ints(params))


def batchnorm_backward_binding(x_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject, aux_addr: PythonObject,
                               params: PythonObject) raises -> PythonObject:
    return _py_run(XC_BATCHNORM_BACKWARD, [_w(x_addr), _w(g_addr), _w(dx_addr), _w(aux_addr)], _ints(params))


def dropout2d_binding(x_addr: PythonObject, y_addr: PythonObject, mask_addr: PythonObject, params: PythonObject,
                      drop_p: PythonObject) raises -> PythonObject:
    return _py_run_f(XC_DROPOUT2D, [_w(x_addr), _w(y_addr), _w(mask_addr)], _ints(params), drop_p)


def spmm_binding(vals_addr: PythonObject, h_addr: PythonObject, out_addr: PythonObject, csr_addr: PythonObject,
                 params: PythonObject) raises -> PythonObject:
    return _py_run(XC_SPMM, [_w(vals_addr), _w(h_addr), _w(out_addr), _w(csr_addr)], _ints(params))


def gcn_norm_binding(w_addr: PythonObject, vals_addr: PythonObject, csr_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_GCN_NORM, [_w(w_addr), _w(vals_addr), _w(csr_addr)], _ints(params))


def pad2d_forward_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_PAD2D_FORWARD, [_w(x_addr), _w(out_addr)], _ints(params))


def pad2d_backward_binding(g_addr: PythonObject, dx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_PAD2D_BACKWARD, [_w(g_addr), _w(dx_addr)], _ints(params))


def adaptive_pool_binding(in_addr: PythonObject, out_addr: PythonObject, idx_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_ADAPTIVE_POOL, [_w(in_addr), _w(out_addr), _w(idx_addr)], _ints(params))


def graph_op_binding(a_addr: PythonObject, b_addr: PythonObject, aux_addr: PythonObject, out_addr: PythonObject,
                     csr_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_GRAPH_OP, [_w(a_addr), _w(b_addr), _w(aux_addr), _w(out_addr), _w(csr_addr)], _ints(params))


def csr_build_host_binding(rows_addr: PythonObject, cols_addr: PythonObject, csr_addr: PythonObject, order_addr: PythonObject,
                      params: PythonObject) raises -> PythonObject:
    return _py_run(XC_CSR_BUILD, [_w(rows_addr), _w(cols_addr), _w(csr_addr), _w(order_addr)], _ints(params))


def perm_gather_host_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_PERM_GATHER, _ws(addrs), _ints(params))


def vals_t_host_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_VALS_T, _ws(addrs), _ints(params))


def gcn_loops_host_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_GCN_LOOPS, _ws(addrs), _ints(params))


def gcn_backward_t_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_GCN_BACKWARD_T, _ws(addrs), _ints(params))


def sage_backward_t_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_SAGE_BACKWARD_T, _ws(addrs), _ints(params))


def res_alloc_binding(n: PythonObject) raises -> PythonObject:
    return _py_run(XC_RES_ALLOC, List[Int](), [Int(py=n)])


def res_free_binding(h: PythonObject) raises -> PythonObject:
    return _py_run(XC_RES_FREE, [Int(py=h)], List[Int]())


def res_upload_binding(h: PythonObject, src_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    return _py_run(XC_RES_UPLOAD, [Int(py=h), _w(src_addr)], [Int(py=n)])


def res_download_binding(h: PythonObject, dst_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    return _py_run(XC_RES_DOWNLOAD, [Int(py=h), _w(dst_addr)], [Int(py=n)])


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_cnn_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_cnn_host() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    try:
        # the native entry the CNN layer doors call (core/xcnn_entry_table.mojo)
        publish_x_cnn_host_entries()
        var m = PythonModuleBuilder("_mojolearn_x_cnn_host")
        m.def_function[x_cnn_host_numeric_mode_binding]("x_cnn_host_numeric_mode")
        m.def_function[x_cnn_host_vendor_binding]("x_cnn_host_vendor")
        m.def_function[x_cnn_host_column_binding]("x_cnn_host_column")
        m.def_function[csr_build_host_binding]("x_cnn_csr_build")
        m.def_function[gcn_loops_host_binding]("x_cnn_gcn_loops")
        m.def_function[perm_gather_host_binding]("x_cnn_perm_gather")
        m.def_function[vals_t_host_binding]("x_cnn_vals_t")
        m.def_function[x_cnn_host_sabotage_binding]("x_cnn_host_sabotage")
        m.def_function[gemm_binding]("x_cnn_gemm")
        m.def_function[conv2d_forward_binding]("x_cnn_conv2d_forward")
        m.def_function[conv2d_backward_binding]("x_cnn_conv2d_backward")
        m.def_function[conv_shape_binding]("x_cnn_conv_shape")
        # TOMBSTONE: x_cnn_conv_block_forward / _backward (non-resident) and the host x_cnn_numeric_mode alias (dead native entries: every caller uses the _r entries / x_cnn_host_numeric_mode) deleted 2026-10-26 by 352103272; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_cnn.patch
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
        m.def_function[gcn_backward_t_binding]("x_cnn_gcn_backward_t")
        m.def_function[sage_backward_t_binding]("x_cnn_sage_backward_t")
        # TOMBSTONE: x_cnn_conv_block_forward / _backward (non-resident) and the host x_cnn_numeric_mode alias (dead native entries: every caller uses the _r entries / x_cnn_host_numeric_mode) deleted 2026-10-26 by 352103272; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_cnn.patch
        m.def_function[vendor_binding]("x_cnn_vendor")
        m.def_function[res_alloc_binding]("x_cnn_res_alloc")
        m.def_function[res_free_binding]("x_cnn_res_free")
        m.def_function[res_upload_binding]("x_cnn_res_upload")
        m.def_function[res_download_binding]("x_cnn_res_download")
        # TOMBSTONE: x_cnn_res_argmax deleted 2026-11-09 by lane/x8b-cnn-resident-predict; code recoverable at 7c9d9d5575.
        m.def_function[conv_block_forward_r_binding]("x_cnn_conv_block_forward_r")
        m.def_function[conv_block_backward_r_binding]("x_cnn_conv_block_backward_r")
        m.def_function[linear_forward_binding]("x_cnn_linear_forward_r")
        m.def_function[linear_backward_binding]("x_cnn_linear_backward_r")
        m.def_function[softmax_xent_binding]("x_cnn_softmax_xent_r")
        m.def_function[sgd_binding]("x_cnn_sgd_r")
        m.def_function[adam_binding]("x_cnn_adam_r")
        m.def_function[idn2_flags_binding]("x_cnn_idn2_flags")
        m.def_function[neural_tape_budget_binding]("x_cnn_neural_tape_budget_bytes")
        m.def_function[neural_numerical_profile_binding]("x_cnn_numerical_profile")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cnn_host: ", e))


def nn14_bounded_im2col_binding() -> PythonObject:
    """Source-arm query for saved-buffer allocation, never data-dependent."""
    return PythonObject(1 if NN14_BOUNDED_IM2COL else 0)
