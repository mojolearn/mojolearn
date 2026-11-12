# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE'S GPU BINDING (algorithm expansion lane 8).
Host addresses in, host addresses out; the work is x_cnn/device.mojo. The CPU
twin is bindings/_mojolearn_x_cnn_host.mojo, same names, same contract."""
from core.experiment_guards_neural import NEURAL_GUARDS_OK
from core.hostptr import f32_ptr, i32_ptr
from std.os import abort
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type
from bindings.py_cpython_table import cpython_table

comptime _TPFLAGS_LIST_SUBCLASS = UInt64(1) << 25  # Py_TPFLAGS_LIST_SUBCLASS
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cnn.ops import NN14_BOUNDED_IM2COL

from x_cnn.ops import CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, conv_params
from x_cnn.ops import PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW, pool_params
from x_cnn.ops import FP, IP
from x_cnn.device import (
    linear_forward_into,
    softmax_xent_into,
    sgd_into,
    adam_into,
    conv_block_forward_into,
    opt_many_resident,
)
from autograd.kernels.cnn import (
    linear_backward_into,
    conv_block_backward_into,
    linear_backward_m,
)
# lane calltable-xcnn (2026-11-11): the operations the CNN layer doors call
# are typed runs in x_cnn/entry_runs.mojo, published at module init as this
# image's native entry (core/xcnn_entry_table.mojo); their Python entries
# below only unpack into plain words and call `x_cnn_run`.
from x_cnn.entry_runs import x_cnn_run, publish_x_cnn_entries
from core.xcnn_entry_table import (
    XC_RES_ALLOC, XC_RES_ALLOC_NOFILL, XC_RES_FREE, XC_RES_UPLOAD, XC_RES_DOWNLOAD, XC_GEMM, XC_CONV2D_FORWARD,
    XC_CONV2D_BACKWARD, XC_MAXPOOL2D_FORWARD, XC_MAXPOOL2D_BACKWARD, XC_AVGPOOL2D_FORWARD, XC_AVGPOOL2D_BACKWARD,
    XC_RELU_FORWARD, XC_RELU_BACKWARD, XC_ADD, XC_MUL, XC_LINEAR_FORWARD, XC_LINEAR_BACKWARD, XC_BATCHNORM_FORWARD,
    XC_BATCHNORM_BACKWARD, XC_DROPOUT2D, XC_SPMM, XC_GCN_NORM, XC_PAD2D_FORWARD, XC_PAD2D_BACKWARD,
    XC_ADAPTIVE_POOL, XC_GRAPH_OP, XC_CSR_BUILD, XC_CSR_UPLOAD, XC_PERM_GATHER, XC_VALS_T, XC_GCN_LOOPS,
    XC_GCN_BACKWARD_T, XC_SAGE_BACKWARD_T, XC_GEMM_M, XC_CONV2D_FORWARD_M, XC_CONV2D_BACKWARD_M,
    XC_MAXPOOL2D_FORWARD_M, XC_MAXPOOL2D_BACKWARD_M, XC_AVGPOOL2D_FORWARD_M, XC_AVGPOOL2D_BACKWARD_M, XC_MAP2_M,
    XC_LINEAR_FORWARD_M, XC_BATCHNORM_FORWARD_M, XC_BATCHNORM_BACKWARD_M, XC_DROPOUT2D_M, XC_SPMM_M,
    XC_ADAPTIVE_POOL_M, XC_GRAPH_OP_M, XC_GCN_NORM_M, XC_PAD2D_M, XC_CHAN_COPY_M,
)
from x_cnn.device import idn_flags
from x_cnn.ops import idn2_flags, neural_tape_budget_bytes, neural_numerical_profile
# TOMBSTONE: MOJOLEARN_AUTOGRAD_CNN_OFF (removed: tape is the only backward) deleted 2026-10-19 by ddd4fa509; code recoverable at ddd4fa509^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_CNN_OFF.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: autograd/rules_cnn.mojo and the device epoch imports (lane x8-cnn-on-executor) deleted 2026-11-09; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/X8_CNN_HAND_EPOCH.patch
# Restored 2026-11-09 by lane/r1-restore-measured-winners as ops cnn_conv_block / cnn_linear: rules_cnn.mojo's block and head
# VJPs (the same conv_block_backward_into / linear_backward_into calls) are autograd/rules_step_ops.mojo's
# CNN_CONV_BLOCK_TAG / CNN_LINEAR_TAG (autograd/cnn_block_op.mojo); the device epoch stays deleted.
# The typed bodies (lane thin-trees-metrics, 2026-10-21), moved word for word
# out of this shell.
from x_cnn.device_entries_impl import _device_int_check


# ------------------------------------------------ cpu3-bindings: input checks on the device
# `_device_int_check` and its two kernels are x_cnn/device_entries_impl.mojo's
# (lane thin-trees-metrics, 2026-10-21).


def _fp(addr: PythonObject) raises -> FP:
    """The caller's float32 array, by address (DEVIATION 5716: the device
    entries copy straight to and from it)."""
    return f32_ptr(Int(py=addr)).unsafe_origin_cast[MutAnyOrigin]()


def _ip(addr: PythonObject) raises -> IP:
    return i32_ptr(Int(py=addr)).unsafe_origin_cast[MutAnyOrigin]()


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

def pool_shape_binding(params: PythonObject) raises -> PythonObject:
    var prm = _pool_prm(params)
    return Python.tuple(Int(prm[PP_OH]), Int(prm[PP_OW]))

def _count(params: PythonObject) raises -> Int:
    var n = Int(py=params[0])
    if n <= 0:
        raise Error("x_cnn: a positive element count is required")
    return n

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
    """`isinstance(o, list)` as CPython's PyList_Check: the type's
    Py_TPFLAGS_LIST_SUBCLASS bit (p3-cnn-sequence-tsa: it ran a builtins
    module lookup and two Python calls per optimizer step and per gather)."""
    var cls = py_type(o)
    return (cpython_table()[].PyType_GetFlags(cls._obj_ptr) & _TPFLAGS_LIST_SUBCLASS) != 0


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
    for k in range(6):  # small-loop(the six sgd hyper words)
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

def adam_binding[resident: Bool = False](w_addr: PythonObject, g_addr: PythonObject, mv_addr: PythonObject, params: PythonObject, hyper: PythonObject) raises -> PythonObject:
    """In place: w and mv = [m (n) | v (n)]. hyper = [step_size, 1 - beta1,
    beta2, 1 - beta2, eps, sqrt(bias_correction2), weight_decay, adamw (0/1),
    1 - lr * weight_decay] (x_cnn/ops.mojo adam_at). Resident, w/g/mv may
    be LISTS of handles (one per parameter, params = their counts)."""
    if Int(py=len(hyper)) != 9:
        raise Error("x_cnn adam: hyper has 9 entries")
    var h = List[Float32]()
    for k in range(9):  # small-loop(the nine adam hyper words)
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

# TOMBSTONE: x_cnn_res_gather (the GPU entry; unused: no caller since lane x8, only python/mojolearn/tests/test_x_cnn_repeat.py) deleted 2026-11-09 by lane/m1-followups-misc; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/XCNN_RES_GATHER.patch


# TOMBSTONE: x_cnn_res_argmax (CNNClassifier.predict's device argmax; the classifier predicts on the x_sequence program executor since lane x8) deleted 2026-11-09 by lane/x8b-cnn-resident-predict; code recoverable at 7c9d9d5575.
# Restore: git apply experiments/removed/X8B_XCNN_RES_ARGMAX.patch

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

def linear_backward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [x, w, g, dx, dW, db]."""
    var s = _lin(params)
    var a = _addrs(addrs, 6)
    var d = Int(py=dev)
    with GILReleased(Python()):
        linear_backward_m(a, d, s[0], s[1], s[2])
    return PythonObject(s[0])

# TOMBSTONE: MOJOLEARN_IDN_CNN_BN_FUSE (DROPPED-noise) deleted 2026-10-10 by lane/postmerge-act-6; code recoverable at 9f83ea479.
# Tried: x_cnn_batchnorm_forward_fused_m / x_cnn_batchnorm_backward_relu_m bindings.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_BN_FUSE.patch; record in docs/TOMBSTONES.md.

# ------------------------------------------------- lane fam-neural `_m` forms
# lane fam-neural (2026-10-04): x_cnn/device.mojo "lane fam-neural: `_m`
# forms". New entries only; no existing entry changes. The same parameter
# checks as the host entries they stand beside. `x_cnn_idn_flags` reports
# which of the lane's switches this build has on (the glue uses an entry
# only when its bit is set; the CPU twin has no such entry, so its glue
# keeps the host entries).


def idn_flags_binding() raises -> PythonObject:
    return PythonObject(idn_flags())

# TOMBSTONE: x_cnn_fit_epoch_r / x_cnn_fit_epoch_d / x_cnn_fit_d / x_cnn_predict_r / x_cnn_epoch_rows / x_cnn_adam_hyper_d (CNNClassifier's hand epoch and resident predict; the classifier is a captured program on the executor, sequence/cnn_classifier_program.mojo) deleted 2026-11-09 by lane/x8-cnn-on-executor; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/X8_CNN_HAND_EPOCH.patch


def idn2_flags_binding() raises -> PythonObject:
    return PythonObject(idn2_flags())


def neural_tape_budget_binding() raises -> PythonObject:
    return PythonObject(neural_tape_budget_bytes())


def neural_numerical_profile_binding() raises -> PythonObject:
    return PythonObject(neural_numerical_profile())


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


# ------------------------------------------------ the table operations' Python entries
# lane calltable-xcnn (2026-11-11): UNPACK only. Each entry turns its Python
# arguments into the plain words of `core/xcnn_entry_table.mojo` and runs
# `x_cnn_run` (the body moved to x_cnn/entry_runs.mojo) with the GIL
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
        r = x_cnn_run(op, a, p, f)
    return PythonObject(r)


def _py_run_f(op: Int, a: List[Int], p: List[Int], drop_p: PythonObject) raises -> PythonObject:
    var f: List[Float64] = [Float64(py=drop_p)]
    var r = 0
    with GILReleased(Python()):
        r = x_cnn_run(op, a, p, f)
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


def linear_forward_py(x_addr: PythonObject, w_addr: PythonObject, b_addr: PythonObject, out_addr: PythonObject,
                      params: PythonObject) raises -> PythonObject:
    return _py_run(XC_LINEAR_FORWARD, [_w(x_addr), _w(w_addr), _w(b_addr), _w(out_addr)], _ints(params))


def linear_backward_py(x_addr: PythonObject, w_addr: PythonObject, g_addr: PythonObject, dx_addr: PythonObject,
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


def csr_build_binding(rows_addr: PythonObject, cols_addr: PythonObject, csr_addr: PythonObject, order_addr: PythonObject,
                      params: PythonObject) raises -> PythonObject:
    return _py_run(XC_CSR_BUILD, [_w(rows_addr), _w(cols_addr), _w(csr_addr), _w(order_addr)], _ints(params))


def perm_gather_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_PERM_GATHER, _ws(addrs), _ints(params))


def vals_t_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_VALS_T, _ws(addrs), _ints(params))


def gcn_loops_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
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


def res_alloc_nofill_binding(n: PythonObject) raises -> PythonObject:
    return _py_run(XC_RES_ALLOC_NOFILL, List[Int](), [Int(py=n)])


def csr_upload_binding(csr_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_CSR_UPLOAD, [_w(csr_addr)], _ints(params))


def gemm_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_GEMM_M, _ws(addrs), _m_words(dev, params))


def conv2d_forward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_CONV2D_FORWARD_M, _ws(addrs), _m_words(dev, params))


def conv2d_backward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_CONV2D_BACKWARD_M, _ws(addrs), _m_words(dev, params))


def maxpool2d_forward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_MAXPOOL2D_FORWARD_M, _ws(addrs), _m_words(dev, params))


def maxpool2d_backward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_MAXPOOL2D_BACKWARD_M, _ws(addrs), _m_words(dev, params))


def avgpool2d_forward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_AVGPOOL2D_FORWARD_M, _ws(addrs), _m_words(dev, params))


def avgpool2d_backward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_AVGPOOL2D_BACKWARD_M, _ws(addrs), _m_words(dev, params))


def map2_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_MAP2_M, _ws(addrs), _m_words(dev, params))


def linear_forward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_LINEAR_FORWARD_M, _ws(addrs), _m_words(dev, params))


def batchnorm_forward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_BATCHNORM_FORWARD_M, _ws(addrs), _m_words(dev, params))


def batchnorm_backward_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_BATCHNORM_BACKWARD_M, _ws(addrs), _m_words(dev, params))


def dropout2d_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject, drop_p: PythonObject) raises -> PythonObject:
    return _py_run_f(XC_DROPOUT2D_M, _ws(addrs), _m_words(dev, params), drop_p)


def spmm_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_SPMM_M, _ws(addrs), _m_words(dev, params))


def adaptive_pool_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_ADAPTIVE_POOL_M, _ws(addrs), _m_words(dev, params))


def graph_op_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_GRAPH_OP_M, _ws(addrs), _m_words(dev, params))


def gcn_norm_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_GCN_NORM_M, _ws(addrs), _m_words(dev, params))


def pad2d_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_PAD2D_M, _ws(addrs), _m_words(dev, params))


def chan_copy_m_binding(addrs: PythonObject, dev: PythonObject, params: PythonObject) raises -> PythonObject:
    return _py_run(XC_CHAN_COPY_M, _ws(addrs), _m_words(dev, params))


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_cnn.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_cnn() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    try:
        # the native entry the CNN layer doors call (core/xcnn_entry_table.mojo)
        publish_x_cnn_entries()
        var m = PythonModuleBuilder("_mojolearn_x_cnn")
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
        m.def_function[linear_forward_py]("x_cnn_linear_forward")
        m.def_function[linear_backward_py]("x_cnn_linear_backward")
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
        m.def_function[res_alloc_nofill_binding]("x_cnn_res_alloc_nofill")
        m.def_function[res_free_binding]("x_cnn_res_free")
        m.def_function[res_upload_binding]("x_cnn_res_upload")
        m.def_function[res_download_binding]("x_cnn_res_download")
        # TOMBSTONE: x_cnn_res_argmax deleted 2026-11-09 by lane/x8b-cnn-resident-predict; code recoverable at 7c9d9d5575.
        m.def_function[conv_block_forward_r_binding]("x_cnn_conv_block_forward_r")
        m.def_function[conv_block_backward_r_binding]("x_cnn_conv_block_backward_r")
        m.def_function[linear_forward_binding[True]]("x_cnn_linear_forward_r")
        m.def_function[linear_backward_binding[True]]("x_cnn_linear_backward_r")
        m.def_function[softmax_xent_binding[True]]("x_cnn_softmax_xent_r")
        m.def_function[sgd_binding[True]]("x_cnn_sgd_r")
        m.def_function[adam_binding[True]]("x_cnn_adam_r")
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
        # TOMBSTONE: MOJOLEARN_IDN_CNN_BN_FUSE (DROPPED-noise) deleted 2026-10-10 by lane/postmerge-act-6; code recoverable at 9f83ea479.
        # Tried: the two fused BatchNorm module functions.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_BN_FUSE.patch; record in docs/TOMBSTONES.md.
        m.def_function[dropout2d_m_binding]("x_cnn_dropout2d_m")
        m.def_function[csr_upload_binding]("x_cnn_csr_upload")
        m.def_function[csr_build_binding]("x_cnn_csr_build")
        m.def_function[gcn_loops_binding]("x_cnn_gcn_loops")
        m.def_function[perm_gather_binding]("x_cnn_perm_gather")
        m.def_function[vals_t_binding]("x_cnn_vals_t")
        m.def_function[idn_flags_binding]("x_cnn_idn_flags")
        m.def_function[idn2_flags_binding]("x_cnn_idn2_flags")
        m.def_function[neural_tape_budget_binding]("x_cnn_neural_tape_budget_bytes")
        m.def_function[neural_numerical_profile_binding]("x_cnn_numerical_profile")
        m.def_function[adaptive_pool_m_binding]("x_cnn_adaptive_pool_m")
        m.def_function[graph_op_m_binding]("x_cnn_graph_op_m")
        m.def_function[gcn_norm_m_binding]("x_cnn_gcn_norm_m")
        m.def_function[pad2d_m_binding]("x_cnn_pad2d_m")
        m.def_function[chan_copy_m_binding]("x_cnn_chan_copy_m")
        m.def_function[spmm_m_binding]("x_cnn_spmm_m")
        m.def_function[gcn_backward_t_binding]("x_cnn_gcn_backward_t")
        m.def_function[sage_backward_t_binding]("x_cnn_sage_backward_t")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cnn: ", e))


def nn14_bounded_im2col_binding() -> PythonObject:
    """Source-arm query for saved-buffer allocation, never data-dependent."""
    return PythonObject(1 if NN14_BOUNDED_IM2COL else 0)
