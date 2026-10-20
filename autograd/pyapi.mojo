# SPDX-License-Identifier: Apache-2.0
"""The address contract of the model-description entries, written once over
`Exec`: `bindings/_mojolearn_x_sequence.mojo` calls these with a
`DeviceExec`, `bindings/_mojolearn_x_sequence_host.mojo` with a `HostExec`.
Python passes the description and the options as integer and float lists,
the user's arrays as they are (an input descriptor, `cast_in_of`: address,
NumPy dtype number, item size, shape, byte strides; Mojo casts them on the
executor) and caller-owned float32 result buffers by address; Mojo validates
them and runs the whole fit or prediction. Nothing is retained after a call.

Description (both entries): prog = [1, input features, layer count, then per
layer (code, arg0, arg1, 0)], one float per layer in ptab
(autograd/model.mojo LAYER_*; docs/plans/AUTOGRAD_API.md). Version 2
(docs/plans/AUTOGRAD_RANKN.md section 5): prog = [2, ndim, d0 .. d7, layer
count, then per layer (code, arg0 .. arg5, 0)], the per-sample input shape
of any rank; X arrives as its (N, product of the sample dims) row-major
matrix view."""
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from sequence.pyapi import fptr, fval, ival
from sequence.recurrent import OptConfig
from autograd.model import FitConfig, ModelSpec, model_fit, model_predict, parse_model
from autograd.ops_api import ops_session, ops_session_close, ops_session_open
from autograd.rankn_api import rn_dims, rn_shape
from autograd.profile import ag_profile_report, ag_profile_reset
from autograd.profile import ag_profile_on, ag_profile_report, ag_profile_reset
from core.cast_in import CastIn, cast_in_kind, label_kind_of
from autograd.cast_ops import gather_labels

# Training options ahead of the per-layer float table in model_fit's fp.
comptime FIT_INTS = 14
comptime FIT_FLOATS = 9


def _ints(p: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(p)):  # driver-loop: over description words
        out.append(ival(p, i))
    return out^


def _floats(p: PythonObject, start: Int) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(start, len(p)):  # driver-loop: over layers' float arguments
        out.append(fval(p, i))
    return out^


def cast_in_of(desc: PythonObject, what: String) raises -> CastIn:
    """The input descriptor [address, NumPy dtype number, item size, rows,
    cols, row stride, column stride] (strides in bytes; a 1-D array of n is
    (n, 1), a scalar (1, 1)) as a checked CastIn."""
    if len(desc) != 7:
        raise Error(what + ": the input descriptor takes 7 integers")
    var c = CastIn(
        ival(desc, 0), cast_in_kind(ival(desc, 1), ival(desc, 2), what), ival(desc, 2),
        ival(desc, 3), ival(desc, 4), ival(desc, 5), ival(desc, 6),
    )
    c.check(what)
    return c


def _seed(ip: PythonObject, at: Int) raises -> UInt64:
    var lo = ival(ip, at)
    var hi = ival(ip, at + 1)
    if lo < 0 or hi < 0 or lo > 0xFFFFFFFF or hi > 0xFFFFFFFF:
        raise Error("model: the seed words must be 32-bit non-negative values")
    return (UInt64(hi) << 32) | UInt64(lo)


def model_fit_py[E: Exec](
    mut ex: E, addrs: PythonObject, prog: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    """addrs = [X (N, D) descriptor, Y descriptor ((N, out) any accepted
    dtype, or N C-contiguous int32 codes), params address (n_params,
    in/out), loss_curve address (epochs, out)];
    ip = [N, loss, target, optimizer kind, optimizer flags, schedule,
    schedule step, batch, epochs, shuffle, seed_lo, seed_hi,
    n_iter_no_change, init];
    fp = [lr, f1 (momentum / beta1), f2 (beta2), eps, weight_decay,
    f7 (dampening), tol, schedule gamma, huber delta] + one float per layer.
    Returns the number of epochs run."""
    if len(addrs) != 4 or len(ip) != FIT_INTS or len(fp) < FIT_FLOATS + 1:
        raise Error("model_fit: requires 4 addresses, 14 integer parameters and 9 floats plus one per layer")
    var spec = parse_model(_ints(prog), _floats(fp, FIT_FLOATS))
    var N = ival(ip, 0)
    var opt = OptConfig(ival(ip, 3), ival(ip, 4), fval(fp, 1), fval(fp, 2), fval(fp, 3), fval(fp, 4), fval(fp, 5))
    var cfg = FitConfig(
        ival(ip, 1), ival(ip, 2), opt, fval(fp, 0), ival(ip, 5), ival(ip, 6), fval(fp, 7),
        ival(ip, 7), ival(ip, 8), ival(ip, 9) != 0, _seed(ip, 10), ival(ip, 12),
        Float64(py=fp[6]), fval(fp, 8), ival(ip, 13) != 0,
    )
    # MOJOLEARN_AG_PROFILE (lane ag-pass2-kernels): this fit's launch
    # profile, counted by the executor, printed after it
    var profile = ag_profile_on()
    if profile:
        ag_profile_reset()
    var n_iter = model_fit(ex, spec, cfg, cast_in_of(addrs[0], "X"), cast_in_of(addrs[1], "y"), N,
                           fptr(addrs[2], "params"), fptr(addrs[3], "loss_curve"))
    if profile:
        print("AG_PROFILE fit=model_fit")
        print(ag_profile_report())
    return PythonObject(n_iter)


def model_predict_py[E: Exec](
    mut ex: E, addrs: PythonObject, prog: PythonObject, ptab: PythonObject, ip: PythonObject,
) raises -> PythonObject:
    """addrs = [X (N, D) descriptor, params address, out address (N, out)]; ptab = one float per layer;
    ip = [N, chunk, transform (0 none, 1 softmax, 2 sigmoid, 3 the (N, 2)
    [1 - p, p] of a one-output sigmoid, 4 the row argmax as N float32
    codes, 5 one-output sigmoid > 0.5 as N float32 codes)]. Returns the
    number of values written."""
    if len(addrs) != 3 or len(ip) != 3:
        raise Error("model_predict: requires 3 addresses and 3 integer parameters")
    var spec = parse_model(_ints(prog), _floats(ptab, 0))
    var N = ival(ip, 0)
    model_predict(ex, spec, cast_in_of(addrs[0], "X"), N, fptr(addrs[1], "params"), fptr(addrs[2], "out"),
                  ival(ip, 1), ival(ip, 2))
    if ival(ip, 2) == 3:
        return PythonObject(N * 2)
    if ival(ip, 2) == 4 or ival(ip, 2) == 5:
        return PythonObject(N)
    return PythonObject(N * spec.out_features())


def model_labels_check(src: PythonObject, classes_addr: PythonObject, codes_addr: PythonObject) raises -> Tuple[CastIn, Int]:
    """`model_labels`' argument checks (both columns): the labels'
    descriptor (one label per row: cols 1) and the label sort's kind."""
    var c = cast_in_of(src, "y")
    if c.cols != 1:
        raise Error("model_labels: the labels must be one per row")
    if Int(py=classes_addr) == 0 or Int(py=codes_addr) == 0:
        raise Error("model_labels: null buffer address")
    return (c, label_kind_of(c.kind, c.itemsize, "y"))


def model_labels_result(k: Int) raises -> PythonObject:
    if k == -2:
        raise Error("mojolearn: y contains a NaN label; NaN is not a class")
    return PythonObject(k)


def model_gather_labels_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """out[i] = classes[codes[i]]. addrs = [codes (n float32 class codes),
    classes (classes_, k items), out (an array of classes_' dtype, n
    items)]; ip = [n, k, item size]. Gathered on the executor
    (autograd/cast_ops.mojo gather_labels). Returns n. Sequential refuses
    non-numeric labels at fit, so classes_ is always a numeric array."""
    if len(addrs) != 3 or len(ip) != 3:
        raise Error("model_gather_labels: requires 3 addresses and [n, k, item size]")
    gather_labels(ex, ival(addrs, 0), ival(ip, 0), ival(addrs, 1), ival(ip, 1), ival(ip, 2), ival(addrs, 2))
    return PythonObject(ival(ip, 0))


def model_n_params_py(prog: PythonObject, ptab: PythonObject) raises -> PythonObject:
    """[parameter count, output width] of a description (validates it)."""
    var spec = parse_model(_ints(prog), _floats(ptab, 0))
    var out = Python.list()
    out.append(PythonObject(spec.n_params))
    out.append(PythonObject(spec.out_features()))
    return out


# ---------------------------------------------------------------- operation sessions
# (autograd/ops_api.mojo). E is the binding's executor, KIND its tag
# (OPS_DEVICE or OPS_HOST): a session opened by one binding is refused by the
# other. Every entry is one call: argument parsing here, all arithmetic in
# the session's executor.


def _handle_list(p: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(p)):  # small-loop(p: an op's input handles, at most three): argument parsing, no data
        out.append(ival(p, i))
    return out^


def _arg_ints(p: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(p)):  # small-loop(p: an op's integer arguments, a shape or axes list at most): argument parsing, no data
        out.append(ival(p, i))
    return out^


def _arg_floats(p: PythonObject) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(p)):  # small-loop(p: an op's float arguments, at most one): argument parsing, no data
        out.append(fval(p, i))
    return out^


def ops_open_py[E: Exec & Movable & Deinitable, KIND: Int](var ex: E, record: PythonObject) raises -> PythonObject:
    """A new session; record = 1 records a graph for backward. Returns its id."""
    return PythonObject(ops_session_open[E, KIND](ex^, Int(py=record) != 0))


def ops_close_py[E: Exec & Movable & Deinitable, KIND: Int](sid: PythonObject) raises -> PythonObject:
    ops_session_close[E, KIND](Int(py=sid))
    return PythonObject(0)


def ops_upload_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, desc: PythonObject, ip: PythonObject,
) raises -> PythonObject:
    """desc: the caller's (rows, cols) array as it is (`cast_in_of`; cast to
    float32 on the executor); ip = [requires_grad], or [requires_grad,
    ndim, d0 .. d(ndim-1)] for a C-order array of another rank whose
    row-major (rows, cols) matrix view `desc` describes (lane
    ag-rankn-python). Returns the new tensor's handle."""
    if len(ip) < 1:
        raise Error("ops_upload: requires [requires_grad] or [requires_grad, ndim, dims...]")
    var src = cast_in_of(desc, "x")
    var p = ops_session[E, KIND](Int(py=sid))
    if len(ip) == 1:
        return PythonObject(p[].upload(src, ival(ip, 0) != 0))
    var nd = ival(ip, 1)
    if nd < 0 or len(ip) != 2 + nd:
        raise Error("ops_upload: the shape takes ndim and ndim dimensions")
    var dims = List[Int](capacity=nd)
    for k in range(nd):  # small-loop(nd: the tensor rank, at most 8): argument parsing, no data
        dims.append(ival(ip, 2 + k))
    return PythonObject(p[].upload_nd(src, rn_shape(dims, "ops: tensor"), ival(ip, 0) != 0))


def ops_download_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, h: PythonObject, addr: PythonObject,
) raises -> PythonObject:
    """The tensor's values into the caller's float32 array of its shape."""
    var p = ops_session[E, KIND](Int(py=sid))
    p[].download(Int(py=h), fptr(addr, "out"))
    return PythonObject(0)


def ops_op_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, code: PythonObject, hs: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    """Operation `code` (autograd/ops_api.mojo OPS_*) on the tensor handles
    `hs` with integer arguments `ip` and float arguments `fp`; returns the
    new tensor's handle."""
    var p = ops_session[E, KIND](Int(py=sid))
    return PythonObject(p[].op(Int(py=code), _handle_list(hs), _arg_ints(ip), _arg_floats(fp)))


def ops_op_packed_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, code: PythonObject, block: PythonObject,
) raises -> PythonObject:
    """`ops_op_py` with the handles and arguments in the session's packed
    int64 block at address `block` (autograd/ops_api.mojo OPS_PACK_*): no
    Python list is walked (lane ag-pass6-host idea G)."""
    var p = ops_session[E, KIND](Int(py=sid))
    return PythonObject(p[].op_packed(Int(py=code), Int(py=block)))


def ops_scalar_py[E: Exec & Movable & Deinitable, KIND: Int](sid: PythonObject, h: PythonObject) raises -> PythonObject:
    """The one value of a (1, 1) tensor."""
    var p = ops_session[E, KIND](Int(py=sid))
    return PythonObject(Float64(p[].scalar(Int(py=h))))


def ops_shape_py[E: Exec & Movable & Deinitable, KIND: Int](sid: PythonObject, h: PythonObject) raises -> PythonObject:
    """The dims of a tensor: [rows, cols] for a matrix, the full shape
    for any other rank (lane ag-rankn-python)."""
    var p = ops_session[E, KIND](Int(py=sid))
    var dims = rn_dims(p[].shape_nd(Int(py=h)))
    var out = Python.list()
    for k in range(len(dims)):  # small-loop(dims: a tensor's rank, at most 8): result packing, no data
        out.append(PythonObject(dims[k]))
    return out


def ops_free_py[E: Exec & Movable & Deinitable, KIND: Int](sid: PythonObject, h: PythonObject) raises -> PythonObject:
    var p = ops_session[E, KIND](Int(py=sid))
    p[].free(Int(py=h))
    return PythonObject(0)


def ops_reset_py[E: Exec & Movable & Deinitable, KIND: Int](sid: PythonObject) raises -> PythonObject:
    """Drop computed tensors and the graph; uploaded tensors stay."""
    var p = ops_session[E, KIND](Int(py=sid))
    p[].reset()
    return PythonObject(0)


def ops_metric_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, kind: PythonObject, hs: PythonObject,
) raises -> PythonObject:
    """Metric `kind` (OPS_METRIC_*) of the handles [y_true, prediction]."""
    var p = ops_session[E, KIND](Int(py=sid))
    return PythonObject(Float64(p[].metric(Int(py=kind), _handle_list(hs))))


def ops_backward_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, h: PythonObject, accumulate: PythonObject,
) raises -> PythonObject:
    """Differentiate the (1, 1) tensor `h` of a recording session."""
    var p = ops_session[E, KIND](Int(py=sid))
    p[].backward(Int(py=h), Int(py=accumulate) != 0)
    return PythonObject(0)


def ops_grad_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, h: PythonObject, addr: PythonObject,
) raises -> PythonObject:
    """The gradient of tensor `h` into the caller's float32 array of its shape."""
    var p = ops_session[E, KIND](Int(py=sid))
    p[].gradient_into(Int(py=h), fptr(addr, "grad"))
    return PythonObject(0)


# Session optimizers (autograd/ops_api.mojo OpsSession.optimizer / step):
# one call each, handles and options only; every number is computed in the
# session.
def _param_handles(p: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(p)):  # small-loop(p: the step's parameter handles): argument parsing, no data
        out.append(ival(p, i))
    return out^


def ops_optimizer_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    """ip = [kind (sequence/ops.mojo OPT_SGD, OPT_ADAM, OPT_ADAMW), nesterov,
    schedule (autograd/model.mojo SCHED_*), step size]; fp = [lr, momentum,
    beta1, beta2, eps, weight_decay, gamma]. Returns the optimizer's id."""
    var p = ops_session[E, KIND](Int(py=sid))
    return PythonObject(p[].optimizer(_arg_ints(ip), _arg_floats(fp)))


def ops_step_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, opt: PythonObject, hs: PythonObject,
) raises -> PythonObject:
    """One in-place update of the parameter handles `hs` by optimizer `opt`."""
    var p = ops_session[E, KIND](Int(py=sid))
    p[].step(Int(py=opt), _param_handles(hs))
    return PythonObject(0)


def ops_scheduler_step_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, opt: PythonObject,
) raises -> PythonObject:
    """Advance optimizer `opt`'s learning-rate schedule by one epoch."""
    var p = ops_session[E, KIND](Int(py=sid))
    p[].scheduler_step(Int(py=opt))
    return PythonObject(0)


def ops_learning_rate_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, opt: PythonObject,
) raises -> PythonObject:
    """Optimizer `opt`'s current learning rate."""
    var p = ops_session[E, KIND](Int(py=sid))
    return PythonObject(Float64(p[].learning_rate(Int(py=opt))))


def ops_zero_grad_py[E: Exec & Movable & Deinitable, KIND: Int](sid: PythonObject) raises -> PythonObject:
    """Forget every trainable tensor's gradient."""
    var p = ops_session[E, KIND](Int(py=sid))
    p[].zero_grad()
    return PythonObject(0)


def ops_optimizer_state_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, opt: PythonObject, h: PythonObject,
) raises -> PythonObject:
    """[step count, beta1^t, beta2^t, mu product, buffer handle 1, 2, 3]
    (0 for a buffer the optimizer does not use)."""
    var p = ops_session[E, KIND](Int(py=sid))
    var st = p[].optimizer_state(Int(py=opt), Int(py=h))
    var out = Python.list()
    out.append(PythonObject(st[0]))
    out.append(PythonObject(Float64(st[1])))
    out.append(PythonObject(Float64(st[2])))
    out.append(PythonObject(Float64(st[3])))
    out.append(PythonObject(st[4]))
    out.append(PythonObject(st[5]))
    out.append(PythonObject(st[6]))
    return out


def ops_load_optimizer_state_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, opt: PythonObject, h: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    """ip = [step count, buffer handle 1, 2, 3]; fp = [beta1^t, beta2^t, mu
    product]: the words ops_optimizer_state returns."""
    var p = ops_session[E, KIND](Int(py=sid))
    p[].load_optimizer_state(Int(py=opt), Int(py=h), _arg_ints(ip), _arg_floats(fp))
    return PythonObject(0)


# The process-wide launch profile (autograd/profile.mojo; MOJOLEARN_AG_PROFILE),
# read by tools/ag_opbench.py per op.
def ops_profile_reset_py() raises -> PythonObject:
    """Clear the launch profile."""
    ag_profile_reset()
    return PythonObject(0)


def ops_profile_report_py() raises -> PythonObject:
    """The launch profile as text (AG_PROFILE lines)."""
    return PythonObject(ag_profile_report())
