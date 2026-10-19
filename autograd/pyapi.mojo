# SPDX-License-Identifier: Apache-2.0
"""The address contract of the model-description entries, written once over
`Exec`: `bindings/_mojolearn_x_sequence.mojo` calls these with a
`DeviceExec`, `bindings/_mojolearn_x_sequence_host.mojo` with a `HostExec`.
Python passes the description and the options as integer and float lists
plus caller-owned C-contiguous float32 buffers by address; Mojo validates
them and runs the whole fit or prediction. Nothing is retained after a call.

Description (both entries): prog = [1, input features, layer count, then per
layer (code, arg0, arg1, 0)], one float per layer in ptab
(autograd/model.mojo LAYER_*; docs/plans/AUTOGRAD_API.md)."""
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from sequence.pyapi import fptr, fval, ival
from sequence.recurrent import OptConfig
from autograd.model import FitConfig, ModelSpec, model_fit, model_predict, parse_model
from autograd.ops_api import ops_session, ops_session_close, ops_session_open
from autograd.profile import ag_profile_report, ag_profile_reset
from autograd.profile import ag_profile_on, ag_profile_report, ag_profile_reset

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


def _seed(ip: PythonObject, at: Int) raises -> UInt64:
    var lo = ival(ip, at)
    var hi = ival(ip, at + 1)
    if lo < 0 or hi < 0 or lo > 0xFFFFFFFF or hi > 0xFFFFFFFF:
        raise Error("model: the seed words must be 32-bit non-negative values")
    return (UInt64(hi) << 32) | UInt64(lo)


def model_fit_py[E: Exec](
    mut ex: E, addrs: PythonObject, prog: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    """addrs = [X (N, D), Y ((N, out) float32, or N int32 codes), params
    (n_params, in/out), loss_curve (epochs, out)];
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
    var n_iter = model_fit(ex, spec, cfg, fptr(addrs[0], "X"), fptr(addrs[1], "Y"), N,
                           fptr(addrs[2], "params"), fptr(addrs[3], "loss_curve"))
    if profile:
        print("AG_PROFILE fit=model_fit")
        print(ag_profile_report())
    return PythonObject(n_iter)


def model_predict_py[E: Exec](
    mut ex: E, addrs: PythonObject, prog: PythonObject, ptab: PythonObject, ip: PythonObject,
) raises -> PythonObject:
    """addrs = [X (N, D), params, out (N, out)]; ptab = one float per layer;
    ip = [N, chunk, transform (0 none, 1 softmax, 2 sigmoid, 3 the (N, 2)
    [1 - p, p] of a one-output sigmoid, 4 the row argmax as N float32
    codes, 5 one-output sigmoid > 0.5 as N float32 codes)]. Returns the
    number of values written."""
    if len(addrs) != 3 or len(ip) != 3:
        raise Error("model_predict: requires 3 addresses and 3 integer parameters")
    var spec = parse_model(_ints(prog), _floats(ptab, 0))
    var N = ival(ip, 0)
    model_predict(ex, spec, fptr(addrs[0], "X"), N, fptr(addrs[1], "params"), fptr(addrs[2], "out"),
                  ival(ip, 1), ival(ip, 2))
    if ival(ip, 2) == 3:
        return PythonObject(N * 2)
    if ival(ip, 2) == 4 or ival(ip, 2) == 5:
        return PythonObject(N)
    return PythonObject(N * spec.out_features())


def model_labels_check(src_addr: PythonObject, n: PythonObject, kind: PythonObject,
                       classes_addr: PythonObject, codes_addr: PythonObject) raises -> Tuple[Int, Int]:
    """`model_labels`' argument checks (both columns): returns (n, kind)."""
    var count = Int(py=n)
    var k_ind = Int(py=kind)
    if count < 1:
        raise Error("model_labels: n must be positive, got " + String(count))
    if k_ind != 0 and k_ind != 1:
        raise Error("model_labels: kind must be 0 (float64) or 1 (int64)")
    if Int(py=src_addr) == 0 or Int(py=classes_addr) == 0 or Int(py=codes_addr) == 0:
        raise Error("model_labels: null buffer address")
    return (count, k_ind)


def model_labels_result(k: Int) raises -> PythonObject:
    if k == -2:
        raise Error("mojolearn: y contains a NaN label; NaN is not a class")
    return PythonObject(k)


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
    for i in range(len(p)):  # small-loop(p: an op's integer arguments, at most four): argument parsing, no data
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
    sid: PythonObject, addr: PythonObject, ip: PythonObject,
) raises -> PythonObject:
    """addr: the caller's C-contiguous float32 (rows, cols); ip = [rows, cols,
    requires_grad]. Returns the new tensor's handle."""
    if len(ip) != 3:
        raise Error("ops_upload: requires [rows, cols, requires_grad]")
    var p = ops_session[E, KIND](Int(py=sid))
    return PythonObject(p[].upload(fptr(addr, "data"), ival(ip, 0), ival(ip, 1), ival(ip, 2) != 0))


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


def ops_scalar_py[E: Exec & Movable & Deinitable, KIND: Int](sid: PythonObject, h: PythonObject) raises -> PythonObject:
    """The one value of a (1, 1) tensor."""
    var p = ops_session[E, KIND](Int(py=sid))
    return PythonObject(Float64(p[].scalar(Int(py=h))))


def ops_shape_py[E: Exec & Movable & Deinitable, KIND: Int](sid: PythonObject, h: PythonObject) raises -> PythonObject:
    """[rows, cols] of a tensor."""
    var p = ops_session[E, KIND](Int(py=sid))
    var shape = p[].shape(Int(py=h))
    var out = Python.list()
    out.append(PythonObject(shape[0]))
    out.append(PythonObject(shape[1]))
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


# The process-wide launch profile (autograd/profile.mojo; MOJOLEARN_AG_PROFILE),
# read by tools/ag_opbench.py per op.
def ops_profile_reset_py() raises -> PythonObject:
    """Clear the launch profile."""
    ag_profile_reset()
    return PythonObject(0)


def ops_profile_report_py() raises -> PythonObject:
    """The launch profile as text (AG_PROFILE lines)."""
    return PythonObject(ag_profile_report())
