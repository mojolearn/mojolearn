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
from bindings.py_balanced import py_call
from sequence.exec_trait import Exec
from providers.python.sequence_pyapi import fptr, fval, ival
from sequence.recurrent import OptConfig
from autograd.model import FIT_STATE_WORDS, FS_EPOCH, IP64, FitConfig, FitValidation, ModelSpec, model_fit, model_fit_state, model_predict, parse_model
from autograd.ops_api import ops_session
from autograd.program import OpsProgram
from autograd.rankn import ShapeN
from sequence.ops import FP
from autograd.rankn_api import rn_dims
from autograd.profile import ag_profile_report, ag_profile_reset
from autograd.profile import ag_profile_on, ag_profile_report, ag_profile_reset
from core.cast_in import CastIn, cast_in_kind, label_kind_of
from autograd.cast_ops import gather_labels
from autograd.weight_import import WI_RECORD, model_import_params, session_load_check, wi_source
from autograd.session_weights import session_load_weights
from autograd.model_execution import model_fit_native, model_predict_native, model_counts_native, model_class_count

# Training options ahead of the per-layer float table in model_fit's fp.
comptime FIT_INTS = 14
comptime FIT_FLOATS = 9
# model_fit with the resumable state: four more addresses and the resume flag.
comptime FIT_ADDRS = 4
comptime FIT_ADDRS_STATE = 8
comptime FIT_INTS_STATE = 15


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


def model_float64_words(values: PythonObject, count: Int = -1) raises -> List[Float64]:
    var result = List[Float64]()
    var n = len(values) if count < 0 else count
    for i in range(n):  # driver-loop: admit model scalar metadata once
        result.append(Float64(py=values[i]))
    return result^


def model_admit_addresses(addrs: PythonObject, target: Int, codes_ready: Bool = False,
                          captured_x: Optional[CastIn] = None) raises -> Tuple[List[Int], List[CastIn]]:
    var n = len(addrs)
    var addresses = List[Int](length=n, fill=0)
    var sources = List[CastIn]()
    sources.append(captured_x.value() if captured_x else cast_in_of(addrs[0], "X"))
    sources.append(CastIn.f32_c(0, 1, 1) if codes_ready else cast_in_of(addrs[1], "y"))
    for i in range(2, n):  # driver-loop: admit stable addresses once, descriptors handled below
        if i < 8 or i > 10:
            addresses[i] = Int(py=addrs[i])
    if n == 13:
        sources.append(cast_in_of(addrs[8], "validation X"))
        sources.append(cast_in_of(addrs[9], "validation y"))
        sources.append(cast_in_of(addrs[10], "training classes") if target != 0 else CastIn.f32_c(0, 1, 1))
    return (addresses^, sources^)


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
    Returns the number of epochs run.

    The resumable form (autograd/model.mojo model_fit_state) adds four
    addresses, [m (n_params float32, in/out), v (n_params float32, in/out),
    state (FIT_STATE_WORDS int64, in/out), the earlier loss curve (float32,
    state[FS_EPOCH] values, read when resuming)], and ip[14] = resume (0/1);
    loss_curve then holds state[FS_EPOCH] + epochs values. Returns the epochs
    completed in all (the earlier ones included)."""
    return _model_fit_run(ex, addrs, prog, ip, fp, 0, 0)


def _model_fit_run[E: Exec](
    mut ex: E, addrs: PythonObject, prog: PythonObject, ip: PythonObject, fp: PythonObject,
    codes_dev: Int, n_classes: Int,
) raises -> PythonObject:
    # Packed public compatibility door: admit once, then call the SAME native
    # whole-model owner as the shared direct descriptor provider.
    var validated = len(addrs) == 13 and len(ip) == 16
    var stateful = (len(addrs) == FIT_ADDRS_STATE and len(ip) == FIT_INTS_STATE) or validated
    var plain = len(addrs) == FIT_ADDRS and len(ip) == FIT_INTS
    if not (stateful or plain) or len(fp) < FIT_FLOATS + 1:
        raise Error("model_fit: requires 4 addresses and 14 integer parameters (or 8 and 15 with the "
                    + "training state) and 9 floats plus one per layer")
    var options = _ints(ip)
    var admitted = model_admit_addresses(addrs, options[2], codes_dev != 0)
    return PythonObject(model_fit_native(ex, admitted[0], admitted[1], _ints(prog), options,
                                        model_float64_words(fp), codes_dev, n_classes))


def model_predict_py[E: Exec](
    mut ex: E, addrs: PythonObject, prog: PythonObject, ptab: PythonObject, ip: PythonObject,
) raises -> PythonObject:
    if len(addrs) != 3 or len(ip) != 3:
        raise Error("model_predict: requires 3 addresses and 3 integer parameters")
    var addresses: List[Int] = [0, Int(py=addrs[1]), Int(py=addrs[2])]
    return PythonObject(model_predict_native(ex, addresses, cast_in_of(addrs[0], "X"),
                                            _ints(prog), model_float64_words(ptab), _ints(ip)))


def model_predict_labels_py[E: Exec](
    mut ex: E, addrs: PythonObject, prog: PythonObject, ptab: PythonObject, ip: PythonObject,
) raises -> PythonObject:
    if len(addrs) != 4 or len(ip) != 5:
        raise Error("model_predict_labels: requires 4 addresses and [N, chunk, transform, k, item size]")
    var addresses: List[Int] = [0, Int(py=addrs[1]), Int(py=addrs[2]), Int(py=addrs[3])]
    return PythonObject(model_predict_native(ex, addresses, cast_in_of(addrs[0], "X"),
                                            _ints(prog), model_float64_words(ptab), _ints(ip), True))


def model_fit_labels_check(lab: PythonObject) raises -> Tuple[CastIn, Int, Int, Int, Int]:
    """`model_fit_labels`' label arguments: lab = [y descriptor (one label
    per row), classes address (n slots of y's dtype), model outputs, loss
    target (1 cross_entropy, 2 bce_with_logits)]. Returns (labels, sort
    kind, classes address, outputs, target)."""
    if len(lab) != 4:
        raise Error("model_fit_labels: the labels take [y descriptor, classes address, outputs, target]")
    var c = cast_in_of(lab[0], "y")
    if c.cols != 1:
        raise Error("model_labels: the labels must be one per row")
    var classes_addr = ival(lab, 1)
    if classes_addr == 0:
        raise Error("model_labels: null buffer address")
    var target = ival(lab, 3)
    if target != 1 and target != 2:
        raise Error("model_fit_labels: target must be 1 (cross_entropy) or 2 (bce_with_logits)")
    return (c, label_kind_of(c.kind, c.itemsize, "y"), classes_addr, ival(lab, 2), target)


def model_fit_labels_py[E: Exec](
    mut ex: E, k: Int, codes: FP, n_out: Int, target: Int, addrs: PythonObject, prog: PythonObject,
    ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    """Sequential.fit of a classifier after the label sort (lane
    migrate-neural-2): `k` classes found, their N codes already in the
    executor buffer `codes`; the class-count refusals of the door, then
    `model_fit_py`'s resumable fit reading the codes there (addrs[1], the
    target slot, is not read; a validation classes descriptor in addrs[10]
    is cut to the k classes). Returns [k, epochs run]."""
    var count = model_class_count(k, n_out, target)
    var n_iter = _model_fit_run(ex, addrs, prog, ip, fp, Int(codes), count)
    var out = Python.list()
    _ = py_call(out.append, PythonObject(count))
    _ = py_call(out.append, n_iter)
    return out


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
    var counts = model_counts_native(_ints(prog), model_float64_words(ptab))
    var out = Python.list()
    _ = py_call(out.append, PythonObject(counts[0]))
    _ = py_call(out.append, PythonObject(counts[1]))
    return out


def _names(p: PythonObject) raises -> List[String]:
    var out = List[String]()
    for i in range(len(p)):  # small-loop(p: one name per source weight tensor): argument parsing, no data
        out.append(String(py=p[i]))
    return out^


def _record_ints(p: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(p)):  # small-loop(p: WI_RECORD words per source weight tensor): argument parsing, no data
        out.append(ival(p, i))
    return out^


def model_import_params_py[E: Exec](
    mut ex: E, prog: PythonObject, ptab: PythonObject, recs: PythonObject, names: PythonObject,
    params_addr: PythonObject,
) raises -> PythonObject:
    """Sequential.load_state_dict: the model's parameters from source
    tensors (autograd/weight_import.mojo: WI_RECORD integers per tensor in
    `recs`, its name in `names`) into the caller's float32 params buffer
    (model_n_params' count). Returns [parameter count, output width, input
    values per sample]."""
    var spec = parse_model(_ints(prog), _floats(ptab, 0))
    model_import_params(ex, spec, _record_ints(recs), _names(names), fptr(params_addr, "params"))
    var out = Python.list()
    _ = py_call(out.append, PythonObject(spec.n_params))
    _ = py_call(out.append, PythonObject(spec.out_features()))
    _ = py_call(out.append, PythonObject(spec.n_inputs))
    return out


# ---------------------------------------------------------------- operation sessions
# (autograd/ops_api.mojo). E is the binding's executor, KIND its tag
# (OPS_DEVICE or OPS_HOST): a session opened by one binding is refused by the
# other. Every entry is one call: argument parsing here, all arithmetic in
# the session's executor.


# TOMBSTONE: the ops_* per-op session doors and lamb_step_py (dead native entries: reached only through their deleted exports) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_sequence.patch


def ops_load_weights_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, recs: PythonObject, names: PythonObject,
) raises -> PythonObject:
    """Session.load_weights: source tensors (autograd/weight_import.mojo
    records, the target handle in word 0) into the session's uploaded
    tensors, in place. Every record is checked before any tensor is
    written. Returns the number of tensors written."""
    var r_ = _record_ints(recs)
    var nm = _names(names)
    return PythonObject(session_load_weights[E, KIND](Int(py=sid), r_, nm))


def ops_shape_py[E: Exec & Movable & Deinitable, KIND: Int](sid: PythonObject, h: PythonObject) raises -> PythonObject:
    """The dims of a tensor: [rows, cols] for a matrix, and the full
    shape of a tensor of any other rank."""
    var p = ops_session[E, KIND](Int(py=sid))
    var dims = rn_dims(p[].shape_nd(Int(py=h)))
    var out = Python.list()
    for k in range(len(dims)):  # small-loop(dims: a tensor's rank, at most 8): result packing, no data
        _ = py_call(out.append, PythonObject(dims[k]))
    return out


# The process-wide launch profile (autograd/profile.mojo; MOJOLEARN_AG_PROFILE),
# read by tools/ag_opbench.py per op.
def ops_profile_reset_py() raises -> PythonObject:
    """Clear the launch profile."""
    ag_profile_reset()
    return PythonObject(0)


def ops_profile_report_py() raises -> PythonObject:
    """The launch profile as text (AG_PROFILE lines)."""
    return PythonObject(ag_profile_report())


# ---------------------------------------------------------------- captured programs
# (autograd/program.mojo; OpsSession.capture / run_program). One call each:
# the replay is the session's, on its executor.
def _dims_out(shapes: List[ShapeN]) raises -> PythonObject:
    var out = Python.list()
    for k in range(len(shapes)):  # small-loop(shapes: a program's inputs or outputs): result packing, no data
        var dims = rn_dims(shapes[k])
        var d = Python.list()
        for j in range(len(dims)):  # small-loop(dims: a tensor's rank, at most 8): result packing, no data
            _ = py_call(d.append, PythonObject(dims[j]))
        _ = py_call(out.append, d)
    return out


def program_info(pid: Int, p: OpsProgram) raises -> PythonObject:
    """[program id, input shapes, output shapes, parameter handles, input
    kinds (autograd/program.mojo PROGRAM_IN_*: 0 float32 values, 1 int32
    ids)]."""
    var res = Python.list()
    _ = py_call(res.append, PythonObject(pid))
    _ = py_call(res.append, _dims_out(p.in_shapes))
    _ = py_call(res.append, _dims_out(p.out_shapes))
    var ph = Python.list()
    for j in range(len(p.params)):  # small-loop(params: a program's parameter handles): result packing, no data
        _ = py_call(ph.append, PythonObject(p.params[j]))
    _ = py_call(res.append, ph)
    var kinds = Python.list()
    for k in range(len(p.in_kinds)):  # small-loop(in_kinds: a program's inputs): result packing, no data
        _ = py_call(kinds.append, PythonObject(p.in_kinds[k]))
    _ = py_call(res.append, kinds)
    return res

