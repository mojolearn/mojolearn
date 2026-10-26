# SPDX-License-Identifier: Apache-2.0
# Python doors for model and session checkpoints (model_checkpoint.mojo):
# argument unpacking and Python results around the typed checkpoint entries.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from sequence.exec_trait import Exec
from sequence.pyapi import py_f64s, py_ints
from autograd.pyapi import program_info
from autograd.ops_api import ops_session
from autograd.model_checkpoint import (
    SEQ_MODEL_INTS, SEQ_OPT_INTS, SEQ_SAVE_INTS, model_checkpoint_info, model_load_checkpoint,
    model_save_checkpoint, ops_load_program, ops_load_state, ops_save_state,
)


def model_save_checkpoint_py(
    path: PythonObject, addrs: PythonObject, prog: PythonObject, ip: PythonObject, fp: PythonObject,
    classes: PythonObject,
) raises -> PythonObject:
    """Write a Sequential checkpoint to `path`.

    addrs = [params (n_params float32), m, v (n_params float32 each),
    state (FIT_STATE_WORDS int64), loss curve (epochs run float32)]; with
    ip[0] = 1 (an inference export) only params is read and the others may
    be 0. prog = the description; ip = [export, loss, target, optimizer,
    optimizer flags, schedule, schedule step, n_iter_no_change, batch size,
    shuffle, n_features, n_outputs, y_1d, epochs run, class count]; fp =
    model_fit's fp (9 options, then one float per layer); classes = the
    classes_ input descriptor (autograd/pyapi.mojo cast_in_of), or None
    when the class count is 0. Returns the file's byte count."""
    var ipw = py_ints(ip)
    var cw = List[Int]()
    if len(ipw) == SEQ_SAVE_INTS and ipw[2 + SEQ_OPT_INTS + SEQ_MODEL_INTS] > 0:
        cw = py_ints(classes)
    return PythonObject(model_save_checkpoint(String(py=path), py_ints(prog), py_ints(addrs), ipw, py_f64s(fp), cw))


def model_checkpoint_info_py(path: PythonObject) raises -> PythonObject:
    """The description and options of a Sequential checkpoint, for Python to
    rebuild the model object: [export (0/1), layers, ints, floats], layers =
    one [code, arg0, arg1, float] per layer, ints = [the 9 option words,
    n_features, n_outputs, y_1d, epochs run, class count, class kind, class
    item size, n_params], floats = [lr, f1, f2, eps, weight_decay, f7, tol,
    schedule gamma, huber delta] (model_fit's fp head)."""
    var r = model_checkpoint_info(String(py=path))
    var out = Python.list()
    _ = py_call(out.append, PythonObject(1 if r.is_export else 0))
    var layers = Python.list()
    for i in range(len(r.layer_floats)):  # small-loop(layers: one record per model layer): result packing, no data
        var rec = Python.list()
        _ = py_call(rec.append, PythonObject(r.layer_ints[3 * i]))
        _ = py_call(rec.append, PythonObject(r.layer_ints[3 * i + 1]))
        _ = py_call(rec.append, PythonObject(r.layer_ints[3 * i + 2]))
        _ = py_call(rec.append, PythonObject(r.layer_floats[i]))
        _ = py_call(layers.append, rec)
    _ = py_call(out.append, layers)
    var ints = Python.list()
    for i in range(len(r.ints)):  # small-loop(ints: the record's option words): result packing, no data
        _ = py_call(ints.append, PythonObject(r.ints[i]))
    _ = py_call(out.append, ints)
    var floats = Python.list()
    for i in range(len(r.floats)):  # small-loop(floats: the nine option floats): result packing, no data
        _ = py_call(floats.append, PythonObject(r.floats[i]))
    _ = py_call(out.append, floats)
    return out


def model_load_checkpoint_py(path: PythonObject, addrs: PythonObject) raises -> PythonObject:
    """Fill the caller's buffers from a Sequential checkpoint: addrs =
    [params (n_params float32), m, v (n_params float32 each), state
    (FIT_STATE_WORDS int64), loss curve (epochs run float32), classes
    (class count values of the class kind and size)]; every address but
    params may be 0 and is then skipped. Returns n_params."""
    var a = py_ints(addrs)
    return PythonObject(model_load_checkpoint(String(py=path), a))


def ops_save_state_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, path: PythonObject, params: PythonObject, optimizer: PythonObject,
    counters: PythonObject,
) raises -> PythonObject:
    """Write the session tensors `params` (handles), the state of
    `optimizer` (its handle, or 0 for none) for each of them, read through
    OpsSession.optimizer_state, and counters = [seed low, seed high, epoch,
    step] to `path`. Returns the byte count."""
    var c = py_ints(counters)
    var file = String(py=path)
    var hs = py_ints(params)
    return PythonObject(ops_save_state[E, KIND](Int(py=sid), file, hs, Int(py=optimizer), c))


def ops_load_state_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, path: PythonObject, requires_grad: PythonObject, optimizer: PythonObject,
) raises -> PythonObject:
    """Upload the parameters of a session checkpoint as new tensors and,
    with `optimizer` (a handle, or 0), restore each one's saved optimizer
    state through OpsSession.load_optimizer_state (the buffers are copied
    from temporary tensors, freed after). Returns [handles, has optimizer
    state, seed low, seed high, epoch, step]."""
    var file = String(py=path)
    var opt = Int(py=optimizer)
    var grad = Int(py=requires_grad) != 0
    var r = ops_load_state[E, KIND](Int(py=sid), file, grad, opt)
    var hs = Python.list()
    for i in range(len(r.handles)):  # small-loop(handles: one per restored parameter tensor): result packing, no data
        _ = py_call(hs.append, PythonObject(r.handles[i]))
    var res = Python.list()
    _ = py_call(res.append, hs)
    _ = py_call(res.append, PythonObject(1 if r.has_opt else 0))
    for i in range(len(r.counters)):  # small-loop(4: the saved counters): result packing, no data
        _ = py_call(res.append, PythonObject(r.counters[i]))
    return res


# TOMBSTONE: the ops_* per-op session doors and lamb_step_py (dead native entries: reached only through their deleted exports) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_sequence.patch


def ops_load_program_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, path: PythonObject,
) raises -> PythonObject:
    """Read a program checkpoint into session `sid`: its parameters become
    new session tensors and the program runs over them. Returns
    program_info (autograd/pyapi.mojo)."""
    var file = String(py=path)
    var pid = ops_load_program[E, KIND](Int(py=sid), file)
    var p = ops_session[E, KIND](Int(py=sid))
    return program_info(pid, p[].program(pid))
