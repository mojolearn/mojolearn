# SPDX-License-Identifier: Apache-2.0
# Python door for control-flow checkpoints (control_checkpoint.mojo):
# argument unpacking and the result dict around the typed entries.
from std.python import Python, PythonObject
from bindings.py_balanced import py_setitem
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session
from autograd.program_pyapi import _integer
from autograd.pyapi import program_info
from autograd.control_checkpoint import control_save, control_load
from bindings.native_json import native_path


def control_save_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, body: PythonObject, condition: PythonObject, otherwise: PythonObject,
    maximum: PythonObject, path: PythonObject, optimizer: Int = 0, seed: UInt64 = 0,
) raises -> PythonObject:
    var loop = otherwise is PythonObject(None)
    var limit = 0
    var other_sid = 0
    var other_id = 0
    if loop:
        limit = _integer(maximum, "max_iterations", 0)
    else:
        other_sid = Int(py=otherwise._s._id)
        other_id = Int(py=otherwise._id)
    var normalized = native_path(path)
    var written = control_save[E, KIND](
        Int(py=sid), Int(py=body._s._id), Int(py=body._id), Int(py=condition._s._id), Int(py=condition._id),
        loop, other_sid, other_id, maximum is not PythonObject(None), limit,
        String(py=normalized), optimizer, seed,
    )
    return PythonObject(written)


def control_load_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, path: PythonObject,
) raises -> PythonObject:
    var normalized = native_path(path)
    var r = control_load[E, KIND](Int(py=sid), String(py=normalized))
    var s = ops_session[E, KIND](Int(py=sid))
    var out = Python.import_module("builtins").dict()
    py_setitem(out, "kind", "while" if r.kind == 1 else "if")
    py_setitem(out, "max_iterations", PythonObject(r.limit) if r.kind == 1 else PythonObject(None))
    py_setitem(out, "body", program_info(r.ids[0], s[].program(r.ids[0])))
    py_setitem(out, "condition", program_info(r.ids[1], s[].program(r.ids[1])))
    py_setitem(out, "otherwise", program_info(r.ids[2], s[].program(r.ids[2])) if r.kind == 0 else PythonObject(None))
    py_setitem(out, "optimizer", PythonObject(r.optimizer) if r.training else PythonObject(None))
    py_setitem(out, "seed", PythonObject(Int(r.seed >> 32)).__lshift__(32).__or__(Int(r.seed & 0xffffffff)))
    return out
