# SPDX-License-Identifier: Apache-2.0
# Python doors for model and session checkpoints (model_checkpoint.mojo):
# argument unpacking and Python results around the typed checkpoint entries.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from sequence.exec_trait import Exec
from providers.python.sequence_pyapi_unpack import py_f64s, py_ints
from providers.python.autograd_pyapi_door import program_info
from autograd.ops_api import ops_session
from autograd.model_checkpoint import (
    SEQ_MODEL_INTS, SEQ_OPT_INTS, SEQ_SAVE_INTS, model_checkpoint_info, model_load_checkpoint,
    model_save_checkpoint, ops_load_program, ops_load_state, ops_save_state,
)


def ops_load_program_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, path: PythonObject,
) raises -> PythonObject:
    """Read a program checkpoint into session `sid`: its parameters become
    new session tensors and the program runs over them. Returns
    program_info (providers/python/autograd_pyapi.mojo)."""
    var file = String(py=path)
    var pid = ops_load_program[E, KIND](Int(py=sid), file)
    var p = ops_session[E, KIND](Int(py=sid))
    return program_info(pid, p[].program(pid))
