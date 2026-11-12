# SPDX-License-Identifier: Apache-2.0
# The owning Session image completes the whole public Program operation.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.native_json import native_path
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session
from autograd.control_checkpoint import control_save
from providers.python.autograd_program_owner_pyapi_door import program_owner_fields, program_native_sequence, program_release_fields
from providers.python.autograd_program_arrays_pyapi_door import program_numeric_scalar, program_admit_value, uint64_input
from providers.python.autograd_program_pyapi import ops_program_execute_py, ops_program_inspect_py, ops_program_fit_native_py, fit_stop_options, _integer
from providers.python.autograd_program_control_pyapi_pack import ops_program_control_py, ops_program_decode_py
from providers.python.autograd_program_control_grad_pyapi_pack import ops_program_control_vjp_py
from providers.python.autograd_program_save_pyapi_door import ops_program_save_fields_py
from providers.python.autograd_owned_program_public_pyapi import ops_owned_program_create_request_py


def _option(value: PythonObject) raises:
    if not program_numeric_scalar(value):
        raise Error("TypeError: Program flags require native boolean or numeric scalars")


def _arguments(args: PythonObject, count: Int) raises:
    if py_type(args) is not py_type(Python.tuple()) or len(args) != count:
        raise Error("TypeError: invalid native Program operation arguments")
