# SPDX-License-Identifier: Apache-2.0
# Compile-only owner: instantiate the true save boundary and writer on HostExec.
# This root is never imported/executed by the provider migration compile check.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from sequence.exec import HostExec
from autograd.ops_api import OPS_HOST
from providers.python.autograd_program_save_pyapi import ops_program_save_native_py


def save_binding(program: PythonObject, path: PythonObject) raises -> PythonObject:
    return ops_program_save_native_py[HostExec, OPS_HOST](program, path)


@export
def PyInit_program_save_boundary_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("program_save_boundary_compile")
        module.def_function[save_binding]("save")
        return module.finalize()
    except error:
        abort(String(error))
