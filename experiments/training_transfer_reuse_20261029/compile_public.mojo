# Compile-only real public imported/captured-program host execution provider.
from std.python import PythonObject
from sequence.exec import HostExec
from autograd.ops_api import OPS_HOST
from providers.python.autograd_program_pyapi import ops_program_execute_py


@export
def compile_program_public(sid: PythonObject, pid: PythonObject, owner_sid: PythonObject,
                           arrays: PythonObject, dynamic: PythonObject,
                           optimize: PythonObject, profile: PythonObject) raises -> PythonObject:
    return ops_program_execute_py[HostExec, OPS_HOST](sid, pid, owner_sid, arrays, dynamic, optimize, profile)
