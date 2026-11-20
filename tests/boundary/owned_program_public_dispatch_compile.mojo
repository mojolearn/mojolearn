# Compile the actual owned-program C entry, selected-image factory and all
# HostExec distributed/partitioned operation owners. This artifact is not run.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings._mojolearn_x_sequence_host import (
    session_public_api_binding, ops_session_public_connect_py,
    ops_session_call_c_binding, ops_session_initialize_c_binding,
    ops_program_public_call_c_binding, owned_program_public_api_binding,
    ops_owned_program_connect_py, ops_owned_program_call_c_binding,
)

@export
def PyInit__mojolearn_owned_program_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_owned_program_public_compile")
        m.def_function[session_public_api_binding]("session_public_api_v1")
        m.def_function[ops_session_public_connect_py]("ops_session_public_connect")
        m.def_py_c_function(ops_session_call_c_binding, "ops_session_call")
        m.def_py_c_function(ops_session_initialize_c_binding, "ops_session_initialize")
        m.def_py_c_function(ops_program_public_call_c_binding, "ops_program_public_call")
        m.def_function[owned_program_public_api_binding]("owned_program_public_api_v1")
        m.def_function[ops_owned_program_connect_py]("ops_owned_program_connect")
        m.def_py_c_function(ops_owned_program_call_c_binding, "ops_owned_program_call")
        return m.finalize()
    except error:
        abort(String(error))
