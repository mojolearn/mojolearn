# Compile the actual Session public admission, descriptor, and HostExec owner.
# This fixture is compiled only; it is not loaded or executed.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings._mojolearn_x_sequence_host import (
    session_public_api_binding, ops_session_public_connect_py,
    ops_session_call_c_binding, ops_session_initialize_c_binding,
)

@export
def PyInit__mojolearn_session_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_session_public_compile")
        m.def_function[session_public_api_binding]("session_public_api_v1")
        m.def_function[ops_session_public_connect_py]("ops_session_public_connect")
        m.def_py_c_function(ops_session_call_c_binding, "ops_session_call")
        m.def_py_c_function(ops_session_initialize_c_binding, "ops_session_initialize")
        return m.finalize()
    except error:
        abort(String(error))
