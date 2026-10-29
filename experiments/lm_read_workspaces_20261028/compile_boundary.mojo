# Compile-only real public forward/read/generate entry points; never execute.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.causal_lm_public import causal_public_c


@export
def PyInit__lm_read_workspaces() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_lm_read_workspaces")
        module.def_py_c_function(causal_public_c[3], "forward")
        module.def_py_c_function(causal_public_c[5], "read")
        module.def_py_c_function(causal_public_c[7], "generate")
        return module.finalize()
    except error:
        abort(String(error))
