# Compile only: all four public numeric-profile operations and native error boundary.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.numeric_profile_native import numeric_profile_c

@export
def PyInit__mojolearn_numeric_profile_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_numeric_profile_compile")
        module.def_py_c_function(numeric_profile_c, "numeric_profile_native_call")
        return module.finalize()
    except error:
        abort(String(error))
