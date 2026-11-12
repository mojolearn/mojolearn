# Compile-only: no runtime or numerical identity qualification.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.tsa_public_native import tsa_c

@export
def PyInit__peer6_autoarima_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_peer6_autoarima_compile")
        module.def_py_c_function(tsa_c, "tsa_native_call")
        return module.finalize()
    except error:
        abort(String(error))
