# Compile-only public backend owner; never imported or executed.
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from bindings.backend_public_native import backend_public_c

@export
def PyInit__mojolearn_backend_public_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_backend_public_compile")
        module.def_py_c_function(backend_public_c, "backend_public_native_call")
        return module.finalize()
    except error:
        abort(String(error))
