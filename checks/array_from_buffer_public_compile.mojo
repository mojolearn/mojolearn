# Compile-only actual Array public dispatcher, including from_buffer routing.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.array_native import array_native_c

@export
def PyInit__mojolearn_array_from_buffer_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_array_from_buffer_compile")
        module.def_py_c_function(array_native_c, "array_native_call")
        return module.finalize()
    except error:
        abort(String(error))
